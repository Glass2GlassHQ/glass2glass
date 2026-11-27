// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::ffi::{c_char, c_int, c_uint, c_ushort, c_void};
use core::future::Future;
use core::pin::Pin;
use core::ptr::NonNull;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName};
use g2g_core::{
    AsyncElement, AudioFormat, BusHandle, BusMessage, Caps, CapsConstraint, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, HardwareError, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::compressed_audio_passthrough::{pad_templates, passthrough_caps, require_format};
use crate::mhas::{MhasAccessUnit, MhasFramer};
use crate::textencoding::{base64_decode, base64_encode};

#[allow(unreachable_pub)]
mod ffi {
    use core::ffi::{c_char, c_int, c_uint, c_ushort, c_void};

    #[repr(C)]
    pub struct MpeghUiManager {
        _opaque: [u8; 0],
    }

    pub const MPEGH_UI_OK: c_int = 0;
    pub const MPEGH_UI_OK_BUT_NO_VALID_DATA: c_int = 7;

    pub const MPEGH_UI_FORCE_RESTART_XML: c_uint = 4;
    pub const MPEGH_UI_NO_CHANGE: c_uint = 1;
    pub const MPEGH_UI_CONTINUES_XML: c_uint = 2;
    pub const MPEGH_UI_INCOMPLETE_XML: c_uint = 4;

    extern "C" {
        pub fn mpegh_UI_Manager_Open() -> *mut MpeghUiManager;
        pub fn mpegh_UI_Manager_Close(manager: *mut MpeghUiManager);
        pub fn mpegh_UI_GetXmlSceneState(
            manager: *mut MpeghUiManager,
            xml_out: *mut c_char,
            xml_out_size: c_uint,
            flags_in: c_uint,
            flags_out: *mut c_uint,
        ) -> c_int;
        pub fn mpegh_UI_ApplyXmlAction(
            manager: *mut MpeghUiManager,
            xml_in: *const c_char,
            xml_in_size: c_uint,
            flags_out: *mut c_uint,
        ) -> c_int;
        pub fn mpegh_UI_FeedMHAS(
            manager: *mut MpeghUiManager,
            mhas_buffer: *mut u8,
            mhas_length: c_uint,
        ) -> c_int;
        pub fn mpegh_UI_UpdateMHAS(
            manager: *mut MpeghUiManager,
            mhas_buffer: *mut u8,
            mhas_buffer_length: c_uint,
            mhas_length: *mut c_uint,
        ) -> c_int;
        pub fn mpegh_UI_SetPersistenceMemory(
            manager: *mut MpeghUiManager,
            persistence_memory_block: *mut c_void,
            persistence_memory_size: c_ushort,
        ) -> c_int;
        pub fn mpegh_UI_GetPersistenceMemory(
            manager: *mut MpeghUiManager,
            persistence_memory_block: *mut *mut c_void,
            persistence_memory_size: *mut c_ushort,
        ) -> c_int;
    }
}

// gst mpeghuimanager's limits: the largest frame it updates, the room mpeghUIManager.h asks
// for on top of the frame, and the longest AudioSceneConfig
const MAX_MPEGH_FRAME_SIZE: usize = 65536;
const UPDATE_MHAS_HEADROOM: usize = 256;
const MAX_CONFIG_LENGTH: usize = 104226;
// ActionEvents queue while no audio scene can take them, so this many is a stuck application
const MAX_PENDING_ACTION_EVENTS: usize = 64;
// mpeghdec keeps about 90 streams of settings in this much persistence memory
const DEFAULT_PERSISTENCE_MEMORY_SIZE: usize = 4096;

fn library_result(code: c_int) -> Result<(), c_int> {
    if code == ffi::MPEGH_UI_OK {
        Ok(())
    } else {
        Err(code)
    }
}

#[derive(Debug)]
struct UiManager {
    handle: NonNull<ffi::MpeghUiManager>,
    // mpeghdec reads and writes this memory, aligned to 16 bits, for as long as the handle is open
    persistence: Vec<u16>,
    scene_state: Vec<u8>,
}

// SAFETY: the handle and the memory it uses are owned by one `UiManager` and only reached
// through `&mut self`, and mpeghdec keeps no thread-local state.
unsafe impl Send for UiManager {}

impl Drop for UiManager {
    fn drop(&mut self) {
        // SAFETY: the handle came from mpegh_UI_Manager_Open and is closed once, here,
        // before `persistence` is freed.
        unsafe { ffi::mpegh_UI_Manager_Close(self.handle.as_ptr()) };
    }
}

impl UiManager {
    fn open(stored_persistence: Option<&[u8]>) -> Result<Self, G2gError> {
        // SAFETY: mpegh_UI_Manager_Open takes no arguments and returns null on failure.
        let handle = unsafe { ffi::mpegh_UI_Manager_Open() };
        let handle = NonNull::new(handle).ok_or(G2gError::Hardware(HardwareError::Other))?;
        let persistence = match stored_persistence {
            Some(bytes) => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&pair| u16::from_ne_bytes(pair))
                .collect(),
            None => alloc::vec![0; DEFAULT_PERSISTENCE_MEMORY_SIZE / 2],
        };
        let mut manager = UiManager {
            handle,
            persistence,
            scene_state: alloc::vec![0; MAX_CONFIG_LENGTH],
        };
        let size = c_ushort::try_from(manager.persistence.len() * 2)
            .map_err(|_| G2gError::CapsMismatch)?;
        // SAFETY: the handle is live and `persistence` holds `size` bytes, which outlive the
        // handle (see the Drop impl).
        let code = unsafe {
            ffi::mpegh_UI_SetPersistenceMemory(
                manager.handle.as_ptr(),
                manager.persistence.as_mut_ptr().cast::<c_void>(),
                size,
            )
        };
        // mpeghdec takes data another version of it wrote, without its settings
        if code != ffi::MPEGH_UI_OK && code != ffi::MPEGH_UI_OK_BUT_NO_VALID_DATA {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        Ok(manager)
    }

    fn feed(&mut self, frame: &mut [u8]) -> Result<(), c_int> {
        let length = c_uint::try_from(frame.len()).expect("bounded by MAX_MPEGH_FRAME_SIZE");
        // SAFETY: the handle is live and `frame` is valid for `length` bytes.
        library_result(unsafe {
            ffi::mpegh_UI_FeedMHAS(self.handle.as_ptr(), frame.as_mut_ptr(), length)
        })
    }

    fn apply_action(&mut self, action_event: &str) -> Result<(), c_int> {
        let length = c_uint::try_from(action_event.len()).expect("bounded by MAX_CONFIG_LENGTH");
        let mut flags_out: c_uint = 0;
        // SAFETY: the handle is live, `action_event` is valid for `length` bytes, which is the
        // length the call reads, and `flags_out` is a writable out parameter.
        library_result(unsafe {
            ffi::mpegh_UI_ApplyXmlAction(
                self.handle.as_ptr(),
                action_event.as_ptr().cast::<c_char>(),
                length,
                &mut flags_out,
            )
        })
    }

    // each AudioSceneConfig completed since the last call, joined from the parts it came in
    fn scene_states(&mut self) -> Result<Vec<String>, G2gError> {
        let mut complete = Vec::new();
        let mut scene: Option<String> = None;
        let mut flags_in: c_uint = 0;
        loop {
            self.scene_state[0] = 0;
            let mut flags_out: c_uint = 0;
            // SAFETY: the handle is live, `scene_state` has room for MAX_CONFIG_LENGTH bytes
            // and `flags_out` is a writable out parameter.
            let code = unsafe {
                ffi::mpegh_UI_GetXmlSceneState(
                    self.handle.as_ptr(),
                    self.scene_state.as_mut_ptr().cast::<c_char>(),
                    MAX_CONFIG_LENGTH as c_uint,
                    flags_in,
                    &mut flags_out,
                )
            };
            flags_in = 0;
            if code != ffi::MPEGH_UI_OK || flags_out & ffi::MPEGH_UI_NO_CHANGE != 0 {
                return Ok(complete);
            }
            if flags_out & ffi::MPEGH_UI_CONTINUES_XML != 0 {
                // the parts before this one were lost to a failed call
                if scene.is_none() {
                    flags_in = ffi::MPEGH_UI_FORCE_RESTART_XML;
                    continue;
                }
            } else {
                scene = None;
            }
            let length = self
                .scene_state
                .iter()
                .position(|&byte| byte == 0)
                .ok_or(G2gError::Hardware(HardwareError::Other))?;
            let part = core::str::from_utf8(&self.scene_state[..length])
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            if part.is_empty() {
                continue;
            }
            let joined = scene.get_or_insert_with(String::new);
            joined.push_str(part);
            if flags_out & ffi::MPEGH_UI_INCOMPLETE_XML == 0 {
                complete.extend(scene.take());
            }
        }
    }

    // the frame grows by the UI packets mpeghdec inserts, into the room after `length`
    fn update(&mut self, buffer: &mut [u8], length: usize) -> Result<usize, c_int> {
        let capacity = c_uint::try_from(buffer.len()).expect("bounded by MAX_MPEGH_FRAME_SIZE");
        let mut updated = c_uint::try_from(length).expect("bounded by MAX_MPEGH_FRAME_SIZE");
        // SAFETY: the handle is live, `buffer` is valid for `capacity` bytes, and `updated`
        // is a writable in and out parameter.
        library_result(unsafe {
            ffi::mpegh_UI_UpdateMHAS(
                self.handle.as_ptr(),
                buffer.as_mut_ptr(),
                capacity,
                &mut updated,
            )
        })?;
        Ok(updated as usize)
    }

    fn persistence_bytes(&mut self) -> Option<Vec<u8>> {
        let mut block: *mut c_void = core::ptr::null_mut();
        let mut size: c_ushort = 0;
        // SAFETY: the handle is live and both out parameters are writable.
        let code = unsafe {
            ffi::mpegh_UI_GetPersistenceMemory(self.handle.as_ptr(), &mut block, &mut size)
        };
        if code != ffi::MPEGH_UI_OK || block.is_null() || size == 0 {
            return None;
        }
        // SAFETY: mpeghdec hands back the `persistence` memory set at open, `size` bytes long.
        let bytes = unsafe { core::slice::from_raw_parts(block.cast::<u8>(), usize::from(size)) };
        Some(bytes.to_vec())
    }
}

// a full AudioSceneConfig lists its presets, the short one mpeghdec reports without a scene does not
fn scene_is_valid(config: &str) -> bool {
    config.contains("<presets>")
}

const UI_MANAGER_FORMATS: &[AudioFormat] = &[AudioFormat::MpeghMhas];

#[derive(Debug, Default)]
pub struct MpeghUiManager {
    manager: Option<UiManager>,
    framer: MhasFramer,
    input_persistence: Option<Vec<u8>>,
    output_persistence: Option<Vec<u8>>,
    config: String,
    scene_valid: bool,
    pending_action_events: VecDeque<String>,
    last_action_event: String,
    sequence: u64,
    bus: Option<BusHandle>,
    log_name: LogName,
}

impl MpeghUiManager {
    pub fn new() -> Self {
        Self::default()
    }

    fn element_name(&self) -> String {
        self.log_name
            .instance()
            .unwrap_or_else(|| short_type_name::<Self>())
            .to_string()
    }

    fn post(&self, message: BusMessage) {
        if let Some(bus) = &self.bus {
            bus.try_post(message);
        }
    }

    // The access unit with the user's settings inserted as UI packets. A frame mpeghdec
    // cannot update still decodes, so it passes through unchanged, as in gst.
    fn update_access_unit(
        &mut self,
        access_unit: MhasAccessUnit,
    ) -> Result<MhasAccessUnit, G2gError> {
        if access_unit.data.len() > MAX_MPEGH_FRAME_SIZE {
            return Ok(access_unit);
        }
        if self.manager.is_none() {
            self.manager = Some(UiManager::open(self.input_persistence.as_deref())?);
        }
        let element = self.element_name();
        let pts_ns = access_unit.pts_ns.unwrap_or(FrameTiming::PTS_NONE);
        let manager = self.manager.as_mut().expect("opened above");
        let length = access_unit.data.len();
        let mut buffer = access_unit.data.clone();
        buffer.resize(length + UPDATE_MHAS_HEADROOM, 0);
        if manager.feed(&mut buffer[..length]).is_err() {
            return Ok(access_unit);
        }

        let mut applied = false;
        let mut results = Vec::new();
        while let Some(action_event) = self.pending_action_events.pop_front() {
            let outcome = manager.apply_action(&action_event);
            // without a scene mpeghdec may not know the groups and presets an event names yet
            if outcome.is_err() && !self.scene_valid {
                self.pending_action_events.push_front(action_event);
                break;
            }
            applied |= outcome.is_ok();
            results.push(BusMessage::MpeghUiEventResult {
                element: element.clone(),
                pts_ns,
                action_event,
                outcome,
            });
        }
        if applied {
            if let Some(bytes) = manager.persistence_bytes() {
                self.output_persistence = Some(bytes);
            }
        }
        let scenes = manager.scene_states()?;
        let updated = manager.update(&mut buffer, length);

        for message in results {
            self.post(message);
        }
        for config in scenes {
            self.scene_valid = scene_is_valid(&config);
            self.config.clone_from(&config);
            self.post(BusMessage::MpeghUiConfig {
                element: element.clone(),
                pts_ns,
                config,
            });
        }
        // mpeghdec refuses to update before the stream's first config and when an upstream
        // UI manager already inserted its packets
        let Ok(updated_length) = updated else {
            return Ok(access_unit);
        };
        buffer.truncate(updated_length);
        Ok(MhasAccessUnit {
            data: buffer,
            pts_ns: access_unit.pts_ns,
        })
    }

    async fn push_access_unit(
        &mut self,
        access_unit: MhasAccessUnit,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let updated = self.update_access_unit(access_unit)?;
        let timing = FrameTiming {
            pts_ns: updated.pts_ns.unwrap_or(FrameTiming::PTS_NONE),
            ..FrameTiming::default()
        };
        let frame = Frame::from_system_bytes(updated.data, timing, self.sequence);
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

impl AsyncElement for MpeghUiManager {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        require_format(UI_MANAGER_FORMATS, upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(passthrough_caps(UI_MANAGER_FORMATS))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        require_format(UI_MANAGER_FORMATS, absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn set_bus(&mut self, bus: BusHandle) {
        self.bus = Some(bus);
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MPEG-H UI manager",
            "Codec/Audio",
            "Applies MPEG-H Audio user interaction to an MHAS stream with Fraunhofer's mpeghdec",
            crate::AARON_BOXER,
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let data = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    for access_unit in self.framer.push(data, frame.timing.pts())? {
                        self.push_access_unit(access_unit, out).await?;
                    }
                }
                PipelinePacket::Eos => {
                    if let Some(access_unit) = self.framer.finish() {
                        self.push_access_unit(access_unit, out).await?;
                    }
                    if let Some(bytes) =
                        self.manager.as_mut().and_then(UiManager::persistence_bytes)
                    {
                        self.output_persistence = Some(bytes);
                    }
                }
                PipelinePacket::Flush => {
                    self.framer.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MPEGHUIMANAGER_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "action-event" => {
                let action_event = value.as_str().ok_or(PropError::Type)?;
                if action_event.is_empty()
                    || action_event.len() > MAX_CONFIG_LENGTH
                    || self.pending_action_events.len() >= MAX_PENDING_ACTION_EVENTS
                {
                    return Err(PropError::Value);
                }
                self.pending_action_events
                    .push_back(action_event.to_string());
                self.last_action_event = action_event.to_string();
            }
            "input-persistence" => {
                // mpeghdec holds the memory it was opened with
                if self.manager.is_some() {
                    return Err(PropError::Value);
                }
                let text = value.as_str().ok_or(PropError::Type)?;
                let bytes = base64_decode(text).ok_or(PropError::Value)?;
                // mpeghdec addresses the memory in 16-bit units with a 16-bit byte count
                if bytes.is_empty() || bytes.len() % 2 != 0 || bytes.len() > usize::from(u16::MAX) {
                    return Err(PropError::Value);
                }
                self.input_persistence = Some(bytes);
            }
            "config" | "output-persistence" => return Err(PropError::ReadOnly),
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let base64 = |bytes: &Option<Vec<u8>>| {
            PropValue::Str(bytes.as_deref().map(base64_encode).unwrap_or_default())
        };
        Some(match name {
            "action-event" => PropValue::Str(self.last_action_event.clone()),
            "config" => PropValue::Str(self.config.clone()),
            "input-persistence" => base64(&self.input_persistence),
            "output-persistence" => base64(&self.output_persistence),
            _ => return None,
        })
    }
}

static MPEGHUIMANAGER_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "action-event",
        PropKind::Str,
        "an ActionEvent XML message to apply at the next frame, the latest one when read",
    ),
    PropertySpec::new(
        "config",
        PropKind::Str,
        "the latest AudioSceneConfig XML message",
    )
    .read_only(),
    PropertySpec::new(
        "input-persistence",
        PropKind::Str,
        "base64 settings an earlier run stored from output-persistence, set before the first frame",
    ),
    PropertySpec::new(
        "output-persistence",
        PropKind::Str,
        "base64 settings to store and pass to input-persistence in a later run",
    )
    .read_only(),
];

impl PadTemplates for MpeghUiManager {
    fn pad_templates() -> Vec<PadTemplate> {
        pad_templates(UI_MANAGER_FORMATS)
    }
}

use core::future::Future;
use core::pin::Pin;
use std::sync::Mutex;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::element::QosMessage;
use g2g_core::frame::Frame;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ClockSync, ConfigureOutcome, Dim, G2gError,
    HardwareError, Interlace, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PresentationPacer, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
    MAX_LATENESS_PROPERTY, QOS_INTERVAL_PROPERTY,
};
use ndk::hardware_buffer_format::HardwareBufferFormat;
use ndk::native_window::NativeWindow;

use crate::clock::wait_to_present;

const DEFAULT_CHANNEL: &str = "default";
const RGBA_BYTES_PER_PIXEL: usize = 4;

// Sinks draw while holding this lock, so detach_surface returns only after the last draw.
static SURFACES: Mutex<BTreeMap<String, NativeWindow>> = Mutex::new(BTreeMap::new());

pub fn attach_surface(channel: &str, window: NativeWindow) {
    surfaces().insert(channel.to_string(), window);
}

pub fn detach_surface(channel: &str) {
    surfaces().remove(channel);
}

fn surfaces() -> std::sync::MutexGuard<'static, BTreeMap<String, NativeWindow>> {
    SURFACES.lock().expect("a sink panicked while drawing")
}

// Copies RGBA frames into the window attached under its channel, and drops them while none is.
#[derive(Debug)]
pub struct SurfaceSink {
    channel: String,
    width: usize,
    height: usize,
    presented: u64,
    pacer: PresentationPacer,
}

impl Default for SurfaceSink {
    fn default() -> Self {
        Self::new()
    }
}

impl SurfaceSink {
    pub fn new() -> Self {
        Self {
            channel: DEFAULT_CHANNEL.to_string(),
            width: 0,
            height: 0,
            presented: 0,
            pacer: PresentationPacer::for_video_sink(),
        }
    }

    pub fn presented(&self) -> u64 {
        self.presented
    }

    fn present(&mut self, frame: &Frame) -> Result<(), G2gError> {
        let bytes = frame
            .domain
            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
        let row_bytes = self.width * RGBA_BYTES_PER_PIXEL;
        if bytes.len() != row_bytes * self.height {
            return Err(G2gError::CapsMismatch);
        }
        let surfaces = surfaces();
        let Some(window) = surfaces.get(&self.channel) else {
            return Ok(());
        };
        let hardware = |_| G2gError::Hardware(HardwareError::Other);
        let width = i32::try_from(self.width).map_err(|_| G2gError::CapsMismatch)?;
        let height = i32::try_from(self.height).map_err(|_| G2gError::CapsMismatch)?;
        window
            .set_buffers_geometry(width, height, Some(HardwareBufferFormat::R8G8B8A8_UNORM))
            .map_err(hardware)?;
        let mut buffer = window.lock(None).map_err(hardware)?;
        if (buffer.width(), buffer.height()) != (self.width, self.height) {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        let lines = buffer
            .lines()
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        for (line, row) in lines.zip(bytes.chunks_exact(row_bytes)) {
            for (pixel, byte) in line.iter_mut().zip(row) {
                pixel.write(*byte);
            }
        }
        self.presented += 1;
        Ok(())
    }
}

impl AsyncElement for SurfaceSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&rgba_any())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(rgba_any()))
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pacer.take_qos()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SURFACESINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if name == "channel" {
            self.channel = value.as_str().ok_or(PropError::Type)?.to_string();
            return Ok(());
        }
        self.pacer
            .set_property(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if name == "channel" {
            return Some(PropValue::Str(self.channel.clone()));
        }
        self.pacer.get_property(name)
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(Caps::RawVideo {
                    format: RawVideoFormat::Rgba8,
                    width,
                    height,
                    ..
                }) => {
                    let fixed = |dim: Dim| dim.fixed().ok_or(G2gError::CapsMismatch);
                    self.width = fixed(width)? as usize;
                    self.height = fixed(height)? as usize;
                }
                PipelinePacket::CapsChanged(_) => return Err(G2gError::CapsMismatch),
                PipelinePacket::DataFrame(frame) => {
                    let paced = self.pacer.judge(frame.timing.pts_ns, self.presented);
                    if wait_to_present(paced).await {
                        self.present(&frame)?;
                    }
                }
                PipelinePacket::Segment(segment) => self.pacer.set_segment(segment),
                PipelinePacket::Flush => self.pacer.flush(),
                _ => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for SurfaceSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(rgba_any()))])
    }
}

fn rgba_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

static SURFACESINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "channel",
        PropKind::Str,
        "name the application attaches its Surface under (default \"default\")",
    ),
    MAX_LATENESS_PROPERTY,
    QOS_INTERVAL_PROPERTY,
];

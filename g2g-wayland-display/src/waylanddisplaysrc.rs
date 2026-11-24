// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;
use std::path::{Path, PathBuf};
use std::time::Duration;

use g2g_core::caps::{fps_period_ns, framerate_q16_from_fraction};
use g2g_core::frame::Frame;
use g2g_core::log::Target;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AllocationParams, Caps, CapsConstraint, CapsSet, CaptureAnchor, ClockSync, Colorimetry,
    ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError, HardwareError, Interlace,
    LatencyReport, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PixelAspect, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};
use g2g_plugins::numbuffers::{
    finished_at_zero_limit, get_num_buffers, num_buffers_property, set_num_buffers,
};
use g2g_plugins::windowinput::WindowInput;
use smithay::reexports::calloop::channel::{channel, Channel, Sender};
use tokio::sync::{oneshot, watch};

use crate::display::{
    run_display, DisplayChannels, DisplaySettings, DisplaySize, FrameReply, Pacing, Picture,
    Rendering,
};
use crate::gpu::{GpuFormat, GpuOutput};

#[derive(Debug, Clone, Copy)]
struct OutputFormat {
    format: RawVideoFormat,
    colorimetry: Colorimetry,
}

// drm's abgr8888 is R, G, B, A in memory
const RGBA_OUTPUT: OutputFormat = OutputFormat {
    format: RawVideoFormat::Rgba8,
    colorimetry: Colorimetry::UNKNOWN,
};
const NV12_OUTPUT: OutputFormat = OutputFormat {
    format: RawVideoFormat::Nv12,
    colorimetry: Colorimetry::BT709,
};
pub const SOFTWARE_RENDERING: &str = "software";
const DEFAULT_RENDER_NODE: &str = "/dev/dri/renderD128";
const DEFAULT_WIDTH: u32 = 1920;
const DEFAULT_HEIGHT: u32 = 1080;
const DEFAULT_FRAMERATE: u32 = 60;
const REPEAT_AFTER: Duration = Duration::from_secs(1);
const SYS_CLASS_DRM: &str = "/sys/class/drm";

// the kernel driver bound to a render node's device, such as amdgpu or nvidia
pub fn render_node_driver(render_node: &Path) -> std::io::Result<String> {
    let not_a_node = || std::io::Error::from(std::io::ErrorKind::InvalidInput);
    let name = render_node.file_name().ok_or_else(not_a_node)?;
    let driver = std::fs::read_link(Path::new(SYS_CLASS_DRM).join(name).join("device/driver"))?;
    driver
        .file_name()
        .and_then(|driver| driver.to_str())
        .map(str::to_owned)
        .ok_or_else(not_a_node)
}

pub type InputSender = Sender<WindowInput>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayNames {
    // clients set WAYLAND_DISPLAY to this socket name
    pub wayland_socket: String,
    // X11 clients set DISPLAY to this, present when the source runs Xwayland
    pub x11_display: Option<String>,
}

pub struct WaylandDisplaySrc {
    width: u32,
    height: u32,
    framerate: u32,
    frame_limit: u64,
    render_node: String,
    xwayland: bool,
    output_memory: MemoryDomainKind,
    format: RawVideoFormat,
    configured: bool,
    clock_sync: Option<ClockSync>,
    input_sender: InputSender,
    input: Option<Channel<WindowInput>>,
    names: watch::Sender<Option<DisplayNames>>,
}

impl core::fmt::Debug for WaylandDisplaySrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WaylandDisplaySrc")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("framerate", &self.framerate)
            .field("frame_limit", &self.frame_limit)
            .field("render_node", &self.render_node)
            .field("xwayland", &self.xwayland)
            .field("output_memory", &self.output_memory)
            .field("format", &self.format)
            .field("names", &*self.names.borrow())
            .finish_non_exhaustive()
    }
}

impl Default for WaylandDisplaySrc {
    fn default() -> Self {
        let (input_sender, input) = channel();
        Self {
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            framerate: DEFAULT_FRAMERATE,
            frame_limit: u64::MAX,
            render_node: String::from(DEFAULT_RENDER_NODE),
            xwayland: false,
            output_memory: MemoryDomainKind::System,
            format: RGBA_OUTPUT.format,
            configured: false,
            clock_sync: None,
            input_sender,
            input: Some(input),
            names: watch::channel(None).0,
        }
    }
}

impl WaylandDisplaySrc {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_framerate(mut self, framerate: u32) -> Self {
        self.framerate = framerate;
        self
    }

    pub fn with_num_buffers(mut self, frame_limit: u64) -> Self {
        self.frame_limit = frame_limit;
        self
    }

    pub fn with_render_node(mut self, render_node: impl Into<String>) -> Self {
        self.render_node = render_node.into();
        self
    }

    pub fn with_xwayland(mut self) -> Self {
        self.xwayland = true;
        self
    }

    pub fn render_node(&self) -> &str {
        &self.render_node
    }

    pub fn input_sender(&self) -> InputSender {
        self.input_sender.clone()
    }

    pub fn display_names(&self) -> watch::Receiver<Option<DisplayNames>> {
        // None until the run opens the wayland socket and, with Xwayland, its window manager runs
        self.names.subscribe()
    }

    fn rendering(&self) -> Result<Rendering, G2gError> {
        if self.render_node == SOFTWARE_RENDERING {
            return Ok(Rendering::Software);
        }
        let output = match self.output_memory {
            MemoryDomainKind::DmaBuf => GpuOutput::DmaBuf,
            MemoryDomainKind::System => GpuOutput::System,
            _ => return Err(G2gError::AllocationConflict),
        };
        let format = match self.format {
            RawVideoFormat::Rgba8 => GpuFormat::Rgba,
            RawVideoFormat::Nv12 => GpuFormat::Nv12,
            _ => return Err(G2gError::CapsMismatch),
        };
        Ok(Rendering::Gpu {
            render_node: PathBuf::from(&self.render_node),
            output,
            format,
        })
    }

    fn output_formats(&self) -> Vec<OutputFormat> {
        // nv12 halves both dimensions for its chroma plane
        let even = self.width.is_multiple_of(2) && self.height.is_multiple_of(2);
        if self.render_node == SOFTWARE_RENDERING || !even {
            return Vec::from([RGBA_OUTPUT]);
        }
        Vec::from([RGBA_OUTPUT, NV12_OUTPUT])
    }

    fn caps_for(&self, output: OutputFormat) -> Result<Caps, G2gError> {
        let framerate =
            framerate_q16_from_fraction(self.framerate, 1).ok_or(G2gError::CapsMismatch)?;
        Ok(Caps::RawVideo {
            format: output.format,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(framerate),
            interlace: Interlace::Progressive,
            colorimetry: output.colorimetry,
            pixel_aspect: PixelAspect::SQUARE,
        })
    }

    pub(crate) fn rgba_caps(&self) -> Result<Caps, G2gError> {
        self.caps_for(RGBA_OUTPUT)
    }

    fn produced_caps(&self) -> Result<CapsSet, G2gError> {
        let alternatives = self
            .output_formats()
            .into_iter()
            .map(|output| self.caps_for(output))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CapsSet::from_alternatives(alternatives))
    }
}

#[derive(Debug)]
struct DisplayThread {
    frame_requests: Option<Sender<FrameReply>>,
    handle: Option<std::thread::JoinHandle<Result<(), G2gError>>>,
}

impl DisplayThread {
    fn request_frame(&self, reply: FrameReply) -> bool {
        self.frame_requests
            .as_ref()
            .is_some_and(|frame_requests| frame_requests.send(reply).is_ok())
    }

    fn join(&mut self) -> Result<(), G2gError> {
        // the display loop stops once its frame request channel closes
        self.frame_requests = None;
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        handle
            .join()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?
    }
}

impl Drop for DisplayThread {
    fn drop(&mut self) {
        // a cancelled run must not leave the compositor and Xwayland running past the process exit
        if let Err(error) = self.join() {
            g2g_core::g2g_warn!(
                Target::category("WaylandDisplaySrc"),
                "the display thread failed after the run was cancelled: {error:?}"
            );
        }
    }
}

impl SourceLoop for WaylandDisplaySrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.rgba_caps())
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(self.produced_caps().map(CapsConstraint::Produces))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let mut accepted = None;
        for output in self.output_formats() {
            if *absolute_caps == self.caps_for(output)? {
                accepted = Some(output.format);
            }
        }
        self.format = accepted.ok_or(G2gError::CapsMismatch)?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Wayland display source",
            "Source/Video",
            "Runs a headless Wayland compositor and produces its output picture",
            g2g_plugins::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPERTIES: &[PropertySpec] = &[
            PropertySpec::new("width", PropKind::Uint, "output width in pixels")
                .with_default("1920"),
            PropertySpec::new("height", PropKind::Uint, "output height in pixels")
                .with_default("1080"),
            PropertySpec::new(
                "framerate",
                PropKind::Uint,
                "highest output rate in frames per second, also the refresh clients see",
            )
            .with_default("60"),
            PropertySpec::new(
                "render-node",
                PropKind::Str,
                "DRM render node the compositor renders with, or software for the CPU renderer",
            )
            .with_default(DEFAULT_RENDER_NODE),
            PropertySpec::new(
                "xwayland",
                PropKind::Bool,
                "run Xwayland so X11 apps can connect, needs the Xwayland binary on PATH",
            )
            .with_default("false"),
            num_buffers_property("frames to render then EOS (-1 = forever)"),
        ];
        PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let as_u32 = |value: &PropValue| {
            value
                .as_uint()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or(PropError::Type)
        };
        match name {
            "width" => self.width = as_u32(&value)?,
            "height" => self.height = as_u32(&value)?,
            "framerate" => self.framerate = as_u32(&value)?,
            "render-node" => {
                self.render_node = value.as_str().ok_or(PropError::Type)?.to_owned();
            }
            "xwayland" => self.xwayland = value.as_bool().ok_or(PropError::Type)?,
            "num-buffers" => set_num_buffers(&mut self.frame_limit, &value)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "width" => Some(PropValue::Uint(self.width.into())),
            "height" => Some(PropValue::Uint(self.height.into())),
            "framerate" => Some(PropValue::Uint(self.framerate.into())),
            "render-node" => Some(PropValue::Str(self.render_node.clone())),
            "xwayland" => Some(PropValue::Bool(self.xwayland)),
            "num-buffers" => Some(get_num_buffers(self.frame_limit)),
            _ => None,
        }
    }

    fn output_memory(&self) -> MemoryDomainKind {
        self.output_memory
    }

    fn output_domains(&self) -> DomainSet {
        if self.render_node == SOFTWARE_RENDERING {
            return DomainSet::only(MemoryDomainKind::System);
        }
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::DmaBuf)
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        if let Ok(resolved) = params.clone().resolve_for_producer(self.output_domains()) {
            self.output_memory = resolved.domain;
        }
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.clock_sync = Some(sync);
    }

    fn latency(&self) -> LatencyReport {
        LatencyReport::live(fps_period_ns(self.framerate), None)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            if finished_at_zero_limit(self.frame_limit, out).await? {
                return Ok(0);
            }
            let rendering = self.rendering()?;
            let input = self.input.take().ok_or(G2gError::NotConfigured)?;
            let period_ns = fps_period_ns(self.framerate);
            let size = DisplaySize {
                width: self.width,
                height: self.height,
            };
            let pacing = Pacing {
                frame_period: Duration::from_nanos(period_ns),
                repeat_after: REPEAT_AFTER,
            };
            let settings = DisplaySettings {
                size,
                pacing,
                rendering,
                xwayland: self.xwayland,
            };
            let (frame_request_sender, frame_requests) = channel::<FrameReply>();
            let (names_sender, names_receiver) = oneshot::channel();
            let handle = std::thread::Builder::new()
                .name(String::from("g2g-waylanddisplaysrc"))
                .spawn(move || {
                    run_display(
                        settings,
                        DisplayChannels {
                            frame_requests,
                            input,
                            names: names_sender,
                        },
                    )
                })
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            let mut display = DisplayThread {
                frame_requests: Some(frame_request_sender),
                handle: Some(handle),
            };
            let names = match names_receiver.await {
                Ok(Ok(names)) => names,
                Ok(Err(error)) => {
                    display.join()?;
                    return Err(error);
                }
                Err(_) => {
                    display.join()?;
                    return Err(G2gError::Hardware(HardwareError::Other));
                }
            };
            g2g_core::g2g_info!(
                Target::category("WaylandDisplaySrc"),
                "clients connect with WAYLAND_DISPLAY={} DISPLAY={}",
                names.wayland_socket,
                names.x11_display.as_deref().unwrap_or("(no Xwayland)")
            );
            self.names.send_replace(Some(names));

            let clock_sync = self.clock_sync.clone();
            let mut anchor = CaptureAnchor::new();
            let mut elapsed = 0u64;
            let mut previous_arrival_ns: Option<u64> = None;
            let mut sequence = 0u64;
            let mut downstream_open = true;
            while sequence < self.frame_limit {
                let (reply, picture) = oneshot::channel();
                if !display.request_frame(reply) {
                    break;
                }
                let Ok(picture) = picture.await else {
                    break;
                };
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                // frames come only when the picture changes
                elapsed += previous_arrival_ns.map_or(0, |previous| arrival_ns - previous);
                previous_arrival_ns = Some(arrival_ns);
                let pts = match &clock_sync {
                    Some(sync) => anchor.stamp(sync, elapsed, period_ns),
                    None => elapsed,
                };
                let domain = match picture {
                    Picture::System(bytes) => {
                        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice()))
                    }
                    Picture::DmaBuf(dmabuf) => MemoryDomain::DmaBuf(dmabuf),
                };
                let frame = Frame {
                    domain,
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: period_ns,
                        capture_ns: pts,
                        arrival_ns,
                        keyframe: true,
                        ..FrameTiming::default()
                    },
                    sequence,
                    meta: Default::default(),
                };
                if out.push(PipelinePacket::DataFrame(frame)).await.is_err() {
                    downstream_open = false;
                    break;
                }
                sequence += 1;
            }
            display.join()?;
            if downstream_open {
                out.push(PipelinePacket::Eos).await?;
            }
            Ok(sequence)
        })
    }
}

impl PadTemplates for WaylandDisplaySrc {
    fn pad_templates() -> Vec<PadTemplate> {
        let alternatives = [RGBA_OUTPUT, NV12_OUTPUT].map(|output| Caps::RawVideo {
            format: output.format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: Interlace::Progressive,
            colorimetry: output.colorimetry,
            pixel_aspect: PixelAspect::SQUARE,
        });
        Vec::from([PadTemplate::source(CapsSet::from_alternatives(
            alternatives.into(),
        ))])
    }
}

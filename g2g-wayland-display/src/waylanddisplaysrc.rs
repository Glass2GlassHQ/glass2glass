// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;
use std::path::{Path, PathBuf};
use std::time::Duration;

use g2g_core::caps::framerate_q16_period_ns;
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
use g2g_plugins::videosizeandrate::{
    requested_dimension, VideoSizeAndRate, MAX_REQUESTED_DIMENSION_TEXT, MAX_REQUESTED_FPS,
    MIN_REQUESTED_DIMENSION_TEXT, REQUESTED_DIMENSION,
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
const MIN_REQUESTED_FPS: u32 = 1;
const MIN_REQUESTED_FPS_TEXT: &str = "1";
const MAX_REQUESTED_FPS_TEXT: &str = "1000";
const REQUESTED_FRAMERATE: Rate = Rate::Range {
    min_q16: MIN_REQUESTED_FPS << 16,
    max_q16: MAX_REQUESTED_FPS << 16,
};
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
    size_and_rate: VideoSizeAndRate,
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
            .field("size_and_rate", &self.size_and_rate)
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
            size_and_rate: VideoSizeAndRate::new(
                DEFAULT_WIDTH,
                DEFAULT_HEIGHT,
                DEFAULT_FRAMERATE << 16,
            ),
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
        self.size_and_rate.pin_width(width);
        self.size_and_rate.pin_height(height);
        self
    }

    pub fn with_framerate(mut self, framerate: u32) -> Self {
        self.size_and_rate.pin_framerate_q16(framerate << 16);
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

    fn output_formats(&self) -> &'static [OutputFormat] {
        if self.render_node == SOFTWARE_RENDERING {
            return &[RGBA_OUTPUT];
        }
        &[RGBA_OUTPUT, NV12_OUTPUT]
    }

    fn produced_caps(&self) -> CapsSet {
        self.size_and_rate
            .produced_caps(self.output_formats(), REQUESTED_FRAMERATE, caps_for)
    }
}

// nv12 halves both dimensions for its chroma plane
fn fits_format(format: RawVideoFormat, dimension: &Dim) -> bool {
    format != RawVideoFormat::Nv12 || dimension.fixed().is_none_or(|d| d.is_multiple_of(2))
}

fn caps_for(output: OutputFormat, width: Dim, height: Dim, framerate: Rate) -> Option<Caps> {
    if !fits_format(output.format, &width) || !fits_format(output.format, &height) {
        return None;
    }
    Some(Caps::RawVideo {
        format: output.format,
        width,
        height,
        framerate,
        interlace: Interlace::Progressive,
        colorimetry: output.colorimetry,
        pixel_aspect: PixelAspect::SQUARE,
    })
}

pub(crate) const DECLARED_CAPS: Caps = Caps::RawVideo {
    format: RGBA_OUTPUT.format,
    width: REQUESTED_DIMENSION,
    height: REQUESTED_DIMENSION,
    framerate: REQUESTED_FRAMERATE,
    interlace: Interlace::Progressive,
    colorimetry: RGBA_OUTPUT.colorimetry,
    pixel_aspect: PixelAspect::SQUARE,
};

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
        let rgba = caps_for(
            RGBA_OUTPUT,
            Dim::Fixed(self.size_and_rate.width()),
            Dim::Fixed(self.size_and_rate.height()),
            Rate::Fixed(self.size_and_rate.framerate_q16()),
        );
        core::future::ready(rgba.ok_or(G2gError::CapsMismatch))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(self.produced_caps())))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format,
            width,
            height,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        let fits = fits_format(*format, width) && fits_format(*format, height);
        if !fits || !self.produced_caps().accepts(absolute_caps) {
            return Err(G2gError::CapsMismatch);
        }
        self.size_and_rate.configure(absolute_caps)?;
        self.format = *format;
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
                .with_range(MIN_REQUESTED_DIMENSION_TEXT, MAX_REQUESTED_DIMENSION_TEXT)
                .with_default("1920"),
            PropertySpec::new("height", PropKind::Uint, "output height in pixels")
                .with_range(MIN_REQUESTED_DIMENSION_TEXT, MAX_REQUESTED_DIMENSION_TEXT)
                .with_default("1080"),
            PropertySpec::new(
                "framerate",
                PropKind::Uint,
                "highest output rate in frames per second, also the refresh clients see",
            )
            .with_range(MIN_REQUESTED_FPS_TEXT, MAX_REQUESTED_FPS_TEXT)
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
        match name {
            "width" => self.size_and_rate.pin_width(requested_dimension(&value)?),
            "height" => self.size_and_rate.pin_height(requested_dimension(&value)?),
            "framerate" => {
                let fps = value.as_uint().ok_or(PropError::Type)?;
                let fps = u32::try_from(fps)
                    .ok()
                    .filter(|fps| (MIN_REQUESTED_FPS..=MAX_REQUESTED_FPS).contains(fps))
                    .ok_or(PropError::Value)?;
                self.size_and_rate.pin_framerate_q16(fps << 16);
            }
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
            "width" => Some(PropValue::Uint(self.size_and_rate.width().into())),
            "height" => Some(PropValue::Uint(self.size_and_rate.height().into())),
            "framerate" => Some(PropValue::Uint(
                (self.size_and_rate.framerate_q16() >> 16).into(),
            )),
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
        let period_ns = framerate_q16_period_ns(self.size_and_rate.framerate_q16()).unwrap_or(0);
        LatencyReport::live(period_ns, None)
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
            let period_ns = framerate_q16_period_ns(self.size_and_rate.framerate_q16())
                .ok_or(G2gError::CapsMismatch)?;
            let size = DisplaySize {
                width: self.size_and_rate.width(),
                height: self.size_and_rate.height(),
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

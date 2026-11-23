// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::num::NonZeroU32;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, FrameCallbackData},
    delegate_dispatch2, delegate_registry,
    dmabuf::{DmabufFeedback, DmabufHandler, DmabufState},
    output::{OutputHandler, OutputState},
    reexports::calloop::{
        channel::{channel, Channel, Event as ChanEvent, Sender as CalloopSender},
        EventLoop,
    },
    reexports::calloop_wayland_source::WaylandSource,
    reexports::client::{
        delegate_noop,
        globals::{registry_queue_init, GlobalListContents},
        protocol::{
            wl_buffer, wl_output, wl_registry, wl_shm, wl_subsurface::WlSubsurface, wl_surface,
        },
        Connection, Dispatch, QueueHandle, WEnum,
    },
    reexports::protocols::wp::color_representation::v1::client::{
        wp_color_representation_manager_v1::{self, WpColorRepresentationManagerV1},
        wp_color_representation_surface_v1::{Coefficients, Range, WpColorRepresentationSurfaceV1},
    },
    reexports::protocols::wp::commit_timing::v1::client::{
        wp_commit_timer_v1::WpCommitTimerV1, wp_commit_timing_manager_v1::WpCommitTimingManagerV1,
    },
    reexports::protocols::wp::fifo::v1::client::{
        wp_fifo_manager_v1::WpFifoManagerV1, wp_fifo_v1::WpFifoV1,
    },
    reexports::protocols::wp::linux_dmabuf::zv1::client::{
        zwp_linux_buffer_params_v1, zwp_linux_dmabuf_feedback_v1,
        zwp_linux_dmabuf_v1::{self, ZwpLinuxDmabufV1},
    },
    reexports::protocols::wp::presentation_time::client::{
        wp_presentation::{self, WpPresentation},
        wp_presentation_feedback::{self, WpPresentationFeedback},
    },
    reexports::protocols::wp::viewporter::client::{
        wp_viewport::WpViewport, wp_viewporter::WpViewporter,
    },
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::{
        xdg::{
            window::{Window, WindowConfigure, WindowDecorations, WindowHandler},
            XdgShell,
        },
        WaylandSurface,
    },
    shm::{
        slot::{Buffer, SlotPool},
        Shm, ShmHandler,
    },
    subcompositor::SubcompositorState,
};

use crate::capsfilter::DRM_FOURCC_FORMATS;
use crate::clock::wait_to_present;
use crate::display_aspect::{
    display_size, letterbox, ViewportRectangle, FORCE_ASPECT_RATIO_NAME,
    FORCE_ASPECT_RATIO_PROPERTY,
};
use crate::dmabufimport::DmaBufImportCache;
use crate::mathf::median;
use crate::paddedrows::dmabuf_plane_layout;
use crate::pixel::{carries_yuv, even_dims_required, frame_byte_size, row_bytes};
use crate::videoflip::frame_orientation;
use crate::videofx::any_geometry;
use crate::worker_ready::Handshake;
#[cfg(feature = "metadata")]
use crate::{
    drmcard::{draw_composition, same_composition, Card, DEFAULT_CARD},
    overlaycomposition::{RENDER_OVERLAYS_NAME, RENDER_OVERLAYS_PROPERTY},
};
use g2g_core::caps_parse::parse_raw_format;
use g2g_core::element::{PresentationStats, QosMessage};
use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{DomainSet, MemoryDomain, MemoryDomainKind, OwnedDmaBuf, SystemSlice};
use g2g_core::meta::{Orientation, OrientationSet};
use g2g_core::metrics::{monotonic_ns, LatencyHistogram, LatencySnapshot};
use g2g_core::{
    AsyncElement, BusHandle, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockPriority,
    ClockSync, Colorimetry, ConfigureOutcome, Dim, ElementMetadata, G2gError, HardwareError,
    MatrixCoefficients, OutputSink, PadTemplate, PadTemplates, PipelineClock, PipelinePacket,
    PixelAspect, PresentationPacer, PropError, PropKind, PropValue, PropertySpec, RawVideoFormat,
    MAX_LATENESS_PROPERTY, PROCESSING_DEADLINE_PROPERTY, QOS_INTERVAL_PROPERTY, SYNC_PROPERTY,
};

enum WorkerCmd {
    Frame(QueuedFrame),
    Resize(Picture),
    Shutdown,
}

enum QueuedPixels {
    System(SystemSlice),
    Imported(OwnedDmaBuf),
}

struct QueuedFrame {
    pixels: QueuedPixels,
    colorimetry: Colorimetry,
    presented: PresentedFrame,
    orientation: Orientation,
    ack: tokio::sync::oneshot::Sender<()>,
    #[cfg(feature = "metadata")]
    overlay: Option<g2g_core::OverlayCompositionMeta>,
}

struct PresentationDeadline {
    clock: Arc<dyn PipelineClock + Send + Sync>,
    deadline_ns: u64,
}

struct PresentedFrame {
    arrival_ns: u64,
    deadline: Option<PresentationDeadline>,
}

struct CommittedFrame {
    frame: PresentedFrame,
    committed_ns: u64,
}

#[derive(Debug, Default)]
struct DeadlineErrors {
    samples: Vec<i64>,
    latest: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PacingPolicy {
    #[default]
    Block,
    DropOldest,
}

// compositors send no frame callbacks to a surface they are not painting
const FRAME_CALLBACK_SILENCE_NS: u64 = 250_000_000;

// 8 bytes per sample, about 2.4 hours at 30 fps
const PER_FRAME_SAMPLE_CAPACITY: usize = 1 << 18;

#[derive(Debug)]
struct LatencyRecorder {
    histogram: LatencyHistogram,
    samples: std::sync::Mutex<Vec<u64>>,
}

impl LatencyRecorder {
    fn new() -> Self {
        Self {
            histogram: LatencyHistogram::new(),
            samples: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn record(&self, dur_ns: u64) {
        self.histogram.record(dur_ns);
        let mut samples = self.samples.lock().expect("latency samples lock");
        if samples.len() < PER_FRAME_SAMPLE_CAPACITY {
            samples.push(dur_ns);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ScheduleDrift {
    clock_ns: i64,
    media_ns: i64,
    deadline_ns: i64,
}

#[derive(Debug, Default)]
struct ScheduleDriftTrace {
    base: Option<(u64, u64, u64, u64)>,
    last_log_ns: u64,
}

impl ScheduleDriftTrace {
    const INTERVAL_NS: u64 = 1_000_000_000;

    fn sample(
        &mut self,
        mono_ns: u64,
        clock_ns: u64,
        pts_ns: u64,
        deadline_ns: u64,
    ) -> Option<ScheduleDrift> {
        let base = *self
            .base
            .get_or_insert((mono_ns, clock_ns, pts_ns, deadline_ns));
        if self.last_log_ns != 0 && mono_ns.saturating_sub(self.last_log_ns) < Self::INTERVAL_NS {
            return None;
        }
        self.last_log_ns = mono_ns;
        let elapsed = |now: u64, then: u64| now as i64 - then as i64;
        let mono = elapsed(mono_ns, base.0);
        Some(ScheduleDrift {
            clock_ns: elapsed(clock_ns, base.1) - mono,
            media_ns: elapsed(pts_ns, base.2) - mono,
            deadline_ns: elapsed(deadline_ns, base.3) - mono,
        })
    }
}

pub struct WaylandSink {
    title: String,
    app_id: String,
    cmd_tx: Option<CalloopSender<WorkerCmd>>,
    worker: Option<JoinHandle<()>>,
    picture: Option<Picture>,
    colorimetry: Colorimetry,
    compositor_formats: OnceLock<CompositorFormats>,
    force_aspect_ratio: bool,
    fullscreen: bool,
    frames_presented: Arc<AtomicU64>,
    latency: Arc<LatencyRecorder>,
    frames_dropped: Arc<AtomicU64>,
    last_frame_callback_ns: Arc<AtomicU64>,
    present_delay_ns: Arc<AtomicU64>,
    pacing: PacingPolicy,
    pacer: PresentationPacer,
    deadline_errors: Arc<std::sync::Mutex<DeadlineErrors>>,
    schedule_drift: ScheduleDriftTrace,
    #[cfg(feature = "metadata")]
    render_overlays: bool,
    #[cfg(feature = "metadata")]
    overlay_device: String,
}

impl g2g_core::log::LogSource for WaylandSink {
    fn log_category(&self) -> &'static str {
        g2g_core::log::short_type_name::<Self>()
    }
}

impl core::fmt::Debug for WaylandSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WaylandSink")
            .field("title", &self.title)
            .field("app_id", &self.app_id)
            .field("picture", &self.picture)
            .field(
                "frames_presented",
                &self.frames_presented.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl Default for WaylandSink {
    fn default() -> Self {
        Self::new()
    }
}

impl WaylandSink {
    pub fn new() -> Self {
        Self {
            title: String::from("glass2glass"),
            app_id: String::from("io.glass2glass.WaylandSink"),
            cmd_tx: None,
            worker: None,
            picture: None,
            colorimetry: Colorimetry::UNKNOWN,
            compositor_formats: OnceLock::new(),
            force_aspect_ratio: true,
            fullscreen: false,
            frames_presented: Arc::new(AtomicU64::new(0)),
            latency: Arc::new(LatencyRecorder::new()),
            frames_dropped: Arc::new(AtomicU64::new(0)),
            last_frame_callback_ns: Arc::new(AtomicU64::new(0)),
            present_delay_ns: Arc::new(AtomicU64::new(0)),
            pacing: PacingPolicy::default(),
            pacer: PresentationPacer::for_video_sink(),
            deadline_errors: Arc::new(std::sync::Mutex::new(DeadlineErrors::default())),
            schedule_drift: ScheduleDriftTrace::default(),
            #[cfg(feature = "metadata")]
            render_overlays: false,
            #[cfg(feature = "metadata")]
            overlay_device: String::from(DEFAULT_CARD),
        }
    }

    pub fn with_pacing(mut self, pacing: PacingPolicy) -> Self {
        self.pacing = pacing;
        self
    }

    pub fn with_max_lateness_ns(mut self, ns: u64) -> Self {
        self.pacer.set_max_lateness_ns(ns);
        self
    }

    pub fn with_qos_interval_ns(mut self, ns: u64) -> Self {
        self.pacer.set_report_interval_ns(ns);
        self
    }

    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.pacer.set_bus(bus);
        self
    }

    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped.load(Ordering::Relaxed)
    }

    pub fn late_dropped(&self) -> u64 {
        self.pacer.late_dropped()
    }

    pub fn with_title<S: Into<String>>(mut self, title: S) -> Self {
        self.title = title.into();
        self
    }

    pub fn with_app_id<S: Into<String>>(mut self, app_id: S) -> Self {
        self.app_id = app_id.into();
        self
    }

    pub fn frames_presented(&self) -> u64 {
        self.frames_presented.load(Ordering::Relaxed)
    }

    pub fn latency_snapshot(&self) -> LatencySnapshot {
        self.latency.histogram.snapshot()
    }

    pub fn latency_samples(&self) -> Vec<u64> {
        self.latency
            .samples
            .lock()
            .expect("latency samples lock")
            .clone()
    }

    pub fn deadline_error_samples(&self) -> Vec<i64> {
        self.deadline_errors
            .lock()
            .expect("deadline errors lock")
            .samples
            .clone()
    }

    pub fn clock_sync(&self) -> Option<&ClockSync> {
        self.pacer.clock_sync()
    }

    fn compositor_formats(&self) -> &CompositorFormats {
        self.compositor_formats.get_or_init(|| {
            probe_compositor_formats().unwrap_or_else(|error| {
                g2g_core::g2g_error!(
                    self,
                    "cannot read the Wayland compositor's formats: {error}"
                );
                CompositorFormats::default()
            })
        })
    }

    fn shutdown(&mut self) {
        if let Some(tx) = self.cmd_tx.take() {
            let _ = tx.send(WorkerCmd::Shutdown);
        }
        if let Some(join) = self.worker.take() {
            let _ = join.join();
        }
    }
}

impl Drop for WaylandSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Debug)]
struct WaylandClock;
impl PipelineClock for WaylandClock {
    fn now_ns(&self) -> u64 {
        monotonic_ns()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Picture {
    format: RawVideoFormat,
    width: u32,
    height: u32,
    display_width: u32,
    display_height: u32,
}

impl Picture {
    fn of(caps: &Caps, force_aspect_ratio: bool) -> Result<Self, G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            pixel_aspect,
            ..
        } = *caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        let (even_width, even_height) = even_dims_required(format);
        if (even_width && width % 2 != 0) || (even_height && height % 2 != 0) {
            return Err(G2gError::CapsMismatch);
        }
        let (display_width, display_height) = if force_aspect_ratio {
            display_size(width, height, pixel_aspect)
        } else {
            (width, height)
        };
        Ok(Self {
            format,
            width,
            height,
            display_width,
            display_height,
        })
    }
}

#[derive(Debug, Default)]
struct CompositorFormats {
    shm: Vec<RawVideoFormat>,
    dmabuf: Vec<RawVideoFormat>,
}

impl CompositorFormats {
    fn accepted_caps(&self) -> CapsSet {
        let mut formats: Vec<RawVideoFormat> = Vec::new();
        for format in self.shm.iter().chain(&self.dmabuf) {
            if !formats.contains(format) {
                formats.push(*format);
            }
        }
        // decoders produce NV12, and the solver prefers the first alternative
        formats.sort_by_key(|format| *format != RawVideoFormat::Nv12);
        CapsSet::from_alternatives(formats.into_iter().map(any_geometry).collect())
    }
}

// drm_fourcc.h
const DRM_FORMAT_MOD_LINEAR: u64 = 0;
const DRM_FORMAT_ARGB8888: u32 = u32::from_le_bytes(*b"AR24");
const DRM_FORMAT_XRGB8888: u32 = u32::from_le_bytes(*b"XR24");

fn drm_format(format: RawVideoFormat) -> Option<u32> {
    let (fourcc, _) = DRM_FOURCC_FORMATS
        .iter()
        .find(|(_, gst_name)| *gst_name == format.gst_name())?;
    Some(u32::from_le_bytes(fourcc.as_bytes().try_into().ok()?))
}

fn raw_video_format(drm_format: u32) -> Option<RawVideoFormat> {
    let fourcc = drm_format.to_le_bytes();
    let (_, gst_name) = DRM_FOURCC_FORMATS
        .iter()
        .find(|(known, _)| known.as_bytes() == fourcc)?;
    parse_raw_format(gst_name)
}

fn shm_format(format: RawVideoFormat) -> Option<wl_shm::Format> {
    // wl_shm codes ARGB8888 and XRGB8888 as 0 and 1 instead of their DRM fourccs
    match drm_format(format)? {
        DRM_FORMAT_ARGB8888 => Some(wl_shm::Format::Argb8888),
        DRM_FORMAT_XRGB8888 => Some(wl_shm::Format::Xrgb8888),
        code => wl_shm::Format::try_from(code).ok(),
    }
}

fn drm_format_of_shm(format: WEnum<wl_shm::Format>) -> u32 {
    match format {
        WEnum::Value(wl_shm::Format::Argb8888) => DRM_FORMAT_ARGB8888,
        WEnum::Value(wl_shm::Format::Xrgb8888) => DRM_FORMAT_XRGB8888,
        WEnum::Value(format) => format as u32,
        WEnum::Unknown(code) => code,
    }
}

#[derive(Default)]
struct FormatProbe {
    shm: Vec<u32>,
    dmabuf: Vec<u32>,
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for FormatProbe {
    fn event(
        _: &mut Self,
        _: &wl_registry::WlRegistry,
        _: wl_registry::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_shm::WlShm, ()> for FormatProbe {
    fn event(
        probe: &mut Self,
        _: &wl_shm::WlShm,
        event: wl_shm::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_shm::Event::Format { format } = event {
            probe.shm.push(drm_format_of_shm(format));
        }
    }
}

impl Dispatch<ZwpLinuxDmabufV1, ()> for FormatProbe {
    fn event(
        probe: &mut Self,
        _: &ZwpLinuxDmabufV1,
        event: zwp_linux_dmabuf_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwp_linux_dmabuf_v1::Event::Modifier {
            format,
            modifier_hi,
            modifier_lo,
        } = event
        {
            if (u64::from(modifier_hi) << 32 | u64::from(modifier_lo)) == DRM_FORMAT_MOD_LINEAR {
                probe.dmabuf.push(format);
            }
        }
    }
}

fn probe_compositor_formats() -> Result<CompositorFormats, Box<dyn std::error::Error>> {
    let conn = Connection::connect_to_env()?;
    let (globals, mut queue) = registry_queue_init::<FormatProbe>(&conn)?;
    let qh = queue.handle();
    globals.bind::<wl_shm::WlShm, _, _>(&qh, 1..=1, ())?;
    // linux-dmabuf sends its format table as modifier events to version 3 only
    let dmabuf = globals.bind::<ZwpLinuxDmabufV1, _, _>(&qh, 3..=3, ()).ok();
    let mut probe = FormatProbe::default();
    queue.roundtrip(&mut probe)?;
    if let Some(dmabuf) = dmabuf {
        dmabuf.destroy();
    }
    let known = |codes: &[u32]| codes.iter().copied().filter_map(raw_video_format).collect();
    Ok(CompositorFormats {
        shm: known(&probe.shm),
        dmabuf: known(&probe.dmabuf),
    })
}

fn importable_caps() -> CapsSet {
    CapsSet::from_alternatives(
        DRM_FOURCC_FORMATS
            .iter()
            .filter_map(|(_, gst_name)| parse_raw_format(gst_name))
            .map(any_geometry)
            .collect(),
    )
}

impl PadTemplates for WaylandSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(importable_caps())])
    }
}

impl AsyncElement for WaylandSink {
    fn latency(&self) -> g2g_core::LatencyReport {
        self.pacer.latency()
    }

    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::Provider,
            alloc::sync::Arc::new(WaylandClock),
        ))
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::DmaBuf)
    }

    fn absorbed_orientations(&self) -> OrientationSet {
        if cfg!(feature = "metadata") {
            OrientationSet::ALL
        } else {
            OrientationSet::EMPTY
        }
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pacer.take_qos()
    }

    fn presentation_stats(&self) -> Option<PresentationStats> {
        Some(PresentationStats {
            presented: self.frames_presented(),
            dropped: self.frames_dropped(),
            late_dropped: self.late_dropped(),
        })
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(self.compositor_formats().accepted_caps())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.pacer.adopt_framerate(absolute_caps);
        let picture = Picture::of(absolute_caps, self.force_aspect_ratio)?;
        if let Caps::RawVideo { colorimetry, .. } = absolute_caps {
            self.colorimetry = *colorimetry;
        }
        let formats = self.compositor_formats();
        if !formats.shm.contains(&picture.format) && !formats.dmabuf.contains(&picture.format) {
            g2g_core::g2g_error!(
                self,
                "the compositor takes no {} buffers",
                picture.format.gst_name()
            );
            return Err(G2gError::CapsMismatch);
        }

        // RtspSrc's first caps carry placeholder dimensions until the SPS arrives
        if self.worker.is_some() {
            match self.picture {
                Some(shown) if shown == picture => return Ok(ConfigureOutcome::Accepted),
                Some(shown) if shown.format == picture.format => {
                    let tx = self.cmd_tx.as_ref().ok_or(G2gError::NotConfigured)?;
                    tx.send(WorkerCmd::Resize(picture))
                        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    self.picture = Some(picture);
                    return Ok(ConfigureOutcome::Accepted);
                }
                _ => self.shutdown(),
            }
        }

        let (tx, rx) = channel::<WorkerCmd>();
        let presented = Arc::clone(&self.frames_presented);
        let dropped = Arc::clone(&self.frames_dropped);
        let latency = Arc::clone(&self.latency);
        let deadline_errors = Arc::clone(&self.deadline_errors);
        self.last_frame_callback_ns
            .store(monotonic_ns(), Ordering::Relaxed);
        let last_frame_callback_ns = Arc::clone(&self.last_frame_callback_ns);
        self.present_delay_ns.store(0, Ordering::Relaxed);
        let present_delay_ns = Arc::clone(&self.present_delay_ns);
        let title = self.title.clone();
        let app_id = self.app_id.clone();
        let fullscreen = self.fullscreen;
        let force_aspect_ratio = self.force_aspect_ratio;
        #[cfg(feature = "metadata")]
        let overlay_device = self
            .render_overlays
            .then(|| std::path::PathBuf::from(&self.overlay_device));

        let ready = Arc::new(Handshake::new());
        let ready_for_worker = Arc::clone(&ready);

        let join = thread::Builder::new()
            .name(String::from("g2g-waylandsink"))
            .spawn(move || {
                if let Err(e) = worker_main(
                    picture,
                    fullscreen,
                    force_aspect_ratio,
                    #[cfg(feature = "metadata")]
                    overlay_device,
                    title,
                    app_id,
                    rx,
                    presented,
                    dropped,
                    latency,
                    deadline_errors,
                    last_frame_callback_ns,
                    present_delay_ns,
                    ready_for_worker,
                ) {
                    std::eprintln!("g2g-waylandsink worker error: {e:?}");
                }
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        if !ready.wait(Duration::from_secs(5)) {
            let _ = tx.send(WorkerCmd::Shutdown);
            let _ = join.join();
            return Err(G2gError::Hardware(HardwareError::Other));
        }

        self.cmd_tx = Some(tx);
        self.worker = Some(join);
        self.picture = Some(picture);
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Wayland video sink",
            "Sink/Video",
            "Presents raw video to a Wayland surface through wl_shm or linux-dmabuf",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("title", PropKind::Str, "window title").with_default("glass2glass"),
            PropertySpec::new("app-id", PropKind::Str, "Wayland xdg app id")
                .with_default("io.glass2glass.WaylandSink"),
            FORCE_ASPECT_RATIO_PROPERTY,
            PropertySpec::new(
                "fullscreen",
                PropKind::Bool,
                "make the window fullscreen, the picture scaled to the output",
            )
            .with_default("false"),
            MAX_LATENESS_PROPERTY,
            PROCESSING_DEADLINE_PROPERTY,
            QOS_INTERVAL_PROPERTY,
            SYNC_PROPERTY,
            #[cfg(feature = "metadata")]
            RENDER_OVERLAYS_PROPERTY,
            #[cfg(feature = "metadata")]
            PropertySpec::new(
                "overlay-device",
                PropKind::Str,
                "DRM device node the render-overlays buffers are allocated on",
            )
            .with_default(DEFAULT_CARD),
        ];
        PROPS
    }

    #[cfg(feature = "metadata")]
    fn meta_requests(&self) -> g2g_core::meta::MetaRequests {
        crate::overlaycomposition::render_requests(self.render_overlays)
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "title" => {
                self.title = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "app-id" => {
                self.app_id = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            FORCE_ASPECT_RATIO_NAME => {
                self.force_aspect_ratio = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "fullscreen" => {
                self.fullscreen = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            #[cfg(feature = "metadata")]
            RENDER_OVERLAYS_NAME => {
                self.render_overlays = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            #[cfg(feature = "metadata")]
            "overlay-device" => {
                self.overlay_device = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            _ => self
                .pacer
                .set_property(name, &value)
                .unwrap_or(Err(PropError::Unknown)),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "title" => Some(PropValue::Str(self.title.clone())),
            "app-id" => Some(PropValue::Str(self.app_id.clone())),
            FORCE_ASPECT_RATIO_NAME => Some(PropValue::Bool(self.force_aspect_ratio)),
            "fullscreen" => Some(PropValue::Bool(self.fullscreen)),
            #[cfg(feature = "metadata")]
            RENDER_OVERLAYS_NAME => Some(PropValue::Bool(self.render_overlays)),
            #[cfg(feature = "metadata")]
            "overlay-device" => Some(PropValue::Str(self.overlay_device.clone())),
            _ => self.pacer.get_property(name),
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::CapsChanged(caps) = &packet {
                self.pacer.adopt_framerate(caps);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let orientation = frame_orientation(&frame);
                    #[cfg(feature = "metadata")]
                    let overlay = frame
                        .meta
                        .get::<g2g_core::OverlayCompositionMeta>()
                        .filter(|_| self.render_overlays)
                        .cloned();
                    let Frame { domain, timing, .. } = frame;
                    if !matches!(domain, MemoryDomain::DmaBuf(_)) {
                        domain.require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    }
                    let t_in = monotonic_ns();
                    let presented = self.frames_presented.load(Ordering::Relaxed);
                    self.pacer
                        .set_present_delay_ns(self.present_delay_ns.load(Ordering::Relaxed));
                    let latest_error = self
                        .deadline_errors
                        .lock()
                        .expect("deadline errors lock")
                        .latest
                        .take();
                    if let Some(error_ns) = latest_error {
                        self.pacer.record_presented_error_ns(error_ns);
                    }
                    let pace = self.pacer.judge(&timing, presented);
                    let deadline_ns = self.pacer.last_deadline_ns();
                    let wait_ns = match pace {
                        g2g_core::Pace::Wait(n) => n,
                        _ => 0,
                    };
                    if !wait_to_present(pace).await {
                        g2g_core::g2g_log!(self, "late-drop pts={}", timing.pts_ns);
                        return Ok(());
                    }
                    let picture = self.picture.ok_or(G2gError::NotConfigured)?;
                    let formats = self.compositor_formats();
                    let pixels = match domain {
                        MemoryDomain::DmaBuf(dmabuf) => {
                            if !formats.dmabuf.contains(&picture.format) {
                                g2g_core::g2g_error!(
                                    self,
                                    "the compositor takes no linear {} dma-bufs",
                                    picture.format.gst_name()
                                );
                                return Err(G2gError::CapsMismatch);
                            }
                            QueuedPixels::Imported(dmabuf)
                        }
                        MemoryDomain::System(pixels) => {
                            if !formats.shm.contains(&picture.format) {
                                g2g_core::g2g_error!(
                                    self,
                                    "the compositor takes {} only as dma-bufs, put videoconvert ahead of waylandsink",
                                    picture.format.gst_name()
                                );
                                return Err(G2gError::CapsMismatch);
                            }
                            let frame_bytes =
                                frame_byte_size(picture.format, picture.width, picture.height);
                            if pixels.as_slice().len() < frame_bytes {
                                return Err(G2gError::CapsMismatch);
                            }
                            QueuedPixels::System(pixels)
                        }
                        _ => return Err(G2gError::UnsupportedDomain),
                    };

                    let tx = self.cmd_tx.as_ref().ok_or(G2gError::NotConfigured)?;
                    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
                    let deadline =
                        deadline_ns
                            .zip(self.pacer.clock_sync())
                            .map(|(deadline_ns, sync)| PresentationDeadline {
                                clock: Arc::clone(&sync.clock),
                                deadline_ns,
                            });
                    tx.send(WorkerCmd::Frame(QueuedFrame {
                        pixels,
                        colorimetry: self.colorimetry,
                        presented: PresentedFrame {
                            arrival_ns: timing.arrival_ns,
                            deadline,
                        },
                        orientation,
                        ack: ack_tx,
                        #[cfg(feature = "metadata")]
                        overlay,
                    }))
                    .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    match self.pacing {
                        PacingPolicy::Block => {
                            let silence =
                                core::time::Duration::from_nanos(FRAME_CALLBACK_SILENCE_NS);
                            let last_callback_ns =
                                self.last_frame_callback_ns.load(Ordering::Relaxed);
                            if monotonic_ns().saturating_sub(last_callback_ns)
                                < FRAME_CALLBACK_SILENCE_NS
                            {
                                match tokio::time::timeout(silence, ack_rx).await {
                                    Ok(acked) => acked
                                        .map_err(|_| G2gError::Hardware(HardwareError::Other))?,
                                    Err(_) => g2g_core::g2g_log!(
                                        self,
                                        "frame callback overdue, presenting unthrottled"
                                    ),
                                }
                            }
                        }
                        PacingPolicy::DropOldest => {
                            drop(ack_rx);
                        }
                    }
                    // the elected clock is the audio sink's in an A/V graph
                    let clock_now_ns = self.pacer.clock_sync().map(|s| s.now_ns());
                    if let (Some(deadline), Some(clock_now)) = (deadline_ns, clock_now_ns) {
                        if let Some(drift) = self.schedule_drift.sample(
                            monotonic_ns(),
                            clock_now,
                            timing.pts_ns,
                            deadline,
                        ) {
                            g2g_core::g2g_log!(
                                self,
                                "pace clock-mono={}us media-mono={}us deadline-mono={}us \
                                 wait={}us age_in={}us late_dropped={}",
                                drift.clock_ns / 1_000,
                                drift.media_ns / 1_000,
                                drift.deadline_ns / 1_000,
                                wait_ns / 1_000,
                                t_in.saturating_sub(timing.arrival_ns) / 1_000,
                                self.pacer.late_dropped()
                            );
                        }
                    }
                    Ok(())
                }
                PipelinePacket::Segment(seg) => {
                    self.pacer.set_segment(seg);
                    Ok(())
                }
                PipelinePacket::Flush => {
                    self.pacer.flush();
                    Ok(())
                }
                PipelinePacket::CapsChanged(_) => Ok(()),
                PipelinePacket::Eos => {
                    self.shutdown();
                    Ok(())
                }
                _ => Ok(()),
            }
        })
    }
}

struct TimedCommits {
    fifo: WpFifoV1,
    timer: WpCommitTimerV1,
}

struct ColorRepresentation {
    surface: WpColorRepresentationSurfaceV1,
    shown: Option<(Coefficients, Range)>,
}

fn coefficients_and_range(colorimetry: Colorimetry) -> (Coefficients, Range) {
    let conversion = colorimetry.yuv_conversion();
    let coefficients = match conversion.matrix {
        MatrixCoefficients::Bt709 => Coefficients::Bt709,
        MatrixCoefficients::Bt2020Ncl => Coefficients::Bt2020,
        _ => Coefficients::Bt601,
    };
    let range = if conversion.full_range {
        Range::Full
    } else {
        Range::Limited
    };
    (coefficients, range)
}

struct WorkerState {
    registry_state: RegistryState,
    output_state: OutputState,
    shm: Shm,
    pool: SlotPool,
    buffer: Option<Buffer>,
    dmabuf_state: DmabufState,
    imports: DmaBufImportCache<wl_buffer::WlBuffer>,
    // held until the compositor sends wl_buffer.release for their buffer
    on_screen: Vec<(wl_buffer::WlBuffer, OwnedDmaBuf)>,
    window: Window,
    window_viewport: WpViewport,
    black_buffer: Buffer,
    picture_surface: wl_surface::WlSurface,
    picture_subsurface: WlSubsurface,
    picture_viewport: WpViewport,
    qh: QueueHandle<WorkerState>,
    format: RawVideoFormat,
    width: u32,
    height: u32,
    display_width: u32,
    display_height: u32,
    force_aspect_ratio: bool,
    configured_size: ConfiguredSize,
    configured: bool,
    closing: bool,
    exit: bool,
    ready: Option<Arc<Handshake>>,
    presented: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    latency: Arc<LatencyRecorder>,
    deadline_errors: Arc<std::sync::Mutex<DeadlineErrors>>,
    presentation: WpPresentation,
    presentation_clock: Option<u32>,
    timed_commits: Option<TimedCommits>,
    color_representation: Option<ColorRepresentation>,
    supported_coefficients: Vec<(Coefficients, Range)>,
    present_delay_ns: Arc<AtomicU64>,
    commit_to_screen_ns: VecDeque<u64>,
    unresolved_feedback: usize,
    last_feedback_ns: u64,
    last_frame_callback_ns: Arc<AtomicU64>,
    pending: Option<QueuedFrame>,
    // wayland keeps the buffer transform until the client sets another
    orientation: Orientation,
    picture_has_buffer: bool,
    pending_ack: Option<tokio::sync::oneshot::Sender<()>>,
    #[cfg(feature = "metadata")]
    overlay: Option<OverlaySurface>,
}

const XRGB8888_BYTES_PER_PIXEL: usize = 4;

#[allow(clippy::too_many_arguments)]
fn worker_main(
    picture: Picture,
    fullscreen: bool,
    force_aspect_ratio: bool,
    #[cfg(feature = "metadata")] overlay_device: Option<std::path::PathBuf>,
    title: String,
    app_id: String,
    rx: Channel<WorkerCmd>,
    presented: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    latency: Arc<LatencyRecorder>,
    deadline_errors: Arc<std::sync::Mutex<DeadlineErrors>>,
    last_frame_callback_ns: Arc<AtomicU64>,
    present_delay_ns: Arc<AtomicU64>,
    ready: Arc<Handshake>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (width, height) = (picture.width, picture.height);
    let conn = Connection::connect_to_env()?;
    let (globals, event_queue) = registry_queue_init(&conn)?;
    let qh = event_queue.handle();

    let mut event_loop: EventLoop<WorkerState> = EventLoop::try_new()?;
    let loop_handle = event_loop.handle();
    WaylandSource::new(conn.clone(), event_queue).insert(loop_handle.clone())?;

    let compositor = CompositorState::bind(&globals, &qh)?;
    let xdg_shell = XdgShell::bind(&globals, &qh)?;
    let subcompositor =
        SubcompositorState::bind(compositor.wl_compositor().clone(), &globals, &qh)?;
    let viewporter: WpViewporter = globals.bind(&qh, 1..=1, ())?;
    let presentation: WpPresentation = globals.bind(&qh, 1..=1, ())?;
    let shm = Shm::bind(&globals, &qh)?;
    let dmabuf_state = DmabufState::new(&globals, &qh);

    let surface = compositor.create_surface(&qh);
    let window = xdg_shell.create_window(surface, WindowDecorations::RequestServer, &qh);
    window.set_title(&title);
    window.set_app_id(&app_id);
    if fullscreen {
        window.set_fullscreen(None);
    }
    let window_viewport = viewporter.get_viewport(window.wl_surface(), &qh, ());
    let (picture_subsurface, picture_surface) =
        subcompositor.create_subsurface(window.wl_surface().clone(), &qh);
    let picture_viewport = viewporter.get_viewport(&picture_surface, &qh, ());
    picture_subsurface.set_desync();
    let fifo_manager = globals.bind::<WpFifoManagerV1, _, _>(&qh, 1..=1, ()).ok();
    let commit_timing_manager = globals
        .bind::<WpCommitTimingManagerV1, _, _>(&qh, 1..=1, ())
        .ok();
    let timed_commits = match (fifo_manager, commit_timing_manager) {
        (Some(fifo_manager), Some(commit_timing_manager)) => Some(TimedCommits {
            fifo: fifo_manager.get_fifo(&picture_surface, &qh, ()),
            timer: commit_timing_manager.get_timer(&picture_surface, &qh, ()),
        }),
        _ => None,
    };
    let color_representation =
        match globals.bind::<WpColorRepresentationManagerV1, _, _>(&qh, 1..=1, ()) {
            Ok(manager) if carries_yuv(picture.format) => Some(ColorRepresentation {
                surface: manager.get_surface(&picture_surface, &qh, ()),
                shown: None,
            }),
            _ => None,
        };
    g2g_core::g2g_log!(
        Target::category(short_type_name::<WaylandSink>()),
        "{}",
        if timed_commits.is_some() {
            "commits frames at their deadlines with wp_fifo_v1 and wp_commit_timing_v1"
        } else {
            "paces commits by frame callbacks"
        }
    );
    #[cfg(feature = "metadata")]
    let overlay = match overlay_device {
        Some(device) => Some(
            OverlaySurface::new(
                &subcompositor,
                &viewporter,
                window.wl_surface(),
                &device,
                (width, height),
                &dmabuf_state,
                &qh,
            )
            .map_err(|error| std::format!("overlay buffers on {}: {error:?}", device.display()))?,
        ),
        None => None,
    };
    window.commit();

    let mut pool = SlotPool::new(frame_byte_size(picture.format, width, height), &shm)?;
    let (black_buffer, black_pixel) = pool.create_buffer(
        1,
        1,
        XRGB8888_BYTES_PER_PIXEL as i32,
        wl_shm::Format::Xrgb8888,
    )?;
    black_pixel.fill(0);

    let mut state = WorkerState {
        registry_state: RegistryState::new(&globals),
        output_state: OutputState::new(&globals, &qh),
        shm,
        pool,
        buffer: None,
        dmabuf_state,
        imports: DmaBufImportCache::new(),
        on_screen: Vec::new(),
        window,
        window_viewport,
        black_buffer,
        picture_surface,
        picture_subsurface,
        picture_viewport,
        qh: qh.clone(),
        format: picture.format,
        width,
        height,
        display_width: picture.display_width,
        display_height: picture.display_height,
        force_aspect_ratio,
        configured_size: (None, None),
        configured: false,
        closing: false,
        exit: false,
        ready: Some(ready),
        presented,
        dropped,
        latency,
        deadline_errors,
        presentation,
        presentation_clock: None,
        timed_commits,
        color_representation,
        supported_coefficients: Vec::new(),
        present_delay_ns,
        commit_to_screen_ns: VecDeque::new(),
        unresolved_feedback: 0,
        last_feedback_ns: 0,
        last_frame_callback_ns,
        pending: None,
        orientation: Orientation::Identity,
        picture_has_buffer: false,
        pending_ack: None,
        #[cfg(feature = "metadata")]
        overlay,
    };

    loop_handle.insert_source(rx, |event, _, state: &mut WorkerState| match event {
        ChanEvent::Msg(WorkerCmd::Frame(queued)) => {
            if state.configured {
                state.draw(queued);
            } else {
                state.pending = Some(queued);
            }
        }
        ChanEvent::Msg(WorkerCmd::Resize(picture)) => {
            if let Err(error) = state.resize(picture) {
                g2g_core::g2g_error!(
                    Target::category(short_type_name::<WaylandSink>()),
                    "cannot resize the picture: {error:?}"
                );
                state.exit = true;
            }
        }
        ChanEvent::Msg(WorkerCmd::Shutdown) | ChanEvent::Closed => {
            state.closing = true;
        }
    })?;

    while !state.exit {
        event_loop.dispatch(Some(Duration::from_millis(100)), &mut state)?;
        // compositors send no feedback for a commit on a surface they are not painting
        let feedback_silent =
            monotonic_ns().saturating_sub(state.last_feedback_ns) >= FRAME_CALLBACK_SILENCE_NS;
        if state.closing && (state.unresolved_feedback == 0 || feedback_silent) {
            state
                .dropped
                .fetch_add(state.unresolved_feedback as u64, Ordering::Relaxed);
            state.exit = true;
        }
    }
    Ok(())
}

type ConfiguredSize = (Option<NonZeroU32>, Option<NonZeroU32>);

#[derive(Debug, PartialEq, Eq)]
struct WindowLayout {
    window_width: u32,
    window_height: u32,
    picture: ViewportRectangle,
}

fn window_layout(
    (display_width, display_height): (u32, u32),
    orientation: Orientation,
    (configured_width, configured_height): ConfiguredSize,
    force_aspect_ratio: bool,
) -> WindowLayout {
    let (width, height) = if orientation.swaps_dims() {
        (display_height, display_width)
    } else {
        (display_width, display_height)
    };
    let window_width = configured_width.map_or(width, NonZeroU32::get);
    let window_height = configured_height.map_or(height, NonZeroU32::get);
    let picture = if force_aspect_ratio {
        letterbox(
            width,
            height,
            PixelAspect::SQUARE,
            window_width,
            window_height,
        )
    } else {
        ViewportRectangle::whole(window_width, window_height)
    };
    WindowLayout {
        window_width,
        window_height,
        picture,
    }
}

impl WorkerState {
    fn commit_layout(&self) {
        self.picture_subsurface.set_sync();
        self.set_layout(self.orientation);
        self.picture_surface.commit();
        #[cfg(feature = "metadata")]
        if let Some(overlay) = &self.overlay {
            overlay.surface.commit();
        }
        self.window.commit();
        self.picture_subsurface.set_desync();
    }

    fn resize(&mut self, picture: Picture) -> Result<(), G2gError> {
        self.width = picture.width;
        self.height = picture.height;
        self.display_width = picture.display_width;
        self.display_height = picture.display_height;
        self.buffer = None;
        #[cfg(feature = "metadata")]
        if let Some(overlay) = &mut self.overlay {
            overlay.resize(
                (picture.width, picture.height),
                &self.dmabuf_state,
                &self.qh,
            )?;
        }
        if self.configured {
            self.commit_layout();
        }
        Ok(())
    }

    fn set_layout(&self, orientation: Orientation) {
        let layout = window_layout(
            (self.display_width, self.display_height),
            orientation,
            self.configured_size,
            self.force_aspect_ratio,
        );
        self.window_viewport
            .set_destination(layout.window_width as i32, layout.window_height as i32);
        self.picture_subsurface
            .set_position(layout.picture.x as i32, layout.picture.y as i32);
        self.picture_viewport
            .set_destination(layout.picture.width as i32, layout.picture.height as i32);
        #[cfg(feature = "metadata")]
        if let Some(overlay) = &self.overlay {
            overlay
                .subsurface
                .set_position(layout.picture.x as i32, layout.picture.y as i32);
            overlay
                .viewport
                .set_destination(layout.picture.width as i32, layout.picture.height as i32);
        }
    }

    #[cfg(feature = "metadata")]
    fn update_overlay(
        &mut self,
        composition: Option<g2g_core::OverlayCompositionMeta>,
        turned: bool,
    ) -> Result<bool, G2gError> {
        let size = (self.width, self.height);
        let Some(overlay) = &mut self.overlay else {
            return Ok(false);
        };
        let appeared = overlay.show(composition, size)?;
        // wayland applies a new transform and viewport with the surface's next commit
        if turned {
            overlay.surface.commit();
        }
        Ok(appeared)
    }

    fn fill_shm_buffer(&mut self, pixels: &SystemSlice) -> Result<(), G2gError> {
        let frame_bytes = frame_byte_size(self.format, self.width, self.height);
        let hardware = || G2gError::Hardware(HardwareError::Other);
        // sctk's Buffer::canvas covers plane 0 only, the slot covers every plane
        let buffer = match self.buffer.take() {
            Some(buffer) if buffer.slot().canvas(&mut self.pool).is_some() => buffer,
            _ => {
                let format = shm_format(self.format).ok_or(G2gError::CapsMismatch)?;
                let stride = i32::try_from(row_bytes(self.format, self.width as usize))
                    .map_err(|_| G2gError::CapsMismatch)?;
                let slot = self.pool.new_slot(frame_bytes).map_err(|_| hardware())?;
                self.pool
                    .create_buffer_in(&slot, self.width as i32, self.height as i32, stride, format)
                    .map_err(|_| hardware())?
            }
        };
        let canvas = buffer.slot().canvas(&mut self.pool).ok_or_else(hardware)?;
        canvas[..frame_bytes].copy_from_slice(&pixels.as_slice()[..frame_bytes]);
        self.buffer = Some(buffer);
        Ok(())
    }

    fn import(&mut self, dmabuf: &OwnedDmaBuf) -> Result<wl_buffer::WlBuffer, G2gError> {
        let drm_format = drm_format(self.format).ok_or(G2gError::CapsMismatch)?;
        let (layout, size) = dmabuf_plane_layout(dmabuf, self.format, self.width, self.height)
            .ok_or(G2gError::CapsMismatch)?;
        let (dmabuf_state, qh) = (&self.dmabuf_state, &self.qh);
        let (width, height) = (self.width as i32, self.height as i32);
        self.imports.get_or_import(
            dmabuf,
            size,
            || {
                let params = dmabuf_state
                    .create_params(qh)
                    .map_err(|_| G2gError::UnsupportedDomain)?;
                // SAFETY: the fd is a live dma-buf owned by `dmabuf`, and the request is written before this returns.
                let fd = unsafe { std::os::fd::BorrowedFd::borrow_raw(dmabuf.as_raw()) };
                for index in 0..layout.count() {
                    let plane = layout.plane(index).ok_or(G2gError::CapsMismatch)?;
                    let to_u32 =
                        |value: usize| u32::try_from(value).map_err(|_| G2gError::CapsMismatch);
                    params.add(
                        fd,
                        index as u32,
                        to_u32(plane.offset)?,
                        to_u32(plane.stride)?,
                        DRM_FORMAT_MOD_LINEAR,
                    );
                }
                let (wl_buffer, _) = params.create_immed(
                    width,
                    height,
                    drm_format,
                    zwp_linux_buffer_params_v1::Flags::empty(),
                    qh,
                );
                Ok(wl_buffer)
            },
            |wl_buffer| {
                wl_buffer.destroy();
                Ok(())
            },
        )
    }

    fn draw(&mut self, queued: QueuedFrame) {
        let QueuedFrame {
            pixels,
            colorimetry,
            presented,
            orientation,
            ack,
            #[cfg(feature = "metadata")]
            overlay,
        } = queued;
        let width = self.width as i32;
        let height = self.height as i32;
        let attach = match pixels {
            QueuedPixels::System(pixels) => {
                if self.fill_shm_buffer(&pixels).is_err() {
                    let _ = ack.send(());
                    return;
                }
                None
            }
            QueuedPixels::Imported(dmabuf) => match self.import(&dmabuf) {
                Ok(wl_buffer) => Some((wl_buffer, dmabuf)),
                Err(error) => {
                    g2g_core::g2g_error!(
                        Target::category(short_type_name::<WaylandSink>()),
                        "the compositor cannot show this dma-buf as linear {}: {error:?}",
                        self.format.gst_name()
                    );
                    self.exit = true;
                    return;
                }
            },
        };

        let turned = orientation != self.orientation;
        if turned {
            self.picture_surface
                .set_buffer_transform(buffer_transform(orientation));
            #[cfg(feature = "metadata")]
            if let Some(overlay) = &self.overlay {
                overlay
                    .surface
                    .set_buffer_transform(buffer_transform(orientation));
            }
            // a transposing turn swaps the surface size, not the buffer's
            self.set_layout(orientation);
        }
        #[cfg(feature = "metadata")]
        let overlay_appeared = match self.update_overlay(overlay, turned) {
            Ok(appeared) => appeared,
            Err(error) => {
                g2g_core::g2g_error!(
                    Target::category(short_type_name::<WaylandSink>()),
                    "cannot show the overlay composition: {error:?}"
                );
                self.exit = true;
                return;
            }
        };
        #[cfg(not(feature = "metadata"))]
        let overlay_appeared = false;
        let target = match (&self.timed_commits, &presented.deadline) {
            (Some(_), Some(deadline)) => match self.presentation_time_of(deadline) {
                Some(target_ns) => Some(Duration::from_nanos(target_ns)),
                None => {
                    self.exit = true;
                    return;
                }
            },
            _ => None,
        };
        self.set_color_representation(colorimetry);
        let surface = &self.picture_surface;
        // weston repaints a subsurface for its own commits only once a parent commit has placed it on an output
        let commit_parent = turned || !self.picture_has_buffer || overlay_appeared;
        if commit_parent {
            // wayland applies a sync subsurface's state with its parent's next commit
            self.picture_subsurface.set_sync();
        }
        match &self.timed_commits {
            Some(timed) => {
                timed.fifo.wait_barrier();
                timed.fifo.set_barrier();
                if let Some(target) = target {
                    let seconds = target.as_secs();
                    timed.timer.set_timestamp(
                        (seconds >> 32) as u32,
                        seconds as u32,
                        target.subsec_nanos(),
                    );
                }
            }
            None => {
                // smithay-client-toolkit passes the surface to `CompositorHandler::frame` in `FrameCallbackData`
                surface.frame(&self.qh, FrameCallbackData(surface.clone()));
            }
        }
        let committed = CommittedFrame {
            frame: presented,
            committed_ns: monotonic_ns(),
        };
        self.presentation.feedback(surface, &self.qh, committed);
        self.unresolved_feedback += 1;
        self.last_feedback_ns = monotonic_ns();
        surface.damage_buffer(0, 0, width, height);
        match attach {
            Some((wl_buffer, dmabuf)) => {
                surface.attach(Some(&wl_buffer), 0, 0);
                self.on_screen.push((wl_buffer, dmabuf));
            }
            None => self
                .buffer
                .as_ref()
                .expect("fill_shm_buffer leaves a buffer")
                .attach_to(surface)
                .expect("attach_to"),
        }
        surface.commit();
        if commit_parent {
            self.window.commit();
            self.picture_subsurface.set_desync();
        }
        self.orientation = orientation;
        self.picture_has_buffer = true;

        if self.timed_commits.is_some() {
            let _ = ack.send(());
            return;
        }
        if let Some(stale) = self.pending_ack.take() {
            let _ = stale.send(());
        }
        self.pending_ack = Some(ack);
    }
}

impl WorkerState {
    fn set_color_representation(&mut self, colorimetry: Colorimetry) {
        let Some(representation) = &mut self.color_representation else {
            return;
        };
        let wanted = coefficients_and_range(colorimetry);
        if representation.shown == Some(wanted) {
            return;
        }
        representation.shown = Some(wanted);
        if self.supported_coefficients.contains(&wanted) {
            representation
                .surface
                .set_coefficients_and_range(wanted.0, wanted.1);
        } else {
            g2g_core::g2g_log!(
                Target::category(short_type_name::<WaylandSink>()),
                "the compositor converts with its default matrix, it takes no {wanted:?}"
            );
        }
    }
}

impl CompositorHandler for WorkerState {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: i32,
    ) {
    }
    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }
    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {
        self.last_frame_callback_ns
            .store(monotonic_ns(), Ordering::Relaxed);
        if let Some(ack) = self.pending_ack.take() {
            let _ = ack.send(());
        }
    }
    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
    fn surface_leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
}

impl WindowHandler for WorkerState {
    fn request_close(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &Window) {
        self.exit = true;
    }

    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &Window,
        configure: WindowConfigure,
        _serial: u32,
    ) {
        let was_first = !self.configured;
        if was_first {
            let window_surface = self.window.wl_surface();
            self.black_buffer
                .attach_to(window_surface)
                .expect("the black buffer is attached once");
            window_surface.damage_buffer(0, 0, 1, 1);
        }
        self.configured_size = configure.new_size;
        self.commit_layout();
        self.configured = true;
        if was_first {
            if let Some(ready) = self.ready.take() {
                ready.notify();
            }
            if let Some(queued) = self.pending.take() {
                self.draw(queued);
            }
        }
    }
}

impl OutputHandler for WorkerState {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl ShmHandler for WorkerState {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl ProvidesRegistryState for WorkerState {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState,];
}

impl DmabufHandler for WorkerState {
    fn dmabuf_state(&mut self) -> &mut DmabufState {
        &mut self.dmabuf_state
    }

    fn dmabuf_feedback(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1,
        _: DmabufFeedback,
    ) {
    }

    fn created(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1,
        _: wl_buffer::WlBuffer,
    ) {
        // buffers come from create_immed, which sends no created event
    }

    fn failed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1,
    ) {
        g2g_core::g2g_error!(
            Target::category(short_type_name::<WaylandSink>()),
            "the compositor refused a linear dma-buf"
        );
        self.exit = true;
    }

    fn released(&mut self, _: &Connection, _: &QueueHandle<Self>, buffer: &wl_buffer::WlBuffer) {
        if let Some(index) = self.on_screen.iter().position(|(held, _)| held == buffer) {
            self.on_screen.swap_remove(index);
        }
        #[cfg(feature = "metadata")]
        if let Some(overlay) = &mut self.overlay {
            for overlay_buffer in overlay
                .buffers
                .iter_mut()
                .filter(|b| &b.wl_buffer == buffer)
            {
                overlay_buffer.on_screen = false;
            }
        }
    }
}

impl Dispatch<WpPresentation, ()> for WorkerState {
    fn event(
        state: &mut Self,
        _: &WpPresentation,
        event: wp_presentation::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wp_presentation::Event::ClockId { clk_id } = event {
            state.presentation_clock = Some(clk_id);
        }
    }
}

fn nanoseconds(seconds: u64, nanoseconds: u32) -> u64 {
    let since =
        Duration::from_secs(seconds).saturating_add(Duration::from_nanos(u64::from(nanoseconds)));
    u64::try_from(since.as_nanos()).unwrap_or(u64::MAX)
}

fn clock_now_ns(clock: u32) -> Option<u64> {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // presentation-time reports a clock_gettime clock id
    // SAFETY: `now` is a live timespec the call writes into.
    let result = unsafe { libc::clock_gettime(clock as libc::clockid_t, &mut now) };
    (result == 0).then(|| nanoseconds(now.tv_sec as u64, now.tv_nsec as u32))
}

impl WorkerState {
    fn presentation_time_of(&self, deadline: &PresentationDeadline) -> Option<u64> {
        let Some(clock) = self.presentation_clock else {
            g2g_core::g2g_error!(
                Target::category(short_type_name::<WaylandSink>()),
                "the compositor sent no presentation clock id"
            );
            return None;
        };
        let Some(now_ns) = clock_now_ns(clock) else {
            g2g_core::g2g_error!(
                Target::category(short_type_name::<WaylandSink>()),
                "cannot read the compositor's presentation clock {clock}"
            );
            return None;
        };
        let until_deadline_ns = deadline.deadline_ns.saturating_sub(deadline.clock.now_ns());
        Some(now_ns.saturating_add(until_deadline_ns))
    }

    fn resolve_feedback(&mut self) {
        self.unresolved_feedback -= 1;
        self.last_feedback_ns = monotonic_ns();
    }
}

// about a second of frames at 30 fps
const PRESENT_DELAY_WINDOW: usize = 30;

impl Dispatch<WpPresentationFeedback, CommittedFrame> for WorkerState {
    fn event(
        state: &mut Self,
        _: &WpPresentationFeedback,
        event: wp_presentation_feedback::Event,
        committed: &CommittedFrame,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let (shown_ns, refresh) = match event {
            wp_presentation_feedback::Event::Presented {
                tv_sec_hi,
                tv_sec_lo,
                tv_nsec,
                refresh,
                ..
            } => (
                nanoseconds(u64::from(tv_sec_hi) << 32 | u64::from(tv_sec_lo), tv_nsec),
                refresh,
            ),
            wp_presentation_feedback::Event::Discarded => {
                state.dropped.fetch_add(1, Ordering::Relaxed);
                state.resolve_feedback();
                return;
            }
            _ => return,
        };
        state.resolve_feedback();
        state.presented.fetch_add(1, Ordering::Relaxed);
        let Some(clock) = state.presentation_clock else {
            g2g_core::g2g_error!(
                Target::category(short_type_name::<WaylandSink>()),
                "the compositor sent a presentation time before its clock id"
            );
            state.exit = true;
            return;
        };
        // mutter reports presentation times later than the feedback arrives
        let Some(shown_from_now_ns) = clock_now_ns(clock).map(|now| shown_ns as i64 - now as i64)
        else {
            g2g_core::g2g_error!(
                Target::category(short_type_name::<WaylandSink>()),
                "cannot read the compositor's presentation clock {clock}"
            );
            state.exit = true;
            return;
        };
        let frame = &committed.frame;
        let shown_monotonic_ns = monotonic_ns().saturating_add_signed(shown_from_now_ns);
        let present_delay_ns = if state.timed_commits.is_some() {
            // compositors hold a timed commit until its target time
            u64::from(refresh)
        } else {
            state
                .commit_to_screen_ns
                .push_back(shown_monotonic_ns.saturating_sub(committed.committed_ns));
            if state.commit_to_screen_ns.len() > PRESENT_DELAY_WINDOW {
                state.commit_to_screen_ns.pop_front();
            }
            median(state.commit_to_screen_ns.iter().copied().collect())
        };
        state
            .present_delay_ns
            .store(present_delay_ns, Ordering::Relaxed);
        if frame.arrival_ns != 0 {
            state
                .latency
                .record(shown_monotonic_ns.saturating_sub(frame.arrival_ns));
        }
        if let Some(deadline) = &frame.deadline {
            let shown_on_clock_ns = deadline
                .clock
                .now_ns()
                .saturating_add_signed(shown_from_now_ns);
            let error_ns = shown_on_clock_ns as i64 - deadline.deadline_ns as i64;
            let mut errors = state.deadline_errors.lock().expect("deadline errors lock");
            if errors.samples.len() < PER_FRAME_SAMPLE_CAPACITY {
                errors.samples.push(error_ns);
            }
            errors.latest = Some(error_ns);
        }
    }
}

impl Dispatch<WpColorRepresentationManagerV1, ()> for WorkerState {
    fn event(
        state: &mut Self,
        _: &WpColorRepresentationManagerV1,
        event: wp_color_representation_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wp_color_representation_manager_v1::Event::SupportedCoefficientsAndRanges {
            coefficients: WEnum::Value(coefficients),
            range: WEnum::Value(range),
        } = event
        {
            state.supported_coefficients.push((coefficients, range));
        }
    }
}

delegate_dispatch2!(WorkerState);
delegate_registry!(WorkerState);
delegate_noop!(WorkerState: ignore WpViewporter);
delegate_noop!(WorkerState: ignore WpViewport);
delegate_noop!(WorkerState: ignore WpFifoManagerV1);
delegate_noop!(WorkerState: ignore WpFifoV1);
delegate_noop!(WorkerState: ignore WpCommitTimingManagerV1);
delegate_noop!(WorkerState: ignore WpCommitTimerV1);
delegate_noop!(WorkerState: ignore WpColorRepresentationSurfaceV1);

#[cfg(feature = "metadata")]
struct OverlayBuffer {
    dumb: drm::control::dumbbuffer::DumbBuffer,
    wl_buffer: wl_buffer::WlBuffer,
    on_screen: bool,
}

#[cfg(feature = "metadata")]
struct OverlaySurface {
    surface: wl_surface::WlSurface,
    subsurface: WlSubsurface,
    viewport: WpViewport,
    card: Card,
    buffers: Vec<OverlayBuffer>,
    shown: Option<g2g_core::OverlayCompositionMeta>,
    layer: Vec<u8>,
    row: Vec<u8>,
}

#[cfg(feature = "metadata")]
impl OverlaySurface {
    fn new(
        subcompositor: &SubcompositorState,
        viewporter: &WpViewporter,
        parent: &wl_surface::WlSurface,
        device: &std::path::Path,
        size: (u32, u32),
        dmabuf_state: &DmabufState,
        qh: &QueueHandle<WorkerState>,
    ) -> Result<Self, G2gError> {
        let card = Card::open(device)?;
        let buffers = overlay_buffers(&card, size, dmabuf_state, qh)?;
        let (subsurface, surface) = subcompositor.create_subsurface(parent.clone(), qh);
        let viewport = viewporter.get_viewport(&surface, qh, ());
        subsurface.set_desync();
        Ok(Self {
            surface,
            subsurface,
            viewport,
            card,
            buffers,
            shown: None,
            layer: Vec::new(),
            row: Vec::new(),
        })
    }

    fn resize(
        &mut self,
        size: (u32, u32),
        dmabuf_state: &DmabufState,
        qh: &QueueHandle<WorkerState>,
    ) -> Result<(), G2gError> {
        use drm::control::Device as _;
        for buffer in self.buffers.drain(..) {
            buffer.wl_buffer.destroy();
            self.card
                .destroy_dumb_buffer(buffer.dumb)
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        }
        self.buffers = overlay_buffers(&self.card, size, dmabuf_state, qh)?;
        self.surface.attach(None, 0, 0);
        self.shown = None;
        Ok(())
    }

    fn show(
        &mut self,
        composition: Option<g2g_core::OverlayCompositionMeta>,
        (width, height): (u32, u32),
    ) -> Result<bool, G2gError> {
        let unchanged = match (&composition, &self.shown) {
            (None, None) => true,
            (Some(next), Some(shown)) => same_composition(next, shown),
            _ => false,
        };
        if unchanged {
            return Ok(false);
        }
        let Some(composition) = composition else {
            self.surface.attach(None, 0, 0);
            self.surface.commit();
            self.shown = None;
            return Ok(false);
        };
        let Some(buffer) = self.buffers.iter_mut().find(|buffer| !buffer.on_screen) else {
            return Ok(false);
        };
        draw_composition(
            &self.card,
            &mut buffer.dumb,
            &composition,
            (width, height),
            &mut self.layer,
            &mut self.row,
        )?;
        self.surface.attach(Some(&buffer.wl_buffer), 0, 0);
        self.surface
            .damage_buffer(0, 0, width as i32, height as i32);
        self.surface.commit();
        buffer.on_screen = true;
        let appeared = self.shown.is_none();
        self.shown = Some(composition);
        Ok(appeared)
    }
}

#[cfg(feature = "metadata")]
fn overlay_buffers(
    card: &Card,
    (width, height): (u32, u32),
    dmabuf_state: &DmabufState,
    qh: &QueueHandle<WorkerState>,
) -> Result<Vec<OverlayBuffer>, G2gError> {
    use drm::buffer::Buffer as _;
    use drm::control::Device as _;
    let hardware = |_| G2gError::Hardware(HardwareError::Other);
    let drm_format = drm_format(RawVideoFormat::Bgra8).ok_or(G2gError::CapsMismatch)?;
    // one buffer is written while the compositor still reads the other
    (0..2)
        .map(|_| {
            let dumb = card
                .create_dumb_buffer((width, height), drm::buffer::DrmFourcc::Argb8888, 32)
                .map_err(hardware)?;
            let fd = card
                .buffer_to_prime_fd(dumb.handle(), drm::CLOEXEC | drm::RDWR)
                .map_err(hardware)?;
            let params = dmabuf_state
                .create_params(qh)
                .map_err(|_| G2gError::UnsupportedDomain)?;
            params.add(
                std::os::fd::AsFd::as_fd(&fd),
                0,
                0,
                dumb.pitch(),
                DRM_FORMAT_MOD_LINEAR,
            );
            let (wl_buffer, _) = params.create_immed(
                width as i32,
                height as i32,
                drm_format,
                zwp_linux_buffer_params_v1::Flags::empty(),
                qh,
            );
            Ok(OverlayBuffer {
                dumb,
                wl_buffer,
                on_screen: false,
            })
        })
        .collect()
}

fn buffer_transform(orientation: Orientation) -> wl_output::Transform {
    // set_buffer_transform takes the turn the buffer already has
    match orientation.inverse() {
        Orientation::Identity => wl_output::Transform::Normal,
        // wl_output rotations are counter-clockwise
        Orientation::Rotate90Ccw => wl_output::Transform::_90,
        Orientation::Rotate180 => wl_output::Transform::_180,
        Orientation::Rotate90Cw => wl_output::Transform::_270,
        // wl_output flips around the vertical axis before it rotates
        Orientation::HorizontalMirror => wl_output::Transform::Flipped,
        Orientation::Transpose => wl_output::Transform::Flipped90,
        Orientation::VerticalMirror => wl_output::Transform::Flipped180,
        Orientation::Transverse => wl_output::Transform::Flipped270,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_picture_fits_inside_the_configured_window() {
        const PICTURE: (u32, u32) = (1920, 1080);
        let panel = (NonZeroU32::new(1024), NonZeroU32::new(600));
        let layout = |orientation, configured_size, force_aspect_ratio| {
            let layout = window_layout(PICTURE, orientation, configured_size, force_aspect_ratio);
            (layout.window_width, layout.window_height, layout.picture)
        };
        let rectangle = |x, y, width, height| ViewportRectangle {
            x,
            y,
            width,
            height,
        };
        assert_eq!(
            layout(Orientation::Identity, panel, true),
            (1024, 600, rectangle(0, 12, 1024, 576))
        );
        assert_eq!(
            layout(Orientation::Rotate90Cw, panel, true),
            (1024, 600, rectangle(343, 0, 337, 600))
        );
        assert_eq!(
            layout(Orientation::Identity, panel, false),
            (1024, 600, rectangle(0, 0, 1024, 600))
        );
        assert_eq!(
            layout(Orientation::Identity, (None, None), true),
            (1920, 1080, rectangle(0, 0, 1920, 1080))
        );
        assert_eq!(
            layout(Orientation::Rotate90Cw, (None, None), true),
            (1080, 1920, rectangle(0, 0, 1080, 1920))
        );
        assert_eq!(
            layout(Orientation::Identity, (NonZeroU32::new(800), None), true),
            (800, 1080, rectangle(0, 315, 800, 450))
        );
    }
    use g2g_core::{BusMessage, Dim, Rate, VideoCodec, VIDEO_SINK_MAX_LATENESS_NS};

    #[test]
    fn buffer_transform_inverts_the_descriptor() {
        use wl_output::Transform;
        assert_eq!(buffer_transform(Orientation::Identity), Transform::Normal);
        assert_eq!(buffer_transform(Orientation::Rotate90Cw), Transform::_90);
        assert_eq!(buffer_transform(Orientation::Rotate180), Transform::_180);
        assert_eq!(buffer_transform(Orientation::Rotate90Ccw), Transform::_270);
        assert_eq!(
            buffer_transform(Orientation::HorizontalMirror),
            Transform::Flipped
        );
        assert_eq!(
            buffer_transform(Orientation::Transpose),
            Transform::Flipped90
        );
        assert_eq!(
            buffer_transform(Orientation::VerticalMirror),
            Transform::Flipped180
        );
        assert_eq!(
            buffer_transform(Orientation::Transverse),
            Transform::Flipped270
        );
    }

    #[test]
    fn advertises_that_it_absorbs_orientation() {
        let expected = if cfg!(feature = "metadata") {
            OrientationSet::ALL
        } else {
            OrientationSet::EMPTY
        };
        assert_eq!(WaylandSink::new().absorbed_orientations(), expected);
    }

    #[test]
    fn dmabufs_import_by_their_drm_format_code() {
        // drm_fourcc.h values of DRM_FORMAT_NV12 and DRM_FORMAT_YUYV
        assert_eq!(drm_format(RawVideoFormat::Nv12), Some(0x3231_564e));
        assert_eq!(drm_format(RawVideoFormat::Yuyv), Some(0x5659_5559));
    }

    #[test]
    fn declares_system_memory_and_dmabuf_only() {
        // a wrong declaration makes `nvdec ! waylandsink` fail with UnsupportedDomain
        assert_eq!(
            WaylandSink::new().input_domains(),
            DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::DmaBuf)
        );
    }

    #[test]
    fn intercept_passes_through_any_format() {
        let sink = WaylandSink::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert_eq!(sink.intercept_caps(&h264), Ok(h264));
    }

    #[test]
    fn intercept_passes_through_nv12() {
        let sink = WaylandSink::new();
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        };
        assert_eq!(sink.intercept_caps(&nv12), Ok(nv12));
    }

    #[test]
    fn accepts_the_compositor_formats_nv12_first() {
        let formats = CompositorFormats {
            shm: alloc::vec![
                RawVideoFormat::Bgrx8,
                RawVideoFormat::Nv12,
                RawVideoFormat::Yuyv
            ],
            dmabuf: alloc::vec![RawVideoFormat::Nv12, RawVideoFormat::Uyvy],
        };
        let expected: Vec<Caps> = [
            RawVideoFormat::Nv12,
            RawVideoFormat::Bgrx8,
            RawVideoFormat::Yuyv,
            RawVideoFormat::Uyvy,
        ]
        .into_iter()
        .map(any_geometry)
        .collect();
        assert_eq!(formats.accepted_caps().alternatives(), expected.as_slice());
    }

    #[test]
    fn shm_codes_map_to_formats_and_back() {
        // wl_shm codes XRGB8888 as 1, every other format as its DRM fourcc
        let xrgb = drm_format_of_shm(WEnum::Value(wl_shm::Format::Xrgb8888));
        assert_eq!(raw_video_format(xrgb), Some(RawVideoFormat::Bgrx8));
        assert_eq!(
            shm_format(RawVideoFormat::Bgrx8),
            Some(wl_shm::Format::Xrgb8888)
        );
        let nv12 = drm_format_of_shm(WEnum::Value(wl_shm::Format::Nv12));
        assert_eq!(raw_video_format(nv12), Some(RawVideoFormat::Nv12));
        assert_eq!(shm_format(RawVideoFormat::Nv12), Some(wl_shm::Format::Nv12));
        let rgb565 = drm_format_of_shm(WEnum::Value(wl_shm::Format::Rgb565));
        assert_eq!(raw_video_format(rgb565), None);
    }

    #[test]
    fn colour_representation_follows_the_caps_colorimetry() {
        assert_eq!(
            coefficients_and_range(Colorimetry::BT709),
            (Coefficients::Bt709, Range::Limited)
        );
        assert_eq!(
            coefficients_and_range(Colorimetry::UNKNOWN),
            (Coefficients::Bt601, Range::Limited)
        );
    }

    #[test]
    fn configure_rejects_undecoded_video() {
        let mut sink = WaylandSink::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert_eq!(
            sink.configure_pipeline(&h264).err(),
            Some(G2gError::CapsMismatch)
        );
        assert!(
            sink.worker.is_none(),
            "no worker should be spawned on rejected caps"
        );
    }

    #[test]
    fn set_clock_sync_enables_pts_pacing() {
        let mut sink = WaylandSink::new();
        assert!(!sink.pacer.is_paced());
        let sync = ClockSync::new(Arc::new(WaylandClock), 0);
        AsyncElement::set_clock_sync(&mut sink, sync);
        assert!(sink.pacer.is_paced(), "clock sync stored, PTS pacing on");
    }

    #[test]
    fn qos_lateness_decision_respects_the_bound() {
        let sink = WaylandSink::new();
        assert!(
            !sink.pacer.is_too_late(0, VIDEO_SINK_MAX_LATENESS_NS),
            "within the default bound"
        );
        assert!(
            sink.pacer.is_too_late(0, VIDEO_SINK_MAX_LATENESS_NS + 1),
            "past the default bound"
        );

        let strict = WaylandSink::new().with_max_lateness_ns(0);
        assert!(!strict.pacer.is_too_late(100, 100), "on time is not late");
        assert!(
            strict.pacer.is_too_late(100, 101),
            "1ns past the deadline is late"
        );

        let tol = WaylandSink::new().with_max_lateness_ns(10);
        assert!(!tol.pacer.is_too_late(100, 110), "within tolerance");
        assert!(tol.pacer.is_too_late(100, 111), "beyond tolerance");
    }

    struct ManualClock(Arc<AtomicU64>);
    impl PipelineClock for ManualClock {
        fn now_ns(&self) -> u64 {
            self.0.load(Ordering::Relaxed)
        }
    }

    struct NullOut;
    impl OutputSink for NullOut {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            packet_slot.take();
            core::task::Poll::Ready(Ok(g2g_core::PushOutcome::Accepted))
        }
    }

    fn nv12_frame(pts_ns: u64) -> Frame {
        use g2g_core::frame::FrameTiming;
        use g2g_core::memory::{MemoryDomain, SystemSlice};
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            timing: FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            sequence: 0,
            meta: Default::default(),
        }
    }

    #[tokio::test]
    async fn qos_drops_a_late_frame_and_posts_to_the_bus() {
        // a late frame drops before any compositor I/O
        use g2g_core::clock::PlayAnchor;
        let (bus, handle) = g2g_core::Bus::new(4);
        let clock = Arc::new(AtomicU64::new(1_000_000));
        let anchor = PlayAnchor::new();
        anchor.stamp(0);
        let mut sink = WaylandSink::new().with_max_lateness_ns(0).with_bus(handle);
        AsyncElement::set_clock_sync(
            &mut sink,
            ClockSync::with_play_anchor(Arc::new(ManualClock(clock)), 0, anchor),
        );

        // a PTS 0 frame is 1 ms late on this clock
        let mut out = NullOut;
        sink.process(PipelinePacket::DataFrame(nv12_frame(0)), &mut out)
            .await
            .unwrap();

        assert_eq!(sink.late_dropped(), 1, "the late frame was dropped");
        let upstream = AsyncElement::take_qos(&mut sink).expect("upstream QoS report");
        assert_eq!(upstream.jitter_ns, 1_000_000);
        assert!(
            AsyncElement::take_qos(&mut sink).is_none(),
            "the report is consumed once"
        );
        match bus.try_recv() {
            Some(BusMessage::Qos {
                running_time_ns,
                jitter_ns,
                dropped,
                ..
            }) => {
                assert_eq!(running_time_ns, 0, "deadline reported");
                assert_eq!(jitter_ns, 1_000_000, "1 ms late");
                assert_eq!(dropped, 1, "cumulative drop count");
            }
            other => panic!("expected a Qos message, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn qos_default_does_not_drop() {
        let clock = Arc::new(AtomicU64::new(5_000_000));
        let mut sink = WaylandSink::new();
        AsyncElement::set_clock_sync(&mut sink, ClockSync::new(Arc::new(ManualClock(clock)), 0));
        assert!(
            !sink.pacer.is_too_late(0, 5_000_000),
            "anchored first frame on time"
        );
        assert_eq!(sink.late_dropped(), 0);
    }

    #[test]
    fn running_time_uses_pts_without_segment() {
        let sink = WaylandSink::new();
        assert_eq!(sink.pacer.running_time(50_000_000), Some(50_000_000));
    }

    #[test]
    fn running_time_maps_through_segment_and_clips() {
        // an accurate seek to 70 ms clips the frames before it
        let mut sink = WaylandSink::new();
        let seg = g2g_core::Segment::for_flush_seek(&g2g_core::Seek::flush_to(70_000_000), None);
        sink.pacer.set_segment(seg);
        assert_eq!(
            sink.pacer.running_time(66_000_000),
            None,
            "pre-target frame clips"
        );
        assert_eq!(
            sink.pacer.running_time(70_000_000),
            Some(0),
            "the target frame is running-time zero after a flushing seek"
        );
    }

    #[test]
    fn configure_rejects_odd_dims() {
        let mut sink = WaylandSink::new();
        let odd = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(641),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        };
        match sink.configure_pipeline(&odd) {
            Err(G2gError::CapsMismatch) => {}
            other => panic!("expected CapsMismatch on odd dims, got {other:?}"),
        }
    }
}

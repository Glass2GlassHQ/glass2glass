//! libcamera capture source. Streams NV12 (else YUYV) frames from a camera via
//! the libcamera stack, the modern Linux camera path. Linux-only (`libcamera`
//! feature); links the system libcamera through the `libcamera` crate.
//!
//! libcamera handles UVC webcams through its `uvcvideo` pipeline handler (so it
//! covers the same devices as [`V4l2Src`](crate::v4l2src::V4l2Src)), plus
//! CSI/ISP cameras (Raspberry Pi, embedded SoCs) that need an ISP pipeline V4L2
//! alone cannot drive.
//!
//! Pipeline shape: `LibCameraSrc -> [VideoConvert] -> sink`. The source asks
//! libcamera for NV12 (planar, decoder / ML / GPU friendly) and falls back to
//! YUYV (the near-universal UVC packed format) when the camera does not offer
//! NV12; `VideoConvert` unpacks YUYV downstream when needed.
//!
//! libcamera is callback-driven and its objects are thread-affine, so all
//! libcamera work (manager, camera, requests, completion callback) runs on a
//! dedicated capture thread that feeds the async `run` loop over a bounded
//! channel, the same structure as `V4l2Src`. The format / geometry is
//! negotiated up front in [`intercept_caps`](LibCameraSrc::intercept_caps)
//! (libcamera may adjust the requested size), and the capture thread
//! re-configures the camera under that exact format. Keeping no libcamera
//! handle in the struct between negotiation and `run` keeps `LibCameraSrc`
//! `Send` and sidesteps the thread-affinity contract.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError,
    HardwareError, LatencyReport, MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat,
};

use libcamera::camera::CameraConfigurationStatus;
use libcamera::camera_manager::CameraManager;
use libcamera::framebuffer::AsFrameBuffer;
use libcamera::framebuffer_allocator::{FrameBuffer, FrameBufferAllocator};
use libcamera::framebuffer_map::MemoryMappedFrameBuffer;
use libcamera::geometry::Size;
use libcamera::pixel_format::PixelFormat;
use libcamera::request::ReuseFlag;
use libcamera::stream::StreamRole;

/// Default advisory frame rate when the caller does not specify one. Geometry
/// is left at `0` (let libcamera pick its ViewFinder default) unless requested.
const DEFAULT_FPS: u32 = 30;
/// FrameBuffer ring depth requested from libcamera. Doubles as the async channel
/// bound, so the capture thread blocks (backpressure) rather than outrunning the
/// pipeline.
const BUFFER_COUNT: usize = 4;

/// NV12 (planar 4:2:0), the preferred output: Y plane then interleaved UV.
const PF_NV12: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'N', b'V', b'1', b'2']), 0);
/// YUYV (packed 4:2:2), the UVC-universal fallback.
const PF_YUYV: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'Y', b'V']), 0);

/// Map a libcamera / OS failure to the generic hardware-error arm. libcamera
/// has no dedicated `HardwareError` variant; errno is preserved where one is
/// available, else `-1`.
fn lc_err(e: &std::io::Error) -> G2gError {
    G2gError::Hardware(HardwareError::Io(e.raw_os_error().unwrap_or(-1)))
}

fn lc_other() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// Translate a libcamera `PixelFormat` to our `RawVideoFormat`. Only the two
/// formats the source negotiates are recognised.
fn map_format(pf: PixelFormat) -> Option<RawVideoFormat> {
    if pf == PF_NV12 {
        Some(RawVideoFormat::Nv12)
    } else if pf == PF_YUYV {
        Some(RawVideoFormat::Yuyv)
    } else {
        None
    }
}

#[derive(Debug)]
pub struct LibCameraSrc {
    /// Index into libcamera's enumerated camera list (default 0).
    camera_index: usize,
    /// Requested geometry; `0` means "let libcamera pick its default".
    req_width: u32,
    req_height: u32,
    /// Best-effort frame rate, used for PTS stamping and the latency report.
    /// libcamera frame-duration enforcement (`FrameDurationLimits`) is a
    /// follow-up; the camera captures at its own default cadence today.
    req_fps: u32,
    /// Stop after this many frames; `0` = run until the pipeline shuts down.
    frame_limit: u64,
    /// Cached negotiation result: (format, width, height, fps). Set by
    /// [`negotiate`](Self::negotiate), consumed by `run`.
    negotiated: Option<(RawVideoFormat, u32, u32, u32)>,
    configured: bool,
}

impl LibCameraSrc {
    /// Capture from the first enumerated camera at the default geometry.
    pub fn new() -> Self {
        Self {
            camera_index: 0,
            req_width: 0,
            req_height: 0,
            req_fps: DEFAULT_FPS,
            frame_limit: 0,
            negotiated: None,
            configured: false,
        }
    }

    /// Select which enumerated camera to open (default 0).
    pub fn with_camera(mut self, index: usize) -> Self {
        self.camera_index = index;
        self
    }

    /// Request a capture geometry. libcamera may adjust it; the negotiated size
    /// is what the chain sees.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.req_width = width;
        self.req_height = height;
        self
    }

    /// Set the advisory frame rate (PTS / latency only for now).
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.req_fps = fps;
        self
    }

    /// Stop after `n` frames (`0` = unlimited).
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// Probe the camera: acquire it, generate a ViewFinder configuration, try
    /// NV12 then YUYV, validate, and read back the format libcamera settled on.
    /// The camera is released before `run`. Caches the result for `run` and for
    /// repeat `caps_constraint` calls during re-fixate.
    fn negotiate(&mut self) -> Result<Caps, G2gError> {
        if let Some((format, w, h, fps)) = self.negotiated {
            return Ok(raw_caps(format, w, h, fps));
        }

        let mgr = CameraManager::new().map_err(|e| lc_err(&e))?;
        let cameras = mgr.cameras();
        let cam = cameras.get(self.camera_index).ok_or_else(lc_other)?;
        let cam = cam.acquire().map_err(|e| lc_err(&e))?;

        // Try NV12 first, then YUYV; accept whichever survives validation
        // unchanged. Fall back to libcamera's default format if neither holds.
        let mut chosen: Option<(RawVideoFormat, u32, u32)> = None;
        for pf in [PF_NV12, PF_YUYV] {
            let mut cfgs = cam
                .generate_configuration(&[StreamRole::ViewFinder])
                .ok_or_else(lc_other)?;
            {
                let mut cfg = cfgs.get_mut(0).ok_or_else(lc_other)?;
                cfg.set_pixel_format(pf);
                if self.req_width > 0 && self.req_height > 0 {
                    cfg.set_size(Size {
                        width: self.req_width,
                        height: self.req_height,
                    });
                }
            }
            if matches!(cfgs.validate(), CameraConfigurationStatus::Invalid) {
                continue;
            }
            let cfg = cfgs.get(0).ok_or_else(lc_other)?;
            if cfg.get_pixel_format() == pf {
                let size = cfg.get_size();
                chosen = Some((map_format(pf).unwrap(), size.width, size.height));
                break;
            }
        }

        // Fallback: whatever the default ViewFinder config validates to, if it
        // is a format we can carry.
        let (format, w, h) = match chosen {
            Some(c) => c,
            None => {
                let mut cfgs = cam
                    .generate_configuration(&[StreamRole::ViewFinder])
                    .ok_or_else(lc_other)?;
                if self.req_width > 0 && self.req_height > 0 {
                    let mut cfg = cfgs.get_mut(0).ok_or_else(lc_other)?;
                    cfg.set_size(Size {
                        width: self.req_width,
                        height: self.req_height,
                    });
                }
                if matches!(cfgs.validate(), CameraConfigurationStatus::Invalid) {
                    return Err(G2gError::CapsMismatch);
                }
                let cfg = cfgs.get(0).ok_or_else(lc_other)?;
                let format = map_format(cfg.get_pixel_format()).ok_or(G2gError::CapsMismatch)?;
                let size = cfg.get_size();
                (format, size.width, size.height)
            }
        };

        self.negotiated = Some((format, w, h, self.req_fps));
        Ok(raw_caps(format, w, h, self.req_fps))
    }
}

impl Default for LibCameraSrc {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the `Caps::RawVideo` for a negotiated format/geometry/rate.
fn raw_caps(format: RawVideoFormat, w: u32, h: u32, fps: u32) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(fps << 16),
    }
}

impl SourceLoop for LibCameraSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        // The probe is synchronous (acquire + configure + validate, no
        // streaming), so no async body is needed.
        core::future::ready(self.negotiate())
    }

    /// Produces the format libcamera settles on during the probe, so a chain
    /// built on the camera takes the native arc-consistency path. Mirrors
    /// `V4l2Src`.
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(
            self.negotiate().map(|caps| CapsConstraint::Produces(CapsSet::one(caps))),
        )
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.negotiated.is_none() {
            return Err(G2gError::NotConfigured);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "libcamera camera source",
            "Source/Video",
            "Captures video from a camera via libcamera (NV12 / YUYV)",
            "g2g",
        )
    }

    /// Live source: contributes one frame period so the sink keeps a frame in
    /// hand and never runs dry waiting on capture.
    fn latency(&self) -> LatencyReport {
        let fps = self.negotiated.map(|(_, _, _, f)| f).unwrap_or(self.req_fps);
        let period_ns = if fps > 0 { 1_000_000_000 / fps as u64 } else { 0 };
        LatencyReport::live(period_ns, None)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        LIBCAMERA_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "camera" => {
                self.camera_index = value.as_uint().ok_or(PropError::Type)? as usize;
                Ok(())
            }
            "width" => {
                self.req_width = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "height" => {
                self.req_height = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "framerate" => {
                self.req_fps = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "camera" => Some(PropValue::Uint(self.camera_index as u64)),
            "width" => Some(PropValue::Uint(self.req_width as u64)),
            "height" => Some(PropValue::Uint(self.req_height as u64)),
            "framerate" => Some(PropValue::Uint(self.req_fps as u64)),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let (format, w, h, fps) = self.negotiated.ok_or(G2gError::NotConfigured)?;
            let limit = self.frame_limit;
            let index = self.camera_index;
            let pf = match format {
                RawVideoFormat::Nv12 => PF_NV12,
                RawVideoFormat::Yuyv => PF_YUYV,
                // negotiate only ever stores Nv12 / Yuyv.
                _ => return Err(G2gError::CapsMismatch),
            };

            // Bounded channel: the capture thread blocks once the pipeline is
            // BUFFER_COUNT frames behind, so memory stays bounded (backpressure).
            let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(BUFFER_COUNT);

            // All libcamera interaction lives on this thread: the objects are
            // thread-affine and completions arrive on a libcamera callback.
            let handle = std::thread::spawn(move || -> Result<(), G2gError> {
                capture_loop(index, pf, w, h, limit, tx)
            });

            let pts_step_ns = if fps > 0 { 1_000_000_000 / fps as u64 } else { 0 };
            let mut seq = 0u64;
            while let Some(bytes) = rx.recv().await {
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                let pts = seq * pts_step_ns;
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: pts_step_ns,
                        capture_ns: pts,
                        arrival_ns,
                        keyframe: true, // raw frames are each independently presentable
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
                seq += 1;
            }

            // Surface a capture-thread failure that produced nothing, rather
            // than masking it as a clean EOS.
            let thread_result = handle.join().unwrap_or_else(|_| Err(lc_other()));
            if seq == 0 {
                thread_result?;
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

/// The blocking libcamera capture loop, run on its own thread. Acquires the
/// camera, configures it for `pf` at `w`x`h`, allocates and queues a request
/// ring, and forwards each completed frame's packed bytes over `tx` until the
/// frame limit is hit or the receiver is dropped.
fn capture_loop(
    index: usize,
    pf: PixelFormat,
    w: u32,
    h: u32,
    limit: u64,
    tx: tokio::sync::mpsc::Sender<Vec<u8>>,
) -> Result<(), G2gError> {
    let mgr = CameraManager::new().map_err(|e| lc_err(&e))?;
    let cameras = mgr.cameras();
    let cam = cameras.get(index).ok_or_else(lc_other)?;
    let mut cam = cam.acquire().map_err(|e| lc_err(&e))?;

    let mut cfgs = cam
        .generate_configuration(&[StreamRole::ViewFinder])
        .ok_or_else(lc_other)?;
    {
        let mut cfg = cfgs.get_mut(0).ok_or_else(lc_other)?;
        cfg.set_pixel_format(pf);
        cfg.set_size(Size { width: w, height: h });
    }
    if matches!(cfgs.validate(), CameraConfigurationStatus::Invalid) {
        return Err(G2gError::CapsMismatch);
    }
    cam.configure(&mut cfgs).map_err(|e| lc_err(&e))?;

    // The stream handle is owned once copied out (it points into `cfgs`, which
    // stays alive for the whole function); the `cfg` borrow ends here.
    let stream = cfgs.get(0).ok_or_else(lc_other)?.stream().ok_or_else(lc_other)?;

    let mut alloc = FrameBufferAllocator::new(&cam);
    let buffers = alloc.alloc(&stream).map_err(|e| lc_err(&e))?;
    let mapped = buffers
        .into_iter()
        .map(MemoryMappedFrameBuffer::new)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| lc_other())?;

    // One request per buffer; each owns its mapped framebuffer.
    let mut reqs = Vec::with_capacity(mapped.len());
    for buf in mapped {
        let mut req = cam.create_request(None).ok_or_else(lc_other)?;
        req.add_buffer(&stream, buf).map_err(|e| lc_err(&e))?;
        reqs.push(req);
    }

    // Completed requests return on a libcamera callback thread; forward them to
    // this loop over a std channel.
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    cam.on_request_completed(move |req| {
        let _ = done_tx.send(req);
    });

    cam.start(None).map_err(|e| lc_err(&e))?;
    for req in reqs.drain(..) {
        cam.queue_request(req).map_err(|(_, e)| lc_err(&e))?;
    }

    let mut count = 0u64;
    while limit == 0 || count < limit {
        // First frame can be slow (exposure/AGC settle); allow generous time.
        let mut req = match done_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(r) => r,
            Err(_) => break,
        };

        // Pack every plane's used bytes contiguously: for NV12 this yields
        // Y followed by interleaved UV (tight NV12); for YUYV the single
        // packed plane.
        let payload = {
            let fb: &MemoryMappedFrameBuffer<FrameBuffer> =
                req.buffer(&stream).ok_or_else(lc_other)?;
            let planes = fb.data();
            let meta = fb.metadata();
            let mut payload = Vec::new();
            for (i, plane) in planes.iter().enumerate() {
                let used = meta
                    .as_ref()
                    .and_then(|m| m.planes().get(i))
                    .map(|p| p.bytes_used as usize)
                    .unwrap_or(plane.len())
                    .min(plane.len());
                payload.extend_from_slice(&plane[..used]);
            }
            payload
        };

        // Err means the pipeline shut down (receiver dropped).
        if tx.blocking_send(payload).is_err() {
            break;
        }
        count += 1;

        // Recycle the request's buffers and re-queue it for the next frame.
        req.reuse(ReuseFlag::REUSE_BUFFERS);
        if cam.queue_request(req).is_err() {
            break;
        }
    }

    cam.stop().map_err(|e| lc_err(&e))?;
    Ok(())
}

static LIBCAMERA_PROPS: &[PropertySpec] = &[
    PropertySpec::new("camera", PropKind::Uint, "camera index (libcamera enumeration order)"),
    PropertySpec::new("width", PropKind::Uint, "requested capture width in pixels"),
    PropertySpec::new("height", PropKind::Uint, "requested capture height in pixels"),
    PropertySpec::new("framerate", PropKind::Uint, "advisory frame rate (PTS / latency)"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_before_negotiation_is_not_configured() {
        // configure_pipeline must reject until intercept_caps has probed.
        let mut src = LibCameraSrc::new();
        let caps = raw_caps(RawVideoFormat::Nv12, 640, 480, 30);
        let err = src
            .configure_pipeline(&caps)
            .expect_err("configure without negotiate must fail");
        assert_eq!(err, G2gError::NotConfigured);
    }

    #[test]
    fn properties_round_trip() {
        let mut src = LibCameraSrc::new();
        src.set_property("camera", PropValue::Uint(1)).unwrap();
        src.set_property("width", PropValue::Uint(1280)).unwrap();
        src.set_property("height", PropValue::Uint(720)).unwrap();
        src.set_property("framerate", PropValue::Uint(60)).unwrap();
        assert_eq!(src.get_property("camera"), Some(PropValue::Uint(1)));
        assert_eq!(src.get_property("width"), Some(PropValue::Uint(1280)));
        assert_eq!(src.get_property("height"), Some(PropValue::Uint(720)));
        assert_eq!(src.get_property("framerate"), Some(PropValue::Uint(60)));
        assert_eq!(
            src.set_property("nope", PropValue::Uint(0)),
            Err(PropError::Unknown)
        );
        assert_eq!(
            src.set_property("camera", PropValue::Str("x".into())),
            Err(PropError::Type)
        );
    }

    #[test]
    fn maps_known_formats_only() {
        assert_eq!(map_format(PF_NV12), Some(RawVideoFormat::Nv12));
        assert_eq!(map_format(PF_YUYV), Some(RawVideoFormat::Yuyv));
        let rgb = PixelFormat::new(u32::from_le_bytes([b'R', b'G', b'2', b'4']), 0);
        assert_eq!(map_format(rgb), None);
    }
}

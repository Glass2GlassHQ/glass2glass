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
    PropValue, PropertySpec, Rate, RawVideoFormat, VideoCodec,
};

use libcamera::camera::CameraConfigurationStatus;
use libcamera::camera_manager::CameraManager;
use libcamera::control::ControlList;
use libcamera::controls::FrameDurationLimits;
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

/// NV12 (planar 4:2:0), the preferred raw output: Y plane then interleaved UV.
const PF_NV12: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'N', b'V', b'1', b'2']), 0);
/// YUYV (packed 4:2:2), the UVC-universal raw fallback.
const PF_YUYV: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'Y', b'V']), 0);
/// MJPEG (on-camera baseline JPEG per frame). The UVC high-frame-rate path:
/// compressing on the camera fits resolutions over USB that uncompressed YUYV
/// cannot. Decoded downstream by `MjpegDec`. Note the fourcc is `MJPG`.
const PF_MJPEG: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'M', b'J', b'P', b'G']), 0);

/// The output the source negotiates with the camera. Carries the libcamera
/// pixel format and the `Caps` it maps to (raw vs compressed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutKind {
    Nv12,
    Yuyv,
    Mjpeg,
}

impl OutKind {
    fn pixel_format(self) -> PixelFormat {
        match self {
            OutKind::Nv12 => PF_NV12,
            OutKind::Yuyv => PF_YUYV,
            OutKind::Mjpeg => PF_MJPEG,
        }
    }

    fn from_pixel_format(pf: PixelFormat) -> Option<Self> {
        if pf == PF_NV12 {
            Some(OutKind::Nv12)
        } else if pf == PF_YUYV {
            Some(OutKind::Yuyv)
        } else if pf == PF_MJPEG {
            Some(OutKind::Mjpeg)
        } else {
            None
        }
    }

    /// The `Caps` this output produces at the negotiated geometry / rate. Raw
    /// formats map to `RawVideo`; MJPEG to `CompressedVideo{Mjpeg}`.
    fn caps(self, w: u32, h: u32, fps: u32) -> Caps {
        let width = Dim::Fixed(w);
        let height = Dim::Fixed(h);
        let framerate = Rate::Fixed(fps << 16);
        match self {
            OutKind::Nv12 => Caps::RawVideo { format: RawVideoFormat::Nv12, width, height, framerate },
            OutKind::Yuyv => Caps::RawVideo { format: RawVideoFormat::Yuyv, width, height, framerate },
            OutKind::Mjpeg => {
                Caps::CompressedVideo { codec: VideoCodec::Mjpeg, width, height, framerate }
            }
        }
    }
}

/// Map a libcamera / OS failure to the generic hardware-error arm. libcamera
/// has no dedicated `HardwareError` variant; errno is preserved where one is
/// available, else `-1`.
fn lc_err(e: &std::io::Error) -> G2gError {
    G2gError::Hardware(HardwareError::Io(e.raw_os_error().unwrap_or(-1)))
}

fn lc_other() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

#[derive(Debug)]
pub struct LibCameraSrc {
    /// Index into libcamera's enumerated camera list (default 0).
    camera_index: usize,
    /// Requested geometry; `0` means "let libcamera pick its default".
    req_width: u32,
    req_height: u32,
    /// Capture frame rate. Enforced on the camera via a fixed
    /// `FrameDurationLimits` (min == max) at `start`, and also used for PTS
    /// stamping and the latency report. `0` lets the camera run at its default
    /// cadence.
    req_fps: u32,
    /// Stop after this many frames; `0` = run until the pipeline shuts down.
    frame_limit: u64,
    /// Request MJPEG (compressed) output instead of raw NV12/YUYV. MJPEG fits
    /// high frame rates / resolutions over USB that uncompressed YUYV cannot;
    /// `MjpegDec` decodes it downstream.
    prefer_mjpeg: bool,
    /// Cached negotiation result: (output kind, width, height, fps). Set by
    /// [`negotiate`](Self::negotiate), consumed by `run`.
    negotiated: Option<(OutKind, u32, u32, u32)>,
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
            prefer_mjpeg: false,
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

    /// Set the capture frame rate. Enforced on the camera via
    /// `FrameDurationLimits`; `0` keeps the camera's default cadence.
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.req_fps = fps;
        self
    }

    /// Stop after `n` frames (`0` = unlimited).
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// Request MJPEG output (compressed, `CompressedVideo{Mjpeg}`) instead of
    /// raw NV12/YUYV. Pair with `MjpegDec` downstream. Enables high frame rates
    /// the uncompressed YUYV mode cannot sustain over USB.
    pub fn with_mjpeg(mut self, on: bool) -> Self {
        self.prefer_mjpeg = on;
        self
    }

    /// Probe the camera: acquire it, generate a ViewFinder configuration, try
    /// NV12 then YUYV, validate, and read back the format libcamera settled on.
    /// The camera is released before `run`. Caches the result for `run` and for
    /// repeat `caps_constraint` calls during re-fixate.
    fn negotiate(&mut self) -> Result<Caps, G2gError> {
        if let Some((kind, w, h, fps)) = self.negotiated {
            return Ok(kind.caps(w, h, fps));
        }

        let mgr = CameraManager::new().map_err(|e| lc_err(&e))?;
        let cameras = mgr.cameras();
        let cam = cameras.get(self.camera_index).ok_or_else(lc_other)?;
        let cam = cam.acquire().map_err(|e| lc_err(&e))?;

        // MJPEG mode asks for MJPEG only; raw mode prefers NV12 then YUYV.
        // Accept whichever candidate survives validation unchanged.
        let candidates: &[OutKind] = if self.prefer_mjpeg {
            &[OutKind::Mjpeg]
        } else {
            &[OutKind::Nv12, OutKind::Yuyv]
        };

        let mut chosen: Option<(OutKind, u32, u32)> = None;
        for &kind in candidates {
            let mut cfgs = cam
                .generate_configuration(&[StreamRole::ViewFinder])
                .ok_or_else(lc_other)?;
            {
                let mut cfg = cfgs.get_mut(0).ok_or_else(lc_other)?;
                cfg.set_pixel_format(kind.pixel_format());
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
            if cfg.get_pixel_format() == kind.pixel_format() {
                let size = cfg.get_size();
                chosen = Some((kind, size.width, size.height));
                break;
            }
        }

        let (kind, w, h) = match chosen {
            Some(c) => c,
            // MJPEG mode is explicit: if the camera does not offer MJPEG there
            // is no sensible raw fallback (the caller asked for compressed).
            None if self.prefer_mjpeg => return Err(G2gError::CapsMismatch),
            // Raw mode: take whatever the default ViewFinder config validates
            // to, if it is a format we can carry.
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
                let kind =
                    OutKind::from_pixel_format(cfg.get_pixel_format()).ok_or(G2gError::CapsMismatch)?;
                let size = cfg.get_size();
                (kind, size.width, size.height)
            }
        };

        self.negotiated = Some((kind, w, h, self.req_fps));
        Ok(kind.caps(w, h, self.req_fps))
    }
}

impl Default for LibCameraSrc {
    fn default() -> Self {
        Self::new()
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
            "Captures video from a camera via libcamera (NV12 / YUYV / MJPEG)",
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
            // "mjpeg" / "raw" select the output kind; raw auto-picks NV12/YUYV.
            "format" => {
                self.prefer_mjpeg = match value.as_str().ok_or(PropError::Type)? {
                    "mjpeg" => true,
                    "raw" => false,
                    _ => return Err(PropError::Value),
                };
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
            "format" => Some(PropValue::Str(if self.prefer_mjpeg { "mjpeg" } else { "raw" }.into())),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let (kind, w, h, fps) = self.negotiated.ok_or(G2gError::NotConfigured)?;
            let limit = self.frame_limit;
            let index = self.camera_index;
            let pf = kind.pixel_format();

            // Bounded channel: the capture thread blocks once the pipeline is
            // BUFFER_COUNT frames behind, so memory stays bounded (backpressure).
            let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(BUFFER_COUNT);

            // All libcamera interaction lives on this thread: the objects are
            // thread-affine and completions arrive on a libcamera callback.
            let handle = std::thread::spawn(move || -> Result<(), G2gError> {
                capture_loop(index, pf, w, h, fps, limit, tx)
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
    fps: u32,
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

    // Cap the capture rate when a frame rate was requested. The minimum frame
    // duration (microseconds) bounds the *fastest* rate, so it caps fps at the
    // request; the maximum is left generous (a 1 fps floor) so a camera that
    // cannot sustain the requested rate falls back to its own ceiling instead
    // of collapsing. Setting min == max forces an exact interval, which a UVC
    // camera that cannot meet it (e.g. 30 fps uncompressed at higher
    // resolutions) handles by dropping to a much lower rate, so we avoid that.
    let start_controls = if fps > 0 {
        let min_us = (1_000_000 / fps as i64).max(1);
        let max_us = 1_000_000i64; // 1 fps floor; does not bind in practice.
        let mut ctrls = ControlList::new();
        ctrls
            .set(FrameDurationLimits([min_us, max_us.max(min_us)]))
            .map_err(|_| lc_other())?;
        Some(ctrls)
    } else {
        None
    };
    cam.start(start_controls.as_deref()).map_err(|e| lc_err(&e))?;
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
    PropertySpec::new("framerate", PropKind::Uint, "capture frame rate (enforced via FrameDurationLimits)"),
    PropertySpec::new("format", PropKind::Str, "output: raw (NV12/YUYV) | mjpeg (pair with mjpegdec)"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_before_negotiation_is_not_configured() {
        // configure_pipeline must reject until intercept_caps has probed.
        let mut src = LibCameraSrc::new();
        let caps = OutKind::Nv12.caps(640, 480, 30);
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
        // format property toggles MJPEG output.
        assert_eq!(src.get_property("format"), Some(PropValue::Str("raw".into())));
        src.set_property("format", PropValue::Str("mjpeg".into())).unwrap();
        assert_eq!(src.get_property("format"), Some(PropValue::Str("mjpeg".into())));
        assert_eq!(
            src.set_property("format", PropValue::Str("bogus".into())),
            Err(PropError::Value)
        );
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
    fn outkind_maps_pixel_formats_and_caps() {
        assert_eq!(OutKind::from_pixel_format(PF_NV12), Some(OutKind::Nv12));
        assert_eq!(OutKind::from_pixel_format(PF_YUYV), Some(OutKind::Yuyv));
        assert_eq!(OutKind::from_pixel_format(PF_MJPEG), Some(OutKind::Mjpeg));
        let rgb = PixelFormat::new(u32::from_le_bytes([b'R', b'G', b'2', b'4']), 0);
        assert_eq!(OutKind::from_pixel_format(rgb), None);

        // Raw kinds produce RawVideo; MJPEG produces CompressedVideo{Mjpeg}.
        assert!(matches!(
            OutKind::Nv12.caps(640, 480, 30),
            Caps::RawVideo { format: RawVideoFormat::Nv12, .. }
        ));
        assert!(matches!(
            OutKind::Mjpeg.caps(1280, 720, 30),
            Caps::CompressedVideo { codec: VideoCodec::Mjpeg, .. }
        ));
    }

    #[test]
    fn with_mjpeg_sets_preference() {
        let src = LibCameraSrc::new().with_mjpeg(true);
        assert_eq!(src.get_property("format"), Some(PropValue::Str("mjpeg".into())));
    }
}

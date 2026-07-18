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
use alloc::string::{String, ToString};
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
use libcamera::control::{Control, ControlInfoMap, ControlList};
use libcamera::controls::{
    AeEnable, AnalogueGain, Brightness, Contrast, ExposureTime, FrameDurationLimits, Saturation,
};
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
            OutKind::Nv12 => Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width,
                height,
                framerate,
            },
            OutKind::Yuyv => Caps::RawVideo {
                format: RawVideoFormat::Yuyv,
                width,
                height,
                framerate,
            },
            OutKind::Mjpeg => Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                width,
                height,
                framerate,
            },
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

/// Resolve the camera to open: when `id` is set, the index of the first camera
/// whose `id()` contains that substring; otherwise `index` unchanged. Reads
/// each camera's id through a short-lived handle, so it adds no borrow to the
/// returned value.
fn resolve_camera_index(
    cameras: &libcamera::camera_manager::CameraList<'_>,
    index: usize,
    id: Option<&str>,
) -> Option<usize> {
    match id {
        Some(want) => {
            (0..cameras.len()).find(|&i| cameras.get(i).is_some_and(|c| c.id().contains(want)))
        }
        None => Some(index),
    }
}

/// Set a control only if the camera advertises it. libcamera throws a C++
/// `std::out_of_range` (aborting the process across the FFI boundary) when a
/// control list passed to `start` carries an id the pipeline handler does not
/// support, so an unsupported control (e.g. `AnalogueGain` on a UVC webcam that
/// exposes only exposure) must be skipped, not set. Returns whether it was set.
fn set_if_supported<C: Control>(ctrls: &mut ControlList, infos: &ControlInfoMap, val: C) -> bool {
    if infos.count(C::ID) > 0 {
        // `set` only fails on a value/type mismatch we control here, so ignore.
        let _ = ctrls.set(val);
        true
    } else {
        false
    }
}

#[derive(Debug)]
pub struct LibCameraSrc {
    /// Index into libcamera's enumerated camera list (default 0). Ignored when
    /// `camera_id` is set.
    camera_index: usize,
    /// Select the camera by an id substring instead of by index (libcamera ids
    /// are stable across reboots, unlike enumeration order). The first camera
    /// whose `id()` contains this string is used.
    camera_id: Option<String>,
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
    /// Auto-exposure override. `None` = leave libcamera's default (AE on); a
    /// manual `exposure_us` / `gain` implies AE off. With AE on in a dim room
    /// the camera lengthens exposure and the frame rate collapses, so capping
    /// exposure is the real lever for high fps in low light.
    ae_enable: Option<bool>,
    /// Manual exposure time in microseconds (implies AE off). A short exposure
    /// lets the camera hit a high frame rate; pair with `gain` to keep the
    /// image bright.
    exposure_us: Option<i32>,
    /// Manual analogue gain (sensor ISO multiplier; implies AE off). Brightens a
    /// short exposure at the cost of noise.
    gain: Option<f32>,
    /// Brightness in [-1.0, 1.0] (0 = default). Unlike exposure / gain this is a
    /// post-capture image adjustment, so it brightens a short-exposure frame
    /// *without* lowering the frame rate, the low-light lever that keeps fps.
    brightness: Option<f32>,
    /// Contrast (1.0 = default).
    contrast: Option<f32>,
    /// Colour saturation (1.0 = default; 0 = greyscale).
    saturation: Option<f32>,
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
            camera_id: None,
            req_width: 0,
            req_height: 0,
            req_fps: DEFAULT_FPS,
            frame_limit: 0,
            prefer_mjpeg: false,
            ae_enable: None,
            exposure_us: None,
            gain: None,
            brightness: None,
            contrast: None,
            saturation: None,
            negotiated: None,
            configured: false,
        }
    }

    /// Select which enumerated camera to open by index (default 0).
    pub fn with_camera(mut self, index: usize) -> Self {
        self.camera_index = index;
        self
    }

    /// Select the camera by an id substring (e.g. a USB path fragment or a
    /// model string). Stable across reboots, unlike enumeration order; takes
    /// precedence over [`with_camera`](Self::with_camera).
    pub fn with_camera_id<S: Into<String>>(mut self, id: S) -> Self {
        self.camera_id = Some(id.into());
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

    /// Turn auto-exposure on or off explicitly. With AE on (the default) the
    /// camera lengthens exposure in low light and the frame rate drops; turn it
    /// off and set [`with_exposure`](Self::with_exposure) to hold a high rate.
    pub fn with_auto_exposure(mut self, on: bool) -> Self {
        self.ae_enable = Some(on);
        self
    }

    /// Set a manual exposure time in microseconds (disables auto-exposure). A
    /// short exposure (e.g. 8000 us) keeps the frame interval short enough for a
    /// high frame rate even in a dim room; compensate brightness with
    /// [`with_gain`](Self::with_gain).
    pub fn with_exposure(mut self, micros: i32) -> Self {
        self.exposure_us = Some(micros);
        self.ae_enable = Some(false);
        self
    }

    /// Set a manual analogue gain (disables auto-exposure). Brightens a short
    /// exposure at the cost of sensor noise.
    pub fn with_gain(mut self, gain: f32) -> Self {
        self.gain = Some(gain);
        self.ae_enable = Some(false);
        self
    }

    /// Set image brightness in `[-1.0, 1.0]` (0 = default). A post-capture
    /// adjustment that does not change the exposure time, so it brightens a
    /// short-exposure low-light frame without lowering the frame rate. Applied
    /// only if the camera supports it.
    pub fn with_brightness(mut self, brightness: f32) -> Self {
        self.brightness = Some(brightness);
        self
    }

    /// Set contrast (1.0 = default). Applied only if the camera supports it.
    pub fn with_contrast(mut self, contrast: f32) -> Self {
        self.contrast = Some(contrast);
        self
    }

    /// Set colour saturation (1.0 = default, 0 = greyscale). Applied only if the
    /// camera supports it.
    pub fn with_saturation(mut self, saturation: f32) -> Self {
        self.saturation = Some(saturation);
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
        let idx = resolve_camera_index(&cameras, self.camera_index, self.camera_id.as_deref())
            .ok_or_else(lc_other)?;
        let cam = cameras.get(idx).ok_or_else(lc_other)?;
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
                let kind = OutKind::from_pixel_format(cfg.get_pixel_format())
                    .ok_or(G2gError::CapsMismatch)?;
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
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
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
            self.negotiate()
                .map(|caps| CapsConstraint::Produces(CapsSet::one(caps))),
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
        let fps = self
            .negotiated
            .map(|(_, _, _, f)| f)
            .unwrap_or(self.req_fps);
        let period_ns = if fps > 0 {
            1_000_000_000 / fps as u64
        } else {
            0
        };
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
            "camera-id" => {
                self.camera_id = Some(value.as_str().ok_or(PropError::Type)?.to_string());
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
            "auto-exposure" => {
                self.ae_enable = Some(value.as_bool().ok_or(PropError::Type)?);
                Ok(())
            }
            // Manual exposure (us) / gain both imply auto-exposure off.
            "exposure" => {
                self.exposure_us = Some(value.as_int().ok_or(PropError::Type)? as i32);
                self.ae_enable = Some(false);
                Ok(())
            }
            "gain" => {
                self.gain = Some(value.as_double().ok_or(PropError::Type)? as f32);
                self.ae_enable = Some(false);
                Ok(())
            }
            "brightness" => {
                self.brightness = Some(value.as_double().ok_or(PropError::Type)? as f32);
                Ok(())
            }
            "contrast" => {
                self.contrast = Some(value.as_double().ok_or(PropError::Type)? as f32);
                Ok(())
            }
            "saturation" => {
                self.saturation = Some(value.as_double().ok_or(PropError::Type)? as f32);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "camera" => Some(PropValue::Uint(self.camera_index as u64)),
            "camera-id" => self.camera_id.clone().map(PropValue::Str),
            "width" => Some(PropValue::Uint(self.req_width as u64)),
            "height" => Some(PropValue::Uint(self.req_height as u64)),
            "framerate" => Some(PropValue::Uint(self.req_fps as u64)),
            "format" => Some(PropValue::Str(
                if self.prefer_mjpeg { "mjpeg" } else { "raw" }.into(),
            )),
            "auto-exposure" => self.ae_enable.map(PropValue::Bool),
            "exposure" => self.exposure_us.map(|e| PropValue::Int(e as i64)),
            "gain" => self.gain.map(|g| PropValue::Double(g as f64)),
            "brightness" => self.brightness.map(|b| PropValue::Double(b as f64)),
            "contrast" => self.contrast.map(|c| PropValue::Double(c as f64)),
            "saturation" => self.saturation.map(|s| PropValue::Double(s as f64)),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let (kind, w, h, fps) = self.negotiated.ok_or(G2gError::NotConfigured)?;
            let setup = CaptureSetup {
                index: self.camera_index,
                camera_id: self.camera_id.clone(),
                pf: kind.pixel_format(),
                w,
                h,
                limit: self.frame_limit,
            };
            let controls = CamControls {
                fps,
                ae_enable: self.ae_enable,
                exposure_us: self.exposure_us,
                gain: self.gain,
                brightness: self.brightness,
                contrast: self.contrast,
                saturation: self.saturation,
            };

            // Bounded channel: the capture thread blocks once the pipeline is
            // BUFFER_COUNT frames behind, so memory stays bounded (backpressure).
            let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(BUFFER_COUNT);

            // All libcamera interaction lives on this thread: the objects are
            // thread-affine and completions arrive on a libcamera callback.
            let handle = std::thread::spawn(move || -> Result<(), G2gError> {
                capture_loop(setup, controls, tx)
            });

            let pts_step_ns = if fps > 0 {
                1_000_000_000 / fps as u64
            } else {
                0
            };
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
/// Everything the capture thread needs to open and configure the stream
/// (camera selection, pixel format, geometry, frame limit). Bundled so the
/// thread entry point stays within a sane argument count.
#[derive(Debug)]
struct CaptureSetup {
    index: usize,
    camera_id: Option<String>,
    pf: PixelFormat,
    w: u32,
    h: u32,
    limit: u64,
}

/// Camera-side controls applied at `start`: the frame-rate cap plus the
/// exposure / gain overrides. All `Copy`, so they cross to the capture thread.
#[derive(Debug, Clone, Copy)]
struct CamControls {
    fps: u32,
    ae_enable: Option<bool>,
    exposure_us: Option<i32>,
    gain: Option<f32>,
    brightness: Option<f32>,
    contrast: Option<f32>,
    saturation: Option<f32>,
}

fn capture_loop(
    setup: CaptureSetup,
    controls: CamControls,
    tx: tokio::sync::mpsc::Sender<Vec<u8>>,
) -> Result<(), G2gError> {
    let CaptureSetup {
        index,
        camera_id,
        pf,
        w,
        h,
        limit,
    } = setup;
    let mgr = CameraManager::new().map_err(|e| lc_err(&e))?;
    let cameras = mgr.cameras();
    let idx = resolve_camera_index(&cameras, index, camera_id.as_deref()).ok_or_else(lc_other)?;
    let cam = cameras.get(idx).ok_or_else(lc_other)?;
    let mut cam = cam.acquire().map_err(|e| lc_err(&e))?;

    let mut cfgs = cam
        .generate_configuration(&[StreamRole::ViewFinder])
        .ok_or_else(lc_other)?;
    {
        let mut cfg = cfgs.get_mut(0).ok_or_else(lc_other)?;
        cfg.set_pixel_format(pf);
        cfg.set_size(Size {
            width: w,
            height: h,
        });
    }
    if matches!(cfgs.validate(), CameraConfigurationStatus::Invalid) {
        return Err(G2gError::CapsMismatch);
    }
    cam.configure(&mut cfgs).map_err(|e| lc_err(&e))?;

    // The stream handle is owned once copied out (it points into `cfgs`, which
    // stays alive for the whole function); the `cfg` borrow ends here.
    let stream = cfgs
        .get(0)
        .ok_or_else(lc_other)?
        .stream()
        .ok_or_else(lc_other)?;

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

    // Build the start controls: the frame-rate cap plus any exposure / gain
    // overrides, each applied only if the camera advertises it (see
    // `set_if_supported`). With auto-exposure on (the default) the camera
    // lengthens exposure in low light and the frame rate collapses, so a manual
    // short exposure (AE off) is the real lever for a high rate in a dim room;
    // the FrameDurationLimits cap alone cannot beat the exposure-bound rate.
    let infos = cam.controls();
    let mut ctrls = ControlList::new();
    let mut any = false;

    if controls.fps > 0 {
        // Minimum frame duration (microseconds) bounds the *fastest* rate, so
        // it caps fps at the request; the maximum is left generous (a 1 fps
        // floor) so a camera that cannot sustain the requested rate falls back
        // to its own ceiling instead of collapsing on an exact min == max.
        let min_us = (1_000_000 / controls.fps as i64).max(1);
        let max_us = 1_000_000i64.max(min_us);
        any |= set_if_supported(&mut ctrls, infos, FrameDurationLimits([min_us, max_us]));
    }
    // A manual exposure or gain implies auto-exposure off (AE would otherwise
    // override the fixed values); an explicit `ae_enable` always wins.
    let ae = controls
        .ae_enable
        .or_else(|| (controls.exposure_us.is_some() || controls.gain.is_some()).then_some(false));
    if let Some(ae) = ae {
        any |= set_if_supported(&mut ctrls, infos, AeEnable(ae));
    }
    if let Some(us) = controls.exposure_us {
        any |= set_if_supported(&mut ctrls, infos, ExposureTime(us));
    }
    if let Some(g) = controls.gain {
        any |= set_if_supported(&mut ctrls, infos, AnalogueGain(g));
    }
    // Image adjustments: independent of exposure, so they brighten / tune a
    // short-exposure frame without touching the frame rate.
    if let Some(b) = controls.brightness {
        any |= set_if_supported(&mut ctrls, infos, Brightness(b));
    }
    if let Some(c) = controls.contrast {
        any |= set_if_supported(&mut ctrls, infos, Contrast(c));
    }
    if let Some(s) = controls.saturation {
        any |= set_if_supported(&mut ctrls, infos, Saturation(s));
    }
    let start_controls = any.then_some(ctrls);
    cam.start(start_controls.as_deref())
        .map_err(|e| lc_err(&e))?;
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
    PropertySpec::new(
        "camera",
        PropKind::Uint,
        "camera index (libcamera enumeration order)",
    ),
    PropertySpec::new(
        "camera-id",
        PropKind::Str,
        "select camera by id substring (stable across reboots)",
    ),
    PropertySpec::new("width", PropKind::Uint, "requested capture width in pixels"),
    PropertySpec::new(
        "height",
        PropKind::Uint,
        "requested capture height in pixels",
    ),
    PropertySpec::new(
        "framerate",
        PropKind::Uint,
        "capture frame rate (enforced via FrameDurationLimits)",
    ),
    PropertySpec::new(
        "format",
        PropKind::Str,
        "output: raw (NV12/YUYV) | mjpeg (pair with mjpegdec)",
    ),
    PropertySpec::new(
        "auto-exposure",
        PropKind::Bool,
        "auto-exposure on/off (off lifts the low-light fps cap)",
    ),
    PropertySpec::new(
        "exposure",
        PropKind::Int,
        "manual exposure time in microseconds (disables AE)",
    ),
    PropertySpec::new(
        "gain",
        PropKind::Double,
        "manual analogue gain (disables AE)",
    ),
    PropertySpec::new(
        "brightness",
        PropKind::Double,
        "image brightness [-1,1]; brightens without lowering fps",
    ),
    PropertySpec::new(
        "contrast",
        PropKind::Double,
        "image contrast (1.0 = default)",
    ),
    PropertySpec::new(
        "saturation",
        PropKind::Double,
        "colour saturation (1.0 = default, 0 = grey)",
    ),
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
        assert_eq!(
            src.get_property("format"),
            Some(PropValue::Str("raw".into()))
        );
        src.set_property("format", PropValue::Str("mjpeg".into()))
            .unwrap();
        assert_eq!(
            src.get_property("format"),
            Some(PropValue::Str("mjpeg".into()))
        );
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
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                ..
            }
        ));
        assert!(matches!(
            OutKind::Mjpeg.caps(1280, 720, 30),
            Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                ..
            }
        ));
    }

    #[test]
    fn with_mjpeg_sets_preference() {
        let src = LibCameraSrc::new().with_mjpeg(true);
        assert_eq!(
            src.get_property("format"),
            Some(PropValue::Str("mjpeg".into()))
        );
    }

    #[test]
    fn manual_exposure_implies_ae_off() {
        // A bare source reports no exposure overrides.
        let src = LibCameraSrc::new();
        assert_eq!(src.get_property("auto-exposure"), None);
        assert_eq!(src.get_property("exposure"), None);

        // Setting exposure or gain turns auto-exposure off automatically.
        let src = LibCameraSrc::new().with_exposure(8000);
        assert_eq!(src.get_property("exposure"), Some(PropValue::Int(8000)));
        assert_eq!(
            src.get_property("auto-exposure"),
            Some(PropValue::Bool(false))
        );

        let src = LibCameraSrc::new().with_gain(4.0);
        assert_eq!(src.get_property("gain"), Some(PropValue::Double(4.0)));
        assert_eq!(
            src.get_property("auto-exposure"),
            Some(PropValue::Bool(false))
        );

        // The property setters mirror the builders.
        let mut src = LibCameraSrc::new();
        src.set_property("exposure", PropValue::Int(5000)).unwrap();
        assert_eq!(
            src.get_property("auto-exposure"),
            Some(PropValue::Bool(false))
        );
        src.set_property("auto-exposure", PropValue::Bool(true))
            .unwrap();
        assert_eq!(
            src.get_property("auto-exposure"),
            Some(PropValue::Bool(true))
        );
        assert_eq!(
            src.set_property("gain", PropValue::Str("x".into())),
            Err(PropError::Type)
        );
    }

    #[test]
    fn image_adjustments_round_trip_without_touching_ae() {
        // Brightness / contrast / saturation are image adjustments, NOT
        // exposure, so they must not flip auto-exposure off.
        // Values exactly representable in both f32 and f64 (the setters narrow
        // to f32) so the round-trip compares cleanly.
        let src = LibCameraSrc::new()
            .with_brightness(0.5)
            .with_contrast(1.25)
            .with_saturation(0.0);
        assert_eq!(src.get_property("brightness"), Some(PropValue::Double(0.5)));
        assert_eq!(src.get_property("contrast"), Some(PropValue::Double(1.25)));
        assert_eq!(src.get_property("saturation"), Some(PropValue::Double(0.0)));
        assert_eq!(
            src.get_property("auto-exposure"),
            None,
            "brightness must not disable AE"
        );
    }

    #[test]
    fn camera_id_selects_by_name() {
        let src = LibCameraSrc::new();
        assert_eq!(src.get_property("camera-id"), None);
        let mut src = LibCameraSrc::new().with_camera_id("usb-1.2");
        assert_eq!(
            src.get_property("camera-id"),
            Some(PropValue::Str("usb-1.2".into()))
        );
        src.set_property("camera-id", PropValue::Str("front".into()))
            .unwrap();
        assert_eq!(
            src.get_property("camera-id"),
            Some(PropValue::Str("front".into()))
        );
    }
}

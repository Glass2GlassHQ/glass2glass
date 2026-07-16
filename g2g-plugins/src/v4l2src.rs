//! V4L2 capture source. Streams packed YUYV (4:2:2) frames off a UVC
//! `/dev/videoN` device via mmap streaming I/O. Linux-only (`v4l2` feature).
//!
//! Pipeline shape: `V4l2Src -> VideoConvert(Yuyv -> Nv12/I420/Rgba8) -> sink`.
//! YUYV is the near-universal UVC output; `VideoConvert` unpacks it (M89).
//!
//! V4L2's ioctls are blocking, so the capture loop runs on a dedicated std
//! thread that feeds the async `run` loop over a bounded channel. The format
//! is negotiated up front in [`intercept_caps`](V4l2Src::intercept_caps) (the
//! driver may adjust the requested geometry and frame rate), and the capture
//! thread re-opens the device under that exact format. Keeping the device out
//! of the struct between negotiation and `run` sidesteps `Send`/borrow
//! entanglement with the mmap stream, which borrows the device.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError,
    HardwareError, LatencyReport, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::prelude::{Device, MmapStream};
use v4l::video::capture::Parameters;
use v4l::video::Capture;
use v4l::{Format, FourCC};

/// Default capture geometry / rate used when the caller does not specify one.
const DEFAULT_WIDTH: u32 = 640;
const DEFAULT_HEIGHT: u32 = 480;
const DEFAULT_FPS: u32 = 30;
/// mmap buffer-ring depth requested from the driver. Doubles as the async
/// channel bound, so the capture thread blocks (backpressure) rather than
/// outrunning the pipeline.
const BUFFER_COUNT: u32 = 4;
/// The only fourcc we negotiate. UVC cameras universally support it.
const YUYV: &[u8; 4] = b"YUYV";

/// Map a V4L2 / OS error to the reserved `G2gError::V4l2` arm, preserving the
/// errno where one exists.
fn v4l2_err(e: &std::io::Error) -> G2gError {
    G2gError::Hardware(HardwareError::V4l2(e.raw_os_error().unwrap_or(-1)))
}

#[derive(Debug)]
pub struct V4l2Src {
    device: String,
    req_width: u32,
    req_height: u32,
    req_fps: u32,
    /// 0 means run until error or downstream shutdown; otherwise stop after
    /// this many frames and emit EOS (the test / bounded-capture path).
    frame_limit: u64,
    /// Driver-chosen `(width, height, fps)`, filled by `intercept_caps`. The
    /// driver may snap the request to a supported mode, so these are the real
    /// numbers the capture thread and the emitted caps use.
    negotiated: Option<(u32, u32, u32)>,
    configured: bool,
}

impl V4l2Src {
    /// Capture from `device` (e.g. `/dev/video0`) at the default 640x480 / 30.
    pub fn new(device: impl Into<String>) -> Self {
        Self {
            device: device.into(),
            req_width: DEFAULT_WIDTH,
            req_height: DEFAULT_HEIGHT,
            req_fps: DEFAULT_FPS,
            frame_limit: 0,
            negotiated: None,
            configured: false,
        }
    }

    /// Request a capture size. The driver may snap to the nearest supported
    /// mode; the negotiated caps reflect what it actually chose.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.req_width = width;
        self.req_height = height;
        self
    }

    /// Request a frame rate in fps. Best-effort: the driver may pick another.
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.req_fps = fps;
        self
    }

    /// Stop after `n` frames and emit EOS. Without this the source runs until
    /// an error or until downstream drops (no EOS on its own).
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// Open the device, set YUYV at the requested geometry, and read back what
    /// the driver actually chose. The probe device is dropped before `run`.
    fn negotiate(&mut self) -> Result<Caps, G2gError> {
        let dev = Device::with_path(&self.device).map_err(|e| v4l2_err(&e))?;
        let fmt = Format::new(self.req_width, self.req_height, FourCC::new(YUYV));
        let actual = dev.set_format(&fmt).map_err(|e| v4l2_err(&e))?;
        if &actual.fourcc.repr != YUYV {
            // The device cannot produce YUYV (it snapped to MJPEG or similar).
            // A format-flexible source / decode-through-MJPEG path is future
            // work; for now this is an unsupported configuration.
            return Err(G2gError::CapsMismatch);
        }
        // Frame rate is best-effort: many UVC cams ignore set_params for some
        // modes, so fall back to the request when the read-back is unusable.
        let fps = match dev.set_params(&Parameters::with_fps(self.req_fps)) {
            Ok(p) if p.interval.numerator > 0 => p.interval.denominator / p.interval.numerator,
            _ => self.req_fps,
        };
        self.negotiated = Some((actual.width, actual.height, fps));
        Ok(Caps::RawVideo {
            format: RawVideoFormat::Yuyv,
            width: Dim::Fixed(actual.width),
            height: Dim::Fixed(actual.height),
            framerate: Rate::Fixed(fps << 16),
        })
    }
}

impl SourceLoop for V4l2Src {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        // The probe ioctls are quick and synchronous; no need for an async body.
        core::future::ready(self.negotiate())
    }

    /// Produces the YUYV caps the driver settles on during the ioctl probe, so a
    /// chain built on the camera takes the native arc-consistency path. Mirrors
    /// `UdpSrc`; the probe is synchronous, so no async body is needed.
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
            "V4L2 camera source",
            "Source/Video",
            "Captures video from a V4L2 device (YUYV)",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("device", PropKind::Str, "V4L2 device node (e.g. /dev/video0)")
                .with_default("/dev/video0"),
            PropertySpec::new("width", PropKind::Uint, "requested capture width (driver may snap)"),
            PropertySpec::new("height", PropKind::Uint, "requested capture height (driver may snap)"),
            PropertySpec::new("framerate", PropKind::Uint, "requested capture rate, fps (best effort)"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "device" => {
                self.device = value.as_str().ok_or(PropError::Type)?.to_string();
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
            "device" => Some(PropValue::Str(self.device.clone())),
            "width" => Some(PropValue::Uint(self.req_width as u64)),
            "height" => Some(PropValue::Uint(self.req_height as u64)),
            "framerate" => Some(PropValue::Uint(self.req_fps as u64)),
            _ => None,
        }
    }

    /// Live source: contributes one frame period of latency so the sink keeps a
    /// frame in hand and never runs dry waiting on capture.
    fn latency(&self) -> LatencyReport {
        let fps = self.negotiated.map(|(_, _, f)| f).unwrap_or(self.req_fps);
        let period_ns = if fps > 0 { 1_000_000_000 / fps as u64 } else { 0 };
        LatencyReport::live(period_ns, None)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let (w, h, fps) = self.negotiated.ok_or(G2gError::NotConfigured)?;
            let limit = self.frame_limit;
            let device = self.device.clone();
            let expected = (w as usize) * (h as usize) * 2;

            // Bounded channel: the capture thread blocks once the pipeline is
            // BUFFER_COUNT frames behind, so we don't grow memory unboundedly.
            let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(BUFFER_COUNT as usize);

            // Blocking V4L2 capture on its own thread. It owns the device and
            // the mmap stream (which borrows the device), copies each frame's
            // payload out of the mmap buffer, and hands it to the async side.
            let handle = std::thread::spawn(move || -> Result<(), G2gError> {
                let dev = Device::with_path(&device).map_err(|e| v4l2_err(&e))?;
                dev.set_format(&Format::new(w, h, FourCC::new(YUYV)))
                    .map_err(|e| v4l2_err(&e))?;
                let _ = dev.set_params(&Parameters::with_fps(fps));
                let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, BUFFER_COUNT)
                    .map_err(|e| v4l2_err(&e))?;

                let mut count = 0u64;
                while limit == 0 || count < limit {
                    let (buf, meta) = stream.next().map_err(|e| v4l2_err(&e))?;
                    let n = (meta.bytesused as usize).min(buf.len());
                    let mut payload = Vec::with_capacity(n);
                    payload.extend_from_slice(&buf[..n]);
                    // Err means the receiver was dropped (pipeline shut down).
                    if tx.blocking_send(payload).is_err() {
                        break;
                    }
                    count += 1;
                }
                Ok(())
            });

            let pts_step_ns = if fps > 0 { 1_000_000_000 / fps as u64 } else { 0 };
            let mut seq = 0u64;
            while let Some(bytes) = rx.recv().await {
                // A short frame (driver hiccup) can't be unpacked safely; skip
                // it rather than push a malformed buffer downstream.
                if bytes.len() < expected {
                    continue;
                }
                // Source-side wall-clock stamp for glass-to-glass latency, same
                // convention as VideoTestSrc / RtspSrc.
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
            let thread_result = handle
                .join()
                .unwrap_or(Err(G2gError::Hardware(HardwareError::V4l2(-1))));
            if seq == 0 {
                thread_result?;
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

impl PadTemplates for V4l2Src {
    /// Always produces packed YUYV; a constructed instance fixes the geometry
    /// and rate during `intercept_caps`.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(g2g_core::CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Yuyv,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_set_requested_config() {
        let src = V4l2Src::new("/dev/video0")
            .with_size(1280, 720)
            .with_fps(60)
            .with_frame_limit(10);
        assert_eq!(src.device, "/dev/video0");
        assert_eq!((src.req_width, src.req_height, src.req_fps), (1280, 720, 60));
        assert_eq!(src.frame_limit, 10);
    }

    #[test]
    fn run_before_negotiation_is_not_configured() {
        // configure_pipeline must reject when intercept_caps never ran, so the
        // capture thread is never spawned against an un-negotiated device.
        let mut src = V4l2Src::new("/dev/video0");
        let err = src
            .configure_pipeline(&Caps::RawVideo {
                format: RawVideoFormat::Yuyv,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Fixed(30 << 16),
            })
            .expect_err("configure without negotiate must fail");
        assert_eq!(err, G2gError::NotConfigured);
    }
}

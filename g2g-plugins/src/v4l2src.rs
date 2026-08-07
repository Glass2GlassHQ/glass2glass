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
use v4l::control::{Control, Value as ControlValue};
use v4l::io::traits::CaptureStream;
use v4l::prelude::{Device, MmapStream};
use v4l::v4l_sys::{
    V4L2_CID_AUTO_WHITE_BALANCE, V4L2_CID_EXPOSURE_ABSOLUTE, V4L2_CID_EXPOSURE_AUTO,
    V4L2_CID_FOCUS_ABSOLUTE, V4L2_CID_FOCUS_AUTO, V4L2_CID_WHITE_BALANCE_TEMPERATURE,
};
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

/// ENODEV, reported when `device-id` names a camera that is not attached.
const ENODEV: i32 = 19;

/// How a camera control's value is carried, which decides both the property
/// kind and the V4L2 control payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlKind {
    Switch,
    Amount,
}

impl ControlKind {
    const fn prop_kind(self) -> PropKind {
        match self {
            ControlKind::Switch => PropKind::Bool,
            ControlKind::Amount => PropKind::Uint,
        }
    }
}

/// One camera control this element drives, exposed under the name `v4l2-ctl`
/// uses for it.
#[derive(Debug, Clone, Copy)]
struct ControlSpec {
    name: &'static str,
    id: u32,
    kind: ControlKind,
    blurb: &'static str,
}

/// The controls, in the order they are applied: an auto switch comes before
/// the manual value it gates, because a driver rejects a manual exposure or
/// focus while its automatic mode is still on.
///
/// `exposure-auto` is a menu, not a switch: 0 auto, 1 manual, 2 shutter
/// priority, 3 aperture priority (`V4L2_EXPOSURE_*`), and most UVC cameras
/// implement only 1 and 3.
const CONTROLS: [ControlSpec; 6] = [
    ControlSpec {
        name: "white-balance-temperature-auto",
        id: V4L2_CID_AUTO_WHITE_BALANCE,
        kind: ControlKind::Switch,
        blurb: "automatic white balance",
    },
    ControlSpec {
        name: "exposure-auto",
        id: V4L2_CID_EXPOSURE_AUTO,
        kind: ControlKind::Amount,
        blurb: "exposure mode: 0 auto, 1 manual, 2 shutter priority, 3 aperture priority",
    },
    ControlSpec {
        name: "focus-auto",
        id: V4L2_CID_FOCUS_AUTO,
        kind: ControlKind::Switch,
        blurb: "continuous auto focus",
    },
    ControlSpec {
        name: "white-balance-temperature",
        id: V4L2_CID_WHITE_BALANCE_TEMPERATURE,
        kind: ControlKind::Amount,
        blurb: "white balance colour temperature, kelvin (needs auto off)",
    },
    ControlSpec {
        name: "exposure-absolute",
        id: V4L2_CID_EXPOSURE_ABSOLUTE,
        kind: ControlKind::Amount,
        blurb: "exposure time in 100 us units (needs exposure-auto=1)",
    },
    ControlSpec {
        name: "focus-absolute",
        id: V4L2_CID_FOCUS_ABSOLUTE,
        kind: ControlKind::Amount,
        blurb: "focus distance, driver-defined units (needs focus-auto=false)",
    },
];

/// The property spec of control `index`, so the table above is the only place
/// a control's name / kind / description is written.
const fn control_prop(index: usize) -> PropertySpec {
    PropertySpec::new(
        CONTROLS[index].name,
        CONTROLS[index].kind.prop_kind(),
        CONTROLS[index].blurb,
    )
}

/// The values set on this element, `None` for a control left alone. Indexed
/// like [`CONTROLS`].
type ControlValues = [Option<i64>; CONTROLS.len()];

/// Apply every set control in table order, one ioctl each (`set_controls`
/// refuses a batch spanning two control classes, and these span the user and
/// camera classes). A driver that rejects one fails the negotiation: the
/// caller asked for a control this camera does not have.
fn apply_controls(dev: &Device, values: &ControlValues) -> Result<(), G2gError> {
    for (spec, value) in CONTROLS.iter().zip(values) {
        let Some(value) = *value else { continue };
        let value = match spec.kind {
            ControlKind::Switch => ControlValue::Boolean(value != 0),
            ControlKind::Amount => ControlValue::Integer(value),
        };
        dev.set_control(Control { id: spec.id, value })
            .map_err(|e| v4l2_err(&e))?;
    }
    Ok(())
}

/// One completed capture handed to the async loop: the payload bytes plus the
/// driver's buffer timestamp in nanoseconds (`None` when the driver left it at
/// zero, which is how an absent timestamp shows up).
#[derive(Debug)]
struct Captured {
    bytes: Vec<u8>,
    timestamp_ns: Option<u64>,
}

/// The driver's `timeval` capture time as nanoseconds. All-zero means the
/// driver never stamped the buffer.
fn buffer_timestamp_ns(ts: &v4l::timestamp::Timestamp) -> Option<u64> {
    if ts.sec == 0 && ts.usec == 0 {
        return None;
    }
    let sec = u64::try_from(ts.sec).ok()?;
    let usec = u64::try_from(ts.usec).ok()?;
    Some(
        sec.saturating_mul(1_000_000_000)
            .saturating_add(usec.saturating_mul(1_000)),
    )
}

#[derive(Debug)]
pub struct V4l2Src {
    device: String,
    /// Persistent id from the device monitor; when set it decides `device` at
    /// negotiation instead of the node path, so a saved pipeline survives a
    /// replug that renumbered `/dev/videoN`.
    device_id: String,
    device_resolved: bool,
    controls: ControlValues,
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
            device_id: String::new(),
            device_resolved: false,
            controls: [None; CONTROLS.len()],
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

    /// Request a frame rate in fps. Best-effort: the driver may pick another,
    /// and many UVC cams free-run below the one they accepted. PTS comes from
    /// the driver's buffer timestamps, so it holds either way; this rate only
    /// feeds the advertised caps, the latency report, and the fallback period.
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

    /// Select the camera by the persistent id the device monitor reports for
    /// it, resolved to a node path at negotiation. Overrides
    /// [`new`](Self::new)'s path.
    pub fn with_device_id(mut self, id: impl Into<String>) -> Self {
        self.device_id = id.into();
        self.device_resolved = false;
        self
    }

    /// Point `device` at whatever node carries `device-id` now. Once per
    /// instance: the resolution is a full enumeration, and negotiation runs
    /// more than once.
    fn resolve_device(&mut self) -> Result<(), G2gError> {
        if self.device_id.is_empty() || self.device_resolved {
            return Ok(());
        }
        let path = crate::v4l2device::resolve_device_id(&self.device_id)
            .ok_or(G2gError::Hardware(HardwareError::V4l2(ENODEV)))?;
        self.device = path;
        self.device_resolved = true;
        Ok(())
    }

    /// Open the device, set YUYV at the requested geometry, and read back what
    /// the driver actually chose. The probe device is dropped before `run`.
    fn negotiate(&mut self) -> Result<Caps, G2gError> {
        self.resolve_device()?;
        let dev = Device::with_path(&self.device).map_err(|e| v4l2_err(&e))?;
        // Controls are device state, not per-fd, so applying them on the probe
        // handle holds for the capture thread's own open.
        apply_controls(&dev, &self.controls)?;
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
            interlace: g2g_core::Interlace::Any,
        })
    }
}

impl SourceLoop for V4l2Src {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
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
            "V4L2 camera source",
            "Source/Video",
            "Captures video from a V4L2 device (YUYV)",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "device",
                PropKind::Str,
                "V4L2 device node (e.g. /dev/video0)",
            )
            .with_default("/dev/video0"),
            PropertySpec::new(
                "device-id",
                PropKind::Str,
                "persistent device id from the device monitor; resolved to a node path at \
                 negotiation, so a saved pipeline survives a replug",
            ),
            PropertySpec::new(
                "width",
                PropKind::Uint,
                "requested capture width (driver may snap)",
            ),
            PropertySpec::new(
                "height",
                PropKind::Uint,
                "requested capture height (driver may snap)",
            ),
            PropertySpec::new(
                "framerate",
                PropKind::Uint,
                "requested capture rate, fps (best effort)",
            ),
            control_prop(0),
            control_prop(1),
            control_prop(2),
            control_prop(3),
            control_prop(4),
            control_prop(5),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(index) = CONTROLS.iter().position(|c| c.name == name) {
            self.controls[index] = Some(match CONTROLS[index].kind {
                ControlKind::Switch => i64::from(value.as_bool().ok_or(PropError::Type)?),
                ControlKind::Amount => i64::try_from(value.as_uint().ok_or(PropError::Type)?)
                    .map_err(|_| PropError::Value)?,
            });
            return Ok(());
        }
        match name {
            "device" => {
                self.device = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "device-id" => {
                self.device_id = value.as_str().ok_or(PropError::Type)?.to_string();
                self.device_resolved = false;
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
        if let Some(index) = CONTROLS.iter().position(|c| c.name == name) {
            let value = self.controls[index]?;
            return Some(match CONTROLS[index].kind {
                ControlKind::Switch => PropValue::Bool(value != 0),
                ControlKind::Amount => PropValue::Uint(value as u64),
            });
        }
        match name {
            "device" => Some(PropValue::Str(self.device.clone())),
            "device-id" => Some(PropValue::Str(self.device_id.clone())),
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
        let period_ns = if fps > 0 {
            1_000_000_000 / fps as u64
        } else {
            0
        };
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
            let (tx, mut rx) = tokio::sync::mpsc::channel::<Captured>(BUFFER_COUNT as usize);

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

                loop {
                    let (buf, meta) = stream.next().map_err(|e| v4l2_err(&e))?;
                    let n = (meta.bytesused as usize).min(buf.len());
                    let mut payload = Vec::with_capacity(n);
                    payload.extend_from_slice(&buf[..n]);
                    let captured = Captured {
                        bytes: payload,
                        timestamp_ns: buffer_timestamp_ns(&meta.timestamp),
                    };
                    // Err means the receiver was dropped (limit reached or the
                    // pipeline shut down).
                    if tx.blocking_send(captured).is_err() {
                        break;
                    }
                }
                Ok(())
            });

            let pts_step_ns = if fps > 0 {
                1_000_000_000 / fps as u64
            } else {
                0
            };
            // The driver stamps every buffer with the time of capture, so the
            // PTS tracks the rate the camera actually held rather than the one
            // that was asked for. The two differ whenever the camera cannot
            // hold the request (auto-exposure lengthens the frame duration in
            // low light), and stamping the request there would compress the
            // timeline: a recording would play back faster than it was shot.
            let mut epoch_ns: Option<u64> = None;
            let mut prev_pts = 0u64;
            let mut seq = 0u64;
            while let Some(captured) = rx.recv().await {
                // A short frame (driver hiccup) can't be unpacked safely; skip
                // it rather than push a malformed buffer downstream.
                if captured.bytes.len() < expected {
                    continue;
                }
                // Source-side wall-clock stamp for glass-to-glass latency, same
                // convention as VideoTestSrc / RtspSrc.
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                let pts = match captured.timestamp_ns {
                    Some(ts) => ts.saturating_sub(*epoch_ns.get_or_insert(ts)),
                    None => seq * pts_step_ns,
                };
                // The gap the previous frame occupied. The nominal period
                // covers the first frame and any repeated timestamp.
                let duration_ns = match pts.checked_sub(prev_pts) {
                    Some(d) if seq > 0 && d > 0 => d,
                    _ => pts_step_ns,
                };
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        captured.bytes.into_boxed_slice(),
                    )),
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns,
                        capture_ns: pts,
                        arrival_ns,
                        keyframe: true, // raw frames are each independently presentable
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
                prev_pts = pts;
                seq += 1;
                // The limit counts emitted frames, so a skipped short buffer
                // is captured again rather than lost from the count.
                if limit > 0 && seq >= limit {
                    break;
                }
            }

            // Drop the receiver first: a capture thread blocked in send must
            // fail out of it before the join below can succeed.
            drop(rx);

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
        Vec::from([PadTemplate::source(g2g_core::CapsSet::one(
            Caps::RawVideo {
                format: RawVideoFormat::Yuyv,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
            },
        ))])
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
        assert_eq!(
            (src.req_width, src.req_height, src.req_fps),
            (1280, 720, 60)
        );
        assert_eq!(src.frame_limit, 10);
    }

    #[test]
    fn every_control_is_a_declared_property() {
        let src = V4l2Src::new("/dev/video0");
        let specs = SourceLoop::properties(&src);
        for spec in CONTROLS {
            let declared = specs
                .iter()
                .find(|p| p.name == spec.name)
                .unwrap_or_else(|| panic!("{} is not declared", spec.name));
            assert_eq!(declared.kind, spec.kind.prop_kind());
        }
        // ids must be distinct, or one control silently overwrites another.
        for (i, a) in CONTROLS.iter().enumerate() {
            assert!(!CONTROLS[..i].iter().any(|b| b.id == a.id), "{}", a.name);
        }
    }

    #[test]
    fn control_properties_round_trip_through_their_kind() {
        let mut src = V4l2Src::new("/dev/video0");
        // unset controls report nothing, so nothing is applied to the driver.
        assert!(SourceLoop::get_property(&src, "exposure-absolute").is_none());
        assert!(src.controls.iter().all(Option::is_none));

        SourceLoop::set_property(&mut src, "exposure-auto", PropValue::Uint(1)).expect("menu");
        SourceLoop::set_property(&mut src, "exposure-absolute", PropValue::Uint(250))
            .expect("amount");
        SourceLoop::set_property(&mut src, "focus-auto", PropValue::Bool(false)).expect("switch");
        assert_eq!(
            SourceLoop::get_property(&src, "exposure-absolute"),
            Some(PropValue::Uint(250))
        );
        assert_eq!(
            SourceLoop::get_property(&src, "focus-auto"),
            Some(PropValue::Bool(false))
        );
        // a switch set from a number (or the reverse) is a type error, not a
        // silently coerced control value.
        assert_eq!(
            SourceLoop::set_property(&mut src, "focus-auto", PropValue::Uint(1)),
            Err(PropError::Type)
        );

        // the auto switches precede the manual values they gate.
        let order: Vec<&str> = CONTROLS.iter().map(|c| c.name).collect();
        for (auto, manual) in [
            ("exposure-auto", "exposure-absolute"),
            ("focus-auto", "focus-absolute"),
            (
                "white-balance-temperature-auto",
                "white-balance-temperature",
            ),
        ] {
            let auto_at = order.iter().position(|n| *n == auto).expect("auto");
            let manual_at = order.iter().position(|n| *n == manual).expect("manual");
            assert!(auto_at < manual_at, "{auto} must precede {manual}");
        }
    }

    #[test]
    fn device_id_overrides_the_node_path_and_fails_loud_when_absent() {
        let mut src =
            V4l2Src::new("/dev/video0").with_device_id("no-such-bus:No Camera:/dev/video9");
        assert_eq!(
            SourceLoop::get_property(&src, "device-id"),
            Some(PropValue::Str(
                "no-such-bus:No Camera:/dev/video9".to_string()
            ))
        );
        // an id nothing carries must not silently fall back to `device`.
        assert_eq!(
            src.resolve_device(),
            Err(G2gError::Hardware(HardwareError::V4l2(ENODEV)))
        );
        assert_eq!(src.device, "/dev/video0");

        // set through the property path, an attached camera's own id resolves
        // to its node. Skipped where no camera is present (CI).
        use g2g_core::runtime::DeviceProvider;
        let devices = crate::v4l2device::V4l2DeviceProvider::new()
            .probe()
            .unwrap_or_default();
        let Some(first) = devices.first() else { return };
        let mut src = V4l2Src::new("/dev/null");
        SourceLoop::set_property(
            &mut src,
            "device-id",
            PropValue::Str(first.persistent_id.clone()),
        )
        .expect("device-id");
        src.resolve_device().expect("resolve an attached camera");
        assert_eq!(
            SourceLoop::get_property(&src, "device"),
            Some(PropValue::Str(first.props[0].1.clone()))
        );
    }

    /// The whole M944 path against a real camera: an id the provider minted
    /// selects the node, and a control set through a property reaches the
    /// driver. Self-skips where no camera (CI) or none with a switch control.
    #[test]
    fn live_camera_resolves_its_id_and_applies_a_control() {
        use g2g_core::runtime::DeviceProvider;
        let devices = crate::v4l2device::V4l2DeviceProvider::new()
            .probe()
            .unwrap_or_default();
        let Some(camera) = devices.first() else {
            return;
        };
        let path = camera.props[0].1.clone();
        let dev = Device::with_path(&path).expect("open the probed camera");

        // the first switch control this camera reports; its current value is
        // restored at the end, so the test leaves the camera as it found it.
        let Some((spec, before)) = CONTROLS
            .iter()
            .filter(|c| c.kind == ControlKind::Switch)
            .find_map(|c| match dev.control(c.id) {
                Ok(Control {
                    value: ControlValue::Boolean(on),
                    ..
                }) => Some((c, on)),
                _ => None,
            })
        else {
            return;
        };

        let mut src = V4l2Src::new("/dev/null");
        SourceLoop::set_property(
            &mut src,
            "device-id",
            PropValue::Str(camera.persistent_id.clone()),
        )
        .expect("device-id");
        SourceLoop::set_property(&mut src, spec.name, PropValue::Bool(!before)).expect("control");
        let caps = g2g_core::runtime::block_on(SourceLoop::intercept_caps(&mut src))
            .expect("negotiate the resolved camera");
        assert!(matches!(caps, Caps::RawVideo { .. }));
        assert_eq!(
            SourceLoop::get_property(&src, "device"),
            Some(PropValue::Str(path.clone()))
        );
        assert_eq!(
            dev.control(spec.id).expect("read back").value,
            ControlValue::Boolean(!before),
            "{} did not reach the driver",
            spec.name
        );

        dev.set_control(Control {
            id: spec.id,
            value: ControlValue::Boolean(before),
        })
        .expect("restore");
    }

    #[test]
    fn buffer_timestamp_converts_and_detects_absence() {
        use v4l::timestamp::Timestamp;
        assert_eq!(
            buffer_timestamp_ns(&Timestamp::new(12, 500_000)),
            Some(12_500_000_000)
        );
        // An unstamped buffer must not be mistaken for the epoch.
        assert_eq!(buffer_timestamp_ns(&Timestamp::new(0, 0)), None);
        // Microseconds alone are a valid stamp right after the monotonic epoch.
        assert_eq!(buffer_timestamp_ns(&Timestamp::new(0, 1)), Some(1_000));
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
                interlace: g2g_core::Interlace::Any,
            })
            .expect_err("configure without negotiate must fail");
        assert_eq!(err, G2gError::NotConfigured);
    }
}

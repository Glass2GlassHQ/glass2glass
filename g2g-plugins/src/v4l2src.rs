//! V4L2 capture source. Streams frames off a `/dev/videoN` device via mmap
//! streaming I/O. Linux-only (`v4l2` feature).
//!
//! The device decides which formats exist and negotiation decides which one is
//! used: the probe enumerates the device's pixel formats, advertises every one
//! [`CapturePixelFormat`] covers, and the capture thread runs whichever the
//! solver fixed. Packed YUYV comes first in that set, so a chain that accepts
//! anything (`V4l2Src -> VideoConvert -> sink`, M89) still gets the raw UVC
//! output it always did. Pinning the link to `image/jpeg` instead selects the
//! camera's MJPEG mode, which fits resolutions and frame rates over USB that
//! uncompressed YUYV cannot; `MjpegDec` decodes it downstream.
//!
//! V4L2's ioctls are blocking, so the capture loop runs on a dedicated std
//! thread that feeds the async `run` loop over a bounded channel. The formats
//! are probed in [`intercept_caps`](V4l2Src::intercept_caps) (the driver may
//! adjust the requested geometry and frame rate per format), and the capture
//! thread re-opens the device under the negotiated one. Keeping the device out
//! of the struct between negotiation and `run` sidesteps `Send`/borrow
//! entanglement with the mmap stream, which borrows the device.
//!
//! `io-mode=dmabuf` replaces that copy with an export: the driver's MMAP buffers
//! are exported once (`VIDIOC_EXPBUF`) as dma-buf fds and each frame carries a
//! share of the fd its buffer was filled into, so a GPU consumer imports the
//! camera buffer directly. The buffer is the frame, so it goes back to the
//! driver only once every share of it has dropped; see [`ExportedQueue`].

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::{OwnedDmaBuf, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError,
    HardwareError, LatencyReport, MemoryDomain, MemoryDomainKind, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate,
};

use crate::capturepixelformat::CapturePixelFormat;

use std::sync::Arc;

use v4l::buffer::Type;
use v4l::control::{Control, Value as ControlValue};
use v4l::device::Handle;
use v4l::io::traits::CaptureStream;
use v4l::memory::Memory;
use v4l::prelude::{Device, MmapStream};
use v4l::timestamp::Timestamp;
use v4l::v4l2::vidioc;
use v4l::v4l_sys::{
    v4l2_buffer, v4l2_exportbuffer, v4l2_requestbuffers, V4L2_CID_AUTO_WHITE_BALANCE,
    V4L2_CID_EXPOSURE_ABSOLUTE, V4L2_CID_EXPOSURE_AUTO, V4L2_CID_FOCUS_ABSOLUTE,
    V4L2_CID_FOCUS_AUTO, V4L2_CID_WHITE_BALANCE_TEMPERATURE,
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
/// Frames the dmabuf path lets sit in the channel. A dmabuf frame *is* the
/// driver's buffer until it drops, so this stays below [`BUFFER_COUNT`]: the
/// queued channel plus the one frame the pipeline is working on still leaves the
/// driver a buffer to fill.
const DMABUF_INFLIGHT: usize = BUFFER_COUNT as usize - 2;
/// How long the dmabuf capture loop waits before re-checking for a released
/// buffer when every one of them is still downstream. Well under a frame period,
/// so a release is picked up promptly.
const DMABUF_RELEASE_POLL: core::time::Duration = core::time::Duration::from_millis(1);

/// The V4L2 fourccs this element carries, in the order they are advertised.
/// Packed YUYV leads: it is the near-universal UVC raw output, and a chain that
/// pins nothing takes the first alternative. MJPEG comes last because it needs
/// a decoder downstream. A device format missing from this table is skipped,
/// since there is no `Caps` to carry it on.
const FOURCCS: [(&[u8; 4], CapturePixelFormat); 4] = [
    (b"YUYV", CapturePixelFormat::Yuyv),
    (b"NV12", CapturePixelFormat::Nv12),
    (b"YU12", CapturePixelFormat::I420),
    (b"MJPG", CapturePixelFormat::Mjpeg),
];

/// The format a V4L2 fourcc carries, `None` for one no `Caps` covers.
pub fn format_for_fourcc(fourcc: &[u8; 4]) -> Option<CapturePixelFormat> {
    FOURCCS
        .iter()
        .find(|(code, _)| *code == fourcc)
        .map(|(_, format)| *format)
}

/// How a captured buffer leaves the element, the `io-mode` property's values.
/// `Auto` and `Mmap` both copy the mmap'd buffer into system memory (`Auto` is
/// what a driver-chosen default would pick, and MMAP is the only streaming
/// method this element implements).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum IoMode {
    #[default]
    Auto,
    Mmap,
    /// Export the MMAP buffers as dma-buf fds and emit frames in
    /// [`MemoryDomain::DmaBuf`], no copy.
    DmaBuf,
}

/// The `io-mode` values this element accepts, and the only place their names are
/// written. The V4L2 methods it does not implement (`rw`, `userptr`,
/// `dmabuf-import`) are absent, so asking for one is refused.
const IO_MODES: [(&str, IoMode); 3] = [
    ("auto", IoMode::Auto),
    ("mmap", IoMode::Mmap),
    ("dmabuf", IoMode::DmaBuf),
];

impl IoMode {
    fn name(self) -> &'static str {
        IO_MODES
            .iter()
            .find(|(_, mode)| *mode == self)
            .map(|(name, _)| *name)
            .unwrap_or("auto")
    }

    /// Whether a capture format can be carried in this mode. An exported dma-buf
    /// describes no payload length, so a compressed format's variable-length
    /// access unit (MJPEG) cannot be read out of the fd alone: only raw formats
    /// are offered for dmabuf export.
    fn carries(self, format: CapturePixelFormat) -> bool {
        self != IoMode::DmaBuf || format.raw_format().is_some()
    }
}

/// One capture mode the device confirmed: the fourcc to set plus the geometry
/// and rate the driver reported back for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Mode {
    fourcc: [u8; 4],
    format: CapturePixelFormat,
    width: u32,
    height: u32,
    fps: u32,
}

impl Mode {
    fn caps(&self) -> Caps {
        self.format.caps(self.width, self.height, self.fps)
    }
}

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

/// What a completed capture carries: bytes copied out of the mmap'd buffer, or a
/// share of the dma-buf fd the driver filled (no copy).
#[derive(Debug)]
enum CapturedPayload {
    Bytes(Vec<u8>),
    DmaBuf(OwnedDmaBuf),
}

/// One completed capture handed to the async loop: the payload plus the driver's
/// buffer timestamp in nanoseconds (`None` when the driver left it at zero, which
/// is how an absent timestamp shows up).
#[derive(Debug)]
struct Captured {
    payload: CapturedPayload,
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

/// One ioctl on the capture fd, with the errno kept in the reserved V4L2 error
/// arm.
///
/// # Safety
/// `arg` must be the argument struct `request` is defined over (the
/// `vidioc::VIDIOC_*` constant names it), since the kernel reads and writes it
/// through the pointer.
unsafe fn capture_ioctl<T>(
    fd: core::ffi::c_int,
    request: vidioc::_IOC_TYPE,
    arg: &mut T,
) -> Result<(), G2gError> {
    // SAFETY: the argument type is this function's contract, and `arg` is a live
    // exclusive borrow for the whole call.
    unsafe { v4l::v4l2::ioctl(fd, request, arg as *mut T as *mut core::ffi::c_void) }
        .map_err(|e| v4l2_err(&e))
}

/// A zeroed V4L2 ioctl argument. Every one of these is plain kernel-ABI data
/// whose reserved fields must be zero.
fn zeroed_arg<T>() -> T {
    // SAFETY: the V4L2 argument structs are `repr(C)` plain data (integers and
    // unions of integers) for which all-zero is the "nothing requested" value.
    unsafe { core::mem::zeroed() }
}

/// A capture-buffer descriptor for the MMAP queue, pre-filled for `index`.
fn mmap_buffer_arg(index: u32) -> v4l2_buffer {
    let mut buf: v4l2_buffer = zeroed_arg();
    buf.type_ = Type::VideoCapture as u32;
    buf.memory = Memory::Mmap as u32;
    buf.index = index;
    buf
}

/// The driver's MMAP buffers, exported once as dma-buf fds and handed downstream
/// without a copy.
///
/// The invariant this type exists to hold: a dequeued buffer is the frame, so it
/// must not go back to the driver while any share of its fd is alive. The element
/// keeps its own share of every buffer for the whole stream (so an fd is never
/// closed mid-stream and `VIDIOC_EXPBUF` runs once per buffer, not once per
/// frame), and [`refill`](Self::refill) re-queues exactly those buffers whose
/// share count is back to that one reference. With all [`BUFFER_COUNT`] buffers
/// held downstream there is nothing to dequeue and the capture loop waits, which
/// is backpressure rather than a deadlock: [`DMABUF_INFLIGHT`] bounds the
/// pipeline below `BUFFER_COUNT` so it cannot hold them all.
struct ExportedQueue {
    /// The device handle the ioctls go to, kept so the fd outlives the queue.
    handle: Arc<Handle>,
    /// The element's own share of each exported buffer, indexed by V4L2 buffer
    /// index.
    buffers: Vec<OwnedDmaBuf>,
    /// Whether the driver currently owns buffer `i`.
    queued: Vec<bool>,
    streaming: bool,
}

/// Hand-written because the `v4l` crate's device `Handle` is not `Debug`.
impl core::fmt::Debug for ExportedQueue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExportedQueue")
            .field("fd", &self.handle.fd())
            .field("buffers", &self.buffers)
            .field("queued", &self.queued)
            .field("streaming", &self.streaming)
            .finish()
    }
}

impl ExportedQueue {
    /// Allocate the MMAP buffers and export each one as a dma-buf fd. `stride` is
    /// the driver's `bytesperline` for the negotiated format, carried on every
    /// exported buffer so a consumer can address rows.
    fn new(dev: &Device, stride: u32) -> Result<Self, G2gError> {
        let handle = dev.handle();
        let fd = handle.fd();

        let mut request: v4l2_requestbuffers = zeroed_arg();
        request.count = BUFFER_COUNT;
        request.type_ = Type::VideoCapture as u32;
        request.memory = Memory::Mmap as u32;
        // SAFETY: `v4l2_requestbuffers` is VIDIOC_REQBUFS's argument type.
        unsafe { capture_ioctl(fd, vidioc::VIDIOC_REQBUFS, &mut request) }?;
        // A driver may grant fewer buffers than asked for, but none at all leaves
        // nothing to capture into.
        if request.count == 0 {
            return Err(G2gError::Hardware(HardwareError::V4l2(-1)));
        }

        let mut buffers = Vec::with_capacity(request.count as usize);
        for index in 0..request.count {
            let mut export: v4l2_exportbuffer = zeroed_arg();
            export.type_ = Type::VideoCapture as u32;
            export.index = index;
            // SAFETY: `v4l2_exportbuffer` is VIDIOC_EXPBUF's argument type.
            unsafe { capture_ioctl(fd, vidioc::VIDIOC_EXPBUF, &mut export) }?;
            if export.fd < 0 {
                return Err(G2gError::Hardware(HardwareError::V4l2(-1)));
            }
            // SAFETY: VIDIOC_EXPBUF just minted this fd for this process and
            // nothing else owns it; `OwnedDmaBuf` closes it on the last share.
            buffers.push(unsafe { OwnedDmaBuf::from_raw(export.fd, stride, 0) });
        }

        let queued = buffers.iter().map(|_| false).collect();
        Ok(Self {
            handle,
            buffers,
            queued,
            streaming: false,
        })
    }

    fn queue(&mut self, index: usize) -> Result<(), G2gError> {
        let mut buf = mmap_buffer_arg(index as u32);
        // SAFETY: `v4l2_buffer` is VIDIOC_QBUF's argument type.
        unsafe { capture_ioctl(self.handle.fd(), vidioc::VIDIOC_QBUF, &mut buf) }?;
        self.queued[index] = true;
        Ok(())
    }

    /// Queue every buffer and start streaming.
    fn start(&mut self) -> Result<(), G2gError> {
        for index in 0..self.buffers.len() {
            self.queue(index)?;
        }
        let mut buf_type = Type::VideoCapture as core::ffi::c_int;
        // SAFETY: VIDIOC_STREAMON takes a buffer-type int.
        unsafe { capture_ioctl(self.handle.fd(), vidioc::VIDIOC_STREAMON, &mut buf_type) }?;
        self.streaming = true;
        Ok(())
    }

    /// Hand back every buffer whose frame downstream has dropped, and report how
    /// many the driver now holds. Zero means every buffer is still in flight, so
    /// there is nothing to dequeue until a consumer releases one.
    fn refill(&mut self) -> Result<usize, G2gError> {
        for index in 0..self.buffers.len() {
            if !self.queued[index] && self.buffers[index].share_count() == 1 {
                self.queue(index)?;
            }
        }
        Ok(self.queued.iter().filter(|queued| **queued).count())
    }

    /// Block until the driver has filled a buffer, then take that buffer out of
    /// the queue. Returns its index, the bytes the driver wrote, and its capture
    /// timestamp.
    fn dequeue(&mut self) -> Result<(usize, u32, Option<u64>), G2gError> {
        self.handle
            .poll(libc::POLLIN, -1)
            .map_err(|e| v4l2_err(&e))?;
        let mut buf = mmap_buffer_arg(0);
        // SAFETY: `v4l2_buffer` is VIDIOC_DQBUF's argument type.
        unsafe { capture_ioctl(self.handle.fd(), vidioc::VIDIOC_DQBUF, &mut buf) }?;
        let index = buf.index as usize;
        // The index comes from the driver, and indexing on it would panic if a
        // buggy one reported a buffer outside the queue it allocated.
        if index >= self.buffers.len() {
            return Err(G2gError::Hardware(HardwareError::V4l2(-1)));
        }
        self.queued[index] = false;
        Ok((
            index,
            buf.bytesused,
            buffer_timestamp_ns(&Timestamp::from(buf.timestamp)),
        ))
    }
}

impl Drop for ExportedQueue {
    fn drop(&mut self) {
        if self.streaming {
            let mut buf_type = Type::VideoCapture as core::ffi::c_int;
            // SAFETY: VIDIOC_STREAMOFF takes a buffer-type int. A teardown error
            // has nowhere to go: closing the device fd releases the queue anyway.
            let _ =
                unsafe { capture_ioctl(self.handle.fd(), vidioc::VIDIOC_STREAMOFF, &mut buf_type) };
        }
        // No REQBUFS(0) here: a driver refuses to free buffers while an exported
        // dma-buf fd is still open, and a frame may well outlive the stream. The
        // queue is released when the device fd closes, and the buffer memory when
        // the last share of its fd drops.
    }
}

/// The dmabuf capture loop, run on the capture thread. Exports the MMAP buffers
/// once, then hands each dequeued buffer downstream as a share of its fd.
fn capture_dmabuf(
    device: &str,
    mode: Mode,
    min_bytes: usize,
    tx: &tokio::sync::mpsc::Sender<Captured>,
) -> Result<(), G2gError> {
    let dev = Device::with_path(device).map_err(|e| v4l2_err(&e))?;
    let format = Format::new(mode.width, mode.height, FourCC::new(&mode.fourcc));
    // The driver's own bytesperline for the format it accepted, not a computed
    // one: a device may pad rows.
    let actual = dev.set_format(&format).map_err(|e| v4l2_err(&e))?;
    let _ = dev.set_params(&Parameters::with_fps(mode.fps));

    let mut queue = ExportedQueue::new(&dev, actual.stride)?;
    queue.start()?;
    loop {
        while queue.refill()? == 0 {
            std::thread::sleep(DMABUF_RELEASE_POLL);
        }
        let (index, bytesused, timestamp_ns) = queue.dequeue()?;
        // A short frame (a driver hiccup) cannot be unpacked. Leaving it unsent
        // re-queues its buffer on the next pass instead of pushing a malformed
        // frame down the pipeline.
        if (bytesused as usize) < min_bytes {
            continue;
        }
        let captured = Captured {
            payload: CapturedPayload::DmaBuf(queue.buffers[index].clone()),
            timestamp_ns,
        };
        // Err means the receiver was dropped (limit reached or the pipeline shut
        // down).
        if tx.blocking_send(captured).is_err() {
            break;
        }
    }
    Ok(())
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
    io_mode: IoMode,
    /// 0 means run until error or downstream shutdown; otherwise stop after
    /// this many frames and emit EOS (the test / bounded-capture path).
    frame_limit: u64,
    /// Every mode the device confirmed during the probe, in advertised
    /// preference order. Cached: negotiation runs more than once, and each
    /// probe is a round of ioctls. Cleared when a property changes the request.
    modes: Vec<Mode>,
    /// The mode negotiation settled on, filled by `configure_pipeline` from the
    /// caps the solver fixed. The capture thread runs exactly this.
    chosen: Option<Mode>,
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
            io_mode: IoMode::default(),
            frame_limit: 0,
            modes: Vec::new(),
            chosen: None,
            configured: false,
        }
    }

    /// Request a capture size. The driver may snap to the nearest supported
    /// mode; the negotiated caps reflect what it actually chose.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.req_width = width;
        self.req_height = height;
        self.modes.clear();
        self
    }

    /// Request a frame rate in fps. Best-effort: the driver may pick another,
    /// and many UVC cams free-run below the one they accepted. PTS comes from
    /// the driver's buffer timestamps, so it holds either way; this rate only
    /// feeds the advertised caps, the latency report, and the fallback period.
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.req_fps = fps;
        self.modes.clear();
        self
    }

    /// Stop after `n` frames and emit EOS. Without this the source runs until
    /// an error or until downstream drops (no EOS on its own).
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// Choose how buffers leave the element. [`IoMode::DmaBuf`] exports the
    /// driver's capture buffers and emits them in [`MemoryDomain::DmaBuf`]
    /// without a copy, which restricts the advertised formats to the raw ones
    /// (an exported fd carries no payload length, so MJPEG cannot travel that
    /// way).
    pub fn with_io_mode(mut self, mode: IoMode) -> Self {
        self.io_mode = mode;
        self
    }

    /// Select the camera by the persistent id the device monitor reports for
    /// it, resolved to a node path at negotiation. Overrides
    /// [`new`](Self::new)'s path.
    pub fn with_device_id(mut self, id: impl Into<String>) -> Self {
        self.device_id = id.into();
        self.device_resolved = false;
        self.modes.clear();
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

    /// Probe the device: for every fourcc it reports that [`FOURCCS`] covers,
    /// set that format at the requested geometry and read back what the driver
    /// actually chose. Caches the confirmed modes in preference order; the
    /// probe device is dropped before `run`.
    fn probe_modes(&mut self) -> Result<&[Mode], G2gError> {
        if !self.modes.is_empty() {
            return Ok(&self.modes);
        }
        self.resolve_device()?;
        let dev = Device::with_path(&self.device).map_err(|e| v4l2_err(&e))?;
        // Controls are device state, not per-fd, so applying them on the probe
        // handle holds for the capture thread's own open.
        apply_controls(&dev, &self.controls)?;
        let reported = dev.enum_formats().map_err(|e| v4l2_err(&e))?;

        // Kept so a probe that confirms nothing reports why (a camera already
        // streaming in another process refuses every set_format with EBUSY)
        // instead of a bare caps mismatch.
        let mut refusal: Option<G2gError> = None;
        for (fourcc, format) in FOURCCS {
            if !reported.iter().any(|d| &d.fourcc.repr == fourcc) {
                continue;
            }
            let fmt = Format::new(self.req_width, self.req_height, FourCC::new(fourcc));
            // A driver that rejects one format, substitutes another, or reports
            // a degenerate geometry leaves that format out rather than failing
            // the whole probe: the remaining formats may still work.
            let actual = match dev.set_format(&fmt) {
                Ok(actual) => actual,
                Err(e) => {
                    refusal.get_or_insert_with(|| v4l2_err(&e));
                    continue;
                }
            };
            if &actual.fourcc.repr != fourcc || actual.width == 0 || actual.height == 0 {
                continue;
            }
            // Frame rate is per format and best-effort: many UVC cams ignore
            // set_params for some modes, so fall back to the request when the
            // read-back is unusable.
            let fps = match dev.set_params(&Parameters::with_fps(self.req_fps)) {
                Ok(p) if p.interval.numerator > 0 => p.interval.denominator / p.interval.numerator,
                _ => self.req_fps,
            };
            self.modes.push(Mode {
                fourcc: *fourcc,
                format,
                width: actual.width,
                height: actual.height,
                fps,
            });
        }

        if self.modes.is_empty() {
            // Either the device refused every format, or it offers nothing this
            // element can carry (a greyscale or bayer-only sensor, say).
            return Err(refusal.unwrap_or(G2gError::CapsMismatch));
        }
        Ok(&self.modes)
    }

    /// The confirmed modes negotiation may settle on, in preference order: every
    /// one the device offers, less those the current `io-mode` cannot carry.
    fn advertised_modes(&mut self) -> Result<Vec<Mode>, G2gError> {
        let io_mode = self.io_mode;
        let modes: Vec<Mode> = self
            .probe_modes()?
            .iter()
            .copied()
            .filter(|mode| io_mode.carries(mode.format))
            .collect();
        // A camera with nothing but an MJPEG mode has no format to export, so
        // dmabuf capture fails negotiation rather than silently copying.
        if modes.is_empty() {
            return Err(G2gError::CapsMismatch);
        }
        Ok(modes)
    }

    /// Every format the device confirmed, in preference order, as the set the
    /// solver picks from.
    fn produced_caps(&mut self) -> Result<CapsSet, G2gError> {
        Ok(CapsSet::from_alternatives(
            self.advertised_modes()?.iter().map(Mode::caps).collect(),
        ))
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
        // Single-caps callers get the preferred format (YUYV where the device
        // has it); the whole set travels through `caps_constraint`.
        core::future::ready(
            self.advertised_modes()
                .and_then(|modes| modes.first().map(Mode::caps).ok_or(G2gError::CapsMismatch)),
        )
    }

    /// Produces every format the device confirmed during the ioctl probe, in
    /// preference order, so the solver settles the link on the one downstream
    /// wants (raw for a convert / display chain, `image/jpeg` for a decode
    /// chain) and the chain takes the native arc-consistency path. The probe is
    /// synchronous, so no async body is needed.
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(self.produced_caps().map(CapsConstraint::Produces))
    }

    /// Records which advertised mode the solver fixed, which is what the
    /// capture thread then runs.
    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.modes.is_empty() {
            return Err(G2gError::NotConfigured);
        }
        let chosen = self
            .modes
            .iter()
            .filter(|m| self.io_mode.carries(m.format))
            .find(|m| m.caps().intersect(absolute_caps).is_ok())
            .copied()
            .ok_or(G2gError::CapsMismatch)?;
        self.chosen = Some(chosen);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "V4L2 camera source",
            "Source/Video",
            "Captures video from a V4L2 device (YUYV / NV12 / I420 / MJPEG)",
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
            PropertySpec::new(
                "num-buffers",
                PropKind::Int,
                "frames to emit then EOS (-1 = forever)",
            )
            .with_default("-1"),
            PropertySpec::new(
                "io-mode",
                PropKind::Str,
                "how buffers leave the element: auto | mmap (copy to system memory) | dmabuf \
                 (export the capture buffer, raw formats only)",
            )
            .with_default("auto")
            .with_enum_values("auto | mmap | dmabuf"),
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
        // Every property here feeds the probe (which device, which geometry and
        // rate, which controls), so the cached modes no longer describe it.
        self.modes.clear();
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
            "num-buffers" => {
                let n = value.as_int().ok_or(PropError::Type)?;
                self.frame_limit = if n < 0 { 0 } else { n as u64 };
                Ok(())
            }
            "io-mode" => {
                let name = value.as_str().ok_or(PropError::Type)?;
                // A V4L2 method this element does not implement is refused, not
                // quietly treated as the default.
                self.io_mode = IO_MODES
                    .iter()
                    .find(|(nick, _)| *nick == name)
                    .map(|(_, mode)| *mode)
                    .ok_or(PropError::Value)?;
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
            "num-buffers" => Some(PropValue::Int(if self.frame_limit == 0 {
                -1
            } else {
                self.frame_limit as i64
            })),
            "io-mode" => Some(PropValue::Str(self.io_mode.name().to_string())),
            _ => None,
        }
    }

    /// `io-mode=dmabuf` hands the driver's capture buffer downstream as a dma-buf
    /// fd; every other mode copies it into system memory.
    fn output_memory(&self) -> MemoryDomainKind {
        match self.io_mode {
            IoMode::DmaBuf => MemoryDomainKind::DmaBuf,
            IoMode::Auto | IoMode::Mmap => MemoryDomainKind::System,
        }
    }

    /// Live source: contributes one frame period of latency so the sink keeps a
    /// frame in hand and never runs dry waiting on capture.
    fn latency(&self) -> LatencyReport {
        let fps = self
            .chosen
            .or_else(|| self.modes.first().copied())
            .map(|m| m.fps)
            .unwrap_or(self.req_fps);
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
            let mode = self.chosen.ok_or(G2gError::NotConfigured)?;
            let (w, h, fps) = (mode.width, mode.height, mode.fps);
            let limit = self.frame_limit;
            let device = self.device.clone();
            // A fixed-size format's short frame (a driver hiccup) cannot be
            // unpacked, so it is dropped. MJPEG's length varies per frame, so
            // only an empty buffer is dropped there.
            let min_bytes = mode.format.frame_bytes(w, h).unwrap_or(1);

            // Bounded channel: the capture thread blocks once the pipeline is
            // BUFFER_COUNT frames behind, so we don't grow memory unboundedly.
            // The dmabuf path bounds it lower, since a queued frame there holds a
            // capture buffer the driver still needs.
            let io_mode = self.io_mode;
            let queued_frames = match io_mode {
                IoMode::DmaBuf => DMABUF_INFLIGHT,
                IoMode::Auto | IoMode::Mmap => BUFFER_COUNT as usize,
            };
            let (tx, mut rx) = tokio::sync::mpsc::channel::<Captured>(queued_frames);

            // Blocking V4L2 capture on its own thread. It owns the device and
            // the mmap stream (which borrows the device), copies each frame's
            // payload out of the mmap buffer, and hands it to the async side.
            // In dmabuf mode it exports the buffers instead and copies nothing.
            let handle = std::thread::spawn(move || -> Result<(), G2gError> {
                if io_mode == IoMode::DmaBuf {
                    return capture_dmabuf(&device, mode, min_bytes, &tx);
                }
                let dev = Device::with_path(&device).map_err(|e| v4l2_err(&e))?;
                dev.set_format(&Format::new(w, h, FourCC::new(&mode.fourcc)))
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
                        payload: CapturedPayload::Bytes(payload),
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
                // Skip a short frame rather than push a malformed buffer
                // downstream. The dmabuf path checks the driver's byte count on
                // the capture thread instead, where the buffer can still be
                // re-queued.
                if let CapturedPayload::Bytes(bytes) = &captured.payload {
                    if bytes.len() < min_bytes {
                        continue;
                    }
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
                let domain = match captured.payload {
                    CapturedPayload::Bytes(bytes) => {
                        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice()))
                    }
                    CapturedPayload::DmaBuf(buffer) => MemoryDomain::DmaBuf(buffer),
                };
                let frame = Frame {
                    domain,
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns,
                        capture_ns: pts,
                        arrival_ns,
                        // raw frames and MJPEG's per-frame JPEGs are each
                        // independently decodable
                        keyframe: true,
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
    /// Every format the element can carry, in preference order. Which of them a
    /// given device offers, and at what geometry and rate, is decided by the
    /// probe in `intercept_caps`.
    fn pad_templates() -> Vec<PadTemplate> {
        let alternatives = FOURCCS
            .iter()
            .map(|(_, format)| match format.raw_format() {
                Some(raw) => Caps::RawVideo {
                    format: raw,
                    width: Dim::Any,
                    height: Dim::Any,
                    framerate: Rate::Any,
                    interlace: g2g_core::Interlace::Any,
                },
                None => Caps::CompressedVideo {
                    codec: g2g_core::VideoCodec::Mjpeg,
                    width: Dim::Any,
                    height: Dim::Any,
                    framerate: Rate::Any,
                },
            })
            .collect();
        Vec::from([PadTemplate::source(CapsSet::from_alternatives(
            alternatives,
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
    fn io_mode_round_trips_and_refuses_the_methods_it_does_not_implement() {
        let mut src = V4l2Src::new("/dev/video0");
        // the default copies into system memory, as it always did.
        assert_eq!(
            SourceLoop::get_property(&src, "io-mode"),
            Some(PropValue::Str("auto".to_string()))
        );
        assert_eq!(SourceLoop::output_memory(&src), MemoryDomainKind::System);

        SourceLoop::set_property(&mut src, "io-mode", PropValue::Str("dmabuf".to_string()))
            .expect("dmabuf");
        assert_eq!(
            SourceLoop::get_property(&src, "io-mode"),
            Some(PropValue::Str("dmabuf".to_string()))
        );
        // the output domain follows the mode, so a downstream GPU consumer sees
        // what it will actually be handed.
        assert_eq!(SourceLoop::output_memory(&src), MemoryDomainKind::DmaBuf);

        // a V4L2 method this element does not implement must be refused, not
        // accepted and then ignored.
        for unsupported in ["userptr", "dmabuf-import", "rw", ""] {
            assert_eq!(
                SourceLoop::set_property(
                    &mut src,
                    "io-mode",
                    PropValue::Str(unsupported.to_string())
                ),
                Err(PropError::Value),
                "{unsupported} must not be accepted"
            );
        }
        // the refusals left the mode alone.
        assert_eq!(SourceLoop::output_memory(&src), MemoryDomainKind::DmaBuf);
        assert_eq!(
            V4l2Src::new("/dev/video0")
                .with_io_mode(IoMode::DmaBuf)
                .io_mode,
            IoMode::DmaBuf
        );
    }

    #[test]
    fn dmabuf_export_carries_only_the_raw_formats() {
        // an exported fd describes no payload length, so MJPEG's variable-length
        // access unit must not be advertised for dmabuf capture. Every raw
        // format still is, in both modes.
        for (_, format) in FOURCCS {
            let raw = format.raw_format().is_some();
            assert_eq!(IoMode::DmaBuf.carries(format), raw, "{format:?}");
            assert!(IoMode::Auto.carries(format), "{format:?}");
            assert!(IoMode::Mmap.carries(format), "{format:?}");
        }
        // every accepted nick maps back to the name it was set with, or
        // get_property would report a different mode than was asked for.
        for (name, mode) in IO_MODES {
            assert_eq!(mode.name(), name);
        }
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
            .configure_pipeline(&CapturePixelFormat::Yuyv.caps(640, 480, 30))
            .expect_err("configure without negotiate must fail");
        assert_eq!(err, G2gError::NotConfigured);
    }

    #[test]
    fn yuyv_leads_the_advertised_formats_and_mjpeg_trails() {
        // A chain that pins nothing takes the first alternative, so raw YUYV
        // must stay the default and MJPEG (needs a decoder) must come last.
        let order: Vec<CapturePixelFormat> = FOURCCS.iter().map(|(_, f)| *f).collect();
        assert_eq!(order.first(), Some(&CapturePixelFormat::Yuyv));
        assert_eq!(order.last(), Some(&CapturePixelFormat::Mjpeg));
        // the fourcc table is the only place a code is written, and no code
        // may appear twice (one format would shadow the other).
        for (i, (code, _)) in FOURCCS.iter().enumerate() {
            assert!(!FOURCCS[..i].iter().any(|(c, _)| c == code));
            assert_eq!(format_for_fourcc(code), Some(FOURCCS[i].1));
        }
        assert_eq!(format_for_fourcc(b"GREY"), None);
    }

    /// The whole advertised set must fixate, or a chain that pins one of the
    /// formats fails negotiation instead of selecting it.
    #[test]
    fn every_advertised_format_fixates() {
        for (_, format) in FOURCCS {
            let caps = format.caps(1280, 720, 30);
            assert_eq!(caps.fixate().expect("fixates"), caps);
        }
    }
}

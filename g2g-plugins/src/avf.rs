//! M738: AVFoundation capture.
//!
//! Two `SourceLoop` sources, the macOS analog of `Camera2Src` / `AAudioSrc`:
//! - [`AvfVideoSrc`]: default camera -> NV12 frames (packed System bytes by
//!   default, retained IOSurface-backed `CVPixelBuffer`s in `cv-output` mode,
//!   so `avfvideosrc ! metalvideosink` presents camera frames with no CPU
//!   copy).
//! - [`AvfAudioSrc`]: default microphone -> interleaved S16 PCM.
//!
//! Both drive one `AVCaptureSession` (device -> `AVCaptureDeviceInput` ->
//! data output) whose delegate delivers sample buffers on a private serial
//! dispatch queue; a mutex-guarded queue hands them to the element's run loop,
//! the same callback-boundary shape as the Core Audio elements.
//!
//! Camera and microphone are permission-gated (TCC) and absent on the CI
//! runner, so the tests probe: no default device (or a denied open) surfaces
//! as a structured hardware error, asserted as the graceful path; real capture
//! is validated on a Mac with devices, like the Android capture elements were
//! on a Pixel.

use core::ffi::c_void;
use core::future::Future;
use core::pin::Pin;

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use dispatch2::{DispatchQueue, DispatchRetained};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{define_class, msg_send, AllocAnyThread, DefinedClass};
use objc2_av_foundation::{
    AVCaptureAudioDataOutput, AVCaptureAudioDataOutputSampleBufferDelegate, AVCaptureConnection,
    AVCaptureDevice, AVCaptureDeviceInput, AVCaptureOutput, AVCaptureSession,
    AVCaptureSessionPreset640x480, AVCaptureVideoDataOutput,
    AVCaptureVideoDataOutputSampleBufferDelegate, AVMediaTypeAudio, AVMediaTypeVideo,
};
use objc2_avf_audio::{
    AVFormatIDKey, AVLinearPCMBitDepthKey, AVLinearPCMIsFloatKey, AVLinearPCMIsNonInterleaved,
    AVNumberOfChannelsKey, AVSampleRateKey,
};
use objc2_core_audio_types::kAudioFormatLinearPCM;
use objc2_core_foundation::{CFRetained, CFString};
use objc2_core_media::{CMBlockBuffer, CMSampleBuffer};
use objc2_core_video::{
    kCVPixelBufferPixelFormatTypeKey, CVPixelBuffer, CVPixelBufferGetHeight,
    CVPixelBufferGetIOSurface, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
};
use objc2_foundation::{NSDictionary, NSNumber, NSObject, NSObjectProtocol, NSString};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    HardwareError, MemoryDomain, OutputSink, OwnedCvPixelBuffer, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

use crate::cvnv12::{pack_nv12_locked, CvBufferOwner};

use alloc::boxed::Box;
use alloc::vec::Vec;

/// How long the run loop waits for the delegate before checking liveness.
const IO_TIMEOUT: Duration = Duration::from_secs(1);
/// Consecutive empty waits before the source surfaces a dead device.
const MAX_IDLE_WAITS: u32 = 3;

fn hw() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// One captured camera frame, retained straight from the delegate.
struct CapturedVideo {
    buf: CFRetained<CVPixelBuffer>,
    width: u32,
    height: u32,
    pixel_format: u32,
    io_surface_backed: bool,
}

// SAFETY: the retained pixel buffer crosses from the delegate queue to the run
// loop through the mutex; CoreFoundation retain/release is thread-safe and the
// pixels are immutable after capture.
unsafe impl Send for CapturedVideo {}

/// Shared between a capture delegate (on its dispatch queue) and the element's
/// run loop.
#[derive(Default)]
struct Shared<T> {
    filled: Mutex<VecDeque<T>>,
    cv: Condvar,
}

struct VideoIvars {
    shared: Arc<Shared<CapturedVideo>>,
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements; no Drop impl.
    #[unsafe(super(NSObject))]
    #[name = "G2gAvfVideoDelegate"]
    #[ivars = VideoIvars]
    struct VideoDelegate;

    unsafe impl NSObjectProtocol for VideoDelegate {}

    unsafe impl AVCaptureVideoDataOutputSampleBufferDelegate for VideoDelegate {
        #[unsafe(method(captureOutput:didOutputSampleBuffer:fromConnection:))]
        unsafe fn captureOutput_didOutputSampleBuffer_fromConnection(
            &self,
            _output: &AVCaptureOutput,
            sample_buffer: &CMSampleBuffer,
            _connection: &AVCaptureConnection,
        ) {
            // SAFETY: the sample buffer is valid for the callback; the image
            // buffer is retained (+1) by the accessor, keeping it alive past
            // the delegate's scope as Apple's docs require.
            let Some(image) = (unsafe { sample_buffer.image_buffer() }) else {
                return;
            };
            // A CVImageBufferRef IS a CVPixelBufferRef here (video output).
            // SAFETY: same-representation CF types; the retain moves over.
            let buf: CFRetained<CVPixelBuffer> = unsafe { CFRetained::cast_unchecked(image) };
            let width = CVPixelBufferGetWidth(&buf) as u32;
            let height = CVPixelBufferGetHeight(&buf) as u32;
            let pixel_format = CVPixelBufferGetPixelFormatType(&buf);
            let io_surface_backed = CVPixelBufferGetIOSurface(Some(&buf)).is_some();
            let shared = &self.ivars().shared;
            shared.filled.lock().unwrap().push_back(CapturedVideo {
                buf,
                width,
                height,
                pixel_format,
                io_surface_backed,
            });
            shared.cv.notify_one();
        }
    }
);

impl VideoDelegate {
    fn new(shared: Arc<Shared<CapturedVideo>>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(VideoIvars { shared });
        // SAFETY: plain NSObject init.
        unsafe { msg_send![super(this), init] }
    }
}

struct AudioIvars {
    shared: Arc<Shared<Vec<u8>>>,
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements; no Drop impl.
    #[unsafe(super(NSObject))]
    #[name = "G2gAvfAudioDelegate"]
    #[ivars = AudioIvars]
    struct AudioDelegate;

    unsafe impl NSObjectProtocol for AudioDelegate {}

    unsafe impl AVCaptureAudioDataOutputSampleBufferDelegate for AudioDelegate {
        #[unsafe(method(captureOutput:didOutputSampleBuffer:fromConnection:))]
        unsafe fn captureOutput_didOutputSampleBuffer_fromConnection(
            &self,
            _output: &AVCaptureOutput,
            sample_buffer: &CMSampleBuffer,
            _connection: &AVCaptureConnection,
        ) {
            // SAFETY: the sample buffer is valid for the callback; the block
            // buffer accessor retains it for this scope.
            let Some(block) = (unsafe { sample_buffer.data_buffer() }) else {
                return;
            };
            let mut total_len: usize = 0;
            let mut len_at: usize = 0;
            let mut data_ptr: *mut core::ffi::c_char = core::ptr::null_mut();
            // SAFETY: out-params are valid slots; the block is live.
            let st = unsafe {
                CMBlockBuffer::data_pointer(&block, 0, &mut len_at, &mut total_len, &mut data_ptr)
            };
            if st != 0 || data_ptr.is_null() || total_len == 0 {
                return;
            }
            // SAFETY: the block guarantees `total_len` contiguous bytes.
            let bytes = unsafe { core::slice::from_raw_parts(data_ptr as *const u8, total_len) };
            let shared = &self.ivars().shared;
            shared.filled.lock().unwrap().push_back(bytes.to_vec());
            shared.cv.notify_one();
        }
    }
);

impl AudioDelegate {
    fn new(shared: Arc<Shared<Vec<u8>>>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(AudioIvars { shared });
        // SAFETY: plain NSObject init.
        unsafe { msg_send![super(this), init] }
    }
}

/// A running capture session plus everything that must stay alive with it.
struct SessionState<D> {
    session: Retained<AVCaptureSession>,
    _delegate: Retained<D>,
    _queue: DispatchRetained<DispatchQueue>,
}

impl<D> Drop for SessionState<D> {
    fn drop(&mut self) {
        // SAFETY: live session; stopping tears the capture graph down.
        unsafe { self.session.stopRunning() };
    }
}

impl<D> core::fmt::Debug for SessionState<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SessionState").finish_non_exhaustive()
    }
}

/// Build the session skeleton shared by both sources: default device of
/// `media_type` -> device input -> session. `None` device (absent hardware or
/// denied TCC permission) surfaces as the structured hardware error the CI
/// probe asserts.
unsafe fn open_session(
    media_type: Option<&'static objc2_av_foundation::AVMediaType>,
) -> Result<Retained<AVCaptureSession>, G2gError> {
    let media_type = media_type.ok_or_else(hw)?;
    // SAFETY: the media type is a valid static; a nil default device is the
    // no-hardware / no-permission case.
    let device =
        unsafe { AVCaptureDevice::defaultDeviceWithMediaType(media_type) }.ok_or_else(hw)?;
    // SAFETY: the device is live; a deny (TCC) or busy device errors here.
    let input =
        unsafe { AVCaptureDeviceInput::deviceInputWithDevice_error(&device) }.map_err(|_| hw())?;
    let session = AVCaptureSession::new();
    // SAFETY: freshly created session; the input was just opened for it.
    unsafe {
        if !session.canAddInput(&input) {
            return Err(hw());
        }
        session.addInput(&input);
    }
    Ok(session)
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/// Captures NV12 frames from the default camera (VGA preset).
#[derive(Debug)]
pub struct AvfVideoSrc {
    target_buffers: u64,
    /// Emit retained IOSurface-backed `CVPixelBuffer`s (the M735 zero-copy
    /// domain) instead of packing NV12 to system memory.
    cv_output: bool,
    state: Option<SessionState<VideoDelegate>>,
    shared: Arc<Shared<CapturedVideo>>,
    configured: bool,
}

// SAFETY: the ObjC objects are used on the element's owning task only (the
// platform elements' single-thread-executor contract); the delegate hands
// frames over through the mutex-guarded shared queue.
unsafe impl Send for AvfVideoSrc {}

impl AvfVideoSrc {
    /// A camera source emitting `target_buffers` frames then EOS
    /// (`u64::MAX` = capture until stopped).
    pub fn new(target_buffers: u64) -> Self {
        Self {
            target_buffers,
            cv_output: false,
            state: None,
            shared: Arc::new(Shared::default()),
            configured: false,
        }
    }

    /// Emit zero-copy `CvPixelBuffer` frames (camera -> Metal with no CPU
    /// copy). Also settable as the `cv-output` property.
    pub fn with_cv_output(mut self) -> Self {
        self.cv_output = true;
        self
    }

    fn caps(&self) -> Caps {
        // The VGA session preset pins the geometry; the rate is nominal (the
        // camera paces itself, per-frame PTS carries the real timing).
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    fn open(&mut self) -> Result<(), G2gError> {
        if self.state.is_some() {
            return Ok(());
        }
        // SAFETY: static access; open_session validates everything else.
        let session = unsafe { open_session(AVMediaTypeVideo)? };
        // SAFETY: fresh session; VGA is universally supported.
        unsafe { session.setSessionPreset(AVCaptureSessionPreset640x480) };

        let output = AVCaptureVideoDataOutput::new();
        // Pin the delivered format to '420v' NV12 (every Apple camera pipeline
        // supports the bi-planar 4:2:0 formats).
        // SAFETY: the CF and NS string types are toll-free bridged; the key
        // static is valid.
        let key: &NSString =
            unsafe { &*(kCVPixelBufferPixelFormatTypeKey as *const CFString as *const NSString) };
        let fourcc = NSNumber::numberWithUnsignedInt(crate::cvnv12::K_CV_PIXEL_FORMAT_420V);
        let value: &AnyObject = &fourcc;
        let settings = NSDictionary::from_slices(&[key], &[value]);
        // SAFETY: fresh output; the settings dictionary is well-formed.
        unsafe { output.setVideoSettings(Some(&settings)) };

        let queue = DispatchQueue::new("g2g.avfvideosrc", None);
        let delegate = VideoDelegate::new(self.shared.clone());
        // SAFETY: delegate + queue outlive the session (held in SessionState).
        unsafe {
            output.setSampleBufferDelegate_queue(
                Some(ProtocolObject::from_ref(&*delegate)),
                Some(&queue),
            );
            if !session.canAddOutput(&output) {
                return Err(hw());
            }
            session.addOutput(&output);
            session.startRunning();
        }
        self.state = Some(SessionState {
            session,
            _delegate: delegate,
            _queue: queue,
        });
        Ok(())
    }
}

impl SourceLoop for AvfVideoSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.open()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "cv-output",
            PropKind::Bool,
            "emit retained IOSurface-backed CVPixelBuffers (zero-copy) instead of packed NV12",
        )
        .with_default("false")];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "cv-output" => {
                self.cv_output = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "cv-output" => Some(PropValue::Bool(self.cv_output)),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let mut seq = 0u64;
            let mut idle = 0u32;
            while seq < self.target_buffers {
                let captured = {
                    let mut filled = self.shared.filled.lock().unwrap();
                    loop {
                        if let Some(c) = filled.pop_front() {
                            idle = 0;
                            break Some(c);
                        }
                        let (guard, timeout) = self
                            .shared
                            .cv
                            .wait_timeout(filled, IO_TIMEOUT)
                            .map_err(|_| hw())?;
                        filled = guard;
                        if timeout.timed_out() && filled.is_empty() {
                            idle += 1;
                            break None;
                        }
                    }
                };
                let Some(c) = captured else {
                    if idle >= MAX_IDLE_WAITS {
                        // The camera delivered nothing for several deadlines:
                        // dead capture, surface rather than hang.
                        return Err(hw());
                    }
                    continue;
                };

                let arrival_ns = g2g_core::metrics::monotonic_ns();
                let domain = if self.cv_output {
                    let ptr = CFRetained::as_ptr(&c.buf).as_ptr() as u64;
                    MemoryDomain::CvPixelBuffer(OwnedCvPixelBuffer::new(
                        ptr,
                        c.width,
                        c.height,
                        c.pixel_format,
                        c.io_surface_backed,
                        Arc::new(CvBufferOwner(c.buf)),
                    ))
                } else {
                    let packed = pack_nv12_locked(&c.buf, c.width as usize, c.height as usize)
                        .ok_or_else(hw)?;
                    MemoryDomain::System(SystemSlice::from_boxed(packed))
                };
                let frame = Frame {
                    domain,
                    timing: FrameTiming {
                        pts_ns: arrival_ns,
                        dts_ns: arrival_ns,
                        duration_ns: 0,
                        capture_ns: arrival_ns,
                        arrival_ns,
                        keyframe: false,
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                seq += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            self.state = None; // stops the session
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

// ---------------------------------------------------------------------------
// Microphone
// ---------------------------------------------------------------------------

/// Captures interleaved S16 PCM from the default microphone.
#[derive(Debug)]
pub struct AvfAudioSrc {
    sample_rate: u32,
    channels: u8,
    target_buffers: u64,
    state: Option<SessionState<AudioDelegate>>,
    shared: Arc<Shared<Vec<u8>>>,
    configured: bool,
}

// SAFETY: same contract as `AvfVideoSrc`.
unsafe impl Send for AvfAudioSrc {}

impl AvfAudioSrc {
    /// A mic source at `sample_rate` / `channels` (S16LE), emitting
    /// `target_buffers` buffers then EOS (`u64::MAX` = capture until stopped).
    pub fn new(sample_rate: u32, channels: u8, target_buffers: u64) -> Self {
        Self {
            sample_rate: sample_rate.max(1),
            channels: channels.max(1),
            target_buffers,
            state: None,
            shared: Arc::new(Shared::default()),
            configured: false,
        }
    }

    fn caps(&self) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: self.channels,
            sample_rate: self.sample_rate,
        }
    }

    fn open(&mut self) -> Result<(), G2gError> {
        if self.state.is_some() {
            return Ok(());
        }
        // SAFETY: static access; open_session validates everything else.
        let session = unsafe { open_session(AVMediaTypeAudio)? };

        let output = AVCaptureAudioDataOutput::new();
        // Pin the delivered format to interleaved packed S16 at our rate.
        // SAFETY: the key statics are valid NSStrings on macOS.
        let settings = unsafe {
            let keys = [
                AVFormatIDKey.ok_or_else(hw)?,
                AVSampleRateKey.ok_or_else(hw)?,
                AVNumberOfChannelsKey.ok_or_else(hw)?,
                AVLinearPCMBitDepthKey.ok_or_else(hw)?,
                AVLinearPCMIsFloatKey.ok_or_else(hw)?,
                AVLinearPCMIsNonInterleaved.ok_or_else(hw)?,
            ];
            let fmt = NSNumber::numberWithUnsignedInt(kAudioFormatLinearPCM);
            let rate = NSNumber::numberWithDouble(self.sample_rate as f64);
            let ch = NSNumber::numberWithUnsignedInt(self.channels as u32);
            let bits = NSNumber::numberWithUnsignedInt(16);
            let no = NSNumber::numberWithBool(false);
            let values: [&AnyObject; 6] = [&fmt, &rate, &ch, &bits, &no, &no];
            NSDictionary::from_slices(&keys, &values)
        };
        // SAFETY: fresh output; the settings dictionary is well-formed.
        unsafe { output.setAudioSettings(Some(&settings)) };

        let queue = DispatchQueue::new("g2g.avfaudiosrc", None);
        let delegate = AudioDelegate::new(self.shared.clone());
        // SAFETY: delegate + queue outlive the session (held in SessionState).
        unsafe {
            output.setSampleBufferDelegate_queue(
                Some(ProtocolObject::from_ref(&*delegate)),
                Some(&queue),
            );
            if !session.canAddOutput(&output) {
                return Err(hw());
            }
            session.addOutput(&output);
            session.startRunning();
        }
        self.state = Some(SessionState {
            session,
            _delegate: delegate,
            _queue: queue,
        });
        Ok(())
    }
}

impl SourceLoop for AvfAudioSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.open()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let frame_bytes = 2 * self.channels as usize;
            let ns_per_frame = 1_000_000_000u64 / self.sample_rate as u64;

            let mut total_frames = 0u64;
            let mut seq = 0u64;
            let mut idle = 0u32;
            while seq < self.target_buffers {
                let chunk = {
                    let mut filled = self.shared.filled.lock().unwrap();
                    loop {
                        if let Some(c) = filled.pop_front() {
                            idle = 0;
                            break Some(c);
                        }
                        let (guard, timeout) = self
                            .shared
                            .cv
                            .wait_timeout(filled, IO_TIMEOUT)
                            .map_err(|_| hw())?;
                        filled = guard;
                        if timeout.timed_out() && filled.is_empty() {
                            idle += 1;
                            break None;
                        }
                    }
                };
                let Some(chunk) = chunk else {
                    if idle >= MAX_IDLE_WAITS {
                        return Err(hw());
                    }
                    continue;
                };
                let frames = (chunk.len() / frame_bytes) as u64;
                if frames == 0 {
                    continue;
                }
                let pts_ns = total_frames * ns_per_frame;
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(chunk.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns,
                        dts_ns: pts_ns,
                        duration_ns: frames * ns_per_frame,
                        capture_ns: pts_ns,
                        arrival_ns: g2g_core::metrics::monotonic_ns(),
                        keyframe: false,
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                total_frames += frames;
                seq += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            self.state = None; // stops the session
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

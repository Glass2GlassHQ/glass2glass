//! M739: ScreenCaptureKit display capture.
//!
//! [`ScreenCaptureSrc`] is a `SourceLoop` source that captures the main
//! display as NV12 frames through an `SCStream`: shareable-content lookup ->
//! `SCContentFilter` over the first display -> `SCStreamConfiguration` pinned
//! to `'420v'` at the display geometry -> an `SCStreamOutput` delegate feeding
//! the same delegate-to-run-loop handoff as the AVFoundation camera (shared in
//! `cvnv12`). Packed System NV12 by default, `cv-output` for zero-copy
//! IOSurface-backed frames.
//!
//! Screen recording is TCC permission-gated: on a denied or headless host the
//! shareable-content lookup returns nothing, surfacing as the structured
//! hardware error the CI probe asserts (real capture is validated on a Mac
//! with the permission granted).

use core::future::Future;
use core::pin::Pin;
use core::ptr::NonNull;

use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use block2::RcBlock;
use dispatch2::{DispatchQueue, DispatchRetained};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{define_class, msg_send, AllocAnyThread, DefinedClass};
use objc2_core_foundation::CFRetained;
use objc2_core_media::CMSampleBuffer;
use objc2_foundation::{NSArray, NSError, NSObject, NSObjectProtocol};
use objc2_screen_capture_kit::{
    SCContentFilter, SCShareableContent, SCStream, SCStreamConfiguration, SCStreamOutput,
    SCStreamOutputType, SCWindow,
};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError, HardwareError,
    MemoryDomain, OutputSink, OwnedCvPixelBuffer, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, Rate, RawVideoFormat,
};

use crate::cvnv12::{
    captured_video_from_sample, pack_nv12_locked, CapturedVideo, CvBufferOwner, Shared,
    K_CV_PIXEL_FORMAT_420V,
};

use alloc::boxed::Box;

/// How long the run loop waits for the delegate before checking liveness, and
/// how long the shareable-content lookup may take.
const IO_TIMEOUT: Duration = Duration::from_secs(1);
/// Consecutive empty waits before the source surfaces a dead capture.
const MAX_IDLE_WAITS: u32 = 3;

fn hw() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// A retained ObjC pointer crossing the completion-block boundary as a plain
/// integer (the block runs on an arbitrary queue).
struct RetainedPtr(usize);
// SAFETY: carries ownership of one retain; the receiver adopts it exactly once.
unsafe impl Send for RetainedPtr {}

struct OutputIvars {
    shared: Arc<Shared<CapturedVideo>>,
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements; no Drop impl.
    #[unsafe(super(NSObject))]
    #[name = "G2gSckStreamOutput"]
    #[ivars = OutputIvars]
    struct StreamOutput;

    unsafe impl NSObjectProtocol for StreamOutput {}

    unsafe impl SCStreamOutput for StreamOutput {
        #[unsafe(method(stream:didOutputSampleBuffer:ofType:))]
        unsafe fn stream_did_output_sample_buffer_of_type(
            &self,
            _stream: &SCStream,
            sample_buffer: &CMSampleBuffer,
            of_type: SCStreamOutputType,
        ) {
            if of_type != SCStreamOutputType::Screen {
                return;
            }
            let Some(captured) = captured_video_from_sample(sample_buffer) else {
                return;
            };
            let shared = &self.ivars().shared;
            shared.filled.lock().unwrap().push_back(captured);
            shared.cv.notify_one();
        }
    }
);

impl StreamOutput {
    fn new(shared: Arc<Shared<CapturedVideo>>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(OutputIvars { shared });
        // SAFETY: plain NSObject init.
        unsafe { msg_send![super(this), init] }
    }
}

/// The running stream plus everything that must stay alive with it.
struct StreamState {
    stream: Retained<SCStream>,
    _output: Retained<StreamOutput>,
    _queue: DispatchRetained<DispatchQueue>,
    width: u32,
    height: u32,
}

impl Drop for StreamState {
    fn drop(&mut self) {
        // SAFETY: live stream; a fire-and-forget stop (no completion handler).
        unsafe { self.stream.stopCaptureWithCompletionHandler(None) };
    }
}

impl core::fmt::Debug for StreamState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StreamState")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

/// Fetch the shareable content synchronously (the API is completion-block
/// only). `None` means denied screen-recording permission or no display.
fn shareable_content() -> Option<Retained<SCShareableContent>> {
    let (tx, rx) = mpsc::channel::<Option<RetainedPtr>>();
    let block = RcBlock::new(
        move |content: *mut SCShareableContent, _error: *mut NSError| {
            let ptr = NonNull::new(content).map(|p| {
                // SAFETY: retain inside the callback so the pointer stays valid
                // past it; the receiver adopts this +1.
                let retained = unsafe { Retained::retain(p.as_ptr()) }.expect("non-null");
                RetainedPtr(Retained::into_raw(retained) as usize)
            });
            let _ = tx.send(ptr);
        },
    );
    // SAFETY: the block is valid for the call; ScreenCaptureKit copies it.
    unsafe { SCShareableContent::getShareableContentWithCompletionHandler(&block) };
    let ptr = rx.recv_timeout(5 * IO_TIMEOUT).ok()??;
    // SAFETY: adopts the +1 the block took.
    Some(unsafe { Retained::from_raw(ptr.0 as *mut SCShareableContent) }.expect("non-null"))
}

/// Captures the main display as NV12 frames.
#[derive(Debug)]
pub struct ScreenCaptureSrc {
    target_buffers: u64,
    /// Emit retained IOSurface-backed `CVPixelBuffer`s (the M735 zero-copy
    /// domain) instead of packing NV12 to system memory.
    cv_output: bool,
    state: Option<StreamState>,
    shared: Arc<Shared<CapturedVideo>>,
    configured: bool,
}

// SAFETY: the ObjC objects are used on the element's owning task only (the
// platform elements' single-thread-executor contract); the delegate hands
// frames over through the mutex-guarded shared queue.
unsafe impl Send for ScreenCaptureSrc {}

impl ScreenCaptureSrc {
    /// A display-capture source emitting `target_buffers` frames then EOS
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

    /// Emit zero-copy `CvPixelBuffer` frames (display -> Metal with no CPU
    /// copy). Also settable as the `cv-output` property.
    pub fn with_cv_output(mut self) -> Self {
        self.cv_output = true;
        self
    }

    /// Open the stream (if not already) and report the display geometry as the
    /// produced caps; the rate is nominal (the display paces itself).
    fn ensure_open(&mut self) -> Result<Caps, G2gError> {
        if let Some(st) = &self.state {
            return Ok(nv12_caps(st.width, st.height));
        }
        let content = shareable_content().ok_or_else(hw)?;
        // SAFETY: live content object.
        let displays = unsafe { content.displays() };
        let display = displays.iter().next().ok_or_else(hw)?;
        // SAFETY: live display; the CGDisplay-sized ints are positive.
        let (width, height) = unsafe {
            (
                display.width().max(2) as u32,
                display.height().max(2) as u32,
            )
        };
        // Even dims for NV12 chroma subsampling.
        let (width, height) = (width & !1, height & !1);

        // SAFETY: plain object construction with live arguments.
        let state = unsafe {
            let excluded = NSArray::<SCWindow>::from_slice(&[]);
            let filter = SCContentFilter::initWithDisplay_excludingWindows(
                SCContentFilter::alloc(),
                &display,
                &excluded,
            );
            let config = SCStreamConfiguration::new();
            config.setWidth(width as usize);
            config.setHeight(height as usize);
            config.setPixelFormat(K_CV_PIXEL_FORMAT_420V);

            let stream = SCStream::initWithFilter_configuration_delegate(
                SCStream::alloc(),
                &filter,
                &config,
                None,
            );
            let queue = DispatchQueue::new("g2g.screencapturesrc", None);
            let output = StreamOutput::new(self.shared.clone());
            stream
                .addStreamOutput_type_sampleHandlerQueue_error(
                    ProtocolObject::from_ref(&*output),
                    SCStreamOutputType::Screen,
                    Some(&queue),
                )
                .map_err(|_| hw())?;
            // Fire-and-forget start: a failed start delivers no frames and the
            // run loop's liveness deadline surfaces it.
            stream.startCaptureWithCompletionHandler(None);
            StreamState {
                stream,
                _output: output,
                _queue: queue,
                width,
                height,
            }
        };
        self.state = Some(state);
        Ok(nv12_caps(width, height))
    }
}

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

impl SourceLoop for ScreenCaptureSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.ensure_open())
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        let caps = self.ensure_open();
        core::future::ready(caps.map(|c| CapsConstraint::Produces(CapsSet::one(c))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.ensure_open()?;
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
                        // The stream delivered nothing for several deadlines
                        // (denied permission surfaces here too when the start
                        // failed asynchronously): dead capture, surface it.
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
            self.state = None; // stops the stream
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

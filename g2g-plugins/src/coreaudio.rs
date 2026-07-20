//! M737: macOS audio via the AudioToolbox `AudioQueue` C API.
//!
//! Two elements, the macOS analog of the WASAPI / ALSA / AAudio audio elements:
//! - [`CoreAudioSink`]: an `AsyncElement` sink that renders interleaved PCM
//!   (`PcmS16Le` / `PcmF32Le`) to the default output device.
//! - [`CoreAudioSrc`]: a `SourceLoop` source that captures interleaved PCM from
//!   the default input device (the microphone; permission-gated in a real app).
//!
//! `AudioQueue` is callback-driven (the queue's own thread returns spent render
//! buffers / delivers filled capture buffers), so each element shares a boxed,
//! mutex-guarded state with its callback: the sink keeps a free-buffer list it
//! blocks on when the queue is saturated (real device back-pressure), the source
//! keeps a filled-chunk queue its run loop drains. Everything else (the queue
//! ref, start/stop) runs on the element's owning task under the same
//! single-thread-executor `Send` contract as the other platform elements.
//!
//! The CI runner has no audio hardware, so the tests probe: a failed open is
//! reported and asserted as the graceful error path (like the Android
//! permission-gated probes); with a device present they render / capture for
//! real.

use core::ffi::c_void;
use core::future::Future;
use core::pin::Pin;
use core::ptr::{self, NonNull};

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use objc2_audio_toolbox::{
    AudioQueueAllocateBuffer, AudioQueueBufferRef, AudioQueueDispose, AudioQueueEnqueueBuffer,
    AudioQueueFlush, AudioQueueNewInput, AudioQueueNewOutput, AudioQueueRef, AudioQueueStart,
    AudioQueueStop,
};
use objc2_core_audio_types::{
    kAudioFormatFlagIsFloat, kAudioFormatFlagIsPacked, kAudioFormatFlagIsSignedInteger,
    kAudioFormatLinearPCM, AudioStreamBasicDescription, AudioStreamPacketDescription,
    AudioTimeStamp,
};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming,
    G2gError, HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

use alloc::boxed::Box;
use alloc::vec::Vec;

/// Render buffers in flight; the sink blocks when all are queued (device-paced
/// back-pressure, the AudioQueue analog of a blocking write).
const SINK_BUFFERS: usize = 4;
/// Capture buffers in flight.
const SRC_BUFFERS: usize = 8;
/// Bytes per queue buffer (~42 ms of 48 kHz stereo S16).
const BUF_BYTES: u32 = 16 * 1024;
/// How long to wait on the queue's callback before declaring the device wedged.
const IO_TIMEOUT: Duration = Duration::from_secs(1);
/// Consecutive empty capture waits before the source surfaces a dead device.
const MAX_IDLE_WAITS: u32 = 3;

fn hw() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// Bytes per sample for a g2g PCM format (None for compressed audio).
fn bytes_per_sample(format: AudioFormat) -> Option<usize> {
    match format {
        AudioFormat::PcmS16Le => Some(2),
        AudioFormat::PcmF32Le => Some(4),
        _ => None,
    }
}

/// Validate that `caps` is interleaved PCM and extract `(format, channels, rate)`.
fn pcm_params(caps: &Caps) -> Result<(AudioFormat, u8, u32), G2gError> {
    match caps {
        Caps::Audio {
            format,
            channels,
            sample_rate,
        } if bytes_per_sample(*format).is_some() && *channels > 0 && *sample_rate > 0 => {
            Ok((*format, *channels, *sample_rate))
        }
        _ => Err(G2gError::CapsMismatch),
    }
}

/// The Core Audio stream description for interleaved packed PCM. AudioQueue
/// takes this as authoritative (it converts to the device format internally),
/// so the negotiated caps are exactly what plays / records.
fn asbd(
    format: AudioFormat,
    channels: u8,
    sample_rate: u32,
) -> Option<AudioStreamBasicDescription> {
    let (bits, flags) = match format {
        AudioFormat::PcmS16Le => (16u32, kAudioFormatFlagIsSignedInteger),
        AudioFormat::PcmF32Le => (32u32, kAudioFormatFlagIsFloat),
        _ => return None,
    };
    let bytes_per_frame = (bits / 8) * channels as u32;
    Some(AudioStreamBasicDescription {
        mSampleRate: sample_rate as f64,
        mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: flags | kAudioFormatFlagIsPacked,
        mBytesPerPacket: bytes_per_frame,
        mFramesPerPacket: 1,
        mBytesPerFrame: bytes_per_frame,
        mChannelsPerFrame: channels as u32,
        mBitsPerChannel: bits,
        mReserved: 0,
    })
}

/// A queue buffer pointer that can cross the callback-thread boundary. The
/// buffer is owned by its `AudioQueue` and stays valid until the queue is
/// disposed.
#[derive(Debug, Clone, Copy)]
struct BufPtr(AudioQueueBufferRef);
// SAFETY: the pointer is only dereferenced by whichever side currently holds
// the buffer (the queue while playing / filling, the element after the
// callback hands it over through the mutex), never by both at once.
unsafe impl Send for BufPtr {}

// ---------------------------------------------------------------------------
// Sink (render / playback)
// ---------------------------------------------------------------------------

/// Shared with the render callback (which runs on the queue's own thread): the
/// buffers the queue has finished playing, ready to refill.
#[derive(Debug, Default)]
struct SinkShared {
    free: Mutex<Vec<BufPtr>>,
    cv: Condvar,
}

/// Render callback: the queue is done with `buf`; hand it back to `process`.
///
/// SAFETY: invoked by the queue's thread with the `*mut SinkShared` refcon
/// passed at create, which (boxed in `SinkState`) outlives the queue.
unsafe extern "C-unwind" fn sink_callback(
    user: *mut c_void,
    _aq: AudioQueueRef,
    buf: AudioQueueBufferRef,
) {
    // SAFETY: see above; the shared state is alive for the queue's life.
    let shared = unsafe { &*(user as *const SinkShared) };
    shared.free.lock().unwrap().push(BufPtr(buf));
    shared.cv.notify_one();
}

struct SinkState {
    queue: AudioQueueRef,
    shared: Box<SinkShared>,
    started: bool,
}

impl Drop for SinkState {
    fn drop(&mut self) {
        // SAFETY: the queue is live; immediate dispose stops callbacks before
        // returning, so `shared` (dropped after this) is not used again.
        unsafe {
            AudioQueueDispose(self.queue, true);
        }
    }
}

impl core::fmt::Debug for SinkState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SinkState")
            .field("started", &self.started)
            .finish_non_exhaustive()
    }
}

/// Renders interleaved PCM to the default Core Audio output device.
#[derive(Debug)]
pub struct CoreAudioSink {
    state: Option<SinkState>,
    format: AudioFormat,
    channels: u8,
    sample_rate: u32,
    configured: bool,
    rendered: u64,
}

// SAFETY: the queue ref is only used on the element's owning task (the
// single-thread-executor contract shared by the platform elements); the
// callback touches only the mutex-guarded `SinkShared`.
unsafe impl Send for CoreAudioSink {}

impl Default for CoreAudioSink {
    fn default() -> Self {
        Self::new()
    }
}

impl CoreAudioSink {
    pub fn new() -> Self {
        Self {
            state: None,
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            configured: false,
            rendered: 0,
        }
    }

    /// Whether the default output device opens (tests probe with this and
    /// report the denial otherwise, like the Android permission probes).
    pub fn device_available() -> bool {
        let mut probe = Self::new();
        probe.open().is_ok()
    }

    /// Count of PCM frames handed to the device. Useful in tests.
    pub fn rendered(&self) -> u64 {
        self.rendered
    }

    fn open(&mut self) -> Result<(), G2gError> {
        let mut desc = asbd(self.format, self.channels, self.sample_rate).ok_or_else(hw)?;
        let mut shared = Box::new(SinkShared::default());
        let mut queue: AudioQueueRef = ptr::null_mut();
        // SAFETY: `desc` and the out slot outlive the call; the refcon points at
        // the boxed shared state, which the returned `SinkState` keeps alive for
        // the queue's whole life. A null run loop uses the queue's own thread.
        let st = unsafe {
            AudioQueueNewOutput(
                NonNull::from(&mut desc),
                Some(sink_callback),
                shared.as_mut() as *mut SinkShared as *mut c_void,
                None,
                None,
                0,
                NonNull::from(&mut queue),
            )
        };
        if st != 0 || queue.is_null() {
            return Err(hw());
        }
        // Pre-allocate the buffer pool; every buffer starts on the free list.
        {
            let mut free = shared.free.lock().unwrap();
            for _ in 0..SINK_BUFFERS {
                let mut buf: AudioQueueBufferRef = ptr::null_mut();
                // SAFETY: live queue; valid out slot.
                let st =
                    unsafe { AudioQueueAllocateBuffer(queue, BUF_BYTES, NonNull::from(&mut buf)) };
                if st != 0 || buf.is_null() {
                    // SAFETY: dispose the half-built queue (buffers go with it).
                    unsafe { AudioQueueDispose(queue, true) };
                    return Err(hw());
                }
                free.push(BufPtr(buf));
            }
        }
        self.state = Some(SinkState {
            queue,
            shared,
            started: false,
        });
        Ok(())
    }

    /// Queue a whole PCM payload, chunking through the buffer pool and blocking
    /// on the queue's pace when all buffers are in flight.
    fn write_all(&mut self, pcm: &[u8]) -> Result<(), G2gError> {
        let bps = bytes_per_sample(self.format).ok_or(G2gError::CapsMismatch)?;
        let frame_bytes = bps * self.channels as usize;
        if frame_bytes == 0 || pcm.len() < frame_bytes {
            return Ok(());
        }
        let state = self.state.as_mut().ok_or(G2gError::NotConfigured)?;
        let mut off = 0usize;
        while off < pcm.len() {
            let buf = {
                let mut free = state.shared.free.lock().unwrap();
                loop {
                    if let Some(b) = free.pop() {
                        break b;
                    }
                    let (guard, timeout) = state
                        .shared
                        .cv
                        .wait_timeout(free, IO_TIMEOUT)
                        .map_err(|_| hw())?;
                    free = guard;
                    if timeout.timed_out() && free.is_empty() {
                        // No buffer came back within the deadline: the device
                        // is wedged; surface rather than spin.
                        return Err(hw());
                    }
                }
            };
            // SAFETY: the buffer is ours until re-enqueued (came off the free
            // list); capacity was allocated as BUF_BYTES.
            let n = unsafe {
                let b = &mut *buf.0;
                let cap = b.mAudioDataBytesCapacity as usize;
                let n = cap.min(pcm.len() - off);
                // Whole frames only, so a chunk boundary never splits a sample.
                let n = (n / frame_bytes) * frame_bytes;
                core::ptr::copy_nonoverlapping(
                    pcm[off..].as_ptr(),
                    b.mAudioData.as_ptr() as *mut u8,
                    n,
                );
                b.mAudioDataByteSize = n as u32;
                n
            };
            if n == 0 {
                return Err(G2gError::CapsMismatch);
            }
            // SAFETY: live queue; the buffer belongs to it.
            let st = unsafe { AudioQueueEnqueueBuffer(state.queue, buf.0, 0, ptr::null()) };
            if st != 0 {
                return Err(hw());
            }
            off += n;
            self.rendered += (n / frame_bytes) as u64;
            if !state.started {
                // Start only once data is queued (an empty queue underruns).
                // SAFETY: live queue; null start time = now.
                let st = unsafe { AudioQueueStart(state.queue, ptr::null()) };
                if st != 0 {
                    return Err(hw());
                }
                state.started = true;
            }
        }
        Ok(())
    }
}

impl AsyncElement for CoreAudioSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        pcm_params(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::LegacySink(Box::new(|c: &Caps| {
            pcm_params(c)?;
            Ok(c.clone())
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, channels, sample_rate) = pcm_params(absolute_caps)?;
        self.format = format;
        self.channels = channels;
        self.sample_rate = sample_rate;
        self.open()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.write_all(slice.as_slice())?;
                }
                PipelinePacket::Eos => {
                    if let Some(st) = self.state.as_ref() {
                        if st.started {
                            // Drain what is queued, then stop asynchronously.
                            // SAFETY: live queue.
                            unsafe {
                                AudioQueueFlush(st.queue);
                                AudioQueueStop(st.queue, false);
                            }
                        }
                    }
                }
                // PCM caps are fixed at configure; a mid-stream change would need
                // a queue rebuild (not in v1). Control packets are consumed.
                PipelinePacket::CapsChanged(_)
                | PipelinePacket::Flush
                | PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for CoreAudioSink {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |format| Caps::Audio {
            format,
            channels: 2,
            sample_rate: 48_000,
        };
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(Vec::from([
            pcm(AudioFormat::PcmS16Le),
            pcm(AudioFormat::PcmF32Le),
        ])))])
    }
}

// ---------------------------------------------------------------------------
// Source (capture)
// ---------------------------------------------------------------------------

/// Shared with the capture callback: filled chunks the run loop drains.
#[derive(Debug, Default)]
struct SrcShared {
    filled: Mutex<VecDeque<Vec<u8>>>,
    cv: Condvar,
}

/// Capture callback: copy the delivered bytes out and re-queue the buffer.
///
/// SAFETY: invoked by the queue's thread with the `*mut SrcShared` refcon
/// passed at create, which (boxed in `SrcState`) outlives the queue.
unsafe extern "C-unwind" fn src_callback(
    user: *mut c_void,
    aq: AudioQueueRef,
    buf: AudioQueueBufferRef,
    _start_time: NonNull<AudioTimeStamp>,
    _num_packets: u32,
    _packet_descs: *const AudioStreamPacketDescription,
) {
    // SAFETY: see above; the buffer is ours until re-enqueued.
    let shared = unsafe { &*(user as *const SrcShared) };
    let bytes = unsafe {
        let b = &*buf;
        core::slice::from_raw_parts(
            b.mAudioData.as_ptr() as *const u8,
            b.mAudioDataByteSize as usize,
        )
    };
    if !bytes.is_empty() {
        shared.filled.lock().unwrap().push_back(bytes.to_vec());
        shared.cv.notify_one();
    }
    // SAFETY: hand the buffer back to the live queue for the next fill.
    unsafe {
        AudioQueueEnqueueBuffer(aq, buf, 0, ptr::null());
    }
}

struct SrcState {
    queue: AudioQueueRef,
    shared: Box<SrcShared>,
}

impl Drop for SrcState {
    fn drop(&mut self) {
        // SAFETY: live queue; immediate dispose stops callbacks before
        // returning, so `shared` (dropped after this) is not used again.
        unsafe {
            AudioQueueDispose(self.queue, true);
        }
    }
}

impl core::fmt::Debug for SrcState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SrcState").finish_non_exhaustive()
    }
}

/// Captures interleaved PCM from the default Core Audio input device.
#[derive(Debug)]
pub struct CoreAudioSrc {
    sample_rate: u32,
    channels: u8,
    target_buffers: u64,
    state: Option<SrcState>,
    configured: bool,
}

// SAFETY: same single-thread-executor contract as `CoreAudioSink`.
unsafe impl Send for CoreAudioSrc {}

impl CoreAudioSrc {
    /// A capture source at `sample_rate` / `channels` (S16LE), emitting
    /// `target_buffers` buffers then EOS (`u64::MAX` = capture until stopped).
    pub fn new(sample_rate: u32, channels: u8, target_buffers: u64) -> Self {
        Self {
            sample_rate: sample_rate.max(1),
            channels: channels.max(1),
            target_buffers,
            state: None,
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
        let mut desc =
            asbd(AudioFormat::PcmS16Le, self.channels, self.sample_rate).ok_or_else(hw)?;
        let mut shared = Box::new(SrcShared::default());
        let mut queue: AudioQueueRef = ptr::null_mut();
        // SAFETY: same contract as the sink's create call.
        let st = unsafe {
            AudioQueueNewInput(
                NonNull::from(&mut desc),
                Some(src_callback),
                shared.as_mut() as *mut SrcShared as *mut c_void,
                None,
                None,
                0,
                NonNull::from(&mut queue),
            )
        };
        if st != 0 || queue.is_null() {
            return Err(hw());
        }
        for _ in 0..SRC_BUFFERS {
            let mut buf: AudioQueueBufferRef = ptr::null_mut();
            // SAFETY: live queue; valid out slot; the buffer is enqueued for
            // capture right away.
            let st = unsafe {
                let st = AudioQueueAllocateBuffer(queue, BUF_BYTES, NonNull::from(&mut buf));
                if st == 0 && !buf.is_null() {
                    AudioQueueEnqueueBuffer(queue, buf, 0, ptr::null())
                } else {
                    st
                }
            };
            if st != 0 {
                // SAFETY: dispose the half-built queue.
                unsafe { AudioQueueDispose(queue, true) };
                return Err(hw());
            }
        }
        self.state = Some(SrcState { queue, shared });
        Ok(())
    }
}

impl SourceLoop for CoreAudioSrc {
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
        let state = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        // SAFETY: live queue; null start time = now.
        let st = unsafe { AudioQueueStart(state.queue, ptr::null()) };
        if st != 0 {
            return Err(hw());
        }
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
                    let state = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
                    let mut filled = state.shared.filled.lock().unwrap();
                    loop {
                        if let Some(c) = filled.pop_front() {
                            idle = 0;
                            break Some(c);
                        }
                        let (guard, timeout) = state
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
                        // The device delivered nothing for several deadlines:
                        // dead capture, surface rather than hang.
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

            if let Some(st) = self.state.as_ref() {
                // SAFETY: live queue; stop immediately, capture is done.
                unsafe {
                    AudioQueueStop(st.queue, true);
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

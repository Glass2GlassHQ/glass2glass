use core::future::poll_fn;
use core::task::{Poll, Waker};

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::Target;
use g2g_core::memory::SystemSlice;
use g2g_core::time::ticks_to_ns;
use g2g_core::{
    CaptureAnchor, ClockSync, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
};

// 64 periods is 640 ms at the usual 10 ms period
const CAPTURE_QUEUE_DEPTH: usize = 64;

#[cfg(any(target_os = "linux", target_os = "windows"))]
const DEVICE_OPEN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

// what a device thread that never reported back is surfaced as
#[cfg(any(target_os = "linux", target_os = "windows"))]
const DEVICE_OPEN_TIMED_OUT_CODE: i32 = -1;

#[derive(Debug)]
struct CapturedChunk {
    first_frame: u64,
    frames: u64,
    bytes: Vec<u8>,
}

#[derive(Debug, Default)]
struct QueueState {
    chunks: VecDeque<CapturedChunk>,
    // counts dropped chunks too, so a drop leaves a gap in the stamps
    next_frame: u64,
    waker: Option<Waker>,
    senders: usize,
    receiver_dropped: bool,
    failure: Option<G2gError>,
}

// a full queue drops its oldest chunk, as gst audiobasesrc skips ahead when downstream lags
#[derive(Debug)]
pub(crate) struct CaptureSender {
    state: Arc<Mutex<QueueState>>,
    frame_bytes: usize,
}

#[derive(Debug)]
pub(crate) struct CaptureReceiver {
    state: Arc<Mutex<QueueState>>,
}

pub(crate) fn capture_queue(
    frame_bytes: usize,
) -> Result<(CaptureSender, CaptureReceiver), G2gError> {
    if frame_bytes == 0 {
        return Err(G2gError::CapsMismatch);
    }
    let state = Arc::new(Mutex::new(QueueState {
        senders: 1,
        ..QueueState::default()
    }));
    let sender = CaptureSender {
        state: Arc::clone(&state),
        frame_bytes,
    };
    Ok((sender, CaptureReceiver { state }))
}

fn wake(waker: Option<Waker>) {
    if let Some(waker) = waker {
        waker.wake();
    }
}

impl CaptureSender {
    // false once the receiver is gone
    pub(crate) fn push(&self, bytes: Vec<u8>) -> bool {
        let frames = (bytes.len() / self.frame_bytes) as u64;
        let mut state = self.state.lock().unwrap();
        if state.receiver_dropped {
            return false;
        }
        let first_frame = state.next_frame;
        state.next_frame = first_frame.saturating_add(frames);
        if state.chunks.len() == CAPTURE_QUEUE_DEPTH {
            state.chunks.pop_front();
        }
        state.chunks.push_back(CapturedChunk {
            first_frame,
            frames,
            bytes,
        });
        let waker = state.waker.take();
        drop(state);
        wake(waker);
        true
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn pushed_frames(&self) -> u64 {
        self.state.lock().unwrap().next_frame
    }

    #[cfg(any(test, feature = "pipewire", target_os = "macos", target_os = "android"))]
    pub(crate) fn fail(&self, error: G2gError) {
        let mut state = self.state.lock().unwrap();
        state.failure = Some(error);
        let waker = state.waker.take();
        drop(state);
        wake(waker);
    }
}

impl Clone for CaptureSender {
    fn clone(&self) -> Self {
        self.state.lock().unwrap().senders += 1;
        Self {
            state: Arc::clone(&self.state),
            frame_bytes: self.frame_bytes,
        }
    }
}

impl Drop for CaptureSender {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        state.senders -= 1;
        let waker = state.waker.take();
        drop(state);
        wake(waker);
    }
}

impl CaptureReceiver {
    async fn next(&mut self) -> Result<Option<CapturedChunk>, G2gError> {
        poll_fn(|cx| {
            let mut state = self.state.lock().unwrap();
            if let Some(chunk) = state.chunks.pop_front() {
                return Poll::Ready(Ok(Some(chunk)));
            }
            if let Some(error) = state.failure.take() {
                return Poll::Ready(Err(error));
            }
            if state.senders == 0 {
                return Poll::Ready(Ok(None));
            }
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        })
        .await
    }
}

impl Drop for CaptureReceiver {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        state.receiver_dropped = true;
        state.chunks.clear();
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub(crate) fn wait_for_device(
    ready: &std::sync::mpsc::Receiver<Result<(), i32>>,
) -> Result<(), i32> {
    ready
        .recv_timeout(DEVICE_OPEN_TIMEOUT)
        .unwrap_or(Err(DEVICE_OPEN_TIMED_OUT_CODE))
}

#[derive(Debug)]
pub(crate) struct CaptureLoopSettings {
    pub(crate) log_category: &'static str,
    pub(crate) sample_rate: u32,
    pub(crate) clock_sync: Option<ClockSync>,
    // u64::MAX for no limit
    pub(crate) buffer_limit: u64,
}

// pushes Eos only if downstream is still open, after stop_device has run
pub(crate) async fn run_capture(
    mut receiver: CaptureReceiver,
    settings: CaptureLoopSettings,
    out: &mut dyn OutputSink,
    stop_device: impl FnOnce(),
) -> Result<u64, G2gError> {
    let rate = settings.sample_rate;
    if rate == 0 {
        drop(receiver);
        stop_device();
        return Err(G2gError::CapsMismatch);
    }
    let log = Target::category(settings.log_category);
    let mut anchor = CaptureAnchor::new();
    let mut expected_frame = 0u64;
    let mut sequence = 0u64;
    let mut downstream_open = true;
    let mut failure = None;
    while sequence < settings.buffer_limit {
        let chunk = match receiver.next().await {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(error) => {
                failure = Some(error);
                break;
            }
        };
        if chunk.frames == 0 {
            continue;
        }
        if chunk.first_frame > expected_frame {
            g2g_core::g2g_warn!(
                log,
                "can't record audio fast enough, dropped {} frames",
                chunk.first_frame - expected_frame
            );
        }
        expected_frame = chunk.first_frame + chunk.frames;
        let start_ns = ticks_to_ns(chunk.first_frame, rate);
        let duration_ns = ticks_to_ns(expected_frame, rate) - start_ns;
        // the chunk finished capturing when it arrived, so its span is the lead
        let pts_ns = match &settings.clock_sync {
            Some(sync) => anchor.stamp(sync, start_ns, duration_ns),
            None => start_ns,
        };
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(chunk.bytes.into_boxed_slice())),
            timing: FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns,
                capture_ns: pts_ns,
                arrival_ns: g2g_core::metrics::monotonic_ns(),
                keyframe: false,
            },
            sequence,
            meta: Default::default(),
        };
        if out.push(PipelinePacket::DataFrame(frame)).await.is_err() {
            downstream_open = false;
            break;
        }
        sequence += 1;
    }

    drop(receiver);
    stop_device();

    if let Some(error) = failure {
        return Err(error);
    }
    if downstream_open {
        out.push(PipelinePacket::Eos).await?;
    }
    Ok(sequence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::task::Context;
    use g2g_core::runtime::block_on;
    use g2g_core::PushOutcome;

    const RATE_44K: u32 = 44_100;
    // 10 ms at 44.1 kHz, where whole-ns-per-frame stamping loses 0.7 ns a frame
    const FRAMES_PER_CHUNK: usize = 441;
    const CHUNK_NS: u64 = 10_000_000;
    const MONO_S16_FRAME_BYTES: usize = 2;

    #[derive(Default)]
    struct TimingSink {
        stamps: Vec<(u64, u64)>,
        eos: bool,
    }

    impl OutputSink for TimingSink {
        fn poll_push(
            &mut self,
            _cx: &mut Context<'_>,
            packet: &mut Option<PipelinePacket>,
        ) -> Poll<Result<PushOutcome, G2gError>> {
            match packet.take().expect("a packet") {
                PipelinePacket::DataFrame(frame) => self
                    .stamps
                    .push((frame.timing.pts_ns, frame.timing.duration_ns)),
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn settings(buffer_limit: u64) -> CaptureLoopSettings {
        CaptureLoopSettings {
            log_category: "audiocapture",
            sample_rate: RATE_44K,
            clock_sync: None,
            buffer_limit,
        }
    }

    fn chunk() -> Vec<u8> {
        vec![0u8; FRAMES_PER_CHUNK * MONO_S16_FRAME_BYTES]
    }

    #[test]
    fn stamps_do_not_drift_at_44_1_khz() {
        let (sender, receiver) = capture_queue(MONO_S16_FRAME_BYTES).expect("queue");
        for _ in 0..CAPTURE_QUEUE_DEPTH {
            assert!(sender.push(chunk()));
        }
        drop(sender);
        let mut sink = TimingSink::default();
        let pushed =
            block_on(run_capture(receiver, settings(u64::MAX), &mut sink, || {})).expect("capture");
        assert_eq!(pushed, CAPTURE_QUEUE_DEPTH as u64);
        for (index, &(pts_ns, duration_ns)) in sink.stamps.iter().enumerate() {
            assert_eq!(pts_ns, index as u64 * CHUNK_NS);
            assert_eq!(duration_ns, CHUNK_NS);
        }
        assert!(sink.eos);
    }

    #[test]
    fn a_full_queue_drops_the_oldest_chunk_and_leaves_a_gap() {
        let dropped = 2;
        let (sender, receiver) = capture_queue(MONO_S16_FRAME_BYTES).expect("queue");
        for _ in 0..CAPTURE_QUEUE_DEPTH + dropped {
            assert!(sender.push(chunk()));
        }
        drop(sender);
        let mut sink = TimingSink::default();
        let pushed =
            block_on(run_capture(receiver, settings(u64::MAX), &mut sink, || {})).expect("capture");
        assert_eq!(pushed, CAPTURE_QUEUE_DEPTH as u64);
        assert_eq!(sink.stamps[0].0, dropped as u64 * CHUNK_NS);
    }

    #[test]
    fn a_device_failure_ends_the_capture_without_eos() {
        let (sender, receiver) = capture_queue(MONO_S16_FRAME_BYTES).expect("queue");
        assert!(sender.push(chunk()));
        sender.fail(G2gError::CapsMismatch);
        let mut stopped = false;
        let mut sink = TimingSink::default();
        let result = block_on(run_capture(receiver, settings(u64::MAX), &mut sink, || {
            stopped = true
        }));
        assert!(result.is_err());
        assert!(stopped);
        assert_eq!(sink.stamps.len(), 1);
        assert!(!sink.eos);
    }

    #[test]
    fn the_buffer_limit_stops_the_capture_and_the_device_side_sees_it() {
        let (sender, receiver) = capture_queue(MONO_S16_FRAME_BYTES).expect("queue");
        assert!(sender.push(chunk()));
        assert!(sender.push(chunk()));
        let mut sink = TimingSink::default();
        let pushed =
            block_on(run_capture(receiver, settings(1), &mut sink, || {})).expect("capture");
        assert_eq!(pushed, 1);
        assert!(sink.eos);
        assert!(!sender.push(chunk()));
    }
}

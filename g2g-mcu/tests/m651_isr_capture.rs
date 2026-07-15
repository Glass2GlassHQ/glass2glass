//! M651: the ISR/DMA-driven capture concurrency model. A real MCU captures in
//! interrupt context (a DMA-completion ISR) while the pipeline runs in the main
//! context; the two hand frames across that boundary through `SpscFrameRing`.
//! These host tests exercise the hand-off under genuine concurrency (a producer
//! thread standing in for the ISR), complementing the on-target proof
//! (`examples/g2g-qemu`'s `isr_capture` bin, where a SysTick interrupt is the
//! real producer). Two properties: (1) with the producer paced, every captured
//! frame reaches the pipeline in capture order (lossless, in-order); (2) when the
//! producer outruns the consumer, frames are dropped and counted, never
//! corrupted or reordered (bounded back-pressure).

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::{step_source_sink, SpscCaptureSrc, SpscFrameRing, StaticSink, Step};

const BYTES: usize = 8;
const FRAME_NS: u64 = 1_000_000;

/// Stamp a frame with its capture sequence number (first 4 bytes, LE).
fn stamp(buf: &mut [u8; BYTES], seq: u32) {
    buf[0..4].copy_from_slice(&seq.to_le_bytes());
}

fn frame_seq(frame: &Frame) -> u32 {
    let MemoryDomain::System(s) = &frame.domain else {
        panic!("capture frames are System-domain");
    };
    let b = s.as_slice();
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// A sink that asserts every frame arrives in strict capture order (no loss, no
/// reorder) and counts them.
struct InOrderSink {
    next_expected: u32,
    count: u32,
    ordered: bool,
}

impl StaticSink for InOrderSink {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        if frame_seq(&frame) != self.next_expected {
            self.ordered = false;
        }
        self.next_expected += 1;
        self.count += 1;
        Ok(())
    }
}

#[test]
fn isr_style_producer_feeds_the_pipeline_lossless_and_in_order() {
    // The ring is the ISR-to-pipeline hand-off buffer; N=4 slots (3 usable) is a
    // realistic double-buffer-plus-slack. TARGET frames far exceed N, forcing
    // many wraps under concurrency.
    let ring: SpscFrameRing<4, BYTES> = SpscFrameRing::new();
    const TARGET: u32 = 500;

    thread::scope(|s| {
        // Producer (stands in for the DMA-completion ISR): publishes TARGET
        // frames, each stamped with its capture sequence. It retries on a full
        // ring (paces to the consumer) so this run is lossless.
        s.spawn(|| {
            for seq in 0..TARGET {
                while ring.produce(|b| stamp(b, seq)).is_err() {
                    thread::yield_now(); // ring full: let the consumer drain
                }
            }
        });

        // Consumer: drive the pipeline (capture source -> in-order sink) one
        // frame at a time, idling (here a yield; WFI on hardware) while the ring
        // is empty. `step_source_sink` is the same frame-at-a-time runner a C/RTOS
        // superloop uses.
        let mut src = SpscCaptureSrc::new(&ring, thread::yield_now, FRAME_NS)
            .with_frame_limit(TARGET);
        let mut sink = InOrderSink { next_expected: 0, count: 0, ordered: true };
        loop {
            match step_source_sink(&mut src, &mut sink).expect("clean step") {
                Step::Advanced => {}
                Step::Eos => break,
                Step::Pending => unreachable!("stages are synchronous"),
            }
        }
        // Lossless + in-order is the sink's count and ordering: all TARGET frames
        // arrived, each in capture sequence. (The producer retries on a full ring
        // rather than dropping, so `overruns` here counts transient full-ring
        // retries, not lost frames; the no-loss guarantee is the count below.)
        assert!(sink.ordered, "every frame reached the pipeline in capture order");
        assert_eq!(sink.count, TARGET, "no captured frame was lost");
    });
}

#[test]
fn a_fast_producer_drops_and_counts_frames_without_reorder() {
    // Back-pressure: the producer (ISR) does NOT pace, dropping on a full ring,
    // while the consumer is deliberately slow. Some frames are lost, but the ones
    // delivered are a strictly increasing subsequence (in order, uncorrupted),
    // and the accounting balances: delivered + overruns == attempts.
    let ring: SpscFrameRing<4, BYTES> = SpscFrameRing::new();
    const ATTEMPTS: u32 = 2000;
    let done = AtomicBool::new(false);

    thread::scope(|s| {
        s.spawn(|| {
            for seq in 0..ATTEMPTS {
                // ISR semantics: try once, drop (count) if full, never block.
                let _ = ring.produce(|b| stamp(b, seq));
            }
            done.store(true, Ordering::Release);
        });

        // Slow consumer draining the ring directly (a busy pipeline).
        let mut delivered: u32 = 0;
        let mut last: Option<u32> = None;
        let mut ordered = true;
        loop {
            match ring.borrow() {
                Some(slice) => {
                    let b = slice.as_slice();
                    let seq = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    if let Some(prev) = last {
                        if seq <= prev {
                            ordered = false; // a reorder or a stale/duplicated slot
                        }
                    }
                    last = Some(seq);
                    delivered += 1;
                    drop(slice);
                    ring.release();
                    // Be slow, so the producer overruns.
                    for _ in 0..50 {
                        std::hint::spin_loop();
                    }
                }
                None => {
                    if done.load(Ordering::Acquire) && ring.is_empty() {
                        break;
                    }
                    thread::yield_now();
                }
            }
        }

        let overruns = ring.overruns();
        assert!(ordered, "delivered frames are strictly increasing (no reorder/corruption)");
        assert!(overruns > 0, "a fast producer over a slow consumer must overrun (got {overruns})");
        assert_eq!(delivered + overruns, ATTEMPTS, "every attempt was either delivered or counted as an overrun");
    });
}

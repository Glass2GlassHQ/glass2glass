//! M664 `audiomixer` PTS-based alignment. Drives the real `AudioMixer` element
//! through `MultiInputElement::process`, checking that buffers land on a shared
//! output timeline by PTS (not arrival): overlapping ranges sum, a PTS gap mixes
//! as silence, a finished input keeps contributing silence while others run, and
//! zero / duplicate PTS append continuously.
//!
//! Mono S16LE at 1000 Hz: one sample-frame is 1 ms, so a PTS of `k` ms maps to
//! sample-frame `k` and the emitted PTS/duration read back in whole ms.

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, Caps, G2gError, MemoryDomain, MultiInputElement, OutputSink, PipelinePacket,
};
use g2g_plugins::audiomixer::AudioMixer;

const MS: u64 = 1_000_000;

fn out_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 1,
        sample_rate: 1000,
    }
}

/// A mono S16LE frame of `samples` at `pts_ms`.
fn frame(pts_ms: u64, samples: &[i16]) -> PipelinePacket {
    let mut bytes = alloc_bytes(samples.len() * 2);
    for (s, chunk) in samples.iter().zip(bytes.chunks_exact_mut(2)) {
        chunk.copy_from_slice(&s.to_le_bytes());
    }
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes)),
        timing: FrameTiming {
            pts_ns: pts_ms * MS,
            ..FrameTiming::default()
        },
        sequence: 0,
        meta: Default::default(),
    })
}

fn alloc_bytes(n: usize) -> Box<[u8]> {
    vec![0u8; n].into_boxed_slice()
}

#[derive(Default)]
struct Collect {
    /// (pts_ns, duration_ns, samples) per emitted DataFrame.
    frames: Vec<(u64, u64, Vec<i16>)>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    let samples = s
                        .as_slice()
                        .chunks_exact(2)
                        .map(|c| i16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    self.frames
                        .push((f.timing.pts_ns, f.timing.duration_ns, samples));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

async fn feed(mixer: &mut AudioMixer, input: usize, packet: PipelinePacket, sink: &mut Collect) {
    mixer.process(input, packet, sink).await.expect("process");
}

#[tokio::test]
async fn overlapping_ranges_sum_and_offset_is_silence() {
    let mut m = AudioMixer::new(2, out_caps());
    let mut sink = Collect::default();

    // input 0 covers frames 0..4; input 1 arrives late at 2 ms covering 2..4.
    feed(&mut m, 0, frame(0, &[100, 100, 100, 100]), &mut sink).await;
    // input 1 has delivered nothing yet, so nothing is safe to emit.
    assert!(
        sink.frames.is_empty(),
        "cannot emit before every input covers the span"
    );

    feed(&mut m, 1, frame(2, &[10, 10]), &mut sink).await;

    assert_eq!(
        sink.frames.len(),
        1,
        "one contiguous span emits once both inputs cover it"
    );
    let (pts, dur, samples) = &sink.frames[0];
    assert_eq!(*pts, 0);
    assert_eq!(*dur, 4 * MS);
    // frames 0,1: input 0 alone (input 1 silent); frames 2,3: summed.
    assert_eq!(samples, &[100, 100, 110, 110]);
}

#[tokio::test]
async fn pts_gap_within_an_input_is_silence() {
    let mut m = AudioMixer::new(1, out_caps());
    let mut sink = Collect::default();

    feed(&mut m, 0, frame(0, &[50]), &mut sink).await;
    // Jump to 2 ms, leaving frame 1 uncovered.
    feed(&mut m, 0, frame(2, &[70]), &mut sink).await;

    assert_eq!(sink.frames.len(), 2);
    assert_eq!(sink.frames[0], (0, MS, vec![50]));
    // frame 1 is the silent gap, frame 2 the new buffer.
    assert_eq!(sink.frames[1], (MS, 2 * MS, vec![0, 70]));
}

#[tokio::test]
async fn early_eos_input_mixed_as_silence_while_others_run() {
    let mut m = AudioMixer::new(2, out_caps());
    let mut sink = Collect::default();

    feed(&mut m, 0, frame(0, &[100, 100]), &mut sink).await;
    feed(&mut m, 1, frame(0, &[10]), &mut sink).await;
    // Both cover frame 0 -> that span emits (summed).
    assert_eq!(sink.frames.len(), 1);
    assert_eq!(sink.frames[0], (0, MS, vec![110]));

    // input 1 ends; input 0's remaining frame 1 now emits with input 1 silent.
    feed(&mut m, 1, PipelinePacket::Eos, &mut sink).await;
    assert_eq!(sink.frames.len(), 2);
    assert_eq!(sink.frames[1], (MS, MS, vec![100]));

    // input 0 keeps going; the finished input contributes nothing.
    feed(&mut m, 0, frame(2, &[100]), &mut sink).await;
    assert_eq!(sink.frames.len(), 3);
    assert_eq!(sink.frames[2], (2 * MS, MS, vec![100]));
}

#[tokio::test]
async fn zero_and_duplicate_pts_append_continuously() {
    let mut m = AudioMixer::new(1, out_caps());
    let mut sink = Collect::default();

    feed(&mut m, 0, frame(0, &[1, 2]), &mut sink).await;
    // Same PTS again: append after the input's own position, do not overwrite.
    feed(&mut m, 0, frame(0, &[3, 4]), &mut sink).await;

    assert_eq!(sink.frames.len(), 2);
    assert_eq!(sink.frames[0], (0, 2 * MS, vec![1, 2]));
    assert_eq!(sink.frames[1], (2 * MS, 2 * MS, vec![3, 4]));
}

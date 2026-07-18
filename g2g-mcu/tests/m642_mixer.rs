//! M642: the saturating Q15-gain mixer and the const-arity fan-in runner it
//! rides on. The mixer's spec is its own math (unlike the wire codecs there
//! is no external peer to be bit-exact against), so the oracle is an i64
//! reference computed independently in the test, plus the corners that
//! distinguish a correct fixed-point mixer from a plausible one: positive and
//! negative saturation, the double-`i16::MIN` product sum that overflows an
//! i32 accumulator, round-half-up, and exact cancellation of equal-and-
//! opposite gains.

mod util;

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{run_sources_fanin_sink, StaticFanIn2, StaticSink, StaticSource};
use g2g_mcu::mixer::mix_q15;
use g2g_mcu::Mixer;
use util::{block_on, frame_of, le_bytes, payload};

/// The reference the element must match: unbounded integer mix, then round
/// and clamp, written independently of the `const fn` under test.
fn mix_ref(a: i16, b: i16, ga: i16, gb: i16) -> i16 {
    let acc = i64::from(a) * i64::from(ga) + i64::from(b) * i64::from(gb);
    let v = (acc + (1 << 14)) >> 15;
    v.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

#[test]
fn mix_matches_reference_across_the_sample_domain() {
    // Every `a` value against a spread of partners and gain pairs: full
    // 65536-sample coverage on one axis, corners on the others.
    let partners = [-32768i16, -32767, -1, 0, 1, 12345, 32766, 32767];
    let gains = [
        (16384i16, 16384i16),
        (32767, 32767),
        (-32768, -32768),
        (32767, -32768),
        (0, 0),
    ];
    for a in i16::MIN..=i16::MAX {
        for &b in &partners {
            for &(ga, gb) in &gains {
                assert_eq!(
                    mix_q15(a, b, ga, gb),
                    mix_ref(a, b, ga, gb),
                    "a={a} b={b} ga={ga} gb={gb}"
                );
            }
        }
    }
}

#[test]
fn mix_corners() {
    // Positive and negative saturation.
    assert_eq!(mix_q15(32767, 32767, 32767, 32767), 32767, "positive clip");
    assert_eq!(
        mix_q15(-32768, -32768, 32767, 32767),
        -32768,
        "negative clip"
    );
    // Both products at +2^30: the sum is 2^31, one past i32::MAX, so this
    // input distinguishes the i64 accumulator from an overflowing i32 one.
    assert_eq!(
        mix_q15(-32768, -32768, -32768, -32768),
        32767,
        "i32 accumulator would overflow"
    );
    // Round-half-up: 0.5 in the Q15 remainder rounds away from zero (up).
    assert_eq!(mix_q15(1, 0, 16384, 0), 1, "half rounds up");
    // Equal-and-opposite gains cancel identical inputs exactly.
    for x in [-32768i16, -12345, 0, 1, 32767] {
        assert_eq!(
            mix_q15(x, x, 16384, -16384),
            0,
            "exact cancellation at x={x}"
        );
    }
    // Half + half of the same signal reconstructs it exactly (0.5 is exact
    // in Q15, unlike unity).
    for x in [-32768i16, -3, 0, 5, 32767] {
        assert_eq!(
            mix_q15(x, x, 16384, 16384),
            x,
            "0.5 + 0.5 is identity at x={x}"
        );
    }
}

/// Emits `chunks`-sized frames of `samples` from a ring, PTS = frame index.
struct PcmSource<'r, const B: usize> {
    ring: &'r StaticLendRing<1, B>,
    samples: Vec<i16>,
    chunk: usize,
    idx: usize,
}

impl<const B: usize> StaticSource for PcmSource<'_, B> {
    async fn next(&mut self) -> Result<Option<Frame>, G2gError> {
        let start = self.idx * self.chunk;
        if start >= self.samples.len() {
            return Ok(None);
        }
        let end = (start + self.chunk).min(self.samples.len());
        let frame = frame_of(
            self.ring,
            &le_bytes(&self.samples[start..end]),
            self.idx as u64,
            self.idx as u64,
        );
        self.idx += 1;
        Ok(Some(frame))
    }
}

/// Copies every payload (and the frame timing) out of the pipeline.
struct Collect {
    frames: Vec<(u64, Vec<u8>)>,
}

impl StaticSink for Collect {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        self.frames
            .push((frame.timing.pts_ns, payload(&frame).to_vec()));
        Ok(())
    }
}

#[test]
fn fanin_graph_mixes_two_streams() {
    // Two deterministic signals long enough to cross several frames.
    let sig_a: Vec<i16> = (0..96)
        .map(|i| ((i * 977) % 30000 - 15000) as i16)
        .collect();
    let sig_b: Vec<i16> = (0..96)
        .map(|i| ((i * 331) % 24000 - 12000) as i16)
        .collect();
    let ring_a: StaticLendRing<1, 64> = StaticLendRing::new();
    let ring_b: StaticLendRing<1, 64> = StaticLendRing::new();
    let out_ring: StaticLendRing<1, 64> = StaticLendRing::new();
    let src_a = PcmSource {
        ring: &ring_a,
        samples: sig_a.clone(),
        chunk: 16,
        idx: 0,
    };
    let src_b = PcmSource {
        ring: &ring_b,
        samples: sig_b.clone(),
        chunk: 16,
        idx: 0,
    };
    // SAFETY: the ring outlives every frame in this test.
    let mixer = unsafe { Mixer::with_ring(20000, -9000, &out_ring) };
    let mut sink = Collect { frames: Vec::new() };
    block_on(run_sources_fanin_sink(src_a, src_b, mixer, &mut sink)).expect("graph runs to EOS");

    assert_eq!(sink.frames.len(), 6, "96 samples in 16-sample frames");
    let mixed: Vec<i16> = sink
        .frames
        .iter()
        .flat_map(|(_, p)| p.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])))
        .collect();
    let expect: Vec<i16> = sig_a
        .iter()
        .zip(&sig_b)
        .map(|(&a, &b)| mix_ref(a, b, 20000, -9000))
        .collect();
    assert_eq!(
        mixed, expect,
        "every output sample is the Q15 mix of its input pair"
    );
    // Input `a` is the timing master: output PTS follows it frame for frame.
    let pts: Vec<u64> = sink.frames.iter().map(|(pts, _)| *pts).collect();
    assert_eq!(pts, [0, 1, 2, 3, 4, 5], "timing inherited from input a");
}

#[test]
fn fanin_ends_at_the_shorter_source() {
    let ring_a: StaticLendRing<1, 64> = StaticLendRing::new();
    let ring_b: StaticLendRing<1, 64> = StaticLendRing::new();
    let out_ring: StaticLendRing<1, 64> = StaticLendRing::new();
    let src_a = PcmSource {
        ring: &ring_a,
        samples: vec![0i16; 64],
        chunk: 16,
        idx: 0,
    };
    let src_b = PcmSource {
        ring: &ring_b,
        samples: vec![0i16; 32],
        chunk: 16,
        idx: 0,
    };
    // SAFETY: the ring outlives every frame in this test.
    let mixer = unsafe { Mixer::with_ring(16384, 16384, &out_ring) };
    let mut sink = Collect { frames: Vec::new() };
    block_on(run_sources_fanin_sink(src_a, src_b, mixer, &mut sink)).expect("clean EOS");
    assert_eq!(
        sink.frames.len(),
        2,
        "stream ends when the shorter input ends"
    );
}

#[test]
fn mixer_rejects_malformed_pairs() {
    let ring: StaticLendRing<2, 64> = StaticLendRing::new();
    let out_ring: StaticLendRing<1, 64> = StaticLendRing::new();
    // SAFETY: the ring outlives every frame in this test.
    let mut mixer = unsafe { Mixer::with_ring(16384, 16384, &out_ring) };

    // Length mismatch between the inputs.
    let a = frame_of(&ring, &le_bytes(&[1, 2, 3]), 0, 0);
    let b = frame_of(&ring, &le_bytes(&[1, 2]), 0, 0);
    assert!(
        matches!(block_on(mixer.process2(a, b)), Err(G2gError::CapsMismatch)),
        "unequal payloads are a caps bug, not a truncation"
    );

    // A payload that is not whole 16-bit samples.
    let a = frame_of(&ring, &[1u8, 2, 3], 0, 0);
    let b = frame_of(&ring, &[4u8, 5, 6], 0, 0);
    assert!(
        matches!(block_on(mixer.process2(a, b)), Err(G2gError::CapsMismatch)),
        "odd byte count is rejected"
    );
}

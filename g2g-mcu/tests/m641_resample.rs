//! M641: the fixed-point polyphase resampler. No external implementation is
//! bit-comparable (every resampler picks its own filter), so the oracle is
//! analytic ground truth: the generated tables' invariants (exact per-phase
//! unity sums, accumulator headroom), exact DC gain through the elements,
//! in-band tone SNR by least-squares sine fit (sidesteps fractional group
//! delay), alias rejection when decimating, and chunked streaming being
//! byte-identical to one-shot processing (the state carry across frames).

mod util;

use g2g_core::error::G2gError;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticTransform;
use g2g_mcu::resample::{table_for, MAX_TAPS};
use g2g_mcu::resample_tables::COEFF_SHIFT;
use g2g_mcu::{Resampler, SampleRate};
use util::{block_on, frame_of, le_bytes, payload};

use SampleRate::{Hz16000, Hz48000, Hz8000};

const PAIRS: [(SampleRate, SampleRate); 6] = [
    (Hz8000, Hz16000),
    (Hz16000, Hz48000),
    (Hz8000, Hz48000),
    (Hz16000, Hz8000),
    (Hz48000, Hz16000),
    (Hz48000, Hz8000),
];

fn hz(r: SampleRate) -> f64 {
    match r {
        Hz8000 => 8000.0,
        Hz16000 => 16000.0,
        Hz48000 => 48000.0,
    }
}

/// Resample `samples` through a fresh element in one frame.
fn run_once(from: SampleRate, to: SampleRate, samples: &[i16]) -> Vec<u8> {
    let ring: StaticLendRing<1, { 96 * 1024 }> = StaticLendRing::new();
    let input_ring: StaticLendRing<1, { 32 * 1024 }> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut rs = unsafe { Resampler::with_ring(from, to, &ring) };
    let input = frame_of(&input_ring, &le_bytes(samples), 0, 0);
    let out = block_on(rs.process(input))
        .expect("resample")
        .expect("frame");
    payload(&out).to_vec()
}

fn to_samples(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[test]
fn table_invariants() {
    for (from, to) in PAIRS {
        let t = table_for(from, to).expect("table");
        assert_eq!(t.coeffs.len(), t.l as usize * t.taps, "phase-major layout");
        assert!(t.taps <= MAX_TAPS, "history window covers every table");
        for p in 0..t.l as usize {
            let branch = &t.coeffs[p * t.taps..(p + 1) * t.taps];
            let sum: i32 = branch.iter().map(|&c| c as i32).sum();
            assert_eq!(
                sum,
                1 << COEFF_SHIFT,
                "phase {p} of {from:?}->{to:?} sums to unity"
            );
            let abs: i32 = branch.iter().map(|&c| (c as i32).abs()).sum();
            assert!(abs < 60_000, "i32 accumulator headroom, phase {p}");
        }
    }
    for r in [Hz8000, Hz16000, Hz48000] {
        assert!(table_for(r, r).is_none(), "same rate is identity");
    }
}

#[test]
fn dc_gain_is_exact() {
    for (from, to) in PAIRS {
        let t = table_for(from, to).expect("table");
        let n = 4 * t.taps.max(64);
        let out = to_samples(&run_once(from, to, &vec![1000i16; n]));
        // Skip the filter's fill-in transient; every settled output must be
        // exactly the input level (per-phase sums are exactly unity).
        let settle = 2 * t.taps * t.l as usize / t.m as usize;
        let tail = &out[settle.min(out.len())..];
        assert!(
            !tail.is_empty(),
            "enough settled output for {from:?}->{to:?}"
        );
        assert!(
            tail.iter().all(|&s| s == 1000),
            "{from:?}->{to:?}: settled DC must be exact, got {:?}",
            &tail[..tail.len().min(8)]
        );
    }
}

/// Least-squares fit of `a*sin + b*cos` at frequency `f_out` (cycles/sample)
/// and the residual RMS: measures tone fidelity without needing the filter's
/// fractional group delay.
fn sine_fit_snr(out: &[i16], f_out: f64) -> f64 {
    let (mut ss, mut sc, mut cc, mut ys, mut yc) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (i, &y) in out.iter().enumerate() {
        let (s, c) = (2.0 * std::f64::consts::PI * f_out * i as f64).sin_cos();
        ss += s * s;
        sc += s * c;
        cc += c * c;
        ys += y as f64 * s;
        yc += y as f64 * c;
    }
    let det = ss * cc - sc * sc;
    let a = (ys * cc - yc * sc) / det;
    let b = (yc * ss - ys * sc) / det;
    let mut signal = 0f64;
    let mut noise = 0f64;
    for (i, &y) in out.iter().enumerate() {
        let (s, c) = (2.0 * std::f64::consts::PI * f_out * i as f64).sin_cos();
        let fit = a * s + b * c;
        signal += fit * fit;
        noise += (y as f64 - fit) * (y as f64 - fit);
    }
    10.0 * (signal / noise.max(1e-12)).log10()
}

#[test]
fn in_band_tone_snr() {
    // A 997 Hz tone (non-harmonic of the frame sizes) through every pair.
    for (from, to) in PAIRS {
        let t = table_for(from, to).expect("table");
        let n = 8192;
        let f_in = 997.0 / hz(from);
        let tone: Vec<i16> = (0..n)
            .map(|i| (28_000.0 * (2.0 * std::f64::consts::PI * f_in * i as f64).sin()) as i16)
            .collect();
        let out = to_samples(&run_once(from, to, &tone));
        let settle = 4 * t.taps.max(32) * t.l as usize / t.m as usize;
        let steady = &out[settle..out.len() - 4];
        let snr = sine_fit_snr(steady, 997.0 / hz(to));
        assert!(snr > 55.0, "{from:?}->{to:?}: tone SNR {snr:.1} dB");
    }
}

#[test]
fn decimation_rejects_aliases() {
    // A 5 kHz tone is above the 4 kHz Nyquist of an 8 kHz output: after
    // 48k->8k it must be attenuated into the filter's stopband, not folded
    // in at full level.
    let n = 16_384;
    let f_in = 5000.0 / hz(Hz48000);
    let tone: Vec<i16> = (0..n)
        .map(|i| (28_000.0 * (2.0 * std::f64::consts::PI * f_in * i as f64).sin()) as i16)
        .collect();
    let out = to_samples(&run_once(Hz48000, Hz8000, &tone));
    let t = table_for(Hz48000, Hz8000).unwrap();
    let steady = &out[(2 * t.taps / t.m as usize)..];
    let rms =
        (steady.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / steady.len() as f64).sqrt();
    let rejection_db = 20.0 * (28_000.0 / rms.max(1e-9)).log10();
    assert!(rejection_db > 50.0, "alias rejection {rejection_db:.1} dB");
}

#[test]
fn chunked_streaming_matches_one_shot() {
    // The state (history + fractional position) must make frame boundaries
    // invisible: odd-sized chunks concatenate to exactly the one-shot bytes.
    let n = 2000;
    let sig: Vec<i16> = (0..n)
        .map(|i| ((i * 37 % 256) as i16 - 128) * 199)
        .collect();
    for (from, to) in [(Hz8000, Hz48000), (Hz48000, Hz8000), (Hz16000, Hz48000)] {
        let whole = run_once(from, to, &sig);
        let out_ring: StaticLendRing<2, { 96 * 1024 }> = StaticLendRing::new();
        // SAFETY: the rings outlive every frame in this test.
        let mut rs = unsafe { Resampler::with_ring(from, to, &out_ring) };
        let mut chunked = Vec::new();
        for (i, chunk) in sig.chunks(311).enumerate() {
            let input_ring: StaticLendRing<1, 1024> = StaticLendRing::new();
            let input = frame_of(&input_ring, &le_bytes(chunk), 0, i as u64);
            let out = block_on(rs.process(input))
                .expect("resample")
                .expect("frame");
            chunked.extend_from_slice(payload(&out));
        }
        assert_eq!(
            chunked, whole,
            "{from:?}->{to:?}: chunking must be invisible"
        );
    }
}

#[test]
fn element_mechanics() {
    // Same rate: identity pass-through, zero copy.
    let input_ring: StaticLendRing<1, 64> = StaticLendRing::new();
    let ring: StaticLendRing<1, 64> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut same = unsafe { Resampler::with_ring(Hz8000, Hz8000, &ring) };
    let bytes = le_bytes(&[1, -2, 3, -4]);
    let input = frame_of(&input_ring, &bytes, 9, 3);
    let out = block_on(same.process(input)).expect("ok").expect("frame");
    assert_eq!(payload(&out), bytes, "identity payload");
    assert_eq!((out.timing.pts_ns, out.sequence), (9, 3));

    // Torn payload.
    let input_ring2: StaticLendRing<1, 64> = StaticLendRing::new();
    // SAFETY: as above.
    let mut up = unsafe { Resampler::with_ring(Hz8000, Hz16000, &ring) };
    let torn = frame_of(&input_ring2, &[1, 2, 3], 0, 0);
    assert_eq!(
        block_on(up.process(torn)).unwrap_err(),
        G2gError::CapsMismatch
    );

    // Timing/sequence inheritance through a real conversion.
    let input_ring3: StaticLendRing<1, 64> = StaticLendRing::new();
    let big_ring: StaticLendRing<1, 256> = StaticLendRing::new();
    // SAFETY: as above.
    let mut up = unsafe { Resampler::with_ring(Hz8000, Hz16000, &big_ring) };
    let input = frame_of(&input_ring3, &le_bytes(&[100; 16]), 777, 5);
    let out = block_on(up.process(input)).expect("ok").expect("frame");
    assert_eq!(payload(&out).len(), 64, "16 samples in -> 32 out at 2x");
    assert_eq!((out.timing.pts_ns, out.sequence), (777, 5));
}

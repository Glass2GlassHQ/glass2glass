//! M638: the G.711 fixed-point codec, validated against the reference peer.
//! The sample conversions are checked bit-exact against ffmpeg over the
//! *entire* domain (all 65536 encoder inputs, all 256 decoder codes, both
//! laws), the strongest oracle a codec can have, persisted as `Oracle`
//! conformance evidence. The ring-lend transform elements are exercised on
//! host rings: exact payload conversion, timing/sequence inheritance, framing
//! validation, and ring back-pressure surfacing.
//!
//! There is no runtime zero-alloc test on purpose: `g2g-mcu` depends on
//! `g2g-core` with `default-features = false`, so the `alloc` crate does not
//! exist in its build; a codec path that needed the heap would fail to
//! compile (the crate's thumb cross-check in CI), which is stronger than a
//! counter.

mod util;

use std::process::Command;

use g2g_core::error::G2gError;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticTransform;
use g2g_mcu::g711::{alaw_decode, alaw_encode, mulaw_decode, mulaw_encode};
use g2g_mcu::{G711Dec, G711Enc, Law};
use util::{block_on, frame_of, le_bytes, payload};

#[test]
fn known_reconstruction_values() {
    // Mu-law anchors (ITU table values): both zero codes, the extremes.
    assert_eq!(mulaw_decode(0xFF), 0, "+0");
    assert_eq!(mulaw_decode(0x7F), 0, "-0");
    assert_eq!(mulaw_decode(0x80), 32124, "max positive");
    assert_eq!(mulaw_decode(0x00), -32124, "max negative");
    // A-law anchors (sign bit set on the wire = positive).
    assert_eq!(alaw_decode(0xD5), 8, "+0 level");
    assert_eq!(alaw_decode(0x55), -8, "-0 level");
    assert_eq!(alaw_decode(0xAA), 32256, "max positive");
    assert_eq!(alaw_decode(0x2A), -32256, "max negative");
    // Encode anchors: zero lands on the positive zero code.
    assert_eq!(mulaw_encode(0), 0xFF);
    assert_eq!(alaw_encode(0), 0xD5);
    assert_eq!(mulaw_encode(i16::MAX), 0x80);
    assert_eq!(alaw_encode(i16::MAX), 0xAA);
}

#[test]
fn codeword_transparency() {
    for c in 0..=255u8 {
        // A-law: strict code transparency.
        assert_eq!(alaw_encode(alaw_decode(c)), c, "alaw code {c:#04x}");
        // Mu-law has two zero codes; -0 (0x7F) collapses onto +0 (0xFF) like
        // every reference implementation. All other codes round-trip.
        let expect = if c == 0x7F { 0xFF } else { c };
        assert_eq!(mulaw_encode(mulaw_decode(c)), expect, "mulaw code {c:#04x}");
    }
}

#[test]
fn reconstruction_error_bounded() {
    // Nearest-level quantization: the error is at most half the widest
    // segment step (1024 for both laws) plus the >>2 domain floor.
    for s in (i16::MIN..=i16::MAX).step_by(7) {
        let mu = mulaw_decode(mulaw_encode(s)) as i32 - s as i32;
        let a = alaw_decode(alaw_encode(s)) as i32 - s as i32;
        // Mu-law tops out at 32124, so the outermost inputs sit further from
        // the last level than half a step; A-law likewise at 32256.
        let bound = if s.unsigned_abs() >= 31 * 1024 { 660 } else { 520 };
        assert!(mu.abs() <= bound, "mulaw err {mu} at {s}");
        assert!(a.abs() <= bound, "alaw err {a} at {s}");
    }
}

/// Run ffmpeg converting `input` (written to a temp file) between raw
/// formats, returning the output bytes. Temp files, not pipes: the 128 KiB
/// sweep would deadlock a single-threaded pipe writer.
fn ffmpeg_convert(in_fmt: &str, out_fmt: &str, input: &[u8]) -> Vec<u8> {
    let dir = std::env::temp_dir();
    let stamp = std::process::id();
    let src = dir.join(format!("g2g-m638-{stamp}-in.{in_fmt}"));
    let dst = dir.join(format!("g2g-m638-{stamp}-out.{out_fmt}"));
    std::fs::write(&src, input).expect("write temp input");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", in_fmt, "-ar", "8000", "-ac", "1", "-i"])
        .arg(&src)
        .args(["-f", out_fmt])
        .arg(&dst)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg {in_fmt} -> {out_fmt}");
    let out = std::fs::read(&dst).expect("read temp output");
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&dst);
    out
}

#[test]
fn ffmpeg_oracle_bit_exact_full_domain() {
    // Self-skip on a box without ffmpeg (the CI conformance job has it).
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }

    // Decode: all 256 codes, both laws, bit-exact against ffmpeg's decoder.
    let codes: Vec<u8> = (0..=255u8).collect();
    for (fmt, dec) in [("mulaw", mulaw_decode as fn(u8) -> i16), ("alaw", alaw_decode)] {
        let theirs = ffmpeg_convert(fmt, "s16le", &codes);
        let ours: Vec<u8> = le_bytes(&codes.iter().map(|&c| dec(c)).collect::<Vec<_>>());
        assert_eq!(ours, theirs, "{fmt} decode differs from ffmpeg");
    }

    // Encode: the full 65536-sample sweep, both laws, bit-exact against
    // ffmpeg's encoder (same nearest-reconstruction quantization).
    let sweep: Vec<i16> = (i16::MIN..=i16::MAX).collect();
    let sweep_bytes = le_bytes(&sweep);
    for (fmt, enc) in [("mulaw", mulaw_encode as fn(i16) -> u8), ("alaw", alaw_encode)] {
        let theirs = ffmpeg_convert("s16le", fmt, &sweep_bytes);
        let ours: Vec<u8> = sweep.iter().map(|&s| enc(s)).collect();
        assert_eq!(ours, theirs, "{fmt} encode differs from ffmpeg");
    }

    // A full-domain bit-exact match against a named external implementation
    // is Oracle-tier conformance evidence; persist it for the maturity table.
    use g2g_core::conformance::{ConformanceDimension, Evidence};
    for element in ["g711enc", "g711dec"] {
        g2g_plugins::conformance::persist::record_evidence(
            element,
            &Evidence::new(ConformanceDimension::Oracle)
                .peer("ffmpeg")
                .codec("g711")
                .detail("bit-exact full-domain sweep, mu-law + A-law"),
        )
        .expect("persist oracle evidence");
    }
}

#[test]
fn encoder_element_converts_and_inherits_timing() {
    let samples: Vec<i16> = vec![0, 1, -1, i16::MAX, i16::MIN, 12345, -12345, 42];
    let input_ring: StaticLendRing<1, 16> = StaticLendRing::new();
    let out_ring: StaticLendRing<2, 8> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut enc = unsafe { G711Enc::with_ring(Law::Mulaw, &out_ring) };

    let input = frame_of(&input_ring, &le_bytes(&samples), 777, 9);
    let out = block_on(enc.process(input)).expect("encode").expect("frame");
    let expect: Vec<u8> = samples.iter().map(|&s| mulaw_encode(s)).collect();
    assert_eq!(payload(&out), expect, "per-sample companding");
    assert_eq!(out.timing.pts_ns, 777, "timing inherited");
    assert_eq!(out.sequence, 9, "sequence inherited");
}

#[test]
fn decoder_element_round_trips_the_encoder() {
    let samples: Vec<i16> = (-64..64).map(|i| i * 257).collect();
    let input_ring: StaticLendRing<1, 256> = StaticLendRing::new();
    let mid_ring: StaticLendRing<2, 128> = StaticLendRing::new();
    let out_ring: StaticLendRing<2, 256> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut enc = unsafe { G711Enc::with_ring(Law::Alaw, &mid_ring) };
    // SAFETY: as above.
    let mut dec = unsafe { G711Dec::with_ring(Law::Alaw, &out_ring) };

    let input = frame_of(&input_ring, &le_bytes(&samples), 5, 1);
    let mid = block_on(enc.process(input)).expect("encode").expect("frame");
    let out = block_on(dec.process(mid)).expect("decode").expect("frame");
    let expect: Vec<u8> =
        le_bytes(&samples.iter().map(|&s| alaw_decode(alaw_encode(s))).collect::<Vec<_>>());
    assert_eq!(payload(&out), expect, "decode(encode(x)) through the elements");
}

#[test]
fn framing_and_sizing_are_validated() {
    let input_ring: StaticLendRing<1, 64> = StaticLendRing::new();
    let out_ring: StaticLendRing<2, 8> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut enc = unsafe { G711Enc::with_ring(Law::Mulaw, &out_ring) };

    // Torn 16-bit sample: rejected before touching the ring.
    let torn = frame_of(&input_ring, &[1, 2, 3], 0, 0);
    assert_eq!(block_on(enc.process(torn)).unwrap_err(), G2gError::CapsMismatch);

    // Output larger than a slot (32 bytes in -> 16 out > 8): a sizing bug.
    let big = frame_of(&input_ring, &[0u8; 32], 0, 0);
    assert_eq!(block_on(enc.process(big)).unwrap_err(), G2gError::CapsMismatch);
}

#[test]
fn ring_exhaustion_surfaces_as_pool_exhausted() {
    let input_ring: StaticLendRing<1, 8> = StaticLendRing::new();
    let out_ring: StaticLendRing<1, 8> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut enc = unsafe { G711Enc::with_ring(Law::Mulaw, &out_ring) };

    let first = block_on(enc.process(frame_of(&input_ring, &[0, 0], 0, 0)))
        .expect("encode")
        .expect("frame");
    // The only output slot is still lent out to `first`.
    let input_ring2: StaticLendRing<1, 8> = StaticLendRing::new();
    let err = block_on(enc.process(frame_of(&input_ring2, &[0, 0], 0, 1))).unwrap_err();
    assert_eq!(err, G2gError::PoolExhausted);
    drop(first);
    // Slot returned: encoding proceeds again.
    let input_ring3: StaticLendRing<1, 8> = StaticLendRing::new();
    block_on(enc.process(frame_of(&input_ring3, &[0, 0], 0, 2))).expect("encode").expect("frame");
}

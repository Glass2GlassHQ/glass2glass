//! M639: the IMA ADPCM codec (WAV / DVI4 block layout, mono), validated
//! against the reference peer in all three directions: our encode of a
//! full-range signal is byte-identical to ffmpeg's `adpcm_ima_wav` encoder,
//! our decode of ffmpeg's stream matches ffmpeg's own decode exactly, and
//! ffmpeg decodes *our* stream to exactly what we decode. Persisted as
//! `Oracle` conformance evidence. A pre-8.1 ffmpeg decodes 4-bit IMA-WAV
//! with the superseded multiplicative arithmetic; the oracle test classifies
//! that by output and skips the decode-exactness asserts against it (M765).
//! The block functions and ring-lend elements are exercised on host rings
//! like the G.711 ones (`m638_g711.rs`); the no-alloc claim is compile-time
//! (see there).

mod util;

use std::process::Command;

use g2g_core::error::G2gError;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticTransform;
use g2g_mcu::adpcm::{decode_block, encode_block, samples_per_block, BLOCK_HEADER};
use g2g_mcu::{AdpcmDec, AdpcmEnc, ImaState};
use util::{block_on, frame_of, le_bytes, payload};

/// The WAV-default block size ffmpeg uses (mono: 2041 samples per block).
const BLOCK: usize = 1024;

/// A deterministic full-range test signal: a swept sine plus LCG noise, hard
/// on the step adaptation (fast transients + quiet stretches).
fn signal(n: usize) -> Vec<i16> {
    let mut v = Vec::with_capacity(n);
    let mut lcg: u32 = 0x1234_5678;
    for i in 0..n {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (lcg >> 16) as i16 as i32 / 8;
        let t = i as f64;
        let sine = (32_000.0 * (t * 0.01 + t * t * 1e-6).sin()) as i32;
        v.push((sine * 3 / 4 + noise).clamp(-32768, 32767) as i16)
    }
    v
}

fn encode_stream(samples: &[i16], block_bytes: usize) -> Vec<u8> {
    let spb = samples_per_block(block_bytes);
    let mut out = vec![0u8; samples.len() / spb * block_bytes];
    let mut index = 0u8;
    for (s, d) in le_bytes(samples)
        .chunks_exact(spb * 2)
        .zip(out.chunks_exact_mut(block_bytes))
    {
        index = encode_block(index, s, d).expect("sized exactly");
    }
    out
}

fn decode_stream(bytes: &[u8], block_bytes: usize) -> Vec<u8> {
    let spb = samples_per_block(block_bytes);
    let mut out = vec![0u8; bytes.len() / block_bytes * spb * 2];
    for (s, d) in bytes
        .chunks_exact(block_bytes)
        .zip(out.chunks_exact_mut(spb * 2))
    {
        decode_block(s, d).expect("sized exactly");
    }
    out
}

/// Minimal mono WAVE_FORMAT_IMA_ADPCM (0x11) wrapper around raw blocks, so
/// ffmpeg can read our stream.
fn wav_of(data: &[u8], block_bytes: u16, rate: u32) -> Vec<u8> {
    let spb = samples_per_block(block_bytes as usize) as u32;
    let mut h = Vec::new();
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(4 + 28 + 12 + 8 + data.len() as u32).to_le_bytes());
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&20u32.to_le_bytes());
    h.extend_from_slice(&0x11u16.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes());
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&(rate * block_bytes as u32 / spb).to_le_bytes());
    h.extend_from_slice(&block_bytes.to_le_bytes());
    h.extend_from_slice(&4u16.to_le_bytes());
    h.extend_from_slice(&2u16.to_le_bytes());
    h.extend_from_slice(&(spb as u16).to_le_bytes());
    h.extend_from_slice(b"fact");
    h.extend_from_slice(&4u32.to_le_bytes());
    h.extend_from_slice(&(data.len() as u32 / block_bytes as u32 * spb).to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&(data.len() as u32).to_le_bytes());
    let mut wav = h;
    wav.extend_from_slice(data);
    wav
}

/// The legacy (pre-8.1 ffmpeg) multiplicative IMA reconstruction
/// (`diff = (2n+1)*step >> 3`), kept only to classify an old oracle's decode
/// output; the codec under test implements the IMA spec's bit-serial form the
/// modern decoder uses. An independent reference, not a copy of the unit.
fn decode_stream_multiplicative(bytes: &[u8], block_bytes: usize) -> Vec<u8> {
    const STEPS: [u16; 89] = [
        7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60,
        66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371,
        408, 449, 494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878,
        2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845,
        8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086,
        29794, 32767,
    ];
    const INDEX: [i8; 8] = [-1, -1, -1, -1, 2, 4, 6, 8];
    let mut out = Vec::with_capacity(bytes.len());
    for blk in bytes.chunks_exact(block_bytes) {
        let mut pred = i16::from_le_bytes([blk[0], blk[1]]) as i32;
        let mut idx = (blk[2] as usize).min(88);
        out.extend_from_slice(&(pred as i16).to_le_bytes());
        for byte in &blk[BLOCK_HEADER..] {
            for nib in [byte & 0xF, byte >> 4] {
                let step = STEPS[idx] as i32;
                let diff = ((2 * (nib & 7) as i32 + 1) * step) >> 3;
                pred = (pred + if nib & 8 != 0 { -diff } else { diff }).clamp(-32768, 32767);
                idx = (idx as i32 + INDEX[(nib & 7) as usize] as i32).clamp(0, 88) as usize;
                out.extend_from_slice(&(pred as i16).to_le_bytes());
            }
        }
    }
    out
}

/// The `data` chunk of a WAV file.
fn wav_data(wav: &[u8]) -> Vec<u8> {
    let mut i = 12;
    while i + 8 <= wav.len() {
        let id = &wav[i..i + 4];
        let sz = u32::from_le_bytes(wav[i + 4..i + 8].try_into().unwrap()) as usize;
        if id == b"data" {
            return wav[i + 8..i + 8 + sz].to_vec();
        }
        i += 8 + sz + (sz & 1);
    }
    panic!("no data chunk");
}

#[test]
fn sample_arithmetic_anchors() {
    // From zero state, a zero delta still nudges by step>>3 truncated: step 7
    // -> diff 0; the nibble is 0 and the state cools to index 0 (already 0).
    let mut st = ImaState::default();
    assert_eq!(st.encode_sample(0), 0);
    assert_eq!(
        st,
        ImaState {
            predictor: 0,
            step_index: 0
        }
    );
    // A full-scale jump saturates the nibble and heats the step index by 8.
    let mut st = ImaState::default();
    let nib = st.encode_sample(i16::MAX);
    assert_eq!(nib, 7);
    assert_eq!(st.step_index, 8);
    // Decode of a max-magnitude negative nibble walks the predictor down.
    let mut st = ImaState {
        predictor: 0,
        step_index: 88,
    };
    let s = st.decode_sample(0xF);
    assert_eq!(s, -32768, "clamped to i16 range");
    assert_eq!(st.step_index, 88, "index stays clamped at the top");
}

#[test]
fn block_round_trip_and_state_carry() {
    let spb = samples_per_block(12); // tiny blocks: 17 samples each
    let sig = signal(spb * 3);
    let enc = encode_stream(&sig, 12);
    assert_eq!(enc.len(), 36);
    // Every block header carries the block's first sample verbatim.
    for (blk, samples) in enc.chunks_exact(12).zip(sig.chunks_exact(spb)) {
        assert_eq!(i16::from_le_bytes([blk[0], blk[1]]), samples[0]);
    }
    // Round trip: sample 0 of each block is exact; the rest tracks within
    // the quantizer's reach (coarse for tiny blocks, exact structure checked
    // bit-for-bit against ffmpeg in the oracle test).
    let dec = decode_stream(&enc, 12);
    let dec_samples: Vec<i16> = dec
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(dec_samples.len(), sig.len());
    for (blk_idx, (d, s)) in dec_samples
        .chunks_exact(spb)
        .zip(sig.chunks_exact(spb))
        .enumerate()
    {
        assert_eq!(d[0], s[0], "block {blk_idx} anchor sample");
    }
    // Undersized / oversized slices are rejected, never panicking.
    let mut small = [0u8; BLOCK_HEADER - 1];
    assert!(encode_block(0, &le_bytes(&sig[..spb]), &mut small).is_none());
    assert!(decode_block(&enc[..BLOCK_HEADER - 1], &mut [0u8; 34]).is_none());
    assert!(
        decode_block(&enc[..12], &mut [0u8; 12]).is_none(),
        "wrong dst size"
    );
}

#[test]
fn ffmpeg_oracle_bit_exact_three_ways() {
    // Self-skip on a box without ffmpeg (the CI conformance job has it).
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }
    let dir = std::env::temp_dir();
    let stamp = std::process::id();
    let f = |name: &str| dir.join(format!("g2g-m639-{stamp}-{name}"));

    let spb = samples_per_block(BLOCK);
    let sig = signal(spb * 8);
    let ours_enc = encode_stream(&sig, BLOCK);

    // 1. Encode: ffmpeg encodes the same signal; the streams must be
    //    byte-identical (same quantizer arithmetic, same state carry).
    std::fs::write(f("sig.s16"), le_bytes(&sig)).unwrap();
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            "8000",
            "-ac",
            "1",
            "-i",
        ])
        .arg(f("sig.s16"))
        .args(["-c:a", "adpcm_ima_wav", "-block_size", &BLOCK.to_string()])
        .arg(f("theirs.wav"))
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg encode");
    let theirs_enc = wav_data(&std::fs::read(f("theirs.wav")).unwrap());
    assert_eq!(ours_enc, theirs_enc, "encoded stream differs from ffmpeg");

    // 2. Decode of the reference stream: must match ffmpeg's own s16 output.
    //    ffmpeg >= 8.1 decodes 4-bit IMA-WAV with the IMA spec's bit-serial
    //    expansion (which the codec under test implements); older ffmpeg used
    //    the multiplicative form. Classify the installed oracle by its own
    //    output: a legacy decoder skips the decode-exactness asserts (its
    //    arithmetic is the superseded one), anything else must match us.
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(f("theirs.wav"))
        .args(["-f", "s16le"])
        .arg(f("theirs.s16"))
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg decode");
    let ffmpeg_dec = std::fs::read(f("theirs.s16")).unwrap();
    let strict = decode_stream(&theirs_enc, BLOCK) == ffmpeg_dec;
    if !strict {
        assert_eq!(
            decode_stream_multiplicative(&theirs_enc, BLOCK),
            ffmpeg_dec,
            "ffmpeg's decode matches neither the bit-serial nor the legacy multiplicative arithmetic"
        );
        eprintln!(
            "legacy (pre-8.1) ffmpeg IMA decoder detected; skipping decode-exactness asserts"
        );
    }

    // 3. The reference decodes *our* stream to exactly our reconstruction
    //    (bit-serial both sides, so only meaningful against a modern oracle).
    if strict {
        std::fs::write(f("ours.wav"), wav_of(&ours_enc, BLOCK as u16, 8000)).unwrap();
        let status = Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"])
            .arg(f("ours.wav"))
            .args(["-f", "s16le"])
            .arg(f("ffdec.s16"))
            .status()
            .expect("run ffmpeg");
        assert!(status.success(), "ffmpeg decodes our wav");
        assert_eq!(
            std::fs::read(f("ffdec.s16")).unwrap(),
            decode_stream(&ours_enc, BLOCK),
            "ffmpeg reconstructs our stream differently"
        );
    }

    for name in [
        "sig.s16",
        "theirs.wav",
        "theirs.s16",
        "ours.wav",
        "ffdec.s16",
    ] {
        let _ = std::fs::remove_file(f(name));
    }

    // Full-signal bit-exactness against a named external implementation:
    // Oracle-tier evidence. Honest about what ran: a legacy oracle only
    // verified the encode direction.
    use g2g_core::conformance::{ConformanceDimension, Evidence};
    let detail = if strict {
        "bit-exact encode + decode + cross-decode"
    } else {
        "bit-exact encode (legacy ffmpeg decoder, decode asserts skipped)"
    };
    for element in ["adpcmenc", "adpcmdec"] {
        g2g_plugins::conformance::persist::record_evidence(
            element,
            &Evidence::new(ConformanceDimension::Oracle)
                .peer("ffmpeg")
                .codec("adpcm-ima-wav")
                .detail(detail),
        )
        .expect("persist oracle evidence");
    }
}

#[test]
fn elements_round_trip_whole_blocks() {
    const BB: usize = 12; // 17 samples per block
    let spb = samples_per_block(BB);
    let sig = signal(spb * 2);
    let input_ring: StaticLendRing<1, 128> = StaticLendRing::new();
    let mid_ring: StaticLendRing<2, 64> = StaticLendRing::new();
    let out_ring: StaticLendRing<2, 128> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut enc = unsafe { AdpcmEnc::with_ring(&mid_ring, BB) };
    // SAFETY: as above.
    let mut dec = unsafe { AdpcmDec::with_ring(&out_ring, BB) };

    let input = frame_of(&input_ring, &le_bytes(&sig), 42, 7);
    let mid = block_on(enc.process(input))
        .expect("encode")
        .expect("frame");
    assert_eq!(
        payload(&mid),
        encode_stream(&sig, BB),
        "element matches the block fns"
    );
    assert_eq!(mid.timing.pts_ns, 42, "timing inherited");
    assert_eq!(mid.sequence, 7, "sequence inherited");

    let out = block_on(dec.process(mid)).expect("decode").expect("frame");
    assert_eq!(payload(&out), decode_stream(&encode_stream(&sig, BB), BB));
}

#[test]
fn encoder_state_carries_across_frames() {
    const BB: usize = 12;
    let spb = samples_per_block(BB);
    let sig = signal(spb * 2);
    let ring_a: StaticLendRing<1, 64> = StaticLendRing::new();
    let ring_b: StaticLendRing<1, 64> = StaticLendRing::new();
    let mid_ring: StaticLendRing<2, 64> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut enc = unsafe { AdpcmEnc::with_ring(&mid_ring, BB) };

    // One frame per block: together they must equal the single-stream encode
    // (the step index carries across frames like ffmpeg carries it across
    // blocks).
    let (a, b) = sig.split_at(spb);
    let fa = block_on(enc.process(frame_of(&ring_a, &le_bytes(a), 0, 0)))
        .unwrap()
        .unwrap();
    let fb = block_on(enc.process(frame_of(&ring_b, &le_bytes(b), 0, 1)))
        .unwrap()
        .unwrap();
    let mut both = payload(&fa).to_vec();
    both.extend_from_slice(payload(&fb));
    assert_eq!(
        both,
        encode_stream(&sig, BB),
        "state must carry across frames"
    );
}

#[test]
fn framing_is_validated() {
    const BB: usize = 12;
    let input_ring: StaticLendRing<1, 64> = StaticLendRing::new();
    let out_ring: StaticLendRing<1, 64> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut enc = unsafe { AdpcmEnc::with_ring(&out_ring, BB) };
    // SAFETY: as above.
    let mut dec = unsafe { AdpcmDec::with_ring(&out_ring, BB) };

    // Not a whole number of blocks' samples.
    let torn = frame_of(&input_ring, &[0u8; 10], 0, 0);
    assert_eq!(
        block_on(enc.process(torn)).unwrap_err(),
        G2gError::CapsMismatch
    );
    let input_ring2: StaticLendRing<1, 64> = StaticLendRing::new();
    let torn = frame_of(&input_ring2, &[0u8; 10], 0, 0);
    assert_eq!(
        block_on(dec.process(torn)).unwrap_err(),
        G2gError::CapsMismatch
    );
}

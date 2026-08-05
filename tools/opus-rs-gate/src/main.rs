//! Correctness gate for replacing the libopus FFI decode path with a pure-Rust
//! one (M916). Decodes identical packet sequences through `opus-rs` (the only
//! complete pure-Rust Opus implementation on crates.io, a port of libopus 1.6)
//! and through libopus, and reports how far apart the PCM lands.
//!
//! Three inputs: the RFC 8251 conformance vectors, the repo's own Ogg Opus
//! fixtures, and libopus-encoded 48 kHz 20 ms streams matching what `OpusEnc`
//! emits. Run it with `tools/opus-rs-gate.sh`, which fetches the vectors.
//!
//! `opus-rs` is a port of the same algorithm, not an independent decoder, so a
//! correct one agrees with libopus to near float precision. [`MIN_SNR_DB`] is
//! therefore set far below that: clearing it means "unambiguously the same
//! decoder", and missing it is not a tolerance question.
//!
//! Verdict at 0.1.26: FAIL, by a wide margin. Numbers in the run output.

use std::fs;
use std::path::Path;
use std::process::ExitCode;

use audiopus::coder::{Decoder as LibDecoder, Encoder as LibEncoder};
use audiopus::packet::Packet;
use audiopus::{Application, Bitrate, Channels, MutSignals, SampleRate};
use opus_rs::OpusDecoder as RsDecoder;

/// Longest Opus frame is 120 ms: 5760 samples per channel at 48 kHz.
const MAX_FRAME_SAMPLES: usize = 5760;

/// SNR (dB) of opus-rs output against libopus output below which the candidate
/// is not the same decoder. A port of libopus decoding the same packets sits
/// near float precision (80 dB+); the reference path's own agreement with the
/// shipped RFC 8251 output measures 74 dB and up.
const MIN_SNR_DB: f64 = 40.0;

/// SNR of the libopus reference path against the shipped `.dec` output, below
/// which the harness itself is suspect rather than the candidate.
const MIN_SELFCHECK_SNR_DB: f64 = 60.0;

fn snr(reference: &[i16], test: &[i16]) -> f64 {
    let n = reference.len().min(test.len());
    let (mut signal, mut error) = (0.0f64, 0.0f64);
    for k in 0..n {
        let r = f64::from(reference[k]);
        let d = r - f64::from(test[k]);
        signal += r * r;
        error += d * d;
    }
    if error == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (signal / error).log10()
    }
}

/// Split an `opus_demo` `.bit` file into packets: each is a big-endian byte
/// length, a big-endian range-coder final state (unused here), then the payload.
fn bit_packets(bytes: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        i += 8;
        if len == 0 || len > 1500 || i + len > bytes.len() {
            break;
        }
        out.push(&bytes[i..i + len]);
        i += len;
    }
    out
}

/// Packets of a single-stream Ogg file, header packets included. Enough for the
/// repo's own fixtures; not a general demuxer.
fn ogg_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut partial: Vec<u8> = Vec::new();
    let mut i = 0usize;
    while i + 27 <= data.len() && &data[i..i + 4] == b"OggS" {
        let n_segments = data[i + 26] as usize;
        let table = i + 27;
        let Some(payload) = table.checked_add(n_segments) else {
            break;
        };
        if payload > data.len() {
            break;
        }
        let mut at = payload;
        for &seg in &data[table..payload] {
            let seg = seg as usize;
            if at + seg > data.len() {
                return packets;
            }
            partial.extend_from_slice(&data[at..at + seg]);
            at += seg;
            if seg < 255 {
                packets.push(core::mem::take(&mut partial));
            }
        }
        i = at;
    }
    packets
}

/// Decode with libopus, returning the PCM and each packet's per-channel sample
/// count so the candidate's output can be held to the same timeline.
fn decode_libopus(packets: &[&[u8]], channels: usize) -> (Vec<i16>, Vec<usize>) {
    let chans = if channels == 1 {
        Channels::Mono
    } else {
        Channels::Stereo
    };
    let mut dec = LibDecoder::new(SampleRate::Hz48000, chans).unwrap();
    let mut pcm = Vec::new();
    let mut durations = Vec::new();
    for p in packets {
        let mut buf = vec![0i16; MAX_FRAME_SAMPLES * channels];
        let packet = Packet::try_from(*p).expect("libopus rejected a packet");
        let signals = MutSignals::try_from(&mut buf[..]).unwrap();
        let n = dec.decode(Some(packet), signals, false).unwrap();
        durations.push(n);
        pcm.extend_from_slice(&buf[..n * channels]);
    }
    (pcm, durations)
}

struct RsRun {
    pcm: Vec<i16>,
    rejected: usize,
    errors: Vec<String>,
}

/// Decode with opus-rs. A rejected or panicking packet contributes silence for
/// its libopus duration, so the two outputs stay sample-aligned: an unaligned
/// comparison would report a meaningless SNR.
fn decode_opus_rs(packets: &[&[u8]], channels: usize, durations: &[usize]) -> RsRun {
    let mut dec = RsDecoder::new(48_000, channels).unwrap();
    let mut pcm: Vec<i16> = Vec::new();
    let mut rejected = 0usize;
    let mut errors: Vec<String> = Vec::new();
    let note = |errors: &mut Vec<String>, e: String| {
        if !errors.contains(&e) {
            errors.push(e);
        }
    };
    for (p, &want) in packets.iter().zip(durations) {
        let mut buf = vec![0f32; MAX_FRAME_SAMPLES * channels];
        let decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            dec.decode(p, MAX_FRAME_SAMPLES, &mut buf)
        }));
        let n = match decoded {
            Ok(Ok(n)) => n.min(want),
            Ok(Err(e)) => {
                rejected += 1;
                note(&mut errors, e.to_string());
                0
            }
            Err(_) => {
                rejected += 1;
                note(&mut errors, "PANIC".to_string());
                0
            }
        };
        pcm.extend(
            buf[..n * channels]
                .iter()
                .map(|v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16),
        );
        pcm.resize(pcm.len() + (want - n) * channels, 0);
    }
    RsRun {
        pcm,
        rejected,
        errors,
    }
}

/// One comparison line. Returns whether it cleared the gate.
fn compare(label: &str, packets: &[&[u8]], channels: usize) -> bool {
    let (lib, durations) = decode_libopus(packets, channels);
    let rs = decode_opus_rs(packets, channels, &durations);
    let db = snr(&lib, &rs.pcm);
    let pass = rs.rejected == 0 && db >= MIN_SNR_DB;
    println!(
        "  {:<24} ch={channels} packets={:>5} rejected={:>5} ({:>5.1}%) bit-exact={:<5} snr={:>7.2} dB  {}{}",
        label,
        packets.len(),
        rs.rejected,
        100.0 * rs.rejected as f64 / packets.len() as f64,
        lib == rs.pcm,
        db,
        if pass { "PASS" } else { "FAIL" },
        if rs.errors.is_empty() {
            String::new()
        } else {
            format!("  {:?}", rs.errors)
        }
    );
    pass
}

/// Decode each conformance vector through libopus and check it against the
/// shipped reference output, so a candidate failure cannot be blamed on the
/// `.bit` parsing or the packet ordering.
fn self_check(dir: &Path) -> bool {
    println!("libopus reference path vs the shipped RFC 8251 output:");
    let mut ok = true;
    for n in 1..=12 {
        let bytes = fs::read(dir.join(format!("testvector{n:02}.bit"))).unwrap();
        let reference: Vec<i16> = fs::read(dir.join(format!("testvector{n:02}.dec")))
            .unwrap()
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let (lib, _) = decode_libopus(&bit_packets(&bytes), 2);
        let db = snr(&reference, &lib);
        let aligned = lib.len() == reference.len();
        let pass = aligned && db >= MIN_SELFCHECK_SNR_DB;
        ok &= pass;
        println!(
            "  testvector{n:02} samples={:>8} (reference {:>8}) snr={db:>7.2} dB  {}",
            lib.len(),
            reference.len(),
            if pass { "PASS" } else { "FAIL" }
        );
    }
    ok
}

fn conformance(dir: &Path) -> bool {
    println!("\nopus-rs vs libopus on the RFC 8251 vectors, 48 kHz stereo:");
    let mut ok = true;
    for n in 1..=12 {
        let bytes = fs::read(dir.join(format!("testvector{n:02}.bit"))).unwrap();
        ok &= compare(&format!("testvector{n:02}"), &bit_packets(&bytes), 2);
    }
    ok
}

fn fixtures(dir: &Path) -> bool {
    println!("\nopus-rs vs libopus on the repo's Ogg Opus fixtures:");
    let mut ok = true;
    for (name, channels) in [("opus_mono_48k.opus", 1), ("opus_stereo_48k.opus", 2)] {
        let data = fs::read(dir.join(name)).unwrap();
        let all = ogg_packets(&data);
        // Drop OpusHead + OpusTags: they are codec config, not audio.
        let audio: Vec<&[u8]> = all
            .iter()
            .filter(|p| !p.starts_with(b"OpusHead") && !p.starts_with(b"OpusTags"))
            .map(|p| p.as_slice())
            .collect();
        ok &= compare(name, &audio, channels);
    }
    ok
}

/// The stream shape `OpusEnc` produces: 48 kHz, 20 ms, one frame per packet, a
/// fixed channel count. If the candidate cannot match libopus here it cannot
/// stand in for it anywhere.
fn synthetic() -> bool {
    println!("\nopus-rs vs libopus on libopus-encoded 48 kHz 20 ms streams:");
    let mut ok = true;
    for (channels, chans) in [(1usize, Channels::Mono), (2, Channels::Stereo)] {
        for (app, name) in [
            (Application::Audio, "audio"),
            (Application::Voip, "voip"),
            (Application::LowDelay, "lowdelay"),
        ] {
            let mut enc = LibEncoder::new(SampleRate::Hz48000, chans, app).unwrap();
            enc.set_bitrate(Bitrate::BitsPerSecond(96_000)).unwrap();
            let frame = 960usize;
            let mut phase = 0.0f32;
            let mut packets: Vec<Vec<u8>> = Vec::new();
            for _ in 0..100 {
                let mut input = vec![0i16; frame * channels];
                for s in input.chunks_exact_mut(channels) {
                    phase += 2.0 * std::f32::consts::PI * 440.0 / 48_000.0;
                    let v = (phase.sin() * 12_000.0) as i16;
                    s[0] = v;
                    if channels == 2 {
                        s[1] = v / 2;
                    }
                }
                let mut buf = vec![0u8; 4000];
                let n = enc.encode(&input, &mut buf).unwrap();
                buf.truncate(n);
                packets.push(buf);
            }
            let refs: Vec<&[u8]> = packets.iter().map(|p| p.as_slice()).collect();
            ok &= compare(&format!("440 Hz tone, {name}"), &refs, channels);
        }
    }
    ok
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let vectors = args
        .next()
        .expect("usage: opus-rs-gate <vector-dir> <fixture-dir>");
    let fixture_dir = args
        .next()
        .expect("usage: opus-rs-gate <vector-dir> <fixture-dir>");
    let vectors = Path::new(&vectors);
    let fixture_dir = Path::new(&fixture_dir);

    // opus-rs panics rather than returning an error on some malformed input;
    // the default hook would bury the report under backtraces.
    std::panic::set_hook(Box::new(|_| {}));

    let harness_ok = self_check(vectors);
    let gate_ok = conformance(vectors) & fixtures(fixture_dir) & synthetic();
    let _ = std::panic::take_hook();

    println!();
    if !harness_ok {
        println!("HARNESS SUSPECT: the libopus path does not reproduce the shipped output.");
        return ExitCode::FAILURE;
    }
    if gate_ok {
        println!("GATE PASSED: opus-rs matches libopus within {MIN_SNR_DB} dB everywhere.");
        ExitCode::SUCCESS
    } else {
        println!("GATE FAILED: opus-rs does not reproduce libopus. Stay on the FFI path.");
        ExitCode::FAILURE
    }
}

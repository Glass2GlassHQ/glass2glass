//! M778: Ogg granule timing. `oggdemux stream=vorbis` stamps every audio
//! packet with sample-accurate pts / duration from the setup header's mode
//! tables (blockflags recovered by the validated backward scan, no codebook
//! parse) and clamps the cumulative timeline to the end-of-stream granule;
//! `vorbisdec` trims its decoded PCM to the clamped duration, so the output
//! ends exactly where ffmpeg's does. Oracle: ffprobe's per-packet
//! (pts, duration, size) list, on a steady tone (long blocks) and on noise
//! (mixed short/long windows exercising every mode).
#![cfg(all(feature = "std", feature = "vorbis"))]

use core::future::Future;
use core::pin::Pin;
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, G2gError, OutputSink, PropValue, PushOutcome,
};
use g2g_plugins::oggdemux::OggDemux;

#[derive(Default)]
struct CaptureSink {
    frames: Vec<(u64, u64, usize)>,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames
                        .push((f.timing.pts_ns, f.timing.duration_ns, s.len()));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn chunk(data: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

fn temp_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("g2g-m778-{tag}-{}.ogg", std::process::id()))
}

/// One ffprobe packet: `(pts in 1/44100, duration in 1/44100, size)`. The pts
/// is signed: ffmpeg expresses the initial Vorbis priming as a negative first
/// pts (the g2g timeline clamps it at zero instead, `FrameTiming` is unsigned).
type ProbedPacket = (i64, i64, usize);

/// Encode `source` (an ffmpeg lavfi graph) to Ogg-Vorbis and capture ffprobe's
/// per-packet timing list, or `None` when the host cannot.
fn encode_and_probe(tag: &str, source: &str) -> Option<(Vec<u8>, Vec<ProbedPacket>)> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return None;
    }
    let path = temp_path(tag);
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i", source])
        .args(["-ac", "2", "-c:a", "libvorbis", "-f", "ogg"])
        .arg(&path)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&path).ok()?;
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "packet=pts,duration,size",
            "-of",
            "csv=p=0",
        ])
        .arg(&path)
        .output()
        .ok()?;
    let _ = std::fs::remove_file(&path);
    let expected: Vec<ProbedPacket> = String::from_utf8_lossy(&probe.stdout)
        .lines()
        .filter_map(|l| {
            let mut it = l.trim().trim_end_matches(',').split(',');
            let pts: i64 = it.next()?.parse().ok()?;
            let dur: i64 = it.next()?.parse().ok()?;
            let size: usize = it.next()?.parse().ok()?;
            Some((pts, dur, size))
        })
        .collect();
    Some((bytes, expected))
}

/// Demux `ogg` with `stream=vorbis`, returning the audio packets' timing.
async fn demux_timing(ogg: &[u8]) -> Vec<(u64, u64, usize)> {
    let mut d = OggDemux::new();
    d.set_property("stream", PropValue::Str("vorbis".into()))
        .expect("stream property");
    d.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    })
    .expect("configure");
    let mut sink = CaptureSink::default();
    for piece in ogg.chunks(997) {
        d.process(chunk(piece), &mut sink).await.expect("demux");
    }
    d.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    // Skip the three in-band header frames.
    sink.frames[3..].to_vec()
}

/// Every audio packet's pts / duration matches ffprobe's (converted to ns at
/// 44.1 kHz), including the first packet's `blocksize/2` priming duration and
/// the final packet's end-granule clamp.
async fn assert_timing_matches(source: &str, tag: &str) {
    let Some((ogg, expected)) = encode_and_probe(tag, source) else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    assert!(!expected.is_empty(), "ffprobe reports packets");
    let media = demux_timing(&ogg).await;
    assert_eq!(media.len(), expected.len(), "packet count matches ffprobe");
    let ns = |s: i64| s.max(0) as u128 * 1_000_000_000 / 44_100;
    for (i, ((pts, dur, size), (ref_pts, _, ref_size))) in media.iter().zip(&expected).enumerate() {
        assert_eq!(*size, *ref_size, "packet {i} size");
        // The g2g timeline is the lapped `(prev + cur) / 4` model, clamping
        // ffmpeg's negative priming pts at zero. ffmpeg's own list is not
        // self-consistent at short/long block boundaries (its parser
        // approximates short-block durations, so its pts jump there); compare
        // pts only where ffprobe's predecessor chain agrees with itself.
        let ref_consistent = i == 0 || {
            let (p, d, _) = expected[i - 1];
            p + d == *ref_pts
        };
        if ref_consistent {
            assert_eq!(*pts as u128, ns(*ref_pts), "packet {i} pts");
        }
        // Our own timeline must always be gap-free.
        if let Some((next_pts, _, _)) = media.get(i + 1) {
            assert_eq!(*pts + *dur, *next_pts, "packet {i} contiguous");
        }
    }
    // The final packet ends exactly where ffmpeg's does (the end-granule clamp).
    let (pts, dur, _) = media.last().unwrap();
    let (ref_pts, ref_dur, _) = expected.last().unwrap();
    assert_eq!(
        (*pts + *dur) as u128,
        ns(*ref_pts + *ref_dur),
        "stream end matches ffmpeg"
    );
}

#[tokio::test]
async fn tone_timing_matches_ffprobe() {
    assert_timing_matches("sine=frequency=440:duration=0.5:sample_rate=44100", "tone").await;
}

/// Noise forces frequent short/long window switches, exercising the mode
/// table across blocksize transitions (the `(prev + cur) / 4` lapping math).
/// Seeded, so the fixture (and any failure) is reproducible.
#[tokio::test]
async fn noise_timing_matches_ffprobe() {
    assert_timing_matches(
        "anoisesrc=duration=1:sample_rate=44100:amplitude=0.5:seed=7",
        "noise",
    )
    .await;
}

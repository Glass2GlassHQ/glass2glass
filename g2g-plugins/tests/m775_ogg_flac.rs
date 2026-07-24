//! M775: Ogg-FLAC. `oggdemux stream=flac` maps the `\x7fFLAC` logical stream:
//! caps from the embedded STREAMINFO, the native `fLaC` header forwarded
//! in-band, one FLAC frame per packet timed by block size, with ffprobe's
//! packet list as the framing oracle and ffmpeg's own decode as the PCM one
//! (FLAC is lossless, so the comparison is bit-exact). Bare `decodebin` sniffs
//! the codec via the primary-stream hook, and a `Caps::Audio{Flac}` decode
//! chain auto-inserts `flacparse` (the elementary `.flac` path).
#![cfg(all(feature = "std", feature = "ffmpeg"))]

use core::future::Future;
use core::pin::Pin;
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::parse_launch;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, G2gError, OutputSink, PropValue,
    PushOutcome,
};
use g2g_plugins::oggdemux::OggDemux;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl g2g_core::PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Parse, negotiate, and run a launch line; frames consumed at the sink.
async fn run_line(line: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = g2g_core::runtime::run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    stats.frames_consumed
}

/// The element type names in the built graph, in topological order.
fn chain_names(line: &str) -> Vec<String> {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let vg = graph.finish().expect("valid graph");
    vg.topo()
        .iter()
        .filter_map(|&n| vg.element(n).map(|e| e.log_category().to_string()))
        .collect()
}

#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    frames: Vec<(u64, Vec<u8>)>,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames.push((f.timing.pts_ns, s.to_vec()));
                    }
                }
                _ => {}
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

/// One ffprobe packet: `(pts in 1/44100, size)`.
type ProbedPacket = (u64, usize);

fn temp_path(tag: &str, ext: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("g2g-m775-{tag}-{}.{ext}", std::process::id()))
}

/// A real 0.5 s 44.1 kHz stereo tone encoded by ffmpeg into Ogg-FLAC, plus
/// ffprobe's packet list as the framing oracle, or `None` when the host cannot.
fn encode_ogg_flac(tag: &str) -> Option<(Vec<u8>, Vec<ProbedPacket>)> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return None;
    }
    let path = temp_path(tag, "oga");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.5:sample_rate=44100",
            "-ac",
            "2",
            "-c:a",
            "flac",
            "-f",
            "ogg",
        ])
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
            "packet=pts,size",
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
            let pts: u64 = it.next()?.parse().ok()?;
            let size: usize = it.next()?.parse().ok()?;
            Some((pts, size))
        })
        .collect();
    Some((bytes, expected))
}

/// ffmpeg's own decode of `path` to interleaved S16LE, the lossless PCM oracle.
fn ffmpeg_pcm(path: &std::path::Path) -> Option<Vec<u8>> {
    let out = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-i"])
        .arg(path)
        .args(["-f", "s16le", "-c:a", "pcm_s16le", "pipe:1"])
        .output()
        .ok()?;
    out.status.success().then_some(out.stdout)
}

/// `oggdemux stream=flac` maps the logical stream: STREAMINFO caps, the native
/// `fLaC` header first (last-block flag set), then ffprobe's exact packet list
/// with sample-accurate timestamps.
#[tokio::test]
async fn demuxed_frames_match_ffprobe_packets() {
    let Some((oga, expected)) = encode_ogg_flac("frames") else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    assert!(!expected.is_empty(), "ffprobe reports packets");

    let mut d = OggDemux::new();
    d.set_property("stream", PropValue::Str("flac".into()))
        .expect("stream property");
    d.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    })
    .expect("configure");
    let mut sink = CaptureSink::default();
    // Odd-sized chunks, so pages straddle input boundaries.
    for piece in oga.chunks(997) {
        d.process(chunk(piece), &mut sink).await.expect("demux");
    }
    d.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");

    assert_eq!(
        sink.caps,
        vec![Caps::Audio {
            format: AudioFormat::Flac,
            channels: 2,
            sample_rate: 44_100,
        }],
        "STREAMINFO caps"
    );

    // Frame 0 is the embedded native header, re-terminated for standalone use.
    let (first_pts, header) = &sink.frames[0];
    assert!(header.starts_with(b"fLaC"), "native header forwarded first");
    assert_eq!(header[4] & 0x80, 0x80, "last-metadata-block flag set");
    assert_eq!(*first_pts, 0);

    let media = &sink.frames[1..];
    assert_eq!(media.len(), expected.len(), "frame count matches ffprobe");
    for (i, ((pts_ns, data), (ref_pts, ref_size))) in media.iter().zip(&expected).enumerate() {
        assert_eq!(data.len(), *ref_size, "frame {i} size matches ffprobe");
        let ref_ns = *ref_pts as u128 * 1_000_000_000 / 44_100;
        assert_eq!(*pts_ns as u128, ref_ns, "frame {i} pts matches ffprobe");
    }
}

/// The full launch chain decodes Ogg-FLAC to PCM bit-exact with ffmpeg's own
/// decode of the same file.
#[tokio::test]
async fn launch_line_decodes_bit_exact_vs_ffmpeg() {
    let Some((oga, _)) = encode_ogg_flac("decode") else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    let src = temp_path("decode-src", "oga");
    std::fs::write(&src, &oga).unwrap();
    let reference = ffmpeg_pcm(&src).expect("ffmpeg decodes the fixture");
    let out = temp_path("decode-out", "raw");
    let _ = std::fs::remove_file(&out);

    let line = format!(
        "filesrc location={} ! oggdemux stream=flac ! ffmpegaudiodec ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=44100,channels=2 ! filesink location={}",
        src.display(),
        out.display()
    );
    assert!(run_line(&line).await > 0, "{line}");
    let pcm = std::fs::read(&out).expect("pcm written");
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
    assert_eq!(pcm, reference, "lossless decode is bit-exact vs ffmpeg");
}

/// Bare `decodebin` on an Ogg-FLAC file: the primary-stream hook sniffs the
/// `\x7fFLAC` packet and selects `oggdemux stream=flac`, so a FLAC (not Opus)
/// decode chain is plugged, with `flacparse` auto-inserted ahead of the decoder,
/// and the composed chain decodes bit-exact vs ffmpeg.
#[tokio::test]
async fn ogg_flac_bare_decodebin_plugs_flac_chain() {
    let Some((oga, _)) = encode_ogg_flac("autoplug") else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    let src = temp_path("autoplug", "oga");
    std::fs::write(&src, &oga).unwrap();
    let reference = ffmpeg_pcm(&src).expect("ffmpeg decodes the fixture");
    let sink_tail = "audioconvert ! audio/x-raw,format=S16LE,rate=44100,channels=2";
    let names = chain_names(&format!(
        "filesrc location={} ! decodebin ! {sink_tail} ! fakesink",
        src.display()
    ));

    let out = temp_path("autoplug-out", "raw");
    let _ = std::fs::remove_file(&out);
    let line = format!(
        "filesrc location={} ! decodebin ! {sink_tail} ! filesink location={}",
        src.display(),
        out.display()
    );
    assert!(run_line(&line).await > 0, "{line}");
    let pcm = std::fs::read(&out).expect("pcm written");
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
    assert_eq!(pcm, reference, "lossless decode is bit-exact vs ffmpeg");

    assert!(
        names.iter().any(|n| n == "OggDemux"),
        "oggdemux was plugged: {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "FlacParse"),
        "flacparse was auto-inserted: {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "FfmpegAudioDec"),
        "a FLAC decoder was plugged: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n == "OpusDec"),
        "the Opus default was overridden: {names:?}"
    );
}

/// Bare `decodebin` on an elementary `.flac` file auto-inserts `flacparse`
/// ahead of the decoder (M774 required an explicit `flacparse` in the line),
/// and the chain decodes bit-exact vs ffmpeg.
#[tokio::test]
async fn flac_file_bare_decodebin_inserts_flacparse() {
    let Some((_, _)) = encode_ogg_flac("probe") else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    // A native `.flac` sibling of the Ogg fixture.
    let src = temp_path("elementary", "flac");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.5:sample_rate=44100",
            "-ac",
            "2",
            "-c:a",
            "flac",
        ])
        .arg(&src)
        .status()
        .expect("ffmpeg runs");
    assert!(status.success(), "ffmpeg encodes .flac");
    let reference = ffmpeg_pcm(&src).expect("ffmpeg decodes the fixture");

    let sink_tail = "audioconvert ! audio/x-raw,format=S16LE,rate=44100,channels=2";
    let names = chain_names(&format!(
        "filesrc location={} ! decodebin ! {sink_tail} ! fakesink",
        src.display()
    ));
    assert!(
        names.iter().any(|n| n == "FlacParse"),
        "flacparse was auto-inserted: {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "FfmpegAudioDec"),
        "a FLAC decoder was plugged: {names:?}"
    );

    let out = temp_path("elementary-out", "raw");
    let _ = std::fs::remove_file(&out);
    let line = format!(
        "filesrc location={} ! decodebin ! {sink_tail} ! filesink location={}",
        src.display(),
        out.display()
    );
    assert!(run_line(&line).await > 0, "{line}");
    let pcm = std::fs::read(&out).expect("pcm written");
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
    assert_eq!(pcm, reference, "lossless decode is bit-exact vs ffmpeg");
}

/// A lone `playbin uri=` on an Ogg-FLAC file and on an elementary `.flac` file
/// builds and runs via the audio playbin hook (the `file://` fallback is a
/// self-demuxing MP4 video source, so both were unplayable before it).
#[tokio::test]
async fn playbin_plays_flac_and_ogg_flac() {
    let Some((oga, _)) = encode_ogg_flac("playbin") else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    let src = temp_path("playbin", "oga");
    std::fs::write(&src, &oga).unwrap();
    assert!(
        run_line(&format!("playbin uri=file://{}", src.display())).await > 0,
        "playbin plays ogg-flac"
    );
    let _ = std::fs::remove_file(&src);

    let src = temp_path("playbin", "flac");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.5:sample_rate=44100",
            "-ac",
            "2",
            "-c:a",
            "flac",
        ])
        .arg(&src)
        .status()
        .expect("ffmpeg runs");
    assert!(status.success(), "ffmpeg encodes .flac");
    assert!(
        run_line(&format!("playbin uri=file://{}", src.display())).await > 0,
        "playbin plays elementary flac"
    );
    let _ = std::fs::remove_file(&src);
}

/// Regression: an Ogg-Opus file through bare `decodebin` still plugs the Opus
/// decoder (the hook declines, leaving the demux's default Opus port).
#[cfg(feature = "opus")]
#[test]
fn ogg_opus_bare_decodebin_still_plugs_opus() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("opus", "opus");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.5:sample_rate=48000",
            "-ac",
            "2",
            "-c:a",
            "libopus",
            "-f",
            "ogg",
        ])
        .arg(&src)
        .status()
        .expect("ffmpeg runs");
    if !status.success() {
        eprintln!("skipping: ffmpeg lacks libopus");
        return;
    }
    let names = chain_names(&format!(
        "filesrc location={} ! decodebin ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=48000,channels=2 ! fakesink",
        src.display()
    ));
    let _ = std::fs::remove_file(&src);
    assert!(
        names.iter().any(|n| n == "OpusDec"),
        "the Opus chain is unchanged: {names:?}"
    );
}

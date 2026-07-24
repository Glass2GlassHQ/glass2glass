//! M777: Vorbis decode. `oggdemux stream=vorbis` forwards the three header
//! packets in-band and the audio packets whole; the pure-Rust `vorbisdec`
//! (symphonia) stashes ident + setup, decodes, and stamps PCM from decoded
//! sample counts. Oracles: ffprobe's packet list for the demux framing, and
//! ffmpeg's own decode for the PCM (lossy codec, same bitstream: samples must
//! agree within 1 LSB of int16 rounding; g2g does not yet trim the final
//! block to the end granule, so a bounded tail beyond ffmpeg's length is
//! allowed). Auto paths: bare `decodebin` sniffs the codec, `playbin uri=`
//! plays a `.ogg` end to end.
#![cfg(all(feature = "std", feature = "vorbis"))]

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
    frames: Vec<Vec<u8>>,
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
                        self.frames.push(s.to_vec());
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

fn temp_path(tag: &str, ext: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("g2g-m777-{tag}-{}.{ext}", std::process::id()))
}

/// A real 0.5 s 44.1 kHz stereo tone encoded by ffmpeg (libvorbis) into Ogg,
/// plus ffprobe's packet sizes as the framing oracle, or `None` when the host
/// cannot.
fn encode_ogg_vorbis(tag: &str) -> Option<(Vec<u8>, Vec<usize>)> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return None;
    }
    let path = temp_path(tag, "ogg");
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
            "libvorbis",
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
            "packet=size",
            "-of",
            "csv=p=0",
        ])
        .arg(&path)
        .output()
        .ok()?;
    let _ = std::fs::remove_file(&path);
    let sizes: Vec<usize> = String::from_utf8_lossy(&probe.stdout)
        .lines()
        .filter_map(|l| l.trim().trim_end_matches(',').parse().ok())
        .collect();
    Some((bytes, sizes))
}

/// ffmpeg's own decode of `path` to interleaved S16LE, the PCM oracle.
fn ffmpeg_pcm(path: &std::path::Path) -> Option<Vec<u8>> {
    let out = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-i"])
        .arg(path)
        .args(["-f", "s16le", "-c:a", "pcm_s16le", "pipe:1"])
        .output()
        .ok()?;
    out.status.success().then_some(out.stdout)
}

/// Assert `pcm` matches the ffmpeg reference: every overlapping sample within
/// 1 LSB (independent float->int rounding), at least as long as the reference,
/// and any untrimmed tail bounded by one max Vorbis block (8192 frames).
fn assert_pcm_matches(pcm: &[u8], reference: &[u8], channels: usize) {
    assert!(
        pcm.len() >= reference.len(),
        "g2g decode at least as long as ffmpeg's ({} vs {})",
        pcm.len(),
        reference.len()
    );
    assert!(
        pcm.len() - reference.len() <= 8192 * channels * 2,
        "untrimmed tail bounded by one max block ({} extra bytes)",
        pcm.len() - reference.len()
    );
    let mut max_diff = 0i32;
    for (a, b) in pcm.chunks_exact(2).zip(reference.chunks_exact(2)) {
        let x = i16::from_le_bytes([a[0], a[1]]) as i32;
        let y = i16::from_le_bytes([b[0], b[1]]) as i32;
        max_diff = max_diff.max((x - y).abs());
    }
    assert!(
        max_diff <= 1,
        "samples within 1 LSB of ffmpeg, got {max_diff}"
    );
}

/// `oggdemux stream=vorbis` forwards the three prefixed headers first, then
/// exactly ffprobe's packet list.
#[tokio::test]
async fn demuxed_packets_match_ffprobe() {
    let Some((ogg, sizes)) = encode_ogg_vorbis("frames") else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    assert!(!sizes.is_empty(), "ffprobe reports packets");

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

    assert_eq!(
        sink.caps,
        vec![Caps::Audio {
            format: AudioFormat::Vorbis,
            channels: 2,
            sample_rate: 44_100,
        }],
        "identification-header caps"
    );
    assert!(sink.frames.len() > 3, "headers + audio came out");
    assert!(sink.frames[0].starts_with(b"\x01vorbis"), "ident first");
    assert!(sink.frames[1].starts_with(b"\x03vorbis"), "comment second");
    assert!(sink.frames[2].starts_with(b"\x05vorbis"), "setup third");
    let media: Vec<usize> = sink.frames[3..].iter().map(|f| f.len()).collect();
    assert_eq!(media, sizes, "audio packets match ffprobe's list");
}

/// The explicit launch chain decodes to PCM matching ffmpeg's own decode.
#[tokio::test]
async fn launch_line_decodes_matching_ffmpeg() {
    let Some((ogg, _)) = encode_ogg_vorbis("decode") else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    let src = temp_path("decode-src", "ogg");
    std::fs::write(&src, &ogg).unwrap();
    let reference = ffmpeg_pcm(&src).expect("ffmpeg decodes the fixture");
    let out = temp_path("decode-out", "raw");
    let _ = std::fs::remove_file(&out);

    let line = format!(
        "filesrc location={} ! oggdemux stream=vorbis ! vorbisdec ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=44100,channels=2 ! filesink location={}",
        src.display(),
        out.display()
    );
    assert!(run_line(&line).await > 0, "{line}");
    let pcm = std::fs::read(&out).expect("pcm written");
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
    assert_pcm_matches(&pcm, &reference, 2);
}

/// Bare `decodebin` sniffs the Vorbis stream (primary-stream hook) and plugs
/// `oggdemux stream=vorbis ! vorbisdec`; the chain decodes matching ffmpeg.
#[tokio::test]
async fn bare_decodebin_plugs_vorbis_chain() {
    let Some((ogg, _)) = encode_ogg_vorbis("autoplug") else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    let src = temp_path("autoplug", "ogg");
    std::fs::write(&src, &ogg).unwrap();
    let reference = ffmpeg_pcm(&src).expect("ffmpeg decodes the fixture");
    let sink_tail = "audioconvert ! audio/x-raw,format=S16LE,rate=44100,channels=2";
    let names = chain_names(&format!(
        "filesrc location={} ! decodebin ! {sink_tail} ! fakesink",
        src.display()
    ));
    assert!(
        names.iter().any(|n| n == "OggDemux") && names.iter().any(|n| n == "VorbisDec"),
        "vorbis chain was plugged: {names:?}"
    );

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
    assert_pcm_matches(&pcm, &reference, 2);
}

/// A lone `playbin uri=` on an Ogg-Vorbis file builds and runs via the audio
/// playbin hook.
#[tokio::test]
async fn playbin_plays_ogg_vorbis() {
    let Some((ogg, _)) = encode_ogg_vorbis("playbin") else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    let src = temp_path("playbin", "ogg");
    std::fs::write(&src, &ogg).unwrap();
    assert!(
        run_line(&format!("playbin uri=file://{}", src.display())).await > 0,
        "playbin plays ogg-vorbis"
    );
    let _ = std::fs::remove_file(&src);
}

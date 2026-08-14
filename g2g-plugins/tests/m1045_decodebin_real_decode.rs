//! M1045 - a `decodebin` line RUN over real media. The other decodebin tests
//! (M193, M196, M482) stop at parse time: they assert the spliced chain, not that
//! a single decoded pixel came out. This runs `filesrc ! decodebin ! ...` over the
//! committed A/V MP4 fixture to EOS and checks the output against ffmpeg's own
//! decode of the same file, so the auto-plugged chain is validated by its pixels.
//!
//! A demuxed file negotiates placeholder geometry (the decoder has not seen an
//! SPS when the solver runs) and refines it with the first `CapsChanged`, so the
//! geometry assertion reads the observer's runtime caps, not the solved ones.
//!
//! Needs decoders in the auto-plug pool (ffmpeg).

#![cfg(all(feature = "std", feature = "ffmpeg"))]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::runtime::{parse_launch, run_graph_observed, Observer, RunStats, TelemetrySnapshot};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

/// A 0.28 s H.264 + AAC MP4 muxed by ffmpeg. ffprobe reports 64x48 @ 25 fps and
/// 7 video frames, plus 14 AAC frames of mono 44.1 kHz audio.
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av_h264_aac.mp4"
);
const FIXTURE_WIDTH: usize = 64;
const FIXTURE_HEIGHT: usize = 48;
const FIXTURE_VIDEO_FRAMES: usize = 7;
const FIXTURE_AUDIO_SAMPLES: usize = 14 * 1024;
/// I420 is 8-bit luma plus two quarter-size chroma planes.
const I420_FRAME_BYTES: usize = FIXTURE_WIDTH * FIXTURE_HEIGHT * 3 / 2;
const S16_BYTES: usize = 2;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m1045-{tag}-{}.raw", std::process::id()))
}

/// Parse, negotiate, and run a launch line to EOS, with the caps each link
/// really carried.
async fn run_line(line: &str) -> (RunStats, TelemetrySnapshot) {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let observer = Observer::new();
    let stats = run_graph_observed(graph, &ZeroClock, 4, &observer, None)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    (stats, observer.snapshot())
}

/// The caps every link carried while running, as `gst`-style strings.
fn observed_caps(snapshot: &TelemetrySnapshot) -> Vec<String> {
    snapshot
        .edges
        .iter()
        .filter_map(|e| e.observed_caps.clone())
        .collect()
}

/// ffmpeg's own decode of the fixture's video track to I420, the pixel oracle:
/// the auto-plugged decoder is the same libavcodec, so this is bit-exact.
fn ffmpeg_i420() -> Option<Vec<u8>> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i", FIXTURE])
        .args(["-an", "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"])
        .output()
        .ok()?;
    out.status.success().then_some(out.stdout)
}

/// A bare `decodebin` over the committed MP4 decodes real pixels: the raw link
/// carries the fixture's true geometry while running, and the bytes reaching the
/// sink are ffmpeg's own decode of the same file, frame count included.
#[tokio::test]
async fn decodebin_decodes_the_fixture_bit_exact() {
    let Some(reference) = ffmpeg_i420() else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    assert_eq!(
        reference.len(),
        FIXTURE_VIDEO_FRAMES * I420_FRAME_BYTES,
        "ffprobe's frame count and geometry agree with ffmpeg's decode"
    );

    let out = temp_path("video");
    let _ = std::fs::remove_file(&out);
    // The capsfilter pins the pixel format only: the geometry has to come from
    // the decoded stream.
    let line = format!(
        "filesrc location={FIXTURE} ! decodebin ! videoconvert ! video/x-raw,format=I420 \
         ! filesink location={}",
        out.display()
    );
    let (stats, snapshot) = run_line(&line).await;
    let decoded = std::fs::read(&out).expect("decoded frames written");
    let _ = std::fs::remove_file(&out);

    let raw = observed_caps(&snapshot);
    assert!(
        raw.iter().any(|c| c.contains("video/x-raw")
            && c.contains("format=I420")
            && c.contains(&format!("width={FIXTURE_WIDTH}"))
            && c.contains(&format!("height={FIXTURE_HEIGHT}"))),
        "a link carried decoded {FIXTURE_WIDTH}x{FIXTURE_HEIGHT} I420: {raw:?}"
    );
    assert_eq!(
        decoded.len() / I420_FRAME_BYTES,
        FIXTURE_VIDEO_FRAMES,
        "every coded frame came out decoded ({} bytes)",
        decoded.len()
    );
    assert_eq!(
        decoded, reference,
        "decoded pixels match ffmpeg's own decode"
    );
    assert!(stats.frames_consumed > 0, "the sink consumed frames");
    eprintln!(
        "decodebin decoded {} I420 frames of {FIXTURE_WIDTH}x{FIXTURE_HEIGHT} ({} bytes), \
         bit-exact vs ffmpeg; frames_consumed={}",
        decoded.len() / I420_FRAME_BYTES,
        decoded.len(),
        stats.frames_consumed
    );
}

/// `decodebin name=d` fans the same file out to both tracks and RUNS: the video
/// branch writes ffmpeg's pixels, the audio branch one PCM sample per coded AAC
/// sample. The AAC decode is not compared byte for byte because ffmpeg's own
/// decode trims the encoder priming this pipeline keeps.
#[tokio::test]
async fn decodebin_fanout_runs_both_branches() {
    let Some(reference) = ffmpeg_i420() else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    let video_out = temp_path("fanout-video");
    let audio_out = temp_path("fanout-audio");
    let _ = std::fs::remove_file(&video_out);
    let _ = std::fs::remove_file(&audio_out);

    let line = format!(
        "filesrc location={FIXTURE} ! decodebin name=d  \
         d.video_0 ! videoconvert ! video/x-raw,format=I420 ! filesink location={}  \
         d.audio_0 ! audioconvert ! audio/x-raw,format=S16LE,rate=44100,channels=1 \
         ! filesink location={}",
        video_out.display(),
        audio_out.display()
    );
    let (_, snapshot) = run_line(&line).await;
    let video = std::fs::read(&video_out).expect("video branch wrote frames");
    let audio = std::fs::read(&audio_out).expect("audio branch wrote samples");
    let _ = std::fs::remove_file(&video_out);
    let _ = std::fs::remove_file(&audio_out);

    assert_eq!(video, reference, "video branch decoded ffmpeg's pixels");
    assert_eq!(
        audio.len() / S16_BYTES,
        FIXTURE_AUDIO_SAMPLES,
        "audio branch decoded every AAC frame ({} bytes)",
        audio.len()
    );
    let caps = observed_caps(&snapshot);
    assert!(
        caps.iter().any(|c| c.contains("audio/x-raw")),
        "the audio branch carried decoded PCM: {caps:?}"
    );
    eprintln!(
        "decodebin fan-out decoded {} video frames and {} audio samples",
        video.len() / I420_FRAME_BYTES,
        audio.len() / S16_BYTES
    );
}

//! M1076 YUV4MPEG2: `y4mdec` reads a `.y4m` file's raw frames, `typefind` and
//! `decodebin` reach it by content, and `y4menc` writes a file back that ffmpeg
//! and GStreamer read.
//!
//! The fixtures and every expected value come from ffmpeg / ffprobe, checked in
//! next to this test so it needs neither at run time:
//!
//! ```text
//! ffmpeg -v error -y -f lavfi -i testsrc2=size=64x48:rate=25 -frames:v 5 \
//!   -pix_fmt yuv420p tests/fixtures/y4m_64x48_i420.y4m
//! ffmpeg -v error -y -f lavfi -i testsrc2=size=64x48:rate=25 -frames:v 5 \
//!   -pix_fmt yuv422p tests/fixtures/y4m_64x48_i422.y4m
//! ffprobe -v error -show_format -show_streams -count_frames -of json <file> > <file>.json
//! ```
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{is_raw_video, parse_launch, run_graph, GraphNode, Registry};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, Dim, G2gError, Graph, Interlace, OutputSink,
    PipelineClock, PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::registry::default_registry;
use g2g_plugins::y4m::Y4mDec;

const I420_FIXTURE: &str = "y4m_64x48_i420.y4m";
const I422_FIXTURE: &str = "y4m_64x48_i422.y4m";

/// Bytes per input buffer when the decoder is driven directly: an odd size, so
/// frames straddle buffer boundaries the way `filesrc` chunks make them.
const CHUNK_LEN: usize = 997;

/// The line before each frame's planes, and what separates the stream header
/// from the first of them.
const FRAME_MARKER: &[u8] = b"FRAME\n";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m1076-{tag}-{}.y4m", std::process::id()))
}

/// What the checked-in `ffprobe` output says about a fixture.
struct Probe {
    width: u32,
    height: u32,
    format: RawVideoFormat,
    framerate_q16: u32,
    frames: u64,
}

impl Probe {
    /// Read the `ffprobe -of json` output checked in beside a fixture. The
    /// fixtures are single-stream, so the first value of each key is the one
    /// wanted and the reader needs no JSON object model.
    fn read(name: &str) -> Probe {
        let path = fixture_path(name).with_extension("json");
        let json = std::fs::read_to_string(&path).expect("the probe output is checked in");
        let value =
            |key: &str| json_value(&json, key).unwrap_or_else(|| panic!("{key} in {path:?}"));
        let rate = value("r_frame_rate");
        let (num, den) = rate.split_once('/').expect("a num/den framerate");
        Probe {
            width: value("width").parse().expect("a width"),
            height: value("height").parse().expect("a height"),
            format: pixel_format(&value("pix_fmt")),
            framerate_q16: ((num.parse::<u64>().expect("a numerator") << 16)
                / den.parse::<u64>().expect("a denominator")) as u32,
            frames: value("nb_read_frames").parse().expect("a frame count"),
        }
    }

    fn caps(&self) -> Caps {
        Caps::RawVideo {
            format: self.format,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(self.framerate_q16),
            interlace: Interlace::Progressive,
        }
    }

    fn frame_bytes(&self) -> usize {
        self.format
            .unpadded_frame_bytes(self.width, self.height)
            .expect("a planar frame size") as usize
    }

    fn frame_period_ns(&self) -> u64 {
        1_000_000_000u64 * 65536 / self.framerate_q16 as u64
    }
}

/// The g2g format an ffprobe `pix_fmt` names. Only the two the fixtures use:
/// anything else means the fixtures were regenerated with a different `-pix_fmt`
/// and the test should say so rather than guess.
fn pixel_format(pix_fmt: &str) -> RawVideoFormat {
    match pix_fmt {
        "yuv420p" => RawVideoFormat::I420,
        "yuv422p" => RawVideoFormat::I422,
        other => panic!("no g2g format for the probed pix_fmt {other}"),
    }
}

/// The value of `"key"` in `json`, as text: a quoted string without its quotes,
/// or a bare number.
fn json_value(json: &str, key: &str) -> Option<String> {
    let at = json.find(&format!("\"{key}\""))? + key.len() + 2;
    let rest = json[at..].trim_start().strip_prefix(':')?.trim_start();
    match rest.strip_prefix('"') {
        Some(quoted) => Some(quoted[..quoted.find('"')?].to_string()),
        None => Some(
            rest[..rest.find([',', '\n', '}'])?]
                .trim()
                .trim_end_matches(',')
                .to_string(),
        ),
    }
}

/// A y4m stream split into its header line (without the newline) and one byte
/// run per frame, the `FRAME` lines dropped. The test's own reader, so what it
/// compares against is the file rather than the element under test.
fn split_stream(stream: &[u8], frame_bytes: usize) -> (String, Vec<Vec<u8>>) {
    let end = stream
        .iter()
        .position(|&b| b == b'\n')
        .expect("a stream header line");
    let header = String::from_utf8(stream[..end].to_vec()).expect("an ASCII header");
    let mut frames = Vec::new();
    let mut at = end + 1;
    while at < stream.len() {
        assert_eq!(
            &stream[at..at + FRAME_MARKER.len()],
            FRAME_MARKER,
            "a FRAME line introduces every frame"
        );
        at += FRAME_MARKER.len();
        frames.push(stream[at..at + frame_bytes].to_vec());
        at += frame_bytes;
    }
    (header, frames)
}

/// The `W` / `H` / `F` / `I` / `C` parameters of a stream header, keyed by their
/// tag letter. The optional `A` (pixel aspect) and `X` (writer extensions) are
/// kept too, so a caller can assert on what a writer did or did not emit.
fn header_parameters(header: &str) -> Vec<(char, String)> {
    header
        .strip_prefix("YUV4MPEG2 ")
        .expect("the y4m signature")
        .split(' ')
        .filter(|token| !token.is_empty())
        .map(|token| {
            let mut characters = token.chars();
            let tag = characters.next().expect("a parameter tag");
            (tag, characters.as_str().to_string())
        })
        .collect()
}

fn parameter(header: &str, tag: char) -> Option<String> {
    header_parameters(header)
        .into_iter()
        .find(|(t, _)| *t == tag)
        .map(|(_, value)| value)
}

#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    frames: Vec<Frame>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        match packet_slot.take().expect("poll_push without a packet") {
            PipelinePacket::CapsChanged(caps) => self.caps.push(caps),
            PipelinePacket::DataFrame(frame) => self.frames.push(frame),
            _ => {}
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn y4m_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Y4m,
    }
}

/// Drive `Y4mDec` over a whole file in [`CHUNK_LEN`] pieces, then end the stream.
async fn decode_fixture(name: &str) -> CaptureSink {
    let bytes = std::fs::read(fixture_path(name)).expect("the fixture is checked in");
    let mut element = Y4mDec::new();
    element
        .configure_pipeline(&y4m_caps())
        .expect("a y4m byte stream is accepted");
    let mut sink = CaptureSink::default();
    for piece in bytes.chunks(CHUNK_LEN) {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(piece.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        element
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .expect("the stream parses");
    }
    element
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("the stream ends on a frame boundary");
    sink
}

#[tokio::test]
async fn every_fixture_frame_reaches_the_sink() {
    for name in [I420_FIXTURE, I422_FIXTURE] {
        let probe = Probe::read(name);
        let reg = default_registry();
        let line = format!(
            "filesrc location={} ! y4mdec ! fakesink",
            fixture_path(name).display()
        );
        let graph = parse_launch(&reg, &line).expect("the line parses and negotiates");
        let stats = run_graph(graph, &ZeroClock, 4)
            .await
            .expect("the pipeline runs");
        assert_eq!(
            stats.frames_consumed, probe.frames,
            "{name}: one buffer per y4m FRAME"
        );
    }
}

#[tokio::test]
async fn decoded_frames_carry_the_probed_caps_timing_and_bytes() {
    for name in [I420_FIXTURE, I422_FIXTURE] {
        let probe = Probe::read(name);
        let sink = decode_fixture(name).await;

        assert_eq!(
            sink.caps,
            vec![probe.caps()],
            "{name}: the concrete geometry, format and rate are announced once, before the first frame"
        );
        assert_eq!(sink.frames.len() as u64, probe.frames, "{name}");

        let period_ns = probe.frame_period_ns();
        for (index, frame) in sink.frames.iter().enumerate() {
            assert_eq!(
                frame.timing.pts_ns,
                index as u64 * period_ns,
                "{name}: frame {index} sits on the frame grid"
            );
            assert_eq!(frame.timing.duration_ns, period_ns, "{name}: frame {index}");
        }

        // The first frame's bytes are the file's own, straight after the first
        // FRAME line.
        let file = std::fs::read(fixture_path(name)).expect("the fixture is checked in");
        let (_, frames) = split_stream(&file, probe.frame_bytes());
        assert_eq!(
            sink.frames[0]
                .domain
                .as_system_slice()
                .expect("system bytes"),
            frames[0].as_slice(),
            "{name}: the first frame is copied byte for byte out of the file"
        );
    }
}

#[tokio::test]
async fn typefind_and_decodebin_reach_the_decoder_without_naming_it() {
    let probe = Probe::read(I420_FIXTURE);
    let path = fixture_path(I420_FIXTURE);
    let reg = default_registry();
    for line in [
        format!(
            "filesrc location={} ! typefind ! y4mdec ! fakesink",
            path.display()
        ),
        format!("filesrc location={} ! decodebin ! fakesink", path.display()),
    ] {
        let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
        let stats = run_graph(graph, &ZeroClock, 4)
            .await
            .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
        assert_eq!(stats.frames_consumed, probe.frames, "`{line}`");
    }
}

/// The auto-plug search itself, so the element `decodebin` picked is named.
#[test]
fn decodebin_splices_the_y4m_decoder() {
    let reg: Registry = default_registry();
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(FileSrc::new(
        fixture_path(I420_FIXTURE),
        y4m_caps(),
    )));
    let sink = graph.add_sink(GraphNode::element(FakeSink::new()));
    let inserted = reg
        .decodebin(&mut graph, src, sink, &y4m_caps(), &is_raw_video, 4)
        .expect("a y4m byte stream reaches raw video");
    assert_eq!(
        inserted.len(),
        1,
        "the decoder alone, no parser ahead of it"
    );
    assert_eq!(
        graph
            .element(inserted[0])
            .expect("the spliced decoder")
            .log_category(),
        g2g_core::log::short_type_name::<Y4mDec>(),
    );
}

/// A decode / encode round trip reproduces the fixture's frames exactly and a
/// header describing the same stream. Not a whole-file comparison: ffmpeg writes
/// an `A` pixel-aspect parameter and an `X` extension g2g's caps carry nothing
/// to reproduce, so the header is compared parameter by parameter.
#[tokio::test]
async fn a_round_trip_reproduces_the_frames_and_the_header_fields() {
    for name in [I420_FIXTURE, I422_FIXTURE] {
        let probe = Probe::read(name);
        let out = temp_path(name);
        let _ = std::fs::remove_file(&out);
        let reg = default_registry();
        let line = format!(
            "filesrc location={} ! y4mdec ! y4menc ! filesink location={}",
            fixture_path(name).display(),
            out.display()
        );
        let graph = parse_launch(&reg, &line).expect("the line parses and negotiates");
        run_graph(graph, &ZeroClock, 4)
            .await
            .expect("the pipeline runs");

        let source = std::fs::read(fixture_path(name)).expect("the fixture is checked in");
        let written = std::fs::read(&out).expect("the round trip wrote a file");
        let _ = std::fs::remove_file(&out);

        let (source_header, source_frames) = split_stream(&source, probe.frame_bytes());
        let (written_header, written_frames) = split_stream(&written, probe.frame_bytes());
        assert_eq!(
            written_frames, source_frames,
            "{name}: every frame's planes survive the round trip"
        );
        assert_eq!(written_frames.len() as u64, probe.frames, "{name}");
        for tag in ['W', 'H', 'F', 'I', 'C'] {
            assert_eq!(
                parameter(&written_header, tag),
                parameter(&source_header, tag),
                "{name}: the {tag} parameter of `{written_header}` vs `{source_header}`"
            );
        }
        assert_eq!(
            parameter(&written_header, 'A'),
            None,
            "{name}: no pixel aspect is claimed, the caps carry none"
        );
    }
}

/// The written file as ffprobe reads it, `None` when ffprobe is not installed
/// (CI lacks it).
fn ffprobe_json(path: &std::path::Path) -> Option<String> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-show_format", "-show_streams"])
        .args(["-count_frames", "-of", "json"])
        .arg(path)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `videotestsrc` is RGBA only and y4m holds planar YUV, so `videoconvert` sits
/// between them: the encoder refusing RGBA at negotiation is what puts it there.
#[tokio::test]
async fn an_encoded_file_reads_back_in_ffprobe() {
    const FRAMES: u64 = 3;
    // videotestsrc's registered defaults.
    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 240;

    let out = temp_path("encoded");
    let _ = std::fs::remove_file(&out);
    let reg = default_registry();
    let line = format!(
        "videotestsrc num-buffers={FRAMES} ! videoconvert ! y4menc ! filesink location={}",
        out.display()
    );
    let graph = parse_launch(&reg, &line).expect("the line parses and negotiates");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");

    let Some(json) = ffprobe_json(&out) else {
        eprintln!("skipping the ffprobe read-back: no ffprobe on PATH");
        let _ = std::fs::remove_file(&out);
        return;
    };
    let value = |key: &str| json_value(&json, key).unwrap_or_else(|| panic!("{key} in {json}"));
    assert_eq!(value("width"), WIDTH.to_string());
    assert_eq!(value("height"), HEIGHT.to_string());
    assert_eq!(value("nb_read_frames"), FRAMES.to_string());
    assert_eq!(
        value("pix_fmt"),
        "yuv420p",
        "videoconvert picks the first planar format y4menc accepts"
    );
    let _ = std::fs::remove_file(&out);
}

/// GStreamer's own `y4mdec` reads what `y4menc` wrote. Ignored by default: it
/// needs `gst-launch-1.0` installed, which CI does not have.
///
/// ```text
/// cargo test -p g2g-plugins --features std --test m1076_y4m -- --ignored
/// ```
#[tokio::test]
#[ignore]
async fn gstreamer_reads_an_encoded_file() {
    let out = temp_path("gst-interop");
    let _ = std::fs::remove_file(&out);
    let reg = default_registry();
    let line = format!(
        "videotestsrc num-buffers=3 ! videoconvert ! y4menc ! filesink location={}",
        out.display()
    );
    let graph = parse_launch(&reg, &line).expect("the line parses and negotiates");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");

    let status = Command::new("gst-launch-1.0")
        .arg("-q")
        .args(["filesrc", &format!("location={}", out.display())])
        .args(["!", "y4mdec", "!", "fakesink"])
        .status()
        .expect("gst-launch-1.0 is installed");
    let _ = std::fs::remove_file(&out);
    assert!(status.success(), "gst-launch-1.0 read the file: {status}");
}

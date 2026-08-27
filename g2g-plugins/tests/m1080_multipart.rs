//! M1080 MIME multipart: `multipartdemux` splits the
//! `multipart/x-mixed-replace` stream an MJPEG-over-HTTP server pushes into its
//! JPEG parts, and `multipartmux` writes one back that ffmpeg and GStreamer read.
//!
//! The fixture is ffmpeg's own mpjpeg output, checked in next to this test so it
//! needs neither ffmpeg nor a camera at run time:
//!
//! ```text
//! ffmpeg -v error -y -f lavfi -i testsrc2=size=64x48 -t 1 -r 5 \
//!   -f mpjpeg tests/fixtures/multipart_64x48_jpeg.mjpg
//! ```
//!
//! Every expected value is read out of that file: the part count, and each
//! part's bytes.
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, G2gError, OutputSink, PipelineClock, PushOutcome,
};
use g2g_plugins::multipart::MultipartDemux;
use g2g_plugins::registry::default_registry;

const FIXTURE: &str = "multipart_64x48_jpeg.mjpg";

/// The boundary ffmpeg's mpjpeg muxer writes.
const FIXTURE_BOUNDARY: &str = "ffmpeg";

/// Bytes per input buffer when the demuxer is driven directly: an odd size, so
/// parts straddle buffer boundaries the way `filesrc` chunks make them.
const CHUNK_LEN: usize = 997;

/// The JPEG start-of-image marker every part's body opens with.
const SOI: [u8; 2] = [0xFF, 0xD8];

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
    std::env::temp_dir().join(format!("g2g-m1080-{tag}-{}.mjpg", std::process::id()))
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// The boundary line that opens a part in `boundary`'s stream. ffmpeg writes a
/// bare boundary after the last part too, so a header has to follow for the line
/// to open one.
fn part_opener(boundary: &str) -> Vec<u8> {
    format!("--{boundary}\r\nContent-").into_bytes()
}

/// The parts of a multipart stream, read the plain way (no g2g element): from
/// each opening boundary, skip the header block, then take the body up to the
/// CRLF before the next boundary.
fn parts_of(bytes: &[u8], boundary: &str) -> Vec<Vec<u8>> {
    let opener = part_opener(boundary);
    let blank_line = b"\r\n\r\n";
    let terminator = format!("\r\n--{boundary}").into_bytes();
    let mut parts = Vec::new();
    let mut at = 0;
    while let Some(rel) = find(&bytes[at..], &opener) {
        let headers = at + rel + opener.len();
        let Some(blank) = find(&bytes[headers..], blank_line) else {
            break;
        };
        let body = headers + blank + blank_line.len();
        let end = body + find(&bytes[body..], &terminator).expect("a part ends at a boundary");
        parts.push(Vec::from(&bytes[body..end]));
        at = end;
    }
    parts
}

/// How many parts a stream carries, counted from its boundary lines alone.
fn part_count(bytes: &[u8], boundary: &str) -> usize {
    let opener = part_opener(boundary);
    bytes
        .windows(opener.len())
        .filter(|window| *window == opener.as_slice())
        .count()
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

/// Drive `MultipartDemux` over a whole stream in [`CHUNK_LEN`] pieces, then end
/// it.
async fn demux(bytes: &[u8]) -> CaptureSink {
    let mut element = MultipartDemux::new();
    element
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Multipart,
        })
        .expect("a multipart byte stream is accepted");
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
        .expect("the stream ends between parts");
    sink
}

#[test]
fn the_fixture_carries_the_parts_its_boundary_lines_promise() {
    let bytes = std::fs::read(fixture_path(FIXTURE)).expect("the fixture is checked in");
    let parts = parts_of(&bytes, FIXTURE_BOUNDARY);
    assert_eq!(parts.len(), part_count(&bytes, FIXTURE_BOUNDARY));
    assert!(!parts.is_empty(), "ffmpeg wrote at least one part");
    for (index, part) in parts.iter().enumerate() {
        assert!(part.starts_with(&SOI), "part {index} is a JPEG");
    }
}

#[tokio::test]
async fn every_part_reaches_the_sink() {
    let bytes = std::fs::read(fixture_path(FIXTURE)).expect("the fixture is checked in");
    let reg = default_registry();
    let line = format!(
        "filesrc location={} ! multipartdemux ! fakesink",
        fixture_path(FIXTURE).display()
    );
    let graph = parse_launch(&reg, &line).expect("the line parses and negotiates");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    assert_eq!(
        stats.frames_consumed as usize,
        part_count(&bytes, FIXTURE_BOUNDARY),
        "one buffer per multipart part"
    );
}

#[tokio::test]
async fn each_demuxed_part_is_the_body_between_its_headers_and_its_boundary() {
    let bytes = std::fs::read(fixture_path(FIXTURE)).expect("the fixture is checked in");
    let expected = parts_of(&bytes, FIXTURE_BOUNDARY);
    let sink = demux(&bytes).await;

    assert_eq!(sink.frames.len(), expected.len());
    assert_eq!(
        sink.caps.len(),
        1,
        "the JPEG caps are announced once, before the first part"
    );
    for (index, (frame, want)) in sink.frames.iter().zip(&expected).enumerate() {
        let got = frame.domain.as_system_slice().expect("system bytes");
        assert!(got.starts_with(&SOI), "part {index} starts at the SOI");
        assert_eq!(got, want.as_slice(), "part {index} is copied byte for byte");
        assert_eq!(
            frame.timing.pts_ns, 0,
            "the transport carries no timestamps"
        );
        assert!(frame.timing.keyframe, "every JPEG decodes on its own");
    }
}

/// The demuxer feeds the JPEG decoder without a parser between them: one part is
/// one access unit.
#[cfg(feature = "mjpeg")]
#[tokio::test]
async fn the_parts_decode() {
    let bytes = std::fs::read(fixture_path(FIXTURE)).expect("the fixture is checked in");
    let reg = default_registry();
    let line = format!(
        "filesrc location={} ! multipartdemux ! mjpegdec ! fakesink",
        fixture_path(FIXTURE).display()
    );
    let graph = parse_launch(&reg, &line).expect("the line parses and negotiates");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    assert_eq!(
        stats.frames_consumed as usize,
        part_count(&bytes, FIXTURE_BOUNDARY),
        "every part decoded"
    );
}

/// Write the fixture's parts back out under ffmpeg's own boundary, returning the
/// muxed file's path. `tag` names the file, so tests running side by side do not
/// share one.
async fn remux(tag: &str) -> PathBuf {
    let out = temp_path(tag);
    let _ = std::fs::remove_file(&out);
    let reg = default_registry();
    let line = format!(
        "filesrc location={} ! multipartdemux ! multipartmux boundary={FIXTURE_BOUNDARY} ! filesink location={}",
        fixture_path(FIXTURE).display(),
        out.display()
    );
    let graph = parse_launch(&reg, &line).expect("the line parses and negotiates");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    out
}

#[tokio::test]
async fn a_round_trip_reproduces_every_part() {
    let source = std::fs::read(fixture_path(FIXTURE)).expect("the fixture is checked in");
    let out = remux("roundtrip").await;
    let written = std::fs::read(&out).expect("the round trip wrote a file");
    let _ = std::fs::remove_file(&out);

    assert_eq!(
        parts_of(&written, FIXTURE_BOUNDARY),
        parts_of(&source, FIXTURE_BOUNDARY),
        "every part survives demux and remux"
    );
    assert!(
        written.ends_with(format!("--{FIXTURE_BOUNDARY}--\r\n").as_bytes()),
        "the stream is closed on Eos"
    );
}

/// ffprobe reads the muxed file as an mpjpeg stream and counts the same parts.
/// Ignored by default: it needs ffprobe installed, which CI does not have.
///
/// ```text
/// cargo test -p g2g-plugins --features std --test m1080_multipart -- --ignored
/// ```
#[tokio::test]
#[ignore]
async fn ffprobe_counts_the_muxed_parts() {
    let source = std::fs::read(fixture_path(FIXTURE)).expect("the fixture is checked in");
    let out = remux("ffprobe").await;
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-f", "mpjpeg", "-count_frames"])
        .args(["-show_streams", "-of", "json"])
        .arg(&out)
        .output()
        .expect("ffprobe is installed");
    let _ = std::fs::remove_file(&out);
    let json = String::from_utf8_lossy(&probe.stdout);
    let frames = json
        .split("\"nb_read_frames\":")
        .nth(1)
        .and_then(|rest| rest.split('"').nth(1))
        .expect("ffprobe reported a frame count");
    assert_eq!(
        frames.parse::<usize>().expect("a frame count"),
        part_count(&source, FIXTURE_BOUNDARY)
    );
}

/// GStreamer's own `multipartdemux` reads what `multipartmux` wrote, one buffer
/// per part. Ignored by default: it needs `gst-launch-1.0` installed, which CI
/// does not have.
#[tokio::test]
#[ignore]
async fn gstreamer_reads_the_muxed_parts() {
    let source = std::fs::read(fixture_path(FIXTURE)).expect("the fixture is checked in");
    let out = remux("gst").await;
    let run = Command::new("gst-launch-1.0")
        .arg("-v")
        .args(["filesrc", &format!("location={}", out.display())])
        .args(["!", "multipartdemux", "!", "fakesink", "silent=false"])
        .output()
        .expect("gst-launch-1.0 is installed");
    let _ = std::fs::remove_file(&out);
    assert!(run.status.success(), "gst-launch-1.0 read the file");
    // `fakesink silent=false` posts one `chain` last-message per buffer, which
    // `-v` prints.
    let buffers = String::from_utf8_lossy(&run.stdout)
        .lines()
        .filter(|line| line.contains("chain"))
        .count();
    assert_eq!(buffers, part_count(&source, FIXTURE_BOUNDARY));
}

/// The whole MJPEG-over-HTTP leg: a local server hands the stream out and
/// `httpsrc` feeds the demuxer. Ignored by default: it needs `python3` and a
/// listening socket.
///
/// ```text
/// cargo test -p g2g-plugins --features std,http-src --test m1080_multipart -- --ignored
/// ```
#[cfg(feature = "http-src")]
#[tokio::test]
#[ignore]
async fn an_http_server_feeds_the_demuxer() {
    /// Where the throwaway server listens.
    const PORT: u16 = 18080;

    let bytes = std::fs::read(fixture_path(FIXTURE)).expect("the fixture is checked in");
    let served = std::env::temp_dir().join(format!("g2g-m1080-http-{}", std::process::id()));
    std::fs::create_dir_all(&served).expect("a directory to serve");
    std::fs::write(served.join(FIXTURE), &bytes).expect("the served copy");

    let mut server = Command::new("python3")
        .args(["-m", "http.server", &PORT.to_string()])
        .args(["--bind", "127.0.0.1"])
        .current_dir(&served)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("python3 is installed");
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let reg = default_registry();
    let line = format!(
        "httpsrc location=http://127.0.0.1:{PORT}/{FIXTURE} bytestream-format=multipart ! multipartdemux ! fakesink"
    );
    let graph = parse_launch(&reg, &line).expect("the line parses and negotiates");
    let stats = run_graph(graph, &ZeroClock, 4).await;
    let _ = server.kill();
    let _ = std::fs::remove_dir_all(&served);

    let stats = stats.expect("the pipeline runs");
    assert_eq!(
        stats.frames_consumed as usize,
        part_count(&bytes, FIXTURE_BOUNDARY),
        "every part arrived over HTTP"
    );
}

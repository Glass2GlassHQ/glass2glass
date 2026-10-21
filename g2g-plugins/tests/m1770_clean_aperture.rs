#![cfg(all(
    feature = "std",
    feature = "metadata",
    target_os = "linux",
    feature = "ffmpeg"
))]

use std::path::PathBuf;

use g2g_core::graph::Graph;
use g2g_core::runtime::{
    parse_launch, run_graph_observed, GraphNode, GraphNodeRef, Observer, TelemetrySnapshot,
};
use g2g_core::{DotAnnotations, PipelineClock, RawVideoFormat};
use g2g_plugins::registry::default_registry;

// tools/clean-aperture-fixture.sh writes each fixture and ffmpeg's default decode of it
struct Fixture {
    file: &'static str,
    reference: &'static str,
}

const H264: Fixture = Fixture {
    file: concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/h264_clean_aperture.mov"
    ),
    reference: concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/h264_clean_aperture.mov.yuv420p"
    ),
};
const AV1: Fixture = Fixture {
    file: concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/av1_clean_aperture.mp4"
    ),
    reference: concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/av1_clean_aperture.mp4.yuv420p"
    ),
};
const MPEG4: Fixture = Fixture {
    file: concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mpeg4_clean_aperture.mp4"
    ),
    reference: concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mpeg4_clean_aperture.mp4.yuv420p"
    ),
};
// the fixture script's 45x33 aperture, which ffmpeg rounds down to whole 4:2:0 chroma samples
const CROPPED_WIDTH: usize = 44;
const CROPPED_HEIGHT: usize = 32;
const FIXTURE_FRAMES: usize = 3;
const I420_FRAME_BYTES: usize = CROPPED_WIDTH * CROPPED_HEIGHT * 3 / 2;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn reference(fixture: &Fixture) -> Vec<u8> {
    let reference = std::fs::read(fixture.reference).expect("the reference decode is checked in");
    assert_eq!(
        reference.len(),
        FIXTURE_FRAMES * I420_FRAME_BYTES,
        "ffmpeg decodes every frame at the cropped size"
    );
    reference
}

fn output_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m1770-{tag}-{}.yuv", std::process::id()))
}

// the bytes the graph writes to `out`, with the caps every link carried
async fn run(
    tag: &str,
    graph: impl FnOnce(&str) -> Graph<GraphNode>,
) -> (Vec<u8>, TelemetrySnapshot) {
    let out = output_path(tag);
    let _ = std::fs::remove_file(&out);
    let graph = graph(&out.display().to_string());
    let observer = Observer::new();
    run_graph_observed(graph, &ZeroClock, 4, &observer, None)
        .await
        .unwrap_or_else(|e| panic!("runs the {tag} graph: {e:?}"));
    let decoded = std::fs::read(&out).expect("decoded frames written");
    let _ = std::fs::remove_file(&out);
    (decoded, observer.snapshot())
}

// the bytes the line writes to `filesink location={out}`, with the caps every link carried
async fn run_line(tag: &str, line: impl Fn(&str) -> String) -> (Vec<u8>, TelemetrySnapshot) {
    run(tag, |out| {
        let line = line(out);
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"))
    })
    .await
}

fn assert_carried_cropped_caps(snapshot: &TelemetrySnapshot) {
    let carried: Vec<&str> = snapshot
        .edges
        .iter()
        .filter_map(|edge| edge.observed_caps.as_deref())
        .collect();
    assert!(
        carried.iter().any(|caps| {
            caps.contains("video/x-raw")
                && caps.contains(&format!("width={CROPPED_WIDTH}"))
                && caps.contains(&format!("height={CROPPED_HEIGHT}"))
        }),
        "a link carried {CROPPED_WIDTH}x{CROPPED_HEIGHT} raw video: {carried:?}"
    );
}

fn assert_cropped_as_ffmpeg_does(fixture: &Fixture, decoded: &[u8], snapshot: &TelemetrySnapshot) {
    let reference = reference(fixture);
    assert_carried_cropped_caps(snapshot);
    assert_eq!(decoded.len(), reference.len(), "frame count and size");
    assert!(decoded == reference, "the cropped pixels are ffmpeg's");
}

fn decodebin_line(fixture: &Fixture) -> impl Fn(&str) -> String + '_ {
    move |out| {
        format!(
            "filesrc location={} ! decodebin ! videoconvert ! video/x-raw,format=I420 \
             ! filesink location={out}",
            fixture.file
        )
    }
}

#[tokio::test]
async fn decodebin_crops_to_the_clean_aperture_as_ffmpeg_does() {
    let (decoded, snapshot) = run_line("decodebin", decodebin_line(&H264)).await;
    assert_cropped_as_ffmpeg_does(&H264, &decoded, &snapshot);
}

#[tokio::test]
async fn a_bare_videocrop_after_the_decoder_crops_to_the_clean_aperture() {
    let (decoded, snapshot) = run_line("videocrop", |out| {
        format!(
            "filesrc location={} ! qtdemux ! avdec_h264 ! videocrop ! videoconvert \
             ! video/x-raw,format=I420 ! filesink location={out}",
            H264.file
        )
    })
    .await;
    assert_cropped_as_ffmpeg_does(&H264, &decoded, &snapshot);
}

#[tokio::test]
async fn mp4src_frames_carry_the_clean_aperture_to_a_bare_videocrop() {
    let registry = default_registry();
    let element = |name: &str| {
        registry
            .make_element(name)
            .unwrap_or_else(|| panic!("{name} is registered"))
    };
    let (decoded, snapshot) = run("mp4src", |out| {
        let mut graph: Graph<GraphNode> = Graph::new();
        let source = graph.add_source(GraphNodeRef::source(g2g_plugins::mp4src::Mp4Src::new(
            H264.file,
        )));
        let decoder = graph.add_transform(GraphNodeRef::Element(element("avdec_h264")));
        let crop = graph.add_transform(GraphNodeRef::Element(element("videocrop")));
        let convert = graph.add_transform(GraphNodeRef::element(
            g2g_plugins::videoconvert::VideoConvert::new(RawVideoFormat::I420),
        ));
        let sink = graph.add_sink(GraphNodeRef::element(g2g_plugins::filesink::FileSink::new(
            out,
        )));
        for (from, to) in [
            (source, decoder),
            (decoder, crop),
            (crop, convert),
            (convert, sink),
        ] {
            graph.link(from, to).expect("links the chain");
        }
        graph
    })
    .await;
    assert_cropped_as_ffmpeg_does(&H264, &decoded, &snapshot);
}

#[tokio::test]
async fn a_bare_uridecodebin_crops_to_the_clean_aperture() {
    let (decoded, snapshot) = run_line("uridecodebin", |out| {
        format!(
            "uridecodebin uri=file://{} ! videoconvert ! video/x-raw,format=I420 \
             ! filesink location={out}",
            H264.file
        )
    })
    .await;
    assert_cropped_as_ffmpeg_does(&H264, &decoded, &snapshot);
}

#[tokio::test]
async fn the_uridecodebin_fan_out_crops_its_video_branch() {
    let (decoded, snapshot) = run_line("uridecodebin-fanout", |out| {
        format!(
            "uridecodebin uri=file://{} name=d  d.video_0 ! videoconvert \
             ! video/x-raw,format=I420 ! filesink location={out}",
            H264.file
        )
    })
    .await;
    assert_cropped_as_ffmpeg_does(&H264, &decoded, &snapshot);
}

#[tokio::test]
async fn the_decodebin_fan_out_crops_its_video_branch() {
    let (decoded, snapshot) = run_line("decodebin-fanout", |out| {
        format!(
            "filesrc location={} ! decodebin name=d  d.video_0 ! videoconvert \
             ! video/x-raw,format=I420 ! filesink location={out}",
            H264.file
        )
    })
    .await;
    assert_cropped_as_ffmpeg_does(&H264, &decoded, &snapshot);
}

// playbin's video sink is whatever display sink the machine has, so the graph is read instead of run
#[test]
fn playbin_crops_its_video_branch() {
    let line = format!("playbin uri=file://{}", H264.file);
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let dot = graph.to_dot(
        "pipeline",
        |node| graph.element(node).map(|e| e.log_category().to_string()),
        &DotAnnotations::default(),
    );
    assert!(
        dot.contains("label=\"VideoCrop\""),
        "the video branch has a videocrop: {dot}"
    );
}

#[tokio::test]
async fn decodebin_crops_av1_in_mp4_to_the_clean_aperture() {
    let (decoded, snapshot) = run_line("decodebin-av1", decodebin_line(&AV1)).await;
    assert_cropped_as_ffmpeg_does(&AV1, &decoded, &snapshot);
}

#[tokio::test]
async fn decodebin_crops_mpeg4_part2_in_mp4_to_the_clean_aperture() {
    let (decoded, snapshot) = run_line("decodebin-mpeg4", decodebin_line(&MPEG4)).await;
    assert_cropped_as_ffmpeg_does(&MPEG4, &decoded, &snapshot);
}

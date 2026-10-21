#![cfg(all(
    feature = "std",
    feature = "metadata",
    target_os = "linux",
    feature = "ffmpeg"
))]

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph_observed, Observer, TelemetrySnapshot};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

// tools/clean-aperture-fixture.sh writes the fixture and ffmpeg's default decode of it
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_clean_aperture.mov"
);
const REFERENCE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_clean_aperture.mov.yuv420p"
);
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

fn reference() -> Vec<u8> {
    let reference = std::fs::read(REFERENCE).expect("the reference decode is checked in");
    assert_eq!(
        reference.len(),
        FIXTURE_FRAMES * I420_FRAME_BYTES,
        "ffmpeg decodes every frame at the cropped size"
    );
    reference
}

// the bytes the line writes to `filesink location={out}`, with the caps every link carried
async fn run_line(tag: &str, line: impl Fn(&str) -> String) -> (Vec<u8>, TelemetrySnapshot) {
    let out: PathBuf =
        std::env::temp_dir().join(format!("g2g-m1770-{tag}-{}.yuv", std::process::id()));
    let _ = std::fs::remove_file(&out);
    let line = line(&out.display().to_string());
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let observer = Observer::new();
    run_graph_observed(graph, &ZeroClock, 4, &observer, None)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let decoded = std::fs::read(&out).expect("decoded frames written");
    let _ = std::fs::remove_file(&out);
    (decoded, observer.snapshot())
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

#[tokio::test]
async fn decodebin_crops_to_the_clean_aperture_as_ffmpeg_does() {
    let reference = reference();
    let (decoded, snapshot) = run_line("decodebin", |out| {
        format!(
            "filesrc location={FIXTURE} ! decodebin ! videoconvert ! video/x-raw,format=I420 \
             ! filesink location={out}"
        )
    })
    .await;
    assert_carried_cropped_caps(&snapshot);
    assert_eq!(decoded.len(), reference.len(), "frame count and size");
    assert!(decoded == reference, "the cropped pixels are ffmpeg's");
}

#[tokio::test]
async fn a_bare_videocrop_after_the_decoder_crops_to_the_clean_aperture() {
    let reference = reference();
    let (decoded, snapshot) = run_line("videocrop", |out| {
        format!(
            "filesrc location={FIXTURE} ! qtdemux ! avdec_h264 ! videocrop ! videoconvert \
             ! video/x-raw,format=I420 ! filesink location={out}"
        )
    })
    .await;
    assert_carried_cropped_caps(&snapshot);
    assert_eq!(decoded.len(), reference.len(), "frame count and size");
    assert!(decoded == reference, "the cropped pixels are ffmpeg's");
}

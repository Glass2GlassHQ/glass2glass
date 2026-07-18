//! M171: subtitle text overlay. The `subparse` SRT/WebVTT parsers, the
//! `TextOverlay` element rendering cues by PTS, and the registry / `gst-launch`
//! wiring (`textoverlay`, `location=`).

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;
use g2g_plugins::subparse::{parse_webvtt, Cue, CueSettings, TextAlign};
use g2g_plugins::textoverlay::TextOverlay;

/// A clock pinned at zero (the parsed pipelines don't depend on wall time).
struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("g2g_m171_{}_{}", std::process::id(), name))
}

#[test]
fn webvtt_round_trips_through_the_parser() {
    let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n<b>First</b> line\n\n00:00:03.000 --> 00:00:05.000 align:start\nSecond\n";
    let cues = parse_webvtt(vtt);
    assert_eq!(
        cues,
        vec![
            Cue {
                start_ns: 1_000_000_000,
                end_ns: 3_000_000_000,
                text: "First line".into(),
                settings: CueSettings::default(),
            },
            Cue {
                start_ns: 3_000_000_000,
                end_ns: 5_000_000_000,
                text: "Second".into(),
                settings: CueSettings {
                    align: TextAlign::Start,
                    ..CueSettings::default()
                },
            },
        ]
    );
}

#[test]
fn registry_exposes_textoverlay_with_a_location_property() {
    let reg = default_registry();
    assert!(
        reg.element_names().contains(&"textoverlay"),
        "textoverlay registered"
    );
    let dump = reg.inspect("textoverlay").expect("inspectable");
    assert!(
        dump.contains("location"),
        "location property listed:\n{dump}"
    );
}

#[tokio::test]
async fn textoverlay_runs_in_a_parsed_pipeline() {
    // videotestsrc emits RGBA8, which textoverlay consumes directly; with no
    // cues it is an identity transform, so this proves the registry wiring and
    // negotiation, not rendering (that is unit-tested in the element).
    let reg = default_registry();
    let graph = parse_launch(&reg, "videotestsrc num-buffers=4 ! textoverlay ! fakesink")
        .expect("pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("pipeline runs");
    assert_eq!(
        stats.frames_consumed, 4,
        "all frames reached the sink through the overlay"
    );
}

#[test]
fn location_property_loads_an_srt_file() {
    let path = temp_path("subs.srt");
    std::fs::write(
        &path,
        "1\n00:00:01,000 --> 00:00:04,000\nHELLO WORLD\n\n2\n00:00:05,000 --> 00:00:06,000\nDONE\n",
    )
    .expect("write fixture");

    // Drive the element the way the launch parser does: default-construct, then
    // apply the `location=` property as a textual value.
    use g2g_core::{AsyncElement, PropValue};
    let mut ov = TextOverlay::new();
    assert_eq!(ov.cue_count(), 0);
    ov.set_property(
        "location",
        PropValue::Str(path.to_string_lossy().into_owned()),
    )
    .expect("location accepted");
    assert_eq!(ov.cue_count(), 2, "both cues loaded from the file");

    let _ = std::fs::remove_file(&path);
}

#![cfg(feature = "analytics")]

use g2g_core::runtime::{parse_launch, run_graph, Registry};
use g2g_core::{Frame, G2gError};
use g2g_plugins::appsink::{register_appsink_pull, AppSinkPull, Pull};
use g2g_plugins::clock::WallClock;
use pyo3::prelude::*;

const FIXTURE_MODULE: &str = "text_output_element";
const FRAME_COUNT: usize = 3;
const VIDEO_CAPS: &str = "video/x-raw,format=RGBA,width=64,height=48";
const LINK_CAPACITY: usize = 4;

fn fixtures_on_path() {
    static SET: std::sync::Once = std::sync::Once::new();
    SET.call_once(|| {
        std::env::set_var(
            "PYTHONPATH",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
        );
    });
}

fn registry() -> Registry {
    fixtures_on_path();
    let mut registry = g2g_plugins::registry::default_registry();
    g2g_python::register(&mut registry);
    registry
}

fn run(line: &str) -> Result<(), G2gError> {
    let graph = parse_launch(&registry(), line).expect("the launch line parses");
    let clock = WallClock::new();
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap()
        .block_on(run_graph(graph, &clock, LINK_CAPACITY))
        .map(|_| ())
}

#[derive(Debug, Default)]
struct Pulled {
    frames: Vec<Frame>,
    ended: bool,
}

fn drain(pull: &AppSinkPull) -> Pulled {
    let mut pulled = Pulled::default();
    loop {
        match pull.try_pull() {
            Pull::Frame(frame) => pulled.frames.push(frame),
            Pull::Ended => {
                pulled.ended = true;
                return pulled;
            }
            Pull::Empty => return pulled,
        }
    }
}

fn bytes(frame: &Frame) -> Vec<u8> {
    frame
        .domain
        .as_system_slice()
        .expect("a system memory frame")
        .to_vec()
}

fn video_without_pyelement(channel: &str) -> Vec<Frame> {
    let pull = register_appsink_pull(channel);
    run(&format!(
        "videotestsrc num-buffers={FRAME_COUNT} ! {VIDEO_CAPS} ! appsink channel={channel}"
    ))
    .expect("the reference run completes");
    drain(&pull).frames
}

fn assert_routed(test: &str, branches_text_first: bool) {
    let video_channel = format!("m1241-{test}-video");
    let text_channel = format!("m1241-{test}-text");
    let video_pull = register_appsink_pull(&video_channel);
    let text_pull = register_appsink_pull(&text_channel);
    let video_branch = format!("cap.src ! appsink channel={video_channel}");
    let text_branch = format!("cap.text_src ! appsink channel={text_channel}");
    let branches = if branches_text_first {
        format!("{text_branch} {video_branch}")
    } else {
        format!("{video_branch} {text_branch}")
    };
    run(&format!(
        "videotestsrc num-buffers={FRAME_COUNT} ! {VIDEO_CAPS} ! \
         pyelement module={FIXTURE_MODULE} class=Captioner name=cap {branches}"
    ))
    .expect("the run completes");

    let video = drain(&video_pull);
    let text = drain(&text_pull);
    let reference = video_without_pyelement(&format!("m1241-{test}-reference"));
    assert!(video.ended && text.ended, "both branches end");
    assert_eq!(video.frames.len(), FRAME_COUNT);
    for (frame, reference) in video.frames.iter().zip(&reference) {
        assert_eq!(
            bytes(frame),
            bytes(reference),
            "src carries the frame unchanged"
        );
    }
    let captions: Vec<(String, u64)> = text
        .frames
        .iter()
        .map(|frame| {
            (
                String::from_utf8(bytes(frame)).unwrap(),
                frame.timing.pts_ns,
            )
        })
        .collect();
    let expected: Vec<(String, u64)> = video
        .frames
        .iter()
        .enumerate()
        .map(|(n, frame)| (format!("caption {n}"), frame.timing.pts_ns))
        .collect();
    assert_eq!(captions, expected);
}

#[test]
fn text_src_carries_the_captions_and_src_the_video() {
    assert_routed("video-first", false);
}

#[test]
fn branches_route_by_pad_name_not_link_order() {
    assert_routed("text-first", true);
}

#[test]
fn text_src_feeds_a_textoverlay_text_pad() {
    let channel = "m1241-overlay";
    let pull = register_appsink_pull(channel);
    run(&format!(
        "videotestsrc num-buffers={FRAME_COUNT} ! {VIDEO_CAPS} ! tee name=t \
         t. ! queue ! pyelement module={FIXTURE_MODULE} class=Captioner name=cap \
         cap.src ! fakesink \
         cap.text_src ! queue ! overlay.text_sink \
         t. ! queue ! textoverlay name=overlay ! appsink channel={channel}"
    ))
    .expect("the run completes");

    let overlaid = drain(&pull);
    assert!(overlaid.ended);
    assert_eq!(overlaid.frames.len(), FRAME_COUNT);
}

#[test]
fn an_unknown_source_pad_fails_at_launch_naming_it() {
    let line = format!(
        "videotestsrc num-buffers={FRAME_COUNT} ! {VIDEO_CAPS} ! \
         pyelement module={FIXTURE_MODULE} class=Captioner name=cap \
         cap.src ! fakesink cap.subtitle_src ! fakesink"
    );
    let Err(error) = parse_launch(&registry(), &line) else {
        panic!("a pyelement with a subtitle_src pad launched");
    };
    assert!(error.to_string().contains("'subtitle_src'"), "{error}");
}

#[test]
fn emitting_to_text_src_without_the_pad_fails_naming_it() {
    let result = run(&format!(
        "videotestsrc num-buffers={FRAME_COUNT} ! {VIDEO_CAPS} ! \
         pyelement module={FIXTURE_MODULE} class=RecordingCaptioner ! fakesink"
    ));

    assert!(result.is_err());
    let errors: Vec<String> = Python::attach(|py| {
        PyModule::import(py, FIXTURE_MODULE)?
            .getattr("EMIT_ERRORS")?
            .extract()
    })
    .expect("the fixture records its emit errors");
    assert!(
        errors.iter().any(|error| error.contains("'text_src'")),
        "{errors:?}"
    );
}

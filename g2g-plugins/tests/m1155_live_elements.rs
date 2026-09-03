//! M1155: `togglerecord` starts and stops several streams on one decision.
//!
//! Two halves. A hand-driven pair (an H.264 main whose keyframes decide, a PCM
//! secondary at its own buffer rate) checks that the secondary forwards exactly
//! the span the main recorded and lands on the same eaten-gap timeline. Then a
//! `parse_launch` line checks the same element reached through text: the `record`
//! property gates a single stream, and `group=` joins two tee branches so the
//! secondary only passes what the main decided.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, ChannelLayout, Colorimetry, Dim, Frame, FrameTiming, G2gError,
    Interlace, OutputSink, PipelineClock, PipelinePacket, PropValue, PushOutcome, Rate,
    RawVideoFormat, VideoCodec,
};
use g2g_plugins::registry::default_registry;
use g2g_plugins::togglerecord::{RecordGroup, ToggleRecord};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// 25 fps video, so a keyframe every 4th frame is one interval boundary every
/// 160 ms.
const VIDEO_FRAME_NS: u64 = 40_000_000;
const VIDEO_KEYFRAME_EVERY: u64 = 4;
/// 1024 samples at 48 kHz, rounded to a flat 20 ms so the expected timestamps
/// stay readable.
const AUDIO_FRAME_NS: u64 = 20_000_000;
const VIDEO_FRAMES: u64 = 10;
const AUDIO_FRAMES: u64 = 20;

/// The keyframe the recording starts on, and the one it stops before.
const RECORD_START_NS: u64 = 4 * VIDEO_FRAME_NS;
const RECORD_STOP_NS: u64 = 8 * VIDEO_FRAME_NS;

fn h264() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(25 << 16),
        colorimetry: Colorimetry::UNKNOWN,
    }
}

fn pcm() -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 2,
        sample_rate: 48_000,
        channel_layout: ChannelLayout::UNSPECIFIED,
    }
}

fn rgba() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(25 << 16),
        interlace: Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

fn frame(pts_ns: u64, duration_ns: u64, keyframe: bool) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 32]))),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns,
            keyframe,
            ..FrameTiming::default()
        },
        pts_ns / duration_ns,
    ))
}

/// Records the timestamp of every frame that got through.
#[derive(Default)]
struct Collect {
    pts: Vec<u64>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        if let PipelinePacket::DataFrame(frame) = packet {
            self.pts.push(frame.timing.pts_ns);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Drives the whole video stream through `main`, asking for a recording from
/// frame 1 and stopping it from frame 5, then the whole audio stream through
/// `secondary`. Returns what each forwarded.
async fn run_pair(is_live: bool) -> (Vec<u64>, Vec<u64>) {
    let group = RecordGroup::new();
    let mut main = ToggleRecord::main(group.clone()).with_is_live(is_live);
    let mut secondary = ToggleRecord::secondary(group.clone()).with_is_live(is_live);
    main.configure_pipeline(&h264()).expect("H.264 passes");
    secondary.configure_pipeline(&pcm()).expect("PCM passes");

    let mut video_out = Collect::default();
    let mut audio_out = Collect::default();
    for index in 0..VIDEO_FRAMES {
        if index == 1 {
            group.set_record(true);
        }
        if index == 5 {
            group.set_record(false);
        }
        let keyframe = index % VIDEO_KEYFRAME_EVERY == 0;
        main.process(
            frame(index * VIDEO_FRAME_NS, VIDEO_FRAME_NS, keyframe),
            &mut video_out,
        )
        .await
        .expect("the main stream runs");
    }
    for index in 0..AUDIO_FRAMES {
        secondary
            .process(
                frame(index * AUDIO_FRAME_NS, AUDIO_FRAME_NS, false),
                &mut audio_out,
            )
            .await
            .expect("the secondary runs");
    }
    (video_out.pts, audio_out.pts)
}

#[tokio::test]
async fn the_secondary_forwards_the_span_the_main_stream_decided() {
    let (video, audio) = run_pair(false).await;

    // The main asked to record at frame 1 but only started at the keyframe at
    // frame 4, and asked to stop at frame 5 but only stopped before the keyframe
    // at frame 8: frames 4..7, pulled back to a timeline starting at zero.
    let expected_video: Vec<u64> = (0..4).map(|i| i * VIDEO_FRAME_NS).collect();
    assert_eq!(video, expected_video, "the recording is whole GOPs");

    // Every audio buffer whose timestamp lies in [160 ms, 320 ms), on the same
    // eaten-gap timeline.
    let expected_audio: Vec<u64> = (0..AUDIO_FRAMES)
        .map(|i| i * AUDIO_FRAME_NS)
        .filter(|pts| (RECORD_START_NS..RECORD_STOP_NS).contains(pts))
        .map(|pts| pts - RECORD_START_NS)
        .collect();
    assert_eq!(
        audio, expected_audio,
        "the secondary cut on the video's keyframe decision"
    );
    assert_eq!(
        audio.first(),
        video.first(),
        "both streams open the recording at the same output time"
    );
}

#[tokio::test]
async fn is_live_leaves_both_streams_on_their_input_timeline() {
    let (video, audio) = run_pair(true).await;

    let expected_video: Vec<u64> = (4..8).map(|i| i * VIDEO_FRAME_NS).collect();
    assert_eq!(video, expected_video, "no gap eating on the main stream");
    let expected_audio: Vec<u64> = (0..AUDIO_FRAMES)
        .map(|i| i * AUDIO_FRAME_NS)
        .filter(|pts| (RECORD_START_NS..RECORD_STOP_NS).contains(pts))
        .collect();
    assert_eq!(audio, expected_audio, "nor on the secondary");
}

#[tokio::test]
async fn a_launch_line_gates_a_single_stream_on_record() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=4 ! togglerecord record=true ! fakesink",
    )
    .expect("the togglerecord line parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("runs");
    assert_eq!(
        stats.frames_consumed, 4,
        "raw video is all cut points, so recording starts at the first frame"
    );

    let reg = default_registry();
    let graph =
        parse_launch(&reg, "videotestsrc num-buffers=4 ! togglerecord ! fakesink").expect("parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("runs");
    assert_eq!(
        stats.frames_consumed, 0,
        "`record` defaults to false, so nothing is written"
    );
}

#[tokio::test]
async fn a_named_group_joins_two_branches_of_a_launch_line() {
    // Held for the whole test: the group is what a `togglerecord group=` element
    // looks up, and dropping it would let the second run build a fresh one.
    let group = RecordGroup::named("m1155-branches");
    group.set_record(true);

    let reg = default_registry();
    let line = "videotestsrc num-buffers=4 ! tee name=t \
                ! queue ! togglerecord group=m1155-branches ! fakesink \
                t. ! queue ! togglerecord group=m1155-branches main=false ! fakesink";
    let graph = parse_launch(&reg, line).expect("the two-branch group line parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("runs");
    assert_eq!(
        stats.frames_consumed, 8,
        "both branches recorded all four frames"
    );
}

#[test]
fn a_launch_line_reads_the_record_flag_back_through_the_group() {
    let reg = default_registry();
    let mut element = reg
        .make_element("togglerecord")
        .expect("registered under the gst name");
    element
        .set_property("group", PropValue::Str("m1155-readback".into()))
        .expect("group is settable");
    let group = RecordGroup::named("m1155-readback");
    group.set_record(true);
    assert_eq!(
        element.get_property("record"),
        Some(PropValue::Bool(true)),
        "the group's flag is what the element reports"
    );
    assert_eq!(
        element.get_property("recording"),
        Some(PropValue::Bool(false)),
        "asking to record is not yet recording"
    );
}

#[test]
fn togglerecord_passes_through_any_caps() {
    // A group member constrains neither side: it decides when a frame passes,
    // never what shape it has.
    let element = ToggleRecord::new();
    for caps in [h264(), pcm(), rgba()] {
        assert_eq!(
            element.intercept_caps(&caps).expect("any caps pass"),
            caps.clone()
        );
    }
}

#[tokio::test]
async fn a_parked_secondary_wakes_when_the_main_stream_advances() {
    let group = RecordGroup::new();
    let mut main = ToggleRecord::main(group.clone());
    let mut secondary = ToggleRecord::secondary(group.clone());
    main.configure_pipeline(&rgba()).expect("raw video passes");
    secondary.configure_pipeline(&pcm()).expect("PCM passes");
    group.set_record(true);

    let mut main_out = Collect::default();
    let mut secondary_out = Collect::default();
    // `join!` polls in order, so the secondary reaches its wait first and the
    // main stream only produces after a yield. A lost wakeup hangs here, so the
    // whole pair is bounded.
    let paired = async {
        let advance = async {
            tokio::task::yield_now().await;
            assert!(
                !group.recording(),
                "the secondary is waiting on a decision the main stream has not made"
            );
            main.process(frame(0, VIDEO_FRAME_NS, true), &mut main_out)
                .await
        };
        let follow = secondary.process(frame(0, AUDIO_FRAME_NS, false), &mut secondary_out);
        let (followed, advanced) = tokio::join!(follow, advance);
        followed.expect("the secondary runs");
        advanced.expect("the main stream runs");
    };
    tokio::time::timeout(core::time::Duration::from_secs(5), paired)
        .await
        .expect("the main stream's advance woke the parked secondary");

    assert_eq!(main_out.pts, vec![0]);
    assert_eq!(
        secondary_out.pts,
        vec![0],
        "the secondary forwarded once the decision existed"
    );
}

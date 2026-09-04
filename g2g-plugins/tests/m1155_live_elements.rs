//! M1155: the two live-stream elements, `togglerecord` and `livesync`.
//!
//! `togglerecord` starts and stops several streams on one decision. A
//! hand-driven pair (an H.264 main whose keyframes decide, a PCM secondary at its
//! own buffer rate) checks that the secondary forwards exactly the span the main
//! recorded and lands on the same eaten-gap timeline. Then a `parse_launch` line
//! checks the same element reached through text: the `record` property gates a
//! single stream, and `group=` joins two tee branches so the secondary only
//! passes what the main decided.
//!
//! `livesync` keeps a stalling live input's output going. Hand-driven against a
//! caller-supplied clock, it fills a video gap with the last frame repeated on
//! cadence and an audio gap with silence the size of the last buffer, drops a
//! frame behind the timeline it already emitted, and follows one that is further
//! behind than `late-threshold`. A `parse_launch` line checks that a single
//! inbound link builds it as a fan-in and that the frames reach the sink.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, ChannelLayout, Colorimetry, Dim, Frame, FrameTiming, G2gError,
    Interlace, MultiInputElement, OutputSink, PipelineClock, PipelinePacket, PropValue,
    PushOutcome, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::livesync::LiveSync;
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

/// Records the timestamp, span and bytes of every frame that got through.
#[derive(Default)]
struct Collect {
    pts: Vec<u64>,
    durations: Vec<u64>,
    payloads: Vec<Vec<u8>>,
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
            self.durations.push(frame.timing.duration_ns);
            self.payloads.push(
                frame
                    .domain
                    .as_system_slice()
                    .expect("the tests push system memory")
                    .to_vec(),
            );
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

/// Nanoseconds in a second, for turning a sample rate into a buffer size.
const NS_PER_SECOND: u64 = 1_000_000_000;
/// Bytes in one hand-driven `livesync` video buffer. Each frame is filled with
/// its own index, so a repeat is visible in the bytes.
const LIVESYNC_VIDEO_BYTES: usize = 32;
/// Frames fed on cadence before the stall.
const LIVESYNC_LEAD_FRAMES: u64 = 2;
/// Output slots the stall covers, and so filler frames expected.
const LIVESYNC_GAP_FRAMES: u64 = 3;
/// Two frame periods behind the output timeline before `livesync` follows the
/// input instead of dropping it.
const LIVESYNC_LATE_THRESHOLD_NS: u64 = 2 * VIDEO_FRAME_NS;
/// Frames the late-arrival test feeds on cadence first: enough that a frame a
/// threshold and more behind still lands on a non-negative timestamp.
const LIVESYNC_ONCADENCE_FRAMES: u64 = 4;
/// Fill byte of the frames that arrive behind the timeline, distinct from every
/// on-cadence frame's index.
const LIVESYNC_LATE_FILL: u8 = 0xAA;
/// Buffers the `parse_launch` line asks `videotestsrc` for.
const LIVESYNC_LAUNCH_BUFFERS: u64 = 4;
/// Bytes one `PcmS16Le` sample takes, the format `pcm()` names.
const PCM_S16LE_SAMPLE_BYTES: usize = 2;

fn livesync_video_frame(pts_ns: u64, fill: u8) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(
            vec![fill; LIVESYNC_VIDEO_BYTES].into_boxed_slice(),
        )),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: VIDEO_FRAME_NS,
            ..FrameTiming::default()
        },
        pts_ns / VIDEO_FRAME_NS,
    ))
}

/// Bytes one `AUDIO_FRAME_NS` buffer of `pcm()` occupies.
fn audio_buffer_bytes() -> usize {
    let Caps::Audio {
        format,
        channels,
        sample_rate,
        ..
    } = pcm()
    else {
        panic!("pcm() is audio caps");
    };
    let sample_bytes = match format {
        AudioFormat::PcmS16Le => PCM_S16LE_SAMPLE_BYTES,
        other => panic!("pcm() is 16-bit, got {other:?}"),
    };
    (sample_rate as u64 * AUDIO_FRAME_NS / NS_PER_SECOND) as usize
        * channels as usize
        * sample_bytes
}

fn livesync_audio_buffer(pts_ns: u64, bytes: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: AUDIO_FRAME_NS,
            ..FrameTiming::default()
        },
        pts_ns / AUDIO_FRAME_NS,
    ))
}

#[tokio::test]
async fn livesync_fills_a_video_stall_with_the_last_frame() {
    let mut sync = LiveSync::new();
    sync.configure_pipeline(0, &rgba())
        .expect("raw video passes");
    assert_eq!(
        MultiInputElement::tick_interval_ns(&sync),
        Some(VIDEO_FRAME_NS),
        "the tick period is one frame period of the negotiated caps"
    );

    // Wall time tracks the timeline: every frame arrives exactly when it is due.
    let mut out = Collect::default();
    for index in 0..LIVESYNC_LEAD_FRAMES {
        let pts_ns = index * VIDEO_FRAME_NS;
        sync.handle(
            0,
            livesync_video_frame(pts_ns, index as u8),
            pts_ns,
            &mut out,
        )
        .await
        .expect("the lead frames pass");
    }

    // Nothing arrives for the whole gap. One tick at the last missed slot fills
    // every one of them.
    let resume_pts_ns = (LIVESYNC_LEAD_FRAMES + LIVESYNC_GAP_FRAMES) * VIDEO_FRAME_NS;
    sync.handle(
        0,
        PipelinePacket::Tick,
        resume_pts_ns - VIDEO_FRAME_NS,
        &mut out,
    )
    .await
    .expect("the tick fills the gap");
    sync.handle(
        0,
        livesync_video_frame(resume_pts_ns, LIVESYNC_LEAD_FRAMES as u8),
        resume_pts_ns,
        &mut out,
    )
    .await
    .expect("the input resumes");

    let emitted = LIVESYNC_LEAD_FRAMES + LIVESYNC_GAP_FRAMES + 1;
    let expected_pts: Vec<u64> = (0..emitted).map(|slot| slot * VIDEO_FRAME_NS).collect();
    assert_eq!(out.pts, expected_pts, "the output timeline has no hole");
    assert_eq!(out.durations, vec![VIDEO_FRAME_NS; emitted as usize]);

    let last_real = out.payloads[LIVESYNC_LEAD_FRAMES as usize - 1].clone();
    for filler in 0..LIVESYNC_GAP_FRAMES as usize {
        assert_eq!(
            out.payloads[LIVESYNC_LEAD_FRAMES as usize + filler],
            last_real,
            "the filler is the last real frame's bytes"
        );
    }
    assert_ne!(
        out.payloads.last(),
        Some(&last_real),
        "and the resumed frame is not"
    );

    assert_eq!(
        sync.get_property("in"),
        Some(PropValue::Uint(LIVESYNC_LEAD_FRAMES + 1))
    );
    assert_eq!(sync.get_property("out"), Some(PropValue::Uint(emitted)));
    assert_eq!(
        sync.get_property("duplicate"),
        Some(PropValue::Uint(LIVESYNC_GAP_FRAMES))
    );
    assert_eq!(sync.get_property("drop"), Some(PropValue::Uint(0)));
}

#[tokio::test]
async fn livesync_fills_an_audio_stall_with_silence() {
    let mut sync = LiveSync::new();
    sync.configure_pipeline(0, &pcm()).expect("PCM passes");

    let mut out = Collect::default();
    let tone = vec![LIVESYNC_LATE_FILL; audio_buffer_bytes()];
    sync.handle(0, livesync_audio_buffer(0, &tone), 0, &mut out)
        .await
        .expect("the first buffer passes");
    sync.handle(0, PipelinePacket::Tick, AUDIO_FRAME_NS, &mut out)
        .await
        .expect("the tick fills the next slot");

    assert_eq!(out.pts, vec![0, AUDIO_FRAME_NS]);
    assert_eq!(out.durations, vec![AUDIO_FRAME_NS; 2]);
    assert_eq!(out.payloads[0], tone);
    assert_eq!(
        out.payloads[1],
        vec![0u8; audio_buffer_bytes()],
        "S16LE silence is all-zero, the size and span of the last buffer"
    );
    assert_eq!(sync.get_property("duplicate"), Some(PropValue::Uint(1)));
}

#[tokio::test]
async fn livesync_drops_a_late_frame_and_follows_a_much_later_one() {
    let mut sync = LiveSync::new().with_late_threshold_ns(LIVESYNC_LATE_THRESHOLD_NS);
    sync.configure_pipeline(0, &rgba())
        .expect("raw video passes");

    let mut out = Collect::default();
    for index in 0..LIVESYNC_ONCADENCE_FRAMES {
        let pts_ns = index * VIDEO_FRAME_NS;
        sync.handle(
            0,
            livesync_video_frame(pts_ns, index as u8),
            pts_ns,
            &mut out,
        )
        .await
        .expect("the lead frames pass");
    }

    // The next output is due at LIVESYNC_ONCADENCE_FRAMES periods. This one is exactly
    // the threshold behind it, so it is dropped rather than followed.
    let next_pts_ns = LIVESYNC_ONCADENCE_FRAMES * VIDEO_FRAME_NS;
    let late_pts_ns = next_pts_ns - LIVESYNC_LATE_THRESHOLD_NS;
    sync.handle(
        0,
        livesync_video_frame(late_pts_ns, LIVESYNC_LATE_FILL),
        next_pts_ns,
        &mut out,
    )
    .await
    .expect("the late frame is handled");
    let on_cadence: Vec<u64> = (0..LIVESYNC_ONCADENCE_FRAMES)
        .map(|slot| slot * VIDEO_FRAME_NS)
        .collect();
    assert_eq!(
        out.pts, on_cadence,
        "the late frame did not reach the output"
    );
    assert_eq!(sync.get_property("drop"), Some(PropValue::Uint(1)));

    // One period further behind clears the threshold, so the timeline follows it.
    let resync_pts_ns = late_pts_ns - VIDEO_FRAME_NS;
    sync.handle(
        0,
        livesync_video_frame(resync_pts_ns, LIVESYNC_LATE_FILL),
        next_pts_ns + VIDEO_FRAME_NS,
        &mut out,
    )
    .await
    .expect("the far-behind frame is handled");
    assert_eq!(
        out.pts.last(),
        Some(&resync_pts_ns),
        "the output timeline restarted on the far-behind frame"
    );
    assert_eq!(
        sync.get_property("drop"),
        Some(PropValue::Uint(1)),
        "following the input is not a drop"
    );
    assert_eq!(
        sync.get_property("in"),
        Some(PropValue::Uint(LIVESYNC_ONCADENCE_FRAMES + 2))
    );
}

#[tokio::test]
async fn a_launch_line_builds_livesync_as_a_one_input_fan_in() {
    let reg = default_registry();
    let line = format!("videotestsrc num-buffers={LIVESYNC_LAUNCH_BUFFERS} ! livesync ! fakesink");
    let graph = parse_launch(&reg, &line).expect("the one-input livesync line parses");
    // A real clock, so the fan-in arm gets the deadline tick the element asked
    // for; the source outruns it, so no filler is due.
    let stats = run_graph(graph, &WallClock::new(), 4)
        .await
        .expect("runs to EOS");
    assert!(
        stats.frames_consumed >= LIVESYNC_LAUNCH_BUFFERS,
        "the sink saw every source frame, got {}",
        stats.frames_consumed
    );
}

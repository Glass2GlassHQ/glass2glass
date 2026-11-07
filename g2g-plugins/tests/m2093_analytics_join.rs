#![cfg(feature = "metadata")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AnalyticsMeta, BBox, Caps, Colorimetry, Dim, Interlace, MultiInputElement, ObjectDetection,
    Rate, RawVideoFormat,
};
use g2g_plugins::analyticsjoin::AnalyticsJoin;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const FRAME_PERIOD_NS: u64 = 33_333_333;
const VIDEO_FRAMES: u64 = 9;
const DETECTION_STRIDE: u64 = 3;

fn rgba() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30 << 16),
        interlace: Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

fn frame_at(pts_ns: u64, duration_ns: u64) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 4].into_boxed_slice())),
        FrameTiming {
            pts_ns,
            duration_ns,
            ..FrameTiming::default()
        },
        0,
    )
}

fn video_frame(index: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(frame_at(index * FRAME_PERIOD_NS, FRAME_PERIOD_NS))
}

// the box's x position identifies which detection frame it came from
fn detections_frame(pts_ns: u64, x: f32) -> PipelinePacket {
    let mut analytics = AnalyticsMeta::new();
    analytics.add_detection(ObjectDetection {
        bbox: BBox {
            x,
            y: 0.5,
            w: 0.1,
            h: 0.1,
        },
        label: 32,
        confidence: 0.9,
    });
    let mut frame = frame_at(pts_ns, 0);
    frame.meta.attach(analytics);
    PipelinePacket::DataFrame(frame)
}

fn configured_join() -> AnalyticsJoin {
    let mut join = AnalyticsJoin::new();
    join.configure_pipeline(AnalyticsJoin::VIDEO, &rgba())
        .expect("the video pad takes raw video");
    join.configure_pipeline(AnalyticsJoin::ANALYTICS, &rgba())
        .expect("the analytics pad takes any caps");
    join
}

async fn push(
    join: &mut AnalyticsJoin,
    input: usize,
    packet: PipelinePacket,
    sink: &mut CollectSink,
) {
    join.process(input, packet, sink).await.expect("process");
}

fn data_frames(sink: &CollectSink) -> Vec<&Frame> {
    sink.packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .collect()
}

fn detection_x(frame: &Frame) -> Option<f32> {
    let analytics = frame.meta.get::<AnalyticsMeta>()?;
    analytics
        .detections()
        .next()
        .map(|detection| detection.bbox.x)
}

fn detection_x_for_index(index: u64) -> f32 {
    index as f32 / VIDEO_FRAMES as f32
}

#[tokio::test]
async fn detections_land_on_the_video_frames_at_their_time() {
    let mut join = configured_join();
    let mut sink = CollectSink::default();
    // video runs ahead of the detector branch, as it does behind inference
    for index in 0..VIDEO_FRAMES {
        push(
            &mut join,
            AnalyticsJoin::VIDEO,
            video_frame(index),
            &mut sink,
        )
        .await;
    }
    assert!(
        data_frames(&sink).is_empty(),
        "video waits until the detector branch reaches its time"
    );
    for index in (0..VIDEO_FRAMES).step_by(DETECTION_STRIDE as usize) {
        let pts_ns = index * FRAME_PERIOD_NS;
        push(
            &mut join,
            AnalyticsJoin::ANALYTICS,
            detections_frame(pts_ns, detection_x_for_index(index)),
            &mut sink,
        )
        .await;
    }
    push(
        &mut join,
        AnalyticsJoin::ANALYTICS,
        PipelinePacket::Eos,
        &mut sink,
    )
    .await;

    let frames = data_frames(&sink);
    assert_eq!(frames.len() as u64, VIDEO_FRAMES);
    for (index, frame) in frames.iter().enumerate() {
        let index = index as u64;
        assert_eq!(
            frame.timing.pts_ns,
            index * FRAME_PERIOD_NS,
            "video keeps its order"
        );
        let expected = index
            .is_multiple_of(DETECTION_STRIDE)
            .then(|| detection_x_for_index(index));
        assert_eq!(detection_x(frame), expected, "frame {index}");
    }
}

#[tokio::test]
async fn a_restamped_detection_matches_within_half_a_frame() {
    let mut join = configured_join();
    let mut sink = CollectSink::default();
    let inside_ns = FRAME_PERIOD_NS / 2 - 1;
    let outside_ns = FRAME_PERIOD_NS / 2 + 1;
    for index in 0..4 {
        push(
            &mut join,
            AnalyticsJoin::VIDEO,
            video_frame(index),
            &mut sink,
        )
        .await;
    }
    push(
        &mut join,
        AnalyticsJoin::ANALYTICS,
        detections_frame(FRAME_PERIOD_NS + inside_ns, 0.5),
        &mut sink,
    )
    .await;
    push(
        &mut join,
        AnalyticsJoin::ANALYTICS,
        detections_frame(3 * FRAME_PERIOD_NS + outside_ns, 0.75),
        &mut sink,
    )
    .await;
    push(
        &mut join,
        AnalyticsJoin::ANALYTICS,
        PipelinePacket::Eos,
        &mut sink,
    )
    .await;

    let found: Vec<Option<f32>> = data_frames(&sink).into_iter().map(detection_x).collect();
    assert_eq!(found, [None, Some(0.5), None, None]);
}

#[tokio::test]
async fn an_ended_detector_branch_releases_the_held_video_in_order() {
    let mut join = configured_join();
    let mut sink = CollectSink::default();
    push(&mut join, AnalyticsJoin::VIDEO, video_frame(0), &mut sink).await;
    push(
        &mut join,
        AnalyticsJoin::VIDEO,
        PipelinePacket::CapsChanged(rgba()),
        &mut sink,
    )
    .await;
    push(&mut join, AnalyticsJoin::VIDEO, video_frame(1), &mut sink).await;
    assert!(sink.packets.is_empty());
    push(
        &mut join,
        AnalyticsJoin::ANALYTICS,
        PipelinePacket::Eos,
        &mut sink,
    )
    .await;

    assert!(matches!(
        sink.packets.as_slice(),
        [
            PipelinePacket::DataFrame(first),
            PipelinePacket::CapsChanged(_),
            PipelinePacket::DataFrame(second),
        ] if first.timing.pts_ns == 0
            && second.timing.pts_ns == FRAME_PERIOD_NS
            && detection_x(first).is_none()
            && detection_x(second).is_none()
    ));
}

#[cfg(feature = "analytics-json")]
#[tokio::test]
async fn a_launch_line_joins_a_rate_reduced_detector_branch() {
    use g2g_plugins::appsink::{register_appsink_pull, Pull};

    // videorate restamps onto a 10 fps grid from zero, so the detection sits 1 ns after its video frame
    const SOURCE_FRAMES: u64 = 6;
    const DETECTED_FRAME: u64 = 3;
    let detected_pts_ns = DETECTED_FRAME * FRAME_PERIOD_NS;
    let detection_x_pixels = 16.0;

    let directory = std::env::temp_dir().join(format!("g2g-m2093-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let records = directory.join("records.jsonl");
    std::fs::write(
        &records,
        format!(
            "{{\"pts\":{},\"detections\":[{{\"label\":\"ball\",\"x\":{detection_x_pixels},\"y\":8,\"w\":4,\"h\":4,\"score\":0.9}}]}}\n",
            detected_pts_ns as f64 / 1e9
        ),
    )
    .unwrap();

    let channel = "m2093-joined";
    let pull = register_appsink_pull(channel);
    let line = format!(
        "analyticsjoin name=j ! appsink channel={channel} \
         videotestsrc num-buffers={SOURCE_FRAMES} ! video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate=30/1 ! tee name=t \
         t. ! videorate framerate=10/1 ! metareplay location={} ! j.analytics \
         t. ! j.video",
        records.to_string_lossy()
    );
    let graph = g2g_core::runtime::parse_launch(&g2g_plugins::registry::default_registry(), &line)
        .expect("the join line builds");
    let clock = g2g_plugins::clock::WallClock::new();
    let run = g2g_core::runtime::run_graph(graph, &clock, 4);
    let collect = async {
        let mut found = Vec::new();
        loop {
            match pull.try_pull() {
                Pull::Frame(frame) => found.push((frame.timing.pts_ns, detection_x(&frame))),
                Pull::Empty => tokio::task::yield_now().await,
                Pull::Ended => return found,
            }
        }
    };
    let (result, found) = tokio::join!(run, collect);
    result.expect("the join line runs");
    let expected: Vec<(u64, Option<f32>)> = (0..SOURCE_FRAMES)
        .map(|index| {
            let pts_ns = index * FRAME_PERIOD_NS;
            let x = (index == DETECTED_FRAME).then_some(detection_x_pixels / WIDTH as f32);
            (pts_ns, x)
        })
        .collect();
    assert_eq!(found, expected);
    let _ = std::fs::remove_dir_all(&directory);
}

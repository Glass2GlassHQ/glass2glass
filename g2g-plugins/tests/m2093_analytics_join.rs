#![cfg(feature = "metadata")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AnalyticsMeta, AnalyticsNode, BBox, Caps, Colorimetry, Dim, Interlace, MultiInputElement,
    ObjectDetection, Rate, RawVideoFormat, RelationKind, Tracking,
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

#[tokio::test]
async fn a_second_branch_adds_to_the_detections_already_on_the_frame() {
    let mut join = configured_join();
    let mut sink = CollectSink::default();
    let mut players = AnalyticsMeta::new();
    let player = players.add_detection(ObjectDetection {
        bbox: BBox {
            x: 0.25,
            y: 0.5,
            w: 0.1,
            h: 0.2,
        },
        label: 2,
        confidence: 0.8,
    });
    let track = players.push(AnalyticsNode::Tracking(Tracking { object_id: 7 }));
    players.relate(player, track, RelationKind::Tracks);
    let PipelinePacket::DataFrame(mut video) = video_frame(0) else {
        unreachable!()
    };
    video.meta.attach(players);

    push(
        &mut join,
        AnalyticsJoin::VIDEO,
        PipelinePacket::DataFrame(video),
        &mut sink,
    )
    .await;
    push(
        &mut join,
        AnalyticsJoin::ANALYTICS,
        detections_frame(0, 0.75),
        &mut sink,
    )
    .await;

    let frames = data_frames(&sink);
    let analytics = frames[0].meta.get::<AnalyticsMeta>().expect("analytics");
    let xs: Vec<f32> = analytics
        .detections()
        .map(|detection| detection.bbox.x)
        .collect();
    assert_eq!(
        xs,
        [0.25, 0.75],
        "the frame keeps its detection and gains the branch's"
    );
    assert_eq!(
        analytics.tracking_id(player),
        Some(7),
        "existing relations still resolve"
    );
    assert_eq!(analytics.tracking_id(analytics.nodes.len() - 1), None);
}

#[derive(Default)]
struct ScriptedSource {
    packets: Vec<PipelinePacket>,
    wait_before_pushing: Option<std::sync::Arc<tokio::sync::Notify>>,
    signal_when_pushed: Option<std::sync::Arc<tokio::sync::Notify>>,
}

fn scripted(packets: Vec<PipelinePacket>) -> ScriptedSource {
    ScriptedSource {
        packets,
        ..ScriptedSource::default()
    }
}

impl g2g_core::runtime::SourceLoop for ScriptedSource {
    type RunFuture<'a>
        = std::pin::Pin<Box<dyn std::future::Future<Output = Result<u64, g2g_core::G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = std::future::Ready<Result<Caps, g2g_core::G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        std::future::ready(Ok(rgba()))
    }
    fn configure_pipeline(
        &mut self,
        _caps: &Caps,
    ) -> Result<g2g_core::ConfigureOutcome, g2g_core::G2gError> {
        Ok(g2g_core::ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn g2g_core::OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if let Some(wait) = &self.wait_before_pushing {
                wait.notified().await;
            }
            let count = self.packets.len() as u64;
            for packet in self.packets.drain(..) {
                out.push(packet).await?;
            }
            if let Some(signal) = &self.signal_when_pushed {
                signal.notify_one();
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(count)
        })
    }
}

#[derive(Clone, Default)]
struct SharedSink(std::sync::Arc<std::sync::Mutex<Vec<Seen>>>);

#[derive(Debug, Clone, PartialEq)]
enum Seen {
    Caps {
        width: u32,
    },
    Frame {
        bytes: usize,
        detection_x: Option<f32>,
    },
}

impl g2g_core::AsyncElement for SharedSink {
    type ProcessFuture<'a>
        = std::future::Ready<Result<(), g2g_core::G2gError>>
    where
        Self: 'a;
    fn intercept_caps(&self, caps: &Caps) -> Result<Caps, g2g_core::G2gError> {
        Ok(caps.clone())
    }
    fn caps_constraint_as_sink(&self) -> g2g_core::CapsConstraint<'_> {
        g2g_core::CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(
        &mut self,
        _caps: &Caps,
    ) -> Result<g2g_core::ConfigureOutcome, g2g_core::G2gError> {
        Ok(g2g_core::ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn g2g_core::OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let seen = match packet {
            PipelinePacket::CapsChanged(Caps::RawVideo {
                width: Dim::Fixed(width),
                ..
            }) => Seen::Caps { width },
            PipelinePacket::DataFrame(frame) => Seen::Frame {
                bytes: frame.domain.as_system_slice().unwrap().len(),
                detection_x: detection_x(&frame),
            },
            _ => return std::future::ready(Ok(())),
        };
        self.0.lock().unwrap().push(seen);
        std::future::ready(Ok(()))
    }
}

struct NullClock;
impl g2g_core::PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

// decoders announce a stream before they know its framerate
#[tokio::test]
async fn video_caps_with_an_unknown_framerate_mid_stream_keep_the_join_running() {
    use g2g_core::runtime::{run_graph, GraphNode};
    let unknown_rate = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
    };
    let mut video = vec![PipelinePacket::CapsChanged(unknown_rate)];
    video.extend((0..DETECTION_STRIDE).map(video_frame));
    let analytics = vec![detections_frame(0, 0.5)];

    let seen = SharedSink::default();
    let mut graph: g2g_core::Graph<GraphNode> = g2g_core::Graph::new();
    let video_source = graph.add_source(GraphNode::source(scripted(video)));
    let analytics_source = graph.add_source(GraphNode::source(scripted(analytics)));
    let join = graph.add_muxer(GraphNode::muxer(AnalyticsJoin::new()), 2);
    let sink = graph.add_sink(GraphNode::element(seen.clone()));
    graph
        .link(
            video_source,
            join.input(u8::try_from(AnalyticsJoin::VIDEO).unwrap()),
        )
        .unwrap();
    graph
        .link(
            analytics_source,
            join.input(u8::try_from(AnalyticsJoin::ANALYTICS).unwrap()),
        )
        .unwrap();
    graph.link(join.output(), sink).unwrap();

    run_graph(graph, &NullClock, 4)
        .await
        .expect("the join runs");

    let found: Vec<Option<f32>> = seen
        .0
        .lock()
        .unwrap()
        .iter()
        .filter_map(|seen| match seen {
            Seen::Frame { detection_x, .. } => Some(*detection_x),
            Seen::Caps { .. } => None,
        })
        .collect();
    assert_eq!(found, [Some(0.5), None, None]);
}

fn rgba_frame(width: u32, index: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(
            vec![0u8; (width * HEIGHT * 4) as usize].into_boxed_slice(),
        )),
        FrameTiming {
            pts_ns: index * FRAME_PERIOD_NS,
            duration_ns: FRAME_PERIOD_NS,
            ..FrameTiming::default()
        },
        index,
    ))
}

#[tokio::test]
async fn a_size_change_reaches_downstream_after_the_frames_held_before_it() {
    use g2g_core::runtime::{run_graph, GraphNode};
    const NARROW: u32 = WIDTH / 2;
    const WIDE_FRAMES: u64 = 2;
    const NARROW_FRAMES: u64 = 8;
    let narrow_caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(NARROW),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30 << 16),
        interlace: Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
    };
    let mut video: Vec<PipelinePacket> = (0..WIDE_FRAMES).map(|i| rgba_frame(WIDTH, i)).collect();
    video.push(PipelinePacket::CapsChanged(narrow_caps));
    video.extend((WIDE_FRAMES..WIDE_FRAMES + NARROW_FRAMES).map(|i| rgba_frame(NARROW, i)));
    let last_pts_ns = (WIDE_FRAMES + NARROW_FRAMES - 1) * FRAME_PERIOD_NS;
    // with link capacity 4, the join has taken the size change once all ten video packets are sent
    let video_sent = std::sync::Arc::new(tokio::sync::Notify::new());
    let video_source = ScriptedSource {
        packets: video,
        signal_when_pushed: Some(video_sent.clone()),
        ..ScriptedSource::default()
    };
    let analytics_source = ScriptedSource {
        packets: vec![detections_frame(last_pts_ns, 0.5)],
        wait_before_pushing: Some(video_sent),
        ..ScriptedSource::default()
    };

    let seen = SharedSink::default();
    let mut graph: g2g_core::Graph<GraphNode> = g2g_core::Graph::new();
    let video_node = graph.add_source(GraphNode::source(video_source));
    let analytics_node = graph.add_source(GraphNode::source(analytics_source));
    let join = graph.add_muxer(GraphNode::muxer(AnalyticsJoin::new()), 2);
    let sink = graph.add_sink(GraphNode::element(seen.clone()));
    graph
        .link(
            video_node,
            join.input(u8::try_from(AnalyticsJoin::VIDEO).unwrap()),
        )
        .unwrap();
    graph
        .link(
            analytics_node,
            join.input(u8::try_from(AnalyticsJoin::ANALYTICS).unwrap()),
        )
        .unwrap();
    graph.link(join.output(), sink).unwrap();

    run_graph(graph, &NullClock, 4)
        .await
        .expect("the join runs");

    let mut width = WIDTH;
    let mut frames = 0;
    for seen in seen.0.lock().unwrap().iter() {
        match seen {
            Seen::Caps { width: changed } => width = *changed,
            Seen::Frame { bytes, .. } => {
                frames += 1;
                assert_eq!(
                    *bytes,
                    (width * HEIGHT * 4) as usize,
                    "frame {frames} matches the caps before it"
                );
            }
        }
    }
    assert_eq!(frames, WIDE_FRAMES + NARROW_FRAMES);
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

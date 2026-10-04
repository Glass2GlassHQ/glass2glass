//! M1214: `intersink` / `intersrc` link independent graphs in one process.
//!
//! `default_registry` is `std`-gated, so this file is too.
#![cfg(feature = "std")]

use std::collections::BTreeSet;
use std::task::{Context, Poll};
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph, RunStats, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, Colorimetry, Dim, FrameTiming, G2gError, OutputSink, PipelineClock,
    PipelinePacket, PropValue, PropertySpec, PushOutcome, Rate, TextFormat, VideoCodec,
};
use g2g_plugins::appsink::{register_appsink_pull, AppSinkPull, Pull};
use g2g_plugins::appsrc::{register_appsrc, AppSrcFeed};
use g2g_plugins::inter::{InterSink, InterSrc};
use g2g_plugins::registry::default_registry;

const WIDTH: usize = 4;
const HEIGHT: usize = 2;
const RGBA_BYTES_PER_PIXEL: usize = 4;
const FRAME_BYTES: usize = WIDTH * HEIGHT * RGBA_BYTES_PER_PIXEL;
const FRAMERATE: u64 = 30;
const FRAME_PERIOD_NS: u64 = 1_000_000_000 / FRAMERATE;
const TEST_SOURCE_FRAMES: u64 = 10;
const LINK_CAPACITY: usize = 4;
const RUN_DEADLINE: Duration = Duration::from_secs(20);
const CONSUMER_HEAD_START: Duration = Duration::from_millis(200);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn video_caps() -> String {
    format!("video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate={FRAMERATE}/1")
}

fn text_caps() -> Caps {
    Caps::Text {
        format: TextFormat::Utf8,
    }
}

async fn run_launch(line: String) -> Result<RunStats, G2gError> {
    let registry = default_registry();
    let graph = parse_launch(&registry, &line).expect("parses");
    run_graph(graph, &ZeroClock, LINK_CAPACITY).await
}

fn appsrc_producer(name: &str) -> String {
    format!(
        "appsrc channel={name} caps={} ! intersink producer-name={name}",
        video_caps()
    )
}

fn appsink_consumer(name: &str, channel: &str) -> String {
    format!(
        "intersrc producer-name={name} ! appsink channel={channel} caps={}",
        video_caps()
    )
}

fn frame_fill(index: u64) -> u8 {
    (index % u64::from(u8::MAX)) as u8
}

// a consumer that has a frame is subscribed, so the end of stream reaches it
async fn push_until_each_received(
    feed: &AppSrcFeed,
    pulls: &[&AppSinkPull],
) -> (BTreeSet<u64>, Vec<Vec<Frame>>) {
    let mut pushed = BTreeSet::new();
    let mut received: Vec<Vec<Frame>> = pulls.iter().map(|_| Vec::new()).collect();
    let mut index = 0u64;
    while received.iter().any(Vec::is_empty) {
        let pts = index * FRAME_PERIOD_NS;
        if feed.push(&[frame_fill(index); FRAME_BYTES], pts) {
            pushed.insert(pts);
            index += 1;
        }
        tokio::task::yield_now().await;
        for (pull, frames) in pulls.iter().zip(received.iter_mut()) {
            if let Pull::Frame(frame) = pull.try_pull() {
                frames.push(frame);
            }
        }
    }
    while !feed.end_of_stream() {
        tokio::task::yield_now().await;
    }
    (pushed, received)
}

async fn drain(pull: &AppSinkPull, frames: &mut Vec<Frame>) {
    while let Some(frame) = pull.pull().await {
        frames.push(frame);
    }
}

fn assert_subsequence_of_pushed(frames: &[Frame], pushed: &BTreeSet<u64>) {
    assert!(!frames.is_empty(), "the consumer received frames");
    let mut previous_pts = None;
    for frame in frames {
        let pts = frame.timing.pts_ns;
        assert!(pushed.contains(&pts), "pts {pts} was pushed");
        assert!(previous_pts < Some(pts), "frames stay in order");
        previous_pts = Some(pts);
        let bytes = frame.domain.as_system_slice().expect("system memory");
        assert_eq!(bytes, &[frame_fill(pts / FRAME_PERIOD_NS); FRAME_BYTES][..]);
    }
}

#[derive(Default)]
struct CollectingOutput {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectingOutput {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        self.packets.extend(packet.take());
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn named_sink(name: &str) -> InterSink {
    let mut sink = InterSink::new();
    sink.set_property("producer-name", PropValue::Str(name.into()))
        .expect("producer-name");
    sink
}

fn named_source(name: &str) -> InterSrc {
    let mut source = InterSrc::new();
    source
        .set_property("producer-name", PropValue::Str(name.into()))
        .expect("producer-name");
    source
}

async fn publish(sink: &mut InterSink, pts_ns: u64, keyframe: bool) {
    sink.process(
        data_frame(pts_ns, keyframe),
        &mut CollectingOutput::default(),
    )
    .await
    .expect("publishing never waits on a consumer");
}

fn data_frame(pts_ns: u64, keyframe: bool) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8]))),
        FrameTiming {
            pts_ns,
            keyframe,
            ..FrameTiming::default()
        },
        0,
    ))
}

#[tokio::test]
async fn consumer_receives_the_producers_frames_and_ends() {
    let name = "inter_frames";
    let channel = "inter_frames_out";
    let feed = register_appsrc(name);
    let pull = register_appsink_pull(channel);
    let driver = async {
        let (pushed, mut received) = push_until_each_received(&feed, &[&pull]).await;
        drain(&pull, &mut received[0]).await;
        (pushed, received.remove(0))
    };
    let (consumer, producer, (pushed, frames)) = tokio::time::timeout(RUN_DEADLINE, async {
        tokio::join!(
            run_launch(appsink_consumer(name, channel)),
            run_launch(appsrc_producer(name)),
            driver
        )
    })
    .await
    .expect("both graphs end");
    consumer.expect("the consumer graph runs");
    producer.expect("the producer graph runs");
    assert_subsequence_of_pushed(&frames, &pushed);
    assert!(matches!(pull.try_pull(), Pull::Ended));
}

#[tokio::test]
async fn consumer_started_before_its_producer_waits_and_connects() {
    let name = "inter_waiting";
    let channel = "inter_waiting_out";
    let pull = register_appsink_pull(channel);
    let producer = async {
        tokio::time::sleep(CONSUMER_HEAD_START).await;
        assert!(
            matches!(pull.try_pull(), Pull::Empty),
            "the consumer waits instead of failing"
        );
        run_launch(format!(
            "videotestsrc num-buffers={TEST_SOURCE_FRAMES} width={WIDTH} height={HEIGHT} \
             ! intersink producer-name={name}"
        ))
        .await
    };
    let mut frames = Vec::new();
    let (consumer, producer, ()) = tokio::time::timeout(RUN_DEADLINE, async {
        tokio::join!(
            run_launch(appsink_consumer(name, channel)),
            producer,
            drain(&pull, &mut frames)
        )
    })
    .await
    .expect("both graphs end");
    consumer.expect("the consumer graph runs");
    producer.expect("the producer graph runs");
    assert!(!frames.is_empty(), "the consumer received frames");
    assert!(frames.len() as u64 <= TEST_SOURCE_FRAMES);
    for frame in &frames {
        assert_eq!(
            frame.domain.as_system_slice().map(<[u8]>::len),
            Some(FRAME_BYTES)
        );
    }
}

#[tokio::test]
async fn every_consumer_receives_the_stream() {
    let name = "inter_two_consumers";
    let channels = ["inter_two_consumers_a", "inter_two_consumers_b"];
    let feed = register_appsrc(name);
    let pulls = channels.map(register_appsink_pull);
    let driver = async {
        let (pushed, mut received) = push_until_each_received(&feed, &[&pulls[0], &pulls[1]]).await;
        for (pull, frames) in pulls.iter().zip(received.iter_mut()) {
            drain(pull, frames).await;
        }
        (pushed, received)
    };
    let (first, second, producer, (pushed, received)) = tokio::time::timeout(RUN_DEADLINE, async {
        tokio::join!(
            run_launch(appsink_consumer(name, channels[0])),
            run_launch(appsink_consumer(name, channels[1])),
            run_launch(appsrc_producer(name)),
            driver
        )
    })
    .await
    .expect("all graphs end");
    first.expect("the first consumer graph runs");
    second.expect("the second consumer graph runs");
    producer.expect("the producer graph runs");
    for frames in &received {
        assert_subsequence_of_pushed(frames, &pushed);
    }
}

#[tokio::test]
async fn a_second_producer_on_a_live_name_fails_to_configure() {
    let name = "inter_taken";
    let mut first = named_sink(name);
    first
        .configure_pipeline(&text_caps())
        .expect("first claims the name");
    let mut second = named_sink(name);
    assert!(second.configure_pipeline(&text_caps()).is_err());
    let graph = run_launch(format!(
        "videotestsrc num-buffers={TEST_SOURCE_FRAMES} ! intersink producer-name={name}"
    ))
    .await;
    assert!(graph.is_err(), "a launched second producer fails too");
    drop(first);
    second
        .configure_pipeline(&text_caps())
        .expect("the name is free once the first producer is gone");
}

#[tokio::test]
async fn dropping_the_producer_ends_its_consumers() {
    let name = "inter_dropped";
    let channel = "inter_dropped_out";
    let pull = register_appsink_pull(channel);
    let mut sink = named_sink(name);
    sink.configure_pipeline(&text_caps()).expect("claims");
    let driver = async {
        let mut index = 0;
        while !matches!(pull.try_pull(), Pull::Frame(_)) {
            publish(&mut sink, index, true).await;
            index += 1;
            tokio::task::yield_now().await;
        }
        drop(sink);
        while pull.pull().await.is_some() {}
    };
    let (consumer, ()) = tokio::time::timeout(RUN_DEADLINE, async {
        tokio::join!(
            run_launch(format!(
                "intersrc producer-name={name} ! appsink channel={channel}"
            )),
            driver
        )
    })
    .await
    .expect("the consumer ends once its producer is gone");
    consumer.expect("the consumer graph runs");
}

async fn run_to_end(source: &mut InterSrc) -> Vec<u64> {
    let mut output = CollectingOutput::default();
    let emitted = source.run(&mut output).await.expect("runs");
    assert!(matches!(output.packets.last(), Some(PipelinePacket::Eos)));
    let pts: Vec<u64> = output
        .packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame.timing.pts_ns),
            _ => None,
        })
        .collect();
    assert_eq!(emitted, pts.len() as u64);
    pts
}

#[tokio::test]
async fn a_full_consumer_queue_drops_its_oldest_frames() {
    const MAX_BUFFERS: u64 = 2;
    const PUBLISHED: u64 = 5;
    let name = "inter_drop_oldest";
    let mut sink = named_sink(name);
    sink.configure_pipeline(&text_caps()).expect("claims");
    let mut source = named_source(name);
    source
        .set_property("max-buffers", PropValue::Uint(MAX_BUFFERS))
        .expect("max-buffers");
    assert_eq!(source.intercept_caps().await, Ok(text_caps()));
    source.configure_pipeline(&text_caps()).expect("configures");
    for pts in 0..PUBLISHED {
        publish(&mut sink, pts, true).await;
    }
    drop(sink);
    assert_eq!(
        run_to_end(&mut source).await,
        ((PUBLISHED - MAX_BUFFERS)..PUBLISHED).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn a_consumer_joining_compressed_video_mid_stream_starts_at_a_keyframe() {
    let name = "inter_mid_stream";
    let caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(WIDTH as u32),
        height: Dim::Fixed(HEIGHT as u32),
        framerate: Rate::Any,
        colorimetry: Colorimetry::UNKNOWN,
    };
    let mut sink = named_sink(name);
    sink.configure_pipeline(&caps).expect("claims");
    publish(&mut sink, 0, true).await;
    let mut source = named_source(name);
    assert_eq!(source.intercept_caps().await, Ok(caps.clone()));
    source.configure_pipeline(&caps).expect("configures");
    let joined = [(1, false), (2, true), (3, false)];
    for (pts, keyframe) in joined {
        publish(&mut sink, pts, keyframe).await;
    }
    drop(sink);
    assert_eq!(run_to_end(&mut source).await, [2, 3]);
}

fn declared_default(specs: &[PropertySpec], name: &str) -> Option<PropValue> {
    let spec = specs.iter().find(|spec| spec.name == name)?;
    spec.parse_value(spec.default?).ok()
}

#[test]
fn properties_round_trip() {
    const MAX_BUFFERS: u64 = 3;
    let name = PropValue::Str("inter_properties".into());
    let mut sink = InterSink::new();
    assert_eq!(
        sink.get_property("producer-name"),
        declared_default(sink.properties(), "producer-name")
    );
    sink.set_property("producer-name", name.clone()).unwrap();
    assert_eq!(sink.get_property("producer-name"), Some(name.clone()));

    let mut source = InterSrc::new();
    for property in ["producer-name", "max-buffers"] {
        assert_eq!(
            source.get_property(property),
            declared_default(source.properties(), property),
            "{property} reports its declared default"
        );
    }
    source.set_property("producer-name", name.clone()).unwrap();
    assert_eq!(source.get_property("producer-name"), Some(name));
    source
        .set_property("max-buffers", PropValue::Uint(MAX_BUFFERS))
        .unwrap();
    assert_eq!(
        source.get_property("max-buffers"),
        Some(PropValue::Uint(MAX_BUFFERS))
    );
    assert!(
        source
            .set_property("max-buffers", PropValue::Uint(0))
            .is_err(),
        "a consumer queue needs room for one frame"
    );
}

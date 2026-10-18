//! `KvsWebRtcSink` as master and `KvsWebRtcSrc` as viewer against the real
//! Amazon Kinesis Video Streams signalling service, in one process. Ignored by
//! default: it needs AWS credentials allowed to use the kinesisvideo signalling
//! APIs and an existing signalling channel.
//!
//! ```sh
//! aws kinesisvideo create-signaling-channel --channel-name g2g-smoke
//! eval "$(aws configure export-credentials --format env)"
//! export AWS_REGION=us-west-2 G2G_KVS_CHANNEL=g2g-smoke
//! cargo test -p g2g-plugins --features webrtc-kvs --test kvswebrtc_smoke -- --ignored --nocapture
//! aws kinesisvideo delete-signaling-channel --channel-arn <arn from describe-signaling-channel>
//! ```
//!
//! `G2G_DEBUG=KvsWebRtc*:debug` shows the signalling steps. Green means both
//! elements resolved the channel, signed their WebSocket connections, traded
//! SDP and candidates through KVS, and the viewer received the master's video
//! starting on a keyframe.

#![cfg(all(target_os = "linux", feature = "webrtc-kvs"))]

mod numbuffers_common;
mod webrtc_streams_common;

use core::task::{Context, Poll};
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, FrameTiming, G2gError, MemoryDomain, MultiOutputSource, OutputSink,
    PipelinePacket, PushOutcome, VideoCodec,
};
use g2g_plugins::kvswebrtcsink::KvsWebRtcSink;
use g2g_plugins::kvswebrtcsrc::KvsWebRtcSrc;
use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
use g2g_plugins::IceTransportPolicy;
use numbuffers_common::CollectPorts;
use webrtc_streams_common::{data_frames, fixed_caps, synthetic_stream, FPS};

const CHANNEL_VARIABLE: &str = "G2G_KVS_CHANNEL";
const VIDEO_CODEC: WebRtcVideoCodec = WebRtcVideoCodec::H264;
const PUBLISH_SECONDS: u32 = 6;
// the master is connected and waiting before the viewer offers
const VIEWER_START_DELAY: Duration = Duration::from_secs(3);
const DRAIN_BEFORE_EOS: Duration = Duration::from_millis(500);
const RUN_TIMEOUT: Duration = Duration::from_secs(60);

struct Discard;

impl OutputSink for Discard {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        packet.take();
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

async fn publish(sink: &mut KvsWebRtcSink, units: &[Vec<u8>]) {
    let interval = Duration::from_secs(1) / FPS;
    for (sequence, unit) in units.iter().enumerate() {
        let timing = FrameTiming {
            pts_ns: sequence as u64 * interval.as_nanos() as u64,
            ..FrameTiming::default()
        };
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(unit.clone().into_boxed_slice())),
            timing,
            sequence as u64,
        );
        sink.process(PipelinePacket::DataFrame(frame), &mut Discard)
            .await
            .expect("the master publishes");
        tokio::time::sleep(interval).await;
    }
    tokio::time::sleep(DRAIN_BEFORE_EOS).await;
    sink.process(PipelinePacket::Eos, &mut Discard)
        .await
        .unwrap();
}

async fn master_streams_to_a_viewer(policy: IceTransportPolicy) {
    g2g_core::log::init_from_env();
    let channel = std::env::var(CHANNEL_VARIABLE)
        .expect("G2G_KVS_CHANNEL names an existing signalling channel");
    let units = synthetic_stream(VIDEO_CODEC, (PUBLISH_SECONDS * FPS) as usize);
    let mut sink = KvsWebRtcSink::new()
        .with_channel_name(channel.clone())
        .with_ice_transport_policy(policy);
    sink.configure_pipeline(&fixed_caps(VIDEO_CODEC)).unwrap();
    let mut src = KvsWebRtcSrc::new()
        .with_channel_name(channel)
        .with_video_codec(VIDEO_CODEC)
        .with_ice_transport_policy(policy)
        .with_frame_limit(units.len() as u64);
    let mut out = CollectPorts::new(src.output_count());

    let (_, emitted) = tokio::time::timeout(RUN_TIMEOUT, async {
        tokio::join!(publish(&mut sink, &units), async {
            tokio::time::sleep(VIEWER_START_DELAY).await;
            src.run(&mut out).await
        })
    })
    .await
    .expect("the viewer run ends");
    let emitted = emitted.expect("the viewer run succeeds");
    eprintln!("viewer emitted {emitted} access units");

    let [video, _audio] = out.packets.as_slice() else {
        panic!("two output ports")
    };
    assert!(
        matches!(
            video.first(),
            Some(PipelinePacket::CapsChanged(Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }))
        ),
        "{:?}",
        video.first()
    );
    assert!(matches!(video.last(), Some(PipelinePacket::Eos)));
    let video_frames = data_frames(video);
    let first = video_frames.first().expect("video frames arrived");
    assert!(first.timing.keyframe, "video starts on a keyframe");
}

#[tokio::test]
#[ignore = "needs AWS credentials and a KVS signalling channel (see the module docs)"]
async fn master_streams_to_a_viewer_through_kvs() {
    master_streams_to_a_viewer(IceTransportPolicy::All).await;
}

// neither side offers a host or reflexive candidate, so the media went through the service's TURN
#[tokio::test]
#[ignore = "needs AWS credentials and a KVS signalling channel (see the module docs)"]
async fn master_streams_to_a_viewer_through_the_kvs_relay() {
    master_streams_to_a_viewer(IceTransportPolicy::Relay).await;
}

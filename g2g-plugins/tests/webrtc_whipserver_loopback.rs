//! `WhipServerSrc` against the g2g WHIP client (`WebRtcSink`) on localhost, once
//! per carried video codec and once for Opus audio. Synthetic access units, no
//! media server and no fixture files, so it runs by default.

#![cfg(all(target_os = "linux", feature = "webrtc"))]

mod webrtc_streams_common;

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::time::Duration;

use tokio::sync::mpsc;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, FrameTiming, G2gError, MemoryDomain, MultiOutputSink,
    MultiOutputSource, OutputSink, PipelinePacket, PropValue, PushOutcome,
};
use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
use g2g_plugins::webrtcsink::WebRtcSink;
use g2g_plugins::whipserversrc::WhipServerSrc;
use webrtc_streams_common::{fixed_caps, payload, synthetic_stream, FPS};

const VIDEO_PORT: usize = 0;
const AUDIO_PORT: usize = 1;
const VIDEO_UNITS: usize = 60;
const AUDIO_PACKETS: usize = 50;
const AUDIO_PACKET_BYTES: usize = 20;
const AUDIO_PACKET_INTERVAL: Duration = Duration::from_millis(20);
const OPUS_CHANNELS: u8 = 2;
const OPUS_SAMPLE_RATE: u32 = 48_000;
// let the sender's pacer flush before EOS deletes the resource
const DRAIN_BEFORE_EOS: Duration = Duration::from_millis(500);
const RUN_TIMEOUT: Duration = Duration::from_secs(20);
const MIN_DELIVERED_PERCENT: usize = 90;
/// Every packet the server pushed, per output port.
struct Collector {
    packets: mpsc::UnboundedSender<(usize, PipelinePacket)>,
}

impl MultiOutputSink for Collector {
    fn poll_push_to(
        &mut self,
        _cx: &mut Context<'_>,
        port: usize,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        let packet = packet.take().expect("push without a packet");
        let _ = self.packets.send((port, packet));
        Poll::Ready(Ok(PushOutcome::Accepted))
    }

    fn port_count(&self) -> usize {
        2
    }
}

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

fn opus_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: OPUS_CHANNELS,
        sample_rate: OPUS_SAMPLE_RATE,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Publish `units` through a `WebRtcSink` at `interval`, then EOS.
async fn publish(whip_url: String, caps: Caps, units: &[Vec<u8>], interval: Duration) {
    let mut sink = WebRtcSink::new(whip_url);
    sink.configure_pipeline(&caps).unwrap();
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
            .unwrap();
        tokio::time::sleep(interval).await;
    }
    tokio::time::sleep(DRAIN_BEFORE_EOS).await;
    sink.process(PipelinePacket::Eos, &mut Discard)
        .await
        .unwrap();
}

/// Run a server for `video_codec`, publish `units` with `caps` into it, and
/// return what each output port received.
async fn loopback(
    video_codec: WebRtcVideoCodec,
    caps: Caps,
    units: Vec<Vec<u8>>,
    interval: Duration,
) -> [Vec<PipelinePacket>; 2] {
    let host_addr = format!("http://127.0.0.1:{}", free_port());
    let mut server = WhipServerSrc::new(host_addr.clone());
    server
        .set_property("video-codec", PropValue::Str(video_codec.name().into()))
        .unwrap();
    let (packets, mut received) = mpsc::unbounded_channel();
    let mut collector = Collector { packets };

    let mut ports: [Vec<PipelinePacket>; 2] = [Vec::new(), Vec::new()];
    let run: Pin<Box<dyn Future<Output = Result<u64, G2gError>> + '_>> = server.run(&mut collector);
    let client = async {
        // both caps announcements follow the bind, so the server is listening
        let mut announced = Vec::new();
        for _ in 0..2 {
            announced.push(received.recv().await.unwrap());
        }
        publish(format!("{host_addr}/whip"), caps, &units, interval).await;
        announced
    };
    let (result, announced) =
        tokio::time::timeout(RUN_TIMEOUT, async { tokio::join!(run, client) })
            .await
            .expect("loopback run timed out");
    result.expect("server run failed");
    for (port, packet) in announced {
        ports[port].push(packet);
    }
    while let Ok((port, packet)) = received.try_recv() {
        ports[port].push(packet);
    }
    ports
}

fn data_frames(packets: &[PipelinePacket]) -> Vec<&Frame> {
    packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .collect()
}

fn bytes(frame: &Frame) -> &[u8] {
    match &frame.domain {
        MemoryDomain::System(slice) => slice.as_slice(),
        _ => panic!("expected system memory"),
    }
}

fn assert_ends_with_eos(ports: &[Vec<PipelinePacket>; 2]) {
    for (port, packets) in ports.iter().enumerate() {
        assert!(
            matches!(packets.last(), Some(PipelinePacket::Eos)),
            "port {port} did not end with EOS"
        );
    }
}

fn min_delivered(sent: usize) -> usize {
    sent * MIN_DELIVERED_PERCENT / 100
}

#[tokio::test]
async fn every_video_codec_publishes_into_the_server() {
    let interval = Duration::from_secs(1) / FPS;
    for codec in WebRtcVideoCodec::ALL {
        let units = synthetic_stream(codec, VIDEO_UNITS);
        let ports = loopback(codec, fixed_caps(codec), units.clone(), interval).await;

        let Some(PipelinePacket::CapsChanged(Caps::CompressedVideo {
            codec: announced, ..
        })) = ports[VIDEO_PORT].first()
        else {
            panic!("{codec:?}: video port did not open with video caps");
        };
        assert_eq!(*announced, codec.video_codec());

        let frames = data_frames(&ports[VIDEO_PORT]);
        assert!(
            frames.len() >= min_delivered(units.len()),
            "{codec:?}: {} of {} units arrived",
            frames.len(),
            units.len()
        );
        assert_eq!(bytes(frames[0]), &units[0][..], "{codec:?}");
        assert!(frames[0].timing.keyframe, "{codec:?}");
        assert!(data_frames(&ports[AUDIO_PORT]).is_empty(), "{codec:?}");
        assert_ends_with_eos(&ports);
    }
}

#[tokio::test]
async fn opus_publishes_into_the_audio_output() {
    let packets: Vec<Vec<u8>> = (0..AUDIO_PACKETS)
        .map(|index| payload(AUDIO_PACKET_BYTES, index as u8))
        .collect();
    let ports = loopback(
        WebRtcVideoCodec::default(),
        opus_caps(),
        packets.clone(),
        AUDIO_PACKET_INTERVAL,
    )
    .await;

    assert!(matches!(
        ports[AUDIO_PORT].first(),
        Some(PipelinePacket::CapsChanged(Caps::Audio {
            format: AudioFormat::Opus,
            ..
        }))
    ));
    let frames = data_frames(&ports[AUDIO_PORT]);
    assert!(
        frames.len() >= min_delivered(packets.len()),
        "{} of {} packets arrived",
        frames.len(),
        packets.len()
    );
    assert_eq!(bytes(frames[0]), &packets[0][..]);
    assert!(data_frames(&ports[VIDEO_PORT]).is_empty());
    assert_ends_with_eos(&ports);
}

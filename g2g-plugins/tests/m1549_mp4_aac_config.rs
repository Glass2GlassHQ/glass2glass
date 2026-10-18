//! M1549: an MP4 AAC track whose object type ADTS cannot signal (ER AAC, ELD,
//! USAC) leaves the demuxer as raw access units led by its AudioSpecificConfig,
//! and `aacparse` forwards both unchanged.
#![cfg(feature = "std")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/eld_mp4.rs"]
mod eld_mp4;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, G2gError, MultiOutputElement,
    MultiOutputSink, PropValue, PushOutcome,
};
use g2g_plugins::aacparse::AacParse;
use g2g_plugins::mp4demux::Mp4Demux;
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};

use collect_sink::CollectSink;
use eld_mp4::{access_units, progressive_eld_mp4, ELD_CONFIG};

fn data_frame(data: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

fn frame_bytes(packets: &[PipelinePacket]) -> Vec<Vec<u8>> {
    packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(
                frame
                    .domain
                    .require_system_slice("test")
                    .expect("system frame")
                    .to_vec(),
            ),
            _ => None,
        })
        .collect()
}

fn caps_changes(packets: &[PipelinePacket]) -> Vec<Caps> {
    packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::CapsChanged(caps) => Some(caps.clone()),
            _ => None,
        })
        .collect()
}

fn assert_config_leads_raw_units(frames: &[Vec<u8>]) {
    let expected: Vec<Vec<u8>> = std::iter::once(ELD_CONFIG.to_vec())
        .chain(access_units())
        .collect();
    assert_eq!(
        frames, expected,
        "the AudioSpecificConfig leads, then each access unit unframed"
    );
}

async fn single_output_demux(file: Vec<u8>) -> Vec<PipelinePacket> {
    let mut demux = Mp4Demux::new();
    demux
        .set_property("stream", PropValue::Str("aac".into()))
        .expect("stream=aac");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4,
        })
        .expect("configure");
    let mut sink = CollectSink::default();
    demux
        .process(data_frame(file), &mut sink)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    sink.packets
}

#[derive(Default)]
struct PortCapture {
    frames: Vec<Vec<u8>>,
}

impl MultiOutputSink for PortCapture {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        _port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.frames.extend(frame_bytes(&[packet]));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }

    fn port_count(&self) -> usize {
        1
    }
}

#[tokio::test]
async fn mp4demux_sends_eld_config_ahead_of_raw_units() {
    let packets = single_output_demux(progressive_eld_mp4()).await;
    assert!(
        packets.iter().any(|packet| matches!(
            packet,
            PipelinePacket::CapsChanged(Caps::Audio {
                format: AudioFormat::Aac,
                ..
            })
        )),
        "AAC caps announced"
    );
    assert_config_leads_raw_units(&frame_bytes(&packets));
}

#[tokio::test]
async fn multi_output_demux_sends_eld_config_ahead_of_raw_units() {
    let file = progressive_eld_mp4();
    let streams = forwardable_streams(&file);
    assert_eq!(streams.len(), 1, "one AAC track discovered");
    let mut demux = Mp4DemuxN::new(vec![Mp4Port {
        track_id: streams[0].track_id,
        caps: streams[0].caps.clone(),
    }]);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .expect("configure");
    let mut capture = PortCapture::default();
    demux
        .process(data_frame(file), &mut capture)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut capture)
        .await
        .expect("eos");
    assert_config_leads_raw_units(&capture.frames);
}

#[tokio::test]
async fn aacparse_forwards_config_and_raw_units_unchanged() {
    let demuxed = single_output_demux(progressive_eld_mp4()).await;
    let mut parse = AacParse::new();
    parse
        .configure_pipeline(&Caps::Audio {
            format: AudioFormat::Aac,
            channels: 0,
            sample_rate: 0,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .expect("configure");
    let demuxed_caps = caps_changes(&demuxed);
    let mut sink = CollectSink::default();
    for packet in demuxed {
        parse.process(packet, &mut sink).await.expect("parse");
    }
    parse
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    assert_config_leads_raw_units(&frame_bytes(&sink.packets));
    assert_eq!(
        caps_changes(&sink.packets),
        demuxed_caps,
        "only the demuxer's caps pass, nothing read out of the raw bytes"
    );
}

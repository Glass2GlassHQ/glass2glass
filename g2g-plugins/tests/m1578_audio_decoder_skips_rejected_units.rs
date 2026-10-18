//! M1578: `FfmpegAudioDec` drops an access unit libavcodec rejects and decodes
//! on, as the ffmpeg CLI does, instead of failing the pipeline.
#![cfg(all(target_os = "linux", feature = "ffmpeg"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use std::path::PathBuf;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, AudioFormat, Caps, ANY_CHANNELS};
use g2g_plugins::ffmpegaudiodec::FfmpegAudioDec;

use collect_sink::CollectSink;

const FIXTURE: &str = "aac_stereo_44100_adts.aac";
const ADTS_HEADER_LEN: usize = 7;
// libavcodec rejects an ADTS frame of all-zero payload as invalid data
const CORRUPT_PAYLOAD_BYTE: u8 = 0x00;

fn data_frame(data: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

// the 13-bit frame length across bytes 3..6 of the header
fn adts_frame_len(header: &[u8]) -> usize {
    (usize::from(header[3] & 0x03) << 11)
        | (usize::from(header[4]) << 3)
        | usize::from(header[5] >> 5)
}

async fn decode(packets: Vec<PipelinePacket>) -> Vec<u8> {
    let mut decoder = FfmpegAudioDec::new();
    decoder
        .configure_pipeline(&Caps::Audio {
            format: AudioFormat::Aac,
            channels: ANY_CHANNELS,
            sample_rate: 0,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .expect("configure");
    let mut sink = CollectSink::default();
    for packet in packets.into_iter().chain([PipelinePacket::Eos]) {
        decoder.process(packet, &mut sink).await.expect("decode");
    }
    sink.packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(
                frame
                    .domain
                    .require_system_slice("test")
                    .expect("pcm")
                    .to_vec(),
            ),
            _ => None,
        })
        .flatten()
        .collect()
}

#[tokio::test]
async fn a_rejected_unit_is_skipped() {
    let adts = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(FIXTURE),
    )
    .expect("the fixture is checked in");
    let first_len = adts_frame_len(&adts);
    let (first, rest) = adts.split_at(first_len);
    let mut corrupt = first[..ADTS_HEADER_LEN].to_vec();
    corrupt.resize(first_len, CORRUPT_PAYLOAD_BYTE);

    let clean = decode(vec![data_frame(adts.clone())]).await;
    let with_rejected = decode(vec![
        data_frame(first.to_vec()),
        data_frame(corrupt),
        data_frame(rest.to_vec()),
    ])
    .await;
    assert!(!clean.is_empty(), "the fixture decodes");
    assert!(
        with_rejected == clean,
        "the rejected unit adds and removes nothing"
    );
}

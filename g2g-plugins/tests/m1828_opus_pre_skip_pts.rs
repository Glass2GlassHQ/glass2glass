//! M1828: decoded Opus is presented from its first kept sample, so the pts run
//! the pre-skip behind the container's packet stamps, the way ffmpeg and gst
//! report them.
#![cfg(all(feature = "opus", feature = "std"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::time::ticks_to_ns;
use g2g_core::{AsyncElement, AudioFormat, ByteStreamEncoding, Caps};
use g2g_plugins::oggdemux::OggDemux;
use g2g_plugins::opusdec::OpusDec;
use g2g_plugins::opusparse::OPUS_RATE_HZ;

use collect_sink::CollectSink;

// libopus output in Ogg, whose OpusHead carries the encoder's pre-skip
const FIXTURE: &str = "opus_stereo_48k.opus";
const BYTES_PER_SAMPLE: usize = 2;

fn data_frame(data: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

#[tokio::test]
async fn decoded_opus_pts_count_only_the_kept_samples() {
    let path = format!("{}/tests/fixtures/{FIXTURE}", env!("CARGO_MANIFEST_DIR"));
    let file = std::fs::read(path).expect("the fixture is checked in");

    let mut demux = OggDemux::new();
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        })
        .expect("configure");
    let mut demuxed = CollectSink::default();
    demux
        .process(data_frame(file), &mut demuxed)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut demuxed)
        .await
        .expect("eos");

    let mut decoder = OpusDec::new();
    decoder
        .configure_pipeline(&Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: OPUS_RATE_HZ,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .expect("configure");
    let mut decoded = CollectSink::default();
    for packet in demuxed.packets {
        decoder.process(packet, &mut decoded).await.expect("decode");
    }

    let frames: Vec<&Frame> = decoded
        .packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .collect();
    assert!(frames.len() > 2, "the fixture decodes to several frames");
    let mut samples_before = 0u64;
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(
            frame.timing.pts_ns,
            ticks_to_ns(samples_before, OPUS_RATE_HZ),
            "frame {index} starts where the samples before it end"
        );
        let bytes = frame
            .domain
            .require_system_slice("test")
            .expect("pcm")
            .len();
        let samples = (bytes / (2 * BYTES_PER_SAMPLE)) as u64;
        assert_eq!(frame.timing.duration_ns, ticks_to_ns(samples, OPUS_RATE_HZ));
        samples_before += samples;
    }
}

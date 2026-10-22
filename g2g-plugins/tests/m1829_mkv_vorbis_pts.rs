//! M1829: Vorbis out of Matroska is stamped with each packet's own timestamp,
//! the frames of a laced block spaced by their lapped durations, as ffmpeg
//! reports them.
#![cfg(all(feature = "std", feature = "vorbis"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::time::ticks_to_ns;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, PropValue, ANY_CHANNELS, ANY_SAMPLE_RATE,
};
use g2g_plugins::mkvdemux::MkvDemux;
use g2g_plugins::vorbisdec::VorbisDec;

use collect_sink::CollectSink;

// fate-suite mkv/xiph_lacing.mka: Xiph-laced Vorbis blocks on a millisecond timestamp grid
const FIXTURE: &str = "vorbis_xiph_laced.mka";
// ffprobe -show_frames -show_entries frame=pts_time -of csv=p=0 on the fixture
const FFPROBE_PTS: &str = "vorbis_xiph_laced.mka.pts";
const NS_PER_SECOND: f64 = 1e9;

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

async fn run<E: AsyncElement>(
    element: &mut E,
    caps: &Caps,
    packets: Vec<PipelinePacket>,
) -> Vec<PipelinePacket> {
    element.configure_pipeline(caps).expect("configure");
    let mut sink = CollectSink::default();
    for packet in packets {
        element.process(packet, &mut sink).await.expect("process");
    }
    element
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    sink.packets
}

#[tokio::test]
async fn laced_vorbis_in_matroska_takes_the_packet_timestamps() {
    let file = std::fs::read(fixture(FIXTURE)).expect("the fixture is checked in");
    let expected: Vec<u64> = std::fs::read_to_string(fixture(FFPROBE_PTS))
        .expect("the probe output is checked in")
        .lines()
        .map(|seconds| (seconds.parse::<f64>().expect("seconds") * NS_PER_SECOND).round() as u64)
        .collect();

    let mut demux = MkvDemux::new();
    demux
        .set_property("stream", PropValue::Str("vorbis".into()))
        .expect("stream=vorbis");
    let demuxed = run(
        &mut demux,
        &Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        },
        vec![PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(file.into_boxed_slice())),
            FrameTiming::default(),
            0,
        ))],
    )
    .await;
    let decoded = run(
        &mut VorbisDec::new(),
        &Caps::Audio {
            format: AudioFormat::Vorbis,
            channels: ANY_CHANNELS,
            sample_rate: ANY_SAMPLE_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        },
        demuxed,
    )
    .await;

    let sample_rate = decoded
        .iter()
        .find_map(|packet| match packet {
            PipelinePacket::CapsChanged(Caps::Audio { sample_rate, .. }) => Some(*sample_rate),
            _ => None,
        })
        .expect("the decoder announced its caps");
    // ffmpeg spaces a lace by a duration cut to its timebase, so it drifts under a sample
    let tolerance_ns = ticks_to_ns(1, sample_rate);
    let pts: Vec<u64> = decoded
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame.timing.pts_ns),
            _ => None,
        })
        .collect();
    assert_eq!(pts.len(), expected.len(), "one frame per decoded packet");
    for (index, (got, want)) in pts.iter().zip(&expected).enumerate() {
        assert!(
            got.abs_diff(*want) <= tolerance_ns,
            "frame {index}: {got} ns, ffprobe {want} ns"
        );
    }
}

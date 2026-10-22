//! M1827: the encoder delay and padding an MP3's LAME tag or an M4A's
//! `iTunSMPB` tag records go downstream as a `Segment`, and the decoder drops
//! the samples outside it, the way ffmpeg and gst trim them.
#![cfg(all(target_os = "linux", feature = "ffmpeg"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;
mod ffprobe;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, PropValue, Segment, ANY_CHANNELS,
};
use g2g_plugins::ffmpegaudiodec::FfmpegAudioDec;
use g2g_plugins::mp4demux::Mp4Demux;
use g2g_plugins::mpegaudioparse::MpegAudioParse;

use collect_sink::CollectSink;
use ffprobe::{fixture_path, Probe};

// ffmpeg's libmp3lame output, whose Info frame carries a LAME tag with the delay and padding
const MP3_FIXTURE: &str = "mp3_stereo_44100_id3v2.mp3";
// fate-suite audiomatch/tones_nero_44100_stereo_aac_lc.m4a: Nero AAC with iTunSMPB and no edit list
const ITUNES_FIXTURE: &str = "aac_lc_stereo_44100_itunsmpb.m4a";
const BYTES_PER_SAMPLE: usize = 2;
// an odd size, so frames straddle the input buffers
const CHUNK_LEN: usize = 997;

struct Decoded {
    pcm: Vec<u8>,
    bytes_per_sample_frame: usize,
}

fn data_frame(data: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

fn any_layout(format: AudioFormat) -> Caps {
    Caps::Audio {
        format,
        channels: ANY_CHANNELS,
        sample_rate: 0,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

async fn run<E: AsyncElement>(
    element: &mut E,
    packets: Vec<PipelinePacket>,
) -> Vec<PipelinePacket> {
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

async fn decode(format: AudioFormat, packets: Vec<PipelinePacket>) -> Decoded {
    let mut decoder = FfmpegAudioDec::new();
    decoder
        .configure_pipeline(&any_layout(format))
        .expect("configure");
    let decoded = run(&mut decoder, packets).await;
    let channels = decoded
        .iter()
        .find_map(|packet| match packet {
            PipelinePacket::CapsChanged(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels,
                ..
            }) => Some(usize::from(*channels)),
            _ => None,
        })
        .expect("the decoder announced its PCM caps");
    let pcm = decoded
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .flat_map(|frame| {
            frame
                .domain
                .require_system_slice("test")
                .expect("pcm")
                .to_vec()
        })
        .collect();
    Decoded {
        pcm,
        bytes_per_sample_frame: channels * BYTES_PER_SAMPLE,
    }
}

fn the_segment(packets: &[PipelinePacket]) -> Segment {
    let mut segments = packets.iter().filter_map(|packet| match packet {
        PipelinePacket::Segment(segment) => Some(*segment),
        _ => None,
    });
    let segment = segments
        .next()
        .expect("the encoder delay goes out as a segment");
    assert!(segments.next().is_none(), "one segment");
    segment
}

// the same packets with the segment left out, copied since a frame is not Clone
fn without_segments(packets: &[PipelinePacket]) -> Vec<PipelinePacket> {
    packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(data_frame(
                frame.domain.require_system_slice("test").expect("coded"),
            )),
            PipelinePacket::CapsChanged(caps) => Some(PipelinePacket::CapsChanged(caps.clone())),
            _ => None,
        })
        .collect()
}

fn sample_bytes(decoded: &Decoded, samples: std::ops::Range<u64>) -> std::ops::Range<usize> {
    samples.start as usize * decoded.bytes_per_sample_frame
        ..samples.end as usize * decoded.bytes_per_sample_frame
}

#[tokio::test]
async fn lame_tag_drops_the_mp3_delay_and_padding() {
    let probe = Probe::read(MP3_FIXTURE);
    // ffprobe reports the trimmed stream: its start is the delay, its duration the samples kept
    let time_base: u64 = probe
        .text("time_base")
        .strip_prefix("1/")
        .and_then(|denominator| denominator.parse().ok())
        .expect("a 1/n time base");
    let to_samples = |ticks: &str| {
        ticks.parse::<u64>().expect("a tick count") * u64::from(probe.sample_rate) / time_base
    };
    let delay = to_samples(&probe.text("start_pts"));
    let kept = to_samples(&probe.text("duration_ts"));

    let bytes = std::fs::read(fixture_path(MP3_FIXTURE)).expect("the fixture is checked in");
    let mut parser = MpegAudioParse::new();
    parser
        .configure_pipeline(&any_layout(AudioFormat::Mp3))
        .expect("configure");
    let parsed = run(
        &mut parser,
        bytes.chunks(CHUNK_LEN).map(data_frame).collect(),
    )
    .await;
    let segment = the_segment(&parsed);

    let whole = decode(AudioFormat::Mp3, without_segments(&parsed)).await;
    let trimmed = decode(AudioFormat::Mp3, parsed).await;
    let kept_bytes = sample_bytes(&whole, delay..delay + kept);
    assert!(kept_bytes.end < whole.pcm.len(), "the padding is cut");
    assert_eq!(trimmed.pcm.len(), kept_bytes.len(), "{segment:?}");
    assert!(
        trimmed.pcm == whole.pcm[kept_bytes],
        "the decode drops exactly the delay and padding samples"
    );
}

// the iTunSMPB value's hex fields: a zero, the priming, the padding, the length
fn itunes_priming_and_length(file: &[u8]) -> (u64, u64) {
    let tag = file
        .windows(8)
        .position(|window| window == b"iTunSMPB")
        .expect("the fixture has the tag");
    let kind_at = tag
        + file[tag..]
            .windows(4)
            .position(|w| w == b"data")
            .expect("its data box");
    let box_start = kind_at - 4;
    let box_len = u32::from_be_bytes(file[box_start..kind_at].try_into().expect("a size")) as usize;
    // the data box's type and locale words come before the text
    let text = &file[kind_at + 4 + 8..box_start + box_len];
    let fields: Vec<u64> = std::str::from_utf8(text)
        .expect("utf-8")
        .split_whitespace()
        .map(|field| u64::from_str_radix(field, 16).expect("hex"))
        .collect();
    (fields[1], fields[3])
}

#[tokio::test]
async fn itunes_gapless_tag_drops_the_aac_priming_and_padding() {
    let file = std::fs::read(fixture_path(ITUNES_FIXTURE)).expect("the fixture is checked in");
    let (priming, length) = itunes_priming_and_length(&file);

    let mut demux = Mp4Demux::new();
    demux
        .set_property("stream", PropValue::Str("aac".into()))
        .expect("stream=aac");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4,
        })
        .expect("configure");
    let demuxed = run(&mut demux, vec![data_frame(&file)]).await;
    let segment = the_segment(&demuxed);

    let whole = decode(AudioFormat::Aac, without_segments(&demuxed)).await;
    let trimmed = decode(AudioFormat::Aac, demuxed).await;
    let kept_bytes = sample_bytes(&whole, priming..priming + length);
    assert!(kept_bytes.end < whole.pcm.len(), "the padding is cut");
    assert_eq!(trimmed.pcm.len(), kept_bytes.len(), "{segment:?}");
    assert!(
        trimmed.pcm == whole.pcm[kept_bytes],
        "the decode drops exactly the priming and padding samples"
    );
}

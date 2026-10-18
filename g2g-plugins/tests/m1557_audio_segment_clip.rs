//! M1557: an audio decoder trims its PCM to the upstream segment, sample-accurate
//! at both edges, and `qtdemux` sends an AAC track's edit list as that segment, so
//! the priming goes the way ffmpeg drops it.
#![cfg(all(target_os = "linux", feature = "ffmpeg"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use std::path::PathBuf;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::time::{ns_to_ticks, ticks_to_ns};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, PropValue, Segment, ANY_CHANNELS,
};
use g2g_plugins::ffmpegaudiodec::FfmpegAudioDec;
use g2g_plugins::mp4demux::Mp4Demux;

use collect_sink::CollectSink;

const FIXTURE: &str = "aac_stereo_44100_adts.aac";
// audio only, its elst starting the presentation past the AAC priming
const EDIT_LIST_FIXTURE: &str = "audio_aac_48000.m4a";
const BYTES_PER_SAMPLE: usize = 2;
// inside the first 1024-sample AAC frame
const SEGMENT_START_SAMPLE: u64 = 300;
// inside a later frame, so the frames past it are dropped
const SEGMENT_STOP_SAMPLE: u64 = 5_000;

struct Decoded {
    pcm: Vec<u8>,
    first_pts_ns: u64,
    channels: usize,
    sample_rate: u32,
}

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(path).expect("the fixture is checked in")
}

fn data_frame(data: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

async fn decode(packets: Vec<PipelinePacket>) -> Decoded {
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
    for packet in packets {
        decoder.process(packet, &mut sink).await.expect("decode");
    }
    decoder
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");

    let (channels, sample_rate) = sink
        .packets
        .iter()
        .find_map(|packet| match packet {
            PipelinePacket::CapsChanged(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels,
                sample_rate,
                ..
            }) => Some((usize::from(*channels), *sample_rate)),
            _ => None,
        })
        .expect("the decoder announced its PCM caps");
    let frames: Vec<&Frame> = sink
        .packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .collect();
    let pcm = frames
        .iter()
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
        first_pts_ns: frames.first().expect("decoded audio").timing.pts_ns,
        channels,
        sample_rate,
    }
}

#[tokio::test]
async fn ffmpeg_audio_decoder_clips_to_the_segment() {
    // the whole ADTS file goes in as one buffer, so one packet decodes to many frames
    let adts = || data_frame(fixture(FIXTURE));
    let whole = decode(vec![adts()]).await;
    let rate = whole.sample_rate;
    let segment = Segment {
        start: ticks_to_ns(SEGMENT_START_SAMPLE, rate),
        stop: Some(ticks_to_ns(SEGMENT_STOP_SAMPLE, rate)),
        ..Segment::new()
    };
    let clipped = decode(vec![PipelinePacket::Segment(segment), adts()]).await;

    let bytes_per_sample_frame = whole.channels * BYTES_PER_SAMPLE;
    let kept_samples = (SEGMENT_STOP_SAMPLE - SEGMENT_START_SAMPLE) as usize;
    assert!(
        whole.pcm.len() > SEGMENT_STOP_SAMPLE as usize * bytes_per_sample_frame,
        "the fixture runs past the segment stop"
    );
    assert_eq!(
        clipped.pcm.len() / bytes_per_sample_frame,
        kept_samples,
        "only the samples inside the segment are left"
    );
    assert_eq!(clipped.first_pts_ns, segment.start);
    let start_byte = SEGMENT_START_SAMPLE as usize * bytes_per_sample_frame;
    let stop_byte = SEGMENT_STOP_SAMPLE as usize * bytes_per_sample_frame;
    assert!(
        clipped.pcm == whole.pcm[start_byte..stop_byte],
        "the kept samples are the unclipped decode's samples in the segment"
    );
}

// the first edit's media_time, in media timescale ticks
fn be32_after(file: &[u8], kind: &[u8; 4], offset: usize) -> u64 {
    let at = file
        .windows(4)
        .position(|window| window == kind)
        .expect("the fixture has the box")
        + kind.len()
        + offset;
    u64::from(u32::from_be_bytes(file[at..at + 4].try_into().unwrap()))
}

// version and flags, entry count, then the first entry's segment duration and media time
fn elst_media_time(file: &[u8]) -> u64 {
    be32_after(file, b"elst", 4 + 4 + 4)
}

fn elst_segment_duration(file: &[u8]) -> u64 {
    be32_after(file, b"elst", 4 + 4)
}

// version and flags, creation and modification times, then the timescale
fn mvhd_timescale(file: &[u8]) -> u64 {
    be32_after(file, b"mvhd", 4 + 4 + 4)
}

async fn demux_audio(file: Vec<u8>) -> Vec<PipelinePacket> {
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

#[tokio::test]
async fn mp4demux_edit_list_drops_the_aac_priming() {
    let file = fixture(EDIT_LIST_FIXTURE);
    let priming = elst_media_time(&file);
    let edit_duration_movie_ticks = elst_segment_duration(&file);
    let movie_timescale = mvhd_timescale(&file);
    let demuxed = demux_audio(file).await;
    let segment = demuxed
        .iter()
        .find_map(|packet| match packet {
            PipelinePacket::Segment(segment) => Some(*segment),
            _ => None,
        })
        .expect("the demuxer sends the edit list as a segment");
    let unsegmented = demuxed
        .iter()
        .filter(|packet| !matches!(packet, PipelinePacket::Segment(_)))
        .map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => data_frame(
                frame
                    .domain
                    .require_system_slice("test")
                    .expect("adts")
                    .to_vec(),
            ),
            PipelinePacket::CapsChanged(caps) => PipelinePacket::CapsChanged(caps.clone()),
            other => panic!("unexpected demuxer output {other:?}"),
        })
        .collect();
    let whole = decode(unsegmented).await;
    let trimmed = decode(demuxed).await;

    let rate = whole.sample_rate;
    assert_eq!(ns_to_ticks(segment.start, rate), priming);
    assert_eq!(trimmed.first_pts_ns, segment.start);
    let frame_bytes = whole.channels * BYTES_PER_SAMPLE;
    let priming_bytes = priming as usize * frame_bytes;
    let edit_samples = edit_duration_movie_ticks * u64::from(rate) / movie_timescale;
    let edit_end_bytes = (priming + edit_samples) as usize * frame_bytes;
    assert!(
        edit_end_bytes < whole.pcm.len(),
        "the edit cuts the padding"
    );
    assert!(
        trimmed.pcm == whole.pcm[priming_bytes..edit_end_bytes],
        "the decode drops exactly the priming and padding samples"
    );
}

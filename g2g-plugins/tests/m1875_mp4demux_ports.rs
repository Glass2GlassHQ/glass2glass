#![cfg(feature = "std")]

use core::task::{Context, Poll};

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, G2gError, MultiOutputElement, MultiOutputSink,
    PushOutcome, Segment,
};
use g2g_plugins::mp4demux::Mp4Demux;
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

// ffmpeg -f lavfi -i testsrc=size=64x48:rate=25:duration=3 -f lavfi -i sine=frequency=440:duration=3
//   -c:v libx264 -g 10 -bf 2 -c:a aac -movflags frag_keyframe+empty_moov+separate_moof
// one moof per track, the video one first, and its video starts two frames late
const FRAGMENT_PER_TRACK: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_aac_fragment_per_track.mp4"
);
// the same with +delay_moov, which writes an edit list per track
const FRAGMENT_PER_TRACK_EDIT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_aac_fragment_per_track_edit.mp4"
);
// the same with -movflags frag_keyframe alone: the moov holds the first samples
const FRAG_KEYFRAME: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_aac_frag_keyframe.mp4"
);
const NANOS_PER_SECOND: u64 = 1_000_000_000;
const FRAME_RATE: u64 = 25;
// x264 with -bf 2 delays presentation by two frames
const B_FRAME_DELAY_NS: u64 = 2 * NANOS_PER_SECOND / FRAME_RATE;
// ffmpeg's AAC encoder primes 1024 samples, at the sine source's 44100 Hz
const AAC_PRIMING_NS: u64 = 1024 * NANOS_PER_SECOND / 44_100;
const READ_CHUNK_LEN: usize = 4096;

fn file_chunks(file: &[u8]) -> impl Iterator<Item = PipelinePacket> + '_ {
    file.chunks(READ_CHUNK_LEN).map(|chunk| {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        ))
    })
}

struct PortSegments(Vec<Vec<Segment>>);

impl MultiOutputSink for PortSegments {
    fn poll_push_to(
        &mut self,
        _cx: &mut Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::Segment(segment)) = packet_slot.take() {
            self.0[port].push(segment);
        }
        Poll::Ready(Ok(PushOutcome::Accepted))
    }

    fn port_count(&self) -> usize {
        self.0.len()
    }
}

// every track of `path` on its own port, as the playbin fan-out builds it, with each port's segments
async fn segments_per_track(path: &str) -> Vec<Vec<Segment>> {
    let file = std::fs::read(path).expect("fixture reads");
    let streams = forwardable_streams(&file);
    let mut demux = Mp4DemuxN::new(
        streams
            .iter()
            .map(|stream| Mp4Port {
                track_id: stream.track_id,
                caps: stream.caps.clone(),
            })
            .collect(),
    );
    MultiOutputElement::configure_pipeline(
        &mut demux,
        &Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        },
    )
    .expect("configure");
    let mut sink = PortSegments(vec![Vec::new(); streams.len()]);
    for packet in file_chunks(&file).chain([PipelinePacket::Eos]) {
        MultiOutputElement::process(&mut demux, packet, &mut sink)
            .await
            .expect("demux");
    }
    assert!(streams[0].video && !streams[1].video, "video, then audio");
    sink.0
}

fn starts(segments: &[Segment]) -> Vec<u64> {
    segments.iter().map(|segment| segment.start).collect()
}

#[tokio::test]
async fn each_fragmented_track_opens_at_its_own_edit() {
    let segments = segments_per_track(FRAGMENT_PER_TRACK_EDIT).await;
    assert_eq!(starts(&segments[0]), [B_FRAME_DELAY_NS]);
    let audio = starts(&segments[1]);
    assert_eq!(audio.len(), 1);
    assert!(
        audio[0].abs_diff(AAC_PRIMING_NS) <= 1,
        "the audio segment starts after its priming, not at the video's start: {audio:?}"
    );
}

#[tokio::test]
async fn audio_that_starts_before_the_first_video_fragment_is_kept() {
    let segments = segments_per_track(FRAGMENT_PER_TRACK).await;
    for port in &segments {
        assert!(
            starts(port).iter().all(|&start| start == 0),
            "the audio starts at 0, so no port is cut to the video's late start: {segments:?}"
        );
    }
}

#[tokio::test]
async fn a_flush_keeps_the_stream_start_of_a_reread_file() {
    let file = std::fs::read(FRAGMENT_PER_TRACK).expect("fixture reads");
    let mut demux = Mp4Demux::new();
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4,
        })
        .expect("configure");
    let mut sink = CollectSink::default();
    let packets = file_chunks(&file)
        .chain([
            PipelinePacket::Flush,
            PipelinePacket::Segment(Segment::new()),
        ])
        .chain(file_chunks(&file))
        .chain([PipelinePacket::Eos]);
    for packet in packets {
        demux.process(packet, &mut sink).await.expect("demux");
    }
    let segment_starts: Vec<u64> = sink
        .packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::Segment(segment) => Some(segment.start),
            _ => None,
        })
        .collect();
    assert_eq!(
        segment_starts,
        [B_FRAME_DELAY_NS, B_FRAME_DELAY_NS],
        "the lone video track opens at its first picture, before and after the flush"
    );
}

#[tokio::test]
async fn samples_outside_the_fragments_fail_the_run() {
    let file = std::fs::read(FRAG_KEYFRAME).expect("fixture reads");
    let mut demux = Mp4Demux::new();
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4,
        })
        .expect("configure");
    let mut sink = CollectSink::default();
    let mut ran = Ok(());
    for packet in file_chunks(&file).chain([PipelinePacket::Eos]) {
        ran = demux.process(packet, &mut sink).await;
        if ran.is_err() {
            break;
        }
    }
    assert_eq!(ran, Err(G2gError::CapsMismatch));
}

// a selected track whose ContentEncoding the demuxer cannot undo fails the stream before any of its stored bytes go out

#![cfg(feature = "std")]

use std::path::Path;

use g2g_core::fanout::MultiOutputSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, G2gError, MultiOutputElement, OutputSink, PushOutcome,
};
use g2g_plugins::matroska::MatroskaDemuxer;
use g2g_plugins::mkvdemux::{
    forwardable_streams, subtitle_streams, MkvDemux, MkvDemuxN, MkvStream,
};

#[derive(Default)]
struct FrameCount(usize);

impl FrameCount {
    fn count(&mut self, packet: PipelinePacket) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if matches!(packet, PipelinePacket::DataFrame(_)) {
            self.0 += 1;
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl OutputSink for FrameCount {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.count(packet_slot.take().expect("poll_push without a packet"))
    }
}

impl MultiOutputSink for FrameCount {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        _port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.count(packet_slot.take().expect("poll_push without a packet"))
    }

    fn port_count(&self) -> usize {
        1
    }
}

fn fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(path).expect("the fixture is committed")
}

fn file_packet(name: &str) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(fixture(name).into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

fn parsed(name: &str) -> MatroskaDemuxer {
    let mut demux = MatroskaDemuxer::new();
    demux.push_data(&fixture(name));
    demux
}

const MATROSKA: Caps = Caps::ByteStream {
    encoding: ByteStreamEncoding::Matroska,
};

async fn assert_both_demuxers_refuse(name: &str, stream: MkvStream) {
    let mut single = MkvDemux::new().with_stream(stream);
    single.configure_pipeline(&MATROSKA).expect("configure");
    let mut frames = FrameCount::default();
    let result = single.process(file_packet(name), &mut frames).await;
    assert_eq!(result, Err(G2gError::CapsMismatch), "MkvDemux on {name}");
    assert_eq!(frames.0, 0, "MkvDemux on {name} emits no frame");

    let mut multi = MkvDemuxN::new(vec![stream]);
    multi.configure_pipeline(&MATROSKA).expect("configure");
    let mut frames = FrameCount::default();
    let result = multi.process(file_packet(name), &mut frames).await;
    assert_eq!(result, Err(G2gError::CapsMismatch), "MkvDemuxN on {name}");
    assert_eq!(frames.0, 0, "MkvDemuxN on {name} emits no frame");
}

// no key is supplied, which flags the track with or without matroska-decrypt
#[tokio::test]
async fn an_encrypted_vp9_track_with_no_key_fails_the_stream() {
    assert_both_demuxers_refuse("webm_vp9_encrypted.webm", MkvStream::Vp9).await;
}

#[cfg(not(feature = "matroska-decrypt"))]
#[tokio::test]
async fn an_encrypted_opus_track_fails_the_stream_without_matroska_decrypt() {
    assert_both_demuxers_refuse("webm_opus_encrypted.webm", MkvStream::Opus).await;
}

// playbin and the fallbacksrc fan-out build their demuxer ports from these lists
#[test]
fn flagged_tracks_are_left_out_of_the_automatic_stream_lists() {
    assert!(!forwardable_streams(&parsed("av1_321x181_24fps.mkv")).is_empty());
    assert!(forwardable_streams(&parsed("webm_vp9_encrypted.webm")).is_empty());
    assert!(!subtitle_streams(&parsed("mkv_ass_plain.mks")).is_empty());
    #[cfg(not(feature = "matroska-compression"))]
    assert!(subtitle_streams(&parsed("mkv_ass_bzip2.mks")).is_empty());
}

#[cfg(not(feature = "matroska-compression"))]
#[tokio::test]
async fn a_bzip2_track_fails_the_stream_without_matroska_compression() {
    assert_both_demuxers_refuse(
        "mkv_ass_bzip2.mks",
        MkvStream::Subtitle(g2g_core::TextFormat::Ssa),
    )
    .await;
}

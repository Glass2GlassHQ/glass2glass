#![cfg(feature = "std")]

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, G2gError, OutputSink, PropValue, PushOutcome, Rate,
};
use g2g_plugins::mkvdemux::MkvDemux;
use g2g_plugins::mp4demux::Mp4Demux;

// av1_321x181.obu stream-copied by `ffmpeg -framerate 24`
const FIXTURE_FPS: u32 = 24;
const MP4_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av1_321x181_24fps.mp4"
);
const MKV_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av1_321x181_24fps.mkv"
);
const READ_CHUNK_LEN: usize = 4096;

#[derive(Default)]
struct CapsSink {
    caps: Vec<Caps>,
}

impl OutputSink for CapsSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::CapsChanged(caps)) = packet_slot.take() {
            self.caps.push(caps);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

async fn demuxed_framerates(
    demux: &mut impl AsyncElement,
    encoding: ByteStreamEncoding,
    path: &str,
) -> Vec<Rate> {
    demux
        .configure_pipeline(&Caps::ByteStream { encoding })
        .expect("configure");
    let mut sink = CapsSink::default();
    let file = std::fs::read(path).expect("fixture reads");
    for chunk in file.chunks(READ_CHUNK_LEN) {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        demux
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .expect("demux");
    }
    demux
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    sink.caps
        .into_iter()
        .filter_map(|caps| match caps {
            Caps::CompressedVideo { framerate, .. } => Some(framerate),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn mp4demux_reads_the_rate_of_a_constant_sample_duration() {
    let rates =
        demuxed_framerates(&mut Mp4Demux::new(), ByteStreamEncoding::Mp4, MP4_FIXTURE).await;
    assert_eq!(rates, vec![Rate::Fixed(FIXTURE_FPS << 16)]);
}

#[tokio::test]
async fn mkvdemux_reads_the_rate_of_the_default_duration() {
    let mut demux = MkvDemux::new();
    demux
        .set_property("stream", PropValue::Str("av1".into()))
        .expect("stream property");
    let rates = demuxed_framerates(&mut demux, ByteStreamEncoding::Matroska, MKV_FIXTURE).await;
    assert_eq!(rates, vec![Rate::Fixed(FIXTURE_FPS << 16)]);
}

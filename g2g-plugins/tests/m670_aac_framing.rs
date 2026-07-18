//! M670 `AacParse` validated against real ffmpeg-encoded AAC in both framings:
//! ADTS (elementary stream) and LOAS/LATM (broadcast). Each fixture is a short
//! sine encoded by ffmpeg; the parser must recover the channel count and sample
//! rate and emit a `CapsChanged` before forwarding the frame.

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, G2gError, MemoryDomain, OutputSink, PipelinePacket,
};
use g2g_plugins::aacparse::AacParse;

const ADTS: &[u8] = include_bytes!("fixtures/aac_stereo_44100.adts");
const LATM: &[u8] = include_bytes!("fixtures/aac_stereo_48000.latm");

#[derive(Default)]
struct Collect {
    caps: Vec<(u8, u32)>,
    data_frames: usize,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(Caps::Audio {
                    channels,
                    sample_rate,
                    ..
                }) => {
                    self.caps.push((channels, sample_rate));
                }
                PipelinePacket::DataFrame(_) => self.data_frames += 1,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

async fn refine(stream: &[u8]) -> Collect {
    let mut parse = AacParse::new();
    let sentinel = Caps::Audio {
        format: AudioFormat::Aac,
        channels: 0,
        sample_rate: 0,
    };
    parse.configure_pipeline(&sentinel).expect("configures");
    let mut sink = Collect::default();
    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(stream.to_vec().into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };
    parse
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("process");
    sink
}

#[tokio::test]
async fn adts_stream_refines_to_stereo_44100() {
    let sink = refine(ADTS).await;
    assert_eq!(
        sink.caps,
        vec![(2, 44_100)],
        "real ADTS refined to stereo/44100"
    );
    assert_eq!(sink.data_frames, 1, "the frame is forwarded after the caps");
}

#[tokio::test]
async fn loas_latm_stream_refines_to_stereo_48000() {
    let sink = refine(LATM).await;
    assert_eq!(
        sink.caps,
        vec![(2, 48_000)],
        "real LOAS/LATM refined to stereo/48000"
    );
    assert_eq!(sink.data_frames, 1);
}

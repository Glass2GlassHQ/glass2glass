//! M670 `AacParse` validated against real ffmpeg-encoded AAC in both framings:
//! ADTS (elementary stream) and LOAS/LATM (broadcast). Each fixture is a short
//! sine encoded by ffmpeg; the parser must recover the channel count and sample
//! rate and emit a `CapsChanged` before the frames it describes. The ADTS
//! fixture is also split into one access unit per buffer (M1074), so its frame
//! count is the number of ADTS headers in the file; the LATM stream is framed by
//! its container already and is forwarded buffer for buffer.

use g2g_core::element::PushOutcome;
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
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
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
    parse
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    sink
}

/// Walk the ADTS headers of `stream` and count the access units, the framing the
/// parser has to reproduce. The 13-bit `aac_frame_length` spans bytes 3..6.
fn adts_access_units(stream: &[u8]) -> usize {
    const ADTS_HEADER_LEN: usize = 7;
    let mut units = 0;
    let mut pos = 0;
    while pos + ADTS_HEADER_LEN <= stream.len() {
        assert_eq!(
            (stream[pos], stream[pos + 1] & 0xF6),
            (0xFF, 0xF0),
            "the fixture is back-to-back ADTS"
        );
        let len = (((stream[pos + 3] & 0x03) as usize) << 11)
            | ((stream[pos + 4] as usize) << 3)
            | ((stream[pos + 5] >> 5) as usize);
        if pos + len > stream.len() {
            break;
        }
        units += 1;
        pos += len;
    }
    units
}

#[tokio::test]
async fn adts_stream_refines_to_stereo_44100() {
    let sink = refine(ADTS).await;
    assert_eq!(
        sink.caps,
        vec![(2, 44_100)],
        "real ADTS refined to stereo/44100"
    );
    assert_eq!(
        sink.data_frames,
        adts_access_units(ADTS),
        "one buffer per ADTS access unit"
    );
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

//! M300 VP9-in-MP4: `Mp4MuxN` muxes a VP9 video track and an AAC audio track
//! into one ISO-BMFF byte stream. VP9 frames are stored verbatim (no AVCC
//! reframing) behind a `vp09` VisualSampleEntry + `vpcC` VPCodecConfigurationBox;
//! the keyframe flag comes from the VP9 uncompressed frame header. Drives the
//! element directly with synthetic VP9 frames + ADTS AAC, and asserts the moov
//! carries the VP9 sample entry (no avcC). A real `vp9enc` end-to-end run needs
//! libvpx, so the codec config is also checked structurally by ffprobe out of
//! band.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, Caps, Dim, G2gError, MultiInputElement, OutputSink, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::mp4muxn::Mp4MuxN;

fn vp9_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::Vp9,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn aac_caps() -> Caps {
    Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 48000 }
}

#[derive(Default)]
struct CaptureSink {
    bytes: Vec<u8>,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    self.bytes.extend_from_slice(s.as_slice());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
        0,
    ))
}

/// A VP9 frame whose uncompressed header byte marks it a key frame (marker 0b10,
/// profile 0, show_existing 0, frame_type 0), then arbitrary payload.
fn vp9_key() -> Vec<u8> {
    vec![0x80, 0x49, 0x83, 0x42, 0x00, 0x11, 0x22]
}

/// A minimal ADTS AAC access unit (7-byte header + payload) at 48 kHz stereo.
fn adts_au(payload: &[u8]) -> Vec<u8> {
    let frame_len = payload.len() + 7;
    let mut au = vec![
        0xFF,
        0xF1,
        (1 << 6) | (3 << 2),
        ((2 & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F,
        0xFC,
    ];
    au.extend_from_slice(payload);
    au
}

fn count(haystack: &[u8], needle: &[u8]) -> usize {
    haystack.windows(needle.len()).filter(|w| *w == needle).count()
}

#[tokio::test]
async fn muxes_vp9_and_aac_into_one_mp4() {
    let mut mux = Mp4MuxN::new(2);
    mux.configure_pipeline(0, &vp9_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = CaptureSink::default();

    mux.process(0, frame(vp9_key(), 0), &mut sink).await.unwrap();
    mux.process(1, frame(adts_au(&[0x01, 0x02, 0x03]), 0), &mut sink).await.unwrap();
    mux.process(0, frame(vp9_key(), 33_000_000), &mut sink).await.unwrap();
    mux.process(1, frame(adts_au(&[0x04, 0x05]), 21_000_000), &mut sink).await.unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink).await.unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink).await.unwrap();

    let out = &sink.bytes;
    assert_eq!(&out[4..8], b"ftyp", "ISO-BMFF starts with ftyp");
    assert_eq!(count(out, b"trak"), 2, "two tracks (video + audio)");
    assert_eq!(count(out, b"vp09"), 1, "VP9 sample entry");
    assert_eq!(count(out, b"vpcC"), 1, "VPCodecConfigurationBox");
    assert_eq!(count(out, b"avcC"), 0, "no avcC: VP9 is not H.264");
    assert_eq!(count(out, b"esds"), 1, "the AAC track still carries its esds");
    assert_eq!(mux.emitted(), 4, "all four access units muxed");
}

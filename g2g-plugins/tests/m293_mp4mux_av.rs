//! M293 A/V fragmented-MP4 muxer: `Mp4MuxN` muxes an H.264 video track and an
//! AAC audio track into one ISO-BMFF byte stream, interleaving access units by
//! PTS. Drives the multi-input element directly with synthetic video AUs (SPS +
//! PPS + IDR, then P slices) and ADTS AAC AUs, and asserts the muxed stream is a
//! well-formed two-track fMP4 (ftyp + a moov carrying both an `avcC` video and an
//! `esds` audio sample entry, then interleaved `moof`/`mdat` fragments
//! referencing both tracks). End-to-end playability is covered by the manual
//! `videotestsrc ! x264enc ! mp4mux. audiotestsrc ! avenc_aac ! m.` ffprobe check.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, Caps, Dim, G2gError, MultiInputElement, OutputSink, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::mp4muxn::Mp4MuxN;

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
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
    frames: u64,
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
                    self.frames += 1;
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

fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}

/// A minimal ADTS AAC access unit (7-byte header + payload) at 48 kHz stereo.
fn adts_au(payload: &[u8]) -> Vec<u8> {
    let frame_len = payload.len() + 7;
    let sr_index = 3u8; // 48000
    let channels = 2u8;
    let mut au = vec![
        0xFF,
        0xF1,
        (1 << 6) | (sr_index << 2) | ((channels >> 2) & 1),
        ((channels & 3) << 6) | ((frame_len >> 11) & 3) as u8,
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
async fn muxes_two_tracks_into_one_iso_bmff_stream() {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];

    let mut mux = Mp4MuxN::new(2);
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = CaptureSink::default();

    // Interleave: video at 0/33ms, audio at 0/21/42ms. The PTS-ordered merge
    // releases nothing until both inputs have queued an AU, then drains in order.
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink).await.unwrap();
    mux.process(1, frame(adts_au(&[0x01, 0x02, 0x03]), 0), &mut sink).await.unwrap();
    mux.process(1, frame(adts_au(&[0x04, 0x05]), 21_000_000), &mut sink).await.unwrap();
    mux.process(0, frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000), &mut sink).await.unwrap();
    mux.process(1, frame(adts_au(&[0x06, 0x07]), 42_000_000), &mut sink).await.unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink).await.unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink).await.unwrap();

    let out = &sink.bytes;
    assert_eq!(&out[4..8], b"ftyp", "ISO-BMFF starts with ftyp");
    // The moov carries both tracks.
    assert_eq!(count(out, b"trak"), 2, "two trak boxes (video + audio)");
    assert_eq!(count(out, b"trex"), 2, "two trex (one per track)");
    assert_eq!(count(out, b"avcC"), 1, "H.264 video sample entry");
    assert_eq!(count(out, b"esds"), 1, "AAC audio sample entry");
    assert_eq!(count(out, b"vide"), 1, "a video handler");
    assert_eq!(count(out, b"soun"), 1, "a sound handler");
    // Every access unit became a moof+mdat fragment (3 audio + 2 video).
    assert_eq!(count(out, b"moof"), 5, "one moof per access unit");
    assert_eq!(count(out, b"mdat"), 5, "one mdat per access unit");
    assert_eq!(mux.emitted(), 5, "five byte-stream frames forwarded");
}

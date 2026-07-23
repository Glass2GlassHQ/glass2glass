//! M294 A/V Matroska muxer: `MkvMuxN` muxes an H.264 video track and an AAC
//! audio track into one Matroska byte stream, interleaving access units by PTS.
//! Drives the multi-input element directly with synthetic video AUs (SPS + PPS +
//! IDR, then a P slice) and ADTS AAC AUs, then demuxes the result back through
//! `MatroskaDemuxer` and asserts both tracks and their frames are recovered, with
//! the per-track `CodecPrivate` (avcC record / AAC ASC) written. End-to-end
//! playability is covered by the manual `videotestsrc ! x264enc ! matroskamux.
//! audiotestsrc ! avenc_aac ! m.` ffprobe check.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, Caps, Dim, G2gError, MultiInputElement, OutputSink, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::matroska::{MatroskaDemuxer, MkvCodec};
use g2g_plugins::mkvmuxn::MkvMuxN;

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn aac_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: 48000,
    }
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
                if let Some(s) = f.domain.as_system_slice() {
                    self.bytes.extend_from_slice(s);
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
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
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
    haystack
        .windows(needle.len())
        .filter(|w| *w == needle)
        .count()
}

#[tokio::test]
async fn muxes_two_tracks_into_one_matroska_stream() {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];

    let mut mux = MkvMuxN::new(2);
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = CaptureSink::default();

    // Interleave: video at 0/33ms, audio at 0/21/42ms. The PTS-ordered merge
    // releases nothing until both inputs have queued an AU, then drains in order.
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0x01, 0x02, 0x03]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0x04, 0x05]), 21_000_000), &mut sink)
        .await
        .unwrap();
    mux.process(
        0,
        frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000),
        &mut sink,
    )
    .await
    .unwrap();
    mux.process(1, frame(adts_au(&[0x06, 0x07]), 42_000_000), &mut sink)
        .await
        .unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();

    let out = &sink.bytes;
    // A Matroska stream: EBML header + Segment + Tracks, with both CodecIDs and a
    // CodecPrivate per track (0x63A2).
    assert!(
        count(out, b"matroska") >= 1,
        "matroska DocType (H.264 is not WebM)"
    );
    assert!(count(out, b"V_MPEG4/ISO/AVC") == 1, "H.264 CodecID");
    assert!(count(out, b"A_AAC") == 1, "AAC CodecID");
    assert!(
        count(out, &[0x63, 0xA2]) >= 2,
        "a CodecPrivate element per track"
    );
    assert!(mux.emitted() >= 5, "five access units muxed");

    // Demux the produced Matroska back: two tracks recovered with their params.
    let mut d = MatroskaDemuxer::new();
    d.push_data(out);
    let tracks = d.tracks();
    assert_eq!(tracks.len(), 2, "video + audio tracks announced");
    assert_eq!(tracks[0].number, 1);
    assert_eq!(tracks[0].codec, MkvCodec::H264);
    assert_eq!((tracks[0].width, tracks[0].height), (320, 240));
    assert_eq!(tracks[1].number, 2);
    assert_eq!(tracks[1].codec, MkvCodec::Aac);
    assert_eq!(tracks[1].channels, 2);
    assert_eq!(tracks[1].sample_rate, 48_000);

    // Frames come back split by track, in PTS order, the audio de-ADTS'd to its
    // 2/3-byte payloads and the video AVCC length-prefixed.
    let frames = d.take_frames();
    let video: Vec<_> = frames.iter().filter(|f| f.track == 1).collect();
    let audio: Vec<_> = frames.iter().filter(|f| f.track == 2).collect();
    assert_eq!(video.len(), 2, "two video frames");
    assert_eq!(audio.len(), 3, "three audio frames");
    // The first audio payload was [0x01,0x02,0x03] behind a 7-byte ADTS header;
    // the muxer strips ADTS, so the recovered frame is exactly those 3 bytes.
    assert_eq!(
        audio[0].data,
        vec![0x01, 0x02, 0x03],
        "AAC payload, ADTS stripped"
    );
    assert_eq!(audio[1].data, vec![0x04, 0x05]);
    // The IDR video frame is AVCC: each NALU 4-byte length-prefixed, so it is
    // longer than the raw NALU bytes and carries no Annex-B start code.
    assert!(video[0].keyframe, "the IDR is a keyframe");
    assert!(!video[1].keyframe, "the P slice is not");
    assert!(
        count(&video[0].data, &[0, 0, 0, 1]) == 0,
        "no Annex-B start codes (AVCC framing)"
    );
}

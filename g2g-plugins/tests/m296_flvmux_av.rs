//! M296 A/V FLV muxer: `FlvMuxN` muxes an H.264 video track and an AAC audio
//! track into one FLV byte stream, interleaving access units by PTS and writing
//! the decoder-config sequence headers up front. Drives the multi-input element
//! directly with synthetic video AUs (SPS + PPS + IDR, then a P slice) and ADTS
//! AAC AUs, then demuxes the result back through `FlvDemuxer` and asserts both
//! tracks' media frames are recovered. End-to-end playability is covered by the
//! manual `videotestsrc ! x264enc ! flvmux. audiotestsrc ! avenc_aac ! m.`
//! ffprobe check.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, Caps, Dim, G2gError, MultiInputElement, OutputSink, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::flv::{FlvDemuxer, FlvTrack};
use g2g_plugins::flvmuxn::FlvMuxN;

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
async fn muxes_two_tracks_into_one_flv_stream() {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];

    let mut mux = FlvMuxN::new(2);
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = CaptureSink::default();

    // Interleave: video at 0/33ms (IDR then a P slice), audio at 0/21/42ms.
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
    // An FLV stream with both tracks present (flags bit0 video | bit2 audio).
    assert_eq!(&out[0..3], b"FLV", "starts with the FLV signature");
    assert_eq!(out[4], 0x05, "audio + video present flag");
    // The decoder-config sequence headers are written up front: a video config
    // tag body begins 0x17,0x00 and an audio config tag body 0xAF,0x00.
    assert!(
        count(out, &[0x17, 0x00, 0x00, 0x00, 0x00]) >= 1,
        "AVC sequence header"
    );
    assert!(count(out, &[0xAF, 0x00]) >= 1, "AAC sequence header");
    assert_eq!(mux.emitted(), 5, "five media frames muxed");

    // Demux back: the FLV demuxer skips the sequence headers and recovers the
    // media access units, the audio de-ADTS'd and the video AVCC-framed.
    let mut d = FlvDemuxer::new();
    d.push_data(out);
    let units = d.take_units();
    let video: Vec<_> = units
        .iter()
        .filter(|u| u.track == FlvTrack::Video)
        .collect();
    let audio: Vec<_> = units
        .iter()
        .filter(|u| u.track == FlvTrack::Audio)
        .collect();
    assert_eq!(video.len(), 2, "two video media frames");
    assert_eq!(audio.len(), 3, "three audio media frames");
    assert_eq!(
        audio[0].data,
        vec![0x01, 0x02, 0x03],
        "AAC payload, ADTS stripped"
    );
    assert_eq!(audio[0].pts_ms, 0);
    assert_eq!(audio[1].pts_ms, 21);
    // The IDR video AU is AVCC length-prefixed, so it carries no Annex-B start code.
    assert!(
        count(&video[0].data, &[0, 0, 0, 1]) == 0,
        "AVCC framing (no start codes)"
    );
}

//! M377 - app-driven stream selection, the playbin SELECT_STREAMS analog. After
//! a demuxer announces its streams (M376), the app names a stream id to forward
//! via a `StreamSelectController`; the demuxer switches its single output to that
//! track, re-negotiates caps, and confirms the active id on the bus
//! (`BusMessage::StreamsSelected`).
//!
//! Mux an A/V Matroska stream (H.264 video + AAC audio) with `MkvMuxN`. Drive
//! `MkvDemux` (defaulting to the video track) header-first so it announces the
//! collection with no frames yet; then select the audio track and feed the
//! Clusters: the demuxer switches mid-stream, emitting a CapsChanged from video
//! to audio and forwarding the AAC payloads, not the video.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::StreamSelectController;
use g2g_core::{
    AudioFormat, Bus, BusMessage, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement,
    OutputSink, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};
use g2g_plugins::mkvmuxn::MkvMuxN;

const CLUSTER_ID: [u8; 4] = [0x1F, 0x43, 0xB6, 0x75];

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}
fn aac_caps() -> Caps {
    Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 48_000 }
}

#[derive(Default)]
struct Collect {
    bytes: Vec<u8>,
}
impl OutputSink for Collect {
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

/// Records the forwarded frame payloads and every CapsChanged caps, so the test
/// can see the output stream switch.
#[derive(Default)]
struct Tap {
    frames: Vec<Vec<u8>>,
    caps: Vec<Caps>,
}
impl OutputSink for Tap {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(Frame { domain: MemoryDomain::System(s), .. }) => {
                    self.frames.push(s.as_slice().to_vec());
                }
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                _ => {}
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
fn data_frame(bytes: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming::default(),
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

/// Mux an H.264 + AAC stream; the two AAC payloads are recognizable so the test
/// can confirm the audio track is what gets forwarded after the switch.
async fn mux_av() -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];

    let mut mux = MkvMuxN::new(2);
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = Collect::default();
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink).await.unwrap();
    mux.process(1, frame(adts_au(&[0xA1, 0xA2, 0xA3]), 0), &mut sink).await.unwrap();
    mux.process(0, frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000), &mut sink).await.unwrap();
    mux.process(1, frame(adts_au(&[0xB4, 0xB5]), 21_000_000), &mut sink).await.unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink).await.unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.bytes
}

#[tokio::test]
async fn mkvdemux_switches_its_output_to_the_selected_stream() {
    let file = mux_av().await;
    let first_cluster = file.windows(4).position(|w| w == CLUSTER_ID).unwrap();
    let (header, clusters) = file.split_at(first_cluster);

    let (bus, handle) = Bus::new(16);
    let select = StreamSelectController::new();
    // Default to the video track; selection will switch it to audio.
    let mut demux = MkvDemux::new()
        .with_stream(MkvStream::H264)
        .with_bus(handle)
        .with_stream_select(select.clone());
    demux
        .configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::Matroska })
        .expect("configure");

    let mut tap = Tap::default();

    // 1. Feed the header only: Tracks parses, the collection is announced, the
    // default (video) caps are emitted, but no Cluster has arrived so no frames.
    demux.process(data_frame(header), &mut tap).await.unwrap();
    assert!(
        tap.caps.iter().any(|c| matches!(c, Caps::CompressedVideo { codec: VideoCodec::H264, .. })),
        "default video caps emitted before selection"
    );
    assert!(tap.frames.is_empty(), "no frames before any Cluster");

    // 2. Discover the announced audio stream id and select it.
    let mut audio_id = None;
    while let Some(msg) = bus.try_recv() {
        if let BusMessage::StreamCollection(c) = msg {
            audio_id =
                c.streams_of_type(g2g_core::StreamType::Audio).next().map(|s| s.id.clone());
        }
    }
    let audio_id = audio_id.expect("the collection announced an audio stream");
    assert_eq!(audio_id, "matroska-track-2");
    select.select(vec![audio_id.clone()]);

    // 3. Feed the Clusters: the demuxer switches mid-stream to the audio track,
    // emits a CapsChanged from video to audio, and forwards the AAC payloads.
    demux.process(data_frame(clusters), &mut tap).await.unwrap();

    assert!(
        tap.caps.last().map(|c| matches!(c, Caps::Audio { format: AudioFormat::Aac, .. })).unwrap_or(false),
        "the switch re-negotiated caps to the audio stream"
    );
    assert_eq!(
        tap.frames,
        vec![vec![0xA1u8, 0xA2, 0xA3], vec![0xB4, 0xB5]],
        "the AAC payloads are forwarded (the audio track), not the video",
    );

    // 4. The selection is confirmed on the bus.
    let selected: Vec<_> = core::iter::from_fn(|| bus.try_recv())
        .filter_map(|m| match m {
            BusMessage::StreamsSelected { ids } => Some(ids),
            _ => None,
        })
        .collect();
    assert_eq!(selected, vec![vec![audio_id]], "StreamsSelected confirms the active id");
}

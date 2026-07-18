//! M378 - multi-output Matroska demuxer (the decodebin core). One Matroska byte
//! stream in, N elementary streams out, one per output port. `MkvDemuxN` parses
//! the container once and routes each track's access units to its port by codec,
//! emitting each port's concrete caps before its first frame. This is what lets a
//! single demuxer feed multiple decode branches (audio + video together) in one
//! pipeline, the playbin / decodebin model.
//!
//! Mux an A/V Matroska stream (H.264 video + AAC audio) with `MkvMuxN`, then demux
//! it with a two-port `MkvDemuxN` (port 0 = H.264, port 1 = AAC) and assert each
//! port receives its own CapsChanged and only its track's frames.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::fanout::MultiOutputSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, Bus, BusMessage, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement,
    MultiOutputElement, OutputSink, PushOutcome, Rate, StreamType, VideoCodec,
};
use g2g_plugins::mkvdemux::{MkvDemuxN, MkvStream};
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
        sample_rate: 48_000,
    }
}

/// A `MultiOutputSink` recording, per port, the forwarded frame payloads and the
/// caps of each CapsChanged.
#[derive(Default)]
struct PortTap {
    frames: Vec<Vec<Vec<u8>>>,
    caps: Vec<Vec<Caps>>,
}
impl PortTap {
    fn new(ports: usize) -> Self {
        Self {
            frames: vec![Vec::new(); ports],
            caps: vec![Vec::new(); ports],
        }
    }
}
impl MultiOutputSink for PortTap {
    fn push_to<'a>(
        &'a mut self,
        port: usize,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(s),
                    ..
                }) => {
                    self.frames[port].push(s.as_slice().to_vec());
                }
                PipelinePacket::CapsChanged(c) => self.caps[port].push(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
    fn port_count(&self) -> usize {
        self.frames.len()
    }
}

// --- A/V mux fixture (shared shape with m294 / m377) ---
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
async fn mux_av() -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let mut mux = MkvMuxN::new(2);
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = Collect::default();
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0xA1, 0xA2, 0xA3]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(
        0,
        frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000),
        &mut sink,
    )
    .await
    .unwrap();
    mux.process(1, frame(adts_au(&[0xB4, 0xB5]), 21_000_000), &mut sink)
        .await
        .unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    sink.bytes
}

fn data_frame(bytes: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

#[tokio::test]
async fn mkvdemuxn_splits_av_onto_two_ports() {
    let file = mux_av().await;

    let (bus, handle) = Bus::new(16);
    // Port 0 = video (H.264), port 1 = audio (AAC).
    let mut demux = MkvDemuxN::new(vec![MkvStream::H264, MkvStream::Aac]).with_bus(handle);
    assert_eq!(demux.port_count(), 2);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure");

    let mut tap = PortTap::new(2);
    demux.process(data_frame(&file), &mut tap).await.unwrap();

    // Each port announced its own elementary-stream caps once, ahead of frames.
    assert_eq!(tap.caps[0].len(), 1, "video port announced caps once");
    assert!(
        matches!(
            tap.caps[0][0],
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }
        ),
        "video port retypes to H.264 caps"
    );
    assert_eq!(tap.caps[1].len(), 1, "audio port announced caps once");
    assert!(
        matches!(
            tap.caps[1][0],
            Caps::Audio {
                format: AudioFormat::Aac,
                sample_rate: 48_000,
                ..
            }
        ),
        "audio port retypes to AAC caps"
    );

    // Port 1 gets exactly the AAC payloads (de-ADTS'd), in PTS order; the video
    // payloads land only on port 0. The single demuxer fed both branches.
    assert_eq!(
        tap.frames[1],
        vec![vec![0xA1u8, 0xA2, 0xA3], vec![0xB4, 0xB5]],
        "audio port carries only the AAC track"
    );
    assert_eq!(
        tap.frames[0].len(),
        2,
        "video port carries the two H.264 access units"
    );
    assert!(
        tap.frames[0].iter().all(|f| !f.is_empty()) && tap.frames[0] != tap.frames[1],
        "video frames are distinct from the audio frames"
    );

    // The collection was still announced (discovery works on the multi-output path).
    let announced = core::iter::from_fn(|| bus.try_recv())
        .filter_map(|m| match m {
            BusMessage::StreamCollection(c) => Some(c),
            _ => None,
        })
        .next()
        .expect("a StreamCollection was announced");
    assert_eq!(announced.streams_of_type(StreamType::Video).count(), 1);
    assert_eq!(announced.streams_of_type(StreamType::Audio).count(), 1);
}

#[tokio::test]
async fn mkvdemuxn_leaves_an_unselected_stream_dark() {
    // Two ports, but both video renditions the file does not carry (it has H.264 +
    // AAC): the AAC track is dropped (no port), the H.265 port stays dark, and the
    // H.264 port carries the video. Proves routing drops unported tracks.
    let file = mux_av().await;
    let mut demux = MkvDemuxN::new(vec![MkvStream::H264, MkvStream::H265]);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure");

    let mut tap = PortTap::new(2);
    demux.process(data_frame(&file), &mut tap).await.unwrap();

    assert_eq!(tap.frames[0].len(), 2, "H.264 port carries the video");
    assert!(
        tap.frames[1].is_empty(),
        "the H.265 port stays dark (no such track)"
    );
    assert!(tap.caps[1].is_empty(), "a dark port announces no caps");
}

//! M381 - app-driven stream selection on the multi-output demuxer. A
//! `StreamSelectController` re-maps which stream each `MkvDemuxN` port carries:
//! the app names a stream id per port (from the M376 collection), and the demuxer
//! re-routes accordingly, re-arming each changed port's CapsChanged and confirming
//! the active ids on the bus. The multi-output counterpart of M377 (single-output
//! switch).
//!
//! Start a two-port demux as [video, audio]; after the collection is announced,
//! re-select to swap the ports to [audio, video] and feed the Clusters: port 0 now
//! carries the AAC track and port 1 the H.264 track.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::StreamSelectController;
use g2g_core::fanout::{MultiOutputElement, MultiOutputSink};
use g2g_core::{
    AudioFormat, Bus, BusMessage, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement,
    OutputSink, PushOutcome, Rate, StreamType, VideoCodec,
};
use g2g_plugins::mkvdemux::{MkvDemuxN, MkvStream};
use g2g_plugins::mkvmuxn::MkvMuxN;

const CLUSTER_ID: [u8; 4] = [0x1F, 0x43, 0xB6, 0x75];

fn h264_caps() -> Caps {
    Caps::CompressedVideo { codec: VideoCodec::H264, width: Dim::Fixed(320), height: Dim::Fixed(240), framerate: Rate::Fixed(30 << 16) }
}
fn aac_caps() -> Caps {
    Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 48_000 }
}

#[derive(Default)]
struct PortTap {
    frames: Vec<Vec<Vec<u8>>>,
    caps: Vec<Vec<Caps>>,
}
impl PortTap {
    fn new(ports: usize) -> Self {
        Self { frames: vec![Vec::new(); ports], caps: vec![Vec::new(); ports] }
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
                PipelinePacket::DataFrame(Frame { domain: MemoryDomain::System(s), .. }) => {
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
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink).await.unwrap();
    mux.process(1, frame(adts_au(&[0xA1, 0xA2, 0xA3]), 0), &mut sink).await.unwrap();
    mux.process(0, frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000), &mut sink).await.unwrap();
    mux.process(1, frame(adts_au(&[0xB4, 0xB5]), 21_000_000), &mut sink).await.unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink).await.unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.bytes
}

#[tokio::test]
async fn selection_remaps_which_stream_each_port_carries() {
    let file = mux_av().await;
    let first_cluster = file.windows(4).position(|w| w == CLUSTER_ID).unwrap();
    let (header, clusters) = file.split_at(first_cluster);

    let (bus, handle) = Bus::new(16);
    let select = StreamSelectController::new();
    // Ports start [video, audio]; selection will swap them.
    let mut demux = MkvDemuxN::new(vec![MkvStream::H264, MkvStream::Aac])
        .with_bus(handle)
        .with_stream_select(select.clone());
    demux
        .configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::Matroska })
        .expect("configure");

    let mut tap = PortTap::new(2);

    // 1. Header only: the collection is announced, no Cluster yet so no frames.
    demux.process(data_frame(header), &mut tap).await.unwrap();
    assert!(tap.frames.iter().all(|p| p.is_empty()), "no frames before any Cluster");

    // 2. Discover the announced ids and select a SWAP: port 0 <- audio, port 1 <- video.
    let mut video_id = None;
    let mut audio_id = None;
    while let Some(msg) = bus.try_recv() {
        if let BusMessage::StreamCollection(c) = msg {
            video_id = c.streams_of_type(StreamType::Video).next().map(|s| s.id.clone());
            audio_id = c.streams_of_type(StreamType::Audio).next().map(|s| s.id.clone());
        }
    }
    let video_id = video_id.expect("video id");
    let audio_id = audio_id.expect("audio id");
    select.select(vec![audio_id.clone(), video_id.clone()]);

    // 3. Feed the Clusters: the swap is applied before routing, so port 0 now
    // carries the AAC track and port 1 the H.264 track.
    demux.process(data_frame(clusters), &mut tap).await.unwrap();

    assert!(
        tap.caps[0].last().map(|c| matches!(c, Caps::Audio { format: AudioFormat::Aac, .. })).unwrap_or(false),
        "port 0 re-mapped to the audio stream"
    );
    assert_eq!(tap.frames[0], vec![vec![0xA1u8, 0xA2, 0xA3], vec![0xB4, 0xB5]], "port 0 carries AAC after the swap");
    assert!(
        tap.caps[1].last().map(|c| matches!(c, Caps::CompressedVideo { codec: VideoCodec::H264, .. })).unwrap_or(false),
        "port 1 re-mapped to the video stream"
    );
    assert_eq!(tap.frames[1].len(), 2, "port 1 carries the two H.264 access units after the swap");

    // 4. The active selection is confirmed on the bus.
    let selected: Vec<_> = core::iter::from_fn(|| bus.try_recv())
        .filter_map(|m| match m {
            BusMessage::StreamsSelected { ids } => Some(ids),
            _ => None,
        })
        .collect();
    assert_eq!(selected, vec![vec![audio_id, video_id]], "StreamsSelected confirms the per-port ids");
}

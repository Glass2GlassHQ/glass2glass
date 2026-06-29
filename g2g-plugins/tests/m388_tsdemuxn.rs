//! M388 - multi-output MPEG-TS demuxer `TsDemuxN`. One TS byte stream in, N
//! elementary streams out, one per output port; the demuxer parses the transport
//! stream once and routes each PES access unit to the port whose codec matches the
//! PMT `stream_type`, emitting each port's caps before its first frame. The
//! MPEG-TS sibling of `MkvDemuxN`, what lets a single demuxer feed audio + video
//! decode branches in one pipeline (the `playbin uri=*.ts` fan-out core).
//!
//! Build an A/V transport stream by hand (PAT + 2-stream PMT + video/audio PES),
//! demux it with a two-port `TsDemuxN` (port 0 = H.264, port 1 = AAC), and assert
//! each port gets its own CapsChanged and only its stream's access units.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::fanout::MultiOutputSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, Bus, BusMessage, ByteStreamEncoding, Caps, G2gError, MultiOutputElement,
    PushOutcome, VideoCodec,
};
use g2g_plugins::mpegts::{STREAM_TYPE_AAC, STREAM_TYPE_H264};
use g2g_plugins::tsdemux::{TsDemuxN, TsStream};

const TS_SYNC: u8 = 0x47;
const TS_PACKET_LEN: usize = 188;

// --- MPEG-TS section / PES builders (mirroring the tsdemux unit-test helpers) ---
fn ts_packet(pid: u16, pusi: bool, payload: &[u8]) -> Vec<u8> {
    const ROOM: usize = TS_PACKET_LEN - 4;
    let mut p = vec![0u8; TS_PACKET_LEN];
    p[0] = TS_SYNC;
    p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) as u8 & 0x1F);
    p[2] = (pid & 0xFF) as u8;
    let l = payload.len();
    if l == ROOM {
        p[3] = 0x10;
        p[4..].copy_from_slice(payload);
    } else {
        p[3] = 0x30;
        let af_len = ROOM - 1 - l;
        p[4] = af_len as u8;
        if af_len >= 1 {
            p[5] = 0x00;
            for b in p.iter_mut().take(6 + (af_len - 1)).skip(6) {
                *b = 0xFF;
            }
        }
        p[5 + af_len..].copy_from_slice(payload);
    }
    p
}
fn psi(pid: u16, table_id: u8, body: &[u8]) -> Vec<u8> {
    let section_length = body.len() + 4;
    let mut s = vec![table_id, 0xB0 | ((section_length >> 8) as u8 & 0x0F), (section_length & 0xFF) as u8];
    s.extend_from_slice(body);
    s.extend_from_slice(&[0, 0, 0, 0]);
    let mut payload = vec![0u8];
    payload.extend_from_slice(&s);
    ts_packet(pid, true, &payload)
}
fn pat(pmt_pid: u16) -> Vec<u8> {
    psi(0x0000, 0x00, &[0, 1, 0xC1, 0, 0, 0, 1, 0xE0 | (pmt_pid >> 8) as u8 & 0x1F, pmt_pid as u8])
}
fn pmt2(v_pid: u16, v_type: u8, a_pid: u16, a_type: u8) -> Vec<u8> {
    psi(
        0x1000,
        0x02,
        &[
            0x00, 0x01, 0xC1, 0x00, 0x00,
            0xE0 | (v_pid >> 8) as u8 & 0x1F, v_pid as u8,
            0xF0, 0x00,
            v_type, 0xE0 | (v_pid >> 8) as u8 & 0x1F, v_pid as u8, 0xF0, 0x00,
            a_type, 0xE0 | (a_pid >> 8) as u8 & 0x1F, a_pid as u8, 0xF0, 0x00,
        ],
    )
}
/// A PES with an explicit `stream_id` (video 0xE0, audio 0xC0), no PTS.
fn pes_id(stream_id: u8, es: &[u8]) -> Vec<u8> {
    let mut p = vec![0x00, 0x00, 0x01, stream_id];
    let header = [0x80u8, 0x00, 0x00];
    let len = header.len() + es.len();
    p.push((len >> 8) as u8);
    p.push((len & 0xFF) as u8);
    p.extend_from_slice(&header);
    p.extend_from_slice(es);
    p
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

fn data_frame(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

/// Build an A/V transport stream: PAT + PMT (H.264 video pid 256, AAC audio pid
/// 257) + two access units per stream (each PES completes when the next PUSI for
/// that PID arrives, the last on flush).
fn av_ts(v_aus: &[&[u8]], a_aus: &[&[u8]]) -> Vec<u8> {
    let v_pid = 0x0100u16;
    let a_pid = 0x0101u16;
    let mut ts = pat(0x1000);
    ts.extend_from_slice(&pmt2(v_pid, STREAM_TYPE_H264, a_pid, STREAM_TYPE_AAC));
    let n = v_aus.len().max(a_aus.len());
    for i in 0..n {
        if let Some(au) = v_aus.get(i) {
            ts.extend_from_slice(&ts_packet(v_pid, true, &pes_id(0xE0, au)));
        }
        if let Some(au) = a_aus.get(i) {
            ts.extend_from_slice(&ts_packet(a_pid, true, &pes_id(0xC0, au)));
        }
    }
    ts
}

#[tokio::test]
async fn tsdemuxn_splits_av_onto_two_ports() {
    let v0: &[u8] = &[0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1e];
    let v1: &[u8] = &[0, 0, 0, 1, 0x41, 0x9a, 0x00];
    let a0: &[u8] = &[0xFF, 0xF1, 0x50, 0x80, 0x01, 0x02];
    let a1: &[u8] = &[0xFF, 0xF1, 0x50, 0x80, 0x03];
    let ts = av_ts(&[v0, v1], &[a0, a1]);

    let (bus, handle) = Bus::new(16);
    let mut demux = TsDemuxN::new(vec![TsStream::H264, TsStream::Aac]).with_bus(handle);
    demux
        .configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs })
        .expect("configure");

    let mut tap = PortTap::new(2);
    demux.process(data_frame(ts), &mut tap).await.unwrap();
    // EOS flushes the final in-flight PES of each stream and routes it.
    demux.process(PipelinePacket::Eos, &mut tap).await.unwrap();

    // Port 0 carries the two H.264 access units; port 1 the two AAC ones.
    assert_eq!(tap.frames[0], vec![v0.to_vec(), v1.to_vec()], "video AUs on port 0");
    assert_eq!(tap.frames[1], vec![a0.to_vec(), a1.to_vec()], "audio AUs on port 1");

    // Each port announced its elementary caps once, before its frames.
    assert_eq!(tap.caps[0].len(), 1, "one CapsChanged on the video port");
    assert!(matches!(tap.caps[0][0], Caps::CompressedVideo { codec: VideoCodec::H264, .. }));
    assert_eq!(tap.caps[1].len(), 1, "one CapsChanged on the audio port");
    assert!(matches!(tap.caps[1][0], Caps::Audio { format: AudioFormat::Aac, .. }));

    // The same StreamCollection (M386) is announced for discovery.
    let collections = core::iter::from_fn(|| bus.try_recv())
        .filter(|m| matches!(m, BusMessage::StreamCollection(_)))
        .count();
    assert_eq!(collections, 1, "the program's streams are announced once");
}

#[tokio::test]
async fn tsdemuxn_leaves_an_absent_streams_port_dark() {
    // Two ports requested (video + audio), but the multiplex carries only video.
    let v0: &[u8] = &[0, 0, 0, 1, 0x67, 0x42];
    let v1: &[u8] = &[0, 0, 0, 1, 0x41, 0x9a];
    let ts = av_ts(&[v0, v1], &[]);

    let mut demux = TsDemuxN::new(vec![TsStream::H264, TsStream::Aac]);
    demux
        .configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs })
        .unwrap();
    let mut tap = PortTap::new(2);
    demux.process(data_frame(ts), &mut tap).await.unwrap();
    demux.process(PipelinePacket::Eos, &mut tap).await.unwrap();

    assert_eq!(tap.frames[0].len(), 2, "video port carries its AUs");
    assert!(tap.frames[1].is_empty(), "audio port stays dark (no audio in the multiplex)");
    assert!(tap.caps[1].is_empty(), "a dark port announces nothing");
}

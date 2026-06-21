//! M167 RTMP ingest end to end: a test publisher connects to `RtmpSrc` over a
//! real loopback TCP socket, performs the simple handshake + connect /
//! createStream / publish flow, and pushes one video message. `RtmpSrc` reframes
//! it into an FLV byte stream that `flvdemux` recovers the access unit from.

#![cfg(feature = "rtmp")]

use core::future::Future;
use core::pin::Pin;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome,
};
use g2g_plugins::flv::{FlvDemuxer, FlvTrack};
use g2g_plugins::rtmpsrc::RtmpSrc;

#[derive(Default)]
struct CaptureSink {
    flv: Vec<u8>,
    eos: bool,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let MemoryDomain::System(s) = &f.domain {
                        self.flv.extend_from_slice(s.as_slice());
                    }
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn push_u24(out: &mut Vec<u8>, v: u32) {
    out.push((v >> 16) as u8);
    out.push((v >> 8) as u8);
    out.push(v as u8);
}

fn amf_string(out: &mut Vec<u8>, s: &str) {
    out.push(0x02);
    out.extend_from_slice(&(s.len() as u16).to_be_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn amf_number(out: &mut Vec<u8>, v: f64) {
    out.push(0x00);
    out.extend_from_slice(&v.to_be_bytes());
}

/// One `fmt 0` chunk carrying a whole (sub-128-byte) message.
fn chunk(csid: u8, msg_type: u8, msid: u32, ts: u32, payload: &[u8]) -> Vec<u8> {
    let mut c = vec![csid & 0x3F];
    push_u24(&mut c, ts & 0x00FF_FFFF);
    push_u24(&mut c, payload.len() as u32);
    c.push(msg_type);
    c.extend_from_slice(&msid.to_le_bytes());
    c.extend_from_slice(payload);
    c
}

/// The publisher side: handshake, the publish command flow, then one video tag.
fn publish_video(addr: std::net::SocketAddr, au: Vec<u8>) {
    let mut stream = TcpStream::connect(addr).unwrap();
    // C0 + C1, then read S0+S1+S2 (3073 bytes), then C2.
    stream.write_all(&[3]).unwrap();
    stream.write_all(&[0u8; 1536]).unwrap();
    let mut s0s1s2 = [0u8; 1 + 1536 + 1536];
    stream.read_exact(&mut s0s1s2).unwrap();
    stream.write_all(&[0u8; 1536]).unwrap();

    let mut connect = Vec::new();
    amf_string(&mut connect, "connect");
    amf_number(&mut connect, 1.0);
    connect.push(0x05);
    stream.write_all(&chunk(3, 20, 0, 0, &connect)).unwrap();

    let mut create = Vec::new();
    amf_string(&mut create, "createStream");
    amf_number(&mut create, 2.0);
    create.push(0x05);
    stream.write_all(&chunk(3, 20, 0, 0, &create)).unwrap();

    let mut publish = Vec::new();
    amf_string(&mut publish, "publish");
    amf_number(&mut publish, 0.0);
    publish.push(0x05);
    amf_string(&mut publish, "key");
    amf_string(&mut publish, "live");
    stream.write_all(&chunk(3, 20, 1, 0, &publish)).unwrap();

    let mut vbody = vec![0x17u8, 0x01, 0x00, 0x00, 0x00]; // keyframe|AVC, NALU, cts 0
    vbody.extend_from_slice(&au);
    stream.write_all(&chunk(6, 9, 1, 40, &vbody)).unwrap();

    // Close the write side so the server sees EOS; drain any server replies.
    stream.shutdown(std::net::Shutdown::Write).unwrap();
    let _ = stream.read_to_end(&mut Vec::new());
}

#[tokio::test]
async fn rtmp_publish_demuxes_to_access_units() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    let au = vec![0u8, 0, 0, 3, 0x65, 0x11, 0x22]; // AVCC: 4-byte length=3 + NAL
    let au_for_client = au.clone();
    thread::spawn(move || publish_video(addr, au_for_client));

    let mut src = RtmpSrc::from_listener(listener).unwrap();
    src.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::Flv }).unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();

    assert!(sink.eos, "connection close ends the source");
    let mut demux = FlvDemuxer::new();
    demux.push_data(&sink.flv);
    let units = demux.take_units();
    assert_eq!(units.len(), 1, "the published video tag demuxes to one access unit");
    assert_eq!(units[0].track, FlvTrack::Video);
    assert_eq!(units[0].data, au, "RtmpSrc -> flvdemux recovers the AVCC access unit");
    assert_eq!(units[0].pts_ms, 40);
}

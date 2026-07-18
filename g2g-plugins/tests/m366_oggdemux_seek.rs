//! M366 - Ogg/Opus demuxer seek (`OggDemux` over a seekable `FileSrc`). Drives an
//! upstream byte-seek and re-syncs from the packet at or after the target. The
//! Ogg demuxer carries no per-packet PTS, so the element now accumulates one from
//! each Opus packet's decoded duration (TOC byte, 48 kHz); every audio packet is
//! a resync point.
//!
//! The clip is five 20 ms Opus packets (PTS 0, 20, 40, 60, 80 ms). A seek to
//! 50 ms resumes from the 60 ms packet.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, PipelinePacket};
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{ByteStreamEncoding, Caps, G2gError, Seek};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::oggdemux::OggDemux;

use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m366_{}_{}.opus", std::process::id(), name))
}

/// One Ogg page carrying `packets` (each laced into 255-byte segments).
fn page(header_type: u8, serial: u32, seq: u32, packets: &[&[u8]]) -> Vec<u8> {
    let mut table = Vec::new();
    let mut body = Vec::new();
    for p in packets {
        let mut n = p.len();
        loop {
            let seg = n.min(255);
            table.push(seg as u8);
            n -= seg;
            if seg < 255 {
                break;
            }
        }
        body.extend_from_slice(p);
    }
    let mut out = b"OggS".to_vec();
    out.push(0); // version
    out.push(header_type);
    out.extend_from_slice(&0u64.to_le_bytes()); // granule
    out.extend_from_slice(&serial.to_le_bytes());
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // CRC (ignored on read)
    out.push(table.len() as u8);
    out.extend_from_slice(&table);
    out.extend_from_slice(&body);
    out
}

fn opus_head(channels: u8) -> Vec<u8> {
    let mut h = b"OpusHead".to_vec();
    h.push(1);
    h.push(channels);
    h.extend_from_slice(&[0, 0]);
    h.extend_from_slice(&48_000u32.to_le_bytes());
    h.extend_from_slice(&[0, 0, 0]);
    h
}

/// OpusHead (BOS) + OpusTags + five 20 ms audio packets (TOC 0x08 = SILK NB
/// 20 ms, one frame), each tagged with a distinct second byte.
fn synthetic_ogg() -> Vec<u8> {
    let serial = 0x0BAD_F00D;
    let pkts: Vec<Vec<u8>> = (0..5u8).map(|i| vec![0x08, 0xA0 + i]).collect();
    let refs: Vec<&[u8]> = pkts.iter().map(|p| p.as_slice()).collect();
    let mut s = Vec::new();
    s.extend_from_slice(&page(0x02, serial, 0, &[&opus_head(2)]));
    s.extend_from_slice(&page(0x00, serial, 1, &[b"OpusTags\0\0\0\0"]));
    s.extend_from_slice(&page(0x00, serial, 2, &refs));
    s
}

#[derive(Default)]
struct Capture {
    frames: Vec<Vec<u8>>,
    flushes: usize,
    segments: usize,
}
impl OutputSink for Capture {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(s),
                    ..
                }) => {
                    self.frames.push(s.as_slice().to_vec());
                }
                PipelinePacket::Flush => self.flushes += 1,
                PipelinePacket::Segment(_) => self.segments += 1,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

struct Chain<'a> {
    demux: &'a mut OggDemux,
    capture: &'a mut Capture,
}
impl OutputSink for Chain<'_> {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            self.demux.process(packet, self.capture).await?;
            Ok(PushOutcome::Accepted)
        })
    }
}

#[tokio::test]
async fn oggdemux_seeks_to_the_target_packet_over_filesrc() {
    let path = temp_path("seek");
    std::fs::write(&path, synthetic_ogg()).unwrap();

    let byte = SeekController::new();
    let time = SeekController::new();
    // Seek to 50 ms: resume from the first packet at/after it, the 60 ms one.
    time.seek(Seek::flush_to(50_000_000));

    let mut src = FileSrc::new(
        &path,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        },
    )
    .with_chunk_size(16)
    .with_seek(byte.clone());
    let mut demux = OggDemux::new().with_seek(time.clone(), byte.clone());

    let caps = {
        let c: Pin<Box<dyn Future<Output = _>>> = Box::pin(src.intercept_caps());
        c.await.expect("probe")
    };
    src.configure_pipeline(&caps).expect("configure src");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        })
        .expect("configure demux");

    let mut capture = Capture::default();
    {
        let mut chain = Chain {
            demux: &mut demux,
            capture: &mut capture,
        };
        src.run(&mut chain).await.expect("filesrc runs");
    }

    assert!(
        capture.flushes >= 1,
        "the upstream byte-seek flushed downstream"
    );
    assert!(capture.segments >= 1, "a resume segment was emitted");
    // Packets at 0,20,40 ms dropped; resume from 60 ms (0xA3) and 80 ms (0xA4).
    assert_eq!(
        capture.frames,
        vec![vec![0x08u8, 0xA3], vec![0x08u8, 0xA4]],
        "resumed from the 60 ms packet, pre-target packets discarded"
    );
    let _ = std::fs::remove_file(&path);
}

//! M363 - MPEG-TS demuxer seek (`TsDemux` over a seekable `FileSrc`). Like the
//! fMP4 case (M362), the demuxer drives an upstream byte-seek and re-syncs from
//! the keyframe at or after the target. TS units carry no keyframe flag, so the
//! demuxer detects an IDR/IRAP in the access unit itself (`annexb::au_is_keyframe`).
//!
//! The clip has IDR access units at AU 0 and AU 4 (TS repeats SPS/PPS in-band
//! before each IDR, so the resumed AU is independently decodable). A seek to a
//! time between them resumes from AU 4.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{ByteStreamEncoding, Caps, Dim, G2gError, Rate, Seek, VideoCodec};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::tsdemux::TsDemux;
use g2g_plugins::tsmux::TsMux;

use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m363_{}_{}.ts", std::process::id(), name))
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn au_frame(bytes: Vec<u8>, pts_ns: u64, seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: 33_333_333,
            ..FrameTiming::default()
        },
        sequence: seq,
        meta: Default::default(),
    }
}

#[derive(Default)]
struct Bytes {
    out: Vec<u8>,
}
impl OutputSink for Bytes {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(Frame {
                domain: MemoryDomain::System(s),
                ..
            }) = packet
            {
                self.out.extend_from_slice(s.as_slice());
            }
            Ok(PushOutcome::Accepted)
        })
    }
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

/// FileSrc -> TsDemux -> capture, in one task.
struct Chain<'a> {
    demux: &'a mut TsDemux,
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

async fn make_ts(aus: &[Vec<u8>]) -> Vec<u8> {
    let mut mux = TsMux::new();
    mux.configure_pipeline(&h264_caps()).unwrap();
    let mut bytes = Bytes::default();
    for (i, au) in aus.iter().enumerate() {
        mux.process(
            PipelinePacket::DataFrame(au_frame(au.clone(), i as u64 * 33_333_333, i as u64)),
            &mut bytes,
        )
        .await
        .unwrap();
    }
    mux.process(PipelinePacket::Eos, &mut bytes).await.unwrap();
    bytes.out
}

#[tokio::test]
async fn tsdemux_seeks_to_the_target_keyframe_over_filesrc() {
    // IDR access units (SPS+PPS+IDR, in-band) at AU 0 and AU 4; P slices between.
    let sc = &[0, 0, 0, 1][..];
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11, 0x22];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr = |tag: u8| [sc, &sps, sc, &pps, sc, &[0x65, tag]].concat();
    let p = |f: u8| [sc, &[0x41, f][..]].concat();
    let aus = vec![idr(0xA0), p(1), p(2), p(3), idr(0xA4), p(5), p(6), p(7)];

    let ts = make_ts(&aus).await;
    let path = temp_path("seek");
    std::fs::write(&path, &ts).unwrap();

    let byte = SeekController::new();
    let time = SeekController::new();
    // Seek to 100 ms: resume from the next keyframe, AU 4 (133 ms).
    time.seek(Seek::flush_to(100_000_000));

    let mut src = FileSrc::new(&path, Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs })
        .with_chunk_size(188) // one TS packet per chunk: byte-seek observed mid-read
        .with_seek(byte.clone());
    let mut demux = TsDemux::new().with_seek(time.clone(), byte.clone());

    let caps = {
        let c: Pin<Box<dyn Future<Output = _>>> = Box::pin(src.intercept_caps());
        c.await.expect("probe")
    };
    src.configure_pipeline(&caps).expect("configure src");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
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

    // Re-synced from AU 4: the tail idr(0xA4), p5, p6, p7.
    assert_eq!(
        capture.frames.len(),
        4,
        "resumed from the target keyframe (AU 4)"
    );
    assert!(
        capture.flushes >= 1,
        "the upstream byte-seek flushed downstream"
    );
    assert!(capture.segments >= 1, "a resume segment was emitted");
    assert!(
        capture.frames[0].windows(2).any(|w| w == [0x65, 0xA4]),
        "first resumed AU is AU 4's IDR"
    );
    assert!(
        !capture
            .frames
            .iter()
            .any(|f| f.windows(2).any(|w| w == [0x65, 0xA0])),
        "the pre-target keyframe (AU 0) was discarded"
    );
    let _ = std::fs::remove_file(&path);
}

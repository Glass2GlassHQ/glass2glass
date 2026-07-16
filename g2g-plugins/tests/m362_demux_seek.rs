//! M362 - demuxer seek (`Fmp4Demux` over a seekable `FileSrc`). A demuxer is a
//! transform with no random access; it becomes seek-aware by driving its
//! upstream byte source. The app seeks the demuxer (time); the demuxer asks
//! `FileSrc` to reposition (byte offset 0, re-scan), resets its parser on the
//! returned `Flush`, and re-syncs from the keyframe at or after the target.
//!
//! The clip has keyframes at AU 0 and AU 4. A seek to a time between them must
//! resume from AU 4 (the first keyframe >= target), with the parameter sets
//! re-prepended so a decoder can start, and a fresh segment emitted.
//!
//! The pipeline is driven as a manual `FileSrc -> Fmp4Demux -> capture` chain so
//! the test exercises the real byte-seek round-trip (FileSrc repositions, the
//! demuxer resets and re-syncs) without the graph runner's startup negotiation,
//! which is orthogonal here (`Fmp4Demux` advertises runtime-refined geometry, so
//! a real graph puts an `h264parse` after it to fixate).

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{ByteStreamEncoding, Caps, Dim, G2gError, Rate, Seek, VideoCodec};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::fmp4demux::Fmp4Demux;
use g2g_plugins::mp4mux::Mp4Mux;

use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m362_{}_{}.mp4", std::process::id(), name))
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(64),
        height: Dim::Fixed(48),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn au_frame(bytes: Vec<u8>, pts_ns: u64, seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming { pts_ns, dts_ns: pts_ns, duration_ns: 33_333_333, ..FrameTiming::default() },
        sequence: seq,
        meta: Default::default(),
    }
}

/// Concatenate the ISO-BMFF byte stream `Mp4Mux` forwards.
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
            if let PipelinePacket::DataFrame(Frame { domain: MemoryDomain::System(s), .. }) = packet {
                self.out.extend_from_slice(s.as_slice());
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Records the demuxed access units, the `Flush`, the resume segment, and caps.
#[derive(Default)]
struct Capture {
    frames: Vec<Vec<u8>>,
    flushes: usize,
    segments: usize,
    caps: usize,
}
impl OutputSink for Capture {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(Frame { domain: MemoryDomain::System(s), .. }) => {
                    self.frames.push(s.as_slice().to_vec());
                }
                PipelinePacket::Flush => self.flushes += 1,
                PipelinePacket::Segment(_) => self.segments += 1,
                PipelinePacket::CapsChanged(_) => self.caps += 1,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Adapter sink that feeds each packet `FileSrc` emits straight into the demuxer,
/// whose own output lands in `capture`: a `FileSrc -> Fmp4Demux -> capture` chain
/// in one task, so the byte-seek the demuxer drives round-trips to `FileSrc`.
struct Chain<'a> {
    demux: &'a mut Fmp4Demux,
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

/// Mux the access units to an fMP4 byte buffer via `Mp4Mux`.
async fn make_fmp4(aus: &[Vec<u8>]) -> Vec<u8> {
    let mut mux = Mp4Mux::new();
    mux.configure_pipeline(&h264_caps()).unwrap();
    let mut bytes = Bytes::default();
    for (i, au) in aus.iter().enumerate() {
        mux.process(PipelinePacket::DataFrame(au_frame(au.clone(), i as u64 * 33_333_333, i as u64)), &mut bytes)
            .await
            .unwrap();
    }
    mux.process(PipelinePacket::Eos, &mut bytes).await.unwrap();
    bytes.out
}

/// Run a `FileSrc(path) -> Fmp4Demux -> Capture` chain to EOS.
async fn run_chain(src: &mut FileSrc, demux: &mut Fmp4Demux) -> Capture {
    let caps = {
        let c: Pin<Box<dyn Future<Output = _>>> = Box::pin(src.intercept_caps());
        c.await.expect("probe")
    };
    src.configure_pipeline(&caps).expect("configure src");
    demux
        .configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff })
        .expect("configure demux");
    let mut capture = Capture::default();
    {
        let mut chain = Chain { demux, capture: &mut capture };
        src.run(&mut chain).await.expect("filesrc runs");
    }
    capture
}

#[tokio::test]
async fn fmp4demux_seeks_to_the_target_keyframe_over_filesrc() {
    // 8 AUs; keyframes (IDR, NAL type 5) at index 0 and 4.
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11, 0x22];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let sc = &[0, 0, 0, 1][..];
    let idr0: Vec<u8> = [sc, &sps, sc, &pps, sc, &[0x65, 0xA0]].concat();
    let idr4: Vec<u8> = [sc, &[0x65, 0xA4]].concat(); // bare IDR: param sets re-prepended on resume
    let p = |f: u8| [sc, &[0x41, f][..]].concat();
    let aus = vec![idr0, p(1), p(2), p(3), idr4, p(5), p(6), p(7)];

    let fmp4 = make_fmp4(&aus).await;
    let path = temp_path("seek");
    std::fs::write(&path, &fmp4).unwrap();

    // `byte` drives FileSrc (and is the demuxer's upstream); `time` is the
    // app-facing time seek. Seek to 100 ms: between AU 3 (100 ms) and the next
    // keyframe AU 4 (133 ms), so the demuxer must resume from AU 4.
    let byte = SeekController::new();
    let time = SeekController::new();
    time.seek(Seek::flush_to(100_000_000));

    let mut src = FileSrc::new(&path, Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff })
        .with_chunk_size(64) // many chunks: FileSrc observes the byte-seek mid-read
        .with_seek(byte.clone());
    let mut demux = Fmp4Demux::new().with_seek(time.clone(), byte.clone());

    let cap = run_chain(&mut src, &mut demux).await;

    // Re-synced from AU 4: exactly the tail idr4, p5, p6, p7.
    assert_eq!(cap.frames.len(), 4, "resumed from the target keyframe (AU 4) to the end");
    assert!(cap.flushes >= 1, "the upstream byte-seek flushed downstream");
    assert!(cap.segments >= 1, "a resume segment was emitted");
    assert_eq!(cap.caps, 1, "caps announced exactly once despite the re-scan");

    // The first resumed frame is the IDR with parameter sets re-prepended.
    let first = &cap.frames[0];
    assert!(first.windows(2).any(|w| w == [0x65, 0xA4]), "resumed at AU 4's IDR");
    assert!(first.windows(2).any(|w| w == [0x67, 0x42]), "parameter sets re-prepended for resume");
    // No pre-target frames leaked through.
    assert!(
        !cap.frames.iter().any(|f| f.windows(2).any(|w| w == [0x65, 0xA0])),
        "the pre-target keyframe (AU 0) was discarded"
    );

    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn no_seek_demuxes_the_whole_clip_over_filesrc() {
    let sc = &[0, 0, 0, 1][..];
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11, 0x22];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr0: Vec<u8> = [sc, &sps, sc, &pps, sc, &[0x65, 0xA0]].concat();
    let p = |f: u8| [sc, &[0x41, f][..]].concat();
    let aus = vec![idr0, p(1), p(2), p(3)];
    let fmp4 = make_fmp4(&aus).await;
    let path = temp_path("noseek");
    std::fs::write(&path, &fmp4).unwrap();

    let mut src = FileSrc::new(&path, Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff })
        .with_chunk_size(64);
    let mut demux = Fmp4Demux::new();
    let cap = run_chain(&mut src, &mut demux).await;

    // All four access units recovered when not seeking (no flush, one caps).
    assert_eq!(cap.frames.len(), 4);
    assert_eq!(cap.flushes, 0);
    assert_eq!(cap.caps, 1);
    let _ = std::fs::remove_file(&path);
}

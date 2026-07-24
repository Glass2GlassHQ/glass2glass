//! M365 - FLV demuxer seek (`FlvDemux` over a seekable `FileSrc`). Drives an
//! upstream byte-seek and re-syncs from the keyframe at or after the target,
//! using the FLV video tag's frame-type flag.
//!
//! The clip has keyframe video tags at 0 ms and 120 ms, with interframes between.
//! A seek to 80 ms resumes from the 120 ms keyframe.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, PipelinePacket};
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{ByteStreamEncoding, Caps, G2gError, Seek};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::flv::{FlvMuxer, FlvTrack};
use g2g_plugins::flvdemux::{FlvDemux, FlvStream};

use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m365_{}_{}.flv", std::process::id(), name))
}

/// Build an FLV byte stream: video tags `(pts_ms, keyframe, au)`.
fn make_flv(tags: &[(u32, bool, Vec<u8>)]) -> Vec<u8> {
    let mut mux = FlvMuxer::new(FlvTrack::Video);
    let mut out = Vec::new();
    for (pts, kf, au) in tags {
        out.extend_from_slice(&mux.push_video(au, *pts, 0, *kf));
    }
    out
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
    demux: &'a mut FlvDemux,
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
async fn flvdemux_seeks_to_the_target_keyframe_over_filesrc() {
    // Valid AVCC payloads (4-byte length prefix); the element re-frames them
    // Annex-B on the way out (M662).
    let flv = make_flv(&[
        (0, true, vec![0, 0, 0, 2, 0x65, 0x01]),
        (40, false, vec![0, 0, 0, 2, 0x41, 0x02]),
        (80, false, vec![0, 0, 0, 2, 0x41, 0x03]),
        (120, true, vec![0, 0, 0, 2, 0x65, 0x04]),
        (160, false, vec![0, 0, 0, 2, 0x41, 0x05]),
    ]);
    let path = temp_path("seek");
    std::fs::write(&path, &flv).unwrap();

    let byte = SeekController::new();
    let time = SeekController::new();
    // Seek to 80 ms: resume from the next keyframe at 120 ms.
    time.seek(Seek::flush_to(80_000_000));

    let mut src = FileSrc::new(
        &path,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Flv,
        },
    )
    .with_chunk_size(16)
    .with_seek(byte.clone());
    let mut demux = FlvDemux::new()
        .with_stream(FlvStream::H264)
        .with_seek(time.clone(), byte.clone());

    let caps = {
        let c: Pin<Box<dyn Future<Output = _>>> = Box::pin(src.intercept_caps());
        c.await.expect("probe")
    };
    src.configure_pipeline(&caps).expect("configure src");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Flv,
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
    assert_eq!(
        capture.frames,
        vec![vec![0, 0, 0, 1, 0x65, 0x04], vec![0, 0, 0, 1, 0x41, 0x05]],
        "resumed from the 120 ms keyframe to the end, pre-target frames discarded"
    );
    let _ = std::fs::remove_file(&path);
}

//! M363 - MPEG-TS demuxer seek (`TsDemux` over a seekable `FileSrc`). Like the
//! fMP4 case (M362), the demuxer drives an upstream byte-seek and re-syncs from
//! the keyframe at or before the target. TS units carry no keyframe flag, so the
//! demuxer detects an IDR/IRAP in the access unit itself (`annexb::au_is_keyframe`).
//!
//! The clip has IDR access units at AU 0 and AU 4 (TS repeats SPS/PPS in-band
//! before each IDR, so the resumed AU is independently decodable). A seek to a
//! time after AU 5 resumes from AU 4.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{
    ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement, Rate, Seek, SeekFlags, Segment,
    VideoCodec,
};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::tsdemux::TsDemux;
use g2g_plugins::tsmuxn::TsMux;

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
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
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
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
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
    frame_pts_ns: Vec<u64>,
    flushes: usize,
    segment_starts: Vec<u64>,
    // each segment with the number of frames out before it
    segments: Vec<(usize, Segment)>,
    // issued through the controller once this many frames are out
    seek_after_frames: Option<(usize, SeekController, Seek)>,
}
impl OutputSink for Capture {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(s),
                    timing,
                    ..
                }) => {
                    self.frames.push(s.as_slice().to_vec());
                    self.frame_pts_ns.push(timing.pts_ns);
                    if let Some((frames, controller, seek)) = &self.seek_after_frames {
                        if self.frames.len() == *frames {
                            controller.seek(*seek);
                        }
                    }
                }
                PipelinePacket::Flush => self.flushes += 1,
                PipelinePacket::Segment(segment) => {
                    self.segment_starts.push(segment.start);
                    self.segments.push((self.frames.len(), segment));
                }
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
    fn poll_push(
        &mut self,
        cx: &mut core::task::Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet.take().expect("poll_push without a packet");
        let mut fut = self.demux.process(packet, self.capture);
        match fut.as_mut().poll(cx) {
            core::task::Poll::Ready(r) => {
                core::task::Poll::Ready(r.map(|()| PushOutcome::Accepted))
            }
            // In-memory test elements never block: their only awaits are
            // pushes into always-ready capture sinks.
            core::task::Poll::Pending => panic!("element future did not resolve in one poll"),
        }
    }
}

async fn make_ts(aus: &[Vec<u8>]) -> Vec<u8> {
    let mut mux = TsMux::new(1);
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    let mut bytes = Bytes::default();
    for (i, au) in aus.iter().enumerate() {
        mux.process(
            0,
            PipelinePacket::DataFrame(au_frame(au.clone(), i as u64 * 33_333_333, i as u64)),
            &mut bytes,
        )
        .await
        .unwrap();
    }
    mux.process(0, PipelinePacket::Eos, &mut bytes)
        .await
        .unwrap();
    bytes.out
}

// IDR access units (SPS+PPS+IDR, in-band) at AU 0 and AU 4; P slices between.
fn two_gop_access_units() -> Vec<Vec<u8>> {
    let sc = &[0, 0, 0, 1][..];
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11, 0x22];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr = |tag: u8| [sc, &sps, sc, &pps, sc, &[0x65, tag]].concat();
    let p = |f: u8| [sc, &[0x41, f][..]].concat();
    vec![idr(0xA0), p(1), p(2), p(3), idr(0xA4), p(5), p(6), p(7)]
}

// 180 ms after the first PTS, between AU 5 (167 ms) and AU 6 (200 ms), so playback resumes from AU 4
const SECOND_GOP_TARGET_NS: u64 = 180_000_000;
const SECOND_GOP_FRAMES: usize = 4;

// FileSrc -> TsDemux -> `capture` over the two-GOP clip, `time` seeking the demuxer
async fn run_chain(name: &str, time: SeekController, capture: Capture) -> Capture {
    let ts = make_ts(&two_gop_access_units()).await;
    let path = temp_path(name);
    std::fs::write(&path, &ts).unwrap();

    let byte = SeekController::new();
    let mut src = FileSrc::new(&path, Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs })
        .with_chunk_size(188) // one TS packet per chunk: byte-seek observed mid-read
        .with_seek(byte.clone());
    let mut demux = TsDemux::new().with_seek(time, byte);

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

    let mut capture = capture;
    {
        let mut chain = Chain {
            demux: &mut demux,
            capture: &mut capture,
        };
        src.run(&mut chain).await.expect("filesrc runs");
    }
    let _ = std::fs::remove_file(&path);
    capture
}

#[tokio::test]
async fn tsdemux_seeks_to_the_target_keyframe_over_filesrc() {
    let time = SeekController::new();
    time.seek(Seek::flush_to(SECOND_GOP_TARGET_NS));
    let capture = run_chain("seek", time, Capture::default()).await;

    // Re-synced from AU 4: the tail idr(0xA4), p5, p6, p7.
    assert_eq!(
        capture.frames.len(),
        SECOND_GOP_FRAMES,
        "resumed from the keyframe before the target (AU 4)"
    );
    assert!(
        capture.flushes >= 1,
        "the upstream byte-seek flushed downstream"
    );
    let segment_start = *capture.segment_starts.last().expect("a resume segment");
    assert!(
        capture.frame_pts_ns[1] < segment_start && segment_start <= capture.frame_pts_ns[2],
        "the resume segment starts at the target, between AU 5 and AU 6"
    );
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
}

#[tokio::test]
async fn a_non_flushing_seek_carries_running_time_on_from_the_last_frame_out() {
    let frames_before_seek = 2;
    let time = SeekController::new();
    let capture = Capture {
        seek_after_frames: Some((
            frames_before_seek,
            time.clone(),
            Seek {
                flags: SeekFlags::NONE,
                ..Seek::flush_to(SECOND_GOP_TARGET_NS)
            },
        )),
        ..Capture::default()
    };
    let capture = run_chain("nonflushing", time, capture).await;

    assert_eq!(capture.flushes, 0, "playback carries on through the seek");
    let (_, opening) = capture.segments[0];
    let &(frames_out, seek_segment) = capture.segments.last().expect("the seek sent a segment");
    assert!(frames_out >= frames_before_seek);
    assert_eq!(
        seek_segment.base,
        opening
            .to_running_time(capture.frame_pts_ns[frames_out - 1])
            .expect("the frame played inside the opening segment"),
        "running time carries on from the last frame out"
    );
    assert_eq!(capture.frames.len() - frames_out, SECOND_GOP_FRAMES);
}

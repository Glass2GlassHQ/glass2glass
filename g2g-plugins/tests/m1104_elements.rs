//! M1104: PNM stills, a tone source, DTMF generate/detect, videoanalyse,
//! scenechange, debugspy.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, PropValue, PushOutcome, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::debugspy::DebugSpy;
use g2g_plugins::pnm::{PnmDec, PnmEnc};
use g2g_plugins::registry::default_registry;
use g2g_plugins::scenechange::SceneChange;
use g2g_plugins::typefind::{sniff_caps, still_image_caps};
use g2g_plugins::videoanalyse::VideoAnalyse;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Default)]
struct Collect {
    frames: Vec<Vec<u8>>,
    keyframes: Vec<bool>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(f)) = packet_slot.take() {
            self.keyframes.push(f.timing.keyframe);
            if let Some(s) = f.domain.as_system_slice() {
                self.frames.push(s.to_vec());
            }
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn rgba(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..w * h {
        v.extend_from_slice(&[r, g, b, 255]);
    }
    v
}

fn raw_rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
    }
}

fn frame(bytes: Vec<u8>) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: 0,
            dts_ns: 0,
            duration_ns: 0,
            capture_ns: 0,
            arrival_ns: 0,
            keyframe: false,
        },
        sequence: 0,
        meta: Default::default(),
    }
}

fn push<E: AsyncElement>(element: &mut E, bytes: Vec<u8>) -> Collect {
    let mut sink = Collect::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(element.process(PipelinePacket::DataFrame(frame(bytes)), &mut sink))
        .expect("process");
    sink
}

#[test]
fn pnmenc_pnmdec_roundtrip_rgb() {
    let pixels = rgba(4, 2, 10, 20, 30);
    let mut enc = PnmEnc::new();
    enc.configure_pipeline(&raw_rgba(4, 2)).unwrap();
    let encoded = push(&mut enc, pixels.clone());
    assert_eq!(encoded.frames.len(), 1);

    let mut dec = PnmDec::new();
    dec.configure_pipeline(&still_image_caps(VideoCodec::Pnm))
        .unwrap();
    let decoded = push(&mut dec, encoded.frames[0].clone());
    assert_eq!(decoded.frames.len(), 1);
    // Decoder emits packed RGB, alpha dropped.
    assert_eq!(decoded.frames[0].len(), 4 * 2 * 3);
    assert_eq!(&decoded.frames[0][..3], &[10, 20, 30]);
}

#[test]
fn pnmenc_ascii_is_p3() {
    let mut enc = PnmEnc::new();
    enc.set_property("ascii", PropValue::Bool(true)).unwrap();
    enc.configure_pipeline(&raw_rgba(2, 1)).unwrap();
    let out = push(&mut enc, rgba(2, 1, 1, 2, 3));
    assert!(out.frames[0].starts_with(b"P3\n"));
}

#[test]
fn typefind_types_pnm() {
    let ppm = b"P6\n2 1\n255\n\x01\x02\x03\x04\x05\x06";
    assert_eq!(sniff_caps(ppm), Some(still_image_caps(VideoCodec::Pnm)));
}

#[test]
fn videoanalyse_flat_red_is_bright() {
    let mut e = VideoAnalyse::new();
    e.configure_pipeline(&raw_rgba(4, 4)).unwrap();
    push(&mut e, rgba(4, 4, 255, 0, 0));
    assert!(
        e.luma_average() > 0.1,
        "red has BT.709 luma, got {}",
        e.luma_average()
    );
    assert!(e.luma_variance() < 1e-9);
}

#[test]
fn scenechange_marks_cut() {
    let mut e = SceneChange::new();
    e.configure_pipeline(&raw_rgba(8, 8)).unwrap();
    // Fill the past window with near-identical frames, then a white flash.
    for _ in 0..6 {
        push(&mut e, rgba(8, 8, 16, 16, 16));
    }
    let cut = push(&mut e, rgba(8, 8, 255, 255, 255));
    assert!(e.changes() >= 1, "white after black is a cut");
    assert!(cut.keyframes.last().copied().unwrap_or(false));
}

#[test]
fn debugspy_hashes_buffer() {
    let mut e = DebugSpy::new();
    e.configure_pipeline(&raw_rgba(2, 2)).unwrap();
    push(&mut e, rgba(2, 2, 1, 2, 3));
    assert_eq!(e.seen(), 1);
    assert_eq!(e.last_checksum().len(), 40, "sha1 hex");
}

#[tokio::test]
async fn launch_lines_run() {
    let reg = default_registry();
    for (line, n) in [
        ("videotestsrc num-buffers=1 ! pnmenc ! pnmdec ! fakesink", 1),
        ("tonegeneratesrc num-buffers=2 freq=880 ! fakesink", 2),
        ("dtmfsrc number=4 num-buffers=3 ! dtmfdetect ! fakesink", 3),
    ] {
        let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line}: {e}"));
        let stats = run_graph(graph, &ZeroClock, 4)
            .await
            .unwrap_or_else(|e| panic!("{line}: {e:?}"));
        assert_eq!(stats.frames_consumed, n, "{line}");
    }
}

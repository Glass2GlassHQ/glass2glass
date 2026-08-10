//! M899: VobSub (DVD subpicture) bitmap subtitles, validated against ffmpeg.
//!
//! The fixture is the hand-authored `.idx` / `.sub` pair of the `vobsub_fixture`
//! module, and ffmpeg is the reference peer on both sides of it: its `vobsub`
//! demuxer reads the pair and its Matroska muxer writes the `S_VOBSUB` track
//! (`CodecPrivate`, block framing and timing all ffmpeg's), and its `dvdsub`
//! decoder renders the same cues through `overlay` for the pixel-for-pixel
//! comparison against g2g's decode composited by `Compositor`.
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement, OutputSink,
    PushOutcome, Rate, RawVideoFormat, SubPictureFormat,
};
use g2g_plugins::compositor::{Compositor, CompositorPad};
use g2g_plugins::conformance::persist;
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};
use g2g_plugins::registry::default_registry;
use g2g_plugins::vobsubdec::VobSubDec;

mod vobsub_fixture;
use vobsub_fixture::{author_vobsub, cues, have_ffmpeg, CUE_DURATION_NS, H, W};

// ---- ffmpeg ----

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m899-{}-{name}", std::process::id()))
}

/// Fixture length in seconds, and therefore the reference burn-in's frame count
/// at one frame per second.
const DURATION_S: u32 = 9;

/// Let ffmpeg mux the authored cues into Matroska over an H.264 video track.
/// Everything about the resulting `S_VOBSUB` track (CodecID, `CodecPrivate`,
/// block framing, timestamps) is ffmpeg's.
fn mux_mkv(idx: &PathBuf, out: &PathBuf) {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args([
            "-f",
            "lavfi",
            "-i",
            &format!("color=c=black:s=320x240:r=10:d={DURATION_S}"),
        ])
        .arg("-i")
        .arg(idx)
        .args(["-map", "0:v", "-map", "1:s"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:s", "copy"])
        .arg(out)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg muxed the S_VOBSUB fixture");
}

/// ffmpeg's own burn-in: its `dvdsub` decoder rendered through `overlay` onto
/// black, one RGB frame per second. Blending happens in RGB (not the default
/// YUV) so chroma subsampling cannot smear the reference.
fn reference_burn_in(idx: &PathBuf, raw: &PathBuf) -> Vec<Vec<u8>> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args([
            "-f",
            "lavfi",
            "-i",
            &format!("color=c=black:s={W}x{H}:r=1:d={DURATION_S}"),
        ])
        .arg("-i")
        .arg(idx)
        .args([
            "-filter_complex",
            "[0:v]format=rgba[b];[b][1:s]overlay=format=rgb[v]",
        ])
        .args(["-map", "[v]", "-pix_fmt", "rgb24", "-f", "rawvideo"])
        .arg(raw)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg burned the reference frames");
    let bytes = std::fs::read(raw).expect("read reference frames");
    let stride = (W * H * 3) as usize;
    assert_eq!(bytes.len(), stride * DURATION_S as usize);
    bytes.chunks(stride).map(|c| c.to_vec()).collect()
}

// ---- g2g plumbing ----

#[derive(Default)]
struct CaptureSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");

        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn data(bytes: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

fn rgba_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(25 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

/// One decoded canvas: its PTS, its declared duration, and its pixels.
struct Canvas {
    pts_ns: u64,
    duration_ns: u64,
    rgba: Vec<u8>,
}

/// Demux the fixture's `S_VOBSUB` track and decode it, both with the real
/// elements: `mkvdemux stream=vobsub ! vobsubdec`.
async fn demux_and_decode(mkv: &[u8]) -> Vec<Canvas> {
    let mut demux = MkvDemux::new().with_stream(MkvStream::VobSub);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("mkvdemux accepts a Matroska byte stream");
    let mut demuxed = CaptureSink::default();
    for chunk in mkv.chunks(4096) {
        demux
            .process(data(chunk.to_vec(), 0), &mut demuxed)
            .await
            .expect("demux");
    }
    demux
        .process(PipelinePacket::Eos, &mut demuxed)
        .await
        .expect("demux eos");

    let mut dec = VobSubDec::new();
    dec.configure_pipeline(&Caps::SubPicture {
        format: SubPictureFormat::VobSub,
    })
    .expect("vobsubdec accepts a VobSub stream");
    let mut decoded = CaptureSink::default();
    for packet in demuxed.packets {
        dec.process(packet, &mut decoded).await.expect("decode");
    }
    decoded
        .packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(Canvas {
                pts_ns: f.timing.pts_ns,
                duration_ns: f.timing.duration_ns,
                rgba: f.domain.as_system_slice().expect("system frame").to_vec(),
            }),
            _ => None,
        })
        .collect()
}

/// Bounding box of the non-transparent pixels of an RGBA canvas, and how many
/// there are. `None` for a fully transparent one.
fn opaque_bbox(rgba: &[u8]) -> Option<(u32, u32, u32, u32, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (W, H, 0u32, 0u32);
    let mut count = 0usize;
    for y in 0..H {
        for x in 0..W {
            if rgba[((y * W + x) * 4 + 3) as usize] != 0 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
                count += 1;
            }
        }
    }
    (count > 0).then_some((x0, y0, x1, y1, count))
}

/// Bounding box of the non-black pixels of an RGB frame, and how many.
fn lit_bbox(rgb: &[u8]) -> Option<(u32, u32, u32, u32, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (W, H, 0u32, 0u32);
    let mut count = 0usize;
    for y in 0..H {
        for x in 0..W {
            let at = ((y * W + x) * 3) as usize;
            if rgb[at..at + 3] != [0, 0, 0] {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
                count += 1;
            }
        }
    }
    (count > 0).then_some((x0, y0, x1, y1, count))
}

#[test]
fn vobsubdec_and_its_gst_alias_build_from_a_launch_line() {
    let reg = default_registry();
    for name in ["vobsubdec", "dvdsubdec"] {
        assert!(
            reg.make_element(name).is_some(),
            "`{name}` builds from the registry"
        );
    }
    assert!(reg.element_names().contains(&"vobsubdec"));
    let graph = g2g_core::runtime::parse_launch(
        &reg,
        "filesrc location=x.mkv bytestream-format=matroska ! matroskademux stream=vobsub ! vobsubdec width=720 height=480 ! fakesink",
    );
    assert!(graph.is_ok(), "launch line parses: {:?}", graph.err());
}

#[tokio::test]
async fn ffmpeg_muxed_vobsub_cues_demux_and_decode_to_the_authored_rectangles() {
    if !have_ffmpeg() {
        eprintln!("skipping m899 vobsub decode: no ffmpeg on PATH");
        return;
    }
    let (idx, sub, mkv) = (
        temp_path("decode.idx"),
        temp_path("decode.sub"),
        temp_path("decode.mkv"),
    );
    author_vobsub(&idx, &sub);
    mux_mkv(&idx, &mkv);

    let canvases = demux_and_decode(&std::fs::read(&mkv).expect("read mkv")).await;
    // the opening empty canvas, then a shown / cleared pair per cue
    assert_eq!(canvases.len(), 1 + 2 * cues().len(), "canvas count");
    assert!(
        opaque_bbox(&canvases[0].rgba).is_none(),
        "the stream opens on an empty canvas"
    );

    for (i, cue) in cues().iter().enumerate() {
        let shown = &canvases[1 + i * 2];
        let cleared = &canvases[2 + i * 2];
        let pts = (cue.pts_s * 1_000_000_000.0) as u64;
        assert_eq!(shown.pts_ns, pts, "cue {i} shows at its block PTS");
        assert_eq!(
            shown.duration_ns, CUE_DURATION_NS,
            "cue {i} runs to the control sequence's stop date"
        );
        assert_eq!(
            cleared.pts_ns,
            pts + CUE_DURATION_NS,
            "cue {i} clears at its hide time"
        );
        assert!(
            opaque_bbox(&cleared.rgba).is_none(),
            "cue {i}'s clear canvas is fully transparent"
        );
        assert_eq!(
            opaque_bbox(&shown.rgba),
            Some((
                cue.x,
                cue.y,
                cue.x + cue.w - 1,
                cue.y + cue.h - 1,
                (cue.w * cue.h) as usize
            )),
            "cue {i} lands on its authored display rectangle"
        );
    }

    persist::record_evidence(
        "vobsubdec",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("vobsub")
            .detail("g2g demuxes and decodes an ffmpeg-muxed S_VOBSUB track to the authored cue rectangles and times"),
    )
    .expect("record oracle evidence");

    for p in [idx, sub, mkv] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn compositing_the_decoded_cues_matches_ffmpegs_dvdsub_burn_in() {
    if !have_ffmpeg() {
        eprintln!("skipping m899 vobsub burn-in: no ffmpeg on PATH");
        return;
    }
    let (idx, sub, mkv, raw) = (
        temp_path("burn.idx"),
        temp_path("burn.sub"),
        temp_path("burn.mkv"),
        temp_path("burn.raw"),
    );
    author_vobsub(&idx, &sub);
    mux_mkv(&idx, &mkv);
    let reference = reference_burn_in(&idx, &raw);

    let canvases = demux_and_decode(&std::fs::read(&mkv).expect("read mkv")).await;

    // The compositor is the consumer the decoder is written for: the subtitle
    // canvases are a sparse overlay input it holds between frames, so the clear
    // canvas is what ends a cue.
    let mut comp = Compositor::new(
        W,
        H,
        Vec::from([
            CompositorPad::at(0, 0),
            CompositorPad::at(0, 0).with_zorder(1),
        ]),
    );
    comp.configure_pipeline(0, &rgba_caps(W, H)).expect("base");
    comp.configure_pipeline(1, &rgba_caps(W, H)).expect("cues");

    let black: Vec<u8> = [0u8, 0, 0, 255]
        .iter()
        .cycle()
        .take((W * H * 4) as usize)
        .copied()
        .collect();
    let mut composed = CaptureSink::default();
    let mut next = 0usize;
    for second in 0..DURATION_S as u64 {
        let now = second * 1_000_000_000;
        // Deliver every canvas whose PTS has arrived, in order, then the base
        // frame that releases one composited output.
        while next < canvases.len() && canvases[next].pts_ns <= now {
            comp.process(1, data(canvases[next].rgba.clone(), now), &mut composed)
                .await
                .expect("overlay input");
            next += 1;
        }
        comp.process(0, data(black.clone(), now), &mut composed)
            .await
            .expect("base input");
    }

    let frames: Vec<Vec<u8>> = composed
        .packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => {
                Some(f.domain.as_system_slice().expect("system frame").to_vec())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        frames.len(),
        DURATION_S as usize,
        "one output per base frame"
    );

    let mut lit_frames = 0;
    for (i, (ours, theirs)) in frames.iter().zip(&reference).enumerate() {
        assert_eq!(
            lit_bbox(&rgba_to_rgb(ours)),
            lit_bbox(theirs),
            "frame {i}: g2g's composited subtitle covers the same pixels as ffmpeg's burn-in"
        );
        let mut worst = 0u8;
        for px in 0..(W * H) as usize {
            for c in 0..3 {
                worst = worst.max(ours[px * 4 + c].abs_diff(theirs[px * 3 + c]));
            }
        }
        assert!(
            worst <= 2,
            "frame {i}: worst channel difference vs ffmpeg is {worst}"
        );
        if lit_bbox(theirs).is_some() {
            lit_frames += 1;
        }
    }
    // Two frames per cue sit strictly inside its display window at 1 fps; a run
    // that burned nothing would otherwise compare two black sequences.
    assert_eq!(
        lit_frames,
        2 * cues().len(),
        "ffmpeg burned a subtitle on the expected frames"
    );

    persist::record_evidence(
        "vobsubdec",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("vobsub")
            .detail("compositing g2g's decoded cues matches ffmpeg's dvdsub overlay burn-in pixel for pixel"),
    )
    .expect("record oracle evidence");

    for p in [idx, sub, mkv, raw] {
        let _ = std::fs::remove_file(p);
    }
}

fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4).flat_map(|p| p[..3].to_vec()).collect()
}

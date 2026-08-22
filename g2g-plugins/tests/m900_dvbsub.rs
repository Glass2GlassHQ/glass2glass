//! M900: DVB subtitles (ETSI EN 300 743), validated against ffmpeg.
//!
//! ffmpeg has a `dvbsub` encoder but refuses text-to-bitmap subtitle encoding,
//! so the fixture goes the one route ffmpeg does allow: the hand-authored VobSub
//! `.idx` / `.sub` pair of the `vobsub_fixture` module transcoded bitmap to
//! bitmap (`-c:s dvbsub`) into an MPEG-TS with a real `subtitling_descriptor`,
//! and that transport stream remuxed to Matroska so the `S_DVBSUB` track carries
//! the `CodecPrivate` page ids ffmpeg's own demuxer synthesizes. Everything
//! about the DVB bitstream, its segments, palette and page ids is ffmpeg's, and
//! its `dvbsub` decoder renders the same cues through `overlay` for the
//! pixel-for-pixel comparison against g2g's decode composited by `Compositor`.
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Bus, BusMessage, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement,
    OutputSink, PushOutcome, Rate, RawVideoFormat, StreamType, SubPictureFormat,
};
use g2g_plugins::compositor::{Compositor, CompositorPad};
use g2g_plugins::conformance::persist;
use g2g_plugins::dvbsub::PageIds;
use g2g_plugins::dvbsubdec::DvbSubDec;
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};
use g2g_plugins::registry::default_registry;
use g2g_plugins::tsdemux::{TsDemux, TsStream};

mod vobsub_fixture;
use vobsub_fixture::{author_vobsub, cues, have_ffmpeg, H, W};

/// Fixture length in seconds, and therefore the reference burn-in's frame count
/// at one frame per second.
const DURATION_S: u32 = 9;

/// The page ids ffmpeg's `subtitling_descriptor` declares for its single
/// subtitle stream, and the type byte beside them (DVB subtitles, no monitor
/// aspect ratio criticality).
const PAGE_IDS: PageIds = PageIds {
    composition: 1,
    ancillary: 1,
};
const SUBTITLING_TYPE: u8 = 0x10;

/// The transport stream's own timebase start: ffmpeg's mux places the first
/// video frame at 1.6 s, and every subtitle PTS rides that offset. The Matroska
/// remux normalizes it away, which is why the burn-in comparison uses the mkv.
const TS_START_NS: u64 = 1_600_000_000;

// ---- fixture ----

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m900-{}-{name}", std::process::id()))
}

/// Transcode the authored VobSub cues to DVB subtitles in an MPEG-TS, then remux
/// that to Matroska. Both files' subtitle bitstreams are ffmpeg's `dvbsub`
/// encoder's; only the TS carries the `subtitling_descriptor`, and the remux is
/// what turns it into the `S_DVBSUB` `CodecPrivate`.
fn author_fixture(idx: &PathBuf, ts: &PathBuf, mkv: &PathBuf) {
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
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:s", "dvbsub"])
        .arg(ts)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg transcoded dvdsub to dvbsub");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(ts)
        .args(["-map", "0", "-c", "copy"])
        .arg(mkv)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg remuxed the S_DVBSUB fixture");
}

/// ffmpeg's own burn-in: its `dvbsub` decoder rendered through `overlay` onto
/// black, one RGB frame per second. Blending happens in RGB (not the default
/// YUV) so chroma subsampling cannot smear the reference.
fn reference_burn_in(mkv: &PathBuf, raw: &PathBuf) -> Vec<Vec<u8>> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args([
            "-f",
            "lavfi",
            "-i",
            &format!("color=c=black:s={W}x{H}:r=1:d={DURATION_S}"),
        ])
        .arg("-i")
        .arg(mkv)
        .args([
            "-filter_complex",
            "[0:v]format=rgba[b];[b][1:s]overlay=format=rgb[v]",
        ])
        .args(["-map", "[v]", "-pix_fmt", "rgb24"])
        // The decoder's 30 s page_time_out keeps the sub2video input alive long
        // past the video, so the output is bounded to the video's own frames.
        .args(["-frames:v", &DURATION_S.to_string(), "-f", "rawvideo"])
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

/// The `DataFrame` payloads and `CapsChanged` a demuxer emitted.
struct Demuxed {
    caps: Vec<Caps>,
    frames: Vec<(u64, Vec<u8>)>,
}

fn split(packets: Vec<PipelinePacket>) -> Demuxed {
    let mut caps = Vec::new();
    let mut frames = Vec::new();
    for p in packets {
        match p {
            PipelinePacket::CapsChanged(c) => caps.push(c),
            PipelinePacket::DataFrame(f) => frames.push((
                f.timing.pts_ns,
                f.domain.as_system_slice().expect("system frame").to_vec(),
            )),
            _ => {}
        }
    }
    Demuxed { caps, frames }
}

/// Demux the transport stream's DVB subtitle PID, with a bus so the PMT's
/// `StreamCollection` is observable too.
async fn demux_ts(ts: &[u8]) -> (Demuxed, Vec<g2g_core::Stream>) {
    let (bus, handle) = Bus::new(64);
    let mut demux = TsDemux::new()
        .with_stream(TsStream::DvbSub)
        .with_bus(handle);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
        .expect("tsdemux accepts an MPEG-TS byte stream");
    let mut out = CaptureSink::default();
    for chunk in ts.chunks(4096) {
        demux
            .process(data(chunk.to_vec(), 0), &mut out)
            .await
            .expect("demux");
    }
    demux
        .process(PipelinePacket::Eos, &mut out)
        .await
        .expect("demux eos");
    let streams = core::iter::from_fn(|| bus.try_recv())
        .filter_map(|m| match m {
            BusMessage::StreamCollection(c) => Some(c),
            _ => None,
        })
        .flat_map(|c| c.streams().to_vec())
        .collect();
    (split(out.packets), streams)
}

/// Demux the Matroska file's `S_DVBSUB` track.
async fn demux_mkv(mkv: &[u8]) -> Demuxed {
    let mut demux = MkvDemux::new().with_stream(MkvStream::DvbSub);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("mkvdemux accepts a Matroska byte stream");
    let mut out = CaptureSink::default();
    for chunk in mkv.chunks(4096) {
        demux
            .process(data(chunk.to_vec(), 0), &mut out)
            .await
            .expect("demux");
    }
    demux
        .process(PipelinePacket::Eos, &mut out)
        .await
        .expect("demux eos");
    split(out.packets)
}

/// One decoded canvas: its PTS and its pixels.
struct Canvas {
    pts_ns: u64,
    rgba: Vec<u8>,
}

/// Run the real decoder over a demuxer's frames.
async fn decode(frames: Vec<(u64, Vec<u8>)>) -> Vec<Canvas> {
    let mut dec = DvbSubDec::new();
    dec.configure_pipeline(&Caps::SubPicture {
        format: SubPictureFormat::DvbSub,
    })
    .expect("dvbsubdec accepts a DVB subtitle stream");
    let mut out = CaptureSink::default();
    for (pts_ns, bytes) in frames {
        dec.process(data(bytes, pts_ns), &mut out)
            .await
            .expect("decode");
    }
    dec.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("decode eos");
    out.packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(Canvas {
                pts_ns: f.timing.pts_ns,
                rgba: f.domain.as_system_slice().expect("system frame").to_vec(),
            }),
            _ => None,
        })
        .collect()
}

/// Bounding box of the non-transparent pixels of an RGBA canvas, and how many
/// there are. `None` for a fully transparent one.
fn opaque_bbox(rgba: &[u8]) -> Option<(u32, u32, u32, u32, usize)> {
    bbox(rgba, 4, |px| px[3] != 0)
}

/// Bounding box of the non-black pixels of an RGB frame, and how many.
fn lit_bbox(rgb: &[u8]) -> Option<(u32, u32, u32, u32, usize)> {
    bbox(rgb, 3, |px| px[..3] != [0, 0, 0])
}

fn bbox(
    buf: &[u8],
    bpp: usize,
    lit: impl Fn(&[u8]) -> bool,
) -> Option<(u32, u32, u32, u32, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (W, H, 0u32, 0u32);
    let mut count = 0usize;
    for y in 0..H {
        for x in 0..W {
            let at = (y * W + x) as usize * bpp;
            if lit(&buf[at..at + bpp]) {
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

/// The rectangle each authored cue occupies on the display, as the DVB region
/// the transcode produced: the same rectangle, since ffmpeg's dvbsub encoder
/// keeps the source subpicture's placement.
fn cue_rects() -> Vec<(u32, u32, u32, u32, usize)> {
    cues()
        .iter()
        .map(|c| (c.x, c.y, c.x + c.w - 1, c.y + c.h - 1, (c.w * c.h) as usize))
        .collect()
}

/// Assert the canvases are the opening empty one, then a shown / clear pair per
/// cue at `times`.
fn assert_cue_canvases(canvases: &[Canvas], times: &[(u64, u64)]) {
    assert_eq!(canvases.len(), 1 + 2 * times.len(), "canvas count");
    assert!(
        opaque_bbox(&canvases[0].rgba).is_none(),
        "the stream opens on an empty canvas"
    );
    for (i, (rect, (shown_ns, cleared_ns))) in cue_rects().iter().zip(times).enumerate() {
        let shown = &canvases[1 + i * 2];
        let cleared = &canvases[2 + i * 2];
        assert_eq!(shown.pts_ns, *shown_ns, "cue {i} shows at its display set");
        assert_eq!(
            opaque_bbox(&shown.rgba),
            Some(*rect),
            "cue {i} lands on the transcoded region rectangle"
        );
        assert_eq!(cleared.pts_ns, *cleared_ns, "cue {i} clears at its end");
        assert!(
            opaque_bbox(&cleared.rgba).is_none(),
            "cue {i}'s clear canvas is fully transparent"
        );
    }
}

// ---- tests ----

#[test]
fn dvbsubdec_builds_from_a_launch_line() {
    let reg = default_registry();
    assert!(reg.make_element("dvbsubdec").is_some());
    assert!(reg.element_names().contains(&"dvbsubdec"));
    // No gst alias: gst's `dvbsuboverlay` is a video-overlay convenience
    // element, not a bare decoder, so nothing here should answer to that name.
    assert!(
        reg.make_element("dvbsuboverlay").is_none(),
        "`dvbsuboverlay` is a video overlay, not this decoder"
    );
    let graph = g2g_core::runtime::parse_launch(
        &reg,
        "filesrc location=x.ts bytestream-format=mpegts ! tsdemux stream=dvbsub ! dvbsubdec width=1920 height=1080 page-id=1 ! fakesink",
    );
    assert!(graph.is_ok(), "launch line parses: {:?}", graph.err());
}

#[tokio::test]
async fn tsdemux_selects_the_subtitling_descriptor_stream_and_forwards_its_page_ids() {
    if !have_ffmpeg() {
        eprintln!("skipping m900 tsdemux selection: no ffmpeg on PATH");
        return;
    }
    let (idx, sub, ts, mkv) = (
        temp_path("sel.idx"),
        temp_path("sel.sub"),
        temp_path("sel.ts"),
        temp_path("sel.mkv"),
    );
    author_vobsub(&idx, &sub);
    author_fixture(&idx, &ts, &mkv);

    let (demuxed, streams) = demux_ts(&std::fs::read(&ts).expect("read ts")).await;
    let subs: Vec<_> = streams
        .iter()
        .filter(|s| {
            s.caps
                == Caps::SubPicture {
                    format: SubPictureFormat::DvbSub,
                }
        })
        .collect();
    assert_eq!(
        subs.len(),
        1,
        "the PMT's private stream with a subtitling_descriptor is a DVB subtitle stream"
    );
    assert_eq!(subs[0].stream_type, StreamType::Text);

    // The descriptor's page ids reach the decoder in band, ahead of the cues.
    assert_eq!(
        demuxed.frames[0].1,
        g2g_plugins::dvbsub::page_id_blob(PAGE_IDS, SUBTITLING_TYPE),
        "the first frame is the subtitling_descriptor's page ids"
    );
    assert_eq!(
        demuxed.frames.len(),
        1 + 4,
        "then the four display sets ffmpeg encoded"
    );
    // Each display set is a PES data field: data_identifier then subtitle_stream_id.
    for (_, payload) in &demuxed.frames[1..] {
        assert_eq!(&payload[..2], &[0x20, 0x00], "PES data field header");
    }

    for p in [idx, sub, ts, mkv] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn mkvdemux_selects_the_s_dvbsub_track_and_forwards_its_codec_private() {
    if !have_ffmpeg() {
        eprintln!("skipping m900 mkvdemux selection: no ffmpeg on PATH");
        return;
    }
    let (idx, sub, ts, mkv) = (
        temp_path("mkvsel.idx"),
        temp_path("mkvsel.sub"),
        temp_path("mkvsel.ts"),
        temp_path("mkvsel.mkv"),
    );
    author_vobsub(&idx, &sub);
    author_fixture(&idx, &ts, &mkv);

    let demuxed = demux_mkv(&std::fs::read(&mkv).expect("read mkv")).await;
    assert_eq!(
        demuxed.caps,
        vec![Caps::SubPicture {
            format: SubPictureFormat::DvbSub
        }],
        "the S_DVBSUB track types as a DVB subtitle stream"
    );
    assert_eq!(
        demuxed.frames[0].1,
        g2g_plugins::dvbsub::page_id_blob(PAGE_IDS, SUBTITLING_TYPE),
        "the CodecPrivate page ids go out ahead of the display sets"
    );
    assert_eq!(demuxed.frames.len(), 1 + 4);
    // A Matroska block is the bare segment stream, without the PES data field.
    for (_, payload) in &demuxed.frames[1..] {
        assert_eq!(payload[0], 0x0f, "the block starts on a segment sync byte");
    }

    for p in [idx, sub, ts, mkv] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn both_carriages_decode_to_the_same_cue_canvases() {
    if !have_ffmpeg() {
        eprintln!("skipping m900 decode: no ffmpeg on PATH");
        return;
    }
    let (idx, sub, ts, mkv) = (
        temp_path("dec.idx"),
        temp_path("dec.sub"),
        temp_path("dec.ts"),
        temp_path("dec.mkv"),
    );
    author_vobsub(&idx, &sub);
    author_fixture(&idx, &ts, &mkv);

    // The Matroska remux normalizes the transport stream's start offset away, so
    // its cue times are the authored ones and the TS's are those plus the offset.
    let mkv_times: Vec<(u64, u64)> = Vec::from([
        (1_500_000_000, 3_548_000_000),
        (5_500_000_000, 7_548_000_000),
    ]);
    let ts_times: Vec<(u64, u64)> = mkv_times
        .iter()
        .map(|(a, b)| (a + TS_START_NS, b + TS_START_NS))
        .collect();

    let from_mkv = demux_mkv(&std::fs::read(&mkv).expect("read mkv")).await;
    assert_cue_canvases(&decode(from_mkv.frames).await, &mkv_times);

    let (from_ts, _) = demux_ts(&std::fs::read(&ts).expect("read ts")).await;
    assert_cue_canvases(&decode(from_ts.frames).await, &ts_times);

    persist::record_evidence(
        "dvbsubdec",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("dvbsub")
            .detail("g2g decodes ffmpeg's dvbsub display sets to the authored cue rectangles out of both the MPEG-TS and Matroska carriages"),
    )
    .expect("record oracle evidence");

    for p in [idx, sub, ts, mkv] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn compositing_the_decoded_cues_matches_ffmpegs_dvbsub_burn_in() {
    if !have_ffmpeg() {
        eprintln!("skipping m900 burn-in: no ffmpeg on PATH");
        return;
    }
    let (idx, sub, ts, mkv, raw) = (
        temp_path("burn.idx"),
        temp_path("burn.sub"),
        temp_path("burn.ts"),
        temp_path("burn.mkv"),
        temp_path("burn.raw"),
    );
    author_vobsub(&idx, &sub);
    author_fixture(&idx, &ts, &mkv);
    let reference = reference_burn_in(&mkv, &raw);

    let demuxed = demux_mkv(&std::fs::read(&mkv).expect("read mkv")).await;
    let canvases = decode(demuxed.frames).await;

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
        // g2g runs the same BT.601 fixed-point CLUT conversion the reference
        // decoder does, so the burned pixels are identical, not merely close.
        assert_eq!(
            worst, 0,
            "frame {i}: g2g's rendered colours differ from ffmpeg's"
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
        "dvbsubdec",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("dvbsub")
            .detail("compositing g2g's decoded display sets matches ffmpeg's dvbsub overlay burn-in pixel for pixel"),
    )
    .expect("record oracle evidence");

    for p in [idx, sub, ts, mkv, raw] {
        let _ = std::fs::remove_file(p);
    }
}

fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| p[..3].to_vec())
        .collect()
}

//! M929: MPEG program stream (`.mpg` / `.vob`) demuxing, MPEG-1/2 video, and the
//! DVD subpicture track a VOB carries.
//!
//! The fixtures are ffmpeg's own program streams: an MPEG-1 `.mpg` (MPEG-1 packs
//! and MPEG-1 PES headers) and an MPEG-2 `.vob` (MPEG-2 packs, MPEG-2 PES, a
//! `private_stream_2` navigation stream, and a subpicture track stream-copied
//! from the hand-authored `.idx` / `.sub` pair of the `vobsub_fixture` module).
//! ffmpeg cannot encode text subtitles to `dvdsub`, so the subpicture bytes are
//! authored here and copied into the VOB rather than transcoded.
//!
//! What is asserted: the PS magic types, both pack / PES flavours demux, the
//! sequence header fixes the video caps, timestamps are monotonic, the first
//! access unit is a keyframe, the subpicture track reassembles and renders
//! through `vobsubdec` on its default palette, MPEG-2 in MPEG-TS reaches
//! `VideoCodec::Mpeg2`, and malformed input never panics.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::frame::{Frame, PipelinePacket};
use g2g_core::{frame::FrameTiming, memory::SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, Dim, G2gError, MemoryDomain, OutputSink,
    PadDirection, PadTemplates, PropValue, PushOutcome, Rate, SubPictureFormat, VideoCodec,
};
use g2g_plugins::psdemux::{
    forwardable_streams, parse_sequence_header, subpicture_streams, PsDemux, PsDemuxer, PsStream,
};
use g2g_plugins::registry::default_registry;
use g2g_plugins::tsdemux::{TsDemux, TsStream};
use g2g_plugins::typefind::sniff;
use g2g_plugins::vobsub::parse_idx;
use g2g_plugins::vobsubdec::VobSubDec;

mod vobsub_fixture;
use vobsub_fixture::{author_vobsub, cues, have_ffmpeg, CUE_DURATION_NS};

/// The VOB's video geometry: NTSC, so it differs from both the fixture `.idx`'s
/// PAL `size:` line and `vobsubdec`'s own 720x576 default. The cue rectangles
/// the fixture authors all fit inside it, so a demuxer that failed to pass the
/// video's geometry through would render on the wrong canvas and be caught.
const VOB_W: u32 = 720;
const VOB_H: u32 = 480;

// ---- harness ----

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        self.packets.push(packet);
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

impl Collect {
    fn frames(&self) -> Vec<&Frame> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    fn caps_changes(&self) -> Vec<Caps> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m929-{}-{name}", std::process::id()))
}

fn ffmpeg(args: &[&str]) {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .output()
        .expect("run ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn data_frame(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

fn ps_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegPs,
    }
}

fn bytes(frame: &Frame) -> Vec<u8> {
    frame
        .domain
        .as_system_slice()
        .expect("system frame")
        .to_vec()
}

/// Run the whole file through one `PsDemux` selection, in 4 KiB chunks so the
/// packet realignment across input frames runs too.
async fn demux(file: &[u8], stream: PsStream) -> Collect {
    let mut el = PsDemux::new().with_stream(stream);
    el.configure_pipeline(&ps_caps())
        .expect("mpegpsdemux accepts a program stream");
    let mut sink = Collect::default();
    for chunk in file.chunks(4096) {
        el.process(data_frame(chunk.to_vec()), &mut sink)
            .await
            .expect("demux a chunk");
    }
    el.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("flush at end of stream");
    sink
}

/// An MPEG-1 program stream: MPEG-1 packs, MPEG-1 PES headers, mpeg1video + mp2.
fn author_mpg(path: &Path, seconds: &str) {
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size=352x288:rate=25:duration={seconds}"),
        "-f",
        "lavfi",
        "-i",
        &format!("sine=frequency=440:duration={seconds}"),
        "-c:v",
        "mpeg1video",
        "-b:v",
        "1000k",
        "-g",
        "12",
        "-c:a",
        "mp2",
        "-b:a",
        "128k",
        "-f",
        "mpeg",
        path.to_str().unwrap(),
    ]);
}

/// An MPEG-2 program stream carrying the authored subpicture track: MPEG-2
/// packs, MPEG-2 PES, `private_stream_2` navigation packets, and a `dvdsub`
/// stream copied straight from the fixture pair. The video is authored at the
/// fixture's own display size, which is what the demuxer's synthesized `.idx`
/// then declares.
fn author_vob(path: &Path, idx: &PathBuf, sub: &PathBuf) {
    author_vobsub(idx, sub);
    // Derived from the caller's own path: the tests run concurrently, so a
    // shared intermediate name would be deleted out from under another one.
    let video = path.with_extension("src.mpg");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={VOB_W}x{VOB_H}:rate=25:duration=8"),
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=8",
        "-c:v",
        "mpeg2video",
        "-b:v",
        "2000k",
        "-g",
        "12",
        "-c:a",
        "mp2",
        "-b:a",
        "128k",
        "-f",
        "mpeg",
        video.to_str().unwrap(),
    ]);
    ffmpeg(&[
        "-i",
        idx.to_str().unwrap(),
        "-i",
        video.to_str().unwrap(),
        "-map",
        "1:v",
        "-map",
        "1:a",
        "-map",
        "0:s",
        "-c",
        "copy",
        "-f",
        "dvd",
        path.to_str().unwrap(),
    ]);
    let _ = std::fs::remove_file(video);
}

// ---- typefind + launch ----

#[test]
fn the_pack_start_code_types_a_program_stream() {
    let mut header = Vec::from([0x00u8, 0x00, 0x01, 0xBA, 0x44, 0x00, 0x04, 0x00]);
    header.resize(512, 0);
    assert_eq!(sniff(&header), Some(ByteStreamEncoding::MpegPs));
    // The prefix alone is not a pack: a bare start code prefix must not type.
    assert_eq!(sniff(&[0x00, 0x00, 0x01, 0xE0, 0x00, 0x10]), None);
    assert_eq!(sniff(&[]), None);
}

#[test]
fn mpegpsdemux_builds_from_a_launch_line() {
    let reg = default_registry();
    assert!(reg.element_names().contains(&"mpegpsdemux"));
    let graph = g2g_core::runtime::parse_launch(
        &reg,
        "filesrc location=movie.vob ! mpegpsdemux stream=subpicture ! vobsubdec ! fakesink",
    );
    assert!(graph.is_ok(), "launch line parses: {:?}", graph.err());
}

#[test]
fn the_stream_property_round_trips() {
    let mut el = PsDemux::new();
    assert!(el.properties().iter().any(|s| s.name == "stream"));
    for (name, want) in [
        ("mpeg2", PsStream::Mpeg2),
        ("mp2", PsStream::Mp2),
        ("ac3", PsStream::Ac3),
        ("subpicture", PsStream::SubPicture),
    ] {
        el.set_property("stream", PropValue::Str(name.into()))
            .expect("a known selection is accepted");
        assert_eq!(el.stream(), want);
        assert_eq!(el.get_property("stream"), Some(PropValue::Str(name.into())));
    }
    assert!(el
        .set_property("stream", PropValue::Str("lpcm".into()))
        .is_err());
}

// ---- MPEG-1 program stream ----

#[tokio::test]
async fn an_mpeg1_program_stream_demuxes_video_and_audio() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let path = temp_path("mpeg1.mpg");
    author_mpg(&path, "3");
    let file = std::fs::read(&path).expect("read the fixture");

    // Stream discovery: probing the file finds one video and one audio stream.
    let mut probe = PsDemuxer::new();
    probe.push_data(&file);
    let infos = forwardable_streams(&probe);
    assert_eq!(infos.len(), 2, "video + audio discovered: {infos:?}");
    assert!(infos[0].video && infos[0].stream == PsStream::Mpeg2);
    assert_eq!(infos[1].stream, PsStream::Mp2);

    let video = demux(&file, PsStream::Mpeg2).await;
    let frames = video.frames();
    assert!(frames.len() >= 20, "access units: {}", frames.len());

    // The sequence header fixes the caps, and never at an unfixatable Any.
    let caps = video.caps_changes();
    assert_eq!(
        caps.first(),
        Some(&Caps::CompressedVideo {
            codec: VideoCodec::Mpeg2,
            width: Dim::Fixed(352),
            height: Dim::Fixed(288),
            framerate: Rate::Fixed(25 << 16),
        }),
        "the sequence header's geometry is announced once, concretely"
    );
    assert_eq!(caps.len(), 1, "and is not re-announced: {caps:?}");

    // The first access unit opens on a sequence header, so it is a sync point.
    assert!(
        bytes(frames[0]).starts_with(&[0x00, 0x00, 0x01, 0xB3]),
        "the first unit opens the sequence"
    );
    assert!(
        parse_sequence_header(&bytes(frames[0])).is_some(),
        "the first unit carries a decodable sequence header"
    );

    let pts: Vec<u64> = frames.iter().map(|f| f.timing.pts_ns).collect();
    assert!(
        pts.windows(2).all(|w| w[1] >= w[0]),
        "video timestamps are monotonic: {:?}",
        &pts[..8.min(pts.len())]
    );
    assert!(pts.last().unwrap() > &2_000_000_000, "the clip runs 3s");

    let audio = demux(&file, PsStream::Mp2).await;
    let aframes = audio.frames();
    assert!(!aframes.is_empty(), "mp2 packets forwarded");
    assert_eq!(
        audio.caps_changes(),
        Vec::new(),
        "an audio pad has nothing to refine"
    );
    let apts: Vec<u64> = aframes.iter().map(|f| f.timing.pts_ns).collect();
    assert!(apts.windows(2).all(|w| w[1] >= w[0]), "audio pts monotonic");
    let _ = std::fs::remove_file(path);
}

// ---- MPEG-2 VOB, subpictures included ----

#[tokio::test]
async fn a_vob_demuxes_video_audio_and_subpictures() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let (vob, idx, sub) = (temp_path("sub.vob"), temp_path("f.idx"), temp_path("f.sub"));
    author_vob(&vob, &idx, &sub);
    let file = std::fs::read(&vob).expect("read the VOB");

    // The navigation (`private_stream_2`) packets a VOB carries are skipped, so
    // discovery names only the three real streams.
    let mut probe = PsDemuxer::new();
    probe.push_data(&file);
    assert_eq!(
        probe.streams().len(),
        3,
        "video + audio + subpicture: {:?}",
        probe.streams()
    );
    let infos = forwardable_streams(&probe);
    assert_eq!(infos.len(), 2, "the A/V fan-out leaves subpictures out");
    assert_eq!(
        subpicture_streams(&probe).len(),
        1,
        "and the subpicture track is discoverable on its own"
    );

    let video = demux(&file, PsStream::Mpeg2).await;
    assert_eq!(
        video.caps_changes().first(),
        Some(&Caps::CompressedVideo {
            codec: VideoCodec::Mpeg2,
            width: Dim::Fixed(VOB_W),
            height: Dim::Fixed(VOB_H),
            framerate: Rate::Fixed(25 << 16),
        })
    );
    assert!(video.frames().len() >= 100, "8s of 25fps video");

    let subs = demux(&file, PsStream::SubPicture).await;
    let frames = subs.frames();
    let expected = cues();
    assert_eq!(
        frames.len(),
        1 + expected.len(),
        "the synthesized .idx, then one frame per cue"
    );

    // The pad opens on a `.idx` text carrying the video's own geometry, which is
    // the canvas the cues are placed on. No palette: the decoder's default one
    // renders a program stream's palette-less cues.
    let config = parse_idx(&bytes(frames[0])).expect("the first frame is .idx text");
    assert_eq!(
        config.size,
        Some((VOB_W, VOB_H)),
        "the video's own geometry, not the decoder's default"
    );
    assert_eq!(config.palette, None);

    for (i, cue) in expected.iter().enumerate() {
        let frame = frames[1 + i];
        let spu = bytes(frame);
        assert_eq!(
            u16::from_be_bytes([spu[0], spu[1]]) as usize,
            spu.len(),
            "cue {i} is delimited by its own declared size"
        );
        assert_eq!(
            frame.timing.duration_ns, CUE_DURATION_NS,
            "cue {i} carries the unit's own hide time"
        );
        // ffmpeg rebases the muxed timestamps, so the spacing is what is stable.
        if i > 0 {
            let dt = frame.timing.pts_ns - frames[i].timing.pts_ns;
            let want = ((cue.pts_s - expected[i - 1].pts_s) * 1e9) as u64;
            assert!(
                dt.abs_diff(want) < 50_000_000,
                "cue {i} keeps its spacing: {dt} vs {want}"
            );
        }
    }
    for p in [vob, idx, sub] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn a_demuxed_subpicture_renders_on_the_default_palette() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let (vob, idx, sub) = (
        temp_path("render.vob"),
        temp_path("r.idx"),
        temp_path("r.sub"),
    );
    author_vob(&vob, &idx, &sub);
    let file = std::fs::read(&vob).expect("read the VOB");
    let demuxed = demux(&file, PsStream::SubPicture).await;

    let mut dec = VobSubDec::new();
    dec.configure_pipeline(&Caps::SubPicture {
        format: SubPictureFormat::VobSub,
    })
    .expect("vobsubdec accepts a subpicture pad");
    let mut sink = Collect::default();
    for packet in demuxed.packets {
        if let PipelinePacket::DataFrame(f) = packet {
            let timing = f.timing;
            let data = bytes(&f);
            dec.process(
                PipelinePacket::DataFrame(Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
                    timing,
                    0,
                )),
                &mut sink,
            )
            .await
            .expect("decode a cue");
        }
    }

    // The canvas geometry follows the `.idx` the demuxer synthesized.
    assert!(
        sink.caps_changes().iter().any(|c| matches!(
            c,
            Caps::RawVideo {
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } if *w == VOB_W && *h == VOB_H
        )),
        "the decoder retargets to the demuxed geometry: {:?}",
        sink.caps_changes()
    );

    let expected = cues();
    let canvases = sink.frames();
    assert!(
        canvases.len() > expected.len(),
        "the opening blank canvas plus a canvas per cue: {}",
        canvases.len()
    );
    // Each cue is a painted canvas then a clearing one, so the last cue's
    // picture is the frame before the last: opaque pixels only inside the
    // authored rectangle, and nothing left standing after it.
    let last = bytes(canvases[canvases.len() - 2]);
    assert_eq!(last.len(), (VOB_W * VOB_H * 4) as usize);
    assert!(
        bytes(canvases[canvases.len() - 1])
            .chunks(4)
            .all(|p| p[3] == 0),
        "the cue is cleared at its hide time"
    );
    let cue = &expected[expected.len() - 1];
    let mut opaque = 0usize;
    for y in 0..VOB_H {
        for x in 0..VOB_W {
            if last[((y * VOB_W + x) * 4 + 3) as usize] == 0 {
                continue;
            }
            opaque += 1;
            assert!(
                x >= cue.x && x < cue.x + cue.w && y >= cue.y && y < cue.y + cue.h,
                "an opaque pixel at ({x},{y}) is outside the authored rectangle"
            );
        }
    }
    assert!(opaque > 1000, "the cue paints its box: {opaque} pixels");
    for p in [vob, idx, sub] {
        let _ = std::fs::remove_file(p);
    }
}

// ---- MPEG-2 over MPEG-TS ----

#[tokio::test]
async fn mpeg2_video_in_a_transport_stream_reaches_the_mpeg2_selection() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let path = temp_path("mpeg2.ts");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=25:duration=2",
        "-c:v",
        "mpeg2video",
        "-b:v",
        "800k",
        "-g",
        "12",
        "-f",
        "mpegts",
        path.to_str().unwrap(),
    ]);
    let file = std::fs::read(&path).expect("read the transport stream");

    let mut el = TsDemux::new().with_stream(TsStream::Mpeg2);
    el.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .expect("tsdemux accepts a transport stream");
    let mut sink = Collect::default();
    for chunk in file.chunks(4096) {
        el.process(data_frame(chunk.to_vec()), &mut sink)
            .await
            .expect("demux a chunk");
    }
    el.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("flush");

    let frames = sink.frames();
    assert!(frames.len() >= 20, "access units: {}", frames.len());
    assert!(
        parse_sequence_header(&bytes(frames[0])).is_some(),
        "the first unit carries the sequence header"
    );
    // The PMT names stream_type 0x02, which the fan-out reports as video.
    let mut probe = g2g_plugins::mpegts::TsDemuxer::new();
    let mut off = 0;
    while off + 188 <= file.len() {
        if file[off] == 0x47 {
            probe.push_packet(&file[off..off + 188]);
        }
        off += 188;
    }
    let infos = g2g_plugins::tsdemux::forwardable_streams(&probe);
    assert!(
        infos.iter().any(|i| i.stream == TsStream::Mpeg2
            && i.video
            && matches!(
                i.caps,
                Caps::CompressedVideo {
                    codec: VideoCodec::Mpeg2,
                    ..
                }
            )),
        "the PMT's MPEG-2 stream is a video branch: {infos:?}"
    );
    let _ = std::fs::remove_file(path);
}

// ---- launch fan-out ----

/// A named `mpegpsdemux` fans a VOB out to a video, an audio and a subpicture
/// pad. The subpicture one is what MPEG-TS has no equivalent of: a program
/// stream's subtitle track is selectable as `d.text_0`.
#[test]
fn named_pads_fan_a_vob_out_including_the_subpicture_track() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let (vob, idx, sub) = (temp_path("fan.vob"), temp_path("n.idx"), temp_path("n.sub"));
    author_vob(&vob, &idx, &sub);
    let p = vob.display();
    let reg = default_registry();
    let line = format!(
        "filesrc location={p} bytestream-format=mpegps ! mpegpsdemux name=d  \
         d.video_0 ! fakesink  d.audio_0 ! fakesink  d.text_0 ! vobsubdec ! fakesink"
    );
    let graph = g2g_core::runtime::parse_launch(&reg, &line)
        .unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let vg = graph.finish().expect("valid graph");
    let demuxes: Vec<g2g_core::graph::NodeKind> = vg
        .topo()
        .iter()
        .map(|&n| vg.kind(n))
        .filter(|k| matches!(k, g2g_core::graph::NodeKind::Tee(_)))
        .collect();
    assert_eq!(
        demuxes,
        [g2g_core::graph::NodeKind::Tee(3)],
        "video + audio + subpicture ports"
    );
    for path in [vob, idx, sub] {
        let _ = std::fs::remove_file(path);
    }
}

/// `.vob` / `.mpg` type by extension, so a bare launch line needs no
/// `bytestream-format=`.
#[test]
fn the_file_extensions_type_a_program_stream() {
    let reg = default_registry();
    for name in ["movie.vob", "clip.mpg", "clip.MPEG"] {
        let line = format!("filesrc location={name} ! mpegpsdemux ! fakesink");
        assert!(
            g2g_core::runtime::parse_launch(&reg, &line).is_ok(),
            "{name} types as a program stream"
        );
    }
}

// ---- decode ----

#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
#[tokio::test]
async fn a_demuxed_program_stream_decodes_to_frames() {
    use g2g_plugins::ffmpegdec::FfmpegVideoDec;

    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let path = temp_path("decode.mpg");
    author_mpg(&path, "1");
    let file = std::fs::read(&path).expect("read the fixture");
    let demuxed = demux(&file, PsStream::Mpeg2).await;

    let mut dec = FfmpegVideoDec::new();
    dec.configure_pipeline(&Caps::CompressedVideo {
        codec: VideoCodec::Mpeg2,
        width: Dim::Fixed(352),
        height: Dim::Fixed(288),
        framerate: Rate::Fixed(25 << 16),
    })
    .expect("libavcodec opens the MPEG-2 decoder");
    let mut sink = Collect::default();
    for packet in demuxed.packets {
        if let PipelinePacket::DataFrame(f) = packet {
            let data = bytes(&f);
            dec.process(data_frame(data), &mut sink)
                .await
                .expect("decode an access unit");
        }
    }
    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain the decoder");

    let frames = sink.frames();
    assert!(frames.len() >= 20, "decoded frames: {}", frames.len());
    // 352x288 4:2:0 is 1.5 bytes per pixel.
    assert_eq!(bytes(frames[0]).len(), 352 * 288 * 3 / 2);
    let _ = std::fs::remove_file(path);
}

/// The MPEG-2 half of the same path: a VOB's video decodes as well as an
/// MPEG-1 `.mpg`'s, and `playbin uri=` builds the whole graph for it.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
#[tokio::test]
async fn a_vob_decodes_and_playbin_builds_its_graph() {
    use g2g_plugins::ffmpegdec::FfmpegVideoDec;

    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let (vob, idx, sub) = (temp_path("dec.vob"), temp_path("d.idx"), temp_path("d.sub"));
    author_vob(&vob, &idx, &sub);
    let file = std::fs::read(&vob).expect("read the VOB");
    let demuxed = demux(&file, PsStream::Mpeg2).await;

    let mut dec = FfmpegVideoDec::new();
    dec.configure_pipeline(&Caps::CompressedVideo {
        codec: VideoCodec::Mpeg2,
        width: Dim::Fixed(VOB_W),
        height: Dim::Fixed(VOB_H),
        framerate: Rate::Fixed(25 << 16),
    })
    .expect("libavcodec opens the MPEG-2 decoder");
    let mut sink = Collect::default();
    for packet in demuxed.packets {
        if let PipelinePacket::DataFrame(f) = packet {
            let data = bytes(&f);
            dec.process(data_frame(data), &mut sink)
                .await
                .expect("decode an access unit");
        }
    }
    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain the decoder");
    let frames = sink.frames();
    assert!(frames.len() >= 150, "8s of 25fps: {}", frames.len());
    assert_eq!(
        bytes(frames[0]).len(),
        (VOB_W * VOB_H) as usize * 3 / 2,
        "the decoder ran at the sequence header's geometry"
    );

    // The A/V fan-out: `playbin uri=file://x.vob` builds video and audio
    // branches, the subpicture track staying off the auto-plugged path.
    let reg = default_registry();
    let line = format!("playbin uri=file://{}", vob.display());
    let graph = g2g_core::runtime::parse_launch(&reg, &line)
        .unwrap_or_else(|e| panic!("playbin builds `{line}`: {e}"));
    let vg = graph.finish().expect("valid graph");
    let demuxes: Vec<g2g_core::graph::NodeKind> = vg
        .topo()
        .iter()
        .map(|&n| vg.kind(n))
        .filter(|k| matches!(k, g2g_core::graph::NodeKind::Tee(_)))
        .collect();
    assert_eq!(
        demuxes,
        [g2g_core::graph::NodeKind::Tee(2)],
        "video + audio branches"
    );
    for path in [vob, idx, sub] {
        let _ = std::fs::remove_file(path);
    }
}

// ---- robustness ----

/// Every count and length in a program stream is attacker-controlled, so the
/// parser has to survive whatever a file claims: a truncated pack, a packet
/// length past the end of the data, a subpicture unit that declares a size it
/// never delivers, and plain noise.
#[test]
fn malformed_input_never_panics() {
    let cases: Vec<(&str, Vec<u8>)> = Vec::from([
        ("empty", Vec::new()),
        ("bare prefix", Vec::from([0x00u8, 0x00, 0x01])),
        (
            "pack header cut short",
            Vec::from([0x00u8, 0x00, 0x01, 0xBA]),
        ),
        (
            "pack with an impossible marker",
            Vec::from([0x00u8, 0x00, 0x01, 0xBA, 0xFF, 0xFF, 0xFF, 0xFF]),
        ),
        (
            "zero packet length",
            Vec::from([0x00u8, 0x00, 0x01, 0xE0, 0x00, 0x00, 0x00, 0x00, 0x01, 0xE0]),
        ),
        (
            "packet length past the end",
            Vec::from([0x00u8, 0x00, 0x01, 0xE0, 0xFF, 0xFF, 0x80, 0x00, 0x00]),
        ),
        (
            "PES header data length past the end",
            Vec::from([
                0x00u8, 0x00, 0x01, 0xE0, 0x00, 0x08, 0x80, 0x80, 0xFF, 0x11, 0x22,
            ]),
        ),
        (
            "private stream with no substream byte",
            Vec::from([0x00u8, 0x00, 0x01, 0xBD, 0x00, 0x03, 0x80, 0x00, 0x00]),
        ),
        ("MPEG-1 header of nothing but stuffing", {
            let mut v = Vec::from([0x00u8, 0x00, 0x01, 0xE0, 0x00, 0x20]);
            v.resize(38, 0xFF);
            v
        }),
        ("all zeroes", vec![0u8; 4096]),
        ("start code storm", start_code_storm(4096)),
        ("counting noise", (0..=255u8).cycle().take(4096).collect()),
    ]);
    for (name, data) in cases {
        let mut demux = PsDemuxer::new();
        demux.push_data(&data);
        demux.flush();
        let units = demux.take_units();
        // Whatever survives must at least be self-consistent.
        for u in &units {
            assert!(!u.data.is_empty(), "{name}: an empty unit was emitted");
        }
        // Byte-at-a-time feeding exercises every partial-header path.
        let mut demux = PsDemuxer::new();
        for b in &data {
            demux.push_data(&[*b]);
        }
        demux.flush();
        let _ = demux.take_units();
    }
}

/// A subpicture unit whose declared size never arrives must not accumulate
/// without bound, and must emit nothing.
#[test]
fn a_subpicture_claiming_a_size_it_never_delivers_emits_nothing() {
    let mut demux = PsDemuxer::new();
    // A `private_stream_1` PES: MPEG-2 header with a PTS, substream 0x20, then a
    // unit declaring 0xFFFF bytes of which only two ever arrive.
    let body = Vec::from([
        0x80u8, 0x80, 0x05, 0x21, 0x00, 0x01, 0x00, 0x01, 0x20, 0xFF, 0xFF,
    ]);
    let mut packet = Vec::from([0x00u8, 0x00, 0x01, 0xBD]);
    packet.extend_from_slice(&(body.len() as u16).to_be_bytes());
    packet.extend_from_slice(&body);
    // Repeat it: each repetition opens a new unit (it carries a PTS), so nothing
    // accumulates across them.
    for _ in 0..1000 {
        demux.push_data(&packet);
    }
    assert!(
        demux.take_units().is_empty(),
        "an undelivered unit emits nothing"
    );
    assert_eq!(
        demux.streams().len(),
        1,
        "the substream is still discovered"
    );
}

#[test]
fn a_reserved_sequence_header_is_rejected() {
    // 00 00 01 B3, then width 352 / height 288 and frame_rate_code 0 (forbidden).
    let au = Vec::from([0x00u8, 0x00, 0x01, 0xB3, 0x16, 0x01, 0x20, 0x10]);
    assert!(parse_sequence_header(&au).is_none(), "code 0 is forbidden");
    let au = Vec::from([0x00u8, 0x00, 0x01, 0xB3, 0x16, 0x01, 0x20, 0x13]);
    let seq = parse_sequence_header(&au).expect("code 3 is 25 fps");
    assert_eq!(
        (seq.width, seq.height, seq.framerate_q16),
        (352, 288, 25 << 16)
    );
    // A zero dimension cannot fixate, so it is not accepted either.
    let au = Vec::from([0x00u8, 0x00, 0x01, 0xB3, 0x00, 0x00, 0x00, 0x13]);
    assert!(parse_sequence_header(&au).is_none());
    // Truncated after the start code.
    assert!(parse_sequence_header(&[0x00, 0x00, 0x01, 0xB3, 0x16]).is_none());
}

fn start_code_storm(n: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut id = 0u8;
    while out.len() < n {
        out.extend_from_slice(&[0x00, 0x00, 0x01, id]);
        id = id.wrapping_add(1);
    }
    out
}

/// The source pad advertises every selection, so an auto-plug search can reach
/// each of them, and the video one never at an unfixatable `Any`.
#[test]
fn the_source_pad_advertises_every_selection() {
    let templates = PsDemux::pad_templates();
    let source = templates
        .iter()
        .find(|t| t.direction == PadDirection::Source)
        .expect("a source pad");
    let wants = [
        Caps::CompressedVideo {
            codec: VideoCodec::Mpeg2,
            width: Dim::Range {
                min: 16,
                max: 65_535,
            },
            height: Dim::Range {
                min: 16,
                max: 65_535,
            },
            framerate: Rate::Range {
                min_q16: 1 << 16,
                max_q16: 240 << 16,
            },
        },
        Caps::Audio {
            format: AudioFormat::Mp2,
            channels: 0,
            sample_rate: 0,
        },
        Caps::Audio {
            format: AudioFormat::Ac3,
            channels: 0,
            sample_rate: 0,
        },
        Caps::SubPicture {
            format: SubPictureFormat::VobSub,
        },
    ];
    for want in wants {
        let g2g_core::PadCaps::Fixed(set) = &source.caps else {
            panic!("a source pad names concrete caps");
        };
        assert!(set.accepts(&want), "the pad offers {want:?}");
    }
    let mut probe = PsDemuxer::new();
    probe.push_data(&[]);
    assert!(forwardable_streams(&probe).is_empty(), "nothing seen yet");
}

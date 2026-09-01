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

use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::frame::{Frame, PipelinePacket};
use g2g_core::{frame::FrameTiming, memory::SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, Dim, G2gError, MemoryDomain,
    MultiInputElement, MultiOutputElement, OutputSink, PadDirection, PadTemplates, PropValue,
    PushOutcome, Rate, SubPictureFormat, VideoCodec,
};
/// A multi-output sink recording each port's packets in order.
struct PortTap {
    ports: Vec<Vec<PipelinePacket>>,
}

impl PortTap {
    fn new(n: usize) -> Self {
        Self {
            ports: (0..n).map(|_| Vec::new()).collect(),
        }
    }
}

impl g2g_core::MultiOutputSink for PortTap {
    fn port_count(&self) -> usize {
        self.ports.len()
    }

    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");

        self.ports[port].push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

use g2g_plugins::psdemux::{
    forwardable_streams, parse_sequence_header, subpicture_streams, PsDemux, PsDemuxN, PsDemuxer,
    PsStream,
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN
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
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
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
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
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

    // `playbin uri=file://x.vob` fans the disc out. This fixture carries a
    // subpicture track, so since M931 the graph is the compositing overlay:
    // video, audio and subpicture ports.
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
        [g2g_core::graph::NodeKind::Tee(3)],
        "video + audio + subpicture branches"
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
        // An MPEG-2 pack whose stuffing count names bytes that never arrive:
        // the header is complete but the pack it declares is not.
        ("MPEG-2 pack truncated inside its stuffing", {
            let mut v = Vec::from([0x00u8, 0x00, 0x01, 0xBA]);
            v.extend_from_slice(&[0x44, 0x00, 0x04, 0x00, 0x04, 0x01, 0x00, 0x03, 0xF8]);
            v.push(0x07); // pack_stuffing_length = 7, so the pack is 21 bytes
            v
        }),
        (
            "MPEG-1 pack truncated after its marker",
            Vec::from([0x00u8, 0x00, 0x01, 0xBA, 0x21]),
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

/// A truncated pack must leave the parser waiting, not reading past its buffer.
/// Both pack layouts declare a length from bytes inside the header itself, so a
/// file cut short mid-pack once made the parser drain past the end and panic.
#[test]
fn a_truncated_pack_waits_for_the_rest_instead_of_overrunning() {
    // MPEG-2: 14 header bytes present, stuffing_length 7 promising a 21-byte pack.
    let mut head = Vec::from([0x00u8, 0x00, 0x01, 0xBA]);
    head.extend_from_slice(&[0x44, 0x00, 0x04, 0x00, 0x04, 0x01, 0x00, 0x03, 0xF8, 0x07]);
    assert_eq!(head.len(), 14);
    let mut demux = PsDemuxer::new();
    demux.push_data(&head);
    assert!(demux.streams().is_empty(), "nothing decodable yet");

    // The pack completes and a video packet follows: the parser picks up where
    // it left off rather than having discarded the stream.
    let mut rest = vec![0xFFu8; 7]; // the promised stuffing
    rest.extend_from_slice(&pes(0xE0, Some(9_000), &video_unit(352, 288, 1)));
    demux.push_data(&rest);
    demux.flush();
    assert_eq!(
        demux.streams().len(),
        1,
        "the completed pack's packet is parsed"
    );
    assert_eq!(
        demux.sequence().map(|s| (s.width, s.height)),
        Some((352, 288))
    );

    // MPEG-1: the '0010' marker promises 12 bytes, only 5 are present.
    let mut demux = PsDemuxer::new();
    demux.push_data(&[0x00, 0x00, 0x01, 0xBA, 0x21]);
    assert!(demux.take_units().is_empty());
}

/// A spliced program stream (concatenated titles) changes geometry mid-file, and
/// the video caps have to follow it: the refinement is not a one-shot.
#[tokio::test]
async fn a_mid_stream_geometry_change_is_announced() {
    let mut file = Vec::new();
    for (i, (w, h)) in [(352u32, 288u32), (352, 288), (704, 576), (704, 576)]
        .into_iter()
        .enumerate()
    {
        file.extend_from_slice(&PACK);
        let pts = 9_000 * (i as u64 + 1);
        file.extend_from_slice(&pes(0xE0, Some(pts), &video_unit(w, h, 1)));
    }
    let caps = demux(&file, PsStream::Mpeg2).await.caps_changes();
    let geometry: Vec<(Dim, Dim)> = caps
        .iter()
        .filter_map(|c| match c {
            Caps::CompressedVideo { width, height, .. } => Some((width.clone(), height.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        geometry,
        [
            (Dim::Fixed(352), Dim::Fixed(288)),
            (Dim::Fixed(704), Dim::Fixed(576)),
        ],
        "the new geometry is announced, and neither is repeated"
    );
}

/// A long run with no start code at all must not make every fragment re-scan the
/// whole buffer: a crafted file would otherwise cost quadratic work. Feeding a
/// megabyte in small fragments finishes promptly if the scan is linear.
#[test]
fn a_stream_with_no_start_codes_does_not_rescan_quadratically() {
    let mut file = Vec::new();
    file.extend_from_slice(&PACK);
    // One PES whose payload is 60 KiB of a byte that can never open a start code.
    file.extend_from_slice(&pes(0xE0, Some(9_000), &vec![0x42u8; 60_000]));
    let mut demux = PsDemuxer::new();
    let start = std::time::Instant::now();
    for _ in 0..64 {
        demux.push_data(&file);
    }
    demux.flush();
    assert!(
        start.elapsed() < std::time::Duration::from_secs(5),
        "the start-code scan stayed linear: {:?}",
        start.elapsed()
    );
}

/// A program stream cut mid-subpicture must yield only whole cues: the
/// half-arrived unit can never reach its declared size, so it is dropped rather
/// than decoded onto a short canvas. Every canvas downstream is a full frame.
#[tokio::test]
async fn a_stream_cut_mid_subpicture_yields_only_whole_frames() {
    // One complete cue, then the opening half of a second one.
    let cue = spu_unit(40, 60, 120, 40);
    let mut file = Vec::new();
    // Two pictures, so a video access unit completes (and its sequence header
    // parses) before the first cue, the way a real title plays.
    for i in 0..2 {
        file.extend_from_slice(&PACK);
        file.extend_from_slice(&pes(0xE0, Some(9_000 * (i + 1)), &video_unit(720, 480, 1)));
    }
    file.extend_from_slice(&PACK);
    file.extend_from_slice(&pes_private(0x20, Some(27_000), &cue));
    file.extend_from_slice(&PACK);
    // Truncated: the unit declares its full size but only a third arrives.
    file.extend_from_slice(&pes_private(0x20, Some(180_000), &cue[..cue.len() / 3]));

    let demuxed = demux(&file, PsStream::SubPicture).await;
    let frames = demuxed.frames();
    assert_eq!(
        frames.len(),
        2,
        "the synthesized .idx and the one complete cue, not the truncated one"
    );
    let config = parse_idx(&bytes(frames[0])).expect("the pad opens on .idx text");
    assert_eq!(config.size, Some((720, 480)));
    let spu = bytes(frames[1]);
    assert_eq!(
        u16::from_be_bytes([spu[0], spu[1]]) as usize,
        spu.len(),
        "the emitted cue is exactly its declared size"
    );

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
    let canvases = sink.frames();
    assert!(!canvases.is_empty(), "the complete cue rendered");
    let full = (720 * 480 * 4) as usize;
    for (i, c) in canvases.iter().enumerate() {
        assert_eq!(
            bytes(c).len(),
            full,
            "canvas {i} is a whole 720x480 frame, never a partial one"
        );
    }
}

/// AC-3 frames straddle the sector-cut PES packets of a program stream, so the
/// demuxer has to realign them: a decoder is handed whole syncframes, not the
/// fragments the container happens to carry. Authored so no frame starts where a
/// packet does after the first.
#[tokio::test]
async fn ac3_frames_split_across_packets_are_realigned() {
    // Six 384-byte syncframes (frmsizecod 8, fscod 0 -> 128 words) back to back.
    const FRAME: usize = 256;
    let stream: Vec<u8> = (0..8).flat_map(|i| ac3_syncframe(FRAME, i as u8)).collect();
    assert_eq!(stream.len(), FRAME * 8);

    // Cut it into packets that deliberately do not fall on frame boundaries.
    let cuts = [100usize, 300, 700, 1100, 1500, stream.len()];
    let mut file = Vec::from(PACK);
    let mut at = 0usize;
    for cut in cuts {
        let chunk = &stream[at..cut];
        // The DVD pointer is 1-based from the byte after it, and 0 when no frame
        // starts in this packet.
        let first_in_chunk = (at..cut).find(|o| o % FRAME == 0).map(|o| o - at);
        let ptr = first_in_chunk.map_or(0, |o| o + 1) as u16;
        file.extend_from_slice(&PACK);
        file.extend_from_slice(&pes_ac3(ptr, Some(9_000 + at as u64), chunk));
        at = cut;
    }

    let frames = demux(&file, PsStream::Ac3).await;
    let emitted: Vec<Vec<u8>> = frames.frames().iter().map(|f| bytes(f)).collect();
    let total: usize = emitted.iter().map(|u| u.len()).sum();
    assert_eq!(
        total % FRAME,
        0,
        "every emitted byte belongs to a whole syncframe"
    );
    assert_eq!(total / FRAME, 8, "all eight frames survive realignment");
    // Each emitted unit opens on a syncword and is a whole number of frames.
    for (i, u) in emitted.iter().enumerate() {
        assert_eq!(&u[..2], &[0x0B, 0x77], "unit {i} opens on a syncframe");
        assert_eq!(u.len() % FRAME, 0, "unit {i} is whole frames");
    }
    // The payload survives byte for byte, in order.
    let joined: Vec<u8> = emitted.concat();
    assert_eq!(joined, stream, "the frames are the authored ones, in order");
}

/// A byte range cut from the middle of a VOB joins mid-GOP: the first pictures
/// arrive with no sequence header ahead of them, so nothing downstream knows the
/// geometry and libavcodec fails the whole stream on "invalid frame dimensions".
/// The demuxer drops to the first sync point instead, the tune-in convention the
/// rest of the tree follows.
#[tokio::test]
async fn a_mid_gop_tune_in_drops_to_the_first_sequence_header() {
    let mut file = Vec::new();
    // Two pictures that are not sync points, then the sequence, then one more.
    for (i, unit) in [
        picture_unit(2),
        picture_unit(3),
        video_unit(720, 480, 1),
        picture_unit(2),
    ]
    .into_iter()
    .enumerate()
    {
        file.extend_from_slice(&PACK);
        file.extend_from_slice(&pes(0xE0, Some(9_000 * (i as u64 + 1)), &unit));
    }

    let out = demux(&file, PsStream::Mpeg2).await;
    let frames = out.frames();
    assert_eq!(
        frames.len(),
        2,
        "only the sequence-header unit and what follows it"
    );

    let first = bytes(frames[0]);
    assert!(
        first.starts_with(&[0x00, 0x00, 0x01, 0xB3]),
        "the stream opens on the sequence header"
    );
    assert!(
        parse_sequence_header(&first).is_some(),
        "so a decoder can size its frames"
    );
    assert!(
        frames[0].timing.keyframe,
        "the first unit is flagged an independently-decodable point"
    );
    assert!(!frames[1].timing.keyframe, "the P-picture after it is not");

    // And the geometry is announced before any frame goes out.
    assert_eq!(
        out.caps_changes().first(),
        Some(&Caps::CompressedVideo {
            codec: VideoCodec::Mpeg2,
            width: Dim::Fixed(720),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(25 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN
        })
    );
    let first_caps = out
        .packets
        .iter()
        .position(|p| matches!(p, PipelinePacket::CapsChanged(_)));
    let first_frame = out
        .packets
        .iter()
        .position(|p| matches!(p, PipelinePacket::DataFrame(_)));
    assert!(first_caps < first_frame, "caps lead the first frame");
}

/// The fan-out subtitle route must open on the same synthesized `.idx` the
/// single-output demuxer sends. Without it a `playbin` / named-pad subtitle
/// branch renders on `vobsubdec`'s 720x576 default, so an NTSC disc's cues land
/// on the wrong canvas.
#[tokio::test]
async fn the_fanout_subpicture_port_opens_on_the_idx_config() {
    let cue = spu_unit(40, 60, 120, 40);
    let mut file = Vec::new();
    for i in 0..2 {
        file.extend_from_slice(&PACK);
        file.extend_from_slice(&pes(0xE0, Some(9_000 * (i + 1)), &video_unit(720, 480, 1)));
    }
    file.extend_from_slice(&PACK);
    file.extend_from_slice(&pes_private(0x20, Some(27_000), &cue));

    let mut el = PsDemuxN::new(Vec::from([PsStream::Mpeg2, PsStream::SubPicture]));
    el.configure_pipeline(&ps_caps()).expect("configure");
    let mut tap = PortTap::new(2);
    for chunk in file.chunks(4096) {
        el.process(data_frame(chunk.to_vec()), &mut tap)
            .await
            .expect("demux a chunk");
    }
    el.process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("flush");

    let frames: Vec<Vec<u8>> = tap.ports[1]
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(bytes(f)),
            _ => None,
        })
        .collect();
    assert_eq!(frames.len(), 2, "the .idx config, then the cue");
    let config = parse_idx(&frames[0]).expect("the port opens on .idx text");
    assert_eq!(
        config.size,
        Some((720, 480)),
        "carrying the video's own geometry, not the decoder's default"
    );
    assert_eq!(
        u16::from_be_bytes([frames[1][0], frames[1][1]]) as usize,
        frames[1].len(),
        "the cue follows it whole"
    );
}

/// M931: `playbin uri=file.vob` on a disc with a subpicture track builds the
/// compositing overlay graph, and a disc without one builds the plain A/V
/// fan-out. Needs a decoder for the video, so it runs under the `ffmpeg` feature.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
#[test]
fn playbin_builds_a_subpicture_overlay_only_when_the_disc_has_one() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let reg = default_registry();

    // With subpictures: a 2-input compositor blends the cues and the demux fans
    // video, audio and subpicture.
    let (vob, idx, sub) = (
        temp_path("pb.vob"),
        temp_path("pb.idx"),
        temp_path("pb.sub"),
    );
    author_vob(&vob, &idx, &sub);
    let graph =
        g2g_core::runtime::parse_launch(&reg, &format!("playbin uri=file://{}", vob.display()))
            .unwrap_or_else(|e| panic!("playbin builds the subtitled disc: {e}"));
    let vg = graph.finish().expect("valid graph");
    let kinds: Vec<g2g_core::graph::NodeKind> = vg.topo().iter().map(|&n| vg.kind(n)).collect();
    assert!(
        kinds
            .iter()
            .any(|k| matches!(k, g2g_core::graph::NodeKind::Muxer(2))),
        "a 2-input compositor blends the cues: {kinds:?}"
    );
    assert!(
        kinds
            .iter()
            .any(|k| matches!(k, g2g_core::graph::NodeKind::Tee(3))),
        "the demux fans video, audio and subpicture: {kinds:?}"
    );

    // Without a subpicture track: the plain A/V fan-out, no compositor.
    let plain = temp_path("plain.mpg");
    author_mpg(&plain, "3");
    let graph =
        g2g_core::runtime::parse_launch(&reg, &format!("playbin uri=file://{}", plain.display()))
            .unwrap_or_else(|e| panic!("playbin builds the plain disc: {e}"));
    let vg = graph.finish().expect("valid graph");
    let kinds: Vec<g2g_core::graph::NodeKind> = vg.topo().iter().map(|&n| vg.kind(n)).collect();
    assert!(
        !kinds
            .iter()
            .any(|k| matches!(k, g2g_core::graph::NodeKind::Muxer(_))),
        "no subpicture track, no overlay: {kinds:?}"
    );
    for p in [vob, idx, sub, plain] {
        let _ = std::fs::remove_file(p);
    }
}

/// The pixels the overlay exists for. A cue demuxed from the authored VOB and
/// decoded is composited over a flat video frame: inside the cue's window the
/// composite differs from the video, and only within the authored rectangle;
/// a frame outside the window is the video untouched.
#[tokio::test]
async fn composited_frames_carry_the_cue_only_inside_its_window_and_rectangle() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let (vob, idx, sub) = (
        temp_path("px.vob"),
        temp_path("px.idx"),
        temp_path("px.sub"),
    );
    author_vob(&vob, &idx, &sub);
    let file = std::fs::read(&vob).expect("read the VOB");
    let demuxed = demux(&file, PsStream::SubPicture).await;

    // Decode the demuxed cues the way the overlay branch does.
    let mut dec = VobSubDec::new();
    dec.configure_pipeline(&Caps::SubPicture {
        format: SubPictureFormat::VobSub,
    })
    .expect("vobsubdec configures");
    let mut cues = Collect::default();
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
                &mut cues,
            )
            .await
            .expect("decode a cue");
        }
    }
    let canvases = cues.frames();
    assert!(canvases.len() >= 2, "painted canvases and clearing ones");
    // The last painted canvas is the frame before the final clear.
    let painted = canvases[canvases.len() - 2];
    let cue = cues_last();

    // Composite it over a flat video frame of a known colour.
    const FLAT: [u8; 4] = [17, 34, 51, 255];
    let mut comp = g2g_plugins::compositor::Compositor::new(
        VOB_W,
        VOB_H,
        Vec::from([
            g2g_plugins::compositor::CompositorPad::at(0, 0),
            g2g_plugins::compositor::CompositorPad::at(0, 0),
        ]),
    );
    let rgba = Caps::RawVideo {
        format: g2g_core::RawVideoFormat::Rgba8,
        width: Dim::Fixed(VOB_W),
        height: Dim::Fixed(VOB_H),
        framerate: Rate::Fixed(25 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    };
    for pad in 0..2 {
        comp.configure_pipeline(pad, &rgba).expect("configure");
    }
    let flat = |pts: u64| {
        let mut buf = Vec::with_capacity((VOB_W * VOB_H * 4) as usize);
        for _ in 0..VOB_W * VOB_H {
            buf.extend_from_slice(&FLAT);
        }
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
            FrameTiming {
                pts_ns: pts,
                dts_ns: pts,
                ..FrameTiming::default()
            },
            0,
        ))
    };
    let mut out = Collect::default();
    comp.process(
        1,
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes(painted).into_boxed_slice())),
            painted.timing,
            0,
        )),
        &mut out,
    )
    .await
    .expect("the cue canvas in");
    comp.process(0, flat(painted.timing.pts_ns), &mut out)
        .await
        .expect("video in");

    let composited = out.frames();
    assert!(!composited.is_empty(), "the compositor emitted a frame");
    let inside = bytes(composited[composited.len() - 1]);
    assert_eq!(inside.len(), (VOB_W * VOB_H * 4) as usize);

    let mut changed = 0usize;
    for y in 0..VOB_H {
        for x in 0..VOB_W {
            let at = ((y * VOB_W + x) * 4) as usize;
            if inside[at..at + 4] == FLAT {
                continue;
            }
            changed += 1;
            assert!(
                (cue.x..cue.x + cue.w).contains(&x) && (cue.y..cue.y + cue.h).contains(&y),
                "a changed pixel at ({x},{y}) is outside the authored cue rectangle"
            );
        }
    }
    assert!(changed > 100, "the cue painted its rectangle: {changed} px");

    // Outside the window: the clearing canvas composites to the flat video.
    let clear = canvases[canvases.len() - 1];
    let mut out2 = Collect::default();
    comp.process(
        1,
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes(clear).into_boxed_slice())),
            clear.timing,
            0,
        )),
        &mut out2,
    )
    .await
    .expect("the clear canvas in");
    comp.process(0, flat(clear.timing.pts_ns), &mut out2)
        .await
        .expect("video in");
    let after = out2.frames();
    if let Some(f) = after.last() {
        let px = bytes(f);
        assert!(
            px.chunks(4).all(|p| p == FLAT),
            "after the cue's hide time the video is untouched"
        );
    }
    for p in [vob, idx, sub] {
        let _ = std::fs::remove_file(p);
    }
}

/// The last cue the fixture authors, whose rectangle the composite must respect.
fn cues_last() -> vobsub_fixture::Cue {
    let mut all = cues();
    all.pop().expect("the fixture authors cues")
}

// --- synthetic program stream builders ---

/// An MPEG-2 pack header with no stuffing.
const PACK: [u8; 14] = [
    0x00, 0x00, 0x01, 0xBA, 0x44, 0x00, 0x04, 0x00, 0x04, 0x01, 0x00, 0x03, 0xF8, 0x00,
];

/// A 33-bit timestamp in the five-byte PES field layout.
fn pts_field(pts90: u64) -> [u8; 5] {
    let p = pts90 & 0x1_ffff_ffff;
    [
        0x20 | ((p >> 29) as u8 & 0x0e) | 1,
        (p >> 22) as u8,
        (p >> 14) as u8 | 1,
        (p >> 7) as u8,
        ((p << 1) as u8) | 1,
    ]
}

/// One MPEG-2 PES packet.
fn pes(stream_id: u8, pts90: Option<u64>, payload: &[u8]) -> Vec<u8> {
    let mut body = Vec::from([0x80u8, 0x00, 0x00]);
    if let Some(p) = pts90 {
        body[1] = 0x80;
        body[2] = 0x05;
        body.extend_from_slice(&pts_field(p));
    }
    body.extend_from_slice(payload);
    let mut out = Vec::from([0x00u8, 0x00, 0x01, stream_id]);
    out.extend_from_slice(&(body.len() as u16).to_be_bytes());
    out.extend_from_slice(&body);
    out
}

/// A video access unit with no sequence header: just a picture of
/// `coding_type` (2 = P, 3 = B), the shape a mid-GOP tune-in starts on.
fn picture_unit(coding_type: u8) -> Vec<u8> {
    let mut out = Vec::from([
        0x00,
        0x00,
        0x01,
        0x00,
        0x00,
        (coding_type << 3) | 0x07,
        0xFF,
    ]);
    out.extend_from_slice(&[0x42u8; 64]);
    out
}

/// A video access unit: a sequence header at `w` x `h` and 25 fps, then one
/// picture of `coding_type`, then filler standing in for coded data.
fn video_unit(w: u32, h: u32, coding_type: u8) -> Vec<u8> {
    let mut out = Vec::from([
        0x00,
        0x00,
        0x01,
        0xB3,
        (w >> 4) as u8,
        (((w & 0xF) << 4) | (h >> 8)) as u8,
        h as u8,
        0x13, // aspect_ratio 1, frame_rate_code 3 (25 fps)
    ]);
    out.extend_from_slice(&[
        0x00,
        0x00,
        0x01,
        0x00,
        0x00,
        (coding_type << 3) | 0x07,
        0xFF,
    ]);
    out.extend_from_slice(&[0x42u8; 64]);
    out
}

/// One `private_stream_1` PES packet on a subpicture substream.
fn pes_private(substream: u8, pts90: Option<u64>, payload: &[u8]) -> Vec<u8> {
    let mut body = Vec::from([substream]);
    body.extend_from_slice(payload);
    pes(0xBD, pts90, &body)
}

/// One `private_stream_1` PES packet on AC-3 substream 0x80, carrying the DVD
/// substream header: a frame-header count and the 1-based pointer to the first
/// access unit starting in this packet.
fn pes_ac3(first_au_ptr: u16, pts90: Option<u64>, payload: &[u8]) -> Vec<u8> {
    let mut body = Vec::from([0x80u8, 1]);
    body.extend_from_slice(&first_au_ptr.to_be_bytes());
    body.extend_from_slice(payload);
    pes(0xBD, pts90, &body)
}

/// An AC-3 syncframe of exactly `len` bytes. `frmsizecod` 8 at `fscod` 0 is 128
/// 16-bit words (256 bytes) at 48 kHz; `tag` marks the frame so a reordering or
/// a dropped fragment is visible in the reassembled bytes.
fn ac3_syncframe(len: usize, tag: u8) -> Vec<u8> {
    assert_eq!(len, 256, "the authored frmsizecod is 256 bytes");
    let mut f = Vec::from([0x0Bu8, 0x77, 0x00, 0x00, 0x08]);
    f.resize(len, tag);
    f
}

/// A minimal subpicture unit: a size field, a control-sequence offset, a scrap
/// of pixel data, and one control sequence that shows then hides the cue.
fn spu_unit(x: u32, y: u32, w: u32, h: u32) -> Vec<u8> {
    let pixel_data = [0x00u8; 16];
    let data_end = 4 + pixel_data.len();
    let (x2, y2) = (x + w - 1, y + h - 1);
    let show = Vec::from([
        0x03,
        0x10,
        0x32, // colormap
        0x04,
        0xFF,
        0xF0, // alpha
        0x05,
        (x >> 4) as u8,
        (((x & 0xf) << 4) | (x2 >> 8)) as u8,
        x2 as u8,
        (y >> 4) as u8,
        (((y & 0xf) << 4) | (y2 >> 8)) as u8,
        y2 as u8,
        0x06,
        0x00,
        0x04,
        0x00,
        0x04, // field offsets
        0x01,
        0xFF,
    ]);
    let hide = [0x02u8, 0xFF];
    let seq1 = data_end;
    let seq2 = seq1 + 4 + show.len();
    let total = seq2 + 4 + hide.len();
    let mut out = Vec::new();
    out.extend_from_slice(&(total as u16).to_be_bytes());
    out.extend_from_slice(&(seq1 as u16).to_be_bytes());
    out.extend_from_slice(&pixel_data);
    out.extend_from_slice(&0u16.to_be_bytes());
    out.extend_from_slice(&(seq2 as u16).to_be_bytes());
    out.extend_from_slice(&show);
    out.extend_from_slice(&180u16.to_be_bytes());
    out.extend_from_slice(&(seq2 as u16).to_be_bytes());
    out.extend_from_slice(&hide);
    assert_eq!(out.len(), total);
    out
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
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

/// A DVD title's PTS starts wherever the disc puts it (a mid-title slice can
/// open at hundreds of seconds), so each stream must open with a `Segment`
/// mapping its first PTS to running time 0: without one a paced sink holds
/// every frame until that offset passes on the wall clock.
#[tokio::test]
async fn a_stream_start_segment_precedes_the_first_frame() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let path = temp_path("segment.mpg");
    author_mpg(&path, "1");
    let file = std::fs::read(&path).expect("read the fixture");

    let video = demux(&file, PsStream::Mpeg2).await;
    let seg_at = video
        .packets
        .iter()
        .position(|p| matches!(p, PipelinePacket::Segment(_)))
        .expect("a stream-start segment is emitted");
    let frame_at = video
        .packets
        .iter()
        .position(|p| matches!(p, PipelinePacket::DataFrame(_)))
        .expect("frames follow");
    assert!(seg_at < frame_at, "segment before the first frame");
    let PipelinePacket::Segment(seg) = &video.packets[seg_at] else {
        unreachable!();
    };
    let first_pts = video.frames()[0].timing.pts_ns;
    assert_eq!(seg.start, first_pts, "segment start is the first PTS");
    assert_eq!(
        seg.to_running_time(first_pts),
        Some(0),
        "the first frame presents immediately"
    );
}

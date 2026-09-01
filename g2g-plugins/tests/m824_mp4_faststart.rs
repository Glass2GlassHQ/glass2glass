//! M824: `mp4mux faststart=true`, the progressive layout with its `moov` ahead
//! of the `mdat`. A reader then has the whole sample index before it has read
//! any media, so playback over a network needs no seek to the end of the file
//! (the `qtmux faststart` / `ffmpeg -movflags +faststart` layout).
//!
//! The chunk offsets shift by the `moov`'s own size, which is the part that can
//! go wrong: every check here demuxes the faststart file and compares it against
//! the moov-at-end progressive file of the same frames, and hands both to ffmpeg
//! to decode.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{
    parse_launch, run_graph, DynSourceLoop, Registry, SourceFactory, SourceLoop,
};
use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, ConfigureOutcome, Dim, G2gError, MultiInputElement,
    MultiOutputElement, MultiOutputSink, OutputSink, PipelineClock, PropValue, PushOutcome, Rate,
    VideoCodec,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};
use g2g_plugins::mp4muxn::Mp4MuxN;
use g2g_plugins::registry::default_registry;

const WIDTH: usize = 320;
const HEIGHT: usize = 240;

#[derive(Default)]
struct CaptureSink {
    bytes: Vec<u8>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.bytes.extend_from_slice(s);
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Per-port capture of a demuxer's output: the packets and the refined caps.
#[derive(Default)]
struct PortCapture {
    ports: Vec<Vec<(Vec<u8>, FrameTiming)>>,
    caps: Vec<Option<Caps>>,
}

impl MultiOutputSink for PortCapture {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if self.ports.len() <= port {
                self.ports.resize(port + 1, Vec::new());
                self.caps.resize(port + 1, None);
            }
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.ports[port].push((s.to_vec(), f.timing));
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps[port] = Some(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }

    fn port_count(&self) -> usize {
        self.ports.len().max(1)
    }
}

/// One input pad's worth of muxer input: its negotiated caps, the concrete caps
/// a demuxer refined to at runtime, and its frames.
struct MuxTrack {
    nego: Caps,
    refined: Option<Caps>,
    frames: Vec<(Vec<u8>, FrameTiming)>,
}

fn frame(data: Vec<u8>, timing: FrameTiming) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        timing,
        0,
    ))
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m824-{}-{name}", std::process::id()))
}

/// The top-level 4ccs of a file, in order.
fn top_level_boxes(file: &[u8]) -> Vec<[u8; 4]> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + 8 <= file.len() {
        let size = u32::from_be_bytes(file[at..at + 4].try_into().unwrap()) as usize;
        // `size 1` means the real size is the 64-bit largesize that follows.
        let size = if size == 1 {
            u64::from_be_bytes(file[at + 8..at + 16].try_into().unwrap()) as usize
        } else {
            size
        };
        if size < 8 || at + size > file.len() {
            break;
        }
        out.push(file[at + 4..at + 8].try_into().unwrap());
        at += size;
    }
    out
}

/// ffprobe's `key=value` lines for one stream selector, plus the container
/// duration under `format_duration`.
fn probe(path: &PathBuf, select: &str) -> Vec<(String, String)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", select])
        .args([
            "-show_entries",
            "stream=codec_name,width,height,channels,sample_rate,duration,start_time,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1",
        ])
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(
        out.status.success() && out.stderr.is_empty(),
        "ffprobe read {} without complaint: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    let mut seen_stream_duration = false;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().split_once('='))
        .map(|(k, v)| {
            let key = if k == "duration" && seen_stream_duration {
                "format_duration"
            } else {
                if k == "duration" {
                    seen_stream_duration = true;
                }
                k
            };
            (key.to_string(), v.to_string())
        })
        .collect()
}

fn field<'a>(probed: &'a [(String, String)], key: &str) -> &'a str {
    probed
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("ffprobe reported {key}, got {probed:?}"))
}

fn decode(path: &PathBuf, args: &[&str]) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(args)
        .arg("-")
        .output()
        .expect("run ffmpeg");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && err.is_empty(),
        "ffmpeg decoded {} cleanly: {err}",
        path.display()
    );
    assert!(!out.stdout.is_empty(), "ffmpeg decoded something");
    out.stdout
}

fn decode_video(path: &PathBuf) -> Vec<u8> {
    decode(path, &["-f", "rawvideo", "-pix_fmt", "yuv420p"])
}

fn decode_audio(path: &PathBuf) -> Vec<u8> {
    decode(path, &["-f", "s16le", "-c:a", "pcm_s16le"])
}

/// Whether this ffmpeg build has `name` as an encoder.
fn has_encoder(name: &str) -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(name))
        .unwrap_or(false)
}

/// ffmpeg-authored 1.0 s H.264 + AAC MP4 (no B-frames: g2g's MP4 demux carries
/// PTS only, so a reordered stream would not survive the round trip).
fn author_av(path: &PathBuf) -> Option<Vec<u8>> {
    if !has_encoder("libx264") || !has_encoder("aac") {
        eprintln!("skipping: this ffmpeg has no libx264 / aac encoder");
        return None;
    }
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i"])
        .arg(format!("testsrc2=size={WIDTH}x{HEIGHT}:rate=30:duration=1"))
        .args(["-f", "lavfi", "-i"])
        .arg("sine=frequency=440:duration=1:sample_rate=48000")
        .args([
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0", "-g", "15",
        ])
        .args(["-c:a", "aac", "-ac", "2", "-ar", "48000"])
        .arg(path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the A/V fixture");
    Some(std::fs::read(path).expect("read fixture"))
}

/// Mux `tracks` into one progressive MP4, with or without faststart,
/// interleaving the pads by PTS the way a runner does.
async fn mux(tracks: &[MuxTrack], faststart: bool) -> Vec<u8> {
    let mut m = Mp4MuxN::new(tracks.len())
        .with_fragmented(false)
        .with_faststart(faststart);
    let mut sink = CaptureSink::default();
    for (i, t) in tracks.iter().enumerate() {
        m.configure_pipeline(i, &t.nego).expect("configure mp4mux");
    }
    for (i, t) in tracks.iter().enumerate() {
        if let Some(caps) = &t.refined {
            m.process(i, PipelinePacket::CapsChanged(caps.clone()), &mut sink)
                .await
                .expect("caps");
        }
    }
    let mut cursors = vec![0usize; tracks.len()];
    loop {
        let next = (0..tracks.len())
            .filter(|&i| cursors[i] < tracks[i].frames.len())
            .min_by_key(|&i| tracks[i].frames[cursors[i]].1.pts_ns);
        let Some(i) = next else { break };
        let (data, timing) = tracks[i].frames[cursors[i]].clone();
        cursors[i] += 1;
        m.process(i, frame(data, timing), &mut sink)
            .await
            .expect("mux");
    }
    for i in 0..tracks.len() {
        m.process(i, PipelinePacket::Eos, &mut sink)
            .await
            .expect("mux eos");
    }
    sink.bytes
}

/// Demux an MP4 into per-port packets plus each port's refined caps.
async fn demux_mp4(file: &[u8]) -> (Vec<Caps>, PortCapture) {
    let streams = forwardable_streams(file);
    assert!(!streams.is_empty(), "the file has forwardable tracks");
    let ports: Vec<Mp4Port> = streams
        .iter()
        .map(|s| Mp4Port {
            track_id: s.track_id,
            caps: s.caps.clone(),
        })
        .collect();
    let mut d = Mp4DemuxN::new(ports);
    d.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .expect("configure mp4demux");
    let mut tap = PortCapture::default();
    tap.ports.resize(streams.len(), Vec::new());
    tap.caps.resize(streams.len(), None);
    d.process(frame(file.to_vec(), FrameTiming::default()), &mut tap)
        .await
        .expect("demux");
    d.process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("demux eos");
    (streams.into_iter().map(|s| s.caps).collect(), tap)
}

/// The headline: a faststart file leads with its `moov`, and is otherwise the
/// same movie as the moov-at-end progressive one. Checked three ways: g2g's own
/// demuxer reads the same packets out of both (so every shifted `stco` offset
/// lands on its sample), ffprobe reports the same streams, and ffmpeg decodes
/// both to identical pictures and samples.
#[tokio::test]
async fn faststart_writes_the_moov_before_the_mdat() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("src.mp4");
    let Some(file) = author_av(&src) else {
        return;
    };
    let (nego, tap) = demux_mp4(&file).await;
    assert_eq!(nego.len(), 2, "a video and an audio track");
    let tracks: Vec<MuxTrack> = (0..nego.len())
        .map(|i| MuxTrack {
            nego: nego[i].clone(),
            refined: tap.caps[i].clone(),
            frames: tap.ports[i].clone(),
        })
        .collect();

    let fast = mux(&tracks, true).await;
    let plain = mux(&tracks, false).await;

    assert_eq!(
        top_level_boxes(&fast),
        vec![*b"ftyp", *b"moov", *b"mdat"],
        "faststart: the index precedes the media"
    );
    assert_eq!(
        top_level_boxes(&plain),
        vec![*b"ftyp", *b"mdat", *b"moov"],
        "the default progressive layout is unchanged"
    );
    assert_eq!(fast.len(), plain.len(), "the same boxes, reordered");

    // g2g reads its own faststart file: same packets as the moov-at-end file,
    // byte for byte, which is only true if every shifted chunk offset is right.
    let (fast_nego, fast_tap) = demux_mp4(&fast).await;
    let (plain_nego, plain_tap) = demux_mp4(&plain).await;
    assert_eq!(fast_nego.len(), plain_nego.len(), "same track count");
    for port in 0..fast_nego.len() {
        let after: Vec<&Vec<u8>> = fast_tap.ports[port].iter().map(|(b, _)| b).collect();
        let before: Vec<&Vec<u8>> = plain_tap.ports[port].iter().map(|(b, _)| b).collect();
        assert_eq!(
            after, before,
            "port {port}: the two layouts hold the same packets"
        );
        let source: Vec<&Vec<u8>> = tap.ports[port].iter().map(|(b, _)| b).collect();
        assert_eq!(after, source, "port {port}: and the same the source had");
    }

    let fast_path = temp_path("out-faststart.mp4");
    let plain_path = temp_path("out-progressive.mp4");
    std::fs::write(&fast_path, &fast).expect("write faststart");
    std::fs::write(&plain_path, &plain).expect("write progressive");

    // ffprobe reads the same streams out of both.
    for select in ["v:0", "a:0"] {
        let f = probe(&fast_path, select);
        println!("ffprobe faststart {select}: {f:?}");
        assert_eq!(
            f,
            probe(&plain_path, select),
            "{select}: same stream report"
        );
    }
    let video = probe(&fast_path, "v:0");
    assert_eq!(field(&video, "codec_name"), "h264");
    assert_eq!(field(&video, "width"), WIDTH.to_string());
    assert_eq!(field(&video, "height"), HEIGHT.to_string());
    assert_eq!(
        field(&video, "nb_frames"),
        tap.ports[0].len().to_string(),
        "every access unit is in the sample table"
    );

    // And decodes both to the same media.
    let fast_video = decode_video(&fast_path);
    assert_eq!(
        fast_video,
        decode_video(&plain_path),
        "the two layouts decode to the same pictures"
    );
    assert_eq!(
        fast_video.len() / (WIDTH * HEIGHT * 3 / 2),
        tap.ports[0].len(),
        "every muxed picture comes back"
    );
    assert_eq!(
        decode_audio(&fast_path),
        decode_audio(&plain_path),
        "and the same samples"
    );

    // The reference muxer's own faststart remux of our file has the same layout.
    let reference = temp_path("out-ffmpeg-faststart.mp4");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&plain_path)
        .args(["-c", "copy", "-movflags", "+faststart"])
        .arg(&reference)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg remuxed with +faststart");
    let ref_boxes = top_level_boxes(&std::fs::read(&reference).expect("read reference"));
    assert_eq!(
        ref_boxes.iter().position(|b| b == b"moov"),
        Some(1),
        "ffmpeg's faststart puts the moov right after the ftyp too: {ref_boxes:?}"
    );
    assert!(
        ref_boxes.iter().position(|b| b == b"mdat") > ref_boxes.iter().position(|b| b == b"moov"),
        "and the mdat after it: {ref_boxes:?}"
    );

    persist::record_evidence(
        "mp4mux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264+aac")
            .detail("ffprobe reads and ffmpeg decodes a moov-first (faststart) g2g MP4"),
    )
    .expect("record oracle evidence");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&fast_path);
    let _ = std::fs::remove_file(&plain_path);
    let _ = std::fs::remove_file(&reference);
}

#[test]
fn faststart_is_a_declared_property() {
    let mut m = Mp4MuxN::new(2);
    assert!(
        m.properties().iter().any(|s| s.name == "faststart"),
        "declared, so parse_launch can look up its kind"
    );
    assert_eq!(m.get_property("faststart"), Some(PropValue::Bool(false)));
    m.set_property("faststart", PropValue::Bool(true))
        .expect("settable");
    assert_eq!(m.get_property("faststart"), Some(PropValue::Bool(true)));
}

// --- launch line ----------------------------------------------------------

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn aac_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: 48_000,
    }
}

/// Emits a fixed script of (access unit, pts_ns) for one elementary stream.
struct AuSrc {
    caps: Caps,
    aus: Vec<(Vec<u8>, u64)>,
}

impl SourceLoop for AuSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps.clone()))
    }
    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        let aus = self.aus.clone();
        Box::pin(async move {
            for (i, (au, pts)) in aus.iter().enumerate() {
                let f = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
                    FrameTiming {
                        pts_ns: *pts,
                        ..FrameTiming::default()
                    },
                    i as u64,
                );
                out.push(PipelinePacket::DataFrame(f)).await?;
                tokio::task::yield_now().await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(aus.len() as u64)
        })
    }
}

fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}

fn build_h264() -> Box<dyn DynSourceLoop> {
    let key = annexb(&[
        &[0x67, 0x42, 0x00, 0x1E, 0x88],
        &[0x68, 0xCE, 0x3C, 0x80],
        &[0x65, 0x11],
    ]);
    Box::new(AuSrc {
        caps: h264_caps(),
        aus: vec![
            (key, 0),
            (annexb(&[&[0x41, 0x22]]), 40_000_000),
            (annexb(&[&[0x41, 0x33]]), 80_000_000),
        ],
    })
}

fn build_aac() -> Box<dyn DynSourceLoop> {
    let adts = |tail: u8| vec![0xFFu8, 0xF1, 0x4C, 0x80, 0x00, 0x1F, 0xFC, tail];
    Box::new(AuSrc {
        caps: aac_caps(),
        aus: vec![(adts(0xAA), 20_000_000), (adts(0xBB), 60_000_000)],
    })
}

fn registry_with_av_sources() -> Registry {
    let mut reg = default_registry();
    reg.register_source(SourceFactory::new("h264src", h264_caps(), build_h264));
    reg.register_source(SourceFactory::new("aacsrc", aac_caps(), build_aac));
    reg
}

/// The property is reachable from a launch line: the fan-in `mp4mux` takes
/// `faststart=true` and the file it writes leads with its `moov`.
#[tokio::test]
async fn launch_line_sets_faststart_on_the_fan_in_muxer() {
    let out = temp_path("launch.mp4");
    let _ = std::fs::remove_file(&out);
    let reg = registry_with_av_sources();
    let graph = parse_launch(
        &reg,
        &format!(
            "h264src ! m.   aacsrc ! m.   mp4mux name=m fragmented=false faststart=true ! filesink location={}",
            out.display()
        ),
    )
    .expect("fan-in mp4mux accepts faststart");
    run_graph(graph, &ZeroClock, 4).await.expect("runs");

    let file = std::fs::read(&out).expect("the launch line wrote a file");
    assert_eq!(
        top_level_boxes(&file),
        vec![*b"ftyp", *b"moov", *b"mdat"],
        "the launch-built muxer wrote a moov-first file"
    );
    let _ = std::fs::remove_file(&out);
}

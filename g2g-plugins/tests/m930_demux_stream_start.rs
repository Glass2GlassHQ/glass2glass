//! M930: every demuxer opens its stream the same way.
//!
//! A demuxer used to emit a `Segment` only when resuming from a seek, so a
//! stream whose first timestamp is large (a broadcast capture, a mid-title byte
//! slice, any mid-stream join) reached a paced sink with no running-time
//! mapping, and the sink held every frame until that wall-clock offset passed.
//! Each demuxer now emits a stream-start `Segment` mapping its first emitted
//! timestamp to running time 0, and a fan-out shares one origin across its ports
//! so A/V stay aligned.
//!
//! The MPEG-2 tune-in case rides along: those pictures carry no geometry of
//! their own, so a transport stream joined mid-GOP must drop to the first
//! sequence header or libavcodec fails the whole stream on its dimensions.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, G2gError, MultiOutputElement, MultiOutputSink,
    OutputSink, PushOutcome, Segment,
};
use g2g_plugins::mkvdemux::{MkvDemux, MkvDemuxN, MkvStream};
use g2g_plugins::psdemux::parse_sequence_header;
use g2g_plugins::tsdemux::{TsDemux, TsDemuxN, TsStream};

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

impl MultiOutputSink for PortTap {
    fn port_count(&self) -> usize {
        self.ports.len()
    }

    fn push_to<'a>(
        &'a mut self,
        port: usize,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        self.ports[port].push(packet);
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m930-{}-{name}", std::process::id()))
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

fn bytes(frame: &Frame) -> Vec<u8> {
    frame
        .domain
        .as_system_slice()
        .expect("system frame")
        .to_vec()
}

/// The opening `Segment` of a packet run, asserted to precede the first frame,
/// with that frame's timestamp. Returns `(segment, first pts)`.
fn opening_segment(packets: &[PipelinePacket]) -> (Segment, u64) {
    let seg_at = packets
        .iter()
        .position(|p| matches!(p, PipelinePacket::Segment(_)))
        .expect("a stream-start segment is emitted");
    let frame_at = packets
        .iter()
        .position(|p| matches!(p, PipelinePacket::DataFrame(_)))
        .expect("frames follow");
    assert!(
        seg_at < frame_at,
        "the segment leads the first frame ({seg_at} vs {frame_at})"
    );
    let PipelinePacket::Segment(seg) = &packets[seg_at] else {
        unreachable!();
    };
    let PipelinePacket::DataFrame(f) = &packets[frame_at] else {
        unreachable!();
    };
    (*seg, f.timing.pts_ns)
}

/// The mapping every stream-start segment must establish: the first frame
/// presents at running time 0, however large its timestamp.
fn assert_maps_to_zero(seg: &Segment, first_pts: u64) {
    assert_eq!(seg.start, first_pts, "segment start is the first timestamp");
    assert_eq!(
        seg.to_running_time(first_pts),
        Some(0),
        "so the first frame presents immediately"
    );
}

// ---- fixtures ----

/// A transport stream whose timestamps start far from zero, the way a broadcast
/// capture's do. `-output_ts_offset` moves the muxer's clock.
fn author_ts(path: &Path, codec: &str, offset_s: &str) {
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=25:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-c:v",
        codec,
        "-g",
        "12",
        "-b:v",
        "800k",
        "-c:a",
        "aac",
        "-output_ts_offset",
        offset_s,
        "-f",
        "mpegts",
        path.to_str().unwrap(),
    ]);
}

/// Feed a whole file to a single-output demuxer in 4 KiB chunks.
async fn run_single(el: &mut dyn AsyncElementDyn, file: &[u8]) -> Collect {
    let mut sink = Collect::default();
    for chunk in file.chunks(4096) {
        el.feed(data_frame(chunk.to_vec()), &mut sink).await;
    }
    el.feed(PipelinePacket::Eos, &mut sink).await;
    sink
}

/// Minimal object-safe shim so one runner drives either demuxer.
trait AsyncElementDyn {
    fn feed<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut Collect,
    ) -> Pin<Box<dyn Future<Output = ()> + 'a>>;
}

impl<T: AsyncElement> AsyncElementDyn for T {
    fn feed<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut Collect,
    ) -> Pin<Box<dyn Future<Output = ()> + 'a>> {
        Box::pin(async move {
            self.process(packet, out).await.expect("demux a chunk");
        })
    }
}

// ---- MPEG-TS ----

#[tokio::test]
async fn tsdemux_opens_on_a_stream_start_segment() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let path = temp_path("offset.ts");
    author_ts(&path, "libx264", "3600");
    let file = std::fs::read(&path).expect("read the fixture");

    let mut el = TsDemux::new().with_stream(TsStream::H264);
    el.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .expect("tsdemux accepts a transport stream");
    let sink = run_single(&mut el, &file).await;

    let (seg, first_pts) = opening_segment(&sink.packets);
    assert!(
        first_pts > 3_000_000_000,
        "the capture's timestamps start an hour in: {first_pts}"
    );
    assert_maps_to_zero(&seg, first_pts);
    // And only one: a segment per frame would reset the sink's mapping.
    let segments = sink
        .packets
        .iter()
        .filter(|p| matches!(p, PipelinePacket::Segment(_)))
        .count();
    assert_eq!(segments, 1, "the stream-start segment goes out once");
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn tsdemuxn_ports_share_one_running_time_origin() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let path = temp_path("fanout.ts");
    author_ts(&path, "libx264", "3600");
    let file = std::fs::read(&path).expect("read the fixture");

    let mut el = TsDemuxN::new(Vec::from([TsStream::H264, TsStream::Aac]));
    el.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .expect("configure");
    let mut tap = PortTap::new(2);
    for chunk in file.chunks(4096) {
        el.process(data_frame(chunk.to_vec()), &mut tap)
            .await
            .expect("demux a chunk");
    }
    el.process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("flush");

    let (video_seg, video_pts) = opening_segment(&tap.ports[0]);
    let (audio_seg, _) = opening_segment(&tap.ports[1]);
    assert_maps_to_zero(&video_seg, video_pts);
    // One origin across the ports: a per-port base would shift A against V by
    // the difference between their first timestamps.
    assert_eq!(
        video_seg.start, audio_seg.start,
        "both ports map from the same origin"
    );
    let _ = std::fs::remove_file(path);
}

/// A transport stream joined mid-GOP: MPEG-2 pictures carry no geometry, so the
/// demuxer must drop to the first sequence header rather than hand a decoder a
/// stream it cannot size.
#[tokio::test]
async fn tsdemux_mpeg2_joins_at_a_sequence_header() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let path = temp_path("mid.ts");
    author_ts(&path, "mpeg2video", "0");
    let whole = std::fs::read(&path).expect("read the fixture");

    // Cut at a packet boundary well into the stream, which lands mid-GOP: the
    // PAT/PMT recur, so the demuxer still routes, but the first access units it
    // reassembles are dependent pictures.
    let sync = whole.iter().position(|&b| b == 0x47).expect("a sync byte");
    let packets = (whole.len() - sync) / 188;
    let cut = sync + (packets / 3) * 188;
    let tail = &whole[cut..];

    let mut el = TsDemux::new().with_stream(TsStream::Mpeg2);
    el.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .expect("configure");
    let sink = run_single(&mut el, tail).await;

    let frames: Vec<&Frame> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();
    assert!(!frames.is_empty(), "the join recovers at the next sequence");
    assert!(
        parse_sequence_header(&bytes(frames[0])).is_some(),
        "the first unit out carries the sequence header"
    );
    let (seg, first_pts) = opening_segment(&sink.packets);
    assert_maps_to_zero(&seg, first_pts);

    // The whole file, read from its start, keeps every unit: the drop only ever
    // applies to what precedes the first sequence header.
    let mut el = TsDemux::new().with_stream(TsStream::Mpeg2);
    el.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .expect("configure");
    let full = run_single(&mut el, &whole).await;
    let full_frames = full
        .packets
        .iter()
        .filter(|p| matches!(p, PipelinePacket::DataFrame(_)))
        .count();
    assert!(
        full_frames > frames.len(),
        "the whole file yields more than its tail ({full_frames} vs {})",
        frames.len()
    );
    let _ = std::fs::remove_file(path);
}

// ---- Matroska ----

#[tokio::test]
async fn mkvdemux_opens_on_a_stream_start_segment() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let path = temp_path("clip.mkv");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=25:duration=2",
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "300k",
        "-f",
        "matroska",
        path.to_str().unwrap(),
    ]);
    let file = std::fs::read(&path).expect("read the fixture");

    let mut el = MkvDemux::new().with_stream(MkvStream::Vp9);
    el.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Matroska,
    })
    .expect("matroskademux accepts a Matroska stream");
    let sink = run_single(&mut el, &file).await;

    let (seg, first_pts) = opening_segment(&sink.packets);
    assert_maps_to_zero(&seg, first_pts);
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn mkvdemuxn_ports_share_one_running_time_origin() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let path = temp_path("av.mkv");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=25:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "300k",
        "-c:a",
        "libopus",
        "-f",
        "matroska",
        path.to_str().unwrap(),
    ]);
    let file = std::fs::read(&path).expect("read the fixture");

    let mut el = MkvDemuxN::new(Vec::from([MkvStream::Vp9, MkvStream::Opus]));
    el.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Matroska,
    })
    .expect("configure");
    let mut tap = PortTap::new(2);
    for chunk in file.chunks(4096) {
        el.process(data_frame(chunk.to_vec()), &mut tap)
            .await
            .expect("demux a chunk");
    }
    el.process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("flush");

    let (video_seg, video_pts) = opening_segment(&tap.ports[0]);
    let (audio_seg, _) = opening_segment(&tap.ports[1]);
    assert_maps_to_zero(&video_seg, video_pts);
    assert_eq!(
        video_seg.start, audio_seg.start,
        "both ports map from the same origin"
    );
    let _ = std::fs::remove_file(path);
}

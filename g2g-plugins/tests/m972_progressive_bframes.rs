//! M972: the progressive MP4 layout with real B-frames. A reordered H.264
//! stream presents its frames in a different order than it decodes them, which
//! `ctts` records as `pts - dts` per sample. `Mp4DemuxN` reads the source's
//! `ctts` and carries the decode timestamp on the frame, and `Mp4MuxN`'s
//! progressive layout puts it back, so a remux keeps every frame's composition
//! time.
//!
//! Oracle: ffprobe's per-packet `pts_time` / `dts_time` on the source and on the
//! remux, compared relative to each file's first packet (g2g writes no edit
//! list, so the two timelines share their shape, not their origin), plus
//! ffmpeg's own decode of the remux.
#![cfg(feature = "std")]

use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    ByteStreamEncoding, Caps, G2gError, MultiInputElement, MultiOutputElement, MultiOutputSink,
    OutputSink, PushOutcome,
};
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};
use g2g_plugins::mp4muxn::Mp4MuxN;

const WIDTH: usize = 320;
const HEIGHT: usize = 240;
/// 1 s at 25 fps.
const FRAME_COUNT: usize = 25;
/// Half a 90 kHz tick, the muxer's video timescale, in seconds.
const TICK_SLACK: f64 = 0.5 / 90_000.0;

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

/// Per-port capture of the demuxer's output.
#[derive(Default)]
struct PortCapture {
    ports: Vec<Vec<(Vec<u8>, FrameTiming)>>,
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
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.ports[port].push((s.to_vec(), f.timing));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }

    fn port_count(&self) -> usize {
        self.ports.len()
    }
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
    std::env::temp_dir().join(format!("g2g-m972-{}-{name}", std::process::id()))
}

/// A 1 s 25 fps H.264 fixture with two consecutive B-frames per group, so the
/// stream really reorders (the encoder emits each anchor picture ahead of the
/// B-frames that reference it).
fn author_bframes(path: &Path) -> bool {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg(format!("testsrc2=size={WIDTH}x{HEIGHT}:rate=25:duration=1"))
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .args(["-bf", "2", "-g", "15"])
        .arg(path)
        .status()
        .expect("run ffmpeg");
    status.success()
}

/// ffprobe's per-packet `(pts_time, dts_time)` for the first video stream, in
/// stored (decode) order.
fn packet_times(path: &Path) -> Vec<(f64, f64)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args([
            "-show_entries",
            "packet=pts_time,dts_time",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(out.status.success(), "ffprobe read {}", path.display());
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().trim_end_matches(',').split_once(','))
        .map(|(p, d)| (p.parse().expect("pts_time"), d.parse().expect("dts_time")))
        .collect()
}

/// Each time less the first packet's, so two files whose timelines start at a
/// different origin still compare.
fn relative(times: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let (p0, d0) = times[0];
    times.iter().map(|(p, d)| (p - p0, d - d0)).collect()
}

/// ffmpeg's decode of `path` to raw frames, failing on any decode complaint.
fn decode_video(path: &Path) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .output()
        .expect("run ffmpeg");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && err.is_empty(),
        "ffmpeg decoded {} cleanly: {err}",
        path.display()
    );
    out.stdout
}

/// Demux an MP4's first track into its frames plus its caps.
async fn demux_video(file: &[u8]) -> (Caps, Vec<(Vec<u8>, FrameTiming)>) {
    let streams = forwardable_streams(file);
    assert_eq!(streams.len(), 1, "the fixture has one track");
    let ports: Vec<Mp4Port> = streams
        .iter()
        .map(|s| Mp4Port {
            track_id: s.track_id,
            caps: s.caps.clone(),
        })
        .collect();
    let caps = streams[0].caps.clone();
    let mut demux = Mp4DemuxN::new(ports);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .expect("configure mp4demux");
    let mut tap = PortCapture::default();
    tap.ports.resize(1, Vec::new());
    demux
        .process(frame(file.to_vec(), FrameTiming::default()), &mut tap)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("demux eos");
    (caps, tap.ports.remove(0))
}

/// Mux one video track into a progressive (moov-at-end) MP4, in the order the
/// frames arrive, which is decode order.
async fn mux_progressive(caps: &Caps, frames: &[(Vec<u8>, FrameTiming)]) -> Vec<u8> {
    let mut mux = Mp4MuxN::new(1).with_fragmented(false);
    let mut sink = CaptureSink::default();
    mux.configure_pipeline(0, caps).expect("configure mp4mux");
    for (data, timing) in frames {
        mux.process(0, frame(data.clone(), *timing), &mut sink)
            .await
            .expect("mux");
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    sink.bytes
}

/// The demuxer carries a real decode timestamp: the fixture reorders, so its
/// frames arrive out of presentation order, each with a DTS ahead of its PTS.
#[tokio::test]
async fn demuxed_frames_carry_a_decode_timestamp() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("carry.mp4");
    assert!(author_bframes(&src), "ffmpeg authored the B-frame fixture");
    let bytes = std::fs::read(&src).expect("read fixture");
    let (_, frames) = demux_video(&bytes).await;
    let _ = std::fs::remove_file(&src);

    assert_eq!(frames.len(), FRAME_COUNT, "every frame demuxed");
    let reordered = frames.iter().filter(|(_, t)| t.dts_ns < t.pts_ns).count();
    assert!(
        reordered > FRAME_COUNT / 2,
        "most frames decode before they present, got {reordered}"
    );
    let decode_order_ok = frames.windows(2).all(|w| w[0].1.dts_ns < w[1].1.dts_ns);
    assert!(decode_order_ok, "frames arrive in decode order");
    let presents_out_of_order = frames.windows(2).any(|w| w[0].1.pts_ns > w[1].1.pts_ns);
    assert!(
        presents_out_of_order,
        "the fixture really reorders: some frame presents before its predecessor"
    );
}

/// A progressive remux of a reordered stream keeps every frame's composition
/// time: ffprobe reads back the source's own pts / dts shape, and ffmpeg decodes
/// the file without complaint.
#[tokio::test]
async fn progressive_remux_keeps_composition_times() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("remux-src.mp4");
    assert!(author_bframes(&src), "ffmpeg authored the B-frame fixture");
    let bytes = std::fs::read(&src).expect("read fixture");
    let (caps, frames) = demux_video(&bytes).await;
    let muxed = mux_progressive(&caps, &frames).await;

    let out = temp_path("remux-out.mp4");
    std::fs::write(&out, &muxed).expect("write remux");
    let reference = packet_times(&src);
    let remuxed = packet_times(&out);
    assert_eq!(reference.len(), FRAME_COUNT, "the source has every packet");
    assert_eq!(remuxed.len(), FRAME_COUNT, "the remux has every packet");
    for (i, ((pts, dts), (ref_pts, ref_dts))) in relative(&remuxed)
        .iter()
        .zip(relative(&reference))
        .enumerate()
    {
        assert!(
            (pts - ref_pts).abs() <= TICK_SLACK,
            "packet {i} presents at {pts}, source {ref_pts}"
        );
        assert!(
            (dts - ref_dts).abs() <= TICK_SLACK,
            "packet {i} decodes at {dts}, source {ref_dts}"
        );
    }

    let frame_bytes = WIDTH * HEIGHT * 3 / 2;
    assert_eq!(
        decode_video(&out).len(),
        FRAME_COUNT * frame_bytes,
        "ffmpeg decodes the remux to every frame"
    );
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
}

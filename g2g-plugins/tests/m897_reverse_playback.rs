//! M897 - reverse playback (`rate < 0`) through the MP4 path.
//!
//! A reverse seek reaches `Mp4Src`, which walks the sync samples (`stss`)
//! backward from the segment `stop` and emits one whole GOP at a time in decode
//! order, so a forward-only decoder can decode it. `GopReverse` sits after the
//! decoder and re-emits each decoded GOP in descending PTS, and the reverse
//! `Segment` maps those descending timestamps to ascending running time at the
//! sink (`Segment::to_running_time` measures reverse from `stop`).
//!
//! The fixture is a progressive (`stss`-carrying) MP4 built in-process from the
//! repo's Annex-B H.264 clip, four IDR-led GOPs of five frames.
#![cfg(feature = "std")]

use std::path::{Path, PathBuf};

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{
    AsyncClock, AsyncElement, Caps, Dim, G2gError, MultiInputElement, OutputSink, PipelineClock,
    PushOutcome, Rate, Seek, VideoCodec,
};
use g2g_plugins::gopreverse::GopReverse;
use g2g_plugins::mp4muxn::Mp4MuxN;
use g2g_plugins::mp4src::Mp4Src;
use g2g_plugins::syncsink::SyncSink;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const GOP: usize = 5;
const FRAME_NS: u64 = 33_333_333;

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30 << 16),
    }
}

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
        core::task::Poll::Ready({
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
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

    fn pts(&self) -> Vec<u64> {
        self.frames().iter().map(|f| f.timing.pts_ns).collect()
    }
}

/// A clock fixed at 0 that records every deadline it is asked to sleep until:
/// the running-time order in which the sink presented frames.
#[derive(Clone, Default)]
struct RecordingClock {
    deadlines: std::sync::Arc<std::sync::Mutex<Vec<u64>>>,
}

impl PipelineClock for RecordingClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

impl AsyncClock for RecordingClock {
    type SleepFuture<'a> = core::future::Ready<()>;
    fn sleep_until_ns(&self, deadline_ns: u64) -> core::future::Ready<()> {
        self.deadlines.lock().unwrap().push(deadline_ns);
        core::future::ready(())
    }
}

struct NullSink;
impl OutputSink for NullSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Byte offsets of each NAL payload (just past its start code).
fn start_code_offsets(data: &[u8]) -> Vec<usize> {
    let mut offs = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                offs.push(i + 3);
                i += 3;
                continue;
            }
            if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                offs.push(i + 4);
                i += 4;
                continue;
            }
        }
        i += 1;
    }
    offs
}

/// Split an Annex-B stream into per-picture access units: a VCL NAL (type 1/5)
/// closes an AU, carrying any preceding SPS/PPS/SEI with it.
fn split_access_units(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut units = Vec::new();
    let mut cur = Vec::new();
    let starts = start_code_offsets(stream);
    for (k, &begin) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(stream.len());
        let nal = &stream[begin..end];
        cur.extend_from_slice(&[0, 0, 0, 1]);
        cur.extend_from_slice(nal);
        if matches!(nal.first().map(|b| b & 0x1F), Some(1) | Some(5)) {
            units.push(core::mem::take(&mut cur));
        }
    }
    units
}

/// The repo's ten-frame clip (two IDR-led GOPs of five) played twice, so the
/// fixture has four GOPs and a mid-stream seek has somewhere to land.
fn access_units() -> Vec<Vec<u8>> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/h264_640x480.h264"
    );
    let es = std::fs::read(path).expect("read the H.264 fixture");
    let once = split_access_units(&es);
    assert_eq!(once.len(), 2 * GOP, "the clip is two GOPs of five");
    let mut aus = once.clone();
    aus.extend(once);
    aus
}

fn au_frame(bytes: Vec<u8>, index: usize) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: index as u64 * FRAME_NS,
            dts_ns: index as u64 * FRAME_NS,
            duration_ns: FRAME_NS,
            ..FrameTiming::default()
        },
        sequence: index as u64,
        meta: Default::default(),
    })
}

#[derive(Default)]
struct ByteSink {
    bytes: Vec<u8>,
}

impl OutputSink for ByteSink {
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

/// Write a progressive (non-fragmented, so `stss`-indexed) MP4 of the fixture
/// clip and return its path.
async fn fixture_mp4(name: &str) -> PathBuf {
    let mut mux = Mp4MuxN::new(1).with_fragmented(false);
    mux.configure_pipeline(0, &h264_caps()).expect("configure");
    let mut sink = ByteSink::default();
    for (i, au) in access_units().into_iter().enumerate() {
        mux.process(0, au_frame(au, i), &mut sink)
            .await
            .expect("mux");
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    let path = std::env::temp_dir().join(format!("g2g_m897_{}_{name}.mp4", std::process::id()));
    std::fs::write(&path, &sink.bytes).expect("write the fixture");
    path
}

/// Run `Mp4Src` over `path` to EOS, applying `seek` (if any) before the first
/// frame, and return everything it pushed.
async fn run_src(path: &Path, seek: Option<Seek>) -> Collect {
    let ctl = SeekController::new();
    let mut src = Mp4Src::new(path).with_seek(ctl.clone());
    let caps = src.intercept_caps().await.expect("probe the file");
    src.configure_pipeline(&caps).expect("configure");
    if let Some(s) = seek {
        ctl.seek(s);
    }
    let mut out = Collect::default();
    src.run(&mut out).await.expect("run to EOS");
    out
}

/// The forward PTS list (decode order), the reference every reverse expectation
/// is built from.
async fn forward_pts(path: &Path) -> Vec<u64> {
    let pts = run_src(path, None).await.pts();
    assert_eq!(pts.len(), 4 * GOP, "four GOPs of five samples");
    pts
}

/// Feed `packets` through a fresh `GopReverse`, returning what it emitted.
async fn through_gop_reverse(packets: Vec<PipelinePacket>) -> Collect {
    let mut rev = GopReverse::new();
    rev.configure_pipeline(&h264_caps()).expect("configure");
    let mut out = Collect::default();
    for p in packets {
        rev.process(p, &mut out).await.expect("reverse");
    }
    out
}

fn segment_of(packets: &[PipelinePacket]) -> g2g_core::Segment {
    packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::Segment(s) => Some(*s),
            _ => None,
        })
        .expect("the seek emitted a segment")
}

#[tokio::test]
async fn reverse_seek_walks_gops_backward_in_decode_order() {
    let path = fixture_mp4("walk").await;
    let forward = forward_pts(&path).await;
    let duration = *forward.last().expect("frames") + FRAME_NS;

    let out = run_src(&path, Some(Seek::reverse(0, duration))).await;

    // The flushing reverse seek announces itself before any frame.
    assert!(
        matches!(out.packets.first(), Some(PipelinePacket::Flush)),
        "a flushing seek flushes first"
    );
    let seg = segment_of(&out.packets);
    assert_eq!(seg.rate, -1.0);
    assert_eq!(seg.stop, Some(duration));

    // GOPs newest first, each one forward (decode order) so the decoder can
    // decode it: 10..14, then 5..9, then 0..4 ... in reverse GOP order.
    let expected: Vec<u64> = forward
        .chunks(GOP)
        .rev()
        .flat_map(|g| g.iter().copied())
        .collect();
    assert_eq!(out.pts(), expected);
    // Every GOP starts on a sync sample, so a decoder can resume there.
    for (i, f) in out.frames().iter().enumerate() {
        assert_eq!(
            f.timing.keyframe,
            i % GOP == 0,
            "frame {i} of the reverse walk"
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn gop_reverse_emits_each_gop_in_descending_pts() {
    let path = fixture_mp4("order").await;
    let forward = forward_pts(&path).await;
    let duration = *forward.last().expect("frames") + FRAME_NS;

    let out = run_src(&path, Some(Seek::reverse(0, duration))).await;
    let reversed = through_gop_reverse(out.packets).await;

    // Strictly decreasing PTS: reverse presentation order, end to start.
    let mut want = forward.clone();
    want.sort_unstable_by(|a, b| b.cmp(a));
    assert_eq!(reversed.pts(), want);
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn reverse_playback_presents_in_ascending_running_time() {
    let path = fixture_mp4("runtime").await;
    let forward = forward_pts(&path).await;
    let duration = *forward.last().expect("frames") + FRAME_NS;

    let out = run_src(&path, Some(Seek::reverse(0, duration))).await;
    let reversed = through_gop_reverse(out.packets).await;

    let clock = RecordingClock::default();
    let mut sink = SyncSink::new(clock.clone());
    sink.configure_pipeline(&h264_caps()).expect("configure");
    let mut null = NullSink;
    for p in reversed.packets {
        sink.process(p, &mut null).await.expect("present");
    }

    assert_eq!(
        sink.received(),
        forward.len() as u64,
        "every frame presented"
    );
    assert_eq!(sink.clipped(), 0, "the segment spans the whole file");
    let deadlines = clock.deadlines.lock().unwrap().clone();
    assert_eq!(deadlines.len(), forward.len());
    assert!(
        deadlines.windows(2).all(|w| w[0] < w[1]),
        "descending PTS maps to ascending running time: {deadlines:?}"
    );
    // The newest frame plays first (running time 0), the oldest last.
    assert_eq!(deadlines[0], duration - forward[forward.len() - 1]);
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn mid_stream_reverse_seek_lands_on_the_gop_boundary() {
    let path = fixture_mp4("midseek").await;
    let forward = forward_pts(&path).await;
    // A stop inside the third GOP (sample 12) and a start inside the second
    // (sample 7): playback covers GOPs 2 and 1 only.
    let seek = Seek::reverse(forward[7], forward[12]);

    let out = run_src(&path, Some(seek)).await;

    let expected: Vec<u64> = forward[GOP..3 * GOP]
        .chunks(GOP)
        .rev()
        .flatten()
        .copied()
        .collect();
    assert_eq!(
        out.pts(),
        expected,
        "the walk opens on the sync sample of the GOP holding `stop` and ends \
         with the GOP holding `start`"
    );
    assert!(
        out.frames()[0].timing.keyframe,
        "playback resumes on a keyframe"
    );
    assert_eq!(
        out.pts()[0],
        forward[2 * GOP],
        "the GOP holding `stop`, not the sample at `stop`"
    );

    // The samples above `stop` are emitted as decode references, and the sink
    // clips them out of the segment instead of presenting them.
    let reversed = through_gop_reverse(out.packets).await;
    let clock = RecordingClock::default();
    let mut sink = SyncSink::new(clock.clone());
    sink.configure_pipeline(&h264_caps()).expect("configure");
    let mut null = NullSink;
    for p in reversed.packets {
        sink.process(p, &mut null).await.expect("present");
    }
    assert_eq!(sink.clipped(), 4, "samples 13, 14 and 5, 6 are outside");
    assert_eq!(sink.received(), 6, "samples 7..12 are inside");
    let deadlines = clock.deadlines.lock().unwrap().clone();
    assert!(
        deadlines.windows(2).all(|w| w[0] < w[1]),
        "ascending running time: {deadlines:?}"
    );
    let _ = std::fs::remove_file(&path);
}

/// The decode leg: the same frames decoded through the reverse GOP walk are
/// bit-exact with a forward decode of the same file. Needs libavcodec.
#[cfg(feature = "ffmpeg")]
mod decode {
    use super::*;
    use g2g_core::runtime::{run_graph, GraphNode};
    use g2g_core::Graph;
    use g2g_plugins::ffmpegdec::{FfmpegVideoDec, OutputFormat};
    use std::collections::BTreeMap;

    /// Decode `packets` (which end in the source's `Eos`, draining the decoder)
    /// with a fresh software decoder. The decoder does not forward `Eos` (the
    /// runner does), so it is re-appended for the next stage.
    async fn decode(packets: Vec<PipelinePacket>) -> Collect {
        let mut dec = FfmpegVideoDec::new().with_output_format(OutputFormat::I420);
        let narrowed = dec.intercept_caps(&h264_caps()).expect("H.264 accepted");
        dec.configure_pipeline(&narrowed).expect("libavcodec opens");
        let mut out = Collect::default();
        for p in packets {
            dec.process(p, &mut out).await.expect("decode");
        }
        out.packets.push(PipelinePacket::Eos);
        out
    }

    fn pixels(c: &Collect) -> BTreeMap<u64, Vec<u8>> {
        c.frames()
            .iter()
            .map(|f| {
                let bytes = f.domain.as_system_slice().expect("system frame").to_vec();
                (f.timing.pts_ns, bytes)
            })
            .collect()
    }

    /// The whole chain under the real runner: `mp4src ! avdec_h264 ! gopreverse
    /// ! syncsink` with a reverse seek armed before the run. Nothing here is
    /// hand-fed, so the runner's own `Segment` / `Eos` delivery is what carries
    /// reverse playback.
    #[tokio::test]
    async fn reverse_playback_runs_as_a_graph() {
        let path = fixture_mp4("graph").await;
        let forward = forward_pts(&path).await;
        let duration = *forward.last().expect("frames") + FRAME_NS;

        let ctl = SeekController::new();
        ctl.seek(Seek::reverse(0, duration));
        let clock = RecordingClock::default();

        let mut g: Graph<GraphNode> = Graph::new();
        let src = g.add_source(GraphNode::source(Mp4Src::new(&path).with_seek(ctl)));
        let dec = g.add_transform(GraphNode::element(
            FfmpegVideoDec::new().with_output_format(OutputFormat::I420),
        ));
        let rev = g.add_transform(GraphNode::element(GopReverse::new()));
        let sink = g.add_sink(GraphNode::element(SyncSink::new(clock.clone())));
        g.link(src, dec).expect("link");
        g.link(dec, rev).expect("link");
        g.link(rev, sink).expect("link");

        let stats = run_graph(g, &clock, 4).await.expect("graph runs");
        assert_eq!(stats.frames_consumed, forward.len() as u64);

        let deadlines = clock.deadlines.lock().unwrap().clone();
        assert_eq!(deadlines.len(), forward.len(), "every frame scheduled");
        assert!(
            deadlines.windows(2).all(|w| w[0] < w[1]),
            "reverse playback presents in ascending running time: {deadlines:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn reverse_decode_is_bit_exact_with_forward_decode() {
        let path = fixture_mp4("decode").await;
        let forward_src = run_src(&path, None).await;
        let forward = pixels(&decode(forward_src.packets).await);
        assert_eq!(forward.len(), 4 * GOP, "every frame decoded forward");

        let duration = *forward.keys().last().expect("frames") + FRAME_NS;
        let reverse_src = run_src(&path, Some(Seek::reverse(0, duration))).await;
        let decoded = decode(reverse_src.packets).await;
        let reversed = through_gop_reverse(decoded.packets).await;

        let pts = reversed.pts();
        assert_eq!(pts.len(), forward.len(), "the same frames come back");
        assert!(
            pts.windows(2).all(|w| w[0] > w[1]),
            "decoded frames arrive newest first: {pts:?}"
        );
        for f in reversed.frames() {
            let want = forward
                .get(&f.timing.pts_ns)
                .unwrap_or_else(|| panic!("no forward frame at {}", f.timing.pts_ns));
            assert_eq!(
                f.domain.as_system_slice().expect("system frame"),
                want.as_slice(),
                "frame at {} differs from the forward decode",
                f.timing.pts_ns
            );
        }
        let _ = std::fs::remove_file(&path);
    }
}

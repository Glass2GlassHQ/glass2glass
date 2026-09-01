//! M997: `FfmpegVideoDec` acts on a relayed QoS report instead of only passing
//! it on, by skipping non-reference pictures (`AVDISCARD_NONREF`) while the sink
//! reports lateness.
//!
//! Two levels. `nonref_skip_sheds_work_and_recovers` drives the real decoder with
//! a downstream that reports lateness for the first part of the stream, and checks
//! the three things that matter: work was shed, only non-reference pictures went
//! missing (every frame still emitted is byte-identical to the same picture from a
//! full decode, so no reference chain was broken), and once the reports stop the
//! decoder is back to decoding everything. `qos_report_reaches_the_decoder_...`
//! runs a real graph (AU source -> decoder -> late sink) and shows the routing the
//! `qos` property picks: with it on the decoder consumes the report, with it off
//! the report walks past to the source, as before.
//!
//! Fixture: `tests/fixtures/h264_640x480_bframes.h264`, 12 pictures in two IDR-led
//! GOPs whose 6 disposable B-frames carry `nal_ref_idc == 0`. It is fed three
//! times over (each pass opens on an IDR with its parameter sets, so the
//! concatenation is a valid stream) to give the skip budget room to drain well
//! before the end.

#![cfg(all(target_os = "linux", feature = "ffmpeg"))]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{run_graph, GraphNodeRef, SourceLoop};
use g2g_core::{
    AsyncClock, Caps, ConfigureOutcome, Dim, G2gError, Graph, PipelineClock, PropValue, QosMessage,
    Rate, VideoCodec,
};
use g2g_plugins::ffmpegdec::FfmpegVideoDec;
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::syncsink::SyncSink;

const FIXTURE: &[u8] = include_bytes!("fixtures/h264_640x480_bframes.h264");
const FIXTURE_FRAMES: usize = 12;
const PASSES: usize = 3;
const FRAMES: usize = FIXTURE_FRAMES * PASSES;
const FRAME_NS: u64 = 33_333_333;
/// 100 ms behind: three frame periods, so each report arms a three-picture skip.
const LATE_NS: i64 = 100_000_000;
/// Access unit the downstream stops reporting lateness at, leaving the rest of
/// the stream as the recovery runway.
const PRESSURE_UNTIL: usize = 24;

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn data_frame(bytes: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        sequence: 0,
        meta: Default::default(),
    })
}

/// Collects frames, and reports lateness on every push while `pressure` is set:
/// the synthetic version of a sink running behind the clock.
struct QosSink {
    frames: Vec<(u64, Vec<u8>)>,
    pressure: Arc<AtomicBool>,
}

impl QosSink {
    fn new(pressure: Arc<AtomicBool>) -> Self {
        Self {
            frames: Vec::new(),
            pressure,
        }
    }

    fn idle() -> Self {
        Self::new(Arc::new(AtomicBool::new(false)))
    }
}

impl OutputSink for QosSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        use core::task::Poll;
        if let Some(PipelinePacket::DataFrame(f)) = packet.take() {
            let bytes = f
                .domain
                .as_system_slice()
                .expect("decoded system frame")
                .to_vec();
            self.frames.push((f.timing.pts_ns, bytes));
        }
        if self.pressure.load(Ordering::Relaxed) {
            return Poll::Ready(Ok(PushOutcome::Qos(QosMessage {
                jitter_ns: LATE_NS,
                running_time_ns: 0,
            })));
        }
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Split the fixture into access units with the real `H264Parse`, the framing the
/// decoder sees in a pipeline, then repeat it `PASSES` times.
async fn access_units() -> Vec<Vec<u8>> {
    let mut parse = H264Parse::reframing();
    parse
        .configure_pipeline(&h264_caps())
        .expect("h264parse accepts the stream");
    let mut sink = QosSink::idle();
    parse
        .process(data_frame(FIXTURE.to_vec(), 0), &mut sink)
        .await
        .expect("parse the fixture");
    parse
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain at EOS");
    let pass: Vec<Vec<u8>> = sink.frames.into_iter().map(|(_, b)| b).collect();
    assert_eq!(pass.len(), FIXTURE_FRAMES, "one access unit per picture");
    let mut aus = Vec::with_capacity(FRAMES);
    for _ in 0..PASSES {
        aus.extend(pass.iter().cloned());
    }
    aus
}

/// PTSs of the access units whose slices are non-reference (`nal_ref_idc == 0`),
/// ie the pictures nothing else depends on. The oracle for "only disposable work
/// was dropped".
fn nonreference_pts(aus: &[Vec<u8>]) -> BTreeSet<u64> {
    let mut out = BTreeSet::new();
    for (i, au) in aus.iter().enumerate() {
        let mut first_slice = None;
        let mut k = 0;
        while k + 3 < au.len() {
            if au[k] == 0 && au[k + 1] == 0 && au[k + 2] == 1 {
                let header = au[k + 3];
                if (1..=5).contains(&(header & 0x1f)) && first_slice.is_none() {
                    first_slice = Some(header);
                }
                k += 3;
            } else {
                k += 1;
            }
        }
        let header = first_slice.unwrap_or_else(|| panic!("access unit {i} carries no slice"));
        if (header >> 5) & 0x3 == 0 {
            out.insert(i as u64 * FRAME_NS);
        }
    }
    assert!(
        !out.is_empty(),
        "the fixture must contain non-reference pictures for this test to mean anything"
    );
    out
}

/// Feed every access unit through a decoder, with the downstream reporting
/// lateness until access unit `pressure_until`. Returns the decoder and its
/// emitted pts -> bytes.
async fn decode(
    mut dec: FfmpegVideoDec,
    pressure_until: usize,
) -> (FfmpegVideoDec, BTreeMap<u64, Vec<u8>>) {
    let aus = access_units().await;
    dec.configure_pipeline(&h264_caps())
        .expect("libavcodec opens the H.264 decoder");
    let pressure = Arc::new(AtomicBool::new(false));
    let mut sink = QosSink::new(pressure.clone());
    for (i, au) in aus.into_iter().enumerate() {
        pressure.store(i < pressure_until, Ordering::Relaxed);
        dec.process(data_frame(au, i as u64 * FRAME_NS), &mut sink)
            .await
            .expect("decode");
    }
    pressure.store(false, Ordering::Relaxed);
    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain at EOS");
    (dec, sink.frames.into_iter().collect())
}

#[tokio::test]
async fn nonref_skip_sheds_work_and_recovers() {
    let aus = access_units().await;
    let nonref = nonreference_pts(&aus);

    // Reference: no lateness reported, so every picture is decoded.
    let (idle, full) = decode(FfmpegVideoDec::new().with_qos(true), 0).await;
    assert_eq!(full.len(), FRAMES, "the full decode emits every picture");
    assert_eq!(idle.qos_skipped(), 0, "no report, nothing skipped");

    // Under sustained lateness the decoder sheds non-reference pictures.
    let (dec, shed) = decode(FfmpegVideoDec::new().with_qos(true), PRESSURE_UNTIL).await;
    assert!(dec.qos_skipped() > 0, "access units fed in skip mode");
    assert!(
        shed.len() < full.len(),
        "some decode work was shed ({} of {} pictures emitted)",
        shed.len(),
        full.len()
    );

    // Correctness: every frame still emitted is bit-identical to the same
    // picture from the full decode, and the missing ones are all non-reference,
    // so no reference chain was broken.
    for (pts, bytes) in &shed {
        let want = full
            .get(pts)
            .expect("emitted a picture the full decode did not");
        assert!(
            bytes == want,
            "picture at {pts} ns differs from the full decode: a dropped reference corrupted it"
        );
    }
    let missing: BTreeSet<u64> = full
        .keys()
        .filter(|p| !shed.contains_key(p))
        .copied()
        .collect();
    assert!(!missing.is_empty(), "nothing was dropped");
    assert!(
        missing.is_subset(&nonref),
        "dropped pictures {missing:?} are not all non-reference (non-reference: {nonref:?})"
    );

    // Recovery: the reports stopped, the budget drained, and the tail of the
    // stream is whole again.
    assert!(
        !dec.skipping_nonref(),
        "the budget ran out, so full decoding is back"
    );
    let budget = LATE_NS as u64 / FRAME_NS;
    let tail_start = (PRESSURE_UNTIL as u64 + budget + 1) * FRAME_NS;
    let tail: Vec<u64> = full.keys().filter(|p| **p >= tail_start).copied().collect();
    assert!(!tail.is_empty(), "the recovery runway is empty");
    for pts in tail {
        assert!(
            shed.contains_key(&pts),
            "picture at {pts} ns is past the skip budget and should have been decoded"
        );
    }

    eprintln!(
        "m997: {} of {FRAMES} pictures emitted under lateness, {} non-reference dropped, \
         {} access units fed in skip mode",
        shed.len(),
        missing.len(),
        dec.qos_skipped()
    );
}

#[tokio::test]
async fn qos_off_ignores_reports_and_decodes_everything() {
    let (dec, frames) = decode(FfmpegVideoDec::new(), PRESSURE_UNTIL).await;
    assert!(!dec.qos(), "off by default");
    assert_eq!(dec.qos_skipped(), 0, "reports are not acted on");
    assert_eq!(frames.len(), FRAMES, "every picture is decoded");
}

#[test]
fn qos_properties_round_trip_and_gate_the_relay() {
    let mut dec = FfmpegVideoDec::new();
    assert!(
        !AsyncElement::handles_qos(&dec),
        "the runner relays by default"
    );
    assert_eq!(dec.get_property("qos"), Some(PropValue::Bool(false)));
    assert_eq!(
        dec.get_property("max-skip-frames"),
        Some(PropValue::Uint(30))
    );

    dec.set_property("qos", PropValue::Bool(true))
        .expect("set qos");
    dec.set_property("max-skip-frames", PropValue::Uint(5))
        .expect("set max-skip-frames");
    assert!(
        AsyncElement::handles_qos(&dec),
        "with qos on the report stops here instead of being relayed"
    );
    assert_eq!(dec.get_property("qos"), Some(PropValue::Bool(true)));
    assert_eq!(dec.max_skip_frames(), 5);
}

/// Source emitting the fixture's access units, counting the QoS reports it is
/// handed (the reports that walked past the decoder).
struct AuSrc {
    aus: Vec<Vec<u8>>,
    qos_seen: Arc<AtomicU64>,
}

impl SourceLoop for AuSrc {
    type RunFuture<'a> = BoxFuture<'a, Result<u64, G2gError>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let aus = core::mem::take(&mut self.aus);
            let count = aus.len() as u64;
            for (i, au) in aus.into_iter().enumerate() {
                if let PushOutcome::Qos(_) = out.push(data_frame(au, i as u64 * FRAME_NS)).await? {
                    self.qos_seen.fetch_add(1, Ordering::Relaxed);
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(count)
        })
    }
}

/// A clock pinned far past every frame's deadline, with an instant sleep, so the
/// sink drops each frame as late and reports it upstream.
#[derive(Debug)]
struct LateClock;
impl PipelineClock for LateClock {
    fn now_ns(&self) -> u64 {
        10_000_000_000
    }
}
impl AsyncClock for LateClock {
    type SleepFuture<'a> = core::future::Ready<()>;
    fn sleep_until_ns(&self, _deadline_ns: u64) -> core::future::Ready<()> {
        core::future::ready(())
    }
}

/// `qos` decides who acts on the sink's lateness: the decoder that can shed
/// non-reference work, or (as before) the source at the top of the graph.
/// Returns the access units the decoder fed in skip mode and the reports the
/// source was handed.
async fn run_late_graph(qos: bool) -> (u64, u64) {
    let qos_seen = Arc::new(AtomicU64::new(0));
    let mut src = AuSrc {
        aus: access_units().await,
        qos_seen: qos_seen.clone(),
    };
    let mut dec = FfmpegVideoDec::new().with_qos(qos);
    let mut sink = SyncSink::new(LateClock).with_max_lateness_ns(0);

    {
        let mut g: Graph<GraphNodeRef> = Graph::new();
        let s = g.add_source(GraphNodeRef::source_ref(&mut src));
        let d = g.add_transform(GraphNodeRef::element_ref(&mut dec));
        let k = g.add_sink(GraphNodeRef::element_ref(&mut sink));
        g.link(s, d).expect("link");
        g.link(d, k).expect("link");
        run_graph(g, &LateClock, 4).await.expect("graph runs");
    }
    assert!(sink.dropped() > 0, "the late sink reported lateness");
    (dec.qos_skipped(), qos_seen.load(Ordering::Relaxed))
}

#[tokio::test]
async fn qos_report_reaches_the_decoder_instead_of_the_source() {
    let (skipped, source_reports) = run_late_graph(true).await;
    assert!(
        skipped > 0,
        "the decoder acted on the sink's report and shed non-reference work"
    );
    assert_eq!(
        source_reports, 0,
        "the report stopped at the decoder rather than being relayed"
    );

    let (skipped_off, source_reports_off) = run_late_graph(false).await;
    assert_eq!(skipped_off, 0, "with qos off the decoder sheds nothing");
    assert!(
        source_reports_off > 0,
        "and the report is relayed to the source, as M175 had it"
    );
}

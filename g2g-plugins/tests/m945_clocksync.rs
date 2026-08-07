//! M945 `clocksync`: a pass-through transform that holds each buffer until its
//! PTS is due on the pipeline clock, so an upstream running as fast as the CPU
//! allows feeds downstream at real time.
//!
//! The pacing tests drive a virtual clock that jumps to each deadline instead of
//! sleeping, so they assert the schedule the element asked for without spending
//! it. The launch test is the one real-time run: it proves the element paces
//! under the runner, with its own clock, from a `gst-launch` line.
//!
//! std-gated: `cargo test -p g2g-plugins --features std --test m945_clocksync`.
#![cfg(feature = "std")]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncClock, AsyncElement, Caps, ClockSync, DynAsyncClock, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, PropValue, PropertySpec, PushOutcome,
    RawVideoFormat,
};
use g2g_plugins::clocksync::ClockSyncTransform;
use g2g_plugins::registry::default_registry;

/// A clock whose timer jumps straight to the deadline: the element's pacing is
/// then readable as the time on the clock, at no wall-clock cost.
#[derive(Debug, Default)]
struct VirtualClock {
    now_ns: AtomicU64,
}

impl PipelineClock for VirtualClock {
    fn now_ns(&self) -> u64 {
        self.now_ns.load(Ordering::Acquire)
    }

    fn as_ticker(&self) -> Option<&dyn DynAsyncClock> {
        Some(self)
    }
}

impl AsyncClock for VirtualClock {
    type SleepFuture<'a>
        = Ready<()>
    where
        Self: 'a;

    fn sleep_until_ns(&self, deadline_ns: u64) -> Ready<()> {
        self.now_ns.fetch_max(deadline_ns, Ordering::AcqRel);
        ready(())
    }
}

#[derive(Default)]
struct CaptureSink {
    frames: Vec<u64>,
    caps: Vec<Caps>,
}

impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => self.frames.push(f.timing.pts_ns),
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: g2g_core::Dim::Fixed(2),
        height: g2g_core::Dim::Fixed(2),
        framerate: g2g_core::Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Progressive,
    }
}

fn frame(pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 16].into_boxed_slice())),
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        pts_ns,
    ))
}

/// Run `element` over frames at 0 / 33 / 66 ms on a virtual clock, returning the
/// clock time it ended at and what came out.
async fn pace(element: &mut ClockSyncTransform) -> (u64, Vec<u64>) {
    let clock = Arc::new(VirtualClock::default());
    element
        .configure_pipeline(&caps())
        .expect("transparent caps");
    element.set_clock_sync(ClockSync::new(clock.clone(), 0));
    let mut sink = CaptureSink::default();
    for pts in [0, 33_000_000, 66_000_000] {
        element.process(frame(pts), &mut sink).await.expect("paced");
    }
    (clock.now_ns(), sink.frames)
}

/// The point of the element: the stream leaves at the rate its PTS describes,
/// however fast it arrived.
#[tokio::test]
async fn each_buffer_waits_for_its_pts() {
    let mut e = ClockSyncTransform::new();
    let (elapsed, out) = pace(&mut e).await;
    assert_eq!(out, vec![0, 33_000_000, 66_000_000], "nothing is dropped");
    // The first buffer anchors the stream and goes straight out; the other two
    // are held until their PTS, so 66 ms of clock passes over three frames.
    assert_eq!(elapsed, 66_000_000);
    assert_eq!(e.forwarded(), 3);
}

/// `sync=false` is the escape hatch: an identity again, no clock consulted.
#[tokio::test]
async fn sync_false_forwards_immediately() {
    let mut e = ClockSyncTransform::new().with_sync(false);
    let (elapsed, out) = pace(&mut e).await;
    assert_eq!(out, vec![0, 33_000_000, 66_000_000]);
    assert_eq!(elapsed, 0, "no time was spent waiting");
}

/// `ts-offset` shifts the whole schedule, including the anchoring buffer.
#[tokio::test]
async fn ts_offset_delays_the_schedule() {
    let mut e = ClockSyncTransform::new().with_ts_offset_ns(10_000_000);
    let (elapsed, out) = pace(&mut e).await;
    assert_eq!(out.len(), 3);
    assert_eq!(elapsed, 76_000_000, "66 ms of stream, 10 ms later");

    // An offset past the frame period shifts the schedule once, not once per
    // buffer: 66 ms of stream, 200 ms later, not 3 x 200 ms.
    let mut e = ClockSyncTransform::new().with_ts_offset_ns(200_000_000);
    let (elapsed, _) = pace(&mut e).await;
    assert_eq!(elapsed, 266_000_000);

    // Negative: the schedule pulls in, and never earlier than arrival.
    let mut e = ClockSyncTransform::new().with_ts_offset_ns(-40_000_000);
    let (elapsed, _) = pace(&mut e).await;
    assert_eq!(elapsed, 26_000_000, "the first two deadlines clamp to zero");
}

/// Caps are none of this element's business: whatever arrives is what leaves.
#[tokio::test]
async fn caps_pass_through_unchanged() {
    let mut e = ClockSyncTransform::new();
    assert_eq!(e.intercept_caps(&caps()).expect("any caps"), caps());
    e.configure_pipeline(&caps()).expect("accepts");
    let mut sink = CaptureSink::default();
    e.process(PipelinePacket::CapsChanged(caps()), &mut sink)
        .await
        .expect("forwarded");
    assert_eq!(sink.caps, vec![caps()]);
}

/// Both halves of each knob, so a `gst-launch` line can set it.
#[test]
fn properties_round_trip() {
    let declares = |specs: &[PropertySpec], name: &str| specs.iter().any(|s| s.name == name);
    let mut e = ClockSyncTransform::new();
    assert!(declares(e.properties(), "sync"));
    assert!(declares(e.properties(), "ts-offset"));
    assert_eq!(e.get_property("sync"), Some(PropValue::Bool(true)));
    e.set_property("sync", PropValue::Bool(false)).unwrap();
    assert_eq!(e.get_property("sync"), Some(PropValue::Bool(false)));
    e.set_property("ts-offset", PropValue::Int(-5_000_000))
        .unwrap();
    assert_eq!(
        e.get_property("ts-offset"),
        Some(PropValue::Int(-5_000_000))
    );
    assert!(e.set_property("sync", PropValue::Int(1)).is_err());
    assert!(e.set_property("nope", PropValue::Bool(true)).is_err());
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// The real-time proof: the same 10 frames take a third of a second with
/// `clocksync` in the line and no time at all without it. The runner elects no
/// clock here (nothing in the line provides one), so this also covers the
/// element's own monotonic fallback.
#[tokio::test]
async fn a_launch_line_paces_in_real_time() {
    async fn run(line: &str) -> (core::time::Duration, g2g_core::runtime::RunStats) {
        let reg = default_registry();
        let graph = parse_launch(&reg, line).expect("pipeline parses");
        let started = Instant::now();
        let stats = run_graph(graph, &ZeroClock, 4)
            .await
            .expect("pipeline runs");
        (started.elapsed(), stats)
    }

    let src = "videotestsrc num-buffers=10 framerate=30/1 width=32 height=32";
    let (paced, stats) = run(&format!("{src} ! clocksync ! fakesink")).await;
    assert_eq!(stats.frames_consumed, 10);
    // Nine gaps of 33.3 ms after the anchoring frame: 300 ms, less a margin for
    // a timer that fires early.
    assert!(
        paced.as_millis() >= 280,
        "paced run took only {paced:?}, so nothing waited"
    );

    let (unpaced, stats) = run(&format!("{src} ! clocksync sync=false ! fakesink")).await;
    assert_eq!(stats.frames_consumed, 10);
    assert!(
        unpaced < paced / 2,
        "sync=false still took {unpaced:?} against the paced {paced:?}"
    );
}

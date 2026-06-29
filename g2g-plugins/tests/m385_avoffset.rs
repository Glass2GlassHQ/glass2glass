//! M385 - A/V offset transform. `AvOffset` shifts every frame's PTS/DTS by a
//! signed `offset` (ns), the g2g form of GStreamer playbin's `av-offset`: put one
//! on a branch of a multi-stream playback graph (typically audio) to re-align it.
//! Positive delays, negative advances (clamped at 0), zero is a pass-through.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::parse_launch;
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, OutputSink, PipelinePacket, PushOutcome, Rate,
    RawVideoFormat,
};

use g2g_plugins::avoffset::AvOffset;
use g2g_plugins::registry::default_registry;

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn frame(pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8]) as Box<[u8]>)),
        FrameTiming { pts_ns, dts_ns: pts_ns, duration_ns: 1000, ..FrameTiming::default() },
        0,
    ))
}

#[derive(Default)]
struct Collect {
    pts: Vec<u64>,
    dts: Vec<u64>,
}
impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                self.pts.push(f.timing.pts_ns);
                self.dts.push(f.timing.dts_ns);
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

async fn run_offset(offset_ns: i64, ptss: &[u64]) -> Collect {
    let mut el = AvOffset::new(offset_ns);
    el.configure_pipeline(&caps()).unwrap();
    let mut sink = Collect::default();
    for &p in ptss {
        el.process(frame(p), &mut sink).await.unwrap();
    }
    sink
}

#[tokio::test]
async fn positive_offset_delays_pts_and_dts() {
    let out = run_offset(500, &[0, 1000, 2000]).await;
    assert_eq!(out.pts, vec![500, 1500, 2500], "positive offset delays PTS");
    assert_eq!(out.dts, vec![500, 1500, 2500], "DTS shifts with PTS");
}

#[tokio::test]
async fn negative_offset_advances_and_clamps_at_zero() {
    let out = run_offset(-500, &[0, 1000]).await;
    // 0 - 500 clamps to 0 (a PTS cannot go negative); 1000 - 500 = 500.
    assert_eq!(out.pts, vec![0, 500], "negative offset advances, clamped at 0");
}

#[tokio::test]
async fn zero_offset_is_passthrough() {
    let out = run_offset(0, &[0, 1000, 2000]).await;
    assert_eq!(out.pts, vec![0, 1000, 2000], "zero offset leaves the timeline untouched");
}

#[test]
fn registered_with_offset_property() {
    let reg = default_registry();
    // Registered under the gst-style name with a settable signed `offset`.
    let mut el = reg.make_element("avoffset").expect("avoffset is registered");
    el.set_property("offset", g2g_core::PropValue::Int(40_000_000)).expect("offset is settable");
    assert_eq!(el.get_property("offset"), Some(g2g_core::PropValue::Int(40_000_000)));

    // And it parses + applies from a text pipeline (the av-offset use: delay audio).
    let graph = parse_launch(
        &reg,
        "audiotestsrc num-buffers=1 ! avoffset offset=40000000 ! fakesink",
    )
    .expect("pipeline with avoffset builds");
    assert_eq!(graph.node_count(), 3, "source, avoffset, sink");
}

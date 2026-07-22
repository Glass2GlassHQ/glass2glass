//! M466 `gstwrap`: host a real GStreamer element inside a g2g graph.
//!
//! Needs the host GStreamer runtime + dev libs and `gst-plugins-good` (for
//! `videoflip`); like the g2g-bridge smoke scripts, run it locally, not in CI:
//!
//! ```sh
//! cargo test -p g2g-plugins --features gstreamer --test gstwrap
//! ```
//!
//! The test drives the element directly (a crafted input frame + a capturing
//! output sink, the g2g graph boundary) rather than through `parse_launch`,
//! because the launch DSL v1 cannot carry a quoted property value with spaces
//! (`element="videoflip method=horizontal-flip"`). It asserts the pixels come
//! back horizontally flipped, which only a real GStreamer `videoflip` produces.

use g2g_core::element::BoxFuture;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, G2gError, OutputSink, PipelineClock, PipelinePacket, PropValue, PushOutcome,
};

use g2g_plugins::capsfilter::parse_caps;
use g2g_plugins::gstwrap::GstWrap;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Collects the bytes of every `DataFrame` the element emits.
#[derive(Default)]
struct Collect {
    frames: Vec<Vec<u8>>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[tokio::test]
async fn hosts_a_real_gstreamer_videoflip() {
    // A 2x1 RGBA frame: left pixel opaque white, right pixel a distinct colour.
    let caps =
        parse_caps("video/x-raw,format=RGBA,width=2,height=1,framerate=1/1").expect("caps parse");

    let mut el = GstWrap::new();
    el.set_property(
        "element",
        PropValue::Str("videoflip method=horizontal-flip".into()),
    )
    .expect("element property");
    el.configure_pipeline(&caps)
        .expect("gst pipeline builds (needs host GStreamer + gst-plugins-good videoflip)");

    let input: Vec<u8> = vec![0xFF, 0xFF, 0xFF, 0xFF, 0x10, 0x20, 0x30, 0x40];
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(input.into_boxed_slice())),
        FrameTiming {
            pts_ns: 0,
            dts_ns: 0,
            ..FrameTiming::default()
        },
        0,
    );

    let mut sink = Collect::default();
    el.process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("process frame");
    // EOS flushes videoflip's buffered frame; drain collects it.
    el.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("process eos");

    assert_eq!(
        sink.frames.len(),
        1,
        "the hosted GStreamer element produced one frame"
    );
    // Horizontal flip of a 2x1 image swaps the two pixels.
    assert_eq!(
        sink.frames[0],
        vec![0x10, 0x20, 0x30, 0x40, 0xFF, 0xFF, 0xFF, 0xFF],
        "pixels came back horizontally flipped by the real GStreamer videoflip"
    );
}

/// The quote-aware launch tokenizer carries a multi-word element description into
/// `gstwrap` from a gst-launch line, so a hosted GStreamer element runs straight
/// from `g2g-launch` (not only via the programmatic API).
#[tokio::test]
async fn runs_from_a_launch_line_with_a_spaced_property() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=3 ! gstwrap element=\"videoflip method=horizontal-flip\" ! fakesink",
    )
    .expect("quoted gstwrap line parses and builds");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("pipeline runs");
    assert_eq!(
        stats.frames_consumed, 3,
        "all frames flowed through the hosted GStreamer element"
    );
}

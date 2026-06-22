//! A third-party g2g element, registered and run via the text launcher.
//!
//! `cargo run -p g2g-plugins --features std --example third_party_element`
//!
//! Shows the whole extension path a downstream crate follows against the
//! published `g2g-core` / `g2g-plugins` (no g2g source changes): implement
//! `AsyncElement`, add `PadTemplates` + `metadata` (so `g2g-inspect` sees it),
//! expose a `register(&mut Registry)`, then use the element by name in a
//! `gst-launch` line. A source is the same with `SourceLoop` + `register_source`;
//! a muxer with `MultiInputElement` + `register_muxer`.

use core::future::Future;
use core::pin::Pin;

use g2g_core::runtime::{parse_launch, run_graph, LaunchFactory, Registry};
use g2g_core::{
    AsyncElement, Caps, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

/// A trivial pass-through video transform that counts frames. Swap the body of
/// `process` for real per-frame work.
#[derive(Debug, Default)]
struct FrameCounter {
    seen: u64,
    configured: bool,
}

impl AsyncElement for FrameCounter {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        // Accept whatever flows in (a real element would narrow here).
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    self.seen += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // The runner emits the single EOS; a transform must not forward it.
                PipelinePacket::Eos => {}
                // CapsChanged / Flush / Segment flow through unchanged.
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Frame counter",
            "Filter/Analyzer/Video",
            "Counts data frames and forwards them unchanged.",
            "third-party",
        )
    }
}

impl PadTemplates for FrameCounter {
    fn pad_templates() -> Vec<PadTemplate> {
        let any = CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        });
        Vec::from([PadTemplate::sink(any.clone()), PadTemplate::source(any)])
    }
}

/// The convention: a downstream crate exposes `register` so an application can
/// add its elements to any registry (exactly what `g2g_python::register` does).
fn register(registry: &mut Registry) {
    registry.register_launch(LaunchFactory::of::<FrameCounter>("framecounter", || {
        Box::new(FrameCounter::default())
    }));
}

fn main() {
    let mut registry = default_registry();
    register(&mut registry);

    let line = "videotestsrc num-buffers=5 ! framecounter ! fakesink";
    let graph = parse_launch(&registry, line).expect("pipeline parses");

    let rt = tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
    let stats = rt.block_on(run_graph(graph, &WallClock::new(), 4)).expect("pipeline runs");
    println!("ran `{line}`");
    println!("frames emitted: {}, consumed: {}", stats.frames_emitted, stats.frames_consumed);
}

//! An out-of-tree third-party g2g plugin, built as a dynamically loadable
//! `cdylib`. This is the whole author workflow for extending a packaged
//! `g2g-launch` without recompiling g2g (DESIGN_TODO "Dynamic plugin loading via
//! cargo", M201):
//!
//! 1. `cargo new --lib`, set `crate-type = ["cdylib"]`.
//! 2. Depend on `g2g-core` (the element traits) + `g2g-plugin` (the SDK).
//! 3. Implement `AsyncElement` + `PadTemplates` for your element.
//! 4. `declare_plugin!` to export the C-ABI entry points.
//! 5. `cargo build --release`, drop the `.so` in `$G2G_PLUGIN_PATH`, and use the
//!    element by name in a `gst-launch` line.

use core::future::Future;
use core::pin::Pin;

use g2g_core::{
    AsyncElement, Caps, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, Rate, RawVideoFormat,
};

/// A trivial pass-through video transform that counts the frames it sees. The
/// stand-in for whatever real per-frame work a third-party element does.
#[derive(Debug, Default)]
pub struct ExampleFilter {
    seen: u64,
}

impl AsyncElement for ExampleFilter {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Example filter",
            "Filter/Effect/Video",
            "Counts data frames and forwards them unchanged (dynamic-plugin demo).",
            "third-party",
        )
    }
}

impl PadTemplates for ExampleFilter {
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

// Export the C-ABI entry points (`g2g_plugin_abi` + `g2g_plugin_register`) the
// host loader looks up. The element is then usable as `examplefilter`.
g2g_plugin::declare_plugin! {
    elements: [
        ("examplefilter", ExampleFilter, || Box::new(ExampleFilter::default())),
    ]
}

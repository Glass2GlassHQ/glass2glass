//! `PyAggregator`: a gst-python-ml batched element hosted as a g2g muxer.
//!
//! N-in-1-out: collects one frame from each contributing input (via
//! [`InputAggregator`]), runs one Python batch call, and emits a single anchor
//! frame carrying any aggregate metadata. This is the `BaseAggregator`
//! (multi-pad batched inference) shape on g2g's [`MultiInputElement`].
//!
//! The Python contract is `g2g_process_batch(buffers, width, height, fmt, meta)`
//! where `buffers` is a list of writable buffer-protocol views (one per
//! contributing input), `meta` is the analytics sink. Because `MultiInputElement`
//! is N-in-1-out, only the anchor (input-0) frame is emitted; the aggregate
//! result travels as the anchor's `AnalyticsMeta` (the batched-inference-attaches
//! -detections use). Per-stream results would need a demux, which the trait does
//! not provide. v1 assumes every input shares one geometry/format (the output);
//! `batch_size`-style temporal accumulation and per-input formats are follow-ups.

use core::future::Future;
use core::pin::Pin;

use g2g_core::{
    Caps, ConfigureOutcome, Dim, Frame, G2gError, InputAggregator, MultiInputElement, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

/// A gst-python-ml batched element hosted as a first-class g2g aggregator.
#[derive(Debug)]
pub struct PyAggregator {
    // Read only when spawning the worker (the `python` build).
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    module: String,
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    class: String,
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    draw_label: bool,
    inputs: usize,
    /// Caps accepted on every input pad (and produced on the output).
    accept: Caps,
    /// The negotiated caps, captured at configure time (shared by all inputs and
    /// the output in v1).
    fixed: Option<Caps>,
    agg: InputAggregator<Frame>,
    emitted: u64,
    /// The hosted Python element on its GIL-owning worker thread, spawned once
    /// at the first input's configure. Present only in the `python` build.
    #[cfg(feature = "python")]
    worker: Option<crate::host::PyWorker>,
}

impl PyAggregator {
    /// Host `class` from Python `module` as an `inputs`-way batching aggregator.
    pub fn new(module: impl Into<String>, class: impl Into<String>, inputs: usize) -> Self {
        Self {
            module: module.into(),
            class: class.into(),
            draw_label: false,
            inputs,
            accept: Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            },
            fixed: None,
            agg: InputAggregator::new(inputs),
            emitted: 0,
            #[cfg(feature = "python")]
            worker: None,
        }
    }

    /// Override the accepted (and produced) caps.
    pub fn with_accept(mut self, caps: Caps) -> Self {
        self.accept = caps;
        self
    }

    /// Set the `draw-label` flag forwarded to the Python element.
    pub fn with_draw_label(mut self, on: bool) -> Self {
        self.draw_label = on;
        self
    }

    /// Count of batches emitted downstream. Useful in tests.
    pub fn emitted_count(&self) -> u64 {
        self.emitted
    }

    #[cfg(feature = "python")]
    async fn run_batch(&self, frames: Vec<Frame>, caps: &Caps) -> Result<Vec<Frame>, G2gError> {
        self.worker
            .as_ref()
            .ok_or(G2gError::NotConfigured)?
            .run_batch(frames, caps)
            .await
    }

    #[cfg(not(feature = "python"))]
    async fn run_batch(&self, _frames: Vec<Frame>, _caps: &Caps) -> Result<Vec<Frame>, G2gError> {
        Err(G2gError::UnsupportedDomain)
    }

    /// Emit every batch currently complete: one frame per contributing input ->
    /// one Python batch call -> push the anchor frame (carrying metadata).
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let caps = self.fixed.clone().ok_or(G2gError::NotConfigured)?;
        while let Some(round) = self.agg.take_round() {
            let frames: Vec<Frame> = round.into_iter().map(|(_input, frame)| frame).collect();
            let processed = self.run_batch(frames, &caps).await?;
            if let Some(anchor) = processed.into_iter().next() {
                out.push(PipelinePacket::DataFrame(anchor)).await?;
                self.emitted += 1;
            }
        }
        Ok(())
    }
}

impl MultiInputElement for PyAggregator {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.accept)
    }

    fn configure_pipeline(
        &mut self,
        _input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        absolute_caps.intersect(&self.accept)?;
        self.fixed = Some(absolute_caps.clone());
        #[cfg(feature = "python")]
        {
            if self.module.is_empty() || self.class.is_empty() {
                return Err(G2gError::NotConfigured);
            }
            // Spawn the worker once (configure is called per input).
            if self.worker.is_none() {
                // Property forwarding to the hosted aggregator instance is a
                // follow-up; transforms (PyTransform) forward theirs today.
                self.worker = Some(crate::host::PyWorker::spawn(
                    &self.module,
                    &self.class,
                    self.draw_label,
                    &[],
                )?);
            }
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.fixed.clone().ok_or(G2gError::NotConfigured)
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    self.agg.push(input, frame);
                    self.drain(out).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // v1: every input shares the fixed caps; validate and do not
                    // re-emit (the merged output caps are unchanged).
                    c.intersect(&self.accept)?;
                }
                // Per-input EOS: this input contributes its drained frames, then
                // drops out of future rounds. The runner emits the merged EOS,
                // so the element must not forward it.
                PipelinePacket::Eos => {
                    self.agg.mark_ended(input);
                    self.drain(out).await?;
                }
                // Segment / Flush are stream-control the fan-in runner owns;
                // a batching muxer has nothing to add.
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PYAGGREGATOR_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "module" => {
                self.module = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "class" => {
                self.class = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "draw-label" => {
                self.draw_label = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "module" => Some(PropValue::Str(self.module.clone())),
            "class" => Some(PropValue::Str(self.class.clone())),
            "draw-label" => Some(PropValue::Bool(self.draw_label)),
            _ => None,
        }
    }
}

/// `PyAggregator`'s settable properties (the runtime / `gst-launch` face). The
/// input count comes from link degree (the muxer factory), not a property.
static PYAGGREGATOR_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "module",
        PropKind::Str,
        "Python module to import (the aggregator element)",
    ),
    PropertySpec::new(
        "class",
        PropKind::Str,
        "class within the module to instantiate",
    ),
    PropertySpec::new(
        "draw-label",
        PropKind::Bool,
        "overlay the inferred label on the anchor frame",
    )
    .with_default("false"),
];

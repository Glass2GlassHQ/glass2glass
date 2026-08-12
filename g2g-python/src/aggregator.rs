//! `PyAggregator`: a gst-python-ml batched element hosted as a g2g muxer.
//!
//! N-in-1-out: collects one frame from each contributing input (via
//! [`InputAggregator`]), runs one Python batch call, and emits a single anchor
//! frame carrying any aggregate metadata. This is the `BaseAggregator`
//! (multi-pad batched inference) shape on g2g's [`MultiInputElement`].
//!
//! The Python contract is `g2g_process_batch(buffers, width, height, fmt, meta)`
//! where `buffers` is a list of writable buffer-protocol views (one per
//! contributing input), `meta` is the analytics sink. GPU-resident inputs take the
//! CUDA shape instead (M986): `g2g_process_cuda_batch(planes, width, height, meta)`
//! with one `(luma, chroma)` `__cuda_array_interface__` pair per input, so a
//! batched detector reads every stream's decoded surface with no readback. Because `MultiInputElement`
//! is N-in-1-out, only the anchor (input-0) frame is emitted; the aggregate
//! result travels as the anchor's `AnalyticsMeta` (the batched-inference-attaches
//! -detections use). Per-stream results would need a demux, which the trait does
//! not provide. v1 assumes every input shares one geometry/format;
//! `batch_size`-style temporal accumulation and per-input formats are follow-ups.
//! The output may be a different media type than the input (`output-caps=`), for
//! the audio-in / text-out families.

use core::future::Future;
use core::pin::Pin;

use g2g_core::{
    Caps, ConfigureOutcome, Dim, Frame, G2gError, InputAggregator, MultiInputElement, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

use crate::props::{fixed_caps, hosted_element_props};

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
    /// Whether the hosted element batches GPU-resident CUDA frames, read through
    /// `g2g_process_cuda_batch` and passed through untouched.
    cuda_frames: bool,
    inputs: usize,
    /// Caps accepted on every input pad.
    accept: Caps,
    /// Caps produced on the output, when the hosted element emits a different
    /// media type than it reads (audio in / text out, and the rest of the
    /// gst-python-ml aggregator families). Unset means it emits what it read.
    produce: Option<Caps>,
    /// The negotiated input caps, captured at configure time (shared by all
    /// inputs in v1).
    fixed: Option<Caps>,
    agg: InputAggregator<Frame>,
    emitted: u64,
    /// Element properties forwarded verbatim to the hosted Python instance at
    /// spawn, in the order they were set.
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    params: Vec<(String, PropValue)>,
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
            cuda_frames: false,
            inputs,
            accept: Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
            },
            produce: None,
            fixed: None,
            agg: InputAggregator::new(inputs),
            emitted: 0,
            params: Vec::new(),
            #[cfg(feature = "python")]
            worker: None,
        }
    }

    /// Override the accepted input caps.
    pub fn with_accept(mut self, caps: Caps) -> Self {
        self.accept = caps;
        self
    }

    /// Emit `caps` downstream instead of the negotiated input caps, for a hosted
    /// element that changes media type.
    pub fn with_produce(mut self, caps: Caps) -> Self {
        self.produce = Some(caps);
        self
    }

    /// Set the `draw-label` flag forwarded to the Python element.
    pub fn with_draw_label(mut self, on: bool) -> Self {
        self.draw_label = on;
        self
    }

    /// Batch GPU-resident CUDA frames: they reach the hosted element as
    /// `__cuda_array_interface__` planes through `g2g_process_cuda_batch`, and the
    /// anchor flows on still device-resident. Also drives what this element asks
    /// each input branch to allocate
    /// (see [`MultiInputElement::propose_allocation_for_input`]).
    pub fn with_cuda_frames(mut self, on: bool) -> Self {
        self.cuda_frames = on;
        self
    }

    /// Count of frames emitted downstream, which is one per batch unless the
    /// hosted element emits several buffers from one. Useful in tests.
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
    /// one Python batch call -> push the anchor frame (carrying metadata), or
    /// each of the buffers the element emitted in its stead.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let caps = self.fixed.clone().ok_or(G2gError::NotConfigured)?;
        while let Some(round) = self.agg.take_round() {
            let frames: Vec<Frame> = round.into_iter().map(|(_input, frame)| frame).collect();
            for processed in self.run_batch(frames, &caps).await? {
                out.push(PipelinePacket::DataFrame(processed)).await?;
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
                self.worker = Some(crate::host::PyWorker::spawn(
                    &self.module,
                    &self.class,
                    self.draw_label,
                    &self.params,
                )?);
            }
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.produce
            .clone()
            .or_else(|| self.fixed.clone())
            .ok_or(G2gError::NotConfigured)
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

    /// Ask every input's branch to allocate in the domain the hosted code reads,
    /// so a decoder that can do either keeps its frames device-resident for a
    /// `cuda-frames` batch (the per-pad form of what `PyTransform` proposes). Only
    /// the domain and the frame size are constrained: this element allocates
    /// nothing of its own.
    fn propose_allocation_for_input(
        &self,
        _input: usize,
        caps: &Caps,
    ) -> Option<g2g_core::AllocationParams> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        } = caps
        else {
            return None;
        };
        let size = crate::format::frame_bytes(*format, *width, *height);
        Some(if self.cuda_frames {
            g2g_core::AllocationParams::cuda(size, 1, 1)
        } else {
            g2g_core::AllocationParams::system(size, 1)
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
            "cuda-frames" => {
                self.cuda_frames = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "input-caps" => {
                self.accept = fixed_caps(value.as_str().ok_or(PropError::Type)?)?;
                Ok(())
            }
            "output-caps" => {
                self.produce = Some(fixed_caps(value.as_str().ok_or(PropError::Type)?)?);
                Ok(())
            }
            // Any other declared property is forwarded to the hosted Python
            // instance, the same way `PyTransform` forwards its own.
            other if PYAGGREGATOR_PROPS.iter().any(|s| s.name == other) => {
                if let Some(slot) = self.params.iter_mut().find(|(k, _)| k == other) {
                    slot.1 = value;
                } else {
                    self.params.push((other.to_string(), value));
                }
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
            "cuda-frames" => Some(PropValue::Bool(self.cuda_frames)),
            "input-caps" => Some(PropValue::Str(self.accept.to_gst_string())),
            "output-caps" => self
                .produce
                .as_ref()
                .map(|c| PropValue::Str(c.to_gst_string())),
            other => self
                .params
                .iter()
                .find(|(k, _)| k == other)
                .map(|(_, v)| v.clone()),
        }
    }
}

/// `PyAggregator`'s settable properties (the runtime / `gst-launch` face). The
/// input count comes from link degree (the muxer factory), not a property.
static PYAGGREGATOR_PROPS: &[PropertySpec] = hosted_element_props![
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
    PropertySpec::new(
        "cuda-frames",
        PropKind::Bool,
        "batch GPU-resident CUDA frames (needs g2g_process_cuda_batch, NV12 / P010)",
    )
    .with_default("false"),
    PropertySpec::new(
        "input-caps",
        PropKind::Str,
        "caps accepted on every input pad, e.g. audio/x-raw,format=S16LE,rate=16000",
    )
    .with_default("video/x-raw,format=RGBA"),
    PropertySpec::new(
        "output-caps",
        PropKind::Str,
        "caps produced downstream when the hosted element changes media type, e.g. text/x-raw,format=utf8",
    ),
];

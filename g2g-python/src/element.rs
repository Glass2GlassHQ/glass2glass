//! [`PyTransform`]: a gst-python-ml element shell hosted as a g2g transform.
//!
//! This is the Rust mirror of gst-python-ml's `backend/gst` `BaseTransform`:
//! it negotiates caps, then on each frame hands the buffer to a hosted Python
//! instance and pushes the result downstream. The negotiation half is pure
//! Rust and always compiles; the per-frame Python call lives in [`crate::host`]
//! behind the `python` feature.
//!
//! Caps model: an overlay/inference-in-place element (the `ActionTask` shape)
//! takes a raw-video frame and returns one in the same format, so it is a
//! non-boundary transform whose output caps equal its input. `output-caps=`
//! makes it a boundary instead ([`AsyncElement::is_format_boundary`] plus a
//! `DerivedOutput` constraint naming the declared caps, like `g2g-ml`'s
//! `OrtInference`): the single-chain gst-python-ml families read one media type
//! and write another (audio in and a transcript out, text in and speech out),
//! and their output is not the size of their input, so the hosted element
//! returns it through `meta.emit` (see [`crate::host`]).
//!
//! Memory-domain model (M985): the frame is read where it lies and forwarded
//! untouched, so both pads carry the one domain the hosted code reads, System or
//! (under `cuda-frames`) CUDA. See [`AsyncElement::input_domains`] below.

use core::future::Future;
use core::pin::Pin;

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    ElementMetadata, Frame, G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

use crate::format::{format_from_py, format_to_py, frame_bytes};
use crate::props::{fixed_caps, hosted_element_props};

/// A gst-python-ml element hosted as a first-class g2g transform.
#[derive(Debug)]
pub struct PyTransform {
    /// Python module to import, e.g. `"action"` (a gst-python-ml element shell
    /// running under `PYML_BACKEND=g2g`).
    module: String,
    /// Class within the module to instantiate, e.g. `"ActionTransform"`.
    class: String,
    /// Caps this element accepts on its sink pad. Default: RGBA at any
    /// geometry / rate. A real element derives this from the Python class's
    /// declared sink-pad template; `with_accept` overrides it meanwhile.
    accept: Caps,
    /// Caps produced downstream when the hosted element emits a different media
    /// type than it reads: audio in / text out for transcription, and the rest
    /// of the 1-in-1-out gst-python-ml families. Unset means it emits what it
    /// read (the overlay / detector shape).
    produce: Option<Caps>,
    /// Overlay flag bridged to the Python task (an example backend-declared
    /// property; `ActionTask` reads `self.draw_label`).
    draw_label: bool,
    /// Whether the hosted element works on GPU-resident CUDA frames, which it
    /// reads through `g2g_process_cuda` and passes through untouched. It decides
    /// both halves of this element's memory-domain declaration: the frame arrives
    /// and leaves in the one domain, so the two cannot disagree.
    cuda_frames: bool,
    /// Element properties forwarded verbatim to the hosted Python instance at
    /// construction (e.g. `model-name`, `engine-name`, `device`): the gst-python
    /// GObject-property analog. The Python class declares these (via the g2g
    /// backend's `GObject` shim); the host `setattr`s them on the instance with
    /// `-` mapped to `_`. Kept in insertion order for deterministic application.
    params: Vec<(String, PropValue)>,
    configured: bool,
    /// The negotiated, fully fixed input caps captured at configure time, so
    /// `process` knows the concrete geometry / format to hand Python.
    fixed: Option<Caps>,
    emitted: u64,
    /// The hosted Python element on its own GIL-owning worker thread, spawned
    /// at configure time. Present only in the `python` build.
    #[cfg(feature = "python")]
    worker: Option<crate::host::PyWorker>,
}

impl PyTransform {
    /// Host the `class` from Python `module`. The instance is created at
    /// `configure_pipeline` time (under the GIL), not here, so construction
    /// stays cheap and infallible like the other elements' `new`.
    pub fn new(module: impl Into<String>, class: impl Into<String>) -> Self {
        Self {
            module: module.into(),
            class: class.into(),
            accept: Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
            },
            produce: None,
            draw_label: false,
            cuda_frames: false,
            params: Vec::new(),
            configured: false,
            fixed: None,
            emitted: 0,
            #[cfg(feature = "python")]
            worker: None,
        }
    }

    /// Override the accepted sink caps (e.g. to host an NV12 element). The
    /// supported set may carry `Any` dims/rate: negotiation fixes them against
    /// concrete upstream caps, so `process` always sees a fixed format.
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

    /// The caps this element puts on its source pad for `input`: the declared
    /// output when it changes media type, else the input it passes through.
    fn output_for(&self, input: &Caps) -> Caps {
        self.produce.clone().unwrap_or_else(|| input.clone())
    }

    /// Set the `draw-label` overlay flag forwarded to the Python task.
    pub fn with_draw_label(mut self, on: bool) -> Self {
        self.draw_label = on;
        self
    }

    /// Host an element that works on GPU-resident CUDA frames: they reach it as
    /// `__cuda_array_interface__` planes through `g2g_process_cuda` and flow on
    /// still device-resident. Sets this element's whole memory-domain story (see
    /// [`AsyncElement::input_domains`]); the hosted class must define
    /// `g2g_process_cuda` and the caps must be semi-planar (NV12 / P010).
    pub fn with_cuda_frames(mut self, on: bool) -> Self {
        self.cuda_frames = on;
        self
    }

    /// The one memory domain frames arrive and leave in.
    fn domain(&self) -> MemoryDomainKind {
        if self.cuda_frames {
            MemoryDomainKind::Cuda
        } else {
            MemoryDomainKind::System
        }
    }

    /// Count of frames pushed downstream. Useful in tests.
    pub fn emitted_count(&self) -> u64 {
        self.emitted
    }

    #[cfg(feature = "python")]
    async fn run(&self, frame: Frame) -> Result<Vec<Frame>, G2gError> {
        let worker = self.worker.as_ref().ok_or(G2gError::NotConfigured)?;
        let caps = self.fixed.as_ref().ok_or(G2gError::NotConfigured)?;
        worker.run(frame, caps).await
    }

    #[cfg(not(feature = "python"))]
    async fn run(&self, _frame: Frame) -> Result<Vec<Frame>, G2gError> {
        // The per-frame Python call embeds CPython via pyo3 and lives behind
        // the `python` feature. The default build negotiates caps but cannot
        // run frames; build with `--features python`.
        Err(G2gError::UnsupportedDomain)
    }
}

impl AsyncElement for PyTransform {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// The hosted element reads and writes the frame in place, so the output
    /// caps equal the input (when it is in the accepted set) unless
    /// `output-caps=` declared a different media type. Declaring this native
    /// constraint (rather than the default legacy intercept-only path, whose
    /// output the solver leaves unconstrained) lets the graph solver derive this
    /// element's output edge and lets the runtime forward-caps resolve steer a
    /// mid-stream `CapsChanged` (e.g. an upstream decoder's first-frame caps)
    /// cleanly through it, instead of stalling on an unconstrained boundary.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // Stating both sides lets the solver narrow the input link too. Derivation
        // only pushes forward, so wavparse's any-rate any-channels fixates alone.
        if let Some(produce) = self.produce.clone() {
            return CapsConstraint::Mapping(vec![(
                CapsSet::one(self.accept.clone()),
                CapsSet::one(produce),
            )]);
        }
        let accept = self.accept.clone();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            match input.intersect(&accept) {
                Ok(_) => CapsSet::one(input.clone()),
                Err(_) => CapsSet::from_alternatives(Vec::new()),
            }
        }))
    }

    /// A hosted element that declares `output-caps=` turns one media type into
    /// another (audio into a transcript), which is what a boundary is.
    fn is_format_boundary(&self) -> bool {
        self.produce.is_some()
    }

    /// The legacy-bridge half of the constraint above, for the runner paths that
    /// derive a boundary element's output side through this hook.
    fn propose_output_caps(&self, input: &Caps) -> Caps {
        self.output_for(input)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.accept)
    }

    /// The hosted element reads the frame where it already is and forwards it
    /// untouched, so the domain it emits is the domain it consumes: System bytes
    /// over the buffer protocol, or CUDA device memory over
    /// `__cuda_array_interface__` under `cuda-frames`. Declaring one domain on
    /// both pads keeps that relation honest, so the domain-converter auto-plug
    /// splices a download / upload *ahead* of this element when upstream cannot
    /// deliver what the hosted code reads, and splices nothing after it (the
    /// frame really does leave in the declared domain).
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(self.domain())
    }

    fn output_memory(&self) -> MemoryDomainKind {
        self.domain()
    }

    /// Ask upstream to allocate in the domain the hosted code can read, so a
    /// multi-domain producer (an NVDEC that can keep frames on the device or
    /// download them) settles on it rather than needing a converter node. Only
    /// the domain is constrained: this element allocates nothing of its own, so
    /// it imposes no buffer count or alignment, and the size is one frame.
    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        } = caps
        else {
            return None;
        };
        let size = frame_bytes(*format, *width, *height);
        Some(if self.cuda_frames {
            AllocationParams::cuda(size, 1, 1)
        } else {
            AllocationParams::system(size, 1)
        })
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        absolute_caps.intersect(&self.accept)?;
        self.fixed = Some(absolute_caps.clone());
        #[cfg(feature = "python")]
        {
            // A registry-built `pyelement` starts with empty module/class until
            // `module=`/`class=` properties are applied; fail clearly here
            // rather than importing the empty module.
            if self.module.is_empty() || self.class.is_empty() {
                return Err(G2gError::NotConfigured);
            }
            // Spawn the worker once: a re-configure (re-negotiation) must not tear
            // down and re-init the hosted instance, which would discard a loaded
            // model. Matches `PySource` / `PyAggregator`.
            if self.worker.is_none() {
                self.worker = Some(crate::host::PyWorker::spawn(
                    &self.module,
                    &self.class,
                    self.draw_label,
                    &self.params,
                )?);
            }
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    for output in self.run(frame).await? {
                        self.emitted += 1;
                        out.push(PipelinePacket::DataFrame(output)).await?;
                    }
                }
                // A mid-stream change to anything outside the accepted set is a
                // hard error; otherwise announce this element's own output side
                // so downstream stays in step.
                PipelinePacket::CapsChanged(c) => {
                    c.intersect(&self.accept)?;
                    let announce = self.output_for(&c);
                    out.push(PipelinePacket::CapsChanged(announce)).await?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // Stateless per-frame host: nothing buffered to drain.
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
            "Python ML element host",
            "Filter/Effect/Video",
            "Hosts a gst-python-ml element shell as a g2g transform via embedded CPython.",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PYTRANSFORM_PROPS
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
            "format" => {
                let parsed = format_from_py(value.as_str().ok_or(PropError::Type)?)
                    .ok_or(PropError::Value)?;
                let Caps::RawVideo { format, .. } = &mut self.accept else {
                    return Err(PropError::Value);
                };
                *format = parsed;
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
            // Any other property goes to the hosted Python instance. Which names
            // are real is the class's to say, so a typo is caught when it loads.
            other => {
                crate::props::forward(&mut self.params, other, value);
                Ok(())
            }
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "module" => Some(PropValue::Str(self.module.clone())),
            "class" => Some(PropValue::Str(self.class.clone())),
            "draw-label" => Some(PropValue::Bool(self.draw_label)),
            "cuda-frames" => Some(PropValue::Bool(self.cuda_frames)),
            "format" => match &self.accept {
                Caps::RawVideo { format, .. } => {
                    Some(PropValue::Str(format_to_py(*format).to_string()))
                }
                _ => None,
            },
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

impl PadTemplates for PyTransform {
    /// Advertise the default accepted format (RGBA, any geometry) on both pads
    /// for `gst-inspect` / autoplug. A `pyelement` is a same-format transform,
    /// so sink and source carry the same set. (`with_accept` can host another
    /// format programmatically; the launch template reflects the default.)
    fn pad_templates() -> Vec<PadTemplate> {
        let rgba = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        };
        let set = CapsSet::one(rgba);
        Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
    }
}

/// `PyTransform`'s settable properties (the runtime / `gst-launch` face).
static PYTRANSFORM_PROPS: &[PropertySpec] = hosted_element_props![
    PropertySpec::new(
        "module",
        PropKind::Str,
        "Python module to import (the element shell)",
    ),
    PropertySpec::new(
        "class",
        PropKind::Str,
        "class within the module to instantiate",
    ),
    PropertySpec::new(
        "draw-label",
        PropKind::Bool,
        "overlay the inferred label on the frame",
    )
    .with_default("false"),
    PropertySpec::new(
        "format",
        PropKind::Str,
        "pixel format the hosted element accepts (RGBA | BGRA | NV12 | I420 | YUY2 | P010_10LE)",
    )
    .with_default("RGBA"),
    PropertySpec::new(
        "cuda-frames",
        PropKind::Bool,
        "host an element that reads GPU-resident CUDA frames (needs g2g_process_cuda, NV12 / P010)",
    )
    .with_default("false"),
    PropertySpec::new(
        "input-caps",
        PropKind::Str,
        "caps accepted on the sink pad, e.g. audio/x-raw,format=S16LE,rate=16000",
    )
    .with_default("video/x-raw,format=RGBA"),
    PropertySpec::new(
        "output-caps",
        PropKind::Str,
        "caps produced downstream when the hosted element changes media type, e.g. text/x-raw,format=utf8",
    ),
];

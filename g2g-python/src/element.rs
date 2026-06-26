//! [`PyTransform`]: a gst-python-ml element shell hosted as a g2g transform.
//!
//! This is the Rust mirror of gst-python-ml's `backend/gst` `BaseTransform`:
//! it negotiates caps, then on each frame hands the buffer to a hosted Python
//! instance and pushes the result downstream. The negotiation half is pure
//! Rust and always compiles; the per-frame Python call lives in [`crate::host`]
//! behind the `python` feature.
//!
//! Caps model: an overlay/inference-in-place element (the `ActionTask` shape)
//! takes a raw-video frame and returns one in the same format, so this is a
//! non-boundary transform whose output caps equal its input. A future
//! format-changing variant (e.g. raw-video in, `Caps::Tensor` out) would set
//! [`AsyncElement::is_format_boundary`] and a `DerivedOutput` constraint, like
//! `g2g-ml`'s `OrtInference`.

use core::future::Future;
use core::pin::Pin;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, Frame,
    G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, Rate, RawVideoFormat,
};

/// A gst-python-ml element hosted as a first-class g2g transform.
#[derive(Debug)]
pub struct PyTransform {
    /// Python module to import, e.g. `"action"` (a gst-python-ml element shell
    /// running under `GSTML_BACKEND=g2g`).
    module: String,
    /// Class within the module to instantiate, e.g. `"ActionTransform"`.
    class: String,
    /// Caps this element accepts on its sink pad. Default: RGBA at any
    /// geometry / rate. A real element derives this from the Python class's
    /// declared sink-pad template; `with_accept` overrides it meanwhile.
    accept: Caps,
    /// Overlay flag bridged to the Python task (an example backend-declared
    /// property; `ActionTask` reads `self.draw_label`).
    draw_label: bool,
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
            },
            draw_label: false,
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

    /// Set the `draw-label` overlay flag forwarded to the Python task.
    pub fn with_draw_label(mut self, on: bool) -> Self {
        self.draw_label = on;
        self
    }

    /// Count of frames pushed downstream. Useful in tests.
    pub fn emitted_count(&self) -> u64 {
        self.emitted
    }

    #[cfg(feature = "python")]
    async fn run(&self, frame: Frame) -> Result<Frame, G2gError> {
        let worker = self.worker.as_ref().ok_or(G2gError::NotConfigured)?;
        let caps = self.fixed.as_ref().ok_or(G2gError::NotConfigured)?;
        worker.run(frame, caps).await
    }

    #[cfg(not(feature = "python"))]
    async fn run(&self, _frame: Frame) -> Result<Frame, G2gError> {
        // The per-frame Python call embeds CPython via pyo3 and lives behind
        // the `python` feature. The default build negotiates caps but cannot
        // run frames; build with `--features python`.
        Err(G2gError::UnsupportedDomain)
    }
}

impl AsyncElement for PyTransform {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Passthrough identity: the hosted element reads and writes the frame in
    /// place, so the output caps equal the input (when it is in the accepted
    /// set). Declaring this native constraint (rather than the default legacy
    /// intercept-only path, whose output the solver leaves unconstrained) lets
    /// the graph solver derive this element's output edge and lets the runtime
    /// forward-caps resolve steer a mid-stream `CapsChanged` (e.g. an upstream
    /// decoder's first-frame caps) cleanly through it, instead of stalling on an
    /// unconstrained boundary.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let accept = self.accept.clone();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            match input.intersect(&accept) {
                Ok(_) => CapsSet::one(input.clone()),
                Err(_) => CapsSet::from_alternatives(Vec::new()),
            }
        }))
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.accept)
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
            self.worker = Some(crate::host::PyWorker::spawn(
                &self.module,
                &self.class,
                self.draw_label,
                &self.params,
            )?);
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
                    let output = self.run(frame).await?;
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(output)).await?;
                }
                // Non-boundary, same-format transform: a mid-stream change to
                // anything outside the accepted set is a hard error; otherwise
                // forward it so downstream stays in step.
                PipelinePacket::CapsChanged(c) => {
                    c.intersect(&self.accept)?;
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // Stateless per-frame host: nothing buffered to drain.
                PipelinePacket::Eos => {}
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
            // Any other *declared* property is forwarded to the hosted Python
            // instance (a gst-python-ml element's own GObject property, e.g.
            // model-name / engine-name). Stored in order; re-setting replaces.
            // An undeclared name is a typo, rejected like any other element.
            other if PYTRANSFORM_PROPS.iter().any(|s| s.name == other) => {
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
            other => self.params.iter().find(|(k, _)| k == other).map(|(_, v)| v.clone()),
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
        };
        let set = CapsSet::one(rgba);
        Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
    }
}

/// `PyTransform`'s settable properties (the runtime / `gst-launch` face).
static PYTRANSFORM_PROPS: &[PropertySpec] = &[
    PropertySpec::new("module", PropKind::Str, "Python module to import (the element shell)"),
    PropertySpec::new("class", PropKind::Str, "class within the module to instantiate"),
    PropertySpec::new("draw-label", PropKind::Bool, "overlay the inferred label on the frame")
        .with_default("false"),
    // Common ML tunables declared by the gst-python-ml backend BaseTransform.
    // These are forwarded to the hosted Python instance (a property absent from
    // the Python class is simply set as an attribute it ignores). Declaring them
    // here lets `gst-launch` type and accept `model-name=...` etc.
    PropertySpec::new("model-name", PropKind::Str, "pre-trained model name or local path"),
    PropertySpec::new(
        "engine-name",
        PropKind::Str,
        "ML engine: pytorch, onnx, tensorflow, tflite, openvino, ...",
    ),
    PropertySpec::new("device", PropKind::Str, "inference device: cpu, cuda, cuda:0, ..."),
    PropertySpec::new("batch-size", PropKind::Int, "number of items to process in a batch"),
    PropertySpec::new("frame-stride", PropKind::Int, "how often to process a frame"),
    PropertySpec::new("input-format", PropKind::Str, "input tensor layout: auto, nhwc, or nchw"),
    PropertySpec::new("post-process", PropKind::Str, "post-processing format for raw output"),
    PropertySpec::new("device-queue-id", PropKind::Int, "DeviceQueue id from the pool to use"),
    PropertySpec::new("compile", PropKind::Bool, "enable torch.compile for the model")
        .with_default("false"),
    PropertySpec::new("track", PropKind::Bool, "enable object tracking (detectors)")
        .with_default("false"),
];

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
    AsyncElement, Caps, ConfigureOutcome, Dim, ElementMetadata, Frame, G2gError, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
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
    configured: bool,
    /// The negotiated, fully fixed input caps captured at configure time, so
    /// `process` knows the concrete geometry / format to hand Python.
    fixed: Option<Caps>,
    emitted: u64,
    /// The live Python element instance, created at configure time under the
    /// GIL. Present only in the `python` build.
    #[cfg(feature = "python")]
    instance: Option<pyo3::Py<pyo3::PyAny>>,
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
            configured: false,
            fixed: None,
            emitted: 0,
            #[cfg(feature = "python")]
            instance: None,
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
    fn run(&mut self, frame: Frame) -> Result<Frame, G2gError> {
        let instance = self.instance.as_ref().ok_or(G2gError::NotConfigured)?;
        let caps = self.fixed.as_ref().ok_or(G2gError::NotConfigured)?;
        crate::host::run_transform(instance, frame, caps)
    }

    #[cfg(not(feature = "python"))]
    fn run(&mut self, _frame: Frame) -> Result<Frame, G2gError> {
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

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.accept)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        absolute_caps.intersect(&self.accept)?;
        self.fixed = Some(absolute_caps.clone());
        #[cfg(feature = "python")]
        {
            self.instance =
                Some(crate::host::instantiate(&self.module, &self.class, self.draw_label)?);
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
                    let output = self.run(frame)?;
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

/// `PyTransform`'s settable properties (the runtime / `gst-launch` face).
static PYTRANSFORM_PROPS: &[PropertySpec] = &[
    PropertySpec::new("module", PropKind::Str, "Python module to import (the element shell)"),
    PropertySpec::new("class", PropKind::Str, "class within the module to instantiate"),
    PropertySpec::new("draw-label", PropKind::Bool, "overlay the inferred label on the frame")
        .with_default("false"),
];

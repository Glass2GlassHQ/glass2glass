use core::future::Future;
use core::pin::Pin;

use g2g_core::memory::DomainSet;
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropValue, PropertySpec,
};

use crate::hosted_input::{
    default_accept, HostedInput, CLASS_PROPERTY, CUDA_FRAMES_PROPERTY, INPUT_CAPS_PROPERTY,
    MODULE_PROPERTY,
};
use crate::props::{hosted_element_props, HostedClassProperties};

/// A gst-python-ml element hosted as a g2g sink: it reads every frame through
/// the same hooks as [`PyTransform`](crate::PyTransform) and sends nothing on.
#[derive(Debug)]
pub struct PySink {
    input: HostedInput,
}

impl PySink {
    /// Host the `class` from Python `module`. The instance is created at
    /// `configure_pipeline` time, as for [`PyTransform`](crate::PyTransform).
    pub fn new(module: impl Into<String>, class: impl Into<String>) -> Self {
        Self {
            input: HostedInput::new(module.into(), class.into()),
        }
    }
}

impl AsyncElement for PySink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(self.input.accept.clone()))
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.input.intercept_caps(upstream_caps)
    }

    fn input_domains(&self) -> DomainSet {
        self.input.input_domains()
    }

    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        self.input.propose_allocation(caps)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.input.configure(absolute_caps, false)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.input.is_configured() {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => self.input.consume(frame).await,
                PipelinePacket::CapsChanged(caps) => self.input.caps_changed(caps),
                PipelinePacket::Eos => self.input.finish().await,
                _ => Ok(()),
            }
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Python sink",
            "Sink",
            "Hosts a gst-python-ml element shell as a g2g sink via embedded CPython.",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PYSINK_CLASS_PROPS.for_class(&self.input.module, &self.input.class)
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.input.set_property(name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.input.get_property(name)
    }
}

impl PadTemplates for PySink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(default_accept()))])
    }
}

static PYSINK_CLASS_PROPS: HostedClassProperties = HostedClassProperties::new(PYSINK_PROPS);

static PYSINK_PROPS: &[PropertySpec] = hosted_element_props![
    MODULE_PROPERTY,
    CLASS_PROPERTY,
    CUDA_FRAMES_PROPERTY,
    INPUT_CAPS_PROPERTY,
];

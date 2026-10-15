use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use g2g_core::memory::DomainSet;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    MultiOutputElement, MultiOutputSink, OutputSink, PipelinePacket, PropError, PropValue,
    PropertySpec, PushOutcome, TextFormat,
};

use crate::element::PyTransform;

pub(crate) const SRC_PAD: &str = "src";
pub(crate) const TEXT_SRC_PAD: &str = "text_src";

const DEFAULT_SRC_PORT: usize = 0;
const DEFAULT_TEXT_SRC_PORT: usize = 1;

fn text_caps() -> Caps {
    Caps::Text {
        format: TextFormat::Utf8,
    }
}

// a `pyelement` whose launch line links both its src and its text_src pad
#[derive(Debug)]
pub(crate) struct PyTransformWithText {
    transform: PyTransform,
    src_port: usize,
    text_src_port: usize,
    text_caps_sent: bool,
}

impl PyTransformWithText {
    pub(crate) fn new() -> Self {
        Self {
            transform: PyTransform::new("", "").with_text_output(),
            src_port: DEFAULT_SRC_PORT,
            text_src_port: DEFAULT_TEXT_SRC_PORT,
            text_caps_sent: false,
        }
    }

    async fn push_text(
        &mut self,
        packet: PipelinePacket,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        if !self.text_caps_sent {
            self.text_caps_sent = true;
            out.push_to(self.text_src_port, PipelinePacket::CapsChanged(text_caps()))
                .await?;
        }
        out.push_to(self.text_src_port, packet).await?;
        Ok(())
    }
}

struct PortSink<'a> {
    out: &'a mut dyn MultiOutputSink,
    port: usize,
}

impl OutputSink for PortSink<'_> {
    fn poll_push(
        &mut self,
        cx: &mut Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        self.out.poll_push_to(cx, self.port, packet)
    }

    fn begin_push(&mut self) {
        self.out.begin_push_to(self.port);
    }
}

impl MultiOutputElement for PyTransformWithText {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        AsyncElement::intercept_caps(&self.transform, upstream_caps)
    }

    // a batch input is checked per stream, which a caps set cannot say
    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        AsyncElement::configure_pipeline(&mut self.transform, absolute_caps)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let text_src_copy = match &packet {
                PipelinePacket::Segment(segment) => Some(PipelinePacket::Segment(*segment)),
                PipelinePacket::Flush => Some(PipelinePacket::Flush),
                _ => None,
            };
            let mut src = PortSink {
                out: &mut *out,
                port: self.src_port,
            };
            let text = self.transform.process_packet(packet, &mut src).await?;
            if let Some(packet) = text_src_copy {
                self.push_text(packet, out).await?;
            }
            for frame in text {
                self.push_text(PipelinePacket::DataFrame(frame), out)
                    .await?;
            }
            Ok(())
        })
    }

    fn port_caps_constraint(&self, port: usize) -> Option<CapsConstraint<'_>> {
        if port == self.src_port {
            return Some(self.transform.caps_constraint_as_transform());
        }
        (port == self.text_src_port).then(|| CapsConstraint::Produces(CapsSet::one(text_caps())))
    }

    fn name_output_ports(&mut self, pad_names: &[&str]) -> Result<(), usize> {
        let mut src_port = None;
        let mut text_src_port = None;
        for (port, name) in pad_names.iter().enumerate() {
            let taken = match *name {
                "" | SRC_PAD => &mut src_port,
                TEXT_SRC_PAD => &mut text_src_port,
                _ => return Err(port),
            };
            if taken.replace(port).is_some() {
                return Err(port);
            }
        }
        if let (Some(src_port), Some(text_src_port)) = (src_port, text_src_port) {
            self.src_port = src_port;
            self.text_src_port = text_src_port;
        }
        Ok(())
    }

    fn input_domains(&self) -> DomainSet {
        AsyncElement::input_domains(&self.transform)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AsyncElement::properties(&self.transform)
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        AsyncElement::set_property(&mut self.transform, name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        AsyncElement::get_property(&self.transform, name)
    }

    fn metadata(&self) -> ElementMetadata {
        AsyncElement::metadata(&self.transform)
    }
}

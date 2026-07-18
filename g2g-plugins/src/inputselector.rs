//! Input selector (`input-selector`). Forwards frames from exactly one of its N
//! inputs, chosen by the `active-pad` property; frames on the other inputs are
//! dropped. The g2g analog of GStreamer's `input-selector`, switchable live.
//! `no_std`.
//!
//! The inputs are assumed the same format (the switch is transparent), so the
//! output caps follow input 0.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::{
    Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, MultiInputElement,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

#[derive(Debug)]
pub struct InputSelector {
    inputs: usize,
    active: usize,
    configured: Vec<Option<Caps>>,
}

impl InputSelector {
    pub fn new(inputs: usize) -> Self {
        assert!(inputs > 0, "InputSelector needs at least one input");
        Self { inputs, active: 0, configured: vec![None; inputs] }
    }

    pub fn with_active(mut self, active: usize) -> Self {
        self.active = active.min(self.inputs - 1);
        self
    }
}

impl MultiInputElement for InputSelector {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn output_follows_input(&self) -> Option<usize> {
        Some(0)
    }

    fn configure_pipeline(&mut self, input: usize, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured[input] = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.configured[0].clone().ok_or(G2gError::NotConfigured)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new("Input selector", "Generic", "Forwards one of N inputs", "g2g")
    }

    fn properties(&self) -> &'static [PropertySpec] {
        INPUTSELECTOR_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "active-pad" => {
                let idx = value.as_uint().ok_or(PropError::Type)? as usize;
                if idx >= self.inputs {
                    return Err(PropError::Value);
                }
                self.active = idx;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "active-pad" => Some(PropValue::Uint(self.active as u64)),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            // Only the active input reaches the output; drop the rest. Eos is
            // aggregated by the runner, so it is never forwarded here.
            if input == self.active {
                if let PipelinePacket::Eos = packet {
                    return Ok(());
                }
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

static INPUTSELECTOR_PROPS: &[PropertySpec] =
    &[PropertySpec::new("active-pad", PropKind::Uint, "index of the input to forward")];

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Frame, FrameTiming, MemoryDomain, PushOutcome, SystemSlice};

    #[derive(Default)]
    struct CollectSink {
        seq: Vec<u64>,
    }
    impl OutputSink for CollectSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            if let PipelinePacket::DataFrame(f) = &packet {
                self.seq.push(f.sequence);
            }
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }
    }

    fn frame(seq: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 4].into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: seq,
            meta: Default::default(),
        })
    }

    #[tokio::test]
    async fn forwards_only_active_input() {
        let mut s = InputSelector::new(2);
        let mut out = CollectSink::default();
        s.process(0, frame(10), &mut out).await.unwrap();
        s.process(1, frame(20), &mut out).await.unwrap(); // dropped (input 1 inactive)
        s.set_property("active-pad", PropValue::Uint(1)).unwrap();
        s.process(0, frame(30), &mut out).await.unwrap(); // dropped now
        s.process(1, frame(40), &mut out).await.unwrap();
        assert_eq!(out.seq, vec![10, 40]);
    }

    #[test]
    fn active_pad_out_of_range_rejected() {
        let mut s = InputSelector::new(2);
        assert_eq!(s.set_property("active-pad", PropValue::Uint(5)).unwrap_err(), PropError::Value);
    }
}

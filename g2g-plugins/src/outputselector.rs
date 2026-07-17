//! Output selector (`output-selector`). Routes the single input to exactly one of
//! its N outputs, chosen by the `active-pad` property; the other outputs receive
//! no frames. The g2g analog of GStreamer's `output-selector`, switchable live.
//! `no_std`.
//!
//! Every output carries the input caps (a broadcast-tee fan-out, not a content
//! demux), so timing / caps changes are broadcast to all outputs while only the
//! active output gets data. That keeps an inactive branch negotiated, so a live
//! switch to it works without renegotiation.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    Caps, CapsConstraint, ConfigureOutcome, G2gError, MultiOutputElement, MultiOutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

#[derive(Debug)]
pub struct OutputSelector {
    ports: usize,
    active: usize,
}

impl OutputSelector {
    pub fn new(ports: usize) -> Self {
        assert!(ports > 0, "OutputSelector needs at least one output");
        Self { ports, active: 0 }
    }

    pub fn with_active(mut self, active: usize) -> Self {
        self.active = active.min(self.ports - 1);
        self
    }
}

impl MultiOutputElement for OutputSelector {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        OUTPUTSELECTOR_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "active-pad" => {
                let idx = value.as_uint().ok_or(PropError::Type)? as usize;
                if idx >= self.ports {
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
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                // Data goes only to the active output.
                PipelinePacket::DataFrame(_) => {
                    out.push_to(self.active, packet).await?;
                }
                // Caps / timing / flush keep every branch in sync so a switch
                // works. PipelinePacket is not Clone, so rebuild per port.
                PipelinePacket::CapsChanged(caps) => {
                    for port in 0..self.ports {
                        out.push_to(port, PipelinePacket::CapsChanged(caps.clone())).await?;
                    }
                }
                PipelinePacket::Segment(seg) => {
                    for port in 0..self.ports {
                        out.push_to(port, PipelinePacket::Segment(seg)).await?;
                    }
                }
                PipelinePacket::Flush => {
                    for port in 0..self.ports {
                        out.push_to(port, PipelinePacket::Flush).await?;
                    }
                }
                // The runner broadcasts the merged Eos to every port after this
                // returns; the element must not forward it.
                PipelinePacket::Eos => {}
                _ => {}
            }
            Ok(())
        })
    }
}

static OUTPUTSELECTOR_PROPS: &[PropertySpec] =
    &[PropertySpec::new("active-pad", PropKind::Uint, "index of the output to route to")];

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use g2g_core::{Frame, FrameTiming, MemoryDomain, PushOutcome, SystemSlice};

    #[derive(Default)]
    struct CollectSink {
        // per-port received sequences.
        got: Vec<Vec<u64>>,
    }
    impl CollectSink {
        fn new(ports: usize) -> Self {
            Self { got: (0..ports).map(|_| Vec::new()).collect() }
        }
    }
    impl MultiOutputSink for CollectSink {
        fn push_to<'a>(
            &'a mut self,
            port: usize,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            if let PipelinePacket::DataFrame(f) = &packet {
                self.got[port].push(f.sequence);
            }
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }
        fn port_count(&self) -> usize {
            self.got.len()
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
    async fn routes_to_active_output_only() {
        let mut s = OutputSelector::new(2);
        let mut out = CollectSink::new(2);
        s.process(frame(1), &mut out).await.unwrap();
        s.set_property("active-pad", PropValue::Uint(1)).unwrap();
        s.process(frame(2), &mut out).await.unwrap();
        assert_eq!(out.got[0], vec![1]);
        assert_eq!(out.got[1], vec![2]);
    }

    #[test]
    fn active_pad_out_of_range_rejected() {
        let mut s = OutputSelector::new(2);
        assert_eq!(s.set_property("active-pad", PropValue::Uint(9)).unwrap_err(), PropError::Value);
    }
}

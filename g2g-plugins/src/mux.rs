//! Interleaving muxer (M10). Combines N input streams into one output by
//! forwarding every input's frames straight through (each frame already
//! carries its own `caps`). Negotiation is per-input: each pad accepts and
//! records its own caps; the merged output caps are fixed at construction.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::{Caps, ConfigureOutcome, G2gError, MultiInputElement, OutputSink, PipelinePacket};

#[derive(Debug)]
pub struct InterleaveMux {
    inputs: usize,
    output: Caps,
    configured: Vec<Option<Caps>>,
}

impl InterleaveMux {
    pub fn new(inputs: usize, output: Caps) -> Self {
        assert!(inputs > 0, "InterleaveMux needs at least one input");
        Self { inputs, output, configured: vec![None; inputs] }
    }

    /// The caps input pad `input` was configured with, i.e. the result of
    /// that input's independent negotiation. `None` before configuration.
    pub fn input_caps(&self, input: usize) -> Option<&Caps> {
        self.configured.get(input).and_then(|c| c.as_ref())
    }
}

impl MultiInputElement for InterleaveMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Each input negotiates independently; the interleave accepts any
        // per-input caps (frames carry their own caps downstream).
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.configured[input] = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.output.clone())
    }

    fn process<'a>(
        &'a mut self,
        _input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            out.push(packet).await?;
            Ok(())
        })
    }
}

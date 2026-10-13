use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{
    BatchMeta, BatchStream, Caps, ConfigureOutcome, ElementMetadata, Frame, FrameTiming, G2gError,
    InputAggregator, MemoryDomain, MultiInputElement, OutputSink, PipelinePacket, SystemSlice,
};

/// The `streammux` element: N raw video inputs in, one [`Caps::Batch`] frame
/// out per round of one frame from every input still running, the frames
/// riding in its [`BatchMeta`]. Modelled on gst-plugins-rs's
/// `analyticscombiner`. A late input holds the batch back: there is no timeout.
#[derive(Debug)]
pub struct BatchMux {
    streams: Vec<Option<Caps>>,
    aggregator: InputAggregator<Frame>,
    batches: u64,
}

impl BatchMux {
    /// A mux batching `inputs` streams.
    pub fn new(inputs: usize) -> Self {
        Self {
            streams: vec![None; inputs],
            aggregator: InputAggregator::new(inputs),
            batches: 0,
        }
    }

    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        while let Some(round) = self.aggregator.take_round() {
            let timing = round_timing(&round);
            let streams = round
                .into_iter()
                .map(|(index, frame)| {
                    let caps = self.streams[index].clone().ok_or(G2gError::NotConfigured)?;
                    Ok(BatchStream {
                        index,
                        caps,
                        frame: Some(frame),
                    })
                })
                .collect::<Result<Vec<_>, G2gError>>()?;
            let mut batch = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(Box::new([]))),
                timing,
                self.batches,
            );
            batch.meta.attach(BatchMeta { streams });
            self.batches += 1;
            out.push(PipelinePacket::DataFrame(batch)).await?;
        }
        Ok(())
    }
}

fn round_timing(round: &[(usize, Frame)]) -> FrameTiming {
    match round.iter().find(|(index, _)| *index == 0) {
        Some((_, frame)) => frame.timing,
        None => round
            .iter()
            .map(|(_, frame)| frame.timing)
            .min_by_key(|timing| timing.pts_ns)
            .unwrap_or_default(),
    }
}

impl MultiInputElement for BatchMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.streams.len()
    }

    fn output_batches_inputs(&self) -> bool {
        true
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::RawVideo { .. } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let caps = self.intercept_caps(input, absolute_caps)?;
        *self.streams.get_mut(input).ok_or(G2gError::CapsMismatch)? = Some(caps);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        let streams = self
            .streams
            .iter()
            .cloned()
            .collect::<Option<Vec<Caps>>>()
            .ok_or(G2gError::NotConfigured)?;
        Ok(Caps::Batch { streams })
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Stream mux",
            "Mux/Batch",
            "Batches one raw video frame from every input into one frame",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    if !matches!(frame.domain, MemoryDomain::System(_)) {
                        return Err(G2gError::UnsupportedDomain);
                    }
                    // a teed batch then shares its frames instead of copying them
                    frame.domain.make_shareable();
                    self.aggregator.push(input, frame);
                    self.drain(out).await?;
                }
                PipelinePacket::Eos => {
                    self.aggregator.mark_ended(input);
                    self.drain(out).await?;
                }
                PipelinePacket::Segment(_) | PipelinePacket::Flush if input == 0 => {
                    out.push(packet).await?;
                }
                _ => {}
            }
            Ok(())
        })
    }
}

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    BatchMeta, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError,
    MultiOutputElement, MultiOutputSink, PipelinePacket,
};

/// The `streamdemux` element: one [`Caps::Batch`] frame in, each of its
/// streams' frames out on the port of the mux input it came from, the inverse
/// of [`BatchMux`](crate::batchmux::BatchMux). Modelled on gst-plugins-rs's
/// `analyticssplitter`. The launch name differs from the Rust name because
/// [`StreamDemux`](crate::streamdemux::StreamDemux) is the content-routing
/// demux.
#[derive(Debug)]
pub struct BatchDemux {
    ports: usize,
    streams: Vec<Caps>,
}

impl BatchDemux {
    /// A demux with `ports` output ports, one per stream of the batch.
    pub fn new(ports: usize) -> Self {
        Self {
            ports,
            streams: Vec::new(),
        }
    }

    fn batch_streams(&self, caps: &Caps) -> Result<Vec<Caps>, G2gError> {
        let Caps::Batch { streams } = caps else {
            return Err(G2gError::CapsMismatch);
        };
        if streams.len() != self.ports {
            return Err(G2gError::PortCountMismatch {
                ports: self.ports,
                streams: streams.len(),
            });
        }
        Ok(streams.clone())
    }
}

impl MultiOutputElement for BatchDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Batch { .. } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    // a caps set cannot say "a batch of any streams", configure_pipeline checks it
    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.streams = self.batch_streams(absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn ports_unbatch(&self) -> bool {
        true
    }

    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        self.streams.get(port).cloned()
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Stream demux",
            "Demux/Batch",
            "Splits a batch back into one stream per output port",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(mut batch) => {
                    let meta = batch
                        .meta
                        .get_mut::<BatchMeta>()
                        .ok_or(G2gError::CapsMismatch)?;
                    for stream in &mut meta.streams {
                        if stream.index >= self.ports {
                            return Err(G2gError::CapsMismatch);
                        }
                        let frame = stream.frame.take().ok_or(G2gError::BatchFrameTaken {
                            index: stream.index,
                        })?;
                        out.push_to(stream.index, PipelinePacket::DataFrame(frame))
                            .await?;
                    }
                }
                PipelinePacket::CapsChanged(caps) => {
                    let streams = self.batch_streams(&caps)?;
                    for (port, stream) in streams.iter().enumerate() {
                        if self.streams.get(port) != Some(stream) {
                            out.push_to(port, PipelinePacket::CapsChanged(stream.clone()))
                                .await?;
                        }
                    }
                    self.streams = streams;
                }
                PipelinePacket::Segment(segment) => {
                    for port in 0..self.ports {
                        out.push_to(port, PipelinePacket::Segment(segment)).await?;
                    }
                }
                PipelinePacket::Flush => {
                    for port in 0..self.ports {
                        out.push_to(port, PipelinePacket::Flush).await?;
                    }
                }
                _ => {}
            }
            Ok(())
        })
    }
}

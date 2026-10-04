use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, MemoryDomain,
    OutputSink, PipelinePacket,
};

use crate::gpu::GpuContext;
use crate::wgpubuffer::{whole_word_size, wrap_buffer};

#[derive(Debug, Default)]
pub struct WgpuUpload {
    context: Option<GpuContext>,
    configured: bool,
}

impl WgpuUpload {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_context(mut self, context: GpuContext) -> Self {
        self.context = Some(context);
        self
    }

    async fn context(&mut self) -> Result<&GpuContext, G2gError> {
        let context = match self.context.take() {
            Some(context) => context,
            None => GpuContext::headless().await?,
        };
        Ok(self.context.insert(context))
    }
}

// the buffer is padded to a whole word for wgpu, the frame keeps the exact length
fn upload(context: &GpuContext, bytes: &[u8]) -> Result<MemoryDomain, G2gError> {
    let size = whole_word_size(bytes.len() as u64).ok_or(G2gError::CapsMismatch)?;
    let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpuupload"),
        size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    buffer
        .slice(..)
        .get_mapped_range_mut()
        .slice(..bytes.len())
        .copy_from_slice(bytes);
    buffer.unmap();
    Ok(MemoryDomain::WgpuBuffer(wrap_buffer(
        &context.device,
        &context.queue,
        buffer,
        bytes.len(),
    )))
}

impl AsyncElement for WgpuUpload {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "wgpu upload",
            "Filter/Converter/Video/GPU",
            "Copies a system-memory frame into a GPU-resident wgpu buffer",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::WgpuBuffer
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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
                PipelinePacket::DataFrame(mut frame) => {
                    let category = g2g_core::log::short_type_name::<Self>();
                    let context = self.context().await?;
                    frame.domain = upload(context, frame.domain.require_system_slice(category)?)?;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // The runner forwards end-of-stream after `process` returns.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

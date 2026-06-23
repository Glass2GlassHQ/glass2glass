//! `CudaToWgpu`: the CUDA->wgpu zero-copy bridge element (M220).
//!
//! Consumes a decoded NV12 frame in CUDA device memory
//! ([`MemoryDomain::Cuda`], from `FfmpegVideoDec` in `NvdecCuda` mode) and emits
//! it as a wgpu external-memory texture ([`MemoryDomain::WgpuTexture`]) that
//! `WgpuPreprocess`'s M217 surface-import path samples directly. The pixels
//! never leave the GPU: NVDEC's planes are copied device->device into a Vulkan
//! image shared with CUDA (no PCIe download, unlike `CudaDownload`).
//!
//! The transport primitives live in `g2g_plugins::cudawgpu` (raw Vulkan + the
//! CUDA external-memory FFI); this element wires them into the pipeline and
//! produces a `WgpuNv12Texture`-owned frame on its interop device, which
//! `WgpuPreprocess` then adopts (the M217 device-identity pattern).
//!
//! Caps are `Identity(NV12)`: only the memory domain changes, Cuda -> WgpuTexture
//! (caps do not encode the domain), so the element drops into an
//! `NvdecCuda -> WgpuPreprocess` chain without changing negotiation.
//!
//! Linux + NVIDIA only (`cuda-wgpu` feature). v1 allocates a fresh shared image
//! per frame; a reuse pool is a follow-up.

use core::future::Future;
use core::pin::Pin;

use std::sync::Arc;

use g2g_core::memory::OwnedWgpuTexture;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, Frame, G2gError,
    MemoryDomain, OutputSink, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::cudawgpu::{
    create_interop_device, cuda_copy_nv12_planes, export_nv12_image, wrap_as_texture, InteropDevice,
};

use crate::wgpupreprocess::WgpuNv12Texture;

/// NV12 with open geometry: the element's identity caps set (see module docs).
fn nv12_any() -> CapsSet {
    CapsSet::one(Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    })
}

/// CUDA NV12 -> wgpu external-memory texture bridge. See the module docs.
#[derive(Debug, Default)]
pub struct CudaToWgpu {
    configured: bool,
    /// The Vulkan wgpu device with `VK_KHR_external_memory_fd`, built lazily on
    /// the first frame (device creation is async) and reused.
    interop: Option<InteropDevice>,
    /// Frames bridged CUDA -> wgpu texture.
    converted: u64,
}

impl CudaToWgpu {
    pub fn new() -> Self {
        Self::default()
    }

    /// Frames bridged so far. Useful in tests.
    pub fn converted(&self) -> u64 {
        self.converted
    }
}

impl AsyncElement for CudaToWgpu {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Only the domain changes; narrow upstream against NV12. The native
        // solver uses the Identity constraint below instead.
        for alt in nv12_any().alternatives() {
            if let Ok(narrowed) = upstream_caps.intersect(alt) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(nv12_any())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !nv12_any().accepts(absolute_caps) {
            return Err(G2gError::CapsMismatch);
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
                    let MemoryDomain::Cuda(buf) = &frame.domain else {
                        // GPU-input only: a System frame is the CPU path's job.
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // Copy the plane geometry out before borrowing the device.
                    let (ctx, luma, luma_p, chroma, chroma_p, w, h) = (
                        buf.context,
                        buf.luma_ptr,
                        buf.luma_pitch,
                        buf.chroma_ptr,
                        buf.chroma_pitch,
                        buf.width,
                        buf.height,
                    );

                    if self.interop.is_none() {
                        self.interop = Some(create_interop_device().await?);
                    }
                    let interop = self.interop.as_ref().unwrap();

                    // SAFETY: `interop.device` has VK_KHR_external_memory_fd; the
                    // plane pointers are valid NV12 device memory in `ctx` (the
                    // decoder pins them via the frame's keep-alive, and `frame`
                    // outlives this copy). `export_nv12_image` matches w/h.
                    let texture = unsafe {
                        let shared = export_nv12_image(&interop.device, w, h)?;
                        cuda_copy_nv12_planes(&shared, ctx, luma, luma_p, chroma, chroma_p, w, h)?;
                        wrap_as_texture(&interop.device, shared)
                    };

                    let owner =
                        WgpuNv12Texture::new(interop.device.clone(), interop.queue.clone(), texture);
                    let domain =
                        MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(w, h, Arc::new(owner)));
                    self.converted += 1;
                    let out_frame = Frame {
                        domain,
                        timing: frame.timing,
                        sequence: frame.sequence,
                        meta: Default::default(),
                    };
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

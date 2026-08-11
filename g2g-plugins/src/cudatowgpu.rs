//! `CudaToWgpu`: the CUDA->wgpu zero-copy bridge element (M220).
//!
//! Consumes a decoded NV12 frame in CUDA device memory
//! ([`MemoryDomain::Cuda`], from `FfmpegVideoDec` in `NvdecCuda` mode) and emits
//! it as a wgpu external-memory texture ([`MemoryDomain::WgpuTexture`]) that
//! `WgpuPreprocess`'s M217 surface-import path samples directly. The pixels
//! never leave the GPU: NVDEC's planes are copied device->device into a Vulkan
//! image shared with CUDA (no PCIe download, unlike `CudaDownload`).
//!
//! The transport primitives live in [`crate::cudawgpu`] (raw Vulkan + the CUDA
//! external-memory FFI); this element wires them into the pipeline and produces a
//! [`WgpuNv12Texture`](crate::gpu::WgpuNv12Texture)-owned frame on the shared
//! interop device, which `WgpuPreprocess` then adopts (the M217 device-identity
//! pattern) and a windowed `WgpuSink` blits.
//!
//! Caps are `Identity(NV12)`: only the memory domain changes, Cuda -> WgpuTexture
//! (caps do not encode the domain), so the element drops into an
//! `NvdecCuda -> WgpuPreprocess` chain without changing negotiation.
//!
//! Linux + NVIDIA only (`cuda-wgpu` feature). Frames reuse shared images from a
//! [`CudaWgpuPool`] (M254): the Vulkan image, its CUDA import, and the
//! `wgpu::Texture` are allocated once and recycled when the downstream frame is
//! released (via a drop guard on the emitted keep-alive), so per frame only the
//! two device->device plane copies and a sync run. A recycled entry may still be
//! sampled by an in-flight wgpu submission, so the device is drained
//! (`Device::poll`) before its image is overwritten.
//!
//! The pooled images are imported into the *decoder's* CUDA context, and the
//! driver wedges if such an import is destroyed after its context is gone, so the
//! pool is released at end of stream. A consumer that parks emitted frames past
//! the end of the pipeline hits the same wedge: present or copy each frame and
//! let it go.

use core::future::Future;
use core::pin::Pin;

use std::sync::Arc;

use alloc::boxed::Box;

use crate::cudawgpu::{shared_interop_device, CudaWgpuPool, InteropDevice};
use crate::gpu::WgpuNv12Texture;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::{DomainSet, MemoryDomainKind, OwnedWgpuTexture};
use g2g_core::{
    g2g_error, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, Frame, G2gError,
    HardwareError, MemoryDomain, OutputSink, PipelinePacket, Rate, RawVideoFormat,
};

/// How long end-of-stream waits for the consumer to release the frames it is
/// still presenting before giving up on tearing their shared images down.
const RECLAIM_TIMEOUT: core::time::Duration = core::time::Duration::from_secs(2);

/// How often that wait re-checks, yielding in between so the consumer's arm runs.
const RECLAIM_POLL: core::time::Duration = core::time::Duration::from_millis(2);

/// NV12 with open geometry: the element's identity caps set (see module docs).
fn nv12_any() -> CapsSet {
    CapsSet::one(Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
    })
}

/// CUDA NV12 -> wgpu external-memory texture bridge. See the module docs.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::cudatowgpu::CudaToWgpu;
///
/// let bridge = CudaToWgpu::new();
/// assert_eq!(bridge.converted(), 0);
/// ```
#[derive(Debug, Default)]
pub struct CudaToWgpu {
    configured: bool,
    /// The Vulkan wgpu device with `VK_KHR_external_memory_fd`, taken from the
    /// process-wide slot on the first frame (device creation is async) and reused.
    interop: Option<Arc<InteropDevice>>,
    /// Reuse pool of shared NV12 images (allocation + CUDA import amortized).
    pool: CudaWgpuPool,
    /// NV12 geometry of the pooled entries; a change rebuilds the pool.
    dims: Option<(u32, u32)>,
    /// Frames bridged CUDA -> wgpu texture.
    converted: u64,
    /// Runner-assigned instance name, so this element's error lines name it.
    log_name: LogName,
}

impl LogSource for CudaToWgpu {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

impl CudaToWgpu {
    pub fn new() -> Self {
        Self::default()
    }

    /// Frames bridged so far. Useful in tests.
    pub fn converted(&self) -> u64 {
        self.converted
    }

    /// Wait for the sink to give back the frames it is still presenting, then
    /// destroy the shared images while the decoder's CUDA context is alive.
    /// Leaks them instead if the wait runs out: destroying an image a consumer
    /// still holds wedges the driver, and a wedge is not a trade for a leak.
    async fn release_pool(&mut self) {
        let Some(interop) = &self.interop else {
            return;
        };
        let deadline = std::time::Instant::now() + RECLAIM_TIMEOUT;
        while !self.pool.all_returned() {
            if std::time::Instant::now() >= deadline {
                g2g_error!(
                    self,
                    "a consumer still holds decoded frames at end of stream; \
                     leaking their shared images rather than tearing them down"
                );
                return;
            }
            tokio::time::sleep(RECLAIM_POLL).await;
        }
        self.pool.destroy_returned(&interop.device);
    }

    /// Drop the reuse pool's free entries, forcing the next frame to allocate +
    /// import a fresh shared image. Exposed for benchmarks that A/B the pool
    /// against the per-frame-allocation path; not needed in normal use.
    pub fn reset_pool(&mut self) {
        self.pool = CudaWgpuPool::new();
    }
}

impl AsyncElement for CudaToWgpu {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
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

    fn set_instance_name(&mut self, name: alloc::string::String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: alloc::string::String) {
        self.log_name.set_category(category);
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(nv12_any())
    }

    /// The whole point of the element: CUDA device memory in, a wgpu texture out.
    /// Declaring both ends lets the M354 auto-plug pick this as the converter for
    /// a `Cuda -> WgpuTexture` edge, and stops it splicing another one after.
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::Cuda)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::WgpuTexture
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
                        self.interop = Some(shared_interop_device().await?);
                    }
                    // A geometry change invalidates pooled entries of the old
                    // size; tear them down under the same three conditions.
                    if self.dims != Some((w, h)) {
                        self.dims = Some((w, h));
                        self.release_pool().await;
                        self.pool = CudaWgpuPool::new();
                    }
                    let interop = self.interop.as_ref().unwrap();

                    // Reuse a pooled shared image, or build one. A recycled entry's
                    // image may still be sampled by an in-flight wgpu submission, so
                    // drain the device before its planes are overwritten.
                    let entry = match self.pool.take_free() {
                        Some(entry) => {
                            interop
                                .device
                                .poll(wgpu::PollType::Wait {
                                    submission_index: None,
                                    timeout: None,
                                })
                                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                            entry
                        }
                        // SAFETY: `interop.device` has VK_KHR_external_memory_fd; `ctx`
                        // is the decoder's CUDA context where the planes are valid.
                        None => unsafe { self.pool.build_entry(&interop.device, ctx, w, h)? },
                    };

                    // Copy this frame's planes into the entry's persistent CUDA array.
                    // SAFETY: the plane pointers are valid NV12 device memory in `ctx`
                    // (pinned by the frame's keep-alive, and `frame` outlives this
                    // copy); the entry was imported for (w, h) in `ctx`.
                    unsafe { entry.copy_planes(luma, luma_p, chroma, chroma_p, w, h)? };

                    // Hand a clone downstream; the entry returns to the pool when the
                    // emitted keep-alive (and its drop guard) is released.
                    let texture = entry.texture().clone();
                    let guard = self.pool.in_flight(entry);
                    let owner = WgpuNv12Texture::with_recycle(
                        interop.device.clone(),
                        interop.queue.clone(),
                        texture,
                        Box::new(guard),
                    );
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
                PipelinePacket::Eos => {
                    // Last point at which the decoder's CUDA context is certainly
                    // still alive, which is one of the three conditions destroying
                    // these imports needs. The other two are the frames being back
                    // from the sink and the GPU being idle: wait for the first
                    // (yielding, so the sink's arm can drain and return them) and
                    // let `destroy_returned` do the second.
                    self.release_pool().await;
                }
                // future PipelinePacket variants (non_exhaustive): pass-through
                // transform, forward unknown ordered control packets unchanged.
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

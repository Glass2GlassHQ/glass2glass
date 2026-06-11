use alloc::boxed::Box;

#[cfg(feature = "runtime")]
use crate::pool::PooledBuffer;

#[derive(Debug)]
pub enum MemoryDomain {
    System(SystemSlice),
    DmaBuf(OwnedDmaBuf),
    VulkanTexture(OwnedVulkanTexture),
    WebGPUBuffer(OwnedWebGPUBuffer),
    /// NVIDIA CUDA device memory. Carries raw device pointers (the decoded
    /// frame stays on the GPU), so a downstream GPU consumer can use it with
    /// no device->host copy. The backing allocation is owned elsewhere (eg an
    /// ffmpeg `CUDA`-hwframe `AVFrame`); see [`OwnedCudaBuffer`].
    Cuda(OwnedCudaBuffer),
}

/// The memory domain of a [`MemoryDomain`] without its payload. Used by the
/// allocation query (M12) so a consumer can name the kind of memory it wants
/// allocated without holding a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MemoryDomainKind {
    #[default]
    System,
    DmaBuf,
    VulkanTexture,
    WebGPUBuffer,
    Cuda,
}

impl MemoryDomain {
    /// The payload-free discriminant of this domain.
    pub fn kind(&self) -> MemoryDomainKind {
        match self {
            MemoryDomain::System(_) => MemoryDomainKind::System,
            MemoryDomain::DmaBuf(_) => MemoryDomainKind::DmaBuf,
            MemoryDomain::VulkanTexture(_) => MemoryDomainKind::VulkanTexture,
            MemoryDomain::WebGPUBuffer(_) => MemoryDomainKind::WebGPUBuffer,
            MemoryDomain::Cuda(_) => MemoryDomainKind::Cuda,
        }
    }
}

/// CPU-memory slice. The backing buffer may be a freshly-allocated `Box<[u8]>`
/// (`from_boxed`) or a pool-recycled buffer (`from_pool`). Dropping the
/// `SystemSlice` releases the underlying storage — in the pooled case, the
/// buffer returns to its pool automatically.
#[derive(Debug)]
pub struct SystemSlice {
    inner: SystemSliceInner,
}

#[derive(Debug)]
enum SystemSliceInner {
    Owned(Box<[u8]>),
    #[cfg(feature = "runtime")]
    Pooled(PooledBuffer<Box<[u8]>>),
}

impl SystemSlice {
    pub fn from_boxed(bytes: Box<[u8]>) -> Self {
        Self { inner: SystemSliceInner::Owned(bytes) }
    }

    #[cfg(feature = "runtime")]
    pub fn from_pool(buffer: PooledBuffer<Box<[u8]>>) -> Self {
        Self { inner: SystemSliceInner::Pooled(buffer) }
    }

    pub fn as_slice(&self) -> &[u8] {
        match &self.inner {
            SystemSliceInner::Owned(b) => b,
            #[cfg(feature = "runtime")]
            SystemSliceInner::Pooled(p) => p.as_ref(),
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match &mut self.inner {
            SystemSliceInner::Owned(b) => b,
            #[cfg(feature = "runtime")]
            SystemSliceInner::Pooled(p) => p.as_mut(),
        }
    }
}

#[derive(Debug)]
pub struct OwnedDmaBuf {
    fd: i32,
    pub stride: u32,
    pub offset: u32,
}

impl OwnedDmaBuf {
    /// # Safety
    /// `fd` must be a valid DMABUF descriptor with no other owner; the caller
    /// transfers ownership to this struct.
    pub unsafe fn from_raw(fd: i32, stride: u32, offset: u32) -> Self {
        Self { fd, stride, offset }
    }

    pub fn as_raw(&self) -> i32 {
        self.fd
    }
}

impl Drop for OwnedDmaBuf {
    fn drop(&mut self) {
        // DMABUF is Linux-only. On std+linux, close the fd. On other targets
        // (Wasm, RTOS without libc) we leak; a custom close hook registered
        // by the owning BufferPool is the planned no_std story.
        #[cfg(all(target_os = "linux", feature = "std"))]
        {
            extern "C" {
                fn close(fd: i32) -> i32;
            }
            // SAFETY: `from_raw` is the only constructor and is `unsafe`,
            // requiring callers to certify sole ownership of the fd.
            unsafe {
                close(self.fd);
            }
        }
    }
}

#[derive(Debug)]
pub struct OwnedVulkanTexture {
    pub handle: u64,
    pub allocation_id: u64,
}

#[derive(Debug)]
pub struct OwnedWebGPUBuffer {
    pub buffer_id: u64,
}

/// A decoded NV12 picture left in CUDA device memory. Holds the two NV12
/// plane device pointers (luma Y, interleaved chroma UV) with their row
/// pitches, the CUDA context they are valid in, and a keep-alive owner that
/// pins the backing allocation for as long as the pointers are referenced.
///
/// The device memory itself is not owned by this struct: an ffmpeg
/// `CUDA`-hwframe decoder owns it as part of an `AVFrame`. `g2g-core` cannot
/// link CUDA, so the producing element hands over its owning handle boxed as
/// a [`CudaKeepAlive`]; dropping this buffer drops that box, releasing the
/// frame back to its hwframe pool. The pointers stay valid for exactly the
/// lifetime of the keep-alive.
#[derive(Debug)]
pub struct OwnedCudaBuffer {
    /// `CUdeviceptr` (as `u64`) of the luma (Y) plane.
    pub luma_ptr: u64,
    /// `CUdeviceptr` (as `u64`) of the interleaved chroma (UV, NV12) plane.
    pub chroma_ptr: u64,
    /// Row pitch in bytes of the luma plane (>= width; the decoder aligns it).
    pub luma_pitch: u32,
    /// Row pitch in bytes of the chroma plane.
    pub chroma_pitch: u32,
    /// Visible picture dimensions in pixels.
    pub width: u32,
    pub height: u32,
    /// `CUcontext` (as `u64`) the pointers are valid in. A consumer pushes
    /// this context (`cuCtxPushCurrent`) before touching the memory.
    pub context: u64,
    /// Pins the backing allocation (eg the decoder's `AVFrame`) for the life
    /// of the pointers. Dropping it releases the allocation.
    keep_alive: Box<dyn CudaKeepAlive>,
}

impl OwnedCudaBuffer {
    /// Wrap CUDA device pointers with the owner that keeps them valid.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        luma_ptr: u64,
        chroma_ptr: u64,
        luma_pitch: u32,
        chroma_pitch: u32,
        width: u32,
        height: u32,
        context: u64,
        keep_alive: Box<dyn CudaKeepAlive>,
    ) -> Self {
        Self {
            luma_ptr,
            chroma_ptr,
            luma_pitch,
            chroma_pitch,
            width,
            height,
            context,
            keep_alive,
        }
    }

    /// The keep-alive owner, exposed so a consumer that imports the memory
    /// into another API (eg CUDA external memory) can take shared ownership.
    pub fn keep_alive(&self) -> &dyn CudaKeepAlive {
        self.keep_alive.as_ref()
    }
}

/// Owner token kept alongside an [`OwnedCudaBuffer`]'s device pointers. The
/// CUDA memory is owned by the producing element (typically an ffmpeg
/// `AVFrame` from a `CUDA` hwframe pool); `g2g-core` cannot link CUDA, so the
/// element boxes its owning handle as this trait object. Dropping the box
/// releases the backing allocation. `Send` so a frame can cross the runner's
/// worker-thread boundaries like every other domain.
pub trait CudaKeepAlive: core::fmt::Debug + Send {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicBool, Ordering};

    /// Stands in for a producer's owning handle (eg an ffmpeg `AVFrame`):
    /// flips a shared flag on drop so the test can prove the keep-alive owner
    /// is released exactly when the buffer is.
    #[derive(Debug)]
    struct FlagOnDrop(Arc<AtomicBool>);
    impl Drop for FlagOnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    impl CudaKeepAlive for FlagOnDrop {}

    #[test]
    fn cuda_domain_reports_cuda_kind() {
        let dropped = Arc::new(AtomicBool::new(false));
        let buf = OwnedCudaBuffer::new(
            0x1000,
            0x2000,
            2048,
            2048,
            1920,
            1080,
            0xC0FFEE,
            Box::new(FlagOnDrop(dropped.clone())),
        );
        let domain = MemoryDomain::Cuda(buf);
        assert_eq!(domain.kind(), MemoryDomainKind::Cuda);
    }

    #[test]
    fn dropping_cuda_buffer_releases_keep_alive() {
        let dropped = Arc::new(AtomicBool::new(false));
        let buf = OwnedCudaBuffer::new(
            0x1000,
            0x2000,
            2048,
            2048,
            1920,
            1080,
            0xC0FFEE,
            Box::new(FlagOnDrop(dropped.clone())),
        );
        assert!(!dropped.load(Ordering::SeqCst), "owner alive while buffer held");
        assert_eq!(buf.luma_ptr, 0x1000);
        assert_eq!(buf.chroma_ptr, 0x2000);
        drop(buf);
        assert!(
            dropped.load(Ordering::SeqCst),
            "dropping the buffer must release the backing allocation"
        );
    }
}

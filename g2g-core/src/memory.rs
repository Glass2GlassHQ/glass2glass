use alloc::boxed::Box;

#[cfg(feature = "runtime")]
use crate::pool::PooledBuffer;

#[derive(Debug)]
pub enum MemoryDomain {
    System(SystemSlice),
    DmaBuf(OwnedDmaBuf),
    VulkanTexture(OwnedVulkanTexture),
    WebGPUBuffer(OwnedWebGPUBuffer),
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
}

impl MemoryDomain {
    /// The payload-free discriminant of this domain.
    pub fn kind(&self) -> MemoryDomainKind {
        match self {
            MemoryDomain::System(_) => MemoryDomainKind::System,
            MemoryDomain::DmaBuf(_) => MemoryDomainKind::DmaBuf,
            MemoryDomain::VulkanTexture(_) => MemoryDomainKind::VulkanTexture,
            MemoryDomain::WebGPUBuffer(_) => MemoryDomainKind::WebGPUBuffer,
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

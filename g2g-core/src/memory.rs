use alloc::boxed::Box;

#[derive(Debug)]
pub enum MemoryDomain {
    System(SystemSlice),
    DmaBuf(OwnedDmaBuf),
    VulkanTexture(OwnedVulkanTexture),
    WebGPUBuffer(OwnedWebGPUBuffer),
}

#[derive(Debug)]
pub struct SystemSlice {
    bytes: Box<[u8]>,
}

impl SystemSlice {
    pub fn from_boxed(bytes: Box<[u8]>) -> Self {
        Self { bytes }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
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
        // M4: close the fd via libc on std targets or a registered close hook
        // on no_std targets. Leaks the fd in M0 — acceptable for skeleton.
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

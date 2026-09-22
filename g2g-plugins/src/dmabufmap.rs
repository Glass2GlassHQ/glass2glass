//! Read-only CPU mapping of a DMABUF, for the sinks that hand their payload to
//! a device that takes a host pointer (the Linux audio sinks). Importing the fd
//! instead of taking a system copy is what keeps a dma-buf producer
//! (`localdmabufsrc`, `appsrc`, a capture device) from paying a download.
//!
//! The kernel wants CPU access bracketed by `DMA_BUF_IOCTL_SYNC` so its caches
//! are coherent with whatever device wrote the buffer, so the mapping holds the
//! bracket for its lifetime: START on open, END on drop. A buffer whose exporter
//! implements no `begin_cpu_access` answers `ENOTTY`, which is not a failure to
//! map, only the absence of the hook.

use core::ffi::c_void;

use g2g_core::memory::OwnedDmaBuf;
use g2g_core::G2gError;

/// `DMA_BUF_IOCTL_SYNC`: `_IOW('b', 0, struct dma_buf_sync)`, one `__u64` of
/// flags.
const DMA_BUF_IOCTL_SYNC: libc::c_ulong = 0x4008_6200;
/// `DMA_BUF_SYNC_READ`, and the START / END halves of the access bracket.
const DMA_BUF_SYNC_READ: u64 = 1 << 0;
const DMA_BUF_SYNC_END: u64 = 1 << 2;

/// A DMABUF mapped for reading, unmapped on drop.
#[derive(Debug)]
pub struct DmaBufReadMap {
    address: *mut c_void,
    /// The whole mapping, which the kernel only hands out entire.
    length: usize,
    /// Where the frame's payload starts inside it.
    offset: usize,
    fd: i32,
}

impl DmaBufReadMap {
    /// Map `dmabuf` and take its payload from `offset` to the end of the
    /// buffer. The payload is the whole remainder: a dma-buf carries no length
    /// of its own, so a producer sizes the buffer to the frame it exports.
    ///
    /// The kernel maps a dma-buf only whole (`dma_buf_mmap` refuses a non-zero
    /// page offset), so the offset is applied to the mapping, not to `mmap`.
    pub fn read(dmabuf: &OwnedDmaBuf) -> Result<Self, G2gError> {
        let fd = dmabuf.as_raw();
        // SAFETY: `fd` is an open dma-buf descriptor the frame owns; SEEK_END on
        // one reports its size, which is how a consumer learns it.
        let size = unsafe { libc::lseek(fd, 0, libc::SEEK_END) };
        if size <= 0 {
            return Err(G2gError::UnsupportedDomain);
        }
        let length = size as usize;
        let offset = dmabuf.offset as usize;
        if offset >= length {
            return Err(G2gError::UnsupportedDomain);
        }
        // SAFETY: a null hint lets the kernel place the mapping; `fd` is an open
        // dma-buf and `length` is the size just read off it.
        let address = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                length,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(G2gError::UnsupportedDomain);
        }
        sync(fd, DMA_BUF_SYNC_READ);
        Ok(Self {
            address,
            length,
            offset,
            fd,
        })
    }

    /// The payload: the mapping from the buffer's offset on.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the mapping is live for `self`'s lifetime and covers
        // `self.length` readable bytes, of which `offset` is inside.
        unsafe {
            core::slice::from_raw_parts(
                (self.address as *const u8).add(self.offset),
                self.length - self.offset,
            )
        }
    }
}

impl Drop for DmaBufReadMap {
    fn drop(&mut self) {
        sync(self.fd, DMA_BUF_SYNC_READ | DMA_BUF_SYNC_END);
        // SAFETY: `address` / `length` are the mapping this type made and has
        // not unmapped before.
        unsafe { libc::munmap(self.address, self.length) };
    }
}

/// One half of the CPU-access bracket. Best effort: an exporter with no
/// `begin_cpu_access` hook answers `ENOTTY`, and the mapping is still readable.
fn sync(fd: i32, flags: u64) {
    // SAFETY: the ioctl reads one `__u64` through the pointer, which is a live
    // local for the duration of the call.
    unsafe {
        libc::ioctl(fd, DMA_BUF_IOCTL_SYNC, &flags as *const u64);
    }
}

/// A frame's bytes, wherever its memory lives: borrowed from a system slice, or
/// mapped in place from a dma-buf. The sinks that feed a host pointer to a
/// device read their payload through this, so accepting the dma-buf domain
/// costs them one match rather than a second data path.
#[derive(Debug)]
pub enum FrameBytes<'a> {
    System(&'a [u8]),
    Mapped(DmaBufReadMap),
}

impl FrameBytes<'_> {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::System(slice) => slice,
            Self::Mapped(map) => map.as_slice(),
        }
    }
}

/// Read a frame's payload from either domain. `element` names the caller for
/// the domain-mismatch diagnostic.
pub fn frame_bytes<'a>(
    frame: &'a g2g_core::frame::Frame,
    element: &'static str,
) -> Result<FrameBytes<'a>, G2gError> {
    match &frame.domain {
        g2g_core::memory::MemoryDomain::DmaBuf(dmabuf) => {
            Ok(FrameBytes::Mapped(DmaBufReadMap::read(dmabuf)?))
        }
        domain => Ok(FrameBytes::System(domain.require_system_slice(element)?)),
    }
}

/// Both domains these sinks take: a system buffer, or a dma-buf they map.
pub fn audio_input_domains() -> g2g_core::memory::DomainSet {
    g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
        .with(g2g_core::memory::MemoryDomainKind::DmaBuf)
}

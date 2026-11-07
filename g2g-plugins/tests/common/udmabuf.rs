use g2g_core::memory::OwnedDmaBuf;

/// `UDMABUF_CREATE`: `_IOW('u', 0x42, struct udmabuf_create)`.
const UDMABUF_CREATE: libc::c_ulong = 0x4018_7542;

#[repr(C)]
struct UdmabufCreate {
    memfd: u32,
    flags: u32,
    offset: u64,
    size: u64,
}

pub(crate) fn udmabuf_of(bytes: &[u8]) -> Option<OwnedDmaBuf> {
    let name = c"g2g-udmabuf";
    // SAFETY: opening a device path and creating a memfd are plain syscalls
    // whose arguments are valid for the call.
    let (device, memfd) = unsafe {
        let device = libc::open(c"/dev/udmabuf".as_ptr(), libc::O_RDWR);
        if device < 0 {
            return None;
        }
        let memfd = libc::memfd_create(name.as_ptr(), libc::MFD_ALLOW_SEALING);
        if memfd < 0 {
            libc::close(device);
            return None;
        }
        (device, memfd)
    };
    // SAFETY: `memfd` is the fresh descriptor above; the write covers exactly
    // the bytes just sized for, and the seal is what udmabuf requires.
    let created = unsafe {
        let sized = libc::ftruncate(memfd, bytes.len() as libc::off_t) == 0
            && libc::write(memfd, bytes.as_ptr().cast(), bytes.len()) == bytes.len() as isize
            && libc::fcntl(memfd, libc::F_ADD_SEALS, libc::F_SEAL_SHRINK) == 0;
        if !sized {
            libc::close(memfd);
            libc::close(device);
            return None;
        }
        // udmabuf exports whole pages only
        let create = UdmabufCreate {
            memfd: memfd as u32,
            flags: 0,
            offset: 0,
            size: bytes.len() as u64,
        };
        let fd = libc::ioctl(device, UDMABUF_CREATE, &create as *const UdmabufCreate);
        libc::close(memfd);
        libc::close(device);
        fd
    };
    if created < 0 {
        return None;
    }
    // SAFETY: `created` is a fresh dma-buf fd this test solely owns, handed to
    // `OwnedDmaBuf` which closes it on drop.
    Some(unsafe { OwnedDmaBuf::from_raw(created, 0, 0) })
}

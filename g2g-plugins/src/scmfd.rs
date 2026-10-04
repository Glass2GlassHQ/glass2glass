//! SCM_RIGHTS file-descriptor passing over a Unix socket (`local-dmabuf`,
//! `unixfd`).
//!
//! A dma-buf is a *file descriptor*, not plain bytes: to share it with another
//! process it must be sent as `SCM_RIGHTS` ancillary data of a `sendmsg`, which
//! makes the kernel install a dup of the fd in the receiver (both fds then
//! reference the same underlying buffer, kernel-refcounted). This is the
//! fundamental difference from the CUDA IPC path (see `localipc`), whose
//! 64-byte handle rides any byte transport.
//!
//! The FFI is hand-rolled (no `libc` / `nix` dep, matching the repo's
//! self-contained-feature style) and Linux + LP64 only (dma-buf is Linux; the
//! `cmsghdr` / `msghdr` layouts below assume `size_t` == 8). Struct sizes are
//! asserted at compile time.
//!
//! The public entry points are the non-blocking [`send_with_fds`] /
//! [`recv_with_fds`] raw-syscall wrappers (one message carries up to
//! [`MAX_FDS`] descriptors, a multi-plane buffer needs one per memory) and their
//! single-fd forms. The graph elements drive them through tokio readiness.

use core::ffi::{c_int, c_void};
use core::mem::size_of;

use alloc::vec::Vec;

use std::io;
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};

/// `SOL_SOCKET` (ancillary-data socket level).
const SOL_SOCKET: c_int = 1;
/// `SCM_RIGHTS` (the ancillary-data type that carries fds).
const SCM_RIGHTS: c_int = 1;
/// `MSG_NOSIGNAL`: a write to a closed peer returns EPIPE instead of raising
/// SIGPIPE.
const MSG_NOSIGNAL: c_int = 0x4000;
/// `MSG_CMSG_CLOEXEC`: received fds get O_CLOEXEC set atomically (no fd leak
/// across an exec between recvmsg and our own close).
const MSG_CMSG_CLOEXEC: c_int = 0x4000_0000;
/// `MSG_CTRUNC`: the peer attached more fds than the control buffer holds, and
/// the kernel closed the ones that did not fit.
const MSG_CTRUNC: c_int = 0x8;

/// Most descriptors one message carries. GStreamer caps a buffer at 16
/// memories, and `unixfd` sends one fd per memory.
pub const MAX_FDS: usize = 16;

/// `struct iovec`: a single (base, len) I/O buffer.
#[repr(C)]
struct IoVec {
    iov_base: *mut c_void,
    iov_len: usize,
}

/// `struct msghdr` (Linux LP64). `repr(C)` reproduces the ABI padding (a 4-byte
/// hole after each `c_int` before the next pointer / `usize`).
#[repr(C)]
struct MsgHdr {
    msg_name: *mut c_void,
    msg_namelen: u32,
    msg_iov: *mut IoVec,
    msg_iovlen: usize,
    msg_control: *mut c_void,
    msg_controllen: usize,
    msg_flags: c_int,
}

/// `struct cmsghdr` (Linux LP64): the header in front of each ancillary item.
#[repr(C)]
struct CmsgHdr {
    cmsg_len: usize,
    cmsg_level: c_int,
    cmsg_type: c_int,
}

const CMSG_HEADER_LEN: usize = size_of::<CmsgHdr>();
const FD_LEN: usize = size_of::<c_int>();
/// Room for one `SCM_RIGHTS` item of [`MAX_FDS`] descriptors.
const CONTROL_LEN: usize = cmsg_space(MAX_FDS * FD_LEN);

// The layouts above are load-bearing (the kernel writes / reads these exact
// offsets); assert them so a bad edit fails to compile rather than silently
// corrupting ancillary data.
const _: () = {
    assert!(size_of::<CmsgHdr>() == 16);
    assert!(size_of::<MsgHdr>() == 56);
    assert!(size_of::<IoVec>() == 16);
};

/// An 8-aligned ancillary-data buffer, the alignment `CMSG_ALIGN` assumes.
#[repr(C, align(8))]
struct ControlBuffer([u8; CONTROL_LEN]);

/// `CMSG_ALIGN` on LP64.
const fn cmsg_align(len: usize) -> usize {
    (len + size_of::<usize>() - 1) & !(size_of::<usize>() - 1)
}

/// `CMSG_SPACE`: header plus `data_len` bytes, padded to the next item.
const fn cmsg_space(data_len: usize) -> usize {
    cmsg_align(CMSG_HEADER_LEN) + cmsg_align(data_len)
}

extern "C" {
    fn sendmsg(fd: c_int, msg: *const MsgHdr, flags: c_int) -> isize;
    fn recvmsg(fd: c_int, msg: *mut MsgHdr, flags: c_int) -> isize;
}

/// Send `buf` (which must be non-empty; `SCM_RIGHTS` needs at least one data
/// byte) over socket `sock`, attaching `fds` (at most [`MAX_FDS`]) as one
/// ancillary item. Returns the number of data bytes sent. Non-blocking: an
/// EAGAIN surfaces as [`io::ErrorKind::WouldBlock`] (the caller retries on
/// writability).
///
/// The fds (when present) are attached to the *first* byte of this send, so a
/// caller sending a record in one shot attaches them once and sends the
/// remainder (on a short write) without them.
pub fn send_with_fds(sock: c_int, buf: &[u8], fds: &[c_int]) -> io::Result<usize> {
    debug_assert!(!buf.is_empty(), "SCM_RIGHTS needs at least one data byte");
    if fds.len() > MAX_FDS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "more descriptors than one message carries",
        ));
    }
    let mut iov = IoVec {
        iov_base: buf.as_ptr() as *mut c_void,
        iov_len: buf.len(),
    };
    let mut control = ControlBuffer([0; CONTROL_LEN]);
    let data_len = fds.len() * FD_LEN;
    let header = CmsgHdr {
        cmsg_len: CMSG_HEADER_LEN + data_len,
        cmsg_level: SOL_SOCKET,
        cmsg_type: SCM_RIGHTS,
    };
    control.0[..size_of::<usize>()].copy_from_slice(&header.cmsg_len.to_ne_bytes());
    control.0[size_of::<usize>()..size_of::<usize>() + FD_LEN]
        .copy_from_slice(&header.cmsg_level.to_ne_bytes());
    control.0[size_of::<usize>() + FD_LEN..CMSG_HEADER_LEN]
        .copy_from_slice(&header.cmsg_type.to_ne_bytes());
    for (index, fd) in fds.iter().enumerate() {
        let start = CMSG_HEADER_LEN + index * FD_LEN;
        control.0[start..start + FD_LEN].copy_from_slice(&fd.to_ne_bytes());
    }
    let msg = MsgHdr {
        msg_name: core::ptr::null_mut(),
        msg_namelen: 0,
        msg_iov: &mut iov,
        msg_iovlen: 1,
        msg_control: if fds.is_empty() {
            core::ptr::null_mut()
        } else {
            control.0.as_mut_ptr() as *mut c_void
        },
        msg_controllen: if fds.is_empty() {
            0
        } else {
            cmsg_space(data_len)
        },
        msg_flags: 0,
    };
    // SAFETY: `msg` points at a live iovec and, when fds are attached, at a
    // control buffer holding one well-formed SCM_RIGHTS item. The caller owns
    // the socket fd for the duration of the call.
    let n = unsafe { sendmsg(sock, &msg, MSG_NOSIGNAL) };
    if n < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(n as usize)
}

/// [`send_with_fds`] with at most one fd.
pub fn send_with_fd(sock: c_int, buf: &[u8], fd: Option<c_int>) -> io::Result<usize> {
    send_with_fds(sock, buf, fd.as_slice())
}

/// Receive up to `buf.len()` bytes over socket `sock`, appending every fd that
/// arrives as `SCM_RIGHTS` ancillary data to `fds`. Non-blocking (EAGAIN ->
/// `WouldBlock`).
///
/// A control buffer is supplied on *every* call: on a stream socket, reading
/// past the byte an fd is attached to *without* a control buffer makes the
/// kernel discard the fd, so the caller must never do a plain read across a
/// frame boundary. A peer that attached more than [`MAX_FDS`] fails the read,
/// and the fds that did arrive are closed.
pub fn recv_with_fds(sock: c_int, buf: &mut [u8], fds: &mut Vec<OwnedFd>) -> io::Result<usize> {
    let mut iov = IoVec {
        iov_base: buf.as_mut_ptr() as *mut c_void,
        iov_len: buf.len(),
    };
    let mut control = ControlBuffer([0; CONTROL_LEN]);
    let mut msg = MsgHdr {
        msg_name: core::ptr::null_mut(),
        msg_namelen: 0,
        msg_iov: &mut iov,
        msg_iovlen: 1,
        msg_control: control.0.as_mut_ptr() as *mut c_void,
        msg_controllen: CONTROL_LEN,
        msg_flags: 0,
    };
    // SAFETY: `msg` points at a live iovec and a CONTROL_LEN-byte control
    // buffer, and the kernel writes at most that many ancillary bytes.
    let n = unsafe { recvmsg(sock, &mut msg, MSG_CMSG_CLOEXEC) };
    if n < 0 {
        return Err(io::Error::last_os_error());
    }
    let received_before = fds.len();
    let written = msg.msg_controllen.min(CONTROL_LEN);
    let mut offset = 0;
    // Walk every item the kernel wrote, guarding each length against the
    // buffer: a malformed item must not be read as fds.
    while offset + CMSG_HEADER_LEN <= written {
        let field = |start: usize, len: usize| &control.0[offset + start..offset + start + len];
        let cmsg_len = usize::from_ne_bytes(field(0, size_of::<usize>()).try_into().unwrap());
        let level = c_int::from_ne_bytes(field(size_of::<usize>(), FD_LEN).try_into().unwrap());
        let kind = c_int::from_ne_bytes(
            field(size_of::<usize>() + FD_LEN, FD_LEN)
                .try_into()
                .unwrap(),
        );
        if cmsg_len < CMSG_HEADER_LEN || offset + cmsg_len > written {
            break;
        }
        if level == SOL_SOCKET && kind == SCM_RIGHTS {
            let count = (cmsg_len - CMSG_HEADER_LEN) / FD_LEN;
            for index in 0..count {
                let raw = c_int::from_ne_bytes(
                    field(CMSG_HEADER_LEN + index * FD_LEN, FD_LEN)
                        .try_into()
                        .unwrap(),
                );
                // SAFETY: the kernel just installed `raw` in this process for
                // this message, so nothing else owns it.
                fds.push(unsafe { OwnedFd::from_raw_fd(raw) });
            }
        }
        offset += cmsg_align(cmsg_len);
    }
    if msg.msg_flags & MSG_CTRUNC != 0 {
        fds.truncate(received_before);
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "peer attached more descriptors than one message carries",
        ));
    }
    Ok(n as usize)
}

/// Receive up to `buf.len()` bytes over socket `sock`, capturing a single fd if
/// one arrives as `SCM_RIGHTS` ancillary data. Returns `(bytes, Some(fd))` when
/// an fd accompanied this chunk. More than one fails the read and closes them.
pub fn recv_with_fd(sock: c_int, buf: &mut [u8]) -> io::Result<(usize, Option<c_int>)> {
    let mut fds = Vec::new();
    let n = recv_with_fds(sock, buf, &mut fds)?;
    if fds.len() > 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "more than one descriptor where one was expected",
        ));
    }
    Ok((n, fds.pop().map(IntoRawFd::into_raw_fd)))
}

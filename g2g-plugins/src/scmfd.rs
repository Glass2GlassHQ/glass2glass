//! SCM_RIGHTS file-descriptor passing over a Unix socket (`local-dmabuf`).
//!
//! A dma-buf is a *file descriptor*, not plain bytes: to share it with another
//! process it must be sent as `SCM_RIGHTS` ancillary data of a `sendmsg`, which
//! makes the kernel install a dup of the fd in the receiver (both fds then
//! reference the same underlying buffer, kernel-refcounted). This is the
//! fundamental difference from the CUDA IPC path (see [`crate::localipc`]), whose
//! 64-byte handle rides any byte transport.
//!
//! The FFI is hand-rolled (no `libc` / `nix` dep, matching the repo's
//! self-contained-feature style) and Linux + LP64 only (dma-buf is Linux; the
//! `cmsghdr` / `msghdr` layouts below assume `size_t` == 8). Struct sizes are
//! asserted at compile time.
//!
//! The two public entry points are the non-blocking [`send_with_fd`] /
//! [`recv_with_fd`] raw-syscall wrappers; the graph elements drive them through
//! tokio readiness (`UnixStream::try_io`).

use core::ffi::{c_int, c_void};

use std::io;

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

/// A `cmsghdr` sized for exactly one fd, laid out to match the kernel's
/// `CMSG_SPACE(sizeof(int))` on LP64: the 16-byte header, the 4-byte fd, and 4
/// bytes of tail padding (total 24, 8-aligned). Using a typed struct sidesteps
/// hand-rolled `CMSG_*` alignment arithmetic.
#[repr(C)]
struct SingleFdCmsg {
    cmsg_len: usize,
    cmsg_level: c_int,
    cmsg_type: c_int,
    fd: c_int,
    _pad: c_int,
}

// The layouts above are load-bearing (the kernel writes / reads these exact
// offsets); assert them so a bad edit fails to compile rather than silently
// corrupting ancillary data.
const _: () = {
    assert!(core::mem::size_of::<SingleFdCmsg>() == 24);
    assert!(core::mem::size_of::<MsgHdr>() == 56);
    assert!(core::mem::size_of::<IoVec>() == 16);
    // With `repr(C)` the fields before `fd` (usize + two c_int) occupy exactly 16
    // bytes, so `fd` sits at offset 16 == CMSG_ALIGN(sizeof(cmsghdr)); the size==24
    // assert above pins the trailing pad. `offset_of!` would state this directly
    // but is not stable at MSRV 1.75.
    assert!(core::mem::align_of::<SingleFdCmsg>() == 8);
};

/// `CMSG_LEN(sizeof(int))` on LP64: 16-byte header + 4-byte fd.
const CMSG_LEN_ONE_FD: usize = 16 + 4;

extern "C" {
    fn sendmsg(fd: c_int, msg: *const MsgHdr, flags: c_int) -> isize;
    fn recvmsg(fd: c_int, msg: *mut MsgHdr, flags: c_int) -> isize;
}

/// Send `buf` (which must be non-empty; `SCM_RIGHTS` needs at least one data
/// byte) over socket `sock`, optionally attaching `fd` as ancillary data.
/// Returns the number of data bytes sent. Non-blocking: an EAGAIN surfaces as
/// [`io::ErrorKind::WouldBlock`] (the caller retries on writability).
///
/// The fd (when present) is attached to the *first* byte of this send, so a
/// caller sending a record in one shot attaches the fd once and sends the
/// remainder (on a short write) without it.
pub fn send_with_fd(sock: c_int, buf: &[u8], fd: Option<c_int>) -> io::Result<usize> {
    debug_assert!(!buf.is_empty(), "SCM_RIGHTS needs at least one data byte");
    let mut iov = IoVec {
        iov_base: buf.as_ptr() as *mut c_void,
        iov_len: buf.len(),
    };
    let mut cmsg = SingleFdCmsg {
        cmsg_len: CMSG_LEN_ONE_FD,
        cmsg_level: SOL_SOCKET,
        cmsg_type: SCM_RIGHTS,
        fd: fd.unwrap_or(0),
        _pad: 0,
    };
    let msg = MsgHdr {
        msg_name: core::ptr::null_mut(),
        msg_namelen: 0,
        msg_iov: &mut iov,
        msg_iovlen: 1,
        msg_control: if fd.is_some() {
            (&mut cmsg as *mut SingleFdCmsg) as *mut c_void
        } else {
            core::ptr::null_mut()
        },
        msg_controllen: if fd.is_some() {
            core::mem::size_of::<SingleFdCmsg>()
        } else {
            0
        },
        msg_flags: 0,
    };
    // SAFETY: `msg` points at a live iovec (and cmsg iff an fd is attached); the
    // socket fd is owned by the caller for the duration of the call.
    let n = unsafe { sendmsg(sock, &msg, MSG_NOSIGNAL) };
    if n < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(n as usize)
}

/// Receive up to `buf.len()` bytes over socket `sock`, capturing a single fd if
/// one arrives as `SCM_RIGHTS` ancillary data. Returns `(bytes, Some(fd))` when
/// an fd accompanied this chunk. Non-blocking (EAGAIN -> `WouldBlock`).
///
/// A control buffer is supplied on *every* call: on a stream socket, reading
/// past the byte an fd is attached to *without* a control buffer makes the
/// kernel discard the fd, so the caller must never do a plain read across a
/// frame boundary.
pub fn recv_with_fd(sock: c_int, buf: &mut [u8]) -> io::Result<(usize, Option<c_int>)> {
    let mut iov = IoVec {
        iov_base: buf.as_mut_ptr() as *mut c_void,
        iov_len: buf.len(),
    };
    let mut cmsg = SingleFdCmsg {
        cmsg_len: 0,
        cmsg_level: 0,
        cmsg_type: 0,
        fd: -1,
        _pad: 0,
    };
    let mut msg = MsgHdr {
        msg_name: core::ptr::null_mut(),
        msg_namelen: 0,
        msg_iov: &mut iov,
        msg_iovlen: 1,
        msg_control: (&mut cmsg as *mut SingleFdCmsg) as *mut c_void,
        msg_controllen: core::mem::size_of::<SingleFdCmsg>(),
        msg_flags: 0,
    };
    // SAFETY: `msg` points at a live iovec and a 24-byte control buffer; the
    // kernel writes at most that many ancillary bytes.
    let n = unsafe { recvmsg(sock, &mut msg, MSG_CMSG_CLOEXEC) };
    if n < 0 {
        return Err(io::Error::last_os_error());
    }
    // An fd arrived iff the kernel wrote a well-formed SCM_RIGHTS cmsg covering
    // one fd. Guard every field (never trust the peer): a truncated or unexpected
    // control message must not be read as a valid fd.
    let got_fd = msg.msg_controllen >= CMSG_LEN_ONE_FD
        && cmsg.cmsg_len == CMSG_LEN_ONE_FD
        && cmsg.cmsg_level == SOL_SOCKET
        && cmsg.cmsg_type == SCM_RIGHTS
        && cmsg.fd >= 0;
    let fd = if got_fd { Some(cmsg.fd) } else { None };
    Ok((n as usize, fd))
}

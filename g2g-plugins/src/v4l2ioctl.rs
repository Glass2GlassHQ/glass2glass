// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::{G2gError, HardwareError};
use v4l::v4l2::vidioc;

pub(crate) fn v4l2_err(e: &std::io::Error) -> G2gError {
    G2gError::Hardware(HardwareError::V4l2(e.raw_os_error().unwrap_or(-1)))
}

/// One ioctl on a V4L2 fd, with the errno kept in the reserved V4L2 error
/// arm.
///
/// # Safety
/// `arg` must be the argument struct `request` is defined over (the
/// `vidioc::VIDIOC_*` constant names it), since the kernel reads and writes it
/// through the pointer.
pub(crate) unsafe fn v4l2_ioctl<T>(
    fd: core::ffi::c_int,
    request: vidioc::_IOC_TYPE,
    arg: &mut T,
) -> Result<(), G2gError> {
    // SAFETY: the argument type is this function's contract, and `arg` is a live
    // exclusive borrow for the whole call.
    unsafe { v4l::v4l2::ioctl(fd, request, arg as *mut T as *mut core::ffi::c_void) }
        .map_err(|e| v4l2_err(&e))
}

/// A zeroed V4L2 ioctl argument. Every one of these is plain kernel-ABI data
/// whose reserved fields must be zero.
pub(crate) fn zeroed_arg<T>() -> T {
    // SAFETY: the V4L2 argument structs are `repr(C)` plain data (integers and
    // unions of integers) for which all-zero is the "nothing requested" value.
    unsafe { core::mem::zeroed() }
}

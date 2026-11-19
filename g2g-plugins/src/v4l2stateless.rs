// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;
use core::ffi::c_void;
use core::mem::{offset_of, size_of};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};

use g2g_core::G2gError;
use v4l::v4l2::vidioc;
use v4l::v4l_sys::{
    v4l2_buf_type_V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
    v4l2_buf_type_V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, v4l2_capability, v4l2_fmtdesc,
    v4l2_memory_V4L2_MEMORY_MMAP, v4l2_plane, v4l2_requestbuffers, V4L2_CAP_VIDEO_M2M_MPLANE,
};

use crate::v4l2ioctl::{v4l2_err, v4l2_ioctl, zeroed_arg};

// linux/videodev2.h from kernel 5.11 on. These are declared here because the
// v4l2-sys-mit bindings come from the build host's headers, and older ones lack
// the request fd fields.
pub(crate) const V4L2_CTRL_WHICH_CUR_VAL: u32 = 0;
pub(crate) const V4L2_CTRL_WHICH_REQUEST_VAL: u32 = 0x0f01_0000;
pub(crate) const V4L2_BUF_FLAG_ERROR: u32 = 0x0000_0040;
pub(crate) const V4L2_BUF_FLAG_REQUEST_FD: u32 = 0x0080_0000;
pub(crate) const V4L2_PIX_FMT_H264_SLICE: u32 = u32::from_le_bytes(*b"S264");

pub(crate) const OUTPUT_TYPE: u32 = v4l2_buf_type_V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
pub(crate) const CAPTURE_TYPE: u32 = v4l2_buf_type_V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

// generous for any picture a codec accepts, H.264 level limits keep a coded frame below its raw size
pub(crate) const BITSTREAM_BYTES_PER_PIXEL: u32 = 1;

// linux/media.h, the same encoding on every architecture Linux shares the generic ioctl layout on
const MEDIA_IOC_REQUEST_ALLOC: vidioc::_IOC_TYPE = 0x8004_7c05_u32 as vidioc::_IOC_TYPE;
const MEDIA_REQUEST_IOC_QUEUE: vidioc::_IOC_TYPE = 0x7c80;
const MEDIA_REQUEST_IOC_REINIT: vidioc::_IOC_TYPE = 0x7c81;

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct V4l2ExtControl {
    id: u32,
    size: u32,
    reserved2: u32,
    value: *mut c_void,
}

#[repr(C)]
#[derive(Debug)]
struct V4l2ExtControls {
    which: u32,
    count: u32,
    error_idx: u32,
    request_fd: i32,
    reserved: u32,
    controls: *mut V4l2ExtControl,
}

#[repr(C)]
#[derive(Debug)]
pub(crate) struct V4l2Buffer {
    pub(crate) index: u32,
    pub(crate) buffer_type: u32,
    pub(crate) bytesused: u32,
    pub(crate) flags: u32,
    pub(crate) field: u32,
    pub(crate) timestamp: libc::timeval,
    pub(crate) timecode: [u32; 4],
    pub(crate) sequence: u32,
    pub(crate) memory: u32,
    pub(crate) planes: *mut v4l2_plane,
    pub(crate) length: u32,
    pub(crate) reserved2: u32,
    pub(crate) request_fd: i32,
}

// sizes and offsets printed by a C program built against the kernel 7.3 uAPI headers
const _: () = assert!(size_of::<V4l2ExtControl>() == 20);
const _: () = assert!(offset_of!(V4l2ExtControl, value) == 12);
const _: () = assert!(size_of::<V4l2ExtControls>() == 32);
const _: () = assert!(offset_of!(V4l2ExtControls, request_fd) == 12);
const _: () = assert!(offset_of!(V4l2ExtControls, controls) == 24);
const _: () = assert!(size_of::<V4l2Buffer>() == 88);
const _: () = assert!(offset_of!(V4l2Buffer, timestamp) == 24);
const _: () = assert!(offset_of!(V4l2Buffer, sequence) == 56);
const _: () = assert!(offset_of!(V4l2Buffer, planes) == 64);
const _: () = assert!(offset_of!(V4l2Buffer, length) == 72);
const _: () = assert!(offset_of!(V4l2Buffer, request_fd) == 80);

pub(crate) fn open_node(path: &Path) -> Result<OwnedFd, G2gError> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| v4l2_err(&e))?;
    Ok(OwnedFd::from(file))
}

fn m2m_queue_takes(video: &OwnedFd, buffer_type: u32, pixelformat: u32) -> bool {
    let mut capability: v4l2_capability = zeroed_arg();
    // SAFETY: `v4l2_capability` is VIDIOC_QUERYCAP's argument type.
    if unsafe { v4l2_ioctl(video.as_raw_fd(), vidioc::VIDIOC_QUERYCAP, &mut capability) }.is_err() {
        return false;
    }
    if capability.device_caps & V4L2_CAP_VIDEO_M2M_MPLANE == 0 {
        return false;
    }
    (0..)
        .map_while(|index| {
            let mut format: v4l2_fmtdesc = zeroed_arg();
            format.index = index;
            format.type_ = buffer_type;
            // SAFETY: `v4l2_fmtdesc` is VIDIOC_ENUM_FMT's argument type.
            unsafe { v4l2_ioctl(video.as_raw_fd(), vidioc::VIDIOC_ENUM_FMT, &mut format) }
                .ok()
                .map(|()| format.pixelformat)
        })
        .any(|format| format == pixelformat)
}

// sysfs lists the media controller of a video node as a `mediaN` entry of its device directory
fn media_node(video: &Path) -> Option<PathBuf> {
    let name = video.file_name()?;
    let device = Path::new("/sys/class/video4linux")
        .join(name)
        .join("device");
    std::fs::read_dir(device)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.file_name())
        .find(|entry| entry.to_string_lossy().starts_with("media"))
        .map(|entry| Path::new("/dev").join(entry))
}

#[derive(Debug)]
pub(crate) struct M2mNodes {
    pub(crate) video: PathBuf,
    pub(crate) media: PathBuf,
}

// the first memory-to-memory video node whose `buffer_type` queue takes `pixelformat` and that has a media controller
pub(crate) fn find_m2m_nodes(
    buffer_type: u32,
    pixelformat: u32,
) -> Result<Option<M2mNodes>, G2gError> {
    let mut videos: Vec<PathBuf> = std::fs::read_dir("/dev")
        .map_err(|e| v4l2_err(&e))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("video"))
        })
        .collect();
    videos.sort();
    Ok(videos
        .into_iter()
        .filter(|video| {
            open_node(video).is_ok_and(|fd| m2m_queue_takes(&fd, buffer_type, pixelformat))
        })
        .find_map(|video| {
            let media = media_node(&video)?;
            Some(M2mNodes { video, media })
        }))
}

#[derive(Debug)]
pub(crate) struct Mapping {
    address: *mut u8,
    length: usize,
}

// SAFETY: the mapping is owned by one codec session and unmapped once, on drop.
unsafe impl Send for Mapping {}

impl Mapping {
    pub(crate) fn of_buffer(
        video: &OwnedFd,
        buffer_type: u32,
        index: u32,
    ) -> Result<Self, G2gError> {
        let mut plane: v4l2_plane = zeroed_arg();
        query_planes(video, buffer_type, index, core::slice::from_mut(&mut plane))?;
        Self::of_plane(video, &plane)
    }

    pub(crate) fn of_planes(
        video: &OwnedFd,
        buffer_type: u32,
        index: u32,
        plane_count: usize,
    ) -> Result<Vec<Self>, G2gError> {
        let mut planes: Vec<v4l2_plane> = (0..plane_count).map(|_| zeroed_arg()).collect();
        query_planes(video, buffer_type, index, &mut planes)?;
        planes
            .iter()
            .map(|plane| Self::of_plane(video, plane))
            .collect()
    }

    fn of_plane(video: &OwnedFd, plane: &v4l2_plane) -> Result<Self, G2gError> {
        let length = plane.length as usize;
        // SAFETY: the kernel filled `mem_offset` for an MMAP plane.
        let offset = unsafe { plane.m.mem_offset };
        // SAFETY: the offset and length are the ones VIDIOC_QUERYBUF reported for this buffer.
        let address = unsafe {
            v4l::v4l2::mmap(
                core::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                video.as_raw_fd(),
                libc::off_t::from(offset),
            )
        }
        .map_err(|e| v4l2_err(&e))?;
        Ok(Self {
            address: address.cast(),
            length,
        })
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping is `length` bytes and lives as long as `self`.
        unsafe { core::slice::from_raw_parts(self.address, self.length) }
    }

    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the mapping is `length` bytes, writable, and borrowed exclusively.
        unsafe { core::slice::from_raw_parts_mut(self.address, self.length) }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the address and length are the ones mmap returned. An unmap
        // error has nowhere to go and leaves only the address space reserved.
        let _ = unsafe { v4l::v4l2::munmap(self.address.cast(), self.length) };
    }
}

fn query_planes(
    video: &OwnedFd,
    buffer_type: u32,
    index: u32,
    planes: &mut [v4l2_plane],
) -> Result<(), G2gError> {
    let mut buffer = planes_buffer_arg(buffer_type, index, v4l2_memory_V4L2_MEMORY_MMAP, planes);
    // SAFETY: `V4l2Buffer` matches the kernel's `v4l2_buffer`, which
    // VIDIOC_QUERYBUF takes, and `planes` outlives the call.
    unsafe { v4l2_ioctl(video.as_raw_fd(), vidioc::VIDIOC_QUERYBUF, &mut buffer) }
}

pub(crate) fn planes_buffer_arg(
    buffer_type: u32,
    index: u32,
    memory: u32,
    planes: &mut [v4l2_plane],
) -> V4l2Buffer {
    let mut buffer: V4l2Buffer = zeroed_arg();
    buffer.index = index;
    buffer.buffer_type = buffer_type;
    buffer.memory = memory;
    buffer.length = planes.len() as u32;
    buffer.planes = planes.as_mut_ptr();
    buffer
}

pub(crate) fn buffer_arg(buffer_type: u32, index: u32, plane: &mut v4l2_plane) -> V4l2Buffer {
    planes_buffer_arg(
        buffer_type,
        index,
        v4l2_memory_V4L2_MEMORY_MMAP,
        core::slice::from_mut(plane),
    )
}

pub(crate) fn request_buffers(
    video: &OwnedFd,
    buffer_type: u32,
    memory: u32,
    count: u32,
) -> Result<u32, G2gError> {
    let mut request: v4l2_requestbuffers = zeroed_arg();
    request.count = count;
    request.type_ = buffer_type;
    request.memory = memory;
    // SAFETY: `v4l2_requestbuffers` is VIDIOC_REQBUFS's argument type.
    unsafe { v4l2_ioctl(video.as_raw_fd(), vidioc::VIDIOC_REQBUFS, &mut request) }?;
    Ok(request.count)
}

pub(crate) fn set_stream(video: &OwnedFd, buffer_type: u32, on: bool) -> Result<(), G2gError> {
    let mut stream_type = buffer_type as core::ffi::c_int;
    let request = if on {
        vidioc::VIDIOC_STREAMON
    } else {
        vidioc::VIDIOC_STREAMOFF
    };
    // SAFETY: VIDIOC_STREAMON and VIDIOC_STREAMOFF take a buffer-type int.
    unsafe { v4l2_ioctl(video.as_raw_fd(), request, &mut stream_type) }?;
    Ok(())
}

pub(crate) fn set_controls(
    video: &OwnedFd,
    which: u32,
    request_fd: i32,
    controls: &mut [V4l2ExtControl],
) -> Result<(), G2gError> {
    let mut arg = V4l2ExtControls {
        which,
        count: controls.len() as u32,
        error_idx: 0,
        request_fd,
        reserved: 0,
        controls: controls.as_mut_ptr(),
    };
    // SAFETY: `V4l2ExtControls` matches the kernel's `v4l2_ext_controls`, which
    // VIDIOC_S_EXT_CTRLS takes, and every control payload outlives the call.
    unsafe { v4l2_ioctl(video.as_raw_fd(), vidioc::VIDIOC_S_EXT_CTRLS, &mut arg) }?;
    Ok(())
}

pub(crate) fn control<T>(id: u32, payload: &mut T) -> V4l2ExtControl {
    V4l2ExtControl {
        id,
        size: size_of::<T>() as u32,
        reserved2: 0,
        value: (payload as *mut T).cast(),
    }
}

// a menu control carries its value in the union, with no payload size
pub(crate) fn menu_control(id: u32, value: i32) -> V4l2ExtControl {
    V4l2ExtControl {
        id,
        size: 0,
        reserved2: 0,
        value: value as isize as *mut c_void,
    }
}

// stateless codecs name a reference picture by the vb2 timestamp of the buffer that carried it
pub(crate) fn timestamp_ns(picture_number: u64) -> u64 {
    picture_number * 1000
}

pub(crate) fn buffer_timestamp(picture_number: u64) -> libc::timeval {
    let timestamp_us = timestamp_ns(picture_number) / 1000;
    libc::timeval {
        tv_sec: (timestamp_us / 1_000_000) as libc::time_t,
        tv_usec: (timestamp_us % 1_000_000) as libc::suseconds_t,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RequestError {
    TimedOut,
    Device(G2gError),
}

impl From<G2gError> for RequestError {
    fn from(error: G2gError) -> Self {
        Self::Device(error)
    }
}

#[derive(Debug)]
pub(crate) struct MediaRequest(OwnedFd);

impl MediaRequest {
    pub(crate) fn alloc(media: &OwnedFd) -> Result<Self, G2gError> {
        let mut request_fd: core::ffi::c_int = -1;
        // SAFETY: MEDIA_IOC_REQUEST_ALLOC writes the new request's fd into an int.
        unsafe { v4l2_ioctl(media.as_raw_fd(), MEDIA_IOC_REQUEST_ALLOC, &mut request_fd) }?;
        // SAFETY: the kernel just created this fd for this process and nothing else owns it.
        Ok(Self(unsafe { OwnedFd::from_raw_fd(request_fd) }))
    }

    pub(crate) fn fd(&self) -> i32 {
        self.0.as_raw_fd()
    }

    pub(crate) fn reinit(&self) -> Result<(), G2gError> {
        // SAFETY: MEDIA_REQUEST_IOC_REINIT takes no argument, and the request is idle or complete.
        unsafe { v4l::v4l2::ioctl(self.fd(), MEDIA_REQUEST_IOC_REINIT, core::ptr::null_mut()) }
            .map_err(|e| v4l2_err(&e))
    }

    pub(crate) fn queue_and_wait(&self, timeout_ms: i32) -> Result<(), RequestError> {
        // SAFETY: MEDIA_REQUEST_IOC_QUEUE takes no argument.
        unsafe { v4l::v4l2::ioctl(self.fd(), MEDIA_REQUEST_IOC_QUEUE, core::ptr::null_mut()) }
            .map_err(|e| v4l2_err(&e))?;
        let mut poll = libc::pollfd {
            fd: self.fd(),
            events: libc::POLLPRI,
            revents: 0,
        };
        // SAFETY: one live pollfd for the duration of the call.
        let ready = unsafe { libc::poll(&mut poll, 1, timeout_ms) };
        if ready < 0 {
            return Err(v4l2_err(&std::io::Error::last_os_error()).into());
        }
        if ready == 0 {
            return Err(RequestError::TimedOut);
        }
        Ok(())
    }
}

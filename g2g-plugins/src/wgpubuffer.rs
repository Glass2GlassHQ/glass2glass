use core::any::Any;

use alloc::sync::Arc;

use g2g_core::memory::{OwnedWgpuBuffer, WgpuBufferKeepAlive};

/// Owner for a plain `wgpu::Buffer` and the device and queue it lives on: what
/// [`wrap_buffer`] attaches, so `wgputodmabuf` can recover the buffer and copy
/// from it and `wgpudownload` can read it back.
#[derive(Debug)]
pub struct PlainWgpuBuffer {
    buffer: wgpu::Buffer,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl PlainWgpuBuffer {
    /// The wrapped buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

impl WgpuBufferKeepAlive for PlainWgpuBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Wrap a `wgpu::Buffer` (with `COPY_SRC` usage) and the device and queue it
/// lives on as a `WgpuBuffer` frame domain of `len` valid bytes.
pub fn wrap_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: wgpu::Buffer,
    len: usize,
) -> OwnedWgpuBuffer {
    OwnedWgpuBuffer::new(
        len,
        Arc::new(PlainWgpuBuffer {
            buffer,
            device: device.clone(),
            queue: queue.clone(),
        }),
    )
}

// buffer copies and storage bindings move whole 4-byte words
pub(crate) fn whole_word_size(len: u64) -> Option<u64> {
    len.checked_next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
}

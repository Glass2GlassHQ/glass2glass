use std::sync::Mutex;

use g2g_core::{G2gError, HardwareError};

const WRITE_ALIGNMENT: usize = wgpu::COPY_BUFFER_ALIGNMENT as usize;

// `write_buffer` takes whole 4-byte words only, so the last partial word goes up zero padded
pub(crate) fn write_padded(queue: &wgpu::Queue, buffer: &wgpu::Buffer, bytes: &[u8]) {
    let aligned = bytes.len() - bytes.len() % WRITE_ALIGNMENT;
    if aligned > 0 {
        queue.write_buffer(buffer, 0, &bytes[..aligned]);
    }
    let tail = &bytes[aligned..];
    if !tail.is_empty() {
        let mut word = [0u8; WRITE_ALIGNMENT];
        word[..tail.len()].copy_from_slice(tail);
        queue.write_buffer(buffer, aligned as u64, &word);
    }
}

// a `MAP_READ` buffer kept across frames, rebuilt only for a larger readback
#[derive(Debug, Default)]
pub(crate) struct ReadbackStaging {
    buffer: Mutex<Option<wgpu::Buffer>>,
}

impl ReadbackStaging {
    // submits `encoder` with a copy of `source[..len]` appended and waits for the bytes
    pub(crate) fn read_back(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mut encoder: wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        len: u64,
    ) -> Result<Vec<u8>, G2gError> {
        let mut cached = self.buffer.lock().expect("readback staging lock poisoned");
        let staging = match cached.take() {
            Some(buffer) if buffer.size() >= len => buffer,
            _ => device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: len,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, len);
        queue.submit([encoder.finish()]);

        let slice = staging.slice(..len);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        rx.recv()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let bytes = slice.get_mapped_range().to_vec();
        staging.unmap();
        *cached = Some(staging);
        Ok(bytes)
    }
}

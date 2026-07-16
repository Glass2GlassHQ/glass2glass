//! Inline GPU tensor preprocessing via wgpu compute (DESIGN.md §5.1).
//!
//! `WgpuPreprocess` is the hardware-first preprocessing pillar: an
//! `AsyncElement` that takes an NV12 video frame and emits a normalized f32
//! NCHW RGB tensor (`Caps::RawVideo{Nv12} -> Caps::Tensor{F32,[1,3,H,W],Nchw}`),
//! doing the BT.601 colour conversion and the `value / 255` normalization in a
//! wgpu compute shader rather than on the CPU. It produces the same tensor
//! contract `OrtInference` builds on the CPU, so it composes with the existing
//! tensor graph (`-> TensorBatcher -> inference -> TensorPostprocess`).
//!
//! Both ends of the compute can now stay on the GPU:
//! - **Output (M215, [`with_gpu_output`](WgpuPreprocess::with_gpu_output)):** the
//!   f32 tensor is left in a `wgpu::Buffer` (`MemoryDomain::WgpuBuffer`) instead
//!   of read back to `MemoryDomain::System`, so `WgpuInference` binds it on-device.
//! - **Input (M217, surface-import):** when the NV12 frame arrives already on the
//!   GPU as a `MemoryDomain::WgpuTexture` (an R8Uint texture in standard NV12
//!   byte layout, see [`WgpuNv12Texture`]), the element samples it straight into
//!   the compute pass on the producer's own device, with no CPU upload. The
//!   default `MemoryDomain::System` path (upload NV12 bytes to a storage buffer)
//!   is unchanged.
//!
//! With both ends GPU-resident, `surface -> WgpuPreprocess -> WgpuInference` runs
//! with the pixels never touching the CPU. A real GPU NV12 decoder
//! (`DmaBuf`/`D3D11Texture`/CUDA import into a wgpu texture) is the producer that
//! slots in upstream; until one lands, [`nv12_to_gpu_texture`] stands in for it.
//! RGBA input (normalize only, no colour convert) is a small follow-up.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, HardwareError,
    MemoryDomain, OutputSink, OwnedWgpuBuffer, OwnedWgpuTexture, PipelinePacket, Rate,
    RawVideoFormat, TensorDType, TensorLayout, TensorShape, WgpuBufferKeepAlive, WgpuKeepAlive,
};

/// 8x8 invocations per workgroup; the dispatch covers ceil(W/8) x ceil(H/8).
const WORKGROUP: u32 = 8;

/// NV12 -> normalized planar RGB (BT.601 limited range), in a compute pass.
/// The NV12 bytes arrive as a packed `array<u32>`; `out` is the f32 NCHW
/// tensor (R plane, then G, then B), each value in `[0, 1]`.
const SHADER: &str = r#"
struct Dims { width: u32, height: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> nv12: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn load_byte(i: u32) -> f32 {
    let word = nv12[i / 4u];
    let shift = (i % 4u) * 8u;
    return f32((word >> shift) & 0xFFu);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = dims.width;
    let h = dims.height;
    if (x >= w || y >= h) { return; }

    let luma_index = y * w + x;
    let yv = load_byte(luma_index);
    // NV12: w*h luma bytes, then interleaved Cb,Cr at half resolution.
    let uv_base = w * h;
    let uv_index = uv_base + (y / 2u) * w + (x / 2u) * 2u;
    let cb = load_byte(uv_index) - 128.0;
    let cr = load_byte(uv_index + 1u) - 128.0;

    let yy = (yv - 16.0) * 1.164383;
    let r = yy + 1.596027 * cr;
    let g = yy - 0.391762 * cb - 0.812968 * cr;
    let b = yy + 2.017232 * cb;

    let area = w * h;
    out[luma_index] = clamp(r, 0.0, 255.0) / 255.0;
    out[area + luma_index] = clamp(g, 0.0, 255.0) / 255.0;
    out[2u * area + luma_index] = clamp(b, 0.0, 255.0) / 255.0;
}
"#;

/// Surface-import variant of `SHADER` (M217): the NV12 frame arrives as an
/// R8Uint texture of size `width x (height * 3/2)` holding the bytes in the
/// standard NV12 layout (Y plane in rows `[0, h)`, interleaved Cb,Cr in rows
/// `[h, h*3/2)`), so the byte at logical index `i` is texel `(i % w, i / w)`.
/// `textureLoad` reads the exact integer byte (no sampler, no filtering), so the
/// math and the output are identical to the storage-buffer path. `out` is the
/// same f32 NCHW tensor.
const TEX_SHADER: &str = r#"
struct Dims { width: u32, height: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var nv12: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = dims.width;
    let h = dims.height;
    if (x >= w || y >= h) { return; }

    let yv = f32(textureLoad(nv12, vec2<i32>(i32(x), i32(y)), 0).r);
    // UV is half-resolution, packed in the rows after the Y plane: the Cb,Cr
    // pair for this pixel sits at column (x/2)*2 of row h + y/2.
    let cx = i32((x / 2u) * 2u);
    let cy = i32(h + y / 2u);
    let cb = f32(textureLoad(nv12, vec2<i32>(cx, cy), 0).r) - 128.0;
    let cr = f32(textureLoad(nv12, vec2<i32>(cx + 1, cy), 0).r) - 128.0;

    let yy = (yv - 16.0) * 1.164383;
    let r = yy + 1.596027 * cr;
    let g = yy - 0.391762 * cb - 0.812968 * cr;
    let b = yy + 2.017232 * cb;

    let area = w * h;
    let li = y * w + x;
    out[li] = clamp(r, 0.0, 255.0) / 255.0;
    out[area + li] = clamp(g, 0.0, 255.0) / 255.0;
    out[2u * area + li] = clamp(b, 0.0, 255.0) / 255.0;
}
"#;

/// Surface-import variant for an already-RGB frame (M304): the input is an
/// `Rgba8Unorm` texture whose YCbCr->RGB conversion already happened upstream
/// (the Android `MediaCodecDec` GPU path samples the decoded `AHardwareBuffer`
/// through an immutable ycbcr sampler). `textureLoad` returns normalized f32
/// already, so this just writes R,G,B into the NCHW tensor, no colour math.
#[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
const TEX_SHADER_RGBA: &str = r#"
struct Dims { width: u32, height: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var img: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = dims.width;
    let h = dims.height;
    if (x >= w || y >= h) { return; }

    let c = textureLoad(img, vec2<i32>(i32(x), i32(y)), 0);
    let area = w * h;
    let li = y * w + x;
    out[li] = c.r;
    out[area + li] = c.g;
    out[2u * area + li] = c.b;
}
"#;

/// The host BT.601 reference matching `SHADER`, kept public so the test (and a
/// CPU-fallback caller) can compare against the GPU output. Returns the f32
/// NCHW RGB tensor for one NV12 frame.
pub fn nv12_to_rgb_tensor(nv12: &[u8], width: usize, height: usize) -> Vec<f32> {
    let area = width * height;
    let uv_base = area;
    let byte = |i: usize| nv12[i] as f32;
    let mut out = vec![0f32; 3 * area];
    for y in 0..height {
        for x in 0..width {
            let li = y * width + x;
            let yv = byte(li);
            let uvi = uv_base + (y / 2) * width + (x / 2) * 2;
            let cb = byte(uvi) - 128.0;
            let cr = byte(uvi + 1) - 128.0;
            let yy = (yv - 16.0) * 1.164383;
            let r = (yy + 1.596027 * cr).clamp(0.0, 255.0) / 255.0;
            let g = (yy - 0.391762 * cb - 0.812968 * cr).clamp(0.0, 255.0) / 255.0;
            let b = (yy + 2.017232 * cb).clamp(0.0, 255.0) / 255.0;
            out[li] = r;
            out[area + li] = g;
            out[2 * area + li] = b;
        }
    }
    out
}

/// GPU resources sized to a fixed `W x H`, built lazily on the first frame.
#[derive(Debug)]
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    nv12_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    nv12_len: usize,
    nv12_padded: usize,
    out_bytes: usize,
}

/// Surface-import GPU resources (M217): the texture-sampling pipeline and the
/// output buffers, built lazily on the first GPU-texture frame, on the device
/// that frame's texture lives on (a texture is bindable only on its own device).
/// No input buffer: the input is the incoming texture, bound per frame, so the
/// bind group is rebuilt per dispatch.
#[derive(Debug)]
struct TexGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    dims_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    out_bytes: usize,
}

#[derive(Debug)]
pub struct WgpuPreprocess {
    width: u32,
    height: u32,
    configured: bool,
    gpu: Option<Gpu>,
    /// Surface-import resources, built on the first GPU-texture frame from that
    /// frame's device (M217). Separate from `gpu` because the texture path binds
    /// a sampled texture, not a storage buffer, and adopts the producer's device.
    tex_gpu: Option<TexGpu>,
    /// RGBA surface-import resources (M304), built on the first RGBA GPU-texture
    /// frame. Separate pipeline from `tex_gpu` (samples `texture_2d<f32>`, no
    /// YCbCr math); the input is already-converted RGBA from `MediaCodecDec`.
    #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
    tex_rgba_gpu: Option<TexGpu>,
    last_caps: Option<Caps>,
    emitted: u64,
    /// When set, emit the tensor as a GPU-resident `MemoryDomain::WgpuBuffer`
    /// (no GPU->CPU read-back) instead of `MemoryDomain::System` (M215). Lets a
    /// downstream GPU consumer read the tensor on-device.
    gpu_output: bool,
}

impl Default for WgpuPreprocess {
    fn default() -> Self {
        Self::new()
    }
}

impl WgpuPreprocess {
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            configured: false,
            gpu: None,
            tex_gpu: None,
            #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
            tex_rgba_gpu: None,
            last_caps: None,
            emitted: 0,
            gpu_output: false,
        }
    }

    /// Emit the tensor GPU-resident (`MemoryDomain::WgpuBuffer`) rather than
    /// reading it back to system memory (M215): the compute output stays in a
    /// `wgpu::Buffer`, so a downstream GPU consumer reads it with no
    /// GPU->CPU copy. A CPU consumer recovers the bytes via the buffer owner's
    /// `read_back`. Default off (the system-memory variant).
    pub fn with_gpu_output(mut self) -> Self {
        self.gpu_output = true;
        self
    }

    /// Count of tensor `DataFrame`s pushed downstream. Useful in tests.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn supported_input(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    fn tensor_caps(&self) -> Caps {
        Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, 3, self.height, self.width]),
            layout: TensorLayout::Nchw,
        }
    }

    async fn ensure_gpu(&mut self) -> Result<(), G2gError> {
        if self.gpu.is_some() {
            return Ok(());
        }
        self.gpu = Some(build_gpu(self.width, self.height).await?);
        Ok(())
    }

    /// Upload the NV12 frame, run the compute pass, and read the f32 tensor
    /// back as little-endian bytes (the `OrtInference` output byte format).
    /// Blocks the calling task on `poll(Wait)`; offloading the GPU round-trip
    /// to a blocking pool is a follow-up.
    fn dispatch(&self, nv12: &[u8]) -> Result<Box<[u8]>, G2gError> {
        let gpu = self.gpu.as_ref().ok_or(G2gError::NotConfigured)?;
        if nv12.len() < gpu.nv12_len {
            return Err(G2gError::CapsMismatch);
        }
        let mut padded = vec![0u8; gpu.nv12_padded];
        padded[..gpu.nv12_len].copy_from_slice(&nv12[..gpu.nv12_len]);
        gpu.queue.write_buffer(&gpu.nv12_buf, 0, &padded);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nv12->rgb"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            let gx = self.width.div_ceil(WORKGROUP);
            let gy = self.height.div_ceil(WORKGROUP);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        encoder.copy_buffer_to_buffer(&gpu.out_buf, 0, &gpu.staging, 0, gpu.out_bytes as u64);
        gpu.queue.submit([encoder.finish()]);

        let slice = gpu.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        rx.recv()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        let bytes = slice.get_mapped_range().to_vec().into_boxed_slice();
        gpu.staging.unmap();
        Ok(bytes)
    }

    /// GPU-output variant of [`dispatch`](Self::dispatch) (M215): run the same
    /// compute, then copy the result into a fresh per-frame `wgpu::Buffer` (a
    /// GPU->GPU copy, on-device) and hand it downstream, with NO map / poll /
    /// read-back. The fresh buffer is `STORAGE | COPY_SRC` so a downstream GPU
    /// consumer can bind it, or read it back via the owner. A per-frame buffer
    /// (not the shared `out_buf`) so the next frame's compute does not clobber a
    /// buffer still in flight downstream.
    fn dispatch_gpu(&self, nv12: &[u8]) -> Result<OwnedWgpuBuffer, G2gError> {
        let gpu = self.gpu.as_ref().ok_or(G2gError::NotConfigured)?;
        if nv12.len() < gpu.nv12_len {
            return Err(G2gError::CapsMismatch);
        }
        let mut padded = vec![0u8; gpu.nv12_padded];
        padded[..gpu.nv12_len].copy_from_slice(&nv12[..gpu.nv12_len]);
        gpu.queue.write_buffer(&gpu.nv12_buf, 0, &padded);

        let frame_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preprocess-tensor"),
            size: gpu.out_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nv12->rgb"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            let gx = self.width.div_ceil(WORKGROUP);
            let gy = self.height.div_ceil(WORKGROUP);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        // On-device copy into the per-frame buffer; no read-back to the CPU.
        encoder.copy_buffer_to_buffer(&gpu.out_buf, 0, &frame_buf, 0, gpu.out_bytes as u64);
        gpu.queue.submit([encoder.finish()]);

        let owner =
            WgpuBufferOwner::new(gpu.device.clone(), gpu.queue.clone(), frame_buf, gpu.out_bytes);
        Ok(OwnedWgpuBuffer::new(gpu.out_bytes, std::sync::Arc::new(owner)))
    }

    /// Build the surface-import pipeline and output buffers on `device` (M217).
    /// Idempotent: built once, on the first GPU-texture frame, because the
    /// device is only known once such a frame arrives.
    fn ensure_tex_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.tex_gpu.is_some() {
            return;
        }
        self.tex_gpu = Some(build_tex_gpu(device, queue, self.width, self.height));
    }

    /// Surface-import dispatch (M217): sample the incoming NV12 texture straight
    /// into the compute pass on its own device, no CPU upload. Returns the tensor
    /// domain, GPU-resident (`WgpuBuffer`) when `gpu_output` is set or read back
    /// to `System` otherwise, mirroring [`dispatch`] / [`dispatch_gpu`]. The bind
    /// group is rebuilt per frame because the input texture changes per frame.
    fn dispatch_tex(&self, owner: &WgpuNv12Texture) -> Result<MemoryDomain, G2gError> {
        let tg = self.tex_gpu.as_ref().ok_or(G2gError::NotConfigured)?;
        let texture = owner.texture();
        // The texture must hold the NV12 frame in the standard byte layout:
        // width x (height + height/2), one byte per texel (R8Uint).
        if texture.width() != self.width || texture.height() != self.height + self.height / 2 {
            return Err(G2gError::CapsMismatch);
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let layout = tg.pipeline.get_bind_group_layout(0);
        let bind_group = tg.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nv12-tex-binding"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tg.dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tg.out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder =
            tg.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nv12-tex->rgb"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&tg.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let gx = self.width.div_ceil(WORKGROUP);
            let gy = self.height.div_ceil(WORKGROUP);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        if self.gpu_output {
            // Fresh per-frame buffer, like dispatch_gpu, so the next frame's
            // compute can't clobber one still in flight downstream.
            let frame_buf = tg.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("preprocess-tensor"),
                size: tg.out_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&tg.out_buf, 0, &frame_buf, 0, tg.out_bytes as u64);
            tg.queue.submit([encoder.finish()]);
            let owner =
                WgpuBufferOwner::new(tg.device.clone(), tg.queue.clone(), frame_buf, tg.out_bytes);
            Ok(MemoryDomain::WgpuBuffer(OwnedWgpuBuffer::new(
                tg.out_bytes,
                std::sync::Arc::new(owner),
            )))
        } else {
            encoder.copy_buffer_to_buffer(&tg.out_buf, 0, &tg.staging, 0, tg.out_bytes as u64);
            tg.queue.submit([encoder.finish()]);
            let slice = tg.staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            tg.device
                .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            rx.recv()
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            let bytes = slice.get_mapped_range().to_vec().into_boxed_slice();
            tg.staging.unmap();
            Ok(MemoryDomain::System(SystemSlice::from_boxed(bytes)))
        }
    }

    /// Try to consume the GPU texture as an already-RGB `WgpuRgbaTexture` (the
    /// M304 Android decode path). Returns `Ok(None)` if the keep-alive is not one
    /// (so the caller falls through to `UnsupportedDomain`).
    #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
    fn try_dispatch_rgba(
        &mut self,
        any: &dyn core::any::Any,
    ) -> Result<Option<MemoryDomain>, G2gError> {
        let Some(owner) = any.downcast_ref::<g2g_plugins::mediacodec_wgpu::WgpuRgbaTexture>() else {
            return Ok(None);
        };
        self.ensure_tex_rgba_gpu(owner.device(), owner.queue());
        Ok(Some(self.dispatch_tex_rgba(owner)?))
    }

    /// No RGBA-texture producer off the Android `mediacodec-wgpu` path.
    #[cfg(not(all(target_os = "android", feature = "mediacodec-wgpu")))]
    fn try_dispatch_rgba(
        &mut self,
        _any: &dyn core::any::Any,
    ) -> Result<Option<MemoryDomain>, G2gError> {
        Ok(None)
    }

    #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
    fn ensure_tex_rgba_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.tex_rgba_gpu.is_some() {
            return;
        }
        self.tex_rgba_gpu = Some(build_tex_rgba_gpu(device, queue, self.width, self.height));
    }

    /// RGBA surface-import dispatch (M304): sample the already-converted RGBA
    /// texture straight into the tensor on its own device, no colour math, no CPU
    /// upload. Mirrors [`dispatch_tex`] but binds an `Rgba8Unorm` `texture_2d<f32>`
    /// sized `width x height` (not the NV12 `x 3/2`).
    #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
    fn dispatch_tex_rgba(
        &self,
        owner: &g2g_plugins::mediacodec_wgpu::WgpuRgbaTexture,
    ) -> Result<MemoryDomain, G2gError> {
        let tg = self.tex_rgba_gpu.as_ref().ok_or(G2gError::NotConfigured)?;
        let texture = owner.texture();
        if texture.width() != self.width || texture.height() != self.height {
            return Err(G2gError::CapsMismatch);
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let layout = tg.pipeline.get_bind_group_layout(0);
        let bind_group = tg.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rgba-tex-binding"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tg.dims_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry { binding: 2, resource: tg.out_buf.as_entire_binding() },
            ],
        });

        let mut encoder =
            tg.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rgba-tex->tensor"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&tg.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let gx = self.width.div_ceil(WORKGROUP);
            let gy = self.height.div_ceil(WORKGROUP);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        if self.gpu_output {
            let frame_buf = tg.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("preprocess-tensor"),
                size: tg.out_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&tg.out_buf, 0, &frame_buf, 0, tg.out_bytes as u64);
            tg.queue.submit([encoder.finish()]);
            let owner =
                WgpuBufferOwner::new(tg.device.clone(), tg.queue.clone(), frame_buf, tg.out_bytes);
            Ok(MemoryDomain::WgpuBuffer(OwnedWgpuBuffer::new(
                tg.out_bytes,
                std::sync::Arc::new(owner),
            )))
        } else {
            encoder.copy_buffer_to_buffer(&tg.out_buf, 0, &tg.staging, 0, tg.out_bytes as u64);
            tg.queue.submit([encoder.finish()]);
            let slice = tg.staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            tg.device
                .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            rx.recv()
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            let bytes = slice.get_mapped_range().to_vec().into_boxed_slice();
            tg.staging.unmap();
            Ok(MemoryDomain::System(SystemSlice::from_boxed(bytes)))
        }
    }
}

/// Owns a GPU-resident linear tensor buffer: the `wgpu::Buffer` holding an f32
/// tensor, plus the device / queue that produced it (needed to read it back or
/// to keep submitting work on the same device). Boxed as the
/// [`WgpuBufferKeepAlive`] of a [`MemoryDomain::WgpuBuffer`]; a downstream GPU
/// consumer downcasts to bind the buffer directly, or calls
/// [`read_back`](Self::read_back) for the CPU bytes.
///
/// First produced by [`WgpuPreprocess`] in GPU-output mode (M215, the f32 NCHW
/// RGB tensor); it is also the owner `WgpuInference` emits for its GPU-resident
/// logits, so the same downcast recovers either producer's buffer. A consumer
/// that adopts [`device`](Self::device) / [`queue`](Self::queue) keeps the
/// tensor on the same device, so its work serializes after the producer's on the
/// shared queue with no CPU round-trip (M216).
#[derive(Debug)]
pub struct WgpuBufferOwner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffer: wgpu::Buffer,
    len: usize,
}

impl WgpuBufferOwner {
    /// Wrap a GPU buffer with the device / queue that produced it, for handing
    /// downstream as a [`MemoryDomain::WgpuBuffer`]. `len` is the valid f32
    /// payload length in bytes.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, buffer: wgpu::Buffer, len: usize) -> Self {
        Self { device, queue, buffer, len }
    }

    /// The backing GPU buffer, for a downstream GPU consumer to bind directly.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// The device that produced the buffer, so a downstream GPU consumer can
    /// adopt it and bind the buffer (a `wgpu::Buffer` is bindable only on its
    /// own device) rather than reading back to the CPU.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue paired with [`device`](Self::device). Submitting the consumer's
    /// work here orders it after the producer's already-submitted work, so the
    /// buffer is ready without an explicit fence or read-back.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Copy the tensor back to the CPU (the deferred read-back a CPU consumer
    /// pays, instead of the element paying it for every frame): copy into a
    /// `MAP_READ` staging buffer, map, and return the little-endian f32 bytes.
    pub fn read_back(&self) -> Result<Vec<u8>, G2gError> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preprocess-readback"),
            size: self.len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.len as u64);
        self.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        rx.recv()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let bytes = slice.get_mapped_range().to_vec();
        staging.unmap();
        Ok(bytes)
    }
}

impl WgpuBufferKeepAlive for WgpuBufferOwner {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// Owns a GPU-resident NV12 frame for surface-import into [`WgpuPreprocess`]
/// (M217): an R8Uint `wgpu::Texture` of size `width x (height * 3/2)` holding the
/// bytes in the standard NV12 layout (Y plane, then interleaved Cb,Cr), plus the
/// device / queue it lives on. Boxed as the [`WgpuKeepAlive`] of a
/// [`MemoryDomain::WgpuTexture`]; `WgpuPreprocess` downcasts to recover the
/// texture and adopt its device (a texture is bindable only on its own device),
/// so the NV12 pixels are sampled straight into the compute pass with no CPU
/// upload. A real GPU NV12 decoder is the intended producer; until one lands,
/// [`nv12_to_gpu_texture`] builds one from system bytes.
pub struct WgpuNv12Texture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    texture: wgpu::Texture,
    /// Optional drop guard whose `Drop` recycles the backing image (e.g. a
    /// `CudaWgpuPool` return handle from `CudaToWgpu`). Type-erased so this stays
    /// decoupled from the producer; `None` for non-pooled producers like
    /// `nv12_to_gpu_texture`. Held only to run its `Drop` when the frame releases.
    _recycle: Option<Box<dyn core::any::Any + Send + Sync>>,
}

impl core::fmt::Debug for WgpuNv12Texture {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WgpuNv12Texture")
            .field("texture", &self.texture)
            .field("pooled", &self._recycle.is_some())
            .finish()
    }
}

impl WgpuNv12Texture {
    /// Wrap an NV12 R8Uint texture with the device / queue it lives on.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, texture: wgpu::Texture) -> Self {
        Self { device, queue, texture, _recycle: None }
    }

    /// Like [`new`](Self::new), but carries a drop guard recycled when the frame
    /// is released (a pooled producer hands back a `CudaWgpuPool` return handle).
    pub fn with_recycle(
        device: wgpu::Device,
        queue: wgpu::Queue,
        texture: wgpu::Texture,
        recycle: Box<dyn core::any::Any + Send + Sync>,
    ) -> Self {
        Self { device, queue, texture, _recycle: Some(recycle) }
    }

    /// The backing NV12 texture, for the importer to sample directly.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// The device the texture lives on; the importer adopts it to bind the
    /// texture rather than uploading the frame to its own device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue paired with [`device`](Self::device).
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

impl WgpuKeepAlive for WgpuNv12Texture {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl AsyncElement for WgpuPreprocess {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // M304: also accept an already-RGB GPU texture (from the Android decode
        // path); fall back to the NV12 input otherwise.
        #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
        if let Ok(rgba) = upstream_caps.intersect(&Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }) {
            return Ok(rgba);
        }
        upstream_caps.intersect(&self.supported_input())
    }

    /// Native `DerivedOutput`: NV12 at fixed even geometry in, the matching
    /// `[1, 3, H, W]` f32 tensor out. Other input yields an empty set, so the
    /// solver rejects it at negotiation time.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } if w % 2 == 0 && h % 2 == 0 => CapsSet::one(Caps::Tensor {
                dtype: TensorDType::F32,
                shape: TensorShape::new([1, 3, *h, *w]),
                layout: TensorLayout::Nchw,
            }),
            // M304: already-RGB GPU texture input maps to the same NCHW tensor.
            #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } => CapsSet::one(Caps::Tensor {
                dtype: TensorDType::F32,
                shape: TensorShape::new([1, 3, *h, *w]),
                layout: TensorLayout::Nchw,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } if w % 2 == 0 && h % 2 == 0 => {
                self.width = *w;
                self.height = *h;
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            // M304: already-RGB GPU texture input (no chroma subsampling, so any
            // fixed geometry is fine).
            #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } => {
                self.width = *w;
                self.height = *h;
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let domain = match &frame.domain {
                        // System input: upload the NV12 bytes to a storage buffer
                        // and run the compute on the element's own device.
                        MemoryDomain::System(slice) => {
                            self.ensure_gpu().await?;
                            // GPU-output mode keeps the tensor on the device
                            // (M215); otherwise read it back to system memory.
                            if self.gpu_output {
                                MemoryDomain::WgpuBuffer(self.dispatch_gpu(slice.as_slice())?)
                            } else {
                                MemoryDomain::System(SystemSlice::from_boxed(
                                    self.dispatch(slice.as_slice())?,
                                ))
                            }
                        }
                        // Surface-import (M217): the NV12 frame is already a GPU
                        // texture. Adopt its device and sample it directly, no
                        // CPU upload. A foreign keep-alive we cannot bind.
                        MemoryDomain::WgpuTexture(owned) => {
                            let any = owned.keep_alive().as_any();
                            if let Some(owner) = any.downcast_ref::<WgpuNv12Texture>() {
                                self.ensure_tex_gpu(owner.device(), owner.queue());
                                self.dispatch_tex(owner)?
                            } else if let Some(domain) = self.try_dispatch_rgba(any)? {
                                // M304: already-RGB texture from the Android decode path.
                                domain
                            } else {
                                return Err(G2gError::UnsupportedDomain);
                            }
                        }
                        _ => return Err(G2gError::UnsupportedDomain),
                    };
                    let new_caps = self.tensor_caps();
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                        self.last_caps = Some(new_caps);
                    }
                    let tensor = Frame {
                        domain,
                        // preprocessing is per-frame: the tensor inherits the
                        // source timing so glass-to-glass latency stays traceable.
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(tensor)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // geometry is pinned at configure; a mid-stream change to
                    // anything but NV12 is a hard error.
                    c.intersect(&self.supported_input())?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is a timing marker: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // stateless per-frame conversion: nothing to drain.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Whether a wgpu adapter is available on this host. Tests skip gracefully
/// when no GPU is present, like the other hardware-gated elements.
pub async fn gpu_available() -> bool {
    wgpu::Instance::default()
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .is_ok()
}

/// Map any wgpu request/poll error to a structured hardware failure.
fn gpu_err<E>(_e: E) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

async fn build_gpu(width: u32, height: u32) -> Result<Gpu, G2gError> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map_err(gpu_err)?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(gpu_err)?;

    let area = width as usize * height as usize;
    let nv12_len = area * 3 / 2;
    let nv12_padded = nv12_len.div_ceil(4) * 4;
    let out_bytes = 3 * area * 4;

    let nv12_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nv12-in"),
        size: nv12_padded as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rgb-tensor-out"),
        size: out_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: out_bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dims"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut dims = [0u8; 16];
    dims[0..4].copy_from_slice(&width.to_le_bytes());
    dims[4..8].copy_from_slice(&height.to_le_bytes());
    queue.write_buffer(&dims_buf, 0, &dims);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nv12-rgb-normalize"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("nv12-rgb-normalize"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nv12-rgb-binding"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dims_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: nv12_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    Ok(Gpu {
        device,
        queue,
        pipeline,
        bind_group,
        nv12_buf,
        out_buf,
        staging,
        nv12_len,
        nv12_padded,
        out_bytes,
    })
}

/// Build the surface-import resources on an already-existing device (the one the
/// incoming NV12 texture lives on), M217. Unlike [`build_gpu`] it requests no
/// adapter / device: a texture is bindable only on its own device, so the
/// importer adopts the producer's rather than creating its own.
fn build_tex_gpu(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> TexGpu {
    let area = width as usize * height as usize;
    let out_bytes = 3 * area * 4;

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rgb-tensor-out"),
        size: out_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: out_bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dims"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut dims = [0u8; 16];
    dims[0..4].copy_from_slice(&width.to_le_bytes());
    dims[4..8].copy_from_slice(&height.to_le_bytes());
    queue.write_buffer(&dims_buf, 0, &dims);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nv12-tex-rgb-normalize"),
        source: wgpu::ShaderSource::Wgsl(TEX_SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("nv12-tex-rgb-normalize"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    TexGpu {
        device: device.clone(),
        queue: queue.clone(),
        pipeline,
        dims_buf,
        out_buf,
        staging,
        out_bytes,
    }
}

/// Build the RGBA surface-import resources (M304): same output buffers + dims as
/// [`build_tex_gpu`], but the `TEX_SHADER_RGBA` pipeline that samples an
/// `Rgba8Unorm` texture (no YCbCr math). The input texture is `width x height`.
#[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
fn build_tex_rgba_gpu(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> TexGpu {
    let area = width as usize * height as usize;
    let out_bytes = 3 * area * 4;

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rgb-tensor-out"),
        size: out_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: out_bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dims"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut dims = [0u8; 16];
    dims[0..4].copy_from_slice(&width.to_le_bytes());
    dims[4..8].copy_from_slice(&height.to_le_bytes());
    queue.write_buffer(&dims_buf, 0, &dims);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rgba-tex-tensor"),
        source: wgpu::ShaderSource::Wgsl(TEX_SHADER_RGBA.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rgba-tex-tensor"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    TexGpu { device: device.clone(), queue: queue.clone(), pipeline, dims_buf, out_buf, staging, out_bytes }
}

/// Stand-in for a GPU NV12 decoder until one lands (M217): upload NV12 system
/// bytes to a GPU R8Uint texture of size `width x (height * 3/2)` (the standard
/// NV12 byte layout) on a fresh wgpu device, and return it as the
/// `MemoryDomain::WgpuTexture` domain [`WgpuPreprocess`] surface-imports. A real
/// GPU decoder (`DmaBuf`/`D3D11Texture`/CUDA import) produces this domain
/// directly; this exists so the surface-import path is exercisable end-to-end.
pub async fn nv12_to_gpu_texture(
    nv12: &[u8],
    width: u32,
    height: u32,
) -> Result<MemoryDomain, G2gError> {
    if width % 2 != 0 || height % 2 != 0 {
        return Err(G2gError::CapsMismatch);
    }
    let tex_rows = height + height / 2;
    if nv12.len() < (width * tex_rows) as usize {
        return Err(G2gError::CapsMismatch);
    }

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map_err(gpu_err)?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(gpu_err)?;

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("nv12-surface"),
        size: wgpu::Extent3d { width, height: tex_rows, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // write_texture has no 256-byte bytes_per_row constraint (it stages
    // internally), so the unaligned NV12 width is fine.
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &nv12[..(width * tex_rows) as usize],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width),
            rows_per_image: Some(tex_rows),
        },
        wgpu::Extent3d { width, height: tex_rows, depth_or_array_layers: 1 },
    );

    let owner = WgpuNv12Texture::new(device, queue, texture);
    Ok(MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
        width,
        height,
        std::sync::Arc::new(owner),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_grayscale_is_linear_luma() {
        // neutral chroma (128,128) -> R=G=B = (Y-16)*1.164383/255
        let nv12 = [16u8, 235, 126, 100, 128, 128];
        let t = nv12_to_rgb_tensor(&nv12, 2, 2);
        // R, G, B planes are identical for grayscale.
        assert!((t[0] - 0.0).abs() < 1e-4, "Y=16 -> 0");
        assert!((t[1] - 1.0).abs() < 1e-4, "Y=235 -> 1");
        for plane in 0..3 {
            for px in 0..4 {
                assert!((t[plane * 4 + px] - t[px]).abs() < 1e-6, "grayscale planes equal");
            }
        }
    }

    #[test]
    fn intercept_narrows_nv12_and_rejects_rgba() {
        let e = WgpuPreprocess::new();
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        let rgba = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        assert!(e.intercept_caps(&nv12).is_ok());
        assert_eq!(e.intercept_caps(&rgba), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn configure_requires_even_nv12_geometry() {
        let mut e = WgpuPreprocess::new();
        let odd = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(3),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert_eq!(
            e.configure_pipeline(&odd).err(),
            Some(G2gError::CapsMismatch),
            "4:2:0 needs even dims"
        );
        assert!(!e.configured);
    }
}

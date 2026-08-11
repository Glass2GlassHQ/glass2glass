//! Inline GPU tensor preprocessing via wgpu compute (DESIGN.md §5.1).
//!
//! `WgpuPreprocess` is the hardware-first preprocessing pillar: an
//! `AsyncElement` that takes an NV12 or packed-YUYV video frame and emits a
//! normalized f32 NCHW RGB tensor (`Caps::RawVideo -> Caps::Tensor{F32,
//! [1,3,H,W],Nchw}`), doing the BT.601 colour conversion and the `value / 255`
//! normalization in a wgpu compute shader rather than on the CPU. It produces the
//! same tensor contract `OrtInference` builds on the CPU, so it composes with the
//! existing tensor graph (`-> TensorBatcher -> inference -> TensorPostprocess`).
//! YUYV is what a UVC webcam captures, so a camera reaches the tensor with no
//! `videoconvert` in front.
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
//! - **Input (M990, dma-buf import, `dmabuf-wgpu` feature, Linux):** a
//!   `MemoryDomain::DmaBuf` frame from a capture / decode path is imported
//!   with Vulkan external memory into a buffer aliasing the same pixels and bound
//!   into the compute pass. The frame's row stride and plane offset reach the
//!   shader in the dims uniform, so a padded capture buffer needs no repack. A
//!   webcam's capture buffer is CPU-backed, which only an integrated GPU can bind,
//!   so [`with_import_adapter`](WgpuPreprocess::with_import_adapter) picks the GPU
//!   the import opens on (M993). Windows D3D11 surface import is the remaining
//!   input path.
//!
//! With both ends GPU-resident, `capture / decode -> WgpuPreprocess ->
//! WgpuInference` runs with the pixels never touching the CPU.
//! [`nv12_to_gpu_texture`] builds a GPU-texture frame from system bytes for the
//! surface-import path when no GPU producer is in the graph. RGBA input (normalize
//! only, no colour convert) is a small follow-up.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, HardwareError,
    MemoryDomain, OutputSink, OwnedWgpuBuffer, OwnedWgpuTexture, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, RawVideoFormat, TensorDType, TensorLayout,
    TensorShape, WgpuBufferKeepAlive,
};
// The GPU-resident NV12 frame owner lives with the interop code that produces it
// (the CUDA and dma-buf bridges); re-exported so this module's consumers keep
// naming it here.
pub use g2g_plugins::gpu::WgpuNv12Texture;

/// 8x8 invocations per workgroup; the dispatch covers ceil(W/8) x ceil(H/8).
const WORKGROUP: u32 = 8;

/// YUV bytes -> normalized planar RGB (BT.601 limited range), in a compute pass.
/// The frame bytes arrive as a packed `array<u32>`; `out` is the f32 NCHW tensor
/// (R plane, then G, then B), each value in `[0, 1]`.
///
/// `$sample` is the only part that differs per pixel format: it reads `yv`, `cb`,
/// `cr` for pixel `(x, y)`, whose row starts at byte `row`. Everything else, the
/// bindings and the colour math, is shared, hence a macro rather than a const:
/// `concat!` splices the sample step in at compile time.
///
/// `dims.stride` / `dims.base` locate the pixels inside the input buffer: a
/// system-memory frame is tightly packed from byte 0 (`stride` = the format's tight
/// row bytes, `base == 0`), an imported dma-buf can have a padded row stride and a
/// nonzero plane offset (M990). The output tensor is always tightly packed.
macro_rules! yuv_shader {
    ($sample:expr) => {
        concat!(
            r#"
struct Dims { width: u32, height: u32, stride: u32, base: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> pixels: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn load_byte(i: u32) -> f32 {
    let word = pixels[i / 4u];
    let shift = (i % 4u) * 8u;
    return f32((word >> shift) & 0xFFu);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = dims.width;
    let h = dims.height;
    let stride = dims.stride;
    if (x >= w || y >= h) { return; }
    let row = dims.base + y * stride;
"#,
            $sample,
            r#"
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
"#
        )
    };
}

/// NV12 input: `h` luma rows, then interleaved Cb,Cr rows at half height, all at
/// the same row stride.
const SHADER: &str = yuv_shader!(
    r#"
    let yv = load_byte(row + x);
    let uv_index = dims.base + stride * h + (y / 2u) * stride + (x / 2u) * 2u;
    let cb = load_byte(uv_index) - 128.0;
    let cr = load_byte(uv_index + 1u) - 128.0;
"#
);

/// Packed YUYV (4:2:2) input, what a UVC webcam captures: four bytes
/// `Y0 Cb Y1 Cr` per pixel pair, one row after another, so a pixel's chroma is
/// its pair's and only the luma byte differs between the pair's two pixels.
const YUYV_SHADER: &str = yuv_shader!(
    r#"
    let pair = row + (x / 2u) * 4u;
    let yv = load_byte(pair + (x % 2u) * 2u);
    let cb = load_byte(pair + 1u) - 128.0;
    let cr = load_byte(pair + 3u) - 128.0;
"#
);

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

/// One pixel's BT.601 limited-range conversion, the host mirror of the shaders'
/// colour math (`cb` / `cr` already centred on 0). Returns normalized R, G, B.
fn bt601_rgb(yv: f32, cb: f32, cr: f32) -> [f32; 3] {
    let yy = (yv - 16.0) * 1.164383;
    [
        (yy + 1.596027 * cr).clamp(0.0, 255.0) / 255.0,
        (yy - 0.391762 * cb - 0.812968 * cr).clamp(0.0, 255.0) / 255.0,
        (yy + 2.017232 * cb).clamp(0.0, 255.0) / 255.0,
    ]
}

/// Write one pixel's RGB into the NCHW tensor's three planes.
fn write_pixel(out: &mut [f32], area: usize, index: usize, rgb: [f32; 3]) {
    out[index] = rgb[0];
    out[area + index] = rgb[1];
    out[2 * area + index] = rgb[2];
}

/// The host BT.601 reference matching `SHADER`, kept public so the test (and a
/// CPU-fallback caller) can compare against the GPU output. Returns the f32
/// NCHW RGB tensor for one tightly-packed NV12 frame.
pub fn nv12_to_rgb_tensor(nv12: &[u8], width: usize, height: usize) -> Vec<f32> {
    let area = width * height;
    let uv_base = area;
    let byte = |i: usize| nv12[i] as f32;
    let mut out = vec![0f32; 3 * area];
    for y in 0..height {
        for x in 0..width {
            let li = y * width + x;
            let uvi = uv_base + (y / 2) * width + (x / 2) * 2;
            let rgb = bt601_rgb(byte(li), byte(uvi) - 128.0, byte(uvi + 1) - 128.0);
            write_pixel(&mut out, area, li, rgb);
        }
    }
    out
}

/// The host reference matching `YUYV_SHADER`, the packed-4:2:2 counterpart of
/// [`nv12_to_rgb_tensor`]: one tightly-packed YUYV frame (`Y0 Cb Y1 Cr` per pixel
/// pair, `2 * width` bytes per row) to the same f32 NCHW RGB tensor.
pub fn yuyv_to_rgb_tensor(yuyv: &[u8], width: usize, height: usize) -> Vec<f32> {
    let area = width * height;
    let byte = |i: usize| yuyv[i] as f32;
    let mut out = vec![0f32; 3 * area];
    for y in 0..height {
        for x in 0..width {
            let li = y * width + x;
            let pair = y * width * 2 + (x / 2) * 4;
            let rgb = bt601_rgb(
                byte(pair + (x % 2) * 2),
                byte(pair + 1) - 128.0,
                byte(pair + 3) - 128.0,
            );
            write_pixel(&mut out, area, li, rgb);
        }
    }
    out
}

/// The raw formats the element reads, in preference order: NV12 and packed YUYV
/// through a compute shader, plus already-converted RGBA where a GPU decoder hands
/// it over as a texture (M304).
fn input_formats() -> &'static [RawVideoFormat] {
    #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
    return &[
        RawVideoFormat::Rgba8,
        RawVideoFormat::Nv12,
        RawVideoFormat::Yuyv,
    ];
    #[cfg(not(all(target_os = "android", feature = "mediacodec-wgpu")))]
    return &[RawVideoFormat::Nv12, RawVideoFormat::Yuyv];
}

/// `format` at any geometry: what the element advertises and intersects upstream
/// caps against.
fn any_geometry(format: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
    }
}

/// Whether the compute pass can read `format` at this geometry: 4:2:0 chroma
/// needs both dimensions even, packed 4:2:2 only an even width.
fn geometry_ok(format: RawVideoFormat, width: u32, height: u32) -> bool {
    match format {
        RawVideoFormat::Nv12 => width.is_multiple_of(2) && height.is_multiple_of(2),
        RawVideoFormat::Yuyv => width.is_multiple_of(2),
        #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
        RawVideoFormat::Rgba8 => true,
        _ => false,
    }
}

/// The storage-buffer compute shader that reads `format`'s bytes. `None` for a
/// format with no such path (RGBA input arrives as a texture, never as bytes).
fn shader_for(format: RawVideoFormat) -> Option<&'static str> {
    match format {
        RawVideoFormat::Nv12 => Some(SHADER),
        RawVideoFormat::Yuyv => Some(YUYV_SHADER),
        _ => None,
    }
}

/// GPU resources sized to a fixed `W x H` and pixel format, built lazily on the
/// first frame.
#[derive(Debug)]
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    input_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    input_len: usize,
    input_padded: usize,
    out_bytes: usize,
}

/// GPU resources for an import path: the pipeline and the output buffers, with no
/// input buffer of their own, because the input arrives with the frame and is
/// bound per dispatch. Built lazily on the first such frame, on the device that
/// frame's memory lives on (a texture or imported buffer is bindable only on its
/// own device), so the bind group is rebuilt per dispatch.
///
/// Serves the GPU-texture surface-import (M217) and the dma-buf import (M990).
#[derive(Debug)]
struct ImportGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    dims_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    out_bytes: usize,
}

/// # Example
///
/// ```no_run
/// use g2g_ml::wgpupreprocess::WgpuPreprocess;
///
/// let preprocess = WgpuPreprocess::new().with_gpu_output();
/// ```
#[derive(Debug)]
pub struct WgpuPreprocess {
    width: u32,
    height: u32,
    /// Input pixel format from the negotiated caps: which compute shader runs and
    /// how many bytes a frame is.
    format: RawVideoFormat,
    configured: bool,
    gpu: Option<Gpu>,
    /// Surface-import resources, built on the first GPU-texture frame from that
    /// frame's device (M217). Separate from `gpu` because the texture path binds
    /// a sampled texture, not a storage buffer, and adopts the producer's device.
    tex_gpu: Option<ImportGpu>,
    /// RGBA surface-import resources (M304), built on the first RGBA GPU-texture
    /// frame. Separate pipeline from `tex_gpu` (samples `texture_2d<f32>`, no
    /// YCbCr math); the input is already-converted RGBA from `MediaCodecDec`.
    #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
    tex_rgba_gpu: Option<ImportGpu>,
    /// dma-buf import resources (M990), built on the first dma-buf frame. Separate
    /// from `gpu` because the import needs a device carrying the Vulkan
    /// external-memory extensions, which the element's own device does not have.
    #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
    dmabuf_gpu: Option<ImportGpu>,
    /// The dma-buf import itself, holding the producer-sync state it keeps across
    /// frames (M990).
    #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
    dmabuf_importer: g2g_plugins::dmabufwgpu::DmaBufImporter,
    /// Which GPU the dma-buf import opens on (M993): a webcam's CPU-backed
    /// capture buffer binds only on an integrated one.
    #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
    import_adapter: g2g_plugins::dmabufwgpu::ImportAdapter,
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
            format: RawVideoFormat::Nv12,
            configured: false,
            gpu: None,
            tex_gpu: None,
            #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
            tex_rgba_gpu: None,
            #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
            dmabuf_gpu: None,
            #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
            dmabuf_importer: g2g_plugins::dmabufwgpu::DmaBufImporter::new(),
            #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
            import_adapter: g2g_plugins::dmabufwgpu::ImportAdapter::default(),
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

    /// Import dma-buf frames on a chosen GPU instead of the most capable one
    /// (M993): a capture buffer from a USB webcam is CPU-backed, which a discrete
    /// GPU refuses to bind and an integrated one accepts. Same knob as the
    /// `import-adapter` property; the default is unchanged, which is what a
    /// GPU-exported dma-buf needs.
    #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
    pub fn with_import_adapter(mut self, adapter: g2g_plugins::dmabufwgpu::ImportAdapter) -> Self {
        self.import_adapter = adapter;
        self
    }

    /// Count of tensor `DataFrame`s pushed downstream. Useful in tests.
    pub fn emitted(&self) -> u64 {
        self.emitted
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
        self.gpu = Some(build_gpu(self.width, self.height, self.format).await?);
        Ok(())
    }

    /// Upload the frame, run the compute pass, and read the f32 tensor
    /// back as little-endian bytes (the `OrtInference` output byte format).
    /// Blocks the calling task on `poll(Wait)`; offloading the GPU round-trip
    /// to a blocking pool is a follow-up.
    fn dispatch(&self, pixels: &[u8]) -> Result<Box<[u8]>, G2gError> {
        let gpu = self.gpu.as_ref().ok_or(G2gError::NotConfigured)?;
        if pixels.len() < gpu.input_len {
            return Err(G2gError::CapsMismatch);
        }
        let mut padded = vec![0u8; gpu.input_padded];
        padded[..gpu.input_len].copy_from_slice(&pixels[..gpu.input_len]);
        gpu.queue.write_buffer(&gpu.input_buf, 0, &padded);

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
    fn dispatch_gpu(&self, pixels: &[u8]) -> Result<OwnedWgpuBuffer, G2gError> {
        let gpu = self.gpu.as_ref().ok_or(G2gError::NotConfigured)?;
        if pixels.len() < gpu.input_len {
            return Err(G2gError::CapsMismatch);
        }
        let mut padded = vec![0u8; gpu.input_padded];
        padded[..gpu.input_len].copy_from_slice(&pixels[..gpu.input_len]);
        gpu.queue.write_buffer(&gpu.input_buf, 0, &padded);

        let frame_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preprocess-tensor"),
            size: gpu.out_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
        // On-device copy into the per-frame buffer; no read-back to the CPU.
        encoder.copy_buffer_to_buffer(&gpu.out_buf, 0, &frame_buf, 0, gpu.out_bytes as u64);
        gpu.queue.submit([encoder.finish()]);

        let owner = WgpuBufferOwner::new(
            gpu.device.clone(),
            gpu.queue.clone(),
            frame_buf,
            gpu.out_bytes,
        );
        Ok(OwnedWgpuBuffer::new(
            gpu.out_bytes,
            std::sync::Arc::new(owner),
        ))
    }

    /// Build the surface-import pipeline and output buffers on `device` (M217).
    /// Idempotent: built once, on the first GPU-texture frame, because the
    /// device is only known once such a frame arrives.
    fn ensure_tex_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.tex_gpu.is_some() {
            return;
        }
        self.tex_gpu = Some(build_import_gpu(
            device,
            queue,
            self.width,
            self.height,
            TEX_SHADER,
            "nv12-tex-rgb-normalize",
        ));
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

        let mut encoder = tg
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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

        finish_import(tg, encoder, self.gpu_output)
    }

    /// Try to consume the GPU texture as an already-RGB `WgpuRgbaTexture` (the
    /// M304 Android decode path). Returns `Ok(None)` if the keep-alive is not one
    /// (so the caller falls through to `UnsupportedDomain`).
    #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
    fn try_dispatch_rgba(
        &mut self,
        any: &dyn core::any::Any,
    ) -> Result<Option<MemoryDomain>, G2gError> {
        let Some(owner) = any.downcast_ref::<g2g_plugins::mediacodec_wgpu::WgpuRgbaTexture>()
        else {
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
        self.tex_rgba_gpu = Some(build_import_gpu(
            device,
            queue,
            self.width,
            self.height,
            TEX_SHADER_RGBA,
            "rgba-tex-tensor",
        ));
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

        let mut encoder = tg
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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

        finish_import(tg, encoder, self.gpu_output)
    }

    /// dma-buf import dispatch (M990): import the frame's dma-buf as a Vulkan
    /// buffer aliasing the same memory and bind it straight into the compute
    /// pass, no CPU upload. The frame's row stride and plane offset go to the
    /// shader in the dims uniform, so a padded capture buffer needs no repack.
    /// Returns `UnsupportedDomain` when the driver cannot bind the fd (a CPU-backed
    /// dma-buf on a discrete GPU: see `with_import_adapter`).
    #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
    async fn dispatch_dmabuf(
        &mut self,
        dmabuf: &g2g_core::memory::OwnedDmaBuf,
    ) -> Result<MemoryDomain, G2gError> {
        use g2g_plugins::dmabufwgpu::create_import_device_on;

        let shader = shader_for(self.format).ok_or(G2gError::CapsMismatch)?;
        let plane_bytes = self
            .format
            .frame_bytes(u64::from(dmabuf.stride), u64::from(self.height))
            .ok_or(G2gError::CapsMismatch)?;
        // The shader reads the plane as `array<u32>`, so the binding must be a
        // whole number of words. dma-buf memory is page granular, so rounding up
        // stays inside the allocation. The stride and offset come from the
        // producer, so fold them with checked ops.
        let size = u64::from(dmabuf.offset)
            .checked_add(plane_bytes)
            .and_then(|s| s.checked_next_multiple_of(4))
            .ok_or(G2gError::CapsMismatch)?;
        if size == 0 {
            return Err(G2gError::CapsMismatch);
        }
        if self.dmabuf_gpu.is_none() {
            let (device, queue) = create_import_device_on(self.import_adapter).await?;
            self.dmabuf_gpu = Some(build_import_gpu(
                &device,
                &queue,
                self.width,
                self.height,
                shader,
                "dmabuf-rgb-normalize",
            ));
        }
        let device = self
            .dmabuf_gpu
            .as_ref()
            .ok_or(G2gError::NotConfigured)?
            .device
            .clone();
        let input = self.dmabuf_importer.import(&device, dmabuf, size).await?;
        self.run_dmabuf_pass(&input, dmabuf.stride, dmabuf.offset)
    }

    /// Run the compute pass over an imported dma-buf buffer. Split from
    /// [`dispatch_dmabuf`](Self::dispatch_dmabuf) so the import (which needs
    /// `&mut self` for the cached producer semaphore) is done before the resources
    /// are borrowed.
    #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
    fn run_dmabuf_pass(
        &self,
        input: &wgpu::Buffer,
        stride: u32,
        base: u32,
    ) -> Result<MemoryDomain, G2gError> {
        let ig = self.dmabuf_gpu.as_ref().ok_or(G2gError::NotConfigured)?;
        ig.queue.write_buffer(
            &ig.dims_buf,
            0,
            &dims_bytes(self.width, self.height, stride, base),
        );
        let layout = ig.pipeline.get_bind_group_layout(0);
        let bind_group = ig.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dmabuf-binding"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ig.dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ig.out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ig
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dmabuf->rgb"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&ig.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let gx = self.width.div_ceil(WORKGROUP);
            let gy = self.height.div_ceil(WORKGROUP);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        finish_import(ig, encoder, self.gpu_output)
    }
}

/// Finish an import dispatch: submit `encoder` and return the tensor, either
/// GPU-resident in a fresh per-frame buffer (`gpu_output`, so the next frame's
/// compute cannot clobber one still in flight downstream) or read back to system
/// memory. Shared by every import path, whose only difference is how the input
/// was bound.
fn finish_import(
    ig: &ImportGpu,
    mut encoder: wgpu::CommandEncoder,
    gpu_output: bool,
) -> Result<MemoryDomain, G2gError> {
    if gpu_output {
        let frame_buf = ig.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preprocess-tensor"),
            size: ig.out_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&ig.out_buf, 0, &frame_buf, 0, ig.out_bytes as u64);
        ig.queue.submit([encoder.finish()]);
        let owner =
            WgpuBufferOwner::new(ig.device.clone(), ig.queue.clone(), frame_buf, ig.out_bytes);
        return Ok(MemoryDomain::WgpuBuffer(OwnedWgpuBuffer::new(
            ig.out_bytes,
            std::sync::Arc::new(owner),
        )));
    }
    encoder.copy_buffer_to_buffer(&ig.out_buf, 0, &ig.staging, 0, ig.out_bytes as u64);
    ig.queue.submit([encoder.finish()]);
    let slice = ig.staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ig.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    rx.recv()
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    let bytes = slice.get_mapped_range().to_vec().into_boxed_slice();
    ig.staging.unmap();
    Ok(MemoryDomain::System(SystemSlice::from_boxed(bytes)))
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
        Self {
            device,
            queue,
            buffer,
            len,
        }
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.len as u64);
        self.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
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
        Ok(bytes)
    }
}

impl WgpuBufferKeepAlive for WgpuBufferOwner {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl AsyncElement for WgpuPreprocess {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        input_formats()
            .iter()
            .find_map(|format| upstream_caps.intersect(&any_geometry(*format)).ok())
            .ok_or(G2gError::CapsMismatch)
    }

    /// Native `DerivedOutput`: a readable format at fixed geometry in, the matching
    /// `[1, 3, H, W]` f32 tensor out. Other input yields an empty set, so the
    /// solver rejects it at negotiation time.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo {
                format,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } if geometry_ok(*format, *w, *h) => CapsSet::one(Caps::Tensor {
                dtype: TensorDType::F32,
                shape: TensorShape::new([1, 3, *h, *w]),
                layout: TensorLayout::Nchw,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !geometry_ok(*format, *w, *h) {
            return Err(G2gError::CapsMismatch);
        }
        self.width = *w;
        self.height = *h;
        self.format = *format;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        WGPU_PREPROCESS_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "gpu-output" => {
                self.gpu_output = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
            "import-adapter" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                self.import_adapter = g2g_plugins::dmabufwgpu::ImportAdapter::from_name(text)
                    .ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "gpu-output" => Some(PropValue::Bool(self.gpu_output)),
            #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
            "import-adapter" => Some(PropValue::Str(self.import_adapter.name().into())),
            _ => None,
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
                        // dma-buf import (M990): the NV12 frame is a dma-buf from a
                        // capture / decode path. Import it into a Vulkan buffer that
                        // aliases the same memory and bind that, no CPU upload.
                        #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
                        MemoryDomain::DmaBuf(dmabuf) => self.dispatch_dmabuf(dmabuf).await?,
                        _ => return Err(G2gError::UnsupportedDomain),
                    };
                    let new_caps = self.tensor_caps();
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
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
                    // geometry and format are pinned at configure; a mid-stream
                    // change to another format is a hard error.
                    c.intersect(&any_geometry(self.format))?;
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

/// Whether the tensor stays GPU-resident, so a `gst-launch` line can pick the
/// keep-on-GPU path without the builder.
const GPU_OUTPUT_PROP: PropertySpec = PropertySpec::new(
    "gpu-output",
    PropKind::Bool,
    "emit the tensor as a GPU buffer instead of reading it back to system memory",
);

#[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
static WGPU_PREPROCESS_PROPS: &[PropertySpec] = &[
    GPU_OUTPUT_PROP,
    g2g_plugins::dmabufwgpu::IMPORT_ADAPTER_PROP,
];

#[cfg(not(all(target_os = "linux", feature = "dmabuf-wgpu")))]
static WGPU_PREPROCESS_PROPS: &[PropertySpec] = &[GPU_OUTPUT_PROP];

/// Every readable raw format at any geometry in; no source template, because the
/// output tensor's shape follows the negotiated input geometry.
#[cfg(feature = "launch")]
impl g2g_core::PadTemplates for WgpuPreprocess {
    fn pad_templates() -> Vec<g2g_core::PadTemplate> {
        let formats = input_formats().iter().copied().map(any_geometry).collect();
        Vec::from([g2g_core::PadTemplate::sink(CapsSet::from_alternatives(
            formats,
        ))])
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

async fn build_gpu(width: u32, height: u32, format: RawVideoFormat) -> Result<Gpu, G2gError> {
    let source = shader_for(format).ok_or(G2gError::CapsMismatch)?;
    let stride = format.row_stride(width).ok_or(G2gError::CapsMismatch)?;
    let input_len = format
        .frame_bytes(u64::from(stride), u64::from(height))
        .ok_or(G2gError::CapsMismatch)? as usize;

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
    // The shader reads the frame as `array<u32>`, so the buffer is a whole
    // number of words.
    let input_padded = input_len.div_ceil(4) * 4;
    let out_bytes = 3 * area * 4;

    let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixels-in"),
        size: input_padded as u64,
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
    // System memory is tightly packed from byte 0.
    queue.write_buffer(&dims_buf, 0, &dims_bytes(width, height, stride, 0));

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("yuv-rgb-normalize"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("yuv-rgb-normalize"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("yuv-rgb-binding"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dims_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_buf.as_entire_binding(),
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
        input_buf,
        out_buf,
        staging,
        input_len,
        input_padded,
        out_bytes,
    })
}

/// The 16-byte `Dims` uniform every pipeline binds: the frame geometry, plus
/// where the input pixels sit in their buffer (`stride` = input row stride,
/// `base` = byte offset of the first luma byte). The texture pipelines read only
/// the geometry.
fn dims_bytes(width: u32, height: u32, stride: u32, base: u32) -> [u8; 16] {
    let mut dims = [0u8; 16];
    dims[0..4].copy_from_slice(&width.to_le_bytes());
    dims[4..8].copy_from_slice(&height.to_le_bytes());
    dims[8..12].copy_from_slice(&stride.to_le_bytes());
    dims[12..16].copy_from_slice(&base.to_le_bytes());
    dims
}

/// Build the import resources on an already-existing device: the one the incoming
/// frame's GPU memory lives on (a texture) or the one that can import its fd (a
/// dma-buf). Unlike [`build_gpu`] it requests no adapter / device, and allocates
/// no input buffer, because the input arrives with the frame. `shader` picks what
/// that input is: `TEX_SHADER` for an NV12 texture (M217), the imported dma-buf
/// buffer's format shader (M990, see [`shader_for`]), `TEX_SHADER_RGBA` for an
/// already-RGB texture (M304).
fn build_import_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    shader: &str,
    label: &str,
) -> ImportGpu {
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
    // A dma-buf frame rewrites this per dispatch with its own stride / offset.
    queue.write_buffer(&dims_buf, 0, &dims_bytes(width, height, width, 0));

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    ImportGpu {
        device: device.clone(),
        queue: queue.clone(),
        pipeline,
        dims_buf,
        out_buf,
        staging,
        out_bytes,
    }
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
    if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
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
        size: wgpu::Extent3d {
            width,
            height: tex_rows,
            depth_or_array_layers: 1,
        },
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
        wgpu::Extent3d {
            width,
            height: tex_rows,
            depth_or_array_layers: 1,
        },
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
                assert!(
                    (t[plane * 4 + px] - t[px]).abs() < 1e-6,
                    "grayscale planes equal"
                );
            }
        }
    }

    #[test]
    fn yuyv_reference_grayscale_is_linear_luma() {
        // one pixel pair with neutral chroma: Y0=16 -> 0, Y1=235 -> 1, R=G=B.
        let yuyv = [16u8, 128, 235, 128];
        let t = yuyv_to_rgb_tensor(&yuyv, 2, 1);
        assert!((t[0] - 0.0).abs() < 1e-4, "Y=16 -> 0");
        assert!((t[1] - 1.0).abs() < 1e-4, "Y=235 -> 1");
        for plane in 0..3 {
            for px in 0..2 {
                assert!(
                    (t[plane * 2 + px] - t[px]).abs() < 1e-6,
                    "grayscale planes equal"
                );
            }
        }
        // The same picture as NV12 gives the same tensor, so the two references
        // agree on the colour math and only differ in where the bytes sit.
        let nv12 = [16u8, 235, 128, 128];
        assert_eq!(t, nv12_to_rgb_tensor(&nv12, 2, 1));
    }

    #[test]
    fn intercept_narrows_the_readable_formats_and_rejects_others() {
        let e = WgpuPreprocess::new();
        let raw = |format| Caps::RawVideo {
            format,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        };
        assert_eq!(
            e.intercept_caps(&raw(RawVideoFormat::Nv12)),
            Ok(raw(RawVideoFormat::Nv12))
        );
        assert_eq!(
            e.intercept_caps(&raw(RawVideoFormat::Yuyv)),
            Ok(raw(RawVideoFormat::Yuyv))
        );
        assert_eq!(
            e.intercept_caps(&raw(RawVideoFormat::Rgba8)),
            Err(G2gError::CapsMismatch)
        );
        assert_eq!(
            e.intercept_caps(&raw(RawVideoFormat::I420)),
            Err(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn configure_takes_yuyv_at_odd_height_but_not_odd_width() {
        let yuyv = |w, h| Caps::RawVideo {
            format: RawVideoFormat::Yuyv,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        };
        let mut e = WgpuPreprocess::new();
        // 4:2:2 subsamples horizontally only, so an odd row count is fine.
        assert!(e.configure_pipeline(&yuyv(640, 481)).is_ok());
        assert_eq!(e.format, RawVideoFormat::Yuyv);
        assert_eq!(
            WgpuPreprocess::new()
                .configure_pipeline(&yuyv(641, 480))
                .err(),
            Some(G2gError::CapsMismatch),
            "an odd width has no complete pixel pair"
        );
    }

    /// M993: the import-adapter knob is settable by name, defaults to the choice a
    /// GPU-exported dma-buf needs, and refuses an unknown value.
    #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
    #[test]
    fn import_adapter_property_round_trips() {
        let mut e = WgpuPreprocess::new();
        assert!(WGPU_PREPROCESS_PROPS
            .iter()
            .any(|p| p.name == "import-adapter"));
        assert_eq!(
            e.get_property("import-adapter"),
            Some(PropValue::Str("high-performance".into()))
        );
        e.set_property("import-adapter", PropValue::Str("integrated".into()))
            .expect("integrated is a valid choice");
        assert_eq!(
            e.import_adapter,
            g2g_plugins::dmabufwgpu::ImportAdapter::Integrated
        );
        assert_eq!(
            e.set_property("import-adapter", PropValue::Str("magic".into())),
            Err(PropError::Value)
        );
    }

    #[test]
    fn configure_requires_even_nv12_geometry() {
        let mut e = WgpuPreprocess::new();
        let odd = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(3),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        };
        assert_eq!(
            e.configure_pipeline(&odd).err(),
            Some(G2gError::CapsMismatch),
            "4:2:0 needs even dims"
        );
        assert!(!e.configured);
    }
}

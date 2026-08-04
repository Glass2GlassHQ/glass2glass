//! wgpu compute compositor (M853): the GPU sibling of the CPU
//! [`Compositor`](crate::compositor) for HD / many-input scale. Same
//! N-in-1-out [`MultiInputElement`] surface, same [`CompositorPad`] semantics
//! (position, z-order, per-pad alpha, optional bilinear resize) and the same
//! latest-wins overlay cadence; only the pixel work moves to a compute shader.
//!
//! RGBA8 only (the planar YUV mixing stays on the CPU element). Input frames
//! arrive in system memory and are uploaded into one packed storage buffer; an
//! overlay is re-uploaded only when a new frame lands, so a slow overlay costs
//! nothing per output frame.
//!
//! **One dispatch per output frame.** Every pad is bound at once and each
//! invocation walks them in paint order for its own output pixel, so ordered
//! source-over needs no inter-dispatch barrier and the cost is one pass however
//! many inputs there are.
//!
//! **Bit-exact with the CPU compositor.** The shader repeats the integer blend
//! (`paint::blend_px`) and the Q16 bilinear mapping in `u32`/`i32`, not float,
//! so a GPU frame matches [`Compositor::compose`](crate::compositor::Compositor)
//! byte for byte. The fixed-point mapping stays inside `u32` for dimensions up
//! to [`MAX_DIM`], which `configure_pipeline` enforces.
//!
//! Output is [`MemoryDomain::System`] by default (feeds any CPU sink); with
//! [`with_gpu_output`](WgpuCompositor::with_gpu_output) the composite stays on
//! the device as a [`MemoryDomain::WgpuTexture`] for
//! [`WgpuSink`](crate::wgpusink) or an embedder's own render graph. Share the
//! consumer's [`GpuContext`] via [`with_context`](WgpuCompositor::with_context)
//! so that handoff is copy-free.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use crate::compositor::{paint_order, CompositorPad, CompositorState};
use crate::gpu::{gpu_err, GpuContext, WgpuTextureKeepAlive};
use g2g_core::frame::Frame;
use g2g_core::memory::{OwnedWgpuTexture, SystemSlice};
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain,
    MultiInputElement, OutputSink, PipelinePacket, Rate, RawVideoFormat,
};

/// Largest input / canvas edge the shader's fixed-point resize stays exact for
/// (the `(2*d+1) * s * 32768 / dst` mapping must not overflow `u32`).
pub const MAX_DIM: u32 = 16384;

const WORKGROUP: u32 = 8;

/// Compositing compute shader. `pads` arrives pre-sorted in paint order and
/// carries each input's placement plus its offset into the packed `src` buffer;
/// one invocation owns one output pixel and blends every pad covering it.
const SHADER: &str = r#"
struct Params {
    out_w: u32,
    out_h: u32,
    pad_count: u32,
    bg: u32,
    stride_words: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct Pad {
    x0: i32,
    y0: i32,
    dw: u32,
    dh: u32,
    sw: u32,
    sh: u32,
    alpha: u32,
    off: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pads: array<Pad>;
@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

fn unpack(v: u32) -> vec4<u32> {
    return vec4<u32>(v & 0xffu, (v >> 8u) & 0xffu, (v >> 16u) & 0xffu, (v >> 24u) & 0xffu);
}

fn pack(v: vec4<u32>) -> u32 {
    return v.x | (v.y << 8u) | (v.z << 16u) | (v.w << 24u);
}

// integer source-over, the exact arithmetic of paint::blend_px.
fn blend(d: vec4<u32>, s: vec4<u32>, galpha: u32) -> vec4<u32> {
    let a = (s.w * galpha + 127u) / 255u;
    let inv = 255u - a;
    return vec4<u32>(
        (s.x * a + d.x * inv + 127u) / 255u,
        (s.y * a + d.y * inv + 127u) / 255u,
        (s.z * a + d.z * inv + 127u) / 255u,
        a + d.w * inv / 255u,
    );
}

// center-aligned Q16 source coordinate: ((d + 0.5) * s / dst - 0.5), clamped.
// Split around the division so the numerator never leaves u32.
fn map_q16(d: u32, s: u32, dst: u32, hi: i32) -> i32 {
    let n = (2u * d + 1u) * s;
    let q = (n / dst) * 32768u + ((n % dst) * 32768u) / dst;
    return clamp(i32(q) - 32768, 0, hi);
}

fn sample_scaled(p: Pad, dx: u32, dy: u32) -> vec4<u32> {
    let fx = map_q16(dx, p.sw, p.dw, i32((p.sw - 1u) << 16u));
    let fy = map_q16(dy, p.sh, p.dh, i32((p.sh - 1u) << 16u));
    let x0 = u32(fx >> 16);
    let y0 = u32(fy >> 16);
    let x1 = min(x0 + 1u, p.sw - 1u);
    let y1 = min(y0 + 1u, p.sh - 1u);
    let tx = u32((fx >> 8) & 0xff);
    let ty = u32((fy >> 8) & 0xff);
    let s00 = unpack(src[p.off + y0 * p.sw + x0]);
    let s01 = unpack(src[p.off + y0 * p.sw + x1]);
    let s10 = unpack(src[p.off + y1 * p.sw + x0]);
    let s11 = unpack(src[p.off + y1 * p.sw + x1]);
    let top = s00 * (256u - tx) + s01 * tx;
    let bot = s10 * (256u - tx) + s11 * tx;
    return (top * (256u - ty) + bot * ty) >> vec4<u32>(16u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.out_w || gid.y >= params.out_h) {
        return;
    }
    var acc = unpack(params.bg);
    for (var i = 0u; i < params.pad_count; i = i + 1u) {
        let p = pads[i];
        let dx = i32(gid.x) - p.x0;
        let dy = i32(gid.y) - p.y0;
        if (dx < 0 || dy < 0 || dx >= i32(p.dw) || dy >= i32(p.dh)) {
            continue;
        }
        var px: vec4<u32>;
        if (p.dw == p.sw && p.dh == p.sh) {
            px = unpack(src[p.off + u32(dy) * p.sw + u32(dx)]);
        } else {
            px = sample_scaled(p, u32(dx), u32(dy));
        }
        acc = blend(acc, px, p.alpha);
    }
    dst[gid.y * params.stride_words + gid.x] = pack(acc);
}
"#;

/// One pad as the shader reads it: placement on the canvas, source geometry, and
/// where that input's pixels start in the packed source buffer.
#[derive(Debug, Clone, Copy)]
struct GpuPad {
    x0: i32,
    y0: i32,
    dw: u32,
    dh: u32,
    sw: u32,
    sh: u32,
    alpha: u32,
    off: u32,
}

/// Device resources sized to the current input geometry. Rebuilt when an input's
/// geometry changes (the packed source layout is derived from it).
#[derive(Debug)]
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    src_buf: wgpu::Buffer,
    pads_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// Byte offset of each input's pixels in `src_buf`.
    offsets: Vec<usize>,
    /// Row pitch of `out_buf`, padded to the 256-byte alignment a
    /// buffer -> texture copy requires.
    row_bytes: usize,
}

/// GPU compositor: N RGBA8 inputs blended into one canvas by a compute shader.
#[derive(Debug)]
pub struct WgpuCompositor {
    out_w: u32,
    out_h: u32,
    framerate_q16: u32,
    pads: Vec<CompositorPad>,
    /// Same latest-wins cadence as the CPU compositor: input 0 queues and
    /// releases one output frame each, every other input keeps only its newest.
    state: CompositorState,
    /// Inputs whose cached bytes are already in `src_buf`, so an overlay that
    /// has not moved is not re-uploaded per output frame.
    uploaded: Vec<bool>,
    background: [u8; 4],
    ctx: Option<GpuContext>,
    gpu: Option<Gpu>,
    gpu_output: bool,
}

impl WgpuCompositor {
    /// An `out_w` x `out_h` RGBA8 canvas at 30 fps with one [`CompositorPad`]
    /// per input (input 0 is the timing driver). Panics if `pads` is empty.
    pub fn new(out_w: u32, out_h: u32, pads: Vec<CompositorPad>) -> Self {
        assert!(!pads.is_empty(), "WgpuCompositor needs at least one input");
        let n = pads.len();
        Self {
            out_w,
            out_h,
            framerate_q16: 30 << 16,
            pads,
            state: CompositorState::new(n),
            uploaded: vec![false; n],
            background: [0, 0, 0, 255],
            ctx: None,
            gpu: None,
            gpu_output: false,
        }
    }

    /// Set the output framerate in nominal fps. Labels the output caps; the emit
    /// cadence follows input 0 (put a `VideoRate` downstream to resample).
    pub fn with_framerate(mut self, fps: u32) -> Self {
        self.framerate_q16 = fps << 16;
        self
    }

    /// Set the RGBA8 background the inputs composite over (default opaque black).
    pub fn with_background(mut self, rgba: [u8; 4]) -> Self {
        self.background = rgba;
        self
    }

    /// Composite on a shared [`GpuContext`] instead of opening a private device.
    /// Pass the downstream [`WgpuSink`](crate::wgpusink)'s context so a
    /// GPU-output texture presents with no copy.
    pub fn with_context(mut self, ctx: GpuContext) -> Self {
        self.ctx = Some(ctx);
        self
    }

    /// Emit the canvas GPU-resident ([`MemoryDomain::WgpuTexture`]) instead of
    /// reading it back to system memory: the composite stays on the device for a
    /// GPU consumer. Default off.
    pub fn with_gpu_output(mut self) -> Self {
        self.gpu_output = true;
        self
    }

    /// Number of composited frames emitted so far (one per input-0 frame).
    pub fn emitted(&self) -> u64 {
        self.state.emitted()
    }

    fn output(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(self.out_w),
            height: Dim::Fixed(self.out_h),
            framerate: Rate::Fixed(self.framerate_q16),
        }
    }

    fn accepted(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    /// Build the device resources for the current input geometry. Every input
    /// must be configured by now (the packed source layout needs all of them).
    async fn ensure_gpu(&mut self) -> Result<(), G2gError> {
        if self.gpu.is_some() {
            return Ok(());
        }
        let ctx = match self.ctx.clone() {
            Some(ctx) => ctx,
            None => GpuContext::headless().await?,
        };
        let gpu = self.build_gpu(ctx.device, ctx.queue)?;
        self.gpu = Some(gpu);
        self.uploaded.iter_mut().for_each(|u| *u = false);
        Ok(())
    }

    fn build_gpu(&self, device: wgpu::Device, queue: wgpu::Queue) -> Result<Gpu, G2gError> {
        // Pack the inputs back to back; an unconfigured input still gets a slot
        // (zero-sized), so offsets stay indexed by input number.
        let mut offsets = Vec::with_capacity(self.pads.len());
        let mut total = 0usize;
        for i in 0..self.pads.len() {
            offsets.push(total);
            let (w, h) = self.state.geometry(i).unwrap_or((0, 0));
            total += w as usize * h as usize * 4;
        }
        let row_bytes = (self.out_w as usize * 4).div_ceil(256) * 256;
        let out_bytes = row_bytes * self.out_h as usize;

        let src_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compositor-src"),
            // A zero-sized storage binding is invalid; keep one word minimum.
            size: total.max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pads_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compositor-pads"),
            size: (self.pads.len() * core::mem::size_of::<GpuPad>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compositor-params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compositor-canvas"),
            size: out_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compositor-readback"),
            size: out_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compositor-blend"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compositor-blend"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compositor-binding"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pads_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: src_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        Ok(Gpu {
            device,
            queue,
            pipeline,
            bind_group,
            src_buf,
            pads_buf,
            params_buf,
            out_buf,
            staging,
            offsets,
            row_bytes,
        })
    }

    /// The pads the shader should walk, in paint order: only inputs that are
    /// configured, non-degenerate, and have pixels available.
    fn gpu_pads(&self, offsets: &[usize]) -> Vec<GpuPad> {
        let mut out = Vec::with_capacity(self.pads.len());
        for i in paint_order(&self.pads) {
            let Some((sw, sh)) = self.state.geometry(i) else {
                continue;
            };
            if sw == 0 || sh == 0 {
                continue;
            }
            // Input 0 is the frame driving this output; the rest need a cached one.
            if i != 0 && self.state.latest(i).is_none() {
                continue;
            }
            let pad = self.pads[i];
            let (dw, dh) = pad.size.unwrap_or((sw, sh));
            if dw == 0 || dh == 0 {
                continue;
            }
            out.push(GpuPad {
                x0: pad.xpos,
                y0: pad.ypos,
                dw,
                dh,
                sw,
                sh,
                alpha: pad.alpha as u32,
                off: (offsets[i] / 4) as u32,
            });
        }
        out
    }

    /// Upload whatever changed and run one compositing dispatch. `base0` is the
    /// input-0 frame currently driving output; overlays come from their cached
    /// latest bytes and are uploaded only when newer than the device copy.
    fn dispatch(&mut self, base0: &[u8]) -> Result<(), G2gError> {
        let gpu = self.gpu.as_ref().ok_or(G2gError::NotConfigured)?;
        let pads = self.gpu_pads(&gpu.offsets);

        // Disjoint field borrows: the cached frames are read while `uploaded` is
        // updated.
        let (state, uploaded) = (&self.state, &mut self.uploaded);
        for (i, up) in uploaded.iter_mut().enumerate() {
            let Some((w, h)) = state.geometry(i) else {
                continue;
            };
            let need = w as usize * h as usize * 4;
            if need == 0 {
                continue;
            }
            let src: &[u8] = if i == 0 {
                base0
            } else {
                if *up {
                    continue;
                }
                match state.latest(i) {
                    Some((_, bytes)) => bytes,
                    None => continue,
                }
            };
            if src.len() < need {
                return Err(G2gError::CapsMismatch);
            }
            gpu.queue
                .write_buffer(&gpu.src_buf, gpu.offsets[i] as u64, &src[..need]);
            *up = true;
        }

        let mut pad_bytes = Vec::with_capacity(self.pads.len() * 32);
        for p in &pads {
            for w in [
                p.x0 as u32,
                p.y0 as u32,
                p.dw,
                p.dh,
                p.sw,
                p.sh,
                p.alpha,
                p.off,
            ] {
                pad_bytes.extend_from_slice(&w.to_le_bytes());
            }
        }
        // The pads buffer is sized for every input; a shorter paint list leaves
        // the tail stale, which `pad_count` keeps the shader from reading.
        if !pad_bytes.is_empty() {
            gpu.queue.write_buffer(&gpu.pads_buf, 0, &pad_bytes);
        }

        let mut params = [0u8; 32];
        for (slot, v) in [
            self.out_w,
            self.out_h,
            pads.len() as u32,
            u32::from_le_bytes(self.background),
            (gpu.row_bytes / 4) as u32,
        ]
        .into_iter()
        .enumerate()
        {
            params[slot * 4..slot * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        gpu.queue.write_buffer(&gpu.params_buf, 0, &params);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("composite"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.dispatch_workgroups(
                self.out_w.div_ceil(WORKGROUP),
                self.out_h.div_ceil(WORKGROUP),
                1,
            );
        }
        gpu.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Read the composited canvas back to tightly-packed system memory.
    fn read_canvas(&self) -> Result<Box<[u8]>, G2gError> {
        let gpu = self.gpu.as_ref().ok_or(G2gError::NotConfigured)?;
        let bytes = gpu.row_bytes * self.out_h as usize;
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&gpu.out_buf, 0, &gpu.staging, 0, bytes as u64);
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
            .map_err(gpu_err)?;
        rx.recv().map_err(gpu_err)?.map_err(gpu_err)?;

        let mapped = slice.get_mapped_range();
        let tight = self.out_w as usize * 4;
        let mut out = Vec::with_capacity(tight * self.out_h as usize);
        for row in 0..self.out_h as usize {
            let start = row * gpu.row_bytes;
            out.extend_from_slice(&mapped[start..start + tight]);
        }
        drop(mapped);
        gpu.staging.unmap();
        Ok(out.into_boxed_slice())
    }

    /// Copy the composited canvas into a fresh per-frame texture handed
    /// downstream. Fresh (not the shared buffer) so the next frame's dispatch
    /// cannot clobber a canvas still in flight.
    fn canvas_texture(&self) -> Result<wgpu::Texture, G2gError> {
        let gpu = self.gpu.as_ref().ok_or(G2gError::NotConfigured)?;
        let size = wgpu::Extent3d {
            width: self.out_w,
            height: self.out_h,
            depth_or_array_layers: 1,
        };
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("compositor-canvas"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &gpu.out_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(gpu.row_bytes as u32),
                    rows_per_image: Some(self.out_h),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            size,
        );
        gpu.queue.submit([encoder.finish()]);
        Ok(texture)
    }

    /// Composite `base0` and wrap the result as the next output frame, in
    /// whichever memory domain this compositor was built for.
    fn compose_frame(&mut self, base0: &[u8], timing: FrameTiming) -> Result<Frame, G2gError> {
        self.dispatch(base0)?;
        let domain = if self.gpu_output {
            MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
                self.out_w,
                self.out_h,
                Arc::new(WgpuTextureKeepAlive(self.canvas_texture()?)),
            ))
        } else {
            MemoryDomain::System(SystemSlice::from_boxed(self.read_canvas()?))
        };
        Ok(Frame::new(domain, timing, self.state.next_sequence()))
    }
}

impl MultiInputElement for WgpuCompositor {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.pads.len()
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.accepted())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(self.accepted()))
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(self.output())))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if *w > MAX_DIM || *h > MAX_DIM || self.out_w > MAX_DIM || self.out_h > MAX_DIM {
            return Err(G2gError::CapsMismatch);
        }
        if self.state.set_geometry(input, *w, *h) {
            // The packed source layout is derived from every input's geometry.
            self.gpu = None;
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.output())
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let (w, h) = self.state.geometry(input).ok_or(G2gError::NotConfigured)?;
                    let Some(src) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let need = w as usize * h as usize * 4;
                    if src.len() < need {
                        return Err(G2gError::CapsMismatch);
                    }
                    self.state.ingest(input, frame.timing, src[..need].into());
                    if input != 0 {
                        // The overlay's new bytes need re-uploading.
                        self.uploaded[input] = false;
                    }

                    while let Some((timing, base)) = self.state.take_due() {
                        self.ensure_gpu().await?;
                        let frame = self.compose_frame(&base, timing)?;
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                    }
                }
                // A per-input geometry change invalidates that input's queued
                // frames and the packed source layout; the output caps are fixed,
                // so nothing is forwarded.
                PipelinePacket::CapsChanged(Caps::RawVideo {
                    format: RawVideoFormat::Rgba8,
                    width: Dim::Fixed(w),
                    height: Dim::Fixed(h),
                    ..
                }) => {
                    if self.state.geometry(input) != Some((w, h)) {
                        if w > MAX_DIM || h > MAX_DIM {
                            return Err(G2gError::CapsMismatch);
                        }
                        self.state.set_geometry(input, w, h);
                        self.state.clear(input);
                        self.uploaded[input] = false;
                        self.gpu = None;
                    }
                }
                // A flush on an overlay drops its cached frame so a stale overlay
                // never lingers, and re-arms startup so it is waited for again.
                PipelinePacket::Flush => {
                    self.state.flush(input);
                    self.uploaded[input] = false;
                }
                PipelinePacket::Eos | PipelinePacket::Segment(_) => {}
                _ => {}
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compositor::Compositor;
    use g2g_core::PushOutcome;

    /// One device for the whole test binary, built under a lock: opening several
    /// wgpu devices concurrently crashes some drivers (seen as a SIGSEGV inside
    /// the NVIDIA driver when these tests each opened their own). `None` when the
    /// host has no adapter (CI), so every GPU test skips.
    async fn shared_ctx() -> Option<GpuContext> {
        static CTX: tokio::sync::Mutex<Option<GpuContext>> = tokio::sync::Mutex::const_new(None);
        let mut slot = CTX.lock().await;
        if slot.is_none() {
            *slot = GpuContext::headless().await.ok();
        }
        slot.clone()
    }

    fn rgba_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    fn solid(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..w * h {
            v.extend_from_slice(&rgba);
        }
        v
    }

    /// A horizontal red -> blue ramp with a vertical alpha ramp, so bilinear
    /// resize and per-pixel alpha both have something non-uniform to chew on.
    fn ramp(w: u32, h: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let t = (x * 255 / w.max(1)) as u8;
                v.extend_from_slice(&[255 - t, t / 2, t, (y * 255 / h.max(1)) as u8]);
            }
        }
        v
    }

    #[derive(Default)]
    struct FrameSink {
        frames: Vec<Frame>,
    }
    impl OutputSink for FrameSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(frame) = packet {
                    self.frames.push(frame);
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn frame_of(bytes: &[u8]) -> Frame {
        Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        )
    }

    /// Drive an element to one composited frame: overlays first (so priming
    /// completes), then the input-0 base. Returns the system-memory canvas.
    async fn run_cpu(mut comp: Compositor, geoms: &[(u32, u32)], frames: &[Vec<u8>]) -> Vec<u8> {
        let mut sink = FrameSink::default();
        for (i, (w, h)) in geoms.iter().enumerate() {
            comp.configure_pipeline(i, &rgba_caps(*w, *h)).unwrap();
        }
        for i in (1..geoms.len()).rev() {
            comp.process(
                i,
                PipelinePacket::DataFrame(frame_of(&frames[i])),
                &mut sink,
            )
            .await
            .unwrap();
        }
        comp.process(
            0,
            PipelinePacket::DataFrame(frame_of(&frames[0])),
            &mut sink,
        )
        .await
        .unwrap();
        let f = sink.frames.pop().expect("cpu frame");
        f.domain.as_system_slice().unwrap().to_vec()
    }

    async fn run_gpu(
        mut comp: WgpuCompositor,
        geoms: &[(u32, u32)],
        frames: &[Vec<u8>],
    ) -> (WgpuCompositor, Frame) {
        let mut sink = FrameSink::default();
        for (i, (w, h)) in geoms.iter().enumerate() {
            comp.configure_pipeline(i, &rgba_caps(*w, *h)).unwrap();
        }
        for i in (1..geoms.len()).rev() {
            comp.process(
                i,
                PipelinePacket::DataFrame(frame_of(&frames[i])),
                &mut sink,
            )
            .await
            .unwrap();
        }
        comp.process(
            0,
            PipelinePacket::DataFrame(frame_of(&frames[0])),
            &mut sink,
        )
        .await
        .unwrap();
        let f = sink.frames.pop().expect("gpu frame");
        (comp, f)
    }

    /// Composite the same scene on both elements and assert byte equality.
    /// Skips (passes) when the host has no adapter.
    async fn assert_parity(
        pads: Vec<CompositorPad>,
        out: (u32, u32),
        background: [u8; 4],
        geoms: &[(u32, u32)],
        frames: &[Vec<u8>],
        what: &str,
    ) {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping GPU compositor {what} test");
            return;
        };
        let cpu = Compositor::new(out.0, out.1, pads.clone()).with_background(background);
        let want = run_cpu(cpu, geoms, frames).await;
        let gpu = WgpuCompositor::new(out.0, out.1, pads)
            .with_background(background)
            .with_context(ctx);
        let (_, frame) = run_gpu(gpu, geoms, frames).await;
        let got = frame.domain.as_system_slice().unwrap().to_vec();
        assert_eq!(got.len(), want.len(), "{what}: canvas size");
        let bad = got
            .iter()
            .zip(want.iter())
            .position(|(a, b)| a != b)
            .map(|i| (i, got[i], want[i]));
        assert!(bad.is_none(), "{what}: first mismatch {bad:?}");
    }

    #[tokio::test]
    async fn matches_cpu_on_opaque_overlay() {
        assert_parity(
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(16, 16).with_zorder(1),
            ]),
            (64, 48),
            [10, 20, 30, 255],
            &[(64, 48), (32, 16)],
            &[solid(64, 48, [200, 40, 10, 255]), ramp(32, 16)],
            "opaque overlay",
        )
        .await;
    }

    #[tokio::test]
    async fn matches_cpu_on_zorder_and_negative_offset() {
        // Three overlapping pads, painted in a different order than their index,
        // one hanging off the top-left so clipping is exercised too.
        assert_parity(
            Vec::from([
                CompositorPad::at(0, 0).with_zorder(2),
                CompositorPad::at(-8, -4).with_zorder(5),
                CompositorPad::at(8, 8).with_zorder(1),
            ]),
            (40, 40),
            [0, 0, 0, 255],
            &[(40, 40), (24, 24), (24, 24)],
            &[
                solid(40, 40, [30, 30, 200, 255]),
                ramp(24, 24),
                solid(24, 24, [0, 255, 0, 255]),
            ],
            "z-order + clipping",
        )
        .await;
    }

    #[tokio::test]
    async fn matches_cpu_on_per_pad_alpha() {
        assert_parity(
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(4, 4).with_zorder(1).with_alpha(77),
                CompositorPad::at(2, 10).with_zorder(2).with_alpha(200),
            ]),
            (32, 32),
            [90, 90, 90, 255],
            &[(32, 32), (16, 16), (16, 16)],
            &[
                solid(32, 32, [255, 0, 0, 255]),
                ramp(16, 16),
                solid(16, 16, [0, 0, 255, 128]),
            ],
            "per-pad alpha",
        )
        .await;
    }

    #[tokio::test]
    async fn matches_cpu_on_scaled_pads() {
        // Up- and downscaled overlays: the shader's fixed-point bilinear must
        // land on the same samples as the CPU one.
        assert_parity(
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(2, 2).with_zorder(1).with_size(24, 20),
                CompositorPad::at(30, 8).with_zorder(2).with_size(9, 7),
            ]),
            (48, 40),
            [0, 0, 0, 255],
            &[(48, 40), (7, 5), (32, 32)],
            &[solid(48, 40, [12, 200, 60, 255]), ramp(7, 5), ramp(32, 32)],
            "bilinear resize",
        )
        .await;
    }

    #[tokio::test]
    async fn gpu_output_texture_matches_cpu() {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping GPU compositor texture test");
            return;
        };
        let pads = Vec::from([
            CompositorPad::at(0, 0),
            CompositorPad::at(6, 6).with_zorder(1).with_alpha(160),
        ]);
        let geoms = [(32, 32), (16, 16)];
        let frames = [solid(32, 32, [10, 120, 240, 255]), ramp(16, 16)];

        let want = run_cpu(Compositor::new(32, 32, pads.clone()), &geoms, &frames).await;

        let gpu = WgpuCompositor::new(32, 32, pads)
            .with_context(ctx.clone())
            .with_gpu_output();
        let (_comp, frame) = run_gpu(gpu, &geoms, &frames).await;

        let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
            panic!("gpu-output mode emits a WgpuTexture frame");
        };
        assert_eq!((owned.width, owned.height), (32, 32));
        let tex = crate::gpu::texture_of(owned).expect("texture keep-alive");
        let got = crate::gpu::read_rgba_texture(&ctx, tex);
        assert_eq!(got, want, "GPU-resident canvas matches the CPU compositor");
    }

    #[tokio::test]
    async fn stale_overlay_is_reused_until_a_newer_one_lands() {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping GPU compositor cadence test");
            return;
        };
        // Two output frames off one overlay upload: the second must still see
        // the overlay (the upload cache must not drop it), and a new overlay
        // must take effect on the third.
        let pads = Vec::from([
            CompositorPad::at(0, 0),
            CompositorPad::at(0, 0).with_zorder(1).with_size(4, 4),
        ]);
        let mut comp = WgpuCompositor::new(8, 8, pads).with_context(ctx);
        comp.configure_pipeline(0, &rgba_caps(8, 8)).unwrap();
        comp.configure_pipeline(1, &rgba_caps(4, 4)).unwrap();
        let mut sink = FrameSink::default();
        let base = solid(8, 8, [255, 0, 0, 255]);

        comp.process(
            1,
            PipelinePacket::DataFrame(frame_of(&solid(4, 4, [0, 255, 0, 255]))),
            &mut sink,
        )
        .await
        .unwrap();
        for _ in 0..2 {
            comp.process(0, PipelinePacket::DataFrame(frame_of(&base)), &mut sink)
                .await
                .unwrap();
        }
        comp.process(
            1,
            PipelinePacket::DataFrame(frame_of(&solid(4, 4, [0, 0, 255, 255]))),
            &mut sink,
        )
        .await
        .unwrap();
        comp.process(0, PipelinePacket::DataFrame(frame_of(&base)), &mut sink)
            .await
            .unwrap();

        assert_eq!(sink.frames.len(), 3, "one output per input-0 frame");
        let px = |f: &Frame| {
            let b = f.domain.as_system_slice().unwrap();
            [b[0], b[1], b[2], b[3]]
        };
        assert_eq!(px(&sink.frames[0]), [0, 255, 0, 255], "overlay painted");
        assert_eq!(
            px(&sink.frames[1]),
            [0, 255, 0, 255],
            "overlay still cached"
        );
        assert_eq!(px(&sink.frames[2]), [0, 0, 255, 255], "newer overlay wins");
    }

    #[test]
    fn rejects_non_rgba_and_oversized_geometry() {
        let mut comp = WgpuCompositor::new(64, 64, Vec::from([CompositorPad::at(0, 0)]));
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(64),
            height: Dim::Fixed(64),
            framerate: Rate::Fixed(30 << 16),
        };
        assert!(matches!(
            comp.configure_pipeline(0, &nv12),
            Err(G2gError::CapsMismatch)
        ));
        assert!(matches!(
            comp.configure_pipeline(0, &rgba_caps(MAX_DIM + 1, 64)),
            Err(G2gError::CapsMismatch)
        ));
        assert!(comp.configure_pipeline(0, &rgba_caps(64, 64)).is_ok());
        assert!(matches!(
            comp.output(),
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(64),
                ..
            }
        ));
    }
}

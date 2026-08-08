//! wgpu compute compositor (M853): the GPU sibling of the CPU
//! [`Compositor`](crate::compositor) for HD / many-input scale. Same
//! N-in-1-out [`MultiInputElement`] surface, same [`CompositorPad`] semantics
//! (position, z-order, per-pad alpha, optional bilinear resize), the same
//! latest-wins overlay cadence and the same
//! [`with_timed_output`](WgpuCompositor::with_timed_output) hold on a stalled
//! input 0; only the pixel work moves to a compute shader.
//!
//! RGBA8 only (the planar YUV mixing stays on the CPU element). A system-memory
//! frame is uploaded into one packed storage buffer, and re-uploaded only when a
//! new frame lands, so a slow overlay costs nothing per output frame. A
//! [`MemoryDomain::WgpuTexture`] frame (M874) is instead bound as a sampled
//! texture and composited where it already is: no download, no upload, no copy.
//! Pads mix freely, and texture inputs with
//! [`with_gpu_output`](WgpuCompositor::with_gpu_output) keep the whole composite
//! on the device. An input texture must be `Rgba8Unorm`, bindable, and created on
//! the compositor's own device: a GPU producer shares that device through
//! [`with_context`](WgpuCompositor::with_context), the same convention
//! [`WgpuSink`](crate::wgpusink) uses, and a foreign-device texture is a loud
//! wgpu validation failure rather than a silent copy back through memory.
//!
//! The compositing shader is generated per device build, since the number of
//! texture pads decides its bindings; a pad switching between bytes and a texture
//! rebuilds the device state.
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
//! to [`MAX_DIM`], which `configure_pipeline` enforces. A bound texture enters
//! that arithmetic through `textureLoad` (never a sampler), whose unorm texels
//! round back to the same 8-bit integers a packed pad carries.
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
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use crate::compositor::{
    color_property, color_value, dim_property, frame_period_ns, framerate_property, pad_property,
    paint_order, set_pad_property, CompositorPad, CompositorState, WGPU_COMPOSITOR_PROPS,
};
use crate::gpu::{gpu_err, texture_of, GpuContext, WgpuTextureKeepAlive};
use g2g_core::frame::Frame;
use g2g_core::memory::{OwnedWgpuTexture, SystemSlice};
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError,
    MemoryDomain, MultiInputElement, OutputSink, PipelinePacket, PropError, PropValue,
    PropertySpec, Rate, RawVideoFormat,
};

/// Largest input / canvas edge the shader's fixed-point resize stays exact for
/// (the `(2*d+1) * s * 32768 / dst` mapping must not overflow `u32`).
pub const MAX_DIM: u32 = 16384;

const WORKGROUP: u32 = 8;

/// `u32`s per `Pad` in the pads buffer, matching the WGSL struct.
const PAD_WORDS: usize = 9;

/// First bind-group slot for a texture pad; 0..=3 are the buffers.
const TEX_BINDING_BASE: u32 = 4;

/// Head of the compositing compute shader: the bindings every dispatch has and
/// the integer arithmetic shared with the CPU compositor. `pads` arrives
/// pre-sorted in paint order and carries each input's placement, its offset into
/// the packed `src` buffer and which source it reads.
const SHADER_HEAD: &str = r#"
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
    // 0 reads the packed src buffer, else the (1-based) texture binding.
    src_kind: u32,
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
"#;

/// Tail of the compositing shader, after the generated texture bindings and
/// their `load_tex`: one source fetch that both the 1:1 and the resized path go
/// through, so a pad's pixels come from the packed buffer or its bound texture
/// with identical arithmetic downstream.
const SHADER_TAIL: &str = r#"
fn fetch(p: Pad, x: u32, y: u32) -> vec4<u32> {
    if (p.src_kind == 0u) {
        return unpack(src[p.off + y * p.sw + x]);
    }
    return load_tex(p.src_kind, vec2<i32>(i32(x), i32(y)));
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
    let s00 = fetch(p, x0, y0);
    let s01 = fetch(p, x1, y0);
    let s10 = fetch(p, x0, y1);
    let s11 = fetch(p, x1, y1);
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
            px = fetch(p, u32(dx), u32(dy));
        } else {
            px = sample_scaled(p, u32(dx), u32(dy));
        }
        acc = blend(acc, px, p.alpha);
    }
    dst[gid.y * params.stride_words + gid.x] = pack(acc);
}
"#;

/// The shader for `tex_count` texture pads: one `texture_2d<f32>` binding each,
/// and a `load_tex` switch over them (a fixed binding per pad, no binding_array,
/// so the base wgpu feature set is enough). `textureLoad` never samples: it
/// returns the texel's unorm floats, and rounding them back to 0..=255 recovers
/// the exact 8-bit integers the shared blend arithmetic works in, keeping a
/// texture pad bit-exact with the CPU compositor.
fn shader_source(tex_count: usize) -> String {
    let mut s = String::from(SHADER_HEAD);
    for slot in 0..tex_count {
        let binding = TEX_BINDING_BASE + slot as u32;
        s += &format!("@group(0) @binding({binding}) var tex{slot}: texture_2d<f32>;\n");
    }
    s += "\nfn load_tex(kind: u32, xy: vec2<i32>) -> vec4<u32> {\n";
    s += "    var v = vec4<f32>(0.0);\n    switch kind {\n";
    for slot in 0..tex_count {
        let kind = slot + 1;
        s += &format!("        case {kind}u: {{ v = textureLoad(tex{slot}, xy, 0); }}\n");
    }
    s += "        default: {}\n    }\n    return vec4<u32>(round(v * 255.0));\n}\n";
    s += SHADER_TAIL;
    s
}

/// One input's pixels as the compositor cached them: system-memory bytes to
/// upload into the packed source buffer, or a producer's texture bound and
/// sampled where it already is, never copied.
#[derive(Debug)]
enum Source {
    Bytes(Box<[u8]>),
    Texture(OwnedWgpuTexture),
}

impl Source {
    fn is_texture(&self) -> bool {
        matches!(self, Self::Texture(_))
    }

    fn texture(&self) -> Option<&wgpu::Texture> {
        match self {
            Self::Texture(owned) => texture_of(owned),
            Self::Bytes(_) => None,
        }
    }
}

/// One pad as the shader reads it: placement on the canvas, source geometry, and
/// where its pixels come from (the packed buffer at `off`, or the texture
/// binding `src_kind` selects).
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
    src_kind: u32,
}

impl GpuPad {
    /// The pad as the shader's `Pad` struct, field order included.
    fn words(&self) -> [u32; PAD_WORDS] {
        [
            self.x0 as u32,
            self.y0 as u32,
            self.dw,
            self.dh,
            self.sw,
            self.sh,
            self.alpha,
            self.off,
            self.src_kind,
        ]
    }
}

/// Device resources sized to the current input geometry and source kinds.
/// Rebuilt when an input's geometry changes (the packed source layout is derived
/// from it) or when a pad switches between bytes and a texture (so are the
/// layout and the shader).
#[derive(Debug)]
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    /// Kept to rebuild the bind group when a texture pad's handle changes.
    layout: wgpu::BindGroupLayout,
    /// The group as built with this device: usable as-is while no pad is a
    /// texture, otherwise a starting point rebuilt per dispatch.
    bind_group: wgpu::BindGroup,
    src_buf: wgpu::Buffer,
    pads_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// Byte offset of each input's pixels in `src_buf`. A texture input gets a
    /// zero-sized slot: its pixels are never packed there.
    offsets: Vec<usize>,
    /// Inputs bound as textures, in binding order: a pad's position here + 1 is
    /// its `src_kind`, and [`TEX_BINDING_BASE`] + position its binding.
    tex_inputs: Vec<usize>,
    /// Stands in for a texture pad with nothing cached yet (pre-first-frame or
    /// post-flush), so every declared binding has a texture. Never sampled: such
    /// a pad is left out of the paint list.
    dummy_tex: wgpu::Texture,
    /// Row pitch of `out_buf`, padded to the 256-byte alignment a
    /// buffer -> texture copy requires.
    row_bytes: usize,
}

impl Gpu {
    /// Views for each texture pad's binding, in `tex_inputs` order.
    fn texture_views(&self, sources: &[Option<&Source>]) -> Vec<wgpu::TextureView> {
        self.tex_inputs
            .iter()
            .map(|&i| {
                let tex = sources[i]
                    .and_then(Source::texture)
                    .unwrap_or(&self.dummy_tex);
                tex.create_view(&Default::default())
            })
            .collect()
    }
}

/// Bind the four fixed buffers plus one texture view per texture pad. The pads
/// buffer is bound whole; `pad_count` keeps the shader off any stale tail.
fn build_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    bufs: [&wgpu::Buffer; 4],
    views: &[wgpu::TextureView],
) -> wgpu::BindGroup {
    let mut entries: Vec<wgpu::BindGroupEntry> = bufs
        .iter()
        .enumerate()
        .map(|(i, buf)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buf.as_entire_binding(),
        })
        .collect();
    for (slot, view) in views.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: TEX_BINDING_BASE + slot as u32,
            resource: wgpu::BindingResource::TextureView(view),
        });
    }
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compositor-binding"),
        layout,
        entries: &entries,
    })
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
    /// Caches either system-memory bytes or a texture, per input.
    state: CompositorState<Source>,
    /// Inputs whose cached bytes are already in `src_buf`, so an overlay that
    /// has not moved is not re-uploaded per output frame. Bytes pads only: a
    /// texture pad is bound, never uploaded.
    uploaded: Vec<bool>,
    /// Inputs delivering textures rather than bytes, as observed from their
    /// frames. Both the generated shader and the packed source layout derive
    /// from this, so a pad switching kind rebuilds the device state.
    textures: Vec<bool>,
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
            textures: vec![false; n],
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

    /// Keep emitting at the output framerate while input 0 stalls, re-compositing
    /// its last frame once per frame period with the overlays as they stand
    /// (zero-order-hold). Off by default. Needs a pipeline clock that can sleep on
    /// a deadline, as on the CPU element.
    pub fn with_timed_output(mut self) -> Self {
        self.state.set_hold(true);
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
            interlace: g2g_core::Interlace::Any,
        }
    }

    fn accepted(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
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
        // Pack the inputs back to back; an unconfigured input, and one bound as
        // a texture, still gets a (zero-sized) slot, so offsets stay indexed by
        // input number.
        let mut offsets = Vec::with_capacity(self.pads.len());
        let mut total = 0usize;
        for i in 0..self.pads.len() {
            offsets.push(total);
            let (w, h) = match self.textures[i] {
                true => (0, 0),
                false => self.state.geometry(i).unwrap_or((0, 0)),
            };
            total += w as usize * h as usize * 4;
        }
        let tex_inputs: Vec<usize> = (0..self.pads.len()).filter(|&i| self.textures[i]).collect();
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
            size: (self.pads.len() * PAD_WORDS * 4) as u64,
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

        let dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("compositor-unbound-pad"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compositor-blend"),
            source: wgpu::ShaderSource::Wgsl(shader_source(tex_inputs.len()).into()),
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
        // Every texture pad starts on the dummy; the first dispatch rebinds it to
        // whatever frame it has cached by then.
        let dummy_views: Vec<wgpu::TextureView> = tex_inputs
            .iter()
            .map(|_| dummy_tex.create_view(&Default::default()))
            .collect();
        let bind_group = build_bind_group(
            &device,
            &layout,
            [&params_buf, &pads_buf, &src_buf, &out_buf],
            &dummy_views,
        );
        Ok(Gpu {
            device,
            queue,
            pipeline,
            layout,
            bind_group,
            src_buf,
            pads_buf,
            params_buf,
            out_buf,
            staging,
            offsets,
            tex_inputs,
            dummy_tex,
            row_bytes,
        })
    }

    /// The pads the shader should walk, in paint order: only inputs that are
    /// configured, non-degenerate, and have pixels available.
    fn gpu_pads(&self, gpu: &Gpu) -> Vec<GpuPad> {
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
            let (dw, dh) = pad.dest_size(sw, sh);
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
                off: (gpu.offsets[i] / 4) as u32,
                // A texture pad reads its own binding, the rest the packed buffer.
                src_kind: match gpu.tex_inputs.iter().position(|&t| t == i) {
                    Some(slot) => slot as u32 + 1,
                    None => 0,
                },
            });
        }
        out
    }

    /// Upload whatever changed and run one compositing dispatch. `base0` is the
    /// input-0 frame currently driving output; overlays come from their cached
    /// latest pixels, bytes uploaded only when newer than the device copy and
    /// textures bound where they are.
    fn dispatch(&mut self, base0: &Source) -> Result<(), G2gError> {
        let gpu = self.gpu.as_ref().ok_or(G2gError::NotConfigured)?;
        let pads = self.gpu_pads(gpu);
        // Each input's pixels for this output frame: input 0 drives it, the rest
        // contribute their cached latest.
        let sources: Vec<Option<&Source>> = (0..self.pads.len())
            .map(|i| match i {
                0 => Some(base0),
                _ => self.state.latest(i).map(|(_, s)| s),
            })
            .collect();

        // Disjoint field borrows: the cached frames are read while `uploaded` is
        // updated.
        let uploaded = &mut self.uploaded;
        for (i, up) in uploaded.iter_mut().enumerate() {
            let Some((w, h)) = self.state.geometry(i) else {
                continue;
            };
            let need = w as usize * h as usize * 4;
            if need == 0 {
                continue;
            }
            if i != 0 && *up {
                continue;
            }
            // A texture pad is bound, not uploaded; there is nothing to pack.
            let Some(Source::Bytes(bytes)) = sources[i] else {
                continue;
            };
            if bytes.len() < need {
                return Err(G2gError::CapsMismatch);
            }
            gpu.queue
                .write_buffer(&gpu.src_buf, gpu.offsets[i] as u64, &bytes[..need]);
            *up = true;
        }

        let mut pad_bytes = Vec::with_capacity(pads.len() * PAD_WORDS * 4);
        for p in &pads {
            for w in p.words() {
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

        // A texture pad's handle changes with every frame it delivers, so the
        // group is rebuilt around this frame's textures; with no texture pad the
        // one built with the device stands.
        let rebound = match gpu.tex_inputs.is_empty() {
            true => None,
            false => Some(build_bind_group(
                &gpu.device,
                &gpu.layout,
                [&gpu.params_buf, &gpu.pads_buf, &gpu.src_buf, &gpu.out_buf],
                &gpu.texture_views(&sources),
            )),
        };

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("composite"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, rebound.as_ref().unwrap_or(&gpu.bind_group), &[]);
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

    /// The pixels to cache for a delivered frame. A GPU texture is kept by
    /// reference and composited where it lies (no copy, no upload); a
    /// system-memory frame is copied out of its slice as before. Either way the
    /// geometry must be the one negotiated for that input.
    fn source_of(&self, domain: &MemoryDomain, w: u32, h: u32) -> Result<Source, G2gError> {
        if let MemoryDomain::WgpuTexture(owned) = domain {
            if (owned.width, owned.height) != (w, h) {
                return Err(G2gError::CapsMismatch);
            }
            // A foreign producer's keep-alive holds a texture this element cannot
            // reach, so it cannot be bound.
            let tex = texture_of(owned).ok_or(G2gError::UnsupportedDomain)?;
            // It is sampled in place: it must be bindable, and hold the RGBA8 the
            // caps promise (the blend reads its texels as 8-bit integers).
            if tex.format() != wgpu::TextureFormat::Rgba8Unorm
                || !tex.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING)
            {
                return Err(G2gError::CapsMismatch);
            }
            return Ok(Source::Texture(owned.clone()));
        }
        let Some(src) = domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        let need = w as usize * h as usize * 4;
        if src.len() < need {
            return Err(G2gError::CapsMismatch);
        }
        Ok(Source::Bytes(src[..need].into()))
    }

    /// Composite `base0` and wrap the result as the next output frame, in
    /// whichever memory domain this compositor was built for.
    fn compose_frame(&mut self, base0: &Source, timing: FrameTiming) -> Result<Frame, G2gError> {
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
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "GPU video compositor",
            "Filter/Editing/Video",
            "Composites several video inputs onto one timed output canvas on the GPU",
            "g2g",
        )
    }
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.pads.len()
    }

    /// Only with timed output on: the arm then ticks once per output frame period,
    /// which is when a zero-order-hold frame may be due.
    fn tick_interval_ns(&self) -> Option<u64> {
        match self.state.hold_enabled() {
            true => Some(frame_period_ns(self.framerate_q16)).filter(|&ns| ns > 0),
            false => None,
        }
    }

    fn properties(&self) -> &'static [PropertySpec] {
        WGPU_COMPOSITOR_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(applied) = set_pad_property(&mut self.pads, name, &value) {
            return applied;
        }
        match name {
            // The canvas geometry sizes the device buffers, so a change rebuilds
            // them on the next dispatch.
            "width" => {
                self.out_w = dim_property(&value)?;
                self.gpu = None;
            }
            "height" => {
                self.out_h = dim_property(&value)?;
                self.gpu = None;
            }
            "framerate" => self.framerate_q16 = framerate_property(&value)?,
            "background-color" => self.background = color_property(&value)?,
            "timed-output" => self.state.set_hold(value.as_bool().ok_or(PropError::Type)?),
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = pad_property(&self.pads, name) {
            return Some(value);
        }
        Some(match name {
            "width" => PropValue::Uint(self.out_w as u64),
            "height" => PropValue::Uint(self.out_h as u64),
            "framerate" => PropValue::Fraction((self.framerate_q16 >> 16) as i32, 1),
            "background-color" => color_value(self.background),
            "timed-output" => PropValue::Bool(self.state.hold_enabled()),
            _ => return None,
        })
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
                    let source = self.source_of(&frame.domain, w, h)?;
                    // The shader text and the packed source layout both derive
                    // from which pads are textures, so a switch rebuilds them.
                    if self.textures[input] != source.is_texture() {
                        self.textures[input] = source.is_texture();
                        self.gpu = None;
                        if input == 0 {
                            // A retained frame of the other kind no longer fits
                            // the layout it would be composited under.
                            self.state.drop_held();
                        }
                    }
                    if input != 0 {
                        // The overlay's new bytes need re-uploading.
                        self.uploaded[input] = false;
                    }
                    self.state.ingest(input, frame.timing, source);

                    while let Some((timing, base)) = self.state.take_due() {
                        self.ensure_gpu().await?;
                        let frame = self.compose_frame(&base, timing)?;
                        self.state.record_emitted(timing, base);
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                    }
                }
                // Zero-order-hold: input 0 has not delivered for a whole output
                // period, so re-composite the frame it last did (bytes still in
                // the packed buffer, or the texture still bound) with the overlays
                // as they now stand.
                PipelinePacket::Tick => {
                    let period = frame_period_ns(self.framerate_q16);
                    if let Some((timing, base)) = self.state.take_tick_due(period) {
                        self.ensure_gpu().await?;
                        let frame = self.compose_frame(&base, timing)?;
                        self.state.record_held(timing, base);
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
    use crate::compositor::{Compositor, PENDING_CAP};
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
            interlace: g2g_core::Interlace::Any,
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

    /// An RGBA8 texture frame on `ctx`'s device: what a GPU producer sharing the
    /// compositor's context hands over, composited without a copy.
    fn texture_frame(ctx: &GpuContext, w: u32, h: u32, pixels: &[u8]) -> Frame {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test-input"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        Frame::new(
            MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
                w,
                h,
                Arc::new(WgpuTextureKeepAlive(texture)),
            )),
            FrameTiming::default(),
            0,
        )
    }

    /// Drive the GPU element to one composited frame, delivering input `i`'s
    /// pixels as a texture when `tex[i]` and in system memory otherwise.
    /// Overlays first, so priming completes before the input-0 frame.
    async fn run_gpu_mixed(
        mut comp: WgpuCompositor,
        ctx: &GpuContext,
        geoms: &[(u32, u32)],
        frames: &[Vec<u8>],
        tex: &[bool],
    ) -> (WgpuCompositor, Frame) {
        let mut sink = FrameSink::default();
        for (i, (w, h)) in geoms.iter().enumerate() {
            comp.configure_pipeline(i, &rgba_caps(*w, *h)).unwrap();
        }
        let frame_for = |i: usize| match tex[i] {
            true => texture_frame(ctx, geoms[i].0, geoms[i].1, &frames[i]),
            false => frame_of(&frames[i]),
        };
        for i in (1..geoms.len()).rev() {
            comp.process(i, PipelinePacket::DataFrame(frame_for(i)), &mut sink)
                .await
                .unwrap();
        }
        comp.process(0, PipelinePacket::DataFrame(frame_for(0)), &mut sink)
            .await
            .unwrap();
        let f = sink.frames.pop().expect("gpu frame");
        (comp, f)
    }

    /// Composite the same scene on the CPU element and on the GPU one with the
    /// `tex` inputs delivered as bound textures, then assert byte equality: a
    /// texture pad must be as bit-exact as an uploaded one.
    async fn assert_texture_parity(
        pads: Vec<CompositorPad>,
        out: (u32, u32),
        background: [u8; 4],
        geoms: &[(u32, u32)],
        frames: &[Vec<u8>],
        tex: &[bool],
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
            .with_context(ctx.clone());
        let (_, frame) = run_gpu_mixed(gpu, &ctx, geoms, frames, tex).await;
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

    #[tokio::test]
    async fn texture_overlay_matches_cpu() {
        // A system-memory base with a GPU overlay bound in place: the two source
        // kinds mix on one canvas and still match the CPU element byte for byte.
        assert_texture_parity(
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(16, 16).with_zorder(1).with_alpha(200),
            ]),
            (64, 48),
            [10, 20, 30, 255],
            &[(64, 48), (32, 16)],
            &[solid(64, 48, [200, 40, 10, 255]), ramp(32, 16)],
            &[false, true],
            "texture overlay",
        )
        .await;
    }

    #[tokio::test]
    async fn all_texture_inputs_match_cpu() {
        // Every pad a texture, overlapping, painted out of index order, one
        // clipped off the top-left.
        assert_texture_parity(
            Vec::from([
                CompositorPad::at(0, 0).with_zorder(2),
                CompositorPad::at(-8, -4).with_zorder(5),
                CompositorPad::at(8, 8).with_zorder(1).with_alpha(77),
            ]),
            (40, 40),
            [0, 0, 0, 255],
            &[(40, 40), (24, 24), (24, 24)],
            &[
                solid(40, 40, [30, 30, 200, 255]),
                ramp(24, 24),
                solid(24, 24, [0, 255, 0, 255]),
            ],
            &[true, true, true],
            "all-texture inputs",
        )
        .await;
    }

    #[tokio::test]
    async fn scaled_texture_pads_match_cpu() {
        // Up- and downscaled texture pads: textureLoad plus the Q16 mapping must
        // land on the same samples as the CPU bilinear.
        assert_texture_parity(
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(2, 2).with_zorder(1).with_size(24, 20),
                CompositorPad::at(30, 8).with_zorder(2).with_size(9, 7),
            ]),
            (48, 40),
            [0, 0, 0, 255],
            &[(48, 40), (7, 5), (32, 32)],
            &[solid(48, 40, [12, 200, 60, 255]), ramp(7, 5), ramp(32, 32)],
            &[false, true, true],
            "scaled texture pads",
        )
        .await;
    }

    #[tokio::test]
    async fn newer_texture_overlay_wins() {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping GPU compositor texture cadence test");
            return;
        };
        // Two outputs off one texture overlay (it stays bound), then a newer
        // texture must take effect on the third.
        let pads = Vec::from([
            CompositorPad::at(0, 0),
            CompositorPad::at(0, 0).with_zorder(1).with_size(4, 4),
        ]);
        let mut comp = WgpuCompositor::new(8, 8, pads).with_context(ctx.clone());
        comp.configure_pipeline(0, &rgba_caps(8, 8)).unwrap();
        comp.configure_pipeline(1, &rgba_caps(4, 4)).unwrap();
        let mut sink = FrameSink::default();
        let base = solid(8, 8, [255, 0, 0, 255]);

        let green = texture_frame(&ctx, 4, 4, &solid(4, 4, [0, 255, 0, 255]));
        comp.process(1, PipelinePacket::DataFrame(green), &mut sink)
            .await
            .unwrap();
        for _ in 0..2 {
            comp.process(0, PipelinePacket::DataFrame(frame_of(&base)), &mut sink)
                .await
                .unwrap();
        }
        let blue = texture_frame(&ctx, 4, 4, &solid(4, 4, [0, 0, 255, 255]));
        comp.process(1, PipelinePacket::DataFrame(blue), &mut sink)
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
        assert_eq!(px(&sink.frames[0]), [0, 255, 0, 255], "texture painted");
        assert_eq!(
            px(&sink.frames[1]),
            [0, 255, 0, 255],
            "same texture still bound"
        );
        assert_eq!(
            px(&sink.frames[2]),
            [0, 0, 255, 255],
            "newer texture rebinds"
        );
    }

    #[tokio::test]
    async fn flushed_texture_pad_is_left_out() {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping GPU compositor texture flush test");
            return;
        };
        // A flushed texture pad has nothing to bind, so its declared binding falls
        // back to the dummy and the pad drops out of the paint list: the startup
        // overflow then emits the base overlay-less rather than sampling garbage.
        let pads = Vec::from([
            CompositorPad::at(0, 0),
            CompositorPad::at(0, 0).with_zorder(1),
        ]);
        let mut comp = WgpuCompositor::new(4, 4, pads).with_context(ctx.clone());
        comp.configure_pipeline(0, &rgba_caps(4, 4)).unwrap();
        comp.configure_pipeline(1, &rgba_caps(4, 4)).unwrap();
        let mut sink = FrameSink::default();

        let overlay = texture_frame(&ctx, 4, 4, &solid(4, 4, [0, 255, 0, 255]));
        comp.process(1, PipelinePacket::DataFrame(overlay), &mut sink)
            .await
            .unwrap();
        comp.process(1, PipelinePacket::Flush, &mut sink)
            .await
            .unwrap();
        let base = solid(4, 4, [255, 0, 0, 255]);
        for _ in 0..=PENDING_CAP {
            comp.process(0, PipelinePacket::DataFrame(frame_of(&base)), &mut sink)
                .await
                .unwrap();
        }

        assert_eq!(
            sink.frames.len(),
            1,
            "one overlay-less frame off the overflow"
        );
        let b = sink.frames[0].domain.as_system_slice().unwrap();
        assert_eq!(
            [b[0], b[1], b[2], b[3]],
            [255, 0, 0, 255],
            "base only, the flushed texture contributes nothing"
        );
    }

    #[tokio::test]
    async fn texture_geometry_mismatch_is_rejected() {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping GPU compositor texture caps test");
            return;
        };
        let mut comp = WgpuCompositor::new(8, 8, Vec::from([CompositorPad::at(0, 0)]))
            .with_context(ctx.clone());
        comp.configure_pipeline(0, &rgba_caps(8, 8)).unwrap();
        let mut sink = FrameSink::default();
        let small = texture_frame(&ctx, 4, 4, &solid(4, 4, [0, 255, 0, 255]));
        assert!(matches!(
            comp.process(0, PipelinePacket::DataFrame(small), &mut sink)
                .await,
            Err(G2gError::CapsMismatch)
        ));
        assert!(sink.frames.is_empty(), "nothing composited");
    }

    #[tokio::test]
    async fn texture_input_to_gpu_output_stays_on_the_device() {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping GPU compositor texture-to-texture test");
            return;
        };
        // Texture in, texture out: nothing touches system memory between the
        // producer and the consumer, and the canvas still matches the CPU element.
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
        let (comp, frame) = run_gpu_mixed(gpu, &ctx, &geoms, &frames, &[true, true]).await;

        // Both inputs are bound textures, so nothing was packed for upload: the
        // source buffer is the one-word minimum a storage binding needs.
        assert_eq!(
            comp.gpu.as_ref().expect("device built").src_buf.size(),
            4,
            "texture inputs are sampled in place, never uploaded"
        );

        let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
            panic!("gpu-output mode emits a WgpuTexture frame");
        };
        assert_eq!((owned.width, owned.height), (32, 32));
        let tex = crate::gpu::texture_of(owned).expect("texture keep-alive");
        assert_eq!(
            crate::gpu::read_rgba_texture(&ctx, tex),
            want,
            "GPU-resident canvas off texture inputs matches the CPU compositor"
        );
    }

    /// Drive a timed compositor through one held frame: overlay, base, then two
    /// ticks. The first tick closes the period the real frame arrived in, the
    /// second finds an empty period and emits the zero-order-hold frame.
    async fn run_held<M: MultiInputElement>(
        comp: &mut M,
        geoms: &[(u32, u32)],
        frames: &[Vec<u8>],
    ) -> Vec<Frame> {
        let mut sink = FrameSink::default();
        for (i, (w, h)) in geoms.iter().enumerate() {
            comp.configure_pipeline(i, &rgba_caps(*w, *h)).unwrap();
        }
        comp.process(
            1,
            PipelinePacket::DataFrame(frame_of(&frames[1])),
            &mut sink,
        )
        .await
        .unwrap();
        comp.process(
            0,
            PipelinePacket::DataFrame(frame_of(&frames[0])),
            &mut sink,
        )
        .await
        .unwrap();
        for _ in 0..2 {
            comp.process(0, PipelinePacket::Tick, &mut sink)
                .await
                .unwrap();
        }
        sink.frames
    }

    #[tokio::test]
    async fn held_frame_matches_the_cpu_compositor() {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping GPU compositor hold test");
            return;
        };
        let pads = Vec::from([
            CompositorPad::at(0, 0),
            CompositorPad::at(4, 4).with_zorder(1).with_alpha(180),
        ]);
        let geoms = [(32, 32), (16, 16)];
        let frames = [solid(32, 32, [200, 40, 10, 255]), ramp(16, 16)];

        let mut cpu = Compositor::new(32, 32, pads.clone()).with_timed_output();
        let cpu_frames = run_held(&mut cpu, &geoms, &frames).await;
        let mut gpu = WgpuCompositor::new(32, 32, pads)
            .with_context(ctx)
            .with_timed_output();
        let gpu_frames = run_held(&mut gpu, &geoms, &frames).await;

        assert_eq!(gpu_frames.len(), 2, "the real frame plus one held frame");
        assert_eq!(
            cpu_frames.len(),
            gpu_frames.len(),
            "same cadence as the CPU"
        );
        let bytes = |f: &Frame| f.domain.as_system_slice().unwrap().to_vec();
        assert_eq!(
            bytes(&gpu_frames[1]),
            bytes(&cpu_frames[1]),
            "the held composite is bit-exact with the CPU element"
        );
        assert_eq!(
            gpu_frames[1].timing.pts_ns, cpu_frames[1].timing.pts_ns,
            "and carries the same advanced timestamp"
        );
        assert_eq!(
            gpu_frames[1].timing.pts_ns,
            1_000_000_000 * 65536 / (30 << 16),
            "one 30 fps period past the real frame"
        );
    }

    #[tokio::test]
    async fn held_texture_base_picks_up_a_newer_texture_overlay() {
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping GPU compositor texture hold test");
            return;
        };
        // Both inputs GPU-resident: the retained base texture stays bound across
        // ticks while the overlay it composites with moves on.
        let pads = Vec::from([
            CompositorPad::at(0, 0),
            CompositorPad::at(0, 0).with_zorder(1),
        ]);
        let mut comp = WgpuCompositor::new(8, 8, pads)
            .with_context(ctx.clone())
            .with_timed_output();
        comp.configure_pipeline(0, &rgba_caps(8, 8)).unwrap();
        comp.configure_pipeline(1, &rgba_caps(4, 4)).unwrap();
        let mut sink = FrameSink::default();

        let green = texture_frame(&ctx, 4, 4, &solid(4, 4, [0, 255, 0, 255]));
        comp.process(1, PipelinePacket::DataFrame(green), &mut sink)
            .await
            .unwrap();
        let base = texture_frame(&ctx, 8, 8, &solid(8, 8, [255, 0, 0, 255]));
        comp.process(0, PipelinePacket::DataFrame(base), &mut sink)
            .await
            .unwrap();
        let blue = texture_frame(&ctx, 4, 4, &solid(4, 4, [0, 0, 255, 255]));
        comp.process(1, PipelinePacket::DataFrame(blue), &mut sink)
            .await
            .unwrap();
        for _ in 0..2 {
            comp.process(0, PipelinePacket::Tick, &mut sink)
                .await
                .unwrap();
        }

        assert_eq!(sink.frames.len(), 2, "one held frame off the empty period");
        let at = |f: &Frame, i: usize| {
            let b = f.domain.as_system_slice().unwrap();
            [b[i], b[i + 1], b[i + 2], b[i + 3]]
        };
        assert_eq!(at(&sink.frames[0], 0), [0, 255, 0, 255], "first overlay");
        assert_eq!(
            at(&sink.frames[1], 0),
            [0, 0, 255, 255],
            "the held frame composites the newer overlay texture"
        );
        // Outside the 4x4 overlay: the retained base texture is still what shows.
        assert_eq!(
            at(&sink.frames[1], (7 * 8 + 7) * 4),
            [255, 0, 0, 255],
            "the retained base texture stayed bound"
        );
    }

    #[test]
    fn rejects_non_rgba_and_oversized_geometry() {
        let mut comp = WgpuCompositor::new(64, 64, Vec::from([CompositorPad::at(0, 0)]));
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(64),
            height: Dim::Fixed(64),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
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

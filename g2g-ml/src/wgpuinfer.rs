//! GPU-resident tensor inference via wgpu compute (DESIGN.md §5.2, M216).
//!
//! `WgpuInference` is the consumer half of the keep-on-GPU inference branch
//! `WgpuPreprocess::with_gpu_output` (M215) opened: it takes the f32 NCHW tensor
//! straight out of the producer's `wgpu::Buffer` and runs the inference on the
//! GPU, so the tensor never makes the GPU->CPU->GPU round-trip an opaque-backend
//! consumer (burn / ort) would force. v1 ships the same single linear layer as
//! `BurnInference` (`output = input . W + b`), run as a wgpu matmul compute pass,
//! so its output is bit-for-bit comparable on the CPU and against the burn path.
//!
//! The trick that makes it zero-copy is device identity: a `wgpu::Buffer` is
//! bindable only on the `wgpu::Device` that created it, so the element does not
//! own a device. It adopts the producer's device / queue (carried by the
//! incoming [`WgpuBufferOwner`]) on the first frame, binds the input buffer
//! directly, and submits its compute on the producer's queue, which orders it
//! after the producer's already-submitted work with no fence or read-back.
//!
//! Input is `Caps::Tensor{F32,[1,C,H,W],Nchw}` in `MemoryDomain::WgpuBuffer`
//! (anything else is `UnsupportedDomain`); output is read back to
//! `MemoryDomain::System` by default or left GPU-resident with
//! [`with_gpu_output`](WgpuInference::with_gpu_output) for a downstream GPU
//! consumer, so a stack of these elements runs a small CNN entirely on-device.
//!
//! Beyond the matmul `linear`, the element offers a small op zoo, each a compute
//! pass on the producer's device: `conv2d` (the keystone, M261), the weightless
//! activations `relu` / `sigmoid`, and `maxpool2d` / `avgpool2d`. The weighted
//! ops bind (meta, input, weights, bias, out); the weightless ones bind only
//! (meta, input, out). Chaining them GPU-resident (conv -> relu -> pool -> ...)
//! is a real CNN body with no GPU->CPU round-trip between layers. Trained weights
//! load via `conv2d_from_safetensors` (M262).

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, HardwareError,
    MemoryDomain, OutputSink, OwnedWgpuBuffer, PipelinePacket, TensorDType, TensorLayout,
    TensorShape,
};

use crate::wgpupreprocess::WgpuBufferOwner;

/// One invocation per output; the dispatch covers ceil(N / 64).
const WORKGROUP: u32 = 64;

/// `out = input . W + b`: input is the flat `K`-length f32 tensor, `W` the
/// row-major `[K, N]` weight matrix (element `(k, n)` at `k * N + n`, matching
/// burn's `[K, N]` layout), `b` the `[N]` bias. One invocation accumulates one
/// output, reading the input buffer the producer left on the device.
const SHADER: &str = r#"
struct Dims { k: u32, n: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= dims.n) { return; }
    var acc = bias[n];
    for (var k = 0u; k < dims.k; k = k + 1u) {
        acc = acc + input[k] * weights[k * dims.n + n];
    }
    out[n] = acc;
}
"#;

/// A single same-padding, stride-1 2D convolution over the `[1, Cin, H, W]`
/// (NCHW) f32 tensor `WgpuPreprocess` emits: `out[oc, y, x] = bias[oc] + sum over
/// (ic, ky, kx) input[ic, y+ky-padH, x+kx-padW] * weights[oc, ic, ky, kx]`, zero
/// outside the input (the standard same-pad convention). Weights are
/// `[Cout, Cin, KH, KW]` row-major; output is `[1, Cout, H, W]`. One invocation
/// per output element, accumulating over the `Cin * KH * KW` receptive field, so
/// the whole convolution stays on the device the producer's buffer lives on. This
/// is the keystone op that lets the GPU-resident chain run an actual CNN layer,
/// not just the matmul `SHADER` above.
const CONV_SHADER: &str = r#"
struct Conv { cin: u32, cout: u32, kh: u32, kw: u32, h: u32, w: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> c: Conv;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = c.cout * c.h * c.w;
    if (idx >= total) { return; }
    let ox = idx % c.w;
    let oy = (idx / c.w) % c.h;
    let oc = idx / (c.w * c.h);
    let pad_h = c.kh / 2u;
    let pad_w = c.kw / 2u;
    var acc = bias[oc];
    for (var ic = 0u; ic < c.cin; ic = ic + 1u) {
        for (var ky = 0u; ky < c.kh; ky = ky + 1u) {
            let iy = i32(oy) + i32(ky) - i32(pad_h);
            if (iy < 0 || iy >= i32(c.h)) { continue; }
            for (var kx = 0u; kx < c.kw; kx = kx + 1u) {
                let ix = i32(ox) + i32(kx) - i32(pad_w);
                if (ix < 0 || ix >= i32(c.w)) { continue; }
                let in_idx = (ic * c.h + u32(iy)) * c.w + u32(ix);
                let w_idx = ((oc * c.cin + ic) * c.kh + ky) * c.kw + kx;
                acc = acc + input[in_idx] * weights[w_idx];
            }
        }
    }
    out[idx] = acc;
}
"#;

/// Activation kind tag in the [`ACT_SHADER`] meta uniform.
const ACT_RELU: u32 = 0;
const ACT_SIGMOID: u32 = 1;

/// An elementwise activation over the flat `n`-length tensor, shape-preserving:
/// `kind` 0 is ReLU (`max(x, 0)`), 1 is the logistic sigmoid. A weightless op,
/// so it binds only (meta, input, out), not the conv/linear weight + bias. ReLU
/// is the nonlinearity that keeps a stack of conv layers from collapsing into a
/// single linear map, the reason a multi-layer CNN needs it between layers.
const ACT_SHADER: &str = r#"
struct Act { n: u32, kind: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> a: Act;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= a.n) { return; }
    let x = input[i];
    if (a.kind == 1u) {
        out[i] = 1.0 / (1.0 + exp(-x));
    } else {
        out[i] = max(x, 0.0);
    }
}
"#;

/// Pooling kind tag in the [`POOL_SHADER`] meta uniform.
const POOL_MAX: u32 = 0;
const POOL_AVG: u32 = 1;

/// A `KH x KW` spatial pool with stride `(SH, SW)`, no padding, over the
/// `[Cin, H, W]` NCHW tensor: `kind` 0 is max-pool, 1 is average-pool. Output is
/// `[Cin, OH, OW]` with `OH = (H - KH) / SH + 1`, `OW = (W - KW) / SW + 1` (the
/// host computes them and passes them in). One invocation per output element,
/// reducing over its `KH x KW` window. The downsampler that shrinks the feature
/// map between CNN stages; a weightless op like the activation.
const POOL_SHADER: &str = r#"
struct Pool { c: u32, h: u32, w: u32, kh: u32, kw: u32, sh: u32, sw: u32, oh: u32, ow: u32, kind: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Pool;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.c * p.oh * p.ow;
    if (idx >= total) { return; }
    let ox = idx % p.ow;
    let oy = (idx / p.ow) % p.oh;
    let ch = idx / (p.ow * p.oh);
    // Top-left of the window is always in-bounds, so it seeds the max reduction.
    let base = (ch * p.h + oy * p.sh) * p.w + ox * p.sw;
    var m = input[base];
    var acc = 0.0;
    for (var ky = 0u; ky < p.kh; ky = ky + 1u) {
        let iy = oy * p.sh + ky;
        for (var kx = 0u; kx < p.kw; kx = kx + 1u) {
            let ix = ox * p.sw + kx;
            let v = input[(ch * p.h + iy) * p.w + ix];
            m = max(m, v);
            acc = acc + v;
        }
    }
    if (p.kind == 1u) {
        out[idx] = acc / f32(p.kh * p.kw);
    } else {
        out[idx] = m;
    }
}
"#;

/// The host reference matching `SHADER`, kept public so the test (and a CPU
/// caller) can compare against the GPU output. `input` is the flat `K`-length
/// f32 tensor; returns the `[N]` logits.
pub fn linear_reference(input: &[f32], weights: &[f32], bias: &[f32]) -> Vec<f32> {
    let n = bias.len();
    let mut out = vec![0f32; n];
    for (col, o) in out.iter_mut().enumerate() {
        let mut acc = bias[col];
        for (row, &x) in input.iter().enumerate() {
            acc += x * weights[row * n + col];
        }
        *o = acc;
    }
    out
}

/// The host reference matching [`CONV_SHADER`]: a same-padding, stride-1 conv over
/// the NCHW `input` (`[Cin, H, W]`), `weights` `[Cout, Cin, KH, KW]`, `bias`
/// `[Cout]`, returning `[Cout, H, W]`. Public so the test compares the GPU conv
/// against it.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_reference(
    input: &[f32],
    cin: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    h: usize,
    w: usize,
    weights: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let (pad_h, pad_w) = (kh / 2, kw / 2);
    let mut out = vec![0f32; cout * h * w];
    for oc in 0..cout {
        for oy in 0..h {
            for ox in 0..w {
                let mut acc = bias[oc];
                for ic in 0..cin {
                    for ky in 0..kh {
                        let iy = oy as isize + ky as isize - pad_h as isize;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..kw {
                            let ix = ox as isize + kx as isize - pad_w as isize;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            let in_idx = (ic * h + iy as usize) * w + ix as usize;
                            let w_idx = ((oc * cin + ic) * kh + ky) * kw + kx;
                            acc += input[in_idx] * weights[w_idx];
                        }
                    }
                }
                out[(oc * h + oy) * w + ox] = acc;
            }
        }
    }
    out
}

/// Host reference for [`ACT_SHADER`] ReLU: `max(x, 0)` elementwise. Public so the
/// chaining test can fold it into a CPU reference.
pub fn relu_reference(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x.max(0.0)).collect()
}

/// Host reference for [`ACT_SHADER`] sigmoid: `1 / (1 + e^-x)` elementwise.
pub fn sigmoid_reference(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

/// Shared host pooling reference: a `kh x kw` window, stride `(sh, sw)`, no pad,
/// over the `[c, h, w]` NCHW tensor, reducing each window by max (`is_max`) or
/// mean. Returns `[c, oh, ow]`.
#[allow(clippy::too_many_arguments)]
fn pool_reference(
    is_max: bool,
    input: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
) -> Vec<f32> {
    let oh = (h - kh) / sh + 1;
    let ow = (w - kw) / sw + 1;
    let mut out = vec![0f32; c * oh * ow];
    for ch in 0..c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut m = f32::NEG_INFINITY;
                let mut acc = 0f32;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let v = input[(ch * h + oy * sh + ky) * w + ox * sw + kx];
                        m = m.max(v);
                        acc += v;
                    }
                }
                out[(ch * oh + oy) * ow + ox] =
                    if is_max { m } else { acc / (kh * kw) as f32 };
            }
        }
    }
    out
}

/// Host reference matching [`POOL_SHADER`] max-pool. Public for the chaining test.
#[allow(clippy::too_many_arguments)]
pub fn maxpool2d_reference(
    input: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
) -> Vec<f32> {
    pool_reference(true, input, c, h, w, kh, kw, sh, sw)
}

/// Host reference matching [`POOL_SHADER`] average-pool.
#[allow(clippy::too_many_arguments)]
pub fn avgpool2d_reference(
    input: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
) -> Vec<f32> {
    pool_reference(false, input, c, h, w, kh, kw, sh, sw)
}

/// f32 slice as little-endian bytes, for `queue.write_buffer` (no bytemuck dep).
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// GPU resources built lazily on the first frame, on the device the producer's
/// buffer was created on. The input buffer is per-frame (it arrives with each
/// frame), so the bind group and output buffer are rebuilt per dispatch; the
/// pipeline and the weight / bias / meta buffers are built once.
#[derive(Debug)]
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    meta_buf: wgpu::Buffer,
    /// `Some` for the weighted ops (linear / conv2d), `None` for the weightless
    /// ones (activation / pooling), whose shader binds only (meta, input, out).
    weight_buf: Option<wgpu::Buffer>,
    bias_buf: Option<wgpu::Buffer>,
}

#[derive(Debug)]
pub struct WgpuInference {
    /// Input tensor shape (`[1, 3, H, W]` for a linear layer, `[1, Cin, H, W]`
    /// for a conv), the caps this element accepts.
    in_shape: Vec<u32>,
    /// Output tensor shape (`[1, N]` logits for linear, `[1, Cout, H, W]` for conv).
    out_shape: Vec<u32>,
    /// Layer weights, packed for the active `shader` (`[K, N]` linear, `[Cout, Cin,
    /// KH, KW]` conv).
    weights: Vec<f32>,
    /// Layer bias (`[N]` linear, `[Cout]` conv).
    bias: Vec<f32>,
    /// Input payload length in bytes, validated against each frame.
    in_bytes: usize,
    /// Output payload length in bytes.
    out_bytes: usize,
    /// Output element count; the dispatch covers `ceil(dispatch_n / 64)`.
    dispatch_n: u32,
    /// The WGSL compute shader for the active op ([`SHADER`] or [`CONV_SHADER`]).
    shader: &'static str,
    /// Pre-packed bytes of the op's uniform meta buffer (`{k, n}` for linear,
    /// `{cin, cout, kh, kw, h, w}` for conv).
    meta: Vec<u8>,
    configured: bool,
    /// Built on the first frame from the producer's device; see [`Gpu`].
    gpu: Option<Gpu>,
    last_caps: Option<Caps>,
    emitted: u64,
    /// When set, emit the output GPU-resident (`MemoryDomain::WgpuBuffer`) for a
    /// downstream GPU consumer; default reads it back to `MemoryDomain::System`.
    gpu_output: bool,
}

impl WgpuInference {
    /// A linear layer over the `[1, 3, H, W]` f32 tensor `WgpuPreprocess` emits.
    /// `weights` is the row-major `[K, N]` matrix (`K = 3 * width * height`) and
    /// `bias` is `[N]`; `N` is `bias.len()`. Matches `BurnInference::linear`'s
    /// contract, so the same weights yield the same logits on either backend.
    /// Fails loud on a dimension mismatch.
    pub fn linear(
        width: u32,
        height: u32,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, G2gError> {
        let n = bias.len();
        let k = 3 * width as usize * height as usize;
        if n == 0 || k == 0 || weights.len() != k * n {
            return Err(G2gError::CapsMismatch);
        }
        let mut meta = vec![0u8; 16];
        meta[0..4].copy_from_slice(&(k as u32).to_le_bytes());
        meta[4..8].copy_from_slice(&(n as u32).to_le_bytes());
        Ok(Self {
            in_shape: vec![1, 3, height, width],
            out_shape: vec![1, n as u32],
            weights,
            bias,
            in_bytes: k * 4,
            out_bytes: n * 4,
            dispatch_n: n as u32,
            shader: SHADER,
            meta,
            configured: false,
            gpu: None,
            last_caps: None,
            emitted: 0,
            gpu_output: false,
        })
    }

    /// A single same-padding, stride-1 2D convolution over the `[1, Cin, H, W]`
    /// (NCHW) f32 tensor `WgpuPreprocess` emits, leaving the `[1, Cout, H, W]`
    /// result on the GPU. `weights` is `[Cout, Cin, KH, KW]` row-major, `bias` is
    /// `[Cout]`; the kernel runs in the [`CONV_SHADER`] compute pass on the
    /// producer's device, no CPU upload. The keystone op for running an actual CNN
    /// layer on the GPU-resident chain. Fails loud on a dimension mismatch.
    /// `conv2d_reference` is the matching host check.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        cin: u32,
        cout: u32,
        kh: u32,
        kw: u32,
        height: u32,
        width: u32,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, G2gError> {
        if cin == 0 || cout == 0 || kh == 0 || kw == 0 || height == 0 || width == 0 {
            return Err(G2gError::CapsMismatch);
        }
        if bias.len() != cout as usize
            || weights.len() != (cout * cin * kh * kw) as usize
        {
            return Err(G2gError::CapsMismatch);
        }
        let in_elems = (cin * height * width) as usize;
        let out_elems = (cout * height * width) as usize;
        let dims = [cin, cout, kh, kw, height, width];
        let mut meta = vec![0u8; 32];
        for (i, d) in dims.iter().enumerate() {
            meta[i * 4..i * 4 + 4].copy_from_slice(&d.to_le_bytes());
        }
        Ok(Self {
            in_shape: vec![1, cin, height, width],
            out_shape: vec![1, cout, height, width],
            weights,
            bias,
            in_bytes: in_elems * 4,
            out_bytes: out_elems * 4,
            dispatch_n: out_elems as u32,
            shader: CONV_SHADER,
            meta,
            configured: false,
            gpu: None,
            last_caps: None,
            emitted: 0,
            gpu_output: false,
        })
    }

    /// Build a [`conv2d`](Self::conv2d) layer from trained weights in a parsed
    /// safetensors file (M262): reads the `[Cout, Cin, KH, KW]` weight tensor and
    /// the `[Cout]` bias by name, infers the kernel dimensions from the weight
    /// shape, and takes the spatial input size (`height`, `width`) from the
    /// runtime caps. Importing a different checkpoint is "parse a different file";
    /// the architecture stays this compiled element. Fails loud on a missing
    /// tensor, a non-F32 / non-4D weight, or a `[Cout]`-mismatched bias.
    pub fn conv2d_from_safetensors(
        st: &crate::safetensors::SafeTensors<'_>,
        weight_key: &str,
        bias_key: &str,
        height: u32,
        width: u32,
    ) -> Result<Self, G2gError> {
        let wt = st.get(weight_key).map_err(|_| G2gError::CapsMismatch)?;
        let [cout, cin, kh, kw] = match wt.shape {
            [a, b, c, d] => [*a as u32, *b as u32, *c as u32, *d as u32],
            _ => return Err(G2gError::CapsMismatch),
        };
        let weights = wt.to_f32().map_err(|_| G2gError::CapsMismatch)?;
        let bias = st
            .get(bias_key)
            .and_then(|b| b.to_f32())
            .map_err(|_| G2gError::CapsMismatch)?;
        Self::conv2d(cin, cout, kh, kw, height, width, weights, bias)
    }

    /// Shared builder for the weightless ops (activation, pooling): no weight /
    /// bias tensor, so `ensure_gpu` skips those buffers and `dispatch` binds the
    /// 3-entry (meta, input, out) layout. Byte sizes and the dispatch count come
    /// from the shapes (`product` of the dims).
    fn new_weightless(
        in_shape: Vec<u32>,
        out_shape: Vec<u32>,
        shader: &'static str,
        meta: Vec<u8>,
    ) -> Self {
        let in_elems: u32 = in_shape.iter().product();
        let out_elems: u32 = out_shape.iter().product();
        Self {
            in_shape,
            out_shape,
            weights: Vec::new(),
            bias: Vec::new(),
            in_bytes: in_elems as usize * 4,
            out_bytes: out_elems as usize * 4,
            dispatch_n: out_elems,
            shader,
            meta,
            configured: false,
            gpu: None,
            last_caps: None,
            emitted: 0,
            gpu_output: false,
        }
    }

    /// An elementwise activation (`kind`) over the `[1, C, H, W]` tensor,
    /// shape-preserving. Runs [`ACT_SHADER`] on the producer's device.
    fn activation(kind: u32, channels: u32, height: u32, width: u32) -> Result<Self, G2gError> {
        if channels == 0 || height == 0 || width == 0 {
            return Err(G2gError::CapsMismatch);
        }
        let n = channels * height * width;
        let mut meta = vec![0u8; 16];
        meta[0..4].copy_from_slice(&n.to_le_bytes());
        meta[4..8].copy_from_slice(&kind.to_le_bytes());
        let shape = vec![1, channels, height, width];
        Ok(Self::new_weightless(shape.clone(), shape, ACT_SHADER, meta))
    }

    /// A ReLU activation over the `[1, C, H, W]` tensor (`max(x, 0)`,
    /// elementwise). The nonlinearity that goes between conv layers so a stack of
    /// them does not collapse into a single linear map. Weightless: no upload.
    pub fn relu(channels: u32, height: u32, width: u32) -> Result<Self, G2gError> {
        Self::activation(ACT_RELU, channels, height, width)
    }

    /// A logistic-sigmoid activation over the `[1, C, H, W]` tensor
    /// (`1 / (1 + e^-x)`, elementwise).
    pub fn sigmoid(channels: u32, height: u32, width: u32) -> Result<Self, G2gError> {
        Self::activation(ACT_SIGMOID, channels, height, width)
    }

    /// A `kh x kw` spatial pool (`kind`), stride `(sh, sw)`, no padding, over the
    /// `[1, C, H, W]` tensor, leaving `[1, C, OH, OW]` on the GPU. Runs
    /// [`POOL_SHADER`]. Fails loud on a zero dim or a window larger than the input.
    #[allow(clippy::too_many_arguments)]
    fn pool(
        kind: u32,
        channels: u32,
        height: u32,
        width: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
    ) -> Result<Self, G2gError> {
        if channels == 0 || height == 0 || width == 0 || kh == 0 || kw == 0 || sh == 0 || sw == 0 {
            return Err(G2gError::CapsMismatch);
        }
        if kh > height || kw > width {
            return Err(G2gError::CapsMismatch);
        }
        let oh = (height - kh) / sh + 1;
        let ow = (width - kw) / sw + 1;
        let dims = [channels, height, width, kh, kw, sh, sw, oh, ow, kind];
        let mut meta = vec![0u8; 48];
        for (i, d) in dims.iter().enumerate() {
            meta[i * 4..i * 4 + 4].copy_from_slice(&d.to_le_bytes());
        }
        Ok(Self::new_weightless(
            vec![1, channels, height, width],
            vec![1, channels, oh, ow],
            POOL_SHADER,
            meta,
        ))
    }

    /// A `kh x kw` stride-`(sh, sw)` max-pool over the `[1, C, H, W]` tensor (the
    /// CNN downsampler), output `[1, C, OH, OW]`. `maxpool2d_reference` matches it.
    #[allow(clippy::too_many_arguments)]
    pub fn maxpool2d(
        channels: u32,
        height: u32,
        width: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
    ) -> Result<Self, G2gError> {
        Self::pool(POOL_MAX, channels, height, width, kh, kw, sh, sw)
    }

    /// A `kh x kw` stride-`(sh, sw)` average-pool over the `[1, C, H, W]` tensor,
    /// output `[1, C, OH, OW]`. `avgpool2d_reference` matches it.
    #[allow(clippy::too_many_arguments)]
    pub fn avgpool2d(
        channels: u32,
        height: u32,
        width: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
    ) -> Result<Self, G2gError> {
        Self::pool(POOL_AVG, channels, height, width, kh, kw, sh, sw)
    }

    /// Whether the active op uploads a weight + bias tensor (linear / conv2d) or
    /// is weightless (activation / pooling). Drives the buffer set and bind-group
    /// layout.
    fn is_weighted(&self) -> bool {
        !self.weights.is_empty()
    }

    /// Emit the logits GPU-resident (`MemoryDomain::WgpuBuffer`) instead of
    /// reading them back to system memory, so a downstream GPU consumer (a GPU
    /// softmax / argmax, say) reads them on-device. Default off.
    pub fn with_gpu_output(mut self) -> Self {
        self.gpu_output = true;
        self
    }

    /// Count of logit `DataFrame`s pushed downstream. Useful in tests.
    pub fn inferred_count(&self) -> u64 {
        self.emitted
    }

    fn supported_input(&self) -> Caps {
        Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape(self.in_shape.clone()),
            layout: TensorLayout::Nchw,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape(self.out_shape.clone()),
            layout: TensorLayout::Nchw,
        }
    }

    /// Build the pipeline and upload the weight / bias / meta buffers on the
    /// device the producer's buffer lives on. Idempotent: built once, on the
    /// first frame, because the device is only known once a frame arrives.
    fn ensure_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.gpu.is_some() {
            return;
        }
        let meta_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("infer-meta"),
            size: self.meta.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&meta_buf, 0, &self.meta);

        // Weightless ops (activation / pooling) have no weight or bias tensor; their
        // shader binds only (meta, input, out), so these buffers stay `None`.
        let (weight_buf, bias_buf) = if self.is_weighted() {
            let weight_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("infer-weights"),
                size: (self.weights.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&weight_buf, 0, &f32_bytes(&self.weights));

            let bias_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("infer-bias"),
                size: (self.bias.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&bias_buf, 0, &f32_bytes(&self.bias));
            (Some(weight_buf), Some(bias_buf))
        } else {
            (None, None)
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpu-infer"),
            source: wgpu::ShaderSource::Wgsl(self.shader.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wgpu-infer"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        self.gpu = Some(Gpu {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            meta_buf,
            weight_buf,
            bias_buf,
        });
    }

    /// Bind the producer's input buffer directly, run the matmul into a fresh
    /// per-frame output buffer, and return it for the caller to read back or to
    /// hand downstream GPU-resident. No fence: the dispatch is submitted on the
    /// producer's queue, after the producer's own submission.
    fn dispatch(&self, input: &wgpu::Buffer) -> Result<wgpu::Buffer, G2gError> {
        let gpu = self.gpu.as_ref().ok_or(G2gError::NotConfigured)?;

        // GPU-output mode needs COPY_SRC so a downstream consumer (or a CPU
        // read-back) can copy it out; the CPU path needs it for the staging copy.
        let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("infer-logits"),
            size: self.out_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // The bindings follow the active shader's layout: weighted ops bind
        // (meta=0, input=1, weights=2, bias=3, out=4); weightless ops bind
        // (meta=0, input=1, out=2). The pipeline's auto-derived layout matches.
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: gpu.meta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
        ];
        match (&gpu.weight_buf, &gpu.bias_buf) {
            (Some(weights), Some(bias)) => {
                entries.push(wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weights.as_entire_binding(),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bias.as_entire_binding(),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                });
            }
            _ => entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            }),
        }

        let layout = gpu.pipeline.get_bind_group_layout(0);
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-infer-binding"),
            layout: &layout,
            entries: &entries,
        });

        let mut encoder =
            gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-linear"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.dispatch_n.div_ceil(WORKGROUP), 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        Ok(out_buf)
    }

    /// Read the logits buffer back to little-endian f32 bytes (the
    /// `OrtInference` / `BurnInference` output format), paying the GPU->CPU copy
    /// the System path owes.
    fn read_back(&self, logits: &wgpu::Buffer) -> Result<Box<[u8]>, G2gError> {
        let gpu = self.gpu.as_ref().ok_or(G2gError::NotConfigured)?;
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("infer-readback"),
            size: self.out_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(logits, 0, &staging, 0, self.out_bytes as u64);
        gpu.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        rx.recv()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let bytes = slice.get_mapped_range().to_vec().into_boxed_slice();
        staging.unmap();
        Ok(bytes)
    }
}

impl AsyncElement for WgpuInference {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.supported_input())
    }

    /// Native `DerivedOutput`: the `[1, 3, H, W]` tensor in, the `[1, N]` logits
    /// out. Non-matching input yields an empty set, rejected at solve.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let supported = self.supported_input();
        let out = self.output_caps();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            if input.intersect(&supported).is_ok() {
                CapsSet::one(out.clone())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Validate caps cheaply; the GPU pipeline is built lazily on the first
        // frame, once the producer's device is known.
        absolute_caps.intersect(&self.supported_input())?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
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
                    // The whole point of this element: the tensor is already on
                    // the GPU. A System frame is the CPU path's job (BurnInference).
                    let MemoryDomain::WgpuBuffer(owned) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    if owned.len != self.in_bytes {
                        return Err(G2gError::CapsMismatch);
                    }
                    // Recover the producer's device / queue / buffer. A foreign
                    // owner (some other producer's keep-alive) we cannot bind.
                    let owner = owned
                        .keep_alive()
                        .as_any()
                        .downcast_ref::<WgpuBufferOwner>()
                        .ok_or(G2gError::UnsupportedDomain)?;

                    self.ensure_gpu(owner.device(), owner.queue());
                    let logits = self.dispatch(owner.buffer())?;

                    let new_caps = self.output_caps();
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                        self.last_caps = Some(new_caps);
                    }

                    let domain = if self.gpu_output {
                        let gpu = self.gpu.as_ref().ok_or(G2gError::NotConfigured)?;
                        // Box the logits as the same shared owner the preprocess
                        // stage uses, so a downstream downcast is identical
                        // whichever stage produced the buffer.
                        let owner = WgpuBufferOwner::new(
                            gpu.device.clone(),
                            gpu.queue.clone(),
                            logits,
                            self.out_bytes,
                        );
                        MemoryDomain::WgpuBuffer(OwnedWgpuBuffer::new(
                            self.out_bytes,
                            std::sync::Arc::new(owner),
                        ))
                    } else {
                        MemoryDomain::System(SystemSlice::from_boxed(self.read_back(&logits)?))
                    };

                    let tensor = Frame {
                        domain,
                        // per-frame inference: inherit source timing so latency
                        // stays traceable.
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(tensor)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // geometry is pinned at construction; anything else is a
                    // hard error.
                    c.intersect(&self.supported_input())?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is a timing marker: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // stateless per-frame inference: nothing to drain.
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

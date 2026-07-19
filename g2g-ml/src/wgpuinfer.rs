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

/// Inference-mode batch-norm: a per-channel affine `out = scale[c]*in + shift[c]`,
/// where `scale`/`shift` are folded from `gamma/beta/running_mean/running_var` on
/// the host ([`WgpuInference::batch_norm`]). A weighted op reusing the (meta,
/// input, weights=scale, bias=shift, out) binding of the conv/linear layout, so
/// no new bind-group shape is needed. `c` is the channel count, `hw = H*W`.
const BN_SHADER: &str = r#"
struct Bn { c: u32, hw: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> b: Bn;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> shift: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = b.c * b.hw;
    if (i >= total) { return; }
    let ch = i / b.hw;
    out[i] = scale[ch] * input[i] + shift[ch];
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

/// Elementwise add of two equal-length tensors: `out[i] = a[i] + b[i]`. The
/// residual/skip primitive, a two-input op (unlike every other op here, which is
/// single-input): `a` is the running tensor, `b` the saved skip tensor from an
/// earlier layer. Binds (meta{n}, a, b, out); used only inside a
/// [`ResidualStack`], which supplies both operands.
const ADD_SHADER: &str = r#"
struct Add { n: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<uniform> c: Add;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= c.n) { return; }
    out[i] = a[i] + b[i];
}
"#;

/// Host reference for [`ADD_SHADER`]: elementwise add of two equal-length
/// tensors. Public so a residual block can be checked on the CPU without a GPU.
pub fn add_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

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
                out[(ch * oh + oy) * ow + ox] = if is_max { m } else { acc / (kh * kw) as f32 };
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

/// Host reference for inference-mode batch normalization over an `[C, hw]`
/// (row-major, `hw = H*W`) tensor: per channel `y = gamma*(x - mean)/sqrt(var +
/// eps) + beta`, folded to the affine `scale*x + shift` the GPU pass applies.
/// Matches [`WgpuInference::batch_norm`].
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_reference(
    input: &[f32],
    channels: usize,
    hw: usize,
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; channels * hw];
    for c in 0..channels {
        let scale = gamma[c] / (var[c] + eps).sqrt();
        let shift = beta[c] - mean[c] * scale;
        for i in 0..hw {
            out[c * hw + i] = scale * input[c * hw + i] + shift;
        }
    }
    out
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
    /// A two-input elementwise op (the residual [`ADD_SHADER`]): its `dispatch`
    /// binds a second input (the skip tensor) rather than the single-input layout.
    /// Only [`WgpuInference::add`] sets it; used inside a [`ResidualStack`].
    binary: bool,
}

/// One layer in a whole-model stack imported from a single safetensors file by
/// [`WgpuInference::stack_from_safetensors`]. The architecture (this list) stays
/// compiled; only the weights are the file, so importing a different checkpoint
/// is "parse a different file". Tensor names follow the PyTorch convention
/// (`conv1.weight`, `bn1.running_mean`, `fc.weight`, ...).
#[derive(Debug, Clone)]
pub enum StackLayer {
    /// Same-pad stride-1 conv; reads `{name}.weight` `[Cout,Cin,KH,KW]` +
    /// `{name}.bias` `[Cout]`. The channel count updates to `Cout`.
    Conv2d { name: String },
    /// Inference batch-norm; reads `{name}.weight`(gamma), `.bias`(beta),
    /// `.running_mean`, `.running_var`, each `[C]`.
    BatchNorm { name: String, eps: f32 },
    /// ReLU nonlinearity (shape-preserving).
    Relu,
    /// Logistic sigmoid (shape-preserving).
    Sigmoid,
    /// Max pool; updates the spatial dims to `((H-kh)/sh+1, (W-kw)/sw+1)`.
    MaxPool2d { kh: u32, kw: u32, sh: u32, sw: u32 },
    /// Average pool; updates the spatial dims like [`Self::MaxPool2d`].
    AvgPool2d { kh: u32, kw: u32, sh: u32, sw: u32 },
    /// Global average pool over the full spatial extent, giving `[1, C, 1, 1]`.
    GlobalAvgPool,
    /// Fully-connected head; reads `{name}.weight` `[K, N]` (row-major, input-
    /// major to match the matmul shader) + `{name}.bias` `[N]`. `K` must equal the
    /// running `C*H*W`; the output shape becomes `[1, N]`.
    Linear { name: String },
    /// Save the running tensor into a named skip register (shape-preserving,
    /// no GPU op) so a later [`Self::AddSkip`] can add it back, expressing the
    /// `y = f(x) + x` residual topology a straight chain cannot. Only valid in a
    /// [`ResidualStack`] (`residual_stack_from_safetensors`).
    SaveSkip { slot: String },
    /// Add the tensor saved under `slot` (by an earlier [`Self::SaveSkip`]) to the
    /// running tensor, elementwise (the skip connection's join). The saved tensor's
    /// shape must equal the running `[C, H, W]`; shape-preserving.
    AddSkip { slot: String },
}

/// One step in a [`ResidualStack`]: a single-input op, a skip-save marker, or a
/// two-input add that joins a saved skip tensor.
#[derive(Debug)]
enum ResidualStep {
    /// A single-input GPU op (conv / bn / act / pool / linear).
    Op(WgpuInference),
    /// Save the running tensor under this slot for a later [`Self::Add`].
    Save(String),
    /// An elementwise-add op that joins the tensor saved under this slot.
    Add(WgpuInference, String),
}

/// A model imported with skip/residual connections
/// ([`WgpuInference::residual_stack_from_safetensors`]). Unlike the flat
/// `Vec<WgpuInference>` a straight chain produces, this carries the skip topology
/// (save / add markers) and runs the whole thing on the GPU with a single
/// read-back via [`run`](Self::run), so a ResNet-style block runs on-device with
/// no GPU->CPU round-trip between layers.
#[derive(Debug)]
pub struct ResidualStack {
    steps: Vec<ResidualStep>,
}

impl ResidualStack {
    /// Run the whole model on the GPU-resident `input` (a `MemoryDomain::WgpuBuffer`
    /// f32 NCHW tensor, as `WgpuPreprocess::with_gpu_output` emits), adopting the
    /// producer's device / queue, and read the final tensor back to
    /// `MemoryDomain::System`. Every intermediate (and every saved skip tensor)
    /// stays on the device; only the final tensor is read back. Fails loud on a
    /// non-GPU input, a foreign buffer owner, or an add against a not-yet-saved slot.
    pub fn run(&mut self, input: Frame) -> Result<Frame, G2gError> {
        use std::collections::BTreeMap;
        let MemoryDomain::WgpuBuffer(owned) = &input.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let owner = owned
            .keep_alive()
            .as_any()
            .downcast_ref::<WgpuBufferOwner>()
            .ok_or(G2gError::UnsupportedDomain)?;
        let (device, queue) = (owner.device().clone(), owner.queue().clone());

        // The running GPU buffer, threaded op to op; saved skip tensors keyed by slot.
        let mut cur: wgpu::Buffer = owner.buffer().clone();
        let mut saved: BTreeMap<String, wgpu::Buffer> = BTreeMap::new();
        // out_bytes of the op that produced `cur`, for the final read-back.
        let mut cur_bytes = owned.len;
        // Index of the last op that ran (for its read_back / gpu handle).
        let mut last_op: Option<usize> = None;

        for i in 0..self.steps.len() {
            match &mut self.steps[i] {
                ResidualStep::Op(op) => {
                    op.ensure_gpu(&device, &queue);
                    cur = op.dispatch(&cur)?;
                    cur_bytes = op.out_bytes;
                    last_op = Some(i);
                }
                ResidualStep::Save(slot) => {
                    saved.insert(slot.clone(), cur.clone());
                }
                ResidualStep::Add(op, slot) => {
                    let skip = saved.get(slot).ok_or(G2gError::CapsMismatch)?.clone();
                    op.ensure_gpu(&device, &queue);
                    cur = op.dispatch_binary(&cur, &skip)?;
                    cur_bytes = op.out_bytes;
                    last_op = Some(i);
                }
            }
        }

        // Read the final tensor back through the op that produced it (it owns the
        // matching device / out_bytes). A stack with no op at all is a misuse.
        let idx = last_op.ok_or(G2gError::NotConfigured)?;
        let bytes = match &self.steps[idx] {
            ResidualStep::Op(op) | ResidualStep::Add(op, _) => op.read_back(&cur)?,
            ResidualStep::Save(_) => return Err(G2gError::NotConfigured),
        };
        debug_assert_eq!(
            bytes.len(),
            cur_bytes,
            "read-back length tracks the final op"
        );
        Ok(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes)),
            timing: input.timing,
            sequence: 0,
            meta: Default::default(),
        })
    }
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
        Self::linear_shaped(vec![1, 3, height, width], weights, bias)
    }

    /// A linear layer over a flat input of any NCHW shape (`K = product(in_shape)`),
    /// the generalization of [`linear`](Self::linear) used for a fully-connected
    /// head after a global pool (`[1, C, 1, 1]`). `weights` is row-major `[K, N]`,
    /// `bias` is `[N]`. Fails loud on a dimension mismatch.
    pub fn linear_shaped(
        in_shape: Vec<u32>,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, G2gError> {
        let n = bias.len();
        // The shape may come from a safetensors file: bound the rank (tensor
        // caps carry at most MAX_TENSOR_RANK dims) and fold with checked
        // arithmetic.
        if TensorShape::from_slice(&in_shape).is_none() {
            return Err(G2gError::CapsMismatch);
        }
        let k = in_shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
            .and_then(|v| usize::try_from(v).ok())
            .ok_or(G2gError::CapsMismatch)?;
        if n == 0 || k == 0 || weights.len() != k.checked_mul(n).ok_or(G2gError::CapsMismatch)? {
            return Err(G2gError::CapsMismatch);
        }
        let mut meta = vec![0u8; 16];
        meta[0..4].copy_from_slice(&(k as u32).to_le_bytes());
        meta[4..8].copy_from_slice(&(n as u32).to_le_bytes());
        Ok(Self {
            in_shape,
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
            binary: false,
        })
    }

    /// Inference-mode batch normalization over the `[1, C, H, W]` tensor: per
    /// channel `y = gamma*(x - mean)/sqrt(var + eps) + beta`, the training-time
    /// `running_mean`/`running_var` frozen into a per-channel affine `scale*x +
    /// shift` computed here on the host (so the GPU pass is a cheap multiply-add
    /// reusing the conv/linear weight+bias buffers: `scale` as weights, `shift` as
    /// bias). The op nearly every real CNN needs between conv and activation.
    /// `gamma`/`beta`/`mean`/`var` are each `[C]`. `batch_norm_reference` matches
    /// it. Fails loud on a dimension mismatch.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm(
        channels: u32,
        height: u32,
        width: u32,
        gamma: Vec<f32>,
        beta: Vec<f32>,
        mean: Vec<f32>,
        var: Vec<f32>,
        eps: f32,
    ) -> Result<Self, G2gError> {
        if channels == 0 || height == 0 || width == 0 {
            return Err(G2gError::CapsMismatch);
        }
        let c = channels as usize;
        if [gamma.len(), beta.len(), mean.len(), var.len()]
            .iter()
            .any(|&l| l != c)
        {
            return Err(G2gError::CapsMismatch);
        }
        let hw = (height as u64)
            .checked_mul(width as u64)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or(G2gError::CapsMismatch)?;
        // Fold the running stats into a per-channel scale/shift on the host.
        let mut scale = vec![0f32; c];
        let mut shift = vec![0f32; c];
        for i in 0..c {
            let s = gamma[i] / (var[i] + eps).sqrt();
            scale[i] = s;
            shift[i] = beta[i] - mean[i] * s;
        }
        let elems = (channels as u64 * hw as u64) as usize;
        let mut meta = vec![0u8; 16];
        meta[0..4].copy_from_slice(&channels.to_le_bytes());
        meta[4..8].copy_from_slice(&hw.to_le_bytes());
        Ok(Self {
            in_shape: vec![1, channels, height, width],
            out_shape: vec![1, channels, height, width],
            weights: scale,
            bias: shift,
            in_bytes: elems * 4,
            out_bytes: elems * 4,
            dispatch_n: elems as u32,
            shader: BN_SHADER,
            meta,
            configured: false,
            gpu: None,
            last_caps: None,
            emitted: 0,
            gpu_output: false,
            binary: false,
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
        // The dims may come from an untrusted safetensors shape
        // (`conv2d_from_safetensors`), so fold every element-count product with
        // checked u64 arithmetic: an overflow fails the build instead of
        // panicking (debug) or wrapping to a value that admits a mismatched
        // weight buffer or undersizes the GPU buffers / dispatch count.
        let prod = |dims: &[u32]| -> Option<usize> {
            dims.iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
                .and_then(|n| usize::try_from(n).ok())
        };
        let weight_len = prod(&[cout, cin, kh, kw]).ok_or(G2gError::CapsMismatch)?;
        if bias.len() != cout as usize || weights.len() != weight_len {
            return Err(G2gError::CapsMismatch);
        }
        let in_elems = prod(&[cin, height, width]).ok_or(G2gError::CapsMismatch)?;
        let out_elems = prod(&[cout, height, width]).ok_or(G2gError::CapsMismatch)?;
        let in_bytes = in_elems.checked_mul(4).ok_or(G2gError::CapsMismatch)?;
        let out_bytes = out_elems.checked_mul(4).ok_or(G2gError::CapsMismatch)?;
        let dispatch_n = u32::try_from(out_elems).map_err(|_| G2gError::CapsMismatch)?;
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
            in_bytes,
            out_bytes,
            dispatch_n,
            shader: CONV_SHADER,
            meta,
            configured: false,
            gpu: None,
            last_caps: None,
            emitted: 0,
            gpu_output: false,
            binary: false,
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

    /// Import a whole multi-layer model as a chain of GPU ops from `specs` + one
    /// safetensors file, the generalization of [`conv2d_from_safetensors`] from a
    /// single layer to a full stack. Walks `specs`, pulling each layer's tensors
    /// by name and tracking the running `[C, H, W]` shape so every layer's dims
    /// follow the previous layer's output; the input shape is `[in_c, in_h, in_w]`.
    /// Every op but the last is set GPU-resident (`with_gpu_output`), so the whole
    /// model runs on-device with a single read-back at the end. Fails loud on a
    /// missing tensor or a shape that does not line up. Returns the chain in order;
    /// feed a `[1, in_c, in_h, in_w]` tensor to the first and read the last.
    pub fn stack_from_safetensors(
        specs: &[StackLayer],
        st: &crate::safetensors::SafeTensors<'_>,
        in_c: u32,
        in_h: u32,
        in_w: u32,
    ) -> Result<Vec<WgpuInference>, G2gError> {
        let (mut c, mut h, mut w) = (in_c, in_h, in_w);
        let mut chain: Vec<WgpuInference> = Vec::with_capacity(specs.len());
        for spec in specs {
            match spec {
                // Skip/residual layers describe a non-linear topology, which a flat
                // Vec cannot express: build with `residual_stack_from_safetensors`.
                StackLayer::SaveSkip { .. } | StackLayer::AddSkip { .. } => {
                    return Err(G2gError::CapsMismatch);
                }
                _ => chain.push(Self::build_layer(spec, st, &mut c, &mut h, &mut w)?),
            }
        }
        // Keep every intermediate on the GPU; only the final layer reads back.
        let last = chain.len().saturating_sub(1);
        for (i, op) in chain.iter_mut().enumerate() {
            if i != last {
                op.gpu_output = true;
            }
        }
        Ok(chain)
    }

    /// Build one non-skip [`StackLayer`] from `st`, given the running `(c, h, w)`
    /// shape (updated in place to the layer's output). Shared by the linear
    /// [`stack_from_safetensors`] and the [`ResidualStack`] builder so the two
    /// stay in lockstep. `SaveSkip` / `AddSkip` are not layers here (they carry
    /// no weights and are handled by the residual builder's control flow).
    fn build_layer(
        spec: &StackLayer,
        st: &crate::safetensors::SafeTensors<'_>,
        c: &mut u32,
        h: &mut u32,
        w: &mut u32,
    ) -> Result<Self, G2gError> {
        let vec_by_name = |name: &str| -> Result<Vec<f32>, G2gError> {
            st.get(name)
                .and_then(|t| t.to_f32())
                .map_err(|_| G2gError::CapsMismatch)
        };
        Ok(match spec {
            StackLayer::Conv2d { name } => {
                let layer = Self::conv2d_from_safetensors(
                    st,
                    &format!("{name}.weight"),
                    &format!("{name}.bias"),
                    *h,
                    *w,
                )?;
                *c = layer.out_shape[1]; // Cout
                layer
            }
            StackLayer::BatchNorm { name, eps } => Self::batch_norm(
                *c,
                *h,
                *w,
                vec_by_name(&format!("{name}.weight"))?,
                vec_by_name(&format!("{name}.bias"))?,
                vec_by_name(&format!("{name}.running_mean"))?,
                vec_by_name(&format!("{name}.running_var"))?,
                *eps,
            )?,
            StackLayer::Relu => Self::relu(*c, *h, *w)?,
            StackLayer::Sigmoid => Self::sigmoid(*c, *h, *w)?,
            StackLayer::MaxPool2d { kh, kw, sh, sw } => {
                let layer = Self::maxpool2d(*c, *h, *w, *kh, *kw, *sh, *sw)?;
                (*h, *w) = (layer.out_shape[2], layer.out_shape[3]);
                layer
            }
            StackLayer::AvgPool2d { kh, kw, sh, sw } => {
                let layer = Self::avgpool2d(*c, *h, *w, *kh, *kw, *sh, *sw)?;
                (*h, *w) = (layer.out_shape[2], layer.out_shape[3]);
                layer
            }
            StackLayer::GlobalAvgPool => {
                let layer = Self::avgpool2d(*c, *h, *w, *h, *w, *h, *w)?;
                (*h, *w) = (1, 1);
                layer
            }
            StackLayer::Linear { name } => {
                let wt = st
                    .get(&format!("{name}.weight"))
                    .map_err(|_| G2gError::CapsMismatch)?;
                let [k, n] = match wt.shape {
                    [a, b] => [*a as u32, *b as u32],
                    _ => return Err(G2gError::CapsMismatch),
                };
                // K must equal the flattened running tensor (input-major [K, N]).
                let k_expected = (*c as u64) * (*h as u64) * (*w as u64);
                if k as u64 != k_expected {
                    return Err(G2gError::CapsMismatch);
                }
                let weights = wt.to_f32().map_err(|_| G2gError::CapsMismatch)?;
                let bias = vec_by_name(&format!("{name}.bias"))?;
                let layer = Self::linear_shaped(vec![1, *c, *h, *w], weights, bias)?;
                (*c, *h, *w) = (n, 1, 1);
                layer
            }
            StackLayer::SaveSkip { .. } | StackLayer::AddSkip { .. } => {
                return Err(G2gError::CapsMismatch);
            }
        })
    }

    /// Import a whole model **with skip/residual connections** as a GPU-resident
    /// [`ResidualStack`], the non-linear-topology generalization of
    /// [`stack_from_safetensors`]. Beyond the ordinary layers, a `SaveSkip { slot }`
    /// records the running tensor under a name and a later `AddSkip { slot }` adds
    /// it back elementwise (a ResNet-style `y = f(x) + x` block). Tracks the running
    /// `[C, H, W]` shape exactly like the linear builder; an `AddSkip` requires the
    /// saved tensor's shape to match. Fails loud on a missing tensor, an unknown
    /// slot, or a shape that does not line up.
    pub fn residual_stack_from_safetensors(
        specs: &[StackLayer],
        st: &crate::safetensors::SafeTensors<'_>,
        in_c: u32,
        in_h: u32,
        in_w: u32,
    ) -> Result<ResidualStack, G2gError> {
        use std::collections::{BTreeMap, BTreeSet};
        let (mut c, mut h, mut w) = (in_c, in_h, in_w);
        // The shape the running tensor had when each slot was saved, so an AddSkip
        // both validates against it and builds an add op of the right size.
        let mut slot_shapes: BTreeMap<String, (u32, u32, u32)> = BTreeMap::new();
        let mut declared: BTreeSet<String> = BTreeSet::new();
        let mut steps: Vec<ResidualStep> = Vec::with_capacity(specs.len());
        for spec in specs {
            match spec {
                StackLayer::SaveSkip { slot } => {
                    slot_shapes.insert(slot.clone(), (c, h, w));
                    declared.insert(slot.clone());
                    steps.push(ResidualStep::Save(slot.clone()));
                }
                StackLayer::AddSkip { slot } => {
                    // The saved tensor must exist and match the running shape.
                    let &(sc, sh, sw) = slot_shapes.get(slot).ok_or(G2gError::CapsMismatch)?;
                    if (sc, sh, sw) != (c, h, w) {
                        return Err(G2gError::CapsMismatch);
                    }
                    steps.push(ResidualStep::Add(Self::add(c, h, w)?, slot.clone()));
                }
                _ => steps.push(ResidualStep::Op(Self::build_layer(
                    spec, st, &mut c, &mut h, &mut w,
                )?)),
            }
        }
        Ok(ResidualStack { steps })
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
            binary: false,
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

    /// An elementwise add over the `[1, C, H, W]` tensor: `out = a + b`, where `a`
    /// is the running tensor and `b` a same-shape skip tensor from an earlier
    /// layer (the residual/skip primitive). A two-input op, run only inside a
    /// [`ResidualStack`] (which supplies the skip operand); `add_reference`
    /// matches it. Fails loud on a zero dimension.
    pub fn add(channels: u32, height: u32, width: u32) -> Result<Self, G2gError> {
        if channels == 0 || height == 0 || width == 0 {
            return Err(G2gError::CapsMismatch);
        }
        let n = (channels as u64)
            .checked_mul(height as u64)
            .and_then(|v| v.checked_mul(width as u64))
            .and_then(|v| u32::try_from(v).ok())
            .ok_or(G2gError::CapsMismatch)?;
        let shape = vec![1, channels, height, width];
        let mut meta = vec![0u8; 16];
        meta[0..4].copy_from_slice(&n.to_le_bytes());
        let mut op = Self::new_weightless(shape.clone(), shape, ADD_SHADER, meta);
        op.binary = true;
        Ok(op)
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
            shape: TensorShape::from_slice(&self.in_shape).expect("rank validated at construction"),
            layout: TensorLayout::Nchw,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::from_slice(&self.out_shape)
                .expect("rank validated at construction"),
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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu.meta_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.as_entire_binding(),
            },
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

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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

    /// Dispatch the two-input elementwise [`ADD_SHADER`]: `out = a + b`, binding
    /// (meta=0, a=1, b=2, out=3). `a` is the running tensor, `b` the skip tensor;
    /// both must be `out_bytes` long. Used by [`ResidualStack::run`] for a skip
    /// connection. Fails if the op was not built by [`WgpuInference::add`].
    fn dispatch_binary(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
    ) -> Result<wgpu::Buffer, G2gError> {
        let gpu = self.gpu.as_ref().ok_or(G2gError::NotConfigured)?;
        if !self.binary {
            return Err(G2gError::NotConfigured);
        }
        let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("infer-add"),
            size: self.out_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu.meta_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
        ];
        let layout = gpu.pipeline.get_bind_group_layout(0);
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-infer-add-binding"),
            layout: &layout,
            entries: &entries,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-add"),
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
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(logits, 0, &staging, 0, self.out_bytes as u64);
        gpu.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
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
        staging.unmap();
        Ok(bytes)
    }
}

impl AsyncElement for WgpuInference {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
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
                        out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
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
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

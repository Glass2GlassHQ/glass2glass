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
//! Input is `Caps::Tensor{F32,[1,3,H,W],Nchw}` in `MemoryDomain::WgpuBuffer`
//! (anything else is `UnsupportedDomain`); output is the `[1, N]` logits, read
//! back to `MemoryDomain::System` by default or left GPU-resident with
//! [`with_gpu_output`](WgpuInference::with_gpu_output) for a downstream GPU
//! consumer. Richer layers and a trained-weight loader are follow-ups, same as
//! the burn path; the `AsyncElement` / caps contract here is what they slot into.

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
    weight_buf: wgpu::Buffer,
    bias_buf: wgpu::Buffer,
}

#[derive(Debug)]
pub struct WgpuInference {
    width: u32,
    height: u32,
    num_outputs: usize,
    /// Row-major `[K, N]` weight matrix, `K = 3 * W * H`.
    weights: Vec<f32>,
    /// `[N]` bias.
    bias: Vec<f32>,
    /// Input payload length in bytes (`K * 4`), validated against each frame.
    in_bytes: usize,
    /// Output payload length in bytes (`N * 4`).
    out_bytes: usize,
    configured: bool,
    /// Built on the first frame from the producer's device; see [`Gpu`].
    gpu: Option<Gpu>,
    last_caps: Option<Caps>,
    emitted: u64,
    /// When set, emit the logits GPU-resident (`MemoryDomain::WgpuBuffer`) for a
    /// downstream GPU consumer; default reads them back to `MemoryDomain::System`.
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
        let num_outputs = bias.len();
        let k = 3 * width as usize * height as usize;
        if num_outputs == 0 || k == 0 || weights.len() != k * num_outputs {
            return Err(G2gError::CapsMismatch);
        }
        Ok(Self {
            width,
            height,
            num_outputs,
            weights,
            bias,
            in_bytes: k * 4,
            out_bytes: num_outputs * 4,
            configured: false,
            gpu: None,
            last_caps: None,
            emitted: 0,
            gpu_output: false,
        })
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
            shape: TensorShape(vec![1, 3, self.height, self.width]),
            layout: TensorLayout::Nchw,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape(vec![1, self.num_outputs as u32]),
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
        let k = (self.in_bytes / 4) as u32;
        let n = self.num_outputs as u32;

        let meta_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("infer-meta"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut meta = [0u8; 16];
        meta[0..4].copy_from_slice(&k.to_le_bytes());
        meta[4..8].copy_from_slice(&n.to_le_bytes());
        queue.write_buffer(&meta_buf, 0, &meta);

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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpu-linear"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wgpu-linear"),
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

        let layout = gpu.pipeline.get_bind_group_layout(0);
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-linear-binding"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu.meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu.weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gpu.bias_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
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
            pass.dispatch_workgroups((self.num_outputs as u32).div_ceil(WORKGROUP), 1, 1);
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

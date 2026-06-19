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
//! This is the system-memory variant: the NV12 bytes are uploaded to a storage
//! buffer and the f32 tensor is read back to `MemoryDomain::System`. The
//! zero-copy path (binding a decoder's `DmaBuf`/`D3D11Texture` surface straight
//! into the compute pass and emitting a GPU-resident tensor domain) is the
//! follow-up; it needs the surface-import handshake and a GPU tensor domain in
//! core. RGBA input (normalize only, no colour convert) is likewise a small
//! follow-up.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, HardwareError,
    MemoryDomain, OutputSink, PipelinePacket, Rate, RawVideoFormat, TensorDType, TensorLayout,
    TensorShape,
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

#[derive(Debug)]
pub struct WgpuPreprocess {
    width: u32,
    height: u32,
    configured: bool,
    gpu: Option<Gpu>,
    last_caps: Option<Caps>,
    emitted: u64,
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
            last_caps: None,
            emitted: 0,
        }
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
            shape: TensorShape(vec![1, 3, self.height, self.width]),
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
}

impl AsyncElement for WgpuPreprocess {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
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
                shape: TensorShape(vec![1, 3, *h, *w]),
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.ensure_gpu().await?;
                    let bytes = self.dispatch(slice.as_slice())?;
                    let new_caps = self.tensor_caps();
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                        self.last_caps = Some(new_caps);
                    }
                    let tensor = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes)),
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

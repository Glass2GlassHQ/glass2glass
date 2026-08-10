#![cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
//! M990: a `MemoryDomain::DmaBuf` NV12 frame binds into `WgpuPreprocess`'s compute
//! pass through a Vulkan external-memory import, with no CPU round trip.
//!
//! `WgpuToDmaBuf` stands in for the capture / decode producer: a discrete GPU can
//! bind only a GPU-visible dma-buf (not a udmabuf or a USB webcam's), so a
//! GPU-exported one is the producer available on a machine without a VAAPI
//! decoder. The tensor must match the host BT.601 reference and, bit for bit, the
//! same pixels through the System upload path. The third case gives the dma-buf a
//! padded row stride (what a v4l2 capture produces) and asserts the shader reads
//! it in place, with no repack.
//!
//! Needs a GPU with Vulkan dma-buf export + import; CI-excluded.
//!   cargo test -p g2g-ml --features dmabuf-wgpu --test m990_dmabuf_preprocess -- --nocapture

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, Caps, Dim, G2gError, OutputSink, PushOutcome, Rate, RawVideoFormat};
use g2g_ml::wgpupreprocess::{gpu_available, nv12_to_rgb_tensor, WgpuBufferOwner, WgpuPreprocess};
use g2g_plugins::wgpudmabuf::WgpuToDmaBuf;

/// Frame geometry the tensor is built at.
const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
/// Extra bytes per row for the padded-stride case.
const ROW_PADDING: u32 = 6;

/// parallel per-test device creation intermittently segfaults in the NVIDIA driver
/// (the recorded wgpu gotcha), so each GPU test takes this for its whole body.
static GPU_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

impl Collect {
    fn frame(&mut self) -> Frame {
        self.packets
            .drain(..)
            .find_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .expect("element pushed a tensor frame")
    }
}

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

/// An NV12 frame laid out with row stride `stride`: `height` luma rows, then
/// `height / 2` interleaved Cb,Cr rows, each `stride` bytes with only the first
/// `width` meaningful. Padding is 0xFF, a value that shifts the output if the
/// shader ever reads it. Chroma varies per block, so a swapped Cb/Cr or a wrong
/// row would change the tensor.
fn padded_nv12(stride: u32, width: u32, height: u32) -> Vec<u8> {
    let (stride, width, height) = (stride as usize, width as usize, height as usize);
    let mut out = vec![0xFFu8; stride * (height + height / 2)];
    for y in 0..height {
        for x in 0..width {
            out[y * stride + x] = (16 + (x * 13 + y * 29) % 220) as u8;
        }
    }
    let chroma = height * stride;
    for cy in 0..height / 2 {
        for cx in (0..width).step_by(2) {
            out[chroma + cy * stride + cx] = ((cx * 7 + cy * 11) % 256) as u8;
            out[chroma + cy * stride + cx + 1] = ((cx * 23 + cy * 5) % 256) as u8;
        }
    }
    out
}

/// The same pixels repacked at `stride == width`, the layout the host reference
/// and the System upload path take.
fn tight_nv12(padded: &[u8], stride: u32, width: u32, height: u32) -> Vec<u8> {
    let (stride, width, height) = (stride as usize, width as usize, height as usize);
    let mut out = Vec::with_capacity(width * (height + height / 2));
    for row in 0..height + height / 2 {
        out.extend_from_slice(&padded[row * stride..row * stride + width]);
    }
    out
}

fn frame_f32(frame: &Frame) -> Vec<f32> {
    let slice = frame
        .domain
        .as_system_slice()
        .expect("tensor frame is System memory");
    slice
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Export `bytes` as a GPU-allocated dma-buf NV12 frame: the stand-in producer.
/// The export element derives the row stride from the caps width, so passing
/// `width + padding` yields a padded frame. `None` when no GPU can export.
async fn export_nv12_dmabuf(bytes: &[u8], stride_width: u32, height: u32) -> Option<Frame> {
    let mut export = WgpuToDmaBuf::new();
    let (device, queue) = export.gpu().await.ok()?;
    let source = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nv12-export-src"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&source, 0, bytes);

    export
        .configure_pipeline(&nv12_caps(stride_width, height))
        .expect("export configure");
    let mut sink = Collect::default();
    export
        .process(
            PipelinePacket::DataFrame(Frame {
                domain: MemoryDomain::WgpuBuffer(WgpuToDmaBuf::wrap_buffer(
                    &device,
                    source,
                    bytes.len(),
                )),
                timing: FrameTiming {
                    pts_ns: 7_000,
                    ..FrameTiming::default()
                },
                sequence: 3,
                meta: Default::default(),
            }),
            &mut sink,
        )
        .await
        .expect("export process");
    let frame = sink.frame();
    let MemoryDomain::DmaBuf(dmabuf) = &frame.domain else {
        panic!("export produced a dma-buf frame");
    };
    assert_eq!(dmabuf.stride, stride_width, "export used the caps stride");
    Some(frame)
}

/// Run a dma-buf frame through `WgpuPreprocess` at `WIDTH x HEIGHT`. `None` when
/// the driver cannot bind the fd (skip), which the element reports as
/// `UnsupportedDomain`.
async fn preprocess_dmabuf(frame: Frame, gpu_output: bool) -> Option<Frame> {
    let mut element = if gpu_output {
        WgpuPreprocess::new().with_gpu_output()
    } else {
        WgpuPreprocess::new()
    };
    element
        .configure_pipeline(&nv12_caps(WIDTH, HEIGHT))
        .expect("preprocess configure");
    let mut sink = Collect::default();
    match element
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
    {
        Ok(()) => {}
        Err(G2gError::UnsupportedDomain) => {
            eprintln!("SKIP: dma-buf import unsupported on this driver (export succeeded)");
            return None;
        }
        Err(e) => panic!("dma-buf preprocess failed: {e:?}"),
    }
    assert_eq!(element.emitted(), 1, "one tensor per frame");
    Some(sink.frame())
}

/// The same NV12 bytes through the System upload path, for a bit-exact comparison
/// against the import path (same shader, same GPU, so any difference is the
/// import's).
async fn preprocess_system(nv12: Vec<u8>) -> Vec<f32> {
    let mut element = WgpuPreprocess::new();
    element
        .configure_pipeline(&nv12_caps(WIDTH, HEIGHT))
        .expect("preprocess configure");
    let mut sink = Collect::default();
    element
        .process(
            PipelinePacket::DataFrame(Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(nv12.into_boxed_slice())),
                timing: FrameTiming::default(),
                sequence: 0,
                meta: Default::default(),
            }),
            &mut sink,
        )
        .await
        .expect("system preprocess");
    frame_f32(&sink.frame())
}

/// The tensor must match the host BT.601 reference (float tolerance) and be
/// non-grayscale, proving the chroma rows were read.
fn assert_matches_reference(got: &[f32], nv12: &[u8]) {
    let area = (WIDTH * HEIGHT) as usize;
    let expect = nv12_to_rgb_tensor(nv12, WIDTH as usize, HEIGHT as usize);
    assert_eq!(got.len(), 3 * area, "NCHW tensor length");
    for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "tensor[{i}] = {g}, reference {e}: imported pixels differ"
        );
    }
    assert!(
        (0..area).any(|px| (got[px] - got[area + px]).abs() > 1e-3),
        "coloured chroma must break R == G, else the chroma rows were not read"
    );
}

#[tokio::test]
async fn dmabuf_import_matches_system_upload() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let nv12 = padded_nv12(WIDTH, WIDTH, HEIGHT);
    let Some(frame) = export_nv12_dmabuf(&nv12, WIDTH, HEIGHT).await else {
        eprintln!("skipping: no GPU dma-buf export on this host");
        return;
    };
    assert_eq!(frame.timing.pts_ns, 7_000);
    let Some(tensor) = preprocess_dmabuf(frame, false).await else {
        return;
    };
    assert_eq!(tensor.timing.pts_ns, 7_000, "tensor inherits source timing");

    let got = frame_f32(&tensor);
    assert_matches_reference(&got, &nv12);
    assert_eq!(
        got,
        preprocess_system(nv12).await,
        "imported dma-buf and uploaded system memory give the same tensor, bit for bit"
    );
    eprintln!(
        "PASS: dma-buf import == system upload, {} values",
        got.len()
    );
}

#[tokio::test]
async fn dmabuf_import_with_gpu_output_stays_resident() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let nv12 = padded_nv12(WIDTH, WIDTH, HEIGHT);
    let Some(frame) = export_nv12_dmabuf(&nv12, WIDTH, HEIGHT).await else {
        eprintln!("skipping: no GPU dma-buf export on this host");
        return;
    };
    let Some(tensor) = preprocess_dmabuf(frame, true).await else {
        return;
    };

    let MemoryDomain::WgpuBuffer(owned) = &tensor.domain else {
        panic!("gpu-output mode keeps the tensor in a GPU buffer");
    };
    let owner = owned
        .keep_alive()
        .as_any()
        .downcast_ref::<WgpuBufferOwner>()
        .expect("tensor buffer owner");
    let got: Vec<f32> = owner
        .read_back()
        .expect("read back the GPU tensor")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_matches_reference(&got, &nv12);
    eprintln!("PASS: dma-buf in, GPU-resident tensor out, never touched the CPU");
}

#[tokio::test]
async fn padded_stride_dmabuf_needs_no_repack() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let stride = WIDTH + ROW_PADDING;
    let padded = padded_nv12(stride, WIDTH, HEIGHT);
    let Some(frame) = export_nv12_dmabuf(&padded, stride, HEIGHT).await else {
        eprintln!("skipping: no GPU dma-buf export on this host");
        return;
    };
    let Some(tensor) = preprocess_dmabuf(frame, false).await else {
        return;
    };

    let got = frame_f32(&tensor);
    let tight = tight_nv12(&padded, stride, WIDTH, HEIGHT);
    assert_matches_reference(&got, &tight);
    assert_eq!(
        got,
        preprocess_system(tight).await,
        "a padded dma-buf gives the same tensor as the repacked pixels"
    );
    eprintln!("PASS: stride {stride} for width {WIDTH} read in place, no repack");
}

//! M559: GPU -> DMABUF export round trip. `WgpuToDmaBuf` exports a GPU-resident
//! wgpu buffer as a dma-buf fd; `DmaBufToWgpu` re-imports it on a *separate*
//! wgpu device; the bytes are read back and must match, proving the export
//! element works AND that the exported fd is an independent reference (the element
//! frees its own Vulkan handles right after export). Covers packed RGBA and 8-bit
//! NV12 (plane-aware size). Needs a GPU with Vulkan dma-buf export+import;
//! CI-excluded.
//!   cargo test -p g2g-plugins --features dmabuf-wgpu --test m559_wgpu_dmabuf_export -- --nocapture
#![cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, OutputSink, PipelinePacket, PushOutcome, Rate,
    RawVideoFormat,
};

use g2g_plugins::dmabufwgpu::{DmaBufToWgpu, DmaBufWgpuBuffer};
use g2g_plugins::wgpudmabuf::WgpuToDmaBuf;

/// Grabs the single frame an element pushes.
#[derive(Default)]
struct CaptureSink {
    frame: Option<Frame>,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                self.frame = Some(f);
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn caps(format: RawVideoFormat, w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Export a `size`-byte pattern buffer of `format` (WxH) to a dma-buf and
/// re-import it on a second device; return the bytes read back, or `None` if the
/// GPU / driver cannot do the round trip (skip).
async fn roundtrip(format: RawVideoFormat, w: u32, h: u32, size: usize) -> Option<Vec<u8>> {
    let c = caps(format, w, h);

    // Export side: a source buffer with a known pattern on the export device.
    let mut exp = WgpuToDmaBuf::new();
    let (dev, queue) = exp.gpu().await.ok()?;
    let pattern: Vec<u8> = (0..size).map(|i| (i.wrapping_mul(7).wrapping_add(3)) as u8).collect();
    let src = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("export-src"),
        size: size as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&src, 0, &pattern);

    let frame_in = Frame {
        domain: MemoryDomain::WgpuBuffer(WgpuToDmaBuf::wrap_buffer(&dev, src, size)),
        timing: FrameTiming { pts_ns: 1234, ..FrameTiming::default() },
        sequence: 42,
        meta: Default::default(),
    };
    exp.configure_pipeline(&c).expect("export configure");
    let mut cap = CaptureSink::default();
    exp.process(PipelinePacket::DataFrame(frame_in), &mut cap).await.expect("export process");
    let dmabuf_frame = cap.frame.take().expect("export emitted a frame");
    assert!(matches!(dmabuf_frame.domain, MemoryDomain::DmaBuf(_)), "export produced a dma-buf");
    // Timing / sequence pass through the export.
    assert_eq!(dmabuf_frame.sequence, 42);
    assert_eq!(dmabuf_frame.timing.pts_ns, 1234);
    assert_eq!(exp.exported(), 1);

    // Import side: a SEPARATE device (DmaBufToWgpu builds its own).
    let mut imp = DmaBufToWgpu::new();
    imp.configure_pipeline(&c).expect("import configure");
    let mut cap2 = CaptureSink::default();
    match imp.process(PipelinePacket::DataFrame(dmabuf_frame), &mut cap2).await {
        Ok(()) => {}
        Err(G2gError::UnsupportedDomain) => {
            eprintln!("SKIP: dma-buf import unsupported on this driver (export succeeded)");
            return None;
        }
        Err(e) => panic!("import failed: {e:?}"),
    }
    let wgpu_frame = cap2.frame.take().expect("import emitted a frame");
    let MemoryDomain::WgpuBuffer(owned) = &wgpu_frame.domain else {
        panic!("import produced a wgpu buffer");
    };
    assert_eq!(owned.len, size, "imported buffer is the full plane-aware size");
    let buf = owned
        .keep_alive()
        .as_any()
        .downcast_ref::<DmaBufWgpuBuffer>()
        .expect("imported owner")
        .buffer();

    // Read back from the import device.
    let idev = imp.device().expect("import device");
    let iq = imp.queue().expect("import queue");
    let staging = idev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = idev.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size as u64);
    iq.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    idev.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("poll");
    rx.recv().expect("readback channel").expect("map readback");
    let got = slice.get_mapped_range().to_vec();

    assert_eq!(got, pattern, "pixels survived GPU -> dma-buf -> GPU across devices");
    Some(got)
}

#[tokio::test]
async fn roundtrip_rgba8() {
    const W: u32 = 8;
    const H: u32 = 8;
    const SIZE: usize = (W * H * 4) as usize; // packed 4bpp
    if roundtrip(RawVideoFormat::Rgba8, W, H, SIZE).await.is_some() {
        eprintln!("PASS rgba8: {SIZE} bytes exported + re-imported on a 2nd device, exact");
    }
}

#[tokio::test]
async fn roundtrip_nv12() {
    const W: u32 = 8;
    const H: u32 = 8;
    const SIZE: usize = (W * H + W * H / 2) as usize; // luma + interleaved chroma
    if roundtrip(RawVideoFormat::Nv12, W, H, SIZE).await.is_some() {
        eprintln!("PASS nv12: {SIZE} bytes (plane-aware) exported + re-imported, exact");
    }
}

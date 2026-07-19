//! M562: zero-stall GPU->DMABUF sync. `WgpuToDmaBuf::with_external_semaphore(true)`
//! signals a timeline semaphore on each frame's copy submit (instead of blocking on
//! `device.poll(Wait)`) and attaches the exported semaphore fd + per-frame value to
//! the emitted dma-buf. A sem-aware `DmaBufToWgpu` on a *separate* device imports
//! the semaphore once and host-waits each frame's value before reading, so the
//! bytes must still come back exact. Runs several frames to exercise the monotonic
//! timeline and the producer's lazy `dst` reclamation (freed once the copy retires,
//! not by a stall). Needs a GPU with Vulkan dma-buf + external-semaphore export;
//! CI-excluded.
//!   cargo test -p g2g-plugins --features dmabuf-wgpu --test m561_dmabuf_semaphore_sync -- --nocapture
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

fn caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[tokio::test]
async fn timeline_sync_roundtrip_multi_frame() {
    const W: u32 = 8;
    const H: u32 = 8;
    const SIZE: usize = (W * H * 4) as usize;
    const FRAMES: u64 = 4;
    let c = caps(W, H);

    // Producer with the zero-stall external-semaphore mode ON.
    let mut exp = WgpuToDmaBuf::new().with_external_semaphore(true);
    let Ok((dev, queue)) = exp.gpu().await else {
        eprintln!("SKIP: no Vulkan export device / external-semaphore extension");
        return;
    };
    exp.configure_pipeline(&c).expect("export configure");

    // Consumer builds its own (separate) device.
    let mut imp = DmaBufToWgpu::new();
    imp.configure_pipeline(&c).expect("import configure");

    for n in 0..FRAMES {
        // Distinct pattern per frame so a missed wait (reading a stale / torn
        // buffer) would surface as a mismatch.
        let pattern: Vec<u8> = (0..SIZE)
            .map(|i| (i.wrapping_mul(7).wrapping_add(n as usize * 101)) as u8)
            .collect();
        let src = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export-src"),
            size: SIZE as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&src, 0, &pattern);

        let frame_in = Frame {
            domain: MemoryDomain::WgpuBuffer(WgpuToDmaBuf::wrap_buffer(&dev, src, SIZE)),
            timing: FrameTiming::default(),
            sequence: n,
            meta: Default::default(),
        };
        let mut cap = CaptureSink::default();
        exp.process(PipelinePacket::DataFrame(frame_in), &mut cap)
            .await
            .expect("export process");
        let dmabuf_frame = cap.frame.take().expect("export emitted a frame");

        // The emitted dma-buf carries the timeline semaphore fd + a monotonic value.
        let MemoryDomain::DmaBuf(ref db) = dmabuf_frame.domain else {
            panic!("export produced a dma-buf");
        };
        assert!(db.sync_fd().is_some(), "frame {n} carries a sync fd");
        assert_eq!(
            db.sync_value(),
            Some(n + 1),
            "monotonic timeline value (1-based)"
        );

        // Consumer imports + host-waits the value before handing the buffer on.
        let mut cap2 = CaptureSink::default();
        match imp
            .process(PipelinePacket::DataFrame(dmabuf_frame), &mut cap2)
            .await
        {
            Ok(()) => {}
            Err(G2gError::UnsupportedDomain) => {
                eprintln!("SKIP: dma-buf import unsupported on this driver (export + sync worked)");
                return;
            }
            Err(e) => panic!("import failed on frame {n}: {e:?}"),
        }
        let wgpu_frame = cap2.frame.take().expect("import emitted a frame");
        let MemoryDomain::WgpuBuffer(owned) = &wgpu_frame.domain else {
            panic!("import produced a wgpu buffer");
        };
        let buf = owned
            .keep_alive()
            .as_any()
            .downcast_ref::<DmaBufWgpuBuffer>()
            .expect("imported owner")
            .buffer();

        // Read back on the import device; must equal the pattern (proves the
        // host wait ordered the read after the producer's copy).
        let idev = imp.device().expect("import device");
        let iq = imp.queue().expect("import queue");
        let staging = idev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: SIZE as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = idev.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, SIZE as u64);
        iq.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        idev.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("poll");
        rx.recv().expect("readback channel").expect("map readback");
        let got = slice.get_mapped_range().to_vec();
        assert_eq!(
            got, pattern,
            "frame {n}: pixels exact through timeline-synced dma-buf handoff"
        );
    }

    assert_eq!(
        exp.exported(),
        FRAMES,
        "all frames exported without a producer stall"
    );
    eprintln!(
        "PASS: {FRAMES} frames GPU->dma-buf->GPU across devices, ordered by an exported timeline \
         semaphore (producer never called poll(Wait)); every frame byte-exact"
    );
}

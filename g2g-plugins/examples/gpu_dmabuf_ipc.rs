//! End-to-end zero-copy GPU frame egress across a process boundary (M559):
//! `WgpuToDmaBuf -> DmaBufSink` in one process, `DmaBufSrc -> DmaBufToWgpu` in
//! another, with only a dma-buf fd crossing (over a Unix socket via SCM_RIGHTS).
//! A GPU-resident frame leaves the producer process and lands GPU-resident in the
//! consumer process with no CPU copy and no PCIe round trip.
//!
//! This wires together the whole stack proven in pieces:
//! `WgpuToDmaBuf` (M559) exports a wgpu buffer as a dma-buf fd, `DmaBufSink`
//! (M557) passes that fd to the peer via SCM_RIGHTS, `DmaBufSrc` receives it, and
//! `DmaBufToWgpu` re-imports it on the peer's GPU.
//!
//! The lifetime story also composes: the export frees its own Vulkan handles at
//! once (the dma-buf fd is an independent reference), and once the sink sends the
//! fd the kernel's SCM_RIGHTS dup keeps the buffer alive for the consumer.
//!
//! One binary, both ends (self-re-exec with a `child` arg + socket path). The
//! parent renders a known pattern into a wgpu buffer for N frames, exports each,
//! and hands the dma-buf to `DmaBufSink`, then Eos. The child runs
//! `DmaBufSrc -> DmaBufToWgpu -> VerifySink`, reading each imported GPU buffer
//! back and checking the pattern.
//!
//! Run on a Linux GPU host with Vulkan dma-buf export+import (validated on the
//! RTX 3060), not CI:
//!   cargo run -p g2g-plugins --features dmabuf-wgpu,local-dmabuf --example gpu_dmabuf_ipc

use core::future::Future;
use core::pin::Pin;
use std::process::Command;

use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, OutputSink, PipelineClock,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};

use g2g_plugins::dmabufwgpu::{DmaBufToWgpu, DmaBufWgpuBuffer};
use g2g_plugins::localdmabuf::{DmaBufSink, DmaBufSrc};
use g2g_plugins::wgpudmabuf::WgpuToDmaBuf;

const W: u32 = 16;
const H: u32 = 16;
const SIZE: usize = (W * H * 4) as usize; // packed RGBA
const N: u64 = 4;

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Deterministic per-frame pattern (depends on the sequence so a mis-ordered or
/// mis-mapped frame is caught).
fn pattern(seq: u64) -> Vec<u8> {
    (0..SIZE)
        .map(|i| {
            ((seq as usize)
                .wrapping_mul(37)
                .wrapping_add(i)
                .wrapping_mul(7)) as u8
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() == Some("child") {
        let path = args.next().expect("socket path");
        std::process::exit(run_child(&path));
    }
    run_parent();
}

// ---- parent: render + export + DmaBufSink ----

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

struct NullOut;
impl OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn run_parent() {
    let path = std::env::temp_dir()
        .join(format!("g2g-gpudmabuf-{}.sock", std::process::id()))
        .to_string_lossy()
        .into_owned();
    let _ = std::fs::remove_file(&path);
    let exe = std::env::current_exe().expect("current_exe");
    let mut child = Command::new(exe)
        .arg("child")
        .arg(&path)
        .spawn()
        .expect("spawn child");

    let mut waited = 0;
    while !std::path::Path::new(&path).exists() {
        std::thread::sleep(std::time::Duration::from_millis(50));
        waited += 1;
        assert!(waited < 200, "child never bound {path}");
    }

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    // `G2G_DMABUF_SEM=1` exercises the zero-stall external-semaphore path (M561):
    // the producer signals a timeline semaphore on each copy submit instead of
    // blocking on `device.poll(Wait)`, ships the semaphore fd once, and the child
    // host-waits each frame's value before reading. Default (off) is the M560
    // poll(Wait) path. Mirrors the CUDA example's `G2G_ZEROCOPY=1` toggle.
    let use_sem = std::env::var("G2G_DMABUF_SEM").is_ok();
    rt.block_on(async {
        let mut exp = WgpuToDmaBuf::new().with_external_semaphore(use_sem);
        let (dev, queue) = exp
            .gpu()
            .await
            .expect("Vulkan dma-buf export device (need a GPU)");
        exp.configure_pipeline(&caps()).expect("export configure");
        let mut sink = DmaBufSink::new(path);
        sink.configure_pipeline(&caps()).expect("sink configure");
        let mut null = NullOut;

        for seq in 0..N {
            // "Render": fill a wgpu buffer on the export device with the pattern.
            let src = dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame"),
                size: SIZE as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&src, 0, &pattern(seq));
            let frame = Frame {
                domain: MemoryDomain::WgpuBuffer(WgpuToDmaBuf::wrap_buffer(&dev, src, SIZE)),
                timing: FrameTiming {
                    pts_ns: seq * 1000,
                    ..FrameTiming::default()
                },
                sequence: seq,
                meta: Default::default(),
            };
            // Export to a dma-buf, then ship the fd to the peer.
            let mut cap = CaptureSink { frame: None };
            exp.process(PipelinePacket::DataFrame(frame), &mut cap)
                .await
                .expect("export");
            let dmabuf_frame = cap.frame.take().expect("exported a dma-buf frame");
            sink.process(PipelinePacket::DataFrame(dmabuf_frame), &mut null)
                .await
                .expect("send dma-buf");
        }
        sink.process(PipelinePacket::Eos, &mut null)
            .await
            .expect("eos");
    });

    let status = child.wait().expect("wait child");
    let _ = std::fs::remove_file(
        std::env::temp_dir().join(format!("g2g-gpudmabuf-{}.sock", std::process::id())),
    );
    assert!(
        status.success(),
        "child failed to receive/verify GPU frames: {status:?}"
    );
    let sync = if use_sem {
        "ordered by an exported timeline semaphore (producer never blocked on the copy)"
    } else {
        "the producer drained each copy with poll(Wait) before sending"
    };
    println!(
        "PASS: {N} GPU frames left this process as dma-buf fds (SCM_RIGHTS) and were re-imported \
         GPU-resident in the child, every pixel verified (no CPU copy, no PCIe); {sync}"
    );
}

// ---- child: DmaBufSrc -> DmaBufToWgpu -> VerifySink ----

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// The child sink: owns the `DmaBufToWgpu` importer and, per received dma-buf
/// frame, imports it to the GPU, reads it back, and checks the pattern. Owning the
/// importer (rather than placing it as a separate transform in the chain runner)
/// keeps its device/queue reachable for the read-back.
#[derive(Default)]
struct VerifySink {
    imp: DmaBufToWgpu,
    verified: u64,
    bad: bool,
}
impl AsyncElement for VerifySink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // The imported buffer's size is plane-aware, so the importer needs caps.
        self.imp.configure_pipeline(c)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let PipelinePacket::DataFrame(ref frame) = packet else {
                return Ok(());
            };
            let seq = frame.sequence;
            // Import the received dma-buf onto the GPU.
            let mut inner = CaptureSink { frame: None };
            self.imp.process(packet, &mut inner).await?;
            let imported = inner.frame.take().expect("importer emitted a frame");
            let MemoryDomain::WgpuBuffer(owned) = &imported.domain else {
                self.bad = true;
                return Ok(());
            };
            let buf = owned
                .keep_alive()
                .as_any()
                .downcast_ref::<DmaBufWgpuBuffer>()
                .expect("imported owner")
                .buffer();
            let device = self.imp.device().expect("import device");
            let queue = self.imp.queue().expect("import queue");
            if read_back(device, queue, buf) == pattern(seq) {
                self.verified += 1;
            } else {
                self.bad = true;
            }
            Ok(())
        })
    }
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u8> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: SIZE as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, SIZE as u64);
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("poll");
    rx.recv().expect("readback channel").expect("map readback");
    slice.get_mapped_range().to_vec()
}

fn run_child(path: &str) -> i32 {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let mut src = DmaBufSrc::new(path).with_frame_limit(N);
        let mut sink = VerifySink::default();
        let clock = ZeroClock;
        let run =
            run_simple_pipeline(&mut src, &mut sink, &clock, LatencyProfile::Live.link_capacity())
                .await;
        match run {
            Ok(_) if sink.verified == N && !sink.bad => {
                println!(
                    "  child: received + GPU-imported + verified all {N} frames across the process boundary"
                );
                0
            }
            Ok(_) => {
                eprintln!("child: verified {}/{N} (bad={})", sink.verified, sink.bad);
                1
            }
            Err(e) => {
                eprintln!("child: pipeline error: {e:?}");
                1
            }
        }
    })
}

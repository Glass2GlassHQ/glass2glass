//! Cross-process validation of the local CUDA transport elements (M556 phase 2,
//! `local-ipc`). Two *separate processes* move NV12 frames GPU->GPU with no PCIe
//! round trip: the producer's VRAM is shared to the consumer through a CUDA IPC
//! handle over a Unix socket, and only the handle + descriptor cross.
//!
//! One binary, both ends (self-re-exec with a `child` arg + socket path):
//!   parent: create a CUDA context, allocate an NV12 buffer, fill it with a known
//!           pattern, and drive `LocalCudaSink` for N frames + Eos.
//!   child : run `LocalCudaSrc -> VerifySink`; the source maps the parent's VRAM,
//!           copies it device->device into its own buffer, acks, and emits it;
//!           VerifySink reads each frame back and checks the pattern.
//!
//! Run on a machine with an NVIDIA GPU (not CI, like the rest of the CUDA stack):
//!   cargo run -p g2g-plugins --features local-ipc --example local_cuda_transport

use core::future::Future;
use core::pin::Pin;
use std::process::Command;
use std::sync::Arc;

use g2g_core::memory::{CudaKeepAlive, OwnedCudaBuffer};
use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};

use g2g_plugins::localcuda::{LocalCudaSink, LocalCudaSrc};
use g2g_plugins::localipc;

const W: u32 = 64;
const H: u32 = 48;
const N: u64 = 5;

/// span of a packed NV12 frame (tight pitch = width): luma W*H + chroma W*ceil(H/2).
fn span() -> usize {
    (W as usize) * (H as usize) + (W as usize) * (H as usize).div_ceil(2)
}

fn pattern(i: usize) -> u8 {
    ((i as u32).wrapping_mul(2_654_435_761) >> 11) as u8
}

/// Select the receiver mode. `G2G_ZEROCOPY=1` exercises `LocalCudaSrc::zero_copy`
/// (emit the producer's mapped VRAM directly, no receive-side copy); otherwise
/// the default device->device copy mode. The child inherits this env.
fn zero_copy() -> bool {
    std::env::var("G2G_ZEROCOPY").map(|v| v == "1").unwrap_or(false)
}

fn nv12() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() == Some("child") {
        let path = args.next().expect("socket path");
        std::process::exit(run_child(&path));
    }
    run_parent();
}

// ---- parent: fill VRAM, drive LocalCudaSink ----

#[derive(Debug)]
struct ParentCtx(u64);
// SAFETY: single-thread (current_thread runtime); the context is thread-affine.
unsafe impl Send for ParentCtx {}
// SAFETY: see above (single-thread; no concurrent access).
unsafe impl Sync for ParentCtx {}
impl Drop for ParentCtx {
    fn drop(&mut self) {
        localipc::destroy_context(self.0);
    }
}

#[derive(Debug)]
struct ParentAlloc {
    dptr: u64,
    _ctx: Arc<ParentCtx>,
}
impl Drop for ParentAlloc {
    fn drop(&mut self) {
        // SAFETY: our allocation, freed once; context still pinned + current.
        unsafe {
            let _ = localipc::free(self.dptr);
        }
    }
}
impl CudaKeepAlive for ParentAlloc {}

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
    // Unique socket path; spawn the child (the receiver) which binds it.
    let path = std::env::temp_dir()
        .join(format!("g2g-localcuda-{}.sock", std::process::id()))
        .to_string_lossy()
        .into_owned();
    let _ = std::fs::remove_file(&path);
    let exe = std::env::current_exe().expect("current_exe");
    let mut child =
        Command::new(exe).arg("child").arg(&path).spawn().expect("spawn child");

    // Wait for the child to bind the socket (it does so during negotiation).
    let mut waited = 0;
    while !std::path::Path::new(&path).exists() {
        std::thread::sleep(std::time::Duration::from_millis(50));
        waited += 1;
        assert!(waited < 200, "child never bound {path}");
    }

    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let ctx = Arc::new(ParentCtx(
            localipc::init_context(0).expect("CUDA context (need an NVIDIA GPU)"),
        ));
        let sz = span();
        let base = localipc::alloc(sz).expect("cuMemAlloc");
        let host: Vec<u8> = (0..sz).map(pattern).collect();
        // SAFETY: `base` is a live `sz`-byte allocation in the current context.
        unsafe { localipc::htod(base, &host).expect("htod") };

        // NV12 in one packed allocation: luma at base, chroma at base + W*H,
        // tight pitch = width. Shared by all frames (same VRAM, same pattern).
        let buf = OwnedCudaBuffer::new(
            base,
            base + (W * H) as u64,
            W,
            W,
            W,
            H,
            ctx.0,
            Arc::new(ParentAlloc { dptr: base, _ctx: Arc::clone(&ctx) }),
        );

        let mut sink = LocalCudaSink::new(path);
        sink.configure_pipeline(&nv12()).expect("configure");
        let mut null = NullOut;
        for seq in 0..N {
            let frame = Frame {
                domain: MemoryDomain::Cuda(buf.clone()),
                timing: FrameTiming { pts_ns: seq * 1000, ..FrameTiming::default() },
                sequence: seq,
                meta: Default::default(),
            };
            sink.process(PipelinePacket::DataFrame(frame), &mut null)
                .await
                .expect("send frame");
        }
        sink.process(PipelinePacket::Eos, &mut null).await.expect("eos");
        assert_eq!(sink.sent(), N, "all frames sent + acked");
        // `buf` drops here (last ref) -> ParentAlloc frees the VRAM; the child
        // has already copied every frame (it acked each before we advanced).
    });

    let status = child.wait().expect("wait child");
    let _ = std::fs::remove_file(
        std::env::temp_dir().join(format!("g2g-localcuda-{}.sock", std::process::id())),
    );
    assert!(status.success(), "child failed to receive/verify frames: {status:?}");
    let mode = if zero_copy() { "zero-copy (direct-mapped)" } else { "device->device copy" };
    println!(
        "PASS [{mode}]: {N} NV12 frames crossed a process boundary GPU->GPU via \
         CUDA IPC (no PCIe); child verified every pixel"
    );
}

// ---- child: LocalCudaSrc -> VerifySink ----

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Reads each received CUDA frame back to host and checks the pattern the parent
/// wrote. Counts verified frames.
#[derive(Default)]
struct VerifySink {
    verified: u64,
    bad: bool,
}
impl AsyncElement for VerifySink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>> where Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                if let MemoryDomain::Cuda(buf) = &frame.domain {
                    let sz = span();
                    let mut host = vec![0u8; sz];
                    // SAFETY: `luma_ptr` is the base of the receiver's own dest
                    // allocation (>= sz bytes) in the current (Src) context.
                    unsafe { localipc::dtoh(&mut host, buf.luma_ptr)? };
                    if host.iter().enumerate().all(|(i, b)| *b == pattern(i)) {
                        self.verified += 1;
                    } else {
                        self.bad = true;
                    }
                }
            }
            Ok(())
        })
    }
}

fn run_child(path: &str) -> i32 {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let mut src = LocalCudaSrc::new(path).with_frame_limit(N);
        if zero_copy() {
            src = src.zero_copy();
        }
        let mut sink = VerifySink::default();
        let clock = ZeroClock;
        let res = run_simple_pipeline(
            &mut src,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        )
        .await;
        match res {
            Ok(_) if sink.verified == N && !sink.bad => {
                println!("  child: received + verified all {N} frames GPU->GPU");
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

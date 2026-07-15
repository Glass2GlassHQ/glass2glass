//! Cross-process zero-copy validation for the CUDA IPC primitive (M556,
//! `local-ipc`). Proves two *separate processes* map the SAME VRAM: the only
//! thing transmitted is the 64-byte IPC handle (plus the size), no frame bytes.
//!
//! The one binary is both ends (it re-execs itself with a `child` argument):
//!   parent: create a CUDA context, allocate device memory, upload a known byte
//!           pattern, export an IPC handle, spawn the child, write the descriptor
//!           to the child's stdin, and wait for it.
//!   child : create its own context on the same device, open the handle (mapping
//!           the parent's allocation, no copy), read it back device->host, and
//!           verify every byte matches the pattern the parent wrote.
//! The parent asserts the child exited 0 (verified) before it frees, which also
//! satisfies the "exporter stays alive until the importer opens" contract.
//!
//! Run on a machine with an NVIDIA GPU (not CI, like the rest of the CUDA stack):
//!   cargo run -p g2g-plugins --features local-ipc --example cuda_ipc_smoke

use std::io::{Read, Write};
use std::process::{Command, Stdio};

use g2g_plugins::localipc::{
    self, CudaIpcDescriptor, CudaIpcHandle, CUDA_IPC_HANDLE_SIZE,
};

/// Bytes shared across the two processes (a stand-in for one video frame).
const SIZE: usize = 1920 * 1080 * 3 / 2; // one 1080p NV12 frame's worth

/// The deterministic pattern the parent writes and the child verifies.
fn pattern(i: usize) -> u8 {
    ((i as u32).wrapping_mul(2_654_435_761) >> 13) as u8
}

fn main() {
    let is_child = std::env::args().any(|a| a == "child");
    if is_child {
        std::process::exit(run_child());
    }
    run_parent();
}

/// Parent: allocate + fill VRAM, export the handle, hand it to a child process,
/// and assert the child mapped the same VRAM and read the pattern back.
fn run_parent() {
    let ctx = localipc::init_context(0).expect("create CUDA context (need an NVIDIA GPU)");
    let dptr = localipc::alloc(SIZE).expect("cuMemAlloc");

    // Upload the known pattern host->device.
    let mut host: Vec<u8> = (0..SIZE).map(pattern).collect();
    // SAFETY: `dptr` is a live SIZE-byte allocation in the current context.
    unsafe { localipc::htod(dptr, &host).expect("htod") };

    // Export a handle to the whole allocation (plain bytes, transport-agnostic).
    // SAFETY: `dptr` is the base of a live cuMemAlloc allocation.
    let handle: CudaIpcHandle = unsafe { localipc::ipc_export(dptr).expect("ipc export") };
    let desc = CudaIpcDescriptor { handle, size: SIZE as u64 };

    // Spawn a fresh process (its own CUDA context) and hand it the descriptor
    // over stdin. Wait for it, so our allocation stays alive until it has opened
    // the handle (the CUDA IPC lifetime contract).
    let exe = std::env::current_exe().expect("current_exe");
    let mut child = Command::new(exe)
        .arg("child")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn child");
    child
        .stdin
        .take()
        .expect("child stdin")
        .write_all(&desc.to_bytes())
        .expect("write descriptor");
    let out = child.wait_with_output().expect("wait child");

    // Cleanup only after the child is done (it has opened + closed its mapping).
    // Scrub host so a false pass can't come from stale memory.
    host.fill(0);
    // SAFETY: `dptr` is our live allocation, freed once; `ctx` created by us.
    unsafe { localipc::free(dptr).expect("free") };
    localipc::destroy_context(ctx);

    print!("{}", String::from_utf8_lossy(&out.stdout));
    assert!(
        out.status.success(),
        "child failed to read the parent's VRAM over the IPC handle: {:?}",
        out.status
    );
    println!(
        "PASS: two processes shared {SIZE} bytes of VRAM zero-copy; only the \
         {CUDA_IPC_HANDLE_SIZE}-byte handle crossed"
    );
}

/// Child: read the descriptor, open the handle (mapping the parent's VRAM),
/// verify the pattern. Exit 0 on match, 1 on any mismatch / error.
fn run_child() -> i32 {
    let mut bytes = Vec::new();
    if std::io::stdin().read_to_end(&mut bytes).is_err() {
        eprintln!("child: read stdin failed");
        return 1;
    }
    let Some(desc) = CudaIpcDescriptor::from_bytes(&bytes) else {
        eprintln!("child: short descriptor");
        return 1;
    };
    let size = desc.size as usize;

    // Own context on the same device; the parent's allocation is still live.
    let ctx = match localipc::init_context(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("child: context: {e:?}");
            return 1;
        }
    };
    // SAFETY: a context on device 0 is current; the exporter is still alive.
    let mapped = match unsafe { localipc::ipc_open(&desc.handle) } {
        Ok(p) => p,
        Err(e) => {
            eprintln!("child: ipc_open: {e:?}");
            localipc::destroy_context(ctx);
            return 1;
        }
    };

    // Read the SHARED VRAM back to host and verify byte-for-byte.
    let mut host = vec![0u8; size];
    // SAFETY: `mapped` addresses the parent's live `size`-byte allocation.
    let read = unsafe { localipc::dtoh(&mut host, mapped) };
    let mut ok = read.is_ok();
    if ok {
        for (i, b) in host.iter().enumerate() {
            if *b != pattern(i) {
                eprintln!("child: mismatch at byte {i}: got {b}, want {}", pattern(i));
                ok = false;
                break;
            }
        }
    } else {
        eprintln!("child: dtoh failed: {read:?}");
    }

    // SAFETY: `mapped` came from ipc_open, closed once; then destroy our context.
    unsafe {
        let _ = localipc::ipc_close(mapped);
    }
    localipc::destroy_context(ctx);

    if ok {
        println!("  child: opened the handle and verified all {size} bytes match");
        0
    } else {
        1
    }
}

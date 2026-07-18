//! Cross-process validation of the vendor-neutral DMABUF transport elements
//! (M557, `local-dmabuf`). Two *separate processes* move a frame with no copy:
//! the producer's dma-buf file descriptor is passed to the consumer as
//! `SCM_RIGHTS` ancillary data over a Unix socket, so both processes reference
//! the same underlying buffer (kernel-refcounted) and only the fd + a small
//! descriptor cross.
//!
//! The dma-buf here is a `udmabuf` (a genuine dma-buf built from a sealed memfd
//! via `/dev/udmabuf`): CPU-mappable and GPU-agnostic, so the transport is
//! validated end-to-end without any GPU driver. A GPU-exported dma-buf would ride
//! the exact same path; the receive side would then import it with the
//! `dmabuf-wgpu` element instead of mmap-ing it.
//!
//! One binary, both ends (self-re-exec with a `child` arg + socket path):
//!   parent: create a udmabuf, mmap it, fill it with a known pattern, wrap it as
//!           an `OwnedDmaBuf`, and drive `DmaBufSink` for N frames + Eos.
//!   child : run `DmaBufSrc -> VerifySink`; the source wraps each received fd as a
//!           dma-buf frame; VerifySink mmaps it and checks the pattern.
//!
//! Run on Linux with `/dev/udmabuf` accessible (not CI):
//!   cargo run -p g2g-plugins --features local-dmabuf --example local_dmabuf_transport

use core::ffi::{c_char, c_int, c_uint, c_void};
use core::future::Future;
use core::pin::Pin;
use std::os::fd::{FromRawFd, OwnedFd};
use std::process::Command;

use g2g_core::memory::{MemoryDomain, OwnedDmaBuf};
use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, OutputSink,
    PipelineClock, PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};

use g2g_plugins::localdmabuf::{DmaBufSink, DmaBufSrc};

const W: u32 = 64;
const H: u32 = 48;
const N: u64 = 5;

/// Allocation size shared over the transport: a packed NV12 frame (tight pitch =
/// width) rounded up to a page, since `udmabuf` requires a page-aligned size. The
/// whole allocation is patterned + verified.
fn span() -> usize {
    let nv12 = (W as usize) * (H as usize) + (W as usize) * (H as usize).div_ceil(2);
    nv12.div_ceil(4096) * 4096
}

fn pattern(i: usize) -> u8 {
    ((i as u32).wrapping_mul(2_654_435_761) >> 11) as u8
}

fn nv12() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
    }
}

// ---- minimal udmabuf FFI (example-local; a real element would not create its
// own dma-bufs, it would receive them from a GPU / capture producer) ----

const MFD_CLOEXEC: c_uint = 0x0001;
const MFD_ALLOW_SEALING: c_uint = 0x0002;
const F_ADD_SEALS: c_int = 1033;
const F_SEAL_SHRINK: c_int = 0x0002;
const O_RDWR: c_int = 2;
const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const MAP_SHARED: c_int = 1;
/// `_IOW('u', 0x42, struct udmabuf_create)` on generic Linux (sizeof == 24).
const UDMABUF_CREATE: u64 = 0x4018_7542;
const UDMABUF_FLAGS_CLOEXEC: u32 = 0x01;

#[repr(C)]
struct UdmabufCreate {
    memfd: u32,
    flags: u32,
    offset: u64,
    size: u64,
}

extern "C" {
    fn memfd_create(name: *const c_char, flags: c_uint) -> c_int;
    fn ftruncate(fd: c_int, length: i64) -> c_int;
    fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    fn ioctl(fd: c_int, request: u64, ...) -> c_int;
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        off: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
    fn close(fd: c_int) -> c_int;
}

/// Create a `udmabuf` of `size` bytes, returning an owned dma-buf fd. The backing
/// memfd is sealed with `F_SEAL_SHRINK` (a udmabuf requirement) and closed; the
/// dma-buf keeps the memory alive.
fn create_udmabuf(size: usize) -> OwnedFd {
    // SAFETY: standard libc sequence; every fd is checked and ownership is
    // transferred into an `OwnedFd` at the end (or the process aborts on failure).
    unsafe {
        let memfd = memfd_create(
            b"g2g-udmabuf\0".as_ptr() as *const c_char,
            MFD_CLOEXEC | MFD_ALLOW_SEALING,
        );
        assert!(memfd >= 0, "memfd_create failed");
        assert_eq!(ftruncate(memfd, size as i64), 0, "ftruncate failed");
        assert_eq!(
            fcntl(memfd, F_ADD_SEALS, F_SEAL_SHRINK),
            0,
            "F_SEAL_SHRINK failed"
        );
        let dev = open(b"/dev/udmabuf\0".as_ptr() as *const c_char, O_RDWR);
        assert!(
            dev >= 0,
            "open /dev/udmabuf failed (need access; user in the right group?)"
        );
        let create = UdmabufCreate {
            memfd: memfd as u32,
            flags: UDMABUF_FLAGS_CLOEXEC,
            offset: 0,
            size: size as u64,
        };
        let dmabuf = ioctl(dev, UDMABUF_CREATE, &create as *const UdmabufCreate);
        assert!(dmabuf >= 0, "UDMABUF_CREATE ioctl failed");
        close(dev);
        close(memfd);
        OwnedFd::from_raw_fd(dmabuf)
    }
}

/// Fill (`write=true`) or verify a dma-buf fd's first `size` bytes against
/// [`pattern`] via an mmap. Returns true on a verify match.
fn map_and_check(fd: c_int, size: usize, write: bool) -> bool {
    // SAFETY: `fd` is a live CPU-mappable dma-buf of >= `size` bytes; the mapping
    // is unmapped before return.
    unsafe {
        let prot = if write {
            PROT_READ | PROT_WRITE
        } else {
            PROT_READ
        };
        let ptr = mmap(core::ptr::null_mut(), size, prot, MAP_SHARED, fd, 0);
        assert!(ptr as isize != -1, "mmap dma-buf failed");
        let bytes = core::slice::from_raw_parts_mut(ptr as *mut u8, size);
        let ok = if write {
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = pattern(i);
            }
            true
        } else {
            bytes.iter().enumerate().all(|(i, b)| *b == pattern(i))
        };
        munmap(ptr, size);
        ok
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

// ---- parent: fill a udmabuf, drive DmaBufSink ----

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
        .join(format!("g2g-localdmabuf-{}.sock", std::process::id()))
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

    let sz = span();
    let dmabuf_fd = create_udmabuf(sz);
    use std::os::fd::AsRawFd;
    assert!(map_and_check(dmabuf_fd.as_raw_fd(), sz, true), "fill mmap");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let raw = {
            use std::os::fd::IntoRawFd;
            dmabuf_fd.into_raw_fd()
        };
        // stride = tight width, offset 0 (packed NV12 in one allocation).
        // SAFETY: `raw` is a live, solely-owned dma-buf fd (from the OwnedFd we
        // just consumed); OwnedDmaBuf takes ownership and closes it once on drop.
        let owned = unsafe { OwnedDmaBuf::from_raw(raw, W, 0) };

        let mut sink = DmaBufSink::new(path);
        sink.configure_pipeline(&nv12()).expect("configure");
        let mut null = NullOut;
        for seq in 0..N {
            let frame = Frame {
                // Clone shares the fd (Arc refcount); each send dups it to the child.
                domain: MemoryDomain::DmaBuf(owned.clone()),
                timing: FrameTiming {
                    pts_ns: seq * 1000,
                    ..FrameTiming::default()
                },
                sequence: seq,
                meta: Default::default(),
            };
            sink.process(PipelinePacket::DataFrame(frame), &mut null)
                .await
                .expect("send frame");
        }
        sink.process(PipelinePacket::Eos, &mut null)
            .await
            .expect("eos");
        assert_eq!(sink.sent(), N, "all frames sent");
        // `owned` drops here (last local ref) -> closes our fd; the child's dup(s)
        // keep the buffer alive until it is done.
    });

    let status = child.wait().expect("wait child");
    let _ = std::fs::remove_file(
        std::env::temp_dir().join(format!("g2g-localdmabuf-{}.sock", std::process::id())),
    );
    assert!(
        status.success(),
        "child failed to receive/verify frames: {status:?}"
    );
    println!(
        "PASS: {N} dma-buf frames crossed a process boundary via SCM_RIGHTS fd \
         passing (no copy, vendor-neutral); child mmap-verified every byte"
    );
}

// ---- child: DmaBufSrc -> VerifySink ----

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Default)]
struct VerifySink {
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
                if let MemoryDomain::DmaBuf(dmabuf) = &frame.domain {
                    if map_and_check(dmabuf.as_raw(), span(), false) {
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
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let mut src = DmaBufSrc::new(path).with_frame_limit(N);
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
                println!("  child: received + mmap-verified all {N} frames");
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

//! C seam ABI proof (M650): a C-driven, frame-stepped g2g pipeline. Where
//! `examples/g2g-freertos` links the whole pipeline as a Rust static library
//! the C app *calls into*, this crate proves the inverse integration a C/RTOS
//! shop actually needs: C code *is* the peripheral. The board registers its C
//! capture routine and its C network stack as function pointers
//! (`g2g_audio_egress_init`), then drives the pipeline one frame at a time from
//! its own superloop (`g2g_audio_egress_step`) with control returning to C
//! between frames. No Rust adapter is hand-written; the existing C drivers feed
//! and drain a real g2g graph (`capture -> G.711 mu-law encode -> RTP`).
//!
//! The seam adapters ([`g2g_mcu::cffi`]) and the frame-at-a-time runner
//! ([`g2g_core::step_source_sink`]) carry the same heap-free / panic-free
//! guarantees as the rest of the MCU path: this crate links for a bare
//! Cortex-M target with no allocator and no reachable panic, asserted by
//! `tools/cffi-check.sh`, then runs on the host from `harness.c` (a real C
//! caller supplying C capture + send callbacks) with its wire compared against
//! the pipeline's own Rust reference over an identical input.
//!
//! [`g2g_mcu::cffi`]: g2g_mcu::cffi

#![no_std]

use core::cell::UnsafeCell;
use core::ffi::c_void;

use g2g_core::staticpool::StaticLendRing;
use g2g_core::{
    drive_ready, run_source_transform_sink, step_source_sink, MediaClock, SinkChain, Step,
};
use g2g_mcu::cffi::{CaptureFn, SendFn};
use g2g_mcu::{CFrameGrabber, CPacketSender, FrameGrabber, G711Enc, GrabberSrc, Law, PacketSender, RtpSink};

// Required by `no_std`, but unreachable: the archive has no `core::panicking`
// symbols (checked by `tools/cffi-check.sh`), so nothing can call this.
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// 20 ms of 8 kHz mono S16LE: 160 samples, 320 input bytes, 160 mu-law out.
const SAMPLES_PER_FRAME: usize = 160;
const CAP_BYTES: usize = SAMPLES_PER_FRAME * 2;
const ULAW_BYTES: usize = SAMPLES_PER_FRAME;
const FRAME_NS: u64 = 20_000_000;
const CLOCK_HZ: u32 = 8000;
const PT_PCMU: u8 = 0;
/// The fixed RTP identity the reference uses; a C caller comparing against
/// [`g2g_audio_egress_reference`] must `init` with this SSRC and sequence 0 (see
/// `G2G_AUDIO_EGRESS_REF_SSRC` in `include/g2g_cffi.h`).
const REF_SSRC: u32 = 0x0A0B_0C0D;

// ── Step return codes (documented in include/g2g_cffi.h) ──────────────────────
const STEP_ADVANCED: i32 = 1;
const STEP_EOS: i32 = 0;
const STEP_ERROR: i32 = -1;
const STEP_PENDING: i32 = -2;
const STEP_UNINIT: i32 = -3;

// The C-driven pipeline lives in statics: the lend rings are `const`-constructed
// (so the zero-copy lend is sound by construction), and the assembled pipeline
// is stored in a singleton cell set by `init`. One pipeline per library, the MCU
// singleton pattern.
static CAP_RING: StaticLendRing<1, CAP_BYTES> = StaticLendRing::new();
static ENC_RING: StaticLendRing<1, ULAW_BYTES> = StaticLendRing::new();

type Source = GrabberSrc<'static, CFrameGrabber, 1, CAP_BYTES>;
type Sink = SinkChain<G711Enc<'static, 1, ULAW_BYTES>, RtpSink<CPacketSender>>;

struct Pipeline {
    src: Source,
    sink: Sink,
}

struct PipelineCell(UnsafeCell<Option<Pipeline>>);

// SAFETY: the target is a single-threaded MCU superloop; `init` / `step` /
// `reset` are documented to be called only from that one context, so there is
// never concurrent access to the cell. (The same single-thread contract the
// `g2g-mcu` FFI seams and the whole no-alloc executor model assume.)
unsafe impl Sync for PipelineCell {}

static PIPELINE: PipelineCell = PipelineCell(UnsafeCell::new(None));

/// Wire a C-driven audio egress pipeline (`capture -> G.711 mu-law -> RTP`) from
/// the caller's C capture and send callbacks. Returns 0. Replaces any pipeline a
/// prior `init` left in place.
///
/// # Safety
/// `capture` and `send` must stay valid functions for the life of the pipeline
/// (until `g2g_audio_egress_reset` or the next `init`), and the two ctx pointers
/// valid handles to pass them. Must be called from the single pipeline thread.
#[no_mangle]
pub unsafe extern "C" fn g2g_audio_egress_init(
    capture: CaptureFn,
    capture_ctx: *mut c_void,
    send: SendFn,
    send_ctx: *mut c_void,
    ssrc: u32,
    sequence: u16,
) -> i32 {
    // SAFETY: the caller's contract (documented above) guarantees the callbacks
    // and ctx handles are valid for the pipeline's life.
    let (grabber, sender) =
        unsafe { (CFrameGrabber::new(capture, capture_ctx), CPacketSender::new(send, send_ctx)) };
    let src = GrabberSrc::new(grabber, &CAP_RING, FRAME_NS);
    let enc = G711Enc::new(Law::Mulaw, &ENC_RING);
    let rtp = RtpSink::new(sender, MediaClock::audio(CLOCK_HZ), PT_PCMU, ssrc, sequence);
    let pipeline = Pipeline { src, sink: SinkChain(enc, rtp) };
    // SAFETY: single-threaded per the contract; no other reference to the cell
    // is live during this write.
    unsafe { *PIPELINE.0.get() = Some(pipeline) };
    0
}

/// Run exactly one frame (`capture -> encode -> send`) and return control to the
/// caller. Returns 1 when a packet was emitted, 0 at end of stream, -1 on a
/// stage error, -2 if a stage suspended (use a real executor), -3 if `init` has
/// not run. This is the yield-to-C superloop entry: call it once per frame.
///
/// # Safety
/// Must be called from the single pipeline thread, after `init`.
#[no_mangle]
pub unsafe extern "C" fn g2g_audio_egress_step() -> i32 {
    // SAFETY: single-threaded per the contract; the returned reference is the
    // sole live borrow of the cell for the duration of the step.
    let slot = unsafe { &mut *PIPELINE.0.get() };
    let Some(pipeline) = slot.as_mut() else {
        return STEP_UNINIT;
    };
    match step_source_sink(&mut pipeline.src, &mut pipeline.sink) {
        Ok(Step::Advanced) => STEP_ADVANCED,
        Ok(Step::Eos) => STEP_EOS,
        Ok(Step::Pending) => STEP_PENDING,
        Err(_) => STEP_ERROR,
    }
}

/// Drop the pipeline, releasing the C seams. A fresh `init` may follow.
///
/// # Safety
/// Must be called from the single pipeline thread.
#[no_mangle]
pub unsafe extern "C" fn g2g_audio_egress_reset() {
    // SAFETY: single-threaded per the contract.
    unsafe { *PIPELINE.0.get() = None };
}

/// Fill `buf` (`len` bytes, mono S16LE) with the reference capture ramp starting
/// at absolute sample index `global_sample`: a deterministic signed sweep the
/// proof harness uses as its C capture so the C-seam run and
/// [`g2g_audio_egress_reference`] feed byte-identical input, isolating the seam.
/// Exposed so the C harness and the Rust reference share ONE definition of the
/// input (no cross-language reimplementation to drift).
///
/// # Safety
/// `buf` must be a valid writable buffer of `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn g2g_audio_egress_fill_ramp(buf: *mut u8, len: usize, global_sample: u32) {
    // SAFETY: the caller guarantees `buf` is writable for `len` bytes.
    let buf = unsafe { core::slice::from_raw_parts_mut(buf, len) };
    fill_ramp(buf, global_sample);
}

/// The input frame size (S16LE bytes) a `capture` callback is handed per frame.
#[no_mangle]
pub extern "C" fn g2g_audio_egress_frame_bytes() -> usize {
    CAP_BYTES
}

/// Run the SAME pipeline with native Rust seams over the reference ramp, for
/// `frames` frames, and return its wire checksum (sum of every emitted byte plus
/// the packet count in the high 32 bits). A C-seam run over the identical ramp
/// and RTP identity must reproduce this exactly, which is what proves the C seam
/// is byte-transparent. Uses its own rings, independent of the C-driven pipeline.
#[no_mangle]
pub extern "C" fn g2g_audio_egress_reference(frames: u32) -> u64 {
    let cap_ring: StaticLendRing<1, CAP_BYTES> = StaticLendRing::new();
    let enc_ring: StaticLendRing<1, ULAW_BYTES> = StaticLendRing::new();
    // SAFETY: both rings outlive the run (the runner drains before they drop).
    let src = unsafe { GrabberSrc::with_ring(RampGrabber { next: 0 }, &cap_ring, FRAME_NS) }
        .with_frame_limit(frames);
    // SAFETY: the encoder's ring likewise outlives the run.
    let enc = unsafe { G711Enc::with_ring(Law::Mulaw, &enc_ring) };
    let mut rtp =
        RtpSink::new(SumSender::default(), MediaClock::audio(CLOCK_HZ), PT_PCMU, REF_SSRC, 0);
    match drive_ready(run_source_transform_sink(src, enc, &mut rtp)) {
        Some(Ok(())) => {}
        _ => return 0,
    }
    rtp.free().checksum()
}

/// The reference capture ramp: sample at absolute index `g` is
/// `((g & 0xff) - 128) * 128` (a signed sweep exercising several G.711
/// segments). Panic-free (slice pattern, constant array indices), so it stays in
/// the no-alloc / panic-free archive.
fn fill_ramp(buf: &mut [u8], global_sample: u32) {
    let mut g = global_sample;
    for chunk in buf.chunks_exact_mut(2) {
        let sample = (((g & 0xff) as i32 - 128) * 128) as i16;
        let bytes = sample.to_le_bytes();
        let [lo, hi] = chunk else { continue };
        *lo = bytes[0];
        *hi = bytes[1];
        g = g.wrapping_add(1);
    }
}

/// The native Rust capture used by [`g2g_audio_egress_reference`]: fills each
/// frame from [`fill_ramp`], tracking the absolute sample index across frames.
struct RampGrabber {
    next: u32,
}

impl FrameGrabber for RampGrabber {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, g2g_core::error::G2gError> {
        fill_ramp(buf, self.next);
        self.next = self.next.wrapping_add((buf.len() / 2) as u32);
        Ok(buf.len())
    }
}

/// The native Rust sender used by the reference: folds the wire into a checksum
/// the C harness matches (byte sum plus packet count in the high bits).
#[derive(Default)]
struct SumSender {
    sum: u64,
    packets: u32,
}

impl SumSender {
    fn checksum(&self) -> u64 {
        self.sum.wrapping_add((self.packets as u64) << 32)
    }
}

impl PacketSender for SumSender {
    async fn send(
        &mut self,
        header: &[u8; g2g_core::rtp::RTP_HEADER_LEN],
        payload: &[u8],
    ) -> Result<(), g2g_core::error::G2gError> {
        for &b in header.iter().chain(payload.iter()) {
            self.sum = self.sum.wrapping_add(b as u64);
        }
        self.packets = self.packets.wrapping_add(1);
        Ok(())
    }
}

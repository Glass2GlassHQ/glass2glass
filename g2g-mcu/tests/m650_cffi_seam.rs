//! M650: the C FFI seam adapters (`CFrameGrabber` / `CPacketSender`) let C code
//! be a g2g peripheral. This proves the seam is byte-transparent: the SAME
//! pipeline (`capture -> G.711 mu-law encode -> RTP`) driven through the C
//! callbacks produces the exact wire the native Rust seams produce, so the
//! adapter's function-pointer marshalling adds nothing and drops nothing. The
//! G.711 / RTP correctness itself is proven elsewhere (M638 bit-exact vs ffmpeg,
//! M643 ffmpeg RTP peer); this test isolates the C seam. It also drives the C
//! path through `g2g_core::step_source_sink`, exercising the frame-at-a-time
//! (C-superloop) runner the FFI ABI uses.

use core::ffi::c_void;

use g2g_core::rtp::RTP_HEADER_LEN;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{
    run_source_transform_sink, step_source_sink, MediaClock, SinkChain, Step,
};
use g2g_mcu::{
    CFrameGrabber, CPacketSender, FrameGrabber, G711Enc, GrabberSrc, Law, PacketSender, RtpSink,
};

// 20 ms of 8 kHz mono S16LE = 160 samples: input 320 bytes, mu-law out 160.
const SAMPLES_PER_FRAME: usize = 160;
const CAP_BYTES: usize = SAMPLES_PER_FRAME * 2;
const ULAW_BYTES: usize = SAMPLES_PER_FRAME;
const FRAME_NS: u64 = 20_000_000;
const FRAMES: u32 = 25;
const SSRC: u32 = 0x0A0B_0C0D;

/// The shared, deterministic capture pattern used by BOTH the native grabber and
/// the C capture callback, so the two runs differ ONLY in the seam mechanism
/// (a direct trait impl vs a C function pointer). A signed ramp that sweeps a
/// range of amplitudes and both signs, so the G.711 encoder exercises several
/// segments (a constant tone would not).
fn fill_ramp(buf: &mut [u8], next: &mut u32) {
    for pair in buf.chunks_exact_mut(2) {
        let sample = (((*next & 0xff) as i32 - 128) * 100) as i16;
        let bytes = sample.to_le_bytes();
        pair[0] = bytes[0];
        pair[1] = bytes[1];
        *next = next.wrapping_add(1);
    }
}

/// The shared wire fold (a checksum over every emitted byte plus a packet count
/// in the high bits), used by both the native sender and the C send callback.
fn fold(sum: &mut u64, packets: &mut u32, header: &[u8], payload: &[u8]) {
    for &b in header.iter().chain(payload.iter()) {
        *sum = sum.wrapping_add(b as u64);
    }
    *packets += 1;
}

fn checksum(sum: u64, packets: u32) -> u64 {
    sum.wrapping_add((packets as u64) << 32)
}

// ── Native Rust seams (the reference) ─────────────────────────────────────────

struct RampGrabber {
    next: u32,
}

impl FrameGrabber for RampGrabber {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, g2g_core::error::G2gError> {
        fill_ramp(buf, &mut self.next);
        Ok(buf.len())
    }
}

struct SumSender {
    sum: u64,
    packets: u32,
}

impl PacketSender for SumSender {
    async fn send(
        &mut self,
        header: &[u8; RTP_HEADER_LEN],
        payload: &[u8],
    ) -> Result<(), g2g_core::error::G2gError> {
        fold(&mut self.sum, &mut self.packets, header, payload);
        Ok(())
    }
}

// ── C seams (the same logic behind `extern "C"` function pointers) ────────────

struct CapCtx {
    next: u32,
}

extern "C" fn c_capture(ctx: *mut c_void, buf: *mut u8, len: usize) -> isize {
    // SAFETY: the test passes a live `&mut CapCtx` as `ctx` and a valid
    // `buf`/`len` from the lent ring slot.
    let (ctx, buf) =
        unsafe { (&mut *(ctx as *mut CapCtx), core::slice::from_raw_parts_mut(buf, len)) };
    fill_ramp(buf, &mut ctx.next);
    len as isize
}

struct SumCtx {
    sum: u64,
    packets: u32,
}

extern "C" fn c_send(
    ctx: *mut c_void,
    header: *const u8,
    header_len: usize,
    payload: *const u8,
    payload_len: usize,
) -> i32 {
    // SAFETY: the test passes a live `&mut SumCtx` as `ctx`; RtpSink passes
    // valid header/payload pointers for their lengths.
    let (ctx, header, payload) = unsafe {
        (
            &mut *(ctx as *mut SumCtx),
            core::slice::from_raw_parts(header, header_len),
            core::slice::from_raw_parts(payload, payload_len),
        )
    };
    fold(&mut ctx.sum, &mut ctx.packets, header, payload);
    0
}

/// Run the pipeline with the native Rust seams, to end of stream.
fn run_native() -> u64 {
    let cap_ring: StaticLendRing<1, CAP_BYTES> = StaticLendRing::new();
    let enc_ring: StaticLendRing<1, ULAW_BYTES> = StaticLendRing::new();
    // SAFETY: both rings outlive the pipeline (the runner drains before this
    // frame drops them), satisfying `with_ring`'s contract.
    let src = unsafe { GrabberSrc::with_ring(RampGrabber { next: 0 }, &cap_ring, FRAME_NS) }
        .with_frame_limit(FRAMES);
    // SAFETY: the encoder's ring likewise outlives the runner.
    let enc = unsafe { G711Enc::with_ring(Law::Mulaw, &enc_ring) };
    let mut rtp = RtpSink::new(SumSender { sum: 0, packets: 0 }, MediaClock::audio(8000), 0, SSRC, 0);
    g2g_core::drive_ready(run_source_transform_sink(src, enc, &mut rtp))
        .expect("native pipeline is synchronous")
        .expect("native pipeline runs clean");
    let sender = rtp.free();
    checksum(sender.sum, sender.packets)
}

/// Run the SAME pipeline with the C-callback seams, driven one frame at a time
/// through `step_source_sink` (the C-superloop runner), to `FRAMES` frames.
fn run_c_seams() -> u64 {
    let cap_ring: StaticLendRing<1, CAP_BYTES> = StaticLendRing::new();
    let enc_ring: StaticLendRing<1, ULAW_BYTES> = StaticLendRing::new();
    let mut cap_ctx = CapCtx { next: 0 };
    let mut sum_ctx = SumCtx { sum: 0, packets: 0 };
    // SAFETY: the callbacks stay valid for the whole run, and the ctx pointers
    // reference stack locals that outlive the pipeline below.
    let (grabber, sender) = unsafe {
        (
            CFrameGrabber::new(c_capture, &mut cap_ctx as *mut CapCtx as *mut c_void),
            CPacketSender::new(c_send, &mut sum_ctx as *mut SumCtx as *mut c_void),
        )
    };
    // SAFETY: both rings outlive the pipeline (each stepped frame drops within
    // its iteration before the rings drop at end of function).
    let mut src = unsafe { GrabberSrc::with_ring(grabber, &cap_ring, FRAME_NS) };
    // SAFETY: the encoder's ring likewise outlives the stepped pipeline.
    let enc = unsafe { G711Enc::with_ring(Law::Mulaw, &enc_ring) };
    let mut rtp = RtpSink::new(sender, MediaClock::audio(8000), 0, SSRC, 0);
    let mut sink = SinkChain(enc, &mut rtp);
    for i in 0..FRAMES {
        match step_source_sink(&mut src, &mut sink).expect("clean step") {
            Step::Advanced => {}
            other => panic!("frame {i}: expected Advanced, got {other:?}"),
        }
    }
    checksum(sum_ctx.sum, sum_ctx.packets)
}

#[test]
fn c_seams_are_byte_transparent_vs_native() {
    let native = run_native();
    let c = run_c_seams();
    assert_ne!(c, 0, "the C-seam pipeline actually emitted RTP");
    assert_eq!(
        c, native,
        "the C capture/send seams deliver exactly the wire the native seams do"
    );
}

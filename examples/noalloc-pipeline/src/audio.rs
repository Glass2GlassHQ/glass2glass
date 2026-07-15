//! The flagship deterministic audio graph (M644): `capture -> convert ->
//! resample -> mix -> encode -> RTP` composed as ONE static pipeline from the
//! real `g2g-mcu` elements, heap-free end to end. This is the reference
//! chain of the deterministic-audio wedge: the same graph an MCU sends from
//! a board (the [`PacketSender`] is the board's UDP stack there), proven
//! here with mock capture peripherals and a checksumming sender so every
//! byte of RTP output is asserted.
//!
//! Two capture branches feed a fan-in ([`run_sources_fanin_sink`], the M642
//! const-arity runner), each a fused linear chain in a source slot
//! ([`SourceChain`], the static `source ! transform` bin):
//!
//! - branch A: a DMIC-shaped 8 kHz S16LE capture (1 kHz tone), used as the
//!   mix's timing master;
//! - branch B: a SAI-shaped 48 kHz capture delivering 24-bit-in-32 slots
//!   (2 kHz tone) -> [`PcmConvert`] (the `convert` stage) -> [`Resampler`]
//!   48 -> 8 kHz (the `resample` stage);
//!
//! then `mix` ([`Mixer`], 0.5/0.5 Q15 gains) -> `encode` ([`G711Enc`],
//! mu-law) -> `RTP` ([`RtpSink`], PCMU/PT 0, one 10 ms packet per frame)
//! fused into the sink slot ([`SinkChain`]). The mix link's `Caps::Audio`
//! is negotiated at build (black-boxed intersect, like the video pipeline's
//! `Caps::Tensor` link), so the audio caps kind joins the proof coverage.
//!
//! Everything is fixed-point, so the wire bytes are bit-exact across
//! targets: [`AUDIO_EXPECTED_CHECKSUM`] is pinned by the host test
//! (`tests/flagship.rs`, which also validates the DSP semantics against an
//! independent float reference) and re-asserted on the emulated Cortex-M by
//! `g2g-qemu`, which is the determinism claim made machine-checkable.

use core::hint::black_box;

use g2g_core::error::G2gError;
use g2g_core::rtp::RTP_HEADER_LEN;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{
    drive_ready, run_sources_fanin_sink, AudioFormat, Caps, Chain, MediaClock, SinkChain,
    SourceChain,
};
use g2g_mcu::rtp::PacketSender;
use g2g_mcu::{
    FrameGrabber, G711Enc, GrabberSrc, Law, Mixer, PcmConvert, Resampler, RtpSink, SampleRate,
};

/// 10 ms frames: the chain's packet time.
const FRAME_NS: u64 = 10_000_000;
/// Half a second of audio.
pub const FRAMES: u32 = 50;
/// Branch A: 8 kHz S16 capture, 80 samples per frame.
const A_BYTES: usize = 80 * 2;
/// Branch B: 48 kHz capture in 32-bit slots, 480 samples per frame.
const B_BYTES: usize = 480 * 4;
/// B after `convert` (S16).
const CONV_BYTES: usize = 480 * 2;
/// B after `resample` (divide by 6) and the mix output: 80 samples S16.
const MIX_BYTES: usize = 80 * 2;
/// The encoded payload: one mu-law byte per sample.
const ULAW_BYTES: usize = 80;
/// The RTP stream identity (an MCU picks these at boot; fixed here so the
/// wire bytes, and with them the checksum, are fully deterministic).
const SSRC: u32 = 0x6767_6767;
const PT_PCMU: u8 = 0;

/// 1 kHz at 8 kHz sampling, amplitude 8000 (branch A's tone).
static TONE_A: [i16; 8] = [0, 5657, 8000, 5657, 0, -5657, -8000, -5657];
/// 2 kHz at 48 kHz sampling, amplitude 6000 (branch B's tone).
static TONE_B: [i16; 24] = [
    0, 1553, 3000, 4243, 5196, 5796, 6000, 5796, 5196, 4243, 3000, 1553, 0, -1553, -3000, -4243,
    -5196, -5796, -6000, -5796, -5196, -4243, -3000, -1553,
];

/// The wire checksum a correct run produces: the wrapping sum of every RTP
/// byte (header + payload) plus the packet count folded into the high bits
/// (byte sums stay far below 2^32, so the two never collide). Pinned by the
/// host test in `tests/flagship.rs`, which independently validates the
/// stream's structure and DSP content; the QEMU proof then re-asserts the
/// same constant on the Cortex-M ISA, making the cross-target bit-exactness
/// claim machine-checked.
pub const AUDIO_EXPECTED_CHECKSUM: u64 = 214_749_012_174;

/// A deterministic tone standing in for a capture DMA: repeats a period
/// table, S16LE (`wide = false`, the DMIC shape) or left-justified 32-bit
/// slots (`wide = true`, the SAI shape). Phase persists across frames. `pub`
/// so the M646 generated-graph proof (`mcugen-audio`) can drive its emitted
/// pipeline with the same capture peripherals this reference uses.
pub struct ToneGrabber {
    table: &'static [i16],
    pos: usize,
    wide: bool,
}

/// Branch A's capture peripheral: the 8 kHz S16 tone (the mix timing master).
pub fn tone_a() -> ToneGrabber {
    ToneGrabber { table: &TONE_A, pos: 0, wide: false }
}

/// Branch B's capture peripheral: the 48 kHz 24-in-32-slot tone.
pub fn tone_b() -> ToneGrabber {
    ToneGrabber { table: &TONE_B, pos: 0, wide: true }
}

impl ToneGrabber {
    fn next_sample(&mut self) -> i16 {
        let s = self.table.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        if self.pos >= self.table.len() {
            self.pos = 0;
        }
        s
    }
}

impl FrameGrabber for ToneGrabber {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        if self.wide {
            for slot in buf.chunks_exact_mut(4) {
                let bytes = ((self.next_sample() as i32) << 16).to_le_bytes();
                // Slice patterns, not copy_from_slice: no length-mismatch
                // panic path may enter the no-alloc archive.
                let [b0, b1, b2, b3] = slot else { continue };
                [*b0, *b1, *b2, *b3] = bytes;
            }
        } else {
            for pair in buf.chunks_exact_mut(2) {
                let bytes = self.next_sample().to_le_bytes();
                let [b0, b1] = pair else { continue };
                [*b0, *b1] = bytes;
            }
        }
        Ok(buf.len())
    }
}

/// A [`PacketSender`] that checksums the wire instead of sending it: the
/// proof-side stand-in for a board's UDP stack.
#[derive(Default)]
pub struct SumSender {
    sum: u64,
    packets: u32,
}

impl SumSender {
    /// The checksum in [`AUDIO_EXPECTED_CHECKSUM`]'s format.
    pub fn checksum(&self) -> u64 {
        self.sum.wrapping_add((self.packets as u64) << 32)
    }
}

impl PacketSender for SumSender {
    async fn send(&mut self, header: &[u8; RTP_HEADER_LEN], payload: &[u8]) -> Result<(), G2gError> {
        for &b in header.iter().chain(payload.iter()) {
            self.sum = self.sum.wrapping_add(b as u64);
        }
        self.packets += 1;
        Ok(())
    }
}

/// A [`SumSender`] that additionally stamps every packet with a
/// caller-supplied clock read: the timing/jitter report's probe (M645). The
/// QEMU harness passes a SysTick read and runs under `-icount`, where
/// virtual time is a pure function of the instruction stream, so the stamps
/// measure the pipeline's deterministic per-frame execution cost, not host
/// scheduling noise. Fixed storage, one `u32` per frame: heap-free like
/// everything else on this path.
pub struct TimedSumSender<F: FnMut() -> u32> {
    sum: SumSender,
    now: F,
    stamps: [u32; FRAMES as usize],
    n: usize,
}

impl<F: FnMut() -> u32> TimedSumSender<F> {
    /// A timing probe reading the clock through `now`.
    pub fn new(now: F) -> Self {
        Self { sum: SumSender::default(), now, stamps: [0; FRAMES as usize], n: 0 }
    }

    /// The checksum in [`AUDIO_EXPECTED_CHECKSUM`]'s format (the timing run
    /// re-verifies the wire, so a wrong pipeline cannot report timings).
    pub fn checksum(&self) -> u64 {
        self.sum.checksum()
    }

    /// The clock stamp taken as each packet completed, in emission order.
    pub fn stamps(&self) -> &[u32] {
        // `n` never exceeds the array (send() guards), but slice defensively.
        self.stamps.get(..self.n.min(FRAMES as usize)).unwrap_or(&[])
    }
}

impl<F: FnMut() -> u32> PacketSender for TimedSumSender<F> {
    async fn send(&mut self, header: &[u8; RTP_HEADER_LEN], payload: &[u8]) -> Result<(), G2gError> {
        self.sum.send(header, payload).await?;
        if let Some(slot) = self.stamps.get_mut(self.n) {
            *slot = (self.now)();
            self.n += 1;
        }
        Ok(())
    }
}

/// Negotiate the mix link's `Caps::Audio` (produced side vs accepted side).
/// Both sides pass through `black_box` so the audio arm of the un-inlined
/// `Caps::intersect` is genuinely in the archive, covered by the no-alloc +
/// panic-free proofs (the video pipeline does the same for `Caps::Tensor`).
fn negotiate_mix_link() -> Option<Caps> {
    let mixed =
        || Caps::Audio { format: AudioFormat::PcmS16Le, channels: 1, sample_rate: 8000 };
    let produced = black_box(mixed());
    let accepted = black_box(mixed());
    produced.intersect(&accepted).ok()
}

/// Build and run the flagship graph over stack-local rings, emitting every
/// RTP packet through `sender` and returning it for inspection (checksum on
/// the proof targets, full wire capture in the host test). A failed caps
/// negotiation returns the sender untouched (its zero checksum fails the
/// comparison honestly).
pub async fn run_audio_with<S: PacketSender>(sender: S) -> S {
    if negotiate_mix_link().is_none() {
        return sender;
    }
    let ring_a: StaticLendRing<1, A_BYTES> = StaticLendRing::new();
    let ring_b: StaticLendRing<1, B_BYTES> = StaticLendRing::new();
    let ring_conv: StaticLendRing<1, CONV_BYTES> = StaticLendRing::new();
    let ring_res: StaticLendRing<1, MIX_BYTES> = StaticLendRing::new();
    let ring_mix: StaticLendRing<1, MIX_BYTES> = StaticLendRing::new();
    let ring_enc: StaticLendRing<1, ULAW_BYTES> = StaticLendRing::new();

    // SAFETY (all four `with_ring`s): every ring above outlives the graph:
    // the runner drains the pipeline (each lent frame is dropped within its
    // iteration) before this future completes and drops the rings.
    let src_a =
        unsafe { GrabberSrc::with_ring(tone_a(), &ring_a, FRAME_NS) }.with_frame_limit(FRAMES);
    let cap_b =
        unsafe { GrabberSrc::with_ring(tone_b(), &ring_b, FRAME_NS) }.with_frame_limit(FRAMES);
    let convert = unsafe { PcmConvert::with_ring(&ring_conv) };
    let resample =
        unsafe { Resampler::with_ring(SampleRate::Hz48000, SampleRate::Hz8000, &ring_res) };
    let mixer = unsafe { Mixer::with_ring(16384, 16384, &ring_mix) };
    let encode = unsafe { G711Enc::with_ring(Law::Mulaw, &ring_enc) };

    // The whole flagship graph in one const-arity call: fused branch B in a
    // source slot, fused encode -> RTP in the sink slot.
    let src_b = SourceChain(cap_b, Chain(convert, resample));
    let mut rtp = RtpSink::new(sender, MediaClock::audio(8000), PT_PCMU, SSRC, 0);
    let _ = run_sources_fanin_sink(src_a, src_b, mixer, SinkChain(encode, &mut rtp)).await;
    rtp.free()
}

/// [`run_audio_with`] a checksumming sender, driven by the safe single-poll
/// executor: the proof-target entry ([`AUDIO_EXPECTED_CHECKSUM`] when
/// everything worked; a `Pending` pipeline yields 0, failing honestly).
pub fn run_audio() -> u64 {
    drive_ready(run_audio_with(SumSender::default()))
        .map(|s| s.checksum())
        .unwrap_or(0)
}

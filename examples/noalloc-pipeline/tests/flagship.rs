//! Host validation of the flagship audio graph (M644): capture -> convert ->
//! resample -> mix -> encode -> RTP as one static pipeline.
//!
//! Three layers of assertion:
//! 1. wire structure: 50 PCMU packets, sequential, timestamps in exact 8 kHz
//!    media ticks, marker on the first packet only;
//! 2. DSP semantics against an independent float reference: the decoded
//!    mu-law stream must contain the two source tones at their mixed
//!    amplitudes (least-squares projection at 1 kHz and 2 kHz) with small
//!    residual, proving convert, resample, mix, and encode did signal
//!    processing, not just plumbing;
//! 3. the pinned checksum: the exact wire bytes fold to
//!    [`AUDIO_EXPECTED_CHECKSUM`], the same constant `g2g-qemu` re-asserts on
//!    the Cortex-M ISA, which makes the fixed-point chain's cross-target
//!    bit-exactness a machine-checked claim rather than a slogan.

use g2g_core::error::G2gError;
use g2g_core::rtp::RTP_HEADER_LEN;
use g2g_core::drive_ready;
use g2g_mcu::g711::mulaw_decode;
use g2g_mcu::rtp::PacketSender;
use noalloc_pipeline::audio::{run_audio, run_audio_with, AUDIO_EXPECTED_CHECKSUM, FRAMES};
use noalloc_pipeline::{
    run, run_display_banded_with, run_display_with, StubBus, NoDelay, BANDS_PER_REFRESH,
    EXPECTED_CHECKSUM, PANEL_H, PANEL_W, STRIPE,
};

/// Captures every packet (header + payload concatenated), the host-side
/// inspection twin of the proof targets' checksumming sender.
#[derive(Default)]
struct Collect {
    packets: Vec<Vec<u8>>,
}

impl PacketSender for Collect {
    async fn send(&mut self, header: &[u8; RTP_HEADER_LEN], payload: &[u8]) -> Result<(), G2gError> {
        let mut pkt = header.to_vec();
        pkt.extend_from_slice(payload);
        self.packets.push(pkt);
        Ok(())
    }
}

fn wire() -> Vec<Vec<u8>> {
    drive_ready(run_audio_with(Collect::default())).expect("static graph never suspends").packets
}

#[test]
fn wire_structure_is_the_contracted_rtp_stream() {
    let packets = wire();
    assert_eq!(packets.len(), FRAMES as usize, "one 10 ms packet per frame");
    for (i, pkt) in packets.iter().enumerate() {
        assert_eq!(pkt.len(), RTP_HEADER_LEN + 80, "12-byte header + 80 mu-law bytes");
        assert_eq!(pkt[0], 0x80, "V=2, no padding/extension/CSRCs");
        assert_eq!(pkt[1] & 0x7F, 0, "static PT 0 = PCMU");
        assert_eq!(pkt[1] & 0x80 != 0, i == 0, "marker only on the first packet");
        assert_eq!(u16::from_be_bytes([pkt[2], pkt[3]]), i as u16, "sequential");
        let ts = u32::from_be_bytes([pkt[4], pkt[5], pkt[6], pkt[7]]);
        assert_eq!(ts, i as u32 * 80, "PTS in exact 8 kHz ticks: 80 per 10 ms");
    }
}

/// Least-squares projection of `x` onto sin/cos at `freq_hz` (8 kHz
/// sampling): the tone's amplitude regardless of phase. Written against
/// f64 math, fully independent of the fixed-point chain under test.
fn tone_amplitude(x: &[f64], freq_hz: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * freq_hz / 8000.0;
    let n = x.len() as f64;
    let (mut s, mut c) = (0.0, 0.0);
    for (i, &v) in x.iter().enumerate() {
        s += v * (w * i as f64).sin();
        c += v * (w * i as f64).cos();
    }
    (2.0 / n * s).hypot(2.0 / n * c)
}

#[test]
fn decoded_stream_contains_the_mixed_tones() {
    let packets = wire();
    let samples: Vec<f64> = packets
        .iter()
        .flat_map(|p| p[RTP_HEADER_LEN..].iter().map(|&b| mulaw_decode(b) as f64))
        .collect();
    assert_eq!(samples.len(), FRAMES as usize * 80);
    // Skip two frames of resampler warmup, then measure over whole periods
    // of both tones (3840 = 480 x 8 = 960 x 4).
    let x = &samples[160..4000];
    let amp_a = tone_amplitude(x, 1000.0);
    let amp_b = tone_amplitude(x, 2000.0);
    // Branch A: 8000 x 0.5 gain; branch B: 6000 x 0.5, through convert (a
    // lossless <<16 >>16 round trip) and the 48 -> 8 kHz resampler (passband
    // ~unity). Tolerance covers mu-law quantization + filter ripple.
    assert!((amp_a - 4000.0).abs() < 120.0, "1 kHz tone at half amplitude, got {amp_a:.1}");
    assert!((amp_b - 3000.0).abs() < 120.0, "2 kHz tone at half amplitude, got {amp_b:.1}");
    // The two tones must be essentially all of the signal: small residual
    // proves nothing else (distortion, aliasing, framing glitches) leaked in.
    let w1 = 2.0 * std::f64::consts::PI * 1000.0 / 8000.0;
    let w2 = 2.0 * std::f64::consts::PI * 2000.0 / 8000.0;
    let proj = |f: f64, phase_fn: fn(f64) -> f64| {
        let n = x.len() as f64;
        2.0 / n * x.iter().enumerate().map(|(i, &v)| v * phase_fn(f * i as f64)).sum::<f64>()
    };
    let (a1, b1) = (proj(w1, f64::sin), proj(w1, f64::cos));
    let (a2, b2) = (proj(w2, f64::sin), proj(w2, f64::cos));
    let residual_power: f64 = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let fit = a1 * (w1 * i as f64).sin()
                + b1 * (w1 * i as f64).cos()
                + a2 * (w2 * i as f64).sin()
                + b2 * (w2 * i as f64).cos();
            (v - fit) * (v - fit)
        })
        .sum::<f64>()
        / x.len() as f64;
    let signal_power = (amp_a * amp_a + amp_b * amp_b) / 2.0;
    let snr_db = 10.0 * (signal_power / residual_power).log10();
    assert!(snr_db > 30.0, "the stream is the two tones and little else, SNR {snr_db:.1} dB");
}

#[test]
fn checksum_is_pinned_and_deterministic() {
    // The checksum entry (what QEMU and the executor proofs run) must fold
    // the exact wire bytes this test inspected.
    let packets = wire();
    let byte_sum: u64 =
        packets.iter().flat_map(|p| p.iter()).fold(0u64, |s, &b| s.wrapping_add(b as u64));
    let expected = byte_sum.wrapping_add((packets.len() as u64) << 32);
    assert_eq!(
        run_audio(),
        expected,
        "the checksum entry reproduces the captured wire"
    );
    assert_eq!(
        run_audio(),
        AUDIO_EXPECTED_CHECKSUM,
        "the pinned cross-target constant (update AUDIO_EXPECTED_CHECKSUM only for an intentional pipeline change)"
    );
    // Bit-exact reproducibility on this target; QEMU extends the claim to
    // the Cortex-M ISA via the same constant.
    assert_eq!(wire(), packets, "two runs, identical wire bytes");
}

/// The video display pipeline (camera -> transform -> `SpiDisplaySink`) over
/// the stub panel. `run()`'s checksum is otherwise only checked by the C
/// harness + QEMU; assert it here too, and drive the board-agnostic
/// `run_display_with` (the exact entry the `g2g-esp32p4` esp-hal harness calls)
/// against a fresh stub bus so the generic seam is covered, not just the
/// stub-bound `run_async` wrapper.
#[test]
fn display_pipeline_puts_the_expected_bytes_on_the_panel() {
    assert_eq!(run(), EXPECTED_CHECKSUM, "the wire checksum is the pinned display protocol");

    // Same run, but driving the generic runner directly with a caller-supplied
    // panel (a real board hands its HAL's SpiDevice/OutputPin/DelayNs the same
    // way): the checksum must match, proving the backend seam is transparent.
    let bus = StubBus::new();
    let mut delay = NoDelay;
    drive_ready(run_display_with(bus.spi(), bus.dc(), &mut delay))
        .expect("static chain never suspends")
        .expect("the display pipeline runs to completion");
    assert_eq!(bus.checksum(), EXPECTED_CHECKSUM, "board-agnostic runner, identical wire");
}

/// The Tier-1.5 full-panel path: 240x240 streamed to a banded sink from a
/// 15 KB ring. The precise per-band wire is asserted in g2g-mcu's
/// `m629_spi_display`; here we prove the runner wires the banded source + sink
/// and completes one full refresh deterministically over the stub panel.
#[test]
fn banded_full_panel_refresh_runs_over_the_stub() {
    assert_eq!(PANEL_H % STRIPE, 0, "the stripe tiles the panel");
    let run_once = || {
        let bus = StubBus::new();
        let mut delay = NoDelay;
        drive_ready(run_display_banded_with(bus.spi(), bus.dc(), &mut delay))
            .expect("static chain never suspends")
            .expect("one full 240x240 refresh completes");
        bus.checksum()
    };
    let sum = run_once();
    // Something was actually written (window params + pixels), and the wire is
    // deterministic across refreshes. The checksum sums byte *values*, so a
    // mostly-black test pattern is small; the exact per-band wire is asserted
    // in g2g-mcu's m629_spi_display.
    assert!(sum > 0, "a full refresh put bytes on the wire");
    assert_eq!(sum, run_once(), "two refreshes, identical wire");
    assert_eq!(BANDS_PER_REFRESH, 15, "240 / 16 = 15 bands per refresh");
    let _ = (PANEL_W, PANEL_H);
}

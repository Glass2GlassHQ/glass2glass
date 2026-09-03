//! M1152: `AudioResample`'s stop-band rejection under heavy downsampling. A
//! 19 kHz tone at 96 kHz has no representation below the 8 kHz output's
//! Nyquist, so every sample the resampler emits for it is kernel leakage, and
//! the measured level is the kernel's rejection at a 12:1 ratio. A second probe
//! in the middle of the output band checks the same kernel is still flat where
//! signal is meant to pass.

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, ChannelLayout, Frame, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelinePacket, PushOutcome,
};
use g2g_plugins::audioresample::AudioResample;

/// The probe's rate pair: a 12:1 decimation.
const IN_RATE: u32 = 96_000;
const OUT_RATE: u32 = 8_000;

/// Out-of-band tone, above the output Nyquist and folding to 3 kHz. The 12:1
/// ratio steps 3/8 of a turn per output sample here, so the measurement covers
/// eight phases of the tone instead of landing on its zero crossings (which
/// 20 kHz, half a turn per output sample, would do).
const STOPBAND_TONE_HZ: f64 = 19_000.0;

/// In-band tone, well inside the output's 4 kHz Nyquist.
const PASSBAND_TONE_HZ: f64 = 1_000.0;

/// Peak amplitude of both probe tones.
const TONE_AMPLITUDE: f64 = 0.5;

/// Probe length in input samples: one second at [`IN_RATE`].
const INPUT_FRAMES: usize = IN_RATE as usize;

/// Output samples skipped before measuring: at the head of the stream the
/// kernel window reaches past sample 0 and reads it held.
const WARMUP_OUTPUT_FRAMES: usize = 512;

/// Rejection asserted at [`STOPBAND_TONE_HZ`]. The kernel measures -132 dB,
/// the floor its f32 accumulation can reach; a fixed-width window at this
/// ratio measures -103 dB, so the threshold sits between them.
const STOPBAND_REJECTION_DB: f64 = -120.0;

/// Pass-band amplitude error allowed at 1 kHz.
const PASSBAND_TOLERANCE_DB: f64 = 0.5;

#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn mono_f32(rate: u32) -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmF32Le,
        channels: 1,
        sample_rate: rate,
        channel_layout: ChannelLayout::UNSPECIFIED,
    }
}

fn tone_bytes(frequency_hz: f64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(INPUT_FRAMES * size_of::<f32>());
    for n in 0..INPUT_FRAMES {
        // reduce the phase to one turn first, so the far end of the probe keeps
        // the same precision as the near end.
        let turns = frequency_hz * n as f64 / f64::from(IN_RATE);
        let value = TONE_AMPLITUDE * (std::f64::consts::TAU * turns.fract()).sin();
        bytes.extend_from_slice(&(value as f32).to_le_bytes());
    }
    bytes
}

fn frame(bytes: Vec<u8>) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

/// Resample one probe tone from [`IN_RATE`] to [`OUT_RATE`] at the default
/// quality and return the output's RMS over the tone's own RMS, in dB.
fn relative_output_db(frequency_hz: f64) -> f64 {
    let mut resample = AudioResample::new(OUT_RATE);
    resample
        .configure_pipeline(&mono_f32(IN_RATE))
        .expect("f32 mono input accepted");
    let mut sink = CollectSink::default();
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(resample.process(
            PipelinePacket::DataFrame(frame(tone_bytes(frequency_hz))),
            &mut sink,
        ))
        .expect("the resampler processed the probe");

    let PipelinePacket::DataFrame(out) = sink.packets.last().expect("an output packet") else {
        panic!("expected a DataFrame downstream");
    };
    let bytes = out
        .domain
        .require_system_slice("test")
        .expect("system memory out");
    let samples: Vec<f32> = bytes
        .as_chunks::<{ size_of::<f32>() }>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert!(
        samples.len() > WARMUP_OUTPUT_FRAMES,
        "the probe outlasts the kernel's warm-up, got {} samples",
        samples.len()
    );

    let measured = &samples[WARMUP_OUTPUT_FRAMES..];
    let mean_square = measured
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / measured.len() as f64;
    let tone_rms = TONE_AMPLITUDE / 2f64.sqrt();
    20.0 * (mean_square.sqrt() / tone_rms).log10()
}

#[test]
fn out_of_band_tone_is_rejected_under_heavy_downsampling() {
    let rejection = relative_output_db(STOPBAND_TONE_HZ);
    assert!(
        rejection < STOPBAND_REJECTION_DB,
        "{STOPBAND_TONE_HZ} Hz leaked into the {OUT_RATE} Hz output at {rejection} dB, \
         want under {STOPBAND_REJECTION_DB} dB"
    );
}

#[test]
fn in_band_tone_keeps_its_amplitude_under_heavy_downsampling() {
    let level = relative_output_db(PASSBAND_TONE_HZ);
    assert!(
        level.abs() < PASSBAND_TOLERANCE_DB,
        "{PASSBAND_TONE_HZ} Hz came out at {level} dB, want within \
         {PASSBAND_TOLERANCE_DB} dB"
    );
}

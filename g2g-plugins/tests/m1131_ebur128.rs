//! M1131: `ebur128` measures ITU-R BS.1770-4 loudness and passes the audio
//! through untouched.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, PushOutcome,
};
use g2g_plugins::ebur128::Ebur128;
use g2g_plugins::registry::default_registry;

const RATE: u32 = 48_000;
const CHANNELS: usize = 2;
const TONE_HZ: f64 = 1000.0;
/// Full scale: the sine touches +-1.0.
const TONE_AMPLITUDE: f64 = 1.0;
/// One buffer per 100 ms, which is also the meter's default report interval.
const BUFFER_FRAMES: usize = (RATE / 10) as usize;

/// BS.1770-4's calibration term, the `-0.691` of
/// `L = -0.691 + 10 log10(sum_channels G * z)`.
const LOUDNESS_OFFSET_LU: f64 = -0.691;
/// ffmpeg's `ebur128` filter on the same signal
/// (`ffmpeg -f lavfi -i "aevalsrc=exprs=sin(2*PI*1000*t)|sin(2*PI*1000*t):s=48000:d=20:c=stereo"
/// -af ebur128 -f null -`) reports this integrated loudness. It prints one
/// decimal, so that is how close the two can be held.
const FFMPEG_INTEGRATED_LUFS: f64 = 0.0;
const FFMPEG_PRINT_RESOLUTION_LU: f64 = 0.05;
/// The meter runs its biquads in f32 and its logarithm through the crate's
/// `no_std` `log2`, so a hundredth of a LU is as tight as a derived value holds.
const DERIVED_TOLERANCE_LU: f64 = 0.01;

/// BS.1770-4's gated-measurement grid: 400 ms blocks overlapping by 75%, and the
/// absolute gate they have to clear.
const BLOCK_DURATION_MS: usize = 400;
const STEP_DURATION_MS: usize = 100;
const MS_PER_SECOND: usize = 1_000;
const ABSOLUTE_GATE_LUFS: f64 = -70.0;
/// How far under the gate the quiet passage of the gating test sits.
const GATE_MARGIN_LU: f64 = 10.0;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Default)]
struct Collect {
    bytes: Vec<u8>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            if let Some(slice) = frame.domain.as_system_slice() {
                self.bytes.extend_from_slice(slice);
            }
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmF32Le,
        channels: CHANNELS as u8,
        sample_rate: RATE,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// `frames` sample frames of a 1 kHz sine at `amplitude` on every channel, as
/// interleaved F32LE.
fn sine(frames: usize, start_frame: usize, amplitude: f64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(frames * CHANNELS * 4);
    for index in 0..frames {
        let phase = core::f64::consts::TAU * TONE_HZ * (start_frame + index) as f64 / RATE as f64;
        let value = (amplitude * phase.sin()) as f32;
        for _ in 0..CHANNELS {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn frame(bytes: Vec<u8>) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    )
}

/// Push `bytes` through the meter in 100 ms buffers, then `Eos`.
fn measure(meter: &mut Ebur128, bytes: &[u8]) -> Collect {
    let mut sink = Collect::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime.block_on(async {
        for buffer in bytes.chunks(BUFFER_FRAMES * CHANNELS * 4) {
            meter
                .process(PipelinePacket::DataFrame(frame(buffer.to_vec())), &mut sink)
                .await
                .unwrap();
        }
        meter.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    });
    sink
}

fn configured() -> Ebur128 {
    let mut meter = Ebur128::new();
    meter.configure_pipeline(&caps()).unwrap();
    meter
}

/// The loudness a sine at `amplitude` on every channel has to measure.
///
/// Each channel's mean square is `amplitude^2 / 2`, K-weighting scales it by
/// `|H(1 kHz)|^2`, and both channels carry BS.1770's front weight of 1, so the
/// weighted sum is `channels * amplitude^2 / 2 * |H|^2` and
/// `L = -0.691 + 10 log10(that)`.
fn derived_sine_loudness(meter: &Ebur128, amplitude: f64) -> f64 {
    let response = meter.response_at(TONE_HZ);
    let weighted_power = CHANNELS as f64 * (amplitude * amplitude / 2.0) * response * response;
    LOUDNESS_OFFSET_LU + 10.0 * weighted_power.log10()
}

/// The amplitude a 1 kHz sine needs to measure `loudness` LUFS, the inverse of
/// [`derived_sine_loudness`].
fn amplitude_for_loudness(meter: &Ebur128, loudness: f64) -> f64 {
    TONE_AMPLITUDE * 10f64.powf((loudness - derived_sine_loudness(meter, TONE_AMPLITUDE)) / 20.0)
}

#[test]
fn a_full_scale_sine_measures_the_derived_loudness() {
    let mut meter = configured();
    let expected = derived_sine_loudness(&meter, TONE_AMPLITUDE);
    measure(&mut meter, &sine(RATE as usize * 20, 0, TONE_AMPLITUDE));

    let integrated = meter.integrated_lufs().expect("20 s of tone is measured");
    assert!(
        (integrated - expected).abs() < DERIVED_TOLERANCE_LU,
        "integrated {integrated} LUFS against the derived {expected} LUFS"
    );
    assert!(
        (expected - FFMPEG_INTEGRATED_LUFS).abs() < FFMPEG_PRINT_RESOLUTION_LU,
        "the derived {expected} LUFS is what ffmpeg's ebur128 prints"
    );
    // over a steady tone the three windows agree.
    let momentary = meter.momentary_lufs().expect("a 400 ms block is measured");
    let short_term = meter.short_term_lufs().expect("a 3 s window is measured");
    assert!(
        (momentary - expected).abs() < DERIVED_TOLERANCE_LU,
        "{momentary}"
    );
    assert!(
        (short_term - expected).abs() < DERIVED_TOLERANCE_LU,
        "{short_term}"
    );
}

#[test]
fn a_passage_under_the_absolute_gate_stays_out_of_the_integrated_measurement() {
    let tone_ms = 5_000usize;
    let tone_frames = RATE as usize * tone_ms / MS_PER_SECOND;

    let mut meter = configured();
    // A tone this quiet sits well below the -70 LUFS gate, but is not digital
    // silence, so the gate itself is what has to reject it.
    let quiet_amplitude = amplitude_for_loudness(&meter, ABSOLUTE_GATE_LUFS - GATE_MARGIN_LU);
    let mut bytes = sine(tone_frames, 0, TONE_AMPLITUDE);
    bytes.extend_from_slice(&sine(tone_frames, tone_frames, quiet_amplitude));
    measure(&mut meter, &bytes);
    let with_quiet_tail = meter
        .integrated_lufs()
        .expect("the loud passage is measured");

    // Only the blocks holding some of the loud passage survive the gate: the
    // whole ones at full power, and the ones straddling the two passages at the
    // fraction of a block the loud passage still covers.
    let whole_blocks = (tone_ms - BLOCK_DURATION_MS) / STEP_DURATION_MS + 1;
    let straddling = BLOCK_DURATION_MS / STEP_DURATION_MS - 1;
    let straddling_power: f64 = (1..=straddling)
        .map(|steps| (BLOCK_DURATION_MS - steps * STEP_DURATION_MS) as f64)
        .sum::<f64>()
        / BLOCK_DURATION_MS as f64;
    let expected = derived_sine_loudness(&meter, TONE_AMPLITUDE)
        + 10.0
            * ((whole_blocks as f64 + straddling_power) / (whole_blocks + straddling) as f64)
                .log10();

    assert!(
        (with_quiet_tail - expected).abs() < DERIVED_TOLERANCE_LU,
        "integrated {with_quiet_tail} LUFS against the derived {expected} LUFS; \
         an ungated mean over an equally long quiet tail would have lost 3 LU"
    );
    // the momentary window is ungated, so it does follow the quiet passage down.
    let momentary = meter.momentary_lufs().expect("the quiet tail is measured");
    assert!(
        (momentary - (ABSOLUTE_GATE_LUFS - GATE_MARGIN_LU)).abs() < DERIVED_TOLERANCE_LU,
        "momentary {momentary} LUFS at the end of the quiet tail"
    );
}

/// The runner re-announces the solved caps mid-stream, which must not restart
/// the integrated measurement.
#[test]
fn unchanged_caps_do_not_reset_the_measurement() {
    let mut meter = configured();
    let mut sink = Collect::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let seconds = 2usize;
    runtime.block_on(async {
        for second in 0..seconds {
            let start = second * RATE as usize;
            meter
                .process(
                    PipelinePacket::DataFrame(frame(sine(RATE as usize, start, TONE_AMPLITUDE))),
                    &mut sink,
                )
                .await
                .unwrap();
            meter
                .process(PipelinePacket::CapsChanged(caps()), &mut sink)
                .await
                .unwrap();
        }
        meter.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    });

    let expected = derived_sine_loudness(&meter, TONE_AMPLITUDE);
    let integrated = meter
        .integrated_lufs()
        .expect("the caps announcement did not wipe the measurement");
    assert!(
        (integrated - expected).abs() < DERIVED_TOLERANCE_LU,
        "integrated {integrated} LUFS after {seconds} caps announcements"
    );
}

#[test]
fn the_meter_passes_the_audio_through_byte_for_byte() {
    let mut meter = configured();
    let input = sine(BUFFER_FRAMES * 3, 0, TONE_AMPLITUDE);
    let collected = measure(&mut meter, &input);
    assert_eq!(collected.bytes, input);
}

#[test]
fn post_messages_false_leaves_the_meter_silent() {
    let mut meter = configured();
    meter
        .set_property("post-messages", g2g_core::PropValue::Bool(false))
        .unwrap();
    let input = sine(RATE as usize, 0, TONE_AMPLITUDE);
    let collected = measure(&mut meter, &input);
    assert_eq!(collected.bytes, input, "the audio still flows");
    assert!(meter.integrated_lufs().is_none(), "nothing was measured");
}

#[tokio::test]
async fn launch_runs_the_meter_in_a_pipeline() {
    let registry = default_registry();
    let graph = parse_launch(
        &registry,
        "audiotestsrc num-buffers=5 ! ebur128 interval=200000000 ! fakesink",
    )
    .expect("the ebur128 pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the ebur128 pipeline runs");
    assert_eq!(stats.frames_consumed, 5, "a passthrough drops nothing");
}

//! M1085 audio filters: `audiochannelmix`, `audiomixmatrix`, `stereo`,
//! `audiofirfilter`, `audioiirfilter`, `removesilence`, `audiobuffersplit` and
//! `speed`. Every assertion drives the real element through `process` and
//! measures what came out: the gains a mix applied, the impulse response a
//! kernel produced, the buffers a re-framer cut, the samples a rate change
//! left. Expected values are derived from the named constants, never spelled
//! out as a literal output.
#![cfg(feature = "std")]

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::parse_launch;
use g2g_core::segment::Segment;
use g2g_core::{
    AsyncElement, AudioFormat, Bus, BusMessage, Caps, G2gError, MemoryDomain, OutputSink,
    PropValue, PushOutcome,
};
use g2g_plugins::audiobuffersplit::AudioBufferSplit;
use g2g_plugins::audiochannelmix::AudioChannelMix;
use g2g_plugins::audiofirfilter::AudioFirFilter;
use g2g_plugins::audioiirfilter::AudioIirFilter;
use g2g_plugins::audiomixmatrix::{AudioMixMatrix, MixMatrixMode};
use g2g_plugins::registry::default_registry;
use g2g_plugins::removesilence::{RemoveSilence, VAD_WINDOW_SAMPLES};
use g2g_plugins::speed::Speed;
use g2g_plugins::stereo::Stereo;

const RATE: u32 = 48_000;
const NS_PER_SECOND: u64 = 1_000_000_000;

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
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

impl Collect {
    /// Every data frame's samples, decoded from F32LE and concatenated.
    fn samples(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for frame in self.data_frames() {
            let bytes = frame.domain.as_system_slice().expect("system frame");
            for chunk in bytes.as_chunks::<4>().0 {
                out.push(f32::from_le_bytes(*chunk));
            }
        }
        out
    }

    fn data_frames(&self) -> Vec<&Frame> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    /// The caps of the first `CapsChanged` the element emitted.
    fn first_caps(&self) -> Option<&Caps> {
        self.packets.iter().find_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c),
            _ => None,
        })
    }

    fn segments(&self) -> Vec<&Segment> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::Segment(s) => Some(s),
                _ => None,
            })
            .collect()
    }
}

fn caps(channels: u8) -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmF32Le,
        channels,
        sample_rate: RATE,
    }
}

fn to_frame(samples: &[f32], channels: usize, pts_ns: u64) -> PipelinePacket {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    let frames = (samples.len() / channels) as u64;
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: frames * NS_PER_SECOND / RATE as u64,
            ..Default::default()
        },
        0,
    ))
}

/// Nanoseconds `frames` sample frames occupy at [`RATE`].
fn frames_to_ns(frames: u64) -> u64 {
    frames * NS_PER_SECOND / RATE as u64
}

/// Push one buffer through a configured element and collect what came out.
async fn run_one<E: AsyncElement>(
    element: &mut E,
    samples: &[f32],
    channels: usize,
    pts_ns: u64,
) -> Collect {
    let mut out = Collect::default();
    element
        .process(to_frame(samples, channels, pts_ns), &mut out)
        .await
        .expect("the element accepts the buffer");
    out
}

// ---------------------------------------------------------------------------
// audiochannelmix
// ---------------------------------------------------------------------------

/// A stereo pair whose two channels are plainly distinguishable.
const LEFT_SAMPLE: f32 = 0.5;
const RIGHT_SAMPLE: f32 = -0.25;
/// Gains for a half-and-half fold of one side into the other.
const HALF_GAIN: f64 = 0.5;

#[tokio::test]
async fn audiochannelmix_applies_each_of_the_four_gains() {
    let mut mix = AudioChannelMix::new()
        .with_left_to_left(1.0 - HALF_GAIN)
        .with_left_to_right(HALF_GAIN)
        .with_right_to_left(HALF_GAIN)
        .with_right_to_right(1.0 - HALF_GAIN);
    mix.configure_pipeline(&caps(2)).expect("stereo configures");

    let input = [LEFT_SAMPLE, RIGHT_SAMPLE];
    let out = run_one(&mut mix, &input, 2, 0).await;
    let mixed = out.samples();

    let left = (1.0 - HALF_GAIN) * LEFT_SAMPLE as f64 + HALF_GAIN * RIGHT_SAMPLE as f64;
    let right = HALF_GAIN * LEFT_SAMPLE as f64 + (1.0 - HALF_GAIN) * RIGHT_SAMPLE as f64;
    assert_eq!(mixed.len(), input.len());
    assert!((mixed[0] as f64 - left).abs() < 1e-6, "got {}", mixed[0]);
    assert!((mixed[1] as f64 - right).abs() < 1e-6, "got {}", mixed[1]);
    // shape untouched.
    assert_eq!(out.first_caps(), Some(&caps(2)));
}

#[tokio::test]
async fn audiochannelmix_refuses_a_mono_stream() {
    let mut mix = AudioChannelMix::new();
    assert_eq!(
        mix.configure_pipeline(&caps(1)).unwrap_err(),
        G2gError::CapsMismatch
    );
}

// ---------------------------------------------------------------------------
// audiomixmatrix
// ---------------------------------------------------------------------------

/// A stereo-to-mono downmix, one gain per input channel.
const DOWNMIX_LEFT_GAIN: f64 = 0.25;
const DOWNMIX_RIGHT_GAIN: f64 = 0.75;

#[tokio::test]
async fn audiomixmatrix_downmix_is_the_weighted_sum_and_changes_the_caps() {
    let matrix = format!("{DOWNMIX_LEFT_GAIN},{DOWNMIX_RIGHT_GAIN}");
    let mut mix = AudioMixMatrix::new()
        .with_in_channels(2)
        .with_out_channels(1)
        .with_matrix(&matrix);
    mix.configure_pipeline(&caps(2)).expect("stereo configures");

    let input = [LEFT_SAMPLE, RIGHT_SAMPLE, RIGHT_SAMPLE, LEFT_SAMPLE];
    let out = run_one(&mut mix, &input, 2, 0).await;
    let mixed = out.samples();

    assert_eq!(mixed.len(), input.len() / 2, "two channels folded into one");
    let first = DOWNMIX_LEFT_GAIN * LEFT_SAMPLE as f64 + DOWNMIX_RIGHT_GAIN * RIGHT_SAMPLE as f64;
    let second = DOWNMIX_LEFT_GAIN * RIGHT_SAMPLE as f64 + DOWNMIX_RIGHT_GAIN * LEFT_SAMPLE as f64;
    assert!((mixed[0] as f64 - first).abs() < 1e-6, "got {}", mixed[0]);
    assert!((mixed[1] as f64 - second).abs() < 1e-6, "got {}", mixed[1]);
    // the mono output caps reach downstream before the first frame.
    assert_eq!(out.first_caps(), Some(&caps(1)));
    assert_eq!(mix.intercept_caps(&caps(2)).unwrap(), caps(1));
}

#[tokio::test]
async fn audiomixmatrix_first_channels_keeps_the_leading_channels() {
    let mut mix = AudioMixMatrix::new()
        .with_mode(MixMatrixMode::FirstChannels)
        .with_out_channels(2);
    mix.configure_pipeline(&caps(4)).expect("quad configures");

    let input = [LEFT_SAMPLE, RIGHT_SAMPLE, 1.0, -1.0];
    let out = run_one(&mut mix, &input, 4, 0).await;
    assert_eq!(out.samples(), vec![LEFT_SAMPLE, RIGHT_SAMPLE]);
    assert_eq!(out.first_caps(), Some(&caps(2)));
}

#[tokio::test]
async fn audiomixmatrix_refuses_a_matrix_that_does_not_fit_the_stream() {
    let mut mix = AudioMixMatrix::new()
        .with_in_channels(2)
        .with_out_channels(1)
        .with_matrix("0.5,0.5");
    assert_eq!(
        mix.configure_pipeline(&caps(4)).unwrap_err(),
        G2gError::CapsMismatch
    );
}

// ---------------------------------------------------------------------------
// stereo
// ---------------------------------------------------------------------------

/// gst keeps the widening factor ten times the `stereo` property.
const STEREO_FACTOR_SCALE: f64 = 10.0;
/// A widening setting whose factor (2) is easy to check by hand.
const WIDENING: f64 = 0.2;

#[tokio::test]
async fn stereo_widens_the_channel_difference_and_leaves_a_centred_pair() {
    let mut widen = Stereo::new().with_stereo(WIDENING);
    widen
        .configure_pipeline(&caps(2))
        .expect("stereo configures");

    // frame 0 is centred (both channels identical), frame 1 is off centre.
    let input = [LEFT_SAMPLE, LEFT_SAMPLE, LEFT_SAMPLE, RIGHT_SAMPLE];
    let out = run_one(&mut widen, &input, 2, 0).await;
    let widened = out.samples();

    assert_eq!(widened[0], LEFT_SAMPLE, "a centred pair is untouched");
    assert_eq!(widened[1], LEFT_SAMPLE);

    let average = (LEFT_SAMPLE as f64 + RIGHT_SAMPLE as f64) / 2.0;
    let factor = WIDENING * STEREO_FACTOR_SCALE;
    let left = average + (LEFT_SAMPLE as f64 - average) * factor;
    let right = average + (RIGHT_SAMPLE as f64 - average) * factor;
    assert!(
        (widened[2] as f64 - left).abs() < 1e-6,
        "got {}",
        widened[2]
    );
    assert!(
        (widened[3] as f64 - right).abs() < 1e-6,
        "got {}",
        widened[3]
    );
}

#[tokio::test]
async fn stereo_inactive_is_a_pass_through() {
    let mut widen = Stereo::new().with_active(false).with_stereo(WIDENING);
    widen
        .configure_pipeline(&caps(2))
        .expect("stereo configures");
    let input = [LEFT_SAMPLE, RIGHT_SAMPLE];
    let out = run_one(&mut widen, &input, 2, 0).await;
    assert_eq!(out.samples(), input.to_vec());
}

// ---------------------------------------------------------------------------
// audiofirfilter
// ---------------------------------------------------------------------------

/// A three-tap kernel whose taps are distinct, so a recovered impulse response
/// pins the tap order as well as the values.
const TEST_KERNEL: [f64; 3] = [0.25, 0.5, 0.125];

fn kernel_text(taps: &[f64]) -> String {
    taps.iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

#[tokio::test]
async fn audiofirfilter_of_an_impulse_returns_the_kernel() {
    let mut fir = AudioFirFilter::new().with_kernel(&kernel_text(&TEST_KERNEL));
    fir.configure_pipeline(&caps(1)).expect("mono configures");

    // an impulse followed by enough zeros to shift the whole kernel out.
    let mut input = vec![0.0f32; TEST_KERNEL.len()];
    input[0] = 1.0;
    let mut out = run_one(&mut fir, &input, 1, 0).await;
    fir.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("eos drains the tail");

    let response = out.samples();
    assert_eq!(response.len(), TEST_KERNEL.len());
    for (index, tap) in TEST_KERNEL.iter().enumerate() {
        assert!(
            (response[index] as f64 - tap).abs() < 1e-6,
            "tap {index}: got {}",
            response[index]
        );
    }
}

#[tokio::test]
async fn audiofirfilter_latency_drops_that_many_leading_samples() {
    const LATENCY_SAMPLES: u64 = 1;
    let mut fir = AudioFirFilter::new()
        .with_kernel(&kernel_text(&TEST_KERNEL))
        .with_latency(LATENCY_SAMPLES);
    fir.configure_pipeline(&caps(1)).expect("mono configures");

    let mut input = vec![0.0f32; TEST_KERNEL.len()];
    input[0] = 1.0;
    let out = run_one(&mut fir, &input, 1, 0).await;
    let response = out.samples();
    // the head is gone, so the response starts at the second tap.
    assert_eq!(response.len(), TEST_KERNEL.len() - LATENCY_SAMPLES as usize);
    assert!((response[0] as f64 - TEST_KERNEL[1]).abs() < 1e-6);
}

#[tokio::test]
async fn audiofirfilter_rejects_a_kernel_that_is_not_a_number_list() {
    let mut fir = AudioFirFilter::new();
    assert!(fir
        .set_property("kernel", PropValue::Str("0.25,oops".into()))
        .is_err());
}

// ---------------------------------------------------------------------------
// audioiirfilter
// ---------------------------------------------------------------------------

/// A one-pole low-pass: `y[n] = POLE_GAIN * x[n] + POLE_GAIN * y[n-1]`.
const POLE_GAIN: f64 = 0.5;
/// Output samples the impulse-response check walks.
const IIR_RESPONSE_SAMPLES: usize = 4;

#[tokio::test]
async fn audioiirfilter_one_pole_matches_the_hand_computed_recurrence() {
    let mut iir = AudioIirFilter::new()
        .with_b(&POLE_GAIN.to_string())
        .with_a(&format!("1,{}", -POLE_GAIN));
    iir.configure_pipeline(&caps(1)).expect("mono configures");

    let mut input = vec![0.0f32; IIR_RESPONSE_SAMPLES];
    input[0] = 1.0;
    let out = run_one(&mut iir, &input, 1, 0).await;
    let response = out.samples();

    // impulse response of y[n] = g*x[n] + g*y[n-1] is g^(n+1).
    assert_eq!(response.len(), IIR_RESPONSE_SAMPLES);
    let mut expected = POLE_GAIN;
    for (index, value) in response.iter().enumerate() {
        assert!(
            (*value as f64 - expected).abs() < 1e-6,
            "sample {index}: got {value}, expected {expected}"
        );
        expected *= POLE_GAIN;
    }
}

#[tokio::test]
async fn audioiirfilter_state_carries_across_buffers() {
    let mut iir = AudioIirFilter::new()
        .with_b(&POLE_GAIN.to_string())
        .with_a(&format!("1,{}", -POLE_GAIN));
    iir.configure_pipeline(&caps(1)).expect("mono configures");

    let impulse = [1.0f32];
    let first = run_one(&mut iir, &impulse, 1, 0).await;
    assert!((first.samples()[0] as f64 - POLE_GAIN).abs() < 1e-6);

    // the decay continues into the next buffer rather than restarting.
    let silence = [0.0f32];
    let second = run_one(&mut iir, &silence, 1, frames_to_ns(1)).await;
    assert!((second.samples()[0] as f64 - POLE_GAIN * POLE_GAIN).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// removesilence
// ---------------------------------------------------------------------------

/// One test buffer, long enough to fill the detector's zero-crossing window.
const VAD_BUFFER_FRAMES: usize = VAD_WINDOW_SAMPLES * 4;
/// A tone well below half the sample rate, so it crosses zero rarely, which is
/// what the reference's detector reads as voice.
const VOICE_TONE_HZ: f64 = 1000.0;
const VOICE_AMPLITUDE: f64 = 0.5;

fn voice(frames: usize) -> Vec<f32> {
    (0..frames)
        .map(|i| {
            let t = i as f64 / RATE as f64;
            (VOICE_AMPLITUDE * (core::f64::consts::TAU * VOICE_TONE_HZ * t).sin()) as f32
        })
        .collect()
}

#[tokio::test]
async fn removesilence_drops_a_silent_buffer_and_keeps_a_loud_one() {
    let mut strip = RemoveSilence::new().with_remove(true);
    strip.configure_pipeline(&caps(1)).expect("mono configures");

    let loud = voice(VAD_BUFFER_FRAMES);
    let kept = run_one(&mut strip, &loud, 1, 0).await;
    assert_eq!(
        kept.data_frames().len(),
        1,
        "a tone reaches the output untouched"
    );
    assert_eq!(kept.samples(), loud);

    let quiet = vec![0.0f32; VAD_BUFFER_FRAMES];
    let dropped = run_one(
        &mut strip,
        &quiet,
        1,
        frames_to_ns(VAD_BUFFER_FRAMES as u64),
    )
    .await;
    assert!(
        dropped.data_frames().is_empty(),
        "silence never reaches the output"
    );
}

#[tokio::test]
async fn removesilence_without_remove_forwards_the_silence() {
    let mut strip = RemoveSilence::new();
    strip.configure_pipeline(&caps(1)).expect("mono configures");
    let quiet = vec![0.0f32; VAD_BUFFER_FRAMES];
    let out = run_one(&mut strip, &quiet, 1, 0).await;
    assert_eq!(out.data_frames().len(), 1);
}

#[tokio::test]
async fn removesilence_squash_closes_the_gap_the_drop_left() {
    let mut strip = RemoveSilence::new().with_remove(true).with_squash(true);
    strip.configure_pipeline(&caps(1)).expect("mono configures");

    let buffer_ns = frames_to_ns(VAD_BUFFER_FRAMES as u64);
    let quiet = vec![0.0f32; VAD_BUFFER_FRAMES];
    let dropped = run_one(&mut strip, &quiet, 1, 0).await;
    assert!(dropped.data_frames().is_empty());

    let loud = voice(VAD_BUFFER_FRAMES);
    let kept = run_one(&mut strip, &loud, 1, buffer_ns).await;
    let frames = kept.data_frames();
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].timing.pts_ns, 0,
        "the surviving buffer moves back over the dropped one"
    );
}

#[tokio::test]
async fn removesilence_posts_the_silence_transitions_on_the_bus() {
    let (bus, handle) = Bus::new(8);
    let mut strip = RemoveSilence::new()
        .with_remove(true)
        .with_silent(false)
        .with_bus(handle);
    strip.configure_pipeline(&caps(1)).expect("mono configures");

    let buffer_ns = frames_to_ns(VAD_BUFFER_FRAMES as u64);
    let quiet = vec![0.0f32; VAD_BUFFER_FRAMES];
    run_one(&mut strip, &quiet, 1, 0).await;
    let loud = voice(VAD_BUFFER_FRAMES);
    run_one(&mut strip, &loud, 1, buffer_ns).await;

    let mut posted = Vec::new();
    while let Some(message) = bus.try_recv() {
        if let BusMessage::Info(text) = message {
            posted.push(text);
        }
    }
    assert_eq!(strip.silence_transitions(), posted.len() as u64);
    assert_eq!(posted.len(), 2, "silence detected, then finished");
    assert!(posted[0].contains("silence_detected"), "got {}", posted[0]);
    assert!(posted[1].contains("silence_finished"), "got {}", posted[1]);
    // the second message names the buffer that ended the silence.
    assert!(
        posted[1].contains(&buffer_ns.to_string()),
        "got {}",
        posted[1]
    );
}

#[tokio::test]
async fn removesilence_silent_keeps_the_bus_quiet() {
    let (bus, handle) = Bus::new(8);
    let mut strip = RemoveSilence::new().with_remove(true).with_bus(handle);
    strip.configure_pipeline(&caps(1)).expect("mono configures");
    run_one(&mut strip, &vec![0.0f32; VAD_BUFFER_FRAMES], 1, 0).await;
    assert!(bus.try_recv().is_none(), "silent is on by default");
    assert_eq!(strip.silence_transitions(), 0);
}

// ---------------------------------------------------------------------------
// audiobuffersplit
// ---------------------------------------------------------------------------

/// A 10 ms output buffer, which divides [`RATE`] exactly.
const SPLIT_DURATION: (i32, i32) = (1, 100);
/// Input buffers that do not line up with the output size at all.
const RAGGED_INPUT_FRAMES: usize = 700;
const RAGGED_INPUT_BUFFERS: usize = 5;

#[tokio::test]
async fn audiobuffersplit_cuts_equal_buffers_with_contiguous_timestamps() {
    let mut split =
        AudioBufferSplit::new().with_output_buffer_duration(SPLIT_DURATION.0, SPLIT_DURATION.1);
    split.configure_pipeline(&caps(1)).expect("mono configures");
    let frames_per_buffer = RATE as u64 * SPLIT_DURATION.0 as u64 / SPLIT_DURATION.1 as u64;
    assert_eq!(split.samples_per_buffer(), frames_per_buffer);

    let mut out = Collect::default();
    for index in 0..RAGGED_INPUT_BUFFERS {
        let start = (index * RAGGED_INPUT_FRAMES) as u64;
        let samples = vec![index as f32; RAGGED_INPUT_FRAMES];
        split
            .process(to_frame(&samples, 1, frames_to_ns(start)), &mut out)
            .await
            .expect("the split accepts the buffer");
    }

    let frames = out.data_frames();
    let total_frames = (RAGGED_INPUT_FRAMES * RAGGED_INPUT_BUFFERS) as u64;
    assert_eq!(
        frames.len() as u64,
        total_frames / frames_per_buffer,
        "every whole output buffer was cut"
    );
    for (index, frame) in frames.iter().enumerate() {
        let bytes = frame.domain.as_system_slice().expect("system frame");
        assert_eq!(
            bytes.len() as u64 / 4,
            frames_per_buffer,
            "buffer {index} holds one output duration"
        );
        let expected_pts = frames_to_ns(index as u64 * frames_per_buffer);
        assert_eq!(frame.timing.pts_ns, expected_pts, "buffer {index} pts");
        assert_eq!(
            frame.timing.duration_ns,
            frames_to_ns(frames_per_buffer),
            "buffer {index} duration"
        );
    }
}

#[tokio::test]
async fn audiobuffersplit_eos_flushes_the_partial_tail_unless_strict() {
    let frames_per_buffer = RATE as u64 * SPLIT_DURATION.0 as u64 / SPLIT_DURATION.1 as u64;
    let partial_frames = (frames_per_buffer / 2) as usize;

    for (strict, expected_frames) in [(false, 1), (true, 0)] {
        let mut split = AudioBufferSplit::new()
            .with_output_buffer_duration(SPLIT_DURATION.0, SPLIT_DURATION.1)
            .with_strict_buffer_size(strict);
        split.configure_pipeline(&caps(1)).expect("mono configures");

        let mut out = run_one(&mut split, &vec![1.0f32; partial_frames], 1, 0).await;
        assert!(out.data_frames().is_empty(), "not a whole buffer yet");
        split
            .process(PipelinePacket::Eos, &mut out)
            .await
            .expect("eos is accepted");
        assert_eq!(
            out.data_frames().len(),
            expected_frames,
            "strict-buffer-size={strict}"
        );
    }
}

#[tokio::test]
async fn audiobuffersplit_gapless_fills_a_gap_with_silence() {
    // The gap here is shorter than the default 40 ms alignment threshold and
    // than the default one-second discont wait, so both are turned off to make
    // the jump count on the buffer that makes it.
    let mut split = AudioBufferSplit::new()
        .with_output_buffer_duration(SPLIT_DURATION.0, SPLIT_DURATION.1)
        .with_gapless(true)
        .with_alignment_threshold(0)
        .with_discont_wait(0);
    split.configure_pipeline(&caps(1)).expect("mono configures");
    let frames_per_buffer = RATE as u64 * SPLIT_DURATION.0 as u64 / SPLIT_DURATION.1 as u64;

    // one whole output buffer, then a jump of the same length.
    let block = vec![1.0f32; frames_per_buffer as usize];
    let mut out = run_one(&mut split, &block, 1, 0).await;
    let gap_frames = frames_per_buffer * 2;
    let resumed = to_frame(&block, 1, frames_to_ns(gap_frames + frames_per_buffer));
    split
        .process(resumed, &mut out)
        .await
        .expect("the split accepts the buffer after the gap");

    let samples = out.samples();
    let expected_frames = frames_per_buffer + gap_frames + frames_per_buffer;
    assert_eq!(
        samples.len() as u64,
        expected_frames,
        "the gap was filled rather than skipped"
    );
    let gap_start = frames_per_buffer as usize;
    let gap_end = gap_start + gap_frames as usize;
    assert!(
        samples[gap_start..gap_end].iter().all(|s| *s == 0.0),
        "the fill is silence"
    );
}

// ---------------------------------------------------------------------------
// speed
// ---------------------------------------------------------------------------

const SPEED_FACTOR: f64 = 2.0;
const SPEED_INPUT_FRAMES: usize = 1000;

#[tokio::test]
async fn speed_two_halves_the_sample_count_and_the_duration() {
    let mut speed = Speed::new().with_speed(SPEED_FACTOR);
    speed.configure_pipeline(&caps(1)).expect("mono configures");

    let input: Vec<f32> = (0..SPEED_INPUT_FRAMES).map(|i| i as f32).collect();
    let out = run_one(&mut speed, &input, 1, 0).await;
    let resampled = out.samples();

    let expected_frames = SPEED_INPUT_FRAMES as f64 / SPEED_FACTOR;
    assert!(
        (resampled.len() as f64 - expected_frames).abs() <= 1.0,
        "got {} frames, expected about {expected_frames}",
        resampled.len()
    );

    let frame = out.data_frames()[0];
    assert_eq!(frame.timing.pts_ns, 0);
    assert_eq!(
        frame.timing.duration_ns,
        frames_to_ns(resampled.len() as u64),
        "the duration follows the sample count at the unchanged caps rate"
    );
    // the caps rate is untouched: only the sample count moved.
    assert_eq!(out.first_caps(), Some(&caps(1)));
}

#[tokio::test]
async fn speed_compresses_the_segment_it_forwards() {
    let mut speed = Speed::new().with_speed(SPEED_FACTOR);
    speed.configure_pipeline(&caps(1)).expect("mono configures");

    let start_ns = NS_PER_SECOND;
    let stop_ns = NS_PER_SECOND * 3;
    let segment = Segment {
        start: start_ns,
        stop: Some(stop_ns),
        position: start_ns,
        ..Segment::new()
    };
    let mut out = Collect::default();
    speed
        .process(PipelinePacket::Segment(segment), &mut out)
        .await
        .expect("the segment is accepted");

    let forwarded: Vec<Segment> = out.segments().into_iter().copied().collect();
    assert_eq!(forwarded.len(), 1);
    assert_eq!(forwarded[0].start, (start_ns as f64 / SPEED_FACTOR) as u64);
    assert_eq!(
        forwarded[0].stop,
        Some((stop_ns as f64 / SPEED_FACTOR) as u64)
    );

    // the output timeline starts where the compressed segment does.
    let input: Vec<f32> = (0..SPEED_INPUT_FRAMES).map(|i| i as f32).collect();
    speed
        .process(to_frame(&input, 1, start_ns), &mut out)
        .await
        .expect("the buffer is accepted");
    let frame = out.data_frames()[0];
    assert_eq!(frame.timing.pts_ns, forwarded[0].start);
}

// ---------------------------------------------------------------------------
// launch registration
// ---------------------------------------------------------------------------

#[test]
fn every_m1085_element_builds_from_a_launch_line() {
    let registry = default_registry();
    for name in [
        "audiochannelmix",
        "audiomixmatrix",
        "stereo",
        "audiofirfilter",
        "audioiirfilter",
        "removesilence",
        "audiobuffersplit",
        "speed",
    ] {
        let line = format!("fakesrc ! {name} ! fakesink");
        parse_launch(&registry, &line).unwrap_or_else(|e| panic!("{name} parses: {e}"));
    }
}

#[test]
fn launch_sets_the_array_and_fraction_properties() {
    let registry = default_registry();
    let line = format!(
        "fakesrc ! audiofirfilter kernel={} latency=1 ! \
         audioiirfilter a=1,-0.5 b=0.5 ! \
         audiomixmatrix in-channels=2 out-channels=1 matrix=0.5,0.5 ! \
         audiobuffersplit output-buffer-duration=1/100 ! fakesink",
        kernel_text(&TEST_KERNEL)
    );
    parse_launch(&registry, &line).expect("the array and fraction properties parse");
}

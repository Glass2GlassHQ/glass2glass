//! M1218: `webrtcdsp` and `webrtcechoprobe` driven through `process` with the
//! real processor. Every signal is generated here from a seeded generator, and
//! every threshold is a named constant.
#![cfg(feature = "webrtcdsp")]

use core::f64::consts::TAU;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, G2gError, MemoryDomain, OutputSink, PropError, PropValue,
    PropertySpec, PushOutcome,
};
use g2g_plugins::gst_compat::{gst_equivalent, GstEquivalent};
use g2g_plugins::registry::default_registry;
use g2g_plugins::webrtcdsp::{WebrtcDsp, WebrtcEchoProbe};

const RATE: u32 = 16_000;
const NS_PER_SECOND: u64 = 1_000_000_000;
const SEED: u64 = 0x1218_5eed_0bad_cafe;
const PERIOD_FRAMES: usize = RATE as usize / 100;
const CHUNK_FRAMES: usize = 2 * PERIOD_FRAMES;
const SIGNAL_SECONDS: usize = 10;
const CONVERGENCE_SECONDS: usize = 4;
const ECHO_DELAY_MS: usize = 40;
const ECHO_GAIN: f32 = 0.5;
const FAR_END_AMPLITUDE: f32 = 0.4;
// a slow envelope makes the noise burst like speech
const ENVELOPE_HZ: f64 = 3.0;
const TONE_HZ: f64 = 500.0;
const TONE_AMPLITUDE: f64 = 0.1;
const NOISE_AMPLITUDE: f32 = 0.05;
const NOISE_TONE_AMPLITUDE: f64 = 0.3;

const MIN_ECHO_REDUCTION_DB: f64 = 30.0;
const MIN_DOUBLE_TALK_ECHO_REDUCTION_DB: f64 = 6.0;
const TONE_TOLERANCE_DB: f64 = 3.0;
const MIN_NOISE_REDUCTION_DB: f64 = 6.0;
const BYPASS_TOLERANCE_DB: f64 = 0.5;

const ODD_CHUNK_FRAMES: usize = 113;
const ODD_TOTAL_FRAMES: usize = RATE as usize + 37;
const FIRST_PTS_NS: u64 = 5 * NS_PER_SECOND;

const MISMATCHED_RATE: u32 = 48_000;
// outside the GStreamer rate set the elements take
const UNSUPPORTED_RATE: u32 = 44_100;
const I16_SCALE: f32 = 32768.0;

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
    fn data_frames(&self) -> impl Iterator<Item = &Frame> {
        self.packets.iter().filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
    }

    fn samples(&self) -> Vec<f32> {
        let mut samples = Vec::new();
        for frame in self.data_frames() {
            let bytes = frame.domain.as_system_slice().expect("system frame");
            for chunk in bytes.as_chunks::<2>().0 {
                samples.push(i16::from_le_bytes(*chunk) as f32 / I16_SCALE);
            }
        }
        samples
    }
}

fn caps(rate: u32) -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 1,
        sample_rate: rate,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

fn frames_to_ns(frames: usize) -> u64 {
    frames as u64 * NS_PER_SECOND / RATE as u64
}

fn frame(samples: &[f32], pts_ns: u64) -> PipelinePacket {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        let quantized = (sample.clamp(-1.0, 1.0) * (I16_SCALE - 1.0)).round() as i16;
        bytes.extend_from_slice(&quantized.to_le_bytes());
    }
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: frames_to_ns(samples.len()),
            ..Default::default()
        },
        0,
    ))
}

struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn speech_like_far_end(frames: usize) -> Vec<f32> {
    let mut noise = Noise(SEED);
    (0..frames)
        .map(|index| {
            let time = index as f64 / RATE as f64;
            let envelope = 0.5 + 0.5 * (TAU * ENVELOPE_HZ * time).sin();
            FAR_END_AMPLITUDE * envelope as f32 * noise.next()
        })
        .collect()
}

fn echo_of(far_end: &[f32]) -> Vec<f32> {
    let delay = RATE as usize * ECHO_DELAY_MS / 1000;
    (0..far_end.len())
        .map(|index| match index.checked_sub(delay) {
            Some(source) => ECHO_GAIN * far_end[source],
            None => 0.0,
        })
        .collect()
}

fn tone(frames: usize, frequency: f64, amplitude: f64) -> Vec<f32> {
    (0..frames)
        .map(|index| (amplitude * (TAU * frequency * index as f64 / RATE as f64).sin()) as f32)
        .collect()
}

fn mix(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(a, b)| a + b).collect()
}

fn power_db(samples: &[f32]) -> f64 {
    let energy: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    10.0 * (energy / samples.len() as f64).log10()
}

// the window's component at `frequency`, any phase, and what is left without it
fn split_tone(samples: &[f32], frequency: f64) -> (f64, Vec<f32>) {
    let count = samples.len() as f64;
    let angle = |index: usize| TAU * frequency * index as f64 / RATE as f64;
    let (mut sine, mut cosine) = (0.0, 0.0);
    for (index, sample) in samples.iter().enumerate() {
        sine += *sample as f64 * angle(index).sin();
        cosine += *sample as f64 * angle(index).cos();
    }
    let (sine, cosine) = (2.0 * sine / count, 2.0 * cosine / count);
    let residual = samples
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let fitted = sine * angle(index).sin() + cosine * angle(index).cos();
            (*sample as f64 - fitted) as f32
        })
        .collect();
    (sine.hypot(cosine), residual)
}

fn measured(samples: &[f32]) -> &[f32] {
    &samples[CONVERGENCE_SECONDS * RATE as usize..]
}

fn dsp(probe_name: &str) -> WebrtcDsp {
    WebrtcDsp::new()
        .with_probe(probe_name)
        .with_noise_suppression(false)
        .with_gain_control(false)
        .with_high_pass_filter(false)
}

// far end through the probe and near end through the dsp, chunk by chunk with
// matching timestamps, as a live playback and capture pair would interleave
async fn run_pair(
    probe: &mut WebrtcEchoProbe,
    dsp: &mut WebrtcDsp,
    far_end: &[f32],
    near_end: &[f32],
) -> Vec<f32> {
    let mut played = Collect::default();
    let mut processed = Collect::default();
    for (index, (far, near)) in far_end
        .chunks(CHUNK_FRAMES)
        .zip(near_end.chunks(CHUNK_FRAMES))
        .enumerate()
    {
        let pts_ns = frames_to_ns(index * CHUNK_FRAMES);
        probe
            .process(frame(far, pts_ns), &mut played)
            .await
            .expect("the probe takes the far end");
        dsp.process(frame(near, pts_ns), &mut processed)
            .await
            .expect("the dsp takes the near end");
    }
    dsp.process(PipelinePacket::Eos, &mut processed)
        .await
        .expect("the dsp drains at Eos");
    assert_eq!(
        played.samples().len(),
        far_end.len(),
        "the probe forwards the far end"
    );
    processed.samples()
}

async fn cancel_echo(probe_name: &str, echo_cancel: bool, near_end: &[f32]) -> Vec<f32> {
    let far_end = speech_like_far_end(SIGNAL_SECONDS * RATE as usize);
    let mut probe = WebrtcEchoProbe::new().with_probe_name(probe_name);
    probe
        .configure_pipeline(&caps(RATE))
        .expect("probe configures");
    let mut dsp = dsp(probe_name).with_echo_cancel(echo_cancel);
    dsp.configure_pipeline(&caps(RATE)).expect("dsp configures");
    run_pair(&mut probe, &mut dsp, &far_end, near_end).await
}

#[tokio::test]
async fn echo_cancellation_removes_the_echo_and_bypass_keeps_it() {
    let echo = echo_of(&speech_like_far_end(SIGNAL_SECONDS * RATE as usize));
    let echo_db = power_db(measured(&echo));

    let cancelled = cancel_echo("m1218-echo", true, &echo).await;
    let reduction_db = echo_db - power_db(measured(&cancelled));
    println!("echo reduction: {reduction_db:.1} dB");
    assert!(
        reduction_db >= MIN_ECHO_REDUCTION_DB,
        "echo reduced by {reduction_db:.1} dB"
    );

    let bypassed = cancel_echo("m1218-echo-bypass", false, &echo).await;
    let change_db = power_db(measured(&bypassed)) - echo_db;
    println!("echo-cancel=false level change: {change_db:.2} dB");
    assert!(
        change_db.abs() <= BYPASS_TOLERANCE_DB,
        "level changed by {change_db:.2} dB"
    );
}

#[tokio::test]
async fn double_talk_keeps_the_near_end_tone() {
    let frames = SIGNAL_SECONDS * RATE as usize;
    let echo = echo_of(&speech_like_far_end(frames));
    let talker = tone(frames, TONE_HZ, TONE_AMPLITUDE);
    let near_end = mix(&echo, &talker);

    let output = cancel_echo("m1218-double-talk", true, &near_end).await;
    let (tone_amplitude, residual) = split_tone(measured(&output), TONE_HZ);
    let tone_change_db = 20.0 * (tone_amplitude / TONE_AMPLITUDE).log10();
    let echo_reduction_db = power_db(measured(&echo)) - power_db(&residual);
    println!("double talk: tone {tone_change_db:.2} dB, echo reduction {echo_reduction_db:.1} dB");
    assert!(
        tone_change_db.abs() <= TONE_TOLERANCE_DB,
        "tone changed by {tone_change_db:.2} dB"
    );
    assert!(
        echo_reduction_db >= MIN_DOUBLE_TALK_ECHO_REDUCTION_DB,
        "echo reduced by {echo_reduction_db:.1} dB"
    );
}

#[tokio::test]
async fn noise_suppression_switched_on_at_runtime_lowers_the_noise() {
    let half = SIGNAL_SECONDS * RATE as usize / 2;
    let mut noise = Noise(SEED);
    let hiss: Vec<f32> = (0..2 * half)
        .map(|_| NOISE_AMPLITUDE * noise.next())
        .collect();
    let input = mix(&hiss, &tone(2 * half, TONE_HZ, NOISE_TONE_AMPLITUDE));

    let mut dsp = WebrtcDsp::new()
        .with_echo_cancel(false)
        .with_noise_suppression(false)
        .with_gain_control(false)
        .with_high_pass_filter(false);
    dsp.configure_pipeline(&caps(RATE)).expect("dsp configures");
    let mut out = Collect::default();
    for (index, chunk) in input.chunks(CHUNK_FRAMES).enumerate() {
        if index * CHUNK_FRAMES == half {
            dsp.set_property("noise-suppression", PropValue::Bool(true))
                .expect("noise-suppression is settable");
        }
        dsp.process(frame(chunk, frames_to_ns(index * CHUNK_FRAMES)), &mut out)
            .await
            .expect("the dsp takes the stream");
    }
    let output = out.samples();
    let settle = CONVERGENCE_SECONDS * RATE as usize / 2;
    let noise_db = |range: core::ops::Range<usize>| {
        let (_, residual) = split_tone(&output[range], TONE_HZ);
        power_db(&residual)
    };
    let input_noise_db = power_db(&hiss);
    let off_change_db = noise_db(settle..half) - input_noise_db;
    let on_reduction_db = input_noise_db - noise_db(half + settle..2 * half);
    println!(
        "noise-suppression=false change {off_change_db:.2} dB, =true reduction {on_reduction_db:.1} dB"
    );
    assert!(
        off_change_db.abs() <= BYPASS_TOLERANCE_DB,
        "noise changed by {off_change_db:.2} dB with suppression off"
    );
    assert!(
        on_reduction_db >= MIN_NOISE_REDUCTION_DB,
        "noise reduced by {on_reduction_db:.1} dB"
    );
}

#[tokio::test]
async fn odd_frames_come_out_complete_and_in_order() {
    let input = speech_like_far_end(ODD_TOTAL_FRAMES);
    let mut dsp = WebrtcDsp::new().with_echo_cancel(false);
    dsp.configure_pipeline(&caps(RATE)).expect("dsp configures");
    let mut out = Collect::default();
    for (index, chunk) in input.chunks(ODD_CHUNK_FRAMES).enumerate() {
        let pts_ns = FIRST_PTS_NS + frames_to_ns(index * ODD_CHUNK_FRAMES);
        dsp.process(frame(chunk, pts_ns), &mut out)
            .await
            .expect("the dsp takes an odd-sized frame");
    }
    dsp.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("the dsp drains at Eos");

    assert_eq!(out.samples().len(), input.len());
    let mut emitted_frames = 0;
    for frame in out.data_frames() {
        let expected_ns = FIRST_PTS_NS + frames_to_ns(emitted_frames);
        assert!(
            frame.timing.pts_ns.abs_diff(expected_ns) <= 1,
            "pts {} where {expected_ns} was due",
            frame.timing.pts_ns
        );
        emitted_frames += frame.domain.as_system_slice().expect("system").len() / 2;
    }
    assert_eq!(dsp.latency().min_ns, frames_to_ns(PERIOD_FRAMES));
}

#[test]
fn echo_cancel_without_a_probe_fails_to_configure() {
    let mut dsp = WebrtcDsp::new().with_probe("m1218-nobody");
    assert_eq!(
        dsp.configure_pipeline(&caps(RATE)).unwrap_err(),
        G2gError::NotConfigured
    );
}

#[test]
fn probe_and_dsp_at_different_rates_fail_to_configure() {
    let name = "m1218-rate";
    let mut probe = WebrtcEchoProbe::new().with_probe_name(name);
    probe
        .configure_pipeline(&caps(MISMATCHED_RATE))
        .expect("probe configures");
    let mut dsp = WebrtcDsp::new().with_probe(name);
    assert_eq!(
        dsp.configure_pipeline(&caps(RATE)).unwrap_err(),
        G2gError::CapsMismatch
    );
}

#[test]
fn unsupported_rate_is_refused() {
    let mut dsp = WebrtcDsp::new().with_echo_cancel(false);
    assert_eq!(
        dsp.configure_pipeline(&caps(UNSUPPORTED_RATE)).unwrap_err(),
        G2gError::CapsMismatch
    );
}

fn declared_default(specs: &[PropertySpec], name: &str) -> PropValue {
    let spec = specs
        .iter()
        .find(|spec| spec.name == name)
        .unwrap_or_else(|| panic!("`{name}` is declared"));
    spec.parse_value(spec.default.expect("a declared default"))
        .unwrap_or_else(|_| panic!("`{name}`'s default parses"))
}

#[test]
fn properties_round_trip_with_their_declared_defaults() {
    let changed = [
        ("probe", PropValue::Str("m1218-speaker".into())),
        ("echo-cancel", PropValue::Bool(false)),
        ("noise-suppression", PropValue::Bool(false)),
        (
            "noise-suppression-level",
            PropValue::Str("very-high".into()),
        ),
        ("gain-control", PropValue::Bool(false)),
        ("gain-control-mode", PropValue::Str("fixed-digital".into())),
        ("target-level-dbfs", PropValue::Int(10)),
        ("compression-gain-db", PropValue::Int(20)),
        ("high-pass-filter", PropValue::Bool(false)),
    ];
    let mut dsp = WebrtcDsp::new();
    let specs = dsp.properties();
    assert_eq!(specs.len(), changed.len());
    for (name, value) in changed {
        assert_eq!(
            dsp.get_property(name),
            Some(declared_default(specs, name)),
            "{name}"
        );
        dsp.set_property(name, value.clone()).expect(name);
        assert_eq!(dsp.get_property(name), Some(value), "{name}");
    }
    assert_eq!(
        dsp.set_property("noise-suppression-level", PropValue::Str("loud".into())),
        Err(PropError::Value)
    );
    let (_, maximum) = specs
        .iter()
        .find(|spec| spec.name == "compression-gain-db")
        .and_then(|spec| spec.range)
        .expect("compression-gain-db declares a range");
    let beyond: i64 = maximum.parse::<i64>().expect("numeric maximum") + 1;
    assert_eq!(
        dsp.set_property("compression-gain-db", PropValue::Int(beyond)),
        Err(PropError::Value)
    );

    let mut probe = WebrtcEchoProbe::new();
    assert_eq!(
        probe.get_property("probe-name"),
        Some(declared_default(probe.properties(), "probe-name"))
    );
    probe
        .set_property("probe-name", PropValue::Str("m1218-speaker".into()))
        .expect("probe-name is settable");
    assert_eq!(
        probe.get_property("probe-name"),
        Some(PropValue::Str("m1218-speaker".into()))
    );
}

#[test]
fn both_gstreamer_names_resolve() {
    let registry = default_registry();
    for name in ["webrtcdsp", "webrtcechoprobe"] {
        assert_eq!(
            gst_equivalent(&registry, name),
            GstEquivalent::Available,
            "{name}"
        );
    }
}

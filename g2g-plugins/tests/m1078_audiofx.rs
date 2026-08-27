//! M1078 `audiofx` filters: `audiodynamic`, `audioinvert`, `audiokaraoke`, the
//! windowed-sinc pair (`audiowsinclimit` / `audiowsincband`) and the Chebyshev
//! pair (`audiocheblimit` / `audiochebband`). Every assertion is a measurement
//! of what the filter did to a signal (the gain it applied at a frequency, the
//! peak it left, the cancellation it achieved), not a hard-coded output buffer.
#![cfg(feature = "std")]

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, G2gError, MemoryDomain, OutputSink, PipelineClock, PropValue,
    PushOutcome,
};
use g2g_plugins::audiochebband::AudioChebBand;
use g2g_plugins::audiocheblimit::AudioChebLimit;
use g2g_plugins::audiodynamic::{AudioDynamic, DynamicCharacteristics, DynamicMode};
use g2g_plugins::audiofx::{BandMode, FirWindow, LimitMode};
use g2g_plugins::audioinvert::AudioInvert;
use g2g_plugins::audiokaraoke::AudioKaraoke;
use g2g_plugins::audiowsincband::AudioWsincBand;
use g2g_plugins::audiowsinclimit::AudioWsincLimit;
use g2g_plugins::registry::default_registry;

const RATE: u32 = 48_000;
const NS_PER_SECOND: u64 = 1_000_000_000;
/// One test buffer: 100 ms, long enough for an IIR to settle inside it.
const BUFFER_SAMPLES: usize = 4_800;
const BUFFERS: usize = 4;
/// Test tone amplitude: half scale, so a filter that overshoots still fits.
const AMPLITUDE: f64 = 0.5;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

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
        for packet in &self.packets {
            if let PipelinePacket::DataFrame(frame) = packet {
                let bytes = frame.domain.as_system_slice().expect("system frame");
                for chunk in bytes.as_chunks::<4>().0 {
                    out.push(f32::from_le_bytes(*chunk));
                }
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

/// `frames` sample frames of a sine at `frequency`, the same tone on every
/// channel, starting at sample index `start`.
fn sine(frequency: f64, frames: usize, channels: usize, start: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames * channels);
    for i in 0..frames {
        let t = (start + i) as f64 / RATE as f64;
        let value = (AMPLITUDE * (core::f64::consts::TAU * frequency * t).sin()) as f32;
        for _ in 0..channels {
            out.push(value);
        }
    }
    out
}

fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    (sum / samples.len() as f64).sqrt()
}

fn db(ratio: f64) -> f64 {
    20.0 * ratio.max(f64::MIN_POSITIVE).log10()
}

/// Drive `element` with `BUFFERS` buffers of a sine, then `Eos`, and return the
/// gain it applied, measured over the second half of the output so a filter's
/// start-up transient is not counted.
async fn measure_gain<E: AsyncElement>(element: &mut E, frequency: f64, channels: usize) -> f64 {
    element
        .configure_pipeline(&caps(channels as u8))
        .expect("f32 caps configure");
    let mut out = Collect::default();
    for buffer in 0..BUFFERS {
        let start = buffer * BUFFER_SAMPLES;
        let samples = sine(frequency, BUFFER_SAMPLES, channels, start);
        let pts = (start as u64) * NS_PER_SECOND / RATE as u64;
        element
            .process(to_frame(&samples, channels, pts), &mut out)
            .await
            .expect("filter accepts the buffer");
    }
    element
        .process(PipelinePacket::Eos, &mut out)
        .await
        .expect("filter accepts eos");

    let filtered = out.samples();
    let reference = sine(frequency, BUFFERS * BUFFER_SAMPLES, channels, 0);
    let half = filtered.len() / 2;
    rms(&filtered[half..]) / rms(&reference)
}

// ---------------------------------------------------------------------------
// audiowsinclimit / audiowsincband
// ---------------------------------------------------------------------------

/// A Hamming window's peak sidelobe is 41 dB below its main lobe, so a
/// windowed-sinc kernel rejects a tone well inside its stop band by at least
/// that much.
const HAMMING_STOPBAND_DB: f64 = 41.0;
/// How far the pass band may deviate from unity gain.
const PASSBAND_TOLERANCE_DB: f64 = 1.0;
/// Kernel length used by the frequency tests: the reference's default, whose
/// Hamming transition width (about 3.3 / length in normalized frequency) is far
/// narrower than the octaves between the test tones and the cutoffs.
const TEST_KERNEL_TAPS: i64 = 101;
const CUTOFF_HZ: f64 = 1000.0;
const PASSBAND_TONE_HZ: f64 = 200.0;
const STOPBAND_TONE_HZ: f64 = 5000.0;
const BAND_LOWER_HZ: f64 = 1000.0;
const BAND_UPPER_HZ: f64 = 4000.0;
const IN_BAND_TONE_HZ: f64 = 2500.0;
const OUT_OF_BAND_TONE_HZ: f64 = 15000.0;

#[tokio::test]
async fn wsinclimit_low_pass_keeps_the_low_tone_and_rejects_the_high_one() {
    let mut low = AudioWsincLimit::new()
        .with_mode(LimitMode::LowPass)
        .with_cutoff(CUTOFF_HZ)
        .with_length(TEST_KERNEL_TAPS)
        .with_window(FirWindow::Hamming);
    let passband = db(measure_gain(&mut low, PASSBAND_TONE_HZ, 1).await);
    assert!(
        passband.abs() < PASSBAND_TOLERANCE_DB,
        "200 Hz passes at {passband:.2} dB"
    );

    let mut low = AudioWsincLimit::new()
        .with_mode(LimitMode::LowPass)
        .with_cutoff(CUTOFF_HZ)
        .with_length(TEST_KERNEL_TAPS)
        .with_window(FirWindow::Hamming);
    let stopband = db(measure_gain(&mut low, STOPBAND_TONE_HZ, 1).await);
    assert!(
        stopband <= -HAMMING_STOPBAND_DB,
        "5 kHz is only {stopband:.2} dB down"
    );
}

#[tokio::test]
async fn wsinclimit_high_pass_is_the_mirror() {
    let mut high = AudioWsincLimit::new()
        .with_mode(LimitMode::HighPass)
        .with_cutoff(CUTOFF_HZ)
        .with_length(TEST_KERNEL_TAPS)
        .with_window(FirWindow::Hamming);
    let stopband = db(measure_gain(&mut high, PASSBAND_TONE_HZ, 1).await);
    assert!(
        stopband <= -HAMMING_STOPBAND_DB,
        "200 Hz is only {stopband:.2} dB down"
    );

    let mut high = AudioWsincLimit::new()
        .with_mode(LimitMode::HighPass)
        .with_cutoff(CUTOFF_HZ)
        .with_length(TEST_KERNEL_TAPS)
        .with_window(FirWindow::Hamming);
    let passband = db(measure_gain(&mut high, STOPBAND_TONE_HZ, 1).await);
    assert!(
        passband.abs() < PASSBAND_TOLERANCE_DB,
        "5 kHz passes at {passband:.2} dB"
    );
}

fn wsincband(mode: BandMode) -> AudioWsincBand {
    AudioWsincBand::new()
        .with_mode(mode)
        .with_lower_frequency(BAND_LOWER_HZ)
        .with_upper_frequency(BAND_UPPER_HZ)
        .with_length(TEST_KERNEL_TAPS)
        .with_window(FirWindow::Hamming)
}

#[tokio::test]
async fn wsincband_band_pass_keeps_the_band() {
    let inside = db(measure_gain(&mut wsincband(BandMode::BandPass), IN_BAND_TONE_HZ, 1).await);
    assert!(
        inside.abs() < PASSBAND_TOLERANCE_DB,
        "2.5 kHz passes at {inside:.2} dB"
    );
    let outside =
        db(measure_gain(&mut wsincband(BandMode::BandPass), OUT_OF_BAND_TONE_HZ, 1).await);
    assert!(
        outside <= -HAMMING_STOPBAND_DB,
        "15 kHz is only {outside:.2} dB down"
    );
}

#[tokio::test]
async fn wsincband_band_reject_is_the_complement() {
    let inside = db(measure_gain(&mut wsincband(BandMode::BandReject), IN_BAND_TONE_HZ, 1).await);
    assert!(
        inside <= -HAMMING_STOPBAND_DB,
        "2.5 kHz is only {inside:.2} dB down"
    );
    let outside =
        db(measure_gain(&mut wsincband(BandMode::BandReject), OUT_OF_BAND_TONE_HZ, 1).await);
    assert!(
        outside.abs() < PASSBAND_TOLERANCE_DB,
        "15 kHz passes at {outside:.2} dB"
    );
}

#[tokio::test]
async fn wsinclimit_keeps_the_sample_count_and_the_first_timestamp() {
    let mut low = AudioWsincLimit::new()
        .with_mode(LimitMode::LowPass)
        .with_cutoff(CUTOFF_HZ)
        .with_length(TEST_KERNEL_TAPS);
    low.configure_pipeline(&caps(1)).expect("configures");
    let latency = low.latency_samples();
    assert_eq!(latency, (TEST_KERNEL_TAPS as usize - 1) / 2);

    let mut out = Collect::default();
    for buffer in 0..BUFFERS {
        let start = buffer * BUFFER_SAMPLES;
        let samples = sine(PASSBAND_TONE_HZ, BUFFER_SAMPLES, 1, start);
        let pts = (start as u64) * NS_PER_SECOND / RATE as u64;
        low.process(to_frame(&samples, 1, pts), &mut out)
            .await
            .unwrap();
    }
    // before Eos the group delay is still owed.
    assert_eq!(out.samples().len(), BUFFERS * BUFFER_SAMPLES - latency);
    low.process(PipelinePacket::Eos, &mut out).await.unwrap();
    assert_eq!(
        out.samples().len(),
        BUFFERS * BUFFER_SAMPLES,
        "the tail pushed at Eos restores the input's sample count"
    );
    let frames = out.data_frames();
    assert_eq!(frames[0].timing.pts_ns, 0, "the group delay is taken off");
    let last = frames.last().expect("frames were pushed");
    let expected_last_pts =
        ((BUFFERS * BUFFER_SAMPLES - latency) as u64) * NS_PER_SECOND / RATE as u64;
    assert_eq!(last.timing.pts_ns, expected_last_pts);
}

// ---------------------------------------------------------------------------
// audiocheblimit / audiochebband
// ---------------------------------------------------------------------------

const CHEB_POLES: i64 = 4;
const CHEB_RIPPLE_DB: f64 = 0.5;
/// Chebyshev type 1 attenuates by `10*log10(1 + eps^2 * T_n(r)^2)` at frequency
/// ratio `r`, with `eps^2 = 10^(ripple/10) - 1` and `T_n(r) = cosh(n *
/// acosh(r))`. The bilinear transform warps the axis, so `r` is
/// `tan(pi*f/rate) / tan(pi*fc/rate)`, and normalizing for unity gain at DC
/// lifts an even-order response by the full ripple, which comes back off here.
fn chebyshev_type1_stopband_db(stop_hz: f64, cutoff_hz: f64, poles: f64, ripple_db: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let r = (pi * stop_hz / RATE as f64).tan() / (pi * cutoff_hz / RATE as f64).tan();
    let epsilon_squared = 10f64.powf(ripple_db / 10.0) - 1.0;
    let chebyshev = (poles * r.acosh()).cosh();
    10.0 * (1.0 + epsilon_squared * chebyshev * chebyshev).log10() - ripple_db
}

/// Slack between the ideal response and a measurement taken from a finite
/// buffer of a real signal.
const CHEBYSHEV_MEASUREMENT_MARGIN_DB: f64 = 2.0;

#[tokio::test]
async fn cheblimit_low_pass_reaches_its_predicted_stop_band() {
    let mut low = AudioChebLimit::new()
        .with_mode(LimitMode::LowPass)
        .with_cutoff(CUTOFF_HZ)
        .with_poles(CHEB_POLES)
        .with_ripple(CHEB_RIPPLE_DB)
        .with_type(1);
    let passband = db(measure_gain(&mut low, PASSBAND_TONE_HZ, 1).await);
    assert!(
        passband.abs() < CHEB_RIPPLE_DB + PASSBAND_TOLERANCE_DB,
        "200 Hz passes at {passband:.2} dB"
    );

    let mut low = AudioChebLimit::new()
        .with_mode(LimitMode::LowPass)
        .with_cutoff(CUTOFF_HZ)
        .with_poles(CHEB_POLES)
        .with_ripple(CHEB_RIPPLE_DB)
        .with_type(1);
    let stopband = db(measure_gain(&mut low, STOPBAND_TONE_HZ, 1).await);
    let expected = chebyshev_type1_stopband_db(
        STOPBAND_TONE_HZ,
        CUTOFF_HZ,
        CHEB_POLES as f64,
        CHEB_RIPPLE_DB,
    );
    assert!(
        stopband <= -(expected - CHEBYSHEV_MEASUREMENT_MARGIN_DB),
        "5 kHz is {stopband:.2} dB down, the order predicts {expected:.2} dB"
    );
}

#[tokio::test]
async fn chebband_band_pass_keeps_the_band() {
    let mut band = AudioChebBand::new()
        .with_mode(BandMode::BandPass)
        .with_lower_frequency(BAND_LOWER_HZ)
        .with_upper_frequency(BAND_UPPER_HZ)
        .with_poles(CHEB_POLES)
        .with_ripple(CHEB_RIPPLE_DB);
    let inside = db(measure_gain(&mut band, 2000.0, 1).await);
    assert!(
        inside.abs() < CHEB_RIPPLE_DB + PASSBAND_TOLERANCE_DB,
        "the band centre passes at {inside:.2} dB"
    );

    let mut band = AudioChebBand::new()
        .with_mode(BandMode::BandPass)
        .with_lower_frequency(BAND_LOWER_HZ)
        .with_upper_frequency(BAND_UPPER_HZ)
        .with_poles(CHEB_POLES)
        .with_ripple(CHEB_RIPPLE_DB);
    let outside = db(measure_gain(&mut band, OUT_OF_BAND_TONE_HZ, 1).await);
    // the element's own response, cross-checked against GStreamer's in the
    // module tests, is what a measurement has to reproduce.
    let predicted = db(band.response_at(OUT_OF_BAND_TONE_HZ));
    assert!(
        (outside - predicted).abs() < CHEBYSHEV_MEASUREMENT_MARGIN_DB,
        "15 kHz measured {outside:.2} dB, the cascade predicts {predicted:.2} dB"
    );
}

// ---------------------------------------------------------------------------
// audiodynamic
// ---------------------------------------------------------------------------

const COMPRESSOR_THRESHOLD: f64 = 0.5;
const COMPRESSOR_RATIO: f64 = 0.25;

#[tokio::test]
async fn compressor_pulls_a_full_scale_peak_onto_the_hard_knee() {
    let mut element = AudioDynamic::new()
        .with_mode(DynamicMode::Compressor)
        .with_characteristics(DynamicCharacteristics::HardKnee)
        .with_threshold(COMPRESSOR_THRESHOLD)
        .with_ratio(COMPRESSOR_RATIO);
    element.configure_pipeline(&caps(1)).expect("configures");

    // a full-scale sine, so the peak actually reaches 1.0.
    let mut samples = Vec::new();
    for i in 0..BUFFER_SAMPLES {
        let t = i as f64 / RATE as f64;
        samples.push((core::f64::consts::TAU * 1000.0 * t).sin() as f32);
    }
    let mut out = Collect::default();
    element
        .process(to_frame(&samples, 1, 0), &mut out)
        .await
        .unwrap();

    let peak = out.samples().iter().fold(0.0f32, |acc, s| acc.max(s.abs())) as f64;
    let input_peak = samples.iter().fold(0.0f32, |acc, s| acc.max(s.abs())) as f64;
    // gstaudiodynamic.c, hard-knee compressor: past the threshold the curve is
    // t + (x - t) * ratio.
    let expected = COMPRESSOR_THRESHOLD + (input_peak - COMPRESSOR_THRESHOLD) * COMPRESSOR_RATIO;
    assert!(
        (peak - expected).abs() < 1e-3,
        "peak {peak} should be the hard-knee value {expected}"
    );
}

const EXPANDER_THRESHOLD: f64 = 0.5;
const EXPANDER_RATIO: f64 = 2.0;

#[tokio::test]
async fn expander_squelches_a_tone_under_its_zero_crossing() {
    let mut element = AudioDynamic::new()
        .with_mode(DynamicMode::Expander)
        .with_characteristics(DynamicCharacteristics::HardKnee)
        .with_threshold(EXPANDER_THRESHOLD)
        .with_ratio(EXPANDER_RATIO);
    element.configure_pipeline(&caps(1)).expect("configures");

    // gstaudiodynamic.c, hard-knee expander: everything under the zero crossing
    // t - t/ratio is squelched, so a tone that never reaches it goes silent.
    let zero_crossing = EXPANDER_THRESHOLD - EXPANDER_THRESHOLD / EXPANDER_RATIO;
    let quiet_amplitude = zero_crossing * 0.9;
    let mut samples = Vec::new();
    for i in 0..BUFFER_SAMPLES {
        let t = i as f64 / RATE as f64;
        samples.push((quiet_amplitude * (core::f64::consts::TAU * 1000.0 * t).sin()) as f32);
    }
    let mut out = Collect::default();
    element
        .process(to_frame(&samples, 1, 0), &mut out)
        .await
        .unwrap();
    assert!(
        out.samples().iter().all(|s| *s == 0.0),
        "a tone under the zero crossing is squelched"
    );

    // and a tone above the threshold passes untouched.
    let mut element = AudioDynamic::new()
        .with_mode(DynamicMode::Expander)
        .with_characteristics(DynamicCharacteristics::HardKnee)
        .with_threshold(EXPANDER_THRESHOLD)
        .with_ratio(EXPANDER_RATIO);
    element.configure_pipeline(&caps(1)).expect("configures");
    let loud: Vec<f32> = samples
        .iter()
        .map(|s| (*s as f64 / quiet_amplitude) as f32)
        .collect();
    let mut out = Collect::default();
    element
        .process(to_frame(&loud, 1, 0), &mut out)
        .await
        .unwrap();
    let peak = out.samples().iter().fold(0.0f32, |acc, s| acc.max(s.abs())) as f64;
    assert!((peak - 1.0).abs() < 1e-3, "a loud peak passes, got {peak}");
}

// ---------------------------------------------------------------------------
// audioinvert
// ---------------------------------------------------------------------------

#[tokio::test]
async fn invert_negates_at_full_degree_and_silences_at_half() {
    let samples = sine(1000.0, 512, 1, 0);

    let mut full = AudioInvert::new().with_degree(1.0);
    full.configure_pipeline(&caps(1)).expect("configures");
    let mut out = Collect::default();
    full.process(to_frame(&samples, 1, 0), &mut out)
        .await
        .unwrap();
    for (input, output) in samples.iter().zip(out.samples()) {
        assert!(
            (output + input).abs() < 1e-6,
            "{output} is not the negation of {input}"
        );
    }

    let mut half = AudioInvert::new().with_degree(0.5);
    half.configure_pipeline(&caps(1)).expect("configures");
    let mut out = Collect::default();
    half.process(to_frame(&samples, 1, 0), &mut out)
        .await
        .unwrap();
    assert!(
        out.samples().iter().all(|s| s.abs() < 1e-6),
        "half inversion cancels the signal"
    );
}

// ---------------------------------------------------------------------------
// audiokaraoke
// ---------------------------------------------------------------------------

/// How far down a centre-panned tone has to land for the effect to be doing its
/// job. GStreamer's own `audiokaraoke` puts a 1 kHz centre tone 32.5 dB down
/// with these defaults.
const KARAOKE_CENTRE_REJECTION_DB: f64 = 30.0;
const KARAOKE_TONE_HZ: f64 = 1000.0;

#[tokio::test]
async fn karaoke_cancels_a_centre_panned_tone() {
    let mut element = AudioKaraoke::new();
    let centre = db(measure_gain(&mut element, KARAOKE_TONE_HZ, 2).await);
    assert!(
        centre <= -KARAOKE_CENTRE_REJECTION_DB,
        "the centre tone is only {centre:.2} dB down"
    );
}

#[tokio::test]
async fn karaoke_passes_anti_phase_channels() {
    let mut element = AudioKaraoke::new();
    element.configure_pipeline(&caps(2)).expect("configures");
    let mono = sine(KARAOKE_TONE_HZ, BUFFER_SAMPLES, 1, 0);
    let mut stereo = Vec::with_capacity(mono.len() * 2);
    for sample in &mono {
        stereo.push(*sample);
        stereo.push(-*sample);
    }
    let mut out = Collect::default();
    element
        .process(to_frame(&stereo, 2, 0), &mut out)
        .await
        .unwrap();
    let gain = db(rms(&out.samples()) / rms(&stereo));
    // an anti-phase pair has no centre to cancel: subtracting the other channel
    // doubles each side.
    assert!(
        (gain - db(2.0)).abs() < 1.0,
        "anti-phase content came out at {gain:.2} dB"
    );
}

// ---------------------------------------------------------------------------
// launch lines
// ---------------------------------------------------------------------------

const SMOKE_BUFFERS: u64 = 10;

async fn launch_consumes_ten_buffers(element: &str) {
    let reg = default_registry();
    let line =
        format!("audiotestsrc num-buffers={SMOKE_BUFFERS} ! audioconvert ! {element} ! fakesink");
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("`{line}` parses: {e:?}"));
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("`{line}` runs: {e:?}"));
    assert_eq!(
        stats.frames_consumed, SMOKE_BUFFERS,
        "`{line}` delivered every buffer"
    );
}

#[tokio::test]
async fn audiodynamic_runs_in_a_text_pipeline() {
    launch_consumes_ten_buffers("audiodynamic mode=compressor threshold=0.5 ratio=0.25").await;
}

#[tokio::test]
async fn audioinvert_runs_in_a_text_pipeline() {
    launch_consumes_ten_buffers("audioinvert degree=1.0").await;
}

#[tokio::test]
async fn audiokaraoke_runs_in_a_text_pipeline() {
    launch_consumes_ten_buffers("audiokaraoke level=1.0 mono-level=0.5").await;
}

#[tokio::test]
async fn audiocheblimit_runs_in_a_text_pipeline() {
    launch_consumes_ten_buffers("audiocheblimit mode=low-pass cutoff=1000 poles=4 ripple=0.5")
        .await;
}

#[tokio::test]
async fn audiochebband_runs_in_a_text_pipeline() {
    launch_consumes_ten_buffers(
        "audiochebband mode=band-pass lower-frequency=1000 upper-frequency=4000",
    )
    .await;
}

/// A windowed-sinc filter holds its group delay back and pushes it at `Eos`, so
/// ten input buffers leave the sink as eleven, the same as in GStreamer.
async fn launch_wsinc(element: &str) {
    let reg = default_registry();
    let line =
        format!("audiotestsrc num-buffers={SMOKE_BUFFERS} ! audioconvert ! {element} ! fakesink");
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("`{line}` parses: {e:?}"));
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("`{line}` runs: {e:?}"));
    assert_eq!(
        stats.frames_consumed,
        SMOKE_BUFFERS + 1,
        "`{line}` delivered every buffer plus the tail"
    );
}

#[tokio::test]
async fn audiowsinclimit_runs_in_a_text_pipeline() {
    launch_wsinc("audiowsinclimit mode=low-pass cutoff=1000 length=101 window=hamming").await;
}

#[tokio::test]
async fn audiowsincband_runs_in_a_text_pipeline() {
    launch_wsinc(
        "audiowsincband mode=band-pass lower-frequency=1000 upper-frequency=4000 length=101",
    )
    .await;
}

/// Every new element answers `properties()` for the names its launch line uses.
#[test]
fn every_filter_declares_the_properties_its_launch_line_sets() {
    let dynamic = AudioDynamic::new();
    for name in ["mode", "characteristics", "threshold", "ratio"] {
        assert!(
            dynamic.properties().iter().any(|s| s.name == name),
            "audiodynamic declares {name}"
        );
        assert!(dynamic.get_property(name).is_some());
    }
    let karaoke = AudioKaraoke::new();
    for name in ["level", "mono-level", "filter-band", "filter-width"] {
        assert!(
            karaoke.properties().iter().any(|s| s.name == name),
            "audiokaraoke declares {name}"
        );
    }
    assert_eq!(
        AudioInvert::new().get_property("degree"),
        Some(PropValue::Double(0.0))
    );
}

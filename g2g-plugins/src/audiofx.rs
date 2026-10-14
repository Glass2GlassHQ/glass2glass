//! Filter kernels shared by the `audiofx` transforms: the PCM boundary every
//! one of them sits behind, the windowed-sinc FIR (`audiowsinclimit` /
//! `audiowsincband`) and the Chebyshev IIR cascade (`audiocheblimit` /
//! `audiochebband`). The coefficient math is ported from GStreamer's
//! `gst/audiofx/*.c`, so a g2g pipeline and a gst pipeline shape a signal the
//! same way.
//!
//! Every filter reads interleaved `PcmS16Le` or `PcmF32Le`, works in f32 (f64
//! inside the kernels) and writes the format back unchanged, so `audioconvert`
//! is placed ahead of them for any other format.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, G2gError, MemoryDomain, PadTemplate, PropError,
    PropValue,
};

pub(crate) use crate::audioconvert::{decode, encode};
use crate::fft::FftPlan;
use crate::fifobuffer::FifoBuffer;
use crate::mathf;

/// The sample formats the audiofx filters handle directly.
pub(crate) const AUDIOFX_FORMATS: [AudioFormat; 2] = [AudioFormat::PcmS16Le, AudioFormat::PcmF32Le];

/// The nominal shape the pad templates advertise, matching the other audio
/// filters' templates.
const TEMPLATE_CHANNELS: u8 = 2;
const TEMPLATE_RATE: u32 = 48_000;

/// Interleaved PCM the audiofx filters accept: one of [`AUDIOFX_FORMATS`], a
/// non-zero channel count, and exactly `required_channels` when the filter only
/// works on one layout (`audiokaraoke` is stereo).
pub(crate) fn accept_audio(
    caps: &Caps,
    required_channels: Option<u8>,
) -> Result<(AudioFormat, usize, u32), G2gError> {
    match caps {
        Caps::Audio {
            format,
            channels,
            sample_rate,
            ..
        } if AUDIOFX_FORMATS.contains(format)
            && *channels > 0
            && required_channels.is_none_or(|want| want == *channels) =>
        {
            Ok((*format, *channels as usize, *sample_rate))
        }
        _ => Err(G2gError::CapsMismatch),
    }
}

/// Output caps equal input caps: the filters change samples, not shape.
pub(crate) fn passthrough_constraint(required_channels: Option<u8>) -> CapsConstraint<'static> {
    CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
        match accept_audio(input, required_channels) {
            Ok(_) => CapsSet::one(input.clone()),
            Err(_) => CapsSet::from_alternatives(Vec::new()),
        }
    }))
}

/// Sink and source templates over [`AUDIOFX_FORMATS`].
pub(crate) fn pad_templates(channels: u8) -> Vec<PadTemplate> {
    let pcm = |format| Caps::Audio {
        format,
        channels,
        sample_rate: TEMPLATE_RATE,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    };
    let set = CapsSet::from_alternatives(AUDIOFX_FORMATS.map(pcm).to_vec());
    Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
}

/// The stereo-or-any template shape the filters declare.
pub(crate) fn default_pad_templates() -> Vec<PadTemplate> {
    pad_templates(TEMPLATE_CHANNELS)
}

// f32 and encoded buffers an element reuses across frames instead of allocating per frame
#[derive(Debug, Default)]
pub(crate) struct PcmScratch {
    samples: Vec<f32>,
    encoded: Vec<u8>,
}

impl PcmScratch {
    pub(crate) fn decode(&mut self, src: &[u8], format: AudioFormat) -> &[f32] {
        crate::audioconvert::decode_into(&mut self.samples, src, format);
        &self.samples
    }

    // a ragged trailing partial sample is dropped, as decode and encode always did
    pub(crate) fn transform_frame(
        &mut self,
        mut frame: Frame,
        format: AudioFormat,
        element: &'static str,
        transform: impl FnOnce(&mut [f32]),
    ) -> Result<Frame, G2gError> {
        crate::audioconvert::decode_into(
            &mut self.samples,
            frame.domain.require_system_slice(element)?,
            format,
        );
        transform(&mut self.samples);
        self.encoded.clear();
        crate::audioconvert::encode_into(&mut self.encoded, &self.samples, format);
        match &mut frame.domain {
            MemoryDomain::System(slice) if slice.as_slice().len() == self.encoded.len() => {
                slice.as_mut_slice().copy_from_slice(&self.encoded);
            }
            domain => {
                *domain =
                    MemoryDomain::System(SystemSlice::from_boxed(self.encoded.as_slice().into()));
            }
        }
        Ok(frame)
    }
}

/// Hold a mixed sample inside full scale. gst's `audiochannelmix` and `stereo`
/// are S16-only and clamp to the integer range; the g2g filters work in f32, so
/// the same saturation lands at +-1.
pub(crate) fn clamp_sample(value: f64) -> f32 {
    value.clamp(-1.0, 1.0) as f32
}

/// Read a `Double` property and reject anything outside `[min, max]`, the way a
/// GObject float property clamps its range at the spec.
pub(crate) fn double_in_range(value: PropValue, min: f64, max: f64) -> Result<f64, PropError> {
    let v = value.as_double().ok_or(PropError::Type)?;
    if v < min || v > max {
        return Err(PropError::Value);
    }
    Ok(v)
}

/// The separator of a coefficient list property. `PropKind` has no array kind,
/// so gst's `<a,b,c>` GstValueArray properties (`audiofirfilter kernel`,
/// `audioiirfilter a` / `b`, one row of `audiomixmatrix matrix`) are carried as
/// a `Str` holding this list.
pub(crate) const COEFFICIENT_SEPARATOR: char = ',';

/// Parse a comma-separated list of decimal floats. An empty (or blank) string
/// is an empty list; anything that is not a float is a [`PropError::Value`].
pub(crate) fn parse_coefficients(text: &str) -> Result<Vec<f64>, PropError> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let mut values = Vec::new();
    for entry in text.split(COEFFICIENT_SEPARATOR) {
        let value = entry.trim().parse::<f64>().map_err(|_| PropError::Value)?;
        if !value.is_finite() {
            return Err(PropError::Value);
        }
        values.push(value);
    }
    Ok(values)
}

/// A coefficient list back as the text [`parse_coefficients`] reads.
pub(crate) fn format_coefficients(values: &[f64]) -> String {
    use core::fmt::Write;
    let mut out = String::new();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(COEFFICIENT_SEPARATOR);
        }
        let _ = write!(out, "{value}");
    }
    out
}

/// Read an `Int` property and reject anything outside `[min, max]`.
pub(crate) fn int_in_range(value: PropValue, min: i64, max: i64) -> Result<i64, PropError> {
    let v = value.as_int().ok_or(PropError::Type)?;
    if v < min || v > max {
        return Err(PropError::Value);
    }
    Ok(v)
}

/// Low-pass / high-pass selector, the `mode` enum of `audiowsinclimit` and
/// `audiocheblimit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitMode {
    LowPass,
    HighPass,
}

impl LimitMode {
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s {
            "low-pass" => Some(Self::LowPass),
            "high-pass" => Some(Self::HighPass),
            _ => None,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::LowPass => "low-pass",
            Self::HighPass => "high-pass",
        }
    }
}

/// The spellings and order `gst-inspect` prints for a limit `mode`.
pub(crate) const LIMIT_MODE_VALUES: &str = "low-pass | high-pass";

/// Band-pass / band-reject selector, the `mode` enum of `audiowsincband` and
/// `audiochebband`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandMode {
    BandPass,
    BandReject,
}

impl BandMode {
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s {
            "band-pass" => Some(Self::BandPass),
            "band-reject" => Some(Self::BandReject),
            _ => None,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::BandPass => "band-pass",
            Self::BandReject => "band-reject",
        }
    }
}

/// The spellings and order `gst-inspect` prints for a band `mode`.
pub(crate) const BAND_MODE_VALUES: &str = "band-pass | band-reject";

/// Window applied to the sinc kernel, the `window` enum of the two windowed-sinc
/// filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirWindow {
    Hamming,
    Blackman,
    Gaussian,
    Cosine,
    Hann,
}

/// The spellings and order `gst-inspect` prints for `window`.
pub(crate) const FIR_WINDOW_VALUES: &str = "hamming | blackman | gaussian | cosine | hann";

impl FirWindow {
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s {
            "hamming" => Some(Self::Hamming),
            "blackman" => Some(Self::Blackman),
            "gaussian" => Some(Self::Gaussian),
            "cosine" => Some(Self::Cosine),
            "hann" => Some(Self::Hann),
            _ => None,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Hamming => "hamming",
            Self::Blackman => "blackman",
            Self::Gaussian => "gaussian",
            Self::Cosine => "cosine",
            Self::Hann => "hann",
        }
    }

    /// The window's value at tap `i` of a `len`-tap kernel.
    fn at(self, i: usize, len: usize) -> f64 {
        let pi = core::f64::consts::PI;
        let i = i as f64;
        let span = (len - 1) as f64;
        match self {
            Self::Hamming => 0.54 - 0.46 * mathf::cos(2.0 * pi * i / span),
            Self::Blackman => {
                0.42 - 0.5 * mathf::cos(2.0 * pi * i / span)
                    + 0.08 * mathf::cos(4.0 * pi * i / span)
            }
            Self::Gaussian => {
                let x = 3.0 / len as f64 * (2.0 * i - span);
                mathf::exp(-0.5 * x * x)
            }
            Self::Cosine => mathf::cos(pi * i / span - pi / 2.0),
            Self::Hann => 0.5 * (1.0 - mathf::cos(2.0 * pi * i / span)),
        }
    }
}

/// Windowed-sinc low-pass kernel at `cutoff` Hz, normalized for unity gain at
/// DC. `len` is odd, so the peak lands on the centre tap.
fn lowpass_kernel(cutoff: f64, rate: u32, len: usize, window: FirWindow) -> Vec<f64> {
    let w = 2.0 * core::f64::consts::PI * (cutoff / rate as f64);
    let middle = (len - 1) as f64 / 2.0;
    let mut kernel = vec![0.0f64; len];
    let mut sum = 0.0;
    for (i, tap) in kernel.iter_mut().enumerate() {
        let offset = i as f64 - middle;
        *tap = if offset == 0.0 {
            w
        } else {
            mathf::sin(w * offset) / offset
        };
        *tap *= window.at(i, len);
        sum += *tap;
    }
    for tap in kernel.iter_mut() {
        *tap /= sum;
    }
    kernel
}

/// Spectral inversion: negate the kernel and add unity at the centre tap, which
/// turns a low-pass into its complementary high-pass.
fn spectral_invert(kernel: &mut [f64]) {
    let len = kernel.len();
    for tap in kernel.iter_mut() {
        *tap = -*tap;
    }
    if len % 2 == 1 {
        kernel[(len - 1) / 2] += 1.0;
    } else {
        kernel[len / 2 - 1] += 0.5;
        kernel[len / 2] += 0.5;
    }
}

/// `audiowsinclimit`'s kernel: a windowed-sinc low-pass, spectrally inverted for
/// high-pass.
pub(crate) fn limit_kernel(
    mode: LimitMode,
    cutoff: f64,
    rate: u32,
    len: usize,
    window: FirWindow,
) -> Vec<f64> {
    let cutoff = cutoff.clamp(0.0, rate as f64 / 2.0);
    let mut kernel = lowpass_kernel(cutoff, rate, len, window);
    if mode == LimitMode::HighPass {
        spectral_invert(&mut kernel);
    }
    kernel
}

/// `audiowsincband`'s kernel: the sum of a low-pass at `lower` and a high-pass
/// at `upper` is a band-reject, spectrally inverted for band-pass.
pub(crate) fn band_kernel(
    mode: BandMode,
    lower: f64,
    upper: f64,
    rate: u32,
    len: usize,
    window: FirWindow,
) -> Vec<f64> {
    let nyquist = rate as f64 / 2.0;
    let mut lower = lower.clamp(0.0, nyquist);
    let mut upper = upper.clamp(0.0, nyquist);
    if lower > upper {
        core::mem::swap(&mut lower, &mut upper);
    }
    let low = lowpass_kernel(lower, rate, len, window);
    let mut high = lowpass_kernel(upper, rate, len, window);
    spectral_invert(&mut high);
    let mut kernel: Vec<f64> = low.iter().zip(high.iter()).map(|(l, h)| l + h).collect();
    if mode == BandMode::BandPass {
        for tap in kernel.iter_mut() {
            *tap = -*tap;
        }
        kernel[len / 2] += 1.0;
    }
    kernel
}

// gst audiowsinclimit and audiowsincband kernel length default and bounds
pub(crate) const WSINC_DEFAULT_LENGTH: i64 = 101;
pub(crate) const WSINC_LENGTH_MIN: i64 = 3;
pub(crate) const WSINC_LENGTH_MAX: i64 = 256_000;

// a linear-phase kernel needs an odd tap count so its peak lands on one sample
pub(crate) fn odd_wsinc_length(length: i64) -> usize {
    let length = length.clamp(WSINC_LENGTH_MIN, WSINC_LENGTH_MAX);
    (if length % 2 == 0 { length + 1 } else { length }) as usize
}

// gst audiofxbasefirfilter switches to FFT overlap-save convolution at this kernel length
const FFT_THRESHOLD: usize = 32;
// gst filters 4 * kernel_length samples per FFT block
const FFT_BLOCK_KERNELS: usize = 4;

/// Time-domain FIR over interleaved frames, the reference's low-latency path.
#[derive(Debug, Default)]
struct DirectFir {
    kernel: Vec<f64>,
    taps: usize,
    channels: usize,
    // per channel, each sample stored at `pos` and `pos + taps`, so the newest `taps` are one slice
    history: Vec<f32>,
    pos: usize,
}

impl DirectFir {
    fn new(kernel: Vec<f64>, channels: usize) -> Self {
        let taps = kernel.len().max(1);
        Self {
            kernel,
            taps,
            channels,
            history: vec![0.0; 2 * taps * channels],
            pos: 0,
        }
    }

    fn reset(&mut self) {
        self.history.fill(0.0);
        self.pos = 0;
    }

    fn run(&mut self, input: &[f32], output: &mut Vec<f32>) {
        if self.channels == 0 {
            return;
        }
        let taps = self.taps;
        for frame in input.chunks_exact(self.channels) {
            for (channel, &sample) in frame.iter().enumerate() {
                let ring = &mut self.history[channel * 2 * taps..][..2 * taps];
                ring[self.pos] = sample;
                ring[self.pos + taps] = sample;
                let newest_first = ring[self.pos + 1..=self.pos + taps].iter().rev();
                let acc = self
                    .kernel
                    .iter()
                    .zip(newest_first)
                    .fold(0.0f64, |acc, (coefficient, &sample)| {
                        acc + coefficient * sample as f64
                    });
                output.push(acc as f32);
            }
            self.pos = (self.pos + 1) % taps;
        }
    }
}

/// FFT overlap-save FIR, gst's path for long kernels. Output comes a block at a
/// time, so a call can return fewer frames than it was given.
#[derive(Debug)]
struct FftFir {
    plan: FftPlan,
    kernel_re: Vec<f64>,
    kernel_im: Vec<f64>,
    taps: usize,
    channels: usize,
    // per channel, the last `taps - 1` inputs the next block overlaps
    overlap: Vec<f64>,
    pending: FifoBuffer<f32>,
    re: Vec<f64>,
    im: Vec<f64>,
}

impl FftFir {
    fn new(kernel: &[f64], channels: usize) -> Self {
        let taps = kernel.len();
        let plan = FftPlan::new((FFT_BLOCK_KERNELS * taps).next_power_of_two());
        let size = plan.size();
        let mut kernel_re = vec![0.0; size];
        kernel_re[..taps].copy_from_slice(kernel);
        let mut kernel_im = vec![0.0; size];
        plan.forward(&mut kernel_re, &mut kernel_im);
        Self {
            plan,
            kernel_re,
            kernel_im,
            taps,
            channels,
            overlap: vec![0.0; (taps - 1) * channels],
            pending: FifoBuffer::default(),
            re: vec![0.0; size],
            im: vec![0.0; size],
        }
    }

    fn reset(&mut self) {
        self.overlap.fill(0.0);
        self.pending.clear();
    }

    // new input frames one block filters
    fn block_frames(&self) -> usize {
        self.plan.size() - (self.taps - 1)
    }

    fn run(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.pending.extend_from_slice(input);
        let block_samples = self.block_frames() * self.channels;
        while self.pending.len() >= block_samples {
            self.filter_block(output);
        }
    }

    fn flush(&mut self, output: &mut Vec<f32>) {
        let frames = self.pending.len() / self.channels;
        if frames == 0 {
            return;
        }
        let padded = self.block_frames() * self.channels;
        self.pending.resize(padded, 0.0);
        let start = output.len();
        self.filter_block(output);
        output.truncate(start + frames * self.channels);
    }

    fn filter_block(&mut self, output: &mut Vec<f32>) {
        let frames = self.block_frames();
        let overlap_len = self.taps - 1;
        let start = output.len();
        output.resize(start + frames * self.channels, 0.0);
        for channel in 0..self.channels {
            let overlap = &mut self.overlap[channel * overlap_len..][..overlap_len];
            self.re[..overlap_len].copy_from_slice(overlap);
            let block = &self.pending.as_slice()[..frames * self.channels];
            let new_inputs = block.iter().skip(channel).step_by(self.channels);
            for (slot, &sample) in self.re[overlap_len..].iter_mut().zip(new_inputs) {
                *slot = sample as f64;
            }
            let newest = &self.re[self.plan.size() - overlap_len..];
            overlap.copy_from_slice(newest);
            self.im.fill(0.0);
            self.plan.forward(&mut self.re, &mut self.im);
            let spectrum = self.re.iter_mut().zip(self.im.iter_mut());
            for ((re, im), (kr, ki)) in spectrum.zip(self.kernel_re.iter().zip(&self.kernel_im)) {
                let product_re = *re * kr - *im * ki;
                *im = *re * ki + *im * kr;
                *re = product_re;
            }
            self.plan.inverse(&mut self.re, &mut self.im);
            let filtered = output[start..]
                .iter_mut()
                .skip(channel)
                .step_by(self.channels);
            for (slot, &value) in filtered.zip(&self.re[overlap_len..]) {
                *slot = value as f32;
            }
        }
        self.pending.discard(frames * self.channels);
    }
}

#[derive(Debug)]
enum FirConvolver {
    Direct(DirectFir),
    Fft(FftFir),
}

impl Default for FirConvolver {
    fn default() -> Self {
        Self::Direct(DirectFir::default())
    }
}

impl FirConvolver {
    fn new(kernel: Vec<f64>, channels: usize) -> Self {
        if kernel.len() >= FFT_THRESHOLD {
            Self::Fft(FftFir::new(&kernel, channels))
        } else {
            Self::Direct(DirectFir::new(kernel, channels))
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Direct(fir) => fir.reset(),
            Self::Fft(fir) => fir.reset(),
        }
    }

    fn run(&mut self, input: &[f32], output: &mut Vec<f32>) {
        match self {
            Self::Direct(fir) => fir.run(input, output),
            Self::Fft(fir) => fir.run(input, output),
        }
    }

    // the input a block-based convolver still holds
    fn flush(&mut self, output: &mut Vec<f32>) {
        match self {
            Self::Direct(_) => {}
            Self::Fft(fir) => fir.flush(output),
        }
    }
}

/// The group delay a linear-phase kernel of `taps` taps introduces.
fn linear_phase_latency(taps: usize) -> usize {
    taps.saturating_sub(1) / 2
}

/// The streaming half of a windowed-sinc filter: the kernel, the history, and
/// the group-delay bookkeeping `gst_audio_fx_base_fir_filter_transform` does.
/// A linear-phase FIR delays everything by `(length - 1) / 2` samples, so the
/// leading output frames of that many samples are dropped and the same count is
/// convolved out of the history at `Eos`, leaving the output aligned with the
/// input and the sample count unchanged.
#[derive(Debug, Default)]
pub(crate) struct FirStream {
    convolver: FirConvolver,
    channels: usize,
    latency: usize,
    /// output frames still to be dropped to remove the group delay.
    skip: usize,
    /// output frames pushed so far, the pts cursor of the stream.
    emitted_frames: u64,
    started: bool,
    output: Vec<f32>,
    // where the samples to push start in `output`, past the dropped group delay
    output_start: usize,
}

impl FirStream {
    /// Configure with the group delay the caller worked out: the linear-phase
    /// one for the windowed-sinc filters, `audiofirfilter`'s `latency` property
    /// for an arbitrary kernel, which has no delay the filter can derive.
    pub(crate) fn configure_with_latency(
        &mut self,
        kernel: Vec<f64>,
        channels: usize,
        latency: usize,
    ) {
        self.latency = latency;
        self.channels = channels.max(1);
        self.convolver = FirConvolver::new(kernel, self.channels);
        self.skip = self.latency;
        self.emitted_frames = 0;
        self.started = false;
    }

    /// Swap in a rebuilt kernel. Once samples are flowing the history is cleared
    /// rather than drained (the reference pushes its residue, which a property
    /// setter here has no output pad for), so the next `latency` samples carry a
    /// transient.
    pub(crate) fn set_kernel(&mut self, kernel: Vec<f64>) {
        let latency = linear_phase_latency(kernel.len());
        self.set_kernel_with_latency(kernel, latency);
    }

    /// Swap in a rebuilt kernel and its group delay together.
    pub(crate) fn set_kernel_with_latency(&mut self, kernel: Vec<f64>, latency: usize) {
        self.latency = latency;
        self.convolver = FirConvolver::new(kernel, self.channels.max(1));
        if !self.started {
            self.skip = self.latency;
        }
    }

    pub(crate) fn reset(&mut self) {
        self.convolver.reset();
        self.skip = self.latency;
        self.emitted_frames = 0;
        self.started = false;
    }

    pub(crate) fn latency(&self) -> usize {
        self.latency
    }

    /// Whether any input has reached the filter yet.
    pub(crate) fn started(&self) -> bool {
        self.started
    }

    /// Output frames pushed so far: the offset the next buffer's pts sits at.
    pub(crate) fn emitted_frames(&self) -> u64 {
        self.emitted_frames
    }

    /// The samples the last `run` or `drain` produced. Early in the stream that
    /// is shorter than the input, and can be empty.
    pub(crate) fn filtered(&self) -> &[f32] {
        &self.output[self.output_start..]
    }

    /// Filter one interleaved buffer into [`filtered`](Self::filtered).
    pub(crate) fn run(&mut self, input: &[f32]) {
        self.started = true;
        self.output.clear();
        self.convolver.run(input, &mut self.output);
        self.drop_group_delay();
    }

    /// The tail the history still holds at `Eos`, convolved against silence.
    pub(crate) fn drain(&mut self) {
        self.output.clear();
        self.output_start = 0;
        if !self.started {
            return;
        }
        let silence = vec![0.0f32; self.latency * self.channels];
        self.convolver.run(&silence, &mut self.output);
        self.convolver.flush(&mut self.output);
        self.drop_group_delay();
    }

    fn drop_group_delay(&mut self) {
        let dropped = (self.skip * self.channels).min(self.output.len());
        self.skip -= dropped / self.channels;
        self.output_start = dropped;
        self.emitted_frames += ((self.output.len() - dropped) / self.channels) as u64;
    }
}

/// The PCM plumbing both windowed-sinc elements sit on: caps, the sample
/// boundary, and the timestamps that follow from dropping the group delay.
#[derive(Debug)]
pub(crate) struct FirTransform {
    stream: FirStream,
    format: AudioFormat,
    channels: usize,
    rate: u32,
    caps: Option<Caps>,
    start_pts_ns: u64,
    /// Everything but pts / dts / duration is carried over from the last input.
    timing: FrameTiming,
    emitted: u64,
    scratch: PcmScratch,
}

impl Default for FirTransform {
    fn default() -> Self {
        Self {
            stream: FirStream::default(),
            format: AUDIOFX_FORMATS[0],
            channels: 0,
            rate: 0,
            caps: None,
            start_pts_ns: 0,
            timing: FrameTiming::default(),
            emitted: 0,
            scratch: PcmScratch::default(),
        }
    }
}

impl FirTransform {
    pub(crate) fn configure(&mut self, caps: &Caps, kernel: Vec<f64>) -> Result<(), G2gError> {
        let latency = linear_phase_latency(kernel.len());
        self.configure_with_latency(caps, kernel, latency)
    }

    /// Configure with the caller's group delay, for `audiofirfilter`'s
    /// `latency` property.
    pub(crate) fn configure_with_latency(
        &mut self,
        caps: &Caps,
        kernel: Vec<f64>,
        latency: usize,
    ) -> Result<(), G2gError> {
        let (format, channels, rate) = accept_audio(caps, None)?;
        self.format = format;
        self.channels = channels;
        self.rate = rate;
        self.caps = Some(caps.clone());
        self.stream
            .configure_with_latency(kernel, channels, latency);
        Ok(())
    }

    pub(crate) fn caps(&self) -> Option<&Caps> {
        self.caps.as_ref()
    }

    pub(crate) fn rate(&self) -> u32 {
        self.rate
    }

    pub(crate) fn set_kernel(&mut self, kernel: Vec<f64>) {
        self.stream.set_kernel(kernel);
    }

    pub(crate) fn set_kernel_with_latency(&mut self, kernel: Vec<f64>, latency: usize) {
        self.stream.set_kernel_with_latency(kernel, latency);
    }

    pub(crate) fn latency(&self) -> usize {
        self.stream.latency()
    }

    pub(crate) fn reset(&mut self) {
        self.stream.reset();
        self.start_pts_ns = 0;
    }

    /// Filter one input frame. Nothing comes back until the leading group delay
    /// has been dropped.
    pub(crate) fn filter(
        &mut self,
        frame: &Frame,
        element: &'static str,
    ) -> Result<Option<Frame>, G2gError> {
        let src = frame.domain.require_system_slice(element)?;
        if !self.stream.started() {
            self.start_pts_ns = frame.timing.pts_ns;
        }
        self.timing = frame.timing;
        self.stream.run(self.scratch.decode(src, self.format));
        Ok(self.wrap())
    }

    /// The kernel's tail at `Eos`.
    pub(crate) fn drain(&mut self) -> Option<Frame> {
        self.stream.drain();
        self.wrap()
    }

    fn wrap(&mut self) -> Option<Frame> {
        let samples = self.stream.filtered();
        if samples.is_empty() {
            return None;
        }
        let frames = (samples.len() / self.channels.max(1)) as u64;
        let offset = self.stream.emitted_frames() - frames;
        let pts = self
            .start_pts_ns
            .saturating_add(crate::audioconvert::samples_to_ns(offset, self.rate));
        let timing = FrameTiming {
            pts_ns: pts,
            dts_ns: pts,
            duration_ns: crate::audioconvert::samples_to_ns(frames, self.rate),
            ..self.timing
        };
        let encoded = encode(samples, self.format);
        let sequence = self.emitted;
        self.emitted += 1;
        Some(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(encoded)),
            timing,
            sequence,
            meta: Default::default(),
        })
    }
}

/// One cascade section, up to order four: the band filters pair their poles so
/// their sections are quartic. `y[n] = sum b[i]*x[n-i] + sum a[i]*y[n-i]`, the
/// sign convention the reference's cascade multiplication assumes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IirSection {
    pub(crate) b: [f64; 5],
    /// `a[0]` is unused: the denominator is monic.
    pub(crate) a: [f64; 5],
}

impl IirSection {
    /// The identity section, used before the rate is known.
    pub(crate) fn identity() -> Self {
        let mut b = [0.0; 5];
        b[0] = 1.0;
        Self { b, a: [0.0; 5] }
    }

    /// A section that mutes its input.
    fn zero() -> Self {
        Self {
            b: [0.0; 5],
            a: [0.0; 5],
        }
    }

    /// |H(z)| at a point on the unit circle, where `z^-1` is the conjugate.
    fn magnitude(&self, zr: f64, zi: f64) -> f64 {
        let (mut nr, mut ni) = (0.0, 0.0);
        let (mut dr, mut di) = (1.0, 0.0);
        // powers of z^-1 = conj(z).
        let (mut pr, mut pi) = (1.0, 0.0);
        for i in 0..5 {
            nr += self.b[i] * pr;
            ni += self.b[i] * pi;
            if i > 0 {
                dr -= self.a[i] * pr;
                di -= self.a[i] * pi;
            }
            let next_r = pr * zr + pi * zi;
            let next_i = pi * zr - pr * zi;
            pr = next_r;
            pi = next_i;
        }
        let numerator = mathf::sqrt(nr * nr + ni * ni);
        let denominator = mathf::sqrt(dr * dr + di * di);
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

/// Direct-form-I state of one section on one channel.
#[derive(Debug, Clone, Copy, Default)]
struct IirSectionState {
    x: [f64; 4],
    y: [f64; 4],
}

impl IirSectionState {
    fn step(&mut self, section: &IirSection, x0: f64) -> f64 {
        let mut acc = section.b[0] * x0;
        for i in 0..4 {
            acc += section.b[i + 1] * self.x[i] + section.a[i + 1] * self.y[i];
        }
        self.x = [x0, self.x[0], self.x[1], self.x[2]];
        self.y = [acc, self.y[0], self.y[1], self.y[2]];
        acc
    }
}

/// A cascade of [`IirSection`]s with per-channel state.
#[derive(Debug, Default)]
pub(crate) struct IirCascade {
    sections: Vec<IirSection>,
    /// `sections.len() * channels` states, indexed `section * channels + channel`.
    state: Vec<IirSectionState>,
    channels: usize,
}

impl IirCascade {
    pub(crate) fn set_sections(&mut self, sections: Vec<IirSection>, channels: usize) {
        self.sections = sections;
        self.channels = channels.max(1);
        self.state = vec![IirSectionState::default(); self.sections.len() * self.channels];
    }

    pub(crate) fn reset(&mut self) {
        for state in self.state.iter_mut() {
            *state = IirSectionState::default();
        }
    }

    // new coefficients for the same number of sections, the filter state carries over
    pub(crate) fn retune(&mut self, sections: &[IirSection]) {
        self.sections.copy_from_slice(sections);
    }

    /// |H(z)| of the whole cascade at a point on the unit circle.
    pub(crate) fn magnitude(&self, zr: f64, zi: f64) -> f64 {
        self.sections
            .iter()
            .map(|s| s.magnitude(zr, zi))
            .product::<f64>()
    }

    /// Divide the cascade's gain by `gain`.
    pub(crate) fn normalize_gain(&mut self, gain: f64) {
        if gain == 0.0 {
            return;
        }
        // one section carries the whole correction; the cascade is a product.
        if let Some(first) = self.sections.first_mut() {
            for coefficient in first.b.iter_mut() {
                *coefficient /= gain;
            }
        }
    }

    /// Scale the numerators so the cascade has unity gain at `(zr, zi)`.
    pub(crate) fn normalize_at(&mut self, zr: f64, zi: f64) {
        self.normalize_gain(self.magnitude(zr, zi));
    }

    /// Filter interleaved samples in place.
    pub(crate) fn run(&mut self, samples: &mut [f32]) {
        if self.sections.is_empty() || self.channels == 0 {
            return;
        }
        for (index, sample) in samples.iter_mut().enumerate() {
            *sample = self.filter_sample(index % self.channels, *sample as f64) as f32;
        }
    }

    pub(crate) fn filter_sample(&mut self, channel: usize, mut value: f64) -> f64 {
        for (section_index, section) in self.sections.iter().enumerate() {
            value = self.state[section_index * self.channels + channel].step(section, value);
        }
        value
    }
}

/// The z-plane prototype of one Chebyshev pole pair: a low-pass at frequency 1
/// carried through the bilinear transform, `(x0, x1, x2, y1, y2)`.
///
/// `p` is the 1-based pair index, `np` the pole count of the prototype, `kind`
/// 1 (all-pole, pass-band ripple) or 2 (zeros in the stop band).
fn chebyshev_prototype(p: usize, np: usize, kind: u8, ripple: f64) -> [f64; 5] {
    let pi = core::f64::consts::PI;
    let np_f = np as f64;

    // pole location on the unit circle for a low-pass at frequency 1.
    let angle = (pi / 2.0) * (2.0 * p as f64 - 1.0) / np_f;
    let mut rp = -mathf::sin(angle);
    let mut ip = mathf::cos(angle);

    // ripple moves the pole from the circle onto an ellipse.
    if kind == 1 && ripple > 0.0 {
        let es = mathf::sqrt(mathf::powf(10.0, ripple / 10.0) - 1.0);
        let vx = (1.0 / np_f) * mathf::asinh(1.0 / es);
        rp *= mathf::sinh(vx);
        ip *= mathf::cosh(vx);
    } else if kind == 2 {
        let es = mathf::sqrt(mathf::powf(10.0, ripple / 10.0) - 1.0);
        let vx = (1.0 / np_f) * mathf::asinh(es);
        rp *= mathf::sinh(vx);
        ip *= mathf::cosh(vx);
    }

    // type 2 inverts the pole and puts a zero on the unit circle.
    let mut iz = 0.0;
    if kind == 2 {
        let magnitude = rp * rp + ip * ip;
        rp /= magnitude;
        ip /= magnitude;
        let zero_angle = pi / (np_f * 2.0) + ((p as f64 - 1.0) * pi) / np_f;
        iz = mathf::cos(zero_angle);
        iz /= iz * iz;
    }

    // bilinear transform, substituting s by (2/t)*((z-1)/(z+1)) with
    // t = 2*tan(0.5).
    let t = 2.0 * mathf::tan(0.5);
    let m = rp * rp + ip * ip;
    let d = 4.0 - 4.0 * rp * t + m * t * t;
    let (x0, x1, x2) = if kind == 1 {
        let x0 = (t * t) / d;
        (x0, 2.0 * x0, x0)
    } else {
        let x0 = (t * t * iz * iz + 4.0) / d;
        (x0, (-8.0 + 2.0 * iz * iz * t * t) / d, x0)
    };
    let y1 = (8.0 - 2.0 * m * t * t) / d;
    let y2 = (-4.0 - 4.0 * rp * t - m * t * t) / d;
    [x0, x1, x2, y1, y2]
}

/// The Chebyshev knobs both `audiocheb*` elements carry.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChebSettings {
    /// 1 puts the ripple in the pass band, 2 in the stop band.
    pub(crate) kind: u8,
    pub(crate) poles: usize,
    pub(crate) ripple_db: f64,
    pub(crate) channels: usize,
}

/// `audiocheblimit`'s cascade: the prototype's `z^-1` is substituted by the
/// low-pass or high-pass all-pass section that moves the cutoff to `cutoff`.
fn cheb_limit_cascade(
    mode: LimitMode,
    cutoff: f64,
    rate: u32,
    settings: ChebSettings,
) -> Vec<IirSection> {
    let ChebSettings {
        kind,
        poles,
        ripple_db: ripple,
        ..
    } = settings;
    if rate == 0 {
        return Vec::from([IirSection::identity()]);
    }
    let nyquist = rate as f64 / 2.0;
    if cutoff >= nyquist {
        return Vec::from([match mode {
            LimitMode::LowPass => IirSection::identity(),
            LimitMode::HighPass => IirSection::zero(),
        }]);
    }
    if cutoff <= 0.0 {
        return Vec::from([match mode {
            LimitMode::LowPass => IirSection::zero(),
            LimitMode::HighPass => IirSection::identity(),
        }]);
    }

    let omega = 2.0 * core::f64::consts::PI * (cutoff / rate as f64);
    let k = match mode {
        LimitMode::LowPass => mathf::sin((1.0 - omega) / 2.0) / mathf::sin((1.0 + omega) / 2.0),
        LimitMode::HighPass => -mathf::cos((omega + 1.0) / 2.0) / mathf::cos((omega - 1.0) / 2.0),
    };

    let mut sections = Vec::with_capacity(poles / 2);
    for p in 1..=poles / 2 {
        let [x0, x1, x2, y1, y2] = chebyshev_prototype(p, poles, kind, ripple);
        let d = 1.0 + y1 * k - y2 * k * k;
        let mut b = [0.0; 5];
        let mut a = [0.0; 5];
        b[0] = (x0 + k * (-x1 + k * x2)) / d;
        b[1] = (x1 + k * k * x1 - 2.0 * k * (x0 + x2)) / d;
        b[2] = (x0 * k * k - x1 * k + x2) / d;
        a[1] = (2.0 * k + y1 + y1 * k * k - 2.0 * y2 * k) / d;
        a[2] = (-k * k - y1 * k + y2) / d;
        if mode == LimitMode::HighPass {
            a[1] = -a[1];
            b[1] = -b[1];
        }
        sections.push(IirSection { b, a });
    }
    sections
}

/// `audiochebband`'s cascade: the prototype's `z^-1` is substituted by the
/// second-order all-pass section that maps the low-pass onto the band, so each
/// pole pair becomes a quartic section.
fn cheb_band_cascade(
    mode: BandMode,
    lower: f64,
    upper: f64,
    rate: u32,
    settings: ChebSettings,
) -> Vec<IirSection> {
    let ChebSettings {
        kind,
        poles,
        ripple_db: ripple,
        ..
    } = settings;
    if rate == 0 {
        return Vec::from([IirSection::identity()]);
    }
    if upper <= lower {
        return Vec::from([match mode {
            BandMode::BandPass => IirSection::zero(),
            BandMode::BandReject => IirSection::identity(),
        }]);
    }
    let lower = lower.max(0.0);
    let upper = upper.min(rate as f64 / 2.0);

    let pi = core::f64::consts::PI;
    let w0 = 2.0 * pi * (lower / rate as f64);
    let w1 = 2.0 * pi * (upper / rate as f64);
    let a = mathf::cos((w1 + w0) / 2.0) / mathf::cos((w1 - w0) / 2.0);
    let (alpha, beta) = match mode {
        BandMode::BandPass => {
            let b = mathf::tan(0.5) / mathf::tan((w1 - w0) / 2.0);
            ((2.0 * a * b) / (1.0 + b), (b - 1.0) / (b + 1.0))
        }
        BandMode::BandReject => {
            let b = mathf::tan(0.5) * mathf::tan((w1 - w0) / 2.0);
            ((2.0 * a) / (1.0 + b), (1.0 - b) / (1.0 + b))
        }
    };

    // the prototype has half as many poles as the band filter's order.
    let prototype_poles = poles / 2;
    let mut sections = Vec::with_capacity(poles / 4);
    for p in 1..=poles / 4 {
        let [x0, x1, x2, y1, y2] = chebyshev_prototype(p, prototype_poles, kind, ripple);
        let mut b = [0.0; 5];
        let mut coefficients = [0.0; 5];
        match mode {
            BandMode::BandPass => {
                let d = 1.0 + beta * (y1 - beta * y2);
                b[0] = (x0 + beta * (-x1 + beta * x2)) / d;
                b[1] = (alpha * (-2.0 * x0 + x1 + beta * x1 - 2.0 * beta * x2)) / d;
                b[2] = (-x1 - beta * beta * x1
                    + 2.0 * beta * (x0 + x2)
                    + alpha * alpha * (x0 - x1 + x2))
                    / d;
                b[3] = (alpha * (x1 + beta * (-2.0 * x0 + x1) - 2.0 * x2)) / d;
                b[4] = (beta * (beta * x0 - x1) + x2) / d;
                coefficients[1] = (alpha * (2.0 + y1 + beta * y1 - 2.0 * beta * y2)) / d;
                coefficients[2] = (-y1 - beta * beta * y1 - alpha * alpha * (1.0 + y1 - y2)
                    + 2.0 * beta * (-1.0 + y2))
                    / d;
                coefficients[3] = (alpha * (y1 + beta * (2.0 + y1) - 2.0 * y2)) / d;
                coefficients[4] = (-beta * beta - beta * y1 + y2) / d;
            }
            BandMode::BandReject => {
                let d = -1.0 + beta * (beta * y2 + y1);
                b[0] = (-x0 - beta * x1 - beta * beta * x2) / d;
                b[1] = (alpha * (2.0 * x0 + x1 + beta * x1 + 2.0 * beta * x2)) / d;
                b[2] = (-x1
                    - beta * beta * x1
                    - 2.0 * beta * (x0 + x2)
                    - alpha * alpha * (x0 + x1 + x2))
                    / d;
                b[3] = (alpha * (x1 + beta * (2.0 * x0 + x1) + 2.0 * x2)) / d;
                b[4] = (-beta * beta * x0 - beta * x1 - x2) / d;
                coefficients[1] = (alpha * (-2.0 + y1 + beta * y1 + 2.0 * beta * y2)) / d;
                coefficients[2] = -(y1
                    + beta * beta * y1
                    + 2.0 * beta * (-1.0 + y2)
                    + alpha * alpha * (-1.0 + y1 + y2))
                    / d;
                coefficients[3] = (alpha * (beta * (-2.0 + y1) + y1 + 2.0 * y2)) / d;
                coefficients[4] = -(-beta * beta + beta * y1 + y2) / d;
            }
        }
        sections.push(IirSection { b, a: coefficients });
    }
    sections
}

/// `audiocheblimit`'s filter: the cascade normalized for unity gain at DC
/// (low-pass) or at Nyquist (high-pass).
pub(crate) fn cheb_limit_filter(
    mode: LimitMode,
    cutoff: f64,
    rate: u32,
    settings: ChebSettings,
) -> IirCascade {
    let mut cascade = IirCascade::default();
    cascade.set_sections(
        cheb_limit_cascade(mode, cutoff, rate, settings),
        settings.channels,
    );
    match mode {
        LimitMode::LowPass => cascade.normalize_at(1.0, 0.0),
        LimitMode::HighPass => cascade.normalize_at(-1.0, 0.0),
    }
    cascade
}

/// `audiochebband`'s filter: the cascade normalized for unity gain at the band
/// centre (band-pass), or at the geometric mean of DC and Nyquist gain
/// (band-reject), as the reference does.
pub(crate) fn cheb_band_filter(
    mode: BandMode,
    lower: f64,
    upper: f64,
    rate: u32,
    settings: ChebSettings,
) -> IirCascade {
    let mut cascade = IirCascade::default();
    cascade.set_sections(
        cheb_band_cascade(mode, lower, upper, rate, settings),
        settings.channels,
    );
    if rate == 0 {
        return cascade;
    }
    match mode {
        BandMode::BandReject => {
            let gain = mathf::sqrt(cascade.magnitude(1.0, 0.0) * cascade.magnitude(-1.0, 0.0));
            cascade.normalize_gain(gain);
        }
        BandMode::BandPass => {
            let pi = core::f64::consts::PI;
            let centre = pi * (lower + upper) / rate as f64;
            cascade.normalize_at(mathf::cos(centre), mathf::sin(centre));
        }
    }
    cascade
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Amplitude response of a kernel at `frequency`, evaluated as the magnitude
    /// of its discrete-time Fourier transform.
    fn kernel_gain(kernel: &[f64], frequency: f64, rate: u32) -> f64 {
        let w = 2.0 * core::f64::consts::PI * frequency / rate as f64;
        let mut re = 0.0;
        let mut im = 0.0;
        for (i, tap) in kernel.iter().enumerate() {
            re += tap * mathf::cos(w * i as f64);
            im -= tap * mathf::sin(w * i as f64);
        }
        mathf::sqrt(re * re + im * im)
    }

    const TEST_KERNEL_TAPS: usize = 101;
    const TEST_CHANNELS: usize = 2;
    // odd, so buffers straddle FFT block boundaries
    const TEST_BUFFER_FRAMES: usize = 97;

    fn test_signal(frames: usize) -> Vec<f32> {
        (0..frames * TEST_CHANNELS)
            .map(|i| ((i * 37) % 101) as f32 / 101.0 - 0.5)
            .collect()
    }

    fn test_kernel() -> Vec<f64> {
        limit_kernel(
            LimitMode::LowPass,
            1000.0,
            48_000,
            TEST_KERNEL_TAPS,
            FirWindow::Hamming,
        )
    }

    #[test]
    fn fft_overlap_save_matches_direct_convolution() {
        let kernel = test_kernel();
        assert!(kernel.len() >= FFT_THRESHOLD);
        let input = test_signal(3001);
        let mut expected = Vec::new();
        DirectFir::new(kernel.clone(), TEST_CHANNELS).run(&input, &mut expected);
        let mut fft = FftFir::new(&kernel, TEST_CHANNELS);
        let mut got = Vec::new();
        for buffer in input.chunks(TEST_BUFFER_FRAMES * TEST_CHANNELS) {
            fft.run(buffer, &mut got);
        }
        fft.flush(&mut got);
        assert_eq!(got.len(), expected.len());
        for (index, (fast, slow)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (fast - slow).abs() < 1e-5,
                "sample {index}: {fast} vs {slow}"
            );
        }
    }

    #[test]
    fn a_long_kernel_stream_keeps_the_sample_count() {
        let kernel = test_kernel();
        let latency = linear_phase_latency(kernel.len());
        let frames = 1000;
        let mut stream = FirStream::default();
        stream.configure_with_latency(kernel, TEST_CHANNELS, latency);
        let mut total = 0;
        for buffer in test_signal(frames).chunks(TEST_BUFFER_FRAMES * TEST_CHANNELS) {
            stream.run(buffer);
            total += stream.filtered().len();
        }
        stream.drain();
        total += stream.filtered().len();
        assert_eq!(total, frames * TEST_CHANNELS);
        assert_eq!(stream.emitted_frames(), frames as u64);
    }

    #[test]
    fn lowpass_kernel_has_unity_dc_gain() {
        let kernel = limit_kernel(LimitMode::LowPass, 1000.0, 48_000, 101, FirWindow::Hamming);
        assert!((kernel.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((kernel_gain(&kernel, 0.0, 48_000) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn highpass_is_the_lowpass_complement() {
        let low = limit_kernel(LimitMode::LowPass, 1000.0, 48_000, 101, FirWindow::Hamming);
        let high = limit_kernel(LimitMode::HighPass, 1000.0, 48_000, 101, FirWindow::Hamming);
        // the two sum to a unit impulse, so their responses sum to one.
        for (i, (l, h)) in low.iter().zip(high.iter()).enumerate() {
            let expected = if i == 50 { 1.0 } else { 0.0 };
            assert!((l + h - expected).abs() < 1e-12, "tap {i}");
        }
    }

    #[test]
    fn band_kernels_are_complements() {
        let pass = band_kernel(
            BandMode::BandPass,
            1000.0,
            4000.0,
            48_000,
            101,
            FirWindow::Hamming,
        );
        let reject = band_kernel(
            BandMode::BandReject,
            1000.0,
            4000.0,
            48_000,
            101,
            FirWindow::Hamming,
        );
        assert!(kernel_gain(&pass, 2500.0, 48_000) > 0.9);
        assert!(kernel_gain(&reject, 2500.0, 48_000) < 0.1);
        assert!(kernel_gain(&pass, 100.0, 48_000) < 0.1);
        assert!(kernel_gain(&reject, 100.0, 48_000) > 0.9);
    }

    #[test]
    fn chebyshev_lowpass_passes_dc_and_stops_nyquist() {
        let mut cascade = IirCascade::default();
        cascade.set_sections(
            cheb_limit_cascade(
                LimitMode::LowPass,
                1000.0,
                48_000,
                ChebSettings {
                    kind: 1,
                    poles: 4,
                    ripple_db: 0.5,
                    channels: 1,
                },
            ),
            1,
        );
        cascade.normalize_at(1.0, 0.0);
        assert!(
            (cascade.magnitude(1.0, 0.0) - 1.0).abs() < 1e-9,
            "unity at DC"
        );
        assert!(
            cascade.magnitude(-1.0, 0.0) < 1e-4,
            "the stop band reaches Nyquist"
        );
    }
}

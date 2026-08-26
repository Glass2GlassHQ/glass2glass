//! Pitch-preserving audio time stretcher (`scaletempo`). A playback rate other
//! than 1 is played by consuming the input faster or slower while the output
//! keeps the original pitch, so 2x speech stays speech instead of turning into
//! a chipmunk.
//!
//! WSOLA over fixed strides, the same structure as gst `scaletempo`: each
//! output stride is `stride` ms long and starts with an `overlap` fraction that
//! crossfades the previous stride's tail into the new input, and the new
//! input's position is picked by cross-correlating that tail against a `search`
//! ms window so the two halves line up in phase. The input pointer then slides
//! by `stride * rate` frames, which is where the length change comes from.
//!
//! The rate comes from the [`Segment`](g2g_core::segment::Segment)'s `rate`.
//! Following gst, the forwarded segment carries `applied_rate = rate` and
//! `rate = 1.0`, and the output is re-stamped onto the compressed timeline
//! (`start + (pts - start) / rate`), so downstream sees an ordinary rate-1
//! stream whose stream time still recovers the media position.
//!
//! A rate of 1 (within [`RATE_PASSTHROUGH_EPSILON`]) is a pure pass-through:
//! frames are forwarded with their bytes and timestamps untouched and nothing
//! is buffered. Reverse playback (a negative rate) is out of scope and is
//! forwarded the same untouched way.
//!
//! Interleaved `PcmS16Le` / `PcmF32Le` only, matching gst's format set minus
//! F64LE. Anything else is refused in negotiation so `audioconvert` is placed
//! ahead. The stretch itself runs on an f32 copy of the queue, so the s16 path
//! is not bit-identical to gst's fixed-point arithmetic. CPU-only, `no_std`
//! baseline.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use crate::audioconvert::{read_sample, sample_bytes, samples_to_ns, write_sample};

/// gst `scaletempo`'s default `stride`, the length of one output stride.
const DEFAULT_STRIDE_MS: u32 = 30;

/// gst's default `overlap`, the fraction of a stride that is crossfaded.
const DEFAULT_OVERLAP: f64 = 0.2;

/// gst's default `search`, how far the best-overlap search may shift the input.
const DEFAULT_SEARCH_MS: u32 = 14;

/// The same values as declared text, for `gst-inspect`.
const DEFAULT_STRIDE_TEXT: &str = "30";
const DEFAULT_OVERLAP_TEXT: &str = "0.2";
const DEFAULT_SEARCH_TEXT: &str = "14";

/// A rate this close to 1 is passed through instead of stretched: the length
/// change is under a sample per stride and the buffering would cost more than
/// it corrects.
const RATE_PASSTHROUGH_EPSILON: f64 = 1e-6;

const MILLISECONDS_PER_SECOND: f64 = 1000.0;

/// The PCM formats the stretcher runs on, gst `scaletempo`'s set minus F64LE
/// (g2g has no 64-bit PCM).
const STRETCH_FORMATS: [AudioFormat; 2] = [AudioFormat::PcmS16Le, AudioFormat::PcmF32Le];

/// # Example
///
/// ```no_run
/// use g2g_plugins::scaletempo::ScaleTempo;
///
/// let stretch = ScaleTempo::new().with_stride_ms(20);
/// ```
#[derive(Debug)]
pub struct ScaleTempo {
    stride_ms: u32,
    overlap: f64,
    search_ms: u32,
    /// Effective playback rate, from the last `Segment`.
    rate: f64,
    input: Option<(AudioFormat, u8, u32)>,
    configured: bool,
    last_caps: Option<Caps>,
    /// Set by a property or caps change; the window sizes are re-derived on the
    /// next frame.
    windows_stale: bool,
    stride_frames: usize,
    overlap_frames: usize,
    search_frames: usize,
    /// Frames copied straight out of the queue after the crossfaded overlap.
    standing_frames: usize,
    /// Crossfade ramp, one weight per overlap frame.
    blend_ramp: Vec<f32>,
    /// Correlation taper `i * (overlap_frames - i)`, one per overlap frame.
    correlation_taper: Vec<f32>,
    /// Tail of the previous output stride, crossfaded into the next one.
    overlap_tail: Vec<f32>,
    /// The tapered overlap tail, the fixed half of the cross-correlation.
    correlation_reference: Vec<f32>,
    /// Input samples waiting to be turned into strides.
    queue: Vec<f32>,
    queue_capacity: usize,
    /// Input samples to drop before the queue refills again.
    samples_to_slide: usize,
    /// Fractional frames of the scaled stride carried into the next slide.
    stride_error: f64,
    /// Output timeline: pts of the first output sample, and the sample frames
    /// emitted since it.
    base_pts: Option<u64>,
    emitted_frames: u64,
    segment_start: u64,
    sequence: u64,
}

impl Default for ScaleTempo {
    fn default() -> Self {
        Self::new()
    }
}

impl ScaleTempo {
    /// gst's defaults: 30 ms strides, 20 % overlap, 14 ms of search.
    pub fn new() -> Self {
        Self {
            stride_ms: DEFAULT_STRIDE_MS,
            overlap: DEFAULT_OVERLAP,
            search_ms: DEFAULT_SEARCH_MS,
            rate: 1.0,
            input: None,
            configured: false,
            last_caps: None,
            windows_stale: true,
            stride_frames: 0,
            overlap_frames: 0,
            search_frames: 0,
            standing_frames: 0,
            blend_ramp: Vec::new(),
            correlation_taper: Vec::new(),
            overlap_tail: Vec::new(),
            correlation_reference: Vec::new(),
            queue: Vec::new(),
            queue_capacity: 0,
            samples_to_slide: 0,
            stride_error: 0.0,
            base_pts: None,
            emitted_frames: 0,
            segment_start: 0,
            sequence: 0,
        }
    }

    pub fn with_stride_ms(mut self, stride_ms: u32) -> Self {
        self.stride_ms = stride_ms;
        self.windows_stale = true;
        self
    }

    pub fn with_overlap(mut self, overlap: f64) -> Self {
        self.overlap = overlap;
        self.windows_stale = true;
        self
    }

    pub fn with_search_ms(mut self, search_ms: u32) -> Self {
        self.search_ms = search_ms;
        self.windows_stale = true;
        self
    }

    fn accept_input(&self, caps: &Caps) -> Result<(AudioFormat, u8, u32), G2gError> {
        let Caps::Audio {
            format,
            channels,
            sample_rate,
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !STRETCH_FORMATS.contains(format)
            || *channels == ANY_CHANNELS
            || *sample_rate == ANY_SAMPLE_RATE
        {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *channels, *sample_rate))
    }

    /// Whether the rate is forward and far enough from 1 to be worth stretching.
    fn stretching(&self) -> bool {
        let near_one = self.rate > 1.0 - RATE_PASSTHROUGH_EPSILON
            && self.rate < 1.0 + RATE_PASSTHROUGH_EPSILON;
        self.rate > 0.0 && !near_one
    }

    /// A media timestamp on the compressed output timeline.
    fn map_pts(&self, pts: u64) -> u64 {
        let Some(span) = pts.checked_sub(self.segment_start) else {
            return pts;
        };
        self.segment_start
            .saturating_add((span as f64 / self.rate) as u64)
    }

    /// Drop everything buffered and restart the output timeline.
    fn reset_stream(&mut self) {
        self.queue.clear();
        self.overlap_tail.fill(0.0);
        self.samples_to_slide = 0;
        self.stride_error = 0.0;
        self.base_pts = None;
        self.emitted_frames = 0;
    }

    fn derive_windows(&mut self, channels: usize, sample_rate: u32) {
        let frames_per_ms = sample_rate as f64 / MILLISECONDS_PER_SECOND;
        self.stride_frames = ((self.stride_ms as f64 * frames_per_ms) as usize).max(1);
        self.overlap_frames = (self.stride_frames as f64 * self.overlap.clamp(0.0, 1.0)) as usize;
        self.standing_frames = self.stride_frames - self.overlap_frames;
        // one overlap frame leaves nothing to correlate over.
        self.search_frames = if self.overlap_frames <= 1 {
            0
        } else {
            (self.search_ms as f64 * frames_per_ms) as usize
        };

        let overlap_frames = self.overlap_frames as f32;
        self.blend_ramp = (0..self.overlap_frames)
            .map(|i| i as f32 / overlap_frames)
            .collect();
        self.correlation_taper = (0..self.overlap_frames)
            .map(|i| i as f32 * (overlap_frames - i as f32))
            .collect();
        let overlap_samples = self.overlap_frames * channels;
        self.overlap_tail = vec![0.0; overlap_samples];
        self.correlation_reference = vec![0.0; overlap_samples];
        self.queue_capacity =
            (self.search_frames + self.stride_frames + self.overlap_frames) * channels;
        self.queue = Vec::with_capacity(self.queue_capacity);
        self.samples_to_slide = 0;
        self.stride_error = 0.0;
        self.windows_stale = false;
    }

    /// Slide the queue forward by the pending amount and top it up from
    /// `input`, returning how much of `input` was taken from `offset` on.
    fn fill_queue(&mut self, input: &[f32], offset: usize) -> usize {
        let start = offset;
        let mut offset = offset;
        if self.samples_to_slide > 0 {
            if self.samples_to_slide < self.queue.len() {
                self.queue.drain(..self.samples_to_slide);
                self.samples_to_slide = 0;
            } else {
                self.samples_to_slide -= self.queue.len();
                self.queue.clear();
                let skipped = self.samples_to_slide.min(input.len() - offset);
                self.samples_to_slide -= skipped;
                offset += skipped;
            }
        }
        let room = self.queue_capacity.saturating_sub(self.queue.len());
        let taken = room.min(input.len() - offset);
        self.queue.extend_from_slice(&input[offset..offset + taken]);
        offset + taken - start
    }

    /// Frame offset into the queue whose samples line up best with the previous
    /// stride's tail.
    fn best_overlap_offset(&mut self, search_frames: usize, channels: usize) -> usize {
        // the taper is zero on frame 0, so the correlation starts a frame in.
        let overlap_samples = self.overlap_frames * channels;
        for i in channels..overlap_samples {
            self.correlation_reference[i] =
                self.correlation_taper[i / channels] * self.overlap_tail[i];
        }
        let mut best_offset = 0;
        let mut best_score = f64::NEG_INFINITY;
        for offset in 0..search_frames {
            let base = offset * channels;
            let mut score = 0.0f64;
            for i in channels..overlap_samples {
                score += (self.correlation_reference[i] * self.queue[base + i]) as f64;
            }
            if score > best_score {
                best_score = score;
                best_offset = offset;
            }
        }
        best_offset
    }

    /// Append one output stride and set up the next input slide.
    fn emit_stride(&mut self, search_frames: usize, channels: usize, out: &mut Vec<f32>) {
        let base = if search_frames > 0 {
            self.best_overlap_offset(search_frames, channels) * channels
        } else {
            0
        };
        let overlap_samples = self.overlap_frames * channels;
        for i in 0..overlap_samples {
            let tail = self.overlap_tail[i];
            out.push(tail + self.blend_ramp[i / channels] * (self.queue[base + i] - tail));
        }
        let standing = base + overlap_samples;
        out.extend_from_slice(&self.queue[standing..standing + self.standing_frames * channels]);
        let next_tail = base + self.stride_frames * channels;
        self.overlap_tail
            .copy_from_slice(&self.queue[next_tail..next_tail + overlap_samples]);

        let slide = self.stride_frames as f64 * self.rate + self.stride_error;
        let whole = slide as usize;
        self.stride_error = slide - whole as f64;
        self.samples_to_slide = whole * channels;
    }

    /// Turn as much of `input` (plus whatever was queued) into whole strides as
    /// the buffer allows. At EOS the search window shrinks to what is left
    /// instead of waiting for a full queue.
    fn run_strides(&mut self, input: &[f32], channels: usize, at_eos: bool) -> Vec<f32> {
        let mut out = Vec::new();
        let mut consumed = self.fill_queue(input, 0);
        let minimum_frames = self.stride_frames + self.overlap_frames;
        loop {
            let queued_frames = self.queue.len() / channels;
            if queued_frames < minimum_frames {
                break;
            }
            let search_frames = if self.queue.len() >= self.queue_capacity {
                self.search_frames
            } else if at_eos {
                self.search_frames.min(queued_frames - minimum_frames)
            } else {
                break;
            };
            self.emit_stride(search_frames, channels, &mut out);
            consumed += self.fill_queue(input, consumed);
        }
        out
    }

    async fn push_samples(
        &mut self,
        samples: Vec<f32>,
        format: AudioFormat,
        channels: usize,
        sample_rate: u32,
        source: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        if samples.is_empty() {
            return Ok(());
        }
        let mut bytes = Vec::with_capacity(samples.len() * sample_bytes(format));
        for sample in &samples {
            write_sample(&mut bytes, *sample, format);
        }
        let frames = (samples.len() / channels) as u64;
        let base = self.base_pts.unwrap_or(0);
        let pts = base.saturating_add(samples_to_ns(self.emitted_frames, sample_rate));
        let end = base.saturating_add(samples_to_ns(self.emitted_frames + frames, sample_rate));
        self.emitted_frames += frames;
        let timing = FrameTiming {
            pts_ns: pts,
            dts_ns: pts,
            duration_ns: end - pts,
            capture_ns: source.capture_ns,
            arrival_ns: source.arrival_ns,
            keyframe: source.keyframe,
        };
        let domain = MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice()));
        let frame = Frame::new(domain, timing, self.sequence);
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

impl AsyncElement for ScaleTempo {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Scaletempo",
            "Filter/Effect/Rate/Audio",
            "Stretches audio to the playback rate without changing its pitch",
            "g2g",
        )
    }

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// Retiming only: the output caps equal the input.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio { format, .. } if STRETCH_FORMATS.contains(format) => {
                CapsSet::one(input.clone())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.input = Some(self.accept_input(absolute_caps)?);
        self.windows_stale = true;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let (format, channel_count, sample_rate) = self.input.ok_or(G2gError::NotConfigured)?;
            let channels = channel_count as usize;
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let bytes_per_frame = sample_bytes(format) * channels;
                    let bytes = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    if bytes_per_frame == 0 || bytes.len() % bytes_per_frame != 0 {
                        return Err(G2gError::CapsMismatch);
                    }

                    let caps = Caps::Audio {
                        format,
                        channels: channel_count,
                        sample_rate,
                    };
                    if self.last_caps.as_ref() != Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_caps = Some(caps);
                    }

                    if !self.stretching() {
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        return Ok(());
                    }
                    if self.windows_stale {
                        self.derive_windows(channels, sample_rate);
                    }
                    if self.base_pts.is_none() {
                        self.base_pts = Some(self.map_pts(frame.timing.pts().unwrap_or(0)));
                    }
                    let width = sample_bytes(format);
                    let input: Vec<f32> = bytes
                        .chunks_exact(width)
                        .map(|at| read_sample(at, format))
                        .collect();
                    let source = frame.timing;
                    let stretched = self.run_strides(&input, channels, false);
                    self.push_samples(stretched, format, channels, sample_rate, source, out)
                        .await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    let new_input = self.accept_input(&caps)?;
                    if self.input != Some(new_input) {
                        self.input = Some(new_input);
                        self.windows_stale = true;
                        self.reset_stream();
                    }
                }
                PipelinePacket::Segment(mut segment) => {
                    self.rate = segment.rate;
                    self.segment_start = segment.start;
                    self.reset_stream();
                    if self.stretching() {
                        segment.applied_rate = segment.rate;
                        segment.rate = 1.0;
                        segment.stop = segment.stop.map(|stop| self.map_pts(stop));
                        segment.position = self.map_pts(segment.position);
                    }
                    out.push(PipelinePacket::Segment(segment)).await?;
                }
                PipelinePacket::Flush => {
                    self.reset_stream();
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // the runner emits the end itself; this drains what is left.
                PipelinePacket::Eos => {
                    if self.stretching() && !self.windows_stale {
                        let tail = self.run_strides(&[], channels, true);
                        let source = FrameTiming::default();
                        self.push_samples(tail, format, channels, sample_rate, source, out)
                            .await?;
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SCALETEMPO_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "stride" => {
                self.stride_ms = value.as_uint().ok_or(PropError::Type)? as u32;
                self.windows_stale = true;
            }
            "overlap" => {
                self.overlap = value.as_double().ok_or(PropError::Type)?;
                self.windows_stale = true;
            }
            "search" => {
                self.search_ms = value.as_uint().ok_or(PropError::Type)? as u32;
                self.windows_stale = true;
            }
            "rate" => return Err(PropError::Value),
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "stride" => Some(PropValue::Uint(self.stride_ms as u64)),
            "overlap" => Some(PropValue::Double(self.overlap)),
            "search" => Some(PropValue::Uint(self.search_ms as u64)),
            "rate" => Some(PropValue::Double(self.rate)),
            _ => None,
        }
    }
}

/// `ScaleTempo`'s properties (M1075): gst `scaletempo`'s three window settings
/// plus its read-only current rate.
static SCALETEMPO_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "stride",
        PropKind::Uint,
        "length in milliseconds to output each stride",
    )
    .with_default(DEFAULT_STRIDE_TEXT),
    PropertySpec::new(
        "overlap",
        PropKind::Double,
        "fraction of a stride to overlap",
    )
    .with_default(DEFAULT_OVERLAP_TEXT),
    PropertySpec::new(
        "search",
        PropKind::Uint,
        "length in milliseconds to search for the best overlap position",
    )
    .with_default(DEFAULT_SEARCH_TEXT),
    PropertySpec::new("rate", PropKind::Double, "current playback rate").read_only(),
];

impl PadTemplates for ScaleTempo {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |format| Caps::Audio {
            format,
            channels: ANY_CHANNELS,
            sample_rate: ANY_SAMPLE_RATE,
        };
        let set = CapsSet::from_alternatives(STRETCH_FORMATS.map(pcm).to_vec());
        Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    #[test]
    fn declared_defaults_match_the_constants() {
        let element = ScaleTempo::new();
        assert_eq!(
            element.get_property("stride"),
            Some(PropValue::Uint(DEFAULT_STRIDE_MS as u64))
        );
        assert_eq!(
            element.get_property("search"),
            Some(PropValue::Uint(DEFAULT_SEARCH_MS as u64))
        );
        assert_eq!(
            element.get_property("overlap"),
            Some(PropValue::Double(DEFAULT_OVERLAP))
        );
        for (name, text) in [
            ("stride", DEFAULT_STRIDE_TEXT),
            ("overlap", DEFAULT_OVERLAP_TEXT),
            ("search", DEFAULT_SEARCH_TEXT),
        ] {
            let spec = SCALETEMPO_PROPS
                .iter()
                .find(|s| s.name == name)
                .expect("declared");
            assert_eq!(spec.default, Some(text), "{name}");
        }
    }

    #[test]
    fn only_a_rate_away_from_one_stretches() {
        let mut element = ScaleTempo::new();
        assert!(!element.stretching(), "rate 1 passes through");
        element.rate = 1.0 + RATE_PASSTHROUGH_EPSILON / 2.0;
        assert!(!element.stretching(), "inside the epsilon");
        element.rate = 2.0;
        assert!(element.stretching());
        element.rate = -2.0;
        assert!(!element.stretching(), "reverse is out of scope");
    }

    #[test]
    fn windows_come_from_the_millisecond_settings() {
        let mut element = ScaleTempo::new();
        element.derive_windows(2, RATE);
        let per_ms = RATE as usize / 1000;
        assert_eq!(element.stride_frames, DEFAULT_STRIDE_MS as usize * per_ms);
        assert_eq!(
            element.overlap_frames,
            (element.stride_frames as f64 * DEFAULT_OVERLAP) as usize
        );
        assert_eq!(element.search_frames, DEFAULT_SEARCH_MS as usize * per_ms);
        assert_eq!(
            element.standing_frames + element.overlap_frames,
            element.stride_frames
        );
        assert_eq!(
            element.queue_capacity,
            (element.search_frames + element.stride_frames + element.overlap_frames) * 2
        );
    }

    #[test]
    fn configure_rejects_the_formats_audioconvert_has_to_handle() {
        let mut element = ScaleTempo::new();
        for format in [AudioFormat::PcmU8, AudioFormat::PcmS24Le, AudioFormat::Opus] {
            let caps = Caps::Audio {
                format,
                channels: 2,
                sample_rate: RATE,
            };
            assert_eq!(
                element.configure_pipeline(&caps).unwrap_err(),
                G2gError::CapsMismatch,
                "{format:?}"
            );
        }
        for format in STRETCH_FORMATS {
            let caps = Caps::Audio {
                format,
                channels: 2,
                sample_rate: RATE,
            };
            assert!(element.configure_pipeline(&caps).is_ok(), "{format:?}");
        }
    }
}

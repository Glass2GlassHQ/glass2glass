//! Buffer re-framer (`audiobuffersplit`). Re-cuts an interleaved PCM stream
//! into buffers of one fixed duration and re-stamps them off a sample counter,
//! so a stream arriving in ragged chunks leaves in even ones. Preserves format,
//! channel count, and sample rate, and never touches a sample value. CPU-only,
//! `no_std` baseline.
//!
//! Matches GStreamer's `audiobuffersplit`. `output-buffer-duration` is a
//! fraction of a second; when `rate * numerator` does not divide the
//! denominator the remainder accumulates and every so often a buffer carries
//! one extra sample, so the average duration is exact. Timestamps run
//! `resync pts + samples * 1e9 / rate` from the last resync rather than
//! accumulating, so a rate that does not divide a second (44100) never drifts.
//!
//! A timestamp that deviates from the expected one by more than
//! `alignment-threshold`, and keeps deviating for `discont-wait`, is a
//! discontinuity: the pending buffer is flushed (or discarded, under
//! `strict-buffer-size`) and the timestamp grid restarts. Under `gapless` the
//! gap is filled with silence, or the overlap trimmed off the incoming buffer,
//! instead, unless the gap exceeds `max-silence-time`.
//!
//! Alignment is measured against each frame's pts, not its running time: g2g
//! forwards the [`Segment`](g2g_core::segment::Segment) unchanged, and the two
//! coincide for the ordinary rate-1 segment starting at 0. Reverse playback is
//! out of scope.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use crate::audioconvert::{ns_to_samples, pcm_formats, sample_bytes, samples_to_ns, silence_byte};

/// gst's default `output-buffer-duration`, 20 ms.
const DEFAULT_OUTPUT_BUFFER_DURATION: (i32, i32) = (1, 50);
const DEFAULT_OUTPUT_BUFFER_DURATION_TEXT: &str = "1/50";

/// gst's default `alignment-threshold`, 40 ms in ns.
const DEFAULT_ALIGNMENT_THRESHOLD_NS: u64 = 40_000_000;
const DEFAULT_ALIGNMENT_THRESHOLD_TEXT: &str = "40000000";

/// gst's default `discont-wait`, one second in ns.
const DEFAULT_DISCONT_WAIT_NS: u64 = 1_000_000_000;
const DEFAULT_DISCONT_WAIT_TEXT: &str = "1000000000";

const DEFAULT_OUTPUT_BUFFER_SIZE: u64 = 0;
const OUTPUT_BUFFER_SIZE_MAX: u64 = i32::MAX as u64;
const DEFAULT_STRICT_BUFFER_SIZE: bool = false;
const DEFAULT_GAPLESS: bool = false;
const DEFAULT_MAX_SILENCE_TIME_NS: u64 = 0;

const DEFAULT_OUTPUT_BUFFER_SIZE_TEXT: &str = "0";
const DEFAULT_STRICT_BUFFER_SIZE_TEXT: &str = "false";
const DEFAULT_GAPLESS_TEXT: &str = "false";
const DEFAULT_MAX_SILENCE_TIME_TEXT: &str = "0";

/// Sample frames one silence chunk covers when a gapless gap is filled, so a
/// long gap costs a bounded allocation per push. gst uses one second.
const SILENCE_CHUNK_SECONDS: u64 = 1;

/// The timestamp-alignment state gst keeps in a `GstAudioStreamAlign`: where
/// the next buffer is expected, and since when it has been off.
#[derive(Debug, Default)]
struct StreamAlign {
    /// Offset (in sample frames) the next buffer should start at.
    next_offset: Option<u64>,
    /// Expected time of the first deviating buffer, while `discont-wait` runs.
    discont_time: Option<u64>,
}

impl StreamAlign {
    fn reset(&mut self) {
        self.next_offset = None;
        self.discont_time = None;
    }

    /// Whether this buffer starts a discontinuity, and advance the expectation.
    fn process(
        &mut self,
        forced: bool,
        timestamp_ns: u64,
        frames: u64,
        rate: u32,
        alignment_threshold_ns: u64,
        discont_wait_ns: u64,
    ) -> bool {
        let start_offset = ns_to_samples(timestamp_ns, rate);
        let end_offset = start_offset + frames;
        let mut discont = forced;

        match self.next_offset {
            None => discont = true,
            Some(next) if !discont => {
                let diff = start_offset.abs_diff(next);
                let expected_time = samples_to_ns(next, rate);
                let max_sample_diff = ns_to_samples(alignment_threshold_ns, rate).max(1);
                if diff >= max_sample_diff {
                    if discont_wait_ns == 0 {
                        discont = true;
                    } else {
                        match self.discont_time {
                            None => {
                                if expected_time.abs_diff(timestamp_ns) >= discont_wait_ns {
                                    discont = true;
                                } else {
                                    self.discont_time = Some(expected_time);
                                }
                            }
                            Some(since) => {
                                if timestamp_ns.abs_diff(since) >= discont_wait_ns {
                                    discont = true;
                                    self.discont_time = None;
                                }
                            }
                        }
                    }
                } else {
                    self.discont_time = None;
                }
            }
            Some(_) => {}
        }

        if discont {
            self.next_offset = Some(end_offset);
            self.discont_time = None;
        } else {
            self.next_offset = self.next_offset.map(|next| next + frames);
        }
        discont
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::audiobuffersplit::AudioBufferSplit;
///
/// // 10 ms buffers.
/// let split = AudioBufferSplit::new().with_output_buffer_duration(1, 100);
/// ```
#[derive(Debug)]
pub struct AudioBufferSplit {
    /// `output-buffer-duration` as a fraction of a second.
    duration_numerator: i32,
    duration_denominator: i32,
    output_buffer_size: u64,
    strict_buffer_size: bool,
    gapless: bool,
    alignment_threshold_ns: u64,
    discont_wait_ns: u64,
    max_silence_time_ns: u64,

    input: Option<(AudioFormat, u8, u32)>,
    configured: bool,
    last_caps: Option<Caps>,

    /// Whole sample frames one output buffer holds, and the leftover fraction
    /// of a frame per buffer that accumulates into an extra one.
    samples_per_buffer: u64,
    error_per_buffer: u64,
    accumulated_error: u64,

    /// Bytes not yet cut into an output buffer.
    adapter: Vec<u8>,
    /// Output frames emitted since the last resync, `None` before the first.
    current_offset: Option<u64>,
    /// Timestamp the current run of output buffers is measured from.
    resync_pts: u64,
    align: StreamAlign,
    /// Frames still to be trimmed off the incoming data in gapless mode.
    drop_frames: u64,
    emitted: u64,
}

impl Default for AudioBufferSplit {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioBufferSplit {
    pub fn new() -> Self {
        Self {
            duration_numerator: DEFAULT_OUTPUT_BUFFER_DURATION.0,
            duration_denominator: DEFAULT_OUTPUT_BUFFER_DURATION.1,
            output_buffer_size: DEFAULT_OUTPUT_BUFFER_SIZE,
            strict_buffer_size: DEFAULT_STRICT_BUFFER_SIZE,
            gapless: DEFAULT_GAPLESS,
            alignment_threshold_ns: DEFAULT_ALIGNMENT_THRESHOLD_NS,
            discont_wait_ns: DEFAULT_DISCONT_WAIT_NS,
            max_silence_time_ns: DEFAULT_MAX_SILENCE_TIME_NS,
            input: None,
            configured: false,
            last_caps: None,
            samples_per_buffer: 0,
            error_per_buffer: 0,
            accumulated_error: 0,
            adapter: Vec::new(),
            current_offset: None,
            resync_pts: 0,
            align: StreamAlign::default(),
            drop_frames: 0,
            emitted: 0,
        }
    }

    pub fn with_output_buffer_duration(mut self, numerator: i32, denominator: i32) -> Self {
        if numerator > 0 && denominator > 0 {
            self.duration_numerator = numerator;
            self.duration_denominator = denominator;
            self.update_samples_per_buffer();
        }
        self
    }

    pub fn with_output_buffer_size(mut self, bytes: u64) -> Self {
        self.output_buffer_size = bytes.min(OUTPUT_BUFFER_SIZE_MAX);
        self.update_samples_per_buffer();
        self
    }

    pub fn with_strict_buffer_size(mut self, strict: bool) -> Self {
        self.strict_buffer_size = strict;
        self
    }

    pub fn with_gapless(mut self, gapless: bool) -> Self {
        self.gapless = gapless;
        self
    }

    pub fn with_alignment_threshold(mut self, threshold_ns: u64) -> Self {
        self.alignment_threshold_ns = threshold_ns;
        self
    }

    pub fn with_discont_wait(mut self, wait_ns: u64) -> Self {
        self.discont_wait_ns = wait_ns;
        self
    }

    pub fn with_max_silence_time(mut self, time_ns: u64) -> Self {
        self.max_silence_time_ns = time_ns;
        self
    }

    /// Frames one output buffer holds, once the rate is known. Zero until then,
    /// and zero if the settings ask for less than one frame per buffer.
    pub fn samples_per_buffer(&self) -> u64 {
        self.samples_per_buffer
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
        if !pcm_formats().contains(format)
            || *channels == 0
            || *channels == ANY_CHANNELS
            || *sample_rate == 0
            || *sample_rate == ANY_SAMPLE_RATE
        {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *channels, *sample_rate))
    }

    fn bytes_per_frame(&self) -> usize {
        match self.input {
            Some((format, channels, _)) => sample_bytes(format) * channels as usize,
            None => 0,
        }
    }

    /// The buffer duration in effect: `output-buffer-size` when it is set (it
    /// names a frame count at the stream's rate), else the fraction property.
    fn effective_duration(&self) -> (u64, u64) {
        let bytes_per_frame = self.bytes_per_frame();
        let Some((_, _, rate)) = self.input else {
            return (
                self.duration_numerator as u64,
                self.duration_denominator as u64,
            );
        };
        if self.output_buffer_size > 0 && bytes_per_frame > 0 {
            return (
                self.output_buffer_size / bytes_per_frame as u64,
                rate as u64,
            );
        }
        (
            self.duration_numerator as u64,
            self.duration_denominator as u64,
        )
    }

    fn update_samples_per_buffer(&mut self) {
        let Some((_, _, rate)) = self.input else {
            self.samples_per_buffer = 0;
            return;
        };
        let (numerator, denominator) = self.effective_duration();
        if denominator == 0 {
            self.samples_per_buffer = 0;
            return;
        }
        let frames = rate as u64 * numerator;
        self.samples_per_buffer = frames / denominator;
        self.error_per_buffer = frames % denominator;
        self.accumulated_error = 0;
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let input = self.accept_input(caps)?;
        if self.input != Some(input) {
            self.input = Some(input);
            self.reset_stream();
        }
        self.update_samples_per_buffer();
        if self.samples_per_buffer == 0 {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(())
    }

    fn reset_stream(&mut self) {
        self.adapter.clear();
        self.current_offset = None;
        self.resync_pts = 0;
        self.align.reset();
        self.drop_frames = 0;
        self.accumulated_error = 0;
    }

    /// Bytes the next output buffer takes: the whole frames per buffer, plus
    /// one more frame whenever the accumulated remainder has reached a frame.
    fn next_buffer_bytes(&self, bytes_per_frame: usize) -> usize {
        let (_, denominator) = self.effective_duration();
        let mut frames = self.samples_per_buffer;
        if denominator > 0 && self.error_per_buffer + self.accumulated_error >= denominator {
            frames += 1;
        }
        frames as usize * bytes_per_frame
    }

    /// Cut whole buffers out of the adapter. `force` also emits a final short
    /// one from whatever is left.
    async fn emit_buffers(
        &mut self,
        force: bool,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let Some((format, channels, rate)) = self.input else {
            return Err(G2gError::NotConfigured);
        };
        let bytes_per_frame = sample_bytes(format) * channels as usize;
        if bytes_per_frame == 0 {
            return Err(G2gError::CapsMismatch);
        }
        let caps = Caps::Audio {
            format,
            channels,
            sample_rate: rate,
        };
        let (_, denominator) = self.effective_duration();

        let mut size = self.next_buffer_bytes(bytes_per_frame);
        while (size > 0 && self.adapter.len() >= size) || (force && !self.adapter.is_empty()) {
            let take = size.min(self.adapter.len());
            if take == 0 {
                break;
            }
            let bytes: Vec<u8> = self.adapter.drain(..take).collect();
            let frames = (take / bytes_per_frame) as u64;
            let offset = self.current_offset.unwrap_or(0);
            let pts = self.resync_pts.saturating_add(samples_to_ns(offset, rate));
            let end = self
                .resync_pts
                .saturating_add(samples_to_ns(offset + frames, rate));
            self.current_offset = Some(offset + frames);
            if denominator > 0 {
                self.accumulated_error =
                    (self.accumulated_error + self.error_per_buffer) % denominator;
            }

            if self.last_caps.as_ref() != Some(&caps) {
                out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                self.last_caps = Some(caps.clone());
            }
            let timing = FrameTiming {
                pts_ns: pts,
                dts_ns: pts,
                duration_ns: end.saturating_sub(pts),
                ..Default::default()
            };
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                timing,
                self.emitted,
            );
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;

            size = self.next_buffer_bytes(bytes_per_frame);
        }
        Ok(())
    }

    /// Realign the output grid to an incoming buffer. In gapless mode a gap is
    /// filled with silence and an overlap is trimmed off the input; otherwise
    /// the grid restarts at this buffer's timestamp.
    async fn handle_discont(
        &mut self,
        pts_ns: u64,
        input_frames: u64,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let Some((format, channels, rate)) = self.input else {
            return Err(G2gError::NotConfigured);
        };
        let bytes_per_frame = sample_bytes(format) * channels as usize;
        let available_frames = (self.adapter.len() / bytes_per_frame.max(1)) as u64;

        let forced = self.current_offset.is_none();
        let mut discont = self.align.process(
            forced,
            pts_ns,
            input_frames,
            rate,
            self.alignment_threshold_ns,
            self.discont_wait_ns,
        );
        if !discont {
            return Ok(());
        }
        self.drop_frames = 0;

        if let (true, Some(current)) = (self.gapless, self.current_offset) {
            let received = current + available_frames;
            if pts_ns < self.resync_pts {
                // The stream jumped back before the grid's origin: everything
                // buffered plus the rewind is stale.
                self.drop_frames = received + ns_to_samples(self.resync_pts - pts_ns, rate);
                discont = false;
            } else {
                let new_offset = ns_to_samples(pts_ns - self.resync_pts, rate);
                if new_offset > received {
                    let silence_frames = new_offset - received;
                    let silence_ns = samples_to_ns(silence_frames, rate);
                    let too_long =
                        self.max_silence_time_ns != 0 && silence_ns > self.max_silence_time_ns;
                    if !too_long {
                        self.push_silence(silence_frames, bytes_per_frame, rate, format, out)
                            .await?;
                        discont = false;
                    }
                } else if new_offset < received {
                    self.drop_frames = received - new_offset;
                    discont = false;
                }
            }
        }

        if discont {
            if self.strict_buffer_size {
                self.adapter.clear();
            } else {
                self.emit_buffers(true, out).await?;
            }
            self.current_offset = Some(0);
            self.accumulated_error = 0;
            self.resync_pts = pts_ns;
        }
        Ok(())
    }

    /// Fill a gapless gap, in bounded chunks so a long gap does not allocate
    /// the whole silence at once.
    async fn push_silence(
        &mut self,
        mut frames: u64,
        bytes_per_frame: usize,
        rate: u32,
        format: AudioFormat,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let chunk_frames = rate as u64 * SILENCE_CHUNK_SECONDS;
        while frames > 0 {
            let take = frames.min(chunk_frames);
            let filled = self.adapter.len() + take as usize * bytes_per_frame;
            self.adapter.resize(filled, silence_byte(format));
            self.emit_buffers(false, out).await?;
            frames -= take;
        }
        Ok(())
    }
}

impl AsyncElement for AudioBufferSplit {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// Re-framing only: the output caps equal the input.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio { format, .. } if pcm_formats().contains(format) => {
                CapsSet::one(input.clone())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configure(absolute_caps)?;
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let bytes_per_frame = self.bytes_per_frame();
                    let bytes = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    if bytes_per_frame == 0 || bytes.len() % bytes_per_frame != 0 {
                        return Err(G2gError::CapsMismatch);
                    }
                    let input_frames = (bytes.len() / bytes_per_frame) as u64;
                    self.handle_discont(frame.timing.pts_ns, input_frames, out)
                        .await?;

                    let dropped = self.drop_frames.min(input_frames);
                    self.drop_frames -= dropped;
                    let from = dropped as usize * bytes_per_frame;
                    if from < bytes.len() {
                        self.adapter.extend_from_slice(&bytes[from..]);
                    }
                    self.emit_buffers(false, out).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    self.configure(&c)?;
                }
                PipelinePacket::Flush => {
                    self.reset_stream();
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // The runner emits the end itself; this flushes the tail, which
                // `strict-buffer-size` discards instead.
                PipelinePacket::Eos => {
                    if self.strict_buffer_size {
                        self.adapter.clear();
                    } else {
                        self.emit_buffers(true, out).await?;
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
        AUDIOBUFFERSPLIT_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Audio Buffer Split",
            "Audio/Filter",
            "Splits raw audio buffers into equal sized chunks",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "output-buffer-duration" => {
                let (numerator, denominator) = value.as_fraction().ok_or(PropError::Type)?;
                if numerator <= 0 || denominator <= 0 {
                    return Err(PropError::Value);
                }
                self.duration_numerator = numerator;
                self.duration_denominator = denominator;
                self.update_samples_per_buffer();
            }
            "output-buffer-size" => {
                let bytes = value.as_uint().ok_or(PropError::Type)?;
                if bytes > OUTPUT_BUFFER_SIZE_MAX {
                    return Err(PropError::Value);
                }
                self.output_buffer_size = bytes;
                self.update_samples_per_buffer();
            }
            "strict-buffer-size" => {
                self.strict_buffer_size = value.as_bool().ok_or(PropError::Type)?
            }
            "gapless" => self.gapless = value.as_bool().ok_or(PropError::Type)?,
            "alignment-threshold" => {
                self.alignment_threshold_ns = value.as_uint().ok_or(PropError::Type)?
            }
            "discont-wait" => self.discont_wait_ns = value.as_uint().ok_or(PropError::Type)?,
            "max-silence-time" => {
                self.max_silence_time_ns = value.as_uint().ok_or(PropError::Type)?
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "output-buffer-duration" => Some(PropValue::Fraction(
                self.duration_numerator,
                self.duration_denominator,
            )),
            "output-buffer-size" => Some(PropValue::Uint(self.output_buffer_size)),
            "strict-buffer-size" => Some(PropValue::Bool(self.strict_buffer_size)),
            "gapless" => Some(PropValue::Bool(self.gapless)),
            "alignment-threshold" => Some(PropValue::Uint(self.alignment_threshold_ns)),
            "discont-wait" => Some(PropValue::Uint(self.discont_wait_ns)),
            "max-silence-time" => Some(PropValue::Uint(self.max_silence_time_ns)),
            _ => None,
        }
    }
}

static AUDIOBUFFERSPLIT_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "output-buffer-duration",
        PropKind::Fraction,
        "output block size in seconds",
    )
    .with_range("1/2147483647", "2147483647/1")
    .with_default(DEFAULT_OUTPUT_BUFFER_DURATION_TEXT),
    PropertySpec::new(
        "output-buffer-size",
        PropKind::Uint,
        "output block size in bytes, takes precedence over the duration when non-zero",
    )
    .with_range("0", "2147483647")
    .with_default(DEFAULT_OUTPUT_BUFFER_SIZE_TEXT),
    PropertySpec::new(
        "strict-buffer-size",
        PropKind::Bool,
        "discard the last samples at Eos or a discont if they are too few to fill a buffer",
    )
    .with_default(DEFAULT_STRICT_BUFFER_SIZE_TEXT),
    PropertySpec::new(
        "gapless",
        PropKind::Bool,
        "insert silence / drop samples instead of restarting the timestamp grid",
    )
    .with_default(DEFAULT_GAPLESS_TEXT),
    PropertySpec::new(
        "alignment-threshold",
        PropKind::Uint,
        "timestamp alignment threshold in nanoseconds",
    )
    .with_default(DEFAULT_ALIGNMENT_THRESHOLD_TEXT),
    PropertySpec::new(
        "discont-wait",
        PropKind::Uint,
        "nanoseconds a timestamp has to keep deviating before it counts as a discontinuity",
    )
    .with_default(DEFAULT_DISCONT_WAIT_TEXT),
    PropertySpec::new(
        "max-silence-time",
        PropKind::Uint,
        "do not insert silence in gapless mode if the gap exceeds this period in ns (0 disables)",
    )
    .with_default(DEFAULT_MAX_SILENCE_TIME_TEXT),
];

impl PadTemplates for AudioBufferSplit {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |format| Caps::Audio {
            format,
            channels: ANY_CHANNELS,
            sample_rate: ANY_SAMPLE_RATE,
        };
        let set = CapsSet::from_alternatives(pcm_formats().iter().copied().map(pcm).collect());
        vec![PadTemplate::sink(set.clone()), PadTemplate::source(set)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    fn caps(rate: u32) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: rate,
        }
    }

    #[test]
    fn declared_defaults_match_the_constants() {
        let element = AudioBufferSplit::new();
        assert_eq!(
            element.get_property("output-buffer-duration"),
            Some(PropValue::Fraction(
                DEFAULT_OUTPUT_BUFFER_DURATION.0,
                DEFAULT_OUTPUT_BUFFER_DURATION.1
            ))
        );
        assert_eq!(
            element.get_property("alignment-threshold"),
            Some(PropValue::Uint(DEFAULT_ALIGNMENT_THRESHOLD_NS))
        );
        assert_eq!(
            element.get_property("discont-wait"),
            Some(PropValue::Uint(DEFAULT_DISCONT_WAIT_NS))
        );
    }

    #[test]
    fn samples_per_buffer_follows_the_rate_and_the_fraction() {
        let mut element = AudioBufferSplit::new();
        element.configure(&caps(RATE)).unwrap();
        assert_eq!(
            element.samples_per_buffer(),
            RATE as u64 * DEFAULT_OUTPUT_BUFFER_DURATION.0 as u64
                / DEFAULT_OUTPUT_BUFFER_DURATION.1 as u64
        );
        assert_eq!(element.error_per_buffer, 0, "48000/50 divides");
    }

    #[test]
    fn a_rate_that_does_not_divide_leaves_a_remainder() {
        let mut element = AudioBufferSplit::new().with_output_buffer_duration(1, 7);
        element.configure(&caps(44_100)).unwrap();
        assert_eq!(element.samples_per_buffer(), 44_100 / 7);
        assert_eq!(element.error_per_buffer, 44_100 % 7);
    }

    #[test]
    fn output_buffer_size_takes_precedence() {
        let mut element = AudioBufferSplit::new().with_output_buffer_size(4 * 100);
        element.configure(&caps(RATE)).unwrap();
        // 100 F32 mono frames, whatever the duration fraction says.
        assert_eq!(element.samples_per_buffer(), 100);
    }

    #[test]
    fn a_duration_below_one_frame_is_refused() {
        let mut element = AudioBufferSplit::new().with_output_buffer_duration(1, 1_000_000);
        assert_eq!(
            element.configure(&caps(RATE)).unwrap_err(),
            G2gError::CapsMismatch
        );
    }

    #[test]
    fn the_first_buffer_is_always_a_resync() {
        let mut align = StreamAlign::default();
        assert!(align.process(false, 0, 480, RATE, DEFAULT_ALIGNMENT_THRESHOLD_NS, 0));
        // a contiguous follow-on is not.
        assert!(!align.process(
            false,
            samples_to_ns(480, RATE),
            480,
            RATE,
            DEFAULT_ALIGNMENT_THRESHOLD_NS,
            0
        ));
    }

    #[test]
    fn a_jump_past_the_threshold_is_a_discont() {
        let mut align = StreamAlign::default();
        align.process(false, 0, 480, RATE, DEFAULT_ALIGNMENT_THRESHOLD_NS, 0);
        let jump = samples_to_ns(480, RATE) + DEFAULT_ALIGNMENT_THRESHOLD_NS * 2;
        assert!(align.process(false, jump, 480, RATE, DEFAULT_ALIGNMENT_THRESHOLD_NS, 0));
    }

    #[test]
    fn discont_wait_defers_the_discont() {
        let mut align = StreamAlign::default();
        let wait_ns = DEFAULT_DISCONT_WAIT_NS;
        align.process(false, 0, 480, RATE, DEFAULT_ALIGNMENT_THRESHOLD_NS, wait_ns);
        let jump = samples_to_ns(480, RATE) + DEFAULT_ALIGNMENT_THRESHOLD_NS * 2;
        assert!(
            !align.process(
                false,
                jump,
                480,
                RATE,
                DEFAULT_ALIGNMENT_THRESHOLD_NS,
                wait_ns
            ),
            "the deviation has not held for discont-wait yet"
        );
        assert!(align.process(
            false,
            jump + wait_ns,
            480,
            RATE,
            DEFAULT_ALIGNMENT_THRESHOLD_NS,
            wait_ns
        ));
    }
}

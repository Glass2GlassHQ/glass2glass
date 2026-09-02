//! Audio reverser (`audioreverse`), the audio half of reverse playback and the
//! analog of ffmpeg's `areverse` (GStreamer has no such element). It collects
//! `chunk-duration` of interleaved PCM and re-emits it as one buffer with the
//! sample frames in the opposite order, so the content plays backwards.
//!
//! [`GopReverse`](crate::gopreverse::GopReverse) does the same for decoded
//! video: it re-emits a GOP in descending PTS. Audio has no GOP, so the batch is
//! a duration instead, and the reversal happens inside the buffer rather than
//! across buffers. Timestamps follow the same rule either way: a chunk keeps the
//! time of the samples it was cut from, so a forward-stamped stream comes out
//! monotonically ascending and a reverse-playback feed (chunks arriving newest
//! first, under a `rate < 0` segment) comes out descending, which is what the
//! reverse segment maps to ascending running time.
//!
//! [`ScaleTempo`](crate::scaletempo::ScaleTempo) and [`Speed`](crate::speed::Speed)
//! leave a negative rate alone; this element is what actually reverses the
//! samples. Unlike `GopReverse` it reverses every stream it sees, not only a
//! reverse segment: it is placed in a pipeline on purpose.
//!
//! `chunk-duration` bounds how much is buffered. Zero buffers the whole stream
//! and reverses it at `Eos`, the exact `areverse` behaviour, at the cost of
//! holding all of it. A partial final chunk is flushed reversed at `Eos`, so no
//! samples are lost. CPU-only `no_std`.

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

use crate::audioconvert::{ns_to_samples, pcm_formats, sample_bytes, samples_to_ns};

/// One second of audio per reversed buffer.
const DEFAULT_CHUNK_DURATION_NS: u64 = 1_000_000_000;
const DEFAULT_CHUNK_DURATION_TEXT: &str = "1000000000";

/// Reverse the order of the sample frames in an interleaved buffer, leaving the
/// channels inside each frame as they are.
fn reverse_frames(bytes: &mut [u8], bytes_per_frame: usize) {
    let frames = bytes.len() / bytes_per_frame;
    for near in 0..frames / 2 {
        let far = frames - 1 - near;
        let (head, tail) = bytes.split_at_mut(far * bytes_per_frame);
        let head_frame = &mut head[near * bytes_per_frame..(near + 1) * bytes_per_frame];
        head_frame.swap_with_slice(&mut tail[..bytes_per_frame]);
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::audioreverse::AudioReverse;
///
/// // half-second batches instead of the default second.
/// let reverse = AudioReverse::new().with_chunk_duration(500_000_000);
/// ```
#[derive(Debug)]
pub struct AudioReverse {
    chunk_duration_ns: u64,
    input: Option<(AudioFormat, u8, u32)>,
    configured: bool,
    last_caps: Option<Caps>,

    /// Sample frames one reversed buffer holds, zero while the whole stream is
    /// batched (`chunk-duration` 0) or before the rate is known.
    chunk_frames: u64,
    /// Bytes not yet cut into a reversed buffer.
    adapter: Vec<u8>,
    /// Timestamp of the sample the adapter was last anchored at, and the frames
    /// emitted since. A buffer arriving on an empty adapter re-anchors both.
    anchor_ns: u64,
    anchor_offset: u64,
    emitted: u64,
}

impl Default for AudioReverse {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioReverse {
    pub fn new() -> Self {
        Self {
            chunk_duration_ns: DEFAULT_CHUNK_DURATION_NS,
            input: None,
            configured: false,
            last_caps: None,
            chunk_frames: 0,
            adapter: Vec::new(),
            anchor_ns: 0,
            anchor_offset: 0,
            emitted: 0,
        }
    }

    pub fn with_chunk_duration(mut self, duration_ns: u64) -> Self {
        self.chunk_duration_ns = duration_ns;
        self.update_chunk_frames();
        self
    }

    /// Sample frames one reversed buffer holds. Zero means the whole stream is
    /// held until `Eos`.
    pub fn chunk_frames(&self) -> u64 {
        self.chunk_frames
    }

    fn accept_input(&self, caps: &Caps) -> Result<(AudioFormat, u8, u32), G2gError> {
        let Caps::Audio {
            format,
            channels,
            sample_rate,
            ..
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

    fn update_chunk_frames(&mut self) {
        self.chunk_frames = match self.input {
            Some((_, _, rate)) => ns_to_samples(self.chunk_duration_ns, rate),
            None => 0,
        };
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let input = self.accept_input(caps)?;
        if self.input != Some(input) {
            self.input = Some(input);
            self.reset_stream();
        }
        self.update_chunk_frames();
        self.configured = true;
        Ok(())
    }

    fn reset_stream(&mut self) {
        self.adapter.clear();
        self.anchor_ns = 0;
        self.anchor_offset = 0;
    }

    /// Cut whole chunks out of the adapter, reversed. `force` also emits a final
    /// short one from whatever is left.
    async fn emit_chunks(&mut self, force: bool, out: &mut dyn OutputSink) -> Result<(), G2gError> {
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
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        let chunk_bytes = self.chunk_frames as usize * bytes_per_frame;

        while (chunk_bytes > 0 && self.adapter.len() >= chunk_bytes)
            || (force && !self.adapter.is_empty())
        {
            let take = if chunk_bytes > 0 {
                chunk_bytes.min(self.adapter.len())
            } else {
                self.adapter.len()
            };
            let mut bytes: Vec<u8> = self.adapter.drain(..take).collect();
            reverse_frames(&mut bytes, bytes_per_frame);

            let frames = (take / bytes_per_frame) as u64;
            let pts = self
                .anchor_ns
                .saturating_add(samples_to_ns(self.anchor_offset, rate));
            let end = self
                .anchor_ns
                .saturating_add(samples_to_ns(self.anchor_offset + frames, rate));
            self.anchor_offset += frames;

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
        }
        Ok(())
    }
}

impl AsyncElement for AudioReverse {
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

    /// Re-ordering only: the output caps equal the input.
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
                    if self.adapter.is_empty() {
                        self.anchor_ns = frame.timing.pts_ns;
                        self.anchor_offset = 0;
                    }
                    self.adapter.extend_from_slice(bytes);
                    self.emit_chunks(false, out).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    self.configure(&c)?;
                }
                PipelinePacket::Flush => {
                    self.reset_stream();
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // The runner emits the end itself; this flushes the last partial
                // chunk, reversed like the rest.
                PipelinePacket::Eos => self.emit_chunks(true, out).await?,
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AUDIOREVERSE_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Audio reverser",
            "Filter/Effect/Audio",
            "Re-emits each chunk of audio with its samples in the opposite order",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "chunk-duration" => {
                self.chunk_duration_ns = value.as_uint().ok_or(PropError::Type)?;
                self.update_chunk_frames();
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "chunk-duration" => Some(PropValue::Uint(self.chunk_duration_ns)),
            _ => None,
        }
    }
}

static AUDIOREVERSE_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "chunk-duration",
    PropKind::Uint,
    "audio reversed per output buffer in nanoseconds, 0 holds the whole stream until the end",
)
.with_default(DEFAULT_CHUNK_DURATION_TEXT)];

impl PadTemplates for AudioReverse {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |format| Caps::Audio {
            format,
            channels: ANY_CHANNELS,
            sample_rate: ANY_SAMPLE_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        let set = CapsSet::from_alternatives(pcm_formats().iter().copied().map(pcm).collect());
        vec![PadTemplate::sink(set.clone()), PadTemplate::source(set)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    fn caps() -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 1,
            sample_rate: RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    #[test]
    fn declared_default_matches_the_constant() {
        let element = AudioReverse::new();
        assert_eq!(
            element.get_property("chunk-duration"),
            Some(PropValue::Uint(DEFAULT_CHUNK_DURATION_NS))
        );
    }

    #[test]
    fn chunk_frames_follow_the_rate() {
        let mut element = AudioReverse::new().with_chunk_duration(DEFAULT_CHUNK_DURATION_NS / 2);
        element.configure(&caps()).unwrap();
        assert_eq!(element.chunk_frames(), (RATE / 2) as u64);
    }

    #[test]
    fn zero_duration_batches_the_whole_stream() {
        let mut element = AudioReverse::new().with_chunk_duration(0);
        element.configure(&caps()).unwrap();
        assert_eq!(element.chunk_frames(), 0);
    }

    #[test]
    fn frames_reverse_and_channels_stay_put() {
        // two stereo S16 frames: (1, 2) then (3, 4).
        let mut bytes = Vec::new();
        for sample in [1i16, 2, 3, 4] {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        reverse_frames(&mut bytes, 2 * 2);
        let out: Vec<i16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|s| i16::from_le_bytes(*s))
            .collect();
        assert_eq!(out, [3, 4, 1, 2]);
    }

    #[test]
    fn an_odd_frame_count_keeps_its_middle() {
        let mut bytes = Vec::from([1u8, 2, 3]);
        reverse_frames(&mut bytes, 1);
        assert_eq!(bytes, [3, 2, 1]);
    }
}

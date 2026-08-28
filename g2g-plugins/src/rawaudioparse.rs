//! Headerless raw-audio framer (`rawaudioparse`): a `ByteStream{Raw}` dump in
//! (a `.pcm` file, arbitrary chunks from `filesrc`), timestamped `Audio` buffers
//! out, each holding whole sample frames.
//!
//! The file says nothing about the samples, so the format, rate and channel
//! count are properties and the output caps state what they declare. Each input
//! chunk is cut at the last whole sample frame it holds and the remainder waits
//! for the next chunk, so a buffer boundary never splits a sample. Timestamps
//! run from the sample count, since the byte stream carries none.
//!
//! `format=alaw` / `format=mulaw` emit `Audio{Alaw}` / `Audio{Mulaw}` for the
//! `alawdec` / `mulawdec` pair rather than PCM; the sample is one byte per
//! channel there.
//!
//! Only the interleaved layout is read (`Caps::Audio` describes an interleaved
//! stream), so gst's `interleaved` and `channel-positions` are not exposed.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_warn, AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::audioconvert::{audio_format_from_str, audio_format_to_str, sample_bytes};

/// gst `rawaudioparse` defaults: 16-bit PCM, 44.1 kHz, stereo.
const DEFAULT_FORMAT: AudioFormat = AudioFormat::PcmS16Le;
const DEFAULT_SAMPLE_RATE: u32 = 44_100;
const DEFAULT_CHANNELS: u8 = 2;

/// The same values as declared text, for `gst-inspect`.
const DEFAULT_PCM_FORMAT_TEXT: &str = "S16LE";
const DEFAULT_SAMPLE_RATE_TEXT: &str = "44100";
const DEFAULT_CHANNELS_TEXT: &str = "2";

/// Highest sample rate and channel count accepted, so a bogus property fails
/// there rather than sizing a buffer from it.
const MAX_SAMPLE_RATE: u64 = 768_000;
const MAX_CHANNELS: u64 = 64;

const NS_PER_SECOND: u128 = 1_000_000_000;

static RAWAUDIOPARSE_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "format",
        PropKind::Str,
        "encoding of the raw stream: pcm | alaw | mulaw",
    )
    .with_default("pcm"),
    PropertySpec::new(
        "pcm-format",
        PropKind::Str,
        "sample format when format=pcm: S16LE | F32LE | S24LE | S32LE | U8",
    )
    .with_default(DEFAULT_PCM_FORMAT_TEXT),
    PropertySpec::new("sample-rate", PropKind::Uint, "sample rate in Hz")
        .with_default(DEFAULT_SAMPLE_RATE_TEXT)
        .with_range("1", "768000"),
    PropertySpec::new("num-channels", PropKind::Uint, "channels in the raw stream")
        .with_default(DEFAULT_CHANNELS_TEXT)
        .with_range("1", "64"),
];

/// The `format` property's non-PCM values, and the caps format each emits.
const COMPANDED_FORMATS: &[(&str, AudioFormat)] =
    &[("alaw", AudioFormat::Alaw), ("mulaw", AudioFormat::Mulaw)];

/// # Example
///
/// ```no_run
/// use g2g_plugins::rawaudioparse::RawAudioParse;
///
/// // gst-launch equivalent:
/// // filesrc location=tone.pcm ! rawaudioparse sample-rate=48000 num-channels=1
/// let parser = RawAudioParse::new().with_rate(48_000).with_channels(1);
/// ```
#[derive(Debug)]
pub struct RawAudioParse {
    format: AudioFormat,
    sample_rate: u32,
    channels: u8,
    configured: bool,
    caps_sent: bool,
    /// Bytes left over from the last chunk: fewer than one sample frame.
    partial: Vec<u8>,
    /// Sample frames emitted, the presentation-time counter.
    samples: u64,
    sequence: u64,
    log_name: LogName,
}

impl Default for RawAudioParse {
    fn default() -> Self {
        Self::new()
    }
}

impl RawAudioParse {
    pub fn new() -> Self {
        Self {
            format: DEFAULT_FORMAT,
            sample_rate: DEFAULT_SAMPLE_RATE,
            channels: DEFAULT_CHANNELS,
            configured: false,
            caps_sent: false,
            partial: Vec::new(),
            samples: 0,
            sequence: 0,
            log_name: LogName::default(),
        }
    }

    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn with_channels(mut self, channels: u8) -> Self {
        self.channels = channels;
        self
    }

    /// Buffers emitted so far.
    pub fn buffers_emitted(&self) -> u64 {
        self.sequence
    }

    /// Bytes one sample frame occupies: one sample per channel. A-law / mu-law
    /// samples are a single byte each.
    fn frame_bytes(&self) -> Option<usize> {
        let sample = match self.format {
            AudioFormat::Alaw | AudioFormat::Mulaw => 1,
            format => sample_bytes(format),
        };
        sample
            .checked_mul(usize::from(self.channels))
            .filter(|bytes| *bytes > 0)
    }

    fn output_caps(&self) -> Caps {
        Caps::Audio {
            format: self.format,
            channels: self.channels,
            sample_rate: self.sample_rate,
        }
    }

    fn input_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Raw,
        }
    }

    /// Push `bytes` (already whole sample frames) as one buffer, timestamped
    /// from the running sample count.
    async fn emit(&mut self, bytes: Vec<u8>, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let frame_bytes = self.frame_bytes().ok_or(G2gError::CapsMismatch)?;
        if !self.caps_sent {
            out.push(PipelinePacket::CapsChanged(self.output_caps()))
                .await?;
            self.caps_sent = true;
        }
        let rate = u128::from(self.sample_rate);
        let ns = |samples: u64| (u128::from(samples) * NS_PER_SECOND / rate) as u64;
        let samples = (bytes.len() / frame_bytes) as u64;
        let pts_ns = ns(self.samples);
        let duration_ns = ns(samples);
        self.samples += samples;
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns,
                // Raw samples: every buffer stands alone.
                keyframe: true,
                ..FrameTiming::default()
            },
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

impl AsyncElement for RawAudioParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Raw audio framer",
            "Codec/Parser/Audio",
            "Frames a headerless raw audio dump using its declared format, rate and channels",
            "g2g",
        )
    }

    /// Reads host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let caps = self.output_caps();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Raw,
            } => CapsSet::one(caps.clone()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Raw
            }
        ) {
            return Err(G2gError::CapsMismatch);
        }
        // Without a sample size there is nothing to cut the stream into.
        if self.frame_bytes().is_none() || self.sample_rate == 0 {
            return Err(G2gError::CapsMismatch);
        }
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    let frame_bytes = self.frame_bytes().ok_or(G2gError::CapsMismatch)?;
                    let mut bytes = core::mem::take(&mut self.partial);
                    bytes.extend_from_slice(slice);
                    let whole = bytes.len() - bytes.len() % frame_bytes;
                    self.partial = bytes.split_off(whole);
                    if !bytes.is_empty() {
                        self.emit(bytes, out).await?;
                    }
                }
                PipelinePacket::Eos => {
                    if !self.partial.is_empty() {
                        g2g_warn!(
                            self,
                            "dropped {} trailing bytes, short of a sample frame",
                            self.partial.len()
                        );
                        self.partial.clear();
                    }
                }
                PipelinePacket::Flush => {
                    self.partial.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
                // The declared shape replaces the byte stream's caps, which
                // carry no format.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        RAWAUDIOPARSE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "format" => {
                let text = value.as_str().ok_or(PropError::Type)?.to_ascii_lowercase();
                if text == "pcm" {
                    // The sample format is `pcm-format`'s to state; keep the one
                    // already set unless it is a companded format.
                    if matches!(self.format, AudioFormat::Alaw | AudioFormat::Mulaw) {
                        self.format = DEFAULT_FORMAT;
                    }
                    return Ok(());
                }
                self.format = COMPANDED_FORMATS
                    .iter()
                    .find(|(name, _)| *name == text)
                    .map(|(_, format)| *format)
                    .ok_or(PropError::Value)?;
            }
            "pcm-format" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                self.format = audio_format_from_str(text).ok_or(PropError::Value)?;
            }
            "sample-rate" => {
                let rate = value.as_uint().ok_or(PropError::Type)?;
                if rate == 0 || rate > MAX_SAMPLE_RATE {
                    return Err(PropError::Value);
                }
                self.sample_rate = rate as u32;
            }
            "num-channels" => {
                let channels = value.as_uint().ok_or(PropError::Type)?;
                if channels == 0 || channels > MAX_CHANNELS {
                    return Err(PropError::Value);
                }
                self.channels = channels as u8;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "format" => {
                let text = COMPANDED_FORMATS
                    .iter()
                    .find(|(_, format)| *format == self.format)
                    .map_or("pcm", |(name, _)| name);
                Some(PropValue::Str(text.into()))
            }
            "pcm-format" => Some(PropValue::Str(audio_format_to_str(self.format).into())),
            "sample-rate" => Some(PropValue::Uint(u64::from(self.sample_rate))),
            "num-channels" => Some(PropValue::Uint(u64::from(self.channels))),
            _ => None,
        }
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }
}

impl LogSource for RawAudioParse {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

impl PadTemplates for RawAudioParse {
    fn pad_templates() -> Vec<PadTemplate> {
        // `Caps::Audio` has no open fields, so the template pins the declared
        // default shape; an instance emits its own.
        let audio = Caps::Audio {
            format: DEFAULT_FORMAT,
            channels: DEFAULT_CHANNELS,
            sample_rate: DEFAULT_SAMPLE_RATE,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(CapsSet::one(audio)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use g2g_core::PushOutcome;

    #[derive(Default)]
    struct RecordingSink {
        packets: Vec<PipelinePacket>,
    }

    impl OutputSink for RecordingSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            self.packets.push(packet);
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    impl RecordingSink {
        fn caps(&self) -> Vec<&Caps> {
            self.packets
                .iter()
                .filter_map(|p| match p {
                    PipelinePacket::CapsChanged(c) => Some(c),
                    _ => None,
                })
                .collect()
        }

        fn frames(&self) -> Vec<&Frame> {
            self.packets
                .iter()
                .filter_map(|p| match p {
                    PipelinePacket::DataFrame(f) => Some(f),
                    _ => None,
                })
                .collect()
        }
    }

    fn parser() -> RawAudioParse {
        let mut parser = RawAudioParse::new().with_rate(48_000).with_channels(2);
        parser
            .configure_pipeline(&RawAudioParse::input_caps())
            .expect("a raw byte stream");
        parser
    }

    fn data_frame(bytes: Vec<u8>) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming::default(),
            0,
        ))
    }

    #[tokio::test]
    async fn holds_back_a_split_sample_frame() {
        let mut parser = parser();
        let mut sink = RecordingSink::default();
        // Stereo S16LE: 4 bytes a sample frame, so a 6-byte chunk splits one.
        parser
            .process(data_frame(vec![1u8; 6]), &mut sink)
            .await
            .expect("the chunk parses");
        parser
            .process(data_frame(vec![2u8; 6]), &mut sink)
            .await
            .expect("the chunk parses");
        let sizes: Vec<usize> = sink
            .frames()
            .iter()
            .map(|f| f.domain.as_system_slice().expect("system").len())
            .collect();
        assert_eq!(sizes, vec![4, 8], "each buffer is whole sample frames");
        // The held-back pair of bytes is carried into the second buffer.
        let second = sink.frames()[1].domain.as_system_slice().expect("system");
        assert_eq!(&second[..2], &[1u8, 1u8]);
    }

    #[tokio::test]
    async fn timestamps_run_from_the_sample_count() {
        let mut parser = parser();
        let mut sink = RecordingSink::default();
        // 480 sample frames = 10 ms at 48 kHz, stereo S16LE.
        const CHUNK_BYTES: usize = 480 * 4;
        const CHUNK_NS: u64 = 10_000_000;
        for _ in 0..3 {
            parser
                .process(data_frame(vec![0u8; CHUNK_BYTES]), &mut sink)
                .await
                .expect("the chunk parses");
        }
        let times: Vec<(u64, u64)> = sink
            .frames()
            .iter()
            .map(|f| (f.timing.pts_ns, f.timing.duration_ns))
            .collect();
        assert_eq!(
            times,
            vec![
                (0, CHUNK_NS),
                (CHUNK_NS, CHUNK_NS),
                (2 * CHUNK_NS, CHUNK_NS)
            ]
        );
        assert_eq!(
            sink.caps(),
            vec![&Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 48_000,
            }]
        );
    }

    #[tokio::test]
    async fn companded_format_emits_one_byte_samples() {
        let mut parser = RawAudioParse::new().with_rate(8_000).with_channels(1);
        parser
            .set_property("format", PropValue::Str("mulaw".into()))
            .expect("mu-law is a format value");
        parser
            .configure_pipeline(&RawAudioParse::input_caps())
            .expect("a raw byte stream");
        let mut sink = RecordingSink::default();
        parser
            .process(data_frame(vec![0xFFu8; 3]), &mut sink)
            .await
            .expect("the chunk parses");
        assert_eq!(
            sink.caps(),
            vec![&Caps::Audio {
                format: AudioFormat::Mulaw,
                channels: 1,
                sample_rate: 8_000,
            }]
        );
        // One byte per sample, so nothing is held back.
        assert_eq!(
            sink.frames()[0]
                .domain
                .as_system_slice()
                .expect("system")
                .len(),
            3
        );
    }

    #[tokio::test]
    async fn drops_a_trailing_partial_sample_frame() {
        let mut parser = parser();
        let mut sink = RecordingSink::default();
        parser
            .process(data_frame(vec![9u8; 5]), &mut sink)
            .await
            .expect("the chunk parses");
        parser
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("the tail flushes");
        assert_eq!(sink.frames().len(), 1);
        assert_eq!(parser.buffers_emitted(), 1);
    }

    #[test]
    fn format_and_pcm_format_do_not_overwrite_each_other() {
        let mut parser = RawAudioParse::new();
        parser
            .set_property("pcm-format", PropValue::Str("F32LE".into()))
            .expect("a PCM sample format");
        parser
            .set_property("format", PropValue::Str("pcm".into()))
            .expect("pcm is the default encoding");
        assert_eq!(
            parser.get_property("pcm-format"),
            Some(PropValue::Str("F32LE".into())),
            "format=pcm keeps the chosen sample format"
        );
        parser
            .set_property("format", PropValue::Str("alaw".into()))
            .expect("a-law is a format value");
        assert_eq!(
            parser.get_property("format"),
            Some(PropValue::Str("alaw".into()))
        );
        // Back to PCM: the companded format cannot stay as the sample format.
        parser
            .set_property("format", PropValue::Str("pcm".into()))
            .expect("pcm is the default encoding");
        assert_eq!(
            parser.get_property("pcm-format"),
            Some(PropValue::Str(DEFAULT_PCM_FORMAT_TEXT.into()))
        );
    }

    #[test]
    fn refuses_an_absurd_rate_or_channel_count() {
        let mut parser = RawAudioParse::new();
        assert_eq!(
            parser.set_property("sample-rate", PropValue::Uint(0)),
            Err(PropError::Value)
        );
        assert_eq!(
            parser.set_property("sample-rate", PropValue::Uint(1_000_000)),
            Err(PropError::Value)
        );
        assert_eq!(
            parser.set_property("num-channels", PropValue::Uint(65)),
            Err(PropError::Value)
        );
        assert_eq!(
            parser.set_property("pcm-format", PropValue::Str("S8".into())),
            Err(PropError::Value)
        );
    }
}

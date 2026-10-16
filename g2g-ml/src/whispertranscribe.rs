//! Speech-to-text on whisper.cpp (`whisper` feature, design/ml.md).
//!
//! `WhisperTranscribe` takes 16 kHz mono f32 PCM and emits one UTF-8
//! `Caps::Text` frame per transcribed segment, stamped with the segment's start
//! and length on the input timeline. Audio is gathered into windows of
//! `chunk-seconds` and each full window is transcribed in one call, so text
//! arrives one window late; what is left at EOS is transcribed then. Upstream
//! delivers the fixed format (`audioconvert format=F32LE channels=1 !
//! audioresample samplerate=16000` from anything else), the element does not
//! convert.

use core::future::Future;
use core::pin::Pin;

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperError,
    WhisperState,
};

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomain, MemoryDomainKind, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ChannelLayout, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, HardwareError, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, TextFormat,
};

const LOG_CATEGORY: &str = "WhisperTranscribe";
/// Whisper models are trained on 16 kHz mono audio.
pub const SAMPLE_RATE: u32 = 16_000;
const CHANNELS: u8 = 1;
const BYTES_PER_SAMPLE: usize = core::mem::size_of::<f32>();
const DEFAULT_CHUNK_SECONDS: f64 = 10.0;
const DEFAULT_CHUNK_SECONDS_TEXT: &str = "10.0";
/// whisper.cpp rejects input shorter than one second, so a shorter tail is
/// padded with silence up to this.
const MIN_SAMPLES: usize = SAMPLE_RATE as usize;
const DEFAULT_LANGUAGE: &str = "en";
/// whisper.cpp's language name that turns on detection.
pub const AUTO_LANGUAGE: &str = "auto";
/// Segment timestamps come back in centiseconds.
const SEGMENT_TIMESTAMP_UNIT_NS: u64 = 10_000_000;
const NS_PER_SECOND: f64 = 1_000_000_000.0;
/// Greedy decoding keeps one candidate per step.
const GREEDY_BEST_OF: i32 = 1;

/// A transcribed segment on the input timeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub text: String,
    pub pts_ns: u64,
    pub duration_ns: u64,
}

struct LoadedWhisper {
    _context: WhisperContext,
    state: WhisperState,
}

// the whisper handles have no `Debug` impl
impl core::fmt::Debug for LoadedWhisper {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("LoadedWhisper")
            .finish_non_exhaustive()
    }
}

/// Transcribes 16 kHz mono f32 audio into timed text frames.
///
/// # Example
///
/// ```no_run
/// use g2g_ml::whispertranscribe::WhisperTranscribe;
///
/// // gst-launch style: whispertranscribe model=ggml-tiny.en.bin chunk-seconds=5.0
/// let element = WhisperTranscribe::from_file("ggml-tiny.en.bin")
///     .unwrap()
///     .with_chunk_seconds(5.0);
/// ```
#[derive(Debug)]
pub struct WhisperTranscribe {
    model_path: Option<String>,
    language: String,
    translate: bool,
    chunk_seconds: f64,
    threads: u32,
    model: Option<LoadedWhisper>,
    pcm: Vec<f32>,
    /// Input-timeline time of `pcm[0]`.
    chunk_start_ns: u64,
    sequence: u64,
    caps_emitted: bool,
    configured: bool,
}

impl Default for WhisperTranscribe {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperTranscribe {
    /// A model-less element; `model=` (or [`Self::from_file`]) loads one.
    pub fn new() -> Self {
        Self {
            model_path: None,
            language: String::from(DEFAULT_LANGUAGE),
            translate: false,
            chunk_seconds: DEFAULT_CHUNK_SECONDS,
            threads: 0,
            model: None,
            pcm: Vec::new(),
            chunk_start_ns: 0,
            sequence: 0,
            caps_emitted: false,
            configured: false,
        }
    }

    /// Load a ggml whisper model (`ggml-*.bin`).
    pub fn from_file(path: &str) -> Result<Self, G2gError> {
        let mut element = Self::new();
        element.load_model(path)?;
        Ok(element)
    }

    /// The spoken language as whisper names it (`en`, `de`, ...), or
    /// [`AUTO_LANGUAGE`] to detect it per window (the `language` property).
    pub fn with_language(mut self, language: &str) -> Self {
        self.language = String::from(language);
        self
    }

    /// Translate the speech to English instead of transcribing it (the
    /// `translate` property).
    pub fn with_translate(mut self, translate: bool) -> Self {
        self.translate = translate;
        self
    }

    /// How much audio each transcription call covers (the `chunk-seconds`
    /// property).
    pub fn with_chunk_seconds(mut self, seconds: f64) -> Self {
        self.chunk_seconds = seconds;
        self
    }

    /// Decoder threads, `0` for whisper.cpp's default (the `threads` property).
    pub fn with_threads(mut self, threads: u32) -> Self {
        self.threads = threads;
        self
    }

    fn load_model(&mut self, path: &str) -> Result<(), G2gError> {
        let context = WhisperContext::new_with_params(path, WhisperContextParameters::default())
            .map_err(whisper_err)?;
        let state = context.create_state().map_err(whisper_err)?;
        self.model = Some(LoadedWhisper {
            _context: context,
            state,
        });
        self.model_path = Some(String::from(path));
        Ok(())
    }

    /// The one input format: 16 kHz mono f32.
    pub fn input_caps() -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: CHANNELS,
            sample_rate: SAMPLE_RATE,
            channel_layout: ChannelLayout::UNSPECIFIED,
        }
    }

    fn accepts_input(caps: &Caps) -> bool {
        matches!(
            caps,
            Caps::Audio {
                format: AudioFormat::PcmF32Le,
                channels: CHANNELS,
                sample_rate: SAMPLE_RATE,
                ..
            }
        )
    }

    pub fn text_caps() -> Caps {
        Caps::Text {
            format: TextFormat::Utf8,
        }
    }

    fn chunk_samples(&self) -> usize {
        ((self.chunk_seconds * f64::from(SAMPLE_RATE)) as usize).max(MIN_SAMPLES)
    }

    /// Transcribe `samples` as audio starting at `start_ns` on the input timeline.
    pub fn transcribe(&mut self, samples: &[f32], start_ns: u64) -> Result<Vec<Segment>, G2gError> {
        let Self {
            model,
            language,
            translate,
            threads,
            ..
        } = self;
        let model = model.as_mut().ok_or(G2gError::NotConfigured)?;

        let mut params = FullParams::new(SamplingStrategy::Greedy {
            best_of: GREEDY_BEST_OF,
        });
        params.set_language(Some(language.as_str()));
        params.set_translate(*translate);
        if *threads > 0 {
            params.set_n_threads(i32::try_from(*threads).unwrap_or(i32::MAX));
        }
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        model.state.full(params, samples).map_err(whisper_err)?;

        let mut segments = Vec::new();
        for segment in model.state.as_iter() {
            let text = segment.to_str_lossy().map_err(whisper_err)?;
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            let t0 = u64::try_from(segment.start_timestamp()).unwrap_or(0);
            let t1 = u64::try_from(segment.end_timestamp()).unwrap_or(t0);
            segments.push(Segment {
                text: String::from(text),
                pts_ns: start_ns.saturating_add(t0.saturating_mul(SEGMENT_TIMESTAMP_UNIT_NS)),
                duration_ns: t1
                    .saturating_sub(t0)
                    .saturating_mul(SEGMENT_TIMESTAMP_UNIT_NS),
            });
        }
        Ok(segments)
    }

    async fn push_segments(
        &mut self,
        segments: Vec<Segment>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        if segments.is_empty() {
            return Ok(());
        }
        if !self.caps_emitted {
            out.push(PipelinePacket::CapsChanged(Self::text_caps()))
                .await?;
            self.caps_emitted = true;
        }
        for segment in segments {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(
                    segment.text.into_bytes().into_boxed_slice(),
                )),
                FrameTiming {
                    pts_ns: segment.pts_ns,
                    duration_ns: segment.duration_ns,
                    ..FrameTiming::default()
                },
                self.sequence,
            );
            self.sequence += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }

    /// Transcribe every full window in the buffer, and the remainder too when
    /// `flush` is set.
    async fn drain(&mut self, flush: bool, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let chunk = self.chunk_samples();
        while self.pcm.len() >= chunk {
            let window: Vec<f32> = self.pcm.drain(..chunk).collect();
            let start_ns = self.chunk_start_ns;
            self.chunk_start_ns += samples_to_ns(chunk);
            let segments = self.transcribe(&window, start_ns)?;
            self.push_segments(segments, out).await?;
        }
        if !flush || self.pcm.is_empty() {
            return Ok(());
        }
        let mut tail = core::mem::take(&mut self.pcm);
        let start_ns = self.chunk_start_ns;
        self.chunk_start_ns += samples_to_ns(tail.len());
        tail.resize(tail.len().max(MIN_SAMPLES), 0.0);
        let segments = self.transcribe(&tail, start_ns)?;
        self.push_segments(segments, out).await
    }

    async fn collect(&mut self, frame: &Frame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let bytes = frame
            .domain
            .as_system_slice()
            .ok_or(G2gError::UnsupportedDomain)?;
        if self.pcm.is_empty() {
            if let Some(pts_ns) = frame.timing.pts() {
                self.chunk_start_ns = pts_ns;
            }
        }
        self.pcm.extend(
            bytes
                .as_chunks::<BYTES_PER_SAMPLE>()
                .0
                .iter()
                .map(|sample| f32::from_le_bytes(*sample)),
        );
        self.drain(false, out).await
    }
}

fn samples_to_ns(samples: usize) -> u64 {
    (samples as f64 / f64::from(SAMPLE_RATE) * NS_PER_SECOND) as u64
}

fn whisper_err(error: WhisperError) -> G2gError {
    g2g_core::g2g_error!(
        g2g_core::log::Target::category(LOG_CATEGORY),
        "whisper.cpp call failed: {error}"
    );
    G2gError::Hardware(HardwareError::Other)
}

impl AsyncElement for WhisperTranscribe {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Whisper transcription",
            "Filter/Audio/Text",
            "Transcribes 16 kHz mono speech into timed text with whisper.cpp",
            "g2g",
        )
    }

    /// Reads the samples out of host memory, so it takes system frames only.
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::accepts_input(upstream_caps) {
            Ok(Self::text_caps())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if WhisperTranscribe::accepts_input(input) {
                CapsSet::one(WhisperTranscribe::text_caps())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !Self::accepts_input(absolute_caps) {
            return Err(G2gError::CapsMismatch);
        }
        if self.model.is_none() {
            return Err(G2gError::NotConfigured);
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
                PipelinePacket::DataFrame(frame) => self.collect(&frame, out).await,
                // The element's own output caps come back from the runner and
                // must be forwarded; the audio input caps produce no packet.
                PipelinePacket::CapsChanged(caps) => {
                    if caps != Self::text_caps() {
                        return Ok(());
                    }
                    self.caps_emitted = true;
                    out.push(PipelinePacket::CapsChanged(caps))
                        .await
                        .map(|_| ())
                }
                // The runner forwards EOS after this returns, so the tail still
                // gets out in front of it.
                PipelinePacket::Eos => self.drain(true, out).await,
                PipelinePacket::Flush => {
                    self.pcm.clear();
                    out.push(PipelinePacket::Flush).await.map(|_| ())
                }
                other => out.push(other).await.map(|_| ()),
            }
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        WHISPER_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "model" => {
                let path = value.as_str().ok_or(PropError::Type)?;
                // a path that does not load is a bad value, so the launch line fails loud
                self.load_model(path).map_err(|_| PropError::Value)
            }
            "language" => {
                self.language = String::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            "translate" => {
                self.translate = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "chunk-seconds" => {
                let seconds = value.as_double().ok_or(PropError::Type)?;
                if seconds <= 0.0 {
                    return Err(PropError::Value);
                }
                self.chunk_seconds = seconds;
                Ok(())
            }
            "threads" => {
                let threads = value.as_uint().ok_or(PropError::Type)?;
                self.threads = u32::try_from(threads).map_err(|_| PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "model" => Some(PropValue::Str(self.model_path.clone().unwrap_or_default())),
            "language" => Some(PropValue::Str(self.language.clone())),
            "translate" => Some(PropValue::Bool(self.translate)),
            "chunk-seconds" => Some(PropValue::Double(self.chunk_seconds)),
            "threads" => Some(PropValue::Uint(u64::from(self.threads))),
            _ => None,
        }
    }
}

impl PadTemplates for WhisperTranscribe {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(CapsSet::one(Self::text_caps())),
        ])
    }
}

static WHISPER_PROPS: &[PropertySpec] = &[
    PropertySpec::new("model", PropKind::Str, "path to the ggml whisper model"),
    PropertySpec::new(
        "language",
        PropKind::Str,
        "spoken language as whisper names it, or auto to detect it",
    )
    .with_default(DEFAULT_LANGUAGE),
    PropertySpec::new(
        "translate",
        PropKind::Bool,
        "translate the speech to English instead of transcribing it",
    )
    .with_default("false"),
    PropertySpec::new(
        "chunk-seconds",
        PropKind::Double,
        "seconds of audio each transcription call covers",
    )
    .with_default(DEFAULT_CHUNK_SECONDS_TEXT),
    PropertySpec::new(
        "threads",
        PropKind::Uint,
        "decoder threads, 0 for whisper.cpp's default",
    )
    .with_default("0"),
];

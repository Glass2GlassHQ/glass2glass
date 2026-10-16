//! Text generation on llama.cpp (`llama` feature, design/ml.md).
//!
//! `LlamaChat` answers each UTF-8 `Caps::Text` frame as one chat turn on a GGUF
//! model and emits the reply as a text frame with the input's timing. Turns are
//! independent: every frame starts a fresh context holding only the system
//! prompt and that frame's text, so a digest stream (`textdigest ! llamachat`)
//! gets one answer per digest with no history between them. A model with a chat
//! template gets the system and user messages through it; one without gets the
//! system prompt and the text as a raw completion prompt.

use core::future::Future;
use core::num::NonZeroU32;
use core::pin::Pin;
use std::sync::OnceLock;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomain, MemoryDomainKind, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    HardwareError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, TextFormat,
};

const LOG_CATEGORY: &str = "LlamaChat";
const DEFAULT_MAX_TOKENS: u32 = 256;
const DEFAULT_MAX_TOKENS_TEXT: &str = "256";
const DEFAULT_CONTEXT_SIZE: u32 = 2048;
const DEFAULT_CONTEXT_SIZE_TEXT: &str = "2048";
/// Zero picks the most likely token every step.
const DEFAULT_TEMPERATURE: f64 = 0.0;
const DEFAULT_TEMPERATURE_TEXT: &str = "0.0";
const SYSTEM_ROLE: &str = "system";
const USER_ROLE: &str = "user";
/// The one sequence every token of a turn belongs to.
const SEQUENCE_ID: i32 = 0;
/// `sample` index for the logits of the last token decoded.
const LAST_TOKEN_LOGITS: i32 = -1;
/// What separates the system prompt from the text in a raw completion prompt.
const RAW_PROMPT_SEPARATOR: &str = "\n";

/// The backend can be started once per process.
fn backend() -> Result<&'static LlamaBackend, G2gError> {
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| LlamaBackend::init().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| llama_err(error))
}

/// Answers each text frame as one chat turn on a GGUF model.
///
/// # Example
///
/// ```no_run
/// use g2g_ml::llamachat::LlamaChat;
///
/// // gst-launch style: llamachat model=qwen.gguf system-prompt="Reply in one line." max-tokens=64
/// let element = LlamaChat::from_file("qwen.gguf")
///     .unwrap()
///     .with_system_prompt("Reply in one line.")
///     .with_max_tokens(64);
/// ```
pub struct LlamaChat {
    model_path: Option<String>,
    system_prompt: String,
    max_tokens: u32,
    context_size: u32,
    temperature: f64,
    seed: u32,
    threads: u32,
    model: Option<LlamaModel>,
    sequence: u64,
    caps_emitted: bool,
    configured: bool,
}

// `LlamaModel` has no `Debug` impl
impl core::fmt::Debug for LlamaChat {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("LlamaChat")
            .field("model_path", &self.model_path)
            .field("system_prompt", &self.system_prompt)
            .field("max_tokens", &self.max_tokens)
            .field("context_size", &self.context_size)
            .field("temperature", &self.temperature)
            .field("seed", &self.seed)
            .field("threads", &self.threads)
            .field("loaded", &self.model.is_some())
            .finish_non_exhaustive()
    }
}

impl Default for LlamaChat {
    fn default() -> Self {
        Self::new()
    }
}

impl LlamaChat {
    /// A model-less element; `model=` (or [`Self::from_file`]) loads one.
    pub fn new() -> Self {
        Self {
            model_path: None,
            system_prompt: String::new(),
            max_tokens: DEFAULT_MAX_TOKENS,
            context_size: DEFAULT_CONTEXT_SIZE,
            temperature: DEFAULT_TEMPERATURE,
            seed: 0,
            threads: 0,
            model: None,
            sequence: 0,
            caps_emitted: false,
            configured: false,
        }
    }

    /// Load a GGUF model.
    pub fn from_file(path: &str) -> Result<Self, G2gError> {
        let mut element = Self::new();
        element.load_model(path)?;
        Ok(element)
    }

    /// The system message every turn opens with, empty for none (the
    /// `system-prompt` property).
    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = String::from(prompt);
        self
    }

    /// The most tokens one reply may run to (the `max-tokens` property).
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Context window in tokens, prompt and reply together (the `context-size`
    /// property).
    pub fn with_context_size(mut self, context_size: u32) -> Self {
        self.context_size = context_size;
        self
    }

    /// Sampling temperature, `0` for greedy decoding (the `temperature`
    /// property).
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sampling seed, used when the temperature is above zero (the `seed`
    /// property).
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Decode threads, `0` for llama.cpp's default (the `threads` property).
    pub fn with_threads(mut self, threads: u32) -> Self {
        self.threads = threads;
        self
    }

    fn load_model(&mut self, path: &str) -> Result<(), G2gError> {
        // the crate's loader debug-asserts the path exists
        if !std::path::Path::new(path).is_file() {
            return Err(llama_err(&format_args!("no model file at {path}")));
        }
        let backend = backend()?;
        let model = LlamaModel::load_from_file(backend, path, &LlamaModelParams::default())
            .map_err(|error| llama_err(&error))?;
        self.model = Some(model);
        self.model_path = Some(String::from(path));
        Ok(())
    }

    pub fn text_caps() -> Caps {
        Caps::Text {
            format: TextFormat::Utf8,
        }
    }

    /// The prompt as the model's chat template renders the system and user
    /// messages, or the two joined raw when the model has none.
    fn prompt(&self, model: &LlamaModel, text: &str) -> Result<String, G2gError> {
        let Ok(template) = model.chat_template(None) else {
            if self.system_prompt.is_empty() {
                return Ok(String::from(text));
            }
            return Ok([self.system_prompt.as_str(), text].join(RAW_PROMPT_SEPARATOR));
        };
        let mut messages = Vec::new();
        if !self.system_prompt.is_empty() {
            messages.push(
                LlamaChatMessage::new(String::from(SYSTEM_ROLE), self.system_prompt.clone())
                    .map_err(|error| llama_err(&error))?,
            );
        }
        messages.push(
            LlamaChatMessage::new(String::from(USER_ROLE), String::from(text))
                .map_err(|error| llama_err(&error))?,
        );
        model
            .apply_chat_template(&template, &messages, true)
            .map_err(|error| llama_err(&error))
    }

    fn sampler(&self) -> LlamaSampler {
        if self.temperature <= 0.0 {
            return LlamaSampler::greedy();
        }
        LlamaSampler::chain_simple([
            LlamaSampler::temp(self.temperature as f32),
            LlamaSampler::dist(self.seed),
        ])
    }

    /// One turn: the reply to `text`.
    pub fn answer(&self, text: &str) -> Result<String, G2gError> {
        let backend = backend()?;
        let model = self.model.as_ref().ok_or(G2gError::NotConfigured)?;
        let prompt = self.prompt(model, text)?;
        let vocab = model.vocab();
        let prompt_tokens = vocab.tokenize(prompt.as_bytes(), true, true);
        let context_size = self.context_size as usize;
        if prompt_tokens.len() + self.max_tokens as usize > context_size {
            g2g_core::g2g_error!(
                g2g_core::log::Target::category(LOG_CATEGORY),
                "prompt of {} tokens plus max-tokens={} exceeds context-size={}",
                prompt_tokens.len(),
                self.max_tokens,
                self.context_size
            );
            return Err(G2gError::Hardware(HardwareError::Other));
        }

        let mut params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.context_size))
            .with_n_batch(self.context_size);
        if self.threads > 0 {
            let threads = i32::try_from(self.threads).unwrap_or(i32::MAX);
            params = params.with_n_threads(threads).with_n_threads_batch(threads);
        }
        let mut context = model
            .new_context(backend, params)
            .map_err(|error| llama_err(&error))?;

        let mut batch = LlamaBatch::new(context_size, 1);
        let last = prompt_tokens.len().saturating_sub(1);
        for (position, token) in prompt_tokens.iter().enumerate() {
            batch
                .add(*token, position as i32, &[SEQUENCE_ID], position == last)
                .map_err(|error| llama_err(&error))?;
        }
        context
            .decode(&mut batch)
            .map_err(|error| llama_err(&error))?;

        let mut sampler = self.sampler();
        let mut reply = Vec::new();
        let first_reply_position = prompt_tokens.len() as i32;
        for position in (first_reply_position..).take(self.max_tokens as usize) {
            let token = sampler.sample(&context, LAST_TOKEN_LOGITS);
            if vocab.is_eog(token) {
                break;
            }
            reply.extend(vocab.token_to_piece(token, false, None));
            batch.clear();
            batch
                .add(token, position, &[SEQUENCE_ID], true)
                .map_err(|error| llama_err(&error))?;
            context
                .decode(&mut batch)
                .map_err(|error| llama_err(&error))?;
        }
        Ok(String::from(String::from_utf8_lossy(&reply).trim()))
    }

    async fn reply_to(&mut self, frame: &Frame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let bytes = frame
            .domain
            .as_system_slice()
            .ok_or(G2gError::UnsupportedDomain)?;
        let reply = self.answer(&String::from_utf8_lossy(bytes))?;
        if !self.caps_emitted {
            out.push(PipelinePacket::CapsChanged(Self::text_caps()))
                .await?;
            self.caps_emitted = true;
        }
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(
                reply.into_bytes().into_boxed_slice(),
            )),
            frame.timing,
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await.map(|_| ())
    }
}

fn llama_err(error: &dyn core::fmt::Display) -> G2gError {
    g2g_core::g2g_error!(
        g2g_core::log::Target::category(LOG_CATEGORY),
        "llama.cpp call failed: {error}"
    );
    G2gError::Hardware(HardwareError::Other)
}

impl AsyncElement for LlamaChat {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "LLM chat",
            "Filter/Text",
            "Answers each text frame as one chat turn on a llama.cpp model",
            "g2g",
        )
    }

    /// Reads the text out of host memory, so it takes system frames only.
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if *upstream_caps == Self::text_caps() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if *input == LlamaChat::text_caps() {
                CapsSet::one(LlamaChat::text_caps())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if *absolute_caps != Self::text_caps() {
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
                PipelinePacket::DataFrame(frame) => self.reply_to(&frame, out).await,
                // The element's own output caps come back from the runner and
                // must be forwarded, which also covers the first reply's emission.
                PipelinePacket::CapsChanged(caps) => {
                    if caps != Self::text_caps() {
                        return Ok(());
                    }
                    self.caps_emitted = true;
                    out.push(PipelinePacket::CapsChanged(caps))
                        .await
                        .map(|_| ())
                }
                other => out.push(other).await.map(|_| ()),
            }
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        LLAMA_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "model" => {
                let path = value.as_str().ok_or(PropError::Type)?;
                // a path that does not load is a bad value, so the launch line fails loud
                self.load_model(path).map_err(|_| PropError::Value)
            }
            "system-prompt" => {
                self.system_prompt = String::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            "max-tokens" => {
                let max_tokens = value.as_uint().ok_or(PropError::Type)?;
                self.max_tokens = u32::try_from(max_tokens).map_err(|_| PropError::Value)?;
                Ok(())
            }
            "context-size" => {
                let context_size = value.as_uint().ok_or(PropError::Type)?;
                self.context_size = u32::try_from(context_size)
                    .ok()
                    .filter(|size| *size > 0)
                    .ok_or(PropError::Value)?;
                Ok(())
            }
            "temperature" => {
                let temperature = value.as_double().ok_or(PropError::Type)?;
                if temperature < 0.0 {
                    return Err(PropError::Value);
                }
                self.temperature = temperature;
                Ok(())
            }
            "seed" => {
                let seed = value.as_uint().ok_or(PropError::Type)?;
                self.seed = u32::try_from(seed).map_err(|_| PropError::Value)?;
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
            "system-prompt" => Some(PropValue::Str(self.system_prompt.clone())),
            "max-tokens" => Some(PropValue::Uint(u64::from(self.max_tokens))),
            "context-size" => Some(PropValue::Uint(u64::from(self.context_size))),
            "temperature" => Some(PropValue::Double(self.temperature)),
            "seed" => Some(PropValue::Uint(u64::from(self.seed))),
            "threads" => Some(PropValue::Uint(u64::from(self.threads))),
            _ => None,
        }
    }
}

impl PadTemplates for LlamaChat {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::text_caps())),
            PadTemplate::source(CapsSet::one(Self::text_caps())),
        ])
    }
}

static LLAMA_PROPS: &[PropertySpec] = &[
    PropertySpec::new("model", PropKind::Str, "path to the GGUF model"),
    PropertySpec::new(
        "system-prompt",
        PropKind::Str,
        "system message every turn opens with, empty for none",
    ),
    PropertySpec::new(
        "max-tokens",
        PropKind::Uint,
        "most tokens one reply may run to",
    )
    .with_default(DEFAULT_MAX_TOKENS_TEXT),
    PropertySpec::new(
        "context-size",
        PropKind::Uint,
        "context window in tokens, prompt and reply together",
    )
    .with_default(DEFAULT_CONTEXT_SIZE_TEXT),
    PropertySpec::new(
        "temperature",
        PropKind::Double,
        "sampling temperature, 0 picks the most likely token every step",
    )
    .with_default(DEFAULT_TEMPERATURE_TEXT),
    PropertySpec::new(
        "seed",
        PropKind::Uint,
        "sampling seed, used when the temperature is above zero",
    )
    .with_default("0"),
    PropertySpec::new(
        "threads",
        PropKind::Uint,
        "decode threads, 0 for llama.cpp's default",
    )
    .with_default("0"),
];

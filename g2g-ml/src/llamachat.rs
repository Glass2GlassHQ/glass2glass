//! Text generation on llama.cpp (`llama` feature, design/ml.md).
//!
//! `LlamaChat` answers each UTF-8 `Caps::Text` frame as one chat turn on a GGUF
//! model and emits the reply as a text frame with the input's timing. By default
//! turns are independent: every frame starts a fresh context holding only the
//! system prompt and that frame's text, so a digest stream (`textdigest !
//! llamachat`) gets one answer per digest with no history between them. With
//! `history-turns=N` the last N exchanges go ahead of the new text and one
//! context stays alive, so a turn feeds only the tokens its KV cache lacks. A
//! model with a chat template gets the messages through it; one without gets
//! them joined as a raw completion prompt.

use core::future::Future;
use core::num::NonZeroU32;
use core::pin::Pin;
use std::collections::VecDeque;
use std::sync::OnceLock;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

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
const ASSISTANT_ROLE: &str = "assistant";
/// The one sequence every token of a turn belongs to.
const SEQUENCE_ID: i32 = 0;
/// `sample` index for the logits of the last token decoded.
const LAST_TOKEN_LOGITS: i32 = -1;
/// What separates the messages in a raw completion prompt.
pub const RAW_PROMPT_SEPARATOR: &str = "\n";
/// llama.cpp's `n_gpu_layers` value for every layer, its default.
#[cfg(any(feature = "llama-cuda", feature = "llama-vulkan"))]
const EVERY_GPU_LAYER: i64 = -1;
#[cfg(any(feature = "llama-cuda", feature = "llama-vulkan"))]
const EVERY_GPU_LAYER_TEXT: &str = "-1";

/// The backend can be started once per process.
fn backend() -> Result<&'static LlamaBackend, G2gError> {
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| LlamaBackend::init().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| llama_err(error))
}

/// One user message and the reply to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exchange {
    pub user: String,
    pub assistant: String,
}

/// The context kept across turns while history is on.
struct Conversation {
    context: LlamaContext<'static>,
    /// The tokens in the KV cache, by position.
    cached_tokens: Vec<LlamaToken>,
    context_size: u32,
    threads: u32,
}

/// What one turn sampled.
struct Generated {
    text: String,
    /// Every sampled token the context decoded, so the end-of-generation token is absent.
    tokens: Vec<LlamaToken>,
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
    history_turns: u32,
    history: VecDeque<Exchange>,
    #[cfg(any(feature = "llama-cuda", feature = "llama-vulkan"))]
    gpu_layers: Option<u32>,
    // declared before `model` so its context drops before the model it points into
    conversation: Option<Conversation>,
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
            .field("history_turns", &self.history_turns)
            .field("history", &self.history)
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
            history_turns: 0,
            history: VecDeque::new(),
            #[cfg(any(feature = "llama-cuda", feature = "llama-vulkan"))]
            gpu_layers: None,
            conversation: None,
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

    /// Keep the last `turns` exchanges and feed them ahead of each new text,
    /// `0` for independent turns (the `history-turns` property).
    pub fn with_history_turns(mut self, turns: u32) -> Self {
        self.set_history_turns(turns);
        self
    }

    fn set_history_turns(&mut self, turns: u32) {
        self.history_turns = turns;
        self.trim_history();
        if turns == 0 {
            self.conversation = None;
        }
    }

    fn trim_history(&mut self) {
        while self.history.len() > self.history_turns as usize {
            self.history.pop_front();
        }
    }

    /// The exchanges the next turn feeds ahead of its text, oldest first.
    pub fn history(&self) -> &VecDeque<Exchange> {
        &self.history
    }

    /// Forget the kept exchanges (the `reset` property).
    pub fn reset(&mut self) {
        self.history.clear();
    }

    #[cfg(not(any(feature = "llama-cuda", feature = "llama-vulkan")))]
    fn model_params(&self) -> LlamaModelParams {
        LlamaModelParams::default()
    }

    #[cfg(any(feature = "llama-cuda", feature = "llama-vulkan"))]
    fn model_params(&self) -> LlamaModelParams {
        match self.gpu_layers {
            Some(layers) => LlamaModelParams::default().with_n_gpu_layers(layers),
            None => LlamaModelParams::default(),
        }
    }

    fn load_model(&mut self, path: &str) -> Result<(), G2gError> {
        // the crate's loader debug-asserts the path exists
        if !std::path::Path::new(path).is_file() {
            return Err(llama_err(&format_args!("no model file at {path}")));
        }
        let backend = backend()?;
        let model = LlamaModel::load_from_file(backend, path, &self.model_params())
            .map_err(|error| llama_err(&error))?;
        self.conversation = None;
        self.model = Some(model);
        self.model_path = Some(String::from(path));
        Ok(())
    }

    pub fn text_caps() -> Caps {
        Caps::Text {
            format: TextFormat::Utf8,
        }
    }

    /// The system prompt, the kept exchanges and `text` as role and content,
    /// in order.
    fn messages<'a>(&'a self, text: &'a str) -> Vec<(&'static str, &'a str)> {
        let mut messages = Vec::new();
        if !self.system_prompt.is_empty() {
            messages.push((SYSTEM_ROLE, self.system_prompt.as_str()));
        }
        for exchange in &self.history {
            messages.push((USER_ROLE, exchange.user.as_str()));
            messages.push((ASSISTANT_ROLE, exchange.assistant.as_str()));
        }
        messages.push((USER_ROLE, text));
        messages
    }

    /// The prompt as the model's chat template renders the messages, or their
    /// contents joined raw when the model has none.
    fn prompt(&self, model: &LlamaModel, text: &str) -> Result<String, G2gError> {
        let messages = self.messages(text);
        let Ok(template) = model.chat_template(None) else {
            let contents: Vec<&str> = messages.iter().map(|(_, content)| *content).collect();
            return Ok(contents.join(RAW_PROMPT_SEPARATOR));
        };
        let messages = messages
            .into_iter()
            .map(|(role, content)| {
                LlamaChatMessage::new(String::from(role), String::from(content))
                    .map_err(|error| llama_err(&error))
            })
            .collect::<Result<Vec<_>, _>>()?;
        model
            .apply_chat_template(&template, &messages, true)
            .map_err(|error| llama_err(&error))
    }

    fn prompt_tokens(&self, text: &str) -> Result<Vec<LlamaToken>, G2gError> {
        let model = self.model.as_ref().ok_or(G2gError::NotConfigured)?;
        let prompt = self.prompt(model, text)?;
        Ok(model.vocab().tokenize(prompt.as_bytes(), true, true))
    }

    /// The prompt tokens for `text`, after dropping the oldest exchanges until
    /// the prompt and a full reply fit the context.
    fn fitted_prompt_tokens(&mut self, text: &str) -> Result<Vec<LlamaToken>, G2gError> {
        loop {
            let prompt_tokens = self.prompt_tokens(text)?;
            if prompt_tokens.len() + self.max_tokens as usize <= self.context_size as usize {
                return Ok(prompt_tokens);
            }
            if self.history.pop_front().is_none() {
                g2g_core::g2g_error!(
                    g2g_core::log::Target::category(LOG_CATEGORY),
                    "prompt of {} tokens plus max-tokens={} exceeds context-size={}",
                    prompt_tokens.len(),
                    self.max_tokens,
                    self.context_size
                );
                return Err(G2gError::Hardware(HardwareError::Other));
            }
        }
    }

    fn context_params(&self) -> LlamaContextParams {
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.context_size))
            .with_n_batch(self.context_size);
        if self.threads == 0 {
            return params;
        }
        let threads = i32::try_from(self.threads).unwrap_or(i32::MAX);
        params.with_n_threads(threads).with_n_threads_batch(threads)
    }

    /// The kept context when its settings still match, else a new empty one.
    fn kept_conversation(&mut self) -> Result<Conversation, G2gError> {
        if let Some(conversation) = self.conversation.take() {
            if conversation.context_size == self.context_size
                && conversation.threads == self.threads
            {
                return Ok(conversation);
            }
        }
        let model = self.model.as_ref().ok_or(G2gError::NotConfigured)?;
        let context = model
            .new_context(backend()?, self.context_params())
            .map_err(|error| llama_err(&error))?;
        // SAFETY: llama.cpp keeps its own pointer to the model, valid until `self.model` drops or
        // is replaced, and the `conversation` field is cleared or dropped before either happens.
        let context =
            unsafe { core::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(context) };
        Ok(Conversation {
            context,
            cached_tokens: Vec::new(),
            context_size: self.context_size,
            threads: self.threads,
        })
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

    /// One turn: the reply to `text`, after the kept exchanges when
    /// `history-turns` is above zero.
    pub fn answer(&mut self, text: &str) -> Result<String, G2gError> {
        let prompt_tokens = self.fitted_prompt_tokens(text)?;
        if self.history_turns == 0 {
            let model = self.model.as_ref().ok_or(G2gError::NotConfigured)?;
            let mut context = model
                .new_context(backend()?, self.context_params())
                .map_err(|error| llama_err(&error))?;
            return Ok(self.generate(&mut context, &prompt_tokens, 0)?.text);
        }

        let mut conversation = self.kept_conversation()?;
        let shared = conversation
            .cached_tokens
            .iter()
            .zip(&prompt_tokens)
            .take_while(|(cached, wanted)| cached == wanted)
            .count();
        // the last prompt token is fed again even when cached, its logits pick the first reply token
        let reused = shared.min(prompt_tokens.len().saturating_sub(1));
        let first_stale = u32::try_from(reused).map_err(|error| llama_err(&error))?;
        conversation
            .context
            .kv_cache_seq_rm(SEQUENCE_ID, Some(first_stale), None)
            .map_err(|error| llama_err(&error))?;
        let generated = self.generate(&mut conversation.context, &prompt_tokens, reused)?;
        conversation.cached_tokens = [prompt_tokens, generated.tokens].concat();
        self.conversation = Some(conversation);
        self.history.push_back(Exchange {
            user: String::from(text),
            assistant: generated.text.clone(),
        });
        self.trim_history();
        Ok(generated.text)
    }

    /// Decode `prompt_tokens` from position `reused` on, then sample the reply.
    fn generate(
        &self,
        context: &mut LlamaContext<'_>,
        prompt_tokens: &[LlamaToken],
        reused: usize,
    ) -> Result<Generated, G2gError> {
        let model = self.model.as_ref().ok_or(G2gError::NotConfigured)?;
        let vocab = model.vocab();
        let mut batch = LlamaBatch::new(self.context_size as usize, 1);
        let last = prompt_tokens.len().saturating_sub(1);
        for (position, token) in prompt_tokens.iter().enumerate().skip(reused) {
            batch
                .add(*token, position as i32, &[SEQUENCE_ID], position == last)
                .map_err(|error| llama_err(&error))?;
        }
        context
            .decode(&mut batch)
            .map_err(|error| llama_err(&error))?;

        let mut sampler = self.sampler();
        let mut reply = Vec::new();
        let mut tokens = Vec::new();
        let first_reply_position = prompt_tokens.len() as i32;
        for position in (first_reply_position..).take(self.max_tokens as usize) {
            let token = sampler.sample(context, LAST_TOKEN_LOGITS);
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
            tokens.push(token);
        }
        Ok(Generated {
            text: String::from(String::from_utf8_lossy(&reply).trim()),
            tokens,
        })
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
            "history-turns" => {
                let turns = value.as_uint().ok_or(PropError::Type)?;
                self.set_history_turns(u32::try_from(turns).map_err(|_| PropError::Value)?);
                Ok(())
            }
            "reset" => {
                if value.as_bool().ok_or(PropError::Type)? {
                    self.reset();
                }
                Ok(())
            }
            #[cfg(any(feature = "llama-cuda", feature = "llama-vulkan"))]
            "n-gpu-layers" => {
                let layers = value.as_int().ok_or(PropError::Type)?;
                self.gpu_layers = if layers == EVERY_GPU_LAYER {
                    None
                } else {
                    Some(u32::try_from(layers).map_err(|_| PropError::Value)?)
                };
                // the layer split is fixed at load, so a loaded model loads again
                match self.model_path.clone() {
                    Some(path) => self.load_model(&path).map_err(|_| PropError::Value),
                    None => Ok(()),
                }
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
            "history-turns" => Some(PropValue::Uint(u64::from(self.history_turns))),
            "reset" => Some(PropValue::Bool(false)),
            #[cfg(any(feature = "llama-cuda", feature = "llama-vulkan"))]
            "n-gpu-layers" => Some(PropValue::Int(
                self.gpu_layers.map_or(EVERY_GPU_LAYER, i64::from),
            )),
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
    PropertySpec::new(
        "history-turns",
        PropKind::Uint,
        "exchanges kept and fed ahead of each new text, 0 for independent turns",
    )
    .with_default("0"),
    PropertySpec::new(
        "reset",
        PropKind::Bool,
        "set true to forget the kept exchanges, reads back false",
    )
    .with_default("false"),
    #[cfg(any(feature = "llama-cuda", feature = "llama-vulkan"))]
    PropertySpec::new(
        "n-gpu-layers",
        PropKind::Int,
        "model layers offloaded to the GPU, -1 for every layer",
    )
    .with_default(EVERY_GPU_LAYER_TEXT),
];

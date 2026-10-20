//! `LlamaChat` against the real llama.cpp runtime on the stories260K GGUF
//! (`tools/llama-fixture.sh` into the gitignored `fixtures/llama`); skips when
//! absent. The model is a TinyStories toy with no chat template, so the prompt
//! goes in raw and the reply continues the story. Run:
//!
//! ```powershell
//! cargo test -p g2g-ml --features llama --test llama_chat
//! ```
//!
//! With `llama-cuda` or `llama-vulkan` the same tests run on the GPU.

#![cfg(feature = "llama")]

use std::path::PathBuf;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{G2gError, PropError, PropValue};
use g2g_ml::llamachat::{Exchange, LlamaChat, RAW_PROMPT_SEPARATOR};

const MODEL: &str = "stories260K.gguf";
const PROMPT: &str = "Once upon a time";
const SECOND_PROMPT: &str = "The dog ran to the park";
const THIRD_PROMPT: &str = "Then it started to rain";
const KEPT_TURNS: u32 = 2;
const LONG_REPLY_TOKENS: u32 = 32;
const SHORT_REPLY_TOKENS: u32 = 4;
/// Too small for the prompt plus any reply.
const TINY_CONTEXT: u32 = 8;
/// Room for one long exchange, not two.
const ONE_EXCHANGE_CONTEXT: u32 = 2 * LONG_REPLY_TOKENS;
const NS_PER_SECOND: u64 = 1_000_000_000;

fn model_path() -> Option<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/llama")
        .join(MODEL);
    if !path.exists() {
        eprintln!(
            "llama fixture absent ({}); run tools/llama-fixture.sh. skipping.",
            path.display()
        );
        return None;
    }
    Some(path.to_str().unwrap().to_string())
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
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn text_frame(text: &str, pts_ns: u64) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(
            text.as_bytes().to_vec().into_boxed_slice(),
        )),
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        0,
    )
}

#[test]
fn greedy_replies_are_deterministic_and_capped_by_max_tokens() {
    let Some(model) = model_path() else {
        return;
    };
    let mut element = LlamaChat::from_file(&model)
        .expect("model loads")
        .with_max_tokens(LONG_REPLY_TOKENS);
    let long = element.answer(PROMPT).expect("reply");
    assert!(!long.is_empty(), "empty reply");
    assert_eq!(
        element.answer(PROMPT).expect("reply"),
        long,
        "greedy decoding repeats itself"
    );

    let short = element
        .with_max_tokens(SHORT_REPLY_TOKENS)
        .answer(PROMPT)
        .expect("reply");
    assert!(
        short.len() < long.len() && long.starts_with(&short),
        "the capped reply {short:?} is not a shorter prefix of {long:?}"
    );
}

#[test]
fn a_prompt_that_does_not_fit_the_context_fails_loud() {
    let Some(model) = model_path() else {
        return;
    };
    let mut element = LlamaChat::from_file(&model)
        .expect("model loads")
        .with_context_size(TINY_CONTEXT)
        .with_max_tokens(TINY_CONTEXT);
    assert!(element.answer(PROMPT).is_err());
}

#[tokio::test]
async fn a_text_frame_comes_back_as_one_reply_with_its_timing() {
    let Some(model) = model_path() else {
        return;
    };
    let mut element = LlamaChat::from_file(&model)
        .expect("model loads")
        .with_max_tokens(SHORT_REPLY_TOKENS);
    element
        .configure_pipeline(&LlamaChat::text_caps())
        .expect("configure");
    let pts_ns = 7 * NS_PER_SECOND;
    let mut sink = Collect::default();
    element
        .process(
            PipelinePacket::DataFrame(text_frame(PROMPT, pts_ns)),
            &mut sink,
        )
        .await
        .expect("frame");
    let [PipelinePacket::CapsChanged(caps), PipelinePacket::DataFrame(frame)] = &sink.packets[..]
    else {
        panic!(
            "expected CapsChanged then one text frame, got {:?}",
            sink.packets
        );
    };
    assert_eq!(caps, &LlamaChat::text_caps());
    assert_eq!(
        frame.timing.pts_ns, pts_ns,
        "the reply keeps the prompt's timing"
    );
    let reply = std::str::from_utf8(frame.domain.as_system_slice().unwrap()).unwrap();
    assert!(!reply.is_empty() && reply != PROMPT, "reply was {reply:?}");
}

#[test]
fn properties_round_trip_and_a_bad_model_is_a_bad_value() {
    let mut element = LlamaChat::new();
    assert_eq!(
        element.set_property("model", PropValue::Str("/nonexistent/nope.gguf".into())),
        Err(PropError::Value)
    );
    element
        .set_property("system-prompt", PropValue::Str("Reply in one line.".into()))
        .unwrap();
    element
        .set_property("max-tokens", PropValue::Uint(12))
        .unwrap();
    element
        .set_property("context-size", PropValue::Uint(512))
        .unwrap();
    element
        .set_property("temperature", PropValue::Double(0.7))
        .unwrap();
    element.set_property("seed", PropValue::Uint(42)).unwrap();
    element.set_property("threads", PropValue::Uint(3)).unwrap();
    element
        .set_property("history-turns", PropValue::Uint(4))
        .unwrap();
    assert_eq!(
        element.set_property("context-size", PropValue::Uint(0)),
        Err(PropError::Value)
    );
    assert_eq!(
        element.set_property("temperature", PropValue::Double(-1.0)),
        Err(PropError::Value)
    );
    assert_eq!(
        element.set_property("max-tokens", PropValue::Str("12".into())),
        Err(PropError::Type)
    );
    assert_eq!(
        element.get_property("system-prompt"),
        Some(PropValue::Str("Reply in one line.".into()))
    );
    assert_eq!(
        element.get_property("max-tokens"),
        Some(PropValue::Uint(12))
    );
    assert_eq!(
        element.get_property("context-size"),
        Some(PropValue::Uint(512))
    );
    assert_eq!(
        element.get_property("temperature"),
        Some(PropValue::Double(0.7))
    );
    assert_eq!(element.get_property("seed"), Some(PropValue::Uint(42)));
    assert_eq!(element.get_property("threads"), Some(PropValue::Uint(3)));
    assert_eq!(
        element.get_property("history-turns"),
        Some(PropValue::Uint(4))
    );
    assert!(
        matches!(
            element.configure_pipeline(&LlamaChat::text_caps()),
            Err(G2gError::NotConfigured)
        ),
        "no model, no run"
    );
}

fn long_chat(model: &str) -> LlamaChat {
    LlamaChat::from_file(model)
        .expect("model loads")
        .with_max_tokens(LONG_REPLY_TOKENS)
}

/// The reply a history-less element gives to the raw prompt the kept turns render to.
fn reply_to_rendered(model: &str, turns: &[&str]) -> String {
    long_chat(model)
        .answer(&turns.join(RAW_PROMPT_SEPARATOR))
        .expect("reply")
}

#[test]
fn kept_exchanges_go_ahead_of_the_new_text() {
    let Some(model) = model_path() else {
        return;
    };
    let mut chat = long_chat(&model).with_history_turns(KEPT_TURNS);
    let first = chat.answer(PROMPT).expect("first reply");
    let second = chat.answer(SECOND_PROMPT).expect("second reply");
    assert_eq!(
        chat.history().iter().cloned().collect::<Vec<_>>(),
        [
            Exchange {
                user: PROMPT.into(),
                assistant: first.clone()
            },
            Exchange {
                user: SECOND_PROMPT.into(),
                assistant: second.clone()
            },
        ]
    );
    // the second turn decodes only its new tokens, and must match a full recompute
    assert_eq!(
        second,
        reply_to_rendered(&model, &[PROMPT, &first, SECOND_PROMPT]),
        "the second turn did not see the first exchange"
    );
    assert_ne!(
        second,
        long_chat(&model).answer(SECOND_PROMPT).expect("reply"),
        "the history did not change the answer"
    );

    let third = chat.answer(THIRD_PROMPT).expect("third reply");
    assert_eq!(
        third,
        reply_to_rendered(
            &model,
            &[PROMPT, &first, SECOND_PROMPT, &second, THIRD_PROMPT]
        ),
        "the third turn did not see both kept exchanges"
    );
    let users: Vec<&str> = chat.history().iter().map(|e| e.user.as_str()).collect();
    assert_eq!(
        users,
        [SECOND_PROMPT, THIRD_PROMPT],
        "more than the kept number of exchanges stays"
    );
}

#[test]
fn reset_forgets_the_kept_exchanges() {
    let Some(model) = model_path() else {
        return;
    };
    let mut chat = long_chat(&model).with_history_turns(KEPT_TURNS);
    chat.answer(SECOND_PROMPT).expect("first reply");
    chat.set_property("reset", PropValue::Bool(true)).unwrap();
    assert!(chat.history().is_empty());
    assert_eq!(chat.get_property("reset"), Some(PropValue::Bool(false)));
    assert_eq!(
        chat.answer(PROMPT).expect("reply"),
        long_chat(&model).answer(PROMPT).expect("reply"),
        "the forgotten exchange still shaped the answer"
    );
}

#[test]
fn an_exchange_that_no_longer_fits_the_context_is_dropped() {
    let Some(model) = model_path() else {
        return;
    };
    let mut chat = long_chat(&model)
        .with_context_size(ONE_EXCHANGE_CONTEXT)
        .with_history_turns(KEPT_TURNS);
    chat.answer(PROMPT).expect("first reply");
    let second = chat.answer(SECOND_PROMPT).expect("second reply");
    let users: Vec<&str> = chat.history().iter().map(|e| e.user.as_str()).collect();
    assert_eq!(
        users,
        [SECOND_PROMPT],
        "the exchange that did not fit stays"
    );
    assert_eq!(
        second,
        long_chat(&model)
            .with_context_size(ONE_EXCHANGE_CONTEXT)
            .answer(SECOND_PROMPT)
            .expect("reply")
    );
}

#[test]
fn no_history_keeps_turns_independent() {
    let Some(model) = model_path() else {
        return;
    };
    let mut chat = long_chat(&model);
    let alone = chat.answer(SECOND_PROMPT).expect("reply");
    chat.answer(PROMPT).expect("reply");
    assert!(chat.history().is_empty());
    assert_eq!(chat.answer(SECOND_PROMPT).expect("reply"), alone);
}

#[cfg(any(feature = "llama-cuda", feature = "llama-vulkan"))]
#[test]
fn every_layer_is_offloaded_by_default_and_zero_keeps_the_model_on_the_cpu() {
    let default_layers = llama_cpp_2::model::params::LlamaModelParams::default().n_gpu_layers();
    let mut element = LlamaChat::new();
    let declared = element
        .properties()
        .iter()
        .find(|spec| spec.name == "n-gpu-layers")
        .and_then(|spec| spec.default)
        .expect("n-gpu-layers declares a default");
    assert_eq!(declared, default_layers.to_string());
    assert_eq!(
        element.get_property("n-gpu-layers"),
        Some(PropValue::Int(i64::from(default_layers)))
    );
    assert_eq!(
        element.set_property("n-gpu-layers", PropValue::Int(-2)),
        Err(PropError::Value)
    );
    let Some(model) = model_path() else {
        return;
    };
    element
        .set_property("model", PropValue::Str(model.clone()))
        .unwrap();
    let mut on_gpu = element.with_max_tokens(LONG_REPLY_TOKENS);
    let gpu_reply = on_gpu.answer(PROMPT).expect("reply");
    on_gpu
        .set_property("n-gpu-layers", PropValue::Int(0))
        .expect("reloads");
    assert_eq!(on_gpu.get_property("n-gpu-layers"), Some(PropValue::Int(0)));
    let cpu_reply = on_gpu.answer(PROMPT).expect("reply");
    assert!(!gpu_reply.is_empty() && !cpu_reply.is_empty());
}

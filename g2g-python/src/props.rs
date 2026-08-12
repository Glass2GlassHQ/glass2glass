//! The property face shared by every hosted gst-python-ml element.
//!
//! A `pyelement` and a `pyaggregator` host the same family of Python classes, so
//! they accept the same tunables and forward them the same way. One list here so
//! adding a property to one host cannot leave the other behind, plus the parsing
//! for the caps-valued properties both hosts read themselves rather than forward.

use g2g_core::{Caps, CapsSet, PropError};

/// The one concrete caps a `input-caps=` / `output-caps=` description names.
/// A description that names a set (a `{a,b}` list, an unbounded geometry) has no
/// single answer, so it is rejected rather than silently resolved.
pub(crate) fn fixed_caps(desc: &str) -> Result<Caps, PropError> {
    CapsSet::from_gst_string(desc)
        .and_then(|set| set.fixate())
        .ok_or(PropError::Value)
}

/// Build a host element's property list: its own entries first, then every
/// property forwarded verbatim to the hosted Python instance. `PropertySpec` and
/// `PropKind` resolve at the call site, which already imports both.
macro_rules! hosted_element_props {
    ($($own:expr),+ $(,)?) => {
        &[
            $($own,)+
        // Common ML tunables declared by the gst-python-ml backend BaseTransform.
        // These are forwarded to the hosted Python instance (a property absent from
        // the Python class is simply set as an attribute it ignores). Declaring them
        // here lets `gst-launch` type and accept `model-name=...` etc.
        PropertySpec::new(
            "model-name",
            PropKind::Str,
            "pre-trained model name or local path",
        ),
        PropertySpec::new(
            "engine-name",
            PropKind::Str,
            "ML engine: pytorch, onnx, tensorflow, tflite, openvino, ...",
        ),
        PropertySpec::new(
            "device",
            PropKind::Str,
            "inference device: cpu, cuda, cuda:0, ...",
        ),
        PropertySpec::new(
            "batch-size",
            PropKind::Int,
            "number of items to process in a batch",
        ),
        PropertySpec::new(
            "frame-stride",
            PropKind::Int,
            "how often to process a frame",
        ),
        PropertySpec::new(
            "input-format",
            PropKind::Str,
            "input tensor layout: auto, nhwc, or nchw",
        ),
        PropertySpec::new(
            "post-process",
            PropKind::Str,
            "post-processing format for raw output",
        ),
        PropertySpec::new(
            "device-queue-id",
            PropKind::Int,
            "DeviceQueue id from the pool to use",
        ),
        PropertySpec::new(
            "compile",
            PropKind::Bool,
            "enable torch.compile for the model",
        )
        .with_default("false"),
        PropertySpec::new(
            "track",
            PropKind::Bool,
            "enable object tracking (detectors)",
        )
        .with_default("false"),
        // Per-element knobs the gst-python-ml elements declare. Forwarded the same
        // way, so the pipeline line a gst-python-ml README prints runs here too.
        PropertySpec::new("acks", PropKind::Int, "Kafka producer acknowledgement mode"),
        PropertySpec::new("broker", PropKind::Str, "Kafka broker host:port"),
        PropertySpec::new("caption-file", PropKind::Str, "file to read captions from"),
        PropertySpec::new("colormap", PropKind::Str, "colormap the visualization uses"),
        PropertySpec::new(
            "compression-type",
            PropKind::Str,
            "Kafka message compression",
        ),
        PropertySpec::new("cooldown", PropKind::Int, "seconds between repeated alerts"),
        PropertySpec::new("draw-alert", PropKind::Bool, "overlay a fired alert"),
        PropertySpec::new(
            "draw-heatmap",
            PropKind::Bool,
            "overlay the anomaly heatmap",
        ),
        PropertySpec::new("draw-text", PropKind::Bool, "overlay the recognized text"),
        PropertySpec::new("framerate", PropKind::Str, "framerate the element assumes"),
        PropertySpec::new("gallery-path", PropKind::Str, "directory of known faces"),
        PropertySpec::new(
            "gate",
            PropKind::Bool,
            "pass audio only while voice is present",
        ),
        PropertySpec::new("history-length", PropKind::Uint, "how many items to keep"),
        PropertySpec::new(
            "initial-prompt",
            PropKind::Str,
            "prompt seeding transcription",
        ),
        PropertySpec::new("iou-threshold", PropKind::Double, "IoU a track match needs"),
        PropertySpec::new("labels", PropKind::Str, "comma separated candidate labels"),
        PropertySpec::new("language", PropKind::Str, "language code, e.g. en or ko"),
        PropertySpec::new("linger-ms", PropKind::Int, "Kafka producer linger"),
        PropertySpec::new("llm-model-name", PropKind::Str, "model the LLM stage loads"),
        PropertySpec::new(
            "max-age",
            PropKind::Int,
            "frames a track survives unmatched",
        ),
        PropertySpec::new(
            "max-masks",
            PropKind::Int,
            "most masks to segment per frame",
        ),
        PropertySpec::new("max-queue-size", PropKind::Uint, "buffers to queue per pad"),
        PropertySpec::new("max-tokens", PropKind::Int, "most tokens to generate"),
        PropertySpec::new(
            "message-timeout-ms",
            PropKind::Int,
            "Kafka delivery timeout",
        ),
        PropertySpec::new("meta-path", PropKind::Str, "metadata file to draw from"),
        PropertySpec::new(
            "min-hits",
            PropKind::Int,
            "matches before a track is reported",
        ),
        PropertySpec::new("mode", PropKind::Str, "how the element is prompted"),
        PropertySpec::new("mqtt-broker", PropKind::Str, "MQTT broker host:port"),
        PropertySpec::new("mqtt-topic", PropKind::Str, "MQTT topic to publish on"),
        PropertySpec::new("normalize", PropKind::Bool, "L2-normalize the embedding"),
        PropertySpec::new("num-frames", PropKind::Int, "frames one inference consumes"),
        PropertySpec::new("num-streams", PropKind::Int, "streams the element handles"),
        PropertySpec::new("output-dim", PropKind::Int, "embedding dimensions to keep"),
        PropertySpec::new("prompt", PropKind::Str, "prompt sent with each frame"),
        PropertySpec::new(
            "reference-path",
            PropKind::Str,
            "reference features to score against",
        ),
        PropertySpec::new("rules", PropKind::Str, "JSON rules an alert fires on"),
        PropertySpec::new("scale-factor", PropKind::Int, "upscaling factor"),
        PropertySpec::new(
            "schema-file",
            PropKind::Str,
            "JSON schema for published messages",
        ),
        PropertySpec::new("source-id", PropKind::Str, "id published with each message"),
        PropertySpec::new("speaker", PropKind::Str, "synthesized voice"),
        PropertySpec::new("src", PropKind::Str, "source language code"),
        PropertySpec::new("stem", PropKind::Str, "separated stem to output"),
        PropertySpec::new("streaming", PropKind::Bool, "emit as the stream arrives"),
        PropertySpec::new(
            "system-prompt",
            PropKind::Str,
            "system prompt for the model",
        ),
        PropertySpec::new("target", PropKind::Str, "target language code"),
        PropertySpec::new("temperature", PropKind::Double, "generation temperature"),
        PropertySpec::new("text", PropKind::Str, "text to embed"),
        PropertySpec::new(
            "threshold",
            PropKind::Double,
            "score a result needs to count",
        ),
        PropertySpec::new("timeout", PropKind::Int, "seconds to wait for a response"),
        PropertySpec::new("top-k", PropKind::Int, "how many top results to keep"),
        PropertySpec::new("topic", PropKind::Str, "Kafka topic to publish on"),
        PropertySpec::new("tracker-type", PropKind::Str, "tracking algorithm"),
        PropertySpec::new("tracking", PropKind::Bool, "draw track ids and trails"),
        PropertySpec::new(
            "translate",
            PropKind::Bool,
            "translate rather than transcribe",
        ),
        PropertySpec::new("url", PropKind::Str, "endpoint the element posts to"),
        PropertySpec::new(
            "visualize",
            PropKind::Bool,
            "render the result onto the frame",
        ),
        PropertySpec::new("webhook-url", PropKind::Str, "URL an alert posts to"),
        ]
    };
}

pub(crate) use hosted_element_props;

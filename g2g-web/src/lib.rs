//! Browser entry points for the glass2glass WebAssembly pipeline.
//!
//! Each `#[wasm_bindgen]` function wires a g2g graph out of `g2g-plugins`' wasm
//! elements and drives it on the browser event loop via `spawn_local`. The runner
//! future is executor agnostic (spin-based channels), so the browser drives it
//! exactly as tokio drives it natively, single threaded (no SharedArrayBuffer, so
//! no cross-origin-isolation headers are required to deploy this).
//!
//! This crate is a thin, deployable cdylib shim: all the real work lives in the
//! reusable elements (`WebSocketSrc`, `WebCodecsDecode`, `CanvasSink`, ...). It is
//! kept out of the workspace so its cdylib / std / wasm-bindgen footprint never
//! touches the no_std baseline. Build it for wasm32 only.

// Everything here is browser-only; guard the whole module so a stray native
// `cargo check` on this excluded crate is a clean no-op rather than a link error
// against the wasm32-gated elements.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{run_linear_chain, run_simple_pipeline, run_source_transform_sink};
use g2g_core::{Caps, Dim, Rate, VideoCodec};

use g2g_plugins::analyticsoverlay::AnalyticsOverlay;
use g2g_plugins::canvassink::CanvasSink;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::wasmclock::WasmClock;
use g2g_plugins::webcodecsdecode::WebCodecsDecode;
use g2g_plugins::websocketsrc::WebSocketSrc;

use portability_core::overlay_stages;
pub mod webortdetect;
use webortdetect::WebOrtDetect;

/// Install the panic hook once at module load, so a Rust panic surfaces as a
/// readable `console.error` instead of an opaque `unreachable` trap. Called
/// automatically by wasm-bindgen when the module initializes.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Log a pipeline's terminal result to the browser console, so a negotiation /
/// decode error (otherwise swallowed by the fire-and-forget entry points) is
/// visible. `n` is the graph name.
fn report<T: core::fmt::Debug>(n: &str, r: Result<T, g2g_core::G2gError>) {
    match r {
        Ok(stats) => web_sys::console::log_1(&JsValue::from_str(&format!(
            "g2g[{n}]: finished ok: {stats:?}"
        ))),
        Err(e) => web_sys::console::error_1(&JsValue::from_str(&format!(
            "g2g[{n}]: pipeline error: {e:?}"
        ))),
    }
}

/// H.264 elementary-stream caps with unknown geometry: the decoder derives the
/// real dimensions from the in-band SPS, announced mid-stream via `CapsChanged`.
/// Shared by every ingest entry point.
///
/// Geometry is a wide placeholder *Range*, not `Dim::Any`: negotiation must
/// fixate the link (Phase-2 `fixate()` rejects `Any`), and a Range fixates to
/// its minimum. Minimum 0 is deliberate, so the fixated 0x0 makes the decoder's
/// `width != 0` guard skip the coded-dims hint and let the browser
/// `VideoDecoder` size itself from the in-band SPS. Framerate mirrors
/// `RtspSrc`'s placeholder (1..240 fps). See the `intercept_caps must survive
/// fixate` design note.
fn h264_ingest_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Range { min: 0, max: 8192 },
        height: Dim::Range { min: 0, max: 8192 },
        framerate: Rate::Range { min_q16: 1 << 16, max_q16: 240 << 16 },
    }
}

/// Open `url`, treat the binary frames as an H.264 Annex-B elementary stream, and
/// run `WebSocketSrc -> FakeSink` to completion on the browser event loop. A
/// minimal ingest smoke entry (no decode); returns immediately, runs as a spawned
/// task.
#[wasm_bindgen]
pub fn run_websocket_ingest(url: String) {
    spawn_local(async move {
        let mut src = WebSocketSrc::new(url, h264_ingest_caps());
        let mut sink = FakeSink::new();
        let clock = WasmClock::new();
        let _ = run_simple_pipeline(&mut src, &mut sink, &clock, 8).await;
    });
}

/// Open `url`, decode the H.264 Annex-B access units it delivers with the browser
/// `VideoDecoder`, and run `WebSocketSrc -> WebCodecsDecode -> FakeSink`. The first
/// in-browser receive-to-decoded-pixels pipeline; expects one access unit per
/// WebSocket message, starting at a keyframe.
#[wasm_bindgen]
pub fn run_websocket_decode(url: String) {
    spawn_local(async move {
        let mut src = WebSocketSrc::new(url, h264_ingest_caps());
        let mut dec = WebCodecsDecode::new();
        let mut sink = FakeSink::new();
        let clock = WasmClock::new();
        let _ = run_source_transform_sink(&mut src, &mut dec, &mut sink, &clock, 8).await;
    });
}

/// WebSocket ingest -> WebCodecs decode -> canvas: the first in-browser
/// glass-to-glass receive pipeline. `canvas_id` is the id of an existing
/// `<canvas>`. Expects one H.264 access unit per WebSocket message, starting at a
/// keyframe.
#[wasm_bindgen]
pub fn run_websocket_to_canvas(url: String, canvas_id: String) {
    spawn_local(async move {
        let mut src = WebSocketSrc::new(url, h264_ingest_caps());
        let mut dec = WebCodecsDecode::new();
        let mut sink = CanvasSink::new(canvas_id);
        let clock = WasmClock::new();
        report("ws->decode->canvas", run_source_transform_sink(&mut src, &mut dec, &mut sink, &clock, 8).await);
    });
}

/// WebSocket ingest -> decode -> **detect** -> **overlay** -> canvas: the
/// in-browser object-detection chain (Architecture A, Stage 1). Runs a four-hop
/// linear graph `WebSocketSrc -> WebCodecsDecode -> WebDetect -> AnalyticsOverlay
/// -> CanvasSink` via `run_linear_chain` (cooperative on the single browser
/// thread, no worker), reusing the CPU box-draw overlay and 2D canvas sink
/// unchanged. `WebDetect` attaches synthetic detections decoded through the real
/// `DetectionPostprocess`; Stage 2 swaps its synthetic tensor for an `ort-web`
/// YOLOv8 run. Expects one H.264 access unit per WebSocket message, from a
/// keyframe. `canvas_id` is the id of an existing `<canvas>` with no prior WebGPU
/// context (the 2D path).
#[wasm_bindgen]
pub fn run_websocket_detect_to_canvas(url: String, canvas_id: String) {
    spawn_local(async move {
        let mut src = WebSocketSrc::new(url, h264_ingest_caps());
        let mut dec = WebCodecsDecode::new();
        // The detect + overlay stages come from the SHARED portability core, the
        // exact same `overlay_stages()` the native runner uses.
        let mut stages = overlay_stages(3);
        let mut sink = CanvasSink::new(canvas_id);
        let clock = WasmClock::new();
        let transforms: Vec<&mut dyn DynAsyncElement> =
            vec![&mut dec, &mut stages.detect, &mut stages.overlay];
        report(
            "ws->decode->detect->overlay->canvas",
            run_linear_chain(&mut src, transforms, &mut sink, &clock, 8).await,
        );
    });
}

/// WebSocket ingest -> decode -> **real ONNX YOLOv8 detect** -> overlay -> canvas
/// (Architecture A, Stage 2). Same four-hop linear chain as
/// [`run_websocket_detect_to_canvas`], but `WebDetect`'s synthetic box is replaced
/// by `WebOrtDetect`, which runs a real YOLOv8 model through ONNX Runtime Web
/// (loaded from the CDN by `ort-shim.js`) and decodes the output through the same
/// `DetectionPostprocess`. `model_url` is where the browser fetches the `.onnx`
/// (serve it same-origin, e.g. `models/yolov8n.onnx`). First frame is slow (model
/// download + ort-web wasm compile + session create); steady state is one CPU
/// inference per frame. `canvas_id` is a `<canvas>` with no prior WebGPU context.
#[wasm_bindgen]
pub fn run_websocket_ortdetect_to_canvas(url: String, canvas_id: String, model_url: String) {
    spawn_local(async move {
        let mut src = WebSocketSrc::new(url, h264_ingest_caps());
        let mut dec = WebCodecsDecode::new();
        let mut det = WebOrtDetect::new(model_url);
        let mut overlay = AnalyticsOverlay::new().with_thickness(3);
        let mut sink = CanvasSink::new(canvas_id);
        let clock = WasmClock::new();
        let transforms: Vec<&mut dyn DynAsyncElement> =
            vec![&mut dec, &mut det, &mut overlay];
        report(
            "ws->decode->ortdetect->overlay->canvas",
            run_linear_chain(&mut src, transforms, &mut sink, &clock, 4).await,
        );
    });
}

/// WebSocket ingest -> decode -> **remote detect** -> overlay -> canvas
/// (Architecture B: server-side inference, thin browser client). The same graph
/// as [`run_websocket_ortdetect_to_canvas`], but the detect element is the
/// **generic** `WsWireTransform` (M555): it ships each decoded frame to a native
/// peer at `detect_url` over the g2g-core wire codec and emits the processed frame
/// it returns (with `AnalyticsMeta` boxes attached), which `AnalyticsOverlay`
/// draws. This is the distributed-graph primitive, not a bespoke shim: the browser
/// element knows nothing about detection, and the peer (a `RemoteWsSrc`-style wire
/// server running the real `OrtInference` -> `DetectionPostprocess` chain) runs
/// whatever subgraph it likes. Inference moved off the browser by swapping one
/// generic element. `detect_url` is the server's WebSocket (e.g. `ws://127.0.0.1:9602`).
#[wasm_bindgen]
pub fn run_websocket_remotedetect_to_canvas(url: String, canvas_id: String, detect_url: String) {
    use g2g_plugins::wswiretransform::WsWireTransform;

    spawn_local(async move {
        let mut src = WebSocketSrc::new(url, h264_ingest_caps());
        let mut dec = WebCodecsDecode::new();
        let mut det = WsWireTransform::new(detect_url);
        let mut overlay = AnalyticsOverlay::new().with_thickness(3);
        let mut sink = CanvasSink::new(canvas_id);
        let clock = WasmClock::new();
        let transforms: Vec<&mut dyn DynAsyncElement> =
            vec![&mut dec, &mut det, &mut overlay];
        report(
            "ws->decode->remotedetect->overlay->canvas",
            run_linear_chain(&mut src, transforms, &mut sink, &clock, 4).await,
        );
    });
}

/// WebSocket ingest -> WebCodecs decode -> **wire-codec offload** to a native
/// subgraph (M554, the distributed-graph primitive over WebSocket). Decodes the
/// H.264 access units from `url` in the browser, then ships each decoded RGBA
/// frame to a native `RemoteWsSrc` at `wire_url` via `WsWireSink`, which
/// serializes the whole `PipelinePacket` stream with the *same* g2g-core wire
/// codec a native `RemoteWsSink` uses. The graph
/// (`WebSocketSrc -> WebCodecsDecode -> WsWireSink`) cuts its edge right after
/// decode and runs whatever tail the server wires behind `RemoteWsSrc` (e.g.
/// `-> detect -> ...`), so this is the media-agnostic generalization of the
/// bespoke M549 `WebRemoteDetect` shim: the browser no longer needs an element
/// that knows about detection, only the generic transport. `wire_url` is the
/// server's WebSocket (e.g. `ws://127.0.0.1:9601`).
#[wasm_bindgen]
pub fn run_websocket_decode_offload_to_wire(url: String, wire_url: String) {
    use g2g_plugins::wswiresink::WsWireSink;

    spawn_local(async move {
        let mut src = WebSocketSrc::new(url, h264_ingest_caps());
        let mut dec = WebCodecsDecode::new();
        let mut sink = WsWireSink::new(wire_url);
        let clock = WasmClock::new();
        report(
            "ws->decode->wire-offload",
            run_source_transform_sink(&mut src, &mut dec, &mut sink, &clock, 8).await,
        );
    });
}

/// WebSocket ingest -> WebCodecs decode -> **WebGPU** canvas: the zero-copy path.
/// The decoder keeps each frame GPU-resident (`with_gpu_output`) and the sink imports
/// it as a `GPUExternalTexture` and renders it, with no CPU readback (the browser
/// analog of the native Vulkan Video -> wgpu wedge). `canvas_id` must be a `<canvas>`
/// with no prior 2D context (a canvas' context type is fixed on first acquisition).
#[wasm_bindgen]
pub fn run_websocket_to_webgpu_canvas(url: String, canvas_id: String) {
    use g2g_plugins::webgpucanvassink::WebGpuCanvasSink;

    spawn_local(async move {
        let mut src = WebSocketSrc::new(url, h264_ingest_caps());
        let mut dec = WebCodecsDecode::new().with_gpu_output();
        let mut sink = WebGpuCanvasSink::new(canvas_id);
        let clock = WasmClock::new();
        report("ws->decode->webgpu", run_source_transform_sink(&mut src, &mut dec, &mut sink, &clock, 8).await);
    });
}

/// Browser EGRESS: synthetic capture -> WebCodecs encode -> WebSocket send.
/// Generates an animated 320x240 RGBA test pattern, encodes it to H.264 Annex-B with
/// the browser `VideoEncoder`, and streams each access unit to `url` over a WebSocket
/// (`PatternSrc -> WebCodecsEncode -> WebSocketSink`). A native receiver can save the
/// bytes as a playable `.h264`. Runs 150 frames (~10 s at 15 fps) then closes.
#[wasm_bindgen]
pub fn run_pattern_encode_to_websocket(url: String) {
    use g2g_plugins::patternsrc::PatternSrc;
    use g2g_plugins::webcodecsencode::WebCodecsEncode;
    use g2g_plugins::websocketsink::WebSocketSink;

    spawn_local(async move {
        let mut src = PatternSrc::new(320, 240, 15, 150);
        let mut enc = WebCodecsEncode::new();
        let mut sink = WebSocketSink::new(url);
        let clock = WasmClock::new();
        report("pattern->encode->ws", run_source_transform_sink(&mut src, &mut enc, &mut sink, &clock, 8).await);
    });
}

/// WebSocket ingest -> WebCodecs decode -> **WebGPU inference**: the zero-copy
/// GPU-ML path. The decoder keeps each frame GPU-resident and `WebGpuCanvasSink`
/// (in `with_inference` mode) runs a per-pixel nearest-centroid classifier over it
/// with no CPU readback, presenting the class map. The browser analog of the native
/// GPU-ML pipeline (decode -> GPU compute -> result); a full CNN/ONNX head slots in
/// where the classifier WGSL is.
#[wasm_bindgen]
pub fn run_websocket_to_webgpu_inference(url: String, canvas_id: String) {
    use g2g_plugins::webgpucanvassink::WebGpuCanvasSink;

    spawn_local(async move {
        let mut src = WebSocketSrc::new(url, h264_ingest_caps());
        let mut dec = WebCodecsDecode::new().with_gpu_output();
        let mut sink = WebGpuCanvasSink::new(canvas_id).with_inference();
        let clock = WasmClock::new();
        report("ws->decode->webgpu-infer", run_source_transform_sink(&mut src, &mut dec, &mut sink, &clock, 8).await);
    });
}

/// WebSocket ingest -> WebCodecs decode -> **WebGPU CNN**: a real 2-layer
/// convolutional network (conv3x3 -> ReLU -> conv3x3) run over each decoded frame,
/// zero-copy from the GPU external texture, presenting its feature map. The browser
/// GPU-ML wedge with actual conv layers; trained weights / deeper nets slot into the
/// same shader.
#[wasm_bindgen]
pub fn run_websocket_to_webgpu_cnn(url: String, canvas_id: String) {
    use g2g_plugins::webgpucanvassink::WebGpuCanvasSink;

    spawn_local(async move {
        let mut src = WebSocketSrc::new(url, h264_ingest_caps());
        let mut dec = WebCodecsDecode::new().with_gpu_output();
        let mut sink = WebGpuCanvasSink::new(canvas_id).with_cnn();
        let clock = WasmClock::new();
        report("ws->decode->webgpu-cnn", run_source_transform_sink(&mut src, &mut dec, &mut sink, &clock, 8).await);
    });
}

/// Browser EGRESS from a real camera: `getUserMedia` capture -> WebCodecs encode ->
/// WebSocket send. Opens a `width` x `height` webcam via `WebCameraSrc`, encodes to
/// H.264 Annex-B, and streams each access unit to `url`
/// (`WebCameraSrc -> WebCodecsEncode -> WebSocketSink`) so a native receiver can save
/// a playable `.h264`. Requires camera permission (a secure context + user grant, or
/// a fake device). Runs `frames` frames (0 = until the track ends) then closes.
#[wasm_bindgen]
pub fn run_camera_encode_to_websocket(url: String, width: u32, height: u32, frames: f64) {
    use g2g_plugins::webcamerasrc::WebCameraSrc;
    use g2g_plugins::webcodecsencode::WebCodecsEncode;
    use g2g_plugins::websocketsink::WebSocketSink;

    spawn_local(async move {
        let mut src = WebCameraSrc::new(width, height, frames as u64);
        let mut enc = WebCodecsEncode::new();
        let mut sink = WebSocketSink::new(url);
        let clock = WasmClock::new();
        report("camera->encode->ws", run_source_transform_sink(&mut src, &mut enc, &mut sink, &clock, 8).await);
    });
}

/// WebRTC data-channel ingest -> WebCodecs decode -> canvas. `channel` is an
/// already-open `RtcDataChannel` the application negotiated; same decode and
/// presentation path as [`run_websocket_to_canvas`].
#[wasm_bindgen]
pub fn run_datachannel_to_canvas(channel: web_sys::RtcDataChannel, canvas_id: String) {
    use g2g_plugins::webrtcsrc::WebRtcSrc;

    spawn_local(async move {
        let mut src = WebRtcSrc::new(channel, h264_ingest_caps());
        let mut dec = WebCodecsDecode::new();
        let mut sink = CanvasSink::new(canvas_id);
        let clock = WasmClock::new();
        report("ws->decode->canvas", run_source_transform_sink(&mut src, &mut dec, &mut sink, &clock, 8).await);
    });
}

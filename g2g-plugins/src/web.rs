//! Browser entry points: wire a wasm pipeline and drive it on the event loop
//! via `spawn_local`. The runner future is executor-agnostic (spin-based
//! channels), so the browser drives it exactly as tokio drives it natively.

use alloc::string::String;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;

use g2g_core::runtime::run_simple_pipeline;
use g2g_core::{Caps, Dim, Rate, VideoCodec};

use crate::fakesink::FakeSink;
use crate::wasmclock::WasmClock;
use crate::websocketsrc::WebSocketSrc;

/// Open `url`, treat the binary frames as an H.264 Annex-B elementary stream,
/// and run `WebSocketSrc -> FakeSink` to completion on the browser event loop.
/// Returns immediately; the pipeline runs as a spawned task. A minimal smoke
/// entry for M39; decode + canvas pipelines land in M40/M41.
#[wasm_bindgen]
pub fn run_websocket_ingest(url: String) {
    spawn_local(async move {
        let caps = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let mut src = WebSocketSrc::new(url, caps);
        let mut sink = FakeSink::new();
        let clock = WasmClock::new();
        let _ = run_simple_pipeline(&mut src, &mut sink, &clock, 8).await;
    });
}

/// Open `url`, decode the H.264 Annex-B access units it delivers with the
/// browser `VideoDecoder`, and run `WebSocketSrc -> WebCodecsDecode -> FakeSink`
/// on the browser event loop. The first in-browser receive-to-decoded-pixels
/// pipeline (M40); a canvas sink lands in M41. Requires the stream to send one
/// access unit per WebSocket message, starting at a keyframe.
#[cfg(feature = "web-codecs")]
#[wasm_bindgen]
pub fn run_websocket_decode(url: String) {
    use crate::webcodecsdecode::WebCodecsDecode;
    use g2g_core::runtime::run_source_transform_sink;

    spawn_local(async move {
        let caps = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let mut src = WebSocketSrc::new(url, caps);
        let mut dec = WebCodecsDecode::new();
        let mut sink = FakeSink::new();
        let clock = WasmClock::new();
        let _ = run_source_transform_sink(&mut src, &mut dec, &mut sink, &clock, 8).await;
    });
}

/// WebSocket ingest -> WebCodecs decode -> canvas: the first in-browser
/// glass-to-glass receive pipeline (M41). `canvas_id` is the id of an existing
/// `<canvas>` element. Expects one H.264 access unit per WebSocket message,
/// starting at a keyframe.
#[cfg(feature = "web-codecs")]
#[wasm_bindgen]
pub fn run_websocket_to_canvas(url: String, canvas_id: String) {
    use crate::canvassink::CanvasSink;
    use crate::webcodecsdecode::WebCodecsDecode;
    use g2g_core::runtime::run_source_transform_sink;

    spawn_local(async move {
        let caps = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let mut src = WebSocketSrc::new(url, caps);
        let mut dec = WebCodecsDecode::new();
        let mut sink = CanvasSink::new(canvas_id);
        let clock = WasmClock::new();
        let _ = run_source_transform_sink(&mut src, &mut dec, &mut sink, &clock, 8).await;
    });
}

/// WebRTC data-channel ingest -> WebCodecs decode -> canvas (M42). `channel` is
/// an already-open `RtcDataChannel` the application negotiated; same decode and
/// presentation path as `run_websocket_to_canvas`.
#[cfg(feature = "web-codecs")]
#[wasm_bindgen]
pub fn run_datachannel_to_canvas(channel: web_sys::RtcDataChannel, canvas_id: String) {
    use crate::canvassink::CanvasSink;
    use crate::webcodecsdecode::WebCodecsDecode;
    use crate::webrtcsrc::WebRtcSrc;
    use g2g_core::runtime::run_source_transform_sink;

    spawn_local(async move {
        let caps = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let mut src = WebRtcSrc::new(channel, caps);
        let mut dec = WebCodecsDecode::new();
        let mut sink = CanvasSink::new(canvas_id);
        let clock = WasmClock::new();
        let _ = run_source_transform_sink(&mut src, &mut dec, &mut sink, &clock, 8).await;
    });
}

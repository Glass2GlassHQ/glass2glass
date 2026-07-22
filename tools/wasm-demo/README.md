# glass2glass browser pipeline demo

A runnable, end-to-end proof of the g2g WebAssembly pipeline: a native WebSocket
server streams an H.264 fixture to the browser, where a g2g graph compiled to wasm
(`WebSocketSrc → WebCodecsDecode → CanvasSink`) decodes it with the browser's own
hardware decoder (WebCodecs) and presents it to a `<canvas>`.

The whole pipeline is **single-threaded** (driven on the browser event loop via
`spawn_local`), so unlike thread-based wasm ports (e.g. gst.wasm, which needs
`SharedArrayBuffer`) this requires **no cross-origin-isolation (COOP/COEP) headers** to
deploy. The wasm artifact is ~200 KB (it ships no codec: WebCodecs uses the browser's).

## Prerequisites

- `rustup target add wasm32-unknown-unknown`
- `cargo install wasm-pack`
- `python3` (for the static file server) and a WebCodecs-capable browser (Chrome/Edge;
  Firefox with `dom.media.webcodecs.enabled`).

## Run (three steps)

```sh
# 1. Build the wasm module + JS glue into ./pkg
./build.sh

# 2. Stream the fixture over WebSocket (one access unit per message, looping)
cargo run --release --manifest-path ws-fixture-server/Cargo.toml
#    -> serving ws://127.0.0.1:8080

# 3. Serve this directory and open it
./serve.sh          # -> http://127.0.0.1:8000/
```

Open <http://127.0.0.1:8000/>, pick a presentation mode, click **Start**:

- **Canvas 2D (readback)** &mdash; `WebCodecsDecode` copies each frame to system RGBA;
  `CanvasSink` paints it with `putImageData`.
- **WebGPU (zero-copy)** &mdash; `WebCodecsDecode` keeps the frame GPU-resident
  (`with_gpu_output`); `WebGpuCanvasSink` imports the `VideoFrame` as a
  `GPUExternalTexture` and samples it in a render pass, **no CPU readback** (the
  browser analog of the native Vulkan&nbsp;Video&rarr;wgpu wedge). Needs a
  WebGPU-capable browser (`navigator.gpu`).

## Send (browser encode &rarr; egress)

The reverse direction: the browser captures, encodes, and sends. Click **Send** on
the page (or call `run_pattern_encode_to_websocket(url)`):
`PatternSrc` (an animated RGBA test pattern) &rarr; `WebCodecsEncode` (browser
`VideoEncoder`, H.264 Annex-B) &rarr; `WebSocketSink` (one access unit per message).
Run the receiver to capture it as a playable file:

```sh
cargo run --release --manifest-path ws-recv-server/Cargo.toml   # ws://127.0.0.1:8081
# click Send; when it finishes:
ffplay received.h264          # or: vlc received.h264
```

(A real camera source via getUserMedia + `MediaStreamTrackProcessor` is a follow-up;
the synthetic pattern keeps the encode + egress path self-contained.)

## Object detection (in-browser and server-side)

The detect modes add a detection stage to the linear chain
(`decode → detect → overlay → canvas`, one graph, the detect element swapped):

- **synthetic box** &mdash; `WebDetect` attaches a fixed box decoded through the
  real `g2g-ml` `DetectionPostprocess` (proves the g2g half; no model).
- **real YOLOv8 / ort-web** &mdash; `WebOrtDetect` runs a real ONNX YOLOv8n in the
  browser via ONNX Runtime Web (loaded from the CDN, single-threaded), decoding
  the output through the same `DetectionPostprocess` (Architecture A).
- **server-side / remote** &mdash; `WebRemoteDetect` ships each decoded frame to the
  native `detect-server`, which runs the **real g2g native chain**
  (`OrtInference → ONNX Runtime → DetectionPostprocess`) and returns the boxes.
  Same graph, inference moved off the browser (Architecture B).

```sh
# Fetch the model + build a sample clip with real objects (git-ignored, ~13 MB):
./get-detection-assets.sh

# Stream the sample clip instead of the default fixture:
cargo run --release --manifest-path ws-fixture-server/Cargo.toml -- 127.0.0.1:8080 models/bus_640.h264 15

# For the server-side mode, also run the detection server (native ONNX Runtime):
cargo run --release --manifest-path detect-server/Cargo.toml -- 127.0.0.1:8090 models/yolov8n.onnx
```

Then pick a detect mode and **Start**. The in-browser YOLO runs at a few fps
(CPU-wasm); the server-side path keeps up with the stream (native ORT) and is the
"thin client, offload the model" shape.

### Headless test (ort-web MVP)

`headless/run-ortdetect.mjs` drives the ort-web chain end to end in a
WebCodecs-capable Chromium against a committed, deterministic model fixture
(`fixtures/tiny-detect.onnx`, ~0.6 KB, regenerable via
`uv run --with onnx --with numpy fixtures/gen-tiny-detect.py`). The fixture plants
exactly two detections per frame, so the test asserts the model loads, every frame
yields two decoded detections, and the overlay boxes render to the canvas. It needs
`playwright` (`npm i -D playwright`) and a full Chromium; env overrides `G2G_CHROME`,
`G2G_WS_SERVER_BIN` (prebuilt `ws-fixture-server`, else `cargo run`), `G2G_FIXTURE`.

```sh
./build.sh                        # build pkg/ first
node headless/run-ortdetect.mjs   # -> PASS ...
```

## Pieces

| File | Role |
| :--- | :--- |
| `../../g2g-web/` | the deployable cdylib: `#[wasm_bindgen]` entry points wiring g2g elements |
| `build.sh` | `wasm-pack build g2g-web --target web` into `pkg/` (sets the WebCodecs unstable cfg) |
| `index.html` | canvas + `run_websocket_to_canvas(url, "video")` |
| `ws-fixture-server/` | native tokio + tokio-tungstenite server; splits the Annex-B fixture into access units and streams them on a timer (the receive demo's source) |
| `ws-recv-server/` | native receiver for the send demo; appends the browser's encoded access units to `received.h264` |
| `serve.sh` | `python3 -m http.server` (no special headers) |
| `fixtures/` | committed tiny deterministic ONNX detector (`tiny-detect.onnx`) + `gen-tiny-detect.py` |
| `headless/` | `run-ortdetect.mjs` + `ortdetect.html`: headless validation of the ort-web chain |

## Notes

- The fixture defaults to `g2g-plugins/tests/fixtures/h264_640x480.h264`; pass another
  path (and fps) as args to the server.
- `WebCodecsDecode` configures lazily from the first in-band SPS, so the stream must
  start on a keyframe (the server loops from the IDR, so it does).
- Other entry points in `g2g-web` (all exported into `pkg/g2g_web.js`):
  `run_websocket_to_webgpu_canvas` (the zero-copy WebGPU path above),
  `run_websocket_decode` (decode to a fakesink), `run_websocket_ingest` (ingest only),
  and `run_datachannel_to_canvas` (WebRTC data-channel ingest instead of WebSocket).

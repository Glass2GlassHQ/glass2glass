# Portability: one pipeline, five targets

g2g's core is pure Rust, `no_std + alloc`, and sans-IO, so the *same typed graph*
runs unchanged across the whole hardware spectrum: **MCU · RTOS · CPU · GPU ·
WASM**. The graph, the `AsyncElement` traits, the `Caps` negotiation, and the
`run_graph` runner are identical on every target; only the *deployment shell*
(which executor, which source/sink) changes.

This document is the proof, not the claim: each target below is backed by code
you can run.

## The showcase pipeline

A detection-overlay pipeline: `source → detect → overlay → sink`. The two
processing stages are the portability core:

- `SyntheticDetect` — attaches detection metadata, decoded through the real
  `g2g-ml` `DetectionPostprocess` (a planted box, so no model is needed and the
  result is identical everywhere; swap in `OrtInference` / ort-web / a remote
  server without touching the graph, see the Architecture A/B detection arc).
- `AnalyticsOverlay` — draws the boxes onto the RGBA frame (CPU, `no_std`).

Both come from a **single shared definition**,
[`portability_core::overlay_stages()`](examples/portability-core/src/lib.rs).
The native runner and the browser build construct their middle-of-graph from that
exact function; only the source and sink differ:

| | source | processing (shared) | sink |
| :--- | :--- | :--- | :--- |
| **CPU** (`portability-native`) | `VideoTestSrc` | `overlay_stages()` | `FileSink` |
| **WASM** (`g2g-web`) | `WebSocketSrc → WebCodecsDecode` | `overlay_stages()` | `CanvasSink` |

Same `SyntheticDetect`, same `AnalyticsOverlay`, same `run_linear_chain` runner,
compiled once for native and once for `wasm32`.

## The proof matrix

| Target | What was verified | Evidence |
| :--- | :--- | :--- |
| **MCU** | the `no_std + alloc` core (caps algebra, `Frame`, the no-alloc `StaticLendRing` DMA capture ring, the bounded channel) compiles to bare-metal Cortex-M | **~1.4 KB** of g2g `.text` (`examples/g2g-size` for `thumbv7em-none-eabihf`; the exercised `g2g_min` slice is 916 B) |
| **RTOS** | the graph runs on the Embassy executor with `embassy-sync` stack channels | `m43_embassy` passes |
| **CPU** | `VideoTestSrc → overlay_stages() → FileSink` renders one annotated frame | native runner writes a 640×480 RGBA frame with the box drawn at `[0.25, 0.25, 0.5, 0.5]` |
| **GPU** | frames processed GPU-resident (wgpu on the RTX 3060), matching the CPU reference | `wgpu_preprocess` 4/4 (NV12 → f32 NCHW tensor, no CPU round-trip); the zero-copy Vulkan Video → `wgpu::Texture` decode path is validated for H.264/H.265/AV1 |
| **WASM** | the **same** `overlay_stages()` runs in a real browser: `WebSocketSrc → WebCodecsDecode → overlay_stages() → CanvasSink` | headless-Chromium harness draws the box over the decoded video (box-border pixels are the exact overlay palette where the plain path shows video) |

## Spike: QNX (safety-certified RTOS)

Not a matrix row yet (compile-checked, not run), but a Tier-0 portability spike
for the ISO 26262 / IEC 62304 automotive/medical market where QNX is the
reference platform. QNX is a POSIX microkernel on application processors
(aarch64 / x86-64), so this is the `std`-capable path, not the MCU one. Verified
locally (no QNX SDP needed, `cargo +nightly ... -Zbuild-std=core,alloc`, for
both `aarch64-unknown-nto-qnx800` and `x86_64-pc-nto-qnx800`):

- **Compiles today, zero code changes:** `g2g-core` (the no-alloc subset **and**
  the full `alloc` + dynamic `runtime` layer: caps solver, autoplug, dynamic
  `Graph`), `g2g-mcu` (the whole peripheral catalog), and the `g2g-plugins`
  `no_std` baseline. The portable pure-Rust surface is QNX-ready as-is.
- **Why it's clean:** every OS/HW element is gated by a *specific* `target_os`
  (`"linux"` / `"windows"` / `"macos"` / `"android"`), never `cfg(unix)`, so the
  Linux HW paths (VAAPI, DRM/KMS, dma-buf, v4l2, ALSA/PipeWire) are *excluded* on
  `nto`, not accidentally pulled in.
- **Tier 1 (needs the QNX SDP, no g2g blocker in sight):** the `std` elements
  (file I/O, the RTP/RTSP/SRT transports). `std` pulls `tokio`, so "does `tokio`
  build on QNX 8" is the one dependency question to settle.
- **Tier 2 (needs an SoC + partner):** QNX-native HW, a QNX Screen display sink,
  the vendor VPU (bridged through the M650 C-seam), GPU. New `target_os = "nto"`
  elements.

## Reproduce

```sh
# CPU: the native runner (shared overlay_stages), writes an annotated frame.
cd examples/portability-native
cargo run --release -- annotated.rgba 640 480
ffmpeg -f rawvideo -pix_fmt rgba -s 640x480 -i annotated.rgba annotated.png   # view

# WASM: the same overlay_stages in the browser.
bash tools/wasm-demo/build.sh
cargo run --release --manifest-path tools/wasm-demo/ws-fixture-server/Cargo.toml
tools/wasm-demo/serve.sh   # open http://127.0.0.1:8000/ , pick "Detect (synthetic box)"

# MCU: the no_std core footprint on Cortex-M.
cd examples/g2g-size
cargo build --release --target thumbv7em-none-eabihf
size target/thumbv7em-none-eabihf/release/libg2g_size.a

# RTOS: the Embassy executor smoke.
cargo test -p g2g-plugins --features embassy --test m43_embassy

# QNX (Tier-0 spike): the portable core compiles for the QNX ARM target
# (nightly + build-std; no QNX SDP required for this non-linking lib check).
cargo +nightly check -p g2g-core --features alloc,runtime \
  --target aarch64-unknown-nto-qnx800 -Zbuild-std=core,alloc

# GPU: wgpu preprocess on the device (serialize device creation).
cargo test -p g2g-ml --features wgpu --test wgpu_preprocess -- --test-threads=1
```

## Why it holds

The processing loop needs only `core` and `alloc`; every OS- or hardware-coupled
element lives behind a cargo feature. So the target is a property of the
*deployment binary*, not the graph. That is the whole thesis: you write the
pipeline once and move a stage between an MCU, an edge box, a GPU server, or the
browser (even across a network boundary, see the remote-detection server) by
swapping the source/sink or a single element, never the graph.

See [DESIGN.md](DESIGN.md) §1 and the [README](README.md#portability-one-pipeline-five-targets).

# DESIGN_TODO

Open work tracked against the architecture in [DESIGN.md](DESIGN.md). Items
here are deferrals from the spec, follow-ups blocked on a concrete driver or
upstream fix, and forward-looking tracks that the current architecture
anticipates but hasn't yet built.

## Negotiation (DESIGN.md §4.13)

- **Forward coordinator re-solve walk** (Caps-β). The current mid-stream
  re-solve uses a startup downstream-feasibility snapshot
  (`solver::downstream_feasibility`). A downstream `DerivedOutput` element
  that must re-derive on a mid-stream input change isn't covered — its
  envelope was snapshotted against the startup input. The forward
  coordinator walk (request/reply through each arm, gathering each
  element's current constraint contribution) is the missing piece; design
  is settled but build is gated on a real driver where a downstream
  decoder sits below another format-changing transform.
- **Dynamic pads / request pads.** `tee::request-pad` and
  `mux::request-pad` style runtime branch/input addition. The fan-out and
  muxer are currently static.
- **Mid-stream element hot-swap.** `ElementSlot::swap` scaffolding exists
  (§4.8.2); the live swap of a real element under load isn't wired up.
- **Preference algebra.** `CapsPreferences` is a placeholder; the solver
  uses constraint-internal ordering for tie-breaks. A real
  competing-constraint scenario is needed to drive a concrete preference
  algebra (sum-of-indices is the placeholder).

## Receive / decode (DESIGN.md §4.11)

- **VaapiH264Dec on AMD desktop.** cros-codecs hard-codes a 16×16 initial
  `VAContext` and uses ChromeOS-specific GBM flags
  (`GBM_BO_USE_HW_VIDEO_DECODER`, `NV12` contiguous). Both fail on the
  Mesa `radeonsi` GBM provider. The clean fix is upstream: a cros-codecs
  surface backend that allocates VAAPI surfaces directly through libva
  (`vaCreateSurfaces`) instead of routing through GBM. Until then, `ffmpeg`
  is the Linux AMD path.
- **ffmpeg VAAPI hwaccel.** Open the `h264_vaapi` codec with an attached
  `AVHWDeviceContext(VAAPI)`, register a `get_format` callback claiming
  `AV_PIX_FMT_VAAPI`, and `av_hwframe_transfer_data` the decoded surface
  into `System` memory. Stays inside `FfmpegH264Dec`; the public
  `AsyncElement` shape doesn't change. Useful on Intel iGPUs and AMD
  desktop while the cros-codecs upstream fix is pending.
- **Zero-copy `MemoryDomain::DmaBuf` from `VaapiH264Dec`.** The
  GBM-allocated surface is already a DMA-buf; exposing its fd via
  `OwnedDmaBuf` needs a refcount story to keep the surface alive until
  downstream releases it.
- **H.265 in `VaapiH264Dec`.** The cros-codecs stateless framework supports
  it; a sibling element keyed on `VideoCodec::H265` is straightforward.
- **Upstream `Reconfigure` driven by `VaapiH264Dec` `FormatChanged`.**
  Resolution change is observed (`DecoderEvent::FormatChanged` → fresh
  `CapsChanged` downstream) but not yet plumbed as an upstream
  `Reconfigure`.
- **`MfDecode` zero-copy + DXVA.** D3D11 zero-copy output via the
  `MemoryDomain::D3D11Texture` variant, DXVA hardware acceleration
  (`MF_SA_D3D11_AWARE`), strided NV12 output. The software-decoder path
  currently assumes `stride == width`.
- **10-bit pixel formats in `FfmpegH264Dec`.** `YUV420P10` / `P010`.
  4:4:4 is accepted with chroma box-averaged down to 4:2:0; the 10-bit
  layout is endianness / bit-position-specific and was not added without
  a libav host to verify on.

## CUDA / display (DESIGN.md §4.11.5)

- **`CudaGlSink` first compile + e2e.** The sink draft is in tree (EGL on a
  Wayland surface via `wl_egl_window`, `glow` GL ES 3 program, NV12 shader,
  per-frame map/copy/unmap via the CUDA-GL interop FFI), but was authored
  off-Linux. The first compile pass on Linux+NVIDIA and the manual
  `wayland_smoke`-style benchmark (`rtspsrc → h264parse →
  ffmpegdec[NvdecCuda] → CudaGlSink`) versus the `NvdecCuvid → WaylandSink`
  system-memory baseline are owed.
- **GL-on-KMS variant of `CudaGlSink`.** Wayland is the dev-loop path; KMS
  / GBM is the production tty path. Re-uses the CUDA-GL interop core; only
  the windowing changes.
- **CUDA ↔ Vulkan external memory.** Importing a Vulkan image's memory
  into CUDA via `cudaImportExternalMemory` is the long-term direction; the
  GL path is the pragmatic first deliverable.
- **Real downstream consumer for the β allocation re-cascade.** The
  in-tree decoders record the sink's mid-stream proposal but the MFT and
  CUDA output pools are fixed at codec open; a pool that actually re-sizes
  on the mid-stream proposal exercises the cascade end-to-end. (The cascade
  itself is built and covered by a fake transform.)

## Egress (DESIGN.md §4.12)

- **RTCP sender reports.** RFC 3550 SR generation on the existing
  `UdpSink`.
- **RTSP `ANNOUNCE` / `RECORD` ingest.** The Wowza-style egress
  handshake. Sandbox blocks port 554, so this is bring-up + manual
  validation, not CI-testable.

## Embedded (DESIGN.md §6.2.1)

- **`EmbassyClock` HAL tick.** The tick is selected at the cargo feature;
  driving it on real hardware needs a HAL time driver. Host verification
  via the `block_on` pipeline is in place.
- **Full `embassy-executor` multi-task integration.** Pipelines run today
  under `embassy-futures::block_on`; the multi-task executor path uses
  the same runner futures.
- **Fixed DMA-ring capture `SourceLoop`.** A no-alloc end-to-end frame
  flow: a lifetime-carrying `SystemSlice` wires `StaticBufferPool` into
  the zero-copy path. This is the last piece of the strict no-heap
  embedded story.

## Browser / Wasm (DESIGN.md §6.3.1)

- **In-browser runtime validation.** `WebSocketSrc → H264Parse →
  WebCodecsDecode → CanvasSink` compiles for `wasm32-unknown-unknown` but
  the live `WebSocket` receive + `performance.now()` pacing is owed a
  `wasm-bindgen-test` (or manual) run.
- **WebGPU-texture zero-copy sink.** `MemoryDomain::WebGPUBuffer` into a
  `GPUTexture` needs the async device handshake to live in the
  `OwnedWebGPUBuffer` keep-alive (`request_adapter` / `request_device`
  are async, `configure_pipeline` is not).
- **Web Workers executor.** `spawn_local` drives pipelines on the main
  thread; off-main-thread Workers need JS bootstrap infrastructure.
- **HEVC in `WebCodecsDecode`.** The hook is parameterized; the
  per-codec setup parallels the H.264 path.

## ML (DESIGN.md §5)

- **ONNX import via `burn-import`.** Build-time codegen; the
  `BurnInference` `AsyncElement` shape is what trained-weight modules
  slot into.
- **Trained-weight `Module` path for `BurnInference`.** Richer layers
  (conv, attention) once the codegen lands.
- **Decoder DMA-BUF / D3D11 surface import into `WgpuPreprocess`.** Today
  `WgpuPreprocess` uploads NV12 to a storage buffer; binding a decoder's
  surface directly into the compute pass needs the surface-import
  handshake and a GPU tensor `MemoryDomain` for the output.

## Documentation

- Architecture diagrams in [docs/](docs/) (the Pages site is text-only).
- Per-element rustdoc pass: ensure every public element type has an
  example block.

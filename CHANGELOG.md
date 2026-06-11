# Changelog

Pre-release. Work is tracked by milestone (Mn) following the roadmap in `DESIGN.md` §4.10.
Nothing is published yet; all versions are `0.1.0`.

## Unreleased

### M18 item 6: pad templates for the Windows decode/display elements

- `MfDecode` and `D3D11Sink` now implement `PadTemplates`, so a tool can
  introspect their static caps and check link compatibility before either is
  constructed (`gst_element_factory_get_static_pad_templates` analog). Extends
  the existing coverage (`VideoTestSrc` / `FakeSink` / `H264Parse`) to the
  Windows GPU path. `MfDecode`: H.264 sink pad + NV12 source pad (the memory
  domain is not encoded in caps, so the templates are backend-independent).
  `D3D11Sink`: a terminal NV12 sink pad, no source pad.
- New integration test `windows_decode_to_display_chain_links_by_type`
  (gated on `mf-decode` + `d3d11-sink`) proves the whole chain is
  introspectable pre-instantiation: `H264Parse -> MfDecode -> D3D11Sink` all
  link by type, while an RGBA source is correctly rejected at the decoder.
  Plus element-local unit tests for each template. VERIFIED on the Windows dev
  host: `cargo test -p g2g-plugins --features "mf-decode d3d11-sink"` (34 lib +
  the chain test) and clippy green; default workspace unaffected.

### W1: allocation-query handshake for the D3D11 path

- Mirrors C3 step 3 on the Windows side, completing the W1 <-> C3 symmetry.
  `D3D11Sink::propose_allocation` returns `AllocationParams::d3d11(...)`: a
  `MemoryDomainKind::D3D11Texture` proposal sized to the NV12 frame
  (`w*h*3/2`, even dims guaranteed by the sink), with pool headroom
  (`min_buffers = 3`) and 256-byte alignment. The runner conveys it to the
  upstream decoder's `configure_allocation`.
- `MfDecode::configure_allocation` records the proposal (`requested_alloc()`
  accessor), like `FfmpegH264Dec`. On the `with_d3d11` path a texture request
  is honoured by construction (it already emits `D3D11Texture` frames); the
  software path can't satisfy it, so the request stays recorded for diagnostics
  rather than silently changing the output domain.
- Two GPU-free unit tests (`D3D11Sink` proposes the right size/align/headroom;
  `MfDecode` records a conveyed D3D11 proposal). VERIFIED on the Windows dev
  host: `cargo test -p g2g-plugins --features "mf-decode d3d11-sink"` (32
  passed) and clippy green; the platform-agnostic conveyance stays covered by
  `m12_allocation.rs`.

### W1 (Phase 4): `D3D11Sink` present sink

- Completes the Windows zero-copy decode -> display track. `D3D11Sink`
  (`d3d11-sink` feature, Windows-only) consumes `MemoryDomain::D3D11Texture`
  frames from `MfDecode::with_d3d11()` and presents them in a Win32 window via
  a DXGI flip-model swapchain. The NV12 -> RGB colour convert runs on the GPU
  through a D3D11 video processor (`VideoProcessorBlt`), so the decoded texture
  never leaves the GPU. The Windows analog of `CudaGlSink`.
- Same worker-thread model as `WaylandSink` / `CudaGlSink`: a dedicated thread
  owns the window + message pump + D3D11 objects (both thread-affine); the sink
  struct holds only `Send` handles (an mpsc sender + atomics). The decoded
  `OwnedD3D11Texture` is `Send`, so it crosses to the worker and the texture
  (and its owning `IMFSample`) stays pinned until presented. NV12-in-D3D11
  only (`UnsupportedDomain` otherwise); per-frame ack gives backpressure;
  mid-stream geometry change respawns the worker.
- The swapchain + video processor are created lazily on the first frame from
  that frame's `ID3D11Device` (the decoder's device), since a D3D11 resource
  and the views over it must share a device, avoiding a second device + texture
  sharing. The window is created up front (no device needed).
- All COM (D3D11 video device/context/processor/views, the input/output view
  descriptors, the DXGI swapchain, and the Win32 window + message loop) was
  verified against the fetched `windows-0.62.2` source per AGENTS.md. Adds
  `windows` features `Win32_UI_WindowsAndMessaging`, `Win32_System_LibraryLoader`,
  `Win32_Graphics_Gdi`.
- Three GPU-free unit tests (intercept pass-through, non-NV12 reject, odd-dim
  reject). VERIFIED on the Windows dev host: `cargo test -p g2g-plugins
  --features d3d11-sink` (19 passed, the full present path COMPILES) and
  `cargo clippy --features d3d11-sink` green. The actual present (real GPU
  decode into textures shown in a window) is owed as a user-side run on a GPU
  machine; the dev host can do it. Acceptance test: `rtspsrc -> h264parse ->
  mfdecode[with_d3d11] -> d3d11sink`, a visible window with decoded video.

### W1 (Phase 3): `MfDecode` zero-copy D3D11 texture output

- Completes the Windows zero-copy decode track. With `with_d3d11()`, `MfDecode`
  now emits `MemoryDomain::D3D11Texture` frames instead of reading the decoded
  pixels back to system memory: the decoded NV12 stays in the GPU texture, so a
  DXGI / D3D11 consumer (a future swapchain present sink) takes the handoff
  without a GPU->CPU copy. The Windows analog of C3's `NvdecCuda` output.
- `extract_texture` pulls the `ID3D11Texture2D` and its subresource index out of
  the DXVA output sample's `IMFDXGIBuffer` (`GetResource` +
  `GetSubresourceIndex`) and wraps them in an `OwnedD3D11Texture`. The
  keep-alive (`SampleOwner`) owns the `IMFSample`, so the texture stays valid
  until the consumer drops the frame, then the sample returns to the decoder's
  output texture pool. `Send` under the same MTA contract as the decoder.
- `DecodedPicture` gained a `DecodedPayload` enum (`System(Box<[u8]>)` vs
  `D3D11(OwnedD3D11Texture)`); `process_output` branches on the active device
  (texture extraction on the DXVA path, the Phase-2 system readback otherwise),
  and `process` maps the payload to the frame's `MemoryDomain`. The software
  path is byte-identical.
- VERIFIED on the Windows dev host: `cargo test -p g2g-plugins --features
  mf-decode` (27 passed, the full D3D11 texture path COMPILES) and
  `cargo clippy --features mf-decode` green. The DXVA decode runtime (GPU decode
  of a real H.264 stream into textures, and a D3D11 consumer to display them) is
  owed as a user-side run on a GPU machine; the dev host can do it. The present
  sink (the D3D11 consumer, analog of `CudaGlSink`) is the next phase.

### W1 (Phase 2): `MfDecode` DXVA / D3D11 hardware decode

- `MfDecode::with_d3d11()` opts the Windows decoder into DXVA hardware decode.
  `configure_pipeline` then creates a hardware D3D11 device
  (`D3D11CreateDevice`, `D3D_DRIVER_TYPE_HARDWARE`, video support, multithread
  protection on, which Media Foundation requires) and a Media Foundation DXGI
  device manager (`MFCreateDXGIDeviceManager` + `ResetDevice`), and hands it to
  the MFT via `MFT_MESSAGE_SET_D3D_MANAGER` before setting the media types. The
  decode then runs on the GPU. Default stays the MS software decoder.
- The sync `CLSID_MSH264DecoderMFT` does DXVA in-place when given a D3D
  manager, so no async-MFT / `MFTEnumEx` event loop is needed: the existing
  synchronous `ProcessInput`/`ProcessOutput` drain still drives it. With a D3D
  manager the MFT allocates its own output samples
  (`MFT_OUTPUT_STREAM_PROVIDES_SAMPLES`); `process_output` detects that flag
  and passes a null sample so the MFT fills it, instead of pre-allocating a
  system buffer. The software path is byte-identical to before.
- Phase 2 reads the (D3D11-backed) output sample back to packed system NV12 via
  the existing `copy_sample`, so every current sink keeps working with
  hardware decode. The zero-copy `MemoryDomain::D3D11Texture` output (no
  readback) is Phase 3.
- `DecoderState` holds the D3D11 device and DXGI manager for the decoder's
  lifetime; the device outlives every output sample. New GPU-free builder unit
  test (`with_d3d11` toggles the opt-in). VERIFIED on the Windows dev host:
  `cargo test -p g2g-plugins --features mf-decode` (27 passed, the D3D11 path
  COMPILES) and `cargo clippy --features mf-decode` green. The actual DXVA
  decode runtime (device creation + GPU decode of a real H.264 stream) is owed
  as a user-side run on a machine with a GPU; the Windows dev host can do it.
- Adds `windows` crate features `Win32_Graphics_Direct3D11`,
  `Win32_Graphics_Direct3D`, `Win32_Graphics_Dxgi`, `Win32_Graphics_Dxgi_Common`.

### W1 (Phase 1): Direct3D 11 memory domain foundation

- First phase of the Windows zero-copy decode -> display track, the analog of
  the C3 CUDA track for the `MfDecode` path. Goal: keep DXVA-decoded NV12
  resident in a D3D11 texture instead of copying it to system memory, so a
  DXGI / D3D11 consumer (a swapchain present sink) takes the handoff without a
  GPU->CPU copy. Phase 1 lands only the platform-agnostic core types (compiles
  and tests on the Windows dev host); the `MfDecode` DXVA output (Phase 2) and
  the present sink (Phase 3) follow.
- New `MemoryDomain::D3D11Texture(OwnedD3D11Texture)` variant and matching
  `MemoryDomainKind::D3D11Texture`, mirroring the handle-based `Cuda` variant
  (no `windows`-crate link in `no_std` core). `OwnedD3D11Texture` carries the
  `ID3D11Texture2D` pointer, the subresource index of this frame within it
  (DXVA decoders hand out a texture *array* whose subresources are the decoded
  surfaces), the visible dims, the `DXGI_FORMAT`, the `ID3D11Device`, and a
  keep-alive owner.
- `D3D11KeepAlive` trait (`Debug + Send`): the producing element boxes its
  owning handle (a Media Foundation `IMFSample` / `IMFDXGIBuffer`) as a trait
  object so core never links the `windows` crate; dropping the buffer drops
  the box and releases the sample back to the decoder.
- `AllocationParams::d3d11(size, count, align)` constructor, the Windows analog
  of `AllocationParams::cuda`, so a DXGI consumer can request texture-resident
  buffers via the M12 allocation query.
- Two unit tests in `memory.rs` (`kind()` maps the new variant; a `FlagOnDrop`
  keep-alive proves the backing texture is released exactly when the buffer
  drops). VERIFIED on the Windows dev host: full workspace tests (114 core
  passed), workspace clippy, and the no_std baseline all green.

### M13: `MfDecode` NV12 stride handling (Windows)

- `MfDecode` no longer assumes the Media Foundation decoder's NV12 output is
  tightly packed. The MFT can report an `MF_MT_DEFAULT_STRIDE` larger than the
  frame width (hardware MFTs align rows up; the MS software decoder packs
  tightly), in which case the contiguous output buffer carries per-row padding
  that would feed garbage to the packed-NV12 sinks (`WaylandSink`, `KmsSink`).
- `set_nv12_output` now also reads the output type's `MF_MT_DEFAULT_STRIDE`
  (floored at `width`, so a missing or bottom-up value degrades to the
  packed assumption) and caches it on `DecoderState`. `copy_sample` strips the
  padding via a new pure `pack_nv12(src, width, height, stride)` that copies
  `width` bytes from each of the `height + height/2` source rows (Y plane +
  half-height interleaved UV) into a tightly-packed `width*height*3/2` buffer.
  `stride == width` stays a row-wise copy; a short source buffer leaves the
  tail zeroed rather than panicking.
- Four GPU-free unit tests for `pack_nv12` (packed identity, stride
  de-padding, bad-geometry reject, short-source fail-safe). VERIFIED on the
  Windows dev host: `cargo test -p g2g-plugins --features mf-decode` (26
  passed) and `cargo clippy --features mf-decode` green.

### C3 (Phase 3, step 3): allocation-query handshake for the CUDA path

- Completes the C3 roadmap (DESIGN-C3-cuda.md §4 step 3): wires the M12
  allocation query so the GPU sink declares it wants device memory and the
  CUDA decoder records the request. The cross-element conveyance machinery
  itself already existed (and is tested generically in `m12_allocation.rs`,
  including the two CUDA-domain cases); this connects the real elements.
- `CudaGlSink::propose_allocation` returns `AllocationParams::cuda(...)`: a
  `MemoryDomainKind::Cuda` proposal sized to the NV12 frame
  (`cuda::nv12_byte_size`), with pool headroom (`min_buffers = 3`: the frame
  on the GL thread plus the one the runner link holds) and 256-byte GPU
  alignment. Returns `None` until the geometry is fixed. The runner conveys
  this to the upstream decoder's `configure_allocation`.
- `FfmpegH264Dec::configure_allocation` records the proposal
  (`requested_alloc()` accessor). On the `NvdecCuda` backend a Cuda request is
  honoured by construction (it already emits device-resident frames); the
  system-memory backends can't satisfy it, so the request stays recorded for
  diagnostics rather than silently changing the output domain. The runner
  calls `configure_allocation` *after* `configure_pipeline` (the decoder is
  already open), so the recorded `min_buffers` is not yet used to size the
  hwframe pool's `extra_hw_frames` (still the fixed `8` from Phase 2); doing so
  is a future optimization that needs the allocation query moved ahead of
  decoder open, noted in the field docs.
- New GPU-free unit tests: `cuda::nv12_byte_size` (even/odd dims),
  `CudaGlSink` proposes Cuda with the right size/align/headroom, and
  `FfmpegH264Dec` records a conveyed Cuda proposal. These run on Linux under
  `--features cuda-gl` / `--features ffmpeg` (the modules are Linux-gated);
  the platform-agnostic conveyance remains covered on the Windows host by
  `m12_allocation.rs`. Default workspace build/test/clippy green.

### C3 (Phase 3, step 2b): `CudaGlSink` (first draft, owed first compile)

- `g2g-plugins::cudaglsink::CudaGlSink` behind a new `cuda-gl` feature
  (implies `cuda`; Linux + NVIDIA). The zero-copy-ish display payoff: keeps
  `Backend::NvdecCuda` decoded NV12 on the GPU and presents it via CUDA-GL
  interop (device->texture copy + NV12->RGB fragment shader) on a Wayland EGL
  surface, removing both the device->host copy `CudaDownload` pays and the CPU
  colour convert `WaylandSink` pays (DESIGN-C3-cuda.md §3.2, §4 step 2).
- `CudaGlInterop` (in `g2g-plugins::cuda`, gated `cuda-gl`) is the CUDA side:
  registers the two GL textures once (`cuGraphicsGLRegisterImage`,
  write-discard), then per frame maps them, `cuMemcpy2D`s each NV12 plane
  device->`cudaArray`, and unmaps; unregisters on drop. This consumes the
  step-2a interop FFI. `make_context_current` pushes the ffmpeg CUDA context
  onto the sink's GL worker thread.
- Sink structure mirrors the proven `WaylandSink`: a dedicated worker thread
  owns the Wayland connection + EGL/GL context (both thread-affine); the sink
  struct holds only `Send` handles (a `calloop` channel + atomics). The
  decoded `OwnedCudaBuffer` is `Send`, so it crosses to the worker and the
  device frame stays pinned until presented. NV12-in-CUDA only (a
  system-memory frame is rejected `UnsupportedDomain`); per-frame ack gives
  compositor-paced backpressure; mid-stream geometry change respawns the
  worker (M16 5j).
- New deps under the `cuda-gl` feature (Linux-gated, verified current):
  `khronos-egl` 6 (`static`, links libEGL), `glow` 0.17 (GL ES wrapper),
  `wayland-egl` 0.32 (`wl_egl_window` from the SCTK surface). The Appendix A
  vertex + fragment shaders (from step 2a) drive a fullscreen quad with two
  NV12 textures (luma `R8`, chroma `RG8`), GL ES 3 for the single/two-channel
  formats.
- THREE GPU-free unit tests (intercept pass-through, non-NV12 reject, odd-dim
  reject) lock the negotiation surface.
- VERIFICATION: this is a FIRST DRAFT. The module is `cuda-gl` + Linux +
  NVIDIA-gated and is NOT compiled on the Windows dev host; the EGL/GL/Wayland
  worker is owed a first compile and an e2e on the Linux+GPU box. The crate-API
  spots most likely to need a small fixup carry inline `// VERIFY:` notes (the
  `wl_display` / `wl_surface` raw-pointer accessors on `wayland-client` 0.31,
  glow 0.17's `tex_image_2d` pixel-source parameter, and the
  `eglGetProcAddress` cast for glow's loader). Acceptance test: a
  `wayland_smoke`-style benchmark `rtspsrc -> h264parse ->
  ffmpegdec[NvdecCuda] -> CudaGlSink`, p50/p95 versus the `NvdecCuvid ->
  WaylandSink` system-memory baseline. Default workspace build/test/clippy and
  no_std core baseline remain green.

### C3 (Phase 3, step 2a): CUDA-GL interop foundation

- Groundwork for `CudaGlSink` (DESIGN-C3-cuda.md §3.2, §4 step 2), the real
  zero-copy-ish display path: decoded NV12 stays on the GPU and is presented
  via CUDA-GL interop (`cuGraphicsMapResources` -> mapped `cudaArray` ->
  device->array `cuMemcpy2D` -> fragment-shader YCbCr->RGB), removing the
  PCIe round-trip and the CPU colour convert. Staged ahead of the EGL/Wayland
  windowing (step 2b) so the verifiable pieces land first, mirroring how
  Phase 1's core types preceded Phase 2's FFI.
- Extends the `g2g-plugins::cuda` FFI (behind the existing `cuda` feature,
  Linux + NVIDIA) with the CUDA-GL interop entry points, verified against the
  CUDA Driver API docs (`CUDA_GL` / `CUDA_GRAPHICS` groups):
  `cuGraphicsGLRegisterImage`, `cuGraphicsUnregisterResource`,
  `cuGraphicsMapResources`, `cuGraphicsUnmapResources`,
  `cuGraphicsSubResourceGetMappedArray`, plus the `CU_MEMORYTYPE_ARRAY` /
  `WRITE_DISCARD` / `GL_TEXTURE_2D` constants and opaque handle aliases. These
  live in `libcuda` itself, so no extra link. Marked `#[allow(dead_code)]`
  until step 2b calls them.
- `nv12_gl_uploads(width, height)` computes the per-plane device->`cudaArray`
  copy extents: a full-res `R8` luma texture (1 byte/texel) and a half-res
  `RG8` interleaved CbCr texture (2 bytes/texel). Pure geometry, unit-tested
  for even and odd dims.
- The NV12->RGB fragment shader and its paired vertex shader land verbatim
  from DESIGN-C3-cuda.md Appendix A as `FRAGMENT_SHADER_NV12` / `VERTEX_SHADER`
  consts (GLSL ES 1.00, BT.601 limited range). A test locks the `y_tex` /
  `uv_tex` sampler contract the CUDA upload side depends on.
- VERIFICATION: the module is `cuda`-feature + Linux + NVIDIA-gated and does
  not compile on the Windows dev host; the GPU-free unit tests (plane extents,
  shader contract) run under `cargo test -p g2g-plugins --features cuda` on
  the Linux box. Step 2b (EGL context on a Wayland surface, GL program, the
  map/copy/unmap render loop) is the remaining lift and needs an EGL/GL
  binding-crate decision. Default workspace build/clippy green.

### C3 (Phase 3, step 1): `CudaDownload` device->host bring-up element

- New `g2g-plugins::cuda::CudaDownload` transform behind a new `cuda`
  feature (Linux + NVIDIA, implies std). Copies a `Backend::NvdecCuda`
  `MemoryDomain::Cuda` NV12 frame back to system memory (device->host
  `cuMemcpy2D`, honouring the device-side row pitch) so a CUDA-resident
  stream reaches the existing CPU sinks (`WaylandSink` / `KmsSink`). It
  negates the zero-copy latency win, but it makes the `NvdecCuda` decode
  path end-to-end usable and testable (frame counts, geometry) before the
  real `CudaGlSink` exists. This is the low-risk Phase 3 bring-up element
  (DESIGN-C3-cuda.md §3.4, §4 step 1).
- Caps surface is `Identity(NV12)` with open geometry: input and output are
  the same NV12 description (caps do not encode the memory domain), so the
  element drops into any `NvdecCuda -> sink` chain without changing
  negotiation; only the frame's domain changes (`Cuda -> System`). A frame
  already in system memory passes through untouched, so it is a safe no-op
  on the `Software` / `NvdecCuvid` backends.
- Packed NV12 destination: luma plane (`width*height`) then interleaved
  chroma (`2*ceil(w/2)*ceil(h/2)`); for even dims the standard
  `width*height*3/2`. The per-plane copy descriptors (`nv12_plane_copies`)
  are pure geometry, unit-tested for even and odd dimensions without a GPU.
- CUDA bindings: thin hand-rolled FFI linking `libcuda` directly (no crate
  dep), per DESIGN-C3-cuda.md §6 (`cudarc` has no GL-interop wrappers and
  fights the foreign-`CUcontext` ownership). Exactly the surface this
  element needs: `cuCtxPushCurrent_v2` / `cuCtxPopCurrent_v2` (push the
  ffmpeg-owned context the pointers are valid in, always pop) and
  `cuMemcpy2D_v2`. `#[repr(C)] CudaMemcpy2D` mirrors `cuda.h` field-for-field
  (verified against the CUDA Driver API docs). Every `unsafe` block carries
  a `// SAFETY:` note.
- New `HardwareError::Cuda(i32)` variant carries the raw `CUresult` on a
  driver failure, mirroring `Vulkan(i32)` / `MediaFoundation(i32)`. Core
  change; compiles in std and no_std.
- Five GPU-free unit tests (`Identity(NV12)` constraint, non-NV12 reject,
  intercept narrowing, even- and odd-dimension plane packing).
- VERIFICATION: the device-copy path is `cuda`-feature + Linux + NVIDIA-GPU
  only and does not compile on the Windows dev host; first-compile and the
  e2e (`rtspsrc -> h264parse -> ffmpegdec[NvdecCuda] -> CudaDownload ->
  WaylandSink`) are owed on the Linux+GPU box, same as the rest of C3. The
  default workspace build/test/clippy and the no_std core baseline are green.

### C3 (Phase 2): NVDEC CUDA hwframe output (`Backend::NvdecCuda`)

- New `FfmpegH264Dec` backend that keeps decoded NV12 resident in GPU
  memory. Unlike `Backend::NvdecCuvid` (standalone `h264_cuvid` codec,
  copies NV12 back to system memory), `NvdecCuda` attaches an
  `AV_HWDEVICE_TYPE_CUDA` device to the generic `h264` decoder and
  installs a `get_format` hook selecting `AV_PIX_FMT_CUDA`, the canonical
  `hw_decode.c` true-hwaccel pattern. Decoded frames are emitted as
  `MemoryDomain::Cuda` carrying the two NV12 plane device pointers, row
  pitches, dims, and the `CUcontext`; the owning `AVFrame` is boxed as the
  buffer's `CudaKeepAlive`, so the device memory is released back to the
  hwframe pool exactly when a downstream consumer drops the frame. Removes
  cuvid's device->host copy; the latency payoff lands once a CUDA-consuming
  sink (Phase 3) takes the handoff copy-free.
- Output is always NV12 (the device frame's native layout):
  `with_backend(NvdecCuda)` pins `OutputFormat::Nv12` and `configure_pipeline`
  rejects an I420 request loud (a GPU colour convert would be needed, out of
  scope). Negotiation surface is unchanged (H.264 in, NV12 out, same
  geometry) so chains compose identically to the other backends; only the
  emitted frame's memory domain differs, which caps do not encode.
- `extra_hw_frames = 8` gives the decoder pool headroom for the frames held
  in flight downstream (link capacity) plus its own reference set, so the
  pool does not starve under `LatencyProfile::Live`. `low_delay` is on (as
  for cuvid); the cuvid-private `surfaces` knob does not apply to the
  generic hwaccel.
- The decode loop allocates a fresh `AVFrame` per drained picture (the CUDA
  path moves the whole frame into the keep-alive, so it cannot be reused
  scratch); the system-memory paths (`Software`, `NvdecCuvid`) are otherwise
  byte-identical. `DecodedPicture` gained a `DecodedPayload` enum
  (`System(Box<[u8]>)` vs `Cuda(OwnedCudaBuffer)`).
- Raw FFI via `ffmpeg_next::ffi` (re-exported `ffmpeg-sys-next`):
  `av_hwdevice_ctx_create`, `av_buffer_ref`/`unref`, a `get_cuda_format`
  C callback, and an `AVCUDADeviceContextHead` `#[repr(C)]` mirror of the
  public `AVCUDADeviceContext` head (read the `CUcontext` without depending
  on ffmpeg-sys-next having bound the optional CUDA header). Every `unsafe`
  block carries a `// SAFETY:` note; `CudaFrameOwner` asserts `Send` under
  the same ownership-transfer contract as the decoder.
- Three GPU-free unit tests lock the builder surface (NvdecCuda forces NV12
  + low-delay, overrides a prior I420, and its caps constraint is NV12).
- VERIFICATION: this path is `ffmpeg`-feature + Linux + NVIDIA-GPU only and
  does not compile on the Windows dev host; first-compile and the e2e decode
  are owed on the Linux+GPU box, same as the existing rtsp/kms/wayland code.

### C3 (Phase 1): CUDA memory domain foundation

- First phase of the zero-copy NVDEC -> GPU display track. Goal: keep
  `Backend::NvdecCuvid`'s decoded NV12 resident in CUDA device memory
  instead of cuvid's default device->host copy, so a GPU consumer
  (display scanout) takes the handoff copy-free. Phase 1 lands only the
  platform-agnostic core types (compiles and tests on the Windows dev
  host); the ffmpeg hwframe output (Phase 2) and the KmsSink CUDA
  consumer (Phase 3) follow.
- New `MemoryDomain::Cuda(OwnedCudaBuffer)` variant and matching
  `MemoryDomainKind::Cuda`, mirroring the existing handle-based
  `VulkanTexture` / `WebGPUBuffer` variants (no CUDA link in `no_std`
  core). `OwnedCudaBuffer` carries the two NV12 plane device pointers
  (luma Y, interleaved chroma UV) with row pitches, visible dims, the
  `CUcontext` the pointers are valid in, and a keep-alive owner.
- `CudaKeepAlive` trait (`Debug + Send`): the producing element boxes
  its owning handle (an ffmpeg `CUDA`-hwframe `AVFrame`) as a trait
  object so core never links CUDA; dropping the buffer drops the box and
  releases the allocation back to the hwframe pool. Pointers stay valid
  for exactly the keep-alive's lifetime.
- `AllocationParams::cuda(size, count, align)` constructor so a GPU
  consumer can request device-resident buffers via the M12 allocation
  query. This makes `MemoryDomainKind::Cuda` the first cross-element
  pool domain that crosses a real producer/consumer boundary, the
  driver M18 item 1 (allocation re-cascade beta) was waiting on.
- Two unit tests in `memory.rs`: `kind()` maps the new variant, and a
  `FlagOnDrop` keep-alive proves the backing allocation is released
  exactly when the buffer drops. no_std baseline (`--no-default-features`),
  default core build, full workspace tests, and workspace clippy all
  green; the change touches only `g2g-core` (`memory.rs`, `query.rs`,
  `lib.rs`).
- Two GPU-free runner tests in `m12_allocation.rs` prove the wiring claim:
  the allocation query conveys a consumer's `MemoryDomainKind::Cuda`
  proposal end-to-end to the producer, and the CUDA domain survives a
  transform fold (most-demanding size/align win, consumer dictates the
  domain). These exercise the real linear runners with fake elements and
  run on the Windows host.

### LatencyProfile knob on runners

- New `LatencyProfile` enum + `LinkCapacity` newtype in
  `g2g_core::runtime`. Runner signatures change from
  `link_capacity: usize` to `link_capacity: impl Into<LinkCapacity>`,
  so callers express intent (`LatencyProfile::Live`,
  `LatencyProfile::Throughput`, `LatencyProfile::Custom(n)`) instead of
  remembering the steady-state floor formula `2 * cap * frame_period`.
  Test code keeps working — `4usize` still composes via
  `From<usize> for LinkCapacity`.
- `LatencyProfile::Live` -> capacity 2 (~67 ms floor at 60 fps; RTSP ->
  decode -> display). `Throughput` -> 8 (~267 ms; batch / file
  ingest). `Custom(n)` -> caller picks; clamps 0 to 1 so a misconfigured
  env var doesn't deadlock the producer.
- `wayland_smoke` defaults to `LatencyProfile::Live` (was hard-coded
  to 8 with `G2G_LINK_CAP` override). The env var is now a
  bisection-tooling override, not the only path to a low-latency run.
  The recipe in `project_wayland_smoke_recipe.md` no longer needs
  `G2G_LINK_CAP=1` to express live intent.
- 5 unit tests in `g2g-core/src/runtime/runner.rs` (`profile_tests`)
  lock the profile -> capacity mapping, the zero-clamp, and the
  `Into<LinkCapacity>` composition so a refactor can't silently
  drift the defaults.

### RtspSrc: stash post-SETUP session, skip duplicate connect

- `intercept_caps`'s probe used to DESCRIBE + SETUP, extract caps, and
  *drop* the session — `run`'s first session attempt re-paid the same
  round-trips on the same server. Now the probe stashes the post-SETUP
  `Session<Described>` in `RtspSrc::stashed_session` alongside the
  discovered caps; `run_session` takes it on the first attempt and
  goes straight to PLAY. Reconnects after a network failure rebuild
  from scratch (by definition the stashed session is gone once the
  connection drops).
- Shared `connect_describe_setup` helper extracts the DESCRIBE +
  SETUP step from `run_session`; both the probe path (`probe_session`)
  and the reconnect path call it. `probe_caps_with_reconnect` renamed
  to `probe_session_with_reconnect`, returns a `StashedSession`
  carrying session + video_idx + caps.
- `run_rtsp` and `run_session` take `&mut RtspSrc` (was `&RtspSrc`) so
  the stash can be `take`n at the boundary between probe and run.

### M17: split `VideoFormat` into codec vs. raw pixel layout

- `Caps::Video { format: VideoFormat, .. }` (where `VideoFormat`
  conflated `H264`/`H265`/`Av1`/`Vp9` with `Nv12`/`I420`/`Rgba8`/`Bgra8`)
  is gone. Replaced by two distinct `Caps` variants:
  - `Caps::CompressedVideo { codec: VideoCodec, width, height, framerate }`
  - `Caps::RawVideo { format: RawVideoFormat, width, height, framerate }`
- New enums: `VideoCodec { H264, H265, Av1, Vp9 }` and
  `RawVideoFormat { Nv12, I420, Rgba8, Bgra8 }`. Old `VideoFormat`
  removed.
- A raw sink (`waylandsink`, `kmssink`) offered compressed input now
  fails negotiation structurally via `Caps::intersect` variant
  mismatch, not a runtime format compare. Mirrors GStreamer's
  `video/x-h264` vs `video/x-raw` distinction.
- New `Caps::dims(&self) -> Option<(&Dim, &Dim, &Rate)>` helper for
  element code that needs geometry without caring whether the link is
  pre- or post-decode. Several pattern-match sites collapsed to use it.
- `Caps::intersect` / `is_fixed` / `fixate` updated for the new
  variants; `is_fixed` now delegates to `dims()`.
- Both video variants keep `width/height/framerate` for now. Honest
  answer is GStreamer drops them on `video/x-h264` because they live
  in SPS, but our solver + RtspSrc placeholder Range + Range-as-
  placeholder convention all hang off geometry on compressed caps.
  Dropping it is a deeper rework overlapping workaround #1's
  redesign; out of scope here.
- No `codec_data` / `extras` field added. Latent need (SPS+PPS in
  negotiation, profile/level) but no current consumer; add when first
  driver lands.
- `AcceptsAny` still matches any `Caps` variant — no split into
  `AcceptsAnyRaw` / `AcceptsAnyCompressed`. Sinks that need to
  discriminate already pattern-match the format.
- Migration touched ~145 `Caps::Video` constructors and ~206
  `VideoFormat::` references across the workspace (sources, decoders,
  sinks, solver, fan-in/out, ~16 test files). Most was mechanical;
  helper functions in tests gained both `video()` (raw) and
  `compressed()` variants. `OutputFormat::video_format()` renamed to
  `raw_format()` in `ffmpegdec.rs`.
- Verified: `cargo test --workspace` (175 passed, 0 failed),
  `cargo test --features rtsp` (80 passed), `cargo test --features
  "rtsp ffmpeg"` (95 passed), `cargo check --features vaapi`,
  `cargo check --features wayland-sink`, no_std baseline.

### M18 item 5: async `SourceLoop::intercept_caps`

- Closes workaround #1 (RtspSrc placeholder Range). Sources can now
  perform I/O during negotiation and return real caps instead of a wide
  placeholder. `SourceLoop` gains `type CapsFuture<'a>` and
  `fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a>`; the
  default `caps_constraint()` is also async and awaits `intercept_caps`.
  `&self` → `&mut self` so a source can stash state (the discovered
  caps, an open session) for `run` to reuse.
- `DynSourceLoop` (the erased mirror used by fan-in / muxer) becomes
  `BoxFuture<'a, Result<Caps, G2gError>>`-returning. Same shape as
  `DynAsyncElement::process`.
- Runtime call sites — `runner.rs::run_simple_chain`, `run_source_fanout`,
  `coordinator::negotiate_source_transform_sink`, `fanin.rs::run_fanin_sink`
  / `run_muxer_sink` — `.await` the new future.
  `negotiate_source_transform_sink` is now `async fn`.
- `RtspSrc` refactor: `intercept_caps` performs DESCRIBE + SETUP, parses
  H.264 `VideoParameters`, drops the session, and caches the discovered
  caps for `caps_constraint` re-fixate retries. `with_expected_dims` is
  the offline fast path (returns caller-supplied geometry without I/O,
  framerate stays a fixable Range). Failures flow through the reconnect
  policy: a transient connect drop during negotiation retries with the
  same backoff `run` uses for mid-session drops; structural failures
  (bad URL, no H.264 stream) surface immediately.
- Test sources mechanically migrated (~16 files): each gains
  `type CapsFuture<'a> = core::future::Ready<...>` and a one-line
  `intercept_caps` returning `core::future::ready(...)`.
- New `m18_async_intercept_caps` test proves the runner genuinely awaits
  the caps future: a source `tokio::time::sleep`s in `intercept_caps`,
  and the pipeline's elapsed time must be at least the sleep interval.
  A sync stub would finish in microseconds.
- RtspSrc test surface updated:
  `rtspsrc_intercept_caps_with_expected_dims_skips_probe_and_fixates`
  replaces the old placeholder-Range assertion;
  `rtspsrc_intercept_caps_probes_and_fails_on_unreachable_url` covers
  the new probe-failure pathway. `rtspsrc_with_reconnect_retries_then_fails`
  now exercises probe-side reconnect (probe failure goes through the
  same retry+backoff loop `run` uses for session drops).
- Documents the structural cost: probe and `run`'s session-open are
  distinct connections, so the server pays for two DESCRIBEs at
  startup. A future optimization is to stash the post-SETUP session
  and consume it in `run`.

### caps-nego: drop per-frame `Frame.caps`

- Removed the `caps: Caps` field from `Frame` in `g2g-core/src/frame.rs`.
  Element-level caps state (set in `configure_pipeline`, updated on
  `PipelinePacket::CapsChanged`) is the single source of truth;
  `CapsChanged` events delimit caps epochs before the first affected
  `DataFrame`. The field was write-only on every hot path (grep
  confirmed zero readers in production code), so every produced frame
  was paying a `Caps::clone` for no consumer. Matches the lesson
  modern GStreamer applied when it moved caps off `GstBuffer` onto the
  pad.
- Updated all `Frame` constructors (sources, decoders, tests) to drop
  the field. `make_frame` test helpers that took a `caps: Caps` lost
  that parameter; call sites updated.
- `PickyByWidthSink` in `m8_negotiation` now tracks the current width
  via accepted `configure_pipeline` and `CapsChanged` events through
  `process()` instead of reading `frame.caps`. The
  `mid_stream_reconfigure_round_trip` data-widths assertion changed
  from `[640, 1280, 640]` to `[640, 640, 640]`: the rejected 1280
  `CapsChanged` never reaches the sink (the runner intercepts and
  converts to a `Reconfigure`), so the sink's tracked width never
  advances to 1280. The configure_widths assertion still proves the
  reconfigure round trip.
- `FakeReorderDecoder` in `m16_workaround3_phase_a` seeds `input_caps`
  from `configure_pipeline`'s `absolute_caps` so the first
  `DataFrame` (which arrives before any runtime `CapsChanged`) still
  has the correct input caps tagged for queue reorder accounting.
- Removed now-orphan `any_h264_caps()` in `rtspsrc.rs`.

### NVDEC backend low-latency defaults + `wayland_smoke` knobs

- `FfmpegH264Dec::with_backend(Backend::NvdecCuvid)` now enables
  low-latency tuning by default: `h264_cuvid surfaces=4` (down from
  cuvid's default 25, which adds ~25 frames of in-decoder buffering)
  and the `AV_CODEC_FLAG_LOW_DELAY` codec flag (release each picture as
  soon as it's decoded, no reorder hold). Switching back to
  `Backend::Software` clears the tuning so the sw path stays at
  libavcodec defaults. Override either knob via
  `with_cuvid_surfaces(Some(n))` / `with_low_delay(bool)` *after*
  `with_backend`. Wired in `configure_pipeline` via `open_as_with` +
  `Dictionary` + `Context::set_flags`. Closes the gap where the first
  `wayland_smoke` run with `NvdecCuvid` saw p50 = 163 ms (~80 ms over
  the link-cap floor) — almost entirely cuvid's `surfaces=25` pipeline
  depth, recoverable without a CUDA memory domain.
- Four new unit tests:
  `software_backend_does_not_set_cuvid_defaults`,
  `nvdec_backend_defaults_to_low_latency_tuning`,
  `switching_back_to_software_clears_nvdec_tuning`,
  `cuvid_surfaces_override_survives_after_with_backend`. Defaults are
  policy, not implementation detail; locking them so a refactor can't
  silently revert.
- `wayland_smoke` gains `G2G_TARGET_FRAMES` (default 60) so a NVDEC
  benchmark can amortize cuvid's startup tax (libnvcuvid load + CUDA
  context + surface pool can be 1-2 s). Test timeout now scales:
  `30 + max(30, target * 100 ms / 1000)` so a 600-frame steady-state
  run fits cleanly. Steady-state p50/p95 only become meaningful for
  `target >= ~300`.

### NVDEC backend for `FfmpegH264Dec` (NVIDIA hardware H.264 decode)

- New `Backend` enum and `FfmpegH264Dec::with_backend(Backend)`. Defaults
  to `Backend::Software` (existing libavcodec built-in H.264 decoder, no
  behavior change). `Backend::NvdecCuvid` opens the `h264_cuvid` standalone
  codec via `codec::decoder::find_by_name("h264_cuvid")`; if libavcodec
  wasn't built with cuvid or `libnvcuvid.so` isn't reachable at runtime,
  `configure_pipeline` fails loud with `Hardware(Other)` so the caller's
  backend choice is honoured rather than silently downgraded.
- No new Cargo feature: cuvid is a runtime libavcodec lookup, not a
  link-time dep. No raw `ffmpeg-sys` hwaccel plumbing
  (`AVHWDeviceContext`, `get_format`, `av_hwframe_transfer_data`) is
  needed because cuvid is a standalone codec that emits NV12 directly to
  system memory.
- Caps surface unchanged: the `DerivedOutput` constraint (H.264 in →
  NV12/I420 out, same geometry, framerate forwarded) is identical
  across backends, so existing solver wiring, mixed-chain tests, and the
  M16 workaround #3 Phase A consistency check apply verbatim.
- `copy_yuv420` extended to accept `Pixel::NV12` source frames (cuvid's
  native output) in addition to `YUV420P`/`YUVJ420P` (software path).
  Four (source-layout × output-layout) branches, each honouring source
  pitch. NV12-source → I420-output de-interleaves; NV12-source →
  NV12-output is a row copy.
- New unit tests `default_backend_is_software`,
  `with_backend_overrides_default`, and
  `caps_constraint_independent_of_backend` (NVDEC and software backends
  produce identical solver constraints — chains compose the same way).
- Visual e2e (`rtspsrc → h264parse → ffmpegdec[NvdecCuvid] →
  wayland/kms`) is user-side: CI has no GPU, and the existing rtsp
  sandbox block already constrains live tests to manual verification.

### M18: GStreamer parity push (item-by-item from DESIGN-M16-caps-nego.md §13.4)

- **Item 6: pad templates (declarative, pre-instantiation metadata).**
  New `pad_template` module (g2g-core, runtime feature): `PadDirection`,
  `PadCaps` (`Fixed(CapsSet)` / `Any`), `PadTemplate`, and a `PadTemplates`
  trait whose `pad_templates()` is an associated function (no `&self`), so
  a tool inspects an element *type* without constructing it, the analog of
  GStreamer's `gst_element_factory_get_static_pad_templates`. A template is
  the static superset of what the type can do; a constructed instance's
  `caps_constraint_as_*` is a subset. `pad_link(producer, consumer)` runs
  the negotiation `solve_linear` against two templates and returns the
  fixated caps or a structured `NegotiationFailure` (`EmptyLink` =
  incompatible, `Unfixable` = compatible but geometry/framerate still
  open); `types_can_link::<A, B>()` is the convenience boolean (treating
  `Unfixable` as compatible, since static templates routinely leave
  geometry open until instance time). `PadTemplates` implemented for
  `VideoTestSrc` (RGBA source), `FakeSink` (wildcard sink), and `H264Parse`
  (H.264 sink + source). Unit tests in the module plus integration test
  `m18_pad_templates.rs` (introspection without construction;
  `VideoTestSrc -> FakeSink` compatible, `VideoTestSrc -> H264Parse`
  `EmptyLink`, direction-awareness). no_std default core build, core
  runtime suite, full workspace tests, and workspace clippy all green.

- **Item 3 (Phase C): fan-out per-branch re-solve (FO-2) with FO-1
  strict default.** A mid-stream `CapsChanged` broadcast across the
  fan-out is now re-solved per branch against that branch's declared
  `caps_constraint_as_sink()` before `configure_pipeline` (Phase B applied
  per branch), closing the gap where the fan-out runner skipped the solver
  gate. Because each branch runs in its own arm, the re-solves are
  concurrent for free. FO-1 strict default: a branch whose constraint
  rejects the new caps fails the whole fan-out loud (`CapsMismatch`),
  matching GStreamer's `tee`-with-rejecting-downstream; the
  `AllowBranchDrop` graceful-degradation policy stays a future opt-in.
  `DynAsyncElement` gains a dyn-safe `caps_constraint_as_sink` (blanket
  impl forwards to `AsyncElement`) so `Box`-erased branch sinks can be
  re-solved; `re_solve_downstream_sink` is refactored to share a
  `re_solve_against_sink_constraint` core with the new
  `re_solve_downstream_dyn_sink`. New integration test
  `m18_fanout_phase_c.rs`: FO-2 accept (a geometry change every branch
  admits reaches each branch's `process`) and FO-1 strict reject (one
  RGBA-only branch fails the fan-out on an NV12 switch, and never sees the
  rejected caps). no_std core build, core suite, and the std plugins suite
  all green.

- **Item 1 (Session D follow-up): fan-out branch α.** Completes the α
  story for non-linear topologies. `DynAsyncElement` gains dyn-safe
  `propose_allocation` / `configure_allocation` (blanket impl forwards to
  `AsyncElement`); `coordinator::realloc_local_dyn` is the `Box`-erased
  counterpart of `realloc_local`. `run_source_fanout` now re-allocates
  each branch locally after the FO-2 re-solve applies the new caps. As in
  the linear case, the fan-out runner never configures branch allocation
  at startup, so the per-branch re-allocation is solely α. Covered by an
  added assertion in `m18_fanout_phase_c.rs` (each branch records exactly
  one re-allocation sized from the new caps).

- **Item 2 (Phase C): muxer per-input re-solve (MX-1) + input-derived
  output re-emit (MX-2).** A per-input mid-stream `CapsChanged` is now
  re-solved against the muxer's per-input constraint and applied to that
  pad (`MultiInputElement::configure_pipeline(i, ..)`), then consumed,
  instead of being forwarded raw to the output as the previous
  `process`-everything path did (which leaked input-side caps as the
  merged output). After the per-input reconfigure the runner re-derives
  the merged output and, only when it changed, eagerly emits one
  downstream `CapsChanged` (MX-2). Both run inside the existing single
  muxer task, which already owns `&mut mux` and serializes all inputs, so
  no β coordinator restructure is needed (workaround3 §10.4). Strict
  failure: a per-input caps the muxer can't accept fails loud
  (`CapsMismatch`); the structured reverse-`Renegotiate`-per-input variant
  is β-gated (the muxer task can't reach an input's reconfigure slot
  pre-β). The startup per-input and output negotiation are refactored into
  shared `solve_mux_input` / `solve_mux_output` helpers that the mid-stream
  paths reuse (no duplicated solver call). New integration test
  `m18_mux_phase_c.rs`: MX-1 (input pad re-solved, input `CapsChanged` not
  leaked, static `InterleaveMux` output unchanged) and MX-2 (a
  derived-output muxer emits exactly one downstream `CapsChanged` with the
  re-derived output). Closes the runtime half of item 2 (MX-3 trait
  surface already landed); no_std core build, core suite, and the std
  plugins suite all green.

- **Item 1 (Session D): α element-local re-allocation.** First
  observable M18 behavior change. New `coordinator::realloc_local`: when
  a mid-stream `CapsChanged` is applied to an element, the runner
  re-derives that element's own allocation params from the new caps
  (`propose_allocation`) and stores them (`configure_allocation`) before
  the element processes the notification. Wired at the three
  statically-typed mid-stream apply sites: `run_simple_pipeline` (sink),
  `run_source_transform_sink` (transform and sink). No cross-element
  cascade, that is β (Session E). Element-local only, so safe under the
  per-`Frame.caps` invariant: in-flight old-caps frames keep their
  old-pool buffers. Previously M12 allocation ran solely at startup, and
  in a 3-element chain the sink's `configure_allocation` was never called
  at all (its proposal feeds the transform); now a mid-stream geometry
  change re-allocates it. Fan-out branch sinks are excluded for now:
  `DynAsyncElement` does not expose the allocation hooks, so per-branch α
  lands with the FO-2 dyn-trait extension. New integration test
  `m18_alpha_realloc.rs` (one re-allocation sized from the new caps on a
  mid-stream change; none without). no_std core build, core suite, and
  the std plugins suite all green.

- **Item 1 (Session C): startup negotiation relocated to the
  coordinator.** Pure refactor, no behavior change. The
  `source → transform → sink` startup negotiation (the `solve_linear` +
  per-link `configure_pipeline` cascade with bounded `ReFixate` retry)
  moves verbatim out of `run_source_transform_sink` into
  `coordinator::negotiate_source_transform_sink`, returning a
  `LinearNegotiation { source_link, sink_link }` that names the per-link
  caps the β re-cascade will reconfigure. `MAX_FIXATION_ATTEMPTS` moves
  with it (still used by `run_simple_pipeline` via import). The runner now
  calls the routine and uses `sink_link` exactly where the loop produced
  `negotiated_caps`. Verified by the unchanged M8/M16/M18/pipeline_smoke
  integration suites (all green) plus the no_std core build; the next
  session (D) adds α element-local re-allocation hooks, the first
  observable M18 behavior change.

- **Item 1 (Session B): coordinator control-channel scaffolding.** First
  step of the allocation re-cascade β restructure
  (DESIGN-M16-workaround3-reconfigure.md §9.4 β; R2 single-task
  coordinator, R3 out-of-band channel). New `runtime::coordinator`
  module: `CoordinatorEvent`, `CoordinatorHandle` (clonable producer over
  the in-house mpsc), `Coordinator` task, and a `coordinator(capacity)`
  constructor. `run_source_transform_sink` now spawns the coordinator as
  a fourth join arm; the sink arm reports a `CoordinatorEvent::CapsChanged`
  for each applied mid-stream `CapsChanged` and the stub only counts them.
  The channel closes when the sink arm drops its handle, so the
  coordinator terminates with the pipeline. No data-plane behavior change:
  frames and EOS flow exactly as before; this only validates the channel
  topology before Session C moves startup negotiation in and Session E
  turns each event into a real `Recascade`. New
  `RunStats.coordinator_events` field (`0` for runners without a
  coordinator). New integration test `m18_coordinator.rs` (event observed
  on applied caps change; zero events and clean termination otherwise).
  no_std core build, core suite, and the std plugins suite all green.

- **Item 3 (partial): fan-out constraint migration.** Symmetric move
  to item 2 on the `MultiOutputElement` side. Adds
  `MultiOutputElement::caps_constraint_as_input(&self) ->
  CapsConstraint<'_>` default method to the trait (wraps
  `intercept_caps(...)` as a `LegacySink` per the existing legacy
  bridge pattern). `Router` (g2g-core) overrides to return
  `AcceptsAny` — it broadcasts upstream caps verbatim with no per-
  branch format restriction. `run_source_fanout` calls the new
  method instead of constructing an inline `LegacySink`. Closes the
  structural prerequisite for Phase C FO-2 (per-branch downstream
  re-solve once a mid-stream `CapsChanged` crosses the fan-out
  boundary); the runtime execution still needs the coordinator
  restructure (workaround3 §9 β). New unit test
  `router_input_constraint_is_wildcard` in `fanout.rs`. Existing
  M9 router/gate/merger integration tests unchanged; full workspace
  + Linux feature matrix + no_std build all green.

- **Item 2 (partial): muxer constraint migration.** Adds
  `MultiInputElement::caps_constraint_as_input(idx) -> CapsConstraint<'_>`
  and `MultiInputElement::caps_constraint_for_output() ->
  Result<CapsConstraint<'_>, G2gError>` default methods to the trait.
  Default `caps_constraint_as_input` wraps `intercept_caps(idx, ...)`
  as a `LegacySink` per-pad legacy bridge; default
  `caps_constraint_for_output` eagerly evaluates `output_caps()` and
  wraps as `LegacySource`. `InterleaveMux` (g2g-plugins) overrides
  both: per-input returns `AcceptsAny` (the muxer forwards
  per-frame-tagged caps straight through), output returns
  `Produces(CapsSet::one(self.output.clone()))` (static at
  construction). `run_muxer_sink` now calls these trait methods
  instead of constructing the inline `LegacySink` for each input pair
  and using a direct `output_caps().fixate()` for the downstream sink
  hop; the muxer→sink edge goes through `solve_linear` with an
  `AcceptsAny` sink-side constraint to preserve the contract that
  the sink's `intercept_caps` is not consulted for this hop. This
  closes the structural prerequisite for Phase C MX-1 (per-input
  mid-stream re-solve) and MX-2 (eager output `CapsChanged` on
  per-input change); the runtime execution of those still requires
  the coordinator restructure (workaround3 §9 β) and lands later.
  New unit tests `mux::tests::per_input_constraint_is_wildcard` and
  `mux::tests::output_constraint_is_produces_with_configured_output`.
  Existing `m10_muxer` integration tests and full workspace +
  Linux-feature matrix unchanged.

### M16: Caps negotiation redesign (CSP framing)
- Design doc `DESIGN-M16-caps-nego.md` (§§1-10) recasts negotiation as a
  constraint-satisfaction problem with a solver, and documents the
  6-step migration plan (§8).
- Step 1: `CapsSet` algebra in `g2g-core::caps`. Preference-ordered
  alternatives over `Caps`; `one`, `from_alternatives`, `intersect`
  (preserves self's outer order, dedupes equal results), `fixate`
  (picks the highest-preference fixable alternative). `Caps` remains
  the *fixed* runtime description; `CapsSet` is the negotiation-time
  vocabulary. Re-exported from the crate root. Unit-tested for empty
  intersection, preference preservation, dedup, and fixate fallback.
- Adjacent design debt acknowledged (no code change): `VideoFormat`
  conflates compressed codecs (H264, H265, Av1, Vp9) and raw pixel
  layouts (Nv12, I420, Rgba8, Bgra8) in a single enum, shoehorned
  into `Caps::Video { format, ... }`. GStreamer keeps these as
  separate media types (`video/x-h264` vs `video/x-raw, format=...`).
  Documented in `DESIGN-M16-caps-nego.md §11` and memory
  (`architecture_codec_vs_raw_format.md`). M17-sized refactor;
  M16 continues on the current shape.

- Latency observability: `VideoTestSrc` now stamps
  `FrameTiming::arrival_ns` at frame emission (std-gated, falls back
  to 0 in no_std), matching `RtspSrc`'s convention. `FakeSink` holds
  a `LatencyHistogram` and records `monotonic_ns() - arrival_ns` per
  received `DataFrame` whose `arrival_ns` is non-zero (std-gated),
  with a `latency_snapshot()` accessor returning a `LatencySnapshot`
  (count/mean/max/p50/p95/p99 nanoseconds, log2 buckets). `KmsSink`
  gets the same treatment: per-frame recording after page-flip
  submission, with a `latency_snapshot()` accessor; the timing point
  is page-flip submission rather than vblank completion, which
  under-reports true scanout latency by up to one refresh interval
  but is good enough as a regression guard. New regression test
  `videotestsrc_to_fakesink_latency_under_25ms` in
  `pipeline_smoke.rs` asserts max + p99 latency stay under 25ms for
  the all-in-memory `videotestsrc → fakesink` chain through the M16
  solver — catches order-of-magnitude regressions (lock contention,
  blocking I/O, runner serialization) while tolerating shared-CI
  variance. WaylandSink's existing histogram is unchanged. Tested
  across base, ffmpeg, rtsp, wayland-sink, kms-sink, std, and
  no-default-features builds.
- Workaround #3 Phase B (sink-side downstream subgraph re-solve):
  `run_simple_pipeline` and `run_source_transform_sink` now route every
  mid-stream forward `CapsChanged` arriving at the sink through a new
  `re_solve_downstream_sink()` helper before calling
  `configure_pipeline`. The helper runs `solve_linear` over
  `[LegacySource(new_caps), sink.caps_constraint_as_sink()]` and
  returns the assigned sink-input caps. A `NegotiationFailure::EmptyLink`
  here means the sink's declared constraint rejects the boundary's
  output — the runner drops the forward `CapsChanged` and signals a
  reverse `Reconfigure::Renegotiate` upstream (§7 forward × reverse:
  the request hits the transform's link, not the source, so the
  failure surfaces *at the boundary*). `NegotiationFailure::Unfixable`
  (e.g. a decoder leaving `Rate::Any` because framerate isn't
  pixel-level data) is treated as success and `new_caps` is passed
  through unchanged — fixation is a startup-negotiation concern, not
  a mid-stream re-solve one. The observable change in the current
  3-element runner: a sink with a restrictive
  `Accepts(set)` constraint can short-circuit a hostile mid-stream
  `CapsChanged` via the solver even if its legacy `configure_pipeline`
  would have silently accepted. New
  `g2g-plugins/tests/m16_workaround3_phase_b.rs`: a `HostileBoundary`
  fake transform injects a non-NV12 mid-stream `CapsChanged`; a
  `PickySink` declares `Accepts(NV12)` for the solver but accepts
  every shape in `configure_pipeline`. Asserts the hostile caps never
  reach `process` while a matching NV12 geometry-change still
  propagates. Full Linux feature matrix
  (`ffmpeg rtsp wayland-sink kms-sink`) green.
- Workaround #3 Phase A (retired for linear topology): decoders
  (`FfmpegH264Dec`, `MfDecode`, `VaapiH264Dec`) stop silently swallowing
  input `PipelinePacket::CapsChanged`. On arrival they validate the
  format (loud `CapsMismatch` on a hostile mid-stream H.264 → VP9
  switch; previously dropped silently) and record the caps into a new
  `input_caps` field. The output `CapsChanged` is still emitted at the
  decode boundary from decoded-frame geometry, preserving the §3
  ordering invariant. Each decoder's `DerivedOutput` closure now
  delegates to a free `derive_output_caps()` helper so a
  `debug_assert!` before each output `CapsChanged` push verifies
  decode-time geometry agrees with the closure applied to the recorded
  input (via `Caps::intersect` non-empty — handles the
  `Fixed`-vs-`Any` framerate asymmetry without false positives). New
  test `g2g-plugins/tests/m16_workaround3_phase_a.rs`: a
  `FakeReorderDecoder` simulates B-frame reorder; asserts the output
  `CapsChanged(B)` appears strictly between the last A-tagged frame
  and the first B-tagged frame (the regression guard against any
  future "eager forward" temptation), plus the loud-reject case. All
  44 g2g-plugins lib + integration suites green across
  `ffmpeg + rtsp + wayland-sink + kms-sink + vaapi` on Fedora.
  Decisions D2 (single source of truth via the closure) and D4
  (acknowledge, don't swallow) implemented. D1 (carrier) and D3
  (runner subgraph re-solve) not load-bearing for Phase A; they
  arrive with Phase B alongside the §7 forward × reverse
  `Reconfigure` race resolution.
- Workaround #3 design (no code): `DESIGN-M16-workaround3-reconfigure.md`
  turns the deferred forward in-band reconfigure into an implementable,
  phased spec. Key finding: decoders swallow their input `CapsChanged`
  and self-detect output geometry from decoded frames, which is
  correctly ordered; the naive "re-derive and forward eagerly" fix
  corrupts the stream across a B-frame reorder boundary (old-geometry
  frames still draining). The reconfigure must stay at the decode
  boundary. Phased plan: A) validate + record the input caps instead of
  silently dropping (linear, CI-testable), B) runner re-solves the
  downstream subgraph on a boundary `CapsChanged` (allocators /
  multi-element downstream), C) non-linear topologies. Decisions D1-D4
  (reuse `CapsChanged` vs new packet; derive via the `DerivedOutput`
  closure; runner subgraph re-solve; acknowledge-don't-swallow) flagged
  for sign-off.
- Cleanup (workaround #2 fully retired): removed the non-NV12 no-op
  accept branch from `WaylandSink` / `KmsSink` `configure_pipeline`.
  Reachability audit: every chain that reaches these sinks (incl. the
  `wayland_smoke` / `kms_smoke` `rtsp → ffmpegdec → sink` tests) now goes
  through a native `DerivedOutput` decoder, so the solver assigns the
  sink link NV12 (fixated to `Fixed`) at startup; the branch only fired
  on a decoder-less H.264→display chain, which exists in no test,
  example, or real pipeline. The sinks now reject non-NV12 input loud
  (`CapsMismatch`) instead of silently accepting undisplayable caps.
  `intercept_caps` stays pass-through (the solver hands it NV12).
  Updated tests: `configure_accepts_h264_as_deferred_noop` →
  `configure_rejects_non_nv12`; `intercept_passes_through_h264_for_deferred_configure`
  → `intercept_passes_through_any_format` (both sinks). Linux-gated
  (`wayland-sink` / `kms-sink`), so not compiled on the Windows host; a
  Linux + Wayland/DRM e2e is owed alongside the other deferred visual
  checks.
- Step 6 (DESIGN §8): `CapsFilter` element + ACCEPT_CAPS surface.
  `CapsConstraint::accepts(&Caps)` and `CapsSet::accepts(&Caps)` answer
  the ACCEPT_CAPS query (§7) as a pure check against the declared
  constraint, no negotiation: set-shapes test membership, `Mapping`
  tests its input sides, `DerivedOutput` tests input validity, wildcards
  accept anything, legacy bridges defer to their wrapped callbacks.
  `g2g-plugins::capsfilter::CapsFilter` is a pass-through transform whose
  native constraint is `Identity(set)`; inserted anywhere it pins the
  link to a concrete `CapsSet` (e.g. ahead of an `AcceptsAny` sink). It
  narrows on the legacy/mixed path via `intercept_caps`, validates its
  configured caps and any mid-stream `CapsChanged` against the filter
  (loud `CapsMismatch` on violation), and forwards data unchanged. New
  tests: `accept_caps_query_checks_constraint_set` (core) and three
  `capsfilter` lib tests. Completes the §8 migration plan.
- Workaround #1 (partial, opt-in): `RtspSrc::with_expected_dims(w, h)`.
  The RTSP handshake (and the real SDP geometry) only completes inside
  `run`, so `intercept_caps` defaults to a wide placeholder `Dim::Range`
  that survives fixation, with the real dims arriving later via
  `CapsChanged`. Callers who know the camera resolution can now declare
  it up front: `intercept_caps` then advertises fixed dims, so the
  chain negotiates the real geometry at startup and a downstream sink
  sizes its surface once instead of building at the placeholder min and
  rebuilding on the first `CapsChanged`. If the SDP disagrees, the
  mid-stream `CapsChanged` still corrects it. The placeholder remains
  the default; full auto-detection still needs an async
  `SourceLoop::intercept_caps` (the "big" redesign, deferred). New unit
  tests `intercept_caps_defaults_to_placeholder_range` and
  `with_expected_dims_advertises_fixed_geometry`; verified with
  `cargo test/clippy -p g2g-plugins --features rtsp`.
- Cleanup (post-5m): removed `FfmpegH264Dec`'s now-dead
  `is_format_boundary` and `propose_output_caps` overrides (and their
  two unit tests). The decoder is native (`DerivedOutput` since 5k), so
  the runner dispatches through `caps_constraint_as_transform` and never
  consults these forward-half hooks; `is_format_boundary` has no runtime
  consumer at all (it was a forward-declaration for a redesign that
  ended up keying on the constraint surface instead). The `AsyncElement`
  trait methods themselves stay: `propose_output_caps` is still live for
  *unmigrated* legacy transforms via the solver's mixed-cascade path
  (`solver.rs` `LegacyTransform { propose_output }`). Pure deletion; not
  compiled on the Windows dev host (`ffmpeg` is Linux-gated).
  `WaylandSink`/`KmsSink` non-NV12 branches were left in place: the
  in-code note from steps 5e/5j marks them still load-bearing for
  all-legacy chains, contradicting the "no longer load-bearing"
  expectation, so they need a separate decision before removal.
- Step 5m: `VaapiH264Dec` (Linux cros-codecs VAAPI) overrides
  `caps_constraint_as_transform` to return
  `CapsConstraint::DerivedOutput(closure)`, identical in shape to steps
  5k/5l. The closure validates H.264 input and emits NV12 at the same
  dims/framerate; non-H.264 input yields an empty `CapsSet` so the
  solver rejects at negotiation time. The VAAPI backend only emits
  NV12, so the closure has no output-format choice. New unit test
  `caps_constraint_is_derived_output_h264_to_nv12`. Not compiled on the
  Windows dev host (the `vaapi` feature is Linux-only); change is
  byte-identical in shape to the verified 5l/5k pattern.
- Step 5l: `MfDecode` (Windows Media Foundation) overrides
  `caps_constraint_as_transform` to return
  `CapsConstraint::DerivedOutput(closure)`, mirroring step 5k. The
  closure validates H.264 input and emits NV12 at the same
  dims/framerate; non-H.264 input yields an empty `CapsSet` so the
  solver rejects at negotiation time. The MFT only emits NV12, so the
  closure has no output-format choice. Mixed chains get real per-link
  caps from the solver: H.264 to the decoder, NV12 to the sink. New
  unit test `caps_constraint_is_derived_output_h264_to_nv12` covers
  the H.264→NV12 derivation and the non-H.264 rejection. Verified with
  `cargo clippy/test -p g2g-plugins --features mf-decode` (Windows).
- Step 5k: `FfmpegH264Dec` overrides `caps_constraint_as_transform`
  to return `CapsConstraint::DerivedOutput(closure)`. The closure
  validates H.264 input and emits the chosen output format
  (`Nv12`/`I420`) at the same dims/framerate; non-H.264 input
  yields an empty `CapsSet` so the solver rejects at negotiation
  time. With the decoder native, the production chain
  `rtsp → ffmpegdec → sink` becomes mixed: the solver returns
  `[H264, Nv12]` per-link, the runner feeds the decoder H.264
  (what its `configure_pipeline` requires) and the sink Nv12.
  Coupled with 5j (NV12 sinks tolerate mid-stream dim changes),
  the placeholder-dim NV12 at startup → real-dim mid-stream
  `CapsChanged` transition rebuilds the surface cleanly instead of
  refusing. New unit test
  `caps_constraint_is_derived_output_h264_to_chosen_format` covers
  the H.264→NV12 derivation and the non-H.264 rejection. All 99
  g2g-core + 10 ffmpegdec lib tests + every integration suite green
  across base / rtsp / ffmpeg feature sets. Visual verification of
  the manual `rtsp → ffmpegdec → wayland/kms` chain is on the user.

- Step 5j (reordered): NV12 display sinks (`WaylandSink`, `KmsSink`)
  tolerate mid-stream geometry changes. Previously `configure_pipeline`
  with new dims after the worker / framebuffer pool was up returned
  `CapsMismatch`; now it tears down the existing worker/slots
  (`self.shutdown()` / `self.teardown()`) and falls through to
  fresh setup at the new geometry. Same-dims is still a no-op.
  This unblocks the next step (5k: migrate `FfmpegH264Dec` to
  `DerivedOutput`): mixed chains will land NV12 at startup with
  placeholder dims (RtspSrc workaround #1, fixated to min), and the
  real geometry will arrive via mid-stream `CapsChanged` after SPS
  parse — both transitions now succeed instead of the second
  refusing. No new tests in this commit: the rebuild path runs
  through `worker_main` (Wayland session) / `allocate_slots` (real
  DRM card), neither testable in CI. Visual verification belongs to
  the user's manual e2e run.

- Step 5h: `CapsConstraint::IdentityAny` wildcard transform variant
  for pass-through transforms (probe / metering / tee) whose
  `intercept_caps` is `Ok(upstream.clone())`. Native solver couples
  input and output links to be equal without narrowing either by a
  set; surrounding endpoints determine the actual caps.
  `IdentityTransform` (g2g-plugins) overrides
  `caps_constraint_as_transform` to return `IdentityAny`.
  Existing 3-element pipeline_smoke test
  `source_identity_sink_3_element_pipeline` now exercises the
  all-native arc-consistency path
  (`VideoTestSrc Produces → IdentityTransform IdentityAny →
  FakeSink AcceptsAny`). 3 new solver tests cover the coupling, a
  mixed-chain pass-through, and rejection of `IdentityAny` in
  endpoint positions. 99 g2g-core tests + 11 plugin lib +
  every integration suite green.

- Step 5g: first native transform. `H264Parse` overrides
  `caps_constraint_as_transform` to return
  `Identity(CapsSet::one(Caps::Video { format: H264, dims: Any }))`.
  The native solver couples its input and output links and enforces
  the H.264 format requirement during arc consistency instead of via
  the dynamic `intercept_caps` callback. New tests:
  - `h264parse::caps_constraint_is_identity_h264_any` (unit).
  - `pipeline_smoke::h264parse_identity_negotiates_in_mixed_chain`
    drives a real `run_source_transform_sink` chain
    `LegacySource(H264) → H264Parse Identity → AcceptsAny FakeSink`
    and verifies negotiation + EOS propagation through the mixed
    cascade. The source is an inline EOS-only stub (no real H.264
    bytes needed — `H264Parse::process` for `Eos` is pass-through).
  96 g2g-core tests + 6 pipeline_smoke + every integration suite
  green.
- Step 5f (revised): first native source + `SourceLoop` trait
  integration. Original 5f scope (workaround #1 placeholder dims)
  bumped — properly fixing it needs async `intercept_caps` (SDP
  DESCRIBE), and it's symbiotic with #2 so fixing alone unblocks
  nothing visible.
  - `SourceLoop` gains `caps_constraint(&self) -> Result<CapsConstraint<'_>, G2gError>`
    default method returning `LegacySource(intercept_caps()?)`.
  - `run_simple_pipeline`, `run_source_transform_sink`, and
    `run_source_fanout` call `source.caps_constraint()` instead of
    constructing `LegacySource` inline. `ReFixate` retry uses
    `LegacySource(counter)` fallback (counter-proposals are a legacy
    concept). `run_muxer_sink` stays on `intercept_caps` because
    `DynSourceLoop` doesn't yet expose `caps_constraint` — no
    migrated muxer sources exist, so adding it is deferred until
    needed.
  - `VideoTestSrc` overrides `caps_constraint` to return
    `Produces(CapsSet::one(self.caps()))`. Production chain
    `videotestsrc → FakeSink` (both native) now exercises the
    all-native arc-consistency solver path with backward
    propagation, instead of the mixed cascade. Behavior unchanged.
  - 1 new solver test (`all_native_produces_to_accepts_any_passes_through`).
    96 g2g-core tests + every integration suite green.
- Step 5e: correctness fix — `solve_legacy_cascade` reverts to
  intercept-only (bit-compatible with the pre-M16 cascade). Step 4b
  had incorrectly called the format-boundary's `propose_output_caps`
  during the legacy cascade, producing per-link caps the legacy
  workaround-#2 sinks (`WaylandSink`, `KmsSink`) can't consume —
  e.g. NV12 at placeholder dims at startup, with the deferred-setup
  branch then refusing the mid-stream real-dim `CapsChanged`.
  Restored: every all-legacy chain element receives the single
  fixated `Caps` from the cascade's final intercept, matching
  pre-M16. The CI suite missed the regression because the e2e
  `rtsp → ffmpeg → waylandsink` test is `#[ignore]`d (sandbox
  blocks port 554).

  **Revised 5d claim**: per-link configure benefits mixed chains
  (one or both endpoints native). All-legacy chains keep the
  single-fixated-caps model and workaround #2 stays load-bearing
  for them — until those sinks migrate (which requires workaround
  #1, the placeholder-dims fix, to land first so per-link NV12
  carries real dims at startup).

  Updated `legacy_cascade_with_boundary_transform` test and
  `architecture_caps_nego_debt` memory.
- Step 5d: per-link configure (deletes caps-nego workaround #2).
  Two coupled changes:
  - `solve_legacy_cascade` no longer clobbers upstream link slots
    with the final fixated caps. Each link is fixated independently
    and carries its own intercept-narrowed value, so format-changing
    boundaries (decoder: H264 in / NV12 out) keep their per-link
    identity.
  - `run_source_transform_sink` extracts both `src_caps = links[0]`
    and `sink_caps = links[1]` from the solver and passes each
    element the side it expects: `source.configure_pipeline(src_caps)`,
    `transform.configure_pipeline(src_caps)` (input side — what
    decoders like `FfmpegH264Dec` validate against), and
    `sink.configure_pipeline(sink_caps)`. M12 allocation queries
    use the downstream-facing `sink_caps`.

  **Regression caught and fixed**: in step 5c, migrating `FakeSink`
  to `AcceptsAny` routed the rtsp e2e chain through the mixed
  cascade, which correctly returned `links=[H264, NV12]`. But the
  pre-5d runner fed `links.last()=NV12` to every element, including
  the decoder, which rejects NV12 with `CapsMismatch`. The e2e test
  is `#[ignore]`d so CI missed it. New regression test
  `format_changing_transform_receives_input_side_caps` in
  `pipeline_smoke.rs` covers this shape without needing ffmpeg.

  **Workaround #2 retired**: `WaylandSink` / `KmsSink` now receive
  NV12 directly at startup (when downstream of a decoder), so their
  "Caps::Video { .. } => no-op" deferred-setup branch is no longer
  load-bearing. The branch is left in place for safety; cleanup is
  a follow-up. Updated `legacy_cascade_with_boundary_transform`
  test to assert per-link semantics.
- Step 5c: `CapsConstraint::AcceptsAny` wildcard sink variant for
  debug / probe / passthrough sinks whose `intercept_caps` is
  `Ok(upstream.clone())`. Solver treats it as no-op narrowing on
  the link (upstream's produced caps flow through unchanged) and
  enforces it at the chain's tail; in a native chain a middle
  `AcceptsAny` is silently invisible, in mixed/legacy forward
  cascade an interior `AcceptsAny` returns `EndpointShapeMismatch`.
  `FakeSink` and `syncsink` (g2g-plugins) override
  `caps_constraint_as_sink` to return `AcceptsAny`; chains
  containing them now exercise the mixed-cascade path. `identity`
  (transform-shape pass-through) stays on the legacy bridge —
  needs an `Identity`-with-wildcard variant which is a separate
  gap. 3 new solver tests; all 95 g2g-core tests + integration
  suites green.
- Step 5b: trait integration so individual elements can migrate.
  `AsyncElement` gains two default methods —
  `caps_constraint_as_sink()` and `caps_constraint_as_transform()` —
  each returning the legacy bridge (`LegacySink` / `LegacyTransform`)
  for today's `intercept_caps` + `propose_output_caps`. The runner
  (`run_simple_pipeline` and `run_source_transform_sink`) now calls
  these methods instead of constructing the bridge inline; migrated
  elements override to return a native `CapsConstraint` and chains
  containing them hit the mixed-cascade solver path. Behavior is
  identical for every existing element (all defaults). Bridge helpers
  `legacy_sink_constraint` / `legacy_transform_constraint` relaxed
  to `?Sized` so they can be called from the trait's default methods
  on `&Self`.
- Step 5a: mixed-chain support in the solver. Chains that mix
  legacy and native `CapsConstraint` variants now route to a new
  `solve_mixed_cascade` (single forward pass that handles every
  variant). Legacy variants and `DerivedOutput` require upstream to
  fixate to one concrete `Caps`, which migration chains satisfy
  because the source is typically single-alternative. Backward
  arc-consistency (Identity / Mapping filtering against downstream
  sinks) is not applied in the mixed path; once a chain becomes
  fully native, dispatch routes it back to the arc-consistency
  solver and backward propagation is restored.
  `NegotiationFailure::MixedLegacyAndNative` is no longer returned
  (kept as a variant in case future mixed shapes need it).
  4 new tests cover legacy-source/native-sink, native-source/legacy-sink,
  native-source/legacy-decoder/native-sink, and sink rejection in a
  mixed chain.
- Step 4c: roll the solver-via-legacy-bridge pattern out to the
  remaining linear runner entry points. `run_simple_pipeline`
  (source → sink) and `run_source_fanout` (source → fanout, with the
  fanout as the linear "sink" of the chain — downstream sinks
  broadcast-receive the fixated caps and don't participate in
  narrowing). `run_muxer_sink` solves each source ↔ muxer-input pair
  via the solver, wrapping the muxer's per-input `intercept_caps`
  as a `LegacySink`. `run_fanin_sink` stays direct: each source
  self-fixates with no peer narrowing, so there's no chain to solve.
  The muxer's aggregated-output → sink half stays as direct
  `fixate()` because today's runner intentionally does *not* call
  `sink.intercept_caps` for that hop (the muxer output is the
  canonical merged caps).
- Step 4b: `run_source_transform_sink` startup negotiation routes
  through `solve_linear` via the legacy bridge. The pre-M16 inline
  cascade (`source.intercept_caps` → `transform.intercept_caps` →
  `sink.intercept_caps` → `fixate`) is replaced by building
  `LegacySource` / `LegacyTransform` / `LegacySink` constraints and
  calling the solver; the cascade output is bit-compatible with what
  the inline path produced. `ReFixate` retry stays in the runner (the
  solver doesn't model counter-proposals); on each retry the
  `LegacySource` seed becomes the counter and the solver re-runs.
  Mid-stream `CapsChanged` paths are untouched. All existing tests
  pass (89 g2g-core, 14 g2g-plugins rtsp lib, all integration
  suites).
- Step 4a: legacy bridge into the solver. `CapsConstraint` gains a
  `'a` lifetime parameter and three transitional variants —
  `LegacySource(Caps)`, `LegacyTransform { intercept, propose_output }`,
  `LegacySink(intercept)` — that capture today's `AsyncElement`
  callbacks. `legacy_transform_constraint(&T)` / `legacy_sink_constraint(&T)`
  helpers wrap a borrowed element. The solver dispatches: all-native
  chains take arc consistency, all-legacy chains take
  `solve_legacy_cascade` (forward cascade that mirrors today's runner,
  then fixates the final caps and propagates to upstream link slots
  the same way `configure_pipeline` is called today). Mixed chains
  return `NegotiationFailure::MixedLegacyAndNative` until step 5
  migrates individual elements. 6 new tests cover the cascade,
  pass-through and boundary transforms, intercept failure, mixed
  chain rejection, and the AsyncElement → LegacyTransform bridge.
- Step 3: linear-pipeline caps solver in
  `g2g-core::runtime::solver` (feature `runtime`).
  `solve_linear(&[&CapsConstraint]) -> Result<Vec<Caps>, NegotiationFailure>`
  walks a source → transform* → sink chain with arc consistency:
  seed endpoint links from `Produces` / `Accepts`, forward+backward
  sweep until fixed point, fixate each link to one concrete `Caps`.
  Handles all four interior constraint shapes — `Identity` couples
  input and output, `Mapping` filters pre-enumerated (in, out) pairs
  to the surviving set, `DerivedOutput` fires once its input link
  fixates. `NegotiationFailure` reports `EmptyLink` /
  `EndpointShapeMismatch` / `Unfixable` / `Degenerate`; `Cyclic` is
  reserved for the non-linear solver. `CapsSet::union` added to
  support the `Mapping` path. 10 unit tests cover the minimal chain,
  empty/disjoint links, degenerate input, endpoint-shape errors,
  preference tie-break, identity coupling and mismatch, derived
  output, and mapping pair selection.
- Step 2: `FormatElement` trait + `CapsConstraint` enum in new
  `g2g-core::format_element` module. `CapsConstraint` variants:
  `Accepts` (sinks), `Produces` (sources), `Identity` (pass-through
  transforms), `Mapping(Vec<(in, out)>)` (pre-enumerated codecs),
  `DerivedOutput(Fn(&Caps) -> CapsSet)` (decoders reading SPS).
  `configure_link(input, output)` replaces `configure_pipeline`;
  boundary elements see distinct sides, sources/sinks see `None` on
  the unused side. `CapsPreferences` is a placeholder for the
  tie-break algebra (DESIGN-M16 §10). Coexists with `AsyncElement`;
  the legacy-bridge blanket impl lands with the solver in step 3,
  because its shape is dictated by what the solver consumes.

### M12: Live-source surface (latency + allocation + clock election)
- Latency query: `LatencyReport { live, min_ns, max_ns }` + `query` module,
  GStreamer-style latency triple with `combine`/`aggregate` (min and max sum
  along the path, unbounded `max_ns = None` is infectious, liveness sticky) and
  `is_unsatisfiable`. `ZERO` contributes `max_ns = Some(0)` (non-buffering),
  distinct from `None` (unbounded buffering). `AsyncElement::latency()` /
  `SourceLoop::latency()` default methods (return `ZERO`); live sources and
  buffering transforms override.
- Allocation query: `AllocationParams { size_bytes, min_buffers, align, domain }`
  + `MemoryDomainKind` (and `MemoryDomain::kind()`). `AsyncElement::propose_allocation`
  / `configure_allocation` and `SourceLoop::configure_allocation` default
  methods let a consumer propose a buffer pool that its producer allocates into
  (zero-copy handoff). `AllocationParams::merge` folds an upstream element's own
  requirement into a downstream proposal (most-demanding size/count/alignment
  wins; consumer dictates domain).
- Clock distribution: `ClockPriority` (`SystemFallback` < `Provider` <
  `LiveSource`), `ClockCandidate` (priority + shared `Arc<dyn PipelineClock>`),
  and `elect_clock` (highest priority wins, ties resolve to the most upstream).
  `AsyncElement::provide_clock` / `SourceLoop::provide_clock` default methods
  let a live source or sink offer its clock; the runner adopts the elected clock
  over the supplied fallback.
- Linear runners (`run_simple_pipeline`, `run_source_transform_sink`) fold the
  configured chain into `RunStats::latency`, resolve the allocation query into
  `RunStats::allocation`, and elect the pipeline clock into
  `RunStats::{clock_priority, base_time_ns}` after negotiation. Fan-in /
  fan-out runners report neutral values (topology aggregation deferred).
- M12 complete: with M8–M12 done, `g2g` reaches dynamic-pipeline feature parity
  with GStreamer (per DESIGN.md §4.10) while keeping the static typed layer.

### M15: RTSP reconnect + long-running stability soak
- `RtspSrc` now supports reconnect-with-backoff. Off by default for
  backwards compatibility; opt in with `.with_reconnect(max_attempts)`
  and optionally `.with_reconnect_backoff(initial_ms, max_ms)`.
  Exponential backoff caps at `max_ms`. Network/protocol errors trigger
  retry; a graceful server-side end-of-stream (retina's demuxer returning
  `None`, typical for VOD finishing) terminates without retry.
- `run_rtsp` refactored into an outer reconnect orchestrator and inner
  `run_session`. State threaded across sessions:
  - cumulative `sequence` counter so downstream sees monotonic IDs;
  - `pts_base_ns` offset so per-session PTS continues monotonically
    across reconnects (with a deliberate 1 s gap marking the boundary).
- A `PipelinePacket::Flush` is emitted before each reconnect so the
  decoder flushes its codec state and sinks reset `last_sequence`.
- `tests/rtsp_soak.rs` (#[ignore]) is a long-running stability soak
  configurable via `G2G_SOAK_SECONDS` (default 30) and `G2G_SOAK_MIN_FRAMES`
  (default `seconds * 20`). Asserts monotonic `sequence` and `pts_ns`,
  and that the pipeline reaches the frame floor — catches PTS regressions
  and stalls that the 2-second smokes miss. Module docs cover the manual
  reconnect-exercise workflow (Ctrl-C the publisher, watch it resume).
- `tests/rtsp_smoke.rs` gains `rtspsrc_with_reconnect_retries_then_fails`
  which exercises the reconnect orchestrator against an unreachable URL,
  asserting the source retries `max_attempts` times before surfacing an
  error.

### M14: Wayland display sink (Linux, NV12, desktop-dev convenience)
- `WaylandSink` element (`wayland-sink` feature, Linux-only): opens an
  `xdg_toplevel` window on the running compositor and presents NV12 frames
  via `wl_shm` after software conversion to XRGB8888 (BT.601 limited range).
  Designed as the desktop-dev companion to `KmsSink` — same NV12 input
  contract so the upstream pipeline stays identical.
- Threading: a dedicated worker thread owns all Wayland state (Connection,
  EventQueue, SlotPool); the sink struct holds only a calloop channel and
  an `Arc<AtomicU64>` counter, both `Send + Sync`. SCTK's
  `calloop_wayland_source` multiplexes Wayland events and frame arrivals
  in a single event loop. A one-shot Mutex/Condvar handshake gates the
  sink-side `configure_pipeline` on the first compositor `configure`.
- Pulls `smithay-client-toolkit` 0.20 (transitively bringing the
  `wayland-*` family) under `[target.'cfg(target_os = "linux")'.dependencies]`.
- Constraints (v1): NV12 only, fixed input dims, no scaling (compositor
  letterboxes/clips if its configure suggests a different size), no PTS
  pacing, software conversion only (zero-copy via `zwp_linux_dmabuf_v1` is
  deferred).
- `KmsSink` is the production low-latency sink; `WaylandSink` is for
  iterating on the pipeline inside your desktop session without dropping
  to a tty.

### M14: KMS/DRM display sink (Linux, NV12)
- `KmsSink` element (`kms-sink` feature, Linux-only): primary-plane scanout
  of NV12 `DataFrame`s on the first connected connector + CRTC of the
  configured DRM device (defaults to `/dev/dri/card0`). Two-buffer dumb-
  buffer pool; first frame goes through `set_crtc`, subsequent frames page-
  flip and the next submission blocks on the prior flip's `PageFlip` event
  so the buffer being overwritten is off scanout (tearing-free).
- `FfmpegH264Dec::with_output_format(OutputFormat::Nv12)` (M14 prerequisite,
  separate commit) interleaves the U/V planes after decode without swscale;
  same total byte length as I420. I420 remains the default.
- New optional, target-gated deps: `drm` 0.15 + `drm-fourcc` 2 under
  `[target.'cfg(target_os = "linux")'.dependencies]`; `kms-sink` implies
  `std`. No `unsafe` and no GBM dependency — pure dumb-buffer path.
- Constraints (v1): NV12 only, fixed input dims (mid-stream geometry change
  not supported), no letterboxing/scaling (buffer scans out at native dims;
  smaller-than-mode video shows at origin with stale framebuffer around it),
  requires DRM master (tty or DRM lease; a running compositor will block).
- Deferred (v2): overlay-plane path with src/dst rectangles for proper
  letterboxing; async page flips for lower latency; Wayland sink as a
  desktop-dev convenience using the same NV12 input contract.

### M13: End-to-end RTSP → ffmpeg decode (Linux software path)
- `RtspSrc::intercept_caps` now advertises fixate-friendly `Dim::Range` /
  `Rate::Range` instead of `Any`. `Caps::fixate()` rejects `Any` and aborted
  Phase 2 negotiation before any network handshake; the placeholder is
  overwritten by the SDP-derived `CapsChanged` emitted from `run`.
- `RtspSrc` drops every `VideoFrame` until the first `is_random_access_point`
  IDR. retina's `FrameFormat::SIMPLE` only prepends SPS/PPS on keyframes, so
  mid-GOP tune-in (typical for live RTSP servers like MediaMTX) would feed
  parameterless slices to the decoder and stall with "non-existing PPS 0".
- New `rtsp_ffmpeg_e2e` integration test (`rtsp` + `ffmpeg` features, Linux):
  `RtspSrc → FfmpegH264Dec → FakeSink` over `run_source_transform_sink`,
  asserting decoded I420 frames reach the sink with a fixed-dim `CapsChanged`
  preceding the first `DataFrame`. Module docs include a MediaMTX + ffmpeg
  recipe for a deterministic local fixture.

### M13: Windows hardware decode (Media Foundation)
- `MfDecode` element (`mf-decode` feature, Windows-only): wraps the Media
  Foundation H.264 Decoder MFT (`CLSID_MSH264DecoderMFT`, an `IMFTransform`).
  Consumes Annex-B H.264 `DataFrame`s and emits decoded NV12 frames as
  `MemoryDomain::System`, with a `CapsChanged(Nv12)` before the first frame and
  on each decoder stream change. Implements the canonical feed/drain MFT loop
  (`ProcessInput`/`ProcessOutput`, `NEED_MORE_INPUT`/`STREAM_CHANGE`,
  `COMMAND_DRAIN` on EOS, `COMMAND_FLUSH` on seek).
- New optional, target-gated dependency: `windows` 0.62 under
  `[target.'cfg(windows)'.dependencies]`; the `mf-decode` feature implies `std`.
- `HardwareError::MediaFoundation(i32)` carries the failing `HRESULT`.
- Constraint: COM is initialised MTA; the element is thread-affine and intended
  for a single-thread executor (asserted `Send` under a documented contract).
- Deferred: D3D11 zero-copy output (needs a new `MemoryDomain` variant), DXVA
  hardware acceleration, and strided (`MF_MT_DEFAULT_STRIDE`) NV12 copy.

### M11: Application control surface
- `Bus` + cloneable `BusHandle` + `BusMessage` (Eos/Error/Warning/Custom): mp-sc message channel so elements notify the app asynchronously without back-references. Non-blocking `try_post`; `try_recv`/`recv` on the app side.
- `LinkInterceptor` probes: a `Pass`/`Drop` interceptor installed on a link's `SenderSink` via a runtime `ProbeSlot` (GStreamer pad-probe equivalent). Empty by default, so existing links are unaffected.
- `PipelinePacket::Flush`: non-terminal seek-flush packet; elements reset position (sinks drop `last_sequence`) and forward/broadcast it, the stream resumes afterwards.
- Deferred: blocking probe action.

### M10: True muxer fan-in
- `MultiInputElement` trait + `run_muxer_sink`: combine all N inputs into one output (vs the M9 Merger selector), with per-input caps negotiation and EOS aggregation (one `Eos` after every input ends).
- `InterleaveMux` element; tagged-merge-channel runner so one `&mut` muxer task processes all inputs without a select primitive.

### M9: Dynamic fan-out / fan-in
- `Router` (1->N): routes each frame to an atomic-selected output port, broadcasts `CapsChanged` to all ports.
- `Gate` (1->1): forwards or drops data by an atomic flag; a plain `AsyncElement`, so it needs no new runner.
- `Merger` (N->1): selects one active input, drains the rest, emits one merged `Eos` after every input ends.
- Multi-output / fan-in plumbing: `MultiOutputSink`, `MultiOutputElement`, `DynSourceLoop`, `join_all`, and the `run_source_fanout` / `run_fanin_sink` runners. Each primitive has a cloneable control handle (`RouterHandle` / `GateHandle` / `MergerHandle`).
- Deferred: `BranchSlot` + `SwapPolicy` variants (need a task spawner).

### M8: Caps negotiation + dynamic element swap
- Caps algebra: `Dim`/`Rate`/`Caps` `intersect`, `is_fixed`, `fixate` (Phase 1 narrowing, Phase 2 fixation per §4.2).
- Runners fixate negotiated caps before configuring elements; bounded `ReFixate` retry; mid-stream `CapsChanged` cascade and upstream `Reconfigure` sideband.
- `ElementSlot` + `SwapHandle`: lock-free atomic hot-swap of one element mid-stream; `DynAsyncElement` blanket impl so any element boxes into a slot.

## M0-M5 (prior)
- M0/M1: Cargo workspace scaffold and minimal `no_std + alloc` pipeline runtime (bounded channel, `Join2`, linear runner).
- M2: async `OutputSink`, identity transform, 3-element runner.
- M3: `AsyncClock` + `WallClock` + `SyncSink` for PTS-paced presentation.
- M4: Arc-recycled `BufferPool` with async `acquire`; DMABUF close-on-drop fix.
- M5: `RtspSrc` element wrapping `retina`.

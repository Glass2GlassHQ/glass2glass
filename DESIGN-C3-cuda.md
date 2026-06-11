# C3 — Zero-copy NVDEC → GPU display (CUDA memory domain)

> Phased plan for keeping NVDEC-decoded NV12 resident on the GPU and
> presenting it without a device->host round-trip. Phases 1 and 2 have
> landed (see `CHANGELOG.md`); this doc records the design and the Phase 3
> research that redirected the consumer away from KMS/dmabuf.

## 1. Goal

`Backend::NvdecCuvid` runs the standalone `h264_cuvid` codec, which decodes
on the GPU and then copies NV12 back to system memory. The pipeline then
pays a second host copy in `copy_yuv420` and, for `WaylandSink`, a CPU
NV12 -> XRGB conversion. The glass-to-glass floor is dominated by those
host copies and the CPU convert.

C3 keeps the decoded NV12 in CUDA device memory end-to-end so a GPU
consumer (display) takes the handoff without a PCIe round-trip.

## 2. What landed (Phases 1-2)

- **`MemoryDomain::Cuda(OwnedCudaBuffer)` + `MemoryDomainKind::Cuda`**
  (`g2g-core`, platform-agnostic, `no_std`). `OwnedCudaBuffer` carries the
  two NV12 plane device pointers (luma Y, interleaved chroma UV), row
  pitches, dims, the `CUcontext`, and a boxed `CudaKeepAlive` owner. Core
  never links CUDA: the producing element supplies the owner as a trait
  object, and dropping the buffer releases the backing allocation.
- **`AllocationParams::cuda(...)`** makes `MemoryDomainKind::Cuda` the first
  cross-element pool domain crossing a real producer/consumer boundary (the
  driver the M18 item-1 allocation re-cascade beta was missing). Conveyance
  is covered by GPU-free runner tests in `m12_allocation.rs`.
- **`Backend::NvdecCuda`** (`g2g-plugins`, `ffmpeg` feature, Linux+NVIDIA):
  the generic `h264` decoder with an `AV_HWDEVICE_TYPE_CUDA` device and a
  `get_format` hook selecting `AV_PIX_FMT_CUDA`. Emits `MemoryDomain::Cuda`
  frames; the owning `AVFrame` is the keep-alive. NV12 only. Not yet
  compiled/verified (Linux+GPU owed).

## 3. Phase 3: the consumer problem

A CUDA memory domain only pays off with a GPU-side consumer. The original
plan was "`KmsSink` zero-copy scanout via dmabuf". Research says that is the
wrong mechanism on NVIDIA.

### 3.1 Why not KMS / dmabuf

- CUDA can export device memory to a dma-buf fd via
  `cuMemGetHandleForAddressRange(..., CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD)`,
  but **only for VMM-allocated memory** (`cuMemCreate` / `cuMemMap`). NVDEC
  decoder frames come from libavcodec's CUDA hwframe pool, not VMM, so they
  cannot be exported directly: it would need a device->VMM copy first,
  re-introducing a copy and defeating the point.
- The NVIDIA proprietary driver historically **exports** dma-buf but will
  not **import** dma-buf created by other drivers, and KMS scanout of a
  CUDA-exported dma-buf through `nvidia-drm` is unproven/fragile. The whole
  `KmsSink` dumb-buffer path assumes CPU-writable buffers anyway.

Conclusion: do not route CUDA frames through `KmsSink` / dmabuf.

### 3.2 CUDA <-> OpenGL interop (recommended)

The well-trodden path on the NVIDIA proprietary driver (what GStreamer's
`nvcodec` + `glimagesink` and NVIDIA's `FramePresenterGL` sample do):

1. Create a GL context on the display surface (EGL).
2. Register a GL texture (or PBO) with `cuGraphicsGLRegisterImage` /
   `cuGraphicsGLRegisterBuffer` once.
3. Per frame: `cuGraphicsMapResources`, then `cudaMemcpy2D`
   (device -> device, honouring the source pitch) the NV12 planes into the
   GL resource, then `cuGraphicsUnmapResources`. No host round-trip.
4. Sample the two NV12 planes (Y + interleaved UV) in a fragment shader,
   convert BT.601/709 to RGB on the GPU, present via `eglSwapBuffers`.

This works because it is the driver's own CUDA/GL interop, not cross-driver
dma-buf. It is not literally zero-copy (one device->device copy into the GL
texture), but it removes the PCIe round-trip and the CPU colour convert,
which is the latency win.

### 3.3 CUDA <-> Vulkan external memory (alternative)

Import a Vulkan image's memory into CUDA via `cudaImportExternalMemory`
(VK exports `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD`), write the decoded
NV12 into it, present with Vulkan. More modern and the long-term direction,
but a heavier stack to stand up than GL. Defer.

### 3.4 `CudaDownload` fallback (low-risk safety net)

A transform that copies a `MemoryDomain::Cuda` frame to
`MemoryDomain::System` (NV12, `cudaMemcpy2D` device -> host). It negates the
latency win but lets a `NvdecCuda` stream reach the existing CPU sinks
(`WaylandSink` / `KmsSink`) for correctness and bring-up before the GL sink
exists. Its caps constraint is `Identity(NV12)`; only the domain changes.
This is the first thing to build in Phase 3 because it makes the CUDA
backend usable and testable end-to-end (frame counts, geometry) even though
the device pointers themselves still need a GPU to exercise.

## 4. Recommended Phase 3 plan

1. **`CudaDownload`** element (§3.4). Smallest CUDA FFI surface
   (`cuMemcpy2D` D->H + context push/pop). Unblocks `NvdecCuda -> download ->
   existing sink` so the decode path is verifiable before any GL work.
2. **`CudaGlSink`** (new sink, §3.2): EGL context on a Wayland surface
   (reuse the `WaylandSink` windowing approach via SCTK + `wl_egl_window`)
   or GBM/KMS for the tty case; CUDA-GL registered texture; NV12 shader.
   This is the real zero-copy-ish display payoff. Largest lift; new EGL/GL
   + CUDA-GL FFI surface, behind a new feature (e.g. `cuda-gl`).
3. Wire the allocation query so `CudaGlSink` proposes
   `MemoryDomainKind::Cuda` and `NvdecCuda` honours it (the cross-element
   handshake Phase 1 already conveys).

## 5. Verification gaps

Everything in Phases 2-3 is `ffmpeg`/CUDA + Linux + NVIDIA-GPU and does not
compile on the Windows dev host. CI has no GPU. All of it is owed as
user-side e2e, same constraint as the existing rtsp/kms/wayland elements.
A `wayland_smoke`-style manual benchmark (`rtspsrc -> h264parse ->
ffmpegdec[NvdecCuda] -> CudaGlSink`) is the acceptance test; compare p50/p95
against the `NvdecCuvid -> WaylandSink` system-memory baseline.

## 6. Open decisions

- **GL-on-Wayland vs GL-on-KMS first.** Wayland reuses `WaylandSink`'s
  windowing and is the dev-loop sink; KMS/GBM is the production tty path.
  Lean Wayland first (faster iteration), KMS second.
- **Feature naming / crate placement** for the CUDA FFI: a thin internal
  `cuda` module in `g2g-plugins` vs a dedicated dependency
  (`cudarc` / `cust`). A maintained crate (e.g. `cudarc`) would avoid
  hand-rolled driver-API FFI; evaluate before writing raw bindings.

## 7. References

- CUDA dma-buf export limits: NVIDIA Developer Forums "CUDA and Linux
  DMA-BUF"; `open-gpu-kernel-modules` discussion #243 (export yes, import
  no).
- CUDA VMM / shareable handles: CUDA Driver API "Virtual Memory
  Management" and `CUDA_VA` group docs.
- CUDA-GL interop: CUDA Driver API `CUDA_GL` group; GStreamer
  `gst-plugins-bad` nvcodec GL interop (MR !614); NVIDIA
  `FramePresenterGL` (Video Codec SDK samples).

# Rerun `re_video` proof-of-concept: a vendor-neutral hardware-decode extension point

This directory captures a working proof that **Rerun's `re_video` can host an
out-of-tree hardware video decoder** through a small, backend-agnostic extension
point, and that glass2glass' vendor-neutral **Vulkan Video** decoder plugs into it,
including the zero-copy path where a decoded `wgpu::Texture` is rendered directly by
`re_renderer` with no CPU readback.

The design is deliberately shaped for upstreaming: **`re_video` gains a generic
registry and a generic GPU-resident frame type, and depends on nothing from
glass2glass.** The g2g binding lives entirely in a separate `re_video_g2g` crate.
That is the difference from the earlier iteration of this PoC (which wired g2g
directly into `re_video` behind a `g2g-vulkan` feature): the coupling is now one
directional and the extension point is vendor-neutral, so the `re_video` change is
an RFC candidate on its own, independent of glass2glass.

The sibling in-repo test `g2g-plugins/tests/m508_revideo_adapter.rs` proves the
adapter (`g2g_plugins::revideo::VulkanStreamDecoder`) satisfies re_video's
chunk-at-a-time contract; this PoC closes the loop by driving the *real* `re_video`
crate with g2g registered as a backend.

## What the fork changes

Against a `re_video` checkout (see pin below), the patch
`re_video-g2g-vulkan.patch` makes three sets of changes.

### 1. `re_video`: a vendor-neutral extension point (no glass2glass dependency)

- **Hardware-decode registry** (`decode/mod.rs`). A process registers a factory via
  `re_video::register_hw_video_decoder(...)`; `new_decoder` consults registered
  backends *before* its built-in software decoders. A factory either **handles** a
  video (`HwDecoderAttempt::Handled(Ok(decoder))`) or **declines**
  (`HwDecoderAttempt::NotSupported(sender)`, handing the output channel back) so the
  search falls through to the next backend or to re_video's own ffmpeg / dav1d path.
  A backend-agnostic `DecodeError::HwDecoder(String)` variant carries backend errors.
  None of this names any specific decoder crate.
- **Generic GPU-resident frame** (`decode/gpu_texture.rs`, new), behind a new
  `gpu-textures` cargo feature. `GpuVideoFrame { texture, device, queue, adapter }`
  is a plain `wgpu::Texture` plus the wgpu handles it is bound to, with a
  `read_rgba()` verification helper. `FrameContent` gains a feature-gated
  `gpu_texture: Option<GpuVideoFrame>`; when `Some`, `data` is empty (the frame was
  never read back) and a consumer renders the texture directly.
- **`Cargo.toml`.** The `gpu-textures` feature pulls an *optional* `wgpu` dependency;
  the default build takes no `wgpu` dependency and no glass2glass dependency at all.

### 2. `re_video_g2g` (new crate): the glass2glass backend

An out-of-tree crate that implements the extension point:

- `re_video_g2g::register()` registers a factory that maps H.264 / H.265 / AV1 onto
  `g2g_plugins::revideo::VulkanStreamDecoder`, declining (so re_video falls back to
  software) when no Vulkan decode device opens (`can_open_device`), and declining
  codecs g2g's Vulkan path doesn't cover (VP8 / VP9 / images).
- `impl AsyncDecoder for G2gVulkanDecoder` runs the decoder on its **own OS thread**
  fed by a command channel, so `submit_chunk` is non-blocking (the async contract
  Rerun's player expects) and frames arrive on the output channel as they decode. Its
  `Drop` **joins** that worker, so all Vulkan/wgpu teardown completes on the worker
  before drop returns.
- For **real container input** it takes codec parameters out of band: `extract_config`
  pulls the `avcC` / `hvcC` / `av1C` record from the video's `stsd` box and builds via
  `VulkanStreamDecoder::from_config`, which reframes the length-prefixed (AVCC) samples
  to Annex-B itself; a raw elementary stream (no `stsd`) falls back to in-band
  parameters from the first chunk.
- Tier B (zero-copy) is opt-in via `re_video_g2g::set_prefer_gpu_textures(true)`: the
  backend produces a `re_video::GpuVideoFrame` from g2g's decoded texture and the decode
  device's wgpu handles.
- `Cargo.toml` path-deps `g2g-plugins` (feature `vulkan-video`) and `g2g-core` (feature
  `runtime`, for `block_on`), plus `re_video` with `features = ["gpu-textures"]`.

### 3. `re_renderer`: consume the decoded texture with no copy

Behind a new `gpu-textures` feature (`= ["re_video/gpu-textures", "dep:re_video_g2g"]`):

- **Adopt an external texture with no copy.** `GpuTexturePool::adopt_external_texture`
  wraps an externally-created `wgpu::Texture` as a re_renderer `GpuTexture` with a real
  pool handle (so bind-group creation resolves it), backed by a new
  `DynamicResourcePool::alloc_external`. A `GpuTextureInternal::external` flag makes the
  pool **never** call `destroy()` on it (that would double-free a texture re_renderer
  does not own); the image is freed only when the source and this handle drop, via wgpu's
  reference counting. **This piece is fully generic** (it names no decoder) and is the
  smallest independently-upstreamable change.
- **Construction-site fix.** `re_renderer`'s own `video/chunk_decoder.rs` builds a
  `FrameContent`; the feature-gated `gpu_texture` field is set to `None` there too, so
  re_renderer compiles when re_video's `gpu-textures` is on.

The `re_video_g2g` path deps use an absolute path to this repo's `g2g-plugins` /
`g2g-core`; adjust it in the patch if your checkout lives elsewhere.

## Why the split matters (make-or-break: one wgpu)

`re_video::GpuVideoFrame` is built on Rerun's workspace `wgpu` (29.0); `re_video_g2g`
brings glass2glass' `wgpu` (29.x) via `g2g-plugins`. Cargo unifies both to a single
`wgpu` crate version, so the decoded texture **handle crosses** from g2g into
`re_video::GpuVideoFrame` with no copy and the types match. If those ever diverge to
semver-incompatible majors, Tier B stops compiling. (Verified here: both resolve to
`wgpu 29.0.3`.)

## Result (RTX 3060, this host)

Four tests on the real crates. Three in `re_video_g2g` (the backend), one in
`re_renderer` (the render last-mile):

```
# re_video_g2g:
mp4_decode:    all 10 H.264 frames from a real .mp4 (avcC + AVCC samples) via re_video::load_mp4 + new_decoder, bit-exact I420 vs ffmpeg
gpu_texture:   10 GPU-resident RGBA textures from a real .mp4 via re_video::new_decoder, zero-copy (no CPU readback in decode); match ffmpeg
vulkan_decode: all 10 H.264 frames decoded through re_video::new_decoder (threaded g2g backend), bit-exact I420 vs ffmpeg

# re_renderer:
g2g_tier_b_render: g2g-decoded wgpu::Texture adopted + rendered by re_renderer's rectangle renderer (offscreen, on the decode device), render output matches the decoded texture (interior mean abs diff ~3.5)
```

- **`re_video_g2g/tests/mp4_decode.rs` (real container).** Demuxes an actual `.mp4`
  with re_video's own `VideoDataDescription::load_mp4`, then feeds every sample (a
  byte-span read via the player's `VideoSliceSource`, exactly as Rerun's player builds
  chunks) through `new_decoder`. Parameter sets come out of band from the `avcC` box and
  the samples are AVCC; the g2g backend handles both. Every one of the 10 frames is
  **bit-exact I420** vs a software ffmpeg reference. This proves g2g decodes what Rerun
  actually logs, not just a raw elementary stream.
- **`re_video_g2g/tests/gpu_texture.rs` (Tier B, zero-copy).** Same real `.mp4` through
  the same API, with `set_prefer_gpu_textures(true)`: each `Frame` arrives GPU-resident
  on `FrameContent::gpu_texture` (a `re_video::GpuVideoFrame`) with empty `data`. The
  test reads each texture back only to verify it, matching the software reference via the
  same BT.601 matrix. This is the moat: hardware decode straight into a `wgpu::Texture`.
- **`re_video_g2g/tests/vulkan_decode.rs` (raw elementary stream).** Builds a
  `VideoDataDescription` for the raw H.264 fixture, feeds one access-unit chunk per
  frame, and asserts the same bit-exactness. Clean exit (no double-free) on the threaded
  teardown path that previously crashed.
- **`re_renderer/tests/g2g_tier_b_render.rs` (render last-mile).** Drives a Tier B frame
  into `re_renderer`'s real render path and reads back the RENDER OUTPUT (not the raw
  texture) to prove re_renderer sampled it (below).

### Teardown crash: fixed

An earlier version drove g2g through re_video's built-in `SyncDecoderWrapper`, whose
`Drop` **detaches** (does not join) its worker; that thread's Vulkan teardown raced
process exit and produced a non-deterministic `free(): invalid size` double-free. Two
fixes: (1) g2g `open_decode_device` (M510) enumerates Vulkan adapters and picks a
discrete GPU that passes the codec's decode probe, instead of trusting
`request_adapter(HighPerformance)` (which on a multi-GPU host can hand back an adapter
with no decode queue for the codec); (2) the `re_video_g2g` adapter owns its worker
thread and **joins it on `Drop`** rather than reusing `SyncDecoderWrapper`.
`g2g-plugins/tests/vulkan_thread_teardown.rs` proves create+decode+drop on joined worker
threads is clean.

### Codec caveat (AV1 off-main-thread)

H.264 and H.265 Vulkan decode are **bit-identical whether run on the main thread or a
spawned worker** (proven by `h264_/h265_decode_matches_across_threads`), so the threaded
integration is bit-exact for them. AV1 decode is bit-exact on the main thread but exhibits
a small, run-varying residual on the late (compound / temporal-MV) inter frames when
driven from a spawned thread on this host's NVIDIA driver; every g2g-fed `Std*` param is
byte-identical across threads and the GPU op sequence is identical, so the residual is
isolated to the driver's AV1 decode, not g2g. This PoC therefore demonstrates
bit-exactness with H.264.

## Reproduce

1. Clone Rerun and check out the pinned commit (a partial/blobless clone is fine):

   ```
   git clone https://github.com/rerun-io/rerun
   cd rerun
   git checkout ef9d94e9cf1af999a114bb0b815abcd3f0c0c94c
   ```

2. Apply the patch (adjust the g2g path inside `re_video_g2g/Cargo.toml` if needed):

   ```
   git apply /path/to/glass2glass/docs/rerun-poc/re_video-g2g-vulkan.patch
   ```

3. Provide the test fixtures (not embedded here to keep the patch text-only): the raw
   H.264 stream, a software `yuv420p` reference, and a real `.mp4` (remux the stream, so
   `-c:v copy` keeps the bytes identical, giving `avcC` + AVCC samples). Both the
   `re_video_g2g` and `re_renderer` tests read them from `re_video`'s fixtures dir:

   ```
   mkdir -p crates/utils/re_video/tests/fixtures
   cd crates/utils/re_video/tests/fixtures
   cp /path/to/glass2glass/g2g-plugins/tests/fixtures/h264_640x480.h264 .
   ffmpeg -y -i h264_640x480.h264 -f rawvideo -pix_fmt yuv420p h264_640x480_ref.yuv
   ffmpeg -y -i h264_640x480.h264 -c:v copy -f mp4 h264_640x480.mp4
   cd -
   ```

4. Build + run (system stable toolchain works; the repo pins 1.92.0 but 1.96 builds it):

   ```
   # The backend crate (Tier A + Tier B), skips gracefully with no decode adapter:
   cargo +stable test -p re_video_g2g --release -- --nocapture

   # The render last-mile (needs re_renderer's gpu-textures feature):
   cargo +stable test -p re_renderer --features gpu-textures \
     --release --test g2g_tier_b_render -- --nocapture
   ```

   Every test skips gracefully (no assertion) on a host with no Vulkan H.264 decode
   adapter or no distinct compute queue.

## Tier B (true zero-copy): device identity

Tier A is GPU decode -> CPU I420 readback -> re_renderer re-uploads + YUV->RGB. Tier B
keeps the frame GPU-resident: the backend hands `re_renderer` the decoded `wgpu::Texture`
directly (YUV->RGB already applied by g2g's `VkSamplerYcbcrConversion` compute pass),
skipping both the readback and re_renderer's upload + colour convert.

**Device-identity constraint.** Zero-copy does *not* work by re_renderer handing its
device to the decoder. g2g must **create** the decode device itself: it enables Vulkan
video-decode queue families (and a distinct compute family for the ycbcr pass) that a
render-only device does not request. So the texture is bound to *g2g's* device, and the
correct direction is the reverse: `re_renderer` runs on the decode device, built on the
`GpuVideoFrame`'s carried `adapter` / `device` / `queue`. On a single-GPU host that
device is the display GPU too. On a split host (decode dGPU + present iGPU, like this
one) the decode and display devices cannot be the same, so a cross-device copy is
unavoidable and Tier A is the honest path there.

### The render last-mile

`re_renderer/tests/g2g_tier_b_render.rs` builds `RenderContext` on the carried device,
adopts the decoded texture with `adopt_external_texture` (no copy), wraps it as an
unorm-RGBA `ColormappedTexture` (sRGB decode off: g2g's output is already BT.601 RGB in
`Rgba8Unorm`), and draws it with re_renderer's real `RectangleRenderer` into an offscreen
`ViewBuilder` (top-left-origin orthographic, 1:1 texel->pixel, MSAA off). A scheduled
screenshot reads the render output back; it matches the decoded texture's own contents to
within a ~1.4% mean channel difference (sub-texel nearest sampling), i.e. re_renderer
sampled that exact GPU frame.

## Still owed (in Rerun's viewer)

- Negotiating the Tier A/B choice from which device `re_renderer` renders on, rather than
  the `set_prefer_gpu_textures` PoC setter.
- Wiring texture adoption into Rerun's actual video space-view (this PoC proves the render
  path with a standalone offscreen view).
- The upstream RFC: propose the `re_video` registry + `GpuVideoFrame` + the generic
  `re_renderer` `adopt_external_texture` as the extension point, with `re_video_g2g` as
  the reference implementer.

Pinned Rerun commit: `ef9d94e9cf1af999a114bb0b815abcd3f0c0c94c`

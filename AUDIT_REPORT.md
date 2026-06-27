# Code audit report (2026-06)

Source: `/code-audit-pro` run over the workspace, interrupted midway. This file
records what was fixed, what is left, and which parts of the tree the audit
reached versus which still need a pass.

## Summary

- 81 fix commits landed this session (`786779d..HEAD`, oldest is
  `786779d reject zero-fan tee/muxer in graph finish`; everything at/below
  `f6b7a7b` is prior milestone work, not audit fixes).
- Each commit is one logical fix. Relevant unit tests were run before every
  commit on this Windows host.
- Nothing is pushed (`git push` fails here: no publickey for the remote).
- 8 findings remain open: code that no toolchain on this Windows host can build.
  Per decision, the low-risk ones were committed flagged-unverified and the
  unsafe-FFI leaks are documented below with exact fixes.

## Verification matrix (this host)

CAN build/test here: Windows (MF, WASAPI, D3D11), `wasm32-unknown-unknown`
(`--cfg=web_sys_unstable_apis`), webrtc (str0m, pure Rust), vello-overlay (real
GPU adapter present, render tests run), pyo3.

CANNOT build here: CUDA / nvdec / cudawgpu / cudaglsink / cudakmssink / glnv12
(Linux + NVIDIA), Apple VideoToolbox vtdecode / vtencode (macOS), Android
camera2src / mediacodec* (no `aarch64-linux-android` target or NDK), ffmpeg
ffmpeg* (Linux), pipewire (Linux).

## Resolved after the Windows session (Linux + Android host)

A follow-up pass on a Linux + NVIDIA box (CUDA, ffmpeg, pipewire present) and a
connected Android device closed the toolchain gap:

- The 8 "compile-unverified" fixes below all build + clippy clean on Linux; the
  ffmpeg ones pass their tests (199 cases). No regressions.
- Clippy was not run on the Windows host: two committed fixes left `-Werror`
  clippy regressions (`subparse` needless `Some(?)` `8ef417b`; `h264parse`
  `u32::MAX` tautology `397025f`) and the `wavsink` conformant-float-header fix
  left a stale std-gated test (`m34` `62a77ec`). All three fixed.
- `mediacodec_wgpu` convert leak: RESOLVED (`dfe7095`), on-device validated.
- camera2src / mediacodecdec packer dedup: RESOLVED (`d577173`), on-device
  validated. (The MediaCodec queue/poll skeleton dedup is still open.)

## Open findings

### Committed, compile-unverified here (low risk)

Single-file logic / arithmetic / dead-code changes that preserve the
non-error-path behavior. NOTE: all confirmed to compile + clippy clean on a
Linux + NVIDIA host in the follow-up pass; the macOS `vt*` two still need a Mac.

| Finding | Commit |
| :--- | :--- |
| vtdecode: drop dead `input_caps` field | `5947321` |
| vtencode: reject CapsChanged geometry != session | `7483738` |
| ffmpegaacenc: u128 PTS-to-ns, no overflow | `79e25bc` |
| ffmpegdec: bound + flush `pts_to_arrival` | `86bcc28` |
| ffmpegenc: drain pts map on output | `d796278` |
| pipewiresink: drain PCM tail before EOS teardown | `ef948ff` |
| glnv12: cache shader locations at link | `b73a32c` |
| cudawgpu: remove dead `cuda_copy_nv12_planes` | `5a430dd` |

### Documented, not committed (unsafe-FFI cleanup / cross-file move)

These need the platform toolchain to verify handle cleanup or feature-gated
wiring. Exact fixes:

- **nvdec leak** (`nvdec.rs`): the CUDA context + cuvid ctx-lock are created in
  `open()` but only owned/freed by `CuvidDecoder` (created lazily on the first
  decoded picture). If configured but no decoder is ever created, `NvDec::drop`
  only destroys the parser, leaking both. Fix: in `NvDec::drop`, when
  `decoder_owner.is_none()`, also `cuvid_ctx_lock_destroy(ctx_lock)` then
  `cu_ctx_destroy(context)`, mirroring `CuvidDecoder::drop`.

- **cudawgpu export leak** (`cudawgpu.rs`, `build_entry`): `export_nv12_image`
  returns a `SharedNv12Image` (Vulkan image/memory/FD) that has no `Drop`. If
  the following `import_image_into_cuda(&shared, ...)?` fails, `shared` drops
  without freeing. Fix: match the import result; on `Err`, free the image/memory
  and close the FD (same ash calls the texture drop callback uses) before
  returning.

- **cudawgpu primary-ctx leak** (`cudawgpu.rs:372` and the sibling at `:512`):
  `cuDevicePrimaryCtxRetain` is matched by a release later, but the `check(..)?`
  calls in between early-return past the release, leaking the retain and leaving
  the context current. Fix: wrap the retain in an RAII guard (e.g.
  `PrimaryCtxRelease(i32)`) so release runs on every path.

- **cudaglsink EGL leak** (`cudaglsink.rs:292`): a mid-stream resolution change
  calls `shutdown()` to respawn the worker, but the worker's EGL
  display/context/surface are not destroyed on exit, so each resize leaks an EGL
  context + surface. Fix: destroy them in the worker terminate path
  (`eglDestroySurface` / `eglDestroyContext` / `eglTerminate`), or hold them in a
  guard dropped when the worker loop exits.

- **camera2src / mediacodecdec dedup** (`camera2src.rs:433`,
  `mediacodecdec.rs:695`): the `YUV_420_888 -> NV12` packer was byte-identical,
  differing only in the return wrapper. RESOLVED (`d577173`): extracted to
  `yuv420::pack_yuv420_to_nv12`. The MediaCodec queue/poll skeleton + flag
  constants are still duplicated decoder/encoder; that larger dedup is left as a
  follow-up. Validated on device.

- **ffmpegaacenc missing test**: the AAC encode core has no end-to-end test;
  writing one needs a Linux ffmpeg build to run it.

## Earlier deferral (test strength, not a bug)

`g2g-plugins/tests/m188_stacked_auto.rs` asserts frame count, not output bytes.
An appsink byte-capture attempt was reverted (claim-once `configure_pipeline`
reset the capture). The capsfilter pin already fails the run on a wrong
negotiated format, so negotiation is covered; byte-level assertion would need a
larger programmatic-graph rewrite.

## Coverage map

The audit was interrupted, so absence of a finding in an area below does not
prove it is clean: it may simply not have been reached. Findings clustered as
follows.

### Audited (findings fixed)

- **Core runtime:** fan-in pad rollback, sink EOS preroll double-decrement,
  fan-out arm error surfacing, COW pooled-buffer reclaim, preroll re-arm on
  down-transition, `accumulate_seek` edges, zero-fan tee/muxer rejection,
  `short_type_name` for generics, shared `block_on`, decodebin splice dedup.
- **Parsers / bitstream:** h264parse, h265parse, flvdemux AMF0 recursion,
  subparse timestamp math.
- **Demux / mux:** fmp4 (sample-count alloc, stss, tfdt, contiguous-mdat),
  matroska/mkv (laced pts, live cluster, per-frame copy), ogg reassembly, mp4muxn
  (moov track numbering, per-frame copy), mkvmuxn/flvmuxn copy, filesink, filesink
  creation, fmp4demux malformed box, tsmux/tsmuxn, registry (alias resolve, dual
  registration listing).
- **RTP / RTCP / network / streaming:** rtpjitter, rtpdepay, rtcp jitter,
  mpegts PSI spanning, rtspserver (src/sink, ports, content-length, control
  buffer), srt (syn cookie, loss-list), turn permission ordering, udpsrc redrain,
  uri/fetch query handling, webrtc (src callback detach, duplex pad routing,
  session/sink/whep TURN dedup), rtmp csid map.
- **Codecs / ML:** av1enc (pts map bound, bitrate-change flush), mfencode
  (async MFT, framerate caps), mfvideosrc receiver-drop, safetensors header
  nesting cap, detect control-packet forwarding.
- **Audio / video transforms:** wavsink float header, audioconvert non-pcm
  reject, wasapisink/src, videorate fraction readback, videotestsrc bar wrap,
  compositor stale frame, video filters output caps, overlay blend share.
- **HLS / DASH:** hls iv char-boundary, mpd segment cap, hlssrc probe reuse,
  sampleaes pattern offsets.
- **Web:** webcodecsdecode timestamp, wgpusink surface reconfigure.
- **Platform codecs:** the 8 open findings above (vt*, ffmpeg*, pipewire,
  gl/cuda, nvdec, cudaglsink, mediacodec_wgpu, camera2src).

### Not yet audited (recommend a pass)

No findings surfaced here and the run did not clearly reach them:

- **g2g-core algebra/negotiation:** `caps.rs`, `solver.rs`, `autoplug.rs`,
  `graph.rs`, `pool.rs`/`staticpool.rs`, `memory.rs`, `aggregator.rs`, `query.rs`,
  `segment.rs`, `clock.rs`, `link.rs`, `fanout.rs`. Only the runner/fanin/seek
  surface was touched; the core type algebra and the autoplug solver were not.
- **g2g-ml inference path:** `burninfer.rs`, `ortinfer.rs`, `wgpuinfer.rs`,
  `wgpupreprocess.rs`, `postprocess.rs`, `cudatowgpu.rs` (only `safetensors.rs`
  and `detect.rs` were touched).
- **g2g-python / g2g-pyapi / g2g-capi hosting:** only the capi run-thread spawn
  and host `attach_metadata` were touched; `element.rs`, `source.rs`, `format.rs`,
  `aggregator.rs`, `host.rs`, and the pyapi surface were not.
- **g2g-enterprise** (`batcher.rs`) and **g2g-bridge** (`lib.rs`): untouched.
- **Plain transform / audio / video elements:** opus (dec/enc/parse), vpx/vp8/
  vp9/av1 parse, mjpeg (dec/enc), mp4 (box/mux/src/audiosink/audiosrc), mkvdemux,
  tsdemux, streamdemux, uridecodebin, typefind, audiomixer/audioresample/
  audiopanorama/volume/audiotestsrc, videoconvert/videoscale/videocrop/videoflip/
  videobox/videobalance/alpha/pixel/yuv, capsfilter, identity, fakesink/filesrc/
  httpsrc, dashsrc/rtmpsrc, vaapidec, nvenc, kmssink, v4l2src, libcamerasrc/
  libcamera_dmabuf, alsasink/pulsesink/pwaudio/aaudio, srtcrypto/srtsink,
  ulpfec/rtx, onvif, websocketsrc, canvassink, bitmapfont, encoder_base, annexb,
  h264util, mux, syncsink, poolstage, plugin_loader.

### Suggested priority for the resumed audit

1. **g2g-core caps/solver/autoplug** (negotiation correctness affects every
   pipeline; highest blast radius).
2. **g2g-ml inference + pre/post-process** (untrusted model I/O, tensor shape
   math, GPU buffer sizing).
3. **Untouched demuxers** parsing untrusted input: `mp4box`/`mp4src`,
   `mkvdemux`, `tsdemux`, `oggdemux` deeper, `dashsrc`/`rtmpsrc` (same
   untrusted-length class as the fmp4/matroska bugs already found).
4. **g2g-python hosting boundary** (FFI + GIL + lifetime correctness).
5. Remaining plain transforms last (lowest risk).

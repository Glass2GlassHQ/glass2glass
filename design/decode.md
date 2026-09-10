# Hardware decode

Decoder and encoder elements: VAAPI, Windows Media Foundation, libavcodec,
NVDEC / NVENC and Vulkan Video, plus the RTSP receive pipeline that feeds them
and the GPU-resident output paths.

Part of the design in [README.md](README.md).

## Decoder element contract

`RtspSrc` then `H264Parse` cover encoded-bitstream processing: mux, re-stream,
record. Decoded-pixel output, which ML inference, display and colour-space
conversion need, uses a decoder `AsyncElement` that accepts
`Caps::CompressedVideo { codec: H264 | H265, .. }` and emits
`Caps::RawVideo { format: Nv12 | I420, .. }` backed by `MemoryDomain::System`,
`MemoryDomain::DmaBuf`, `MemoryDomain::Cuda` or `MemoryDomain::D3D11Texture`
depending on backend.

## VaapiDec (Linux VAAPI, cros-codecs)

`VaapiDec<C>` (`g2g-plugins/src/vaapidec.rs`, feature `vaapi`,
`cfg(target_os = "linux")`) is built on `cros-codecs` with its `vaapi` backend.
The `VaapiCodec` binding picks the stateless decoder and NAL splitter, giving
`VaapiH264Dec` and `VaapiH265Dec`. cros-codecs parses the bitstream and manages
the DPB, and the decode runs on the GPU through libva.

`VaapiDec` is Intel-only. cros-codecs allocates output surfaces through ChromeOS
GBM extensions (`GBM_BO_USE_HW_VIDEO_DECODER`, contiguous NV12) that Mesa
`radeonsi` does not provide, so the element cannot start on AMD desktop GPUs.
The Linux hardware-decode ranking is `VulkanVideoDec` first, vendor-neutral and
picked by the domain-aware search for GPU-domain consumers, with ffmpeg's VAAPI
hwaccel (`Backend::Vaapi`) as the hardware route into system memory.

- Input `Caps::CompressedVideo { codec: C::CODEC, .. }`: `intercept_caps`
  intersects with the element's codec and rejects everything else.
- Output `Caps::RawVideo { format: Nv12, .. }` in `MemoryDomain::System`, a CPU
  copy out of the GBM-allocated surface.
- `GbmDevice::open("/dev/dri/renderD128")`, configurable via
  `VaapiH264Dec::with_render_node`, allocates `GenericDmaVideoFrame` surfaces,
  one per output picture from the decoder's allocator callback.
- The first `decode()` call surfaces `DecodeError::CheckEvents`. The element
  drains events, takes the SPS-derived `StreamInfo` on `FormatChanged`, and
  re-feeds the same NAL.
- `PipelinePacket::Flush` forwards `decoder.flush()` downstream. EOS flushes the
  decoder, drains the DPB, emits `Eos`.
- `libva::Display` is `Rc<Display>` and therefore `!Send`. `unsafe impl Send`
  rests on the runner's ownership model, move not share.

```text
H.264 Annex-B  (MemoryDomain::System)
       │
       ▼
┌───────────────────────────────┐
│  VaapiH264Dec                 │
│   cros-codecs StatelessDecoder│
│   <H264, VaapiBackend<...>>   │
│   DPB + B-frame reorder       │
└───────────┬───────────────────┘
            │  NV12 row-copied out of GBM surface
            ▼
    downstream AsyncElement
```

## MfDecode (Windows Media Foundation Transform)

`MfDecode` (`g2g-plugins/src/mfdecode.rs`, feature `mf-decode`,
`cfg(target_os = "windows")`) wraps `CLSID_MSH264DecoderMFT` via `windows-rs` in
an MTA COM apartment. Input `Caps::CompressedVideo { codec: H264, .. }`, rejected
at `intercept_caps` otherwise. Output `Caps::RawVideo { format: Nv12, .. }` in
`MemoryDomain::System`, a CPU copy out of the MFT output buffer.
`PipelinePacket::Flush` forwards `MFT_MESSAGE_COMMAND_FLUSH` downstream, and EOS
sends `MFT_MESSAGE_COMMAND_DRAIN` to flush the B-frame reorder buffer before
emitting `Eos`. The type is `!Send` by default because of COM, and
`unsafe impl Send` rests on MTA free-threading: the MS H.264 decoder MFT is
callable from any MTA thread without marshaling.

`MfEncode` (feature `mf-encode`) wraps `CLSID_MSH264EncoderMFT` with
`MF_LOW_LATENCY` set, so no B-frames, converting
`Caps::RawVideo { format: Nv12 }` to `Caps::CompressedVideo { codec: H264 }`,
Annex-B framed. `MfAacEncode` / `MfAacDecode` (feature `mf-aac`) cover the AAC
audio path.

## FfmpegH264Dec (ffmpeg / libavcodec)

`FfmpegH264Dec` (`g2g-plugins/src/ffmpegdec.rs`, feature `ffmpeg`,
`cfg(target_os = "linux")`) wraps system libavcodec via `ffmpeg-next`. Input caps
are `Caps::CompressedVideo { codec: H264, .. }`. The backend is selectable:

| `Backend` variant | Codec opened | Output domain | Notes |
| :--- | :--- | :--- | :--- |
| `Software` | `h264` | `System` | Software decode, broadest hardware coverage. |
| `NvdecCuvid` | `h264_cuvid` | `System` | GPU decode, host copy. Pairs with CPU sinks. |
| `NvdecCuda` | `h264` + `AV_HWDEVICE_TYPE_CUDA` | `Cuda` | Zero-copy device-memory output. |
| `Vaapi` | `h264` + `AV_HWDEVICE_TYPE_VAAPI` | `System` | GPU decode, surface downloaded with `av_hwframe_transfer_data`. The Linux AMD / Intel hardware path, working on Mesa `radeonsi` where `VaapiH264Dec` cannot. Pin the render node with `with_vaapi_device` or the `device` property. Launch name `ffmpegvaapidec`. |

### Output formats

Output caps are `Caps::RawVideo` with the layout chosen by `with_output_format`
or the `output-format` property. All are also pad-template alternatives, so a
downstream that pins one auto-plugs a decoder built for it.

- `I420` is the default, libavcodec's native 8-bit 4:2:0. `Nv12` is a U/V
  interleave with no swscale.
- `I422` / `I444` preserve a High 4:2:2 / 4:4:4 source's chroma. A 4:4:4 source
  feeding an `I420` / `Nv12` request is box-averaged down.
- 10-bit and 12-bit sources (High 10 / Main10) keep their depth: `I420p10`
  through `I444p12` are a lossless 2-byte-per-sample plane copy.
- `P010` is the semi-planar 10-bit layout, NV12's shape with the value in each
  16-bit word's top bits, which 10-bit samplers and overlay planes take. It is
  packed from a planar 10-bit software decode or taken verbatim from a `P010LE`
  hardware frame.
- `OutputFormat::Auto` emits the source's own chroma and depth, advertising the
  whole set at negotiation and fixing the concrete format per frame via
  `CapsChanged`.
- `YUVJ*P` is accepted with the same plane layout as its studio-range sibling.

Mismatches needing a real conversion, chroma upsampling or an 8-bit request from
a 10-bit source, are rejected with `CapsMismatch` instead of silently converted.
Put a `videoconvert` downstream, which takes the planar 10-bit and 12-bit family.
Bit-exact against ffmpeg's own raw decode of a High 10 clip for `I420p10` and
`P010`.

### Feed loop and threading

One access unit per `Packet::copy`. PTS is forwarded verbatim because libavcodec
echoes it back on the decoded frame. `send_packet()` then `receive_frame()`
drained until `EAGAIN`. `PipelinePacket::Flush` calls `decoder.flush()`, and EOS
calls `send_eof()` plus a final drain before forwarding `Eos`.
`ffmpeg::decoder::Video` wraps a raw `*mut AVCodecContext` and is `!Send` by
default, so `unsafe impl Send` rests on the same ownership-transfer argument as
`MfDecode` and `VaapiH264Dec`.

## FfmpegH264Enc

`FfmpegH264Enc` (`g2g-plugins/src/ffmpegenc.rs`, feature `ffmpeg`,
`cfg(target_os = "linux")`) is the encode-side mirror:
`Caps::RawVideo { format: I420, .. }` in,
`Caps::CompressedVideo { codec: H264, .. }` Annex-B out, via `ffmpeg-next`. It
gives the Linux production path a hardware H.264 encoder, the codec `WebRtcSink`,
`RtpH264Packetizer` and the RTSP server require. The other Linux encoders are
AV1, VP8/9 and MJPEG, which those H.264-only sinks do not accept.

| `Backend` variant | Encoder opened | Notes |
| :--- | :--- | :--- |
| `Nvenc` (default) | `h264_nvenc` | NVIDIA NVENC, hardware, realtime. The server-side render-and-stream path wants this. Fails loud at configure if absent. |
| `Software` | `libx264` | Portable CPU fallback for CI and no-GPU hosts, present only if libavcodec was built `--enable-libx264`. |

`max_b_frames = 0`, so output is in presentation order with no reorder hold.
Parameter sets are in band: the `GLOBAL_HEADER` flag is not set, so SPS/PPS ride
each IDR, the Annex-B stream a network sink expects. Each backend gets a
low-latency preset and tune, `p4`/`ll`/CBR/`delay=0` for NVENC and
`veryfast`/`zerolatency` for libx264. A downstream PLI
(`Reconfigure::ForceKeyframe`) forces an IDR on the next frame via `pict_type`.
The input frame's nanosecond PTS is mapped through the encoder's frame-index PTS
(`time_base = 1/fps`) and recovered on the output packet, surviving any reorder.

A round-trip test on the RTX 3060 encodes I420 through `Nvenc` and `Software` and
decodes back through `FfmpegVideoDec`, asserting Annex-B framing and I420 at the
original geometry. The `ffmpeg` feature is CI-excluded for libav version
sensitivity, so this runs on libav hosts.

## NvEnc

`NvEnc` (`g2g-plugins/src/nvenc.rs`, feature `nvenc` which implies `cuda`,
`cfg(target_os = "linux")`) is the zero-copy device-resident H.264 encoder. The
ffmpeg `Nvenc` backend takes system-memory I420 and copies it into libavcodec.
`NvEnc` ingests an NVDEC/CUDA NV12 surface (`MemoryDomain::Cuda`) in place and
drives the NVIDIA Video Codec SDK (`nvEncodeAPI`) directly, so pixels never leave
the GPU. It closes the native `FfmpegH264Dec(NvdecCuda) -> NvEnc` loop with no
PCIe download, the encode-side mirror of the `CudaToWgpu` import bridge
([ml.md](ml.md)), and is the egress half of the server-side
render-and-stream path fed by the wgpu-to-CUDA hand-off.

- Caps: `Caps::RawVideo { format: Nv12 | Rgba8 | Bgra8, .. }` in,
  `Caps::CompressedVideo { codec: H264, .. }` Annex-B out, a native
  `DerivedOutput` at the same dims and framerate. Caps do not encode the memory
  domain, so negotiation is identical to a system encoder. At runtime the frame
  must be `MemoryDomain::Cuda`, else `UnsupportedDomain`, the symmetric contract
  `FfmpegH264Enc` upholds for `System`.
- NV12 input, the NVDEC hwframe domain, must be a contiguous surface: chroma at
  `luma_ptr + luma_pitch * height`, one base pointer plus pitch. RGBA input, the
  GPU-render domain reached via `WgpuToCuda`, is a single packed plane at
  `luma_ptr` with `luma_pitch = width * 4`, registered as NVENC `ABGR` for wgpu
  `Rgba8` byte order or `ARGB`, NVENC converting to H.264 colour internally.
- Bindings are hand-rolled FFI. As in the `cuda` module
  (`g2g-plugins/src/cuda.rs`), `cudarc` is not used: the element links
  `libnvidia-encode` and `libcuda` directly. The SDK's version-tagged structs are
  transcribed `#[repr(C)]` with compile-time size assertions
  (`const _: () = assert!(size_of::<T>() == N)`) checked against the installed
  `nvEncodeAPI.h` (SDK 13.0, field offsets verified with `offsetof`), so a
  mismatched SDK fails the build rather than corrupting the wire layout. The one
  field-heavy codec-config union stays opaque as a correctly-sized `[u32; N]`:
  the driver fills it via `nvEncGetEncodePresetConfigEx` and only rate control
  and GOP are overwritten.
- The session opens lazily on the first frame, on that frame's `CUcontext`, the
  NVDEC source's context. Per frame: `nvEncRegisterResource`
  (`CUDADEVICEPTR`, NV12), `nvEncMapInputResource`, `nvEncEncodePicture`,
  `nvEncLockBitstream` to copy out Annex-B, then unlock, unmap, unregister.
- Low latency: preset P4 with the LOW_LATENCY tuning info, CBR, no B-frames
  (`frameIntervalP = 1`), and an infinite GOP (`NVENC_INFINITE_GOPLENGTH`) so
  IDRs are emitted only on the first frame and on a downstream PLI
  (`Reconfigure::ForceKeyframe`). Each forced IDR sets `OUTPUT_SPSPPS` so in-band
  parameter sets ride it. The NV12 nanosecond PTS round-trips through NVENC's
  `inputTimeStamp`.
- HEVC sits alongside H.264: `with_codec(VideoCodec::H265)` or the `codec`
  property switches the encode GUID to `NV_ENC_CODEC_HEVC_GUID` and the output
  caps to `CompressedVideo{H265}`, the path otherwise identical. P010 input maps
  to the 10-bit buffer format and the HEVC Main10 profile. P010 with
  `codec=h264` is rejected because NVENC has no 10-bit H.264.
- `gop-size` (-1 means infinite, the low-latency default) and
  `repeat-sequence-header` write `gopLength` / `idrPeriod` / `repeatSPSPPS`,
  re-applied live through `nvEncReconfigureEncoder`. The output-bitstream-buffer
  pool and runtime bitrate retarget are in place.
- `input_domains = {Cuda}`, so a CPU-side NV12 source gets a `CudaUpload` spliced
  in by the converter auto-plug ([caps.md](caps.md), the allocation
  cascade). The encoder itself
  stays Cuda-only.
- An on-hardware round-trip on the RTX 3060 synthesizes a CUDA-resident NV12
  surface, encodes through `NvEnc`, and decodes the Annex-B back through
  `FfmpegVideoDec` to the original geometry. It skips with no NVIDIA GPU, and the
  `nvenc` feature is CI-excluded.
- The session is a raw NVENC handle plus CUDA context driven through `&mut self`
  only, so `unsafe impl Send` rests on the same ownership-transfer argument as
  `FfmpegH264Enc`.

## NvDec

`NvDec` (`g2g-plugins/src/nvdec.rs`, feature `nvdec` which implies `cuda`,
`cfg(target_os = "linux")`) is the decode half of the pair, the mirror of
`NvEnc`. It promotes NVIDIA hardware decode from the `FfmpegH264Dec`
`Backend::NvdecCuda` flag, which reaches NVDEC through libavcodec's cuvid
hwaccel, to a first-class element driving the NVCUVID parser and decoder API
directly. With `NvDec -> ... -> NvEnc` both native, the whole H.264 transcode
loop stays on the GPU and out of libavcodec.

- Caps: `Caps::CompressedVideo { codec: H264, .. }` Annex-B in,
  `Caps::RawVideo { format: Nv12, .. }` out, a native `DerivedOutput`. The
  runtime `CapsChanged` carries the cropped display geometry the bitstream
  declares.
- Multi-domain output: `output_domains = {Cuda, System}`, reconciled in
  `configure_allocation` against the negotiated proposal
  (`resolve_for_producer`, the allocation cascade in
  [caps.md](caps.md)). A CUDA-capable consumer
  keeps each surface device-resident, the default `MemoryDomain::Cuda`. A
  System-only consumer makes the decoder download through `cuda::download_nv12`
  before emitting. Downstream demand alone decides.
- Callback model: a parser (`cuvidCreateVideoParser`) is fed the elementary
  stream and synchronously invokes three callbacks from inside
  `cuvidParseVideoData`. A sequence callback creates the `CUvideodecoder` once
  the SPS geometry is known, a decode callback runs `cuvidDecodePicture`, and a
  display callback fires when a frame is ready in display order. The display
  callback cannot `await`, so it maps the surface with `cuvidMapVideoFrame64` and
  pushes a ready frame onto a queue that `process` drains after the parse
  returns. The callbacks reach element state through a `*mut DecoderState` passed
  as the parser user-data, targeting a heap `Box` so it survives the runner
  moving the element between worker threads.
- Bindings are hand-rolled FFI: `libnvcuvid` and `libcuda` directly, no `cudarc`.
  NVCUVID exports real symbols with no `CreateInstance` dispatch table, unlike
  NVENC, so the calls are plain `extern "C"`. Structs are transcribed
  `#[repr(C)]` with compile-time size assertions against the installed
  `cuviddec.h` / `nvcuvid.h`. The per-picture `CUVIDPICPARAMS` is opaque: the
  parser fills it and the pointer goes straight to `cuvidDecodePicture`.
- Frame lifetime: each output frame carries a `CudaKeepAlive` that calls
  `cuvidUnmapVideoFrame64` on drop plus an `Arc` to the decoder, so the decoder
  and its CUDA context outlive any frame in flight. Decoder, context lock and
  context are destroyed in that order once the last frame is released. The
  element owns its own CUDA context, created at configure.
- Device selection: every element creating its own CUDA context takes the ordinal
  from a `cuda-device-id` property, that is `NvDec`, `CudaUpload`,
  `LocalCudaSrc`, `FfmpegVideoDec` on `Backend::NvdecCuda`, and read-only on the
  `WgpuToCuda` bridge whose device is fixed by the wgpu device it was built over.
  The spec is declared once in `g2g-plugins/src/cudadeviceid.rs` so name, range
  and default cannot drift. It defaults to 0, is read when the context opens, and
  a later set is refused with `PropError::ReadOnly` because frames already in
  flight carry the old ordinal. Each emitted `OwnedCudaBuffer` reports it as
  `device_ordinal`, so a consumer can tell which GPU a surface lives on.
- HEVC and AV1 sit alongside H.264: input caps accept
  `CompressedVideo{H264|H265|Av1}`, and the codec is inferred and mapped to the
  `cudaVideoCodec` the parser and decoder are created for. A 10-bit stream
  decodes to a `P016` surface announced as `RawVideoFormat::P010`.
- A mid-stream resolution change reconfigures the live decoder in place with
  `cuvidReconfigureDecoder` when the new size fits, else rebuilds it. The CUDA
  context rides a separate `Arc` so in-flight frames survive the rebuild.
- Display delay defaults to a low-latency 1, settable via `max-display-delay`
  in the range 0 to 16.
- An on-hardware test on the RTX 3060 runs the full native loop, a synthesized
  CUDA NV12 surface encoded by `NvEnc` and decoded by `NvDec` back to CUDA NV12,
  asserting geometry and, via a small device-to-host copy, real luma content. It
  skips with no NVIDIA GPU, and the `nvdec` feature is CI-excluded.

## End-to-end RTSP pipeline

```
RtspSrc ──► H264Parse ──► [decoder] ──► [ML / display / encode]
(System / H264)            (System / DmaBuf / Cuda / D3D11Texture; NV12)
```

| Platform | Decoder element | Feature | Output |
| :--- | :--- | :--- | :--- |
| Linux software | `FfmpegH264Dec` (`Software`) | `ffmpeg` | `System` / I420 |
| Linux + NVIDIA | `FfmpegH264Dec` (`NvdecCuvid` / `NvdecCuda`) | `ffmpeg` + `cuda` | `System` / `Cuda` / NV12 |
| Linux + VAAPI | `VaapiH264Dec` / `VaapiH265Dec` | `vaapi` | `System` / NV12 |
| Windows | `MfDecode` | `mf-decode` | `System` / NV12 |

### RtspSrc

`RtspSrc` connects via `retina` using RTSP/RTP over TCP and negotiates H.264 with
`FrameFormat::SIMPLE` (Annex-B), or accepts AVCC framing detected per buffer. The
first SPS gives geometry. Framerate comes from the VUI `timing_info`
(`time_scale / (2 * num_units_in_tick)`) when present, else `Rate::Any`.
`RtspSrc::with_credentials` supplies the DESCRIBE/SETUP account, threaded into
retina's `SessionOptions`.

### RtspSrcN

`RtspSrcN` (`rtspsrcn.rs`, same feature) plays the stream's audio track alongside
its video: one retina session, output 0 the video and output 1 the first audio
stream the SDP offers, AAC via `mpeg4-generic` or G.711 via `pcma` / `pcmu`, as a
`MultiOutputSource` driven by `run_fanout_session`. Both streams take the same
lower transport and one `FrameFormat::SIMPLE` SETUP, so AAC arrives ADTS-framed
and needs no out-of-band config.

The pads run on different RTP clocks, so each frame's PTS is
`Timestamp::elapsed`, normal play time against the server's `RTP-Info` origin
which spans both streams, converted at that stream's clock rate and rebased on
the session's first frame: one timeline starting at zero with the stream's A/V
offset intact. `InitialTimestampPolicy::Permissive` keeps a server that omits
`rtptime` playable, each stream starting at its own first packet.

The audio pad negotiates at the decoder-facing caps, not the SDP's. A compressed
`sample_rate` is matched for equality (the negotiation lifecycle in
[README.md](README.md)), so AAC advertises the
`0/0` sentinel the demuxers use and G.711 the 8 kHz rate its decoder declares.

Pad count is fixed before the run, so the `playbin uri=rtsp://...` hook DESCRIBEs
first (`blocking_probe_tracks`) and declines to the single-pad `RtspSrc` chain
for a video-only stream. A launch line declares the count by linking pads:
`rtspsrcn name=s location=... s. ! ... s. ! ...`.

### OnvifSrc

`OnvifSrc` (`onvif` feature) is the ONVIF control plane in front of `RtspSrc`. An
ONVIF camera does not stream over ONVIF: its SOAP services tell you the RTSP URL.
`discover` sends one WS-Discovery `Probe` to the `239.255.255.250:3702` multicast
group and collects each camera's device-service URL from the `ProbeMatch`
`XAddrs`. `resolve_stream_uri` then runs `GetCapabilities`, `GetProfiles`,
`GetStreamUri`, authenticated with a WS-Security `UsernameToken` digest
(`Base64(SHA1(nonce ++ created ++ password))`).

The element resolves the RTSP URI lazily during negotiation (`intercept_caps`),
builds an inner `RtspSrc` once forwarding the same credentials, since cameras
gate the media stream behind the device account, and delegates the rest of the
`SourceLoop` to it. The SOAP layer is hand-rolled, fixed request templates plus
`roxmltree` response reads, to avoid the git-only `onvif` / `schema` crate tree.
The footprint is reqwest, roxmltree, sha1, base64 and getrandom. Its scope is
discovery and stream-URI resolution.

### ONVIF analytics metadata

The camera's scene description is a separate RTSP track, `m=application` with an
`a=rtpmap` encoding name of `vnd.onvif.metadata` or `vnd.onvif.metadata.gzip`
(Streaming Specification 5.2.2.4). `RtspSrcN` subscribes it on a pad of its own
when `onvif-metadata` is set, after video and audio. The property is off by
default, so an existing consumer and the `playbin` hook negotiate what they did
before, and when a launch line links only two pads the metadata pad takes the
audio one's slot rather than growing the element past what was linked.

retina concatenates the RTP payloads to the marker bit, which closes one XML
document. The element inflates a gzip member itself with `miniz_oxide` and
bounded output, and emits the whole `tt:MetadataStream` document as one
`Caps::OnvifMetadata` frame. The Streaming Specification writes the compressed
encoding name `vnd.onvif.metadata+gzip` while retina matches
`vnd.onvif.metadata.gzip`, and a set-up stream retina cannot depacketize fails
the whole session, so a track advertised with the specification's spelling, or
with an EXI coding which g2g has no decoder for, is logged and left alone.

### onvifmetadataparse

`onvifmetadataparse` (`onvifmetadata.rs`, `onvif` feature) splits a document into
one frame per `tt:Frame`, sharing the input buffer rather than copying it, and
attaches that frame's objects as an `AnalyticsMeta`: one `ObjectDetection` per
`tt:Object` with a usable `tt:BoundingBox`, a `Tracking` node holding its
`ObjectId` related to it by `Tracks`, and a `Contains` relation for a `Parent`
attribute naming an object in the same frame. Class names come from either
encoding of `tt:Class`, the current `tt:Type Likelihood="..."` list and the
legacy `tt:ClassCandidate` pairs, the likeliest one naming the detection.

Coordinates pass through the `tt:Transformation` stack (`t' = v·s + t`,
`s' = u·s`) into the ONVIF normalized frame system, then are remapped from
`[-1, 1]` about the picture centre with `y` up to the `BBox` convention of
`[0, 1]` from the top-left with `y` down. Elements are matched by namespace and
local name, never by prefix. A payload holding several concatenated
`<?xml ...?>` roots is cut at each declaration and parsed separately. Frame and
object counts and element depth are bounded, and a malformed document yields no
output.

### onvifmetadatacombiner

Sync is by wall clock, not RTP time: the Streaming Specification gives the
metadata track's RTP timestamps no meaning and makes a `tt:Frame`'s `UtcTime` the
name of the picture it describes. `WallClockMeta` (`g2g-core`, nanoseconds since
the Unix epoch) carries that instant on both sides. `RtspSrcN` keeps the latest
RTCP sender report per stream and computes each frame's wall clock as the
report's NTP instant plus the signed 32-bit RTP difference converted at the
stream's clock rate, so a timestamp wrap reads as a frame just before the report
rather than most of a wrap after it. Before a stream's first report its frames
carry no wall clock.

`onvifmetadatacombiner` merges the two pads by that clock, falling back to PTS on
the play timeline when either side lacks it. It holds each video frame for
`latency` (default 200 ms) of stream time, attaches the metadata whose instant
falls in the frame's window, its own duration else the next frame's start,
appends to whatever `AnalyticsMeta` a detector upstream already wrote rather than
replacing it, and drops metadata more than `max-lateness` (default 200 ms) behind
the video. An EOS on either pad flushes what is held, so a silent metadata pad
never stalls the video.

## Zero-copy NVDEC to CUDA to GPU display

`Backend::NvdecCuvid` decodes on the GPU but copies NV12 back to system memory,
and the glass-to-glass floor is then dominated by the PCIe round-trip plus the
sink's CPU NV12-to-XRGB convert. The CUDA-resident path keeps decoded NV12 in
device memory end-to-end so a GPU consumer such as a display takes the handoff
without a host round-trip.

`MemoryDomain::Cuda(OwnedCudaBuffer)` lives in `g2g-core` and is
platform-agnostic. `OwnedCudaBuffer` carries the two NV12 plane device pointers,
luma Y and interleaved chroma UV, row pitches, dims, the `CUcontext`, and a boxed
`CudaKeepAlive` owner. Core never links CUDA: the producing element supplies the
owner as a trait object, and dropping the buffer releases the backing allocation.
`AllocationParams::cuda(...)` makes `MemoryDomainKind::Cuda` a cross-element pool
domain in the allocation negotiation ([caps.md](caps.md)).

`Backend::NvdecCuda` opens the generic `h264` codec with an
`AV_HWDEVICE_TYPE_CUDA` device and a `get_format` hook selecting
`AV_PIX_FMT_CUDA`. The resulting `AVFrame` is the keep-alive that owns the device
pointers wrapped into `OwnedCudaBuffer`.

### CUDA and GL interop

CUDA can only export VMM-allocated memory (`cuMemCreate` / `cuMemMap`) to a
dma-buf fd, and NVDEC decoder frames come from libavcodec's CUDA hwframe pool,
not VMM. The NVIDIA proprietary driver also does not import foreign dma-bufs
reliably through `nvidia-drm`. Presentation therefore uses CUDA-GL interop, the
path NVIDIA's `FramePresenterGL` sample takes:

1. Create an EGL context on the display surface.
2. Register a GL texture with `cuGraphicsGLRegisterImage` once.
3. Per frame: `cuGraphicsMapResources`, `cudaMemcpy2D` (device→device,
   honouring source pitch) the NV12 planes into the GL resource,
   `cuGraphicsUnmapResources`.
4. Sample Y + interleaved UV in a fragment shader (BT.601/709 limited range),
   present via `eglSwapBuffers`.

This is not strictly zero-copy, one device-to-device copy goes into the GL
texture, but it removes the PCIe round-trip and the CPU colour convert.

Bindings are hand-rolled FFI. `cudarc` has no CUDA-GL interop wrappers such as
`cuGraphicsGLRegisterImage`, and its safe API assumes it owns the `CudaContext`,
whereas the `CUcontext` is created and owned by ffmpeg's hwdevice and carried on
`OwnedCudaBuffer`. The needed surface is small: `cuCtxPushCurrent_v2` /
`_PopCurrent_v2`, `cuMemcpy2D_v2`, and the GL-interop quartet
`cuGraphicsGLRegisterImage` / `cuGraphicsMapResources` /
`cuGraphicsSubResourceGetMappedArray` / `cuGraphicsUnmapResources`. The plugin
links `libcuda` directly.

### CudaDownload, CudaGlSink, CudaKmsSink

- `CudaDownload` (`cuda` feature) is an `Identity(NV12)` transform copying a
  `MemoryDomain::Cuda` frame to `MemoryDomain::System` via device-to-host
  `cuMemcpy2D`. It negates the latency win but lets a `NvdecCuda` stream reach
  the existing CPU sinks for correctness and bring-up.
- `CudaGlSink` (`cuda-gl` feature, Linux + NVIDIA) holds an EGL context on a
  Wayland surface (`wl_egl_window` from SCTK), a `glow` GL ES 3 program with the
  two NV12 textures, and the per-frame map, copy and unmap render loop via the
  CUDA-GL interop entry points. On an RTX 3060 its present latency is about
  10.7x lower than `NvdecCuvid -> WaylandSink` at 1080p.
- `CudaKmsSink` (`cuda-kms` feature, Linux + NVIDIA) is the tty and
  no-compositor counterpart: the same CUDA-GL interop and NV12-to-RGB shader,
  shared via the `glnv12` module, but EGL renders into a GBM surface scanned out
  via DRM page-flips instead of a Wayland surface. It needs DRM master, a bare VT
  or a DRM lease. The shared render half is the validated `CudaGlSink` path.

## Vulkan Video (vendor-neutral GPU-resident decode)

The NVDEC to CUDA to wgpu path is vendor-locked: CUDA has no AMD or Intel analog,
so a wgpu-based consumer such as a game engine or visualization viewer that wants
hardware decode straight into its own render device gets it only on NVIDIA.
`VulkanVideoDec` decodes with `VK_KHR_video_queue` and `VK_KHR_video_decode_*` on
the same Vulkan device wgpu already runs, so the decoded `VkImage` is imported as
a `wgpu::Texture` with no download and no second interop bridge. One element
covers AMD (RADV), NVIDIA and Intel (ANV), each validated as hardware is
available.

### Capability probe

`vulkanvideo::probe_decode_caps` reaches the adapter's raw `ash` handles via
`as_hal::<Vulkan>()`, finds a decode-capable queue family, and queries
`vkGetPhysicalDeviceVideoCapabilitiesKHR` for H.264, H.265 and AV1. It returns
the coded-extent range, DPB slot and active-reference budget, and the
`DPB_AND_OUTPUT_COINCIDE` flag that `intercept_caps` and DPB sizing negotiate
against. On the RTX 3060: H.264 to 4096 square, H.265 and AV1 to 8192 square,
output coincides with the DPB.

The query returns a generic `ERROR_INITIALIZATION_FAILED` unless the
codec-specific output caps struct (`VkVideoDecodeH264/H265/AV1CapabilitiesKHR`)
is chained alongside `VkVideoDecodeCapabilitiesKHR`, with a
`VkVideoDecodeUsageInfoKHR` on the profile.

### Device and session setup

The element is mostly reuse. The `VkImage` to `wgpu::Texture` import
(`cudawgpu.rs` / `dmabufwgpu.rs` `texture_from_raw` plus
`TextureMemory::External`, [ml.md](ml.md)), custom Vulkan device
creation with extra extensions from the `cuda-wgpu` device path, the multiplanar
NV12-to-RGBA `VkSamplerYcbcrConversion` compute pass shared with the Android
`mediacodec-wgpu` decoder, the Annex-B plus SPS/PPS front-end (`h264parse` and
h265parse), and the allocation-domain auto-plug (the allocation cascade in
[caps.md](caps.md)) all
already exist.

New is the decode session itself: a `VkDevice` with a
`VK_QUEUE_VIDEO_DECODE_BIT_KHR` queue adopted into wgpu via `create_from_hal`,
the integration point that matters since wgpu will not request a decode queue on
its own, a `vkGetPhysicalDeviceVideoCapabilitiesKHR` probe feeding
`intercept_caps`, a `VkVideoSessionKHR` / `VkVideoSessionParametersKHR` whose
`Std*` parameter structs are populated from the parsed SPS/PPS/VPS, DPB
reference-slot management, and the `vkCmdDecodeVideoKHR` recording with output
pipelined through the YCbCr pass on an in-flight ring. The `Std*` mapping is the
correctness-critical part, one mapping module per codec, re-emitted on mid-stream
change via `CapsChanged`.

A session's `maxCodedExtent` is the device's maximum, not the stream's geometry.
It is only an upper bound and each picture resource carries its real extent.
Sizing the session to the picture made the NVIDIA driver refuse whole small
geometries with `ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR`.

Every image the session writes, DPB slot and decode output, is created at the
picture extent rounded up to the device's `pictureAccessGranularity`, the unit in
which a decode accesses a picture resource. The readback copy takes the picture's
own extent back out of that image, and the YCbCr compute pass is given the padded
extent so its normalized coordinates still land on the picture's texels. The
padding never reaches an output frame.

Session and DPB rebuild mid-stream on any in-band parameter-set change, keyed by
a byte fingerprint of the AU's parameter sets. That covers geometry and
same-geometry content such as a profile or entropy-mode switch, while
byte-identical keyframe re-sends keep the session. The outgoing decoder's
pipelined tail is flushed first so no frame is lost.

Output caps are `Caps::RawVideo { format: Rgba8, .. }` in
`MemoryDomain::WgpuTexture`, optionally `VulkanTexture` or multiplanar NV12.
Negotiation and the frame keep-alive follow the `NvDec` multi-domain pattern.

### H.264

Session plus `Std*` SPS/PPS mapping, IDR then full-DPB P-frame decode bit-exact
against the ffmpeg software decoder, the zero-copy `VkSamplerYcbcrConversion`
NV12-to-RGBA import into a `wgpu::Texture`, the `VulkanVideoDec` streaming
element and its `WgpuSink` present, and `produces(WgpuTexture)` auto-plug.

The `WgpuTexture` output offers `Rgba8` first and `Nv12` second. When the solved
caps pin `NV12`, the converter copies the decoded slot's two planes into a fresh
image (`YcbcrConverter::copy_nv12`, a plane-wise `vkCmdCopyImage` on the compute
queue in place of the ycbcr dispatch) and imports it as a `TextureFormat::NV12`
wgpu texture behind `WgpuNv12Texture`, so a consumer samples the planes through
`Plane0` / `Plane1` views with no colour conversion. The decode device requests
`TEXTURE_FORMAT_NV12` when the adapter offers it.

The same copy backs the `VulkanTexture` output domain: the frame carries the raw
`VkImage` handle, `VkFormat` and geometry in `OwnedVulkanTexture`, and its
keep-alive downcasts to `VulkanImageOwner` for the `ash` device and the wgpu
texture that owns the image, so a Vulkan-native consumer such as an encoder or
presenter reads the picture where the decoder left it, idle in
`SHADER_READ_ONLY_OPTIMAL`.

### Texture layouts

The caps say a `WgpuTexture` frame is NV12 but not which of the two NV12 texture
layouts it is, so a consumer reads that off the texture per frame.
`gpu::texture_layout` maps `R8Uint` to the packed plane the CUDA and dma-buf
bridges allocate, `TextureFormat::NV12` to the decoder's two planes, and an
uncompressed colour format to a finished picture.

wgpu refuses to bind a multi-planar texture whole, so the planes come from
`gpu::nv12_plane_views`: `Plane0` as full-size `R8Unorm` and `Plane1` as
half-size `Rg8Unorm`, each a `texture_2d<f32>`. `WgpuSink` holds a blit pipeline
per layout and picks one per frame, so one sink negotiated for NV12 renders the
same picture from a decoder's texture and from a system-memory upload of the same
clip. Its two NV12 fragment stages share one `ycbcr_to_rgb` step and both fetch
the nearest chroma texel, since a filtered chroma sample sits a quarter texel off
the chroma grid and shifts colour across edges. `g2g-ml`'s `WgpuPreprocess` does
the same on the compute side, with a second import pipeline binding the two plane
views. The consumers that only read a finished picture, `WgpuCompositor`,
`VulkanHdrSink` and the Bevy plugin, reject either NV12 layout rather than
sampling a luma plane as colour.

### Streaming and random-access models

Two consumption models sit on the same `H264DpbDecoder`. The streaming
`VulkanVideoDec` is a push `AsyncElement` for pipelines. `VulkanVideoPlayer` is a
random-access pull frame server (`frame_at(pts)` / `frame_at_index`) that indexes
GOPs and POC (`index_pictures`), `reset`s and decodes forward from the enclosing
random-access point on a seek (`decode_range_to_texture`), and caches decoded
textures keyed by decoding index. It is the timeline scrubber a wgpu
visualization viewer needs, whose native decode is typically CPU software plus a
GPU upload copy, and the `vulkan_video_scrubber` example drives it interactively.

The player drives H.264 and H.265, the codec sniffed by `sniff_annexb_codec`
behind a `PlayerDecoder` enum. Its seek point is the nearest IRAP, an IDR for
H.264 and an IDR / CRA / BLA for H.265 (`PictureMeta::is_random_access`), so
scrubbing into a late open-GOP GOP tunes in at that GOP's CRA and discards its
RASL instead of decoding from the leading IDR. A leading picture, a CRA's RASL /
RADL whose POC precedes its CRA, instead seeks from the random-access point
before that CRA and decodes continuously through it so its references exist.

A forward seek within reach keeps decoding rather than re-decoding from the
keyframe, so linear playback is O(n) coded pictures, not O(n^2). Display order is
by GOP then POC since POC resets at each IDR. The decoded-frame cache is LRU and
bounded by both a frame count and a byte budget, the bound that matters at 4K and
8K where a count alone pins gigabytes, and `set_cache_traversed` optionally
caches every traversed picture on a decode range so a backward scrub within a GOP
is free.

Decoded frames are consumable by an application-owned wgpu render pipeline, not
just `WgpuSink`: the imported RGBA texture carries `TEXTURE_BINDING`, so a
foreign pipeline on the shared decode device samples it zero-copy
(`m500_vulkan_video_embed` plus the `vulkan_video_engine_embed` example), the
integration primitive a Bevy viewer-renderer consumer builds on.

### H.265

`parse_h265_vps/sps/pps` plus `extract_h265_parameter_sets` read the RBSP, which
starts at `nal[2..]` because the NAL header is two bytes, including the full
`profile_tier_level` and the short-term reference-picture sets, parsed to
canonical explicit form. An inter-RPS-predicted set is derived per H.265 7.4.8.
`to_std_h265_params` maps them onto the `StdVideoH265*` layout, returning a
`StdH265Params` bundle that owns the pointee blocks the SPS/VPS reference by
pointer: profile-tier-level, DPB manager, short-term RPS list.
`create_h265_session`, via `open_h265_decode_device`, builds the
`VkVideoSession` plus parameters, driver-validated on the RTX 3060.

`H265DpbDecoder` decodes pixels: per-picture slice-segment-header parse
(`parse_h265_slice_header`), picture-order-count per 8.3.1,
reference-picture-set DPB management with every IRAP a clean reset, and the
reference-slot lists (`RefPicSetStCurrBefore/After`, POC-keyed reference info)
handed to `vkCmdDecodeVideoKHR`, reusing the H.264 DPB machinery. The whole
fixture of IDR and CRA GOPs decodes bit-exact against the ffmpeg software decoder
on the 3060, also straight to GPU-resident RGBA `wgpu::Texture`s.

NVIDIA's Vulkan HEVC slice-header parser needs a 3-byte start code (`00 00 01`).
A 4-byte one breaks every non-IDR slice, while the IDR tolerates it via the
picture info, so the H.265 path frames slices with 3 bytes and H.264 keeps 4.

### Long-term reference pictures

The SPS long-term table rides `pLongTermRefPicsSps`. The slice header's long-term
entries, SPS-indexed and slice-coded with the accumulated `DeltaPocMsbCycleLt`
per 7.4.7.1, resolve against the DPB by full POC when the MSB cycle is present
and by POC lsb alone otherwise. The RPS prune keeps long-term-listed pictures,
`RefPicSetLtCurr` carries the used-by-current slots, and each reference's `Std*`
info flags its short-term or long-term marking. That marking changes the driver's
MV scaling, so a wrong one corrupts prediction silently instead of erroring.

Two slice-RPS rules follow. An inline `st_ref_pic_set` coded in a slice header
does carry `delta_idx_minus1` when inter-RPS-predicted, and missing it desyncs
every later field. `NumDeltaPocsOfRefRpsIdx` must be the referenced set's delta
count, not 0, for the driver's own slice-header re-parse. All 500 frames of the
JCT-VC `LTRPSPS_A_Qualcomm_1` conformance vector decode bit-exact against ffmpeg
on the 3060, and the GPU-texture path shares the same DPB machinery.

### Reference marking

Which decoded pictures stay available as references is the stream's decision, not
the decoder's. `dec_ref_pic_marking()` in each slice header either leaves it to
the default sliding window, evicting the smallest `FrameNumWrap`, or names the
pictures to retire, which is what x264 does for its B-pyramid. Reading the
marking means walking past the reference-list modification and the prediction
weight table first, so `poc::parse_h264_slice_marking` continues the shared slice
parse through them and returns the operations as `H264RefPicMarking`,
fixed-capacity so the header stays `Copy`. The DPB applies the short-term
operations and refuses a long-term operation rather than keep feeding the driver
a reference the stream has retired. Running the sliding window regardless
diverges from the reference set the driver builds its L0 / L1 lists against.

### B-frames and display order

The driver builds the L0 / L1 reference lists from the DPB and the per-picture
POC the decoder supplies, H.264 from every DPB slot's FrameNum and POC and H.265
from the `RefPicSetStCurrBefore/After` split by POC sign, so a
bidirectionally-predicted frame reconstructs bit-exact. What B-frames change is
order: a frame is coded after pictures that precede it on screen.

The whole-stream `decode_all` / `decode_all_to_textures` index the stream's POCs
(`index_pictures`) and reorder the coding-order output into display order via
`reorder_to_display_order`, keyed by coded-video-sequence and POC so POC resets
at each keyframe group correctly. For an I/P stream this is the identity.

The low-level streaming `decode_push` stays in coding order, since a low-latency
consumer such as the `streamdec` adapter reorders by PTS itself. The
`VulkanVideoDec` element does reorder its system NV12 path: `decode_push_meta`
returns one `PictureMeta` per submitted picture carrying the POC the decode
already computed, no second pass, and the element feeds retired frames through a
small `ReorderBuffer` keyed by the same coded-video-sequence and POC. The
GPU-texture path streams the same way: `decode_push_to_textures` decodes each
AU's pictures with the DPB and POC state intact across calls, since the
whole-stream `decode_all_to_textures` indexing pass resets it and cannot stream,
and the element reorders them through a texture `ReorderBuffer`. AV1 needs
neither buffer: its display order is the bitstream's op order, so the element
op-walks each temporal unit (`decode_display` / `decode_display_to_textures`, the
DPB persisting across calls), which also makes `show_existing_frame` re-displays
and per-frame film-grain synthesis work when streamed.

NVIDIA's driver retains the `pStdSequenceHeader` / `pColorConfig` pointers handed
to `vkCreateVideoSessionParametersKHR` and dereferences them per decode, so
`Av1DecodeSession` owns a stable boxed copy for its lifetime. Dropping the Std
block after creation gives small, nondeterministic pixel corruption.

The `ReorderBuffer` releases the whole previous coded video sequence at each
keyframe, where POC resets, and bumps the lowest-POC held frame once a sequence
exceeds the stream's own declared reorder depth: H.264 VUI
`bitstream_restriction` `max_num_reorder_frames` or H.265
`sps_max_num_reorder_pics`, with the DPB slot count as the fallback bound when
the stream declares none. An I/P stream emits without hold and a long GOP does
not buffer unbounded. `Eos` and a reconfig boundary drain it in display order,
and a `Flush` (seek) discards it, for H.264 and H.265. AV1 stays in coding order
there, its display order coming from `show_existing_frame` / `order_hint`,
handled whole-stream by `decode_all`. Verified bit-exact against the software
decoder's display-order output for H.264 and closed-GOP H.265 B-frame clips on
the 3060, with the element's AU-by-AU streaming output matching that oracle byte
for byte.

The DPB is flushed only at an IRAP with `NoRaslOutputFlag == 1`, every IDR and
BLA and a CRA only as the first picture, so full-stream H.265 open-GOP decodes
bit-exact: a mid-stream CRA keeps the references its RASL followers use. After a
`reset` (a seek) the CRA is the first picture, so `NoRaslOutputFlag == 1` and its
RASL leading pictures, which reference now-absent pre-CRA frames, are discarded
rather than decoded against a flushed DPB. `h265_is_rasl` plus a `skip_rasl` flag
set from each IRAP's `NoRaslOutputFlag` is checked before POC derivation so a
dropped RASL leaves no trace. The flag is 0 in continuous decoding, so full-stream
open-GOP is unchanged.

### Colour space

Decoded YUV is converted to RGB with the stream's actual colour space, not a
fixed matrix. A `VideoColorSpace`, colour matrix plus quantization range, is
resolved at decoder build time from the H.264 / H.265 VUI colour description
(`parse_vui_color`, one helper since the VUI colour prefix is identical in both
codecs) or the AV1 `color_config`, keyed by the CICP `matrix_coefficients`
codepoint. Unspecified falls back by resolution, the ffmpeg heuristic. Both the
CPU `nv12_to_rgba`, a general Kr/Kb luma-weight formula over studio and full
range, and the GPU `YcbcrConverter`, whose `VkSamplerYcbcrConversion` is built
with the matching `YCBCR_601/709/2020` model and `ITU_NARROW/FULL` range, apply
it, so BT.709 HD and BT.2020 content get the right matrix instead of BT.601.

### 10-bit decode

HDR is 10-bit, so the decoder is not fixed to 8-bit NV12. The session derives its
bit depth from the SPS and, for a 10-bit HEVC stream, selects the Main 10 profile
and the `G10X6` two-plane 4:2:0 output format, 16-bit samples with the value in
the top 10 bits. The shared `DpbCore` scales its readback sizing to 2 bytes per
sample, and `Nv12Frame::bit_depth` marks the layout. HEVC Main 10 and AV1 Main
10-bit, whose `av1_profile(bit_depth)` comes from `color_config.BitDepth`, both
decode bit-exact against the software decoder on the 3060.

The GPU-texture path carries 10-bit too: the `YcbcrConverter` picks its formats
from the decode bit depth, so a `G10X6` frame samples through a 10-bit
`VkSamplerYcbcrConversion` and stores into an `R16G16B16A16_SFLOAT` image, the
`rgba16f` compute shader, imported as a `Rgba16Float` `wgpu::Texture` matching
the CPU reference under the stream's matrix. The float target preserves the full
10-bit precision and is where the transfer stage operates.

### HDR transfer and tone mapping

The fixed-function ycbcr hardware does the matrix and range but not the transfer
function, so an HDR (PQ / HLG) stream reaches the compute pass as its raw
transfer-encoded R'G'B'. `VideoColorSpace` carries a `TransferFunction` (PQ =
CICP 16, HLG = 18, else SDR) resolved from the stream, and the
`create_*_dpb_decoder_gpu_tonemap` constructors turn on a transfer stage in the
`rgba16f` shader, selected by a push constant: EOTF (PQ ST 2084 / HLG B67), then
BT.2390 EETF display mapping (maxRGB, source 1000 to target 100 nits), then
BT.2020 to BT.709 gamut, then BT.709 OETF, yielding display-ready SDR. GPU output
matches a CPU port of the same pipeline, and the transfer math is unit-pinned to
spec anchors. It is opt-in: the default GPU path stays passthrough, matrix and
range only, with the stream's PQ / HLG encoding preserved in the float target for
the HDR swapchain.

### HDR swapchain present

`vulkanhdrsink` (`hdr-present`). wgpu 29's surface config has no colour-space
setting, so `VulkanHdrSink` owns a raw `VK_KHR_swapchain` on the decode device's
`VkInstance`. The present extensions, `VK_KHR_swapchain` and
`VK_EXT_hdr_metadata` when advertised, are enabled conditionally in
`open_decode_device` so a decode-only GPU is unaffected.

It negotiates the best colour space the surface offers, `HDR10_ST2084` PQ, else
`EXTENDED_SRGB_LINEAR` scRGB, else SDR, and presents the passthrough PQ
`Rgba16Float` texture by a raw `vkCmdBlitImage` into the acquired swapchain
image. The acquire, blit and present chain is ordered by GPU semaphores,
`image_available` plus a per-image `render_finished`, with one in-flight fence
waited at the top of the next frame so nothing stalls mid-present. BT.2020
mastering metadata is attached via `vkSetHdrMetadataEXT` when available.

Surface-format and colour-space selection and metadata construction are
unit-tested. The on-screen present is validated live via
`examples/vulkan_video_hdr_on_screen.rs`, since HDR depends on display and
compositor. This completes the HDR chain: colour matrix, 10-bit decode, 10-bit
GPU texture, PQ/HLG tone-map, HDR10 present.

### AV1

AV1 is not NAL or Annex-B framed: `av1_obus` walks the low-overhead OBU stream by
its LEB128 size fields, bounds-checked. `parse_av1_sequence_header` reads the
sequence header OBU, operating points, optional timing and decoder-model info,
order-hint config and the full `color_config`, into an `Av1SequenceHeader`, which
`to_std_av1_seq_header` maps onto `StdVideoAV1SequenceHeader` plus an owned
`StdVideoAV1ColorConfig` block. The Std AV1 color enums are numeric-equal to the
AV1 spec codepoints, so they cast directly. `av1_frame_infos` classifies each
coded frame from its frame-header lead. GPU-free unit tests cover a real libaom
640x480 fixture. The full uncompressed frame header parses through
`parse_av1_frame_header` plus all sub-parses, validated field-by-field against
ffmpeg `trace_headers`.

The session is `open_av1_decode_device` plus `av1_profile` plus
`create_av1_session`, the last carrying the Std sequence header in
`VkVideoDecodeAV1SessionParametersCreateInfoKHR`. Parameter creation makes the
driver validate the mapping. `Av1DpbDecoder` maps the header onto
`StdVideoDecodeAV1PictureInfo` plus sub-structs (`to_std_av1_picture_info`) and
manages AV1's reference model: a pool of `NUM_REF_FRAMES + 1` physical DPB
images, `ref_frame_idx` to slot mapping, `refresh_frame_flags` remap, per-tile
offsets, `vkCmdDecodeVideoKHR`. The whole fixture, 1 key plus 9 inter frames
including compound and temporal-MV inter frames, decodes bit-exact against the
ffmpeg software decoder on the 3060, SAD per pixel 0 on every frame.

That needed one non-obvious default: the loop-filter reference deltas from
`setup_past_independence` are `[INTRA=1, LAST/LAST2/LAST3=0, GOLDEN=-1,
BWDREF=0, ALTREF2=-1, ALTREF=-1]`. The ALTREF2 and ALTREF entries are -1, not 0.
Defaulting them to 0 leaves in-loop deblocking mis-configured for compound blocks
referencing the alt frames, a small residual on inter frames past the first.

`av1_tile_layout` parses the `OBU_FRAME` tile-group header plus the per-tile
`TileSizeBytes` size prefixes into the driver's per-tile offset and size lists,
and a 2x2 and a 4x4 libaom clip decode bit-exact on the 3060. A stream where
decode order differs from display order takes a synchronous reorder-aware path:
`scan_ops` builds the op list, non-shown alt-ref frames decode into the DPB
without emitting, and each `show_existing_frame` emits the referenced stored slot
at its display position.

Film grain is synthesized on the decoded NV12. The 3060 exposes only
`DPB_AND_OUTPUT_COINCIDE` for AV1, so the driver cannot apply grain, which needs
a distinct output image, and `apply_film_grain_nv12` runs the full AV1 grain
synthesis (spec 7.18.3, ported from the re_rav1d scalar reference) on the
grain-free hardware reconstruction, bit-exact against dav1d for luma and chroma.
The GPU-texture path applies the same grain: since the ycbcr compute pass
produces the grain-free reconstruction, `grained_slot_to_texture` reads the
displayed slot back to NV12, the GPU DPB images carrying `TRANSFER_SRC`, runs
`apply_film_grain_nv12`, and uploads the result to the RGBA texture. Grain is
output-only, so the read-back leaves the DPB reference untouched and a grain-free
displayed frame stays on the zero-copy GPU convert.

`StdVideoAV1LoopRestoration::LoopRestorationSize` is the `1 + lr_unit_shift`
encoding, not the pixel unit size, matching ffmpeg's Vulkan hwaccel. Getting it
wrong desyncs the whole frame, so Wiener and SGR loop restoration depend on it.

### DpbCore and the two submit paths

All three `*DpbDecoder`s fold their GPU plumbing onto one codec-agnostic
`DpbCore`: device, session, DPB image pool, readback buffer, command pool, and
the `record_decode` barrier plus begin/decode/end recording. The codec-specific
decoders carry only the `Std*` mapping and reference bookkeeping. `DpbCore` runs
two submit paths off that one recorder.

The texture path (`decode_all_to_textures`, the player) converts each decoded
slot to an RGBA `wgpu::Texture` through a persistent `YcbcrConverter`. The ycbcr
conversion, sampler, descriptor-set layout and compute pipeline are built once,
not rebuilt per picture, with formats chosen from the decode bit depth, 8-bit
`G8_B8R8` to `Rgba8Unorm` or 10-bit `G10X6` to `Rgba16Float`. It chains the
decode to its conversion with a `sem_dc` semaphore: the decode is submitted on
the decode queue signalling `sem_dc` with no fence, and the compute pass on the
compute queue waits `sem_dc`, so the per-picture CPU prep, RGBA image plus memory
allocation plus views plus descriptor set, overlaps the decode's GPU execution
with no mid-picture fence wait. That is about 1.9x over the naive per-picture
rebuild plus double fence wait, about 690 fps at 640x480 on the 3060.

It is not pipelined across pictures. The conversion transitions the DPB slot in
place and the next decode references that slot, so a decode must wait the
previous slot's conversion restore, and the required cross-queue semaphore is
exactly what forbids cross-frame overlap. An intermediate NV12 copy would
decouple them at the cost of the copy.

The system NV12 path (`decode_all`) is pipelined through a fixed-depth in-flight
ring: `DECODE_RING_DEPTH`, a second `RESET_COMMAND_BUFFER` command pool with a
persistent command buffer and fence per slot, and one readback buffer sized
`DECODE_RING_DEPTH` frames so each slot copies to its own region. Each picture is
recorded and submitted without waiting. The oldest slot is retired, fence waited,
its NV12 read back and its bitstream freed, only when the ring wraps onto it, and
a final drain collects the tail. In-order execution on the single decode queue
preserves DPB reference correctness because references are CPU-side bookkeeping,
so only the readback buffer needs per-slot isolation. `reset` (seek) and `Drop`
drain the ring first. This keeps CPU record and fence-wait latency behind GPU
decode work instead of stalling after every picture, about 16% higher batch
decode throughput on the 3060 measured on H.264, and the bit-exact-vs-ffmpeg
guards go through this path unchanged.

The streaming `VulkanVideoDec` element decodes one access unit per `process`
call, so it drains per AU by design. The ring win is on the batch `decode_all`
used by the player and tests.

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

The parsers `h264parse`, `h265parse`, `av1parse`, `mpegvideoparse`,
`mpeg4videoparse` and `vc1parse` take the frame rate from the bitstream (the SPS
VUI `timing_info`, the sequence header or VOL, `av1parse` reads none). When the
bitstream codes none they keep the rate an upstream `CapsChanged` declared, such
as a demuxer's container rate, else `Rate::Any`. They never take it from
`configure_pipeline`, whose caps carry the negotiation placeholder fixated to
1 fps.

The decoders skip that placeholder too. `ffmpegdec`, `dav1ddec` and `rav1ddec`
ignore the rate in the first `configure_pipeline` call and take it from the
runner's later calls, which carry the caps an upstream `CapsChanged` declared.

In re-framing mode `h265parse` drops the RASL access units of a BLA, and of a
CRA that starts the stream, follows an end-of-sequence or end-of-bitstream NAL,
or follows a `Flush` or `Segment`. Those pictures reference pictures before the
random access point that the decoder never received (`NoRaslOutputFlag == 1`,
H.265 8.1.3). RASL pictures of any other CRA decode normally and pass through.
A `Segment` that only lowers the previous one's start (same rate, base and stop), as a
demuxer sends for a unit presenting earlier, continues the bitstream and leaves the
filter alone.

The access-unit framer behind `h264parse`, `h265parse` and the MPEG video parsers
releases a buffer's trailing access unit at once only while every buffer so far
has begun on an access-unit start, at least two in a row. One buffer that begins
inside an access unit turns that off for the rest of the stream, and the framer
goes back to holding each access unit until the next one starts. MPEG-TS muxers
like Apple's fill each PES to a packet boundary, so a PES can begin with the
previous picture's last bytes while some PES still open on a picture by chance.

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
| `Vaapi` | `h264` + `AV_HWDEVICE_TYPE_VAAPI` | `System` | GPU decode, surface downloaded with `av_hwframe_transfer_data`, NV12 unless `output-format` says otherwise. The Linux AMD / Intel hardware path for H.264 and H.265. Pin the render node with `with_vaapi_device` or the `device` property. Launch name `ffmpegvaapidec`, which the gst `va*dec` / `vaapi*dec` names resolve to. |

The Linux hardware-decode ranking is `VulkanVideoDec` first, vendor-neutral and
picked by the domain-aware search for GPU-domain consumers, with `Vaapi` as the
hardware route into system memory.

The other codecs libavcodec decodes open the same way. A ProRes decoder gets
the profile fourcc as its `codec_tag`, which picks its sample depth. Theora's identification,
comment and setup packets arrive in band ahead of its frames, from `oggdemux`
or `matroskademux`. The decoder holds them, opens libavcodec's `theora` decoder
on them Xiph-laced as extradata, and drops the copies a demuxer re-sends after
a flush.

### Output formats

Output caps are `Caps::RawVideo` with the layout chosen by `with_output_format`
or the `output-format` property. All are also pad-template alternatives, so a
downstream that pins one auto-plugs a decoder built for it. A downstream taking
any raw format gets an `OutputFormat::Auto` decoder, since I420 leads the
template.

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
- `A420` is I420 followed by a full-resolution 8-bit alpha plane. `Auto` emits
  it for a `YUVA420P` source (VP6 with alpha), all four planes copied, and
  `videoconvert` drops the alpha for I420 or carries it into RGBA / BGRA.
- `A444p12` is `I444p12` followed by a full-resolution 12-bit alpha plane, the
  `YUVA444P12` ProRes 4444 decodes to with transparency. `videoconvert` narrows
  that alpha to 8 bits for RGBA and widens 8-bit alpha by repeating its top bits,
  so opaque stays 4095.

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
`MfDecode`.

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
- Auto-plug skips `NvDec` for a stream outside the size range
  `cuvidGetDecoderCaps` reports for its codec, AV1 below 128x128 on the RTX
  3060. The factory registers an input check (`ElementFactory::accepting_input`)
  that the search runs after the sink template matches. The first check queries
  device 0 through a temporary CUDA context and caches the result for the
  process. With no CUDA device every size passes. The IVF probe reports the
  header size so `filesrc ! decodebin` knows it when it plugs. The solver does
  not refuse: a demuxer's placeholder caps fixate to its range minimum before the
  real size arrives.
- An on-hardware test on the RTX 3060 runs the full native loop, a synthesized
  CUDA NV12 surface encoded by `NvEnc` and decoded by `NvDec` back to CUDA NV12,
  asserting geometry and, via a small device-to-host copy, real luma content. It
  skips with no NVIDIA GPU, and the `nvdec` feature is CI-excluded.
- Before it creates or reconfigures a decoder, the sequence callback asks
  `cuvidGetDecoderCaps` about the stream's codec, chroma format and bit depth,
  caching each answer for the element's lifetime, and refuses the stream with
  `CapsMismatch` and a logged reason when the GPU has no decoder for it, cannot
  output it to NV12 or P016, or the coded size is outside the minimum, maximum or
  macroblock limit. A CUDA or CUVID call that still fails logs the call and the
  driver's error name from `cuGetErrorName`. The open-caps probe behind autoplug
  stays silent, so a host without a GPU logs nothing.
- nvdec gives NVIDIA's parser one VP9 frame per parse call, splitting each
  superframe itself: the parser builds picture parameters the decoder rejects for
  some superframes taken whole (two hidden frames then a show_existing_frame),
  and the same frames parsed one at a time decode bit-exact. H.264 streams with
  more than one slice group (FMO) are refused from the PPS: the CUVID picture
  parameters carry FMO fields, but on an RTX 2080 Ti `cuvidDecodePicture` rejects
  those pictures and ffmpeg's `h264_cuvid` returns wrong ones.

## Decoder cost model

When the input caps carry a fixed picture size, `decodebin` picks the decode
chain with the lowest per-frame cost rather than the shortest one (M1718, the
rule is in [caps.md](caps.md)). The constants are one table in
`g2g-plugins/src/registry.rs`, applied with `ElementFactory::cost` and
`Registry::register_transfer_cost`.

A container's picture size reaches the plan through the primary-stream hook's
stream caps, read from the file's first 4 MiB. MP4 takes it from the sample
entry and Matroska from the Tracks pixel size. MPEG-TS takes it from the first
SPS in the video PID's PES units within the 4 MiB prefix. The last unit is
completed at the end of the prefix, so a one-frame file is sized too. The
decodebin primary-stream hook and the playbin / fan-out probe share this read.
A size the prefix does not give, or one outside the 16..65535 placeholder range,
leaves the stream unpriced, and the chain falls back to registration order.

Measured on 2026-10-20 on a Ryzen 7 6800U with an RTX 3060, while other builds
were running: a release `g2g-launch`, `filesrc ! decodebin ! fakesink` over a
300-frame clip, wall time divided by frames.

| input | `ffmpegdec` | `nvdec` + download | `vulkanvideodec` + download |
| :--- | :--- | :--- | :--- |
| 1920x1080 H.264 (2.07 MP, 3.1 MB NV12) | 1.3 ms | 3.3 ms | 4.6 ms |
| 3840x2160 HEVC (8.29 MP, 12.4 MB NV12) | 5.0 ms | 4.9 ms | 7.3 ms |

Derived constants: `ffmpegdec` 600 us per megapixel (0.63 and 0.60 measured),
`nvdec` 2800 us per frame, `vulkanvideodec` 3700 us per frame, `cudadownload`
170 us per MB and `wgpudownload` 290 us per MB. The download is priced for one
frame of the decoder's chosen output format, NV12 at 1.5 bytes per pixel. A
System-memory consumer crosses over at about 8 MP, so 4K HEVC takes `nvdec` and
1080p keeps `ffmpegdec`. The larger gain at 4K is CPU time: 12.6 s for the clip
in software against 0.5 s with `nvdec`.

`dav1ddec`, `rav1ddec`, the VA-API decoders and the platform decoders are not
measured. Their cost is unknown, so a chain through one is never priced and the
score order picks as before.

A size reaches the search only from a probe that reads it: a raw Annex-B H.264
or H.265 file through `filesrc` (the content sniff parses the SPS), an IVF
header, and the SDP of an RTSP stream with audio that `uridecodebin` or
`playbin` probes. MP4, Matroska and MPEG-TS probes report no picture size, and
`rtspsrc ! decodebin` has no parse-time DESCRIBE, so those inputs keep the
shortest chain.

With `ffmpeg` built, `ffmpegdec` decodes AV1 ahead of `rav1ddec` (rank -10) and
`dav1ddec` (unpriced), so a test that needs the AV1 track selected checks the
demuxer's `stream` or the source's caps, not the decoder name.

## End-to-end RTSP pipeline

```
RtspSrc ──► H264Parse ──► [decoder] ──► [ML / display / encode]
(System / H264)            (System / DmaBuf / Cuda / D3D11Texture; NV12)
```

| Platform | Decoder element | Feature | Output |
| :--- | :--- | :--- | :--- |
| Linux software | `FfmpegH264Dec` (`Software`) | `ffmpeg` | `System` / I420 |
| Linux + NVIDIA | `FfmpegH264Dec` (`NvdecCuvid` / `NvdecCuda`) | `ffmpeg` + `cuda` | `System` / `Cuda` / NV12 |
| Linux + VAAPI | `FfmpegH264Dec` (`Vaapi`) | `ffmpeg` | `System` / NV12 |
| Windows | `MfDecode` | `mf-decode` | `System` / NV12 |

### RtspSrc and RtspSrcN

`RtspSrcN` (`rtspsrcn.rs`, `rtsp` feature) is the one RTSP client. It connects
via `retina` using RTSP/RTP over TCP (or the `protocols` order) with
`FrameFormat::SIMPLE`, so H.264 and H.265 arrive Annex-B with SPS/PPS ahead of
every keyframe, and it drops access units until the first keyframe of each
session so a mid-GOP tune-in never feeds a decoder slices without parameter
sets. Output 0 is the video, output 1 the first audio stream the SDP offers,
AAC via `mpeg4-generic` or G.711 via `pcma` / `pcmu`, as a `MultiOutputSource`.
Every track takes the same lower transport and one `FrameFormat::SIMPLE` SETUP,
so AAC arrives ADTS-framed and needs no out-of-band config. `with_credentials`
supplies the DESCRIBE/SETUP account, threaded into retina's `SessionOptions`.

`RtspSrc` (`rtspsrc.rs`) is `OnePortSource<RtspSrcN>`: a video-only `RtspSrcN`
run as a `SourceLoop`, its port 0 the single output. `OnePortSource`
(`oneport.rs`) takes the properties the single-output form exposes, here every
`RtspSrcN` property except `audio-format` and `onvif-metadata`, which pick the
pad set.

Negotiation connects, DESCRIBEs and SETUPs every subscribed track, retrying a
failed connect per the reconnect policy, so the video pad advertises the real
codec and geometry. `RtspSrcN` does this in `resolve_output_caps`, which the
graph runner awaits for a fan-out source and `OnePortSource` awaits in
`intercept_caps`. The first session of `run` PLAYs that set-up session instead
of connecting again. A server that keeps the parameter sets out of the SDP
leaves the geometry a placeholder range with the real codec, refined by a
`CapsChanged` once the first SPS arrives. `with_expected_dims` (or `width` and
`height`) skips the connect and advertises fixed H.264 geometry. Framerate comes
from the VUI `timing_info` (`time_scale / (2 * num_units_in_tick)`) when
present, else a fixable placeholder range.

The source reports live latency with a zero minimum. On a network failure with
`reconnect` set it pushes a `Flush` to every open pad, waits the backoff, and
connects afresh. The emitted timeline jumps a second past the last session's
highest PTS, but only after a session that emitted media, so a run of failed
attempts does not move it.

A server that closes the connection mid-stream ends a recording, an SDP whose
`a=range` has an end such as `npt=0-30`, with `Eos`. On a live feed the close
is a lost session when `reconnect` is set, failing with
`G2gError::Network(NetworkFailure::Stream)` once the attempts run out, and an
`Eos` without it, as GStreamer's `rtspsrc` ends on a server EOF.

Both RTSP clients fail DESCRIBE, SETUP, PLAY and depacketizer setup with
`G2gError::Network(NetworkFailure::Connect)`, and any error after media starts
with `G2gError::RtspStream`, logging retina's message on the element's category.
Retina's error kind is private, so a bad NAL and a dropped transport map to the
same variant.

The pads run on different RTP clocks, so each frame's PTS is
`Timestamp::elapsed`, normal play time against the server's `RTP-Info` origin
which spans both streams, converted at that stream's clock rate and rebased on
the session's first frame: one timeline starting at zero with the stream's A/V
offset intact. `InitialTimestampPolicy::Permissive` keeps a server that omits
`rtptime` playable, each stream starting at its own first packet. With one
stream set up retina starts it at its first packet whatever the policy, so the
video-only form stamps from its first emitted frame.

The audio pad negotiates at the decoder-facing caps, not the SDP's. A compressed
`sample_rate` is matched for equality (the negotiation lifecycle in
[README.md](README.md)), so AAC advertises the
`0/0` sentinel the demuxers use and G.711 the 8 kHz rate its decoder declares.

Pad count is fixed before the run, so the `playbin uri=rtsp://...` hook DESCRIBEs
first (`blocking_probe_tracks`) and declines to the single-pad `RtspSrc` chain
for a video-only stream. A launch line declares the count by linking pads:
`rtspsrcn name=s location=... s. ! ... s. ! ...`. gst's `rtspsrc` has a pad per
stream, so `rtspsrc` is registered both as the `RtspSrc` source and as this
fan-out source, and the parser builds the fan-out form when more than one of its
pads is linked, or when the one linked branch does not take video. The fan-out
form always has a second pad, audio unless `onvif-metadata` takes it, and each bare `s.` takes the first free
pad whose kind its first element takes, as gst links a pad by caps.

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

The `tt:Event` part becomes one more output frame, after the analytics ones,
carrying an `OnvifEventMeta` (defined in `onvifmetadata.rs`, not `g2g-core`)
with one `OnvifEventMessage` per `wsnt:NotificationMessage` directly under a
`tt:Event`: the `wsnt:Topic` text as written, the `tt:Message` `UtcTime`, its
`PropertyOperation` (`Initialized`, `Changed`, `Deleted`) and the `tt:SimpleItem`
name-value pairs of its `Source`, `Key` and `Data` groups. Field names follow
`onvif.xsd` (`tt:Message`, `tt:ItemList`), `metadatastream.xsd`
(`tt:EventStream`) and WS-BaseNotification `b-2.xsd`. `tt:ElementItem`s are
skipped. The frame takes the input document's `WallClockMeta` rather than a
message's `UtcTime`, as GStreamer's `onvifmetadataparse` times non-frame
elements by the buffer that carried them. A message that breaks the schema (no
readable `UtcTime`, an unknown `PropertyOperation`, an item without its `Name`
or `Value`) is dropped alone. Messages per document
(`MAX_EVENT_MESSAGES_PER_DOCUMENT`), items per message (`MAX_ITEMS_PER_MESSAGE`)
and topic, name and value length (`MAX_EVENT_TEXT_BYTES`) are bounded: past the
first bound the rest are dropped, past the other two the message is.

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
replacing it, appends event messages to the frame's `OnvifEventMeta` the same
way, and drops metadata more than `max-lateness` (default 200 ms) behind
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
`DPB_AND_OUTPUT_COINCIDE` flag that `intercept_caps`, DPB sizing and the decode
model negotiate against. On the RTX 3060: H.264 to 4096 square, H.265 and AV1 to
8192 square, output coincides with the DPB. On the Radeon 680M under Mesa RADV:
H.264 to 4096 square, H.265 and AV1 to 8192x4352, output distinct from the DPB.

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
which a decode accesses a picture resource. The readback copy takes the picture's own extent back out of that
image, and the YCbCr compute pass is given the padded extent so its normalized
coordinates still land on the picture's texels. The padding never reaches an
output frame: `output_extent` is the picture size the caller gets, `coded_extent`
the size every picture resource is bound at, which the spec requires to stay
inside the device's coded-extent range.

Below the device's minimum coded extent no codec decodes correctly. H.264 comes
out wrong on RADV or loses the device on NVIDIA. AV1 comes out flat gray on
NVIDIA, whose AV1 minimum is 128x128. H.265 comes out wrong on the RTX 2080 Ti,
whose minimum is 144x144, though the RTX 3060 decodes it correctly. Session
creation refuses a stream of any codec below the minimum with `StreamRefused`,
which names the cause and which the element reports as `CapsMismatch`. An AV1 frame can be coded smaller than its
sequence's maximum size, so the decoder also refuses each AV1 frame coded below
the minimum. Above the device's maximum coded extent the session clamps the
picture and decodes zeros, so session creation refuses an H.264 or H.265 stream
larger than the maximum the same way (the RTX 3060's H.265 maximum is 8192,
below `PICSIZE_A` / `PICSIZE_B` at 8440). An AV1 sequence maximum may exceed the
device's while its frames do not, so AV1 is refused per frame instead. Auto-plug skips `vulkanvideodec` for the same sizes through an input
check (`ElementFactory::accepting_input`), so `decodebin` takes a software
decoder. The check compares an H.264 stream at its macroblock-aligned coded size,
which is what the device minimum applies to. The check reads the decode capabilities of
the GPU the element would open, once per process on first use, and accepts every
input when no Vulkan video device answers.

The output is the stream's crop rectangle of the coded picture: the H.264 SPS
frame crop or the H.265 SPS conformance window, so a 1080p H.264 stream coded at
1920x1088 comes out 1920x1080. The session computes it once from the SPS as
`output_offset` plus `output_extent`, and every output path reads from that
offset: the readback copy, the YCbCr compute pass (pushed as the sample origin),
and the two-plane texture copy. The H.265 VUI default display window is not
applied, matching ffmpeg's default. A crop that reaches outside the coded picture
or leaves it empty fails session creation with `UnsupportedStream`.

Only AV1 produces an odd picture size, since its frame size is any value, while
H.264 and H.265 crop a 4:2:0 picture in two-sample units. The images are always
even, so the readback copies `ceil(w/2) x ceil(h/2)` CbCr pairs out of them, the
tight NV12 / P010 layout of the core caps, with the chroma plane starting on a
4-byte boundary of the readback buffer as a copy on a transfer-only queue needs.
Film grain repeats the last luma column for the chroma average of an odd width,
as dav1d does. The RGBA outputs carry any size. Vulkan and wgpu both reject a
4:2:0 multi-planar image with an odd side, so the two-plane `NV12` / `P010`
texture output of an odd picture is a texture one column or row larger, filled
from the coded image (decoded samples past the picture edge, since the coded
size is aligned) or left zero on the film-grain upload, and the frame's
`OwnedWgpuTexture` / `OwnedVulkanTexture` carries the picture size. A consumer
reads the picture at that size, never at the texture's: `wgpudownload` copies
each plane at the picture extent, which gives the tight layout again, and
`wgpusink` indexes both NV12 blits with the negotiated size.

A driver reporting `DPB_AND_OUTPUT_COINCIDE` decodes into its reference slot, and
one image per slot serves as both. A driver that does not, Mesa RADV for all
three codecs, writes the picture into a second image in the same decode
operation. There the format query runs twice, once for `VIDEO_DECODE_DST_KHR` and
once for `VIDEO_DECODE_DPB_KHR` alone (the combined query returns no format at
all, which is what used to fail every session create), and each DPB slot is
created in the reference format with the DPB usage bit alone, paired one to one
with an output image in the decode format carrying the output bit plus
`TRANSFER_SRC` and, on the GPU path, `SAMPLED`. The decode binds the slot as its
setup reference and the paired image as `dstPictureResource`; reference lists
keep naming slot images. Every reader, the readback copy, the YCbCr pass, the
two-plane copy and the film-grain read-back, takes the paired output image in
`VIDEO_DECODE_DST_KHR` layout instead of the slot in `VIDEO_DECODE_DPB_KHR`, and
skips the transition back that keeps a coincide slot a valid reference. Pairing
per slot rather than per decode is what makes AV1 `show_existing_frame` and
reorder re-display need no copy: a slot's picture stays readable until that slot
is decoded into again, which cannot happen while it is still referenced.

A driver that does not report `SEPARATE_REFERENCE_IMAGES`, the NVIDIA driver for
all three codecs, requires every reference slot of a
session to be a layer of one image. There the DPB is one image with an array
layer per slot and a single-layer view per slot. The picture resource binds that
view, so its `baseArrayLayer` is 0, and every barrier, readback copy, plane copy
and sampling view names the slot's layer: a barrier over the whole image would
take the other slots' references to `UNDEFINED` with it. A driver that reports the
capability, RADV, gets one image per slot. Distinct-model output images are never
references and stay one image per slot on both.

A video-decode-only queue family, which is what RADV exposes, supports no
transfer commands, so the readback copy cannot ride in the decode's own command
buffer. There the copy is recorded on the compute family's queue instead and
submitted after a per-ring-slot semaphore the decode signals, and every readable
picture image is `CONCURRENT` across the families that touch it. The decode queue
copies in its own command buffer when its family allows it, which is one
submission per picture instead of two.

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

The `WgpuTexture` output offers `Rgba8` first, then `Nv12` and `P010`. When the
solved caps pin either two-plane format, the converter copies the decoded slot's
two planes into a fresh image of the decoder's own format
(`YcbcrConverter::copy_nv12`, a plane-wise `vkCmdCopyImage` on the compute queue
in place of the ycbcr dispatch) and imports it as a `TextureFormat::NV12` (8-bit
`G8_B8R8`) or `TextureFormat::P010` (10-bit `G10X6`) wgpu texture behind
`WgpuNv12Texture`, so a consumer samples the planes through `Plane0` / `Plane1`
views with no colour conversion. Those views are in each plane's own format, so
the copy image is created `MUTABLE_FORMAT | EXTENDED_USAGE`, as wgpu creates its
own two-plane textures. The decode device requests whichever of
`TEXTURE_FORMAT_NV12` / `TEXTURE_FORMAT_P010` the adapter offers.

A stream's bit depth is only known once its parameter sets parse, long after the
caps solve, so the source pad offers both formats and the solve picks `Nv12`
when downstream takes either. The solved caps the runner hands the decoder
therefore choose only two-plane against RGBA: every path announces the format
its frames carry, `Nv12` or `P010`, from the decoded bit depth, in a
`CapsChanged` ahead of the first frame. A downstream capsfilter pinning the other
format refuses that `CapsChanged` with `CapsMismatch`, so samples are never
mislabelled or truncated.

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
bridges allocate, `TextureFormat::NV12` and `TextureFormat::P010` to the
decoder's two planes, and an uncompressed colour format to a finished picture.

wgpu refuses to bind a multi-planar texture whole, so the planes come from
`gpu::nv12_plane_views`: full-size `Plane0` and half-size `Plane1`, each viewed
in the plane format the texture's own format dictates (`R8Unorm` / `Rg8Unorm` for
NV12, `R16Unorm` / `Rg16Unorm` for P010) and each a `texture_2d<f32>`. Plane
samples are normalized, so one shader reads either depth. `WgpuSink` holds a blit
pipeline per layout and picks one per frame, so one sink negotiated for NV12
renders the same picture from a decoder's texture and from a system-memory upload
of the same clip. Its two NV12 fragment stages share one `ycbcr_to_rgb` step and both fetch
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
reference-picture-set DPB management, and the reference-slot lists
(`RefPicSetStCurrBefore/After`, POC-keyed reference info) handed to
`vkCmdDecodeVideoKHR`, reusing the H.264 DPB machinery. The whole fixture of
IDR and CRA GOPs decodes bit-exact against the ffmpeg software decoder on the
3060, also straight to GPU-resident RGBA `wgpu::Texture`s.

The decoder keeps the newest VPS, SPS and PPS of each id and parses each slice
against the sets its PPS names. When a picture's sets differ from the ones its
session parameters hold, it creates new session parameters, draining the ring
first. The element rebuilds the session only when the active SPS of an access
unit's first slice changes the picture size, conformance window, DPB size or
bit depth, and that SPS can be activated by a slice in an access unit that
carries no SPS. Scaling lists, coded or default, and non-uniform tile widths go
into the Std SPS and PPS. The DPB is sized from the highest sub-layer's
`sps_max_dec_pic_buffering`. Every decoded picture stays a short-term reference
until a later reference-picture set drops it, so sub-layer non-reference
pictures still serve higher sub-layers. Output follows C.5.2: the reorder
buffer bumps on `sps_max_num_reorder_pics` or when the references plus the
waiting pictures exceed the DPB. It drops pictures with `pic_output_flag` 0. At
an IRAP that starts a coded video sequence it drops the waiting pictures on
`no_output_of_prior_pics_flag`, or on a CRA. An end-of-sequence NAL makes the
next CRA start a sequence, which skips its RASL pictures and resets the
order-count MSB.

Both paths frame slices with a 3-byte start code (`00 00 01`). NVIDIA's Vulkan
HEVC slice-header parser breaks every non-IDR slice on a 4-byte one, and ANV
starts the H.264 slice read three bytes past each slice offset.

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
fixed-capacity so the header stays `Copy`. The DPB applies every marking
operation of 8.2.5.4, long-term ones included: a long-term picture carries
`used_for_long_term_reference` and its `LongTermFrameIdx` as the FrameNum in the
driver's reference info, and MMCO 5 resets the POC state and frame_num and
starts a new reorder group. A frame_num gap under
`gaps_in_frame_num_value_allowed_flag` makes the 8.2.5.2 non-existing frames
short-term references that hold a DPB slot and go to the driver with
`is_non_existing`, because a driver whose firmware builds the L0 / L1 lists from
the reference slots (RADV on VCN) counts them. Vulkan only takes a slot a decode
has set up since the session reset as a reference, so when the SPS allows gaps
the session's first IDR is also decoded into every other slot, and a gap frame
can take any of them. A stream that does not start at an IDR leaves a gap frame
in a slot nothing has set up out of the driver's list.

Parameter sets are kept by id. The decoder parses each slice against its SPS
and PPS tables and recreates only the `VkVideoSessionParametersKHR` when a
picture's sets differ from what the session holds, after draining the ring, so a
PPS re-sent with new content before a non-IDR picture keeps the DPB. The
element rebuilds the session only when the SPS shape changes (extent, crop,
reference count, chroma, bit depth). POC type 1 carries its offset cycle into
the std SPS, `PicOrderCnt` goes to the driver as the real top and bottom field
counts, and the H.264 reorder hold falls back to the level-derived
`MaxDpbFrames` from `h264parse::sps_reorder_frames` when the SPS has no VUI
`max_num_reorder_frames`.

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

A consumer that wants the unconverted 10-bit planes pins `P010` instead: the
`G10X6` copy imports as a `TextureFormat::P010` texture, and the raw
`VulkanTexture` domain hands out the `G10X6` image itself. Both are bit-exact
with the system `P010` readback of the same clip.

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

`av1_tile_layout` parses each tile-group header plus the per-tile
`TileSizeBytes` size prefixes into the driver's per-tile offset and size lists,
and a 2x2 and a 4x4 libaom clip decode bit-exact on the 3060. A frame is an
`OBU_FRAME` or an `OBU_FRAME_HEADER` followed by `OBU_TILE_GROUP`s, written to the
bitstream buffer back to back, and its tile groups must cover the frame's tiles
in order. A frame header that arrives before the frame's last tile group is a copy
(spec 7.5) and is skipped, as is every `OBU_REDUNDANT_FRAME_HEADER`. Only the last
shown frame of each temporal unit is emitted, which in a scalable stream is its
highest spatial layer, the output of aomdec and dav1d with `all_layers` off. The
OBU walk first drops every OBU outside operating point 0 (spec 5.3.1: an OBU with
an extension header whose `temporal_id` or `spatial_id` is not in
`operating_point_idc[0]`), as dav1d and aomdec do by default. Each frame is read
back and converted at its own `UpscaledWidth x FrameHeight`, the size dav1d
outputs, so a frame coded below the sequence size leaves at that size behind a
`CapsChanged`. The readback layout is per picture, bounded by the session's
output extent, and a frame larger than the session is refused. The
driver takes an AV1 frame's size only from the picture resource `codedExtent`, so
each frame and each reference is bound at its own `UpscaledWidth x FrameHeight`:
the base layer of `svc-L2T1` is 640x360 under the 1280x720 layer that predicts
from it. A stream where decode order differs from display order takes a
synchronous reorder-aware path: `av1_scan_ops` builds the op list, non-shown alt-ref frames decode into the DPB
without emitting, and each `show_existing_frame` emits the referenced stored slot
at its display position.

Film grain is synthesized on the decoded NV12. The session never asks the driver
to apply it, on either decode model, and `apply_film_grain_nv12` runs the full AV1 grain
synthesis (spec 7.18.3, ported from the re_rav1d scalar reference) on the
grain-free hardware reconstruction, bit-exact against dav1d for luma and chroma.
The GPU-texture path applies the same grain: since the ycbcr compute pass
produces the grain-free reconstruction, `grained_picture_to_texture` reads the
displayed picture back to NV12, the GPU picture images carrying `TRANSFER_SRC`, runs
`apply_film_grain_nv12`, and uploads the result into a texture in whichever layout
the caps settled on: RGBA through the CPU convert, or a `TextureFormat::NV12`
texture written one `write_texture` per plane aspect when the caps pinned the
two-plane output. Grain is output-only, so the read-back leaves the DPB reference
untouched and a grain-free displayed frame stays on the zero-copy GPU convert.
The CPU convert and two-plane upload are 8-bit. A grained 10-bit picture is
uploaded instead into a staging image of the DPB slots' format and extent and
takes the converter pass a grain-free 10-bit slot takes, to `Rgba16Float`
(tone-mapped when enabled) or a P010 texture.

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
a final drain collects the tail. `reset` (seek) and `Drop` drain the ring first.
This keeps CPU record and fence-wait latency behind GPU decode work instead of
stalling after every picture, about 16% higher batch decode throughput on the
3060 measured on H.264, and the bit-exact-vs-ffmpeg guards go through this path
unchanged.

Submission order on one queue does not by itself order one decode's reads after
another's writes, so `record_decode` opens every decode after the session's first
with a barrier whose source is the decode stage with decode read and write
access. It transitions the target slot's layer, and on the distinct model the
output image, after the earlier decodes that read or wrote them, and a global
memory barrier from decode writes to decode reads makes the references those
decodes wrote visible to this one. The first decode after a `RESET` takes the top
of the pipe as its source, since the reset drained every earlier decode.
Where the readback copy runs on another queue nothing orders a decode after the
copy of the same slot's previous picture, so a decode into a slot first retires
the ring entries through the newest one that decoded into it. On the coincide
model that copy also moves the slot image out of the decode layout and back, so
the decode retires through the newest entry for any slot it references as well.

The ycbcr pass and the two-plane copy wait the decode's semaphore at the compute
and transfer stage respectively, and their opening layout transitions take that
same stage as their source, which puts the transitions after the wait instead of
ahead of it at the top of the queue.

The streaming `VulkanVideoDec` element decodes one access unit per `process`
call, so it drains per AU by design. The ring win is on the batch `decode_all`
used by the player and tests.

## Conformance per backend

calliope's Fluster conformance run with each decoder pinned through
`G2G_PLUGIN_FEATURE_RANK` (see the calliope README), on an RTX 3060 with the
NVIDIA Vulkan driver, 2026-10-14. A pass is an output MD5 equal to the vector's
official one.

| Decoder | H.264 JVT-AVC_V1 | H.265 HEVC_V1 + RExt | AV1-TEST-VECTORS |
| :--- | ---: | ---: | ---: |
| `ffmpegdec` | 130 / 135 | 172 / 188 (2026-10-20) | 0 / 240 |
| `nvdec` | 129 / 135 | 141 / 188 | 202 / 240 |
| `vulkanvideodec` | 99 / 135 (2026-10-20) | 142 / 188 (2026-10-21) | 204 / 240 (2026-10-21) |
| `dav1ddec` | | | 240 / 240 (2026-10-21) |
| `rav1ddec` | | | 240 / 240 (2026-10-21) |

The H.264 failures every backend shares are the Extended-profile FMO and SP
slice vectors, which ffmpeg does not decode either. Every `vulkanvideodec`
H.264 failure is a nonzero exit naming the coding tool: field pictures are
refused at the slice header (MBAFF frame pictures decode), FMO at the PPS and
SP slices at the slice header, so a stream the decoder cannot handle never ends
as an empty or wrong success.

`vulkanvideodec` decodes AV1-TEST-VECTORS at 204 / 240 on the RTX 3060 (NVIDIA)
and 240 / 240 on RADV (Rembrandt), 2026-10-21. The NVIDIA gap is the 36
`size-16..66` vectors below the device minimum coded extent, which decode on
RADV. The frame-header parse passes CDEF
secondary strengths as coded and keeps each reference slot's loop-filter deltas,
segmentation features and global-motion models for `load_previous()`. Film grain
is synthesized on the CPU at 8 and 10 bits from the grain-free luma, bit-exact
with dav1d.

Both software AV1 decoders open the backend with `all_layers` off, so a scalable
stream outputs one picture per temporal unit at its highest spatial layer
(operating point 0). This is what aomdec, ffmpeg's libdav1d wrapper and the
conformance md5s produce. GStreamer's `dav1ddec` has no property for it, so g2g
exposes none.

A mid-stream `configure_pipeline` (the runner calls it on every input
`CapsChanged`, for example av1parse refining the picture size) keeps the backend
decoder, so pictures still held by frame threading are not lost. av1parse turns
off early release of the trailing temporal unit for good once one input buffer
begins inside a unit, the same rule as the H.264 / H.265 access-unit framer.

Every `vulkanvideodec` H.265 failure is also a nonzero exit naming the coding
tool, raised when the parameter sets parse, before the driver sees the stream.
The sessions are Main and Main 10, so a chroma format other than 4:2:0, a bit
depth above 10, unequal luma and chroma depths, and SPS or PPS range extension
or screen content tools are all refused. Wavefront parallel processing on a
picture one coding tree block wide is refused too. RADV (Rembrandt, pinned with
`VK_DRIVER_FILES`) gave 146 of 188 for H.265 and 97 of 135 for H.264 on 2026-10-20.

`ffmpegdec`'s 16 HEVC misses are libavcodec's own: the ffmpeg command line on
libavcodec 61 (ffmpeg 7.1) and 62 (ffmpeg 8.1) misses the same 16 vectors, the
extended-precision and high-throughput 4:4:4 16-bit RExt streams, the 14-bit
CABAC bypass-alignment ones, `Bitdepth_A` / `Bitdepth_B`, `TSUNEQBD_A`,
`VPSSPSPPS_A` and `SAODBLK_A` / `SAODBLK_B`. The count is the same with the
decoder pinned through `G2G_PLUGIN_FEATURE_RANK` and through plain `decodebin`.

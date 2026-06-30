//! Standard source / sink / transform elements for `glass2glass`.
//!
//! Per the spec (§2), this crate is `no_std + alloc` at baseline. Network
//! and OS-coupled elements (RTSP source via `retina`, V4L2, wgpu sinks)
//! live behind cargo features that imply `std`.

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]

extern crate alloc;

pub mod aacparse;
pub mod appsink;
pub mod appsrc;
pub mod opusparse;
pub mod vp8parse;
pub mod vp9parse;
pub mod av1parse;
pub mod audioconvert;
pub mod audioresample;
pub mod audiotestsrc;
pub mod volume;
pub mod audiopanorama;
pub mod audiomixer;
pub mod capsfilter;
pub mod fakesink;
pub mod h264parse;
pub mod h265parse;
pub mod identity;
pub mod avoffset;
pub mod mux;
pub mod streamdemux;
pub mod tsmuxn;
// Shared integer source-over blend used by the compositor and CPU overlays.
mod paint;

// Software RGBA8 compositor (fan-in pixel mixer): PiP / grids / overlays.
pub mod compositor;
// Analytics overlay (M101): draws AnalyticsMeta detection boxes onto RGBA8.
// Needs the per-frame metadata graph, so it is gated on `analytics`.
#[cfg(feature = "analytics")]
pub mod analyticsoverlay;
// Shared wgpu device context for the GPU elements (M103): a producer and a sink
// must share one device for a copy-free WgpuTexture handoff.
#[cfg(any(feature = "vello-overlay", feature = "wgpu-sink", feature = "cuda-wgpu"))]
pub mod gpu;
// Vello GPU companion to analyticsoverlay (M102): renders boxes with the Vello
// 2D renderer into a wgpu texture (MemoryDomain::WgpuTexture, kept on GPU).
#[cfg(feature = "vello-overlay")]
pub mod vellooverlay;
// GPU presentation sink (M103): presents MemoryDomain::WgpuTexture frames by
// blitting onto an offscreen target or a caller-provided wgpu::Surface.
#[cfg(feature = "wgpu-sink")]
pub mod wgpusink;
pub mod videoconvert;
pub mod videoscale;
pub mod videorate;
pub mod videocrop;
pub mod videoflip;
pub mod videobalance;
pub mod videobox;
pub mod alpha;
// Subtitle cue parsing (SRT / WebVTT) and the embedded bitmap font, both no_std,
// feeding the `textoverlay` element below.
pub mod subparse;
pub mod bitmapfont;
pub mod textoverlay;
// CEA-608/708 closed captions carried in-band in H.264/H.265 SEI (no_std).
pub mod cea;
// Shared pixel-format helpers for the packed-RGBA elements (videobalance, alpha).
mod pixel;
// Sans-IO H.264 RTP packetizer (RFC 3550 + 6184), the live-egress foundation.
pub mod rtppay;
// Sans-IO H.264 RTP depayloader, the receive-side inverse of rtppay.
pub mod rtpdepay;
// Sans-IO RTP jitter buffer (reorder / loss / dup detection) between a socket
// and the depayloader, the receive-side network-resilience stage.
pub mod rtpjitter;
// Sans-IO RTCP (RFC 3550 SR/RR/BYE + RFC 4585 Generic NACK) and RFC 3550
// reception statistics: the RTP control / feedback channel.
pub mod rtcp;
// Sans-IO RFC 4588 RTP retransmission (RTX) framing: wraps a resent packet in a
// distinct payload type with the original sequence number prepended.
pub mod rtx;
// Sans-IO RTP forward error correction (ULPFEC, RFC 5109): XOR repair packets
// that recover a single per-group loss with no round trip.
pub mod ulpfec;
// Sans-IO FlexFEC (RFC 8627): repair packets on a dedicated FEC SSRC with a
// variable-length mask, protecting more than ULPFEC's 16 packets and enabling
// 2-D (row + column) recovery of bursts.
pub mod flexfec;
// uridecodebin front door: URI-scheme handlers for Registry::build_uridecodebin
// (file:// -> Mp4Src, udp:// -> UdpSrc, rtsp:// -> RtspSrc, v4l2:// -> V4l2Src),
// each gated to its source's feature.
#[cfg(feature = "std")]
pub mod uridecodebin;
// A Registry pre-populated with the standard elements for parse_launch /
// gst-inspect (M107). std (the Registry is std).
#[cfg(feature = "std")]
pub mod registry;
// GStreamer porting helpers: gst->g2g element map + launch linter (M200). std
// (uses the Registry + parse_launch).
#[cfg(feature = "std")]
pub mod gst_compat;
// Dynamic (`dlopen`) plugin loader for third-party `cdylib` plugins built with
// the `g2g-plugin` SDK (M201). Behind `plugin-loader` (pulls `libloading`); the
// loaded elements register into a `Registry` the parser then uses by name.
#[cfg(feature = "plugin-loader")]
pub mod plugin_loader;
// Annex-B NAL splitting shared by rtppay (RTP) and h264util (WebCodecs).
mod annexb;
// Shared seek helper for byte-stream demuxers (M362): drives an upstream
// byte-seek (FileSrc) and re-syncs from the returned Flush.
mod demuxseek;
// MPEG-TS demuxer parsing core (no_std): PAT/PMT/PES -> elementary access units.
pub mod mpegts;
// MPEG-TS demuxer element (no_std): Caps::ByteStream{MpegTs} -> H.264, wrapping
// the mpegts parser.
pub mod tsdemux;
// MPEG-TS muxer element (no_std): one elementary stream -> Caps::ByteStream{MpegTs},
// the inverse of tsdemux.
pub mod tsmux;
// Matroska / WebM demuxer parsing core (no_std): EBML -> Tracks + Cluster frames.
pub mod matroska;
// Matroska / WebM demuxer element (no_std): Caps::ByteStream{Matroska} -> one
// selected elementary stream, wrapping the matroska parser.
pub mod mkvdemux;
// Matroska / WebM muxer element (no_std): one elementary stream ->
// Caps::ByteStream{Matroska}, the inverse of mkvdemux.
pub mod mkvmux;
// Multi-track Matroska / WebM muxer element: N elementary streams (A/V) ->
// Caps::ByteStream{Matroska}, the fan-in analog of mkvmux. std-gated: reuses the
// MP4 family's NAL / ADTS / avcC helpers, and the A/V case is only reachable with
// the std+ffmpeg encoders.
#[cfg(feature = "std")]
pub mod mkvmuxn;
// Ogg demuxer parsing core (no_std): OggS pages -> elementary-stream packets.
pub mod ogg;
// Ogg demuxer element (no_std): Caps::ByteStream{Ogg} -> Opus, wrapping the
// ogg parser.
pub mod oggdemux;
// FLV demuxer parsing core (no_std): FLV tags -> elementary access units.
pub mod flv;
// FLV demuxer element (no_std): Caps::ByteStream{Flv} -> H.264 / AAC, wrapping
// the flv parser.
pub mod flvdemux;
// FLV muxer element (no_std): one elementary stream -> Caps::ByteStream{Flv}, the
// inverse of flvdemux.
pub mod flvmux;
// Multi-track FLV muxer element: a video + audio elementary stream (A/V) ->
// Caps::ByteStream{Flv}, the fan-in analog of flvmux. std-gated: reuses the MP4
// family's NAL / ADTS / avcC helpers, like mkvmuxn.
#[cfg(feature = "std")]
pub mod flvmuxn;
// Container content sniffing (no_std): guess a ByteStreamEncoding from a header.
pub mod typefind;
pub mod videotestsrc;
// Pool-backed passthrough transform whose buffer pool is rebuilt to the
// downstream allocation proposal, including the mid-stream β re-cascade.
pub mod poolstage;

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
pub mod clock;
#[cfg(feature = "std")]
pub mod filesink;
#[cfg(feature = "std")]
pub mod filesrc;
#[cfg(feature = "std")]
mod audio;
#[cfg(feature = "std")]
mod mp4box;
#[cfg(feature = "std")]
pub mod mp4mux;
#[cfg(feature = "std")]
pub mod mp4muxn;
#[cfg(feature = "std")]
pub mod fmp4mux;
#[cfg(feature = "std")]
pub mod mp4src;
#[cfg(feature = "std")]
pub mod mp4demuxn;
#[cfg(feature = "std")]
pub mod gaplesssrc;
#[cfg(feature = "std")]
pub mod mp4audiosink;
#[cfg(feature = "std")]
pub mod mp4audiosrc;
#[cfg(feature = "std")]
pub mod syncsink;
#[cfg(feature = "std")]
pub mod wavsink;

#[cfg(feature = "rtsp")]
pub mod rtspsrc;

// ONVIF camera discovery + RTSP stream-URI resolution (OnvifSrc). Resolves a
// camera's RTSP URL over SOAP, then delegates to RtspSrc; implies `rtsp`.
#[cfg(feature = "onvif")]
pub mod onvif;

// Sans-IO RTSP 1.0 server responder (always compiled) and the tokio TCP serving
// sink (egress: hosts a pipeline's H.264 as an RTSP endpoint).
pub mod rtspserver;
#[cfg(feature = "rtsp-server")]
pub mod rtspserversink;
#[cfg(feature = "rtsp-server")]
pub mod rtspserversrc;

// Sans-IO SRT (Secure Reliable Transport) wire layer + handshake + ARQ (always
// compiled); the tokio caller sink / listener source sit behind the `srt` feature.
pub mod srt;
#[cfg(feature = "srt")]
pub mod srtcrypto;
#[cfg(feature = "srt")]
pub mod srtsink;
#[cfg(feature = "srt")]
pub mod srtsrc;

// UDP egress sink (M47): drives the M46 RtpH264Packetizer and sends RTP over a
// tokio UdpSocket, the send-side inverse of RtspSrc's receive path.
#[cfg(feature = "udp-egress")]
pub mod udpsink;

// Native WebRTC elements on the sans-IO str0m stack (ICE/DTLS/SRTP), gated
// behind the std `webrtc` feature. WebRtcSink = WHIP egress, WebRtcWhepSrc =
// WHEP ingest; webrtc_util holds the shared ICE/SDP-POST helpers. Distinct from
// the wasm-only data-channel webrtcsrc.
#[cfg(feature = "webrtc")]
mod turn;
#[cfg(feature = "webrtc")]
mod webrtc_util;
#[cfg(feature = "webrtc")]
pub mod webrtcduplex;
#[cfg(feature = "webrtc")]
pub mod webrtcsink;
#[cfg(feature = "webrtc")]
pub mod webrtcsession;
#[cfg(feature = "webrtc")]
pub mod webrtcwhepsession;
#[cfg(feature = "webrtc")]
pub mod webrtcwhepsrc;

// UDP ingress source (M91): receives RTP on a tokio UdpSocket and depayloads
// H.264 (rtpdepay) into Annex-B access units, the receive-side inverse of
// UdpSink. Raw RTP (no RTSP/SDP); see module docs.
#[cfg(feature = "udp-ingress")]
pub mod udpsrc;

// Media Foundation decode is Windows-only. The `windows` dependency is
// target-gated, so the module only exists when building for Windows with the
// `mf-decode` feature; enabling the feature on other platforms is a no-op.
#[cfg(all(target_os = "windows", feature = "mf-decode"))]
pub mod mfdecode;

// VideoToolbox H.264 decode is macOS-only, the macOS counterpart of mfdecode.
// The objc2 framework dependencies are target-gated, so the module only exists
// when building for macOS with the `vtdecode` feature; enabling the feature on
// other platforms is a no-op (first element of the macOS platform track, M218).
#[cfg(all(target_os = "macos", feature = "vtdecode"))]
pub mod vtdecode;
// VideoToolbox H.264 encode (M231), the encode counterpart of vtdecode.
#[cfg(all(target_os = "macos", feature = "vtencode"))]
pub mod vtencode;

// NDK MediaCodec H.264 decode is Android-only, the Android counterpart of
// vtdecode / mfdecode. The `ndk` dependency is target-gated, so the module only
// exists when building for Android with the `mediacodec` feature; enabling it on
// other platforms is a no-op (first element of the Android platform track, M219).
#[cfg(all(target_os = "android", feature = "mediacodec"))]
pub mod mediacodecdec;

// Buffer-flag constants + the dequeue-input-with-retries skeleton shared by the
// MediaCodec decode / encode elements below.
#[cfg(all(target_os = "android", feature = "mediacodec"))]
mod mediacodec_common;

// M306: Android MediaCodec H.264/H.265 encode (NV12 -> Annex-B), the encode
// mirror of mediacodecdec and the Android analog of mfencode / vtencode.
#[cfg(all(target_os = "android", feature = "mediacodec"))]
pub mod mediacodecenc;

// M307: Android AAudio PCM render (AAudioSink) + capture (AAudioSrc), the Android
// analog of the WASAPI / ALSA / PulseAudio audio elements.
#[cfg(all(target_os = "android", feature = "aaudio"))]
pub mod aaudio;

// M308: Android camera capture via the NDK Camera2 API (raw ndk-sys), capturing
// YUV_420_888 into an ImageReader and packing NV12. The Android analog of v4l2src.
#[cfg(all(target_os = "android", feature = "camera2"))]
pub mod camera2src;

// M304: Android MediaCodec -> wgpu/Vulkan zero-copy bridge. Imports the decoded
// AImage's AHardwareBuffer into a wgpu Vulkan device for a device-local copy
// into a sampled texture (no CPU NV12 readback). The Android analog of cudawgpu.
#[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
pub mod mediacodec_wgpu;

// Media Foundation H.264 encode, the encode-side mirror of mfdecode. Same
// Windows-only target gate; enabling the feature elsewhere is a no-op.
#[cfg(all(target_os = "windows", feature = "mf-encode"))]
pub mod mfencode;

// Media Foundation AAC audio encode/decode. Windows-only; MfAacEncode is an
// enumerated encoder, MfAacDecode wraps CLSID_MSAACDecMFT.
#[cfg(all(target_os = "windows", feature = "mf-aac"))]
pub mod mfaacencode;
#[cfg(all(target_os = "windows", feature = "mf-aac"))]
pub mod mfaacdecode;

// Shared frame-emission loop for the packet-producing encoders below.
#[cfg(any(
    feature = "av1-encode",
    feature = "vpx",
    feature = "opus",
    feature = "ffmpeg",
    feature = "nvenc"
))]
mod encoder_base;

// AV1 software encode via the pure-Rust rav1e crate (cross-platform).
#[cfg(feature = "av1-encode")]
pub mod av1enc;

// AV1 decode via libdav1d (FFI through the `dav1d` crate). Not pure Rust; links
// system libdav1d, gated behind the `dav1d` feature.
#[cfg(feature = "dav1d")]
pub mod dav1ddec;

// AV1 decode via `re_rav1d`, the pure-Rust port of dav1d. Same caps as `dav1ddec`,
// no system deps; gated behind the `rav1d` feature.
#[cfg(feature = "rav1d")]
pub mod rav1ddec;

// VP8/VP9 software encode via libvpx (FFI through vpx-encode). Not pure Rust;
// links system libvpx, gated behind the `vpx` feature.
#[cfg(feature = "vpx")]
pub mod vpxenc;

// Motion-JPEG decode via the pure-Rust zune-jpeg crate (no system deps).
#[cfg(feature = "mjpeg")]
pub mod mjpegdec;

// Motion-JPEG encode via the pure-Rust jpeg-encoder crate (no system deps).
#[cfg(feature = "mjpeg-encode")]
pub mod mjpegenc;

// Opus audio encode + decode via libopus (FFI through audiopus). Not pure Rust;
// links libopus (system or bundled-and-built), gated behind the `opus` feature.
#[cfg(feature = "opus")]
pub mod opusdec;
#[cfg(feature = "opus")]
pub mod opusenc;

// HTTP(S) byte-stream source via reqwest (the fetch layer under HLS/DASH).
#[cfg(feature = "http-src")]
pub mod httpsrc;

// Shared HTTP fetch + URL helpers for the adaptive-streaming sources.
#[cfg(feature = "http-src")]
mod fetch;

// Shared throughput-driven ABR estimator for the adaptive-streaming sources.
#[cfg(any(feature = "hls", feature = "dash"))]
mod abr;

// HLS playlist parser (pure, no_std baseline) and the HlsSrc segment source.
pub mod hls;
#[cfg(feature = "hls")]
pub mod hlssrc;
// HLS SAMPLE-AES per-sample decryptor (runs after the demuxer).
#[cfg(feature = "hls")]
pub mod sampleaesdecrypt;

// RTMP: the sans-IO protocol (always compiled) and the tokio TCP source (ingest)
// + sink (egress).
pub mod rtmp;
#[cfg(feature = "rtmp")]
pub mod rtmpsink;
#[cfg(feature = "rtmp")]
pub mod rtmpsrc;

// DASH MPD parser and the DashSrc segment source.
#[cfg(feature = "dash")]
pub mod mpd;
#[cfg(feature = "dash")]
pub mod dashsrc;

// Fragmented-MP4 / CMAF parsing (shared) and the byte-stream demuxer. In the
// std MP4 family (shares mp4box with mp4src/fmp4mux).
#[cfg(feature = "std")]
mod fmp4;
#[cfg(feature = "std")]
pub mod fmp4demux;
// Shared cbcs (MPEG-CENC) sample decryption for the HLS fMP4 and MP4 demux paths.
#[cfg(any(feature = "hls", feature = "mp4-cenc"))]
mod cenc;

// Worker-readiness latch shared by the platform display sinks below.
#[cfg(any(
    all(target_os = "windows", feature = "d3d11-sink"),
    all(target_os = "linux", feature = "wayland-sink"),
    all(target_os = "linux", feature = "cuda-gl"),
))]
mod worker_ready;

// YUV_420_888 -> NV12 packer shared by the Android ndk-image elements
// (camera2src, mediacodecdec).
#[cfg(all(target_os = "android", any(feature = "camera2", feature = "mediacodec")))]
mod yuv420;

// D3D11 present sink: displays MemoryDomain::D3D11Texture frames via a DXGI
// swapchain + D3D11 video processor. Windows-only; the analog of CudaGlSink.
#[cfg(all(target_os = "windows", feature = "d3d11-sink"))]
pub mod d3d11sink;

// WASAPI render sink: plays PCM on the default audio endpoint (shared mode).
// Windows-only; the audible-output end of the M25 audio path.
#[cfg(all(target_os = "windows", feature = "wasapi-sink"))]
pub mod wasapisink;

// WASAPI capture source: captures PCM from the default audio endpoint.
// Windows-only; the input mirror of WasapiSink.
#[cfg(all(target_os = "windows", feature = "wasapi-src"))]
pub mod wasapisrc;

// Media Foundation camera capture source: drains frames from a video capture
// device via an IMFSourceReader. Windows-only; the video sibling of WasapiSrc.
#[cfg(all(target_os = "windows", feature = "mf-video-src"))]
pub mod mfvideosrc;

// VAAPI H.264 decode via cros-codecs is Linux-only. The dependency is
// target-gated; enabling the feature on other platforms is a no-op.
#[cfg(all(target_os = "linux", feature = "vaapi"))]
pub mod vaapidec;

// ffmpeg/libavcodec H.264 decode is Linux-only here (the ffmpeg-next dep is
// target-gated). Currently software decode; VAAPI hwaccel is a follow-up that
// stays inside this module and does not change the public AsyncElement shape.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
pub mod ffmpegdec;

// Audio decode via libavcodec (AAC -> interleaved PcmS16Le), the audio sibling
// of `ffmpegdec`. Same Linux + `ffmpeg` gate.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
pub mod ffmpegaudiodec;

// H.264 encode via libavcodec (NVENC / libx264), the encode-side mirror of
// ffmpegdec (M266). Same Linux + ffmpeg gating.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
pub mod ffmpegenc;

// AAC-LC audio encode via libavcodec, the audio companion of ffmpegenc (M292).
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
pub mod ffmpegaacenc;

// Pure chroma-resampling math for the decoders (YUV444P -> 4:2:0 downsample).
// Compiled for the Linux ffmpeg build that uses it and under cfg(test) so the
// resampling logic is host-testable without libavcodec.
#[cfg(any(test, all(target_os = "linux", feature = "ffmpeg")))]
mod yuv;

// KMS/DRM display sink for NV12 frames. Linux-only (drm + drm-fourcc deps are
// target-gated). Requires DRM master at runtime; see module docs.
#[cfg(all(target_os = "linux", feature = "kms-sink"))]
pub mod kmssink;

// V4L2 capture source (UVC webcams etc.). Linux-only; streams packed YUYV
// (4:2:2) off /dev/videoN via mmap on a dedicated capture thread. See module
// docs.
#[cfg(all(target_os = "linux", feature = "v4l2"))]
pub mod v4l2src;

// libcamera capture source (NV12 / YUYV) via the system libcamera stack. The
// modern Linux camera path: covers UVC webcams plus CSI/ISP cameras. Linux-only.
#[cfg(all(target_os = "linux", feature = "libcamera"))]
pub mod libcamerasrc;

// Zero-copy libcamera -> GPU dma-buf import feasibility probe (Linux + GPU).
#[cfg(all(target_os = "linux", feature = "libcamera-dmabuf"))]
pub mod libcamera_dmabuf;

// Wayland display sink (NV12 -> XRGB8888 via wl_shm). Linux-only;
// desktop-dev convenience sink — see module docs.
#[cfg(all(target_os = "linux", feature = "wayland-sink"))]
pub mod waylandsink;

// Linux audio render sinks: the audible-output end of the audio path, the
// analogs of the Windows-only WasapiSink. Each links a different system audio
// stack and is target-gated to Linux behind its own feature.
// ALSA (libasound), the lowest-level path.
#[cfg(all(target_os = "linux", feature = "alsa-sink"))]
pub mod alsasink;
// PulseAudio / PipeWire-pulse via the blocking libpulse "simple" API.
#[cfg(all(target_os = "linux", feature = "pulse-sink"))]
pub mod pulsesink;
// PipeWire audio render sink + capture source (the modern Linux media layer).
// Both elements share the `pipewire` feature, the pipewire-rs crate, and the
// pwaudio SPA-format helper.
#[cfg(all(target_os = "linux", feature = "pipewire"))]
mod pwaudio;
#[cfg(all(target_os = "linux", feature = "pipewire"))]
pub mod pipewiresink;
#[cfg(all(target_os = "linux", feature = "pipewire"))]
pub mod pipewiresrc;

// CUDA device-memory consumers (C3 Phase 3). `CudaDownload` copies a
// `MemoryDomain::Cuda` NV12 frame back to system memory so a `NvdecCuda`
// stream reaches the CPU sinks. Hand-rolled libcuda FFI; Linux + NVIDIA only.
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod cuda;

// Native NVENC H.264 encode (M269): `NvEnc` ingests a CUDA NV12 surface (the
// NVDEC hwframe domain) and drives the NVIDIA Video Codec SDK directly, so the
// encode runs GPU-resident with no device->host read-back, the zero-copy mirror
// of the `cuda-wgpu` import bridge. Hand-rolled libnvidia-encode + libcuda FFI;
// Linux + NVIDIA only.
#[cfg(all(target_os = "linux", feature = "nvenc"))]
pub mod nvenc;

// Native NVDEC H.264 decode (M270): `NvDec` is the decode half of the
// gst-`nvcodec`-style pair, mirror of `NvEnc`. It drives the NVCUVID
// parser+decoder API directly (no libavcodec), emitting CUDA NV12 surfaces
// (`MemoryDomain::Cuda`) for a zero-copy handoff to the GPU consumers / `NvEnc`.
// Hand-rolled libnvcuvid + libcuda FFI; Linux + NVIDIA only.
#[cfg(all(target_os = "linux", feature = "nvdec"))]
pub mod nvdec;

// Shared NV12 GL ES render state for the CUDA-GL sinks (program + textures +
// per-frame CUDA upload + draw); the platform present stays in each sink.
#[cfg(all(target_os = "linux", any(feature = "cuda-gl", feature = "cuda-kms")))]
pub(crate) mod glnv12;

// CUDA-GL zero-copy-ish display sink: keeps decoded NV12 on the GPU and
// presents it via CUDA-GL interop on a Wayland EGL surface. Linux + NVIDIA.
#[cfg(all(target_os = "linux", feature = "cuda-gl"))]
pub mod cudaglsink;

// CUDA-GL display sink on DRM/KMS: the tty / no-compositor counterpart of
// cudaglsink, presenting via a GBM surface + page-flips. Linux + NVIDIA.
#[cfg(all(target_os = "linux", feature = "cuda-kms"))]
pub mod cudakmssink;

// CUDA<->wgpu zero-copy interop: imports a Vulkan external-memory image into
// CUDA so NVDEC NV12 reaches WgpuPreprocess on the GPU. Linux + NVIDIA.
#[cfg(all(target_os = "linux", feature = "cuda-wgpu"))]
pub mod cudawgpu;

// Browser / WebAssembly target (DESIGN.md §6.3), behind the `web` feature:
// WasmClock (performance.now + setTimeout), WebSocketSrc ingest, and a
// spawn_local browser entry. The wasm bindings are target-gated to wasm32, so
// enabling `web` elsewhere is a no-op, like mf-decode on Linux.
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod wasmclock;
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod web;
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod websocketsrc;

// CanvasSink (M41): present decoded RGBA frames to an HTML canvas. WebRtcSrc
// (M42): ingest over a provided RtcDataChannel. Both stable web-sys.
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod canvassink;
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod webrtcsrc;

// Pure helpers shared by the wasm elements (ms->ns conversion, the
// callback->async Inbox bridge). Compiled for the wasm `web` build and under
// `cfg(test)` so the logic is unit-testable on the host.
#[cfg(any(test, all(target_arch = "wasm32", feature = "web")))]
mod webutil;

// WebCodecs hardware decode (M40), behind the `web-codecs` feature (implies
// `web`). The build needs RUSTFLAGS=--cfg=web_sys_unstable_apis. H.264 -> RGBA.
#[cfg(all(target_arch = "wasm32", feature = "web-codecs"))]
pub mod webcodecsdecode;

// H.264 Annex-B helpers (NAL split, keyframe detection, codec string). Pure
// no_std; used by h264parse (keyframe flag) and WebCodecsDecode.
mod h264util;

// Embassy RTOS clock backend (M43): the embedded deployment-profile clock over
// embassy-time, the no_std analog of WallClock / WasmClock.
#[cfg(feature = "embassy")]
pub mod embassyclock;

// Embassy zero-alloc inter-task packet link (M45): PacketChannel over
// embassy-sync, the §6.2 stack-channel backend.
#[cfg(feature = "embassy-link")]
pub mod embassylink;

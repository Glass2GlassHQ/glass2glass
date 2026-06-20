//! Standard source / sink / transform elements for `glass2glass`.
//!
//! Per the spec (§2), this crate is `no_std + alloc` at baseline. Network
//! and OS-coupled elements (RTSP source via `retina`, V4L2, wgpu sinks)
//! live behind cargo features that imply `std`.

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]

extern crate alloc;

pub mod aacparse;
pub mod opusparse;
pub mod vp8parse;
pub mod vp9parse;
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
pub mod mux;
// Software RGBA8 compositor (fan-in pixel mixer): PiP / grids / overlays.
pub mod compositor;
// Analytics overlay (M101): draws AnalyticsMeta detection boxes onto RGBA8.
// Needs the per-frame metadata graph, so it is gated on `analytics`.
#[cfg(feature = "analytics")]
pub mod analyticsoverlay;
// Shared wgpu device context for the GPU elements (M103): a producer and a sink
// must share one device for a copy-free WgpuTexture handoff.
#[cfg(any(feature = "vello-overlay", feature = "wgpu-sink"))]
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
// uridecodebin front door: URI-scheme handlers for Registry::build_uridecodebin
// (file:// -> Mp4Src, udp:// -> UdpSrc, rtsp:// -> RtspSrc, v4l2:// -> V4l2Src),
// each gated to its source's feature.
#[cfg(feature = "std")]
pub mod uridecodebin;
// A Registry pre-populated with the standard elements for parse_launch /
// gst-inspect (M107). std (the Registry is std).
#[cfg(feature = "std")]
pub mod registry;
// Annex-B NAL splitting shared by rtppay (RTP) and h264util (WebCodecs).
mod annexb;
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
// Container content sniffing (no_std): guess a ByteStreamEncoding from a header.
pub mod typefind;
pub mod videotestsrc;

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
pub mod mp4sink;
#[cfg(feature = "std")]
pub mod mp4src;
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

// UDP egress sink (M47): drives the M46 RtpH264Packetizer and sends RTP over a
// tokio UdpSocket, the send-side inverse of RtspSrc's receive path.
#[cfg(feature = "udp-egress")]
pub mod udpsink;

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

// VAAPI H.264 decode via cros-codecs is Linux-only. The dependency is
// target-gated; enabling the feature on other platforms is a no-op.
#[cfg(all(target_os = "linux", feature = "vaapi"))]
pub mod vaapidec;

// ffmpeg/libavcodec H.264 decode is Linux-only here (the ffmpeg-next dep is
// target-gated). Currently software decode; VAAPI hwaccel is a follow-up that
// stays inside this module and does not change the public AsyncElement shape.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
pub mod ffmpegdec;

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

// Wayland display sink (NV12 -> XRGB8888 via wl_shm). Linux-only;
// desktop-dev convenience sink — see module docs.
#[cfg(all(target_os = "linux", feature = "wayland-sink"))]
pub mod waylandsink;

// CUDA device-memory consumers (C3 Phase 3). `CudaDownload` copies a
// `MemoryDomain::Cuda` NV12 frame back to system memory so a `NvdecCuda`
// stream reaches the CPU sinks. Hand-rolled libcuda FFI; Linux + NVIDIA only.
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod cuda;

// CUDA-GL zero-copy-ish display sink: keeps decoded NV12 on the GPU and
// presents it via CUDA-GL interop on a Wayland EGL surface. Linux + NVIDIA.
#[cfg(all(target_os = "linux", feature = "cuda-gl"))]
pub mod cudaglsink;

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

// H.264 Annex-B helpers for WebCodecsDecode (NAL split, keyframe, codec
// string). Pure; compiled under cfg(test) so the logic is host-testable.
#[cfg(any(test, all(target_arch = "wasm32", feature = "web-codecs")))]
mod h264util;

// Embassy RTOS clock backend (M43): the embedded deployment-profile clock over
// embassy-time, the no_std analog of WallClock / WasmClock.
#[cfg(feature = "embassy")]
pub mod embassyclock;

// Embassy zero-alloc inter-task packet link (M45): PacketChannel over
// embassy-sync, the §6.2 stack-channel backend.
#[cfg(feature = "embassy-link")]
pub mod embassylink;

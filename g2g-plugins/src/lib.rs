//! Standard source / sink / transform elements for `glass2glass`.
//!
//! Per the spec (§2), this crate is `no_std + alloc` at baseline. Network
//! and OS-coupled elements (RTSP source via `retina`, V4L2, wgpu sinks)
//! live behind cargo features that imply `std`.
//!
//! OS-, GPU-, and device-coupled elements are experimental (Tier 3 in
//! `STABILITY.md`): they compile, and some have host tests, but their runtime
//! is not a CI promise. `g2g-inspect` prints `Stability   experimental` on
//! those factories. The pure-Rust set is provisional, not frozen.

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]

extern crate alloc;

// Drive a leaf future to completion by spinning. Only for the `#[cfg(fuzzing)]`
// element shims: they parse buffered bytes into a synchronous sink and never
// await real IO, so a no-op waker never leaves them pending.
#[cfg(fuzzing)]
pub(crate) fn fuzz_block_on<F: core::future::Future>(f: F) -> F::Output {
    use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(core::ptr::null(), &VT)
    }
    fn noop(_: *const ()) {}
    static VT: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VT)) };
    let mut cx = Context::from_waker(&waker);
    let mut f = f;
    // SAFETY: `f` is owned here and never moved again before it is dropped.
    let mut f = unsafe { core::pin::Pin::new_unchecked(&mut f) };
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
    }
}

pub mod aacparse;
// IMA ADPCM codec elements (M1073), wrapping the heap-free `g2g-mcu` block math.
pub mod ac3parse;
pub mod adpcm;
// Native FLAC stream parser (M774): frame-splits a `.flac` byte stream (the
// re-framing `h264parse` analog for audio) and refines caps from STREAMINFO.
pub mod flacparse;
// G.711 (mu-law / A-law) codec elements (M1073), wrapping the heap-free
// `g2g-mcu` companding math.
pub mod g711;

pub mod appsink;
pub mod appsrc;
pub mod audioamplify;
pub mod audiobuffersplit;
pub mod audiochannelmix;
pub mod audiochebband;
pub mod audiocheblimit;
pub mod audioconvert;
pub mod audiodynamic;
pub mod audioecho;
pub mod audiofirfilter;
// Filter kernels the audiofx transforms share (windowed-sinc FIR, Chebyshev
// IIR) plus their common PCM boundary.
pub mod audiofx;
pub mod audioiirfilter;
pub mod audioinvert;
pub mod audiokaraoke;
pub mod audiomixer;
pub mod audiomixmatrix;
pub mod audiopanorama;
pub mod audiorate;
pub mod audioresample;
pub mod audiotestsrc;
// `tonegeneratesrc`: a sine at `freq` / `volume`.
pub mod audiowsincband;
pub mod audiowsinclimit;
pub mod av1parse;
pub mod avoffset;
pub mod tonegeneratesrc;
// `dtmfsrc` / `dtmfdetect`: ITU-T Q.23 tones and a Goertzel detector.
pub mod dtmf;
// Byte corrupter: overwrites bytes at random, to prove a parser survives them.
pub mod breakmydata;
pub mod capsfilter;
// Caps rewriter: overwrites the caps travelling with a stream, data untouched.
pub mod capssetter;
// What a capture source's pixel format means on a link, shared by the capture
// sources so their fourcc tables map to one set of caps.
pub mod capturepixelformat;
// Buffer digests, for checking a codec change is bit-exact.
pub mod checksumsink;
// `debugspy`: passthrough that hashes each buffer.
pub mod debugspy;
// Byte-stream re-chunker: step-aligned random buffer sizes.
pub mod chopmydata;
pub mod concat;
pub mod cutter;
// Source reading the payload carried inside a `data:` URI.
pub mod dataurisrc;
pub mod deinterleave;
pub mod equalizer;
// Pass-through that turns a failure from downstream into a dropped buffer.
pub mod errorignore;
// The media-typed fake sinks: raw video only, raw audio only.
pub mod fakemediasink;
pub mod fakesink;
pub mod fakesrc;
// Decoded-GOP reverser (M897): the presentation half of reverse playback.
pub mod gopreverse;
pub mod h264parse;
pub mod h265parse;
pub mod identity;
pub mod imagefreeze;
pub mod inputselector;
pub mod interleave;
pub mod level;
pub mod mux;
pub mod nalparse;
// Shared access-unit framing for the start-code elementary streams that are not
// NAL streams (MPEG-1/2 video, MPEG-4 Part 2, VC-1).
pub mod startcodeparse;
// Legacy video parsers over that core.
pub mod mpeg4videoparse;
pub mod mpegvideoparse;
#[cfg(feature = "offload")]
pub mod offload;
pub mod opusparse;
pub mod outputselector;
pub mod poc;
pub mod progressreport;
pub mod vc1parse;
// Headerless raw framers: a `.pcm` / `.yuv` dump cut into buffers from the
// format and geometry its properties declare.
pub mod rawaudioparse;
pub mod rawvideoparse;
pub mod scaletempo;
// Split-file source: the files matching a pattern read as one byte stream.
// Reads a directory, so std.
#[cfg(feature = "std")]
pub mod splitfilesrc;
// Still-image framing: a JPEG / PNG byte stream cut into whole images.
pub mod stillparse;
// Deterministic pseudo-randomness shared by the test / debug elements.
mod random;
// Byte re-chunking shared by the random / step-aligned buffer-size transforms.
mod rechunk;
// Byte-stream re-chunker: random buffer sizes, for shaking out parsers that
// depend on where their input is cut.
pub mod removesilence;
pub mod rndbuffersize;
pub mod spectrum;
pub mod speed;
pub mod stereo;
pub mod streamdemux;
// Bus tag injector: posts a hand-written tag list for a stream that carries none.
pub mod taginject;
// The `tags=` property, gst taglist syntax, shared by taginject and the tag writers.
pub mod tagproperty;
pub mod tsmuxn;
// Closable pass-through: drops data while `drop=true`, for muting one tee branch.
pub mod valve;
pub mod volume;
pub mod vp8parse;
pub mod vp9parse;
// Shared integer source-over blend used by the compositor and CPU overlays.
mod mathf;
mod paint;
mod xmlutil;
// Shared gst `num-buffers` property conversion used by the source elements.
mod numbuffers;
// Shared `cuda-device-id` property used by the elements that open a CUDA context.
#[cfg(all(target_os = "linux", any(feature = "cuda", feature = "ffmpeg")))]
mod cudadeviceid;

// Software RGBA8 compositor (fan-in pixel mixer): PiP / grids / overlays.
pub mod compositor;
// Conformance batteries (M614): exercise real elements to derive their maturity
// records, so `g2g-inspect --maturity` reports validation observed, not claimed.
pub mod conformance;
// Analytics overlay (M101): draws AnalyticsMeta detection boxes onto RGBA8.
// Needs the per-frame metadata graph, so it is gated on `analytics`.
#[cfg(feature = "analytics")]
pub mod analyticsoverlay;
// Shared wgpu device context for the GPU elements (M103): a producer and a sink
// must share one device for a copy-free WgpuTexture handoff.
#[cfg(any(
    feature = "vello-overlay",
    feature = "wgpu-sink",
    feature = "cuda-wgpu",
    feature = "vulkan-video",
    feature = "mediacodec-wgpu"
))]
pub mod gpu;
// GPU / compute device discovery (M939): wgpu adapters, CUDA devices, VAAPI nodes.
#[cfg(any(feature = "wgpu-sink", feature = "cuda", feature = "vaapi"))]
pub mod gpudevice;
// Re-export wgpu so a downstream consumer (a viewer wiring g2g's GPU-texture
// decode into its renderer) can name `wgpu::Texture` / build on a shared device
// with the EXACT wgpu version g2g's textures are bound to. A version mismatch
// would make the handle types incompatible.
#[cfg(any(
    feature = "vello-overlay",
    feature = "wgpu-sink",
    feature = "cuda-wgpu",
    feature = "vulkan-video"
))]
pub use wgpu;
// Vello GPU companion to analyticsoverlay (M102): renders boxes with the Vello
// 2D renderer into a wgpu texture (MemoryDomain::WgpuTexture, kept on GPU).
#[cfg(feature = "vello-overlay")]
pub mod vellooverlay;
// GPU presentation sink (M103): presents MemoryDomain::WgpuTexture frames by
// blitting onto an offscreen target or a caller-provided wgpu::Surface.
pub mod alpha;
pub mod aspectratiocrop;
pub mod chromahold;
#[cfg(feature = "std")]
pub mod clockoverlay;
pub mod coloreffects;
pub mod deinterlace;
pub mod gamma;
pub mod gaussianblur;
pub mod smooth;
pub mod tensorconvert;
pub mod timeoverlay;
pub mod videobalance;
pub mod videobox;
pub mod videoconvert;
pub mod videoconvertscale;
pub mod videocrop;
pub mod videodiff;
// `videoanalyse`: luma average / variance of each frame.
pub mod videoanalyse;
// `scenechange`: SAD-based shot-change detector.
pub mod scenechange;
pub mod videoflip;
// Shared negotiation and frame loop behind the CPU video-effect transforms.
pub(crate) mod videofx;
pub mod videomedian;
pub mod videorate;
pub mod videoscale;
pub mod wavenc;
pub mod wavparse;
pub mod zebrastripe;
// AIFF / AU PCM containers, the APEv2 tag reader, and the wire conversion and
// mux loop they share.
pub mod aiff;
pub mod apedemux;
pub mod au;
mod audiocontainer;
pub mod gaudieffects;
// `hsvfilter` / `hsvdetector`: HSV transform and colour-box detector.
pub mod hsv;
mod pcmendian;
// `roundedcorners`: transparent corner arcs.
pub mod roundedcorners;
#[cfg(test)]
mod testutil;
// MIME multipart (`multipart/x-mixed-replace`) reader + writer: the MJPEG-over-
// HTTP transport an IP camera pushes.
pub mod multipart;
// wgpu compute companion to `compositor` (M853): RGBA8 fan-in blending in one
// compute dispatch, System or MemoryDomain::WgpuTexture out. Shares the wgpu
// GPU feature with the sink that consumes its textures.
#[cfg(feature = "wgpu-sink")]
pub mod wgpucompositor;
#[cfg(feature = "wgpu-sink")]
pub mod wgpusink;
// Windowed wgpu display sink (`wgpusink` on a launch line): owns an
// xdg_toplevel, builds the wgpu::Surface over it, and drives WgpuSink on it.
#[cfg(all(target_os = "linux", feature = "wgpu-present"))]
pub mod wgpupresent;
// YUV4MPEG2 (`.y4m`) reader + writer: raw planar YUV in a file, WAV's video
// counterpart.
pub mod y4m;
// Subtitle cue parsing (SRT / WebVTT) and the embedded bitmap font, both no_std,
// feeding the `textoverlay` element below.
pub mod bitmapfont;
pub mod subparse;
pub mod textoverlay;
// The writers that invert `subparse`: timed cues out as a SubRip / WebVTT
// document, over the cue bookkeeping both share.
pub mod srtenc;
pub mod subenc;
pub mod webvttenc;
// Shaping / bidi / system-font discovery behind the overlay's horizontal path.
#[cfg(feature = "text-shaping")]
pub mod textshape;
// Shared H.264/H.265 SEI message walk + the HDR10 static-metadata payloads.
pub mod sei;
// CEA-608/708 closed captions carried in-band in H.264/H.265 SEI (no_std).
pub mod cea;
// MISB ST 0601 KLV telemetry (STANAG 4609): codec + klvdecode element (no_std).
pub mod klv;
// MISB ST 0903 VMTI moving-target reports, nested in ST 0601 tag 74 (no_std).
pub mod vmti;
// VobSub (DVD subpicture) bitmap subtitles: the SPU / .idx codec (no_std) plus
// the `vobsubdec` element that renders cues to RGBA canvases.
pub mod vobsub;
pub mod vobsubdec;
// Sidecar `.idx` / `.sub` pair as a VobSub stream (needs the filesystem).
#[cfg(feature = "std")]
pub mod vobsubsrc;
// DVB subtitles (ETSI EN 300 743): the segment-stream codec (no_std) plus the
// `dvbsubdec` element that renders display sets to RGBA canvases.
pub mod dvbsub;
pub mod dvbsubdec;
// Blu-ray PGS / HDMV subtitles: the segment-stream codec (no_std) plus the
// `pgsdec` element that renders display sets to RGBA canvases.
pub mod pgs;
pub mod pgsdec;
// The visible end of those three decoders: blends their cue canvases onto video.
pub mod subpictureoverlay;
// EBU teletext (ETSI EN 300 706): the page-assembly codec (no_std) plus the
// `teletextdec` element that turns a subtitle page into plain-text cues.
pub mod teletext;
pub mod teletextdec;
// Cursor-on-Target bridge (M811): the ST 0601 -> CoT XML event builder (no_std)
// plus the `cotsink` TAK egress element, which needs `udp-egress`.
pub mod cotsink;
// Closed-caption extraction element: compressed video in, timed text cues out.
pub mod ccextract;
// Closed-caption insertion element: compressed video + cues in, SEI'd video out.
pub mod ccinsert;
// Closed-caption transport converter: cc_data / CDP / S334-1A / raw CEA-608.
pub mod ccconverter;
// Closed-caption combiner: video + captions in, video carrying caption meta out.
#[cfg(feature = "metadata")]
pub mod cccombiner;
// MISB ST 0604 MISP time stamps in H.264 / H.265 SEI (STANAG 4609): codec +
// misptimeinsert / misptimeextract elements (no_std).
pub mod misptime;
// Shared pixel-format helpers: packed-RGBA layout (videobalance, alpha) and the
// planar plane / frame sizing the format-agnostic filters need (deinterlace).
mod pixel;
// Where a capture driver's / decoder's padded rows sit, shared by the producers
// that either declare them (`PlaneLayout`) or pack them tight.
#[cfg(any(feature = "metadata", feature = "v4l2", feature = "pipewire"))]
mod paddedrows;
// Sans-IO RFC 4566 SDP: the shared media-section scanner plus the RTP/AVP
// mapping from a media description to Caps (payload type, codec, clock rate,
// fmtp parameter-set geometry), so an RTP receiver configures from the SDP a
// sender publishes instead of a declared hint. no_std+alloc.
pub mod sdp;
// Sans-IO H.264 RTP packetizer (RFC 3550 + 6184), the live-egress foundation.
pub mod rtppay;
// Sans-IO H.264 RTP depayloader, the receive-side inverse of rtppay.
pub mod rtpdepay;
// Sans-IO KLV metadata RTP payloader / depayloader (RFC 6597, SMPTE ST 336).
pub mod rtpklv;

// ST 2110-30 PCM audio over RTP (M595): sans-IO packetizer / depacketizer for
// uncompressed L16 / L24, RTP timestamps from the PTP media clock. no_std+alloc.
pub mod st2110audio;

// ST 2110-40 ancillary data over RTP (M596): SMPTE ST 291 ANC packets (captions,
// timecode) per RFC 8331, sans-IO packetizer / depacketizer with 10-bit-word
// parity + checksum. no_std+alloc.
pub mod st2110anc;

// ST 2110-20 uncompressed video over RTP (M599, RFC 4175): sans-IO packetizer /
// depacketizer slicing a frame into SRD line runs, RGBA 8-bit + YCbCr-4:2:2 8/10-bit.
// no_std+alloc.
pub mod st2110video;

// ST 2110-22 JPEG XS over RTP (M604, RFC 9134): sans-IO packetizer / depacketizer
// slicing a JPEG XS codestream into codestream-mode packets, RTP timestamps from
// the PTP media clock. no_std+alloc.
pub mod st2110jxs;

// ST 2110-7 seamless protection (M608): sans-IO receive-side de-duplication merging
// redundant RTP streams by sequence number (first arrival wins), essence-agnostic.
// no_std+alloc.
pub mod st2110dup;

// ST 2110-21 sender pacing (M609): sans-IO traffic-shaping schedule spreading a
// frame's RTP packets across the frame period (linear / gapped), with a conformance
// check. A sink realizes it by sleeping to each packet's offset. no_std+alloc.
pub mod st2110pacing;

// ST 2110 SDP (M601, RFC 4566 + SMPTE ST 2110-10/-20/-30/-40): sans-IO generator /
// parser for the out-of-band stream description (essence, PT, address/port, PTP
// reference clock) a receiver configures from. no_std+alloc.
pub mod st2110sdp;

// ST 2110-30 audio network elements (M597): a sink (AsyncElement) and source
// (SourceLoop) wrapping the sans-IO -30 core over UDP, RTP timestamps off the
// PTP media clock. std (UdpSocket), behind the `st2110` feature.
#[cfg(feature = "st2110")]
pub mod st2110audiortp;

// ST 2110-40 caption network elements (M598): a sink (AsyncElement, compressed
// video in -> -40 UDP) and source (SourceLoop, -40 UDP -> text cues) wrapping the
// sans-IO -40 core, bridged to the CEA-608/708 stack via CDPs, RTP timestamps off
// the PTP media clock. std (UdpSocket), behind the `st2110` feature.
#[cfg(feature = "st2110")]
pub mod st2110ancrtp;

// ST 2110-20 video network elements (M599): a sink (AsyncElement, packed raw video
// in -> RFC 4175 UDP) and source (SourceLoop, UDP -> raw video frames) wrapping the
// sans-IO -20 core, RTP timestamps off the PTP media clock. std (UdpSocket), behind
// the `st2110` feature.
#[cfg(feature = "st2110")]
pub mod st2110videortp;

// ST 2110-22 JPEG XS network elements (M604): a sink (AsyncElement, JPEG XS
// codestream in -> RFC 9134 UDP) and source (SourceLoop, UDP -> codestream frames)
// wrapping the sans-IO -22 core, RTP timestamps off the PTP media clock. std
// (UdpSocket), behind the `st2110` feature.
#[cfg(feature = "st2110")]
pub mod st2110jxsrtp;
// Shared RTP H.264 receive loop (jitter + RTCP RR/NACK + FEC/RTX + depayload):
// the receive path both UdpSrc (raw RTP) and RtspServerSrc (RTSP RECORD) ride.
// Gated to the ingest features that supply tokio (its UDP transport).
#[cfg(any(feature = "udp-ingress", feature = "rtsp-server"))]
pub mod rtprecv;
// Sans-IO RTP jitter buffer (reorder / loss / dup detection) between a socket
// and the depayloader, the receive-side network-resilience stage.
pub mod rtpjitter;
// Sans-IO RTCP (RFC 3550 SR/RR/BYE + RFC 4585 Generic NACK) and RFC 3550
// reception statistics: the RTP control / feedback channel.
pub mod rtcp;
// Sans-IO RFC 7714 AES-GCM protection for RTP and RTCP. The feature keeps
// cryptographic dependencies out of builds that use plain RTP only.
#[cfg(feature = "srtp")]
pub mod srtp;
#[cfg(feature = "srtp")]
pub mod srtpdec;
#[cfg(feature = "srtp")]
pub mod srtpenc;
// DTLS-SRTP (RFC 5764): the handshake that keys the RFC 7714 layer, and the
// element pair that runs it over the media socket.
#[cfg(feature = "dtls-srtp")]
pub mod dtlssrtp;
#[cfg(feature = "dtls-srtp")]
pub mod dtlssrtpdec;
#[cfg(feature = "dtls-srtp")]
pub mod dtlssrtpenc;
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
// Sans-IO SMPTE ST 2022-1 (Pro-MPEG COP3) FEC: the 2-D row/column XOR repair
// streams professional MPEG-TS-over-RTP contribution links expect.
pub mod st2022fec;
// uridecodebin front door: URI-scheme handlers for Registry::build_uridecodebin
// (file:// -> Mp4Src, udp:// -> UdpSrc, rtsp:// -> RtspSrc, v4l2:// -> V4l2Src),
// each gated to its source's feature.
#[cfg(feature = "std")]
pub mod uridecodebin;
// A Registry pre-populated with the standard elements for parse_launch /
// gst-inspect (M107). std (the Registry is std).
#[cfg(feature = "std")]
pub mod registry;
// The device-provider analog of `registry` (M939): the standard
// `DeviceMonitor` assembly.
#[cfg(feature = "std")]
pub mod devicemon;
// GStreamer porting helpers: gst->g2g element map + launch linter (M200). std
// (uses the Registry + parse_launch).
#[cfg(feature = "std")]
pub mod gst_compat;
// Declarative graph format (M578): build a `Graph` from a JSON / YAML document,
// the structured sibling of the `gst-launch` text parser. Behind `declarative`
// (pulls serde + serde_json); `declarative-yaml` adds the YAML front-end.
#[cfg(feature = "declarative")]
pub mod declarative;
// Embedded Rhai scripting (M579/M580): a script that BUILDS a graph
// (`script::build_from_script`), and the `scriptelement` runtime transform whose
// per-frame logic is a Rhai `process(frame)`. Behind `script-rhai` (pulls rhai).
#[cfg(feature = "script-rhai")]
pub mod script;
// Dynamic (`dlopen`) plugin loader for third-party `cdylib` plugins built with
// the `g2g-plugin` SDK (M201). Behind `plugin-loader` (pulls `libloading`); the
// loaded elements register into a `Registry` the parser then uses by name.
#[cfg(feature = "plugin-loader")]
pub mod plugin_loader;
// Tokio thread-per-arm executor for the opt-in multicore graph runner
// (`run_graph_threaded`). Needs std (tokio) + multi-thread (Send graph).
#[cfg(all(feature = "std", feature = "multi-thread"))]
mod graphthreads;
#[cfg(all(feature = "std", feature = "multi-thread"))]
pub use graphthreads::TokioThreadSpawner;
// Annex-B NAL splitting shared by rtppay (RTP) and h264util (WebCodecs).
mod annexb;
// RIFF chunk walking shared by the WAVE and AVI elements.
mod riff;
// AVI container parsing / writing (M1071), behind `avidemux` and `avimux`.
mod avi;
// AVI demuxer elements (M1071): one `ByteStream{Avi}` in, its streams out.
pub mod avidemux;
// AVI muxer elements (M1071): one video plus an optional audio stream in, a
// `ByteStream{Avi}` out.
pub mod avimux;
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
// MPEG program stream demuxer (no_std): Caps::ByteStream{MpegPs} -> one
// elementary stream, the `.mpg` / `.vob` sibling of tsdemux.
pub mod mpeg2video;
pub mod psdemux;
// Frame headers of the self-syncing audio bitstreams (AC-3, MPEG audio), shared
// by the audio decoder's frame splitting, psdemux's frame realignment and
// mpegaudioparse.
mod audioframe;
// ID3v1 / ID3v2 tag parsing and writing (no_std), shared by id3demux,
// mpegaudioparse and id3v2mux.
mod id3;
// ID3 tag stripper element (no_std): a tagged byte stream in, the payload out.
pub mod id3demux;
// ID3v2 tag writer element (no_std): the same byte stream with its leading tag
// rewritten from the `tags` property.
pub mod id3v2mux;
// APEv2 tag writer element (no_std): the tag block appended at the tail.
pub mod apev2mux;
// Xing/Info VBR header writer (no_std): the seek header a VBR `.mp3` needs.
pub mod xingmux;
// VorbisComment parsing and writing (no_std), shared by the Ogg demuxer and
// muxer and the vorbistag / flactag writers.
mod vorbiscomment;
// Vorbis comment-header rewriter element (no_std).
pub mod vorbistag;
// Native FLAC VORBIS_COMMENT block rewriter element (no_std).
pub mod flactag;
// MPEG audio parser element (no_std): an `.mp3` / `.mp2` byte stream -> one
// MPEG audio frame per buffer, the framing ffmpegaudiodec takes.
pub mod mpegaudioparse;
// Non-blocking link to a blocking audio device worker thread, shared by the
// ALSA and PulseAudio sinks.
#[cfg(any(feature = "alsa-sink", feature = "pulse-sink"))]
mod audioworker;
// Matroska / WebM demuxer parsing core (no_std): EBML -> Tracks + Cluster frames.
pub mod matroska;
// Matroska / WebM demuxer element (no_std): Caps::ByteStream{Matroska} -> one
// selected elementary stream, wrapping the matroska parser.
pub mod mkvdemux;
// IVF demuxer element (no_std): Caps::ByteStream{Ivf} -> the VP8 / VP9 / AV1
// video elementary stream, the raw libvpx / libaom conformance-vector container.
pub mod ivfdemux;
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
// Ogg muxer element (no_std): one Opus / Vorbis / FLAC stream ->
// Caps::ByteStream{Ogg}, the inverse of oggdemux.
pub mod oggmux;
// Multi-stream Ogg muxer element (no_std): N audio streams -> one grouped
// Caps::ByteStream{Ogg}, the fan-in analog of oggmux.
pub mod oggmuxn;
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
// Wall-clock pacing transform: holds each buffer until its PTS is due, turning
// an as-fast-as-possible upstream into a live-paced stream.
#[cfg(feature = "std")]
pub mod clocksync;
// Stall detector: fails the run when no data crosses it within `timeout`. std
// because the deadline is wall time and the timer is a tokio task.
#[cfg(feature = "std")]
pub mod filesink;
#[cfg(feature = "std")]
pub mod filesrc;
#[cfg(feature = "std")]
pub mod watchdog;
// Frame-rate reporter around a child display sink. std (it reads a clock and
// builds its child through the Registry).
#[cfg(feature = "std")]
pub mod fpsdisplaysink;
// Raw file-descriptor source and sink: unix only, a `RawFd` is what the `fd`
// property names.
#[cfg(all(feature = "std", unix))]
pub mod fd;
// Spill-to-storage byte buffer (M861): turns a pushed, non-seekable byte stream
// into a seekable one by absorbing it into a temp file.
#[cfg(feature = "std")]
pub mod downloadbuffer;
// Record / replay: dump the packet stream to a file and play it back, for
// deterministic repro of bugs that need a live source.
#[cfg(feature = "std")]
pub mod multifilesink;
#[cfg(feature = "std")]
pub mod multifilesrc;
#[cfg(feature = "std")]
pub mod record;
#[cfg(feature = "std")]
pub mod splitmuxsink;
// HLS packager: cuts a muxed byte stream into segment files plus a rolling
// m3u8 media playlist (M896).
#[cfg(feature = "std")]
pub mod hlssink;
// Subtitle/text file source: a .srt/.vtt/.ssa/.ttml file as a Text stream.
#[cfg(feature = "std")]
mod audio;
#[cfg(feature = "std")]
pub mod fmp4mux;
#[cfg(feature = "std")]
pub mod gaplesssrc;
#[cfg(feature = "std")]
pub mod mp4audiosink;
#[cfg(feature = "std")]
pub mod mp4audiosrc;
#[cfg(feature = "std")]
mod mp4box;
#[cfg(feature = "std")]
pub mod mp4demuxn;
#[cfg(feature = "std")]
pub mod mp4mux;
#[cfg(feature = "std")]
pub mod mp4muxn;
#[cfg(feature = "std")]
pub mod mp4src;
#[cfg(feature = "std")]
pub mod subtitlesrc;
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
// Per-publisher ingest session machinery shared by the two ingest elements.
#[cfg(feature = "rtsp-server")]
mod rtspingest;
#[cfg(feature = "rtsp-server")]
pub mod rtspserversink;
#[cfg(feature = "rtsp-server")]
pub mod rtspserversrc;
#[cfg(feature = "rtsp-server")]
pub mod rtspserversrcn;

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
pub mod webrtc_simulcast;
#[cfg(feature = "webrtc")]
mod webrtc_util;
#[cfg(all(feature = "webrtc", fuzzing))]
pub use turn::fuzz_parse as turn_fuzz_parse;
#[cfg(all(feature = "webrtc", fuzzing))]
pub use webrtc_util::fuzz_parse as stun_fuzz_parse;
#[cfg(feature = "webrtc")]
pub mod webrtcdata;
#[cfg(feature = "webrtc")]
pub mod webrtcduplex;
#[cfg(feature = "webrtc")]
pub mod webrtcsession;
#[cfg(feature = "webrtc")]
pub mod webrtcsink;
#[cfg(feature = "webrtc")]
pub mod webrtcwhepsession;
#[cfg(feature = "webrtc")]
pub mod webrtcwhepsrc;

// Native LiveKit publisher + subscriber (T4): WebSocket + protobuf signaller
// layered over the str0m engine. `livekit_signal` is the transport/protocol seam
// (JWT + hand-rolled protobuf), `livekitsink` the publish element, `livekitsrc`
// the room subscriber (answers the server-offered subscriber PC). Gated behind
// `webrtc-livekit` (implies `webrtc`, adds the WebSocket client + JWT crypto).
#[cfg(feature = "webrtc-livekit")]
pub mod livekit_signal;
#[cfg(feature = "webrtc-livekit")]
pub mod livekitduplex;
#[cfg(feature = "webrtc-livekit")]
pub mod livekitsink;
#[cfg(feature = "webrtc-livekit")]
pub mod livekitsrc;

// UDP ingress source (M91): receives RTP on a tokio UdpSocket and depayloads
// H.264 (rtpdepay) into Annex-B access units, the receive-side inverse of
// UdpSink. Caps come from a published SDP or the stream's SPS; see module docs.
#[cfg(feature = "udp-ingress")]
pub mod udpsrc;

// Plain TCP byte-stream elements (M1068): TcpServerSrc / TcpClientSrc read a
// socket into DataFrame chunks the way FileSrc reads a file, TcpServerSink /
// TcpClientSink write those bytes back out. Raw bytes, no framing of their own.
#[cfg(feature = "tcp")]
pub mod tcp;

// Shared-memory IPC pair (M1081): ShmSink serves frames through a POSIX shm
// area announced over a unix control socket, ShmSrc maps that area and copies
// each announced block out. The wire is GStreamer's shmpipe protocol, so either
// end can be a gst shmsink / shmsrc.
#[cfg(all(unix, feature = "shm"))]
pub mod shm;
#[cfg(all(unix, feature = "shm"))]
pub mod shmpipe;

// Distributed-graph transport pair (M551): RemoteSink (TCP client) serializes
// the PipelinePacket stream (g2g-core wire codec) and RemoteSrc (TCP listener)
// reconstructs it, so any graph edge can be cut and the downstream subgraph run
// across a process / machine boundary. Behind the `remote` feature (std + tokio).
// JSON tooling shared by `g2g-inspect --json` and the `g2g-mcp` server: registry
// dump, launch-line validation, bounded run.
#[cfg(feature = "tooling-json")]
pub mod toolingjson;

// Edge content preview (observe feature): sampled packet -> JSON thumbnail /
// waveform / hexdump for the dashboard edge tap.
#[cfg(feature = "observe")]
pub mod preview;

// Live pipeline dashboard transport (observe feature): serves Observer telemetry
// + bus events over one WS/HTTP port to the static dashboard page. Used by
// `g2g-launch --observe`.
#[cfg(feature = "observe")]
pub mod dashboard;

// In-terminal live telemetry (tui feature): the same Observer tap drawn with
// ratatui. Used by `g2g-launch --tui`.
#[cfg(feature = "tui")]
pub mod tui;

#[cfg(feature = "remote")]
pub mod remotesink;
#[cfg(feature = "remote")]
pub mod remotesrc;

// Shared helper for the distributed-graph transports (map a g2g-core wire codec
// error to the pipeline error type); used by the TCP pair, the native WebSocket
// pair, and the browser WsWireSink alike. The `web` arm is wasm32-gated to match
// where WsWireSink (its only web-side user) is compiled, so a native `web` build
// does not leave map_wire unused.
#[cfg(any(
    feature = "remote",
    feature = "remote-ws",
    feature = "webtransport",
    all(target_arch = "wasm32", feature = "web")
))]
mod remotewire;

// Shared receive-side core for the distributed-graph source elements (TCP
// RemoteSrc + WebSocket RemoteWsSrc + WebTransport RemoteWtSrc), parameterized
// over the transport.
#[cfg(any(feature = "remote", feature = "remote-ws", feature = "webtransport"))]
pub mod remotesource;

// Shared send-side core for the distributed-graph sink elements (TCP RemoteSink
// + WebSocket RemoteWsSink + WebTransport RemoteWtSink), parameterized over the
// transport.
#[cfg(any(feature = "remote", feature = "remote-ws", feature = "webtransport"))]
pub mod remoteclient;

// Shared core for the distributed-graph remote-transform elements (WebSocket
// RemoteWsTransform + WebTransport RemoteWtTransform): the FIFO frame-out /
// processed-frame-back round trip, parameterized over the transport.
#[cfg(any(feature = "remote-ws", feature = "webtransport"))]
pub mod remotetransform;

// Shared `host`/`address` + `port` property get/set for the network source/sink
// elements (SocketAddr-backed). Collapses the identical string->IpAddr and
// bounds-checked-uint->port dispatch that each of these elements would otherwise
// copy.
#[cfg(any(
    feature = "remote",
    feature = "remote-ws",
    feature = "webtransport",
    feature = "rtmp",
    feature = "rtsp-server",
    feature = "srt",
    feature = "udp-ingress",
    feature = "udp-egress",
))]
mod netprop;

// Shared byte-stream carrier pieces (M1068, M1079): the frame shape a received
// chunk takes, the MPEG-TS packet geometry a datagram is cut on, and the
// container list a raw wire sink advertises.
#[cfg(any(
    feature = "tcp",
    feature = "srt",
    feature = "udp-ingress",
    feature = "udp-egress",
))]
pub mod bytestream;

// WebSocket sibling of the M551 pair (M554): RemoteWsSink (WebSocket client) +
// RemoteWsSrc (WebSocket server) carry the same wire-codec PipelinePacket stream,
// one packet per binary WebSocket message, so a browser peer (which speaks only
// WebSocket) can join the same distributed primitive. Behind `remote-ws`.
#[cfg(feature = "remote-ws")]
mod remotewsio;
#[cfg(feature = "remote-ws")]
pub mod remotewssink;
#[cfg(feature = "remote-ws")]
pub mod remotewssrc;
// RemoteWsTransform (M555): a media-agnostic remote transform. Ships each input
// packet to a remote stage over one WebSocket and emits the processed packet it
// gets back, so a middle stage (e.g. inference) runs on another machine. The
// bidirectional, round-trip generalization of the browser WebRemoteDetect shim.
#[cfg(feature = "remote-ws")]
pub mod remotewstransform;

// WebTransport sibling of the same family (M901): RemoteWtSink (client) +
// RemoteWtSrc (server) + RemoteWtTransform carry the identical wire-codec stream
// over one reliable bidirectional QUIC stream (HTTP/3 CONNECT), so a peer that
// speaks WebTransport (a browser, or a native QUIC client) joins the same
// distributed primitive. Behind `webtransport`.
#[cfg(feature = "webtransport")]
pub mod remotewtio;
#[cfg(feature = "webtransport")]
pub mod remotewtsink;
#[cfg(feature = "webtransport")]
pub mod remotewtsrc;
#[cfg(feature = "webtransport")]
pub mod remotewttransform;

// Native IETF MoQ Transport draft-16 (M902 / M903): the wire codec plus the
// session driver over the M901 carrier, the `moqtsink` publisher that maps a
// fragmented-MP4 stream onto MOQT groups and objects, and the `moqtsrc`
// subscriber that puts the objects back in order. Behind `moqt`.
#[cfg(feature = "moqt")]
pub mod moqt;
#[cfg(feature = "moqt")]
pub mod moqtsessionsrc;
#[cfg(feature = "moqt")]
pub mod moqtsink;
#[cfg(feature = "moqt")]
pub mod moqtsrc;

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
// macOS Metal present sink (M736): NV12 (System bytes or the M735 zero-copy
// CvPixelBuffer domain) rendered to a CAMetalLayer drawable.
#[cfg(all(target_os = "macos", feature = "metal-sink"))]
pub mod metalvideosink;
// macOS Core Audio render + capture via AudioToolbox AudioQueue (M737).
#[cfg(all(target_os = "macos", feature = "coreaudio"))]
pub mod coreaudio;

// Core Audio device discovery: the HAL's input / output devices, for the
// device monitor.
#[cfg(all(target_os = "macos", feature = "coreaudio"))]
pub mod coreaudiodevice;
// Shared CVPixelBuffer helpers (NV12 pack + zero-copy keep-alive + the capture
// delegate handoff) for the macOS video elements.
#[cfg(all(
    target_os = "macos",
    any(
        feature = "vtdecode",
        feature = "avfoundation",
        feature = "screencapture"
    )
))]
pub(crate) mod cvnv12;
// AVFoundation camera + mic capture (M738).
#[cfg(all(target_os = "macos", feature = "avfoundation"))]
pub mod avf;

// AVFoundation camera discovery: the cameras a discovery session lists, for
// the device monitor.
#[cfg(all(target_os = "macos", feature = "avfoundation"))]
pub mod avfdevice;
// ScreenCaptureKit display capture (M739).
#[cfg(all(target_os = "macos", feature = "screencapture"))]
pub mod sck;

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
pub mod mfaacdecode;
#[cfg(all(target_os = "windows", feature = "mf-aac"))]
pub mod mfaacencode;

// Shared frame-emission loop for the packet-producing encoders below (and
// `gstwrap`, which emits its hosted element's output frames the same way).
#[cfg(any(
    feature = "av1-encode",
    feature = "vpx",
    feature = "opus",
    feature = "ffmpeg",
    feature = "nvenc",
    feature = "gstreamer"
))]
mod encoder_base;

// AV1 software encode via the pure-Rust rav1e crate (cross-platform).
#[cfg(feature = "av1-encode")]
pub mod av1enc;

// Shared AV1 decoder element body (the macro both backends expand), so the
// libdav1d and re_rav1d elements differ only in the backend crate they name.
#[cfg(any(feature = "dav1d", feature = "rav1d"))]
mod av1dec;

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

// PNG stills via the pure-Rust png crate (no system deps).
#[cfg(feature = "png")]
pub mod pngdec;
#[cfg(feature = "png")]
pub mod pngenc;

// WebP stills via the pure-Rust image-webp crate (no system deps).
#[cfg(feature = "webp")]
pub mod webpdec;

// Byte-stream framing and header geometry for the still-image formats.
mod stillframe;
// Geometry bounds and packed RGB(A) output shared by the still-image codecs.
mod stillimage;
// Netpbm stills (PBM / PGM / PPM): `pnmenc` / `pnmdec`, no extra crate.
pub mod pnm;

// Opus audio encode + decode via libopus (FFI through audiopus). Not pure Rust;
// links libopus (system or bundled-and-built), gated behind the `opus` feature.
#[cfg(feature = "opus")]
pub mod opusdec;
#[cfg(feature = "opus")]
pub mod opusenc;
// Vorbis decode, pure Rust via symphonia. Gated behind the `vorbis` feature.
#[cfg(feature = "vorbis")]
pub mod vorbisdec;

// HTTP(S) byte-stream source via reqwest (the fetch layer under HLS/DASH).
#[cfg(feature = "http-src")]
pub mod httpsrc;

// Shared HTTP fetch + URL helpers for the adaptive-streaming sources (not
// HttpSrc itself, which streams its response body directly).
#[cfg(any(feature = "hls", feature = "dash"))]
mod fetch;

// Shared throughput-driven ABR estimator for the adaptive-streaming sources.
#[cfg(any(feature = "hls", feature = "dash"))]
mod abr;

// Shared duration-keyed prebuffer window for the adaptive segment loops.
#[cfg(any(feature = "hls", feature = "dash"))]
mod segprebuf;

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
// RTMP "genuine FMS/FP" HMAC-SHA256 digest (complex) handshake, gated to the
// rtmp feature that supplies the crypto; the sans-IO core uses the simple one.
#[cfg(feature = "rtmp")]
pub mod rtmphandshake;
#[cfg(feature = "rtmp")]
pub mod rtmpsink;
#[cfg(feature = "rtmp")]
pub mod rtmpsrc;

// DASH MPD parser and the DashSrc segment source.
#[cfg(feature = "dash")]
pub mod dashsrc;
#[cfg(feature = "dash")]
pub mod mpd;

// Fragmented-MP4 / CMAF parsing (shared) and the byte-stream demuxer. In the
// std MP4 family (shares mp4box with mp4src/fmp4mux).
#[cfg(feature = "std")]
mod fmp4;
#[cfg(feature = "std")]
pub mod fmp4demux;
// Progressive / whole-file MP4 demuxer (M479): the single-output, buffer-to-Eos
// sibling of fmp4demux, for a bare `filesrc location=X.mp4 ! decodebin`.
#[cfg(feature = "std")]
pub mod mp4demux;
// MPEG Common Encryption: protection metadata parsing plus the shared key store
// and (behind `hls` / `mp4-cenc`) sample decryption for the fMP4 demux paths.
#[cfg(feature = "std")]
pub mod cenc;

// Worker-readiness latch shared by the platform display sinks below.
#[cfg(any(
    all(target_os = "windows", feature = "d3d11-sink"),
    all(target_os = "linux", feature = "wayland-sink"),
    all(target_os = "linux", feature = "cuda-gl"),
    all(target_os = "linux", feature = "gl-sink"),
    all(target_os = "linux", feature = "wgpu-present"),
))]
mod worker_ready;

// YUV_420_888 -> NV12 packer shared by the Android ndk-image elements
// (camera2src, mediacodecdec).
#[cfg(all(
    target_os = "android",
    any(feature = "camera2", feature = "mediacodec")
))]
mod yuv420;

// D3D11 present sink: displays MemoryDomain::D3D11Texture frames via a DXGI
// swapchain + D3D11 video processor. Windows-only; the analog of CudaGlSink.
#[cfg(all(target_os = "windows", feature = "d3d11-sink"))]
pub mod d3d11sink;

// WASAPI render sink: plays PCM on the selected audio endpoint (shared mode).
// Windows-only; the audible-output end of the M25 audio path.
#[cfg(all(target_os = "windows", feature = "wasapi-sink"))]
pub mod wasapisink;

// WASAPI capture source: captures PCM from the selected audio endpoint.
// Windows-only; the input mirror of WasapiSink.
#[cfg(all(target_os = "windows", feature = "wasapi-src"))]
pub mod wasapisrc;

// Endpoint selection + mix-format mapping shared by the two WASAPI elements
// and the endpoint provider.
#[cfg(all(
    target_os = "windows",
    any(feature = "wasapi-src", feature = "wasapi-sink")
))]
mod wasapipcm;

// WASAPI endpoint discovery: the active render / capture endpoints, with
// IMMNotificationClient hotplug.
#[cfg(all(
    target_os = "windows",
    any(feature = "wasapi-src", feature = "wasapi-sink")
))]
pub mod wasapidevice;

// Media Foundation camera capture source: drains frames from a video capture
// device via an IMFSourceReader. Windows-only; the video sibling of WasapiSrc.
#[cfg(all(target_os = "windows", feature = "mf-video-src"))]
pub mod mfvideosrc;

// Media Foundation camera discovery: the video capture devices MF enumerates,
// with the native modes mfvideosrc can deliver.
#[cfg(all(target_os = "windows", feature = "mf-video-src"))]
pub mod mfdevice;

// VAAPI H.264 / H.265 decode via cros-codecs is Linux-only. The dependency is
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

// PTP system clock (M593 phase C): reads the OS PTP-disciplined CLOCK_TAI on a
// worker and drives a g2g-core PtpClock, so a linuxptp-synced host offers a
// grandmaster clock to election. Linux-only (CLOCK_TAI); needs libc.
#[cfg(all(target_os = "linux", feature = "ptp"))]
pub mod ptpsystemclock;

// ptp4l management-socket query (M998): asks a local linuxptp for its port states
// and offset from master, so PtpSystemClock can tell real grandmaster lock from a
// readable CLOCK_TAI. Linux-only, `ptp` feature.
#[cfg(all(target_os = "linux", feature = "ptp"))]
pub mod ptp4l;

// In-process software PTP client (M594): speaks PTP over UDP (SLAVE mode) and
// disciplines a g2g-core PtpClock itself, so an endpoint without an OS PTP
// daemon can lock to a grandmaster. Needs privileged ports + a grandmaster; see
// module docs. std (via the `ptp` feature).
#[cfg(feature = "ptp")]
pub mod ptpclient;

// V4L2 capture source (UVC webcams etc.). Linux-only; streams packed YUYV
// (4:2:2) off /dev/videoN via mmap on a dedicated capture thread. See module
// docs.
#[cfg(all(target_os = "linux", feature = "v4l2"))]
pub mod v4l2src;

// V4L2 device discovery: enumerates /dev/videoN capture nodes with the probed
// modes of every format v4l2src carries, for the device monitor.
#[cfg(all(target_os = "linux", feature = "v4l2"))]
pub mod v4l2device;

// libcamera capture source (NV12 / YUYV) via the system libcamera stack. The
// modern Linux camera path: covers UVC webcams plus CSI/ISP cameras. Linux-only.
#[cfg(all(target_os = "linux", feature = "libcamera"))]
pub mod libcamerasrc;

// Zero-copy libcamera -> GPU dma-buf import feasibility probe (Linux + GPU).
#[cfg(all(target_os = "linux", feature = "libcamera-dmabuf"))]
pub mod libcamera_dmabuf;

// Zero-copy DMABUF -> wgpu buffer import element (Linux + GPU).
#[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
pub mod dmabufwgpu;

// Zero-copy wgpu buffer -> DMABUF export element (M559): the producer half that
// pairs with dmabufwgpu's importer, so a GPU frame leaves the process via a
// dma-buf fd (feed it to DmaBufSink). Needs Vulkan dma-buf export support.
#[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
pub mod wgpudmabuf;

// Vendor-neutral GPU-resident hardware video decode via Vulkan Video
// (VK_KHR_video_queue). Linux + Windows (both expose the extensions on RADV /
// ANV / the NVIDIA proprietary driver). See DESIGN.md 4.11.6.
#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]
pub mod vulkanvideo;

// HDR swapchain present (M575): present a decoded HDR texture to an on-screen
// swapchain with an HDR colour space + mastering metadata. A raw ash swapchain on
// the decode device (wgpu 29 cannot express a swapchain colour space). See
// DESIGN.md 4.11.6.
#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "hdr-present"
))]
pub mod vulkanhdrsink;

// Streaming-decoder adapter presenting the Vulkan Video decoders in the
// chunk-at-a-time CPU-frame (I420) shape a wgpu viewer's async decoder consumes
// (the wgpu-texture wedge); see the module docs.
#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]
pub mod streamdec;

// Reverse GStreamer bridge (`gstwrap`): host an unported GStreamer element
// inside a g2g graph. Drives `appsrc ! <element> ! appsink` via a C helper.
#[cfg(feature = "gstreamer")]
pub mod gstwrap;

// Wayland display sink (NV12 -> XRGB8888 via wl_shm). Linux-only;
// desktop-dev convenience sink — see module docs.
#[cfg(all(target_os = "linux", feature = "wayland-sink"))]
pub mod waylandsink;

// Linux audio render sinks and capture sources: the two ends of the audio path,
// the analogs of the Windows-only WasapiSink / WasapiSrc. Each links a different
// system audio stack and is target-gated to Linux behind its own feature; the
// sink and source of one stack share a helper module for the format / channel
// mapping they both need.
// ALSA (libasound), the lowest-level path.
#[cfg(all(target_os = "linux", any(feature = "alsa-sink", feature = "alsa-src")))]
mod alsapcm;
#[cfg(all(target_os = "linux", feature = "alsa-sink"))]
pub mod alsasink;
#[cfg(all(target_os = "linux", feature = "alsa-src"))]
pub mod alsasrc;
// ALSA device discovery: the PCM hint list as capture / playback devices.
#[cfg(all(target_os = "linux", any(feature = "alsa-sink", feature = "alsa-src")))]
pub mod alsadevice;
// PulseAudio / PipeWire-pulse via the blocking libpulse "simple" API.
#[cfg(all(
    target_os = "linux",
    any(feature = "pulse-sink", feature = "pulse-src")
))]
mod pulsepcm;
#[cfg(all(target_os = "linux", feature = "pulse-sink"))]
pub mod pulsesink;
#[cfg(all(target_os = "linux", feature = "pulse-src"))]
pub mod pulsesrc;
// PipeWire audio render sink + capture source and the video capture source (the
// modern Linux media layer). The elements share the `pipewire` feature and the
// pipewire-rs crate; the SPA pod helpers split per media type (pwaudio /
// pwvideo).
#[cfg(all(target_os = "linux", feature = "pipewire"))]
pub mod pipewiresink;
#[cfg(all(target_os = "linux", feature = "pipewire"))]
pub mod pipewiresrc;
#[cfg(all(target_os = "linux", feature = "pipewire"))]
pub mod pipewirevideosrc;
#[cfg(all(target_os = "linux", feature = "pipewire"))]
mod pwaudio;
#[cfg(all(target_os = "linux", feature = "pipewire"))]
mod pwvideo;
// The xdg-desktop-portal ScreenCast handshake `pipewirevideosrc portal=true`
// runs to get a screen-capture node on a Wayland desktop.
#[cfg(all(target_os = "linux", feature = "portal"))]
pub mod screencastportal;
// Device discovery over the PipeWire graph, the one Linux backend with native
// hotplug events (M939).
#[cfg(all(target_os = "linux", feature = "pipewire"))]
pub mod pwdevice;

// CUDA device-memory consumers (C3 Phase 3). `CudaDownload` copies a
// `MemoryDomain::Cuda` NV12 frame back to system memory so a `NvdecCuda`
// stream reaches the CPU sinks. Hand-rolled libcuda FFI; Linux + NVIDIA only.
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub mod cuda;

// Local zero-copy IPC over CUDA IPC memory handles (M556): share a device
// allocation with another same-machine + same-GPU process with no
// device->host->device copy. The handle is plain bytes, so it rides any
// transport. Linux + NVIDIA only (via the `cuda` gate).
#[cfg(all(target_os = "linux", feature = "local-ipc"))]
pub mod localipc;

// LocalCudaSink / LocalCudaSrc (M556 phase 2): the GPU-resident analog of the
// RemoteSink/RemoteSrc pair, carrying a MemoryDomain::Cuda NV12 frame to a
// same-machine peer over a Unix socket via a CUDA IPC handle (no PCIe round
// trip; the receive side takes one on-GPU device->device copy).
#[cfg(all(target_os = "linux", feature = "local-ipc"))]
pub mod localcuda;

// SCM_RIGHTS fd passing over a Unix socket (M557): hand-rolled sendmsg/recvmsg
// FFI used by the DMABUF local transport to move a dma-buf's file descriptor
// (which, unlike a CUDA IPC handle, is not plain bytes) between processes.
#[cfg(all(target_os = "linux", feature = "local-dmabuf"))]
pub mod scmfd;

// DmaBufSink / DmaBufSrc (M557): the vendor-neutral analog of LocalCudaSink/Src,
// carrying a MemoryDomain::DmaBuf frame to a same-machine peer over a Unix socket
// by passing the dma-buf fd as SCM_RIGHTS ancillary data (kernel-refcounted, so
// no per-frame ack). Linux only.
#[cfg(all(target_os = "linux", feature = "local-dmabuf"))]
pub mod localdmabuf;

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

// JPEG XS encode / decode (M605): `SvtJpegXsEnc` / `SvtJpegXsDec`, the ST 2110-22
// compressed essence. Hand-rolled FFI to Intel SVT-JPEG-XS (libSvtJpegxs, ISO/IEC
// 21122), struct layouts asserted against SvtJpegxs*.h; build.rs links it via
// pkg-config. Linux-only, behind the `jpegxs` feature.
#[cfg(all(target_os = "linux", feature = "jpegxs"))]
pub mod svtjpegxs;

// Shared GL ES render state for the EGL display sinks (program + textures +
// per-frame upload + draw); the platform present stays in each sink.
#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-gl", feature = "cuda-kms", feature = "gl-sink")
))]
pub(crate) mod glnv12;

// Shared Wayland window + present loop for the sinks that own their window on a
// worker thread; each supplies only the renderer that draws one frame.
#[cfg(all(
    target_os = "linux",
    any(feature = "cuda-gl", feature = "gl-sink", feature = "wgpu-present")
))]
pub(crate) mod waylandwindow;

// The compositor-reachable check the auto sink aliases resolve through, shared
// by every display sink that presents on Wayland.
#[cfg(all(
    target_os = "linux",
    any(
        feature = "wayland-sink",
        feature = "cuda-gl",
        feature = "gl-sink",
        feature = "wgpu-present"
    )
))]
pub(crate) mod waylanddisplay;

// EGL + GL ES renderer over that worker for the GL sinks; each sink supplies
// only its per-frame upload.
#[cfg(all(target_os = "linux", any(feature = "cuda-gl", feature = "gl-sink")))]
pub(crate) mod glwindow;

// Vendor-neutral GL ES display sink: system-memory NV12 / RGBA presented through
// EGL on Wayland, NV12->RGB converted on the GPU. Linux-only, no CUDA.
#[cfg(all(target_os = "linux", feature = "gl-sink"))]
pub mod glsink;

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

// CudaToWgpu: the element wiring cudawgpu's transport into a graph, so an NVDEC
// CUDA frame reaches a wgpu consumer (preprocess, present) with no PCIe
// round-trip. Linux + NVIDIA.
#[cfg(all(target_os = "linux", feature = "cuda-wgpu"))]
pub mod cudatowgpu;

// Browser / WebAssembly target (DESIGN.md §6.3), behind the `web` feature:
// WasmClock (performance.now + setTimeout) and WebSocketSrc ingest. The wasm
// bindings are target-gated to wasm32, so enabling `web` elsewhere is a no-op,
// like mf-decode on Linux. The deployable `#[wasm_bindgen]` browser entry points
// that wire these into a graph live in the excluded `g2g-web` cdylib crate.
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod wasmclock;
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod websocketsrc;

// WebSocketSink (M542): browser egress, send frame bytes over a WebSocket.
// PatternSrc (M542): synthetic animated RGBA source, the "capture" side of the
// browser send demo when no camera is present.
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod patternsrc;
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod websocketsink;

// WsWireSink (M554): the browser send half of the distributed-graph primitive.
// Ships wire-encoded PipelinePackets to a native RemoteWsSrc over a WebSocket,
// speaking the identical g2g-core wire codec (the media-agnostic generalization
// of the bespoke M549 WebRemoteDetect shim).
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod wswiresink;

// WsWireTransform (M555): the browser remote-transform. Offloads a middle stage
// to a native peer over one WebSocket (send frame, receive processed frame back),
// the generic replacement for the bespoke WebRemoteDetect detection shim.
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod wswiretransform;

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

// WebCodecs hardware ENCODE (M542): WebCodecsEncode wraps the browser VideoEncoder,
// RGBA -> H.264 Annex-B. Same `web-codecs` feature + unstable cfg.
#[cfg(all(target_arch = "wasm32", feature = "web-codecs"))]
pub mod webcodecsencode;

// Camera capture (M544): WebCameraSrc opens getUserMedia and reads the track's
// VideoFrames via a MediaStreamTrackProcessor, emitting RGBA (the real capture
// side of the egress pipeline). Shares `copy_out_rgba` with WebCodecsDecode, so
// it rides the `web-codecs` feature + unstable cfg.
#[cfg(all(target_arch = "wasm32", feature = "web-codecs"))]
pub mod webcamerasrc;

// WebGPU zero-copy presentation (M541): WebGpuCanvasSink imports the decoded
// VideoFrame (WebCodecsDecode GPU-texture output) as a GPUExternalTexture and
// renders it to a <canvas> WebGPU context, no CPU readback. Needs the `web-gpu`
// feature and RUSTFLAGS=--cfg=web_sys_unstable_apis.
#[cfg(all(target_arch = "wasm32", feature = "web-gpu"))]
pub mod webgpucanvassink;

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

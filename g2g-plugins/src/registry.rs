//! A pre-populated element [`Registry`] (M107), so a `gst-launch` text pipeline
//! and `gst-inspect` work out of the box without the caller hand-registering
//! every element.
//!
//! [`default_registry`] registers the standard `no_std`-baseline elements under
//! their conventional names: the test sources, the video and audio transform
//! chains, and the `fakesink` / `filesink` sinks. Each is default-constructed and
//! then configured by the parser from its `key=value` properties (M104/M106).
//!
//! `std`-only (the `Registry` is). Feature- and platform-gated elements (the
//! opus / av1 / vpx / mjpeg codecs, `fmp4demux`, the rtsp / udp / http / hls /
//! dash / rtmp network sources and sinks, and the Linux v4l2 / ffmpeg / vaapi /
//! wayland / kms / alsa / pulse elements) are registered by
//! [`register_feature_gated`], each block `#[cfg]`-gated like its module, so they
//! appear in `gst-inspect` / `parse_launch` when their feature is enabled.
//! `filesrc` is registered (M112): its `bytestream-format` property supplies the
//! container type a raw byte stream lacks, so `filesrc location=x.ts
//! bytestream-format=mpegts ! tsdemux` works as text.

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::runtime::{ElementFactory, LaunchFactory, MuxerFactory, Registry, SourceFactory};
use g2g_core::{AudioFormat, ByteStreamEncoding, Caps, Dim, Rate, RawVideoFormat};

use crate::aacparse::AacParse;
use crate::alpha::Alpha;
use crate::audioconvert::AudioConvert;
use crate::audiomixer::AudioMixer;
use crate::audiopanorama::AudioPanorama;
use crate::audioresample::AudioResample;
use crate::audiotestsrc::AudioTestSrc;
use crate::capsfilter::CapsFilter;
use crate::fakesink::FakeSink;
use crate::filesink::FileSink;
use crate::filesrc::FileSrc;
use crate::flvdemux::FlvDemux;
use crate::flvmux::FlvMux;
use crate::h264parse::H264Parse;
use crate::h265parse::H265Parse;
use crate::identity::IdentityTransform;
use crate::mkvdemux::MkvDemux;
use crate::mux::InterleaveMux;
use crate::mkvmux::MkvMux;
use crate::oggdemux::OggDemux;
use crate::opusparse::OpusParse;
use crate::vp8parse::Vp8Parse;
use crate::vp9parse::Vp9Parse;
use crate::av1parse::Av1Parse;
use crate::videobalance::VideoBalance;
use crate::videobox::VideoBox;
use crate::videoconvert::VideoConvert;
use crate::videocrop::VideoCrop;
use crate::videoflip::{FlipMethod, VideoFlip};
use crate::videorate::VideoRate;
use crate::videoscale::VideoScale;
use crate::textoverlay::TextOverlay;
use crate::videotestsrc::VideoTestSrc;
use crate::volume::Volume;
use crate::tsdemux::TsDemux;
use crate::tsmux::TsMux;

// Feature- (and platform-) gated elements, registered when their feature is on so
// `gst-inspect`, `gst-inspect --all`, and `parse_launch` see them. Each registers
// exactly as its `#[cfg]` in `lib.rs` gates the module.
#[cfg(feature = "opus")]
use crate::{opusdec::OpusDec, opusenc::OpusEnc};
#[cfg(feature = "av1-encode")]
use crate::av1enc::Av1Enc;
#[cfg(feature = "vpx")]
use crate::vpxenc::VpxEnc;
#[cfg(feature = "mjpeg")]
use crate::mjpegdec::MjpegDec;
#[cfg(feature = "mjpeg-encode")]
use crate::mjpegenc::MjpegEnc;
use crate::fmp4demux::Fmp4Demux;
#[cfg(feature = "rtsp")]
use crate::rtspsrc::RtspSrc;
#[cfg(feature = "udp-ingress")]
use crate::udpsrc::UdpSrc;
#[cfg(feature = "udp-egress")]
use crate::udpsink::UdpSink;
#[cfg(feature = "rtsp-server")]
use crate::rtspserversink::RtspServerSink;
#[cfg(feature = "rtsp-server")]
use crate::rtspserversrc::RtspServerSrc;
#[cfg(feature = "srt")]
use crate::srtsink::SrtSink;
#[cfg(feature = "srt")]
use crate::srtsrc::SrtSrc;
#[cfg(feature = "http-src")]
use crate::httpsrc::HttpSrc;
#[cfg(feature = "hls")]
use crate::hlssrc::HlsSrc;
#[cfg(feature = "dash")]
use crate::dashsrc::DashSrc;
#[cfg(feature = "rtmp")]
use crate::rtmpsink::RtmpSink;
#[cfg(feature = "rtmp")]
use crate::rtmpsrc::RtmpSrc;
#[cfg(all(target_os = "linux", feature = "wayland-sink"))]
use crate::waylandsink::WaylandSink;
#[cfg(feature = "webrtc")]
use crate::webrtcsink::WebRtcSink;
#[cfg(all(target_os = "linux", feature = "alsa-sink"))]
use crate::alsasink::AlsaSink;
#[cfg(all(target_os = "linux", feature = "pulse-sink"))]
use crate::pulsesink::PulseSink;
#[cfg(all(target_os = "linux", feature = "v4l2"))]
use crate::v4l2src::V4l2Src;
#[cfg(all(target_os = "linux", feature = "kms-sink"))]
use crate::kmssink::KmsSink;
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
use crate::ffmpegdec::{Backend as FfmpegBackend, FfmpegH264Dec};
#[cfg(all(target_os = "linux", feature = "vaapi"))]
use crate::vaapidec::VaapiH264Dec;

/// A [`Registry`] pre-populated with the standard elements, ready for
/// [`parse_launch`](g2g_core::runtime::parse_launch) and
/// [`inspect`](g2g_core::runtime::Registry::inspect).
///
/// ```text
/// videotestsrc num-buffers=10 ! videoconvert format=nv12 ! videoscale width=320 height=240 ! fakesink
/// audiotestsrc num-buffers=5 freq=440 ! audioconvert channels=1 ! audioresample samplerate=16000 ! fakesink
/// ```
pub fn default_registry() -> Registry {
    let mut reg = Registry::new();

    // Sources. The output caps are the autoplug `decodebin` input; the parser
    // only calls the constructor and applies properties.
    reg.register_source(SourceFactory::new(
        "videotestsrc",
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        },
        || Box::new(VideoTestSrc::new(320, 240, 30, 0)),
    ));
    reg.register_source(SourceFactory::new(
        "audiotestsrc",
        Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 },
        || Box::new(AudioTestSrc::new(48_000, 2, 440, 0)),
    ));
    // The output caps are a nominal default; the `bytestream-format` property
    // (incl. `auto`) sets the real container per instance before negotiation.
    reg.register_source(SourceFactory::new(
        "filesrc",
        Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs },
        || Box::new(FileSrc::new("", Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs })),
    ));
    // Application push source (M233): the real caps come from its `caps`
    // property; buffers arrive from `appsrc::register_appsrc`.
    reg.register_source(SourceFactory::new(
        "appsrc",
        crate::appsrc::registered_output_caps(),
        || Box::new(crate::appsrc::AppSrc::new()),
    ));

    // Video transforms.
    reg.register_launch(LaunchFactory::of::<VideoConvert>("videoconvert", || {
        // Caps-driven by default (M186): a bare `videoconvert` takes its output
        // format from a downstream capsfilter, or passes through.
        Box::new(VideoConvert::auto())
    }));
    reg.register_launch(LaunchFactory::of::<VideoScale>("videoscale", || {
        Box::new(VideoScale::new(0, 0))
    }));
    reg.register_launch(LaunchFactory::of::<VideoCrop>("videocrop", || {
        Box::new(VideoCrop::new(0, 0, 0, 0))
    }));
    reg.register_launch(LaunchFactory::of::<VideoFlip>("videoflip", || {
        Box::new(VideoFlip::new(FlipMethod::Identity))
    }));
    reg.register_launch(LaunchFactory::of::<VideoBalance>("videobalance", || {
        Box::new(VideoBalance::new())
    }));
    reg.register_launch(LaunchFactory::of::<Alpha>("alpha", || Box::new(Alpha::new())));
    reg.register_launch(LaunchFactory::of::<VideoBox>("videobox", || Box::new(VideoBox::new())));
    // Subtitle overlay (M171): the `location=` property loads an SRT / WebVTT
    // file (std), so cues render by PTS without hand-built Rust.
    reg.register_launch(LaunchFactory::of::<TextOverlay>("textoverlay", || {
        Box::new(TextOverlay::new())
    }));
    // VideoRate / IdentityTransform have no pad templates declared.
    reg.register_launch(LaunchFactory::new("videorate", Vec::new(), || {
        Box::new(VideoRate::new(30.0))
    }));

    // Audio transforms.
    reg.register_launch(LaunchFactory::of::<AudioConvert>("audioconvert", || {
        Box::new(AudioConvert::new(AudioFormat::PcmS16Le, 2))
    }));
    reg.register_launch(LaunchFactory::of::<AudioResample>("audioresample", || {
        // Caps-driven by default (M187): a bare `audioresample` takes its output
        // rate from a downstream capsfilter, or passes through.
        Box::new(AudioResample::auto())
    }));
    reg.register_launch(LaunchFactory::of::<Volume>("volume", || Box::new(Volume::new())));
    reg.register_launch(LaunchFactory::of::<AudioPanorama>("audiopanorama", || {
        Box::new(AudioPanorama::new())
    }));

    // Demuxers + parsers + passthrough.
    reg.register_launch(LaunchFactory::of::<TsDemux>("tsdemux", || Box::new(TsDemux::new())));
    reg.register_launch(LaunchFactory::of::<MkvDemux>("matroskademux", || Box::new(MkvDemux::new())));
    reg.register_launch(LaunchFactory::of::<TsMux>("mpegtsmux", || Box::new(TsMux::new())));
    reg.register_launch(LaunchFactory::of::<MkvMux>("matroskamux", || Box::new(MkvMux::new())));
    reg.register_launch(LaunchFactory::of::<OggDemux>("oggdemux", || Box::new(OggDemux::new())));
    reg.register_launch(LaunchFactory::of::<Fmp4Demux>("fmp4demux", || Box::new(Fmp4Demux::new())));
    reg.register_launch(LaunchFactory::of::<FlvDemux>("flvdemux", || Box::new(FlvDemux::new())));
    reg.register_launch(LaunchFactory::of::<FlvMux>("flvmux", || Box::new(FlvMux::new())));
    reg.register_launch(LaunchFactory::of::<H264Parse>("h264parse", || Box::new(H264Parse::new())));
    reg.register_launch(LaunchFactory::of::<H265Parse>("h265parse", || Box::new(H265Parse::new())));
    reg.register_launch(LaunchFactory::of::<AacParse>("aacparse", || Box::new(AacParse::new())));
    reg.register_launch(LaunchFactory::of::<OpusParse>("opusparse", || Box::new(OpusParse::new())));
    reg.register_launch(LaunchFactory::of::<Vp8Parse>("vp8parse", || Box::new(Vp8Parse::new())));
    reg.register_launch(LaunchFactory::of::<Vp9Parse>("vp9parse", || Box::new(Vp9Parse::new())));
    reg.register_launch(LaunchFactory::of::<Av1Parse>("av1parse", || Box::new(Av1Parse::new())));
    reg.register_launch(LaunchFactory::new("identity", Vec::new(), || {
        Box::new(IdentityTransform::new())
    }));
    // The inline caps-filter shorthand (`! video/x-raw,width=320 !`) builds this
    // by name with a `caps` property; see runtime::parse_launch.
    reg.register_launch(LaunchFactory::new("capsfilter", Vec::new(), || {
        Box::new(CapsFilter::default())
    }));

    // Fan-in muxer (M122). `funnel` is the structural N-to-1 forwarder for text
    // fan-in (`funnel name=m ! sink   a ! m.   b ! m.`); the parser derives the
    // input count from link degree. The output caps are a nominal default (frames
    // carry their own caps downstream), matching `videotestsrc`'s default.
    reg.register_muxer(MuxerFactory::new("funnel", |inputs| {
        Box::new(InterleaveMux::new(
            inputs,
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(320),
                height: Dim::Fixed(240),
                framerate: Rate::Fixed(30 << 16),
            },
        ))
    }));
    // Summing audio fan-in (M130): adds aligned S16LE buffers from N inputs. The
    // output caps are a nominal default matching `audiotestsrc`.
    reg.register_muxer(MuxerFactory::new("audiomixer", |inputs| {
        Box::new(AudioMixer::new(
            inputs,
            Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 },
        ))
    }));
    // Multi-stream MPEG-TS fan-in (M208): the A+V container case. `mpegtsmux` is
    // registered both as a single-input launch element (the `tsmux::TsMux` above)
    // and here as a fan-in muxer (`tsmuxn::TsMux`); the parser picks by link
    // degree (`make_element` for one input, `make_muxer` for several), so the one
    // name covers both the `! mpegtsmux !` and `v.! m.  a.! m.  mpegtsmux name=m`
    // shapes the way gst's request sink pads do. Each input's PMT stream type is
    // learned from its negotiated caps; AUs interleave by PTS (M204).
    reg.register_muxer(MuxerFactory::new("mpegtsmux", |inputs| {
        Box::new(crate::tsmuxn::TsMux::new(inputs))
    }));

    // Sinks.
    reg.register_launch(LaunchFactory::of::<FakeSink>("fakesink", || Box::new(FakeSink::new())));
    // Application pull/callback sink (M233): hands buffers to a callback set via
    // `appsink::set_appsink_callback`.
    reg.register_launch(LaunchFactory::of::<crate::appsink::AppSink>("appsink", || {
        Box::new(crate::appsink::AppSink::new())
    }));
    reg.register_launch(LaunchFactory::of::<FileSink>("filesink", || Box::new(FileSink::new(""))));
    #[cfg(feature = "rtmp")]
    reg.register_launch(LaunchFactory::of::<RtmpSink>("rtmpsink", || Box::new(RtmpSink::new(""))));

    register_feature_gated(&mut reg);
    register_aliases(&mut reg);
    register_autoplug_candidates(&mut reg);
    register_uri_handlers(&mut reg);

    reg
}

/// Register the `uri=` scheme handlers (M196) so `uridecodebin` / `playbin` in a
/// text pipeline can build their source from a URI. Each handler is gated to its
/// source's feature (the same gate as in [`uridecodebin`](crate::uridecodebin)),
/// so a scheme is available only when its source is compiled in.
fn register_uri_handlers(reg: &mut Registry) {
    // file:// -> Mp4Src (self-demuxing MP4, emits H.264). Available under std.
    reg.register_uri(crate::uridecodebin::file_handler());
    #[cfg(feature = "udp-ingress")]
    reg.register_uri(crate::uridecodebin::udp_handler());
    #[cfg(feature = "rtsp")]
    reg.register_uri(crate::uridecodebin::rtsp_handler());
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    reg.register_uri(crate::uridecodebin::v4l2_handler());
}

/// Register the parsers and decoders as auto-plug [`ElementFactory`] candidates
/// (M193), so the decode-chain search (`Registry::autoplug` / the `decodebin`
/// parser node / `build_playbin`) has elements to compose. These are the same
/// element types already registered for the text parser via `register_launch`;
/// here they additionally carry their pad templates into the search. Parsers
/// bridge a byte / elementary stream to a fixed compressed codec; decoders bridge
/// a compressed codec to raw. The build closures ignore the chosen output caps
/// (these elements take their format from negotiation), matching the
/// parameterless launch constructors. Decoders mirror their feature gates, so the
/// search only routes through a decoder when its backend is compiled in.
fn register_autoplug_candidates(reg: &mut Registry) {
    // Parsers (baseline): elementary-stream framing, no external deps.
    reg.register(ElementFactory::of::<H264Parse>("h264parse", |_| Box::new(H264Parse::new())));
    reg.register(ElementFactory::of::<H265Parse>("h265parse", |_| Box::new(H265Parse::new())));
    reg.register(ElementFactory::of::<AacParse>("aacparse", |_| Box::new(AacParse::new())));
    reg.register(ElementFactory::of::<OpusParse>("opusparse", |_| Box::new(OpusParse::new())));
    reg.register(ElementFactory::of::<Vp8Parse>("vp8parse", |_| Box::new(Vp8Parse::new())));
    reg.register(ElementFactory::of::<Vp9Parse>("vp9parse", |_| Box::new(Vp9Parse::new())));
    reg.register(ElementFactory::of::<Av1Parse>("av1parse", |_| Box::new(Av1Parse::new())));

    // Demuxers (baseline, M194): a container byte stream in, one selected
    // elementary stream out. They are 1-in/1-out (an instance forwards one stream,
    // chosen by codec), so the chain search composes them like any other element:
    // ByteStream{container} -> tsdemux/... -> CompressedVideo|Audio -> decoder ->
    // raw. Built parameterless = the default (video) stream, which matches both
    // the search's first-alternative choice and the decodebin macro's by-name
    // build, so the two decode paths stay consistent.
    reg.register(ElementFactory::of::<TsDemux>("tsdemux", |_| Box::new(TsDemux::new())));
    reg.register(ElementFactory::of::<MkvDemux>("matroskademux", |_| Box::new(MkvDemux::new())));
    reg.register(ElementFactory::of::<Fmp4Demux>("fmp4demux", |_| Box::new(Fmp4Demux::new())));
    reg.register(ElementFactory::of::<OggDemux>("oggdemux", |_| Box::new(OggDemux::new())));
    reg.register(ElementFactory::of::<FlvDemux>("flvdemux", |_| Box::new(FlvDemux::new())));

    // Decoders (feature- + platform-gated, same gate as the launch registration).
    #[cfg(feature = "opus")]
    reg.register(ElementFactory::of::<OpusDec>("opusdec", |_| Box::new(OpusDec::new())));
    #[cfg(feature = "mjpeg")]
    reg.register(ElementFactory::of::<MjpegDec>("mjpegdec", |_| Box::new(MjpegDec::new())));
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register(ElementFactory::of::<FfmpegH264Dec>("ffmpegdec", |_| Box::new(FfmpegH264Dec::new())));
    // ffmpeg VAAPI hwaccel backend as a distinct name (M237). Same element type
    // as ffmpegdec, constructed with `Backend::Vaapi`; the libva device defaults
    // to the VA display's choice (a `device=` property is a follow-up).
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register(ElementFactory::of::<FfmpegH264Dec>("ffmpegvaapidec", |_| {
        Box::new(FfmpegH264Dec::new().with_backend(FfmpegBackend::Vaapi))
    }));
    #[cfg(all(target_os = "linux", feature = "vaapi"))]
    reg.register(ElementFactory::of::<VaapiH264Dec>("vaapidec", |_| Box::new(VaapiH264Dec::new())));
}

/// Register gst-canonical-name aliases (M192) so pasted `gst-launch` lines using
/// GStreamer's element names resolve to the g2g equivalents. Each alias resolves
/// at construction time to the first of its targets that is actually registered,
/// so the decoder / display aliases work only when their feature is on, and the
/// `auto*sink` aliases fall back through the available display / audio sinks to
/// `fakesink` (always present), which keeps a tutorial line running headless.
fn register_aliases(reg: &mut Registry) {
    // Auto sinks: prefer a real display / audio sink, fall back to fakesink.
    reg.register_alias("autovideosink", &["waylandsink", "kmssink", "fakesink"]);
    reg.register_alias("autoaudiosink", &["alsasink", "pulsesink", "fakesink"]);
    // Common desktop video-sink names map onto whatever display sink we have.
    for name in ["xvimagesink", "ximagesink", "glimagesink"] {
        reg.register_alias(name, &["waylandsink", "kmssink", "fakesink"]);
    }
    // Decoders: GStreamer's libav / VA-API names -> the g2g decoders. The VA-API
    // names prefer the ffmpeg VAAPI hwaccel (`ffmpegvaapidec`, works on Mesa
    // radeonsi) and fall back to the cros-codecs `vaapidec` when only that
    // feature is on; the alias resolves to the first registered target.
    reg.register_alias("avdec_h264", &["ffmpegdec"]);
    reg.register_alias("vaapih264dec", &["ffmpegvaapidec", "vaapidec"]);
    reg.register_alias("vah264dec", &["ffmpegvaapidec", "vaapidec"]);
    // VPx encoders: gst splits vp8enc / vp9enc; g2g has one vpxenc.
    reg.register_alias("vp8enc", &["vpxenc"]);
    reg.register_alias("vp9enc", &["vpxenc"]);
}

/// Register the feature- and platform-gated elements. Each block compiles only
/// when its `#[cfg]` (the same gate as the module in `lib.rs`) holds, so a build
/// without the feature is unchanged. Sources whose constructor needs a runtime
/// value (a URL, a socket, a device) are default-built with a placeholder; the
/// real value comes from a property / builder before use (the placeholder only
/// has to be side-effect-free, since `inspect` default-builds to read metadata).
#[allow(unused_variables)]
fn register_feature_gated(reg: &mut Registry) {
    // Codecs (cross-platform).
    #[cfg(feature = "opus")]
    {
        reg.register_launch(LaunchFactory::of::<OpusEnc>("opusenc", || Box::new(OpusEnc::new())));
        reg.register_launch(LaunchFactory::of::<OpusDec>("opusdec", || Box::new(OpusDec::new())));
    }
    #[cfg(feature = "av1-encode")]
    reg.register_launch(LaunchFactory::of::<Av1Enc>("av1enc", || Box::new(Av1Enc::new())));
    #[cfg(feature = "vpx")]
    reg.register_launch(LaunchFactory::of::<VpxEnc>("vpxenc", || Box::new(VpxEnc::new())));
    #[cfg(feature = "mjpeg")]
    reg.register_launch(LaunchFactory::of::<MjpegDec>("mjpegdec", || Box::new(MjpegDec::new())));
    #[cfg(feature = "mjpeg-encode")]
    reg.register_launch(LaunchFactory::of::<MjpegEnc>("mjpegenc", || Box::new(MjpegEnc::new())));

    // Network sources / sinks.
    #[cfg(feature = "rtsp")]
    reg.register_source(SourceFactory::new(
        "rtspsrc",
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        },
        || Box::new(RtspSrc::new("")),
    ));
    #[cfg(feature = "udp-ingress")]
    reg.register_source(SourceFactory::new(
        "udpsrc",
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        },
        || Box::new(UdpSrc::new("0.0.0.0:5004".parse().unwrap())),
    ));
    #[cfg(feature = "udp-egress")]
    reg.register_launch(LaunchFactory::of::<UdpSink>("udpsink", || {
        Box::new(UdpSink::new("127.0.0.1:5004".parse().unwrap()))
    }));
    #[cfg(feature = "rtsp-server")]
    reg.register_launch(LaunchFactory::of::<RtspServerSink>("rtspserversink", || {
        Box::new(RtspServerSink::new("0.0.0.0:8554".parse().unwrap()))
    }));
    #[cfg(feature = "rtsp-server")]
    reg.register_source(SourceFactory::new(
        "rtspserversrc",
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        },
        || Box::new(RtspServerSrc::new("0.0.0.0:8554".parse().unwrap())),
    ));
    #[cfg(feature = "srt")]
    reg.register_source(SourceFactory::new(
        "srtsrc",
        Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs },
        || Box::new(SrtSrc::new("0.0.0.0:9000".parse().unwrap())),
    ));
    #[cfg(feature = "srt")]
    reg.register_launch(LaunchFactory::of::<SrtSink>("srtsink", || {
        Box::new(SrtSink::new("127.0.0.1:9000".parse().unwrap()))
    }));
    #[cfg(feature = "http-src")]
    reg.register_source(SourceFactory::new(
        "httpsrc",
        Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs },
        || Box::new(HttpSrc::new("", Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs })),
    ));
    #[cfg(feature = "hls")]
    reg.register_source(SourceFactory::new(
        "hlssrc",
        Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs },
        || Box::new(HlsSrc::new("")),
    ));
    #[cfg(feature = "dash")]
    reg.register_source(SourceFactory::new(
        "dashsrc",
        Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff },
        || Box::new(DashSrc::new("")),
    ));
    #[cfg(feature = "rtmp")]
    reg.register_source(SourceFactory::new(
        "rtmpsrc",
        Caps::ByteStream { encoding: ByteStreamEncoding::Flv },
        || Box::new(RtmpSrc::new("0.0.0.0:1935".parse().unwrap())),
    ));

    // Linux capture / decode / display.
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    reg.register_source(SourceFactory::new(
        "v4l2src",
        Caps::RawVideo {
            format: RawVideoFormat::Yuyv,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        },
        || Box::new(V4l2Src::new("/dev/video0")),
    ));
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register_launch(LaunchFactory::of::<FfmpegH264Dec>("ffmpegdec", || {
        Box::new(FfmpegH264Dec::new())
    }));
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register_launch(LaunchFactory::of::<FfmpegH264Dec>("ffmpegvaapidec", || {
        Box::new(FfmpegH264Dec::new().with_backend(FfmpegBackend::Vaapi))
    }));
    #[cfg(all(target_os = "linux", feature = "vaapi"))]
    reg.register_launch(LaunchFactory::of::<VaapiH264Dec>("vaapidec", || {
        Box::new(VaapiH264Dec::new())
    }));
    #[cfg(all(target_os = "linux", feature = "wayland-sink"))]
    reg.register_launch(LaunchFactory::new("waylandsink", Vec::new(), || {
        Box::new(WaylandSink::new())
    }));
    // WebRTC WHIP egress; the `location` property targets the endpoint. The URL
    // defaults empty (set it via `webrtcsink location=...`); publishing starts
    // on the first frame.
    #[cfg(feature = "webrtc")]
    reg.register_launch(LaunchFactory::new("webrtcsink", Vec::new(), || {
        Box::new(WebRtcSink::new(""))
    }));
    #[cfg(all(target_os = "linux", feature = "kms-sink"))]
    reg.register_launch(LaunchFactory::new("kmssink", Vec::new(), || Box::new(KmsSink::new())));
    #[cfg(all(target_os = "linux", feature = "alsa-sink"))]
    reg.register_launch(LaunchFactory::of::<AlsaSink>("alsasink", || Box::new(AlsaSink::new())));
    #[cfg(all(target_os = "linux", feature = "pulse-sink"))]
    reg.register_launch(LaunchFactory::of::<PulseSink>("pulsesink", || Box::new(PulseSink::new())));
}

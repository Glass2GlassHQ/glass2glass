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
#[cfg(feature = "std")]
use crate::mp4mux::Mp4Mux;
use crate::oggdemux::OggDemux;
use crate::opusparse::OpusParse;
use crate::vp8parse::Vp8Parse;
use crate::vp9parse::Vp9Parse;
use crate::av1parse::Av1Parse;
use crate::videobalance::VideoBalance;
use crate::videobox::VideoBox;
use crate::tensorconvert::TensorConvert;
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
#[cfg(feature = "dav1d")]
use crate::dav1ddec::Dav1dDec;
#[cfg(feature = "rav1d")]
use crate::rav1ddec::Rav1dDec;
#[cfg(feature = "mjpeg")]
use crate::mjpegdec::MjpegDec;
#[cfg(feature = "mjpeg-encode")]
use crate::mjpegenc::MjpegEnc;
use crate::fmp4demux::Fmp4Demux;
#[cfg(feature = "rtsp")]
use crate::rtspsrc::RtspSrc;
#[cfg(feature = "onvif")]
use crate::onvif::OnvifSrc;
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
#[cfg(feature = "webrtc")]
use crate::webrtcwhepsrc::WebRtcWhepSrc;
#[cfg(all(target_os = "linux", feature = "alsa-sink"))]
use crate::alsasink::AlsaSink;
#[cfg(all(target_os = "linux", feature = "pulse-sink"))]
use crate::pulsesink::PulseSink;
#[cfg(all(target_os = "linux", feature = "v4l2"))]
use crate::v4l2src::V4l2Src;
#[cfg(all(target_os = "linux", feature = "libcamera"))]
use crate::libcamerasrc::LibCameraSrc;
#[cfg(all(target_os = "linux", feature = "kms-sink"))]
use crate::kmssink::KmsSink;
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
use crate::ffmpegdec::{Backend as FfmpegBackend, FfmpegH264Dec};
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
use crate::ffmpegenc::{Backend as FfmpegEncBackend, FfmpegH264Enc};
#[cfg(all(target_os = "linux", feature = "vaapi"))]
use crate::vaapidec::VaapiH264Dec;
#[cfg(all(target_os = "linux", feature = "nvdec"))]
use crate::nvdec::NvDec;
#[cfg(all(target_os = "linux", feature = "nvenc"))]
use crate::nvenc::NvEnc;
#[cfg(all(target_os = "android", feature = "mediacodec"))]
use crate::mediacodecdec::MediaCodecDec;
#[cfg(all(target_os = "android", feature = "mediacodec"))]
use crate::mediacodecenc::MediaCodecEnc;
#[cfg(all(target_os = "android", feature = "aaudio"))]
use crate::aaudio::{AAudioSink, AAudioSrc};
#[cfg(all(target_os = "android", feature = "camera2"))]
use crate::camera2src::Camera2Src;

/// A [`Registry`] pre-populated with the standard elements, ready for
/// [`parse_launch`](g2g_core::runtime::parse_launch) and
/// [`inspect`](g2g_core::runtime::Registry::inspect).
///
/// ```text
/// videotestsrc num-buffers=10 ! videoconvert format=nv12 ! videoscale width=320 height=240 ! fakesink
/// audiotestsrc num-buffers=5 freq=440 ! audioconvert channels=1 ! audioresample samplerate=16000 ! fakesink
/// ```
/// The decode-chain parser injector (M421): an auto-plugged decoder is fed one
/// access unit per packet by splicing an access-unit-re-framing `h264parse` ahead
/// of it, the way GStreamer's `decodebin` always inserts a parser. Returns `None`
/// for codecs without a re-framing parser (the input decodes directly). H.264
/// (M421) and H.265 (M425) re-frame to one access unit per packet; audio still
/// decodes directly.
fn video_parser_provider(input: &Caps) -> Option<Box<dyn g2g_core::element::DynAsyncElement>> {
    match input {
        Caps::CompressedVideo { codec: g2g_core::VideoCodec::H264, .. } => {
            Some(Box::new(crate::h264parse::H264Parse::reframing()))
        }
        Caps::CompressedVideo { codec: g2g_core::VideoCodec::H265, .. } => {
            Some(Box::new(crate::h265parse::H265Parse::reframing()))
        }
        _ => None,
    }
}

pub fn default_registry() -> Registry {
    let mut reg = Registry::new();
    // Auto-plugged decode chains splice a re-framing parser before the decoder
    // (M421), so a decoder fed un-access-unit-aligned input (e.g. one MPEG-TS PES
    // that is not one coded picture) does not mis-parse.
    reg.set_parser_provider(video_parser_provider);

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
        // num-buffers defaults to forever (the property's documented `-1`),
        // matching gst videotestsrc; a launch line bounds it with `num-buffers=N`.
        || Box::new(VideoTestSrc::new(320, 240, 30, u64::MAX)),
    ));
    reg.register_source(SourceFactory::new(
        "audiotestsrc",
        Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 },
        // num-buffers defaults to forever (the property's documented `-1`),
        // matching gst audiotestsrc; a launch line bounds it with `num-buffers=N`.
        || Box::new(AudioTestSrc::new(48_000, 2, 440, u64::MAX)),
    ));
    // Android AAudio mic capture (M307); the device may open with different
    // actuals, reported as the produced caps. `aaudiosrc` is the gst analog.
    #[cfg(all(target_os = "android", feature = "aaudio"))]
    reg.register_source(SourceFactory::new(
        "aaudiosrc",
        Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 },
        || Box::new(AAudioSrc::new(48_000, 2, u64::MAX)),
    ));
    // Android camera capture (M308); 640x480 NV12 default. `camerasrc` /
    // `ahcsrc` are the gst analogs.
    #[cfg(all(target_os = "android", feature = "camera2"))]
    reg.register_source(SourceFactory::new(
        "camera2src",
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        },
        || Box::new(Camera2Src::new(640, 480, u64::MAX)),
    ));
    // The output caps are a nominal default; the `bytestream-format` property
    // (incl. `auto`) sets the real container per instance before negotiation.
    reg.register_source(SourceFactory::new(
        "filesrc",
        Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs },
        || Box::new(FileSrc::new("", Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs })),
    ));
    // Subtitle / text file source (M433): a `.srt` / `.vtt` / `.ssa` / `.ttml`
    // file as a `Text` stream, feeding `subparse` (overlay or caption authoring).
    // The `format` is sniffed from the `location` extension unless set explicitly.
    reg.register_source(SourceFactory::new(
        "subtitlesrc",
        Caps::Text { format: g2g_core::TextFormat::Srt },
        || Box::new(crate::subtitlesrc::SubtitleSrc::new("", g2g_core::TextFormat::Srt)),
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
    // Tensor dtype converter (M441): quantize/dequantize, the tensor sibling of
    // videoconvert. A bare instance quantizes to uint8 (scale 1, zp 0); the real
    // affine params come from the `scale` / `zero-point` / `dtype` properties.
    reg.register_launch(LaunchFactory::of::<TensorConvert>("tensorconvert", || {
        Box::new(TensorConvert::quantize(g2g_core::TensorDType::U8, 1.0, 0))
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
    // Closed-caption extractor (M429): mines CEA-608 / CEA-708 captions from a
    // compressed H.264 / H.265 stream's SEI into timed text cues (default CC1),
    // e.g. `... ! h264parse ! ccextract ! textoverlay ...` on a teed branch.
    reg.register_launch(LaunchFactory::of::<crate::ccextract::CcExtract>("ccextract", || {
        Box::new(crate::ccextract::CcExtract::new())
    }));
    // Detection-box overlay (M102): draws the frame's `AnalyticsMeta` bounding
    // boxes onto the RGBA frame, so a detector's output is visible downstream
    // (e.g. `... ! analyticsoverlay ! videoconvert ! autovideosink`). No pad
    // templates declared (caps-driven via intercept_caps). Gated on `analytics`,
    // the metadata graph it reads.
    #[cfg(feature = "analytics")]
    reg.register_launch(LaunchFactory::new("analyticsoverlay", Vec::new(), || {
        Box::new(crate::analyticsoverlay::AnalyticsOverlay::new())
    }));
    // VideoRate / IdentityTransform have no pad templates declared.
    reg.register_launch(LaunchFactory::new("videorate", Vec::new(), || {
        // Caps-driven by default (M290): `videorate ! caps,framerate=N` sets the
        // rate; `videorate framerate=N` still works via the property; bare
        // `videorate` passes the input rate through.
        Box::new(VideoRate::auto())
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
    // Fragmented-MP4 / ISO-BMFF muxer (M291), the gst `mp4mux`/`qtmux` analog:
    // `... ! x264enc ! mp4mux ! filesink location=out.mp4`. std-gated like its
    // module (it shares the `fmp4mux` box writer).
    #[cfg(feature = "std")]
    reg.register_launch(LaunchFactory::of::<Mp4Mux>("mp4mux", || Box::new(Mp4Mux::new())));
    reg.register_launch(LaunchFactory::of::<OggDemux>("oggdemux", || Box::new(OggDemux::new())));
    reg.register_launch(LaunchFactory::of::<Fmp4Demux>("fmp4demux", || Box::new(Fmp4Demux::new())));
    reg.register_launch(LaunchFactory::of::<FlvDemux>("flvdemux", || Box::new(FlvDemux::new())));
    reg.register_launch(LaunchFactory::of::<FlvMux>("flvmux", || Box::new(FlvMux::new())));
    // Re-framing mode (M421): a `gst-launch` `h264parse` access-unit-aligns its
    // output (one coded picture per buffer), matching GStreamer's `h264parse`, so
    // `... ! tsdemux ! h264parse ! <decoder> ! ...` feeds the decoder correctly.
    reg.register_launch(LaunchFactory::of::<H264Parse>("h264parse", || Box::new(H264Parse::reframing())));
    reg.register_launch(LaunchFactory::of::<H265Parse>("h265parse", || Box::new(H265Parse::reframing())));
    reg.register_launch(LaunchFactory::of::<AacParse>("aacparse", || Box::new(AacParse::new())));
    reg.register_launch(LaunchFactory::of::<OpusParse>("opusparse", || Box::new(OpusParse::new())));
    reg.register_launch(LaunchFactory::of::<Vp8Parse>("vp8parse", || Box::new(Vp8Parse::new())));
    reg.register_launch(LaunchFactory::of::<Vp9Parse>("vp9parse", || Box::new(Vp9Parse::new())));
    reg.register_launch(LaunchFactory::of::<Av1Parse>("av1parse", || Box::new(Av1Parse::new())));
    reg.register_launch(LaunchFactory::new("identity", Vec::new(), || {
        Box::new(IdentityTransform::new())
    }));
    // A/V offset (M385): shifts PTS/DTS by `offset=` ns; the av-offset sync knob.
    reg.register_launch(LaunchFactory::new("avoffset", Vec::new(), || {
        Box::new(crate::avoffset::AvOffset::new(0))
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
    // Multi-track fragmented-MP4 fan-in (M293): the A/V container case. Like
    // `mpegtsmux`, `mp4mux` is registered both as a single-input launch element
    // (`mp4mux::Mp4Mux` above) and here as a fan-in muxer (`mp4muxn::Mp4MuxN`);
    // the parser picks by link degree, so one name covers `! mp4mux !` and
    // `v.! m.  a.! m.  mp4mux name=m`. Video + AAC audio interleave by PTS.
    #[cfg(feature = "std")]
    reg.register_muxer(MuxerFactory::new("mp4mux", |inputs| {
        Box::new(crate::mp4muxn::Mp4MuxN::new(inputs))
    }));
    // Multi-track Matroska / WebM fan-in (M294): the A/V container case. Like
    // `mpegtsmux`, `matroskamux` is registered both as a single-input launch
    // element (`mkvmux::MkvMux` above) and here as a fan-in muxer
    // (`mkvmuxn::MkvMuxN`); the parser picks by link degree, so one name covers
    // `! matroskamux !` and `v.! m.  a.! m.  matroskamux name=m`. H.26x video +
    // AAC audio interleave by PTS. std-gated like the `mp4mux` fan-in above.
    #[cfg(feature = "std")]
    reg.register_muxer(MuxerFactory::new("matroskamux", |inputs| {
        Box::new(crate::mkvmuxn::MkvMuxN::new(inputs))
    }));
    // Multi-track FLV fan-in (M296): the A/V container case, FLV's one-video +
    // one-audio model. Like the others, `flvmux` is both a single-input launch
    // element (`flvmux::FlvMux` above) and this fan-in muxer (`flvmuxn::FlvMuxN`);
    // the parser picks by link degree. H.264 video + AAC audio interleave by PTS,
    // with the decoder-config sequence headers written up front. std-gated.
    #[cfg(feature = "std")]
    reg.register_muxer(MuxerFactory::new("flvmux", |inputs| {
        Box::new(crate::flvmuxn::FlvMuxN::new(inputs))
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
    // playbin uri=file://x auto-fan-out: each hook probes the container and builds
    // a multi-stream graph, declining (so the next hook / single-stream playbin
    // takes over) for a container it does not parse. MKV (M382), MPEG-TS (M389),
    // then fragmented MP4 (M392).
    reg.register_playbin(crate::uridecodebin::mkv_playbin);
    reg.register_playbin(crate::uridecodebin::ts_playbin);
    reg.register_playbin(crate::uridecodebin::mp4_playbin);
    // hls:// fan-out (M395): probe the master playlist, fan its variant's muxed TS
    // streams out; the hls_handler is the single-stream fallback it declines to.
    #[cfg(feature = "hls")]
    {
        reg.register_uri(crate::uridecodebin::hls_handler());
        reg.register_playbin(crate::uridecodebin::hls_playbin);
    }
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
/// a compressed codec to raw. Most build closures ignore the chosen output caps
/// (these elements take their format from negotiation), matching the
/// parameterless launch constructors; the ffmpeg decoders are the exception, as
/// they have a fixed-at-construction output layout (NV12 / I420) that must match
/// the alternative the search settled on (see [`ffmpegdec_output_format`]).
/// Decoders mirror their feature gates, so the search only routes through a
/// decoder when its backend is compiled in.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
fn ffmpegdec_output_format(out: &Caps) -> crate::ffmpegdec::OutputFormat {
    use crate::ffmpegdec::OutputFormat;
    // The source pad template lists NV12 before I420, so a `is_raw_video` target
    // settles on NV12 (the layout KMS / waylandsink want); an I420-only sink drives
    // I420. Anything that is not raw video falls back to I420.
    match out {
        Caps::RawVideo { format: RawVideoFormat::Nv12, .. } => OutputFormat::Nv12,
        _ => OutputFormat::I420,
    }
}

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
    // AV1 decode via libdav1d (software, System memory): an auto-plug candidate
    // for AV1 -> I420, alongside av1parse.
    #[cfg(feature = "dav1d")]
    reg.register(ElementFactory::of::<Dav1dDec>("dav1ddec", |_| Box::new(Dav1dDec::new())));
    // Pure-Rust AV1 decode via re_rav1d (software, System memory): same AV1 -> I420
    // candidate. Negative rank so libdav1d (hand-written asm, faster) wins the
    // auto-plug tiebreak when both are built; rav1ddec is the portable fallback and
    // the sole AV1 decoder on pure-Rust targets.
    #[cfg(feature = "rav1d")]
    reg.register(ElementFactory::of::<Rav1dDec>("rav1ddec", |_| Box::new(Rav1dDec::new())).rank(-10));
    // Honor the output format the auto-plug search chose for this hop
    // (`ChainLink::output`): the source pad template advertises both NV12 and
    // I420, so a strict-NV12 sink (KMS / waylandsink) makes the search settle on
    // NV12, and the decoder must be built to emit it. Ignoring `out` here built a
    // fixed-I420 decoder under a chain promised NV12, so the runner's forward-caps
    // pre-fix (sink's NV12 accept-set) hit the decoder's `format != output_format`
    // arm and failed startup negotiation. Default to I420 for a loose target.
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register(ElementFactory::of::<FfmpegH264Dec>("ffmpegdec", |out| {
        Box::new(FfmpegH264Dec::new().with_output_format(ffmpegdec_output_format(out)))
    }));
    // AAC (and other libavcodec audio codecs) -> interleaved PcmS16Le (M422), the
    // audio sibling of ffmpegdec, in the auto-plug pool so a decode chain reaches
    // raw audio (e.g. an MPEG-TS / HLS AAC track).
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register(ElementFactory::of::<crate::ffmpegaudiodec::FfmpegAudioDec>("ffmpegaudiodec", |_| {
        Box::new(crate::ffmpegaudiodec::FfmpegAudioDec::new())
    }));
    // ffmpeg VAAPI hwaccel backend as a distinct name (M237). Same element type
    // as ffmpegdec, constructed with `Backend::Vaapi`; the libva device defaults
    // to the VA display's choice (a `device=` property is a follow-up).
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register(
        ElementFactory::of::<FfmpegH264Dec>("ffmpegvaapidec", |out| {
            Box::new(
                FfmpegH264Dec::new()
                    .with_backend(FfmpegBackend::Vaapi)
                    .with_output_format(ffmpegdec_output_format(out)),
            )
        })
        .hardware(),
    );
    #[cfg(all(target_os = "linux", feature = "vaapi"))]
    reg.register(
        ElementFactory::of::<VaapiH264Dec>("vaapidec", |_| Box::new(VaapiH264Dec::new())).hardware(),
    );
    // Native NVDEC (M270), registered last so a default (System-memory) auto-plug
    // still picks a CPU decoder: `NvDec` emits NV12 in CUDA device memory, which
    // caps geometry / format does not encode. M276 makes that domain a first-class
    // auto-plug feature: the factory is tagged `produces(Cuda)`, so the
    // domain-aware search (`decodebin_preferring(.., Cuda)`) prefers `NvDec` for a
    // GPU consumer, while a plain `decodebin` (preference `System`) is unchanged.
    #[cfg(all(target_os = "linux", feature = "nvdec"))]
    reg.register(
        ElementFactory::of::<NvDec>("nvdec", |_| Box::new(NvDec::new()))
            .produces(g2g_core::MemoryDomainKind::Cuda)
            .hardware(),
    );
    // Android hardware video decode via the NDK MediaCodec (M219/M302); one
    // factory per codec (the MIME is fixed at construction). Reachable from
    // g2g-launch on-device; the gst analog is `amcviddec-<component>`.
    #[cfg(all(target_os = "android", feature = "mediacodec"))]
    reg.register(
        ElementFactory::of::<MediaCodecDec>("mediacodecdec", |_| Box::new(MediaCodecDec::h264()))
            .hardware(),
    );
    #[cfg(all(target_os = "android", feature = "mediacodec"))]
    reg.register(
        ElementFactory::of::<MediaCodecDec>("mediacodecdech265", |_| Box::new(MediaCodecDec::h265()))
            .hardware(),
    );
    // Android hardware video encode via the NDK MediaCodec (M306); launch-only
    // (encoders are not auto-plug candidates), one factory per codec. The gst
    // analog is `amcvidenc-<component>`.
    #[cfg(all(target_os = "android", feature = "mediacodec"))]
    reg.register_launch(LaunchFactory::of::<MediaCodecEnc>("mediacodecenc", || {
        Box::new(MediaCodecEnc::h264())
    }));
    #[cfg(all(target_os = "android", feature = "mediacodec"))]
    reg.register_launch(LaunchFactory::of::<MediaCodecEnc>("mediacodecench265", || {
        Box::new(MediaCodecEnc::h265())
    }));
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
    // AV1 decode: gst's libav name -> the libdav1d decoder, falling back to the
    // pure-Rust re_rav1d decoder when only the `rav1d` feature is built.
    reg.register_alias("avdec_av1", &["dav1ddec", "rav1ddec"]);
    reg.register_alias("vah264dec", &["ffmpegvaapidec", "vaapidec"]);
    // VPx encoders: gst splits vp8enc / vp9enc; g2g has one vpxenc.
    reg.register_alias("vp8enc", &["vpxenc"]);
    reg.register_alias("vp9enc", &["vpxenc"]);
    // gst's libav software H.264 encoder name -> the ffmpeg encoder, software
    // first (`x264enc`, libx264), falling back to the NVENC-backed `ffmpegenc`
    // when only that is registered. The native NVENC encoder owns `nvh264enc`.
    reg.register_alias("avenc_h264", &["x264enc", "ffmpegenc"]);
    // QuickTime / MP4 muxer names -> the one fMP4 muxer (inert without std).
    reg.register_alias("qtmux", &["mp4mux"]);
    // gst's short AAC encoder name -> the libavcodec AAC encoder.
    reg.register_alias("aacenc", &["avenc_aac"]);
    // GStreamer's nvcodec names -> the native g2g NVENC / NVDEC elements. Resolve
    // to the registered target only when the feature is on (else the alias is
    // inert), the same first-registered rule as the VA-API names above.
    reg.register_alias("nvh264dec", &["nvdec"]);
    reg.register_alias("nvh264enc", &["nvenc"]);
    reg.register_alias("nvv4l2h264enc", &["nvenc"]);
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
    #[cfg(feature = "dav1d")]
    reg.register_launch(LaunchFactory::of::<Dav1dDec>("dav1ddec", || Box::new(Dav1dDec::new())));
    #[cfg(feature = "rav1d")]
    reg.register_launch(LaunchFactory::of::<Rav1dDec>("rav1ddec", || Box::new(Rav1dDec::new())));
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
    // ONVIF camera source: set the device service URL + account via
    // `onvifsrc location=... user=... password=...`. The H.264 output caps
    // match RtspSrc (the resolved RTSP stream the element delegates to).
    #[cfg(feature = "onvif")]
    reg.register_source(SourceFactory::new(
        "onvifsrc",
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        },
        || Box::new(OnvifSrc::new("")),
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
    // WebRTC WHEP ingest; the `location` property targets the endpoint. The URL
    // defaults empty (set it via `webrtcsrc location=...`); the handshake runs
    // when the source starts.
    #[cfg(feature = "webrtc")]
    reg.register_source(SourceFactory::new(
        "webrtcsrc",
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        },
        || Box::new(WebRtcWhepSrc::new("")),
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
    // libcamera capture: NV12 (else YUYV). Geometry/format are negotiated with
    // the camera at startup, so the declared caps are fully open.
    #[cfg(all(target_os = "linux", feature = "libcamera"))]
    reg.register_source(SourceFactory::new(
        "libcamerasrc",
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        },
        || Box::new(LibCameraSrc::new()),
    ));
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register_launch(LaunchFactory::of::<FfmpegH264Dec>("ffmpegdec", || {
        Box::new(FfmpegH264Dec::new())
    }));
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register_launch(LaunchFactory::of::<crate::ffmpegaudiodec::FfmpegAudioDec>(
        "ffmpegaudiodec",
        || Box::new(crate::ffmpegaudiodec::FfmpegAudioDec::new()),
    ));
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register_launch(LaunchFactory::of::<FfmpegH264Dec>("ffmpegvaapidec", || {
        Box::new(FfmpegH264Dec::new().with_backend(FfmpegBackend::Vaapi))
    }));
    // ffmpeg / libavcodec H.264 *encoder* (M266 / M274), the encode-side mirror of
    // ffmpegdec. `ffmpegenc` defaults to the NVENC backend (`h264_nvenc`); the
    // explicit `x264enc` name opens the libx264 software encoder for hosts without
    // an NVIDIA GPU. Launch-only: an encoder is never an auto-plug candidate.
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register_launch(LaunchFactory::of::<FfmpegH264Enc>("ffmpegenc", || {
        Box::new(FfmpegH264Enc::new())
    }));
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register_launch(LaunchFactory::of::<FfmpegH264Enc>("x264enc", || {
        Box::new(FfmpegH264Enc::new().with_backend(FfmpegEncBackend::Software))
    }));
    // libavcodec AAC-LC audio encoder (M292), the gst `avenc_aac` analog and the
    // Linux audio-encode path for the A/V muxers; the `aacenc` alias is added in
    // `register_aliases`.
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register_launch(LaunchFactory::of::<crate::ffmpegaacenc::FfmpegAacEnc>("avenc_aac", || {
        Box::new(crate::ffmpegaacenc::FfmpegAacEnc::new())
    }));
    #[cfg(all(target_os = "linux", feature = "vaapi"))]
    reg.register_launch(LaunchFactory::of::<VaapiH264Dec>("vaapidec", || {
        Box::new(VaapiH264Dec::new())
    }));
    // Native NVIDIA Video Codec SDK elements (M269 / M270): zero-copy CUDA NV12
    // <-> H.264, the gst-`nvcodec`-style pair. Explicit-select by name.
    #[cfg(all(target_os = "linux", feature = "nvdec"))]
    reg.register_launch(LaunchFactory::of::<NvDec>("nvdec", || Box::new(NvDec::new())));
    #[cfg(all(target_os = "linux", feature = "nvenc"))]
    reg.register_launch(LaunchFactory::of::<NvEnc>("nvenc", || Box::new(NvEnc::new())));
    #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
    reg.register_launch(LaunchFactory::of::<crate::dmabufwgpu::DmaBufToWgpu>("dmabuftowgpu", || {
        Box::new(crate::dmabufwgpu::DmaBufToWgpu::new())
    }));
    // Reverse GStreamer bridge: host an unported GStreamer element in a g2g graph.
    // No pad templates (caps are what the negotiation settles + the `output-caps`
    // property declares), like `identity`.
    #[cfg(feature = "gstreamer")]
    reg.register_launch(LaunchFactory::new("gstwrap", Vec::new(), || {
        Box::new(crate::gstwrap::GstWrap::new())
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
    // Android AAudio PCM render (M307); the gst analog is `aaudiosink`.
    #[cfg(all(target_os = "android", feature = "aaudio"))]
    reg.register_launch(LaunchFactory::of::<AAudioSink>("aaudiosink", || Box::new(AAudioSink::new())));
}

#[cfg(all(test, target_os = "linux", feature = "nvenc", feature = "nvdec"))]
mod nv_registry_tests {
    use super::*;

    /// The native NVENC / NVDEC elements (M269 / M270) and their gst-canonical
    /// aliases resolve to constructible elements. `new()` touches no CUDA (the
    /// session / context open lazily at configure), so this runs without a GPU.
    #[test]
    fn nvcodec_elements_and_aliases_resolve() {
        let reg = default_registry();
        for name in ["nvenc", "nvdec", "nvh264enc", "nvh264dec", "nvv4l2h264enc"] {
            assert!(reg.make_element(name).is_some(), "registry resolves `{name}`");
        }
        // The native decoder is also an auto-plug candidate (registered after the
        // CPU decoders so it does not out-rank them; see register_autoplug_candidates).
        assert!(reg.element_names().contains(&"nvdec"), "nvdec is an autoplug factory");
    }
}

#[cfg(all(test, target_os = "linux", feature = "nvdec", feature = "ffmpeg"))]
mod domain_aware_autoplug_tests {
    use super::*;
    use g2g_core::runtime::is_raw_video;
    use g2g_core::{Caps, Dim, MemoryDomainKind, Rate, VideoCodec};

    fn h264() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    /// M276: the memory feature gates auto-plug by domain. A default (System)
    /// decode of H.264 stays on the CPU decoder; requesting `Cuda` prefers the
    /// native NVDEC. Needs no GPU (the search reads pad-template + feature
    /// metadata; nothing is constructed or run).
    #[test]
    fn cuda_preference_selects_nvdec_over_cpu_decoder() {
        let reg = default_registry();
        // Default selection: NvDec (registered last, tagged Cuda) does not hijack
        // the system-memory path; the CPU decoder is chosen.
        let cpu = reg.autoplug_names(&h264(), &is_raw_video, 4).expect("a decoder reaches raw");
        assert_eq!(cpu.last(), Some(&"ffmpegdec"), "default decode stays on the CPU: {cpu:?}");
        // Cuda preference: the domain-aware search prefers the native NVDEC.
        let gpu = reg
            .autoplug_names_preferring(&h264(), &is_raw_video, 4, MemoryDomainKind::Cuda)
            .expect("a decoder reaches raw");
        assert_eq!(gpu.last(), Some(&"nvdec"), "Cuda preference prefers NvDec: {gpu:?}");
    }
}

#[cfg(all(test, target_os = "linux", feature = "ffmpeg"))]
mod ffmpeg_enc_registry_tests {
    use super::*;

    /// The ffmpeg H.264 encoder (M266 / M274) resolves under its native name, the
    /// software `x264enc` name, and the gst `avenc_h264` alias. `new()` opens no
    /// libavcodec context (that happens at configure), so this needs no GPU.
    #[test]
    fn ffmpeg_encoder_and_alias_resolve() {
        let reg = default_registry();
        for name in ["ffmpegenc", "x264enc", "avenc_h264"] {
            assert!(reg.make_element(name).is_some(), "registry resolves `{name}`");
        }
    }
}

#[cfg(test)]
mod muxer_alias_tests {
    use super::*;

    /// `qtmux` aliases `mp4mux`; resolving it as a fan-in muxer lets an A/V
    /// pipeline written `... qtmux name=m` build the multi-input MP4 muxer.
    #[test]
    fn qtmux_alias_resolves_as_a_fan_in_muxer() {
        let reg = default_registry();
        assert!(reg.make_muxer("qtmux", 2).is_some(), "qtmux resolves to the mp4mux fan-in");
        assert!(reg.make_muxer("mp4mux", 2).is_some(), "the alias target still builds directly");
    }

    #[test]
    fn dual_registered_muxers_are_listed_once() {
        let reg = default_registry();
        let names = reg.element_names();
        let mut seen = alloc::collections::BTreeSet::new();
        for n in &names {
            assert!(seen.insert(*n), "element `{n}` listed more than once");
        }
        // mp4mux is registered as both a launch element and a fan-in muxer.
        assert_eq!(names.iter().filter(|n| **n == "mp4mux").count(), 1);
    }
}

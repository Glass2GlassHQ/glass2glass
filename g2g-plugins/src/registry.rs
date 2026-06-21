//! A pre-populated element [`Registry`] (M107), so a `gst-launch` text pipeline
//! and `gst-inspect` work out of the box without the caller hand-registering
//! every element.
//!
//! [`default_registry`] registers the standard `no_std`-baseline elements under
//! their conventional names: the test sources, the video and audio transform
//! chains, and the `fakesink` / `filesink` sinks. Each is default-constructed and
//! then configured by the parser from its `key=value` properties (M104/M106).
//!
//! `std`-only (the `Registry` is). Feature-gated capture / decode / display
//! elements (`v4l2src`, `ffmpeg`, `waylandsink`, ...) are not registered here yet;
//! a caller adds them to the returned registry with `register_source` /
//! `register_launch` as their features are enabled. `filesrc` is registered
//! (M112): its `bytestream-format` property supplies the container type a raw
//! byte stream lacks, so `filesrc location=x.ts bytestream-format=mpegts ! tsdemux`
//! works as text.

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::runtime::{LaunchFactory, MuxerFactory, Registry, SourceFactory};
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
#[cfg(feature = "http-src")]
use crate::httpsrc::HttpSrc;
#[cfg(feature = "hls")]
use crate::hlssrc::HlsSrc;
#[cfg(feature = "dash")]
use crate::dashsrc::DashSrc;
#[cfg(feature = "rtmp")]
use crate::rtmpsrc::RtmpSrc;
#[cfg(all(target_os = "linux", feature = "wayland-sink"))]
use crate::waylandsink::WaylandSink;
#[cfg(all(target_os = "linux", feature = "alsa-sink"))]
use crate::alsasink::AlsaSink;
#[cfg(all(target_os = "linux", feature = "pulse-sink"))]
use crate::pulsesink::PulseSink;
#[cfg(all(target_os = "linux", feature = "v4l2"))]
use crate::v4l2src::V4l2Src;
#[cfg(all(target_os = "linux", feature = "kms-sink"))]
use crate::kmssink::KmsSink;
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
use crate::ffmpegdec::FfmpegH264Dec;
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

    // Video transforms.
    reg.register_launch(LaunchFactory::of::<VideoConvert>("videoconvert", || {
        Box::new(VideoConvert::new(RawVideoFormat::Rgba8))
    }));
    reg.register_launch(LaunchFactory::of::<VideoScale>("videoscale", || {
        Box::new(VideoScale::new(0, 0))
    }));
    reg.register_launch(LaunchFactory::of::<VideoCrop>("videocrop", || {
        Box::new(VideoCrop::new(0, 0, 0, 0))
    }));
    reg.register_launch(LaunchFactory::of::<VideoFlip>("videoflip", || {
        Box::new(VideoFlip::new(FlipMethod::HorizontalMirror))
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
        Box::new(AudioResample::new(48_000))
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

    // Sinks.
    reg.register_launch(LaunchFactory::of::<FakeSink>("fakesink", || Box::new(FakeSink::new())));
    reg.register_launch(LaunchFactory::of::<FileSink>("filesink", || Box::new(FileSink::new(""))));

    register_feature_gated(&mut reg);

    reg
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
    #[cfg(all(target_os = "linux", feature = "vaapi"))]
    reg.register_launch(LaunchFactory::of::<VaapiH264Dec>("vaapidec", || {
        Box::new(VaapiH264Dec::new())
    }));
    #[cfg(all(target_os = "linux", feature = "wayland-sink"))]
    reg.register_launch(LaunchFactory::new("waylandsink", Vec::new(), || {
        Box::new(WaylandSink::new())
    }));
    #[cfg(all(target_os = "linux", feature = "kms-sink"))]
    reg.register_launch(LaunchFactory::new("kmssink", Vec::new(), || Box::new(KmsSink::new())));
    #[cfg(all(target_os = "linux", feature = "alsa-sink"))]
    reg.register_launch(LaunchFactory::of::<AlsaSink>("alsasink", || Box::new(AlsaSink::new())));
    #[cfg(all(target_os = "linux", feature = "pulse-sink"))]
    reg.register_launch(LaunchFactory::of::<PulseSink>("pulsesink", || Box::new(PulseSink::new())));
}

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
//! `register_feature_gated`, each block `#[cfg]`-gated like its module, so they
//! appear in `gst-inspect` / `parse_launch` when their feature is enabled.
//! `filesrc` is registered (M112): its `bytestream-format` property supplies the
//! container type a raw byte stream lacks, so `filesrc location=x.ts
//! bytestream-format=mpegts ! tsdemux` works as text.

use alloc::boxed::Box;
use alloc::vec::Vec;

#[cfg(feature = "script-rhai")]
use g2g_core::runtime::DemuxFactory;
use g2g_core::runtime::{
    ElementFactory, EncoderChoice, LaunchFactory, MuxerFactory, Registry, SourceFactory,
};
use g2g_core::{AudioFormat, ByteStreamEncoding, Caps, Dim, Rate, RawVideoFormat};

use crate::aacparse::AacParse;
use crate::ac3parse::Ac3Parse;
use crate::adpcm::{AdpcmDec, AdpcmEnc};
use crate::aiff::{AiffMux, AiffParse};
use crate::alpha::Alpha;
use crate::apedemux::ApeDemux;
use crate::au::{AuMux, AuParse};
use crate::audioconvert::AudioConvert;
use crate::audiomixer::AudioMixer;
use crate::audiopanorama::AudioPanorama;
use crate::audiorate::AudioRate;
use crate::audioresample::AudioResample;
use crate::audiotestsrc::AudioTestSrc;
use crate::av1parse::Av1Parse;
use crate::capsfilter::CapsFilter;
use crate::colorspace::Colorspace;
use crate::compositor::{Compositor, CompositorPad};
use crate::debugspy::DebugSpy;
use crate::downloadbuffer::DownloadBuffer;
use crate::dtmf::{DtmfDetect, DtmfSrc};
use crate::fakesink::FakeSink;
use crate::fakesrc::FakeSrc;
use crate::filesink::FileSink;
use crate::filesrc::FileSrc;
use crate::flacparse::FlacParse;
use crate::flvdemux::FlvDemux;
use crate::flvmux::FlvMux;
use crate::g711::{AlawDec, AlawEnc, MulawDec, MulawEnc};
use crate::gaudieffects::{Burn, Chromium, Dilate, Dodge, Exclusion, Solarize};
use crate::h264parse::H264Parse;
use crate::h265parse::H265Parse;
use crate::hsv::{HsvDetector, HsvFilter};
use crate::id3demux::Id3Demux;
use crate::identity::IdentityTransform;
use crate::ivfdemux::IvfDemux;
use crate::mkvdemux::MkvDemux;
use crate::mkvmux::MkvMux;
#[cfg(feature = "std")]
use crate::mp4mux::Mp4Mux;
use crate::mpeg4videoparse::Mpeg4VideoParse;
use crate::mpegaudioparse::MpegAudioParse;
use crate::mpegvideoparse::MpegVideoParse;
use crate::mux::InterleaveMux;
use crate::oggdemux::OggDemux;
use crate::oggmux::OggMux;
use crate::opusparse::OpusParse;
use crate::pnm::{PnmDec, PnmEnc};
use crate::rawaudioparse::RawAudioParse;
use crate::rawvideoparse::RawVideoParse;
use crate::record::{RecordSink, ReplaySrc};
use crate::roundedcorners::RoundedCorners;
use crate::scenechange::SceneChange;
use crate::stillparse::{JpegParse, PngParse};
use crate::tensorconvert::TensorConvert;
use crate::textoverlay::TextOverlay;
use crate::tonegeneratesrc::ToneGenerateSrc;
use crate::tsdemux::TsDemux;
use crate::tsmux::TsMux;
use crate::valve::Valve;
use crate::vc1parse::Vc1Parse;
use crate::videoanalyse::VideoAnalyse;
use crate::videobalance::VideoBalance;
use crate::videobox::VideoBox;
use crate::videoconvert::VideoConvert;
use crate::videoconvertscale::VideoConvertScale;
use crate::videocrop::VideoCrop;
use crate::videoflip::{Orientation, VideoFlip};
use crate::videorate::VideoRate;
use crate::videoscale::VideoScale;
use crate::videotestsrc::VideoTestSrc;
use crate::volume::Volume;
use crate::vp8parse::Vp8Parse;
use crate::vp9parse::Vp9Parse;
use crate::wavenc::WavEnc;
use crate::wavparse::WavParse;
use crate::y4m::{Y4mDec, Y4mEnc};

// Feature- (and platform-) gated elements, registered when their feature is on so
// `gst-inspect`, `gst-inspect --all`, and `parse_launch` see them. Each registers
// exactly as its `#[cfg]` in `lib.rs` gates the module.
#[cfg(all(target_os = "android", feature = "aaudio"))]
use crate::aaudio::{AAudioSink, AAudioSrc};
#[cfg(all(target_os = "linux", feature = "alsa-sink"))]
use crate::alsasink::AlsaSink;
#[cfg(all(target_os = "linux", feature = "alsa-src"))]
use crate::alsasrc::AlsaSrc;
#[cfg(feature = "av1-encode")]
use crate::av1enc::Av1Enc;
#[cfg(all(target_os = "android", feature = "camera2"))]
use crate::camera2src::Camera2Src;
#[cfg(feature = "dash")]
use crate::dashsrc::DashSrc;
#[cfg(feature = "dav1d")]
use crate::dav1ddec::Dav1dDec;
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
use crate::ffmpegdec::{Backend as FfmpegBackend, FfmpegH264Dec};
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
use crate::ffmpegenc::{Backend as FfmpegEncBackend, FfmpegH264Enc};
use crate::fmp4demux::Fmp4Demux;
#[cfg(feature = "hls")]
use crate::hlssrc::HlsSrc;
#[cfg(feature = "http-src")]
use crate::httpsrc::HttpSrc;
#[cfg(all(target_os = "linux", feature = "kms-sink"))]
use crate::kmssink::KmsSink;
#[cfg(all(target_os = "linux", feature = "libcamera"))]
use crate::libcamerasrc::LibCameraSrc;
#[cfg(all(target_os = "linux", feature = "local-ipc"))]
use crate::localcuda::{LocalCudaSink, LocalCudaSrc};
#[cfg(all(target_os = "linux", feature = "local-dmabuf"))]
use crate::localdmabuf::{DmaBufSink, DmaBufSrc};
#[cfg(all(target_os = "android", feature = "mediacodec"))]
use crate::mediacodecdec::MediaCodecDec;
#[cfg(all(target_os = "android", feature = "mediacodec"))]
use crate::mediacodecenc::MediaCodecEnc;
#[cfg(feature = "mjpeg")]
use crate::mjpegdec::MjpegDec;
#[cfg(feature = "mjpeg-encode")]
use crate::mjpegenc::MjpegEnc;
#[cfg(feature = "moqt")]
use crate::moqtsink::MoqtSink;
#[cfg(feature = "moqt")]
use crate::moqtsrc::MoqtSrc;
use crate::mp4demux::Mp4Demux;
#[cfg(all(target_os = "linux", feature = "nvdec"))]
use crate::nvdec::NvDec;
#[cfg(all(target_os = "linux", feature = "nvenc"))]
use crate::nvenc::NvEnc;
#[cfg(feature = "onvif")]
use crate::onvif::OnvifSrc;
#[cfg(all(target_os = "linux", feature = "pipewire"))]
use crate::pipewiresink::PipeWireSink;
#[cfg(all(target_os = "linux", feature = "pipewire"))]
use crate::pipewiresrc::PipeWireSrc;
#[cfg(all(target_os = "linux", feature = "pipewire"))]
use crate::pipewirevideosrc::PipeWireVideoSrc;
#[cfg(feature = "png")]
use crate::pngdec::PngDec;
#[cfg(feature = "png")]
use crate::pngenc::PngEnc;
#[cfg(all(target_os = "linux", feature = "pulse-sink"))]
use crate::pulsesink::PulseSink;
#[cfg(all(target_os = "linux", feature = "pulse-src"))]
use crate::pulsesrc::PulseSrc;
#[cfg(feature = "rav1d")]
use crate::rav1ddec::Rav1dDec;
#[cfg(feature = "remote")]
use crate::remotesink::RemoteSink;
#[cfg(feature = "remote")]
use crate::remotesrc::RemoteSrc;
#[cfg(feature = "remote-ws")]
use crate::remotewssink::RemoteWsSink;
#[cfg(feature = "remote-ws")]
use crate::remotewssrc::RemoteWsSrc;
#[cfg(feature = "remote-ws")]
use crate::remotewstransform::RemoteWsTransform;
#[cfg(feature = "webtransport")]
use crate::remotewtsink::RemoteWtSink;
#[cfg(feature = "webtransport")]
use crate::remotewtsrc::RemoteWtSrc;
#[cfg(feature = "webtransport")]
use crate::remotewttransform::RemoteWtTransform;
#[cfg(feature = "rtmp")]
use crate::rtmpsink::RtmpSink;
#[cfg(feature = "rtmp")]
use crate::rtmpsrc::RtmpSrc;
#[cfg(feature = "rtsp-server")]
use crate::rtspserversink::RtspServerSink;
#[cfg(feature = "rtsp-server")]
use crate::rtspserversrc::RtspServerSrc;
#[cfg(feature = "rtsp")]
use crate::rtspsrc::RtspSrc;
use crate::scaletempo::ScaleTempo;
#[cfg(all(unix, feature = "shm"))]
use crate::shm::{ShmSink, ShmSrc};
#[cfg(feature = "srtp")]
use crate::srtpdec::SrtpDec;
#[cfg(feature = "srtp")]
use crate::srtpenc::SrtpEnc;
#[cfg(feature = "srt")]
use crate::srtsink::SrtSink;
#[cfg(feature = "srt")]
use crate::srtsrc::SrtSrc;
#[cfg(all(target_os = "linux", feature = "jpegxs"))]
use crate::svtjpegxs::{SvtJpegXsDec, SvtJpegXsEnc};
#[cfg(feature = "tcp")]
use crate::tcp::{TcpClientSink, TcpClientSrc, TcpServerSink, TcpServerSrc};
#[cfg(feature = "udp-egress")]
use crate::udpsink::UdpSink;
#[cfg(feature = "udp-ingress")]
use crate::udpsrc::UdpSrc;
#[cfg(all(target_os = "linux", feature = "v4l2"))]
use crate::v4l2src::V4l2Src;
#[cfg(all(target_os = "linux", feature = "vaapi"))]
use crate::vaapidec::{VaapiH264Dec, VaapiH265Dec};
#[cfg(feature = "vorbis")]
use crate::vorbisdec::VorbisDec;
#[cfg(feature = "vpx")]
use crate::vpxenc::VpxEnc;
#[cfg(all(target_os = "linux", feature = "wayland-sink"))]
use crate::waylandsink::WaylandSink;
#[cfg(feature = "webp")]
use crate::webpdec::WebPDec;
#[cfg(feature = "webrtc")]
use crate::webrtcsink::WebRtcSink;
#[cfg(feature = "webrtc")]
use crate::webrtcwhepsrc::WebRtcWhepSrc;
#[cfg(feature = "opus")]
use crate::{opusdec::OpusDec, opusenc::OpusEnc};

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
/// of it, the way GStreamer's `decodebin` always inserts a parser. Names the
/// registered launch element (M676: the name-based `decodebin` expansion in
/// `parse_launch` shares this mapping; both launch registrations construct the
/// re-framing form). Returns `None` for codecs without a re-framing parser (the
/// input decodes directly). H.264 (M421) and H.265 (M425) re-frame to one access
/// unit per packet; FLAC frame-aligns via `flacparse` (M775, a bare `.flac` byte
/// stream carries no frame lengths), and MPEG audio, AAC and AC-3 frame-align via
/// `mpegaudioparse` / `aacparse` / `ac3parse` (M1065, M1074: one self-syncing
/// frame per packet); other audio decodes directly. JPEG and PNG frame into
/// whole images via `jpegparse` / `pngparse` (M1087), which the still-image
/// decoders need: they take one complete image per buffer.
fn decode_parser_provider(input: &Caps) -> Option<&'static str> {
    match input {
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            ..
        } => Some("h264parse"),
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H265,
            ..
        } => Some("h265parse"),
        Caps::Audio {
            format: AudioFormat::Flac,
            ..
        } => Some("flacparse"),
        Caps::Audio {
            format: AudioFormat::Mp2 | AudioFormat::Mp3,
            ..
        } => Some("mpegaudioparse"),
        Caps::Audio {
            format: AudioFormat::Aac,
            ..
        } => Some("aacparse"),
        Caps::Audio {
            format: AudioFormat::Ac3,
            ..
        } => Some("ac3parse"),
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::Mjpeg,
            ..
        } => Some("jpegparse"),
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::Png,
            ..
        } => Some("pngparse"),
        _ => None,
    }
}

/// gst `imagesequencesrc`'s default output rate.
const IMAGE_SEQUENCE_FPS: u32 = 30;

/// Encoders that produce each coded stream an encoding profile can name, most
/// preferred first (hardware before software, native before FFI), for the
/// `encodebin` expansion. The registry takes the first candidate this build
/// compiled in; the props pin a multi-codec encoder to that codec.
fn encode_provider(target: &Caps) -> Option<&'static [EncoderChoice]> {
    use g2g_core::VideoCodec as V;
    static H264: &[EncoderChoice] = &[
        EncoderChoice::with_props("nvenc", &[("codec", "h264")]),
        EncoderChoice::plain("vtenc_h264"),
        EncoderChoice::plain("mediacodecenc"),
        EncoderChoice::plain("x264enc"),
        EncoderChoice::plain("ffmpegenc"),
    ];
    static H265: &[EncoderChoice] = &[
        EncoderChoice::with_props("nvenc", &[("codec", "hevc")]),
        EncoderChoice::plain("vtenc_h265"),
        EncoderChoice::plain("mediacodecench265"),
    ];
    static VP8: &[EncoderChoice] = &[EncoderChoice::with_props("vpxenc", &[("codec", "vp8")])];
    static VP9: &[EncoderChoice] = &[EncoderChoice::with_props("vpxenc", &[("codec", "vp9")])];
    static AV1: &[EncoderChoice] = &[EncoderChoice::plain("av1enc")];
    static MJPEG: &[EncoderChoice] = &[EncoderChoice::plain("mjpegenc")];
    static PNG: &[EncoderChoice] = &[EncoderChoice::plain("pngenc")];
    static PNM: &[EncoderChoice] = &[EncoderChoice::plain("pnmenc")];
    static JPEGXS: &[EncoderChoice] = &[EncoderChoice::plain("jpegxsenc")];
    static AAC: &[EncoderChoice] = &[EncoderChoice::plain("avenc_aac")];
    static OPUS: &[EncoderChoice] = &[EncoderChoice::plain("opusenc")];
    static MULAW: &[EncoderChoice] = &[EncoderChoice::plain("mulawenc")];
    static ALAW: &[EncoderChoice] = &[EncoderChoice::plain("alawenc")];
    static ADPCM: &[EncoderChoice] = &[EncoderChoice::plain("adpcmenc")];
    Some(match target {
        Caps::CompressedVideo { codec, .. } => match codec {
            V::H264 => H264,
            V::H265 => H265,
            V::Vp8 => VP8,
            V::Vp9 => VP9,
            V::Av1 => AV1,
            V::Mjpeg => MJPEG,
            V::Png => PNG,
            V::Pnm => PNM,
            V::JpegXs => JPEGXS,
            _ => return None,
        },
        Caps::Audio { format, .. } => match format {
            AudioFormat::Aac => AAC,
            AudioFormat::Opus => OPUS,
            AudioFormat::Mulaw => MULAW,
            AudioFormat::Alaw => ALAW,
            AudioFormat::ImaAdpcm => ADPCM,
            _ => return None,
        },
        _ => return None,
    })
}

/// Muxers that write each container an encoding profile can name, for the
/// `encodebin` expansion. IVF and the raw byte stream have no writer here, so a
/// profile naming one fails loud.
fn container_muxer_provider(container: &Caps) -> Option<&'static [&'static str]> {
    let Caps::ByteStream { encoding } = container else {
        return None;
    };
    Some(match encoding {
        ByteStreamEncoding::MpegTs => &["mpegtsmux"],
        ByteStreamEncoding::Matroska => &["matroskamux"],
        ByteStreamEncoding::Mp4 | ByteStreamEncoding::IsoBmff => &["mp4mux"],
        ByteStreamEncoding::Avi => &["avimux"],
        ByteStreamEncoding::Ogg => &["oggmux"],
        ByteStreamEncoding::Flv => &["flvmux"],
        ByteStreamEncoding::Wav => &["wavenc"],
        ByteStreamEncoding::Aiff => &["aiffmux"],
        ByteStreamEncoding::Au => &["avmux_au"],
        ByteStreamEncoding::Y4m => &["y4menc"],
        ByteStreamEncoding::Multipart => &["multipartmux"],
        _ => return None,
    })
}

pub fn default_registry() -> Registry {
    let mut reg = Registry::new();
    // Auto-plugged decode chains splice a re-framing parser before the decoder
    // (M421), so a decoder fed un-access-unit-aligned input (e.g. one MPEG-TS PES
    // that is not one coded picture) does not mis-parse.
    reg.set_parser_provider(decode_parser_provider);
    // M1089: `encodebin profile=` picks its encoders and muxer through these.
    reg.set_encoder_provider(encode_provider);
    reg.set_muxer_provider(container_muxer_provider);
    // A parsed pipeline whose producer and consumer disagree on a memory domain
    // gets the bridge spliced in (M354): `nvdec ! wgpusink` keeps the frame on
    // the GPU, `nvdec ! waylandsink` downloads it.
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    reg.set_domain_converter(crate::cuda::cuda_domain_converter);

    // Sources. The output caps are the autoplug `decodebin` input; the parser
    // only calls the constructor and applies properties.
    reg.register_source(SourceFactory::new(
        "videotestsrc",
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        },
        // num-buffers defaults to forever (the property's documented `-1`),
        // matching gst videotestsrc; a launch line bounds it with `num-buffers=N`.
        || Box::new(VideoTestSrc::new(320, 240, 30, u64::MAX)),
    ));
    reg.register_source(SourceFactory::new(
        "audiotestsrc",
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        },
        // num-buffers defaults to forever (the property's documented `-1`),
        // matching gst audiotestsrc; a launch line bounds it with `num-buffers=N`.
        || Box::new(AudioTestSrc::new(48_000, 2, 440, u64::MAX)),
    ));
    // `tonegeneratesrc`: 44100 Hz stereo sine at `freq` / `volume`.
    reg.register_source(SourceFactory::new(
        "tonegeneratesrc",
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 44_100,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        },
        || Box::new(ToneGenerateSrc::new()),
    ));
    // `dtmfsrc`: 8 kHz mono DTMF, packetized at `interval` ms.
    reg.register_source(SourceFactory::new(
        "dtmfsrc",
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 1,
            sample_rate: 8_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        },
        || Box::new(DtmfSrc::new()),
    ));
    // Android AAudio mic capture (M307); the device may open with different
    // actuals, reported as the produced caps. `aaudiosrc` is the gst analog.
    #[cfg(all(target_os = "android", feature = "aaudio"))]
    reg.register_source(
        SourceFactory::new(
            "aaudiosrc",
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 48_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            },
            || Box::new(AAudioSrc::new(48_000, 2, u64::MAX)),
        )
        .with_experimental(),
    );
    // Android camera capture (M308); 640x480 NV12 default. `camerasrc` /
    // `ahcsrc` are the gst analogs.
    #[cfg(all(target_os = "android", feature = "camera2"))]
    reg.register_source(
        SourceFactory::new(
            "camera2src",
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            || Box::new(Camera2Src::new(640, 480, u64::MAX)),
        )
        .with_experimental(),
    );
    // The output caps are a nominal default; a bare launch `filesrc` derives its
    // type from the `location` extension (M478), and the `bytestream-format`
    // property (incl. `auto`) overrides that per instance before negotiation.
    reg.register_source(SourceFactory::new(
        "filesrc",
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
        || Box::new(FileSrc::untyped()),
    ));
    // Synthetic byte source (M1070): `fakesrc num-buffers=20 sizemax=4096 ! ...`
    // drives a graph without a file, a device or a network. Same byte-stream type
    // as an untyped `filesrc`, so a `typefind` after it behaves the same.
    reg.register_source(SourceFactory::new(
        "fakesrc",
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
        || Box::new(FakeSrc::new()),
    ));
    // File-descriptor source (M1070): reads an already-open descriptor, so a
    // pipeline can sit in a shell pipe (`fdsrc fd=0 ! typefind ! decodebin`).
    // Unix only, like the module.
    #[cfg(unix)]
    reg.register_source(SourceFactory::new(
        "fdsrc",
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
        || Box::new(crate::fd::FdSrc::default()),
    ));
    // Subtitle / text file source (M433): a `.srt` / `.vtt` / `.ssa` / `.ttml`
    // file as a `Text` stream, feeding `subparse` (overlay or caption authoring).
    // The `format` is sniffed from the `location` extension unless set explicitly.
    reg.register_source(SourceFactory::new(
        "subtitlesrc",
        Caps::Text {
            format: g2g_core::TextFormat::Srt,
        },
        || {
            Box::new(crate::subtitlesrc::SubtitleSrc::new(
                "",
                g2g_core::TextFormat::Srt,
            ))
        },
    ));
    // VobSub sidecar source (M926): a DVD subtitle `.idx` / `.sub` pair sitting
    // next to a video, e.g. `vobsubsrc location=movie.idx ! vobsubdec ! c.` .
    // `sub-location` overrides the derived `.sub` path, `language` picks one of
    // an `.idx`'s indexed languages.
    reg.register_source(SourceFactory::new(
        "vobsubsrc",
        Caps::SubPicture {
            format: g2g_core::SubPictureFormat::VobSub,
        },
        || Box::new(crate::vobsubsrc::VobSubSrc::new("")),
    ));
    // Image-sequence source (M-gap): reads img%05d.jpg style sequences, Motion-JPEG
    // by default so `multifilesrc location=img%05d.jpg ! mjpegdec ! ...` works.
    reg.register_source(SourceFactory::new(
        "multifilesrc",
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::Mjpeg,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        },
        || Box::new(crate::multifilesrc::MultiFileSrc::new("")),
    ));
    // Image-sequence source with a stated rate (M1088): the same file walk as
    // `multifilesrc`, stamped on a framerate grid so a folder of stills plays
    // as a clip. gst's default pattern and rate.
    reg.register_source(SourceFactory::new(
        "imagesequencesrc",
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::Mjpeg,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Fixed(IMAGE_SEQUENCE_FPS << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        },
        || {
            Box::new(
                crate::multifilesrc::MultiFileSrc::new("%05d")
                    .with_framerate(IMAGE_SEQUENCE_FPS, 1),
            )
        },
    ));
    // Split-file source (M1088): the parts of one recording read back as one
    // byte stream. The declared caps are a placeholder: the real type comes from
    // the first part's extension at negotiation.
    reg.register_source(SourceFactory::new(
        "splitfilesrc",
        Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::MpegTs,
        },
        || Box::new(crate::splitfilesrc::SplitFileSrc::new("")),
    ));
    // `data:` URI source (M1088): the payload inside the URI, typed by sniffing
    // it. The declared caps are a placeholder, replaced once the URI is read.
    reg.register_source(SourceFactory::new(
        "dataurisrc",
        Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::Raw,
        },
        || Box::new(crate::dataurisrc::DataUriSrc::new("")),
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
    // Colorimetry converter (M1127): a bare `colorspace` takes its target from a
    // downstream capsfilter, or passes through.
    reg.register_launch(LaunchFactory::of::<Colorspace>("colorspace", || {
        Box::new(Colorspace::new())
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
    // Convert and scale in one element: takes both from a downstream capsfilter
    // unless the `format` / `width` / `height` properties pin them.
    reg.register_launch(LaunchFactory::of::<VideoConvertScale>(
        "videoconvertscale",
        || Box::new(VideoConvertScale::auto()),
    ));
    reg.register_launch(LaunchFactory::of::<WavEnc>("wavenc", || {
        Box::new(WavEnc::new())
    }));
    reg.register_launch(LaunchFactory::of::<WavParse>("wavparse", || {
        Box::new(WavParse::new())
    }));
    reg.register_launch(LaunchFactory::of::<AiffMux>("aiffmux", || {
        Box::new(AiffMux::new())
    }));
    reg.register_launch(LaunchFactory::of::<AiffParse>("aiffparse", || {
        Box::new(AiffParse::new())
    }));
    reg.register_launch(LaunchFactory::of::<AuMux>("avmux_au", || {
        Box::new(AuMux::new())
    }));
    reg.register_launch(LaunchFactory::of::<AuParse>("auparse", || {
        Box::new(AuParse::new())
    }));
    reg.register_launch(LaunchFactory::of::<Y4mEnc>("y4menc", || {
        Box::new(Y4mEnc::new())
    }));
    reg.register_launch(LaunchFactory::of::<Y4mDec>("y4mdec", || {
        Box::new(Y4mDec::new())
    }));
    reg.register_launch(LaunchFactory::of::<crate::multipart::MultipartDemux>(
        "multipartdemux",
        || Box::new(crate::multipart::MultipartDemux::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::multipart::MultipartMux>(
        "multipartmux",
        || Box::new(crate::multipart::MultipartMux::new()),
    ));
    reg.register_launch(LaunchFactory::of::<VideoCrop>("videocrop", || {
        Box::new(VideoCrop::new(0, 0, 0, 0))
    }));
    reg.register_launch(LaunchFactory::of::<VideoFlip>("videoflip", || {
        Box::new(VideoFlip::new(Orientation::Identity))
    }));
    reg.register_launch(LaunchFactory::of::<VideoBalance>("videobalance", || {
        Box::new(VideoBalance::new())
    }));
    // Software video effects (M1084).
    reg.register_launch(
        LaunchFactory::of::<crate::aspectratiocrop::AspectRatioCrop>("aspectratiocrop", || {
            Box::new(crate::aspectratiocrop::AspectRatioCrop::new())
        }),
    );
    reg.register_launch(LaunchFactory::of::<crate::chromahold::ChromaHold>(
        "chromahold",
        || Box::new(crate::chromahold::ChromaHold::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::coloreffects::ColorEffects>(
        "coloreffects",
        || Box::new(crate::coloreffects::ColorEffects::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::gaussianblur::GaussianBlur>(
        "gaussianblur",
        || Box::new(crate::gaussianblur::GaussianBlur::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::smooth::Smooth>("smooth", || {
        Box::new(crate::smooth::Smooth::new())
    }));
    reg.register_launch(LaunchFactory::of::<crate::videodiff::VideoDiff>(
        "videodiff",
        || Box::new(crate::videodiff::VideoDiff::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::videomedian::VideoMedian>(
        "videomedian",
        || Box::new(crate::videomedian::VideoMedian::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::zebrastripe::ZebraStripe>(
        "zebrastripe",
        || Box::new(crate::zebrastripe::ZebraStripe::new()),
    ));
    reg.register_launch(LaunchFactory::of::<VideoAnalyse>("videoanalyse", || {
        Box::new(VideoAnalyse::new())
    }));
    reg.register_launch(LaunchFactory::of::<SceneChange>("scenechange", || {
        Box::new(SceneChange::new())
    }));
    // `dtmfdetect`: 8 kHz mono Goertzel.
    reg.register_launch(LaunchFactory::of::<DtmfDetect>("dtmfdetect", || {
        Box::new(DtmfDetect::new())
    }));
    reg.register_launch(LaunchFactory::of::<PnmEnc>("pnmenc", || {
        Box::new(PnmEnc::new())
    }));
    reg.register_launch(LaunchFactory::of::<PnmDec>("pnmdec", || {
        Box::new(PnmDec::new())
    }));
    // GStreamer gaudieffects (M1103).
    reg.register_launch(LaunchFactory::of::<Solarize>("solarize", || {
        Box::new(Solarize::new())
    }));
    reg.register_launch(LaunchFactory::of::<Chromium>("chromium", || {
        Box::new(Chromium::new())
    }));
    reg.register_launch(LaunchFactory::of::<Exclusion>("exclusion", || {
        Box::new(Exclusion::new())
    }));
    reg.register_launch(LaunchFactory::of::<Dodge>("dodge", || {
        Box::new(Dodge::new())
    }));
    reg.register_launch(LaunchFactory::of::<Burn>("burn", || Box::new(Burn::new())));
    reg.register_launch(LaunchFactory::of::<Dilate>("dilate", || {
        Box::new(Dilate::new())
    }));
    reg.register_launch(LaunchFactory::of::<HsvFilter>("hsvfilter", || {
        Box::new(HsvFilter::new())
    }));
    reg.register_launch(LaunchFactory::of::<HsvDetector>("hsvdetector", || {
        Box::new(HsvDetector::new())
    }));
    reg.register_launch(LaunchFactory::of::<RoundedCorners>(
        "roundedcorners",
        || Box::new(RoundedCorners::new()),
    ));
    reg.register_launch(LaunchFactory::of::<Alpha>("alpha", || {
        Box::new(Alpha::new())
    }));
    reg.register_launch(LaunchFactory::of::<VideoBox>("videobox", || {
        Box::new(VideoBox::new())
    }));
    reg.register_launch(LaunchFactory::of::<crate::gamma::Gamma>("gamma", || {
        Box::new(crate::gamma::Gamma::new())
    }));
    reg.register_launch(LaunchFactory::of::<crate::deinterlace::Deinterlace>(
        "deinterlace",
        || Box::new(crate::deinterlace::Deinterlace::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::timeoverlay::TimeOverlay>(
        "timeoverlay",
        || Box::new(crate::timeoverlay::TimeOverlay::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::clockoverlay::ClockOverlay>(
        "clockoverlay",
        || Box::new(crate::clockoverlay::ClockOverlay::new()),
    ));
    // Subtitle overlay (M171): the `location=` property loads an SRT / WebVTT
    // file (std), so cues render by PTS without hand-built Rust.
    reg.register_launch(LaunchFactory::of::<TextOverlay>("textoverlay", || {
        Box::new(TextOverlay::new())
    }));
    // Subtitle parser (M477): a structured subtitle document (`Text{Srt/WebVtt/
    // Ssa/Ttml}`) in, timed plain `Text{Utf8}` cues out, so a launch line can turn
    // a `subtitlesrc` file (or a demuxed `stpp`/TTML text pad) into overlayable
    // cues: `subtitlesrc location=x.srt ! subparse ! textoverlay name=o`.
    reg.register_launch(LaunchFactory::of::<crate::subparse::SubParse>(
        "subparse",
        || Box::new(crate::subparse::SubParse::new()),
    ));
    // Subtitle writers (M1096), the inverse of `subparse`: timed `Text{Utf8}` cues
    // in, a SubRip / WebVTT document out, e.g.
    // `subtitlesrc location=x.vtt ! subparse ! srtenc ! filesink location=x.srt`.
    reg.register_launch(LaunchFactory::of::<crate::srtenc::SrtEnc>("srtenc", || {
        Box::new(crate::srtenc::SrtEnc::new())
    }));
    reg.register_launch(LaunchFactory::of::<crate::webvttenc::WebVttEnc>(
        "webvttenc",
        || Box::new(crate::webvttenc::WebVttEnc::new()),
    ));
    // Closed-caption transport converter (M1096): the same triples re-laid from
    // one byte layout into another, e.g. an MP4 caption track's packed `cc_data`
    // out as the CDPs an ancillary-data packetizer sends:
    // `... ! ccconverter in-format=cc_data out-format=cdp ! ...`.
    reg.register_launch(LaunchFactory::of::<crate::ccconverter::CcConverter>(
        "ccconverter",
        || Box::new(crate::ccconverter::CcConverter::new()),
    ));
    // Closed-caption extractor (M429): mines CEA-608 / CEA-708 captions from a
    // compressed H.264 / H.265 stream's SEI into timed text cues (default CC1),
    // e.g. `... ! h264parse ! ccextract ! textoverlay ...` on a teed branch.
    reg.register_launch(LaunchFactory::of::<crate::ccextract::CcExtract>(
        "ccextract",
        || Box::new(crate::ccextract::CcExtract::new()),
    ));
    // MISP time stamp elements (M809): the video-side half of STANAG 4609 time
    // correlation. `misptimeinsert` writes an ST 0604 microsecond time SEI into
    // each access unit; `misptimeextract` mines it back out as `ts=` text on a
    // teed branch, e.g. `... ! h264parse ! misptimeextract ! textoverlay ...`.
    reg.register_launch(LaunchFactory::of::<crate::misptime::MispTimeInsert>(
        "misptimeinsert",
        || Box::new(crate::misptime::MispTimeInsert::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::misptime::MispTimeExtract>(
        "misptimeextract",
        || Box::new(crate::misptime::MispTimeExtract::new()),
    ));
    // KLV telemetry decoder (M800): a demuxed STANAG 4609 metadata stream's ST
    // 0601 local sets become timed text lines,
    // e.g. `tsdemux stream=klv ! klvdecode ! textoverlay name=o`.
    reg.register_launch(LaunchFactory::of::<crate::klv::KlvDecode>(
        "klvdecode",
        || Box::new(crate::klv::KlvDecode::new()),
    ));
    // VobSub (DVD subpicture) decoder (M899): bitmap cues become full-frame
    // transparent RGBA canvases a compositor paints over the video,
    // e.g. `mkvdemux stream=vobsub ! vobsubdec ! c.` .
    reg.register_launch(LaunchFactory::of::<crate::vobsubdec::VobSubDec>(
        "vobsubdec",
        || Box::new(crate::vobsubdec::VobSubDec::new()),
    ));
    // DVB subtitle decoder (M900): the broadcast bitmap-subtitle sibling of
    // `vobsubdec`, e.g. `tsdemux stream=dvbsub ! dvbsubdec ! c.` . No gst alias:
    // gst's `dvbsuboverlay` is a video-overlay element, not a bare decoder.
    reg.register_launch(LaunchFactory::of::<crate::dvbsubdec::DvbSubDec>(
        "dvbsubdec",
        || Box::new(crate::dvbsubdec::DvbSubDec::new()),
    ));
    // Blu-ray PGS subtitle decoder (M925): the HDMV bitmap-subtitle sibling of
    // `dvbsubdec`, e.g. `mkvdemux stream=pgs ! pgsdec ! c.` . No gst alias: gst
    // has no PGS decoder element.
    reg.register_launch(LaunchFactory::of::<crate::pgsdec::PgsDec>("pgsdec", || {
        Box::new(crate::pgsdec::PgsDec::new())
    }));
    // EBU teletext subtitle decoder (M924): a demuxed teletext stream's subtitle
    // page becomes plain-text cues,
    // e.g. `tsdemux stream=teletext ! teletextdec page=888 ! textoverlay name=o`.
    reg.register_launch(LaunchFactory::of::<crate::teletextdec::TeletextDec>(
        "teletextdec",
        || Box::new(crate::teletextdec::TeletextDec::new()),
    ));
    // Detection-box overlay (M102): draws the frame's `AnalyticsMeta` bounding
    // boxes onto the RGBA frame, so a detector's output is visible downstream
    // (e.g. `... ! analyticsoverlay ! videoconvert ! autovideosink`). No pad
    // templates declared (caps-driven via intercept_caps). Gated on `analytics`,
    // the metadata graph it reads.
    #[cfg(feature = "analytics")]
    reg.register_launch(LaunchFactory::new("analyticsoverlay", Vec::new(), || {
        Box::new(crate::analyticsoverlay::AnalyticsOverlay::new())
    }));
    // Still-frame stream generator (M1067): `imagefreeze num-buffers=N` bounds
    // the run, a bare `imagefreeze` repeats the first frame indefinitely. No pad
    // templates declared (caps-driven via intercept_caps).
    reg.register_launch(LaunchFactory::new("imagefreeze", Vec::new(), || {
        Box::new(crate::imagefreeze::ImageFreeze::new())
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
        // Caps-driven by default: a bare `audioconvert` takes its output format /
        // channels from a downstream capsfilter, or passes the input through.
        Box::new(AudioConvert::auto())
    }));
    reg.register_launch(LaunchFactory::of::<AudioResample>("audioresample", || {
        // Caps-driven by default (M187): a bare `audioresample` takes its output
        // rate from a downstream capsfilter, or passes through.
        Box::new(AudioResample::auto())
    }));
    // Timestamp corrector (M1066): silence fills a gap, overlapping samples are
    // dropped, so the PCM stream downstream is contiguous.
    reg.register_launch(LaunchFactory::of::<AudioRate>("audiorate", || {
        Box::new(AudioRate::new())
    }));
    // Pitch-preserving time stretcher (M1075): a segment rate other than 1 is
    // played at the original pitch.
    reg.register_launch(LaunchFactory::of::<ScaleTempo>("scaletempo", || {
        Box::new(ScaleTempo::new())
    }));
    // Channel picker (M1072): the single-output form of `deinterleave`, so
    // `... ! deinterleave channel=1 ! ...` takes one channel without pad syntax.
    // The fan-out form of the same name is registered below; the parser picks by
    // link degree.
    reg.register_launch(LaunchFactory::of::<crate::deinterleave::Deinterleave>(
        "deinterleave",
        || Box::new(crate::deinterleave::Deinterleave::new()),
    ));
    reg.register_launch(LaunchFactory::of::<Volume>("volume", || {
        Box::new(Volume::new())
    }));
    reg.register_launch(LaunchFactory::of::<AudioPanorama>("audiopanorama", || {
        Box::new(AudioPanorama::new())
    }));
    reg.register_launch(LaunchFactory::of::<crate::audioamplify::AudioAmplify>(
        "audioamplify",
        || Box::new(crate::audioamplify::AudioAmplify::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::audioecho::AudioEcho>(
        "audioecho",
        || Box::new(crate::audioecho::AudioEcho::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::audiodynamic::AudioDynamic>(
        "audiodynamic",
        || Box::new(crate::audiodynamic::AudioDynamic::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::audioinvert::AudioInvert>(
        "audioinvert",
        || Box::new(crate::audioinvert::AudioInvert::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::audiokaraoke::AudioKaraoke>(
        "audiokaraoke",
        || Box::new(crate::audiokaraoke::AudioKaraoke::new()),
    ));
    reg.register_launch(
        LaunchFactory::of::<crate::audiowsinclimit::AudioWsincLimit>("audiowsinclimit", || {
            Box::new(crate::audiowsinclimit::AudioWsincLimit::new())
        }),
    );
    reg.register_launch(LaunchFactory::of::<crate::audiowsincband::AudioWsincBand>(
        "audiowsincband",
        || Box::new(crate::audiowsincband::AudioWsincBand::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::audiocheblimit::AudioChebLimit>(
        "audiocheblimit",
        || Box::new(crate::audiocheblimit::AudioChebLimit::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::audiochebband::AudioChebBand>(
        "audiochebband",
        || Box::new(crate::audiochebband::AudioChebBand::new()),
    ));
    // Channel mixers, generic-coefficient filters, re-framer and rate changer
    // (M1085). `audiofirfilter` / `audioiirfilter` take their coefficients as
    // comma-separated lists and `audiomixmatrix` its rows separated by `;`,
    // since `PropKind` has no array kind.
    reg.register_launch(
        LaunchFactory::of::<crate::audiochannelmix::AudioChannelMix>("audiochannelmix", || {
            Box::new(crate::audiochannelmix::AudioChannelMix::new())
        }),
    );
    reg.register_launch(LaunchFactory::of::<crate::audiomixmatrix::AudioMixMatrix>(
        "audiomixmatrix",
        || Box::new(crate::audiomixmatrix::AudioMixMatrix::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::stereo::Stereo>("stereo", || {
        Box::new(crate::stereo::Stereo::new())
    }));
    reg.register_launch(LaunchFactory::of::<crate::audiofirfilter::AudioFirFilter>(
        "audiofirfilter",
        || Box::new(crate::audiofirfilter::AudioFirFilter::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::audioiirfilter::AudioIirFilter>(
        "audioiirfilter",
        || Box::new(crate::audioiirfilter::AudioIirFilter::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::removesilence::RemoveSilence>(
        "removesilence",
        || Box::new(crate::removesilence::RemoveSilence::new()),
    ));
    reg.register_launch(
        LaunchFactory::of::<crate::audiobuffersplit::AudioBufferSplit>("audiobuffersplit", || {
            Box::new(crate::audiobuffersplit::AudioBufferSplit::new())
        }),
    );
    reg.register_launch(LaunchFactory::of::<crate::speed::Speed>("speed", || {
        Box::new(crate::speed::Speed::new())
    }));
    // Audio half of reverse playback (M1130): reverses the samples inside each
    // `chunk-duration` batch, the way `gopreverse` reverses a decoded GOP.
    reg.register_launch(LaunchFactory::of::<crate::audioreverse::AudioReverse>(
        "audioreverse",
        || Box::new(crate::audioreverse::AudioReverse::new()),
    ));
    // Level meter + silence detector (passthrough analyzers): measurements are
    // read via getters, the g2g analog of gst posting them on the bus.
    reg.register_launch(LaunchFactory::of::<crate::level::Level>("level", || {
        Box::new(crate::level::Level::new())
    }));
    reg.register_launch(LaunchFactory::of::<crate::cutter::Cutter>("cutter", || {
        Box::new(crate::cutter::Cutter::new())
    }));
    reg.register_launch(LaunchFactory::of::<crate::equalizer::Equalizer3Bands>(
        "equalizer-3bands",
        || Box::new(crate::equalizer::Equalizer3Bands::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::spectrum::Spectrum>(
        "spectrum",
        || Box::new(crate::spectrum::Spectrum::new()),
    ));
    // EBU R128 loudness (M1131): momentary / short-term / gated integrated LUFS.
    reg.register_launch(LaunchFactory::of::<crate::ebur128::Ebur128>(
        "ebur128",
        || Box::new(crate::ebur128::Ebur128::new()),
    ));

    // Demuxers + parsers + passthrough.
    reg.register_launch(LaunchFactory::of::<TsDemux>("tsdemux", || {
        Box::new(TsDemux::new())
    }));
    reg.register_launch(LaunchFactory::of::<crate::psdemux::PsDemux>(
        "mpegpsdemux",
        || Box::new(crate::psdemux::PsDemux::new()),
    ));
    reg.register_launch(LaunchFactory::of::<MkvDemux>("matroskademux", || {
        Box::new(MkvDemux::new())
    }));
    reg.register_launch(LaunchFactory::of::<IvfDemux>("ivfdemux", || {
        Box::new(IvfDemux::new())
    }));
    // AVI (M1071): `filesrc location=X.avi ! avidemux ! ...` (the video stream,
    // or the `stream=` selection). A multi-branch `avidemux name=d  d.video_0 !
    // ...  d.audio_0 ! ...` fans out via the demux-select hook (AviDemuxN).
    reg.register_launch(LaunchFactory::of::<crate::avidemux::AviDemux>(
        "avidemux",
        || Box::new(crate::avidemux::AviDemux::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::avimux::AviMux>("avimux", || {
        Box::new(crate::avimux::AviMux::new())
    }));
    reg.register_launch(LaunchFactory::of::<TsMux>("mpegtsmux", || {
        Box::new(TsMux::new())
    }));
    reg.register_launch(LaunchFactory::of::<MkvMux>("matroskamux", || {
        Box::new(MkvMux::new())
    }));
    // Fragmented-MP4 / ISO-BMFF muxer (M291), the gst `mp4mux`/`qtmux` analog:
    // `... ! x264enc ! mp4mux ! filesink location=out.mp4`. std-gated like its
    // module (it shares the `fmp4mux` box writer).
    #[cfg(feature = "std")]
    reg.register_launch(LaunchFactory::of::<Mp4Mux>("mp4mux", || {
        Box::new(Mp4Mux::new())
    }));
    reg.register_launch(LaunchFactory::of::<OggDemux>("oggdemux", || {
        Box::new(OggDemux::new())
    }));
    reg.register_launch(LaunchFactory::of::<OggMux>("oggmux", || {
        Box::new(OggMux::new())
    }));
    reg.register_launch(LaunchFactory::of::<Fmp4Demux>("fmp4demux", || {
        Box::new(Fmp4Demux::new())
    }));
    // Progressive MP4 single-output demux (M479): `filesrc location=X.mp4 ! qtdemux
    // ! h264parse ! ...` (the video track). A multi-branch `qtdemux name=d
    // d.video_0 ! ... d.audio_0 ! ...` still fans out via the demux-select hook
    // (Mp4DemuxN); this covers the single-stream / video-only file.
    #[cfg(feature = "std")]
    reg.register_launch(LaunchFactory::of::<Mp4Demux>("qtdemux", || {
        Box::new(Mp4Demux::new())
    }));
    reg.register_launch(LaunchFactory::of::<FlvDemux>("flvdemux", || {
        Box::new(FlvDemux::new())
    }));
    reg.register_launch(LaunchFactory::of::<FlvMux>("flvmux", || {
        Box::new(FlvMux::new())
    }));
    // Re-framing mode (M421): a `gst-launch` `h264parse` access-unit-aligns its
    // output (one coded picture per buffer), matching GStreamer's `h264parse`, so
    // `... ! tsdemux ! h264parse ! <decoder> ! ...` feeds the decoder correctly.
    reg.register_launch(LaunchFactory::of::<H264Parse>("h264parse", || {
        Box::new(H264Parse::reframing())
    }));
    reg.register_launch(LaunchFactory::of::<H265Parse>("h265parse", || {
        Box::new(H265Parse::reframing())
    }));
    reg.register_launch(LaunchFactory::of::<AacParse>("aacparse", || {
        Box::new(AacParse::new())
    }));
    reg.register_launch(LaunchFactory::of::<FlacParse>("flacparse", || {
        Box::new(FlacParse::new())
    }));
    reg.register_launch(LaunchFactory::of::<MpegAudioParse>(
        "mpegaudioparse",
        || Box::new(MpegAudioParse::new()),
    ));
    reg.register_launch(LaunchFactory::of::<Ac3Parse>("ac3parse", || {
        Box::new(Ac3Parse::new())
    }));
    // Still-image framers (M1087): a JPEG / PNG byte stream into whole images,
    // which is what `mjpegdec` / `pngdec` need (they take one image per buffer).
    reg.register_launch(LaunchFactory::of::<JpegParse>("jpegparse", || {
        Box::new(JpegParse::new())
    }));
    reg.register_launch(LaunchFactory::of::<PngParse>("pngparse", || {
        Box::new(PngParse::new())
    }));
    // Headerless raw framers (M1086): the file declares nothing, so the shape
    // comes from the properties. Named explicitly, never auto-plugged: a
    // `ByteStream{Raw}` link carries no geometry to pick them by.
    reg.register_launch(LaunchFactory::of::<RawVideoParse>("rawvideoparse", || {
        Box::new(RawVideoParse::new())
    }));
    reg.register_launch(LaunchFactory::of::<RawAudioParse>("rawaudioparse", || {
        Box::new(RawAudioParse::new())
    }));
    // Takes any byte stream (the tags are not part of the media type), so it
    // declares no pad templates, the way `typefind` does.
    reg.register_launch(LaunchFactory::new("id3demux", Vec::new(), || {
        Box::new(Id3Demux::new())
    }));
    // Tag writers (M1094). The three that take any byte stream declare no pad
    // templates, like `id3demux`; `xingmux` / `vorbistag` / `flactag` are pinned
    // to the format whose header they rewrite.
    reg.register_launch(LaunchFactory::new("id3v2mux", Vec::new(), || {
        Box::new(crate::id3v2mux::Id3V2Mux::new())
    }));
    reg.register_launch(LaunchFactory::new("apev2mux", Vec::new(), || {
        Box::new(crate::apev2mux::ApeV2Mux::new())
    }));
    reg.register_launch(LaunchFactory::new("apedemux", Vec::new(), || {
        Box::new(ApeDemux::new())
    }));
    reg.register_launch(LaunchFactory::of::<crate::xingmux::XingMux>(
        "xingmux",
        || Box::new(crate::xingmux::XingMux::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::vorbistag::VorbisTag>(
        "vorbistag",
        || Box::new(crate::vorbistag::VorbisTag::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::flactag::FlacTag>(
        "flactag",
        || Box::new(crate::flactag::FlacTag::new()),
    ));
    reg.register_launch(LaunchFactory::of::<OpusParse>("opusparse", || {
        Box::new(OpusParse::new())
    }));
    reg.register_launch(LaunchFactory::of::<Vp8Parse>("vp8parse", || {
        Box::new(Vp8Parse::new())
    }));
    reg.register_launch(LaunchFactory::of::<Vp9Parse>("vp9parse", || {
        Box::new(Vp9Parse::new())
    }));
    reg.register_launch(LaunchFactory::of::<Av1Parse>("av1parse", || {
        Box::new(Av1Parse::new())
    }));
    // Legacy video parsers (M1095): the start-code elementary streams that are
    // not NAL streams. `vc1parse` frames advanced profile only; simple / main
    // carry no start codes.
    reg.register_launch(LaunchFactory::of::<MpegVideoParse>(
        "mpegvideoparse",
        || Box::new(MpegVideoParse::new()),
    ));
    reg.register_launch(LaunchFactory::of::<Mpeg4VideoParse>(
        "mpeg4videoparse",
        || Box::new(Mpeg4VideoParse::new()),
    ));
    reg.register_launch(LaunchFactory::of::<Vc1Parse>("vc1parse", || {
        Box::new(Vc1Parse::new())
    }));
    reg.register_launch(LaunchFactory::new("identity", Vec::new(), || {
        Box::new(IdentityTransform::new())
    }));
    // `debugspy`: passthrough hasher, any caps.
    reg.register_launch(LaunchFactory::new("debugspy", Vec::new(), || {
        Box::new(DebugSpy::new())
    }));
    // Closable pass-through (M1070): `valve drop=true` mutes one branch of a tee
    // without rebuilding the graph. No pad templates, like `identity`.
    reg.register_launch(LaunchFactory::new("valve", Vec::new(), || {
        Box::new(Valve::new())
    }));
    // Mid-graph content sniffing: re-declares the caps of a byte stream from its
    // own leading bytes, for a source that could only guess (`srtsrc ! typefind !
    // decodebin`). No pad templates, like `identity`: it passes data through and
    // only the sniff decides the type.
    reg.register_launch(LaunchFactory::new("typefind", Vec::new(), || {
        Box::new(crate::typefind::TypeFind::new())
    }));
    // Wall-clock pacing (M945): `sync=false` makes it an identity again.
    reg.register_launch(LaunchFactory::new("clocksync", Vec::new(), || {
        Box::new(crate::clocksync::ClockSyncTransform::new())
    }));
    // Reverse playback (M897): re-emits each decoded GOP in descending PTS, so a
    // `rate < 0` seek plays backwards through a forward-only decoder.
    reg.register_launch(LaunchFactory::new("gopreverse", Vec::new(), || {
        Box::new(crate::gopreverse::GopReverse::new())
    }));
    // Spill-to-storage buffer (M861): absorbs a pushed byte stream into a temp
    // file and serves it seekably, so `httpsrc bytestream-format=mp4 !
    // downloadbuffer ! qtdemux` plays a moov-at-end MP4 that the pushed stream
    // alone cannot (its output is the whole-file `ByteStream{Mp4}`).
    reg.register_launch(LaunchFactory::of::<DownloadBuffer>(
        "downloadbuffer",
        || Box::new(DownloadBuffer::new()),
    ));
    // Progress report passthrough: counts frames / bytes, logs periodically.
    reg.register_launch(LaunchFactory::new("progressreport", Vec::new(), || {
        Box::new(crate::progressreport::ProgressReport::new())
    }));
    // Stall detector (M1077): `watchdog timeout=2000` fails a run whose source
    // went silent. No pad templates, like `identity`.
    reg.register_launch(LaunchFactory::new("watchdog", Vec::new(), || {
        Box::new(crate::watchdog::Watchdog::new())
    }));
    // Caps rewriter (M1077): `capssetter caps="video/x-raw,framerate=60/1"`
    // corrects what a source declared, without touching the data.
    reg.register_launch(LaunchFactory::new("capssetter", Vec::new(), || {
        Box::new(crate::capssetter::CapsSetter::new())
    }));
    // Tag injector (M1077): posts a hand-written tag list on the bus.
    reg.register_launch(LaunchFactory::new("taginject", Vec::new(), || {
        Box::new(crate::taginject::TagInject::new())
    }));
    // Byte-stream re-chunker (M1077): randomly sized buffers, for shaking out a
    // parser that depends on where its input is cut.
    reg.register_launch(LaunchFactory::new("rndbuffersize", Vec::new(), || {
        Box::new(crate::rndbuffersize::RndBufferSize::new())
    }));
    // Byte-stream re-chunker (M1083): the step-aligned sibling of
    // `rndbuffersize`, so every cut lands on `step-size`.
    reg.register_launch(LaunchFactory::new("chopmydata", Vec::new(), || {
        Box::new(crate::chopmydata::ChopMyData::new())
    }));
    // Byte corrupter (M1083): `breakmydata probability=0.01 seed=7` proves a
    // parser downstream fails the parse instead of panicking.
    reg.register_launch(LaunchFactory::new("breakmydata", Vec::new(), || {
        Box::new(crate::breakmydata::BreakMyData::new())
    }));
    // Failure absorber (M1083): keeps the run going when the branch below it
    // dies. No pad templates, like `identity`.
    reg.register_launch(LaunchFactory::new("errorignore", Vec::new(), || {
        Box::new(crate::errorignore::ErrorIgnore::new())
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
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
        ))
    }));
    // Sequential concatenation and live input switch: both N-in / 1-out, so the
    // parser builds them by link degree (input count from the branches linked in).
    reg.register_muxer(MuxerFactory::new("concat", |inputs| {
        Box::new(crate::concat::Concat::new(inputs))
    }));
    reg.register_muxer(MuxerFactory::new("input-selector", |inputs| {
        Box::new(crate::inputselector::InputSelector::new(inputs))
    }));
    // Live output switch: 1-in / N-out, built by the demux link degree.
    reg.register_demux(g2g_core::runtime::DemuxFactory::new(
        "output-selector",
        |outputs| Box::new(crate::outputselector::OutputSelector::new(outputs)),
    ));
    // Channel splitter fan-out (M1072): one N-channel PCM stream in, one mono
    // stream per port out, port count from the `d.` link degree:
    // `audiotestsrc channels=2 ! deinterleave name=d  d.src_0 ! fakesink
    // d.src_1 ! fakesink`. Each port announces its mono caps at run time, so the
    // input's channel count has to equal the port count.
    reg.register_demux(g2g_core::runtime::DemuxFactory::new(
        "deinterleave",
        |outputs| Box::new(crate::deinterleave::DeinterleaveN::new(outputs)),
    ));
    // Subtitle-overlay fan-in (M477): the launch-line sibling of the single-input
    // `textoverlay` above, the analog of GStreamer's `textoverlay` text_sink
    // request pad. A `TextOverlayN` merges an RGBA8 video pad (input 0) and a timed
    // `Text{Utf8}` pad (input 1) by PTS, painting cues onto the video:
    // `d.video_0 ! ffmpegdec ! videoconvert ! o.   d.text_0 ! o.   textoverlay
    // name=o ! videoconvert ! autovideosink`. Registered both as a single-input
    // launch element and here as a fan-in muxer; the parser picks by link degree
    // (M122), the same one-name-two-roles model as `mpegtsmux`. Always 2-input, so
    // link exactly the video and text branches.
    reg.register_muxer(MuxerFactory::new("textoverlay", |_inputs| {
        Box::new(crate::textoverlay::TextOverlayN::new())
    }));
    // Bitmap-subtitle overlay fan-in (M1005): the same shape for cues that are
    // pixels rather than text. An RGBA8 video pad (input 0) and the RGBA8
    // canvases a subpicture decoder paints (input 1), merged by PTS:
    // `d.video_0 ! avdec_h264 ! videoconvert ! o.video   d.text_0 ! vobsubdec !
    // o.text   subpictureoverlay name=o ! videoconvert ! autovideosink`. Always
    // 2-input, so link exactly the video and subpicture branches.
    reg.register_muxer(MuxerFactory::new("subpictureoverlay", |_inputs| {
        Box::new(crate::subpictureoverlay::SubPictureOverlay::new())
    }));
    // Closed-caption fan-in (M1096): a video branch on `c.video` (input 0, whose
    // caps the output follows) and a closed-caption branch on `c.caption` (input
    // 1), merged by PTS so each video frame leaves carrying the caption triples
    // that belong with it, ready for a `ccinsert` in meta-sourced mode. Always
    // 2-input, so link exactly the video and caption branches.
    #[cfg(feature = "metadata")]
    reg.register_muxer(MuxerFactory::new("cccombiner", |_inputs| {
        Box::new(crate::cccombiner::CcCombiner::new())
    }));
    // Picture-in-picture / grid video fan-in (M876): the gst `compositor` analog,
    // built by link degree like the muxers above (one pad per branch linked in,
    // input 0 the timing driver and backmost layer). The canvas is a nominal
    // default matching `videotestsrc`; `width` / `height` / `framerate` /
    // `background-color` / `format` retarget it, and gst's request-pad placement
    // (`sink_1::xpos`) is flattened to `sinkN-xpos` and friends:
    // `videotestsrc ! c.  videotestsrc ! c.  compositor name=c width=640
    // height=480 sink1-xpos=320 sink1-zorder=1 ! videoconvert ! autovideosink`.
    reg.register_muxer(MuxerFactory::new("compositor", |inputs| {
        Box::new(Compositor::new(
            320,
            240,
            (0..inputs).map(|_| CompositorPad::at(0, 0)).collect(),
        ))
    }));
    // The GPU sibling, same property surface (RGBA8 only, no `format`).
    #[cfg(feature = "wgpu-sink")]
    reg.register_muxer(MuxerFactory::new("wgpucompositor", |inputs| {
        Box::new(crate::wgpucompositor::WgpuCompositor::new(
            320,
            240,
            (0..inputs).map(|_| CompositorPad::at(0, 0)).collect(),
        ))
    }));
    // Summing audio fan-in (M130): sums N S16LE inputs on a PTS-aligned timeline
    // (M664). The output caps are a nominal default matching `audiotestsrc`; its
    // channel count / rate drive the PTS-to-sample-frame mapping.
    reg.register_muxer(MuxerFactory::new("audiomixer", |inputs| {
        Box::new(AudioMixer::new(
            inputs,
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 48_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            },
        ))
    }));
    // Channel interleaver fan-in (M1072): N mono pads in, one N-channel stream
    // out, pad count from the link degree. The output shape is declared before
    // the pads negotiate, so a non-default one is set on the element:
    // `interleave name=i format=F32LE rate=44100`.
    reg.register_muxer(MuxerFactory::new("interleave", |inputs| {
        Box::new(crate::interleave::Interleave::new(inputs))
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
    // Multi-stream Ogg fan-in (M790): the grouped-bitstream case. Like
    // `mpegtsmux`, `oggmux` is both a single-input launch element
    // (`oggmux::OggMux` above) and this fan-in muxer (`oggmuxn::OggMuxN`); the
    // parser picks by link degree, so one name covers `! oggmux !` and
    // `a.! m.  b.! m.  oggmux name=m`. Each pad becomes its own logical
    // bitstream; packets interleave by PTS (M204).
    // AVI fan-in (M1071): one video stream plus an optional audio one. Like
    // `mpegtsmux`, `avimux` is both a single-input launch element
    // (`avimux::AviMux` above) and this fan-in muxer (`avimux::AviMuxN`); the
    // parser picks by link degree, so one name covers `! avimux !` and
    // `v.! m.video_0  a.! m.audio_0  avimux name=m`.
    reg.register_muxer(MuxerFactory::new("avimux", |inputs| {
        Box::new(crate::avimux::AviMuxN::new(inputs))
    }));
    reg.register_muxer(MuxerFactory::new("oggmux", |inputs| {
        Box::new(crate::oggmuxn::OggMuxN::new(inputs))
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
    reg.register_launch(LaunchFactory::of::<FakeSink>("fakesink", || {
        Box::new(FakeSink::new())
    }));
    // The media-typed fake sinks (M1083): `fakesink` behind a pad that takes
    // only decoded video / only PCM, so a branch whose decode is missing fails
    // to negotiate instead of being silently swallowed.
    reg.register_launch(LaunchFactory::of::<crate::fakemediasink::FakeVideoSink>(
        "fakevideosink",
        || Box::new(crate::fakemediasink::FakeVideoSink::new()),
    ));
    reg.register_launch(LaunchFactory::of::<crate::fakemediasink::FakeAudioSink>(
        "fakeaudiosink",
        || Box::new(crate::fakemediasink::FakeAudioSink::new()),
    ));
    // Digest sink (M1083): one `<pts> <digest>` line per buffer, for checking a
    // codec change is bit-exact.
    reg.register_launch(LaunchFactory::of::<crate::checksumsink::ChecksumSink>(
        "checksumsink",
        || Box::new(crate::checksumsink::ChecksumSink::new()),
    ));
    // Frame-rate reporter (M1083): wraps the display sink `video-sink` names and
    // reports the rate it achieves.
    reg.register_launch(LaunchFactory::new("fpsdisplaysink", Vec::new(), || {
        Box::new(crate::fpsdisplaysink::FpsDisplaySink::new())
    }));
    // Application pull/callback sink (M233): hands buffers to a callback set via
    // `appsink::set_appsink_callback`.
    reg.register_launch(LaunchFactory::of::<crate::appsink::AppSink>(
        "appsink",
        || Box::new(crate::appsink::AppSink::new()),
    ));
    reg.register_launch(LaunchFactory::of::<FileSink>("filesink", || {
        Box::new(FileSink::new(""))
    }));
    // File-descriptor sink (M1070): writes to an already-open descriptor, the
    // other half of a shell pipe. Unix only, like the module.
    #[cfg(unix)]
    reg.register_launch(LaunchFactory::of::<crate::fd::FdSink>("fdsink", || {
        Box::new(crate::fd::FdSink::default())
    }));
    // Raw-PCM WAV file sink: `... ! audioconvert ! wavsink location=out.wav`.
    reg.register_launch(LaunchFactory::of::<crate::wavsink::WavSink>(
        "wavsink",
        || Box::new(crate::wavsink::WavSink::new("")),
    ));
    // Record / replay pair: record the packet stream to a file, play it back as a source.
    reg.register_launch(LaunchFactory::of::<RecordSink>("recordsink", || {
        Box::new(RecordSink::new(""))
    }));
    reg.register_source(SourceFactory::new(
        "replaysrc",
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
        || Box::new(ReplaySrc::new("")),
    ));
    reg.register_launch(LaunchFactory::of::<crate::multifilesink::MultiFileSink>(
        "multifilesink",
        || Box::new(crate::multifilesink::MultiFileSink::new("")),
    ));
    reg.register_launch(LaunchFactory::of::<crate::splitmuxsink::SplitMuxSink>(
        "splitmuxsink",
        || Box::new(crate::splitmuxsink::SplitMuxSink::new("")),
    ));
    // HLS packager, fed by a muxer: `... ! tsmux ! hlssink location=seg%05d.ts`.
    reg.register_launch(LaunchFactory::of::<crate::hlssink::HlsSink>(
        "hlssink",
        || Box::new(crate::hlssink::HlsSink::default()),
    ));
    #[cfg(feature = "rtmp")]
    reg.register_launch(LaunchFactory::of::<RtmpSink>("rtmpsink", || {
        Box::new(RtmpSink::new(""))
    }));

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
    reg.register_playbin(crate::uridecodebin::ps_playbin);
    // Lone-audio-stream files the container hooks decline: Ogg (Opus / FLAC)
    // and elementary audio (`.flac`), M775.
    reg.register_playbin(crate::uridecodebin::audio_playbin);
    // Explicit-demux fan-out (M476): a named `matroskademux` / `tsdemux` / `qtdemux`
    // fed by a file source, with several output-pad refs (`d.video_0`, `d.audio_0`),
    // probes its file and builds the multi-output demuxer honoring the selection.
    reg.register_demux_select(crate::uridecodebin::mkv_demux_select);
    reg.register_demux_select(crate::uridecodebin::ts_demux_select);
    reg.register_demux_select(crate::uridecodebin::mp4_demux_select);
    reg.register_demux_select(crate::uridecodebin::ogg_demux_select);
    reg.register_demux_select(crate::uridecodebin::ps_demux_select);
    reg.register_demux_select(crate::uridecodebin::avi_demux_select);
    // `decodebin` fan-out (M482): `filesrc location=x ! decodebin name=d  d.video_0
    // ! ...  d.audio_0 ! ...` probes the file, builds the multi-output demuxer, and
    // auto-plugs a decoder onto each port (the decode-per-port sibling of the above).
    reg.register_decodebin_select(crate::uridecodebin::mkv_decodebin_select);
    reg.register_decodebin_select(crate::uridecodebin::ts_decodebin_select);
    reg.register_decodebin_select(crate::uridecodebin::mp4_decodebin_select);
    reg.register_decodebin_select(crate::uridecodebin::ps_decodebin_select);
    reg.register_decodebin_select(crate::uridecodebin::avi_decodebin_select);
    // Bare `filesrc location=X ! decodebin` primary-stream selection (M746): an
    // audio-only container's single-stream demux defaults to a video port; the hook
    // sniffs the file and selects the real (audio) stream instead.
    reg.register_primary_stream(crate::uridecodebin::ts_primary_stream);
    reg.register_primary_stream(crate::uridecodebin::mp4_primary_stream);
    reg.register_primary_stream(crate::uridecodebin::mkv_primary_stream);
    reg.register_primary_stream(crate::uridecodebin::ogg_primary_stream);
    reg.register_primary_stream(crate::uridecodebin::ps_primary_stream);
    reg.register_primary_stream(crate::uridecodebin::avi_primary_stream);
    // hls:// fan-out (M395): probe the master playlist, fan its variant's muxed TS
    // streams out; the hls_handler is the single-stream fallback it declines to.
    #[cfg(feature = "hls")]
    {
        reg.register_uri(crate::uridecodebin::hls_handler());
        reg.register_playbin(crate::uridecodebin::hls_playbin);
    }
    #[cfg(feature = "udp-ingress")]
    reg.register_uri(crate::uridecodebin::udp_handler());
    // rtsp:// fan-out (M1122): DESCRIBE the stream and, when it carries audio as
    // well as video, play both off one session; the rtsp_handler is the
    // video-only fallback it declines to.
    #[cfg(feature = "rtsp")]
    {
        reg.register_uri(crate::uridecodebin::rtsp_handler());
        reg.register_playbin(crate::uridecodebin::rtsp_playbin);
    }
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
    // The source pad template lists NV12 before I420, so a `is_raw_video` target
    // settles on NV12 (the layout KMS / waylandsink want); an I420-only sink drives
    // I420. Anything that is not raw video falls back to I420.
    ffmpegdec_pinned_output_format(out).unwrap_or(crate::ffmpegdec::OutputFormat::I420)
}

/// Autoplug output format for the **software** `ffmpegdec` (M685/M686): default
/// to `Auto` for a loose target so the decoder keeps the source chroma and a
/// downstream that pins 4:2:2 / 4:4:4 (a capsfilter) negotiates it. NV12 stays
/// explicit for overlay-plane sinks. Distinct from [`ffmpegdec_output_format`],
/// which the fixed-format hwaccel path (`ffmpegvaapidec`) keeps using.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
fn ffmpegdec_sw_output_format(out: &Caps) -> crate::ffmpegdec::OutputFormat {
    ffmpegdec_pinned_output_format(out).unwrap_or(crate::ffmpegdec::OutputFormat::Auto)
}

/// The decoder output layout a *pinned* raw-video target demands: a downstream
/// that fixes NV12, a non-4:2:0 chroma, or a 10-/12-bit format (M887) must get a
/// decoder built for exactly that, or its `CapsChanged` is rejected at startup.
/// `None` for an I420 / non-raw target, whose fallback differs per backend.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
fn ffmpegdec_pinned_output_format(out: &Caps) -> Option<crate::ffmpegdec::OutputFormat> {
    use crate::ffmpegdec::OutputFormat;
    let Caps::RawVideo { format, .. } = out else {
        return None;
    };
    Some(match format {
        RawVideoFormat::Nv12 => OutputFormat::Nv12,
        RawVideoFormat::I422 => OutputFormat::I422,
        RawVideoFormat::I444 => OutputFormat::I444,
        RawVideoFormat::I420p10 => OutputFormat::I420p10,
        RawVideoFormat::I420p12 => OutputFormat::I420p12,
        RawVideoFormat::I422p10 => OutputFormat::I422p10,
        RawVideoFormat::I422p12 => OutputFormat::I422p12,
        RawVideoFormat::I444p10 => OutputFormat::I444p10,
        RawVideoFormat::I444p12 => OutputFormat::I444p12,
        RawVideoFormat::P010 => OutputFormat::P010,
        _ => return None,
    })
}

fn register_autoplug_candidates(reg: &mut Registry) {
    // Parsers (baseline): elementary-stream framing, no external deps.
    reg.register(ElementFactory::of::<H264Parse>("h264parse", |_| {
        Box::new(H264Parse::new())
    }));
    reg.register(ElementFactory::of::<H265Parse>("h265parse", |_| {
        Box::new(H265Parse::new())
    }));
    reg.register(ElementFactory::of::<FlacParse>("flacparse", |_| {
        Box::new(FlacParse::new())
    }));
    reg.register(ElementFactory::of::<MpegAudioParse>(
        "mpegaudioparse",
        |_| Box::new(MpegAudioParse::new()),
    ));
    reg.register(ElementFactory::of::<Ac3Parse>("ac3parse", |_| {
        Box::new(Ac3Parse::new())
    }));
    reg.register(ElementFactory::of::<AacParse>("aacparse", |_| {
        Box::new(AacParse::new())
    }));
    reg.register(ElementFactory::of::<OpusParse>("opusparse", |_| {
        Box::new(OpusParse::new())
    }));
    reg.register(ElementFactory::of::<Vp8Parse>("vp8parse", |_| {
        Box::new(Vp8Parse::new())
    }));
    reg.register(ElementFactory::of::<Vp9Parse>("vp9parse", |_| {
        Box::new(Vp9Parse::new())
    }));
    reg.register(ElementFactory::of::<Av1Parse>("av1parse", |_| {
        Box::new(Av1Parse::new())
    }));

    // Demuxers (baseline, M194): a container byte stream in, one selected
    // elementary stream out. They are 1-in/1-out (an instance forwards one stream,
    // chosen by codec), so the chain search composes them like any other element:
    // ByteStream{container} -> tsdemux/... -> CompressedVideo|Audio -> decoder ->
    // raw. Built parameterless = the default (video) stream, which matches both
    // the search's first-alternative choice and the decodebin macro's by-name
    // build, so the two decode paths stay consistent.
    reg.register(ElementFactory::of::<TsDemux>("tsdemux", |_| {
        Box::new(TsDemux::new())
    }));
    reg.register(ElementFactory::of::<crate::psdemux::PsDemux>(
        "mpegpsdemux",
        |_| Box::new(crate::psdemux::PsDemux::new()),
    ));
    reg.register(ElementFactory::of::<MkvDemux>("matroskademux", |_| {
        Box::new(MkvDemux::new())
    }));
    reg.register(ElementFactory::of::<IvfDemux>("ivfdemux", |_| {
        Box::new(IvfDemux::new())
    }));
    // AVI (M1071): `ByteStream{Avi}` -> the stream it selects, so
    // `filesrc location=x.avi ! decodebin` auto-plugs this.
    reg.register(ElementFactory::of::<crate::avidemux::AviDemux>(
        "avidemux",
        |_| Box::new(crate::avidemux::AviDemux::new()),
    ));
    // RIFF/WAVE (M1030): `ByteStream{Wav}` -> the PCM it carries, so
    // `filesrc location=x.wav ! decodebin` auto-plugs this.
    reg.register(ElementFactory::of::<WavParse>("wavparse", |_| {
        Box::new(WavParse::new())
    }));
    // AIFF / AU (M1102): the same auto-plug for `.aiff` / `.au`.
    reg.register(ElementFactory::of::<AiffParse>("aiffparse", |_| {
        Box::new(AiffParse::new())
    }));
    reg.register(ElementFactory::of::<AuParse>("auparse", |_| {
        Box::new(AuParse::new())
    }));
    // YUV4MPEG2 (M1076): `ByteStream{Y4m}` -> the raw frames it carries, so
    // `filesrc location=x.y4m ! decodebin` auto-plugs this.
    reg.register(ElementFactory::of::<Y4mDec>("y4mdec", |_| {
        Box::new(Y4mDec::new())
    }));
    // MIME multipart (M1080): `ByteStream{Multipart}` -> the JPEG parts it
    // carries, so `httpsrc bytestream-format=multipart ! decodebin` auto-plugs
    // this ahead of the JPEG decoder.
    reg.register(ElementFactory::of::<crate::multipart::MultipartDemux>(
        "multipartdemux",
        |_| Box::new(crate::multipart::MultipartDemux::new()),
    ));
    reg.register(ElementFactory::of::<Fmp4Demux>("fmp4demux", |_| {
        Box::new(Fmp4Demux::new())
    }));
    // Whole-file / progressive MP4 (M479): `ByteStream{Mp4}` -> the video track, so
    // `filesrc location=X.mp4 ! decodebin` auto-plugs a demuxer (the fragmented
    // `fmp4demux` above stays on the streaming `IsoBmff` that HLS / DASH produce).
    reg.register(ElementFactory::of::<Mp4Demux>("qtdemux", |_| {
        Box::new(Mp4Demux::new())
    }));
    reg.register(ElementFactory::of::<OggDemux>("oggdemux", |_| {
        Box::new(OggDemux::new())
    }));
    reg.register(ElementFactory::of::<FlvDemux>("flvdemux", |_| {
        Box::new(FlvDemux::new())
    }));

    // Decoders (feature- + platform-gated, same gate as the launch registration).
    // Telephony codecs (M1073): baseline, so `rtspsrc ! decodebin` reaches PCM
    // on a PCMU / PCMA camera and `filesrc ! wavparse ! decodebin` on an
    // ADPCM WAV.
    reg.register(ElementFactory::of::<MulawDec>("mulawdec", |_| {
        Box::new(MulawDec::new())
    }));
    reg.register(ElementFactory::of::<AlawDec>("alawdec", |_| {
        Box::new(AlawDec::new())
    }));
    reg.register(ElementFactory::of::<AdpcmDec>("adpcmdec", |_| {
        Box::new(AdpcmDec::new())
    }));
    #[cfg(feature = "opus")]
    reg.register(ElementFactory::of::<OpusDec>("opusdec", |_| {
        Box::new(OpusDec::new())
    }));
    #[cfg(feature = "vorbis")]
    reg.register(ElementFactory::of::<VorbisDec>("vorbisdec", |_| {
        Box::new(VorbisDec::new())
    }));
    #[cfg(feature = "mjpeg")]
    reg.register(ElementFactory::of::<MjpegDec>("mjpegdec", |_| {
        Box::new(MjpegDec::new())
    }));
    // Still images (M1050): typefind types a `.png` / `.webp` file as a
    // one-frame CompressedVideo stream, and these are what `decodebin` plugs
    // behind it to reach raw RGBA.
    #[cfg(feature = "png")]
    reg.register(ElementFactory::of::<PngDec>("pngdec", |_| {
        Box::new(PngDec::new())
    }));
    reg.register(ElementFactory::of::<PnmDec>("pnmdec", |_| {
        Box::new(PnmDec::new())
    }));
    #[cfg(feature = "webp")]
    reg.register(ElementFactory::of::<WebPDec>("webpdec", |_| {
        Box::new(WebPDec::new())
    }));
    // AV1 decode via libdav1d (software, System memory): an auto-plug candidate
    // for AV1 -> I420, alongside av1parse.
    #[cfg(feature = "dav1d")]
    reg.register(ElementFactory::of::<Dav1dDec>("dav1ddec", |_| {
        Box::new(Dav1dDec::new())
    }));
    // Pure-Rust AV1 decode via re_rav1d (software, System memory): same AV1 -> I420
    // candidate. Negative rank so libdav1d (hand-written asm, faster) wins the
    // auto-plug tiebreak when both are built; rav1ddec is the portable fallback and
    // the sole AV1 decoder on pure-Rust targets.
    #[cfg(feature = "rav1d")]
    reg.register(
        ElementFactory::of::<Rav1dDec>("rav1ddec", |_| Box::new(Rav1dDec::new())).rank(-10),
    );
    // Honor the output format the auto-plug search chose for this hop
    // (`ChainLink::output`): the source pad template advertises both NV12 and
    // I420, so a strict-NV12 sink (KMS / waylandsink) makes the search settle on
    // NV12, and the decoder must be built to emit it. Ignoring `out` here built a
    // fixed-I420 decoder under a chain promised NV12, so the runner's forward-caps
    // pre-fix (sink's NV12 accept-set) hit the decoder's `format != output_format`
    // arm and failed startup negotiation. Default to I420 for a loose target.
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register(ElementFactory::of::<FfmpegH264Dec>("ffmpegdec", |out| {
        Box::new(FfmpegH264Dec::new().with_output_format(ffmpegdec_sw_output_format(out)))
    }));
    // AAC (and other libavcodec audio codecs) -> interleaved PcmS16Le (M422), the
    // audio sibling of ffmpegdec, in the auto-plug pool so a decode chain reaches
    // raw audio (e.g. an MPEG-TS / HLS AAC track).
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register(ElementFactory::of::<crate::ffmpegaudiodec::FfmpegAudioDec>(
        "ffmpegaudiodec",
        |_| Box::new(crate::ffmpegaudiodec::FfmpegAudioDec::new()),
    ));
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
        ElementFactory::of::<VaapiH264Dec>("vaapidec", |_| Box::new(VaapiH264Dec::new()))
            .hardware(),
    );
    #[cfg(all(target_os = "linux", feature = "vaapi"))]
    reg.register(
        ElementFactory::of::<VaapiH265Dec>("vaapidech265", |_| Box::new(VaapiH265Dec::new()))
            .hardware(),
    );
    // Native NVDEC (M270), registered last so a default (System-memory) auto-plug
    // still picks a CPU decoder: `NvDec` emits NV12 in CUDA device memory, which
    // caps geometry / format does not encode. M276 makes that domain a first-class
    // auto-plug feature: the factory is tagged `produces(Cuda)`, so the
    // domain-aware search prefers `NvDec` for a GPU consumer (M989 reads that
    // preference off the consumer's declared `input_domains`), while a consumer
    // that declares no domain requirement keeps the CPU decoder.
    #[cfg(all(target_os = "linux", feature = "nvdec"))]
    reg.register(
        ElementFactory::of::<NvDec>("nvdec", |_| Box::new(NvDec::new()))
            .produces(g2g_core::MemoryDomainKind::Cuda)
            .hardware(),
    );
    // Vendor-neutral Vulkan Video hardware decode (M493; H.264 / H.265 / AV1 since
    // M517), the wgpu-texture analog of NvDec: tagged `produces(WgpuTexture)`, so a
    // WgpuTexture-preferring domain-aware search picks it for a wgpu consumer (the
    // copy-free wedge into a game engine / visualization viewer) for any of the
    // three codecs its sink template advertises, while a plain (System) `decodebin`
    // is unchanged (a WgpuTexture producer is a domain mismatch there, as Cuda is).
    #[cfg(feature = "vulkan-video")]
    reg.register(
        ElementFactory::of::<crate::vulkanvideo::VulkanVideoDec>("vulkanvideodec", |_| {
            Box::new(crate::vulkanvideo::VulkanVideoDec::new())
        })
        .produces(g2g_core::MemoryDomainKind::WgpuTexture)
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
        ElementFactory::of::<MediaCodecDec>("mediacodecdech265", |_| {
            Box::new(MediaCodecDec::h265())
        })
        .hardware(),
    );
    // Android hardware video encode via the NDK MediaCodec (M306); launch-only
    // (encoders are not auto-plug candidates), one factory per codec. The gst
    // analog is `amcvidenc-<component>`.
    #[cfg(all(target_os = "android", feature = "mediacodec"))]
    reg.register_launch(
        LaunchFactory::of::<MediaCodecEnc>("mediacodecenc", || Box::new(MediaCodecEnc::h264()))
            .with_experimental(),
    );
    #[cfg(all(target_os = "android", feature = "mediacodec"))]
    reg.register_launch(
        LaunchFactory::of::<MediaCodecEnc>("mediacodecench265", || Box::new(MediaCodecEnc::h265()))
            .with_experimental(),
    );
    // macOS hardware video decode via VideoToolbox (M218/M534); one factory per
    // codec, like the MediaCodec pair. `vtdec` matches the gst applemedia name.
    // Registered twice like `ffmpegdec`: as an auto-plug candidate and as a
    // launch factory (a bare name in a text pipeline resolves via the latter).
    #[cfg(all(target_os = "macos", feature = "vtdecode"))]
    reg.register(
        ElementFactory::of::<crate::vtdecode::VtDecode>("vtdec", |_| {
            Box::new(crate::vtdecode::VtDecode::h264())
        })
        .hardware(),
    );
    #[cfg(all(target_os = "macos", feature = "vtdecode"))]
    reg.register(
        ElementFactory::of::<crate::vtdecode::VtDecode>("vtdech265", |_| {
            Box::new(crate::vtdecode::VtDecode::h265())
        })
        .hardware(),
    );
    #[cfg(all(target_os = "macos", feature = "vtdecode"))]
    reg.register_launch(
        LaunchFactory::of::<crate::vtdecode::VtDecode>("vtdec", || {
            Box::new(crate::vtdecode::VtDecode::h264())
        })
        .with_experimental(),
    );
    #[cfg(all(target_os = "macos", feature = "vtdecode"))]
    reg.register_launch(
        LaunchFactory::of::<crate::vtdecode::VtDecode>("vtdech265", || {
            Box::new(crate::vtdecode::VtDecode::h265())
        })
        .with_experimental(),
    );
    // macOS hardware video encode via VideoToolbox (M231/M534); launch-only
    // (encoders are not auto-plug candidates), under the gst applemedia names.
    #[cfg(all(target_os = "macos", feature = "vtencode"))]
    reg.register_launch(
        LaunchFactory::of::<crate::vtencode::VtEncode>("vtenc_h264", || {
            Box::new(crate::vtencode::VtEncode::h264())
        })
        .with_experimental(),
    );
    #[cfg(all(target_os = "macos", feature = "vtencode"))]
    reg.register_launch(
        LaunchFactory::of::<crate::vtencode::VtEncode>("vtenc_h265", || {
            Box::new(crate::vtencode::VtEncode::h265())
        })
        .with_experimental(),
    );
    // macOS Metal present sink (M736); the display sink on this platform, so
    // it also backs the `autovideosink` alias below.
    #[cfg(all(target_os = "macos", feature = "metal-sink"))]
    reg.register_launch(
        LaunchFactory::of::<crate::metalvideosink::MetalVideoSink>("metalvideosink", || {
            Box::new(crate::metalvideosink::MetalVideoSink::new())
        })
        .with_experimental(),
    );
    // macOS Core Audio render (M737); `osxaudiosink` is the gst analog and an
    // alias below, and `autoaudiosink` falls back to it on this platform.
    #[cfg(all(target_os = "macos", feature = "coreaudio"))]
    reg.register_launch(
        LaunchFactory::of::<crate::coreaudio::CoreAudioSink>("coreaudiosink", || {
            Box::new(crate::coreaudio::CoreAudioSink::new())
        })
        .with_experimental(),
    );
    // macOS Core Audio mic capture (M737); `osxaudiosrc` is the gst analog.
    #[cfg(all(target_os = "macos", feature = "coreaudio"))]
    reg.register_source(
        SourceFactory::new(
            "coreaudiosrc",
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 48_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            },
            || Box::new(crate::coreaudio::CoreAudioSrc::new(48_000, 2, u64::MAX)),
        )
        .with_experimental(),
    );
    // AVFoundation camera capture (M738), VGA NV12; `avfvideosrc` matches gst.
    #[cfg(all(target_os = "macos", feature = "avfoundation"))]
    reg.register_source(
        SourceFactory::new(
            "avfvideosrc",
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Fixed(30 << 16),
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            || Box::new(crate::avf::AvfVideoSrc::new(u64::MAX)),
        )
        .with_experimental(),
    );
    // AVFoundation mic capture (M738); `avfaudiosrc` matches gst's osxaudiosrc
    // sibling naming.
    #[cfg(all(target_os = "macos", feature = "avfoundation"))]
    reg.register_source(
        SourceFactory::new(
            "avfaudiosrc",
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 48_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            },
            || Box::new(crate::avf::AvfAudioSrc::new(48_000, 2, u64::MAX)),
        )
        .with_experimental(),
    );
    // ScreenCaptureKit display capture (M739). The registered caps are nominal;
    // the source reports the real display geometry at negotiation.
    #[cfg(all(target_os = "macos", feature = "screencapture"))]
    reg.register_source(
        SourceFactory::new(
            "screencapturesrc",
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(1920),
                height: Dim::Fixed(1080),
                framerate: Rate::Fixed(30 << 16),
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            || Box::new(crate::sck::ScreenCaptureSrc::new(u64::MAX)),
        )
        .with_experimental(),
    );
    // Windows capture / render (M943): registered so the device monitor's
    // `Device::create` can build what it discovered, and so a launch line can
    // name them. The registered caps are nominal; each element reports the real
    // endpoint / camera shape at negotiation.
    #[cfg(all(target_os = "windows", feature = "mf-video-src"))]
    reg.register_source(
        SourceFactory::new(
            "mfvideosrc",
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Fixed(30 << 16),
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            || Box::new(crate::mfvideosrc::MfVideoSrc::new()),
        )
        .with_experimental(),
    );
    #[cfg(all(target_os = "windows", feature = "wasapi-src"))]
    reg.register_source(
        SourceFactory::new(
            "wasapisrc",
            Caps::Audio {
                format: AudioFormat::PcmF32Le,
                channels: 2,
                sample_rate: 48_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            },
            || Box::new(crate::wasapisrc::WasapiSrc::new(u64::MAX)),
        )
        .with_experimental(),
    );
    #[cfg(all(target_os = "windows", feature = "wasapi-sink"))]
    reg.register_launch(
        LaunchFactory::of::<crate::wasapisink::WasapiSink>("wasapisink", || {
            Box::new(crate::wasapisink::WasapiSink::new())
        })
        .with_experimental(),
    );
}

/// Register gst-canonical-name aliases (M192) so pasted `gst-launch` lines using
/// GStreamer's element names resolve to the g2g equivalents. Each alias resolves
/// at construction time to the first of its targets that is actually registered,
/// so the decoder / display aliases work only when their feature is on, and the
/// `auto*sink` aliases fall back through the available display / audio sinks to
/// `fakesink` (always present), which keeps a tutorial line running headless.
fn register_aliases(reg: &mut Registry) {
    // Auto sinks: prefer a real display / audio sink, fall back to fakesink.
    // `wgpusink` leads: it is the only one that presents a GPU-resident frame
    // without a round trip through system memory.
    reg.register_alias(
        "autovideosink",
        &[
            "wgpusink",
            "waylandsink",
            "kmssink",
            "metalvideosink",
            "fakesink",
        ],
    );
    reg.register_alias(
        "autoaudiosink",
        &[
            "alsasink",
            "pulsesink",
            "coreaudiosink",
            "wasapisink",
            "fakesink",
        ],
    );
    // Auto sources: the first capture element this build actually has, in
    // platform order. No `videotestsrc` / `audiotestsrc` last resort: a capture
    // line that silently produced a test pattern would look like it worked.
    reg.register_alias(
        "autovideosrc",
        &[
            "v4l2src",
            "libcamerasrc",
            "pipewirevideosrc",
            "avfvideosrc",
            "mfvideosrc",
            "camera2src",
        ],
    );
    reg.register_alias(
        "autoaudiosrc",
        &[
            "alsasrc",
            "pulsesrc",
            "pipewiresrc",
            "coreaudiosrc",
            "wasapisrc",
            "avfaudiosrc",
            "aaudiosrc",
        ],
    );
    // gst's name for the DVD subpicture decoder.
    reg.register_alias("dvdsubdec", &["vobsubdec"]);
    // gst ships the ID3v2 writer twice under two names.
    reg.register_alias("id3mux", &["id3v2mux"]);
    // gst's older names for the raw framers, and its `unaligned*parse` bins:
    // those exist there to re-cut a stream whose buffers do not land on frame
    // boundaries, which these framers do for any input chunking.
    reg.register_alias("videoparse", &["rawvideoparse"]);
    reg.register_alias("unalignedvideoparse", &["rawvideoparse"]);
    reg.register_alias("audioparse", &["rawaudioparse"]);
    reg.register_alias("unalignedaudioparse", &["rawaudioparse"]);
    // `udpsink` takes the `clients` list gst splits into a second element.
    // `dynudpsink` is not an alias: its destination rides on per-buffer metadata,
    // which this sink has no equivalent of.
    reg.register_alias("multiudpsink", &["udpsink"]);
    // gst's macOS audio element names.
    reg.register_alias("osxaudiosink", &["coreaudiosink", "fakesink"]);
    reg.register_alias("osxaudiosrc", &["coreaudiosrc"]);
    // Common desktop video-sink names map onto whatever display sink we have.
    for name in ["xvimagesink", "ximagesink"] {
        reg.register_alias(name, &["waylandsink", "kmssink", "fakesink"]);
    }
    // `glimagesink` is the real EGL / GL ES `GlSink` when the `gl-sink` feature
    // is built; without it the name falls back like the X names above.
    #[cfg(not(all(target_os = "linux", feature = "gl-sink")))]
    reg.register_alias("glimagesink", &["waylandsink", "kmssink", "fakesink"]);
    reg.register_alias(
        "glsink",
        &["glimagesink", "waylandsink", "kmssink", "fakesink"],
    );
    // Decoders: GStreamer's libav / VA-API names -> the g2g decoders. The VA-API
    // names prefer the ffmpeg VAAPI hwaccel (`ffmpegvaapidec`, works on Mesa
    // radeonsi) and fall back to the cros-codecs `vaapidec` when only that
    // feature is on; the alias resolves to the first registered target.
    // `avdec_h264` falls back to VideoToolbox `vtdec` on macOS builds without
    // the ffmpeg feature.
    reg.register_alias("avdec_h264", &["ffmpegdec", "vtdec"]);
    reg.register_alias("vaapih264dec", &["ffmpegvaapidec", "vaapidec"]);
    // AV1 decode: gst's libav name and its aom plugin name -> the libdav1d
    // decoder, falling back to the pure-Rust re_rav1d decoder when only the
    // `rav1d` feature is built.
    reg.register_alias("avdec_av1", &["dav1ddec", "rav1ddec"]);
    reg.register_alias("av1dec", &["dav1ddec", "rav1ddec"]);
    reg.register_alias("vah264dec", &["ffmpegvaapidec", "vaapidec"]);
    // H.265 has no ffmpeg VAAPI hwaccel element here, so both gst names go
    // straight to the cros-codecs decoder.
    reg.register_alias("vaapih265dec", &["vaapidech265"]);
    reg.register_alias("vah265dec", &["vaapidech265"]);
    // VPx encoders: gst splits vp8enc / vp9enc; g2g has one vpxenc.
    reg.register_alias("vp8enc", &["vpxenc"]);
    reg.register_alias("vp9enc", &["vpxenc"]);
    // gst's libav software H.264 encoder name -> the ffmpeg encoder, software
    // first (`x264enc`, libx264), falling back to the NVENC-backed `ffmpegenc`
    // when only that is registered. The native NVENC encoder owns `nvh264enc`.
    reg.register_alias("avenc_h264", &["x264enc", "ffmpegenc"]);
    // QuickTime / MP4 muxer names -> the one fMP4 muxer (inert without std).
    reg.register_alias("qtmux", &["mp4mux"]);
    // gst's HLS sinks bundle their own muxer; g2g's takes one upstream, so the
    // names map to the single packager (the launch line keeps its `tsmux !`).
    for name in ["hlssink2", "hlssink3", "hlscmafsink"] {
        reg.register_alias(name, &["hlssink"]);
    }
    // gst-plugins-rs names for elements g2g has under the C plugin name.
    reg.register_alias("rsaudioecho", &["audioecho"]);
    reg.register_alias("rspngenc", &["pngenc"]);
    reg.register_alias("rav1enc", &["av1enc"]);
    reg.register_alias("reqwesthttpsrc", &["httpsrc"]);
    reg.register_alias("rsfilesrc", &["filesrc"]);
    reg.register_alias("rsfilesink", &["filesink"]);
    reg.register_alias("rsflvdemux", &["flvdemux"]);
    reg.register_alias("rswebpdec", &["webpdec"]);
    reg.register_alias("claxondec", &["ffmpegaudiodec"]);
    reg.register_alias("lewtondec", &["vorbisdec"]);
    reg.register_alias("isomp4mux", &["mp4mux"]);
    reg.register_alias("isofmp4mux", &["mp4mux"]);
    reg.register_alias("cmafmux", &["mp4mux"]);
    reg.register_alias("dashmp4mux", &["mp4mux"]);
    reg.register_alias("whepsrc", &["webrtcsrc"]);
    reg.register_alias("whipsink", &["webrtcsink"]);
    // gst's short AAC encoder name -> the libavcodec AAC encoder.
    reg.register_alias("aacenc", &["avenc_aac"]);
    // GStreamer's nvcodec names -> the native g2g NVENC / NVDEC elements. Resolve
    // to the registered target only when the feature is on (else the alias is
    // inert), the same first-registered rule as the VA-API names above.
    reg.register_alias("nvh264dec", &["nvdec"]);
    reg.register_alias("nvh264enc", &["nvenc"]);
    reg.register_alias("nvv4l2h264enc", &["nvenc"]);
    // WebM is Matroska, and `matroskamux` carries the `streamable` property the
    // gst name shares.
    reg.register_alias("webmmux", &["matroskamux"]);
    // gst's audio and video mixers -> the one mixer / compositor each. The
    // compositor takes videomixer's per-pad `xpos` / `ypos` / `zorder` / `alpha`.
    reg.register_alias("adder", &["audiomixer"]);
    reg.register_alias("liveadder", &["audiomixer"]);
    reg.register_alias("videomixer", &["compositor"]);
    // The remaining libav / plain decoder names the one ffmpeg decoder covers:
    // `ffmpegdec` decodes VP8, VP9, MPEG-2 and MPEG-4 part 2, `ffmpegaudiodec`
    // decodes AAC, MP2, MP3, AC-3 and FLAC.
    for name in [
        "vp8dec",
        "vp9dec",
        "mpeg2dec",
        "avdec_vp8",
        "avdec_vp9",
        "avdec_mpeg2video",
        "avdec_mpeg4",
    ] {
        reg.register_alias(name, &["ffmpegdec"]);
    }
    for name in [
        "mpg123audiodec",
        "flacdec",
        "a52dec",
        "faad",
        "fdkaacdec",
        "avdec_mp3",
        "avdec_aac",
        "avdec_ac3",
        "avdec_flac",
    ] {
        reg.register_alias(name, &["ffmpegaudiodec"]);
    }
    // gst's rtmp2 sink takes the same `location` URL as `rtmpsink`. `rtmp2src`
    // has no alias: `rtmpsrc` listens for a publisher on `address` / `port`.
    reg.register_alias("rtmp2sink", &["rtmpsink"]);
}

/// One feature-gated launch element: the name a pipeline writes, the cargo
/// feature that compiles it, and whether this build has it.
///
/// `compiled_in` mirrors the element's `#[cfg]` in `register_feature_gated`;
/// the catalog itself is un-cfg'd, so a build that lacks an element can still
/// name the feature that would provide it.
#[derive(Debug, Clone, Copy)]
pub struct FeatureGatedElement {
    pub name: &'static str,
    pub feature: &'static str,
    pub compiled_in: bool,
}

/// Every launch element name that only exists with a cargo feature, so an
/// "unknown element" can be answered with "rebuild with this feature" instead of
/// "no such element". `glimagesink` is deliberately absent: without `gl-sink` the
/// name resolves anyway, as an alias onto whatever display sink is built.
pub static FEATURE_GATED_ELEMENTS: &[FeatureGatedElement] = &{
    // Each row's gate is built from the feature (and the platform, where the
    // element is target-gated), so it cannot drift from the row it describes.
    macro_rules! rows {
        ($($name:literal => $feature:literal $(on $os:literal)?;)*) => {
            [$(FeatureGatedElement {
                name: $name,
                feature: $feature,
                compiled_in: cfg!(all(feature = $feature $(, target_os = $os)?)),
            }),*]
        };
    }
    rows! {
        "scriptelement" => "script-rhai";
        "scriptrouter" => "script-rhai";
        "cccombiner" => "metadata";
        "opusenc" => "opus";
        "opusdec" => "opus";
        "vorbisdec" => "vorbis";
        "av1enc" => "av1-encode";
        "vpxenc" => "vpx";
        "mjpegdec" => "mjpeg";
        "mjpegenc" => "mjpeg-encode";
        "pngdec" => "png";
        "pngenc" => "png";
        "webpdec" => "webp";
        "dav1ddec" => "dav1d";
        "rav1ddec" => "rav1d";
        "vulkanvideodec" => "vulkan-video";
        "rtspsrc" => "rtsp";
        "rtspsrcn" => "rtsp";
        "onvifsrc" => "onvif";
        "udpsrc" => "udp-ingress";
        "udpsink" => "udp-egress";
        // The alias needs its own row: without `udp-egress` there is no
        // `udpsink` for it to resolve to, so the name reads as unknown.
        "multiudpsink" => "udp-egress";
        "cotsink" => "udp-egress";
        "rtspserversink" => "rtsp-server";
        "rtspserversrc" => "rtsp-server";
        "rtspserversrcn" => "rtsp-server";
        "srtsrc" => "srt";
        "srtsink" => "srt";
        "srtpenc" => "srtp";
        "srtpdec" => "srtp";
        "dtlssrtpenc" => "dtls-srtp";
        "dtlssrtpdec" => "dtls-srtp";
        "tcpserversrc" => "tcp";
        "tcpclientsrc" => "tcp";
        "tcpserversink" => "tcp";
        "tcpclientsink" => "tcp";
        "shmsrc" => "shm";
        "shmsink" => "shm";
        "remotesrc" => "remote";
        "remotesink" => "remote";
        "remotewssrc" => "remote-ws";
        "remotewssink" => "remote-ws";
        "remotewstransform" => "remote-ws";
        "remotewtsrc" => "webtransport";
        "remotewtsink" => "webtransport";
        "remotewttransform" => "webtransport";
        "moqtsink" => "moqt";
        "moqtsrc" => "moqt";
        "moqtsessionsrc" => "moqt";
        "webrtcsrc" => "webrtc";
        "webrtcsink" => "webrtc";
        "webrtcsessionsink" => "webrtc";
        "webrtcwhepsessionsrc" => "webrtc";
        "livekitsink" => "webrtc-livekit";
        "livekitsrc" => "webrtc-livekit";
        "httpsrc" => "http-src";
        "hlssrc" => "hls";
        "dashsrc" => "dash";
        "rtmpsrc" => "rtmp";
        "rtmpsink" => "rtmp";
        "analyticsoverlay" => "analytics";
        "wgpucompositor" => "wgpu-sink";
        "gstwrap" => "gstreamer";
        "mp4mux" => "std";
        "localcudasrc" => "local-ipc" on "linux";
        "localcudasink" => "local-ipc" on "linux";
        "dmabufsrc" => "local-dmabuf" on "linux";
        "dmabufsink" => "local-dmabuf" on "linux";
        "v4l2src" => "v4l2" on "linux";
        "libcamerasrc" => "libcamera" on "linux";
        "ffmpegdec" => "ffmpeg" on "linux";
        "ffmpegaudiodec" => "ffmpeg" on "linux";
        "ffmpegvaapidec" => "ffmpeg" on "linux";
        "ffmpegenc" => "ffmpeg" on "linux";
        "x264enc" => "ffmpeg" on "linux";
        "avenc_aac" => "ffmpeg" on "linux";
        "vaapidec" => "vaapi" on "linux";
        "vaapidech265" => "vaapi" on "linux";
        "nvdec" => "nvdec" on "linux";
        "nvenc" => "nvenc" on "linux";
        "jpegxsenc" => "jpegxs" on "linux";
        "jpegxsdec" => "jpegxs" on "linux";
        "dmabuftowgpu" => "dmabuf-wgpu" on "linux";
        "wgputodmabuf" => "dmabuf-wgpu" on "linux";
        "waylandsink" => "wayland-sink" on "linux";
        "wgpusink" => "wgpu-present" on "linux";
        "kmssink" => "kms-sink" on "linux";
        "alsasink" => "alsa-sink" on "linux";
        "alsasrc" => "alsa-src" on "linux";
        "pulsesink" => "pulse-sink" on "linux";
        "pulsesrc" => "pulse-src" on "linux";
        "pipewiresink" => "pipewire" on "linux";
        "pipewiresrc" => "pipewire" on "linux";
        "pipewirevideosrc" => "pipewire" on "linux";
        "aaudiosrc" => "aaudio" on "android";
        "aaudiosink" => "aaudio" on "android";
        "camera2src" => "camera2" on "android";
        "mediacodecdec" => "mediacodec" on "android";
        "mediacodecdech265" => "mediacodec" on "android";
        "mediacodecenc" => "mediacodec" on "android";
        "mediacodecench265" => "mediacodec" on "android";
        "vtdec" => "vtdecode" on "macos";
        "vtdech265" => "vtdecode" on "macos";
        "vtenc_h264" => "vtencode" on "macos";
        "vtenc_h265" => "vtencode" on "macos";
        "metalvideosink" => "metal-sink" on "macos";
        "coreaudiosink" => "coreaudio" on "macos";
        "coreaudiosrc" => "coreaudio" on "macos";
        "avfvideosrc" => "avfoundation" on "macos";
        "avfaudiosrc" => "avfoundation" on "macos";
        "screencapturesrc" => "screencapture" on "macos";
        "mfvideosrc" => "mf-video-src" on "windows";
        "wasapisrc" => "wasapi-src" on "windows";
        "wasapisink" => "wasapi-sink" on "windows";
    }
};

/// The cargo feature that would compile the launch element named `name`, `None`
/// if the name is not in [`FEATURE_GATED_ELEMENTS`].
pub fn required_feature(name: &str) -> Option<&'static str> {
    FEATURE_GATED_ELEMENTS
        .iter()
        .find(|element| element.name == name)
        .map(|element| element.feature)
}

/// Register the feature- and platform-gated elements. Each block compiles only
/// when its `#[cfg]` (the same gate as the module in `lib.rs`) holds, so a build
/// without the feature is unchanged. Sources whose constructor needs a runtime
/// value (a URL, a socket, a device) are default-built with a placeholder; the
/// real value comes from a property / builder before use (the placeholder only
/// has to be side-effect-free, since `inspect` default-builds to read metadata).
#[allow(unused_variables)]
fn register_feature_gated(reg: &mut Registry) {
    // Rhai script transform (M580): `scriptelement script=... ! ...` runs a
    // per-frame `process(frame)` over a raw-video buffer. Pure Rust, no system dep.
    #[cfg(feature = "script-rhai")]
    reg.register_launch(LaunchFactory::of::<crate::script::ScriptElement>(
        "scriptelement",
        || Box::new(crate::script::ScriptElement::new()),
    ));
    // Rhai routing demux (M583): `scriptrouter name=r  r.0 ! ...  r.1 ! ...` sends
    // each buffer to the output port a `route(frame)` script picks (e.g. fan to
    // per-consumer `appsink` channels). Output count derived from the pad refs.
    #[cfg(feature = "script-rhai")]
    reg.register_demux(DemuxFactory::new("scriptrouter", |outputs| {
        Box::new(crate::script::ScriptRouter::new(outputs))
    }));

    // Timestamp burn-in (M1114): `videotestsrc ! timestampburn ! x264enc` marks
    // each frame with the CLOCK_MONOTONIC time it left, for the same-metric
    // latency bench to read back after the stream has crossed a network.
    #[cfg(all(unix, feature = "latency-bench"))]
    reg.register_launch(LaunchFactory::of::<crate::timestampburn::TimestampBurn>(
        "timestampburn",
        || Box::new(crate::timestampburn::TimestampBurn::new()),
    ));

    // Codecs (cross-platform).
    reg.register_launch(LaunchFactory::of::<MulawEnc>("mulawenc", || {
        Box::new(MulawEnc::new())
    }));
    reg.register_launch(LaunchFactory::of::<MulawDec>("mulawdec", || {
        Box::new(MulawDec::new())
    }));
    reg.register_launch(LaunchFactory::of::<AlawEnc>("alawenc", || {
        Box::new(AlawEnc::new())
    }));
    reg.register_launch(LaunchFactory::of::<AlawDec>("alawdec", || {
        Box::new(AlawDec::new())
    }));
    reg.register_launch(LaunchFactory::of::<AdpcmEnc>("adpcmenc", || {
        Box::new(AdpcmEnc::new())
    }));
    reg.register_launch(LaunchFactory::of::<AdpcmDec>("adpcmdec", || {
        Box::new(AdpcmDec::new())
    }));
    #[cfg(feature = "opus")]
    {
        reg.register_launch(LaunchFactory::of::<OpusEnc>("opusenc", || {
            Box::new(OpusEnc::new())
        }));
        reg.register_launch(LaunchFactory::of::<OpusDec>("opusdec", || {
            Box::new(OpusDec::new())
        }));
    }
    #[cfg(feature = "vorbis")]
    reg.register_launch(LaunchFactory::of::<VorbisDec>("vorbisdec", || {
        Box::new(VorbisDec::new())
    }));
    #[cfg(feature = "av1-encode")]
    reg.register_launch(LaunchFactory::of::<Av1Enc>("av1enc", || {
        Box::new(Av1Enc::new())
    }));
    #[cfg(feature = "vpx")]
    reg.register_launch(LaunchFactory::of::<VpxEnc>("vpxenc", || {
        Box::new(VpxEnc::new())
    }));
    #[cfg(feature = "mjpeg")]
    reg.register_launch(LaunchFactory::of::<MjpegDec>("mjpegdec", || {
        Box::new(MjpegDec::new())
    }));
    #[cfg(feature = "dav1d")]
    reg.register_launch(LaunchFactory::of::<Dav1dDec>("dav1ddec", || {
        Box::new(Dav1dDec::new())
    }));
    #[cfg(feature = "rav1d")]
    reg.register_launch(LaunchFactory::of::<Rav1dDec>("rav1ddec", || {
        Box::new(Rav1dDec::new())
    }));
    #[cfg(feature = "mjpeg-encode")]
    reg.register_launch(LaunchFactory::of::<MjpegEnc>("mjpegenc", || {
        Box::new(MjpegEnc::new())
    }));
    #[cfg(feature = "png")]
    reg.register_launch(LaunchFactory::of::<PngDec>("pngdec", || {
        Box::new(PngDec::new())
    }));
    #[cfg(feature = "png")]
    reg.register_launch(LaunchFactory::of::<PngEnc>("pngenc", || {
        Box::new(PngEnc::new())
    }));
    #[cfg(feature = "webp")]
    reg.register_launch(LaunchFactory::of::<WebPDec>("webpdec", || {
        Box::new(WebPDec::new())
    }));

    // Network sources / sinks.
    #[cfg(feature = "rtsp")]
    reg.register_source(SourceFactory::new(
        "rtspsrc",
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        },
        || Box::new(RtspSrc::new("")),
    ));
    // Multi-track playback of the same stream (M1122): one session, video on the
    // first linked pad and the SDP's audio on the second
    // (`rtspsrcn name=s location=...  s. ! ...  s. ! ...`).
    #[cfg(feature = "rtsp")]
    reg.register_fanout_src(g2g_core::runtime::FanoutSrcFactory::new(
        "rtspsrcn",
        |outputs| Box::new(crate::rtspsrcn::RtspSrcN::new("").with_outputs(outputs)),
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        },
        || Box::new(OnvifSrc::new("")),
    ));
    // Plain TCP byte streams (M1068). The declared caps are nominal, like
    // `srtsrc`'s: nothing on the wire says what the bytes are, so
    // `bytestream-format` names the container and a downstream `typefind` can
    // re-declare it from the content.
    #[cfg(feature = "tcp")]
    reg.register_source(SourceFactory::new(
        "tcpserversrc",
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
        || Box::new(TcpServerSrc::default()),
    ));
    #[cfg(feature = "tcp")]
    reg.register_source(SourceFactory::new(
        "tcpclientsrc",
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
        || Box::new(TcpClientSrc::default()),
    ));
    // Shared memory (M1081). The declared caps are nominal for the same reason
    // as the TCP sources': the shmpipe protocol has no field for caps, so
    // `bytestream-format` or `caps` says what the bytes are.
    #[cfg(all(unix, feature = "shm"))]
    reg.register_source(SourceFactory::new(
        "shmsrc",
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
        || Box::new(ShmSrc::default()),
    ));
    #[cfg(feature = "udp-ingress")]
    reg.register_source(SourceFactory::new(
        "udpsrc",
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        },
        || Box::new(WebRtcWhepSrc::new("")),
    ));
    #[cfg(feature = "tcp")]
    reg.register_launch(LaunchFactory::of::<TcpServerSink>("tcpserversink", || {
        Box::new(TcpServerSink::default())
    }));
    #[cfg(feature = "tcp")]
    reg.register_launch(LaunchFactory::of::<TcpClientSink>("tcpclientsink", || {
        Box::new(TcpClientSink::default())
    }));
    #[cfg(all(unix, feature = "shm"))]
    reg.register_launch(LaunchFactory::of::<ShmSink>("shmsink", || {
        Box::new(ShmSink::default())
    }));
    #[cfg(feature = "udp-egress")]
    reg.register_launch(LaunchFactory::of::<UdpSink>("udpsink", || {
        Box::new(UdpSink::new("127.0.0.1:5004".parse().unwrap()))
    }));
    // RFC 7714 packet protection (M1098): both build keyless and refuse to
    // configure until `key=` supplies one.
    #[cfg(feature = "srtp")]
    reg.register_launch(LaunchFactory::of::<SrtpEnc>("srtpenc", || {
        Box::new(SrtpEnc::default())
    }));
    #[cfg(feature = "srtp")]
    reg.register_launch(LaunchFactory::of::<SrtpDec>("srtpdec", || {
        Box::new(SrtpDec::default())
    }));
    // DTLS-SRTP key delivery (M1100): the pair runs a handshake over the media
    // socket, so neither takes a `key=`. `dtlssrtpenc` is a fan-in, one pad per
    // flow with the flow read from each pad's caps, and `dtlssrtpdec` a fan-out
    // with RTP on port 0 and RTCP on port 1. The two find each other by
    // `connection-id`:
    // `rtp. ! e.  rtcp. ! e.  dtlssrtpenc name=e connection-id=x is-client=true
    //  ! udpsink   udpsrc bytestream-format=dtls ! dtlssrtpdec name=d
    //  connection-id=x  d. ! ...  d. ! ...`.
    #[cfg(feature = "dtls-srtp")]
    reg.register_muxer(MuxerFactory::new("dtlssrtpenc", |inputs| {
        Box::new(crate::dtlssrtpenc::DtlsSrtpEnc::new(inputs))
    }));
    #[cfg(feature = "dtls-srtp")]
    reg.register_demux(g2g_core::runtime::DemuxFactory::new(
        "dtlssrtpdec",
        |outputs| Box::new(crate::dtlssrtpdec::DtlsSrtpDec::new(outputs)),
    ));
    // Cursor-on-Target bridge (M811): a demuxed STANAG 4609 metadata stream's ST
    // 0601 local sets become CoT events on a TAK network, e.g.
    // `tsdemux stream=klv ! cotsink host=239.2.3.1 port=6969`.
    #[cfg(feature = "udp-egress")]
    reg.register_launch(LaunchFactory::of::<crate::cotsink::CotSink>(
        "cotsink",
        || {
            Box::new(crate::cotsink::CotSink::new(
                "239.2.3.1:6969".parse().unwrap(),
            ))
        },
    ));
    #[cfg(feature = "rtsp-server")]
    reg.register_launch(LaunchFactory::of::<RtspServerSink>(
        "rtspserversink",
        || Box::new(RtspServerSink::new("0.0.0.0:8554".parse().unwrap())),
    ));
    #[cfg(feature = "rtsp-server")]
    reg.register_source(SourceFactory::new(
        "rtspserversrc",
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        },
        || Box::new(RtspServerSrc::new("0.0.0.0:8554".parse().unwrap())),
    ));
    // Concurrent multi-publisher ingest (M863): one endpoint, one recording
    // publisher per linked pad (`rtspserversrcn name=s  s. ! ...  s. ! ...`).
    #[cfg(feature = "rtsp-server")]
    reg.register_fanout_src(g2g_core::runtime::FanoutSrcFactory::new(
        "rtspserversrcn",
        |n| {
            Box::new(crate::rtspserversrcn::RtspServerSrcN::new(
                "0.0.0.0:8554".parse().unwrap(),
                n,
            ))
        },
    ));
    #[cfg(feature = "srt")]
    reg.register_source(SourceFactory::new(
        "srtsrc",
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
        || Box::new(SrtSrc::new("0.0.0.0:9000".parse().unwrap())),
    ));
    #[cfg(feature = "srt")]
    reg.register_launch(LaunchFactory::of::<SrtSink>("srtsink", || {
        Box::new(SrtSink::new("127.0.0.1:9000".parse().unwrap()))
    }));
    // Distributed-graph transport pair (M551). `remotesrc` produces whatever the
    // sender negotiates: the declared caps are a nominal catalog default, since
    // the real caps are discovered from the wire on connect (`intercept_caps`).
    #[cfg(feature = "remote")]
    reg.register_source(SourceFactory::new(
        "remotesrc",
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        },
        || Box::new(RemoteSrc::new("0.0.0.0:9600".parse().unwrap())),
    ));
    #[cfg(feature = "remote")]
    reg.register_launch(LaunchFactory::of::<RemoteSink>("remotesink", || {
        Box::new(RemoteSink::new("127.0.0.1:9600".parse().unwrap()))
    }));
    // WebSocket sibling of the M551 pair (M554): same wire codec, one packet per
    // binary WebSocket message. `remotewssrc` also discovers its caps from the
    // wire on connect, so the declared caps here are a nominal catalog default.
    #[cfg(feature = "remote-ws")]
    reg.register_source(SourceFactory::new(
        "remotewssrc",
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        },
        || Box::new(RemoteWsSrc::new("0.0.0.0:9601".parse().unwrap())),
    ));
    #[cfg(feature = "remote-ws")]
    reg.register_launch(LaunchFactory::of::<RemoteWsSink>("remotewssink", || {
        Box::new(RemoteWsSink::new("ws://127.0.0.1:9601"))
    }));
    // Remote-transform (M555): offload a middle stage over a WebSocket, round-trip.
    #[cfg(feature = "remote-ws")]
    reg.register_launch(LaunchFactory::of::<RemoteWsTransform>(
        "remotewstransform",
        || Box::new(RemoteWsTransform::new("ws://127.0.0.1:9602")),
    ));
    // WebTransport sibling of the same family (M901): the same wire codec over one
    // reliable bidirectional QUIC stream. `remotewtsrc` also discovers its caps
    // from the wire on connect, so the declared caps here are a nominal catalog
    // default; it needs `certificate` / `private-key` to start.
    #[cfg(feature = "webtransport")]
    reg.register_source(SourceFactory::new(
        "remotewtsrc",
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        },
        || Box::new(RemoteWtSrc::new("0.0.0.0:9603".parse().unwrap())),
    ));
    #[cfg(feature = "webtransport")]
    reg.register_launch(LaunchFactory::of::<RemoteWtSink>("remotewtsink", || {
        Box::new(RemoteWtSink::new("https://127.0.0.1:9603"))
    }));
    #[cfg(feature = "webtransport")]
    reg.register_launch(LaunchFactory::of::<RemoteWtTransform>(
        "remotewttransform",
        || Box::new(RemoteWtTransform::new("https://127.0.0.1:9604")),
    ));
    // MoQ Transport publisher (M902) and subscriber (M903): fMP4 in, MOQT groups
    // and objects out to an IETF relay over the same WebTransport carrier, and
    // back the other way.
    #[cfg(feature = "moqt")]
    reg.register_launch(LaunchFactory::of::<MoqtSink>("moqtsink", || {
        Box::new(MoqtSink::new("https://127.0.0.1:4443/", "g2g"))
    }));
    // The multi-track shape of the same subscriber: one session, one pad per
    // track (`moqtsessionsrc name=s tracks=1.m4s,2.m4s  s. ! ...  s. ! ...`).
    #[cfg(feature = "moqt")]
    reg.register_fanout_src(g2g_core::runtime::FanoutSrcFactory::new(
        "moqtsessionsrc",
        |outputs| {
            Box::new(
                crate::moqtsessionsrc::MoqtSessionSrc::new("https://127.0.0.1:4443/", "g2g")
                    .with_outputs(outputs),
            )
        },
    ));
    #[cfg(feature = "moqt")]
    reg.register_source(SourceFactory::new(
        "moqtsrc",
        Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        },
        || Box::new(MoqtSrc::new("https://127.0.0.1:4443/", "g2g")),
    ));
    // Local zero-copy transports (M556 / M557): same-machine GPU-resident (CUDA
    // IPC) and vendor-neutral (DMABUF over SCM_RIGHTS) sink/src pairs. Like the
    // remote pair, the source discovers its real caps from the peer on connect, so
    // the declared caps here are a nominal catalog default; the `location` property
    // sets the Unix socket path.
    #[cfg(all(target_os = "linux", feature = "local-ipc"))]
    reg.register_source(
        SourceFactory::new(
            "localcudasrc",
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            || Box::new(LocalCudaSrc::new("/tmp/g2g-localcuda.sock")),
        )
        .with_experimental(),
    );
    #[cfg(all(target_os = "linux", feature = "local-ipc"))]
    reg.register_launch(
        LaunchFactory::of::<LocalCudaSink>("localcudasink", || {
            Box::new(LocalCudaSink::new("/tmp/g2g-localcuda.sock"))
        })
        .with_experimental(),
    );
    #[cfg(all(target_os = "linux", feature = "local-dmabuf"))]
    reg.register_source(
        SourceFactory::new(
            "dmabufsrc",
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            || Box::new(DmaBufSrc::new("/tmp/g2g-dmabuf.sock")),
        )
        .with_experimental(),
    );
    #[cfg(all(target_os = "linux", feature = "local-dmabuf"))]
    reg.register_launch(
        LaunchFactory::of::<DmaBufSink>("dmabufsink", || {
            Box::new(DmaBufSink::new("/tmp/g2g-dmabuf.sock"))
        })
        .with_experimental(),
    );
    #[cfg(feature = "http-src")]
    reg.register_source(SourceFactory::new(
        "httpsrc",
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
        || {
            Box::new(HttpSrc::new(
                "",
                Caps::ByteStream {
                    encoding: ByteStreamEncoding::MpegTs,
                },
            ))
        },
    ));
    #[cfg(feature = "hls")]
    reg.register_source(SourceFactory::new(
        "hlssrc",
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
        || Box::new(HlsSrc::new("")),
    ));
    #[cfg(feature = "dash")]
    reg.register_source(SourceFactory::new(
        "dashsrc",
        Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        },
        || Box::new(DashSrc::new("")),
    ));
    #[cfg(feature = "rtmp")]
    reg.register_source(SourceFactory::new(
        "rtmpsrc",
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Flv,
        },
        || Box::new(RtmpSrc::new("0.0.0.0:1935".parse().unwrap())),
    ));

    // Linux capture / decode / display.
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    reg.register_source(
        SourceFactory::new(
            "v4l2src",
            Caps::RawVideo {
                format: RawVideoFormat::Yuyv,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            || Box::new(V4l2Src::new("/dev/video0")),
        )
        .with_experimental(),
    );
    // libcamera capture: NV12 (else YUYV). Geometry/format are negotiated with
    // the camera at startup, so the declared caps are fully open.
    #[cfg(all(target_os = "linux", feature = "libcamera"))]
    reg.register_source(
        SourceFactory::new(
            "libcamerasrc",
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            || Box::new(LibCameraSrc::new()),
        )
        .with_experimental(),
    );
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register_launch(LaunchFactory::of::<FfmpegH264Dec>("ffmpegdec", || {
        // Auto preserves source chroma (M685/M686): a `decodebin` chain whose
        // downstream pins 4:2:2 / 4:4:4 negotiates it, while a 4:2:0 source (or
        // an I420 request) still resolves to I420. A downstream that needs NV12
        // sets `output-format=nv12` explicitly.
        Box::new(FfmpegH264Dec::new().with_output_format(crate::ffmpegdec::OutputFormat::Auto))
    }));
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register_launch(LaunchFactory::of::<crate::ffmpegaudiodec::FfmpegAudioDec>(
        "ffmpegaudiodec",
        || Box::new(crate::ffmpegaudiodec::FfmpegAudioDec::new()),
    ));
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    reg.register_launch(
        LaunchFactory::of::<FfmpegH264Dec>("ffmpegvaapidec", || {
            Box::new(FfmpegH264Dec::new().with_backend(FfmpegBackend::Vaapi))
        })
        .with_experimental(),
    );
    // Vendor-neutral Vulkan Video hardware decoder (M493; H.264 / H.265 / AV1 since
    // M517): compressed video in, NV12 system memory or (zero-copy) RGBA
    // WgpuTexture out, on the same Vulkan device wgpu runs (AMD/NVIDIA/Intel). The
    // launch name; it is also an auto-plug candidate (registered in
    // `register_autoplug_candidates`, preferred for a WgpuTexture consumer).
    #[cfg(feature = "vulkan-video")]
    reg.register_launch(
        LaunchFactory::of::<crate::vulkanvideo::VulkanVideoDec>("vulkanvideodec", || {
            Box::new(crate::vulkanvideo::VulkanVideoDec::new())
        })
        .with_experimental(),
    );
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
    reg.register_launch(LaunchFactory::of::<crate::ffmpegaacenc::FfmpegAacEnc>(
        "avenc_aac",
        || Box::new(crate::ffmpegaacenc::FfmpegAacEnc::new()),
    ));
    #[cfg(all(target_os = "linux", feature = "vaapi"))]
    reg.register_launch(
        LaunchFactory::of::<VaapiH264Dec>("vaapidec", || Box::new(VaapiH264Dec::new()))
            .with_experimental(),
    );
    #[cfg(all(target_os = "linux", feature = "vaapi"))]
    reg.register_launch(
        LaunchFactory::of::<VaapiH265Dec>("vaapidech265", || Box::new(VaapiH265Dec::new()))
            .with_experimental(),
    );
    // Native NVIDIA Video Codec SDK elements (M269 / M270): zero-copy CUDA NV12
    // <-> H.264, the gst-`nvcodec`-style pair. Explicit-select by name.
    #[cfg(all(target_os = "linux", feature = "nvdec"))]
    reg.register_launch(
        LaunchFactory::of::<NvDec>("nvdec", || Box::new(NvDec::new())).with_experimental(),
    );
    #[cfg(all(target_os = "linux", feature = "nvenc"))]
    reg.register_launch(
        LaunchFactory::of::<NvEnc>("nvenc", || Box::new(NvEnc::new())).with_experimental(),
    );
    // JPEG XS codec (M605): the ST 2110-22 compressed essence, via SVT-JPEG-XS.
    #[cfg(all(target_os = "linux", feature = "jpegxs"))]
    reg.register_launch(
        LaunchFactory::of::<SvtJpegXsEnc>("jpegxsenc", || Box::new(SvtJpegXsEnc::new()))
            .with_experimental(),
    );
    #[cfg(all(target_os = "linux", feature = "jpegxs"))]
    reg.register_launch(
        LaunchFactory::of::<SvtJpegXsDec>("jpegxsdec", || Box::new(SvtJpegXsDec::new()))
            .with_experimental(),
    );
    #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
    reg.register_launch(
        LaunchFactory::of::<crate::dmabufwgpu::DmaBufToWgpu>("dmabuftowgpu", || {
            Box::new(crate::dmabufwgpu::DmaBufToWgpu::new())
        })
        .with_experimental(),
    );
    // Export mirror (M559): a GPU-resident wgpu buffer out to a dma-buf fd.
    #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
    reg.register_launch(
        LaunchFactory::of::<crate::wgpudmabuf::WgpuToDmaBuf>("wgputodmabuf", || {
            Box::new(crate::wgpudmabuf::WgpuToDmaBuf::new())
        })
        .with_experimental(),
    );
    // Reverse GStreamer bridge: host an unported GStreamer element in a g2g graph.
    // No pad templates (caps are what the negotiation settles + the `output-caps`
    // property declares), like `identity`.
    #[cfg(feature = "gstreamer")]
    reg.register_launch(
        LaunchFactory::new("gstwrap", Vec::new(), || {
            Box::new(crate::gstwrap::GstWrap::new())
        })
        .with_experimental(),
    );
    // Each of the three presents on Wayland, so each declares the same
    // compositor check; it only changes what the `auto*sink` aliases fall
    // through to, never what a pipeline naming the sink outright does.
    #[cfg(all(target_os = "linux", feature = "wayland-sink"))]
    reg.register_launch(
        LaunchFactory::new("waylandsink", Vec::new(), || Box::new(WaylandSink::new()))
            .with_usable(crate::waylanddisplay::compositor_reachable)
            .with_experimental(),
    );
    // Vendor-neutral EGL / GL ES display sink under its gst name; it declares
    // NV12 + RGBA pad templates, so decodebin can auto-plug onto it.
    #[cfg(all(target_os = "linux", feature = "gl-sink"))]
    reg.register_launch(
        LaunchFactory::of::<crate::glsink::GlSink>("glimagesink", || {
            Box::new(crate::glsink::GlSink::new())
        })
        .with_usable(crate::waylanddisplay::compositor_reachable)
        .with_experimental(),
    );
    // Windowed wgpu display sink: it takes GPU-resident frames as they are, so a
    // decoder that keeps them on the GPU reaches the screen with no upload. Its
    // NV12 + RGBA pad templates let decodebin auto-plug onto it.
    #[cfg(all(target_os = "linux", feature = "wgpu-present"))]
    reg.register_launch(
        LaunchFactory::of::<crate::wgpupresent::WgpuPresentSink>("wgpusink", || {
            Box::new(crate::wgpupresent::WgpuPresentSink::new())
        })
        .with_usable(crate::waylanddisplay::compositor_reachable)
        .with_experimental(),
    );
    // WebRTC WHIP egress; the `location` property targets the endpoint. The URL
    // defaults empty (set it via `webrtcsink location=...`); publishing starts
    // on the first frame.
    #[cfg(feature = "webrtc")]
    reg.register_launch(LaunchFactory::new("webrtcsink", Vec::new(), || {
        Box::new(WebRtcSink::new(""))
    }));
    // Multi-track WHIP session sink (M725): a terminal fan-in whose track kinds
    // come from each linked pad's caps (video pads group as simulcast layers in
    // pad order): `v ! s.  a ! s.  webrtcsessionsink name=s location=...`.
    #[cfg(feature = "webrtc")]
    reg.register_muxer(MuxerFactory::new("webrtcsessionsink", |inputs| {
        Box::new(crate::webrtcsession::WebRtcSessionSink::new("").with_inputs(inputs))
    }));
    // LiveKit publisher, the same terminal fan-in shape with room/credential
    // properties (`livekitsink name=s url=... room=... api-key=...`).
    #[cfg(feature = "webrtc-livekit")]
    reg.register_muxer(MuxerFactory::new("livekitsink", |inputs| {
        Box::new(crate::livekitsink::LiveKitSink::new("", "", "g2g").with_inputs(inputs))
    }));
    // Terminal fan-out session sources (M727): output 0 = H.264 video, output
    // 1 = Opus audio (`livekitsrc name=s url=...  s. ! ...  s. ! ...`).
    #[cfg(feature = "webrtc-livekit")]
    reg.register_fanout_src(g2g_core::runtime::FanoutSrcFactory::new(
        "livekitsrc",
        |_n| Box::new(crate::livekitsrc::LiveKitSrc::new("", "", "g2g-sub")),
    ));
    // Same shape for the multi-track WHEP subscriber; `location` targets the
    // endpoint (`webrtcwhepsessionsrc name=s location=...`).
    #[cfg(feature = "webrtc")]
    reg.register_fanout_src(g2g_core::runtime::FanoutSrcFactory::new(
        "webrtcwhepsessionsrc",
        |_n| Box::new(crate::webrtcwhepsession::WebRtcWhepSessionSrc::new("")),
    ));
    #[cfg(all(target_os = "linux", feature = "kms-sink"))]
    reg.register_launch(
        LaunchFactory::new("kmssink", Vec::new(), || Box::new(KmsSink::new())).with_experimental(),
    );
    #[cfg(all(target_os = "linux", feature = "alsa-sink"))]
    reg.register_launch(
        LaunchFactory::of::<AlsaSink>("alsasink", || Box::new(AlsaSink::new())).with_experimental(),
    );
    #[cfg(all(target_os = "linux", feature = "pulse-sink"))]
    reg.register_launch(
        LaunchFactory::of::<PulseSink>("pulsesink", || Box::new(PulseSink::new()))
            .with_experimental(),
    );
    // Linux audio capture (M886), the non-PipeWire mic paths.
    #[cfg(all(target_os = "linux", feature = "alsa-src"))]
    reg.register_source(
        SourceFactory::new(
            "alsasrc",
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 48_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            },
            || Box::new(AlsaSrc::new()),
        )
        .with_experimental(),
    );
    #[cfg(all(target_os = "linux", feature = "pulse-src"))]
    reg.register_source(
        SourceFactory::new(
            "pulsesrc",
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 48_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            },
            || Box::new(PulseSrc::new()),
        )
        .with_experimental(),
    );
    // PipeWire: audio render / capture plus video capture (M890). The audio
    // capture element opens S16LE stereo at 48 kHz.
    #[cfg(all(target_os = "linux", feature = "pipewire"))]
    reg.register_launch(
        LaunchFactory::of::<PipeWireSink>("pipewiresink", || Box::new(PipeWireSink::new()))
            .with_experimental(),
    );
    #[cfg(all(target_os = "linux", feature = "pipewire"))]
    reg.register_source(
        SourceFactory::new(
            "pipewiresrc",
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 48_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            },
            || Box::new(PipeWireSrc::new()),
        )
        .with_experimental(),
    );
    // Geometry and format are negotiated with the node at startup, so the
    // declared caps stay open (like v4l2src / libcamerasrc).
    #[cfg(all(target_os = "linux", feature = "pipewire"))]
    reg.register_source(
        SourceFactory::new(
            "pipewirevideosrc",
            Caps::RawVideo {
                format: RawVideoFormat::I420,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            || Box::new(PipeWireVideoSrc::new()),
        )
        .with_experimental(),
    );
    // Android AAudio PCM render (M307); the gst analog is `aaudiosink`.
    #[cfg(all(target_os = "android", feature = "aaudio"))]
    reg.register_launch(
        LaunchFactory::of::<AAudioSink>("aaudiosink", || Box::new(AAudioSink::new()))
            .with_experimental(),
    );
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
            assert!(
                reg.make_element(name).is_some(),
                "registry resolves `{name}`"
            );
        }
        // The native decoder is also an auto-plug candidate (registered after the
        // CPU decoders so it does not out-rank them; see register_autoplug_candidates).
        assert!(
            reg.element_names().contains(&"nvdec"),
            "nvdec is an autoplug factory"
        );
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
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
        let cpu = reg
            .autoplug_names(&h264(), &is_raw_video, 4)
            .expect("a decoder reaches raw");
        assert_eq!(
            cpu.last(),
            Some(&"ffmpegdec"),
            "default decode stays on the CPU: {cpu:?}"
        );
        // Cuda preference: the domain-aware search prefers the native NVDEC.
        let gpu = reg
            .autoplug_names_preferring(&h264(), &is_raw_video, 4, MemoryDomainKind::Cuda)
            .expect("a decoder reaches raw");
        assert_eq!(
            gpu.last(),
            Some(&"nvdec"),
            "Cuda preference prefers NvDec: {gpu:?}"
        );
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
            assert!(
                reg.make_element(name).is_some(),
                "registry resolves `{name}`"
            );
        }
    }
}

#[cfg(test)]
mod feature_catalog_tests {
    use super::*;

    /// The catalog has to agree with the live registry both ways: a name listed
    /// here that resolves without its feature is a baseline element wrongly
    /// listed (the lint would advise a pointless rebuild), and a name that does
    /// not resolve with its feature on means the row names the wrong feature.
    #[test]
    fn the_feature_catalog_matches_the_live_registry() {
        let reg = default_registry();
        for element in FEATURE_GATED_ELEMENTS {
            assert_eq!(
                reg.knows_element(element.name),
                element.compiled_in,
                "`{}` (feature `{}`)",
                element.name,
                element.feature
            );
        }
    }

    #[test]
    fn every_catalog_name_is_listed_once() {
        let mut seen = alloc::collections::BTreeSet::new();
        for element in FEATURE_GATED_ELEMENTS {
            assert!(seen.insert(element.name), "`{}` listed twice", element.name);
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
        assert!(
            reg.make_muxer("qtmux", 2).is_some(),
            "qtmux resolves to the mp4mux fan-in"
        );
        assert!(
            reg.make_muxer("mp4mux", 2).is_some(),
            "the alias target still builds directly"
        );
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

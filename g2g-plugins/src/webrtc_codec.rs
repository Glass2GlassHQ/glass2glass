//! The media the str0m WebRTC elements carry: the video codecs they packetize
//! and depacketize, the sinks' accepted caps, the sources' placeholder caps, and
//! the `video-codec` property every receiving element exposes.

use alloc::vec::Vec;

use str0m::format::Codec;
use str0m::media::{Frequency, MediaKind, MediaTime};
use str0m::RtcConfig;

use g2g_core::{
    AudioFormat, Caps, Dim, PropError, PropKind, PropValue, PropertySpec, Rate, VideoCodec,
};

/// A video codec the WebRTC elements carry, each with a str0m packetizer. The
/// pipeline framing matches what str0m reads and writes: Annex-B for H.264 and
/// H.265, one raw frame for VP8 and VP9, one low-overhead temporal unit for AV1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WebRtcVideoCodec {
    #[default]
    H264,
    Vp8,
    Vp9,
    H265,
    Av1,
}

impl WebRtcVideoCodec {
    /// Every carried codec, in the order the sinks list their accepted caps.
    pub const ALL: [Self; 5] = [Self::H264, Self::Vp8, Self::Vp9, Self::H265, Self::Av1];

    pub fn from_video_codec(codec: VideoCodec) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.video_codec() == codec)
    }

    pub fn video_codec(self) -> VideoCodec {
        match self {
            Self::H264 => VideoCodec::H264,
            Self::Vp8 => VideoCodec::Vp8,
            Self::Vp9 => VideoCodec::Vp9,
            Self::H265 => VideoCodec::H265,
            Self::Av1 => VideoCodec::Av1,
        }
    }

    pub(crate) fn str0m(self) -> Codec {
        match self {
            Self::H264 => Codec::H264,
            Self::Vp8 => Codec::Vp8,
            Self::Vp9 => Codec::Vp9,
            Self::H265 => Codec::H265,
            Self::Av1 => Codec::Av1,
        }
    }

    /// Turn this codec on in an `Rtc` under construction.
    pub(crate) fn enable(self, config: RtcConfig) -> RtcConfig {
        match self {
            Self::H264 => config.enable_h264(true),
            Self::Vp8 => config.enable_vp8(true),
            Self::Vp9 => config.enable_vp9(true),
            Self::H265 => config.enable_h265(true),
            Self::Av1 => config.enable_av1(true),
        }
    }

    /// The `video-codec` property value.
    pub fn name(self) -> &'static str {
        match self {
            Self::H264 => "h264",
            Self::Vp8 => "vp8",
            Self::Vp9 => "vp9",
            Self::H265 => "h265",
            Self::Av1 => "av1",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|c| c.name().eq_ignore_ascii_case(name))
    }

    /// The codec of a compressed-video caps, `None` for audio or a codec with no
    /// str0m packetizer.
    pub(crate) fn from_caps(caps: &Caps) -> Option<Self> {
        match caps {
            Caps::CompressedVideo { codec, .. } => Self::from_video_codec(*codec),
            _ => None,
        }
    }

    /// Sink caps: this codec at any geometry and framerate.
    pub(crate) fn caps_any(self) -> Caps {
        Caps::CompressedVideo {
            codec: self.video_codec(),
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    /// Source caps: geometry is unknown until the in-band parameter sets, so a
    /// `Range` placeholder that fixates (a downstream parser recovers the real
    /// dimensions and re-announces them). `fixate()` rejects `Dim::Any`.
    pub(crate) fn caps_placeholder(self) -> Caps {
        Caps::CompressedVideo {
            codec: self.video_codec(),
            width: Dim::Range {
                min: PLACEHOLDER_MIN_DIM,
                max: PLACEHOLDER_MAX_DIM,
            },
            height: Dim::Range {
                min: PLACEHOLDER_MIN_DIM,
                max: PLACEHOLDER_MAX_DIM,
            },
            framerate: Rate::Range {
                min_q16: PLACEHOLDER_MIN_FPS << 16,
                max_q16: PLACEHOLDER_MAX_FPS << 16,
            },
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    /// A depacketized unit in the pipeline framing. The AV1 RTP payload format
    /// carries no temporal delimiter OBU, so one goes back on the front.
    pub(crate) fn pipeline_unit(self, depacketized: &[u8]) -> Vec<u8> {
        let opens_with_delimiter = depacketized
            .first()
            .is_some_and(|b| (b >> OBU_TYPE_SHIFT) & OBU_TYPE_MASK == OBU_TEMPORAL_DELIMITER);
        if self != Self::Av1 || opens_with_delimiter {
            return depacketized.to_vec();
        }
        let mut unit = AV1_TEMPORAL_DELIMITER.to_vec();
        unit.extend_from_slice(depacketized);
        unit
    }

    /// Whether a depacketized access unit of this codec is a keyframe.
    pub(crate) fn is_keyframe(self, au: &[u8]) -> bool {
        match self {
            Self::Vp8 => crate::vp8parse::is_keyframe(au),
            Self::Vp9 => crate::vp9parse::is_keyframe(au),
            Self::H264 | Self::H265 | Self::Av1 => {
                crate::annexb::au_is_keyframe(self.video_codec(), au)
            }
        }
    }
}

/// A low-overhead temporal delimiter OBU: type 2 with has_size_field, size 0.
const AV1_TEMPORAL_DELIMITER: [u8; 2] = [0x12, 0x00];
const OBU_TYPE_SHIFT: u8 = 3;
const OBU_TYPE_MASK: u8 = 0x0f;
const OBU_TEMPORAL_DELIMITER: u8 = 2;

const PLACEHOLDER_MIN_DIM: u32 = 2;
const PLACEHOLDER_MAX_DIM: u32 = 8192;
const PLACEHOLDER_MIN_FPS: u32 = 1;
const PLACEHOLDER_MAX_FPS: u32 = 240;

/// The `video-codec` property of the receiving elements: which codec the source
/// offers and emits, fixed before negotiation because a graph fixates its caps
/// before the handshake can tell what the peer sends.
pub(crate) const VIDEO_CODEC_PROP: PropertySpec = PropertySpec::new(
    "video-codec",
    PropKind::Str,
    "video codec to negotiate and emit: h264 | vp8 | vp9 | h265 | av1",
)
.with_default("h264")
.with_enum_values("h264 | vp8 | vp9 | h265 | av1");

pub(crate) fn set_video_codec_property(
    slot: &mut WebRtcVideoCodec,
    value: &PropValue,
) -> Result<(), PropError> {
    let name = value.as_str().ok_or(PropError::Type)?;
    *slot = WebRtcVideoCodec::parse(name).ok_or(PropError::Value)?;
    Ok(())
}

pub(crate) fn get_video_codec_property(codec: WebRtcVideoCodec) -> PropValue {
    PropValue::Str(codec.name().into())
}

/// WebRTC Opus is 48 kHz; mono and stereo are both accepted.
pub(crate) const OPUS_SAMPLE_RATE: u32 = 48_000;

pub(crate) fn opus_caps(channels: u8) -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels,
        sample_rate: OPUS_SAMPLE_RATE,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// Everything a sending element accepts on one pad: each video codec at any
/// geometry, then stereo and mono Opus.
pub(crate) fn sink_caps_alternatives() -> Vec<Caps> {
    WebRtcVideoCodec::ALL
        .into_iter()
        .map(WebRtcVideoCodec::caps_any)
        .chain([opus_caps(2), opus_caps(1)])
        .collect()
}

/// Which media a pad carries. WebRTC needs the codec, the m-line `MediaKind`,
/// and the RTP clock to agree, so one pad is one track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Track {
    /// Video on a 90 kHz clock.
    Video(WebRtcVideoCodec),
    /// Opus audio on a 48 kHz clock.
    Audio,
}

impl Track {
    /// The track a pad's fixated caps select, `None` for caps WebRTC cannot carry.
    pub(crate) fn from_caps(caps: &Caps) -> Option<Self> {
        match caps {
            Caps::Audio {
                format: AudioFormat::Opus,
                ..
            } => Some(Track::Audio),
            _ => WebRtcVideoCodec::from_caps(caps).map(Track::Video),
        }
    }

    pub(crate) fn is_video(self) -> bool {
        matches!(self, Track::Video(_))
    }

    pub(crate) fn media_kind(self) -> MediaKind {
        match self {
            Track::Video(_) => MediaKind::Video,
            Track::Audio => MediaKind::Audio,
        }
    }

    pub(crate) fn codec(self) -> Codec {
        match self {
            Track::Video(codec) => codec.str0m(),
            Track::Audio => Codec::Opus,
        }
    }

    /// Turn this track's codec on in an `Rtc` under construction.
    pub(crate) fn enable(self, config: RtcConfig) -> RtcConfig {
        match self {
            Track::Video(codec) => codec.enable(config),
            Track::Audio => config.enable_opus(true),
        }
    }

    pub(crate) fn frequency(self) -> Frequency {
        match self {
            Track::Video(_) => Frequency::NINETY_KHZ,
            Track::Audio => Frequency::FORTY_EIGHT_KHZ,
        }
    }

    fn rate_hz(self) -> u64 {
        match self {
            Track::Video(_) => 90_000,
            Track::Audio => 48_000,
        }
    }

    /// Map a nanosecond PTS to this track's RTP `MediaTime`. `u128` intermediate
    /// avoids overflow on large timestamps.
    pub(crate) fn media_time(self, pts_ns: u64) -> MediaTime {
        let ticks = (pts_ns as u128 * self.rate_hz() as u128 / 1_000_000_000) as u64;
        MediaTime::new(ticks, self.frequency())
    }

    /// A depacketized unit in the pipeline framing (see
    /// `WebRtcVideoCodec::pipeline_unit`), Opus packets as they are.
    pub(crate) fn pipeline_unit(self, depacketized: &[u8]) -> Vec<u8> {
        match self {
            Track::Video(codec) => codec.pipeline_unit(depacketized),
            Track::Audio => depacketized.to_vec(),
        }
    }

    /// Whether a depacketized access unit is a keyframe. Every Opus packet is
    /// independently decodable, so audio is never flagged.
    pub(crate) fn is_keyframe(self, au: &[u8]) -> bool {
        match self {
            Track::Video(codec) => codec.is_keyframe(au),
            Track::Audio => false,
        }
    }
}

/// Enable every distinct codec in `tracks` on an `Rtc` under construction.
pub(crate) fn enable_tracks(
    mut config: RtcConfig,
    tracks: impl IntoIterator<Item = Track>,
) -> RtcConfig {
    let mut seen: Vec<Track> = Vec::new();
    for track in tracks {
        if !seen.contains(&track) {
            config = track.enable(config);
            seen.push(track);
        }
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_codec_round_trips_through_its_name_and_caps() {
        for codec in WebRtcVideoCodec::ALL {
            assert_eq!(WebRtcVideoCodec::parse(codec.name()), Some(codec));
            assert_eq!(WebRtcVideoCodec::from_caps(&codec.caps_any()), Some(codec));
            assert_eq!(
                Track::from_caps(&codec.caps_placeholder()),
                Some(Track::Video(codec))
            );
            assert_eq!(Track::Video(codec).codec(), codec.str0m());
        }
        assert_eq!(WebRtcVideoCodec::parse("mjpeg"), None);
        assert_eq!(Track::from_caps(&opus_caps(2)), Some(Track::Audio));
        assert_eq!(Track::from_caps(&opus_caps(1)), Some(Track::Audio));
    }

    #[test]
    fn mjpeg_is_not_carried() {
        let caps = Caps::CompressedVideo {
            codec: VideoCodec::Mjpeg,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert_eq!(Track::from_caps(&caps), None);
    }

    #[test]
    fn video_codec_property_parses_and_rejects() {
        let mut slot = WebRtcVideoCodec::default();
        set_video_codec_property(&mut slot, &PropValue::Str("AV1".into())).unwrap();
        assert_eq!(slot, WebRtcVideoCodec::Av1);
        assert_eq!(
            set_video_codec_property(&mut slot, &PropValue::Str("theora".into())),
            Err(PropError::Value)
        );
        assert_eq!(
            set_video_codec_property(&mut slot, &PropValue::Int(1)),
            Err(PropError::Type)
        );
        assert_eq!(get_video_codec_property(slot), PropValue::Str("av1".into()));
    }

    #[test]
    fn video_codec_property_accepts_every_name_on_a_launch_line() {
        for codec in WebRtcVideoCodec::ALL {
            assert_eq!(
                VIDEO_CODEC_PROP.parse_value(codec.name()),
                Ok(PropValue::Str(codec.name().into())),
                "{codec:?}"
            );
        }
        assert!(VIDEO_CODEC_PROP.parse_value("theora").is_err());
    }

    #[test]
    fn media_time_uses_the_track_clock() {
        let video = Track::Video(WebRtcVideoCodec::Vp8);
        assert_eq!(video.media_time(1_000_000_000).numer(), 90_000);
        assert_eq!(video.media_time(33_333_333).numer(), 2_999);
        assert_eq!(video.frequency(), Frequency::NINETY_KHZ);
        assert_eq!(Track::Audio.media_time(1_000_000_000).numer(), 48_000);
        assert_eq!(Track::Audio.media_time(20_000_000).numer(), 960);
        assert_eq!(Track::Audio.frequency(), Frequency::FORTY_EIGHT_KHZ);
    }

    #[test]
    fn av1_units_get_their_temporal_delimiter_back() {
        let sequence_header_obu = [0x0a, 0x01, 0x55];
        assert_eq!(
            WebRtcVideoCodec::Av1.pipeline_unit(&sequence_header_obu),
            [0x12, 0x00, 0x0a, 0x01, 0x55]
        );
        let already_delimited = [0x12, 0x00, 0x0a, 0x01, 0x55];
        assert_eq!(
            WebRtcVideoCodec::Av1.pipeline_unit(&already_delimited),
            already_delimited
        );
        assert_eq!(WebRtcVideoCodec::Vp8.pipeline_unit(&[0x10, 0]), [0x10, 0]);
        assert_eq!(Track::Audio.pipeline_unit(&[0xfc]), [0xfc]);
    }

    #[test]
    fn placeholder_caps_fixate() {
        for codec in WebRtcVideoCodec::ALL {
            assert!(codec.caps_placeholder().fixate().is_ok());
        }
    }

    #[test]
    fn keyframe_detection_per_codec() {
        // vp8 frame tag bit 0 clear = key frame, start code, 16x16
        let vp8_key = [0x10, 0x02, 0x00, 0x9d, 0x01, 0x2a, 16, 0, 16, 0, 0, 0, 0, 0];
        assert!(WebRtcVideoCodec::Vp8.is_keyframe(&vp8_key));
        let mut vp8_inter = vp8_key;
        vp8_inter[0] |= 1;
        assert!(!WebRtcVideoCodec::Vp8.is_keyframe(&vp8_inter));
        // h265 IDR_W_RADL (type 19) behind a start code
        assert!(WebRtcVideoCodec::H265.is_keyframe(&[0, 0, 0, 1, 19 << 1, 1, 0xaf]));
        assert!(!WebRtcVideoCodec::H265.is_keyframe(&[0, 0, 0, 1, 1 << 1, 1, 0xaf]));
        assert!(!Track::Audio.is_keyframe(&[0xfc, 0xff, 0xfe]));
    }
}

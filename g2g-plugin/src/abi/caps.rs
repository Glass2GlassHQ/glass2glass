//! Frozen numeric codes for the caps vocabulary, and the conversion between
//! [`Caps`] and [`FfiCaps`].
//!
//! The host's caps enums are `#[non_exhaustive]` and gain variants as new
//! formats land, so their *discriminants* cannot be an ABI: a plugin built a
//! year ago would silently reinterpret a reordered enum. Every variant that
//! crosses v2 therefore gets an explicit code here, **assigned once and never
//! reused**. A code the host does not know fails conversion, and a host `Caps`
//! with no code fails too, rather than being coerced into a neighbouring
//! variant.

use std::vec::Vec;

use g2g_core::caps::{
    AudioFormat, ByteStreamEncoding, Caps, CapsSet, Dim, Interlace, Rate, RawVideoFormat,
    TextFormat, VideoCodec,
};

use super::{
    FfiAudioCaps, FfiByteStreamCaps, FfiCaps, FfiCapsBody, FfiCompressedVideoCaps, FfiDim, FfiRate,
    FfiRawVideoCaps, FfiTextCaps, CAPS_AUDIO, CAPS_BYTE_STREAM, CAPS_COMPRESSED_VIDEO,
    CAPS_RAW_VIDEO, CAPS_TEXT, DIM_ANY, DIM_FIXED, DIM_RANGE, INTERLACE_ANY, INTERLACE_INTERLEAVED,
    INTERLACE_PROGRESSIVE, MAX_CAPS_ALTERNATIVES,
};

/// Why a caps value could not cross the boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapsCodeError {
    /// The plugin used a caps discriminant this host does not know.
    UnknownTag(u32),
    /// The plugin used a format / codec code this host does not know, or the
    /// host holds a variant with no code in this ABI generation.
    UnknownFormat {
        /// Which table was consulted (`"raw video format"`, `"video codec"`, ...).
        table: &'static str,
        /// The offending code, or `u32::MAX` when a host variant had none.
        code: u32,
    },
    /// A `Dim` / `Rate` kind outside `DIM_ANY` / `DIM_FIXED` / `DIM_RANGE`, or
    /// an interlace code outside its three values.
    BadEnumValue {
        /// Which field.
        field: &'static str,
        /// The offending value.
        value: u32,
    },
    /// A channel count above 255 (the host stores it in a `u8`).
    BadChannelCount(u32),
    /// A caps kind the host has but v2 does not carry (tensor, KLV, closed
    /// caption, sub-picture).
    UnsupportedCapsKind,
    /// More alternatives in one set than [`MAX_CAPS_ALTERNATIVES`].
    TooManyAlternatives(usize),
    /// A non-empty alternative list with a null pointer.
    NullAlternatives,
}

impl core::fmt::Display for CapsCodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CapsCodeError::UnknownTag(t) => write!(f, "unknown caps tag {t}"),
            CapsCodeError::UnknownFormat { table, code } => {
                write!(f, "unknown {table} code {code}")
            }
            CapsCodeError::BadEnumValue { field, value } => {
                write!(f, "invalid {field} value {value}")
            }
            CapsCodeError::BadChannelCount(c) => write!(f, "channel count {c} exceeds 255"),
            CapsCodeError::UnsupportedCapsKind => {
                f.write_str("this caps kind does not cross the v2 plugin ABI")
            }
            CapsCodeError::TooManyAlternatives(n) => {
                write!(f, "{n} caps alternatives exceeds the limit")
            }
            CapsCodeError::NullAlternatives => f.write_str("null caps alternative list"),
        }
    }
}

impl std::error::Error for CapsCodeError {}

/// A frozen code assignment: `(code, variant)` pairs, looked up in both
/// directions. One table per enum keeps the two directions from drifting, which
/// is the failure that would make a plugin and host disagree about what a
/// stream is.
macro_rules! code_table {
    ($vis:vis $name:ident : $ty:ty { $($code:literal => $variant:expr),* $(,)? }) => {
        $vis const $name: &[(u32, $ty)] = &[ $( ($code, $variant) ),* ];
    };
}

code_table! {
    pub RAW_VIDEO_FORMAT_CODES: RawVideoFormat {
        1 => RawVideoFormat::Nv12,
        2 => RawVideoFormat::I420,
        3 => RawVideoFormat::Rgba8,
        4 => RawVideoFormat::Bgra8,
        5 => RawVideoFormat::Yuyv,
        6 => RawVideoFormat::I420p10,
        7 => RawVideoFormat::I420p12,
        8 => RawVideoFormat::I422,
        9 => RawVideoFormat::I422p10,
        10 => RawVideoFormat::I422p12,
        11 => RawVideoFormat::I444,
        12 => RawVideoFormat::I444p10,
        13 => RawVideoFormat::I444p12,
        14 => RawVideoFormat::P010,
    }
}

code_table! {
    pub VIDEO_CODEC_CODES: VideoCodec {
        1 => VideoCodec::H264,
        2 => VideoCodec::H265,
        3 => VideoCodec::Av1,
        4 => VideoCodec::Vp8,
        5 => VideoCodec::Vp9,
        6 => VideoCodec::Mjpeg,
        7 => VideoCodec::Mpeg4Part2,
        8 => VideoCodec::Mpeg2,
        9 => VideoCodec::SorensonH263,
        10 => VideoCodec::Vp6 { alpha: false },
        11 => VideoCodec::Vp6 { alpha: true },
        12 => VideoCodec::JpegXs,
        13 => VideoCodec::Pnm,
    }
}

code_table! {
    pub AUDIO_FORMAT_CODES: AudioFormat {
        1 => AudioFormat::Aac,
        2 => AudioFormat::Opus,
        3 => AudioFormat::Mp2,
        4 => AudioFormat::Ac3,
        5 => AudioFormat::Mp3,
        6 => AudioFormat::Speex,
        7 => AudioFormat::Flac,
        8 => AudioFormat::Vorbis,
        9 => AudioFormat::PcmS16Le,
        10 => AudioFormat::PcmF32Le,
        11 => AudioFormat::PcmS24Le,
        12 => AudioFormat::PcmS32Le,
        13 => AudioFormat::PcmU8,
        14 => AudioFormat::Mulaw,
        15 => AudioFormat::Alaw,
        16 => AudioFormat::ImaAdpcm,
    }
}

code_table! {
    pub BYTE_STREAM_ENCODING_CODES: ByteStreamEncoding {
        1 => ByteStreamEncoding::MpegTs,
        2 => ByteStreamEncoding::Matroska,
        3 => ByteStreamEncoding::Ogg,
        4 => ByteStreamEncoding::Flv,
        5 => ByteStreamEncoding::IsoBmff,
        6 => ByteStreamEncoding::Mp4,
        7 => ByteStreamEncoding::Ivf,
        8 => ByteStreamEncoding::MpegPs,
    }
}

code_table! {
    pub TEXT_FORMAT_CODES: TextFormat {
        1 => TextFormat::Utf8,
        2 => TextFormat::PangoMarkup,
        3 => TextFormat::Srt,
        4 => TextFormat::WebVtt,
        5 => TextFormat::Ssa,
        6 => TextFormat::Ttml,
        7 => TextFormat::Teletext,
    }
}

fn decode<T: Copy>(table: &[(u32, T)], name: &'static str, code: u32) -> Result<T, CapsCodeError> {
    table
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, v)| *v)
        .ok_or(CapsCodeError::UnknownFormat { table: name, code })
}

fn encode<T: Copy + PartialEq>(
    table: &[(u32, T)],
    name: &'static str,
    value: T,
) -> Result<u32, CapsCodeError> {
    table
        .iter()
        .find(|(_, v)| *v == value)
        .map(|(c, _)| *c)
        .ok_or(CapsCodeError::UnknownFormat {
            table: name,
            code: u32::MAX,
        })
}

fn dim_into_ffi(dim: &Dim) -> FfiDim {
    match dim {
        Dim::Any => FfiDim {
            kind: DIM_ANY,
            min: 0,
            max: 0,
        },
        Dim::Fixed(v) => FfiDim {
            kind: DIM_FIXED,
            min: *v,
            max: *v,
        },
        Dim::Range { min, max } => FfiDim {
            kind: DIM_RANGE,
            min: *min,
            max: *max,
        },
    }
}

fn dim_from_ffi(dim: &FfiDim) -> Result<Dim, CapsCodeError> {
    match dim.kind {
        DIM_ANY => Ok(Dim::Any),
        DIM_FIXED => Ok(Dim::Fixed(dim.min)),
        DIM_RANGE => Ok(Dim::Range {
            min: dim.min,
            max: dim.max,
        }),
        other => Err(CapsCodeError::BadEnumValue {
            field: "dimension kind",
            value: other,
        }),
    }
}

fn rate_into_ffi(rate: &Rate) -> FfiRate {
    match rate {
        Rate::Any => FfiRate {
            kind: DIM_ANY,
            min_q16: 0,
            max_q16: 0,
        },
        Rate::Fixed(v) => FfiRate {
            kind: DIM_FIXED,
            min_q16: *v,
            max_q16: *v,
        },
        Rate::Range { min_q16, max_q16 } => FfiRate {
            kind: DIM_RANGE,
            min_q16: *min_q16,
            max_q16: *max_q16,
        },
    }
}

fn rate_from_ffi(rate: &FfiRate) -> Result<Rate, CapsCodeError> {
    match rate.kind {
        DIM_ANY => Ok(Rate::Any),
        DIM_FIXED => Ok(Rate::Fixed(rate.min_q16)),
        DIM_RANGE => Ok(Rate::Range {
            min_q16: rate.min_q16,
            max_q16: rate.max_q16,
        }),
        other => Err(CapsCodeError::BadEnumValue {
            field: "framerate kind",
            value: other,
        }),
    }
}

fn interlace_into_ffi(interlace: Interlace) -> u32 {
    match interlace {
        Interlace::Any => INTERLACE_ANY,
        Interlace::Progressive => INTERLACE_PROGRESSIVE,
        Interlace::Interleaved => INTERLACE_INTERLEAVED,
    }
}

fn interlace_from_ffi(value: u32) -> Result<Interlace, CapsCodeError> {
    match value {
        INTERLACE_ANY => Ok(Interlace::Any),
        INTERLACE_PROGRESSIVE => Ok(Interlace::Progressive),
        INTERLACE_INTERLEAVED => Ok(Interlace::Interleaved),
        other => Err(CapsCodeError::BadEnumValue {
            field: "interlace",
            value: other,
        }),
    }
}

/// Convert a host [`Caps`] into its ABI form. A caps kind v2 does not carry
/// (tensor, KLV, closed caption, sub-picture) or a format with no frozen code
/// fails rather than being approximated.
pub fn caps_into_ffi(caps: &Caps) -> Result<FfiCaps, CapsCodeError> {
    let (tag, body) = match caps {
        Caps::RawVideo {
            format,
            width,
            height,
            framerate,
            interlace,
            ..
        } => (
            CAPS_RAW_VIDEO,
            FfiCapsBody {
                raw_video: FfiRawVideoCaps {
                    format: encode(RAW_VIDEO_FORMAT_CODES, "raw video format", *format)?,
                    width: dim_into_ffi(width),
                    height: dim_into_ffi(height),
                    framerate: rate_into_ffi(framerate),
                    interlace: interlace_into_ffi(*interlace),
                },
            },
        ),
        Caps::CompressedVideo {
            codec,
            width,
            height,
            framerate,
            ..
        } => (
            CAPS_COMPRESSED_VIDEO,
            FfiCapsBody {
                compressed_video: FfiCompressedVideoCaps {
                    codec: encode(VIDEO_CODEC_CODES, "video codec", *codec)?,
                    width: dim_into_ffi(width),
                    height: dim_into_ffi(height),
                    framerate: rate_into_ffi(framerate),
                },
            },
        ),
        Caps::Audio {
            format,
            channels,
            sample_rate,
        } => (
            CAPS_AUDIO,
            FfiCapsBody {
                audio: FfiAudioCaps {
                    format: encode(AUDIO_FORMAT_CODES, "audio format", *format)?,
                    channels: u32::from(*channels),
                    sample_rate: *sample_rate,
                },
            },
        ),
        Caps::ByteStream { encoding } => (
            CAPS_BYTE_STREAM,
            FfiCapsBody {
                byte_stream: FfiByteStreamCaps {
                    encoding: encode(
                        BYTE_STREAM_ENCODING_CODES,
                        "byte stream encoding",
                        *encoding,
                    )?,
                },
            },
        ),
        Caps::Text { format } => (
            CAPS_TEXT,
            FfiCapsBody {
                text: FfiTextCaps {
                    format: encode(TEXT_FORMAT_CODES, "text format", *format)?,
                },
            },
        ),
        _ => return Err(CapsCodeError::UnsupportedCapsKind),
    };
    Ok(FfiCaps {
        tag,
        reserved: 0,
        body,
    })
}

/// Convert an ABI caps value back to a host [`Caps`]. Every discriminant,
/// format code, and range kind is checked before the union is read, so a
/// malformed value fails the call instead of selecting the wrong union member.
pub fn caps_from_ffi(caps: &FfiCaps) -> Result<Caps, CapsCodeError> {
    match caps.tag {
        CAPS_RAW_VIDEO => {
            // SAFETY: the tag has just been matched, and the tag is the ABI's
            // sole authority on which union member is live.
            let v = unsafe { caps.body.raw_video };
            Ok(Caps::RawVideo {
                format: decode(RAW_VIDEO_FORMAT_CODES, "raw video format", v.format)?,
                width: dim_from_ffi(&v.width)?,
                height: dim_from_ffi(&v.height)?,
                framerate: rate_from_ffi(&v.framerate)?,
                interlace: interlace_from_ffi(v.interlace)?,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            })
        }
        CAPS_COMPRESSED_VIDEO => {
            // SAFETY: as above, the matched tag selects this member.
            let v = unsafe { caps.body.compressed_video };
            Ok(Caps::CompressedVideo {
                codec: decode(VIDEO_CODEC_CODES, "video codec", v.codec)?,
                width: dim_from_ffi(&v.width)?,
                height: dim_from_ffi(&v.height)?,
                framerate: rate_from_ffi(&v.framerate)?,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            })
        }
        CAPS_AUDIO => {
            // SAFETY: as above, the matched tag selects this member.
            let v = unsafe { caps.body.audio };
            let channels =
                u8::try_from(v.channels).map_err(|_| CapsCodeError::BadChannelCount(v.channels))?;
            Ok(Caps::Audio {
                format: decode(AUDIO_FORMAT_CODES, "audio format", v.format)?,
                channels,
                sample_rate: v.sample_rate,
            })
        }
        CAPS_BYTE_STREAM => {
            // SAFETY: as above, the matched tag selects this member.
            let v = unsafe { caps.body.byte_stream };
            Ok(Caps::ByteStream {
                encoding: decode(
                    BYTE_STREAM_ENCODING_CODES,
                    "byte stream encoding",
                    v.encoding,
                )?,
            })
        }
        CAPS_TEXT => {
            // SAFETY: as above, the matched tag selects this member.
            let v = unsafe { caps.body.text };
            Ok(Caps::Text {
                format: decode(TEXT_FORMAT_CODES, "text format", v.format)?,
            })
        }
        other => Err(CapsCodeError::UnknownTag(other)),
    }
}

/// Read a plugin-declared caps set (a pad template) into a host [`CapsSet`].
/// An empty set is legal and means "any" on a sink pad, "same as the input" on
/// a source pad.
///
/// # Safety
/// When `count > 0`, `alternatives` must point at `count` initialised
/// [`FfiCaps`] values that stay valid for the call.
pub unsafe fn caps_set_from_ffi(
    alternatives: *const FfiCaps,
    count: usize,
) -> Result<CapsSet, CapsCodeError> {
    if count == 0 {
        return Ok(CapsSet::from_alternatives(Vec::new()));
    }
    if count > MAX_CAPS_ALTERNATIVES {
        return Err(CapsCodeError::TooManyAlternatives(count));
    }
    if alternatives.is_null() {
        return Err(CapsCodeError::NullAlternatives);
    }
    // SAFETY: the caller guarantees `count` initialised values at
    // `alternatives`, and `count` is now known to be within the bound.
    let slice = unsafe { core::slice::from_raw_parts(alternatives, count) };
    let mut out = Vec::with_capacity(count);
    for c in slice {
        out.push(caps_from_ffi(c)?);
    }
    Ok(CapsSet::from_alternatives(out))
}

/// Whether a caps value would survive fixation: no `Any` dimension or
/// framerate. A plugin's declared *source* caps become the caps its wrapper
/// returns from `intercept_caps`, and an `Any` there cannot be fixated, so the
/// registration is refused instead of failing later inside the solver.
pub(super) fn is_fixable(caps: &Caps) -> bool {
    caps.fixate().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every code table must be injective in both directions, or two formats
    /// would collide and a stream would be decoded as the wrong thing.
    fn assert_bijective<T: Copy + PartialEq + core::fmt::Debug>(table: &[(u32, T)]) {
        for (i, (code, variant)) in table.iter().enumerate() {
            assert_ne!(*code, 0, "code 0 is reserved for 'none'");
            for (other_code, other_variant) in &table[i + 1..] {
                assert_ne!(code, other_code, "duplicate code {code}");
                assert_ne!(variant, other_variant, "duplicate variant {variant:?}");
            }
        }
    }

    #[test]
    fn code_tables_are_bijective() {
        assert_bijective(RAW_VIDEO_FORMAT_CODES);
        assert_bijective(VIDEO_CODEC_CODES);
        assert_bijective(AUDIO_FORMAT_CODES);
        assert_bijective(BYTE_STREAM_ENCODING_CODES);
        assert_bijective(TEXT_FORMAT_CODES);
    }

    #[test]
    fn raw_video_caps_round_trip() {
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(640),
            height: Dim::Range { min: 2, max: 480 },
            framerate: Rate::Fixed(30 << 16),
            interlace: Interlace::Progressive,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let ffi = caps_into_ffi(&caps).expect("raw video crosses v2");
        assert_eq!(caps_from_ffi(&ffi).expect("and comes back"), caps);
    }

    #[test]
    fn every_coded_variant_round_trips() {
        for (_, format) in RAW_VIDEO_FORMAT_CODES {
            let caps = Caps::RawVideo {
                format: *format,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            };
            let ffi = caps_into_ffi(&caps).expect("coded format crosses");
            assert_eq!(caps_from_ffi(&ffi).expect("and comes back"), caps);
        }
        for (_, codec) in VIDEO_CODEC_CODES {
            let caps = Caps::CompressedVideo {
                codec: *codec,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            };
            let ffi = caps_into_ffi(&caps).expect("coded codec crosses");
            assert_eq!(caps_from_ffi(&ffi).expect("and comes back"), caps);
        }
        for (_, format) in AUDIO_FORMAT_CODES {
            let caps = Caps::Audio {
                format: *format,
                channels: 2,
                sample_rate: 48_000,
            };
            let ffi = caps_into_ffi(&caps).expect("coded audio crosses");
            assert_eq!(caps_from_ffi(&ffi).expect("and comes back"), caps);
        }
    }

    #[test]
    fn unknown_format_code_is_refused() {
        // A hostile or newer plugin naming a format this host has no code for
        // must fail, not land on whatever variant sits at that index.
        let ffi = FfiCaps {
            tag: CAPS_RAW_VIDEO,
            reserved: 0,
            body: FfiCapsBody {
                raw_video: FfiRawVideoCaps {
                    format: 9999,
                    width: FfiDim {
                        kind: DIM_FIXED,
                        min: 16,
                        max: 16,
                    },
                    height: FfiDim {
                        kind: DIM_FIXED,
                        min: 16,
                        max: 16,
                    },
                    framerate: FfiRate {
                        kind: DIM_ANY,
                        min_q16: 0,
                        max_q16: 0,
                    },
                    interlace: INTERLACE_ANY,
                },
            },
        };
        assert!(matches!(
            caps_from_ffi(&ffi),
            Err(CapsCodeError::UnknownFormat { .. })
        ));
    }

    #[test]
    fn unknown_tag_is_refused_without_reading_the_union() {
        let ffi = FfiCaps {
            tag: 4242,
            reserved: 0,
            body: FfiCapsBody {
                text: FfiTextCaps { format: 1 },
            },
        };
        let err = caps_from_ffi(&ffi).expect_err("an unknown tag is refused");
        assert_eq!(err, CapsCodeError::UnknownTag(4242));
    }

    #[test]
    fn bad_dimension_kind_is_refused() {
        let ffi = FfiCaps {
            tag: CAPS_COMPRESSED_VIDEO,
            reserved: 0,
            body: FfiCapsBody {
                compressed_video: FfiCompressedVideoCaps {
                    codec: 1,
                    width: FfiDim {
                        kind: 77,
                        min: 0,
                        max: 0,
                    },
                    height: FfiDim {
                        kind: DIM_ANY,
                        min: 0,
                        max: 0,
                    },
                    framerate: FfiRate {
                        kind: DIM_ANY,
                        min_q16: 0,
                        max_q16: 0,
                    },
                },
            },
        };
        assert!(matches!(
            caps_from_ffi(&ffi),
            Err(CapsCodeError::BadEnumValue { .. })
        ));
    }

    #[test]
    fn channel_count_above_a_byte_is_refused() {
        let ffi = FfiCaps {
            tag: CAPS_AUDIO,
            reserved: 0,
            body: FfiCapsBody {
                audio: FfiAudioCaps {
                    format: 1,
                    channels: 100_000,
                    sample_rate: 48_000,
                },
            },
        };
        let err = caps_from_ffi(&ffi).expect_err("an out-of-range channel count is refused");
        assert_eq!(err, CapsCodeError::BadChannelCount(100_000));
    }

    #[test]
    fn a_caps_kind_outside_v2_is_refused_not_approximated() {
        let err = caps_into_ffi(&Caps::Klv).expect_err("KLV does not cross v2");
        assert_eq!(err, CapsCodeError::UnsupportedCapsKind);
    }

    #[test]
    fn an_over_long_alternative_list_is_refused_before_dereferencing() {
        // The length is checked before the pointer is read, so a hostile count
        // cannot walk off the end of a real (short) array.
        let one = FfiCaps {
            tag: CAPS_TEXT,
            reserved: 0,
            body: FfiCapsBody {
                text: FfiTextCaps { format: 1 },
            },
        };
        // SAFETY: `one` is a live local. The count is deliberately absurd, which
        // is exactly the case that must be caught before the pointer is used.
        let err = unsafe { caps_set_from_ffi(&one as *const _, usize::MAX) }
            .expect_err("an absurd count is refused");
        assert!(matches!(err, CapsCodeError::TooManyAlternatives(_)));
    }

    #[test]
    fn a_null_alternative_list_with_a_count_is_refused() {
        // SAFETY: the null pointer is the case under test; nothing is read.
        let err = unsafe { caps_set_from_ffi(core::ptr::null(), 3) }.expect_err("null is refused");
        assert_eq!(err, CapsCodeError::NullAlternatives);
    }
}

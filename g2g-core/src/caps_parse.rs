//! `gst-launch` caps-string parsing: the inverse of [`Caps::to_gst_string`].
//!
//! Kept next to the printer so the two stay in step, and so a crate that cannot
//! depend on `g2g-plugins` can still turn a caps property string into a
//! [`CapsSet`].

use alloc::vec::Vec;

use crate::caps::{
    pcm_from_gst_format, AudioFormat, ByteStreamEncoding, Caps, CapsSet, Colorimetry, Dim,
    Interlace, Rate, RawVideoFormat, SubPictureFormat, TextFormat, VideoCodec, PCM_FORMATS,
};
use crate::channels::ChannelLayout;

/// The raw pixel formats a format-less `video/x-raw` expands to (M184). Order is
/// the preference the solver fixates by when several survive; in practice the
/// upstream format narrows it to one.
const RAW_VIDEO_FORMATS: [RawVideoFormat; 5] = [
    RawVideoFormat::Nv12,
    RawVideoFormat::I420,
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Yuyv,
];

/// The raw sample formats a format-less `audio/x-raw` expands to (M184).
fn raw_audio_formats() -> Vec<AudioFormat> {
    PCM_FORMATS.iter().map(|(format, _)| *format).collect()
}

/// The plain-text formats a `text/x-raw` with no `format=` covers.
const TEXT_FORMATS: [TextFormat; 2] = [TextFormat::Utf8, TextFormat::PangoMarkup];

/// A parsed caps field value: a fixed scalar (`width=640`), a `[min,max]` range
/// (`width=[1,1920]`), or a `{a,b,...}` list (`format={I420,NV12}`). A range maps
/// to `Dim::Range` / `Rate::Range` within one caps; a list expands to alternatives
/// in the returned `CapsSet` (the gst idiom, so a launch caps filter constrains
/// negotiation without over-fixing).
enum FieldVal<'a> {
    One(&'a str),
    Range(&'a str, &'a str),
    List(Vec<&'a str>),
}

/// Split on top-level commas only, so the commas inside a `[..]` range or `{..}`
/// list are not mistaken for field separators.
pub(crate) fn split_top_commas(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '[' | '{' => depth += 1,
            ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

fn parse_field_val(v: &str) -> FieldVal<'_> {
    let v = v.trim();
    if let Some(inner) = v.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let p = split_top_commas(inner);
        // gst ranges are `[min,max]`; a third `step` element is ignored.
        if p.len() >= 2 {
            return FieldVal::Range(p[0].trim(), p[1].trim());
        }
    }
    if let Some(inner) = v.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
        return FieldVal::List(
            split_top_commas(inner)
                .into_iter()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect(),
        );
    }
    FieldVal::One(v)
}

// Expand a dimension field into its constraint(s): a fixed value or `Any`, a
// `Range`, or (for a list) one `Fixed` per alternative. A present-but-unparseable
// fixed value stays lenient (`Any`), as before; a range / list must parse or the
// whole caps is rejected (`None`).
fn expand_dim(fv: Option<&FieldVal>) -> Option<Vec<Dim>> {
    Some(match fv {
        None => alloc::vec![Dim::Any],
        Some(FieldVal::One(s)) => alloc::vec![s.parse::<u32>().map_or(Dim::Any, Dim::Fixed)],
        Some(FieldVal::Range(a, b)) => {
            alloc::vec![Dim::Range {
                min: a.parse().ok()?,
                max: b.parse().ok()?
            }]
        }
        Some(FieldVal::List(xs)) => xs
            .iter()
            .map(|x| x.parse::<u32>().ok().map(Dim::Fixed))
            .collect::<Option<Vec<_>>>()?,
    })
}

fn expand_rate(fv: Option<&FieldVal>) -> Option<Vec<Rate>> {
    Some(match fv {
        None => alloc::vec![Rate::Any],
        Some(FieldVal::One(s)) => alloc::vec![parse_rate(s).unwrap_or(Rate::Any)],
        Some(FieldVal::Range(a, b)) => {
            alloc::vec![Rate::Range {
                min_q16: rate_q16(a)?,
                max_q16: rate_q16(b)?
            }]
        }
        Some(FieldVal::List(xs)) => xs
            .iter()
            .map(|x| parse_rate(x))
            .collect::<Option<Vec<_>>>()?,
    })
}

fn expand_raw_format(fv: Option<&FieldVal>) -> Option<Vec<RawVideoFormat>> {
    Some(match fv {
        None => RAW_VIDEO_FORMATS.to_vec(),
        Some(FieldVal::One(s)) => alloc::vec![parse_raw_format(s)?],
        Some(FieldVal::List(xs)) => xs
            .iter()
            .map(|x| parse_raw_format(x))
            .collect::<Option<Vec<_>>>()?,
        Some(FieldVal::Range(..)) => return None, // a format range is meaningless
    })
}

fn expand_text_format(fv: Option<&FieldVal>) -> Option<Vec<TextFormat>> {
    Some(match fv {
        None => TEXT_FORMATS.to_vec(),
        Some(FieldVal::One(s)) => alloc::vec![parse_text_format(s)?],
        Some(FieldVal::List(xs)) => xs
            .iter()
            .map(|x| parse_text_format(x))
            .collect::<Option<Vec<_>>>()?,
        Some(FieldVal::Range(..)) => return None,
    })
}

fn expand_audio_format(fv: Option<&FieldVal>) -> Option<Vec<AudioFormat>> {
    Some(match fv {
        None => raw_audio_formats(),
        Some(FieldVal::One(s)) => alloc::vec![parse_audio_format(s)?],
        Some(FieldVal::List(xs)) => xs
            .iter()
            .map(|x| parse_audio_format(x))
            .collect::<Option<Vec<_>>>()?,
        Some(FieldVal::Range(..)) => return None,
    })
}

// An absent `colorimetry` is the wildcard (matching `to_gst_string`, which
// omits an unknown one); a present value must parse (preset name or 4-part
// form) or the whole caps is rejected. No list/range: GStreamer's colorimetry
// is one flat string value.
fn expand_colorimetry(fv: Option<&FieldVal>) -> Option<Colorimetry> {
    match fv {
        None => Some(Colorimetry::UNKNOWN),
        Some(FieldVal::One(s)) => Colorimetry::from_gst_string(s),
        Some(_) => None,
    }
}

// An absent `channel-mask` is the wildcard (matching `to_gst_string`, which
// omits an unspecified layout). A present one is gst's `(bitmask)0x...` in gst
// bit order; a mask naming a speaker g2g has no position for rejects the whole
// caps rather than dropping that channel silently. No list/range: gst's
// channel-mask is one flat bitmask value.
fn expand_channel_mask(fv: Option<&FieldVal>) -> Option<ChannelLayout> {
    let Some(FieldVal::One(s)) = fv else {
        return match fv {
            None => Some(ChannelLayout::UNSPECIFIED),
            Some(_) => None,
        };
    };
    let v = s.trim();
    let v = v.strip_prefix("(bitmask)").unwrap_or(v).trim();
    let mask = match v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16).ok()?,
        None => v.parse::<u64>().ok()?,
    };
    ChannelLayout::from_gst_mask(mask)
}

// `Caps::Audio` holds scalar channels (u8) / sample_rate (u32) with no range
// type, so a range is rejected; a list expands to alternatives.
fn expand_u8(fv: Option<&FieldVal>, default: u8) -> Option<Vec<u8>> {
    Some(match fv {
        None => alloc::vec![default],
        Some(FieldVal::One(s)) => alloc::vec![s.parse().unwrap_or(default)],
        Some(FieldVal::List(xs)) => xs
            .iter()
            .map(|x| x.parse::<u8>().ok())
            .collect::<Option<Vec<_>>>()?,
        Some(FieldVal::Range(..)) => return None,
    })
}

fn expand_u32(fv: Option<&FieldVal>, default: u32) -> Option<Vec<u32>> {
    Some(match fv {
        None => alloc::vec![default],
        Some(FieldVal::One(s)) => alloc::vec![s.parse().unwrap_or(default)],
        Some(FieldVal::List(xs)) => xs
            .iter()
            .map(|x| x.parse::<u32>().ok())
            .collect::<Option<Vec<_>>>()?,
        Some(FieldVal::Range(..)) => return None,
    })
}

impl CapsSet {
    /// Parse a `gst-launch` caps description (`media/type,field=value,...`) into a
    /// [`CapsSet`], the inverse of [`Caps::to_gst_string`]. Field values may be
    /// fixed (`width=640`), a `[min,max]` range (`width=[1,1920]`, mapped to
    /// `Dim::Range` / `Rate::Range`), or a `{a,b,...}` list (`format={I420,NV12}`,
    /// expanded to alternatives). A `video/x-raw` / `audio/x-raw` with no `format`
    /// expands to all supported raw formats at the given geometry (the
    /// gst-idiomatic format-less caps). `None` on an unknown media type or an
    /// unparseable range / list. Format values are case-insensitive (GStreamer's
    /// uppercase or the historical lowercase, M182).
    pub fn from_gst_string(desc: &str) -> Option<CapsSet> {
        let mut parts = split_top_commas(desc).into_iter();
        let media = parts.next()?.trim();
        let fields: Vec<(&str, FieldVal)> = parts
            .filter_map(|p| p.split_once('='))
            .map(|(k, v)| (k.trim(), parse_field_val(v)))
            .collect();
        let fv = |key: &str| fields.iter().find(|(k, _)| *k == key).map(|(_, v)| v);

        // Cartesian product of the list-valued fields; range fields stay as one
        // `Range` inside each alternative.
        let compressed_set = |codec: VideoCodec| -> Option<CapsSet> {
            let (widths, heights, rates) = (
                expand_dim(fv("width"))?,
                expand_dim(fv("height"))?,
                expand_rate(fv("framerate"))?,
            );
            let colorimetry = expand_colorimetry(fv("colorimetry"))?;
            let mut alts = Vec::new();
            for w in &widths {
                for h in &heights {
                    for r in &rates {
                        alts.push(compressed(
                            codec,
                            w.clone(),
                            h.clone(),
                            r.clone(),
                            colorimetry,
                        ));
                    }
                }
            }
            Some(CapsSet::from_alternatives(alts))
        };
        let bytestream = |encoding: ByteStreamEncoding| -> Option<CapsSet> {
            Some(CapsSet::one(Caps::ByteStream { encoding }))
        };
        let audio_set = |formats: &[AudioFormat]| -> Option<CapsSet> {
            let (channels, rates) = (
                expand_u8(fv("channels"), 2)?,
                expand_u32(fv("rate"), 48_000)?,
            );
            let channel_layout = expand_channel_mask(fv("channel-mask"))?;
            let mut alts = Vec::new();
            for &format in formats {
                for &ch in &channels {
                    for &sr in &rates {
                        alts.push(Caps::Audio {
                            format,
                            channels: ch,
                            sample_rate: sr,
                            channel_layout,
                        });
                    }
                }
            }
            Some(CapsSet::from_alternatives(alts))
        };

        let text_set = |format| Some(CapsSet::one(Caps::Text { format }));

        match media {
            "video/x-raw" => {
                let (formats, widths, heights, rates) = (
                    expand_raw_format(fv("format"))?,
                    expand_dim(fv("width"))?,
                    expand_dim(fv("height"))?,
                    expand_rate(fv("framerate"))?,
                );
                // An absent `interlace-mode` is the wildcard (a filter should not
                // constrain what it does not name), matching `to_gst_string`, which
                // prints the field only for `interleaved`.
                let interlace = match fv("interlace-mode") {
                    None => Interlace::Any,
                    Some(FieldVal::One(s)) if *s == "progressive" => Interlace::Progressive,
                    Some(FieldVal::One(s)) if *s == "interleaved" => Interlace::Interleaved,
                    Some(_) => return None,
                };
                let colorimetry = expand_colorimetry(fv("colorimetry"))?;
                let mut alts = Vec::new();
                for &format in &formats {
                    for w in &widths {
                        for h in &heights {
                            for r in &rates {
                                alts.push(Caps::RawVideo {
                                    format,
                                    width: w.clone(),
                                    height: h.clone(),
                                    framerate: r.clone(),
                                    interlace,
                                    colorimetry,
                                });
                            }
                        }
                    }
                }
                Some(CapsSet::from_alternatives(alts))
            }
            "audio/x-raw" => audio_set(&expand_audio_format(fv("format"))?),
            "video/x-h264" => compressed_set(VideoCodec::H264),
            "video/x-h265" => compressed_set(VideoCodec::H265),
            "video/x-vp8" => compressed_set(VideoCodec::Vp8),
            "video/x-vp9" => compressed_set(VideoCodec::Vp9),
            "video/x-av1" => compressed_set(VideoCodec::Av1),
            "image/jpeg" => compressed_set(VideoCodec::Mjpeg),
            "image/png" => compressed_set(VideoCodec::Png),
            "image/webp" => compressed_set(VideoCodec::WebP),
            "image/x-portable-anymap"
            | "image/x-portable-bitmap"
            | "image/x-portable-graymap"
            | "image/x-portable-pixmap" => compressed_set(VideoCodec::Pnm),
            // gst tells VC-1 from the older WMV versions with `wmvversion` /
            // `format`, fields this caps string carries no room for.
            "video/x-wmv" => compressed_set(VideoCodec::Vc1),
            // The legacy Flash codecs, under the names gst's flvdemux emits.
            "video/x-flash-video" => compressed_set(VideoCodec::SorensonH263),
            "video/x-vp6-flash" => compressed_set(VideoCodec::Vp6 { alpha: false }),
            "video/x-vp6-alpha" => compressed_set(VideoCodec::Vp6 { alpha: true }),
            "audio/x-speex" => audio_set(&[AudioFormat::Speex]),
            "audio/x-opus" => audio_set(&[AudioFormat::Opus]),
            "audio/x-ac3" => audio_set(&[AudioFormat::Ac3]),
            "audio/x-flac" => audio_set(&[AudioFormat::Flac]),
            // The companded / ADPCM telephony formats, under the names the
            // printer emits, so a caps string round-trips.
            "audio/x-mulaw" => audio_set(&[AudioFormat::Mulaw]),
            "audio/x-alaw" => audio_set(&[AudioFormat::Alaw]),
            "audio/x-adpcm" => audio_set(&[AudioFormat::ImaAdpcm]),
            "audio/x-vorbis" => audio_set(&[AudioFormat::Vorbis]),
            // gst names AAC `audio/mpeg` (with mpegversion=4, which we don't require).
            "audio/mpeg" => audio_set(&[AudioFormat::Aac]),
            // The text media types `Caps::to_gst_string` prints, so a caps a g2g
            // element announced parses back into the same set.
            "text/x-raw" => Some(CapsSet::from_alternatives(
                expand_text_format(fv("format"))?
                    .into_iter()
                    .map(|format| Caps::Text { format })
                    .collect(),
            )),
            "application/x-subtitle" => text_set(TextFormat::Srt),
            "application/x-subtitle-vtt" => text_set(TextFormat::WebVtt),
            "application/x-ssa" => text_set(TextFormat::Ssa),
            "application/ttml+xml" => text_set(TextFormat::Ttml),
            "private/teletext" => text_set(TextFormat::Teletext),
            "meta/x-klv" => Some(CapsSet::one(Caps::Klv)),
            // The container media types, the inverse of what `to_gst_string`
            // prints for a `ByteStream`, so a caps string round-trips and an
            // encoding profile can name its container. `video/quicktime` parses
            // as the whole-file `Mp4` form (what a muxer writes and a file
            // carries); the streaming `IsoBmff` form is named by the CMAF
            // spelling, since the two share gst's media type.
            "video/mpegts" => bytestream(ByteStreamEncoding::MpegTs),
            "video/x-matroska" | "video/webm" => bytestream(ByteStreamEncoding::Matroska),
            "application/ogg" => bytestream(ByteStreamEncoding::Ogg),
            "video/x-flv" => bytestream(ByteStreamEncoding::Flv),
            "video/quicktime" => bytestream(ByteStreamEncoding::Mp4),
            "video/x-cmaf" => bytestream(ByteStreamEncoding::IsoBmff),
            "video/x-ivf" => bytestream(ByteStreamEncoding::Ivf),
            "video/mpeg-ps" => bytestream(ByteStreamEncoding::MpegPs),
            "audio/x-wav" => bytestream(ByteStreamEncoding::Wav),
            "audio/x-aiff" => bytestream(ByteStreamEncoding::Aiff),
            "audio/x-au" => bytestream(ByteStreamEncoding::Au),
            "video/x-msvideo" => bytestream(ByteStreamEncoding::Avi),
            "application/x-yuv4mpeg" => bytestream(ByteStreamEncoding::Y4m),
            "multipart/x-mixed-replace" => bytestream(ByteStreamEncoding::Multipart),
            "application/octet-stream" => bytestream(ByteStreamEncoding::Raw),
            "application/x-rtp" => bytestream(ByteStreamEncoding::Rtp),
            "application/x-srtp" => bytestream(ByteStreamEncoding::Srtp),
            "application/x-rtcp" => bytestream(ByteStreamEncoding::Rtcp),
            "application/x-srtcp" => bytestream(ByteStreamEncoding::Srtcp),
            "application/x-dtls" => bytestream(ByteStreamEncoding::Dtls),
            "subpicture/x-dvd" => Some(CapsSet::one(Caps::SubPicture {
                format: SubPictureFormat::VobSub,
            })),
            "subpicture/x-dvb" => Some(CapsSet::one(Caps::SubPicture {
                format: SubPictureFormat::DvbSub,
            })),
            "subpicture/x-pgs" => Some(CapsSet::one(Caps::SubPicture {
                format: SubPictureFormat::Pgs,
            })),
            _ => None,
        }
    }
}

fn compressed(
    codec: VideoCodec,
    width: Dim,
    height: Dim,
    framerate: Rate,
    colorimetry: Colorimetry,
) -> Caps {
    Caps::CompressedVideo {
        codec,
        width,
        height,
        framerate,
        colorimetry,
    }
}

// GStreamer caps name formats uppercase (NV12, RGBA, YUY2, S16LE); accept any
// case and the historical lowercase spellings so both port.
pub fn parse_raw_format(s: &str) -> Option<RawVideoFormat> {
    Some(match s.to_ascii_lowercase().as_str() {
        "rgba" => RawVideoFormat::Rgba8,
        "rgb" => RawVideoFormat::Rgb8,
        "bgra" => RawVideoFormat::Bgra8,
        "nv12" => RawVideoFormat::Nv12,
        "i420" => RawVideoFormat::I420,
        "yuyv" | "yuy2" => RawVideoFormat::Yuyv,
        "i420_10le" => RawVideoFormat::I420p10,
        "i420_12le" => RawVideoFormat::I420p12,
        "y42b" => RawVideoFormat::I422,
        "i422_10le" => RawVideoFormat::I422p10,
        "i422_12le" => RawVideoFormat::I422p12,
        "y444" => RawVideoFormat::I444,
        "y444_10le" => RawVideoFormat::I444p10,
        "y444_12le" => RawVideoFormat::I444p12,
        "p010_10le" | "p010" => RawVideoFormat::P010,
        _ => return None,
    })
}

/// The `format=` of a `text/x-raw`, as gst spells it.
fn parse_text_format(s: &str) -> Option<TextFormat> {
    Some(match s.to_ascii_lowercase().as_str() {
        "utf8" => TextFormat::Utf8,
        "pango-markup" => TextFormat::PangoMarkup,
        _ => return None,
    })
}

fn parse_audio_format(s: &str) -> Option<AudioFormat> {
    pcm_from_gst_format(s.trim())
}

/// Parse a framerate `num/den` (or bare integer) into a Q16 fixed-point value.
/// Shared by [`parse_rate`] and the `[min,max]` framerate-range expansion.
fn rate_q16(s: &str) -> Option<u32> {
    Some(match s.trim().split_once('/') {
        Some((n, d)) => {
            let n: u64 = n.trim().parse().ok()?;
            let d: u64 = d.trim().parse().ok()?;
            if d == 0 {
                return None;
            }
            ((n << 16) / d) as u32
        }
        None => (s.trim().parse::<u64>().ok()? << 16) as u32,
    })
}

/// Parse a framerate `num/den` (or bare integer) into a Q16 [`Rate::Fixed`].
fn parse_rate(s: &str) -> Option<Rate> {
    rate_q16(s).map(Rate::Fixed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pcm_format_parses_back_from_what_it_prints() {
        // A format the printer knows and the parser does not makes any caps
        // description carrying it unreadable, far from the format list.
        for (format, _) in PCM_FORMATS {
            let caps = Caps::Audio {
                format,
                channels: 1,
                sample_rate: 16_000,
                channel_layout: crate::ChannelLayout::UNSPECIFIED,
            };
            let printed = caps.to_gst_string();
            let parsed = CapsSet::from_gst_string(&printed)
                .unwrap_or_else(|| panic!("{printed} does not parse back"));
            assert_eq!(parsed.alternatives(), [caps].as_slice(), "{printed}");
        }
    }

    #[test]
    fn parses_high_bit_depth_and_alt_chroma_format_names() {
        // The GStreamer `format=` strings for the planar high-bit-depth / 4:2:2 /
        // 4:4:4 family resolve to the right variant (case-insensitively).
        for (s, want) in [
            ("I420_10LE", RawVideoFormat::I420p10),
            ("i420_12le", RawVideoFormat::I420p12),
            ("Y42B", RawVideoFormat::I422),
            ("I422_10LE", RawVideoFormat::I422p10),
            ("I422_12LE", RawVideoFormat::I422p12),
            ("Y444", RawVideoFormat::I444),
            ("Y444_10LE", RawVideoFormat::I444p10),
            ("Y444_12LE", RawVideoFormat::I444p12),
        ] {
            assert_eq!(parse_raw_format(s), Some(want), "format string {s}");
        }
    }

    #[test]
    fn parse_caps_set_expands_format_less_raw_video() {
        // No `format` -> all supported pixel formats at the fixed geometry (M184).
        let set = CapsSet::from_gst_string("video/x-raw,width=160,height=120").expect("parses");
        assert_eq!(set.alternatives().len(), RAW_VIDEO_FORMATS.len());
        assert!(set.alternatives().iter().all(|c| matches!(
            c,
            Caps::RawVideo {
                width: Dim::Fixed(160),
                height: Dim::Fixed(120),
                ..
            }
        )));
        // Every supported format is represented at that geometry.
        for fmt in RAW_VIDEO_FORMATS {
            assert!(set.alternatives().iter().any(|c| matches!(
                c,
                Caps::RawVideo { format, .. } if *format == fmt
            )));
        }
        // A pinned format still yields exactly one alternative.
        assert_eq!(
            CapsSet::from_gst_string("video/x-raw,format=NV12")
                .unwrap()
                .alternatives()
                .len(),
            1
        );
        // Format-less audio expands to the raw sample formats.
        assert_eq!(
            CapsSet::from_gst_string("audio/x-raw,channels=2")
                .unwrap()
                .alternatives()
                .len(),
            raw_audio_formats().len()
        );
    }

    #[test]
    fn parse_caps_range_maps_to_dim_and_rate_range() {
        // `[min,max]` on width/height -> Dim::Range in one caps (not an expansion).
        let set =
            CapsSet::from_gst_string("video/x-raw,format=nv12,width=[1,1920],height=[1,1080]")
                .unwrap();
        assert_eq!(set.alternatives().len(), 1);
        let Caps::RawVideo { width, height, .. } = &set.alternatives()[0] else {
            panic!()
        };
        assert_eq!(*width, Dim::Range { min: 1, max: 1920 });
        assert_eq!(*height, Dim::Range { min: 1, max: 1080 });
        // A framerate range maps to Rate::Range.
        let set = CapsSet::from_gst_string("video/x-h264,framerate=[0/1,60/1]").unwrap();
        let Caps::CompressedVideo { framerate, .. } = &set.alternatives()[0] else {
            panic!()
        };
        assert!(matches!(framerate, Rate::Range { .. }), "got {framerate:?}");
    }

    /// A text stage in a launch line writes `text/x-raw,format=utf8`, which is
    /// exactly what `Caps::to_gst_string` prints for a UTF-8 text link, so the
    /// two have to agree.
    #[test]
    fn parse_caps_reads_the_text_media_types() {
        let set = CapsSet::from_gst_string("text/x-raw,format=utf8").unwrap();
        assert_eq!(
            set.alternatives(),
            &[Caps::Text {
                format: TextFormat::Utf8
            }]
        );
        // No format is the wildcard over the plain-text ones.
        assert_eq!(
            CapsSet::from_gst_string("text/x-raw")
                .unwrap()
                .alternatives()
                .len(),
            2
        );
        // The structured subtitle types carry their own media type.
        assert_eq!(
            CapsSet::from_gst_string("application/x-subtitle-vtt")
                .unwrap()
                .alternatives(),
            &[Caps::Text {
                format: TextFormat::WebVtt
            }]
        );
        assert!(CapsSet::from_gst_string("text/x-raw,format=nonesuch").is_none());
    }

    #[test]
    fn packet_media_types_round_trip() {
        for (media_type, encoding) in [
            ("application/x-rtp", ByteStreamEncoding::Rtp),
            ("application/x-srtp", ByteStreamEncoding::Srtp),
            ("application/x-rtcp", ByteStreamEncoding::Rtcp),
            ("application/x-srtcp", ByteStreamEncoding::Srtcp),
            ("application/x-dtls", ByteStreamEncoding::Dtls),
            ("audio/x-aiff", ByteStreamEncoding::Aiff),
            ("audio/x-au", ByteStreamEncoding::Au),
        ] {
            let caps = Caps::ByteStream { encoding };
            assert_eq!(caps.to_gst_string(), media_type);
            assert_eq!(
                CapsSet::from_gst_string(media_type).unwrap().alternatives(),
                &[caps]
            );
        }
    }

    #[test]
    fn parse_caps_reads_colorimetry() {
        let set = CapsSet::from_gst_string("video/x-raw,format=NV12,colorimetry=bt709").unwrap();
        let Caps::RawVideo { colorimetry, .. } = &set.alternatives()[0] else {
            panic!()
        };
        assert_eq!(*colorimetry, Colorimetry::BT709);
        // The compressed media types take the field too (a caps filter can pin
        // the bitstream's declared colorimetry).
        let set = CapsSet::from_gst_string("video/x-h264,colorimetry=bt601").unwrap();
        let Caps::CompressedVideo { colorimetry, .. } = &set.alternatives()[0] else {
            panic!()
        };
        assert_eq!(*colorimetry, Colorimetry::BT601);
        // Absent = wildcard; unparseable = the whole caps is rejected.
        let set = CapsSet::from_gst_string("video/x-raw,format=NV12").unwrap();
        let Caps::RawVideo { colorimetry, .. } = &set.alternatives()[0] else {
            panic!()
        };
        assert_eq!(*colorimetry, Colorimetry::UNKNOWN);
        assert!(CapsSet::from_gst_string("video/x-raw,colorimetry=banana").is_none());
        // What the printer emits parses back to the same caps.
        let caps = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
            interlace: Interlace::Any,
            colorimetry: Colorimetry::BT709,
        };
        let printed = caps.to_gst_string();
        assert_eq!(
            CapsSet::from_gst_string(&printed).unwrap().alternatives(),
            &[caps],
            "{printed}"
        );
    }

    #[test]
    fn parse_caps_list_expands_to_alternatives() {
        // `format={I420,NV12}` -> two alternatives, geometry fixed on both.
        let set = CapsSet::from_gst_string("video/x-raw,format={I420,NV12},width=640,height=480")
            .unwrap();
        let fmts: Vec<RawVideoFormat> = set
            .alternatives()
            .iter()
            .map(|c| match c {
                Caps::RawVideo { format, width, .. } => {
                    assert_eq!(*width, Dim::Fixed(640));
                    *format
                }
                _ => panic!("raw video"),
            })
            .collect();
        assert_eq!(fmts.len(), 2);
        assert!(fmts.contains(&RawVideoFormat::I420) && fmts.contains(&RawVideoFormat::Nv12));
        // A width list expands too (cartesian with format): {640,1280} x one format.
        let set = CapsSet::from_gst_string("video/x-raw,format=nv12,width={640,1280}").unwrap();
        assert_eq!(set.alternatives().len(), 2);
        // A malformed range fails the whole caps (rejected, not silently dropped).
        assert!(CapsSet::from_gst_string("video/x-raw,width=[a,b]").is_none());
    }
}

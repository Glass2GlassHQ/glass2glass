//! RFC 4566 SDP: the shared media-section scanner plus the RTP/AVP mapping from
//! a media description to [`Caps`] (M833).
//!
//! An RTP receiver has no in-band stream description, so the SDP a sender
//! publishes (`ffmpeg -sdp_file`, an RTSP `DESCRIBE` body, a file next to the
//! stream) is what tells it the payload type, codec, clock rate, and, via the
//! H.264 / H.265 `fmtp` parameter sets, the geometry. [`SdpMedia::parse`] turns
//! one such description into absolute caps a source can produce.
//!
//! The line scanner ([`scan_sections`]) is the one SDP parser in the workspace:
//! [`crate::st2110sdp`] maps the same sections onto SMPTE ST 2110 essences.
//! Sans-IO (`no_std` + alloc): the caller supplies the text, whether it came
//! from a file, an HTTP body, or an RTSP response.
//!
//! Never trust the stream: an unmappable encoding, a malformed base64 parameter
//! set, or a missing field yields `None` / no geometry rather than a guess.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::{AudioFormat, Caps, Dim, Rate, VideoCodec};

use crate::nalparse::NalCodec;

/// One media section's raw SDP fields, before any codec interpretation: the
/// `m=` line plus the attributes under it, with the session-level connection
/// address and reference clock inherited when the section carries none.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SdpSection {
    /// The `m=` media kind (`video`, `audio`, `application`, ...).
    pub kind: String,
    pub port: u16,
    pub payload_type: u8,
    /// `c=IN IP4 <addr>` (the TTL suffix stripped), section or session level.
    pub address: Option<String>,
    /// `a=rtpmap:` past the payload type, e.g. `H264/90000` or `opus/48000/2`.
    pub rtpmap: Option<String>,
    /// `a=fmtp:` past the payload type, e.g. `packetization-mode=1; sprop-...`.
    pub fmtp: Option<String>,
    /// `a=ptime:` in microseconds.
    pub ptime_us: Option<u32>,
    /// `a=framerate:` as written (`15`, `29.97`).
    pub framerate: Option<String>,
    /// `a=ts-refclk:ptp=...` as `(grandmaster identity, domain)`.
    pub ptp: Option<(String, u8)>,
}

/// Scan every media section of an SDP document. Sections whose `m=` line is
/// malformed are skipped; the session-level `c=` and `a=ts-refclk` lines (those
/// before the first `m=`) are inherited by sections that carry none.
pub fn scan_sections(text: &str) -> Vec<SdpSection> {
    let lines: Vec<&str> = text.lines().map(|l| l.trim_end_matches('\r')).collect();
    let starts: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("m="))
        .map(|(i, _)| i)
        .collect();
    let Some(&first) = starts.first() else {
        return Vec::new();
    };

    let mut session_address = None;
    let mut session_ptp = None;
    for l in &lines[..first] {
        if let Some(c) = l.strip_prefix("c=IN IP4 ") {
            session_address = c.split('/').next().map(|a| a.to_string());
        } else if let Some(a) = l.strip_prefix("a=ts-refclk:ptp=") {
            session_ptp = parse_refclk(a);
        }
    }

    let mut out = Vec::new();
    for (k, &start) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(lines.len());
        let Some(mut section) = scan_one(&lines[start..end]) else {
            continue;
        };
        if section.address.is_none() {
            section.address = session_address.clone();
        }
        if section.ptp.is_none() {
            section.ptp = session_ptp.clone();
        }
        out.push(section);
    }
    out
}

/// Scan the lines of a single media section (`lines[0]` is its `m=` line).
fn scan_one(lines: &[&str]) -> Option<SdpSection> {
    // "m=<kind> <port> <proto> <pt>"; a port of 0 means "not carried here".
    let mut m = lines.first()?.strip_prefix("m=")?.split_whitespace();
    let kind = m.next()?.to_string();
    let port = m.next()?.parse::<u16>().ok()?;
    let _proto = m.next()?;
    let payload_type = m.next()?.parse::<u8>().ok()?;

    let mut section = SdpSection {
        kind,
        port,
        payload_type,
        ..Default::default()
    };
    for line in &lines[1..] {
        if let Some(c) = line.strip_prefix("c=IN IP4 ") {
            section.address = c.split('/').next().map(|a| a.to_string());
        } else if let Some(a) = line.strip_prefix("a=rtpmap:") {
            section.rtpmap = after_pt(a).map(|s| s.to_string());
        } else if let Some(a) = line.strip_prefix("a=fmtp:") {
            section.fmtp = after_pt(a).map(|s| s.to_string());
        } else if let Some(a) = line.strip_prefix("a=ptime:") {
            section.ptime_us = parse_ptime_us(a);
        } else if let Some(a) = line.strip_prefix("a=framerate:") {
            section.framerate = Some(a.trim().to_string());
        } else if let Some(a) = line.strip_prefix("a=ts-refclk:ptp=") {
            section.ptp = parse_refclk(a);
        }
    }
    Some(section)
}

/// Drop the leading `<pt> ` from an `a=rtpmap:` / `a=fmtp:` value.
fn after_pt(s: &str) -> Option<&str> {
    s.split_once(' ').map(|(_, rest)| rest)
}

/// Parse `IEEE1588-2008:<gmid>:<domain>` into `(gmid, domain)`.
fn parse_refclk(a: &str) -> Option<(String, u8)> {
    let rest = a.split_once(':')?.1; // drop the profile
    let (gmid, domain) = rest.rsplit_once(':')?;
    Some((gmid.to_string(), domain.parse::<u8>().ok()?))
}

/// Parse an `a=ptime:` value in (fractional) milliseconds to microseconds.
fn parse_ptime_us(s: &str) -> Option<u32> {
    let s = s.trim();
    match s.split_once('.') {
        Some((ms, frac)) => {
            // Take up to 3 fractional digits (millisecond -> microsecond).
            let mut micros = ms.parse::<u32>().ok()?.checked_mul(1000)?;
            let mut scale = 100u32;
            for c in frac.chars().take(3) {
                micros = micros.checked_add(c.to_digit(10)? * scale)?;
                scale /= 10;
            }
            Some(micros)
        }
        None => s.parse::<u32>().ok()?.checked_mul(1000),
    }
}

/// One RTP/AVP media description resolved to the caps a receiver of it produces.
#[derive(Clone, Debug, PartialEq)]
pub struct SdpMedia {
    /// What a receiver of this stream produces. Video geometry is filled in from
    /// the `fmtp` parameter sets when they are present, and left `Dim::Any` /
    /// `Rate::Any` otherwise (the in-band SPS is authoritative either way).
    pub caps: Caps,
    pub payload_type: u8,
    /// RTP media clock from the `rtpmap` (90000 for video, the sample rate for
    /// audio).
    pub clock_rate: u32,
    /// The `c=` connection address, or `0.0.0.0` when the SDP carries none.
    pub address: String,
    /// The `m=` port. 0 means the SDP does not pin one.
    pub port: u16,
    /// Annex-B parameter sets decoded from `sprop-parameter-sets` (H.264) or
    /// `sprop-vps` / `sprop-sps` / `sprop-pps` (H.265). Empty when the SDP
    /// carries none.
    pub parameter_sets: Vec<u8>,
}

impl SdpMedia {
    /// Map the first media section this understands. `None` when the document
    /// has no section with a mappable `rtpmap` encoding.
    pub fn parse(text: &str) -> Option<Self> {
        Self::parse_all(text).into_iter().next()
    }

    /// Map every media section whose encoding is known, in document order.
    /// Unmappable sections are skipped rather than failing the whole document,
    /// so a program carrying one unsupported codec still configures the rest.
    pub fn parse_all(text: &str) -> Vec<Self> {
        scan_sections(text)
            .into_iter()
            .filter_map(|s| Self::from_section(&s))
            .collect()
    }

    fn from_section(s: &SdpSection) -> Option<Self> {
        let rtpmap = s.rtpmap.as_deref()?;
        let mut parts = rtpmap.split('/');
        let encoding = parts.next()?;
        let clock_rate = parts.next()?.parse::<u32>().ok()?;
        let channels = parts.next().and_then(|c| c.parse::<u8>().ok()).unwrap_or(1);
        if clock_rate == 0 {
            return None;
        }

        let mut parameter_sets = Vec::new();
        let caps = match encoding_kind(encoding)? {
            Encoding::Video(codec) => {
                if let Some(fmtp) = s.fmtp.as_deref() {
                    parameter_sets = sprop_parameter_sets(codec, fmtp);
                }
                let (width, height, sps_fps) = match sps_geometry(codec, &parameter_sets) {
                    Some(g) => (Dim::Fixed(g.width), Dim::Fixed(g.height), g.framerate),
                    None => (Dim::Any, Dim::Any, None),
                };
                Caps::CompressedVideo {
                    codec,
                    width,
                    height,
                    // The SDP's own a=framerate wins over the SPS VUI: it is what
                    // the sender declares it is actually sending.
                    framerate: s
                        .framerate
                        .as_deref()
                        .and_then(parse_q16_fps)
                        .or(sps_fps)
                        .map_or(Rate::Any, Rate::Fixed),
                    colorimetry: g2g_core::Colorimetry::UNKNOWN,
                }
            }
            Encoding::Audio(format) => Caps::Audio {
                format,
                channels,
                sample_rate: clock_rate,
            },
        };
        Some(SdpMedia {
            caps,
            payload_type: s.payload_type,
            clock_rate,
            address: s.address.clone().unwrap_or_else(|| "0.0.0.0".to_string()),
            port: s.port,
            parameter_sets,
        })
    }
}

/// What an `rtpmap` encoding name carries.
enum Encoding {
    Video(VideoCodec),
    Audio(AudioFormat),
}

/// Map an RTP `rtpmap` encoding name (case-insensitive, RFC 4566) to the codec
/// it names. `None` for an encoding g2g has no caps kind for.
fn encoding_kind(name: &str) -> Option<Encoding> {
    // Compare uppercased without allocating: the names are short and ASCII.
    let eq = |want: &str| name.eq_ignore_ascii_case(want);
    Some(match () {
        _ if eq("H264") => Encoding::Video(VideoCodec::H264),
        _ if eq("H265") || eq("HEVC") => Encoding::Video(VideoCodec::H265),
        _ if eq("VP8") => Encoding::Video(VideoCodec::Vp8),
        _ if eq("VP9") => Encoding::Video(VideoCodec::Vp9),
        _ if eq("AV1") || eq("AV1X") => Encoding::Video(VideoCodec::Av1),
        _ if eq("opus") => Encoding::Audio(AudioFormat::Opus),
        _ if eq("mpeg4-generic") || eq("MP4A-LATM") => Encoding::Audio(AudioFormat::Aac),
        _ if eq("PCMU") => Encoding::Audio(AudioFormat::Mulaw),
        _ if eq("PCMA") => Encoding::Audio(AudioFormat::Alaw),
        _ => return None,
    })
}

/// Decode the codec's `fmtp` parameter sets into one Annex-B buffer, in the
/// order a decoder wants them (H.265: VPS, SPS, PPS). A parameter set that is
/// not valid base64 is dropped.
fn sprop_parameter_sets(codec: VideoCodec, fmtp: &str) -> Vec<u8> {
    let keys: &[&str] = match codec {
        VideoCodec::H264 => &["sprop-parameter-sets"],
        VideoCodec::H265 => &["sprop-vps", "sprop-sps", "sprop-pps"],
        _ => return Vec::new(),
    };
    let mut out = Vec::new();
    for key in keys {
        let Some(value) = fmtp_param(fmtp, key) else {
            continue;
        };
        for encoded in value.split(',') {
            let Some(nal) = base64_decode(encoded.trim()) else {
                continue;
            };
            if nal.is_empty() {
                continue;
            }
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(&nal);
        }
    }
    out
}

/// The value of one `key=value` parameter in an `fmtp` line (`;`-separated,
/// case-insensitive key).
fn fmtp_param<'a>(fmtp: &'a str, key: &str) -> Option<&'a str> {
    fmtp.split(';').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        k.trim().eq_ignore_ascii_case(key).then(|| v.trim())
    })
}

/// Geometry from the SPS in `parameter_sets`, for the codecs whose parameter
/// sets carry it.
fn sps_geometry(codec: VideoCodec, parameter_sets: &[u8]) -> Option<crate::nalparse::SpsGeometry> {
    match codec {
        VideoCodec::H264 => crate::h264parse::H264Codec::extract_sps_info(parameter_sets),
        VideoCodec::H265 => crate::h265parse::H265Codec::extract_sps_info(parameter_sets),
        _ => None,
    }
}

/// Parse an `a=framerate:` value (`15`, `29.97`) into Q16 fixed-point fps, the
/// [`Rate::Fixed`] unit. Integer math only (no `f64::round` in `no_std`).
fn parse_q16_fps(s: &str) -> Option<u32> {
    let s = s.trim();
    let (whole, frac) = match s.split_once('.') {
        Some((w, f)) => (w, f),
        None => (s, ""),
    };
    let mut q16 = whole.parse::<u32>().ok()?.checked_mul(1 << 16)?;
    // Fractional digits scaled to 1/65536: 0.97 -> (97 << 16) / 100.
    let mut num = 0u64;
    let mut den = 1u64;
    for c in frac.chars().take(6) {
        num = num.checked_mul(10)?.checked_add(c.to_digit(10)? as u64)?;
        den = den.checked_mul(10)?;
    }
    if den > 1 {
        q16 = q16.checked_add(((num << 16) / den) as u32)?;
    }
    (q16 > 0).then_some(q16)
}

/// Decode standard-alphabet base64 (RFC 4648, padding optional). `None` on any
/// character outside the alphabet or a truncated final group, so a malformed
/// `sprop` never yields half a parameter set. Hand-rolled to keep this module on
/// the `no_std` baseline instead of pulling an optional dependency into it.
pub(crate) fn base64_decode(s: &str) -> Option<Vec<u8>> {
    fn sextet(c: u8) -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a') as u32 + 26,
            b'0'..=b'9' => (c - b'0') as u32 + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        })
    }
    let bytes: &[u8] = s.trim_end_matches('=').as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for &c in bytes {
        acc = (acc << 6) | sextet(c)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    // A leftover group of one sextet (2 bits short of a byte) is truncated input.
    if bits >= 6 {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SDP `ffmpeg -f rtp -sdp_file` writes for a 320x240@15 H.264 stream
    /// (`-flags +global_header`, so the parameter sets ride in the fmtp).
    const FFMPEG_H264_SDP: &str = "v=0\r\n\
        o=- 0 0 IN IP4 127.0.0.1\r\n\
        s=No Name\r\n\
        c=IN IP4 127.0.0.1\r\n\
        t=0 0\r\n\
        a=tool:libavformat 62.12.102\r\n\
        m=video 45999 RTP/AVP 96\r\n\
        a=framerate:15\r\n\
        a=rtpmap:96 H264/90000\r\n\
        a=fmtp:96 packetization-mode=1; \
        sprop-parameter-sets=Z/QADJGWgUH7ARAAAAMAEAAAAwHg8UKq,aM4PGSA=; profile-level-id=F4000C\r\n";

    #[test]
    fn ffmpeg_h264_sdp_maps_to_absolute_caps() {
        let media = SdpMedia::parse(FFMPEG_H264_SDP).expect("maps the H.264 section");
        assert_eq!(media.payload_type, 96);
        assert_eq!(media.clock_rate, 90_000);
        assert_eq!(media.port, 45999);
        assert_eq!(media.address, "127.0.0.1", "session-level c= is inherited");
        assert_eq!(
            media.caps,
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Fixed(320),
                height: Dim::Fixed(240),
                framerate: Rate::Fixed(15 << 16),
                colorimetry: g2g_core::Colorimetry::UNKNOWN
            },
            "geometry comes from the sprop SPS, rate from a=framerate"
        );
        // The parameter sets decode to Annex-B SPS (type 7) then PPS (type 8).
        assert_eq!(&media.parameter_sets[..5], &[0, 0, 0, 1, 0x67]);
        assert!(
            media
                .parameter_sets
                .windows(5)
                .any(|w| w == [0, 0, 0, 1, 0x68]),
            "PPS follows the SPS"
        );
    }

    #[test]
    fn opus_section_maps_to_audio_caps() {
        let text = "v=0\r\n\
            m=audio 5006 RTP/AVP 111\r\n\
            a=rtpmap:111 opus/48000/2\r\n\
            a=fmtp:111 minptime=10;useinbandfec=1\r\n";
        let media = SdpMedia::parse(text).expect("maps the opus section");
        assert_eq!(
            media.caps,
            Caps::Audio {
                format: AudioFormat::Opus,
                channels: 2,
                sample_rate: 48_000,
            }
        );
        assert_eq!(media.payload_type, 111);
    }

    #[test]
    fn parse_all_keeps_both_sections_and_skips_unknown_codecs() {
        let text = "v=0\r\n\
            c=IN IP4 239.1.2.3\r\n\
            m=video 5000 RTP/AVP 96\r\n\
            a=rtpmap:96 H264/90000\r\n\
            m=audio 5002 RTP/AVP 111\r\n\
            a=rtpmap:111 opus/48000/2\r\n\
            m=application 5004 RTP/AVP 100\r\n\
            a=rtpmap:100 nonsense/8000\r\n";
        let all = SdpMedia::parse_all(text);
        assert_eq!(all.len(), 2, "the unmappable section is skipped");
        assert!(matches!(all[0].caps, Caps::CompressedVideo { .. }));
        assert!(matches!(all[1].caps, Caps::Audio { .. }));
        assert!(all.iter().all(|m| m.address == "239.1.2.3"));
        // No fmtp: the codec is known but the geometry is not invented.
        assert_eq!(
            all[0].caps,
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN
            }
        );
    }

    #[test]
    fn malformed_input_yields_nothing_rather_than_a_guess() {
        assert!(SdpMedia::parse("v=0\r\ns=x\r\n").is_none(), "no m= line");
        // A truncated m= line.
        assert!(SdpMedia::parse("m=video 5000\r\na=rtpmap:96 H264/90000\r\n").is_none());
        // A zero clock rate would divide by zero downstream.
        assert!(SdpMedia::parse("m=video 5000 RTP/AVP 96\r\na=rtpmap:96 H264/0\r\n").is_none());
        // Bad base64 in the sprop: the codec still maps, the geometry does not.
        let bad = "m=video 5000 RTP/AVP 96\r\n\
            a=rtpmap:96 H264/90000\r\n\
            a=fmtp:96 sprop-parameter-sets=!!!!,aM4PGSA=\r\n";
        let media = SdpMedia::parse(bad).expect("codec still maps");
        assert!(matches!(
            media.caps,
            Caps::CompressedVideo {
                width: Dim::Any,
                ..
            }
        ));
    }

    #[test]
    fn fractional_framerate_becomes_q16() {
        // 29.97 fps: 29 << 16 plus 0.97 scaled to 1/65536.
        let q16 = parse_q16_fps("29.97").expect("parses");
        assert_eq!(q16 >> 16, 29);
        assert_eq!(q16, (29 << 16) + ((97u32 << 16) / 100));
        assert_eq!(parse_q16_fps("15"), Some(15 << 16));
        assert_eq!(parse_q16_fps("0"), None, "a zero rate is not a rate");
        assert_eq!(parse_q16_fps("x"), None);
    }

    #[test]
    fn base64_rejects_bad_input_and_decodes_unpadded() {
        assert_eq!(base64_decode("aGk="), Some(alloc::vec![b'h', b'i']));
        assert_eq!(base64_decode("aGk"), Some(alloc::vec![b'h', b'i']));
        assert_eq!(base64_decode("a"), None, "a lone sextet is truncated");
        assert_eq!(base64_decode("a b"), None, "space is outside the alphabet");
    }

    #[test]
    fn h265_sprop_sets_are_concatenated_in_decode_order() {
        // Hand-built VPS (type 32) / SPS (33) / PPS (34) NAL headers, base64'd.
        let fmtp = "sprop-vps=QAEM; sprop-sps=QgEB; sprop-pps=RAHA";
        let sets = sprop_parameter_sets(VideoCodec::H265, fmtp);
        let types: Vec<u8> = sets
            .windows(5)
            .filter(|w| w[..4] == [0, 0, 0, 1])
            .map(|w| (w[4] >> 1) & 0x3F)
            .collect();
        assert_eq!(types, alloc::vec![32, 33, 34], "VPS, SPS, then PPS");
    }
}

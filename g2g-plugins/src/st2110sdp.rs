//! ST 2110 SDP (M601): the Session Description (RFC 4566, SMPTE ST 2110-10/-20/
//! -30/-40) that describes a stream so a receiver auto-configures instead of being
//! hand-set. This is the out-of-band half of ST 2110: the essence (-20 video, -30
//! audio, -40 ancillary) carries no geometry / format / clock on the wire, so a
//! sender publishes an SDP and a receiver parses it to learn the payload type,
//! multicast group and port, the pixel sampling / rate / channels, and the PTP
//! reference clock (`a=ts-refclk`, ST 2110-10) all the streams share.
//!
//! Sans-IO string work (pure `no_std` + alloc), CI round-trip testable, so the
//! network elements build an [`St2110Sdp`] to advertise and configure a source
//! from a parsed one. Generation emits the SMPTE-required attributes; parsing is
//! lenient about attribute order and unknown lines (never trust the stream: an
//! `m=` / `rtpmap` it cannot map, or a missing field, yields `None` rather than a
//! wrong guess).

use alloc::format;
use alloc::string::{String, ToString};

use crate::st2110audio::SampleDepth;
use crate::st2110video::Sampling;

/// The essence a stream carries, with the parameters its SDP media / `rtpmap` /
/// `fmtp` lines describe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum St2110Essence {
    /// ST 2110-20 uncompressed video (`rtpmap raw/90000`). `exact_fps` is the
    /// SMPTE `exactframerate` numerator / denominator (e.g. `(60000, 1001)` for
    /// 59.94, `(50, 1)` for 50).
    Video { sampling: Sampling, width: u32, height: u32, exact_fps: (u32, u32) },
    /// ST 2110-22 JPEG XS compressed video (`rtpmap jxsv/90000`, RFC 9134). Rides
    /// under an `m=video` line like -20, but the `fmtp` names JPEG XS
    /// (`packetmode` / `transmode`) alongside the descriptive sampling / geometry.
    JpegXs { sampling: Sampling, width: u32, height: u32, exact_fps: (u32, u32) },
    /// ST 2110-30 PCM audio (`rtpmap L16`/`L24` / rate / channels, `a=ptime`).
    Audio { depth: SampleDepth, sample_rate: u32, channels: u16, ptime_us: u32 },
    /// ST 2110-40 ancillary data (`rtpmap smpte291/90000`).
    Ancillary,
}

/// A buildable / parsed ST 2110 SDP stream description (one media section).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct St2110Sdp {
    pub essence: St2110Essence,
    pub payload_type: u8,
    /// Multicast group (or unicast address) the media flows to.
    pub address: String,
    pub port: u16,
    /// PTP grandmaster clock identity + domain for `a=ts-refclk` (ST 2110-10), if
    /// known. The identity is the EUI-64 gmid text (e.g. `08-00-11-...-00`).
    pub ptp: Option<(String, u8)>,
}

/// The SMPTE `sampling=` name and `depth=` for a [`Sampling`].
fn sampling_sdp(s: Sampling) -> (&'static str, u8) {
    match s {
        Sampling::Rgba8 => ("RGBA", 8),
        Sampling::YCbCr422_8 => ("YCbCr-4:2:2", 8),
        Sampling::YCbCr422_10 => ("YCbCr-4:2:2", 10),
    }
}

/// The [`Sampling`] for an SMPTE `sampling=` name + `depth=`, or `None` if unmapped.
fn sampling_from_sdp(name: &str, depth: u8) -> Option<Sampling> {
    match (name, depth) {
        ("RGBA", 8) => Some(Sampling::Rgba8),
        ("YCbCr-4:2:2", 8) => Some(Sampling::YCbCr422_8),
        ("YCbCr-4:2:2", 10) => Some(Sampling::YCbCr422_10),
        _ => None,
    }
}

impl St2110Sdp {
    /// Serialize to an SDP text block (`\r\n` line endings, per RFC 4566). Emits the
    /// session lines, the essence's `m=` / `c=` / `rtpmap` / `fmtp`, and, when a PTP
    /// grandmaster is known, the ST 2110-10 `a=ts-refclk` / `a=mediaclk` lines.
    pub fn to_sdp(&self) -> String {
        let mut s = String::new();
        push_session_header(&mut s, &self.address);
        self.push_media(&mut s, None);
        s
    }

    /// Append this stream's media-level lines (`m=` / `c=` / `rtpmap` / `fmtp`, an
    /// optional `a=mid`, and the per-media `a=ts-refclk` / `a=mediaclk`) to an
    /// in-progress SDP. Used both by [`Self::to_sdp`] (one stream) and by
    /// [`St2110Session::to_sdp`] (several streams under one session header).
    fn push_media(&self, s: &mut String, mid: Option<usize>) {
        let pt = self.payload_type;
        match &self.essence {
            St2110Essence::Video { sampling, width, height, exact_fps } => {
                let (samp, depth) = sampling_sdp(*sampling);
                s.push_str(&format!("m=video {} RTP/AVP {}\r\n", self.port, pt));
                s.push_str(&format!("c=IN IP4 {}/64\r\n", self.address));
                s.push_str(&format!("a=rtpmap:{pt} raw/90000\r\n"));
                s.push_str(&format!(
                    "a=fmtp:{pt} sampling={samp}; width={width}; height={height}; \
                     exactframerate={}; depth={depth}; colorimetry=BT709; PM=2110GPM; \
                     SSN=ST2110-20:2017; TP=2110TPN\r\n",
                    fps_str(*exact_fps),
                ));
            }
            St2110Essence::JpegXs { sampling, width, height, exact_fps } => {
                let (samp, depth) = sampling_sdp(*sampling);
                s.push_str(&format!("m=video {} RTP/AVP {}\r\n", self.port, pt));
                s.push_str(&format!("c=IN IP4 {}/64\r\n", self.address));
                s.push_str(&format!("a=rtpmap:{pt} jxsv/90000\r\n"));
                // packetmode=0 (codestream mode), transmode=1 (sequential): what the
                // -22 element (`st2110jxs`, RFC 9134) sends.
                s.push_str(&format!(
                    "a=fmtp:{pt} packetmode=0; transmode=1; sampling={samp}; width={width}; \
                     height={height}; exactframerate={}; depth={depth}; colorimetry=BT709; \
                     SSN=ST2110-22:2019; TP=2110TPN\r\n",
                    fps_str(*exact_fps),
                ));
            }
            St2110Essence::Audio { depth, sample_rate, channels, ptime_us } => {
                let enc = match depth {
                    SampleDepth::L16 => "L16",
                    SampleDepth::L24 => "L24",
                };
                s.push_str(&format!("m=audio {} RTP/AVP {}\r\n", self.port, pt));
                s.push_str(&format!("c=IN IP4 {}/64\r\n", self.address));
                s.push_str(&format!("a=rtpmap:{pt} {enc}/{sample_rate}/{channels}\r\n"));
                // ptime in milliseconds (e.g. 1 ms -> "1.000", 125 us -> "0.125").
                s.push_str(&format!(
                    "a=ptime:{}.{:03}\r\n",
                    ptime_us / 1000,
                    ptime_us % 1000
                ));
            }
            St2110Essence::Ancillary => {
                // ST 2110-40 rides under an m=video line per SMPTE.
                s.push_str(&format!("m=video {} RTP/AVP {}\r\n", self.port, pt));
                s.push_str(&format!("c=IN IP4 {}/64\r\n", self.address));
                s.push_str(&format!("a=rtpmap:{pt} smpte291/90000\r\n"));
            }
        }
        if let Some(m) = mid {
            s.push_str(&format!("a=mid:{m}\r\n"));
        }
        if let Some((gmid, domain)) = &self.ptp {
            s.push_str(&format!("a=ts-refclk:ptp=IEEE1588-2008:{gmid}:{domain}\r\n"));
            s.push_str("a=mediaclk:direct=0\r\n");
        }
    }

    /// Parse the first media section of an SDP text block. Returns `None` if there
    /// is no `m=` line, the `rtpmap` encoding is not an ST 2110 essence, or a
    /// required parameter is missing / unmappable.
    pub fn parse(text: &str) -> Option<Self> {
        let mut port = None;
        let mut media_kind = None; // "video" / "audio"
        let mut payload_type = None;
        let mut address = None;
        let mut rtpmap = None; // the encoding string after "<pt> "
        let mut fmtp = None; // the params string after "<pt> "
        let mut ptime_us = None;
        let mut ptp = None;

        for raw in text.lines() {
            let line = raw.trim_end_matches('\r');
            if let Some(m) = line.strip_prefix("m=") {
                // "video <port> RTP/AVP <pt>"; only the first media section.
                if media_kind.is_some() {
                    break;
                }
                let mut it = m.split_whitespace();
                media_kind = Some(it.next()?.to_string());
                port = Some(it.next()?.parse::<u16>().ok()?);
                let _proto = it.next()?; // RTP/AVP
                payload_type = Some(it.next()?.parse::<u8>().ok()?);
            } else if let Some(c) = line.strip_prefix("c=IN IP4 ") {
                // "<addr>/<ttl>" or "<addr>".
                address = Some(c.split('/').next()?.to_string());
            } else if let Some(a) = line.strip_prefix("a=rtpmap:") {
                rtpmap = Some(after_pt(a)?.to_string());
            } else if let Some(a) = line.strip_prefix("a=fmtp:") {
                fmtp = Some(after_pt(a)?.to_string());
            } else if let Some(a) = line.strip_prefix("a=ptime:") {
                // Milliseconds (possibly fractional) -> microseconds.
                ptime_us = parse_ptime_us(a);
            } else if let Some(a) = line.strip_prefix("a=ts-refclk:ptp=") {
                // "IEEE1588-2008:<gmid>:<domain>".
                let rest = a.split_once(':')?.1; // drop the profile
                let (gmid, domain) = rest.rsplit_once(':')?;
                ptp = Some((gmid.to_string(), domain.parse::<u8>().ok()?));
            }
        }

        let essence = match (media_kind.as_deref(), rtpmap.as_deref()) {
            (Some("video"), Some(enc)) if enc.starts_with("raw/") => {
                let (sampling, width, height, exact_fps) = parse_video_fmtp(fmtp.as_deref()?)?;
                St2110Essence::Video { sampling, width, height, exact_fps }
            }
            (Some("video"), Some(enc)) if enc.starts_with("jxsv/") => {
                let (sampling, width, height, exact_fps) = parse_video_fmtp(fmtp.as_deref()?)?;
                St2110Essence::JpegXs { sampling, width, height, exact_fps }
            }
            (Some("video"), Some(enc)) if enc.starts_with("smpte291/") => {
                St2110Essence::Ancillary
            }
            (Some("audio"), Some(enc)) => parse_audio_rtpmap(enc, ptime_us.unwrap_or(1000))?,
            _ => return None,
        };
        Some(St2110Sdp {
            essence,
            payload_type: payload_type?,
            address: address.unwrap_or_else(|| "0.0.0.0".to_string()),
            port: port?,
            ptp,
        })
    }
}

/// Push the RFC 4566 session-level header lines (`v=` / `o=` / `s=` / `t=`) shared
/// by a single-stream SDP and a multi-stream session.
fn push_session_header(s: &mut String, address: &str) {
    s.push_str("v=0\r\n");
    s.push_str(&format!("o=- 0 0 IN IP4 {address}\r\n"));
    s.push_str("s=g2g ST 2110\r\n");
    s.push_str("t=0 0\r\n");
}

/// A full ST 2110 program: several essence streams (video + audio + ancillary)
/// described by one SDP session, each media section tagged with `a=mid`, so a whole
/// program self-describes in a single out-of-band document (ST 2110-10 puts the
/// essences on a shared PTP grandmaster). Each stream keeps its own
/// [`St2110Sdp`]; this just gathers them under one session header.
///
/// (This models a multi-essence program, not ST 2110-7 seamless protection: -7 pairs
/// two *identical* streams under `a=group:DUP`, a distinct redundancy feature.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct St2110Session {
    /// Session origin / connection address (the `o=` line).
    pub address: String,
    /// The essence streams, in `a=mid` order.
    pub media: alloc::vec::Vec<St2110Sdp>,
    /// ST 2110-7 seamless-protection groups: each inner list is the set of media
    /// indices (`a=mid`) carrying the *same* stream on redundant paths, emitted as a
    /// session-level `a=group:DUP <mid>...`. Empty for a plain program.
    pub dup_groups: alloc::vec::Vec<alloc::vec::Vec<usize>>,
}

impl St2110Session {
    /// Serialize the whole program: one session header, then each stream's media
    /// block (with an `a=mid` index), sharing the session.
    pub fn to_sdp(&self) -> String {
        let mut s = String::new();
        push_session_header(&mut s, &self.address);
        // Session-level -7 grouping (before the media sections, per RFC 5888).
        for group in &self.dup_groups {
            let mids: alloc::vec::Vec<String> = group.iter().map(|m| m.to_string()).collect();
            s.push_str(&format!("a=group:DUP {}\r\n", mids.join(" ")));
        }
        for (i, m) in self.media.iter().enumerate() {
            m.push_media(&mut s, Some(i));
        }
        s
    }

    /// Parse every media section of an SDP session into its [`St2110Sdp`]. A
    /// session-level `a=ts-refclk` (before the first `m=`) is inherited by any media
    /// that does not carry its own. Returns `None` if there is no parseable ST 2110
    /// media section. Unmappable sections are skipped (never trust the stream).
    pub fn parse(text: &str) -> Option<Self> {
        let lines: alloc::vec::Vec<&str> =
            text.lines().map(|l| l.trim_end_matches('\r')).collect();
        let m_idx: alloc::vec::Vec<usize> =
            lines.iter().enumerate().filter(|(_, l)| l.starts_with("m=")).map(|(i, _)| i).collect();
        let first_m = *m_idx.first()?;

        // Session-level origin address, default reference clock, and -7 groups.
        let mut address = None;
        let mut session_refclk = None;
        let mut dup_groups = alloc::vec::Vec::new();
        for l in &lines[..first_m] {
            if let Some(o) = l.strip_prefix("o=") {
                address = o.rsplit(' ').next().map(|a| a.to_string());
            } else if l.starts_with("a=ts-refclk:") {
                session_refclk = Some(*l);
            } else if let Some(g) = l.strip_prefix("a=group:DUP ") {
                let mids: alloc::vec::Vec<usize> =
                    g.split_whitespace().filter_map(|m| m.parse().ok()).collect();
                if !mids.is_empty() {
                    dup_groups.push(mids);
                }
            }
        }

        let mut media = alloc::vec::Vec::new();
        for (k, &start) in m_idx.iter().enumerate() {
            let end = m_idx.get(k + 1).copied().unwrap_or(lines.len());
            let section = &lines[start..end];
            let mut chunk = section.join("\r\n");
            // Inherit the session reference clock when the section has none of its own.
            if let Some(refclk) = session_refclk {
                if !section.iter().any(|l| l.starts_with("a=ts-refclk:")) {
                    chunk.push_str("\r\n");
                    chunk.push_str(refclk);
                }
            }
            if let Some(sdp) = St2110Sdp::parse(&chunk) {
                media.push(sdp);
            }
        }
        if media.is_empty() {
            return None;
        }
        let address = address
            .or_else(|| media.first().map(|m| m.address.clone()))
            .unwrap_or_else(|| "0.0.0.0".to_string());
        Some(St2110Session { address, media, dup_groups })
    }
}

/// Format an `exactframerate` (num, den): bare integer when den == 1, else `n/d`.
fn fps_str((num, den): (u32, u32)) -> String {
    if den <= 1 {
        num.to_string()
    } else {
        format!("{num}/{den}")
    }
}

/// Drop the leading `<pt> ` from an `a=rtpmap:` / `a=fmtp:` value, returning the rest.
fn after_pt(s: &str) -> Option<&str> {
    s.split_once(' ').map(|(_, rest)| rest)
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

/// Parse the shared video `a=fmtp:` geometry (`sampling` / `width` / `height` /
/// `depth` / `exactframerate`) into `(sampling, width, height, exact_fps)`, used by
/// both -20 (`raw`) and -22 (`jxsv`). JPEG XS `fmtp` keys we do not map
/// (`packetmode`, `transmode`, ...) are ignored. `None` if a required field is
/// missing or the sampling / depth pair is unmapped.
fn parse_video_fmtp(params: &str) -> Option<(Sampling, u32, u32, (u32, u32))> {
    let mut sampling_name = None;
    let mut width = None;
    let mut height = None;
    let mut depth = None;
    let mut fps = (30u32, 1u32);
    for kv in params.split(';') {
        let (k, v) = kv.split_once('=')?;
        let (k, v) = (k.trim(), v.trim());
        match k {
            "sampling" => sampling_name = Some(v.to_string()),
            "width" => width = v.parse::<u32>().ok(),
            "height" => height = v.parse::<u32>().ok(),
            "depth" => depth = v.parse::<u8>().ok(),
            "exactframerate" => {
                fps = match v.split_once('/') {
                    Some((n, d)) => (n.parse().ok()?, d.parse().ok()?),
                    None => (v.parse().ok()?, 1),
                };
            }
            _ => {}
        }
    }
    let sampling = sampling_from_sdp(&sampling_name?, depth?)?;
    Some((sampling, width?, height?, fps))
}

/// Parse an audio `rtpmap` (`L16`/`L24` `/rate/channels`) into [`St2110Essence::Audio`].
fn parse_audio_rtpmap(enc: &str, ptime_us: u32) -> Option<St2110Essence> {
    let mut it = enc.split('/');
    let depth = match it.next()? {
        "L16" => SampleDepth::L16,
        "L24" => SampleDepth::L24,
        _ => return None,
    };
    let sample_rate = it.next()?.parse::<u32>().ok()?;
    let channels = it.next()?.parse::<u16>().ok()?;
    Some(St2110Essence::Audio { depth, sample_rate, channels, ptime_us })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(sdp: &St2110Sdp) {
        let text = sdp.to_sdp();
        let parsed = St2110Sdp::parse(&text).expect("parses its own output");
        assert_eq!(&parsed, sdp, "SDP round-trips\n{text}");
    }

    #[test]
    fn video_10bit_round_trips() {
        round_trip(&St2110Sdp {
            essence: St2110Essence::Video {
                sampling: Sampling::YCbCr422_10,
                width: 1920,
                height: 1080,
                exact_fps: (60000, 1001),
            },
            payload_type: 96,
            address: "239.10.20.1".to_string(),
            port: 5008,
            ptp: Some(("08-00-11-FF-FE-21-E1-B0".to_string(), 127)),
        });
    }

    #[test]
    fn video_rgba_and_yuyv_round_trip() {
        for sampling in [Sampling::Rgba8, Sampling::YCbCr422_8] {
            round_trip(&St2110Sdp {
                essence: St2110Essence::Video { sampling, width: 1280, height: 720, exact_fps: (50, 1) },
                payload_type: 112,
                address: "192.168.1.5".to_string(),
                port: 5000,
                ptp: None,
            });
        }
    }

    #[test]
    fn jpegxs_round_trips_and_advertises_jxsv() {
        let sdp = St2110Sdp {
            essence: St2110Essence::JpegXs {
                sampling: Sampling::YCbCr422_10,
                width: 1920,
                height: 1080,
                exact_fps: (60, 1),
            },
            payload_type: 112,
            address: "239.22.1.1".to_string(),
            port: 5010,
            ptp: Some(("08-00-11-FF-FE-21-E1-B0".to_string(), 127)),
        };
        let text = sdp.to_sdp();
        assert!(text.contains("a=rtpmap:112 jxsv/90000"), "advertises the -22 rtpmap\n{text}");
        assert!(text.contains("packetmode=0"), "codestream packetization mode\n{text}");
        assert!(text.contains("SSN=ST2110-22:2019"), "names the -22 spec\n{text}");
        round_trip(&sdp);
    }

    #[test]
    fn audio_and_ancillary_round_trip() {
        round_trip(&St2110Sdp {
            essence: St2110Essence::Audio {
                depth: SampleDepth::L24,
                sample_rate: 48_000,
                channels: 2,
                ptime_us: 125,
            },
            payload_type: 97,
            address: "239.30.1.1".to_string(),
            port: 5004,
            ptp: Some(("AA-BB-CC-DD-EE-FF-00-11".to_string(), 0)),
        });
        round_trip(&St2110Sdp {
            essence: St2110Essence::Ancillary,
            payload_type: 100,
            address: "239.40.1.1".to_string(),
            port: 5006,
            ptp: None,
        });
    }

    #[test]
    fn session_bundles_video_audio_anc_and_round_trips() {
        let ptp = Some(("08-00-11-FF-FE-21-E1-B0".to_string(), 127));
        let session = St2110Session {
            address: "239.100.0.1".to_string(),
            media: alloc::vec![
                St2110Sdp {
                    essence: St2110Essence::Video {
                        sampling: Sampling::YCbCr422_10,
                        width: 1920,
                        height: 1080,
                        exact_fps: (60000, 1001),
                    },
                    payload_type: 96,
                    address: "239.100.0.1".to_string(),
                    port: 5000,
                    ptp: ptp.clone(),
                },
                St2110Sdp {
                    essence: St2110Essence::Audio {
                        depth: SampleDepth::L24,
                        sample_rate: 48_000,
                        channels: 2,
                        ptime_us: 125,
                    },
                    payload_type: 97,
                    address: "239.100.0.2".to_string(),
                    port: 5002,
                    ptp: ptp.clone(),
                },
                St2110Sdp {
                    essence: St2110Essence::Ancillary,
                    payload_type: 100,
                    address: "239.100.0.3".to_string(),
                    port: 5004,
                    ptp: ptp.clone(),
                },
            ],
            dup_groups: alloc::vec::Vec::new(),
        };
        let text = session.to_sdp();
        // One session header, three media sections, each tagged with a=mid.
        assert_eq!(text.matches("m=").count(), 3, "three media sections\n{text}");
        assert_eq!(text.matches("v=0").count(), 1, "one session header\n{text}");
        assert!(text.contains("a=mid:0") && text.contains("a=mid:2"), "media tagged\n{text}");

        let parsed = St2110Session::parse(&text).expect("session parses");
        assert_eq!(parsed, session, "the whole program round-trips");
    }

    #[test]
    fn session_dup_group_round_trips() {
        // ST 2110-7: the same video on two redundant paths, grouped a=group:DUP 0 1.
        let video = |addr: &str, port| St2110Sdp {
            essence: St2110Essence::Video {
                sampling: Sampling::YCbCr422_10,
                width: 1920,
                height: 1080,
                exact_fps: (60, 1),
            },
            payload_type: 96,
            address: addr.to_string(),
            port,
            ptp: Some(("08-00-11-FF-FE-21-E1-B0".to_string(), 127)),
        };
        let session = St2110Session {
            address: "239.100.0.1".to_string(),
            media: alloc::vec![video("239.100.0.1", 5000), video("239.101.0.1", 5000)],
            dup_groups: alloc::vec![alloc::vec![0, 1]],
        };
        let text = session.to_sdp();
        assert!(text.contains("a=group:DUP 0 1"), "advertises the -7 duplicate group\n{text}");
        let parsed = St2110Session::parse(&text).expect("parses");
        assert_eq!(parsed, session, "the redundant-pair session round-trips");
    }

    #[test]
    fn session_parses_a_multi_essence_sdp_with_shared_refclk() {
        // A hand-written program SDP: one session-level ts-refclk shared by all media.
        let text = "v=0\r\n\
            o=- 123 45 IN IP4 192.168.0.1\r\n\
            s=SMPTE ST 2110\r\n\
            t=0 0\r\n\
            a=ts-refclk:ptp=IEEE1588-2008:39-A7-94-FF-FE-07-CB-D0:127\r\n\
            m=video 5000 RTP/AVP 96\r\n\
            c=IN IP4 239.100.0.1/64\r\n\
            a=rtpmap:96 raw/90000\r\n\
            a=fmtp:96 sampling=YCbCr-4:2:2; width=1280; height=720; exactframerate=50; depth=10\r\n\
            a=mid:0\r\n\
            m=audio 5002 RTP/AVP 97\r\n\
            c=IN IP4 239.100.0.2/64\r\n\
            a=rtpmap:97 L24/48000/2\r\n\
            a=ptime:1.000\r\n\
            a=mid:1\r\n";
        let session = St2110Session::parse(text).expect("parses");
        assert_eq!(session.media.len(), 2, "video + audio");
        // The session-level PTP clock is inherited by both streams.
        assert!(session.media.iter().all(|m| m.ptp.as_ref().map(|(_, d)| *d) == Some(127)));
        assert!(matches!(session.media[0].essence, St2110Essence::Video { width: 1280, .. }));
        assert!(matches!(session.media[1].essence, St2110Essence::Audio { channels: 2, .. }));
    }

    #[test]
    fn parses_a_realistic_2110_20_sdp() {
        // A hand-written SMPTE ST 2110-20 SDP (attribute order / extra fields the
        // parser tolerates).
        let text = "v=0\r\n\
            o=- 123456 11 IN IP4 192.168.0.1\r\n\
            s=SMPTE ST2110-20\r\n\
            t=0 0\r\n\
            m=video 5000 RTP/AVP 96\r\n\
            c=IN IP4 239.100.0.1/64\r\n\
            a=source-filter: incl IN IP4 239.100.0.1 192.168.0.1\r\n\
            a=rtpmap:96 raw/90000\r\n\
            a=fmtp:96 sampling=YCbCr-4:2:2; width=1920; height=1080; exactframerate=25; depth=10; colorimetry=BT709; PM=2110GPM; SSN=ST2110-20:2017; TP=2110TPN\r\n\
            a=ts-refclk:ptp=IEEE1588-2008:39-A7-94-FF-FE-07-CB-D0:127\r\n\
            a=mediaclk:direct=0\r\n";
        let sdp = St2110Sdp::parse(text).expect("parses");
        assert_eq!(sdp.payload_type, 96);
        assert_eq!(sdp.address, "239.100.0.1");
        assert_eq!(sdp.port, 5000);
        assert_eq!(sdp.ptp, Some(("39-A7-94-FF-FE-07-CB-D0".to_string(), 127)));
        assert_eq!(
            sdp.essence,
            St2110Essence::Video {
                sampling: Sampling::YCbCr422_10,
                width: 1920,
                height: 1080,
                exact_fps: (25, 1),
            }
        );
    }

    #[test]
    fn rejects_non_2110_or_incomplete_sdp() {
        // No m= line.
        assert!(St2110Sdp::parse("v=0\r\ns=x\r\n").is_none());
        // An unmapped codec (H.264) is not an ST 2110 essence.
        let h264 = "m=video 5000 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n";
        assert!(St2110Sdp::parse(h264).is_none());
        // Video with no fmtp (missing geometry) is rejected, not guessed.
        let no_fmtp = "m=video 5000 RTP/AVP 96\r\na=rtpmap:96 raw/90000\r\n";
        assert!(St2110Sdp::parse(no_fmtp).is_none());
        // An unknown sampling / depth combination is rejected.
        let bad = "m=video 5000 RTP/AVP 96\r\na=rtpmap:96 raw/90000\r\n\
            a=fmtp:96 sampling=YCbCr-4:4:4; width=8; height=8; depth=8\r\n";
        assert!(St2110Sdp::parse(bad).is_none());
    }
}

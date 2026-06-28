//! HLS playlist parser (RFC 8216), pure `no_std + alloc`. Parses the two
//! `.m3u8` forms: a *master* playlist (a set of `#EXT-X-STREAM-INF` variant
//! streams for ABR selection) and a *media* playlist (an ordered list of
//! `#EXTINF` segments). [`HlsSrc`](crate::hlssrc) drives this; the parser does no
//! I/O so it is fully unit-testable.
//!
//! A playlist is one form or the other: presence of any `#EXT-X-STREAM-INF`
//! makes it a master, otherwise it is a media playlist. URIs are kept verbatim
//! (absolute or relative); the caller resolves them against the playlist URL.

use alloc::string::String;
use alloc::vec::Vec;

/// One variant stream in a master playlist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variant {
    pub bandwidth: u64,
    pub resolution: Option<(u32, u32)>,
    pub codecs: Option<String>,
    pub uri: String,
}

/// `#EXT-X-KEY` encryption method. `SampleAes` is recognized but unsupported by
/// [`HlsSrc`](crate::hlssrc) (per-sample, not whole-segment encryption); a
/// `METHOD=NONE` tag clears the key rather than producing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyMethod {
    Aes128,
    SampleAes,
}

/// The decryption context a preceding `#EXT-X-KEY` puts in effect for a segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentKey {
    pub method: KeyMethod,
    /// Key resource URI (the caller resolves it against the playlist URL).
    pub uri: String,
    /// Explicit `IV` (16 bytes) from the tag, or `None` to derive it from the
    /// segment's media-sequence number.
    pub iv: Option<[u8; 16]>,
}

/// A byte sub-range of a resource (`#EXT-X-BYTERANGE` / `#EXT-X-MAP:BYTERANGE`):
/// `length` bytes starting at `offset`. Single-file CMAF/fMP4 packs the init
/// segment and every media fragment into one resource addressed by range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    pub offset: u64,
    pub length: u64,
}

/// One media segment in a media playlist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub uri: String,
    /// Segment duration in milliseconds (from `#EXTINF`, seconds * 1000).
    pub duration_ms: u32,
    /// Decryption context from the `#EXT-X-KEY` in effect, `None` if unencrypted.
    pub key: Option<SegmentKey>,
    /// `#EXT-X-BYTERANGE` sub-range of `uri`, `None` for a whole-resource segment.
    pub byte_range: Option<ByteRange>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MasterPlaylist {
    pub variants: Vec<Variant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaPlaylist {
    pub target_duration_secs: u32,
    pub media_sequence: u64,
    pub segments: Vec<Segment>,
    /// `#EXT-X-MAP:URI` initialization segment (fMP4/CMAF): the `ftyp`+`moov`
    /// prepended before the media fragments. `None` for an MPEG-TS playlist.
    pub map_uri: Option<String>,
    /// `#EXT-X-MAP:BYTERANGE` sub-range of `map_uri` (single-file CMAF), `None`
    /// when the init segment is the whole resource.
    pub map_byte_range: Option<ByteRange>,
    /// `#EXT-X-ENDLIST` present: a complete VOD playlist (no live reload).
    pub end_list: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Playlist {
    Master(MasterPlaylist),
    Media(MediaPlaylist),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HlsError {
    /// Missing the leading `#EXTM3U` tag.
    NotAPlaylist,
    /// A tag that requires a following URI line had none.
    DanglingTag,
}

impl MasterPlaylist {
    /// Pick the highest-bandwidth variant at or below `max_bandwidth` (or the
    /// overall highest when `None` / nothing fits). The simplest ABR rule; a
    /// real rate adaptor would track throughput across segments.
    pub fn select(&self, max_bandwidth: Option<u64>) -> Option<&Variant> {
        let under = |v: &&Variant| max_bandwidth.map_or(true, |cap| v.bandwidth <= cap);
        self.variants
            .iter()
            .filter(under)
            .max_by_key(|v| v.bandwidth)
            .or_else(|| self.variants.iter().min_by_key(|v| v.bandwidth))
    }
}

/// Parse a `.m3u8` playlist. Returns master or media form.
pub fn parse(text: &str) -> Result<Playlist, HlsError> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());

    match lines.next() {
        Some("#EXTM3U") => {}
        _ => return Err(HlsError::NotAPlaylist),
    }

    let mut variants = Vec::new();
    let mut segments = Vec::new();
    let mut target_duration_secs = 0u32;
    let mut media_sequence = 0u64;
    let mut map_uri = None;
    let mut map_byte_range = None;
    let mut end_list = false;
    // The `#EXT-X-KEY` carries forward to every following segment until the next
    // one changes or clears it.
    let mut current_key: Option<SegmentKey> = None;
    // A tag carries over to the next URI line: Some(duration_ms) for a segment,
    // or the variant being built for a stream-inf.
    let mut pending_segment: Option<u32> = None;
    let mut pending_variant: Option<Variant> = None;
    // `#EXT-X-BYTERANGE` for the next segment: (length, explicit offset). An
    // absent offset is resolved from the previous sub-range of the same URI.
    let mut pending_byterange: Option<(u64, Option<u64>)> = None;

    for line in lines {
        if let Some(attrs) = line.strip_prefix("#EXT-X-STREAM-INF:") {
            pending_variant = Some(parse_stream_inf(attrs));
        } else if let Some(rest) = line.strip_prefix("#EXTINF:") {
            pending_segment = Some(parse_extinf_ms(rest));
        } else if let Some(rest) = line.strip_prefix("#EXT-X-BYTERANGE:") {
            pending_byterange = Some(parse_byterange(rest));
        } else if let Some(rest) = line.strip_prefix("#EXT-X-TARGETDURATION:") {
            target_duration_secs = rest.trim().parse().unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("#EXT-X-MEDIA-SEQUENCE:") {
            media_sequence = rest.trim().parse().unwrap_or(0);
        } else if let Some(attrs) = line.strip_prefix("#EXT-X-MAP:") {
            let pairs = attr_pairs(attrs);
            map_uri = pairs
                .iter()
                .find(|(k, _)| *k == "URI")
                .map(|(_, v)| String::from(v.trim_matches('"')));
            map_byte_range = pairs
                .iter()
                .find(|(k, _)| *k == "BYTERANGE")
                .and_then(|(_, v)| {
                    let (length, offset) = parse_byterange(v.trim_matches('"'));
                    (length > 0)
                        .then_some(ByteRange { offset: offset.unwrap_or(0), length })
                });
        } else if let Some(attrs) = line.strip_prefix("#EXT-X-KEY:") {
            current_key = parse_key(attrs);
        } else if line == "#EXT-X-ENDLIST" {
            end_list = true;
        } else if line.starts_with('#') {
            // any other tag / comment: ignored
        } else if let Some(mut variant) = pending_variant.take() {
            variant.uri = String::from(line);
            variants.push(variant);
        } else if let Some(duration_ms) = pending_segment.take() {
            let uri = String::from(line);
            let byte_range = pending_byterange.take().map(|(length, offset)| {
                // An absent `@offset` continues from the previous sub-range of the
                // same resource (RFC 8216 §4.3.2.2), else starts at 0.
                let offset = offset.unwrap_or_else(|| {
                    segments
                        .last()
                        .filter(|s: &&Segment| s.uri == uri)
                        .and_then(|s| s.byte_range)
                        .map_or(0, |r| r.offset.saturating_add(r.length))
                });
                ByteRange { offset, length }
            });
            segments.push(Segment { uri, duration_ms, key: current_key.clone(), byte_range });
        }
        // a bare URI with no pending tag is ignored
    }

    if pending_variant.is_some() || pending_segment.is_some() {
        return Err(HlsError::DanglingTag);
    }

    if !variants.is_empty() {
        Ok(Playlist::Master(MasterPlaylist { variants }))
    } else {
        Ok(Playlist::Media(MediaPlaylist {
            target_duration_secs,
            media_sequence,
            segments,
            map_uri,
            map_byte_range,
            end_list,
        }))
    }
}

/// Parse an `#EXT-X-STREAM-INF` attribute list. Unknown attributes are ignored;
/// the URI is filled in by the caller from the following line.
fn parse_stream_inf(attrs: &str) -> Variant {
    let mut bandwidth = 0u64;
    let mut resolution = None;
    let mut codecs = None;
    for (key, value) in attr_pairs(attrs) {
        match key {
            "BANDWIDTH" => bandwidth = value.parse().unwrap_or(0),
            "RESOLUTION" => resolution = parse_resolution(value),
            "CODECS" => codecs = Some(String::from(value.trim_matches('"'))),
            _ => {}
        }
    }
    Variant { bandwidth, resolution, codecs, uri: String::new() }
}

/// Parse an `#EXT-X-KEY` attribute list. `METHOD=NONE` (or a missing/unknown
/// method or a keyed method with no `URI`) yields `None`, clearing encryption.
fn parse_key(attrs: &str) -> Option<SegmentKey> {
    let pairs = attr_pairs(attrs);
    let find = |name: &str| pairs.iter().find(|(k, _)| *k == name).map(|(_, v)| *v);
    let method = match find("METHOD")? {
        "AES-128" => KeyMethod::Aes128,
        "SAMPLE-AES" => KeyMethod::SampleAes,
        _ => return None,
    };
    let uri = String::from(find("URI")?.trim_matches('"'));
    let iv = find("IV").and_then(parse_iv);
    Some(SegmentKey { method, uri, iv })
}

/// `IV=0x<32 hex digits>` -> 16 bytes. Anything else is rejected.
fn parse_iv(value: &str) -> Option<[u8; 16]> {
    let hex = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X"))?;
    // Work on bytes: slicing the &str by index would panic on a non-ASCII byte
    // that lands mid-char in the untrusted attribute.
    let hex = hex.as_bytes();
    if hex.len() != 32 {
        return None;
    }
    let mut iv = [0u8; 16];
    for (i, byte) in iv.iter_mut().enumerate() {
        let hi = (hex[i * 2] as char).to_digit(16)?;
        let lo = (hex[i * 2 + 1] as char).to_digit(16)?;
        *byte = (hi * 16 + lo) as u8;
    }
    Some(iv)
}

/// Split a comma-separated `KEY=VALUE` attribute list, respecting double-quoted
/// values (which may contain commas, as `CODECS` does).
fn attr_pairs(attrs: &str) -> Vec<(&str, &str)> {
    let mut pairs = Vec::new();
    let bytes = attrs.as_bytes();
    let (mut start, mut in_quotes) = (0usize, false);
    let mut i = 0;
    while i <= bytes.len() {
        let at_end = i == bytes.len();
        if !at_end && bytes[i] == b'"' {
            in_quotes = !in_quotes;
        }
        if at_end || (bytes[i] == b',' && !in_quotes) {
            if let Some((k, v)) = attrs[start..i].split_once('=') {
                pairs.push((k.trim(), v.trim()));
            }
            start = i + 1;
        }
        i += 1;
    }
    pairs
}

fn parse_resolution(value: &str) -> Option<(u32, u32)> {
    let (w, h) = value.split_once('x')?;
    Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
}

/// `<n>[@<o>]` (an `#EXT-X-BYTERANGE` value or a MAP `BYTERANGE` attribute) ->
/// (length `n`, optional offset `o`). A missing / malformed length yields `0`,
/// so a bogus tag produces an inert range rather than a parse failure.
fn parse_byterange(rest: &str) -> (u64, Option<u64>) {
    let (n, o) = match rest.trim().split_once('@') {
        Some((n, o)) => (n, Some(o)),
        None => (rest.trim(), None),
    };
    let length = n.trim().parse().unwrap_or(0);
    let offset = o.and_then(|o| o.trim().parse().ok());
    (length, offset)
}

/// `#EXTINF:<seconds>[,<title>]` -> duration in ms. Seconds may be fractional.
fn parse_extinf_ms(rest: &str) -> u32 {
    let secs = rest.split(',').next().unwrap_or("").trim();
    match secs.parse::<f32>() {
        Ok(s) if s.is_finite() && s >= 0.0 => (s * 1000.0) as u32,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_iv_rejects_non_ascii_without_panicking() {
        // 29 hex zeros + a 3-byte char: 32 bytes total but not ASCII, so the old
        // str-index slicing would panic on a char boundary. Must reject cleanly.
        let mut bad = String::from("0x");
        bad.push_str(&"0".repeat(29));
        bad.push('€');
        assert!(parse_iv(&bad).is_none());
        // a valid IV still parses
        assert_eq!(
            parse_iv("0x000102030405060708090a0b0c0d0e0f"),
            Some([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        );
    }

    #[test]
    fn parses_media_playlist_with_fractional_durations() {
        let text = "#EXTM3U\n\
            #EXT-X-VERSION:3\n\
            #EXT-X-TARGETDURATION:10\n\
            #EXT-X-MEDIA-SEQUENCE:7\n\
            #EXTINF:9.009,\n\
            seg0.ts\n\
            #EXTINF:9.009,\n\
            seg1.ts\n\
            #EXTINF:3.003,\n\
            seg2.ts\n\
            #EXT-X-ENDLIST\n";
        let Playlist::Media(m) = parse(text).unwrap() else {
            panic!("expected media playlist");
        };
        assert_eq!(m.target_duration_secs, 10);
        assert_eq!(m.media_sequence, 7);
        assert!(m.end_list);
        assert_eq!(m.segments.len(), 3);
        assert_eq!(
            m.segments[0],
            Segment { uri: "seg0.ts".into(), duration_ms: 9009, key: None, byte_range: None }
        );
        assert_eq!(m.segments[2].duration_ms, 3003);
    }

    #[test]
    fn parses_fmp4_media_playlist_with_init_map() {
        let text = "#EXTM3U\n\
            #EXT-X-TARGETDURATION:4\n\
            #EXT-X-MAP:URI=\"init.mp4\"\n\
            #EXTINF:4.0,\n\
            seg0.m4s\n\
            #EXT-X-ENDLIST\n";
        let Playlist::Media(m) = parse(text).unwrap() else {
            panic!("expected media playlist");
        };
        assert_eq!(m.map_uri.as_deref(), Some("init.mp4"), "EXT-X-MAP init segment recovered");
        assert_eq!(m.segments[0].uri, "seg0.m4s");
    }

    #[test]
    fn parses_byterange_segments_with_implicit_and_explicit_offsets() {
        // Single-file CMAF: init + three fragments are sub-ranges of one resource.
        // The first BYTERANGE gives an explicit offset; the rest continue from the
        // previous sub-range of the same URI (absent @offset).
        let text = "#EXTM3U\n\
            #EXT-X-TARGETDURATION:1\n\
            #EXT-X-MAP:URI=\"all.mp4\",BYTERANGE=\"800@0\"\n\
            #EXTINF:1.0,\n#EXT-X-BYTERANGE:200@800\nall.mp4\n\
            #EXTINF:1.0,\n#EXT-X-BYTERANGE:300\nall.mp4\n\
            #EXTINF:1.0,\n#EXT-X-BYTERANGE:150\nall.mp4\n\
            #EXT-X-ENDLIST\n";
        let Playlist::Media(m) = parse(text).unwrap() else {
            panic!("expected media playlist");
        };
        assert_eq!(m.map_uri.as_deref(), Some("all.mp4"));
        assert_eq!(m.map_byte_range, Some(ByteRange { offset: 0, length: 800 }));
        assert_eq!(m.segments.len(), 3);
        assert_eq!(m.segments[0].byte_range, Some(ByteRange { offset: 800, length: 200 }));
        // Implicit offset continues after the previous sub-range (800+200=1000).
        assert_eq!(m.segments[1].byte_range, Some(ByteRange { offset: 1000, length: 300 }));
        // And again (1000+300=1300).
        assert_eq!(m.segments[2].byte_range, Some(ByteRange { offset: 1300, length: 150 }));
    }

    #[test]
    fn ts_media_playlist_has_no_init_map() {
        let text = "#EXTM3U\n#EXTINF:4.0,\nseg0.ts\n#EXT-X-ENDLIST\n";
        let Playlist::Media(m) = parse(text).unwrap() else {
            panic!("expected media playlist");
        };
        assert_eq!(m.map_uri, None);
    }

    #[test]
    fn parses_master_and_selects_by_bandwidth() {
        let text = "#EXTM3U\n\
            #EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360,CODECS=\"avc1.4d401e,mp4a.40.2\"\n\
            low.m3u8\n\
            #EXT-X-STREAM-INF:BANDWIDTH=2400000,RESOLUTION=1280x720\n\
            high.m3u8\n";
        let Playlist::Master(master) = parse(text).unwrap() else {
            panic!("expected master playlist");
        };
        assert_eq!(master.variants.len(), 2);
        assert_eq!(master.variants[0].resolution, Some((640, 360)));
        assert_eq!(master.variants[0].codecs.as_deref(), Some("avc1.4d401e,mp4a.40.2"));

        // No cap: highest bandwidth wins.
        assert_eq!(master.select(None).unwrap().uri, "high.m3u8");
        // Cap below the high variant: the low one is chosen.
        assert_eq!(master.select(Some(1_000_000)).unwrap().uri, "low.m3u8");
        // Cap below everything: falls back to the lowest.
        assert_eq!(master.select(Some(1)).unwrap().uri, "low.m3u8");
    }

    #[test]
    fn ext_x_key_applies_to_following_segments_until_changed() {
        let text = "#EXTM3U\n\
            #EXT-X-KEY:METHOD=AES-128,URI=\"k1.key\",IV=0x00000000000000000000000000000001\n\
            #EXTINF:4.0,\n\
            seg0.ts\n\
            #EXTINF:4.0,\n\
            seg1.ts\n\
            #EXT-X-KEY:METHOD=NONE\n\
            #EXTINF:4.0,\n\
            seg2.ts\n\
            #EXT-X-KEY:METHOD=AES-128,URI=\"k2.key\"\n\
            #EXTINF:4.0,\n\
            seg3.ts\n\
            #EXT-X-ENDLIST\n";
        let Playlist::Media(m) = parse(text).unwrap() else {
            panic!("expected media playlist");
        };
        let mut iv1 = [0u8; 16];
        iv1[15] = 1;
        assert_eq!(
            m.segments[0].key,
            Some(SegmentKey { method: KeyMethod::Aes128, uri: "k1.key".into(), iv: Some(iv1) }),
        );
        // The key carries forward to the next segment unchanged.
        assert_eq!(m.segments[1].key, m.segments[0].key);
        // METHOD=NONE clears it.
        assert_eq!(m.segments[2].key, None);
        // A new key with no IV defaults to a sequence-derived IV (resolved later).
        assert_eq!(
            m.segments[3].key,
            Some(SegmentKey { method: KeyMethod::Aes128, uri: "k2.key".into(), iv: None }),
        );
    }

    #[test]
    fn rejects_non_playlist() {
        assert_eq!(parse("not a playlist\n"), Err(HlsError::NotAPlaylist));
    }

    #[test]
    fn flags_dangling_segment_tag() {
        let text = "#EXTM3U\n#EXTINF:5.0,\n";
        assert_eq!(parse(text), Err(HlsError::DanglingTag));
    }
}

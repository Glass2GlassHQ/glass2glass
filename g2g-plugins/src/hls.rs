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

/// One media segment in a media playlist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub uri: String,
    /// Segment duration in milliseconds (from `#EXTINF`, seconds * 1000).
    pub duration_ms: u32,
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
    let mut end_list = false;
    // A tag carries over to the next URI line: Some(duration_ms) for a segment,
    // or the variant being built for a stream-inf.
    let mut pending_segment: Option<u32> = None;
    let mut pending_variant: Option<Variant> = None;

    for line in lines {
        if let Some(attrs) = line.strip_prefix("#EXT-X-STREAM-INF:") {
            pending_variant = Some(parse_stream_inf(attrs));
        } else if let Some(rest) = line.strip_prefix("#EXTINF:") {
            pending_segment = Some(parse_extinf_ms(rest));
        } else if let Some(rest) = line.strip_prefix("#EXT-X-TARGETDURATION:") {
            target_duration_secs = rest.trim().parse().unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("#EXT-X-MEDIA-SEQUENCE:") {
            media_sequence = rest.trim().parse().unwrap_or(0);
        } else if let Some(attrs) = line.strip_prefix("#EXT-X-MAP:") {
            map_uri = attr_pairs(attrs)
                .into_iter()
                .find(|(k, _)| *k == "URI")
                .map(|(_, v)| String::from(v.trim_matches('"')));
        } else if line == "#EXT-X-ENDLIST" {
            end_list = true;
        } else if line.starts_with('#') {
            // any other tag / comment: ignored
        } else if let Some(mut variant) = pending_variant.take() {
            variant.uri = String::from(line);
            variants.push(variant);
        } else if let Some(duration_ms) = pending_segment.take() {
            segments.push(Segment { uri: String::from(line), duration_ms });
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
        assert_eq!(m.segments[0], Segment { uri: "seg0.ts".into(), duration_ms: 9009 });
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
    fn rejects_non_playlist() {
        assert_eq!(parse("not a playlist\n"), Err(HlsError::NotAPlaylist));
    }

    #[test]
    fn flags_dangling_segment_tag() {
        let text = "#EXTM3U\n#EXTINF:5.0,\n";
        assert_eq!(parse(text), Err(HlsError::DanglingTag));
    }
}

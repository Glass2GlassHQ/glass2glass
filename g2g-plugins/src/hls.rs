//! HLS playlist parser (RFC 8216), pure `no_std + alloc`. Parses the two
//! `.m3u8` forms: a *master* playlist (a set of `#EXT-X-STREAM-INF` variant
//! streams for ABR selection) and a *media* playlist (an ordered list of
//! `#EXTINF` segments). [`HlsSrc`](crate::hlssrc) drives this; the parser does no
//! I/O so it is fully unit-testable.
//!
//! A playlist is one form or the other: presence of any `#EXT-X-STREAM-INF`
//! makes it a master, otherwise it is a media playlist. URIs are kept verbatim
//! (absolute or relative); the caller resolves them against the playlist URL.
//!
//! [`write_media`] is the other direction, rendering a [`MediaPlaylist`] back to
//! `.m3u8` text for [`HlsSink`](crate::hlssink) to publish.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

/// One variant stream in a master playlist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variant {
    pub bandwidth: u64,
    pub resolution: Option<(u32, u32)>,
    pub codecs: Option<String>,
    pub uri: String,
    /// `AUDIO` group-id this variant binds (its alternate-audio renditions live in
    /// [`MasterPlaylist::renditions`] under this group), `None` when audio is
    /// multiplexed into the variant's own segments.
    pub audio_group: Option<String>,
    /// `SUBTITLES` group-id this variant binds, `None` when it carries none.
    pub subtitles_group: Option<String>,
    /// `VIDEO` group-id this variant binds (alternate video renditions), `None`
    /// for the common single-video case.
    pub video_group: Option<String>,
}

impl Variant {
    /// The `CODECS` attribute split into individual RFC 6381 codec strings
    /// (`avc1.4d401e`, `mp4a.40.2`, ...), empty when the attribute is absent.
    pub fn codec_list(&self) -> Vec<&str> {
        match &self.codecs {
            Some(c) => c
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect(),
            None => Vec::new(),
        }
    }
}

/// The kind of an alternate rendition (`#EXT-X-MEDIA:TYPE=`). `ClosedCaptions`
/// renditions carry no URI (they ride in the video stream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    Audio,
    Video,
    Subtitles,
    ClosedCaptions,
}

/// One alternate rendition declared by `#EXT-X-MEDIA` (RFC 8216 §4.3.4.1): a
/// named, grouped audio / video / subtitle track a variant can bind via its
/// `AUDIO` / `VIDEO` / `SUBTITLES` attribute. The discovery unit behind HLS
/// rendition selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendition {
    pub media_type: MediaType,
    pub group_id: String,
    pub name: String,
    /// `LANGUAGE` (RFC 5646 tag), `None` when the tag omits it.
    pub language: Option<String>,
    /// Rendition playlist URI; `None` for a rendition multiplexed into the
    /// variant's segments (allowed for `AUDIO`/`VIDEO`) or for `CLOSED-CAPTIONS`.
    pub uri: Option<String>,
    /// `DEFAULT=YES`: play this rendition absent an explicit choice.
    pub default: bool,
    /// `AUTOSELECT=YES`: eligible for automatic selection (e.g. by language).
    pub autoselect: bool,
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
    /// The key identifier this key decrypts, from the de-facto `KEYID=0x<32 hex>`
    /// attribute packagers emit for CENC content (RFC 8216 defines no such
    /// attribute; hls.js reads the same one). It binds the playlist key to the
    /// `tenc` / `seig` KID an fMP4 segment names, which is what lets a re-keyed
    /// stream pick the right key per sample.
    pub key_id: Option<[u8; 16]>,
}

/// A byte sub-range of a resource (`#EXT-X-BYTERANGE` / `#EXT-X-MAP:BYTERANGE`):
/// `length` bytes starting at `offset`. Single-file CMAF/fMP4 packs the init
/// segment and every media fragment into one resource addressed by range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    pub offset: u64,
    pub length: u64,
}

/// One partial segment (`#EXT-X-PART`, RFC 8216bis §4.4.4.9): a fraction of a
/// media segment a low-latency server publishes before the whole segment is
/// ready. Its position in [`Segment::parts`] is the Part Index the
/// `_HLS_part` delivery directive names, so a malformed tag fails the parse
/// rather than shifting every index after it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    pub uri: String,
    /// `DURATION` in milliseconds.
    pub duration_ms: u32,
    /// `BYTERANGE` sub-range of `uri`, `None` for a whole-resource part.
    pub byte_range: Option<ByteRange>,
    /// `INDEPENDENT=YES`: the part starts with an independently decodable frame,
    /// so playback can join on it.
    pub independent: bool,
    /// `GAP=YES`: the part is missing and its URI must not be fetched.
    pub gap: bool,
}

/// One media segment in a media playlist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// Empty for the still-being-produced segment at the end of a low-latency
    /// playlist: its `#EXT-X-PART`s are published but its `#EXTINF` and URI are
    /// not, so only [`parts`](Self::parts) can be fetched.
    pub uri: String,
    /// Segment duration in milliseconds (from `#EXTINF`, seconds * 1000), or the
    /// summed part durations for a segment still being produced.
    pub duration_ms: u32,
    /// Decryption context from the `#EXT-X-KEY` in effect, `None` if unencrypted.
    pub key: Option<SegmentKey>,
    /// `#EXT-X-BYTERANGE` sub-range of `uri`, `None` for a whole-resource segment.
    pub byte_range: Option<ByteRange>,
    /// `#EXT-X-GAP`: the segment is missing and its URI must not be fetched
    /// (RFC 8216bis §4.4.4.7). A live packager pads a freshly started playlist
    /// with these, so a client that loads them gets a 404 instead of a stream.
    pub gap: bool,
    /// `#EXT-X-PART` partial segments, in Part Index order. Empty on a plain
    /// playlist; a low-latency one carries them for the segments near the live
    /// edge.
    pub parts: Vec<Part>,
}

impl Segment {
    /// The segment is still being produced: only its published parts exist.
    pub fn incomplete(&self) -> bool {
        self.uri.is_empty()
    }
}

/// `#EXT-X-SERVER-CONTROL` (RFC 8216bis §4.4.3.8): the delivery directives the
/// server accepts, which is what tells a client low-latency playback is on offer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerControl {
    /// `CAN-BLOCK-RELOAD=YES`: a playlist request carrying `_HLS_msn` /
    /// `_HLS_part` is held until that part is published, replacing timed polling.
    pub can_block_reload: bool,
    /// `PART-HOLD-BACK` in ms: how far behind the live edge a client should start.
    pub part_hold_back_ms: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MasterPlaylist {
    pub variants: Vec<Variant>,
    /// Alternate renditions from `#EXT-X-MEDIA` (alternate audio, subtitles, ...),
    /// grouped by `group_id`; a variant binds a group via its `audio_group` /
    /// `subtitles_group` / `video_group`.
    pub renditions: Vec<Rendition>,
}

impl MasterPlaylist {
    /// The renditions in `group_id` of the given `media_type` (e.g. the alternate
    /// audio tracks a variant's `AUDIO` group offers), in declaration order.
    pub fn renditions_in(&self, media_type: MediaType, group_id: &str) -> Vec<&Rendition> {
        self.renditions
            .iter()
            .filter(|r| r.media_type == media_type && r.group_id == group_id)
            .collect()
    }
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
    /// `#EXT-X-PART-INF:PART-TARGET` in ms, present only on a playlist that
    /// publishes partial segments.
    pub part_target_ms: Option<u32>,
    pub server_control: Option<ServerControl>,
}

/// Three part targets is the `PART-HOLD-BACK` a client assumes when the server
/// declares none (RFC 8216bis §4.4.3.8 sets that as the minimum).
const PART_HOLD_BACK_TARGETS: u32 = 3;

impl MediaPlaylist {
    /// Whether the playlist offers low-latency delivery: partial segments plus a
    /// server that holds a playlist request until the next part is published.
    /// Both halves are needed, since parts a client can only poll for on the
    /// `TARGETDURATION` timer arrive no sooner than whole segments.
    pub fn low_latency(&self) -> bool {
        self.part_target_ms.is_some()
            && self
                .server_control
                .is_some_and(|control| control.can_block_reload)
    }

    /// How far behind the live edge to start playing: `PART-HOLD-BACK`, else
    /// three part targets. `0` when the playlist publishes no parts.
    pub fn part_hold_back_ms(&self) -> u32 {
        let default = self
            .part_target_ms
            .map_or(0, |t| t.saturating_mul(PART_HOLD_BACK_TARGETS));
        self.server_control
            .and_then(|control| control.part_hold_back_ms)
            .unwrap_or(default)
    }
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
    /// An `#EXT-X-PART` missing its `DURATION` or `URI`. Part Indices are
    /// positional (the `_HLS_part` directive names one), so a part that cannot
    /// be read fails the playlist instead of shifting the ones after it.
    MalformedPart,
}

impl MasterPlaylist {
    /// Pick the highest-bandwidth variant at or below `max_bandwidth` (or the
    /// overall highest when `None` / nothing fits). The simplest ABR rule; a
    /// real rate adaptor would track throughput across segments.
    pub fn select(&self, max_bandwidth: Option<u64>) -> Option<&Variant> {
        let under = |v: &&Variant| max_bandwidth.is_none_or(|cap| v.bandwidth <= cap);
        self.variants
            .iter()
            .filter(under)
            .max_by_key(|v| v.bandwidth)
            .or_else(|| self.variants.iter().min_by_key(|v| v.bandwidth))
    }

    /// Pick a rendition from `group_id` of `media_type` by `language` preference
    /// (M418): a `LANGUAGE` match wins (exact case-insensitive, then a primary-tag
    /// prefix so `en` matches `en-US`); absent a match or a preference, the
    /// `DEFAULT=YES` rendition, else the first in declaration order. `None` for an
    /// empty group. The `playbin` HLS fan-out uses this to honour a
    /// `#audio-lang=` / `#subtitle-lang=` URI hint when choosing the alternate
    /// audio / subtitle rendition.
    pub fn pick_rendition(
        &self,
        media_type: MediaType,
        group_id: &str,
        language: Option<&str>,
    ) -> Option<&Rendition> {
        let group = self.renditions_in(media_type, group_id);
        if let Some(pref) = language.map(str::trim).filter(|p| !p.is_empty()) {
            if let Some(r) = group.iter().find(|r| {
                r.language
                    .as_deref()
                    .is_some_and(|l| l.eq_ignore_ascii_case(pref))
            }) {
                return Some(r);
            }
            if let Some(r) = group.iter().find(|r| {
                r.language
                    .as_deref()
                    .is_some_and(|l| lang_prefix_match(l, pref))
            }) {
                return Some(r);
            }
        }
        group
            .iter()
            .find(|r| r.default)
            .or_else(|| group.first())
            .copied()
    }
}

/// True when `lang` is a regional refinement of the primary subtag `pref`, e.g.
/// `lang_prefix_match("en-US", "en")`. A bare equal is handled by the caller's
/// case-insensitive exact check, so this only matches the `pref-` prefix form.
fn lang_prefix_match(lang: &str, pref: &str) -> bool {
    let (lb, pb) = (lang.as_bytes(), pref.as_bytes());
    lb.len() > pb.len() && lb[pb.len()] == b'-' && lb[..pb.len()].eq_ignore_ascii_case(pb)
}

/// Parse a `.m3u8` playlist. Returns master or media form.
pub fn parse(text: &str) -> Result<Playlist, HlsError> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());

    match lines.next() {
        Some("#EXTM3U") => {}
        _ => return Err(HlsError::NotAPlaylist),
    }

    let mut variants = Vec::new();
    let mut renditions = Vec::new();
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
    // `#EXT-X-GAP` applies to the next segment.
    let mut pending_gap = false;
    let mut part_target_ms = None;
    let mut server_control = None;
    // `#EXT-X-PART`s seen since the last segment: they precede the `#EXTINF` of
    // the segment they belong to, and a run left over at the end is the segment
    // still being produced.
    let mut pending_parts: Vec<Part> = Vec::new();

    for line in lines {
        if let Some(attrs) = line.strip_prefix("#EXT-X-STREAM-INF:") {
            pending_variant = Some(parse_stream_inf(attrs));
        } else if let Some(attrs) = line.strip_prefix("#EXT-X-MEDIA:") {
            if let Some(r) = parse_media(attrs) {
                renditions.push(r);
            }
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
                    (length > 0).then_some(ByteRange {
                        offset: offset.unwrap_or(0),
                        length,
                    })
                });
        } else if let Some(attrs) = line.strip_prefix("#EXT-X-KEY:") {
            current_key = parse_key(attrs);
        } else if let Some(attrs) = line.strip_prefix("#EXT-X-PART:") {
            pending_parts.push(parse_part(attrs, &pending_parts).ok_or(HlsError::MalformedPart)?);
        } else if let Some(attrs) = line.strip_prefix("#EXT-X-PART-INF:") {
            part_target_ms = attr_pairs(attrs)
                .iter()
                .find(|(k, _)| *k == "PART-TARGET")
                .and_then(|(_, v)| parse_secs_ms(v));
        } else if let Some(attrs) = line.strip_prefix("#EXT-X-SERVER-CONTROL:") {
            let pairs = attr_pairs(attrs);
            let find = |name: &str| pairs.iter().find(|(k, _)| *k == name).map(|(_, v)| *v);
            server_control = Some(ServerControl {
                can_block_reload: find("CAN-BLOCK-RELOAD") == Some("YES"),
                part_hold_back_ms: find("PART-HOLD-BACK").and_then(parse_secs_ms),
            });
        } else if line == "#EXT-X-ENDLIST" {
            end_list = true;
        } else if line == "#EXT-X-GAP" {
            pending_gap = true;
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
            segments.push(Segment {
                uri,
                duration_ms,
                key: current_key.clone(),
                byte_range,
                gap: core::mem::take(&mut pending_gap),
                parts: core::mem::take(&mut pending_parts),
            });
        }
        // a bare URI with no pending tag is ignored
    }

    if pending_variant.is_some() || pending_segment.is_some() {
        return Err(HlsError::DanglingTag);
    }

    // Parts with no `#EXTINF` after them are the segment the server is still
    // producing: it plays as a segment whose only fetchable pieces are its parts.
    if !pending_parts.is_empty() {
        let duration_ms = pending_parts
            .iter()
            .fold(0u32, |sum, p| sum.saturating_add(p.duration_ms));
        segments.push(Segment {
            uri: String::new(),
            duration_ms,
            key: current_key.clone(),
            byte_range: None,
            gap: pending_gap,
            parts: pending_parts,
        });
    }

    if !variants.is_empty() || !renditions.is_empty() {
        Ok(Playlist::Master(MasterPlaylist {
            variants,
            renditions,
        }))
    } else {
        Ok(Playlist::Media(MediaPlaylist {
            target_duration_secs,
            media_sequence,
            segments,
            map_uri,
            map_byte_range,
            end_list,
            part_target_ms,
            server_control,
        }))
    }
}

/// Parse an `#EXT-X-PART` attribute list. `DURATION` and `URI` are required
/// (`None` without them, which the caller turns into a parse failure);
/// `prev_parts` resolves a `BYTERANGE` that omits its `@offset`, which continues
/// from the previous part of the same resource.
fn parse_part(attrs: &str, prev_parts: &[Part]) -> Option<Part> {
    let pairs = attr_pairs(attrs);
    let find = |name: &str| pairs.iter().find(|(k, _)| *k == name).map(|(_, v)| *v);
    let duration_ms = parse_secs_ms(find("DURATION")?)?;
    let uri = String::from(find("URI")?.trim_matches('"'));
    let byte_range = find("BYTERANGE").and_then(|v| {
        let (length, offset) = parse_byterange(v.trim_matches('"'));
        let offset = offset.unwrap_or_else(|| {
            prev_parts
                .iter()
                .rev()
                .find(|p| p.uri == uri)
                .and_then(|p| p.byte_range)
                .map_or(0, |r| r.offset.saturating_add(r.length))
        });
        (length > 0).then_some(ByteRange { offset, length })
    });
    Some(Part {
        uri,
        duration_ms,
        byte_range,
        independent: find("INDEPENDENT") == Some("YES"),
        gap: find("GAP") == Some("YES"),
    })
}

/// Parse an `#EXT-X-STREAM-INF` attribute list. Unknown attributes are ignored;
/// the URI is filled in by the caller from the following line.
fn parse_stream_inf(attrs: &str) -> Variant {
    let mut bandwidth = 0u64;
    let mut resolution = None;
    let mut codecs = None;
    let mut audio_group = None;
    let mut subtitles_group = None;
    let mut video_group = None;
    let unquote = |v: &str| String::from(v.trim_matches('"'));
    for (key, value) in attr_pairs(attrs) {
        match key {
            "BANDWIDTH" => bandwidth = value.parse().unwrap_or(0),
            "RESOLUTION" => resolution = parse_resolution(value),
            "CODECS" => codecs = Some(unquote(value)),
            "AUDIO" => audio_group = Some(unquote(value)),
            "SUBTITLES" => subtitles_group = Some(unquote(value)),
            "VIDEO" => video_group = Some(unquote(value)),
            _ => {}
        }
    }
    Variant {
        bandwidth,
        resolution,
        codecs,
        uri: String::new(),
        audio_group,
        subtitles_group,
        video_group,
    }
}

/// Parse an `#EXT-X-MEDIA` attribute list into a [`Rendition`]. Requires `TYPE`,
/// `GROUP-ID`, and `NAME` (RFC 8216 §4.3.4.1); an unknown / missing `TYPE` or a
/// missing required field yields `None` (the tag is skipped, not a parse error).
fn parse_media(attrs: &str) -> Option<Rendition> {
    let pairs = attr_pairs(attrs);
    let find = |name: &str| pairs.iter().find(|(k, _)| *k == name).map(|(_, v)| *v);
    let unquote = |v: &str| String::from(v.trim_matches('"'));
    let media_type = match find("TYPE")? {
        "AUDIO" => MediaType::Audio,
        "VIDEO" => MediaType::Video,
        "SUBTITLES" => MediaType::Subtitles,
        "CLOSED-CAPTIONS" => MediaType::ClosedCaptions,
        _ => return None,
    };
    Some(Rendition {
        media_type,
        group_id: unquote(find("GROUP-ID")?),
        name: unquote(find("NAME")?),
        language: find("LANGUAGE").map(unquote),
        uri: find("URI").map(unquote),
        default: find("DEFAULT") == Some("YES"),
        autoselect: find("AUTOSELECT") == Some("YES"),
    })
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
    let key_id = find("KEYID").and_then(parse_iv);
    Some(SegmentKey {
        method,
        uri,
        iv,
        key_id,
    })
}

/// `IV=0x<32 hex digits>` -> 16 bytes. Anything else is rejected.
fn parse_iv(value: &str) -> Option<[u8; 16]> {
    let hex = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))?;
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
    parse_secs_ms(rest.split(',').next().unwrap_or("")).unwrap_or(0)
}

/// A fractional-seconds attribute (`#EXTINF`, `PART-TARGET`, `PART-HOLD-BACK`,
/// `DURATION`) -> milliseconds. `None` for anything not a finite non-negative
/// number, and a value beyond `u32::MAX` ms saturates rather than wrapping. The
/// `+ 0.5` rounds: `9.009` is not exactly representable, and truncating it would
/// lose a millisecond off every segment duration (`round` is std-only).
fn parse_secs_ms(value: &str) -> Option<u32> {
    match value.trim().parse::<f64>() {
        Ok(s) if s.is_finite() && s >= 0.0 => Some((s * 1000.0 + 0.5).min(u32::MAX as f64) as u32),
        _ => None,
    }
}

/// Render a media playlist as `.m3u8` text, the inverse of [`parse`]: the tags a
/// player needs to fetch the segments, in RFC 8216 order. `#EXT-X-VERSION` is
/// derived from the features used (6 for an `#EXT-X-MAP`, 4 for byte ranges,
/// else 3 for the fractional `#EXTINF` durations), and `#EXT-X-KEY` is written
/// only where the key in effect changes, including a `METHOD=NONE` to clear it.
pub fn write_media(playlist: &MediaPlaylist) -> String {
    let mut out = String::from("#EXTM3U\n");
    let version = if playlist.map_uri.is_some() {
        6
    } else if playlist.segments.iter().any(|s| s.byte_range.is_some()) {
        4
    } else {
        3
    };
    let _ = writeln!(out, "#EXT-X-VERSION:{version}");
    let _ = writeln!(
        out,
        "#EXT-X-TARGETDURATION:{}",
        playlist.target_duration_secs
    );
    let _ = writeln!(out, "#EXT-X-MEDIA-SEQUENCE:{}", playlist.media_sequence);
    if let Some(uri) = &playlist.map_uri {
        let _ = write!(out, "#EXT-X-MAP:URI=\"{uri}\"");
        if let Some(r) = playlist.map_byte_range {
            let _ = write!(out, ",BYTERANGE=\"{}@{}\"", r.length, r.offset);
        }
        out.push('\n');
    }
    let mut key_in_effect: Option<&SegmentKey> = None;
    for segment in &playlist.segments {
        if segment.key.as_ref() != key_in_effect {
            out.push_str(&key_tag(segment.key.as_ref()));
            key_in_effect = segment.key.as_ref();
        }
        let (secs, millis) = (segment.duration_ms / 1000, segment.duration_ms % 1000);
        let _ = writeln!(out, "#EXTINF:{secs}.{millis:03},");
        if let Some(r) = segment.byte_range {
            let _ = writeln!(out, "#EXT-X-BYTERANGE:{}@{}", r.length, r.offset);
        }
        let _ = writeln!(out, "{}", segment.uri);
    }
    if playlist.end_list {
        out.push_str("#EXT-X-ENDLIST\n");
    }
    out
}

/// The `#EXT-X-KEY` line putting `key` in effect (`METHOD=NONE` for `None`).
fn key_tag(key: Option<&SegmentKey>) -> String {
    let Some(key) = key else {
        return String::from("#EXT-X-KEY:METHOD=NONE\n");
    };
    let method = match key.method {
        KeyMethod::Aes128 => "AES-128",
        KeyMethod::SampleAes => "SAMPLE-AES",
    };
    let mut line = String::new();
    let _ = write!(line, "#EXT-X-KEY:METHOD={method},URI=\"{}\"", key.uri);
    if let Some(iv) = key.iv {
        let _ = write!(line, ",IV={}", hex16(&iv));
    }
    if let Some(kid) = key.key_id {
        let _ = write!(line, ",KEYID={}", hex16(&kid));
    }
    line.push('\n');
    line
}

/// 16 bytes as the `0x<32 hex digits>` form [`parse_iv`] reads back.
fn hex16(bytes: &[u8; 16]) -> String {
    let mut s = String::from("0x");
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
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
            Segment {
                uri: "seg0.ts".into(),
                duration_ms: 9009,
                key: None,
                byte_range: None,
                gap: false,
                parts: Vec::new(),
            }
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
        assert_eq!(
            m.map_uri.as_deref(),
            Some("init.mp4"),
            "EXT-X-MAP init segment recovered"
        );
        assert_eq!(m.segments[0].uri, "seg0.m4s");
    }

    /// A live packager pads a freshly started playlist with `#EXT-X-GAP`
    /// placeholders whose URI it never wrote (mediamtx names them all
    /// `gap.mp4`). The flag has to survive the parse or `HlsSrc` fetches them and
    /// takes a 404.
    #[test]
    fn marks_ext_x_gap_segments() {
        let text = "#EXTM3U\n\
            #EXT-X-TARGETDURATION:1\n\
            #EXT-X-MEDIA-SEQUENCE:1\n\
            #EXT-X-GAP\n\
            #EXTINF:1.0,\n\
            gap.mp4\n\
            #EXTINF:1.0,\n\
            seg2.mp4\n";
        let Playlist::Media(m) = parse(text).unwrap() else {
            panic!("expected media playlist");
        };
        assert!(m.segments[0].gap, "the tagged segment is a gap");
        assert!(
            !m.segments[1].gap,
            "the tag applies to one segment, not the rest of the playlist"
        );
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
        assert_eq!(
            m.map_byte_range,
            Some(ByteRange {
                offset: 0,
                length: 800
            })
        );
        assert_eq!(m.segments.len(), 3);
        assert_eq!(
            m.segments[0].byte_range,
            Some(ByteRange {
                offset: 800,
                length: 200
            })
        );
        // Implicit offset continues after the previous sub-range (800+200=1000).
        assert_eq!(
            m.segments[1].byte_range,
            Some(ByteRange {
                offset: 1000,
                length: 300
            })
        );
        // And again (1000+300=1300).
        assert_eq!(
            m.segments[2].byte_range,
            Some(ByteRange {
                offset: 1300,
                length: 150
            })
        );
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
        assert_eq!(
            master.variants[0].codecs.as_deref(),
            Some("avc1.4d401e,mp4a.40.2")
        );

        // No cap: highest bandwidth wins.
        assert_eq!(master.select(None).unwrap().uri, "high.m3u8");
        // Cap below the high variant: the low one is chosen.
        assert_eq!(master.select(Some(1_000_000)).unwrap().uri, "low.m3u8");
        // Cap below everything: falls back to the lowest.
        assert_eq!(master.select(Some(1)).unwrap().uri, "low.m3u8");
    }

    #[test]
    fn parses_master_renditions_and_variant_group_links() {
        // A master with two alternate audio renditions (an AUDIO group) and a
        // subtitle rendition, plus a variant binding both groups.
        let text = "#EXTM3U\n\
            #EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"aac\",NAME=\"English\",LANGUAGE=\"en\",DEFAULT=YES,AUTOSELECT=YES,URI=\"audio/en.m3u8\"\n\
            #EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"aac\",NAME=\"Français\",LANGUAGE=\"fr\",AUTOSELECT=YES,URI=\"audio/fr.m3u8\"\n\
            #EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID=\"subs\",NAME=\"English\",LANGUAGE=\"en\",URI=\"subs/en.m3u8\"\n\
            #EXT-X-STREAM-INF:BANDWIDTH=2400000,RESOLUTION=1280x720,CODECS=\"avc1.4d401e,mp4a.40.2\",AUDIO=\"aac\",SUBTITLES=\"subs\"\n\
            video/720p.m3u8\n";
        let Playlist::Master(master) = parse(text).unwrap() else {
            panic!("expected master playlist");
        };
        assert_eq!(master.variants.len(), 1);
        let v = &master.variants[0];
        assert_eq!(v.audio_group.as_deref(), Some("aac"));
        assert_eq!(v.subtitles_group.as_deref(), Some("subs"));
        assert_eq!(v.codec_list(), alloc::vec!["avc1.4d401e", "mp4a.40.2"]);

        // The variant's audio group offers two alternate renditions, in order.
        let audio = master.renditions_in(MediaType::Audio, "aac");
        assert_eq!(audio.len(), 2);
        assert_eq!(audio[0].name, "English");
        assert_eq!(audio[0].language.as_deref(), Some("en"));
        assert!(audio[0].default, "first audio rendition is DEFAULT=YES");
        assert_eq!(audio[0].uri.as_deref(), Some("audio/en.m3u8"));
        assert_eq!(audio[1].language.as_deref(), Some("fr"));
        assert!(!audio[1].default);

        let subs = master.renditions_in(MediaType::Subtitles, "subs");
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].uri.as_deref(), Some("subs/en.m3u8"));
    }

    #[test]
    fn pick_rendition_honours_language_then_default() {
        let text = "#EXTM3U\n\
            #EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"a\",NAME=\"English\",LANGUAGE=\"en\",DEFAULT=YES,URI=\"a/en.m3u8\"\n\
            #EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"a\",NAME=\"Français\",LANGUAGE=\"fr-FR\",URI=\"a/fr.m3u8\"\n\
            #EXT-X-STREAM-INF:BANDWIDTH=1,AUDIO=\"a\"\nv.m3u8\n";
        let Playlist::Master(m) = parse(text).unwrap() else {
            panic!("master")
        };

        // Exact language (case-insensitive).
        assert_eq!(
            m.pick_rendition(MediaType::Audio, "a", Some("FR-fr"))
                .unwrap()
                .name,
            "Français"
        );
        // Primary-subtag prefix: `fr` matches `fr-FR`.
        assert_eq!(
            m.pick_rendition(MediaType::Audio, "a", Some("fr"))
                .unwrap()
                .name,
            "Français"
        );
        // No preference -> the DEFAULT=YES rendition.
        assert_eq!(
            m.pick_rendition(MediaType::Audio, "a", None).unwrap().name,
            "English"
        );
        // An unknown language falls back to DEFAULT, not nothing.
        assert_eq!(
            m.pick_rendition(MediaType::Audio, "a", Some("de"))
                .unwrap()
                .name,
            "English"
        );
        // An empty group yields None.
        assert!(m
            .pick_rendition(MediaType::Audio, "nope", Some("en"))
            .is_none());
    }

    #[test]
    fn media_only_renditions_still_form_a_master() {
        // A playlist with #EXT-X-MEDIA but no #EXT-X-STREAM-INF is still a master
        // (renditions present), not mis-parsed as an empty media playlist.
        let text = "#EXTM3U\n\
            #EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"a\",NAME=\"main\",URI=\"a.m3u8\"\n";
        let Playlist::Master(master) = parse(text).unwrap() else {
            panic!("expected master playlist");
        };
        assert!(master.variants.is_empty());
        assert_eq!(master.renditions.len(), 1);
        assert_eq!(master.renditions[0].media_type, MediaType::Audio);
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
            Some(SegmentKey {
                method: KeyMethod::Aes128,
                uri: "k1.key".into(),
                iv: Some(iv1),
                key_id: None
            }),
        );
        // The key carries forward to the next segment unchanged.
        assert_eq!(m.segments[1].key, m.segments[0].key);
        // METHOD=NONE clears it.
        assert_eq!(m.segments[2].key, None);
        // A new key with no IV defaults to a sequence-derived IV (resolved later).
        assert_eq!(
            m.segments[3].key,
            Some(SegmentKey {
                method: KeyMethod::Aes128,
                uri: "k2.key".into(),
                iv: None,
                key_id: None
            }),
        );
    }

    #[test]
    fn written_media_playlist_parses_back_identically() {
        // Every media-playlist feature the writer emits: an fMP4 init map, a
        // rotating key, byte ranges, and a closing ENDLIST.
        let mut iv = [0u8; 16];
        iv[15] = 0x2a;
        let key = SegmentKey {
            method: KeyMethod::Aes128,
            uri: "k.key".into(),
            iv: Some(iv),
            key_id: None,
        };
        let original = MediaPlaylist {
            target_duration_secs: 4,
            media_sequence: 12,
            segments: alloc::vec![
                Segment {
                    uri: "seg0.m4s".into(),
                    duration_ms: 4000,
                    key: None,
                    byte_range: None,
                    gap: false,
                    parts: Vec::new(),
                },
                Segment {
                    uri: "seg1.m4s".into(),
                    duration_ms: 3960,
                    key: Some(key.clone()),
                    byte_range: Some(ByteRange {
                        offset: 900,
                        length: 250,
                    }),
                    gap: false,
                    parts: Vec::new(),
                },
                Segment {
                    uri: "seg2.m4s".into(),
                    duration_ms: 120,
                    key: Some(key),
                    byte_range: None,
                    gap: false,
                    parts: Vec::new(),
                },
            ],
            map_uri: Some("init.mp4".into()),
            map_byte_range: None,
            end_list: true,
            part_target_ms: None,
            server_control: None,
        };
        let text = write_media(&original);
        assert!(
            text.starts_with("#EXTM3U\n#EXT-X-VERSION:6\n"),
            "an EXT-X-MAP playlist declares version 6: {text}"
        );
        assert!(
            text.contains("#EXTINF:3.960,\n"),
            "fractional EXTINF: {text}"
        );
        let Playlist::Media(back) = parse(&text).unwrap() else {
            panic!("expected media playlist");
        };
        assert_eq!(back, original);
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

    /// The low-latency shape a real LL-HLS packager (mediamtx) publishes: parts
    /// precede the `#EXTINF` of the segment they belong to, and the run of parts
    /// left at the end is the segment still being produced.
    const PART_SECS: f64 = 0.2;
    const SEGMENT_SECS: f64 = 1.0;
    const HOLD_BACK_SECS: f64 = 0.5;

    fn ll_playlist(hold_back: Option<f64>) -> String {
        let control = match hold_back {
            Some(h) => alloc::format!("CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK={h:.5}"),
            None => String::from("CAN-BLOCK-RELOAD=YES"),
        };
        alloc::format!(
            "#EXTM3U\n\
             #EXT-X-VERSION:10\n\
             #EXT-X-TARGETDURATION:1\n\
             #EXT-X-SERVER-CONTROL:{control},CAN-SKIP-UNTIL=6.00000\n\
             #EXT-X-PART-INF:PART-TARGET={PART_SECS:.5}\n\
             #EXT-X-MEDIA-SEQUENCE:79\n\
             #EXT-X-MAP:URI=\"init.mp4\"\n\
             #EXTINF:{SEGMENT_SECS:.5},\n\
             seg79.mp4\n\
             #EXT-X-PART:DURATION={PART_SECS:.5},URI=\"part400.mp4\",INDEPENDENT=YES\n\
             #EXT-X-PART:DURATION={PART_SECS:.5},URI=\"part401.mp4\"\n\
             #EXTINF:{SEGMENT_SECS:.5},\n\
             seg80.mp4\n\
             #EXT-X-PART:DURATION={PART_SECS:.5},URI=\"part402.mp4\",INDEPENDENT=YES\n\
             #EXT-X-PART:DURATION={PART_SECS:.5},URI=\"part403.mp4\",GAP=YES\n\
             #EXT-X-PRELOAD-HINT:TYPE=PART,URI=\"part404.mp4\"\n"
        )
    }

    #[test]
    fn parses_low_latency_tags_and_trailing_partial_segment() {
        let part_ms = (PART_SECS * 1000.0) as u32;
        let Playlist::Media(m) = parse(&ll_playlist(Some(HOLD_BACK_SECS))).unwrap() else {
            panic!("expected media playlist");
        };
        assert_eq!(
            m.part_target_ms,
            Some(part_ms),
            "EXT-X-PART-INF PART-TARGET"
        );
        assert_eq!(
            m.server_control,
            Some(ServerControl {
                can_block_reload: true,
                part_hold_back_ms: Some((HOLD_BACK_SECS * 1000.0) as u32),
            })
        );
        assert!(m.low_latency(), "parts plus a blocking-reload server");
        assert_eq!(m.part_hold_back_ms(), (HOLD_BACK_SECS * 1000.0) as u32);

        // Three segments: two complete, then the one being produced.
        assert_eq!(m.segments.len(), 3);
        assert!(
            m.segments[0].parts.is_empty(),
            "an older segment lost its parts"
        );
        assert!(!m.segments[0].incomplete());

        // The parts before an #EXTINF belong to that segment, not the previous one.
        let seg80 = &m.segments[1];
        assert_eq!(seg80.uri, "seg80.mp4");
        assert_eq!(seg80.parts.len(), 2);
        assert_eq!(seg80.parts[0].uri, "part400.mp4");
        assert_eq!(seg80.parts[0].duration_ms, part_ms);
        assert!(seg80.parts[0].independent);
        assert!(!seg80.parts[1].independent);

        // The trailing run of parts is the segment still being produced: no URI,
        // duration summed from its parts, and the GAP part flagged.
        let open = &m.segments[2];
        assert!(open.incomplete(), "no #EXTINF yet, so only its parts exist");
        assert_eq!(open.parts.len(), 2);
        assert_eq!(open.duration_ms, part_ms * 2);
        assert!(!open.parts[0].gap);
        assert!(open.parts[1].gap, "GAP=YES part must not be fetched");
    }

    #[test]
    fn part_hold_back_falls_back_to_three_part_targets() {
        let Playlist::Media(m) = parse(&ll_playlist(None)).unwrap() else {
            panic!("expected media playlist");
        };
        assert_eq!(m.server_control.unwrap().part_hold_back_ms, None);
        assert_eq!(m.part_hold_back_ms(), (PART_SECS * 1000.0) as u32 * 3);
    }

    #[test]
    fn plain_playlist_offers_no_low_latency() {
        let text = "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXTINF:4.0,\nseg0.ts\n";
        let Playlist::Media(m) = parse(text).unwrap() else {
            panic!("expected media playlist");
        };
        assert!(!m.low_latency());
        assert_eq!(m.part_target_ms, None);
        assert_eq!(m.server_control, None);
        assert_eq!(m.part_hold_back_ms(), 0);
        assert!(m.segments[0].parts.is_empty());
    }

    /// A server that declares SERVER-CONTROL without publishing parts (a plain
    /// playlist with delta updates) is not a low-latency one: blocking a reload
    /// on a part index that does not exist would hang.
    #[test]
    fn server_control_without_parts_is_not_low_latency() {
        let text = "#EXTM3U\n#EXT-X-TARGETDURATION:4\n\
                    #EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES\n\
                    #EXTINF:4.0,\nseg0.ts\n";
        let Playlist::Media(m) = parse(text).unwrap() else {
            panic!("expected media playlist");
        };
        assert!(!m.low_latency());
    }

    #[test]
    fn malformed_part_fails_the_parse() {
        // A Part Index is positional (the `_HLS_part` directive names it), so a
        // part that cannot be read has to fail the playlist, not be skipped.
        for attrs in [
            "URI=\"p.mp4\"",               // no DURATION
            "DURATION=0.2",                // no URI
            "DURATION=abc,URI=\"p.mp4\"",  // unparseable DURATION
            "DURATION=-0.2,URI=\"p.mp4\"", // negative DURATION
        ] {
            let text = alloc::format!("#EXTM3U\n#EXT-X-PART:{attrs}\n");
            assert_eq!(
                parse(&text),
                Err(HlsError::MalformedPart),
                "must reject #EXT-X-PART:{attrs}"
            );
        }
    }

    #[test]
    fn part_byterange_continues_from_the_previous_part_of_the_resource() {
        // Single-file CMAF: consecutive parts are sub-ranges of one resource, the
        // second omitting its @offset.
        let text = "#EXTM3U\n\
            #EXT-X-PART-INF:PART-TARGET=0.2\n\
            #EXT-X-PART:DURATION=0.2,URI=\"all.mp4\",BYTERANGE=\"200@800\"\n\
            #EXT-X-PART:DURATION=0.2,URI=\"all.mp4\",BYTERANGE=\"300\"\n";
        let Playlist::Media(m) = parse(text).unwrap() else {
            panic!("expected media playlist");
        };
        let parts = &m.segments[0].parts;
        assert_eq!(
            parts[0].byte_range,
            Some(ByteRange {
                offset: 800,
                length: 200
            })
        );
        assert_eq!(
            parts[1].byte_range,
            Some(ByteRange {
                offset: 1000,
                length: 300
            })
        );
    }
}

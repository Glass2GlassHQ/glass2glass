//! HLS source (HlsSrc, `hls` feature): fetches an `.m3u8` playlist, selects a
//! variant (simple bandwidth-capped ABR), and streams that variant's MPEG-TS
//! media segments downstream as a `Caps::ByteStream{MpegTs}` for `tsdemux`, then
//! `Eos`. The [`hls`](crate::hls) parser does the playlist work; this element
//! adds the fetching (via `reqwest`, like [`HttpSrc`](crate::httpsrc)) and URL
//! resolution.
//!
//! VOD (a playlist with `#EXT-X-ENDLIST`) plays its segments once then `Eos`.
//! Live (no ENDLIST) starts near the live edge (`live_edge_start`: ~3 target
//! durations from the end per RFC 8216 §6.3.3, so playback follows what is being
//! published rather than the stale front of the window), reloads the media
//! playlist on an interval, plays each new segment once (tracked by HLS
//! media-sequence), and ends when ENDLIST finally appears or downstream shuts down.
//!
//! `#EXT-X-KEY:METHOD=AES-128` segments are decrypted in place: the 16-byte key
//! is fetched from the key URI (cached per run) and each segment is AES-128-CBC
//! decrypted with the explicit `IV` or, absent one, the segment media-sequence
//! number as a 128-bit big-endian IV. For `METHOD=SAMPLE-AES` (per-sample, not
//! whole-segment) the fetched key/IV is published to a shared key store
//! ([`with_sample_aes_key_handle`](HlsSrc::with_sample_aes_key_handle)) for a
//! downstream [`SampleAesDecrypt`](crate::sampleaesdecrypt) (TS) or fMP4 demuxer
//! (CENC), and the bytes are forwarded undecrypted; without a handle a
//! SAMPLE-AES playlist is rejected. The init segment (`#EXT-X-MAP`) is assumed
//! unencrypted.
//!
//! A key is published against the byte offset at which its segment enters the
//! emitted stream, not when it is fetched, so a mid-playlist key rotation (a new
//! `#EXT-X-KEY`, including back-to-back rotations and ones that appear on a live
//! reload) takes effect at exactly that segment even though fetching runs ahead
//! of emission. A `KEYID` attribute additionally registers the key under its
//! CENC key identifier, which the fMP4 path matches against the `tenc` / `seig`
//! KID.
//!
//! Single-file CMAF is supported via `#EXT-X-BYTERANGE` (and `#EXT-X-MAP`'s
//! `BYTERANGE`): a segment that carries one fetches only its sub-range with an
//! HTTP `Range` request (M368), the offset continuing from the previous
//! sub-range of the same resource when the tag omits an explicit `@offset`.
//!
//! Low-latency HLS (RFC 8216bis) is followed whenever the playlist offers it:
//! `#EXT-X-PART` partial segments plus `#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD`.
//! The reload then holds the playlist request open with the `_HLS_msn` /
//! `_HLS_part` delivery directives instead of polling on the `TARGETDURATION`
//! timer, each part is fetched and emitted as its own `DataFrame` as it appears,
//! and playback starts `PART-HOLD-BACK` behind the live edge rather than three
//! target durations. `low-latency=false` forces the whole-segment path, and a
//! server that answers a blocking reload with nothing new drops the run back to
//! timed reloads.
//!
//! Scope: in-order segment fetch, one `DataFrame` per segment (per part in
//! low-latency mode). Throughput-driven ABR mid-stream is opt-in
//! ([`with_abr`](HlsSrc::with_abr)); a plain run keeps one variant.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::log::LogSource;
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{
    g2g_debug, AudioFormat, BusHandle, ByteStreamEncoding, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, Dim, ElementMetadata, G2gError, OutputSink, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, Seek, Segment, StreamType, TextFormat, VideoCodec,
};

use crate::abr::BandwidthEstimator;
use crate::fetch::{
    byte_frame, get_bytes, get_range_bytes, get_text, get_text_query, resolve_url,
    MAX_MANIFEST_BYTES, MAX_SEGMENT_BYTES,
};
use crate::fmp4::{TrackHeader, TrackKind};
use crate::hls::{parse, KeyMethod, MasterPlaylist, MediaPlaylist, MediaType, Playlist, Variant};
use crate::sampleaesdecrypt::{SampleAesKey, SampleAesKeyHandle};

/// One decodable stream a master playlist's variant exposes, the unit the
/// `playbin uri=hls://...` fan-out (M395) builds a branch from. The HLS analog of
/// `MkvStreamInfo` / `TsStreamInfo` / `Mp4StreamInfo`, plus the rendition
/// metadata HLS carries (a separate-rendition playlist `uri`, display `name`,
/// `language`).
#[derive(Debug, Clone)]
pub struct HlsStreamInfo {
    pub stream_type: StreamType,
    /// Discovery caps (geometry / channel layout `Any`-or-`0`, refined at runtime
    /// by the demuxer's `CapsChanged`), the branch's negotiation target.
    pub caps: Caps,
    pub video: bool,
    /// A separate alternate-rendition playlist (`#EXT-X-MEDIA:URI`), or `None`
    /// when the stream is multiplexed into the variant's own segments.
    pub uri: Option<String>,
    pub name: String,
    pub language: Option<String>,
}

/// Map an RFC 6381 `CODECS` entry to a `(StreamType, discovery caps, is_video)`.
/// `None` for an unrecognized codec (it is dropped from the discovered streams).
fn codec_to_stream(codec: &str) -> Option<(StreamType, Caps, bool)> {
    let video = |c| {
        (
            StreamType::Video,
            Caps::CompressedVideo {
                codec: c,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            true,
        )
    };
    let audio = |f| {
        (
            StreamType::Audio,
            Caps::Audio {
                format: f,
                channels: 0,
                sample_rate: 0,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            },
            false,
        )
    };
    Some(match codec {
        c if c.starts_with("avc1") || c.starts_with("avc3") => video(VideoCodec::H264),
        c if c.starts_with("hvc1") || c.starts_with("hev1") || c.starts_with("dvh") => {
            video(VideoCodec::H265)
        }
        c if c.starts_with("av01") => video(VideoCodec::Av1),
        c if c.starts_with("vp09") => video(VideoCodec::Vp9),
        c if c.starts_with("vp08") => video(VideoCodec::Vp8),
        c if c.starts_with("mp4a") => audio(AudioFormat::Aac),
        c if c.starts_with("opus") || c.starts_with("Opus") => audio(AudioFormat::Opus),
        _ => return None,
    })
}

/// The decodable streams a master's `variant` exposes (M395 rendition discovery):
/// the streams multiplexed in the variant's own segments (from its `CODECS`) plus
/// the alternate audio renditions its `AUDIO` group offers (each a separate
/// `#EXT-X-MEDIA` playlist). The muxed audio codec is dropped in favour of the
/// group's renditions only when the bound `AUDIO` group is entirely separate
/// playlists; if it has a *URI-less* rendition, per the HLS spec the audio is
/// carried in the variant's own segments (that rendition *is* the muxed track,
/// e.g. the `DEFAULT=YES` entry of Apple's bipbop), so the muxed audio is kept
/// (M422: without this, such a stream played silent, dropped as "a group is bound"
/// yet never surfaced as a separate playlist). Plus the alternate subtitle
/// renditions its `SUBTITLES` group offers (M418: each a separate WebVTT playlist,
/// surfaced as `Caps::Text { WebVtt }`).
pub fn variant_streams(master: &MasterPlaylist, variant: &Variant) -> Vec<HlsStreamInfo> {
    let mut out = Vec::new();
    // The audio rides the variant's own segments (is muxed) when no AUDIO group is
    // bound, or when the bound group has a URI-less rendition (the spec's "this
    // media is present in the variant stream" marker).
    let audio_is_muxed = match &variant.audio_group {
        None => true,
        Some(group) => master
            .renditions_in(MediaType::Audio, group)
            .iter()
            .any(|r| r.uri.is_none()),
    };
    for codec in variant.codec_list() {
        if let Some((stream_type, caps, video)) = codec_to_stream(codec) {
            // Only the muxed codecs ride the variant segments: video always, audio
            // only when it is not exclusively a set of separate renditions.
            if !video && !audio_is_muxed {
                continue;
            }
            let name = if video { "video" } else { "audio" };
            out.push(HlsStreamInfo {
                stream_type,
                caps,
                video,
                uri: None,
                name: String::from(name),
                language: None,
            });
        }
    }
    // Alternate audio renditions (separate playlists) from the variant's group.
    if let Some(group) = &variant.audio_group {
        for r in master.renditions_in(MediaType::Audio, group) {
            let Some(uri) = &r.uri else { continue }; // a URI-less rendition is muxed
            out.push(HlsStreamInfo {
                stream_type: StreamType::Audio,
                caps: Caps::Audio {
                    format: AudioFormat::Aac,
                    channels: 0,
                    sample_rate: 0,
                    channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
                },
                video: false,
                uri: Some(uri.clone()),
                name: r.name.clone(),
                language: r.language.clone(),
            });
        }
    }
    // Alternate subtitle renditions (separate WebVTT playlists) from the variant's
    // SUBTITLES group. A subtitle rendition is always a separate playlist (a
    // URI-less one is skipped). The segments are WebVTT (raw `.vtt` or fMP4 `wvtt`),
    // parsed at run; the branch's negotiation target is `Caps::Text { WebVtt }`.
    if let Some(group) = &variant.subtitles_group {
        for r in master.renditions_in(MediaType::Subtitles, group) {
            let Some(uri) = &r.uri else { continue };
            out.push(HlsStreamInfo {
                stream_type: StreamType::Text,
                caps: Caps::Text {
                    format: TextFormat::WebVtt,
                },
                video: false,
                uri: Some(uri.clone()),
                name: r.name.clone(),
                language: r.language.clone(),
            });
        }
    }
    out
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::hlssrc::HlsSrc;
///
/// let element = HlsSrc::new("https://example.com/master.m3u8")
///     .with_max_bandwidth(4_000_000)
///     .with_prebuffer_ms(2_000);
/// ```
/// Default `prebuffer-ms` on a plain playlist: roughly two segments of a live
/// playlist, so playback does not starve at every segment boundary while the
/// next one is fetched. `with_prebuffer_ms(0)` restores unbuffered emission. A
/// low-latency playlist derives its default from `PART-HOLD-BACK` instead, which
/// is a fraction of this.
pub const DEFAULT_PREBUFFER_MS: u64 = 2_000;

/// Delivery directives on a blocking playlist reload (RFC 8216bis §6.2.5.2):
/// the media sequence number and Part Index the client wants next. The server
/// holds the response until that partial segment is published.
const HLS_MSN_PARAM: &str = "_HLS_msn";
const HLS_PART_PARAM: &str = "_HLS_part";

/// Headroom over one target duration for a blocking reload's deadline: the
/// server should answer within a part duration, so a request outstanding this
/// long means it is not honouring the directives.
const BLOCKING_RELOAD_MARGIN_MS: u64 = 1_000;

/// Blocking reloads that returned nothing new before the run gives up on them
/// and goes back to timed polling.
const BLOCKING_MISS_LIMIT: u32 = 3;

#[derive(Debug)]
pub struct HlsSrc {
    url: String,
    /// ABR cap: select the highest-bandwidth variant at or below this (0 = no
    /// cap, pick the highest available).
    max_bandwidth: u64,
    /// Live-playlist reload interval in ms (0 = derive from `TARGETDURATION`).
    reload_interval_ms: u64,
    /// Container discovered by the negotiation-time probe: `IsoBmff` when the
    /// media playlist has an `#EXT-X-MAP` init segment (fMP4/CMAF), else `MpegTs`.
    /// Memoized so a re-fixate retry skips the probe.
    container: Option<ByteStreamEncoding>,
    /// The resolved playlist the probe already fetched, handed to `run()` so it
    /// reuses the negotiation fetch instead of resolving the same URL again.
    probed: Option<(MediaPlaylist, String)>,
    /// SAMPLE-AES key sink: when set, a `METHOD=SAMPLE-AES` segment publishes its
    /// fetched key/IV here (for a downstream `SampleAesDecrypt`) and the bytes are
    /// forwarded undecrypted. Without it a SAMPLE-AES playlist is rejected.
    sample_aes_key: Option<SampleAesKeyHandle>,
    /// Optional time-seek channel (M367). Unlike `FileSrc` (BYTES format), an
    /// adaptive source resolves a TIME seek to the media segment containing the
    /// target by walking the playlist's `#EXTINF` durations: it emits `Flush`,
    /// jumps to that segment, re-emits the `#EXT-X-MAP` init (so a downstream
    /// `fmp4demux` reset on the flush gets its `moov` again), emits the post-flush
    /// `Segment`, and resumes from there. The CMAF/DASH segment-transition case.
    seek: Option<SeekController>,
    /// Throughput-driven ABR (M371): when set and the playlist is a master, the
    /// run loop measures each segment's download and re-selects the variant whose
    /// declared bandwidth fits the estimate (scaled, and under `max_bandwidth`),
    /// switching the active media playlist and re-emitting the init on a change.
    /// Off by default, so a plain run picks one variant up front and keeps it.
    abr: bool,
    /// Text (subtitle) mode (M419): the playlist is a WebVTT subtitle rendition,
    /// not an A/V variant. The source advertises `Caps::Text { WebVtt }` instead of
    /// a `ByteStream` container and forwards each `.vtt` segment's text (blank-line
    /// separated so a downstream `SubParse` sees each segment's `WEBVTT` /
    /// `X-TIMESTAMP-MAP` header as its own non-cue block). An fMP4 (`wvtt`)
    /// rendition is de-framed to the same WebVTT text (M922). Off by default.
    text: bool,
    /// Start a live playlist from the front of the window (full DVR replay)
    /// instead of near the live edge (M438). Off by default: live playback
    /// follows what is being published. Opt in for a from-the-beginning replay of
    /// the available window. No effect on VOD (always from the front).
    full_replay: bool,
    /// Duration-keyed prebuffer target in ms (0 = off): fetch this much media
    /// ahead before emitting, posting `Buffering` on the attached bus. `None`
    /// derives it from the playlist: `PART-HOLD-BACK` in low-latency mode, else
    /// [`DEFAULT_PREBUFFER_MS`].
    prebuffer_ms: Option<u64>,
    /// Follow the playlist's low-latency tags when it publishes them (M1125):
    /// fetch and emit `#EXT-X-PART` partial segments and block the reload on the
    /// next part. On by default, since a playlist advertising parts and
    /// `CAN-BLOCK-RELOAD` is a server that expects a client to use them; `false`
    /// keeps the whole-segment path on such a playlist.
    low_latency: bool,
    bus: Option<BusHandle>,
    configured: bool,
}

impl HlsSrc {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            max_bandwidth: 0,
            reload_interval_ms: 0,
            container: None,
            probed: None,
            sample_aes_key: None,
            seek: None,
            abr: false,
            text: false,
            full_replay: false,
            prebuffer_ms: None,
            low_latency: true,
            bus: None,
            configured: false,
        }
    }

    /// Buffer this many milliseconds of media (summed `#EXTINF` durations)
    /// before emitting, and again after a flushing seek. `0` disables
    /// prebuffering. The duration-keyed sibling of `HttpSrc::prebuffer-bytes`.
    /// Unset, it is [`DEFAULT_PREBUFFER_MS`], or the playlist's `PART-HOLD-BACK`
    /// in low-latency mode.
    pub fn with_prebuffer_ms(mut self, ms: u64) -> Self {
        self.prebuffer_ms = Some(ms);
        self
    }

    /// Ignore the playlist's low-latency tags: fetch whole segments and reload
    /// on the `TARGETDURATION` timer even where the server offers partial
    /// segments and blocking reloads. The comparison baseline for a
    /// low-latency run, and the escape hatch for a server whose parts misbehave.
    pub fn without_low_latency(mut self) -> Self {
        self.low_latency = false;
        self
    }

    /// Attach the pipeline bus so prebuffering posts
    /// [`g2g_core::BusMessage::Buffering`] level reports.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Start a live playlist from the front of its sliding window (full DVR
    /// replay) rather than near the live edge (M438). Off by default, so live
    /// playback follows what is being published; opt in to replay the whole
    /// available window from the beginning. No effect on VOD (always front-to-end).
    pub fn with_full_replay(mut self) -> Self {
        self.full_replay = true;
        self
    }

    /// Treat the playlist as a WebVTT subtitle rendition (M419): advertise
    /// `Caps::Text { WebVtt }` and forward each `.vtt` segment's text (for a
    /// `SubParse` -> overlay branch), rather than a TS / fMP4 byte stream for a
    /// demuxer. Used by the `playbin` HLS subtitle fan-out for a separate
    /// `#EXT-X-MEDIA:TYPE=SUBTITLES` rendition. Either segment carriage works: a
    /// raw `.vtt` segment forwards its own text, an fMP4 (`wvtt`) one is de-framed
    /// through the fMP4 reader and rendered back to WebVTT (M922), so the branch
    /// is the same graph for both.
    pub fn with_text(mut self) -> Self {
        self.text = true;
        self
    }

    /// Enable throughput-driven ABR (M371): measure each segment's download and
    /// re-select the variant whose declared bandwidth fits the smoothed estimate
    /// (under any `max_bandwidth` cap), switching mid-stream and re-emitting the
    /// init segment on a change. A no-op for a media-only playlist (one
    /// rendition). Off by default (a fixed up-front variant).
    pub fn with_abr(mut self) -> Self {
        self.abr = true;
        self
    }

    /// Make the source time-seekable (M367): `run` polls `controller` before each
    /// segment fetch and, on a flushing seek, selects the media segment containing
    /// the target time (cumulative `#EXTINF` durations, clamped to the last
    /// segment), emits `Flush`, re-emits the `#EXT-X-MAP` init segment for a reset
    /// downstream demuxer, emits the post-flush `Segment`, and resumes there. The
    /// application keeps a clone of the controller to drive scrubbing.
    pub fn with_seek(mut self, controller: SeekController) -> Self {
        self.seek = Some(controller);
        self
    }

    /// Share a SAMPLE-AES key handle with a downstream `SampleAesDecrypt`: HlsSrc
    /// fetches the `#EXT-X-KEY` key/IV and publishes it here, the decryptor reads
    /// it. The auto-wiring path for sample-encrypted streams.
    pub fn with_sample_aes_key_handle(mut self, handle: SampleAesKeyHandle) -> Self {
        self.sample_aes_key = Some(handle);
        self
    }

    /// Cap variant selection to this bitrate (bits/sec); 0 picks the highest.
    pub fn with_max_bandwidth(mut self, bits_per_sec: u64) -> Self {
        self.max_bandwidth = bits_per_sec;
        self
    }

    /// Override the live-playlist reload interval (ms); 0 derives it from the
    /// playlist `TARGETDURATION`.
    pub fn with_reload_interval_ms(mut self, ms: u64) -> Self {
        self.reload_interval_ms = ms;
        self
    }

    fn cap(&self) -> Option<u64> {
        (self.max_bandwidth != 0).then_some(self.max_bandwidth)
    }

    /// Fetch the playlist (resolving master -> media) and decide the segment
    /// container: `IsoBmff` if the media playlist carries an `#EXT-X-MAP` init
    /// segment, else `MpegTs`. Memoized in `self.container`.
    async fn probe(&mut self) -> Result<ByteStreamEncoding, G2gError> {
        if let Some(enc) = self.container {
            return Ok(enc);
        }
        let client = reqwest::Client::new();
        let (media, media_url) = resolve_media(&client, &self.url, self.cap()).await?;
        let enc = if media.map_uri.is_some() {
            ByteStreamEncoding::IsoBmff
        } else {
            ByteStreamEncoding::MpegTs
        };
        self.container = Some(enc);
        self.probed = Some((media, media_url));
        Ok(enc)
    }
}

/// Fetch `url` and resolve a master playlist down to a media playlist, returning
/// it with the URL it came from (for segment-URI resolution and live reload).
async fn resolve_media(
    client: &reqwest::Client,
    url: &str,
    cap: Option<u64>,
) -> Result<(MediaPlaylist, String), G2gError> {
    let text = get_text(client, url, MAX_MANIFEST_BYTES).await?;
    match parse(&text).map_err(|_| G2gError::CapsMismatch)? {
        Playlist::Media(m) => Ok((m, String::from(url))),
        Playlist::Master(master) => {
            let variant = master.select(cap).ok_or(G2gError::CapsMismatch)?;
            let media_url = resolve_url(url, &variant.uri);
            let media_text = get_text(client, &media_url, MAX_MANIFEST_BYTES).await?;
            match parse(&media_text).map_err(|_| G2gError::CapsMismatch)? {
                Playlist::Media(m) => Ok((m, media_url)),
                // A master pointing at another master is malformed.
                Playlist::Master(_) => Err(G2gError::CapsMismatch),
            }
        }
    }
}

/// Select a variant from a master by bandwidth `cap`, fetch its media playlist,
/// and return it with its resolved URL and the chosen variant URI (so ABR can
/// detect a later switch). Used by the ABR path, which keeps the master around.
async fn fetch_variant_media(
    client: &reqwest::Client,
    master: &MasterPlaylist,
    master_url: &str,
    cap: Option<u64>,
) -> Result<(MediaPlaylist, String, String), G2gError> {
    let variant = master.select(cap).ok_or(G2gError::CapsMismatch)?;
    let variant_uri = variant.uri.clone();
    let media_url = resolve_url(master_url, &variant_uri);
    let media_text = get_text(client, &media_url, MAX_MANIFEST_BYTES).await?;
    match parse(&media_text).map_err(|_| G2gError::CapsMismatch)? {
        Playlist::Media(m) => Ok((m, media_url, variant_uri)),
        Playlist::Master(_) => Err(G2gError::CapsMismatch),
    }
}

/// Fetch a 16-byte AES-128 key, memoized by URI (keys rarely rotate, so a small
/// linear cache suffices).
async fn fetch_key(
    client: &reqwest::Client,
    cache: &mut Vec<(String, [u8; 16])>,
    url: &str,
) -> Result<[u8; 16], G2gError> {
    if let Some((_, key)) = cache.iter().find(|(u, _)| u == url) {
        return Ok(*key);
    }
    let bytes = get_bytes(client, url, MAX_MANIFEST_BYTES).await?;
    let key: [u8; 16] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| G2gError::CapsMismatch)?;
    cache.push((String::from(url), key));
    Ok(key)
}

/// One WebVTT subtitle segment as the text a downstream `SubParse` reads. A raw
/// `.vtt` segment is its own text, with a blank line appended so the next
/// segment's `WEBVTT` / `X-TIMESTAMP-MAP` header starts a fresh (non-cue) block.
/// An fMP4 (`wvtt`) segment is de-framed through the shared fMP4 reader and
/// written back as WebVTT, so both carriages of a `#EXT-X-MEDIA:TYPE=SUBTITLES`
/// rendition reach the parser as one `Caps::Text{WebVtt}` stream. `tracks` is the
/// `#EXT-X-MAP` init segment's track list; a self-contained segment (its own
/// `moov`) is parsed on its own.
fn webvtt_segment(bytes: Vec<u8>, tracks: Option<&[TrackHeader]>) -> Vec<u8> {
    if !matches!(
        crate::typefind::sniff(&bytes),
        Some(ByteStreamEncoding::IsoBmff)
    ) {
        let mut b = bytes;
        b.extend_from_slice(b"\n\n");
        return b;
    }
    let own;
    let tracks = match tracks {
        Some(t) => t,
        None => match crate::fmp4::parse_all_tracks(&bytes) {
            Ok(t) => {
                own = t;
                &own
            }
            Err(_) => return Vec::new(),
        },
    };
    webvtt_from_fmp4(&bytes, tracks)
}

/// Render an fMP4 subtitle segment's cues as WebVTT text. The samples de-frame
/// through [`parse_fragments_multi`](crate::fmp4::parse_fragments_multi) (ISO
/// 14496-30 `vttc` / `payl`, the same path the fMP4 demuxer uses), each becoming
/// a cue on the container's own timeline. Empty (`vtte` gap) samples are dropped,
/// blank lines inside a payload are too (they would split the cue block), and a
/// fragment that does not parse yields nothing rather than leaking binary into
/// the text stream.
fn webvtt_from_fmp4(data: &[u8], tracks: &[TrackHeader]) -> Vec<u8> {
    let Ok(samples) = crate::fmp4::parse_fragments_multi(data, tracks, 0, None) else {
        return Vec::new();
    };
    let mut out = alloc::format!("{}\n\n", crate::subparse::WEBVTT_HEADER);
    for (track_id, sample) in samples {
        let is_text = tracks
            .iter()
            .any(|t| t.track_id == track_id && matches!(t.kind, TrackKind::Text { .. }));
        if !is_text {
            continue;
        }
        let text = String::from_utf8_lossy(&sample.annexb);
        let end = sample.pts_ns.saturating_add(sample.duration_ns);
        // WebVTT cues carry no sequence number, so the index is unused.
        out.push_str(&crate::subparse::write_cue_block(
            0,
            sample.pts_ns,
            end,
            &text,
            g2g_core::TextFormat::WebVtt,
        ));
    }
    out.into_bytes()
}

/// The default HLS IV when `#EXT-X-KEY` carries none: the segment media-sequence
/// number as a 128-bit big-endian integer.
fn iv_from_sequence(seq: u64) -> [u8; 16] {
    let mut iv = [0u8; 16];
    iv[8..].copy_from_slice(&seq.to_be_bytes());
    iv
}

/// AES-128-CBC decrypt with PKCS7 padding, in place; returns the plaintext.
fn decrypt_aes128_cbc(
    key: &[u8; 16],
    iv: &[u8; 16],
    mut data: Vec<u8>,
) -> Result<Vec<u8>, G2gError> {
    use aes::cipher::{block_padding::Pkcs7, BlockDecryptMut, KeyIvInit};
    type Aes128CbcDec = cbc::Decryptor<aes::Aes128>;
    let plaintext_len = {
        let plaintext = Aes128CbcDec::new(&(*key).into(), &(*iv).into())
            .decrypt_padded_mut::<Pkcs7>(&mut data)
            .map_err(|_| G2gError::CapsMismatch)?;
        plaintext.len()
    };
    data.truncate(plaintext_len);
    Ok(data)
}

/// The index of the media segment containing `target_ns` and that segment's
/// cumulative start time (ns), walking `#EXTINF` durations. A target past the
/// end clamps to the last segment (GStreamer clamps a seek to the duration).
/// Empty playlist returns `(0, 0)` (the caller's bounds check breaks the loop).
fn segment_for_time(media: &MediaPlaylist, target_ns: u64) -> (usize, u64) {
    let mut start_ns = 0u64;
    let mut last_start = 0u64;
    for (idx, seg) in media.segments.iter().enumerate() {
        let dur_ns = (seg.duration_ms as u64).saturating_mul(1_000_000);
        let end_ns = start_ns.saturating_add(dur_ns);
        if target_ns < end_ns {
            return (idx, start_ns);
        }
        last_start = start_ns;
        start_ns = end_ns;
    }
    (media.segments.len().saturating_sub(1), last_start)
}

/// The first segment index to play on the initial load of a *live* playlist:
/// near the live edge, so playback follows what is being published now instead of
/// replaying the whole sliding window from its (already stale) start. Per RFC 8216
/// §6.3.3 the start is no closer than three target durations from the end (leaving
/// a playback buffer), clamped to the window start when the window is shorter than
/// that. A VOD playlist (`#EXT-X-ENDLIST`) plays from the beginning (index 0).
fn live_edge_start(media: &MediaPlaylist) -> usize {
    if media.end_list {
        return 0;
    }
    let edge_ns = u64::from(media.target_duration_secs.max(1))
        .saturating_mul(3)
        .saturating_mul(1_000_000_000);
    // Walk from the end accumulating durations; the start is the earliest segment
    // that still leaves >= 3 target durations of media ahead of it.
    let mut ahead_ns = 0u64;
    let mut start = 0usize;
    for (i, seg) in media.segments.iter().enumerate().rev() {
        ahead_ns = ahead_ns.saturating_add((seg.duration_ms as u64).saturating_mul(1_000_000));
        start = i;
        if ahead_ns >= edge_ns {
            break;
        }
    }
    start
}

/// The `(media sequence, part index)` to start a low-latency playlist at:
/// `PART-HOLD-BACK` back from the last published part (RFC 8216bis §6.2.4),
/// where a plain client starts three target durations back. The walk continues
/// past the hold-back point to the nearest `INDEPENDENT` part (or the segment
/// start, which always begins with a keyframe), so the decoder joins on a frame
/// it can decode. Clamped to the front of the window.
fn ll_edge_start(media: &MediaPlaylist) -> (u64, usize) {
    let hold_back_ns = u64::from(media.part_hold_back_ms()).saturating_mul(1_000_000);
    let mut behind_ns = 0u64;
    let mut start = (media.media_sequence, 0usize);
    'walk: for (idx, segment) in media.segments.iter().enumerate().rev() {
        let seq = media.media_sequence + idx as u64;
        // A segment whose parts the server has already dropped steps as a whole.
        for step in (0..segment.parts.len().max(1)).rev() {
            let part = segment.parts.get(step);
            let duration_ms = part.map_or(segment.duration_ms, |p| p.duration_ms);
            behind_ns = behind_ns.saturating_add(u64::from(duration_ms).saturating_mul(1_000_000));
            start = (seq, if part.is_some() { step } else { 0 });
            let joinable = step == 0 || part.is_none_or(|p| p.independent);
            if behind_ns >= hold_back_ns && joinable {
                break 'walk;
            }
        }
    }
    start
}

/// The startup prebuffer target in ms: an explicit `prebuffer-ms` wins, else the
/// playlist's `PART-HOLD-BACK` in low-latency mode (holding the two-segment
/// default there would put playback further behind the live edge than the parts
/// gained), else [`DEFAULT_PREBUFFER_MS`].
fn prebuffer_target_ms(explicit: Option<u64>, media: &MediaPlaylist, low_latency: bool) -> u64 {
    explicit.unwrap_or(if low_latency {
        u64::from(media.part_hold_back_ms())
    } else {
        DEFAULT_PREBUFFER_MS
    })
}

/// Whether the playlist publishes anything at or past `(seq, part)`, which is
/// what a blocking reload was asked to wait for. A server that ignores the
/// delivery directives answers at once with the playlist the client already has,
/// and this is how the run notices.
fn has_media_from(media: &MediaPlaylist, seq: u64, part: usize) -> bool {
    media.segments.iter().enumerate().any(|(idx, segment)| {
        let seg_seq = media.media_sequence + idx as u64;
        seg_seq > seq
            || (seg_seq == seq
                && (segment.parts.len() > part || (part == 0 && !segment.incomplete())))
    })
}

impl LogSource for HlsSrc {
    fn log_category(&self) -> &'static str {
        "hlssrc"
    }
}

impl SourceLoop for HlsSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    /// Probe the playlist at negotiation to discover the segment container
    /// (TS vs fMP4), the way `RtspSrc` does its DESCRIBE. The probe is memoized.
    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        Box::pin(async move {
            // Probe regardless (it memoizes the media playlist for `run`); a text
            // subtitle rendition advertises `Text { WebVtt }`, not its container.
            let encoding = self.probe().await?;
            if self.text {
                Ok(Caps::Text {
                    format: TextFormat::WebVtt,
                })
            } else {
                Ok(Caps::ByteStream { encoding })
            }
        })
    }

    async fn caps_constraint(&mut self) -> Result<CapsConstraint<'_>, G2gError> {
        let caps = self.intercept_caps().await?;
        Ok(CapsConstraint::Produces(CapsSet::one(caps)))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            // Subtitle (WebVTT) mode: forward each segment's text with a trailing
            // blank line so the next segment's `WEBVTT` / `X-TIMESTAMP-MAP` header
            // starts a fresh (non-cue) block downstream.
            let text_mode = self.text;
            let client = reqwest::Client::new();
            // ABR keeps the master playlist so the run loop can re-select a variant
            // per segment; `current_variant` tracks the loaded one to detect a
            // switch. Non-ABR reuses the probe's media playlist (one fixed variant).
            let mut master: Option<(MasterPlaylist, String)> = None;
            let mut current_variant: Option<String> = None;
            let mut estimator = BandwidthEstimator::new();
            let (mut media, mut media_url) = if self.abr {
                // Drop any probed media: ABR resolves fresh, keeping the master.
                self.probed = None;
                let text = get_text(&client, &self.url, MAX_MANIFEST_BYTES).await?;
                match parse(&text).map_err(|_| G2gError::CapsMismatch)? {
                    Playlist::Media(m) => (m, self.url.clone()),
                    Playlist::Master(mst) => {
                        let (m, murl, uri) =
                            fetch_variant_media(&client, &mst, &self.url, self.cap()).await?;
                        master = Some((mst, self.url.clone()));
                        current_variant = Some(uri);
                        (m, murl)
                    }
                }
            } else {
                // Reuse the playlist the probe already fetched at negotiation; only
                // resolve again if run() is entered without a prior probe.
                match self.probed.take() {
                    Some(probed) => probed,
                    None => resolve_media(&client, &self.url, self.cap()).await?,
                }
            };

            let mut sequence = 0u64;
            // Low latency: the playlist publishes partial segments and the server
            // holds a reload until the next one, so parts are fetched and emitted
            // as they appear. A subtitle rendition stays on the segment path (its
            // text is rewritten per segment, not per part).
            let mut ll = self.low_latency && !text_mode && media.low_latency();
            // Blocking reloads answered with nothing new: a server that ignores
            // the delivery directives drops the run back to timed reloads.
            let mut blocking_misses = 0u32;
            // Next timed reload on an absolute schedule (None until the first).
            let mut next_reload: Option<tokio::time::Instant> = None;
            // Duration-keyed prebuffer window (0 = pass-through); init segments
            // ride it with duration 0 so ordering survives an ABR re-init.
            let mut window = crate::segprebuf::SegmentPrebuffer::new(
                prebuffer_target_ms(self.prebuffer_ms, &media, ll),
                self.bus.clone(),
            );
            // AES-128 keys fetched once per URI and reused across segments.
            let mut keys: Vec<(String, [u8; 16])> = Vec::new();
            // Sample-encryption key per queued segment, in emit order (`None` for
            // an init segment or a clear one), and the running byte offset of the
            // emitted stream that a published key is keyed by.
            let mut pending_keys: alloc::collections::VecDeque<Option<SampleAesKey>> =
                alloc::collections::VecDeque::new();
            let mut emitted_bytes = 0u64;
            // Next HLS media-sequence number to play, and the Part Index within it
            // (0 = the whole segment); media below that on a live reload was
            // already delivered. A live playlist starts near the live edge
            // (skipping the stale front of the window) instead of its start, so
            // playback follows what is being published; VOD starts at the front. The
            // `full_replay` opt-in starts a live playlist from the window front too.
            let (mut next_seq, mut next_part) = if self.full_replay {
                (media.media_sequence, 0usize)
            } else if ll {
                ll_edge_start(&media)
            } else {
                (media.media_sequence + live_edge_start(&media) as u64, 0)
            };
            // fMP4: the EXT-X-MAP init segment (ftyp+moov) is emitted once, before
            // any media fragment, so a downstream fmp4demux sees the moov first.
            let mut init_emitted = false;
            // Text mode: the init segment is not forwarded but parsed, since an
            // fMP4 (`wvtt`) subtitle rendition needs its `moov` to de-frame the
            // cues out of the fragments that follow.
            let mut init_tracks: Option<Vec<TrackHeader>> = None;
            loop {
                // Index into `media.segments`; a flushing seek repositions it. The
                // matching HLS media-sequence number is `media.media_sequence + idx`.
                let mut idx = 0usize;
                loop {
                    // Apply a pending flushing time seek before the next fetch:
                    // resolve the target time to the segment containing it, flush,
                    // jump there, and re-emit the init segment (the downstream
                    // demuxer reset on the flush needs its moov again).
                    if let Some(seek) = self.seek.as_ref().and_then(|c| c.take_pending()) {
                        if seek.is_flush() {
                            let (target_idx, seg_start_ns) = segment_for_time(&media, seek.start);
                            // Queued lookahead is pre-seek media: drop it (and
                            // re-arm the prebuffer fill) before flushing.
                            window.clear();
                            pending_keys.clear();
                            // The byte coordinate a published key is keyed by
                            // restarts with the re-read stream, matching the
                            // demuxer's reset on this same flush.
                            emitted_bytes = 0;
                            if let Some(handle) = &self.sample_aes_key {
                                handle.lock().expect("key handle poisoned").reset_timeline();
                            }
                            out.push(PipelinePacket::Flush).await?;
                            idx = target_idx;
                            next_seq = media.media_sequence + target_idx as u64;
                            next_part = 0;
                            init_emitted = false;
                            out.push(PipelinePacket::Segment(Segment::for_flush_seek(
                                &Seek::flush_to(seg_start_ns),
                                None,
                            )))
                            .await?;
                        }
                        continue; // re-evaluate from the repositioned index
                    }

                    // fMP4: (re-)emit the EXT-X-MAP init (ftyp+moov) before any
                    // media fragment, so a downstream fmp4demux sees the moov first.
                    if !init_emitted {
                        if let Some(map) = &media.map_uri {
                            let init_url = resolve_url(&media_url, map);
                            let bytes = match media.map_byte_range {
                                Some(r) => {
                                    get_range_bytes(
                                        &client,
                                        &init_url,
                                        r.offset,
                                        r.length,
                                        MAX_SEGMENT_BYTES,
                                    )
                                    .await?
                                }
                                None => get_bytes(&client, &init_url, MAX_SEGMENT_BYTES).await?,
                            };
                            if text_mode {
                                init_tracks = crate::fmp4::parse_all_tracks(&bytes).ok();
                            } else if !bytes.is_empty() {
                                pending_keys.push_back(None);
                                window.admit(bytes, 0);
                            }
                        }
                        init_emitted = true;
                    }

                    // Fetch phase: pull segments into the window while it is
                    // below its duration target (or empty) and segments remain.
                    if idx < media.segments.len() && window.wants_fetch() {
                        let seg_seq = media.media_sequence + idx as u64;
                        // Bytes + elapsed of the segment just fetched, for the ABR
                        // estimator (None when this index was skipped on a live reload).
                        let mut measured: Option<(usize, u64)> = None;
                        let segment = &media.segments[idx];
                        let duration_ns = (segment.duration_ms as u64).saturating_mul(1_000_000);
                        // Fetch this segment's parts rather than the whole thing
                        // when it is still being produced (only its parts exist) or
                        // when it was joined part-way; a complete segment nothing
                        // was taken from is one request instead of several. An
                        // encrypted segment stays whole: the key covers the segment,
                        // not the part. Finishing a segment already joined part-way
                        // does not need low latency still on: dropping back to timed
                        // reloads mid-segment must not leave a hole.
                        let joined_part_way = seg_seq == next_seq && next_part > 0;
                        let use_parts = !segment.parts.is_empty()
                            && segment.key.is_none()
                            && (joined_part_way || (ll && segment.incomplete()));
                        // RFC 8216bis 4.4.4.7: an `#EXT-X-GAP` segment has no media
                        // behind its URI, so step over it instead of fetching a 404.
                        // A live packager pads a freshly started playlist with them.
                        if segment.gap {
                            // Only a gap past the delivery cursor moves it; the
                            // padding before it must not clobber next_part, or
                            // every reload replays the live segment from part 0.
                            if seg_seq + 1 > next_seq {
                                next_seq = seg_seq + 1;
                                next_part = 0;
                            }
                        } else if seg_seq < next_seq {
                            // already delivered
                        } else if use_parts {
                            let first = if seg_seq == next_seq { next_part } else { 0 };
                            for (part_index, part) in segment.parts.iter().enumerate().skip(first) {
                                if part.gap {
                                    continue;
                                }
                                let part_url = resolve_url(&media_url, &part.uri);
                                let t0 = g2g_core::metrics::monotonic_ns();
                                let bytes = match part.byte_range {
                                    Some(r) => {
                                        get_range_bytes(
                                            &client,
                                            &part_url,
                                            r.offset,
                                            r.length,
                                            MAX_SEGMENT_BYTES,
                                        )
                                        .await?
                                    }
                                    None => {
                                        get_bytes(&client, &part_url, MAX_SEGMENT_BYTES).await?
                                    }
                                };
                                g2g_debug!(
                                    self,
                                    "t={t} {stream} part ({seg_seq},{part_index}) fetched: {len} bytes in {ms} ms",
                                    t = g2g_core::metrics::monotonic_ns() / 1_000_000,
                                    stream = media_url.rsplit('/').next().unwrap_or("?"),
                                    len = bytes.len(),
                                    ms = g2g_core::metrics::monotonic_ns().saturating_sub(t0)
                                        / 1_000_000,
                                );
                                if !bytes.is_empty() {
                                    pending_keys.push_back(None);
                                    window.admit(
                                        bytes,
                                        u64::from(part.duration_ms).saturating_mul(1_000_000),
                                    );
                                }
                            }
                            if segment.incomplete() {
                                next_seq = seg_seq;
                                next_part = segment.parts.len();
                            } else {
                                next_seq = seg_seq + 1;
                                next_part = 0;
                            }
                        } else if segment.incomplete() {
                            // Being produced and not part-fetchable here: wait for
                            // the reload that publishes its `#EXTINF` and URI.
                        } else if joined_part_way {
                            // Its remaining parts have aged out of the playlist.
                            // Refetching the whole segment would replay what was
                            // already emitted, so step over the tail.
                            next_seq = seg_seq + 1;
                            next_part = 0;
                        } else {
                            let seg_url = resolve_url(&media_url, &segment.uri);
                            let t0 = g2g_core::metrics::monotonic_ns();
                            let bytes = match segment.byte_range {
                                Some(r) => {
                                    get_range_bytes(
                                        &client,
                                        &seg_url,
                                        r.offset,
                                        r.length,
                                        MAX_SEGMENT_BYTES,
                                    )
                                    .await?
                                }
                                None => get_bytes(&client, &seg_url, MAX_SEGMENT_BYTES).await?,
                            };
                            // Measure the downloaded (pre-decrypt) size against wall time.
                            measured = Some((
                                bytes.len(),
                                g2g_core::metrics::monotonic_ns().saturating_sub(t0),
                            ));
                            g2g_debug!(
                                self,
                                "t={t} {stream} segment {seg_seq} fetched: {len} bytes in {ms} ms",
                                t = g2g_core::metrics::monotonic_ns() / 1_000_000,
                                stream = media_url.rsplit('/').next().unwrap_or("?"),
                                len = bytes.len(),
                                ms = g2g_core::metrics::monotonic_ns().saturating_sub(t0)
                                    / 1_000_000,
                            );
                            // The sample-encryption key travels with its segment
                            // through the window, so it is published when those
                            // bytes are emitted, not now: fetching runs ahead.
                            let mut sample_key = None;
                            let bytes = match &segment.key {
                                None => bytes,
                                Some(key) => {
                                    let key_url = resolve_url(&media_url, &key.uri);
                                    let key_bytes = fetch_key(&client, &mut keys, &key_url).await?;
                                    let iv = key.iv.unwrap_or_else(|| iv_from_sequence(seg_seq));
                                    match key.method {
                                        // Whole-segment: decrypt before the demuxer.
                                        KeyMethod::Aes128 => {
                                            decrypt_aes128_cbc(&key_bytes, &iv, bytes)?
                                        }
                                        // Per-sample: hand the key to the decryptor
                                        // (TS) / demuxer (fMP4) and forward as-is.
                                        KeyMethod::SampleAes => {
                                            let handle = self
                                                .sample_aes_key
                                                .as_ref()
                                                .ok_or(G2gError::CapsMismatch)?;
                                            // A KEYID binds the key to the CENC key
                                            // identifier the segments name, which is
                                            // position-independent, so register it
                                            // as soon as it is known.
                                            if let Some(kid) = key.key_id {
                                                handle
                                                    .lock()
                                                    .expect("key handle poisoned")
                                                    .insert_kid(kid, key_bytes);
                                            }
                                            sample_key = Some(SampleAesKey { key: key_bytes, iv });
                                            bytes
                                        }
                                    }
                                }
                            };
                            let bytes = if text_mode {
                                webvtt_segment(bytes, init_tracks.as_deref())
                            } else {
                                bytes
                            };
                            if !bytes.is_empty() {
                                pending_keys.push_back(sample_key);
                                window.admit(bytes, duration_ns);
                            }
                            next_seq = seg_seq + 1;
                            next_part = 0;
                        }
                        idx += 1;

                        // ABR: feed the measured throughput and, if the best-fitting
                        // variant changed, switch to it (its media playlist), keeping
                        // the aligned index and re-emitting the new variant's init. The
                        // segment borrow above has ended, so reassigning `media` is safe.
                        if let (Some((len, elapsed)), Some((mst, master_url))) =
                            (measured, master.as_ref())
                        {
                            estimator.sample(len, elapsed);
                            if let Some(best) =
                                mst.select(estimator.effective_cap(self.max_bandwidth))
                            {
                                if current_variant.as_deref() != Some(best.uri.as_str()) {
                                    let new_uri = best.uri.clone();
                                    let new_url = resolve_url(master_url, &new_uri);
                                    let text =
                                        get_text(&client, &new_url, MAX_MANIFEST_BYTES).await?;
                                    if let Playlist::Media(m) =
                                        parse(&text).map_err(|_| G2gError::CapsMismatch)?
                                    {
                                        // Variants are time-aligned by media sequence, so
                                        // `idx` / `next_seq` carry over; re-emit the init.
                                        media = m;
                                        media_url = new_url;
                                        current_variant = Some(new_uri);
                                        init_emitted = false;
                                    }
                                }
                            }
                        }
                        continue;
                    }

                    // Emit phase: push the window front downstream. An empty
                    // window with nothing left to fetch ends this pass (live
                    // reload or end of playlist).
                    if let Some(bytes) = window.pop() {
                        // Publish this segment's key against the offset its bytes
                        // start at, so a downstream demuxer decrypts each fragment
                        // with the key its own segment carried.
                        if let Some(key) = pending_keys.pop_front().flatten() {
                            if let Some(handle) = &self.sample_aes_key {
                                handle
                                    .lock()
                                    .expect("key handle poisoned")
                                    .publish_at(emitted_bytes, key);
                            }
                        }
                        emitted_bytes = emitted_bytes.saturating_add(bytes.len() as u64);
                        out.push(PipelinePacket::DataFrame(byte_frame(bytes, sequence)))
                            .await?;
                        sequence += 1;
                        continue;
                    }
                    break;
                }

                if media.end_list {
                    break;
                }
                // Live reload. Low latency holds the request open until the part
                // the run wants next is published (`_HLS_msn` / `_HLS_part`),
                // which is what delivers a part as soon as it exists; a plain
                // playlist waits a reload interval and refetches.
                let target_ms = u64::from(media.target_duration_secs.max(1)) * 1000;
                let held = if ll {
                    let t0 = g2g_core::metrics::monotonic_ns();
                    let answer = get_text_query(
                        &client,
                        &media_url,
                        &[
                            (HLS_MSN_PARAM, next_seq),
                            (HLS_PART_PARAM, next_part as u64),
                        ],
                        core::time::Duration::from_millis(
                            target_ms.saturating_add(BLOCKING_RELOAD_MARGIN_MS),
                        ),
                        MAX_MANIFEST_BYTES,
                    )
                    .await
                    .ok();
                    g2g_debug!(
                        self,
                        "t={t} {stream} blocking reload for ({next_seq},{next_part}): held {ms} ms, {outcome}",
                        t = g2g_core::metrics::monotonic_ns() / 1_000_000,
                        stream = media_url.rsplit('/').next().unwrap_or("?"),
                        ms = g2g_core::metrics::monotonic_ns().saturating_sub(t0) / 1_000_000,
                        outcome = if answer.is_some() { "answered" } else { "miss" },
                    );
                    answer
                } else {
                    None
                };
                let text = match held {
                    Some(text) => text,
                    None => {
                        if ll {
                            // The held request failed or ran past its deadline.
                            blocking_misses += 1;
                        }
                        let interval_ms = if self.reload_interval_ms != 0 {
                            self.reload_interval_ms
                        } else {
                            target_ms
                        };
                        let interval = core::time::Duration::from_millis(interval_ms);
                        // Absolute cadence: sleeping for the interval after the
                        // fetch and emit work makes the true period interval
                        // plus work, and a live client then falls behind the
                        // publisher by the work time every cycle.
                        let due =
                            next_reload.unwrap_or_else(|| tokio::time::Instant::now() + interval);
                        tokio::time::sleep_until(due).await;
                        let now = tokio::time::Instant::now();
                        let scheduled = due + interval;
                        // A full interval behind: reload now, do not burst.
                        next_reload = Some(scheduled.max(now + interval / 2));
                        g2g_debug!(self, "timed reload after {} ms", interval_ms);
                        get_text(&client, &media_url, MAX_MANIFEST_BYTES).await?
                    }
                };
                media = match parse(&text).map_err(|_| G2gError::CapsMismatch)? {
                    Playlist::Media(m) => m,
                    Playlist::Master(_) => return Err(G2gError::CapsMismatch),
                };
                if ll {
                    if has_media_from(&media, next_seq, next_part) {
                        blocking_misses = 0;
                    } else {
                        blocking_misses += 1;
                    }
                }
                ll = self.low_latency
                    && !text_mode
                    && media.low_latency()
                    && blocking_misses < BLOCKING_MISS_LIMIT;
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        HLSSRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "HLS source",
            "Source/Network",
            "Reads an HLS playlist and streams its media segments",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.url = String::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            "max-bandwidth" => match value {
                PropValue::Uint(v) => {
                    self.max_bandwidth = v;
                    Ok(())
                }
                _ => Err(PropError::Type),
            },
            "reload-interval-ms" => match value {
                PropValue::Uint(v) => {
                    self.reload_interval_ms = v;
                    Ok(())
                }
                _ => Err(PropError::Type),
            },
            "full-replay" => match value {
                PropValue::Bool(v) => {
                    self.full_replay = v;
                    Ok(())
                }
                _ => Err(PropError::Type),
            },
            "abr" => match value {
                PropValue::Bool(v) => {
                    self.abr = v;
                    Ok(())
                }
                _ => Err(PropError::Type),
            },
            "prebuffer-ms" => match value {
                PropValue::Uint(v) => {
                    self.prebuffer_ms = Some(v);
                    Ok(())
                }
                _ => Err(PropError::Type),
            },
            "low-latency" => match value {
                PropValue::Bool(v) => {
                    self.low_latency = v;
                    Ok(())
                }
                _ => Err(PropError::Type),
            },
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.url.clone())),
            "max-bandwidth" => Some(PropValue::Uint(self.max_bandwidth)),
            "reload-interval-ms" => Some(PropValue::Uint(self.reload_interval_ms)),
            "full-replay" => Some(PropValue::Bool(self.full_replay)),
            "abr" => Some(PropValue::Bool(self.abr)),
            "prebuffer-ms" => Some(PropValue::Uint(
                self.prebuffer_ms.unwrap_or(DEFAULT_PREBUFFER_MS),
            )),
            "low-latency" => Some(PropValue::Bool(self.low_latency)),
            _ => None,
        }
    }
}

static HLSSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new("location", PropKind::Str, "HLS playlist URL (.m3u8)"),
    PropertySpec::new(
        "max-bandwidth",
        PropKind::Uint,
        "ABR cap in bits/sec; 0 selects the highest-bandwidth variant",
    ),
    PropertySpec::new(
        "reload-interval-ms",
        PropKind::Uint,
        "live-playlist reload interval in ms; 0 derives it from TARGETDURATION",
    ),
    PropertySpec::new(
        "full-replay",
        PropKind::Bool,
        "start a live playlist from the window front (full DVR replay) instead of near the live edge",
    ),
    PropertySpec::new(
        "abr",
        PropKind::Bool,
        "throughput-driven variant switching (measure downloads, re-select mid-stream)",
    )
    .with_default("false"),
    PropertySpec::new(
        "prebuffer-ms",
        PropKind::Uint,
        "media to buffer ahead before emitting, ms; posts Buffering bus messages (0 = off)",
    )
    .with_default("2000"),
    PropertySpec::new(
        "low-latency",
        PropKind::Bool,
        "follow the playlist's LL-HLS tags: fetch EXT-X-PART partial segments and block the reload on the next part",
    )
    .with_default("true"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hls::{parse, Playlist};

    fn master(text: &str) -> MasterPlaylist {
        match parse(text).unwrap() {
            Playlist::Master(m) => m,
            Playlist::Media(_) => panic!("expected master playlist"),
        }
    }

    fn media(text: &str) -> MediaPlaylist {
        match parse(text).unwrap() {
            Playlist::Media(m) => m,
            Playlist::Master(_) => panic!("expected media playlist"),
        }
    }

    /// Two complete one-second segments of five 200 ms parts each, then two
    /// parts of the segment being produced: the shape a live LL-HLS packager
    /// publishes. Only part 0 of a segment is `INDEPENDENT` (it carries the
    /// keyframe), as mediamtx marks them.
    fn ll_media(hold_back: &str) -> MediaPlaylist {
        let mut text = alloc::string::String::from(
            "#EXTM3U\n\
             #EXT-X-TARGETDURATION:1\n\
             #EXT-X-PART-INF:PART-TARGET=0.2\n\
             #EXT-X-MEDIA-SEQUENCE:10\n",
        );
        text.push_str(&alloc::format!(
            "#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES{hold_back}\n"
        ));
        for segment in 0..2 {
            for part in 0..5 {
                let independent = if part == 0 { ",INDEPENDENT=YES" } else { "" };
                text.push_str(&alloc::format!(
                    "#EXT-X-PART:DURATION=0.2,URI=\"s{segment}p{part}.mp4\"{independent}\n"
                ));
            }
            text.push_str(&alloc::format!("#EXTINF:1.0,\nseg{segment}.mp4\n"));
        }
        text.push_str(
            "#EXT-X-PART:DURATION=0.2,URI=\"s2p0.mp4\",INDEPENDENT=YES\n\
             #EXT-X-PART:DURATION=0.2,URI=\"s2p1.mp4\"\n",
        );
        media(&text)
    }

    #[test]
    fn ll_start_snaps_back_to_an_independent_part() {
        // The open segment (12) holds 0.4 s of parts, so a 0.4 s hold-back starts
        // exactly at its part 0.
        assert_eq!(ll_edge_start(&ll_media(",PART-HOLD-BACK=0.4")), (12, 0));

        // 0.5 s reaches one part into the previous segment, whose part 4 is not
        // independently decodable: the start walks back to that segment's part 0,
        // the nearest frame a decoder can join on, rather than stopping there.
        assert_eq!(ll_edge_start(&ll_media(",PART-HOLD-BACK=0.5")), (11, 0));
    }

    /// The whole point of low-latency start: three target durations back (what a
    /// plain client uses) is several segments, PART-HOLD-BACK is a fraction of one.
    #[test]
    fn ll_start_is_nearer_the_live_edge_than_the_plain_one() {
        let m = ll_media(",PART-HOLD-BACK=0.5");
        let (ll_seq, _) = ll_edge_start(&m);
        let plain_seq = m.media_sequence + live_edge_start(&m) as u64;
        assert!(
            ll_seq > plain_seq,
            "low latency starts at {ll_seq}, a plain client at {plain_seq}"
        );
    }

    #[test]
    fn prebuffer_defaults_to_part_hold_back_in_low_latency_mode() {
        let m = ll_media(",PART-HOLD-BACK=0.5");
        assert_eq!(prebuffer_target_ms(None, &m, true), 500);
        // Same playlist with low latency off keeps the two-segment default.
        assert_eq!(
            prebuffer_target_ms(None, &m, false),
            DEFAULT_PREBUFFER_MS,
            "a plain run is unaffected by the LL tags"
        );
        // An explicit prebuffer-ms overrides both.
        assert_eq!(prebuffer_target_ms(Some(120), &m, true), 120);
        assert_eq!(prebuffer_target_ms(Some(120), &m, false), 120);
        // No PART-HOLD-BACK: three part targets.
        assert_eq!(prebuffer_target_ms(None, &ll_media(""), true), 600);
    }

    #[test]
    fn blocking_reload_progress_is_measured_against_the_wanted_part() {
        let m = ll_media(",PART-HOLD-BACK=0.5");
        // The open segment (12) has published parts 0 and 1.
        assert!(has_media_from(&m, 12, 1));
        assert!(!has_media_from(&m, 12, 2), "part 2 is not published yet");
        assert!(!has_media_from(&m, 13, 0), "nor is the next segment");
        // A complete segment counts as media at part index 0.
        assert!(has_media_from(&m, 11, 0));
    }

    #[test]
    fn variant_streams_splits_a_muxed_ts_variant() {
        // A muxed variant (no AUDIO group): CODECS lists both, both are in-segment.
        let m = master(
            "#EXTM3U\n\
             #EXT-X-STREAM-INF:BANDWIDTH=2400000,CODECS=\"avc1.4d401e,mp4a.40.2\"\n\
             v.m3u8\n",
        );
        let streams = variant_streams(&m, &m.variants[0]);
        assert_eq!(streams.len(), 2);
        assert!(
            streams[0].video
                && matches!(
                    streams[0].caps,
                    Caps::CompressedVideo {
                        codec: VideoCodec::H264,
                        ..
                    }
                )
        );
        assert!(
            !streams[1].video
                && matches!(
                    streams[1].caps,
                    Caps::Audio {
                        format: AudioFormat::Aac,
                        ..
                    }
                )
        );
        // Muxed streams ride the variant segments: no separate rendition URI.
        assert!(streams.iter().all(|s| s.uri.is_none()));
    }

    #[test]
    fn variant_streams_uses_separate_audio_renditions() {
        // An AUDIO group: the variant carries video only; audio is two separate
        // rendition playlists (the muxed mp4a codec is dropped in their favour).
        let m = master(
            "#EXTM3U\n\
             #EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"aud\",NAME=\"en\",LANGUAGE=\"en\",DEFAULT=YES,URI=\"a/en.m3u8\"\n\
             #EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"aud\",NAME=\"fr\",LANGUAGE=\"fr\",URI=\"a/fr.m3u8\"\n\
             #EXT-X-STREAM-INF:BANDWIDTH=2400000,CODECS=\"avc1.4d401e,mp4a.40.2\",AUDIO=\"aud\"\n\
             v.m3u8\n",
        );
        let streams = variant_streams(&m, &m.variants[0]);
        assert_eq!(
            streams.len(),
            3,
            "one muxed video + two separate audio renditions"
        );
        assert!(streams[0].video && streams[0].uri.is_none());
        assert_eq!(streams[1].uri.as_deref(), Some("a/en.m3u8"));
        assert_eq!(streams[1].language.as_deref(), Some("en"));
        assert_eq!(streams[2].uri.as_deref(), Some("a/fr.m3u8"));
        assert!(streams[1..].iter().all(|s| !s.video));
    }

    #[test]
    fn variant_streams_discovers_subtitle_renditions() {
        // A SUBTITLES group: each rendition is a separate WebVTT playlist, surfaced
        // as a Caps::Text stream alongside the muxed A/V (M418).
        let m = master(
            "#EXTM3U\n\
             #EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID=\"subs\",NAME=\"English\",LANGUAGE=\"en\",DEFAULT=YES,URI=\"s/en.m3u8\"\n\
             #EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID=\"subs\",NAME=\"French\",LANGUAGE=\"fr\",URI=\"s/fr.m3u8\"\n\
             #EXT-X-STREAM-INF:BANDWIDTH=2400000,CODECS=\"avc1.4d401e,mp4a.40.2\",SUBTITLES=\"subs\"\n\
             v.m3u8\n",
        );
        let streams = variant_streams(&m, &m.variants[0]);
        // muxed video + muxed audio (no AUDIO group) + two subtitle renditions.
        let subs: Vec<_> = streams
            .iter()
            .filter(|s| matches!(s.caps, Caps::Text { .. }))
            .collect();
        assert_eq!(subs.len(), 2, "two subtitle renditions discovered");
        assert_eq!(subs[0].stream_type, StreamType::Text);
        assert_eq!(
            subs[0].caps,
            Caps::Text {
                format: TextFormat::WebVtt
            }
        );
        assert_eq!(subs[0].uri.as_deref(), Some("s/en.m3u8"));
        assert_eq!(subs[0].language.as_deref(), Some("en"));
        assert_eq!(subs[1].uri.as_deref(), Some("s/fr.m3u8"));
        assert!(subs.iter().all(|s| !s.video));
    }
}

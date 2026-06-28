//! HLS source (HlsSrc, `hls` feature): fetches an `.m3u8` playlist, selects a
//! variant (simple bandwidth-capped ABR), and streams that variant's MPEG-TS
//! media segments downstream as a `Caps::ByteStream{MpegTs}` for `tsdemux`, then
//! `Eos`. The [`hls`](crate::hls) parser does the playlist work; this element
//! adds the fetching (via `reqwest`, like [`HttpSrc`](crate::httpsrc)) and URL
//! resolution.
//!
//! VOD (a playlist with `#EXT-X-ENDLIST`) plays its segments once then `Eos`.
//! Live (no ENDLIST) reloads the media playlist on an interval, plays each new
//! segment once (tracked by HLS media-sequence), and ends when ENDLIST finally
//! appears or downstream shuts down.
//!
//! `#EXT-X-KEY:METHOD=AES-128` segments are decrypted in place: the 16-byte key
//! is fetched from the key URI (cached per run) and each segment is AES-128-CBC
//! decrypted with the explicit `IV` or, absent one, the segment media-sequence
//! number as a 128-bit big-endian IV. For `METHOD=SAMPLE-AES` (per-sample, not
//! whole-segment) the fetched key/IV is published to a shared handle
//! ([`with_sample_aes_key_handle`](HlsSrc::with_sample_aes_key_handle)) for a
//! downstream [`SampleAesDecrypt`](crate::sampleaesdecrypt) and the bytes are
//! forwarded undecrypted; without a handle a SAMPLE-AES playlist is rejected. The
//! init segment (`#EXT-X-MAP`) is assumed unencrypted.
//!
//! Single-file CMAF is supported via `#EXT-X-BYTERANGE` (and `#EXT-X-MAP`'s
//! `BYTERANGE`): a segment that carries one fetches only its sub-range with an
//! HTTP `Range` request (M368), the offset continuing from the previous
//! sub-range of the same resource when the tag omits an explicit `@offset`.
//!
//! Scope: in-order segment fetch, one `DataFrame` per segment, a fixed variant
//! (no mid-stream ABR switch). Throughput-driven ABR and live-edge start (skip
//! to the last few segments) are follow-ups (DESIGN_TODO).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Seek, Segment,
};

use crate::fetch::{
    byte_frame, get_bytes, get_range_bytes, get_text, resolve_url, MAX_MANIFEST_BYTES,
    MAX_SEGMENT_BYTES,
};
use crate::abr::BandwidthEstimator;
use crate::hls::{parse, KeyMethod, MasterPlaylist, MediaPlaylist, Playlist};
use crate::sampleaesdecrypt::{SampleAesKey, SampleAesKeyHandle};

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
            configured: false,
        }
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
    let key: [u8; 16] = bytes.as_slice().try_into().map_err(|_| G2gError::CapsMismatch)?;
    cache.push((String::from(url), key));
    Ok(key)
}

/// The default HLS IV when `#EXT-X-KEY` carries none: the segment media-sequence
/// number as a 128-bit big-endian integer.
fn iv_from_sequence(seq: u64) -> [u8; 16] {
    let mut iv = [0u8; 16];
    iv[8..].copy_from_slice(&seq.to_be_bytes());
    iv
}

/// AES-128-CBC decrypt with PKCS7 padding, in place; returns the plaintext.
fn decrypt_aes128_cbc(key: &[u8; 16], iv: &[u8; 16], mut data: Vec<u8>) -> Result<Vec<u8>, G2gError> {
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
            let encoding = self.probe().await?;
            Ok(Caps::ByteStream { encoding })
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
            // AES-128 keys fetched once per URI and reused across segments.
            let mut keys: Vec<(String, [u8; 16])> = Vec::new();
            // Next HLS media-sequence number to play; segments below it on a live
            // reload were already delivered.
            let mut next_seq = media.media_sequence;
            // fMP4: the EXT-X-MAP init segment (ftyp+moov) is emitted once, before
            // any media fragment, so a downstream fmp4demux sees the moov first.
            let mut init_emitted = false;
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
                            let (target_idx, seg_start_ns) =
                                segment_for_time(&media, seek.start);
                            out.push(PipelinePacket::Flush).await?;
                            idx = target_idx;
                            next_seq = media.media_sequence + target_idx as u64;
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
                                    get_range_bytes(&client, &init_url, r.offset, r.length, MAX_SEGMENT_BYTES)
                                        .await?
                                }
                                None => get_bytes(&client, &init_url, MAX_SEGMENT_BYTES).await?,
                            };
                            if !bytes.is_empty() {
                                out.push(PipelinePacket::DataFrame(byte_frame(bytes, sequence)))
                                    .await?;
                                sequence += 1;
                            }
                        }
                        init_emitted = true;
                    }

                    if idx >= media.segments.len() {
                        break;
                    }
                    let seg_seq = media.media_sequence + idx as u64;
                    // Bytes + elapsed of the segment just fetched, for the ABR
                    // estimator (None when this index was skipped on a live reload).
                    let mut measured: Option<(usize, u64)> = None;
                    let segment = &media.segments[idx];
                    if seg_seq >= next_seq {
                        let seg_url = resolve_url(&media_url, &segment.uri);
                        let t0 = g2g_core::metrics::monotonic_ns();
                        let bytes = match segment.byte_range {
                            Some(r) => {
                                get_range_bytes(&client, &seg_url, r.offset, r.length, MAX_SEGMENT_BYTES)
                                    .await?
                            }
                            None => get_bytes(&client, &seg_url, MAX_SEGMENT_BYTES).await?,
                        };
                        // Measure the downloaded (pre-decrypt) size against wall time.
                        measured =
                            Some((bytes.len(), g2g_core::metrics::monotonic_ns().saturating_sub(t0)));
                        let bytes = match &segment.key {
                            None => bytes,
                            Some(key) => {
                                let key_url = resolve_url(&media_url, &key.uri);
                                let key_bytes = fetch_key(&client, &mut keys, &key_url).await?;
                                let iv = key.iv.unwrap_or_else(|| iv_from_sequence(seg_seq));
                                match key.method {
                                    // Whole-segment: decrypt before the demuxer.
                                    KeyMethod::Aes128 => decrypt_aes128_cbc(&key_bytes, &iv, bytes)?,
                                    // Per-sample: publish the key for a downstream
                                    // SampleAesDecrypt and forward the bytes as-is.
                                    KeyMethod::SampleAes => {
                                        let handle = self
                                            .sample_aes_key
                                            .as_ref()
                                            .ok_or(G2gError::CapsMismatch)?;
                                        *handle.lock().expect("key handle poisoned") =
                                            Some(SampleAesKey { key: key_bytes, iv });
                                        bytes
                                    }
                                }
                            }
                        };
                        if !bytes.is_empty() {
                            out.push(PipelinePacket::DataFrame(byte_frame(bytes, sequence))).await?;
                            sequence += 1;
                        }
                        next_seq = seg_seq + 1;
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
                        if let Some(best) = mst.select(estimator.effective_cap(self.max_bandwidth)) {
                            if current_variant.as_deref() != Some(best.uri.as_str()) {
                                let new_uri = best.uri.clone();
                                let new_url = resolve_url(master_url, &new_uri);
                                let text = get_text(&client, &new_url, MAX_MANIFEST_BYTES).await?;
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
                }

                if media.end_list {
                    break;
                }
                // Live: wait a reload interval, then refetch the media playlist.
                let interval_ms = if self.reload_interval_ms != 0 {
                    self.reload_interval_ms
                } else {
                    u64::from(media.target_duration_secs.max(1)) * 1000
                };
                tokio::time::sleep(core::time::Duration::from_millis(interval_ms)).await;
                let text = get_text(&client, &media_url, MAX_MANIFEST_BYTES).await?;
                media = match parse(&text).map_err(|_| G2gError::CapsMismatch)? {
                    Playlist::Media(m) => m,
                    Playlist::Master(_) => return Err(G2gError::CapsMismatch),
                };
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
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.url.clone())),
            "max-bandwidth" => Some(PropValue::Uint(self.max_bandwidth)),
            "reload-interval-ms" => Some(PropValue::Uint(self.reload_interval_ms)),
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
];

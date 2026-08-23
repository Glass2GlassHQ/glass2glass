//! Fragmented-MP4 / CMAF parsing shared by [`Mp4Src`](crate::mp4src) (file
//! source) and [`Fmp4Demux`](crate::fmp4demux) (byte-stream demuxer). Pure
//! `no_std + alloc`: reads the `moov` init (codec, geometry, timescale,
//! parameter sets) and walks `moof`+`mdat` fragments into Annex-B samples.
//!
//! Supported profile: one video track, `trun` v0 with explicit sample sizes,
//! `default-base-is-moof` data offsets landing on the following `mdat`'s
//! payload (what `Mp4Mux` writes and CMAF single-track files share). Anything
//! else fails loud rather than emitting a corrupt bitstream.

use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::{
    AudioFormat, Chapter, ClosedCaptionFormat, G2gError, TagList, TextFormat, VideoCodec,
};

use crate::cea::{parse_cdp, write_cc_data, CcTriple};
use crate::cenc::{
    fragment_sample_crypt, parse_movie_seig, parse_sinf, CencDefaults, CencTrack, SampleCrypt,
};
#[cfg(any(test, feature = "dash"))]
use crate::mp4box::next_box_len;
use crate::mp4box::{
    be32, be64, boxes, boxes_at, find_box, find_path, parse_chpl, parse_esds, parse_esds_video,
    parse_ilst_tags,
};
use crate::opusparse::{opus_head_from_dops, parse_opus_head};

#[derive(Debug)]
pub(crate) struct Header {
    pub(crate) codec: VideoCodec,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) timescale: u32,
    /// Total track duration in nanoseconds from `mdhd`, or `None` when the box
    /// reports `0` (a fragmented / live init segment whose length is unknown
    /// until the fragments arrive). Feeds the M203 `DURATION` query.
    pub(crate) duration_ns: Option<u64>,
    /// Parameter-set NALUs in container order (SPS,PPS for H.264; VPS,SPS,PPS
    /// for H.265), prepended to the first sample if it carries none in-band.
    pub(crate) param_sets: Vec<Vec<u8>>,
    /// Common-encryption metadata from the init's `tenc` (plus any movie-level
    /// `seig` table), `None` for a clear track.
    pub(crate) cenc: Option<CencTrack>,
}

/// In-place sample decryptor: given the crypto resolved for a sample (scheme, IV,
/// pattern, KID, subsample map), the byte offset of its fragment in the source
/// stream, and the sample's bytes, rewrites the protected ranges. The demuxers
/// supply one that resolves the content key from the KID and that offset (a
/// rotating `#EXT-X-KEY` changes key by stream position); a sample whose key is
/// unavailable fails the parse rather than emitting garbage.
pub(crate) type SampleDecrypt<'a> =
    &'a mut dyn FnMut(&SampleCrypt, u64, &mut [u8]) -> Result<(), G2gError>;

#[derive(Debug)]
pub(crate) struct Sample {
    pub(crate) annexb: Vec<u8>,
    pub(crate) pts_ns: u64,
    /// When the sample decodes, which a reordered (B-frame) stream places ahead
    /// of its `pts_ns`. Equal to it for the fragment parsers, which take their
    /// timeline from `tfdt` + decode-order durations and leave the reorder to
    /// the decoder.
    pub(crate) dts_ns: u64,
    pub(crate) duration_ns: u64,
    /// Whether the access unit carries an IDR picture (a seek snap point).
    pub(crate) keyframe: bool,
}

/// Parse the `moov` init box into a [`Header`] (codec, geometry, timescale,
/// parameter sets). `data` must contain the `moov` (a whole init segment or a
/// whole file).
pub(crate) fn parse_header(data: &[u8]) -> Result<Header, G2gError> {
    let moov = find_box(data, b"moov").ok_or(G2gError::CapsMismatch)?;
    let trak = find_box(moov, b"trak").ok_or(G2gError::CapsMismatch)?;

    // tkhd v0: width/height as 16.16 at payload offset 76/80 (after the
    // 4-byte version/flags).
    let tkhd = find_box(trak, b"tkhd").ok_or(G2gError::CapsMismatch)?;
    if tkhd.first() != Some(&0) {
        return Err(G2gError::CapsMismatch);
    }
    let width = be32(tkhd, 76)? >> 16;
    let height = be32(tkhd, 80)? >> 16;

    // mdhd v0: timescale at payload offset 12.
    let mdia = find_box(trak, b"mdia").ok_or(G2gError::CapsMismatch)?;
    let mdhd = find_box(mdia, b"mdhd").ok_or(G2gError::CapsMismatch)?;
    if mdhd.first() != Some(&0) {
        return Err(G2gError::CapsMismatch);
    }
    let timescale = be32(mdhd, 12)?;
    if timescale == 0 {
        return Err(G2gError::CapsMismatch);
    }
    // mdhd v0 duration at payload offset 16, in timescale units. `0` means the
    // length is not yet known (a fragmented init segment), so report `None`.
    let duration_ns = match be32(mdhd, 16)? {
        0 => None,
        units => Some((units as u128 * 1_000_000_000 / timescale as u128) as u64),
    };

    // stsd's first entry is the visual sample entry: avc1/avcC (H.264) or
    // hvc1/hev1 with hvcC (H.265). Its config record carries the parameter sets.
    let stbl = find_path(mdia, &[b"minf", b"stbl"]).ok_or(G2gError::CapsMismatch)?;
    let stsd = find_box(stbl, b"stsd").ok_or(G2gError::CapsMismatch)?;
    // full box: version/flags + entry count, then the first sample entry.
    let entries = stsd.get(8..).ok_or(G2gError::CapsMismatch)?;
    // visual sample entry: 78 bytes of fixed fields before the nested boxes. An
    // encrypted track uses `encv`, carrying the original codec config plus a
    // `sinf` (frma original format + cbcs scheme + tenc defaults).
    let (codec, param_sets, cenc) = if let Some(avc1) = find_box(entries, b"avc1") {
        let children = avc1.get(78..).ok_or(G2gError::CapsMismatch)?;
        let (sps, pps) = parse_avcc(find_box(children, b"avcC").ok_or(G2gError::CapsMismatch)?)?;
        (VideoCodec::H264, Vec::from([sps, pps]), None)
    } else if let Some(hvc1) = find_box(entries, b"hvc1").or_else(|| find_box(entries, b"hev1")) {
        let children = hvc1.get(78..).ok_or(G2gError::CapsMismatch)?;
        let hvcc = find_box(children, b"hvcC").ok_or(G2gError::CapsMismatch)?;
        (VideoCodec::H265, parse_hvcc(hvcc)?, None)
    } else if let Some(mp4v) = find_box(entries, b"mp4v") {
        // MPEG-4 Part 2: the mp4v visual sample entry nests an esds carrying the
        // VOL header as its DecoderSpecificInfo. Confirm objectTypeIndication
        // 0x20 (Visual ISO/IEC 14496-2) so another mp4v-boxed codec is rejected.
        let children = mp4v.get(78..).ok_or(G2gError::CapsMismatch)?;
        let esds = find_box(children, b"esds").ok_or(G2gError::CapsMismatch)?;
        let (oti, dsi) = parse_esds_video(esds)?;
        if oti != 0x20 {
            return Err(G2gError::CapsMismatch);
        }
        // The VOL header is the single config blob (empty if carried in-band).
        let sets = if dsi.is_empty() {
            Vec::new()
        } else {
            Vec::from([dsi])
        };
        (VideoCodec::Mpeg4Part2, sets, None)
    } else if let Some(av01) = find_box(entries, b"av01") {
        (VideoCodec::Av1, parse_av1c_config(av01)?, None)
    } else if let Some(encv) = find_box(entries, b"encv") {
        let children = encv.get(78..).ok_or(G2gError::CapsMismatch)?;
        let sinf = find_box(children, b"sinf").ok_or(G2gError::CapsMismatch)?;
        let cenc = parse_sinf(sinf)?;
        let frma = find_box(sinf, b"frma").ok_or(G2gError::CapsMismatch)?;
        let (codec, param_sets) = match frma.get(0..4) {
            Some(b"avc1") => {
                let avcc = find_box(children, b"avcC").ok_or(G2gError::CapsMismatch)?;
                let (sps, pps) = parse_avcc(avcc)?;
                (VideoCodec::H264, Vec::from([sps, pps]))
            }
            Some(b"hvc1") | Some(b"hev1") => {
                let hvcc = find_box(children, b"hvcC").ok_or(G2gError::CapsMismatch)?;
                (VideoCodec::H265, parse_hvcc(hvcc)?)
            }
            _ => return Err(G2gError::CapsMismatch),
        };
        (codec, param_sets, Some(cenc))
    } else {
        return Err(G2gError::CapsMismatch);
    };

    Ok(Header {
        codec,
        width,
        height,
        timescale,
        duration_ns,
        param_sets,
        cenc: cenc_track(cenc, stbl)?,
    })
}

/// Pair a track's `tenc` defaults with the movie-level `seig` table its `stbl`
/// carries, which a fragment's `sbgp` can index instead of a `traf`-local one.
fn cenc_track(defaults: Option<CencDefaults>, stbl: &[u8]) -> Result<Option<CencTrack>, G2gError> {
    match defaults {
        Some(defaults) => {
            let movie_seig = parse_movie_seig(stbl, defaults.scheme)?;
            Ok(Some(CencTrack {
                defaults,
                movie_seig,
            }))
        }
        None => Ok(None),
    }
}

/// What one track carries: a video elementary stream (codec + geometry +
/// parameter sets), an audio elementary stream (format + channel layout + codec
/// config), or a timed-text stream (subtitle format). The multi-track read-side
/// analog of `TrackInit` in the muxer; clear tracks only (encryption stays
/// single-track via [`parse_header`]).
#[derive(Debug, Clone)]
pub(crate) enum TrackKind {
    Video {
        codec: VideoCodec,
        width: u32,
        height: u32,
        param_sets: Vec<Vec<u8>>,
    },
    Audio {
        format: AudioFormat,
        channels: u8,
        sample_rate: u32,
        /// The track's out-of-band codec config, in the form the elementary
        /// stream carries it: AAC's AudioSpecificConfig (`esds`), or an Opus
        /// `OpusHead` rebuilt from the `dOps` (M791). Empty when there is none.
        config: Vec<u8>,
    },
    /// A timed-text subtitle track. The container carries the per-cue timing
    /// (sample PTS + duration); `format` names the elementary cue payload's syntax
    /// downstream, and `sample` the on-disk sample-entry framing (which selects
    /// de-framing): `tx3g` / `wvtt` cues are plain UTF-8 ([`TextFormat::Utf8`]),
    /// `stpp` samples are TTML documents ([`TextFormat::Ttml`]).
    Text {
        format: TextFormat,
        sample: TextSampleFormat,
    },
    /// A raw closed-caption track (`c608` / `c708`): the caption data is its own
    /// elementary stream rather than SEI inside the video. Samples de-frame from
    /// their container atoms to the `cc_data` triple stream
    /// [`Caps::ClosedCaption`](g2g_core::Caps::ClosedCaption) carries.
    ClosedCaption { format: ClosedCaptionFormat },
}

/// The on-disk timed-text sample format an MP4 text `trak` stores. Distinct from
/// the elementary [`TextFormat`] the de-framed cue carries: both `tx3g` and `wvtt`
/// de-frame to plain UTF-8, but by different framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextSampleFormat {
    /// 3GPP / QuickTime timed text: 2-byte length prefix + UTF-8 cue + style boxes.
    Tx3g,
    /// WebVTT-in-MP4 (ISO 14496-30): per-sample `vttc` / `vtte` boxes; the `payl`
    /// payloads concatenate to the cue's UTF-8 text.
    Wvtt,
    /// TTML-in-MP4 (`stpp`): each sample is a complete XML document, passed through.
    Stpp,
}

/// One track's init data parsed from a `moov/trak`: the `track_ID` (which keys
/// the fragments in [`parse_fragments_multi`]), the media timescale, the
/// elementary-stream kind, the common-encryption metadata for an encrypted track
/// (`None` for a clear one), and the track's own `udta/meta/ilst` metadata
/// (empty when it carries none).
#[derive(Debug, Clone)]
pub(crate) struct TrackHeader {
    pub(crate) track_id: u32,
    pub(crate) timescale: u32,
    pub(crate) kind: TrackKind,
    pub(crate) cenc: Option<CencTrack>,
    pub(crate) tags: TagList,
}

/// Parse every forwardable (`vide` / `soun` / timed-text) track out of a `moov`
/// into a [`TrackHeader`]. The single-track [`parse_header`] reads only the first
/// `trak`; this walks them all (what an A/V `.mp4` carries). Tracks with an
/// unrecognized handler, or whose sample entry names a codec we do not read
/// (`tx3g` / `wvtt` / `stpp` text, `c608` / `c708` captions, and the video codecs
/// [`parse_video_entry`] handles are read, others decline) are skipped, not
/// errors; a malformed video / audio track fails the parse. Errors if no track is
/// forwardable.
pub(crate) fn parse_all_tracks(data: &[u8]) -> Result<Vec<TrackHeader>, G2gError> {
    let moov = find_box(data, b"moov").ok_or(G2gError::CapsMismatch)?;
    let mut tracks = Vec::new();
    for (kind, trak) in boxes(moov) {
        if kind != b"trak" {
            continue;
        }
        if let Some(header) = parse_trak(trak)? {
            tracks.push(header);
        }
    }
    if tracks.is_empty() {
        return Err(G2gError::CapsMismatch);
    }
    Ok(tracks)
}

/// Parse one `trak`. Returns `None` for a handler we do not forward (skip it),
/// `Err` for a malformed video / audio track.
fn parse_trak(trak: &[u8]) -> Result<Option<TrackHeader>, G2gError> {
    // tkhd v0: track_ID at payload offset 12 (4 version/flags + 8 times), then
    // width/height as 16.16 at 76/80.
    let tkhd = find_box(trak, b"tkhd").ok_or(G2gError::CapsMismatch)?;
    if tkhd.first() != Some(&0) {
        return Err(G2gError::CapsMismatch);
    }
    let track_id = be32(tkhd, 12)?;
    let width = be32(tkhd, 76)? >> 16;
    let height = be32(tkhd, 80)? >> 16;

    // mdhd v0: timescale at payload offset 12, duration at 16.
    let mdia = find_box(trak, b"mdia").ok_or(G2gError::CapsMismatch)?;
    let mdhd = find_box(mdia, b"mdhd").ok_or(G2gError::CapsMismatch)?;
    if mdhd.first() != Some(&0) {
        return Err(G2gError::CapsMismatch);
    }
    let timescale = be32(mdhd, 12)?;
    if timescale == 0 {
        return Err(G2gError::CapsMismatch);
    }

    // hdlr handler_type at payload offset 8 (4 version/flags + 4 pre_defined)
    // selects how to read the sample entry.
    let hdlr = find_box(mdia, b"hdlr").ok_or(G2gError::CapsMismatch)?;
    let handler = hdlr.get(8..12).ok_or(G2gError::CapsMismatch)?;

    let stbl = find_path(mdia, &[b"minf", b"stbl"]).ok_or(G2gError::CapsMismatch)?;
    let stsd = find_box(stbl, b"stsd").ok_or(G2gError::CapsMismatch)?;
    let entries = stsd.get(8..).ok_or(G2gError::CapsMismatch)?;

    let (kind, cenc) = match handler {
        // A video sample entry we have no reader for is skipped like an unknown
        // handler: one unsupported track must not fail the whole file.
        b"vide" => match parse_video_entry(entries, width, height)? {
            Some(entry) => entry,
            None => return Ok(None),
        },
        b"soun" => parse_audio_entry(entries, timescale)?,
        // A subtitle / timed-text handler (`text` = 3GPP/QuickTime timed text,
        // `sbtl` / `subt` = MP4 / ISO subtitle). Forwarded only when its sample
        // entry is one we de-frame (`tx3g`); an unrecognized text codec is skipped.
        b"text" | b"sbtl" | b"subt" => match parse_text_entry(entries) {
            Some(kind) => (kind, None),
            None => return Ok(None),
        },
        // A closed-caption handler (`clcp`), whose sample entry names the caption
        // carriage: `c608` / `c708` are read, anything else is skipped.
        b"clcp" => match parse_cc_entry(entries) {
            Some(kind) => (kind, None),
            None => return Ok(None),
        },
        _ => return Ok(None), // hint / metadata / unknown handler: not forwarded
    };
    Ok(Some(TrackHeader {
        track_id,
        timescale,
        kind,
        cenc: cenc_track(cenc, stbl)?,
        tags: parse_ilst_tags(trak),
    }))
}

/// The out-of-band config from an `av01` sample entry's `av1C` record (M779):
/// the trailing configOBUs (typically the sequence header), verbatim, as the
/// one "parameter set". Samples are plain low-overhead OBUs, so an empty
/// configOBUs just means the sequence header rides in-band.
fn parse_av1c_config(av01: &[u8]) -> Result<Vec<Vec<u8>>, G2gError> {
    let children = av01.get(78..).ok_or(G2gError::CapsMismatch)?;
    let av1c = find_box(children, b"av1C").ok_or(G2gError::CapsMismatch)?;
    // marker/version, profile/level, tier/depth flags, delay byte: 4 fixed
    // bytes, then configOBUs.
    let config = av1c.get(4..).unwrap_or_default();
    Ok(if config.is_empty() {
        Vec::new()
    } else {
        Vec::from([config.to_vec()])
    })
}

/// Read a video sample entry (`avc1` / `hvc1` / `hev1`, or the encrypted `encv`)
/// into a [`TrackKind::Video`] plus the cbcs `cenc` defaults for an encrypted
/// track. An `encv` carries the original codec config (`avcC` / `hvcC`) alongside
/// a `sinf` (original format + `cbcs` scheme + `tenc`), the same shape
/// [`parse_header`] reads. Returns `None` for an entry that is well-formed but
/// names a codec with no reader here (MJPEG in MP4, say), so that track is
/// skipped; malformed entry data is still an error.
fn parse_video_entry(
    entries: &[u8],
    width: u32,
    height: u32,
) -> Result<Option<(TrackKind, Option<CencDefaults>)>, G2gError> {
    let (codec, param_sets, cenc) = if let Some(avc1) = find_box(entries, b"avc1") {
        let children = avc1.get(78..).ok_or(G2gError::CapsMismatch)?;
        let (sps, pps) = parse_avcc(find_box(children, b"avcC").ok_or(G2gError::CapsMismatch)?)?;
        (VideoCodec::H264, Vec::from([sps, pps]), None)
    } else if let Some(hvc1) = find_box(entries, b"hvc1").or_else(|| find_box(entries, b"hev1")) {
        let children = hvc1.get(78..).ok_or(G2gError::CapsMismatch)?;
        let hvcc = find_box(children, b"hvcC").ok_or(G2gError::CapsMismatch)?;
        (VideoCodec::H265, parse_hvcc(hvcc)?, None)
    } else if let Some(mp4v) = find_box(entries, b"mp4v") {
        // MPEG-4 Part 2: the mp4v visual sample entry nests an esds carrying the
        // VOL header as its DecoderSpecificInfo. Confirm objectTypeIndication
        // 0x20 (Visual ISO/IEC 14496-2) so another mp4v-boxed codec is declined.
        let children = mp4v.get(78..).ok_or(G2gError::CapsMismatch)?;
        let esds = find_box(children, b"esds").ok_or(G2gError::CapsMismatch)?;
        let (oti, dsi) = parse_esds_video(esds)?;
        if oti != 0x20 {
            // Another codec in an mp4v box (ffmpeg writes MJPEG-in-MP4 this way,
            // objectTypeIndication 0x6C): a valid entry we have no reader for.
            return Ok(None);
        }
        // The VOL header is the single config blob (empty if carried in-band).
        let sets = if dsi.is_empty() {
            Vec::new()
        } else {
            Vec::from([dsi])
        };
        (VideoCodec::Mpeg4Part2, sets, None)
    } else if let Some(av01) = find_box(entries, b"av01") {
        (VideoCodec::Av1, parse_av1c_config(av01)?, None)
    } else if let Some(encv) = find_box(entries, b"encv") {
        let children = encv.get(78..).ok_or(G2gError::CapsMismatch)?;
        let sinf = find_box(children, b"sinf").ok_or(G2gError::CapsMismatch)?;
        let cenc = parse_sinf(sinf)?;
        let frma = find_box(sinf, b"frma").ok_or(G2gError::CapsMismatch)?;
        let (codec, param_sets) = match frma.get(0..4) {
            Some(b"avc1") => {
                let avcc = find_box(children, b"avcC").ok_or(G2gError::CapsMismatch)?;
                let (sps, pps) = parse_avcc(avcc)?;
                (VideoCodec::H264, Vec::from([sps, pps]))
            }
            Some(b"hvc1") | Some(b"hev1") => {
                let hvcc = find_box(children, b"hvcC").ok_or(G2gError::CapsMismatch)?;
                (VideoCodec::H265, parse_hvcc(hvcc)?)
            }
            _ => return Err(G2gError::CapsMismatch),
        };
        (codec, param_sets, Some(cenc))
    } else if visual_sample_entry(entries) {
        // A well-formed VisualSampleEntry naming a codec we do not read: decline
        // so the rest of the file still demuxes.
        return Ok(None);
    } else {
        return Err(G2gError::CapsMismatch);
    };
    Ok(Some((
        TrackKind::Video {
            codec,
            width,
            height,
            param_sets,
        },
        cenc,
    )))
}

/// Whether `entries` (an `stsd`'s sample-entry list) holds at least one entry
/// well-formed enough to be a VisualSampleEntry: a parseable box carrying its 78
/// fixed bytes. Truncated or garbage entry data is not, and fails the parse
/// rather than being silently skipped.
fn visual_sample_entry(entries: &[u8]) -> bool {
    boxes(entries).any(|(_, payload)| payload.len() >= 78)
}

/// Read an audio sample entry into a [`TrackKind::Audio`] plus the cbcs `cenc`
/// defaults for an encrypted track: AAC (`mp4a`/`esds`, or the encrypted
/// `enca`) or Opus (`Opus`/`dOps`, M767). The sample rate is the media
/// timescale (matching `Mp4AudioSrc`).
fn parse_audio_entry(
    entries: &[u8],
    timescale: u32,
) -> Result<(TrackKind, Option<CencDefaults>), G2gError> {
    // Opus: the `dOps` carries the same fields as an `OpusHead`, so it is
    // rebuilt into one and forwarded in-band (M791), the convention `OggDemux`
    // already uses, so the decoder trims the pre-skip and a remux keeps it. The
    // `dOps` OutputChannelCount is authoritative over the sample-entry
    // channelcount. A `dOps` that does not parse leaves the track without config
    // (playable, untrimmed) rather than failing the whole file.
    if let Some(opus) = find_box(entries, b"Opus") {
        let entry_channels = u16::from_be_bytes(
            opus.get(16..18)
                .ok_or(G2gError::CapsMismatch)?
                .try_into()
                .expect("2 bytes"),
        ) as u8;
        let config = find_box(opus.get(28..).unwrap_or(&[]), b"dOps")
            .and_then(opus_head_from_dops)
            .unwrap_or_default();
        let channels = parse_opus_head(&config)
            .map(|(c, _)| c)
            .unwrap_or(entry_channels);
        if channels == 0 {
            return Err(G2gError::CapsMismatch);
        }
        return Ok((
            TrackKind::Audio {
                format: AudioFormat::Opus,
                channels,
                sample_rate: timescale,
                config,
            },
            None,
        ));
    }
    let (entry, cenc) = match find_box(entries, b"mp4a") {
        Some(mp4a) => (mp4a, None),
        None => {
            let enca = find_box(entries, b"enca").ok_or(G2gError::CapsMismatch)?;
            let children = enca.get(28..).ok_or(G2gError::CapsMismatch)?;
            let sinf = find_box(children, b"sinf").ok_or(G2gError::CapsMismatch)?;
            (enca, Some(parse_sinf(sinf)?))
        }
    };
    // AudioSampleEntry: channelcount at offset 16, then 28 bytes before the esds.
    let channels = u16::from_be_bytes(
        entry
            .get(16..18)
            .ok_or(G2gError::CapsMismatch)?
            .try_into()
            .expect("2 bytes"),
    ) as u8;
    if channels == 0 {
        return Err(G2gError::CapsMismatch);
    }
    let esds = find_box(entry.get(28..).ok_or(G2gError::CapsMismatch)?, b"esds")
        .ok_or(G2gError::CapsMismatch)?;
    let config = parse_esds(esds)?;
    Ok((
        TrackKind::Audio {
            format: AudioFormat::Aac,
            channels,
            sample_rate: timescale,
            config,
        },
        cenc,
    ))
}

/// Recognize a timed-text sample entry and map it to the [`TrackKind::Text`] the
/// cue payload carries: `tx3g` (3GPP timed text) and `wvtt` (WebVTT-in-MP4) cues
/// de-frame to plain UTF-8, and `stpp` (TTML-in-MP4) samples are TTML documents.
/// An unrecognized text codec declines (the track is skipped, not an error).
fn parse_text_entry(entries: &[u8]) -> Option<TrackKind> {
    if find_box(entries, b"tx3g").is_some() {
        Some(TrackKind::Text {
            format: TextFormat::Utf8,
            sample: TextSampleFormat::Tx3g,
        })
    } else if find_box(entries, b"wvtt").is_some() {
        Some(TrackKind::Text {
            format: TextFormat::Utf8,
            sample: TextSampleFormat::Wvtt,
        })
    } else if find_box(entries, b"stpp").is_some() {
        Some(TrackKind::Text {
            format: TextFormat::Ttml,
            sample: TextSampleFormat::Stpp,
        })
    } else {
        None
    }
}

/// Recognize a closed-caption sample entry (QuickTime `c608` / `c708`, under a
/// `clcp` handler) and map it to the [`TrackKind::ClosedCaption`] carriage. An
/// unrecognized caption codec declines (the track is skipped, not an error).
fn parse_cc_entry(entries: &[u8]) -> Option<TrackKind> {
    if find_box(entries, b"c608").is_some() {
        Some(TrackKind::ClosedCaption {
            format: ClosedCaptionFormat::Cea608,
        })
    } else if find_box(entries, b"c708").is_some() {
        Some(TrackKind::ClosedCaption {
            format: ClosedCaptionFormat::Cea708,
        })
    } else {
        None
    }
}

/// De-frame one raw-caption sample to the `cc_data` triple stream a
/// [`Caps::ClosedCaption`](g2g_core::Caps::ClosedCaption) link carries, so a
/// container caption track and an SEI caption block feed the same decoder.
fn deframe_cc(raw: &[u8], format: ClosedCaptionFormat) -> Vec<u8> {
    match format {
        ClosedCaptionFormat::Cea608 => deframe_c608(raw),
        ClosedCaptionFormat::Cea708 => deframe_c708(raw),
        // Only the two carriages above are ever surfaced as a track, so a future
        // third yields no caption data rather than a guessed framing.
        _ => Vec::new(),
    }
}

/// De-frame a `c608` sample: a sequence of `cdat` (line-21 field 1) / `cdt2`
/// (field 2) atoms whose payloads are raw CEA-608 byte pairs, re-tagged as
/// `cc_data` triples (`cc_type` 0 for field 1, 1 for field 2). Atom sizes are
/// untrusted: a size below the 8-byte header, past the sample end, or an odd
/// payload length stops the scan, and an atom that is neither `cdat` nor `cdt2` is
/// skipped, so a malformed sample yields the pairs read so far rather than
/// panicking.
fn deframe_c608(raw: &[u8]) -> Vec<u8> {
    let mut triples: Vec<CcTriple> = Vec::new();
    let mut off = 0usize;
    while off.saturating_add(8) <= raw.len() {
        let size =
            u32::from_be_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as usize;
        if size < 8 || off.saturating_add(size) > raw.len() {
            break;
        }
        let cc_type = match &raw[off + 4..off + 8] {
            b"cdat" => Some(0u8),
            b"cdt2" => Some(1u8),
            _ => None,
        };
        let body = &raw[off + 8..off + size];
        if let Some(cc_type) = cc_type {
            if !body.len().is_multiple_of(2) {
                break;
            }
            for pair in body.as_chunks::<2>().0 {
                triples.push(CcTriple {
                    cc_type,
                    b0: pair[0],
                    b1: pair[1],
                });
            }
        }
        off += size;
    }
    write_cc_data(&triples)
}

/// De-frame a `c708` sample: `ccdp` atoms each holding a SMPTE ST 334-2 caption
/// distribution packet, unwrapped to the `cc_data` triples the CDP's ccdata
/// section carries. A CDP that fails to parse (bad identifier, length, or
/// checksum) contributes nothing; atom sizes are untrusted, so a bad one stops the
/// scan.
fn deframe_c708(raw: &[u8]) -> Vec<u8> {
    let mut triples: Vec<CcTriple> = Vec::new();
    let mut off = 0usize;
    while off.saturating_add(8) <= raw.len() {
        let size =
            u32::from_be_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as usize;
        if size < 8 || off.saturating_add(size) > raw.len() {
            break;
        }
        if &raw[off + 4..off + 8] == b"ccdp" {
            if let Some(mut got) = parse_cdp(&raw[off + 8..off + size]) {
                triples.append(&mut got);
            }
        }
        off += size;
    }
    write_cc_data(&triples)
}

/// Strip a 3GPP timed-text (`tx3g`) sample to its UTF-8 cue bytes: a 2-byte
/// big-endian text length, that many UTF-8 bytes, then optional style / modifier
/// boxes (ignored). A zero-length sample is the gap between cues and forwards as
/// an empty string, so a downstream overlay clears. The length is untrusted: a
/// prefix longer than the sample yields an empty cue rather than panicking.
fn deframe_tx3g(raw: &[u8]) -> Vec<u8> {
    let Some(len) = raw
        .get(0..2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]) as usize)
    else {
        return Vec::new();
    };
    raw.get(2..2usize.saturating_add(len))
        .map(<[u8]>::to_vec)
        .unwrap_or_default()
}

/// Strip a WebVTT-in-MP4 (`wvtt`) sample to its UTF-8 cue text (ISO 14496-30). A
/// sample is a sequence of boxes: each `vttc` (cue) carries a `payl` sub-box with
/// the UTF-8 payload, and a `vtte` (empty cue) carries nothing. The `payl`
/// payloads of every `vttc` in the sample concatenate (newline-separated) to the
/// text shown for the sample's time window; an empty / `vtte`-only sample forwards
/// as an empty string so a downstream overlay clears. Box sizes are untrusted: a
/// size below the 8-byte header or past the sample end stops the scan rather than
/// panicking. `sttg` cue settings (placement) are ignored for now.
fn deframe_wvtt(raw: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut off = 0usize;
    while off.saturating_add(8) <= raw.len() {
        let size =
            u32::from_be_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as usize;
        if size < 8 || off.saturating_add(size) > raw.len() {
            break;
        }
        let body = &raw[off + 8..off + size];
        if &raw[off + 4..off + 8] == b"vttc" {
            if let Some(payl) = find_box(body, b"payl") {
                if !out.is_empty() {
                    out.push(b'\n');
                }
                out.extend_from_slice(payl);
            }
        }
        off += size;
    }
    out
}

/// The decode state carried from a `moof` to its following `mdat`: the track id,
/// the per-sample `(size, pts_ns)`, the per-sample `duration_ns`, the per-sample
/// crypto for an encrypted track, and the fragment's start offset in the stream
/// (which keys the rotating content key).
type PendingFragment = (u32, Vec<(u32, u64)>, Vec<u64>, Vec<SampleCrypt>, u64);

/// Walk the `moof`+`mdat` fragments of a multi-track fMP4 and split every sample
/// out, keyed by its `track_ID`. Each `traf`'s `tfhd` names the track, so a
/// fragment is routed to the matching [`TrackHeader`]: video samples are
/// de-framed AVCC->Annex-B with a keyframe scan, audio samples pass through (each
/// is a sync sample). Fragments for an unknown `track_ID` are skipped.
///
/// An encrypted track's samples are decrypted in place via `decrypt` before
/// de-framing, using the crypto the fragment's sample auxiliary information and
/// `seig` groups resolve (see [`fragment_sample_crypt`]); an encrypted track with
/// no `decrypt` supplied fails loud (`CapsMismatch`), so a keyless build never
/// emits garbage. `base_offset` is where `data` sits in the source byte stream,
/// so a rotating key can be selected by fragment position. The multi-track analog
/// of [`parse_fragments`]; a non-conforming fragment is mis-split, not rejected,
/// the same caveat as there.
pub(crate) fn parse_fragments_multi(
    data: &[u8],
    tracks: &[TrackHeader],
    base_offset: u64,
    mut decrypt: Option<SampleDecrypt<'_>>,
) -> Result<Vec<(u32, Sample)>, G2gError> {
    let mut out = Vec::new();
    let mut pending: Option<PendingFragment> = None;

    for (kind, payload, box_at) in boxes_at(data) {
        match kind {
            b"moof" => {
                let traf = find_box(payload, b"traf").ok_or(G2gError::CapsMismatch)?;
                let tfhd = find_box(traf, b"tfhd").ok_or(G2gError::CapsMismatch)?;
                // tfhd: track_ID at payload offset 4 (after version/flags).
                let track_id = be32(tfhd, 4)?;
                let Some(track) = tracks.iter().find(|t| t.track_id == track_id) else {
                    // A fragment for a track we don't forward: hold the id so the
                    // following mdat is skipped, not mis-split into another track.
                    pending = Some((track_id, Vec::new(), Vec::new(), Vec::new(), 0));
                    continue;
                };
                let timescale = track.timescale;
                let tfdt = find_box(traf, b"tfdt").ok_or(G2gError::CapsMismatch)?;
                let base_time = match tfdt.first() {
                    Some(1) => be64(tfdt, 4)?,
                    Some(0) => be32(tfdt, 4)? as u64,
                    _ => return Err(G2gError::CapsMismatch),
                };
                let trun = find_box(traf, b"trun").ok_or(G2gError::CapsMismatch)?;
                let (default_duration, default_size) = tfhd_defaults(tfhd)?;
                let (sizes, durs) = parse_trun(trun, default_duration, default_size, data.len())?;
                // An encrypted track's per-sample IVs / subsample maps live in the
                // fragment's aux info, addressed from the start of this `moof`.
                let crypt = match &track.cenc {
                    Some(c) => {
                        let from_moof = data.get(box_at..).ok_or(G2gError::CapsMismatch)?;
                        fragment_sample_crypt(traf, from_moof, c, sizes.len())?
                    }
                    None => Vec::new(),
                };
                let mut t = base_time;
                let mut tagged = Vec::with_capacity(sizes.len());
                let mut durations = Vec::with_capacity(sizes.len());
                for (size, dur) in sizes.iter().zip(&durs) {
                    tagged.push((*size, timescale_to_ns(t, timescale)));
                    durations.push(timescale_to_ns(*dur as u64, timescale));
                    // base_time / durations are untrusted; saturate, never overflow.
                    t = t.saturating_add(*dur as u64);
                }
                pending = Some((
                    track_id,
                    tagged,
                    durations,
                    crypt,
                    base_offset.saturating_add(box_at as u64),
                ));
            }
            b"mdat" => {
                let Some((track_id, tagged, durations, crypt, frag_at)) = pending.take() else {
                    return Err(G2gError::CapsMismatch); // mdat without moof
                };
                let Some(track) = tracks.iter().find(|t| t.track_id == track_id) else {
                    continue; // skipped (unforwarded track), no samples emitted
                };
                let mut at = 0usize;
                for (i, (size, pts_ns)) in tagged.iter().enumerate() {
                    let raw = payload
                        .get(at..at + *size as usize)
                        .ok_or(G2gError::CapsMismatch)?;
                    at += *size as usize;
                    // Decrypt an encrypted track's sample in place before de-framing.
                    let owned;
                    let bytes: &[u8] = match (&track.cenc, crypt.get(i)) {
                        // A sample a `seig` group declares clear needs no key at
                        // all, so it never reaches the decryptor.
                        (Some(_), Some(sc)) if !sc.protected => raw,
                        (Some(_), Some(sc)) => {
                            let decrypt = decrypt.as_deref_mut().ok_or(G2gError::CapsMismatch)?;
                            let mut buf = raw.to_vec();
                            decrypt(sc, frag_at, &mut buf)?;
                            owned = buf;
                            &owned
                        }
                        // An encrypted track whose fragment described fewer samples
                        // than the `trun` holds: fail rather than emit ciphertext.
                        (Some(_), None) => return Err(G2gError::CapsMismatch),
                        (None, _) => raw,
                    };
                    let (annexb, keyframe) = match &track.kind {
                        TrackKind::Video { codec, .. } => {
                            let annexb = avcc_to_annexb(bytes)?;
                            let kf = contains_keyframe(&annexb, *codec);
                            (annexb, kf)
                        }
                        // Audio access units are stored verbatim; each is a sync point.
                        TrackKind::Audio { .. } => (bytes.to_vec(), true),
                        // A caption sample de-frames to the cc_data triple stream.
                        TrackKind::ClosedCaption { format } => (deframe_cc(bytes, *format), true),
                        // A timed-text cue is independent; strip its sample framing
                        // (tx3g / wvtt) or pass the whole TTML document (stpp).
                        TrackKind::Text { sample, .. } => {
                            let cue = match SampleFraming::of_text(*sample) {
                                SampleFraming::WvttText => deframe_wvtt(bytes),
                                SampleFraming::PassThrough => bytes.to_vec(),
                                _ => deframe_tx3g(bytes),
                            };
                            (cue, true)
                        }
                    };
                    out.push((
                        track_id,
                        Sample {
                            annexb,
                            pts_ns: *pts_ns,
                            dts_ns: *pts_ns,
                            duration_ns: durations[i],
                            keyframe,
                        },
                    ));
                }
            }
            _ => {}
        }
    }
    if pending.is_some() {
        return Err(G2gError::CapsMismatch); // trailing moof without mdat
    }
    Ok(out)
}

// `parse_hvcc` moved to `annexb` (shared with the no_std Matroska demuxer).
pub(crate) use crate::annexb::parse_hvcc;

// The avcC / parameter-set helpers moved to the ungated `annexb` module (M662,
// the no_std FLV demuxer shares them); re-exported so this module's users keep
// their import path.
pub(crate) use crate::annexb::{parse_avcc, prepend_param_sets, starts_with_param_set};

/// Walk the `moof`+`mdat` pairs in `data` and split every sample out of its
/// `mdat`, converting AVCC framing back to Annex-B. `codec` selects the IDR NAL
/// type used to flag keyframes (the seek snap points).
///
/// Assumes each `trun`'s samples are stored contiguously from the start of the
/// following `mdat` payload; the `trun` `data_offset` is not honored. This holds
/// for ffmpeg / CMAF output. A non-conforming fragment that positions its sample
/// data elsewhere in the `mdat` is mis-split, not rejected.
pub(crate) fn parse_fragments(
    data: &[u8],
    timescale: u32,
    codec: VideoCodec,
    cenc: Option<&CencTrack>,
    base_offset: u64,
    mut decrypt: Option<SampleDecrypt<'_>>,
) -> Result<Vec<Sample>, G2gError> {
    let mut samples = Vec::new();
    let mut pending: Option<Vec<(u32, u64)>> = None; // (size, pts_ns) per sample
    let mut durations: Vec<u64> = Vec::new();
    let mut pending_crypt: Vec<SampleCrypt> = Vec::new();
    let mut frag_at = 0u64;

    for (kind, payload, box_at) in boxes_at(data) {
        match kind {
            b"moof" => {
                let traf = find_box(payload, b"traf").ok_or(G2gError::CapsMismatch)?;
                let tfdt = find_box(traf, b"tfdt").ok_or(G2gError::CapsMismatch)?;
                let base_time = match tfdt.first() {
                    Some(1) => be64(tfdt, 4)?,
                    Some(0) => be32(tfdt, 4)? as u64,
                    _ => return Err(G2gError::CapsMismatch),
                };
                let trun = find_box(traf, b"trun").ok_or(G2gError::CapsMismatch)?;
                // The `tfhd` is where a CMAF fragment declares the sample duration
                // its `trun` then omits.
                let (default_duration, default_size) = match find_box(traf, b"tfhd") {
                    Some(tfhd) => tfhd_defaults(tfhd)?,
                    None => (0, 0),
                };
                let (sizes, durs) = parse_trun(trun, default_duration, default_size, data.len())?;
                let mut t = base_time;
                let mut tagged = Vec::with_capacity(sizes.len());
                durations.clear();
                for (size, dur) in sizes.iter().zip(&durs) {
                    tagged.push((*size, timescale_to_ns(t, timescale)));
                    durations.push(timescale_to_ns(*dur as u64, timescale));
                    // base_time and durations are untrusted; saturate the running
                    // decode time rather than overflow.
                    t = t.saturating_add(*dur as u64);
                }
                pending = Some(tagged);
                frag_at = base_offset.saturating_add(box_at as u64);
                pending_crypt = match cenc {
                    Some(c) => {
                        let from_moof = data.get(box_at..).ok_or(G2gError::CapsMismatch)?;
                        fragment_sample_crypt(traf, from_moof, c, sizes.len())?
                    }
                    None => Vec::new(),
                };
            }
            b"mdat" => {
                let Some(tagged) = pending.take() else {
                    return Err(G2gError::CapsMismatch); // mdat without moof
                };
                let mut at = 0usize;
                for (i, (size, pts_ns)) in tagged.iter().enumerate() {
                    let raw = payload
                        .get(at..at + *size as usize)
                        .ok_or(G2gError::CapsMismatch)?;
                    let annexb = if cenc.is_some() {
                        // Encrypted: decrypt the sample in place, then de-frame. A
                        // sample a `seig` group declares clear needs no key.
                        let sc = pending_crypt.get(i).ok_or(G2gError::CapsMismatch)?;
                        if sc.protected {
                            let decrypt = decrypt.as_deref_mut().ok_or(G2gError::CapsMismatch)?;
                            let mut buf = raw.to_vec();
                            decrypt(sc, frag_at, &mut buf)?;
                            avcc_to_annexb(&buf)?
                        } else {
                            avcc_to_annexb(raw)?
                        }
                    } else if matches!(codec, VideoCodec::Mpeg4Part2 | VideoCodec::Av1) {
                        // Raw elementary samples (start codes / low-overhead
                        // OBUs): no length-prefix de-framing.
                        raw.to_vec()
                    } else {
                        avcc_to_annexb(raw)?
                    };
                    let keyframe = contains_keyframe(&annexb, codec);
                    samples.push(Sample {
                        annexb,
                        pts_ns: *pts_ns,
                        dts_ns: *pts_ns,
                        duration_ns: durations[i],
                        keyframe,
                    });
                    at += *size as usize;
                }
            }
            _ => {}
        }
    }
    if pending.is_some() {
        return Err(G2gError::CapsMismatch); // trailing moof without mdat
    }
    Ok(samples)
}

/// Parse a *progressive* (non-fragmented) MP4: the classic `ftyp/moov/mdat`
/// layout where the `moov`'s sample tables (`stbl`) describe every sample's size
/// (`stsz`), decode duration (`stts`), composition offset (`ctts`), sync flag
/// (`stss`), and chunk layout (`stsc` + `stco`/`co64`), with the elementary data
/// sitting in `mdat` addressed by absolute file offset. This is what most tools
/// write by default (what `Mp4Src` falls back to when a file has no `moof`).
/// Returns the samples in decode order as Annex-B, the same shape
/// [`parse_fragments`] yields, so `Mp4Src::run` is identical downstream.
///
/// Single video track (the first `trak`, matching [`parse_header`]); the absolute
/// chunk offsets are read straight from `data`, so the `mdat` box framing (and
/// any 64-bit `largesize`) never matters.
pub(crate) fn parse_progressive(data: &[u8], timescale: u32) -> Result<Vec<Sample>, G2gError> {
    let moov = find_box(data, b"moov").ok_or(G2gError::CapsMismatch)?;
    let trak = find_box(moov, b"trak").ok_or(G2gError::CapsMismatch)?;
    let mdia = find_box(trak, b"mdia").ok_or(G2gError::CapsMismatch)?;
    let stbl = find_path(mdia, &[b"minf", b"stbl"]).ok_or(G2gError::CapsMismatch)?;
    // Single video track. H.264/H.265 samples are AVCC length-prefixed and
    // de-frame to Annex-B; MPEG-4 Part 2 (`mp4v`) samples are already raw
    // elementary stream (start codes) and AV1 (`av01`) samples plain
    // low-overhead OBUs, so those pass through verbatim. Absent an stsd,
    // default to Annex-B de-framing (the H.264/H.265 case).
    let framing = find_path(mdia, &[b"minf", b"stbl", b"stsd"])
        .and_then(|stsd| stsd.get(8..))
        .filter(|entries| {
            find_box(entries, b"mp4v").is_some() || find_box(entries, b"av01").is_some()
        })
        .map_or(SampleFraming::Video, |_| SampleFraming::PassThrough);
    parse_progressive_track(data, stbl, timescale, framing)
}

/// How a progressive track's stored samples map to the elementary bytes the
/// pipeline carries: video de-frames AVCC -> Annex-B, audio passes through
/// verbatim (each is a sync sample), and a `tx3g` timed-text sample strips its
/// 2-byte length prefix to the UTF-8 cue.
#[derive(Debug, Clone, Copy)]
enum SampleFraming {
    Video,
    PassThrough,
    Tx3gText,
    WvttText,
    ClosedCaption(ClosedCaptionFormat),
}

impl SampleFraming {
    /// The de-framing a track's [`TrackKind`] selects.
    fn of(kind: &TrackKind) -> Self {
        match kind {
            // MPEG-4 Part 2 samples are raw elementary stream (start codes) and
            // AV1 samples plain low-overhead OBUs, like the single-track case in
            // [`parse_progressive`]; H.264/H.265 are AVCC.
            TrackKind::Video {
                codec: VideoCodec::Mpeg4Part2 | VideoCodec::Av1,
                ..
            } => SampleFraming::PassThrough,
            TrackKind::Video { .. } => SampleFraming::Video,
            TrackKind::Audio { .. } => SampleFraming::PassThrough,
            TrackKind::Text { sample, .. } => Self::of_text(*sample),
            TrackKind::ClosedCaption { format } => SampleFraming::ClosedCaption(*format),
        }
    }

    /// The de-framing a timed-text sample format selects: a `stpp` sample is the
    /// whole TTML document (pass through), `tx3g` / `wvtt` strip their framing.
    fn of_text(sample: TextSampleFormat) -> Self {
        match sample {
            TextSampleFormat::Tx3g => SampleFraming::Tx3gText,
            TextSampleFormat::Wvtt => SampleFraming::WvttText,
            TextSampleFormat::Stpp => SampleFraming::PassThrough,
        }
    }
}

/// Parse the samples of one progressive track from its `stbl` sample tables,
/// addressing the elementary data in `data` by the absolute chunk offsets. The
/// per-track core shared by [`parse_progressive`] (single video track) and
/// [`parse_progressive_multi`] (every A/V / text track). `framing` selects how a
/// stored sample maps to the elementary bytes (AVCC->Annex-B, pass-through, or
/// tx3g text de-framing).
fn parse_progressive_track(
    data: &[u8],
    stbl: &[u8],
    timescale: u32,
    framing: SampleFraming,
) -> Result<Vec<Sample>, G2gError> {
    // stsz: per-sample sizes. A non-zero `default_size` means every sample is
    // that size (no table); otherwise a `sample_count`-long table follows.
    let stsz = find_box(stbl, b"stsz").ok_or(G2gError::CapsMismatch)?;
    let default_size = be32(stsz, 4)?;
    let sample_count = be32(stsz, 8)? as usize;
    // A sample needs at least one byte of media data, so the count cannot exceed
    // the file size. Reject a lying stsz before the per-sample allocations below
    // (the default_size branch fills the Vec, committing physical pages).
    if sample_count > data.len() {
        return Err(G2gError::CapsMismatch);
    }
    let sizes: Vec<u32> = if default_size != 0 {
        alloc::vec![default_size; sample_count]
    } else {
        (0..sample_count)
            .map(|i| be32(stsz, 12 + i * 4))
            .collect::<Result<_, _>>()?
    };

    // stts: decode durations, run-length encoded, expanded to one per sample.
    let stts = find_box(stbl, b"stts").ok_or(G2gError::CapsMismatch)?;
    let mut durations: Vec<u32> = Vec::with_capacity(sample_count);
    for e in 0..be32(stts, 4)? as usize {
        let cnt = be32(stts, 8 + e * 8)? as usize;
        let delta = be32(stts, 12 + e * 8)?;
        durations.resize(durations.len().saturating_add(cnt).min(sample_count), delta);
    }
    durations.resize(sample_count, 0);

    // ctts (optional): composition-time offsets for B-frame reorder. v0 carries
    // them unsigned, v1 signed; `pts = dts + ctts`. Absent => pts == dts.
    let ctts_offsets: Vec<i64> = match find_box(stbl, b"ctts") {
        Some(ctts) => {
            let signed = ctts.first() == Some(&1);
            let mut out: Vec<i64> = Vec::with_capacity(sample_count);
            for e in 0..be32(ctts, 4)? as usize {
                let cnt = be32(ctts, 8 + e * 8)? as usize;
                let raw = be32(ctts, 12 + e * 8)?;
                let off = if signed {
                    raw as i32 as i64
                } else {
                    raw as i64
                };
                let target = out.len().saturating_add(cnt).min(sample_count);
                out.resize(target, off);
            }
            out.resize(sample_count, 0);
            out
        }
        None => alloc::vec![0i64; sample_count],
    };

    // stco (32-bit) or co64 (64-bit): per-chunk file offsets.
    let chunk_offsets: Vec<u64> = if let Some(stco) = find_box(stbl, b"stco") {
        (0..be32(stco, 4)? as usize)
            .map(|c| be32(stco, 8 + c * 4).map(u64::from))
            .collect::<Result<_, _>>()?
    } else {
        let co64 = find_box(stbl, b"co64").ok_or(G2gError::CapsMismatch)?;
        (0..be32(co64, 4)? as usize)
            .map(|c| be64(co64, 8 + c * 8))
            .collect::<Result<_, _>>()?
    };

    // stsc: how many samples sit in each chunk, run-length over chunk ranges.
    // Resolve to a samples-per-chunk count for every chunk.
    let stsc = find_box(stbl, b"stsc").ok_or(G2gError::CapsMismatch)?;
    let stsc_n = be32(stsc, 4)? as usize;
    if stsc_n == 0 {
        return Err(G2gError::CapsMismatch);
    }
    let stsc_entry = |i: usize| -> Result<(u32, u32), G2gError> {
        Ok((be32(stsc, 8 + i * 12)?, be32(stsc, 12 + i * 12)?))
    };
    // Place each sample at its file offset: within a chunk samples are
    // contiguous, so offset advances by the running sample size.
    let mut sample_offsets: Vec<u64> = Vec::with_capacity(sample_count);
    let mut si = 0usize;
    'chunks: for (ci, &chunk_off) in chunk_offsets.iter().enumerate() {
        // The samples-per-chunk for this chunk is the last stsc entry whose
        // (1-based) first_chunk does not exceed it.
        let chunk_1based = (ci + 1) as u32;
        let mut spc = 0u32;
        for e in 0..stsc_n {
            let (first_chunk, samples_per_chunk) = stsc_entry(e)?;
            if first_chunk <= chunk_1based {
                spc = samples_per_chunk;
            } else {
                break;
            }
        }
        let mut at = chunk_off;
        for _ in 0..spc {
            if si >= sample_count {
                break 'chunks;
            }
            sample_offsets.push(at);
            at = at.saturating_add(sizes[si] as u64);
            si += 1;
        }
    }
    if sample_offsets.len() != sample_count {
        return Err(G2gError::CapsMismatch); // stsc/stco disagree with stsz
    }

    // stss: 1-based sync-sample numbers (ascending). Absent => every sample is a
    // sync sample (e.g. all-intra). Used as the keyframe flag (seek snap points).
    // Short-circuit on the first out-of-range entry (like stco/stsz) so a bogus
    // count fails loud instead of spinning the full untrusted range.
    let sync: Option<Vec<u32>> = match find_box(stbl, b"stss") {
        Some(stss) => Some(
            (0..be32(stss, 4)? as usize)
                .map(|i| be32(stss, 8 + i * 4))
                .collect::<Result<_, _>>()?,
        ),
        None => None,
    };

    let mut samples = Vec::with_capacity(sample_count);
    let mut dts: u64 = 0;
    for i in 0..sample_count {
        let off = sample_offsets[i] as usize;
        // `off` comes from an untrusted co64/stco chunk offset, so bound the end
        // with checked arithmetic (a u64 offset near usize::MAX would otherwise
        // overflow the `off + size` add and panic in debug).
        let end = off
            .checked_add(sizes[i] as usize)
            .ok_or(G2gError::CapsMismatch)?;
        let raw = data.get(off..end).ok_or(G2gError::CapsMismatch)?;
        let pts = (dts as i64).saturating_add(ctts_offsets[i]).max(0) as u64;
        let keyframe = match &sync {
            Some(list) => list.binary_search(&((i + 1) as u32)).is_ok(),
            None => true,
        };
        let annexb = match framing {
            SampleFraming::Video => avcc_to_annexb(raw)?,
            SampleFraming::PassThrough => raw.to_vec(),
            SampleFraming::Tx3gText => deframe_tx3g(raw),
            SampleFraming::WvttText => deframe_wvtt(raw),
            SampleFraming::ClosedCaption(format) => deframe_cc(raw, format),
        };
        samples.push(Sample {
            annexb,
            pts_ns: timescale_to_ns(pts, timescale),
            dts_ns: timescale_to_ns(dts, timescale),
            duration_ns: timescale_to_ns(durations[i] as u64, timescale),
            keyframe,
        });
        dts = dts.saturating_add(durations[i] as u64);
    }
    Ok(samples)
}

/// Parse a progressive (`moov`+`mdat`, no `moof`) multi-track file: every A/V /
/// text track's samples, keyed by `track_ID`, in track order. The progressive analog
/// of [`parse_fragments_multi`] (and the multi-track form of [`parse_progressive`]),
/// for files that carry several tracks in classic sample-table layout rather than
/// fragments. Each track's `stbl` is walked independently against its own
/// timescale and de-framing; tracks absent from `tracks` are skipped.
pub(crate) fn parse_progressive_multi(
    data: &[u8],
    tracks: &[TrackHeader],
) -> Result<Vec<(u32, Sample)>, G2gError> {
    let moov = find_box(data, b"moov").ok_or(G2gError::CapsMismatch)?;
    let mut out = Vec::new();
    for track in tracks {
        let Some(trak) = find_trak_by_id(moov, track.track_id) else {
            continue; // a track with no matching trak box
        };
        let mdia = find_box(trak, b"mdia").ok_or(G2gError::CapsMismatch)?;
        let stbl = find_path(mdia, &[b"minf", b"stbl"]).ok_or(G2gError::CapsMismatch)?;
        let framing = SampleFraming::of(&track.kind);
        for s in parse_progressive_track(data, stbl, track.timescale, framing)? {
            out.push((track.track_id, s));
        }
    }
    Ok(out)
}

/// Cap on the chapters a QuickTime chapter track yields. Its sample count is
/// the file's claim, and each sample costs a whole [`Chapter`], so the walk
/// stops rather than sizing the list from the file. (The `chpl` path needs no
/// cap: its count field is one byte.)
const MAX_CHAPTER_TRACK_CHAPTERS: usize = 4096;

/// The chapters a progressive MP4 declares (M1046): the QuickTime chapter text
/// track when the file has one and `data` holds its samples, else the Nero
/// `chpl` list in `moov/udta`. The text track is preferred because it carries a
/// duration per chapter, which `chpl` cannot express; a `moov`-first (faststart)
/// file being read before its `mdat` lands falls back to `chpl`.
pub(crate) fn parse_chapters(data: &[u8]) -> Vec<Chapter> {
    let Some(moov) = find_box(data, b"moov") else {
        return Vec::new();
    };
    let from_track = chapter_track_chapters(data, moov);
    if !from_track.is_empty() {
        return from_track;
    }
    parse_chpl(moov)
}

/// The chapters of the QuickTime chapter text track, empty when the file has
/// none. A media `trak` points at it with a `tref/chap` naming its `track_ID`;
/// that track's samples are the chapter titles, timed by its own sample table,
/// so one sample is one chapter running for that sample's duration.
fn chapter_track_chapters(data: &[u8], moov: &[u8]) -> Vec<Chapter> {
    let chapter_track_id = boxes(moov)
        .filter(|(kind, _)| *kind == b"trak")
        // A `tref/chap` may name several tracks; the first is the chapter track.
        .find_map(|(_, trak)| be32(find_path(trak, &[b"tref", b"chap"])?, 0).ok());
    let Some(chapter_track_id) = chapter_track_id else {
        return Vec::new();
    };
    let Some(trak) = find_trak_by_id(moov, chapter_track_id) else {
        return Vec::new();
    };
    let Some(mdia) = find_box(trak, b"mdia") else {
        return Vec::new();
    };
    // mdhd v0: timescale at payload offset 12, like every other track here.
    let Some(mdhd) = find_box(mdia, b"mdhd").filter(|m| m.first() == Some(&0)) else {
        return Vec::new();
    };
    let Ok(timescale) = be32(mdhd, 12) else {
        return Vec::new();
    };
    if timescale == 0 {
        return Vec::new();
    }
    let Some(stbl) = find_path(mdia, &[b"minf", b"stbl"]) else {
        return Vec::new();
    };
    let Ok(samples) = parse_progressive_track(data, stbl, timescale, SampleFraming::PassThrough)
    else {
        return Vec::new();
    };
    samples
        .iter()
        .take(MAX_CHAPTER_TRACK_CHAPTERS)
        .filter_map(|sample| {
            let title = chapter_sample_title(&sample.annexb)?;
            Some(Chapter {
                start_ns: sample.pts_ns,
                end_ns: Some(sample.pts_ns.saturating_add(sample.duration_ns)),
                title,
                language: None,
                sub_chapters: Vec::new(),
            })
        })
        .collect()
}

/// The title out of one QuickTime chapter-track sample: a big-endian `u16` byte
/// length, the text, then trailing atoms (an `encd` text encoding) this ignores.
/// The text is UTF-8 unless it opens with a byte-order mark, which is how
/// QuickTime and iTunes store a non-ASCII title.
fn chapter_sample_title(sample: &[u8]) -> Option<String> {
    let length = u16::from_be_bytes(sample.get(0..2)?.try_into().ok()?) as usize;
    let text = sample.get(2..2 + length)?;
    match text {
        [0xFE, 0xFF, rest @ ..] => decode_utf16(rest, true),
        [0xFF, 0xFE, rest @ ..] => decode_utf16(rest, false),
        _ => core::str::from_utf8(text).ok().map(String::from),
    }
}

/// Decode UTF-16 code units of the given endianness, or `None` for an odd byte
/// count or an unpaired surrogate.
fn decode_utf16(bytes: &[u8], big_endian: bool) -> Option<String> {
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    let units = bytes.as_chunks::<2>().0.iter().map(|pair| {
        let pair = [pair[0], pair[1]];
        if big_endian {
            u16::from_be_bytes(pair)
        } else {
            u16::from_le_bytes(pair)
        }
    });
    char::decode_utf16(units)
        .collect::<Result<String, _>>()
        .ok()
}

/// The `trak` box (payload) in `moov` whose `tkhd` carries `track_id`, or `None`.
fn find_trak_by_id(moov: &[u8], track_id: u32) -> Option<&[u8]> {
    boxes(moov)
        .filter(|(k, _)| *k == b"trak")
        .map(|(_, t)| t)
        .find(|trak| {
            find_box(trak, b"tkhd")
                .filter(|tkhd| tkhd.first() == Some(&0))
                .and_then(|tkhd| be32(tkhd, 12).ok())
                == Some(track_id)
        })
}

/// A `traf`'s `tfhd` `default_sample_duration` and `default_sample_size`
/// (ISO 14496-12 8.8.7), each `0` when the box does not declare it. A `trun` may
/// lean on either instead of repeating it per sample, which is what ffmpeg's
/// fragmented and CMAF output does (a constant-bitrate audio fragment often
/// carries neither field in its `trun`).
pub(crate) fn tfhd_defaults(tfhd: &[u8]) -> Result<(u32, u32), G2gError> {
    let flags = be32(tfhd, 0)? & 0x00FF_FFFF;
    // version/flags + track_ID, then the optional fields in declaration order.
    let mut at = 8usize;
    if flags & 0x01 != 0 {
        at += 8; // base_data_offset
    }
    if flags & 0x02 != 0 {
        at += 4; // sample_description_index
    }
    let duration = if flags & 0x08 != 0 {
        let d = be32(tfhd, at)?;
        at += 4;
        d
    } else {
        0
    };
    let size = if flags & 0x10 != 0 {
        be32(tfhd, at)?
    } else {
        0
    };
    Ok((duration, size))
}

/// `sample_flags` bit 16: the sample is *not* a sync sample (ISO/IEC 14496-12).
const NON_SYNC: u32 = 0x0001_0000;

/// Whether the first sample a `trun` describes is a sync sample, i.e. whether
/// the fragment it belongs to opens at a random access point. A `trun` that
/// states no flags at all (neither the first-sample override nor per-sample
/// flags) is taken to open at one, which is what a fragment without sample
/// dependency information implies.
pub(crate) fn trun_first_sample_is_sync(trun: &[u8]) -> Result<bool, G2gError> {
    let flags = be32(trun, 0)? & 0x00FF_FFFF;
    // version/flags + sample_count, then the optional fields in declaration order.
    let mut at = 8usize;
    if flags & 0x001 != 0 {
        at += 4; // data_offset
    }
    if flags & 0x004 != 0 {
        return Ok(be32(trun, at)? & NON_SYNC == 0); // first_sample_flags
    }
    if flags & 0x400 != 0 {
        if flags & 0x100 != 0 {
            at += 4; // sample_duration
        }
        if flags & 0x200 != 0 {
            at += 4; // sample_size
        }
        return Ok(be32(trun, at)? & NON_SYNC == 0);
    }
    Ok(true)
}

/// `trun` (v0 or v1); returns (sizes, durations). A `trun` that omits the
/// per-sample duration or size takes `default_duration` / `default_size` from its
/// `tfhd`; a duration neither carries is `0`, but a *size* neither carries is
/// fatal (the samples could not be split). v0 and v1 differ only in the sign of
/// the per-sample composition-time-offset field, which this skips (PTS is taken
/// from `tfdt` + decode-order durations and the decoder reorders), so both parse
/// identically. Real-world muxers (ffmpeg) emit v1 whenever B-frames are present.
pub(crate) fn parse_trun(
    trun: &[u8],
    default_duration: u32,
    default_size: u32,
    fragment_len: usize,
) -> Result<(Vec<u32>, Vec<u32>), G2gError> {
    match trun.first() {
        Some(0) | Some(1) => {}
        _ => return Err(G2gError::CapsMismatch), // unknown trun version
    }
    let flags = be32(trun, 0)? & 0x00FF_FFFF;
    if flags & 0x200 == 0 && default_size == 0 {
        return Err(G2gError::CapsMismatch); // no sample sizes anywhere
    }
    let count = be32(trun, 4)? as usize;
    let mut at = 8usize;
    if flags & 0x1 != 0 {
        at += 4; // data offset (sequential mdat split makes it redundant)
    }
    if flags & 0x4 != 0 {
        at += 4; // first sample flags
    }
    // Each sample consumes at least its 4-byte size plus the optional per-sample
    // fields, so an untrusted `count` cannot exceed the bytes that remain. Reject
    // a lying count before reserving capacity for it.
    let per_sample = if flags & 0x200 != 0 { 4 } else { 0 }
        + if flags & 0x100 != 0 { 4 } else { 0 }
        + if flags & 0x400 != 0 { 4 } else { 0 }
        + if flags & 0x800 != 0 { 4 } else { 0 };
    // Every field defaulted from the `tfhd` leaves the count unconstrained by the
    // `trun`, so bound it by the fragment instead (each sample occupies bytes
    // there); otherwise by the per-sample records the box can hold.
    let bound = trun
        .len()
        .saturating_sub(at)
        .checked_div(per_sample)
        .unwrap_or(fragment_len);
    if count > bound {
        return Err(G2gError::CapsMismatch);
    }
    let mut sizes = Vec::with_capacity(count);
    let mut durations = Vec::with_capacity(count);
    for _ in 0..count {
        let mut duration = default_duration;
        if flags & 0x100 != 0 {
            duration = be32(trun, at)?;
            at += 4;
        }
        if flags & 0x200 != 0 {
            sizes.push(be32(trun, at)?);
            at += 4;
        } else {
            sizes.push(default_size);
        }
        if flags & 0x400 != 0 {
            at += 4; // per-sample flags
        }
        if flags & 0x800 != 0 {
            at += 4; // composition time offset
        }
        durations.push(duration);
    }
    Ok((sizes, durations))
}

/// 4-byte-length-prefixed AVCC NALUs back to Annex-B start codes.
fn avcc_to_annexb(avcc: &[u8]) -> Result<Vec<u8>, G2gError> {
    let mut out = Vec::with_capacity(avcc.len());
    let mut at = 0usize;
    while at < avcc.len() {
        let len = be32(avcc, at)? as usize;
        at += 4;
        let nalu = avcc.get(at..at + len).ok_or(G2gError::CapsMismatch)?;
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nalu);
        at += len;
    }
    Ok(out)
}

/// Whether an Annex-B access unit contains an IDR picture (the keyframe a seek
/// snaps to). NAL boundaries are 4-byte start codes. H.264 IDR is NAL type 5;
/// H.265 IDR is 19/20.
pub(crate) fn contains_keyframe(annexb: &[u8], codec: VideoCodec) -> bool {
    // AV1 units are OBU streams, not start-coded NALs: the frame headers say.
    if codec == VideoCodec::Av1 {
        return crate::av1parse::av1_keyframe(annexb);
    }
    annexb
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == [0, 0, 0, 1])
        .any(|(at, _)| {
            annexb.get(at + 4).is_some_and(|&b| match codec {
                VideoCodec::H265 => matches!((b >> 1) & 0x3F, 19 | 20),
                _ => b & 0x1F == 5,
            })
        })
}

fn timescale_to_ns(t: u64, timescale: u32) -> u64 {
    t.saturating_mul(1_000_000_000) / timescale as u64
}

/// Incremental CMAF chunk splitter for a segment still being written (M888): fed
/// a segment response as its bytes arrive, it yields each complete chunk (the run
/// of boxes up to and including an `mdat`), so a low-latency packager's
/// `styp`+`moof`+`mdat` chunks flow downstream while the rest of the segment is
/// still on the wire. Every fed byte comes back out exactly once and in order
/// ([`CmafChunker::next_chunk`], then the [`CmafChunker::finish`] remainder), so
/// chunked and whole-response consumption hand a demuxer the same byte stream.
///
/// `max` bounds what may be held for one chunk: a box declaring more than that,
/// or a pending run growing past it, fails loud instead of buffering on an
/// attacker-chosen length.
#[cfg(any(test, feature = "dash"))]
#[derive(Debug)]
pub(crate) struct CmafChunker {
    /// Fed bytes not yet yielded.
    buf: Vec<u8>,
    /// How much of `buf`'s front is already framed as whole boxes (none of them
    /// an `mdat`), so a rescan resumes at the box still arriving.
    framed: usize,
    max: usize,
}

#[cfg(any(test, feature = "dash"))]
impl CmafChunker {
    pub(crate) fn new(max: usize) -> Self {
        Self {
            buf: Vec::new(),
            framed: 0,
            max,
        }
    }

    /// Take newly arrived body bytes.
    pub(crate) fn feed(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        if self.buf.len().saturating_add(bytes.len()) > self.max {
            return Err(G2gError::CapsMismatch);
        }
        self.buf.extend_from_slice(bytes);
        Ok(())
    }

    /// The next complete chunk, or `None` while it is still arriving. Call until
    /// it yields `None` after every [`CmafChunker::feed`].
    pub(crate) fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, G2gError> {
        loop {
            let Some(total) = next_box_len(&self.buf[self.framed..])? else {
                return Ok(None);
            };
            if total > self.max {
                return Err(G2gError::CapsMismatch);
            }
            let end = self.framed.saturating_add(total);
            if self.buf.len() < end {
                return Ok(None); // the rest of this box is still in flight
            }
            let mdat = &self.buf[self.framed + 4..self.framed + 8] == b"mdat";
            self.framed = end;
            if mdat {
                let chunk: Vec<u8> = self.buf.drain(..self.framed).collect();
                self.framed = 0;
                return Ok(Some(chunk));
            }
        }
    }

    /// Whatever is left when the body ends: a chunkless response (an init
    /// segment's `ftyp`+`moov`), a trailing box after the last `mdat`, or the tail
    /// of a truncated transfer. Verbatim, so what reaches the demuxer is the
    /// response body either way.
    pub(crate) fn finish(&mut self) -> Option<Vec<u8>> {
        self.framed = 0;
        (!self.buf.is_empty()).then(|| core::mem::take(&mut self.buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// The chunk splitter cuts a CMAF segment after every `mdat` whatever the
    /// arrival boundaries (fed one byte at a time here, the worst case), hands
    /// every fed byte back exactly once and in order, and leaves a trailing box
    /// for `finish`.
    #[test]
    fn cmaf_chunker_splits_after_every_mdat() {
        use crate::mp4box::mp4_box;
        let styp = mp4_box(b"styp", b"cmfscmff");
        let chunk = |n: u8| [mp4_box(b"moof", &[n; 12]), mp4_box(b"mdat", &[n; 20])].concat();
        let mfra = mp4_box(b"mfra", &[0u8; 8]);
        let segment = [styp.clone(), chunk(1), chunk(2), chunk(3), mfra.clone()].concat();

        let mut chunker = CmafChunker::new(1024);
        let mut got = Vec::new();
        for byte in &segment {
            chunker.feed(&[*byte]).expect("under the cap");
            while let Some(c) = chunker.next_chunk().expect("well-formed boxes") {
                got.push(c);
            }
        }
        assert_eq!(
            got,
            vec![[styp, chunk(1)].concat(), chunk(2), chunk(3)],
            "the leading styp rides the first chunk, then one chunk per moof+mdat"
        );
        assert_eq!(
            chunker.finish(),
            Some(mfra),
            "the trailing box comes out last"
        );
        assert_eq!(chunker.finish(), None, "and only once");
        assert_eq!(
            got.concat().len() + 16,
            segment.len(),
            "no byte was dropped"
        );
    }

    /// The cap is the splitter's DoS bound: a box declaring more than it fails
    /// before the bytes are waited on, and so does a pending run growing past it.
    #[test]
    fn cmaf_chunker_bounds_declared_and_pending_sizes() {
        let mut chunker = CmafChunker::new(64);
        // An mdat claiming 1 MiB: rejected on the header, not buffered toward.
        chunker
            .feed(&[0x00, 0x10, 0x00, 0x00, b'm', b'd', b'a', b't'])
            .expect("8 bytes fit");
        assert!(
            chunker.next_chunk().is_err(),
            "over-cap box size fails loud"
        );

        let mut chunker = CmafChunker::new(64);
        assert!(
            chunker.feed(&[0u8; 65]).is_err(),
            "over-cap pending run fails"
        );
    }

    #[test]
    fn avcc_to_annexb_round_trips_length_prefixes() {
        let avcc = [0, 0, 0, 3, 0x67, 1, 2, 0, 0, 0, 2, 0x65, 3];
        let annexb = avcc_to_annexb(&avcc).unwrap();
        assert_eq!(annexb, vec![0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x65, 3]);
        // a truncated NALU fails loud
        assert!(avcc_to_annexb(&[0, 0, 0, 9, 1]).is_err());
    }

    #[test]
    fn trun_parser_reads_the_writer_profile() {
        // flags 0x701: data offset + duration + size + flags, one sample.
        let mut p = vec![0u8, 0, 7, 1];
        p.extend_from_slice(&1u32.to_be_bytes()); // count
        p.extend_from_slice(&120u32.to_be_bytes()); // data offset
        p.extend_from_slice(&3000u32.to_be_bytes()); // duration
        p.extend_from_slice(&77u32.to_be_bytes()); // size
        p.extend_from_slice(&0x0200_0000u32.to_be_bytes()); // sample flags
        let (sizes, durs) = parse_trun(&p, 0, 0, 0).unwrap();
        assert_eq!(sizes, vec![77]);
        assert_eq!(durs, vec![3000]);
    }

    #[test]
    fn timescale_conversion_inverts_the_sink() {
        assert_eq!(timescale_to_ns(90_000, 90_000), 1_000_000_000);
        assert_eq!(timescale_to_ns(2999, 90_000), 33_322_222);
    }

    #[test]
    fn sps_detection_reads_the_first_nal_type() {
        assert!(starts_with_param_set(
            &[0, 0, 0, 1, 0x67, 0xAA],
            VideoCodec::H264
        ));
        assert!(!starts_with_param_set(
            &[0, 0, 0, 1, 0x65, 0xAA],
            VideoCodec::H264
        ));
        assert!(starts_with_param_set(
            &[0, 0, 0, 1, 0x40, 0x01],
            VideoCodec::H265
        ));
        assert!(!starts_with_param_set(
            &[0, 0, 0, 1, 0x26, 0x01],
            VideoCodec::H265
        ));
    }

    #[test]
    fn hvcc_parser_recovers_arrays_in_order() {
        let vps: &[u8] = &[0x40, 0x01, 0xAA];
        let sps: &[u8] = &[0x42, 0x01, 0xBB, 0xCC];
        let pps: &[u8] = &[0x44, 0x01, 0xDD];
        let mut p = vec![0u8; 22];
        p[0] = 1; // configuration version
        p.push(3); // numOfArrays at offset 22
        for (ty, nalu) in [(32u8, vps), (33u8, sps), (34u8, pps)] {
            p.push(0x80 | ty);
            p.extend_from_slice(&1u16.to_be_bytes());
            p.extend_from_slice(&(nalu.len() as u16).to_be_bytes());
            p.extend_from_slice(nalu);
        }
        let sets = parse_hvcc(&p).unwrap();
        assert_eq!(sets, vec![vps.to_vec(), sps.to_vec(), pps.to_vec()]);
    }

    /// A `trun` v1 (signed composition offsets, what ffmpeg writes for B-frame
    /// streams) parses the same as v0: the cts field is skipped either way, so
    /// sizes and durations come out identically. Guards the version gate.
    #[test]
    fn parse_trun_accepts_v0_and_v1() {
        // flags 0x301: data-offset(0x1) + sample-duration(0x100) + sample-size(0x200).
        let build = |version: u8| {
            let mut t = alloc::vec![version, 0x00, 0x03, 0x01];
            t.extend_from_slice(&2u32.to_be_bytes()); // sample count
            t.extend_from_slice(&0u32.to_be_bytes()); // data offset
            for (dur, size) in [(33u32, 1000u32), (33, 1200)] {
                t.extend_from_slice(&dur.to_be_bytes());
                t.extend_from_slice(&size.to_be_bytes());
            }
            t
        };
        let v0 = parse_trun(&build(0), 0, 0, 0).expect("v0 parses");
        let v1 = parse_trun(&build(1), 0, 0, 0).expect("v1 parses");
        assert_eq!(v0, (alloc::vec![1000, 1200], alloc::vec![33, 33]));
        assert_eq!(v0, v1, "v0 and v1 parse identically (cts field is skipped)");
    }

    #[test]
    fn parse_trun_rejects_oversized_count() {
        // flags 0x201 (data-offset + sizes), a huge count but only one sample's
        // worth of bytes: reject instead of reserving gigabytes.
        let mut t = alloc::vec![0u8, 0x00, 0x02, 0x01];
        t.extend_from_slice(&u32::MAX.to_be_bytes()); // count
        t.extend_from_slice(&0u32.to_be_bytes()); // data offset
        t.extend_from_slice(&16u32.to_be_bytes()); // a single sample size
        assert!(parse_trun(&t, 0, 0, 0).is_err());
    }

    /// A minimal progressive (`moov` + `mdat`, no `moof`) file with two AVCC
    /// samples in one chunk parses to two Annex-B samples with the right sizes,
    /// timing, and sync flag (sample 1 only, from `stss`).
    #[test]
    fn parse_progressive_reads_stbl_sample_tables() {
        use crate::mp4box::{full_box, mp4_box};
        // Two AVCC samples: [len=2][0x65 IDR][0xAA], [len=2][0x41 non-IDR][0xBB].
        let mut mdat_body = Vec::new();
        for nal in [[0x65u8, 0xAA], [0x41, 0xBB]] {
            mdat_body.extend_from_slice(&2u32.to_be_bytes());
            mdat_body.extend_from_slice(&nal);
        }
        let sample_size = 6u32; // 4-byte length prefix + 2-byte NAL

        let stsz = {
            let mut b = alloc::vec![0u8; 8]; // default_size = 0, then count
            b[4..8].copy_from_slice(&2u32.to_be_bytes());
            b.extend_from_slice(&sample_size.to_be_bytes());
            b.extend_from_slice(&sample_size.to_be_bytes());
            full_box(b"stsz", 0, 0, &b)
        };
        let stts = {
            let mut b = 1u32.to_be_bytes().to_vec(); // one run
            b.extend_from_slice(&2u32.to_be_bytes()); // count
            b.extend_from_slice(&1000u32.to_be_bytes()); // delta
            full_box(b"stts", 0, 0, &b)
        };
        let stsc = {
            let mut b = 1u32.to_be_bytes().to_vec(); // one entry
            b.extend_from_slice(&1u32.to_be_bytes()); // first_chunk = 1
            b.extend_from_slice(&2u32.to_be_bytes()); // samples_per_chunk = 2
            b.extend_from_slice(&1u32.to_be_bytes()); // sample_desc_index
            full_box(b"stsc", 0, 0, &b)
        };
        let stss = {
            let mut b = 1u32.to_be_bytes().to_vec(); // one sync sample
            b.extend_from_slice(&1u32.to_be_bytes()); // sample number 1 (1-based)
            full_box(b"stss", 0, 0, &b)
        };
        // stco offset is filled once the moov length is known (it is constant in
        // the offset value, so a placeholder build gives the right length).
        let build = |chunk_off: u32| {
            let mut stco_body = 1u32.to_be_bytes().to_vec();
            stco_body.extend_from_slice(&chunk_off.to_be_bytes());
            let stco = full_box(b"stco", 0, 0, &stco_body);
            let mut stbl = Vec::new();
            for t in [&stsz, &stts, &stsc, &stco, &stss] {
                stbl.extend_from_slice(t);
            }
            let stbl = mp4_box(b"stbl", &stbl);
            let minf = mp4_box(b"minf", &stbl);
            let mdia = mp4_box(b"mdia", &minf);
            let trak = mp4_box(b"trak", &mdia);
            mp4_box(b"moov", &trak)
        };
        let moov_len = build(0).len();
        let chunk_off = (moov_len + 8) as u32; // mdat payload starts after its header
        let mut file = build(chunk_off);
        file.extend_from_slice(&mp4_box(b"mdat", &mdat_body));

        let samples = parse_progressive(&file, 1000).expect("progressive parse");
        assert_eq!(samples.len(), 2);
        // AVCC length prefixes became Annex-B start codes.
        assert_eq!(samples[0].annexb, alloc::vec![0, 0, 0, 1, 0x65, 0xAA]);
        assert_eq!(samples[1].annexb, alloc::vec![0, 0, 0, 1, 0x41, 0xBB]);
        assert!(samples[0].keyframe, "sample 1 is in stss");
        assert!(!samples[1].keyframe, "sample 2 is not in stss");
        assert_eq!(samples[0].pts_ns, 0);
        assert_eq!(samples[1].pts_ns, 1_000_000_000); // 1000 / timescale 1000 s
    }

    /// An `mp4v` visual sample entry whose esds objectTypeIndication is 0x20
    /// (Visual ISO/IEC 14496-2) parses to `VideoCodec::Mpeg4Part2` with the VOL
    /// header (the DecoderSpecificInfo) as its single parameter set; a different
    /// objectTypeIndication in an mp4v box is rejected so another codec is not
    /// mistagged as MPEG-4 Part 2.
    #[test]
    fn parse_header_reads_mp4v_mpeg4_part2() {
        use crate::mp4box::{full_box, mp4_box};

        let descriptor = |tag: u8, body: &[u8]| {
            let mut v = vec![tag, body.len() as u8];
            v.extend_from_slice(body);
            v
        };
        // esds with a chosen objectTypeIndication (first DCD byte) and DSI.
        let esds = |oti: u8, dsi_bytes: &[u8]| {
            let dsi = descriptor(0x05, dsi_bytes);
            let mut dcd_body = vec![0u8; 13];
            dcd_body[0] = oti;
            dcd_body.extend_from_slice(&dsi);
            let dcd = descriptor(0x04, &dcd_body);
            let mut es_body = vec![0u8; 3];
            es_body.extend_from_slice(&dcd);
            full_box(b"esds", 0, 0, &descriptor(0x03, &es_body))
        };
        let tkhd = {
            let mut c = vec![0u8; 80];
            c[72..76].copy_from_slice(&(320u32 << 16).to_be_bytes());
            c[76..80].copy_from_slice(&(240u32 << 16).to_be_bytes());
            full_box(b"tkhd", 0, 0, &c)
        };
        let mdhd = {
            let mut c = vec![0u8; 16];
            c[8..12].copy_from_slice(&1000u32.to_be_bytes()); // timescale
            c[12..16].copy_from_slice(&1000u32.to_be_bytes()); // duration
            full_box(b"mdhd", 0, 0, &c)
        };
        // The VOL header carries its own 3-byte start code (VOS start code 0xB0).
        let vol: &[u8] = &[0x00, 0x00, 0x01, 0xB0, 0x08];
        let build = |oti: u8| {
            let mp4v = {
                let mut p = vec![0u8; 78];
                p.extend_from_slice(&esds(oti, vol));
                mp4_box(b"mp4v", &p)
            };
            let mut stsd_body = 1u32.to_be_bytes().to_vec(); // entry count
            stsd_body.extend_from_slice(&mp4v);
            let stsd = full_box(b"stsd", 0, 0, &stsd_body);
            let stbl = mp4_box(b"stbl", &stsd);
            let minf = mp4_box(b"minf", &stbl);
            let mdia = mp4_box(b"mdia", &[mdhd.clone(), minf].concat());
            let trak = mp4_box(b"trak", &[tkhd.clone(), mdia].concat());
            mp4_box(b"moov", &trak)
        };

        let header = parse_header(&build(0x20)).expect("mp4v/esds OTI 0x20 parses");
        assert_eq!(header.codec, VideoCodec::Mpeg4Part2);
        assert_eq!(header.width, 320);
        assert_eq!(header.height, 240);
        assert_eq!(
            header.param_sets,
            vec![vol.to_vec()],
            "VOL header is the config"
        );

        assert!(
            parse_header(&build(0x21)).is_err(),
            "non-Visual objectTypeIndication rejected"
        );
    }

    /// MPEG-4 Part 2 config framing differs from H.264/H.265: the VOL header
    /// already carries 3-byte start codes, so it is prepended verbatim (no
    /// injected 4-byte Annex-B start code), and an access unit already opening
    /// with a VOS/VOL start code counts as carrying its parameter set.
    #[test]
    fn mpeg4_part2_config_framing() {
        let vol: Vec<u8> = vec![0x00, 0x00, 0x01, 0xB0, 0x08];
        let frame: Vec<u8> = vec![0x00, 0x00, 0x01, 0xB6, 0x00]; // I-VOP

        let out = prepend_param_sets(&frame, core::slice::from_ref(&vol), VideoCodec::Mpeg4Part2);
        assert_eq!(
            out,
            [vol.clone(), frame.clone()].concat(),
            "VOL prepended verbatim"
        );

        // H.264 param sets still get a 4-byte start code injected.
        let sps: Vec<u8> = vec![0x67, 0x42];
        let h264 = prepend_param_sets(&frame, core::slice::from_ref(&sps), VideoCodec::H264);
        assert_eq!(
            h264[..4],
            [0, 0, 0, 1],
            "H.264 set gets an Annex-B start code"
        );

        assert!(
            starts_with_param_set(&vol, VideoCodec::Mpeg4Part2),
            "VOS start code detected"
        );
        assert!(
            !starts_with_param_set(&frame, VideoCodec::Mpeg4Part2),
            "an I-VOP is not config"
        );
    }

    /// A two-track fragmented file (an H.264 `vide` trak + an AAC `soun` trak,
    /// then one `moof`+`mdat` per track) parses to two [`TrackHeader`]s with the
    /// right codec/geometry/timescale, and [`parse_fragments_multi`] routes each
    /// fragment to its `track_ID`, de-framing video to Annex-B and passing audio
    /// through. Builds the boxes directly so the test stays a lib unit test.
    #[test]
    fn parse_all_tracks_and_fragments_route_by_track_id() {
        use crate::mp4box::{ftyp, full_box, mp4_box};

        // --- box builders the parser's offsets expect ---------------------
        // tkhd v0: track_ID at payload offset 12, width/height 16.16 at 76/80.
        let tkhd = |track_id: u32, w: u32, h: u32| {
            let mut c = alloc::vec![0u8; 80]; // content after the version/flags
            c[8..12].copy_from_slice(&track_id.to_be_bytes());
            c[72..76].copy_from_slice(&(w << 16).to_be_bytes());
            c[76..80].copy_from_slice(&(h << 16).to_be_bytes());
            full_box(b"tkhd", 0, 0, &c)
        };
        // mdhd v0: timescale at payload offset 12, duration at 16.
        let mdhd = |timescale: u32, duration: u32| {
            let mut c = alloc::vec![0u8; 16];
            c[8..12].copy_from_slice(&timescale.to_be_bytes());
            c[12..16].copy_from_slice(&duration.to_be_bytes());
            full_box(b"mdhd", 0, 0, &c)
        };
        // hdlr: handler_type at payload offset 8.
        let hdlr = |handler: &[u8; 4]| {
            let mut c = alloc::vec![0u8; 20];
            c[4..8].copy_from_slice(handler);
            full_box(b"hdlr", 0, 0, &c)
        };
        let descriptor = |tag: u8, body: &[u8]| {
            let mut v = alloc::vec![tag, body.len() as u8];
            v.extend_from_slice(body);
            v
        };
        let esds = |asc: &[u8]| {
            let dsi = descriptor(0x05, asc);
            let mut dcd_body = alloc::vec![0u8; 13];
            dcd_body.extend_from_slice(&dsi);
            let dcd = descriptor(0x04, &dcd_body);
            let mut es_body = alloc::vec![0u8; 3];
            es_body.extend_from_slice(&dcd);
            let es = descriptor(0x03, &es_body);
            full_box(b"esds", 0, 0, &es)
        };
        let avcc = |sps: &[u8], pps: &[u8]| {
            let mut p = alloc::vec![0u8; 5]; // fixed config bytes
            p.push(0xE1); // reserved bits + sps_count = 1
            p.extend_from_slice(&(sps.len() as u16).to_be_bytes());
            p.extend_from_slice(sps);
            p.push(1); // pps_count
            p.extend_from_slice(&(pps.len() as u16).to_be_bytes());
            p.extend_from_slice(pps);
            mp4_box(b"avcC", &p)
        };
        let stsd = |entry: &[u8]| {
            let mut p = 1u32.to_be_bytes().to_vec(); // entry count
            p.extend_from_slice(entry);
            full_box(b"stsd", 0, 0, &p)
        };
        let trak = |tkhd: &[u8], mdhd: &[u8], hdlr: &[u8], stsd: &[u8]| {
            let minf = mp4_box(b"minf", &mp4_box(b"stbl", stsd));
            let mdia = mp4_box(b"mdia", &[mdhd, hdlr, &minf].concat());
            mp4_box(b"trak", &[tkhd, &mdia].concat())
        };

        let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1e];
        let pps: &[u8] = &[0x68, 0xce];
        let asc: &[u8] = &[0x12, 0x10];

        // avc1 sample entry: 78 fixed bytes then the avcC.
        let avc1 = {
            let mut p = alloc::vec![0u8; 78];
            p.extend_from_slice(&avcc(sps, pps));
            mp4_box(b"avc1", &p)
        };
        // mp4a sample entry: channelcount at offset 16, then 28 bytes before esds.
        let mp4a = {
            let mut p = alloc::vec![0u8; 28];
            p[16..18].copy_from_slice(&2u16.to_be_bytes());
            p.extend_from_slice(&esds(asc));
            mp4_box(b"mp4a", &p)
        };

        let video_trak = trak(
            &tkhd(1, 320, 240),
            &mdhd(90_000, 90_000), // 1 s
            &hdlr(b"vide"),
            &stsd(&avc1),
        );
        let audio_trak = trak(
            &tkhd(2, 0, 0),
            &mdhd(48_000, 48_000), // 1 s
            &hdlr(b"soun"),
            &stsd(&mp4a),
        );
        let moov = mp4_box(b"moov", &[video_trak, audio_trak].concat());

        // --- fragments: one per track, keyed by track_ID via tfhd ---------
        let tfhd = |track_id: u32| full_box(b"tfhd", 0, 0, &track_id.to_be_bytes());
        let tfdt = |base: u64| full_box(b"tfdt", 1, 0, &base.to_be_bytes());
        let trun = |dur: u32, size: u32| {
            let mut p = 1u32.to_be_bytes().to_vec(); // sample count
            p.extend_from_slice(&0u32.to_be_bytes()); // data offset
            p.extend_from_slice(&dur.to_be_bytes());
            p.extend_from_slice(&size.to_be_bytes());
            full_box(b"trun", 0, 0x000301, &p) // data-offset | duration | size
        };
        let moof = |track_id: u32, dur: u32, size: u32| {
            let traf = mp4_box(
                b"traf",
                &[tfhd(track_id), tfdt(0), trun(dur, size)].concat(),
            );
            mp4_box(b"moof", &traf)
        };

        // Video sample: one AVCC NALU (4-byte length + IDR), de-frames to Annex-B.
        let video_sample = alloc::vec![0, 0, 0, 2, 0x65, 0xAA];
        let audio_sample = alloc::vec![0x01u8, 0x02, 0x03]; // raw AAC, passed through

        let mut file = ftyp();
        file.extend_from_slice(&moov);
        file.extend_from_slice(&moof(1, 3000, video_sample.len() as u32));
        file.extend_from_slice(&mp4_box(b"mdat", &video_sample));
        file.extend_from_slice(&moof(2, 1024, audio_sample.len() as u32));
        file.extend_from_slice(&mp4_box(b"mdat", &audio_sample));

        // --- assert: two tracks parsed with the right kinds ---------------
        let tracks = parse_all_tracks(&file).expect("two-track parse");
        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0].track_id, 1);
        match &tracks[0].kind {
            TrackKind::Video {
                codec,
                width,
                height,
                param_sets,
            } => {
                assert_eq!(*codec, VideoCodec::H264);
                assert_eq!((*width, *height), (320, 240));
                assert_eq!(param_sets, &alloc::vec![sps.to_vec(), pps.to_vec()]);
            }
            other => panic!("track 0 should be video, got {other:?}"),
        }
        assert_eq!(tracks[1].track_id, 2);
        match &tracks[1].kind {
            TrackKind::Audio {
                format,
                channels,
                sample_rate,
                config: got,
            } => {
                assert_eq!(*format, AudioFormat::Aac);
                assert_eq!(*channels, 2);
                assert_eq!(*sample_rate, 48_000);
                assert_eq!(got, asc);
            }
            other => panic!("track 1 should be audio, got {other:?}"),
        }

        // --- assert: fragments route to their track and de-frame correctly -
        let samples = parse_fragments_multi(&file, &tracks, 0, None).expect("fragment routing");
        assert_eq!(samples.len(), 2);
        let (vid_id, vid) = &samples[0];
        assert_eq!(*vid_id, 1);
        assert_eq!(vid.annexb, alloc::vec![0, 0, 0, 1, 0x65, 0xAA]);
        assert!(vid.keyframe, "IDR is a keyframe");
        let (aud_id, aud) = &samples[1];
        assert_eq!(*aud_id, 2);
        assert_eq!(aud.annexb, audio_sample, "audio passes through verbatim");
        assert!(aud.keyframe, "every audio AU is a sync sample");
    }

    /// A progressive (`moov`+`mdat`, no `moof`) two-track file (an H.264 `vide`
    /// trak + an AAC `soun` trak, each with classic `stbl` sample tables sharing a
    /// single `mdat`) parses to two tracks and `parse_progressive_multi` routes
    /// each track's samples by `track_ID`, de-framing video to Annex-B and passing
    /// audio through. The progressive analog of the fragmented test above.
    #[test]
    fn parse_progressive_multi_routes_each_track_by_id() {
        use crate::mp4box::{full_box, mp4_box};

        // --- shared leaf box builders (same offsets the parser reads) ------
        let tkhd = |track_id: u32, w: u32, h: u32| {
            let mut c = alloc::vec![0u8; 80];
            c[8..12].copy_from_slice(&track_id.to_be_bytes());
            c[72..76].copy_from_slice(&(w << 16).to_be_bytes());
            c[76..80].copy_from_slice(&(h << 16).to_be_bytes());
            full_box(b"tkhd", 0, 0, &c)
        };
        let mdhd = |timescale: u32| {
            let mut c = alloc::vec![0u8; 16];
            c[8..12].copy_from_slice(&timescale.to_be_bytes());
            full_box(b"mdhd", 0, 0, &c)
        };
        let hdlr = |handler: &[u8; 4]| {
            let mut c = alloc::vec![0u8; 20];
            c[4..8].copy_from_slice(handler);
            full_box(b"hdlr", 0, 0, &c)
        };
        let descriptor = |tag: u8, body: &[u8]| {
            let mut v = alloc::vec![tag, body.len() as u8];
            v.extend_from_slice(body);
            v
        };
        let esds = |asc: &[u8]| {
            let dsi = descriptor(0x05, asc);
            let mut dcd_body = alloc::vec![0u8; 13];
            dcd_body.extend_from_slice(&dsi);
            let dcd = descriptor(0x04, &dcd_body);
            let mut es_body = alloc::vec![0u8; 3];
            es_body.extend_from_slice(&dcd);
            full_box(b"esds", 0, 0, &descriptor(0x03, &es_body))
        };
        let avcc = |sps: &[u8], pps: &[u8]| {
            let mut p = alloc::vec![0u8; 5];
            p.push(0xE1);
            p.extend_from_slice(&(sps.len() as u16).to_be_bytes());
            p.extend_from_slice(sps);
            p.push(1);
            p.extend_from_slice(&(pps.len() as u16).to_be_bytes());
            p.extend_from_slice(pps);
            mp4_box(b"avcC", &p)
        };
        let stsd = |entry: &[u8]| {
            let mut p = 1u32.to_be_bytes().to_vec();
            p.extend_from_slice(entry);
            full_box(b"stsd", 0, 0, &p)
        };
        // sample-table builders (one chunk holding all of a track's samples).
        let stsz = |sizes: &[u32]| {
            let mut b = alloc::vec![0u8; 8]; // default_size 0, then count
            b[4..8].copy_from_slice(&(sizes.len() as u32).to_be_bytes());
            for s in sizes {
                b.extend_from_slice(&s.to_be_bytes());
            }
            full_box(b"stsz", 0, 0, &b)
        };
        let stts = |count: u32, delta: u32| {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&count.to_be_bytes());
            b.extend_from_slice(&delta.to_be_bytes());
            full_box(b"stts", 0, 0, &b)
        };
        let stsc = |spc: u32| {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&1u32.to_be_bytes()); // first_chunk = 1
            b.extend_from_slice(&spc.to_be_bytes()); // samples_per_chunk
            b.extend_from_slice(&1u32.to_be_bytes()); // sample_desc_index
            full_box(b"stsc", 0, 0, &b)
        };
        let stco = |offset: u32| {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&offset.to_be_bytes());
            full_box(b"stco", 0, 0, &b)
        };
        let stss = |sample_no: u32| {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&sample_no.to_be_bytes());
            full_box(b"stss", 0, 0, &b)
        };
        let trak = |tkhd: &[u8], mdhd: &[u8], hdlr: &[u8], stbl: &[u8]| {
            let minf = mp4_box(b"minf", &mp4_box(b"stbl", stbl));
            let mdia = mp4_box(b"mdia", &[mdhd, hdlr, &minf].concat());
            mp4_box(b"trak", &[tkhd, &mdia].concat())
        };

        let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1e];
        let pps: &[u8] = &[0x68, 0xce];
        let asc: &[u8] = &[0x12, 0x10];

        let avc1 = {
            let mut p = alloc::vec![0u8; 78];
            p.extend_from_slice(&avcc(sps, pps));
            mp4_box(b"avc1", &p)
        };
        let mp4a = {
            let mut p = alloc::vec![0u8; 28];
            p[16..18].copy_from_slice(&2u16.to_be_bytes());
            p.extend_from_slice(&esds(asc));
            mp4_box(b"mp4a", &p)
        };

        // Two AVCC video samples + two raw AAC samples, in one mdat each track.
        let video_samples: [&[u8]; 2] = [&[0, 0, 0, 2, 0x65, 0xAA], &[0, 0, 0, 2, 0x41, 0xBB]];
        let audio_samples: [&[u8]; 2] = [&[0xC1, 0xC2, 0xC3], &[0xD1, 0xD2]];
        let v_sizes: Vec<u32> = video_samples.iter().map(|s| s.len() as u32).collect();
        let a_sizes: Vec<u32> = audio_samples.iter().map(|s| s.len() as u32).collect();
        let v_total: u32 = v_sizes.iter().sum();

        // The moov length is constant in the (u32) chunk-offset values, so a
        // placeholder build gives the offsets to fill into the real one.
        let build = |off_v: u32, off_a: u32| {
            let v_stbl = [
                stsd(&avc1),
                stsz(&v_sizes),
                stts(2, 3000),
                stsc(2),
                stco(off_v),
                stss(1),
            ]
            .concat();
            let a_stbl = [
                stsd(&mp4a),
                stsz(&a_sizes),
                stts(2, 1024),
                stsc(2),
                stco(off_a),
            ]
            .concat();
            let video_trak = trak(&tkhd(1, 320, 240), &mdhd(90_000), &hdlr(b"vide"), &v_stbl);
            let audio_trak = trak(&tkhd(2, 0, 0), &mdhd(48_000), &hdlr(b"soun"), &a_stbl);
            mp4_box(b"moov", &[video_trak, audio_trak].concat())
        };
        let moov_len = build(0, 0).len();
        let off_v = (moov_len + 8) as u32; // mdat payload starts after its header
        let off_a = off_v + v_total;
        let mut file = build(off_v, off_a);
        let mut mdat_body = Vec::new();
        for s in video_samples.iter().chain(audio_samples.iter()) {
            mdat_body.extend_from_slice(s);
        }
        file.extend_from_slice(&mp4_box(b"mdat", &mdat_body));

        // No moof: the multi-track progressive path must split both tracks.
        assert!(find_box(&file, b"moof").is_none(), "fixture is progressive");
        let tracks = parse_all_tracks(&file).expect("two tracks");
        assert_eq!(tracks.len(), 2);
        let samples = parse_progressive_multi(&file, &tracks).expect("progressive multi parse");
        assert_eq!(samples.len(), 4, "two video + two audio samples");

        let video: Vec<_> = samples.iter().filter(|(id, _)| *id == 1).collect();
        let audio: Vec<_> = samples.iter().filter(|(id, _)| *id == 2).collect();
        assert_eq!(video.len(), 2);
        assert_eq!(audio.len(), 2);
        // Video de-framed AVCC -> Annex-B; sample 1 (in stss) is the keyframe.
        assert_eq!(video[0].1.annexb, alloc::vec![0, 0, 0, 1, 0x65, 0xAA]);
        assert!(video[0].1.keyframe, "sample 1 is in stss");
        assert!(!video[1].1.keyframe, "sample 2 is not in stss");
        // Audio passed through verbatim; no stss means every sample is a sync point.
        assert_eq!(audio[0].1.annexb, audio_samples[0]);
        assert_eq!(audio[1].1.annexb, audio_samples[1]);
        assert!(audio[0].1.keyframe && audio[1].1.keyframe);
    }

    #[test]
    fn parse_text_entry_recognizes_tx3g_wvtt_stpp() {
        use crate::mp4box::mp4_box;
        // tx3g + wvtt -> plain UTF-8; stpp -> TTML document. The parser only needs
        // the sample-entry box to be present in the stsd entries.
        let tx3g = mp4_box(b"tx3g", &[0u8; 8]);
        assert!(matches!(
            parse_text_entry(&tx3g),
            Some(TrackKind::Text {
                format: TextFormat::Utf8,
                sample: TextSampleFormat::Tx3g
            })
        ));
        let wvtt = mp4_box(b"wvtt", &mp4_box(b"vttC", b"WEBVTT"));
        assert!(matches!(
            parse_text_entry(&wvtt),
            Some(TrackKind::Text {
                format: TextFormat::Utf8,
                sample: TextSampleFormat::Wvtt
            })
        ));
        let stpp = mp4_box(b"stpp", &[0u8; 8]);
        assert!(matches!(
            parse_text_entry(&stpp),
            Some(TrackKind::Text {
                format: TextFormat::Ttml,
                sample: TextSampleFormat::Stpp
            })
        ));
        // A caption codec is not a text codec (it goes through parse_cc_entry).
        assert!(parse_text_entry(&mp4_box(b"c608", &[0u8; 8])).is_none());
    }

    #[test]
    fn parse_cc_entry_recognizes_c608_and_c708() {
        use crate::mp4box::mp4_box;
        assert!(matches!(
            parse_cc_entry(&mp4_box(b"c608", &[0u8; 8])),
            Some(TrackKind::ClosedCaption {
                format: ClosedCaptionFormat::Cea608
            })
        ));
        assert!(matches!(
            parse_cc_entry(&mp4_box(b"c708", &[0u8; 8])),
            Some(TrackKind::ClosedCaption {
                format: ClosedCaptionFormat::Cea708
            })
        ));
        // Another codec under the same handler declines (the track is skipped).
        assert!(parse_cc_entry(&mp4_box(b"tx3g", &[0u8; 8])).is_none());
    }

    #[test]
    fn deframe_c608_tags_cdat_and_cdt2_pairs() {
        use crate::mp4box::mp4_box;
        // cdat pairs are field 1 (cc_type 0), cdt2 pairs field 2 (cc_type 1); the
        // pairs keep their order within each atom, atoms in sample order.
        let mut sample = mp4_box(b"cdat", &[0x94, 0x20, b'H', b'I']);
        sample.extend_from_slice(&mp4_box(b"cdt2", &[0x15, 0x2C]));
        assert_eq!(
            crate::cea::parse_cc_data(&deframe_c608(&sample)),
            alloc::vec![
                CcTriple {
                    cc_type: 0,
                    b0: 0x94,
                    b1: 0x20
                },
                CcTriple {
                    cc_type: 0,
                    b0: b'H',
                    b1: b'I'
                },
                CcTriple {
                    cc_type: 1,
                    b0: 0x15,
                    b1: 0x2C
                },
            ]
        );
        // An unknown atom is skipped, not fatal.
        let mut mixed = mp4_box(b"junk", &[1, 2, 3, 4]);
        mixed.extend_from_slice(&mp4_box(b"cdat", b"OK"));
        assert_eq!(deframe_c608(&mixed).len(), 3);
    }

    #[test]
    fn deframe_c608_survives_a_lying_atom_size() {
        // A size past the sample end, a size below the 8-byte header, an odd
        // payload length, and a truncated header all yield no caption data.
        let mut lying = Vec::from(0x7FFF_FFFFu32.to_be_bytes());
        lying.extend_from_slice(b"cdat");
        lying.extend_from_slice(&[0x94, 0x20]);
        assert!(deframe_c608(&lying).is_empty());
        let mut tiny = Vec::from(4u32.to_be_bytes());
        tiny.extend_from_slice(b"cdat");
        assert!(deframe_c608(&tiny).is_empty());
        let mut odd = Vec::from(11u32.to_be_bytes());
        odd.extend_from_slice(b"cdat");
        odd.extend_from_slice(&[0x94, 0x20, 0x00]);
        assert!(deframe_c608(&odd).is_empty());
        assert!(deframe_c608(&[0, 0, 0]).is_empty());
    }

    #[test]
    fn deframe_c708_unwraps_a_ccdp_packet() {
        use crate::mp4box::mp4_box;
        let triples = alloc::vec![
            CcTriple {
                cc_type: 3,
                b0: 0x21,
                b1: 0x40
            },
            CcTriple {
                cc_type: 2,
                b0: 0x41,
                b1: 0x42
            },
        ];
        let sample = mp4_box(b"ccdp", &crate::cea::build_cdp(&triples, 4, 7));
        assert_eq!(
            crate::cea::parse_cc_data(&deframe_c708(&sample)),
            triples,
            "the CDP's ccdata triples come back out"
        );
        // A CDP whose checksum is wrong contributes nothing, and a lying atom size
        // stops the scan rather than reading past the sample.
        let mut bad = mp4_box(b"ccdp", &crate::cea::build_cdp(&triples, 4, 7));
        *bad.last_mut().expect("checksum byte") ^= 0xFF;
        assert!(deframe_c708(&bad).is_empty());
        let mut lying = Vec::from(0x7FFF_FFFFu32.to_be_bytes());
        lying.extend_from_slice(b"ccdp");
        assert!(deframe_c708(&lying).is_empty());
    }

    #[test]
    fn deframe_wvtt_extracts_payl_payloads() {
        use crate::mp4box::mp4_box;
        // A vttc cue carries a payl payload; multiple cues in one sample join with
        // a newline. A vtte (empty cue) and unknown boxes contribute nothing.
        let cue = |id: &[u8], payl: &str| {
            let mut body = mp4_box(b"iden", id);
            body.extend_from_slice(&mp4_box(b"sttg", b"align:center"));
            body.extend_from_slice(&mp4_box(b"payl", payl.as_bytes()));
            mp4_box(b"vttc", &body)
        };
        let mut sample = cue(b"1", "Hello");
        sample.extend_from_slice(&cue(b"2", "World"));
        assert_eq!(deframe_wvtt(&sample), b"Hello\nWorld");

        // A pure empty-cue sample yields an empty string (clears the overlay).
        let empty = mp4_box(b"vtte", &[]);
        assert!(deframe_wvtt(&empty).is_empty());

        // A truncated box (size past the end) stops the scan, no panic.
        let truncated = [0, 0, 0, 0xFF, b'v', b't', b't', b'c'];
        assert!(deframe_wvtt(&truncated).is_empty());
    }
}

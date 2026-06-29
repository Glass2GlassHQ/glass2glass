//! Fragmented-MP4 / CMAF parsing shared by [`Mp4Src`](crate::mp4src) (file
//! source) and [`Fmp4Demux`](crate::fmp4demux) (byte-stream demuxer). Pure
//! `no_std + alloc`: reads the `moov` init (codec, geometry, timescale,
//! parameter sets) and walks `moof`+`mdat` fragments into Annex-B samples.
//!
//! Supported profile: one video track, `trun` v0 with explicit sample sizes,
//! `default-base-is-moof` data offsets landing on the following `mdat`'s
//! payload (what `Mp4Mux` writes and CMAF single-track files share). Anything
//! else fails loud rather than emitting a corrupt bitstream.

use alloc::vec::Vec;

use g2g_core::{AudioFormat, G2gError, VideoCodec};

use crate::mp4box::{be32, be64, boxes, find_box, find_path, parse_esds};

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
    /// Common-encryption defaults from a `cbcs` `tenc`, `None` for a clear track.
    pub(crate) cenc: Option<CencDefaults>,
}

/// MPEG-CENC `cbcs` track defaults from the init segment's `tenc` box. The IV is
/// the constant IV (cbcs uses `Per_Sample_IV_Size == 0`).
// The pattern / constant-IV fields are consumed by the `hls`-gated decryptor.
#[cfg_attr(not(feature = "hls"), allow(dead_code))]
#[derive(Debug, Clone)]
pub(crate) struct CencDefaults {
    pub(crate) crypt_byte_block: u8,
    pub(crate) skip_byte_block: u8,
    pub(crate) per_sample_iv_size: u8,
    pub(crate) constant_iv: Vec<u8>,
}

/// One `senc` subsample range: `clear` bytes pass through, the next `protected`
/// bytes are sample-encrypted (byte counts over the AVCC sample as stored).
// Fields are consumed by the `hls`-gated decryptor.
#[cfg_attr(not(feature = "hls"), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Subsample {
    pub(crate) clear: u32,
    pub(crate) protected: u32,
}

/// In-place sample decryptor: given a sample's bytes and its `senc` subsample
/// map, rewrites the protected ranges. `fmp4demux` supplies the cbcs one.
pub(crate) type SampleDecrypt<'a> = &'a mut dyn FnMut(&mut [u8], &[Subsample]);

#[derive(Debug)]
pub(crate) struct Sample {
    pub(crate) annexb: Vec<u8>,
    pub(crate) pts_ns: u64,
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
    let stsd = find_path(mdia, &[b"minf", b"stbl", b"stsd"]).ok_or(G2gError::CapsMismatch)?;
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
    } else if let Some(encv) = find_box(entries, b"encv") {
        let children = encv.get(78..).ok_or(G2gError::CapsMismatch)?;
        let sinf = find_box(children, b"sinf").ok_or(G2gError::CapsMismatch)?;
        let cenc = parse_cenc(sinf)?;
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

    Ok(Header { codec, width, height, timescale, duration_ns, param_sets, cenc })
}

/// What one track carries: a video elementary stream (codec + geometry +
/// parameter sets) or an audio elementary stream (format + channel layout +
/// AudioSpecificConfig). The multi-track read-side analog of `TrackInit` in the
/// muxer; clear tracks only (encryption stays single-track via [`parse_header`]).
#[derive(Debug, Clone)]
pub(crate) enum TrackKind {
    Video { codec: VideoCodec, width: u32, height: u32, param_sets: Vec<Vec<u8>> },
    Audio { format: AudioFormat, channels: u8, sample_rate: u32, asc: Vec<u8> },
}

/// One track's init data parsed from a `moov/trak`: the `track_ID` (which keys
/// the fragments in [`parse_fragments_multi`]), the media timescale, the
/// elementary-stream kind, and the cbcs `cenc` defaults for an encrypted track
/// (`None` for a clear one).
#[derive(Debug, Clone)]
pub(crate) struct TrackHeader {
    pub(crate) track_id: u32,
    pub(crate) timescale: u32,
    pub(crate) kind: TrackKind,
    pub(crate) cenc: Option<CencDefaults>,
}

/// Parse every forwardable (`vide` / `soun`) track out of a `moov` into a
/// [`TrackHeader`]. The single-track [`parse_header`] reads only the first
/// `trak`; this walks them all (what an A/V `.mp4` carries). Tracks with an
/// unrecognized handler (text, subtitles) are skipped, not errors; a malformed
/// video / audio track fails the parse. Errors if no track is forwardable.
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

/// Parse one `trak`. Returns `None` for a non-A/V handler (skip it), `Err` for a
/// malformed video / audio track.
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

    let stsd = find_path(mdia, &[b"minf", b"stbl", b"stsd"]).ok_or(G2gError::CapsMismatch)?;
    let entries = stsd.get(8..).ok_or(G2gError::CapsMismatch)?;

    let (kind, cenc) = match handler {
        b"vide" => parse_video_entry(entries, width, height)?,
        b"soun" => parse_audio_entry(entries, timescale)?,
        _ => return Ok(None), // text / subtitle / hint: not forwarded
    };
    Ok(Some(TrackHeader { track_id, timescale, kind, cenc }))
}

/// Read a video sample entry (`avc1` / `hvc1` / `hev1`, or the encrypted `encv`)
/// into a [`TrackKind::Video`] plus the cbcs `cenc` defaults for an encrypted
/// track. An `encv` carries the original codec config (`avcC` / `hvcC`) alongside
/// a `sinf` (original format + `cbcs` scheme + `tenc`), the same shape
/// [`parse_header`] reads.
fn parse_video_entry(
    entries: &[u8],
    width: u32,
    height: u32,
) -> Result<(TrackKind, Option<CencDefaults>), G2gError> {
    let (codec, param_sets, cenc) = if let Some(avc1) = find_box(entries, b"avc1") {
        let children = avc1.get(78..).ok_or(G2gError::CapsMismatch)?;
        let (sps, pps) = parse_avcc(find_box(children, b"avcC").ok_or(G2gError::CapsMismatch)?)?;
        (VideoCodec::H264, Vec::from([sps, pps]), None)
    } else if let Some(hvc1) = find_box(entries, b"hvc1").or_else(|| find_box(entries, b"hev1")) {
        let children = hvc1.get(78..).ok_or(G2gError::CapsMismatch)?;
        let hvcc = find_box(children, b"hvcC").ok_or(G2gError::CapsMismatch)?;
        (VideoCodec::H265, parse_hvcc(hvcc)?, None)
    } else if let Some(encv) = find_box(entries, b"encv") {
        let children = encv.get(78..).ok_or(G2gError::CapsMismatch)?;
        let sinf = find_box(children, b"sinf").ok_or(G2gError::CapsMismatch)?;
        let cenc = parse_cenc(sinf)?;
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
    Ok((TrackKind::Video { codec, width, height, param_sets }, cenc))
}

/// Read an AAC audio sample entry (`mp4a`/`esds`, or the encrypted `enca`) into a
/// [`TrackKind::Audio`] plus the cbcs `cenc` defaults for an encrypted track. The
/// sample rate is the media timescale (matching `Mp4AudioSrc`).
fn parse_audio_entry(
    entries: &[u8],
    timescale: u32,
) -> Result<(TrackKind, Option<CencDefaults>), G2gError> {
    let (entry, cenc) = match find_box(entries, b"mp4a") {
        Some(mp4a) => (mp4a, None),
        None => {
            let enca = find_box(entries, b"enca").ok_or(G2gError::CapsMismatch)?;
            let children = enca.get(28..).ok_or(G2gError::CapsMismatch)?;
            let sinf = find_box(children, b"sinf").ok_or(G2gError::CapsMismatch)?;
            (enca, Some(parse_cenc(sinf)?))
        }
    };
    // AudioSampleEntry: channelcount at offset 16, then 28 bytes before the esds.
    let channels = u16::from_be_bytes(
        entry.get(16..18).ok_or(G2gError::CapsMismatch)?.try_into().expect("2 bytes"),
    ) as u8;
    if channels == 0 {
        return Err(G2gError::CapsMismatch);
    }
    let esds = find_box(entry.get(28..).ok_or(G2gError::CapsMismatch)?, b"esds")
        .ok_or(G2gError::CapsMismatch)?;
    let asc = parse_esds(esds)?;
    Ok((TrackKind::Audio { format: AudioFormat::Aac, channels, sample_rate: timescale, asc }, cenc))
}

/// The decode state carried from a `moof` to its following `mdat`: the track id,
/// the per-sample `(size, pts_ns)`, and the per-sample `duration_ns`.
type PendingFragment = (u32, Vec<(u32, u64)>, Vec<u64>, Vec<Vec<Subsample>>);

/// Per-track sample decryptor for [`parse_fragments_multi`]: given the track's
/// cbcs `cenc` defaults, the sample bytes, and its `senc` subsample map, rewrites
/// the protected ranges in place. The AES lives in the caller (the `mp4-cenc`
/// [`Mp4DemuxN`](crate::mp4demuxn)); `fmp4` stays cipher-free.
pub(crate) type MultiDecrypt<'a> = &'a mut dyn FnMut(&CencDefaults, &mut [u8], &[Subsample]);

/// Walk the `moof`+`mdat` fragments of a multi-track fMP4 and split every sample
/// out, keyed by its `track_ID`. Each `traf`'s `tfhd` names the track, so a
/// fragment is routed to the matching [`TrackHeader`]: video samples are
/// de-framed AVCC->Annex-B with a keyframe scan, audio samples pass through (each
/// is a sync sample). Fragments for an unknown `track_ID` are skipped.
///
/// An encrypted (cbcs) track's samples are decrypted in place via `decrypt`
/// before de-framing, using the per-`traf` `senc` subsample map; an encrypted
/// track with no `decrypt` supplied fails loud (`CapsMismatch`), so a keyless
/// build never emits garbage. The multi-track analog of [`parse_fragments`]; a
/// non-conforming fragment is mis-split, not rejected, the same caveat as there.
pub(crate) fn parse_fragments_multi(
    data: &[u8],
    tracks: &[TrackHeader],
    mut decrypt: Option<MultiDecrypt<'_>>,
) -> Result<Vec<(u32, Sample)>, G2gError> {
    let mut out = Vec::new();
    let mut pending: Option<PendingFragment> = None;

    for (kind, payload) in boxes(data) {
        match kind {
            b"moof" => {
                let traf = find_box(payload, b"traf").ok_or(G2gError::CapsMismatch)?;
                let tfhd = find_box(traf, b"tfhd").ok_or(G2gError::CapsMismatch)?;
                // tfhd: track_ID at payload offset 4 (after version/flags).
                let track_id = be32(tfhd, 4)?;
                let Some(track) = tracks.iter().find(|t| t.track_id == track_id) else {
                    // A fragment for a track we don't forward: hold the id so the
                    // following mdat is skipped, not mis-split into another track.
                    pending = Some((track_id, Vec::new(), Vec::new(), Vec::new()));
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
                let (sizes, durs) = parse_trun(trun)?;
                // An encrypted track carries a `senc` (per-sample subsample maps).
                let subs = match &track.cenc {
                    Some(c) => match find_box(traf, b"senc") {
                        Some(senc) => parse_senc(senc, c.per_sample_iv_size)?,
                        None => Vec::new(),
                    },
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
                pending = Some((track_id, tagged, durations, subs));
            }
            b"mdat" => {
                let Some((track_id, tagged, durations, subs)) = pending.take() else {
                    return Err(G2gError::CapsMismatch); // mdat without moof
                };
                let Some(track) = tracks.iter().find(|t| t.track_id == track_id) else {
                    continue; // skipped (unforwarded track), no samples emitted
                };
                let mut at = 0usize;
                for (i, (size, pts_ns)) in tagged.iter().enumerate() {
                    let raw = payload.get(at..at + *size as usize).ok_or(G2gError::CapsMismatch)?;
                    at += *size as usize;
                    // Decrypt an encrypted track's sample in place before de-framing.
                    let owned;
                    let bytes: &[u8] = match &track.cenc {
                        Some(cenc) => {
                            let decrypt = decrypt.as_deref_mut().ok_or(G2gError::CapsMismatch)?;
                            let mut buf = raw.to_vec();
                            let sub = subs.get(i).map(Vec::as_slice).unwrap_or(&[]);
                            decrypt(cenc, &mut buf, sub);
                            owned = buf;
                            &owned
                        }
                        None => raw,
                    };
                    let (annexb, keyframe) = match &track.kind {
                        TrackKind::Video { codec, .. } => {
                            let annexb = avcc_to_annexb(bytes)?;
                            let kf = contains_keyframe(&annexb, *codec);
                            (annexb, kf)
                        }
                        // Audio access units are stored verbatim; each is a sync point.
                        TrackKind::Audio { .. } => (bytes.to_vec(), true),
                    };
                    out.push((
                        track_id,
                        Sample { annexb, pts_ns: *pts_ns, duration_ns: durations[i], keyframe },
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

/// Read the `cbcs` defaults out of a `sinf`: the `schm` scheme must be `cbcs`,
/// and `schi/tenc` (v1) carries the crypt/skip pattern, per-sample IV size, and
/// constant IV. Rejects other schemes and per-sample-IV (cenc/cbc1) variants.
fn parse_cenc(sinf: &[u8]) -> Result<CencDefaults, G2gError> {
    let schm = find_box(sinf, b"schm").ok_or(G2gError::CapsMismatch)?;
    if schm.get(4..8) != Some(b"cbcs") {
        return Err(G2gError::CapsMismatch);
    }
    let schi = find_box(sinf, b"schi").ok_or(G2gError::CapsMismatch)?;
    let tenc = find_box(schi, b"tenc").ok_or(G2gError::CapsMismatch)?;
    let version = *tenc.first().ok_or(G2gError::CapsMismatch)?;
    let (crypt_byte_block, skip_byte_block) = if version >= 1 {
        let packed = *tenc.get(5).ok_or(G2gError::CapsMismatch)?;
        (packed >> 4, packed & 0x0F)
    } else {
        (0, 0)
    };
    let is_protected = tenc.get(6) == Some(&1);
    let per_sample_iv_size = *tenc.get(7).ok_or(G2gError::CapsMismatch)?;
    // cbcs uses a constant IV (per-sample IV size 0); cenc/cbc1 are out of scope.
    if per_sample_iv_size != 0 {
        return Err(G2gError::CapsMismatch);
    }
    let constant_iv = if is_protected {
        let size = *tenc.get(24).ok_or(G2gError::CapsMismatch)? as usize;
        tenc.get(25..25 + size).ok_or(G2gError::CapsMismatch)?.to_vec()
    } else {
        Vec::new()
    };
    Ok(CencDefaults { crypt_byte_block, skip_byte_block, per_sample_iv_size, constant_iv })
}

/// Parse a `senc` box into per-sample subsample maps (cbcs: no per-sample IV).
/// An empty map for a sample means the whole sample is one protected range.
pub(crate) fn parse_senc(senc: &[u8], per_sample_iv_size: u8) -> Result<Vec<Vec<Subsample>>, G2gError> {
    let flags = be32(senc, 0)? & 0x00FF_FFFF;
    let has_subsamples = flags & 0x2 != 0;
    let count = be32(senc, 4)? as usize;
    let mut at = 8usize;
    // Each sample consumes at least its IV plus a subsample-count field, so an
    // untrusted `count` cannot exceed the remaining bytes. Reject a lying count
    // before reserving capacity for it.
    let min_bytes = (per_sample_iv_size as usize + if has_subsamples { 2 } else { 0 }).max(1);
    if count > senc.len().saturating_sub(at) / min_bytes {
        return Err(G2gError::CapsMismatch);
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        at += per_sample_iv_size as usize;
        let mut subs = Vec::new();
        if has_subsamples {
            let sub_count = u16::from_be_bytes(
                senc.get(at..at + 2).ok_or(G2gError::CapsMismatch)?.try_into().expect("2 bytes"),
            ) as usize;
            at += 2;
            for _ in 0..sub_count {
                let clear = u16::from_be_bytes(
                    senc.get(at..at + 2).ok_or(G2gError::CapsMismatch)?.try_into().expect("2 bytes"),
                ) as u32;
                let protected = be32(senc, at + 2)?;
                at += 6;
                subs.push(Subsample { clear, protected });
            }
        }
        out.push(subs);
    }
    Ok(out)
}

/// Parameter-set NALUs out of an `hvcC` payload, in array order (VPS, SPS,
/// PPS). Fixed 22-byte prefix (config version + 12-byte general PTL +
/// descriptive fields), then `numOfArrays`, then per-array NAL lists.
pub(crate) fn parse_hvcc(hvcc: &[u8]) -> Result<Vec<Vec<u8>>, G2gError> {
    let num_arrays = *hvcc.get(22).ok_or(G2gError::CapsMismatch)?;
    let mut at = 23usize;
    let mut sets = Vec::new();
    for _ in 0..num_arrays {
        // array header byte: array_completeness | reserved | NAL_unit_type.
        at += 1;
        let num_nalus = u16::from_be_bytes(
            hvcc.get(at..at + 2).ok_or(G2gError::CapsMismatch)?.try_into().expect("2 bytes"),
        );
        at += 2;
        for _ in 0..num_nalus {
            let len = u16::from_be_bytes(
                hvcc.get(at..at + 2).ok_or(G2gError::CapsMismatch)?.try_into().expect("2 bytes"),
            ) as usize;
            at += 2;
            let nalu = hvcc.get(at..at + len).ok_or(G2gError::CapsMismatch)?;
            sets.push(nalu.to_vec());
            at += len;
        }
    }
    if sets.is_empty() {
        return Err(G2gError::CapsMismatch);
    }
    Ok(sets)
}

/// First SPS and PPS out of an `avcC` payload.
pub(crate) fn parse_avcc(avcc: &[u8]) -> Result<(Vec<u8>, Vec<u8>), G2gError> {
    // 5 fixed bytes, then SPS count (low 5 bits).
    let sps_count = avcc.get(5).map(|b| b & 0x1F).ok_or(G2gError::CapsMismatch)?;
    if sps_count == 0 {
        return Err(G2gError::CapsMismatch);
    }
    let sps_len = u16::from_be_bytes(
        avcc.get(6..8).ok_or(G2gError::CapsMismatch)?.try_into().expect("2 bytes"),
    ) as usize;
    let sps = avcc.get(8..8 + sps_len).ok_or(G2gError::CapsMismatch)?.to_vec();
    let mut at = 8 + sps_len;
    let pps_count = *avcc.get(at).ok_or(G2gError::CapsMismatch)?;
    if pps_count == 0 {
        return Err(G2gError::CapsMismatch);
    }
    at += 1;
    let pps_len = u16::from_be_bytes(
        avcc.get(at..at + 2).ok_or(G2gError::CapsMismatch)?.try_into().expect("2 bytes"),
    ) as usize;
    at += 2;
    let pps = avcc.get(at..at + pps_len).ok_or(G2gError::CapsMismatch)?.to_vec();
    Ok((sps, pps))
}

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
    cenc: Option<&CencDefaults>,
    mut decrypt: Option<SampleDecrypt<'_>>,
) -> Result<Vec<Sample>, G2gError> {
    let mut samples = Vec::new();
    let mut pending: Option<Vec<(u32, u64)>> = None; // (size, pts_ns) per sample
    let mut durations: Vec<u64> = Vec::new();
    let mut pending_subs: Vec<Vec<Subsample>> = Vec::new();

    for (kind, payload) in boxes(data) {
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
                let (sizes, durs) = parse_trun(trun)?;
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
                pending_subs = match cenc {
                    Some(c) => match find_box(traf, b"senc") {
                        Some(senc) => parse_senc(senc, c.per_sample_iv_size)?,
                        None => Vec::new(),
                    },
                    None => Vec::new(),
                };
            }
            b"mdat" => {
                let Some(tagged) = pending.take() else {
                    return Err(G2gError::CapsMismatch); // mdat without moof
                };
                let mut at = 0usize;
                for (i, (size, pts_ns)) in tagged.iter().enumerate() {
                    let raw = payload.get(at..at + *size as usize).ok_or(G2gError::CapsMismatch)?;
                    let annexb = if cenc.is_some() {
                        // Encrypted: decrypt the sample in place, then de-frame.
                        let decrypt = decrypt.as_deref_mut().ok_or(G2gError::CapsMismatch)?;
                        let mut buf = raw.to_vec();
                        let subs = pending_subs.get(i).map(Vec::as_slice).unwrap_or(&[]);
                        decrypt(&mut buf, subs);
                        avcc_to_annexb(&buf)?
                    } else {
                        avcc_to_annexb(raw)?
                    };
                    let keyframe = contains_keyframe(&annexb, codec);
                    samples.push(Sample {
                        annexb,
                        pts_ns: *pts_ns,
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
    // Single video track: de-frame AVCC -> Annex-B.
    parse_progressive_track(data, stbl, timescale, true)
}

/// Parse the samples of one progressive track from its `stbl` sample tables,
/// addressing the elementary data in `data` by the absolute chunk offsets. The
/// per-track core shared by [`parse_progressive`] (single video track) and
/// [`parse_progressive_multi`] (every A/V track). `video` selects AVCC->Annex-B
/// de-framing; an audio track passes its samples through verbatim.
fn parse_progressive_track(
    data: &[u8],
    stbl: &[u8],
    timescale: u32,
    video: bool,
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
        (0..sample_count).map(|i| be32(stsz, 12 + i * 4)).collect::<Result<_, _>>()?
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
                let off = if signed { raw as i32 as i64 } else { raw as i64 };
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
        Some(stss) => {
            Some((0..be32(stss, 4)? as usize).map(|i| be32(stss, 8 + i * 4)).collect::<Result<_, _>>()?)
        }
        None => None,
    };

    let mut samples = Vec::with_capacity(sample_count);
    let mut dts: u64 = 0;
    for i in 0..sample_count {
        let off = sample_offsets[i] as usize;
        // `off` comes from an untrusted co64/stco chunk offset, so bound the end
        // with checked arithmetic (a u64 offset near usize::MAX would otherwise
        // overflow the `off + size` add and panic in debug).
        let end = off.checked_add(sizes[i] as usize).ok_or(G2gError::CapsMismatch)?;
        let raw = data.get(off..end).ok_or(G2gError::CapsMismatch)?;
        let pts = (dts as i64).saturating_add(ctts_offsets[i]).max(0) as u64;
        let keyframe = match &sync {
            Some(list) => list.binary_search(&((i + 1) as u32)).is_ok(),
            None => true,
        };
        let annexb = if video { avcc_to_annexb(raw)? } else { raw.to_vec() };
        samples.push(Sample {
            annexb,
            pts_ns: timescale_to_ns(pts, timescale),
            duration_ns: timescale_to_ns(durations[i] as u64, timescale),
            keyframe,
        });
        dts = dts.saturating_add(durations[i] as u64);
    }
    Ok(samples)
}

/// Parse a progressive (`moov`+`mdat`, no `moof`) multi-track file: every A/V
/// track's samples, keyed by `track_ID`, in track order. The progressive analog
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
        let video = matches!(track.kind, TrackKind::Video { .. });
        for s in parse_progressive_track(data, stbl, track.timescale, video)? {
            out.push((track.track_id, s));
        }
    }
    Ok(out)
}

/// The `trak` box (payload) in `moov` whose `tkhd` carries `track_id`, or `None`.
fn find_trak_by_id(moov: &[u8], track_id: u32) -> Option<&[u8]> {
    boxes(moov).filter(|(k, _)| *k == b"trak").map(|(_, t)| t).find(|trak| {
        find_box(trak, b"tkhd")
            .filter(|tkhd| tkhd.first() == Some(&0))
            .and_then(|tkhd| be32(tkhd, 12).ok())
            == Some(track_id)
    })
}

/// `trun` (v0 or v1) with explicit sample sizes; returns (sizes, durations) with
/// a zero duration when the stream omits it. v0 and v1 differ only in the sign of
/// the per-sample composition-time-offset field, which this skips (PTS is taken
/// from `tfdt` + decode-order durations and the decoder reorders), so both parse
/// identically. Real-world muxers (ffmpeg) emit v1 whenever B-frames are present.
pub(crate) fn parse_trun(trun: &[u8]) -> Result<(Vec<u32>, Vec<u32>), G2gError> {
    match trun.first() {
        Some(0) | Some(1) => {}
        _ => return Err(G2gError::CapsMismatch), // unknown trun version
    }
    let flags = be32(trun, 0)? & 0x00FF_FFFF;
    if flags & 0x200 == 0 {
        return Err(G2gError::CapsMismatch); // sizes must be explicit
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
    let per_sample = 4
        + if flags & 0x100 != 0 { 4 } else { 0 }
        + if flags & 0x400 != 0 { 4 } else { 0 }
        + if flags & 0x800 != 0 { 4 } else { 0 };
    if count > trun.len().saturating_sub(at) / per_sample {
        return Err(G2gError::CapsMismatch);
    }
    let mut sizes = Vec::with_capacity(count);
    let mut durations = Vec::with_capacity(count);
    for _ in 0..count {
        let mut duration = 0u32;
        if flags & 0x100 != 0 {
            duration = be32(trun, at)?;
            at += 4;
        }
        sizes.push(be32(trun, at)?);
        at += 4;
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

/// Whether the access unit already opens with a parameter-set NAL (so the
/// config-record sets need not be prepended): H.264 SPS(7), H.265 VPS(32).
pub(crate) fn starts_with_param_set(annexb: &[u8], codec: VideoCodec) -> bool {
    if annexb.len() <= 4 || annexb[..4] != [0, 0, 0, 1] {
        return false;
    }
    match codec {
        VideoCodec::H265 => (annexb[4] >> 1) & 0x3F == 32,
        _ => annexb[4] & 0x1F == 7,
    }
}

/// Whether an Annex-B access unit contains an IDR picture (the keyframe a seek
/// snaps to). NAL boundaries are 4-byte start codes. H.264 IDR is NAL type 5;
/// H.265 IDR is 19/20.
pub(crate) fn contains_keyframe(annexb: &[u8], codec: VideoCodec) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

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
        let (sizes, durs) = parse_trun(&p).unwrap();
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
        assert!(starts_with_param_set(&[0, 0, 0, 1, 0x67, 0xAA], VideoCodec::H264));
        assert!(!starts_with_param_set(&[0, 0, 0, 1, 0x65, 0xAA], VideoCodec::H264));
        assert!(starts_with_param_set(&[0, 0, 0, 1, 0x40, 0x01], VideoCodec::H265));
        assert!(!starts_with_param_set(&[0, 0, 0, 1, 0x26, 0x01], VideoCodec::H265));
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
        let v0 = parse_trun(&build(0)).expect("v0 parses");
        let v1 = parse_trun(&build(1)).expect("v1 parses");
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
        assert!(parse_trun(&t).is_err());
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
            let traf = mp4_box(b"traf", &[tfhd(track_id), tfdt(0), trun(dur, size)].concat());
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
            TrackKind::Video { codec, width, height, param_sets } => {
                assert_eq!(*codec, VideoCodec::H264);
                assert_eq!((*width, *height), (320, 240));
                assert_eq!(param_sets, &alloc::vec![sps.to_vec(), pps.to_vec()]);
            }
            other => panic!("track 0 should be video, got {other:?}"),
        }
        assert_eq!(tracks[1].track_id, 2);
        match &tracks[1].kind {
            TrackKind::Audio { format, channels, sample_rate, asc: got } => {
                assert_eq!(*format, AudioFormat::Aac);
                assert_eq!(*channels, 2);
                assert_eq!(*sample_rate, 48_000);
                assert_eq!(got, asc);
            }
            other => panic!("track 1 should be audio, got {other:?}"),
        }

        // --- assert: fragments route to their track and de-frame correctly -
        let samples = parse_fragments_multi(&file, &tracks, None).expect("fragment routing");
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
            let v_stbl = [stsd(&avc1), stsz(&v_sizes), stts(2, 3000), stsc(2), stco(off_v), stss(1)].concat();
            let a_stbl = [stsd(&mp4a), stsz(&a_sizes), stts(2, 1024), stsc(2), stco(off_a)].concat();
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
}

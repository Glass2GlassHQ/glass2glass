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

use g2g_core::{G2gError, VideoCodec};

use crate::mp4box::{be32, be64, boxes, find_box, find_path};

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
                    t += *dur as u64;
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

/// `trun` v0 with explicit sample sizes; returns (sizes, durations) with a
/// zero duration when the stream omits it.
pub(crate) fn parse_trun(trun: &[u8]) -> Result<(Vec<u32>, Vec<u32>), G2gError> {
    if trun.first() != Some(&0) {
        return Err(G2gError::CapsMismatch); // v1 (signed cts) unsupported
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
}

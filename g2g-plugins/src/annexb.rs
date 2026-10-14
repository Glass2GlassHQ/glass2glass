//! Annex-B/AVCC NAL splitting, avcC / parameter-set, and RBSP bitstream
//! helpers, shared by the H.264/H.265 parsers, the WebCodecs payloader
//! (`h264util`), the RTP packetizer (`rtppay`), and the MP4 / FLV demuxers.

use alloc::vec::Vec;

use g2g_core::{G2gError, VideoCodec};

/// Find the next Annex-B start code (`00 00 01` or `00 00 00 01`) at or after
/// `from`. Returns `(start_index, payload_index)`: the offset of the start code
/// and the offset of the NAL byte just past it.
pub(crate) fn next_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let prefix = find_start_code_prefix(data, from)?;
    let payload = prefix + START_CODE_PREFIX.len();
    if prefix > from && data[prefix - 1] == 0 {
        return Some((prefix - 1, payload));
    }
    Some((prefix, payload))
}

pub(crate) const START_CODE_PREFIX: [u8; 3] = [0, 0, 1];

pub(crate) const ANNEX_B_START_CODE: [u8; 4] = [0, 0, 0, 1];

pub(crate) fn find_start_code_prefix(data: &[u8], from: usize) -> Option<usize> {
    let last_byte_offset = START_CODE_PREFIX.len() - 1;
    let mut one_from = from.checked_add(last_byte_offset)?;
    loop {
        let one = one_from + memchr::memchr(1, data.get(one_from..)?)?;
        let prefix = one - last_byte_offset;
        if data[prefix..one] == START_CODE_PREFIX[..last_byte_offset] {
            return Some(prefix);
        }
        one_from = one + 1;
    }
}

/// Iterator over the NAL units of an Annex-B buffer, start codes stripped.
pub(crate) struct NalUnits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for NalUnits<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        let (_, begin) = next_start_code(self.data, self.pos)?;
        let end = match next_start_code(self.data, begin) {
            Some((sc, _)) => sc,
            None => self.data.len(),
        };
        self.pos = end;
        Some(&self.data[begin..end])
    }
}

pub(crate) fn nal_units(data: &[u8]) -> NalUnits<'_> {
    NalUnits { data, pos: 0 }
}

/// Heuristic: a buffer starting with an Annex-B start code is Annex-B, else it
/// is treated as AVCC (length-prefixed). Industry-standard (ffmpeg/gstreamer
/// h264parse do the same); a 4-byte AVCC length that happens to be `00 00 00
/// 01` is implausible for a real access unit and degrades to "no NAL found".
pub(crate) fn is_annex_b(data: &[u8]) -> bool {
    data.starts_with(&[0, 0, 0, 1]) || data.starts_with(&[0, 0, 1])
}

/// Iterator over length-prefixed (AVCC) NAL units: each NAL is preceded by a
/// big-endian length of `len_size` bytes (4 is `lengthSizeMinusOne = 3`, the
/// dominant case and what `retina` emits; an `avcC` record may declare 1..=4).
/// A truncated final length or NAL ends iteration rather than panicking.
pub(crate) struct AvccNals<'a> {
    data: &'a [u8],
    pos: usize,
    len_size: usize,
}

impl<'a> Iterator for AvccNals<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        let start = self.pos.checked_add(self.len_size)?;
        let prefix = self.data.get(self.pos..start)?;
        let len = prefix
            .iter()
            .fold(0usize, |acc, &b| (acc << 8) | b as usize);
        let end = start.checked_add(len)?;
        if end > self.data.len() {
            return None;
        }
        self.pos = end;
        Some(&self.data[start..end])
    }
}

pub(crate) fn avcc_nal_units(data: &[u8]) -> AvccNals<'_> {
    AvccNals {
        data,
        pos: 0,
        len_size: 4,
    }
}

/// [`avcc_nal_units`] with an explicit prefix width, for containers whose config
/// record declares a non-4-byte NAL length (FLV's `avcC` `lengthSizeMinusOne`).
/// `len_size` outside 1..=4 yields an empty iteration.
pub(crate) fn length_prefixed_nal_units(data: &[u8], len_size: usize) -> AvccNals<'_> {
    let len_size = if (1..=4).contains(&len_size) {
        len_size
    } else {
        usize::MAX
    };
    AvccNals {
        data,
        pos: 0,
        len_size,
    }
}

/// NAL iterator over either framing, picked by [`is_annex_b`]. Yields the same
/// NAL payloads (start codes / length prefixes stripped) for a given stream.
pub(crate) enum NalUnitsAny<'a> {
    AnnexB(NalUnits<'a>),
    Avcc(AvccNals<'a>),
}

impl<'a> Iterator for NalUnitsAny<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        match self {
            NalUnitsAny::AnnexB(it) => it.next(),
            NalUnitsAny::Avcc(it) => it.next(),
        }
    }
}

pub(crate) fn nal_units_any(data: &[u8]) -> NalUnitsAny<'_> {
    if is_annex_b(data) {
        NalUnitsAny::AnnexB(nal_units(data))
    } else {
        NalUnitsAny::Avcc(avcc_nal_units(data))
    }
}

/// H.264 NAL unit type: the low 5 bits of the first NAL header byte.
pub(crate) fn h264_nal_type(nal: &[u8]) -> Option<u8> {
    nal.first().map(|b| b & 0x1F)
}

/// Whether an access unit (either framing) begins a keyframe for `codec`: an
/// H.264 IDR (NAL type 5), an H.265 IRAP picture (types 16..=23, covering
/// BLA / IDR / CRA), an MPEG-4 Part 2 I-VOP (VOP start code 0xB6 whose
/// vop_coding_type, the top 2 bits of the next byte, is 0), an MPEG-1/2
/// sync point (see [`mpeg2_start_is_sync`]), or an AV1 key temporal unit (see
/// [`av1_keyframe`](crate::av1parse::av1_keyframe)). Used by the parsers,
/// muxers and demuxers to flag a decodable resume point in a stream whose
/// units carry no keyframe flag of their own.
pub(crate) fn au_is_keyframe(codec: VideoCodec, au: &[u8]) -> bool {
    if codec == VideoCodec::Av1 {
        return crate::av1parse::av1_keyframe(au);
    }
    nal_units_contain_keyframe(codec, nal_units_any(au))
}

/// [`au_is_keyframe`] over an access unit already split into its units (start
/// codes stripped), for any codec but AV1.
pub(crate) fn nal_units_contain_keyframe<'a>(
    codec: VideoCodec,
    mut units: impl Iterator<Item = &'a [u8]>,
) -> bool {
    units.any(|n| match (codec, n.first()) {
        (VideoCodec::H265, Some(b)) => (16..=23).contains(&((b >> 1) & 0x3F)),
        (VideoCodec::Mpeg4Part2, Some(&0xB6)) => n.get(1).is_some_and(|c| c >> 6 == 0),
        (VideoCodec::Mpeg4Part2, _) => false,
        (VideoCodec::Mpeg2, _) => mpeg2_start_is_sync(n),
        (_, Some(b)) => (b & 0x1F) == 5,
        (_, None) => false,
    })
}

/// Whether one MPEG-1/2 video start-code unit (`n[0]` is the start code value,
/// the start code prefix already stripped) marks a decodable resume point: a
/// sequence header (0xB3) or a GOP header (0xB8), both of which only ever
/// precede an I-picture, or a picture header (0x00) whose picture_coding_type
/// is 1 (intra). The coding type is bits 5..3 of the third picture-header byte,
/// the 10-bit temporal_reference occupying the two before it.
fn mpeg2_start_is_sync(n: &[u8]) -> bool {
    match n.first() {
        Some(&0xB3 | &0xB8) => true,
        Some(&0x00) => n.get(2).is_some_and(|b| (b >> 3) & 0x07 == 1),
        _ => false,
    }
}

/// Collect the H.264 SPS (type 7) and PPS (type 8) NAL units from an access unit
/// (either framing), returned as owned copies so the caller can cache them across
/// frames. VideoToolbox builds its `CMVideoFormatDescription` from the parameter
/// sets (supplied out of band), not from NALs inside the decode sample, so the
/// decoder pulls these out and feeds only the VCL NALs to each frame. The RTSP
/// server sink caches them for the same reason: its SDP offers them as
/// `sprop-parameter-sets`.
#[cfg(any(
    all(target_os = "macos", feature = "vtdecode"),
    all(target_os = "android", feature = "mediacodec"),
    feature = "rtsp-server",
    test
))]
pub(crate) fn h264_parameter_sets(au: &[u8]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut sps = Vec::new();
    let mut pps = Vec::new();
    for nal in nal_units_any(au) {
        match h264_nal_type(nal) {
            Some(7) => sps.push(nal.to_vec()),
            Some(8) => pps.push(nal.to_vec()),
            _ => {}
        }
    }
    (sps, pps)
}

/// The H.265 (HEVC) NAL unit type: bits 1..=6 of the first header byte
/// (`(b >> 1) & 0x3F`), where H.264 uses the low 5 bits.
pub(crate) fn h265_nal_type(nal: &[u8]) -> Option<u8> {
    nal.first().map(|b| (b >> 1) & 0x3F)
}

/// VPS, SPS, and PPS NAL lists: the H.265 parameter sets.
#[cfg(any(
    all(target_os = "android", feature = "mediacodec"),
    all(target_os = "macos", feature = "vtdecode"),
    test
))]
pub(crate) type H265ParameterSets = (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>);

/// Collect the H.265 VPS (32), SPS (33), and PPS (34) NAL units from an access
/// unit, as owned copies the caller caches across frames. MediaCodec takes them
/// as the `csd-0` codec-specific data (VPS+SPS+PPS concatenated), the HEVC analog
/// of H.264's separate SPS / PPS; VideoToolbox takes the same three lists to build
/// its HEVC format description.
#[cfg(any(
    all(target_os = "android", feature = "mediacodec"),
    all(target_os = "macos", feature = "vtdecode"),
    test
))]
pub(crate) fn h265_parameter_sets(au: &[u8]) -> H265ParameterSets {
    let mut vps = Vec::new();
    let mut sps = Vec::new();
    let mut pps = Vec::new();
    for nal in nal_units_any(au) {
        match h265_nal_type(nal) {
            Some(32) => vps.push(nal.to_vec()),
            Some(33) => sps.push(nal.to_vec()),
            Some(34) => pps.push(nal.to_vec()),
            _ => {}
        }
    }
    (vps, sps, pps)
}

/// Convert an access unit (Annex-B or AVCC) to AVCC form, each retained NAL
/// preceded by its 4-byte big-endian length (`lengthSizeMinusOne = 3`), keeping
/// only NALs for which `keep` returns true. VideoToolbox decode samples carry the
/// VCL (+ SEI) NALs length-prefixed; the parameter sets live in the format
/// description (see [`h264_parameter_sets`]), so the decoder excludes SPS / PPS /
/// AUD via `keep`. The inverse of [`avcc_nal_units`] for the kept NALs.
// AVCC framing is VideoToolbox-specific; MediaCodec takes Annex-B directly.
#[cfg(any(all(target_os = "macos", feature = "vtdecode"), test))]
pub(crate) fn to_avcc<F: Fn(&[u8]) -> bool>(au: &[u8], keep: F) -> Vec<u8> {
    let mut out = Vec::with_capacity(au.len() + 16);
    for nal in nal_units_any(au) {
        if !keep(nal) {
            continue;
        }
        out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        out.extend_from_slice(nal);
    }
    out
}

/// The NAL length-prefix width of AVCC samples whose container declares none
/// (`lengthSizeMinusOne = 3`, what `retina`, AVI writers and VideoToolbox emit).
pub(crate) const AVCC_LENGTH_SIZE: usize = 4;

/// Rewrite length-prefixed NALs (`len_size`-byte big-endian lengths, 1..=4) as
/// Annex-B, each NAL behind a 4-byte start code. The lengths come from the
/// container, so `None` unless they walk `data` exactly with no empty NAL: a
/// mis-framed sample is refused rather than truncated.
pub(crate) fn length_prefixed_to_annexb(data: &[u8], len_size: usize) -> Option<Vec<u8>> {
    if !(1..=4).contains(&len_size) {
        return None;
    }
    let mut out = Vec::with_capacity(data.len() + ANNEX_B_START_CODE.len());
    let mut consumed = 0usize;
    for nal in length_prefixed_nal_units(data, len_size) {
        if nal.is_empty() {
            return None;
        }
        out.extend_from_slice(&ANNEX_B_START_CODE);
        out.extend_from_slice(nal);
        consumed += len_size + nal.len();
    }
    (consumed == data.len()).then_some(out)
}

/// What an `avcC` / `hvcC` decoder configuration record carries: every
/// parameter set, in record order, and the NAL length-prefix width its samples
/// use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecoderConfigRecord {
    pub parameter_sets: Vec<Vec<u8>>,
    pub nal_length_size: usize,
}

/// `lengthSizeMinusOne` sits in the low 2 bits of this byte of an `avcC`.
const AVCC_LENGTH_SIZE_BYTE: usize = 4;
/// `numOfSequenceParameterSets` sits in the low 5 bits of this byte of an `avcC`.
const AVCC_SPS_COUNT_BYTE: usize = 5;
/// `lengthSizeMinusOne` sits in the low 2 bits of this byte of an `hvcC`.
const HVCC_LENGTH_SIZE_BYTE: usize = 21;
/// `numOfArrays` follows the fixed `hvcC` prefix (version, 12-byte general PTL,
/// descriptive fields).
const HVCC_ARRAY_COUNT_BYTE: usize = 22;

fn record_nal_length_size(byte: u8) -> usize {
    usize::from(byte & 0x03) + 1
}

/// Read `count` u16-length-prefixed NALs at `*at` into `sets`, advancing `*at`.
fn read_record_nals(
    record: &[u8],
    at: &mut usize,
    count: usize,
    sets: &mut Vec<Vec<u8>>,
) -> Result<(), G2gError> {
    for _ in 0..count {
        let length_bytes = record
            .get(*at..at.saturating_add(2))
            .ok_or(G2gError::CapsMismatch)?;
        let len = usize::from(u16::from_be_bytes([length_bytes[0], length_bytes[1]]));
        *at += 2;
        let nal = record
            .get(*at..at.saturating_add(len))
            .ok_or(G2gError::CapsMismatch)?;
        sets.push(nal.to_vec());
        *at += len;
    }
    Ok(())
}

/// Parse an `avcC` (AVCDecoderConfigurationRecord): every SPS then every PPS.
/// Fails on truncation or a record with no SPS or no PPS.
pub(crate) fn parse_avcc(avcc: &[u8]) -> Result<DecoderConfigRecord, G2gError> {
    let length_byte = *avcc
        .get(AVCC_LENGTH_SIZE_BYTE)
        .ok_or(G2gError::CapsMismatch)?;
    let sps_count = usize::from(
        avcc.get(AVCC_SPS_COUNT_BYTE)
            .ok_or(G2gError::CapsMismatch)?
            & 0x1F,
    );
    let mut at = AVCC_SPS_COUNT_BYTE + 1;
    let mut parameter_sets = Vec::new();
    read_record_nals(avcc, &mut at, sps_count, &mut parameter_sets)?;
    let pps_count = usize::from(*avcc.get(at).ok_or(G2gError::CapsMismatch)?);
    at += 1;
    read_record_nals(avcc, &mut at, pps_count, &mut parameter_sets)?;
    if sps_count == 0 || pps_count == 0 {
        return Err(G2gError::CapsMismatch);
    }
    Ok(DecoderConfigRecord {
        parameter_sets,
        nal_length_size: record_nal_length_size(length_byte),
    })
}

/// Parse an `hvcC` (HEVCDecoderConfigurationRecord): every NAL of every array,
/// in array order (VPS, SPS, PPS). Fails on truncation or an empty record.
pub(crate) fn parse_hvcc(hvcc: &[u8]) -> Result<DecoderConfigRecord, G2gError> {
    let length_byte = *hvcc
        .get(HVCC_LENGTH_SIZE_BYTE)
        .ok_or(G2gError::CapsMismatch)?;
    let num_arrays = *hvcc
        .get(HVCC_ARRAY_COUNT_BYTE)
        .ok_or(G2gError::CapsMismatch)?;
    let mut at = HVCC_ARRAY_COUNT_BYTE + 1;
    let mut parameter_sets = Vec::new();
    for _ in 0..num_arrays {
        // array header byte: array_completeness | reserved | NAL_unit_type.
        at += 1;
        let count_bytes = hvcc.get(at..at + 2).ok_or(G2gError::CapsMismatch)?;
        let num_nalus = usize::from(u16::from_be_bytes([count_bytes[0], count_bytes[1]]));
        at += 2;
        read_record_nals(hvcc, &mut at, num_nalus, &mut parameter_sets)?;
    }
    if parameter_sets.is_empty() {
        return Err(G2gError::CapsMismatch);
    }
    Ok(DecoderConfigRecord {
        parameter_sets,
        nal_length_size: record_nal_length_size(length_byte),
    })
}

/// Whether the access unit already opens with a parameter-set NAL (so the
/// config-record sets need not be prepended): H.264 SPS(7), H.265 VPS(32),
/// AV1 a sequence-header OBU anywhere in the temporal unit.
pub(crate) fn starts_with_param_set(annexb: &[u8], codec: VideoCodec) -> bool {
    if codec == VideoCodec::Av1 {
        return crate::av1parse::has_sequence_header(annexb);
    }
    if codec == VideoCodec::Mpeg4Part2 {
        // MPEG-4 Visual uses 3-byte start codes; its config opens with the visual
        // object sequence start code (0xB0) or a video object layer (0x20..=0x2F).
        return match annexb {
            [0, 0, 1, sc, ..] => *sc == 0xB0 || (0x20..=0x2F).contains(sc),
            _ => false,
        };
    }
    if annexb.len() <= 4 || annexb[..4] != [0, 0, 0, 1] {
        return false;
    }
    match codec {
        VideoCodec::H265 => (annexb[4] >> 1) & 0x3F == 32,
        _ => annexb[4] & 0x1F == 7,
    }
}

/// Prepend the out-of-band config-record parameter sets to `annexb` (the first
/// access unit). H.264/H.265 sets are raw NAL bodies, so each gets a 4-byte
/// Annex-B start code; MPEG-4 Part 2's single set (the esds VOL header, which
/// carries its own 3-byte start codes) and AV1's (the `av1C` configOBUs, whole
/// size-delimited OBUs) are prepended verbatim. Shared by the progressive and
/// fragmented MP4 demuxers and the FLV demuxer.
pub(crate) fn prepend_param_sets(
    annexb: &[u8],
    param_sets: &[Vec<u8>],
    codec: VideoCodec,
) -> Vec<u8> {
    let mut out = Vec::new();
    for set in param_sets {
        if !matches!(codec, VideoCodec::Mpeg4Part2 | VideoCodec::Av1) {
            out.extend_from_slice(&[0, 0, 0, 1]);
        }
        out.extend_from_slice(set);
    }
    out.extend_from_slice(annexb);
    out
}

/// Split an Annex-B buffer into NALUs (3- and 4-byte start codes).
pub(crate) fn split_annexb(data: &[u8]) -> Vec<&[u8]> {
    let mut nalus = Vec::new();
    let mut start = None;
    let mut i = 0;
    while i + 2 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let code_start = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
            if let Some(s) = start {
                nalus.push(&data[s..code_start]);
            }
            start = Some(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    if let Some(s) = start {
        if s < data.len() {
            nalus.push(&data[s..]);
        }
    }
    nalus
}

/// NAL unit type, decoded per codec: H.264 packs it in the low 5 bits of byte
/// 0; H.265 in bits 1..6 of byte 0.
pub(crate) fn nalu_type(codec: VideoCodec, nalu: &[u8]) -> u8 {
    let b0 = nalu.first().copied().unwrap_or(0);
    match codec {
        VideoCodec::H265 => (b0 >> 1) & 0x3F,
        _ => b0 & 0x1F,
    }
}

fn find_nalu<'a>(codec: VideoCodec, nalus: &[&'a [u8]], ty: u8) -> Option<&'a [u8]> {
    nalus.iter().copied().find(|n| nalu_type(codec, n) == ty)
}

/// Ordered parameter-set NALUs the moov needs: H.264 SPS(7)+PPS(8), H.265
/// VPS(32)+SPS(33)+PPS(34). Missing any one is a loud error.
pub(crate) fn parameter_sets<'a>(
    codec: VideoCodec,
    nalus: &[&'a [u8]],
) -> Result<Vec<&'a [u8]>, G2gError> {
    let types: &[u8] = match codec {
        VideoCodec::H265 => &[32, 33, 34],
        _ => &[7, 8],
    };
    let sets: Vec<&[u8]> = types
        .iter()
        .map(|ty| find_nalu(codec, nalus, *ty).ok_or(G2gError::CapsMismatch))
        .collect::<Result<_, _>>()?;
    // avcC copies profile/compat/level from sps[1..4]; a shorter SPS is
    // malformed, so fail loud instead of writing a truncated record.
    if !matches!(codec, VideoCodec::H265) && sets[0].len() < 4 {
        return Err(G2gError::CapsMismatch);
    }
    Ok(sets)
}

/// AVCC sample payload: every NALU prefixed with its 4-byte big-endian
/// length (parameter sets stay in-band, which fMP4 players accept).
pub(crate) fn avcc_sample(nalus: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for n in nalus {
        out.extend_from_slice(&(n.len() as u32).to_be_bytes());
        out.extend_from_slice(n);
    }
    out
}

/// The `avcC` AVCDecoderConfigurationRecord body (no box header). `param_sets`
/// is [SPS, PPS]. Also the Matroska `CodecPrivate` for `V_MPEG4/ISO/AVC`.
pub(crate) fn avcc_record(param_sets: &[&[u8]]) -> Vec<u8> {
    let sps = param_sets[0];
    let pps = param_sets[1];
    let mut p = Vec::new();
    p.push(1); // configuration version
    p.extend_from_slice(&sps[1..4.min(sps.len())]); // profile/compat/level
    p.push(0xFC | 3); // 4-byte NALU lengths
    p.push(0xE0 | 1); // 1 SPS
    p.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    p.extend_from_slice(sps);
    p.push(1); // 1 PPS
    p.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    p.extend_from_slice(pps);
    p
}

/// Convert EBSP to RBSP by removing `0x03` emulation-prevention bytes that
/// follow two consecutive zero bytes (H.264 / H.265 share this encoding).
/// Always returns owned bytes for parser simplicity.
pub(crate) fn strip_emulation_prevention(ebsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ebsp.len());
    let mut zeros = 0usize;
    for &b in ebsp {
        if zeros >= 2 && b == 0x03 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

/// Convert RBSP to EBSP by inserting a `0x03` emulation-prevention byte before
/// any byte `<= 0x03` that follows two zero bytes, the inverse of
/// [`strip_emulation_prevention`]. Used by the SAMPLE-AES decryptor to re-escape a
/// NAL after decrypting its de-escaped payload, and by `cea::build_cc_sei` to
/// escape a caption SEI (so it is always available on the no_std baseline).
pub(crate) fn add_emulation_prevention(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len());
    let mut zeros = 0usize;
    for &b in rbsp {
        if zeros >= 2 && b <= 0x03 {
            out.push(0x03);
            zeros = 0;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    out
}

/// Read an SEI `0xFF`-extended value (`payloadType` / `payloadSize`): a run of
/// `0xFF` bytes each adding 255, then a final byte. Returns the value and the
/// index past it, or `None` if the buffer ends mid-value.
pub(crate) fn read_ff_extended(data: &[u8], mut i: usize) -> Option<(usize, usize)> {
    let mut value: usize = 0;
    loop {
        let b = *data.get(i)?;
        i += 1;
        value = value.checked_add(b as usize)?;
        if b != 0xFF {
            return Some((value, i));
        }
    }
}

/// Byte offset of the start code of the first VCL slice NAL in an Annex-B access
/// unit, or `None` if there is none. H.264 VCL NAL types are 1..=5; H.265 VCL
/// types are 0..=31.
pub(crate) fn vcl_start(au: &[u8], codec: VideoCodec) -> Option<usize> {
    let mut pos = 0usize;
    while let Some((sc, begin)) = next_start_code(au, pos) {
        let hdr = *au.get(begin)?;
        let is_vcl = match codec {
            VideoCodec::H265 => ((hdr >> 1) & 0x3F) < 32,
            _ => (1..=5).contains(&(hdr & 0x1F)),
        };
        if is_vcl {
            return Some(sc);
        }
        pos = begin;
    }
    None
}

/// MSB-first bit reader over a byte slice, the shared bitstream cursor for the
/// H.264 / H.265 SPS parsers. All readers return `None` on EOF rather than
/// panicking, so a partial / malformed header propagates as "field unknown"
/// instead of aborting the pipeline.
pub(crate) struct BitReader<'a> {
    buf: &'a [u8],
    next_byte: usize,
    current_byte: u8,
    bits_left_in_current_byte: u32,
    bit_pos: usize,
    skips_emulation_prevention: bool,
    zero_run: usize,
}

const EMULATION_PREVENTION_BYTE: u8 = 0x03;

impl<'a> BitReader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            next_byte: 0,
            current_byte: 0,
            bits_left_in_current_byte: 0,
            bit_pos: 0,
            skips_emulation_prevention: false,
            zero_run: 0,
        }
    }

    // reads the RBSP that strip_emulation_prevention would return, bit_pos counts RBSP bits
    pub(crate) fn skipping_emulation_prevention(ebsp: &'a [u8]) -> Self {
        Self {
            skips_emulation_prevention: true,
            ..Self::new(ebsp)
        }
    }

    fn load_next_byte(&mut self) -> Option<()> {
        let mut byte = *self.buf.get(self.next_byte)?;
        self.next_byte += 1;
        if self.skips_emulation_prevention {
            if self.zero_run >= 2 && byte == EMULATION_PREVENTION_BYTE {
                self.zero_run = 0;
                byte = *self.buf.get(self.next_byte)?;
                self.next_byte += 1;
            }
            self.zero_run = if byte == 0 { self.zero_run + 1 } else { 0 };
        }
        self.current_byte = byte;
        self.bits_left_in_current_byte = 8;
        Some(())
    }

    pub(crate) fn read_bit(&mut self) -> Option<u32> {
        if self.bits_left_in_current_byte == 0 {
            self.load_next_byte()?;
        }
        self.bits_left_in_current_byte -= 1;
        self.bit_pos += 1;
        Some(u32::from(
            (self.current_byte >> self.bits_left_in_current_byte) & 1,
        ))
    }

    /// Read `n` (<= 32) bits MSB-first into a `u32`.
    pub(crate) fn read_bits(&mut self, n: u32) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | self.read_bit()?;
        }
        Some(value)
    }

    /// The current read position in bits from the start of the buffer. Used to
    /// measure the size of a variable-length syntax element (H.265 needs the bit
    /// length of a slice's inline short-term RPS for the decoder).
    #[cfg_attr(not(feature = "vulkan-video"), allow(dead_code))]
    pub(crate) fn bit_pos(&self) -> usize {
        self.bit_pos
    }

    /// Advance past `n` bits without decoding them (e.g. H.265's fixed-size
    /// `profile_tier_level`). `None` if that would run past the end.
    pub(crate) fn skip_bits(&mut self, n: usize) -> Option<()> {
        for _ in 0..n {
            self.read_bit()?;
        }
        Some(())
    }

    /// Unsigned exp-Golomb. Reads leading zeros to determine codeword length,
    /// then `n+1` bits of the codeword value, returns value - 1.
    pub(crate) fn read_ue(&mut self) -> Option<u32> {
        let mut leading_zeros = 0u32;
        loop {
            let b = self.read_bit()?;
            if b == 1 {
                break;
            }
            leading_zeros += 1;
            if leading_zeros > 31 {
                return None;
            }
        }
        let mut val = 1u32;
        for _ in 0..leading_zeros {
            val = (val << 1) | self.read_bit()?;
        }
        Some(val - 1)
    }

    /// H.264 `more_rbsp_data()` (7.2): is there RBSP payload left before the
    /// `rbsp_stop_one_bit`? The stop bit is the last `1` in the buffer, followed
    /// only by zero alignment padding. Reading an optional trailing syntax
    /// element without this check would misread the stop bit as real data (e.g.
    /// a baseline PPS has no `transform_8x8_mode_flag`, so the next bit is the
    /// stop bit, not the flag).
    // Only the `vulkan-video` PPS parser needs this today; keep it compiled in
    // every build (other parsers may adopt it) but do not warn when unused.
    #[cfg_attr(not(feature = "vulkan-video"), allow(dead_code))]
    pub(crate) fn more_rbsp_data(&self) -> bool {
        // runs ahead of bit_pos past a skipped emulation-prevention byte
        let buffer_bit_pos = self.next_byte * 8 - self.bits_left_in_current_byte as usize;
        // Locate the rbsp_stop_one_bit: the last set bit in the whole buffer.
        let Some(stop_byte_idx) = self.buf.iter().rposition(|&b| b != 0) else {
            return false;
        };
        let bit_in_byte = 7 - self.buf[stop_byte_idx].trailing_zeros() as usize;
        buffer_bit_pos < stop_byte_idx * 8 + bit_in_byte
    }

    /// Signed exp-Golomb, mapping ue to se per H.264 SS9.1.1.
    pub(crate) fn read_se(&mut self) -> Option<i32> {
        let ue = self.read_ue()?;
        Some(if ue & 1 == 1 {
            ((ue >> 1) + 1) as i32
        } else {
            -((ue >> 1) as i32)
        })
    }
}

/// MSB-first bit writer, the inverse of [`BitReader`]. Test-only: the codec
/// parsers (`h264parse`, `h265parse`, `av1parse`, `vp9parse`) and `vulkanvideo`
/// hand-build RBSP / bitstream fragments to feed their parsers, and all wrote an
/// identical copy of this before it was shared here.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct BitWriter {
    buf: Vec<u8>,
    bit_pos: usize,
}

#[cfg(test)]
impl BitWriter {
    pub(crate) fn write_bit(&mut self, b: u32) {
        let byte_idx = self.bit_pos / 8;
        if byte_idx >= self.buf.len() {
            self.buf.push(0);
        }
        let bit_off = 7 - (self.bit_pos % 8);
        self.buf[byte_idx] |= ((b & 1) as u8) << bit_off;
        self.bit_pos += 1;
    }

    pub(crate) fn write_bits(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.write_bit((value >> i) & 1);
        }
    }

    /// Unsigned Exp-Golomb (`ue(v)`), the H.264/H.265 RBSP coding.
    pub(crate) fn write_ue(&mut self, v: u32) {
        let v1 = v + 1;
        let n = 31 - v1.leading_zeros();
        for _ in 0..n {
            self.write_bit(0);
        }
        self.write_bits(v1, n + 1);
    }

    pub(crate) fn align_to_byte(&mut self) {
        while !self.bit_pos.is_multiple_of(8) {
            self.write_bit(0);
        }
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn nal_iter_handles_3_and_4_byte_start_codes() {
        // 4-byte start code, then a 3-byte one.
        let mut au = Vec::new();
        au.extend_from_slice(&[0, 0, 0, 1, 0x67, 0x42]);
        au.extend_from_slice(&[0, 0, 1, 0x65, 0x88]);
        let nals: Vec<&[u8]> = nal_units(&au).collect();
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0], &[0x67, 0x42]);
        assert_eq!(nals[1], &[0x65, 0x88]);
    }

    #[test]
    fn avcc_iteration_matches_annexb_for_the_same_nals() {
        let sps: &[u8] = &[0x67, 0x42, 0xC0, 0x1E];
        let idr: &[u8] = &[0x65, 0x88, 0x84, 0x21, 0x0A];

        let mut annexb = Vec::new();
        annexb.extend_from_slice(&[0, 0, 0, 1]);
        annexb.extend_from_slice(sps);
        annexb.extend_from_slice(&[0, 0, 1]);
        annexb.extend_from_slice(idr);

        let mut avcc = Vec::new();
        avcc.extend_from_slice(&(sps.len() as u32).to_be_bytes());
        avcc.extend_from_slice(sps);
        avcc.extend_from_slice(&(idr.len() as u32).to_be_bytes());
        avcc.extend_from_slice(idr);

        assert!(is_annex_b(&annexb));
        assert!(!is_annex_b(&avcc));

        let from_annexb: Vec<&[u8]> = nal_units_any(&annexb).collect();
        let from_avcc: Vec<&[u8]> = nal_units_any(&avcc).collect();
        assert_eq!(from_annexb, vec![sps, idr]);
        assert_eq!(from_avcc, vec![sps, idr], "AVCC yields the same NALs");
    }

    #[test]
    fn length_prefixed_to_annexb_prefixes_each_nal_with_a_start_code() {
        let sps: &[u8] = &[0x67, 0x42, 0xC0, 0x1E];
        let idr: &[u8] = &[0x65, 0x88, 0x84, 0x21, 0x0A];
        let mut avcc = Vec::new();
        avcc.extend_from_slice(&(sps.len() as u32).to_be_bytes());
        avcc.extend_from_slice(sps);
        avcc.extend_from_slice(&(idr.len() as u32).to_be_bytes());
        avcc.extend_from_slice(idr);

        let annexb = length_prefixed_to_annexb(&avcc, AVCC_LENGTH_SIZE).expect("valid walk");
        assert!(is_annex_b(&annexb), "output is Annex-B framed");
        let nals: Vec<&[u8]> = nal_units(&annexb).collect();
        assert_eq!(nals, vec![sps, idr], "the same NALs, now start-code framed");
        // Every NAL is preceded by a 4-byte start code.
        assert_eq!(&annexb[..4], &[0, 0, 0, 1]);
    }

    #[test]
    fn escaped_reader_reads_the_same_bits_as_the_stripped_buffer() {
        // zeros and 0x03 dominate so escape sequences occur at every alignment
        const ALPHABET: [u8; 6] = [0, 0, 0, 3, 1, 0x80];
        const BUFFERS: u32 = 200;
        const BUFFER_LEN: usize = 48;
        let mut state = 0x2545_F491_u32;
        let mut next_random = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for _ in 0..BUFFERS {
            let ebsp: Vec<u8> = (0..BUFFER_LEN)
                .map(|_| ALPHABET[next_random() as usize % ALPHABET.len()])
                .collect();
            let rbsp = strip_emulation_prevention(&ebsp);
            let mut stripped = BitReader::new(&rbsp);
            let mut escaped = BitReader::skipping_emulation_prevention(&ebsp);
            loop {
                let width = next_random() % 12;
                let reads = [
                    (stripped.read_ue(), escaped.read_ue()),
                    (stripped.read_bits(width), escaped.read_bits(width)),
                    (
                        stripped.skip_bits(width as usize).map(|_| 0),
                        escaped.skip_bits(width as usize).map(|_| 0),
                    ),
                ];
                for (from_stripped, from_escaped) in reads {
                    assert_eq!(from_stripped, from_escaped);
                }
                assert_eq!(stripped.bit_pos(), escaped.bit_pos());
                if reads.iter().any(|(read, _)| read.is_none()) {
                    break;
                }
            }
        }
    }

    /// Build an `avcC` body from raw SPS/PPS NALs with the given length size.
    fn build_avcc(sps: &[&[u8]], pps: &[&[u8]], nal_length_size: u8) -> Vec<u8> {
        let mut v = vec![1u8, 0x42, 0x00, 0x1e]; // version + profile/compat/level
        v.push(0xFC | (nal_length_size - 1)); // 111111 + lengthSizeMinusOne
        v.push(0xE0 | (sps.len() as u8 & 0x1F)); // 111 + numSPS
        for s in sps {
            v.extend_from_slice(&(s.len() as u16).to_be_bytes());
            v.extend_from_slice(s);
        }
        v.push(pps.len() as u8);
        for p in pps {
            v.extend_from_slice(&(p.len() as u16).to_be_bytes());
            v.extend_from_slice(p);
        }
        v
    }

    #[test]
    fn parse_avcc_keeps_every_parameter_set_and_the_length_size() {
        let sps: &[u8] = &[0x67, 0x64, 0x00];
        let second_sps: &[u8] = &[0x67, 0x64, 0x01];
        let pps: &[u8] = &[0x68, 0xee, 0x3c];
        let avcc = build_avcc(&[sps, second_sps], &[pps], 4);
        let record = parse_avcc(&avcc).expect("valid avcC");
        assert_eq!(record.nal_length_size, 4);
        assert_eq!(record.parameter_sets, vec![sps, second_sps, pps]);

        // lengthSizeMinusOne == 1 -> 2-byte prefixes.
        let avcc2 = build_avcc(&[sps], &[pps], 2);
        assert_eq!(parse_avcc(&avcc2).unwrap().nal_length_size, 2);

        // Truncated record, or one without a PPS, fails.
        assert!(parse_avcc(&avcc[..7]).is_err());
        assert!(parse_avcc(&build_avcc(&[sps], &[], 4)).is_err());
    }

    #[test]
    fn parse_hvcc_keeps_every_array_in_order() {
        let vps: &[u8] = &[0x40, 0x01, 0x0c];
        let sps: &[u8] = &[0x42, 0x01, 0x01];
        let pps: &[u8] = &[0x44, 0x01];
        let mut hvcc = vec![0u8; HVCC_ARRAY_COUNT_BYTE];
        hvcc[0] = 1;
        hvcc[HVCC_LENGTH_SIZE_BYTE] = 0xFC | 3; // lengthSizeMinusOne = 3 -> size 4
        hvcc.push(3); // numArrays
        for (ty, nal) in [(0x20u8, vps), (0x21, sps), (0x22, pps)] {
            hvcc.push(ty);
            hvcc.extend_from_slice(&1u16.to_be_bytes()); // one NAL
            hvcc.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            hvcc.extend_from_slice(nal);
        }
        let record = parse_hvcc(&hvcc).expect("valid hvcC");
        assert_eq!(record.nal_length_size, 4);
        assert_eq!(record.parameter_sets, vec![vps, sps, pps]);
        assert!(parse_hvcc(&hvcc[..24]).is_err());
    }

    #[test]
    fn length_prefixed_conversion_validates_the_whole_buffer() {
        let sps: &[u8] = &[0x67, 0x42, 0xC0, 0x1E];
        let idr: &[u8] = &[0x65, 0x88, 0x84, 0x21, 0x0A];
        // 2-byte prefixes (a non-default lengthSizeMinusOne).
        let mut avcc = Vec::new();
        avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(sps);
        avcc.extend_from_slice(&(idr.len() as u16).to_be_bytes());
        avcc.extend_from_slice(idr);

        let annexb = length_prefixed_to_annexb(&avcc, 2).expect("valid walk");
        let nals: Vec<&[u8]> = nal_units(&annexb).collect();
        assert_eq!(nals, vec![sps, idr]);

        // Truncated length, trailing garbage, a zero-length NAL, and data that
        // is really Annex-B all fail the exact-consumption walk (the caller
        // forwards the original bytes instead of mis-framing).
        assert!(length_prefixed_to_annexb(&avcc[..avcc.len() - 1], 2).is_none());
        let mut trailing = avcc.clone();
        trailing.push(0);
        assert!(length_prefixed_to_annexb(&trailing, 2).is_none());
        assert!(length_prefixed_to_annexb(&[0, 0], 2).is_none());
        let mut real_annexb = Vec::new();
        real_annexb.extend_from_slice(&[0, 0, 0, 1]);
        real_annexb.extend_from_slice(idr);
        assert!(length_prefixed_to_annexb(&real_annexb, 4).is_none());
    }

    #[test]
    fn avcc_stops_on_a_truncated_length() {
        // 4-byte length says 10 bytes but only 3 follow.
        let mut avcc = Vec::new();
        avcc.extend_from_slice(&10u32.to_be_bytes());
        avcc.extend_from_slice(&[0x67, 0x42, 0xC0]);
        assert_eq!(avcc_nal_units(&avcc).count(), 0, "truncated NAL is dropped");
    }

    #[test]
    fn parameter_sets_extracted_from_mixed_au() {
        // SPS (7), PPS (8), SEI (6), IDR slice (5), in Annex-B.
        let sps: &[u8] = &[0x67, 0x42, 0xE0, 0x1E];
        let pps: &[u8] = &[0x68, 0xCE, 0x3C, 0x80];
        let sei: &[u8] = &[0x06, 0x05, 0x01];
        let idr: &[u8] = &[0x65, 0x88, 0x84, 0x21];
        let mut au = Vec::new();
        for nal in [sps, pps, sei, idr] {
            au.extend_from_slice(&[0, 0, 0, 1]);
            au.extend_from_slice(nal);
        }
        let (got_sps, got_pps) = h264_parameter_sets(&au);
        assert_eq!(got_sps, vec![sps.to_vec()]);
        assert_eq!(got_pps, vec![pps.to_vec()]);
    }

    #[test]
    fn h265_parameter_sets_extracted_from_mixed_au() {
        // VPS (32), SPS (33), PPS (34), SEI (39), IDR slice (19), in Annex-B.
        // H.265 NAL header is 2 bytes; type = (byte0 >> 1) & 0x3F.
        let vps: &[u8] = &[0x40, 0x01, 0x0c];
        let sps: &[u8] = &[0x42, 0x01, 0x01];
        let pps: &[u8] = &[0x44, 0x01, 0xc0];
        let sei: &[u8] = &[0x4e, 0x01, 0x05];
        let idr: &[u8] = &[0x26, 0x01, 0xaf];
        let mut au = Vec::new();
        for nal in [vps, sps, pps, sei, idr] {
            au.extend_from_slice(&[0, 0, 0, 1]);
            au.extend_from_slice(nal);
        }
        let (got_vps, got_sps, got_pps) = h265_parameter_sets(&au);
        assert_eq!(got_vps, vec![vps.to_vec()]);
        assert_eq!(got_sps, vec![sps.to_vec()]);
        assert_eq!(got_pps, vec![pps.to_vec()]);
    }

    #[test]
    fn to_avcc_keeps_vcl_excludes_parameter_sets_and_round_trips() {
        let sps: &[u8] = &[0x67, 0x42, 0xE0, 0x1E];
        let pps: &[u8] = &[0x68, 0xCE];
        let sei: &[u8] = &[0x06, 0x05, 0x01];
        let idr: &[u8] = &[0x65, 0x88, 0x84, 0x21, 0x0A];
        let mut au = Vec::new();
        for nal in [sps, pps, sei, idr] {
            au.extend_from_slice(&[0, 0, 0, 1]);
            au.extend_from_slice(nal);
        }
        // Exclude SPS(7) / PPS(8) / AUD(9); keep SEI + VCL, like the decoder.
        let avcc = to_avcc(&au, |nal| !matches!(h264_nal_type(nal), Some(7..=9)));
        // The kept NALs, recovered by the AVCC iterator, are SEI then IDR.
        let kept: Vec<&[u8]> = avcc_nal_units(&avcc).collect();
        assert_eq!(
            kept,
            vec![sei, idr],
            "parameter sets dropped, order preserved"
        );
    }

    fn annexb_au(nals: &[&[u8]]) -> Vec<u8> {
        nals.iter()
            .flat_map(|nal| [&[0u8, 0, 0, 1][..], nal].concat())
            .collect()
    }

    fn avcc_au(nals: &[&[u8]]) -> Vec<u8> {
        nals.iter()
            .flat_map(|nal| [&(nal.len() as u32).to_be_bytes()[..], nal].concat())
            .collect()
    }

    #[test]
    fn h264_keyframe_is_an_idr_in_either_framing() {
        use g2g_core::VideoCodec::H264;
        let sps: &[u8] = &[0x67, 0x42, 0xE0, 0x1E];
        let pps: &[u8] = &[0x68, 0xCE];
        let idr: &[u8] = &[0x65, 0x88];
        let non_idr: &[u8] = &[0x41, 0x9A];
        for framing in [annexb_au, avcc_au] {
            assert!(au_is_keyframe(H264, &framing(&[sps, pps, idr])));
            assert!(!au_is_keyframe(H264, &framing(&[non_idr])));
        }
    }

    #[test]
    fn h265_keyframe_is_any_irap_in_either_framing() {
        use g2g_core::VideoCodec::H265;
        // nal_unit_type sits in bits 1..6 of the first header byte
        let nal = |nal_unit_type: u8| [nal_unit_type << 1, 0x01];
        const TRAIL_R: u8 = 1;
        const BLA_W_LP: u8 = 16;
        const IDR_W_RADL: u8 = 19;
        const CRA_NUT: u8 = 21;
        for framing in [annexb_au, avcc_au] {
            for irap in [BLA_W_LP, IDR_W_RADL, CRA_NUT] {
                assert!(au_is_keyframe(H265, &framing(&[&nal(irap)])), "{irap}");
            }
            assert!(!au_is_keyframe(H265, &framing(&[&nal(TRAIL_R)])));
        }
    }

    /// MPEG-1/2 sync points: an I-picture, and the sequence / GOP headers that
    /// only ever open one. A P- or B-picture is not a resume point.
    #[test]
    fn mpeg2_keyframes_are_intra_pictures_and_sequence_starts() {
        use g2g_core::VideoCodec::Mpeg2;
        // picture header: temporal_reference then picture_coding_type in bits 5..3.
        let picture = |coding_type: u8| {
            vec![
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                (coding_type << 3) | 0x07,
                0xFF,
            ]
        };
        assert!(au_is_keyframe(Mpeg2, &picture(1)), "an I-picture");
        assert!(!au_is_keyframe(Mpeg2, &picture(2)), "a P-picture");
        assert!(!au_is_keyframe(Mpeg2, &picture(3)), "a B-picture");

        let mut with_sequence = vec![0x00, 0x00, 0x01, 0xB3, 0x16, 0x01, 0x20, 0x13];
        with_sequence.extend_from_slice(&picture(1));
        assert!(au_is_keyframe(Mpeg2, &with_sequence));

        let mut gop_then_p = vec![0x00, 0x00, 0x01, 0xB8, 0x00, 0x00, 0x00, 0x00];
        gop_then_p.extend_from_slice(&picture(2));
        assert!(au_is_keyframe(Mpeg2, &gop_then_p), "a GOP header opens one");

        // A slice, and a truncated picture header, are neither.
        assert!(!au_is_keyframe(Mpeg2, &[0x00, 0x00, 0x01, 0x01, 0xFF]));
        assert!(!au_is_keyframe(Mpeg2, &[0x00, 0x00, 0x01, 0x00, 0x00]));
        assert!(!au_is_keyframe(Mpeg2, &[]));
    }
}

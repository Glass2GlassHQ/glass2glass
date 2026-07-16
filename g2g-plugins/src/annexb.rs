//! Annex-B/AVCC NAL splitting, avcC / parameter-set, and RBSP bitstream
//! helpers, shared by the H.264/H.265 parsers, the WebCodecs payloader
//! (`h264util`), the RTP packetizer (`rtppay`), and the MP4 / FLV demuxers.

use alloc::vec::Vec;

use g2g_core::{G2gError, VideoCodec};

/// Find the next Annex-B start code (`00 00 01` or `00 00 00 01`) at or after
/// `from`. Returns `(start_index, payload_index)`: the offset of the start code
/// and the offset of the NAL byte just past it.
pub(crate) fn next_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                return Some((i, i + 3));
            }
            if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                return Some((i, i + 4));
            }
        }
        i += 1;
    }
    None
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
        let len = prefix.iter().fold(0usize, |acc, &b| (acc << 8) | b as usize);
        let end = start.checked_add(len)?;
        if end > self.data.len() {
            return None;
        }
        self.pos = end;
        Some(&self.data[start..end])
    }
}

pub(crate) fn avcc_nal_units(data: &[u8]) -> AvccNals<'_> {
    AvccNals { data, pos: 0, len_size: 4 }
}

/// [`avcc_nal_units`] with an explicit prefix width, for containers whose config
/// record declares a non-4-byte NAL length (FLV's `avcC` `lengthSizeMinusOne`).
/// `len_size` outside 1..=4 yields an empty iteration.
pub(crate) fn length_prefixed_nal_units(data: &[u8], len_size: usize) -> AvccNals<'_> {
    let len_size = if (1..=4).contains(&len_size) { len_size } else { usize::MAX };
    AvccNals { data, pos: 0, len_size }
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
/// BLA / IDR / CRA), or an MPEG-4 Part 2 I-VOP (VOP start code 0xB6 whose
/// vop_coding_type, the top 2 bits of the next byte, is 0). Used by the demuxer
/// seek path (M362) to snap to a decodable resume point in a stream whose units
/// carry no keyframe flag of their own.
pub(crate) fn au_is_keyframe(codec: g2g_core::VideoCodec, au: &[u8]) -> bool {
    use g2g_core::VideoCodec;
    nal_units_any(au).any(|n| match (codec, n.first()) {
        (VideoCodec::H265, Some(b)) => (16..=23).contains(&((b >> 1) & 0x3F)),
        (VideoCodec::Mpeg4Part2, Some(&0xB6)) => n.get(1).is_some_and(|c| c >> 6 == 0),
        (VideoCodec::Mpeg4Part2, _) => false,
        (_, Some(b)) => (b & 0x1F) == 5,
        (_, None) => false,
    })
}

/// Collect the H.264 SPS (type 7) and PPS (type 8) NAL units from an access unit
/// (either framing), returned as owned copies so the caller can cache them across
/// frames. VideoToolbox builds its `CMVideoFormatDescription` from the parameter
/// sets (supplied out of band), not from NALs inside the decode sample, so the
/// decoder pulls these out and feeds only the VCL NALs to each frame.
#[cfg(any(
    all(target_os = "macos", feature = "vtdecode"),
    all(target_os = "android", feature = "mediacodec"),
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
    all(target_os = "macos", any(feature = "vtdecode", feature = "vtencode")),
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
    all(target_os = "macos", any(feature = "vtdecode", feature = "vtencode")),
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

/// Convert AVCC (4-byte length-prefixed NALs) to Annex-B, each NAL preceded by a
/// 4-byte start code. The inverse of [`to_avcc`]: VideoToolbox's H.264 *encoder*
/// emits length-prefixed NALs, but the g2g pipeline is Annex-B framed
/// (downstream H.264 elements assume start codes), so the encoder converts on the
/// way out. Each NAL gets a 4-byte start code (`00 00 00 01`); the parameter sets
/// are prepended separately on keyframes (they live in the format description,
/// not the sample).
pub(crate) fn avcc_to_annexb(avcc: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(avcc.len() + 16);
    for nal in avcc_nal_units(avcc) {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
    }
    out
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

/// Whether the access unit already opens with a parameter-set NAL (so the
/// config-record sets need not be prepended): H.264 SPS(7), H.265 VPS(32).
pub(crate) fn starts_with_param_set(annexb: &[u8], codec: VideoCodec) -> bool {
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
/// Annex-B start code; MPEG-4 Part 2's single set is the esds VOL header, which
/// already carries its own 3-byte start codes and is prepended verbatim. Shared
/// by the progressive and fragmented MP4 demuxers and the FLV demuxer.
pub(crate) fn prepend_param_sets(
    annexb: &[u8],
    param_sets: &[Vec<u8>],
    codec: VideoCodec,
) -> Vec<u8> {
    let mut out = Vec::new();
    for set in param_sets {
        if codec != VideoCodec::Mpeg4Part2 {
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

/// Whether a NAL begins a keyframe: H.264 IDR (type 5), H.265 any IRAP
/// picture (types 16..=23, covering BLA/IDR/CRA).
pub(crate) fn is_keyframe_nal(codec: VideoCodec, nalu: &[u8]) -> bool {
    let ty = nalu_type(codec, nalu);
    match codec {
        VideoCodec::H265 => (16..=23).contains(&ty),
        _ => ty == 5,
    }
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

/// MSB-first bit reader over a byte slice, the shared bitstream cursor for the
/// H.264 / H.265 SPS parsers. All readers return `None` on EOF rather than
/// panicking, so a partial / malformed header propagates as "field unknown"
/// instead of aborting the pipeline.
pub(crate) struct BitReader<'a> {
    buf: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, bit_pos: 0 }
    }

    pub(crate) fn read_bit(&mut self) -> Option<u32> {
        let byte_idx = self.bit_pos / 8;
        let bit_off = 7 - (self.bit_pos % 8);
        if byte_idx >= self.buf.len() {
            return None;
        }
        let bit = u32::from((self.buf[byte_idx] >> bit_off) & 1);
        self.bit_pos += 1;
        Some(bit)
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
        let new_pos = self.bit_pos.checked_add(n)?;
        if new_pos > self.buf.len() * 8 {
            return None;
        }
        self.bit_pos = new_pos;
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
        let total_bits = self.buf.len() * 8;
        if self.bit_pos >= total_bits {
            return false;
        }
        // Locate the rbsp_stop_one_bit: the last set bit in the whole buffer.
        for byte_idx in (0..self.buf.len()).rev() {
            let b = self.buf[byte_idx];
            if b != 0 {
                let bit_in_byte = 7 - b.trailing_zeros() as usize;
                let stop_bit_pos = byte_idx * 8 + bit_in_byte;
                return self.bit_pos < stop_bit_pos;
            }
        }
        false
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
        while self.bit_pos % 8 != 0 {
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
    fn avcc_to_annexb_prefixes_each_nal_with_a_start_code() {
        let sps: &[u8] = &[0x67, 0x42, 0xC0, 0x1E];
        let idr: &[u8] = &[0x65, 0x88, 0x84, 0x21, 0x0A];
        let mut avcc = Vec::new();
        avcc.extend_from_slice(&(sps.len() as u32).to_be_bytes());
        avcc.extend_from_slice(sps);
        avcc.extend_from_slice(&(idr.len() as u32).to_be_bytes());
        avcc.extend_from_slice(idr);

        let annexb = avcc_to_annexb(&avcc);
        assert!(is_annex_b(&annexb), "output is Annex-B framed");
        let nals: Vec<&[u8]> = nal_units(&annexb).collect();
        assert_eq!(nals, vec![sps, idr], "the same NALs, now start-code framed");
        // Every NAL is preceded by a 4-byte start code.
        assert_eq!(&annexb[..4], &[0, 0, 0, 1]);
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
        assert_eq!(kept, vec![sei, idr], "parameter sets dropped, order preserved");
    }
}

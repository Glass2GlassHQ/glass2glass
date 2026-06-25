//! Pure fragmented-MP4 / CMAF box writer (M24, HEVC in M31): the fMP4 muxing
//! state machine ([`Fmp4Muxer`]) plus the NAL / `avcC` / `hvcC` helpers shared
//! across the container elements. Annex-B H.264/H.265 access units in, an
//! `ftyp`+`moov` init segment then one `moof`+`mdat` fragment per access unit
//! out, so the recording is playable (ffplay/VLC/browsers via MSE) and durable:
//! a truncated live recording stays valid up to the last complete fragment.
//!
//! The `moov` needs the stream's parameter sets, which arrive in-band with the
//! first IDR, so the init segment is emitted on the first access unit; H.264
//! carries `avc1`/`avcC` with SPS+PPS, H.265 `hvc1`/`hvcC` with VPS+SPS+PPS.
//! Samples are AVCC-style 4-byte length-prefixed NALUs. One fragment per access
//! unit favours latency/durability over container overhead (~100 bytes/frame).
//!
//! Wrapped by [`Mp4Mux`](crate::mp4mux) / the A/V [`Mp4MuxN`](crate::mp4muxn)
//! (which forward the bytes to any sink, e.g. `mp4mux ! filesink`); the helpers
//! are reused by the Matroska A/V muxer [`MkvMuxN`](crate::mkvmuxn). Pure
//! `no_std + alloc`.

use alloc::vec::Vec;

use g2g_core::{G2gError, TagList, VideoCodec};

/// 90 kHz media timescale, the conventional choice for video tracks.
const TIMESCALE: u64 = 90_000;
/// Fallback per-frame duration when the stream carries no timing: 1/30 s.
const DEFAULT_DURATION_NS: u64 = 33_333_333;

/// Pure fragmented-MP4 box writer: the muxing state machine wrapped by
/// [`crate::mp4mux::Mp4Mux`] (forwards the bytes downstream, e.g. to a
/// `filesink`) and the A/V [`crate::mp4muxn::Mp4MuxN`]. Annex-B H.264/H.265
/// access units in, an `ftyp`+`moov` init segment then one `moof`+`mdat`
/// fragment per AU out. The init segment is emitted on the first
/// [`push_au`](Self::push_au) because the `moov` needs the stream's in-band
/// parameter sets.
#[derive(Debug)]
pub(crate) struct Fmp4Muxer {
    codec: VideoCodec,
    width: u32,
    height: u32,
    tags: TagList,
    header_written: bool,
    fragments: u64,
    /// Accumulated decode time in media-timescale units (`tfdt`).
    decode_time: u64,
    prev_pts_ns: Option<u64>,
}

impl Fmp4Muxer {
    pub(crate) fn new(codec: VideoCodec, width: u32, height: u32, tags: TagList) -> Self {
        Self {
            codec,
            width,
            height,
            tags,
            header_written: false,
            fragments: 0,
            decode_time: 0,
            prev_pts_ns: None,
        }
    }

    /// Apply a mid-stream caps refinement. A geometry change before the header is
    /// written is absorbed; a codec swap after the `moov` is written is not
    /// expressible and fails loud.
    pub(crate) fn update_caps(&mut self, codec: VideoCodec, width: u32, height: u32) -> Result<(), G2gError> {
        if self.header_written && codec != self.codec {
            return Err(G2gError::CapsMismatch);
        }
        self.codec = codec;
        if width != 0 {
            self.width = width;
        }
        if height != 0 {
            self.height = height;
        }
        Ok(())
    }

    /// Mux one Annex-B access unit, returning the bytes to emit. The first call
    /// prepends the `ftyp`+`moov` init segment. `duration_ns` of 0 derives the
    /// sample duration from the PTS delta (or a default for the first frame).
    pub(crate) fn push_au(
        &mut self,
        annexb: &[u8],
        pts_ns: u64,
        duration_ns: u64,
    ) -> Result<Vec<u8>, G2gError> {
        let nalus = split_annexb(annexb);
        if nalus.is_empty() {
            return Err(G2gError::CapsMismatch);
        }
        let mut out = Vec::new();
        if !self.header_written {
            let param_sets = parameter_sets(self.codec, &nalus)?;
            out.extend_from_slice(&ftyp());
            out.extend_from_slice(&moov(self.codec, self.width, self.height, &param_sets, &self.tags));
            self.header_written = true;
        }
        // duration: explicit, else pts delta, else the default.
        let duration_ns = if duration_ns != 0 {
            duration_ns
        } else {
            match self.prev_pts_ns {
                Some(prev) if pts_ns > prev => pts_ns - prev,
                _ => DEFAULT_DURATION_NS,
            }
        };
        self.prev_pts_ns = Some(pts_ns);
        let duration = ns_to_timescale(duration_ns);
        let sample = avcc_sample(&nalus);
        let is_sync = nalus.iter().any(|n| is_keyframe_nal(self.codec, n));
        let frag = fragment(self.fragments + 1, self.decode_time, duration as u32, &sample, is_sync);
        out.extend_from_slice(&frag);
        self.fragments += 1;
        self.decode_time += duration;
        Ok(out)
    }
}

fn ns_to_timescale(ns: u64) -> u64 {
    // 90 kHz: ns * 90000 / 1e9, reduced to avoid overflow.
    ns.saturating_mul(TIMESCALE / 1000) / 1_000_000
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
fn nalu_type(codec: VideoCodec, nalu: &[u8]) -> u8 {
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

// box primitives (mp4_box/full_box/ftyp/MATRIX) shared across the MP4 elements.
use crate::mp4box::{ftyp, full_box, mp4_box, udta_with_tags, MATRIX};

// --- box writers ----------------------------------------------------------

fn moov(codec: VideoCodec, width: u32, height: u32, param_sets: &[&[u8]], tags: &TagList) -> Vec<u8> {
    let mvhd = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 8]); // creation/modification time
        p.extend_from_slice(&1000u32.to_be_bytes()); // timescale
        p.extend_from_slice(&0u32.to_be_bytes()); // duration (fragmented)
        p.extend_from_slice(&0x00010000u32.to_be_bytes()); // rate 1.0
        p.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
        p.extend_from_slice(&[0u8; 10]); // reserved
        for m in MATRIX {
            p.extend_from_slice(&m.to_be_bytes());
        }
        p.extend_from_slice(&[0u8; 24]); // pre_defined
        p.extend_from_slice(&2u32.to_be_bytes()); // next track id
        full_box(b"mvhd", 0, 0, &p)
    };

    let tkhd = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 8]); // times
        p.extend_from_slice(&1u32.to_be_bytes()); // track id
        p.extend_from_slice(&[0u8; 4]); // reserved
        p.extend_from_slice(&0u32.to_be_bytes()); // duration
        p.extend_from_slice(&[0u8; 16]); // reserved/layer/group/volume
        for m in MATRIX {
            p.extend_from_slice(&m.to_be_bytes());
        }
        p.extend_from_slice(&(width << 16).to_be_bytes()); // 16.16 width
        p.extend_from_slice(&(height << 16).to_be_bytes()); // 16.16 height
        full_box(b"tkhd", 0, 3, &p) // enabled | in_movie
    };

    let mdhd = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 8]);
        p.extend_from_slice(&(TIMESCALE as u32).to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0x55C4u16.to_be_bytes()); // language "und"
        p.extend_from_slice(&[0u8; 2]);
        full_box(b"mdhd", 0, 0, &p)
    };

    let hdlr = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 4]); // pre_defined
        p.extend_from_slice(b"vide");
        p.extend_from_slice(&[0u8; 12]); // reserved
        p.extend_from_slice(b"g2g\0");
        full_box(b"hdlr", 0, 0, &p)
    };

    let sample_entry = visual_sample_entry(codec, width, height, param_sets);

    let stbl = {
        let stsd = {
            let mut p = Vec::new();
            p.extend_from_slice(&1u32.to_be_bytes()); // entry count
            p.extend_from_slice(&sample_entry);
            full_box(b"stsd", 0, 0, &p)
        };
        let empty4 = 0u32.to_be_bytes();
        let stts = full_box(b"stts", 0, 0, &empty4);
        let stsc = full_box(b"stsc", 0, 0, &empty4);
        let stsz = full_box(b"stsz", 0, 0, &[0u8; 8]); // sample size + count
        let stco = full_box(b"stco", 0, 0, &empty4);
        mp4_box(b"stbl", &[stsd, stts, stsc, stsz, stco].concat())
    };

    let minf = {
        let vmhd = full_box(b"vmhd", 0, 1, &[0u8; 8]);
        let dref = {
            let url = full_box(b"url ", 0, 1, &[]); // self-contained
            let mut p = Vec::new();
            p.extend_from_slice(&1u32.to_be_bytes());
            p.extend_from_slice(&url);
            full_box(b"dref", 0, 0, &p)
        };
        let dinf = mp4_box(b"dinf", &dref);
        mp4_box(b"minf", &[vmhd, dinf, stbl].concat())
    };

    let mdia = mp4_box(b"mdia", &[mdhd, hdlr, minf].concat());
    let trak = mp4_box(b"trak", &[tkhd, mdia].concat());

    let mvex = {
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_be_bytes()); // track id
        p.extend_from_slice(&1u32.to_be_bytes()); // default sample description
        p.extend_from_slice(&[0u8; 12]); // default duration/size/flags
        let trex = full_box(b"trex", 0, 0, &p);
        mp4_box(b"mvex", &trex)
    };

    // Optional iTunes-style metadata after the track boxes.
    let udta = udta_with_tags(tags).unwrap_or_default();
    mp4_box(b"moov", &[mvhd, trak, mvex, udta].concat())
}

/// The visual sample entry for the track: `avc1`+`avcC` for H.264,
/// `hvc1`+`hvcC` for H.265. `param_sets` is the ordered set the moov needs
/// (SPS,PPS for H.264; VPS,SPS,PPS for H.265).
pub(crate) fn visual_sample_entry(
    codec: VideoCodec,
    width: u32,
    height: u32,
    param_sets: &[&[u8]],
) -> Vec<u8> {
    let (fourcc, config): (&[u8; 4], Vec<u8>) = match codec {
        VideoCodec::H265 => (b"hvc1", hvcc(param_sets)),
        _ => (b"avc1", avcc(param_sets)),
    };

    let mut p = Vec::new();
    p.extend_from_slice(&[0u8; 6]); // reserved
    p.extend_from_slice(&1u16.to_be_bytes()); // data reference index
    p.extend_from_slice(&[0u8; 16]); // pre_defined/reserved
    p.extend_from_slice(&(width as u16).to_be_bytes());
    p.extend_from_slice(&(height as u16).to_be_bytes());
    p.extend_from_slice(&0x00480000u32.to_be_bytes()); // 72 dpi horiz
    p.extend_from_slice(&0x00480000u32.to_be_bytes()); // 72 dpi vert
    p.extend_from_slice(&[0u8; 4]); // reserved
    p.extend_from_slice(&1u16.to_be_bytes()); // frame count
    p.extend_from_slice(&[0u8; 32]); // compressor name
    p.extend_from_slice(&0x0018u16.to_be_bytes()); // depth 24
    p.extend_from_slice(&0xFFFFu16.to_be_bytes()); // pre_defined -1
    p.extend_from_slice(&config);
    mp4_box(fourcc, &p)
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

/// `avcC` decoder configuration record box. `param_sets` is [SPS, PPS].
fn avcc(param_sets: &[&[u8]]) -> Vec<u8> {
    mp4_box(b"avcC", &avcc_record(param_sets))
}

/// VP8 keyframe flag: the frame tag's bit 0 (`0` = key frame). Shared by the
/// container muxers that store VP8 frames verbatim (Matroska, MP4).
pub(crate) fn vp8_keyframe(frame: &[u8]) -> bool {
    frame.first().is_some_and(|b| b & 1 == 0)
}

/// VP9 keyframe flag from the uncompressed frame header: frame_marker(2)=0b10,
/// profile(2) (+1 reserved bit for profile 3), show_existing_frame(1), then
/// frame_type(1) where `0` = key frame. Superframes are not unpacked (the vpx
/// encoder emits a single frame per buffer). Shared by the VP9-carrying muxers.
pub(crate) fn vp9_keyframe(frame: &[u8]) -> bool {
    let Some(&b0) = frame.first() else { return false };
    let bit = |i: u32| (b0 >> (7 - i)) & 1;
    if ((bit(0) << 1) | bit(1)) != 0b10 {
        return false; // not a valid VP9 frame marker
    }
    let profile = (bit(3) << 1) | bit(2);
    let mut cursor: u32 = 4;
    if profile == 3 {
        cursor += 1; // reserved_zero
    }
    if bit(cursor) == 1 {
        return false; // show_existing_frame: a repeat, not a key frame
    }
    cursor += 1;
    bit(cursor) == 0 // frame_type: 0 = key frame
}

/// `hvcC` decoder configuration record. `param_sets` is [VPS, SPS, PPS]. The
/// 12-byte general profile_tier_level is copied from the SPS (it sits right
/// after the 2-byte NAL header and the 1-byte sps_video_parameter_set_id /
/// max_sub_layers / nesting field); the remaining descriptive fields are set
/// to the 4:2:0 8-bit defaults the MS HEVC encoder produces. Parameter sets
/// stay in-band in each sample regardless, so a player re-parses authoritative
/// values from the SPS.
fn hvcc(param_sets: &[&[u8]]) -> Vec<u8> {
    mp4_box(b"hvcC", &hvcc_record(param_sets))
}

/// The `hvcC` HEVCDecoderConfigurationRecord body (no box header), shared as the
/// Matroska `CodecPrivate` for `V_MPEGH/ISO/HEVC`. See [`hvcc`] for the field
/// layout; `param_sets` is [VPS, SPS, PPS].
pub(crate) fn hvcc_record(param_sets: &[&[u8]]) -> Vec<u8> {
    let vps = param_sets[0];
    let sps = param_sets[1];
    let pps = param_sets[2];

    let mut ptl = [0u8; 12];
    if let Some(src) = sps.get(3..15) {
        ptl.copy_from_slice(src);
    }

    let mut p = Vec::new();
    p.push(1); // configuration version
    p.extend_from_slice(&ptl); // general profile_tier_level (12 bytes)
    p.extend_from_slice(&0xF000u16.to_be_bytes()); // reserved + min_spatial_segmentation_idc 0
    p.push(0xFC); // reserved + parallelismType 0
    p.push(0xFC | 1); // reserved + chromaFormat 4:2:0
    p.push(0xF8); // reserved + bitDepthLumaMinus8 0
    p.push(0xF8); // reserved + bitDepthChromaMinus8 0
    p.extend_from_slice(&0u16.to_be_bytes()); // avgFrameRate (unspecified)
    // constantFrameRate(2)=0 | numTemporalLayers(3)=1 | temporalIdNested(1)=1
    // | lengthSizeMinusOne(2)=3
    p.push((1 << 3) | (1 << 2) | 3);
    p.push(3); // numOfArrays: VPS, SPS, PPS

    for (ty, nalu) in [(32u8, vps), (33u8, sps), (34u8, pps)] {
        p.push(0x80 | ty); // array_completeness=1, NAL_unit_type
        p.extend_from_slice(&1u16.to_be_bytes()); // numNalus
        p.extend_from_slice(&(nalu.len() as u16).to_be_bytes());
        p.extend_from_slice(nalu);
    }
    p
}

/// One `moof`+`mdat` fragment holding a single sample.
fn fragment(
    sequence: u64,
    decode_time: u64,
    duration: u32,
    sample: &[u8],
    is_sync: bool,
) -> Vec<u8> {
    // I-frame: depends on nothing; otherwise depends-on + non-sync.
    let sample_flags: u32 = if is_sync { 0x0200_0000 } else { 0x0101_0000 };

    let build_moof = |data_offset: u32| -> Vec<u8> {
        let mfhd = full_box(b"mfhd", 0, 0, &(sequence as u32).to_be_bytes());
        let tfhd = {
            let p = 1u32.to_be_bytes(); // track id
            full_box(b"tfhd", 0, 0x020000, &p) // default-base-is-moof
        };
        let tfdt = full_box(b"tfdt", 1, 0, &decode_time.to_be_bytes());
        let trun = {
            let mut p = Vec::new();
            p.extend_from_slice(&1u32.to_be_bytes()); // sample count
            p.extend_from_slice(&data_offset.to_be_bytes());
            p.extend_from_slice(&duration.to_be_bytes());
            p.extend_from_slice(&(sample.len() as u32).to_be_bytes());
            p.extend_from_slice(&sample_flags.to_be_bytes());
            // data-offset | duration | size | flags present
            full_box(b"trun", 0, 0x000701, &p)
        };
        let traf = mp4_box(b"traf", &[tfhd, tfdt, trun].concat());
        mp4_box(b"moof", &[mfhd, traf].concat())
    };

    // the trun data offset points past the moof and the mdat header, both
    // size-stable, so one rebuild with the measured size is exact.
    let moof_len = build_moof(0).len() as u32;
    let moof = build_moof(moof_len + 8);
    let mdat = mp4_box(b"mdat", sample);
    [moof, mdat].concat()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn annexb_splitter_handles_both_start_codes() {
        // 4-byte code, 3-byte code, trailing NALU
        let data = [
            0, 0, 0, 1, 0x67, 0xAA, // SPS
            0, 0, 1, 0x68, 0xBB, // PPS
            0, 0, 0, 1, 0x65, 0xCC, 0xDD, // IDR
        ];
        let nalus = split_annexb(&data);
        assert_eq!(nalus.len(), 3);
        assert_eq!(nalu_type(VideoCodec::H264, nalus[0]), 7);
        assert_eq!(nalu_type(VideoCodec::H264, nalus[1]), 8);
        assert_eq!(nalu_type(VideoCodec::H264, nalus[2]), 5);
        assert_eq!(nalus[2], &[0x65, 0xCC, 0xDD]);
    }

    #[test]
    fn hevc_nal_type_and_keyframe_detection() {
        // H.265 NAL type is bits 1..6 of byte 0. VPS=32 -> 0x40, IDR_W_RADL=19
        // -> 0x26, a non-IRAP slice (TRAIL_R=1) -> 0x02.
        assert_eq!(nalu_type(VideoCodec::H265, &[0x40, 0x01]), 32);
        assert_eq!(nalu_type(VideoCodec::H265, &[0x26, 0x01]), 19);
        assert!(is_keyframe_nal(VideoCodec::H265, &[0x26, 0x01])); // IDR
        assert!(!is_keyframe_nal(VideoCodec::H265, &[0x02, 0x01])); // TRAIL_R
        assert!(is_keyframe_nal(VideoCodec::H264, &[0x65])); // IDR
        assert!(!is_keyframe_nal(VideoCodec::H264, &[0x61])); // non-IDR slice
    }

    #[test]
    fn hvcc_carries_three_arrays_and_copies_ptl() {
        // synthetic VPS(32)/SPS(33)/PPS(34); SPS holds a 12-byte PTL after its
        // 3-byte prefix so hvcC copies it verbatim.
        let vps: &[u8] = &[0x40, 0x01, 0xAA];
        let mut sps_v = vec![0x42u8, 0x01, 0x00];
        sps_v.extend((1u8..=12).map(|b| b * 7)); // recognisable PTL bytes
        let sps: &[u8] = &sps_v;
        let pps: &[u8] = &[0x44, 0x01, 0xCC];
        let cfg = hvcc(&[vps, sps, pps]);
        assert_eq!(&cfg[4..8], b"hvcC");
        let payload = &cfg[8..];
        assert_eq!(payload[0], 1, "configuration version");
        assert_eq!(&payload[1..13], &sps_v[3..15], "general PTL copied from SPS");
        assert_eq!(payload[22], 3, "numOfArrays = VPS, SPS, PPS");
        // the three parameter sets must appear in the record
        assert!(cfg.windows(vps.len()).any(|w| w == vps));
        assert!(cfg.windows(pps.len()).any(|w| w == pps));
    }

    #[test]
    fn avcc_sample_length_prefixes_every_nalu() {
        let nalus: Vec<&[u8]> = vec![&[0x67, 1, 2], &[0x65, 3]];
        let s = avcc_sample(&nalus);
        assert_eq!(
            s,
            vec![0, 0, 0, 3, 0x67, 1, 2, 0, 0, 0, 2, 0x65, 3],
            "4-byte BE length before each NALU"
        );
    }

    #[test]
    fn ns_to_timescale_is_90khz() {
        assert_eq!(ns_to_timescale(1_000_000_000), 90_000);
        assert_eq!(ns_to_timescale(33_333_333), 2999);
    }

    #[test]
    fn boxes_carry_size_and_type() {
        let b = mp4_box(b"mdat", &[1, 2, 3]);
        assert_eq!(&b[..4], &11u32.to_be_bytes());
        assert_eq!(&b[4..8], b"mdat");
        assert_eq!(&b[8..], &[1, 2, 3]);
    }

    #[test]
    fn trun_data_offset_points_at_the_mdat_payload() {
        let frag = fragment(1, 0, 3000, &[9, 9, 9, 9], true);
        // moof size from its own header
        let moof_len = u32::from_be_bytes(frag[..4].try_into().unwrap()) as usize;
        // mdat payload begins after the mdat 8-byte header
        let payload_at = moof_len + 8;
        assert_eq!(&frag[payload_at..payload_at + 4], &[9, 9, 9, 9]);
        // the trun's data_offset (relative to moof start) must equal that.
        // locate trun: search for the fourcc and read its data_offset field.
        let pos = frag.windows(4).position(|w| w == b"trun").unwrap();
        let data_offset =
            u32::from_be_bytes(frag[pos + 12..pos + 16].try_into().unwrap()) as usize;
        assert_eq!(data_offset, payload_at);
    }
}

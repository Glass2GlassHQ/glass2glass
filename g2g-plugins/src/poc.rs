//! Picture order count (H.264 8.2.1, H.265 8.3.1) for callers outside a decoder.
//!
//! Display order is written down in exactly one place in a coded H.264 / H.265
//! stream: the picture order count each slice header derives. A demuxer that
//! wants to time an access unit a container left unstamped needs it, so the
//! parses here read the parameter-set and slice-header fields that count and
//! nothing else.
//!
//! They are deliberately permissive. A stream the Vulkan decoder refuses
//! (scaling matrices, 4:4:4, POC type 1, field pictures) still yields a picture
//! order count, because timing such a stream is useful even when this tree
//! cannot decode it. A strict caller wraps these and applies its own rejections
//! on top: `vulkanvideo` parses the slice-header prefix here and then refuses
//! the field pictures its DPB loop cannot run.
//!
//! Every field comes off the wire, so counts are bounded before they drive a
//! loop and the order-count arithmetic is folded in `i64` with saturating ops:
//! a malformed header returns `None` instead of panicking.

use alloc::vec::Vec;

use g2g_core::VideoCodec;

use crate::annexb::{nal_units_any, strip_emulation_prevention, BitReader};

/// Largest `log2_max_frame_num_minus4` / `log2_max_pic_order_cnt_lsb_minus4` the
/// specs allow (H.264 7.4.2.1.1, H.265 7.4.3.2.1). A larger value would shift
/// past the width the counters are held in, so the parse rejects it.
pub(crate) const MAX_LOG2_MINUS4: u32 = 12;

/// Upper bound on `num_ref_frames_in_pic_order_cnt_cycle` (H.264 7.4.2.1.1),
/// which sizes the offset list a POC-type-1 stream codes in its SPS.
const MAX_POC_CYCLE_LENGTH: u32 = 255;

/// How many parameter sets [`AccessUnitPoc`] keeps per stream. Streams carry one
/// or two picture parameter sets; past this a new id replaces the oldest entry
/// rather than growing the table on a stream that churns ids.
const MAX_TRACKED_PPS: usize = 8;

/// The parameter-set fields picture order count derives from, recovered by the
/// H.264 and H.265 SPS parses (see [`SpsGeometry`](crate::nalparse::SpsGeometry)).
///
/// H.265 has only the H.264 type-0 scheme, so it fills `pic_order_cnt_type` 0,
/// `log2_max_pic_order_cnt_lsb` and `separate_colour_plane_flag`, and leaves the
/// `frame_num` and cycle-offset fields at their defaults.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpsPocParams {
    pub pic_order_cnt_type: u8,
    pub log2_max_pic_order_cnt_lsb: u32,
    pub log2_max_frame_num: u32,
    pub delta_pic_order_always_zero_flag: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    /// `offset_for_ref_frame[]`, one per `num_ref_frames_in_pic_order_cnt_cycle`
    /// (empty for every POC type but 1).
    pub offsets_for_ref_frame: Vec<i32>,
    pub frame_mbs_only_flag: bool,
    pub separate_colour_plane_flag: bool,
}

impl SpsPocParams {
    /// The H.264 slice-parse / POC view of these fields.
    pub fn h264_context(&self) -> H264PocContext<'_> {
        H264PocContext {
            separate_colour_plane_flag: self.separate_colour_plane_flag,
            log2_max_frame_num: self.log2_max_frame_num,
            frame_mbs_only_flag: self.frame_mbs_only_flag,
            pic_order_cnt_type: self.pic_order_cnt_type,
            log2_max_pic_order_cnt_lsb: self.log2_max_pic_order_cnt_lsb,
            delta_pic_order_always_zero_flag: self.delta_pic_order_always_zero_flag,
            offset_for_non_ref_pic: self.offset_for_non_ref_pic,
            offset_for_top_to_bottom_field: self.offset_for_top_to_bottom_field,
            offsets_for_ref_frame: &self.offsets_for_ref_frame,
        }
    }

    /// The H.265 slice-parse / POC view of these fields.
    pub fn h265_context(&self) -> H265PocContext {
        H265PocContext {
            log2_max_pic_order_cnt_lsb: self.log2_max_pic_order_cnt_lsb,
            separate_colour_plane_flag: self.separate_colour_plane_flag,
        }
    }
}

// ============================================================================
// H.264
// ============================================================================

/// The sequence-parameter-set fields the H.264 slice-header prefix parse and the
/// POC derivation read. Copy, so a caller holding its own SPS type can build one
/// per picture without allocating.
#[derive(Debug, Clone, Copy)]
pub struct H264PocContext<'a> {
    pub separate_colour_plane_flag: bool,
    /// `log2_max_frame_num_minus4 + 4`, the width of `frame_num`.
    pub log2_max_frame_num: u32,
    pub frame_mbs_only_flag: bool,
    pub pic_order_cnt_type: u8,
    /// `log2_max_pic_order_cnt_lsb_minus4 + 4`, the width of `pic_order_cnt_lsb`.
    pub log2_max_pic_order_cnt_lsb: u32,
    pub delta_pic_order_always_zero_flag: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub offsets_for_ref_frame: &'a [i32],
}

/// The picture-parameter-set field an H.264 slice header needs to be readable up
/// to its order-count syntax, keyed by the id the slice selects it with.
#[derive(Debug, Clone, Copy, Default)]
pub struct H264PpsPoc {
    pub pic_parameter_set_id: u8,
    pub bottom_field_pic_order_in_frame_present_flag: bool,
}

/// The H.264 slice-header prefix: everything up to and including the picture
/// order count syntax, which is where `ref_pic_list_modification` begins.
#[derive(Debug, Clone, Copy, Default)]
pub struct H264SlicePoc {
    pub first_mb_in_slice: u32,
    /// Raw `slice_type` (0..=9); `% 5` gives P/B/I/SP/SI.
    pub slice_type: u32,
    pub pic_parameter_set_id: u8,
    pub frame_num: u32,
    pub field_pic_flag: bool,
    pub bottom_field_flag: bool,
    /// Only meaningful for an IDR.
    pub idr_pic_id: u32,
    /// `nal_unit_type == 5`: an IDR, which resets the order-count state.
    pub is_idr: bool,
    /// `nal_ref_idc` from the NAL header: 0 means the picture is not a reference.
    pub nal_ref_idc: u8,
    /// `pic_order_cnt_lsb` (POC type 0 only; 0 otherwise).
    pub pic_order_cnt_lsb: u32,
    /// `delta_pic_order_cnt_bottom` (POC type 0 with bottom-field POC present).
    pub delta_pic_order_cnt_bottom: i32,
    /// `delta_pic_order_cnt[0..2]` (POC type 1 only).
    pub delta_pic_order_cnt: [i32; 2],
    /// The slice's `dec_ref_pic_marking()`, which only
    /// [`parse_h264_slice_marking`] reads: `None` means it was not parsed, not
    /// that the slice carries none. A decoder that manages a reference list must
    /// refuse a `None` rather than assume the default marking, since a stream
    /// using adaptive marking would then decode against the wrong references.
    pub ref_pic_marking: Option<H264RefPicMarking>,
}

/// One `memory_management_control_operation` of a slice's `dec_ref_pic_marking()`
/// (H.264 7.4.3.3), holding the operands as coded. Operation 0 ends the list and
/// is not represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H264Mmco {
    /// 1: mark a short-term picture unused for reference. The picture is
    /// `CurrPicNum - (difference_of_pic_nums_minus1 + 1)`.
    ShortTermUnused { difference_of_pic_nums_minus1: u32 },
    /// 2: mark a long-term picture unused for reference.
    LongTermUnused { long_term_pic_num: u32 },
    /// 3: give a short-term picture a long-term index.
    AssignLongTerm {
        difference_of_pic_nums_minus1: u32,
        long_term_frame_idx: u32,
    },
    /// 4: set the largest long-term frame index in use (`plus1 == 0` means no
    /// long-term references).
    MaxLongTermIndex { max_long_term_frame_idx_plus1: u32 },
    /// 5: mark every reference picture unused and reset the order count.
    AllUnused,
    /// 6: mark the current picture as a long-term reference.
    CurrentAsLongTerm { long_term_frame_idx: u32 },
}

/// The most operations one `dec_ref_pic_marking()` is read as carrying. Real
/// streams use one or two (x264's B-pyramid unmarks a single short-term
/// reference); a longer list makes the parse refuse the header rather than drop
/// an operation that changes which pictures are references.
pub const MAX_MMCO_OPS: usize = 8;

/// A slice's `dec_ref_pic_marking()` (H.264 7.3.3.3). Fixed-capacity so the
/// slice header stays `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct H264RefPicMarking {
    /// IDR only: the prior pictures need not be output before decoding resumes.
    pub no_output_of_prior_pics: bool,
    /// IDR only: this picture becomes a long-term reference.
    pub long_term_reference: bool,
    /// Non-IDR: the operations below replace the default sliding-window marking.
    pub adaptive: bool,
    ops: [H264Mmco; MAX_MMCO_OPS],
    len: u8,
}

impl Default for H264RefPicMarking {
    fn default() -> Self {
        Self {
            no_output_of_prior_pics: false,
            long_term_reference: false,
            adaptive: false,
            ops: [H264Mmco::AllUnused; MAX_MMCO_OPS],
            len: 0,
        }
    }
}

impl H264RefPicMarking {
    /// The operations in coded order (empty unless `adaptive`).
    pub fn ops(&self) -> &[H264Mmco] {
        &self.ops[..self.len as usize]
    }

    fn push(&mut self, op: H264Mmco) -> Option<()> {
        let slot = self.ops.get_mut(self.len as usize)?;
        *slot = op;
        self.len += 1;
        Some(())
    }
}

/// The picture-parameter-set fields the syntax between the order count and
/// `dec_ref_pic_marking()` is sized by, plus `ChromaArrayType` (from the SPS,
/// which sizes `pred_weight_table`'s chroma entries).
#[derive(Debug, Clone, Copy, Default)]
pub struct H264PpsRefMarking {
    pub redundant_pic_cnt_present_flag: bool,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_idc: u8,
    pub num_ref_idx_l0_default_active_minus1: u32,
    pub num_ref_idx_l1_default_active_minus1: u32,
    pub chroma_array_type: u8,
}

/// Largest `num_ref_idx_lX_active_minus1` the spec allows (H.264 7.4.3), which
/// bounds the `pred_weight_table` loops a malformed header could otherwise run.
const MAX_NUM_REF_IDX_MINUS1: u32 = 31;

/// Iterations the `ref_pic_list_modification` loop is read for before the parse
/// gives up: the list is bounded by the reference count, and a stream coding more
/// than this is refused rather than spun on.
const MAX_REF_LIST_MODIFICATIONS: usize = 64;

impl H264SlicePoc {
    /// Whether this is an I (intra) slice: `slice_type % 5 == 2`.
    pub fn is_intra_slice(&self) -> bool {
        self.slice_type % 5 == 2
    }
}

/// Parse the picture-order-count prefix of an H.264 slice NAL (type 1 or 5),
/// header byte and emulation-prevention included. `pps` is the set of picture
/// parameter sets seen so far; the slice selects one by id.
pub fn parse_h264_slice_poc(
    nal: &[u8],
    sps: &H264PocContext<'_>,
    pps: &[H264PpsPoc],
) -> Option<H264SlicePoc> {
    if nal.is_empty() {
        return None;
    }
    let nal_ref_idc = (nal[0] >> 5) & 0x3;
    let nal_unit_type = nal[0] & 0x1F;
    if nal_unit_type != 1 && nal_unit_type != 5 {
        return None;
    }
    let rbsp = strip_emulation_prevention(&nal[1..]);
    let mut br = BitReader::new(&rbsp);
    parse_h264_slice_poc_bits(&mut br, nal_ref_idc, nal_unit_type == 5, sps, pps)
}

/// [`parse_h264_slice_poc`] carried on through `dec_ref_pic_marking()`, for a
/// decoder that has to keep a reference list. Everything between the two (the
/// reference-list modification and the prediction weight table) is read only to
/// get past it, so `marking` must describe the picture parameter set the slice
/// selects, else the reference marking is read from the wrong bit.
///
/// `None` on a malformed header, exactly as [`parse_h264_slice_poc`].
pub fn parse_h264_slice_marking(
    nal: &[u8],
    sps: &H264PocContext<'_>,
    pps: &[H264PpsPoc],
    marking: &H264PpsRefMarking,
) -> Option<H264SlicePoc> {
    if nal.is_empty() {
        return None;
    }
    let nal_ref_idc = (nal[0] >> 5) & 0x3;
    let nal_unit_type = nal[0] & 0x1F;
    if nal_unit_type != 1 && nal_unit_type != 5 {
        return None;
    }
    let rbsp = strip_emulation_prevention(&nal[1..]);
    let mut br = BitReader::new(&rbsp);
    let mut slice = parse_h264_slice_poc_bits(&mut br, nal_ref_idc, nal_unit_type == 5, sps, pps)?;
    skip_to_ref_pic_marking(&mut br, &slice, marking)?;
    slice.ref_pic_marking = Some(read_dec_ref_pic_marking(&mut br, &slice)?);
    Some(slice)
}

/// Read past `redundant_pic_cnt` .. `pred_weight_table()`, leaving `br` at
/// `dec_ref_pic_marking()` (H.264 7.3.3).
fn skip_to_ref_pic_marking(
    br: &mut BitReader<'_>,
    slice: &H264SlicePoc,
    pps: &H264PpsRefMarking,
) -> Option<()> {
    let slice_type = slice.slice_type % 5;
    let is_p = slice_type == 0 || slice_type == 3;
    let is_b = slice_type == 1;
    let is_intra = slice_type == 2 || slice_type == 4;
    if pps.redundant_pic_cnt_present_flag {
        br.read_ue()?;
    }
    if is_b {
        br.read_bit()?; // direct_spatial_mv_pred_flag
    }
    let mut num_ref_idx_l0_minus1 = pps.num_ref_idx_l0_default_active_minus1;
    let mut num_ref_idx_l1_minus1 = pps.num_ref_idx_l1_default_active_minus1;
    if is_p || is_b {
        if br.read_bit()? == 1 {
            num_ref_idx_l0_minus1 = br.read_ue()?;
            if is_b {
                num_ref_idx_l1_minus1 = br.read_ue()?;
            }
        }
        if num_ref_idx_l0_minus1 > MAX_NUM_REF_IDX_MINUS1
            || num_ref_idx_l1_minus1 > MAX_NUM_REF_IDX_MINUS1
        {
            return None;
        }
    }

    // ref_pic_list_modification (7.3.3.1): each entry is an op plus one operand,
    // op 3 ends the list.
    if !is_intra {
        read_ref_pic_list_modification(br)?;
    }
    if is_b {
        read_ref_pic_list_modification(br)?;
    }

    let weighted = (pps.weighted_pred_flag && is_p) || (pps.weighted_bipred_idc == 1 && is_b);
    if weighted {
        br.read_ue()?; // luma_log2_weight_denom
        if pps.chroma_array_type != 0 {
            br.read_ue()?; // chroma_log2_weight_denom
        }
        read_weight_list(br, num_ref_idx_l0_minus1, pps.chroma_array_type)?;
        if is_b {
            read_weight_list(br, num_ref_idx_l1_minus1, pps.chroma_array_type)?;
        }
    }
    Some(())
}

/// One `ref_pic_list_modification` list (H.264 7.3.3.1).
fn read_ref_pic_list_modification(br: &mut BitReader<'_>) -> Option<()> {
    if br.read_bit()? != 1 {
        return Some(());
    }
    for _ in 0..MAX_REF_LIST_MODIFICATIONS {
        let op = br.read_ue()?;
        if op == 3 {
            return Some(());
        }
        if op > 3 {
            return None;
        }
        br.read_ue()?; // abs_diff_pic_num_minus1 / long_term_pic_num
    }
    None
}

/// One list of `pred_weight_table` entries (H.264 7.3.3.2).
fn read_weight_list(
    br: &mut BitReader<'_>,
    num_ref_idx_minus1: u32,
    chroma_array_type: u8,
) -> Option<()> {
    for _ in 0..=num_ref_idx_minus1 {
        if br.read_bit()? == 1 {
            br.read_se()?; // luma_weight
            br.read_se()?; // luma_offset
        }
        if chroma_array_type != 0 && br.read_bit()? == 1 {
            for _ in 0..2 {
                br.read_se()?; // chroma_weight
                br.read_se()?; // chroma_offset
            }
        }
    }
    Some(())
}

/// `dec_ref_pic_marking()` (H.264 7.3.3.3). A non-reference picture codes none,
/// which reads as the default marking.
fn read_dec_ref_pic_marking(
    br: &mut BitReader<'_>,
    slice: &H264SlicePoc,
) -> Option<H264RefPicMarking> {
    let mut marking = H264RefPicMarking::default();
    if slice.nal_ref_idc == 0 {
        return Some(marking);
    }
    if slice.is_idr {
        marking.no_output_of_prior_pics = br.read_bit()? == 1;
        marking.long_term_reference = br.read_bit()? == 1;
        return Some(marking);
    }
    marking.adaptive = br.read_bit()? == 1;
    if !marking.adaptive {
        return Some(marking);
    }
    for _ in 0..=MAX_MMCO_OPS {
        let op = match br.read_ue()? {
            0 => return Some(marking),
            1 => H264Mmco::ShortTermUnused {
                difference_of_pic_nums_minus1: br.read_ue()?,
            },
            2 => H264Mmco::LongTermUnused {
                long_term_pic_num: br.read_ue()?,
            },
            3 => H264Mmco::AssignLongTerm {
                difference_of_pic_nums_minus1: br.read_ue()?,
                long_term_frame_idx: br.read_ue()?,
            },
            4 => H264Mmco::MaxLongTermIndex {
                max_long_term_frame_idx_plus1: br.read_ue()?,
            },
            5 => H264Mmco::AllUnused,
            6 => H264Mmco::CurrentAsLongTerm {
                long_term_frame_idx: br.read_ue()?,
            },
            _ => return None,
        };
        marking.push(op)?;
    }
    None
}

/// The bit-level half of [`parse_h264_slice_poc`], over an already de-emulated
/// slice RBSP. Leaves `br` positioned where `ref_pic_list_modification` begins,
/// for a caller that keeps reading.
pub(crate) fn parse_h264_slice_poc_bits(
    br: &mut BitReader<'_>,
    nal_ref_idc: u8,
    is_idr: bool,
    sps: &H264PocContext<'_>,
    pps: &[H264PpsPoc],
) -> Option<H264SlicePoc> {
    let first_mb_in_slice = br.read_ue()?;
    let slice_type = br.read_ue()?;
    let pic_parameter_set_id = br.read_ue()?;
    if pic_parameter_set_id > u32::from(u8::MAX) {
        return None;
    }
    let pic_parameter_set_id = pic_parameter_set_id as u8;
    let active = select_pps(pps, pic_parameter_set_id, |p| p.pic_parameter_set_id)?;
    if sps.separate_colour_plane_flag {
        br.read_bits(2)?; // colour_plane_id
    }
    let frame_num = br.read_bits(sps.log2_max_frame_num)?;
    let mut field_pic_flag = false;
    let mut bottom_field_flag = false;
    if !sps.frame_mbs_only_flag {
        field_pic_flag = br.read_bit()? == 1;
        if field_pic_flag {
            bottom_field_flag = br.read_bit()? == 1;
        }
    }
    let idr_pic_id = if is_idr { br.read_ue()? } else { 0 };

    let mut pic_order_cnt_lsb = 0;
    let mut delta_pic_order_cnt_bottom = 0;
    let mut delta_pic_order_cnt = [0i32; 2];
    match sps.pic_order_cnt_type {
        0 => {
            pic_order_cnt_lsb = br.read_bits(sps.log2_max_pic_order_cnt_lsb)?;
            if active.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                delta_pic_order_cnt_bottom = br.read_se()?;
            }
        }
        1 if !sps.delta_pic_order_always_zero_flag => {
            delta_pic_order_cnt[0] = br.read_se()?;
            if active.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                delta_pic_order_cnt[1] = br.read_se()?;
            }
        }
        _ => {}
    }

    Some(H264SlicePoc {
        first_mb_in_slice,
        slice_type,
        pic_parameter_set_id,
        frame_num,
        field_pic_flag,
        bottom_field_flag,
        idr_pic_id,
        is_idr,
        nal_ref_idc,
        pic_order_cnt_lsb,
        delta_pic_order_cnt_bottom,
        delta_pic_order_cnt,
        ref_pic_marking: None,
    })
}

/// Read the picture-parameter-set fields that precede the ones only a decoder
/// needs, leaving `br` positioned after `num_slice_groups_minus1`. Returns the
/// order-count field plus the ids and `num_slice_groups_minus1`, which a strict
/// caller rejects on.
pub(crate) fn parse_h264_pps_poc_bits(
    br: &mut BitReader<'_>,
) -> Option<(H264PpsPoc, u32, u32, u32)> {
    let pic_parameter_set_id = br.read_ue()?;
    let seq_parameter_set_id = br.read_ue()?;
    let entropy_coding_mode_flag = br.read_bit()?;
    let bottom_field_pic_order_in_frame_present_flag = br.read_bit()?;
    let num_slice_groups_minus1 = br.read_ue()?;
    if pic_parameter_set_id > u32::from(u8::MAX) {
        return None;
    }
    Some((
        H264PpsPoc {
            pic_parameter_set_id: pic_parameter_set_id as u8,
            bottom_field_pic_order_in_frame_present_flag:
                bottom_field_pic_order_in_frame_present_flag == 1,
        },
        seq_parameter_set_id,
        entropy_coding_mode_flag,
        num_slice_groups_minus1,
    ))
}

/// H.264 picture-order-count derivation state, carried across the pictures of
/// one stream in decoding order (H.264 8.2.1).
#[derive(Debug, Clone, Default)]
pub struct H264PocState {
    prev_poc_msb: i32,
    prev_poc_lsb: i32,
    prev_frame_num: i32,
    prev_frame_num_offset: i32,
}

impl H264PocState {
    /// The picture order count of the picture `slice` opens, advancing the state.
    pub fn compute(&mut self, sps: &H264PocContext<'_>, slice: &H264SlicePoc) -> i32 {
        match sps.pic_order_cnt_type {
            0 => self.poc_type0(sps, slice),
            1 => self.poc_type1(sps, slice),
            _ => self.poc_type2(sps, slice),
        }
    }

    /// POC type 0 (8.2.1.1): the order count is coded as a wrapping lsb, and the
    /// msb is carried forward from the last reference picture.
    fn poc_type0(&mut self, sps: &H264PocContext<'_>, slice: &H264SlicePoc) -> i32 {
        if slice.is_idr {
            self.prev_poc_msb = 0;
            self.prev_poc_lsb = 0;
        }
        let max_lsb = 1i32 << sps.log2_max_pic_order_cnt_lsb;
        let lsb = slice.pic_order_cnt_lsb as i32;
        let poc_msb = if lsb < self.prev_poc_lsb && (self.prev_poc_lsb - lsb) >= max_lsb / 2 {
            self.prev_poc_msb.saturating_add(max_lsb)
        } else if lsb > self.prev_poc_lsb && (lsb - self.prev_poc_lsb) > max_lsb / 2 {
            self.prev_poc_msb.saturating_sub(max_lsb)
        } else {
            self.prev_poc_msb
        };
        let top = i64::from(poc_msb).saturating_add(i64::from(lsb));
        let bottom = top.saturating_add(i64::from(slice.delta_pic_order_cnt_bottom));
        // Reference pictures update the prev-POC state (per 8.2.1.1);
        // non-reference pictures leave it unchanged.
        if slice.nal_ref_idc != 0 {
            self.prev_poc_msb = poc_msb;
            self.prev_poc_lsb = lsb;
        }
        pick_field(top, bottom, slice)
    }

    /// POC type 1 (8.2.1.2): the order count follows a cycle of per-frame offsets
    /// the SPS declares, indexed by the frame number.
    fn poc_type1(&mut self, sps: &H264PocContext<'_>, slice: &H264SlicePoc) -> i32 {
        let offsets = sps.offsets_for_ref_frame;
        let frame_num = i64::from(slice.frame_num);
        let frame_num_offset = self.frame_num_offset(sps, slice);
        let mut abs_frame_num = if offsets.is_empty() {
            0
        } else {
            frame_num_offset.saturating_add(frame_num)
        };
        if slice.nal_ref_idc == 0 && abs_frame_num > 0 {
            abs_frame_num -= 1;
        }
        let mut expected = 0i64;
        if abs_frame_num > 0 {
            let cycle_length = offsets.len() as i64;
            let cycle_sum: i64 = offsets
                .iter()
                .fold(0i64, |a, o| a.saturating_add(i64::from(*o)));
            let cycles = (abs_frame_num - 1) / cycle_length;
            let in_cycle = ((abs_frame_num - 1) % cycle_length) as usize;
            expected = cycles.saturating_mul(cycle_sum);
            for offset in offsets.iter().take(in_cycle + 1) {
                expected = expected.saturating_add(i64::from(*offset));
            }
        }
        if slice.nal_ref_idc == 0 {
            expected = expected.saturating_add(i64::from(sps.offset_for_non_ref_pic));
        }
        let top = expected.saturating_add(i64::from(slice.delta_pic_order_cnt[0]));
        let bottom = top
            .saturating_add(i64::from(sps.offset_for_top_to_bottom_field))
            .saturating_add(i64::from(slice.delta_pic_order_cnt[1]));
        self.advance_frame_num(slice, frame_num_offset);
        pick_field(top, bottom, slice)
    }

    /// POC type 2 (8.2.1.3): coded order is display order, so the count is
    /// derived from the frame number alone.
    fn poc_type2(&mut self, sps: &H264PocContext<'_>, slice: &H264SlicePoc) -> i32 {
        let frame_num = i64::from(slice.frame_num);
        let frame_num_offset = self.frame_num_offset(sps, slice);
        let poc = if slice.is_idr {
            0
        } else {
            let doubled = frame_num_offset.saturating_add(frame_num).saturating_mul(2);
            if slice.nal_ref_idc == 0 {
                doubled.saturating_sub(1)
            } else {
                doubled
            }
        };
        self.advance_frame_num(slice, frame_num_offset);
        clamp_poc(poc)
    }

    /// `FrameNumOffset` (8.2.1.2 / 8.2.1.3): the number of `frame_num` wraps so
    /// far, so the count keeps rising past a wrap.
    fn frame_num_offset(&self, sps: &H264PocContext<'_>, slice: &H264SlicePoc) -> i64 {
        if slice.is_idr {
            return 0;
        }
        let previous = i64::from(self.prev_frame_num_offset);
        if i64::from(self.prev_frame_num) > i64::from(slice.frame_num) {
            previous.saturating_add(1i64 << sps.log2_max_frame_num)
        } else {
            previous
        }
    }

    fn advance_frame_num(&mut self, slice: &H264SlicePoc, frame_num_offset: i64) {
        self.prev_frame_num_offset = clamp_poc(frame_num_offset);
        self.prev_frame_num = clamp_poc(i64::from(slice.frame_num));
    }
}

/// `PicOrderCnt` from the two field counts (8.2.1): a frame presents at the
/// earlier of the two, a field at its own.
fn pick_field(top: i64, bottom: i64, slice: &H264SlicePoc) -> i32 {
    let poc = if !slice.field_pic_flag {
        top.min(bottom)
    } else if slice.bottom_field_flag {
        bottom
    } else {
        top
    };
    clamp_poc(poc)
}

/// Fold an order count computed in `i64` back into the `i32` the counts are
/// carried in. Only an adversarial stream can reach the clamp.
fn clamp_poc(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// The parameter set a slice selects by id, or the sole one when the slice names
/// an id the caller never supplied (a caller that tracks one parameter set, like
/// a single-stream decoder, still parses such a slice).
fn select_pps<T: Copy>(sets: &[T], id: u8, key: impl Fn(&T) -> u8) -> Option<T> {
    if let Some(found) = sets.iter().find(|s| key(s) == id) {
        return Some(*found);
    }
    match sets {
        [only] => Some(*only),
        _ => None,
    }
}

// ============================================================================
// H.265
// ============================================================================

/// The sequence-parameter-set fields the H.265 slice-segment-header prefix parse
/// and the POC derivation read.
#[derive(Debug, Clone, Copy, Default)]
pub struct H265PocContext {
    /// `log2_max_pic_order_cnt_lsb_minus4 + 4`.
    pub log2_max_pic_order_cnt_lsb: u32,
    pub separate_colour_plane_flag: bool,
}

/// The picture-parameter-set fields an H.265 slice-segment header needs to be
/// readable up to its order count, keyed by the id the slice selects it with.
#[derive(Debug, Clone, Copy, Default)]
pub struct H265PpsPoc {
    pub pps_pic_parameter_set_id: u8,
    pub num_extra_slice_header_bits: u8,
    pub output_flag_present_flag: bool,
}

/// The H.265 slice-segment-header prefix: everything up to and including
/// `slice_pic_order_cnt_lsb`, which is where the reference-picture set begins.
#[derive(Debug, Clone, Copy, Default)]
pub struct H265SlicePoc {
    pub first_slice_segment_in_pic_flag: bool,
    pub nal_unit_type: u8,
    pub slice_pic_parameter_set_id: u8,
    /// 0 = B, 1 = P, 2 = I.
    pub slice_type: u32,
    pub slice_pic_order_cnt_lsb: u32,
    pub is_irap: bool,
    pub is_idr: bool,
}

/// Parse the picture-order-count prefix of an H.265 VCL NAL, two-byte header and
/// emulation-prevention included.
pub fn parse_h265_slice_poc(
    nal: &[u8],
    sps: &H265PocContext,
    pps: &[H265PpsPoc],
) -> Option<H265SlicePoc> {
    if nal.len() < 2 {
        return None;
    }
    let nal_unit_type = (nal[0] >> 1) & 0x3F;
    if nal_unit_type > 31 {
        return None; // not a VCL NAL
    }
    let rbsp = strip_emulation_prevention(&nal[2..]);
    let mut br = BitReader::new(&rbsp);
    parse_h265_slice_poc_bits(&mut br, nal_unit_type, sps, pps)
}

/// The bit-level half of [`parse_h265_slice_poc`], over an already de-emulated
/// slice RBSP. Leaves `br` positioned where the reference-picture set begins,
/// for a caller that keeps reading. A non-first slice segment returns with only
/// its flag set (the fields after it belong to the picture's first segment).
pub(crate) fn parse_h265_slice_poc_bits(
    br: &mut BitReader<'_>,
    nal_unit_type: u8,
    sps: &H265PocContext,
    pps: &[H265PpsPoc],
) -> Option<H265SlicePoc> {
    let is_irap = (16..=23).contains(&nal_unit_type);
    let is_idr = nal_unit_type == 19 || nal_unit_type == 20;
    let first_slice_segment_in_pic_flag = br.read_bit()? == 1;
    if !first_slice_segment_in_pic_flag {
        return Some(H265SlicePoc {
            nal_unit_type,
            is_irap,
            is_idr,
            ..H265SlicePoc::default()
        });
    }
    if is_irap {
        br.read_bit()?; // no_output_of_prior_pics_flag
    }
    let slice_pic_parameter_set_id = br.read_ue()?;
    if slice_pic_parameter_set_id > u32::from(u8::MAX) {
        return None;
    }
    let slice_pic_parameter_set_id = slice_pic_parameter_set_id as u8;
    let active = select_pps(pps, slice_pic_parameter_set_id, |p| {
        p.pps_pic_parameter_set_id
    })?;
    // dependent_slice_segment_flag / slice_segment_address only appear on a
    // segment that is not the picture's first, which returned above.
    for _ in 0..active.num_extra_slice_header_bits {
        br.read_bit()?; // slice_reserved_flag[i]
    }
    let slice_type = br.read_ue()?;
    if active.output_flag_present_flag {
        br.read_bit()?; // pic_output_flag
    }
    if sps.separate_colour_plane_flag {
        br.read_bits(2)?; // colour_plane_id
    }
    // An IDR resets the count, so it codes no lsb.
    let slice_pic_order_cnt_lsb = if is_idr {
        0
    } else {
        br.read_bits(sps.log2_max_pic_order_cnt_lsb)?
    };

    Some(H265SlicePoc {
        first_slice_segment_in_pic_flag: true,
        nal_unit_type,
        slice_pic_parameter_set_id,
        slice_type,
        slice_pic_order_cnt_lsb,
        is_irap,
        is_idr,
    })
}

/// Read the picture-parameter-set fields that precede the ones only a decoder
/// needs, leaving `br` positioned after `num_extra_slice_header_bits`. Also
/// returns `pps_seq_parameter_set_id` and `dependent_slice_segments_enabled_flag`.
pub(crate) fn parse_h265_pps_poc_bits(br: &mut BitReader<'_>) -> Option<(H265PpsPoc, u32, u32)> {
    let pps_pic_parameter_set_id = br.read_ue()?;
    let pps_seq_parameter_set_id = br.read_ue()?;
    let dependent_slice_segments_enabled_flag = br.read_bit()?;
    let output_flag_present_flag = br.read_bit()?;
    let num_extra_slice_header_bits = br.read_bits(3)?;
    if pps_pic_parameter_set_id > u32::from(u8::MAX) {
        return None;
    }
    Some((
        H265PpsPoc {
            pps_pic_parameter_set_id: pps_pic_parameter_set_id as u8,
            num_extra_slice_header_bits: num_extra_slice_header_bits as u8,
            output_flag_present_flag: output_flag_present_flag == 1,
        },
        pps_seq_parameter_set_id,
        dependent_slice_segments_enabled_flag,
    ))
}

/// Whether an H.265 VCL `nal_unit_type` marks a reference picture: the `_R`
/// (odd, < 16) trailing / leading types and every IRAP (16..=23) are references;
/// the `_N` (even, < 16) sub-layer-non-reference types are not.
pub fn h265_nal_is_reference(nal_unit_type: u8) -> bool {
    if nal_unit_type >= 16 {
        nal_unit_type <= 23
    } else {
        nal_unit_type % 2 == 1
    }
}

/// H.265 picture-order-count derivation state, carried across the pictures of
/// one stream in decoding order (H.265 8.3.1, the H.264 type-0 scheme).
#[derive(Debug, Clone, Default)]
pub struct H265PocState {
    prev_poc_msb: i32,
    prev_poc_lsb: i32,
}

impl H265PocState {
    /// The picture order count of the picture `slice` opens, advancing the state.
    pub fn compute(&mut self, sps: &H265PocContext, slice: &H265SlicePoc) -> i32 {
        if slice.is_idr {
            self.prev_poc_msb = 0;
            self.prev_poc_lsb = 0;
            return 0;
        }
        let max_lsb = 1i32 << sps.log2_max_pic_order_cnt_lsb;
        let lsb = slice.slice_pic_order_cnt_lsb as i32;
        let poc_msb = if lsb < self.prev_poc_lsb && (self.prev_poc_lsb - lsb) >= max_lsb / 2 {
            self.prev_poc_msb.saturating_add(max_lsb)
        } else if lsb > self.prev_poc_lsb && (lsb - self.prev_poc_lsb) > max_lsb / 2 {
            self.prev_poc_msb.saturating_sub(max_lsb)
        } else {
            self.prev_poc_msb
        };
        let poc = i64::from(poc_msb).saturating_add(i64::from(lsb));
        // A TemporalId-0 reference picture (not RASL / RADL / SLNR) updates the
        // prev state; sub-layer non-reference types leave it alone.
        if h265_nal_is_reference(slice.nal_unit_type) {
            self.prev_poc_msb = poc_msb;
            self.prev_poc_lsb = lsb;
        }
        clamp_poc(poc)
    }
}

// ============================================================================
// Per-stream access-unit tracking
// ============================================================================

/// Picture order count for the access units of one H.264 / H.265 elementary
/// stream: it latches the picture parameter sets as they flow in band and
/// derives the count of each access unit's first coded picture.
///
/// The sequence parameter set arrives through [`set_sps`](Self::set_sps), since
/// the callers that want this already parse it for geometry. One access unit is
/// taken to hold one coded picture, so a caller must feed access-unit-aligned
/// buffers.
#[derive(Debug, Default)]
pub struct AccessUnitPoc {
    sps: Option<SpsPocParams>,
    h264_pps: Vec<H264PpsPoc>,
    h265_pps: Vec<H265PpsPoc>,
    h264: H264PocState,
    h265: H265PocState,
}

impl AccessUnitPoc {
    /// Adopt the order-count fields of a newly parsed sequence parameter set.
    pub fn set_sps(&mut self, sps: SpsPocParams) {
        if self.sps.as_ref() == Some(&sps) {
            return;
        }
        self.sps = Some(sps);
    }

    /// Feed one access unit: cache the parameter sets it carries and return the
    /// picture order count of the coded picture it opens, or `None` when it
    /// carries no first slice this parse can read.
    pub fn push_access_unit(&mut self, codec: VideoCodec, au: &[u8]) -> Option<i32> {
        match codec {
            VideoCodec::H265 => self.push_h265(au),
            _ => self.push_h264(au),
        }
    }

    fn push_h264(&mut self, au: &[u8]) -> Option<i32> {
        let mut poc = None;
        for nal in nal_units_any(au) {
            let Some(&first) = nal.first() else {
                continue;
            };
            match first & 0x1F {
                8 => {
                    let rbsp = strip_emulation_prevention(&nal[1..]);
                    if let Some((pps, _, _, _)) =
                        parse_h264_pps_poc_bits(&mut BitReader::new(&rbsp))
                    {
                        remember(&mut self.h264_pps, pps, |p| p.pic_parameter_set_id);
                    }
                }
                1 | 5 if poc.is_none() => {
                    let sps = self.sps.as_ref()?;
                    let slice = parse_h264_slice_poc(nal, &sps.h264_context(), &self.h264_pps)?;
                    // Only the picture's first slice advances the state; the
                    // rest repeat its order count.
                    if slice.first_mb_in_slice != 0 {
                        continue;
                    }
                    poc = Some(self.h264.compute(&sps.h264_context(), &slice));
                }
                _ => {}
            }
        }
        poc
    }

    fn push_h265(&mut self, au: &[u8]) -> Option<i32> {
        let mut poc = None;
        for nal in nal_units_any(au) {
            if nal.len() < 2 {
                continue;
            }
            let nal_unit_type = (nal[0] >> 1) & 0x3F;
            if nal_unit_type == 34 {
                let rbsp = strip_emulation_prevention(&nal[2..]);
                if let Some((pps, _, _)) = parse_h265_pps_poc_bits(&mut BitReader::new(&rbsp)) {
                    remember(&mut self.h265_pps, pps, |p| p.pps_pic_parameter_set_id);
                }
            }
            if nal_unit_type <= 31 && poc.is_none() {
                let sps = self.sps.as_ref()?;
                let slice = parse_h265_slice_poc(nal, &sps.h265_context(), &self.h265_pps)?;
                if !slice.first_slice_segment_in_pic_flag {
                    continue;
                }
                poc = Some(self.h265.compute(&sps.h265_context(), &slice));
            }
        }
        poc
    }
}

/// Store a parameter set under its id, replacing the entry with that id. Past
/// [`MAX_TRACKED_PPS`] the oldest entry is dropped, so a stream that churns ids
/// cannot grow the table.
fn remember<T: Copy>(sets: &mut Vec<T>, set: T, key: impl Fn(&T) -> u8) {
    let id = key(&set);
    if let Some(slot) = sets.iter_mut().find(|s| key(s) == id) {
        *slot = set;
        return;
    }
    if sets.len() >= MAX_TRACKED_PPS {
        sets.remove(0);
    }
    sets.push(set);
}

/// Read the H.264 `pic_order_cnt_type` block of an SPS (7.3.2.1.1) into
/// [`SpsPocParams`], with the cycle-offset list bounded. `br` must sit at
/// `pic_order_cnt_type`, and is left after the block.
pub(crate) fn read_h264_poc_block(br: &mut BitReader<'_>, out: &mut SpsPocParams) -> Option<()> {
    let pic_order_cnt_type = br.read_ue()?;
    if pic_order_cnt_type > 2 {
        return None;
    }
    out.pic_order_cnt_type = pic_order_cnt_type as u8;
    match pic_order_cnt_type {
        0 => {
            let log2_minus4 = br.read_ue()?;
            if log2_minus4 > MAX_LOG2_MINUS4 {
                return None;
            }
            out.log2_max_pic_order_cnt_lsb = log2_minus4 + 4;
        }
        1 => {
            out.delta_pic_order_always_zero_flag = br.read_bit()? == 1;
            out.offset_for_non_ref_pic = br.read_se()?;
            out.offset_for_top_to_bottom_field = br.read_se()?;
            let cycle_length = br.read_ue()?;
            if cycle_length > MAX_POC_CYCLE_LENGTH {
                return None;
            }
            out.offsets_for_ref_frame = Vec::with_capacity(cycle_length as usize);
            for _ in 0..cycle_length {
                out.offsets_for_ref_frame.push(br.read_se()?);
            }
        }
        _ => {}
    }
    Some(())
}

/// Skip an H.264 `scaling_list()` block (7.3.2.1.1.1): `count` lists, each
/// either absent or a run of `delta_scale` values whose running scale the loop
/// has to follow to know where the list ends.
pub(crate) fn skip_h264_scaling_lists(br: &mut BitReader<'_>, count: usize) -> Option<()> {
    for i in 0..count {
        if br.read_bit()? == 0 {
            continue;
        }
        let size = if i < 6 { 16 } else { 64 };
        let mut last_scale = 8i32;
        let mut next_scale = 8i32;
        for _ in 0..size {
            if next_scale != 0 {
                let delta = br.read_se()?;
                next_scale = last_scale
                    .saturating_add(delta)
                    .saturating_add(256)
                    .rem_euclid(256);
            }
            if next_scale != 0 {
                last_scale = next_scale;
            }
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annexb::BitWriter;

    /// POC type 0 counts a wrapping lsb: the msb steps once the lsb wraps, so
    /// display order keeps rising past the wrap.
    #[test]
    fn poc_type0_follows_the_lsb_wrap() {
        let sps = H264PocContext {
            separate_colour_plane_flag: false,
            log2_max_frame_num: 4,
            frame_mbs_only_flag: true,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb: 4, // max_lsb 16
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            offsets_for_ref_frame: &[],
        };
        let mut state = H264PocState::default();
        let counts: Vec<i32> = [
            (0u32, true),
            (4, false),
            (8, false),
            (12, false),
            (0, false),
        ]
        .iter()
        .map(|&(lsb, is_idr)| {
            state.compute(
                &sps,
                &H264SlicePoc {
                    is_idr,
                    nal_ref_idc: 1,
                    pic_order_cnt_lsb: lsb,
                    ..H264SlicePoc::default()
                },
            )
        })
        .collect();
        assert_eq!(counts, alloc::vec![0, 4, 8, 12, 16], "the lsb wrap carries");
    }

    /// POC type 1 walks the SPS's offset cycle, so a stream that never codes a
    /// lsb still has an exact display order.
    #[test]
    fn poc_type1_walks_the_offset_cycle() {
        let offsets = [2i32];
        let sps = H264PocContext {
            separate_colour_plane_flag: false,
            log2_max_frame_num: 4,
            frame_mbs_only_flag: true,
            pic_order_cnt_type: 1,
            log2_max_pic_order_cnt_lsb: 4,
            delta_pic_order_always_zero_flag: true,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            offsets_for_ref_frame: &offsets,
        };
        let mut state = H264PocState::default();
        let counts: Vec<i32> = (0..4u32)
            .map(|frame_num| {
                state.compute(
                    &sps,
                    &H264SlicePoc {
                        is_idr: frame_num == 0,
                        nal_ref_idc: 1,
                        frame_num,
                        ..H264SlicePoc::default()
                    },
                )
            })
            .collect();
        assert_eq!(counts, alloc::vec![0, 2, 4, 6]);
    }

    /// A slice header carrying its order count round-trips through the parse,
    /// including the field flags the decoder-strict parser refuses.
    #[test]
    fn slice_poc_parse_reads_a_field_picture() {
        let mut w = BitWriter::default();
        w.write_ue(0); // first_mb_in_slice
        w.write_ue(5); // slice_type (P)
        w.write_ue(0); // pic_parameter_set_id
        w.write_bits(3, 4); // frame_num
        w.write_bit(1); // field_pic_flag
        w.write_bit(1); // bottom_field_flag
        w.write_bits(9, 4); // pic_order_cnt_lsb
        w.align_to_byte();
        let rbsp = w.into_bytes();
        let mut nal = alloc::vec![0x41u8];
        nal.extend_from_slice(&rbsp);

        let sps = H264PocContext {
            separate_colour_plane_flag: false,
            log2_max_frame_num: 4,
            frame_mbs_only_flag: false,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb: 4,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            offsets_for_ref_frame: &[],
        };
        let slice = parse_h264_slice_poc(&nal, &sps, &[H264PpsPoc::default()]).expect("parses");
        assert!(slice.field_pic_flag && slice.bottom_field_flag);
        assert_eq!(slice.frame_num, 3);
        assert_eq!(slice.pic_order_cnt_lsb, 9);
    }

    /// A truncated slice header yields nothing rather than a partial count.
    #[test]
    fn a_truncated_slice_header_is_rejected() {
        let sps = H264PocContext {
            separate_colour_plane_flag: false,
            log2_max_frame_num: 16,
            frame_mbs_only_flag: true,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb: 16,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            offsets_for_ref_frame: &[],
        };
        for len in 1..6 {
            let nal = alloc::vec![0x41u8; len];
            let _ = parse_h264_slice_poc(&nal, &sps, &[H264PpsPoc::default()]);
        }
        assert!(parse_h264_slice_poc(&[0x41, 0x88], &sps, &[H264PpsPoc::default()]).is_none());
    }
}

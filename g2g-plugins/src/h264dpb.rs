// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use crate::annexb::{h264_nal_type, nal_units_any, strip_emulation_prevention, BitReader};
use crate::displayorder::PictureMeta;
use crate::h264parse::{
    h264_poc_context, parse_h264_pps, parse_h264_slice_header, parse_h264_sps, H264Pps,
    H264SliceHeader, H264Sps, H264StreamRefused, H264_NAL_PPS, H264_NAL_SPS,
    MALFORMED_SLICE_HEADER,
};
use g2g_core::meta::CropRectangle;

/// The SPS fields a decode session and its DPB are sized from: a new SPS that
/// differs in any of them needs a new session, any other change does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct H264SessionShape {
    coded_extent: (u32, u32),
    frame_crop_offsets: [u32; 4],
    max_num_ref_frames: u8,
    chroma_format_idc: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
}

impl H264Sps {
    pub(crate) fn session_shape(&self) -> H264SessionShape {
        H264SessionShape {
            coded_extent: self.coded_extent(),
            frame_crop_offsets: [
                self.frame_crop_left_offset,
                self.frame_crop_right_offset,
                self.frame_crop_top_offset,
                self.frame_crop_bottom_offset,
            ],
            max_num_ref_frames: self.max_num_ref_frames,
            chroma_format_idc: self.chroma_format_idc,
            bit_depth_luma_minus8: self.bit_depth_luma_minus8,
            bit_depth_chroma_minus8: self.bit_depth_chroma_minus8,
        }
    }
}

pub(crate) const CHROMA_420_CROP_UNIT: u32 = 2;

// what is left of `picture` once `[left, right, top, bottom]` crop units of `unit` samples are cut off
pub(crate) fn crop_rectangle(
    picture: (u32, u32),
    unit: (u32, u32),
    [left, right, top, bottom]: [u32; 4],
) -> Option<CropRectangle> {
    let axis = |size: u32, unit: u32, start: u32, end: u32| {
        let offset = start.checked_mul(unit)?;
        let kept = size
            .checked_sub(offset)?
            .checked_sub(end.checked_mul(unit)?)?;
        (kept > 0).then_some((offset, kept))
    };
    let (left, width) = axis(picture.0, unit.0, left, right)?;
    let (top, height) = axis(picture.1, unit.1, top, bottom)?;
    Some(CropRectangle {
        left,
        top,
        width,
        height,
    })
}

pub(crate) const H264_NAL_SLICE: u8 = 1;
pub(crate) const H264_NAL_IDR_SLICE: u8 = 5;

/// `MaxFrameNum` (7.4.2.1.1).
pub(crate) fn h264_max_frame_num(sps: &H264Sps) -> u32 {
    1 << (u32::from(sps.log2_max_frame_num_minus4) + 4)
}

/// `FrameNumWrap` of the short-term reference with `frame_num`, seen from the
/// picture with `current_frame_num` (8.2.4.1).
fn h264_frame_num_wrap(frame_num: u32, current_frame_num: u32, max_frame_num: u32) -> i64 {
    if frame_num > current_frame_num {
        i64::from(frame_num) - i64::from(max_frame_num)
    } else {
        i64::from(frame_num)
    }
}

/// `picNumX` of a marking operation on a frame (8.2.5.4.1), whose `CurrPicNum`
/// is its `frame_num`.
fn h264_pic_num(frame_num: u32, difference_of_pic_nums_minus1: u32) -> i64 {
    i64::from(frame_num) - i64::from(difference_of_pic_nums_minus1) - 1
}

/// `TopFieldOrderCnt` and `BottomFieldOrderCnt` of a frame (8.2.1), from its
/// `PicOrderCnt` (the smaller of the two) and the syntax that codes their
/// difference.
pub(crate) fn h264_field_order_counts(
    pic_order_cnt: i32,
    header: &H264SliceHeader,
    sps: &H264Sps,
) -> [i32; 2] {
    let bottom_minus_top = match sps.pic_order_cnt_type {
        0 => i64::from(header.delta_pic_order_cnt_bottom),
        1 => {
            i64::from(sps.offset_for_top_to_bottom_field) + i64::from(header.delta_pic_order_cnt[1])
        }
        _ => 0,
    };
    let top = i64::from(pic_order_cnt) - bottom_minus_top.min(0);
    let saturate = |count: i64| count.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    [saturate(top), saturate(top + bottom_minus_top)]
}

/// The order counts of one H.264 picture.
#[derive(Debug, Clone, Copy)]
pub(crate) struct H264PictureOrder {
    /// The field order counts the picture decodes with.
    pub(crate) field_order_counts: [i32; 2],
    /// The field order counts later pictures reference it with, which
    /// `memory_management_control_operation` 5 resets.
    pub(crate) stored_field_order_counts: [i32; 2],
    /// The picture carries `memory_management_control_operation` 5.
    pub(crate) memory_reset: bool,
}

/// Run the picture-order-count derivation for the picture `header` opens,
/// advancing `poc`. A picture with `memory_management_control_operation` 5
/// restarts the count after it (8.2.1), so its stored field order counts are
/// its own minus the smaller one.
pub(crate) fn h264_picture_order(
    poc: &mut crate::poc::H264PocState,
    header: &H264SliceHeader,
    sps: &H264Sps,
) -> H264PictureOrder {
    let pic_order_cnt = poc.compute(&h264_poc_context(sps), header);
    let field_order_counts = h264_field_order_counts(pic_order_cnt, header, sps);
    let memory_reset = header
        .ref_pic_marking
        .is_some_and(|marking| marking.ops().contains(&crate::poc::H264Mmco::AllUnused));
    let stored_field_order_counts = if memory_reset {
        *poc = crate::poc::H264PocState::default();
        field_order_counts.map(|count| count.saturating_sub(pic_order_cnt))
    } else {
        field_order_counts
    };
    H264PictureOrder {
        field_order_counts,
        stored_field_order_counts,
        memory_reset,
    }
}

/// The `pic_parameter_set_id` a slice NAL selects (7.3.3), read before the rest
/// of the header since the PPS sizes the fields after it.
pub(crate) fn h264_slice_pic_parameter_set_id(nal: &[u8]) -> Result<u8, H264StreamRefused> {
    let mut br = BitReader::skipping_emulation_prevention(nal.get(1..).unwrap_or_default());
    br.read_ue().ok_or(MALFORMED_SLICE_HEADER)?; // first_mb_in_slice
    br.read_ue().ok_or(MALFORMED_SLICE_HEADER)?; // slice_type
    let id = br.read_ue().ok_or(MALFORMED_SLICE_HEADER)?;
    u8::try_from(id).map_err(|_| MALFORMED_SLICE_HEADER)
}

/// A reference picture held in a DPB slot: the metadata the driver matches
/// against slice-header reference lists (`FrameNum`, POC).
#[derive(Debug, Clone, Copy)]
pub(crate) struct RefPic {
    pub(crate) frame_num: u32,
    /// `TopFieldOrderCnt`, `BottomFieldOrderCnt` (8.2.1).
    pub(crate) field_order_counts: [i32; 2],
    /// `LongTermFrameIdx` once the picture is marked long-term (8.2.5.4).
    pub(crate) long_term_frame_idx: Option<u32>,
    // a frame a frame_num gap inferred (8.2.5.2): it holds a slot but was never decoded
    pub(crate) non_existing: bool,
}

pub(crate) const DPB_FULL: H264StreamRefused =
    H264StreamRefused("H.264 stream holds more reference frames than its DPB has slots");
const MISSING_PARAMETER_SET: H264StreamRefused =
    H264StreamRefused("H.264 slice names an SPS or PPS the stream never sent");
// what a lost or corrupt packet leaves behind, which the next keyframe repairs
pub(crate) const H264_DAMAGE_REFUSALS: [H264StreamRefused; 3] = [
    crate::h264parse::MALFORMED_SLICE_HEADER,
    MISSING_PARAMETER_SET,
    DPB_FULL,
];

/// The H.264 reference picture marking state of a DPB (8.2.5): which slots hold
/// reference frames, including the frames a `frame_num` gap inferred, and the
/// long-term index bound. Holds no GPU state.
#[derive(Debug, Clone)]
pub(crate) struct H264References {
    /// Per-slot reference state: `Some` means the slot holds a reference
    /// picture, `None` means it is free.
    slots: alloc::vec::Vec<Option<RefPic>>,
    /// `PrevRefFrameNum` (7.4.3).
    prev_ref_frame_num: u32,
    /// `MaxLongTermFrameIdx` (8.2.5.4), `None` for "no long-term frame indices".
    max_long_term_frame_idx: Option<u32>,
    pub(crate) max_num_ref_frames: usize,
}

impl H264References {
    pub(crate) fn new(num_slots: usize, max_num_ref_frames: usize) -> Self {
        Self {
            slots: alloc::vec![None; num_slots],
            prev_ref_frame_num: 0,
            max_long_term_frame_idx: None,
            max_num_ref_frames,
        }
    }

    /// Mark every picture unused for reference, as an IDR does.
    pub(crate) fn clear(&mut self) {
        self.slots.fill(None);
        self.prev_ref_frame_num = 0;
        self.max_long_term_frame_idx = None;
    }

    /// Every slot holding a reference picture.
    pub(crate) fn active(&self) -> alloc::vec::Vec<(usize, RefPic)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, reference)| reference.map(|r| (slot, r)))
            .collect()
    }

    pub(crate) fn free_slot(&self) -> Option<usize> {
        self.slots.iter().position(|reference| reference.is_none())
    }

    /// Run the marking for the picture just decoded into `target` (8.2.5.1) and
    /// store it as a reference, after the stream's own marking operations or the
    /// sliding window have made room. A non-reference picture (`nal_ref_idc ==
    /// 0`) leaves its slot free.
    pub(crate) fn mark_current(
        &mut self,
        target: usize,
        header: &H264SliceHeader,
        max_frame_num: u32,
        order: &H264PictureOrder,
    ) -> Result<(), H264StreamRefused> {
        if header.nal_ref_idc == 0 || self.max_num_ref_frames == 0 {
            return Ok(());
        }
        let marking = header.ref_pic_marking.ok_or(MALFORMED_SLICE_HEADER)?;
        let long_term_frame_idx = if header.is_idr {
            self.max_long_term_frame_idx = marking.long_term_reference.then_some(0);
            self.max_long_term_frame_idx
        } else if marking.adaptive {
            let long_term_frame_idx =
                self.apply_ref_pic_marking(header.frame_num, max_frame_num, &marking);
            // gst's h264decoder slides the window too when a broken MMCO leaves the DPB full
            self.slide_window(header.frame_num, max_frame_num);
            long_term_frame_idx
        } else {
            self.slide_window(header.frame_num, max_frame_num);
            None
        };
        let frame_num = if order.memory_reset {
            0
        } else {
            header.frame_num
        };
        self.prev_ref_frame_num = frame_num;
        self.slots[target] = Some(RefPic {
            frame_num,
            field_order_counts: order.stored_field_order_counts,
            long_term_frame_idx,
            non_existing: false,
        });
        Ok(())
    }

    /// Apply a slice's adaptive reference marking (8.2.5.4): the stream names
    /// which pictures stop being references, or become long-term ones, instead
    /// of leaving it to the sliding window. Returns the `LongTermFrameIdx` the
    /// current picture takes (operation 6), if any.
    fn apply_ref_pic_marking(
        &mut self,
        frame_num: u32,
        max_frame_num: u32,
        marking: &crate::poc::H264RefPicMarking,
    ) -> Option<u32> {
        let mut current_long_term_frame_idx = None;
        for op in marking.ops() {
            match *op {
                crate::poc::H264Mmco::ShortTermUnused {
                    difference_of_pic_nums_minus1,
                } => {
                    let pic_num = h264_pic_num(frame_num, difference_of_pic_nums_minus1);
                    self.unmark_short_term(pic_num, frame_num, max_frame_num);
                }
                crate::poc::H264Mmco::LongTermUnused { long_term_pic_num } => {
                    self.unmark_long_term(long_term_pic_num);
                }
                crate::poc::H264Mmco::AssignLongTerm {
                    difference_of_pic_nums_minus1,
                    long_term_frame_idx,
                } => {
                    self.unmark_long_term(long_term_frame_idx);
                    let pic_num = h264_pic_num(frame_num, difference_of_pic_nums_minus1);
                    if let Some(slot) = self.short_term_slot(pic_num, frame_num, max_frame_num) {
                        if let Some(reference) = &mut self.slots[slot] {
                            reference.long_term_frame_idx = Some(long_term_frame_idx);
                        }
                    }
                }
                crate::poc::H264Mmco::MaxLongTermIndex {
                    max_long_term_frame_idx_plus1,
                } => {
                    self.max_long_term_frame_idx = max_long_term_frame_idx_plus1.checked_sub(1);
                    let max = self.max_long_term_frame_idx;
                    for reference in &mut self.slots {
                        let beyond = reference
                            .and_then(|r| r.long_term_frame_idx)
                            .is_some_and(|idx| max.is_none_or(|max| idx > max));
                        if beyond {
                            *reference = None;
                        }
                    }
                }
                crate::poc::H264Mmco::AllUnused => {
                    self.slots.fill(None);
                    self.max_long_term_frame_idx = None;
                }
                crate::poc::H264Mmco::CurrentAsLongTerm {
                    long_term_frame_idx,
                } => {
                    self.unmark_long_term(long_term_frame_idx);
                    current_long_term_frame_idx = Some(long_term_frame_idx);
                }
            }
        }
        current_long_term_frame_idx
    }

    /// The slot of the short-term reference whose `PicNum` (its `FrameNumWrap`,
    /// 8.2.4.1) is `pic_num`.
    fn short_term_slot(&self, pic_num: i64, frame_num: u32, max_frame_num: u32) -> Option<usize> {
        self.slots.iter().position(|reference| {
            reference.is_some_and(|r| {
                r.long_term_frame_idx.is_none()
                    && h264_frame_num_wrap(r.frame_num, frame_num, max_frame_num) == pic_num
            })
        })
    }

    fn unmark_short_term(&mut self, pic_num: i64, frame_num: u32, max_frame_num: u32) {
        if let Some(slot) = self.short_term_slot(pic_num, frame_num, max_frame_num) {
            self.slots[slot] = None;
        }
    }

    /// Unmark the long-term reference whose `LongTermFrameIdx` is `idx`; for a
    /// frame it is also its `LongTermPicNum` (8.2.4.1).
    fn unmark_long_term(&mut self, idx: u32) {
        for reference in &mut self.slots {
            if reference.is_some_and(|r| r.long_term_frame_idx == Some(idx)) {
                *reference = None;
            }
        }
    }

    /// The sliding window (8.2.5.3): once the short-term and long-term
    /// references fill `max_num_ref_frames`, the short-term one with the smallest
    /// `FrameNumWrap` stops being a reference.
    fn slide_window(&mut self, frame_num: u32, max_frame_num: u32) {
        let wrap = |other: u32| h264_frame_num_wrap(other, frame_num, max_frame_num);
        while self.slots.iter().flatten().count() >= self.max_num_ref_frames.max(1) {
            let oldest_slot = self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(slot, r)| {
                    r.filter(|r| r.long_term_frame_idx.is_none())
                        .map(|r| (slot, r))
                })
                .min_by_key(|(_, r)| wrap(r.frame_num));
            let Some((slot, _)) = oldest_slot else {
                return;
            };
            self.slots[slot] = None;
        }
    }

    /// Infer the frames a gap in `frame_num` skipped (8.2.5.2): each one runs the
    /// sliding window and then holds a short-term reference slot, so the real
    /// references they push out are the ones the encoder expected to be gone.
    pub(crate) fn fill_frame_num_gap(
        &mut self,
        header: &H264SliceHeader,
        sps: &H264Sps,
    ) -> Result<(), H264StreamRefused> {
        let max_frame_num = h264_max_frame_num(sps);
        let next = (self.prev_ref_frame_num + 1) % max_frame_num;
        let no_gap = header.is_idr
            || sps.gaps_in_frame_num_value_allowed_flag == 0
            || header.frame_num == self.prev_ref_frame_num
            || header.frame_num == next;
        if no_gap {
            return Ok(());
        }
        let mut unused = next;
        while unused != header.frame_num {
            self.slide_window(unused, max_frame_num);
            let slot = self.free_slot().ok_or(DPB_FULL)?;
            self.slots[slot] = Some(RefPic {
                frame_num: unused,
                field_order_counts: [0, 0],
                long_term_frame_idx: None,
                non_existing: true,
            });
            unused = (unused + 1) % max_frame_num;
        }
        self.prev_ref_frame_num = (header.frame_num + max_frame_num - 1) % max_frame_num;
        Ok(())
    }
}

/// The newest SPS and PPS of each id an H.264 stream has sent.
#[derive(Debug, Clone, Default)]
pub(crate) struct H264ParameterSetTable {
    pub(crate) sps: alloc::collections::BTreeMap<u8, H264Sps>,
    pub(crate) pps: alloc::collections::BTreeMap<u8, H264Pps>,
}

impl H264ParameterSetTable {
    #[cfg(feature = "vulkan-video")]
    pub(crate) fn of(ps: &crate::h264parse::H264ParameterSets) -> Self {
        let mut table = Self::default();
        table
            .sps
            .insert(ps.sps.seq_parameter_set_id, ps.sps.clone());
        table
            .pps
            .insert(ps.pps.pic_parameter_set_id, ps.pps.clone());
        table
    }

    /// The PPS a slice selects and the SPS that PPS names.
    pub(crate) fn active(
        &self,
        pic_parameter_set_id: u8,
    ) -> Result<(&H264Sps, &H264Pps), H264StreamRefused> {
        let pps = self
            .pps
            .get(&pic_parameter_set_id)
            .ok_or(MISSING_PARAMETER_SET)?;
        let sps = self
            .sps
            .get(&pps.seq_parameter_set_id)
            .ok_or(MISSING_PARAMETER_SET)?;
        Ok((sps, pps))
    }
}

pub(crate) fn h264_picture_meta(header: &H264SliceHeader, order: &H264PictureOrder) -> PictureMeta {
    let [top, bottom] = order.stored_field_order_counts;
    PictureMeta {
        is_keyframe: header.is_idr || order.memory_reset,
        is_random_access: header.is_idr,
        frame_num: header.frame_num,
        poc: top.min(bottom),
        pic_output_flag: true,
        discards_prior_pictures: false,
        dpb: None,
    }
}

/// One primary coded picture of a stream: its first slice header, its slice
/// NALs (borrowing the input stream), and the parameter sets it was coded with.
#[derive(Debug)]
pub(crate) struct H264Picture<'s> {
    pub(crate) header: H264SliceHeader,
    pub(crate) slices: alloc::vec::Vec<&'s [u8]>,
    pub(crate) sps: H264Sps,
    pub(crate) pps: H264Pps,
}

pub(crate) type PictureUnits<'s> = alloc::vec::Vec<H264Picture<'s>>;

/// Split an Annex-B / AVCC stream into primary coded pictures. Access units
/// break on a VCL NAL with `first_mb_in_slice == 0`. SPS / PPS NALs update
/// the parameter-set table in stream order, so each picture is parsed against,
/// and carries, the sets in effect at its first slice. Other non-VCL NALs are
/// skipped. Borrows `stream`, not the table.
pub(crate) fn split_h264_pictures<'s>(
    parsed_sets: &mut H264ParameterSetTable,
    stream: &'s [u8],
) -> Result<PictureUnits<'s>, H264StreamRefused> {
    let mut pictures = alloc::vec::Vec::new();
    let mut current: Option<H264Picture<'s>> = None;

    for nal in nal_units_any(stream) {
        let Some(nal_type) = h264_nal_type(nal) else {
            continue;
        };
        match nal_type {
            H264_NAL_SPS => {
                let sps = parse_h264_sps(&strip_emulation_prevention(&nal[1..]))?;
                parsed_sets.sps.insert(sps.seq_parameter_set_id, sps);
                continue;
            }
            H264_NAL_PPS => {
                let pps = parse_h264_pps(&strip_emulation_prevention(&nal[1..]))?;
                parsed_sets.pps.insert(pps.pic_parameter_set_id, pps);
                continue;
            }
            H264_NAL_SLICE | H264_NAL_IDR_SLICE => {}
            _ => continue,
        }
        let (sps, pps) = parsed_sets.active(h264_slice_pic_parameter_set_id(nal)?)?;
        let header = parse_h264_slice_header(nal, sps, pps)?;
        if header.first_mb_in_slice == 0 {
            pictures.extend(current.take());
        }
        current
            .get_or_insert_with(|| H264Picture {
                header,
                slices: alloc::vec::Vec::new(),
                sps: sps.clone(),
                pps: pps.clone(),
            })
            .slices
            .push(nal);
    }
    pictures.extend(current);
    Ok(pictures)
}

//! H.264 access-unit parser that refines source-side `Caps`.
//!
//! M6: scans each `DataFrame`'s bitstream for an SPS NAL unit, parses
//! width/height, and emits a `CapsChanged` packet with `Dim::Fixed` values
//! before forwarding the frame. This is the first element that refines
//! caps mid-stream — `RtspSrc` advertises `Dim::Any` at negotiation time
//! because the SPS only lands once bytes flow.
//!
//! Bitstream framing: both Annex-B (00 00 01 / 00 00 00 01 start codes) and
//! AVCC (4-byte length-prefixed NALs, what retina emits by default) are
//! accepted, detected per access unit via `annexb::nal_units_any`. The element
//! refines caps only; it does not rewrite the bitstream between framings.
//!
//! Framerate is recovered from the SPS VUI `timing_info` when present
//! (`time_scale / (2 * num_units_in_tick)`, emitted as a Q16 `Rate::Fixed`);
//! caps carry `Rate::Any` when the VUI omits it.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, VideoCodec,
};

use crate::annexb::{next_start_code, strip_emulation_prevention, BitReader};

/// Upper bound on bytes buffered while waiting for an access-unit boundary in
/// re-framing mode. A real H.264 stream emits start codes frequently, so this is
/// only a guard against an unbounded accumulator on pathological / non-conforming
/// input: past it, the pending bytes are flushed as one access unit rather than
/// grown without limit. A single intra access unit at 4K stays well under it.
const MAX_REFRAME_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Default)]
pub struct H264Parse {
    configured: bool,
    last_emitted_caps: Option<Caps>,
    sps_emitted: u64,
    /// Re-framing mode (M421): when set, the element re-chunks its input into
    /// access-unit-aligned Annex-B buffers (one coded picture per `DataFrame`)
    /// rather than passing buffers through unchanged. A decoder fed un-aligned
    /// units (e.g. one MPEG-TS PES that is not one access unit) mis-parses slice
    /// boundaries; auto-plugged decode chains insert the parser in this mode so
    /// the decoder always sees one access unit per packet, matching what
    /// GStreamer's `decodebin` does with `h264parse`. Off in the default
    /// (caps-refinement-only) construction, so existing explicit / RTSP uses keep
    /// their pass-through framing.
    reframe: bool,
    /// Re-framing accumulator: Annex-B bytes received but not yet emitted as a
    /// complete access unit (the trailing, possibly-incomplete AU is held until
    /// the next AU's start code arrives). Empty outside re-framing mode.
    accum: Vec<u8>,
    /// Timing to stamp the access unit currently at the head of `accum` (captured
    /// when that AU's first byte arrived). Re-framing only.
    au_timing: FrameTiming,
    /// Monotonic sequence number for emitted re-framed access units.
    seq: u64,
    /// Input framing, latched from the first frame (`Some(true)` = Annex-B start
    /// codes, `Some(false)` = AVCC length prefixes). The framing is fixed for a
    /// stream, so it must be decided once: a per-frame guess misclassifies an
    /// Annex-B continuation buffer (one MPEG-TS PES carrying the tail of an access
    /// unit, which does not start with a start code) as AVCC and mangles it.
    /// Re-framing only.
    input_is_annexb: Option<bool>,
}

impl H264Parse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct the parser in access-unit re-framing mode (M421): it accumulates
    /// the input bitstream and emits one access-unit-aligned Annex-B `DataFrame`
    /// per coded picture. Auto-plugged decode chains use this so the decoder is
    /// fed one access unit per packet (see [`reframe`](Self::reframe)).
    pub fn reframing() -> Self {
        Self { reframe: true, ..Self::default() }
    }

    /// Count of `CapsChanged` packets this element has pushed downstream.
    /// Useful for tests asserting that re-emission is suppressed when the
    /// SPS dimensions are unchanged.
    pub fn caps_changes_emitted(&self) -> u64 {
        self.sps_emitted
    }
}

impl AsyncElement for H264Parse {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // H264Parse consumes H.264 at any geometry; intersecting against
        // that narrows the proposal and rejects non-H.264 inputs.
        let supported = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        upstream_caps.intersect(&supported)
    }

    /// M16 step 5g: pass-through identity over H.264 of any geometry.
    /// `Identity(CapsSet::one(...))` is the native shape for transforms
    /// that accept and emit the same caps. With a fully-native chain
    /// the solver couples input and output links and rejects non-H.264
    /// upstream at negotiation time instead of via the dynamic
    /// `intercept_caps` callback.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(CapsSet::one(Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))
    }

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            } => {
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "H.264 parser",
            "Codec/Parser/Video",
            "Parses an H.264 Annex-B stream and refines caps from SPS/PPS",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    if self.reframe {
                        return self.reframe_frame(frame, out).await;
                    }
                    if let g2g_core::MemoryDomain::System(slice) = &frame.domain {
                        // Surface the keyframe flag for trick-mode / keyframe seek
                        // (the parser is the producer that can detect it).
                        let is_keyframe = crate::h264util::h264_au_is_keyframe(slice.as_slice());
                        if let Some(info) = extract_sps_info(slice.as_slice()) {
                            let new_caps = Caps::CompressedVideo {
                                codec: VideoCodec::H264,
                                width: Dim::Fixed(info.width),
                                height: Dim::Fixed(info.height),
                                framerate: info.framerate.map_or(Rate::Any, Rate::Fixed),
                            };
                            if self.last_emitted_caps.as_ref() != Some(&new_caps) {
                                out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                                    .await?;
                                self.last_emitted_caps = Some(new_caps);
                                self.sps_emitted += 1;
                            }
                        }
                        frame.timing.keyframe = is_keyframe;
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    // A seek discontinuity: drop the partial AU rather than splice
                    // pre-seek bytes onto the post-seek stream. Reset SPS tracking
                    // so caps re-emit after the seek.
                    self.accum.clear();
                    self.last_emitted_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {
                    // Flush the final buffered access unit at end of stream, else
                    // the last coded picture would never reach the decoder.
                    if self.reframe && !self.accum.is_empty() {
                        let au = core::mem::take(&mut self.accum);
                        let timing = self.au_timing;
                        self.emit_au(au, timing, out).await?;
                    }
                }
            }
            Ok(())
        })
    }
}

impl H264Parse {
    /// Re-framing path for one input `DataFrame` (M421): normalize to Annex-B,
    /// accumulate, and emit every access unit whose end is now known (its
    /// successor's start code has arrived). The trailing, possibly-incomplete AU
    /// stays buffered until the next call or `Eos`. Non-`System` domains pass
    /// through unchanged (the byte re-framer only applies to host memory).
    async fn reframe_frame(
        &mut self,
        frame: Frame,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let MemoryDomain::System(slice) = &frame.domain else {
            out.push(PipelinePacket::DataFrame(frame)).await?;
            return Ok(());
        };
        // Normalize to Annex-B: a length-prefixed (AVCC) stream is converted, an
        // Annex-B one is appended as-is. The framing is latched from the first
        // frame, since a mid-AU Annex-B continuation buffer (no leading start
        // code) would otherwise be misread as AVCC.
        let bytes = slice.as_slice();
        let is_annexb = *self.input_is_annexb.get_or_insert_with(|| crate::annexb::is_annex_b(bytes));
        if self.accum.is_empty() {
            self.au_timing = frame.timing;
        }
        if is_annexb {
            self.accum.extend_from_slice(bytes);
        } else {
            self.accum.extend_from_slice(&crate::annexb::avcc_to_annexb(bytes));
        }

        // Guard against unbounded growth on non-conforming input: flush what we
        // have as one AU rather than buffering forever.
        if self.accum.len() > MAX_REFRAME_BYTES {
            let au = core::mem::take(&mut self.accum);
            let timing = self.au_timing;
            self.emit_au(au, timing, out).await?;
            return Ok(());
        }

        // Access-unit start offsets in the accumulator. Emit each complete AU
        // (everything before the last start), then retain the trailing AU.
        let starts = h264_au_starts(&self.accum);
        if starts.len() < 2 {
            return Ok(()); // at most one (still-open) AU buffered so far
        }
        let frame_timing = frame.timing;
        let tail = starts[starts.len() - 1];
        // Split off the still-open tail, leaving the complete AUs in `done`.
        let done = self.accum[..tail].to_vec();
        self.accum.drain(..tail);
        for w in starts.windows(2) {
            let (lo, hi) = (w[0], w[1]);
            // The head AU carries the timing captured when it began; AUs that both
            // begin and end inside this buffer take this buffer's timing.
            let timing = if lo == 0 { self.au_timing } else { frame_timing };
            self.emit_au(done[lo..hi].to_vec(), timing, out).await?;
        }
        // The retained tail began within this buffer (its predecessor ended here).
        self.au_timing = frame_timing;
        Ok(())
    }

    /// Emit one access-unit-aligned Annex-B buffer as a `DataFrame`: refine caps
    /// from any SPS it carries (suppressing an unchanged re-emit) and stamp the
    /// keyframe flag, mirroring the pass-through path's per-frame work.
    async fn emit_au(
        &mut self,
        au: Vec<u8>,
        mut timing: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        if au.is_empty() {
            return Ok(());
        }
        if let Some(info) = extract_sps_info(&au) {
            let new_caps = Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Fixed(info.width),
                height: Dim::Fixed(info.height),
                framerate: info.framerate.map_or(Rate::Any, Rate::Fixed),
            };
            if self.last_emitted_caps.as_ref() != Some(&new_caps) {
                out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                self.last_emitted_caps = Some(new_caps);
                self.sps_emitted += 1;
            }
        }
        timing.keyframe = crate::h264util::h264_au_is_keyframe(&au);
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            timing,
            self.seq,
        );
        self.seq += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

/// Start-code offsets in an Annex-B buffer at which a new H.264 access unit
/// begins, per the ISO/IEC 14496-10 access-unit boundary rules. The first NAL
/// always opens the first AU. Once a VCL NAL (types 1..=5) has been seen in the
/// current AU, the next AU begins at: an access-unit delimiter (9), the parameter
/// sets / SEI / prefix NALs that lead a picture (6, 7, 8, 14, 15, 18), or the
/// first VCL NAL of a new coded picture (`first_mb_in_slice == 0`, i.e. the first
/// RBSP bit is 1). Slices 2..N of one picture carry `first_mb_in_slice > 0` and
/// stay in the same AU. This is the framing GStreamer's `h264parse` applies; a
/// decoder needs it because it parses one packet as one coded picture.
fn h264_au_starts(data: &[u8]) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut seen_vcl = false;
    let mut i = 0;
    while let Some((sc, begin)) = next_start_code(data, i) {
        let nal_type = data.get(begin).map(|b| b & 0x1F).unwrap_or(0);
        let is_vcl = (1..=5).contains(&nal_type);
        let starts_au = if !seen_vcl {
            // Leading NALs of the first AU: only the very first opens an AU; the
            // rest (more parameter sets, the first VCL) join it.
            starts.is_empty()
        } else if is_vcl {
            // A new picture's first slice has first_mb_in_slice == 0, encoded as a
            // leading 1 bit in the slice RBSP (ue(v) of 0).
            data.get(begin + 1).map(|b| b & 0x80 != 0).unwrap_or(false)
        } else {
            // A non-VCL that can only lead the next access unit.
            matches!(nal_type, 6 | 7 | 8 | 9 | 14 | 15 | 18)
        };
        if starts_au {
            starts.push(sc);
            seen_vcl = false;
        }
        if is_vcl {
            seen_vcl = true;
        }
        i = begin;
    }
    starts
}

impl PadTemplates for H264Parse {
    /// Consumes and produces H.264 at any geometry (the parser refines
    /// geometry mid-stream from the SPS but never changes media type).
    fn pad_templates() -> Vec<PadTemplate> {
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(h264.clone())),
            PadTemplate::source(CapsSet::one(h264)),
        ])
    }
}

/// Geometry and optional framerate recovered from an SPS.
struct SpsInfo {
    width: u32,
    height: u32,
    /// Framerate as Q16 fixed-point from the VUI `timing_info`, `None` when
    /// the SPS carries no timing.
    framerate: Option<u32>,
}

/// Walk the NALs of `au` (Annex-B or AVCC, auto-detected), returning the info
/// from the first SPS NAL (nal_unit_type == 7) we can fully parse.
fn extract_sps_info(au: &[u8]) -> Option<SpsInfo> {
    for nal in crate::annexb::nal_units_any(au) {
        if nal.is_empty() {
            continue;
        }
        let nal_unit_type = nal[0] & 0x1F;
        if nal_unit_type != 7 {
            continue;
        }
        let rbsp = strip_emulation_prevention(&nal[1..]);
        if let Some(info) = parse_sps(&rbsp) {
            return Some(info);
        }
    }
    None
}

/// Parse the SPS RBSP (post NAL-header byte) for the coded picture dimensions
/// and, when the VUI carries `timing_info`, the framerate. Returns `None` on a
/// parse failure up to the dimensions; a failure past them leaves only the
/// framerate unknown.
fn parse_sps(rbsp: &[u8]) -> Option<SpsInfo> {
    if rbsp.len() < 3 {
        return None;
    }
    let profile_idc = rbsp[0];
    // rbsp[1] = constraint_set flags + reserved zero bits (skipped)
    // rbsp[2] = level_idc (skipped)
    let mut br = BitReader::new(&rbsp[3..]);

    let _sps_id = br.read_ue()?;

    // 4:2:0 unless a high profile signals a different chroma format.
    let mut chroma_format_idc = 1u32;
    let mut separate_colour_plane_flag = 0u32;
    // Profiles that include chroma/scaling/etc. extra header fields.
    if matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        chroma_format_idc = br.read_ue()?;
        if chroma_format_idc == 3 {
            separate_colour_plane_flag = br.read_bit()?;
        }
        let _bit_depth_luma_minus8 = br.read_ue()?;
        let _bit_depth_chroma_minus8 = br.read_ue()?;
        let _qpprime_y_zero_transform_bypass_flag = br.read_bit()?;
        let seq_scaling_matrix_present_flag = br.read_bit()?;
        if seq_scaling_matrix_present_flag == 1 {
            // We don't decode the (optional) scaling lists. M6 test fixtures
            // never set this flag; live streams that do will leave dims
            // unknown until M7 lands a full SPS parser.
            return None;
        }
    }

    let _log2_max_frame_num_minus4 = br.read_ue()?;
    let pic_order_cnt_type = br.read_ue()?;
    if pic_order_cnt_type == 0 {
        let _log2_max_pic_order_cnt_lsb_minus4 = br.read_ue()?;
    } else if pic_order_cnt_type == 1 {
        let _delta_pic_order_always_zero_flag = br.read_bit()?;
        let _offset_for_non_ref_pic = br.read_se()?;
        let _offset_for_top_to_bottom_field = br.read_se()?;
        let num_ref_frames_in_pic_order_cnt_cycle = br.read_ue()?;
        for _ in 0..num_ref_frames_in_pic_order_cnt_cycle {
            let _offset = br.read_se()?;
        }
    }
    let _max_num_ref_frames = br.read_ue()?;
    let _gaps_in_frame_num_value_allowed_flag = br.read_bit()?;

    let pic_width_in_mbs_minus1 = br.read_ue()?;
    let pic_height_in_map_units_minus1 = br.read_ue()?;
    let frame_mbs_only_flag = br.read_bit()?;
    if frame_mbs_only_flag == 0 {
        let _mb_adaptive_frame_field_flag = br.read_bit()?;
    }
    let _direct_8x8_inference_flag = br.read_bit()?;

    let frame_cropping_flag = br.read_bit()?;
    let (crop_left, crop_right, crop_top, crop_bottom) = if frame_cropping_flag == 1 {
        let l = br.read_ue()?;
        let r = br.read_ue()?;
        let t = br.read_ue()?;
        let b = br.read_ue()?;
        (l, r, t, b)
    } else {
        (0, 0, 0, 0)
    };

    // Crop units in luma samples (H.264 7.4.2.1.1). ChromaArrayType 0
    // (monochrome, or 4:4:4 with separate colour planes) crops 1 x (2-fmof);
    // otherwise SubWidthC x SubHeightC*(2-fmof).
    let chroma_array_type = if separate_colour_plane_flag == 1 { 0 } else { chroma_format_idc };
    let (sub_width_c, sub_height_c) = match chroma_array_type {
        1 => (2u32, 2u32), // 4:2:0
        2 => (2, 1),       // 4:2:2
        _ => (1, 1),       // 4:4:4 / monochrome
    };
    // Crop and dimension fields come from untrusted exp-Golomb, so fold with
    // saturating arithmetic (the additions and the *16 would otherwise overflow,
    // panicking in debug and wrapping to bogus caps in release).
    let crop_x = crop_left.saturating_add(crop_right).saturating_mul(sub_width_c);
    let crop_y = crop_top
        .saturating_add(crop_bottom)
        .saturating_mul(sub_height_c.saturating_mul(2u32.saturating_sub(frame_mbs_only_flag)));

    let width = pic_width_in_mbs_minus1
        .saturating_add(1)
        .saturating_mul(16)
        .saturating_sub(crop_x);
    let height = (2 - frame_mbs_only_flag)
        .saturating_mul(pic_height_in_map_units_minus1.saturating_add(1))
        .saturating_mul(16)
        .saturating_sub(crop_y);

    // vui_parameters_present_flag follows frame cropping. Read it without `?`
    // so a stream truncated right here still yields the dimensions.
    let framerate = match br.read_bit() {
        Some(1) => parse_vui_framerate(&mut br),
        _ => None,
    };

    Some(SpsInfo {
        width,
        height,
        framerate,
    })
}

/// Walk the VUI parameters up to `timing_info`, returning the framerate as Q16
/// fixed-point (`time_scale / (2 * num_units_in_tick)`). `None` if the VUI
/// omits timing or ends early; the caller already has the dimensions.
fn parse_vui_framerate(br: &mut BitReader) -> Option<u32> {
    // aspect_ratio_info_present_flag
    if br.read_bit()? == 1 {
        let aspect_ratio_idc = br.read_bits(8)?;
        // 255 = Extended_SAR: explicit sar_width / sar_height follow.
        if aspect_ratio_idc == 255 {
            br.read_bits(16)?; // sar_width
            br.read_bits(16)?; // sar_height
        }
    }
    // overscan_info_present_flag
    if br.read_bit()? == 1 {
        br.read_bit()?; // overscan_appropriate_flag
    }
    // video_signal_type_present_flag
    if br.read_bit()? == 1 {
        br.read_bits(3)?; // video_format
        br.read_bit()?; // video_full_range_flag
        if br.read_bit()? == 1 {
            // colour_description_present_flag
            br.read_bits(8)?; // colour_primaries
            br.read_bits(8)?; // transfer_characteristics
            br.read_bits(8)?; // matrix_coefficients
        }
    }
    // chroma_loc_info_present_flag
    if br.read_bit()? == 1 {
        br.read_ue()?; // chroma_sample_loc_type_top_field
        br.read_ue()?; // chroma_sample_loc_type_bottom_field
    }
    // timing_info_present_flag
    if br.read_bit()? == 1 {
        let num_units_in_tick = br.read_bits(32)?;
        let time_scale = br.read_bits(32)?;
        let _fixed_frame_rate_flag = br.read_bit()?;
        if num_units_in_tick == 0 {
            return None;
        }
        // fps = time_scale / (2 * num_units_in_tick); carry to Q16 in u64.
        let q16 = ((time_scale as u64) << 16) / (2 * num_units_in_tick as u64);
        return u32::try_from(q16).ok();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build a minimal baseline-profile SPS RBSP for `width` x `height`
    /// (both multiples of 16), then frame it in Annex-B. Returns the
    /// full byte stream including a 4-byte start code and NAL header.
    fn build_test_annexb_sps(width: u32, height: u32) -> Vec<u8> {
        assert!(width % 16 == 0 && height % 16 == 0);
        let mut w = BitWriter::default();
        // Post NAL-header SPS fields:
        // seq_parameter_set_id = 0
        w.write_ue(0);
        // log2_max_frame_num_minus4 = 0
        w.write_ue(0);
        // pic_order_cnt_type = 0
        w.write_ue(0);
        // log2_max_pic_order_cnt_lsb_minus4 = 0
        w.write_ue(0);
        // max_num_ref_frames = 1
        w.write_ue(1);
        // gaps_in_frame_num_value_allowed_flag = 0
        w.write_bit(0);
        // pic_width_in_mbs_minus1
        w.write_ue(width / 16 - 1);
        // pic_height_in_map_units_minus1
        w.write_ue(height / 16 - 1);
        // frame_mbs_only_flag = 1
        w.write_bit(1);
        // direct_8x8_inference_flag = 0
        w.write_bit(0);
        // frame_cropping_flag = 0
        w.write_bit(0);
        // vui_parameters_present_flag = 0
        w.write_bit(0);
        // rbsp_trailing_bits: 1 then zero-pad
        w.write_bit(1);
        w.align_to_byte();

        let rbsp = w.into_bytes();

        // Annex-B framing: 00 00 00 01 | nal_header | profile/level bytes | rbsp
        // nal_header for SPS: forbidden_zero_bit=0, nal_ref_idc=3, nal_unit_type=7
        // → (3 << 5) | 7 = 0x67
        // Then the SPS's byte-aligned prefix: profile_idc=66 (baseline),
        // constraint flags + reserved = 0, level_idc=30 — chosen so the
        // parser takes the simple (non-high-profile) branch.
        let mut out = vec![0u8, 0, 0, 1, 0x67, 66, 0, 30];
        out.extend_from_slice(&rbsp);
        out
    }

    #[test]
    fn parse_sps_saturates_adversarial_dimensions() {
        // Untrusted exp-Golomb dimension/crop fields must saturate, not
        // overflow-panic (debug) or wrap to bogus caps (release).
        let mut w = BitWriter::default();
        w.write_ue(0); // seq_parameter_set_id
        w.write_ue(0); // log2_max_frame_num_minus4
        w.write_ue(0); // pic_order_cnt_type
        w.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.write_ue(1); // max_num_ref_frames
        w.write_bit(0); // gaps_in_frame_num_value_allowed_flag
        w.write_ue(300_000_000); // pic_width_in_mbs_minus1 (overflows *16)
        w.write_ue(300_000_000); // pic_height_in_map_units_minus1
        w.write_bit(1); // frame_mbs_only_flag
        w.write_bit(0); // direct_8x8_inference_flag
        w.write_bit(1); // frame_cropping_flag
        w.write_ue(3_000_000_000); // crop_left (sum overflows u32)
        w.write_ue(3_000_000_000); // crop_right
        w.write_ue(3_000_000_000); // crop_top
        w.write_ue(3_000_000_000); // crop_bottom
        w.write_bit(0); // vui_parameters_present_flag
        w.write_bit(1); // rbsp_trailing_bits
        w.align_to_byte();
        let mut au = vec![0u8, 0, 0, 1, 0x67, 66, 0, 30];
        au.extend_from_slice(&w.into_bytes());
        // Must not panic; the dimension *16 saturates to u32::MAX and the even
        // larger crop then saturating-subtracts it back to 0 (vs an overflow
        // panic in debug or a wrapped bogus value in release).
        let info = extract_sps_info(&au).expect("parses without overflow");
        assert_eq!((info.width, info.height), (0, 0));
    }

    /// Build an Annex-B SPS for `width` x `height` carrying a VUI `timing_info`
    /// block. Emulation-prevention bytes are inserted (as a real encoder would
    /// for the 32-bit fields' zero runs) so the parser's de-emulation
    /// round-trips them exactly.
    fn build_annexb_sps_with_timing(
        width: u32,
        height: u32,
        num_units_in_tick: u32,
        time_scale: u32,
    ) -> Vec<u8> {
        assert!(width % 16 == 0 && height % 16 == 0);
        let mut w = BitWriter::default();
        w.write_ue(0); // seq_parameter_set_id
        w.write_ue(0); // log2_max_frame_num_minus4
        w.write_ue(0); // pic_order_cnt_type
        w.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.write_ue(1); // max_num_ref_frames
        w.write_bit(0); // gaps_in_frame_num_value_allowed_flag
        w.write_ue(width / 16 - 1); // pic_width_in_mbs_minus1
        w.write_ue(height / 16 - 1); // pic_height_in_map_units_minus1
        w.write_bit(1); // frame_mbs_only_flag
        w.write_bit(0); // direct_8x8_inference_flag
        w.write_bit(0); // frame_cropping_flag
        w.write_bit(1); // vui_parameters_present_flag
        w.write_bit(0); // aspect_ratio_info_present_flag
        w.write_bit(0); // overscan_info_present_flag
        w.write_bit(0); // video_signal_type_present_flag
        w.write_bit(0); // chroma_loc_info_present_flag
        w.write_bit(1); // timing_info_present_flag
        w.write_bits(num_units_in_tick, 32);
        w.write_bits(time_scale, 32);
        w.write_bit(0); // fixed_frame_rate_flag
        w.write_bit(1); // rbsp_trailing_bits
        w.align_to_byte();
        let rbsp = w.into_bytes();

        // NAL payload after the 0x67 header: profile/constraint/level + RBSP,
        // emulation-prevented like an encoder's output.
        let mut payload = vec![66u8, 0, 30];
        payload.extend_from_slice(&rbsp);
        let payload = crate::annexb::add_emulation_prevention(&payload);

        let mut out = vec![0u8, 0, 0, 1, 0x67];
        out.extend_from_slice(&payload);
        out
    }

    #[derive(Default)]
    struct BitWriter {
        buf: Vec<u8>,
        bit_pos: usize,
    }

    impl BitWriter {
        fn write_bit(&mut self, b: u32) {
            let byte_idx = self.bit_pos / 8;
            if byte_idx >= self.buf.len() {
                self.buf.push(0);
            }
            let bit_off = 7 - (self.bit_pos % 8);
            self.buf[byte_idx] |= ((b & 1) as u8) << bit_off;
            self.bit_pos += 1;
        }

        fn write_bits(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                self.write_bit((value >> i) & 1);
            }
        }

        fn write_ue(&mut self, v: u32) {
            let v1 = v + 1;
            let n = 31 - v1.leading_zeros();
            for _ in 0..n {
                self.write_bit(0);
            }
            self.write_bits(v1, n + 1);
        }

        fn align_to_byte(&mut self) {
            while self.bit_pos % 8 != 0 {
                self.write_bit(0);
            }
        }

        fn into_bytes(self) -> Vec<u8> {
            self.buf
        }
    }

    #[test]
    fn round_trips_a_1280x720_sps() {
        let stream = build_test_annexb_sps(1280, 720);
        let info = extract_sps_info(&stream).expect("SPS must parse");
        assert_eq!((info.width, info.height), (1280, 720));
        assert_eq!(info.framerate, None, "no VUI timing in the baseline fixture");
    }

    #[test]
    fn round_trips_a_1920x1080_sps() {
        let stream = build_test_annexb_sps(1920, 1088);
        // height 1088 because 1080 is not a multiple of 16; the test
        // builder asserts on alignment. Real 1080p streams use cropping.
        let info = extract_sps_info(&stream).expect("SPS must parse");
        assert_eq!((info.width, info.height), (1920, 1088));
    }

    #[test]
    fn parses_an_avcc_framed_sps() {
        // Re-frame the same SPS NAL as AVCC (4-byte length prefix instead of
        // the Annex-B start code) and confirm the dimensions still resolve.
        let annexb = build_test_annexb_sps(1280, 720);
        let nal = &annexb[4..]; // drop the 00 00 00 01 start code
        let mut avcc = (nal.len() as u32).to_be_bytes().to_vec();
        avcc.extend_from_slice(nal);
        let info = extract_sps_info(&avcc).expect("AVCC SPS must parse");
        assert_eq!((info.width, info.height), (1280, 720));
    }

    #[test]
    fn recovers_framerate_from_vui_timing() {
        // num_units_in_tick = 1, time_scale = 60 -> 30 fps.
        let stream = build_annexb_sps_with_timing(1280, 720, 1, 60);
        let info = extract_sps_info(&stream).expect("SPS with VUI must parse");
        assert_eq!((info.width, info.height), (1280, 720));
        assert_eq!(info.framerate, Some(30 << 16), "30 fps in Q16");
    }

    #[test]
    fn ignores_non_sps_nals() {
        // A stream containing only a slice NAL (type 5 = IDR) returns None.
        let stream = [0u8, 0, 0, 1, 0x65, 0xAA, 0xBB, 0xCC];
        assert!(extract_sps_info(&stream).is_none());
    }

    #[test]
    fn returns_none_on_empty_input() {
        assert!(extract_sps_info(&[]).is_none());
    }

    #[test]
    fn strips_emulation_prevention_bytes() {
        // Input "00 00 03 01" should decode to "00 00 01".
        let ebsp = [0u8, 0, 3, 1, 2, 0, 0, 3, 0xFF];
        let rbsp = strip_emulation_prevention(&ebsp);
        assert_eq!(rbsp, [0u8, 0, 1, 2, 0, 0, 0xFF]);
    }

    #[test]
    // the binary grouping aligns to the Exp-Golomb code words, not byte nibbles.
    #[allow(clippy::unusual_byte_groupings)]
    fn bit_reader_decodes_known_ue_codes() {
        // Bits: 1 010 011 00100 → ue values 0, 1, 2, 3
        let bytes = [0b1_010_011_0, 0b0100_0000];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_ue(), Some(0));
        assert_eq!(r.read_ue(), Some(1));
        assert_eq!(r.read_ue(), Some(2));
        assert_eq!(r.read_ue(), Some(3));
    }

    #[test]
    fn parse_vui_framerate_reads_timing_info() {
        // num_units_in_tick = 1001, time_scale = 60000 -> 29.97 fps, exercising
        // the Q16 conversion on a non-integer rate.
        let mut w = BitWriter::default();
        w.write_bit(0); // aspect_ratio_info_present_flag
        w.write_bit(0); // overscan_info_present_flag
        w.write_bit(0); // video_signal_type_present_flag
        w.write_bit(0); // chroma_loc_info_present_flag
        w.write_bit(1); // timing_info_present_flag
        w.write_bits(1001, 32);
        w.write_bits(60000, 32);
        w.write_bit(0); // fixed_frame_rate_flag
        w.align_to_byte();
        let bytes = w.into_bytes();
        let mut br = BitReader::new(&bytes);
        let fps = parse_vui_framerate(&mut br).expect("timing info present");
        let expected = ((60000u64 << 16) / (2 * 1001)) as u32;
        assert_eq!(fps, expected);
        assert_eq!(fps >> 16, 29, "~29.97 fps");
    }

    #[test]
    fn parse_vui_framerate_absent_timing_is_none() {
        let mut w = BitWriter::default();
        w.write_bit(0); // aspect_ratio_info_present_flag
        w.write_bit(0); // overscan_info_present_flag
        w.write_bit(0); // video_signal_type_present_flag
        w.write_bit(0); // chroma_loc_info_present_flag
        w.write_bit(0); // timing_info_present_flag = 0
        w.align_to_byte();
        let bytes = w.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert_eq!(parse_vui_framerate(&mut br), None);
    }

    // -- Element-level tests (drive H264Parse::process directly) -----------

    use core::pin::Pin;
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{FrameTiming, MemoryDomain, PushOutcome};

    #[derive(Default)]
    struct RecordingSink {
        packets: Vec<PipelinePacket>,
    }

    impl OutputSink for RecordingSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                self.packets.push(packet);
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn frame_with_bytes(seq: u64, bytes: Vec<u8>) -> Frame {
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: seq,
            meta: Default::default(),
        }
    }

    fn h264_parse_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let stream = build_test_annexb_sps(1280, 720);
        let frame = frame_with_bytes(0, stream);
        parse
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        assert_eq!(sink.packets.len(), 2, "expected CapsChanged then DataFrame");
        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::CompressedVideo { width, height, .. }) => {
                assert_eq!(*width, Dim::Fixed(1280));
                assert_eq!(*height, Dim::Fixed(720));
            }
            other => panic!("expected CapsChanged first, got {other:?}"),
        }
        assert!(matches!(sink.packets[1], PipelinePacket::DataFrame(_)));
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn emits_caps_from_an_avcc_access_unit() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = RecordingSink::default();

        // Re-frame the SPS NAL as AVCC (4-byte length prefix).
        let annexb = build_test_annexb_sps(1280, 720);
        let nal = &annexb[4..];
        let mut avcc = (nal.len() as u32).to_be_bytes().to_vec();
        avcc.extend_from_slice(nal);

        parse
            .process(PipelinePacket::DataFrame(frame_with_bytes(0, avcc)), &mut sink)
            .await
            .unwrap();

        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::CompressedVideo { width, height, .. }) => {
                assert_eq!(*width, Dim::Fixed(1280));
                assert_eq!(*height, Dim::Fixed(720));
            }
            other => panic!("expected CapsChanged from the AVCC AU, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn does_not_re_emit_caps_when_unchanged() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = RecordingSink::default();

        for seq in 0..3 {
            let stream = build_test_annexb_sps(1280, 720);
            let frame = frame_with_bytes(seq, stream);
            parse
                .process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }

        let caps_count = sink
            .packets
            .iter()
            .filter(|p| matches!(p, PipelinePacket::CapsChanged(_)))
            .count();
        assert_eq!(caps_count, 1, "CapsChanged must fire once for identical SPS");
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn re_emits_caps_on_resolution_change() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let frame_720 = frame_with_bytes(0, build_test_annexb_sps(1280, 720));
        parse
            .process(PipelinePacket::DataFrame(frame_720), &mut sink)
            .await
            .unwrap();

        let frame_1080 = frame_with_bytes(1, build_test_annexb_sps(1920, 1088));
        parse
            .process(PipelinePacket::DataFrame(frame_1080), &mut sink)
            .await
            .unwrap();

        let widths: Vec<Dim> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(Caps::CompressedVideo { width, .. }) => Some(width.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(widths, vec![Dim::Fixed(1280), Dim::Fixed(1920)]);
        assert_eq!(parse.caps_changes_emitted(), 2);
    }

    #[tokio::test]
    async fn rejects_non_h264_caps_in_intercept() {
        let parse = H264Parse::new();
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(parse.intercept_caps(&vp9), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_constraint_is_identity_h264_any() {
        // M16 step 5g: native shape is `Identity` over H.264 with any
        // geometry. With a fully-native chain the solver enforces the
        // input/output coupling and the format requirement during
        // arc-consistency, not via the dynamic `intercept_caps`.
        let parse = H264Parse::new();
        let c = parse.caps_constraint_as_transform();
        match c {
            CapsConstraint::Identity(set) => {
                assert_eq!(
                    set.alternatives(),
                    &[Caps::CompressedVideo {
                        codec: VideoCodec::H264,
                        width: Dim::Any,
                        height: Dim::Any,
                        framerate: Rate::Any,
                    }]
                );
            }
            _ => panic!("expected Identity"),
        }
    }

    // ---- M421 access-unit re-framing ----

    /// An Annex-B VCL slice NAL (type 1). `first` picks `first_mb_in_slice == 0`
    /// (a new picture's first slice: leading RBSP bit 1) vs a continuation slice
    /// (leading bit 0). `tag` makes the payload distinguishable; the payload bytes
    /// avoid `00 00` so they never look like a start code.
    fn annexb_vcl(first: bool, tag: u8) -> Vec<u8> {
        let mut v = vec![0, 0, 0, 1, 0x41]; // start code + NAL header (type 1, ref)
        v.push(if first { 0x80 } else { 0x40 }); // first RBSP byte: MSB = first_mb==0
        v.extend_from_slice(&[0xAA, 0xBB, tag, 0x11]);
        v
    }

    /// Pull the `DataFrame` byte payloads out of a sink, in order.
    fn data_payloads(sink: &RecordingSink) -> Vec<Vec<u8>> {
        sink.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => match &f.domain {
                    MemoryDomain::System(s) => Some(s.as_slice().to_vec()),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    #[test]
    fn au_starts_groups_slices_into_one_picture() {
        // Picture A: two slices (first_mb==0 then a continuation). Picture B: one
        // slice. Two access units, not three.
        let mut stream = annexb_vcl(true, 1);
        stream.extend_from_slice(&annexb_vcl(false, 2)); // same picture (continuation)
        let b_off = stream.len();
        stream.extend_from_slice(&annexb_vcl(true, 3)); // new picture
        let starts = h264_au_starts(&stream);
        assert_eq!(starts, vec![0, b_off], "two access units: A(2 slices) then B");
    }

    #[tokio::test]
    async fn reframing_splits_two_access_units_in_one_buffer() {
        let mut parse = H264Parse::reframing();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = RecordingSink::default();

        // One input buffer carrying two complete pictures back to back.
        let au0 = annexb_vcl(true, 1);
        let au1 = annexb_vcl(true, 2);
        let mut buf = au0.clone();
        buf.extend_from_slice(&au1);
        parse.process(PipelinePacket::DataFrame(frame_with_bytes(0, buf)), &mut sink).await.unwrap();
        // The first AU is emitted once the second AU's start is seen; the second
        // stays buffered until end of stream.
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let payloads = data_payloads(&sink);
        assert_eq!(payloads.len(), 2, "two pictures -> two access-unit DataFrames");
        assert_eq!(payloads[0], au0, "first AU emitted whole");
        assert_eq!(payloads[1], au1, "second AU emitted whole on EOS");
    }

    #[tokio::test]
    async fn reframing_reassembles_an_au_split_across_buffers() {
        // The regression that made HLS video garbage: one Annex-B access unit
        // delivered as two buffers (one MPEG-TS PES carrying the tail), the second
        // with no leading start code. A per-buffer framing guess would misread the
        // tail as AVCC and corrupt it; latching the framing keeps it Annex-B.
        let mut parse = H264Parse::reframing();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let au = annexb_vcl(true, 7);
        let split = 6; // mid-NAL: after the start code + header + first RBSP byte
        parse
            .process(PipelinePacket::DataFrame(frame_with_bytes(0, au[..split].to_vec())), &mut sink)
            .await
            .unwrap();
        parse
            .process(PipelinePacket::DataFrame(frame_with_bytes(1, au[split..].to_vec())), &mut sink)
            .await
            .unwrap();
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let payloads = data_payloads(&sink);
        assert_eq!(payloads.len(), 1, "the split access unit reassembles into one");
        assert_eq!(payloads[0], au, "reassembled bytes are bit-for-bit the original AU");
    }
}

//! H.265 (HEVC) access-unit parser that refines source-side `Caps`.
//!
//! The H.265 sibling of `h264parse`: it scans each `DataFrame` for an SPS NAL
//! (`nal_unit_type == 33`), recovers the coded picture dimensions, and emits a
//! `CapsChanged` with `Dim::Fixed` values before forwarding the frame. This
//! lets a raw H.265 elementary stream (which advertises `Dim::Any` at
//! negotiation, since the SPS only lands once bytes flow) be restreamed or
//! recorded with concrete geometry.
//!
//! H.265's NAL header is two bytes (type is bits `[1..7]` of the first), and
//! the SPS carries a variable-size `profile_tier_level` before the dimensions;
//! for a single-layer stream (`sps_max_sub_layers_minus1 == 0`) that block is a
//! fixed 96 bits. The Annex-B / AVCC framing, the RBSP de-emulation, and the
//! exp-Golomb bit reader are shared with `h264parse` via the `annexb` module.
//!
//! Framerate from the VUI `timing_info` is not recovered yet: in H.265 the VUI
//! sits past the PCM, short-term-ref-pic-set, and long-term-ref loops, too deep
//! to reach safely without a real-stream reference, so caps carry `Rate::Any`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate,
    VideoCodec,
};

use crate::annexb::{
    h265_nal_type, nal_units, next_start_code, strip_emulation_prevention, BitReader,
};

/// H.265 NAL unit type for a sequence parameter set (SPS_NUT).
const SPS_NUT: u8 = 33;
/// H.265 NAL unit types for the video and picture parameter sets.
const VPS_NUT: u8 = 32;
const PPS_NUT: u8 = 34;

/// Re-framing accumulator hard cap: flush rather than buffer forever on
/// non-conforming input (mirrors `h264parse`).
const MAX_REFRAME_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Default)]
pub struct H265Parse {
    configured: bool,
    last_emitted_caps: Option<Caps>,
    sps_emitted: u64,
    /// Re-framing mode (M425): when set, the element re-chunks its input into
    /// access-unit-aligned Annex-B buffers (one coded picture per `DataFrame`)
    /// rather than passing buffers through unchanged. A decoder fed un-aligned
    /// units (e.g. one MPEG-TS PES that is not one access unit) mis-parses slice
    /// boundaries; auto-plugged decode chains insert the parser in this mode so
    /// the decoder always sees one access unit per packet, the HEVC sibling of
    /// the M421 `h264parse` re-framer. Off in the default (caps-refinement-only)
    /// construction, so existing explicit uses keep their pass-through framing.
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
    /// codes, `Some(false)` = HVCC length prefixes). Fixed for a stream, so it is
    /// decided once: a per-frame guess misclassifies a mid-AU Annex-B continuation
    /// buffer (no leading start code) as length-prefixed and mangles it.
    /// Re-framing only.
    input_is_annexb: Option<bool>,
    /// VPS/SPS/PPS re-insertion interval (the gst `config-interval`), in seconds:
    /// `0` = never (default), `-1` = prepend the cached parameter sets to every
    /// IRAP access unit, `N > 0` = prepend at the first IRAP once `N` seconds
    /// elapsed. The HEVC sibling of `h264parse`'s config-interval; applied on the
    /// access-unit-aligned Annex-B (re-framing) output.
    config_interval: i32,
    /// Most-recent VPS / SPS / PPS NAL bodies (start codes stripped), refreshed as
    /// they flow, so an IRAP that lacks inline parameter sets can be prefixed.
    cached_vps: Vec<Vec<u8>>,
    cached_sps: Vec<Vec<u8>>,
    cached_pps: Vec<Vec<u8>>,
    /// PTS (ns) of the last AU the parameter sets were (re-)inserted before.
    last_config_pts_ns: Option<u64>,
}

impl H265Parse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct the parser in access-unit re-framing mode (M425): it accumulates
    /// the input bitstream and emits one access-unit-aligned Annex-B `DataFrame`
    /// per coded picture. Auto-plugged decode chains use this so the decoder is
    /// fed one access unit per packet (the HEVC sibling of `H264Parse::reframing`).
    pub fn reframing() -> Self {
        Self { reframe: true, ..Self::default() }
    }

    /// Count of `CapsChanged` packets pushed downstream, for tests asserting
    /// re-emission is suppressed when the SPS dimensions are unchanged.
    pub fn caps_changes_emitted(&self) -> u64 {
        self.sps_emitted
    }

    /// Set the VPS/SPS/PPS re-insertion interval in seconds (`0` off, `-1` every
    /// IRAP, `N` every N seconds); see [`config_interval`](Self::config_interval).
    pub fn with_config_interval(mut self, seconds: i32) -> Self {
        self.config_interval = seconds;
        self
    }

    /// Refresh the cached parameter sets from `au` and, when re-insertion is due,
    /// return the AU prefixed with the cached VPS/SPS/PPS (Annex-B). `au` must be
    /// Annex-B (the re-framing output always is).
    fn apply_config_interval(&mut self, au: Vec<u8>, pts_ns: u64, keyframe: bool) -> Vec<u8> {
        let mut vps = Vec::new();
        let mut sps = Vec::new();
        let mut pps = Vec::new();
        for nal in nal_units(&au) {
            match h265_nal_type(nal) {
                Some(VPS_NUT) => vps.push(nal.to_vec()),
                Some(SPS_NUT) => sps.push(nal.to_vec()),
                Some(PPS_NUT) => pps.push(nal.to_vec()),
                _ => {}
            }
        }
        let has_config = !sps.is_empty();
        if !vps.is_empty() {
            self.cached_vps = vps;
        }
        if has_config {
            self.cached_sps = sps;
        }
        if !pps.is_empty() {
            self.cached_pps = pps;
        }

        if self.config_interval == 0 || !keyframe {
            return au;
        }
        if has_config {
            self.last_config_pts_ns = Some(pts_ns);
            return au;
        }
        let due = if self.config_interval < 0 {
            true
        } else {
            let interval_ns = (self.config_interval as u64).saturating_mul(1_000_000_000);
            match self.last_config_pts_ns {
                None => true,
                Some(last) => pts_ns.saturating_sub(last) >= interval_ns,
            }
        };
        if !due || self.cached_sps.is_empty() {
            return au;
        }
        let mut out = Vec::with_capacity(au.len() + 96);
        for ps in self.cached_vps.iter().chain(&self.cached_sps).chain(&self.cached_pps) {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(ps);
        }
        out.extend_from_slice(&au);
        self.last_config_pts_ns = Some(pts_ns);
        out
    }

    /// Re-framing data path: accumulate the bitstream and emit each complete
    /// access unit (everything before the last AU start, whose successor's start
    /// code has arrived). The trailing, possibly-incomplete AU stays buffered
    /// until the next call or `Eos`. Non-`System` domains pass through unchanged.
    async fn reframe_frame(
        &mut self,
        frame: Frame,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let MemoryDomain::System(slice) = &frame.domain else {
            out.push(PipelinePacket::DataFrame(frame)).await?;
            return Ok(());
        };
        // Normalize to Annex-B: an HVCC (length-prefixed) stream is converted, an
        // Annex-B one is appended as-is. The framing is latched from the first
        // frame, since a mid-AU Annex-B continuation buffer (no leading start
        // code) would otherwise be misread as length-prefixed.
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

        // Guard against unbounded growth on non-conforming input.
        if self.accum.len() > MAX_REFRAME_BYTES {
            let au = core::mem::take(&mut self.accum);
            let timing = self.au_timing;
            self.emit_au(au, timing, out).await?;
            return Ok(());
        }

        // Emit each complete AU (everything before the last start), retain the tail.
        let starts = h265_au_starts(&self.accum);
        if starts.len() < 2 {
            return Ok(()); // at most one (still-open) AU buffered so far
        }
        let frame_timing = frame.timing;
        let tail = starts[starts.len() - 1];
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
                codec: VideoCodec::H265,
                width: Dim::Fixed(info.width),
                height: Dim::Fixed(info.height),
                framerate: Rate::Any,
            };
            if self.last_emitted_caps.as_ref() != Some(&new_caps) {
                out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                self.last_emitted_caps = Some(new_caps);
                self.sps_emitted += 1;
            }
        }
        timing.keyframe = h265_au_is_keyframe(&au);
        // Re-insert cached VPS/SPS/PPS before this AU when config-interval calls
        // for it (no-op at interval 0 or when the AU already carries them).
        let au = self.apply_config_interval(au, timing.pts_ns, timing.keyframe);
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

impl AsyncElement for H265Parse {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        let supported = Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        upstream_caps.intersect(&supported)
    }

    /// Pass-through identity over H.265 of any geometry (the parser refines
    /// geometry mid-stream from the SPS but never changes media type).
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(CapsSet::one(Caps::CompressedVideo {
            codec: VideoCodec::H265,
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
                codec: VideoCodec::H265,
                ..
            } => {
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "config-interval",
            PropKind::Int,
            "VPS/SPS/PPS re-insertion interval in seconds (0 = off, -1 = every IRAP, N = every N s)",
        )
        .with_range("-1", "3600")
        .with_default("0")];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "config-interval" => {
                let secs = value.as_int().ok_or(PropError::Type)?;
                if !(-1..=3600).contains(&secs) {
                    return Err(PropError::Value);
                }
                self.config_interval = secs as i32;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "config-interval" => Some(PropValue::Int(self.config_interval as i64)),
            _ => None,
        }
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
                PipelinePacket::DataFrame(frame) => {
                    if self.reframe {
                        return self.reframe_frame(frame, out).await;
                    }
                    if let MemoryDomain::System(slice) = &frame.domain {
                        if let Some(info) = extract_sps_info(slice.as_slice()) {
                            let new_caps = Caps::CompressedVideo {
                                codec: VideoCodec::H265,
                                width: Dim::Fixed(info.width),
                                height: Dim::Fixed(info.height),
                                framerate: Rate::Any,
                            };
                            if self.last_emitted_caps.as_ref() != Some(&new_caps) {
                                out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                                    .await?;
                                self.last_emitted_caps = Some(new_caps);
                                self.sps_emitted += 1;
                            }
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    // A seek discontinuity: drop the partial AU rather than splice
                    // pre-seek bytes onto the post-seek stream.
                    self.accum.clear();
                    self.last_emitted_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {
                    // Flush the final buffered access unit, else the last coded
                    // picture would never reach the decoder.
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

impl PadTemplates for H265Parse {
    fn pad_templates() -> Vec<PadTemplate> {
        let h265 = Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(h265.clone())),
            PadTemplate::source(CapsSet::one(h265)),
        ])
    }
}

/// Coded picture dimensions recovered from an SPS (post conformance-window
/// cropping).
struct SpsInfo {
    width: u32,
    height: u32,
}

/// Walk the NALs of `au` (Annex-B or AVCC, auto-detected), returning the info
/// from the first SPS NAL we can parse. H.265 NAL type is bits `[1..7]` of the
/// first header byte.
fn extract_sps_info(au: &[u8]) -> Option<SpsInfo> {
    for nal in crate::annexb::nal_units_any(au) {
        if nal.len() < 2 {
            continue;
        }
        let nal_unit_type = (nal[0] >> 1) & 0x3F;
        if nal_unit_type != SPS_NUT {
            continue;
        }
        // Strip the 2-byte NAL header, then de-emulate the RBSP.
        let rbsp = strip_emulation_prevention(&nal[2..]);
        if let Some(info) = parse_sps(&rbsp) {
            return Some(info);
        }
    }
    None
}

/// Start-code offsets in an Annex-B buffer at which a new H.265 access unit
/// begins, per the ISO/IEC 23008-2 access-unit boundary rules. The first NAL
/// opens the first AU. Once a VCL NAL (`nal_unit_type` 0..=31) has been seen in
/// the current AU, the next AU begins at: a VCL NAL whose
/// `first_slice_segment_in_pic_flag` is 1 (the first coded picture slice, the MSB
/// of the slice RBSP after the 2-byte NAL header), an access-unit delimiter (35),
/// or the parameter-set / prefix-SEI NALs that lead a picture (VPS 32, SPS 33,
/// PPS 34, prefix SEI 39). Slices 2..N of a picture carry the flag as 0 and stay
/// in the same AU. The HEVC sibling of `h264parse`'s `h264_au_starts`.
fn h265_au_starts(data: &[u8]) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut seen_vcl = false;
    let mut i = 0;
    while let Some((sc, begin)) = next_start_code(data, i) {
        let nal_type = data.get(begin).map(|b| (b >> 1) & 0x3F).unwrap_or(0);
        let is_vcl = nal_type <= 31;
        let starts_au = if !seen_vcl {
            // Leading NALs of the first AU: only the very first opens it.
            starts.is_empty()
        } else if is_vcl {
            // A new picture's first slice has first_slice_segment_in_pic_flag == 1,
            // the MSB of the slice RBSP (the byte after the 2-byte NAL header).
            data.get(begin + 2).map(|b| b & 0x80 != 0).unwrap_or(false)
        } else {
            // A non-VCL that can only lead the next access unit.
            matches!(nal_type, 32 | 33 | 34 | 35 | 39)
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

/// True if the access unit contains an IRAP (random-access) picture: a keyframe.
/// HEVC IRAP `nal_unit_type`s are 16..=23 (BLA / IDR / CRA / reserved IRAP).
fn h265_au_is_keyframe(au: &[u8]) -> bool {
    crate::annexb::nal_units_any(au).any(|nal| {
        nal.first()
            .map(|b| (16..=23).contains(&((b >> 1) & 0x3F)))
            .unwrap_or(false)
    })
}

/// Parse the SPS RBSP (H.265 7.3.2.2) up to the conformance window, returning
/// the cropped picture dimensions. `None` on a parse failure before the
/// dimensions resolve.
fn parse_sps(rbsp: &[u8]) -> Option<SpsInfo> {
    let mut br = BitReader::new(rbsp);
    let _sps_video_parameter_set_id = br.read_bits(4)?;
    let sps_max_sub_layers_minus1 = br.read_bits(3)?;
    let _sps_temporal_id_nesting_flag = br.read_bit()?;
    skip_profile_tier_level(&mut br, sps_max_sub_layers_minus1)?;

    let _sps_seq_parameter_set_id = br.read_ue()?;
    let chroma_format_idc = br.read_ue()?;
    let separate_colour_plane_flag = if chroma_format_idc == 3 { br.read_bit()? } else { 0 };
    let pic_width = br.read_ue()?;
    let pic_height = br.read_ue()?;

    let conformance_window_flag = br.read_bit()?;
    let (left, right, top, bottom) = if conformance_window_flag == 1 {
        (br.read_ue()?, br.read_ue()?, br.read_ue()?, br.read_ue()?)
    } else {
        (0, 0, 0, 0)
    };

    // Crop offsets are in chroma sample units, scaled to luma by SubWidthC /
    // SubHeightC (H.265 7.4.3.2.1). ChromaArrayType 0 (monochrome or 4:4:4 with
    // separate colour planes) and 4:4:4 use 1x1.
    let chroma_array_type = if separate_colour_plane_flag == 1 { 0 } else { chroma_format_idc };
    let (sub_width_c, sub_height_c) = match chroma_array_type {
        1 => (2u32, 2u32), // 4:2:0
        2 => (2, 1),       // 4:2:2
        _ => (1, 1),       // 4:4:4 / monochrome
    };
    // Conformance offsets come from untrusted exp-Golomb; saturate the sums so
    // adversarial values cannot overflow before the subtract.
    let width = pic_width.saturating_sub(left.saturating_add(right).saturating_mul(sub_width_c));
    let height = pic_height.saturating_sub(top.saturating_add(bottom).saturating_mul(sub_height_c));
    Some(SpsInfo { width, height })
}

/// Skip `profile_tier_level(1, max_sub_layers_minus1)` (H.265 7.3.3). The
/// general block is a fixed 96 bits (88-bit profile/tier/constraints + 8-bit
/// level); per-sub-layer blocks follow only when `max_sub_layers_minus1 > 0`.
fn skip_profile_tier_level(br: &mut BitReader, max_sub_layers_minus1: u32) -> Option<()> {
    br.skip_bits(88)?; // general profile/tier + constraint/reserved/inbld
    br.skip_bits(8)?; // general_level_idc

    let mut sub_profile_present = [false; 8];
    let mut sub_level_present = [false; 8];
    for i in 0..max_sub_layers_minus1 as usize {
        sub_profile_present[i] = br.read_bit()? == 1;
        sub_level_present[i] = br.read_bit()? == 1;
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            br.read_bits(2)?; // reserved_zero_2bits
        }
    }
    for i in 0..max_sub_layers_minus1 as usize {
        if sub_profile_present[i] {
            br.skip_bits(88)?; // sub_layer profile/tier block
        }
        if sub_level_present[i] {
            br.skip_bits(8)?; // sub_layer_level_idc
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build an Annex-B H.265 SPS for `pic_w` x `pic_h` luma samples at
    /// `chroma_format_idc`, optionally with a conformance window
    /// `(left, right, top, bottom)`. The profile_tier_level is written as 96
    /// zero bits (its content is skipped); the RBSP is emulation-prevented like
    /// a real encoder's output so the parser's de-emulation round-trips it.
    fn build_annexb_sps(
        pic_w: u32,
        pic_h: u32,
        chroma_format_idc: u32,
        conf: Option<(u32, u32, u32, u32)>,
    ) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bits(0, 4); // sps_video_parameter_set_id
        w.write_bits(0, 3); // sps_max_sub_layers_minus1
        w.write_bit(1); // sps_temporal_id_nesting_flag
        for _ in 0..96 {
            w.write_bit(0); // profile_tier_level (single layer = 96 bits)
        }
        w.write_ue(0); // sps_seq_parameter_set_id
        w.write_ue(chroma_format_idc);
        if chroma_format_idc == 3 {
            w.write_bit(0); // separate_colour_plane_flag
        }
        w.write_ue(pic_w);
        w.write_ue(pic_h);
        match conf {
            Some((l, r, t, b)) => {
                w.write_bit(1); // conformance_window_flag
                w.write_ue(l);
                w.write_ue(r);
                w.write_ue(t);
                w.write_ue(b);
            }
            None => w.write_bit(0),
        }
        w.write_bit(1); // rbsp_stop_one_bit
        w.align_to_byte();
        let rbsp = w.into_bytes();
        let ebsp = add_emulation_prevention(&rbsp);

        // 00 00 00 01 | NAL header (type 33, layer 0, tid+1 = 1) | EBSP
        let mut out = vec![0u8, 0, 0, 1, 0x42, 0x01];
        out.extend_from_slice(&ebsp);
        out
    }

    /// Inverse of `annexb::strip_emulation_prevention`: insert `0x03` after each
    /// `00 00` run preceding a byte <= 0x03.
    fn add_emulation_prevention(rbsp: &[u8]) -> Vec<u8> {
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
    fn recovers_dimensions_from_sps() {
        let stream = build_annexb_sps(1920, 1080, 1, None);
        let info = extract_sps_info(&stream).expect("SPS must parse");
        assert_eq!((info.width, info.height), (1920, 1080));
    }

    #[test]
    fn applies_conformance_window_cropping() {
        // 1920x1088 coded, 4:2:0 (SubHeightC = 2), crop 4 chroma rows off the
        // bottom -> 1088 - 2*4 = 1080.
        let stream = build_annexb_sps(1920, 1088, 1, Some((0, 0, 0, 4)));
        let info = extract_sps_info(&stream).expect("SPS with conf window must parse");
        assert_eq!((info.width, info.height), (1920, 1080));
    }

    #[test]
    fn saturates_adversarial_conformance_offsets() {
        // Huge conformance offsets must saturate, not overflow-panic on the sum.
        let huge = 3_000_000_000u32;
        let stream = build_annexb_sps(1920, 1080, 1, Some((huge, huge, huge, huge)));
        let info = extract_sps_info(&stream).expect("parses without overflow");
        assert_eq!((info.width, info.height), (0, 0), "offsets clamp dims to zero");
    }

    #[test]
    fn parses_an_avcc_framed_sps() {
        // Re-frame the SPS NAL as length-prefixed (HVCC-style) and confirm the
        // dimensions still resolve.
        let annexb = build_annexb_sps(1280, 720, 1, None);
        let nal = &annexb[4..]; // drop the 00 00 00 01 start code
        let mut hvcc = (nal.len() as u32).to_be_bytes().to_vec();
        hvcc.extend_from_slice(nal);
        let info = extract_sps_info(&hvcc).expect("length-prefixed SPS must parse");
        assert_eq!((info.width, info.height), (1280, 720));
    }

    #[test]
    fn ignores_non_sps_nals() {
        // A TRAIL_R slice NAL (type 1 -> first byte 0x02) carries no SPS.
        let stream = [0u8, 0, 0, 1, 0x02, 0x01, 0xAA, 0xBB];
        assert!(extract_sps_info(&stream).is_none());
    }

    #[test]
    fn returns_none_on_empty_input() {
        assert!(extract_sps_info(&[]).is_none());
    }

    #[test]
    fn skip_profile_tier_level_advances_96_bits_for_single_layer() {
        // 12 bytes of PTL then a ue(0) = single '1' bit. After the skip the
        // reader must sit exactly on that bit.
        let mut w = BitWriter::default();
        for _ in 0..96 {
            w.write_bit(0);
        }
        w.write_bit(1); // a marker the reader should land on (ue value 0)
        w.align_to_byte();
        let bytes = w.into_bytes();
        let mut br = BitReader::new(&bytes);
        skip_profile_tier_level(&mut br, 0).expect("skips the fixed 96-bit block");
        assert_eq!(br.read_ue(), Some(0), "reader landed just past the PTL");
    }

    // -- Element-level tests (drive H265Parse::process directly) -----------

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

    fn h265_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = H265Parse::new();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let frame = frame_with_bytes(0, build_annexb_sps(1920, 1080, 1, None));
        parse
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        assert_eq!(sink.packets.len(), 2, "expected CapsChanged then DataFrame");
        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::CompressedVideo {
                codec: VideoCodec::H265,
                width,
                height,
                ..
            }) => {
                assert_eq!(*width, Dim::Fixed(1920));
                assert_eq!(*height, Dim::Fixed(1080));
            }
            other => panic!("expected H.265 CapsChanged first, got {other:?}"),
        }
        assert!(matches!(sink.packets[1], PipelinePacket::DataFrame(_)));
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn does_not_re_emit_caps_when_unchanged() {
        let mut parse = H265Parse::new();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        for seq in 0..3 {
            let frame = frame_with_bytes(seq, build_annexb_sps(1280, 720, 1, None));
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
        assert_eq!(caps_count, 1, "CapsChanged fires once for identical SPS");
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn re_emits_caps_on_resolution_change() {
        let mut parse = H265Parse::new();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, build_annexb_sps(1280, 720, 1, None))),
                &mut sink,
            )
            .await
            .unwrap();
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(
                    1,
                    build_annexb_sps(1920, 1080, 1, None),
                )),
                &mut sink,
            )
            .await
            .unwrap();

        let widths: Vec<Dim> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(Caps::CompressedVideo { width, .. }) => {
                    Some(width.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(widths, vec![Dim::Fixed(1280), Dim::Fixed(1920)]);
        assert_eq!(parse.caps_changes_emitted(), 2);
    }

    #[tokio::test]
    async fn rejects_non_h265_caps_in_intercept() {
        let parse = H265Parse::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(parse.intercept_caps(&h264), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_constraint_is_identity_h265_any() {
        let parse = H265Parse::new();
        let c = parse.caps_constraint_as_transform();
        match c {
            CapsConstraint::Identity(set) => {
                assert_eq!(
                    set.alternatives(),
                    &[Caps::CompressedVideo {
                        codec: VideoCodec::H265,
                        width: Dim::Any,
                        height: Dim::Any,
                        framerate: Rate::Any,
                    }]
                );
            }
            _ => panic!("expected Identity"),
        }
    }

    // -- Re-framing (M425) ------------------------------------------------

    /// One Annex-B H.265 VCL NAL (TRAIL_R, type 1). `first` sets
    /// `first_slice_segment_in_pic_flag` (the MSB of the slice RBSP, the byte after
    /// the 2-byte NAL header), so `first == true` opens a new coded picture.
    fn annexb_vcl_h265(first: bool, tag: u8) -> Vec<u8> {
        // start code + NAL header: type 1 (TRAIL_R) = (1 << 1) = 0x02, layer 0;
        // second header byte 0x01 = temporal_id_plus1.
        let mut v = vec![0u8, 0, 0, 1, 0x02, 0x01];
        v.push(if first { 0x80 } else { 0x40 }); // slice RBSP: MSB = first_slice flag
        v.extend_from_slice(&[0xAA, 0xBB, tag, 0x11]);
        v
    }

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
        // Picture A: two slices (first then a continuation). Picture B: one slice.
        // Two access units, not three.
        let mut stream = annexb_vcl_h265(true, 1);
        stream.extend_from_slice(&annexb_vcl_h265(false, 2)); // same picture
        let b_off = stream.len();
        stream.extend_from_slice(&annexb_vcl_h265(true, 3)); // new picture
        let starts = h265_au_starts(&stream);
        assert_eq!(starts, vec![0, b_off], "two access units: A(2 slices) then B");
    }

    #[tokio::test]
    async fn reframing_splits_two_access_units_in_one_buffer() {
        let mut parse = H265Parse::reframing();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let au0 = annexb_vcl_h265(true, 1);
        let au1 = annexb_vcl_h265(true, 2);
        let mut buf = au0.clone();
        buf.extend_from_slice(&au1);
        parse.process(PipelinePacket::DataFrame(frame_with_bytes(0, buf)), &mut sink).await.unwrap();
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let payloads = data_payloads(&sink);
        assert_eq!(payloads.len(), 2, "two pictures -> two access-unit DataFrames");
        assert_eq!(payloads[0], au0, "first AU emitted whole");
        assert_eq!(payloads[1], au1, "second AU emitted whole on EOS");
    }

    #[tokio::test]
    async fn reframing_reassembles_an_au_split_across_buffers() {
        // One Annex-B access unit delivered as two buffers (e.g. one MPEG-TS PES
        // carrying the tail), the second with no leading start code. Latching the
        // framing keeps it Annex-B instead of misreading the tail as length-prefixed.
        let mut parse = H265Parse::reframing();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let au = annexb_vcl_h265(true, 7);
        let split = 7; // mid-NAL: past start code + 2-byte header + first RBSP byte
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

    #[tokio::test]
    async fn reframing_stamps_keyframe_on_irap() {
        // An IDR_W_RADL (type 19) access unit is a keyframe; a TRAIL_R (type 1) is not.
        let mut parse = H265Parse::reframing();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        // IDR: NAL header first byte = (19 << 1) = 0x26, first-slice flag set.
        let mut idr = vec![0u8, 0, 0, 1, 0x26, 0x01, 0x80, 0xAA];
        let trail = annexb_vcl_h265(true, 2);
        idr.extend_from_slice(&trail);
        parse.process(PipelinePacket::DataFrame(frame_with_bytes(0, idr)), &mut sink).await.unwrap();
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let kf: Vec<bool> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f.timing.keyframe),
                _ => None,
            })
            .collect();
        assert_eq!(kf, vec![true, false], "IDR AU is a keyframe, TRAIL_R is not");
    }

    #[test]
    fn config_interval_reinserts_vps_sps_pps_on_irap() {
        let mut p = H265Parse::reframing().with_config_interval(-1);
        // An IRAP AU with VPS (32) + SPS (33) + PPS (34): cached, returned as-is.
        // H.265 NAL header type is bits 1..=6 of the first byte: 32<<1=0x40 etc.
        let mut au1 = vec![0, 0, 0, 1, 0x40, 0x01]; // VPS
        au1.extend_from_slice(&[0, 0, 0, 1, 0x42, 0x01]); // SPS
        au1.extend_from_slice(&[0, 0, 0, 1, 0x44, 0x01]); // PPS
        au1.extend_from_slice(&[0, 0, 0, 1, 0x26, 0x01]); // IDR_W_RADL (19) slice
        let out1 = p.apply_config_interval(au1.clone(), 0, true);
        assert_eq!(out1, au1, "an IRAP that already carries parameter sets is untouched");
        // A later IRAP with no parameter sets gets VPS/SPS/PPS prepended.
        let au2 = vec![0, 0, 0, 1, 0x26, 0x01];
        let out2 = p.apply_config_interval(au2.clone(), 90_000, true);
        assert!(nal_units(&out2).any(|n| h265_nal_type(n) == Some(VPS_NUT)), "result carries a VPS");
        assert!(nal_units(&out2).any(|n| h265_nal_type(n) == Some(SPS_NUT)), "result carries an SPS");
        assert!(nal_units(&out2).any(|n| h265_nal_type(n) == Some(PPS_NUT)), "result carries a PPS");
        assert!(out2.ends_with(&au2), "the original AU is preserved at the tail");
    }
}

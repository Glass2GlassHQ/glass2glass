//! Shared SEI message walk for the H.264 / H.265 elementary streams, plus the
//! HDR10 static-metadata payloads.
//!
//! An access unit's SEI NALs carry a run of `(payloadType, payloadSize, payload)`
//! messages whose framing is identical in both codecs; only the NAL header
//! differs. [`for_each_message`] does that walk once so every consumer (closed
//! captions in [`crate::cea`], the mastering-display / content-light-level
//! parsers here) reads the same bytes without re-scanning the NALs.
//!
//! **Never trust the stream.** Payload sizes and counts are attacker-controlled,
//! so every read is bounds-checked and a malformed message yields nothing rather
//! than panicking.

use alloc::vec::Vec;

use g2g_core::VideoCodec;

use crate::annexb::{
    h264_nal_type, h265_nal_type, nal_units_any, read_ff_extended, strip_emulation_prevention,
    BitReader,
};

/// `pic_timing`, which carries the H.264 SMPTE 12M clock timestamps.
pub const PAYLOAD_PIC_TIMING: usize = 1;
/// `user_data_registered_itu_t_t35`, the ATSC A/53 closed-caption carrier.
pub const PAYLOAD_USER_DATA_REGISTERED: usize = 4;
/// `time_code`, the H.265 SMPTE 12M clock timestamps.
pub const PAYLOAD_TIME_CODE: usize = 136;
/// `mastering_display_colour_volume` (SMPTE ST 2086).
pub const PAYLOAD_MASTERING_DISPLAY: usize = 137;
/// `content_light_level_info` (CTA-861.3 MaxCLL / MaxFALL).
pub const PAYLOAD_CONTENT_LIGHT_LEVEL: usize = 144;

/// Call `f(payload_type, payload)` for every SEI message in one access unit
/// (`au`, Annex-B or AVCC framed) of `codec`, in bitstream order. Messages after
/// a malformed one in the same NAL are skipped; the remaining NALs still parse.
pub fn for_each_message(au: &[u8], codec: VideoCodec, mut f: impl FnMut(usize, &[u8])) {
    for nal in nal_units_any(au) {
        // SEI NAL header + RBSP offset differs by codec: H.264 SEI is NAL type 6
        // with a 1-byte header; H.265 prefix-SEI (39) / suffix-SEI (40) carry a
        // 2-byte header.
        let rbsp_off = match codec {
            VideoCodec::H265 => match h265_nal_type(nal) {
                Some(39) | Some(40) => 2,
                _ => continue,
            },
            _ => match h264_nal_type(nal) {
                Some(6) => 1,
                _ => continue,
            },
        };
        if nal.len() <= rbsp_off {
            continue;
        }
        let rbsp = strip_emulation_prevention(&nal[rbsp_off..]);
        let mut i = 0usize;
        // Stop once only the rbsp_trailing_bits (a lone 0x80) remain.
        while i + 1 < rbsp.len() {
            let Some((payload_type, n)) = read_ff_extended(&rbsp, i) else {
                break;
            };
            i = n;
            let Some((payload_size, n)) = read_ff_extended(&rbsp, i) else {
                break;
            };
            i = n;
            let end = match i.checked_add(payload_size) {
                Some(e) if e <= rbsp.len() => e,
                _ => break,
            };
            f(payload_type, &rbsp[i..end]);
            i = end;
        }
    }
}

/// Read a big-endian `u16` at `off`, or `None` past the end. Only the HDR
/// payload parsers read fixed-width fields.
#[cfg(feature = "metadata")]
fn be16(p: &[u8], off: usize) -> Option<u16> {
    let b = p.get(off..off.checked_add(2)?)?;
    Some(u16::from_be_bytes([b[0], b[1]]))
}

/// Read a big-endian `u32` at `off`, or `None` past the end.
#[cfg(feature = "metadata")]
fn be32(p: &[u8], off: usize) -> Option<u32> {
    let b = p.get(off..off.checked_add(4)?)?;
    Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// Parse a `mastering_display_colour_volume` payload (24 bytes, ST 2086).
/// Chromaticities are coded in increments of 0.00002, the peak luminance in
/// 0.0001 cd/m^2 and the black level likewise. The SEI orders the primaries
/// **G, B, R**; the result is in R, G, B order.
#[cfg(feature = "metadata")]
pub fn parse_mastering_display(p: &[u8]) -> Option<g2g_core::meta::MasteringDisplay> {
    use g2g_core::meta::{Chromaticity, MasteringDisplay};
    if p.len() < 24 {
        return None;
    }
    let chroma = |i: usize| -> Option<Chromaticity> {
        Some(Chromaticity {
            x: be16(p, i * 4)? as f32 / 50_000.0,
            y: be16(p, i * 4 + 2)? as f32 / 50_000.0,
        })
    };
    let (g, b, r) = (chroma(0)?, chroma(1)?, chroma(2)?);
    Some(MasteringDisplay {
        display_primaries: [r, g, b],
        white_point: chroma(3)?,
        // Both luminance fields are u32 in units of 0.0001 cd/m^2.
        max_luminance: be32(p, 16)? as f32 / 10_000.0,
        min_luminance: be32(p, 20)? as f32 / 10_000.0,
    })
}

/// Parse a `content_light_level_info` payload (4 bytes), returning
/// `(MaxCLL, MaxFALL)` in cd/m^2.
#[cfg(feature = "metadata")]
pub fn parse_content_light_level(p: &[u8]) -> Option<(u16, u16)> {
    if p.len() < 4 {
        return None;
    }
    Some((be16(p, 0)?, be16(p, 2)?))
}

/// What an H.264 `pic_timing` SEI needs from the SPS VUI to be parseable at all:
/// the message is not self-delimiting, its leading CPB/DPB delays are sized by
/// the HRD parameters and the clock timestamps are only present when the VUI
/// says so. H.265's `time_code` SEI carries all of this itself, so it needs no
/// context.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PicTimingContext {
    /// `CpbDpbDelaysPresentFlag`: either HRD block is present, so the message
    /// opens with the two delays.
    pub cpb_dpb_delays_present: bool,
    /// Bit widths of `cpb_removal_delay` / `dpb_output_delay`.
    pub cpb_removal_delay_length: u8,
    pub dpb_output_delay_length: u8,
    /// `pic_struct_present_flag`: without it the message carries no timestamps.
    pub pic_struct_present: bool,
    /// Bit width of each clock timestamp's `time_offset`.
    pub time_offset_length: u8,
}

/// Read the `hours / minutes / seconds` of one clock timestamp, whose nesting is
/// identical in H.264 and H.265: a full timestamp codes all three, otherwise
/// each coarser field is present only if the finer one was.
#[cfg(feature = "metadata")]
fn read_hms(br: &mut BitReader, full_timestamp: bool) -> Option<(u8, u8, u8)> {
    if full_timestamp {
        let s = br.read_bits(6)?;
        let m = br.read_bits(6)?;
        let h = br.read_bits(5)?;
        return Some((h as u8, m as u8, s as u8));
    }
    let (mut h, mut m, mut s) = (0u32, 0u32, 0u32);
    if br.read_bit()? == 1 {
        s = br.read_bits(6)?;
        if br.read_bit()? == 1 {
            m = br.read_bits(6)?;
            if br.read_bit()? == 1 {
                h = br.read_bits(5)?;
            }
        }
    }
    Some((h as u8, m as u8, s as u8))
}

/// Parse an H.265 `time_code` payload, returning the first clock timestamp it
/// codes. `None` when it codes none or the payload is malformed.
#[cfg(feature = "metadata")]
pub fn parse_time_code(p: &[u8]) -> Option<g2g_core::meta::TimecodeMeta> {
    let mut br = BitReader::new(p);
    let num_clock_ts = br.read_bits(2)?;
    for _ in 0..num_clock_ts {
        if br.read_bit()? != 1 {
            continue;
        }
        br.read_bit()?; // units_field_based_flag
        br.read_bits(5)?; // counting_type
        let full = br.read_bit()? == 1;
        br.read_bit()?; // discontinuity_flag
        let drop_frame = br.read_bit()? == 1;
        // n_frames is 9 bits here; a count that does not fit a frame number is
        // malformed, so the timestamp is dropped rather than truncated.
        let n_frames = u8::try_from(br.read_bits(9)?).ok()?;
        let (hours, minutes, seconds) = read_hms(&mut br, full)?;
        let time_offset_length = br.read_bits(5)?;
        br.skip_bits(time_offset_length as usize)?;
        return Some(g2g_core::meta::TimecodeMeta {
            hours,
            minutes,
            seconds,
            frames: n_frames,
            drop_frame,
            framerate_q16: None,
        });
    }
    None
}

/// Parse an H.264 `pic_timing` payload with the SPS VUI `ctx` it needs,
/// returning the first clock timestamp it codes.
#[cfg(feature = "metadata")]
pub fn parse_pic_timing(p: &[u8], ctx: PicTimingContext) -> Option<g2g_core::meta::TimecodeMeta> {
    if !ctx.pic_struct_present {
        return None;
    }
    let mut br = BitReader::new(p);
    if ctx.cpb_dpb_delays_present {
        br.skip_bits(ctx.cpb_removal_delay_length as usize)?;
        br.skip_bits(ctx.dpb_output_delay_length as usize)?;
    }
    // NumClockTS per pic_struct (H.264 table D-1).
    let num_clock_ts = match br.read_bits(4)? {
        0..=2 => 1,
        3 | 4 | 7 => 2,
        5 | 6 | 8 => 3,
        _ => return None,
    };
    for _ in 0..num_clock_ts {
        if br.read_bit()? != 1 {
            continue;
        }
        br.read_bits(2)?; // ct_type
        br.read_bit()?; // nuit_field_based_flag
        br.read_bits(5)?; // counting_type
        let full = br.read_bit()? == 1;
        br.read_bit()?; // discontinuity_flag
        let drop_frame = br.read_bit()? == 1;
        let n_frames = br.read_bits(8)? as u8;
        let (hours, minutes, seconds) = read_hms(&mut br, full)?;
        br.skip_bits(ctx.time_offset_length as usize)?;
        return Some(g2g_core::meta::TimecodeMeta {
            hours,
            minutes,
            seconds,
            frames: n_frames,
            drop_frame,
            framerate_q16: None,
        });
    }
    None
}

/// Everything one access unit's SEI messages yield, from a single walk.
#[derive(Debug, Default)]
pub struct SeiInfo {
    /// ATSC A/53 closed-caption `cc_data` triples, in transmission order.
    pub captions: Vec<crate::cea::CcTriple>,
    /// HDR10 static metadata, `None` when the AU carries neither payload.
    #[cfg(feature = "metadata")]
    pub hdr: Option<g2g_core::meta::HdrStaticMeta>,
    /// SMPTE 12M timecode, `None` when the AU codes none.
    #[cfg(feature = "metadata")]
    pub timecode: Option<g2g_core::meta::TimecodeMeta>,
}

/// Walk one access unit's SEI messages once, collecting everything the parser
/// mines from them. `pic_timing` is the SPS VUI context an H.264 `pic_timing`
/// message needs (default for H.265, or when the caller has no SPS yet: the
/// timecode is then simply not recovered).
pub fn parse_au(au: &[u8], codec: VideoCodec, pic_timing: PicTimingContext) -> SeiInfo {
    let mut info = SeiInfo::default();
    #[cfg(feature = "metadata")]
    let mut hdr = g2g_core::meta::HdrStaticMeta::default();
    #[cfg(not(feature = "metadata"))]
    let _ = pic_timing;
    for_each_message(au, codec, |ty, payload| match ty {
        PAYLOAD_USER_DATA_REGISTERED => {
            crate::cea::parse_caption_payload(payload, &mut info.captions)
        }
        #[cfg(feature = "metadata")]
        PAYLOAD_PIC_TIMING => {
            if codec != VideoCodec::H265 {
                info.timecode = parse_pic_timing(payload, pic_timing);
            }
        }
        #[cfg(feature = "metadata")]
        PAYLOAD_TIME_CODE => {
            if codec == VideoCodec::H265 {
                info.timecode = parse_time_code(payload);
            }
        }
        #[cfg(feature = "metadata")]
        PAYLOAD_MASTERING_DISPLAY => {
            if let Some(m) = parse_mastering_display(payload) {
                hdr.mastering = Some(m);
            }
        }
        #[cfg(feature = "metadata")]
        PAYLOAD_CONTENT_LIGHT_LEVEL => {
            if let Some((cll, fall)) = parse_content_light_level(payload) {
                hdr.max_content_light_level = Some(cll);
                hdr.max_frame_average_light_level = Some(fall);
            }
        }
        _ => {}
    });
    #[cfg(feature = "metadata")]
    {
        info.hdr = (!hdr.is_empty()).then_some(hdr);
    }
    info
}

/// Wrap `payload` as an Annex-B SEI NAL of `payload_type` for `codec`, the
/// framing [`for_each_message`] reads back. Test / authoring helper: the caption
/// path has its own builder ([`crate::cea::build_cc_sei`]) that also writes the
/// ATSC header.
pub fn build_sei_nal(payload_type: usize, payload: &[u8], codec: VideoCodec) -> Vec<u8> {
    let mut rbsp = Vec::new();
    write_ff_extended(&mut rbsp, payload_type);
    write_ff_extended(&mut rbsp, payload.len());
    rbsp.extend_from_slice(payload);
    rbsp.push(0x80); // rbsp_trailing_bits

    let mut nal = alloc::vec![0x00, 0x00, 0x00, 0x01];
    match codec {
        // H.265 prefix-SEI: nal_unit_type 39, layer 0, tid 1.
        VideoCodec::H265 => nal.extend_from_slice(&[0x4E, 0x01]),
        _ => nal.push(0x06),
    }
    nal.extend_from_slice(&crate::annexb::add_emulation_prevention(&rbsp));
    nal
}

/// Write an SEI `0xFF`-extended value (the inverse of [`read_ff_extended`]): a
/// run of `0xFF` bytes each worth 255, then the remainder.
pub(crate) fn write_ff_extended(out: &mut Vec<u8>, mut value: usize) {
    while value >= 0xFF {
        out.push(0xFF);
        value -= 0xFF;
    }
    out.push(value as u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "metadata")]
    fn mastering_payload() -> Vec<u8> {
        // BT.2020 primaries in SEI order (G, B, R), D65 white, 1000 / 0.005 nits.
        let mut p = Vec::new();
        for (x, y) in [
            (0.170, 0.797), // G
            (0.131, 0.046), // B
            (0.708, 0.292), // R
            (0.3127, 0.3290),
        ] {
            p.extend_from_slice(&(((x * 50_000.0) as u16).to_be_bytes()));
            p.extend_from_slice(&(((y * 50_000.0) as u16).to_be_bytes()));
        }
        p.extend_from_slice(&10_000_000u32.to_be_bytes()); // 1000 cd/m^2
        p.extend_from_slice(&50u32.to_be_bytes()); // 0.005 cd/m^2
        p
    }

    #[test]
    #[cfg(feature = "metadata")]
    fn parses_a_mastering_display_and_light_level_sei() {
        let mut au = alloc::vec![0x00, 0x00, 0x00, 0x01, 0x65, 0x88];
        let mut cll = Vec::new();
        cll.extend_from_slice(&1200u16.to_be_bytes());
        cll.extend_from_slice(&300u16.to_be_bytes());
        let mut stream = build_sei_nal(
            PAYLOAD_MASTERING_DISPLAY,
            &mastering_payload(),
            VideoCodec::H264,
        );
        stream.extend_from_slice(&build_sei_nal(
            PAYLOAD_CONTENT_LIGHT_LEVEL,
            &cll,
            VideoCodec::H264,
        ));
        stream.append(&mut au);

        let hdr = parse_au(&stream, VideoCodec::H264, PicTimingContext::default())
            .hdr
            .expect("HDR SEI parsed");
        let m = hdr.mastering.expect("mastering display present");
        // Primaries come back in R, G, B order.
        assert!((m.display_primaries[0].x - 0.708).abs() < 1e-4, "red x");
        assert!((m.display_primaries[1].y - 0.797).abs() < 1e-4, "green y");
        assert!((m.display_primaries[2].x - 0.131).abs() < 1e-4, "blue x");
        assert!((m.white_point.x - 0.3127).abs() < 1e-4);
        assert!((m.max_luminance - 1000.0).abs() < 1e-3);
        assert!((m.min_luminance - 0.005).abs() < 1e-6);
        assert_eq!(hdr.max_content_light_level, Some(1200));
        assert_eq!(hdr.max_frame_average_light_level, Some(300));
    }

    #[test]
    #[cfg(feature = "metadata")]
    fn a_truncated_mastering_sei_yields_no_metadata() {
        // A short payload must be rejected, not read past its end.
        let payload = mastering_payload();
        for cut in [0usize, 1, 12, 23] {
            let nal = build_sei_nal(PAYLOAD_MASTERING_DISPLAY, &payload[..cut], VideoCodec::H264);
            assert!(
                parse_au(&nal, VideoCodec::H264, PicTimingContext::default())
                    .hdr
                    .is_none(),
                "truncated to {cut} bytes must not parse"
            );
        }
    }

    #[test]
    #[cfg(feature = "metadata")]
    fn a_stream_without_hdr_sei_has_no_metadata() {
        let au = alloc::vec![0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00];
        assert!(parse_au(&au, VideoCodec::H264, PicTimingContext::default())
            .hdr
            .is_none());
    }

    #[test]
    #[cfg(feature = "metadata")]
    fn parses_an_h265_time_code() {
        use crate::annexb::BitWriter;
        let mut w = BitWriter::default();
        w.write_bits(1, 2); // num_clock_ts
        w.write_bit(1); // clock_timestamp_flag
        w.write_bit(0); // units_field_based_flag
        w.write_bits(0, 5); // counting_type
        w.write_bit(1); // full_timestamp_flag
        w.write_bit(0); // discontinuity_flag
        w.write_bit(1); // cnt_dropped_flag (drop frame)
        w.write_bits(29, 9); // n_frames
        w.write_bits(58, 6); // seconds_value
        w.write_bits(59, 6); // minutes_value
        w.write_bits(10, 5); // hours_value
        w.write_bits(0, 5); // time_offset_length
        w.align_to_byte();

        let tc = parse_time_code(&w.into_bytes()).expect("time_code parsed");
        assert_eq!(
            (tc.hours, tc.minutes, tc.seconds, tc.frames),
            (10, 59, 58, 29)
        );
        assert!(tc.drop_frame);
    }

    #[test]
    #[cfg(feature = "metadata")]
    fn parses_an_h264_pic_timing_with_hrd_delays() {
        use crate::annexb::BitWriter;
        let ctx = PicTimingContext {
            cpb_dpb_delays_present: true,
            cpb_removal_delay_length: 24,
            dpb_output_delay_length: 24,
            pic_struct_present: true,
            time_offset_length: 0,
        };
        let mut w = BitWriter::default();
        w.write_bits(0, 24); // cpb_removal_delay
        w.write_bits(0, 24); // dpb_output_delay
        w.write_bits(0, 4); // pic_struct: one clock timestamp
        w.write_bit(1); // clock_timestamp_flag
        w.write_bits(0, 2); // ct_type
        w.write_bit(0); // nuit_field_based_flag
        w.write_bits(0, 5); // counting_type
        w.write_bit(1); // full_timestamp_flag
        w.write_bit(0); // discontinuity_flag
        w.write_bit(0); // cnt_dropped_flag
        w.write_bits(12, 8); // n_frames
        w.write_bits(34, 6); // seconds_value
        w.write_bits(56, 6); // minutes_value
        w.write_bits(7, 5); // hours_value
        w.align_to_byte();
        let payload = w.into_bytes();

        let tc = parse_pic_timing(&payload, ctx).expect("pic_timing parsed");
        assert_eq!(
            (tc.hours, tc.minutes, tc.seconds, tc.frames),
            (7, 56, 34, 12)
        );
        assert!(!tc.drop_frame);

        // Without the VUI flag the message carries no timestamps at all, so the
        // same bytes must not be mined for one.
        let no_pic_struct = PicTimingContext {
            pic_struct_present: false,
            ..ctx
        };
        assert!(parse_pic_timing(&payload, no_pic_struct).is_none());
        // Mis-sized delays walk off the end rather than inventing a timecode.
        let truncated = PicTimingContext {
            cpb_removal_delay_length: 255,
            ..ctx
        };
        assert!(parse_pic_timing(&payload, truncated).is_none());
    }

    #[test]
    fn walks_h265_prefix_sei_messages() {
        let nal = build_sei_nal(PAYLOAD_CONTENT_LIGHT_LEVEL, &[0, 1, 0, 2], VideoCodec::H265);
        let mut seen = Vec::new();
        for_each_message(&nal, VideoCodec::H265, |ty, p| seen.push((ty, p.to_vec())));
        assert_eq!(seen, alloc::vec![(144usize, alloc::vec![0, 1, 0, 2])]);
    }
}

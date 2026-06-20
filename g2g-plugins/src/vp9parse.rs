//! VP9 frame parser that refines source-side `Caps` from the uncompressed header.
//!
//! The VP9 sibling of `vp8parse`: it reads each frame's uncompressed header (the
//! VP9 bitstream `uncompressed_header()`) and, on a key frame, the coded
//! dimensions, emitting a `CapsChanged` with `Dim::Fixed` width/height before
//! forwarding. A demuxer (mkvdemux) can take geometry from the container Tracks;
//! this recovers it from the bitstream when that is absent.
//!
//! Unlike VP8 the VP9 header is bit-packed (not byte fields): `frame_marker`,
//! profile, the `49 83 42` sync code, a variable-length `color_config`, then the
//! 16-bit `frame_width_minus_1` / `frame_height_minus_1`, all MSB-first, so it
//! uses the shared `annexb::BitReader`. Dimensions are read only from key frames
//! (`frame_type == 0`); inter and `show_existing_frame` frames carry none and
//! forward without a caps change. VP9 has no framerate in the bitstream, so
//! refined caps report `Rate::Any` (matching mkvdemux), refining only geometry.
//! Intra-only inter frames (which can also resize) are a follow-up.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, Rate, VideoCodec,
};

use crate::annexb::BitReader;

/// VP9 colour space signalling RGB (`CS_RGB`); its `color_config` branch omits
/// the `color_range` bit the YUV spaces carry.
const CS_RGB: u32 = 7;

#[derive(Debug, Default)]
pub struct Vp9Parse {
    configured: bool,
    last_emitted_caps: Option<Caps>,
    keyframes_emitted: u64,
}

impl Vp9Parse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count of `CapsChanged` packets pushed downstream, for tests asserting
    /// re-emission is suppressed when the dimensions are unchanged.
    pub fn caps_changes_emitted(&self) -> u64 {
        self.keyframes_emitted
    }
}

impl AsyncElement for Vp9Parse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        let supported = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        upstream_caps.intersect(&supported)
    }

    /// Pass-through identity over VP9 of any geometry (the parser refines
    /// geometry mid-stream from the keyframe but never changes media type).
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(CapsSet::one(Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo { codec: VideoCodec::Vp9, .. } => {
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
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
                    if let g2g_core::MemoryDomain::System(slice) = &frame.domain {
                        if let Some(info) = parse_keyframe(slice.as_slice()) {
                            let new_caps = Caps::CompressedVideo {
                                codec: VideoCodec::Vp9,
                                width: Dim::Fixed(info.width),
                                height: Dim::Fixed(info.height),
                                framerate: Rate::Any,
                            };
                            if self.last_emitted_caps.as_ref() != Some(&new_caps) {
                                out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                                self.last_emitted_caps = Some(new_caps);
                                self.keyframes_emitted += 1;
                            }
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    self.last_emitted_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for Vp9Parse {
    fn pad_templates() -> Vec<PadTemplate> {
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([PadTemplate::sink(CapsSet::one(vp9.clone())), PadTemplate::source(CapsSet::one(vp9))])
    }
}

/// The dimensions (and profile) decoded from a VP9 key frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vp9KeyFrame {
    pub width: u32,
    pub height: u32,
    /// Bitstream profile (0..=3): bit depth + chroma subsampling support. Not
    /// carried in `Caps`, surfaced for completeness.
    pub profile: u8,
}

/// Decode a VP9 key-frame uncompressed header. `None` when the marker / sync
/// code don't match, the frame is not a (shown) key frame, or the header is
/// truncated before the size fields.
fn parse_keyframe(packet: &[u8]) -> Option<Vp9KeyFrame> {
    let mut br = BitReader::new(packet);
    if br.read_bits(2)? != 2 {
        return None; // frame_marker is always 0b10
    }
    let profile_low = br.read_bit()?;
    let profile_high = br.read_bit()?;
    let profile = ((profile_high << 1) | profile_low) as u8;
    if profile == 3 {
        br.read_bit()?; // reserved_zero
    }
    if br.read_bit()? == 1 {
        return None; // show_existing_frame references a stored frame, no new dims
    }
    if br.read_bit()? != 0 {
        return None; // frame_type: only KEY_FRAME (0) carries the size here
    }
    br.read_bit()?; // show_frame
    br.read_bit()?; // error_resilient_mode

    // frame_sync_code: 0x49 0x83 0x42, bit-aligned to the current position.
    if br.read_bits(8)? != 0x49 || br.read_bits(8)? != 0x83 || br.read_bits(8)? != 0x42 {
        return None;
    }

    // color_config: its length depends on profile and colour space.
    if profile >= 2 {
        br.read_bit()?; // ten_or_twelve_bit (bit depth)
    }
    let color_space = br.read_bits(3)?;
    if color_space != CS_RGB {
        br.read_bit()?; // color_range
        if profile == 1 || profile == 3 {
            br.read_bits(3)?; // subsampling_x, subsampling_y, reserved_zero
        }
    } else if profile == 1 || profile == 3 {
        br.read_bit()?; // reserved_zero
    }

    // frame_size: 16-bit minus-1 fields.
    let width = br.read_bits(16)? + 1;
    let height = br.read_bits(16)? + 1;
    Some(Vp9KeyFrame { width, height, profile })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// MSB-first bit writer, the inverse of `annexb::BitReader`, for building
    /// VP9 header fixtures.
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

        fn align_to_byte(&mut self) {
            while self.bit_pos % 8 != 0 {
                self.write_bit(0);
            }
        }

        fn into_bytes(self) -> Vec<u8> {
            self.buf
        }
    }

    /// Build a VP9 key-frame uncompressed header for `width` x `height` at
    /// `profile` (colour space BT.709, 4:2:0).
    fn keyframe(width: u32, height: u32, profile: u8) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bits(2, 2); // frame_marker
        w.write_bit((profile & 1) as u32); // profile_low_bit
        w.write_bit(((profile >> 1) & 1) as u32); // profile_high_bit
        if profile == 3 {
            w.write_bit(0); // reserved_zero
        }
        w.write_bit(0); // show_existing_frame
        w.write_bit(0); // frame_type = KEY_FRAME
        w.write_bit(1); // show_frame
        w.write_bit(0); // error_resilient_mode
        w.write_bits(0x49, 8); // frame_sync_code
        w.write_bits(0x83, 8);
        w.write_bits(0x42, 8);
        if profile >= 2 {
            w.write_bit(0); // ten_or_twelve_bit
        }
        w.write_bits(2, 3); // color_space = CS_BT_709 (not CS_RGB)
        w.write_bit(0); // color_range
        if profile == 1 || profile == 3 {
            w.write_bit(1); // subsampling_x
            w.write_bit(1); // subsampling_y
            w.write_bit(0); // reserved_zero
        }
        w.write_bits(width - 1, 16); // frame_width_minus_1
        w.write_bits(height - 1, 16); // frame_height_minus_1
        w.align_to_byte();
        w.into_bytes()
    }

    #[test]
    fn recovers_1920x1080_profile0() {
        let info = parse_keyframe(&keyframe(1920, 1080, 0)).expect("keyframe must parse");
        assert_eq!((info.width, info.height), (1920, 1080));
        assert_eq!(info.profile, 0);
    }

    #[test]
    fn recovers_profile1_with_subsampling_bits() {
        let info = parse_keyframe(&keyframe(640, 360, 1)).expect("profile-1 keyframe must parse");
        assert_eq!((info.width, info.height), (640, 360));
        assert_eq!(info.profile, 1);
    }

    #[test]
    fn recovers_profile2_with_bit_depth_bit() {
        let info = parse_keyframe(&keyframe(1280, 720, 2)).expect("profile-2 keyframe must parse");
        assert_eq!((info.width, info.height), (1280, 720));
        assert_eq!(info.profile, 2);
    }

    #[test]
    fn rejects_inter_frame() {
        // frame_type = 1 (NON_KEY_FRAME): the parser stops before the size.
        let mut w = BitWriter::default();
        w.write_bits(2, 2);
        w.write_bit(0);
        w.write_bit(0); // profile 0
        w.write_bit(0); // show_existing_frame
        w.write_bit(1); // frame_type = NON_KEY_FRAME
        w.write_bit(1); // show_frame
        w.write_bit(0); // error_resilient_mode
        w.align_to_byte();
        assert!(parse_keyframe(&w.into_bytes()).is_none());
    }

    #[test]
    fn rejects_show_existing_frame() {
        let mut w = BitWriter::default();
        w.write_bits(2, 2);
        w.write_bit(0);
        w.write_bit(0); // profile 0
        w.write_bit(1); // show_existing_frame = 1
        w.write_bits(0, 3); // frame_to_show_map_idx
        w.align_to_byte();
        assert!(parse_keyframe(&w.into_bytes()).is_none());
    }

    #[test]
    fn rejects_bad_marker_and_sync_code() {
        // Wrong frame_marker.
        let mut bad_marker = BitWriter::default();
        bad_marker.write_bits(0, 2);
        bad_marker.align_to_byte();
        assert!(parse_keyframe(&bad_marker.into_bytes()).is_none());

        // Valid profile-0 header up to the sync code (byte-aligned at byte 1),
        // then a corrupted sync byte.
        let mut f = keyframe(640, 480, 0);
        f[1] = 0x00;
        assert!(parse_keyframe(&f).is_none());
    }

    #[test]
    fn rejects_empty_input() {
        assert!(parse_keyframe(&[]).is_none());
    }

    // -- Element-level tests (drive Vp9Parse::process directly) -------------

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

    fn vp9_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = Vp9Parse::new();
        parse.configure_pipeline(&vp9_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let frame = frame_with_bytes(0, keyframe(1920, 1080, 0));
        parse.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();

        assert_eq!(sink.packets.len(), 2, "expected CapsChanged then DataFrame");
        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::CompressedVideo { width, height, .. }) => {
                assert_eq!(*width, Dim::Fixed(1920));
                assert_eq!(*height, Dim::Fixed(1080));
            }
            other => panic!("expected CapsChanged first, got {other:?}"),
        }
        assert!(matches!(sink.packets[1], PipelinePacket::DataFrame(_)));
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn does_not_re_emit_caps_when_unchanged() {
        let mut parse = Vp9Parse::new();
        parse.configure_pipeline(&vp9_caps()).unwrap();
        let mut sink = RecordingSink::default();

        for seq in 0..3 {
            let frame = frame_with_bytes(seq, keyframe(1280, 720, 0));
            parse.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();
        }

        let caps_count =
            sink.packets.iter().filter(|p| matches!(p, PipelinePacket::CapsChanged(_))).count();
        assert_eq!(caps_count, 1, "CapsChanged fires once for identical dimensions");
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn re_emits_caps_on_resolution_change() {
        let mut parse = Vp9Parse::new();
        parse.configure_pipeline(&vp9_caps()).unwrap();
        let mut sink = RecordingSink::default();

        parse
            .process(PipelinePacket::DataFrame(frame_with_bytes(0, keyframe(1280, 720, 0))), &mut sink)
            .await
            .unwrap();
        parse
            .process(PipelinePacket::DataFrame(frame_with_bytes(1, keyframe(1920, 1080, 0))), &mut sink)
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
    async fn rejects_non_vp9_caps_in_intercept() {
        let parse = Vp9Parse::new();
        let vp8 = Caps::CompressedVideo {
            codec: VideoCodec::Vp8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(parse.intercept_caps(&vp8), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_constraint_is_identity_vp9_any() {
        let parse = Vp9Parse::new();
        let c = parse.caps_constraint_as_transform();
        match c {
            CapsConstraint::Identity(set) => {
                assert_eq!(
                    set.alternatives(),
                    &[Caps::CompressedVideo {
                        codec: VideoCodec::Vp9,
                        width: Dim::Any,
                        height: Dim::Any,
                        framerate: Rate::Any,
                    }]
                );
            }
            _ => panic!("expected Identity"),
        }
    }
}

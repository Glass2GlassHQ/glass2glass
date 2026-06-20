//! VP8 frame parser that refines source-side `Caps` from the keyframe header.
//!
//! The VP8 counterpart of `h264parse`: it reads each `DataFrame`'s 3-byte frame
//! tag (RFC 6386 §9.1) and, on a key frame, the uncompressed dimensions, emitting
//! a `CapsChanged` with `Dim::Fixed` width/height before forwarding the frame.
//! A demuxer (mkvdemux) can take geometry from the container Tracks; this lets a
//! VP8 elementary stream lacking that (RTP, raw) recover it from the bitstream.
//!
//! VP8 needs none of the Annex-B / exp-Golomb machinery the H.264 parser shares:
//! the container frames packets, so the packet *is* the frame, and the keyframe
//! header is plain byte fields (frame tag, the `9d 01 2a` start code, then two
//! 16-bit little-endian size words). Dimensions live only in key frames;
//! interframes carry none, so they forward without a caps change. VP8 has no
//! framerate in the bitstream, so refined caps report `Rate::Any` (matching
//! mkvdemux), refining only geometry.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, Rate, VideoCodec,
};

/// VP8 key-frame start code following the 3-byte frame tag (RFC 6386 §9.1).
const VP8_START_CODE: [u8; 3] = [0x9d, 0x01, 0x2a];

#[derive(Debug, Default)]
pub struct Vp8Parse {
    configured: bool,
    last_emitted_caps: Option<Caps>,
    keyframes_emitted: u64,
}

impl Vp8Parse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count of `CapsChanged` packets pushed downstream, for tests asserting
    /// re-emission is suppressed when the dimensions are unchanged.
    pub fn caps_changes_emitted(&self) -> u64 {
        self.keyframes_emitted
    }
}

impl AsyncElement for Vp8Parse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        let supported = Caps::CompressedVideo {
            codec: VideoCodec::Vp8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        upstream_caps.intersect(&supported)
    }

    /// Pass-through identity over VP8 of any geometry (the parser refines
    /// geometry mid-stream from the keyframe but never changes media type).
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(CapsSet::one(Caps::CompressedVideo {
            codec: VideoCodec::Vp8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo { codec: VideoCodec::Vp8, .. } => {
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
                                codec: VideoCodec::Vp8,
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

impl PadTemplates for Vp8Parse {
    fn pad_templates() -> Vec<PadTemplate> {
        let vp8 = Caps::CompressedVideo {
            codec: VideoCodec::Vp8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([PadTemplate::sink(CapsSet::one(vp8.clone())), PadTemplate::source(CapsSet::one(vp8))])
    }
}

/// The dimensions (and version / show flag) decoded from a VP8 key frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vp8KeyFrame {
    pub width: u32,
    pub height: u32,
    /// Bitstream version (0..=3): selects the reconstruction filter and loop
    /// filter (RFC 6386 §9.1). Not carried in `Caps`, surfaced for completeness.
    pub version: u8,
    pub show_frame: bool,
}

/// Decode a VP8 key-frame header. `None` when the packet is too short, is an
/// interframe (no dimensions), lacks the start code, or codes a zero dimension.
fn parse_keyframe(packet: &[u8]) -> Option<Vp8KeyFrame> {
    // 3-byte frame tag + 3-byte start code + two 16-bit size words.
    if packet.len() < 10 {
        return None;
    }
    // Frame tag is a little-endian 24-bit field.
    let tag = packet[0] as u32 | ((packet[1] as u32) << 8) | ((packet[2] as u32) << 16);
    let is_key_frame = tag & 0x1 == 0; // 0 = key frame, 1 = interframe
    if !is_key_frame {
        return None;
    }
    let version = ((tag >> 1) & 0x7) as u8;
    let show_frame = (tag >> 4) & 0x1 == 1;

    if packet[3..6] != VP8_START_CODE {
        return None;
    }
    // 14-bit width / height; the top 2 bits of each size word are the up-scaling
    // factor, ignored here (the coded size is what downstream allocates).
    let width = (packet[6] as u32 | ((packet[7] as u32) << 8)) & 0x3FFF;
    let height = (packet[8] as u32 | ((packet[9] as u32) << 8)) & 0x3FFF;
    if width == 0 || height == 0 {
        return None;
    }
    Some(Vp8KeyFrame { width, height, version, show_frame })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build a VP8 key-frame header for `width` x `height` at `version`, plus a
    /// short dummy payload the parser ignores.
    fn keyframe(width: u32, height: u32, version: u8, show: bool) -> Vec<u8> {
        // key_frame bit = 0; version in bits 1..=3; show in bit 4; first partition
        // size in bits 5..=23 (left 0).
        let tag = ((version as u32 & 0x7) << 1) | ((show as u32) << 4);
        let mut v = vec![
            (tag & 0xFF) as u8,
            ((tag >> 8) & 0xFF) as u8,
            ((tag >> 16) & 0xFF) as u8,
            VP8_START_CODE[0],
            VP8_START_CODE[1],
            VP8_START_CODE[2],
            (width & 0xFF) as u8,
            ((width >> 8) & 0x3F) as u8,
            (height & 0xFF) as u8,
            ((height >> 8) & 0x3F) as u8,
        ];
        v.extend_from_slice(&[0u8; 4]); // partition payload (ignored)
        v
    }

    /// An interframe: the same shape but with the key_frame bit set.
    fn interframe() -> Vec<u8> {
        let mut f = keyframe(640, 480, 0, true);
        f[0] |= 0x1;
        f
    }

    #[test]
    fn recovers_1280x720_keyframe() {
        let info = parse_keyframe(&keyframe(1280, 720, 0, true)).expect("keyframe must parse");
        assert_eq!((info.width, info.height), (1280, 720));
        assert_eq!(info.version, 0);
        assert!(info.show_frame);
    }

    #[test]
    fn decodes_version_and_show_flag() {
        let info = parse_keyframe(&keyframe(320, 240, 3, false)).expect("keyframe must parse");
        assert_eq!(info.version, 3);
        assert!(!info.show_frame);
    }

    #[test]
    fn rejects_interframe() {
        assert!(parse_keyframe(&interframe()).is_none(), "interframes carry no dimensions");
    }

    #[test]
    fn rejects_bad_start_code() {
        let mut f = keyframe(640, 480, 0, true);
        f[4] = 0x00; // corrupt the 9d 01 2a start code
        assert!(parse_keyframe(&f).is_none());
    }

    #[test]
    fn rejects_short_and_zero_dimensions() {
        assert!(parse_keyframe(&[]).is_none());
        assert!(parse_keyframe(&[0u8; 9]).is_none(), "needs 10 bytes");
        assert!(parse_keyframe(&keyframe(0, 480, 0, true)).is_none());
    }

    // -- Element-level tests (drive Vp8Parse::process directly) -------------

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

    fn vp8_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Vp8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = Vp8Parse::new();
        parse.configure_pipeline(&vp8_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let frame = frame_with_bytes(0, keyframe(1280, 720, 0, true));
        parse.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();

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
    async fn interframe_between_keyframes_does_not_re_emit() {
        let mut parse = Vp8Parse::new();
        parse.configure_pipeline(&vp8_caps()).unwrap();
        let mut sink = RecordingSink::default();

        parse
            .process(PipelinePacket::DataFrame(frame_with_bytes(0, keyframe(640, 480, 0, true))), &mut sink)
            .await
            .unwrap();
        parse
            .process(PipelinePacket::DataFrame(frame_with_bytes(1, interframe())), &mut sink)
            .await
            .unwrap();

        let caps_count =
            sink.packets.iter().filter(|p| matches!(p, PipelinePacket::CapsChanged(_))).count();
        assert_eq!(caps_count, 1, "the interframe carries no dimensions, so no second CapsChanged");
        // both frames still forwarded
        let data_count =
            sink.packets.iter().filter(|p| matches!(p, PipelinePacket::DataFrame(_))).count();
        assert_eq!(data_count, 2);
    }

    #[tokio::test]
    async fn re_emits_caps_on_resolution_change() {
        let mut parse = Vp8Parse::new();
        parse.configure_pipeline(&vp8_caps()).unwrap();
        let mut sink = RecordingSink::default();

        parse
            .process(PipelinePacket::DataFrame(frame_with_bytes(0, keyframe(640, 480, 0, true))), &mut sink)
            .await
            .unwrap();
        parse
            .process(PipelinePacket::DataFrame(frame_with_bytes(1, keyframe(1280, 720, 0, true))), &mut sink)
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
        assert_eq!(widths, vec![Dim::Fixed(640), Dim::Fixed(1280)]);
        assert_eq!(parse.caps_changes_emitted(), 2);
    }

    #[tokio::test]
    async fn rejects_non_vp8_caps_in_intercept() {
        let parse = Vp8Parse::new();
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(parse.intercept_caps(&vp9), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_constraint_is_identity_vp8_any() {
        let parse = Vp8Parse::new();
        let c = parse.caps_constraint_as_transform();
        match c {
            CapsConstraint::Identity(set) => {
                assert_eq!(
                    set.alternatives(),
                    &[Caps::CompressedVideo {
                        codec: VideoCodec::Vp8,
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

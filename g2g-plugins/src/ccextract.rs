//! Closed-caption extraction element (M429, `no_std`): tap a compressed
//! H.264 / H.265 video stream, mine the in-band closed-caption `cc_data` triples
//! from each access unit's SEI, decode the selected service, and emit timed
//! `Caps::Text{Utf8}` cues, one frame per cue (the same shape `crate::subparse`
//! emits, so a `TextOverlay` / text sink consumes either interchangeably).
//!
//! Closed captions ride *inside* the compressed bitstream rather than in a
//! container text track (see [`crate::cea`]), so this element sits on the
//! compressed-video link, in parallel with the decoder: tee the parser output,
//! send one branch to the decoder and the other to `CcExtract`. It consumes the
//! access units (it is a branch leaf, not a pass-through), runs
//! [`extract_cc_data`](crate::cea::extract_cc_data) on each, and drives the
//! CEA-608 or CEA-708 state machine selected at construction
//! ([`CcSource`]; default CEA-608 CC1).

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, TextFormat,
    VideoCodec,
};

pub use crate::cea::CcSource;
use crate::cea::{CaptionDecoder, Cea608Channel};
use crate::subparse::Cue;
#[cfg(feature = "metadata")]
use crate::subparse::TextCueMeta;

use core::future::Future;
use core::pin::Pin;

/// Closed-caption extraction element: a compressed `H.264` / `H.265` video stream
/// in, timed `Caps::Text{Utf8}` cue frames out. See the module docs for the
/// teed-branch topology. The service selection ([`CcSource`]) and decode are the
/// shared [`CaptionDecoder`].
#[derive(Debug)]
pub struct CcExtract {
    decoder: CaptionDecoder,
    /// The input codec, fixed at `configure_pipeline`; selects the SEI framing.
    codec: Option<VideoCodec>,
    /// Whether the output `Caps::Text{Utf8}` has been announced downstream.
    caps_emitted: bool,
    /// The most recent access unit's PTS, the end time used to flush a still-shown
    /// caption at `Eos`.
    last_pts: u64,
    sequence: u64,
}

impl Default for CcExtract {
    fn default() -> Self {
        Self::new()
    }
}

impl CcExtract {
    /// A fresh extractor rendering the default source (CEA-608 CC1).
    pub fn new() -> Self {
        Self::for_source(CcSource::default())
    }

    /// A fresh extractor rendering CEA-608 `channel` (CC1..CC4).
    pub fn cea608(channel: Cea608Channel) -> Self {
        Self::for_source(CcSource::Cea608(channel))
    }

    /// A fresh extractor rendering CEA-708 `service` (1 = primary).
    pub fn cea708(service: u8) -> Self {
        Self::for_source(CcSource::Cea708(service))
    }

    /// A fresh extractor rendering `source`.
    pub fn for_source(source: CcSource) -> Self {
        Self {
            decoder: CaptionDecoder::new(source),
            codec: None,
            caps_emitted: false,
            last_pts: 0,
            sequence: 0,
        }
    }

    /// The compressed-video codecs whose SEI this element mines (its sink pad).
    fn input_alternatives() -> CapsSet {
        CapsSet::from_alternatives(Vec::from([
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: g2g_core::Dim::Any,
                height: g2g_core::Dim::Any,
                framerate: g2g_core::Rate::Any,
            },
            Caps::CompressedVideo {
                codec: VideoCodec::H265,
                width: g2g_core::Dim::Any,
                height: g2g_core::Dim::Any,
                framerate: g2g_core::Rate::Any,
            },
        ]))
    }

    fn output_caps() -> Caps {
        Caps::Text { format: TextFormat::Utf8 }
    }

}

/// Emit a run of finished caption cues as `Caps::Text{Utf8}` `DataFrame`s, one per
/// cue, announcing the text caps before the first frame (tracked by
/// `caps_emitted`) and stamping each with `sequence`. Shared by [`CcExtract`] and
/// the ST 2110-40 caption source, which differ only in where the cues come from.
pub(crate) async fn push_cue_frames(
    out: &mut dyn OutputSink,
    cues: Vec<Cue>,
    caps_emitted: &mut bool,
    sequence: &mut u64,
) -> Result<(), G2gError> {
    for cue in cues {
        if !*caps_emitted {
            out.push(PipelinePacket::CapsChanged(Caps::Text { format: TextFormat::Utf8 })).await?;
            *caps_emitted = true;
        }
        let timing = FrameTiming {
            pts_ns: cue.start_ns,
            duration_ns: cue.end_ns.saturating_sub(cue.start_ns),
            ..Default::default()
        };
        let payload = cue.text.into_bytes().into_boxed_slice();
        // Carry the cue placement (608 row / indent, 708 window anchor) as
        // frame-meta so the overlay can honour it (no-op on the ZST baseline).
        #[cfg_attr(not(feature = "metadata"), allow(unused_mut))]
        let mut frame =
            Frame::new(MemoryDomain::System(SystemSlice::from_boxed(payload)), timing, *sequence);
        #[cfg(feature = "metadata")]
        frame.meta.attach(TextCueMeta { settings: cue.settings });
        *sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
    }
    Ok(())
}

impl AsyncElement for CcExtract {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::CompressedVideo { codec: VideoCodec::H264 | VideoCodec::H265, .. } => {
                Ok(upstream_caps.clone())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Decoder-style: a compressed H.264 / H.265 stream in, plain UTF-8 text out,
    /// so the solver negotiates `Text{Utf8}` onto the downstream link while the
    /// sink pad takes the compressed video.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo { codec: VideoCodec::H264 | VideoCodec::H265, .. } => {
                CapsSet::one(Self::output_caps())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo { codec: codec @ (VideoCodec::H264 | VideoCodec::H265), .. } => {
                self.codec = Some(*codec);
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Closed-caption extractor",
            "Codec/Parser/ClosedCaption",
            "Extracts CEA-608 / CEA-708 captions from H.264 / H.265 SEI into timed UTF-8 text cues",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if self.codec.is_none() {
                return Err(G2gError::NotConfigured);
            }
            let cues = match packet {
                PipelinePacket::DataFrame(frame) => {
                    self.last_pts = frame.timing.pts_ns;
                    if let (MemoryDomain::System(slice), Some(codec)) = (&frame.domain, self.codec) {
                        // Drive the decoder from this access unit's bytes; any
                        // newly finished cue is drained and emitted below.
                        let au = slice.as_slice().to_vec();
                        self.decoder.push_au(&au, codec, frame.timing.pts_ns);
                    }
                    self.decoder.take_cues()
                }
                // The output caps are negotiated up front (DerivedOutput) and
                // announced at the first cue; an inbound video caps change carries
                // no caption effect.
                PipelinePacket::CapsChanged(_) => Vec::new(),
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                    Vec::new()
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                    Vec::new()
                }
                // Finalize a still-shown caption at the last access unit's PTS; the
                // runner arm forwards the trailing Eos.
                PipelinePacket::Eos => {
                    self.decoder.flush(self.last_pts);
                    self.decoder.take_cues()
                }
                other => {
                    out.push(other).await?;
                    Vec::new()
                }
            };
            push_cue_frames(out, cues, &mut self.caps_emitted, &mut self.sequence).await
        })
    }
}

impl PadTemplates for CcExtract {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(Self::input_alternatives()),
            PadTemplate::source(CapsSet::one(Self::output_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cea::CcTriple;
    use alloc::vec;
    use g2g_core::PushOutcome;

    // A recording sink that captures the emitted packets so a test can assert on
    // the cue frames the element produced.
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

    /// The text payload of each emitted `DataFrame`, in order.
    fn cue_texts(sink: &RecordingSink) -> Vec<(u64, u64, alloc::string::String)> {
        sink.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => {
                    if let MemoryDomain::System(s) = &f.domain {
                        let text = alloc::string::String::from_utf8_lossy(s.as_slice()).into_owned();
                        Some((f.timing.pts_ns, f.timing.duration_ns, text))
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect()
    }

    /// Build an H.264 access unit: a single SEI NAL carrying a `GA94` `cc_data`
    /// block of `triples`, Annex-B framed. Mirrors the `cea` test fixture.
    fn h264_au_with_triples(triples: &[CcTriple]) -> Vec<u8> {
        // ATSC1 user-data: GA94 user-identifier, type 0x03 (cc_data).
        let mut payload = vec![0xB5, 0x00, 0x31]; // itu-t-t35 country + provider (GA94 below)
        payload.extend_from_slice(&[0x47, 0x41, 0x39, 0x34]); // "GA94"
        payload.push(0x03); // user_data_type_code = cc_data
        // cc_data header: process_cc_data_flag(0x40) | cc_count, then em_data 0xFF.
        let cc_count = triples.len() as u8;
        payload.push(0x40 | (cc_count & 0x1F));
        payload.push(0xFF);
        for t in triples {
            // marker bits 0xF8 | cc_valid(0x04) | cc_type(0..3).
            payload.push(0xF8 | 0x04 | (t.cc_type & 0x03));
            payload.push(t.b0);
            payload.push(t.b1);
        }
        payload.push(0xFF); // marker

        // SEI message: payloadType 4 (user_data_registered_itu_t_t35), size, payload.
        let mut sei = vec![0x04, payload.len() as u8];
        sei.extend_from_slice(&payload);
        sei.push(0x80); // rbsp_trailing

        // SEI NAL (type 6) + Annex-B start code.
        let mut au = vec![0x00, 0x00, 0x00, 0x01, 0x06];
        au.extend_from_slice(&sei);
        au
    }

    fn h264_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: g2g_core::Dim::Any,
            height: g2g_core::Dim::Any,
            framerate: g2g_core::Rate::Any,
        }
    }

    fn data_frame(au: Vec<u8>, pts: u64) -> PipelinePacket {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            FrameTiming { pts_ns: pts, ..Default::default() },
            0,
        );
        PipelinePacket::DataFrame(frame)
    }

    #[test]
    fn negotiates_video_in_text_out() {
        let el = CcExtract::new();
        assert_eq!(el.intercept_caps(&h264_caps()).unwrap(), h264_caps());
        assert!(el
            .intercept_caps(&Caps::Text { format: TextFormat::Utf8 })
            .is_err());
        let derived = match el.caps_constraint_as_transform() {
            CapsConstraint::DerivedOutput(f) => f(&h264_caps()),
            _ => panic!("expected DerivedOutput"),
        };
        assert_eq!(derived.alternatives(), &[CcExtract::output_caps()]);
    }

    #[tokio::test]
    async fn extracts_a_608_pop_on_caption() {
        // A CEA-608 CC1 pop-on caption (RCL, write "HI", EOC to display) then EDM
        // to erase, fed as field-1 (`cc_type` 0) triples over two access units.
        let mut el = CcExtract::new();
        el.configure_pipeline(&h264_caps()).expect("accepts H.264");
        let mut sink = RecordingSink::default();

        // RCL (0x14 0x20) resume caption loading; 'H','I'; EOC (0x14 0x2F) display.
        let au1 = h264_au_with_triples(&[
            CcTriple { cc_type: 0, b0: 0x14, b1: 0x20 },
            CcTriple { cc_type: 0, b0: b'H', b1: b'I' },
            CcTriple { cc_type: 0, b0: 0x14, b1: 0x2F },
        ]);
        el.process(data_frame(au1, 1_000), &mut sink).await.unwrap();
        // EDM (0x14 0x2C) erase displayed memory ends the caption.
        let au2 = h264_au_with_triples(&[CcTriple { cc_type: 0, b0: 0x14, b1: 0x2C }]);
        el.process(data_frame(au2, 5_000), &mut sink).await.unwrap();

        let cues = cue_texts(&sink);
        assert_eq!(cues.len(), 1, "one finished caption");
        assert_eq!(cues[0].2, "HI");
        assert_eq!(cues[0].0, 1_000); // started at the EOC access unit
        assert_eq!(cues[0].0 + cues[0].1, 5_000); // ended at the EDM access unit
        // The output caps are announced before the first cue frame.
        assert!(matches!(
            sink.packets.first(),
            Some(PipelinePacket::CapsChanged(Caps::Text { format: TextFormat::Utf8 }))
        ));
    }

    #[tokio::test]
    async fn renders_only_the_selected_708_service() {
        // The element selects CEA-708 service 1; a field-1 608 triple in the same
        // stream must not produce a cue.
        let mut el = CcExtract::cea708(1);
        el.configure_pipeline(&h264_caps()).expect("accepts H.264");
        let mut sink = RecordingSink::default();
        // A lone 608 pair (no 708 packet) yields nothing from the 708 decoder.
        let au = h264_au_with_triples(&[CcTriple { cc_type: 0, b0: b'H', b1: b'I' }]);
        el.process(data_frame(au, 1_000), &mut sink).await.unwrap();
        el.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        assert!(cue_texts(&sink).is_empty());
    }

    #[tokio::test]
    async fn requires_configure() {
        let mut el = CcExtract::new();
        let mut sink = RecordingSink::default();
        let au = h264_au_with_triples(&[CcTriple { cc_type: 0, b0: b'H', b1: b'I' }]);
        // Processing before configure_pipeline is a NotConfigured error.
        assert!(el.process(data_frame(au, 0), &mut sink).await.is_err());
    }
}

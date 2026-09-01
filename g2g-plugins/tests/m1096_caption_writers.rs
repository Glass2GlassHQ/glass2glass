//! M1096: the caption and subtitle writers. Each one is checked against the
//! reader that already exists for its format, so the assertion is a round trip
//! rather than a restatement of the writer's own output:
//!
//! - `srtenc` / `webvttenc` write documents that [`SubParse`]'s parsers read back
//!   into the cue text and window they started as.
//! - `ccconverter` re-lays caption triples between the four transport layouts;
//!   the captions come out of `CcExtract` as the text the CEA-608 encoder put in.
//! - `cccombiner` attaches a caption stream to the video frames it belongs with,
//!   and `CcInsert::from_meta` + `CcExtract` read those captions back out (the
//!   caption meta needs the `metadata` feature, so those tests are gated on it).

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    Caps, ClosedCaptionFormat, G2gError, OutputSink, PropValue, PushOutcome, TextFormat,
};
use g2g_plugins::ccconverter::CcConverter;
use g2g_plugins::ccextract::CcExtract;
use g2g_plugins::cea::{write_cc_data, Cc608Enc, CcTriple, CEA608_FIELD_1};
use g2g_plugins::srtenc::SrtEnc;
use g2g_plugins::subparse::{parse_srt, parse_webvtt, Cue, CueSettings, WEBVTT_HEADER};
use g2g_plugins::webvttenc::WebVttEnc;

/// One caption byte pair per video frame, the line-21 cadence, at 30 fps.
const FRAME_NS: u64 = 33_333_333;
/// Frames of caption bytes the encoder is drained for: enough to carry the
/// pop-on load, the text and the display command.
const CAPTION_FRAMES: usize = 40;

/// The cues every subtitle round trip starts from. The expected text and window
/// on the far side are read back out of this table, never restated.
const SUBTITLE_CUES: &[(&str, u64, u64)] = &[
    ("HELLO", 1_000_000_000, 2_000_000_000),
    ("SECOND LINE\nAND ANOTHER", 4_500_000_000, 1_250_000_000),
];

#[derive(Default)]
struct RecordingSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for RecordingSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl RecordingSink {
    /// The emitted payloads concatenated as text, the document a `filesink`
    /// downstream would have written.
    fn document(&self) -> String {
        String::from_utf8_lossy(&self.payloads().concat()).into_owned()
    }

    fn payloads(&self) -> Vec<Vec<u8>> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(|s| s.to_vec()),
                _ => None,
            })
            .collect()
    }

    /// Take the emitted frames out, so the next element in the chain can be fed
    /// them (a `Frame` owns its buffer and does not clone).
    #[cfg(feature = "metadata")]
    fn take_frames(&mut self) -> Vec<Frame> {
        core::mem::take(&mut self.packets)
            .into_iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }
}

fn utf8_caps() -> Caps {
    Caps::Text {
        format: TextFormat::Utf8,
    }
}

fn data_frame(payload: Vec<u8>, pts_ns: u64, duration_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            duration_ns,
            ..Default::default()
        },
        0,
    ))
}

fn cue_frame(text: &str, pts_ns: u64, duration_ns: u64) -> PipelinePacket {
    data_frame(text.as_bytes().to_vec(), pts_ns, duration_ns)
}

/// Run every cue of [`SUBTITLE_CUES`] through `encoder` and return the document
/// it wrote.
async fn write_subtitle_document<E: AsyncElement>(encoder: &mut E) -> String {
    let mut sink = RecordingSink::default();
    for (text, pts, duration) in SUBTITLE_CUES {
        encoder
            .process(cue_frame(text, *pts, *duration), &mut sink)
            .await
            .expect("the cue encodes");
    }
    encoder
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("end of stream closes the document");
    sink.document()
}

/// Every cue survived the write / read round trip with its text and window.
fn assert_cues_match_the_fixture(cues: &[Cue]) {
    assert_eq!(cues.len(), SUBTITLE_CUES.len(), "one cue per fixture entry");
    for (cue, (text, pts, duration)) in cues.iter().zip(SUBTITLE_CUES) {
        assert_eq!(cue.text, *text);
        assert_eq!(cue.start_ns, *pts);
        assert_eq!(cue.end_ns, pts + duration);
    }
}

#[tokio::test]
async fn srtenc_writes_a_document_the_subrip_parser_reads_back() {
    let mut encoder = SrtEnc::new();
    encoder
        .configure_pipeline(&utf8_caps())
        .expect("takes UTF-8 cues");
    let document = write_subtitle_document(&mut encoder).await;
    assert_cues_match_the_fixture(&parse_srt(&document));

    // SubRip numbers its cues from 1, which the parser skips over, so it is
    // asserted here rather than through the round trip.
    let numbers: Vec<&str> = document
        .split("\n\n")
        .filter(|block| !block.trim().is_empty())
        .map(|block| block.lines().next().expect("a block has a first line"))
        .collect();
    assert_eq!(numbers, ["1", "2"]);
}

#[tokio::test]
async fn webvttenc_writes_a_document_the_webvtt_parser_reads_back() {
    let mut encoder = WebVttEnc::new();
    encoder
        .configure_pipeline(&utf8_caps())
        .expect("takes UTF-8 cues");
    let document = write_subtitle_document(&mut encoder).await;
    assert!(
        document.starts_with(WEBVTT_HEADER),
        "a .vtt file opens with its signature: {document:?}"
    );
    assert_cues_match_the_fixture(&parse_webvtt(&document));
}

#[tokio::test]
async fn a_cue_free_stream_still_writes_the_webvtt_signature() {
    let mut encoder = WebVttEnc::new();
    encoder
        .configure_pipeline(&utf8_caps())
        .expect("takes UTF-8 cues");
    let mut sink = RecordingSink::default();
    encoder
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("end of stream closes the document");
    assert_eq!(sink.document().trim(), WEBVTT_HEADER);
}

#[tokio::test]
async fn a_blank_cue_writes_no_block() {
    let mut encoder = SrtEnc::new();
    encoder
        .configure_pipeline(&utf8_caps())
        .expect("takes UTF-8 cues");
    let mut sink = RecordingSink::default();
    encoder
        .process(cue_frame("  \n  ", 0, 1_000_000_000), &mut sink)
        .await
        .expect("the blank cue is skipped");
    assert!(sink.payloads().is_empty());
}

#[tokio::test]
async fn the_offset_properties_shift_the_written_window() {
    const START_SHIFT_NS: i64 = -500_000_000;
    const DURATION_SHIFT_NS: i64 = 250_000_000;
    let mut encoder = SrtEnc::new();
    encoder
        .set_property("timestamp", PropValue::Int(START_SHIFT_NS))
        .expect("timestamp is settable");
    encoder
        .set_property("duration", PropValue::Int(DURATION_SHIFT_NS))
        .expect("duration is settable");
    encoder
        .configure_pipeline(&utf8_caps())
        .expect("takes UTF-8 cues");
    let document = write_subtitle_document(&mut encoder).await;

    let cues = parse_srt(&document);
    assert_eq!(cues.len(), SUBTITLE_CUES.len());
    for (cue, (_, pts, duration)) in cues.iter().zip(SUBTITLE_CUES) {
        let start = (*pts as i64 + START_SHIFT_NS) as u64;
        assert_eq!(cue.start_ns, start);
        assert_eq!(cue.end_ns, start + duration + DURATION_SHIFT_NS as u64);
    }
}

/// The caption text the CEA-608 encoder is given, and the window it is shown for.
const CAPTION_CUE_TEXT: &str = "HELLO";

/// Pace one cue out of the repo's own CEA-608 encoder as packed `cc_data`
/// payloads, one per video frame (the line-21 cadence), erasing the caption part
/// way so the decoder sees a finished cue.
fn cea608_cc_data_payloads() -> Vec<Vec<u8>> {
    let mut encoder = Cc608Enc::new();
    encoder.push_cue(&Cue {
        start_ns: 0,
        end_ns: (CAPTION_FRAMES as u64) * FRAME_NS,
        text: String::from(CAPTION_CUE_TEXT),
        settings: CueSettings::default(),
    });
    let mut payloads = Vec::with_capacity(CAPTION_FRAMES);
    for frame in 0..CAPTION_FRAMES {
        // Erase once the caption has been displayed long enough that the pop-on
        // sequence has drained, so the cue closes inside the run.
        if frame == CAPTION_FRAMES - 1 {
            encoder.erase();
        }
        let (b0, b1) = encoder.next_pair();
        payloads.push(write_cc_data(&[CcTriple {
            cc_type: CEA608_FIELD_1,
            b0,
            b1,
        }]));
    }
    payloads
}

/// Decode a run of packed `cc_data` payloads back to caption text with
/// `CcExtract`, the pipeline reader for a raw caption track.
async fn extract_caption_text(payloads: &[Vec<u8>]) -> Vec<String> {
    let mut extractor = CcExtract::new();
    extractor
        .configure_pipeline(&Caps::ClosedCaption {
            format: ClosedCaptionFormat::Cea608,
        })
        .expect("takes a raw caption track");
    let mut sink = RecordingSink::default();
    for (index, payload) in payloads.iter().enumerate() {
        extractor
            .process(
                data_frame(payload.clone(), index as u64 * FRAME_NS, 0),
                &mut sink,
            )
            .await
            .expect("the caption payload decodes");
    }
    extractor
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("end of stream flushes the decoder");
    sink.payloads()
        .iter()
        .map(|p| String::from_utf8_lossy(p).into_owned())
        .collect()
}

/// Push every payload through `converter` and collect what it wrote.
async fn convert_payloads(converter: &mut CcConverter, payloads: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut sink = RecordingSink::default();
    for (index, payload) in payloads.iter().enumerate() {
        converter
            .process(
                data_frame(payload.clone(), index as u64 * FRAME_NS, 0),
                &mut sink,
            )
            .await
            .expect("the caption payload converts");
    }
    sink.payloads()
}

/// A converter reading `from` and writing `to`, configured and ready.
fn converter(from: ClosedCaptionFormat, to: ClosedCaptionFormat) -> CcConverter {
    let mut converter = CcConverter::between(from, to);
    converter
        .configure_pipeline(&Caps::ClosedCaption { format: from })
        .expect("takes its input layout");
    converter
}

#[tokio::test]
async fn every_transport_layout_carries_the_same_captions_back() {
    let cc_data = cea608_cc_data_payloads();
    let expected = extract_caption_text(&cc_data).await;
    assert_eq!(
        expected,
        [String::from(CAPTION_CUE_TEXT)],
        "the packed cc_data reference decodes to the encoded cue"
    );

    // Each layout out and back: the captions that survive must decode the same.
    for layout in [
        ClosedCaptionFormat::Cea708Cdp,
        ClosedCaptionFormat::Cea608S334,
        ClosedCaptionFormat::Cea608Raw,
    ] {
        let mut out = converter(ClosedCaptionFormat::Cea708, layout);
        let carried = convert_payloads(&mut out, &cc_data).await;
        assert_eq!(
            carried.len(),
            cc_data.len(),
            "{layout:?} carries every payload"
        );
        assert_ne!(
            carried, cc_data,
            "{layout:?} is a different byte layout, not a passthrough"
        );

        let mut back = converter(layout, ClosedCaptionFormat::Cea708);
        let recovered = convert_payloads(&mut back, &carried).await;
        assert_eq!(
            extract_caption_text(&recovered).await,
            expected,
            "{layout:?} round trip keeps the caption"
        );
    }
}

#[tokio::test]
async fn a_layout_that_cannot_carry_a_triple_drops_it() {
    // Field 2 pairs cannot ride a field-1 `raw` stream, and DTVCC cannot ride
    // ST 334-1 at all.
    let mixed = write_cc_data(&[
        CcTriple {
            cc_type: 1,
            b0: 0x41,
            b1: 0x42,
        },
        CcTriple {
            cc_type: 3,
            b0: 0x43,
            b1: 0x44,
        },
    ]);
    let mut to_raw = converter(ClosedCaptionFormat::Cea708, ClosedCaptionFormat::Cea608Raw);
    assert!(
        convert_payloads(&mut to_raw, core::slice::from_ref(&mixed))
            .await
            .is_empty(),
        "neither triple belongs to field 1"
    );

    let mut to_s334 = converter(ClosedCaptionFormat::Cea708, ClosedCaptionFormat::Cea608S334);
    let s334 = convert_payloads(&mut to_s334, &[mixed]).await;
    let mut back = converter(ClosedCaptionFormat::Cea608S334, ClosedCaptionFormat::Cea708);
    let recovered = convert_payloads(&mut back, &s334).await;
    assert_eq!(
        recovered,
        [write_cc_data(&[CcTriple {
            cc_type: 1,
            b0: 0x41,
            b1: 0x42,
        }])],
        "the line-21 triple survives and the DTVCC one is dropped"
    );
}

#[cfg(feature = "metadata")]
mod combiner {
    use super::*;
    use g2g_core::meta::CaptionMeta;
    use g2g_core::{MultiInputElement, VideoCodec};
    use g2g_plugins::cccombiner::CcCombiner;
    use g2g_plugins::ccinsert::CcInsert;

    /// A minimal Annex-B access unit: one VCL IDR slice NAL, carrying no captions
    /// of its own.
    fn access_unit() -> Vec<u8> {
        Vec::from([0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00])
    }

    fn h264_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: g2g_core::Dim::Any,
            height: g2g_core::Dim::Any,
            framerate: g2g_core::Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    fn combiner(caption_format: ClosedCaptionFormat) -> CcCombiner {
        let mut combiner = CcCombiner::new();
        combiner
            .configure_pipeline(CcCombiner::VIDEO, &h264_caps())
            .expect("takes the video pad's caps");
        combiner
            .configure_pipeline(
                CcCombiner::CAPTION,
                &Caps::ClosedCaption {
                    format: caption_format,
                },
            )
            .expect("takes the caption pad's caps");
        combiner
    }

    /// Feed the caption payload for each frame in, then the frame, the order a
    /// PTS-ordered merge delivers them, and take the combined frames out.
    async fn combine(combiner: &mut CcCombiner, payloads: &[Vec<u8>]) -> Vec<Frame> {
        let mut sink = RecordingSink::default();
        for (index, payload) in payloads.iter().enumerate() {
            let pts = index as u64 * FRAME_NS;
            combiner
                .process(
                    CcCombiner::CAPTION,
                    data_frame(payload.clone(), pts, 0),
                    &mut sink,
                )
                .await
                .expect("the caption payload queues");
            combiner
                .process(
                    CcCombiner::VIDEO,
                    data_frame(access_unit(), pts, 0),
                    &mut sink,
                )
                .await
                .expect("the video frame passes through");
        }
        sink.take_frames()
    }

    /// The caption triples a frame is carrying.
    fn attached_triples(frame: &Frame) -> Vec<CcTriple> {
        frame
            .meta
            .get::<CaptionMeta>()
            .map(|m| m.iter().map(|t| CcTriple::from(*t)).collect())
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn the_captions_reach_the_video_frames_they_belong_with() {
        let payloads = cea608_cc_data_payloads();
        let mut element = combiner(ClosedCaptionFormat::Cea608);
        let combined = combine(&mut element, &payloads).await;

        assert_eq!(combined.len(), payloads.len(), "every frame is forwarded");
        // Each frame carries exactly the triples of the payload that arrived for
        // it, so the caption pacing survives the combine.
        for (frame, payload) in combined.iter().zip(&payloads) {
            assert_eq!(
                attached_triples(frame),
                g2g_plugins::cea::parse_cc_data(payload),
                "the frame carries its own payload's triples"
            );
        }
    }

    #[tokio::test]
    async fn a_combined_stream_re_encodes_to_captions_ccextract_reads() {
        // cccombiner puts the captions on the access units, CcInsert::from_meta
        // writes them into the SEI, and CcExtract decodes the text back out.
        let payloads = cea608_cc_data_payloads();
        let mut element = combiner(ClosedCaptionFormat::Cea608);
        let combined = combine(&mut element, &payloads).await;

        let mut inserter = CcInsert::from_meta();
        inserter
            .configure_pipeline(CcInsert::VIDEO, &h264_caps())
            .expect("takes H.264 access units");
        let mut inserted = RecordingSink::default();
        for frame in combined {
            inserter
                .process(
                    CcInsert::VIDEO,
                    PipelinePacket::DataFrame(frame),
                    &mut inserted,
                )
                .await
                .expect("the access unit takes the caption SEI");
        }

        let mut extractor = CcExtract::new();
        extractor
            .configure_pipeline(&h264_caps())
            .expect("takes H.264 access units");
        let mut text = RecordingSink::default();
        for frame in inserted.take_frames() {
            extractor
                .process(PipelinePacket::DataFrame(frame), &mut text)
                .await
                .expect("the access unit decodes");
        }
        extractor
            .process(PipelinePacket::Eos, &mut text)
            .await
            .expect("end of stream flushes the decoder");

        let cues: Vec<String> = text
            .payloads()
            .iter()
            .map(|p| String::from_utf8_lossy(p).into_owned())
            .collect();
        assert_eq!(cues, [String::from(CAPTION_CUE_TEXT)]);
    }

    #[tokio::test]
    async fn input_meta_processing_picks_which_captions_survive() {
        let combined_payload = write_cc_data(&[CcTriple {
            cc_type: CEA608_FIELD_1,
            b0: 0x41,
            b1: 0x42,
        }]);
        let existing = CcTriple {
            cc_type: CEA608_FIELD_1,
            b0: 0x43,
            b1: 0x44,
        };
        let queued = g2g_plugins::cea::parse_cc_data(&combined_payload);

        for (nick, expected) in [
            ("append", Vec::from([existing, queued[0]])),
            ("drop", Vec::from([queued[0]])),
            ("favor", Vec::from([existing])),
            ("force", Vec::from([existing])),
        ] {
            let mut element = combiner(ClosedCaptionFormat::Cea608);
            element
                .set_property("input-meta-processing", PropValue::Str(nick.into()))
                .expect("input-meta-processing is settable");
            let mut sink = RecordingSink::default();
            element
                .process(
                    CcCombiner::CAPTION,
                    data_frame(combined_payload.clone(), 0, 0),
                    &mut sink,
                )
                .await
                .expect("the caption payload queues");

            let mut frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(access_unit().into_boxed_slice())),
                FrameTiming::default(),
                0,
            );
            let mut meta = CaptionMeta::new();
            meta.push(existing.into());
            frame.meta.attach(meta);
            element
                .process(
                    CcCombiner::VIDEO,
                    PipelinePacket::DataFrame(frame),
                    &mut sink,
                )
                .await
                .expect("the video frame passes through");

            let combined = sink.take_frames();
            assert_eq!(
                attached_triples(&combined[0]),
                expected,
                "input-meta-processing={nick}"
            );
        }
    }

    #[tokio::test]
    async fn dropping_the_frames_own_captions_needs_nothing_queued() {
        // `drop` discards what the frame arrived carrying, even when the caption
        // pad has sent nothing to put in its place.
        let mut element = combiner(ClosedCaptionFormat::Cea608);
        element
            .set_property("input-meta-processing", PropValue::Str("drop".into()))
            .expect("input-meta-processing is settable");
        let mut frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(access_unit().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        let mut meta = CaptionMeta::new();
        meta.push(
            CcTriple {
                cc_type: CEA608_FIELD_1,
                b0: 0x43,
                b1: 0x44,
            }
            .into(),
        );
        frame.meta.attach(meta);

        let mut sink = RecordingSink::default();
        element
            .process(
                CcCombiner::VIDEO,
                PipelinePacket::DataFrame(frame),
                &mut sink,
            )
            .await
            .expect("the video frame passes through");
        assert!(attached_triples(&sink.take_frames()[0]).is_empty());
    }

    #[tokio::test]
    async fn max_scheduled_drops_the_oldest_queued_payload() {
        const KEPT: u64 = 2;
        let mut element = combiner(ClosedCaptionFormat::Cea608);
        element
            .set_property("max-scheduled", PropValue::Uint(KEPT))
            .expect("max-scheduled is settable");
        let payloads: Vec<Vec<u8>> = (0..4u8)
            .map(|n| {
                write_cc_data(&[CcTriple {
                    cc_type: CEA608_FIELD_1,
                    b0: 0x41 + n,
                    b1: 0x61 + n,
                }])
            })
            .collect();
        let mut sink = RecordingSink::default();
        for payload in &payloads {
            element
                .process(
                    CcCombiner::CAPTION,
                    data_frame(payload.clone(), 0, 0),
                    &mut sink,
                )
                .await
                .expect("the caption payload queues");
        }
        element
            .process(
                CcCombiner::VIDEO,
                data_frame(access_unit(), 0, 0),
                &mut sink,
            )
            .await
            .expect("the video frame passes through");

        let combined = sink.take_frames();
        let kept: Vec<CcTriple> = payloads[payloads.len() - KEPT as usize..]
            .iter()
            .flat_map(|p| g2g_plugins::cea::parse_cc_data(p))
            .collect();
        assert_eq!(
            attached_triples(&combined[0]),
            kept,
            "only the newest payloads survived the cap"
        );
    }
}

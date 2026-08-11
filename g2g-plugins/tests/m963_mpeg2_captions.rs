//! M963: ATSC A/53 closed captions carried in MPEG-2 picture user data. The
//! `cc_data` structure is the one the H.264 / H.265 SEI path already parses; only
//! the wrapper differs (a bare `GA94` identifier after the `user_data` start code,
//! no ITU-T T.35 prefix). Unit under test = the MPEG-2 user-data scan +
//! `cea::extract_cc_data` + `CcExtract` on a `VideoCodec::Mpeg2` link, driven from
//! synthetic access units composed from the real byte layout.
//!
//! The malformed cases are the point: every count and offset in a user-data block
//! is attacker-controlled, so a truncated block, an oversized `cc_count`, a foreign
//! identifier, or an empty block must yield no captions instead of panicking.

#![cfg(feature = "std")]

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    Caps, CapsConstraint, Dim, G2gError, OutputSink, PushOutcome, Rate, TextFormat, VideoCodec,
};
use g2g_plugins::ccextract::CcExtract;
use g2g_plugins::cea::{extract_cc_data, write_cc_data, CcTriple, Cea608};

/// MPEG start code introducing a `user_data` element (ISO 13818-2 6.2.2.2).
const USER_DATA_START_CODE: [u8; 4] = [0x00, 0x00, 0x01, 0xB2];
/// ATSC A/53 `ATSC_identifier`, the first field of a captioned user-data block.
const ATSC_IDENTIFIER: &[u8; 4] = b"GA94";
/// `user_data_type_code` selecting the `cc_data` structure.
const USER_DATA_TYPE_CC: u8 = 0x03;
/// Flags byte: reserved '1' | `process_cc_data_flag` | `additional_data_flag` 0.
const CC_FLAGS: u8 = 0xC0;

/// An ATSC A/53 Part 4 user-data block carrying `triples`: the `GA94` identifier,
/// the `cc_data` type code, the flags byte holding `cc_count`, `em_data`, the
/// triples, then the marker trailer.
fn ga94_block(triples: &[CcTriple]) -> Vec<u8> {
    let mut block = Vec::new();
    block.extend_from_slice(ATSC_IDENTIFIER);
    block.push(USER_DATA_TYPE_CC);
    block.push(CC_FLAGS | (triples.len() as u8 & 0x1F));
    block.push(0xFF); // em_data
    block.extend_from_slice(&write_cc_data(triples));
    block.push(0xFF); // marker_bits
    block
}

/// A synthetic MPEG-2 access unit: sequence header (720x480, 30000/1001),
/// picture header, each `user_data` block, then a slice. The blocks sit between
/// other start codes, so the scan has to walk past them and end each payload at
/// the following start code.
fn mpeg2_au(blocks: &[&[u8]]) -> Vec<u8> {
    let mut au = Vec::from([0x00, 0x00, 0x01, 0xB3, 0x2D, 0x01, 0xE0, 0x24]);
    au.extend_from_slice(&[0x00, 0x00, 0x01, 0x00, 0x00, 0x08, 0xFF]); // picture header
    for block in blocks {
        au.extend_from_slice(&USER_DATA_START_CODE);
        au.extend_from_slice(block);
    }
    au.extend_from_slice(&[0x00, 0x00, 0x01, 0x01, 0x0A, 0xBB]); // slice
    au
}

fn triple(cc_type: u8, b0: u8, b1: u8) -> CcTriple {
    CcTriple { cc_type, b0, b1 }
}

fn mpeg2_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::Mpeg2,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn data_frame(au: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

#[derive(Default)]
struct TextSink {
    cues: Vec<(u64, u64, String)>,
    caps: Vec<Caps>,
}

impl OutputSink for TextSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.cues.push((
                            f.timing.pts_ns,
                            f.timing.duration_ns,
                            String::from_utf8_lossy(s).into_owned(),
                        ));
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[test]
fn extracts_caption_triples_from_picture_user_data() {
    let triples = [
        triple(0, 0x14, 0x20),
        triple(1, b'A', b'B'),
        triple(3, 0x81, 0x02),
    ];
    let au = mpeg2_au(&[&ga94_block(&triples)]);
    assert_eq!(extract_cc_data(&au, VideoCodec::Mpeg2), triples);
}

#[test]
fn extracts_a_trailing_user_data_block_with_no_following_start_code() {
    // The block runs to the end of the access unit: there is no length field, so
    // the payload has to end at the buffer end rather than being dropped.
    let triples = [triple(0, b'H', b'I')];
    let mut au = Vec::from([0x00, 0x00, 0x01, 0x00, 0x00, 0x08, 0xFF]);
    au.extend_from_slice(&USER_DATA_START_CODE);
    au.extend_from_slice(&ga94_block(&triples));
    assert_eq!(extract_cc_data(&au, VideoCodec::Mpeg2), triples);
}

#[test]
fn accumulates_every_user_data_block_in_transmission_order() {
    let first = [triple(0, b'A', b'B')];
    let second = [triple(0, b'C', b'D')];
    // A non-caption block (a foreign identifier) between the two must be skipped
    // without disturbing the ones around it.
    let foreign = Vec::from(b"XYZW\x03\xC0\xFF".as_slice());
    let au = mpeg2_au(&[&ga94_block(&first), &foreign, &ga94_block(&second)]);
    assert_eq!(
        extract_cc_data(&au, VideoCodec::Mpeg2),
        [first[0], second[0]]
    );
}

#[test]
fn decodes_a_608_pop_on_caption_from_user_data() {
    // RCL (0x14 0x20) load the back buffer, write "HI", EOC (0x14 0x2F) display,
    // then EDM (0x14 0x2C) in a later access unit erases it and ends the cue.
    let show = mpeg2_au(&[&ga94_block(&[
        triple(0, 0x14, 0x20),
        triple(0, b'H', b'I'),
        triple(0, 0x14, 0x2F),
    ])]);
    let erase = mpeg2_au(&[&ga94_block(&[triple(0, 0x14, 0x2C)])]);

    let mut decoder = Cea608::new();
    for (au, pts) in [(show, 1_000u64), (erase, 5_000)] {
        for t in extract_cc_data(&au, VideoCodec::Mpeg2) {
            if t.cc_type == 0 {
                decoder.push_pair(t.b0, t.b1, pts);
            }
        }
    }
    let cues = decoder.take_cues();
    assert_eq!(cues.len(), 1, "one finished caption");
    assert_eq!(cues[0].text, "HI");
    assert_eq!((cues[0].start_ns, cues[0].end_ns), (1_000, 5_000));
}

#[tokio::test]
async fn ccextract_captions_an_mpeg2_link() {
    let mut el = CcExtract::new();
    assert_eq!(el.intercept_caps(&mpeg2_caps()).unwrap(), mpeg2_caps());
    let derived = match el.caps_constraint_as_transform() {
        CapsConstraint::DerivedOutput(f) => f(&mpeg2_caps()),
        _ => panic!("expected DerivedOutput"),
    };
    assert_eq!(
        derived.alternatives(),
        &[Caps::Text {
            format: TextFormat::Utf8
        }]
    );
    el.configure_pipeline(&mpeg2_caps())
        .expect("accepts MPEG-2 video");

    let mut sink = TextSink::default();
    let show = mpeg2_au(&[&ga94_block(&[
        triple(0, 0x14, 0x20),
        triple(0, b'H', b'I'),
        triple(0, 0x14, 0x2F),
    ])]);
    el.process(data_frame(show, 1_000), &mut sink)
        .await
        .unwrap();
    let erase = mpeg2_au(&[&ga94_block(&[triple(0, 0x14, 0x2C)])]);
    el.process(data_frame(erase, 5_000), &mut sink)
        .await
        .unwrap();

    assert_eq!(sink.cues.len(), 1);
    assert_eq!(sink.cues[0], (1_000, 4_000, "HI".to_string()));
    assert_eq!(
        sink.caps,
        [Caps::Text {
            format: TextFormat::Utf8
        }]
    );
}

#[test]
fn malformed_user_data_yields_no_captions() {
    let triples = [triple(0, b'H', b'I'), triple(0, b'!', b'?')];
    let good = ga94_block(&triples);

    // cc_count claims 31 triples the block does not hold.
    let mut oversized_count = good.clone();
    oversized_count[5] = CC_FLAGS | 0x1F;
    // A foreign ATSC_identifier is somebody else's user data.
    let mut wrong_identifier = good.clone();
    wrong_identifier[3] = b'3';
    // user_data_type_code other than 0x03 is not cc_data.
    let mut wrong_type_code = good.clone();
    wrong_type_code[4] = 0x04;
    // process_cc_data_flag clear: the block must not be mined.
    let mut flag_clear = good.clone();
    flag_clear[5] = 0x80 | (triples.len() as u8);
    // A start-code emulation inside the payload cuts the block short there.
    let mut emulated_start_code = good.clone();
    emulated_start_code[7..10].copy_from_slice(&[0x00, 0x00, 0x01]);

    let mut cases: Vec<(&str, Vec<u8>)> = Vec::from([
        ("empty block", Vec::new()),
        ("identifier only", Vec::from(ATSC_IDENTIFIER.as_slice())),
        ("oversized cc_count", oversized_count),
        ("wrong identifier", wrong_identifier),
        ("wrong type code", wrong_type_code),
        ("process_cc_data_flag clear", flag_clear),
        ("emulated start code", emulated_start_code),
    ]);
    // Every truncation that loses caption data: each must fail the parse rather
    // than read past the end or emit a partial triple. The last byte is only the
    // marker trailer, so dropping it alone still leaves a well-formed block.
    for cut in 0..good.len() - 1 {
        cases.push(("truncated", good[..cut].to_vec()));
    }

    for (name, block) in cases {
        let au = mpeg2_au(&[&block]);
        assert!(
            extract_cc_data(&au, VideoCodec::Mpeg2).is_empty(),
            "{name}: malformed user data must yield no captions"
        );
    }

    // An access unit truncated inside the start code, and one truncated right
    // after it, must both parse to nothing.
    for cut in 0..USER_DATA_START_CODE.len() {
        let au = &USER_DATA_START_CODE[..cut];
        assert!(extract_cc_data(au, VideoCodec::Mpeg2).is_empty());
    }
    assert!(extract_cc_data(&USER_DATA_START_CODE, VideoCodec::Mpeg2).is_empty());
}

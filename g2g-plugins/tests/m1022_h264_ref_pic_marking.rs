//! M1022: H.264 `dec_ref_pic_marking()`, the slice header's adaptive reference
//! marking.
//!
//! A stream may name which pictures stop being references instead of leaving it
//! to the default sliding window, and x264 does exactly that for its B-pyramid.
//! The marking sits behind the reference-list modification and the prediction
//! weight table, so reading it means walking those correctly: a misread bit
//! yields nonsense operations, not a parse error, which is why this asserts the
//! operations themselves against a stream whose marking is known.

use g2g_plugins::poc::{
    parse_h264_slice_marking, H264Mmco, H264PocContext, H264PpsPoc, H264PpsRefMarking,
    H264RefPicMarking,
};

/// A B-pyramid stream: x264 marks each pyramid B a reference and retires it by
/// hand, which is what makes the marking non-default.
const B_PYRAMID: &[u8] = include_bytes!("fixtures/h264_bpyramid_320x240.h264");
/// A P-only stream, which leaves marking to the sliding window.
const SLIDING_WINDOW: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

/// The fixtures' shared sequence-parameter geometry, read off their SPS by the
/// decoder's own parse; spelled here so this test needs no decoder.
struct Stream {
    log2_max_frame_num: u32,
    log2_max_pic_order_cnt_lsb: u32,
    pic_order_cnt_type: u8,
    pps: H264PpsRefMarking,
}

/// The B-pyramid fixture's own parameters: x264 High profile, CABAC, weighted P
/// prediction, implicit weighted B, 4 default L0 references. A re-encoded fixture
/// fails the parse rather than passing quietly.
fn b_pyramid_stream() -> Stream {
    Stream {
        log2_max_frame_num: 4,
        log2_max_pic_order_cnt_lsb: 6,
        pic_order_cnt_type: 0,
        pps: H264PpsRefMarking {
            redundant_pic_cnt_present_flag: false,
            weighted_pred_flag: true,
            weighted_bipred_idc: 2,
            num_ref_idx_l0_default_active_minus1: 3,
            num_ref_idx_l1_default_active_minus1: 0,
            chroma_array_type: 1,
        },
    }
}

/// x264 Baseline: CAVLC, no weighted prediction, POC type 2.
fn sliding_window_stream() -> Stream {
    Stream {
        log2_max_frame_num: 4,
        log2_max_pic_order_cnt_lsb: 4,
        pic_order_cnt_type: 2,
        pps: H264PpsRefMarking {
            redundant_pic_cnt_present_flag: false,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            num_ref_idx_l0_default_active_minus1: 2,
            num_ref_idx_l1_default_active_minus1: 0,
            chroma_array_type: 1,
        },
    }
}

/// The NAL payloads of a 4-byte-start-code Annex-B stream, header byte included.
fn annex_b_nals(bytes: &[u8]) -> Vec<&[u8]> {
    const START_CODE: [u8; 4] = [0, 0, 0, 1];
    let starts: Vec<usize> = (0..bytes.len().saturating_sub(3))
        .filter(|&i| bytes[i..i + 4] == START_CODE)
        .collect();
    starts
        .iter()
        .enumerate()
        .map(|(k, &begin)| {
            let end = starts.get(k + 1).copied().unwrap_or(bytes.len());
            &bytes[begin + START_CODE.len()..end]
        })
        .collect()
}

/// The reference marking of every slice in `stream`, in coded order.
fn markings(bytes: &[u8], stream: &Stream) -> Vec<H264RefPicMarking> {
    let sps = H264PocContext {
        separate_colour_plane_flag: false,
        log2_max_frame_num: stream.log2_max_frame_num,
        frame_mbs_only_flag: true,
        pic_order_cnt_type: stream.pic_order_cnt_type,
        log2_max_pic_order_cnt_lsb: stream.log2_max_pic_order_cnt_lsb,
        delta_pic_order_always_zero_flag: false,
        offset_for_non_ref_pic: 0,
        offset_for_top_to_bottom_field: 0,
        offsets_for_ref_frame: &[],
    };
    let pps = [H264PpsPoc {
        pic_parameter_set_id: 0,
        bottom_field_pic_order_in_frame_present_flag: false,
    }];
    annex_b_nals(bytes)
        .into_iter()
        .filter(|nal| matches!(nal.first().map(|b| b & 0x1F), Some(1) | Some(5)))
        .map(|nal| {
            parse_h264_slice_marking(nal, &sps, &pps, &stream.pps)
                .expect("every slice header parses")
                .ref_pic_marking
                .expect("the marking parse fills it")
        })
        .collect()
}

#[test]
fn a_b_pyramid_stream_retires_its_own_references() {
    let markings = markings(B_PYRAMID, &b_pyramid_stream());
    assert!(!markings.is_empty(), "the fixture has slices");
    let adaptive: Vec<&H264RefPicMarking> = markings.iter().filter(|m| m.adaptive).collect();
    assert!(
        !adaptive.is_empty(),
        "this fixture is the one that uses adaptive marking; without it the test asserts nothing"
    );
    for marking in adaptive {
        assert!(
            !marking.ops().is_empty(),
            "an adaptive marking carries at least one operation"
        );
        for op in marking.ops() {
            // A misread bit position would land on any operation code at all, so
            // pinning the kind is what makes this a parse check.
            assert!(
                matches!(op, H264Mmco::ShortTermUnused { .. }),
                "x264 retires short-term references, got {op:?}"
            );
        }
    }
}

#[test]
fn a_sliding_window_stream_carries_no_operations() {
    for marking in markings(SLIDING_WINDOW, &sliding_window_stream()) {
        assert!(
            !marking.adaptive,
            "a P-only stream leaves marking to the window"
        );
        assert!(marking.ops().is_empty());
        assert!(!marking.long_term_reference);
    }
}

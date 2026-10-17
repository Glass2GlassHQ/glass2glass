#![allow(dead_code)] // each test file uses a subset

//! Synthetic access units in every codec the WebRTC elements carry, shaped so
//! str0m's packetizers and g2g's keyframe detectors read them as real frames.
//! Payload bytes stay in `1..=251`, so no byte run can emulate an Annex-B start
//! code and split a NAL unit wrongly.

use g2g_core::frame::Frame;
use g2g_core::{Caps, Dim, MemoryDomain, PipelinePacket, Rate};
use g2g_plugins::webrtc_codec::WebRtcVideoCodec;

pub(crate) const WIDTH: u32 = 640;
pub(crate) const HEIGHT: u32 = 480;
pub(crate) const FPS: u32 = 30;
/// Every 30th unit is a keyframe.
pub(crate) const GOP: usize = 30;
/// Keyframe payloads exceed the MTU, so one unit spans several RTP packets.
pub(crate) const KEYFRAME_PAYLOAD: usize = 4000;
pub(crate) const INTER_PAYLOAD: usize = 1500;
const START_CODE: [u8; 4] = [0, 0, 0, 1];

pub(crate) fn fixed_caps(codec: WebRtcVideoCodec) -> Caps {
    Caps::CompressedVideo {
        codec: codec.video_codec(),
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(FPS << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// `n` payload bytes in `1..=251`, varied by `salt`.
pub(crate) fn payload(n: usize, salt: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (((i as u32 + salt as u32) % 251) as u8) + 1)
        .collect()
}

fn leb128(mut value: usize) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return out;
        }
        out.push(byte | 0x80);
    }
}

fn h264_unit(keyframe: bool, salt: u8, sizes: Sizes) -> Vec<u8> {
    let mut au = Vec::new();
    if keyframe {
        // SPS (type 7), PPS (type 8), IDR slice (type 5, nal_ref_idc = 3).
        au.extend_from_slice(&START_CODE);
        au.extend_from_slice(&[0x67, 0x42, 0x00, 0x1f]);
        au.extend_from_slice(&payload(8, 1));
        au.extend_from_slice(&START_CODE);
        au.extend_from_slice(&[0x68, 0xce, 0x3c, 0x80]);
        au.extend_from_slice(&START_CODE);
        au.push(0x65);
        au.extend_from_slice(&payload(sizes.keyframe, salt));
    } else {
        // Non-IDR slice (type 1, nal_ref_idc = 2).
        au.extend_from_slice(&START_CODE);
        au.push(0x41);
        au.extend_from_slice(&payload(sizes.inter, salt));
    }
    au
}

/// HEVC NAL header: six-bit type in the first byte, layer 0, temporal id 1.
fn h265_nal(nal_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut nal = START_CODE.to_vec();
    nal.extend_from_slice(&[nal_type << 1, 0x01]);
    nal.extend_from_slice(payload);
    nal
}

fn h265_unit(keyframe: bool, salt: u8, sizes: Sizes) -> Vec<u8> {
    const VPS: u8 = 32;
    const SPS: u8 = 33;
    const PPS: u8 = 34;
    const IDR_W_RADL: u8 = 19;
    const TRAIL_R: u8 = 1;
    let mut au = Vec::new();
    if keyframe {
        au.extend(h265_nal(VPS, &payload(8, 1)));
        au.extend(h265_nal(SPS, &payload(12, 2)));
        au.extend(h265_nal(PPS, &payload(4, 3)));
        au.extend(h265_nal(IDR_W_RADL, &payload(sizes.keyframe, salt)));
    } else {
        au.extend(h265_nal(TRAIL_R, &payload(sizes.inter, salt)));
    }
    au
}

fn vp8_unit(keyframe: bool, salt: u8, sizes: Sizes) -> Vec<u8> {
    // frame tag: bit 0 = inter frame, version 0, show_frame in bit 4
    const SHOW_FRAME: u8 = 1 << 4;
    const VP8_START_CODE: [u8; 3] = [0x9d, 0x01, 0x2a];
    let mut frame = Vec::new();
    if keyframe {
        frame.extend_from_slice(&[SHOW_FRAME, 0, 0]);
        frame.extend_from_slice(&VP8_START_CODE);
        frame.extend_from_slice(&(WIDTH as u16).to_le_bytes());
        frame.extend_from_slice(&(HEIGHT as u16).to_le_bytes());
        frame.extend_from_slice(&payload(sizes.keyframe, salt));
    } else {
        frame.extend_from_slice(&[SHOW_FRAME | 1, 0, 0]);
        frame.extend_from_slice(&payload(sizes.inter, salt));
    }
    frame
}

fn vp9_unit(keyframe: bool, salt: u8, sizes: Sizes) -> Vec<u8> {
    // uncompressed header, profile 0, 640x480, bt.709: frame_marker, profile,
    // show_existing_frame, frame_type, show_frame, error_resilient, then the
    // sync code, color_space, color_range and frame_size_minus_1 fields
    const KEYFRAME_HEADER: [u8; 9] = [0x82, 0x49, 0x83, 0x42, 0x40, 0x27, 0xf0, 0x1d, 0xf0];
    // frame_type = 1 (inter), show_frame = 1
    const INTER_HEADER: [u8; 1] = [0x86];
    let mut frame = Vec::new();
    if keyframe {
        frame.extend_from_slice(&KEYFRAME_HEADER);
        frame.extend_from_slice(&payload(sizes.keyframe, salt));
    } else {
        frame.extend_from_slice(&INTER_HEADER);
        frame.extend_from_slice(&payload(sizes.inter, salt));
    }
    frame
}

/// One low-overhead OBU: header with has_size_field, leb128 size, payload.
fn av1_obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
    const HAS_SIZE_FIELD: u8 = 0x02;
    let mut obu = vec![(obu_type << 3) | HAS_SIZE_FIELD];
    obu.extend(leb128(payload.len()));
    obu.extend_from_slice(payload);
    obu
}

fn av1_unit(keyframe: bool, salt: u8, sizes: Sizes) -> Vec<u8> {
    const OBU_SEQUENCE_HEADER: u8 = 1;
    const OBU_TEMPORAL_DELIMITER: u8 = 2;
    const OBU_FRAME: u8 = 6;
    // frame header bits: show_existing_frame 0, frame_type KEY (00) / INTER (01), show_frame 1
    const KEY_FRAME_HEADER: u8 = 0b0001_0000;
    const INTER_FRAME_HEADER: u8 = 0b0011_0000;
    let mut unit = av1_obu(OBU_TEMPORAL_DELIMITER, &[]);
    if keyframe {
        unit.extend(av1_obu(OBU_SEQUENCE_HEADER, &payload(12, 1)));
        let mut frame = vec![KEY_FRAME_HEADER];
        frame.extend(payload(sizes.keyframe, salt));
        unit.extend(av1_obu(OBU_FRAME, &frame));
    } else {
        let mut frame = vec![INTER_FRAME_HEADER];
        frame.extend(payload(sizes.inter, salt));
        unit.extend(av1_obu(OBU_FRAME, &frame));
    }
    unit
}

/// Picture payload bytes per unit kind.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sizes {
    pub(crate) keyframe: usize,
    pub(crate) inter: usize,
}

impl Sizes {
    /// Units that span several RTP packets.
    pub(crate) const LARGE: Sizes = Sizes {
        keyframe: KEYFRAME_PAYLOAD,
        inter: INTER_PAYLOAD,
    };
}

/// `frames` access units in `codec` at `sizes`, a keyframe every `GOP` units
/// starting at 0.
pub(crate) fn synthetic_stream_sized(
    codec: WebRtcVideoCodec,
    frames: usize,
    sizes: Sizes,
) -> Vec<Vec<u8>> {
    (0..frames)
        .map(|f| {
            let keyframe = f % GOP == 0;
            let salt = f as u8;
            match codec {
                WebRtcVideoCodec::H264 => h264_unit(keyframe, salt, sizes),
                WebRtcVideoCodec::H265 => h265_unit(keyframe, salt, sizes),
                WebRtcVideoCodec::Vp8 => vp8_unit(keyframe, salt, sizes),
                WebRtcVideoCodec::Vp9 => vp9_unit(keyframe, salt, sizes),
                WebRtcVideoCodec::Av1 => av1_unit(keyframe, salt, sizes),
            }
        })
        .collect()
}

/// [`synthetic_stream_sized`] with units that span several RTP packets.
pub(crate) fn synthetic_stream(codec: WebRtcVideoCodec, frames: usize) -> Vec<Vec<u8>> {
    synthetic_stream_sized(codec, frames, Sizes::LARGE)
}

pub(crate) fn data_frames(packets: &[PipelinePacket]) -> Vec<&Frame> {
    packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .collect()
}

pub(crate) fn frame_bytes(frame: &Frame) -> &[u8] {
    match &frame.domain {
        MemoryDomain::System(slice) => slice.as_slice(),
        other => panic!("frame outside system memory: {other:?}"),
    }
}

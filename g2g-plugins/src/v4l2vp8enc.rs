// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;
use core::mem::{offset_of, size_of};

use g2g_core::memory::MemoryDomain;
use g2g_core::{BitstreamAlignment, ElementMetadata, G2gError, PropertySpec, VideoCodec};

use crate::v4l2stateless::{control, scalar_control};
use crate::v4l2statelessenc::{
    v4l2_encoder_properties, EncodeSession, Picture, QuantizerScale, RateControl, RateSettings,
    V4l2StatelessEnc, V4l2StatelessEncoder, DEFAULT_KEYFRAME_INTERVAL,
};

// linux/v4l2-controls.h of the STMicroelectronics v6.6 kernel, mainline Linux has no VP8 encode controls
const V4L2_CID_STATELESS_VP8_ENCODE_PARAMS: u32 = 0x00a4_09c9;
const V4L2_CID_STATELESS_VP8_ENCODE_QP: u32 = 0x00a4_09ca;
const V4L2_VP8_FRAME_FLAG_SHOWFRAME: u32 = 0x1;
const V4L2_VP8_FRAME_TYPE_KEYFRAME: u8 = 0;
const V4L2_VP8_FRAME_TYPE_INTER: u8 = 1;
const V4L2_PIX_FMT_VP8_FRAME: u32 = u32::from_le_bytes(*b"VP8F");

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct V4l2CtrlVp8EncodeParams {
    flags: u32,
    frame_type: u8,
    color_space: u8,
    clamping_type: u8,
    loop_filter_type: u8,
    loop_filter_level: u8,
    sharpness_level: u8,
    log2_nbr_of_dct_partitions: u8,
    prob_intra: u8,
    prob_last: u8,
    prob_gf: u8,
    copy_buffer_to_golden: u8,
    copy_buffer_to_alternate: u8,
    reference_type: u8,
}

// sizes and offsets printed by a C program built against the STMicroelectronics v6.6 header
const _: () = assert!(size_of::<V4l2CtrlVp8EncodeParams>() == 20);
const _: () = assert!(offset_of!(V4l2CtrlVp8EncodeParams, frame_type) == 4);
const _: () = assert!(offset_of!(V4l2CtrlVp8EncodeParams, prob_intra) == 11);
const _: () = assert!(offset_of!(V4l2CtrlVp8EncodeParams, reference_type) == 16);

// the H1 driver refuses any other loop filter level
const KEYFRAME_LOOP_FILTER_LEVEL: u8 = 26;
const INTER_FRAME_LOOP_FILTER_LEVEL: u8 = 12;
const PROB_INTRA: u8 = 63;
const PROB_LAST: u8 = 255;
const PROB_GOLDEN: u8 = 128;

const VP8_MAX_QUANTIZER: u8 = 127;

// RFC 6386 section 14.1 ac_qlookup
const AC_QUANTIZER_STEPS: [u16; VP8_MAX_QUANTIZER as usize + 1] = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94,
    96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140,
    143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205,
    209, 213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
];

fn vp8_quantizer_step(quantizer: u8) -> f64 {
    f64::from(AC_QUANTIZER_STEPS[usize::from(quantizer)])
}

#[derive(Debug, Default)]
pub struct V4l2Vp8Encoder;

impl V4l2StatelessEncoder for V4l2Vp8Encoder {
    const LOG_CATEGORY: &'static str = "V4l2SlVp8Enc";
    const CODEC: VideoCodec = VideoCodec::Vp8;
    const ALIGNMENT: BitstreamAlignment = BitstreamAlignment::Unspecified;
    const CODED_PIXEL_FORMAT: u32 = V4L2_PIX_FMT_VP8_FRAME;
    const QUANTIZER: QuantizerScale = QuantizerScale {
        max: VP8_MAX_QUANTIZER,
        step: vp8_quantizer_step,
    };
    const RATE_DEFAULTS: RateSettings = RateSettings {
        keyframe_interval: DEFAULT_KEYFRAME_INTERVAL,
        quantizer: 10,
        rate_control: RateControl::ConstantQp,
        bitrate: 0,
        qp_min: 0,
        qp_max: VP8_MAX_QUANTIZER,
        qp_step: 20,
    };

    fn metadata() -> ElementMetadata {
        ElementMetadata::new(
            "V4L2 stateless VP8 encoder",
            "Codec/Encoder/Video/Hardware",
            "Encodes VP8 on a V4L2 stateless encoder through the media request API",
            crate::AARON_BOXER,
        )
    }

    fn properties() -> &'static [PropertySpec] {
        v4l2_encoder_properties!(
            quantizer: "10",
            quantizer_range: "0..127",
            qp_min: "0",
            qp_max: "127",
            qp_step: "20",
            codec: []
        )
    }

    fn encode(
        &mut self,
        session: &mut EncodeSession,
        picture: &Picture,
        keyframe: bool,
        qp: u8,
        input: &MemoryDomain,
    ) -> Result<Vec<u8>, G2gError> {
        let (frame_type, loop_filter_level) = if keyframe {
            (V4L2_VP8_FRAME_TYPE_KEYFRAME, KEYFRAME_LOOP_FILTER_LEVEL)
        } else {
            (V4L2_VP8_FRAME_TYPE_INTER, INTER_FRAME_LOOP_FILTER_LEVEL)
        };
        // the H1 driver writes no golden or altref refresh into the frame header whatever the flags say
        let mut params = V4l2CtrlVp8EncodeParams {
            flags: V4L2_VP8_FRAME_FLAG_SHOWFRAME,
            frame_type,
            loop_filter_level,
            prob_intra: PROB_INTRA,
            prob_last: PROB_LAST,
            prob_gf: PROB_GOLDEN,
            ..Default::default()
        };
        let coded = session.run(
            picture,
            &mut [
                control(V4L2_CID_STATELESS_VP8_ENCODE_PARAMS, &mut params),
                scalar_control(V4L2_CID_STATELESS_VP8_ENCODE_QP, i32::from(qp)),
            ],
            input,
        )?;
        Ok(coded.to_vec())
    }
}

pub type V4l2SlVp8Enc = V4l2StatelessEnc<V4l2Vp8Encoder>;

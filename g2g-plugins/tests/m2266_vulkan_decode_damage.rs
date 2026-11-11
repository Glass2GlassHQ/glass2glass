// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M2266: `vulkanvideodec` decodes on through a damaged access unit. It drops
//! the pictures up to the next keyframe, asks upstream for one, and resumes.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

#[path = "common/annexb.rs"]
mod annexb;
#[path = "common/collect_sink.rs"]
mod collect_sink;

use collect_sink::CollectSink;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, BitstreamAlignment, Caps, Colorimetry, Dim, FrameTiming, MemoryDomain,
    PipelinePacket, PixelAspect, Rate, Reconfigure, VideoCodec,
};
use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoDec, VulkanVideoError};

// IDR P P P P, IDR P P P P
const STREAM: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
const PICTURE: (u32, u32) = (640, 480);
const DAMAGED_ACCESS_UNIT: usize = 2;
const SECOND_KEYFRAME: usize = 5;
// a P slice NAL header with no slice header after it
const TRUNCATED_SLICE: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x41];

static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn has_h264_decode() -> bool {
    match block_on(open_h264_decode_device()) {
        Ok(_) => true,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => false,
        Err(e) => panic!("open the H.264 decode device: {e:?}"),
    }
}

struct Decoded {
    pictures: Vec<Vec<u8>>,
    keyframe_requests: Vec<usize>,
}

fn decode(access_units: Vec<Vec<u8>>) -> Decoded {
    let mut decoder = VulkanVideoDec::new();
    decoder
        .configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(PICTURE.0),
            height: Dim::Fixed(PICTURE.1),
            framerate: Rate::Any,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: PixelAspect::Any,
            alignment: BitstreamAlignment::Unspecified,
        })
        .expect("configure opens the decode device");
    let mut sink = CollectSink::default();
    let mut keyframe_requests = Vec::new();
    for (index, access_unit) in access_units.into_iter().enumerate() {
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(access_unit.into_boxed_slice())),
            timing: FrameTiming {
                pts_ns: index as u64 * 33_000_000,
                ..Default::default()
            },
            sequence: index as u64,
            meta: Default::default(),
        };
        block_on(decoder.process(PipelinePacket::DataFrame(frame), &mut sink))
            .expect("decode access unit");
        if decoder.take_reconfigure() == Some(Reconfigure::ForceKeyframe) {
            keyframe_requests.push(index);
        }
    }
    block_on(decoder.process(PipelinePacket::Eos, &mut sink)).expect("eos drains the decoder");
    let pictures = sink
        .packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => frame.domain.as_system_slice().map(<[u8]>::to_vec),
            _ => None,
        })
        .collect();
    Decoded {
        pictures,
        keyframe_requests,
    }
}

#[test]
fn a_damaged_access_unit_drops_pictures_to_the_next_keyframe_and_asks_for_one() {
    let _gpu = GPU_TEST_LOCK.lock().unwrap();
    if !has_h264_decode() {
        eprintln!("skipping: no Vulkan H.264 decode adapter");
        return;
    }
    let access_units = annexb::access_units(STREAM);
    let clean = decode(access_units.clone());
    assert_eq!(clean.pictures.len(), access_units.len());
    assert!(clean.keyframe_requests.is_empty());

    let mut damaged_units = access_units;
    damaged_units[DAMAGED_ACCESS_UNIT] = TRUNCATED_SLICE.to_vec();
    let damaged = decode(damaged_units);

    let expected: Vec<&Vec<u8>> = clean.pictures[..DAMAGED_ACCESS_UNIT]
        .iter()
        .chain(&clean.pictures[SECOND_KEYFRAME..])
        .collect();
    assert_eq!(damaged.pictures.len(), expected.len());
    assert!(
        damaged
            .pictures
            .iter()
            .zip(expected)
            .all(|(got, want)| got == want),
        "the pictures around the gap match a clean decode"
    );
    assert_eq!(damaged.keyframe_requests, [DAMAGED_ACCESS_UNIT]);
}

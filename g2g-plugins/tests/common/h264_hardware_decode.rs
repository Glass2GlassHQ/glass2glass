// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![allow(dead_code)] // no one test file uses every helper here

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::runtime::block_on;
use g2g_core::{
    AllocationParams, AsyncElement, BitstreamAlignment, Caps, Colorimetry, Dim, FrameTiming,
    G2gError, OutputSink, PipelinePacket, PixelAspect, PushOutcome, Rate, RawVideoFormat,
    Reconfigure, VideoCodec,
};
use g2g_plugins::h264hardwaredec::{H264HardwareDec, H264HardwareDecoder};

pub(crate) struct Fixture {
    pub(crate) name: &'static str,
    pub(crate) stream: &'static [u8],
    pub(crate) picture: (u32, u32),
}

pub(crate) const BFRAMES: Fixture = Fixture {
    name: "h264_640x480_bframes.h264",
    stream: include_bytes!("../fixtures/h264_640x480_bframes.h264"),
    picture: (640, 480),
};
// decodes into a texture taller than the picture
pub(crate) const BOTTOM_PADDED: Fixture = Fixture {
    name: "h264_320x180.h264",
    stream: include_bytes!("../fixtures/h264_320x180.h264"),
    picture: (320, 180),
};
// crops six columns from the left and four rows from the top
pub(crate) const CROPPED_TOP_LEFT: Fixture = Fixture {
    name: "h264_304x180_crop.h264",
    stream: include_bytes!("../fixtures/h264_304x180_crop.h264"),
    picture: (304, 180),
};
pub(crate) const HIGH10: Fixture = Fixture {
    name: "h264_high10_320x240.h264",
    stream: include_bytes!("../fixtures/h264_high10_320x240.h264"),
    picture: (320, 240),
};
// IDR P P P P, IDR P P P P
pub(crate) const TWO_GOPS: Fixture = Fixture {
    name: "h264_640x480.h264",
    stream: include_bytes!("../fixtures/h264_640x480.h264"),
    picture: (640, 480),
};
// the second GOP opens on a non-IDR I picture with a recovery point SEI, then one leading B picture
pub(crate) const OPEN_GOP: Fixture = Fixture {
    name: "h264_320x240_opengop.h264",
    stream: include_bytes!("../fixtures/h264_320x240_opengop.h264"),
    picture: (320, 240),
};
pub(crate) const EIGHT_BIT: u8 = 8;

#[derive(Default)]
pub(crate) struct Collect {
    pub(crate) packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl Collect {
    pub(crate) fn into_frames(self) -> Vec<Frame> {
        self.packets
            .into_iter()
            .filter_map(|packet| match packet {
                PipelinePacket::DataFrame(frame) => Some(frame),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn caps_changes(&self) -> Vec<&Caps> {
        self.packets
            .iter()
            .filter_map(|packet| match packet {
                PipelinePacket::CapsChanged(caps) => Some(caps),
                _ => None,
            })
            .collect()
    }
}

pub(crate) fn h264_caps((width, height): (u32, u32)) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Any,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: PixelAspect::Any,
        alignment: BitstreamAlignment::Unspecified,
    }
}

pub(crate) fn system_frame(bytes: &[u8]) -> Frame {
    Frame::from_system_bytes(bytes.to_vec(), FrameTiming::default(), 0)
}

pub(crate) fn decode<B: H264HardwareDecoder>(
    fixture: &Fixture,
    domain: MemoryDomainKind,
) -> (H264HardwareDec<B>, Collect, Result<(), G2gError>) {
    decode_stream::<B>(fixture.stream, fixture.picture, domain)
}

// the whole stream as one buffer, so every picture shares one timestamp
pub(crate) fn decode_stream<B: H264HardwareDecoder>(
    stream: &[u8],
    picture: (u32, u32),
    domain: MemoryDomainKind,
) -> (H264HardwareDec<B>, Collect, Result<(), G2gError>) {
    let mut decoder = H264HardwareDec::<B>::new();
    decoder.configure_allocation(&AllocationParams {
        size_bytes: 0,
        min_buffers: 1,
        align: 1,
        domain,
        accepts: DomainSet::only(domain),
        ..Default::default()
    });
    decoder
        .configure_pipeline(&h264_caps(picture))
        .expect("configure opens the decode device");
    let mut collect = Collect::default();
    let decoded = block_on(decoder.process(
        PipelinePacket::DataFrame(system_frame(stream)),
        &mut collect,
    ));
    if decoded.is_ok() {
        block_on(decoder.process(PipelinePacket::Eos, &mut collect))
            .expect("eos drains the reorder");
    }
    (decoder, collect, decoded)
}

pub(crate) fn system_frames<B: H264HardwareDecoder>(fixture: &Fixture) -> Vec<Vec<u8>> {
    system_frames_of::<B>(fixture.name, fixture.stream, fixture.picture)
}

pub(crate) fn system_frames_of<B: H264HardwareDecoder>(
    name: &str,
    stream: &[u8],
    picture: (u32, u32),
) -> Vec<Vec<u8>> {
    let (_, collect, decoded) = decode_stream::<B>(stream, picture, MemoryDomainKind::System);
    decoded.expect("decode stream");
    let (width, height) = picture;
    let caps = collect.caps_changes();
    assert!(
        matches!(
            caps.as_slice(),
            [Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(caps_width),
                height: Dim::Fixed(caps_height),
                ..
            }] if (*caps_width, *caps_height) == (width, height)
        ),
        "{name}: one NV12 caps event at the visible size, got {caps:?}"
    );
    collect
        .into_frames()
        .iter()
        .map(|frame| {
            frame
                .domain
                .as_system_slice()
                .expect("system frame")
                .to_vec()
        })
        .collect()
}

pub(crate) fn system_output_is_bit_exact_in_display_order<B: H264HardwareDecoder>() {
    for fixture in [BFRAMES, BOTTOM_PADDED, CROPPED_TOP_LEFT] {
        let frames: Vec<Vec<u8>> = system_frames::<B>(&fixture)
            .iter()
            .map(|nv12| crate::decode_ref::two_plane_to_planar(nv12, EIGHT_BIT, fixture.picture))
            .collect();
        crate::decode_ref::assert_frames_match_reference(
            &frames,
            RawVideoFormat::I420,
            fixture.picture,
            fixture.name,
        );
    }
}

pub(crate) fn latency_counts_the_pictures_the_reorder_holds<B: H264HardwareDecoder>() {
    let (decoder, _, decoded) = decode::<B>(&BFRAMES, MemoryDomainKind::System);
    decoded.expect("decode stream");
    let latency = decoder.latency();
    assert!(latency.min_ns > 0, "{latency:?}");
    assert_eq!(
        latency.max_ns,
        Some(latency.min_ns),
        "the default buffering holds its full depth"
    );
}

pub(crate) fn a_flush_resumes_bit_exact_from_the_next_keyframe<B: H264HardwareDecoder>() {
    let (mut decoder, mut collect, decoded) = decode::<B>(&BFRAMES, MemoryDomainKind::System);
    decoded.expect("decode stream");
    block_on(decoder.process(
        PipelinePacket::DataFrame(system_frame(BFRAMES.stream)),
        &mut collect,
    ))
    .expect("decode the stream again before the seek");
    block_on(decoder.process(PipelinePacket::Flush, &mut collect)).expect("flush");
    let flushed_at = collect.packets.len();
    block_on(decoder.process(
        PipelinePacket::DataFrame(system_frame(BFRAMES.stream)),
        &mut collect,
    ))
    .expect("decode after the flush");
    block_on(decoder.process(PipelinePacket::Eos, &mut collect)).expect("eos drains the reorder");
    let frames: Vec<Vec<u8>> = collect.packets[flushed_at..]
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => frame.domain.as_system_slice(),
            _ => None,
        })
        .map(|nv12| crate::decode_ref::two_plane_to_planar(nv12, EIGHT_BIT, BFRAMES.picture))
        .collect();
    crate::decode_ref::assert_frames_match_reference(
        &frames,
        RawVideoFormat::I420,
        BFRAMES.picture,
        BFRAMES.name,
    );
}

pub(crate) fn high_bit_depth_stream_is_refused<B: H264HardwareDecoder>() {
    let (_, collect, decoded) = decode::<B>(&HIGH10, MemoryDomainKind::System);
    assert_eq!(decoded, Err(G2gError::CapsMismatch));
    assert!(collect.into_frames().is_empty());
}

struct AccessUnitDecode {
    pictures: Vec<Vec<u8>>,
    presentation_times: Vec<u64>,
    keyframe_requests: Vec<usize>,
}

const ACCESS_UNIT_PERIOD_NS: u64 = 33_000_000;

fn system_decoder<B: H264HardwareDecoder>(fixture: &Fixture) -> H264HardwareDec<B> {
    let mut decoder = H264HardwareDec::<B>::new();
    decoder.configure_allocation(&AllocationParams {
        size_bytes: 0,
        min_buffers: 1,
        align: 1,
        domain: MemoryDomainKind::System,
        accepts: DomainSet::only(MemoryDomainKind::System),
        ..Default::default()
    });
    decoder
        .configure_pipeline(&h264_caps(fixture.picture))
        .expect("configure opens the decode device");
    decoder
}

fn decode_access_units<B: H264HardwareDecoder>(
    fixture: &Fixture,
    access_units: Vec<Vec<u8>>,
) -> AccessUnitDecode {
    let mut decoder = system_decoder::<B>(fixture);
    let mut collect = Collect::default();
    let mut keyframe_requests = Vec::new();
    for (index, access_unit) in access_units.iter().enumerate() {
        let mut frame = system_frame(access_unit);
        frame.timing.pts_ns = index as u64 * ACCESS_UNIT_PERIOD_NS;
        block_on(decoder.process(PipelinePacket::DataFrame(frame), &mut collect))
            .expect("decode access unit");
        if decoder.take_reconfigure() == Some(Reconfigure::ForceKeyframe) {
            keyframe_requests.push(index);
        }
    }
    block_on(decoder.process(PipelinePacket::Eos, &mut collect)).expect("eos drains the reorder");
    let frames = collect.into_frames();
    let pictures = frames
        .iter()
        .map(|frame| {
            frame
                .domain
                .as_system_slice()
                .expect("system frame")
                .to_vec()
        })
        .collect();
    AccessUnitDecode {
        pictures,
        presentation_times: frames.iter().map(|frame| frame.timing.pts_ns).collect(),
        keyframe_requests,
    }
}

pub(crate) fn a_damaged_access_unit_drops_pictures_to_the_next_keyframe_and_asks_for_one<
    B: H264HardwareDecoder,
>() {
    const DAMAGED_ACCESS_UNIT: usize = 2;
    const SECOND_KEYFRAME: usize = 5;
    // a P slice NAL header with no slice header after it
    const TRUNCATED_SLICE: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x41];
    let access_units = crate::annexb::access_units(TWO_GOPS.stream);
    let clean = decode_access_units::<B>(&TWO_GOPS, access_units.clone());
    assert_eq!(clean.pictures.len(), access_units.len());
    assert!(clean.keyframe_requests.is_empty());

    let mut damaged_units = access_units;
    damaged_units[DAMAGED_ACCESS_UNIT] = TRUNCATED_SLICE.to_vec();
    let damaged = decode_access_units::<B>(&TWO_GOPS, damaged_units);

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

pub(crate) fn a_flush_resumes_bit_exact_from_a_recovery_point_i_picture<B: H264HardwareDecoder>() {
    const RECOVERY_POINT_ACCESS_UNIT: usize = 9;
    const LEADING_PICTURE_ACCESS_UNIT: usize = 10;
    // the leading B picture shows before the I picture
    const RECOVERY_POINT_DISPLAY_INDEX: usize = 10;
    let access_units = crate::annexb::access_units(OPEN_GOP.stream);
    let clean = decode_access_units::<B>(&OPEN_GOP, access_units.clone());
    assert_eq!(clean.pictures.len(), access_units.len());

    let mut decoder = system_decoder::<B>(&OPEN_GOP);
    let mut collect = Collect::default();
    for access_unit in &access_units {
        block_on(decoder.process(
            PipelinePacket::DataFrame(system_frame(access_unit)),
            &mut collect,
        ))
        .expect("decode before the seek");
    }
    block_on(decoder.process(PipelinePacket::Flush, &mut collect)).expect("flush");
    let flushed_at = collect.packets.len();
    // a seek to the recovery point skips the leading picture, which predicts from the GOP before
    let resumed = access_units[RECOVERY_POINT_ACCESS_UNIT..]
        .iter()
        .enumerate()
        .filter(|&(offset, _)| offset + RECOVERY_POINT_ACCESS_UNIT != LEADING_PICTURE_ACCESS_UNIT);
    for (offset, access_unit) in resumed {
        let mut frame = system_frame(access_unit);
        frame.timing.pts_ns = offset as u64 * 33_000_000;
        block_on(decoder.process(PipelinePacket::DataFrame(frame), &mut collect))
            .expect("decode after the seek");
    }
    block_on(decoder.process(PipelinePacket::Eos, &mut collect)).expect("eos drains the reorder");
    let resumed_pictures: Vec<&[u8]> = collect.packets[flushed_at..]
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => frame.domain.as_system_slice(),
            _ => None,
        })
        .collect();
    let expected: Vec<&[u8]> = clean.pictures[RECOVERY_POINT_DISPLAY_INDEX..]
        .iter()
        .map(Vec::as_slice)
        .collect();
    assert_eq!(resumed_pictures.len(), expected.len());
    assert!(
        resumed_pictures == expected,
        "the pictures from the recovery point match a clean decode"
    );
}

// mp4s with no composition offsets stamp each access unit in decode order
pub(crate) fn decode_order_timestamps_leave_in_rising_order<B: H264HardwareDecoder>() {
    let access_units = crate::annexb::access_units(BFRAMES.stream);
    let count = access_units.len() as u64;
    let decoded = decode_access_units::<B>(&BFRAMES, access_units);
    let expected: Vec<u64> = (0..count)
        .map(|index| index * ACCESS_UNIT_PERIOD_NS)
        .collect();
    assert_eq!(decoded.presentation_times, expected);
}

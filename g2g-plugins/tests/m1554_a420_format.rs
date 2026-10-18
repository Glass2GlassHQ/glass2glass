//! M1554: `A420` is I420 plus a full-resolution alpha plane, and `videoconvert`
//! drops that alpha for I420 and carries it into RGBA.

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

use g2g_core::caps_parse::parse_raw_format;
use g2g_core::memory::SystemSlice;
use g2g_core::wire::{raw_format_from_u8, raw_format_to_u8};
use g2g_core::{
    AsyncElement, Caps, CapsSet, Colorimetry, Dim, Frame, FrameTiming, MemoryDomain,
    PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::videoconvert::{convert, VideoConvert};

// odd on both axes, so the chroma planes round up
const WIDTH: u32 = 7;
const HEIGHT: u32 = 5;
const RGBA_ALPHA_BYTE: usize = 3;
const RGBA_PIXEL_BYTES: usize = 4;

fn raw(format: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

fn luma_samples() -> usize {
    (WIDTH * HEIGHT) as usize
}

// An A420 frame whose alpha plane differs from pixel to pixel.
fn a420_frame() -> Vec<u8> {
    let size = RawVideoFormat::A420
        .unpadded_frame_bytes(WIDTH, HEIGHT)
        .unwrap() as usize;
    (0..size).map(|i| (i * 37 % 251) as u8).collect()
}

fn alpha_plane(frame: &[u8]) -> &[u8] {
    let offset = RawVideoFormat::A420.plane_offset(3, WIDTH, HEIGHT).unwrap() as usize;
    &frame[offset..offset + luma_samples()]
}

async fn convert_with_element(target: RawVideoFormat, input: Vec<u8>) -> Vec<u8> {
    let mut element = VideoConvert::new(target);
    element
        .configure_pipeline(&raw(RawVideoFormat::A420))
        .expect("videoconvert takes A420");
    let mut sink = CollectSink::default();
    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(input.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };
    element
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("the convert runs");
    sink.packets
        .into_iter()
        .find_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => {
                Some(frame.domain.as_system_slice().unwrap().to_vec())
            }
            _ => None,
        })
        .expect("one converted frame")
}

#[test]
fn a420_is_i420_with_a_full_resolution_alpha_plane() {
    let (a420, i420) = (RawVideoFormat::A420, RawVideoFormat::I420);
    assert_eq!(a420.plane_count(), 4);
    assert_eq!(a420.alpha_plane(), Some(3));
    assert_eq!(i420.alpha_plane(), None);
    assert_eq!(a420.chroma_shift(), i420.chroma_shift());
    for plane in 0..i420.plane_count() {
        assert_eq!(
            a420.plane_offset(plane, WIDTH, HEIGHT),
            i420.plane_offset(plane, WIDTH, HEIGHT),
            "plane {plane}"
        );
        assert_eq!(
            a420.plane_stride(plane, WIDTH),
            i420.plane_stride(plane, WIDTH)
        );
        assert_eq!(
            a420.plane_rows(plane, HEIGHT),
            i420.plane_rows(plane, HEIGHT)
        );
    }
    let i420_bytes = i420.unpadded_frame_bytes(WIDTH, HEIGHT).unwrap();
    assert_eq!(a420.plane_offset(3, WIDTH, HEIGHT), Some(i420_bytes));
    assert_eq!(a420.plane_stride(3, WIDTH), Some(WIDTH));
    assert_eq!(a420.plane_rows(3, HEIGHT), Some(HEIGHT));
    assert_eq!(
        a420.unpadded_frame_bytes(WIDTH, HEIGHT),
        Some(i420_bytes + luma_samples() as u64)
    );
}

#[test]
fn a420_caps_parse_and_print_round_trip() {
    assert_eq!(parse_raw_format("A420"), Some(RawVideoFormat::A420));
    assert_eq!(RawVideoFormat::A420.gst_name(), "A420");
    let text = format!("video/x-raw,format=A420,width={WIDTH},height={HEIGHT}");
    let set = CapsSet::from_gst_string(&text).expect("A420 caps parse");
    let [caps] = set.alternatives() else {
        panic!("one alternative, got {set:?}");
    };
    assert!(matches!(
        caps,
        Caps::RawVideo {
            format: RawVideoFormat::A420,
            ..
        }
    ));
    let printed = caps.to_gst_string();
    assert!(printed.contains("format=A420"), "{printed}");
    let reparsed = CapsSet::from_gst_string(&printed).expect("printed caps parse");
    assert_eq!(reparsed.alternatives(), set.alternatives());
    let wire = raw_format_to_u8(RawVideoFormat::A420);
    assert_eq!(raw_format_from_u8(wire).unwrap(), RawVideoFormat::A420);
}

#[tokio::test]
async fn videoconvert_drops_alpha_for_i420() {
    let input = a420_frame();
    let i420_bytes = RawVideoFormat::I420
        .unpadded_frame_bytes(WIDTH, HEIGHT)
        .unwrap() as usize;
    let output = convert_with_element(RawVideoFormat::I420, input.clone()).await;
    assert_eq!(output, input[..i420_bytes]);
}

#[tokio::test]
async fn videoconvert_carries_alpha_into_rgba() {
    let input = a420_frame();
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let i420_bytes = RawVideoFormat::I420
        .unpadded_frame_bytes(WIDTH, HEIGHT)
        .unwrap() as usize;
    let opaque = convert(
        &input[..i420_bytes],
        RawVideoFormat::I420,
        RawVideoFormat::Rgba8,
        w,
        h,
        Colorimetry::UNKNOWN,
    );
    let output = convert_with_element(RawVideoFormat::Rgba8, input.clone()).await;
    assert_eq!(output.len(), opaque.len());
    let alpha = alpha_plane(&input);
    let (converted_pixels, _) = output.as_chunks::<RGBA_PIXEL_BYTES>();
    let (reference_pixels, _) = opaque.as_chunks::<RGBA_PIXEL_BYTES>();
    for (pixel, (converted, reference)) in converted_pixels.iter().zip(reference_pixels).enumerate()
    {
        assert_eq!(
            converted[..RGBA_ALPHA_BYTE],
            reference[..RGBA_ALPHA_BYTE],
            "pixel {pixel} color"
        );
        assert_eq!(
            converted[RGBA_ALPHA_BYTE], alpha[pixel],
            "pixel {pixel} alpha"
        );
    }

    let back = convert(
        &output,
        RawVideoFormat::Rgba8,
        RawVideoFormat::A420,
        w,
        h,
        Colorimetry::UNKNOWN,
    );
    assert_eq!(alpha_plane(&back), alpha);
}

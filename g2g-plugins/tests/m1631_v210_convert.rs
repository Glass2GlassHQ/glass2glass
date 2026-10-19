//! `videoconvert` packs and unpacks v210 byte for byte as ffmpeg does, checked
//! against the fixtures `tools/packed-10bit-fixtures.sh` writes.
#![cfg(feature = "std")]

#[path = "common/raw_convert.rs"]
mod raw_convert;
use raw_convert::{convert, fixture, read_fixture, temp_path};

use g2g_core::RawVideoFormat;

// (v210 file, its planar decode, width, height): 38 and 52 end in a part-filled group
const FFMPEG_ENCODES: [(&str, &str, u32, u32); 3] = [
    ("v210_38x4.v210", "v210_38x4.v210.yuv422p10le", 38, 4),
    ("v210_48x4.v210", "v210_48x4.v210.yuv422p10le", 48, 4),
    ("v210_52x4.v210", "v210_52x4.v210.yuv422p10le", 52, 4),
];
// the 38-wide encode read at 37 pixels, the same row stride
const ODD_WIDTH_DECODE: (&str, &str, u32, u32) = (
    "v210_38x4.v210",
    "v210_38x4_as_37x4.v210.yuv422p10le",
    37,
    4,
);
const PLANAR: RawVideoFormat = RawVideoFormat::I422p10;
const EIGHT_BIT_TARGETS: [RawVideoFormat; 3] = [
    RawVideoFormat::I420,
    RawVideoFormat::Uyvy,
    RawVideoFormat::Bgra8,
];

#[test]
fn the_v210_row_stride_is_the_size_ffmpeg_encodes() {
    for (v210, _, width, height) in FFMPEG_ENCODES {
        let stride = RawVideoFormat::V210.plane_stride(0, width).unwrap();
        assert_eq!(
            read_fixture(v210).len(),
            stride as usize * height as usize,
            "{v210}"
        );
    }
}

#[tokio::test]
async fn v210_unpacks_to_ffmpegs_planar_decode() {
    for (v210, planar, width, height) in FFMPEG_ENCODES.into_iter().chain([ODD_WIDTH_DECODE]) {
        let unpacked = convert(
            &fixture(v210),
            RawVideoFormat::V210,
            PLANAR,
            (width, height),
        )
        .await;
        assert!(
            unpacked == read_fixture(planar),
            "{v210} at {width}x{height}"
        );
    }
}

#[tokio::test]
async fn planar_10_bit_packs_to_ffmpegs_v210() {
    for (v210, planar, width, height) in FFMPEG_ENCODES {
        let packed = convert(
            &fixture(planar),
            PLANAR,
            RawVideoFormat::V210,
            (width, height),
        )
        .await;
        assert!(packed == read_fixture(v210), "{planar}");
    }
}

#[tokio::test]
async fn an_odd_width_frame_packs_and_unpacks_unchanged() {
    let (_, planar, width, height) = ODD_WIDTH_DECODE;
    let packed = convert(
        &fixture(planar),
        PLANAR,
        RawVideoFormat::V210,
        (width, height),
    )
    .await;
    let packed_path = temp_path("odd-width-pack");
    std::fs::write(&packed_path, &packed).expect("temp file written");
    let unpacked = convert(&packed_path, RawVideoFormat::V210, PLANAR, (width, height)).await;
    std::fs::remove_file(&packed_path).ok();
    assert!(unpacked == read_fixture(planar));
}

#[tokio::test]
async fn v210_reaches_8_bit_formats_as_its_planar_decode_does() {
    let (v210, planar, width, height) = FFMPEG_ENCODES[2];
    for target in EIGHT_BIT_TARGETS {
        let from_v210 = convert(
            &fixture(v210),
            RawVideoFormat::V210,
            target,
            (width, height),
        )
        .await;
        let from_planar = convert(&fixture(planar), PLANAR, target, (width, height)).await;
        assert!(from_v210 == from_planar, "{target:?}");
        assert_eq!(
            from_v210.len(),
            target.unpadded_frame_bytes(width, height).unwrap() as usize,
            "{target:?}"
        );
    }
}

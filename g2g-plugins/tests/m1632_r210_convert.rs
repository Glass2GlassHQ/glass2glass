//! `videoconvert` packs and unpacks r210 byte for byte as ffmpeg does, rows laid
//! out unpadded as GStreamer does, checked against the fixtures
//! `tools/packed-10bit-fixtures.sh` writes.
#![cfg(feature = "std")]

#[path = "common/raw_convert.rs"]
mod raw_convert;
use raw_convert::{convert, fixture, read_fixture};

use g2g_core::RawVideoFormat;

// (r210 file, its planar decode, width, height): 37 is odd, ffmpeg pads 37 and 52 but not 64
const FFMPEG_ENCODES: [(&str, &str, u32, u32); 3] = [
    ("r210_37x4.r210", "r210_37x4.r210.gbrp10le", 37, 4),
    ("r210_52x4.r210", "r210_52x4.r210.gbrp10le", 52, 4),
    ("r210_64x4.r210", "r210_64x4.r210.gbrp10le", 64, 4),
];
const PLANAR: RawVideoFormat = RawVideoFormat::Gbr10Le;
const EIGHT_BIT_TARGETS: [RawVideoFormat; 3] = [
    RawVideoFormat::I420,
    RawVideoFormat::Uyvy,
    RawVideoFormat::Bgra8,
];

#[test]
fn gbr_10le_is_laid_out_as_ffmpegs_gbrp10le() {
    for (_, planar, width, height) in FFMPEG_ENCODES {
        let frame_bytes = PLANAR.unpadded_frame_bytes(width, height).unwrap();
        assert_eq!(read_fixture(planar).len() as u64, frame_bytes, "{planar}");
    }
}

#[tokio::test]
async fn r210_unpacks_to_ffmpegs_planar_decode() {
    for (r210, planar, width, height) in FFMPEG_ENCODES {
        let unpacked = convert(
            &fixture(r210),
            RawVideoFormat::R210,
            PLANAR,
            (width, height),
        )
        .await;
        assert!(unpacked == read_fixture(planar), "{r210}");
    }
}

#[tokio::test]
async fn planar_10_bit_rgb_packs_to_ffmpegs_r210() {
    for (r210, planar, width, height) in FFMPEG_ENCODES {
        let packed = convert(
            &fixture(planar),
            PLANAR,
            RawVideoFormat::R210,
            (width, height),
        )
        .await;
        assert!(packed == read_fixture(r210), "{planar}");
    }
}

#[tokio::test]
async fn r210_reaches_8_bit_formats_as_its_planar_decode_does() {
    for (r210, planar, width, height) in FFMPEG_ENCODES {
        for target in EIGHT_BIT_TARGETS {
            let from_r210 = convert(
                &fixture(r210),
                RawVideoFormat::R210,
                target,
                (width, height),
            )
            .await;
            let from_planar = convert(&fixture(planar), PLANAR, target, (width, height)).await;
            assert!(from_r210 == from_planar, "{r210} to {target:?}");
            assert_eq!(
                from_r210.len() as u64,
                target.unpadded_frame_bytes(width, height).unwrap(),
                "{r210} to {target:?}"
            );
        }
    }
}

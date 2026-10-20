#![cfg(feature = "std")]

use std::path::PathBuf;

use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, Dim, PropValue};
use g2g_plugins::filesrc::FileSrc;

const SIZED_FIXTURE: &str = "h264_64x48_bframes.h264";
const SIZED_FIXTURE_DIMENSIONS: (u32, u32) = (64, 48);

#[test]
fn an_annexb_file_probes_its_sps_picture_size() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(SIZED_FIXTURE);
    let mut source = FileSrc::untyped();
    source
        .set_property("location", PropValue::Str(path.to_str().unwrap().into()))
        .unwrap();
    let probed = source.probe_output_caps().expect("the fixture types");
    let Some(Caps::CompressedVideo { width, height, .. }) = probed else {
        panic!("an elementary video stream, got {probed:?}");
    };
    let (expected_width, expected_height) = SIZED_FIXTURE_DIMENSIONS;
    assert_eq!(
        (width, height),
        (Dim::Fixed(expected_width), Dim::Fixed(expected_height))
    );
}

#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
mod decoder_choice {
    use g2g_core::runtime::{is_raw_video, SelectionContext};
    use g2g_core::{Caps, Colorimetry, Dim, Rate, VideoCodec};
    use g2g_plugins::registry::default_registry;

    const AUTOPLUG_MAX_DEPTH: usize = 4;
    const UHD: (u32, u32) = (3840, 2160);
    const FULL_HD: (u32, u32) = (1920, 1080);

    fn decoders_for_system_memory((width, height): (u32, u32)) -> Vec<&'static str> {
        let input = Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate: Rate::Any,
            colorimetry: Colorimetry::UNKNOWN,
        };
        default_registry()
            .autoplug_names_with(
                &input,
                &is_raw_video,
                AUTOPLUG_MAX_DEPTH,
                SelectionContext::default(),
            )
            .expect("an H.265 decode chain")
    }

    #[cfg(feature = "nvdec")]
    #[test]
    fn uhd_hevc_takes_nvdec_and_its_download() {
        assert_eq!(decoders_for_system_memory(UHD), ["nvdec"]);
    }

    #[cfg(not(feature = "nvdec"))]
    #[test]
    fn uhd_hevc_without_nvdec_keeps_the_software_decoder() {
        assert_eq!(decoders_for_system_memory(UHD), ["ffmpegdec"]);
    }

    #[test]
    fn full_hd_hevc_keeps_the_software_decoder() {
        assert_eq!(decoders_for_system_memory(FULL_HD), ["ffmpegdec"]);
    }
}

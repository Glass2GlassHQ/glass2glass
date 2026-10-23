#![cfg(feature = "std")]

use std::path::PathBuf;

use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, Dim, PropValue};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::registry::default_registry;
use g2g_plugins::uridecodebin::ts_decodebin_select;

const SIZED_FIXTURE: &str = "h264_64x48_bframes.h264";
const SIZED_FIXTURE_DIMENSIONS: (u32, u32) = (64, 48);
// made by tools/cost-autoplug-fixtures.sh
const UHD_HEVC_MP4: &str = "h265_3840x2160.mp4";
const UHD_HEVC_MKV: &str = "h265_3840x2160.mkv";
const UHD_HEVC_TS: &str = "h265_3840x2160.ts";
const UHD_HEVC_ONE_FRAME_TS: &str = "h265_3840x2160_one_frame.ts";
const UHD_HEVC_CONTAINER_FIXTURES: [&str; 4] = [
    UHD_HEVC_MP4,
    UHD_HEVC_MKV,
    UHD_HEVC_TS,
    UHD_HEVC_ONE_FRAME_TS,
];
const UHD_HEVC_CONTAINER_DIMENSIONS: (u32, u32) = (3840, 2160);

fn fixture_location(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    path.to_str().unwrap().into()
}

fn probed_file_caps(location: &str) -> Option<Caps> {
    let mut source = FileSrc::untyped();
    source
        .set_property("location", PropValue::Str(location.into()))
        .unwrap();
    source.probe_output_caps().expect("the fixture types")
}

// the caps decodebin plans its decoder chain from, after the demux it plugs for the container
fn container_video_caps(name: &str) -> Caps {
    let location = fixture_location(name);
    let container = probed_file_caps(&location).expect("the fixture is readable");
    let streams = default_registry()
        .container_streams(&location, &container)
        .unwrap_or_else(|| panic!("{name} lists its streams"));
    streams
        .streams
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("{name} has a video stream"))
        .caps
}

fn fixed_size(caps: &Caps) -> (Dim, Dim) {
    let (width, height, _) = caps.dims().expect("video caps");
    (width.clone(), height.clone())
}

#[test]
fn an_annexb_file_probes_its_sps_picture_size() {
    let probed = probed_file_caps(&fixture_location(SIZED_FIXTURE));
    let Some(Caps::CompressedVideo { width, height, .. }) = probed else {
        panic!("an elementary video stream, got {probed:?}");
    };
    let (expected_width, expected_height) = SIZED_FIXTURE_DIMENSIONS;
    assert_eq!(
        (width, height),
        (Dim::Fixed(expected_width), Dim::Fixed(expected_height))
    );
}

#[test]
fn a_container_file_plans_from_its_video_picture_size() {
    let (expected_width, expected_height) = UHD_HEVC_CONTAINER_DIMENSIONS;
    for name in UHD_HEVC_CONTAINER_FIXTURES {
        assert_eq!(
            fixed_size(&container_video_caps(name)),
            (Dim::Fixed(expected_width), Dim::Fixed(expected_height)),
            "{name}"
        );
    }
}

#[test]
fn a_ts_file_fans_out_with_its_video_picture_size() {
    let (expected_width, expected_height) = UHD_HEVC_CONTAINER_DIMENSIONS;
    for name in [UHD_HEVC_TS, UHD_HEVC_ONE_FRAME_TS] {
        let ts = Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        };
        let (_, ports) = ts_decodebin_select(&fixture_location(name), &ts)
            .unwrap_or_else(|| panic!("{name} demuxes"));
        let video = ports
            .first()
            .unwrap_or_else(|| panic!("{name} has a video port"));
        assert_eq!(
            fixed_size(&video.caps),
            (Dim::Fixed(expected_width), Dim::Fixed(expected_height)),
            "{name}"
        );
    }
}

#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
mod decoder_choice {
    use g2g_core::runtime::{is_raw_video, SelectionContext};
    use g2g_core::{Caps, Colorimetry, Dim, Rate, VideoCodec};
    use g2g_plugins::registry::default_registry;

    const AUTOPLUG_MAX_DEPTH: usize = 4;
    const UHD: (u32, u32) = (3840, 2160);
    const FULL_HD: (u32, u32) = (1920, 1080);

    fn sized_hevc((width, height): (u32, u32)) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate: Rate::Any,
            colorimetry: Colorimetry::UNKNOWN,
        }
    }

    fn decoders_for_system_memory(input: &Caps) -> Vec<&'static str> {
        default_registry()
            .autoplug_names_with(
                input,
                &is_raw_video,
                AUTOPLUG_MAX_DEPTH,
                SelectionContext::default(),
            )
            .expect("an H.265 decode chain")
    }

    #[cfg(feature = "nvdec")]
    #[test]
    fn uhd_hevc_takes_nvdec_and_its_download() {
        assert_eq!(decoders_for_system_memory(&sized_hevc(UHD)), ["nvdec"]);
    }

    #[cfg(feature = "nvdec")]
    #[test]
    fn uhd_hevc_in_a_container_takes_nvdec_and_its_download() {
        for name in super::UHD_HEVC_CONTAINER_FIXTURES {
            let input = super::container_video_caps(name);
            assert_eq!(decoders_for_system_memory(&input), ["nvdec"], "{name}");
        }
    }

    #[cfg(not(feature = "nvdec"))]
    #[test]
    fn uhd_hevc_without_nvdec_keeps_the_software_decoder() {
        assert_eq!(decoders_for_system_memory(&sized_hevc(UHD)), ["ffmpegdec"]);
    }

    #[cfg(not(feature = "nvdec"))]
    #[test]
    fn uhd_hevc_in_a_container_without_nvdec_keeps_the_software_decoder() {
        for name in super::UHD_HEVC_CONTAINER_FIXTURES {
            let input = super::container_video_caps(name);
            assert_eq!(decoders_for_system_memory(&input), ["ffmpegdec"], "{name}");
        }
    }

    #[test]
    fn full_hd_hevc_keeps_the_software_decoder() {
        assert_eq!(
            decoders_for_system_memory(&sized_hevc(FULL_HD)),
            ["ffmpegdec"]
        );
    }
}

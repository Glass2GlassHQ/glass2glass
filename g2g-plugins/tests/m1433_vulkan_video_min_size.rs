#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use std::path::PathBuf;

use g2g_core::runtime::block_on;
use g2g_core::{Caps, Dim, G2gError};
use g2g_plugins::ivfdemux::IvfDemux;
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_av1_decode_device, open_h264_decode_device,
    VulkanVideoDevice, VulkanVideoError,
};

mod vulkan_nv12_common;
use vulkan_nv12_common::{launch_frames, H264_CLIP};

// AV1-TEST-VECTORS, below the 128x128 minimum AV1 decode extent of the RTX 3060
const UNDERSIZED_AV1_FIXTURE: &str = "av1-1-b8-01-size-16x16.ivf";

// libtest runs these in parallel and concurrent instance creation SIGSEGVs the NVIDIA loader
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn usable(opened: Result<VulkanVideoDevice, VulkanVideoError>) -> Option<VulkanVideoDevice> {
    match opened {
        Ok(device) => Some(device),
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip: no Vulkan decode adapter for this codec");
            None
        }
        Err(error) => panic!("open decode device: {error:?}"),
    }
}

fn undersized_av1_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(UNDERSIZED_AV1_FIXTURE)
}

fn undersized_av1_caps() -> Caps {
    let file = std::fs::read(undersized_av1_path()).expect("the fixture is checked in");
    IvfDemux::stream_caps(&file).expect("an IVF header")
}

// whether the default AV1 decode GPU reports a minimum above the fixture's size, `None` without one
fn av1_fixture_below_device_minimum() -> Option<bool> {
    let device = usable(block_on(open_av1_decode_device()))?;
    let Caps::CompressedVideo {
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        ..
    } = undersized_av1_caps()
    else {
        panic!("the IVF header carries the picture size");
    };
    let (min_width, min_height) = device.caps().min_coded_extent;
    Some(width < min_width || height < min_height)
}

#[test]
fn an_av1_stream_below_the_device_minimum_fails_loudly() {
    let _gpu = gpu_lock();
    let Some(below_minimum) = av1_fixture_below_device_minimum() else {
        return;
    };
    let decoded = launch_frames(
        &format!(
            "filesrc location={} ! ivfdemux ! vulkanvideodec",
            undersized_av1_path().display()
        ),
        "m1433_undersized_av1",
    );
    if below_minimum {
        assert_eq!(
            decoded.map(|frames| frames.len()),
            Err(G2gError::CapsMismatch),
            "a stream below the minimum must fail, not decode to a flat field"
        );
        return;
    }
    let frames = decoded.expect("a stream at or above the minimum decodes");
    assert!(!frames.is_empty(), "no frames decoded");
    for (index, frame) in frames.iter().enumerate() {
        assert!(
            frame.iter().any(|&sample| sample != frame[0]),
            "frame {index} is a flat field"
        );
    }
}

#[test]
fn an_h264_session_below_the_device_minimum_is_refused() {
    let _gpu = gpu_lock();
    let Some(device) = usable(block_on(open_h264_decode_device())) else {
        return;
    };
    let (min_width, min_height) = device.caps().min_coded_extent;
    let Some(narrower) = min_width.checked_sub(1).filter(|&width| width > 0) else {
        eprintln!("skip: the device decodes any H.264 width");
        return;
    };
    let ps = extract_h264_parameter_sets(H264_CLIP).expect("parse SPS+PPS");
    assert!(matches!(
        device.create_h264_session(&ps, narrower, min_height),
        Err(VulkanVideoError::UnsupportedStream)
    ));
    device
        .create_h264_session(&ps, min_width, min_height)
        .expect("a session at the device minimum");
}

#[cfg(feature = "dav1d")]
#[test]
fn undersized_av1_autoplugs_a_software_decoder_with_vulkanvideodec_preferred() {
    use g2g_core::runtime::{is_raw_video, parse_launch, run_graph, RankOverride};
    use g2g_core::PipelineClock;
    use g2g_plugins::registry::default_registry;

    #[path = "common/ivf.rs"]
    mod ivf;

    const AUTOPLUG_MAX_DEPTH: usize = 4;
    const LINK_CAPACITY: usize = 4;
    const VULKAN_DECODER: &str = "vulkanvideodec";

    struct ZeroClock;
    impl PipelineClock for ZeroClock {
        fn now_ns(&self) -> u64 {
            0
        }
    }

    let _gpu = gpu_lock();
    let Some(below_minimum) = av1_fixture_below_device_minimum() else {
        return;
    };
    let caps = undersized_av1_caps();
    let Caps::CompressedVideo {
        codec,
        framerate,
        colorimetry,
        ..
    } = caps.clone()
    else {
        panic!("the fixture is compressed video: {caps:?}");
    };
    let any_size = Caps::CompressedVideo {
        codec,
        width: Dim::Any,
        height: Dim::Any,
        framerate,
        colorimetry,
    };

    let mut registry = default_registry();
    assert!(registry.override_rank(VULKAN_DECODER, RankOverride::MAX));
    let decoders_for = |input: &Caps| {
        registry
            .autoplug_names(input, &is_raw_video, AUTOPLUG_MAX_DEPTH)
            .expect("a decode chain")
    };
    assert_eq!(decoders_for(&any_size), [VULKAN_DECODER]);
    let chosen = decoders_for(&caps);
    if below_minimum {
        assert!(
            !chosen.is_empty() && !chosen.contains(&VULKAN_DECODER),
            "an undersized stream takes a software decoder: {chosen:?}"
        );
    } else {
        assert_eq!(chosen, [VULKAN_DECODER]);
    }

    let path = undersized_av1_path();
    let file = std::fs::read(&path).expect("the fixture is checked in");
    let line = format!("filesrc location={} ! decodebin ! fakesink", path.display());
    let graph = parse_launch(&registry, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = block_on(run_graph(graph, &ZeroClock, LINK_CAPACITY))
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    assert_eq!(stats.frames_consumed, ivf::ivf_frame_count(&file));
}

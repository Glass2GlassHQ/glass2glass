#![cfg(all(target_os = "linux", feature = "nvdec", feature = "dav1d"))]

use std::path::PathBuf;

use g2g_core::runtime::{is_raw_video, parse_launch, run_graph, RankOverride};
use g2g_core::{AsyncElement, Caps, Dim, PipelineClock};
use g2g_plugins::ivfdemux::IvfDemux;
use g2g_plugins::nvdec::NvDec;
use g2g_plugins::registry::default_registry;

#[path = "common/ivf.rs"]
mod ivf;
use ivf::ivf_frame_count;

// AV1-TEST-VECTORS, smaller than NVDEC's minimum AV1 decode size
const UNDERSIZED_FIXTURE: &str = "av1-1-b8-01-size-16x16.ivf";
const AUTOPLUG_MAX_DEPTH: usize = 4;
const LINK_CAPACITY: usize = 4;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn undersized_av1_autoplugs_a_software_decoder_with_nvdec_preferred() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(UNDERSIZED_FIXTURE);
    let file = std::fs::read(&path).expect("the fixture is checked in");
    let caps = IvfDemux::stream_caps(&file).expect("an IVF header");
    if NvDec::new().configure_pipeline(&caps).is_err() {
        eprintln!("skipping: NVDEC unavailable on this host");
        return;
    }
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
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    };

    let mut registry = default_registry();
    assert!(registry.override_rank("nvdec", RankOverride::MAX));
    let decoders_for = |input: &Caps| {
        registry
            .autoplug_names(input, &is_raw_video, AUTOPLUG_MAX_DEPTH)
            .expect("a decode chain")
    };
    if decoders_for(&any_size) != ["nvdec"] {
        eprintln!("skipping: no AV1 decode on this GPU");
        return;
    }
    let chosen = decoders_for(&caps);
    assert!(
        !chosen.is_empty() && !chosen.contains(&"nvdec"),
        "an undersized stream takes a software decoder: {chosen:?}"
    );

    let line = format!("filesrc location={} ! decodebin ! fakesink", path.display());
    let graph = parse_launch(&registry, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = run_graph(graph, &ZeroClock, LINK_CAPACITY)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    assert_eq!(stats.frames_consumed, ivf_frame_count(&file));
}

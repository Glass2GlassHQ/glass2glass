#![cfg(all(feature = "std", any(feature = "dav1d", feature = "rav1d")))]

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph, LatencyProfile};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

const OBU_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av1_321x181_10bit.obu"
);
// the same stream stream-copied into IVF by ffmpeg
const IVF_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av1_321x181_10bit.ivf"
);
const IVF_FRAME_COUNT_OFFSET: usize = 24;
// far below one temporal unit, so OBUs and their size fields split across reads
const SMALL_BLOCKSIZE: usize = 97;

async fn decode(source: &str, chain: &str, label: &str) -> Vec<u8> {
    let out: PathBuf =
        std::env::temp_dir().join(format!("g2g-m1447-{label}-{}.yuv", std::process::id()));
    let _ = std::fs::remove_file(&out);
    let line = format!(
        "filesrc location={source} blocksize={SMALL_BLOCKSIZE} ! {chain} ! filesink location={}",
        out.display()
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &WallClock::new(), LatencyProfile::Throughput)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let decoded = std::fs::read(&out).expect("decoded frames written");
    let _ = std::fs::remove_file(&out);
    decoded
}

#[tokio::test]
async fn a_raw_obu_file_decodes_like_the_same_stream_in_ivf() {
    let ivf = std::fs::read(IVF_FIXTURE).expect("fixture reads");
    let frames = u32::from_le_bytes(
        ivf[IVF_FRAME_COUNT_OFFSET..IVF_FRAME_COUNT_OFFSET + 4]
            .try_into()
            .expect("u32 field"),
    ) as usize;
    let reference = decode(IVF_FIXTURE, "decodebin", "ivf").await;
    assert!(!reference.is_empty(), "the IVF reference decodes");
    assert_eq!(reference.len() % frames, 0, "whole frames in the reference");

    let typefound = decode(OBU_FIXTURE, "decodebin", "decodebin").await;
    assert_eq!(
        typefound, reference,
        "filesrc ! decodebin sniffs the OBU stream"
    );

    let parsed = decode(OBU_FIXTURE, "av1parse ! decodebin", "av1parse").await;
    assert_eq!(
        parsed, reference,
        "filesrc ! av1parse frames the byte stream"
    );
}

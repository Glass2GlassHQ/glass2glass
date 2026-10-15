#![cfg(all(
    feature = "std",
    target_os = "linux",
    feature = "ffmpeg",
    feature = "dav1d"
))]

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph, LatencyProfile};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

// av1_321x181_10bit.obu stream-copied into IVF by ffmpeg
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av1_321x181_10bit.ivf"
);
const IVF_WIDTH_OFFSET: usize = 12;
const IVF_HEIGHT_OFFSET: usize = 14;
const IVF_FRAME_COUNT_OFFSET: usize = 24;
const TEN_BIT_SAMPLE_BYTES: usize = 2;

fn read_u16_le(bytes: &[u8], offset: usize) -> usize {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]]) as usize
}

fn read_u32_le(bytes: &[u8], offset: usize) -> usize {
    let field: [u8; 4] = bytes[offset..offset + 4].try_into().expect("u32 field");
    u32::from_le_bytes(field) as usize
}

fn i420_10bit_frame_bytes(width: usize, height: usize) -> usize {
    let chroma_samples = width.div_ceil(2) * height.div_ceil(2);
    (width * height + 2 * chroma_samples) * TEN_BIT_SAMPLE_BYTES
}

// videoconvert takes any raw format, so it leaves the decoder output format unpinned
async fn decode_pinned(decoder: &str) -> Vec<u8> {
    let mut registry = default_registry();
    registry
        .apply_rank_spec(&format!("{decoder}:max"))
        .unwrap_or_else(|e| panic!("pins {decoder}: {e:?}"));
    let out: PathBuf =
        std::env::temp_dir().join(format!("g2g-m1431-{decoder}-{}.yuv", std::process::id()));
    let _ = std::fs::remove_file(&out);
    let line = format!(
        "filesrc location={FIXTURE} ! decodebin ! videoconvert \
         ! video/x-raw,format=I420_10LE ! filesink location={}",
        out.display()
    );
    let graph = parse_launch(&registry, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &WallClock::new(), LatencyProfile::Throughput)
        .await
        .unwrap_or_else(|e| panic!("{decoder} runs `{line}`: {e:?}"));
    let decoded = std::fs::read(&out).expect("decoded frames written");
    let _ = std::fs::remove_file(&out);
    decoded
}

#[tokio::test]
async fn ffmpegdec_decodes_ten_bit_av1_into_a_converter_like_dav1d() {
    let ivf = std::fs::read(FIXTURE).expect("fixture reads");
    let width = read_u16_le(&ivf, IVF_WIDTH_OFFSET);
    let height = read_u16_le(&ivf, IVF_HEIGHT_OFFSET);
    let frames = read_u32_le(&ivf, IVF_FRAME_COUNT_OFFSET);

    let reference = decode_pinned("dav1ddec").await;
    assert_eq!(
        reference.len(),
        frames * i420_10bit_frame_bytes(width, height),
        "dav1ddec decodes every {width}x{height} 10-bit frame"
    );
    let decoded = decode_pinned("ffmpegdec").await;
    assert_eq!(decoded, reference, "ffmpegdec matches dav1ddec's samples");
}

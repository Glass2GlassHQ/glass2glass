#![cfg(feature = "wgpu-sink")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{Caps, Dim, PipelineClock, Rate, RawVideoFormat};
use g2g_plugins::appsink::{register_appsink_pull, Pull};
use g2g_plugins::appsrc::register_appsrc;
use g2g_plugins::registry::default_registry;

// NV12 and I420 at 37x5 are 299 bytes, not a whole number of 4-byte words.
const ODD_WIDTH: u32 = 37;
const ODD_HEIGHT: u32 = 5;
const FRAMES: u64 = 2;
const FRAME_PERIOD_NS: u64 = 33_333_333;

// Opening wgpu devices concurrently crashes some drivers.
static GPU_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn has_adapter() -> bool {
    wgpu::Instance::default()
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .is_ok()
}

fn caps(format: RawVideoFormat, width: u32, height: u32) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn frame_pattern(len: usize, frame: u64) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u64).wrapping_mul(7).wrapping_add(frame * 3) as u8)
        .collect()
}

// Runs `appsrc ! {middle} ! appsink` and returns every frame's bytes, with what was fed.
async fn round_trip(
    channel: &str,
    middle: &str,
    format: RawVideoFormat,
    width: u32,
    height: u32,
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let frame_caps = caps(format, width, height);
    let len = format.unpadded_frame_bytes(width, height).unwrap() as usize;
    let sent: Vec<Vec<u8>> = (0..FRAMES).map(|frame| frame_pattern(len, frame)).collect();
    let in_channel = format!("{channel}_in");
    let out_channel = format!("{channel}_out");
    let feed = register_appsrc(&in_channel);
    for (index, bytes) in sent.iter().enumerate() {
        assert!(feed.push(bytes, index as u64 * FRAME_PERIOD_NS));
    }
    feed.end_of_stream();
    let pull = register_appsink_pull(&out_channel);
    let line = format!(
        "appsrc channel={in_channel} caps={} ! {middle} ! appsink channel={out_channel}",
        frame_caps.to_gst_string()
    );
    let graph = parse_launch(&default_registry(), &line).expect("parses");
    run_graph(graph, &ZeroClock, 4).await.expect("runs");
    let mut received = Vec::new();
    while let Pull::Frame(frame) = pull.try_pull() {
        let bytes = frame
            .domain
            .as_system_slice()
            .unwrap_or_else(|| panic!("expected system memory, got {:?}", frame.domain.kind()));
        received.push(bytes.to_vec());
    }
    (received, sent)
}

#[tokio::test]
async fn upload_then_download_returns_the_same_bytes() {
    let _gpu = GPU_LOCK.lock().await;
    if !has_adapter().await {
        eprintln!("no wgpu adapter; skipping");
        return;
    }
    for format in [
        RawVideoFormat::Rgba8,
        RawVideoFormat::Nv12,
        RawVideoFormat::I420,
    ] {
        let channel = format!("m1219_download_{format:?}");
        let (received, sent) = round_trip(
            &channel,
            "wgpuupload ! wgpudownload",
            format,
            ODD_WIDTH,
            ODD_HEIGHT,
        )
        .await;
        assert_eq!(received, sent, "{format:?}");
    }
}

#[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
mod dmabuf {
    use super::*;

    // one dma-buf stride cannot describe an odd-width NV12 or I420 frame
    const EVEN_WIDTH: u32 = 6;
    // 6x2 NV12 and I420 are 18 bytes, not a whole number of 4-byte words
    const SHORT_HEIGHT: u32 = 2;
    const DMABUF_CHAIN: &str = "wgputodmabuf ! dmabuftowgpu ! wgpudownload";

    #[tokio::test]
    async fn upload_then_dmabuf_export_returns_the_same_bytes() {
        let _gpu = GPU_LOCK.lock().await;
        if !has_adapter().await {
            eprintln!("no wgpu adapter; skipping");
            return;
        }
        for format in [
            RawVideoFormat::Rgba8,
            RawVideoFormat::Nv12,
            RawVideoFormat::I420,
        ] {
            let channel = format!("m1219_dmabuf_{format:?}");
            let (received, sent) = round_trip(
                &channel,
                &format!("wgpuupload ! {DMABUF_CHAIN}"),
                format,
                EVEN_WIDTH,
                SHORT_HEIGHT,
            )
            .await;
            assert_eq!(received, sent, "{format:?}");
        }
    }

    #[tokio::test]
    async fn a_system_frame_into_the_export_gets_the_upload_spliced() {
        let _gpu = GPU_LOCK.lock().await;
        if !has_adapter().await {
            eprintln!("no wgpu adapter; skipping");
            return;
        }
        let (received, sent) = round_trip(
            "m1219_spliced",
            DMABUF_CHAIN,
            RawVideoFormat::Nv12,
            EVEN_WIDTH,
            SHORT_HEIGHT,
        )
        .await;
        assert_eq!(received, sent);
    }
}

#![cfg(feature = "std")]

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph_with_bus};
use g2g_core::{Bus, BusMessage, PipelineClock};
use g2g_plugins::registry::default_registry;

// ffmpeg wrote one MJPEG frame a second into a 25 fps stream, padding the gaps with empty chunks
const FIXTURE: &str = "avi_mjpeg_dropped_frames.avi";
const FIXTURE_FRAMES: u64 = 3;
const FRAME_SPACING_NS: u64 = 1_000_000_000;
const STRH_FOURCC: &[u8] = b"strh";
const CHUNK_HEADER_LEN: usize = 8;
const STRH_SAMPLE_SIZE_OFFSET: usize = 44;
const BUS_CAPACITY: usize = 64;
const LINK_CAPACITY: usize = 4;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(FIXTURE)
}

async fn demuxed_pts(path: &std::path::Path) -> Vec<u64> {
    let pipeline = format!(
        "filesrc location={} ! avidemux ! checksumsink",
        path.display()
    );
    let graph = parse_launch(&default_registry(), &pipeline).expect("pipeline parses");
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let run = async move {
        let result = run_graph_with_bus(graph, &ZeroClock, LINK_CAPACITY, &handle).await;
        drop(handle);
        result
    };
    let collect = async {
        let mut pts = Vec::new();
        while let Some(message) = bus.recv().await {
            if let BusMessage::Checksum { pts_ns, .. } = message {
                pts.push(pts_ns);
            }
        }
        pts
    };
    let (result, pts) = tokio::join!(run, collect);
    result.expect("pipeline runs");
    pts
}

fn expected_pts() -> Vec<u64> {
    (0..FIXTURE_FRAMES)
        .map(|frame| frame * FRAME_SPACING_NS)
        .collect()
}

#[tokio::test]
async fn empty_video_chunks_take_their_frame_slots() {
    assert_eq!(demuxed_pts(&fixture_path()).await, expected_pts());
}

#[tokio::test]
async fn video_chunks_count_as_frames_whatever_the_sample_size() {
    let mut file = std::fs::read(fixture_path()).expect("avi fixture");
    let strh = file
        .windows(STRH_FOURCC.len())
        .position(|window| window == STRH_FOURCC)
        .expect("the fixture has a stream header");
    let sample_size_at = strh + CHUNK_HEADER_LEN + STRH_SAMPLE_SIZE_OFFSET;
    // larger than any chunk, so a byte-counted stream would never advance
    let sample_size = u32::try_from(file.len()).expect("small fixture");
    file[sample_size_at..sample_size_at + 4].copy_from_slice(&sample_size.to_le_bytes());
    let path = std::env::temp_dir().join(format!(
        "g2g-m1823-video-sample-size-{}.avi",
        std::process::id()
    ));
    std::fs::write(&path, &file).expect("write the patched avi");
    let pts = demuxed_pts(&path).await;
    std::fs::remove_file(&path).expect("remove the patched avi");
    assert_eq!(pts, expected_pts());
}

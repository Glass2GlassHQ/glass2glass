//! M1025: a `file://` URI is typed from the file's header, not assumed to be MP4.
//!
//! The `file` handler declared H.264-in-MP4 for every path it was given, so
//! `uridecodebin uri=file://clip.h264` built an `Mp4Src` over an elementary
//! stream and died with `CapsMismatch` at the first byte. Both source shapes have
//! to keep working: an ISO-BMFF file still self-demuxes, anything else the sniff
//! recognises becomes a `FileSrc` carrying those caps so the decode chain plugs.
#![cfg(all(feature = "std", feature = "ffmpeg"))]

use g2g_core::runtime::{block_on, is_raw_video, run_graph, Registry};
use g2g_core::PipelineClock;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::registry::default_registry;

/// The runs are not paced; nothing here reads the clock.
struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Link depth for the runs, matching the other decode integration tests.
const LINK_CAPACITY: usize = 4;

/// Search depth for the decode chain, matching the `decodebin` macro.
const MAX_DEPTH: usize = 6;

fn fixture_uri(name: &str) -> String {
    format!(
        "file://{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn decoded_frames(reg: &Registry, uri: &str) -> u64 {
    let graph = reg
        .build_uridecodebin(uri, FakeSink::new(), &is_raw_video, MAX_DEPTH)
        .unwrap_or_else(|e| panic!("{uri} plugs a decode chain, got {e:?}"));
    let stats = block_on(run_graph(graph, &ZeroClock, LINK_CAPACITY))
        .unwrap_or_else(|e| panic!("{uri} runs, got {e:?}"));
    stats.frames_consumed
}

#[test]
fn a_raw_h264_file_uri_decodes() {
    let reg = default_registry();
    assert!(
        decoded_frames(&reg, &fixture_uri("h264_640x480.h264")) > 0,
        "an elementary stream is parsed and decoded, not read as an MP4"
    );
}

#[test]
fn an_mp4_file_uri_still_self_demuxes() {
    let reg = default_registry();
    assert!(
        decoded_frames(&reg, &fixture_uri("av_h264_aac.mp4")) > 0,
        "an ISO-BMFF file keeps its Mp4Src path"
    );
}

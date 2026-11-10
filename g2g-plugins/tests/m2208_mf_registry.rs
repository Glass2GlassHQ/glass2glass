// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M2208: the Media Foundation codec elements in the launch registry.
//! `mfh264dec` decodes and `mfh264enc` encodes from a text pipeline, and
//! `mfaacenc` / `mfaacdec` carry ADTS AAC through `mp4mux` and `qtdemux`.
//!
//! Run with:
//!
//! ```powershell
//! cargo test -p g2g-plugins --features std,mf-decode,mf-encode,mf-aac --test m2208_mf_registry
//! ```
#![cfg(all(
    target_os = "windows",
    feature = "mf-decode",
    feature = "mf-encode",
    feature = "mf-aac"
))]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

const H264_CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

const H264_CLIP_FRAMES: u64 = 10;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp_path(tag: &str, ext: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("g2g-m2208-{}-{}.{}", std::process::id(), tag, ext))
}

async fn frames_consumed(line: &str) -> u64 {
    g2g_core::log::init_from_env();
    let registry = default_registry();
    let graph = parse_launch(&registry, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("`{line}` runs: {e:?}"))
        .frames_consumed
}

#[tokio::test(flavor = "current_thread")]
async fn mfh264dec_decodes_in_a_text_pipeline() {
    let path = temp_path("h264", "h264");
    std::fs::write(&path, H264_CLIP).expect("write fixture");
    let line = format!(
        "filesrc location={} ! h264parse ! mfh264dec ! fakesink",
        path.display()
    );
    let consumed = frames_consumed(&line).await;
    std::fs::remove_file(&path).ok();
    assert_eq!(consumed, H264_CLIP_FRAMES, "every fixture frame decoded");
}

#[tokio::test(flavor = "current_thread")]
async fn mfh264enc_output_decodes_back() {
    let consumed = frames_consumed(
        "videotestsrc num-buffers=5 ! videoconvert ! video/x-raw,format=NV12 \
         ! mfh264enc bitrate=1000000 ! h264parse ! mfh264dec ! fakesink",
    )
    .await;
    assert_eq!(consumed, 5, "every test picture encoded and decoded");
}

#[tokio::test(flavor = "current_thread")]
async fn mfaacenc_output_muxes_and_decodes_back() {
    let path = temp_path("aac", "mp4");
    let encode = format!(
        "audiotestsrc num-buffers=20 ! audioconvert ! audio/x-raw,format=S16LE,rate=48000,channels=2 \
         ! mfaacenc bitrate=160000 ! mp4mux ! filesink location={}",
        path.display()
    );
    frames_consumed(&encode).await;
    let decode = format!(
        "filesrc location={} ! qtdemux ! mfaacdec ! fakesink",
        path.display()
    );
    let consumed = frames_consumed(&decode).await;
    std::fs::remove_file(&path).ok();
    assert!(consumed > 0, "the muxed AAC decodes to PCM");
}

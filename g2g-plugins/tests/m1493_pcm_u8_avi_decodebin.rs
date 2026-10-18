//! M1493: raw PCM in any sample format counts as decoded, so a bare
//! `decodebin` on an unsigned 8-bit PCM AVI plugs no decoder.

#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::registry::default_registry;

const BUFFERS: u64 = 5;

struct ZeroClock;
impl g2g_core::PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn run_line(line: &str) -> u64 {
    let graph =
        parse_launch(&default_registry(), line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"))
        .frames_consumed
}

#[tokio::test]
async fn unsigned_8_bit_avi_audio_reaches_the_sink() {
    let path = std::env::temp_dir().join(format!("g2g-m1493-{}.avi", std::process::id()));
    run_line(&format!(
        "audiotestsrc num-buffers={BUFFERS} ! audioconvert ! audio/x-raw,format=U8,channels=1 \
         ! avimux ! filesink location={}",
        path.display()
    ))
    .await;
    let frames = run_line(&format!(
        "filesrc location={} ! decodebin ! audioconvert ! fakesink",
        path.display()
    ))
    .await;
    std::fs::remove_file(&path).ok();
    assert_eq!(frames, BUFFERS, "every PCM chunk reaches the sink");
}

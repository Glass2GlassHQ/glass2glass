//! Wire-codec server for the glass2glass browser demo's receive direction.
//!
//! A browser can only dial out, so the native side listens: this runs
//! `VideoTestSrc -> RemoteWsSink listen=true`, which serves the whole
//! `PipelinePacket` stream (caps first, then frames with their timing) to
//! whoever connects. The browser runs `run_wire_ingest_to_canvas(url, canvas)`,
//! whose `WsWireSrc` reads exactly that.
//!
//! Usage: `cargo run --release -- [bind=127.0.0.1:9601] [frames=60]`

use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::{AsyncElement, PipelineClock, PropValue};
use g2g_plugins::remotewssink::RemoteWsSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

const DEFAULT_BIND: &str = "127.0.0.1:9601";
const DEFAULT_FRAMES: u64 = 60;
const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const FPS: u32 = 15;

/// The demo has no hardware to pace against; frames leave as fast as the socket
/// takes them.
struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let mut args = std::env::args().skip(1);
    let bind = args.next().unwrap_or_else(|| DEFAULT_BIND.to_string());
    let frames: u64 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_FRAMES);

    let mut source = VideoTestSrc::new(WIDTH, HEIGHT, FPS, frames);
    let mut sink = RemoteWsSink::new(format!("ws://{bind}"));
    sink.set_property("listen", PropValue::Bool(true))
        .expect("listen is a property of the sink");

    println!("wire-serve-server: serving {frames} {WIDTH}x{HEIGHT} frames on ws://{bind}");
    match run_simple_pipeline(
        &mut source,
        &mut sink,
        &ZeroClock,
        LatencyProfile::Live.link_capacity(),
    )
    .await
    {
        Ok(stats) => println!("wire-serve-server: sent {} frames", stats.frames_emitted),
        Err(e) => {
            eprintln!("wire-serve-server: {e:?}");
            std::process::exit(1);
        }
    }
}

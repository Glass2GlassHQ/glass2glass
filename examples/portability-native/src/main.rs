//! Native (CPU) runner for the portability showcase.
//!
//! Builds `VideoTestSrc -> [overlay_transforms] -> FileSink` and writes one
//! annotated RGBA frame. The processing stages come from the exact same
//! `portability_core::overlay_transforms()` the browser build (`g2g-web`) uses;
//! only the source (a test pattern vs a WebCodecs decode) and the sink (a file vs
//! a `<canvas>`) differ. Same elements, same `Caps` negotiation, same
//! `run_linear_chain` runner as the browser.
//!
//! Run: `cargo run --release -- [out.rgba] [width] [height]` (defaults
//! `annotated.rgba 640 480`). View it: `ffmpeg -f rawvideo -pix_fmt rgba -s
//! WxH -i out.rgba out.png`.

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::run_linear_chain;
use g2g_core::PipelineClock;

use g2g_plugins::filesink::FileSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

use portability_core::{overlay_stages, SYNTH_BOX};

/// The pipeline is untimed (one frame, no pacing), so a zero clock suffices.
struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().unwrap_or_else(|| "annotated.rgba".to_string());
    let w: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(640);
    let h: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(480);

    // Source: one RGBA test-pattern frame. Sink: raw RGBA bytes to a file.
    let mut src = VideoTestSrc::new(w, h, 1, 1);
    let mut sink = FileSink::new(&out);
    // The middle of the graph: the SAME processing stages the browser builds
    // (concrete `&mut` so each unsizes to `&mut dyn DynAsyncElement` cleanly).
    let mut stages = overlay_stages(4);
    let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut stages.detect, &mut stages.overlay];

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    match rt.block_on(run_linear_chain(&mut src, transforms, &mut sink, &NullClock, 4)) {
        Ok(stats) => println!(
            "wrote {out}: {w}x{h} RGBA, 1 frame. Synthetic box (normalized) {SYNTH_BOX:?}. stats: {stats:?}\n\
             view: ffmpeg -f rawvideo -pix_fmt rgba -s {w}x{h} -i {out} {out}.png"
        ),
        Err(e) => {
            eprintln!("pipeline error: {e:?}");
            std::process::exit(1);
        }
    }
}

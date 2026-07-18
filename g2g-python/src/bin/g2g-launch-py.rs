//! `g2g-launch-py`: a `gst-launch` analog that also registers the hosted Python
//! elements, so a single text pipeline can run a gst-python-ml element end to
//! end in pure g2g.
//!
//! It is `g2g-launch` plus one line: after [`default_registry`] it calls
//! [`g2g_python::register`], which adds `pyelement` / `pysrc` / `pyaggregator`.
//! The hosted element runs under `GSTML_BACKEND=g2g` (set by the host on first
//! GIL acquisition), so its detections come back as `AnalyticsMeta` and an
//! `analyticsoverlay` can draw them before an `autovideosink`:
//!
//! ```text
//! g2g-launch-py mp4src location=clip.mp4 ! h264parse ! ffmpegdec ! videoconvert \
//!   ! video/x-raw,format=RGBA,width=640,height=640 \
//!   ! pyelement module=objectdetector class=ObjectDetector \
//!       engine-name=onnx model-name=yolo11m.onnx device=cuda:0 \
//!   ! analyticsoverlay ! videoconvert ! autovideosink
//! ```
//!
//! The interpreter must see the gst-python-ml package + its deps: set
//! `PYTHONPATH` (plugin dir + venv site dirs) before launching, exactly as the
//! M322 host test does. Build with `--features launch` (pulls g2g-plugins'
//! ffmpeg / wayland-sink / analytics).

use std::process;
use std::time::{Duration, Instant};

use g2g_core::runtime::{parse_launch, run_graph_with_progress, PipelineProgress};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

// Same steady-state link depth as `g2g-launch` (low latency without starving the
// source); see DESIGN notes on link_capacity dominating glass-to-glass latency.
const LINK_CAPACITY: usize = 4;

const USAGE: &str = "usage: g2g-launch-py [-q] <element> [key=value ...] ! <element> ! ...";

fn main() {
    g2g_core::log::init_from_env();

    // Join the args into one pipeline string; accept a leading `-q`/`--quiet`
    // and skip the common no-op gst-launch flags so a pasted line still runs.
    let mut quiet = false;
    let mut tokens: Vec<String> = Vec::new();
    let mut in_pipeline = false;
    for arg in std::env::args().skip(1) {
        if !in_pipeline && arg.starts_with('-') && arg != "-" {
            match arg.as_str() {
                "-q" | "--quiet" => quiet = true,
                "-e" | "--eos-on-shutdown" | "-m" | "--messages" | "-f" | "--no-fault" | "-t"
                | "--tags" | "-v" | "--verbose" => {}
                "-h" | "--help" => {
                    println!("{USAGE}");
                    return;
                }
                other => eprintln!("g2g-launch-py: ignoring unrecognized option '{other}'"),
            }
            continue;
        }
        in_pipeline = true;
        tokens.push(arg);
    }
    let pipeline = tokens.join(" ");
    if pipeline.trim().is_empty() {
        eprintln!("{USAGE}");
        process::exit(2);
    }

    // Default WAYLAND_DISPLAY so `autovideosink` finds the compositor without the
    // caller exporting it; an explicit value always wins.
    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        std::env::set_var("WAYLAND_DISPLAY", "wayland-0");
        if !quiet {
            eprintln!("WAYLAND_DISPLAY unset, defaulting to wayland-0");
        }
    }

    // The one difference from `g2g-launch`: register the hosted Python elements
    // (`pyelement` / `pysrc` / `pyaggregator`) on top of the standard registry.
    let mut reg = default_registry();
    g2g_python::register(&mut reg);

    let graph = match parse_launch(&reg, &pipeline) {
        Ok(graph) => graph,
        Err(err) => {
            eprintln!("parse error: {err}");
            process::exit(1);
        }
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("build tokio runtime");

    if !quiet {
        println!("Setting pipeline to PLAYING ...");
    }
    let clock = WallClock::new();
    let progress = PipelineProgress::new();
    let started = Instant::now();
    let mut printed_status = false;
    let result = rt.block_on(async {
        let mut run = Box::pin(run_graph_with_progress(
            graph,
            &clock,
            LINK_CAPACITY,
            &progress,
        ));
        loop {
            match tokio::time::timeout(Duration::from_secs(1), &mut run).await {
                Ok(r) => break r,
                Err(_elapsed) => {
                    if !quiet {
                        let pos = match progress.position() {
                            Some(ns) => format!("t={:.1}s", ns as f64 / 1.0e9),
                            None => String::from("prerolling"),
                        };
                        eprint!(
                            "\r  running... {pos} ({:.0}s wall)   ",
                            started.elapsed().as_secs_f64()
                        );
                        use std::io::Write;
                        let _ = std::io::stderr().flush();
                        printed_status = true;
                    }
                }
            }
        }
    });
    if printed_status {
        eprintln!();
    }
    match result {
        Ok(stats) => {
            if !quiet {
                let elapsed = started.elapsed().as_secs_f64();
                print!("{}", stats.report());
                if elapsed > 0.0 {
                    println!(
                        "  run:     {:.2} s wall, {:.1} fps",
                        elapsed,
                        stats.frames_consumed as f64 / elapsed
                    );
                }
            }
        }
        Err(err) => {
            eprintln!("pipeline error: {err:?}");
            process::exit(1);
        }
    }
}

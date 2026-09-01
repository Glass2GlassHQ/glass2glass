//! `g2g-latency-reader`: reads raw I420 frames from standard input, decodes the
//! [`timestampburn`] strip in each one, and reports how long each frame took to
//! reach here from the moment the strip was written.
//!
//! Both sides of `tools/latency-bench-e2e.sh` pipe into this same binary, so the
//! g2g consumer and the GStreamer one are measured by identical code over an
//! identical span, on one machine's `CLOCK_MONOTONIC`.
//!
//! Usage:
//!   ... | g2g-latency-reader --width 1280 --height 720 --frames 300
//!
//! [`timestampburn`]: g2g_plugins::timestampburn

use std::io::Read;
use std::process::ExitCode;

use g2g_plugins::timestampburn::{decode, monotonic_ns};

const NS_PER_MS: f64 = 1_000_000.0;

fn usage() -> ! {
    eprintln!(
        "usage: g2g-latency-reader --width W --height H \
         [--frames N] [--warmup N] [--label TEXT]"
    );
    std::process::exit(2)
}

struct Args {
    width: usize,
    height: usize,
    frames: usize,
    warmup: usize,
    label: String,
}

fn parse_args() -> Args {
    let mut width = 0usize;
    let mut height = 0usize;
    let mut frames = usize::MAX;
    let mut warmup = 0usize;
    let mut label = String::from("latency");
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let Some(value) = argv.next() else { usage() };
        match flag.as_str() {
            "--width" => width = value.parse().unwrap_or_else(|_| usage()),
            "--height" => height = value.parse().unwrap_or_else(|_| usage()),
            "--frames" => frames = value.parse().unwrap_or_else(|_| usage()),
            "--warmup" => warmup = value.parse().unwrap_or_else(|_| usage()),
            "--label" => label = value,
            _ => usage(),
        }
    }
    if width == 0 || height == 0 {
        usage();
    }
    Args {
        width,
        height,
        frames,
        warmup,
        label,
    }
}

/// Nearest-rank percentile of an ascending slice.
fn percentile(sorted: &[u64], pct: usize) -> f64 {
    let index = (sorted.len() * pct / 100).min(sorted.len() - 1);
    sorted[index] as f64 / NS_PER_MS
}

fn main() -> ExitCode {
    let args = parse_args();
    // I420: a full-size luma plane then two half-size chroma planes.
    let frame_bytes = args.width * args.height * 3 / 2;
    let mut frame = vec![0u8; frame_bytes];
    let mut stdin = std::io::stdin().lock();

    let mut samples: Vec<u64> = Vec::new();
    let mut undecodable = 0usize;
    let mut read = 0usize;

    while read < args.warmup + args.frames {
        if stdin.read_exact(&mut frame).is_err() {
            break;
        }
        let now = monotonic_ns();
        read += 1;
        // The warmup covers decoder start-up and, on the g2g publisher, the
        // frames burned while the sink was still waiting for a player.
        if read <= args.warmup {
            continue;
        }
        match decode(&frame[..args.width * args.height], args.width) {
            Some(burned) => samples.push(now.saturating_sub(burned)),
            None => undecodable += 1,
        }
    }

    if samples.is_empty() {
        eprintln!(
            "{}: no frame carried a legible timestamp ({read} read, {undecodable} undecodable)",
            args.label
        );
        return ExitCode::FAILURE;
    }
    samples.sort_unstable();
    println!(
        "{}: n={} p50={:.2}ms p95={:.2}ms p99={:.2}ms max={:.2}ms undecodable={}",
        args.label,
        samples.len(),
        percentile(&samples, 50),
        percentile(&samples, 95),
        percentile(&samples, 99),
        *samples.last().expect("samples is non-empty") as f64 / NS_PER_MS,
        undecodable,
    );
    ExitCode::SUCCESS
}

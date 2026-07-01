//! `g2g-launch`: the `gst-launch` analog. Parses a text pipeline against the
//! standard element registry and runs it to completion.
//!
//! Usage:
//!   g2g-launch [OPTIONS] videotestsrc num-buffers=30 ! videoconvert ! fakesink
//!
//! Leading `gst-launch`-style options are accepted so pasted command lines run
//! verbatim (M191). The remaining arguments are joined with spaces into one
//! pipeline string, parsed by [`g2g_core::runtime::parse_launch`] (M106) against
//! [`g2g_plugins::registry::default_registry`] (M107), and driven on a
//! single-thread tokio runtime against the [`WallClock`]. Requires the `std`
//! feature (registry + wall clock are std-only).
//!
//! Options (the common `gst-launch-1.0` set):
//!   -v, --verbose       print the pipeline with each link's negotiated caps +
//!                       memory domain (falls back to topology if nego fails)
//!   -q, --quiet         suppress the PLAYING / Done progress lines
//!   --dot               dump the parsed pipeline as Graphviz DOT and exit
//!                       (pipe to `dot -Tsvg`); does not run the pipeline
//!   --plugin <path>     load a third-party plugin `.so` before parsing
//!                       (repeatable; needs the `plugin-loader` build feature)
//!   -h, --help          print this help and exit
//!   -e, --eos-on-shutdown, -m, --messages, -f, --no-fault, -t, --tags
//!                       accepted for compatibility (see notes below)
//!
//! Dynamic plugins: with the `plugin-loader` feature, every directory in
//! `$G2G_PLUGIN_PATH` plus each `--plugin <path>` is `dlopen`ed and its elements
//! registered before the pipeline parses, so a packaged binary extends without a
//! rebuild (M201). The plugin's ABI tag must match this build's.
//!
//! Compatibility notes: g2g sources run to their natural EOS (e.g.
//! `num-buffers`) or until the process is killed; there is no run-time
//! cancellation channel yet, so `-e` / `-m` / `-f` / `-t` are recognized and
//! ignored rather than rejected, keeping pasted lines parsing.
//!
//! Display: when built with `wayland-sink` and `WAYLAND_DISPLAY` is unset, it
//! defaults to `wayland-0` so `autovideosink` finds a compositor without the
//! caller exporting it; an existing value is left untouched.
//!
//! Debugging: `G2G_DEBUG` (the `GST_DEBUG` analog, e.g. `G2G_DEBUG=*:debug`)
//! sets per-category log thresholds; `G2G_CAPS_TRACE=1` turns on the
//! caps-negotiation explainer, which narrates the per-edge intersect / fixate
//! decisions (and, on a `CapsMismatch`, names the two conflicting elements and
//! the caps each wanted). Both are read by `g2g_core::log::init_from_env`.

use std::io::Write;
use std::process;
use std::time::{Duration, Instant};

use g2g_core::runtime::{parse_launch, run_graph_with_progress, PipelineProgress};
#[cfg(feature = "multi-thread")]
use g2g_core::runtime::run_graph_threaded_with_progress;
#[cfg(feature = "multi-thread")]
use g2g_plugins::TokioThreadSpawner;
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

// Steady-state link depth. Matches the integration-test default; small enough to
// keep latency low without starving the source (see DESIGN notes on
// link_capacity dominating glass-to-glass latency).
const LINK_CAPACITY: usize = 4;

const USAGE: &str = "usage: g2g-launch [-v] [-q] [--dot] [--threads] [--plugin <path>] [-e] [-m] [-h] \
<element> [key=value ...] ! <element> ! ...";

/// Parsed command-line options plus the leftover pipeline tokens.
#[derive(Default)]
struct Opts {
    verbose: bool,
    quiet: bool,
    help: bool,
    /// Dump the parsed pipeline as Graphviz DOT to stdout and exit without
    /// running it (`--dot`, the `GST_DEBUG_DUMP_DOT_DIR` analog).
    dot: bool,
    /// Plugin `.so` paths from `--plugin` (repeatable), loaded before parsing.
    plugins: Vec<String>,
    /// Run each element on its own OS thread (opt-in multicore, `--threads`),
    /// via `run_graph_threaded`. Off by default: cooperative single-thread has
    /// lower per-frame latency; this trades a per-stage thread handoff for
    /// CPU-bound stages overlapping across cores. Needs the `multi-thread` build.
    threads: bool,
}

/// Split leading `gst-launch`-style flags off the front of the args, returning
/// the options and the remaining pipeline tokens. Only leading `-`/`--` tokens
/// are treated as flags, so a negative property value (always part of a
/// `key=value` token, e.g. `videobox top=-5`) is never mistaken for one. An
/// unrecognized leading flag is warned about and skipped rather than aborting,
/// so an unusual paste still runs.
fn parse_opts(args: impl Iterator<Item = String>) -> (Opts, Vec<String>) {
    let mut opts = Opts::default();
    let mut rest: Vec<String> = Vec::new();
    let mut in_pipeline = false;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        if in_pipeline || !arg.starts_with('-') || arg == "-" {
            in_pipeline = true;
            rest.push(arg);
            continue;
        }
        // `--plugin <path>` and `--plugin=<path>` both supply a value.
        if let Some(path) = arg.strip_prefix("--plugin=") {
            opts.plugins.push(path.to_string());
            continue;
        }
        match arg.as_str() {
            "-v" | "--verbose" => opts.verbose = true,
            "-q" | "--quiet" => opts.quiet = true,
            "-h" | "--help" => opts.help = true,
            "--dot" => opts.dot = true,
            "--threads" => opts.threads = true,
            "--plugin" => match args.next() {
                Some(path) => opts.plugins.push(path),
                None => eprintln!("g2g-launch: --plugin needs a path argument"),
            },
            // Accepted for compatibility (see the module-level notes): these
            // govern live shutdown / bus output, which g2g does not yet expose
            // a run-time channel for. Recognized and ignored so the line runs.
            "-e" | "--eos-on-shutdown" | "-m" | "--messages" | "-f" | "--no-fault"
            | "-t" | "--tags" => {}
            other => eprintln!("g2g-launch: ignoring unrecognized option '{other}'"),
        }
    }
    (opts, rest)
}

/// Load dynamic plugins (`$G2G_PLUGIN_PATH` directories + each `--plugin` path)
/// into `reg` before parsing, so their elements resolve by name. A load failure
/// is fatal: a pipeline naming a plugin element would otherwise fail later with
/// a more confusing "unknown element". Compiled out without `plugin-loader`,
/// where a `--plugin` request is reported rather than silently ignored.
#[cfg(feature = "plugin-loader")]
fn load_plugins(reg: &mut g2g_core::runtime::Registry, plugins: &[String]) {
    use g2g_plugins::plugin_loader;
    match plugin_loader::load_from_env(reg) {
        Ok(loaded) => {
            for p in loaded {
                eprintln!("g2g-launch: loaded plugin {}", p.display());
            }
        }
        Err(err) => {
            eprintln!("g2g-launch: {err}");
            process::exit(1);
        }
    }
    for path in plugins {
        if let Err(err) = plugin_loader::load_plugin(path, reg) {
            eprintln!("g2g-launch: {err}");
            process::exit(1);
        }
        eprintln!("g2g-launch: loaded plugin {path}");
    }
}

#[cfg(not(feature = "plugin-loader"))]
fn load_plugins(_reg: &mut g2g_core::runtime::Registry, plugins: &[String]) {
    if !plugins.is_empty() || std::env::var_os("G2G_PLUGIN_PATH").is_some() {
        eprintln!(
            "g2g-launch: built without the `plugin-loader` feature; \
             --plugin / $G2G_PLUGIN_PATH ignored"
        );
    }
}

fn main() {
    // Honor G2G_DEBUG (the GST_DEBUG analog): install the stderr log sink and
    // apply the category thresholds before the pipeline runs.
    g2g_core::log::init_from_env();

    let (opts, tokens) = parse_opts(std::env::args().skip(1));
    if opts.help {
        println!("{USAGE}");
        return;
    }
    let pipeline = tokens.join(" ");
    if pipeline.trim().is_empty() {
        eprintln!("{USAGE}");
        process::exit(2);
    }

    // Default `WAYLAND_DISPLAY` to the conventional `wayland-0` socket when it is
    // unset, so `autovideosink` / `waylandsink` find a compositor without the
    // caller exporting it first. Only when the display sink is compiled in; an
    // explicit env value always wins. If `wayland-0` is wrong, the sink fails the
    // same way it would have with no value set.
    #[cfg(feature = "wayland-sink")]
    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        std::env::set_var("WAYLAND_DISPLAY", "wayland-0");
        if !opts.quiet {
            eprintln!("WAYLAND_DISPLAY unset, defaulting to wayland-0");
        }
    }

    let mut reg = default_registry();
    load_plugins(&mut reg, &opts.plugins);
    let graph = match parse_launch(&reg, &pipeline) {
        Ok(graph) => graph,
        Err(err) => {
            eprintln!("parse error: {err}");
            // Add a gst->g2g porting hint when one applies.
            let report = g2g_plugins::gst_compat::lint_launch(&reg, &pipeline);
            for hint in &report.findings {
                eprintln!("  hint: {hint}");
            }
            process::exit(1);
        }
    };

    if opts.dot {
        // Dump the pipeline as Graphviz DOT and exit. Negotiate first (probe
        // source caps + solve the whole graph) so each edge carries the chosen
        // caps; on a negotiation failure fall back to a topology-only dump
        // (re-parsing, since negotiation consumed the graph). Each node is
        // labelled with its element's log category; a tee falls back to its kind.
        // `enable_all`: negotiation probes source caps, and a network source's
        // probe opens sockets (see the run path below).
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build tokio runtime");
        match rt.block_on(g2g_core::runtime::negotiate_graph(graph)) {
            Ok((vg, caps, memory)) => print!(
                "{}",
                vg.to_dot(
                    "pipeline",
                    |n| vg.element(n).map(|e| e.log_category().to_string()),
                    &g2g_core::DotAnnotations {
                        edge_caps: Some(&caps),
                        edge_memory: Some(&memory),
                    },
                )
            ),
            Err(err) => {
                eprintln!("g2g-launch: negotiation failed ({err:?}); dumping topology only");
                let graph = match parse_launch(&reg, &pipeline) {
                    Ok(g) => g,
                    Err(e) => {
                        eprintln!("parse error: {e}");
                        process::exit(1);
                    }
                };
                print!(
                    "{}",
                    graph.to_dot(
                        "pipeline",
                        |n| graph.element(n).map(|e| e.log_category().to_string()),
                        &g2g_core::DotAnnotations::default(),
                    )
                );
            }
        }
        return;
    }

    if opts.verbose {
        eprintln!("pipeline: {pipeline}");
        // Show each link's negotiated caps + memory domain (gst `-v` style). The
        // solve lives in `negotiate_graph`, which consumes a graph, so negotiate a
        // freshly-parsed throwaway copy and keep `graph` for the run. On any
        // negotiation failure, fall back to the topology-only wiring dump.
        let negotiated = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()
            .and_then(|rt| {
                let fresh = parse_launch(&reg, &pipeline).ok()?;
                rt.block_on(g2g_core::runtime::negotiate_graph(fresh)).ok()
            });
        match negotiated {
            Some((vg, caps, memory)) => {
                let name = |n: g2g_core::NodeId| {
                    vg.element(n)
                        .map(|el| el.log_category().to_string())
                        .unwrap_or_else(|| format!("n{}", n.0))
                };
                eprintln!("negotiated links ({}):", vg.edge_count());
                for id in 0..vg.edge_count() {
                    let e = vg.edge(id);
                    let caps = caps.get(id).map_or_else(|| "?".to_string(), |c| c.to_gst_string());
                    let mem = memory.get(id).copied().unwrap_or(g2g_core::MemoryDomainKind::System);
                    eprintln!(
                        "  [{id}] {} -> {} : {caps}  mem={mem:?} policy={:?}",
                        name(e.src.node),
                        name(e.dst.node),
                        e.policy
                    );
                }
            }
            None => {
                eprintln!(
                    "links ({}) [negotiation unavailable, topology only]:",
                    graph.edges().len()
                );
                for (i, e) in graph.edges().iter().enumerate() {
                    eprintln!("  [{i}] {:?} -> {:?}  policy={:?}", e.src, e.dst, e.policy);
                }
            }
        }
    }

    // `enable_all` (IO + time), not just time: a network source (HlsSrc, RtspSrc,
    // an http source) opens sockets from the runner task on this ambient runtime,
    // which panics ("IO disabled") under a time-only runtime. Time alone suffices
    // for purely local pipelines, but enabling IO costs nothing and is required the
    // moment a `uri=` resolves to the network.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    if !opts.quiet {
        println!("Setting pipeline to PLAYING ...");
    }
    let clock = WallClock::new();
    let progress = PipelineProgress::new();
    let started = Instant::now();
    // Poll the run future with a 1s timeout so a long-running (e.g. forever, the
    // default `videotestsrc`) pipeline prints a liveness heartbeat instead of
    // looking hung. `timeout` only needs tokio's `time` feature (no `select!`
    // macro). A short pipeline finishes inside the first tick, so it stays quiet.
    let mut printed_status = false;
    let result = rt.block_on(async {
        // `--threads` runs one OS thread per arm (opt-in multicore); the default
        // is the cooperative single-thread runner. Both return the same
        // `Result<RunStats, _>`, boxed to one type so the heartbeat loop is shared.
        type RunFut<'r> = core::pin::Pin<
            Box<dyn core::future::Future<Output = Result<g2g_core::runtime::RunStats, g2g_core::G2gError>> + 'r>,
        >;
        let mut run: RunFut = if opts.threads {
            #[cfg(feature = "multi-thread")]
            {
                Box::pin(run_graph_threaded_with_progress(
                    graph,
                    &clock,
                    LINK_CAPACITY,
                    &progress,
                    &TokioThreadSpawner,
                ))
            }
            #[cfg(not(feature = "multi-thread"))]
            {
                let _ = graph;
                eprintln!(
                    "pipeline error: --threads requires a multi-thread build \
                     (rebuild with --features multi-thread)"
                );
                process::exit(1);
            }
        } else {
            Box::pin(run_graph_with_progress(graph, &clock, LINK_CAPACITY, &progress))
        };
        loop {
            match tokio::time::timeout(Duration::from_secs(1), &mut run).await {
                Ok(r) => break r,
                Err(_elapsed) => {
                    if !opts.quiet {
                        let pos = match progress.position() {
                            Some(ns) => format!("t={:.1}s", ns as f64 / 1.0e9),
                            None => String::from("prerolling"),
                        };
                        eprint!("\r  running... {pos} ({:.0}s wall)   ", started.elapsed().as_secs_f64());
                        let _ = std::io::stderr().flush();
                        printed_status = true;
                    }
                }
            }
        }
    });
    if printed_status {
        eprintln!(); // move off the \r status line before the summary
    }
    match result {
        Ok(stats) => {
            if !opts.quiet {
                // End-of-run report (M287): the RunStats telemetry plus the
                // measured wall-clock throughput this run achieved.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn leading_flags_split_from_pipeline() {
        let (opts, rest) = parse_opts(toks(&["-v", "-e", "videotestsrc", "!", "fakesink"]).into_iter());
        assert!(opts.verbose);
        assert!(!opts.quiet);
        assert_eq!(rest, toks(&["videotestsrc", "!", "fakesink"]));
    }

    #[test]
    fn negative_property_value_is_not_a_flag() {
        // `top=-5` starts the pipeline; a later `-5`-looking token stays put
        // because once a non-flag token is seen, everything after is pipeline.
        let (opts, rest) = parse_opts(toks(&["videobox", "top=-5", "!", "fakesink"]).into_iter());
        assert!(!opts.verbose && !opts.quiet && !opts.help);
        assert_eq!(rest, toks(&["videobox", "top=-5", "!", "fakesink"]));
    }

    #[test]
    fn combined_long_flags_and_quiet() {
        let (opts, rest) = parse_opts(toks(&["--quiet", "--verbose", "fakesink"]).into_iter());
        assert!(opts.quiet && opts.verbose);
        assert_eq!(rest, toks(&["fakesink"]));
    }

    #[test]
    fn dot_flag_splits_from_pipeline() {
        let (opts, rest) = parse_opts(toks(&["--dot", "videotestsrc", "!", "fakesink"]).into_iter());
        assert!(opts.dot);
        assert!(!opts.verbose);
        assert_eq!(rest, toks(&["videotestsrc", "!", "fakesink"]));
    }
}

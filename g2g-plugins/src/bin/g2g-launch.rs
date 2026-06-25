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
//!   -v, --verbose       print the parsed pipeline (elements + link policies)
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
//! Debugging: `G2G_DEBUG` (the `GST_DEBUG` analog, e.g. `G2G_DEBUG=*:debug`)
//! sets per-category log thresholds; `G2G_CAPS_TRACE=1` turns on the
//! caps-negotiation explainer, which narrates the per-edge intersect / fixate
//! decisions (and, on a `CapsMismatch`, names the two conflicting elements and
//! the caps each wanted). Both are read by `g2g_core::log::init_from_env`.

use std::process;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

// Steady-state link depth. Matches the integration-test default; small enough to
// keep latency low without starving the source (see DESIGN notes on
// link_capacity dominating glass-to-glass latency).
const LINK_CAPACITY: usize = 4;

const USAGE: &str = "usage: g2g-launch [-v] [-q] [--dot] [--plugin <path>] [-e] [-m] [-h] \
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
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
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
        // Best-effort `-v`: report the wiring the parse produced (link count and
        // each edge's backpressure policy). Per-pad negotiated caps are not
        // surfaced from the runner yet, so this stops short of gst's caps dump.
        eprintln!("pipeline: {pipeline}");
        eprintln!("links ({}):", graph.edges().len());
        for (i, e) in graph.edges().iter().enumerate() {
            eprintln!("  [{i}] {:?} -> {:?}  policy={:?}", e.src, e.dst, e.policy);
        }
    }

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("build tokio runtime");

    if !opts.quiet {
        println!("Setting pipeline to PLAYING ...");
    }
    let clock = WallClock::new();
    let started = std::time::Instant::now();
    match rt.block_on(run_graph(graph, &clock, LINK_CAPACITY)) {
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

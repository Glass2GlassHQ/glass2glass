//! `g2g-launch`: the `gst-launch` analog. Parses a text pipeline against the
//! standard element registry and runs it to completion.
//!
//! Usage:
//!
//! ```text
//! g2g-launch [OPTIONS] videotestsrc num-buffers=30 ! videoconvert ! fakesink
//! ```
//!
//! Leading `gst-launch`-style options are accepted so pasted command lines run
//! verbatim. The remaining arguments are joined with spaces into one
//! pipeline string, parsed by [`g2g_core::runtime::parse_launch`] against
//! [`g2g_plugins::registry::default_registry`], and driven on a
//! single-thread tokio runtime against the [`WallClock`]. Requires the `std`
//! feature (registry + wall clock are std-only).
//!
//! Options (the common `gst-launch-1.0` set):
//!
//! ```text
//! -v, --verbose       print the pipeline with each link's negotiated caps +
//!                     memory domain (falls back to topology if nego fails),
//!                     and each sink's frame-identity report while it runs
//! -q, --quiet         suppress the PLAYING / Done progress lines
//! --dot               dump the parsed pipeline as Graphviz DOT and exit
//!                     (pipe to `dot -Tsvg`); does not run the pipeline
//! --copy-plan         negotiate and print the memory-domain copy plan (per-hop
//!                     domain trace + transfers + zero-copy verdict), then exit
//! --validate-json     negotiate and print the result as JSON (every node and
//!                     each edge's negotiated caps, or the structured failure),
//!                     then exit; does not run the pipeline
//! --run-json          run the pipeline to EOS and print the same JSON with
//!                     each edge's caps as observed while it ran, so a stream
//!                     that refines mid-run reports its real geometry
//! --tui               run the pipeline under a live terminal UI (element
//!                     latency, link traffic, topology, frame journey);
//!                     needs the `tui` build feature
//! --record-on-error <dir>
//!                     keep a bounded ring of recent packets per link while the
//!                     run goes, and on failure write each link's ring into
//!                     <dir> as a `replaysrc` recording
//! --trace <file>      write each element's per-frame wait and work spans to
//!                     <file> as Chrome trace JSON, for ui.perfetto.dev
//! --seek <seconds>    one flushing seek to that stream time before the first
//!                     frame, then run to the end (`--help` says where the
//!                     first frame lands)
//! --plugin <path>     load a third-party plugin `.so` before parsing
//!                     (repeatable; needs the `plugin-loader` build feature)
//! --graph <file>      build the graph from a declarative JSON / YAML document
//!                     instead of a text pipeline (`declarative` /
//!                     `declarative-yaml` build feature)
//! --script <file>     build the graph from a Rhai script that calls the
//!                     builder API (`script-rhai` build feature)
//! -h, --help          print this help and exit
//! --version           print the version and exit
//! -e, --eos-on-shutdown, -m, --messages, -f, --no-fault, -t, --tags
//!                     accepted for compatibility (see notes below)
//! ```
//!
//! Dynamic plugins: with the `plugin-loader` feature, every directory in
//! `$G2G_PLUGIN_PATH` plus each `--plugin <path>` is `dlopen`ed and its elements
//! registered before the pipeline parses, so a packaged binary extends without a
//! rebuild. The plugin's ABI tag must match this build's. With the
//! `plugin-signing` feature, listing Ed25519 public key files in
//! `$G2G_PLUGIN_TRUSTED_KEYS` makes a matching `<plugin>.sig` mandatory for
//! every one of them.
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
//! Debugging: a failed negotiation names the two conflicting elements and the
//! caps each wanted by default (no env var). `G2G_DEBUG` (the `GST_DEBUG` analog,
//! e.g. `G2G_DEBUG=*:debug`) sets per-category log thresholds; `G2G_CAPS_TRACE=1`
//! adds the successful per-edge intersect / fixate narration on top. Both are
//! read by `g2g_core::log::init_from_env`.

use std::future::Future;
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::process;
use std::sync::Arc;
use std::task::Poll;
use std::time::{Duration, Instant};

#[cfg(feature = "multi-thread")]
use g2g_core::element::BoxFuture;
use g2g_core::runtime::{
    attach_seek, fallback_factory, has_live_source, parse_launch, parse_launch_avoiding,
    FlightRecorder, GraphNode, PipelineProgress, Registry, RunBuilder,
};
#[cfg(any(feature = "observe", feature = "tui"))]
use g2g_core::runtime::{run_graph_observed, run_graph_observed_recorded, Observer};
#[cfg(feature = "multi-thread")]
use g2g_core::runtime::{GraphSpawner, LocalArmFuture};
use g2g_core::{Bus, BusMessage, Seek};
use g2g_core::{Graph, NodeKind};
use g2g_plugins::cli_stdout::write_stdout;
use g2g_plugins::clock::WallClock;
use g2g_plugins::perfetto_trace::PerfettoTrace;
use g2g_plugins::registry::default_registry;
#[cfg(feature = "multi-thread")]
use g2g_plugins::{PinnedThreadSpawner, TokioThreadSpawner};

macro_rules! print_stdout {
    ($($arg:tt)*) => { write_stdout(format_args!($($arg)*)) };
}

macro_rules! println_stdout {
    ($($arg:tt)*) => { write_stdout(format_args!("{}\n", format_args!($($arg)*))) };
}

// Steady-state link depth. Matches the integration-test default; small enough to
// keep latency low without starving the source (see design/README.md on
// link_capacity dominating glass-to-glass latency).
const LINK_CAPACITY: usize = 4;

/// Bus depth for a run: only the tail matters here (the message naming what
/// ended the run), and a slow reader must not stall an element posting one.
const BUS_DEPTH: usize = 64;
const NANOS_PER_MILLISECOND: f64 = 1.0e6;
const STATUS_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, Copy)]
enum ArmThreads {
    Free,
    Pinned,
}

#[cfg(feature = "multi-thread")]
#[derive(Debug)]
enum LaunchSpawner {
    Free(TokioThreadSpawner),
    Pinned(PinnedThreadSpawner),
}

#[cfg(feature = "multi-thread")]
impl GraphSpawner for LaunchSpawner {
    fn spawn_arm(
        &self,
        build: Box<dyn FnOnce() -> LocalArmFuture + Send>,
    ) -> BoxFuture<'static, Result<u64, g2g_core::G2gError>> {
        match self {
            Self::Free(spawner) => spawner.spawn_arm(build),
            Self::Pinned(spawner) => spawner.spawn_arm(build),
        }
    }
}

/// A boxed pipeline run, so the runner entry a flag selects (recorded or not,
/// cooperative or thread-per-arm) is one type the driving loop can hold.
type RunFuture<'r> = core::pin::Pin<
    Box<
        dyn core::future::Future<Output = Result<g2g_core::runtime::RunStats, g2g_core::G2gError>>
            + 'r,
    >,
>;

const USAGE: &str = "usage: g2g-launch [-v] [-q] [--dot] [--copy-plan] [--validate-json] [--run-json] [--threads[=pinned]] [--observe <port>] [--observe-host <addr>] [--tui] [--record-on-error <dir>] [--trace <file>] [--seek <seconds>] [--plugin <path>] [-e] [-m] [-h] [--version] \
<element> [key=value ...] ! <element> ! ...\n       \
g2g-launch [OPTIONS] --graph <file.json|.yaml>   # declarative graph\n       \
g2g-launch [OPTIONS] --script <file.rhai>         # Rhai graph-building script";

const SEEK_HELP: &str = "\n\
--seek <seconds>: one flushing seek to that stream time before the first frame, then run to the end.\n  \
An MP4 file under uridecodebin, or a demuxer after filesrc (qtdemux, tsdemux, matroskademux,\n  \
flvdemux, oggdemux, fmp4demux, also when decodebin plugs it), restarts at the keyframe at or before\n  \
the target and the decoder drops the frames before it, so the first decoded frame is the first one\n  \
at or after the target (ffmpeg -ss). A target past the end fails the run.";

// Printed on `--help` (not on the terse empty-invocation error): a map of the
// rest of the dev tooling, so a `gst-launch` reflex finds the ecosystem. Full
// reference in DEVTOOLS.md.
const SEE_ALSO: &str = "\n\
dev tooling (see DEVTOOLS.md):\n  \
watch a run live      g2g-launch --observe 8787 <pipeline>   then open http://127.0.0.1:8787\n  \
watch it in a shell   g2g-launch --tui <pipeline>             live tables + ASCII topology, q to quit\n  \
why did nego fail     a failed run prints the conflicting elements + caps by default; G2G_CAPS_TRACE=1 adds the per-edge trace\n  \
see the graph         g2g-launch --dot <pipeline> | dot -Tsvg -o pipe.svg\n  \
inspect an element    g2g-inspect [<name>]                    role, caps, and every property\n  \
build it visually     tools/builder/ (React Flow: assemble, import, export a pipeline)\n  \
record / replay       <pipeline> ! recordsink location=cap.g2g   then   replaysrc location=cap.g2g ! ...\n  \
repro a crash         g2g-launch --record-on-error dumps/ <pipeline>   then   replaysrc location=dumps/<file> ! ...\n  \
time every frame      g2g-launch --trace run.json <pipeline>  then open run.json in https://ui.perfetto.dev";

/// Parsed command-line options plus the leftover pipeline tokens.
#[derive(Default)]
struct Opts {
    verbose: bool,
    quiet: bool,
    help: bool,
    version: bool,
    /// Dump the parsed pipeline as Graphviz DOT to stdout and exit without
    /// running it (`--dot`, the `GST_DEBUG_DUMP_DOT_DIR` analog).
    dot: bool,
    /// Negotiate and print the memory-domain copy / allocation plan, then exit
    /// without running (`--copy-plan`): the per-hop domain trace, the transfers,
    /// and whether the graph is zero-copy.
    copy_plan: bool,
    /// Negotiate and print the JSON validate dump (`--validate-json`): every
    /// node and each edge's negotiated caps, or the structured failure. The
    /// machine-readable sibling of `--dot`, for a tool that diffs pipelines.
    validate_json: bool,
    /// Run the pipeline to EOS and print the validate dump with each edge's
    /// caps as observed while it ran (`--run-json`), so a stream whose geometry
    /// only arrives with the data reports what it really carried.
    run_json: bool,
    /// Plugin `.so` paths from `--plugin` (repeatable), loaded before parsing.
    plugins: Vec<String>,
    /// Build the graph from a declarative JSON / YAML file (`--graph <path>`)
    /// instead of a text pipeline. `.json` -> `from_json`, `.yaml` / `.yml`
    /// -> `from_yaml`. Needs the `declarative` (and, for YAML, `declarative-yaml`)
    /// build feature.
    graph: Option<String>,
    /// Build the graph from a Rhai script file (`--script <path>`) that
    /// calls the builder API. Needs the `script-rhai` build feature.
    script: Option<String>,
    /// Run each element on its own OS thread (opt-in multicore, `--threads`),
    /// via `run_graph_threaded`. Off by default: cooperative single-thread has
    /// lower per-frame latency; this trades a per-stage thread handoff for
    /// CPU-bound stages overlapping across cores. Needs the `multi-thread` build.
    /// `--threads=pinned` also pins each arm's thread to a core of its own.
    threads: Option<ArmThreads>,
    /// Run the pipeline while serving the live dashboard on this port
    /// (`--observe <port>`): telemetry + bus events stream over a WebSocket, and
    /// the dashboard page is served on the same port. Needs the `observe` build.
    observe: Option<u16>,
    /// Bind address for `--observe` (`--observe-host <addr>`); default
    /// `127.0.0.1` (loopback only). Set `0.0.0.0` to serve on all interfaces so
    /// another machine can reach the dashboard. The dashboard has no auth and its
    /// telemetry / edge previews expose frame content, so only bind non-loopback
    /// on a trusted network.
    observe_host: Option<String>,
    /// Run the pipeline under the in-terminal live UI (`--tui`): the same
    /// `Observer` telemetry the dashboard serves, drawn in this terminal
    /// instead of a browser. Needs the `tui` build.
    tui: bool,
    /// Directory for the flight recorder's dump (`--record-on-error <dir>`):
    /// while the run goes, every link keeps a bounded ring of its most
    /// recent packets; if the run fails, each ring is written there as a
    /// `replaysrc` recording of the traffic that led to the failure.
    record_on_error: Option<String>,
    trace: Option<String>,
    seek_seconds: Option<String>,
}

/// Parse a `--observe` port, warning and returning `None` on a bad value.
fn parse_port(s: &str) -> Option<u16> {
    match s.parse::<u16>() {
        Ok(p) if p != 0 => Some(p),
        _ => {
            eprintln!("g2g-launch: --observe needs a port in 1..=65535, got '{s}'");
            None
        }
    }
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
        if let Some(path) = arg.strip_prefix("--graph=") {
            opts.graph = Some(path.to_string());
            continue;
        }
        if let Some(path) = arg.strip_prefix("--script=") {
            opts.script = Some(path.to_string());
            continue;
        }
        if let Some(port) = arg.strip_prefix("--observe=") {
            opts.observe = parse_port(port);
            continue;
        }
        if let Some(host) = arg.strip_prefix("--observe-host=") {
            opts.observe_host = Some(host.to_string());
            continue;
        }
        if let Some(dir) = arg.strip_prefix("--record-on-error=") {
            opts.record_on_error = Some(dir.to_string());
            continue;
        }
        if let Some(path) = arg.strip_prefix("--trace=") {
            opts.trace = Some(path.to_string());
            continue;
        }
        if let Some(seconds) = arg.strip_prefix("--seek=") {
            opts.seek_seconds = Some(seconds.to_string());
            continue;
        }
        match arg.as_str() {
            "-v" | "--verbose" => opts.verbose = true,
            "-q" | "--quiet" => opts.quiet = true,
            "-h" | "--help" => opts.help = true,
            "--version" => opts.version = true,
            "--dot" => opts.dot = true,
            "--copy-plan" => opts.copy_plan = true,
            "--validate-json" => opts.validate_json = true,
            "--run-json" => opts.run_json = true,
            "--threads" | "--threads=auto" => opts.threads = Some(ArmThreads::Free),
            "--threads=pinned" => opts.threads = Some(ArmThreads::Pinned),
            "--tui" => opts.tui = true,
            "--plugin" => match args.next() {
                Some(path) => opts.plugins.push(path),
                None => eprintln!("g2g-launch: --plugin needs a path argument"),
            },
            "--graph" => match args.next() {
                Some(path) => opts.graph = Some(path),
                None => eprintln!("g2g-launch: --graph needs a path argument"),
            },
            "--script" => match args.next() {
                Some(path) => opts.script = Some(path),
                None => eprintln!("g2g-launch: --script needs a path argument"),
            },
            "--observe" => match args.next() {
                Some(port) => opts.observe = parse_port(&port),
                None => eprintln!("g2g-launch: --observe needs a port argument"),
            },
            "--observe-host" => match args.next() {
                Some(host) => opts.observe_host = Some(host),
                None => eprintln!("g2g-launch: --observe-host needs an address argument"),
            },
            "--record-on-error" => match args.next() {
                Some(dir) => opts.record_on_error = Some(dir),
                None => eprintln!("g2g-launch: --record-on-error needs a directory argument"),
            },
            "--trace" => match args.next() {
                Some(path) => opts.trace = Some(path),
                None => {
                    eprintln!("g2g-launch: --trace needs a file argument");
                    process::exit(2);
                }
            },
            "--seek" => match args.next() {
                Some(seconds) => opts.seek_seconds = Some(seconds),
                None => {
                    eprintln!("g2g-launch: --seek needs a time in seconds");
                    process::exit(2);
                }
            },
            // Accepted for compatibility (see the module-level notes): these
            // govern live shutdown / bus output, which g2g does not yet expose
            // a run-time channel for. Recognized and ignored so the line runs.
            "-e" | "--eos-on-shutdown" | "-m" | "--messages" | "-f" | "--no-fault" | "-t"
            | "--tags" => {}
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
#[cfg(all(feature = "plugin-loader", not(feature = "plugin-signing")))]
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

/// The same with signature verification: the keys
/// `$G2G_PLUGIN_TRUSTED_KEYS` names gate the scanned directories and each
/// explicit `--plugin` alike, so neither route is the unverified one.
#[cfg(feature = "plugin-signing")]
fn load_plugins(reg: &mut g2g_core::runtime::Registry, plugins: &[String]) {
    use g2g_plugins::plugin_loader::{self, default_policy};
    let fatal = |err: &dyn std::fmt::Display| -> ! {
        eprintln!("g2g-launch: {err}");
        process::exit(1)
    };
    let trusted = plugin_loader::trusted_keys_from_env().unwrap_or_else(|e| fatal(&e));
    match plugin_loader::load_from_env_verified(reg, &trusted) {
        Ok(loaded) => {
            for p in loaded {
                eprintln!("g2g-launch: loaded plugin {}", p.display());
            }
        }
        Err(err) => fatal(&err),
    }
    for path in plugins {
        if let Err(err) = plugin_loader::load_plugin_verified(path, reg, &trusted, &default_policy)
        {
            fatal(&err);
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

/// Build the graph from whichever source the options select: a declarative file
/// (`--graph`), a Rhai script file (`--script`), or the joined text `pipeline`.
/// Callable more than once (the `--dot` / `--verbose` paths build a throwaway
/// copy to negotiate), so it takes only borrows. Errors are pre-formatted.
fn build_graph(reg: &Registry, opts: &Opts, pipeline: &str) -> Result<Graph<GraphNode>, String> {
    if let Some(path) = &opts.graph {
        load_graph_file(reg, path)
    } else if let Some(path) = &opts.script {
        load_script_file(reg, path)
    } else {
        parse_launch(reg, pipeline).map_err(|e| {
            let mut message = format!("parse error: {e}");
            // gst->g2g porting guidance, from the error already in hand (a
            // re-parse would repeat side effects like the uridecodebin file
            // probe, logging the same unreadable path twice).
            for hint in g2g_plugins::gst_compat::explain_parse_error(reg, pipeline, &e) {
                message.push_str("\n  hint: ");
                message.push_str(&hint);
            }
            message
        })
    }
}

/// Load a declarative JSON / YAML graph file (`--graph`). Compiled to a clear
/// error without the `declarative` feature.
#[cfg(feature = "declarative")]
fn load_graph_file(reg: &Registry, path: &str) -> Result<Graph<GraphNode>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read '{path}': {e}"))?;
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".yaml") || lower.ends_with(".yml") {
        #[cfg(feature = "declarative-yaml")]
        {
            return g2g_plugins::declarative::from_yaml(reg, &text).map_err(|e| e.to_string());
        }
        #[cfg(not(feature = "declarative-yaml"))]
        {
            let _ = reg;
            return Err("g2g-launch: YAML graphs need the `declarative-yaml` build feature".into());
        }
    }
    g2g_plugins::declarative::from_json(reg, &text).map_err(|e| e.to_string())
}

#[cfg(not(feature = "declarative"))]
fn load_graph_file(_reg: &Registry, _path: &str) -> Result<Graph<GraphNode>, String> {
    Err("g2g-launch: --graph needs the `declarative` build feature".into())
}

/// Load a Rhai graph-building script file (`--script`). Compiled to a clear
/// error without the `script-rhai` feature.
#[cfg(feature = "script-rhai")]
fn load_script_file(reg: &Registry, path: &str) -> Result<Graph<GraphNode>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read '{path}': {e}"))?;
    g2g_plugins::script::build_from_script(reg, &text).map_err(|e| e.to_string())
}

#[cfg(not(feature = "script-rhai"))]
fn load_script_file(_reg: &Registry, _path: &str) -> Result<Graph<GraphNode>, String> {
    Err("g2g-launch: --script needs the `script-rhai` build feature".into())
}

/// Print the `--validate-json` dump (negotiate without running) and return the
/// process exit code: 0 when the pipeline negotiates, 1 when it does not (the
/// JSON still describes the failure).
#[cfg(feature = "tooling-json")]
fn print_validate_json(reg: &Registry, pipeline: &str) -> i32 {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    let value = rt.block_on(g2g_plugins::toolingjson::validate_json(reg, pipeline));
    println_stdout!("{value}");
    if value["ok"].as_bool().unwrap_or(false) {
        0
    } else {
        1
    }
}

#[cfg(not(feature = "tooling-json"))]
fn print_validate_json(_reg: &Registry, _pipeline: &str) -> i32 {
    eprintln!("g2g-launch: --validate-json needs the `tooling-json` build feature");
    2
}

/// Print the `--run-json` dump: the same shape as `--validate-json`, but the
/// pipeline is run to EOS first and each edge reports the caps that crossed it.
#[cfg(feature = "tooling-json")]
fn print_run_json(reg: &Registry, pipeline: &str) -> i32 {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    let value = rt.block_on(g2g_plugins::toolingjson::observed_graph_json(reg, pipeline));
    println_stdout!("{value}");
    if value["ok"].as_bool().unwrap_or(false) {
        0
    } else {
        1
    }
}

#[cfg(not(feature = "tooling-json"))]
fn print_run_json(_reg: &Registry, _pipeline: &str) -> i32 {
    eprintln!("g2g-launch: --run-json needs the `tooling-json` build feature");
    2
}

fn main() {
    // Honor G2G_DEBUG (the GST_DEBUG analog): install the stderr log sink and
    // apply the category thresholds before the pipeline runs.
    g2g_core::log::init_from_env();

    let (opts, tokens) = parse_opts(std::env::args().skip(1));
    if opts.help {
        println_stdout!("{USAGE}");
        println_stdout!("{SEEK_HELP}");
        println_stdout!("{SEE_ALSO}");
        return;
    }
    if opts.version {
        println_stdout!("g2g-launch {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    let seek_ns = match opts.seek_seconds.as_deref().map(seek_target_ns).transpose() {
        Ok(seek_ns) => seek_ns,
        Err(err) => {
            eprintln!("g2g-launch: {err}");
            process::exit(2);
        }
    };
    let skips_the_run = opts.dot
        || opts.copy_plan
        || opts.validate_json
        || opts.run_json
        || opts.observe.is_some()
        || opts.tui;
    for (flag, given) in [
        ("--seek", seek_ns.is_some()),
        ("--trace", opts.trace.is_some()),
    ] {
        if given && skips_the_run {
            eprintln!(
                "g2g-launch: {flag} takes the plain run, not --dot / --copy-plan / --validate-json / \
                 --run-json / --observe / --tui"
            );
            process::exit(2);
        }
    }
    let pipeline = tokens.join(" ");
    let use_file = opts.graph.is_some() || opts.script.is_some();
    if !use_file && pipeline.trim().is_empty() {
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
    }

    let mut reg = default_registry();
    load_plugins(&mut reg, &opts.plugins);

    if opts.validate_json {
        if use_file {
            eprintln!("g2g-launch: --validate-json takes a text pipeline, not --graph / --script");
            process::exit(2);
        }
        process::exit(print_validate_json(&reg, &pipeline));
    }

    if opts.run_json {
        if use_file {
            eprintln!("g2g-launch: --run-json takes a text pipeline, not --graph / --script");
            process::exit(2);
        }
        process::exit(print_run_json(&reg, &pipeline));
    }

    let graph = match build_graph(&reg, &opts, &pipeline) {
        Ok(graph) => graph,
        Err(err) => {
            eprintln!("{err}");
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
            Ok((vg, caps, memory)) => print_stdout!(
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
                let graph = match build_graph(&reg, &opts, &pipeline) {
                    Ok(g) => g,
                    Err(e) => {
                        eprintln!("{e}");
                        process::exit(1);
                    }
                };
                print_stdout!(
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

    if opts.copy_plan {
        // Negotiate (probe source caps + solve the whole graph), then print the
        // memory-domain copy / allocation plan and exit without running. Shows the
        // domain each hop lives in and every transfer between domains, flagging real
        // frame copies, so a zero-copy pipeline can be confirmed at construction.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build tokio runtime");
        match rt.block_on(g2g_core::runtime::negotiate_graph(graph)) {
            Ok((vg, caps, memory)) => {
                let plan = g2g_core::runtime::copy_plan(&vg, &caps, &memory);
                print_stdout!("{}", plan.to_report());
            }
            Err(err) => {
                eprintln!("g2g-launch: negotiation failed ({err:?}); no copy plan");
                process::exit(1);
            }
        }
        return;
    }

    if opts.verbose {
        match (&opts.graph, &opts.script) {
            (Some(p), _) => eprintln!("graph: {p}"),
            (_, Some(p)) => eprintln!("script: {p}"),
            _ => eprintln!("pipeline: {pipeline}"),
        }
        // Show each link's negotiated caps + memory domain (gst `-v` style). The
        // solve lives in `negotiate_graph`, which consumes a graph, so negotiate a
        // freshly-parsed throwaway copy and keep `graph` for the run. On a
        // negotiation failure, dump the topology-only wiring and stop: the run
        // would fail the same way and log the same ERROR block a second time.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build tokio runtime");
        let fresh = match build_graph(&reg, &opts, &pipeline) {
            Ok(fresh) => fresh,
            Err(err) => {
                eprintln!("{err}");
                process::exit(1);
            }
        };
        match rt.block_on(g2g_core::runtime::negotiate_graph(fresh)) {
            Ok((vg, caps, memory)) => {
                eprintln!("negotiated links ({}):", vg.edge_count());
                for id in 0..vg.edge_count() {
                    let e = vg.edge(id);
                    let caps = caps
                        .get(id)
                        .map_or_else(|| "?".to_string(), |c| c.to_gst_string());
                    let mem = memory
                        .get(id)
                        .copied()
                        .unwrap_or(g2g_core::MemoryDomainKind::System);
                    eprintln!(
                        "  [{id}] {} -> {} : {caps}  mem={mem:?} policy={:?}",
                        node_name(vg.element(e.src.node), vg.kind(e.src.node)),
                        node_name(vg.element(e.dst.node), vg.kind(e.dst.node)),
                        e.policy
                    );
                }
            }
            Err(err) => {
                eprintln!(
                    "links ({}) [negotiation failed, topology only]:",
                    graph.edges().len()
                );
                let graph_node_name = |node| {
                    let kind = graph.node_kind(node).expect("an edge endpoint is a node");
                    node_name(graph.element(node), kind)
                };
                for (i, e) in graph.edges().iter().enumerate() {
                    eprintln!(
                        "  [{i}] {} -> {}  policy={:?}",
                        graph_node_name(e.src.node),
                        graph_node_name(e.dst.node),
                        e.policy
                    );
                }
                eprintln!("pipeline error: {err:?}");
                process::exit(1);
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

    // Live dashboard path: run the pipeline while serving telemetry + events on
    // `--observe <port>`. Diverges (returns / exits), so it must precede the
    // normal run that consumes `graph`.
    if let Some(port) = opts.observe {
        #[cfg(feature = "observe")]
        {
            let host = opts.observe_host.as_deref().unwrap_or("127.0.0.1");
            run_dashboard(
                &rt,
                graph,
                host,
                port,
                opts.quiet,
                opts.record_on_error.as_deref(),
            );
            return;
        }
        #[cfg(not(feature = "observe"))]
        {
            let _ = port;
            eprintln!(
                "pipeline error: --observe requires an observe build \
                 (rebuild with --features observe)"
            );
            process::exit(1);
        }
    }

    // Live terminal UI path: same tap as `--observe`, drawn here instead of in a
    // browser. Diverges like the dashboard path, so it also precedes the normal
    // run that consumes `graph`.
    if opts.tui {
        #[cfg(feature = "tui")]
        {
            run_tui(&rt, graph, opts.quiet, opts.record_on_error.as_deref());
            return;
        }
        #[cfg(not(feature = "tui"))]
        {
            eprintln!(
                "pipeline error: --tui requires a tui build \
                 (rebuild with --features tui)"
            );
            process::exit(1);
        }
    }

    if !opts.quiet {
        println_stdout!("Setting pipeline to PLAYING ...");
    }
    // The auto-plug fallback (M1023). A decoder the search picked can turn out
    // not to decode this stream, and the run dies where another decoder would
    // have played it. Retrying means restarting the pipeline, which is only
    // invisible while nothing has been presented, so the loop below re-plugs on
    // exactly that condition and reports the failure otherwise. A text pipeline
    // only: a `--graph` / `--script` file names its elements, so there is nothing
    // to choose differently.
    let mut avoided: Vec<&'static str> = Vec::new();
    let mut graph = graph;
    if let Some(target_ns) = seek_ns {
        attach_launch_seek(&mut graph, target_ns);
    }
    loop {
        let clock = WallClock::new();
        let progress = PipelineProgress::new();
        let (bus, bus_handle) = Bus::new(BUS_DEPTH);
        // Only built for `--record-on-error`: without it no ring is attached to any
        // link and the run costs exactly what it did before.
        let recorder = opts.record_on_error.as_ref().map(|_| FlightRecorder::new());
        let live = has_live_source(&graph);
        let started = Instant::now();
        // Poll the run future together with the bus, so a message is drained as
        // soon as it is posted (checksumsink waits on a full bus), and print a
        // liveness heartbeat once a second so a long-running (e.g. forever, the
        // default `videotestsrc`) pipeline does not look hung. A pipeline shorter
        // than a second stays quiet.
        let mut printed_status = false;
        let mut failed: Option<String> = None;
        // the run future borrows the spawner for the whole run
        #[cfg(feature = "multi-thread")]
        let spawner = match opts.threads {
            Some(ArmThreads::Pinned) => match PinnedThreadSpawner::from_sysfs() {
                Ok(pinned) => LaunchSpawner::Pinned(pinned),
                Err(error) => {
                    eprintln!("pipeline error: --threads=pinned: {error}");
                    process::exit(1);
                }
            },
            _ => LaunchSpawner::Free(TokioThreadSpawner),
        };
        // a re-plugged retry starts the file over
        let trace = opts.trace.as_deref().map(create_trace);
        let result = rt.block_on(async {
            let mut run_builder = RunBuilder::new(graph, &clock, LINK_CAPACITY)
                .with_progress(&progress)
                .with_bus(&bus_handle);
            if let Some(recorder) = &recorder {
                run_builder = run_builder.with_recorder(recorder);
            }
            if let Some(trace) = &trace {
                run_builder = run_builder.with_visit_recorder(trace.clone());
            }
            // `--threads` runs one OS thread per arm (opt-in multicore); the default
            // is the cooperative single-thread runner. Both return the same
            // `Result<RunStats, _>`, boxed to one type so the heartbeat loop is shared.
            let mut run: RunFuture = if opts.threads.is_some() {
                #[cfg(feature = "multi-thread")]
                {
                    Box::pin(run_builder.run_threaded(&spawner))
                }
                #[cfg(not(feature = "multi-thread"))]
                {
                    let _ = run_builder;
                    eprintln!(
                        "pipeline error: --threads requires a multi-thread build \
                     (rebuild with --features multi-thread)"
                    );
                    process::exit(1);
                }
            } else {
                Box::pin(run_builder.run())
            };
            let mut status_due = std::pin::pin!(tokio::time::sleep(STATUS_INTERVAL));
            std::future::poll_fn(|cx| {
                while let Poll::Ready(Some(message)) = std::pin::pin!(bus.recv()).poll(cx) {
                    handle_bus_message(message, opts.verbose, &mut failed);
                }
                while status_due.as_mut().poll(cx).is_ready() {
                    status_due
                        .as_mut()
                        .reset(tokio::time::Instant::now() + STATUS_INTERVAL);
                    if !opts.quiet && std::io::stderr().is_terminal() {
                        let pos = match progress.position() {
                            Some(ns) => format!("t={:.1}s", ns as f64 / 1.0e9),
                            None => String::from("prerolling"),
                        };
                        eprint!(
                            "\r  running... {pos} ({:.0}s wall)   ",
                            started.elapsed().as_secs_f64()
                        );
                        let _ = std::io::stderr().flush();
                        printed_status = true;
                    }
                }
                run.as_mut().poll(cx)
            })
            .await
        });
        if printed_status {
            eprintln!(); // move off the \r status line before the summary
        }
        if let Some(trace) = trace {
            if let Err(error) = trace.finish() {
                eprintln!("g2g-launch: --trace: {error}");
                process::exit(1);
            }
        }
        drain_bus(&bus, opts.verbose, &mut failed);
        match result {
            Ok(stats) => {
                if !opts.quiet {
                    print_run_summary(&stats, started.elapsed().as_secs_f64());
                }
                // the sink publishes a position for every frame it takes
                if let (Some(seconds), None) = (&opts.seek_seconds, progress.position()) {
                    eprintln!(
                        "pipeline error: no frame followed --seek {seconds}, \
                         the target is past the end of the stream"
                    );
                    process::exit(1);
                }
                return;
            }
            Err(err) => {
                // Re-plug around the element that failed, when the run is one a
                // restart can repeat: nothing presented, a pull source, and a text
                // pipeline that did not name the element itself.
                let replug = (!use_file && !live && progress.position().is_none())
                    .then(|| failed.take())
                    .flatten()
                    .and_then(|failed| fallback_factory(&reg, &pipeline, &failed, &avoided));
                if let Some(factory) = replug {
                    avoided.push(factory);
                    // A parse error here means nothing is left to plug in its
                    // place, so the original failure is what gets reported.
                    if let Ok(next) = parse_launch_avoiding(&reg, &pipeline, &avoided) {
                        eprintln!(
                            "{factory} could not decode this stream ({err:?}), retrying without it"
                        );
                        graph = next;
                        if let Some(target_ns) = seek_ns {
                            attach_launch_seek(&mut graph, target_ns);
                        }
                        continue;
                    }
                }
                eprintln!("pipeline error: {err:?}");
                if let (Some(rec), Some(dir)) = (&recorder, &opts.record_on_error) {
                    dump_flight_recording(rec, dir);
                }
                process::exit(1);
            }
        }
    }
}

// the label the DOT dump gives a node: its element's log category, or its kind for a tee
fn node_name(element: Option<&GraphNode>, kind: NodeKind) -> &'static str {
    element.map_or_else(|| g2g_core::dot::kind_label(kind), |e| e.log_category())
}

fn create_trace(path: &str) -> Arc<PerfettoTrace> {
    match PerfettoTrace::create(Path::new(path)) {
        Ok(trace) => trace,
        Err(error) => {
            eprintln!("g2g-launch: --trace {path}: {error}");
            process::exit(1);
        }
    }
}

fn seek_target_ns(seconds: &str) -> Result<u64, String> {
    let invalid = || format!("--seek needs a time in seconds, got '{seconds}'");
    let parsed: f64 = seconds.parse().map_err(|_| invalid())?;
    let target = Duration::try_from_secs_f64(parsed).map_err(|_| invalid())?;
    u64::try_from(target.as_nanos()).map_err(|_| invalid())
}

fn attach_launch_seek(graph: &mut Graph<GraphNode>, target_ns: u64) {
    if let Err(err) = attach_seek(graph, Seek::flush_to(target_ns)) {
        eprintln!("g2g-launch: --seek: {err}");
        process::exit(1);
    }
}

/// Drain the bus, which nothing else reads on this path: keep the element it
/// named as having ended the run, print each `checksumsink` line on stdout as
/// gst-launch does, and under `-v` print each sink's frame-identity report.
fn drain_bus(bus: &Bus, verbose: bool, failed: &mut Option<String>) {
    while let Some(message) = bus.try_recv() {
        handle_bus_message(message, verbose, failed);
    }
}

fn handle_bus_message(message: BusMessage, verbose: bool, failed: &mut Option<String>) {
    match message {
        BusMessage::ElementError { element, .. } => *failed = Some(element),
        BusMessage::Checksum { pts_ns, digest, .. } => {
            println_stdout!(
                "{}",
                g2g_plugins::checksumsink::checksum_line(pts_ns, digest)
            );
        }
        BusMessage::FrameIdentity { element, stats } if verbose => {
            eprintln!("\r{}", frame_identity_line(&element, &stats));
        }
        _ => {}
    }
}

fn frame_identity_line(element: &str, stats: &g2g_core::FrameIdentityStats) -> String {
    let ms = |ns: i64| ns as f64 / NANOS_PER_MILLISECOND;
    format!(
        "frame identity: {element} frames={} lost={} duplicates={} latency={:.1} ms (min {:.1} mean {:.1} max {:.1})",
        stats.frames,
        stats.lost,
        stats.duplicates,
        ms(stats.latency_ns),
        ms(stats.latency_min_ns),
        ms(stats.latency_mean_ns),
        ms(stats.latency_max_ns),
    )
}

/// Write the flight recorder's per-link rings into the `--record-on-error`
/// directory, naming each recording so the failing traffic can be replayed
/// straight back into a pipeline.
fn dump_flight_recording(recorder: &FlightRecorder, dir: &str) {
    match recorder.dump_to_dir(std::path::Path::new(dir)) {
        Ok(paths) if paths.is_empty() => {
            eprintln!("g2g-launch: no packets were recorded, nothing to dump")
        }
        Ok(paths) => {
            for path in &paths {
                eprintln!("  recorded {}", path.display());
            }
            eprintln!(
                "  replay with: g2g-launch \"replaysrc location={} ! ...\"",
                paths[0].display()
            );
        }
        Err(err) => eprintln!("g2g-launch: could not write the recording: {err:?}"),
    }
}

/// End-of-run report: the RunStats telemetry plus the measured
/// wall-clock throughput this run achieved, and for each presenting sink the
/// rate frames actually reached the display.
fn print_run_summary(stats: &g2g_core::runtime::RunStats, elapsed: f64) {
    print_stdout!("{}", stats.report());
    if elapsed > 0.0 {
        println_stdout!(
            "  run:     {:.2} s wall, {:.1} fps",
            elapsed,
            stats.frames_consumed as f64 / elapsed
        );
        for e in &stats.per_element {
            if let Some(p) = e.presentation {
                println_stdout!(
                    "  display: {:<16} {:.1} fps presented",
                    e.name,
                    p.presented as f64 / elapsed
                );
            }
        }
    }
}

/// Run the pipeline while serving the live dashboard on `host:port`. Sets up an
/// `Observer` (topology + per-element telemetry) and a `Bus` (events), fans bus
/// messages to every WebSocket client through a broadcast channel, and drives the
/// run future against the server with `select!` so the server is dropped when the
/// pipeline ends.
#[cfg(feature = "observe")]
fn run_dashboard(
    rt: &tokio::runtime::Runtime,
    graph: Graph<GraphNode>,
    host: &str,
    port: u16,
    quiet: bool,
    record_dir: Option<&str>,
) {
    use core::pin::pin;
    use tokio::sync::broadcast;

    // Non-loopback binds expose unauthenticated telemetry + frame previews.
    let loopback = host == "127.0.0.1" || host == "::1" || host == "localhost";
    if !loopback {
        eprintln!(
            "g2g-launch: --observe-host {host} serves the dashboard on all matching \
             interfaces with no auth (telemetry + edge previews are readable by anyone \
             who can reach it); use only on a trusted network."
        );
    }

    let clock = WallClock::new();
    let started = Instant::now();
    let recorder = record_dir.map(|_| FlightRecorder::new());
    let result = rt.block_on(async {
        let observer = Observer::new();
        let (bus, bus_handle) = Bus::new(256);
        let (ev_tx, _) = broadcast::channel::<String>(256);

        // Drain the bus and fan each event out to every connected client.
        let drain_tx = ev_tx.clone();
        let drainer = tokio::spawn(async move {
            while let Some(msg) = bus.recv().await {
                if let Some(json) = g2g_plugins::dashboard::event_json(&msg) {
                    let _ = drain_tx.send(json);
                }
            }
        });

        if !quiet {
            // Show a dialable URL: for an all-interfaces bind, loopback is the
            // local one to open (the machine's routable IP reaches it too).
            let shown = if host == "0.0.0.0" || host == "::" { "127.0.0.1" } else { host };
            println_stdout!("dashboard: http://{shown}:{port}  (bound {host}, pipeline running, Ctrl-C to stop)");
        }

        // A bind error ends the server before the run, and we surface it.
        let run: RunFuture = match &recorder {
            Some(rec) => Box::pin(run_graph_observed_recorded(
                graph,
                &clock,
                LINK_CAPACITY,
                &observer,
                Some(&bus_handle),
                rec,
            )),
            None => Box::pin(run_graph_observed(
                graph,
                &clock,
                LINK_CAPACITY,
                &observer,
                Some(&bus_handle),
            )),
        };
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async {
            let _ = shutdown_rx.await;
        };
        let mut server = pin!(g2g_plugins::dashboard::serve(
            observer.clone(),
            ev_tx,
            host,
            port,
            shutdown
        ));
        let run_result = tokio::select! {
            r = run => r,
            e = &mut server => {
                if let Err(err) = e {
                    eprintln!("dashboard: server error: {err}");
                }
                return Ok(Default::default());
            }
        };
        // The clients get the run's last events, its Eos included, before the server stops.
        drop(bus_handle);
        let _ = drainer.await;
        let _ = shutdown_tx.send(());
        if let Err(err) = server.await {
            eprintln!("dashboard: server error: {err}");
        }
        run_result
    });

    match result {
        Ok(stats) => {
            if !quiet {
                print_run_summary(&stats, started.elapsed().as_secs_f64());
            }
        }
        Err(err) => {
            eprintln!("pipeline error: {err:?}");
            if let (Some(rec), Some(dir)) = (&recorder, record_dir) {
                dump_flight_recording(rec, dir);
            }
            process::exit(1);
        }
    }
}

/// Run the pipeline under the in-terminal live UI (`--tui`). Same `Observer`
/// tap as the dashboard, drawn here: the run future is driven against a redraw
/// tick and a keyboard poll with `select!`, so the screen refreshes while the
/// pipeline runs. Quitting (`q` / Esc / Ctrl-C, which raw mode delivers as a
/// keypress rather than a signal) drops the run future, ending the pipeline the
/// same way Ctrl-C ends the dashboard path.
#[cfg(feature = "tui")]
fn run_tui(
    rt: &tokio::runtime::Runtime,
    graph: Graph<GraphNode>,
    quiet: bool,
    record_dir: Option<&str>,
) {
    use g2g_plugins::tui::PipelineTui;

    const REDRAW: Duration = Duration::from_millis(250);
    const INPUT_POLL: Duration = Duration::from_millis(50);

    let mut tui = match PipelineTui::start() {
        Ok(tui) => tui,
        Err(err) => {
            eprintln!("g2g-launch: --tui needs an interactive terminal ({err})");
            process::exit(1);
        }
    };
    let clock = WallClock::new();
    let started = Instant::now();
    let observer = Observer::new();
    let recorder = record_dir.map(|_| FlightRecorder::new());
    // `None` on quit, the run's own result when the pipeline ends first.
    let outcome = rt.block_on(async {
        let mut run: RunFuture = match &recorder {
            Some(rec) => Box::pin(run_graph_observed_recorded(
                graph,
                &clock,
                LINK_CAPACITY,
                &observer,
                None,
                rec,
            )),
            None => Box::pin(run_graph_observed(
                graph,
                &clock,
                LINK_CAPACITY,
                &observer,
                None,
            )),
        };
        let mut redraw = tokio::time::interval(REDRAW);
        let mut input = tokio::time::interval(INPUT_POLL);
        loop {
            tokio::select! {
                result = run.as_mut() => break Some(result),
                _ = redraw.tick() => {
                    if tui.draw(&observer.snapshot()).is_err() {
                        break None;
                    }
                }
                _ = input.tick() => {
                    if tui.handle_input().unwrap_or(true) {
                        break None;
                    }
                }
            }
        }
    });
    // Hand the terminal (and the stderr log sink) back before anything prints.
    drop(tui);

    match outcome {
        Some(Ok(stats)) => {
            if !quiet {
                print_run_summary(&stats, started.elapsed().as_secs_f64());
            }
        }
        Some(Err(err)) => {
            eprintln!("pipeline error: {err:?}");
            if let (Some(rec), Some(dir)) = (&recorder, record_dir) {
                dump_flight_recording(rec, dir);
            }
            process::exit(1);
        }
        None => {
            if !quiet {
                println_stdout!("stopped after {:.2} s", started.elapsed().as_secs_f64());
            }
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
        let (opts, rest) =
            parse_opts(toks(&["-v", "-e", "videotestsrc", "!", "fakesink"]).into_iter());
        assert!(opts.verbose);
        assert!(!opts.quiet);
        assert_eq!(rest, toks(&["videotestsrc", "!", "fakesink"]));
    }

    #[test]
    fn record_on_error_takes_a_directory_either_way() {
        let (spaced, rest) = parse_opts(
            toks(&[
                "--record-on-error",
                "dumps",
                "videotestsrc",
                "!",
                "fakesink",
            ])
            .into_iter(),
        );
        assert_eq!(spaced.record_on_error.as_deref(), Some("dumps"));
        assert_eq!(rest, toks(&["videotestsrc", "!", "fakesink"]));
        let (joined, _) = parse_opts(toks(&["--record-on-error=dumps", "fakesink"]).into_iter());
        assert_eq!(joined.record_on_error.as_deref(), Some("dumps"));
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
    fn tui_flag_splits_from_pipeline() {
        let (opts, rest) =
            parse_opts(toks(&["--tui", "videotestsrc", "!", "fakesink"]).into_iter());
        assert!(opts.tui);
        assert!(opts.observe.is_none());
        assert_eq!(rest, toks(&["videotestsrc", "!", "fakesink"]));
    }

    #[test]
    fn dot_flag_splits_from_pipeline() {
        let (opts, rest) =
            parse_opts(toks(&["--dot", "videotestsrc", "!", "fakesink"]).into_iter());
        assert!(opts.dot);
        assert!(!opts.verbose);
        assert_eq!(rest, toks(&["videotestsrc", "!", "fakesink"]));
    }
}

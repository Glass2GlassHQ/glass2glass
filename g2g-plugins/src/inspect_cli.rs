// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! `g2g-inspect`: the `gst-inspect` analog. Introspects an element registry
//! (the same one the text-launch parser uses).
//!
//! Usage:
//!
//! ```text
//! g2g-inspect                  # list every registerable element
//! g2g-inspect <element>        # dump one element's role, properties, pads
//! g2g-inspect --all            # dump every element in full
//! g2g-inspect --json [name]    # machine-readable registry dump (all or one)
//! g2g-inspect --maturity       # derived conformance maturity per element
//! g2g-inspect --gst <name>     # what a GStreamer element name maps to in g2g
//! g2g-inspect --gst-map        # gst-name/g2g-runtime-name pairs, TSV
//! g2g-inspect --plugin <path>  # load a plugin first, so its elements list
//! g2g-inspect --trusted-key <path>  # only load plugins signed by this key
//! g2g-inspect -h, --help       # print the usage and exit
//! g2g-inspect --version        # print the version and exit
//! ```
//!
//! On a Unix terminal the output is coloured, unless `NO_COLOR` is set, and goes
//! through `$PAGER`, or `less` when it is unset, as `gst-inspect` does.
//!
//! [`run`] takes the registry, so the `g2g-inspect` binary passes
//! [`default_registry`](crate::registry::default_registry) and a launcher with
//! more elements passes its own. Backed by
//! [`g2g_core::runtime::Registry::inspect`]. Requires the `std`
//! feature (the registry is std-only). With the `plugin-loader` feature, plugins
//! from `$G2G_PLUGIN_PATH` and `--plugin <path>` are loaded first so their
//! elements appear in the listing and dumps. With `plugin-signing`, each
//! `--trusted-key <path>` (and each key file in `$G2G_PLUGIN_TRUSTED_KEYS`) adds
//! an Ed25519 public key, and once any key is trusted every plugin must carry a
//! signature from one of them.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::eprintln;
use std::process;
#[cfg(unix)]
use std::sync::atomic::{AtomicI32, Ordering};

use g2g_core::runtime::Registry;

use crate::cli_stdout::write_stdout;
use crate::gst_compat::{element_name_finding, gst_equivalent, gst_name_synonyms, GstEquivalent};

macro_rules! print_stdout {
    ($($arg:tt)*) => { write_stdout(format_args!($($arg)*)) };
}

macro_rules! println_stdout {
    ($($arg:tt)*) => { write_stdout(format_args!("{}\n", format_args!($($arg)*))) };
}

/// The plugin arguments, split out of the command line so what is left is the
/// element name / mode flags.
#[derive(Default)]
struct PluginArgs {
    /// Each `--plugin <path>`: a shared object to load.
    plugins: Vec<String>,
    /// Each `--trusted-key <path>`: an Ed25519 public key file.
    trusted_keys: Vec<String>,
}

/// Pull every `--plugin` / `--trusted-key` (either spelling) out of `raw`,
/// returning them and the remaining arguments.
fn split_plugin_args(raw: Vec<String>) -> (PluginArgs, Vec<String>) {
    let mut found = PluginArgs::default();
    let mut rest = Vec::new();
    let mut iter = raw.into_iter();
    while let Some(arg) = iter.next() {
        let flag = ["--plugin", "--trusted-key"]
            .into_iter()
            .find(|f| arg == *f || arg.starts_with(&format!("{f}=")));
        let Some(flag) = flag else {
            rest.push(arg);
            continue;
        };
        let value = match arg.strip_prefix(&format!("{flag}=")) {
            Some(inline) => Some(inline.to_string()),
            None => iter.next(),
        };
        match value {
            Some(path) if flag == "--plugin" => found.plugins.push(path),
            Some(path) => found.trusted_keys.push(path),
            None => eprintln!("g2g-inspect: {flag} needs a path argument"),
        }
    }
    (found, rest)
}

/// Load `$G2G_PLUGIN_PATH` + each `--plugin` path into `reg` so plugin elements
/// are introspectable. Compiled out without `plugin-loader`.
#[cfg(all(feature = "plugin-loader", not(feature = "plugin-signing")))]
fn load_plugins(reg: &mut Registry, args: &PluginArgs) {
    use crate::plugin_loader;
    if !args.trusted_keys.is_empty() {
        eprintln!(
            "g2g-inspect: built without the `plugin-signing` feature; \
             --trusted-key cannot be honoured"
        );
        process::exit(1);
    }
    if let Err(err) = plugin_loader::load_from_env(reg) {
        eprintln!("g2g-inspect: {err}");
        process::exit(1);
    }
    for path in &args.plugins {
        if let Err(err) = plugin_loader::load_plugin(path, reg) {
            eprintln!("g2g-inspect: {err}");
            process::exit(1);
        }
    }
}

/// The same with signature verification: `--trusted-key` adds to whatever
/// `$G2G_PLUGIN_TRUSTED_KEYS` names, and the resulting set gates both the path
/// scan and each explicit `--plugin`.
#[cfg(feature = "plugin-signing")]
fn load_plugins(reg: &mut Registry, args: &PluginArgs) {
    use crate::plugin_loader::{self, default_policy};
    let fatal = |err: &dyn std::fmt::Display| -> ! {
        eprintln!("g2g-inspect: {err}");
        process::exit(1)
    };
    let mut trusted = plugin_loader::trusted_keys_from_env().unwrap_or_else(|e| fatal(&e));
    for path in &args.trusted_keys {
        if let Err(err) = trusted.trust_key_file(path) {
            fatal(&err);
        }
    }
    if let Err(err) = plugin_loader::load_from_env_verified(reg, &trusted) {
        fatal(&err);
    }
    for path in &args.plugins {
        if let Err(err) = plugin_loader::load_plugin_verified(path, reg, &trusted, &default_policy)
        {
            fatal(&err);
        }
    }
}

#[cfg(not(feature = "plugin-loader"))]
fn load_plugins(_reg: &mut Registry, args: &PluginArgs) {
    if !args.plugins.is_empty()
        || !args.trusted_keys.is_empty()
        || std::env::var_os("G2G_PLUGIN_PATH").is_some()
    {
        eprintln!(
            "g2g-inspect: built without the `plugin-loader` feature; \
             --plugin / --trusted-key / $G2G_PLUGIN_PATH ignored"
        );
    }
}

/// Print the registry (or one element) as JSON: `{"elements":[...]}`. The JSON
/// shape lives in `g2g_plugins::toolingjson`, shared with the MCP server.
#[cfg(feature = "tooling-json")]
fn dump_json(reg: &Registry, name: Option<&str>) {
    match crate::toolingjson::registry_json(reg, name) {
        Ok(v) => println_stdout!(
            "{}",
            serde_json::to_string_pretty(&v).expect("serialize registry")
        ),
        Err(msg) => {
            eprintln!("{msg}");
            process::exit(1);
        }
    }
}

#[cfg(not(feature = "tooling-json"))]
fn dump_json(_reg: &Registry, _name: Option<&str>) {
    eprintln!(
        "g2g-inspect: --json needs the `tooling-json` build feature \
         (rebuild with --features tooling-json)"
    );
    process::exit(1);
}

const USAGE: &str =
    "usage: g2g-inspect [--plugin <path>] [--trusted-key <path>] [<element>]\n       \
g2g-inspect --all | --json [<element>] | --maturity\n       \
g2g-inspect --gst <gstreamer-element> | --gst-map | --gst-scan <source-file.c|.py>\n       \
g2g-inspect -h | --help | --version";

// gst-inspect's less flags: R passes colour, X leaves the text on screen, F skips paging text that fits
#[cfg(unix)]
const LESS_OPTIONS: &str = "RXF";

#[cfg(unix)]
static PAGER_PROCESS_ID: AtomicI32 = AtomicI32::new(0);

// on a terminal, gst-inspect sends stdout, and stderr when it is a terminal too, through `$PAGER` or less
#[cfg(unix)]
fn page_output() {
    use std::io::IsTerminal;
    use std::os::fd::IntoRawFd;
    if !std::io::stdout().is_terminal() {
        return;
    }
    let custom_pager = std::env::var("PAGER").ok();
    let command = custom_pager.as_deref().unwrap_or("less");
    let mut words = command.split_whitespace();
    let Some(program) = words.next() else {
        return;
    };
    let spawned = process::Command::new(program)
        .args(words)
        .env("LESS", LESS_OPTIONS)
        .stdin(process::Stdio::piped())
        .spawn();
    let mut pager = match spawned {
        Ok(pager) => pager,
        Err(err) => {
            if custom_pager.is_some() {
                eprintln!("g2g-inspect: cannot start the pager `{command}`: {err}");
            }
            return;
        }
    };
    let pipe = pager
        .stdin
        .take()
        .expect("the pager's stdin is piped")
        .into_raw_fd();
    let page_stderr = std::io::stderr().is_terminal();
    // SAFETY: `pipe` is an open descriptor this process owns, and dup2 onto the standard descriptors keeps Rust's handles valid
    unsafe {
        libc::dup2(pipe, libc::STDOUT_FILENO);
        if page_stderr {
            libc::dup2(pipe, libc::STDERR_FILENO);
        }
        libc::close(pipe);
    }
    PAGER_PROCESS_ID.store(pager.id() as i32, Ordering::Relaxed);
    // SAFETY: `wait_for_pager` takes no arguments and uses no state that exit tears down first
    unsafe {
        libc::atexit(wait_for_pager);
    }
}

// less reads until its input closes, and the shell prompt returns only after it quits
#[cfg(unix)]
extern "C" fn wait_for_pager() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let mut status = 0;
    // SAFETY: closing the standard descriptors ends the pager's input, and waitpid waits on this process's own child
    unsafe {
        libc::close(libc::STDOUT_FILENO);
        libc::close(libc::STDERR_FILENO);
        libc::waitpid(PAGER_PROCESS_ID.load(Ordering::Relaxed), &mut status, 0);
    }
}

// gst-inspect's escape codes
const RESET: &str = "\x1b[0m";
const YELLOW: &str = "\x1b[33m";
const BRIGHT_BLUE: &str = "\x1b[94m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";

// `Registry::inspect` section headings
const FACTORY_DETAILS_HEADING: &str = "Factory Details:";
const PROPERTIES_HEADING: &str = "Element Properties:";

fn paint(colour: &str, text: &str) -> String {
    format!("{colour}{text}{RESET}")
}

fn coloured_listing_line(line: &str) -> String {
    let name_end = line.find([':', ' ']).unwrap_or(line.len());
    format!("{}{}", paint(GREEN, &line[..name_end]), &line[name_end..])
}

// `name=value` fields after the media type, split on commas outside `[1,16384]` ranges and `{a,b}` lists
fn coloured_caps(caps: &str) -> String {
    let mut depth = 0usize;
    let mut fields = Vec::new();
    let mut start = 0;
    for (at, character) in caps.char_indices() {
        match character {
            '[' | '{' | '(' => depth += 1,
            ']' | '}' | ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                fields.push(&caps[start..at]);
                start = at + 1;
            }
            _ => {}
        }
    }
    fields.push(&caps[start..]);
    let mut out = paint(YELLOW, fields[0]);
    for field in &fields[1..] {
        out.push(',');
        match field.split_once('=') {
            Some((name, value)) => {
                out.push_str(&format!(
                    "{}={}",
                    paint(CYAN, name),
                    paint(BRIGHT_BLUE, value)
                ));
            }
            None => out.push_str(field),
        }
    }
    out
}

// `String. Values: a | b. Default: a` and `flags: readable, writable`
fn coloured_property_attributes(line: &str) -> String {
    line.split(". ")
        .map(|part| match part.split_once(": ") {
            Some((name, value)) => format!("{}: {}", paint(YELLOW, name), paint(CYAN, value)),
            None => paint(GREEN, part),
        })
        .collect::<Vec<_>>()
        .join(". ")
}

fn coloured_dump(dump: &str) -> String {
    let mut section = "";
    let mut out = String::with_capacity(dump.len() * 2);
    for line in dump.split_inclusive('\n') {
        let text = line.trim_end_matches('\n');
        let indent = text.len() - text.trim_start().len();
        let body = &text[indent..];
        let key_end = body.find(' ').unwrap_or(body.len());
        let styled = match indent {
            0 if body.ends_with(':') => {
                section = body;
                paint(YELLOW, body)
            }
            0 => {
                section = "";
                body.to_string()
            }
            _ if section == FACTORY_DETAILS_HEADING => {
                format!(
                    "{}{}",
                    paint(BRIGHT_BLUE, &body[..key_end]),
                    &body[key_end..]
                )
            }
            2 if section == PROPERTIES_HEADING => match body.split_once(": ") {
                Some((name, description)) => format!("{}: {description}", paint(BRIGHT_BLUE, name)),
                None => paint(BRIGHT_BLUE, body),
            },
            _ if section == PROPERTIES_HEADING => coloured_property_attributes(body),
            _ if body.ends_with(':') => paint(BRIGHT_BLUE, body),
            _ => coloured_caps(body),
        };
        out.push_str(&text[..indent]);
        out.push_str(&styled);
        out.push_str(&line[text.len()..]);
    }
    out
}

/// Run `g2g-inspect` over `reg` with the command-line `args` (program name
/// excluded), printing to stdout and exiting the process on an error.
pub fn run(mut reg: Registry, args: Vec<String>) {
    let (plugin_args, rest) = split_plugin_args(args);
    // gst-inspect colours a terminal only, and NO_COLOR is the cross-tool switch to turn colour off
    let colour = cfg!(unix)
        && std::io::IsTerminal::is_terminal(&std::io::stdout())
        && std::env::var_os("NO_COLOR").is_none();
    let shown_dump = |dump: String| if colour { coloured_dump(&dump) } else { dump };
    #[cfg(unix)]
    page_output();
    load_plugins(&mut reg, &plugin_args);
    let mut args = rest.into_iter();
    match args.next() {
        // No element named: list them all, `name: Long-name` per line, the
        // `gst-inspect` index.
        None => {
            for line in reg.element_listing() {
                let line = if colour {
                    coloured_listing_line(&line)
                } else {
                    line
                };
                println_stdout!("{line}");
            }
        }
        Some(flag) if flag == "--help" || flag == "-h" => {
            println_stdout!("{USAGE}");
        }
        Some(flag) if flag == "--version" => {
            println_stdout!("g2g-inspect {}", env!("CARGO_PKG_VERSION"));
        }
        // `--all` / `-a`: the full dump for every registered element, separated
        // by a rule, so the whole catalog can be read or grepped at once.
        Some(flag) if flag == "--all" || flag == "-a" => {
            let mut names = reg.element_names();
            names.sort_unstable();
            let total = names.len();
            for (i, name) in names.into_iter().enumerate() {
                if let Some(dump) = reg.inspect(name) {
                    print_stdout!("{}", shown_dump(dump));
                    if i + 1 < total {
                        println_stdout!("\n{}\n", "-".repeat(60));
                    }
                }
            }
        }
        // `--json [name]`: machine-readable registry dump for the dev tools
        // (visual builder / MCP server). All elements, or one if named.
        Some(flag) if flag == "--json" => {
            dump_json(&reg, args.next().as_deref());
        }
        // `--maturity`: run the in-process conformance battery and print each
        // element's derived maturity (never a hand-authored claim). The batteries
        // exercise real elements with cheap loopback checks; reference-gear /
        // hardware evidence comes from the resource-owning integration tests.
        Some(flag) if flag == "--maturity" => {
            // In-process batteries + any persisted Oracle / Hardware evidence
            // ($G2G_CONFORMANCE_LOG) from the resource-owning conformance tests.
            let report = crate::conformance::persist::full_report();
            print_stdout!("{}", report.to_table());
        }
        // `--gst <name>`: map a GStreamer element name to its g2g equivalent,
        // for porting a pipeline element by element.
        Some(flag) if flag == "--gst" => {
            let Some(gst_name) = args.next() else {
                eprintln!("usage: g2g-inspect --gst <gstreamer-element-name>");
                process::exit(2);
            };
            match gst_equivalent(&reg, &gst_name) {
                GstEquivalent::Available => {
                    println_stdout!("{gst_name}: available in g2g under the same name");
                }
                GstEquivalent::Renamed(g) => {
                    println_stdout!(
                        "{gst_name}: g2g calls it `{g}` (run `g2g-inspect {g}` for details)"
                    );
                }
                GstEquivalent::Unsupported(hint) => {
                    println_stdout!("{gst_name}: no g2g element. {hint}");
                }
                GstEquivalent::NotCompiled(feature) => {
                    println_stdout!(
                        "{gst_name}: a g2g element, but this build left it out; rebuild with `--features {feature}`"
                    );
                }
                GstEquivalent::DidYouMean(near) => {
                    println_stdout!("{gst_name}: unknown to g2g; did you mean `{near}`?");
                    process::exit(1);
                }
                GstEquivalent::Unknown => {
                    println_stdout!("{gst_name}: unknown to g2g; no known equivalent. List elements with `g2g-inspect`.");
                    process::exit(1);
                }
                GstEquivalent::BuildsOnlyOn { platform, feature } => {
                    println_stdout!(
                        "{gst_name}: a g2g element that only builds on {platform}, with `--features {feature}`"
                    );
                }
            }
        }
        // `--gst-map`: every gst element name g2g's runtime reports differently,
        // one `gst-name<TAB>g2g-runtime-name` per line, for a tool pairing the
        // two engines' graph dumps element by element.
        Some(flag) if flag == "--gst-map" => {
            for (gst_name, g2g_name) in gst_name_synonyms(&reg) {
                println_stdout!("{gst_name}\t{g2g_name}");
            }
        }
        // `--gst-scan <file>`: scan a GStreamer application source file (C or
        // Python) and report the element factories it uses that are not portable
        // as-is, plus the dynamic-pipeline APIs that map to a g2g primitive.
        Some(flag) if flag == "--gst-scan" => {
            let Some(path) = args.next() else {
                eprintln!("usage: g2g-inspect --gst-scan <source-file.c|.py>");
                process::exit(2);
            };
            let source = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("cannot read {path}: {e}");
                    process::exit(2);
                }
            };
            let report = crate::gst_compat::scan_source(&reg, &source);
            if report.findings.is_empty() {
                println_stdout!("{path}: every element factory resolves to a g2g element");
            } else {
                println_stdout!("{path}: elements needing attention:");
                for f in &report.findings {
                    println_stdout!("  - {f}");
                }
            }
            for n in &report.notes {
                println_stdout!("  note: {n}");
            }
            if !report.findings.is_empty() {
                process::exit(1);
            }
        }
        Some(name) => match reg.inspect(&name) {
            Some(dump) => print_stdout!("{}", shown_dump(dump)),
            None => {
                eprintln!("No such element: {name}");
                if let Some(hint) = element_name_finding(&reg, &name) {
                    eprintln!("  hint: {hint}");
                }
                process::exit(1);
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn without_escape_codes(text: &str) -> String {
        let mut plain = String::new();
        let mut rest = text;
        while let Some(start) = rest.find('\x1b') {
            plain.push_str(&rest[..start]);
            let end = rest[start..].find('m').expect("escape codes end in m");
            rest = &rest[start + end + 1..];
        }
        plain.push_str(rest);
        plain
    }

    #[test]
    fn colouring_keeps_the_text_and_marks_each_part() {
        let reg = crate::registry::default_registry();
        for name in ["videotestsrc", "videoconvert"] {
            let dump = reg.inspect(name).expect("registered");
            assert_eq!(without_escape_codes(&coloured_dump(&dump)), dump, "{name}");
        }
        let source = coloured_dump(&reg.inspect("videotestsrc").expect("registered"));
        for part in [
            paint(YELLOW, FACTORY_DETAILS_HEADING),
            format!("  {} ", paint(BRIGHT_BLUE, "Long-name")),
            format!("  {}: ", paint(BRIGHT_BLUE, "pattern")),
            paint(GREEN, "String"),
            format!("{}: ", paint(YELLOW, "Default")),
            format!("{}={}", paint(CYAN, "format"), paint(BRIGHT_BLUE, "RGBA")),
            format!(
                "{}={}",
                paint(CYAN, "width"),
                paint(BRIGHT_BLUE, "[1,16384]")
            ),
        ] {
            assert!(source.contains(&part), "{part:?} in {source}");
        }
        assert_eq!(
            coloured_listing_line("aacdec: AAC decoder"),
            format!("{}: AAC decoder", paint(GREEN, "aacdec"))
        );
    }
}

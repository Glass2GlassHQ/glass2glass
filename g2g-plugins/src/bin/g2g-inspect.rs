//! `g2g-inspect`: the `gst-inspect` analog. Introspects the standard element
//! registry (the same one the text-launch parser uses).
//!
//! Usage:
//!   g2g-inspect                  # list every registerable element
//!   g2g-inspect <element>        # dump one element's role, properties, pads
//!   g2g-inspect --all            # dump every element in full
//!   g2g-inspect --gst <name>     # what a GStreamer element name maps to in g2g
//!   g2g-inspect --plugin <path>  # load a plugin first, so its elements list
//!
//! Backed by [`g2g_plugins::registry::default_registry`] and
//! [`g2g_core::runtime::Registry::inspect`] (M105/M107). Requires the `std`
//! feature (the registry is std-only). With the `plugin-loader` feature, plugins
//! from `$G2G_PLUGIN_PATH` and `--plugin <path>` are loaded first so their
//! elements appear in the listing and dumps (M201).

use std::process;

use g2g_plugins::gst_compat::{gst_equivalent, GstEquivalent};
use g2g_plugins::registry::default_registry;

/// Pull every `--plugin <path>` / `--plugin=<path>` out of `raw`, returning the
/// plugin paths and the remaining arguments (the element name / mode flags).
fn split_plugin_args(raw: Vec<String>) -> (Vec<String>, Vec<String>) {
    let mut plugins = Vec::new();
    let mut rest = Vec::new();
    let mut iter = raw.into_iter();
    while let Some(arg) = iter.next() {
        if let Some(path) = arg.strip_prefix("--plugin=") {
            plugins.push(path.to_string());
        } else if arg == "--plugin" {
            match iter.next() {
                Some(path) => plugins.push(path),
                None => eprintln!("g2g-inspect: --plugin needs a path argument"),
            }
        } else {
            rest.push(arg);
        }
    }
    (plugins, rest)
}

/// Load `$G2G_PLUGIN_PATH` + each `--plugin` path into `reg` so plugin elements
/// are introspectable. Compiled out without `plugin-loader`.
#[cfg(feature = "plugin-loader")]
fn load_plugins(reg: &mut g2g_core::runtime::Registry, plugins: &[String]) {
    use g2g_plugins::plugin_loader;
    if let Err(err) = plugin_loader::load_from_env(reg) {
        eprintln!("g2g-inspect: {err}");
        process::exit(1);
    }
    for path in plugins {
        if let Err(err) = plugin_loader::load_plugin(path, reg) {
            eprintln!("g2g-inspect: {err}");
            process::exit(1);
        }
    }
}

#[cfg(not(feature = "plugin-loader"))]
fn load_plugins(_reg: &mut g2g_core::runtime::Registry, plugins: &[String]) {
    if !plugins.is_empty() || std::env::var_os("G2G_PLUGIN_PATH").is_some() {
        eprintln!(
            "g2g-inspect: built without the `plugin-loader` feature; \
             --plugin / $G2G_PLUGIN_PATH ignored"
        );
    }
}

fn main() {
    let (plugins, rest) = split_plugin_args(std::env::args().skip(1).collect());
    let mut reg = default_registry();
    load_plugins(&mut reg, &plugins);
    let mut args = rest.into_iter();
    match args.next() {
        // No element named: list them all, `name: Long-name` per line, the
        // `gst-inspect` index.
        None => {
            for line in reg.element_listing() {
                println!("{line}");
            }
        }
        // `--all` / `-a`: the full dump for every registered element, separated
        // by a rule, so the whole catalog can be read or grepped at once.
        Some(flag) if flag == "--all" || flag == "-a" => {
            let names = reg.element_names();
            let total = names.len();
            for (i, name) in names.into_iter().enumerate() {
                if let Some(dump) = reg.inspect(name) {
                    print!("{dump}");
                    if i + 1 < total {
                        println!("\n{}\n", "-".repeat(60));
                    }
                }
            }
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
                    println!("{gst_name}: available in g2g under the same name");
                }
                GstEquivalent::Renamed(g) => {
                    println!("{gst_name}: g2g calls it `{g}` (run `g2g-inspect {g}` for details)");
                }
                GstEquivalent::Unsupported(hint) => {
                    println!("{gst_name}: no g2g element. {hint}");
                }
                GstEquivalent::Unknown => {
                    println!("{gst_name}: unknown to g2g; no known equivalent. List elements with `g2g-inspect`.");
                    process::exit(1);
                }
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
            let report = g2g_plugins::gst_compat::scan_source(&reg, &source);
            if report.findings.is_empty() {
                println!("{path}: every element factory resolves to a g2g element");
            } else {
                println!("{path}: elements needing attention:");
                for f in &report.findings {
                    println!("  - {f}");
                }
            }
            for n in &report.notes {
                println!("  note: {n}");
            }
            if !report.findings.is_empty() {
                process::exit(1);
            }
        }
        Some(name) => match reg.inspect(&name) {
            Some(dump) => print!("{dump}"),
            None => {
                eprintln!("No such element: {name}");
                process::exit(1);
            }
        },
    }
}

//! `g2g-inspect`: the `gst-inspect` analog. Introspects the standard element
//! registry (the same one the text-launch parser uses).
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
//! ```
//!
//! Backed by [`g2g_plugins::registry::default_registry`] and
//! [`g2g_core::runtime::Registry::inspect`] (M105/M107). Requires the `std`
//! feature (the registry is std-only). With the `plugin-loader` feature, plugins
//! from `$G2G_PLUGIN_PATH` and `--plugin <path>` are loaded first so their
//! elements appear in the listing and dumps (M201). With `plugin-signing`, each
//! `--trusted-key <path>` (and each key file in `$G2G_PLUGIN_TRUSTED_KEYS`) adds
//! an Ed25519 public key, and once any key is trusted every plugin must carry a
//! signature from one of them (M1061).

use std::process;

use g2g_plugins::gst_compat::{gst_equivalent, gst_name_synonyms, GstEquivalent};
use g2g_plugins::registry::default_registry;

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
fn load_plugins(reg: &mut g2g_core::runtime::Registry, args: &PluginArgs) {
    use g2g_plugins::plugin_loader;
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
fn load_plugins(reg: &mut g2g_core::runtime::Registry, args: &PluginArgs) {
    use g2g_plugins::plugin_loader::{self, default_policy};
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
fn load_plugins(_reg: &mut g2g_core::runtime::Registry, args: &PluginArgs) {
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
fn dump_json(reg: &g2g_core::runtime::Registry, name: Option<&str>) {
    match g2g_plugins::toolingjson::registry_json(reg, name) {
        Ok(v) => println!(
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
fn dump_json(_reg: &g2g_core::runtime::Registry, _name: Option<&str>) {
    eprintln!(
        "g2g-inspect: --json needs the `tooling-json` build feature \
         (rebuild with --features tooling-json)"
    );
    process::exit(1);
}

fn main() {
    let (plugin_args, rest) = split_plugin_args(std::env::args().skip(1).collect());
    let mut reg = default_registry();
    load_plugins(&mut reg, &plugin_args);
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
            let report = g2g_plugins::conformance::persist::full_report();
            print!("{}", report.to_table());
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
                GstEquivalent::NotCompiled(feature) => {
                    println!(
                        "{gst_name}: a g2g element, but this build left it out; rebuild with `--features {feature}`"
                    );
                }
                GstEquivalent::DidYouMean(near) => {
                    println!("{gst_name}: unknown to g2g; did you mean `{near}`?");
                    process::exit(1);
                }
                GstEquivalent::Unknown => {
                    println!("{gst_name}: unknown to g2g; no known equivalent. List elements with `g2g-inspect`.");
                    process::exit(1);
                }
            }
        }
        // `--gst-map`: every gst element name g2g's runtime reports differently,
        // one `gst-name<TAB>g2g-runtime-name` per line, for a tool pairing the
        // two engines' graph dumps element by element.
        Some(flag) if flag == "--gst-map" => {
            for (gst_name, g2g_name) in gst_name_synonyms(&reg) {
                println!("{gst_name}\t{g2g_name}");
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

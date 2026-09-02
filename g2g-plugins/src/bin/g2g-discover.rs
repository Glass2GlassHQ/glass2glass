//! `g2g-discover`: the `gst-discoverer-1.0` analog. Reports what a media file
//! holds without a hand-written pipeline: container, elementary streams with
//! their caps, duration, and metadata.
//!
//! Usage:
//!
//! ```text
//! g2g-discover <path-or-uri>          # human-readable report
//! g2g-discover <path-or-uri> --json   # the same as JSON
//! g2g-discover --plugin <path> ...    # load a plugin first
//! g2g-discover --trusted-key <path>   # only load plugins signed by this key
//! ```
//!
//! Backed by [`g2g_plugins::discover::discover`], which sniffs the container,
//! runs a headless probe graph to the first frame, and reads the demuxer's
//! stream collection, tags and duration off the pipeline bus. Nothing is
//! decoded. Local files only: a `file://` URI or a plain path. Requires the
//! `std` feature; `--json` needs `tooling-json`. With `plugin-loader`, plugins
//! from `$G2G_PLUGIN_PATH` and `--plugin <path>` are loaded first so a
//! third-party container element can serve the probe; with `plugin-signing`,
//! each `--trusted-key <path>` adds an Ed25519 public key that plugins must be
//! signed by.

use std::process;

use g2g_plugins::discover::{discover, to_text, Discovery};
use g2g_plugins::registry::default_registry;

/// The plugin arguments, split out of the command line so what is left is the
/// URI / mode flags.
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
            None => eprintln!("g2g-discover: {flag} needs a path argument"),
        }
    }
    (found, rest)
}

/// Load `$G2G_PLUGIN_PATH` + each `--plugin` path into `reg`. Compiled out
/// without `plugin-loader`.
#[cfg(all(feature = "plugin-loader", not(feature = "plugin-signing")))]
fn load_plugins(reg: &mut g2g_core::runtime::Registry, args: &PluginArgs) {
    use g2g_plugins::plugin_loader;
    if !args.trusted_keys.is_empty() {
        eprintln!(
            "g2g-discover: built without the `plugin-signing` feature; \
             --trusted-key cannot be honoured"
        );
        process::exit(1);
    }
    if let Err(err) = plugin_loader::load_from_env(reg) {
        eprintln!("g2g-discover: {err}");
        process::exit(1);
    }
    for path in &args.plugins {
        if let Err(err) = plugin_loader::load_plugin(path, reg) {
            eprintln!("g2g-discover: {err}");
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
        eprintln!("g2g-discover: {err}");
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
            "g2g-discover: built without the `plugin-loader` feature; \
             --plugin / --trusted-key / $G2G_PLUGIN_PATH ignored"
        );
    }
}

/// Print the probe as JSON. The shape lives in `g2g_plugins::toolingjson`,
/// shared with the other dev tools.
#[cfg(feature = "tooling-json")]
fn print_json(info: &Discovery) {
    let value = g2g_plugins::toolingjson::discovery_json(info);
    println!(
        "{}",
        serde_json::to_string_pretty(&value).expect("serialize discovery")
    );
}

#[cfg(not(feature = "tooling-json"))]
fn print_json(_info: &Discovery) {
    eprintln!(
        "g2g-discover: --json needs the `tooling-json` build feature \
         (rebuild with --features tooling-json)"
    );
    process::exit(1);
}

fn main() {
    let (plugin_args, rest) = split_plugin_args(std::env::args().skip(1).collect());
    let mut json = false;
    let mut uri = None;
    for arg in rest {
        match arg.as_str() {
            "--json" => json = true,
            "--help" | "-h" => {
                println!("usage: g2g-discover <path-or-uri> [--json]");
                return;
            }
            _ => uri = Some(arg),
        }
    }
    let Some(uri) = uri else {
        eprintln!("usage: g2g-discover <path-or-uri> [--json]");
        process::exit(2);
    };

    let mut reg = default_registry();
    load_plugins(&mut reg, &plugin_args);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    match rt.block_on(discover(&reg, &uri)) {
        Ok(info) if json => print_json(&info),
        Ok(info) => print!("{}", to_text(&info)),
        Err(err) => {
            eprintln!("g2g-discover: {err}");
            process::exit(1);
        }
    }
}

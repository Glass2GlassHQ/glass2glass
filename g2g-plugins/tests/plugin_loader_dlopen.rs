//! M201: end-to-end dynamic plugin load. Builds the out-of-tree example plugin
//! (`tests/fixtures/example-plugin`) as a `cdylib` exactly as a third party
//! would, `dlopen`s the resulting `.so` into a `Registry`, and runs a pipeline
//! through the loaded `examplefilter` element. The whole "build a plugin with
//! cargo against g2g-devel, drop the .so, run it" path, verified on this host.
//!
//! Requires the `plugin-loader` feature (the loader is gated on it):
//!   cargo test -p g2g-plugins --features plugin-loader --test plugin_loader_dlopen

use std::path::PathBuf;
use std::process::Command;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::clock::WallClock;
use g2g_plugins::plugin_loader::{self, PluginError};
use g2g_plugins::registry::default_registry;

/// The example-plugin fixture crate directory.
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/example-plugin")
}

/// The layout-affecting features in the host's ABI tag (the part after
/// `feat:`), e.g. `["multi-thread"]`. The fixture must be built with the same
/// set or its ABI tag will not match and the loader will (correctly) refuse it.
/// This mirrors a real plugin author building against the host's feature config.
fn host_layout_features() -> Vec<String> {
    let tag = g2g_core::ABI_VERSION;
    let feats = tag
        .rsplit_once("feat:")
        .map(|(_, f)| f.trim())
        .unwrap_or("none");
    if feats == "none" {
        Vec::new()
    } else {
        feats.split(',').map(|s| s.trim().to_string()).collect()
    }
}

/// `cargo build --release` the fixture with the given layout features into the
/// named target subdirectory (separate dirs so concurrent builds with different
/// features do not clobber each other), returning the produced library path.
fn build_fixture_with(features: &[String], target_subdir: &str) -> PathBuf {
    let dir = fixture_dir();
    let target = dir.join(target_subdir);
    let mut cmd = Command::new(env!("CARGO"));
    cmd.arg("build")
        .arg("--release")
        .current_dir(&dir)
        .env("CARGO_TARGET_DIR", &target);
    if !features.is_empty() {
        cmd.arg("--features").arg(features.join(" "));
    }
    let status = cmd
        .status()
        .expect("spawn cargo to build the example plugin");
    assert!(status.success(), "example plugin failed to build");

    // Platform-correct library name: lib<name>.so / .dylib / <name>.dll.
    let name = format!(
        "{}g2g_example_plugin{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let so = target.join("release").join(&name);
    assert!(so.is_file(), "built plugin not found at {}", so.display());
    so
}

/// Build the fixture aligned to the host's ABI (the happy path).
fn build_fixture() -> PathBuf {
    build_fixture_with(&host_layout_features(), "target")
}

#[test]
fn dlopen_plugin_registers_and_runs() {
    let so = build_fixture();

    // The loaded element is absent until the plugin is loaded.
    let mut reg = default_registry();
    assert!(
        !reg.element_names().contains(&"examplefilter"),
        "examplefilter must come from the plugin, not the default registry"
    );

    plugin_loader::load_plugin(&so, &mut reg).expect("plugin loads (ABI must match the host)");

    assert!(
        reg.element_names().contains(&"examplefilter"),
        "the loaded plugin registered `examplefilter`"
    );

    // Run a real pipeline through the dynamically loaded element. Its factory is
    // a fn pointer into the .so, so this also exercises the keep-alive contract.
    let line = "videotestsrc num-buffers=3 ! examplefilter ! fakesink";
    let graph = parse_launch(&reg, line).expect("pipeline using the plugin element parses");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    let stats = rt
        .block_on(run_graph(graph, &WallClock::new(), 4))
        .expect("pipeline runs");
    assert_eq!(
        stats.frames_consumed, 3,
        "all frames flowed through the plugin element"
    );
}

#[test]
fn missing_plugin_fails_to_open() {
    // A path that does not resolve fails cleanly at `dlopen` rather than
    // crashing: the loader probes before trusting.
    let mut reg = default_registry();
    match plugin_loader::load_plugin("/nonexistent/path/to/plugin.so", &mut reg) {
        Err(PluginError::Open { .. }) => {}
        other => panic!("opening a missing plugin should fail with Open, got {other:?}"),
    }
}

#[test]
fn abi_mismatch_is_refused() {
    // The key safety property: a plugin built with a different layout-affecting
    // feature set than the host has a different ABI tag, and the loader refuses
    // it (rather than risk UB passing a differently-sized `Frame` across the
    // boundary). Build the fixture with the OPPOSITE of the host's multi-thread
    // state so the tags are guaranteed to differ, into its own target dir.
    let host = host_layout_features();
    let mismatched: Vec<String> = if host.iter().any(|f| f == "multi-thread") {
        // Host has multi-thread; build the plugin without it.
        host.iter()
            .filter(|f| *f != "multi-thread")
            .cloned()
            .collect()
    } else {
        // Host lacks it; add it to the plugin.
        let mut f = host.clone();
        f.push("multi-thread".to_string());
        f
    };
    let so = build_fixture_with(&mismatched, "target-mismatch");

    let mut reg = default_registry();
    match plugin_loader::load_plugin(&so, &mut reg) {
        Err(PluginError::AbiMismatch { plugin, host, .. }) => {
            assert_ne!(plugin, host, "the tags must actually differ");
        }
        other => panic!("a feature-mismatched plugin must be refused, got {other:?}"),
    }
    assert!(
        !reg.element_names().contains(&"examplefilter"),
        "a refused plugin must not have registered anything"
    );
}

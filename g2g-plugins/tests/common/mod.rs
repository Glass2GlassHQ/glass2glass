//! Building the v2 example plugin, shared by the loader tests that need a real
//! `cdylib` on disk (`plugin_loader_v2`, `m1061_plugin_signing`).

use std::path::PathBuf;
use std::process::Command;

/// The v2 example-plugin fixture crate directory.
pub(crate) fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v2-example-plugin")
}

/// The host's layout-affecting `g2g-core` features, read out of its own ABI tag.
pub(crate) fn host_layout_features() -> Vec<String> {
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

/// The set guaranteed to differ from the host's, by flipping `multi-thread`.
pub(crate) fn mismatched_features() -> Vec<String> {
    let host = host_layout_features();
    if host.iter().any(|f| f == "multi-thread") {
        host.iter()
            .filter(|f| *f != "multi-thread")
            .cloned()
            .collect()
    } else {
        let mut f = host;
        f.push("multi-thread".to_string());
        f
    }
}

/// Build the fixture and return the produced library path. `extra` names
/// additional cargo features; each distinct set gets its own target directory so
/// concurrent test threads do not clobber each other.
pub(crate) fn build_fixture_with(extra: &[&str], target_subdir: &str) -> PathBuf {
    let dir = fixture_dir();
    let target = dir.join(target_subdir);
    let mut cmd = Command::new(env!("CARGO"));
    cmd.arg("build")
        .arg("--release")
        .current_dir(&dir)
        .env("CARGO_TARGET_DIR", &target);
    let mut features = mismatched_features();
    features.extend(extra.iter().map(|f| f.to_string()));
    if !features.is_empty() {
        cmd.arg("--features").arg(features.join(" "));
    }
    let status = cmd.status().expect("spawn cargo to build the v2 plugin");
    assert!(status.success(), "v2 example plugin failed to build");

    let name = format!(
        "{}g2g_v2_example_plugin{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let so = target.join("release").join(&name);
    assert!(so.is_file(), "built plugin not found at {}", so.display());
    so
}

/// The fixture as a well-behaved plugin.
pub(crate) fn build_fixture() -> PathBuf {
    build_fixture_with(&[], "target")
}

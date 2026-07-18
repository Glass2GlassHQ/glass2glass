//! Build script for `g2g-core`: computes the dynamic-plugin ABI tag.
//!
//! Rust has no stable ABI, so a `.so` plugin and the host that `dlopen`s it
//! must agree on (a) the `g2g-core` version, (b) the exact `rustc`, and (c) the
//! layout-affecting feature set, or passing a `Frame` / `Box<dyn ...>` across the
//! boundary is UB. We fold all three into a single string the loader compares.
//!
//! The subtle part is (c): `metadata` changes `FrameMetaSet` / `Frame` size and
//! `multi-thread` changes `ElementBound` (the `Send` bound on the trait
//! objects). A plugin built with a different choice of either is binary
//! incompatible even at the same version + toolchain, so both MUST appear in the
//! tag. Cargo exposes active features to the build script as `CARGO_FEATURE_<NAME>`
//! environment variables, which is how we read them here.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".into());

    // The rustc cargo is driving this build with. Its full version string
    // (incl. commit hash + channel) is what determines layout, so use it whole.
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_version = Command::new(&rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown-rustc".into());

    // Layout-affecting features, in a fixed order so the tag is deterministic.
    // CARGO_FEATURE_<NAME> is set (to "1") only when the feature is active.
    let mut feats: Vec<&str> = Vec::new();
    if std::env::var_os("CARGO_FEATURE_METADATA").is_some() {
        feats.push("metadata");
    }
    if std::env::var_os("CARGO_FEATURE_MULTI_THREAD").is_some() {
        feats.push("multi-thread");
    }
    let feats = if feats.is_empty() {
        "none".to_string()
    } else {
        feats.join(",")
    };

    // The tag a plugin embeds and the loader checks. Pipe-separated so each
    // component is legible in the AbiMismatch error a user sees.
    let tag = format!("g2g-core {version} | {rustc_version} | feat:{feats}");
    println!("cargo:rustc-env=G2G_ABI_VERSION={tag}");
}

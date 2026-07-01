//! Builds the `gstwrap` C helper (`csrc/gstwrap_host.c`) into the crate when the
//! `gstreamer` feature is on. The C file includes the real GStreamer headers (so
//! struct layouts and macros are correct), and pkg-config supplies the include
//! paths and link flags for gstreamer-1.0 + gstreamer-app-1.0.
//!
//! Mirrors `g2g-bridge`'s build.rs. Cargo does not expose package features as
//! `cfg` to build scripts, so the gate is the `CARGO_FEATURE_GSTREAMER` env var
//! Cargo sets when the feature is on. Without it this script does nothing, so the
//! default `no_std` build pulls in neither GStreamer nor a C compiler.

fn main() {
    println!("cargo:rerun-if-changed=csrc/gstwrap_host.c");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GSTREAMER");

    if std::env::var_os("CARGO_FEATURE_GSTREAMER").is_none() {
        return;
    }

    // pkg-config's `.probe()` emits the `cargo:rustc-link-lib` / `-L` lines that
    // link the GStreamer shared libraries the C helper calls into.
    let gst = pkg_config::Config::new()
        .probe("gstreamer-1.0")
        .expect("gstreamer-1.0 dev package (pkg-config) is required for the `gstreamer` feature");
    // gstreamer-app-1.0 for gst_app_src_push_buffer / gst_app_sink_try_pull_sample.
    let app = pkg_config::Config::new()
        .probe("gstreamer-app-1.0")
        .expect("gstreamer-app-1.0 dev package (pkg-config) is required for the `gstreamer` feature");

    let mut build = cc::Build::new();
    build.file("csrc/gstwrap_host.c");
    for path in gst.include_paths.iter().chain(app.include_paths.iter()) {
        build.include(path);
    }
    build.compile("g2g_gstwrap_host");
}

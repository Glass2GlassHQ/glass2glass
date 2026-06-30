//! Builds the GStreamer GObject shell (`csrc/gstglass2glass.c`) into the cdylib
//! when the `gstreamer` feature is on. The C file includes the real GStreamer
//! headers (so struct layouts are correct), and pkg-config supplies the include
//! paths and link flags for gstreamer-1.0 + gstreamer-base-1.0.
//!
//! Cargo does not expose package features as `cfg` to build scripts, so the gate
//! is the `CARGO_FEATURE_GSTREAMER` env var Cargo sets when the feature is on.

fn main() {
    println!("cargo:rerun-if-changed=csrc/gstglass2glass.c");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GSTREAMER");

    if std::env::var_os("CARGO_FEATURE_GSTREAMER").is_none() {
        return;
    }

    let gst = pkg_config::Config::new()
        .probe("gstreamer-1.0")
        .expect("gstreamer-1.0 dev package (pkg-config) is required for the `gstreamer` feature");
    let base = pkg_config::Config::new()
        .probe("gstreamer-base-1.0")
        .expect("gstreamer-base-1.0 dev package (pkg-config) is required");

    let mut build = cc::Build::new();
    build.file("csrc/gstglass2glass.c");
    for path in gst.include_paths.iter().chain(base.include_paths.iter()) {
        build.include(path);
    }
    build.compile("gstglass2glass");

    // The plugin entry points are Rust `#[no_mangle]` exports (see src/ffi.rs),
    // which rustc places in the cdylib's dynamic symbol table; the C object is
    // pulled in because the Rust descriptor references its `plugin_init`. No
    // extra link flags are needed to expose the GStreamer-facing symbols.
}

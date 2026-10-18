//! Builds the `gstwrap` C helper (`csrc/gstwrap_host.c`) into the crate when the
//! `gstreamer` feature is on. The C file includes the real GStreamer headers (so
//! struct layouts and macros are correct), and pkg-config supplies the include
//! paths and link flags for gstreamer-1.0, -app, -allocators and -video.
//!
//! Mirrors `g2g-bridge`'s build.rs. The gate is the `CARGO_FEATURE_GSTREAMER` env
//! var Cargo sets when the feature is on. Without it this script does nothing, so
//! the default `no_std` build pulls in neither GStreamer nor a C compiler.
//!
//! With the `vpx` feature it also runs bindgen over the installed libvpx headers.

fn main() {
    println!("cargo:rerun-if-changed=csrc/gstwrap_host.c");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GSTREAMER");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_JPEGXS");

    // JPEG XS (M605): link Intel SVT-JPEG-XS for `SvtJpegXsEnc` / `Dec`. pkg-config
    // emits the `-L` / `-lSvtJpegxs` lines; the struct layouts the FFI transcribes
    // are guarded by compile-time size assertions in `svtjpegxs.rs`.
    if std::env::var_os("CARGO_FEATURE_JPEGXS").is_some() {
        pkg_config::Config::new()
            .probe("SvtJpegxs")
            .expect("SvtJpegxs dev package (pkg-config) is required for the `jpegxs` feature");
    }

    #[cfg(feature = "vpx")]
    generate_vpx_bindings();

    if std::env::var_os("CARGO_FEATURE_GSTREAMER").is_none() {
        return;
    }

    // pkg-config's `.probe()` emits the `cargo:rustc-link-lib` / `-L` lines that
    // link the GStreamer shared libraries the C helper calls into.
    let gst = pkg_config::Config::new()
        .probe("gstreamer-1.0")
        .expect("gstreamer-1.0 dev package (pkg-config) is required for the `gstreamer` feature");
    // gstreamer-app-1.0 for gst_app_src_push_buffer / gst_app_sink_try_pull_sample.
    let app = pkg_config::Config::new().probe("gstreamer-app-1.0").expect(
        "gstreamer-app-1.0 dev package (pkg-config) is required for the `gstreamer` feature",
    );

    // gstreamer-allocators-1.0 for the dma-buf allocator / memory API.
    let allocators = pkg_config::Config::new()
        .probe("gstreamer-allocators-1.0")
        .expect("gstreamer-allocators-1.0 dev package (pkg-config) is required");
    // gstreamer-video-1.0 for the GstVideoMeta carrying a dma-buf's plane layout.
    let video = pkg_config::Config::new()
        .probe("gstreamer-video-1.0")
        .expect(
            "gstreamer-video-1.0 dev package (pkg-config) is required for the `gstreamer` feature",
        );

    let mut build = cc::Build::new();
    build.file("csrc/gstwrap_host.c");
    for path in gst
        .include_paths
        .iter()
        .chain(app.include_paths.iter())
        .chain(allocators.include_paths.iter())
        .chain(video.include_paths.iter())
    {
        build.include(path);
    }
    build.compile("g2g_gstwrap_host");
}

// Generated against the installed libvpx headers, so the encoder ABI version and
// struct layouts always match the library that gets linked.
#[cfg(feature = "vpx")]
fn generate_vpx_bindings() {
    let vpx = pkg_config::Config::new()
        .probe("vpx")
        .expect("libvpx dev package (pkg-config `vpx`) is required for the `vpx` feature");
    let include_args = vpx
        .include_paths
        .iter()
        .map(|path| format!("-I{}", path.display()));
    let bindings = bindgen::Builder::default()
        .header_contents(
            "vpx_wrapper.h",
            "#include <vpx/vpx_codec.h>\n#include <vpx/vpx_encoder.h>\n#include <vpx/vp8cx.h>\n",
        )
        .clang_args(include_args)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .use_core()
        .prepend_enum_name(false)
        .allowlist_function(
            "vpx_codec_vp8_cx|vpx_codec_vp9_cx|vpx_codec_enc_config_default|vpx_codec_enc_init_ver\
             |vpx_codec_control_|vpx_img_wrap|vpx_codec_encode|vpx_codec_get_cx_data|vpx_codec_destroy",
        )
        .allowlist_type("vp8e_enc_control_id")
        .allowlist_var(
            "VPX_ENCODER_ABI_VERSION|VPX_CODEC_OK|VPX_CODEC_CX_FRAME_PKT\
             |VPX_ERROR_RESILIENT_DEFAULT|VPX_DL_REALTIME|VPX_IMG_FMT_I420",
        )
        .generate()
        .expect("bindgen generates the libvpx encoder bindings");
    let out_dir =
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    bindings
        .write_to_file(out_dir.join("vpx_ffi.rs"))
        .expect("write vpx_ffi.rs to OUT_DIR");
}

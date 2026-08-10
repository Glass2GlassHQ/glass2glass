//! M1010: a glass2glass plugin **written in C**.
//!
//! Compiles `tests/fixtures/c-plugin/plugin.c` against the hand-written header
//! in `g2g-plugin/include/`, `dlopen`s the result, and runs a pipeline through
//! the `cpasser` element it registers.
//!
//! What that proves, in order of how much it matters:
//!
//! 1. The v2 boundary is writable without Rust. No `g2g-core` type, no Rust
//!    ABI, no shared toolchain, and frames still cross with their payload
//!    ownership intact.
//! 2. The header and the Rust type set agree. The C fixture reports `sizeof`
//!    for every ABI struct and the test compares each against the Rust type, so
//!    a field added on one side and forgotten on the other fails here rather
//!    than corrupting memory in a pipeline.
//! 3. A shorter vtable loads. The C element declares a `struct_size` that stops
//!    before the reserved slots and leaves `configure_pipeline` /
//!    `configure_output` null, and the host substitutes its own defaults.
//!
//! Unix only: it shells out to a C compiler with `-shared` / `-dynamiclib`.
//!
//! Requires the `plugin-loader` feature:
//!   cargo test -p g2g-plugins --features plugin-loader --test plugin_c_abi
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::clock::WallClock;
use g2g_plugins::plugin_loader;
use g2g_plugins::registry::default_registry;

/// The C compiler to build the fixture with: `$CC`, else `cc`.
fn compiler() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Compile the C fixture into a shared library and return its path.
fn build_c_plugin() -> PathBuf {
    let source = manifest_dir().join("tests/fixtures/c-plugin/plugin.c");
    let include = manifest_dir().join("../g2g-plugin/include");
    let out_dir = manifest_dir().join("tests/fixtures/c-plugin/build");
    std::fs::create_dir_all(&out_dir).expect("create the C plugin build dir");
    let so = out_dir.join(format!(
        "{}g2gcplugin{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ));

    let shared_flag = if cfg!(target_os = "macos") {
        "-dynamiclib"
    } else {
        "-shared"
    };
    let status = Command::new(compiler())
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-fPIC", "-O1"])
        .arg(shared_flag)
        .arg("-I")
        .arg(&include)
        .arg("-o")
        .arg(&so)
        .arg(&source)
        .status()
        .expect("spawn the C compiler");
    assert!(status.success(), "the C plugin failed to compile");
    assert!(so.is_file(), "no library at {}", so.display());
    so
}

/// `sizeof` for every ABI struct, in the order `g2g_c_plugin_layout` reports.
/// Kept beside the Rust types it is compared against.
fn rust_layout() -> Vec<usize> {
    use g2g_plugin::abi::*;
    use std::mem::size_of;
    vec![
        size_of::<FfiStr>(),
        size_of::<FfiDim>(),
        size_of::<FfiRate>(),
        size_of::<FfiCaps>(),
        size_of::<FfiCapsSet>(),
        size_of::<FfiFrame>(),
        size_of::<FfiPacket>(),
        size_of::<async_ffi::FfiPoll<FfiStatus>>(),
        size_of::<async_ffi::LocalFfiFuture<FfiStatus>>(),
        size_of::<FfiOutputSinkVtable>(),
        size_of::<FfiOutputSink>(),
        size_of::<FfiPropStr>(),
        size_of::<FfiPropValue>(),
        size_of::<FfiPropertySpec>(),
        size_of::<FfiElementMetadata>(),
        size_of::<FfiElementVtable>(),
        size_of::<FfiElementRegistration>(),
        size_of::<FfiRegistrar>(),
        size_of::<FfiCapability>(),
        size_of::<FfiPluginDescriptor>(),
    ]
}

const LAYOUT_NAMES: &[&str] = &[
    "G2gStr",
    "G2gDim",
    "G2gRate",
    "G2gCaps",
    "G2gCapsSet",
    "G2gFrame",
    "G2gPacket",
    "G2gPoll",
    "G2gFuture",
    "G2gOutputSinkVtable",
    "G2gOutputSink",
    "G2gPropStr",
    "G2gPropValue",
    "G2gPropertySpec",
    "G2gElementMetadata",
    "G2gElementVtable",
    "G2gElementRegistration",
    "G2gRegistrar",
    "G2gCapability",
    "G2gPluginDescriptor",
];

/// Read the C side's struct sizes out of the built library.
fn c_layout(so: &Path) -> Vec<usize> {
    let expected = rust_layout().len();
    // SAFETY: loading a library runs its initialisers; this one is built from
    // the fixture source in this repo by the test itself.
    let lib = unsafe { libloading::Library::new(so) }.expect("the C plugin opens");
    // SAFETY: the symbol's type matches `g2g_c_plugin_layout` in plugin.c.
    let probe: libloading::Symbol<unsafe extern "C" fn(*mut usize, usize)> =
        unsafe { lib.get(b"g2g_c_plugin_layout") }.expect("the layout probe is exported");
    let mut out = vec![0usize; expected];
    // SAFETY: `out` has `expected` slots and that is the count passed in.
    unsafe { probe(out.as_mut_ptr(), expected) };
    out
}

#[test]
fn the_c_header_and_the_rust_abi_agree_on_every_struct() {
    // The drift guard. `g2g_plugin_v2.h` is hand-written, so nothing but this
    // stops a field landing on one side only.
    let so = build_c_plugin();
    let from_c = c_layout(&so);
    let from_rust = rust_layout();
    for ((name, c), rust) in LAYOUT_NAMES.iter().zip(&from_c).zip(&from_rust) {
        assert_eq!(
            c, rust,
            "{name}: the C header says {c} bytes, the Rust type is {rust}"
        );
    }
}

#[test]
fn a_c_plugin_loads_and_carries_frames_through_a_pipeline() {
    let so = build_c_plugin();
    let mut reg = default_registry();
    assert!(!reg.element_names().contains(&"cpasser"));

    plugin_loader::load_plugin(&so, &mut reg).expect("a C plugin loads like any other");
    assert!(
        reg.element_names().contains(&"cpasser"),
        "the C plugin registered `cpasser`"
    );

    // Its vtable leaves configure_pipeline and configure_output null; the host
    // fills both with its own default, so negotiation still completes.
    let element = reg.make_element("cpasser").expect("the registry builds it");
    assert_eq!(element.metadata().long_name, "C counting filter");
    drop(element);

    let graph = parse_launch(&reg, "videotestsrc num-buffers=3 ! cpasser ! fakesink")
        .expect("a pipeline using the C element parses");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    let stats = rt
        .block_on(run_graph(graph, &WallClock::new(), 4))
        .expect("pipeline runs");
    assert_eq!(
        stats.frames_consumed, 3,
        "every frame crossed into C and came back out"
    );
}

#[test]
fn a_property_reaches_the_c_element_and_reads_back() {
    let so = build_c_plugin();
    let mut reg = default_registry();
    plugin_loader::load_plugin(&so, &mut reg).expect("the C plugin loads");

    // Read back through the C get_property, then drive the whole thing from a
    // launch line so set_property crosses too.
    let element = reg.make_element("cpasser").expect("the registry builds it");
    assert_eq!(
        element.get_property("enabled"),
        Some(g2g_core::property::PropValue::Bool(true))
    );
    assert_eq!(element.get_property("nonesuch"), None);
    drop(element);

    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=3 ! cpasser enabled=false ! fakesink",
    )
    .expect("pipeline parses");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    let stats = rt
        .block_on(run_graph(graph, &WallClock::new(), 4))
        .expect("pipeline runs");
    assert_eq!(
        stats.frames_consumed, 0,
        "the C element dropped every frame, so the property arrived"
    );
}

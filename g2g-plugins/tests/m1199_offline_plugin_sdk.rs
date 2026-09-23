#![cfg(unix)]

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use g2g_core::runtime::{parse_launch, run_graph, Registry};
use g2g_plugins::clock::WallClock;
use g2g_plugins::plugin_loader;
use g2g_plugins::registry::default_registry;

mod common;
use common::{compile_c_plugin, host_layout_features};

const FRAME_COUNT: u64 = 3;
const LINK_CAPACITY: usize = 4;
const RUST_PLUGIN_PACKAGE: &str = "offline_sdk_plugin";

fn scratch_dir() -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join("m1199-offline-plugin-sdk")
}

fn sdk_prefix() -> &'static Path {
    static STAGED: OnceLock<PathBuf> = OnceLock::new();
    STAGED.get_or_init(|| {
        let scratch = scratch_dir();
        if scratch.exists() {
            std::fs::remove_dir_all(&scratch).expect("clear the previous run");
        }
        let prefix = scratch.join("sdk");
        let bundle_tool =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/plugin-sdk-bundle.sh");
        let status = Command::new("bash")
            .arg(&bundle_tool)
            .arg(&prefix)
            .status()
            .expect("spawn the bundle tool");
        assert!(status.success(), "the bundle tool failed");
        prefix
    })
}

fn frames_through(registry: &Registry, element: &str) -> u64 {
    let line = format!("videotestsrc num-buffers={FRAME_COUNT} ! {element} ! fakesink");
    let graph = parse_launch(registry, &line).expect("pipeline using the plugin element parses");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    runtime
        .block_on(run_graph(graph, &WallClock::new(), LINK_CAPACITY))
        .expect("pipeline runs")
        .frames_consumed
}

fn write_plugin_manifest(project: &Path) {
    let source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/example-plugin/src/lib.rs");
    let mut core_features = vec!["std".to_string(), "runtime".to_string()];
    core_features.extend(host_layout_features());
    let core_features = core_features
        .iter()
        .map(|feature| format!("\"{feature}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let version = env!("CARGO_PKG_VERSION");
    let manifest = format!(
        "[package]\n\
         name = \"{RUST_PLUGIN_PACKAGE}\"\n\
         version = \"0.1.0\"\n\
         edition = \"2021\"\n\
         publish = false\n\
         \n\
         [workspace]\n\
         \n\
         [lib]\n\
         crate-type = [\"cdylib\"]\n\
         path = \"{}\"\n\
         \n\
         [dependencies]\n\
         g2g-core = {{ version = \"{version}\", features = [{core_features}] }}\n\
         g2g-plugin = \"{version}\"\n",
        source.display()
    );
    std::fs::create_dir_all(project).expect("create the plugin project");
    std::fs::write(project.join("Cargo.toml"), manifest).expect("write the plugin manifest");
}

#[test]
fn a_rust_plugin_built_offline_from_the_bundle_loads_and_runs() {
    let prefix = sdk_prefix();
    let project = scratch_dir().join("rust-plugin");
    write_plugin_manifest(&project);
    let empty_cargo_home = scratch_dir().join("empty-cargo-home");
    std::fs::create_dir_all(&empty_cargo_home).expect("create the empty cargo home");
    let target = project.join("target");

    let status = Command::new(env!("CARGO"))
        .args(["build", "--release", "--offline", "--config"])
        .arg(prefix.join("share/g2g/plugin-sdk/config.toml"))
        .current_dir(&project)
        .env("CARGO_HOME", &empty_cargo_home)
        .env("CARGO_TARGET_DIR", &target)
        .status()
        .expect("spawn cargo to build the plugin");
    assert!(status.success(), "the offline plugin build failed");

    let so = target.join("release").join(format!(
        "{}{RUST_PLUGIN_PACKAGE}{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ));
    let mut registry = default_registry();
    plugin_loader::load_plugin(&so, &mut registry)
        .expect("the ABI tag matches, so the bundle carries the host's g2g-core");
    assert_eq!(frames_through(&registry, "examplefilter"), FRAME_COUNT);
}

#[test]
fn a_c_plugin_built_against_the_bundled_header_loads_and_runs() {
    let prefix = sdk_prefix();
    let output = Command::new("pkg-config")
        .args(["--cflags", "g2g-plugin"])
        .env("PKG_CONFIG_PATH", prefix.join("share/pkgconfig"))
        .output()
        .expect("spawn pkg-config");
    assert!(output.status.success(), "pkg-config finds g2g-plugin.pc");
    let include_flags: Vec<OsString> = String::from_utf8(output.stdout)
        .expect("pkg-config prints text")
        .split_whitespace()
        .map(OsString::from)
        .collect();

    let so = compile_c_plugin(&include_flags, &scratch_dir().join("c-plugin"));
    let mut registry = default_registry();
    plugin_loader::load_plugin(&so, &mut registry).expect("the C plugin loads");
    assert_eq!(frames_through(&registry, "cpasser"), FRAME_COUNT);
}

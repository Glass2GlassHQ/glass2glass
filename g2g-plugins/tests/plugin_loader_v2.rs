//! M1010: end-to-end load of a **v2-ABI** plugin.
//!
//! Builds `tests/fixtures/v2-example-plugin` as a standalone `cdylib`,
//! `dlopen`s it into a `Registry`, and runs a pipeline through the loaded
//! `v2counter`.
//!
//! The fixture is built with the **opposite** layout-affecting `g2g-core`
//! feature set from the host, which is precisely what makes a v1 plugin's ABI
//! tag differ and the v1 loader refuse it. Nothing of `g2g-core`'s crosses the
//! v2 boundary, so it must load and run anyway: that is the whole claim.
//!
//! Requires the `plugin-loader` feature:
//!   cargo test -p g2g-plugins --features plugin-loader --test plugin_loader_v2

use std::path::PathBuf;
use std::process::Command;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::clock::WallClock;
use g2g_plugins::plugin_loader::{self, PluginError, PolicyDecision};
use g2g_plugins::registry::default_registry;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v2-example-plugin")
}

/// The host's layout-affecting `g2g-core` features, read out of its own ABI tag.
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

/// The set guaranteed to differ from the host's, by flipping `multi-thread`.
fn mismatched_features() -> Vec<String> {
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
fn build_fixture_with(extra: &[&str], target_subdir: &str) -> PathBuf {
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
fn build_fixture() -> PathBuf {
    build_fixture_with(&[], "target")
}

/// Run three frames through a pipeline containing the loaded element.
fn run(line: &str) -> u64 {
    let mut reg = default_registry();
    let so = build_fixture();
    plugin_loader::load_plugin(&so, &mut reg).expect("a v2 plugin loads regardless of the ABI tag");
    let graph = parse_launch(&reg, line).expect("pipeline parses");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    rt.block_on(run_graph(graph, &WallClock::new(), 4))
        .expect("pipeline runs")
        .frames_consumed
}

#[test]
fn v2_plugin_loads_and_runs_across_an_abi_tag_mismatch() {
    // The fixture is deliberately built against a different g2g-core feature
    // set than the host. Under v1 that is a hard refusal; under v2 it is
    // irrelevant, because nothing of g2g-core's shape crosses the boundary.
    let mut reg = default_registry();
    assert!(
        !reg.element_names().contains(&"v2counter"),
        "v2counter must come from the plugin, not the default registry"
    );

    let so = build_fixture();
    plugin_loader::load_plugin(&so, &mut reg).expect("the v2 plugin loads");
    assert!(
        reg.element_names().contains(&"v2counter"),
        "the loaded plugin registered `v2counter`"
    );

    let graph = parse_launch(&reg, "videotestsrc num-buffers=3 ! v2counter ! fakesink")
        .expect("pipeline using the plugin element parses");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    let stats = rt
        .block_on(run_graph(graph, &WallClock::new(), 4))
        .expect("pipeline runs");
    assert_eq!(
        stats.frames_consumed, 3,
        "every frame crossed the C ABI and came back"
    );
}

#[test]
fn a_property_set_from_the_launch_line_reaches_the_plugin() {
    // `enabled=false` is applied through the vtable's set_property, and the
    // element then drops every frame instead of forwarding it. If the property
    // never arrived, three frames would reach the sink.
    let consumed = run("videotestsrc num-buffers=3 ! v2counter enabled=false ! fakesink");
    assert_eq!(consumed, 0, "the disabled filter swallowed every frame");
}

#[test]
fn the_registry_publishes_the_plugin_element_metadata_and_properties() {
    // The macro builds these from the element type's own `properties()` and
    // `metadata()`, so a `gst-inspect` dump sees a v2 element exactly as it sees
    // a built-in one. Reading `count` back also exercises get_property.
    let so = build_fixture();
    let mut reg = default_registry();
    plugin_loader::load_plugin(&so, &mut reg).expect("the v2 plugin loads");

    let element = reg
        .make_element("v2counter")
        .expect("the registry builds it");
    assert_eq!(element.metadata().long_name, "v2 counting filter");
    let names: Vec<&str> = element.properties().iter().map(|p| p.name).collect();
    assert_eq!(names, ["count", "enabled"]);
    assert!(
        !element.properties()[0].flags.writable,
        "`count` crossed as read-only"
    );
    assert_eq!(
        element.get_property("count"),
        Some(g2g_core::property::PropValue::Uint(0)),
        "a fresh instance has seen nothing"
    );
    assert_eq!(element.get_property("nonesuch"), None);
}

#[test]
fn the_capability_policy_can_refuse_a_plugin_before_it_runs() {
    // The gate: the descriptor declares `v2counter` as a static the host reads
    // with dlsym, so a caller can refuse it without the plugin ever getting
    // control. A refused plugin registers nothing.
    let so = build_fixture();
    let mut reg = default_registry();
    let deny = |declaration: &g2g_plugin::abi::PluginDeclaration| {
        assert_eq!(declaration.name, "g2g-v2-example-plugin");
        assert!(
            declaration.declared_kind("v2counter").is_some(),
            "the element is declared before any plugin code runs"
        );
        PolicyDecision::Deny("not on the allow-list".to_string())
    };
    match plugin_loader::load_plugin_with_policy(&so, &mut reg, &deny) {
        Err(PluginError::PolicyDenied { plugin, reason, .. }) => {
            assert_eq!(plugin, "g2g-v2-example-plugin");
            assert_eq!(reason, "not on the allow-list");
        }
        other => panic!("a denied plugin must not load, got {other:?}"),
    }
    assert!(
        !reg.element_names().contains(&"v2counter"),
        "a refused plugin must not have registered anything"
    );
}

#[test]
fn a_plugin_that_registers_more_than_it_declared_is_refused_whole() {
    // The other half of the gate. This build's `register` adds a second element
    // the descriptor never declared, after a first one that is perfectly valid.
    // The load must fail and commit *nothing*, not keep the good element.
    let so = build_fixture_with(&["undeclared"], "target-undeclared");
    let mut reg = default_registry();
    match plugin_loader::load_plugin(&so, &mut reg) {
        Err(PluginError::V2Invalid { error, .. }) => {
            assert_eq!(
                error.to_string(),
                "plugin registered 'sneaky', which its descriptor never declared"
            );
        }
        other => panic!("an undeclared registration must fail the load, got {other:?}"),
    }
    assert!(
        !reg.element_names().contains(&"v2counter"),
        "the declared element must not be committed either"
    );
    assert!(!reg.element_names().contains(&"sneaky"));
}

#[test]
fn the_v1_loader_still_refuses_what_v2_accepts() {
    // The contrast that gives the test above its meaning: the same feature skew
    // that v2 shrugs off is what the v1 tag check exists to catch. Build the v1
    // example plugin with the mismatched set and confirm it is still refused.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/example-plugin");
    let target = dir.join("target-v2-contrast");
    let mut cmd = Command::new(env!("CARGO"));
    cmd.arg("build")
        .arg("--release")
        .current_dir(&dir)
        .env("CARGO_TARGET_DIR", &target);
    let features = mismatched_features();
    if !features.is_empty() {
        cmd.arg("--features").arg(features.join(" "));
    }
    assert!(
        cmd.status().expect("spawn cargo").success(),
        "v1 example plugin failed to build"
    );
    let name = format!(
        "{}g2g_example_plugin{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let so = target.join("release").join(&name);

    let mut reg = default_registry();
    match plugin_loader::load_plugin(&so, &mut reg) {
        Err(PluginError::AbiMismatch { .. }) => {}
        other => panic!("a feature-mismatched v1 plugin must be refused, got {other:?}"),
    }
}

//! M944: a saved `v4l2src` line names its camera by the device monitor's
//! persistent id and sets camera controls, both through `parse_launch`. The id
//! contains spaces and colons (`bus:card:path`), so this is also the check
//! that a quoted property value survives the parser.
//!
//! Run with the feature: `cargo test -p g2g-plugins --features v4l2 --test
//! m944_device_id_controls`. Self-skips where no camera is attached (CI);
//! the resolution and control paths themselves are unit-tested in
//! `v4l2device` / `v4l2src`.

#![cfg(all(target_os = "linux", feature = "v4l2"))]

use g2g_core::runtime::{parse_launch, DeviceProvider};
use g2g_plugins::registry::default_registry;
use g2g_plugins::v4l2device::{resolve_device_id, V4l2DeviceProvider};

#[test]
fn a_saved_line_selects_the_camera_by_id_and_sets_controls() {
    let devices = V4l2DeviceProvider::new().probe().expect("probe");
    let Some(camera) = devices.first() else {
        return;
    };

    // the id the monitor prints resolves to the node the same probe reports.
    assert_eq!(
        resolve_device_id(&camera.persistent_id).as_deref(),
        Some(camera.props[0].1.as_str())
    );

    let line = format!(
        "v4l2src device-id=\"{}\" white-balance-temperature-auto=true num-buffers=1 ! fakesink",
        camera.persistent_id
    );
    let registry = default_registry();
    // `num-buffers` is not a v4l2src property, so this doubles as the check
    // that an unknown property is rejected rather than silently dropped.
    assert!(parse_launch(&registry, &line).is_err());

    let line = format!(
        "v4l2src device-id=\"{}\" white-balance-temperature-auto=true exposure-auto=1 ! fakesink",
        camera.persistent_id
    );
    parse_launch(&registry, &line).expect("a saved device-id line must parse and build");
}

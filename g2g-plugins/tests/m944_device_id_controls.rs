//! M944: a saved `v4l2src` line names its camera by the device monitor's
//! persistent id and sets camera controls, both through `parse_launch`. The id
//! contains spaces and colons (`bus:card:path`), so this is also the check
//! that a quoted property value survives the parser.
//!
//! M1047 widens the control set the line can carry, and the monitor describes
//! each camera's own controls with the range it accepts.
//!
//! Run with the feature: `cargo test -p g2g-plugins --features v4l2 --test
//! m944_device_id_controls`. The device-id tests self-skip where no camera is
//! attached (CI); the resolution and control paths themselves are unit-tested
//! in `v4l2device` / `v4l2src`.

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
        "v4l2src device-id=\"{}\" white-balance-temperature-auto=true \
         pixel-aspect-ratio=1/1 ! fakesink",
        camera.persistent_id
    );
    let registry = default_registry();
    // `pixel-aspect-ratio` is a gst v4l2src property this element does not
    // implement, so this doubles as the check that an unknown property is
    // rejected rather than silently dropped.
    assert!(parse_launch(&registry, &line).is_err());

    let line = format!(
        "v4l2src device-id=\"{}\" white-balance-temperature-auto=true exposure-auto=1 ! fakesink",
        camera.persistent_id
    );
    parse_launch(&registry, &line).expect("a saved device-id line must parse and build");
}

/// The whole widened control set is settable from a launch line, with the sign
/// the control's range carries. No camera needed: `parse_launch` configures the
/// element, it does not open the device.
#[test]
fn a_launch_line_carries_every_camera_control() {
    let registry = default_registry();
    let line = "v4l2src device=/dev/video0 brightness=-32 contrast=40 saturation=64 hue=-180 \
                gamma=120 gain=8 sharpness=3 backlight-compensation=1 power-line-frequency=1 \
                zoom-absolute=100 pan-absolute=-3600 tilt-absolute=3600 ! fakesink";
    parse_launch(&registry, line).expect("every control must be settable from a launch line");

    // a control that has no signed range takes no negative value, and a
    // property this element does not implement is still refused.
    for refused in [
        "v4l2src device=/dev/video0 zoom-absolute=-1 ! fakesink",
        "v4l2src device=/dev/video0 colour-balance=1 ! fakesink",
    ] {
        assert!(
            parse_launch(&registry, refused).is_err(),
            "{refused} must not parse"
        );
    }
}

/// `extra-controls` reaches the parser as one quoted value even though its own
/// pairs contain `=`, and a malformed list fails the line rather than being
/// dropped.
#[test]
fn a_launch_line_carries_an_extra_controls_list() {
    let registry = default_registry();
    parse_launch(
        &registry,
        "v4l2src device=/dev/video0 extra-controls=\"contrast=40,privacy=1\" ! fakesink",
    )
    .expect("a quoted extra-controls list must parse");

    for refused in [
        "v4l2src device=/dev/video0 extra-controls=\"contrast\" ! fakesink",
        "v4l2src device=/dev/video0 extra-controls=\"contrast=high\" ! fakesink",
    ] {
        assert!(
            parse_launch(&registry, refused).is_err(),
            "{refused} must not parse"
        );
    }
}

/// The monitor describes each camera's controls with the range the driver
/// accepts, under the name an `extra-controls` entry spells. Self-skips where
/// no camera is attached (CI).
#[test]
fn the_monitor_describes_the_cameras_controls() {
    let devices = V4l2DeviceProvider::new().probe().expect("probe");
    let Some(camera) = devices.first() else {
        return;
    };
    let controls: Vec<&(String, String)> = camera
        .detail
        .iter()
        .filter(|(key, _)| key.starts_with("control."))
        .collect();
    assert!(
        !controls.is_empty(),
        "a camera reports controls: {:?}",
        camera.detail
    );
    for (key, range) in controls {
        let name = key
            .strip_prefix("control.")
            .expect("filtered on the prefix");
        assert!(!name.is_empty(), "{key}");
        assert!(
            name.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
            "{key} is not the name an extra-controls entry spells"
        );
        assert!(
            range.contains("step") && range.contains("default"),
            "{range}"
        );
    }
}

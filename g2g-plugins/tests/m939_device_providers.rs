//! M939: the standard device-provider assembly. `default_device_monitor`
//! probes every compiled-in backend on the real host, and each discovered
//! device is well formed and (when it names an element) buildable through the
//! same `default_registry` a launch line uses. Providers activate per feature,
//! so run this with the device features on (v4l2, alsa-src/alsa-sink,
//! pipewire, wgpu-sink, cuda, vaapi); with only `std` it checks the empty
//! assembly.
#![cfg(feature = "std")]

use g2g_plugins::devicemon::default_device_monitor;
use g2g_plugins::registry::default_registry;

#[test]
fn default_monitor_probes_and_devices_are_well_formed() {
    let registry = default_registry();
    let outcome = default_device_monitor().probe();

    // a backend that is not reachable here (no daemon, no hardware) may
    // error, but never with an empty provider name
    for (provider, err) in &outcome.errors {
        assert!(!provider.is_empty(), "unnamed provider errored: {err:?}");
    }

    for device in &outcome.devices {
        assert!(!device.display_name.is_empty(), "{device:?}");
        assert!(!device.klass.is_empty(), "{device:?}");
        assert!(!device.persistent_id.is_empty(), "{device:?}");
        assert!(!device.provider.is_empty(), "{device:?}");
        // devices split on '/' classes; each part is non-empty
        assert!(device.klass.split('/').all(|part| !part.is_empty()));
        // a device that names an element must name one the registry can
        // introspect (and so build); "" is the documented informational case
        if !device.element.is_empty() {
            assert!(
                registry.inspect(device.element).is_some(),
                "{} names unregistered element {}",
                device.persistent_id,
                device.element
            );
        }
    }

    // persistent ids are the monitor's diff key: unique per provider
    let mut ids: Vec<(&str, &str)> = outcome
        .devices
        .iter()
        .map(|d| (d.provider, d.persistent_id.as_str()))
        .collect();
    let before = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(before, ids.len(), "duplicate (provider, persistent_id)");
}

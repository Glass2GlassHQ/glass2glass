//! Standard device-provider assembly (M939): [`default_device_monitor`]
//! mirrors [`registry::default_registry`](crate::registry::default_registry),
//! registering every discovery backend its feature + target allows, so an
//! application (or the `g2g-device-monitor` binary) discovers the same devices
//! a launch line's elements can open.

use g2g_core::runtime::DeviceMonitor;

/// A [`DeviceMonitor`] with every compiled-in provider registered. Filters and
/// the poll interval are left at their defaults; callers add their own.
pub fn default_device_monitor() -> DeviceMonitor {
    #[allow(unused_mut)]
    let mut monitor = DeviceMonitor::new();
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    monitor.register(alloc::boxed::Box::new(
        crate::v4l2device::V4l2DeviceProvider::new(),
    ));
    #[cfg(all(target_os = "linux", any(feature = "alsa-sink", feature = "alsa-src")))]
    monitor.register(alloc::boxed::Box::new(
        crate::alsadevice::AlsaDeviceProvider::new(),
    ));
    #[cfg(all(target_os = "linux", feature = "pipewire"))]
    monitor.register(alloc::boxed::Box::new(
        crate::pwdevice::PipeWireDeviceProvider::new(),
    ));
    #[cfg(any(feature = "wgpu-sink", feature = "cuda", feature = "vaapi"))]
    monitor.register(alloc::boxed::Box::new(
        crate::gpudevice::GpuDeviceProvider::new(),
    ));
    monitor
}

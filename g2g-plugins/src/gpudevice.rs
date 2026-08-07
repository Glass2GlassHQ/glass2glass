//! GPU / compute device discovery (M939): the [`DeviceProvider`] for the
//! `Compute/GPU` class, g2g's extension beyond GStreamer's capture / render
//! device model.
//!
//! One section per compiled feature, concatenated by [`GpuDeviceProvider::probe`]:
//!
//! - `wgpu-sink`: every `wgpu` adapter over `Backends::all()` (Vulkan, GL,
//!   Dx12, Metal, ...), so the same GPU legitimately shows up once per backend
//!   that can drive it.
//! - `cuda` (Linux): every CUDA driver-API device ordinal.
//! - `vaapi` (Linux): every `/dev/dri/renderD*` node.
//!
//! Only the VAAPI nodes name an element (`vaapidec device=...`). A wgpu adapter
//! or a CUDA ordinal is informational: no element in the tree takes an adapter
//! index or a device ordinal as a property today, so those devices carry an
//! empty [`Device::element`] and [`Device::create`] on them fails with
//! `CapsMismatch`. They are still worth listing: the monitor is how an
//! application (and `g2g-device-monitor`) answers "what compute hardware is
//! here", and the ids match what the GPU elements print.
//!
//! Caps are empty on every device: a compute device has no single stream
//! format to advertise, and the codec support of a GPU is probed per element
//! (see `vulkanvideo::probe_decode_caps`), not per device.
//!
//! A section that finds nothing (no driver, no `/dev/dri`) contributes nothing
//! rather than failing the probe: a machine with no GPU is not an error.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::runtime::{Device, DeviceProvider};
use g2g_core::{CapsSet, G2gError};

/// Provider name carried on every device this module produces.
const PROVIDER: &str = "gpu";

/// Class of every device this module produces.
const KLASS: &str = "Compute/GPU";

/// Lists the GPU / compute devices the compiled-in backends can see. See the
/// module docs for what each feature contributes.
#[derive(Debug, Default, Clone, Copy)]
pub struct GpuDeviceProvider;

impl GpuDeviceProvider {
    /// A provider with no configuration; every section is chosen at compile
    /// time by feature.
    pub fn new() -> Self {
        Self
    }
}

impl DeviceProvider for GpuDeviceProvider {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    fn probe(&self) -> Result<Vec<Device>, G2gError> {
        #[allow(unused_mut)]
        let mut devices = Vec::new();
        #[cfg(feature = "wgpu-sink")]
        devices.extend(probe_wgpu());
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        devices.extend(probe_cuda());
        #[cfg(all(target_os = "linux", feature = "vaapi"))]
        devices.extend(probe_vaapi());
        Ok(devices)
    }
}

/// Build one `Compute/GPU` device with this provider's fixed fields.
fn gpu_device(
    display_name: String,
    persistent_id: String,
    element: &'static str,
    props: Vec<(String, String)>,
    detail: Vec<(String, String)>,
) -> Device {
    Device {
        display_name,
        klass: KLASS.to_string(),
        persistent_id,
        caps: CapsSet::from_alternatives(vec![]),
        element,
        props,
        detail,
        provider: PROVIDER,
    }
}

// --- wgpu adapters ---

/// Every wgpu adapter, one device each, deduplicated by persistent id.
#[cfg(feature = "wgpu-sink")]
fn probe_wgpu() -> Vec<Device> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    // enumerate_adapters is async but does no IO worth yielding for; a machine
    // with no working driver yields an empty list rather than an error.
    let adapters = g2g_core::runtime::block_on(instance.enumerate_adapters(wgpu::Backends::all()));

    let mut devices: Vec<Device> = Vec::new();
    for adapter in adapters {
        let info = adapter.get_info();
        let device = wgpu_device(&info);
        // two physically identical cards share vendor+device+name and so
        // collide here; the wgpu info has no per-adapter serial to separate
        // them, and a duplicate persistent id would confuse the monitor's
        // add/remove diff more than a missing second entry does.
        if devices
            .iter()
            .any(|d| d.persistent_id == device.persistent_id)
        {
            continue;
        }
        devices.push(device);
    }
    devices
}

/// Map one adapter's info onto a device: id from the backend + PCI ids + name,
/// everything else display-only detail.
#[cfg(feature = "wgpu-sink")]
fn wgpu_device(info: &wgpu::AdapterInfo) -> Device {
    let backend = info.backend.to_str();
    let mut detail = vec![
        ("backend".to_string(), backend.to_string()),
        (
            "device-type".to_string(),
            device_type_name(info.device_type).to_string(),
        ),
        ("vendor".to_string(), format!("0x{:04x}", info.vendor)),
        ("device".to_string(), format!("0x{:04x}", info.device)),
    ];
    if !info.driver.is_empty() {
        detail.push(("driver".to_string(), info.driver.clone()));
    }
    if !info.device_pci_bus_id.is_empty() {
        detail.push(("pci-bus-id".to_string(), info.device_pci_bus_id.clone()));
    }
    gpu_device(
        info.name.clone(),
        wgpu_persistent_id(backend, info.vendor, info.device, &info.name),
        "",
        Vec::new(),
        detail,
    )
}

/// `wgpu:<backend>:<vendor>:<device>:<name>`. The PCI ids alone do not separate
/// the same card seen through two backends, and the name alone is not unique.
#[cfg(feature = "wgpu-sink")]
fn wgpu_persistent_id(backend: &str, vendor: u32, device: u32, name: &str) -> String {
    format!("wgpu:{backend}:{vendor:04x}:{device:04x}:{name}")
}

/// Lowercase name for the adapter's device type. Software adapters (llvmpipe,
/// WARP) are listed like any other: they are usable compute devices, the detail
/// says what they are.
#[cfg(feature = "wgpu-sink")]
fn device_type_name(kind: wgpu::DeviceType) -> &'static str {
    match kind {
        wgpu::DeviceType::DiscreteGpu => "discrete-gpu",
        wgpu::DeviceType::IntegratedGpu => "integrated-gpu",
        wgpu::DeviceType::VirtualGpu => "virtual-gpu",
        wgpu::DeviceType::Cpu => "cpu",
        wgpu::DeviceType::Other => "other",
    }
}

// --- CUDA devices ---

/// Every CUDA device ordinal the driver reports. No libcuda (or no NVIDIA
/// driver loaded) means no devices, not an error.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn probe_cuda() -> Vec<Device> {
    // SAFETY: cuInit takes only a flags word (0 is the only legal value) and is
    // idempotent, so calling it again after another element already did is
    // defined. A non-zero result means no usable driver.
    let init = unsafe { ffi::cu_init(0) };
    if init != ffi::CUDA_SUCCESS {
        return Vec::new();
    }
    let mut count: i32 = 0;
    // SAFETY: `count` is a live local i32, the out-param the call expects.
    let rc = unsafe { ffi::cu_device_get_count(&mut count) };
    if rc != ffi::CUDA_SUCCESS || count <= 0 {
        return Vec::new();
    }
    (0..count).filter_map(cuda_device).collect()
}

/// One device per ordinal. `None` when the driver refuses the ordinal (a device
/// disappearing between the count and the query).
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn cuda_device(ordinal: i32) -> Option<Device> {
    let mut handle: ffi::CuDevice = 0;
    // SAFETY: `handle` is a live local of the handle type; ordinal came from
    // cuDeviceGetCount.
    if unsafe { ffi::cu_device_get(&mut handle, ordinal) } != ffi::CUDA_SUCCESS {
        return None;
    }
    let mut buf = [0u8; CUDA_NAME_LEN];
    // SAFETY: the call writes at most `len` bytes into `name`; we pass the real
    // length of a live stack buffer and cast to the c_char the ABI wants.
    let rc = unsafe {
        ffi::cu_device_get_name(
            buf.as_mut_ptr() as *mut core::ffi::c_char,
            CUDA_NAME_LEN as i32,
            handle,
        )
    };
    if rc != ffi::CUDA_SUCCESS {
        return None;
    }
    let name = cuda_name(&buf);
    Some(gpu_device(
        name.clone(),
        // ordinal-keyed: imperfect (ordinals renumber when a card is removed)
        // but it is the handle every CUDA element property consumes.
        format!("cuda:{ordinal}:{name}"),
        "",
        Vec::new(),
        vec![("ordinal".to_string(), ordinal.to_string())],
    ))
}

/// Name buffer size cuDeviceGetName is called with. 256 is what the CUDA
/// samples use; the driver truncates and NUL-terminates within it.
#[cfg(all(target_os = "linux", feature = "cuda"))]
const CUDA_NAME_LEN: usize = 256;

/// Decode a NUL-terminated C name buffer: cut at the first NUL, drop trailing
/// whitespace. Not UTF-8 by contract, so lossy.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn cuda_name(buf: &[u8]) -> String {
    let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).trim().to_string()
}

/// Hand-rolled driver-API bindings, matching `cuda.rs`: the `cuda` feature's
/// gate guarantees Linux + an NVIDIA driver, so linking `libcuda` directly
/// beats pulling a wrapper crate for three calls.
#[cfg(all(target_os = "linux", feature = "cuda"))]
mod ffi {
    use core::ffi::c_char;

    /// `CUresult` is a C `enum` (int-sized).
    pub(super) type CuResult = i32;
    /// `CUdevice` is an int handle.
    pub(super) type CuDevice = i32;

    pub(super) const CUDA_SUCCESS: CuResult = 0;

    #[link(name = "cuda")]
    extern "C" {
        /// Initialise the CUDA driver API (flags must be 0). Idempotent.
        #[link_name = "cuInit"]
        pub(super) fn cu_init(flags: u32) -> CuResult;
        /// Number of devices with compute capability, through `count`.
        #[link_name = "cuDeviceGetCount"]
        pub(super) fn cu_device_get_count(count: *mut i32) -> CuResult;
        /// Get a device handle by ordinal.
        #[link_name = "cuDeviceGet"]
        pub(super) fn cu_device_get(device: *mut CuDevice, ordinal: i32) -> CuResult;
        /// Write the device's NUL-terminated name into `name` (at most `len`).
        #[link_name = "cuDeviceGetName"]
        pub(super) fn cu_device_get_name(name: *mut c_char, len: i32, dev: CuDevice) -> CuResult;
    }
}

// --- VAAPI render nodes ---

/// One device per DRM render node. An unreadable or absent `/dev/dri` yields
/// nothing.
#[cfg(all(target_os = "linux", feature = "vaapi"))]
fn probe_vaapi() -> Vec<Device> {
    let Ok(entries) = std::fs::read_dir(DRI_DIR) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|name| is_render_node(name))
        .collect();
    names.sort();
    names
        .iter()
        .map(|name| vaapi_device(&format!("{DRI_DIR}/{name}")))
        .collect()
}

#[cfg(all(target_os = "linux", feature = "vaapi"))]
const DRI_DIR: &str = "/dev/dri";

/// `renderD<n>`, the unprivileged DRM node VA-API opens. `card<n>` is the
/// primary node (needs master / a seat) and is not one of these.
#[cfg(all(target_os = "linux", feature = "vaapi"))]
fn is_render_node(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("renderD") else {
        return false;
    };
    !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit())
}

/// Map a render-node path onto a `vaapidec device=<path>` device.
#[cfg(all(target_os = "linux", feature = "vaapi"))]
fn vaapi_device(path: &str) -> Device {
    gpu_device(
        format!("VAAPI render node {path}"),
        // minor numbers can move across boots when GPUs are added or the
        // probe order changes; the path is still the best handle VA-API takes.
        path.to_string(),
        "vaapidec",
        vec![("device".to_string(), path.to_string())],
        Vec::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_lists_only_compute_gpu_devices() {
        let devices = GpuDeviceProvider::new().probe().expect("probe");
        for device in &devices {
            assert_eq!(device.provider, "gpu");
            assert!(device.has_classes("Compute/GPU"), "{device:?}");
            assert!(!device.persistent_id.is_empty(), "{device:?}");
            assert!(device.caps.alternatives().is_empty(), "{device:?}");
        }
        // ids identify devices for the monitor's diff, so they must be unique
        // even across sections.
        for (i, device) in devices.iter().enumerate() {
            assert!(
                !devices[..i]
                    .iter()
                    .any(|d| d.persistent_id == device.persistent_id),
                "duplicate id {}",
                device.persistent_id
            );
        }
    }

    #[cfg(feature = "wgpu-sink")]
    #[test]
    fn wgpu_id_is_backend_and_pci_keyed() {
        assert_eq!(
            wgpu_persistent_id("vulkan", 0x10de, 0x2504, "NVIDIA GeForce RTX 3060"),
            "wgpu:vulkan:10de:2504:NVIDIA GeForce RTX 3060"
        );
        // the same card under another backend is a different device row
        assert_ne!(
            wgpu_persistent_id("gl", 0x10de, 0x2504, "NVIDIA GeForce RTX 3060"),
            wgpu_persistent_id("vulkan", 0x10de, 0x2504, "NVIDIA GeForce RTX 3060")
        );
    }

    #[cfg(feature = "wgpu-sink")]
    #[test]
    fn wgpu_software_adapters_are_listed_and_labelled() {
        assert_eq!(device_type_name(wgpu::DeviceType::Cpu), "cpu");
        assert_eq!(
            device_type_name(wgpu::DeviceType::DiscreteGpu),
            "discrete-gpu"
        );
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    #[test]
    fn cuda_name_stops_at_nul() {
        let mut buf = [0u8; 16];
        buf[..4].copy_from_slice(b"A100");
        buf[6] = b'X';
        assert_eq!(cuda_name(&buf), "A100");
        assert_eq!(cuda_name(b"NVIDIA GeForce  \0\0"), "NVIDIA GeForce");
        assert_eq!(cuda_name(&[0u8; 8]), "");
        // no NUL at all: take the whole buffer
        assert_eq!(cuda_name(b"RTX 3060"), "RTX 3060");
    }

    #[cfg(all(target_os = "linux", feature = "vaapi"))]
    #[test]
    fn only_render_nodes_are_devices() {
        assert!(is_render_node("renderD128"));
        assert!(is_render_node("renderD129"));
        assert!(!is_render_node("card0"));
        assert!(!is_render_node("renderD"));
        assert!(!is_render_node("renderDx"));
        assert!(!is_render_node("by-path"));
    }

    #[cfg(all(target_os = "linux", feature = "vaapi"))]
    #[test]
    fn vaapi_device_is_launchable() {
        let device = vaapi_device("/dev/dri/renderD128");
        assert_eq!(
            device.launch_fragment(),
            "vaapidec device=/dev/dri/renderD128"
        );
        assert_eq!(device.persistent_id, "/dev/dri/renderD128");
    }
}

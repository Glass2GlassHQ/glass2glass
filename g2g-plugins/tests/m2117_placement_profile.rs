#![cfg(all(target_os = "linux", feature = "placement"))]

#[path = "common/gpu_profile.rs"]
mod gpu_profile;

use g2g_core::memory::GpuVendor;
use g2g_plugins::capsfilter::CapsFilter;
use g2g_plugins::hipconvertscale::HipConvertScale;
use g2g_plugins::hipdownload::HipDownload;
use g2g_plugins::hipupload::HipUpload;
use g2g_plugins::placement::PlacementProfile;
use gpu_profile::{has_device, profile, profile_path, INPUT_WIDTH};

#[tokio::test]
async fn a_profile_run_times_each_gpu_element_on_the_device_it_ran_on() {
    if !has_device(GpuVendor::Amd) {
        return;
    }
    let path = profile_path("m2117-one-gpu");
    let written = profile("amd", &path).await;
    std::fs::remove_file(&path).expect("the profile is removed");

    let elements: Vec<&str> = written
        .elements
        .iter()
        .map(|timing| timing.key.element.as_str())
        .collect();
    assert!(
        elements.contains(&std::any::type_name::<HipUpload>()),
        "{elements:?}"
    );
    let caps_filter = written
        .elements
        .iter()
        .find(|timing| timing.key.element == std::any::type_name::<CapsFilter>())
        .expect("elements that run on no GPU are timed too");
    assert_eq!(caps_filter.key.device, None);
    let convert = written
        .elements
        .iter()
        .find(|timing| timing.key.element == std::any::type_name::<HipConvertScale>())
        .expect("hipconvertscale follows hipupload onto the AMD GPU and is timed there");
    let device = convert
        .key
        .device
        .as_ref()
        .expect("the convert ran on a GPU");
    assert_eq!(device.vendor, GpuVendor::Amd);
    assert!(!device.name.is_empty());
    assert!(
        convert.key.input_caps[0].contains(&format!("width={INPUT_WIDTH}")),
        "{:?}",
        convert.key.input_caps
    );
    let download = written
        .elements
        .iter()
        .find(|timing| timing.key.element == std::any::type_name::<HipDownload>())
        .expect("hipdownload is timed");
    assert_eq!(
        download.key.device.as_ref().map(|device| device.vendor),
        Some(GpuVendor::Amd),
        "hipdownload is timed on the GPU its frames arrive from"
    );
    assert!(convert.cost.host_ns > 0, "{:?}", convert.cost);
    assert!(
        convert.cost.gpu_ns > convert.cost.host_ns,
        "{:?}",
        convert.cost
    );
}

#[tokio::test]
async fn runs_on_two_gpus_merge_into_one_profile_with_copies_both_ways() {
    if !has_device(GpuVendor::Amd) || !has_device(GpuVendor::Nvidia) {
        return;
    }
    let path = profile_path("m2117-two-gpus");
    profile("amd", &path).await;
    profile("nvidia", &path).await;
    let merged = PlacementProfile::read(&path).expect("the profile reads back");
    std::fs::remove_file(&path).expect("the profile is removed");

    let convert_vendors: Vec<GpuVendor> = merged
        .elements
        .iter()
        .filter(|timing| timing.key.element == std::any::type_name::<HipConvertScale>())
        .filter_map(|timing| Some(timing.key.device.as_ref()?.vendor))
        .collect();
    assert_eq!(convert_vendors, [GpuVendor::Amd, GpuVendor::Nvidia]);
    let directions: Vec<(GpuVendor, GpuVendor)> = merged
        .copies
        .iter()
        .map(|copy| (copy.from.vendor, copy.to.vendor))
        .collect();
    assert_eq!(
        directions,
        [
            (GpuVendor::Amd, GpuVendor::Nvidia),
            (GpuVendor::Nvidia, GpuVendor::Amd)
        ]
    );
    assert!(merged.copies.iter().all(|copy| copy.bytes_per_second > 0.0));
}

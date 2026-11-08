#![cfg(all(target_os = "linux", feature = "placement"))]

use std::path::{Path, PathBuf};

use g2g_core::memory::GpuVendor;
use g2g_core::runtime::{parse_launch, RunBuilder};
use g2g_plugins::capsfilter::CapsFilter;
use g2g_plugins::clock::WallClock;
use g2g_plugins::hip::HipDevice;
use g2g_plugins::hipconvertscale::HipConvertScale;
use g2g_plugins::hipupload::HipUpload;
use g2g_plugins::placement::{PlacementProfile, ProfileRun};
use g2g_plugins::registry::default_registry;

const LINK_CAPACITY: usize = 4;
const INPUT_WIDTH: u32 = 1920;

fn line(vendor: &str) -> String {
    format!(
        "videotestsrc num-buffers=40 ! video/x-raw,format=RGBA,width={INPUT_WIDTH},height=1080 \
         ! hipupload vendor={vendor} device-id=0 ! hipconvertscale \
         ! video/x-raw,format=RGBA,width=640,height=640 ! hipdownload ! fakesink"
    )
}

async fn profile(vendor: &str, path: &Path) -> PlacementProfile {
    let mut graph = parse_launch(&default_registry(), &line(vendor)).expect("the line parses");
    let run = ProfileRun::prepare(&mut graph);
    let clock = WallClock::new();
    RunBuilder::new(graph, &clock, LINK_CAPACITY)
        .with_observer(run.observer())
        .run()
        .await
        .expect("the pipeline runs to EOS");
    run.record(path).expect("the profile is written")
}

fn has_device(vendor: GpuVendor) -> bool {
    match HipDevice::open(vendor, 0) {
        Ok(_) => true,
        Err(error) => {
            eprintln!("skipping: no {vendor:?} HIP device 0: {error:?}");
            false
        }
    }
}

fn profile_path(test: &str) -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{test}-{}.json", std::process::id()))
}

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
    assert!(
        !elements.contains(&std::any::type_name::<CapsFilter>()),
        "{elements:?}"
    );
    let convert = written
        .elements
        .iter()
        .find(|timing| timing.key.element == std::any::type_name::<HipConvertScale>())
        .expect("hipconvertscale follows hipupload onto the AMD GPU and is timed there");
    assert_eq!(convert.key.device.vendor, GpuVendor::Amd);
    assert!(!convert.key.device.name.is_empty());
    assert!(
        convert.key.input_caps[0].contains(&format!("width={INPUT_WIDTH}")),
        "{:?}",
        convert.key.input_caps
    );
    assert!(convert.median_ns > 0);
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
        .map(|timing| timing.key.device.vendor)
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

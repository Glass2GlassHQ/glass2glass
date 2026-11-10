// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::path::{Path, PathBuf};

use g2g_core::memory::GpuVendor;
use g2g_core::runtime::{parse_launch, RunBuilder};
use g2g_plugins::clock::WallClock;
use g2g_plugins::hip::HipDevice;
use g2g_plugins::placement::{PlacementProfile, ProfileRun};
use g2g_plugins::registry::default_registry;
use g2g_plugins::TokioThreadSpawner;

pub(crate) const LINK_CAPACITY: usize = 4;
pub(crate) const INPUT_WIDTH: u32 = 1920;

pub(crate) fn line(vendor: &str) -> String {
    format!(
        "videotestsrc num-buffers=40 ! video/x-raw,format=RGBA,width={INPUT_WIDTH},height=1080 \
         ! hipupload vendor={vendor} device-id=0 ! hipconvertscale \
         ! video/x-raw,format=RGBA,width=640,height=640 ! hipdownload ! fakesink"
    )
}

pub(crate) async fn profile(vendor: &str, path: &Path) -> PlacementProfile {
    let mut graph = parse_launch(&default_registry(), &line(vendor)).expect("the line parses");
    let run = ProfileRun::prepare(&mut graph);
    let clock = WallClock::new();
    RunBuilder::new(graph, &clock, LINK_CAPACITY)
        .with_observer(run.observer())
        .run_threaded(&TokioThreadSpawner)
        .await
        .expect("the pipeline runs to EOS");
    run.record(path).expect("the profile is written")
}

pub(crate) fn has_device(vendor: GpuVendor) -> bool {
    match HipDevice::open(vendor, 0) {
        Ok(_) => true,
        Err(error) => {
            eprintln!("skipping: no {vendor:?} HIP device 0: {error:?}");
            false
        }
    }
}

pub(crate) fn profile_path(test: &str) -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{test}-{}.json", std::process::id()))
}

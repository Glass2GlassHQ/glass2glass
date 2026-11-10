// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "placement"))]

#[path = "common/gpu_profile.rs"]
mod gpu_profile;

use std::path::Path;

use g2g_core::memory::GpuVendor;
use g2g_core::runtime::{parse_launch, RunBuilder};
use g2g_core::PropValue;
use g2g_plugins::clock::WallClock;
use g2g_plugins::hipconvertscale::HipConvertScale;
use g2g_plugins::placement::{
    place_launch_graph, PlacementError, PlacementObjective, PlacementProfile, RunnerThreads,
};
use g2g_plugins::registry::default_registry;
use gpu_profile::{has_device, line, profile, profile_path, LINK_CAPACITY};

const SLOWER_THAN_ANY_GPU_NS: u64 = 1_000_000_000;

async fn profile_both_gpus(test: &str) -> Option<std::path::PathBuf> {
    if !has_device(GpuVendor::Amd) || !has_device(GpuVendor::Nvidia) {
        return None;
    }
    let path = profile_path(test);
    profile("amd", &path).await;
    profile("nvidia", &path).await;
    Some(path)
}

fn rewrite(path: &Path, edit: impl FnOnce(&mut PlacementProfile)) {
    let mut profile = PlacementProfile::read(path).expect("the profile reads back");
    edit(&mut profile);
    let text = serde_json::to_string(&profile).expect("the profile serializes");
    std::fs::write(path, text).expect("the profile is rewritten");
}

async fn place(
    pipeline: &str,
    path: &Path,
) -> Result<g2g_core::Graph<g2g_core::runtime::GraphNode>, PlacementError> {
    let registry = default_registry();
    let mut graph = parse_launch(&registry, pipeline).expect("the line parses");
    let dry = parse_launch(&registry, pipeline).expect("the line parses");
    place_launch_graph(
        &registry,
        &mut graph,
        dry,
        path,
        PlacementObjective::Latency,
        RunnerThreads::One,
    )
    .await?;
    Ok(graph)
}

#[tokio::test]
async fn a_stage_placed_on_the_other_gpu_gets_a_copy_and_runs() {
    let Some(path) = profile_both_gpus("m2118-other-gpu").await else {
        return;
    };
    let convert = std::any::type_name::<HipConvertScale>();
    rewrite(&path, |profile| {
        for timing in &mut profile.elements {
            let on_nvidia = timing
                .key
                .device
                .as_ref()
                .is_some_and(|device| device.vendor == GpuVendor::Nvidia);
            if timing.key.element == convert && on_nvidia {
                timing.cost.gpu_ns = SLOWER_THAN_ANY_GPU_NS;
            }
        }
    });
    let graph = place(&line("nvidia"), &path)
        .await
        .expect("the plan succeeds");
    std::fs::remove_file(&path).expect("the profile is removed");

    let placed = (0..graph.node_count() as u32)
        .map(g2g_core::graph::NodeId)
        .filter_map(|node| graph.element(node))
        .find(|element| element.type_name() == convert)
        .expect("the graph keeps hipconvertscale");
    assert_eq!(
        placed.get_property("vendor"),
        Some(PropValue::Str(String::from("amd")))
    );
    assert_eq!(placed.get_property("device-id"), Some(PropValue::Int(0)));
    let clock = WallClock::new();
    RunBuilder::new(graph, &clock, LINK_CAPACITY)
        .run()
        .await
        .expect("frames uploaded on the NVIDIA GPU reach hipconvertscale on the AMD one");
}

#[tokio::test]
async fn a_profile_whose_gpu_has_a_different_name_now_fails_the_plan() {
    let Some(path) = profile_both_gpus("m2118-renamed").await else {
        return;
    };
    rewrite(&path, |profile| {
        for timing in &mut profile.elements {
            let Some(device) = timing.key.device.as_mut() else {
                continue;
            };
            if device.vendor == GpuVendor::Amd {
                device.name = String::from("a GPU from another machine");
            }
        }
    });
    let error = place(&line("nvidia"), &path).await.unwrap_err();
    std::fs::remove_file(&path).expect("the profile is removed");
    assert!(
        matches!(error, PlacementError::DeviceRenamed { .. }),
        "{error}"
    );
}

#[tokio::test]
async fn an_element_the_profile_never_timed_fails_the_plan() {
    let Some(path) = profile_both_gpus("m2118-untimed").await else {
        return;
    };
    let other_properties =
        line("nvidia").replace("hipconvertscale", "hipconvertscale add-borders=false");
    let error = place(&other_properties, &path).await.unwrap_err();
    std::fs::remove_file(&path).expect("the profile is removed");
    assert!(
        matches!(&error, PlacementError::NoTiming { device: None, .. }),
        "{error}"
    );
}

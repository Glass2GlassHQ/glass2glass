// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

use g2g_core::graph::{Graph, NodeId};
use g2g_core::memory::{DomainSet, GpuDevice, GpuVendor, MemoryDomain, MemoryDomainKind};
use g2g_core::runtime::{GraphNode, Observer, StageVisit, TelemetrySnapshot, VisitRecorder};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PipelinePacket,
};

use super::{
    device_property, execution_provider, key_properties, runs_on_a_gpu, CopyTiming, ElementCost,
    ElementKey, ElementTiming, PlacementError, PlacementProfile, ProfiledDevice,
};
use crate::hip::{median_copy_time, HipDevice};

// the first frames include model warm-up and memory pool growth
pub(super) const WARM_UP_FRAMES: usize = 10;

#[derive(Debug, Clone, Copy)]
struct VisitTimes {
    compute_ns: u64,
    poll_ns: u64,
}

#[derive(Debug, Default)]
struct ComputeTimes {
    by_element_name: Mutex<HashMap<String, Vec<VisitTimes>>>,
}

impl VisitRecorder for ComputeTimes {
    fn record(&self, element: &str, visit: StageVisit) {
        let compute_ns = visit
            .exit_ns
            .saturating_sub(visit.enter_ns)
            .saturating_sub(visit.push_wait_ns);
        let mut by_element_name = self
            .by_element_name
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        by_element_name
            .entry(element.to_string())
            .or_default()
            .push(VisitTimes {
                compute_ns,
                poll_ns: visit.poll_ns,
            });
    }
}

fn median(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

impl ComputeTimes {
    // `Err` holds how many frames the element processed
    fn measured(&self, element_name: &str) -> Result<Vec<VisitTimes>, usize> {
        let by_element_name = self
            .by_element_name
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let all = by_element_name
            .get(element_name)
            .map_or(&[][..], Vec::as_slice);
        match all.get(WARM_UP_FRAMES..) {
            Some(measured) if !measured.is_empty() => Ok(measured.to_vec()),
            _ => Err(all.len()),
        }
    }
}

type FrameDevice = Arc<Mutex<Option<GpuDevice>>>;

#[derive(Debug)]
enum DeviceSource {
    Cpu,
    Declared(GpuDevice),
    Frames(FrameDevice),
    Upstream(NodeId),
}

#[derive(Debug)]
struct ProfiledNode {
    node: NodeId,
    hip_sync: Option<NodeId>,
    element: &'static str,
    properties: BTreeMap<String, String>,
    execution_provider: Option<String>,
    device: DeviceSource,
}

#[derive(Debug)]
pub struct ProfileRun {
    nodes: Vec<ProfiledNode>,
    compute_times: Arc<ComputeTimes>,
    observer: Observer,
}

// the profiled element that feeds `node` its frames, skipping elements that pass frames through
fn gpu_upstream(graph: &Graph<GraphNode>, gpu_nodes: &[NodeId], node: NodeId) -> Option<NodeId> {
    let input = graph.edges().iter().find(|edge| edge.dst.node == node)?;
    let source = input.src.node;
    match gpu_nodes.contains(&source) {
        true => Some(source),
        false => gpu_upstream(graph, gpu_nodes, source),
    }
}

impl ProfileRun {
    pub fn prepare(graph: &mut Graph<GraphNode>) -> Self {
        let elements: Vec<NodeId> = (0..graph.node_count() as u32)
            .map(NodeId)
            .filter(|&node| graph.element(node).is_some())
            .collect();
        let gpu_nodes: Vec<NodeId> = elements
            .iter()
            .copied()
            .filter(|&node| {
                graph.element(node).is_some_and(|element| {
                    device_property(element).is_some() && runs_on_a_gpu(element)
                })
            })
            .collect();
        let mut nodes = Vec::new();
        for node in elements {
            let element = graph
                .element(node)
                .expect("the profiled nodes come from the graph's elements");
            let element_name = element.type_name();
            let properties = key_properties(element);
            let execution_provider = execution_provider(element);
            let declared = element.gpu_device();
            let leaves_gpu_work_queued = element.output_memory() == MemoryDomainKind::Hip;
            let output = graph.edges().iter().position(|edge| edge.src.node == node);
            let frame_device = FrameDevice::default();
            let hip_sync = output.filter(|_| leaves_gpu_work_queued).map(|edge| {
                let sync = HipSync {
                    device: None,
                    frame_device: Arc::clone(&frame_device),
                };
                graph.insert_on_edge(edge, GraphNode::element(sync))
            });
            let upstream = gpu_upstream(graph, &gpu_nodes, node);
            let device = match (gpu_nodes.contains(&node), declared, hip_sync, upstream) {
                (false, ..) => DeviceSource::Cpu,
                (true, Some(device), ..) => DeviceSource::Declared(device),
                (true, None, Some(_), _) => DeviceSource::Frames(frame_device),
                (true, None, None, Some(upstream)) => DeviceSource::Upstream(upstream),
                (true, None, None, None) => continue,
            };
            nodes.push(ProfiledNode {
                node,
                hip_sync,
                element: element_name,
                properties,
                execution_provider,
                device,
            });
        }
        let compute_times = Arc::new(ComputeTimes::default());
        let observer = Observer::new().with_visit_recorder(compute_times.clone());
        Self {
            nodes,
            compute_times,
            observer,
        }
    }

    pub fn observer(&self) -> &Observer {
        &self.observer
    }

    pub fn record(self, path: &Path) -> Result<PlacementProfile, PlacementError> {
        let snapshot = self.observer.snapshot();
        let mut profile = PlacementProfile::read_or_new(path)?;
        let mut measured = PlacementProfile::default();
        for profiled in &self.nodes {
            let device = self.device_of(profiled)?;
            measured
                .elements
                .push(profiled.timing(&snapshot, &self.compute_times, device)?);
        }
        profile.merge(measured);
        let devices = profile.devices();
        let mut copies = PlacementProfile::default();
        for from in &devices {
            for to in devices.iter().filter(|to| *to != from) {
                copies.copies.push(measure_copy(from, to)?);
            }
        }
        profile.merge(copies);
        profile.write(path)?;
        Ok(profile)
    }

    fn device_of(&self, profiled: &ProfiledNode) -> Result<Option<GpuDevice>, PlacementError> {
        let no_frames = || PlacementError::TooFewFrames {
            element: profiled.element.to_string(),
            frames: 0,
        };
        match &profiled.device {
            DeviceSource::Cpu => Ok(None),
            DeviceSource::Declared(device) => Ok(Some(*device)),
            DeviceSource::Frames(frame_device) => frame_device
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .map(Some)
                .ok_or_else(no_frames),
            DeviceSource::Upstream(node) => {
                let upstream = self
                    .nodes
                    .iter()
                    .find(|upstream| upstream.node == *node)
                    .ok_or_else(no_frames)?;
                self.device_of(upstream)
            }
        }
    }
}

impl ProfiledNode {
    fn timing(
        &self,
        snapshot: &TelemetrySnapshot,
        compute_times: &ComputeTimes,
        device: Option<GpuDevice>,
    ) -> Result<ElementTiming, PlacementError> {
        let element = || self.element.to_string();
        let measured = |node: NodeId| {
            let name = snapshot
                .nodes
                .get(node.0 as usize)
                .map_or("", |telemetry| telemetry.name.as_str());
            compute_times
                .measured(name)
                .map_err(|frames| PlacementError::TooFewFrames {
                    element: element(),
                    frames,
                })
        };
        let visits = measured(self.node)?;
        let host_ns = median(visits.iter().map(|visit| visit.poll_ns).collect());
        let wait_ns = median(
            visits
                .iter()
                .map(|visit| visit.compute_ns.saturating_sub(visit.poll_ns))
                .collect(),
        );
        let gpu_ns = match self.hip_sync {
            Some(sync) => median(
                measured(sync)?
                    .iter()
                    .map(|visit| visit.compute_ns)
                    .collect(),
            ),
            None => 0,
        };
        let input_caps: Vec<String> = snapshot
            .edges
            .iter()
            .filter(|edge| edge.to == self.node.0 as usize)
            .filter_map(|edge| edge.caps.clone())
            .collect();
        let device = match device {
            Some(device) => Some(ProfiledDevice {
                vendor: device.vendor,
                ordinal: device.ordinal,
                name: HipDevice::open(device.vendor, device.ordinal)?.name()?,
            }),
            None => None,
        };
        Ok(ElementTiming {
            key: ElementKey {
                element: element(),
                properties: self.properties.clone(),
                input_caps,
                device,
                execution_provider: self.execution_provider.clone(),
            },
            cost: ElementCost {
                host_ns,
                wait_ns,
                gpu_ns,
            },
        })
    }
}

pub(super) const COPY_SMALL_BYTES: usize = 4096;
// a 1080p RGBA frame is 8,294,400 bytes
pub(super) const COPY_LARGE_BYTES: usize = 16 << 20;

fn measure_copy(from: &ProfiledDevice, to: &ProfiledDevice) -> Result<CopyTiming, PlacementError> {
    let device = |profiled: &ProfiledDevice| GpuDevice {
        vendor: profiled.vendor,
        ordinal: profiled.ordinal,
    };
    let small = median_copy_time(device(from), device(to), COPY_SMALL_BYTES)?;
    let large = median_copy_time(device(from), device(to), COPY_LARGE_BYTES)?;
    let extra = large
        .checked_sub(small)
        .filter(|extra| !extra.is_zero())
        .ok_or_else(|| PlacementError::CopyNotSlower {
            from: from.clone(),
            to: to.clone(),
        })?;
    Ok(CopyTiming {
        from: from.clone(),
        to: to.clone(),
        overhead_ns: small.as_nanos() as u64,
        bytes_per_second: (COPY_LARGE_BYTES - COPY_SMALL_BYTES) as f64 / extra.as_secs_f64(),
    })
}

#[derive(Debug)]
struct HipSync {
    device: Option<Arc<HipDevice>>,
    frame_device: FrameDevice,
}

impl HipSync {
    fn device_for(&mut self, vendor: GpuVendor, ordinal: i32) -> Result<Arc<HipDevice>, G2gError> {
        let cached = self
            .device
            .as_ref()
            .filter(|device| (device.vendor(), device.ordinal()) == (vendor, ordinal));
        let device = match cached {
            Some(device) => Arc::clone(device),
            None => HipDevice::open(vendor, ordinal)?,
        };
        self.device = Some(Arc::clone(&device));
        *self
            .frame_device
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(GpuDevice { vendor, ordinal });
        Ok(device)
    }
}

impl AsyncElement for HipSync {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "HIP sync",
            "Generic",
            "Waits for the GPU work that wrote each frame to finish",
            crate::AARON_BOXER,
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::Hip)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::Hip
    }

    fn output_domain_follows_input(&self) -> bool {
        true
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::Hip(buffer) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let device = self.device_for(buffer.vendor, buffer.device_ordinal)?;
                    let buffer = buffer.clone();
                    g2g_core::offload::run_blocking(move || device.wait_until_ready(&buffer))
                        .await?;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::graph::{Graph, NodeId};
use g2g_core::memory::GpuDevice;
use g2g_core::runtime::{negotiate_graph, GraphNode, Registry};
use g2g_core::{Caps, Dim, PropValue};

use super::{
    device_property, execution_provider, has_automatic_device, has_property, key_properties,
    runs_on_a_gpu, ElementCost, ElementTiming, PlacementError, PlacementProfile, ProfiledDevice,
};
use crate::hip::HipDevice;
use crate::hipdeviceid::HipDeviceChoice;

pub(super) const MAX_PLACEMENTS: u128 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementObjective {
    Latency,
    Throughput,
}

impl PlacementObjective {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "latency" => Some(Self::Latency),
            "throughput" => Some(Self::Throughput),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunnerThreads {
    One,
    PerElement,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlacedElement {
    pub node: NodeId,
    pub element: &'static str,
    pub device: ProfiledDevice,
    pub execution_provider: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    pub elements: Vec<PlacedElement>,
    pub cost_ns: u64,
}

pub async fn place_launch_graph(
    registry: &Registry,
    graph: &mut Graph<GraphNode>,
    dry: Graph<GraphNode>,
    profile_path: &std::path::Path,
    objective: PlacementObjective,
    runner_threads: RunnerThreads,
) -> Result<Placement, PlacementError> {
    let profile = PlacementProfile::read(profile_path)?;
    // negotiate_graph consumes the graph it solves
    let (_, edge_caps, _) = negotiate_graph(dry)
        .await
        .map_err(PlacementError::Negotiation)?;
    let placement = plan_placement(graph, &edge_caps, &profile, objective, runner_threads)?;
    // parse_launch spliced converters for the devices the line named, before any were placed
    *graph = registry.splice_domain_converters(core::mem::take(graph));
    Ok(placement)
}

impl Placement {
    pub fn report(&self, graph: &Graph<GraphNode>, objective: PlacementObjective) -> String {
        let mut report = String::new();
        for placed in &self.elements {
            let label = graph
                .node_name(placed.node)
                .or_else(|| graph.element(placed.node).map(GraphNode::log_category))
                .unwrap_or(placed.element);
            let provider = placed
                .execution_provider
                .as_deref()
                .map_or(String::new(), |provider| format!(" ({provider})"));
            report.push_str(&format!(
                "placement: {label} on {}{provider}\n",
                placed.device.name
            ));
        }
        report.push_str(&format!(
            "placement: {objective:?} {:.2} ms per frame\n",
            self.cost_ns as f64 / 1e6
        ));
        report
    }
}

pub fn plan_placement(
    graph: &mut Graph<GraphNode>,
    edge_caps: &[Caps],
    profile: &PlacementProfile,
    objective: PlacementObjective,
    runner_threads: RunnerThreads,
) -> Result<Placement, PlacementError> {
    let model = model_of(graph, edge_caps, profile)?;
    let (choice, cost_ns) = model.cheapest(objective, runner_threads)?;
    let mut elements = Vec::new();
    for (&node, &pick) in model.placeable.iter().zip(&choice) {
        let NodeCost::Placeable {
            element,
            candidates,
        } = &model.nodes[node].cost
        else {
            continue;
        };
        let node = NodeId(node as u32);
        let device = model.devices[candidates[pick].device].clone();
        let placed = graph
            .element_mut(node)
            .expect("the cost model takes placeable nodes from the graph's elements");
        place(placed, element, &device)?;
        elements.push(PlacedElement {
            node,
            element,
            device,
            execution_provider: execution_provider(placed),
        });
    }
    Ok(Placement { elements, cost_ns })
}

fn place(
    element: &mut GraphNode,
    name: &'static str,
    device: &ProfiledDevice,
) -> Result<(), PlacementError> {
    let gpu = GpuDevice {
        vendor: device.vendor,
        ordinal: device.ordinal,
    };
    let mut settings: Vec<(&'static str, PropValue)> = Vec::new();
    if let Some(property) = device_property(element) {
        settings.push((property, PropValue::Int(i64::from(device.ordinal))));
    }
    if has_property(element, "vendor") {
        if let Some(vendor) = HipDeviceChoice::of(gpu).get_property("vendor") {
            settings.push(("vendor", vendor));
        }
    }
    for (property, value) in settings {
        element
            .set_property(property, value)
            .map_err(|error| PlacementError::Property {
                element: name.to_string(),
                property,
                error,
            })?;
    }
    Ok(())
}

fn model_of(
    graph: &Graph<GraphNode>,
    edge_caps: &[Caps],
    profile: &PlacementProfile,
) -> Result<CostModel, PlacementError> {
    if edge_caps.len() != graph.edges().len() {
        return Err(PlacementError::EdgeCaps {
            edges: graph.edges().len(),
            caps: edge_caps.len(),
        });
    }
    let mut devices = DeviceTable::default();
    let mut nodes = Vec::with_capacity(graph.node_count());
    for index in 0..graph.node_count() {
        let node = NodeId(index as u32);
        let input_caps: Vec<String> = graph
            .edges()
            .iter()
            .zip(edge_caps)
            .filter(|(edge, _)| edge.dst.node == node)
            .map(|(_, caps)| caps.to_gst_string())
            .collect();
        nodes.push(node_of(
            graph.element(node),
            input_caps,
            profile,
            &mut devices,
        )?);
    }
    let edges = graph
        .edges()
        .iter()
        .zip(edge_caps)
        .map(|(edge, caps)| ModelEdge {
            from: edge.src.node.0 as usize,
            to: edge.dst.node.0 as usize,
            frame_bytes: frame_bytes(caps),
            caps: caps.to_gst_string(),
        })
        .collect();
    let links = devices
        .profiled
        .iter()
        .map(|from| {
            devices
                .profiled
                .iter()
                .map(|to| {
                    profile
                        .copies
                        .iter()
                        .find(|copy| same_gpu(&copy.from, from) && same_gpu(&copy.to, to))
                        .map(|copy| LinkCost {
                            overhead_ns: copy.overhead_ns,
                            bytes_per_second: copy.bytes_per_second,
                        })
                })
                .collect()
        })
        .collect();
    devices.check_names()?;
    Ok(CostModel::new(nodes, edges, devices.profiled, links))
}

fn node_of(
    element: Option<&GraphNode>,
    input_caps: Vec<String>,
    profile: &PlacementProfile,
    devices: &mut DeviceTable,
) -> Result<ModelNode, PlacementError> {
    let Some(element) = element else {
        return Ok(ModelNode {
            cost: NodeCost::Free,
            output: FrameOutput::AsInput,
        });
    };
    let output = match element {
        GraphNode::Element(inner) if inner.output_domain_follows_input() => FrameOutput::AsInput,
        _ if element.output_memory().is_system() => FrameOutput::Cpu,
        _ => FrameOutput::Gpu,
    };
    let type_name = element.type_name();
    let properties = key_properties(element);
    let provider = execution_provider(element);
    let matching: Vec<&ElementTiming> = profile
        .elements
        .iter()
        .filter(|timing| {
            timing.key.element == type_name
                && timing.key.properties == properties
                && timing.key.input_caps == input_caps
                && timing.key.execution_provider == provider
        })
        .collect();
    let Some(property) = device_property(element).filter(|_| runs_on_a_gpu(element)) else {
        let timing = matching
            .into_iter()
            .find(|timing| timing.key.device.is_none())
            .ok_or_else(|| PlacementError::NoCpuTiming {
                element: type_name.to_string(),
                input_caps: input_caps.clone(),
            })?;
        return Ok(ModelNode {
            cost: NodeCost::Host(timing.cost),
            output,
        });
    };
    let no_timing = |device: Option<ProfiledDevice>| PlacementError::NoTiming {
        element: type_name.to_string(),
        input_caps: input_caps.clone(),
        device,
    };
    let on_a_gpu = matching
        .into_iter()
        .filter_map(|timing| Some((timing.key.device.as_ref()?, timing.cost)));
    if has_automatic_device(element, property) {
        let mut candidates: Vec<Candidate> = Vec::new();
        for (device, cost) in on_a_gpu {
            let candidate = Candidate {
                device: devices.index_of(device),
                cost,
            };
            match candidates
                .iter_mut()
                .find(|known| known.device == candidate.device)
            {
                Some(known) if known.cost.total_ns() <= cost.total_ns() => {}
                Some(known) => *known = candidate,
                None => candidates.push(candidate),
            }
        }
        if candidates.is_empty() {
            return Err(no_timing(None));
        }
        return Ok(ModelNode {
            cost: NodeCost::Placeable {
                element: type_name,
                candidates,
            },
            output,
        });
    }
    let Some(fixed) = element.gpu_device() else {
        return Err(PlacementError::DeviceUnknown {
            element: type_name.to_string(),
        });
    };
    let timing = on_a_gpu
        .filter(|(device, _)| device.vendor == fixed.vendor && device.ordinal == fixed.ordinal)
        .min_by_key(|(_, cost)| cost.total_ns());
    let Some((device, cost)) = timing else {
        let name = HipDevice::open(fixed.vendor, fixed.ordinal)?.name()?;
        return Err(no_timing(Some(ProfiledDevice {
            vendor: fixed.vendor,
            ordinal: fixed.ordinal,
            name,
        })));
    };
    Ok(ModelNode {
        cost: NodeCost::Fixed {
            device: devices.index_of(device),
            cost,
        },
        output,
    })
}

fn same_gpu(a: &ProfiledDevice, b: &ProfiledDevice) -> bool {
    (a.vendor, a.ordinal) == (b.vendor, b.ordinal)
}

fn frame_bytes(caps: &Caps) -> Option<u64> {
    match caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        } => format.unpadded_frame_bytes(*width, *height),
        _ => None,
    }
}

#[derive(Debug, Default)]
struct DeviceTable {
    profiled: Vec<ProfiledDevice>,
}

impl DeviceTable {
    fn index_of(&mut self, device: &ProfiledDevice) -> usize {
        match self
            .profiled
            .iter()
            .position(|known| same_gpu(known, device))
        {
            Some(index) => index,
            None => {
                self.profiled.push(device.clone());
                self.profiled.len() - 1
            }
        }
    }

    fn check_names(&self) -> Result<(), PlacementError> {
        for profiled in &self.profiled {
            let found = HipDevice::open(profiled.vendor, profiled.ordinal)?.name()?;
            if found != profiled.name {
                return Err(PlacementError::DeviceRenamed {
                    profiled: profiled.clone(),
                    found,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameOutput {
    Cpu,
    Gpu,
    AsInput,
}

#[derive(Debug, Clone, PartialEq)]
struct Candidate {
    device: usize,
    cost: ElementCost,
}

#[derive(Debug, Clone, PartialEq)]
enum NodeCost {
    Free,
    Host(ElementCost),
    Fixed {
        device: usize,
        cost: ElementCost,
    },
    Placeable {
        element: &'static str,
        candidates: Vec<Candidate>,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct ModelNode {
    cost: NodeCost,
    output: FrameOutput,
}

#[derive(Debug, Clone, PartialEq)]
struct ModelEdge {
    from: usize,
    to: usize,
    frame_bytes: Option<u64>,
    caps: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LinkCost {
    overhead_ns: u64,
    bytes_per_second: f64,
}

#[derive(Debug)]
struct CostModel {
    nodes: Vec<ModelNode>,
    edges: Vec<ModelEdge>,
    devices: Vec<ProfiledDevice>,
    // indexed by source device, then target device
    links: Vec<Vec<Option<LinkCost>>>,
    topological_order: Vec<usize>,
    placeable: Vec<usize>,
}

impl CostModel {
    fn new(
        nodes: Vec<ModelNode>,
        edges: Vec<ModelEdge>,
        devices: Vec<ProfiledDevice>,
        links: Vec<Vec<Option<LinkCost>>>,
    ) -> Self {
        let mut inputs_left = vec![0usize; nodes.len()];
        for edge in &edges {
            inputs_left[edge.to] += 1;
        }
        let mut ready: Vec<usize> = (0..nodes.len())
            .filter(|&node| inputs_left[node] == 0)
            .collect();
        let mut topological_order = Vec::with_capacity(nodes.len());
        while let Some(node) = ready.pop() {
            topological_order.push(node);
            for edge in edges.iter().filter(|edge| edge.from == node) {
                inputs_left[edge.to] -= 1;
                if inputs_left[edge.to] == 0 {
                    ready.push(edge.to);
                }
            }
        }
        let placeable = (0..nodes.len())
            .filter(|&node| matches!(nodes[node].cost, NodeCost::Placeable { .. }))
            .collect();
        Self {
            nodes,
            edges,
            devices,
            links,
            topological_order,
            placeable,
        }
    }

    fn candidate_counts(&self) -> Vec<usize> {
        self.placeable
            .iter()
            .map(|&node| match &self.nodes[node].cost {
                NodeCost::Placeable { candidates, .. } => candidates.len(),
                _ => 1,
            })
            .collect()
    }

    fn cheapest(
        &self,
        objective: PlacementObjective,
        runner_threads: RunnerThreads,
    ) -> Result<(Vec<usize>, u64), PlacementError> {
        let counts = self.candidate_counts();
        let count = counts
            .iter()
            .fold(1u128, |total, &n| total.saturating_mul(n as u128));
        if count > MAX_PLACEMENTS {
            return Err(PlacementError::TooManyPlacements { count });
        }
        let mut choice = vec![0usize; counts.len()];
        let mut best = (
            choice.clone(),
            self.cost(&choice, objective, runner_threads)?,
        );
        loop {
            let next = choice
                .iter()
                .zip(&counts)
                .position(|(&pick, &count)| pick + 1 < count);
            let Some(slot) = next else {
                break;
            };
            choice[slot] += 1;
            choice[..slot].fill(0);
            let cost = self.cost(&choice, objective, runner_threads)?;
            if cost < best.1 {
                best = (choice.clone(), cost);
            }
        }
        Ok(best)
    }

    fn cost(
        &self,
        choice: &[usize],
        objective: PlacementObjective,
        runner_threads: RunnerThreads,
    ) -> Result<u64, PlacementError> {
        let node_count = self.nodes.len();
        let mut device: Vec<Option<usize>> = vec![None; node_count];
        let mut cost: Vec<ElementCost> = vec![ElementCost::default(); node_count];
        for (index, node) in self.nodes.iter().enumerate() {
            match node.cost {
                NodeCost::Host(host) => cost[index] = host,
                NodeCost::Fixed {
                    device: fixed,
                    cost: fixed_cost,
                } => {
                    device[index] = Some(fixed);
                    cost[index] = fixed_cost;
                }
                NodeCost::Free | NodeCost::Placeable { .. } => {}
            }
        }
        for (&node, &pick) in self.placeable.iter().zip(choice) {
            if let NodeCost::Placeable { candidates, .. } = &self.nodes[node].cost {
                device[node] = Some(candidates[pick].device);
                cost[node] = candidates[pick].cost;
            }
        }
        let mut frames_on: Vec<Option<usize>> = vec![None; node_count];
        // GPU work still queued for the frames a node emits, which a copy waits for
        let mut queued_gpu_ns = vec![0u64; node_count];
        let mut finish_ns = vec![0u64; node_count];
        let mut device_load_ns = vec![0u64; self.devices.len()];
        let mut link_load_ns = vec![vec![0u64; self.devices.len()]; self.devices.len()];
        let mut runner_load_ns = 0u64;
        let mut busiest_arm_ns = 0u64;
        for &node in &self.topological_order {
            let mut ready_ns = 0u64;
            let mut input_on: Option<Option<usize>> = None;
            let mut input_queued_gpu_ns = 0u64;
            for edge in self.edges.iter().filter(|edge| edge.to == node) {
                let arriving_on = frames_on[edge.from];
                input_on.get_or_insert(arriving_on);
                input_queued_gpu_ns = input_queued_gpu_ns.max(queued_gpu_ns[edge.from]);
                let copy_ns = match (arriving_on, device[node]) {
                    (Some(from), Some(to)) if from != to && self.owns_device(node) => {
                        let ns = self.copy_ns(from, to, edge)?;
                        link_load_ns[from][to] += ns;
                        busiest_arm_ns = busiest_arm_ns.max(queued_gpu_ns[edge.from] + ns);
                        ns
                    }
                    _ => 0,
                };
                ready_ns = ready_ns.max(finish_ns[edge.from] + copy_ns);
            }
            let input_on = input_on.flatten();
            if !self.owns_device(node) {
                device[node] = input_on;
            }
            (frames_on[node], queued_gpu_ns[node]) = match self.nodes[node].output {
                FrameOutput::Cpu => (None, 0),
                FrameOutput::Gpu => (device[node].or(input_on), cost[node].gpu_ns),
                FrameOutput::AsInput => (input_on, input_queued_gpu_ns),
            };
            let element = cost[node];
            finish_ns[node] = ready_ns + element.total_ns();
            runner_load_ns += element.host_ns;
            busiest_arm_ns = busiest_arm_ns.max(element.host_ns + element.wait_ns);
            if let (Some(on), true) = (device[node], self.owns_device(node)) {
                device_load_ns[on] += element.wait_ns + element.gpu_ns;
            }
        }
        let shared_runner_ns = match runner_threads {
            RunnerThreads::One => runner_load_ns,
            RunnerThreads::PerElement => 0,
        };
        Ok(match objective {
            PlacementObjective::Latency => finish_ns.into_iter().max().unwrap_or(0),
            PlacementObjective::Throughput => device_load_ns
                .into_iter()
                .chain(link_load_ns.into_iter().flatten())
                .chain([busiest_arm_ns, shared_runner_ns])
                .max()
                .unwrap_or(0),
        })
    }

    fn owns_device(&self, node: usize) -> bool {
        matches!(
            self.nodes[node].cost,
            NodeCost::Fixed { .. } | NodeCost::Placeable { .. }
        )
    }

    fn copy_ns(&self, from: usize, to: usize, edge: &ModelEdge) -> Result<u64, PlacementError> {
        let link = self.links[from][to].ok_or_else(|| PlacementError::NoCopyTiming {
            from: self.devices[from].clone(),
            to: self.devices[to].clone(),
        })?;
        let bytes = edge
            .frame_bytes
            .ok_or_else(|| PlacementError::UnsizedCopy {
                caps: edge.caps.clone(),
            })?;
        let transfer_ns = bytes as f64 / link.bytes_per_second * 1e9;
        Ok(link.overhead_ns + transfer_ns as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::memory::GpuVendor;

    const MS: u64 = 1_000_000;
    const FRAME_BYTES: u64 = 8_294_400;
    const FAST_LINK: f64 = 1e12;

    fn devices() -> Vec<ProfiledDevice> {
        [GpuVendor::Amd, GpuVendor::Nvidia]
            .into_iter()
            .map(|vendor| ProfiledDevice {
                vendor,
                ordinal: 0,
                name: format!("{vendor:?} GPU"),
            })
            .collect()
    }

    fn links(overhead_ns: u64) -> Vec<Vec<Option<LinkCost>>> {
        let link = Some(LinkCost {
            overhead_ns,
            bytes_per_second: FAST_LINK,
        });
        vec![vec![None, link], vec![link, None]]
    }

    fn gpu(gpu_ns: u64) -> ElementCost {
        ElementCost {
            gpu_ns,
            ..ElementCost::default()
        }
    }

    fn host(host_ns: u64) -> ElementCost {
        ElementCost {
            host_ns,
            ..ElementCost::default()
        }
    }

    fn source() -> ModelNode {
        ModelNode {
            cost: NodeCost::Free,
            output: FrameOutput::Cpu,
        }
    }

    fn fixed(device: usize, cost: ElementCost, output: FrameOutput) -> ModelNode {
        ModelNode {
            cost: NodeCost::Fixed { device, cost },
            output,
        }
    }

    fn placeable(costs: [ElementCost; 2], output: FrameOutput) -> ModelNode {
        ModelNode {
            cost: NodeCost::Placeable {
                element: "test element",
                candidates: costs
                    .into_iter()
                    .enumerate()
                    .map(|(device, cost)| Candidate { device, cost })
                    .collect(),
            },
            output,
        }
    }

    fn edge(from: usize, to: usize) -> ModelEdge {
        ModelEdge {
            from,
            to,
            frame_bytes: Some(FRAME_BYTES),
            caps: String::from("video/x-raw,format=RGBA,width=1920,height=1080"),
        }
    }

    fn transfer_ns() -> u64 {
        (FRAME_BYTES as f64 / FAST_LINK * 1e9) as u64
    }

    // a convert that is faster on the second GPU feeding a stage fixed on the first
    fn convert_then_fixed_stage(link_overhead_ns: u64) -> CostModel {
        CostModel::new(
            vec![
                source(),
                placeable([gpu(10 * MS), gpu(4 * MS)], FrameOutput::Gpu),
                fixed(0, gpu(5 * MS), FrameOutput::Cpu),
            ],
            vec![edge(0, 1), edge(1, 2)],
            devices(),
            links(link_overhead_ns),
        )
    }

    #[test]
    fn latency_keeps_a_stage_beside_its_consumer_when_the_copy_costs_more_than_it_saves() {
        let model = convert_then_fixed_stage(20 * MS);
        let placed = model
            .cheapest(PlacementObjective::Latency, RunnerThreads::One)
            .unwrap();
        assert_eq!(placed, (vec![0], 15 * MS));
    }

    #[test]
    fn latency_moves_a_stage_to_the_faster_gpu_when_the_copy_is_cheap() {
        let model = convert_then_fixed_stage(MS);
        let placed = model
            .cheapest(PlacementObjective::Latency, RunnerThreads::One)
            .unwrap();
        assert_eq!(placed, (vec![1], 10 * MS + transfer_ns()));
    }

    fn two_parallel_branches() -> CostModel {
        let tee = ModelNode {
            cost: NodeCost::Free,
            output: FrameOutput::AsInput,
        };
        CostModel::new(
            vec![
                source(),
                tee,
                placeable([gpu(10 * MS), gpu(12 * MS)], FrameOutput::Cpu),
                placeable([gpu(10 * MS), gpu(12 * MS)], FrameOutput::Cpu),
            ],
            vec![edge(0, 1), edge(1, 2), edge(1, 3)],
            devices(),
            links(MS),
        )
    }

    #[test]
    fn latency_puts_both_branches_on_the_faster_gpu() {
        let placed = two_parallel_branches()
            .cheapest(PlacementObjective::Latency, RunnerThreads::One)
            .unwrap();
        assert_eq!(placed, (vec![0, 0], 10 * MS));
    }

    #[test]
    fn throughput_spreads_branches_so_no_gpu_carries_both() {
        let (choice, cost_ns) = two_parallel_branches()
            .cheapest(PlacementObjective::Throughput, RunnerThreads::One)
            .unwrap();
        assert_eq!(cost_ns, 12 * MS);
        assert_ne!(choice[0], choice[1]);
    }

    #[test]
    fn throughput_charges_a_copy_the_wait_for_its_producers_gpu_work() {
        let copy_overhead_ns = 2 * MS;
        let producer_gpu_ns = 3 * MS;
        let model = CostModel::new(
            vec![
                source(),
                fixed(0, gpu(producer_gpu_ns), FrameOutput::Gpu),
                placeable([gpu(MS), gpu(MS / 2)], FrameOutput::Cpu),
            ],
            vec![edge(0, 1), edge(1, 2)],
            devices(),
            links(copy_overhead_ns),
        );
        let placed = model
            .cheapest(PlacementObjective::Throughput, RunnerThreads::One)
            .unwrap();
        assert_eq!(placed, (vec![0], producer_gpu_ns + MS));
        let moved = model
            .cost(&[1], PlacementObjective::Throughput, RunnerThreads::One)
            .unwrap();
        assert_eq!(moved, producer_gpu_ns + copy_overhead_ns + transfer_ns());
    }

    #[test]
    fn one_runner_thread_carries_the_host_time_of_every_element() {
        let source_host_ns = 5 * MS;
        let upload_host_ns = 2 * MS;
        let upload = ElementCost {
            host_ns: upload_host_ns,
            ..gpu(MS)
        };
        let model = CostModel::new(
            vec![
                ModelNode {
                    cost: NodeCost::Host(host(source_host_ns)),
                    output: FrameOutput::Cpu,
                },
                fixed(0, upload, FrameOutput::Gpu),
                placeable([gpu(3 * MS), gpu(3 * MS)], FrameOutput::Cpu),
            ],
            vec![edge(0, 1), edge(1, 2)],
            devices(),
            links(MS),
        );
        let throughput = |runner_threads| {
            model
                .cost(&[0], PlacementObjective::Throughput, runner_threads)
                .unwrap()
        };
        assert_eq!(
            throughput(RunnerThreads::One),
            source_host_ns + upload_host_ns
        );
        assert_eq!(throughput(RunnerThreads::PerElement), source_host_ns);
    }

    struct MigraphxInference;

    static INFERENCE_PROPS: &[g2g_core::PropertySpec] = &[
        g2g_core::PropertySpec::new("execution-provider", g2g_core::PropKind::Str, "provider"),
        g2g_core::PropertySpec::new("device-id", g2g_core::PropKind::Int, "device"),
    ];

    impl g2g_core::AsyncElement for MigraphxInference {
        type ProcessFuture<'a> = core::future::Ready<Result<(), g2g_core::G2gError>>;

        fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, g2g_core::G2gError> {
            Ok(upstream.clone())
        }

        fn configure_pipeline(
            &mut self,
            _caps: &Caps,
        ) -> Result<g2g_core::ConfigureOutcome, g2g_core::G2gError> {
            Ok(g2g_core::ConfigureOutcome::Accepted)
        }

        fn input_domains(&self) -> g2g_core::memory::DomainSet {
            g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::Hip)
        }

        fn properties(&self) -> &'static [g2g_core::PropertySpec] {
            INFERENCE_PROPS
        }

        fn get_property(&self, name: &str) -> Option<PropValue> {
            match name {
                "execution-provider" => Some(PropValue::Str(String::from("migraphx"))),
                "device-id" => Some(PropValue::Int(-1)),
                _ => None,
            }
        }

        fn process<'a>(
            &'a mut self,
            _packet: g2g_core::PipelinePacket,
            _out: &'a mut dyn g2g_core::OutputSink,
        ) -> Self::ProcessFuture<'a> {
            core::future::ready(Ok(()))
        }
    }

    #[test]
    fn a_placeable_element_keeps_the_execution_provider_its_line_set() {
        let element = GraphNode::element(MigraphxInference);
        let input_caps = vec![String::from("video/x-raw,format=RGBP")];
        let timing = |device: usize, provider: &str, gpu_ns: u64| ElementTiming {
            key: super::super::ElementKey {
                element: String::from(element.type_name()),
                properties: Default::default(),
                input_caps: input_caps.clone(),
                device: Some(devices()[device].clone()),
                execution_provider: Some(String::from(provider)),
            },
            cost: gpu(gpu_ns),
        };
        let profile = PlacementProfile {
            elements: vec![timing(0, "migraphx", 50 * MS), timing(1, "cuda", 5 * MS)],
            copies: Vec::new(),
        };
        let mut table = DeviceTable::default();
        let node = node_of(Some(&element), input_caps.clone(), &profile, &mut table).unwrap();
        let NodeCost::Placeable { candidates, .. } = node.cost else {
            panic!("device-id -1 leaves the device to the planner: {node:?}");
        };
        let vendors: Vec<GpuVendor> = candidates
            .iter()
            .map(|candidate| table.profiled[candidate.device].vendor)
            .collect();
        assert_eq!(vendors, [GpuVendor::Amd]);
    }

    #[test]
    fn a_search_past_the_limit_fails_instead_of_running() {
        let stages = 21;
        let mut nodes = vec![source()];
        nodes.extend((0..stages).map(|_| placeable([gpu(MS), gpu(MS)], FrameOutput::Cpu)));
        let edges = (0..stages).map(|stage| edge(stage, stage + 1)).collect();
        let model = CostModel::new(nodes, edges, devices(), links(MS));
        let error = model
            .cheapest(PlacementObjective::Latency, RunnerThreads::One)
            .unwrap_err();
        assert!(matches!(
            error,
            PlacementError::TooManyPlacements { count } if count == 1 << stages
        ));
    }

    #[test]
    fn a_copy_of_unsized_frames_fails_instead_of_costing_nothing() {
        let mut model = convert_then_fixed_stage(MS);
        model.edges[1].frame_bytes = None;
        let error = model
            .cost(&[1], PlacementObjective::Latency, RunnerThreads::One)
            .unwrap_err();
        assert!(matches!(error, PlacementError::UnsizedCopy { .. }));
    }
}

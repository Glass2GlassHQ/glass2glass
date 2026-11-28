// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::copyplan::{classify, TransferKind};
use g2g_core::graph::{Graph, NodeId};
use g2g_core::property::PropValue;
use g2g_core::runtime::{GraphNode, Observer};
use g2g_core::{Caps, MemoryDomainKind};
use pyo3::prelude::*;

#[derive(Debug)]
pub(super) struct Report {
    nodes: Vec<Node>,
    pads: Vec<(u8, u8)>,
    last: Vec<String>,
}

#[derive(Debug, Default)]
struct Node {
    category: &'static str,
    capabilities: Option<Capabilities>,
}

#[derive(Debug)]
struct Capabilities {
    accepts: String,
    reads_pixels: bool,
    synchronization: String,
}

impl Capabilities {
    fn of(element: &GraphNode) -> Option<Self> {
        let (Some(PropValue::Str(module)), Some(PropValue::Str(class))) = (
            element.get_property("module"),
            element.get_property("class"),
        ) else {
            return None;
        };
        Python::attach(|py| -> PyResult<Self> {
            let class = py.import(module.as_str())?.getattr(class.as_str())?;
            let caps = class.getattr("FRAME_CAPABILITIES")?;
            let memory: Vec<String> = caps.getattr("memory_types")?.extract()?;
            let formats: Vec<String> = caps.getattr("pixel_formats")?.extract()?;
            Ok(Self {
                accepts: format!("{}/{}", memory.join(","), formats.join(",")),
                reads_pixels: caps.getattr("reads_pixels")?.extract()?,
                synchronization: caps.getattr("synchronization")?.extract()?,
            })
        })
        .ok()
    }
}

impl Report {
    pub(super) fn new(graph: &Graph<GraphNode>) -> Self {
        let nodes = (0..graph.node_count())
            .map(|i| match graph.element(NodeId(i as u32)) {
                Some(element) => Node {
                    category: element.log_category(),
                    capabilities: Capabilities::of(element),
                },
                None => Node::default(),
            })
            .collect();
        let pads = graph
            .edges()
            .iter()
            .map(|edge| (edge.src.index, edge.dst.index))
            .collect();
        Self {
            nodes,
            pads,
            last: Vec::new(),
        }
    }

    pub(super) fn print(&mut self, observer: &Observer) {
        let snapshot = observer.snapshot();
        let kinds: Vec<_> = snapshot
            .edges
            .iter()
            .enumerate()
            .map(|(i, _)| observer.edge_caps(i))
            .collect();
        let domains: Vec<_> = snapshot
            .edges
            .iter()
            .enumerate()
            .map(|(i, _)| observer.edge_memory(i))
            .collect();
        self.last.resize(snapshot.edges.len(), String::new());
        for (i, edge) in snapshot.edges.iter().enumerate() {
            let Some(caps @ (Caps::RawVideo { .. } | Caps::CompressedVideo { .. })) = &kinds[i]
            else {
                continue;
            };
            let spec = edge
                .observed_caps
                .as_deref()
                .or(edge.caps.as_deref())
                .unwrap_or_default();
            let format = spec
                .split(',')
                .find_map(|field| field.strip_prefix("format="))
                .unwrap_or_else(|| spec.split(',').next().unwrap_or_default());
            let from_node = &snapshot.nodes[edge.from];
            let to_node = &snapshot.nodes[edge.to];
            let (src_pad, dst_pad) = self.pads.get(i).copied().unwrap_or_default();
            let memory = domains[i]
                .map(|d| format!("{d:?}"))
                .unwrap_or_else(|| "unknown".into());
            let mut transfers = Vec::new();
            if matches!(caps, Caps::RawVideo { .. }) {
                if let Some(to) = domains[i] {
                    for (j, input) in snapshot.edges.iter().enumerate() {
                        if input.to != edge.from || !matches!(kinds[j], Some(Caps::RawVideo { .. }))
                        {
                            continue;
                        }
                        if let Some(from) = domains[j] {
                            let label = match classify(from, to) {
                                TransferKind::None => continue,
                                TransferKind::DeviceHost if to.is_system() => "download",
                                TransferKind::DeviceHost => "upload",
                                _ => "device interop",
                            };
                            let transfer = format!("{label} {from:?} -> {to:?}");
                            if !transfers.contains(&transfer) {
                                transfers.push(transfer);
                            }
                        }
                    }
                }
            }
            let transfer = if transfers.is_empty() {
                "none".to_owned()
            } else {
                transfers.join(", ")
            };
            let unknown = Node::default();
            let target = self.nodes.get(edge.to).unwrap_or(&unknown);
            let sync = synchronization(target, domains[i]);
            let contract = target
                .capabilities
                .as_ref()
                .map(|cap| format!(" | consumer={}", cap.accepts))
                .unwrap_or_default();
            let from = if from_node.name.is_empty() {
                format!("node{}", edge.from)
            } else {
                from_node.name.clone()
            };
            let to = if to_node.name.is_empty() {
                format!("node{}", edge.to)
            } else {
                to_node.name.clone()
            };
            let line = format!(
                "device-path {from}:{src_pad} -> {to}:{dst_pad} | {memory} {format} | transfer={transfer} | sync={sync}{contract} | {spec}"
            );
            if self.last[i] != line {
                eprintln!("{line}");
                self.last[i] = line;
            }
        }
    }
}

fn synchronization(node: &Node, domain: Option<MemoryDomainKind>) -> String {
    if domain == Some(MemoryDomainKind::Cuda) {
        if let Some(caps) = &node.capabilities {
            if caps.reads_pixels {
                return format!("CUDA stream synchronize at {}", caps.synchronization);
            }
            return "none declared".into();
        }
        if node.category.ends_with("CudaToWgpu") {
            return "CUDA producer readiness wait and interop copy completion".into();
        }
        if node.category.ends_with("CudaDownload") {
            return "CUDA copy completion".into();
        }
    }
    if matches!(
        domain,
        Some(MemoryDomainKind::WgpuTexture | MemoryDomainKind::WgpuBuffer)
    ) && node.category.ends_with("WgpuDownload")
    {
        return "wgpu readback completion".into();
    }
    match domain {
        Some(d) if d.is_system() => "none declared",
        _ => "backend-managed",
    }
    .into()
}

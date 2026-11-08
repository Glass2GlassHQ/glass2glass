use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use g2g_core::graph::{Graph, NodeId};
use g2g_core::memory::{DomainSet, GpuDevice, GpuVendor, MemoryDomain, MemoryDomainKind};
use g2g_core::runtime::{
    traced_gpu_device, GraphNode, Observer, StageVisit, TelemetrySnapshot, VisitRecorder,
};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PipelinePacket,
};
use serde::{Deserialize, Serialize};

use crate::hip::{median_copy_time, HipDevice};

#[derive(Serialize, Deserialize)]
#[serde(remote = "GpuVendor", rename_all = "lowercase")]
enum GpuVendorJson {
    Amd,
    Nvidia,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfiledDevice {
    #[serde(with = "GpuVendorJson")]
    pub vendor: GpuVendor,
    pub ordinal: i32,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementKey {
    pub element: String,
    pub properties: BTreeMap<String, String>,
    pub input_caps: Vec<String>,
    pub device: ProfiledDevice,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementTiming {
    #[serde(flatten)]
    pub key: ElementKey,
    pub median_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CopyTiming {
    pub from: ProfiledDevice,
    pub to: ProfiledDevice,
    pub overhead_ns: u64,
    pub bytes_per_second: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PlacementProfile {
    pub elements: Vec<ElementTiming>,
    pub copies: Vec<CopyTiming>,
}

#[derive(Debug)]
pub enum ProfileError {
    Io(std::io::Error),
    Format(serde_json::Error),
    Gpu(G2gError),
    TooFewFrames {
        element: String,
        frames: usize,
    },
    NoInput {
        element: String,
    },
    CopyNotSlower {
        from: ProfiledDevice,
        to: ProfiledDevice,
    },
}

impl core::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ProfileError::Io(error) => write!(f, "{error}"),
            ProfileError::Format(error) => write!(f, "not a placement profile: {error}"),
            ProfileError::Gpu(error) => write!(f, "GPU query failed: {error:?}"),
            ProfileError::TooFewFrames { element, frames } => write!(
                f,
                "{element} processed {frames} frames, and a timing needs more than the \
                 first {WARM_UP_FRAMES}"
            ),
            ProfileError::NoInput { element } => {
                write!(f, "{element} has no input link with caps")
            }
            ProfileError::CopyNotSlower { from, to } => write!(
                f,
                "copying {COPY_LARGE_BYTES} bytes from {} to {} was no slower than \
                 {COPY_SMALL_BYTES} bytes, so its bandwidth cannot be measured",
                from.name, to.name
            ),
        }
    }
}

impl std::error::Error for ProfileError {}

impl From<G2gError> for ProfileError {
    fn from(error: G2gError) -> Self {
        ProfileError::Gpu(error)
    }
}

impl PlacementProfile {
    pub fn read(path: &Path) -> Result<Self, ProfileError> {
        let text = std::fs::read_to_string(path).map_err(ProfileError::Io)?;
        serde_json::from_str(&text).map_err(ProfileError::Format)
    }

    fn read_or_new(path: &Path) -> Result<Self, ProfileError> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).map_err(ProfileError::Format),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(ProfileError::Io(error)),
        }
    }

    fn write(&self, path: &Path) -> Result<(), ProfileError> {
        let text = serde_json::to_string_pretty(self).map_err(ProfileError::Format)?;
        std::fs::write(path, text).map_err(ProfileError::Io)
    }

    pub fn merge(&mut self, newer: PlacementProfile) {
        for timing in newer.elements {
            self.elements.retain(|old| old.key != timing.key);
            self.elements.push(timing);
        }
        for copy in newer.copies {
            self.copies
                .retain(|old| (&old.from, &old.to) != (&copy.from, &copy.to));
            self.copies.push(copy);
        }
    }

    fn devices(&self) -> Vec<ProfiledDevice> {
        let mut devices: Vec<ProfiledDevice> = Vec::new();
        for timing in &self.elements {
            if !devices.contains(&timing.key.device) {
                devices.push(timing.key.device.clone());
            }
        }
        devices
    }
}

const DEVICE_PROPERTIES: [&str; 2] = ["device-id", "cuda-device-id"];
const BACKEND_PROPERTIES: [&str; 2] = ["vendor", "execution-provider"];

#[derive(Debug)]
struct ProfiledNode {
    node: NodeId,
    hip_sync: Option<NodeId>,
    element: &'static str,
    properties: BTreeMap<String, String>,
    device: GpuDevice,
}

// the first frames include model warm-up and memory pool growth
const WARM_UP_FRAMES: usize = 10;

#[derive(Debug, Default)]
struct ComputeTimes {
    by_element_name: Mutex<HashMap<String, Vec<u64>>>,
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
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        by_element_name
            .entry(element.to_string())
            .or_default()
            .push(compute_ns);
    }
}

impl ComputeTimes {
    // `Err` holds how many frames the element processed
    fn median_ns(&self, element_name: &str) -> Result<u64, usize> {
        let by_element_name = self
            .by_element_name
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let all = by_element_name
            .get(element_name)
            .map_or(&[][..], Vec::as_slice);
        let mut measured = all.get(WARM_UP_FRAMES..).unwrap_or(&[]).to_vec();
        if measured.is_empty() {
            return Err(all.len());
        }
        measured.sort_unstable();
        Ok(measured[measured.len() / 2])
    }
}

#[derive(Debug)]
pub struct ProfileRun {
    nodes: Vec<ProfiledNode>,
    compute_times: Arc<ComputeTimes>,
    observer: Observer,
}

impl ProfileRun {
    pub fn prepare(graph: &mut Graph<GraphNode>) -> Self {
        let mut nodes = Vec::new();
        for index in 0..graph.node_count() as u32 {
            let node = NodeId(index);
            let Some(device) = traced_gpu_device(graph, node) else {
                continue;
            };
            let Some(element) = graph.element(node) else {
                continue;
            };
            let Some(type_name) = element.type_name() else {
                continue;
            };
            let has_device_property = element
                .properties()
                .iter()
                .any(|spec| DEVICE_PROPERTIES.contains(&spec.name));
            if !has_device_property {
                continue;
            }
            let properties = element
                .properties()
                .iter()
                .filter(|spec| spec.flags.writable)
                .filter(|spec| {
                    !DEVICE_PROPERTIES.contains(&spec.name)
                        && !BACKEND_PROPERTIES.contains(&spec.name)
                })
                .filter_map(|spec| {
                    let value = element.get_property(spec.name)?;
                    Some((spec.name.to_string(), value.to_string()))
                })
                .collect();
            let leaves_gpu_work_queued = element.output_memory() == MemoryDomainKind::Hip;
            let output = graph.edges().iter().position(|edge| edge.src.node == node);
            let hip_sync = output
                .filter(|_| leaves_gpu_work_queued)
                .map(|edge| graph.insert_on_edge(edge, GraphNode::element(HipSync::default())));
            nodes.push(ProfiledNode {
                node,
                hip_sync,
                element: type_name,
                properties,
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

    pub fn record(self, path: &Path) -> Result<PlacementProfile, ProfileError> {
        let snapshot = self.observer.snapshot();
        let mut profile = PlacementProfile::read_or_new(path)?;
        let mut measured = PlacementProfile::default();
        for profiled in &self.nodes {
            measured
                .elements
                .push(profiled.timing(&snapshot, &self.compute_times)?);
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
}

impl ProfiledNode {
    fn timing(
        &self,
        snapshot: &TelemetrySnapshot,
        compute_times: &ComputeTimes,
    ) -> Result<ElementTiming, ProfileError> {
        let element = || self.element.to_string();
        let median_of = |node: NodeId| {
            let name = snapshot
                .nodes
                .get(node.0 as usize)
                .map_or("", |telemetry| telemetry.name.as_str());
            compute_times
                .median_ns(name)
                .map_err(|frames| ProfileError::TooFewFrames {
                    element: element(),
                    frames,
                })
        };
        let wait_ns = match self.hip_sync {
            Some(sync) => median_of(sync)?,
            None => 0,
        };
        let median_ns = median_of(self.node)? + wait_ns;
        let input_caps: Vec<String> = snapshot
            .edges
            .iter()
            .filter(|edge| edge.to == self.node.0 as usize)
            .filter_map(|edge| edge.observed_caps.clone().or_else(|| edge.caps.clone()))
            .collect();
        if input_caps.is_empty() {
            return Err(ProfileError::NoInput { element: element() });
        }
        let name = HipDevice::open(self.device.vendor, self.device.ordinal)?.name()?;
        Ok(ElementTiming {
            key: ElementKey {
                element: element(),
                properties: self.properties.clone(),
                input_caps,
                device: ProfiledDevice {
                    vendor: self.device.vendor,
                    ordinal: self.device.ordinal,
                    name,
                },
            },
            median_ns,
        })
    }
}

const COPY_SMALL_BYTES: usize = 4096;
// a 1080p RGBA frame is 8,294,400 bytes
const COPY_LARGE_BYTES: usize = 16 << 20;

fn measure_copy(from: &ProfiledDevice, to: &ProfiledDevice) -> Result<CopyTiming, ProfileError> {
    let device = |profiled: &ProfiledDevice| GpuDevice {
        vendor: profiled.vendor,
        ordinal: profiled.ordinal,
    };
    let small = median_copy_time(device(from), device(to), COPY_SMALL_BYTES)?;
    let large = median_copy_time(device(from), device(to), COPY_LARGE_BYTES)?;
    let extra = large
        .checked_sub(small)
        .filter(|extra| !extra.is_zero())
        .ok_or_else(|| ProfileError::CopyNotSlower {
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

#[derive(Debug, Default)]
struct HipSync {
    device: Option<Arc<HipDevice>>,
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
            "g2g",
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

#[cfg(test)]
mod tests {
    use super::*;

    fn device(ordinal: i32) -> ProfiledDevice {
        ProfiledDevice {
            vendor: GpuVendor::Amd,
            ordinal,
            name: String::from("AMD Radeon 680M"),
        }
    }

    fn timing(model: &str, ordinal: i32, median_ns: u64) -> ElementTiming {
        ElementTiming {
            key: ElementKey {
                element: String::from("g2g_ml::inference::Inference<g2g_ml::ortinfer::OrtBackend>"),
                properties: BTreeMap::from([(String::from("model"), String::from(model))]),
                input_caps: alloc::vec![String::from("video/x-raw,format=RGBP")],
                device: device(ordinal),
            },
            median_ns,
        }
    }

    #[test]
    fn a_later_run_replaces_only_the_timings_it_measured_again() {
        let mut profile = PlacementProfile {
            elements: alloc::vec![timing("yolo11m.onnx", 0, 67), timing("yolo11n.onnx", 0, 9)],
            copies: Vec::new(),
        };
        profile.merge(PlacementProfile {
            elements: alloc::vec![timing("yolo11m.onnx", 0, 70), timing("yolo11m.onnx", 1, 20)],
            copies: Vec::new(),
        });
        let mut medians: Vec<(String, i32, u64)> = profile
            .elements
            .iter()
            .map(|timing| {
                (
                    timing.key.properties["model"].clone(),
                    timing.key.device.ordinal,
                    timing.median_ns,
                )
            })
            .collect();
        medians.sort();
        assert_eq!(
            medians,
            alloc::vec![
                (String::from("yolo11m.onnx"), 0, 70),
                (String::from("yolo11m.onnx"), 1, 20),
                (String::from("yolo11n.onnx"), 0, 9),
            ]
        );
    }

    #[test]
    fn a_profile_reads_back_what_was_written() {
        let profile = PlacementProfile {
            elements: alloc::vec![timing("yolo11m.onnx", 0, 67)],
            copies: alloc::vec![CopyTiming {
                from: device(0),
                to: device(1),
                overhead_ns: 40_000,
                bytes_per_second: 6.0e9,
            }],
        };
        let text = serde_json::to_string(&profile).unwrap();
        assert!(text.contains("\"vendor\":\"amd\""), "{text}");
        assert_eq!(
            serde_json::from_str::<PlacementProfile>(&text).unwrap(),
            profile
        );
    }
}

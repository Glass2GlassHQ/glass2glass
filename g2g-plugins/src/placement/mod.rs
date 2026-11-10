// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

mod plan;
mod profile;

pub use plan::{
    place_launch_graph, plan_placement, PlacedElement, Placement, PlacementObjective, RunnerThreads,
};
pub use profile::ProfileRun;

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::path::Path;

use g2g_core::memory::{GpuVendor, MemoryDomainKind};
use g2g_core::runtime::GraphNode;
use g2g_core::{G2gError, PropError, PropValue};
use serde::{Deserialize, Serialize};

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
    pub device: Option<ProfiledDevice>,
    pub execution_provider: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementCost {
    pub host_ns: u64,
    pub wait_ns: u64,
    pub gpu_ns: u64,
}

impl ElementCost {
    pub fn total_ns(&self) -> u64 {
        self.host_ns + self.wait_ns + self.gpu_ns
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementTiming {
    #[serde(flatten)]
    pub key: ElementKey,
    #[serde(flatten)]
    pub cost: ElementCost,
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
pub enum PlacementError {
    Io(std::io::Error),
    Format(serde_json::Error),
    Gpu(G2gError),
    Negotiation(G2gError),
    TooFewFrames {
        element: String,
        frames: usize,
    },
    CopyNotSlower {
        from: ProfiledDevice,
        to: ProfiledDevice,
    },
    NoTiming {
        element: String,
        input_caps: Vec<String>,
        device: Option<ProfiledDevice>,
    },
    NoCpuTiming {
        element: String,
        input_caps: Vec<String>,
    },
    DeviceRenamed {
        profiled: ProfiledDevice,
        found: String,
    },
    DeviceUnknown {
        element: String,
    },
    NoCopyTiming {
        from: ProfiledDevice,
        to: ProfiledDevice,
    },
    UnsizedCopy {
        caps: String,
    },
    TooManyPlacements {
        count: u128,
    },
    EdgeCaps {
        edges: usize,
        caps: usize,
    },
    Property {
        element: String,
        property: &'static str,
        error: PropError,
    },
}

impl core::fmt::Display for PlacementError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PlacementError::Io(error) => write!(f, "{error}"),
            PlacementError::Format(error) => write!(f, "not a placement profile: {error}"),
            PlacementError::Gpu(error) => write!(f, "GPU query failed: {error:?}"),
            PlacementError::Negotiation(error) => {
                write!(f, "caps negotiation failed: {error:?}")
            }
            PlacementError::TooFewFrames { element, frames } => write!(
                f,
                "{element} processed {frames} frames, and a timing needs more than the \
                 first {}",
                profile::WARM_UP_FRAMES
            ),
            PlacementError::CopyNotSlower { from, to } => write!(
                f,
                "copying {} bytes from {} to {} was no slower than {} bytes, so its \
                 bandwidth cannot be measured",
                profile::COPY_LARGE_BYTES,
                from.name,
                to.name,
                profile::COPY_SMALL_BYTES
            ),
            PlacementError::NoTiming {
                element,
                input_caps,
                device: Some(device),
            } => write!(
                f,
                "the profile has no timing for {element} on {} with these properties and \
                 input caps {input_caps:?}, run --profile with it there",
                device.name
            ),
            PlacementError::NoTiming {
                element,
                input_caps,
                device: None,
            } => write!(
                f,
                "the profile has no timing for {element} with these properties and input \
                 caps {input_caps:?} on any GPU, run --profile with it on each GPU"
            ),
            PlacementError::NoCpuTiming {
                element,
                input_caps,
            } => write!(
                f,
                "the profile has no timing for {element} with these properties and input \
                 caps {input_caps:?}, run --profile with this line"
            ),
            PlacementError::DeviceRenamed { profiled, found } => write!(
                f,
                "the profile measured {:?} device {} as {}, but it is now {found}",
                profiled.vendor, profiled.ordinal, profiled.name
            ),
            PlacementError::DeviceUnknown { element } => write!(
                f,
                "{element} sets device-id without vendor, so its GPU is unknown before it runs"
            ),
            PlacementError::NoCopyTiming { from, to } => write!(
                f,
                "the profile has no copy timing from {} to {}",
                from.name, to.name
            ),
            PlacementError::UnsizedCopy { caps } => {
                write!(f, "a copy of {caps} has no fixed frame size to cost")
            }
            PlacementError::TooManyPlacements { count } => write!(
                f,
                "{count} placements to try, more than the {} limit",
                plan::MAX_PLACEMENTS
            ),
            PlacementError::EdgeCaps { edges, caps } => {
                write!(f, "the graph has {edges} links but {caps} negotiated caps")
            }
            PlacementError::Property {
                element,
                property,
                error,
            } => write!(f, "{element} refused {property}: {error:?}"),
        }
    }
}

impl std::error::Error for PlacementError {}

impl From<G2gError> for PlacementError {
    fn from(error: G2gError) -> Self {
        PlacementError::Gpu(error)
    }
}

impl PlacementProfile {
    pub fn read(path: &Path) -> Result<Self, PlacementError> {
        let text = std::fs::read_to_string(path).map_err(PlacementError::Io)?;
        serde_json::from_str(&text).map_err(PlacementError::Format)
    }

    fn read_or_new(path: &Path) -> Result<Self, PlacementError> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).map_err(PlacementError::Format),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(PlacementError::Io(error)),
        }
    }

    fn write(&self, path: &Path) -> Result<(), PlacementError> {
        let text = serde_json::to_string_pretty(self).map_err(PlacementError::Format)?;
        std::fs::write(path, text).map_err(PlacementError::Io)
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
        for device in self
            .elements
            .iter()
            .filter_map(|timing| timing.key.device.as_ref())
        {
            if !devices.contains(device) {
                devices.push(device.clone());
            }
        }
        devices
    }
}

const DEVICE_PROPERTIES: [&str; 2] = ["device-id", "cuda-device-id"];
const BACKEND_PROPERTIES: [&str; 2] = ["vendor", "execution-provider"];
const AUTOMATIC_DEVICE_ID: i64 = -1;
const AUTOMATIC_VENDOR: &str = "unknown";

fn has_property(element: &GraphNode, name: &str) -> bool {
    element.properties().iter().any(|spec| spec.name == name)
}

fn device_property(element: &GraphNode) -> Option<&'static str> {
    DEVICE_PROPERTIES
        .into_iter()
        .find(|name| has_property(element, name))
}

fn runs_on_a_gpu(element: &GraphNode) -> bool {
    element.gpu_device().is_some()
        || !element.output_memory().is_system()
        || !element.input_domains().contains(MemoryDomainKind::System)
}

fn has_automatic_device(element: &GraphNode, device_property: &str) -> bool {
    let vendor_automatic = !has_property(element, "vendor")
        || element.get_property("vendor") == Some(PropValue::Str(AUTOMATIC_VENDOR.into()));
    element.get_property(device_property) == Some(PropValue::Int(AUTOMATIC_DEVICE_ID))
        && vendor_automatic
}

fn key_properties(element: &GraphNode) -> BTreeMap<String, String> {
    element
        .properties()
        .iter()
        .filter(|spec| spec.flags.writable)
        .filter(|spec| {
            !DEVICE_PROPERTIES.contains(&spec.name) && !BACKEND_PROPERTIES.contains(&spec.name)
        })
        .filter_map(|spec| {
            let value = element.get_property(spec.name)?;
            Some((spec.name.to_string(), value.to_string()))
        })
        .collect()
}

fn execution_provider(element: &GraphNode) -> Option<String> {
    match element.get_property("execution-provider")? {
        PropValue::Str(provider) => Some(provider),
        _ => None,
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

    fn timing(model: &str, ordinal: i32, gpu_ns: u64) -> ElementTiming {
        ElementTiming {
            key: ElementKey {
                element: String::from("g2g_ml::inference::Inference<g2g_ml::ortinfer::OrtBackend>"),
                properties: BTreeMap::from([(String::from("model"), String::from(model))]),
                input_caps: alloc::vec![String::from("video/x-raw,format=RGBP")],
                device: Some(device(ordinal)),
                execution_provider: Some(String::from("migraphx")),
            },
            cost: ElementCost {
                gpu_ns,
                ..ElementCost::default()
            },
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
                    timing.key.device.as_ref().unwrap().ordinal,
                    timing.cost.gpu_ns,
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

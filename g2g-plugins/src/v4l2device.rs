//! V4L2 device discovery (M939): enumerates `/dev/videoN` capture nodes as
//! [`Device`] records so a [`DeviceMonitor`](g2g_core::runtime::DeviceMonitor)
//! lists cameras with the modes their driver actually reports.
//!
//! Only the YUYV fourcc becomes caps, because YUYV is all `v4l2src` can
//! deliver. The other fourccs the node advertises are kept in `detail` so the
//! information is not lost when a decode-through-MJPEG path arrives.
//!
//! V4L2 has no hotplug event source here (udev would be a separate backend),
//! so the provider offers no native watch and the monitor polls and diffs.
//!
//! The mapping half (frame sizes / intervals / capabilities to caps and ids)
//! is pure, so it is unit-tested without a camera attached.
//!
//! [`resolve_device_id`] is the reverse direction, for `v4l2src device-id=`:
//! an id this provider minted earlier back to the node path carrying it now.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::runtime::{Device, DeviceProvider};
use g2g_core::{Caps, CapsSet, Dim, G2gError, Interlace, Rate, RawVideoFormat};

use v4l::capability::Capabilities;
use v4l::fraction::Fraction;
use v4l::frameinterval::FrameIntervalEnum;
use v4l::framesize::FrameSizeEnum;
use v4l::video::Capture;
use v4l::FourCC;

/// The one fourcc `v4l2src` negotiates.
const YUYV: &[u8; 4] = b"YUYV";

/// Upper bound on the caps alternatives one device carries. A driver can
/// report hundreds of size/rate combinations (UVC cameras with many modes,
/// stepwise scalers), and every one of them would be cloned into each hotplug
/// event; 32 covers the useful modes of real hardware.
const MAX_ALTERNATIVES: usize = 32;

/// Discovers V4L2 capture devices for the `v4l2src` element.
#[derive(Debug, Default)]
pub struct V4l2DeviceProvider {
    _private: (),
}

impl V4l2DeviceProvider {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DeviceProvider for V4l2DeviceProvider {
    fn name(&self) -> &'static str {
        "v4l2"
    }

    fn probe(&self) -> Result<Vec<Device>, G2gError> {
        let mut nodes: Vec<String> = v4l::context::enum_devices()
            .iter()
            .map(|n| n.path().to_string_lossy().into_owned())
            .collect();
        // readdir order is arbitrary; a stable list keeps the monitor's diff
        // and any UI listing from reshuffling between probes.
        nodes.sort();
        let mut devices = Vec::new();
        for path in nodes {
            // A node we cannot open or query (permissions, a driver that only
            // does metadata / output) is skipped, never fatal to the probe.
            if let Some(device) = probe_node(&path) {
                devices.push(device);
            }
        }
        Ok(devices)
    }
}

/// Open one node and describe it, or `None` when it is not a usable capture
/// device.
fn probe_node(path: &str) -> Option<Device> {
    let dev = v4l::Device::with_path(path).ok()?;
    let caps = dev.query_caps().ok()?;
    if !caps
        .capabilities
        .contains(v4l::capability::Flags::VIDEO_CAPTURE)
    {
        return None;
    }

    let formats = dev.enum_formats().unwrap_or_default();
    let fourccs: Vec<String> = formats
        .iter()
        .map(|f| f.fourcc.str().unwrap_or("????").to_string())
        .collect();

    let yuyv = FourCC::new(YUYV);
    let mut modes = Vec::new();
    if formats.iter().any(|f| f.fourcc.repr == *YUYV) {
        for size in dev.enum_framesizes(yuyv).unwrap_or_default() {
            let (probe_w, probe_h) = match &size.size {
                FrameSizeEnum::Discrete(d) => (d.width, d.height),
                FrameSizeEnum::Stepwise(s) => (s.min_width, s.min_height),
            };
            let intervals = dev
                .enum_frameintervals(yuyv, probe_w, probe_h)
                .unwrap_or_default()
                .into_iter()
                .map(|i| i.interval)
                .collect();
            modes.push((size.size, intervals));
        }
    }

    let display_name = if caps.card.is_empty() {
        path.to_string()
    } else {
        caps.card.clone()
    };

    Some(Device {
        display_name,
        klass: "Video/Source".to_string(),
        persistent_id: persistent_id(Some(&caps), path),
        caps: yuyv_caps(&modes),
        element: "v4l2src",
        props: Vec::from([("device".to_string(), path.to_string())]),
        detail: Vec::from([
            ("driver".to_string(), caps.driver.clone()),
            ("bus".to_string(), caps.bus.clone()),
            ("formats".to_string(), fourccs.join(" ")),
        ]),
        provider: "v4l2",
    })
}

/// `bus_info:card:path` where the driver reports a bus (stable across reboots
/// for USB / PCI, unlike the node number), else the node path alone. The path
/// stays in the id even then: the RGB and IR nodes of one USB camera share
/// bus_info AND card, and the monitor's diff key must be unique.
fn persistent_id(caps: Option<&Capabilities>, path: &str) -> String {
    match caps {
        Some(c) if !c.bus.is_empty() => {
            let mut id = c.bus.clone();
            id.push(':');
            id.push_str(&c.card);
            id.push(':');
            id.push_str(path);
            id
        }
        _ => path.to_string(),
    }
}

/// The node path a saved `device-id` names now, or `None` when no attached
/// camera carries it. Probes the same list [`V4l2DeviceProvider`] reports, so
/// an id copied out of `g2g-device-monitor` resolves here.
pub fn resolve_device_id(id: &str) -> Option<String> {
    let devices = V4l2DeviceProvider::new().probe().ok()?;
    match_device_id(&devices, id).map(String::from)
}

/// The node path of the device `id` names: the exact id first, then the
/// hardware half of it (bus + card, dropping the node path the id ends with).
/// The fallback is what survives a replug: the same USB port reports the same
/// bus_info, but the kernel may hand the camera a different `/dev/videoN`.
/// Devices arrive sorted by path, so the lowest-numbered node of a multi-node
/// camera wins, which is the capture node on every UVC device.
fn match_device_id<'a>(devices: &'a [Device], id: &str) -> Option<&'a str> {
    if let Some(device) = devices.iter().find(|d| d.persistent_id == id) {
        return node_path(device);
    }
    let wanted = hardware_prefix(id)?;
    devices
        .iter()
        .find(|d| hardware_prefix(&d.persistent_id) == Some(wanted))
        .and_then(node_path)
}

/// The `bus_info:card` half of a [`persistent_id`], or `None` for the bare-path
/// form (which has no hardware identity to match on).
fn hardware_prefix(id: &str) -> Option<&str> {
    let (prefix, tail) = id.rsplit_once(':')?;
    tail.starts_with("/dev/").then_some(prefix)
}

/// The node path a probed device opens, from the `device` property the provider
/// put there.
fn node_path(device: &Device) -> Option<&str> {
    device
        .props
        .iter()
        .find(|(key, _)| key == "device")
        .map(|(_, value)| value.as_str())
}

/// Frame sizes plus the intervals reported for each, as YUYV caps
/// alternatives in driver order.
fn yuyv_caps(modes: &[(FrameSizeEnum, Vec<FrameIntervalEnum>)]) -> CapsSet {
    let mut alternatives = Vec::new();
    for (size, intervals) in modes {
        let Some((width, height)) = size_dims(size) else {
            continue;
        };
        let mut rates = interval_rates(intervals);
        if rates.is_empty() {
            // The driver enumerated the size but no interval for it; the rate
            // is unknown rather than absent.
            rates.push(Rate::Any);
        }
        for framerate in rates {
            if alternatives.len() >= MAX_ALTERNATIVES {
                return CapsSet::from_alternatives(alternatives);
            }
            alternatives.push(Caps::RawVideo {
                format: RawVideoFormat::Yuyv,
                width: width.clone(),
                height: height.clone(),
                framerate,
                interlace: Interlace::Any,
            });
        }
    }
    CapsSet::from_alternatives(alternatives)
}

/// One enumerated frame size as caps dimensions. `None` for a degenerate
/// entry (zero or inverted bounds) rather than caps nothing can satisfy. The
/// stepwise step size has no `Dim` equivalent, so the range is a superset.
fn size_dims(size: &FrameSizeEnum) -> Option<(Dim, Dim)> {
    match size {
        FrameSizeEnum::Discrete(d) => {
            (d.width > 0 && d.height > 0).then_some((Dim::Fixed(d.width), Dim::Fixed(d.height)))
        }
        FrameSizeEnum::Stepwise(s) => {
            if s.min_width == 0
                || s.min_height == 0
                || s.min_width > s.max_width
                || s.min_height > s.max_height
            {
                return None;
            }
            Some((
                Dim::Range {
                    min: s.min_width,
                    max: s.max_width,
                },
                Dim::Range {
                    min: s.min_height,
                    max: s.max_height,
                },
            ))
        }
    }
}

/// Enumerated frame intervals as caps rates, dropping unusable entries.
fn interval_rates(intervals: &[FrameIntervalEnum]) -> Vec<Rate> {
    let mut rates = Vec::new();
    for interval in intervals {
        match interval {
            FrameIntervalEnum::Discrete(f) => {
                if let Some(q16) = rate_q16(f) {
                    rates.push(Rate::Fixed(q16));
                }
            }
            FrameIntervalEnum::Stepwise(s) => {
                // bounds swap: the shortest interval is the highest rate.
                if let (Some(max_q16), Some(min_q16)) = (rate_q16(&s.min), rate_q16(&s.max)) {
                    if min_q16 <= max_q16 {
                        rates.push(Rate::Range { min_q16, max_q16 });
                    }
                }
            }
        }
    }
    rates
}

/// A frame interval (seconds per frame) as Q16 fps, the repo's rate encoding.
/// `None` for a zero term or a rate too large to encode.
fn rate_q16(interval: &Fraction) -> Option<u32> {
    if interval.numerator == 0 || interval.denominator == 0 {
        return None;
    }
    let q16 = ((interval.denominator as u64) << 16) / interval.numerator as u64;
    u32::try_from(q16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use v4l::frameinterval::Stepwise as IntervalStepwise;
    use v4l::framesize::{Discrete, Stepwise as SizeStepwise};

    fn discrete(width: u32, height: u32) -> FrameSizeEnum {
        FrameSizeEnum::Discrete(Discrete { width, height })
    }

    fn fps(n: u32) -> FrameIntervalEnum {
        FrameIntervalEnum::Discrete(Fraction::new(1, n))
    }

    fn caps_of(caps: &Caps) -> (Dim, Dim, Rate) {
        match caps {
            Caps::RawVideo {
                format,
                width,
                height,
                framerate,
                ..
            } => {
                assert_eq!(*format, RawVideoFormat::Yuyv);
                (width.clone(), height.clone(), framerate.clone())
            }
            other => panic!("expected raw video caps, got {other:?}"),
        }
    }

    #[test]
    fn discrete_sizes_and_rates_become_fixed_alternatives() {
        let modes = Vec::from([
            (discrete(640, 480), Vec::from([fps(30), fps(15)])),
            (discrete(1280, 720), Vec::from([fps(10)])),
        ]);
        let set = yuyv_caps(&modes);
        let alts = set.alternatives();
        assert_eq!(alts.len(), 3);
        assert_eq!(
            caps_of(&alts[0]),
            (Dim::Fixed(640), Dim::Fixed(480), Rate::Fixed(30 << 16))
        );
        assert_eq!(
            caps_of(&alts[1]),
            (Dim::Fixed(640), Dim::Fixed(480), Rate::Fixed(15 << 16))
        );
        assert_eq!(
            caps_of(&alts[2]),
            (Dim::Fixed(1280), Dim::Fixed(720), Rate::Fixed(10 << 16))
        );
    }

    #[test]
    fn stepwise_size_and_interval_become_ranges() {
        let size = FrameSizeEnum::Stepwise(SizeStepwise {
            min_width: 160,
            max_width: 1920,
            step_width: 16,
            min_height: 120,
            max_height: 1080,
            step_height: 8,
        });
        let interval = FrameIntervalEnum::Stepwise(IntervalStepwise {
            min: Fraction::new(1, 60),
            max: Fraction::new(1, 5),
            step: Fraction::new(1, 1000),
        });
        let modes = Vec::from([(size, Vec::from([interval]))]);
        let alts = yuyv_caps(&modes).alternatives().to_vec();
        assert_eq!(alts.len(), 1);
        assert_eq!(
            caps_of(&alts[0]),
            (
                Dim::Range {
                    min: 160,
                    max: 1920
                },
                Dim::Range {
                    min: 120,
                    max: 1080
                },
                Rate::Range {
                    min_q16: 5 << 16,
                    max_q16: 60 << 16
                }
            )
        );
    }

    #[test]
    fn degenerate_sizes_and_intervals_are_dropped() {
        let modes = Vec::from([
            (discrete(0, 480), Vec::from([fps(30)])),
            (
                FrameSizeEnum::Stepwise(SizeStepwise {
                    min_width: 1920,
                    max_width: 640,
                    step_width: 1,
                    min_height: 120,
                    max_height: 1080,
                    step_height: 1,
                }),
                Vec::from([fps(30)]),
            ),
            (
                discrete(320, 240),
                Vec::from([FrameIntervalEnum::Discrete(Fraction::new(0, 30))]),
            ),
        ]);
        let alts = yuyv_caps(&modes).alternatives().to_vec();
        // only the last mode survives, with an unknown rate.
        assert_eq!(alts.len(), 1);
        assert_eq!(
            caps_of(&alts[0]),
            (Dim::Fixed(320), Dim::Fixed(240), Rate::Any)
        );
    }

    #[test]
    fn alternatives_are_capped() {
        let modes: Vec<_> = (0..100)
            .map(|i| (discrete(640 + i, 480), Vec::from([fps(30), fps(15)])))
            .collect();
        assert_eq!(yuyv_caps(&modes).alternatives().len(), MAX_ALTERNATIVES);
    }

    #[test]
    fn persistent_id_prefers_bus_info_and_falls_back_to_path() {
        let caps = Capabilities {
            driver: "uvcvideo".to_string(),
            card: "Integrated Camera".to_string(),
            bus: "usb-0000:00:14.0-8".to_string(),
            version: (6, 1, 0),
            capabilities: v4l::capability::Flags::VIDEO_CAPTURE,
        };
        assert_eq!(
            persistent_id(Some(&caps), "/dev/video0"),
            "usb-0000:00:14.0-8:Integrated Camera:/dev/video0"
        );
        // two nodes of one physical camera (RGB + IR) share bus and card; the
        // path keeps their ids distinct
        assert_ne!(
            persistent_id(Some(&caps), "/dev/video0"),
            persistent_id(Some(&caps), "/dev/video2")
        );
        let no_bus = Capabilities {
            bus: String::new(),
            ..caps
        };
        assert_eq!(persistent_id(Some(&no_bus), "/dev/video0"), "/dev/video0");
        assert_eq!(persistent_id(None, "/dev/video2"), "/dev/video2");
    }

    fn probed(bus: &str, card: &str, path: &str) -> Device {
        Device {
            display_name: card.to_string(),
            klass: "Video/Source".to_string(),
            persistent_id: format!("{bus}:{card}:{path}"),
            caps: CapsSet::from_alternatives(Vec::new()),
            element: "v4l2src",
            props: Vec::from([("device".to_string(), path.to_string())]),
            detail: Vec::new(),
            provider: "v4l2",
        }
    }

    #[test]
    fn device_id_resolves_exactly_and_across_a_replug() {
        let devices = Vec::from([
            probed("usb-0000:00:14.0-8", "Integrated Camera", "/dev/video0"),
            probed("usb-0000:00:14.0-8", "Integrated Camera", "/dev/video2"),
            probed("usb-0000:00:14.0-3", "HD Webcam", "/dev/video4"),
        ]);
        // the exact id wins even when a sibling node shares its hardware.
        assert_eq!(
            match_device_id(&devices, "usb-0000:00:14.0-8:Integrated Camera:/dev/video2"),
            Some("/dev/video2")
        );
        // the saved node moved: the same port + card still resolves, to the
        // lowest-numbered node of that camera.
        assert_eq!(
            match_device_id(&devices, "usb-0000:00:14.0-8:Integrated Camera:/dev/video6"),
            Some("/dev/video0")
        );
        // a camera that is not attached resolves to nothing rather than to
        // whatever else happens to be plugged in.
        assert_eq!(
            match_device_id(&devices, "usb-0000:00:14.0-1:Other Camera:/dev/video0"),
            None
        );
        // the bare-path id form has no hardware half, so it matches only itself.
        let bare = Vec::from([Device {
            persistent_id: "/dev/video0".to_string(),
            ..probed("", "", "/dev/video0")
        }]);
        assert_eq!(match_device_id(&bare, "/dev/video0"), Some("/dev/video0"));
        assert_eq!(match_device_id(&bare, "/dev/video1"), None);
    }

    #[test]
    fn probe_describes_any_camera_present_and_tolerates_none() {
        let devices = V4l2DeviceProvider::new().probe().expect("probe");
        for device in devices {
            assert_eq!(device.element, "v4l2src");
            assert_eq!(device.provider, "v4l2");
            assert!(device.has_classes("Video/Source"));
            assert_eq!(device.props.len(), 1);
            assert_eq!(device.props[0].0, "device");
            assert!(device.props[0].1.starts_with("/dev/video"));
            assert!(!device.persistent_id.is_empty());
            assert!(device.launch_fragment().starts_with("v4l2src device="));
        }
    }
}

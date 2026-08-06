//! ALSA device discovery (M939): lists the libasound PCM hints as
//! [`Device`] records, capture hints driving [`AlsaSrc`](crate::alsasrc) and
//! playback hints [`AlsaSink`](crate::alsasink). One hint can serve both
//! directions, and then yields one device per direction.
//!
//! libasound has no hotplug event source of its own, so the provider offers no
//! native watch and a [`DeviceMonitor`](g2g_core::runtime::DeviceMonitor)
//! polls it.
//!
//! Caps are probed by opening each PCM and reading its `hw_params` space: one
//! alternative per supported sample format, at 48 kHz stereo clamped into the
//! device's own ranges. The raw ranges stay in `detail`. A PCM that will not
//! open (busy, permissions) is still listed, with empty caps and the errno in
//! `detail`.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use alsa::device_name::HintIter;
use alsa::pcm::{HwParams, PCM};
use alsa::Direction;

use g2g_core::runtime::{Device, DeviceProvider};
use g2g_core::{AudioFormat, Caps, CapsSet, G2gError, HardwareError};

use crate::alsapcm::FORMATS;

/// [`Device::provider`] for everything this backend finds.
const PROVIDER: &str = "alsa";

/// What the probed caps ask for before clamping into the device's ranges.
const PREFERRED_RATE: u32 = 48_000;
const PREFERRED_CHANNELS: u32 = 2;

/// Lists the libasound PCM hints (`default`, `hw:*`, `plughw:*`, the plugin
/// devices) as capture / playback [`Device`]s.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlsaDeviceProvider;

impl AlsaDeviceProvider {
    /// A provider over the whole hint list (every card).
    pub fn new() -> Self {
        Self
    }
}

impl DeviceProvider for AlsaDeviceProvider {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    fn probe(&self) -> Result<Vec<Device>, G2gError> {
        let hints = HintIter::new_str(None, "pcm").map_err(|e| alsa_err(e.errno()))?;
        let mut out = Vec::new();
        for hint in hints {
            let Some(name) = hint.name else { continue };
            for (dir, mut device) in hint_devices(&name, hint.desc.as_deref(), hint.direction) {
                match probe_pcm(&name, dir) {
                    Ok(probe) => {
                        device.caps = probe.caps();
                        device.detail = probe.detail();
                    }
                    Err(errno) => {
                        device.detail = vec![(
                            String::from("probe-error"),
                            format!("{} open failed, errno {errno}", dir_word(dir)),
                        )]
                    }
                }
                out.push(device);
            }
        }
        Ok(out)
    }
}

fn alsa_err(code: i32) -> G2gError {
    G2gError::Hardware(HardwareError::Alsa(code))
}

fn dir_word(dir: Direction) -> &'static str {
    match dir {
        Direction::Capture => "capture",
        Direction::Playback => "playback",
    }
}

/// The device halves this build lists for a hint: a hint with no direction
/// serves both, and each half counts only when the element that drives it is
/// compiled in.
fn listed_directions(hint: Option<Direction>) -> Vec<Direction> {
    #[allow(unused_mut)]
    let mut out = Vec::new();
    #[cfg(feature = "alsa-src")]
    if hint != Some(Direction::Playback) {
        out.push(Direction::Capture);
    }
    #[cfg(feature = "alsa-sink")]
    if hint != Some(Direction::Capture) {
        out.push(Direction::Playback);
    }
    out
}

/// The (unprobed) devices one hint contributes. `null` is the discard PCM, not
/// a device anyone wants to select.
fn hint_devices(
    name: &str,
    desc: Option<&str>,
    direction: Option<Direction>,
) -> Vec<(Direction, Device)> {
    if name == "null" {
        return Vec::new();
    }
    listed_directions(direction)
        .into_iter()
        .map(|dir| (dir, device_skeleton(name, desc, dir)))
        .collect()
}

/// One device with everything the hint alone decides; caps and detail come
/// from the PCM probe.
fn device_skeleton(name: &str, desc: Option<&str>, dir: Direction) -> Device {
    let (element, klass) = match dir {
        Direction::Capture => ("alsasrc", "Audio/Source"),
        Direction::Playback => ("alsasink", "Audio/Sink"),
    };
    // the first desc line names the card; the rest is boilerplate like
    // "Default Audio Device".
    let display_name = desc
        .and_then(|d| d.lines().next())
        .filter(|line| !line.is_empty())
        .unwrap_or(name);
    Device {
        display_name: display_name.to_string(),
        klass: klass.to_string(),
        // hint names are built from the card name, so they survive a replug;
        // the direction qualifies it because one hint yields both halves.
        persistent_id: format!("{}:{name}", dir_word(dir)),
        caps: CapsSet::from_alternatives(Vec::new()),
        element,
        props: vec![(String::from("device"), name.to_string())],
        detail: Vec::new(),
        provider: PROVIDER,
    }
}

/// The `hw_params` space of one PCM: what it can be opened as.
#[derive(Debug, Clone, PartialEq)]
struct HwProbe {
    rate_min: u32,
    rate_max: u32,
    channels_min: u32,
    channels_max: u32,
    formats: Vec<AudioFormat>,
}

impl HwProbe {
    /// One fixed alternative per supported format, preference order following
    /// [`FORMATS`].
    fn caps(&self) -> CapsSet {
        let sample_rate = clamped(PREFERRED_RATE, self.rate_min, self.rate_max);
        let Ok(channels) = u8::try_from(clamped(
            PREFERRED_CHANNELS,
            self.channels_min,
            self.channels_max,
        )) else {
            return CapsSet::from_alternatives(Vec::new());
        };
        CapsSet::from_alternatives(
            self.formats
                .iter()
                .map(|&format| Caps::Audio {
                    format,
                    channels,
                    sample_rate,
                })
                .collect(),
        )
    }

    /// The ranges the fixed caps above collapsed, kept for display.
    fn detail(&self) -> Vec<(String, String)> {
        vec![
            (String::from("rate-min"), self.rate_min.to_string()),
            (String::from("rate-max"), self.rate_max.to_string()),
            (String::from("channels-min"), self.channels_min.to_string()),
            (String::from("channels-max"), self.channels_max.to_string()),
        ]
    }
}

/// alsa reports each range's ends independently, and a device can report a max
/// below its min; keep the clamp from panicking on that.
fn clamped(want: u32, min: u32, max: u32) -> u32 {
    want.clamp(min, min.max(max))
}

/// Open the PCM nonblocking and read its parameter space. Errno on failure, so
/// one busy device does not fail the whole enumeration.
fn probe_pcm(name: &str, dir: Direction) -> Result<HwProbe, i32> {
    let pcm = PCM::new(name, dir, true).map_err(|e| e.errno())?;
    let hwp = HwParams::any(&pcm).map_err(|e| e.errno())?;
    let formats = FORMATS
        .iter()
        .filter(|(_, fmt)| hwp.test_format(*fmt).is_ok())
        .map(|(format, _)| *format)
        .collect();
    Ok(HwProbe {
        rate_min: hwp.get_rate_min().map_err(|e| e.errno())?,
        rate_max: hwp.get_rate_max().map_err(|e| e.errno())?,
        channels_min: hwp.get_channels_min().map_err(|e| e.errno())?,
        channels_max: hwp.get_channels_max().map_err(|e| e.errno())?,
        formats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(rate: (u32, u32), channels: (u32, u32), formats: &[AudioFormat]) -> HwProbe {
        HwProbe {
            rate_min: rate.0,
            rate_max: rate.1,
            channels_min: channels.0,
            channels_max: channels.1,
            formats: formats.to_vec(),
        }
    }

    #[test]
    fn skeleton_maps_each_direction_onto_its_element() {
        let capture = device_skeleton(
            "hw:0,0",
            Some("HDA Intel PCH, ALC257 Analog\nDefault Audio Device"),
            Direction::Capture,
        );
        assert_eq!(capture.element, "alsasrc");
        assert_eq!(capture.klass, "Audio/Source");
        assert!(capture.has_classes("Source"));
        assert_eq!(capture.display_name, "HDA Intel PCH, ALC257 Analog");
        assert_eq!(capture.provider, "alsa");
        assert_eq!(capture.launch_fragment(), "alsasrc device=hw:0,0");
        // an unprobed skeleton advertises nothing.
        assert!(capture.caps.is_empty());

        let playback = device_skeleton("hw:0,0", None, Direction::Playback);
        assert_eq!(playback.element, "alsasink");
        assert_eq!(playback.klass, "Audio/Sink");
        // no desc: the hint name is the label.
        assert_eq!(playback.display_name, "hw:0,0");
        assert_eq!(playback.launch_fragment(), "alsasink device=hw:0,0");
        // the two halves of one hint must not share an identity, or the
        // monitor's poll-and-diff cannot tell them apart.
        assert_ne!(capture.persistent_id, playback.persistent_id);
    }

    #[test]
    fn listed_directions_follow_the_hint_direction() {
        assert!(!listed_directions(Some(Direction::Playback)).contains(&Direction::Capture));
        assert!(!listed_directions(Some(Direction::Capture)).contains(&Direction::Playback));
        #[cfg(feature = "alsa-src")]
        assert_eq!(
            listed_directions(Some(Direction::Capture)),
            [Direction::Capture]
        );
        #[cfg(feature = "alsa-sink")]
        assert_eq!(
            listed_directions(Some(Direction::Playback)),
            [Direction::Playback]
        );
        // a hint serving both directions yields both devices.
        #[cfg(all(feature = "alsa-src", feature = "alsa-sink"))]
        assert_eq!(
            listed_directions(None),
            [Direction::Capture, Direction::Playback]
        );
    }

    #[test]
    fn hint_devices_skips_null_and_keeps_default() {
        assert!(hint_devices("null", Some("Discard all samples"), None).is_empty());
        let kept = hint_devices("default", Some("Default"), None);
        assert!(!kept.is_empty());
        for (dir, device) in kept {
            assert_eq!(
                device.props,
                [(String::from("device"), String::from("default"))]
            );
            assert_eq!(device.persistent_id, format!("{}:default", dir_word(dir)));
        }
    }

    #[test]
    fn caps_take_the_preferred_rate_and_stereo_when_the_device_allows() {
        let caps = probe(
            (8_000, 192_000),
            (1, 8),
            &[AudioFormat::PcmS16Le, AudioFormat::PcmF32Le],
        )
        .caps();
        assert_eq!(
            caps.alternatives(),
            [
                Caps::Audio {
                    format: AudioFormat::PcmS16Le,
                    channels: 2,
                    sample_rate: 48_000,
                },
                Caps::Audio {
                    format: AudioFormat::PcmF32Le,
                    channels: 2,
                    sample_rate: 48_000,
                },
            ]
        );
    }

    #[test]
    fn caps_clamp_into_a_narrow_device_range() {
        // a mono 8 kHz-only device: both preferences clamp down.
        let hw = probe((8_000, 16_000), (1, 1), &[AudioFormat::PcmS16Le]);
        assert_eq!(
            hw.caps().alternatives(),
            [Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 1,
                sample_rate: 16_000,
            }]
        );
        // nothing about the real range is lost.
        assert_eq!(
            hw.detail(),
            [
                (String::from("rate-min"), String::from("8000")),
                (String::from("rate-max"), String::from("16000")),
                (String::from("channels-min"), String::from("1")),
                (String::from("channels-max"), String::from("1")),
            ]
        );
        // a device that only goes above the preferred rate clamps up.
        let hw = probe((96_000, 192_000), (4, 4), &[AudioFormat::PcmS32Le]);
        assert_eq!(
            hw.caps().alternatives(),
            [Caps::Audio {
                format: AudioFormat::PcmS32Le,
                channels: 4,
                sample_rate: 96_000,
            }]
        );
    }

    #[test]
    fn a_device_with_no_usable_format_advertises_nothing() {
        assert!(probe((8_000, 48_000), (1, 2), &[]).caps().is_empty());
        // a channel count we cannot express is dropped rather than truncated.
        assert!(
            probe((48_000, 48_000), (300, 400), &[AudioFormat::PcmS16Le])
                .caps()
                .is_empty()
        );
    }

    /// Real enumeration: a machine with no sound card (CI) sees no devices,
    /// which is fine; whatever it does see must be well formed.
    #[test]
    fn probe_lists_well_formed_devices() {
        let Ok(devices) = AlsaDeviceProvider::new().probe() else {
            return;
        };
        let mut ids = Vec::new();
        for device in &devices {
            assert_eq!(device.provider, "alsa");
            assert!(matches!(device.element, "alsasrc" | "alsasink"));
            let (key, value) = &device.props[0];
            assert_eq!(key, "device");
            assert!(!value.is_empty());
            assert_ne!(value, "null");
            assert!(!ids.contains(&device.persistent_id));
            ids.push(device.persistent_id.clone());
        }
    }
}

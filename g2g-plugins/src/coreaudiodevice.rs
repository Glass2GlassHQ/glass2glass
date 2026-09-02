//! Core Audio device discovery (M943): the macOS audio half of the
//! [`DeviceMonitor`](g2g_core::runtime::DeviceMonitor), the counterpart of
//! [`alsadevice`](crate::alsadevice) on Linux. The HAL
//! (`kAudioHardwarePropertyDevices`) lists every audio device; one with input
//! channels drives [`CoreAudioSrc`](crate::coreaudio::CoreAudioSrc), one with
//! output channels [`CoreAudioSink`](crate::coreaudio::CoreAudioSink), and a
//! duplex device yields one [`Device`] per direction.
//!
//! Selection and identity are both the device UID: unlike the `AudioDeviceID`,
//! which the HAL reassigns on every boot, the UID is stable, which is why it is
//! the `device=` property and the persistent id.
//!
//! Caps are S16LE at the device's nominal rate and channel count, the shape the
//! AudioQueue elements carry; the raw HAL numbers stay in `detail`.
//!
//! No native watch: the HAL offers `AudioObjectAddPropertyListener`, but its
//! callbacks arrive on a run loop this library does not own, so the monitor
//! polls. (The Windows sibling, `wasapidevice`, does have a push watch.)
//!
//! macOS-only and compile-checked cross-target from Linux; the enumeration is
//! owed a run on a Mac.

use core::ffi::c_void;
use core::mem;
use core::ptr::NonNull;

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use objc2_core_audio::{
    kAudioDevicePropertyDeviceUID, kAudioDevicePropertyNominalSampleRate,
    kAudioDevicePropertyStreamConfiguration, kAudioHardwarePropertyDevices,
    kAudioObjectPropertyElementMain, kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyScopeInput, kAudioObjectPropertyScopeOutput, kAudioObjectSystemObject,
    AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize, AudioObjectID,
    AudioObjectPropertyAddress,
};
use objc2_core_audio_types::AudioBufferList;
use objc2_core_foundation::{CFRetained, CFString};

use g2g_core::runtime::{Device, DeviceProvider};
use g2g_core::{AudioFormat, Caps, CapsSet, G2gError, HardwareError};

/// [`Device::provider`] for everything this backend finds.
const PROVIDER: &str = "coreaudio";

/// Fallback rate for a device that will not report one.
const DEFAULT_RATE: u32 = 48_000;

/// Lists the Core Audio input / output devices.
#[derive(Debug, Default, Clone, Copy)]
pub struct CoreAudioDeviceProvider;

impl CoreAudioDeviceProvider {
    pub fn new() -> Self {
        Self
    }
}

impl DeviceProvider for CoreAudioDeviceProvider {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    fn probe(&self) -> Result<Vec<Device>, G2gError> {
        let mut out = Vec::new();
        for id in device_ids()? {
            // A device that will not name itself cannot be selected later, so
            // it is skipped rather than listed unusably.
            let Some(uid) = string_property(id, kAudioDevicePropertyDeviceUID) else {
                continue;
            };
            let name = string_property(id, kAudioObjectPropertyName).unwrap_or_else(|| uid.clone());
            let rate = nominal_rate(id).unwrap_or(DEFAULT_RATE);
            for direction in [Direction::Capture, Direction::Render] {
                let channels = channel_count(id, direction);
                if channels == 0 {
                    continue;
                }
                out.push(describe(&uid, &name, direction, channels, rate));
            }
        }
        Ok(out)
    }
}

/// Which half of a (possibly duplex) device a listing entry describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Capture,
    Render,
}

impl Direction {
    /// The element that drives this direction, its class, and the id suffix
    /// that keeps the two halves of one duplex device distinct.
    const fn element(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Direction::Capture => ("coreaudiosrc", "Audio/Source", "capture"),
            Direction::Render => ("coreaudiosink", "Audio/Sink", "playback"),
        }
    }

    fn scope(self) -> u32 {
        match self {
            Direction::Capture => kAudioObjectPropertyScopeInput,
            Direction::Render => kAudioObjectPropertyScopeOutput,
        }
    }
}

/// One direction of one device as a [`Device`]. Pure, so the mapping is
/// unit-tested without audio hardware.
fn describe(uid: &str, name: &str, direction: Direction, channels: u8, rate: u32) -> Device {
    let (element, klass, tag) = direction.element();
    Device {
        display_name: name.to_string(),
        klass: klass.to_string(),
        // the UID alone is not unique across a duplex device's two halves, and
        // the monitor's diff key has to be.
        persistent_id: {
            let mut id = String::from(tag);
            id.push(':');
            id.push_str(uid);
            id
        },
        caps: CapsSet::one(Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels,
            sample_rate: rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }),
        element,
        props: Vec::from([("device".to_string(), uid.to_string())]),
        detail: Vec::from([
            ("uid".to_string(), uid.to_string()),
            ("channels".to_string(), channels.to_string()),
            ("nominal-rate".to_string(), rate.to_string()),
        ]),
        provider: PROVIDER,
    }
}

fn address(selector: u32, scope: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain,
    }
}

fn hw() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// Every device the HAL knows.
fn device_ids() -> Result<Vec<AudioObjectID>, G2gError> {
    let mut addr = address(
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
    );
    let system = kAudioObjectSystemObject as AudioObjectID;
    let mut size = 0u32;
    // SAFETY: the address and the out slot outlive the call.
    let status = unsafe {
        AudioObjectGetPropertyDataSize(
            system,
            NonNull::from(&mut addr),
            0,
            core::ptr::null(),
            NonNull::from(&mut size),
        )
    };
    if status != 0 {
        return Err(hw());
    }
    let count = size as usize / mem::size_of::<AudioObjectID>();
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut ids: Vec<AudioObjectID> = vec![0; count];
    // SAFETY: the buffer holds exactly the `size` bytes just reported.
    let status = unsafe {
        AudioObjectGetPropertyData(
            system,
            NonNull::from(&mut addr),
            0,
            core::ptr::null(),
            NonNull::from(&mut size),
            NonNull::new(ids.as_mut_ptr().cast::<c_void>()).ok_or_else(hw)?,
        )
    };
    if status != 0 {
        return Err(hw());
    }
    ids.truncate(size as usize / mem::size_of::<AudioObjectID>());
    Ok(ids)
}

/// A CFString-valued device property as a Rust string.
fn string_property(id: AudioObjectID, selector: u32) -> Option<String> {
    let mut addr = address(selector, kAudioObjectPropertyScopeGlobal);
    let mut value: *const CFString = core::ptr::null();
    let mut size = mem::size_of::<*const CFString>() as u32;
    // SAFETY: the out slot is one pointer, exactly the size declared.
    let status = unsafe {
        AudioObjectGetPropertyData(
            id,
            NonNull::from(&mut addr),
            0,
            core::ptr::null(),
            NonNull::from(&mut size),
            NonNull::from(&mut value).cast::<c_void>(),
        )
    };
    if status != 0 {
        return None;
    }
    let value = NonNull::new(value.cast_mut())?;
    // SAFETY: the HAL hands back a +1 retained CFString, so the retained
    // wrapper takes that reference over and releases it on drop.
    let text = unsafe { CFRetained::from_raw(value) };
    Some(text.to_string())
}

/// The device's nominal sample rate, rounded to whole hertz.
fn nominal_rate(id: AudioObjectID) -> Option<u32> {
    let mut addr = address(
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeGlobal,
    );
    let mut rate = 0f64;
    let mut size = mem::size_of::<f64>() as u32;
    // SAFETY: the out slot is one f64, exactly the size declared.
    let status = unsafe {
        AudioObjectGetPropertyData(
            id,
            NonNull::from(&mut addr),
            0,
            core::ptr::null(),
            NonNull::from(&mut size),
            NonNull::from(&mut rate).cast::<c_void>(),
        )
    };
    if status != 0 || !(rate.is_finite() && rate > 0.0) {
        return None;
    }
    Some(rate.round() as u32)
}

/// Channels the device carries in one direction, 0 when it has none (which is
/// what makes an output-only device an `Audio/Sink` alone). Saturates at the
/// `Caps::Audio` channel width rather than wrapping.
fn channel_count(id: AudioObjectID, direction: Direction) -> u8 {
    let mut addr = address(kAudioDevicePropertyStreamConfiguration, direction.scope());
    let mut size = 0u32;
    // SAFETY: the address and out slot outlive the call.
    let status = unsafe {
        AudioObjectGetPropertyDataSize(
            id,
            NonNull::from(&mut addr),
            0,
            core::ptr::null(),
            NonNull::from(&mut size),
        )
    };
    if status != 0 || (size as usize) < mem::size_of::<AudioBufferList>() {
        return 0;
    }
    // The reply is a variable-length AudioBufferList (a count plus that many
    // AudioBuffers), so the byte buffer is aligned to the header type.
    let words = size as usize / mem::size_of::<AudioBufferList>() + 1;
    let mut storage: Vec<AudioBufferList> = Vec::with_capacity(words);
    // SAFETY: capacity for `words` headers was just reserved; the HAL fills
    // `size` bytes, which is within that.
    let status = unsafe {
        AudioObjectGetPropertyData(
            id,
            NonNull::from(&mut addr),
            0,
            core::ptr::null(),
            NonNull::from(&mut size),
            NonNull::new_unchecked(storage.as_mut_ptr().cast::<c_void>()),
        )
    };
    if status != 0 {
        return 0;
    }
    // The count comes from the same reply as the bytes, but it is read back
    // out of a raw buffer, so it is bounded by what those bytes can hold
    // rather than trusted.
    let capacity = 1
        + (size as usize - mem::size_of::<AudioBufferList>())
            / mem::size_of::<objc2_core_audio_types::AudioBuffer>();
    // SAFETY: the call above initialised the header and the buffers behind it.
    unsafe {
        let list = &*storage.as_ptr();
        let count = (list.mNumberBuffers as usize).min(capacity);
        let buffers = core::slice::from_raw_parts(list.mBuffers.as_ptr(), count);
        let total: u32 = buffers.iter().map(|b| b.mNumberChannels).sum();
        u8::try_from(total).unwrap_or(u8::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_direction_maps_onto_its_element() {
        let capture = describe(
            "AppleUSBAudioEngine:1",
            "USB Mic",
            Direction::Capture,
            1,
            44_100,
        );
        assert_eq!(capture.element, "coreaudiosrc");
        assert!(capture.has_classes("Audio/Source"));
        assert_eq!(
            capture.props,
            [("device".to_string(), "AppleUSBAudioEngine:1".to_string())]
        );
        assert_eq!(
            capture.caps.alternatives(),
            [Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 1,
                sample_rate: 44_100,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            }]
        );

        let render = describe(
            "AppleUSBAudioEngine:1",
            "USB Mic",
            Direction::Render,
            2,
            48_000,
        );
        assert_eq!(render.element, "coreaudiosink");
        assert!(render.has_classes("Audio/Sink"));
        // the two halves of one duplex device must not share an identity, or
        // the monitor's diff cannot tell them apart.
        assert_ne!(capture.persistent_id, render.persistent_id);
        // the selection property stays the bare UID either way.
        assert_eq!(render.props, capture.props);
        assert_eq!(
            render.launch_fragment(),
            "coreaudiosink device=AppleUSBAudioEngine:1"
        );
    }

    /// Real enumeration: whatever this Mac reports must be well formed, and a
    /// machine with no audio device (never, but the probe must not assume) is
    /// simply an empty list.
    #[test]
    fn probe_lists_well_formed_devices() {
        let Ok(devices) = CoreAudioDeviceProvider::new().probe() else {
            return;
        };
        let mut ids = Vec::new();
        for device in &devices {
            assert_eq!(device.provider, PROVIDER);
            assert!(matches!(device.element, "coreaudiosrc" | "coreaudiosink"));
            assert!(!device.props[0].1.is_empty());
            assert!(!ids.contains(&device.persistent_id));
            ids.push(device.persistent_id.clone());
        }
    }
}

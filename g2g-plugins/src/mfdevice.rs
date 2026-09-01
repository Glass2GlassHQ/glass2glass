//! Media Foundation device discovery (M943): the Windows camera half of the
//! [`DeviceMonitor`](g2g_core::runtime::DeviceMonitor), the counterpart of
//! [`v4l2device`](crate::v4l2device) on Linux. `MFEnumDeviceSources` lists the
//! video capture devices, each becoming a [`Device`] that
//! [`MfVideoSrc`](crate::mfvideosrc::MfVideoSrc) opens by `device-path`, the MF
//! symbolic link (also the persistent id: it names the device and the port, so
//! it survives a replug, unlike the enumeration index).
//!
//! Caps are the device's own native modes, filtered to the NV12 / YUY2
//! subtypes the element can deliver. Reading them means activating the source,
//! so a camera another application holds open lists with empty caps rather
//! than failing the probe.
//!
//! Media Foundation offers no hotplug callback (a `WM_DEVICECHANGE` window is
//! the platform answer, which a library has no business creating), so this
//! provider has no native watch and the monitor polls it. The audio half,
//! [`wasapidevice`](crate::wasapidevice), does have one.
//!
//! Everything here is Windows-only and compile-checked cross-target from
//! Linux; the enumeration itself is owed a run on a Windows host.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::Duration;

use windows::Win32::Media::MediaFoundation::{MFShutdown, MFStartup, MFSTARTUP_FULL, MF_VERSION};
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED};

use g2g_core::runtime::{Device, DeviceProvider};
use g2g_core::{Caps, CapsSet, Dim, G2gError, HardwareError, Interlace, Rate};

use crate::mfvideosrc::{enumerate_devices, native_modes, MfDeviceInfo, VideoConfig};

/// [`Device::provider`] for everything this backend finds.
const PROVIDER: &str = "mf";

/// Upper bound on the caps alternatives one device carries, matching the V4L2
/// provider: a UVC camera reports far more modes than a listing can use.
const MAX_ALTERNATIVES: usize = 32;

/// How long the enumeration thread may take before the probe gives up.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Discovers Media Foundation video capture devices for `mfvideosrc`.
#[derive(Debug, Default, Clone, Copy)]
pub struct MfDeviceProvider;

impl MfDeviceProvider {
    pub fn new() -> Self {
        Self
    }
}

impl DeviceProvider for MfDeviceProvider {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    fn probe(&self) -> Result<Vec<Device>, G2gError> {
        // MF is COM, and the monitor's poll thread is not COM-initialised, so
        // the enumeration runs on a thread this call owns end to end (the same
        // shape as MfVideoSrc's own probe).
        let (tx, rx) = std_mpsc::sync_channel::<Result<Vec<Device>, G2gError>>(1);
        thread::Builder::new()
            .name(String::from("g2g-mfdevice-probe"))
            .spawn(move || {
                // SAFETY: COM + MF init on this thread, balanced before exit.
                let result = unsafe {
                    let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
                    let r = MFStartup(MF_VERSION, MFSTARTUP_FULL)
                        .map_err(mf_err)
                        .and_then(|()| probe_devices());
                    let _ = MFShutdown();
                    CoUninitialize();
                    r
                };
                let _ = tx.send(result);
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        rx.recv_timeout(PROBE_TIMEOUT)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?
    }
}

fn mf_err(e: windows::core::Error) -> G2gError {
    G2gError::Hardware(HardwareError::MediaFoundation(e.code().0))
}

/// # Safety
/// Must run on a COM-initialised, MF-started thread.
unsafe fn probe_devices() -> Result<Vec<Device>, G2gError> {
    // SAFETY: enumeration + per-device activation on the owning thread.
    unsafe {
        Ok(enumerate_devices()?
            .into_iter()
            .filter(|info| !info.symbolic_link.is_empty())
            .map(|info| {
                let modes = native_modes(&info.symbolic_link).unwrap_or_default();
                describe(&info, &modes)
            })
            .collect())
    }
}

/// One enumerated device plus its probed modes as a [`Device`]. Pure, so the
/// mapping is unit-tested without a camera.
fn describe(info: &MfDeviceInfo, modes: &[VideoConfig]) -> Device {
    let display_name = if info.friendly_name.is_empty() {
        info.symbolic_link.clone()
    } else {
        info.friendly_name.clone()
    };
    Device {
        display_name,
        klass: "Video/Source".to_string(),
        persistent_id: info.symbolic_link.clone(),
        caps: mode_caps(modes),
        element: "mfvideosrc",
        props: Vec::from([("device-path".to_string(), info.symbolic_link.clone())]),
        detail: Vec::from([("symbolic-link".to_string(), info.symbolic_link.clone())]),
        provider: PROVIDER,
    }
}

/// The device's native modes as caps alternatives, in driver preference order.
/// A mode with a degenerate size or an unusable frame rate is dropped rather
/// than advertised as something nothing can satisfy.
fn mode_caps(modes: &[VideoConfig]) -> CapsSet {
    let mut alternatives = Vec::new();
    for mode in modes {
        if mode.width == 0 || mode.height == 0 || alternatives.len() >= MAX_ALTERNATIVES {
            continue;
        }
        let framerate = match rate_q16(mode.fps_num, mode.fps_den) {
            Some(q16) => Rate::Fixed(q16),
            // the driver reported the size but no usable rate: unknown, not
            // absent.
            None => Rate::Any,
        };
        alternatives.push(Caps::RawVideo {
            format: mode.format.raw_format(),
            width: Dim::Fixed(mode.width),
            height: Dim::Fixed(mode.height),
            framerate,
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        });
    }
    CapsSet::from_alternatives(alternatives)
}

/// An MF `num/den` frame rate as Q16 fps, the repo's rate encoding.
fn rate_q16(num: u32, den: u32) -> Option<u32> {
    if num == 0 || den == 0 {
        return None;
    }
    u32::try_from(((num as u64) << 16) / den as u64).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mfvideosrc::MfPixelFormat;
    use g2g_core::RawVideoFormat;

    fn mode(format: MfPixelFormat, width: u32, height: u32, fps: u32) -> VideoConfig {
        VideoConfig {
            format,
            width,
            height,
            fps_num: fps,
            fps_den: 1,
        }
    }

    fn info() -> MfDeviceInfo {
        MfDeviceInfo {
            friendly_name: "Integrated Camera".to_string(),
            symbolic_link: r"\\?\usb#vid_0bda&pid_5510#01".to_string(),
        }
    }

    #[test]
    fn a_device_is_selected_by_the_symbolic_link_it_is_identified_by() {
        let device = describe(&info(), &[mode(MfPixelFormat::Nv12, 640, 480, 30)]);
        assert_eq!(device.display_name, "Integrated Camera");
        assert!(device.has_classes("Video/Source"));
        assert_eq!(device.element, "mfvideosrc");
        // the id and the property that reopens the device are the same string,
        // which is what makes a saved launch line survive a replug.
        assert_eq!(device.persistent_id, info().symbolic_link);
        assert_eq!(
            device.props,
            [("device-path".to_string(), info().symbolic_link)]
        );
        // a driver with no friendly name still gets a label.
        let unnamed = MfDeviceInfo {
            friendly_name: String::new(),
            ..info()
        };
        assert_eq!(describe(&unnamed, &[]).display_name, unnamed.symbolic_link);
    }

    #[test]
    fn native_modes_become_fixed_caps_alternatives() {
        let modes = [
            mode(MfPixelFormat::Nv12, 1280, 720, 30),
            mode(MfPixelFormat::Yuy2, 640, 480, 60),
        ];
        let caps = mode_caps(&modes);
        assert_eq!(
            caps.alternatives()[0],
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(1280),
                height: Dim::Fixed(720),
                framerate: Rate::Fixed(30 << 16),
                interlace: Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            }
        );
        assert!(matches!(
            caps.alternatives()[1],
            Caps::RawVideo {
                format: RawVideoFormat::Yuyv,
                width: Dim::Fixed(640),
                ..
            }
        ));

        // degenerate geometry is dropped; a zero rate becomes an unknown one.
        let odd = [
            mode(MfPixelFormat::Nv12, 0, 480, 30),
            VideoConfig {
                fps_den: 0,
                ..mode(MfPixelFormat::Nv12, 320, 240, 30)
            },
        ];
        let caps = mode_caps(&odd);
        assert_eq!(caps.alternatives().len(), 1);
        assert!(matches!(
            caps.alternatives()[0],
            Caps::RawVideo {
                framerate: Rate::Any,
                ..
            }
        ));

        // a camera reporting hundreds of modes is capped.
        let many: Vec<_> = (0..100)
            .map(|i| mode(MfPixelFormat::Nv12, 640 + i, 480, 30))
            .collect();
        assert_eq!(mode_caps(&many).alternatives().len(), MAX_ALTERNATIVES);
    }

    #[test]
    fn frame_rate_converts_to_q16() {
        assert_eq!(rate_q16(30, 1), Some(30 << 16));
        // the common NTSC 29.97 spelling.
        assert_eq!(rate_q16(30_000, 1001), Some(1_964_115));
        assert_eq!(rate_q16(0, 1), None);
        assert_eq!(rate_q16(30, 0), None);
    }
}

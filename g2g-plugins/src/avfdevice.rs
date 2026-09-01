//! AVFoundation camera discovery (M943): the macOS camera half of the
//! [`DeviceMonitor`](g2g_core::runtime::DeviceMonitor). An
//! `AVCaptureDeviceDiscoverySession` lists the cameras, each becoming a
//! [`Device`] that [`AvfVideoSrc`](crate::avf::AvfVideoSrc) opens by `device`,
//! the `AVCaptureDevice` unique id (also the persistent id: Apple keeps it
//! stable for a given camera).
//!
//! The advertised caps are what the element actually delivers, NV12 at the VGA
//! session preset, not the camera's full format list: `AvfVideoSrc` pins that
//! preset, so listing 4K here would promise something the element will not
//! produce. The device's model and manufacturer go in `detail`.
//!
//! Enumeration itself needs no camera permission (names and ids are visible
//! without TCC consent); opening one does, which is `AvfVideoSrc`'s problem.
//! There is no native watch: AVFoundation posts device notifications through
//! `NSNotificationCenter`, which needs a run loop this library does not own, so
//! the monitor polls.
//!
//! macOS-only and compile-checked cross-target from Linux; the enumeration is
//! owed a run on a Mac with a camera (the CI runner has none).

use alloc::string::ToString;
use alloc::vec::Vec;

use objc2_av_foundation::{
    AVCaptureDevice, AVCaptureDeviceDiscoverySession, AVCaptureDevicePosition, AVCaptureDeviceType,
    AVCaptureDeviceTypeBuiltInWideAngleCamera, AVCaptureDeviceTypeExternal, AVMediaTypeVideo,
};
use objc2_foundation::NSArray;

use g2g_core::runtime::{Device, DeviceProvider};
use g2g_core::{Caps, CapsSet, Dim, G2gError, Interlace, Rate, RawVideoFormat};

/// [`Device::provider`] for everything this backend finds.
const PROVIDER: &str = "avf";

/// The geometry `AvfVideoSrc` pins with its VGA session preset.
const PRESET_WIDTH: u32 = 640;
const PRESET_HEIGHT: u32 = 480;
const PRESET_FPS: u32 = 30;

/// Discovers AVFoundation cameras for `avfvideosrc`.
#[derive(Debug, Default, Clone, Copy)]
pub struct AvfDeviceProvider;

impl AvfDeviceProvider {
    pub fn new() -> Self {
        Self
    }
}

impl DeviceProvider for AvfDeviceProvider {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    fn probe(&self) -> Result<Vec<Device>, G2gError> {
        // SAFETY: the device-type statics are valid on macOS; the discovery
        // session and the devices it returns are ordinary retained objects.
        unsafe {
            let types: Vec<&AVCaptureDeviceType> = [
                AVCaptureDeviceTypeBuiltInWideAngleCamera,
                AVCaptureDeviceTypeExternal,
            ]
            .into_iter()
            .collect();
            let session =
                AVCaptureDeviceDiscoverySession::discoverySessionWithDeviceTypes_mediaType_position(
                    &NSArray::from_slice(&types),
                    AVMediaTypeVideo,
                    AVCaptureDevicePosition::Unspecified,
                );
            Ok(session.devices().iter().map(|d| describe(&d)).collect())
        }
    }
}

/// One camera as a [`Device`].
///
/// # Safety
/// `camera` must be a live `AVCaptureDevice`.
unsafe fn describe(camera: &AVCaptureDevice) -> Device {
    // SAFETY: plain property reads on a live device.
    let (unique_id, name, model, manufacturer) = unsafe {
        (
            camera.uniqueID().to_string(),
            camera.localizedName().to_string(),
            camera.modelID().to_string(),
            camera.manufacturer().to_string(),
        )
    };
    Device {
        display_name: if name.is_empty() {
            unique_id.clone()
        } else {
            name
        },
        klass: "Video/Source".to_string(),
        persistent_id: unique_id.clone(),
        caps: CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(PRESET_WIDTH),
            height: Dim::Fixed(PRESET_HEIGHT),
            framerate: Rate::Fixed(PRESET_FPS << 16),
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }),
        element: "avfvideosrc",
        props: Vec::from([("device".to_string(), unique_id.clone())]),
        detail: Vec::from([
            ("unique-id".to_string(), unique_id),
            ("model-id".to_string(), model),
            ("manufacturer".to_string(), manufacturer),
        ]),
        provider: PROVIDER,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whatever this Mac has (the CI runner has no camera, which is fine) must
    /// be described well enough to open.
    #[test]
    fn probe_describes_any_camera_present_and_tolerates_none() {
        let devices = AvfDeviceProvider::new().probe().expect("probe");
        for device in devices {
            assert_eq!(device.element, "avfvideosrc");
            assert_eq!(device.provider, PROVIDER);
            assert!(device.has_classes("Video/Source"));
            assert!(!device.persistent_id.is_empty());
            // the id and the property that reopens the camera are the same
            // string, which is what a saved launch line relies on.
            assert_eq!(
                device.props,
                [("device".to_string(), device.persistent_id.clone())]
            );
            assert!(!device.caps.is_empty());
            assert!(device.launch_fragment().starts_with("avfvideosrc device="));
        }
    }
}

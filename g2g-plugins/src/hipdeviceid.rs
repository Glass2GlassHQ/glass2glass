use alloc::sync::Arc;

use g2g_core::memory::{GpuDevice, HipVendor, OwnedHipBuffer};
use g2g_core::{G2gError, PropError, PropKind, PropValue, PropertySpec};

use crate::hip::HipDevice;

pub(crate) const HIP_DEVICE_ID_PROP: PropertySpec = PropertySpec::new(
    "device-id",
    PropKind::Int,
    "HIP device ordinal, -1 for the device the frames arrive on",
)
.with_default("-1")
.with_range("-1", "2147483647");

pub(crate) const HIP_VENDOR_PROP: PropertySpec = PropertySpec::new(
    "vendor",
    PropKind::Str,
    "GPU vendor of the HIP device, unknown tries AMD then NVIDIA",
)
.with_default("unknown")
.with_enum_values("unknown | amd | nvidia");

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct HipDeviceChoice {
    vendor: Option<HipVendor>,
    device_id: Option<i32>,
}

impl HipDeviceChoice {
    pub(crate) fn of(device: GpuDevice) -> Self {
        Self {
            vendor: Some(device.vendor),
            device_id: Some(device.ordinal),
        }
    }

    pub(crate) fn is_set(&self) -> bool {
        self.vendor.is_some() || self.device_id.is_some()
    }

    pub(crate) fn declared(&self) -> Option<GpuDevice> {
        self.vendor.map(|vendor| GpuDevice {
            vendor,
            ordinal: self.device_id.unwrap_or(0),
        })
    }

    pub(crate) fn holds(&self, buffer: &OwnedHipBuffer) -> bool {
        let vendor_matches = self.vendor.is_none_or(|vendor| vendor == buffer.vendor);
        !self.is_set() || (vendor_matches && self.device_id.unwrap_or(0) == buffer.device_ordinal)
    }

    pub(crate) fn open(&self) -> Result<Arc<HipDevice>, G2gError> {
        let ordinal = self.device_id.unwrap_or(0);
        match self.vendor {
            Some(vendor) => HipDevice::open(vendor, ordinal),
            None => HipDevice::open(HipVendor::Amd, ordinal)
                .or_else(|_| HipDevice::open(HipVendor::Nvidia, ordinal)),
        }
    }

    // `None` when `name` is neither `device-id` nor `vendor`
    pub(crate) fn set_property(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        match name {
            "device-id" => Some(self.set_device_id(value)),
            "vendor" => Some(self.set_vendor(value)),
            _ => None,
        }
    }

    fn set_device_id(&mut self, value: &PropValue) -> Result<(), PropError> {
        let device_id = value.as_int().ok_or(PropError::Type)?;
        self.device_id = match device_id {
            -1 => None,
            0..=0x7fff_ffff => Some(device_id as i32),
            _ => return Err(PropError::Value),
        };
        Ok(())
    }

    fn set_vendor(&mut self, value: &PropValue) -> Result<(), PropError> {
        self.vendor = match value.as_str().ok_or(PropError::Type)? {
            "unknown" => None,
            "amd" => Some(HipVendor::Amd),
            "nvidia" => Some(HipVendor::Nvidia),
            _ => return Err(PropError::Value),
        };
        Ok(())
    }

    pub(crate) fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "device-id" => Some(PropValue::Int(self.device_id.map_or(-1, i64::from))),
            "vendor" => Some(PropValue::Str(
                match self.vendor {
                    None => "unknown",
                    Some(HipVendor::Amd) => "amd",
                    Some(HipVendor::Nvidia) => "nvidia",
                }
                .into(),
            )),
            _ => None,
        }
    }
}

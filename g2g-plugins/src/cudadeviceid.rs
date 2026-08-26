//! Shared `cuda-device-id` runtime property for the elements that open their own
//! CUDA context.
//!
//! Every one of them picks a device ordinal at context creation and stamps it
//! onto the `OwnedCudaBuffer` of each frame it emits, so the name, kind, range
//! and default live here once instead of being copied into each element.

use g2g_core::{PropError, PropKind, PropValue, PropertySpec};

/// CUDA device an element opens when nothing sets `cuda-device-id`.
pub(crate) const DEFAULT_CUDA_DEVICE_ID: i32 = 0;

/// Largest ordinal `cuDeviceGet` can take: it is a C `int`.
const MAX_CUDA_DEVICE_ID: i64 = i32::MAX as i64;

/// The `cuda-device-id` spec every element that opens a CUDA context declares,
/// named as gst-nvcodec's elements name it. When the ordinal is read (and
/// whether a later set is refused) is on each element's builder.
pub(crate) const CUDA_DEVICE_ID_PROP: PropertySpec = PropertySpec::new(
    "cuda-device-id",
    PropKind::Int,
    "CUDA device ordinal the element opens its context on, and the ordinal its frames carry",
)
.with_default("0")
.with_range("0", "2147483647");

/// Apply a `cuda-device-id` set to `field`. Refused once `context_open`: the
/// context already exists on the old device and the frames in flight carry its
/// ordinal.
pub(crate) fn set_cuda_device_id(
    field: &mut i32,
    context_open: bool,
    value: &PropValue,
) -> Result<(), PropError> {
    if context_open {
        return Err(PropError::ReadOnly);
    }
    let ordinal = value.as_int().ok_or(PropError::Type)?;
    if !(0..=MAX_CUDA_DEVICE_ID).contains(&ordinal) {
        return Err(PropError::Value);
    }
    *field = ordinal as i32;
    Ok(())
}

/// The read half of [`set_cuda_device_id`].
pub(crate) fn get_cuda_device_id(field: i32) -> PropValue {
    PropValue::Int(field as i64)
}

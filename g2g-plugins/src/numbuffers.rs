//! Shared `num-buffers` runtime-property conversion for the source elements.
//!
//! gst `basesrc` spells the property as a signed int where -1 means "run
//! forever" and n >= 0 means "emit n buffers then EOS". Sources here keep the
//! limit as a `u64` countdown, so the two halves of that conversion live here
//! once rather than being copied into every element's `set_property` /
//! `get_property`.

use g2g_core::property::{PropError, PropValue};

/// Set a `u64` buffer-limit field from a gst-style `num-buffers` value:
/// negative selects unlimited (`u64::MAX`), n >= 0 a bounded run (0 emits
/// nothing).
pub(crate) fn set_num_buffers(limit: &mut u64, value: &PropValue) -> Result<(), PropError> {
    let n = value.as_int().ok_or(PropError::Type)?;
    *limit = if n < 0 { u64::MAX } else { n as u64 };
    Ok(())
}

/// The read half of [`set_num_buffers`]: unlimited reads back as -1.
pub(crate) fn get_num_buffers(limit: u64) -> PropValue {
    PropValue::Int(if limit == u64::MAX { -1 } else { limit as i64 })
}

/// Set a `frame_limit: u64` field (0 = unlimited) from a gst-style
/// `num-buffers` value: -1 selects unlimited, positive n a bounded run. 0 is
/// rejected, since the internal sentinel cannot express "emit none".
#[cfg(any(feature = "rtsp-server", feature = "srt", feature = "udp-ingress"))]
pub(crate) fn set_frame_limit(limit: &mut u64, value: &PropValue) -> Result<(), PropError> {
    match value.as_int().ok_or(PropError::Type)? {
        n if n < 0 => {
            *limit = 0;
            Ok(())
        }
        0 => Err(PropError::Value),
        n => {
            *limit = n as u64;
            Ok(())
        }
    }
}

/// The read half of [`set_frame_limit`]: unlimited reads back as -1.
#[cfg(any(feature = "rtsp-server", feature = "srt", feature = "udp-ingress"))]
pub(crate) fn get_frame_limit(limit: u64) -> PropValue {
    PropValue::Int(match limit {
        0 => -1,
        n => n as i64,
    })
}

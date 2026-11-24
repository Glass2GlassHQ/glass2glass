// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Shared `num-buffers` runtime-property conversion for the source elements.
//!
//! gst `basesrc` spells the property as a signed int where -1 means "run
//! forever" and n >= 0 means "emit n buffers then EOS". Sources here keep the
//! limit as a `u64` countdown, so the two halves of that conversion live here
//! once rather than being copied into every element's `set_property` /
//! `get_property`.

use g2g_core::{PropError, PropKind, PropValue, PropertySpec};

pub const fn num_buffers_property(description: &'static str) -> PropertySpec {
    PropertySpec::new("num-buffers", PropKind::Int, description)
        .with_default("-1")
        .with_range("-1", "9223372036854775807")
}

/// Set a `u64` buffer-limit field from a gst-style `num-buffers` value:
/// negative selects unlimited (`u64::MAX`), n >= 0 a bounded run (0 emits
/// nothing).
pub fn set_num_buffers(limit: &mut u64, value: &PropValue) -> Result<(), PropError> {
    let n = value.as_int().ok_or(PropError::Type)?;
    *limit = if n < 0 { u64::MAX } else { n as u64 };
    Ok(())
}

/// The read half of [`set_num_buffers`]: unlimited reads back as -1.
pub fn get_num_buffers(limit: u64) -> PropValue {
    PropValue::Int(if limit == u64::MAX { -1 } else { limit as i64 })
}

/// A zero limit means "emit nothing, EOS at once", so a source checks this
/// first and never opens its device. Pushes the EOS and reports `true` when the
/// run is already over.
pub async fn finished_at_zero_limit(
    limit: u64,
    out: &mut dyn g2g_core::OutputSink,
) -> Result<bool, g2g_core::G2gError> {
    if limit != 0 {
        return Ok(false);
    }
    out.push(g2g_core::PipelinePacket::Eos).await?;
    Ok(true)
}

/// [`finished_at_zero_limit`] for a multi-pad source: every pad gets the EOS,
/// since a branch left without one never finishes.
#[cfg(any(
    feature = "moqt",
    feature = "rtsp",
    feature = "rtsp-server",
    feature = "webrtc",
    feature = "webrtc-livekit"
))]
pub(crate) async fn finished_at_zero_limit_multi(
    limit: u64,
    pads: usize,
    out: &mut dyn g2g_core::MultiOutputSink,
) -> Result<bool, g2g_core::G2gError> {
    if limit != 0 {
        return Ok(false);
    }
    for port in 0..pads {
        out.push_to(port, g2g_core::PipelinePacket::Eos).await?;
    }
    Ok(true)
}

//! Shared plumbing for the CPU video-effect transforms: negotiation,
//! the `DataFrame` loop, and the caps emit that every one of them repeats.
//!
//! An element implements [`PixelFilter`] (its accepted formats, its per-frame
//! pixel work, and where its [`FilterState`] lives) and forwards the four
//! `AsyncElement` hooks to the free functions here. Geometry-changing filters
//! override [`PixelFilter::output_dims`].

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, Interlace, MemoryDomain,
    OutputSink, PadTemplate, PipelinePacket, PushOutcome, Rate, RawVideoFormat, Reconfigure,
};

use crate::pixel::frame_byte_size;

/// The negotiated stream and the caps bookkeeping a transform needs between
/// frames. Every [`PixelFilter`] owns one.
#[derive(Debug, Default)]
pub(crate) struct FilterState {
    /// Format, dims, and framerate of the configured input stream.
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl FilterState {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// A per-frame CPU pixel transform over system memory.
pub(crate) trait PixelFilter {
    /// The raw formats this filter reads and writes, in preference order.
    const FORMATS: &'static [RawVideoFormat];

    fn state(&self) -> &FilterState;
    fn state_mut(&mut self) -> &mut FilterState;

    /// Output geometry for a `w x h` input. The default preserves it; a
    /// cropping filter overrides, together with [`apply_resized`](Self::apply_resized).
    fn output_dims(&self, _format: RawVideoFormat, w: u32, h: u32) -> (u32, u32) {
        (w, h)
    }

    /// Transform one whole frame in place. `pixels` is exactly
    /// [`frame_byte_size`]`(format, w, h)` bytes, the frame's own buffer.
    fn apply(&mut self, format: RawVideoFormat, w: u32, h: u32, pixels: &mut [u8]);

    /// A geometry-changing filter's whole output frame at
    /// [`output_dims`](Self::output_dims), or `None` to run [`apply`](Self::apply)
    /// in place instead.
    fn apply_resized(
        &mut self,
        _format: RawVideoFormat,
        _w: u32,
        _h: u32,
        _src: &[u8],
    ) -> Option<Box<[u8]>> {
        None
    }

    /// Drop any inter-frame state on a flush. The default keeps none.
    fn reset(&mut self) {}
}

/// The input caps a filter accepts: one of its formats at fixed non-zero dims.
pub(crate) fn accept_input<F: PixelFilter>(
    caps: &Caps,
) -> Result<(RawVideoFormat, u32, u32, Rate), G2gError> {
    let Caps::RawVideo {
        format,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate,
        ..
    } = caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    if !F::FORMATS.contains(format) || *w == 0 || *h == 0 {
        return Err(G2gError::CapsMismatch);
    }
    Ok((*format, *w, *h, framerate.clone()))
}

/// Narrow upstream's offer to the first of the filter's formats it can hold,
/// keeping upstream's geometry and framerate.
pub(crate) fn intercept_caps<F: PixelFilter>(upstream_caps: &Caps) -> Result<Caps, G2gError> {
    for &format in F::FORMATS {
        if let Ok(narrowed) = upstream_caps.intersect(&any_geometry(format)) {
            return Ok(narrowed);
        }
    }
    Err(G2gError::CapsMismatch)
}

/// Record the negotiated input caps.
pub(crate) fn configure<F: PixelFilter>(
    filter: &mut F,
    absolute_caps: &Caps,
) -> Result<ConfigureOutcome, G2gError> {
    let input = accept_input::<F>(absolute_caps)?;
    let state = filter.state_mut();
    state.input = Some(input);
    state.configured = true;
    Ok(ConfigureOutcome::Accepted)
}

/// The `DerivedOutput` constraint of a filter that preserves format, geometry,
/// and framerate.
pub(crate) fn same_caps_constraint<F: PixelFilter>() -> CapsConstraint<'static> {
    CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
        Caps::RawVideo { format, .. } if F::FORMATS.contains(format) => CapsSet::one(input.clone()),
        _ => CapsSet::from_alternatives(Vec::new()),
    }))
}

/// Sink and source templates covering the filter's formats at any geometry.
pub(crate) fn pad_templates<F: PixelFilter>() -> Vec<PadTemplate> {
    pad_templates_for(F::FORMATS)
}

/// Sink and source templates for a passthrough analyser that keeps `formats`.
pub(crate) fn pad_templates_for(formats: &[RawVideoFormat]) -> Vec<PadTemplate> {
    let set = CapsSet::from_alternatives(formats.iter().copied().map(any_geometry).collect());
    Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
}

pub(crate) fn any_geometry(format: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// Whether the pre-send check held the packet just pushed back rather than
/// enqueuing it, to hand this element a downstream `AbsorbOrientation`. The
/// packet has to be pushed again.
fn held_back(outcome: PushOutcome) -> bool {
    matches!(
        outcome,
        PushOutcome::Reconfigure(Reconfigure::AbsorbOrientation(_))
    )
}

/// Push a control packet, repeating it when the arm held it back. The packet is
/// rebuilt rather than cloned: `PipelinePacket` is not `Clone`.
async fn push_control(
    out: &mut dyn OutputSink,
    mut packet: impl FnMut() -> PipelinePacket,
) -> Result<(), G2gError> {
    if held_back(out.push(packet()).await?) {
        out.push(packet()).await?;
    }
    Ok(())
}

/// Run one packet through `filter`: transform a data frame, emit the output
/// caps when they change, forward everything else.
pub(crate) fn drive<'a, F: PixelFilter>(
    filter: &'a mut F,
    packet: PipelinePacket,
    out: &'a mut dyn OutputSink,
) -> Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>> {
    Box::pin(async move {
        if !filter.state().configured {
            return Err(G2gError::NotConfigured);
        }
        match packet {
            PipelinePacket::DataFrame(mut frame) => {
                let Some((format, in_w, in_h, rate)) = filter.state().input.clone() else {
                    return Err(G2gError::NotConfigured);
                };
                let in_bytes = frame_byte_size(format, in_w, in_h);
                let src = frame
                    .domain
                    .require_system_slice(g2g_core::log::short_type_name::<F>())?;
                if src.len() < in_bytes {
                    return Err(G2gError::CapsMismatch);
                }
                if let Some(resized) = filter.apply_resized(format, in_w, in_h, &src[..in_bytes]) {
                    frame.domain = MemoryDomain::System(SystemSlice::from_boxed(resized));
                } else {
                    let MemoryDomain::System(slice) = &mut frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    filter.apply(format, in_w, in_h, &mut slice.as_mut_slice()[..in_bytes]);
                }

                let (out_w, out_h) = filter.output_dims(format, in_w, in_h);
                let new_caps = Caps::RawVideo {
                    format,
                    width: Dim::Fixed(out_w),
                    height: Dim::Fixed(out_h),
                    framerate: rate,
                    interlace: Interlace::Any,
                    colorimetry: g2g_core::Colorimetry::UNKNOWN,
                };
                if filter.state().last_caps.as_ref() != Some(&new_caps) {
                    push_control(out, || PipelinePacket::CapsChanged(new_caps.clone())).await?;
                    filter.state_mut().last_caps = Some(new_caps);
                }
                let sequence = filter.state().emitted;
                filter.state_mut().emitted += 1;
                let out_frame = Frame {
                    domain: frame.domain,
                    timing: frame.timing,
                    sequence,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(out_frame)).await?;
            }
            // `c` is the runner arm's forward *output* caps (it already called
            // configure_pipeline for our input). Forward it and record it to
            // suppress the data path's duplicate emit; do NOT read it back as
            // our input, which would clobber the input with our own output.
            PipelinePacket::CapsChanged(c) => {
                push_control(out, || PipelinePacket::CapsChanged(c.clone())).await?;
                filter.state_mut().last_caps = Some(c);
            }
            PipelinePacket::Flush => {
                filter.reset();
                filter.state_mut().last_caps = None;
                push_control(out, || PipelinePacket::Flush).await?;
            }
            PipelinePacket::Segment(seg) => {
                push_control(out, || PipelinePacket::Segment(seg)).await?;
            }
            PipelinePacket::Eos => {}
            other => {
                out.push(other).await?;
            }
        }
        Ok(())
    })
}

/// `filter` applied to a copy of `src`.
#[cfg(test)]
pub(crate) fn applied<F: PixelFilter>(
    filter: &mut F,
    format: RawVideoFormat,
    w: u32,
    h: u32,
    src: &[u8],
) -> Vec<u8> {
    let mut pixels = src.to_vec();
    filter.apply(format, w, h, &mut pixels);
    pixels
}

#[cfg(test)]
mod tests {
    use g2g_core::AsyncElement;

    use super::*;
    use crate::alpha::Alpha;
    use crate::gamma::Gamma;
    use crate::testutil::{first_caps, CollectSink};
    use crate::videobalance::VideoBalance;

    const WIDTH: u32 = 4;
    const HEIGHT: u32 = 2;

    fn forwards_caps_changed<E: AsyncElement>(mut element: E) {
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        element.configure_pipeline(&caps).unwrap();
        let mut sink = CollectSink::default();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime
            .block_on(element.process(PipelinePacket::CapsChanged(caps.clone()), &mut sink))
            .unwrap();
        assert_eq!(first_caps(&sink.packets), Some(caps));
    }

    #[test]
    fn colour_filters_forward_caps_changed() {
        forwards_caps_changed(Gamma::new());
        forwards_caps_changed(VideoBalance::new());
        forwards_caps_changed(Alpha::new());
    }
}

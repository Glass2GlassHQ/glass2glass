//! Caps filter: a pass-through transform that forces a negotiation-time
//! narrowing (DESIGN-M16-caps-nego.md §7). Data flows through unchanged;
//! the element's only job is to constrain the link to a specific
//! `CapsSet` so the solver narrows the chain to it.
//!
//! Native constraint is `Identity(set)`: input == output, both drawn from
//! the filter set. Insert one anywhere a downstream peer is too permissive
//! (e.g. an `AcceptsAny` sink) and you need to pin a concrete format.
//!
//! Per the transform contract (see `run_source_transform_sink`), this
//! element does NOT emit `Eos` itself — the runner forwards the EOS
//! sentinel after `process(Eos)` returns.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, OutputSink,
    PipelinePacket,
};

#[derive(Debug)]
pub struct CapsFilter {
    filter: CapsSet,
    forwarded: u64,
    configured: bool,
}

impl CapsFilter {
    /// Filter to a single concrete description (the common case: force
    /// one format / geometry).
    pub fn new(caps: Caps) -> Self {
        Self::from_set(CapsSet::one(caps))
    }

    /// Filter to a preference-ordered set of alternatives.
    pub fn from_set(filter: CapsSet) -> Self {
        Self {
            filter,
            forwarded: 0,
            configured: false,
        }
    }

    pub fn forwarded(&self) -> u64 {
        self.forwarded
    }
}

impl AsyncElement for CapsFilter {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Legacy / mixed-cascade path: narrow upstream against the filter,
        // honoring the set's preference order. The native solver uses the
        // `Identity` constraint below instead.
        for alt in self.filter.alternatives() {
            if let Ok(narrowed) = upstream_caps.intersect(alt) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    /// Native pass-through constraint pinned to the filter set. The solver
    /// couples input and output links and narrows both to this set.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(self.filter.clone())
    }

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        // The solver should only ever hand us caps the filter accepts;
        // fail loud if it didn't (a negotiation bug, not a runtime state).
        if !self.filter.accepts(absolute_caps) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(f) => {
                    self.forwarded += 1;
                    out.push(PipelinePacket::DataFrame(f)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // Enforce the filter mid-stream too: a change that the
                    // filter rejects is a pipeline error, surfaced loud.
                    if !self.filter.accepts(&c) {
                        return Err(G2gError::CapsMismatch);
                    }
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Dim, Rate, VideoCodec, RawVideoFormat};

    fn nv12(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        }
    }

    #[test]
    fn caps_constraint_is_identity_of_filter() {
        let f = CapsFilter::new(nv12(1920, 1080));
        let CapsConstraint::Identity(set) = f.caps_constraint_as_transform() else {
            panic!("expected Identity");
        };
        assert_eq!(set.alternatives(), &[nv12(1920, 1080)]);
    }

    #[test]
    fn intercept_narrows_compatible_upstream() {
        // Filter on NV12/any-dims narrows an any-dims upstream to itself
        // and rejects a different format.
        let f = CapsFilter::new(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        });
        assert_eq!(f.intercept_caps(&nv12(1280, 720)), Ok(nv12(1280, 720)));

        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(f.intercept_caps(&h264), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn configure_rejects_caps_outside_filter() {
        let mut f = CapsFilter::new(nv12(1920, 1080));
        assert!(f.configure_pipeline(&nv12(1920, 1080)).is_ok());

        let mut g = CapsFilter::new(nv12(1920, 1080));
        assert_eq!(
            g.configure_pipeline(&nv12(1280, 720)).err(),
            Some(G2gError::CapsMismatch)
        );
    }
}

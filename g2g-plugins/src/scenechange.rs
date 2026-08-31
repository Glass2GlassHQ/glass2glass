//! Shot-change detector (`scenechange`). Passes video through and marks a
//! frame as a keyframe when its luma SAD against the previous picture trips
//! an adaptive threshold over the last few scores (Jim Easterbrook's
//! Schroedinger detector, looking only at past frames so it adds no latency).
//! No properties. I420 plus packed RGBA / BGRA so `videotestsrc ! scenechange`
//! negotiates. CPU-only `no_std`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, RawVideoFormat,
};

use crate::pixel::{frame_byte_size, luma_at};

const FORMATS: [RawVideoFormat; 3] = [
    RawVideoFormat::I420,
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
];

/// Past-score window.
const N_DIFFS: usize = 5;

/// # Example
///
/// ```no_run
/// use g2g_plugins::scenechange::SceneChange;
///
/// let sc = SceneChange::new();
/// ```
#[derive(Debug)]
pub struct SceneChange {
    diffs: [f64; N_DIFFS],
    n_diffs: usize,
    previous: Option<Box<[u8]>>,
    changes: u64,
    last_score: f64,
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    configured: bool,
}

impl Default for SceneChange {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneChange {
    pub fn new() -> Self {
        Self {
            diffs: [0.0; N_DIFFS],
            n_diffs: 0,
            previous: None,
            changes: 0,
            last_score: 0.0,
            input: None,
            configured: false,
        }
    }

    /// Scene changes detected so far.
    pub fn changes(&self) -> u64 {
        self.changes
    }

    /// Mean absolute luma difference of the last pair, 0..255.
    pub fn last_score(&self) -> f64 {
        self.last_score
    }

    fn accept(caps: &Caps) -> Result<(RawVideoFormat, u32, u32, Rate), G2gError> {
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
        if !FORMATS.contains(format) || *w == 0 || *h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *w, *h, framerate.clone()))
    }

    fn score(format: RawVideoFormat, w: u32, h: u32, a: &[u8], b: &[u8]) -> f64 {
        let mut sad = 0u64;
        for y in 0..h {
            for x in 0..w {
                let da = luma_at(format, w, a, x, y) as i32;
                let db = luma_at(format, w, b, x, y) as i32;
                sad += da.abs_diff(db) as u64;
            }
        }
        sad as f64 / ((w as u64) * (h as u64)) as f64
    }

    /// Adaptive threshold on the past window.
    fn is_change(score: f64, diffs: &[f64; N_DIFFS], n_diffs: usize) -> bool {
        if n_diffs < N_DIFFS {
            return false;
        }
        let mut score_min = diffs[0];
        let mut score_max = diffs[0];
        for &d in diffs.iter().take(N_DIFFS - 1).skip(1) {
            score_min = score_min.min(d);
            score_max = score_max.max(d);
        }
        let threshold = 1.8 * score_max - 0.8 * score_min;
        // skip tiny diffs, then trip on a jump vs the window or a score above 50
        score >= 5.0
            && score / threshold >= 1.0
            && ((score > 30.0 && score / diffs[N_DIFFS - 2] > 1.4)
                || score / threshold > 2.3
                || score > 50.0)
    }
}

impl AsyncElement for SceneChange {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Self::accept(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo { format, .. } if FORMATS.contains(format) => {
                CapsSet::one(input.clone())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.input = Some(Self::accept(absolute_caps)?);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Scene change detector",
            "Video/Filter",
            "Detects scene changes in video",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let Some((format, w, h, _)) = self.input else {
                        return Err(G2gError::NotConfigured);
                    };
                    let src = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let need = frame_byte_size(format, w, h);
                    if src.len() < need {
                        return Err(G2gError::CapsMismatch);
                    }
                    let mut timing = frame.timing;
                    if let Some(prev) = self.previous.as_deref() {
                        let score = Self::score(format, w, h, prev, src);
                        self.last_score = score;
                        self.diffs.copy_within(1.., 0);
                        self.diffs[N_DIFFS - 1] = score;
                        self.n_diffs = self.n_diffs.saturating_add(1);
                        if Self::is_change(score, &self.diffs, self.n_diffs) {
                            self.diffs = [0.0; N_DIFFS];
                            self.n_diffs = 0;
                            self.changes += 1;
                            timing.keyframe = true;
                        }
                    }
                    self.previous = Some(src[..need].to_vec().into_boxed_slice());
                    out.push(PipelinePacket::DataFrame(Frame {
                        domain: frame.domain,
                        timing,
                        sequence: frame.sequence,
                        meta: frame.meta,
                    }))
                    .await?;
                }
                PipelinePacket::Flush => {
                    self.previous = None;
                    self.n_diffs = 0;
                    self.diffs = [0.0; N_DIFFS];
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for SceneChange {
    fn pad_templates() -> Vec<PadTemplate> {
        crate::videofx::pad_templates_for(&FORMATS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn identical_frames_are_not_a_cut() {
        let black = vec![16u8; 8 * 8];
        let score = SceneChange::score(RawVideoFormat::I420, 8, 8, &black, &black);
        assert_eq!(score, 0.0);
        assert!(!SceneChange::is_change(score, &[0.0; N_DIFFS], N_DIFFS));
    }

    #[test]
    fn large_jump_is_a_cut() {
        // Past scores sit around 2; a 60-level jump trips score > 50.
        let diffs = [2.0, 2.0, 2.0, 2.0, 60.0];
        assert!(SceneChange::is_change(60.0, &diffs, N_DIFFS));
    }
}

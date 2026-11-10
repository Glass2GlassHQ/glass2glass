use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::VecDeque;

use g2g_core::frame::Frame;
use g2g_core::{
    AnalyticsMeta, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError,
    MultiInputElement, OutputSink, PipelinePacket, Relation,
};

// gst-python-ml's metadata replay allows this window when a frame has no duration.
const UNKNOWN_DURATION_TOLERANCE_NS: u64 = 20_000_000;

pub(crate) fn frame_match_tolerance_ns(duration_ns: u64) -> u64 {
    if duration_ns == 0 {
        UNKNOWN_DURATION_TOLERANCE_NS
    } else {
        duration_ns / 2
    }
}

fn append_analytics(existing: &mut AnalyticsMeta, added: &AnalyticsMeta) -> Result<(), G2gError> {
    match (&existing.class_names, &added.class_names) {
        (Some(kept), Some(other)) if kept != other => return Err(G2gError::CapsMismatch),
        (None, Some(other)) => existing.class_names = Some(other.clone()),
        _ => {}
    }
    let offset = existing.nodes.len();
    existing.nodes.extend(added.nodes.iter().cloned());
    existing
        .relations
        .extend(added.relations.iter().map(|relation| Relation {
            from: relation.from + offset,
            to: relation.to + offset,
            kind: relation.kind,
        }));
    Ok(())
}

#[derive(Debug, Default)]
pub struct AnalyticsJoin {
    video_caps: Option<Caps>,
    held_video: VecDeque<PipelinePacket>,
    unmatched_analytics: VecDeque<(u64, AnalyticsMeta)>,
    latest_analytics_pts_ns: Option<u64>,
    analytics_ended: bool,
}

impl AnalyticsJoin {
    pub const VIDEO: usize = 0;
    pub const ANALYTICS: usize = 1;

    pub fn new() -> Self {
        Self::default()
    }

    fn analytics_passed(&self, pts_ns: u64) -> bool {
        self.analytics_ended
            || self
                .latest_analytics_pts_ns
                .is_some_and(|latest| latest >= pts_ns)
    }

    fn attach_nearest(&mut self, frame: &mut Frame) -> Result<(), G2gError> {
        let pts_ns = frame.timing.pts_ns;
        let tolerance_ns = frame_match_tolerance_ns(frame.timing.duration_ns);
        while self
            .unmatched_analytics
            .front()
            .is_some_and(|(analytics_pts_ns, _)| {
                analytics_pts_ns.saturating_add(tolerance_ns) < pts_ns
            })
        {
            self.unmatched_analytics.pop_front();
        }
        let nearest = self
            .unmatched_analytics
            .iter()
            .filter(|(analytics_pts_ns, _)| analytics_pts_ns.abs_diff(pts_ns) <= tolerance_ns)
            .min_by_key(|(analytics_pts_ns, _)| analytics_pts_ns.abs_diff(pts_ns));
        let Some((_, analytics)) = nearest else {
            return Ok(());
        };
        match frame.meta.get_mut::<AnalyticsMeta>() {
            Some(existing) => append_analytics(existing, analytics),
            None => {
                frame.meta.attach(analytics.clone());
                Ok(())
            }
        }
    }

    async fn release_decided_video(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        while let Some(packet) = self.held_video.pop_front() {
            let PipelinePacket::DataFrame(mut frame) = packet else {
                out.push(packet).await?;
                continue;
            };
            if !self.analytics_passed(frame.timing.pts_ns) {
                self.held_video.push_front(PipelinePacket::DataFrame(frame));
                return Ok(());
            }
            self.attach_nearest(&mut frame)?;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl MultiInputElement for AnalyticsJoin {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        2
    }

    fn output_follows_input(&self) -> Option<usize> {
        Some(Self::VIDEO)
    }

    fn forwards_caps_changes(&self) -> bool {
        true
    }

    fn input_pad_index(
        &self,
        req: &g2g_core::runtime::PadRequest,
        _ordinal: usize,
    ) -> Option<usize> {
        match req.kind {
            g2g_core::runtime::PadKind::Video => Some(Self::VIDEO),
            g2g_core::runtime::PadKind::Any => Some(Self::ANALYTICS),
            _ => None,
        }
    }

    fn intercept_caps(&self, input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match (input, upstream_caps) {
            (Self::ANALYTICS, _) | (Self::VIDEO, Caps::RawVideo { .. }) => {
                Ok(upstream_caps.clone())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        match (input, absolute_caps) {
            (Self::ANALYTICS, _) => Ok(ConfigureOutcome::Accepted),
            (Self::VIDEO, Caps::RawVideo { .. }) => {
                self.video_caps = Some(absolute_caps.clone());
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.video_caps.clone().ok_or(G2gError::NotConfigured)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Analytics join",
            "Filter/Video/Analytics",
            "Attaches the analytics metadata of a detector branch to the video frames at the same time",
            crate::AARON_BOXER,
        )
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match (input, packet) {
                // g2g runner sends the merged Eos once every input has ended.
                (Self::VIDEO, PipelinePacket::Eos) => Ok(()),
                (Self::VIDEO, packet) => {
                    self.held_video.push_back(packet);
                    self.release_decided_video(out).await
                }
                (_, PipelinePacket::Eos) => {
                    self.analytics_ended = true;
                    self.release_decided_video(out).await
                }
                (_, PipelinePacket::DataFrame(frame)) => {
                    let pts_ns = frame.timing.pts_ns;
                    self.latest_analytics_pts_ns = Some(pts_ns);
                    if let Some(analytics) = frame.meta.get::<AnalyticsMeta>() {
                        self.unmatched_analytics
                            .push_back((pts_ns, analytics.clone()));
                    }
                    self.release_decided_video(out).await
                }
                (_, _) => Ok(()),
            }
        })
    }
}

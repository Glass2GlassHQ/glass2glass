// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use g2g_core::progress::stream_time_of;
use g2g_core::segment::Segment;
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, ElementMetadata, FrameTiming, G2gError, OutputSink,
    PipelinePacket,
};

use crate::caps_probe::CapsReporter;
use crate::picture_filters::video_size_of;
use crate::Shared;

#[derive(Debug)]
pub(crate) struct DemuxProbe {
    shared: Arc<Shared>,
    caps: CapsReporter,
    segment: Option<Segment>,
    furthest_stream_time_ns: Option<u64>,
}

impl DemuxProbe {
    pub(crate) fn new(shared: Arc<Shared>) -> Self {
        let caps_shared = shared.clone();
        DemuxProbe {
            shared,
            caps: CapsReporter::new(move |caps| {
                if let Some(size) = video_size_of(caps) {
                    caps_shared.set_demux_size(Some(size));
                }
            }),
            segment: None,
            furthest_stream_time_ns: None,
        }
    }

    fn demuxed(&mut self, pts_ns: u64) {
        if pts_ns == FrameTiming::PTS_NONE {
            return;
        }
        let stream_time_ns = stream_time_of(self.segment.as_ref(), pts_ns);
        let furthest = self
            .furthest_stream_time_ns
            .map_or(stream_time_ns, |furthest| furthest.max(stream_time_ns));
        self.furthest_stream_time_ns = Some(furthest);
        self.shared.set_demuxed_position(Some(furthest));
    }
}

impl AsyncElement for DemuxProbe {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Player demux probe",
            "Filter/Video",
            "Reports the picture size the container declares and how far the demuxer has read",
            "Aaron Boxer <boxerab@protonmail.com>",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.caps.configure(absolute_caps);
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            self.caps.observe(&packet);
            match &packet {
                PipelinePacket::DataFrame(frame) => self.demuxed(frame.timing.pts_ns),
                PipelinePacket::Segment(segment) => self.segment = Some(*segment),
                PipelinePacket::Flush => {
                    self.furthest_stream_time_ns = None;
                    self.shared.set_demuxed_position(None);
                }
                _ => {}
            }
            out.push(packet).await?;
            Ok(())
        })
    }
}

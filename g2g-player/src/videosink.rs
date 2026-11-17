// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use g2g_core::clock::MonotonicClock;
use g2g_core::element::{PresentationStats, QosMessage};
use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ClockCandidate, ClockPriority, ClockSync, ConfigureOutcome,
    ElementMetadata, G2gError, MemoryDomain, OutputSink, Pace, PadTemplate, PadTemplates,
    PipelinePacket, PresentationPacer, PropError, PropValue, PropertySpec, Rate, PACING_PROPERTIES,
};
use g2g_plugins::clock::wait_to_present;
use g2g_plugins::glnv12::{accepted_caps, convert_for_display, DisplayFrame};

use crate::picture_filters::video_size_of;
use crate::{Event, Shared};

pub(crate) const PLAYER_VIDEO_SINK: &str = "playervideosink";

pub(crate) const PLAYER_VIDEO_SINK_METADATA: ElementMetadata = ElementMetadata::new(
    "Player video sink",
    "Sink/Video",
    "Holds each frame until it is due, then hands it to the g2g-player render contexts",
    "Aaron Boxer <boxerab@protonmail.com>",
);

#[derive(Debug)]
pub(crate) struct DueFrame {
    pub(crate) caps: Caps,
    pub(crate) frame: Frame,
}

type UpdateCallback = Arc<dyn Fn() + Send + Sync>;

#[derive(Default)]
pub(crate) struct FrameSlot {
    due: Mutex<Option<DueFrame>>,
    redraw_requested: AtomicBool,
    update: Mutex<Option<UpdateCallback>>,
}

impl std::fmt::Debug for FrameSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameSlot").finish_non_exhaustive()
    }
}

impl FrameSlot {
    fn notify_host(&self) {
        let update = self.update.lock().expect("update callback lock").clone();
        if let Some(update) = update {
            update();
        }
    }

    fn put(&self, frame: DueFrame) {
        *self.due.lock().expect("frame slot lock") = Some(frame);
        self.notify_host();
    }

    pub(crate) fn request_redraw(&self) {
        self.redraw_requested.store(true, Ordering::SeqCst);
        self.notify_host();
    }

    pub(crate) fn take_for_render(&self) -> Option<DueFrame> {
        self.redraw_requested.store(false, Ordering::SeqCst);
        self.due.lock().expect("frame slot lock").take()
    }

    pub(crate) fn wants_redraw(&self) -> bool {
        self.redraw_requested.load(Ordering::SeqCst)
            || self.due.lock().expect("frame slot lock").is_some()
    }

    pub(crate) fn set_update_callback(&self, update: Option<UpdateCallback>) {
        *self.update.lock().expect("update callback lock") = update;
    }
}

#[derive(Debug)]
pub(crate) struct PlayerVideoSink {
    shared: Arc<Shared>,
    caps: Option<Caps>,
    presented: u64,
    pacer: PresentationPacer,
    awaiting_frame_after_flush: bool,
}

impl PlayerVideoSink {
    pub(crate) fn new(shared: Arc<Shared>) -> Self {
        Self {
            shared,
            caps: None,
            presented: 0,
            pacer: PresentationPacer::for_video_sink(),
            awaiting_frame_after_flush: false,
        }
    }

    fn adopt_caps(&mut self, caps: Caps) {
        if self.caps.as_ref() != Some(&caps) {
            let framerate_q16 = match caps.framerate() {
                Rate::Fixed(framerate_q16) if framerate_q16 > 0 => Some(framerate_q16),
                _ => None,
            };
            self.shared.set_framerate_q16(framerate_q16);
            self.shared.set_video_out_params(video_size_of(&caps));
            self.caps = Some(caps);
            self.shared.events.push(Event::VideoReconfig);
        }
    }
}

impl PadTemplates for PlayerVideoSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(accepted_caps())])
    }
}

impl AsyncElement for PlayerVideoSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        PLAYER_VIDEO_SINK_METADATA
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(accepted_caps())
    }

    // TODO: take GPU frames once decoders can be opened on the host's device
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.adopt_caps(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::Provider,
            Arc::new(MonotonicClock),
        ))
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pacer.take_qos()
    }

    fn presentation_stats(&self) -> Option<PresentationStats> {
        Some(PresentationStats {
            presented: self.presented,
            dropped: 0,
            late_dropped: self.pacer.late_dropped(),
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PACING_PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.pacer
            .set_property(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.pacer.get_property(name)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    let caps = self.caps.as_ref().ok_or(G2gError::NotConfigured)?;
                    let paced = self.pacer.judge(frame.timing.pts_ns, self.presented);
                    if paced == Pace::Drop {
                        return Ok(());
                    }
                    let converting = Instant::now();
                    // the render contexts run on the host's UI thread, so they only upload
                    let bytes = frame.domain.require_system_slice(PLAYER_VIDEO_SINK)?;
                    let DisplayFrame { caps, converted } = convert_for_display(caps, bytes)?;
                    if let Some(converted) = converted {
                        frame.domain = MemoryDomain::System(SystemSlice::from_boxed(converted));
                    }
                    // a 4K conversion takes tens of milliseconds, which run before the deadline instead of after it
                    let converted_ns = converting.elapsed().as_nanos() as u64;
                    let paced = match paced {
                        Pace::Wait(ns) => Pace::Wait(ns.saturating_sub(converted_ns)),
                        other => other,
                    };
                    wait_to_present(paced).await;
                    self.presented += 1;
                    self.shared.frames.put(DueFrame { caps, frame });
                    if self.awaiting_frame_after_flush {
                        self.awaiting_frame_after_flush = false;
                        self.shared.seek_landed();
                    }
                }
                PipelinePacket::CapsChanged(caps) => self.adopt_caps(caps),
                PipelinePacket::Segment(segment) => self.pacer.set_segment(segment),
                PipelinePacket::Flush => {
                    self.pacer.flush();
                    if self.shared.take_seek_flush() {
                        self.awaiting_frame_after_flush = true;
                    }
                }
                _ => {}
            }
            Ok(())
        })
    }
}

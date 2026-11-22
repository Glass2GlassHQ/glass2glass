// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use g2g_core::caps::pcm_caps_any_rate_and_channels;
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::progress::SinkPosition;
use g2g_core::runtime::{Flow, StateController};
use g2g_core::state::PipelineState;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, ClockCandidate, ClockPriority, ClockSync,
    ConfigureOutcome, ElementMetadata, Frame, FrameTiming, G2gError, MonotonicClock, OutputSink,
    Pace, PadTemplate, PadTemplates, PipelinePacket, PresentationPacer,
};

use crate::level_meter::audio_params_of;
use crate::{AudioParams, PROPERTY_POLL_INTERVAL};

pub(crate) const PLAYER_AUDIO_SINK: &str = "playeraudiosink";
pub(crate) const PLAYER_AUDIO_SINK_METADATA: ElementMetadata = ElementMetadata::new(
    "Player audio sink",
    "Sink/Audio",
    "Hands interleaved PCM to the g2g-player host at presentation time",
    "Aaron Boxer <boxerab@protonmail.com>",
);

/// A decoded, interleaved PCM buffer borrowed for the duration of the callback.
#[derive(Debug, Clone, Copy)]
pub struct PcmBuffer<'a> {
    pub bytes: &'a [u8],
    pub format: AudioFormat,
    pub params: AudioParams,
    pub timing: FrameTiming,
}

/// Notifications from the player's host audio output.
#[derive(Debug, Clone, Copy)]
pub enum AudioOutputEvent<'a> {
    Samples(PcmBuffer<'a>),
    /// Discard any queued PCM from before the seek.
    Flush,
    /// The current stream reached its end.
    EndOfStream,
}

#[derive(Clone)]
pub(crate) struct AudioCallback(Arc<dyn Fn(AudioOutputEvent<'_>) + Send + Sync>);

impl AudioCallback {
    pub(crate) fn new(callback: impl Fn(AudioOutputEvent<'_>) + Send + Sync + 'static) -> Self {
        Self(Arc::new(callback))
    }
}

impl fmt::Debug for AudioCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioCallback").finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct PendingPcm {
    frame: Frame,
    format: AudioFormat,
    params: AudioParams,
}

#[derive(Debug)]
struct HostAudio {
    callback: AudioCallback,
    state: StateController,
}

#[derive(Debug, Default)]
pub(crate) struct PlayerAudioSink {
    host: Option<HostAudio>,
    pcm: Option<(AudioFormat, AudioParams)>,
    preroll: Option<PendingPcm>,
    pacer: PresentationPacer,
    position: Option<SinkPosition>,
    presented: u64,
}

impl PlayerAudioSink {
    pub(crate) fn new(callback: AudioCallback, state: StateController) -> Self {
        Self {
            host: Some(HostAudio { callback, state }),
            ..Self::default()
        }
    }

    async fn present(&mut self, pending: PendingPcm, generation: u64) -> Result<(), G2gError> {
        let host = self.host.as_ref().ok_or(G2gError::NotConfigured)?;
        let PendingPcm {
            frame,
            format,
            params,
        } = pending;
        loop {
            let flow = host.state.flow_gate(true, generation).await;
            if flow == Flow::Stop || host.state.preroll_generation() != generation {
                return Ok(());
            }
            match self.pacer.judge(&frame.timing, self.presented) {
                Pace::Drop => return Ok(()),
                Pace::Now if host.state.is_playing() => break,
                Pace::Now => continue,
                // A pause or seek can land while the sink waits for a sparse stream.
                Pace::Wait(ns) => {
                    tokio::time::sleep(
                        std::time::Duration::from_nanos(ns).min(PROPERTY_POLL_INTERVAL),
                    )
                    .await;
                }
            }
        }
        let bytes = frame.domain.require_system_bytes(PLAYER_AUDIO_SINK)?;
        (host.callback.0)(AudioOutputEvent::Samples(PcmBuffer {
            bytes: &bytes,
            format,
            params,
            timing: frame.timing,
        }));
        self.presented += 1;
        if let Some(position) = &self.position {
            position.set_at_pts(self.pacer.segment().as_ref(), frame.timing.pts_ns);
        }
        Ok(())
    }
}

impl PadTemplates for PlayerAudioSink {
    fn pad_templates() -> Vec<PadTemplate> {
        vec![PadTemplate::sink(pcm_caps_any_rate_and_channels(|_| true))]
    }
}

impl AsyncElement for PlayerAudioSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn metadata(&self) -> ElementMetadata {
        PLAYER_AUDIO_SINK_METADATA
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if matches!(upstream_caps, Caps::Audio { format, .. } if format.bytes_per_sample().is_some())
        {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(pcm_caps_any_rate_and_channels(|_| true))
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::SystemView)
    }

    fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.intercept_caps(caps)?;
        let Caps::Audio { format, .. } = caps else {
            return Err(G2gError::CapsMismatch);
        };
        let params = audio_params_of(caps).ok_or(G2gError::CapsMismatch)?;
        self.pcm = Some((*format, params));
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

    fn adopt_position(&mut self, position: SinkPosition) -> Option<SinkPosition> {
        self.position = Some(position);
        None
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let host = self.host.as_ref().ok_or(G2gError::NotConfigured)?;
            let generation = host.state.preroll_generation();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let (format, params) = self.pcm.ok_or(G2gError::NotConfigured)?;
                    let pending = PendingPcm {
                        frame,
                        format,
                        params,
                    };
                    if !host.state.is_playing() && self.preroll.is_none() {
                        self.preroll = Some(pending);
                        return Ok(());
                    }
                    if let Some(preroll) = self.preroll.take() {
                        self.present(preroll, generation).await?;
                    }
                    self.present(pending, generation).await?;
                }
                PipelinePacket::Segment(segment) => self.pacer.set_segment(segment),
                PipelinePacket::Flush => {
                    self.preroll = None;
                    self.pacer.flush();
                    if let Some(position) = &self.position {
                        position.forget();
                    }
                    (host.callback.0)(AudioOutputEvent::Flush);
                }
                PipelinePacket::Eos => {
                    if let Some(preroll) = self.preroll.take() {
                        self.present(preroll, generation).await?;
                    }
                    let host = self.host.as_ref().ok_or(G2gError::NotConfigured)?;
                    if host.state.preroll_generation() == generation
                        && host.state.state() != PipelineState::Null
                    {
                        (host.callback.0)(AudioOutputEvent::EndOfStream);
                    }
                }
                _ => {}
            }
            Ok(())
        })
    }
}

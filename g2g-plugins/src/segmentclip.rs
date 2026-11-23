// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Clip raw audio or video to the latest time segment. Audio can lose samples
//! at either edge. Video frames keep their payload and shorten their duration.

use core::future::Future;
use core::num::NonZeroU32;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::caps::framerate_q16_period_ns;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::{
    pcm_caps_any_rate_and_channels, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, Rate, Segment,
};

const NANOS_PER_SECOND: u128 = core::time::Duration::from_secs(1).as_nanos();

/// Clip interleaved PCM to the active segment.
pub type AudioSegmentClip = SegmentClip<true>;

/// Clip raw video to the active segment.
pub type VideoSegmentClip = SegmentClip<false>;

#[derive(Debug, Clone, Copy)]
enum ClipFormat {
    Audio {
        bytes_per_frame: usize,
        sample_rate: NonZeroU32,
    },
    Video {
        frame_period_ns: Option<u64>,
    },
}

/// The `audiosegmentclip` and `videosegmentclip` transforms.
#[derive(Debug, Default)]
pub struct SegmentClip<const AUDIO: bool> {
    format: Option<ClipFormat>,
    segment: Option<Segment>,
}

impl<const AUDIO: bool> SegmentClip<AUDIO> {
    pub fn new() -> Self {
        Self::default()
    }

    fn caps() -> CapsSet {
        if AUDIO {
            pcm_caps_any_rate_and_channels(|_| true)
        } else {
            crate::fakemediasink::raw_video_caps()
        }
    }

    fn configure(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let format = match caps {
            Caps::Audio {
                format,
                channels,
                sample_rate,
                ..
            } if AUDIO
                && *channels > 0
                && crate::audioconvert::RAW_AUDIO_FORMATS.contains(format) =>
            {
                ClipFormat::Audio {
                    bytes_per_frame: format.bytes_per_sample().ok_or(G2gError::CapsMismatch)?
                        * usize::from(*channels),
                    sample_rate: NonZeroU32::new(*sample_rate).ok_or(G2gError::CapsMismatch)?,
                }
            }
            Caps::RawVideo { framerate, .. } if !AUDIO => ClipFormat::Video {
                frame_period_ns: match framerate {
                    Rate::Fixed(q16) => framerate_q16_period_ns(*q16),
                    _ => None,
                },
            },
            _ => return Err(G2gError::CapsMismatch),
        };
        self.format = Some(format);
        Ok(())
    }
}

// Return the time removed at each edge. A frame that fits keeps its original
// timing, even if its duration had to be inferred to test the overlap.
fn clip_timing(
    segment: Segment,
    timing: &mut FrameTiming,
    pts: u64,
    duration: u64,
) -> Option<(u64, u64)> {
    if duration == 0 {
        return segment.presents(pts).then_some((0, 0));
    }
    let end = pts.saturating_add(duration);
    let (start, stop) = segment.clip(pts, Some(end))?;
    let stop = stop?;
    let duration = stop.checked_sub(start).filter(|&duration| duration > 0)?;
    let trimmed = (start - pts, end - stop);
    if trimmed != (0, 0) {
        timing.pts_ns = start;
        timing.duration_ns = duration;
    }
    Some(trimmed)
}

fn clip_audio(
    mut frame: Frame,
    pts: u64,
    segment: Segment,
    bytes_per_frame: usize,
    sample_rate: NonZeroU32,
) -> Result<Option<Frame>, G2gError> {
    let pcm = frame.domain.require_system_slice("audiosegmentclip")?;
    let frames = pcm.len() / bytes_per_frame;
    if frames == 0 {
        return Ok(Some(frame));
    }
    let rate = u128::from(sample_rate.get());
    let duration = if frame.timing.duration_ns == 0 {
        (frames as u128 * NANOS_PER_SECOND / rate).min(u128::from(u64::MAX)) as u64
    } else {
        frame.timing.duration_ns
    };
    let Some((head_ns, tail_ns)) = clip_timing(segment, &mut frame.timing, pts, duration) else {
        return Ok(None);
    };
    // Only whole sample frames can be removed from either edge.
    let to_frames = |ns| (u128::from(ns) * rate / NANOS_PER_SECOND).min(frames as u128) as usize;
    let first = to_frames(head_ns);
    let end = frames - to_frames(tail_ns);
    if first >= end {
        return Ok(None);
    }
    if (first, end) != (0, frames) {
        let pcm = pcm[first * bytes_per_frame..end * bytes_per_frame].to_vec();
        frame.domain = MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice()));
        frame.timing.dts_ns = FrameTiming::DTS_NONE;
    }
    Ok(Some(frame))
}

impl<const AUDIO: bool> AsyncElement for SegmentClip<AUDIO> {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn metadata(&self) -> ElementMetadata {
        if AUDIO {
            ElementMetadata::new(
                "Audio segment clipper",
                "Filter/Audio",
                "Clips PCM samples to the active segment",
                crate::AARON_BOXER,
            )
        } else {
            ElementMetadata::new(
                "Video segment clipper",
                "Filter/Video",
                "Clips raw video timing to the active segment",
                crate::AARON_BOXER,
            )
        }
    }

    fn input_domains(&self) -> DomainSet {
        if AUDIO {
            DomainSet::only(MemoryDomainKind::System)
        } else {
            DomainSet::ALL
        }
    }

    fn output_domain_follows_input(&self) -> bool {
        !AUDIO
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::caps().accepts(upstream_caps) {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(Self::caps())
    }

    fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configure(caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let format = self.format.ok_or(G2gError::NotConfigured)?;
            let packet = match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    if let (Some(segment), Some(pts)) = (self.segment, frame.timing.pts()) {
                        match format {
                            ClipFormat::Audio {
                                bytes_per_frame,
                                sample_rate,
                            } => {
                                let Some(clipped) =
                                    clip_audio(frame, pts, segment, bytes_per_frame, sample_rate)?
                                else {
                                    return Ok(());
                                };
                                frame = clipped;
                            }
                            ClipFormat::Video { frame_period_ns } => {
                                let duration = if frame.timing.duration_ns == 0 {
                                    frame_period_ns.unwrap_or(0)
                                } else {
                                    frame.timing.duration_ns
                                };
                                if clip_timing(segment, &mut frame.timing, pts, duration).is_none()
                                {
                                    return Ok(());
                                }
                            }
                        }
                    }
                    PipelinePacket::DataFrame(frame)
                }
                PipelinePacket::Segment(segment) => {
                    self.segment = Some(segment);
                    PipelinePacket::Segment(segment)
                }
                PipelinePacket::Flush => {
                    self.segment = None;
                    PipelinePacket::Flush
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.configure(&caps)?;
                    PipelinePacket::CapsChanged(caps)
                }
                // The source owns segment playback and the runner forwards EOS.
                PipelinePacket::Eos => return Ok(()),
                other => other,
            };
            out.push(packet).await?;
            Ok(())
        })
    }
}

impl<const AUDIO: bool> PadTemplates for SegmentClip<AUDIO> {
    fn pad_templates() -> Vec<PadTemplate> {
        crate::pcm_passthrough::pad_templates(Self::caps())
    }
}

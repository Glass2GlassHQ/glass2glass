// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! The two media-typed fake sinks, gst's `fakevideosink` and `fakeaudiosink`:
//! [`FakeSink`]'s counting behaviour behind a pad that takes only raw video or
//! only raw PCM. Use one to terminate a branch whose decode you still want
//! negotiated and exercised, where a plain `fakesink` would happily swallow the
//! undecoded stream instead and hide the missing decoder.
//!
//! `silent=false` records a line per buffer in `last-message` and on the debug
//! log, as `fakesink` does.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    Interlace, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropValue,
    PropertySpec, Rate,
};

#[cfg(feature = "std")]
use g2g_core::progress::SinkPosition;
#[cfg(feature = "std")]
use g2g_core::{ClockCandidate, ClockPriority, ClockSync, MonotonicClock, PresentationPacer};

use crate::audioconvert::RAW_AUDIO_FORMATS;
use crate::fakesink::{FakeSink, FAKE_SINK_PROPS};

/// The channel count and rate an audio template leaves open (`Caps::Audio`'s own
/// wildcards).
const ANY_CHANNELS: u8 = 0;
const ANY_SAMPLE_RATE: u32 = 0;

/// Every raw video shape, for the sink's declared constraint and pad template.
pub(crate) fn raw_video_caps() -> CapsSet {
    CapsSet::from_alternatives(
        g2g_core::caps::ALL_RAW_VIDEO_FORMATS
            .iter()
            .map(|&format| Caps::RawVideo {
                format,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
                pixel_aspect: g2g_core::PixelAspect::Any,
            })
            .collect(),
    )
}

/// Every raw audio shape, for the sink's declared constraint and pad template.
fn raw_audio_caps() -> CapsSet {
    CapsSet::from_alternatives(
        RAW_AUDIO_FORMATS
            .iter()
            .map(|&format| Caps::Audio {
                format,
                channels: ANY_CHANNELS,
                sample_rate: ANY_SAMPLE_RATE,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            })
            .collect(),
    )
}

/// What the two sinks share: the counting `FakeSink` they delegate to, whose
/// `silent` / `last-message` pair is the only knob either applies.
#[derive(Debug, Default)]
struct CountingFake {
    inner: FakeSink,
    #[cfg(feature = "std")]
    playout: Option<FakePlayout>,
}

// gst's fakeaudiosink holds each buffer until its time, as the device it stands in for does
#[cfg(feature = "std")]
#[derive(Debug, Default)]
struct FakePlayout {
    pacer: PresentationPacer,
    position: Option<SinkPosition>,
}

impl CountingFake {
    fn new() -> Self {
        Self::default()
    }

    #[cfg(feature = "std")]
    fn paced() -> Self {
        Self {
            playout: Some(FakePlayout::default()),
            ..Self::new()
        }
    }

    #[cfg(not(feature = "std"))]
    fn paced() -> Self {
        Self::new()
    }

    #[cfg(feature = "std")]
    fn provide_clock(&self) -> Option<ClockCandidate> {
        self.playout.as_ref()?;
        // the wait for each buffer is on tokio's timer, which a pipeline run by g2g's block_on lacks
        tokio::runtime::Handle::try_current().ok()?;
        Some(ClockCandidate::new(
            ClockPriority::Provider,
            alloc::sync::Arc::new(MonotonicClock),
        ))
    }

    #[cfg(feature = "std")]
    fn set_clock_sync(&mut self, sync: ClockSync) {
        if let Some(playout) = &mut self.playout {
            playout.pacer.set_clock_sync(sync.clone());
        }
        AsyncElement::set_clock_sync(&mut self.inner, sync);
    }

    #[cfg(feature = "std")]
    fn adopt_position(&mut self, position: SinkPosition) -> Option<SinkPosition> {
        match &mut self.playout {
            Some(playout) if playout.pacer.is_paced() => {
                playout.position = Some(position);
                None
            }
            _ => Some(position),
        }
    }

    #[cfg(feature = "std")]
    async fn play(&mut self, packet: &PipelinePacket) {
        let received = self.inner.received();
        let Some(playout) = &mut self.playout else {
            return;
        };
        match packet {
            PipelinePacket::Segment(segment) => playout.pacer.set_segment(*segment),
            PipelinePacket::Flush => {
                playout.pacer.flush();
                if let Some(position) = &playout.position {
                    position.forget();
                }
            }
            PipelinePacket::DataFrame(frame) => {
                let pts_ns = frame.timing.pts_ns;
                crate::clock::wait_to_present(playout.pacer.judge(&frame.timing, received)).await;
                if let Some(position) = &playout.position {
                    position.set_at_pts(playout.pacer.segment().as_ref(), pts_ns);
                }
            }
            _ => {}
        }
    }
}

/// Both sinks do the same thing with a packet, differing only in what they
/// accept, so the element bodies are this one macro.
macro_rules! fake_media_sink {
    ($name:ident, $fake:path, $accepts:path, $caps:path, $classification:expr, $description:expr) => {
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $name {
            pub fn new() -> Self {
                Self { fake: $fake() }
            }

            /// Buffers received so far.
            pub fn received(&self) -> u64 {
                self.fake.inner.received()
            }

            /// Whether `Eos` has arrived.
            pub fn eos_seen(&self) -> bool {
                self.fake.inner.eos_seen()
            }

            /// The most recent buffer's line, empty while `silent`.
            pub fn last_message(&self) -> &str {
                self.fake.inner.last_message()
            }
        }

        impl AsyncElement for $name {
            type ProcessFuture<'a>
                = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
            where
                Self: 'a;

            fn metadata(&self) -> ElementMetadata {
                ElementMetadata::new(
                    stringify!($name),
                    $classification,
                    $description,
                    crate::AARON_BOXER,
                )
            }

            fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
                if !$accepts(upstream_caps) {
                    return Err(G2gError::CapsMismatch);
                }
                Ok(upstream_caps.clone())
            }

            fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
                CapsConstraint::Accepts($caps())
            }

            fn configure_pipeline(
                &mut self,
                absolute_caps: &Caps,
            ) -> Result<ConfigureOutcome, G2gError> {
                if !$accepts(absolute_caps) {
                    return Err(G2gError::CapsMismatch);
                }
                AsyncElement::configure_pipeline(&mut self.fake.inner, absolute_caps)
            }

            fn process<'a>(
                &'a mut self,
                packet: PipelinePacket,
                out: &'a mut dyn OutputSink,
            ) -> Self::ProcessFuture<'a> {
                Box::pin(async move {
                    #[cfg(feature = "std")]
                    self.fake.play(&packet).await;
                    AsyncElement::process(&mut self.fake.inner, packet, out).await
                })
            }

            #[cfg(feature = "std")]
            fn provide_clock(&self) -> Option<ClockCandidate> {
                self.fake.provide_clock()
            }

            #[cfg(feature = "std")]
            fn set_clock_sync(&mut self, sync: ClockSync) {
                self.fake.set_clock_sync(sync)
            }

            #[cfg(feature = "std")]
            fn adopt_position(&mut self, position: SinkPosition) -> Option<SinkPosition> {
                self.fake.adopt_position(position)
            }

            fn properties(&self) -> &'static [PropertySpec] {
                FAKE_SINK_PROPS
            }

            fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
                AsyncElement::set_property(&mut self.fake.inner, name, value)
            }

            fn get_property(&self, name: &str) -> Option<PropValue> {
                AsyncElement::get_property(&self.fake.inner, name)
            }
        }

        impl PadTemplates for $name {
            fn pad_templates() -> Vec<PadTemplate> {
                vec![PadTemplate::sink($caps())]
            }
        }
    };
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::fakemediasink::FakeVideoSink;
///
/// // gst-launch equivalent: fakevideosink silent=false
/// let sink = FakeVideoSink::new();
/// assert_eq!(sink.received(), 0);
/// ```
#[derive(Debug)]
pub struct FakeVideoSink {
    fake: CountingFake,
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::fakemediasink::FakeAudioSink;
///
/// // gst-launch equivalent: fakeaudiosink silent=false
/// let sink = FakeAudioSink::new();
/// assert_eq!(sink.received(), 0);
/// ```
#[derive(Debug)]
pub struct FakeAudioSink {
    fake: CountingFake,
}

/// Raw (decoded) video only: an undecoded stream is what this sink exists to
/// refuse.
fn is_raw_video(caps: &Caps) -> bool {
    matches!(caps, Caps::RawVideo { .. })
}

/// Raw PCM only. `Caps::is_raw_media` is what separates PCM from the encoded
/// formats `Caps::Audio` also carries.
fn is_raw_audio(caps: &Caps) -> bool {
    matches!(caps, Caps::Audio { .. }) && caps.is_raw_media()
}

fake_media_sink!(
    FakeVideoSink,
    CountingFake::new,
    is_raw_video,
    raw_video_caps,
    "Sink/Video",
    "Discards raw video buffers, counting them"
);
fake_media_sink!(
    FakeAudioSink,
    CountingFake::paced,
    is_raw_audio,
    raw_audio_caps,
    "Sink/Audio",
    "Discards raw audio buffers, counting them"
);

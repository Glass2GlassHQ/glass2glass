//! The two media-typed fake sinks, gst's `fakevideosink` and `fakeaudiosink`:
//! [`FakeSink`]'s counting behaviour behind a pad that takes only raw video or
//! only raw PCM. Use one to terminate a branch whose decode you still want
//! negotiated and exercised, where a plain `fakesink` would happily swallow the
//! undecoded stream instead and hide the missing decoder.
//!
//! `silent=false` records a `chain: <bytes> bytes, pts: <ns>` line per buffer in
//! `last-message` and on the debug log, which is what gst's fakesink prints when
//! its own `silent` is off.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_debug, AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    ElementMetadata, G2gError, Interlace, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

use crate::fakesink::FakeSink;

/// The raw video formats a `fakevideosink` pad advertises: every one g2g has, so
/// nothing negotiable is turned away.
const RAW_VIDEO_FORMATS: [RawVideoFormat; 15] = [
    RawVideoFormat::Nv12,
    RawVideoFormat::I420,
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Rgb8,
    RawVideoFormat::Yuyv,
    RawVideoFormat::I420p10,
    RawVideoFormat::I420p12,
    RawVideoFormat::I422,
    RawVideoFormat::I422p10,
    RawVideoFormat::I422p12,
    RawVideoFormat::I444,
    RawVideoFormat::I444p10,
    RawVideoFormat::I444p12,
    RawVideoFormat::P010,
];

/// The PCM formats a `fakeaudiosink` pad advertises. `Caps::Audio` also carries
/// the encoded formats, which is exactly what this sink refuses.
const RAW_AUDIO_FORMATS: [AudioFormat; 5] = [
    AudioFormat::PcmS16Le,
    AudioFormat::PcmF32Le,
    AudioFormat::PcmS24Le,
    AudioFormat::PcmS32Le,
    AudioFormat::PcmU8,
];

/// The channel count and rate an audio template leaves open (`Caps::Audio`'s own
/// wildcards, M187).
const ANY_CHANNELS: u8 = 0;
const ANY_SAMPLE_RATE: u32 = 0;

/// Every raw video shape, for the sink's declared constraint and pad template.
fn raw_video_caps() -> CapsSet {
    CapsSet::from_alternatives(
        RAW_VIDEO_FORMATS
            .iter()
            .map(|&format| Caps::RawVideo {
                format,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
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

/// What the two sinks share: the counting `FakeSink` they delegate to, plus the
/// `silent` / `last-message` pair that is the only knob either applies.
#[derive(Debug, Default)]
struct CountingFake {
    inner: FakeSink,
    silent: bool,
    last_message: String,
    log_name: LogName,
}

impl CountingFake {
    /// gst's `fakevideosink` / `fakeaudiosink` both default `silent` on.
    fn new() -> Self {
        Self {
            silent: true,
            ..Self::default()
        }
    }

    fn note(&mut self, packet: &PipelinePacket) {
        if self.silent {
            return;
        }
        if let PipelinePacket::DataFrame(frame) = packet {
            let bytes = frame.domain.as_system_slice().map_or(0, <[u8]>::len);
            self.last_message = format!("chain: {bytes} bytes, pts: {}", frame.timing.pts_ns);
            g2g_debug!(self, "{}", self.last_message);
        }
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "silent" => {
                self.silent = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "last-message" => Err(PropError::ReadOnly),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "silent" => Some(PropValue::Bool(self.silent)),
            "last-message" => Some(PropValue::Str(self.last_message.clone())),
            _ => None,
        }
    }
}

impl LogSource for CountingFake {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

/// The `silent` / `last-message` pair both sinks declare, named and defaulted as
/// gst's fake sinks.
static FAKE_MEDIA_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "silent",
        PropKind::Bool,
        "stop recording a line per buffer in last-message",
    )
    .with_default("true"),
    PropertySpec::new(
        "last-message",
        PropKind::Str,
        "the most recent buffer's line, empty while silent",
    )
    .read_only(),
];

/// Both sinks do the same thing with a packet, differing only in what they
/// accept, so the element bodies are this one macro.
macro_rules! fake_media_sink {
    ($name:ident, $accepts:path, $caps:path, $klass:expr, $description:expr) => {
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $name {
            pub fn new() -> Self {
                Self {
                    fake: CountingFake::new(),
                }
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
                &self.fake.last_message
            }
        }

        impl AsyncElement for $name {
            type ProcessFuture<'a>
                = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
            where
                Self: 'a;

            fn metadata(&self) -> ElementMetadata {
                ElementMetadata::new(stringify!($name), $klass, $description, "g2g")
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
                    self.fake.note(&packet);
                    AsyncElement::process(&mut self.fake.inner, packet, out).await
                })
            }

            fn properties(&self) -> &'static [PropertySpec] {
                FAKE_MEDIA_PROPS
            }

            fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
                self.fake.set_property(name, value)
            }

            fn get_property(&self, name: &str) -> Option<PropValue> {
                self.fake.get_property(name)
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
    is_raw_video,
    raw_video_caps,
    "Sink/Video",
    "Discards raw video buffers, counting them"
);
fake_media_sink!(
    FakeAudioSink,
    is_raw_audio,
    raw_audio_caps,
    "Sink/Audio",
    "Discards raw audio buffers, counting them"
);

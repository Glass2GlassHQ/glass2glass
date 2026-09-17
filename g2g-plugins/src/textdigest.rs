//! Incident digest (M1176, `no_std`): joins the text frames of a time window
//! into one text frame per window, the gst-python-ml `pyml_digest` analog. A
//! caption or detection-summary stream becomes one paragraph a minute, which is
//! what a language model or a log reader wants to be handed.
//!
//! Each input becomes the line `At <seconds>s: <text>`. The first frame at or
//! past the end of the open window closes it: the collected lines go out as one
//! frame stamped with the window's start and its length, and the new window
//! starts at that frame. Inputs are consumed, so nothing but digests leaves the
//! element. The open window is flushed at EOS, and dropped on a flush.

use core::fmt::Write as _;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming,
    G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, TextFormat,
};

const DEFAULT_WINDOW_SECONDS: f64 = 60.0;
const DEFAULT_WINDOW_SECONDS_TEXT: &str = "60.0";
const NS_PER_SECOND: f64 = 1_000_000_000.0;
/// What separates two entries of one digest.
const DIGEST_SEPARATOR: &str = "\n";

/// Joins the text frames of a time window into one digest frame.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::textdigest::TextDigest;
///
/// // gst-launch equivalent: textdigest window-seconds=10.0
/// let digest = TextDigest::new().with_window_seconds(10.0);
/// ```
#[derive(Debug)]
pub struct TextDigest {
    window_seconds: f64,
    lines: Vec<String>,
    /// The start of the open window in seconds, `None` before the first frame.
    window_start: Option<f64>,
    sequence: u64,
    caps_emitted: bool,
    configured: bool,
}

impl Default for TextDigest {
    fn default() -> Self {
        Self::new()
    }
}

impl TextDigest {
    pub fn new() -> Self {
        Self {
            window_seconds: DEFAULT_WINDOW_SECONDS,
            lines: Vec::new(),
            window_start: None,
            sequence: 0,
            caps_emitted: false,
            configured: false,
        }
    }

    /// How long a window each digest covers (the `window-seconds` property).
    pub fn with_window_seconds(mut self, seconds: f64) -> Self {
        self.window_seconds = seconds;
        self
    }

    pub(crate) fn text_caps() -> Caps {
        Caps::Text {
            format: TextFormat::Utf8,
        }
    }

    /// The digest of the open window, and the window it covered.
    fn take_window(&mut self) -> Option<(String, f64)> {
        if self.lines.is_empty() {
            return None;
        }
        let start = self.window_start.unwrap_or(0.0);
        let digest = self.lines.join(DIGEST_SEPARATOR);
        self.lines.clear();
        Some((digest, start))
    }

    async fn push_window(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let Some((digest, start)) = self.take_window() else {
            return Ok(());
        };
        if !self.caps_emitted {
            out.push(PipelinePacket::CapsChanged(Self::text_caps()))
                .await?;
            self.caps_emitted = true;
        }
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(
                digest.into_bytes().into_boxed_slice(),
            )),
            FrameTiming {
                pts_ns: (start * NS_PER_SECOND) as u64,
                duration_ns: (self.window_seconds * NS_PER_SECOND) as u64,
                ..FrameTiming::default()
            },
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await.map(|_| ())
    }

    /// Add one input frame to the open window, closing it first when the frame
    /// belongs to the next one.
    async fn collect(&mut self, frame: &Frame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let bytes = frame
            .domain
            .as_system_slice()
            .ok_or(G2gError::UnsupportedDomain)?;
        let text = core::str::from_utf8(bytes).unwrap_or_default();
        let seconds = frame
            .timing
            .pts()
            .map_or(0.0, |pts_ns| pts_ns as f64 / NS_PER_SECOND);
        let start = *self.window_start.get_or_insert(seconds);
        if seconds >= start + self.window_seconds {
            self.push_window(out).await?;
            self.window_start = Some(seconds);
        }
        let mut line = String::new();
        let _ = write!(line, "At {seconds:.1}s: {}", text.trim());
        self.lines.push(line);
        Ok(())
    }
}

impl AsyncElement for TextDigest {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Incident digest",
            "Filter/Text",
            "Joins the text frames of a time window into one digest per window",
            "g2g",
        )
    }

    /// Reads the text out of host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if *upstream_caps == Self::text_caps() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if *input == TextDigest::text_caps() {
                CapsSet::one(TextDigest::text_caps())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if *absolute_caps != Self::text_caps() {
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
                PipelinePacket::DataFrame(frame) => self.collect(&frame, out).await,
                // The element's own output caps, which must be forwarded and
                // suppress the emission before the first digest.
                PipelinePacket::CapsChanged(caps) => {
                    if caps != Self::text_caps() {
                        return Ok(());
                    }
                    self.caps_emitted = true;
                    out.push(PipelinePacket::CapsChanged(caps))
                        .await
                        .map(|_| ())
                }
                // The runner's transform arm forwards the sentinel after this
                // returns, so the last window still gets out in front of it.
                PipelinePacket::Eos => self.push_window(out).await,
                PipelinePacket::Flush => {
                    self.lines.clear();
                    self.window_start = None;
                    out.push(PipelinePacket::Flush).await.map(|_| ())
                }
                other => out.push(other).await.map(|_| ()),
            }
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        TEXTDIGEST_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "window-seconds" => {
                self.window_seconds = value.as_double().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "window-seconds" => Some(PropValue::Double(self.window_seconds)),
            _ => None,
        }
    }
}

impl PadTemplates for TextDigest {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::text_caps())),
            PadTemplate::source(CapsSet::one(Self::text_caps())),
        ])
    }
}

/// `TextDigest`'s settable properties: the length of one window.
static TEXTDIGEST_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "window-seconds",
    PropKind::Double,
    "length of the time window each digest covers",
)
.with_default(DEFAULT_WINDOW_SECONDS_TEXT)];

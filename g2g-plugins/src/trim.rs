//! Trim (M1176, `no_std`): keeps the frames in a presentation-time range and
//! rebases them onto it, so `filesrc ! decodebin ! trim start=… stop=… ! …` cuts
//! a clip out of a decoded file without a seek.
//!
//! A kept frame's pts and dts come out shifted back by `start`, so the clip
//! begins at zero. The first frame at or past `stop` ends the stream: the element
//! pushes `Eos` and drops everything behind it, so a sink downstream closes its
//! file at `stop` rather than at the source's end.
//!
//! A frame with no presentation time cannot be placed in the range, so it is
//! forwarded untouched.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, FrameTiming, G2gError,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

const DEFAULT_START_NS: u64 = 0;
const DEFAULT_START_TEXT: &str = "0";
const DEFAULT_STOP_NS: u64 = u64::MAX;

/// Keeps the frames between `start` and `stop`, rebased onto `start`.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::trim::Trim;
///
/// // gst-launch equivalent: trim start=1000000000 stop=3000000000
/// let trim = Trim::new().with_range(1_000_000_000, 3_000_000_000);
/// assert_eq!(trim.kept(), 0);
/// ```
#[derive(Debug)]
pub struct Trim {
    start_ns: u64,
    stop_ns: u64,
    kept: u64,
    ended: bool,
    configured: bool,
}

impl Default for Trim {
    fn default() -> Self {
        Self::new()
    }
}

impl Trim {
    pub fn new() -> Self {
        Self {
            start_ns: DEFAULT_START_NS,
            stop_ns: DEFAULT_STOP_NS,
            kept: 0,
            ended: false,
            configured: false,
        }
    }

    /// The range to keep, in nanoseconds (the `start` and `stop` properties).
    pub fn with_range(mut self, start_ns: u64, stop_ns: u64) -> Self {
        self.start_ns = start_ns;
        self.stop_ns = stop_ns;
        self
    }

    /// Frames forwarded so far.
    pub fn kept(&self) -> u64 {
        self.kept
    }

    /// The timing a kept frame leaves with: the range's start becomes zero.
    fn rebased(&self, timing: FrameTiming) -> FrameTiming {
        FrameTiming {
            pts_ns: timing.pts_ns.saturating_sub(self.start_ns),
            dts_ns: timing.dts_ns.saturating_sub(self.start_ns),
            ..timing
        }
    }
}

impl AsyncElement for Trim {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Trim",
            "Filter/Generic",
            "Keeps the frames of a presentation-time range, rebased onto its start",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Whatever crosses the element comes out unchanged, so the range applies to
    /// any media type.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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
            if self.ended {
                return Ok(());
            }
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    let Some(pts_ns) = frame.timing.pts() else {
                        self.kept += 1;
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        return Ok(());
                    };
                    if pts_ns >= self.stop_ns {
                        self.ended = true;
                        out.push(PipelinePacket::Eos).await?;
                        return Ok(());
                    }
                    if pts_ns < self.start_ns {
                        return Ok(());
                    }
                    frame.timing = self.rebased(frame.timing);
                    self.kept += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // The runner's transform arm forwards EOS; don't double it.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        TRIM_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "start" => {
                self.start_ns = value.as_uint().ok_or(PropError::Type)?;
                Ok(())
            }
            "stop" => {
                self.stop_ns = value.as_uint().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "start" => Some(PropValue::Uint(self.start_ns)),
            "stop" => Some(PropValue::Uint(self.stop_ns)),
            _ => None,
        }
    }
}

/// `Trim`'s settable properties: the range's ends in nanoseconds.
static TRIM_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "start",
        PropKind::Uint,
        "first presentation time kept, in nanoseconds",
    )
    .with_default(DEFAULT_START_TEXT),
    PropertySpec::new(
        "stop",
        PropKind::Uint,
        "presentation time the stream ends at, in nanoseconds",
    ),
];

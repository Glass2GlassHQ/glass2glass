//! Still-frame stream generator (`imagefreeze`). Holds the first frame it sees
//! and emits copies of it on a fixed framerate grid, turning one decoded image
//! into a video stream. Format and geometry pass through; only the framerate is
//! replaced. Pixels are never touched, so any memory domain rides through
//! (the copies share the buffer handle rather than the bytes).
//!
//! The rate comes from the `framerate` property (25/1 by default): there is no
//! downstream-preferred framerate to inherit, so the output rate is stated
//! rather than negotiated. `num-buffers` bounds the run; unlimited (-1, gst's
//! default) loops inside `process` until the output push fails or the future is
//! cancelled, which is how a shutdown stops it.
//!
//! gst's `allow-replace` and `is-live` are not exposed: the whole run happens
//! inside the first frame's `process` call, so there is no later input frame to
//! swap in, and pacing to a clock belongs to the sink.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsTransform, ConfigureOutcome, Dim, ElementMetadata,
    FieldTransform, FrameTiming, G2gError, Interlace, OutputSink, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, RawVideoFormat, RawVideoShape,
};

use crate::compositor::frame_period_ns;
use crate::numbuffers::{get_num_buffers, set_num_buffers};

/// gst `videotestsrc`-style still rate: the output framerate when the property
/// is left alone.
const DEFAULT_FRAMERATE: (u32, u32) = (25, 1);

/// The same fraction as declared text, for `gst-inspect`.
const DEFAULT_FRAMERATE_TEXT: &str = "25/1";

/// # Example
///
/// ```no_run
/// use g2g_plugins::imagefreeze::ImageFreeze;
///
/// let freeze = ImageFreeze::new().with_framerate(30, 1).with_num_buffers(10);
/// ```
#[derive(Debug)]
pub struct ImageFreeze {
    framerate: (u32, u32),
    /// Frames to emit before the run ends; `u64::MAX` is gst's `-1`
    /// (unlimited).
    num_buffers: u64,
    input: Option<(RawVideoFormat, u32, u32)>,
    configured: bool,
    caps_sent: bool,
    /// Set once a frame has been taken as the still: every later input frame is
    /// ignored.
    frozen: bool,
    emitted: u64,
}

impl Default for ImageFreeze {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageFreeze {
    /// 25 fps, unlimited output.
    pub fn new() -> Self {
        Self {
            framerate: DEFAULT_FRAMERATE,
            num_buffers: u64::MAX,
            input: None,
            configured: false,
            caps_sent: false,
            frozen: false,
            emitted: 0,
        }
    }

    pub fn with_framerate(mut self, numerator: u32, denominator: u32) -> Self {
        if denominator > 0 {
            self.framerate = (numerator, denominator);
        }
        self
    }

    /// `count` frames then stop; `u64::MAX` runs until the pipeline stops it.
    pub fn with_num_buffers(mut self, count: u64) -> Self {
        self.num_buffers = count;
        self
    }

    /// The output framerate in the Q16 fixed-point fps `Rate` carries.
    fn rate_q16(&self) -> u32 {
        let (numerator, denominator) = self.framerate;
        if denominator == 0 {
            return 0;
        }
        u32::try_from((u64::from(numerator) << 16) / u64::from(denominator)).unwrap_or(u32::MAX)
    }

    fn accept_input(&self, caps: &Caps) -> Result<(RawVideoFormat, u32, u32), G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if *width == 0 || *height == 0 {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *width, *height))
    }

    fn output_caps(&self, format: RawVideoFormat, width: u32, height: u32) -> Caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate: Rate::Fixed(self.rate_q16()),
            interlace: Interlace::Any,
        }
    }
}

impl AsyncElement for ImageFreeze {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Still frame stream generator",
            "Filter/Video",
            "Repeats the first frame at a fixed framerate",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // input side: any raw video at the upstream geometry and rate.
        match upstream_caps {
            Caps::RawVideo { .. } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Format and geometry pass through, the framerate becomes the configured
    /// one, so a downstream geometry pin still couples back through the freeze.
    /// A zero rate declares no output shape at all, so the solve fails loud.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let rate_q16 = self.rate_q16();
        let shapes = if rate_q16 > 0 {
            vec![RawVideoShape::PASSTHROUGH
                .with_framerate(FieldTransform::Fixed(Rate::Fixed(rate_q16)))]
        } else {
            Vec::new()
        };
        CapsConstraint::DerivedFields(CapsTransform::RawVideo {
            accept: Vec::new(),
            produce: Vec::new(),
            shapes,
        })
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.rate_q16() == 0 {
            return Err(G2gError::CapsMismatch);
        }
        self.input = Some(self.accept_input(absolute_caps)?);
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
                PipelinePacket::DataFrame(frame) => {
                    // the still is already chosen: later frames have nowhere to go.
                    if self.frozen {
                        return Ok(());
                    }
                    let (format, width, height) = self.input.ok_or(G2gError::NotConfigured)?;
                    self.frozen = true;
                    let mut domain = frame.domain;
                    // refcount the buffer once so each copy is a handle share,
                    // not a deep copy of the pixels.
                    domain.make_shareable();
                    if !self.caps_sent {
                        let caps = self.output_caps(format, width, height);
                        out.push(PipelinePacket::CapsChanged(caps)).await?;
                        self.caps_sent = true;
                    }
                    let period_ns = frame_period_ns(self.rate_q16());
                    while self.emitted < self.num_buffers {
                        let pts = self.emitted.saturating_mul(period_ns);
                        let timing = FrameTiming {
                            pts_ns: pts,
                            dts_ns: pts,
                            duration_ns: period_ns,
                            capture_ns: frame.timing.capture_ns,
                            arrival_ns: frame.timing.arrival_ns,
                            keyframe: frame.timing.keyframe,
                        };
                        let copy = Frame::new(domain.share(), timing, self.emitted);
                        self.emitted += 1;
                        out.push(PipelinePacket::DataFrame(copy)).await?;
                    }
                }
                PipelinePacket::CapsChanged(c) => {
                    self.input = Some(self.accept_input(&c)?);
                }
                PipelinePacket::Flush => {
                    self.frozen = false;
                    self.caps_sent = false;
                    self.emitted = 0;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // the transform arm forwards EOS.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        IMAGEFREEZE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "framerate" => {
                let (numerator, denominator) = value.as_fraction().ok_or(PropError::Type)?;
                if numerator <= 0 || denominator <= 0 {
                    return Err(PropError::Value);
                }
                self.framerate = (numerator as u32, denominator as u32);
            }
            "num-buffers" => set_num_buffers(&mut self.num_buffers, &value)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "framerate" => Some(PropValue::Fraction(
                self.framerate.0 as i32,
                self.framerate.1 as i32,
            )),
            "num-buffers" => Some(get_num_buffers(self.num_buffers)),
            _ => None,
        }
    }
}

/// `ImageFreeze`'s properties (M1067).
static IMAGEFREEZE_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "framerate",
        PropKind::Fraction,
        "output frames per second (e.g. 30/1)",
    )
    .with_default(DEFAULT_FRAMERATE_TEXT),
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "frames to emit before stopping (-1 = unlimited)",
    )
    .with_default("-1"),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(framerate: Rate) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(64),
            height: Dim::Fixed(32),
            framerate,
            interlace: Interlace::Any,
        }
    }

    #[test]
    fn declared_defaults_match_the_constructor() {
        let element = ImageFreeze::new();
        assert_eq!(element.framerate, DEFAULT_FRAMERATE);
        assert_eq!(
            element.get_property("num-buffers"),
            Some(PropValue::Int(-1)),
            "unlimited by default, like gst"
        );
        let framerate = IMAGEFREEZE_PROPS
            .iter()
            .find(|s| s.name == "framerate")
            .expect("framerate is declared");
        assert_eq!(framerate.default, Some(DEFAULT_FRAMERATE_TEXT));
        assert_eq!(
            DEFAULT_FRAMERATE_TEXT, "25/1",
            "the declared text is the pair"
        );
    }

    #[test]
    fn framerate_property_sets_the_output_rate() {
        let mut element = ImageFreeze::new();
        element
            .set_property("framerate", PropValue::Fraction(30000, 1001))
            .unwrap();
        assert_eq!(
            element.get_property("framerate"),
            Some(PropValue::Fraction(30000, 1001)),
            "the fraction round-trips exactly, not through Q16"
        );
        let Caps::RawVideo {
            framerate: Rate::Fixed(q16),
            ..
        } = element.output_caps(RawVideoFormat::Rgba8, 64, 32)
        else {
            panic!("raw video out");
        };
        assert_eq!(q16, ((30000u64 << 16) / 1001) as u32);
        // a zero or negative fraction is refused, never silently kept.
        assert_eq!(
            element.set_property("framerate", PropValue::Fraction(30, 0)),
            Err(PropError::Value)
        );
    }

    #[test]
    fn configure_rejects_compressed_input() {
        let mut element = ImageFreeze::new();
        let h264 = Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Fixed(64),
            height: Dim::Fixed(32),
            framerate: Rate::Any,
        };
        assert_eq!(
            element.configure_pipeline(&h264).unwrap_err(),
            G2gError::CapsMismatch
        );
        assert!(element.configure_pipeline(&caps(Rate::Any)).is_ok());
    }

    #[test]
    fn derived_output_replaces_only_the_framerate() {
        let element = ImageFreeze::new().with_framerate(30, 1);
        let CapsConstraint::DerivedFields(transform) = element.caps_constraint_as_transform()
        else {
            panic!("expected DerivedFields");
        };
        let out = transform.derive(&caps(Rate::Fixed(10 << 16)));
        assert_eq!(out.alternatives(), &[caps(Rate::Fixed(30 << 16))]);
    }
}

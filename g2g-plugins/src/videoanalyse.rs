//! Frame luma meter (`videoanalyse`). A passthrough that measures the average
//! brightness and its variance of each frame. I420 (the Y plane) and packed
//! RGBA / BGRA (BT.709 luma) so `videotestsrc ! videoanalyse` negotiates
//! without a convert. `message=false` skips the measurement.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, Rate, RawVideoFormat,
};

use crate::pixel::{frame_byte_size, luma_at};

const FORMATS: [RawVideoFormat; 3] = [
    RawVideoFormat::I420,
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
];

/// # Example
///
/// ```no_run
/// use g2g_plugins::videoanalyse::VideoAnalyse;
///
/// let analyse = VideoAnalyse::new();
/// ```
#[derive(Debug)]
pub struct VideoAnalyse {
    message: bool,
    luma_average: f64,
    luma_variance: f64,
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    configured: bool,
}

impl Default for VideoAnalyse {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoAnalyse {
    pub fn new() -> Self {
        Self {
            message: true,
            luma_average: 0.0,
            luma_variance: 0.0,
            input: None,
            configured: false,
        }
    }

    /// Mean brightness of the last measured frame, 0..1.
    pub fn luma_average(&self) -> f64 {
        self.luma_average
    }

    /// Brightness variance of the last measured frame.
    pub fn luma_variance(&self) -> f64 {
        self.luma_variance
    }

    fn accept(caps: &Caps) -> Result<(RawVideoFormat, u32, u32, Rate), G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate,
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !FORMATS.contains(format) || *w == 0 || *h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *w, *h, framerate.clone()))
    }

    fn measure(&mut self, format: RawVideoFormat, w: u32, h: u32, src: &[u8]) {
        let n = (w as u64) * (h as u64);
        if n == 0 {
            return;
        }
        let mut sum = 0.0f64;
        for y in 0..h {
            for x in 0..w {
                sum += luma_at(format, w, src, x, y) as f64;
            }
        }
        let mean = (sum / n as f64) / 255.0;
        let mut var = 0.0f64;
        for y in 0..h {
            for x in 0..w {
                let v = luma_at(format, w, src, x, y) as f64 / 255.0 - mean;
                var += v * v;
            }
        }
        self.luma_average = mean;
        self.luma_variance = var / n as f64;
    }
}

impl AsyncElement for VideoAnalyse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Self::accept(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo { format, .. } if FORMATS.contains(format) => {
                CapsSet::one(input.clone())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.input = Some(Self::accept(absolute_caps)?);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Video analyser",
            "Filter/Analyzer/Video",
            "Analyses video for luma average and variance",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if self.message {
                        if let Some((format, w, h, _)) = self.input.clone() {
                            if let Some(src) = frame.domain.as_system_slice() {
                                if src.len() >= frame_byte_size(format, w, h) {
                                    self.measure(format, w, h, src);
                                }
                            }
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    if let Ok(input) = Self::accept(&c) {
                        self.input = Some(input);
                    }
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        ANALYSE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "message" => self.message = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "message" => Some(PropValue::Bool(self.message)),
            _ => None,
        }
    }
}

static ANALYSE_PROPS: &[PropertySpec] =
    &[PropertySpec::new("message", PropKind::Bool, "Post statics messages").with_default("true")];

impl PadTemplates for VideoAnalyse {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        crate::videofx::pad_templates_for(&FORMATS)
    }
}

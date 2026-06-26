//! `PySource`: a Python frame source hosted as a g2g [`SourceLoop`].
//!
//! Each tick hands the Python element a blank, writable buffer-protocol frame
//! via `g2g_produce(buf, width, height, fmt, meta) -> bool`; the element fills
//! it in place and returns `True`, or returns `False` to end the stream. Output
//! caps are fixed by the source's properties, so negotiation is synchronous
//! (`intercept_caps` returns them with no I/O). The same GIL-owning worker
//! thread as the other hosts runs the calls.

use core::future::{Future, Ready};
use core::pin::Pin;

use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

use crate::format::{format_from_py, format_to_py, frame_bytes};

/// A Python frame source hosted as a first-class g2g source.
#[derive(Debug)]
pub struct PySource {
    module: String,
    class: String,
    /// Fixed output caps (concrete format / dims / rate).
    caps: Caps,
    /// Optional cap on frames produced; `None` runs until Python signals EOS.
    num_buffers: Option<u64>,
    configured: bool,
    emitted: u64,
    #[cfg(feature = "python")]
    worker: Option<crate::host::PyWorker>,
}

impl PySource {
    /// Host `class` from Python `module` as a frame source. Default caps are
    /// RGBA 320x240 @ 30; override with [`with_caps`](Self::with_caps) or the
    /// `format` / `width` / `height` / `framerate` properties.
    pub fn new(module: impl Into<String>, class: impl Into<String>) -> Self {
        Self {
            module: module.into(),
            class: class.into(),
            caps: Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(320),
                height: Dim::Fixed(240),
                framerate: Rate::Fixed(30),
            },
            num_buffers: None,
            configured: false,
            emitted: 0,
            #[cfg(feature = "python")]
            worker: None,
        }
    }

    /// Set the fixed output caps (must be fully fixed: concrete format/dims/rate).
    pub fn with_caps(mut self, caps: Caps) -> Self {
        self.caps = caps;
        self
    }

    /// Stop after `n` frames (otherwise run until the Python source returns EOS).
    pub fn with_num_buffers(mut self, n: u64) -> Self {
        self.num_buffers = Some(n);
        self
    }

    /// Count of frames pushed downstream. Useful in tests.
    pub fn emitted_count(&self) -> u64 {
        self.emitted
    }

    /// Mutate the RawVideo caps fields in place (no-op on non-RawVideo caps).
    fn edit_caps(&mut self, f: impl FnOnce(&mut RawVideoFormat, &mut Dim, &mut Dim, &mut Rate)) {
        if let Caps::RawVideo { format, width, height, framerate } = &mut self.caps {
            f(format, width, height, framerate);
        }
    }

    #[cfg(feature = "python")]
    async fn produce_one(&self, frame: Frame) -> Result<Option<Frame>, G2gError> {
        self.worker.as_ref().ok_or(G2gError::NotConfigured)?.run_produce(frame, &self.caps).await
    }

    #[cfg(not(feature = "python"))]
    async fn produce_one(&self, _frame: Frame) -> Result<Option<Frame>, G2gError> {
        Err(G2gError::UnsupportedDomain)
    }

    /// Allocate a zeroed frame of the output geometry, stamped at `seq` /
    /// `pts_step_ns`, for the Python source to fill.
    fn blank_frame(&self, seq: u64, pts_step_ns: u64) -> Result<Frame, G2gError> {
        let Caps::RawVideo { format, width: Dim::Fixed(w), height: Dim::Fixed(h), .. } = &self.caps
        else {
            return Err(G2gError::FixationFailed);
        };
        let bytes = vec![0u8; frame_bytes(*format, *w, *h)].into_boxed_slice();
        let pts = seq.saturating_mul(pts_step_ns);
        Ok(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes)),
            timing: FrameTiming {
                pts_ns: pts,
                dts_ns: pts,
                duration_ns: pts_step_ns,
                capture_ns: pts,
                arrival_ns: 0,
                keyframe: false,
            },
            sequence: seq,
            meta: Default::default(),
        })
    }
}

/// Per-frame PTS step from a fixed framerate, or 0 when the rate is not fixed.
fn pts_step_ns(caps: &Caps) -> u64 {
    match caps {
        Caps::RawVideo { framerate: Rate::Fixed(fps), .. } if *fps > 0 => {
            1_000_000_000u64 / u64::from(*fps)
        }
        _ => 0,
    }
}

impl SourceLoop for PySource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps.clone()))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        absolute_caps.intersect(&self.caps)?;
        self.caps = absolute_caps.clone();
        #[cfg(feature = "python")]
        {
            if self.module.is_empty() || self.class.is_empty() {
                return Err(G2gError::NotConfigured);
            }
            if self.worker.is_none() {
                // Property forwarding to the hosted source instance is a
                // follow-up; transforms (PyTransform) forward theirs today.
                self.worker =
                    Some(crate::host::PyWorker::spawn(&self.module, &self.class, false, &[])?);
            }
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let step = pts_step_ns(&self.caps);
            let mut produced = 0u64;
            loop {
                if let Some(limit) = self.num_buffers {
                    if produced >= limit {
                        break;
                    }
                }
                let blank = self.blank_frame(produced, step)?;
                match self.produce_one(blank).await? {
                    Some(frame) => {
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        produced += 1;
                    }
                    // Python source signalled end of stream.
                    None => break,
                }
            }
            self.emitted = produced;
            out.push(PipelinePacket::Eos).await?;
            Ok(produced)
        })
    }

    /// The output caps are property-fixed, so the auto-plug parser can read them
    /// without negotiation.
    fn configured_output_caps(&self) -> Option<Caps> {
        Some(self.caps.clone())
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PYSOURCE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "module" => {
                self.module = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "class" => {
                self.class = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "num-buffers" => {
                let n = value.as_int().ok_or(PropError::Type)?;
                self.num_buffers = if n < 0 { None } else { Some(n as u64) };
                Ok(())
            }
            "format" => {
                let f = format_from_py(value.as_str().ok_or(PropError::Type)?)
                    .ok_or(PropError::Value)?;
                self.edit_caps(|format, _, _, _| *format = f);
                Ok(())
            }
            "width" => {
                let w = value.as_uint().ok_or(PropError::Type)? as u32;
                self.edit_caps(|_, width, _, _| *width = Dim::Fixed(w));
                Ok(())
            }
            "height" => {
                let h = value.as_uint().ok_or(PropError::Type)? as u32;
                self.edit_caps(|_, _, height, _| *height = Dim::Fixed(h));
                Ok(())
            }
            "framerate" => {
                let (n, d) = value.as_fraction().ok_or(PropError::Type)?;
                if d <= 0 || n <= 0 {
                    return Err(PropError::Value);
                }
                let fps = (n / d) as u32;
                self.edit_caps(|_, _, _, rate| *rate = Rate::Fixed(fps));
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let raw = match &self.caps {
            Caps::RawVideo { format, width, height, framerate } => {
                Some((format, width, height, framerate))
            }
            _ => None,
        };
        match name {
            "module" => Some(PropValue::Str(self.module.clone())),
            "class" => Some(PropValue::Str(self.class.clone())),
            "num-buffers" => Some(PropValue::Int(self.num_buffers.map_or(-1, |n| n as i64))),
            "format" => raw.map(|(f, _, _, _)| PropValue::Str(format_to_py(*f).to_string())),
            "width" => raw.and_then(|(_, w, _, _)| match w {
                Dim::Fixed(v) => Some(PropValue::Uint(u64::from(*v))),
                _ => None,
            }),
            "height" => raw.and_then(|(_, _, h, _)| match h {
                Dim::Fixed(v) => Some(PropValue::Uint(u64::from(*v))),
                _ => None,
            }),
            "framerate" => raw.and_then(|(_, _, _, r)| match r {
                Rate::Fixed(fps) => Some(PropValue::Fraction(*fps as i32, 1)),
                _ => None,
            }),
            _ => None,
        }
    }
}

/// `PySource`'s settable properties (the runtime / `gst-launch` face).
static PYSOURCE_PROPS: &[PropertySpec] = &[
    PropertySpec::new("module", PropKind::Str, "Python module to import (the source element)"),
    PropertySpec::new("class", PropKind::Str, "class within the module to instantiate"),
    PropertySpec::new("format", PropKind::Str, "output pixel format (RGBA | BGRA | NV12 | I420 | YUY2)")
        .with_default("RGBA"),
    PropertySpec::new("width", PropKind::Uint, "output width in pixels").with_default("320"),
    PropertySpec::new("height", PropKind::Uint, "output height in pixels").with_default("240"),
    PropertySpec::new("framerate", PropKind::Fraction, "output framerate").with_default("30/1"),
    PropertySpec::new("num-buffers", PropKind::Int, "frames to produce, or -1 until EOS")
        .with_default("-1"),
];

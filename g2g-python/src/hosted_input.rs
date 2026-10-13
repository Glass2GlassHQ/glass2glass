use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{
    AllocationParams, Caps, Dim, Frame, G2gError, PropError, PropKind, PropValue, PropertySpec,
    Rate, RawVideoFormat,
};

#[cfg(feature = "analytics")]
use g2g_core::{BatchMeta, BatchStream};

use crate::format::frame_bytes;
use crate::props::fixed_caps;

pub(crate) const MODULE_PROPERTY: PropertySpec = PropertySpec::new(
    "module",
    PropKind::Str,
    "Python module to import (the element shell)",
);
pub(crate) const CLASS_PROPERTY: PropertySpec = PropertySpec::new(
    "class",
    PropKind::Str,
    "class within the module to instantiate",
);
pub(crate) const CUDA_FRAMES_PROPERTY: PropertySpec = PropertySpec::new(
    "cuda-frames",
    PropKind::Bool,
    "host an element that reads GPU-resident CUDA frames (needs g2g_process_cuda, NV12 / P010)",
)
.with_default("false");
pub(crate) const INPUT_CAPS_PROPERTY: PropertySpec = PropertySpec::new(
    "input-caps",
    PropKind::Str,
    "caps accepted on the sink pad, e.g. audio/x-raw,format=S16LE,rate=16000",
)
.with_default("video/x-raw,format=RGBA");

pub(crate) fn default_accept() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

// a batch is accepted when each of its streams would be on its own
pub(crate) fn intersect_accepted(caps: &Caps, accept: &Caps) -> Result<Caps, G2gError> {
    #[cfg(feature = "analytics")]
    if let Caps::Batch { streams } = caps {
        let streams = streams
            .iter()
            .map(|stream| stream.intersect(accept))
            .collect::<Result<_, _>>()?;
        return Ok(Caps::Batch { streams });
    }
    caps.intersect(accept)
}

#[cfg(feature = "analytics")]
async fn run_stream_batch(
    worker: &crate::host::PyWorker,
    mut carrier: Frame,
) -> Result<Frame, G2gError> {
    let batch = carrier
        .meta
        .get_mut::<BatchMeta>()
        .ok_or(G2gError::CapsMismatch)?;
    let caps = shared_stream_caps(&batch.streams)?;
    let frames = batch
        .streams
        .iter_mut()
        .map(|stream| {
            stream.frame.take().ok_or(G2gError::BatchFrameTaken {
                index: stream.index,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let frames = worker.run_stream_batch(frames, &caps).await?;
    for (stream, frame) in batch.streams.iter_mut().zip(frames) {
        stream.frame = Some(frame);
    }
    Ok(carrier)
}

// `g2g_process_batch` takes one width, height and format for every buffer
#[cfg(feature = "analytics")]
fn shared_stream_caps(streams: &[BatchStream]) -> Result<Caps, G2gError> {
    let picture = |caps: &Caps| match caps {
        Caps::RawVideo {
            format,
            width,
            height,
            ..
        } => Some((*format, width.clone(), height.clone())),
        _ => None,
    };
    let first = streams.first().ok_or(G2gError::CapsMismatch)?;
    let first_picture = picture(&first.caps);
    if let Some(differing) = streams
        .iter()
        .find(|stream| picture(&stream.caps) != first_picture)
    {
        g2g_core::g2g_error!(
            g2g_core::log::Target::category("pyelement"),
            "batch stream {} is {} but stream {} is {}, pyelement needs one format and geometry across a batch",
            differing.index,
            differing.caps.to_gst_string(),
            first.index,
            first.caps.to_gst_string()
        );
        return Err(G2gError::CapsMismatch);
    }
    Ok(first.caps.clone())
}

#[derive(Debug)]
pub(crate) struct HostedInput {
    pub(crate) module: String,
    pub(crate) class: String,
    pub(crate) accept: Caps,
    pub(crate) cuda_frames: bool,
    params: Vec<(String, PropValue)>,
    configured: bool,
    fixed: Option<Caps>,
    #[cfg(feature = "python")]
    worker: Option<crate::host::PyWorker>,
}

impl HostedInput {
    pub(crate) fn new(module: String, class: String) -> Self {
        Self {
            module,
            class,
            accept: default_accept(),
            cuda_frames: false,
            params: Vec::new(),
            configured: false,
            fixed: None,
            #[cfg(feature = "python")]
            worker: None,
        }
    }

    pub(crate) fn domain(&self) -> MemoryDomainKind {
        if self.cuda_frames {
            MemoryDomainKind::Cuda
        } else {
            MemoryDomainKind::System
        }
    }

    pub(crate) fn input_domains(&self) -> DomainSet {
        DomainSet::only(self.domain())
    }

    pub(crate) fn is_configured(&self) -> bool {
        self.configured
    }

    pub(crate) fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        intersect_accepted(upstream_caps, &self.accept)
    }

    pub(crate) fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        } = caps
        else {
            return None;
        };
        let size = frame_bytes(*format, *width, *height);
        Some(if self.cuda_frames {
            AllocationParams::cuda(size, 1, 1)
        } else {
            AllocationParams::system(size, 1)
        })
    }

    pub(crate) fn configure(
        &mut self,
        absolute_caps: &Caps,
        #[cfg_attr(not(feature = "python"), allow(unused_variables))] draw_label: bool,
    ) -> Result<(), G2gError> {
        self.intercept_caps(absolute_caps)?;
        #[cfg(feature = "analytics")]
        if self.cuda_frames && matches!(absolute_caps, Caps::Batch { .. }) {
            g2g_core::g2g_error!(
                g2g_core::log::Target::category("pyelement"),
                "batch frames reach pyelement in system memory only, set cuda-frames=false"
            );
            return Err(G2gError::UnsupportedDomain);
        }
        self.fixed = Some(absolute_caps.clone());
        #[cfg(feature = "python")]
        {
            // A registry-built element starts with empty module/class until
            // `module=`/`class=` properties are applied; fail clearly here
            // rather than importing the empty module.
            if self.module.is_empty() || self.class.is_empty() {
                return Err(G2gError::NotConfigured);
            }
            // Spawn the worker once: a re-configure (re-negotiation) must not tear
            // down and re-init the hosted instance, which would discard a loaded
            // model. Matches `PySource` / `PyAggregator`.
            if self.worker.is_none() {
                self.worker = Some(crate::host::PyWorker::spawn(
                    &self.module,
                    &self.class,
                    draw_label,
                    &self.params,
                )?);
            }
        }
        self.configured = true;
        Ok(())
    }

    pub(crate) fn caps_changed(&mut self, caps: Caps) -> Result<(), G2gError> {
        self.intercept_caps(&caps)?;
        self.fixed = Some(caps);
        Ok(())
    }

    #[cfg(feature = "python")]
    fn worker_and_caps(&self) -> Result<(&crate::host::PyWorker, &Caps), G2gError> {
        let worker = self.worker.as_ref().ok_or(G2gError::NotConfigured)?;
        let caps = self.fixed.as_ref().ok_or(G2gError::NotConfigured)?;
        Ok((worker, caps))
    }

    #[cfg(feature = "python")]
    pub(crate) async fn run(
        &self,
        frame: Frame,
        produce: Option<&Caps>,
    ) -> Result<Vec<Frame>, G2gError> {
        let (worker, caps) = self.worker_and_caps()?;
        #[cfg(feature = "analytics")]
        if frame.meta.get::<BatchMeta>().is_some() {
            return Ok(vec![run_stream_batch(worker, frame).await?]);
        }
        worker.run(frame, caps, produce).await
    }

    #[cfg(feature = "python")]
    pub(crate) async fn consume(&self, frame: Frame) -> Result<(), G2gError> {
        let (worker, caps) = self.worker_and_caps()?;
        worker.consume(frame, caps).await
    }

    #[cfg(feature = "python")]
    pub(crate) async fn finish(&mut self) -> Result<(), G2gError> {
        let worker = self.worker.as_ref().ok_or(G2gError::NotConfigured)?;
        worker.eos().await?;
        self.worker = None;
        Ok(())
    }

    // The per-frame Python call embeds CPython via pyo3 and lives behind the
    // `python` feature, so the default build negotiates caps but cannot run frames.
    #[cfg(not(feature = "python"))]
    pub(crate) async fn run(
        &self,
        _frame: Frame,
        _produce: Option<&Caps>,
    ) -> Result<Vec<Frame>, G2gError> {
        Err(G2gError::UnsupportedDomain)
    }

    #[cfg(not(feature = "python"))]
    pub(crate) async fn consume(&self, _frame: Frame) -> Result<(), G2gError> {
        Err(G2gError::UnsupportedDomain)
    }

    #[cfg(not(feature = "python"))]
    pub(crate) async fn finish(&mut self) -> Result<(), G2gError> {
        Ok(())
    }

    pub(crate) fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "module" => {
                self.module = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "class" => {
                self.class = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "cuda-frames" => {
                self.cuda_frames = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "input-caps" => {
                self.accept = fixed_caps(value.as_str().ok_or(PropError::Type)?)?;
                Ok(())
            }
            // Any other property goes to the hosted Python instance. Which names
            // are real is the class's to say, so a typo is caught when it loads.
            other => {
                crate::props::forward(&mut self.params, other, value);
                Ok(())
            }
        }
    }

    pub(crate) fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "module" => Some(PropValue::Str(self.module.clone())),
            "class" => Some(PropValue::Str(self.class.clone())),
            "cuda-frames" => Some(PropValue::Bool(self.cuda_frames)),
            "input-caps" => Some(PropValue::Str(self.accept.to_gst_string())),
            other => self
                .params
                .iter()
                .find(|(k, _)| k == other)
                .map(|(_, v)| v.clone()),
        }
    }
}

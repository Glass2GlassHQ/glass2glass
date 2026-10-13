use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{
    AllocationParams, Caps, Dim, Frame, G2gError, PropError, PropKind, PropValue, PropertySpec,
    Rate, RawVideoFormat,
};

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
        upstream_caps.intersect(&self.accept)
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
        absolute_caps.intersect(&self.accept)?;
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
        caps.intersect(&self.accept)?;
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
    pub(crate) async fn run(&self, frame: Frame) -> Result<Vec<Frame>, G2gError> {
        let (worker, caps) = self.worker_and_caps()?;
        worker.run(frame, caps).await
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
    pub(crate) async fn run(&self, _frame: Frame) -> Result<Vec<Frame>, G2gError> {
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

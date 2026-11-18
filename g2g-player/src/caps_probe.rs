// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::fmt;
use std::future::Future;
use std::pin::Pin;

use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, ElementMetadata, G2gError, OutputSink, PipelinePacket,
};

type CapsReport = Box<dyn FnMut(&Caps) + Send>;

pub(crate) struct CapsReporter {
    report: CapsReport,
    unreported: Option<Caps>,
}

impl CapsReporter {
    pub(crate) fn new(report: impl FnMut(&Caps) + Send + 'static) -> Self {
        CapsReporter {
            report: Box::new(report),
            unreported: None,
        }
    }

    pub(crate) fn configure(&mut self, caps: &Caps) {
        self.unreported = Some(caps.clone());
    }

    pub(crate) fn observe(&mut self, packet: &PipelinePacket) {
        match packet {
            PipelinePacket::CapsChanged(caps) => self.unreported = Some(caps.clone()),
            // demuxers replace the placeholder caps negotiation fixed before their first frame
            PipelinePacket::DataFrame(_) => {
                if let Some(caps) = self.unreported.take() {
                    (self.report)(&caps);
                }
            }
            _ => {}
        }
    }
}

pub(crate) struct CapsProbe {
    metadata: ElementMetadata,
    caps: CapsReporter,
}

impl fmt::Debug for CapsReporter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CapsReporter")
            .field("unreported", &self.unreported)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for CapsProbe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CapsProbe")
            .field("metadata", &self.metadata.long_name)
            .finish_non_exhaustive()
    }
}

impl CapsProbe {
    pub(crate) fn new(
        metadata: ElementMetadata,
        report: impl FnMut(&Caps) + Send + 'static,
    ) -> Self {
        CapsProbe {
            metadata,
            caps: CapsReporter::new(report),
        }
    }
}

impl AsyncElement for CapsProbe {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        self.metadata
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.caps.configure(absolute_caps);
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            self.caps.observe(&packet);
            out.push(packet).await?;
            Ok(())
        })
    }
}

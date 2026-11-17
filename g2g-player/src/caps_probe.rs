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

pub(crate) struct CapsProbe {
    metadata: ElementMetadata,
    report: CapsReport,
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
            report: Box::new(report),
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
        (self.report)(absolute_caps);
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::CapsChanged(caps) = &packet {
                (self.report)(caps);
            }
            out.push(packet).await?;
            Ok(())
        })
    }
}

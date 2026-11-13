// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![allow(dead_code)] // each test file uses a subset

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_duplex_session, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket,
};
use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
use g2g_plugins::webrtcduplex::WebRtcDuplexSession;

use crate::webrtc_streams_common::fixed_caps;

pub(crate) const FRAME_INTERVAL: Duration = Duration::from_millis(33);

pub(crate) struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Paces the synthetic units at frame rate, then EOS.
pub(crate) struct PacedSrc {
    pub(crate) codec: WebRtcVideoCodec,
    pub(crate) units: Arc<Vec<Vec<u8>>>,
}

impl SourceLoop for PacedSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(fixed_caps(self.codec)))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(fixed_caps(self.codec)))
                .await?;
            for (seq, unit) in self.units.iter().enumerate() {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(unit.clone().into_boxed_slice())),
                    FrameTiming {
                        pts_ns: seq as u64 * FRAME_INTERVAL.as_nanos() as u64,
                        ..FrameTiming::default()
                    },
                    seq as u64,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                tokio::time::sleep(FRAME_INTERVAL).await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.units.len() as u64)
        })
    }
}

#[derive(Default)]
pub(crate) struct Received {
    pub(crate) caps: Option<Caps>,
    pub(crate) units: Vec<(Vec<u8>, bool)>,
}

pub(crate) struct RecordingSink {
    pub(crate) received: Arc<Mutex<Received>>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let mut received = self.received.lock().unwrap();
            match packet {
                PipelinePacket::CapsChanged(caps) => {
                    received.caps.get_or_insert(caps);
                }
                PipelinePacket::DataFrame(frame) => {
                    let bytes = frame
                        .domain
                        .as_system_slice()
                        .expect("system memory")
                        .to_vec();
                    received.units.push((bytes, frame.timing.keyframe));
                }
                _ => {}
            }
            Ok(())
        })
    }
}

pub(crate) async fn run_peer(
    codec: WebRtcVideoCodec,
    units: Arc<Vec<Vec<u8>>>,
    session: &mut WebRtcDuplexSession,
    received: Arc<Mutex<Received>>,
    clock: &ZeroClock,
) -> Result<g2g_core::runtime::RunStats, G2gError> {
    let mut src = PacedSrc { codec, units };
    let mut sink = RecordingSink { received };
    let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut src];
    let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut sink];
    run_duplex_session(sources, session, sinks, clock, 8).await
}

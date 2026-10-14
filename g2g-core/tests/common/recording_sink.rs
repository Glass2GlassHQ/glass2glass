#![allow(dead_code)] // no one test file uses every helper here

use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_core::caps::CapsSet;
use g2g_core::format_element::CapsConstraint;
use g2g_core::memory::MemoryDomain;
use g2g_core::{AsyncElement, Caps, ConfigureOutcome, Frame, G2gError, OutputSink, PipelinePacket};

/// What the sink saw, in order: every frame's sequence and mark count, and the
/// frame position each `CapsChanged` arrived at.
#[derive(Default, Debug)]
pub(crate) struct Record {
    pub(crate) frames: Vec<(u64, u8)>,
    pub(crate) caps: Vec<(usize, Caps)>,
    pub(crate) eos_count: usize,
}

impl Record {
    pub(crate) fn marks(&self) -> Vec<u8> {
        self.frames.iter().map(|&(_, m)| m).collect()
    }
}

/// Records every packet, and accepts the formats `accepts` names (which is what
/// the runner's downstream-feasibility snapshot is built from). `pace` makes it
/// take its time over each frame, which keeps the shallow link full and so keeps
/// the producer inside a blocking send: that is the state a mutation has to be
/// able to interrupt.
pub(crate) struct RecordingSink {
    pub(crate) record: Arc<Mutex<Record>>,
    pub(crate) accepts: Vec<Caps>,
    pub(crate) pace: Option<Duration>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        if self.accepts.contains(upstream) {
            Ok(upstream.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::from_alternatives(self.accepts.clone()))
    }
    fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.accepts.contains(caps) {
            Ok(ConfigureOutcome::Accepted)
        } else {
            Err(G2gError::CapsMismatch)
        }
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let mut record = self.record.lock().unwrap();
        match packet {
            PipelinePacket::DataFrame(f) => {
                let entry = (f.sequence, mark_of(&f));
                record.frames.push(entry);
                if let Some(pace) = self.pace {
                    drop(record);
                    std::thread::sleep(pace);
                    return core::future::ready(Ok(()));
                }
            }
            PipelinePacket::CapsChanged(caps) => {
                let at = record.frames.len();
                record.caps.push((at, caps));
            }
            PipelinePacket::Eos => record.eos_count += 1,
            _ => {}
        }
        core::future::ready(Ok(()))
    }
}

pub(crate) fn mark_of(frame: &Frame) -> u8 {
    match &frame.domain {
        MemoryDomain::System(s) => s.as_slice()[0],
        other => panic!("unexpected frame domain {other:?}"),
    }
}

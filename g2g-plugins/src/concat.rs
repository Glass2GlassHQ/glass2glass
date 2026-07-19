//! Stream concatenation (`concat`). Plays its N inputs one after another: input
//! 0 forwards until it ends, then input 1, and so on, producing a single output
//! stream (the g2g analog of GStreamer's `concat`). Frames arriving on an input
//! that is not yet active are buffered until its turn. `no_std`.
//!
//! With `adjust-base` (default) each input's timestamps are offset by the total
//! duration of the inputs before it, so the streams play back-to-back on a
//! continuous timeline instead of all starting at zero.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::{
    Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, MultiInputElement,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

#[derive(Debug)]
pub struct Concat {
    inputs: usize,
    active: usize,
    adjust_base: bool,
    configured: Vec<Option<Caps>>,
    /// Packets that arrived on an input before it became active.
    pending: Vec<Vec<PipelinePacket>>,
    ended: Vec<bool>,
    /// Cumulative timeline offset for the active input (sum of prior spans).
    base_offset_ns: u64,
    /// Largest `pts + duration` forwarded for the active input so far.
    active_end_ns: u64,
}

impl Concat {
    pub fn new(inputs: usize) -> Self {
        assert!(inputs > 0, "Concat needs at least one input");
        Self {
            inputs,
            active: 0,
            adjust_base: true,
            configured: vec![None; inputs],
            pending: (0..inputs).map(|_| Vec::new()).collect(),
            ended: vec![false; inputs],
            base_offset_ns: 0,
            active_end_ns: 0,
        }
    }

    /// Forward one packet from the active input, applying the timeline offset to
    /// data frames when `adjust-base` is on, and tracking the input's end time.
    async fn forward(
        &mut self,
        packet: PipelinePacket,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let packet = match packet {
            PipelinePacket::DataFrame(mut frame) => {
                let end = frame.timing.pts_ns.saturating_add(frame.timing.duration_ns);
                if end > self.active_end_ns {
                    self.active_end_ns = end;
                }
                if self.adjust_base {
                    frame.timing.pts_ns = frame.timing.pts_ns.saturating_add(self.base_offset_ns);
                    if frame.timing.dts_ns != 0 {
                        frame.timing.dts_ns =
                            frame.timing.dts_ns.saturating_add(self.base_offset_ns);
                    }
                }
                PipelinePacket::DataFrame(frame)
            }
            other => other,
        };
        out.push(packet).await?;
        Ok(())
    }

    /// The active input ended: fold its span into the base offset and advance to
    /// the next input, flushing any buffered packets and skipping inputs that
    /// already ended before their turn.
    async fn advance(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        loop {
            self.base_offset_ns = self.base_offset_ns.saturating_add(self.active_end_ns);
            self.active_end_ns = 0;
            self.active += 1;
            if self.active >= self.inputs {
                return Ok(());
            }
            let pending = core::mem::take(&mut self.pending[self.active]);
            for packet in pending {
                self.forward(packet, out).await?;
            }
            if !self.ended[self.active] {
                return Ok(());
            }
            // this input was fully buffered and already ended: roll to the next.
        }
    }
}

impl MultiInputElement for Concat {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    /// The output is the same stream as each input in turn; derive it from input 0
    /// (the inputs are assumed the same format, as `concat` requires).
    fn output_follows_input(&self) -> Option<usize> {
        Some(0)
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.configured[input] = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.configured[0].clone().ok_or(G2gError::NotConfigured)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Concat",
            "Generic",
            "Concatenates N streams end to end",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        CONCAT_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "adjust-base" => self.adjust_base = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "adjust-base" => Some(PropValue::Bool(self.adjust_base)),
            // gst exposes the active pad read-only; mirror it as an index.
            "active-pad" => Some(PropValue::Uint(self.active as u64)),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::Eos => {
                    self.ended[input] = true;
                    if input == self.active {
                        self.advance(out).await?;
                    }
                }
                other => {
                    if input == self.active {
                        self.forward(other, out).await?;
                    } else if input > self.active {
                        self.pending[input].push(other);
                    }
                    // input < active: already fully forwarded; drop.
                }
            }
            Ok(())
        })
    }
}

static CONCAT_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "adjust-base",
    PropKind::Bool,
    "offset each input onto a continuous timeline",
)];

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Frame, FrameTiming, MemoryDomain, PushOutcome, SystemSlice};

    #[derive(Default)]
    struct CollectSink {
        pts: Vec<u64>,
    }
    impl OutputSink for CollectSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            if let PipelinePacket::DataFrame(f) = &packet {
                self.pts.push(f.timing.pts_ns);
            }
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }
    }

    fn frame(pts_ns: u64, dur_ns: u64) -> PipelinePacket {
        let timing = FrameTiming {
            pts_ns,
            duration_ns: dur_ns,
            ..FrameTiming::default()
        };
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 4].into_boxed_slice())),
            timing,
            sequence: 0,
            meta: Default::default(),
        })
    }

    #[tokio::test]
    async fn plays_inputs_in_order_with_adjusted_timeline() {
        let mut c = Concat::new(2);
        let mut out = CollectSink::default();
        // input 1 sends first (buffered), then input 0 plays, ends, then input 1.
        c.process(1, frame(0, 100), &mut out).await.unwrap();
        c.process(0, frame(0, 100), &mut out).await.unwrap();
        c.process(0, frame(100, 100), &mut out).await.unwrap();
        c.process(0, PipelinePacket::Eos, &mut out).await.unwrap();
        c.process(1, frame(100, 100), &mut out).await.unwrap();
        c.process(1, PipelinePacket::Eos, &mut out).await.unwrap();
        // input 0: pts 0,100. input 1 offset by input 0's span (200): 200, 300.
        assert_eq!(out.pts, vec![0, 100, 200, 300]);
    }

    #[tokio::test]
    async fn early_ended_input_is_skipped() {
        let mut c = Concat::new(2);
        let mut out = CollectSink::default();
        // input 1 ends before its turn (empty); input 0 plays then concat finishes.
        c.process(1, PipelinePacket::Eos, &mut out).await.unwrap();
        c.process(0, frame(0, 100), &mut out).await.unwrap();
        c.process(0, PipelinePacket::Eos, &mut out).await.unwrap();
        assert_eq!(out.pts, vec![0]);
        assert_eq!(c.active, 2);
    }

    #[tokio::test]
    async fn adjust_base_off_keeps_original_pts() {
        let mut c = Concat::new(2);
        c.set_property("adjust-base", PropValue::Bool(false))
            .unwrap();
        let mut out = CollectSink::default();
        c.process(0, frame(0, 100), &mut out).await.unwrap();
        c.process(0, PipelinePacket::Eos, &mut out).await.unwrap();
        c.process(1, frame(0, 100), &mut out).await.unwrap();
        assert_eq!(out.pts, vec![0, 0]);
    }
}

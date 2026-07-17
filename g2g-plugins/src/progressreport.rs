//! Progress report (`progressreport`). A 1:1 passthrough that counts frames and
//! bytes and, every `update-freq` seconds of stream time, logs a progress line
//! (the g2g analog of GStreamer's `progressreport`). Counts are also exposed via
//! getters. `no_std` (logging goes through the crate log macros).
//!
//! The cadence is keyed on buffer PTS, not a wall clock, so it works the same on
//! a headless / RTOS target with no system clock. `silent` suppresses the log
//! lines without stopping the counting.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;

use g2g_core::log::{short_type_name, LogSource};
use g2g_core::{
    g2g_info, AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

#[derive(Debug)]
pub struct ProgressReport {
    update_freq_s: i64,
    silent: bool,
    frames: u64,
    bytes: u64,
    next_report_ns: u64,
    instance_name: Option<String>,
}

impl Default for ProgressReport {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressReport {
    /// Report every 5s of stream time (the gst default).
    pub fn new() -> Self {
        Self {
            update_freq_s: 5,
            silent: false,
            frames: 0,
            bytes: 0,
            next_report_ns: 0,
            instance_name: None,
        }
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    fn freq_ns(&self) -> u64 {
        (self.update_freq_s.max(1) as u64).saturating_mul(1_000_000_000)
    }
}

impl AsyncElement for ProgressReport {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Pure passthrough of whatever flows through it.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    self.frames += 1;
                    if let MemoryDomain::System(slice) = &frame.domain {
                        self.bytes += slice.as_slice().len() as u64;
                    }
                    let pts = frame.timing.pts_ns;
                    if pts >= self.next_report_ns {
                        if !self.silent {
                            g2g_info!(
                                self,
                                "progress: {} frames, {} bytes, pts {} ns",
                                self.frames,
                                self.bytes,
                                pts
                            );
                        }
                        self.next_report_ns = pts.saturating_add(self.freq_ns());
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // The runner emits the final Eos after process(Eos) returns.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PROGRESSREPORT_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new("Progress report", "Generic", "Periodically reports pipeline progress", "g2g")
    }

    fn set_instance_name(&mut self, name: String) {
        self.instance_name = Some(name);
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "update-freq" => self.update_freq_s = value.as_int().ok_or(PropError::Type)?,
            "silent" => self.silent = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "update-freq" => Some(PropValue::Int(self.update_freq_s)),
            "silent" => Some(PropValue::Bool(self.silent)),
            _ => None,
        }
    }
}

static PROGRESSREPORT_PROPS: &[PropertySpec] = &[
    PropertySpec::new("update-freq", PropKind::Int, "report interval in seconds of stream time"),
    PropertySpec::new("silent", PropKind::Bool, "suppress the log lines (still counts)"),
];

impl LogSource for ProgressReport {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.instance_name.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::pin::Pin;
    use g2g_core::{Frame, FrameTiming, PushOutcome, SystemSlice};

    struct NullSink;
    impl OutputSink for NullSink {
        fn push<'a>(
            &'a mut self,
            _packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }
    }

    fn frame(bytes: usize, pts_ns: u64) -> PipelinePacket {
        let timing = FrameTiming { pts_ns, ..FrameTiming::default() };
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(alloc::vec![0u8; bytes].into_boxed_slice())),
            timing,
            sequence: 0,
            meta: Default::default(),
        })
    }

    #[tokio::test]
    async fn counts_frames_and_bytes() {
        let mut p = ProgressReport::new();
        p.set_property("silent", PropValue::Bool(true)).unwrap();
        let mut out = NullSink;
        p.process(frame(100, 0), &mut out).await.unwrap();
        p.process(frame(200, 1_000_000_000), &mut out).await.unwrap();
        assert_eq!(p.frames(), 2);
        assert_eq!(p.bytes(), 300);
    }

    #[test]
    fn update_freq_round_trips() {
        let mut p = ProgressReport::new();
        p.set_property("update-freq", PropValue::Int(10)).unwrap();
        assert_eq!(p.get_property("update-freq"), Some(PropValue::Int(10)));
        assert_eq!(p.freq_ns(), 10_000_000_000);
    }
}

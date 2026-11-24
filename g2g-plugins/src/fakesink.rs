// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Inspecting sink. Counts frames, records the last sequence number,
//! tracks EOS, and records mid-stream `CapsChanged` events alongside
//! the frame count at which they arrived. Used by tests and to
//! validate runner plumbing.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::log::{short_type_name, ClockTime, LogName, LogSource};
use g2g_core::metrics::{LatencyHistogram, LatencySnapshot};
use g2g_core::{
    g2g_debug, AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Segment,
};

#[cfg(feature = "std")]
use g2g_core::ClockSync;

#[cfg(feature = "std")]
use g2g_core::metrics::monotonic_ns;

/// Position of a recorded `CapsChanged` packet inside the sink's input
/// stream: `frames_before` is the number of `DataFrame` packets that
/// arrived at the sink strictly before this `CapsChanged`.
#[derive(Debug, Clone)]
pub struct CapsChange {
    pub caps: Caps,
    pub frames_before: u64,
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::fakesink::FakeSink;
///
/// let sink = FakeSink::new();
/// assert_eq!(sink.received(), 0);
/// assert!(!sink.eos_seen());
/// ```
#[derive(Debug, Default)]
pub struct FakeSink {
    received: u64,
    last_sequence: Option<u64>,
    eos_seen: bool,
    flushes: u64,
    configured: bool,
    caps_changes: Vec<CapsChange>,
    /// The most recent `Segment` received, and how many have arrived. A stream
    /// opens with one; a flushing seek emits another after the `Flush`.
    last_segment: Option<Segment>,
    segments: u64,
    /// Glass-to-glass latency distribution recorded as
    /// `monotonic_ns() - arrival_ns` on every received `DataFrame`
    /// whose `FrameTiming::arrival_ns` is non-zero. Frames without a
    /// source-side stamp (synthesized by transforms, or sources that
    /// don't stamp) are skipped silently. Test code can pull a
    /// snapshot via [`FakeSink::latency_snapshot`] and assert bounds.
    latency: LatencyHistogram,
    /// Data pointer of each received [`MemoryDomain::SystemView`] frame's
    /// backing, in arrival order. A test compares these to the source's emitted
    /// pointers to prove the frames crossed the pipeline (through a flip) with
    /// zero copies. `System` / GPU frames don't contribute. This is what makes
    /// the sink "stride-aware": it inspects the shared-view domain rather than
    /// assuming a contiguous buffer.
    view_backing_ptrs: Vec<usize>,
    /// Materialized (dense row-major) bytes of the most recent `SystemView`
    /// frame, for correctness checks on a strided transform's output.
    last_view_bytes: Option<Vec<u8>>,
    // the inverse of `silent`, so the derived default matches gst's silent fake sinks
    records_each_buffer: bool,
    last_message: String,
    log_name: LogName,
    #[cfg(feature = "std")]
    clock_sync: Option<ClockSync>,
}

/// The `silent` / `last-message` pair of gst's fake sinks.
pub(crate) static FAKE_SINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "silent",
        PropKind::Bool,
        "stop recording a line per buffer in last-message",
    )
    .with_default("true"),
    PropertySpec::new(
        "last-message",
        PropKind::Str,
        "the most recent buffer's line, empty while silent",
    )
    .read_only(),
];

impl FakeSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn received(&self) -> u64 {
        self.received
    }

    pub fn last_sequence(&self) -> Option<u64> {
        self.last_sequence
    }

    pub fn eos_seen(&self) -> bool {
        self.eos_seen
    }

    /// Number of `Flush` packets seen. A flush resets `last_sequence`.
    pub fn flushes(&self) -> u64 {
        self.flushes
    }

    pub fn caps_changes(&self) -> &[CapsChange] {
        &self.caps_changes
    }

    /// The most recent `Segment` the sink received, or `None` if no SEGMENT has
    /// arrived. Use it to map a frame's timestamp to running time.
    pub fn last_segment(&self) -> Option<Segment> {
        self.last_segment
    }

    /// Number of `Segment` packets seen.
    pub fn segments(&self) -> u64 {
        self.segments
    }

    /// Snapshot of the glass-to-glass latency histogram. Count of zero
    /// means no `arrival_ns`-stamped frames have been received (either
    /// the source doesn't stamp, or no frames have arrived yet).
    pub fn latency_snapshot(&self) -> LatencySnapshot {
        self.latency.snapshot()
    }

    /// Data pointers of received [`MemoryDomain::SystemView`] frames, in
    /// arrival order. Equal to the source's emitted pointers iff every frame
    /// crossed the pipeline with zero copies.
    pub fn view_backing_ptrs(&self) -> &[usize] {
        &self.view_backing_ptrs
    }

    /// Materialized dense bytes of the most recent `SystemView` frame, or
    /// `None` if no shared-view frame has arrived.
    pub fn last_view_bytes(&self) -> Option<&[u8]> {
        self.last_view_bytes.as_deref()
    }

    /// The most recent buffer's line, empty while `silent`.
    pub fn last_message(&self) -> &str {
        &self.last_message
    }

    fn record_buffer(&mut self, bytes: usize, timing: &FrameTiming) {
        self.last_message = format!(
            "chain: {bytes} bytes, pts: {}, running time: {}",
            ClockTime(timing.pts_ns),
            self.running_time()
        );
        g2g_debug!(self, "{}", self.last_message);
    }

    #[cfg(feature = "std")]
    fn running_time(&self) -> String {
        let Some(sync) = &self.clock_sync else {
            return String::from("none");
        };
        let running_ns = sync
            .now_ns()
            .saturating_sub(sync.base_time())
            .saturating_sub(sync.paused_total_ns());
        format!("{}", ClockTime(running_ns))
    }

    #[cfg(not(feature = "std"))]
    fn running_time(&self) -> String {
        String::from("none")
    }
}

impl LogSource for FakeSink {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

impl AsyncElement for FakeSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Declare wildcard sink to the solver. `FakeSink`
    /// accepts whatever upstream produces (its `intercept_caps` is a
    /// pass-through), which is exactly what `CapsConstraint::AcceptsAny`
    /// models. Skips the dynamic intercept callback and lets the
    /// solver propagate upstream caps unchanged.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        FAKE_SINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "silent" => {
                self.records_each_buffer = !value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "last-message" => Err(PropError::ReadOnly),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "silent" => Some(PropValue::Bool(!self.records_each_buffer)),
            "last-message" => Some(PropValue::Str(self.last_message.clone())),
            _ => None,
        }
    }

    #[cfg(feature = "std")]
    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.clock_sync = Some(sync);
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Fake sink",
            "Sink",
            "Discards all buffers (a no-op terminal for testing)",
            crate::AARON_BOXER,
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(f) => {
                    self.last_sequence = Some(f.sequence);
                    self.received += 1;
                    if self.records_each_buffer {
                        let bytes = f.domain.as_system_slice().map_or(0, <[u8]>::len);
                        self.record_buffer(bytes, &f.timing);
                    }
                    // M180: stride-aware path. A shared-view frame is recorded
                    // by its backing pointer (the zero-copy witness) and
                    // materialized for correctness checks; a contiguous frame
                    // needs neither.
                    if let MemoryDomain::SystemView(sv) = &f.domain {
                        self.view_backing_ptrs.push(sv.backing().as_ptr() as usize);
                        self.last_view_bytes = Some(sv.materialize().into_vec());
                    }
                    // Record glass-to-glass latency if the source stamped
                    // arrival_ns. Sub-monotonic stamps (zero, or future
                    // values) are skipped silently — they're either
                    // unstamped frames or a clock-domain mismatch the
                    // sink shouldn't paper over.
                    #[cfg(feature = "std")]
                    {
                        let arrival = f.timing.arrival_ns;
                        if arrival != 0 {
                            let now = monotonic_ns();
                            if now >= arrival {
                                self.latency.record(now - arrival);
                            }
                        }
                    }
                }
                PipelinePacket::Eos => {
                    self.eos_seen = true;
                }
                PipelinePacket::Flush => {
                    // Seek flush: `last_sequence` reports only frames after it.
                    self.flushes += 1;
                    self.last_sequence = None;
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.caps_changes.push(CapsChange {
                        caps,
                        frames_before: self.received,
                    });
                }
                PipelinePacket::Segment(seg) => {
                    self.segments += 1;
                    self.last_segment = Some(seg);
                }
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for FakeSink {
    /// Wildcard sink: accepts any caps, matching the runtime
    /// `caps_constraint_as_sink` of `AcceptsAny`.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

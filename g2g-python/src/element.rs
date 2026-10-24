//! [`PyTransform`]: a gst-python-ml element shell hosted as a g2g transform.
//!
//! This is the Rust mirror of gst-python-ml's `backend/gst` `BaseTransform`:
//! it negotiates caps, then on each frame hands the buffer to a hosted Python
//! instance and pushes the result downstream. The negotiation half is pure
//! Rust and always compiles; the per-frame Python call lives in [`crate::host`]
//! behind the `python` feature.
//!
//! Caps model: an overlay/inference-in-place element (the `ActionTask` shape)
//! takes a raw-video frame and returns one in the same format, so it is a
//! non-boundary transform whose output caps equal its input. `output-caps=`
//! makes it a boundary instead ([`AsyncElement::is_format_boundary`] plus a
//! `DerivedOutput` constraint naming the declared caps, like `g2g-ml`'s
//! `OrtInference`): the single-chain gst-python-ml families read one media type
//! and write another (audio in and a transcript out, text in and speech out),
//! and their output is not the size of their input, so the hosted element
//! returns it through `meta.emit` (see [`crate::host`]).
//!
//! Memory-domain model: the frame is read where it lies and forwarded
//! untouched, so both pads carry the one domain the hosted code reads, System or
//! (under `cuda-frames`) CUDA. See [`AsyncElement::input_domains`] below.

use core::future::Future;
use core::pin::Pin;
use std::collections::VecDeque;
use std::time::Instant;

use g2g_core::metrics::monotonic_ns;

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, Frame, FrameTiming, G2gError, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, PushOutcome, Segment,
    VIDEO_SINK_MAX_LATENESS_NS,
};

use crate::hosted_input::{
    default_accept, intersect_accepted, HostedInput, CLASS_PROPERTY, CUDA_FRAMES_PROPERTY,
    INPUT_CAPS_PROPERTY, MODULE_PROPERTY,
};
use crate::props::{fixed_caps, hosted_element_props, HostedClassProperties};
use g2g_core::caps_parse::parse_raw_format;

const RECENT_PUSH_CAPACITY: usize = 64;
// a run that ran long once must not drop every later frame, a dropped frame never runs to correct it
const RUN_TIME_STALE_AFTER_NS: u64 = 100_000_000;

#[derive(Debug, Clone, Copy)]
struct SinkClock {
    running_time_ns: u64,
    taken_ns: u64,
}

#[derive(Debug, Clone, Copy)]
struct RecentPush {
    running_time_ns: u64,
    pushed_ns: u64,
}

/// A gst-python-ml element hosted as a first-class g2g transform.
#[derive(Debug)]
pub struct PyTransform {
    input: HostedInput,
    /// Caps produced downstream when the hosted element emits a different media
    /// type than it reads: audio in / text out for transcription, and the rest
    /// of the 1-in-1-out gst-python-ml families. Unset means it emits what it
    /// read (the overlay / detector shape).
    produce: Option<Caps>,
    /// Overlay flag bridged to the Python task (an example backend-declared
    /// property; `ActionTask` reads `self.draw_label`).
    draw_label: bool,
    /// Name of the metadata a frame must carry for the hosted element to see it:
    /// a blob header, or `detections` for any detection. Empty runs every frame.
    only_on: g2g_core::OnlyOnGate,
    qos: bool,
    segment: Option<Segment>,
    sink_clock: Option<SinkClock>,
    /// Time from this element pushing a frame to the sink judging it, learned
    /// from a QoS report on a frame in `recent_pushes`.
    downstream_delay_ns: u64,
    recent_pushes: VecDeque<RecentPush>,
    last_run_ns: u64,
    last_run_measured_ns: u64,
    late_dropped: u64,
    emitted: u64,
}

impl PyTransform {
    /// Host the `class` from Python `module`. The instance is created at
    /// `configure_pipeline` time (under the GIL), not here, so construction
    /// stays cheap and infallible like the other elements' `new`.
    pub fn new(module: impl Into<String>, class: impl Into<String>) -> Self {
        Self {
            input: HostedInput::new(module.into(), class.into()),
            produce: None,
            draw_label: false,
            only_on: g2g_core::OnlyOnGate::default(),
            qos: true,
            segment: None,
            sink_clock: None,
            downstream_delay_ns: 0,
            recent_pushes: VecDeque::with_capacity(RECENT_PUSH_CAPACITY),
            last_run_ns: 0,
            last_run_measured_ns: 0,
            late_dropped: 0,
            emitted: 0,
        }
    }

    /// Override the accepted sink caps (e.g. to host an NV12 element). The
    /// supported set may carry `Any` dims/rate: negotiation fixes them against
    /// concrete upstream caps, so `process` always sees a fixed format.
    pub fn with_accept(mut self, caps: Caps) -> Self {
        self.input.accept = caps;
        self
    }

    /// Emit `caps` downstream instead of the negotiated input caps, for a hosted
    /// element that changes media type.
    pub fn with_produce(mut self, caps: Caps) -> Self {
        self.produce = Some(caps);
        self
    }

    /// The caps this element puts on its source pad for `input`: the declared
    /// output when it changes media type, else the input it passes through.
    fn output_for(&self, input: &Caps) -> Caps {
        self.produce.clone().unwrap_or_else(|| input.clone())
    }

    /// Set the `draw-label` overlay flag forwarded to the Python task.
    pub fn with_draw_label(mut self, on: bool) -> Self {
        self.draw_label = on;
        self
    }

    /// Host an element that works on GPU-resident CUDA frames: they reach it as
    /// `__cuda_array_interface__` planes through `g2g_process_cuda` and flow on
    /// still device-resident, as the input or as the planes the hook returns. Sets this element's whole memory-domain story (see
    /// [`AsyncElement::input_domains`]); the hosted class must define
    /// `g2g_process_cuda` and the caps must be semi-planar (NV12 / P010).
    pub fn with_cuda_frames(mut self, on: bool) -> Self {
        self.input.cuda_frames = on;
        self
    }

    pub(crate) fn with_text_output(mut self) -> Self {
        self.input.text_output = true;
        self
    }

    /// Count of frames pushed downstream. Useful in tests.
    pub fn emitted_count(&self) -> u64 {
        self.emitted
    }

    pub fn late_dropped_count(&self) -> u64 {
        self.late_dropped
    }

    fn running_time(&self, frame: &Frame) -> Option<u64> {
        let pts = frame.timing.pts_ns;
        if pts == FrameTiming::PTS_NONE {
            return None;
        }
        match &self.segment {
            Some(segment) => segment.to_running_time(pts),
            None => Some(pts),
        }
    }

    fn is_late(&self, frame: &Frame) -> bool {
        let Some(sink_clock) = self.sink_clock else {
            return false;
        };
        let Some(running_time) = self.running_time(frame) else {
            return false;
        };
        let now_ns = monotonic_ns();
        let elapsed_ns = now_ns.saturating_sub(sink_clock.taken_ns);
        let run_is_fresh =
            now_ns.saturating_sub(self.last_run_measured_ns) <= RUN_TIME_STALE_AFTER_NS;
        let predicted_run_ns = if run_is_fresh { self.last_run_ns } else { 0 };
        let reaches_sink_at = sink_clock
            .running_time_ns
            .saturating_add(elapsed_ns)
            .saturating_add(predicted_run_ns)
            .saturating_add(self.downstream_delay_ns);
        running_time.saturating_add(VIDEO_SINK_MAX_LATENESS_NS) < reaches_sink_at
    }

    async fn push_frame(&mut self, frame: Frame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        self.emitted += 1;
        if let Some(running_time_ns) = self.running_time(&frame) {
            if self.recent_pushes.len() == RECENT_PUSH_CAPACITY {
                self.recent_pushes.pop_front();
            }
            self.recent_pushes.push_back(RecentPush {
                running_time_ns,
                pushed_ns: monotonic_ns(),
            });
        }
        let outcome = out.push(PipelinePacket::DataFrame(frame)).await?;
        self.note_push(outcome);
        Ok(())
    }

    fn note_push(&mut self, outcome: PushOutcome) {
        let PushOutcome::Qos(report) = outcome else {
            return;
        };
        if !self.qos {
            return;
        }
        let reported_push = self
            .recent_pushes
            .iter()
            .rev()
            .find(|push| push.running_time_ns == report.running_time_ns);
        if let Some(push) = reported_push {
            self.downstream_delay_ns = report.taken_ns.saturating_sub(push.pushed_ns);
        }
        self.sink_clock = Some(SinkClock {
            running_time_ns: report
                .running_time_ns
                .saturating_add_signed(report.jitter_ns),
            taken_ns: report.taken_ns,
        });
    }

    // the returned frames are what the hosted class emitted to text_src
    pub(crate) async fn process_packet(
        &mut self,
        packet: PipelinePacket,
        out: &mut dyn OutputSink,
    ) -> Result<Vec<Frame>, G2gError> {
        if !self.input.is_configured() {
            return Err(G2gError::NotConfigured);
        }
        match packet {
            PipelinePacket::DataFrame(frame) => {
                if self.is_late(&frame) {
                    self.late_dropped += 1;
                    return Ok(Vec::new());
                }
                if self.only_on.forwards_untouched(&frame.meta) {
                    self.push_frame(frame, out).await?;
                    return Ok(Vec::new());
                }
                let started = Instant::now();
                let outputs = self.input.run(frame, self.produce.as_ref()).await?;
                self.last_run_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                self.last_run_measured_ns = monotonic_ns();
                for output in outputs.frames {
                    self.push_frame(output, out).await?;
                }
                return Ok(outputs.text);
            }
            // A mid-stream change to anything outside the accepted set is a
            // hard error; otherwise announce this element's own output side
            // so downstream stays in step. Under `output-caps=` the runner's
            // packet carries that output, `configure_pipeline` took the input.
            PipelinePacket::CapsChanged(c) => {
                let announce = self.output_for(&c);
                if self.produce.is_none() {
                    self.input.caps_changed(c)?;
                }
                out.push(PipelinePacket::CapsChanged(announce)).await?;
            }
            PipelinePacket::Flush => {
                self.sink_clock = None;
                self.downstream_delay_ns = 0;
                self.recent_pushes.clear();
                out.push(PipelinePacket::Flush).await?;
            }
            PipelinePacket::Segment(seg) => {
                self.segment = Some(seg);
                out.push(PipelinePacket::Segment(seg)).await?;
            }
            // Stateless per-frame host: nothing buffered to drain.
            PipelinePacket::Eos => {}
            other => {
                out.push(other).await?;
            }
        }
        Ok(Vec::new())
    }
}

impl AsyncElement for PyTransform {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// The hosted element reads and writes the frame in place, so the output
    /// caps equal the input (when it is in the accepted set) unless
    /// `output-caps=` declared a different media type. Declaring this native
    /// constraint (rather than the default legacy intercept-only path, whose
    /// output the solver leaves unconstrained) lets the graph solver derive this
    /// element's output edge and lets the runtime forward-caps resolve steer a
    /// mid-stream `CapsChanged` (e.g. an upstream decoder's first-frame caps)
    /// cleanly through it, instead of stalling on an unconstrained boundary.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // Stating both sides lets the solver narrow the input link too. Derivation
        // only pushes forward, so wavparse's any-rate any-channels fixates alone.
        if let Some(produce) = self.produce.clone() {
            return CapsConstraint::Mapping(vec![(
                CapsSet::one(self.input.accept.clone()),
                CapsSet::one(produce),
            )]);
        }
        let accept = self.input.accept.clone();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            match intersect_accepted(input, &accept) {
                Ok(_) => CapsSet::one(input.clone()),
                Err(_) => CapsSet::from_alternatives(Vec::new()),
            }
        }))
    }

    /// A hosted element that declares `output-caps=` turns one media type into
    /// another (audio into a transcript), which is what a boundary is.
    fn is_format_boundary(&self) -> bool {
        self.produce.is_some()
    }

    /// The legacy-bridge half of the constraint above, for the runner paths that
    /// derive a boundary element's output side through this hook.
    fn propose_output_caps(&self, input: &Caps) -> Caps {
        self.output_for(input)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.input.intercept_caps(upstream_caps)
    }

    /// The hosted element reads the frame where it already is and forwards it
    /// untouched, so the domain it emits is the domain it consumes: System bytes
    /// over the buffer protocol, or CUDA device memory over
    /// `__cuda_array_interface__` under `cuda-frames`. Declaring one domain on
    /// both pads keeps that relation honest, so the domain-converter auto-plug
    /// splices a download / upload *ahead* of this element when upstream cannot
    /// deliver what the hosted code reads, and splices nothing after it (the
    /// frame really does leave in the declared domain).
    fn input_domains(&self) -> DomainSet {
        self.input.input_domains()
    }

    fn output_memory(&self) -> MemoryDomainKind {
        self.input.domain()
    }

    /// Ask upstream to allocate in the domain the hosted code can read, so a
    /// multi-domain producer (an NVDEC that can keep frames on the device or
    /// download them) settles on it rather than needing a converter node. Only
    /// the domain is constrained: this element allocates nothing of its own, so
    /// it imposes no buffer count or alignment, and the size is one frame.
    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        self.input.propose_allocation(caps)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.input.configure(absolute_caps, self.draw_label)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        // meta.emit refuses text_src on this one-output face
        Box::pin(async move { self.process_packet(packet, out).await.map(|_| ()) })
    }

    fn handles_qos(&self) -> bool {
        self.qos
    }

    // Elements upstream still do their work on every frame this one drops.
    fn relays_qos(&self) -> bool {
        true
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Python ML element host",
            "Filter/Effect/Video",
            "Hosts a gst-python-ml element shell as a g2g transform via embedded CPython.",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PYTRANSFORM_CLASS_PROPS.for_class(&self.input.module, &self.input.class)
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "draw-label" => {
                self.draw_label = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "qos" => {
                self.qos = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            g2g_core::OnlyOnGate::PROPERTY => self.only_on.set(&value),
            "format" => {
                let parsed = parse_raw_format(value.as_str().ok_or(PropError::Type)?)
                    .ok_or(PropError::Value)?;
                let Caps::RawVideo { format, .. } = &mut self.input.accept else {
                    return Err(PropError::Value);
                };
                *format = parsed;
                Ok(())
            }
            "output-caps" => {
                self.produce = Some(fixed_caps(value.as_str().ok_or(PropError::Type)?)?);
                Ok(())
            }
            other => self.input.set_property(other, value),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "draw-label" => Some(PropValue::Bool(self.draw_label)),
            "qos" => Some(PropValue::Bool(self.qos)),
            g2g_core::OnlyOnGate::PROPERTY => Some(self.only_on.value()),
            "format" => match &self.input.accept {
                Caps::RawVideo { format, .. } => {
                    Some(PropValue::Str(format.gst_name().to_string()))
                }
                _ => None,
            },
            "output-caps" => self
                .produce
                .as_ref()
                .map(|c| PropValue::Str(c.to_gst_string())),
            other => self.input.get_property(other),
        }
    }
}

impl PadTemplates for PyTransform {
    /// Advertise the default accepted format (RGBA, any geometry) on both pads
    /// for `gst-inspect` / autoplug. A `pyelement` is a same-format transform,
    /// so sink and source carry the same set. (`with_accept` can host another
    /// format programmatically; the launch template reflects the default.)
    fn pad_templates() -> Vec<PadTemplate> {
        let set = CapsSet::one(default_accept());
        Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
    }
}

static PYTRANSFORM_CLASS_PROPS: HostedClassProperties =
    HostedClassProperties::new(PYTRANSFORM_PROPS);

/// `PyTransform`'s settable properties (the runtime / `gst-launch` face).
static PYTRANSFORM_PROPS: &[PropertySpec] = hosted_element_props![
    MODULE_PROPERTY,
    CLASS_PROPERTY,
    PropertySpec::new(
        "draw-label",
        PropKind::Bool,
        "overlay the inferred label on the frame",
    )
    .with_default("false"),
    PropertySpec::new(
        "format",
        PropKind::Str,
        "pixel format the hosted element accepts (RGBA | BGRA | NV12 | I420 | YUY2 | P010_10LE)",
    )
    .with_default("RGBA"),
    CUDA_FRAMES_PROPERTY,
    PropertySpec::new(
        "qos",
        PropKind::Bool,
        "drop any frame a downstream sink's QoS report says would reach it late instead of running the hosted element on it",
    )
    .with_default("true"),
    g2g_core::OnlyOnGate::property_spec(
        "Run only on frames carrying this blob, or any detection when set to detections. Other frames pass through untouched.",
    ),
    INPUT_CAPS_PROPERTY,
    PropertySpec::new(
        "output-caps",
        PropKind::Str,
        "caps produced downstream when the hosted element changes media type, e.g. text/x-raw,format=utf8",
    ),
];

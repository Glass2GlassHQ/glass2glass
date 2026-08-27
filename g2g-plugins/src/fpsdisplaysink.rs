//! Frames-per-second display sink: wraps a real display sink and reports the
//! rate it is actually achieving, the gst `fpsdisplaysink` analog. Put it where
//! the display sink would go (`... ! fpsdisplaysink`) to find out whether a
//! pipeline keeps up, without changing what it does.
//!
//! The child comes from `video-sink`, a registered element name resolved through
//! the same alias chain `autovideosink` uses, so it lands on whatever display
//! sink this build has and on `fakesink` when it has none. gst takes a built
//! `GstElement` there instead, which a launch line cannot write.
//!
//! Every `fps-update-interval` milliseconds of buffers arriving, the counts and
//! rates go on the bus as [`BusMessage::Info`] and on the debug log, in gst's
//! own wording so the two are diffable. Like gst's, the clock is the arriving
//! data, not a timer: a stalled stream reports nothing rather than reporting
//! zero. `Eos` reports the run's maximum, minimum and average one last time.
//!
//! `frames-rendered` and `frames-dropped` are the child's own presentation
//! counters when it keeps them ([`AsyncElement::presentation_stats`], which is
//! where gst reads the sink's QoS stats); a child that counts nothing leaves
//! `frames-rendered` as the buffers handed to it and `frames-dropped` at zero.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::metrics::monotonic_ns;
use g2g_core::{
    g2g_debug, AsyncElement, BusHandle, BusMessage, Caps, CapsConstraint, ClockSync,
    ConfigureOutcome, ElementMetadata, G2gError, OutputSink, PipelinePacket, PresentationStats,
    PropError, PropKind, PropValue, PropertySpec,
};

use crate::registry::default_registry;

// A type alias, not a `use` of the trait, so the trait's methods stay out of
// scope: `FpsDisplaySink` implements `AsyncElement`, and the blanket
// `DynAsyncElement` impl would otherwise make `self.process` ambiguous. Calls on
// the child go through the fully-qualified path (as `SplitMuxSink` does).
type BoxedSink = Box<dyn g2g_core::element::DynAsyncElement>;

/// gst `fpsdisplaysink`'s `video-sink` default, the name it hands to the
/// registry. Resolved through the alias chain, so it is a real display sink
/// wherever one is built.
const DEFAULT_VIDEO_SINK: &str = "autovideosink";
/// The launch name of this element, which cannot also be its own child.
const OWN_LAUNCH_NAME: &str = "fpsdisplaysink";
/// The child every build has, when `video-sink` names nothing registered.
const FALLBACK_VIDEO_SINK: &str = "fakesink";
/// gst `fpsdisplaysink`'s `fps-update-interval` default, in milliseconds.
const DEFAULT_UPDATE_INTERVAL_MS: i64 = 500;
/// gst's bound on `fps-update-interval` (a signed 32-bit millisecond count, and
/// zero would report on every buffer).
const MIN_UPDATE_INTERVAL_MS: i64 = 1;
const MAX_UPDATE_INTERVAL_MS: i64 = i32::MAX as i64;
/// gst's "no measurement yet" value for `max-fps` / `min-fps`.
const NO_MEASUREMENT: f64 = -1.0;

const NS_PER_MS: u64 = 1_000_000;
const NS_PER_SECOND: f64 = 1_000_000_000.0;

/// # Example
///
/// ```no_run
/// use g2g_plugins::fpsdisplaysink::FpsDisplaySink;
///
/// // gst-launch equivalent: fpsdisplaysink video-sink=fakesink fps-update-interval=200
/// let sink = FpsDisplaySink::new();
/// ```
pub struct FpsDisplaySink {
    video_sink: String,
    child: BoxedSink,
    update_interval_ms: i64,
    silent: bool,
    bus: Option<BusHandle>,
    frames_rendered: u64,
    frames_dropped: u64,
    max_fps: f64,
    min_fps: f64,
    last_message: String,
    /// When the first buffer arrived, the origin the average is measured from.
    start_ns: Option<u64>,
    /// When the last report went out, and the counts it reported.
    last_report_ns: u64,
    last_rendered: u64,
    last_dropped: u64,
    log_name: LogName,
}

// `dyn DynAsyncElement` is not Debug, so implement it by hand (like SplitMuxSink).
impl core::fmt::Debug for FpsDisplaySink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FpsDisplaySink")
            .field("video_sink", &self.video_sink)
            .field("update_interval_ms", &self.update_interval_ms)
            .field("frames_rendered", &self.frames_rendered)
            .field("frames_dropped", &self.frames_dropped)
            .finish_non_exhaustive()
    }
}

impl Default for FpsDisplaySink {
    fn default() -> Self {
        Self::new()
    }
}

impl FpsDisplaySink {
    pub fn new() -> Self {
        Self {
            video_sink: DEFAULT_VIDEO_SINK.to_string(),
            child: make_child(DEFAULT_VIDEO_SINK).expect("the fakesink fallback is always built"),
            update_interval_ms: DEFAULT_UPDATE_INTERVAL_MS,
            silent: false,
            bus: None,
            frames_rendered: 0,
            frames_dropped: 0,
            max_fps: NO_MEASUREMENT,
            min_fps: NO_MEASUREMENT,
            last_message: String::new(),
            start_ns: None,
            last_report_ns: 0,
            last_rendered: 0,
            last_dropped: 0,
            log_name: LogName::new(),
        }
    }

    /// Attach the pipeline bus the reports are posted on. Without one they only
    /// reach the debug log.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Name the child display sink, as the `video-sink` property does. Panics on
    /// the one name that cannot be a child, this element's own.
    pub fn with_video_sink(mut self, name: &str) -> Self {
        self.set_child(name)
            .expect("a video-sink other than fpsdisplaysink itself");
        self
    }

    /// Replace the child with the named element, resolved through the registry.
    fn set_child(&mut self, name: &str) -> Result<(), PropError> {
        if name == OWN_LAUNCH_NAME {
            return Err(PropError::Value);
        }
        self.child = make_child(name).ok_or(PropError::Value)?;
        self.video_sink = name.to_string();
        Ok(())
    }

    /// Buffers the child reported rendering, or handed to it when it counts none.
    pub fn frames_rendered(&self) -> u64 {
        self.frames_rendered
    }

    /// Buffers the child reported dropping.
    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped
    }

    /// The most recent report, empty before the first one.
    pub fn last_message(&self) -> &str {
        &self.last_message
    }

    /// Adopt the child's own presentation counters where it keeps them, which is
    /// what gst does with the QoS stats a sink posts.
    fn read_child_counters(&mut self) {
        let Some(PresentationStats {
            presented,
            dropped,
            late_dropped,
        }) = g2g_core::element::DynAsyncElement::presentation_stats(self.child.as_ref())
        else {
            return;
        };
        self.frames_rendered = presented;
        self.frames_dropped = dropped + late_dropped;
    }

    /// Report the rate since the last report, gst's `display_current_fps`.
    fn report_current(&mut self, now_ns: u64) {
        self.read_child_counters();
        let start_ns = self.start_ns.unwrap_or(now_ns);
        let since_last = (now_ns.saturating_sub(self.last_report_ns)) as f64 / NS_PER_SECOND;
        let since_start = (now_ns.saturating_sub(start_ns)) as f64 / NS_PER_SECOND;
        if since_last <= 0.0 || since_start <= 0.0 {
            return;
        }
        let rendered_rate =
            (self.frames_rendered.saturating_sub(self.last_rendered)) as f64 / since_last;
        let drop_rate = (self.frames_dropped.saturating_sub(self.last_dropped)) as f64 / since_last;
        let average = self.frames_rendered as f64 / since_start;

        if self.max_fps == NO_MEASUREMENT || rendered_rate > self.max_fps {
            self.max_fps = rendered_rate;
        }
        if self.min_fps == NO_MEASUREMENT || rendered_rate < self.min_fps {
            self.min_fps = rendered_rate;
        }

        let message = if drop_rate == 0.0 {
            format!(
                "rendered: {}, dropped: {}, current: {rendered_rate:.2}, average: {average:.2}",
                self.frames_rendered, self.frames_dropped
            )
        } else {
            format!(
                "rendered: {}, dropped: {}, fps: {rendered_rate:.2}, drop rate: {drop_rate:.2}",
                self.frames_rendered, self.frames_dropped
            )
        };
        self.publish(message);

        self.last_rendered = self.frames_rendered;
        self.last_dropped = self.frames_dropped;
        self.last_report_ns = now_ns;
    }

    /// Report the whole run, gst's stop-time message.
    fn report_final(&mut self, now_ns: u64) {
        self.read_child_counters();
        let elapsed =
            (now_ns.saturating_sub(self.start_ns.unwrap_or(now_ns))) as f64 / NS_PER_SECOND;
        let average = if elapsed > 0.0 {
            self.frames_rendered as f64 / elapsed
        } else {
            0.0
        };
        self.publish(format!(
            "Max-fps: {:.2}, Min-fps: {:.2}, Average-fps: {average:.2}",
            self.max_fps, self.min_fps
        ));
    }

    /// Put one report on the log and the bus, and keep it as `last-message`.
    fn publish(&mut self, message: String) {
        g2g_debug!(self, "{}", message);
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::Info(message.clone()));
        }
        // gst leaves last-message alone while silent; the report still goes out.
        if !self.silent {
            self.last_message = message;
        }
    }

    /// Count an arriving buffer and report if the interval has passed.
    fn count(&mut self) {
        self.frames_rendered += 1;
        let now_ns = monotonic_ns();
        let start_ns = *self.start_ns.get_or_insert(now_ns);
        if self.last_report_ns == 0 {
            self.last_report_ns = start_ns;
        }
        let interval_ns = (self.update_interval_ms as u64).saturating_mul(NS_PER_MS);
        if now_ns.saturating_sub(self.last_report_ns) > interval_ns {
            self.report_current(now_ns);
        }
    }
}

/// Build the named child through the registry, resolving `autovideosink` and the
/// other aliases; `fakesink` stands in for a name this build does not have.
fn make_child(name: &str) -> Option<BoxedSink> {
    let registry = default_registry();
    registry
        .make_element(name)
        .or_else(|| registry.make_element(FALLBACK_VIDEO_SINK))
}

impl AsyncElement for FpsDisplaySink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "FpsDisplaySink",
            "Sink/Video",
            "Reports the frame rate a wrapped display sink achieves",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        g2g_core::element::DynAsyncElement::intercept_caps(self.child.as_ref(), upstream_caps)
    }

    /// The child decides what the pipeline may feed this.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        g2g_core::element::DynAsyncElement::caps_constraint_as_sink(self.child.as_ref())
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::element::DynAsyncElement::input_domains(self.child.as_ref())
    }

    fn propose_allocation(&self, caps: &Caps) -> Option<g2g_core::AllocationParams> {
        g2g_core::element::DynAsyncElement::propose_allocation(self.child.as_ref(), caps)
    }

    fn configure_allocation(&mut self, params: &g2g_core::AllocationParams) {
        g2g_core::element::DynAsyncElement::configure_allocation(self.child.as_mut(), params);
    }

    fn provide_clock(&self) -> Option<g2g_core::ClockCandidate> {
        g2g_core::element::DynAsyncElement::provide_clock(self.child.as_ref())
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        g2g_core::element::DynAsyncElement::set_clock_sync(self.child.as_mut(), sync);
    }

    fn presentation_stats(&self) -> Option<PresentationStats> {
        g2g_core::element::DynAsyncElement::presentation_stats(self.child.as_ref())
    }

    fn latency(&self) -> g2g_core::LatencyReport {
        g2g_core::element::DynAsyncElement::latency(self.child.as_ref())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        g2g_core::element::DynAsyncElement::configure_pipeline(self.child.as_mut(), absolute_caps)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let ends = matches!(packet, PipelinePacket::Eos);
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                self.count();
            }
            g2g_core::element::DynAsyncElement::process(self.child.as_mut(), packet, out).await?;
            if ends {
                self.report_final(monotonic_ns());
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        FPSDISPLAYSINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "video-sink" => {
                let wanted = value.as_str().ok_or(PropError::Type)?;
                self.set_child(wanted)
            }
            "fps-update-interval" => {
                let ms = value.as_int().ok_or(PropError::Type)?;
                if !(MIN_UPDATE_INTERVAL_MS..=MAX_UPDATE_INTERVAL_MS).contains(&ms) {
                    return Err(PropError::Value);
                }
                self.update_interval_ms = ms;
                Ok(())
            }
            "silent" => {
                self.silent = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "frames-rendered" | "frames-dropped" | "max-fps" | "min-fps" | "last-message" => {
                Err(PropError::ReadOnly)
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "video-sink" => Some(PropValue::Str(self.video_sink.clone())),
            "fps-update-interval" => Some(PropValue::Int(self.update_interval_ms)),
            "silent" => Some(PropValue::Bool(self.silent)),
            "frames-rendered" => Some(PropValue::Uint(self.frames_rendered)),
            "frames-dropped" => Some(PropValue::Uint(self.frames_dropped)),
            "max-fps" => Some(PropValue::Double(self.max_fps)),
            "min-fps" => Some(PropValue::Double(self.min_fps)),
            "last-message" => Some(PropValue::Str(self.last_message.clone())),
            _ => None,
        }
    }
}

impl LogSource for FpsDisplaySink {
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

/// `FpsDisplaySink`'s properties, named and defaulted as gst `fpsdisplaysink`.
static FPSDISPLAYSINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "video-sink",
        PropKind::Str,
        "registered name of the display sink to wrap",
    )
    .with_default(DEFAULT_VIDEO_SINK),
    PropertySpec::new(
        "fps-update-interval",
        PropKind::Int,
        "milliseconds of arriving data between reports",
    )
    .with_range("1", "2147483647")
    .with_default("500"),
    PropertySpec::new(
        "silent",
        PropKind::Bool,
        "stop keeping the report in last-message",
    )
    .with_default("false"),
    PropertySpec::new("frames-rendered", PropKind::Uint, "buffers rendered so far")
        .with_default("0")
        .read_only(),
    PropertySpec::new("frames-dropped", PropKind::Uint, "buffers dropped so far")
        .with_default("0")
        .read_only(),
    PropertySpec::new(
        "max-fps",
        PropKind::Double,
        "highest rate measured, -1 if none",
    )
    .with_default("-1")
    .read_only(),
    PropertySpec::new(
        "min-fps",
        PropKind::Double,
        "lowest rate measured, -1 if none",
    )
    .with_default("-1")
    .read_only(),
    PropertySpec::new("last-message", PropKind::Str, "the most recent report").read_only(),
];

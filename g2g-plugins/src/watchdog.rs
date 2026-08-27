//! Watchdog: fails the pipeline when no packet crosses the element for
//! `timeout` milliseconds, the gst `watchdog` analog. Put it behind a live
//! source so a feed that goes silent tears the run down instead of hanging.
//!
//! Two halves, because the runner calls a transform only when a packet arrives
//! and delivers `PipelinePacket::Tick` to fan-in arms alone:
//!
//! - a timer task (`tokio`, hence the `std` gate) sleeps on the deadline and,
//!   when it passes, posts [`BusMessage::Error`] so an application watching the
//!   bus tears the pipeline down. This is what fires during a stall, and it is
//!   what gst's watchdog does.
//! - `process` compares the wall clock against the last packet's arrival, so a
//!   stall that ends still fails the run with [`G2gError::Timeout`], and so the
//!   timeout is applied even on an executor that hands out no task spawner.
//!
//! Every packet re-arms the timer, `Eos` disarms it.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use core::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::metrics::monotonic_ns;
use g2g_core::{
    g2g_debug, g2g_error, AsyncElement, BusHandle, BusMessage, Caps, CapsConstraint,
    ConfigureOutcome, ElementMetadata, G2gError, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

/// gst `watchdog`'s `timeout` default, in milliseconds.
const DEFAULT_TIMEOUT_MS: u64 = 1000;

const NS_PER_MS: u64 = 1_000_000;

/// What the timer task and `process` share: when the last packet arrived, and
/// whether the stream is still running.
#[derive(Debug, Default)]
struct Deadline {
    /// `monotonic_ns` of the last packet.
    last_packet_ns: AtomicU64,
    /// Cleared at `Eos`, so the timer stops watching a stream that ended.
    armed: AtomicBool,
    /// Set by the timer when the deadline passed.
    expired: AtomicBool,
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::watchdog::Watchdog;
///
/// // gst-launch equivalent: watchdog timeout=2000
/// let element = Watchdog::new().with_timeout_ms(2000);
/// ```
#[derive(Debug)]
pub struct Watchdog {
    timeout_ms: u64,
    bus: Option<BusHandle>,
    deadline: Arc<Deadline>,
    timer: Option<tokio::task::JoinHandle<()>>,
    configured: bool,
    log_name: LogName,
}

impl Default for Watchdog {
    fn default() -> Self {
        Self::new()
    }
}

impl Watchdog {
    pub fn new() -> Self {
        Self {
            timeout_ms: DEFAULT_TIMEOUT_MS,
            bus: None,
            deadline: Arc::new(Deadline::default()),
            timer: None,
            configured: false,
            log_name: LogName::new(),
        }
    }

    /// Milliseconds without a packet before the run fails; `0` disables.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Attach the pipeline bus, so an expired deadline reaches the application
    /// as a [`BusMessage::Error`] while the stream is still stalled.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    fn timeout_ns(&self) -> u64 {
        self.timeout_ms.saturating_mul(NS_PER_MS)
    }

    /// Start the timer task, once. Without a current runtime there is nothing to
    /// sleep on and only the arrival check applies.
    fn start_timer(&mut self) {
        if self.timer.is_some() || self.timeout_ms == 0 {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            g2g_debug!(
                self,
                "no async runtime here, so the timeout only applies as packets arrive"
            );
            return;
        };
        let deadline = self.deadline.clone();
        let bus = self.bus.clone();
        let timeout_ns = self.timeout_ns();
        self.timer = Some(runtime.spawn(async move {
            while deadline.armed.load(Ordering::Acquire) {
                let waited =
                    monotonic_ns().saturating_sub(deadline.last_packet_ns.load(Ordering::Acquire));
                let Some(remaining) = timeout_ns.checked_sub(waited) else {
                    deadline.expired.store(true, Ordering::Release);
                    if let Some(bus) = &bus {
                        bus.try_post(BusMessage::Error(G2gError::Timeout));
                    }
                    return;
                };
                tokio::time::sleep(Duration::from_nanos(remaining)).await;
            }
        }));
    }

    fn arm(&mut self) {
        self.deadline
            .last_packet_ns
            .store(monotonic_ns(), Ordering::Release);
        self.deadline.armed.store(true, Ordering::Release);
    }

    fn disarm(&mut self) {
        self.deadline.armed.store(false, Ordering::Release);
        if let Some(timer) = self.timer.take() {
            timer.abort();
        }
    }

    /// Whether the stream stalled longer than the timeout allows.
    fn expired(&self) -> bool {
        if self.timeout_ms == 0 {
            return false;
        }
        if self.deadline.expired.load(Ordering::Acquire) {
            return true;
        }
        let waited =
            monotonic_ns().saturating_sub(self.deadline.last_packet_ns.load(Ordering::Acquire));
        waited >= self.timeout_ns()
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        if let Some(timer) = self.timer.take() {
            timer.abort();
        }
    }
}

impl AsyncElement for Watchdog {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Watchdog",
            "Generic",
            "Fails the pipeline when no data arrives within a timeout",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Wildcard pass-through: a watchdog watches the clock, not the format.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        self.arm();
        self.start_timer();
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            if self.expired() {
                g2g_error!(self, "no data for {} ms", self.timeout_ms);
                if let Some(bus) = &self.bus {
                    bus.try_post(BusMessage::Error(G2gError::Timeout));
                }
                self.disarm();
                return Err(G2gError::Timeout);
            }
            match packet {
                PipelinePacket::Eos => self.disarm(),
                other => {
                    self.arm();
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        WATCHDOG_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "timeout" => {
                let ms = value.as_uint().ok_or(PropError::Type)?;
                if ms > MAX_TIMEOUT_MS {
                    return Err(PropError::Value);
                }
                self.timeout_ms = ms;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "timeout" => Some(PropValue::Uint(self.timeout_ms)),
            _ => None,
        }
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }
}

impl LogSource for Watchdog {
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

/// Largest `timeout` accepted, gst `watchdog`'s own upper bound (its `timeout`
/// is a signed 32-bit int).
const MAX_TIMEOUT_MS: u64 = i32::MAX as u64;

/// `Watchdog`'s settable properties, named and defaulted as gst `watchdog`.
static WATCHDOG_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "timeout",
    PropKind::Uint,
    "milliseconds without data before the run fails (0 = disabled)",
)
.with_range("0", "2147483647")
.with_default("1000")];

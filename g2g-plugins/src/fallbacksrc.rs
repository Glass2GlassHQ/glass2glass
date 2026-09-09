//! The source half of `fallbacksrc` (M1163): a wrapper that rebuilds the URI
//! source behind it when that source fails, delivers nothing for
//! `restart-timeout`, or (under `restart-on-eos`) ends. Each life of the inner
//! source is stitched onto one timeline through the
//! `ShiftSink` adapter shared with `gaplesssrc`, so a rebuilt file source that
//! starts again at PTS 0 continues where the last life stopped, and the inner
//! `Eos` packets are swallowed: the wrapper's own `Eos` is the only terminal one.
//!
//! Registered on the [`Registry`](g2g_core::runtime::Registry) as its
//! [`RestartSourceHook`](g2g_core::runtime::RestartSourceHook), so the
//! `fallbacksrc uri=X` launch keyword wraps both its main and its `fallback-uri`
//! source with it.
//!
//! With an [`UnblockHandle`] set (M1166, gst's
//! `manual-unblock`), each life waits for the application to release it before
//! it delivers anything, and the wait re-arms on every restart.
//!
//! Semantics follow gst's `fallbacksrc`: a fixed one-second pause before a
//! rebuild, `restart-timeout` as a stall timer on the running source, and
//! `retry-timeout` as the budget for repeated failure, measured from the first
//! failure after the last delivered frame: a rebuild starts only while it would
//! begin inside the budget. gst stores `retry-timeout` but never arms its timer;
//! here it is enforced, and once it runs out the wrapper stops retrying and ends
//! its stream, which is what hands the switch to the fallback for good.
//!
//! Each life start, retry decision and stream end is posted as
//! [`BusMessage::SourceRestart`](g2g_core::BusMessage), gst's
//! read-only `status` and `statistics`: a source arm owns its element for the
//! whole run, so nothing can read a property off it mid-run.

use core::cell::Cell;
use core::fmt;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::time::{Duration, Instant};

use alloc::boxed::Box;
use alloc::string::{String, ToString};

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::query::LatencyReport;
use g2g_core::runtime::{
    select2, DynSourceLoop, Either, FallbackSourceRole, RestartPolicy, SourceLoop, UnblockHandle,
    UriError, UriRebuild,
};
use g2g_core::{
    g2g_info, g2g_warn, BusHandle, BusMessage, Caps, ConfigureOutcome, G2gError, OutputSink,
    PipelinePacket, PushOutcome, SourceRestartReason, SourceRestartStatus,
};

use crate::gaplesssrc::ShiftSink;

/// The pause between a source's death and its rebuild, gst's hardcoded sleep.
pub const RETRY_DELAY: Duration = Duration::from_secs(1);

/// The [`RestartSourceHook`](g2g_core::runtime::RestartSourceHook) registered by
/// the default registry.
pub fn restart_source(
    source: Box<dyn DynSourceLoop>,
    rebuild: UriRebuild,
    policy: RestartPolicy,
    role: FallbackSourceRole,
    unblock: Option<UnblockHandle>,
) -> Box<dyn DynSourceLoop> {
    let wrapper = RestartSrc::new(source, rebuild, policy, Some(role));
    Box::new(match unblock {
        Some(handle) => wrapper.with_unblock_handle(handle),
        None => wrapper,
    })
}

/// A source that rebuilds the URI source it wraps when that source dies. See the
/// module docs. Negotiation and the first configure go through the source given
/// at construction; every rebuilt source is negotiated by the wrapper itself and
/// configured with the caps the wrapper was configured with.
pub struct RestartSrc {
    /// The source about to run. Taken by `run`, which then rebuilds successors.
    current: Option<Box<dyn DynSourceLoop>>,
    rebuild: UriRebuild,
    policy: RestartPolicy,
    /// The caps the runner configured the wrapper with, reused for every rebuild.
    caps: Option<Caps>,
    /// Which `fallbacksrc` source this is, for the bus report. `None` when it was
    /// built directly rather than by the launch keyword.
    role: Option<FallbackSourceRole>,
    /// The application's release control (M1166): with one set, every life waits
    /// for an `unblock` before it delivers.
    unblock: Option<UnblockHandle>,
    bus: Option<BusHandle>,
    log_name: LogName,
}

impl fmt::Debug for RestartSrc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RestartSrc")
            .field("has_current", &self.current.is_some())
            .field("policy", &self.policy)
            .field("caps", &self.caps)
            .field("role", &self.role)
            .field("manual_unblock", &self.unblock.is_some())
            .finish_non_exhaustive()
    }
}

impl RestartSrc {
    pub fn new(
        source: Box<dyn DynSourceLoop>,
        rebuild: UriRebuild,
        policy: RestartPolicy,
        role: Option<FallbackSourceRole>,
    ) -> Self {
        Self {
            current: Some(source),
            rebuild,
            policy,
            caps: None,
            role,
            unblock: None,
            bus: None,
            log_name: LogName::new(),
        }
    }

    /// Hold every life until the application releases it through `handle`
    /// (M1166, gst's `manual-unblock`). The handle re-arms, so a restarted
    /// source waits again.
    pub fn with_unblock_handle(mut self, handle: UnblockHandle) -> Self {
        self.unblock = Some(handle);
        self
    }

    /// Report this source's state to the application, dropping the message when
    /// the bus is full rather than holding up the stream.
    fn report(
        &self,
        status: SourceRestartStatus,
        retries: u64,
        reason: Option<SourceRestartReason>,
    ) {
        let Some(bus) = &self.bus else {
            return;
        };
        bus.try_post(BusMessage::SourceRestart {
            element: self
                .log_name
                .instance()
                .unwrap_or(short_type_name::<Self>())
                .to_string(),
            role: self.role,
            status,
            retries,
            reason,
        });
    }
}

/// Why the running source stopped, for the log line and the retry accounting.
#[derive(Debug)]
enum Death {
    /// `run` returned an error.
    Error(G2gError),
    /// `run` returned cleanly and `restart-on-eos` asked for another life.
    Eos,
    /// Nothing was pushed for `restart-timeout`.
    Stall,
    /// The rebuild itself failed before the source could run.
    Rebuild(UriError),
    /// The rebuilt source refused to negotiate or to take the configured caps.
    Negotiate(G2gError),
}

impl Death {
    fn reason(&self) -> SourceRestartReason {
        match self {
            Death::Error(_) => SourceRestartReason::Error,
            Death::Eos => SourceRestartReason::Eos,
            Death::Stall => SourceRestartReason::Timeout,
            Death::Rebuild(_) => SourceRestartReason::Rebuild,
            Death::Negotiate(_) => SourceRestartReason::Negotiate,
        }
    }
}

impl fmt::Display for Death {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Death::Error(e) => write!(f, "run failed: {e:?}"),
            Death::Eos => f.write_str("ended"),
            Death::Stall => f.write_str("stalled"),
            Death::Rebuild(e) => write!(f, "rebuild failed: {e:?}"),
            Death::Negotiate(e) => write!(f, "rebuilt source refused the caps: {e:?}"),
        }
    }
}

/// When the inner source last handed a packet down, and whether that push is
/// still waiting on downstream. A push blocked on backpressure is not a stall.
#[derive(Debug, Clone, Copy)]
struct Activity {
    last: Instant,
    pushing: bool,
}

/// Records each push's start and completion in the shared `Activity`, so the
/// stall watchdog sees the inner source's liveness through `ShiftSink`.
struct AliveSink<'o> {
    out: &'o mut dyn OutputSink,
    activity: &'o Cell<Activity>,
}

impl OutputSink for AliveSink<'_> {
    fn begin_push(&mut self) {
        self.activity.set(Activity {
            last: Instant::now(),
            pushing: true,
        });
        self.out.begin_push();
    }

    fn poll_push(
        &mut self,
        cx: &mut Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        let result = self.out.poll_push(cx, packet);
        if result.is_ready() {
            self.activity.set(Activity {
                last: Instant::now(),
                pushing: false,
            });
        }
        result
    }
}

/// Resolves once `activity` has been idle for `timeout`; never, for a zero
/// timeout. A push still waiting on downstream keeps the source alive.
async fn stalled(activity: &Cell<Activity>, timeout: Duration) {
    if timeout.is_zero() {
        core::future::pending::<()>().await;
    }
    loop {
        let Activity { last, pushing } = activity.get();
        let deadline = last + timeout;
        let now = Instant::now();
        if !pushing && now >= deadline {
            return;
        }
        let wait = if pushing { timeout } else { deadline - now };
        tokio::time::sleep(wait).await;
    }
}

/// Build a fresh source and bring it to the configured caps.
async fn rebuild_configured(
    rebuild: &UriRebuild,
    caps: &Caps,
) -> Result<Box<dyn DynSourceLoop>, Death> {
    let (mut source, _declared) = rebuild().map_err(Death::Rebuild)?;
    source.intercept_caps().await.map_err(Death::Negotiate)?;
    source.configure_pipeline(caps).map_err(Death::Negotiate)?;
    Ok(source)
}

impl SourceLoop for RestartSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        Box::pin(async move {
            let src = self.current.as_mut().ok_or(G2gError::NotConfigured)?;
            src.intercept_caps().await
        })
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let src = self.current.as_mut().ok_or(G2gError::NotConfigured)?;
        let outcome = src.configure_pipeline(absolute_caps)?;
        self.caps = Some(absolute_caps.clone());
        Ok(outcome)
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }

    fn set_bus(&mut self, bus: BusHandle) {
        self.bus = Some(bus);
    }

    fn latency(&self) -> LatencyReport {
        self.current
            .as_ref()
            .map_or(LatencyReport::ZERO, |src| src.latency())
    }

    fn output_memory(&self) -> MemoryDomainKind {
        self.current
            .as_ref()
            .map_or(MemoryDomainKind::System, |src| src.output_memory())
    }

    fn output_domains(&self) -> DomainSet {
        self.current
            .as_ref()
            .map_or(DomainSet::only(MemoryDomainKind::System), |src| {
                src.output_domains()
            })
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let caps = self.caps.clone().ok_or(G2gError::NotConfigured)?;
            let stall_timeout = Duration::from_nanos(self.policy.restart_timeout_ns);
            let retry_timeout = Duration::from_nanos(self.policy.retry_timeout_ns);
            let mut total = 0u64;
            // Running-time offset for the next life, so the stream stays one
            // monotonic timeline across rebuilds.
            let mut offset = 0u64;
            // The first life is the runner-configured source; each later one is
            // rebuilt here.
            let mut current = self.current.take();
            // When the current run of failures began, cleared by a delivered frame.
            let mut first_failure: Option<Instant> = None;
            // Rebuilds attempted so far, gst's `num-retry`.
            let mut retries = 0u64;
            loop {
                let source = match current.take() {
                    Some(source) => Ok(source),
                    None => rebuild_configured(&self.rebuild, &caps).await,
                };
                let death = match source {
                    Err(death) => death,
                    Ok(mut source) => {
                        if let Some(unblock) = &self.unblock {
                            unblock.wait_release().await;
                        }
                        self.report(SourceRestartStatus::Running, retries, None);
                        let activity = Cell::new(Activity {
                            last: Instant::now(),
                            pushing: false,
                        });
                        // The adapters are scoped so their borrow on `out` ends
                        // before the terminal `Eos` below; the counts are copied
                        // out because a stall drops the inner run future.
                        let (outcome, frames, end) = {
                            let mut alive = AliveSink {
                                out: &mut *out,
                                activity: &activity,
                            };
                            let mut adapter = ShiftSink::new(&mut alive, offset);
                            let outcome = select2(
                                source.run(&mut adapter),
                                stalled(&activity, stall_timeout),
                            )
                            .await;
                            (outcome, adapter.frames, adapter.max_end)
                        };
                        total = total.saturating_add(frames);
                        offset = end;
                        if frames > 0 {
                            first_failure = None;
                        }
                        match outcome {
                            Either::Left(Ok(_)) if !self.policy.restart_on_eos => {
                                self.report(SourceRestartStatus::Stopped, retries, None);
                                out.push(PipelinePacket::Eos).await?;
                                return Ok(total);
                            }
                            Either::Left(Ok(_)) => Death::Eos,
                            Either::Left(Err(e)) => Death::Error(e),
                            Either::Right(()) => Death::Stall,
                        }
                    }
                };
                let now = Instant::now();
                let since = *first_failure.get_or_insert(now);
                // A rebuild only starts inside the budget, so the wait before it
                // counts against it too.
                let next_attempt_at = now.duration_since(since) + RETRY_DELAY;
                if next_attempt_at >= retry_timeout {
                    g2g_warn!(
                        self,
                        "source {death}; retry budget of {retry_timeout:?} spent, ending the stream"
                    );
                    self.report(SourceRestartStatus::Stopped, retries, Some(death.reason()));
                    out.push(PipelinePacket::Eos).await?;
                    return Ok(total);
                }
                retries += 1;
                g2g_info!(self, "source {death}; rebuilding in {RETRY_DELAY:?}");
                self.report(SourceRestartStatus::Retrying, retries, Some(death.reason()));
                tokio::time::sleep(RETRY_DELAY).await;
            }
        })
    }
}

impl LogSource for RestartSrc {
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

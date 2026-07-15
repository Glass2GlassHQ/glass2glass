//! Application sink (`appsink`): the application receives buffers out of a
//! running pipeline (M233 callback, M235 pull), the `gst-appsink` analog.
//!
//! Two delivery modes, selected by what the application registers under the
//! sink's `channel` name *before* launch (the parameterless `fn` factory means
//! the element is handed neither directly, so both are parked in a named
//! global and claimed at `configure_pipeline`):
//!
//! - **Callback** ([`set_appsink_callback`], the GStreamer `new-sample` model):
//!   the element invokes the callback per frame on the run thread with a
//!   borrowed view; copy if you need to keep it. EOS is `data == null, len == 0`.
//! - **Pull** ([`register_appsink_pull`]): the element hands each whole [`Frame`]
//!   to a bounded channel and the application pulls it ([`AppSinkPull`]). The
//!   pulled frame *owns* its bytes (zero-copy: the same `SystemSlice`, including
//!   an `appsrc` foreign lend), valid until the application drops it. A full
//!   channel backpressures the pipeline, the correct slow-consumer behaviour.

use core::ffi::c_void;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use spin::Mutex;

use g2g_core::frame::Frame;
use g2g_core::runtime::{bounded, Receiver, Sender};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, MemoryDomain,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

/// Bounded depth of the element -> application pull channel. A full channel
/// backpressures the pipeline (the slow-consumer behaviour appsink wants).
const PULL_DEPTH: usize = 8;

/// The C callback shape: `(data, len, pts_ns, user)`. `data == null` / `len == 0`
/// signals end-of-stream.
pub type SampleCallback = extern "C" fn(*const u8, usize, u64, *mut c_void);

/// A registered callback plus its opaque user pointer.
struct Slot {
    cb: SampleCallback,
    user: *mut c_void,
}

// SAFETY: the application guarantees (documented C contract) that `cb` and
// `user` are safe to invoke from the pipeline's run thread. The pointers are
// only ever called, never dereferenced by this crate.
unsafe impl Send for Slot {}

impl core::fmt::Debug for Slot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Slot").finish_non_exhaustive()
    }
}

/// One item on the pull channel: a frame or the end-of-stream marker.
#[derive(Debug)]
enum Pulled {
    Frame(Frame),
    Eos,
}

/// How an `appsink channel=<name>` delivers: invoke a callback, or feed a pull
/// channel. Set by [`set_appsink_callback`] / [`register_appsink_pull`].
#[derive(Debug)]
enum SinkMode {
    Callback(Slot),
    Pull(Sender<Pulled>),
}

/// Named delivery modes, keyed by the `appsink channel` property; claimed once
/// by the element at startup. The bridge the parameterless `fn` factory forces.
static SINKS: Mutex<BTreeMap<String, SinkMode>> = Mutex::new(BTreeMap::new());

/// Register the per-frame callback for `appsink channel=<channel>`. Call before
/// launching. Replaces any prior registration under the same name.
pub fn set_appsink_callback(channel: &str, cb: SampleCallback, user: *mut c_void) {
    SINKS.lock().insert(channel.to_string(), SinkMode::Callback(Slot { cb, user }));
}

/// Register `appsink channel=<channel>` in pull mode and return the
/// application's pull handle. Call before launching.
pub fn register_appsink_pull(channel: &str) -> AppSinkPull {
    let (tx, rx) = bounded::<Pulled>(PULL_DEPTH);
    SINKS.lock().insert(channel.to_string(), SinkMode::Pull(tx));
    AppSinkPull { rx }
}

/// Outcome of a non-blocking [`AppSinkPull::try_pull`].
#[derive(Debug)]
pub enum Pull {
    /// A frame is ready.
    Frame(Frame),
    /// No frame pending yet (the pipeline is still running).
    Empty,
    /// The stream has ended; no more frames will arrive.
    Ended,
}

/// The application's pull handle for an `appsink`. Dropping it closes the pull
/// channel; the element then drops frames it cannot deliver.
#[derive(Debug)]
pub struct AppSinkPull {
    rx: Receiver<Pulled>,
}

impl AppSinkPull {
    /// Non-blocking: return the next frame if one is queued.
    pub fn try_pull(&self) -> Pull {
        match self.rx.try_recv() {
            Some(Pulled::Frame(f)) => Pull::Frame(f),
            Some(Pulled::Eos) => Pull::Ended,
            None => Pull::Empty,
        }
    }

    /// Await the next frame; `None` once the stream ends (EOS) or the pipeline
    /// is gone. The application drives this to completion (e.g. a `block_on` on
    /// its own thread) while the pipeline runs on another.
    pub async fn pull(&self) -> Option<Frame> {
        match self.rx.recv().await {
            Some(Pulled::Frame(f)) => Some(f),
            _ => None,
        }
    }
}

/// Application pull/callback sink. Accepts any caps.
#[derive(Debug, Default)]
pub struct AppSink {
    channel: String,
    configured: bool,
    mode: Option<SinkMode>,
    received: u64,
}

impl AppSink {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the delivery `channel` name programmatically (the builder path; the
    /// launch / registry path uses the `channel=` property). Must match the name
    /// passed to [`register_appsink_pull`] / [`set_appsink_callback`].
    pub fn with_channel(mut self, channel: impl Into<String>) -> Self {
        self.channel = channel.into();
        self
    }

    fn channel_name(&self) -> &str {
        if self.channel.is_empty() {
            "default"
        } else {
            &self.channel
        }
    }

    /// Frames delivered so far.
    pub fn received(&self) -> u64 {
        self.received
    }
}

impl AsyncElement for AppSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Claim the registered delivery mode once, and only once: a format- or
        // size-changing upstream transform makes the runner cascade caps a
        // second time, calling `configure_pipeline` again. The claim removes the
        // entry from the global, so a re-configure must not run it again or it
        // would clobber the already-claimed `tx`/callback with `None` and then
        // silently drop every frame (and never forward EOS).
        if self.mode.is_none() {
            self.mode = SINKS.lock().remove(self.channel_name());
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Application sink",
            "Sink",
            "Delivers buffers to the application via callback or pull (g2g_appsink_*)",
            "g2g",
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
                    self.received += 1;
                    match &self.mode {
                        // Callback: deliver a borrowed view of host-visible
                        // memory (GPU-resident frames need a download the v1
                        // path skips; the count still advances).
                        Some(SinkMode::Callback(slot)) => match &f.domain {
                            MemoryDomain::System(s) => {
                                let b = s.as_slice();
                                (slot.cb)(b.as_ptr(), b.len(), f.timing.pts_ns, slot.user);
                            }
                            MemoryDomain::SystemView(sv) => {
                                let b = sv.materialize();
                                (slot.cb)(b.as_ptr(), b.len(), f.timing.pts_ns, slot.user);
                            }
                            _ => {}
                        },
                        // Pull: hand the whole frame over (zero-copy); awaiting
                        // a full channel backpressures the pipeline. A closed
                        // channel (app dropped its handle) drops the frame.
                        Some(SinkMode::Pull(tx)) => {
                            let _ = tx.send(Pulled::Frame(f)).await;
                        }
                        None => {}
                    }
                }
                PipelinePacket::Eos => match &self.mode {
                    Some(SinkMode::Callback(slot)) => {
                        (slot.cb)(core::ptr::null(), 0, 0, slot.user);
                    }
                    Some(SinkMode::Pull(tx)) => {
                        let _ = tx.send(Pulled::Eos).await;
                    }
                    None => {}
                },
                // Control packets are not surfaced to the application in v1.
                PipelinePacket::Flush
                | PipelinePacket::CapsChanged(_)
                | PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        APPSINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "channel" => {
                self.channel = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }
}

static APPSINK_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "channel",
    PropKind::Str,
    "delivery name matching set_appsink_callback / register_appsink_pull (default \"default\")",
)];

impl PadTemplates for AppSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

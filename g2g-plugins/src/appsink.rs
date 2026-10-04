//! Application sink (`appsink`): the application receives buffers out of a
//! running pipeline (M233 callback, M235 pull), the `gst-appsink` analog.
//!
//! Two delivery modes, selected by what the application registers under the
//! sink's `channel` name *before* launch (the parameterless `fn` factory means
//! the element is handed neither directly, so both are parked in a named
//! global). The element claims its registration when its channel is set, or at
//! `configure_pipeline` if none was registered yet, and dropping the element
//! ends the pull, so a graph that fails before it configures still ends it:
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
use core::sync::atomic::{AtomicBool, Ordering};

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use spin::Mutex;

use crate::capsfilter::parse_caps_set;

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::runtime::{bounded, Receiver, Sender};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
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
    SINKS
        .lock()
        .insert(channel.to_string(), SinkMode::Callback(Slot { cb, user }));
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
        // read before try_recv, a last frame sent in between must not read as Ended
        let closed = self.rx.is_closed();
        match self.rx.try_recv() {
            Some(Pulled::Frame(f)) => Pull::Frame(f),
            Some(Pulled::Eos) => Pull::Ended,
            None if closed => Pull::Ended,
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

/// Application pull/callback sink. Accepts any caps unless narrowed by the `caps`
/// property, and every memory domain unless narrowed with
/// [`with_input_domains`](AppSink::with_input_domains).
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::appsink::{register_appsink_pull, AppSink};
///
/// let pull = register_appsink_pull("frames");
/// let sink = AppSink::new().with_channel("frames");
/// let next = pull.try_pull();
/// ```
#[derive(Debug)]
pub struct AppSink {
    channel: String,
    caps: Option<(String, CapsSet)>,
    input_domains: DomainSet,
    configured: bool,
    negotiated: AtomicBool,
    mode: Option<SinkMode>,
    received: u64,
}

impl Default for AppSink {
    fn default() -> Self {
        Self {
            channel: String::new(),
            caps: None,
            input_domains: DomainSet::ALL,
            configured: false,
            negotiated: AtomicBool::new(false),
            mode: None,
            received: 0,
        }
    }
}

impl AppSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_input_domains(mut self, domains: DomainSet) -> Self {
        self.input_domains = domains;
        self
    }

    /// Set the delivery `channel` name programmatically (the builder path; the
    /// launch / registry path uses the `channel=` property). Must match the name
    /// passed to [`register_appsink_pull`] / [`set_appsink_callback`].
    pub fn with_channel(mut self, channel: impl Into<String>) -> Self {
        self.set_channel(channel.into());
        self
    }

    fn set_channel(&mut self, channel: String) {
        self.channel = channel;
        self.mode = SINKS.lock().remove(self.channel_name());
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

impl Drop for AppSink {
    fn drop(&mut self) {
        // registry probes build and drop a default appsink, those must not end a live registration
        if self.mode.is_none() && self.negotiated.load(Ordering::Relaxed) {
            SINKS.lock().remove(self.channel_name());
        }
    }
}

impl AsyncElement for AppSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        let Some((_, set)) = &self.caps else {
            return Ok(upstream_caps.clone());
        };
        set.alternatives()
            .iter()
            .find_map(|alternative| upstream_caps.intersect(alternative).ok())
            .ok_or(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        self.negotiated.store(true, Ordering::Relaxed);
        match &self.caps {
            Some((_, set)) => CapsConstraint::Accepts(set.clone()),
            None => CapsConstraint::AcceptsAny,
        }
    }

    fn input_domains(&self) -> DomainSet {
        self.input_domains
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Claim the registered delivery mode here only if setting the channel
        // found none (a default channel name, or a registration made after the
        // element was built). A format- or size-changing upstream transform
        // makes the runner cascade caps a second time, calling
        // `configure_pipeline` again. The claim removes the entry from the
        // global, so a re-configure must not run it again or it would clobber
        // the already-claimed `tx`/callback with `None` and then silently drop
        // every frame (and never forward EOS).
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
                self.set_channel(value.as_str().ok_or(PropError::Type)?.to_string());
                Ok(())
            }
            "caps" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                let set = parse_caps_set(text).ok_or(PropError::Value)?;
                if set.alternatives().is_empty() {
                    return Err(PropError::Value);
                }
                self.caps = Some((text.to_string(), set));
                Ok(())
            }
            "input-domains" => {
                self.input_domains = parse_domains(value.as_str().ok_or(PropError::Type)?)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "channel" => Some(PropValue::Str(self.channel_name().to_string())),
            "caps" => self
                .caps
                .as_ref()
                .map(|(text, _)| PropValue::Str(text.clone())),
            "input-domains" => Some(PropValue::Str(domain_names(self.input_domains))),
            _ => None,
        }
    }
}

const MEMORY_DOMAIN_NAMES: &[(&str, MemoryDomainKind)] = &[
    ("system", MemoryDomainKind::System),
    ("systemview", MemoryDomainKind::SystemView),
    ("dmabuf", MemoryDomainKind::DmaBuf),
    ("vulkantexture", MemoryDomainKind::VulkanTexture),
    ("webgpubuffer", MemoryDomainKind::WebGPUBuffer),
    ("cuda", MemoryDomainKind::Cuda),
    ("d3d11texture", MemoryDomainKind::D3D11Texture),
    ("cvpixelbuffer", MemoryDomainKind::CvPixelBuffer),
    (
        "webgpuexternaltexture",
        MemoryDomainKind::WebGPUExternalTexture,
    ),
    ("wgputexture", MemoryDomainKind::WgpuTexture),
    ("wgpubuffer", MemoryDomainKind::WgpuBuffer),
];

fn parse_domains(names: &str) -> Result<DomainSet, PropError> {
    names.split(',').try_fold(DomainSet::EMPTY, |set, name| {
        let (_, kind) = MEMORY_DOMAIN_NAMES
            .iter()
            .find(|(known, _)| *known == name.trim())
            .ok_or(PropError::Value)?;
        Ok(set.with(*kind))
    })
}

fn domain_names(set: DomainSet) -> String {
    let names: Vec<&str> = set
        .iter()
        .filter_map(|kind| {
            MEMORY_DOMAIN_NAMES
                .iter()
                .find(|(_, known)| *known == kind)
                .map(|(name, _)| *name)
        })
        .collect();
    names.join(",")
}

static APPSINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "channel",
        PropKind::Str,
        "delivery name matching set_appsink_callback / register_appsink_pull (default \"default\")",
    ),
    PropertySpec::new(
        "caps",
        PropKind::Str,
        "caps to accept, gst-launch syntax (default any)",
    ),
    PropertySpec::new(
        "input-domains",
        PropKind::Str,
        "comma-separated set of memory domains to accept, e.g. dmabuf,system (default every domain)",
    ),
];

impl PadTemplates for AppSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

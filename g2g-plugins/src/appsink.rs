//! Application sink (`appsink`): the application receives buffers out of a
//! running pipeline (M233), the `gst-appsink` analog.
//!
//! v1 is the GStreamer `new-sample`-callback model: register a callback under a
//! channel name *before* launch, and the `appsink channel=<name>` element
//! invokes it once per frame, on the pipeline's run thread, with a borrowed view
//! of the frame bytes (copy if you need to keep them). End-of-stream is the
//! callback with `data == null, len == 0`. A blocking pull API can layer on
//! later; the callback is the lower-level primitive.
//!
//! Like [`crate::appsrc`], the parameterless `fn` factory means the element
//! cannot be handed the callback directly, so it is parked in a named global and
//! claimed at `configure_pipeline`.

use core::ffi::c_void;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use spin::Mutex;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, MemoryDomain,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

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

/// Named callbacks, keyed by the `appsink channel` property. Set by
/// [`set_appsink_callback`] before launch; claimed once by the element.
static SINKS: Mutex<BTreeMap<String, Slot>> = Mutex::new(BTreeMap::new());

/// Register the per-frame callback for `appsink channel=<channel>`. Call before
/// launching. A later registration under the same name replaces the earlier one
/// (until the element claims it at startup).
pub fn set_appsink_callback(channel: &str, cb: SampleCallback, user: *mut c_void) {
    SINKS.lock().insert(channel.to_string(), Slot { cb, user });
}

/// Application pull/callback sink. Accepts any caps; hands each frame's bytes to
/// the registered callback.
#[derive(Debug, Default)]
pub struct AppSink {
    channel: String,
    configured: bool,
    slot: Option<Slot>,
    received: u64,
}

impl core::fmt::Debug for Slot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Slot").finish_non_exhaustive()
    }
}

impl AppSink {
    pub fn new() -> Self {
        Self::default()
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

    /// Invoke the callback for one host-visible buffer, if a callback is set.
    fn deliver(&self, data: &[u8], pts_ns: u64) {
        if let Some(slot) = &self.slot {
            (slot.cb)(data.as_ptr(), data.len(), pts_ns, slot.user);
        }
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
        self.slot = SINKS.lock().remove(self.channel_name());
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Application sink",
            "Sink",
            "Hands each buffer to the application's callback (set_appsink_callback / g2g_appsink_set_callback)",
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
                    // Deliver only host-visible memory; a GPU-resident frame
                    // would need a download the v1 path does not do, so it is
                    // skipped (the frame count still advances).
                    match &f.domain {
                        MemoryDomain::System(s) => self.deliver(s.as_slice(), f.timing.pts_ns),
                        MemoryDomain::SystemView(sv) => {
                            self.deliver(&sv.materialize(), f.timing.pts_ns)
                        }
                        _ => {}
                    }
                }
                PipelinePacket::Eos => {
                    // EOS marker: data == null, len == 0.
                    if let Some(slot) = &self.slot {
                        (slot.cb)(core::ptr::null(), 0, 0, slot.user);
                    }
                }
                // Control packets are not surfaced to the application in v1.
                PipelinePacket::Flush
                | PipelinePacket::CapsChanged(_)
                | PipelinePacket::Segment(_) => {}
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
    "callback name matching set_appsink_callback (default \"default\")",
)];

impl PadTemplates for AppSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

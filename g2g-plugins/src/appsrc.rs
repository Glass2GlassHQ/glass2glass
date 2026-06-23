//! Application source (`appsrc`): the application pushes buffers into a running
//! pipeline (M233), the `gst-appsrc` analog.
//!
//! The element and the application live on different threads (the pipeline runs
//! on its own thread, e.g. via `g2g-capi`), so the hand-off is a bounded
//! [`g2g_core::runtime`] channel: pushing wakes the element's `run` arm through
//! the runtime's `Waker`, the same cross-thread wake the Python host relies on.
//!
//! Because the registry builds elements from a parameterless `fn` (the
//! `gst-launch` model), the application cannot hand the element a channel
//! directly. Instead it [`register_appsrc`]s a named feed *before* launch; the
//! `appsrc channel=<name>` element claims the matching receiver at
//! `configure_pipeline`. The name defaults to `"default"` for the common
//! single-`appsrc` pipeline.
//!
//! v1 copies the pushed bytes into a `System` frame; a zero-copy lend (a foreign
//! buffer with a free callback) is a planned follow-up.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};

use spin::Mutex;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{bounded, Receiver, Sender, SourceLoop};
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    RawVideoFormat,
};
use g2g_core::{Dim, Rate};

use crate::capsfilter::parse_caps_set;

/// Bounded depth of the application -> element feed. A push past this returns
/// "full" rather than blocking, so a fast producer never stalls; the pipeline
/// drains it as it runs.
const FEED_DEPTH: usize = 16;

/// One item crossing the feed channel: a buffer (owned or a zero-copy foreign
/// lend, both a `SystemSlice`) or the end-of-stream marker.
#[derive(Debug)]
enum AppItem {
    Frame { slice: SystemSlice, pts_ns: u64 },
    Eos,
}

/// Named feed receivers, keyed by the `appsrc channel` property. Populated by
/// [`register_appsrc`] (application side) and drained once by the matching
/// element at `configure_pipeline`. A global is the bridge the parameterless
/// `fn` factory forces; entries are removed on pickup so they neither leak nor
/// collide across runs.
static FEEDS: Mutex<BTreeMap<String, Receiver<AppItem>>> = Mutex::new(BTreeMap::new());

/// The application's handle to push buffers into an `appsrc channel=<name>`.
/// Cloneable producer; the feed closes (and the source emits EOS) when every
/// handle drops, or on an explicit [`end_of_stream`](AppSrcFeed::end_of_stream).
#[derive(Debug, Clone)]
pub struct AppSrcFeed {
    tx: Sender<AppItem>,
}

impl AppSrcFeed {
    /// Push one buffer with presentation timestamp `pts_ns`, copying `data` into
    /// an owned `System` frame. Returns `false` if the feed is full (retry
    /// later) or the pipeline has gone away.
    pub fn push(&self, data: &[u8], pts_ns: u64) -> bool {
        self.push_slice(SystemSlice::from_boxed(data.to_vec().into_boxed_slice()), pts_ns)
    }

    /// Push a pre-built buffer, for the zero-copy path: pass a
    /// [`SystemSlice::from_foreign`] to lend the application's bytes without a
    /// copy (the free callback fires when the frame is finally dropped).
    /// Returns `false` (releasing `slice`) if the feed is full or closed.
    pub fn push_slice(&self, slice: SystemSlice, pts_ns: u64) -> bool {
        self.tx.try_send(AppItem::Frame { slice, pts_ns }).is_ok()
    }

    /// Signal end-of-stream: the source emits a final `Eos` and `run` returns.
    pub fn end_of_stream(&self) -> bool {
        self.tx.try_send(AppItem::Eos).is_ok()
    }
}

/// Register a feed under `channel` and return the application's push handle. Call
/// before launching the pipeline that contains `appsrc channel=<channel>`.
pub fn register_appsrc(channel: &str) -> AppSrcFeed {
    let (tx, rx) = bounded::<AppItem>(FEED_DEPTH);
    FEEDS.lock().insert(channel.to_string(), rx);
    AppSrcFeed { tx }
}

/// Nominal output caps for the registry / autoplug descriptor. The real caps
/// come from the `caps` property at parse time (via [`AppSrc::intercept_caps`]).
fn nominal_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Application push source. Set its `caps` (fully fixed, `gst-launch` syntax) and
/// `channel` properties; buffers arrive from the matching [`register_appsrc`].
#[derive(Debug, Default)]
pub struct AppSrc {
    channel: String,
    caps: Option<Caps>,
    configured: bool,
    feed: Option<Receiver<AppItem>>,
    seq: u64,
}

impl AppSrc {
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
}

impl SourceLoop for AppSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.caps.clone().ok_or(G2gError::CapsMismatch))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        let constraint = match &self.caps {
            Some(c) => Ok(CapsConstraint::Produces(CapsSet::one(c.clone()))),
            None => Err(G2gError::CapsMismatch),
        };
        core::future::ready(constraint)
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.caps.is_none() {
            return Err(G2gError::CapsMismatch);
        }
        // Claim the application's feed registered under this channel. Absent one,
        // there is no producer, so fail startup loudly rather than hang.
        self.feed = FEEDS.lock().remove(self.channel_name());
        if self.feed.is_none() {
            return Err(G2gError::NotConfigured);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let feed = self.feed.take().ok_or(G2gError::NotConfigured)?;
            let mut pushed = 0u64;
            // The loop ends when the pattern stops matching: an explicit
            // `AppItem::Eos`, or `None` once every feed handle has dropped.
            while let Some(AppItem::Frame { slice, pts_ns }) = feed.recv().await {
                let frame = Frame {
                    domain: MemoryDomain::System(slice),
                    timing: FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
                    sequence: self.seq,
                    meta: Default::default(),
                };
                self.seq += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
                pushed += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(pushed)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        APPSRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "channel" => {
                self.channel = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "caps" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                let set = parse_caps_set(s).ok_or(PropError::Value)?;
                // appsrc declares a single fixed output; take the first
                // alternative (a fully-specified caps yields exactly one).
                let caps = set.alternatives().first().cloned().ok_or(PropError::Value)?;
                self.caps = Some(caps);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Application source",
            "Source",
            "Emits buffers the application pushes in via register_appsrc / g2g_appsrc_push",
            "g2g",
        )
    }
}

static APPSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new("channel", PropKind::Str, "feed name matching register_appsrc (default \"default\")"),
    PropertySpec::new("caps", PropKind::Str, "fixed output caps, gst-launch syntax (e.g. video/x-raw,format=RGBA,width=320,height=240,framerate=30/1)"),
];

/// The registry needs this source's declared output caps; see [`nominal_caps`].
pub fn registered_output_caps() -> Caps {
    nominal_caps()
}

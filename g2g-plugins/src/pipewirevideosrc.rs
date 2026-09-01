//! Linux PipeWire video capture source. Connects an input stream to a PipeWire
//! video node (camera, another client's published node, a screen-capture node)
//! and streams raw frames downstream in system memory. The video sibling of the
//! audio [`PipeWireSrc`](crate::pipewiresrc::PipeWireSrc).
//!
//! Like the audio element, PipeWire's callback-driven main loop is pinned to one
//! thread, so it runs on a dedicated worker that feeds the async `run` loop over
//! a channel.
//!
//! ## Negotiation
//!
//! `intercept_caps` publishes fixed caps built from the properties (`I420` at the
//! requested geometry / rate), but the connect pod offers our whole format table
//! and an open size / framerate range, so a node with its own fixed mode still
//! links. What the node settled on arrives in `param_changed`, and when it
//! differs from the advertised caps the element emits `CapsChanged` before the
//! first frame. A format outside the table fails the capture instead of being
//! reinterpreted.
//!
//! ## Buffers
//!
//! `io-mode` picks how frames leave the node. `mmap` (the default) connects with
//! `MAP_BUFFERS` and copies each frame out of the mapped block, de-striding
//! padded rows, into `System` memory. `dmabuf` asks the producer for dma-buf
//! memory (a `Buffers` param whose only accepted data type is `SPA_DATA_DmaBuf`)
//! and shares the descriptor downstream as `MemoryDomain::DmaBuf`, no copy: a
//! producer with no dma-buf to give fails the negotiation instead. Because the
//! domain is part of negotiation the mode is a property, not something picked per
//! buffer (GStreamer's `pipewiresrc` gets it from a caps feature instead).
//!
//! The dma-buf path only offers the single-plane formats: a planar frame arrives
//! as one SPA block per plane and `OwnedDmaBuf` carries one fd. The element holds
//! each buffer until every share of its frame is gone, so the producer never
//! overwrites a frame downstream is still reading, and it connects without
//! `RT_PROCESS` so the recycling (a timer plus each `process`) and the dequeue all
//! run on the one loop thread that owns the stream's buffer queues.
//!
//! ## Screen capture
//!
//! `target-object` names a node on the session's own PipeWire remote, in either
//! form `target.object` resolves: a node name or an object serial. A Wayland
//! desktop does not publish its outputs there, so screen capture goes through
//! `portal=true` instead (needs the `portal` feature): the element runs the
//! xdg-desktop-portal ScreenCast handshake, which asks the user to pick a
//! monitor or window, and captures the granted node on the private PipeWire
//! remote the portal hands back. The two are mutually exclusive, since they name
//! nodes on different remotes.

use core::cell::RefCell;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::{OwnedDmaBuf, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError,
    HardwareError, LatencyReport, MemoryDomain, MemoryDomainKind, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate,
    RawVideoFormat,
};

use pipewire as pw;
use pw::spa;
use pw::sys as pw_sys;

use crate::videoconvert::{raw_format_from_str, raw_format_to_str};

#[cfg(feature = "portal")]
use crate::screencastportal::{open_screen_cast, PortalRequest, PortalSourceTypes};

use crate::pwvideo::{
    dmabuf_buffers_pod_bytes, dmabuf_frame, format_pod_bytes, rate_q16, single_plane_row_bytes,
    spa_format, supported_formats, DataBlock, DmaBufFrame, FormatOffer, PlaneLayout, VideoInfo,
    MAX_DIM,
};

/// Requested capture geometry / rate when the caller does not specify one.
const DEFAULT_WIDTH: u32 = 640;
const DEFAULT_HEIGHT: u32 = 480;
const DEFAULT_FPS: u32 = 30;
/// The format the advertised caps carry and the connect pod prefers, per mode.
/// The dma-buf path leads with a single-plane format because that is all its
/// buffers can be (see [`FormatOffer::SinglePlane`]).
const PREFERRED_FORMAT: RawVideoFormat = RawVideoFormat::I420;
const PREFERRED_DMABUF_FORMAT: RawVideoFormat = RawVideoFormat::Bgra8;
/// How often the dma-buf path hands the producer back the buffers downstream has
/// released. A frame in flight blocks the buffer it came in, and `process` only
/// runs when the producer has a buffer to fill, so recycling cannot wait for it.
const RECYCLE_INTERVAL: core::time::Duration = core::time::Duration::from_millis(2);
/// Blocks read from a dequeued buffer. Only a single-block buffer is usable, so a
/// buffer claiming more is read far enough to reject it and no further.
const MAX_BLOCKS: usize = 4;
/// How long the element waits for the worker to report a connected stream, on
/// top of whatever the portal handshake is allowed. The daemon round-trip.
const CONNECT_TIMEOUT: core::time::Duration = core::time::Duration::from_secs(5);
/// How long one blocking read of the worker's setup result takes before the
/// element yields to the executor and tries again, so a long wait (a portal
/// consent dialog) does not stall the other arms of a cooperative graph.
const READY_POLL_INTERVAL: core::time::Duration = core::time::Duration::from_millis(5);
/// Default deadline on each portal step, the consent dialog included.
#[cfg(feature = "portal")]
const DEFAULT_PORTAL_TIMEOUT_SECS: u64 = 60;
/// Portal steps that can each wait a full timeout (CreateSession, SelectSources,
/// Start), which is what the element's own setup deadline has to cover.
#[cfg(feature = "portal")]
const PORTAL_REQUEST_STEPS: u32 = 3;

/// Say the conflict out loud: a `PropError` can only name the one property that
/// was rejected, never the other half of the contradiction.
#[cfg(feature = "portal")]
fn log_portal_target_conflict() {
    g2g_core::g2g_error!(
        g2g_core::log::Target::category("PipeWireVideoSrc"),
        "portal=true and target-object name capture nodes on different PipeWire remotes: set one, not both"
    );
}

/// Where a captured frame lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IoMode {
    /// Copy each frame out of the producer's mapped buffer into system memory.
    #[default]
    MemoryMap,
    /// Share the producer's dma-buf downstream, no copy.
    DmaBuf,
}

impl IoMode {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "mmap" => Some(Self::MemoryMap),
            "dmabuf" => Some(Self::DmaBuf),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::MemoryMap => "mmap",
            Self::DmaBuf => "dmabuf",
        }
    }
}

/// Control message to the loop thread (quit on teardown).
enum Ctrl {
    Terminate,
}

/// Loop-thread to element messages.
enum FromWorker {
    /// The negotiated format, sent on every change (so before the first frame).
    Format(VideoInfo),
    /// One frame copied out of a mapped buffer, and the row pitch it carries.
    /// A pitch of 0 means the rows were packed tight on the way out.
    Frame { bytes: Vec<u8>, first_stride: usize },
    /// One frame still in the producer's dma-buf. The loop thread keeps a share of
    /// its own and recycles the buffer once this one is dropped.
    DmaBuf(OwnedDmaBuf),
    /// The stream negotiated something we cannot carry, handed us a buffer that
    /// disagrees with the negotiated geometry, or went to the error state (a
    /// pinned format the node cannot produce lands here).
    Failed(G2gError),
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::pipewirevideosrc::{IoMode, PipeWireVideoSrc};
///
/// let src = PipeWireVideoSrc::new()
///     .with_size(1920, 1080)
///     .with_fps(30)
///     .with_io_mode(IoMode::DmaBuf);
/// ```
#[derive(Debug)]
pub struct PipeWireVideoSrc {
    /// Node to capture from (`node.name` or object serial); empty = the default
    /// video source the session manager picks.
    target: String,
    req_width: u32,
    req_height: u32,
    req_fps: u32,
    /// Pinned capture format: the connect pod then offers this one alone, so the
    /// node either produces it or negotiation fails. `None` = offer the whole
    /// table and take what the node settles on.
    pin_format: Option<RawVideoFormat>,
    io_mode: IoMode,
    /// `u64::MAX` = run until error or downstream shutdown; else stop after this
    /// many frames and emit EOS. The bounded-capture / test path.
    frame_limit: u64,
    /// Ask xdg-desktop-portal for a screen share instead of capturing a node on
    /// the session's PipeWire remote. Exclusive with `target`.
    #[cfg(feature = "portal")]
    portal: bool,
    #[cfg(feature = "portal")]
    portal_source_types: PortalSourceTypes,
    /// Empty = ask the user; else re-open the grant this token names. Replaced by
    /// the granted token once a handshake succeeds.
    #[cfg(feature = "portal")]
    portal_restore_token: String,
    #[cfg(feature = "portal")]
    portal_timeout_secs: u64,
    /// Whether a consumer asked where the rows are (M1059), so a padded mapped
    /// buffer travels as it is instead of being packed tight.
    keep_row_padding: bool,
    configured: bool,
}

impl Default for PipeWireVideoSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl PipeWireVideoSrc {
    /// Capture from the default video node at 640x480 / 30.
    pub fn new() -> Self {
        Self {
            target: String::new(),
            req_width: DEFAULT_WIDTH,
            req_height: DEFAULT_HEIGHT,
            req_fps: DEFAULT_FPS,
            pin_format: None,
            io_mode: IoMode::MemoryMap,
            frame_limit: u64::MAX,
            #[cfg(feature = "portal")]
            portal: false,
            #[cfg(feature = "portal")]
            portal_source_types: PortalSourceTypes::Monitor,
            #[cfg(feature = "portal")]
            portal_restore_token: String::new(),
            #[cfg(feature = "portal")]
            portal_timeout_secs: DEFAULT_PORTAL_TIMEOUT_SECS,
            keep_row_padding: false,
            configured: false,
        }
    }

    /// Capture from a specific node: its `node.name` or its object serial.
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = target.into();
        self
    }

    /// Request a capture size. The node may keep its own; the negotiated caps
    /// (and a `CapsChanged`) reflect what it actually produces.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.req_width = width;
        self.req_height = height;
        self
    }

    /// Request a frame rate in fps. Best-effort, same as the size.
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.req_fps = fps;
        self
    }

    /// Pin the capture format. Unlike the size and rate this is not best effort:
    /// the connect pod offers this format alone, so a node that cannot produce it
    /// fails the capture and no mid-stream format change can arrive.
    pub fn with_format(mut self, format: RawVideoFormat) -> Self {
        self.pin_format = Some(format);
        self
    }

    /// Stop after `n` frames and emit EOS (0 emits EOS without opening the
    /// stream). Without this the source runs until an error or until downstream
    /// drops.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// Take the producer's dma-buf instead of copying out of a mapped buffer.
    /// The frames then carry [`MemoryDomain::DmaBuf`] and only the single-plane
    /// formats are on offer, so a pinned planar format is rejected.
    pub fn with_io_mode(mut self, mode: IoMode) -> Self {
        self.io_mode = mode;
        self
    }

    /// Capture the screen share xdg-desktop-portal grants, rather than a node on
    /// the session's PipeWire remote. Exclusive with [`Self::with_target`]: the
    /// two name nodes on different remotes, so setting both fails the capture.
    #[cfg(feature = "portal")]
    pub fn with_portal(mut self, source_types: PortalSourceTypes) -> Self {
        self.portal = true;
        self.portal_source_types = source_types;
        self
    }

    /// Re-open the grant `token` names instead of asking the user again. A stale
    /// or unknown token just means the portal asks.
    #[cfg(feature = "portal")]
    pub fn with_portal_restore_token(mut self, token: impl Into<String>) -> Self {
        self.portal_restore_token = token.into();
        self
    }

    /// How long each portal step may take, the consent dialog included.
    #[cfg(feature = "portal")]
    pub fn with_portal_timeout_secs(mut self, seconds: u64) -> Self {
        self.portal_timeout_secs = seconds;
        self
    }

    /// The portal grant and a named target node both say where frames come from,
    /// and they name nodes on different PipeWire remotes, so having both is a
    /// contradiction rather than a preference to resolve.
    #[cfg(feature = "portal")]
    fn portal_conflicts_with_target(&self) -> bool {
        self.portal && !self.target.is_empty()
    }

    /// What the handshake should ask for, or `None` when the portal is off.
    #[cfg(feature = "portal")]
    fn portal_request(&self) -> Option<PortalRequest> {
        self.portal.then(|| PortalRequest {
            source_types: self.portal_source_types,
            restore_token: (!self.portal_restore_token.is_empty())
                .then(|| self.portal_restore_token.clone()),
            timeout: core::time::Duration::from_secs(self.portal_timeout_secs),
        })
    }

    /// The format the caps advertise and the connect pod leads with: the pinned
    /// one, or the mode's default preference. A pin the element cannot map, or one
    /// the mode cannot carry, fails here.
    fn format(&self) -> Result<RawVideoFormat, G2gError> {
        let preferred = match self.io_mode {
            IoMode::MemoryMap => PREFERRED_FORMAT,
            IoMode::DmaBuf => PREFERRED_DMABUF_FORMAT,
        };
        let format = self.pin_format.unwrap_or(preferred);
        spa_format(format).ok_or(G2gError::CapsMismatch)?;
        if self.io_mode == IoMode::DmaBuf && single_plane_row_bytes(format, 1).is_none() {
            return Err(G2gError::CapsMismatch);
        }
        Ok(format)
    }

    /// The formats the connect pod offers behind the preferred one.
    fn format_offer(&self) -> FormatOffer {
        match (self.pin_format, self.io_mode) {
            (Some(_), _) => FormatOffer::PreferredOnly,
            (None, IoMode::DmaBuf) => FormatOffer::SinglePlane,
            (None, IoMode::MemoryMap) => FormatOffer::All,
        }
    }

    /// The caps the element advertises before the node has answered. Fixed, so
    /// `fixate()` has nothing left to do.
    fn caps(&self) -> Result<Caps, G2gError> {
        if self.req_width == 0
            || self.req_height == 0
            || self.req_width > MAX_DIM
            || self.req_height > MAX_DIM
        {
            return Err(G2gError::CapsMismatch);
        }
        Ok(Caps::RawVideo {
            format: self.format()?,
            width: Dim::Fixed(self.req_width),
            height: Dim::Fixed(self.req_height),
            framerate: Rate::Fixed(rate_q16(self.req_fps, 1)),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        })
    }
}

impl SourceLoop for PipeWireVideoSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.caps())
    }

    /// Produces the requested caps, so a chain built on a PipeWire camera takes
    /// the native arc-consistency path (mirrors `V4l2Src`).
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(
            self.caps()
                .map(|c| CapsConstraint::Produces(CapsSet::one(c))),
        )
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        #[cfg(feature = "portal")]
        if self.portal_conflicts_with_target() {
            log_portal_target_conflict();
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        self.caps()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "PipeWire video source",
            "Source/Video",
            "Captures raw video from a PipeWire node",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "target-object",
                PropKind::Str,
                "node name or object serial to capture from (empty = default)",
            ),
            PropertySpec::new(
                "width",
                PropKind::Uint,
                "requested capture width (the node may keep its own)",
            )
            .with_default("640"),
            PropertySpec::new(
                "height",
                PropKind::Uint,
                "requested capture height (the node may keep its own)",
            )
            .with_default("480"),
            PropertySpec::new(
                "framerate",
                PropKind::Uint,
                "requested capture rate, fps (best effort)",
            )
            .with_default("30"),
            PropertySpec::new(
                "format",
                PropKind::Str,
                "pin the capture format: I420 | NV12 | YUY2 | RGBA | BGRA (empty = whatever the node offers)",
            )
            .with_default(""),
            PropertySpec::new(
                "io-mode",
                PropKind::Str,
                "where frames land: mmap (copy into system memory) | dmabuf (share the producer's buffer)",
            )
            .with_default("mmap"),
            PropertySpec::new(
                "num-buffers",
                PropKind::Int,
                "frames to capture then EOS (-1 = forever)",
            )
            .with_default("-1")
            .with_range("-1", "9223372036854775807"),
            #[cfg(feature = "portal")]
            PropertySpec::new(
                "portal",
                PropKind::Bool,
                "capture the screen share xdg-desktop-portal grants (asks the user) instead of a target-object node",
            )
            .with_default("false"),
            #[cfg(feature = "portal")]
            PropertySpec::new(
                "portal-source-types",
                PropKind::Str,
                "what the portal offers to share: monitor | window | any",
            )
            .with_default("monitor"),
            #[cfg(feature = "portal")]
            PropertySpec::new(
                "portal-restore-token",
                PropKind::Str,
                "token from an earlier grant, re-opened without asking (empty = ask; the granted token is logged at info)",
            )
            .with_default(""),
            #[cfg(feature = "portal")]
            PropertySpec::new(
                "portal-timeout",
                PropKind::Uint,
                "seconds to wait for each portal step, the consent dialog included",
            )
            .with_default("60"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "target-object" => {
                let target = value.as_str().ok_or(PropError::Type)?;
                // whichever of the two is set second is the one that gets to fail
                #[cfg(feature = "portal")]
                if self.portal && !target.is_empty() {
                    log_portal_target_conflict();
                    return Err(PropError::Value);
                }
                self.target = target.to_string();
                Ok(())
            }
            "width" => {
                self.req_width = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "height" => {
                self.req_height = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "framerate" => {
                self.req_fps = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "format" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.pin_format = if s.is_empty() {
                    None
                } else {
                    // a name outside the SPA mapping table is an error, never a
                    // silent fall back to the default format
                    let format = raw_format_from_str(s).ok_or(PropError::Value)?;
                    spa_format(format).ok_or(PropError::Value)?;
                    Some(format)
                };
                Ok(())
            }
            "io-mode" => {
                let name = value.as_str().ok_or(PropError::Type)?;
                self.io_mode = IoMode::from_name(name).ok_or(PropError::Value)?;
                Ok(())
            }
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.frame_limit, &value),
            #[cfg(feature = "portal")]
            "portal" => {
                let on = value.as_bool().ok_or(PropError::Type)?;
                if on && !self.target.is_empty() {
                    log_portal_target_conflict();
                    return Err(PropError::Value);
                }
                self.portal = on;
                Ok(())
            }
            #[cfg(feature = "portal")]
            "portal-source-types" => {
                let name = value.as_str().ok_or(PropError::Type)?;
                self.portal_source_types =
                    PortalSourceTypes::from_name(name).ok_or(PropError::Value)?;
                Ok(())
            }
            #[cfg(feature = "portal")]
            "portal-restore-token" => {
                self.portal_restore_token = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            #[cfg(feature = "portal")]
            "portal-timeout" => {
                let seconds = value.as_uint().ok_or(PropError::Type)?;
                if seconds == 0 {
                    return Err(PropError::Value);
                }
                self.portal_timeout_secs = seconds;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "target-object" => Some(PropValue::Str(self.target.clone())),
            "width" => Some(PropValue::Uint(self.req_width as u64)),
            "height" => Some(PropValue::Uint(self.req_height as u64)),
            "framerate" => Some(PropValue::Uint(self.req_fps as u64)),
            "format" => Some(PropValue::Str(
                self.pin_format.map_or("", raw_format_to_str).into(),
            )),
            "io-mode" => Some(PropValue::Str(self.io_mode.as_str().into())),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.frame_limit)),
            #[cfg(feature = "portal")]
            "portal" => Some(PropValue::Bool(self.portal)),
            #[cfg(feature = "portal")]
            "portal-source-types" => Some(PropValue::Str(self.portal_source_types.as_str().into())),
            #[cfg(feature = "portal")]
            "portal-restore-token" => Some(PropValue::Str(self.portal_restore_token.clone())),
            #[cfg(feature = "portal")]
            "portal-timeout" => Some(PropValue::Uint(self.portal_timeout_secs)),
            _ => None,
        }
    }

    /// The domain the frames carry: negotiated up front from `io-mode`, so a
    /// downstream stage sees one domain for the whole capture.
    fn output_memory(&self) -> MemoryDomainKind {
        match self.io_mode {
            IoMode::MemoryMap => MemoryDomainKind::System,
            IoMode::DmaBuf => MemoryDomainKind::DmaBuf,
        }
    }

    /// M1059: a producer whose rows are padded normally has them packed tight
    /// on the way out of a mapped buffer. A consumer that asked for a
    /// `PlaneLayout` reads rows where they lie, so that copy can go.
    fn configure_allocation(&mut self, params: &g2g_core::AllocationParams) {
        #[cfg(feature = "metadata")]
        {
            self.keep_row_padding = params.meta_requests.wants::<g2g_core::meta::PlaneLayout>();
        }
        #[cfg(not(feature = "metadata"))]
        let _ = params;
    }

    /// Live source: contributes one frame period so the sink keeps a frame in
    /// hand and never runs dry waiting on capture (same as `V4l2Src`).
    fn latency(&self) -> LatencyReport {
        let period_ns = if self.req_fps > 0 {
            1_000_000_000 / u64::from(self.req_fps)
        } else {
            0
        };
        LatencyReport::live(period_ns, None)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            if crate::numbuffers::finished_at_zero_limit(self.frame_limit, out).await? {
                return Ok(0);
            }
            let mut advertised = self.caps()?;
            let pod = format_pod_bytes(
                self.format()?,
                self.format_offer(),
                self.req_width,
                self.req_height,
                self.req_fps,
            )?;
            let target = self.target.clone();
            let limit = self.frame_limit;
            let policy = BufferPolicy {
                io_mode: self.io_mode,
                keep_row_padding: self.keep_row_padding,
            };
            #[cfg(feature = "portal")]
            let portal: PortalSetup = self.portal_request();
            #[cfg(not(feature = "portal"))]
            let portal: PortalSetup = ();
            let setup_deadline = setup_deadline(&portal);

            // Frames and format changes cross from the loop thread to here.
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<FromWorker>();
            // Control + a setup-result handshake (surface a connect failure).
            let (ctrl_tx, ctrl_rx) = pw::channel::channel::<Ctrl>();
            let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<SetupResult>(1);

            let handle = std::thread::Builder::new()
                .name(String::from("g2g-pipewirevideosrc"))
                .spawn(move || {
                    match build_and_run(&target, &pod, policy, portal, tx, ctrl_rx, &ready_tx) {
                        Ok(()) => {}
                        Err(code) => {
                            let _ = ready_tx.send(Err(code));
                        }
                    }
                })
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

            // Wait for the stream to connect: the daemon round-trip, plus the
            // portal handshake when there is one. Yields between polls so a
            // consent dialog nobody is answering does not stall sibling arms.
            let connected = await_setup(&ready_rx, setup_deadline).await;
            match connected {
                Ok(_granted) =>
                {
                    #[cfg(feature = "portal")]
                    if let Some(token) = _granted {
                        g2g_core::g2g_info!(
                            g2g_core::log::Target::category("PipeWireVideoSrc"),
                            "portal restore token: {token} (pass as portal-restore-token to skip the dialog)"
                        );
                        self.portal_restore_token = token;
                    }
                }
                Err(e) => {
                    let _ = ctrl_tx.send(Ctrl::Terminate);
                    let _ = handle.join();
                    return Err(e);
                }
            }

            let mut period_ns = if self.req_fps > 0 {
                1_000_000_000 / u64::from(self.req_fps)
            } else {
                0
            };
            let mut seq = 0u64;
            let mut pts = 0u64;
            let mut downstream_open = true;
            let mut failure = None;
            // The format the frames arriving now are in, so a padded frame can
            // say where its rows are.
            let mut current_format: Option<VideoInfo> = None;

            while seq < limit {
                let Some(msg) = rx.recv().await else {
                    break; // worker ended
                };
                // The domain the frame travels in, plus where plane 0 starts in
                // its buffer and how far apart the rows are (0 means tight).
                let (domain, plane0_offset, first_stride) = match msg {
                    FromWorker::Failed(e) => {
                        failure = Some(e);
                        break;
                    }
                    FromWorker::Format(info) => {
                        period_ns = info.frame_period_ns();
                        current_format = Some(info);
                        let caps = info.caps();
                        if caps != advertised {
                            if out
                                .push(PipelinePacket::CapsChanged(caps.clone()))
                                .await
                                .is_err()
                            {
                                downstream_open = false;
                                break;
                            }
                            advertised = caps;
                        }
                        continue;
                    }
                    FromWorker::Frame {
                        bytes,
                        first_stride,
                    } => (
                        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                        0,
                        first_stride,
                    ),
                    // The producer's own buffer, so its rows sit at the
                    // producer's pitch whether or not anybody asked: a consumer
                    // that maps it reads them off the frame instead of the
                    // domain type.
                    FromWorker::DmaBuf(dmabuf) => {
                        let (offset, stride) = (dmabuf.offset as usize, dmabuf.stride as usize);
                        (MemoryDomain::DmaBuf(dmabuf), offset, stride)
                    }
                };
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                let mut frame = Frame {
                    domain,
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: period_ns,
                        capture_ns: pts,
                        arrival_ns,
                        // raw frames are each independently presentable
                        keyframe: true,
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                if let Some(info) = current_format.as_ref() {
                    crate::paddedrows::declare_padded_rows(
                        &mut frame,
                        info.format,
                        info.width as usize,
                        info.height as usize,
                        plane0_offset,
                        first_stride,
                    );
                }
                if out.push(PipelinePacket::DataFrame(frame)).await.is_err() {
                    downstream_open = false;
                    break;
                }
                pts += period_ns;
                seq += 1;
            }

            // Stop the loop and reap the worker.
            let _ = ctrl_tx.send(Ctrl::Terminate);
            let _ = handle.join();

            if let Some(e) = failure {
                return Err(e);
            }
            if downstream_open {
                out.push(PipelinePacket::Eos).await?;
            }
            Ok(seq)
        })
    }
}

impl PadTemplates for PipeWireVideoSrc {
    /// Produces any raw format the element can map; a constructed instance fixes
    /// the geometry / rate from its properties and the format at negotiation.
    fn pad_templates() -> Vec<PadTemplate> {
        let alternatives: Vec<Caps> = supported_formats()
            .map(|format| Caps::RawVideo {
                format,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            })
            .collect();
        Vec::from([PadTemplate::source(CapsSet::from_alternatives(
            alternatives,
        ))])
    }
}

// =================================================================
// Worker thread: the PipeWire capture main loop
// =================================================================

/// What the element hands the worker about the portal: the handshake to run, or
/// nothing at all when the feature is off.
#[cfg(feature = "portal")]
type PortalSetup = Option<PortalRequest>;
#[cfg(not(feature = "portal"))]
type PortalSetup = ();

/// The worker's setup result: connected (carrying the portal's restore token
/// when there was a handshake that produced one), or a PipeWire error code.
type SetupResult = Result<Option<String>, i32>;

/// How long the element gives the worker to report a connected stream. Every
/// portal step is bounded on the worker side, so this only has to cover the sum
/// of them plus the daemon round-trip.
fn setup_deadline(portal: &PortalSetup) -> core::time::Duration {
    #[cfg(feature = "portal")]
    if let Some(request) = portal {
        return request
            .timeout
            .saturating_mul(PORTAL_REQUEST_STEPS)
            .saturating_add(CONNECT_TIMEOUT);
    }
    #[cfg(not(feature = "portal"))]
    let _ = portal;
    CONNECT_TIMEOUT
}

/// Wait for the worker's setup result without blocking the executor for the
/// whole deadline: read in short slices and yield in between.
async fn await_setup(
    ready: &std::sync::mpsc::Receiver<SetupResult>,
    deadline: core::time::Duration,
) -> Result<Option<String>, G2gError> {
    let started = std::time::Instant::now();
    loop {
        match ready.recv_timeout(READY_POLL_INTERVAL) {
            Ok(Ok(granted)) => return Ok(granted),
            Ok(Err(code)) => return Err(G2gError::Hardware(HardwareError::PipeWire(code))),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(G2gError::Hardware(HardwareError::PipeWire(-1)))
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if started.elapsed() >= deadline {
                    return Err(G2gError::Hardware(HardwareError::PipeWire(-1)));
                }
                tokio::task::yield_now().await;
            }
        }
    }
}

/// Run the portal handshake and turn its outcome into what the worker needs: the
/// remote to connect to, the node to target, and the token to report back.
#[cfg(feature = "portal")]
fn run_portal_handshake(
    request: &PortalRequest,
) -> Result<(std::os::fd::OwnedFd, u32, Option<String>), i32> {
    match open_screen_cast(request) {
        Ok(granted) => Ok((granted.remote_fd, granted.node_id, granted.restore_token)),
        Err(e) => {
            g2g_core::g2g_error!(
                g2g_core::log::Target::category("PipeWireVideoSrc"),
                "screencast portal handshake failed: {e}"
            );
            Err(-1)
        }
    }
}

/// What the `process` callback needs to turn a buffer into a frame, refreshed by
/// `param_changed`. `None` until the format is negotiated (or after a failure,
/// which stops further buffers being interpreted).
struct Negotiated {
    info: VideoInfo,
    layout: PlaneLayout,
}

/// A dequeued dma-buf buffer, lent out as a frame. `frame` is the loop thread's
/// own share of the descriptor downstream got: while it is not the last one, the
/// buffer stays out of the producer's hands.
#[derive(Debug)]
struct HeldBuffer {
    buffer: *mut pw_sys::pw_buffer,
    frame: OwnedDmaBuf,
}

/// Buffers lent out on the dma-buf path, shared by the `process` callback and the
/// recycling timer. Both run on the loop thread, which is also the only thread
/// allowed to touch the stream's buffer queues.
type HeldBuffers = Rc<RefCell<Vec<HeldBuffer>>>;

/// How the worker turns a dequeued buffer into a frame: which buffer type the
/// stream negotiates, and whether a padded mapped buffer goes out as it is.
#[derive(Debug, Clone, Copy)]
struct BufferPolicy {
    io_mode: IoMode,
    /// Whether a consumer asked where the rows are, so a padded mapped buffer
    /// goes out as it is instead of being packed tight.
    keep_row_padding: bool,
}

struct UserData {
    negotiated: Option<Negotiated>,
    tx: tokio::sync::mpsc::UnboundedSender<FromWorker>,
    policy: BufferPolicy,
    held: HeldBuffers,
}

fn build_and_run(
    target: &str,
    pod: &[u8],
    policy: BufferPolicy,
    portal: PortalSetup,
    tx: tokio::sync::mpsc::UnboundedSender<FromWorker>,
    ctrl_rx: pw::channel::Receiver<Ctrl>,
    ready: &std::sync::mpsc::SyncSender<SetupResult>,
) -> Result<(), i32> {
    // The handshake talks D-Bus and can sit on a consent dialog, so it runs here
    // on the worker rather than on the executor. It is bounded by its own
    // per-step timeout, so this thread always gets to report a result.
    #[cfg(feature = "portal")]
    let portal = portal.as_ref().map(run_portal_handshake).transpose()?;
    #[cfg(not(feature = "portal"))]
    let portal: Option<(std::os::fd::OwnedFd, u32, Option<String>)> = {
        let () = portal;
        None
    };

    pw::init();
    let mainloop = pw::main_loop::MainLoop::new(None).map_err(|_| -1)?;
    let context = pw::context::Context::new(&mainloop).map_err(|_| -1)?;
    // A portal grant lives on the remote the portal opened for us, never on the
    // session's own, so the node id only means anything over that fd.
    let (core, portal_node, restore_token) = match portal {
        Some((remote_fd, node_id, restore_token)) => (
            context.connect_fd(remote_fd, None).map_err(|_| -1)?,
            Some(node_id),
            restore_token,
        ),
        None => (context.connect(None).map_err(|_| -1)?, None, None),
    };

    // media.type is what the session manager's policy matches on, so it has to
    // be here for the link to be made at all.
    let mut props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Video",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Camera",
    };
    if portal_node.is_some() {
        props.insert(*pw::keys::MEDIA_ROLE, "Screen");
    }
    if !target.is_empty() {
        // spelled out because pipewire-rs gates its TARGET_OBJECT constant
        // behind a crate feature this build does not enable
        props.insert("target.object", target);
    }
    // Rc so the recycling timer can reach the stream the listener is built on.
    let stream =
        Rc::new(pw::stream::Stream::new(&core, "g2g-pipewirevideosrc", props).map_err(|_| -1)?);

    let held: HeldBuffers = Rc::new(RefCell::new(Vec::new()));
    let user_data = UserData {
        negotiated: None,
        tx,
        policy,
        held: Rc::clone(&held),
    };

    let _listener = stream
        .add_local_listener_with_user_data(user_data)
        // A format the node cannot produce (a pinned one it does not offer) is
        // reported as a stream error, not a failed connect: surface it so the
        // capture fails instead of waiting for frames that never come.
        .state_changed(|_, user_data, _old, new| {
            if let pw::stream::StreamState::Error(_) = new {
                let _ = user_data
                    .tx
                    .send(FromWorker::Failed(G2gError::Hardware(
                        HardwareError::PipeWire(-1),
                    )));
            }
        })
        .param_changed(|stream, user_data, id, param| {
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }
            let Some(param) = param else {
                user_data.negotiated = None;
                return;
            };
            match parse_format(param) {
                Ok(info) => {
                    let changed = user_data.negotiated.as_ref().map(|n| n.info) != Some(info);
                    // geometry already bounded by VideoInfo, so a layout exists
                    let Some(layout) = PlaneLayout::new(info.format, info.width, info.height)
                    else {
                        user_data.negotiated = None;
                        let _ = user_data
                            .tx
                            .send(FromWorker::Failed(G2gError::CapsMismatch));
                        return;
                    };
                    user_data.negotiated = Some(Negotiated { info, layout });
                    // The buffers are allocated after this callback, so this is
                    // where the dma-buf demand has to be announced.
                    if user_data.policy.io_mode == IoMode::DmaBuf {
                        let bytes = dmabuf_buffers_pod_bytes();
                        if let Some(pod) = spa::pod::Pod::from_bytes(&bytes) {
                            let _ = stream.update_params(&mut [pod]);
                        }
                    }
                    if changed {
                        let _ = user_data.tx.send(FromWorker::Format(info));
                    }
                }
                Err(e) => {
                    user_data.negotiated = None;
                    let _ = user_data.tx.send(FromWorker::Failed(e));
                }
            }
        })
        .process(|stream, user_data| match user_data.policy.io_mode {
            IoMode::MemoryMap => copy_mapped_frame(stream, user_data),
            IoMode::DmaBuf => share_dmabuf_frame(stream, user_data),
        })
        .register()
        .map_err(|_| -1)?;

    // The dma-buf path holds buffers, so it must not run `process` on the realtime
    // thread: the dequeue and the recycling below have to share one thread. It also
    // wants no mapping (`MAP_BUFFERS` skips dma-buf anyway).
    let flags = match policy.io_mode {
        IoMode::MemoryMap => {
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS
        }
        IoMode::DmaBuf => pw::stream::StreamFlags::AUTOCONNECT,
    };
    let mut params = [spa::pod::Pod::from_bytes(pod).ok_or(-1)?];
    stream
        .connect(
            spa::utils::Direction::Input,
            portal_node,
            flags,
            &mut params,
        )
        .map_err(|_| -1)?;

    // `process` only runs when the producer has a buffer to fill, so a stream whose
    // buffers are all downstream would never be called again: recycle on a timer
    // too, or a slow consumer stalls the capture for good.
    let _recycle_timer = match policy.io_mode {
        IoMode::MemoryMap => None,
        IoMode::DmaBuf => {
            let recycle_stream = Rc::clone(&stream);
            let recycle_held = Rc::clone(&held);
            let timer = mainloop.loop_().add_timer(move |_| {
                requeue_released(&recycle_stream, &mut recycle_held.borrow_mut());
            });
            timer
                .update_timer(Some(RECYCLE_INTERVAL), Some(RECYCLE_INTERVAL))
                .into_sync_result()
                .map_err(|_| -1)?;
            Some(timer)
        }
    };

    let weak = mainloop.downgrade();
    let _recv = ctrl_rx.attach(mainloop.loop_(), move |_ctrl| {
        if let Some(ml) = weak.upgrade() {
            ml.quit();
        }
    });

    let _ = ready.send(Ok(restore_token));
    mainloop.run();
    Ok(())
}

/// The bytes one mapped chunk hands downstream and the row pitch they carry:
/// the producer's rows where they lie when a consumer asked for a
/// `PlaneLayout`, packed tight otherwise. `None` when the chunk disagrees with
/// the negotiated geometry.
fn take_mapped_rows(
    src: &[u8],
    info: &VideoInfo,
    layout: &PlaneLayout,
    stride: usize,
    keep_row_padding: bool,
) -> Option<(Vec<u8>, usize)> {
    let (w, h) = (info.width as usize, info.height as usize);
    if keep_row_padding && stride > crate::pixel::row_bytes(info.format, w) {
        let needed = crate::paddedrows::padded_frame_bytes(info.format, w, h, 0, stride)?;
        return (src.len() >= needed).then(|| (src.to_vec(), stride));
    }
    let mut packed = Vec::with_capacity(layout.frame_bytes());
    layout.copy_tight(src, stride, &mut packed)?;
    Some((packed, 0))
}

/// Copy one mapped buffer out: the `mmap` path's frame.
fn copy_mapped_frame(stream: &pw::stream::StreamRef, user_data: &mut UserData) {
    let Some(negotiated) = user_data.negotiated.as_ref() else {
        return; // no usable format: drop the buffer rather than guess
    };
    let Some(mut buffer) = stream.dequeue_buffer() else {
        return;
    };
    let datas = buffer.datas_mut();
    if datas.is_empty() {
        return;
    }
    let data = &mut datas[0];
    let (offset, size, stride) = {
        let chunk = data.chunk();
        (
            chunk.offset() as usize,
            chunk.size() as usize,
            usize::try_from(chunk.stride()).unwrap_or(0),
        )
    };
    // An empty chunk is a normal tick (the node produced nothing), not a
    // malformed buffer.
    if size == 0 {
        return;
    }
    let Some(mapped) = data.data() else {
        return;
    };
    let fits = offset
        .checked_add(size)
        .is_some_and(|end| end <= mapped.len());
    let taken = fits
        .then(|| {
            take_mapped_rows(
                &mapped[offset..offset + size],
                &negotiated.info,
                &negotiated.layout,
                stride,
                user_data.policy.keep_row_padding,
            )
        })
        .flatten();
    let Some((bytes, first_stride)) = taken else {
        // The buffer disagrees with the negotiated geometry: fail the
        // capture instead of pushing a malformed frame downstream.
        let _ = user_data
            .tx
            .send(FromWorker::Failed(G2gError::CapsMismatch));
        user_data.negotiated = None;
        return;
    };
    let _ = user_data.tx.send(FromWorker::Frame {
        bytes,
        first_stride,
    });
}

/// Share one dma-buf buffer downstream and hold it until every share of the frame
/// is gone. The descriptor handed on is a `dup` of the producer's, so downstream
/// owns what its [`OwnedDmaBuf`] closes while the buffer itself is only recycled
/// here.
fn share_dmabuf_frame(stream: &pw::stream::StreamRef, user_data: &mut UserData) {
    // Take back whatever downstream finished with, so the producer keeps buffers
    // even when this callback goes on to hold one.
    requeue_released(stream, &mut user_data.held.borrow_mut());
    let Some(negotiated) = user_data.negotiated.as_ref() else {
        return;
    };
    // SAFETY: the raw dequeue is what lets a buffer outlive this callback (the safe
    // wrapper requeues on drop). The pointer is null-checked, and every path below
    // either queues it back exactly once or hands it to `held`, which does.
    let buffer = unsafe { stream.dequeue_raw_buffer() };
    if buffer.is_null() {
        return;
    }
    // SAFETY: `buffer` is a live buffer of this stream, so its `spa_buffer` and the
    // `datas` array it points at are valid for the length it reports.
    let blocks = unsafe { read_blocks(buffer) };
    let requeue = |buffer| {
        // SAFETY: `buffer` was just dequeued from `stream` and is queued back once.
        unsafe { stream.queue_raw_buffer(buffer) };
    };
    let fail = |user_data: &mut UserData, error| {
        requeue(buffer);
        let _ = user_data.tx.send(FromWorker::Failed(error));
        user_data.negotiated = None;
    };
    match dmabuf_frame(&blocks, &negotiated.info) {
        DmaBufFrame::Empty => requeue(buffer),
        DmaBufFrame::Unusable => fail(user_data, G2gError::CapsMismatch),
        DmaBufFrame::Ready { fd, stride, offset } => {
            extern "C" {
                fn dup(fd: i32) -> i32;
            }
            // SAFETY: `fd` is the producer's live dma-buf descriptor; `dup` either
            // returns a fresh descriptor for the same buffer or fails.
            let shared_fd = unsafe { dup(fd) };
            if shared_fd < 0 {
                fail(user_data, G2gError::Hardware(HardwareError::Other));
                return;
            }
            // SAFETY: `shared_fd` is a fresh descriptor nobody else owns, so
            // `OwnedDmaBuf` closing it on its last drop is correct.
            let frame = unsafe { OwnedDmaBuf::from_raw(shared_fd, stride, offset) };
            user_data.held.borrow_mut().push(HeldBuffer {
                buffer,
                frame: frame.clone(),
            });
            let _ = user_data.tx.send(FromWorker::DmaBuf(frame));
        }
    }
}

/// The `spa_data` blocks of a dequeued buffer, as much of each as the dma-buf path
/// reads. Stops at [`MAX_BLOCKS`]: only a single-block buffer is usable, so a
/// longer one is read far enough to be rejected and no further.
///
/// # Safety
/// `buffer` must be a buffer this stream handed out and has not taken back.
unsafe fn read_blocks(buffer: *mut pw_sys::pw_buffer) -> Vec<DataBlock> {
    // SAFETY: the caller certifies `buffer` is live, so `spa_buffer` is its own
    // valid buffer description and `datas` covers `n_datas` entries.
    unsafe {
        let spa_buffer = (*buffer).buffer;
        if spa_buffer.is_null() || (*spa_buffer).datas.is_null() {
            return Vec::new();
        }
        let count = ((*spa_buffer).n_datas as usize).min(MAX_BLOCKS);
        let datas = (*spa_buffer).datas;
        let mut blocks = Vec::with_capacity(count);
        for index in 0..count {
            let data = &*datas.add(index);
            let (offset, size, stride) = if data.chunk.is_null() {
                (0, 0, 0)
            } else {
                let chunk = &*data.chunk;
                (chunk.offset, chunk.size, chunk.stride)
            };
            blocks.push(DataBlock {
                data_type: data.type_,
                fd: data.fd,
                offset,
                size,
                stride,
                maxsize: data.maxsize,
            });
        }
        blocks
    }
}

/// Hand the producer back every held buffer whose frame downstream has released.
fn requeue_released(stream: &pw::stream::StreamRef, held: &mut Vec<HeldBuffer>) {
    retire_released(held, |buffer| {
        // SAFETY: `buffer` came from this stream's dequeue and has not been queued
        // back yet; the frame's last share is gone, so nothing is reading it.
        unsafe { stream.queue_raw_buffer(buffer) };
    });
}

/// Drop the held buffers whose frame has no share left but the held one, passing
/// each to `requeue`. Split from the stream call so the lend / release bookkeeping
/// can be tested without a live node.
fn retire_released(held: &mut Vec<HeldBuffer>, mut requeue: impl FnMut(*mut pw_sys::pw_buffer)) {
    held.retain(|entry| {
        if entry.frame.share_count() > 1 {
            return true;
        }
        requeue(entry.buffer);
        false
    });
}

/// Parse a fixated `Format` param into the negotiated geometry / format.
fn parse_format(param: &spa::pod::Pod) -> Result<VideoInfo, G2gError> {
    let (media_type, media_subtype) =
        spa::param::format_utils::parse_format(param).map_err(|_| G2gError::CapsMismatch)?;
    if media_type != spa::param::format::MediaType::Video
        || media_subtype != spa::param::format::MediaSubtype::Raw
    {
        return Err(G2gError::CapsMismatch);
    }
    let mut raw = spa::param::video::VideoInfoRaw::new();
    raw.parse(param).map_err(|_| G2gError::CapsMismatch)?;
    VideoInfo::from_spa(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4x2 BGRA frame the daemon wrote at a 24-byte pitch: 16 bytes of picture
    /// and 8 of padding per row. Returned padded, then tight.
    fn padded_bgra() -> (Vec<u8>, Vec<u8>) {
        let mut padded = Vec::new();
        let mut tight = Vec::new();
        for row in 0..2u8 {
            let picture: Vec<u8> = (0..16).map(|i| row * 16 + i).collect();
            tight.extend_from_slice(&picture);
            padded.extend_from_slice(&picture);
            padded.extend_from_slice(&[0xff; 8]);
        }
        (padded, tight)
    }

    fn bgra_info() -> VideoInfo {
        VideoInfo {
            format: RawVideoFormat::Bgra8,
            width: 4,
            height: 2,
            fps_num: 30,
            fps_denom: 1,
        }
    }

    /// M1059: a mapped chunk is packed tight on the way out, unless a consumer
    /// asked to read the rows where the daemon left them.
    #[test]
    fn a_mapped_chunk_is_packed_unless_a_consumer_asked_for_its_pitch() {
        let (padded, tight) = padded_bgra();
        let info = bgra_info();
        let layout = PlaneLayout::new(info.format, info.width, info.height).expect("a layout");

        let (bytes, stride) =
            take_mapped_rows(&padded, &info, &layout, 24, false).expect("the chunk holds a frame");
        assert_eq!(bytes, tight);
        assert_eq!(stride, 0, "packed rows carry no pitch");

        let (bytes, stride) =
            take_mapped_rows(&padded, &info, &layout, 24, true).expect("the chunk holds a frame");
        assert_eq!(bytes, padded, "the daemon's rows travel untouched");
        assert_eq!(stride, 24);
    }

    /// Rows that are already tight have nothing to declare, so the demand
    /// changes nothing about them.
    #[test]
    fn tight_rows_are_handed_over_the_same_either_way() {
        let (_, tight) = padded_bgra();
        let info = bgra_info();
        let layout = PlaneLayout::new(info.format, info.width, info.height).expect("a layout");
        for requested in [false, true] {
            assert_eq!(
                take_mapped_rows(&tight, &info, &layout, 16, requested),
                Some((tight.clone(), 0))
            );
        }
    }

    /// A chunk shorter than the frame its pitch claims fails the capture rather
    /// than handing downstream a buffer that reads out of bounds.
    #[test]
    fn a_chunk_short_of_the_frame_is_rejected() {
        let (padded, _) = padded_bgra();
        let info = bgra_info();
        let layout = PlaneLayout::new(info.format, info.width, info.height).expect("a layout");
        for requested in [false, true] {
            assert_eq!(
                take_mapped_rows(&padded[..39], &info, &layout, 24, requested),
                None
            );
        }
    }

    #[test]
    fn builders_set_requested_config() {
        let src = PipeWireVideoSrc::new()
            .with_target("g2g-node")
            .with_size(1280, 720)
            .with_fps(60)
            .with_format(RawVideoFormat::Nv12)
            .with_frame_limit(7);
        assert_eq!(src.target, "g2g-node");
        assert_eq!(
            (src.req_width, src.req_height, src.req_fps),
            (1280, 720, 60)
        );
        assert_eq!(src.pin_format, Some(RawVideoFormat::Nv12));
        assert_eq!(src.frame_limit, 7);
        // no limit is the default; the builder spells 0 the way the property
        // does, as a count of zero rather than "forever"
        assert_eq!(PipeWireVideoSrc::new().frame_limit, u64::MAX);
        assert_eq!(PipeWireVideoSrc::new().with_frame_limit(0).frame_limit, 0);
        // mapped buffers unless the caller asks for dma-buf
        assert_eq!(src.io_mode, IoMode::MemoryMap);
        assert_eq!(
            PipeWireVideoSrc::new().with_io_mode(IoMode::DmaBuf).io_mode,
            IoMode::DmaBuf
        );
    }

    /// `io-mode` decides the output domain, the formats on offer and the leading
    /// format, all before the stream connects: the domain is part of negotiation,
    /// so it cannot be picked per buffer.
    #[test]
    fn the_dmabuf_mode_fixes_the_domain_and_the_format_offer() {
        let mut src = PipeWireVideoSrc::new();
        assert_eq!(src.output_memory(), MemoryDomainKind::System);
        assert_eq!(src.format_offer(), FormatOffer::All);
        assert_eq!(src.format(), Ok(RawVideoFormat::I420));

        src.set_property("io-mode", PropValue::Str("dmabuf".to_string()))
            .expect("known prop");
        assert_eq!(src.io_mode, IoMode::DmaBuf);
        assert_eq!(
            src.get_property("io-mode"),
            Some(PropValue::Str("dmabuf".to_string()))
        );
        assert_eq!(src.output_memory(), MemoryDomainKind::DmaBuf);
        assert_eq!(src.format_offer(), FormatOffer::SinglePlane);
        // a single-plane format leads, because one dma-buf block is all a frame gets
        assert_eq!(src.format(), Ok(RawVideoFormat::Bgra8));
        assert!(matches!(
            src.caps(),
            Ok(Caps::RawVideo {
                format: RawVideoFormat::Bgra8,
                ..
            })
        ));

        // a planar pin has no single-block form: rejected up front, not at the
        // first buffer
        src.set_property("format", PropValue::Str("NV12".to_string()))
            .expect("known prop");
        assert_eq!(src.format(), Err(G2gError::CapsMismatch));
        assert_eq!(src.caps(), Err(G2gError::CapsMismatch));
        // and the same pin is fine once frames are copied out again
        src.set_property("io-mode", PropValue::Str("mmap".to_string()))
            .expect("known prop");
        assert_eq!(src.format(), Ok(RawVideoFormat::Nv12));
        // a mode the element does not have is never silently ignored
        assert_eq!(
            src.set_property("io-mode", PropValue::Str("userptr".to_string())),
            Err(PropError::Value)
        );
        assert_eq!(src.io_mode, IoMode::MemoryMap);
    }

    /// The dma-buf path lends the producer's buffer out with the frame: it goes
    /// back only once every share of that frame is gone, so the producer cannot
    /// overwrite a frame downstream is still reading.
    #[test]
    fn a_held_buffer_goes_back_when_its_frame_is_released() {
        use std::os::fd::IntoRawFd;

        let lend = |address: usize| {
            let fd = std::fs::File::open("/dev/null")
                .expect("/dev/null opens")
                .into_raw_fd();
            // SAFETY: a fresh descriptor owned by this test alone, which the
            // `OwnedDmaBuf` closes once on its last drop.
            let frame = unsafe { OwnedDmaBuf::from_raw(fd, 64, 0) };
            let held = HeldBuffer {
                buffer: core::ptr::without_provenance_mut(address),
                frame: frame.clone(),
            };
            (held, frame)
        };
        // the pointers stand in for the producer's buffers: passed back to the
        // requeue, never read
        let (first_held, first_frame) = lend(1);
        let (second_held, second_frame) = lend(2);
        let mut held = Vec::from([first_held, second_held]);

        let mut recycled = Vec::new();
        retire_released(&mut held, |buffer| recycled.push(buffer.addr()));
        assert!(recycled.is_empty(), "both frames are still downstream");
        assert_eq!(held.len(), 2);

        drop(first_frame);
        retire_released(&mut held, |buffer| recycled.push(buffer.addr()));
        assert_eq!(recycled, [1]);
        assert_eq!(held.len(), 1);

        // a second share of the same frame keeps its buffer out (a tee branch)
        let branch = second_frame.clone();
        drop(second_frame);
        retire_released(&mut held, |buffer| recycled.push(buffer.addr()));
        assert_eq!(recycled, [1]);
        drop(branch);
        retire_released(&mut held, |buffer| recycled.push(buffer.addr()));
        assert_eq!(recycled, [1, 2]);
        assert!(held.is_empty());
    }

    #[test]
    fn caps_are_fixed_and_reject_unusable_geometry() {
        let src = PipeWireVideoSrc::new().with_size(320, 240).with_fps(25);
        assert_eq!(
            src.caps(),
            Ok(Caps::RawVideo {
                format: RawVideoFormat::I420,
                width: Dim::Fixed(320),
                height: Dim::Fixed(240),
                framerate: Rate::Fixed(25 << 16),
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN
            })
        );
        // never advertise Dim::Any / a zero dimension: fixate has to be a no-op
        assert_eq!(
            PipeWireVideoSrc::new().with_size(0, 240).caps(),
            Err(G2gError::CapsMismatch)
        );
        assert_eq!(
            PipeWireVideoSrc::new().with_size(320, MAX_DIM + 1).caps(),
            Err(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn properties_round_trip_through_the_launch_path() {
        let mut src = PipeWireVideoSrc::new();
        for (name, value) in [
            ("width", PropValue::Uint(1920)),
            ("height", PropValue::Uint(1080)),
            ("framerate", PropValue::Uint(50)),
            ("target-object", PropValue::Str("cam0".to_string())),
            ("format", PropValue::Str("YUY2".to_string())),
            ("io-mode", PropValue::Str("dmabuf".to_string())),
            ("num-buffers", PropValue::Int(30)),
        ] {
            src.set_property(name, value.clone()).expect("known prop");
            assert_eq!(src.get_property(name), Some(value));
        }
        assert_eq!(
            (src.req_width, src.req_height, src.req_fps),
            (1920, 1080, 50)
        );
        assert_eq!(src.target, "cam0");
        assert_eq!(src.pin_format, Some(RawVideoFormat::Yuyv));
        assert_eq!(src.frame_limit, 30);
        // -1 is no limit, in both directions
        src.set_property("num-buffers", PropValue::Int(-1))
            .expect("known prop");
        assert_eq!(src.frame_limit, u64::MAX);
        assert_eq!(src.get_property("num-buffers"), Some(PropValue::Int(-1)));
        assert_eq!(
            src.set_property("nope", PropValue::Uint(1)),
            Err(PropError::Unknown)
        );
        // every declared property is handled by both halves
        for spec in src.properties() {
            assert!(
                src.get_property(spec.name).is_some(),
                "unhandled property {}",
                spec.name
            );
        }
    }

    #[cfg(feature = "portal")]
    #[test]
    fn the_portal_properties_reach_the_handshake_request() {
        let mut src = PipeWireVideoSrc::new();
        for (name, value) in [
            ("portal", PropValue::Bool(true)),
            ("portal-source-types", PropValue::Str("window".to_string())),
            ("portal-restore-token", PropValue::Str("tok-1".to_string())),
            ("portal-timeout", PropValue::Uint(5)),
        ] {
            src.set_property(name, value.clone()).expect("known prop");
            assert_eq!(src.get_property(name), Some(value));
        }
        let request = src.portal_request().expect("portal is on");
        assert_eq!(request.source_types, PortalSourceTypes::Window);
        assert_eq!(request.restore_token.as_deref(), Some("tok-1"));
        assert_eq!(request.timeout, core::time::Duration::from_secs(5));
        // an empty token means "ask", not "restore an empty grant"
        src.set_property("portal-restore-token", PropValue::Str(String::new()))
            .expect("known prop");
        assert!(src
            .portal_request()
            .expect("portal is on")
            .restore_token
            .is_none());
        assert!(PipeWireVideoSrc::new().portal_request().is_none());
    }

    #[cfg(feature = "portal")]
    #[test]
    fn a_portal_property_outside_its_range_is_refused() {
        let mut src = PipeWireVideoSrc::new();
        assert_eq!(
            src.set_property("portal-source-types", PropValue::Str("desktop".to_string())),
            Err(PropError::Value)
        );
        // a zero deadline would make every handshake fail before it started
        assert_eq!(
            src.set_property("portal-timeout", PropValue::Uint(0)),
            Err(PropError::Value)
        );
        assert_eq!(
            src.set_property("portal", PropValue::Str("yes".to_string())),
            Err(PropError::Type)
        );
    }

    #[cfg(feature = "portal")]
    #[test]
    fn the_portal_and_a_target_node_cannot_both_be_set() {
        // whichever comes second on the launch line fails, in either order
        let mut portal_first = PipeWireVideoSrc::new();
        portal_first
            .set_property("portal", PropValue::Bool(true))
            .expect("known prop");
        assert_eq!(
            portal_first.set_property("target-object", PropValue::Str("cam0".to_string())),
            Err(PropError::Value)
        );
        assert_eq!(portal_first.target, "");

        let mut target_first = PipeWireVideoSrc::new();
        target_first
            .set_property("target-object", PropValue::Str("cam0".to_string()))
            .expect("known prop");
        assert_eq!(
            target_first.set_property("portal", PropValue::Bool(true)),
            Err(PropError::Value)
        );
        assert!(!target_first.portal);

        // clearing the target frees the portal again
        target_first
            .set_property("target-object", PropValue::Str(String::new()))
            .expect("known prop");
        assert!(target_first
            .set_property("portal", PropValue::Bool(true))
            .is_ok());
    }

    #[cfg(feature = "portal")]
    #[test]
    fn the_builders_cannot_smuggle_the_conflict_past_negotiation() {
        let mut src = PipeWireVideoSrc::new()
            .with_target("cam0")
            .with_portal(PortalSourceTypes::Any);
        let caps = src.caps().expect("default caps");
        assert_eq!(
            src.configure_pipeline(&caps).err(),
            Some(G2gError::Hardware(HardwareError::Other))
        );
        assert!(!src.configured);
    }

    #[cfg(feature = "portal")]
    #[test]
    fn the_setup_deadline_covers_every_portal_step() {
        let request = PipeWireVideoSrc::new()
            .with_portal(PortalSourceTypes::Monitor)
            .with_portal_timeout_secs(20)
            .portal_request();
        assert_eq!(
            setup_deadline(&request),
            core::time::Duration::from_secs(20 * 3) + CONNECT_TIMEOUT
        );
        assert_eq!(setup_deadline(&None), CONNECT_TIMEOUT);
    }

    /// A pinned format is what the caps advertise and the only format the connect
    /// pod offers, so the node cannot hand back something else mid-stream.
    #[test]
    fn a_pinned_format_drives_the_caps_and_the_connect_pod() {
        use crate::pwvideo::spa_format;
        use spa::pod::deserialize::PodDeserializer;
        use spa::pod::{ChoiceValue, Value};
        use spa::utils::{Choice, ChoiceEnum, Id};

        let mut src = PipeWireVideoSrc::new().with_size(320, 240).with_fps(30);
        src.set_property("format", PropValue::Str("yuy2".to_string()))
            .expect("known prop");
        assert_eq!(
            src.caps(),
            Ok(Caps::RawVideo {
                format: RawVideoFormat::Yuyv,
                width: Dim::Fixed(320),
                height: Dim::Fixed(240),
                framerate: Rate::Fixed(30 << 16),
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN
            })
        );

        let bytes =
            format_pod_bytes(src.format().unwrap(), src.format_offer(), 320, 240, 30).expect("pod");
        let (_, value) = PodDeserializer::deserialize_any_from(&bytes).expect("pod deserializes");
        let Value::Object(obj) = value else {
            panic!("expected an object pod");
        };
        let format = obj
            .properties
            .iter()
            .find(|p| p.key == spa::param::format::FormatProperties::VideoFormat.as_raw())
            .map(|p| p.value.clone())
            .expect("pod carries a format");
        let Value::Choice(ChoiceValue::Id(Choice(_, ChoiceEnum::Enum { alternatives, .. }))) =
            format
        else {
            panic!("format is an enum choice");
        };
        let pinned = spa_format(RawVideoFormat::Yuyv).unwrap();
        assert_eq!(alternatives, Vec::from([Id(pinned.as_raw())]));

        // an empty pin is the open negotiation the element defaults to
        src.set_property("format", PropValue::Str(String::new()))
            .expect("known prop");
        assert_eq!(src.pin_format, None);
        // and a name the SPA table has no entry for never silently defaults
        for bad in ["y42b", "p010", "nonsense"] {
            assert_eq!(
                src.set_property("format", PropValue::Str(bad.to_string())),
                Err(PropError::Value),
                "{bad} is not a capture format"
            );
        }
        assert_eq!(src.pin_format, None);
    }

    #[test]
    fn configure_marks_the_element_ready() {
        let mut src = PipeWireVideoSrc::new();
        assert!(!src.configured);
        let caps = src.caps().expect("default caps");
        src.configure_pipeline(&caps).expect("fixed caps accepted");
        assert!(src.configured);
    }

    #[test]
    fn pad_template_is_a_raw_video_source() {
        use g2g_core::PadDirection;
        assert!(PipeWireVideoSrc::pad_template(PadDirection::Source).is_some());
        assert!(PipeWireVideoSrc::pad_template(PadDirection::Sink).is_none());
    }

    /// The advertised caps and the negotiated caps compare equal when the node
    /// hands back exactly what was asked for, so no `CapsChanged` is emitted;
    /// any difference produces one.
    #[test]
    fn negotiated_caps_only_differ_when_the_node_differs() {
        let src = PipeWireVideoSrc::new().with_size(320, 240).with_fps(30);
        let same = VideoInfo {
            format: RawVideoFormat::I420,
            width: 320,
            height: 240,
            fps_num: 30,
            fps_denom: 1,
        };
        assert_eq!(same.caps(), src.caps().unwrap());
        let other = VideoInfo {
            format: RawVideoFormat::Yuyv,
            ..same
        };
        assert_ne!(other.caps(), src.caps().unwrap());
    }

    /// The `param_changed` parse: a fixated `Format` param yields the negotiated
    /// info, and a format the element cannot carry fails instead of defaulting.
    #[test]
    fn param_changed_parse_accepts_a_fixated_format_and_rejects_the_rest() {
        use crate::pwvideo::fixed_format_pod_bytes;
        use spa::param::video::VideoFormat;

        let parse = |bytes: &[u8]| {
            let pod = spa::pod::Pod::from_bytes(bytes).expect("pod bytes parse");
            parse_format(pod)
        };
        assert_eq!(
            parse(&fixed_format_pod_bytes(VideoFormat::YUY2, 640, 360, 25)),
            Ok(VideoInfo {
                format: RawVideoFormat::Yuyv,
                width: 640,
                height: 360,
                fps_num: 25,
                fps_denom: 1,
            })
        );
        // YV12 is a real SPA format the element does not carry
        assert_eq!(
            parse(&fixed_format_pod_bytes(VideoFormat::YV12, 640, 360, 25)),
            Err(G2gError::CapsMismatch)
        );
        // and a nonsense geometry never reaches an allocation
        assert_eq!(
            parse(&fixed_format_pod_bytes(VideoFormat::I420, 0, 360, 25)),
            Err(G2gError::CapsMismatch)
        );
    }
}

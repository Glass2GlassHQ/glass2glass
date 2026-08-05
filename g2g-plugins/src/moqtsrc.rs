//! MoQ Transport subscriber source (`moqtsrc`, M903, `moqt` feature): subscribes
//! to a broadcast on an IETF MoQT relay over WebTransport and emits the
//! fragmented-MP4 byte stream it carries.
//!
//! ```text
//! moqtsrc location=https://relay:4443/ namespace=live/cam ! fmp4demux ! ...
//! ```
//!
//! It is the inverse of [`MoqtSink`](crate::moqtsink), and plays either side's
//! broadcast: Cloudflare's `moq-pub` and `moqtsink` publish the same track
//! layout.
//!
//! - the `.catalog` track names the media tracks and the init track. Without it
//!   (`catalog=false`, or a publisher that omits it) the defaults are the
//!   reference ones: `0.mp4` for the init track, `{track_id}.m4s` for media.
//! - the init track's single object is the `ftyp`+`moov`, emitted first so the
//!   demuxer downstream sees a whole fMP4 stream.
//! - each media object is one `moof`+`mdat` fragment, emitted in group and
//!   object order (see [`reassembly`](crate::moqt::reassembly) for the ordering
//!   policy and its bounds).
//!
//! The stream ends on the publisher's PUBLISH_DONE for the media subscription,
//! on the session closing, or on `num-buffers` / `timeout`.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::LogSource;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    g2g_debug, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, HardwareError, MemoryDomain, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use web_transport_quinn::SendStream;

use crate::moqt::catalog::{self, CatalogTrack};
use crate::moqt::coding::{Params, TrackName, TrackNamespace};
use crate::moqt::message::{request_error_code, ControlMessage};
use crate::moqt::reassembly::Reassembler;
use crate::moqt::session::{implementation_name, DataEvent, MoqtSession};
use crate::moqt::v18;
use crate::moqt::{negotiated_version, parse_versions, MoqtVersion};

/// A relay we cannot talk to, or one that violates the protocol, is the same
/// thing to the pipeline: no stream.
fn session_err() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// One track we asked the relay for.
#[derive(Debug)]
struct Subscription {
    request_id: u64,
    /// Set by SUBSCRIBE_OK; the alias is how data streams name this track.
    track_alias: Option<u64>,
    /// The draft-18 request stream's send half. Draft-18 has no UNSUBSCRIBE:
    /// resetting this is how the subscription is cancelled at shutdown.
    request_tx: Option<SendStream>,
    reassembler: Reassembler,
    /// Payloads in order, waiting to be emitted.
    ready: VecDeque<Vec<u8>>,
    /// PUBLISH_DONE arrived, or the relay refused the subscription.
    ended: bool,
}

impl Subscription {
    fn new(request_id: u64, max_groups: usize, max_bytes: usize) -> Self {
        Self {
            request_id,
            track_alias: None,
            request_tx: None,
            reassembler: Reassembler::new(max_groups, max_bytes),
            ready: VecDeque::new(),
            ended: false,
        }
    }
}

/// Subscribes to a MoQ Transport broadcast and emits its fMP4 byte stream.
#[derive(Debug)]
pub struct MoqtSrc {
    location: String,
    cert_hashes: String,
    namespace: String,
    track_name: String,
    init_track: String,
    catalog_track: String,
    use_catalog: bool,
    max_request_id: u64,
    versions: String,
    max_groups: u64,
    max_buffer_bytes: u64,
    max_object_bytes: u64,
    num_buffers: u64,
    timeout_ms: u64,

    configured: bool,
    /// Media track the catalog (or the fallback) selected, for tests and logs.
    selected_track: String,
    objects_received: u64,
    groups_dropped: u64,
    objects_dropped: u64,
}

impl Default for MoqtSrc {
    fn default() -> Self {
        Self::new("https://127.0.0.1:4443/", "g2g")
    }
}

impl MoqtSrc {
    /// Subscribe to `namespace` (a `/`-separated path) on the relay at
    /// `location`.
    pub fn new(location: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            location: location.into(),
            cert_hashes: String::new(),
            namespace: namespace.into(),
            track_name: String::new(),
            init_track: String::from("0.mp4"),
            catalog_track: String::from(".catalog"),
            use_catalog: true,
            max_request_id: 100,
            versions: String::from("18,16"),
            max_groups: 8,
            max_buffer_bytes: 32 * 1024 * 1024,
            max_object_bytes: 16 * 1024 * 1024,
            num_buffers: 0,
            timeout_ms: 15_000,
            configured: false,
            selected_track: String::new(),
            objects_received: 0,
            groups_dropped: 0,
            objects_dropped: 0,
        }
    }

    /// Accept only relay certificates whose SHA-256 digest is listed (hex,
    /// comma-separated) instead of requiring a system root.
    pub fn with_server_certificate_hashes(mut self, hashes: impl Into<String>) -> Self {
        self.cert_hashes = hashes.into();
        self
    }

    /// Subscribe to this media track by name instead of the catalog's first.
    pub fn with_track_name(mut self, name: impl Into<String>) -> Self {
        self.track_name = name.into();
        self
    }

    /// Stop after `n` frames (the init segment counts as one).
    pub fn with_num_buffers(mut self, n: u64) -> Self {
        self.num_buffers = n;
        self
    }

    /// The media track actually subscribed to, once the stream has started.
    pub fn selected_track(&self) -> &str {
        &self.selected_track
    }

    /// Frames the last run handed downstream, init segment included.
    pub fn objects_received(&self) -> u64 {
        self.objects_received
    }

    /// Groups the last run's ordering policy abandoned: a mid-group join, or a
    /// group that never completed inside the buffering bounds.
    pub fn groups_dropped(&self) -> u64 {
        self.groups_dropped
    }

    /// Objects the last run threw away: lost (a hole in a group that ended,
    /// which is what a dropped datagram leaves), late, or duplicated.
    pub fn objects_dropped(&self) -> u64 {
        self.objects_dropped
    }

    fn output_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        }
    }
}

/// The subscriptions and the reorder state they share, identical whichever
/// draft delivers the events: ordering is not version-specific.
struct SubsState {
    max_groups: usize,
    max_bytes: usize,
    subs: Vec<Subscription>,
    /// Data events whose track alias no subscription claims yet. The control
    /// plane and the data streams are different QUIC streams, so a subgroup
    /// can arrive before the SUBSCRIBE_OK that names its alias.
    orphans: VecDeque<DataEvent>,
    orphan_bytes: usize,
}

impl SubsState {
    fn new(max_groups: usize, max_bytes: usize) -> Self {
        Self {
            max_groups,
            max_bytes,
            subs: Vec::new(),
            orphans: VecDeque::new(),
            orphan_bytes: 0,
        }
    }

    fn add(&mut self, request_id: u64) -> usize {
        self.subs.push(Subscription::new(
            request_id,
            self.max_groups,
            self.max_bytes,
        ));
        self.subs.len() - 1
    }

    fn by_request(&mut self, id: u64) -> Option<&mut Subscription> {
        self.subs.iter_mut().find(|s| s.request_id == id)
    }

    fn handle_data(&mut self, event: DataEvent) {
        let alias = match &event {
            DataEvent::StreamOpened { track_alias, .. }
            | DataEvent::StreamClosed { track_alias, .. }
            | DataEvent::Object { track_alias, .. } => *track_alias,
        };
        let Some(at) = self.subs.iter().position(|s| s.track_alias == Some(alias)) else {
            self.hold_orphan(event);
            return;
        };
        self.apply(at, event);
    }

    fn apply(&mut self, at: usize, event: DataEvent) {
        let sub = &mut self.subs[at];
        match event {
            DataEvent::StreamOpened { group_id, .. } => sub.reassembler.stream_opened(group_id),
            DataEvent::StreamClosed { group_id, .. } => sub.reassembler.stream_closed(group_id),
            DataEvent::Object { object, .. } => sub.reassembler.push(object),
        }
        sub.ready.extend(sub.reassembler.drain());
    }

    /// Hold a data event for an alias no subscription has yet, bounded by the
    /// same byte budget the reassembler uses.
    fn hold_orphan(&mut self, event: DataEvent) {
        if let DataEvent::Object { object, .. } = &event {
            self.orphan_bytes = self.orphan_bytes.saturating_add(object.payload.len());
        }
        self.orphans.push_back(event);
        while self.orphan_bytes > self.max_bytes {
            match self.orphans.pop_front() {
                Some(DataEvent::Object { object, .. }) => {
                    self.orphan_bytes = self.orphan_bytes.saturating_sub(object.payload.len());
                }
                Some(_) => {}
                None => break,
            }
        }
    }

    /// Replay the held events for an alias that just became known, in order.
    fn claim_orphans(&mut self, alias: u64) {
        let Some(at) = self.subs.iter().position(|s| s.track_alias == Some(alias)) else {
            return;
        };
        let held = core::mem::take(&mut self.orphans);
        for event in held {
            let matches = match &event {
                DataEvent::StreamOpened { track_alias, .. }
                | DataEvent::StreamClosed { track_alias, .. }
                | DataEvent::Object { track_alias, .. } => *track_alias == alias,
            };
            if matches {
                if let DataEvent::Object { object, .. } = &event {
                    self.orphan_bytes = self.orphan_bytes.saturating_sub(object.payload.len());
                }
                self.apply(at, event);
            } else {
                self.orphans.push_back(event);
            }
        }
    }
}

/// The live half of a draft-16 run: the session plus every subscription on it.
struct Driver {
    namespace: TrackNamespace,
    timeout_ms: u64,
    session: MoqtSession,
    data: tokio::sync::mpsc::UnboundedReceiver<DataEvent>,
    state: SubsState,
    /// The session ended (control stream closed, or the relay went away).
    closed: bool,
}

impl LogSource for MoqtSrc {
    fn log_category(&self) -> &'static str {
        "moqtsrc"
    }
}

impl LogSource for Driver {
    fn log_category(&self) -> &'static str {
        "moqtsrc"
    }
}

impl LogSource for Driver18 {
    fn log_category(&self) -> &'static str {
        "moqtsrc"
    }
}

/// What one [`Driver::pump`] achieved.
#[derive(Debug, PartialEq, Eq)]
enum Pumped {
    Applied,
    /// The session ended: the control stream closed or the relay went away.
    Ended,
    /// Nothing arrived within `timeout`.
    TimedOut,
}

impl Driver {
    /// Send SUBSCRIBE for `name` and return its index in `subs`.
    async fn subscribe(&mut self, name: &str) -> Result<usize, G2gError> {
        let id = self.session.allocate_request_id().ok_or_else(session_err)?;
        self.session
            .send(&ControlMessage::Subscribe {
                id,
                namespace: self.namespace.clone(),
                track_name: TrackName::new(name),
                params: Params::new(),
            })
            .await?;
        g2g_debug!(self, "SUBSCRIBE {name} as request {id}");
        Ok(self.state.add(id))
    }

    /// Wait for one event and apply it.
    async fn pump(&mut self) -> Result<Pumped, G2gError> {
        if self.closed {
            return Ok(Pumped::Ended);
        }
        let timeout = self.timeout_ms;
        let step = {
            let session = &mut self.session;
            let data = &mut self.data;
            let next = async move {
                tokio::select! {
                    control = session.next_control() => Some(Step::Control(control)),
                    event = data.recv() => event.map(Step::Data),
                }
            };
            if timeout == 0 {
                next.await
            } else {
                match tokio::time::timeout(Duration::from_millis(timeout), next).await {
                    Ok(step) => step,
                    Err(_) => return Ok(Pumped::TimedOut),
                }
            }
        };
        match step {
            Some(Step::Control(Some(msg))) => self.handle_control(msg).await?,
            // The control stream ended: so has the session.
            Some(Step::Control(None)) | None => {
                self.closed = true;
                return Ok(Pumped::Ended);
            }
            Some(Step::Data(event)) => self.state.handle_data(event),
        }
        Ok(Pumped::Applied)
    }

    async fn handle_control(&mut self, msg: ControlMessage) -> Result<(), G2gError> {
        g2g_debug!(self, "control: {}", msg.name());
        match msg {
            ControlMessage::SubscribeOk {
                id, track_alias, ..
            } => {
                if let Some(sub) = self.state.by_request(id) {
                    sub.track_alias = Some(track_alias);
                }
                self.state.claim_orphans(track_alias);
            }
            ControlMessage::RequestError { id, error_code, .. } => {
                g2g_debug!(self, "request {id} refused, code {error_code}");
                if let Some(sub) = self.state.by_request(id) {
                    sub.ended = true;
                }
            }
            ControlMessage::PublishDone { id, .. } => {
                if let Some(sub) = self.state.by_request(id) {
                    let tail = sub.reassembler.flush();
                    sub.ready.extend(tail);
                    sub.ended = true;
                }
            }
            ControlMessage::MaxRequestId { request_id } => {
                self.session.set_peer_max_request_id(request_id);
            }
            ControlMessage::GoAway { .. } => self.closed = true,
            // A publisher-side request we do not serve. Draft-16 §4 asks for an
            // explicit refusal rather than silence.
            ControlMessage::Publish { id, .. } | ControlMessage::RequestUpdate { id, .. } => {
                self.session
                    .send(&ControlMessage::RequestError {
                        id,
                        error_code: request_error_code::NOT_SUPPORTED,
                        retry_interval: 0,
                        reason: String::from("not supported"),
                    })
                    .await?;
            }
            // Everything else is a response to a request we did not make, or a
            // message only a publisher acts on: decoded, then ignored.
            _ => {}
        }
        Ok(())
    }

    /// UNSUBSCRIBE every live subscription and close the session.
    async fn shutdown(&mut self) {
        for sub in &self.state.subs {
            if sub.ended {
                continue;
            }
            let _ = self
                .session
                .send(&ControlMessage::Unsubscribe { id: sub.request_id })
                .await;
        }
        self.session.close("done").await;
    }
}

/// Which half of the session produced the next event.
enum Step {
    Control(Option<ControlMessage>),
    Data(DataEvent),
}

/// One response-stream message, keyed by its request id; `None` when the stream
/// ended, which terminates the request (§11.4.1).
type Response = (u64, Option<v18::message::ControlMessage>);

/// The live half of a draft-18 run. Each SUBSCRIBE opened its own bidirectional
/// stream, and a task per stream forwards its responses into one channel, so
/// the pump still waits in one place.
struct Driver18 {
    namespace: TrackNamespace,
    timeout_ms: u64,
    session: v18::session::Session18,
    data: tokio::sync::mpsc::UnboundedReceiver<DataEvent>,
    responses: tokio::sync::mpsc::UnboundedReceiver<Response>,
    response_tx: tokio::sync::mpsc::UnboundedSender<Response>,
    state: SubsState,
    closed: bool,
}

impl Driver18 {
    /// Open a request stream with SUBSCRIBE for `name` and return its index.
    async fn subscribe(&mut self, name: &str) -> Result<usize, G2gError> {
        let id = self.session.allocate_request_id();
        let (tx, rx) = self
            .session
            .open_request(&v18::message::ControlMessage::Subscribe {
                id,
                namespace: self.namespace.clone(),
                track_name: TrackName::new(name),
                params: v18::coding::MessageParams::new(),
            })
            .await?;
        g2g_debug!(self, "SUBSCRIBE {name} as request {id}");
        tokio::spawn(forward_responses(id, rx, self.response_tx.clone()));
        let at = self.state.add(id);
        self.state.subs[at].request_tx = Some(tx);
        Ok(at)
    }

    /// Wait for one event and apply it.
    async fn pump(&mut self) -> Result<Pumped, G2gError> {
        if self.closed || self.session.is_closed() {
            self.closed = true;
            return Ok(Pumped::Ended);
        }
        let timeout = self.timeout_ms;
        let step = {
            let responses = &mut self.responses;
            let data = &mut self.data;
            let next = async move {
                tokio::select! {
                    response = responses.recv() => response.map(Step18::Response),
                    event = data.recv() => event.map(Step18::Data),
                }
            };
            if timeout == 0 {
                next.await
            } else {
                match tokio::time::timeout(Duration::from_millis(timeout), next).await {
                    Ok(step) => step,
                    Err(_) => return Ok(Pumped::TimedOut),
                }
            }
        };
        match step {
            Some(Step18::Response((id, msg))) => self.handle_response(id, msg),
            Some(Step18::Data(event)) => self.state.handle_data(event),
            // Both channels ended: the session is gone.
            None => {
                self.closed = true;
                return Ok(Pumped::Ended);
            }
        }
        Ok(Pumped::Applied)
    }

    fn handle_response(&mut self, id: u64, msg: Option<v18::message::ControlMessage>) {
        use v18::message::ControlMessage as Msg;
        match msg {
            Some(Msg::SubscribeOk { track_alias, .. }) => {
                g2g_debug!(self, "request {id}: SUBSCRIBE_OK, alias {track_alias}");
                if let Some(sub) = self.state.by_request(id) {
                    sub.track_alias = Some(track_alias);
                }
                self.state.claim_orphans(track_alias);
            }
            Some(Msg::RequestError { error_code, .. }) => {
                g2g_debug!(self, "request {id} refused, code {error_code}");
                if let Some(sub) = self.state.by_request(id) {
                    sub.ended = true;
                }
            }
            // PUBLISH_DONE, or the stream ending: either way the subscription
            // is over and what the reassembler still holds is the tail.
            Some(Msg::PublishDone { .. }) | None => {
                if let Some(sub) = self.state.by_request(id) {
                    let tail = sub.reassembler.flush();
                    sub.ready.extend(tail);
                    sub.ended = true;
                }
            }
            // Anything else on a response stream is a message this subscriber
            // did not ask for: decoded, then ignored.
            Some(_) => {}
        }
    }

    /// Cancel every live subscription by resetting its request stream (§3.3.2)
    /// and close the session.
    async fn shutdown(&mut self) {
        for sub in &mut self.state.subs {
            if sub.ended {
                continue;
            }
            if let Some(tx) = sub.request_tx.as_mut() {
                let _ = tx.reset(v18::message::stream_error_code::CANCELLED);
            }
        }
        self.session
            .close(v18::message::session_error_code::NO_ERROR, "done")
            .await;
    }
}

/// Which half of the draft-18 session produced the next event.
enum Step18 {
    Response(Response),
    Data(DataEvent),
}

/// Read one request stream's responses and forward them under its request id.
/// The final `None` reports the stream ending, however it ended.
async fn forward_responses(
    id: u64,
    mut rx: web_transport_quinn::RecvStream,
    out: tokio::sync::mpsc::UnboundedSender<Response>,
) {
    let mut reader = v18::session::MessageReader::new();
    loop {
        match reader.next(&mut rx).await {
            Ok(Some(msg)) => {
                if out.send((id, Some(msg))).is_err() {
                    return;
                }
            }
            _ => {
                let _ = out.send((id, None));
                return;
            }
        }
    }
}

/// The negotiated driver, so the run loop reads one shape whichever draft the
/// server picked.
enum AnyDriver {
    V16(Driver),
    V18(Driver18),
}

impl AnyDriver {
    fn state(&mut self) -> &mut SubsState {
        match self {
            Self::V16(driver) => &mut driver.state,
            Self::V18(driver) => &mut driver.state,
        }
    }

    async fn subscribe(&mut self, name: &str) -> Result<usize, G2gError> {
        match self {
            Self::V16(driver) => driver.subscribe(name).await,
            Self::V18(driver) => driver.subscribe(name).await,
        }
    }

    async fn pump(&mut self) -> Result<Pumped, G2gError> {
        match self {
            Self::V16(driver) => driver.pump().await,
            Self::V18(driver) => driver.pump().await,
        }
    }

    async fn shutdown(&mut self) {
        match self {
            Self::V16(driver) => driver.shutdown().await,
            Self::V18(driver) => driver.shutdown().await,
        }
    }

    /// Pump until the subscription at `at` has a payload, or until it ends, the
    /// session ends, or nothing arrives within `timeout`.
    async fn first_object(&mut self, at: usize) -> Result<Option<Vec<u8>>, G2gError> {
        loop {
            if let Some(payload) = self.state().subs[at].ready.pop_front() {
                return Ok(Some(payload));
            }
            if self.state().subs[at].ended || self.pump().await? != Pumped::Applied {
                return Ok(self.state().subs[at].ready.pop_front());
            }
        }
    }
}

/// Pick the media track: the `track-name` property when set, else the catalog's
/// first entry, else the reference default for a single-track broadcast.
fn select_track(wanted: &str, tracks: &[CatalogTrack]) -> Option<CatalogTrack> {
    if !wanted.is_empty() {
        return Some(
            tracks
                .iter()
                .find(|t| t.name == wanted)
                .cloned()
                .unwrap_or(CatalogTrack {
                    name: wanted.to_string(),
                    init_track: String::new(),
                }),
        );
    }
    tracks.first().cloned()
}

fn byte_frame(bytes: Vec<u8>, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            arrival_ns: g2g_core::metrics::monotonic_ns(),
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    }
}

impl SourceLoop for MoqtSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Self::output_caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(
            Self::output_caps(),
        ))))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if *absolute_caps != Self::output_caps() {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let offered = parse_versions(&self.versions)?;
            let protocols: Vec<&str> = offered.iter().map(|v| v.protocol()).collect();
            let session =
                crate::remotewtio::dial(&self.location, &self.cert_hashes, &protocols).await?;
            let state = SubsState::new(self.max_groups as usize, self.max_buffer_bytes as usize);
            let namespace = TrackNamespace::from_path(&self.namespace);
            let mut driver = match negotiated_version(&session, &offered)? {
                MoqtVersion::V16 => {
                    let session = MoqtSession::connect_over(
                        session,
                        self.max_request_id,
                        &implementation_name(),
                    )
                    .await?;
                    let data = session.start_data_reader(self.max_object_bytes as usize);
                    AnyDriver::V16(Driver {
                        namespace,
                        timeout_ms: self.timeout_ms,
                        session,
                        data,
                        state,
                        closed: false,
                    })
                }
                MoqtVersion::V18 => {
                    let mut session = v18::session::Session18::connect_over(
                        session,
                        &implementation_name(),
                        self.max_object_bytes as usize,
                    )
                    .await?;
                    let data = session.take_data().ok_or_else(session_err)?;
                    let (response_tx, responses) = tokio::sync::mpsc::unbounded_channel();
                    AnyDriver::V18(Driver18 {
                        namespace,
                        timeout_ms: self.timeout_ms,
                        session,
                        data,
                        responses,
                        response_tx,
                        state,
                        closed: false,
                    })
                }
            };

            // The catalog names the tracks. Without it, fall back to the
            // reference layout, which is also what `moq-sub` does.
            let listed = if self.use_catalog {
                let at = driver.subscribe(&self.catalog_track).await?;
                let bytes = driver.first_object(at).await?.unwrap_or_default();
                catalog::parse(&bytes)
            } else {
                Vec::new()
            };
            let selected = select_track(&self.track_name, &listed).unwrap_or(CatalogTrack {
                name: String::from("1.m4s"),
                init_track: String::new(),
            });
            let init_track = if selected.init_track.is_empty() {
                self.init_track.clone()
            } else {
                selected.init_track.clone()
            };

            let at = driver.subscribe(&init_track).await?;
            let Some(init) = driver.first_object(at).await? else {
                driver.shutdown().await;
                return Err(session_err());
            };
            let media = driver.subscribe(&selected.name).await?;

            out.push(PipelinePacket::CapsChanged(Self::output_caps()))
                .await?;
            out.push(PipelinePacket::DataFrame(byte_frame(init, 0)))
                .await?;
            let mut emitted = 1u64;

            let limit = self.num_buffers;
            loop {
                while let Some(payload) = driver.state().subs[media].ready.pop_front() {
                    out.push(PipelinePacket::DataFrame(byte_frame(payload, emitted)))
                        .await?;
                    emitted += 1;
                    if limit != 0 && emitted >= limit {
                        break;
                    }
                }
                if (limit != 0 && emitted >= limit) || driver.state().subs[media].ended {
                    break;
                }
                match driver.pump().await? {
                    Pumped::Applied => {}
                    // A silent relay is the end of the stream: the publisher is
                    // gone, and this is what bounds a live subscription that
                    // nothing else stops.
                    Pumped::Ended | Pumped::TimedOut => break,
                }
            }

            let stats = driver.state().subs[media].reassembler.stats();
            driver.shutdown().await;
            self.selected_track = selected.name;
            self.objects_received = emitted;
            self.groups_dropped = stats.groups_dropped;
            self.objects_dropped = stats.objects_dropped;
            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MoQ Transport source",
            "Source/Network",
            "Subscribes to an IETF MoQ Transport broadcast over WebTransport and emits its fragmented-MP4 stream",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MOQTSRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let string = |v: &PropValue| v.as_str().map(ToString::to_string).ok_or(PropError::Type);
        let uint = |v: &PropValue| v.as_uint().ok_or(PropError::Type);
        match name {
            "location" => self.location = string(&value)?,
            "namespace" => self.namespace = string(&value)?,
            "track-name" => self.track_name = string(&value)?,
            "init-track-name" => self.init_track = string(&value)?,
            "catalog-track-name" => self.catalog_track = string(&value)?,
            "server-certificate-hashes" => self.cert_hashes = string(&value)?,
            "catalog" => self.use_catalog = value.as_bool().ok_or(PropError::Type)?,
            "max-request-id" => self.max_request_id = uint(&value)?,
            "versions" => {
                let list = string(&value)?;
                parse_versions(&list).map_err(|_| PropError::Value)?;
                self.versions = list;
            }
            "max-groups" => self.max_groups = uint(&value)?,
            "max-buffer-bytes" => self.max_buffer_bytes = uint(&value)?,
            "max-object-size" => self.max_object_bytes = uint(&value)?,
            "num-buffers" => self.num_buffers = uint(&value)?,
            "timeout" => self.timeout_ms = uint(&value)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.location.clone())),
            "namespace" => Some(PropValue::Str(self.namespace.clone())),
            "track-name" => Some(PropValue::Str(self.track_name.clone())),
            "init-track-name" => Some(PropValue::Str(self.init_track.clone())),
            "catalog-track-name" => Some(PropValue::Str(self.catalog_track.clone())),
            "server-certificate-hashes" => Some(PropValue::Str(self.cert_hashes.clone())),
            "catalog" => Some(PropValue::Bool(self.use_catalog)),
            "max-request-id" => Some(PropValue::Uint(self.max_request_id)),
            "versions" => Some(PropValue::Str(self.versions.clone())),
            "max-groups" => Some(PropValue::Uint(self.max_groups)),
            "max-buffer-bytes" => Some(PropValue::Uint(self.max_buffer_bytes)),
            "max-object-size" => Some(PropValue::Uint(self.max_object_bytes)),
            "num-buffers" => Some(PropValue::Uint(self.num_buffers)),
            "timeout" => Some(PropValue::Uint(self.timeout_ms)),
            _ => None,
        }
    }
}

static MOQTSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "WebTransport URL of the MoQT relay (e.g. https://host:4443/)",
    )
    .with_default("https://127.0.0.1:4443/"),
    PropertySpec::new(
        "namespace",
        PropKind::Str,
        "broadcast namespace to subscribe to, a /-separated path",
    )
    .with_default("g2g"),
    PropertySpec::new(
        "track-name",
        PropKind::Str,
        "media track to play; empty takes the catalog's first track",
    ),
    PropertySpec::new(
        "init-track-name",
        PropKind::Str,
        "track carrying the ftyp+moov init segment, when the catalog names none",
    )
    .with_default("0.mp4"),
    PropertySpec::new(
        "catalog-track-name",
        PropKind::Str,
        "track carrying the JSON catalog",
    )
    .with_default(".catalog"),
    PropertySpec::new(
        "catalog",
        PropKind::Bool,
        "read the catalog track to discover the tracks",
    )
    .with_default("true"),
    PropertySpec::new(
        "max-request-id",
        PropKind::Uint,
        "MAX_REQUEST_ID advertised to the relay in CLIENT_SETUP (draft-16 sessions only)",
    )
    .with_default("100"),
    PropertySpec::new(
        "versions",
        PropKind::Str,
        "MoQ Transport draft versions offered on CONNECT, comma-separated in preference order; the server's pick decides",
    )
    .with_default("18,16"),
    PropertySpec::new(
        "max-groups",
        PropKind::Uint,
        "groups held while reordering; the oldest is dropped past this",
    )
    .with_default("8"),
    PropertySpec::new(
        "max-buffer-bytes",
        PropKind::Uint,
        "bytes held while reordering; the oldest group is dropped past this",
    )
    .with_default("33554432"),
    PropertySpec::new(
        "max-object-size",
        PropKind::Uint,
        "largest single object accepted from the relay, in bytes",
    )
    .with_default("16777216"),
    PropertySpec::new(
        "num-buffers",
        PropKind::Uint,
        "stop after this many frames, init segment included (0 = unlimited)",
    )
    .with_default("0"),
    PropertySpec::new(
        "timeout",
        PropKind::Uint,
        "give up if nothing arrives for this many ms; before the first frame that fails the run, after it ends the stream (0 = wait forever)",
    )
    .with_default("15000"),
    PropertySpec::new(
        "server-certificate-hashes",
        PropKind::Str,
        "accept only relay certificates with these SHA-256 digests (hex, comma-separated); empty = system roots",
    ),
];

impl PadTemplates for MoqtSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(MoqtSrc::output_caps()))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_selection_prefers_the_property_then_the_catalog() {
        let listed = Vec::from([
            CatalogTrack {
                name: String::from("1.m4s"),
                init_track: String::from("0.mp4"),
            },
            CatalogTrack {
                name: String::from("2.m4s"),
                init_track: String::from("0.mp4"),
            },
        ]);
        assert_eq!(
            select_track("", &listed).map(|t| t.name).as_deref(),
            Some("1.m4s")
        );
        assert_eq!(
            select_track("2.m4s", &listed)
                .map(|t| t.init_track)
                .as_deref(),
            Some("0.mp4"),
            "a named track keeps the catalog's init track"
        );
        // A name the catalog does not list is still subscribed to: the
        // publisher may serve a track it never advertised.
        assert_eq!(
            select_track("9.m4s", &listed).map(|t| t.name).as_deref(),
            Some("9.m4s")
        );
        assert!(select_track("", &[]).is_none());
    }
}

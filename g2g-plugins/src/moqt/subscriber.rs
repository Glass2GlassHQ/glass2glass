//! The subscribing half of MoQ Transport, shared by the single-track
//! [`MoqtSrc`](crate::moqtsrc::MoqtSrc) and the multi-track
//! [`MoqtSessionSrc`](crate::moqtsessionsrc::MoqtSessionSrc): the session
//! driver of each draft, the per-track subscription state, and the reordering
//! that turns objects back into an fMP4 byte stream.
//!
//! One session carries any number of tracks, so everything here is keyed by
//! subscription rather than by element: the two elements differ only in how
//! many tracks they ask for and where the payloads go.
//!
//! A track is established either by our SUBSCRIBE being answered or by the
//! publisher's PUBLISH being accepted (§9.13), so a subscription is matched to
//! its track by name and carries whichever request id established it.

use core::time::Duration;

use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::LogSource;
use g2g_core::memory::SystemSlice;
use g2g_core::{g2g_debug, FrameTiming, G2gError, HardwareError, MemoryDomain};

use web_transport_quinn::SendStream;

use super::catalog::{self, CatalogTrack};
use super::coding::{param, subscription_filter_largest_object, Params, TrackName, TrackNamespace};
use super::message::{request_error_code, ControlMessage, FetchType, JoiningFetch};
use super::reassembly::Reassembler;
use super::session::{implementation_name, DataEvent, MoqtSession};
use super::v18;
use super::{negotiated_version, parse_versions, MoqtVersion};

/// What a subscribing element hands the session driver: everything both
/// elements expose as properties and the driver needs to dial and subscribe.
#[derive(Debug, Clone)]
pub struct SubscriberConfig {
    pub location: String,
    pub cert_hashes: String,
    pub namespace: String,
    pub init_track: String,
    pub catalog_track: String,
    pub use_catalog: bool,
    pub max_request_id: u64,
    pub versions: String,
    pub max_groups: u64,
    pub max_buffer_bytes: u64,
    pub max_object_bytes: u64,
    pub catchup_groups: u64,
    pub timeout_ms: u64,
    /// Tracks a publisher-initiated PUBLISH may establish, beyond the init and
    /// catalog tracks. Empty means any track in the namespace will do.
    pub wanted_tracks: Vec<String>,
}

impl Default for SubscriberConfig {
    fn default() -> Self {
        Self {
            location: String::from("https://127.0.0.1:4443/"),
            cert_hashes: String::new(),
            namespace: String::from("g2g"),
            init_track: String::from("0.mp4"),
            catalog_track: String::from(".catalog"),
            use_catalog: true,
            max_request_id: 100,
            versions: String::from("18,16"),
            max_groups: 8,
            max_buffer_bytes: 32 * 1024 * 1024,
            max_object_bytes: 16 * 1024 * 1024,
            catchup_groups: 0,
            timeout_ms: 15_000,
            wanted_tracks: Vec::new(),
        }
    }
}

/// Dial the relay and complete the handshake of whichever draft the server
/// selected, leaving a driver ready to subscribe.
pub async fn connect(cfg: &SubscriberConfig) -> Result<AnyDriver, G2gError> {
    let offered = parse_versions(&cfg.versions)?;
    let protocols: Vec<&str> = offered.iter().map(|v| v.protocol()).collect();
    let session =
        crate::remotewtio::dial(&cfg.location, &cfg.cert_hashes, &protocols, "default").await?;
    let namespace = TrackNamespace::from_path(&cfg.namespace);
    // A PUBLISH the publisher initiates establishes one of these tracks; with no
    // track named, any track in the namespace will do.
    let mut wanted = Vec::from([cfg.init_track.clone(), cfg.catalog_track.clone()]);
    wanted.extend(cfg.wanted_tracks.iter().cloned());
    let state = SubsState::new(
        cfg.max_groups as usize,
        cfg.max_buffer_bytes as usize,
        namespace.clone(),
        wanted,
        cfg.wanted_tracks.is_empty(),
    );
    Ok(match negotiated_version(&session, &offered)? {
        MoqtVersion::V16 => {
            let session =
                MoqtSession::connect_over(session, cfg.max_request_id, &implementation_name())
                    .await?;
            let data = session.start_data_reader(cfg.max_object_bytes as usize);
            AnyDriver::V16(Driver {
                namespace,
                timeout_ms: cfg.timeout_ms,
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
                cfg.max_object_bytes as usize,
            )
            .await?;
            let data = session.take_data().ok_or_else(session_err)?;
            let requests = session.take_requests().ok_or_else(session_err)?;
            let (response_tx, responses) = tokio::sync::mpsc::unbounded_channel();
            AnyDriver::V18(Driver18 {
                namespace,
                timeout_ms: cfg.timeout_ms,
                session,
                data,
                responses,
                response_tx,
                requests,
                state,
                closed: false,
            })
        }
    })
}

/// Tracks a publisher may establish here with PUBLISH. A publisher cannot make
/// the subscriber hold state for more tracks than this.
const MAX_PUBLISHED_TRACKS: usize = 16;

/// A relay we cannot talk to, or one that violates the protocol, is the same
/// thing to the pipeline: no stream.
pub fn session_err() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// One track we asked the relay for.
#[derive(Debug)]
pub struct Subscription {
    pub request_id: u64,
    /// The track this subscription carries. A publisher-initiated PUBLISH is
    /// matched to it by name, so one track has one subscription however it was
    /// established.
    pub name: String,
    /// Set by SUBSCRIBE_OK; the alias is how data streams name this track.
    pub track_alias: Option<u64>,
    /// The draft-18 request stream's send half. Draft-18 has no UNSUBSCRIBE:
    /// resetting this is how the subscription is cancelled at shutdown.
    pub request_tx: Option<SendStream>,
    /// The request id of the catch-up FETCH joined to this subscription, and
    /// the send half of its draft-18 request stream (dropping that half resets
    /// the stream, which cancels the fetch).
    pub fetch_request: Option<u64>,
    pub fetch_tx: Option<SendStream>,
    /// Whether the catch-up FETCH is still delivering. Live objects wait behind
    /// it, because everything it carries is older.
    pub catchup_pending: bool,
    /// Fetched payloads in the order the publisher wrote them.
    pub fetch_ready: VecDeque<Vec<u8>>,
    /// Objects the catch-up FETCH delivered.
    pub fetch_objects: u64,
    pub reassembler: Reassembler,
    /// Payloads in order, waiting to be emitted.
    pub ready: VecDeque<Vec<u8>>,
    /// Data streams that have closed, against PUBLISH_DONE's stream count.
    pub streams_closed: u64,
    /// PUBLISH_DONE's stream count, when it arrived before all of those
    /// streams had: the message races the data plane, so the subscription
    /// drains the streams it was told about before it ends.
    pub done_after: Option<u64>,
    /// PUBLISH_DONE resolved, or the relay refused the subscription.
    pub ended: bool,
}

impl Subscription {
    pub fn new(request_id: u64, name: String, max_groups: usize, max_bytes: usize) -> Self {
        Self {
            request_id,
            name,
            track_alias: None,
            request_tx: None,
            fetch_request: None,
            fetch_tx: None,
            catchup_pending: false,
            fetch_ready: VecDeque::new(),
            fetch_objects: 0,
            reassembler: Reassembler::new(max_groups, max_bytes),
            ready: VecDeque::new(),
            streams_closed: 0,
            done_after: None,
            ended: false,
        }
    }

    /// Flush what the reassembler still holds and end the subscription.
    pub fn finish(&mut self) {
        let tail = self.reassembler.flush();
        self.ready.extend(tail);
        self.ended = true;
    }

    /// The next payload to emit. Everything the catch-up FETCH delivered comes
    /// out first, and the live objects wait behind it until it has finished.
    pub fn next_payload(&mut self) -> Option<Vec<u8>> {
        if let Some(payload) = self.fetch_ready.pop_front() {
            return Some(payload);
        }
        if self.catchup_pending {
            return None;
        }
        self.ready.pop_front()
    }

    /// Whether this subscription has nothing more to deliver.
    pub fn drained(&self) -> bool {
        self.ended && !self.catchup_pending && self.fetch_ready.is_empty()
    }

    /// PUBLISH_DONE promised `stream_count` data streams: end now if they all
    /// closed already, otherwise once the last one does.
    pub fn publish_done(&mut self, stream_count: u64) {
        if self.streams_closed >= stream_count {
            self.finish();
        } else {
            self.done_after = Some(stream_count);
        }
    }
}

/// The subscriptions and the reorder state they share, identical whichever
/// draft delivers the events: ordering is not version-specific.
#[derive(Debug)]
pub struct SubsState {
    pub max_groups: usize,
    pub max_bytes: usize,
    /// The namespace this run subscribes to: a PUBLISH for another one is not
    /// ours to accept.
    pub namespace: TrackNamespace,
    /// Track names an incoming PUBLISH may establish, and whether any name in
    /// the namespace is acceptable (no `track-name` was set).
    pub wanted: Vec<String>,
    pub accept_any: bool,
    pub subs: Vec<Subscription>,
    /// Data events whose track alias no subscription claims yet. The control
    /// plane and the data streams are different QUIC streams, so a subgroup
    /// can arrive before the SUBSCRIBE_OK that names its alias.
    pub orphans: VecDeque<DataEvent>,
    pub orphan_bytes: usize,
}

impl SubsState {
    pub fn new(
        max_groups: usize,
        max_bytes: usize,
        namespace: TrackNamespace,
        wanted: Vec<String>,
        accept_any: bool,
    ) -> Self {
        Self {
            max_groups,
            max_bytes,
            namespace,
            wanted,
            accept_any,
            subs: Vec::new(),
            orphans: VecDeque::new(),
            orphan_bytes: 0,
        }
    }

    pub fn add(&mut self, request_id: u64, name: &str) -> usize {
        self.subs.push(Subscription::new(
            request_id,
            String::from(name),
            self.max_groups,
            self.max_bytes,
        ));
        self.subs.len() - 1
    }

    pub fn by_name(&self, name: &str) -> Option<usize> {
        self.subs.iter().position(|s| s.name == name)
    }

    /// Whether an incoming PUBLISH establishes a subscription here: it has to
    /// be our namespace, and either a track we already asked for or, when no
    /// track was named, any track the publisher offers.
    pub fn accepts_publish(&self, namespace: &TrackNamespace, name: &str) -> bool {
        if *namespace != self.namespace || self.subs.len() >= MAX_PUBLISHED_TRACKS {
            return false;
        }
        self.by_name(name).is_some() || self.wanted.iter().any(|w| w == name) || self.accept_any
    }

    /// Attach an accepted PUBLISH to the subscription for its track, creating
    /// one when nothing has asked for that track yet, and return its index.
    pub fn establish_published(&mut self, request_id: u64, name: &str, track_alias: u64) -> usize {
        let at = match self.by_name(name) {
            Some(at) => at,
            None => self.add(request_id, name),
        };
        // The publisher's request id identifies the subscription from here on:
        // PUBLISH_DONE and UNSUBSCRIBE both name it.
        self.subs[at].request_id = request_id;
        self.subs[at].track_alias = Some(track_alias);
        self.claim_orphans(track_alias);
        at
    }

    pub fn by_request(&mut self, id: u64) -> Option<&mut Subscription> {
        self.subs.iter_mut().find(|s| s.request_id == id)
    }

    /// The subscription whose catch-up FETCH carries request id `id`.
    pub fn by_fetch(&mut self, id: u64) -> Option<&mut Subscription> {
        self.subs.iter_mut().find(|s| s.fetch_request == Some(id))
    }

    pub fn handle_data(&mut self, event: DataEvent) {
        // A fetch response names its request rather than a track, and arrives
        // in order, so it goes straight to the subscription that asked for it.
        match event {
            DataEvent::FetchObject { request_id, object } => {
                if let Some(sub) = self.by_fetch(request_id) {
                    // A zero-length object is a status marker, not media.
                    if !object.payload.is_empty() {
                        sub.fetch_objects = sub.fetch_objects.saturating_add(1);
                        sub.fetch_ready.push_back(object.payload);
                    }
                }
                return;
            }
            DataEvent::FetchClosed { request_id } => {
                if let Some(sub) = self.by_fetch(request_id) {
                    sub.catchup_pending = false;
                }
                return;
            }
            _ => {}
        }
        let alias = match &event {
            DataEvent::StreamOpened { track_alias, .. }
            | DataEvent::StreamClosed { track_alias, .. }
            | DataEvent::Object { track_alias, .. } => *track_alias,
            // Handled above.
            DataEvent::FetchObject { .. } | DataEvent::FetchClosed { .. } => return,
        };
        let Some(at) = self.subs.iter().position(|s| s.track_alias == Some(alias)) else {
            self.hold_orphan(event);
            return;
        };
        self.apply(at, event);
    }

    pub fn apply(&mut self, at: usize, event: DataEvent) {
        let sub = &mut self.subs[at];
        match event {
            DataEvent::StreamOpened { group_id, .. } => sub.reassembler.stream_opened(group_id),
            DataEvent::StreamClosed { group_id, .. } => {
                sub.reassembler.stream_closed(group_id);
                sub.streams_closed = sub.streams_closed.saturating_add(1);
            }
            DataEvent::Object { object, .. } => sub.reassembler.push(object),
            // Routed by request id before this point.
            DataEvent::FetchObject { .. } | DataEvent::FetchClosed { .. } => {}
        }
        sub.ready.extend(sub.reassembler.drain());
        if sub.done_after.is_some_and(|n| sub.streams_closed >= n) {
            sub.finish();
        }
    }

    /// Hold a data event for an alias no subscription has yet, bounded by the
    /// same byte budget the reassembler uses.
    pub fn hold_orphan(&mut self, event: DataEvent) {
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
    pub fn claim_orphans(&mut self, alias: u64) {
        let Some(at) = self.subs.iter().position(|s| s.track_alias == Some(alias)) else {
            return;
        };
        let held = core::mem::take(&mut self.orphans);
        for event in held {
            let matches = match &event {
                DataEvent::StreamOpened { track_alias, .. }
                | DataEvent::StreamClosed { track_alias, .. }
                | DataEvent::Object { track_alias, .. } => *track_alias == alias,
                // A fetch response is never held: it is routed by request id.
                DataEvent::FetchObject { .. } | DataEvent::FetchClosed { .. } => false,
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
#[derive(Debug)]
pub struct Driver {
    namespace: TrackNamespace,
    timeout_ms: u64,
    session: MoqtSession,
    data: tokio::sync::mpsc::UnboundedReceiver<DataEvent>,
    state: SubsState,
    /// The session ended (control stream closed, or the relay went away).
    closed: bool,
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
pub enum Pumped {
    Applied,
    /// The session ended: the control stream closed or the relay went away.
    Ended,
    /// Nothing arrived within `timeout`.
    TimedOut,
}

impl Driver {
    /// Send SUBSCRIBE for `name` and return its index in `subs`. `catchup` asks
    /// for the Largest Object filter, which draft-16 §9.16.2 requires of any
    /// subscription a joining FETCH names.
    pub async fn subscribe(&mut self, name: &str, catchup: bool) -> Result<usize, G2gError> {
        let id = self.session.allocate_request_id().ok_or_else(session_err)?;
        let mut params = Params::new();
        if catchup {
            params.set_bytes(
                param::SUBSCRIPTION_FILTER,
                subscription_filter_largest_object(),
            );
        }
        self.session
            .send(&ControlMessage::Subscribe {
                id,
                namespace: self.namespace.clone(),
                track_name: TrackName::new(name),
                params,
            })
            .await?;
        g2g_debug!(self, "SUBSCRIBE {name} as request {id}");
        Ok(self.state.add(id, name))
    }

    /// Ask for `groups` groups before the subscription's live edge with a
    /// relative joining FETCH, so the two are contiguous.
    pub async fn fetch_joining(&mut self, at: usize, groups: u64) -> Result<(), G2gError> {
        let id = self.session.allocate_request_id().ok_or_else(session_err)?;
        let joining_request_id = self.state.subs[at].request_id;
        self.session
            .send(&ControlMessage::Fetch {
                id,
                fetch_type: FetchType::RelativeJoining,
                standalone: None,
                joining: Some(JoiningFetch {
                    joining_request_id,
                    joining_start: groups,
                }),
                params: Params::new(),
            })
            .await?;
        g2g_debug!(self, "FETCH {groups} groups back as request {id}");
        self.state.subs[at].fetch_request = Some(id);
        self.state.subs[at].catchup_pending = true;
        Ok(())
    }

    /// Wait for one event and apply it.
    pub async fn pump(&mut self) -> Result<Pumped, G2gError> {
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

    pub async fn handle_control(&mut self, msg: ControlMessage) -> Result<(), G2gError> {
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
                // A refused catch-up costs the buffer, not the stream: the live
                // objects behind it are released.
                if let Some(sub) = self.state.by_fetch(id) {
                    sub.catchup_pending = false;
                }
            }
            ControlMessage::PublishDone {
                id, stream_count, ..
            } => {
                if let Some(sub) = self.state.by_request(id) {
                    sub.publish_done(stream_count);
                }
            }
            ControlMessage::MaxRequestId { request_id } => {
                self.session.set_peer_max_request_id(request_id);
            }
            ControlMessage::GoAway { .. } => self.closed = true,
            // A publisher that initiates the subscription itself: accepting it
            // with PUBLISH_OK is the other way a track is established (§9.13).
            ControlMessage::Publish {
                id,
                namespace,
                track_name,
                track_alias,
                ..
            } => {
                let name = track_name.as_str_lossy();
                if self.state.accepts_publish(&namespace, &name) {
                    g2g_debug!(self, "PUBLISH {name} accepted as request {id}");
                    self.state.establish_published(id, &name, track_alias);
                    self.session
                        .send(&ControlMessage::PublishOk {
                            id,
                            params: Params::new(),
                        })
                        .await?;
                } else {
                    self.session
                        .send(&ControlMessage::RequestError {
                            id,
                            error_code: request_error_code::UNINTERESTED,
                            retry_interval: 0,
                            reason: String::from("not this track"),
                        })
                        .await?;
                }
            }
            // A publisher-side request we do not serve. Draft-16 §4 asks for an
            // explicit refusal rather than silence.
            ControlMessage::RequestUpdate { id, .. } => {
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
    pub async fn shutdown(&mut self) {
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
#[derive(Debug)]
pub enum Step {
    Control(Option<ControlMessage>),
    Data(DataEvent),
}

/// One response-stream message, keyed by its request id; `None` when the stream
/// ended, which terminates the request (§11.4.1).
type Response = (u64, Option<v18::message::ControlMessage>);

/// The live half of a draft-18 run. Each SUBSCRIBE opened its own bidirectional
/// stream, and a task per stream forwards its responses into one channel, so
/// the pump still waits in one place.
#[derive(Debug)]
pub struct Driver18 {
    namespace: TrackNamespace,
    timeout_ms: u64,
    session: v18::session::Session18,
    data: tokio::sync::mpsc::UnboundedReceiver<DataEvent>,
    responses: tokio::sync::mpsc::UnboundedReceiver<Response>,
    response_tx: tokio::sync::mpsc::UnboundedSender<Response>,
    /// Request streams the publisher opens: a PUBLISH here is the other way a
    /// subscription is established.
    requests: tokio::sync::mpsc::UnboundedReceiver<v18::session::PeerRequest>,
    state: SubsState,
    closed: bool,
}

impl Driver18 {
    /// Open a request stream with SUBSCRIBE for `name` and return its index.
    /// `catchup` asks for the Largest Object filter, which is what makes the
    /// subscription joinable by a FETCH (§10.12.2).
    pub async fn subscribe(&mut self, name: &str, catchup: bool) -> Result<usize, G2gError> {
        let id = self.session.allocate_request_id();
        let mut params = v18::coding::MessageParams::new();
        if catchup {
            let filter = v18::message::SubscriptionFilter::LargestObject
                .to_bytes()
                .map_err(|_| session_err())?;
            params
                .set(
                    v18::coding::param::SUBSCRIPTION_FILTER,
                    v18::coding::MsgParam::Bytes(filter),
                )
                .map_err(|_| session_err())?;
        }
        let (tx, rx) = self
            .session
            .open_request(&v18::message::ControlMessage::Subscribe {
                id,
                namespace: self.namespace.clone(),
                track_name: TrackName::new(name),
                params,
            })
            .await?;
        g2g_debug!(self, "SUBSCRIBE {name} as request {id}");
        tokio::spawn(forward_responses(id, rx, self.response_tx.clone()));
        let at = self.state.add(id, name);
        self.state.subs[at].request_tx = Some(tx);
        Ok(at)
    }

    /// Ask for `groups` groups before the subscription's live edge with a
    /// relative joining FETCH on its own request stream.
    pub async fn fetch_joining(&mut self, at: usize, groups: u64) -> Result<(), G2gError> {
        let id = self.session.allocate_request_id();
        let joining_request_id = self.state.subs[at].request_id;
        let (tx, rx) = self
            .session
            .open_request(&v18::message::ControlMessage::Fetch {
                id,
                fetch_type: v18::message::FetchType::RelativeJoining,
                standalone: None,
                joining: Some(v18::message::JoiningFetch {
                    joining_request_id,
                    joining_start: groups,
                }),
                params: v18::coding::MessageParams::new(),
            })
            .await?;
        g2g_debug!(self, "FETCH {groups} groups back as request {id}");
        tokio::spawn(forward_responses(id, rx, self.response_tx.clone()));
        let sub = &mut self.state.subs[at];
        sub.fetch_request = Some(id);
        // Holding the send half open keeps the fetch live: dropping it resets
        // the stream, which is how draft-18 cancels one.
        sub.fetch_tx = Some(tx);
        sub.catchup_pending = true;
        Ok(())
    }

    /// Wait for one event and apply it.
    pub async fn pump(&mut self) -> Result<Pumped, G2gError> {
        if self.closed || self.session.is_closed() {
            self.closed = true;
            return Ok(Pumped::Ended);
        }
        let timeout = self.timeout_ms;
        let step = {
            let responses = &mut self.responses;
            let data = &mut self.data;
            let requests = &mut self.requests;
            let next = async move {
                tokio::select! {
                    response = responses.recv() => response.map(Step18::Response),
                    event = data.recv() => event.map(Step18::Data),
                    request = requests.recv() => request.map(Step18::Request),
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
            Some(Step18::Request(request)) => self.handle_peer_request(request).await?,
            // Every channel ended: the session is gone.
            None => {
                self.closed = true;
                return Ok(Pumped::Ended);
            }
        }
        Ok(Pumped::Applied)
    }

    /// A request stream the publisher opened. PUBLISH for a track this run wants
    /// establishes the subscription (§10.5); anything else is refused on the
    /// stream it arrived on.
    pub async fn handle_peer_request(
        &mut self,
        request: v18::session::PeerRequest,
    ) -> Result<(), G2gError> {
        use v18::message::ControlMessage as Msg;
        let v18::session::PeerRequest { first, mut tx, rx } = request;
        let publish = match first {
            Msg::Publish {
                id,
                namespace,
                track_name,
                track_alias,
                ..
            } => {
                let name = track_name.as_str_lossy();
                self.state
                    .accepts_publish(&namespace, &name)
                    .then_some((id, name, track_alias))
            }
            _ => None,
        };
        let Some((id, name, track_alias)) = publish else {
            let msg = Msg::RequestError {
                error_code: v18::message::request_error_code::UNINTERESTED,
                retry_interval: 0,
                reason: String::from("not this track"),
                redirect: None,
            };
            v18::session::write_message(&mut tx, &msg).await?;
            let _ = tx.finish();
            return Ok(());
        };
        g2g_debug!(self, "PUBLISH {name} accepted as request {id}");
        v18::session::write_message(
            &mut tx,
            &Msg::PublishOk {
                params: v18::coding::MessageParams::new(),
                properties: Params::new(),
            },
        )
        .await?;
        let at = self.state.establish_published(id, &name, track_alias);
        // PUBLISH_DONE arrives on this stream, and resetting our half is how the
        // subscription is cancelled at shutdown.
        self.state.subs[at].request_tx = Some(tx);
        tokio::spawn(forward_responses(id, rx, self.response_tx.clone()));
        Ok(())
    }

    pub fn handle_response(&mut self, id: u64, msg: Option<v18::message::ControlMessage>) {
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
                // A refused catch-up costs the buffer, not the stream.
                if let Some(sub) = self.state.by_fetch(id) {
                    sub.catchup_pending = false;
                }
            }
            // The publisher accepted the catch-up; its objects arrive on the
            // response stream the data plane routes by request id.
            Some(Msg::FetchOk { end, .. }) => {
                g2g_debug!(self, "request {id}: FETCH_OK ending at {end:?}");
            }
            Some(Msg::PublishDone { stream_count, .. }) => {
                if let Some(sub) = self.state.by_request(id) {
                    sub.publish_done(stream_count);
                }
            }
            // The request stream ended. After PUBLISH_DONE that is normal
            // cleanup and the stream-count drain keeps running; without one it
            // is a cancellation, and what the reassembler holds is the tail.
            None => {
                if let Some(sub) = self.state.by_request(id) {
                    if sub.done_after.is_none() && !sub.ended {
                        sub.finish();
                    }
                }
            }
            // Anything else on a response stream is a message this subscriber
            // did not ask for: decoded, then ignored.
            Some(_) => {}
        }
    }

    /// Cancel every live subscription by resetting its request stream (§3.3.2)
    /// and close the session.
    pub async fn shutdown(&mut self) {
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
#[derive(Debug)]
pub enum Step18 {
    Response(Response),
    Data(DataEvent),
    Request(v18::session::PeerRequest),
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
#[derive(Debug)]
pub enum AnyDriver {
    V16(Driver),
    V18(Driver18),
}

impl AnyDriver {
    pub fn state(&mut self) -> &mut SubsState {
        match self {
            Self::V16(driver) => &mut driver.state,
            Self::V18(driver) => &mut driver.state,
        }
    }

    pub async fn subscribe(&mut self, name: &str, catchup: bool) -> Result<usize, G2gError> {
        // A publisher that initiated this track with PUBLISH has established it
        // already, so there is nothing to ask for.
        if let Some(at) = self.state().by_name(name) {
            if self.state().subs[at].track_alias.is_some() {
                return Ok(at);
            }
        }
        match self {
            Self::V16(driver) => driver.subscribe(name, catchup).await,
            Self::V18(driver) => driver.subscribe(name, catchup).await,
        }
    }

    pub async fn fetch_joining(&mut self, at: usize, groups: u64) -> Result<(), G2gError> {
        match self {
            Self::V16(driver) => driver.fetch_joining(at, groups).await,
            Self::V18(driver) => driver.fetch_joining(at, groups).await,
        }
    }

    /// Pump until the subscription at `at` is established: a joining FETCH names
    /// it, and the publisher can only resolve the name once it exists. `false`
    /// when the subscription ended, or nothing arrived, before that.
    pub async fn established(&mut self, at: usize) -> Result<bool, G2gError> {
        loop {
            if self.state().subs[at].track_alias.is_some() {
                return Ok(true);
            }
            if self.state().subs[at].ended || self.pump().await? != Pumped::Applied {
                return Ok(false);
            }
        }
    }

    pub async fn pump(&mut self) -> Result<Pumped, G2gError> {
        match self {
            Self::V16(driver) => driver.pump().await,
            Self::V18(driver) => driver.pump().await,
        }
    }

    pub async fn shutdown(&mut self) {
        match self {
            Self::V16(driver) => driver.shutdown().await,
            Self::V18(driver) => driver.shutdown().await,
        }
    }

    /// Pump until the subscription at `at` has a payload, or until it ends, the
    /// session ends, or nothing arrives within `timeout`.
    pub async fn first_object(&mut self, at: usize) -> Result<Option<Vec<u8>>, G2gError> {
        loop {
            if let Some(payload) = self.state().subs[at].next_payload() {
                return Ok(Some(payload));
            }
            if self.state().subs[at].drained() || self.pump().await? != Pumped::Applied {
                return Ok(self.state().subs[at].next_payload());
            }
        }
    }
}

/// Pick the media track: the `track-name` property when set, else the catalog's
/// first entry, else the reference default for a single-track broadcast.
pub fn select_track(wanted: &str, tracks: &[CatalogTrack]) -> Option<CatalogTrack> {
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

pub fn byte_frame(bytes: Vec<u8>, sequence: u64) -> Frame {
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

impl AnyDriver {
    /// The catalog's track list, or nothing when this run does not read it.
    /// Without the catalog the defaults are the reference layout, which is what
    /// `moq-sub` falls back to as well.
    pub async fn read_catalog(
        &mut self,
        cfg: &SubscriberConfig,
    ) -> Result<Vec<CatalogTrack>, G2gError> {
        if !cfg.use_catalog {
            return Ok(Vec::new());
        }
        let at = self.subscribe(&cfg.catalog_track, false).await?;
        let bytes = self.first_object(at).await?.unwrap_or_default();
        Ok(catalog::parse(&bytes))
    }

    /// The init segment: the one object of the init track, which every media
    /// track of the broadcast shares.
    pub async fn read_init(&mut self, track: &str) -> Result<Option<Vec<u8>>, G2gError> {
        let at = self.subscribe(track, false).await?;
        self.first_object(at).await
    }

    /// Subscribe to a media track, asking for the catch-up groups when the
    /// element wants them. A joining FETCH names the subscription, so it can
    /// only be sent once the publisher has established it.
    pub async fn subscribe_media(
        &mut self,
        name: &str,
        catchup_groups: u64,
    ) -> Result<usize, G2gError> {
        let at = self.subscribe(name, catchup_groups > 0).await?;
        if catchup_groups > 0 && self.established(at).await? {
            self.fetch_joining(at, catchup_groups).await?;
        }
        Ok(at)
    }
}

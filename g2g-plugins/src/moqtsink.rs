//! MoQ Transport publisher sink (`moqtsink`, M902, `moqt` feature): publishes a
//! fragmented-MP4 byte stream to an IETF MoQT relay over WebTransport.
//!
//! ```text
//! ... ! x264enc ! mp4mux ! moqtsink location=https://relay:4443/ namespace=live/cam
//! ```
//!
//! The muxer stays a separate element, so this sink packages whatever writes
//! ISO-BMFF (`mp4mux`, the A/V `mp4muxn`) and never carries a second
//! fragmenter. It walks the incoming boxes the way
//! [`HlsSink`](crate::hlssink) does and maps them onto MOQT's object model:
//!
//! - `ftyp`+`moov` is the init segment: one object, group 0, on its own track
//!   (`0.mp4` by default), which is what `moq-sub` fetches first.
//! - each `moof`+`mdat` pair (with the `styp` / `prft` that open its segment) is
//!   one object on the media track `{track_id}.m4s`. CMAF requires an object to
//!   hold at least one whole chunk, which a `moof`+`mdat` pair is.
//! - a fragment whose first sample is a sync sample starts a new **group**, so a
//!   group is a GOP and a subscriber that joins mid-stream starts at the next
//!   keyframe. A group rides one subgroup stream by default, or `subgroups`
//!   of them round-robin, which stops one object's loss from holding up the
//!   next.
//! - a `.catalog` track carries the JSON track list a browser player reads
//!   (`moq-sub --catalog` reads the same document).
//!
//! `datagrams=true` carries each media object in a QUIC datagram instead:
//! unreliable and bounded by the path MTU, with no head-of-line blocking. An
//! object the path will not take (larger than the MTU, or a peer that accepts no
//! datagrams) falls back to a subgroup stream, since dropping it is not the
//! publisher's call to make. The init and catalog tracks always ride streams:
//! losing either loses the whole broadcast. A group carried by datagrams is
//! closed by an end-of-group status datagram, because no stream ends to say so.
//!
//! The last `cache-groups` groups of every track are kept in memory, so a
//! subscriber can FETCH a range it has already missed (standalone, or joined to
//! one of its subscriptions); a range that was never published or has fallen out
//! of the cache is refused with INVALID_RANGE rather than left hanging.
//! `publish=true` turns the offer round: each track is announced with PUBLISH as
//! soon as the `moov` names it, and a subscriber that accepts one gets the same
//! subscription it would have got by asking.
//!
//! Every group's stream is opened only after SUBSCRIBE_OK for that
//! subscription, so the subscriber can resolve the track alias in the stream
//! header. The session is dialled when the pipeline is configured, and inbound
//! control messages are answered by a pump task on the session's control stream,
//! so a SUBSCRIBE is served whether or not a frame is flowing: before the
//! encoder's first fragment as well as between two of them. The pump and frame
//! publishing take turns on the [`Core`] lock, so an object is never interleaved
//! with the control message that changed what is being served. A media SUBSCRIBE
//! that arrives before the `moov` names any track is held until one does.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::sync::Mutex as StdMutex;

use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use web_transport_quinn::SendStream;

use g2g_core::{
    g2g_debug, AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, HardwareError, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use g2g_core::log::LogSource;

use crate::fmp4::trun_first_sample_is_sync;
use crate::moqt::catalog;
use crate::moqt::coding::{MoqtError, Params, TrackName, TrackNamespace};
use crate::moqt::data::{StreamHeaderType, SubgroupHeader, SubgroupObjectHeader};
use crate::moqt::datagram::DatagramObject;
use crate::moqt::fetch::FetchWriter;
use crate::moqt::message::{
    publish_done_code, request_error_code, stream_error_code, ControlMessage, FetchType,
    JoiningFetch, Location, StandaloneFetch,
};
use crate::moqt::session::{implementation_name, MoqtSession};
use crate::moqt::v18;
use crate::moqt::{negotiated_version, parse_versions, MoqtVersion};
use crate::mp4box::{be32, boxes, find_box, find_path};
use crate::remotewtio::wt_err;

/// The reference publisher writes every subgroup with an explicit subgroup id
/// and an extension-header block (`session/subscribed.rs`), so a relay sees the
/// same header type from us as from `moq-pub`.
const HEADER_TYPE: StreamHeaderType = StreamHeaderType::SubgroupIdExt;

/// SUBSCRIBEs held for a track the `moov` has not named yet. A peer cannot make
/// the queue grow without bound by subscribing to names that do not exist.
const MAX_PENDING_SUBSCRIBES: usize = 64;

/// FETCH responses written at once. A peer cannot make the publisher hold more
/// response streams (and the objects queued on them) than this.
const MAX_ACTIVE_FETCHES: usize = 16;

/// A malformed message we built, or a peer message we could not decode: either
/// way the session is unusable.
fn proto_err(_: MoqtError) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// What an accepted subscription is serving.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Target {
    /// The `.catalog` JSON, one object.
    Catalog,
    /// The `ftyp`+`moov` init segment, one object.
    Init,
    /// A media track, identified by its MP4 track id.
    Media(u32),
}

/// One open subgroup stream of the group in progress.
#[derive(Debug)]
struct SubgroupStream {
    subgroup_id: u64,
    stream: SendStream,
    /// Last object id written here: the next object's delta is measured from it.
    prev_object_id: Option<u64>,
}

/// One accepted SUBSCRIBE and the stream state serving it.
#[derive(Debug)]
struct Subscription {
    request_id: u64,
    track_alias: u64,
    target: Target,
    /// The draft-18 request stream's send half: SUBSCRIBE_OK went on it, and
    /// PUBLISH_DONE follows there. `None` on a draft-16 session, whose
    /// responses ride the control stream instead.
    reply: Option<SendStream>,
    /// The subgroup streams open for the group in progress.
    streams: Vec<SubgroupStream>,
    /// Whether a group boundary has passed since this subscription was accepted.
    /// Until one has, the subscriber joined mid-group and gets nothing.
    serving_group: bool,
    /// The largest location published when this subscription was accepted: what
    /// a joining FETCH against it is contiguous with (draft-16 §9.16.2.1,
    /// draft-18 §10.12.2.1).
    joining: Option<Location>,
    /// Whether the one-object tracks have already delivered their object.
    delivered: bool,
    /// Data streams opened for this subscription, reported in PUBLISH_DONE.
    streams_opened: u64,
}

/// A media track discovered from the `moov`.
#[derive(Debug, Clone)]
struct MediaTrack {
    track_id: u32,
    name: String,
    /// Group counter: bumped at each sync fragment.
    group_id: u64,
    /// Objects already written into the open group.
    objects_in_group: u64,
    /// Whether a group has been opened at all.
    started: bool,
    /// Catalog `selectionParams` fragment, empty when the codec is unrecognized.
    selection_params: String,
    /// The most recently published groups, oldest first, so a FETCH can be
    /// served from memory. Bounded by `cache-groups`.
    cache: VecDeque<CachedGroup>,
}

/// One published group held for FETCH. Object ids run from 0 with no gaps, so
/// the index in `objects` is the object id.
#[derive(Debug, Clone)]
struct CachedGroup {
    group_id: u64,
    objects: Vec<Vec<u8>>,
}

impl MediaTrack {
    /// The largest location published on this track, or `None` before the first
    /// object.
    fn largest(&self) -> Option<Location> {
        self.started.then(|| Location {
            group_id: self.group_id,
            object_id: self.objects_in_group.saturating_sub(1),
        })
    }

    /// Buffer one published object, dropping the oldest group past the bound.
    fn cache_object(&mut self, group_id: u64, starts_group: bool, payload: &[u8], depth: usize) {
        if depth == 0 {
            self.cache.clear();
            return;
        }
        if starts_group || self.cache.back().is_none_or(|g| g.group_id != group_id) {
            self.cache.push_back(CachedGroup {
                group_id,
                objects: Vec::new(),
            });
            while self.cache.len() > depth {
                self.cache.pop_front();
            }
        }
        if let Some(group) = self.cache.back_mut() {
            group.objects.push(payload.to_vec());
        }
    }
}

/// The properties the serving state reads, snapshotted when the session is
/// dialled: from then on the pump task answers control messages without the
/// element.
#[derive(Debug, Clone)]
struct Config {
    namespace: String,
    init_track: String,
    catalog_track: String,
    track_name: String,
    publish_catalog: bool,
    priority: u64,
    datagrams: bool,
    subgroups: u64,
    cache_groups: u64,
    publish: bool,
}

impl Config {
    fn new(namespace: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            init_track: String::from("0.mp4"),
            catalog_track: String::from(".catalog"),
            track_name: String::new(),
            publish_catalog: true,
            priority: 127,
            datagrams: false,
            subgroups: 1,
            cache_groups: 4,
            publish: false,
        }
    }
}

/// What one dial needs: the transport knobs, plus the serving config the session
/// starts with.
#[derive(Debug, Clone)]
struct Dial {
    location: String,
    cert_hashes: String,
    max_request_id: u64,
    versions: String,
    cfg: Config,
}

/// The negotiated session, either draft. What differs is the control plane:
/// draft-16 answers requests on the one control stream, draft-18 on the
/// bidirectional stream each request arrived on.
#[derive(Debug)]
enum Wire {
    V16(MoqtSession),
    V18(v18::session::Session18),
}

impl Wire {
    fn is_closed(&self) -> bool {
        match self {
            Self::V16(session) => session.is_closed(),
            Self::V18(session) => session.is_closed(),
        }
    }
}

/// Counters the sync accessors read. They are bumped from the pump task as well
/// as from `process`, so they live outside the locked core.
#[derive(Debug, Default)]
struct Stats {
    objects_published: AtomicU64,
    datagram_objects: AtomicU64,
    datagram_fallbacks: AtomicU64,
    fetches_served: AtomicU64,
    fetches_cancelled: AtomicU64,
}

impl Stats {
    fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// Where the connect task leaves the control pump's handle, so the element can
/// abort it on drop. A `std` mutex because `Drop` cannot await, and it is only
/// ever held long enough to swap the handle out.
#[derive(Debug, Default)]
struct PumpSlot(StdMutex<Option<JoinHandle<()>>>);

impl PumpSlot {
    fn install(&self, handle: JoinHandle<()>) {
        if let Ok(mut slot) = self.0.lock() {
            if let Some(previous) = slot.replace(handle) {
                previous.abort();
            }
        }
    }

    fn abort(&self) {
        if let Ok(mut slot) = self.0.lock() {
            if let Some(handle) = slot.take() {
                handle.abort();
            }
        }
    }
}

/// How far the session has got. The dial runs in the background from
/// `configure_pipeline` when there is a runtime to spawn on; without one it
/// happens on the first frame instead.
#[derive(Debug)]
enum Connect {
    /// Nothing dialled yet: the next frame does it inline.
    Deferred,
    Dialing(JoinHandle<Result<(), G2gError>>),
    Connected,
}

/// The serving state: the session and everything published over it. Held behind
/// one lock that the control pump and `process` take turns on, so a subscription
/// changes hands between frames and never inside one.
#[derive(Debug)]
struct Core {
    cfg: Config,
    session: Option<Wire>,
    /// Request id of our PUBLISH_NAMESPACE, once sent.
    namespace_request: Option<u64>,
    /// The draft-18 PUBLISH_NAMESPACE request stream's send half. Finishing it
    /// withdraws the namespace, which is what EOS does.
    namespace_stream: Option<SendStream>,
    subscriptions: Vec<Subscription>,
    /// FETCH responses still being written.
    fetches: Vec<ActiveFetch>,
    /// PUBLISHes we sent in `publish` mode, waiting for the subscriber to
    /// accept them.
    pending_publishes: Vec<PendingPublish>,
    /// Whether the tracks have been offered with PUBLISH already.
    published: bool,
    /// Draft-18 PUBLISH response streams the element still has to watch: a task
    /// needs the shared core, which only the element holds.
    publish_watch: Vec<(u64, web_transport_quinn::RecvStream)>,
    /// SUBSCRIBEs for a track the `moov` has not named yet, answered once it
    /// does (or refused then, if it names nothing). The stream is the draft-18
    /// request stream the answer goes on.
    pending_subscribes: Vec<(u64, String, Option<SendStream>)>,

    /// `ftyp`+`moov` as received, the init track's single object.
    init: Vec<u8>,
    /// The catalog JSON, built when the `moov` names the tracks.
    catalog: Vec<u8>,
    tracks: Vec<MediaTrack>,
    /// `styp` / `prft` held until the `moof` they open arrives.
    pending_header: Vec<u8>,
    /// The `moof` (with its segment header) waiting for its `mdat`.
    pending_object: Vec<u8>,
    pending_track: Option<u32>,
    pending_sync: bool,
    /// The pump gave up: a control message it could not answer, or the control
    /// stream ended. The next frame reports it.
    dead: bool,

    stats: Arc<Stats>,
}

/// Publishes an fMP4 byte stream to an IETF MoQ Transport relay.
#[derive(Debug)]
pub struct MoqtSink {
    location: String,
    cert_hashes: String,
    max_request_id: u64,
    versions: String,
    cfg: Config,

    configured: bool,
    core: Arc<Mutex<Core>>,
    stats: Arc<Stats>,
    pump: Arc<PumpSlot>,
    connect: Connect,

    /// The core's track names and catalog as of the last frame, so the sync
    /// accessors can read them without locking.
    track_names: Vec<String>,
    catalog: Vec<u8>,
}

impl Default for MoqtSink {
    fn default() -> Self {
        Self::new("https://127.0.0.1:4443/", "g2g")
    }
}

impl MoqtSink {
    /// Publish to the relay at `location` under `namespace` (a `/`-separated
    /// path, e.g. `live/cam1`).
    pub fn new(location: impl Into<String>, namespace: impl Into<String>) -> Self {
        let cfg = Config::new(namespace);
        let stats = Arc::new(Stats::default());
        Self {
            location: location.into(),
            cert_hashes: String::new(),
            max_request_id: 100,
            versions: String::from("18,16"),
            core: Arc::new(Mutex::new(Core::new(cfg.clone(), Arc::clone(&stats)))),
            cfg,
            configured: false,
            stats,
            pump: Arc::new(PumpSlot::default()),
            connect: Connect::Deferred,
            track_names: Vec::new(),
            catalog: Vec::new(),
        }
    }

    /// Accept only relay certificates whose SHA-256 digest is listed (hex,
    /// comma-separated) instead of requiring a system root, as the M901 carrier
    /// does.
    pub fn with_server_certificate_hashes(mut self, hashes: impl Into<String>) -> Self {
        self.cert_hashes = hashes.into();
        self
    }

    /// Publisher priority written into every subgroup header; smaller is sent
    /// first.
    pub fn with_priority(mut self, priority: u64) -> Self {
        self.cfg.priority = priority;
        self
    }

    /// Carry media objects in datagrams instead of subgroup streams. Off by
    /// default: it trades reliable delivery for no head-of-line blocking.
    pub fn with_datagrams(mut self, datagrams: bool) -> Self {
        self.cfg.datagrams = datagrams;
        self
    }

    /// Spread each group's objects across this many subgroup streams,
    /// round-robin. One (the default) is a stream per group.
    pub fn with_subgroups(mut self, subgroups: u64) -> Self {
        self.cfg.subgroups = subgroups;
        self
    }

    /// Offer every track with PUBLISH once the `moov` names them, instead of
    /// waiting for the peer to SUBSCRIBE.
    pub fn with_publish(mut self, publish: bool) -> Self {
        self.cfg.publish = publish;
        self
    }

    /// Keep this many recently published groups per track, so a subscriber can
    /// FETCH them. Zero caches nothing and refuses every FETCH.
    pub fn with_cache_groups(mut self, groups: u64) -> Self {
        self.cfg.cache_groups = groups;
        self
    }

    /// Objects written to at least one subscriber so far.
    pub fn objects_published(&self) -> u64 {
        self.stats.objects_published.load(Ordering::Relaxed)
    }

    /// Datagrams sent, counted once per subscriber served.
    pub fn datagram_objects(&self) -> u64 {
        self.stats.datagram_objects.load(Ordering::Relaxed)
    }

    /// Objects datagram mode could not send as datagrams (too large for the
    /// path, or a peer that takes none) and put on a subgroup stream instead,
    /// counted the same way.
    pub fn datagram_fallbacks(&self) -> u64 {
        self.stats.datagram_fallbacks.load(Ordering::Relaxed)
    }

    /// FETCH requests accepted and started so far.
    pub fn fetches_served(&self) -> u64 {
        self.stats.fetches_served.load(Ordering::Relaxed)
    }

    /// FETCH responses abandoned part way because the subscriber cancelled.
    pub fn fetches_cancelled(&self) -> u64 {
        self.stats.fetches_cancelled.load(Ordering::Relaxed)
    }

    /// The media track names the `moov` produced, in track order.
    pub fn track_names(&self) -> Vec<String> {
        self.track_names.clone()
    }

    /// The catalog document as published, for tests and for a caller serving it
    /// out of band.
    pub fn catalog(&self) -> &[u8] {
        &self.catalog
    }

    fn dial_params(&self) -> Dial {
        Dial {
            location: self.location.clone(),
            cert_hashes: self.cert_hashes.clone(),
            max_request_id: self.max_request_id,
            versions: self.versions.clone(),
            cfg: self.cfg.clone(),
        }
    }

    /// Make sure the session is up: await the dial started at configure time, or
    /// run it here when there was no runtime then. A dial that failed is
    /// reported and retried on the next frame, since the relay may only be
    /// coming up.
    async fn ready(&mut self) -> Result<(), G2gError> {
        let outcome = match core::mem::replace(&mut self.connect, Connect::Deferred) {
            Connect::Connected => Ok(()),
            Connect::Dialing(handle) => handle
                .await
                .unwrap_or(Err(G2gError::Hardware(HardwareError::Other))),
            Connect::Deferred => {
                connect_session(
                    Arc::clone(&self.core),
                    Arc::clone(&self.pump),
                    self.dial_params(),
                )
                .await
            }
        };
        if outcome.is_ok() {
            self.connect = Connect::Connected;
        }
        outcome
    }
}

impl Drop for MoqtSink {
    fn drop(&mut self) {
        // The pump task holds the core, and through it the session, so without
        // this a dropped element leaves both running.
        self.pump.abort();
        if let Connect::Dialing(handle) = &self.connect {
            handle.abort();
        }
    }
}

/// Dial the relay, complete SETUP, publish the namespace, and start the control
/// pump. Spawned from `configure_pipeline` when there is a runtime, so the
/// namespace is live before the encoder's first fragment.
async fn connect_session(
    core: Arc<Mutex<Core>>,
    pump: Arc<PumpSlot>,
    dial: Dial,
) -> Result<(), G2gError> {
    let mut guard = core.lock().await;
    if guard.session.is_some() {
        return Ok(());
    }
    guard.cfg = dial.cfg;
    let offered = parse_versions(&dial.versions)?;
    let protocols: Vec<&str> = offered.iter().map(|v| v.protocol()).collect();
    let session =
        crate::remotewtio::dial(&dial.location, &dial.cert_hashes, &protocols, "default").await?;
    match negotiated_version(&session, &offered)? {
        MoqtVersion::V16 => {
            let mut session =
                MoqtSession::connect_over(session, dial.max_request_id, &implementation_name())
                    .await?;
            let id = session
                .allocate_request_id()
                .ok_or(G2gError::Hardware(HardwareError::Other))?;
            session
                .send(&ControlMessage::PublishNamespace {
                    id,
                    namespace: guard.namespace_tuple(),
                    params: Params::new(),
                })
                .await?;
            // The pump owns the inbound half from here: nothing else reads it.
            let inbound = session
                .take_control_receiver()
                .ok_or(G2gError::Hardware(HardwareError::Other))?;
            guard.namespace_request = Some(id);
            guard.session = Some(Wire::V16(session));
            drop(guard);
            pump.install(tokio::spawn(pump_control(core, inbound)));
        }
        MoqtVersion::V18 => {
            // The publisher reads no objects, so the data-plane bound only caps
            // what a misbehaving relay could make us buffer.
            let mut session = v18::session::Session18::connect_over(
                session,
                &implementation_name(),
                DATAGRAM_BOUND,
            )
            .await?;
            let id = session.allocate_request_id();
            let (ns_tx, ns_rx) = session
                .open_request(&v18::message::ControlMessage::PublishNamespace {
                    id,
                    namespace: guard.namespace_tuple(),
                    params: v18::coding::MessageParams::new(),
                })
                .await?;
            let requests = session
                .take_requests()
                .ok_or(G2gError::Hardware(HardwareError::Other))?;
            guard.namespace_request = Some(id);
            guard.namespace_stream = Some(ns_tx);
            guard.session = Some(Wire::V18(session));
            drop(guard);
            tokio::spawn(watch_namespace(Arc::clone(&core), ns_rx));
            pump.install(tokio::spawn(pump_requests(core, requests)));
        }
    }
    Ok(())
}

/// The most a draft-18 relay can make the publisher buffer off the data plane
/// it should never use.
const DATAGRAM_BOUND: usize = 64 * 1024;

/// Watch the draft-18 PUBLISH_NAMESPACE response stream: a REQUEST_ERROR (or a
/// rejection before any answer) leaves nothing to serve.
async fn watch_namespace(core: Arc<Mutex<Core>>, mut rx: web_transport_quinn::RecvStream) {
    let mut reader = v18::session::MessageReader::new();
    match reader.next(&mut rx).await {
        Ok(Some(v18::message::ControlMessage::RequestOk { .. })) => {}
        // An error, an unexpected message, or a stream that ended unanswered:
        // the namespace is not published.
        _ => core.lock().await.dead = true,
    }
}

/// Answer draft-18 request streams as they arrive, so a SUBSCRIBE that lands
/// before the first frame (or between two of them) is served when it lands.
async fn pump_requests(
    core: Arc<Mutex<Core>>,
    mut requests: mpsc::UnboundedReceiver<v18::session::PeerRequest>,
) {
    while let Some(request) = requests.recv().await {
        let mut guard = core.lock().await;
        match guard.handle_request(request).await {
            Ok(Some((id, rx))) => {
                drop(guard);
                tokio::spawn(watch_request(Arc::clone(&core), id, rx));
            }
            Ok(None) => {}
            Err(_) => {
                guard.dead = true;
                return;
            }
        }
    }
    // The session ended: no more requests will arrive.
    core.lock().await.dead = true;
}

/// Watch a live subscription's request stream. Draft-18 has no UNSUBSCRIBE: the
/// subscriber cancels by ending the stream, which is when the subscription is
/// dropped. A REQUEST_UPDATE is decoded and left unapplied: the subscription
/// keeps its original shape, which the draft allows the publisher to do with an
/// update it cannot satisfy.
async fn watch_request(core: Arc<Mutex<Core>>, id: u64, mut rx: web_transport_quinn::RecvStream) {
    let mut reader = v18::session::MessageReader::new();
    // Anything but a REQUEST_UPDATE on an established request stream, or its
    // end, means the request is over.
    while let Ok(Some(v18::message::ControlMessage::RequestUpdate { .. })) =
        reader.next(&mut rx).await
    {}
    let mut guard = core.lock().await;
    guard.drop_subscription(id);
    // Draft-18 cancels a FETCH the same way, by resetting its request stream.
    guard.cancel_fetch(id);
}

/// Answer inbound control messages as they arrive, so a SUBSCRIBE that lands
/// before the first frame (or between two of them) is served when it lands.
async fn pump_control(
    core: Arc<Mutex<Core>>,
    mut inbound: mpsc::UnboundedReceiver<ControlMessage>,
) {
    while let Some(msg) = inbound.recv().await {
        let mut guard = core.lock().await;
        if guard.handle_control(msg).await.is_err() {
            guard.dead = true;
            return;
        }
    }
    // The control stream ended: so has the session.
    core.lock().await.dead = true;
}

impl Core {
    fn new(cfg: Config, stats: Arc<Stats>) -> Self {
        Self {
            cfg,
            session: None,
            namespace_request: None,
            namespace_stream: None,
            subscriptions: Vec::new(),
            fetches: Vec::new(),
            pending_publishes: Vec::new(),
            published: false,
            publish_watch: Vec::new(),
            pending_subscribes: Vec::new(),
            init: Vec::new(),
            catalog: Vec::new(),
            tracks: Vec::new(),
            pending_header: Vec::new(),
            pending_object: Vec::new(),
            pending_track: None,
            pending_sync: false,
            dead: false,
            stats,
        }
    }

    fn namespace_tuple(&self) -> TrackNamespace {
        TrackNamespace::from_path(&self.cfg.namespace)
    }

    fn track_names(&self) -> Vec<String> {
        self.tracks.iter().map(|t| t.name.clone()).collect()
    }

    /// Whether the session is still usable. The pump reports what it saw here,
    /// since the pipeline only learns of it when it next has a frame.
    fn alive(&self) -> Result<(), G2gError> {
        if self.dead || self.session.as_ref().is_some_and(Wire::is_closed) {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        Ok(())
    }

    async fn handle_control(&mut self, msg: ControlMessage) -> Result<(), G2gError> {
        g2g_debug!(self, "control: {}", msg.name());
        match msg {
            ControlMessage::Subscribe {
                id,
                namespace,
                track_name,
                ..
            } => {
                self.handle_subscribe(id, namespace, track_name, None)
                    .await?;
            }
            ControlMessage::Unsubscribe { id } => self.drop_subscription(id),
            ControlMessage::PublishOk { id, .. } | ControlMessage::RequestOk { id, .. }
                if self.pending_publishes.iter().any(|p| p.request_id == id) =>
            {
                self.accept_publish(id).await?;
            }
            ControlMessage::RequestError { id, error_code, .. } => {
                // Our namespace publish being refused leaves nothing to serve.
                if Some(id) == self.namespace_request {
                    g2g_debug!(self, "PUBLISH_NAMESPACE rejected, code {error_code}");
                    return Err(G2gError::Hardware(HardwareError::Other));
                }
                // A refused PUBLISH costs that track, not the session.
                self.reject_publish(id);
            }
            ControlMessage::MaxRequestId { request_id } => {
                if let Some(Wire::V16(session)) = self.session.as_mut() {
                    session.set_peer_max_request_id(request_id);
                }
            }
            ControlMessage::PublishNamespaceCancel { .. } | ControlMessage::GoAway { .. } => {
                return Err(G2gError::Hardware(HardwareError::Other));
            }
            ControlMessage::Fetch {
                id,
                fetch_type,
                standalone,
                joining,
                ..
            } => {
                self.handle_fetch(id, fetch_type, standalone, joining, None)
                    .await?;
            }
            ControlMessage::FetchCancel { id } => self.cancel_fetch(id),
            // A subscriber-side request we do not serve. Draft-16 §4 asks for an
            // explicit refusal rather than silence.
            ControlMessage::TrackStatus { id, .. } | ControlMessage::RequestUpdate { id, .. } => {
                self.refuse(
                    id,
                    request_error_code::NOT_SUPPORTED,
                    String::from("not supported"),
                    None,
                )
                .await?;
            }
            // Everything else is a response to a request we did not make, or a
            // message only a subscriber acts on: decoded, then ignored.
            _ => {}
        }
        Ok(())
    }

    /// Handle one draft-18 request stream. Returns the request id and read half
    /// to watch when the request stays live (accepted or held), so the caller
    /// can observe the subscriber cancelling it.
    async fn handle_request(
        &mut self,
        request: v18::session::PeerRequest,
    ) -> Result<Option<(u64, web_transport_quinn::RecvStream)>, G2gError> {
        g2g_debug!(self, "request: {}", request.first.name());
        let v18::session::PeerRequest { first, tx, rx } = request;
        match first {
            v18::message::ControlMessage::Subscribe {
                id,
                namespace,
                track_name,
                ..
            } => {
                let live = self
                    .handle_subscribe(id, namespace, track_name, Some(tx))
                    .await?;
                Ok(live.then_some((id, rx)))
            }
            v18::message::ControlMessage::Fetch {
                id,
                fetch_type,
                standalone,
                joining,
                ..
            } => {
                // The draft-18 bodies carry the same fields under their own
                // types, so the serving side sees one shape.
                let standalone = standalone.map(|body| StandaloneFetch {
                    namespace: body.namespace,
                    track_name: body.track_name,
                    start: body.start,
                    end: body.end,
                });
                let joining = joining.map(|body| JoiningFetch {
                    joining_request_id: body.joining_request_id,
                    joining_start: body.joining_start,
                });
                let fetch_type = match fetch_type {
                    v18::message::FetchType::Standalone => FetchType::Standalone,
                    v18::message::FetchType::RelativeJoining => FetchType::RelativeJoining,
                    v18::message::FetchType::AbsoluteJoining => FetchType::AbsoluteJoining,
                };
                let live = self
                    .handle_fetch(id, fetch_type, standalone, joining, Some(tx))
                    .await?;
                // A live fetch keeps its request stream, because draft-18
                // cancels one by resetting it (§3.3.2).
                Ok(live.then_some((id, rx)))
            }
            // A request the publisher does not serve. §3.3.2: REQUEST_ERROR,
            // then FIN the stream.
            _ => {
                refuse_v18(
                    tx,
                    v18::message::request_error_code::NOT_SUPPORTED,
                    String::from("not supported"),
                )
                .await?;
                Ok(None)
            }
        }
    }

    /// Accept, refuse, or hold one SUBSCRIBE. The media track names come from
    /// the `moov`, and the session is dialled before the encoder has produced
    /// anything, so a subscription can arrive for a media track that does not
    /// exist yet: it is held until the `moov` resolves it. `reply` is the
    /// draft-18 request stream the answer goes on (`None` on draft-16).
    /// `Ok(true)` means the request is live: accepted or held.
    async fn handle_subscribe(
        &mut self,
        id: u64,
        namespace: TrackNamespace,
        track_name: TrackName,
        reply: Option<SendStream>,
    ) -> Result<bool, G2gError> {
        let name = track_name.as_str_lossy();
        let ours = namespace == self.namespace_tuple();
        let target = self.target_of(&namespace, &name);
        if self.subscriptions.iter().any(|s| s.request_id == id)
            || self.pending_subscribes.iter().any(|(held, ..)| *held == id)
        {
            self.refuse(
                id,
                request_error_code::DUPLICATE_SUBSCRIPTION,
                String::from("duplicate request id"),
                reply,
            )
            .await?;
            return Ok(false);
        }
        let Some(target) = target else {
            // Without a `moov` nothing names a media track yet, so hold the
            // request rather than refusing a track that is about to exist.
            if ours
                && self.tracks.is_empty()
                && self.pending_subscribes.len() < MAX_PENDING_SUBSCRIBES
            {
                g2g_debug!(self, "holding SUBSCRIBE {id} for {name} until the moov");
                self.pending_subscribes.push((id, name, reply));
                return Ok(true);
            }
            self.refuse(
                id,
                request_error_code::DOES_NOT_EXIST,
                format!("no track {name}"),
                reply,
            )
            .await?;
            return Ok(false);
        };
        self.accept_subscribe(id, target, reply).await?;
        Ok(true)
    }

    /// SUBSCRIBE_OK, then whatever the new subscription can already be served.
    async fn accept_subscribe(
        &mut self,
        id: u64,
        target: Target,
        reply: Option<SendStream>,
    ) -> Result<(), G2gError> {
        // The reference publisher reuses the request id as the track alias: it
        // is already unique within the session, which is all the draft asks.
        let reply = match reply {
            // Draft-18: the answer goes on the request's own stream, which then
            // stays open to carry PUBLISH_DONE.
            Some(mut tx) => {
                let msg = v18::message::ControlMessage::SubscribeOk {
                    track_alias: id,
                    params: v18::coding::MessageParams::new(),
                    properties: Params::new(),
                };
                v18::session::write_message(&mut tx, &msg).await?;
                Some(tx)
            }
            None => {
                self.send(ControlMessage::SubscribeOk {
                    id,
                    track_alias: id,
                    params: Params::new(),
                    extensions: Params::new(),
                })
                .await?;
                None
            }
        };
        // What a joining FETCH against this subscription ends at: the largest
        // object published when it was accepted.
        let joining = self.largest(&target);
        self.subscriptions.push(Subscription {
            request_id: id,
            track_alias: id,
            target,
            reply,
            streams: Vec::new(),
            serving_group: false,
            joining,
            delivered: false,
            streams_opened: 0,
        });
        self.serve_single_object_tracks().await
    }

    /// Offer every track with PUBLISH, so a subscriber that sends no SUBSCRIBE
    /// still receives the broadcast (`publish=true`). The tracks are only known
    /// once the `moov` names them, so this runs from there.
    async fn publish_tracks(&mut self) -> Result<(), G2gError> {
        if !self.cfg.publish || self.published || self.tracks.is_empty() {
            return Ok(());
        }
        self.published = true;
        let mut targets = Vec::new();
        if self.cfg.publish_catalog {
            targets.push((self.cfg.catalog_track.clone(), Target::Catalog));
        }
        targets.push((self.cfg.init_track.clone(), Target::Init));
        for track in &self.tracks {
            targets.push((track.name.clone(), Target::Media(track.track_id)));
        }
        for (name, target) in targets {
            self.publish_track(&name, target).await?;
        }
        Ok(())
    }

    /// One PUBLISH. The request id doubles as the track alias, as it does for a
    /// subscription we accepted.
    async fn publish_track(&mut self, name: &str, target: Target) -> Result<(), G2gError> {
        let namespace = self.namespace_tuple();
        let track_name = TrackName::new(name);
        let (id, reply, watch) = match self.session.as_mut().ok_or(G2gError::NotConfigured)? {
            Wire::V16(session) => {
                let id = session
                    .allocate_request_id()
                    .ok_or(G2gError::Hardware(HardwareError::Other))?;
                session
                    .send(&ControlMessage::Publish {
                        id,
                        namespace,
                        track_name,
                        track_alias: id,
                        params: Params::new(),
                        extensions: Params::new(),
                    })
                    .await?;
                (id, None, None)
            }
            Wire::V18(session) => {
                let id = session.allocate_request_id();
                let (tx, rx) = session
                    .open_request(&v18::message::ControlMessage::Publish {
                        id,
                        namespace,
                        track_name,
                        track_alias: id,
                        params: v18::coding::MessageParams::new(),
                        properties: Params::new(),
                    })
                    .await?;
                (id, Some(tx), Some(rx))
            }
        };
        g2g_debug!(self, "PUBLISH {name} as request {id}");
        self.pending_publishes.push(PendingPublish {
            request_id: id,
            target,
            reply,
        });
        if let Some(rx) = watch {
            self.publish_watch.push((id, rx));
        }
        Ok(())
    }

    /// The subscriber accepted a PUBLISH: it is a subscription from here on.
    async fn accept_publish(&mut self, id: u64) -> Result<(), G2gError> {
        let Some(at) = self
            .pending_publishes
            .iter()
            .position(|p| p.request_id == id)
        else {
            return Ok(());
        };
        let publish = self.pending_publishes.remove(at);
        g2g_debug!(self, "PUBLISH {id} accepted");
        let joining = self.largest(&publish.target);
        self.subscriptions.push(Subscription {
            request_id: id,
            track_alias: id,
            target: publish.target,
            reply: publish.reply,
            streams: Vec::new(),
            serving_group: false,
            joining,
            delivered: false,
            streams_opened: 0,
        });
        self.serve_single_object_tracks().await
    }

    /// The subscriber refused a PUBLISH, or its stream ended before it answered.
    fn reject_publish(&mut self, id: u64) {
        self.pending_publishes.retain(|p| p.request_id != id);
    }

    /// Take the draft-18 PUBLISH response streams still to be watched.
    fn take_publish_watch(&mut self) -> Vec<(u64, web_transport_quinn::RecvStream)> {
        core::mem::take(&mut self.publish_watch)
    }

    /// Answer the SUBSCRIBEs held for a track name the `moov` had not declared.
    async fn resolve_pending_subscribes(&mut self) -> Result<(), G2gError> {
        for (id, name, reply) in core::mem::take(&mut self.pending_subscribes) {
            let track_id = self
                .tracks
                .iter()
                .find(|t| t.name == name)
                .map(|t| t.track_id);
            match track_id {
                Some(track_id) => {
                    self.accept_subscribe(id, Target::Media(track_id), reply)
                        .await?
                }
                None => {
                    self.refuse(
                        id,
                        request_error_code::DOES_NOT_EXIST,
                        format!("no track {name}"),
                        reply,
                    )
                    .await?
                }
            }
        }
        Ok(())
    }

    async fn refuse(
        &mut self,
        id: u64,
        error_code: u64,
        reason: String,
        reply: Option<SendStream>,
    ) -> Result<(), G2gError> {
        match reply {
            Some(tx) => refuse_v18(tx, error_code, reason).await,
            None => {
                self.send(ControlMessage::RequestError {
                    id,
                    error_code,
                    retry_interval: 0,
                    reason,
                })
                .await
            }
        }
    }

    /// The track a namespace and name select, or `None` when this publisher does
    /// not serve it.
    fn target_of(&self, namespace: &TrackNamespace, name: &str) -> Option<Target> {
        if *namespace != self.namespace_tuple() {
            return None;
        }
        if name == self.cfg.catalog_track && self.cfg.publish_catalog {
            return Some(Target::Catalog);
        }
        if name == self.cfg.init_track {
            return Some(Target::Init);
        }
        self.tracks
            .iter()
            .find(|t| t.name == name)
            .map(|t| Target::Media(t.track_id))
    }

    /// The largest location published on a track, or `None` before its first
    /// object. The init and catalog tracks are one object in group 0.
    fn largest(&self, target: &Target) -> Option<Location> {
        match target {
            Target::Catalog => (!self.catalog.is_empty()).then(Location::default),
            Target::Init => (!self.init.is_empty()).then(Location::default),
            Target::Media(track_id) => self
                .tracks
                .iter()
                .find(|t| t.track_id == *track_id)
                .and_then(MediaTrack::largest),
        }
    }

    /// The oldest group still held for a track, or `None` when nothing is.
    fn oldest_cached(&self, target: &Target) -> Option<u64> {
        match target {
            Target::Catalog => (!self.catalog.is_empty()).then_some(0),
            Target::Init => (!self.init.is_empty()).then_some(0),
            Target::Media(track_id) => self
                .tracks
                .iter()
                .find(|t| t.track_id == *track_id)?
                .cache
                .front()
                .map(|g| g.group_id),
        }
    }

    /// Every cached object of `target` inside the range, in ascending
    /// (group, object) order: the order a fetch response is written in.
    fn cached_objects(
        &self,
        target: &Target,
        start: Location,
        end: Location,
    ) -> Vec<(u64, u64, Vec<u8>)> {
        let single = |payload: &Vec<u8>| {
            if payload.is_empty() || !in_fetch_range(Location::default(), start, end) {
                Vec::new()
            } else {
                Vec::from([(0, 0, payload.clone())])
            }
        };
        match target {
            Target::Catalog => single(&self.catalog),
            Target::Init => single(&self.init),
            Target::Media(track_id) => {
                let Some(track) = self.tracks.iter().find(|t| t.track_id == *track_id) else {
                    return Vec::new();
                };
                let mut out = Vec::new();
                for group in &track.cache {
                    for (object_id, payload) in group.objects.iter().enumerate() {
                        let loc = Location {
                            group_id: group.group_id,
                            object_id: object_id as u64,
                        };
                        if in_fetch_range(loc, start, end) {
                            out.push((loc.group_id, loc.object_id, payload.clone()));
                        }
                    }
                }
                out
            }
        }
    }

    /// What a FETCH asks for, once its track and range are resolved, or the
    /// error code and reason it is refused with. Everything here comes off the
    /// wire, so the range arithmetic is saturating and no allocation is sized
    /// from it: the objects come from the cache, which the publisher bounds.
    fn plan_fetch(
        &self,
        fetch_type: FetchType,
        standalone: Option<StandaloneFetch>,
        joining: Option<JoiningFetch>,
    ) -> Result<(Target, Location, Location), (u64, String)> {
        let missing_body = || {
            (
                request_error_code::INTERNAL_ERROR,
                String::from("fetch body missing"),
            )
        };
        let (target, start, end) = match fetch_type {
            FetchType::Standalone => {
                let body = standalone.ok_or_else(missing_body)?;
                let name = body.track_name.as_str_lossy();
                let target = self.target_of(&body.namespace, &name).ok_or((
                    request_error_code::DOES_NOT_EXIST,
                    format!("no track {name}"),
                ))?;
                (target, body.start, body.end)
            }
            FetchType::RelativeJoining | FetchType::AbsoluteJoining => {
                let body = joining.ok_or_else(missing_body)?;
                let sub = self
                    .subscriptions
                    .iter()
                    .find(|s| s.request_id == body.joining_request_id)
                    .ok_or((
                        request_error_code::INVALID_JOINING_REQUEST_ID,
                        String::from("no such subscription"),
                    ))?;
                let joined_at = sub.joining.ok_or((
                    request_error_code::INVALID_RANGE,
                    String::from("nothing published when the subscription started"),
                ))?;
                let start_group = match fetch_type {
                    FetchType::AbsoluteJoining => body.joining_start,
                    // A relative start before group 0 is the start of the track.
                    _ => joined_at.group_id.saturating_sub(body.joining_start),
                };
                (
                    sub.target.clone(),
                    Location {
                        group_id: start_group,
                        object_id: 0,
                    },
                    // The response ends at the object the subscription starts
                    // after, so the two are contiguous and do not overlap.
                    Location {
                        group_id: joined_at.group_id,
                        object_id: joined_at.object_id.saturating_add(1),
                    },
                )
            }
        };
        if !valid_fetch_range(start, end) {
            return Err((
                request_error_code::INVALID_RANGE,
                String::from("end before start"),
            ));
        }
        let largest = self.largest(&target).ok_or((
            request_error_code::INVALID_RANGE,
            String::from("nothing published yet"),
        ))?;
        if (start.group_id, start.object_id) > (largest.group_id, largest.object_id) {
            return Err((
                request_error_code::INVALID_RANGE,
                String::from("start past the largest object"),
            ));
        }
        let oldest = self.oldest_cached(&target).ok_or((
            request_error_code::INVALID_RANGE,
            String::from("nothing cached"),
        ))?;
        if start.group_id < oldest {
            return Err((
                request_error_code::INVALID_RANGE,
                String::from("group no longer cached"),
            ));
        }
        Ok((target, start, end))
    }

    /// Serve, or refuse, one FETCH. `reply` is the draft-18 request stream the
    /// answer goes on (`None` on draft-16). `Ok(true)` means the response stream
    /// is being written, so the request is live.
    async fn handle_fetch(
        &mut self,
        id: u64,
        fetch_type: FetchType,
        standalone: Option<StandaloneFetch>,
        joining: Option<JoiningFetch>,
        reply: Option<SendStream>,
    ) -> Result<bool, G2gError> {
        self.fetches.retain(|f| !f.task.is_finished());
        if self.fetches.len() >= MAX_ACTIVE_FETCHES {
            self.refuse(
                id,
                request_error_code::INTERNAL_ERROR,
                String::from("too many fetches in flight"),
                reply,
            )
            .await?;
            return Ok(false);
        }
        let (target, start, end) = match self.plan_fetch(fetch_type, standalone, joining) {
            Ok(plan) => plan,
            Err((code, reason)) => {
                g2g_debug!(self, "FETCH {id} refused: {reason}");
                self.refuse(id, code, reason, reply).await?;
                return Ok(false);
            }
        };
        let objects = self.cached_objects(&target, start, end);
        // The response cannot reach past what has been published, and FETCH_OK
        // is where the subscriber learns that.
        let largest = self.largest(&target).unwrap_or_default();
        let end = earlier_end(
            end,
            Location {
                group_id: largest.group_id,
                object_id: largest.object_id.saturating_add(1),
            },
        );
        match reply {
            Some(mut tx) => {
                let msg = v18::message::ControlMessage::FetchOk {
                    end_of_track: false,
                    end,
                    params: v18::coding::MessageParams::new(),
                    properties: Params::new(),
                };
                v18::session::write_message(&mut tx, &msg).await?;
                // Nothing else travels on a fetch's request stream: the objects
                // go on their own stream and a fetch has no PUBLISH_DONE.
                let _ = tx.finish();
            }
            None => {
                self.send(ControlMessage::FetchOk {
                    id,
                    end_of_track: false,
                    end,
                    params: Params::new(),
                    extensions: Params::new(),
                })
                .await?;
            }
        }
        self.start_fetch_stream(id, objects).await?;
        Ok(true)
    }

    /// Open the response stream and hand the objects to a task that writes them.
    /// Writing in its own task is what lets a cancel land between two objects
    /// rather than after the whole range.
    async fn start_fetch_stream(
        &mut self,
        request_id: u64,
        objects: Vec<(u64, u64, Vec<u8>)>,
    ) -> Result<(), G2gError> {
        let priority = self.priority_byte();
        let version = self.wire_version();
        let stream = match self.session.as_mut().ok_or(G2gError::NotConfigured)? {
            Wire::V16(session) => session.open_fetch(request_id, priority).await?,
            Wire::V18(session) => session.open_fetch(request_id, priority).await?,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let task = tokio::spawn(write_fetch_objects(
            version,
            stream,
            objects,
            priority,
            Arc::clone(&cancel),
            Arc::clone(&self.stats),
        ));
        self.fetches.push(ActiveFetch {
            request_id,
            cancel,
            task,
        });
        Stats::bump(&self.stats.fetches_served);
        Ok(())
    }

    /// Stop writing a FETCH response: draft-16 says so with FETCH_CANCEL,
    /// draft-18 by resetting the request stream.
    fn cancel_fetch(&mut self, id: u64) {
        if let Some(fetch) = self.fetches.iter().find(|f| f.request_id == id) {
            // The writer resets the response stream when it sees this, and
            // counts the cancellation: a fetch that had already finished writing
            // is not one that was stopped.
            fetch.cancel.store(true, Ordering::Relaxed);
        }
    }

    fn drop_subscription(&mut self, id: u64) {
        if let Some(at) = self.subscriptions.iter().position(|s| s.request_id == id) {
            let sub = self.subscriptions.remove(at);
            finish_streams(sub.streams);
        }
        self.pending_subscribes.retain(|(held, ..)| *held != id);
    }

    /// Send on the draft-16 control stream. Draft-18 responses ride each
    /// request's own stream, so nothing calls this on a draft-18 session.
    async fn send(&mut self, msg: ControlMessage) -> Result<(), G2gError> {
        match self.session.as_mut() {
            Some(Wire::V16(session)) => session.send(&msg).await,
            Some(Wire::V18(_)) => Err(G2gError::Hardware(HardwareError::Other)),
            None => Err(G2gError::NotConfigured),
        }
    }

    /// Deliver the init and catalog objects to any subscription still waiting
    /// for them. Both tracks hold exactly one object in group 0, so the whole
    /// track is one stream that is finished right away. Called again after each
    /// `moov`, so a subscription that arrived before the init segment existed is
    /// served as soon as it does.
    async fn serve_single_object_tracks(&mut self) -> Result<(), G2gError> {
        for i in 0..self.subscriptions.len() {
            if self.subscriptions[i].delivered {
                continue;
            }
            let payload = match self.subscriptions[i].target {
                Target::Init => self.init.clone(),
                Target::Catalog => self.catalog.clone(),
                Target::Media(_) => continue,
            };
            if payload.is_empty() {
                continue; // the moov has not arrived yet
            }
            let alias = self.subscriptions[i].track_alias;
            let version = self.wire_version();
            let mut stream = self.open_group_stream(alias, 0, 0).await?;
            write_object(version, &mut stream, 0, &payload).await?;
            let _ = stream.finish();
            self.subscriptions[i].delivered = true;
            self.subscriptions[i].streams_opened += 1;
            Stats::bump(&self.stats.objects_published);
        }
        Ok(())
    }

    fn priority_byte(&self) -> u8 {
        self.cfg.priority.min(u64::from(u8::MAX)) as u8
    }

    async fn open_group_stream(
        &mut self,
        track_alias: u64,
        group_id: u64,
        subgroup_id: u64,
    ) -> Result<SendStream, G2gError> {
        let priority = self.priority_byte();
        match self.session.as_mut().ok_or(G2gError::NotConfigured)? {
            Wire::V16(session) => {
                let header = SubgroupHeader {
                    header_type: HEADER_TYPE,
                    track_alias,
                    group_id,
                    subgroup_id: Some(subgroup_id),
                    publisher_priority: priority,
                };
                session.open_subgroup(&header).await
            }
            Wire::V18(session) => {
                let header = v18::data::SubgroupHeader {
                    header_type: v18::data::SubgroupHeaderType::explicit(),
                    track_alias,
                    group_id,
                    subgroup_id: Some(subgroup_id),
                    publisher_priority: Some(priority),
                };
                session.open_subgroup(&header).await
            }
        }
    }

    /// Which draft the live session speaks, for the object encoders.
    fn wire_version(&self) -> MoqtVersion {
        match self.session {
            Some(Wire::V18(_)) => MoqtVersion::V18,
            _ => MoqtVersion::V16,
        }
    }

    /// Walk one input frame's top-level boxes, turning them into objects.
    async fn push_bmff(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        let mut consumed = 0usize;
        // Collect first: a box walk borrows `bytes`, and publishing borrows self.
        let mut objects: Vec<(u32, bool, Vec<u8>)> = Vec::new();
        for (kind, payload) in boxes(bytes) {
            let whole = &bytes[consumed..consumed + payload.len() + 8];
            consumed += whole.len();
            match kind {
                b"ftyp" => self.init.extend_from_slice(whole),
                b"moov" => {
                    self.init.extend_from_slice(whole);
                    self.read_moov(payload)?;
                }
                // A segment header belongs to the fragment it opens, not to the
                // one already sent.
                b"styp" | b"prft" => self.pending_header.extend_from_slice(whole),
                b"moof" => {
                    let traf = find_box(payload, b"traf").ok_or(G2gError::CapsMismatch)?;
                    let tfhd = find_box(traf, b"tfhd").ok_or(G2gError::CapsMismatch)?;
                    let trun = find_box(traf, b"trun").ok_or(G2gError::CapsMismatch)?;
                    self.pending_track = Some(be32(tfhd, 4)?);
                    self.pending_sync = trun_first_sample_is_sync(trun)?;
                    self.pending_object.append(&mut self.pending_header);
                    self.pending_object.extend_from_slice(whole);
                }
                b"mdat" => {
                    let track = self.pending_track.take().ok_or(G2gError::CapsMismatch)?;
                    self.pending_object.extend_from_slice(whole);
                    objects.push((
                        track,
                        self.pending_sync,
                        core::mem::take(&mut self.pending_object),
                    ));
                }
                // Anything else rides with the fragment it precedes.
                _ => self.pending_object.extend_from_slice(whole),
            }
        }
        if consumed != bytes.len() {
            return Err(G2gError::CapsMismatch);
        }
        // The moov may have arrived in this very frame, so a subscription held
        // for a track it names can be answered, and one that was waiting on the
        // init segment or the catalog can be served now.
        if !self.tracks.is_empty() {
            self.resolve_pending_subscribes().await?;
            self.publish_tracks().await?;
        }
        self.serve_single_object_tracks().await?;
        for (track_id, sync, payload) in objects {
            self.publish_object(track_id, sync, &payload).await?;
        }
        Ok(())
    }

    /// Place one media object into its track's group and write it to every
    /// subscriber of that track.
    async fn publish_object(
        &mut self,
        track_id: u32,
        starts_group: bool,
        payload: &[u8],
    ) -> Result<(), G2gError> {
        let Some(at) = self.tracks.iter().position(|t| t.track_id == track_id) else {
            // A fragment for a track the moov never declared.
            return Err(G2gError::CapsMismatch);
        };
        if starts_group {
            if self.tracks[at].started {
                self.close_group(track_id).await;
                self.tracks[at].group_id += 1;
            }
            self.tracks[at].started = true;
            self.tracks[at].objects_in_group = 0;
            for sub in &mut self.subscriptions {
                if sub.target == Target::Media(track_id) {
                    sub.serving_group = true;
                }
            }
        } else if !self.tracks[at].started {
            // Nothing has opened a group yet, so this fragment has no group to
            // belong to: drop it rather than start a GOP mid-way.
            return Ok(());
        }
        let group_id = self.tracks[at].group_id;
        let object_id = self.tracks[at].objects_in_group;
        self.tracks[at].objects_in_group += 1;
        let depth = self.cfg.cache_groups as usize;
        self.tracks[at].cache_object(group_id, starts_group, payload, depth);

        let mut published = false;
        let mut reset = Vec::new();
        for i in 0..self.subscriptions.len() {
            // A subscription that has not seen a group boundary joined
            // mid-group: it waits for the next keyframe.
            if self.subscriptions[i].target != Target::Media(track_id)
                || !self.subscriptions[i].serving_group
            {
                continue;
            }
            if self.cfg.datagrams && self.send_object_datagram(i, group_id, object_id, payload)? {
                Stats::bump(&self.stats.datagram_objects);
                published = true;
                continue;
            }
            if self.cfg.datagrams {
                // Too large for the path MTU, or a peer that takes no
                // datagrams: the object still has to arrive, so it goes on a
                // subgroup stream.
                Stats::bump(&self.stats.datagram_fallbacks);
            }
            if self
                .write_on_subgroup(i, group_id, object_id, payload)
                .await?
            {
                published = true;
            } else {
                reset.push(i);
            }
        }
        for i in reset.into_iter().rev() {
            let sub = self.subscriptions.remove(i);
            finish_streams(sub.streams);
        }
        if published {
            Stats::bump(&self.stats.objects_published);
        }
        Ok(())
    }

    /// Send one object as a datagram to subscription `at`. `Ok(false)` when the
    /// session refused it, which is the caller's cue to use a stream.
    fn send_object_datagram(
        &mut self,
        at: usize,
        group_id: u64,
        object_id: u64,
        payload: &[u8],
    ) -> Result<bool, G2gError> {
        let alias = self.subscriptions[at].track_alias;
        let priority = self.priority_byte();
        match self.session.as_ref().ok_or(G2gError::NotConfigured)? {
            Wire::V16(session) => {
                let object =
                    DatagramObject::media(alias, group_id, object_id, priority, payload.to_vec());
                Ok(session.send_datagram(&object).is_ok())
            }
            Wire::V18(session) => {
                let object = v18::datagram::DatagramObject::media(
                    alias,
                    group_id,
                    object_id,
                    priority,
                    payload.to_vec(),
                );
                Ok(session.send_datagram(&object).is_ok())
            }
        }
    }

    /// Write one object onto subscription `at`'s subgroup stream for it, opening
    /// the stream on first use. `Ok(false)` means this subscriber reset the
    /// stream (it stopped reading), so the subscription is dropped and the rest
    /// keep being served; a session that is actually gone surfaces on the
    /// control stream instead.
    async fn write_on_subgroup(
        &mut self,
        at: usize,
        group_id: u64,
        object_id: u64,
        payload: &[u8],
    ) -> Result<bool, G2gError> {
        let subgroup_id = object_id % self.cfg.subgroups.max(1);
        let slot = match self.subscriptions[at]
            .streams
            .iter()
            .position(|s| s.subgroup_id == subgroup_id)
        {
            Some(slot) => slot,
            None => {
                let alias = self.subscriptions[at].track_alias;
                // Failing to open a stream is a dead session, not a dead
                // subscription: every subscription rides the one connection.
                let stream = self.open_group_stream(alias, group_id, subgroup_id).await?;
                let sub = &mut self.subscriptions[at];
                sub.streams.push(SubgroupStream {
                    subgroup_id,
                    stream,
                    prev_object_id: None,
                });
                sub.streams_opened += 1;
                sub.streams.len() - 1
            }
        };
        let version = self.wire_version();
        let open = &mut self.subscriptions[at].streams[slot];
        // The delta counts the distance to the previous id on *this* stream less
        // one; the first object of a stream takes the delta as its absolute id.
        let delta = match open.prev_object_id {
            Some(prev) => object_id.saturating_sub(prev).saturating_sub(1),
            None => object_id,
        };
        open.prev_object_id = Some(object_id);
        Ok(write_object(version, &mut open.stream, delta, payload)
            .await
            .is_ok())
    }

    /// End the group in progress on `track_id`: finish the subgroup streams
    /// that carried it and, in datagram mode, mark its end. A group carried only
    /// by datagrams has no stream whose close says it is done, so without the
    /// marker the subscriber would hold it until a buffering bound moved it on.
    async fn close_group(&mut self, track_id: u32) {
        let Some(track) = self.tracks.iter().find(|t| t.track_id == track_id) else {
            return;
        };
        let (group_id, objects_in_group) = (track.group_id, track.objects_in_group);
        for i in 0..self.subscriptions.len() {
            if self.subscriptions[i].target != Target::Media(track_id) {
                continue;
            }
            finish_streams(core::mem::take(&mut self.subscriptions[i].streams));
            if !self.cfg.datagrams || !self.subscriptions[i].serving_group {
                continue;
            }
            let alias = self.subscriptions[i].track_alias;
            let priority = self.priority_byte();
            match self.session.as_ref() {
                Some(Wire::V16(session)) => {
                    let marker =
                        DatagramObject::end_of_group(alias, group_id, objects_in_group, priority);
                    let _ = session.send_datagram(&marker);
                }
                Some(Wire::V18(session)) => {
                    let marker = v18::datagram::DatagramObject::end_of_group(
                        alias,
                        group_id,
                        objects_in_group,
                        priority,
                    );
                    let _ = session.send_datagram(&marker);
                }
                None => {}
            }
        }
    }

    /// Read the `moov`: the media tracks it declares and the catalog that
    /// describes them.
    fn read_moov(&mut self, moov: &[u8]) -> Result<(), G2gError> {
        let mut tracks = Vec::new();
        for (kind, trak) in boxes(moov) {
            if kind != b"trak" {
                continue;
            }
            let tkhd = find_box(trak, b"tkhd").ok_or(G2gError::CapsMismatch)?;
            // tkhd v0: track_ID at payload offset 12 (version/flags + two times).
            if tkhd.first() != Some(&0) {
                return Err(G2gError::CapsMismatch);
            }
            let track_id = be32(tkhd, 12)?;
            let stsd = find_path(trak, &[b"mdia", b"minf", b"stbl", b"stsd"])
                .ok_or(G2gError::CapsMismatch)?;
            let entries = stsd.get(8..).ok_or(G2gError::CapsMismatch)?;
            let name = if tracks.is_empty() && !self.cfg.track_name.is_empty() {
                self.cfg.track_name.clone()
            } else {
                format!("{track_id}.m4s")
            };
            tracks.push(MediaTrack {
                track_id,
                name,
                group_id: 0,
                objects_in_group: 0,
                started: false,
                selection_params: selection_params(entries),
                cache: VecDeque::new(),
            });
        }
        if tracks.is_empty() {
            return Err(G2gError::CapsMismatch);
        }
        self.tracks = tracks;
        self.catalog = self.build_catalog().into_bytes();
        Ok(())
    }

    /// The catalog document `moq-sub --catalog`, `moqtsrc` and the browser
    /// player read `initTrack` and each track's `name` from.
    fn build_catalog(&self) -> String {
        let tracks: Vec<(String, String)> = self
            .tracks
            .iter()
            .map(|t| (t.name.clone(), t.selection_params.clone()))
            .collect();
        catalog::build(
            &self.namespace_tuple().to_path(),
            &self.cfg.init_track,
            &tracks,
        )
    }

    /// Finish every open stream, tell each subscriber the track ended, and
    /// close the session.
    async fn finish(&mut self) -> Result<(), G2gError> {
        let started: Vec<u32> = self
            .tracks
            .iter()
            .filter(|t| t.started)
            .map(|t| t.track_id)
            .collect();
        for track_id in started {
            self.close_group(track_id).await;
        }
        let subs = core::mem::take(&mut self.subscriptions);
        for sub in subs {
            finish_streams(sub.streams);
            match sub.reply {
                // Draft-18: PUBLISH_DONE rides the subscription's own request
                // stream, and finishing it is what ends the request.
                Some(mut tx) => {
                    let msg = v18::message::ControlMessage::PublishDone {
                        status_code: v18::message::publish_done_code::TRACK_ENDED,
                        stream_count: sub.streams_opened,
                        reason: String::from("end of stream"),
                    };
                    v18::session::write_message(&mut tx, &msg).await?;
                    let _ = tx.finish();
                }
                None => {
                    self.send(ControlMessage::PublishDone {
                        id: sub.request_id,
                        status_code: publish_done_code::TRACK_ENDED,
                        stream_count: sub.streams_opened,
                        reason: String::from("end of stream"),
                    })
                    .await?;
                }
            }
        }
        // Draft-18 withdraws the namespace by finishing its request stream;
        // draft-16 says so with PUBLISH_NAMESPACE_DONE on the control stream.
        if let Some(mut ns) = self.namespace_stream.take() {
            self.namespace_request = None;
            let _ = ns.finish();
        }
        if let Some(id) = self.namespace_request.take() {
            self.send(ControlMessage::PublishNamespaceDone { id })
                .await?;
        }
        match self.session.as_mut() {
            Some(Wire::V16(session)) => session.close("eos").await,
            Some(Wire::V18(session)) => {
                session
                    .close(v18::message::session_error_code::NO_ERROR, "eos")
                    .await
            }
            None => {}
        }
        Ok(())
    }
}

/// A PUBLISH offered to the peer, waiting to be accepted.
#[derive(Debug)]
struct PendingPublish {
    request_id: u64,
    target: Target,
    /// The draft-18 request stream the answer arrives on, which then carries
    /// PUBLISH_DONE like a subscription's own stream.
    reply: Option<SendStream>,
}

/// Watch one draft-18 PUBLISH response stream: the first message decides
/// whether the track is being received, and the stream ending is how the
/// subscriber says it is done with it.
async fn watch_publish(core: Arc<Mutex<Core>>, id: u64, mut rx: web_transport_quinn::RecvStream) {
    let mut reader = v18::session::MessageReader::new();
    // §10.5 makes PUBLISH_OK shorthand for a REQUEST_OK answering a PUBLISH, so
    // either code point establishes the subscription.
    let accepted = matches!(
        reader.next(&mut rx).await,
        Ok(Some(
            v18::message::ControlMessage::PublishOk { .. }
                | v18::message::ControlMessage::RequestOk { .. }
        ))
    );
    {
        let mut guard = core.lock().await;
        if !accepted {
            guard.reject_publish(id);
            return;
        }
        if guard.accept_publish(id).await.is_err() {
            guard.dead = true;
            return;
        }
    }
    // Anything else on the stream, or its end, means the subscriber is done.
    while let Ok(Some(_)) = reader.next(&mut rx).await {}
    core.lock().await.drop_subscription(id);
}

/// A FETCH response still being written. The writer runs in its own task, so
/// the flag is how a cancel reaches it between two objects.
#[derive(Debug)]
struct ActiveFetch {
    request_id: u64,
    cancel: Arc<AtomicBool>,
    task: JoinHandle<()>,
}

/// Whether `loc` falls in a fetch range. `end` is the draft's encoding: the last
/// object plus one, with an object of 0 meaning the whole group.
fn in_fetch_range(loc: Location, start: Location, end: Location) -> bool {
    let after_start = (loc.group_id, loc.object_id) >= (start.group_id, start.object_id);
    let before_end = loc.group_id < end.group_id
        || (loc.group_id == end.group_id && (end.object_id == 0 || loc.object_id < end.object_id));
    after_start && before_end
}

/// A range whose end is before its start asks for nothing, which the drafts make
/// a refusal rather than an empty response.
fn valid_fetch_range(start: Location, end: Location) -> bool {
    end.group_id > start.group_id
        || (end.group_id == start.group_id
            && (end.object_id == 0 || end.object_id > start.object_id))
}

/// The earlier of two fetch end locations, under the same encoding.
fn earlier_end(a: Location, b: Location) -> Location {
    let key = |l: Location| {
        (
            l.group_id,
            if l.object_id == 0 {
                u64::MAX
            } else {
                l.object_id
            },
        )
    };
    if key(a) <= key(b) {
        a
    } else {
        b
    }
}

/// Write one FETCH response: the objects in ascending order, then a FIN. A
/// cancelled response is reset instead, because a FIN would tell the subscriber
/// the range ended where it did not.
async fn write_fetch_objects(
    version: MoqtVersion,
    mut stream: SendStream,
    objects: Vec<(u64, u64, Vec<u8>)>,
    priority: u8,
    cancel: Arc<AtomicBool>,
    stats: Arc<Stats>,
) {
    let mut writer = FetchWriter::new(version);
    let mut bytes = Vec::new();
    for (group_id, object_id, payload) in objects {
        if cancel.load(Ordering::Relaxed) {
            let _ = stream.reset(stream_error_code::CANCELLED);
            Stats::bump(&stats.fetches_cancelled);
            return;
        }
        bytes.clear();
        if writer
            .object(group_id, object_id, priority, &payload, &mut bytes)
            .is_err()
        {
            let _ = stream.reset(stream_error_code::CANCELLED);
            return;
        }
        if stream.write_all(&bytes).await.is_err() {
            return;
        }
    }
    if cancel.load(Ordering::Relaxed) {
        let _ = stream.reset(stream_error_code::CANCELLED);
        Stats::bump(&stats.fetches_cancelled);
        return;
    }
    let _ = stream.finish();
}

/// Write one object header plus its payload onto an open subgroup stream.
/// `object_id_delta` is the distance to the previous object id on this stream
/// less one, and the first object of a stream takes it as its absolute id: both
/// drafts resolve ids the same way, only the header bytes differ.
async fn write_object(
    version: MoqtVersion,
    stream: &mut SendStream,
    object_id_delta: u64,
    payload: &[u8],
) -> Result<(), G2gError> {
    let mut header = Vec::new();
    match version {
        MoqtVersion::V16 => SubgroupObjectHeader::normal(object_id_delta, payload.len())
            .encode(HEADER_TYPE, &mut header)
            .map_err(proto_err)?,
        MoqtVersion::V18 => v18::data::SubgroupObjectHeader::normal(object_id_delta, payload.len())
            .encode(v18::data::SubgroupHeaderType::explicit(), &mut header)
            .map_err(proto_err)?,
    }
    stream.write_all(&header).await.map_err(wt_err)?;
    stream.write_all(payload).await.map_err(wt_err)
}

/// Refuse a draft-18 request on its own stream: REQUEST_ERROR, then FIN
/// (§3.3.2).
async fn refuse_v18(mut tx: SendStream, error_code: u64, reason: String) -> Result<(), G2gError> {
    let msg = v18::message::ControlMessage::RequestError {
        error_code,
        retry_interval: 0,
        reason,
        redirect: None,
    };
    v18::session::write_message(&mut tx, &msg).await?;
    let _ = tx.finish();
    Ok(())
}

/// Finish every stream of a group that ended, so the subscriber learns nothing
/// more is coming for it.
fn finish_streams(streams: Vec<SubgroupStream>) {
    for mut open in streams {
        let _ = open.stream.finish();
    }
}

/// The catalog `selectionParams` for a sample entry, as a JSON fragment that
/// follows a comma. Empty when the entry names a codec we cannot describe: the
/// track is still listed, just without its parameters.
fn selection_params(entries: &[u8]) -> String {
    if let Some(avc1) = find_box(entries, b"avc1") {
        // Visual sample entry: width/height are 16-bit at payload offsets 24/26,
        // and the avcC follows the 78-byte fixed part.
        let (Ok(width), Ok(height)) = (be16(avc1, 24), be16(avc1, 26)) else {
            return String::new();
        };
        let Some(avcc) = avc1.get(78..).and_then(|c| find_box(c, b"avcC")) else {
            return String::new();
        };
        // avcC: configurationVersion, profile, compatibility, level.
        let Some(p) = avcc.get(1..4) else {
            return String::new();
        };
        return format!(
            ",\"selectionParams\":{{\"codec\":\"avc1.{:02X}{:02X}{:02X}\",\"width\":{width},\"height\":{height}}}",
            p[0], p[1], p[2]
        );
    }
    if let Some(mp4a) = find_box(entries, b"mp4a") {
        // Audio sample entry: channel count at 16, sample rate as 16.16 at 24.
        let (Ok(channels), Ok(rate)) = (be16(mp4a, 16), be32(mp4a, 24)) else {
            return String::new();
        };
        return format!(
            ",\"selectionParams\":{{\"codec\":\"mp4a.40.2\",\"samplerate\":{},\"channelConfig\":\"{channels}\"}}",
            rate >> 16
        );
    }
    String::new()
}

fn be16(data: &[u8], at: usize) -> Result<u16, G2gError> {
    data.get(at..at + 2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]))
        .ok_or(G2gError::CapsMismatch)
}

fn accepted_caps() -> Vec<Caps> {
    Vec::from([Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    }])
}

fn check_caps(caps: &Caps) -> Result<(), G2gError> {
    match caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        } => Ok(()),
        _ => Err(G2gError::CapsMismatch),
    }
}

impl LogSource for MoqtSink {
    fn log_category(&self) -> &'static str {
        "moqtsink"
    }
}

impl LogSource for Core {
    fn log_category(&self) -> &'static str {
        "moqtsink"
    }
}

impl AsyncElement for MoqtSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        check_caps(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::from_alternatives(accepted_caps()))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        check_caps(absolute_caps)?;
        self.configured = true;
        // A sink gets no CapsChanged before its first frame, so this is the only
        // chance to publish the namespace before the encoder has produced
        // anything: a subscriber that attaches during startup is then answered
        // by the pump instead of waiting for a fragment. Without a runtime to
        // spawn on (a sync caller), the first frame dials instead.
        if matches!(self.connect, Connect::Deferred)
            && tokio::runtime::Handle::try_current().is_ok()
        {
            self.connect = Connect::Dialing(tokio::spawn(connect_session(
                Arc::clone(&self.core),
                Arc::clone(&self.pump),
                self.dial_params(),
            )));
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if !self.configured {
                        return Err(G2gError::NotConfigured);
                    }
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.ready().await?;
                    let published = {
                        let mut core = self.core.lock().await;
                        core.alive()?;
                        core.push_bmff(slice).await?;
                        (
                            core.track_names(),
                            core.catalog.clone(),
                            core.take_publish_watch(),
                        )
                    };
                    // A PUBLISH answered on its own request stream needs a task
                    // holding the shared core, which only the element has.
                    for (id, rx) in published.2 {
                        tokio::spawn(watch_publish(Arc::clone(&self.core), id, rx));
                    }
                    (self.track_names, self.catalog) = (published.0, published.1);
                }
                PipelinePacket::CapsChanged(caps) => check_caps(&caps)?,
                PipelinePacket::Eos => {
                    // A dial that never completed leaves nothing to tell the
                    // relay, and its error already surfaced on the frame path.
                    let _ = self.ready().await;
                    self.core.lock().await.finish().await?;
                }
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MOQTSINK_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MoQ Transport sink",
            "Sink/Network",
            "Publishes a fragmented-MP4 stream to an IETF MoQ Transport relay over WebTransport",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let string = |v: &PropValue| v.as_str().map(ToString::to_string).ok_or(PropError::Type);
        match name {
            "location" => self.location = string(&value)?,
            "namespace" => self.cfg.namespace = string(&value)?,
            "track-name" => self.cfg.track_name = string(&value)?,
            "init-track-name" => self.cfg.init_track = string(&value)?,
            "catalog-track-name" => self.cfg.catalog_track = string(&value)?,
            "server-certificate-hashes" => self.cert_hashes = string(&value)?,
            "catalog" => self.cfg.publish_catalog = value.as_bool().ok_or(PropError::Type)?,
            "datagrams" => self.cfg.datagrams = value.as_bool().ok_or(PropError::Type)?,
            "subgroups" => self.cfg.subgroups = value.as_uint().ok_or(PropError::Type)?,
            "cache-groups" => self.cfg.cache_groups = value.as_uint().ok_or(PropError::Type)?,
            "publish" => self.cfg.publish = value.as_bool().ok_or(PropError::Type)?,
            "priority" => self.cfg.priority = value.as_uint().ok_or(PropError::Type)?,
            "max-request-id" => self.max_request_id = value.as_uint().ok_or(PropError::Type)?,
            "versions" => {
                let list = string(&value)?;
                parse_versions(&list).map_err(|_| PropError::Value)?;
                self.versions = list;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.location.clone())),
            "namespace" => Some(PropValue::Str(self.cfg.namespace.clone())),
            "track-name" => Some(PropValue::Str(self.cfg.track_name.clone())),
            "init-track-name" => Some(PropValue::Str(self.cfg.init_track.clone())),
            "catalog-track-name" => Some(PropValue::Str(self.cfg.catalog_track.clone())),
            "server-certificate-hashes" => Some(PropValue::Str(self.cert_hashes.clone())),
            "catalog" => Some(PropValue::Bool(self.cfg.publish_catalog)),
            "datagrams" => Some(PropValue::Bool(self.cfg.datagrams)),
            "subgroups" => Some(PropValue::Uint(self.cfg.subgroups)),
            "cache-groups" => Some(PropValue::Uint(self.cfg.cache_groups)),
            "publish" => Some(PropValue::Bool(self.cfg.publish)),
            "priority" => Some(PropValue::Uint(self.cfg.priority)),
            "max-request-id" => Some(PropValue::Uint(self.max_request_id)),
            "versions" => Some(PropValue::Str(self.versions.clone())),
            _ => None,
        }
    }
}

static MOQTSINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "WebTransport URL of the MoQT relay (e.g. https://host:4443/)",
    )
    .with_default("https://127.0.0.1:4443/"),
    PropertySpec::new(
        "namespace",
        PropKind::Str,
        "broadcast namespace published to the relay, a /-separated path",
    )
    .with_default("g2g"),
    PropertySpec::new(
        "track-name",
        PropKind::Str,
        "name of the first media track; empty names each track {track_id}.m4s",
    ),
    PropertySpec::new(
        "init-track-name",
        PropKind::Str,
        "track carrying the ftyp+moov init segment",
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
        "publish the JSON catalog track",
    )
    .with_default("true"),
    PropertySpec::new(
        "datagrams",
        PropKind::Bool,
        "carry media objects in QUIC datagrams instead of subgroup streams: unreliable and MTU-bounded, with an object too large for the path falling back to a stream",
    )
    .with_default("false"),
    PropertySpec::new(
        "subgroups",
        PropKind::Uint,
        "subgroup streams a group's objects are spread across, round-robin (1 = one stream per group)",
    )
    .with_default("1"),
    PropertySpec::new(
        "priority",
        PropKind::Uint,
        "publisher priority in every subgroup header (0-255, smaller is sent first)",
    )
    .with_default("127"),
    PropertySpec::new(
        "publish",
        PropKind::Bool,
        "offer every track with PUBLISH once the moov names them, instead of waiting for a SUBSCRIBE",
    )
    .with_default("false"),
    PropertySpec::new(
        "cache-groups",
        PropKind::Uint,
        "recently published groups kept per track to answer a FETCH (0 = keep none, every FETCH is refused)",
    )
    .with_default("4"),
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
        "server-certificate-hashes",
        PropKind::Str,
        "accept only relay certificates with these SHA-256 digests (hex, comma-separated); empty = system roots",
    ),
];

impl PadTemplates for MoqtSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(
            accepted_caps(),
        ))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_names_the_init_track_and_each_media_track() {
        let mut core = Core::new(Config::new("live/cam"), Arc::new(Stats::default()));
        core.tracks = Vec::from([MediaTrack {
            track_id: 1,
            name: String::from("1.m4s"),
            group_id: 0,
            objects_in_group: 0,
            started: false,
            selection_params: String::from(
                ",\"selectionParams\":{\"codec\":\"avc1.64000D\",\"width\":320,\"height\":240}",
            ),
            cache: VecDeque::new(),
        }]);
        let catalog = core.build_catalog();
        assert!(catalog.contains("\"namespace\":\"/live/cam\""), "{catalog}");
        assert!(catalog.contains("\"initTrack\":\"0.mp4\""), "{catalog}");
        assert!(catalog.contains("\"name\":\"1.m4s\""), "{catalog}");
        assert!(catalog.contains("\"codec\":\"avc1.64000D\""), "{catalog}");
    }
}

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
//!   keyframe. Each group is one subgroup on its own unidirectional stream.
//! - a `.catalog` track carries the JSON track list a browser player reads
//!   (`moq-sub --catalog` reads the same document).
//!
//! Every group's stream is opened only after SUBSCRIBE_OK for that
//! subscription, so the subscriber can resolve the track alias in the stream
//! header. Inbound control messages are decoded by the session's reader task
//! and applied when the next frame arrives, so a subscription that lands
//! between frames is served on the following one.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use web_transport_quinn::SendStream;

use g2g_core::{
    g2g_debug, AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, HardwareError, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use g2g_core::log::LogSource;

use crate::fmp4::trun_first_sample_is_sync;
use crate::moqt::coding::{MoqtError, Params, TrackName, TrackNamespace};
use crate::moqt::data::{StreamHeaderType, SubgroupHeader, SubgroupObjectHeader};
use crate::moqt::message::{publish_done_code, request_error_code, ControlMessage};
use crate::moqt::session::{implementation_name, MoqtSession};
use crate::mp4box::{be32, boxes, find_box, find_path};
use crate::remotewtio::wt_err;

/// The reference publisher writes every subgroup with an explicit subgroup id
/// and an extension-header block (`session/subscribed.rs`), so a relay sees the
/// same header type from us as from `moq-pub`.
const HEADER_TYPE: StreamHeaderType = StreamHeaderType::SubgroupIdExt;

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

/// One accepted SUBSCRIBE and the stream state serving it.
#[derive(Debug)]
struct Subscription {
    request_id: u64,
    track_alias: u64,
    target: Target,
    /// The open subgroup stream, when a group is in progress.
    stream: Option<SendStream>,
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
}

/// Publishes an fMP4 byte stream to an IETF MoQ Transport relay.
#[derive(Debug)]
pub struct MoqtSink {
    location: String,
    cert_hashes: String,
    namespace: String,
    init_track: String,
    catalog_track: String,
    track_name: String,
    publish_catalog: bool,
    priority: u64,
    max_request_id: u64,

    configured: bool,
    session: Option<MoqtSession>,
    /// Request id of our PUBLISH_NAMESPACE, once sent.
    namespace_request: Option<u64>,
    subscriptions: Vec<Subscription>,

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

    objects_published: u64,
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
        Self {
            location: location.into(),
            cert_hashes: String::new(),
            namespace: namespace.into(),
            init_track: String::from("0.mp4"),
            catalog_track: String::from(".catalog"),
            track_name: String::new(),
            publish_catalog: true,
            priority: 127,
            max_request_id: 100,
            configured: false,
            session: None,
            namespace_request: None,
            subscriptions: Vec::new(),
            init: Vec::new(),
            catalog: Vec::new(),
            tracks: Vec::new(),
            pending_header: Vec::new(),
            pending_object: Vec::new(),
            pending_track: None,
            pending_sync: false,
            objects_published: 0,
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
        self.priority = priority;
        self
    }

    /// Objects written to at least one subscriber so far.
    pub fn objects_published(&self) -> u64 {
        self.objects_published
    }

    /// The media track names the `moov` produced, in track order.
    pub fn track_names(&self) -> Vec<String> {
        self.tracks.iter().map(|t| t.name.clone()).collect()
    }

    /// The catalog document as published, for tests and for a caller serving it
    /// out of band.
    pub fn catalog(&self) -> &[u8] {
        &self.catalog
    }

    fn namespace_tuple(&self) -> TrackNamespace {
        TrackNamespace::from_path(&self.namespace)
    }

    /// Dial the relay, complete SETUP, and publish the namespace. Deferred to
    /// the first frame because the handshake is async.
    async fn ensure_session(&mut self) -> Result<(), G2gError> {
        if self.session.is_some() {
            return Ok(());
        }
        let mut session = MoqtSession::connect(
            &self.location,
            &self.cert_hashes,
            self.max_request_id,
            &implementation_name(),
        )
        .await?;
        let id = session
            .allocate_request_id()
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        session
            .send(&ControlMessage::PublishNamespace {
                id,
                namespace: self.namespace_tuple(),
                params: Params::new(),
            })
            .await?;
        self.namespace_request = Some(id);
        self.session = Some(session);
        Ok(())
    }

    /// Apply every control message the reader task has decoded.
    async fn pump_control(&mut self) -> Result<(), G2gError> {
        while let Some(msg) = self.session.as_mut().and_then(MoqtSession::poll_control) {
            self.handle_control(msg).await?;
        }
        if self.session.as_ref().is_some_and(MoqtSession::is_closed) {
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
            } => self.handle_subscribe(id, namespace, track_name).await?,
            ControlMessage::Unsubscribe { id } => self.drop_subscription(id),
            ControlMessage::RequestError { id, error_code, .. } => {
                // Our namespace publish being refused leaves nothing to serve.
                if Some(id) == self.namespace_request {
                    g2g_debug!(self, "PUBLISH_NAMESPACE rejected, code {error_code}");
                    return Err(G2gError::Hardware(HardwareError::Other));
                }
            }
            ControlMessage::MaxRequestId { request_id } => {
                if let Some(session) = self.session.as_mut() {
                    session.set_peer_max_request_id(request_id);
                }
            }
            ControlMessage::PublishNamespaceCancel { .. } | ControlMessage::GoAway { .. } => {
                return Err(G2gError::Hardware(HardwareError::Other));
            }
            // A subscriber-side request we do not serve. Draft-16 §4 asks for an
            // explicit refusal rather than silence.
            ControlMessage::Fetch { id, .. }
            | ControlMessage::TrackStatus { id, .. }
            | ControlMessage::RequestUpdate { id, .. } => {
                self.send(ControlMessage::RequestError {
                    id,
                    error_code: request_error_code::NOT_SUPPORTED,
                    retry_interval: 0,
                    reason: String::from("not supported"),
                })
                .await?;
            }
            // Everything else is a response to a request we did not make, or a
            // message only a subscriber acts on: decoded, then ignored.
            _ => {}
        }
        Ok(())
    }

    /// Accept or refuse one SUBSCRIBE. The track names come from the `moov`,
    /// which arrives in the same frame that opens the session, and control
    /// messages are applied at the start of the *next* frame, so a media track
    /// is always named by the time a subscription for it can be seen.
    async fn handle_subscribe(
        &mut self,
        id: u64,
        namespace: TrackNamespace,
        track_name: TrackName,
    ) -> Result<(), G2gError> {
        let name = track_name.as_str_lossy();
        let target = if namespace != self.namespace_tuple() {
            None
        } else if name == self.catalog_track && self.publish_catalog {
            Some(Target::Catalog)
        } else if name == self.init_track {
            Some(Target::Init)
        } else {
            self.tracks
                .iter()
                .find(|t| t.name == name)
                .map(|t| Target::Media(t.track_id))
        };
        let Some(target) = target else {
            return self
                .send(ControlMessage::RequestError {
                    id,
                    error_code: request_error_code::DOES_NOT_EXIST,
                    retry_interval: 0,
                    reason: format!("no track {name}"),
                })
                .await;
        };
        if self.subscriptions.iter().any(|s| s.request_id == id) {
            return self
                .send(ControlMessage::RequestError {
                    id,
                    error_code: request_error_code::DUPLICATE_SUBSCRIPTION,
                    retry_interval: 0,
                    reason: String::from("duplicate request id"),
                })
                .await;
        }
        // The reference publisher reuses the request id as the track alias: it
        // is already unique within the session, which is all §10.1 asks.
        self.send(ControlMessage::SubscribeOk {
            id,
            track_alias: id,
            params: Params::new(),
            extensions: Params::new(),
        })
        .await?;
        self.subscriptions.push(Subscription {
            request_id: id,
            track_alias: id,
            target,
            stream: None,
            delivered: false,
            streams_opened: 0,
        });
        self.serve_single_object_tracks().await
    }

    fn drop_subscription(&mut self, id: u64) {
        if let Some(at) = self.subscriptions.iter().position(|s| s.request_id == id) {
            let mut sub = self.subscriptions.remove(at);
            if let Some(stream) = sub.stream.as_mut() {
                let _ = stream.finish();
            }
        }
    }

    async fn send(&mut self, msg: ControlMessage) -> Result<(), G2gError> {
        match self.session.as_mut() {
            Some(session) => session.send(&msg).await,
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
            let mut stream = self.open_group_stream(alias, 0).await?;
            write_object(&mut stream, &payload).await?;
            let _ = stream.finish();
            self.subscriptions[i].delivered = true;
            self.subscriptions[i].streams_opened += 1;
            self.objects_published += 1;
        }
        Ok(())
    }

    async fn open_group_stream(
        &mut self,
        track_alias: u64,
        group_id: u64,
    ) -> Result<SendStream, G2gError> {
        let header = SubgroupHeader {
            header_type: HEADER_TYPE,
            track_alias,
            group_id,
            subgroup_id: Some(0),
            publisher_priority: self.priority.min(u64::from(u8::MAX)) as u8,
        };
        self.session
            .as_mut()
            .ok_or(G2gError::NotConfigured)?
            .open_subgroup(&header)
            .await
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
        // The moov may have arrived in this very frame, so a subscription that
        // was waiting on the init segment can be served now.
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
                self.tracks[at].group_id += 1;
            }
            self.tracks[at].started = true;
            self.tracks[at].objects_in_group = 0;
        } else if !self.tracks[at].started {
            // Nothing has opened a group yet, so this fragment has no group to
            // belong to: drop it rather than start a GOP mid-way.
            return Ok(());
        }
        let group_id = self.tracks[at].group_id;
        self.tracks[at].objects_in_group += 1;

        let mut published = false;
        let mut reset = Vec::new();
        for i in 0..self.subscriptions.len() {
            if self.subscriptions[i].target != Target::Media(track_id) {
                continue;
            }
            if starts_group {
                if let Some(stream) = self.subscriptions[i].stream.as_mut() {
                    let _ = stream.finish();
                }
                let alias = self.subscriptions[i].track_alias;
                // Failing to open a stream is a dead session, not a dead
                // subscription: every subscription rides the one connection.
                let stream = self.open_group_stream(alias, group_id).await?;
                self.subscriptions[i].stream = Some(stream);
                self.subscriptions[i].streams_opened += 1;
            }
            let Some(stream) = self.subscriptions[i].stream.as_mut() else {
                continue; // joined mid-group: wait for the next keyframe
            };
            // A write that fails means this subscriber reset its stream (it
            // stopped reading). Drop the subscription and keep serving the
            // rest; a session that is actually gone surfaces on the control
            // stream instead.
            if write_object(stream, payload).await.is_err() {
                reset.push(i);
                continue;
            }
            published = true;
        }
        for i in reset.into_iter().rev() {
            self.subscriptions.remove(i);
        }
        if published {
            self.objects_published += 1;
        }
        Ok(())
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
            let name = if tracks.is_empty() && !self.track_name.is_empty() {
                self.track_name.clone()
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
            });
        }
        if tracks.is_empty() {
            return Err(G2gError::CapsMismatch);
        }
        self.tracks = tracks;
        self.catalog = self.build_catalog().into_bytes();
        Ok(())
    }

    /// The catalog document (`moq-catalog`'s `Root`), written by hand so the
    /// sink needs no JSON dependency. `moq-sub --catalog` and the browser
    /// player read `initTrack` and each track's `name` from it.
    fn build_catalog(&self) -> String {
        let namespace = self.namespace_tuple().to_path();
        let mut tracks = String::new();
        for (i, track) in self.tracks.iter().enumerate() {
            if i > 0 {
                tracks.push(',');
            }
            tracks.push_str(&format!(
                "{{\"name\":\"{}\",\"initTrack\":\"{}\"{}}}",
                track.name, self.init_track, track.selection_params
            ));
        }
        format!(
            concat!(
                "{{\"version\":1,\"streamingFormat\":1,\"streamingFormatVersion\":\"0.2\",",
                "\"supportsDeltaUpdates\":true,",
                "\"commonTrackFields\":{{\"namespace\":\"{}\",\"packaging\":\"cmaf\",\"renderGroup\":1}},",
                "\"tracks\":[{}]}}"
            ),
            namespace, tracks
        )
    }

    /// Finish every open stream, tell each subscriber the track ended, and
    /// close the session.
    async fn finish(&mut self) -> Result<(), G2gError> {
        let subs = core::mem::take(&mut self.subscriptions);
        for mut sub in subs {
            if let Some(stream) = sub.stream.as_mut() {
                let _ = stream.finish();
            }
            self.send(ControlMessage::PublishDone {
                id: sub.request_id,
                status_code: publish_done_code::TRACK_ENDED,
                stream_count: sub.streams_opened,
                reason: String::from("end of stream"),
            })
            .await?;
        }
        if let Some(id) = self.namespace_request.take() {
            self.send(ControlMessage::PublishNamespaceDone { id })
                .await?;
        }
        if let Some(session) = self.session.as_mut() {
            session.close("eos").await;
        }
        Ok(())
    }
}

/// Write one object header plus its payload onto an open subgroup stream.
///
/// Object ids are consecutive from zero within a group, and the delta is
/// measured from the previous id less one, so it is always zero: the first
/// object's id is the delta itself (`session/subscriber.rs`).
async fn write_object(stream: &mut SendStream, payload: &[u8]) -> Result<(), G2gError> {
    let mut header = Vec::new();
    SubgroupObjectHeader::normal(0, payload.len())
        .encode(HEADER_TYPE, &mut header)
        .map_err(proto_err)?;
    stream.write_all(&header).await.map_err(wt_err)?;
    stream.write_all(payload).await.map_err(wt_err)
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
                    self.ensure_session().await?;
                    self.pump_control().await?;
                    self.push_bmff(slice).await?;
                }
                PipelinePacket::CapsChanged(caps) => check_caps(&caps)?,
                PipelinePacket::Eos => self.finish().await?,
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
            "namespace" => self.namespace = string(&value)?,
            "track-name" => self.track_name = string(&value)?,
            "init-track-name" => self.init_track = string(&value)?,
            "catalog-track-name" => self.catalog_track = string(&value)?,
            "server-certificate-hashes" => self.cert_hashes = string(&value)?,
            "catalog" => self.publish_catalog = value.as_bool().ok_or(PropError::Type)?,
            "priority" => self.priority = value.as_uint().ok_or(PropError::Type)?,
            "max-request-id" => self.max_request_id = value.as_uint().ok_or(PropError::Type)?,
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
            "catalog" => Some(PropValue::Bool(self.publish_catalog)),
            "priority" => Some(PropValue::Uint(self.priority)),
            "max-request-id" => Some(PropValue::Uint(self.max_request_id)),
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
        "priority",
        PropKind::Uint,
        "publisher priority in every subgroup header (0-255, smaller is sent first)",
    )
    .with_default("127"),
    PropertySpec::new(
        "max-request-id",
        PropKind::Uint,
        "MAX_REQUEST_ID advertised to the relay in CLIENT_SETUP",
    )
    .with_default("100"),
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
        let mut sink = MoqtSink::new("https://relay:4443/", "live/cam");
        sink.tracks = Vec::from([MediaTrack {
            track_id: 1,
            name: String::from("1.m4s"),
            group_id: 0,
            objects_in_group: 0,
            started: false,
            selection_params: String::from(
                ",\"selectionParams\":{\"codec\":\"avc1.64000D\",\"width\":320,\"height\":240}",
            ),
        }]);
        let catalog = sink.build_catalog();
        assert!(catalog.contains("\"namespace\":\"/live/cam\""), "{catalog}");
        assert!(catalog.contains("\"initTrack\":\"0.mp4\""), "{catalog}");
        assert!(catalog.contains("\"name\":\"1.m4s\""), "{catalog}");
        assert!(catalog.contains("\"codec\":\"avc1.64000D\""), "{catalog}");
    }
}

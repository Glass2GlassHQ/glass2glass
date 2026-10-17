//! WHIP server source (`WhipServerSrc`): accepts one RFC 9725 WHIP publisher
//! (a browser, OBS, or [`crate::webrtcsink::WebRtcSink`]) and emits its video on
//! output 0 and its Opus audio on output 1. The g2g counterpart of GStreamer's
//! `whipserversrc`.
//!
//! The element is the HTTP server and the SDP answerer. A `POST` of an
//! `application/sdp` offer gets `201 Created` with the answer SDP and a
//! `Location` resource, which the publisher `PATCH`es with trickle candidates
//! and `DELETE`s at teardown. Every response carries CORS headers so a browser
//! page on another origin can publish. The answerer learns its `Mid`s from
//! `Event::MediaAdded`: the first video m-line feeds output 0, the first audio
//! m-line output 1.
//!
//! Limits: one publisher per run (a second `POST` gets `409` and the run ends
//! when the publisher leaves), plain HTTP only, and an ICE restart `PATCH` (a
//! new `a=ice-ufrag`) gets `501`.
//!
//! Video is gated until the first keyframe and a PLI is repeated every second
//! until one arrives.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, UdpSocket};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_native_tls::{native_tls, TlsAcceptor};

use str0m::change::SdpOffer;
use str0m::crypto::from_feature_flags;
use str0m::media::{KeyframeRequestKind, MediaKind, Mid};
use str0m::{Candidate, Event, IceConnectionState, IceCreds, Input, Output, Rtc, RtcConfig};

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource, Target};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_debug, g2g_info, g2g_warn, Caps, FrameTiming, G2gError, HardwareError, MemoryDomain,
    MultiOutputSink, MultiOutputSource, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::filesink::io_err;
use crate::turn::TurnSet;
use crate::webrtc_codec::{
    enable_tracks, get_video_codec_property, opus_caps, set_video_codec_property, Track,
    WebRtcVideoCodec, VIDEO_CODEC_PROP,
};
use crate::webrtc_stats::{WebRtcStatsHandle, STATS_INTERVAL};
use crate::webrtc_util::{
    add_ice_candidates, feed_datagram, media_time_ns, parse_restart_answer_frag, random_hex_id,
    select_host_ip, send_transmit,
};

// each publisher slot owns a pad pair: slot * PORTS_PER_PUBLISHER + VIDEO_PORT / AUDIO_PORT
const VIDEO_PORT: usize = 0;
const AUDIO_PORT: usize = 1;
const PORTS_PER_PUBLISHER: usize = 2;
const DEFAULT_MAX_PUBLISHERS: usize = 1;
// pad indices are u8 in the graph
const MAX_PUBLISHERS: usize = u8::MAX as usize / PORTS_PER_PUBLISHER;
const DATAGRAM_QUEUE_DEPTH: usize = 256;
const PLI_INTERVAL: Duration = Duration::from_secs(1);
pub const DEFAULT_HOST_ADDR: &str = "http://127.0.0.1:8080";
const HTTP_SCHEME: &str = "http://";
const HTTP_DEFAULT_PORT: u16 = 80;
const HTTPS_SCHEME: &str = "https://";
const HTTPS_DEFAULT_PORT: u16 = 443;
const RESOURCE_PATH_PREFIX: &str = "/whip/resource/";
const SDP_CONTENT_TYPE: &str = "application/sdp";
const TRICKLE_CONTENT_TYPE: &str = "application/trickle-ice-sdpfrag";
const TEXT_CONTENT_TYPE: &str = "text/plain";
const BEARER_SCHEME: &str = "Bearer ";
const IF_MATCH_ANY: &str = "*";
const HEAD_TERMINATOR: &[u8] = b"\r\n\r\n";
const MAX_HEAD_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 1024 * 1024;
const READ_CHUNK_BYTES: usize = 4096;
const REQUEST_QUEUE_DEPTH: usize = 16;
const UDP_RECV_BYTES: usize = 2000;
const STEREO: u8 = 2;
const CORS_HEADERS: &[(&str, &str)] = &[
    ("Access-Control-Allow-Origin", "*"),
    (
        "Access-Control-Allow-Methods",
        "POST, PATCH, DELETE, OPTIONS",
    ),
    (
        "Access-Control-Allow-Headers",
        "Authorization, Content-Type, If-Match",
    ),
    ("Access-Control-Expose-Headers", "Location, ETag"),
];

/// WHIP server ingest. See the module docs.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::whipserversrc::WhipServerSrc;
///
/// let source = WhipServerSrc::new("http://0.0.0.0:8190").with_bearer("token");
/// ```
pub struct WhipServerSrc {
    host_addr: String,
    bearer: Option<String>,
    stun_server: Option<String>,
    video_codec: WebRtcVideoCodec,
    frame_limit: u64,
    tls_cert_file: Option<String>,
    tls_key_file: Option<String>,
    max_publishers: usize,
    /// One per publisher slot.
    stats: Vec<WebRtcStatsHandle>,
    log_name: LogName,
}

impl core::fmt::Debug for WhipServerSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WhipServerSrc")
            .field("host_addr", &self.host_addr)
            .field("video_codec", &self.video_codec)
            .finish()
    }
}

impl WhipServerSrc {
    /// Serve WHIP on the host and port of `host_addr` (an `http://` URL).
    pub fn new(host_addr: impl Into<String>) -> Self {
        Self {
            host_addr: host_addr.into(),
            bearer: None,
            stun_server: None,
            video_codec: WebRtcVideoCodec::default(),
            frame_limit: u64::MAX,
            tls_cert_file: None,
            tls_key_file: None,
            max_publishers: DEFAULT_MAX_PUBLISHERS,
            stats: alloc::vec![WebRtcStatsHandle::default()],
            log_name: LogName::new(),
        }
    }

    /// Require `Authorization: Bearer <token>` on every request.
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.bearer = Some(token.into());
        self
    }

    /// Add a server-reflexive candidate from this STUN server (`host:port`).
    pub fn with_stun_server(mut self, server: impl Into<String>) -> Self {
        self.stun_server = Some(server.into());
        self
    }

    /// Accept and emit video in `codec` instead of H.264.
    pub fn with_video_codec(mut self, codec: WebRtcVideoCodec) -> Self {
        self.video_codec = codec;
        self
    }

    /// Stop after `n` access units across both tracks, then EOS on both.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// Serve HTTPS from this PEM certificate chain and PKCS#8 key (the
    /// `host-addr` must then be an `https://` URL).
    pub fn with_tls(mut self, cert_file: impl Into<String>, key_file: impl Into<String>) -> Self {
        self.tls_cert_file = Some(cert_file.into());
        self.tls_key_file = Some(key_file.into());
        self
    }

    /// Accept up to `n` publishers at once, each on its own pad pair (slot `i`
    /// emits video on output `2i` and Opus on `2i + 1`). The run ends when the
    /// last connected publisher leaves.
    pub fn with_max_publishers(mut self, n: usize) -> Self {
        self.set_max_publishers(n);
        self
    }

    fn set_max_publishers(&mut self, n: usize) {
        self.max_publishers = n.clamp(1, MAX_PUBLISHERS);
        self.stats
            .resize_with(self.max_publishers, WebRtcStatsHandle::default);
    }

    /// Live counters of the first publisher slot's connection, readable during
    /// the run.
    pub fn stats_handle(&self) -> WebRtcStatsHandle {
        self.stats[0].clone()
    }

    /// Live counters per publisher slot.
    pub fn publisher_stats_handles(&self) -> Vec<WebRtcStatsHandle> {
        self.stats.clone()
    }

    fn tls_acceptor(&self, tls: bool) -> Result<Option<TlsAcceptor>, String> {
        if !tls {
            return Ok(None);
        }
        let (Some(cert_file), Some(key_file)) = (&self.tls_cert_file, &self.tls_key_file) else {
            return Err("an https host-addr needs tls-cert-file and tls-key-file".to_string());
        };
        let cert =
            std::fs::read(cert_file).map_err(|e| format!("tls-cert-file {cert_file}: {e}"))?;
        let key = std::fs::read(key_file).map_err(|e| format!("tls-key-file {key_file}: {e}"))?;
        let identity = native_tls::Identity::from_pkcs8(&cert, &key)
            .map_err(|e| format!("tls certificate and key: {e}"))?;
        let acceptor =
            native_tls::TlsAcceptor::new(identity).map_err(|e| format!("tls acceptor: {e}"))?;
        Ok(Some(TlsAcceptor::from(acceptor)))
    }

    fn authorized(&self, request: &HttpRequest) -> bool {
        let Some(token) = self.bearer.as_deref() else {
            return true;
        };
        request
            .authorization
            .as_deref()
            .and_then(|value| value.strip_prefix(BEARER_SCHEME))
            .is_some_and(|presented| presented.trim() == token)
    }

    async fn dispatch(
        &self,
        request: &HttpRequest,
        sessions: &mut [Option<Session>],
        datagrams: &mpsc::Sender<ReceivedDatagram>,
    ) -> Dispatched {
        if !self.authorized(request) {
            return Dispatched::keep(HttpResponse::new(Status::Unauthorized));
        }
        if request.method == "POST" {
            return Dispatched::keep(self.accept_publisher(request, sessions, datagrams).await);
        }
        let resource_id = request.path.strip_prefix(RESOURCE_PATH_PREFIX);
        let resource = sessions
            .iter_mut()
            .flatten()
            .find(|s| resource_id == Some(s.resource_id.as_str()));
        match (request.method.as_str(), resource) {
            ("PATCH", Some(s)) => Dispatched::keep(s.trickle(request)),
            ("DELETE", Some(s)) => Dispatched {
                response: HttpResponse::new(Status::Ok),
                ended_slot: Some(s.slot),
            },
            ("PATCH" | "DELETE", None) => Dispatched::keep(HttpResponse::new(Status::NotFound)),
            _ => Dispatched::keep(HttpResponse::new(Status::MethodNotAllowed)),
        }
    }

    async fn accept_publisher(
        &self,
        request: &HttpRequest,
        sessions: &mut [Option<Session>],
        datagrams: &mpsc::Sender<ReceivedDatagram>,
    ) -> HttpResponse {
        let Some(slot) = sessions.iter().position(Option::is_none) else {
            g2g_warn!(
                self,
                "publisher refused, all {} slot(s) taken (max-publishers)",
                sessions.len()
            );
            return HttpResponse::new(Status::Conflict);
        };
        if !has_content_type(request, SDP_CONTENT_TYPE) {
            return HttpResponse::new(Status::UnsupportedMediaType);
        }
        let offer = match SdpOffer::from_sdp_string(&request.body) {
            Ok(offer) => offer,
            Err(e) => {
                g2g_warn!(self, "unparseable offer: {e:?}");
                return HttpResponse::text(Status::BadRequest, format!("bad offer: {e:?}"));
            }
        };
        let socket = match UdpSocket::bind((select_host_ip(), 0)).await {
            Ok(socket) => socket,
            Err(e) => return HttpResponse::text(Status::InternalServerError, e.to_string()),
        };
        let Ok(local) = socket.local_addr() else {
            return HttpResponse::new(Status::InternalServerError);
        };
        let config = RtcConfig::new()
            .set_crypto_provider(Arc::new(from_feature_flags()))
            .set_stats_interval(Some(STATS_INTERVAL))
            .clear_codecs();
        let mut rtc = enable_tracks(config, [Track::Video(self.video_codec), Track::Audio])
            .build(Instant::now());
        // candidates added before the answer ride inside it
        if add_ice_candidates(&mut rtc, &socket, self.stun_server.as_deref())
            .await
            .is_err()
        {
            return HttpResponse::new(Status::InternalServerError);
        }
        let answer = match rtc.sdp_api().accept_offer(offer) {
            Ok(answer) => answer,
            Err(e) => {
                g2g_warn!(self, "offer rejected: {e:?}");
                return HttpResponse::text(Status::BadRequest, format!("offer rejected: {e:?}"));
            }
        };
        let resource_id = random_hex_id();
        g2g_info!(
            self,
            "publisher accepted on slot {slot}, resource {resource_id}, rtc socket {local}"
        );
        let socket = Arc::new(socket);
        let response = HttpResponse::new(Status::Created)
            .with_header("Location", format!("{RESOURCE_PATH_PREFIX}{resource_id}"))
            .with_header("ETag", entity_tag(&resource_id))
            .with_body(SDP_CONTENT_TYPE, answer.to_sdp_string());
        sessions[slot] = Some(Session {
            slot,
            rtc,
            _receiver: spawn_receiver(socket.clone(), slot, local, datagrams.clone()),
            socket,
            local,
            remote_ufrag: parse_restart_answer_frag(&request.body)
                .0
                .map(|creds| creds.ufrag),
            offer_sdp: request.body.clone(),
            resource_id,
            video_mid: None,
            audio_mid: None,
            video_keyframed: false,
            last_pli: Instant::now(),
        });
        response
    }
}

static WHIPSERVERSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "host-addr",
        PropKind::Str,
        "http URL whose host:port the WHIP server binds",
    )
    .with_default(DEFAULT_HOST_ADDR),
    PropertySpec::new(
        "auth-token",
        PropKind::Str,
        "bearer token the publisher must send (empty = no auth)",
    ),
    PropertySpec::new(
        "stun-server",
        PropKind::Str,
        "STUN server host:port for a server-reflexive candidate (empty = host only)",
    ),
    PropertySpec::new(
        "tls-cert-file",
        PropKind::Str,
        "PEM certificate chain served when host-addr is https (needs tls-key-file)",
    ),
    PropertySpec::new(
        "tls-key-file",
        PropKind::Str,
        "PEM PKCS#8 private key matching tls-cert-file",
    ),
    VIDEO_CODEC_PROP,
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "access units across every pad then EOS (-1 = unbounded)",
    )
    .with_default("-1")
    .with_range("-1", "9223372036854775807"),
    PropertySpec::new(
        "max-publishers",
        PropKind::Uint,
        "publishers accepted at once, each on its own video + audio pad pair (slot i = outputs 2i, 2i+1); the run ends when the last one leaves",
    )
    .with_default("1")
    .with_range("1", "127"),
];

impl MultiOutputSource for WhipServerSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn output_count(&self) -> usize {
        self.max_publishers * PORTS_PER_PUBLISHER
    }

    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        if output >= self.output_count() {
            return Err(G2gError::CapsMismatch);
        }
        Ok(match output % PORTS_PER_PUBLISHER {
            VIDEO_PORT => self.video_codec.caps_placeholder(),
            _ => opus_caps(STEREO),
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        WHIPSERVERSRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if name == "num-buffers" {
            return crate::numbuffers::set_num_buffers(&mut self.frame_limit, &value);
        }
        if name == "video-codec" {
            return set_video_codec_property(&mut self.video_codec, &value);
        }
        if name == "max-publishers" {
            let n = value.as_uint().ok_or(PropError::Type)? as usize;
            if !(1..=MAX_PUBLISHERS).contains(&n) {
                return Err(PropError::Value);
            }
            self.set_max_publishers(n);
            return Ok(());
        }
        let v = value.as_str().ok_or(PropError::Type)?;
        let optional = || (!v.is_empty()).then(|| v.into());
        match name {
            "host-addr" => {
                bind_target(v).ok_or(PropError::Value)?;
                self.host_addr = v.into();
            }
            "auth-token" => self.bearer = optional(),
            "stun-server" => self.stun_server = optional(),
            "tls-cert-file" => self.tls_cert_file = optional(),
            "tls-key-file" => self.tls_key_file = optional(),
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "host-addr" => Some(PropValue::Str(self.host_addr.clone())),
            "auth-token" => Some(PropValue::Str(self.bearer.clone().unwrap_or_default())),
            "stun-server" => Some(PropValue::Str(self.stun_server.clone().unwrap_or_default())),
            "tls-cert-file" => Some(PropValue::Str(
                self.tls_cert_file.clone().unwrap_or_default(),
            )),
            "tls-key-file" => Some(PropValue::Str(
                self.tls_key_file.clone().unwrap_or_default(),
            )),
            "video-codec" => Some(get_video_codec_property(self.video_codec)),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.frame_limit)),
            "max-publishers" => Some(PropValue::Uint(self.max_publishers as u64)),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let output_count = self.output_count();
            if crate::numbuffers::finished_at_zero_limit_multi(self.frame_limit, output_count, out)
                .await?
            {
                return Ok(0);
            }
            let hw = || G2gError::Hardware(HardwareError::Other);
            let Some(target) = bind_target(&self.host_addr) else {
                g2g_warn!(self, "host-addr {} is not an http(s) URL", self.host_addr);
                return Err(hw());
            };
            let tls = match self.tls_acceptor(target.tls) {
                Ok(tls) => tls,
                Err(reason) => {
                    g2g_warn!(self, "{reason}");
                    return Err(hw());
                }
            };
            let listener = TcpListener::bind(target.authority.as_str())
                .await
                .map_err(io_err)?;
            g2g_info!(
                self,
                "listening on {} (tls {}), waiting for up to {} WHIP publisher(s)",
                target.authority,
                if tls.is_some() { "on" } else { "off" },
                self.max_publishers
            );
            let (request_tx, mut requests) = mpsc::channel(REQUEST_QUEUE_DEPTH);
            let _accept_task =
                AbortOnDrop(tokio::spawn(accept_connections(listener, tls, request_tx)));

            let video_codec = self.video_codec;
            for port in 0..output_count {
                let caps = match port % PORTS_PER_PUBLISHER {
                    VIDEO_PORT => video_codec.caps_placeholder(),
                    _ => opus_caps(STEREO),
                };
                out.push_to(port, PipelinePacket::CapsChanged(caps)).await?;
            }

            let mut sessions: Vec<Option<Session>> =
                (0..self.max_publishers).map(|_| None).collect();
            let (datagram_tx, mut datagrams) = mpsc::channel(DATAGRAM_QUEUE_DEPTH);
            let mut turn = TurnSet::empty();
            let mut seq = 0u64;
            macro_rules! finish {
                () => {{
                    for port in 0..output_count {
                        out.push_to(port, PipelinePacket::Eos).await?;
                    }
                    return Ok(seq);
                }};
            }
            // a freed slot keeps its pads for the next publisher; the run ends
            // with the last connected publisher
            macro_rules! release {
                ($slot:expr, $reason:expr) => {{
                    let slot: usize = $slot;
                    sessions[slot] = None;
                    g2g_info!(self, "publisher on slot {slot} left ({})", $reason);
                    if sessions.iter().all(Option::is_none) {
                        g2g_info!(self, "last publisher left, ending run");
                        finish!();
                    }
                }};
            }
            // a closed request channel means the accept task died
            macro_rules! answer {
                ($pending:expr) => {{
                    let Some(pending) = $pending else {
                        return Err(hw());
                    };
                    let dispatched = self
                        .dispatch(&pending.request, &mut sessions, &datagram_tx)
                        .await;
                    g2g_debug!(
                        self,
                        "{} {} -> {}",
                        pending.request.method,
                        pending.request.path,
                        dispatched.response.status.code_and_reason().0
                    );
                    let _ = pending.respond.send(dispatched.response);
                    if let Some(slot) = dispatched.ended_slot {
                        release!(slot, "delete");
                    }
                }};
            }

            loop {
                if sessions.iter().all(Option::is_none) {
                    answer!(requests.recv().await);
                    continue;
                }

                let mut frames: Vec<(usize, u64, Vec<u8>)> = Vec::new();
                let mut ended: Vec<(usize, &'static str)> = Vec::new();
                let mut deadline: Option<Instant> = None;
                for s in sessions.iter_mut().flatten() {
                    let slot = s.slot;
                    let first_port = slot * PORTS_PER_PUBLISHER;
                    let session_deadline = loop {
                        match s.rtc.poll_output() {
                            Ok(Output::Timeout(t)) => break t,
                            Ok(Output::Transmit(t)) => {
                                send_transmit(&s.socket, &mut turn, &t).await
                            }
                            Ok(Output::Event(Event::MediaAdded(m))) => {
                                g2g_debug!(
                                    self,
                                    "slot {slot}: {:?} track on mid {:?}",
                                    m.kind,
                                    m.mid
                                );
                                match m.kind {
                                    MediaKind::Video if s.video_mid.is_none() => {
                                        s.video_mid = Some(m.mid)
                                    }
                                    MediaKind::Audio if s.audio_mid.is_none() => {
                                        s.audio_mid = Some(m.mid)
                                    }
                                    _ => {}
                                }
                            }
                            Ok(Output::Event(Event::MediaData(d))) => {
                                let pts_ns = media_time_ns(d.time);
                                if Some(d.mid) == s.video_mid {
                                    if !s.video_keyframed {
                                        if !video_codec.is_keyframe(&d.data) {
                                            continue;
                                        }
                                        g2g_debug!(
                                            self,
                                            "slot {slot}: first video keyframe, forwarding"
                                        );
                                        s.video_keyframed = true;
                                    }
                                    frames.push((
                                        first_port + VIDEO_PORT,
                                        pts_ns,
                                        video_codec.pipeline_unit(&d.data),
                                    ));
                                } else if Some(d.mid) == s.audio_mid {
                                    frames.push((first_port + AUDIO_PORT, pts_ns, d.data.to_vec()));
                                }
                            }
                            Ok(Output::Event(Event::IceConnectionStateChange(
                                IceConnectionState::Disconnected,
                            ))) => {
                                ended.push((slot, "ice disconnected"));
                                break Instant::now();
                            }
                            Ok(Output::Event(Event::IceConnectionStateChange(state))) => {
                                g2g_debug!(self, "slot {slot}: ice {state:?}")
                            }
                            Ok(Output::Event(event)) => {
                                if let Event::PeerStats(p) = &event {
                                    g2g_debug!(
                                        self,
                                        "slot {slot}: peer rx {} B tx {} B rtt {:?} loss {:?}",
                                        p.peer_bytes_rx,
                                        p.peer_bytes_tx,
                                        p.rtt,
                                        p.ingress_loss_fraction
                                    );
                                }
                                self.stats[slot].record(&event);
                            }
                            Err(e) => {
                                g2g_warn!(self, "slot {slot}: rtc failed: {e:?}");
                                ended.push((slot, "rtc failed"));
                                break Instant::now();
                            }
                        }
                    };
                    deadline = Some(deadline.map_or(session_deadline, |d| d.min(session_deadline)));

                    if let Some(mid) = s.video_mid {
                        if !s.video_keyframed && s.last_pli.elapsed() >= PLI_INTERVAL {
                            s.last_pli = Instant::now();
                            if let Some(rx) = s.rtc.direct_api().stream_rx_by_mid(mid, None) {
                                g2g_debug!(self, "slot {slot}: no keyframe yet, sending pli");
                                rx.request_keyframe(KeyframeRequestKind::Pli);
                            }
                        }
                    }
                }
                for (slot, reason) in ended {
                    release!(slot, reason);
                }

                for (port, pts_ns, data) in frames {
                    let keyframe =
                        port % PORTS_PER_PUBLISHER == VIDEO_PORT && video_codec.is_keyframe(&data);
                    let frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(
                            data.into_boxed_slice(),
                        )),
                        timing: FrameTiming {
                            pts_ns,
                            dts_ns: pts_ns,
                            duration_ns: 0,
                            capture_ns: pts_ns,
                            arrival_ns: g2g_core::metrics::monotonic_ns(),
                            keyframe,
                        },
                        sequence: seq,
                        meta: Default::default(),
                    };
                    out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
                    seq += 1;
                    if seq >= self.frame_limit {
                        finish!();
                    }
                }

                let timeout = deadline
                    .unwrap_or_else(Instant::now)
                    .saturating_duration_since(Instant::now());
                tokio::select! {
                    received = datagrams.recv() => {
                        let Some(received) = received else {
                            return Err(hw());
                        };
                        // a datagram from a slot's earlier socket is dropped by the local check
                        let Some(s) = sessions[received.slot].as_mut().filter(|s| s.local == received.local) else {
                            continue;
                        };
                        match received.result {
                            Ok((bytes, source)) => {
                                let _ = feed_datagram(&mut s.rtc, &mut turn, s.local, &bytes, source);
                            }
                            Err(e) => {
                                g2g_warn!(self, "slot {}: rtc socket recv failed: {e}", received.slot);
                                release!(received.slot, "socket recv failed");
                            }
                        }
                    }
                    pending = requests.recv() => answer!(pending),
                    _ = tokio::time::sleep(timeout) => {
                        let now = Instant::now();
                        let mut failed = Vec::new();
                        for s in sessions.iter_mut().flatten() {
                            if s.rtc.handle_input(Input::Timeout(now)).is_err() {
                                failed.push(s.slot);
                            }
                        }
                        for slot in failed {
                            g2g_warn!(self, "slot {slot}: rtc timeout handling failed");
                            release!(slot, "rtc failed");
                        }
                    }
                }
            }
        })
    }
}

/// One datagram from a publisher slot's socket, or the recv error that ended
/// its receive task.
struct ReceivedDatagram {
    slot: usize,
    local: SocketAddr,
    result: std::io::Result<(Vec<u8>, SocketAddr)>,
}

/// Read the slot's socket on its own task into the run loop's one channel, so
/// the loop selects over any number of publishers.
fn spawn_receiver(
    socket: Arc<UdpSocket>,
    slot: usize,
    local: SocketAddr,
    datagrams: mpsc::Sender<ReceivedDatagram>,
) -> AbortOnDrop {
    AbortOnDrop(tokio::spawn(async move {
        let mut buf = alloc::vec![0u8; UDP_RECV_BYTES];
        loop {
            let result = socket
                .recv_from(&mut buf)
                .await
                .map(|(n, source)| (buf[..n].to_vec(), source));
            let failed = result.is_err();
            let delivered = datagrams
                .send(ReceivedDatagram {
                    slot,
                    local,
                    result,
                })
                .await
                .is_ok();
            if failed || !delivered {
                return;
            }
        }
    }))
}

/// One publisher's PeerConnection, on its slot's pad pair.
struct Session {
    slot: usize,
    rtc: Rtc,
    socket: Arc<UdpSocket>,
    local: SocketAddr,
    _receiver: AbortOnDrop,
    resource_id: String,
    remote_ufrag: Option<String>,
    offer_sdp: String,
    video_mid: Option<Mid>,
    audio_mid: Option<Mid>,
    video_keyframed: bool,
    last_pli: Instant,
}

impl Session {
    fn trickle(&mut self, request: &HttpRequest) -> HttpResponse {
        if !has_content_type(request, TRICKLE_CONTENT_TYPE) {
            return HttpResponse::new(Status::UnsupportedMediaType);
        }
        let current_tag = entity_tag(&self.resource_id);
        let stale_tag = request
            .if_match
            .as_deref()
            .is_some_and(|tag| tag != IF_MATCH_ANY && tag != current_tag);
        if stale_tag {
            return HttpResponse::new(Status::PreconditionFailed);
        }
        let (creds, candidates) = parse_restart_answer_frag(&request.body);
        let restart = creds.filter(|c| Some(&c.ufrag) != self.remote_ufrag.as_ref());
        if let Some(creds) = restart {
            return self.restart_ice(creds, &candidates);
        }
        for candidate in candidates {
            self.rtc.add_remote_candidate(candidate);
        }
        HttpResponse::new(Status::NoContent)
    }

    /// RFC 9725 ICE restart: the publisher's new credentials and candidates
    /// arrive as a frag, so re-run the original offer with them swapped in
    /// (str0m restarts its agent when the remote credentials change) and
    /// answer with our new credentials and candidates.
    fn restart_ice(&mut self, creds: IceCreds, candidates: &[Candidate]) -> HttpResponse {
        let offer_sdp = restart_offer(&self.offer_sdp, &creds, candidates);
        let Ok(offer) = SdpOffer::from_sdp_string(&offer_sdp) else {
            return HttpResponse::new(Status::BadRequest);
        };
        let answer = match self.rtc.sdp_api().accept_offer(offer) {
            Ok(answer) => answer,
            Err(e) => {
                return HttpResponse::text(
                    Status::InternalServerError,
                    format!("ice restart rejected: {e:?}"),
                )
            }
        };
        let local = self.rtc.direct_api().local_ice_credentials();
        let mut frag = format!(
            "a=ice-ufrag:{}\r\na=ice-pwd:{}\r\n",
            local.ufrag, local.pass
        );
        for line in sdp_lines(&self.offer_sdp) {
            if line.starts_with("m=") || line.starts_with("a=mid:") {
                frag.push_str(line);
                frag.push_str("\r\n");
            }
            if line.starts_with("a=mid:") {
                break;
            }
        }
        let answer_sdp = answer.to_sdp_string();
        for line in sdp_lines(&answer_sdp).filter(|l| l.starts_with("a=candidate:")) {
            frag.push_str(line);
            frag.push_str("\r\n");
        }
        self.offer_sdp = offer_sdp;
        self.remote_ufrag = Some(creds.ufrag);
        HttpResponse::new(Status::Ok).with_body(TRICKLE_CONTENT_TYPE, frag)
    }
}

fn sdp_lines(sdp: &str) -> impl Iterator<Item = &str> {
    sdp.lines().map(str::trim_end)
}

/// The publisher's offer with `creds` in every ice-ufrag / ice-pwd line, the old
/// candidates dropped, and `candidates` under the first m-line.
fn restart_offer(offer_sdp: &str, creds: &IceCreds, candidates: &[Candidate]) -> String {
    let mut out = String::new();
    let mut inserted = false;
    for line in sdp_lines(offer_sdp) {
        if line.starts_with("a=candidate:") || line == "a=end-of-candidates" {
            continue;
        }
        if line.starts_with("a=ice-ufrag:") {
            out.push_str(&format!("a=ice-ufrag:{}\r\n", creds.ufrag));
        } else if line.starts_with("a=ice-pwd:") {
            out.push_str(&format!("a=ice-pwd:{}\r\n", creds.pass));
        } else {
            out.push_str(line);
            out.push_str("\r\n");
        }
        if !inserted && line.starts_with("a=mid:") {
            for candidate in candidates {
                out.push_str(&format!("a={}\r\n", candidate.to_sdp_string()));
            }
            inserted = true;
        }
    }
    out
}

impl LogSource for WhipServerSrc {
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

struct Dispatched {
    response: HttpResponse,
    /// The slot a `DELETE` ended.
    ended_slot: Option<usize>,
}

impl Dispatched {
    fn keep(response: HttpResponse) -> Self {
        Self {
            response,
            ended_slot: None,
        }
    }
}

struct AbortOnDrop(JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[derive(Debug, PartialEq, Eq)]
struct BindTarget {
    authority: String,
    tls: bool,
}

/// The `host:port` to bind from an `http://` or `https://` URL, the scheme's
/// default port when it has none.
fn bind_target(host_addr: &str) -> Option<BindTarget> {
    let (rest, tls, default_port) = match host_addr.strip_prefix(HTTPS_SCHEME) {
        Some(rest) => (rest, true, HTTPS_DEFAULT_PORT),
        None => (
            host_addr.strip_prefix(HTTP_SCHEME)?,
            false,
            HTTP_DEFAULT_PORT,
        ),
    };
    let authority = rest.split('/').next().unwrap_or_default();
    if authority.is_empty() {
        return None;
    }
    let has_port = authority
        .rsplit_once(':')
        .is_some_and(|(_, port)| !port.contains(']'));
    let authority = if has_port {
        authority.to_string()
    } else {
        format!("{authority}:{default_port}")
    };
    Some(BindTarget { authority, tls })
}

fn entity_tag(resource_id: &str) -> String {
    format!("\"{resource_id}\"")
}

fn has_content_type(request: &HttpRequest, expected: &str) -> bool {
    request
        .content_type
        .as_deref()
        .and_then(|value| value.split(';').next())
        .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case(expected))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Ok,
    Created,
    NoContent,
    BadRequest,
    Unauthorized,
    NotFound,
    MethodNotAllowed,
    Conflict,
    PreconditionFailed,
    PayloadTooLarge,
    UnsupportedMediaType,
    InternalServerError,
    ServiceUnavailable,
}

impl Status {
    fn code_and_reason(self) -> (u16, &'static str) {
        match self {
            Status::Ok => (200, "OK"),
            Status::Created => (201, "Created"),
            Status::NoContent => (204, "No Content"),
            Status::BadRequest => (400, "Bad Request"),
            Status::Unauthorized => (401, "Unauthorized"),
            Status::NotFound => (404, "Not Found"),
            Status::MethodNotAllowed => (405, "Method Not Allowed"),
            Status::Conflict => (409, "Conflict"),
            Status::PreconditionFailed => (412, "Precondition Failed"),
            Status::PayloadTooLarge => (413, "Payload Too Large"),
            Status::UnsupportedMediaType => (415, "Unsupported Media Type"),
            Status::InternalServerError => (500, "Internal Server Error"),
            Status::ServiceUnavailable => (503, "Service Unavailable"),
        }
    }
}

#[derive(Debug)]
struct HttpResponse {
    status: Status,
    headers: Vec<(&'static str, String)>,
    content_type: Option<&'static str>,
    body: String,
}

impl HttpResponse {
    fn new(status: Status) -> Self {
        Self {
            status,
            headers: Vec::new(),
            content_type: None,
            body: String::new(),
        }
    }

    fn text(status: Status, body: String) -> Self {
        Self::new(status).with_body(TEXT_CONTENT_TYPE, body)
    }

    fn with_header(mut self, name: &'static str, value: String) -> Self {
        self.headers.push((name, value));
        self
    }

    fn with_body(mut self, content_type: &'static str, body: String) -> Self {
        self.content_type = Some(content_type);
        self.body = body;
        self
    }

    fn to_bytes(&self) -> Vec<u8> {
        let (code, reason) = self.status.code_and_reason();
        let mut head = format!("HTTP/1.1 {code} {reason}\r\n");
        let content_type = self.content_type.map(|value| ("Content-Type", value));
        let headers = self
            .headers
            .iter()
            .map(|(name, value)| (*name, value.as_str()))
            .chain(content_type)
            .chain(CORS_HEADERS.iter().copied());
        for (name, value) in headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        head.push_str(&format!(
            "Content-Length: {}\r\nConnection: close\r\n\r\n",
            self.body.len()
        ));
        let mut bytes = head.into_bytes();
        bytes.extend_from_slice(self.body.as_bytes());
        bytes
    }
}

#[derive(Debug, Default)]
struct HttpRequest {
    method: String,
    path: String,
    authorization: Option<String>,
    content_type: Option<String>,
    if_match: Option<String>,
    body: String,
}

/// A request handed to the run loop, which owns the session and answers it.
struct PendingRequest {
    request: HttpRequest,
    respond: oneshot::Sender<HttpResponse>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestError {
    Malformed,
    TooLarge,
}

impl RequestError {
    fn response(self) -> HttpResponse {
        match self {
            RequestError::Malformed => HttpResponse::new(Status::BadRequest),
            RequestError::TooLarge => HttpResponse::new(Status::PayloadTooLarge),
        }
    }
}

/// One complete request from `buf`, `None` until the head and body have arrived.
fn parse_request(buf: &[u8]) -> Result<Option<HttpRequest>, RequestError> {
    let head_end = buf
        .windows(HEAD_TERMINATOR.len())
        .position(|window| window == HEAD_TERMINATOR);
    let Some(head_end) = head_end else {
        if buf.len() > MAX_HEAD_BYTES {
            return Err(RequestError::TooLarge);
        }
        return Ok(None);
    };
    if head_end > MAX_HEAD_BYTES {
        return Err(RequestError::TooLarge);
    }
    let head = core::str::from_utf8(&buf[..head_end]).map_err(|_| RequestError::Malformed)?;
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next().unwrap_or_default().split(' ');
    let (Some(method), Some(path)) = (request_line.next(), request_line.next()) else {
        return Err(RequestError::Malformed);
    };
    let mut request = HttpRequest {
        method: method.to_string(),
        path: path.to_string(),
        ..HttpRequest::default()
    };
    let mut content_length = 0usize;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        if name.eq_ignore_ascii_case("Content-Length") {
            content_length = value.parse().map_err(|_| RequestError::Malformed)?;
        } else if name.eq_ignore_ascii_case("Authorization") {
            request.authorization = Some(value.to_string());
        } else if name.eq_ignore_ascii_case("Content-Type") {
            request.content_type = Some(value.to_string());
        } else if name.eq_ignore_ascii_case("If-Match") {
            request.if_match = Some(value.to_string());
        }
    }
    if content_length > MAX_BODY_BYTES {
        return Err(RequestError::TooLarge);
    }
    let body_start = head_end.saturating_add(HEAD_TERMINATOR.len());
    let body_end = body_start.saturating_add(content_length);
    let Some(body) = buf.get(body_start..body_end) else {
        return Ok(None);
    };
    request.body = core::str::from_utf8(body)
        .map_err(|_| RequestError::Malformed)?
        .to_string();
    Ok(Some(request))
}

async fn read_request<S: AsyncRead + Unpin>(stream: &mut S) -> Result<HttpRequest, RequestError> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; READ_CHUNK_BYTES];
    loop {
        if let Some(request) = parse_request(&buf)? {
            return Ok(request);
        }
        let n = stream
            .read(&mut chunk)
            .await
            .map_err(|_| RequestError::Malformed)?;
        if n == 0 {
            return Err(RequestError::Malformed);
        }
        buf.extend_from_slice(&chunk[..n]);
    }
}

/// Answer CORS preflight here, send everything else to the run loop.
async fn answer_request(
    request: HttpRequest,
    requests: &mpsc::Sender<PendingRequest>,
) -> HttpResponse {
    if request.method == "OPTIONS" {
        return HttpResponse::new(Status::NoContent);
    }
    let (respond, response) = oneshot::channel();
    if requests
        .send(PendingRequest { request, respond })
        .await
        .is_err()
    {
        return HttpResponse::new(Status::ServiceUnavailable);
    }
    response
        .await
        .unwrap_or_else(|_| HttpResponse::new(Status::ServiceUnavailable))
}

async fn serve_connection<S: AsyncRead + AsyncWrite + Unpin>(
    mut stream: S,
    requests: mpsc::Sender<PendingRequest>,
) {
    let response = match read_request(&mut stream).await {
        Ok(request) => answer_request(request, &requests).await,
        Err(e) => e.response(),
    };
    let _ = stream.write_all(&response.to_bytes()).await;
    let _ = stream.shutdown().await;
}

async fn accept_connections(
    listener: TcpListener,
    tls: Option<TlsAcceptor>,
    requests: mpsc::Sender<PendingRequest>,
) {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let requests = requests.clone();
                match tls.clone() {
                    Some(acceptor) => {
                        tokio::spawn(async move {
                            // a failed handshake is the client's problem, nothing to answer
                            if let Ok(stream) = acceptor.accept(stream).await {
                                serve_connection(stream, requests).await;
                            }
                        });
                    }
                    None => {
                        tokio::spawn(serve_connection(stream, requests));
                    }
                }
            }
            Err(e) => {
                g2g_warn!(
                    &Target::category(short_type_name::<WhipServerSrc>()),
                    "accept failed: {e}"
                );
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::webrtc_util::add_host_candidate;
    use g2g_core::{AudioFormat, VideoCodec};
    use str0m::change::{SdpAnswer, SdpPendingOffer};
    use str0m::media::Direction;

    fn request(method: &str, path: &str) -> HttpRequest {
        HttpRequest {
            method: method.into(),
            path: path.into(),
            ..HttpRequest::default()
        }
    }

    fn response_text(response: &HttpResponse) -> String {
        String::from_utf8(response.to_bytes()).unwrap()
    }

    #[test]
    fn outputs_follow_the_video_codec() {
        let src = WhipServerSrc::new(DEFAULT_HOST_ADDR).with_video_codec(WebRtcVideoCodec::Vp9);
        assert!(matches!(
            src.output_caps(VIDEO_PORT),
            Ok(Caps::CompressedVideo {
                codec: VideoCodec::Vp9,
                ..
            })
        ));
        assert!(matches!(
            src.output_caps(AUDIO_PORT),
            Ok(Caps::Audio {
                format: AudioFormat::Opus,
                ..
            })
        ));
        assert!(src.output_caps(PORTS_PER_PUBLISHER).is_err());
    }

    #[test]
    fn max_publishers_adds_a_pad_pair_per_slot() {
        let mut src = WhipServerSrc::new(DEFAULT_HOST_ADDR).with_max_publishers(3);
        assert_eq!(src.output_count(), 3 * PORTS_PER_PUBLISHER);
        assert_eq!(src.publisher_stats_handles().len(), 3);
        assert!(matches!(
            src.output_caps(2 * PORTS_PER_PUBLISHER + VIDEO_PORT),
            Ok(Caps::CompressedVideo { .. })
        ));
        assert!(matches!(
            src.output_caps(2 * PORTS_PER_PUBLISHER + AUDIO_PORT),
            Ok(Caps::Audio {
                format: AudioFormat::Opus,
                ..
            })
        ));
        assert!(src.output_caps(3 * PORTS_PER_PUBLISHER).is_err());
        src.set_property("max-publishers", PropValue::Uint(2))
            .unwrap();
        assert_eq!(src.get_property("max-publishers"), Some(PropValue::Uint(2)));
        assert_eq!(src.output_count(), 2 * PORTS_PER_PUBLISHER);
        assert_eq!(
            src.set_property("max-publishers", PropValue::Uint(0)),
            Err(PropError::Value)
        );
        assert_eq!(
            src.set_property("max-publishers", PropValue::Uint(MAX_PUBLISHERS as u64 + 1)),
            Err(PropError::Value)
        );
        let spec = WHIPSERVERSRC_PROPS
            .iter()
            .find(|p| p.name == "max-publishers")
            .unwrap();
        assert_eq!(spec.range, Some(("1", "127")));
        assert_eq!(MAX_PUBLISHERS, 127);
    }

    #[test]
    fn properties_round_trip() {
        let mut src = WhipServerSrc::new(DEFAULT_HOST_ADDR);
        for (name, value) in [
            ("host-addr", "http://0.0.0.0:8190"),
            ("auth-token", "secret"),
            ("stun-server", "stun.example.com:3478"),
            ("tls-cert-file", "/etc/g2g/cert.pem"),
            ("tls-key-file", "/etc/g2g/key.pem"),
            ("video-codec", "av1"),
        ] {
            src.set_property(name, PropValue::Str(value.into()))
                .unwrap();
            assert_eq!(src.get_property(name), Some(PropValue::Str(value.into())));
        }
        src.set_property("num-buffers", PropValue::Int(12)).unwrap();
        assert_eq!(src.get_property("num-buffers"), Some(PropValue::Int(12)));
        assert_eq!(
            src.set_property("video-codec", PropValue::Str("theora".into())),
            Err(PropError::Value)
        );
        assert_eq!(
            src.set_property("host-addr", PropValue::Str("rtsp://h:1".into())),
            Err(PropError::Value)
        );
        src.set_property("auth-token", PropValue::Str(String::new()))
            .unwrap();
        assert_eq!(src.bearer, None);
        for spec in src.properties() {
            assert!(src.get_property(spec.name).is_some(), "{}", spec.name);
        }
    }

    #[test]
    fn launch_line_sets_the_properties() {
        use g2g_core::graph::NodeKind;
        use g2g_core::runtime::parse_launch;

        let registry = crate::registry::default_registry();
        let graph = parse_launch(
            &registry,
            "whipserversrc name=s host-addr=http://0.0.0.0:8190 auth-token=t video-codec=vp8  s. ! fakesink  s. ! fakesink",
        )
        .expect("launch line parses")
        .finish()
        .expect("valid graph");
        let fanouts: Vec<NodeKind> = graph
            .topo()
            .iter()
            .map(|&n| graph.kind(n))
            .filter(|k| matches!(k, NodeKind::FanoutSrc(_)))
            .collect();
        assert_eq!(fanouts, [NodeKind::FanoutSrc(PORTS_PER_PUBLISHER as u8)]);
    }

    #[test]
    fn host_addr_gives_the_bind_address() {
        let plain = |authority: &str| {
            Some(BindTarget {
                authority: authority.into(),
                tls: false,
            })
        };
        assert_eq!(bind_target("http://0.0.0.0:8190"), plain("0.0.0.0:8190"));
        assert_eq!(bind_target("http://[::1]:8190/whip"), plain("[::1]:8190"));
        assert_eq!(bind_target("http://localhost"), plain("localhost:80"));
        assert_eq!(
            bind_target("https://localhost"),
            Some(BindTarget {
                authority: "localhost:443".into(),
                tls: true,
            })
        );
        assert_eq!(bind_target("rtsp://0.0.0.0:8190"), None);
        assert_eq!(bind_target("http://"), None);
    }

    #[test]
    fn parser_reads_head_and_body() {
        let raw = b"POST /whip HTTP/1.1\r\ncontent-LENGTH: 4\r\nCONTENT-TYPE: application/sdp\r\nauthorization: Bearer t\r\n\r\nv=0\n";
        let parsed = parse_request(raw).unwrap().unwrap();
        assert_eq!(parsed.method, "POST");
        assert_eq!(parsed.path, "/whip");
        assert_eq!(parsed.content_type.as_deref(), Some(SDP_CONTENT_TYPE));
        assert_eq!(parsed.authorization.as_deref(), Some("Bearer t"));
        assert_eq!(parsed.body, "v=0\n");
        assert!(parse_request(&raw[..raw.len() - 1]).unwrap().is_none());
        assert!(parse_request(b"POST /whip HTTP/1.1\r\n").unwrap().is_none());
    }

    #[test]
    fn parser_rejects_oversize_and_garbage() {
        let huge_body = format!(
            "POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY_BYTES + 1
        );
        assert_eq!(
            parse_request(huge_body.as_bytes()).unwrap_err(),
            RequestError::TooLarge
        );
        let endless_head = alloc::vec![b'a'; MAX_HEAD_BYTES + 1];
        assert_eq!(
            parse_request(&endless_head).unwrap_err(),
            RequestError::TooLarge
        );
        let bad_length = b"POST / HTTP/1.1\r\nContent-Length: lots\r\n\r\n";
        assert_eq!(
            parse_request(bad_length).unwrap_err(),
            RequestError::Malformed
        );
        let overflowing = format!("POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n", usize::MAX);
        assert_eq!(
            parse_request(overflowing.as_bytes()).unwrap_err(),
            RequestError::TooLarge
        );
    }

    #[tokio::test]
    async fn options_is_answered_with_cors_headers() {
        let (requests, _run_loop) = mpsc::channel(1);
        let response = answer_request(request("OPTIONS", "*"), &requests).await;
        assert_eq!(response.status, Status::NoContent);
        let text = response_text(&response);
        for (name, value) in CORS_HEADERS {
            assert!(text.contains(&format!("{name}: {value}\r\n")), "{name}");
        }
    }

    #[tokio::test]
    async fn dispatch_rejects_without_a_network() {
        let src = WhipServerSrc::new(DEFAULT_HOST_ADDR).with_bearer("secret");
        let mut sessions = [None];
        let (datagrams, _) = mpsc::channel(DATAGRAM_QUEUE_DEPTH);

        let mut wrong_bearer = request("POST", "/whip");
        wrong_bearer.authorization = Some("Bearer guess".into());
        wrong_bearer.content_type = Some(SDP_CONTENT_TYPE.into());
        let dispatched = src.dispatch(&wrong_bearer, &mut sessions, &datagrams).await;
        assert_eq!(dispatched.response.status, Status::Unauthorized);

        let authorized = |method: &str, path: &str| {
            let mut r = request(method, path);
            r.authorization = Some("Bearer secret".into());
            r
        };
        let mut not_sdp = authorized("POST", "/whip");
        not_sdp.content_type = Some("text/plain".into());
        let dispatched = src.dispatch(&not_sdp, &mut sessions, &datagrams).await;
        assert_eq!(dispatched.response.status, Status::UnsupportedMediaType);

        let mut garbage_sdp = authorized("POST", "/whip");
        garbage_sdp.content_type = Some("application/sdp; charset=utf-8".into());
        garbage_sdp.body = "not an offer".into();
        let dispatched = src.dispatch(&garbage_sdp, &mut sessions, &datagrams).await;
        assert_eq!(dispatched.response.status, Status::BadRequest);
        assert!(sessions[0].is_none());

        let unknown = format!("{RESOURCE_PATH_PREFIX}0123");
        for method in ["PATCH", "DELETE"] {
            let dispatched = src
                .dispatch(&authorized(method, &unknown), &mut sessions, &datagrams)
                .await;
            assert_eq!(dispatched.response.status, Status::NotFound, "{method}");
            assert!(dispatched.ended_slot.is_none());
        }
        let dispatched = src
            .dispatch(&authorized("GET", "/whip"), &mut sessions, &datagrams)
            .await;
        assert_eq!(dispatched.response.status, Status::MethodNotAllowed);
    }

    #[test]
    fn created_response_carries_location_and_etag() {
        let response = HttpResponse::new(Status::Created)
            .with_header("Location", format!("{RESOURCE_PATH_PREFIX}ab"))
            .with_header("ETag", entity_tag("ab"))
            .with_body(SDP_CONTENT_TYPE, "v=0\r\n".into());
        let text = response_text(&response);
        assert!(text.starts_with("HTTP/1.1 201 Created\r\n"));
        assert!(text.contains("Location: /whip/resource/ab\r\n"));
        assert!(text.contains("ETag: \"ab\"\r\n"));
        assert!(text.contains("Content-Type: application/sdp\r\n"));
        assert!(text.ends_with("Content-Length: 5\r\nConnection: close\r\n\r\nv=0\r\n"));
    }

    const TLS_CERT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/whip_tls_test_cert.pem"
    );
    const TLS_KEY: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/whip_tls_test_key.pem"
    );

    #[test]
    fn https_needs_both_pem_files() {
        let src = WhipServerSrc::new("https://127.0.0.1:1");
        assert!(src.tls_acceptor(false).unwrap().is_none());
        assert!(src.tls_acceptor(true).is_err());
        assert!(src
            .clone_with_tls(TLS_CERT, "/nonexistent/key.pem")
            .tls_acceptor(true)
            .is_err());
        assert!(src
            .clone_with_tls(TLS_CERT, TLS_KEY)
            .tls_acceptor(true)
            .unwrap()
            .is_some());
    }

    impl WhipServerSrc {
        fn clone_with_tls(&self, cert: &str, key: &str) -> Self {
            WhipServerSrc::new(self.host_addr.clone()).with_tls(cert, key)
        }
    }

    /// A str0m publisher with one host candidate and a sendonly video m-line.
    async fn publisher_offer() -> (Rtc, SdpPendingOffer, String) {
        let config = RtcConfig::new()
            .set_crypto_provider(Arc::new(from_feature_flags()))
            .clear_codecs();
        let mut rtc =
            enable_tracks(config, [Track::Video(WebRtcVideoCodec::H264)]).build(Instant::now());
        let socket = UdpSocket::bind((select_host_ip(), 0)).await.unwrap();
        add_host_candidate(&mut rtc, &socket).unwrap();
        let mut api = rtc.sdp_api();
        api.add_media(MediaKind::Video, Direction::SendOnly, None, None, None);
        let (offer, pending) = api.apply().unwrap();
        (rtc, pending, offer.to_sdp_string())
    }

    fn ufrag_of(sdp: &str) -> String {
        parse_restart_answer_frag(sdp)
            .0
            .expect("ice credentials")
            .ufrag
    }

    fn trickle(resource: &str, body: String) -> HttpRequest {
        let mut patch = request("PATCH", resource);
        patch.content_type = Some(TRICKLE_CONTENT_TYPE.into());
        patch.body = body;
        patch
    }

    #[tokio::test]
    async fn ice_restart_patch_answers_with_new_server_credentials() {
        let src = WhipServerSrc::new(DEFAULT_HOST_ADDR);
        let mut sessions = [None];
        let (datagrams, _) = mpsc::channel(DATAGRAM_QUEUE_DEPTH);
        let (mut publisher, pending, offer_sdp) = publisher_offer().await;
        let mut post = request("POST", "/whip");
        post.content_type = Some(SDP_CONTENT_TYPE.into());
        post.body = offer_sdp;
        let created = src
            .dispatch(&post, &mut sessions, &datagrams)
            .await
            .response;
        assert_eq!(created.status, Status::Created, "{}", created.body);
        let first_ufrag = ufrag_of(&created.body);
        publisher
            .sdp_api()
            .accept_answer(pending, SdpAnswer::from_sdp_string(&created.body).unwrap())
            .unwrap();
        let resource = format!(
            "{RESOURCE_PATH_PREFIX}{}",
            sessions[0].as_ref().unwrap().resource_id
        );

        let (restart_sdp, creds) = {
            let mut api = publisher.sdp_api();
            let creds = api.ice_restart(true);
            let (offer, _pending) = api.apply().unwrap();
            (offer.to_sdp_string(), creds)
        };
        let mut frag = format!(
            "a=ice-ufrag:{}\r\na=ice-pwd:{}\r\n",
            creds.ufrag, creds.pass
        );
        for line in sdp_lines(&restart_sdp).filter(|l| l.starts_with("a=candidate:")) {
            frag.push_str(line);
            frag.push_str("\r\n");
        }
        let restarted = src
            .dispatch(&trickle(&resource, frag), &mut sessions, &datagrams)
            .await
            .response;
        assert_eq!(restarted.status, Status::Ok, "{}", restarted.body);
        assert_eq!(restarted.content_type, Some(TRICKLE_CONTENT_TYPE));
        let new_ufrag = ufrag_of(&restarted.body);
        assert_ne!(new_ufrag, first_ufrag);
        let s = sessions[0].as_mut().unwrap();
        assert_eq!(new_ufrag, s.rtc.direct_api().local_ice_credentials().ufrag);
        assert_eq!(s.remote_ufrag.as_deref(), Some(creds.ufrag.as_str()));
        assert!(
            restarted.body.contains("a=candidate:"),
            "{}",
            restarted.body
        );

        // the same credentials again are a plain trickle, not another restart
        let again = format!(
            "a=ice-ufrag:{}\r\na=ice-pwd:{}\r\n",
            creds.ufrag, creds.pass
        );
        let plain = src
            .dispatch(&trickle(&resource, again), &mut sessions, &datagrams)
            .await;
        assert_eq!(plain.response.status, Status::NoContent);
    }
}

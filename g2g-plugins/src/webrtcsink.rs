//! Native WebRTC egress sink (`WebRtcSink`): publishes a pipeline's encoded
//! video (H.264, VP8, VP9, H.265 or AV1) or Opus to a WHIP server over a WebRTC PeerConnection, built on the sans-IO
//! [`str0m`] stack (ICE / DTLS / SRTP / RTP). This is the native counterpart of
//! the browser-only data-channel [`crate::webrtcsrc::WebRtcSrc`], and the
//! WebRTC analog of `RtmpSink` (publish encoded media to a server endpoint).
//!
//! Shape, mirroring the other tokio network sinks (`UdpSink`, `SrtSink`):
//! - `configure_pipeline` accepts any carried video codec or Opus (`webrtc_codec`).
//! - On the first `DataFrame`, the sink performs the WHIP handshake: it builds a
//!   str0m `Rtc` offering a single send-only m-line in that codec, POSTs the SDP
//!   offer to the WHIP endpoint (reqwest, `application/sdp`), applies the
//!   answer, then spawns a background task that owns the `Rtc` + a tokio
//!   `UdpSocket` and runs str0m's `poll_output` / `handle_input` loop.
//! - Each subsequent access unit is handed to that task over a bounded channel;
//!   the task feeds it to str0m's writer for the codec, which packetizes + SRTP-encrypts
//!   and emits the UDP datagrams. The element itself never touches the `Rtc`
//!   (it lives on the task), so `WebRtcSink` is naturally `Send`.
//! - With `reconnect` set, a session the task loses (the server rejects the ICE
//!   restart, or the socket fails) is replaced: the task POSTs a new offer with
//!   backoff, drops access units meanwhile, then asks the encoder for a keyframe.
//!
//! str0m is sans-IO, so g2g owns the socket and the timer, exactly the contract
//! the `srt` / `rtspserver` modules already follow. The pure-Rust `rust-crypto`
//! backend is selected (no OpenSSL system dep). Behind the `webrtc` feature.
//!
//! Status: compile-validated against str0m 0.20. The live WHIP publish path
//! (real ICE host-candidate selection, the DTLS/SRTP handshake against a real
//! server, and browser playback) is owed an on-network validation, like the
//! other egress elements (`rtmpsink`); the sandbox blocks the required ports.
//!
//! Framing: the sink forwards each access unit as g2g delivers it
//! (Annex-B, the pipeline convention); str0m's packetizer splits NAL
//! units for RTP. Confirming the exact framing str0m expects is part of the
//! owed runtime validation.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use tokio::net::UdpSocket;
use tokio::sync::mpsc;

use str0m::bwe::{Bitrate, BweKind};
use str0m::change::SdpPendingOffer;
use str0m::crypto::from_feature_flags;
use str0m::media::{Direction, Mid, Pt};
use str0m::rtp::AbsCaptureTime;
use str0m::{Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_info, g2g_warn, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, HardwareError, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Reconfigure,
};

use crate::filesink::io_err;
use crate::reconnect::{
    ReconnectPolicy, ReconnectState, DEFAULT_INITIAL_BACKOFF_MS, DEFAULT_MAX_BACKOFF_MS,
    PUBLISHER_RECONNECT_PROP, RECONNECT_BACKOFF_MAX_PROP, RECONNECT_BACKOFF_PROP,
};
use crate::turn::{self, TurnSet};
use crate::webrtc_codec::{
    enable_tracks, frame_capture_time, sink_caps_alternatives, stamp_capture_time, Track,
    WebRtcVideoCodec,
};
use crate::webrtc_receive::UDP_RECV_BYTES;
use crate::webrtc_stats::{WebRtcStatsHandle, STATS_INTERVAL};
use crate::webrtc_util::{
    add_host_candidate, drain_pacer, feed_datagram, ice_restart, negotiate_over_http,
    select_host_ip, send_transmit, trickle_candidates, ResourceHandle, TricklePatch, TurnConfig,
    ICE_RESTART_TIMEOUT, STUN_SERVER_PROP, TURN_PASS_PROP, TURN_SERVER_PROP, TURN_USER_PROP,
};

/// Default bounded depth of the element->session media channel. Backpressures
/// the pipeline if the session task falls behind the encoder.
pub(crate) const DEFAULT_QUEUE_DEPTH: usize = 256;

/// Initial BWE estimate seeded into str0m before any feedback arrives. The
/// congestion controller adapts from here as TWCC/REMB reports come in.
pub(crate) const INITIAL_BITRATE_BPS: u64 = 2_000_000;

/// One encoded access unit handed from the element to the session task.
#[derive(Debug)]
pub(crate) struct MediaUnit {
    pub(crate) pts_ns: u64,
    pub(crate) capture_time: Option<AbsCaptureTime>,
    pub(crate) data: Vec<u8>,
}

/// WHIP-publishing WebRTC egress sink. See the module docs.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::webrtcsink::WebRtcSink;
///
/// let sink = WebRtcSink::new("http://localhost:8889/mystream/whip")
///     .with_bearer("secret-token")
///     .with_stun_server("stun.l.google.com:19302");
/// ```
pub struct WebRtcSink {
    settings: SessionSettings,
    queue_depth: usize,
    configured: bool,
    /// Set on the first frame, after the WHIP handshake spawns the session task.
    tx: Option<mpsc::Sender<MediaUnit>>,
    shared: SessionShared,
    /// Last estimate `take_bitrate` forwarded, so it only reports changes (it is
    /// polled every frame; the encoder need not see the same value repeatedly).
    last_bitrate_sent: u64,
    frames_sent: u64,
}

// what the first WHIP session and every reconnect are built from
#[derive(Debug, Clone)]
struct SessionSettings {
    whip_url: String,
    bearer: Option<String>,
    /// STUN server (`host:port`) for ICE NAT traversal toward a cloud SFU.
    /// `None` = host candidate only (LAN / self-hosted same network).
    stun_server: Option<String>,
    /// TURN relay (`host:port`) + long-term credentials for the NAT cases a
    /// server-reflexive candidate cannot punch through. `None` = no relay.
    turn_server: Option<String>,
    turn_user: String,
    turn_pass: String,
    /// The track, decided from the configured caps. Defaults to H.264 video
    /// until `configure_pipeline` runs.
    track: Track,
    reconnect: ReconnectPolicy,
    log_name: LogName,
}

// state the element and its session task both read or write
#[derive(Debug, Clone, Default)]
struct SessionShared {
    /// The WHIP resource URL (from the handshake), so the element can DELETE it
    /// synchronously on EOS (RFC 9725 teardown). The detached session task cannot
    /// reliably complete a DELETE on clean end, since the runtime may tear it down
    /// first; the element owns the clean-EOS teardown. Shared with the task, which
    /// tears down instead when the session ends first, so exactly one DELETE goes
    /// out either way. A reconnect replaces the URL with the new session's.
    resource: ResourceHandle,
    /// Set by the session task when the remote sends a PLI (keyframe request) or
    /// a reconnect opens a new session; read + cleared by `take_reconfigure`,
    /// which forwards a `Reconfigure::ForceKeyframe` up the reverse channel to the
    /// encoder. Shared because the str0m loop lives on a separate task.
    keyframe_requested: Arc<AtomicBool>,
    /// Latest congestion-control / BWE estimate (bits/second, 0 = none yet),
    /// written by the session task from `Event::EgressBitrateEstimate`. Shared
    /// with the task; `take_bitrate` forwards changes up the reverse channel.
    bitrate_estimate: Arc<AtomicU64>,
    stats: WebRtcStatsHandle,
}

impl LogSource for SessionSettings {
    fn log_category(&self) -> &'static str {
        short_type_name::<WebRtcSink>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

impl LogSource for WebRtcSink {
    fn log_category(&self) -> &'static str {
        self.settings.log_category()
    }
    fn log_instance(&self) -> Option<&str> {
        self.settings.log_instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.settings.log_category_override()
    }
}

impl core::fmt::Debug for WebRtcSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WebRtcSink")
            .field("whip_url", &self.settings.whip_url)
            .field("configured", &self.configured)
            .field("frames_sent", &self.frames_sent)
            .finish()
    }
}

impl WebRtcSink {
    /// Publish to the given WHIP endpoint URL (e.g.
    /// `http://localhost:8889/mystream/whip` on a mediamtx server).
    pub fn new(whip_url: impl Into<String>) -> Self {
        Self {
            settings: SessionSettings {
                whip_url: whip_url.into(),
                bearer: None,
                stun_server: None,
                turn_server: None,
                turn_user: String::new(),
                turn_pass: String::new(),
                track: Track::Video(WebRtcVideoCodec::default()),
                reconnect: ReconnectPolicy {
                    max_attempts: 0,
                    initial_backoff_ms: DEFAULT_INITIAL_BACKOFF_MS,
                    max_backoff_ms: DEFAULT_MAX_BACKOFF_MS,
                },
                log_name: LogName::new(),
            },
            queue_depth: DEFAULT_QUEUE_DEPTH,
            configured: false,
            tx: None,
            shared: SessionShared::default(),
            last_bitrate_sent: 0,
            frames_sent: 0,
        }
    }

    /// Live counters of the WHIP session, readable while it runs.
    pub fn stats_handle(&self) -> WebRtcStatsHandle {
        self.shared.stats.clone()
    }

    /// Attach a bearer token, sent as `Authorization: Bearer <token>` on the
    /// WHIP POST (some servers require it).
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.settings.bearer = Some(token.into());
        self
    }

    /// Override the bounded element->session media-channel depth.
    pub fn with_queue_depth(mut self, depth: usize) -> Self {
        self.queue_depth = depth.max(1);
        self
    }

    /// Set a STUN server (`host:port`, e.g. `stun.l.google.com:19302`) for ICE
    /// NAT traversal. Required to reach a cloud SFU (LiveKit Cloud, etc.) from
    /// behind NAT; unset means host candidate only (works on a LAN).
    pub fn with_stun_server(mut self, server: impl Into<String>) -> Self {
        self.settings.stun_server = Some(server.into());
        self
    }

    /// Set a TURN relay (`host:port`) with long-term credentials. The relay is
    /// the fallback for NAT/firewall situations a STUN server-reflexive
    /// candidate cannot traverse (symmetric NAT, restrictive networks); LiveKit
    /// Cloud and other SFUs sometimes require it. Compose with
    /// [`Self::with_stun_server`]: ICE picks the relay only if the direct paths
    /// fail.
    pub fn with_turn_server(
        mut self,
        server: impl Into<String>,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        self.settings.turn_server = Some(server.into());
        self.settings.turn_user = username.into();
        self.settings.turn_pass = password.into();
        self
    }

    /// Number of access units handed to the WebRTC session so far.
    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    /// Connect the first WHIP session and spawn the task that publishes over it.
    /// Runs on the first frame because it is async (the runner drives `process`
    /// inside a tokio runtime, as for the other network sinks).
    async fn start_publishing(&mut self) -> Result<(), G2gError> {
        let settings = Arc::new(self.settings.clone());
        let first = match connect(settings.clone(), self.shared.resource.clone()).await {
            Ok(connection) => Ok(connection),
            Err(error) if settings.reconnect.max_attempts == 0 => return Err(error),
            Err(error) => Err(format!("{error:?}")),
        };
        let (tx, rx) = mpsc::channel::<MediaUnit>(self.queue_depth);
        tokio::spawn(publish(settings, self.shared.clone(), first, rx));
        self.tx = Some(tx);
        Ok(())
    }
}

// one connected WHIP session, handed from `connect` to `run_session`
#[derive(Debug)]
struct Connection {
    rtc: Rtc,
    socket: UdpSocket,
    local: SocketAddr,
    mid: Mid,
    turn: TurnSet,
    etag: Option<String>,
}

/// Build the `Rtc`, do the WHIP offer/answer exchange and trickle the gathered
/// candidates.
async fn connect(
    settings: Arc<SessionSettings>,
    resource: ResourceHandle,
) -> Result<Connection, G2gError> {
    // ICE needs a routable host candidate; pick the route-local IP via the
    // UDP connect trick (no packet is sent, the OS just resolves the source
    // address for the route). Falls back to loopback for offline use.
    let host_ip = select_host_ip();
    let socket = UdpSocket::bind((host_ip, 0)).await.map_err(io_err)?;
    let local = socket.local_addr().map_err(io_err)?;

    // rust-crypto backend (pure Rust, no OpenSSL); offer only this track's
    // codec so the answer's payload type is unambiguous.
    let config = RtcConfig::new()
        .set_crypto_provider(Arc::new(from_feature_flags()))
        // Congestion control: str0m runs the TWCC/REMB estimator and emits
        // `EgressBitrateEstimate`; the sink relays it to the encoder (BWE).
        .enable_bwe(Some(Bitrate::bps(INITIAL_BITRATE_BPS)))
        .set_stats_interval(Some(STATS_INTERVAL))
        .clear_codecs();
    let config = enable_tracks(config, [settings.track]);
    let mut rtc = config.build(Instant::now());

    // Trickle ICE: the offer carries the host candidate only; reflexive /
    // relay candidates are gathered after the POST and trickled by PATCH.
    add_host_candidate(&mut rtc, &socket)?;

    // Offer a single send-only m-line for the configured track.
    let (offer_sdp, pending, mid): (String, SdpPendingOffer, Mid) = {
        let mut api = rtc.sdp_api();
        let mid = api.add_media(
            settings.track.media_kind(),
            Direction::SendOnly,
            None,
            None,
            None,
        );
        let (offer, pending) = api.apply().ok_or_else(|| {
            g2g_warn!(&*settings, "offer creation failed");
            G2gError::Hardware(HardwareError::Other)
        })?;
        (offer.to_sdp_string(), pending, mid)
    };

    // WHIP: POST the offer, receive the answer SDP + resource URL, apply it.
    let session = negotiate_over_http(
        &mut rtc,
        pending,
        &settings.whip_url,
        settings.bearer.as_deref(),
        &offer_sdp,
    )
    .await?;

    // Gather reflexive / relay candidates and trickle them to the resource.
    let turn: TurnSet = trickle_candidates(
        &mut rtc,
        &socket,
        &offer_sdp,
        &session,
        settings.bearer.as_deref(),
        settings.stun_server.as_deref(),
        TurnConfig {
            server: settings.turn_server.as_deref(),
            user: &settings.turn_user,
            pass: &settings.turn_pass,
        },
    )
    .await;

    resource.set(session.resource);
    Ok(Connection {
        rtc,
        socket,
        local,
        mid,
        turn,
        etag: session.etag,
    })
}

// publishes over `first`, then over each reconnected session
async fn publish(
    settings: Arc<SessionSettings>,
    shared: SessionShared,
    first: Result<Connection, String>,
    mut rx: mpsc::Receiver<MediaUnit>,
) {
    let mut next = first;
    loop {
        let connection = match next {
            Ok(connection) => connection,
            Err(loss) => match reconnect(&settings, &shared, &mut rx, loss).await {
                Some(connection) => connection,
                None => return,
            },
        };
        match run_session(connection, settings.clone(), shared.clone(), &mut rx).await {
            SessionEnd::ElementGone => return,
            SessionEnd::Lost(loss) => next = Err(loss),
        }
    }
}

// the budget of `reconnect` attempts starts again with each loss
async fn reconnect(
    settings: &Arc<SessionSettings>,
    shared: &SessionShared,
    rx: &mut mpsc::Receiver<MediaUnit>,
    mut loss: String,
) -> Option<Connection> {
    let mut retry = ReconnectState::new(&settings.reconnect);
    loop {
        let Some(delay_ms) = retry.next_delay_ms(&settings.reconnect) else {
            g2g_warn!(
                &**settings,
                "publishing to {} failed: {loss}, no reconnect attempt left",
                settings.whip_url
            );
            return None;
        };
        g2g_warn!(
            &**settings,
            "publishing to {} failed: {loss}, reconnect {}/{} in {delay_ms} ms",
            settings.whip_url,
            retry.attempt,
            settings.reconnect.max_attempts
        );
        dropping_units(rx, tokio::time::sleep(Duration::from_millis(delay_ms))).await?;
        match dropping_units(rx, connect(settings.clone(), shared.resource.clone())).await? {
            Ok(connection) => {
                g2g_info!(
                    &**settings,
                    "republishing to {} after {} attempts",
                    settings.whip_url,
                    retry.attempt
                );
                shared.keyframe_requested.store(true, Ordering::Relaxed);
                return Some(connection);
            }
            Err(error) => loss = format!("{error:?}"),
        }
    }
}

// runs `work` while dropping the element's units, `None` once the element is gone
async fn dropping_units<T>(
    rx: &mut mpsc::Receiver<MediaUnit>,
    work: impl Future<Output = T>,
) -> Option<T> {
    let mut work = core::pin::pin!(work);
    loop {
        tokio::select! {
            output = &mut work => return Some(output),
            unit = rx.recv() => {
                unit?;
            }
        }
    }
}

#[derive(Debug)]
enum SessionEnd {
    // the element dropped its sender
    ElementGone,
    // the server or the network ended the session, with the reason for the log
    Lost(String),
}

/// `WebRtcSink`'s settable properties: the WHIP endpoint URL and an optional
/// bearer token, so a `gst-launch` line can target a server without the builder.
static WEBRTCSINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new("location", PropKind::Str, "WHIP endpoint URL to publish to"),
    PropertySpec::new(
        "auth-token",
        PropKind::Str,
        "optional Authorization: Bearer token for the WHIP POST",
    ),
    STUN_SERVER_PROP,
    TURN_SERVER_PROP,
    TURN_USER_PROP,
    TURN_PASS_PROP,
    PUBLISHER_RECONNECT_PROP,
    RECONNECT_BACKOFF_PROP,
    RECONNECT_BACKOFF_MAX_PROP,
];

fn accepted_caps() -> CapsSet {
    CapsSet::from_alternatives(sink_caps_alternatives())
}

impl PadTemplates for WebRtcSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(accepted_caps())])
    }
}

impl AsyncElement for WebRtcSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn set_instance_name(&mut self, name: String) {
        self.settings.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.settings.log_name.set_category(category);
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // One track per sink instance: any carried video codec, or Opus audio.
        Track::from_caps(upstream_caps)
            .map(|_| upstream_caps.clone())
            .ok_or(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(accepted_caps())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.settings.track = Track::from_caps(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "WebRTC sink",
            "Sink/Network/WebRTC",
            "Publishes video (H.264/VP8/VP9/H.265/AV1) or Opus to a WHIP server over WebRTC (str0m: ICE/DTLS/SRTP)",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        WEBRTCSINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(result) = self.settings.reconnect.set_property(name, &value) {
            return result;
        }
        match name {
            // `location` is the gst-canonical name; `whip-url` is accepted too.
            "location" | "whip-url" => {
                self.settings.whip_url = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "auth-token" => {
                let token = value.as_str().ok_or(PropError::Type)?;
                self.settings.bearer = if token.is_empty() {
                    None
                } else {
                    Some(token.into())
                };
                Ok(())
            }
            "stun-server" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.settings.stun_server = if s.is_empty() { None } else { Some(s.into()) };
                Ok(())
            }
            "turn-server" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.settings.turn_server = if s.is_empty() { None } else { Some(s.into()) };
                Ok(())
            }
            "turn-user" => {
                self.settings.turn_user = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "turn-pass" => {
                self.settings.turn_pass = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = self.settings.reconnect.get_property(name) {
            return Some(value);
        }
        match name {
            "location" | "whip-url" => Some(PropValue::Str(self.settings.whip_url.clone())),
            "auth-token" => Some(PropValue::Str(
                self.settings.bearer.clone().unwrap_or_default(),
            )),
            "stun-server" => Some(PropValue::Str(
                self.settings.stun_server.clone().unwrap_or_default(),
            )),
            "turn-server" => Some(PropValue::Str(
                self.settings.turn_server.clone().unwrap_or_default(),
            )),
            "turn-user" => Some(PropValue::Str(self.settings.turn_user.clone())),
            "turn-pass" => Some(PropValue::Str(self.settings.turn_pass.clone())),
            _ => None,
        }
    }

    fn take_reconfigure(&mut self) -> Option<Reconfigure> {
        // The session task set this on a remote PLI or a reconnect; clear and
        // forward it as a keyframe request to the upstream encoder.
        if self
            .shared
            .keyframe_requested
            .swap(false, Ordering::Relaxed)
        {
            Some(Reconfigure::ForceKeyframe)
        } else {
            None
        }
    }

    fn take_bitrate(&mut self) -> Option<u32> {
        // Forward the session task's latest BWE estimate, but only when it
        // changed (this is polled every frame; the encoder need not re-see it).
        let cur = self.shared.bitrate_estimate.load(Ordering::Relaxed);
        if cur != 0 && cur != self.last_bitrate_sent {
            self.last_bitrate_sent = cur;
            Some(cur.min(u32::MAX as u64) as u32)
        } else {
            None
        }
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
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let unit = MediaUnit {
                        pts_ns: frame.timing.pts_ns,
                        capture_time: frame_capture_time(&frame),
                        data: slice.to_vec(),
                    };
                    if self.tx.is_none() {
                        self.start_publishing().await?;
                    }
                    if let Some(tx) = &self.tx {
                        // Bounded send: backpressures the pipeline if the session
                        // task is behind. A closed channel means the session was
                        // lost with no reconnect attempt left.
                        tx.send(unit).await.map_err(|_| G2gError::Shutdown)?;
                    }
                    self.frames_sent += 1;
                }
                // Clean end: DELETE the WHIP resource (RFC 9725 teardown)
                // synchronously here, then let the session task wind down when the
                // channel closes on element drop. Done in the element (not the
                // task) so it completes before the runtime tears the task down.
                PipelinePacket::Eos => {
                    self.shared
                        .resource
                        .delete_once(self.settings.bearer.as_deref())
                        .await;
                }
                // Geometry refinement lives in the in-band SPS, not the m-line.
                PipelinePacket::CapsChanged(_) => {}
                PipelinePacket::Flush => {}
                PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }
}

/// The sans-IO driving loop: owns the `Rtc` and the UDP socket, drains
/// `poll_output` to a deadline, then waits on incoming UDP, an outgoing access
/// unit, or the str0m timeout, whichever comes first.
async fn run_session(
    connection: Connection,
    settings: Arc<SessionSettings>,
    shared: SessionShared,
    rx: &mut mpsc::Receiver<MediaUnit>,
) -> SessionEnd {
    let Connection {
        mut rtc,
        socket,
        local,
        mid,
        mut turn,
        etag,
    } = connection;
    let track = settings.track;
    let bearer = settings.bearer.as_deref();
    let resource = &shared.resource;
    let mut buf = alloc::vec![0u8; UDP_RECV_BYTES];
    // The negotiated payload type for this track's codec, discovered once the
    // writer exists.
    let mut pt: Option<Pt> = None;
    // Keep the TURN allocation + permissions alive while the session runs.
    let mut refresh_at = Instant::now() + turn::REFRESH_INTERVAL;
    // When ICE first went Disconnected; after `ICE_RESTART_TIMEOUT` we attempt an
    // ICE restart (RFC 9725). Cleared on recovery.
    let mut disconnected_since: Option<Instant> = None;

    loop {
        // Drain pending output until str0m asks us to wait for a deadline.
        let deadline = loop {
            match rtc.poll_output() {
                Ok(Output::Timeout(t)) => break t,
                Ok(Output::Transmit(t)) => send_transmit(&socket, &mut turn, &t).await,
                Ok(Output::Event(Event::IceConnectionStateChange(state))) => match state {
                    IceConnectionState::Disconnected => {
                        disconnected_since.get_or_insert_with(Instant::now);
                    }
                    IceConnectionState::Connected | IceConnectionState::Completed => {
                        disconnected_since = None;
                    }
                    _ => {}
                },
                // Remote PLI: the peer lost data and needs a fresh keyframe.
                // Flag it; the element's `take_reconfigure` forwards a
                // `ForceKeyframe` up the reverse channel to the encoder.
                Ok(Output::Event(Event::KeyframeRequest(_))) => {
                    shared.keyframe_requested.store(true, Ordering::Relaxed);
                }
                // Congestion-control estimate: stash the latest target bitrate
                // (TWCC or REMB carry a `Bitrate`); `take_bitrate` relays changes.
                Ok(Output::Event(Event::EgressBitrateEstimate(kind))) => {
                    let bps = match kind {
                        BweKind::Twcc(b) | BweKind::Remb(_, b) => Some(b.as_u64()),
                        // `BweKind` is non-exhaustive; ignore unknown future kinds.
                        _ => None,
                    };
                    if let Some(bps) = bps {
                        shared.bitrate_estimate.store(bps, Ordering::Relaxed);
                    }
                }
                Ok(Output::Event(event)) => shared.stats.record(&event),
                Err(error) => {
                    resource.delete_once(bearer).await;
                    return SessionEnd::Lost(format!("str0m output failed: {error}"));
                }
            }
        };

        // Sustained ICE disconnect: attempt an ICE restart against the resource.
        // A server that rejects the restart PATCH, such as one that restarted and
        // forgot the session, ends the session.
        if disconnected_since.is_some_and(|t| t.elapsed() >= ICE_RESTART_TIMEOUT) {
            disconnected_since = None;
            let Some(url) = resource.url() else {
                return SessionEnd::Lost(
                    "ICE disconnected and the session has no resource URL".into(),
                );
            };
            let restart = ice_restart(&mut rtc, &url, bearer, etag.as_deref()).await;
            if restart != TricklePatch::Accepted {
                resource.delete_once(bearer).await;
                return SessionEnd::Lost(format!(
                    "ICE disconnected and the restart PATCH got {restart:?}"
                ));
            }
        }

        let timeout = deadline.saturating_duration_since(Instant::now());

        tokio::select! {
            // Incoming UDP (STUN / DTLS / RTCP from the peer, or TURN-framed
            // traffic from the relay server).
            r = socket.recv_from(&mut buf) => {
                let (n, source) = match r {
                    Ok(received) => received,
                    Err(error) => {
                        resource.delete_once(bearer).await;
                        return SessionEnd::Lost(format!("UDP receive failed: {error}"));
                    }
                };
                if !feed_datagram(&mut rtc, &mut turn, local, &buf[..n], source) {
                    resource.delete_once(bearer).await;
                    return SessionEnd::Lost("str0m rejected a datagram".into());
                }
            }
            // Keep the TURN allocation + permissions alive.
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(refresh_at)), if !turn.is_empty() => {
                turn.refresh_all(&socket).await;
                refresh_at = Instant::now() + turn::REFRESH_INTERVAL;
            }
            // A closed channel = element drop. Normally the element already
            // deleted the resource on `Eos`, which makes the call below a no-op;
            // it still covers an element dropped without one.
            unit = rx.recv() => {
                let Some(unit) = unit else {
                    // Clean end: flush the pacer tail before dropping the socket.
                    drain_pacer(&mut rtc, &socket, &mut turn).await;
                    resource.delete_once(bearer).await;
                    return SessionEnd::ElementGone;
                };
                if pt.is_none() {
                    if let Some(writer) = rtc.writer(mid) {
                        pt = writer
                            .payload_params()
                            .find(|p| p.spec().codec == track.codec())
                            .map(|p| p.pt());
                    }
                }
                if let Some(p) = pt {
                    let rtp_time = track.media_time(unit.pts_ns);
                    if let Some(writer) = rtc.writer(mid) {
                        let _ = stamp_capture_time(writer, unit.capture_time)
                            .write(p, Instant::now(), rtp_time, unit.data);
                    }
                }
            }
            // str0m's timer fired.
            _ = tokio::time::sleep(timeout) => {
                if let Err(error) = rtc.handle_input(Input::Timeout(Instant::now())) {
                    resource.delete_once(bearer).await;
                    return SessionEnd::Lost(format!("str0m timeout handling failed: {error}"));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::webrtc_codec::opus_caps;
    use g2g_core::{Dim, Rate};

    fn h264_any() -> Caps {
        WebRtcVideoCodec::H264.caps_any()
    }

    #[test]
    fn accepts_every_carried_video_codec() {
        for codec in WebRtcVideoCodec::ALL {
            let mut sink = WebRtcSink::new("http://localhost:8889/s/whip");
            assert!(sink.intercept_caps(&codec.caps_any()).is_ok());
            assert!(sink.configure_pipeline(&codec.caps_any()).is_ok());
            assert_eq!(sink.settings.track, Track::Video(codec));
        }
    }

    #[test]
    fn accepts_h264_rejects_raw() {
        let sink = WebRtcSink::new("http://localhost:8889/s/whip");
        assert!(sink.intercept_caps(&h264_any()).is_ok());
        let rgba = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::Rgba8,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert_eq!(sink.intercept_caps(&rgba), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn configure_requires_h264() {
        let mut sink = WebRtcSink::new("http://localhost:8889/s/whip");
        assert!(sink.configure_pipeline(&h264_any()).is_ok());
        let mut sink2 = WebRtcSink::new("http://localhost:8889/s/whip");
        let raw = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::I420,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert!(matches!(
            sink2.configure_pipeline(&raw),
            Err(G2gError::CapsMismatch)
        ));
    }

    #[test]
    fn builders_set_fields() {
        let sink = WebRtcSink::new("http://h/whip")
            .with_bearer("tok")
            .with_queue_depth(8)
            .with_turn_server("turn:3478", "u", "p");
        assert_eq!(sink.settings.bearer.as_deref(), Some("tok"));
        assert_eq!(sink.queue_depth, 8);
        assert_eq!(sink.settings.turn_server.as_deref(), Some("turn:3478"));
        assert_eq!(sink.settings.turn_user, "u");
        assert_eq!(sink.settings.turn_pass, "p");
        assert_eq!(sink.frames_sent(), 0);
    }

    #[test]
    fn location_and_bearer_properties_round_trip() {
        let mut sink = WebRtcSink::new("http://h/whip");
        sink.set_property("location", PropValue::Str("http://srv:8889/s/whip".into()))
            .unwrap();
        assert_eq!(sink.settings.whip_url, "http://srv:8889/s/whip");
        assert_eq!(
            sink.get_property("location"),
            Some(PropValue::Str("http://srv:8889/s/whip".into()))
        );
        // `whip-url` is an accepted alias for the same field.
        sink.set_property("whip-url", PropValue::Str("http://x/whip".into()))
            .unwrap();
        assert_eq!(sink.settings.whip_url, "http://x/whip");
        sink.set_property("auth-token", PropValue::Str("secret".into()))
            .unwrap();
        assert_eq!(sink.settings.bearer.as_deref(), Some("secret"));
        sink.set_property(
            "stun-server",
            PropValue::Str("stun.l.google.com:19302".into()),
        )
        .unwrap();
        assert_eq!(
            sink.settings.stun_server.as_deref(),
            Some("stun.l.google.com:19302")
        );
        sink.set_property("turn-server", PropValue::Str("relay:3478".into()))
            .unwrap();
        sink.set_property("turn-user", PropValue::Str("u".into()))
            .unwrap();
        sink.set_property("turn-pass", PropValue::Str("p".into()))
            .unwrap();
        assert_eq!(sink.settings.turn_server.as_deref(), Some("relay:3478"));
        assert_eq!(
            sink.get_property("turn-user"),
            Some(PropValue::Str("u".into()))
        );
        // Empty turn-server clears the relay.
        sink.set_property("turn-server", PropValue::Str(String::new()))
            .unwrap();
        assert_eq!(sink.settings.turn_server, None);
        // Empty bearer clears it; unknown name and wrong type are distinct errors.
        sink.set_property("auth-token", PropValue::Str(String::new()))
            .unwrap();
        assert_eq!(sink.settings.bearer, None);
        assert_eq!(
            sink.set_property("nope", PropValue::Str("x".into())),
            Err(PropError::Unknown)
        );
        assert_eq!(
            sink.set_property("location", PropValue::Int(1)),
            Err(PropError::Type)
        );
    }

    #[test]
    fn keyframe_request_flag_surfaces_as_force_keyframe() {
        let sink = WebRtcSink::new("http://h/whip");
        // No PLI yet: nothing to forward.
        let mut s = sink;
        assert_eq!(s.take_reconfigure(), None);
        // Session task observed a remote PLI -> sets the shared flag.
        s.shared.keyframe_requested.store(true, Ordering::Relaxed);
        assert_eq!(s.take_reconfigure(), Some(Reconfigure::ForceKeyframe));
        // Consumed: a second poll yields nothing until the next PLI.
        assert_eq!(s.take_reconfigure(), None);
    }

    #[test]
    fn bitrate_estimate_surfaces_as_take_bitrate_on_change() {
        let mut s = WebRtcSink::new("http://h/whip");
        // No estimate yet.
        assert_eq!(s.take_bitrate(), None);
        // Session task stored a BWE estimate.
        s.shared
            .bitrate_estimate
            .store(1_500_000, Ordering::Relaxed);
        assert_eq!(s.take_bitrate(), Some(1_500_000));
        // Same value again: not re-reported (the encoder need not re-see it).
        assert_eq!(s.take_bitrate(), None);
        // A new estimate is reported.
        s.shared.bitrate_estimate.store(900_000, Ordering::Relaxed);
        assert_eq!(s.take_bitrate(), Some(900_000));
    }

    #[test]
    fn accepts_opus_audio() {
        let sink = WebRtcSink::new("http://h/whip");
        assert!(sink.intercept_caps(&opus_caps(2)).is_ok());
        let mut s = WebRtcSink::new("http://h/whip");
        assert!(s.configure_pipeline(&opus_caps(2)).is_ok());
        assert_eq!(s.settings.track, Track::Audio);
    }
}

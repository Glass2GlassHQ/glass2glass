//! Native LiveKit publisher sink (`LiveKitSink`): joins a LiveKit room and
//! publishes H.264 video (and optional Opus audio) over one str0m PeerConnection,
//! so other room participants can subscribe. This is the T4 signaller layered
//! over the T1 str0m engine, the way gst-plugins-rs layers a LiveKit signaller
//! over `webrtcbin`. The WebSocket + protobuf protocol lives in
//! [`crate::livekit_signal`]; this element owns the media PeerConnection.
//!
//! Shape mirrors [`crate::webrtcsession::WebRtcSessionSink`]: a
//! [`MultiInputElement`] driven by the terminal fan-in runner
//! [`run_fanin_session`](g2g_core::runtime::run_fanin_session), one input per
//! track (video, optionally audio), the track kind read from each input's caps so
//! pad order does not matter. On the first frame it runs the LiveKit join +
//! publish handshake, then spawns a task that owns the `Rtc` + `UdpSocket` + the
//! signalling WebSocket.
//!
//! LiveKit uses two PeerConnections per client. Minting a publish-only token
//! (`canSubscribe = false`) makes the server treat the client-offered publisher
//! PC as primary, so no server subscriber offer arrives and this milestone drives
//! a single `Rtc`. A later ingest milestone answers the server-offered subscriber
//! PC (the answerer role) reusing the same [`crate::livekit_signal`] envelopes.
//!
//! Handshake (per the LiveKit client protocol):
//! `JoinResponse` -> `AddTrackRequest` per track -> `TrackPublishedResponse` ->
//! client `offer` -> server `answer` -> trickle ICE (both ways, publisher
//! target). The `AddTrackRequest.cid` equals the SDP msid track-id str0m emits
//! (we set it explicitly), so the server maps each announced track to its m-line.
//!
//! Behind the `webrtc-livekit` feature. On-network validated against
//! `livekit-server --dev`.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::net::SocketAddr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use tokio::net::{TcpStream, UdpSocket};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use str0m::bwe::{Bitrate, BweKind};
use str0m::change::{SdpAnswer, SdpPendingOffer};
use str0m::crypto::from_feature_flags;
use str0m::media::{Direction, Mid, Pt, Rid};
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    HardwareError, MemoryDomain, MultiInputElement, OutputSink, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, ReverseChannel, VideoCodec,
};

use crate::filesink::io_err;
use crate::livekit_signal::{
    candidate_from_init_json, candidate_init_json, mint_token, signal_ws_url, AddTrackRequest,
    SessionDescription, SignalRequest, SignalResponse, SignalTarget, TrackSource, TrackType,
    VideoGrant, VideoLayer, VideoQuality,
};
use crate::webrtc_simulcast::{
    rids_high_to_low, send_simulcast, KeyframeRoutes, LayerAllocator, SendLayer,
};
use crate::webrtc_util::{add_ice_candidates, feed_datagram, select_host_ip, send_transmit};
use crate::webrtcsink::Track;

/// Default bounded depth of the element->session media channel (per direction).
const DEFAULT_QUEUE_DEPTH: usize = 256;

/// The maximum simulcast video layers on one m-line (rids `q`/`h`/`f`).
const MAX_VIDEO_LAYERS: usize = 3;

/// Access-token lifetime when the sink mints one from api-key/secret.
const TOKEN_TTL_SECS: u64 = 6 * 3600;

/// A split LiveKit signalling WebSocket (client side is TLS-capable).
type WsWrite = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>;
type WsRead = SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>;

/// One encoded access unit handed to the session task, tagged with the m-line
/// writer it targets (`mid` + an optional simulcast `rid`) and its track kind
/// (for the RTP clock + payload type).
#[derive(Debug)]
struct MediaUnit {
    track: Track,
    mid: Mid,
    rid: Option<Rid>,
    pts_ns: u64,
    data: Vec<u8>,
}

/// Where one input pad's access units go on the wire, resolved once the session
/// handshake fixes the m-lines: the track kind, its `mid`, and a simulcast `rid`
/// for a video layer (`None` = the sole video stream or an audio track).
#[derive(Debug, Clone, Copy)]
struct PadSink {
    track: Track,
    mid: Mid,
    rid: Option<Rid>,
}

/// LiveKit publisher sink. See the module docs.
pub struct LiveKitSink {
    /// LiveKit signalling base URL, e.g. `ws://localhost:7880`.
    url: String,
    room: String,
    identity: String,
    api_key: String,
    api_secret: String,
    /// Pre-minted access token; when set, `api_key` / `api_secret` are unused.
    token: Option<String>,
    queue_depth: usize,
    /// Number of video layers on one m-line (1 = single stream, >=2 = simulcast).
    video_layers: usize,
    /// Aggregate send-bitrate cap in bits/second (0 = uncapped), applied to the
    /// BWE estimate before the layer allocator budgets it.
    max_send_bitrate: u64,
    /// Whether an Opus audio track is published alongside the video.
    has_audio: bool,
    /// Number of input pads = video layers + audio, kept in sync with the above.
    track_count: usize,
    /// Track kind per input pad, set in `configure_pipeline`.
    tracks: Vec<Option<Track>>,
    /// Fixated video resolution per input pad (0,0 for audio), recorded in
    /// `configure_pipeline` and used to build each simulcast layer's metadata.
    dims: Vec<(u32, u32)>,
    /// Per-input reverse channel, shared with the fan-in runner: a remote PLI /
    /// BWE naming a track's m-line (+ rid) routes back to the source feeding that
    /// pad.
    reverse: Vec<ReverseChannel>,
    /// Per-input wire target, resolved during the handshake in `start_session`.
    sinks: Vec<Option<PadSink>>,
    /// Set on the first frame, after the join+publish handshake spawns the task.
    tx: Option<mpsc::Sender<MediaUnit>>,
    frames_sent: u64,
}

impl core::fmt::Debug for LiveKitSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LiveKitSink")
            .field("url", &self.url)
            .field("room", &self.room)
            .field("identity", &self.identity)
            .field("track_count", &self.track_count)
            .field("frames_sent", &self.frames_sent)
            .finish()
    }
}

impl LiveKitSink {
    /// Publish into `room` at the LiveKit signalling `url` (`ws://host:7880`),
    /// as participant `identity`. Video only by default; call
    /// [`Self::with_audio`] to add an Opus audio m-line. Set credentials with
    /// [`Self::with_api_key`] (+ secret) or a pre-minted [`Self::with_token`].
    pub fn new(
        url: impl Into<String>,
        room: impl Into<String>,
        identity: impl Into<String>,
    ) -> Self {
        let mut s = Self {
            url: url.into(),
            room: room.into(),
            identity: identity.into(),
            api_key: String::new(),
            api_secret: String::new(),
            token: None,
            queue_depth: DEFAULT_QUEUE_DEPTH,
            video_layers: 1,
            max_send_bitrate: 0,
            has_audio: false,
            track_count: 0,
            tracks: Vec::new(),
            dims: Vec::new(),
            reverse: Vec::new(),
            sinks: Vec::new(),
            tx: None,
            frames_sent: 0,
        };
        s.rebuild_pads();
        s
    }

    /// Publish a second (Opus audio) track alongside the video.
    pub fn with_audio(mut self) -> Self {
        self.has_audio = true;
        self.rebuild_pads();
        self
    }

    /// Publish the video as `layers` simulcast layers on one m-line (each an input
    /// pad, pad 0 = highest resolution). One layer is a plain single stream; two
    /// or more offer `a=rid`/`a=simulcast` so the SFU can forward per subscriber.
    pub fn with_simulcast(mut self, layers: usize) -> Self {
        self.video_layers = layers.clamp(1, MAX_VIDEO_LAYERS);
        self.rebuild_pads();
        self
    }

    /// Resize the per-pad vectors for the current video-layer + audio count.
    fn rebuild_pads(&mut self) {
        let n = self.video_layers + self.has_audio as usize;
        self.track_count = n;
        self.tracks = alloc::vec![None; n];
        self.dims = alloc::vec![(0u32, 0u32); n];
        self.reverse = (0..n).map(|_| ReverseChannel::new()).collect();
        self.sinks = alloc::vec![None; n];
    }

    /// Cap the aggregate send bitrate (bits/second, 0 = uncapped). The BWE
    /// estimate is clamped to this before the simulcast layer allocator budgets
    /// it, so a cap below the layers' combined nominal rate sheds the top layer.
    pub fn with_max_send_bitrate(mut self, bps: u64) -> Self {
        self.max_send_bitrate = bps;
        self
    }

    /// LiveKit API key + secret; the sink mints a publish-only access token.
    pub fn with_api_key(mut self, key: impl Into<String>, secret: impl Into<String>) -> Self {
        self.api_key = key.into();
        self.api_secret = secret.into();
        self
    }

    /// Use a pre-minted access token instead of minting one from key + secret.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Access units handed to the session so far.
    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    /// True once every input pad has been configured.
    fn all_configured(&self) -> bool {
        self.tracks
            .iter()
            .take(self.track_count)
            .all(|t| t.is_some())
    }

    /// The access token to present: the pre-minted one, or a freshly minted
    /// publish-only token (join `room`, `canPublish`, `canSubscribe = false` so
    /// the server makes the publisher PC primary and sends no subscriber offer).
    fn access_token(&self) -> Result<String, G2gError> {
        if let Some(t) = &self.token {
            return Ok(t.clone());
        }
        if self.api_key.is_empty() || self.api_secret.is_empty() {
            std::eprintln!("livekit: no token and no api-key/api-secret set");
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let grant = VideoGrant::publisher(&self.room);
        Ok(mint_token(
            &self.api_key,
            &self.api_secret,
            &self.identity,
            &grant,
            now,
            TOKEN_TTL_SECS,
        ))
    }

    /// Run the LiveKit join + publish handshake and spawn the session task.
    /// Runs on the first frame, once every track kind is known.
    async fn start_session(&mut self) -> Result<(), G2gError> {
        let hw = || G2gError::Hardware(HardwareError::Other);
        let token = self.access_token()?;
        let ws_url = signal_ws_url(&self.url, &token, false);

        let (ws, _resp) = connect_async(&ws_url).await.map_err(|e| {
            std::eprintln!("livekit: WebSocket connect to {} failed: {e}", self.url);
            hw()
        })?;
        let mut ws = ws;

        // Phase A: the server's first envelope is the JoinResponse.
        let join = loop {
            match recv_signal(&mut ws).await? {
                Some(SignalResponse::Join(j)) => break j,
                Some(SignalResponse::Leave) | None => {
                    std::eprintln!("livekit: closed before JoinResponse");
                    return Err(hw());
                }
                // Ignore anything before the join (none expected).
                Some(_) => {}
            }
        };
        let ping_interval = if join.ping_interval > 0 {
            Duration::from_secs(join.ping_interval as u64)
        } else {
            Duration::from_secs(15)
        };

        // Video input pads in order (pad 0 = top layer), each a simulcast layer;
        // one shared video m-line groups them all. Assign rids high->low and read
        // each layer's resolution from its fixated caps.
        let video_inputs: Vec<usize> = self
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == Some(Track::Video))
            .map(|(i, _)| i)
            .collect();
        let rids = rids_high_to_low(video_inputs.len());
        let layers: Vec<SendLayer> = video_inputs
            .iter()
            .zip(rids.iter())
            .map(|(&i, &rid)| SendLayer {
                rid,
                width: self.dims[i].0,
                height: self.dims[i].1,
            })
            .collect();
        let simulcast = layers.len() >= 2;
        // Simulcast enables BWE so the aggregate estimate can budget the layer
        // set (whole-layer on/off in the allocator); the estimate is clamped to
        // `max-send-bitrate` when set.
        let allocator = simulcast.then(|| LayerAllocator::new(&layers, self.max_send_bitrate));

        // Build the publisher Rtc with a send-only m-line per configured track.
        let host_ip = select_host_ip();
        let socket = UdpSocket::bind((host_ip, 0)).await.map_err(io_err)?;
        let local = socket.local_addr().map_err(io_err)?;

        let mut rtc = RtcConfig::new()
            .set_crypto_provider(Arc::new(from_feature_flags()))
            .clear_codecs()
            .enable_h264(true)
            .enable_opus(true)
            .enable_bwe(allocator.as_ref().map(|a| Bitrate::bps(a.initial_bps())))
            .build(Instant::now());
        // Host candidate only for the local dev server; STUN/TURN NAT traversal
        // toward LiveKit Cloud is a follow-up.
        add_ice_candidates(&mut rtc, &socket, None).await?;
        let has_video = !video_inputs.is_empty();
        let has_audio = self.tracks.contains(&Some(Track::Audio));

        // The AddTrackRequest cid must equal the SDP msid track-id, so we choose
        // it and hand it to str0m as the track_id for that m-line.
        let stream_id = format!("g2g-{}", self.identity);
        let video_cid = format!("{stream_id}-video");
        let audio_cid = format!("{stream_id}-audio");

        let (offer_sdp, pending, video_mid, audio_mid): (
            String,
            SdpPendingOffer,
            Option<Mid>,
            Option<Mid>,
        ) = {
            let mut api = rtc.sdp_api();
            let video_mid = has_video.then(|| {
                api.add_media(
                    Track::Video.media_kind(),
                    Direction::SendOnly,
                    Some(stream_id.clone()),
                    Some(video_cid.clone()),
                    send_simulcast(&layers),
                )
            });
            let audio_mid = has_audio.then(|| {
                api.add_media(
                    Track::Audio.media_kind(),
                    Direction::SendOnly,
                    Some(stream_id.clone()),
                    Some(audio_cid.clone()),
                    None,
                )
            });
            let (offer, pending) = api.apply().ok_or_else(hw)?;
            (offer.to_sdp_string(), pending, video_mid, audio_mid)
        };

        // Phase B: announce each track and await its TrackPublishedResponse.
        let mut want_cids: Vec<String> = Vec::new();
        if has_video {
            // The track's geometry is the top layer; `layers` announces each
            // simulcast layer's quality + resolution (empty for a single stream).
            let (top_w, top_h) = layers
                .first()
                .map(|l| (l.width, l.height))
                .unwrap_or((0, 0));
            let layer_meta: Vec<VideoLayer> = if simulcast {
                layers
                    .iter()
                    .map(|l| VideoLayer {
                        quality: VideoQuality::for_rid(l.rid),
                        width: l.width,
                        height: l.height,
                    })
                    .collect()
            } else {
                Vec::new()
            };
            send_signal(
                &mut ws,
                &SignalRequest::AddTrack(AddTrackRequest {
                    cid: video_cid.clone(),
                    name: "video".into(),
                    track_type: TrackType::Video,
                    width: top_w,
                    height: top_h,
                    source: TrackSource::Camera,
                    layers: layer_meta,
                }),
            )
            .await?;
            want_cids.push(video_cid.clone());
        }
        if has_audio {
            send_signal(
                &mut ws,
                &SignalRequest::AddTrack(AddTrackRequest {
                    cid: audio_cid.clone(),
                    name: "audio".into(),
                    track_type: TrackType::Audio,
                    width: 0,
                    height: 0,
                    source: TrackSource::Microphone,
                    layers: Vec::new(),
                }),
            )
            .await?;
            want_cids.push(audio_cid.clone());
        }
        // Buffer any early server trickle so no remote candidate is lost while we
        // drain the TrackPublished / answer messages.
        let mut remote_cands: Vec<String> = Vec::new();
        while !want_cids.is_empty() {
            match recv_signal(&mut ws).await? {
                Some(SignalResponse::TrackPublished(tp)) => {
                    want_cids.retain(|c| *c != tp.cid);
                }
                Some(SignalResponse::Trickle(t)) => stash_candidate(&mut remote_cands, t),
                Some(SignalResponse::Leave) | None => return Err(hw()),
                Some(_) => {}
            }
        }

        // Phase C: send the publisher offer, await the answer, apply it.
        send_signal(
            &mut ws,
            &SignalRequest::Offer(SessionDescription {
                sdp_type: "offer".into(),
                sdp: offer_sdp.clone(),
            }),
        )
        .await?;
        let answer = loop {
            match recv_signal(&mut ws).await? {
                Some(SignalResponse::Answer(sd)) => break sd,
                Some(SignalResponse::Trickle(t)) => stash_candidate(&mut remote_cands, t),
                Some(SignalResponse::Leave) | None => {
                    std::eprintln!("livekit: closed before publisher answer");
                    return Err(hw());
                }
                Some(_) => {}
            }
        };
        let answer = SdpAnswer::from_sdp_string(&answer.sdp).map_err(|e| {
            std::eprintln!("livekit: publisher answer parse failed: {e:?}");
            hw()
        })?;
        rtc.sdp_api().accept_answer(pending, answer).map_err(|e| {
            std::eprintln!("livekit: publisher answer rejected: {e:?}");
            hw()
        })?;

        // Apply any buffered remote candidates now the remote description is set.
        for cand in remote_cands.drain(..) {
            add_remote_candidate(&mut rtc, &cand);
        }

        // Split the signalling socket: the task reads server messages and writes
        // trickle / ping. Trickle our own publisher candidates first (belt and
        // suspenders: they also ride inline in the offer SDP).
        let (mut ws_write, ws_read) = ws.split();
        let mid_str = video_mid
            .or(audio_mid)
            .map(|m| m.to_string())
            .unwrap_or_else(|| "0".to_string());
        for cand in local_candidates(&offer_sdp) {
            let init = candidate_init_json(&cand, &mid_str, 0);
            let _ = ws_write
                .send(Message::Binary(
                    SignalRequest::Trickle(crate::livekit_signal::TrickleRequest {
                        candidate_init: init,
                        target: SignalTarget::Publisher,
                    })
                    .encode(),
                ))
                .await;
        }

        // Resolve each input pad's wire target and per-(mid, rid) keyframe route.
        // A video layer writes to the shared video mid tagged with its rid (None
        // when there is only one layer); audio writes to its own mid.
        let mut routes = KeyframeRoutes::new();
        for (li, &i) in video_inputs.iter().enumerate() {
            if let Some(mid) = video_mid {
                let rid = simulcast.then(|| Rid::from(rids[li]));
                self.sinks[i] = Some(PadSink {
                    track: Track::Video,
                    mid,
                    rid,
                });
                routes.push(mid, rid, self.reverse[i].clone());
            }
        }
        if let Some(ai) = self.tracks.iter().position(|t| *t == Some(Track::Audio)) {
            if let Some(mid) = audio_mid {
                self.sinks[ai] = Some(PadSink {
                    track: Track::Audio,
                    mid,
                    rid: None,
                });
                routes.push(mid, None, self.reverse[ai].clone());
            }
        }
        // BWE is aggregate per connection; route the estimate to the top video
        // layer's source (per-layer allocation is a follow-up milestone).
        let video_reverse = video_inputs
            .first()
            .and_then(|&i| self.reverse.get(i).cloned());

        let (tx, rx) = mpsc::channel::<MediaUnit>(self.queue_depth);
        tokio::spawn(run_session(SessionArgs {
            rtc,
            socket,
            local,
            keyframe_routes: routes,
            video_reverse,
            allocator,
            ws_write,
            ws_read,
            ping_interval,
            rx,
        }));
        self.tx = Some(tx);
        Ok(())
    }
}

/// Send one `SignalRequest` as a binary WebSocket message.
async fn send_signal(
    ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>,
    req: &SignalRequest,
) -> Result<(), G2gError> {
    ws.send(Message::Binary(req.encode())).await.map_err(|e| {
        std::eprintln!("livekit: signal send failed: {e}");
        G2gError::Hardware(HardwareError::Other)
    })
}

/// Read the next server `SignalResponse`, skipping non-binary frames. `Ok(None)`
/// on a clean close.
async fn recv_signal(
    ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>,
) -> Result<Option<SignalResponse>, G2gError> {
    loop {
        match ws.next().await {
            Some(Ok(Message::Binary(bytes))) => {
                return Ok(SignalResponse::decode(&bytes));
            }
            Some(Ok(Message::Close(_))) | None => return Ok(None),
            Some(Ok(_)) => continue,
            Some(Err(e)) => {
                std::eprintln!("livekit: signal recv failed: {e}");
                return Err(G2gError::Hardware(HardwareError::Other));
            }
        }
    }
}

/// Pull the raw candidate string out of a received trickle and stash it.
fn stash_candidate(remote: &mut Vec<String>, t: crate::livekit_signal::TrickleRequest) {
    if let Some(c) = candidate_from_init_json(&t.candidate_init) {
        remote.push(c);
    }
}

/// Add a remote ICE candidate string (`candidate:...`) to `rtc`, ignoring
/// unparseable ones (attacker-controlled signalling: never panic).
fn add_remote_candidate(rtc: &mut Rtc, candidate: &str) {
    if let Ok(c) = Candidate::from_sdp_string(candidate) {
        rtc.add_remote_candidate(c);
    }
}

/// The `a=candidate:` lines from our own offer SDP, as raw candidate strings.
fn local_candidates(offer_sdp: &str) -> Vec<String> {
    offer_sdp
        .lines()
        .filter_map(|l| l.trim_end().strip_prefix("a=candidate:"))
        .map(|c| format!("candidate:{c}"))
        .collect()
}

/// Settable properties, so a `gst-launch` line can target a room without the
/// builder.
static LIVEKIT_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "url",
        PropKind::Str,
        "LiveKit signalling URL (ws://host:7880)",
    ),
    PropertySpec::new("room", PropKind::Str, "room name to join"),
    PropertySpec::new("identity", PropKind::Str, "participant identity"),
    PropertySpec::new("api-key", PropKind::Str, "LiveKit API key (mints a token)"),
    PropertySpec::new(
        "api-secret",
        PropKind::Str,
        "LiveKit API secret (mints a token)",
    ),
    PropertySpec::new(
        "token",
        PropKind::Str,
        "pre-minted access token (overrides api-key/secret)",
    ),
    PropertySpec::new(
        "max-send-bitrate",
        PropKind::Uint,
        "aggregate send bitrate cap in bits/second (0 = uncapped); a cap below the simulcast layers' combined nominal rate sheds the top layer",
    ),
];

fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn opus_stereo() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 48_000,
    }
}

/// The fixated `(width, height)` of a video caps, `(0, 0)` for audio or an
/// unfixated dimension (the layer metadata then simply omits the resolution).
fn video_dims(caps: &Caps) -> (u32, u32) {
    if let Caps::CompressedVideo { width, height, .. } = caps {
        let w = if let Dim::Fixed(w) = width { *w } else { 0 };
        let h = if let Dim::Fixed(h) = height { *h } else { 0 };
        (w, h)
    } else {
        (0, 0)
    }
}

/// The track kind an input's caps select (H.264 video or Opus audio).
fn track_of(caps: &Caps) -> Option<Track> {
    match caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            ..
        } => Some(Track::Video),
        Caps::Audio {
            format: AudioFormat::Opus,
            ..
        } => Some(Track::Audio),
        _ => None,
    }
}

impl MultiInputElement for LiveKitSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.track_count
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match track_of(upstream_caps) {
            Some(_) => Ok(upstream_caps.clone()),
            None => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::from_alternatives(Vec::from([
            h264_any(),
            opus_stereo(),
        ])))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let track = track_of(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        *self.tracks.get_mut(input).ok_or(G2gError::CapsMismatch)? = Some(track);
        if let Some(d) = self.dims.get_mut(input) {
            *d = video_dims(absolute_caps);
        }
        Ok(ConfigureOutcome::Accepted)
    }

    /// Terminal session: no merged output (the network is the destination).
    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(h264_any())
    }

    fn reverse_channel(&self, input: usize) -> Option<ReverseChannel> {
        self.reverse.get(input).cloned()
    }

    fn is_terminal(&self) -> bool {
        true
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "LiveKit sink",
            "Sink/Network/WebRTC",
            "Publishes H.264 (+ Opus) into a LiveKit room (str0m + WebSocket signalling)",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let data = slice.as_slice().to_vec();
                    let pts_ns = frame.timing.pts_ns;
                    // The handshake (first frame) resolves each pad's wire target.
                    if self.tx.is_none() {
                        if !self.all_configured() {
                            return Err(G2gError::NotConfigured);
                        }
                        self.start_session().await?;
                    }
                    let sink = self
                        .sinks
                        .get(input)
                        .copied()
                        .flatten()
                        .ok_or(G2gError::NotConfigured)?;
                    let unit = MediaUnit {
                        track: sink.track,
                        mid: sink.mid,
                        rid: sink.rid,
                        pts_ns,
                        data,
                    };
                    if let Some(tx) = &self.tx {
                        tx.send(unit).await.map_err(|_| G2gError::Shutdown)?;
                    }
                    self.frames_sent += 1;
                }
                PipelinePacket::Eos => {}
                PipelinePacket::CapsChanged(_) => {}
                PipelinePacket::Flush => {}
                PipelinePacket::Segment(_) => {}
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        LIVEKIT_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if name == "max-send-bitrate" {
            self.max_send_bitrate = value.as_uint().ok_or(PropError::Type)?;
            return Ok(());
        }
        let s = value.as_str().ok_or(PropError::Type)?;
        match name {
            "url" => self.url = s.into(),
            "room" => self.room = s.into(),
            "identity" => self.identity = s.into(),
            "api-key" => self.api_key = s.into(),
            "api-secret" => self.api_secret = s.into(),
            "token" => self.token = if s.is_empty() { None } else { Some(s.into()) },
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if name == "max-send-bitrate" {
            return Some(PropValue::Uint(self.max_send_bitrate));
        }
        let v = match name {
            "url" => self.url.clone(),
            "room" => self.room.clone(),
            "identity" => self.identity.clone(),
            "api-key" => self.api_key.clone(),
            "api-secret" => self.api_secret.clone(),
            "token" => self.token.clone().unwrap_or_default(),
            _ => return None,
        };
        Some(PropValue::Str(v))
    }
}

/// Owned inputs for the session task (grouped to dodge the too-many-arguments
/// lint and keep the spawn call readable).
struct SessionArgs {
    rtc: Rtc,
    socket: UdpSocket,
    local: SocketAddr,
    /// Per-(mid, rid) keyframe routing, so a remote PLI fires exactly the layer's
    /// source and never a sibling.
    keyframe_routes: KeyframeRoutes,
    /// Aggregate-BWE target (the top video layer's source).
    video_reverse: Option<ReverseChannel>,
    /// Whole-layer on/off budgeting of the aggregate BWE estimate (simulcast
    /// only). A dropped layer's units are skipped at the writer.
    allocator: Option<LayerAllocator>,
    ws_write: WsWrite,
    ws_read: WsRead,
    ping_interval: Duration,
    rx: mpsc::Receiver<MediaUnit>,
}

/// The sans-IO driving loop: owns the `Rtc`, the UDP socket, and the signalling
/// WebSocket. Drains `poll_output` (routing PLI / BWE back per-track), routes each
/// `MediaUnit` to its track writer, feeds server trickle into str0m, and pings on
/// the JoinResponse interval. Mirrors `WebRtcSessionSink::run_session` with the
/// WebSocket signalling folded in (LiveKit trickles over the WS, not HTTP PATCH).
async fn run_session(mut a: SessionArgs) {
    let mut buf = alloc::vec![0u8; 2000];
    let mut video_pt: Option<Pt> = None;
    let mut audio_pt: Option<Pt> = None;
    let mut ping_at = Instant::now() + a.ping_interval;

    loop {
        let deadline = loop {
            match a.rtc.poll_output() {
                Ok(Output::Timeout(t)) => break t,
                Ok(Output::Transmit(t)) => send_transmit(&a.socket, &mut None, &t).await,
                Ok(Output::Event(Event::IceConnectionStateChange(
                    IceConnectionState::Disconnected,
                ))) => {
                    // LiveKit reconnection (ICE restart / re-join) is a follow-up;
                    // a sustained disconnect just ends the session.
                    return;
                }
                Ok(Output::Event(Event::KeyframeRequest(req))) => {
                    a.keyframe_routes.request_keyframe(req.mid, req.rid);
                }
                Ok(Output::Event(Event::EgressBitrateEstimate(kind))) => {
                    let bps = match kind {
                        BweKind::Twcc(b) | BweKind::Remb(_, b) => Some(b.as_u64()),
                        _ => None,
                    };
                    if let (Some(bps), Some(rc)) = (bps, a.video_reverse.as_ref()) {
                        rc.set_bitrate(bps.min(u32::MAX as u64) as u32);
                    }
                    if let (Some(bps), Some(alloc)) = (bps, a.allocator.as_mut()) {
                        if alloc.update(Instant::now(), bps) {
                            std::eprintln!("livekit: layer set changed at {bps} bps estimate");
                        }
                    }
                }
                Ok(Output::Event(_)) => {}
                Err(_) => return,
            }
        };

        let timeout = deadline.saturating_duration_since(Instant::now());
        tokio::select! {
            r = a.socket.recv_from(&mut buf) => {
                let Ok((n, source)) = r else { return };
                if !feed_datagram(&mut a.rtc, &mut None, a.local, &buf[..n], source) {
                    return;
                }
            }
            msg = a.ws_read.next() => {
                match msg {
                    Some(Ok(Message::Binary(bytes))) => {
                        match SignalResponse::decode(&bytes) {
                            Some(SignalResponse::Trickle(t)) => {
                                if let Some(c) = candidate_from_init_json(&t.candidate_init) {
                                    add_remote_candidate(&mut a.rtc, &c);
                                }
                            }
                            // A server subscriber offer is unexpected for a
                            // publish-only token; ignore it (no subscriber PC).
                            Some(SignalResponse::Leave) => return,
                            _ => {}
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => return,
                    Some(Ok(_)) => {}
                    Some(Err(_)) => return,
                }
            }
            unit = a.rx.recv() => {
                let Some(unit) = unit else { return };
                let mid = unit.mid;
                let pt_slot = match unit.track {
                    Track::Video => &mut video_pt,
                    Track::Audio => &mut audio_pt,
                };
                // The pt is per-m-line, so all layers on the video mid share it.
                if pt_slot.is_none() {
                    if let Some(writer) = a.rtc.writer(mid) {
                        // Prefer a packetization-mode=1 payload type: str0m's
                        // H.264 payloader emits STAP-A / FU-A, which a strict
                        // receiver (Chrome) discards on a mode-0 PT. An SFU
                        // forwards our PT as negotiated, so picking the first
                        // H264 PT here silently broke browser subscribers.
                        let mode1 = writer.payload_params().find(|p| {
                            p.spec().codec == unit.track.codec()
                                && p.spec().format.packetization_mode == Some(1)
                        });
                        let any = writer
                            .payload_params()
                            .find(|p| p.spec().codec == unit.track.codec());
                        *pt_slot = mode1.or(any).map(|p| p.pt());
                    }
                }
                // A starved simulcast layer is shed whole: skip its units.
                let layer_on = match (unit.rid, a.allocator.as_ref()) {
                    (Some(rid), Some(alloc)) => alloc.is_on(rid),
                    _ => true,
                };
                if let Some(p) = *pt_slot {
                    if !layer_on {
                        continue;
                    }
                    let rtp_time = unit.track.media_time(unit.pts_ns);
                    if let Some(writer) = a.rtc.writer(mid) {
                        // Tag the write with the layer's rid so str0m routes it to
                        // that simulcast stream's SSRC (no rid = single stream).
                        let writer = match unit.rid {
                            Some(rid) => writer.rid(rid),
                            None => writer,
                        };
                        let _ = writer.write(p, Instant::now(), rtp_time, unit.data);
                    }
                }
            }
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(ping_at)) => {
                let now_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as i64)
                    .unwrap_or(0);
                let _ = a.ws_write
                    .send(Message::Binary(SignalRequest::Ping(now_ms).encode()))
                    .await;
                ping_at = Instant::now() + a.ping_interval;
            }
            _ = tokio::time::sleep(timeout) => {
                if a.rtc.handle_input(Input::Timeout(Instant::now())).is_err() {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_only_by_default_audio_opt_in() {
        let s = LiveKitSink::new("ws://h:7880", "room", "id");
        assert_eq!(s.input_count(), 1);
        let s = s.with_audio();
        assert_eq!(s.input_count(), 2);
    }

    #[test]
    fn configure_reads_track_kind_from_caps() {
        let mut s = LiveKitSink::new("ws://h:7880", "room", "id").with_audio();
        assert!(s.configure_pipeline(0, &h264_any()).is_ok());
        assert!(s.configure_pipeline(1, &opus_stereo()).is_ok());
        assert_eq!(
            s.tracks,
            alloc::vec![Some(Track::Video), Some(Track::Audio)]
        );
        assert!(s.all_configured());
    }

    #[test]
    fn simulcast_groups_video_layers_and_records_dims() {
        // Two video layers on one m-line: two input pads, both video, each with
        // its fixated resolution recorded for the layer metadata.
        let mut s = LiveKitSink::new("ws://h:7880", "room", "id").with_simulcast(2);
        assert_eq!(s.input_count(), 2);
        let hi = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
        };
        let lo = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Fixed(30 << 16),
        };
        assert!(s.configure_pipeline(0, &hi).is_ok());
        assert!(s.configure_pipeline(1, &lo).is_ok());
        assert_eq!(
            s.tracks,
            alloc::vec![Some(Track::Video), Some(Track::Video)]
        );
        assert_eq!(s.dims, alloc::vec![(640, 480), (320, 240)]);
        assert!(s.all_configured());
    }

    #[test]
    fn simulcast_with_audio_adds_an_extra_pad() {
        let s = LiveKitSink::new("ws://h:7880", "room", "id")
            .with_simulcast(2)
            .with_audio();
        assert_eq!(s.input_count(), 3, "two video layers + one audio");
        assert_eq!(s.video_layers, 2);
        assert!(s.has_audio);
    }

    #[test]
    fn rejects_non_av_caps() {
        let s = LiveKitSink::new("ws://h:7880", "room", "id");
        let raw = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::I420,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert_eq!(s.intercept_caps(0, &raw), Err(G2gError::CapsMismatch));
        assert!(s.intercept_caps(0, &h264_any()).is_ok());
    }

    #[test]
    fn access_token_prefers_preminted() {
        let s = LiveKitSink::new("ws://h:7880", "room", "id").with_token("PRE.MINTED.TOK");
        assert_eq!(s.access_token().unwrap(), "PRE.MINTED.TOK");
    }

    #[test]
    fn access_token_mints_from_key_secret() {
        let s = LiveKitSink::new("ws://h:7880", "room", "id").with_api_key("devkey", "secret");
        let tok = s.access_token().unwrap();
        assert_eq!(tok.split('.').count(), 3, "a minted JWT has three parts");
    }

    #[test]
    fn access_token_errors_without_credentials() {
        let s = LiveKitSink::new("ws://h:7880", "room", "id");
        assert!(s.access_token().is_err());
    }

    #[test]
    fn properties_round_trip() {
        let mut s = LiveKitSink::new("ws://h:7880", "room", "id");
        s.set_property("url", PropValue::Str("ws://srv:7880".into()))
            .unwrap();
        s.set_property("room", PropValue::Str("r2".into())).unwrap();
        s.set_property("api-key", PropValue::Str("k".into()))
            .unwrap();
        s.set_property("api-secret", PropValue::Str("sec".into()))
            .unwrap();
        assert_eq!(
            s.get_property("url"),
            Some(PropValue::Str("ws://srv:7880".into()))
        );
        assert_eq!(s.get_property("room"), Some(PropValue::Str("r2".into())));
        assert_eq!(s.api_key, "k");
        assert_eq!(
            s.set_property("nope", PropValue::Str("x".into())),
            Err(PropError::Unknown)
        );
    }

    #[test]
    fn local_candidates_parsed_from_offer() {
        let offer = "v=0\r\n\
            m=video 9 UDP/TLS/RTP/SAVPF 96\r\n\
            a=candidate:1 1 udp 2113 10.0.0.2 5000 typ host\r\n\
            a=mid:0\r\n";
        let c = local_candidates(offer);
        assert_eq!(
            c,
            alloc::vec!["candidate:1 1 udp 2113 10.0.0.2 5000 typ host"]
        );
    }
}

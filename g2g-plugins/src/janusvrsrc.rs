use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::net::SocketAddr;
use std::time::Instant;

use futures_util::{Sink, StreamExt};
use serde_json::{json, Value};
use tokio::net::UdpSocket;
use tokio_tungstenite::tungstenite::{Error as WsError, Message};

use str0m::crypto::from_feature_flags;
use str0m::media::MediaKind;
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_debug, g2g_info, g2g_warn, Caps, G2gError, HardwareError, MultiOutputSink,
    MultiOutputSource, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

pub use crate::janus_signal::DEFAULT_ENDPOINT;
use crate::janus_signal::{
    failure, id_value, leave, open_session, parse_frame, plugin_data, send_json, trickle_candidate,
    wait_for, Signaller, WsRead, WsWrite, KEEPALIVE_INTERVAL,
};
use crate::turn::TurnSet;
use crate::webrtc_codec::{
    enable_tracks, get_video_codec_property, opus_caps, set_video_codec_property, Track,
    WebRtcVideoCodec, VIDEO_CODEC_PROP,
};
use crate::webrtc_receive::{MediaReceiver, ReceivedUnit, UDP_RECV_BYTES};
use crate::webrtc_util::{
    accept_offer_sdp, add_ice_candidates, feed_datagram, select_host_ip, send_transmit,
    STUN_SERVER_PROP,
};

const VIDEO_PORT: usize = 0;
const AUDIO_PORT: usize = 1;
const OUTPUT_COUNT: usize = 2;
const STEREO: u8 = 2;

pub struct JanusVrSrc {
    endpoint: String,
    room_id: String,
    feed_id: Option<String>,
    api_secret: Option<String>,
    string_ids: bool,
    stun_server: Option<String>,
    video_codec: WebRtcVideoCodec,
    frame_limit: u64,
    log_name: LogName,
}

impl core::fmt::Debug for JanusVrSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("JanusVrSrc")
            .field("endpoint", &self.endpoint)
            .field("room_id", &self.room_id)
            .field("feed_id", &self.feed_id)
            .finish()
    }
}

impl JanusVrSrc {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            room_id: String::new(),
            feed_id: None,
            api_secret: None,
            string_ids: false,
            stun_server: None,
            video_codec: WebRtcVideoCodec::default(),
            frame_limit: u64::MAX,
            log_name: LogName::new(),
        }
    }

    pub fn with_room(mut self, room_id: impl Into<String>) -> Self {
        self.room_id = room_id.into();
        self
    }

    pub fn with_feed_id(mut self, feed_id: impl Into<String>) -> Self {
        self.feed_id = Some(feed_id.into());
        self
    }

    pub fn with_api_secret(mut self, secret: impl Into<String>) -> Self {
        self.api_secret = Some(secret.into());
        self
    }

    pub fn with_string_ids(mut self, string_ids: bool) -> Self {
        self.string_ids = string_ids;
        self
    }

    pub fn with_stun_server(mut self, server: impl Into<String>) -> Self {
        self.stun_server = Some(server.into());
        self
    }

    pub fn with_video_codec(mut self, codec: WebRtcVideoCodec) -> Self {
        self.video_codec = codec;
        self
    }

    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    fn room_value(&self) -> Result<Value, String> {
        id_value(&self.room_id, self.string_ids).ok_or_else(|| {
            format!(
                "room-id {:?} is not a number (set use-string-ids for a string room)",
                self.room_id
            )
        })
    }

    fn configured_feed(&self) -> Result<Option<Value>, String> {
        let Some(feed_id) = &self.feed_id else {
            return Ok(None);
        };
        id_value(feed_id, self.string_ids)
            .map(Some)
            .ok_or_else(|| format!("feed-id {feed_id:?} is not a number"))
    }

    async fn subscribe(&self) -> Result<Subscription, String> {
        let room = self.room_value()?;
        let configured_feed = self.configured_feed()?;
        let (mut ws, signaller) = open_session(&self.endpoint, self.api_secret.as_deref()).await?;

        let feed = match configured_feed {
            Some(feed) => feed,
            None => {
                let list = json!({ "request": "listparticipants", "room": room });
                send_json(&mut ws, signaller.plugin_message(list, None)).await?;
                let participants = wait_for(&mut ws, |m| {
                    (plugin_data(m)["videoroom"] == "participants")
                        .then(|| plugin_data(m)["participants"].clone())
                })
                .await?;
                first_publisher(&participants)
                    .ok_or_else(|| format!("room {} has no publisher", self.room_id))?
            }
        };
        let join = signaller.plugin_message(subscriber_join(room, feed.clone()), None);
        send_json(&mut ws, join).await?;
        let attached = wait_for(&mut ws, |m| {
            (plugin_data(m)["videoroom"] == "attached").then(|| m.clone())
        })
        .await?;
        check_attached_streams(plugin_data(&attached), self.video_codec)?;
        let offer = offer_sdp(&attached).ok_or("attached event carries no sdp offer")?;
        g2g_info!(self, "subscribing to feed {feed} in room {}", self.room_id);

        let socket = UdpSocket::bind((select_host_ip(), 0))
            .await
            .map_err(|e| format!("udp bind: {e}"))?;
        let local = socket.local_addr().map_err(|e| format!("udp bind: {e}"))?;
        let config = RtcConfig::new()
            .set_crypto_provider(Arc::new(from_feature_flags()))
            .clear_codecs();
        let mut rtc = enable_tracks(config, [Track::Video(self.video_codec), Track::Audio])
            .build(Instant::now());
        // candidates added before the answer ride inside it
        add_ice_candidates(&mut rtc, &socket, self.stun_server.as_deref())
            .await
            .map_err(|_| "ice candidate gathering failed".to_string())?;
        answer_offer(&mut rtc, &mut ws, &signaller, offer).await?;
        wait_for(&mut ws, |m| {
            (plugin_data(m)["started"] == "ok").then_some(())
        })
        .await?;
        g2g_debug!(self, "started, rtc socket {local}");

        let (ws_write, ws_read) = ws.split();
        Ok(Subscription {
            rtc,
            socket,
            local,
            signaller,
            ws_write,
            ws_read,
        })
    }
}

struct Subscription {
    rtc: Rtc,
    socket: UdpSocket,
    local: SocketAddr,
    signaller: Signaller,
    ws_write: WsWrite,
    ws_read: WsRead,
}

fn subscriber_join(room: Value, feed: Value) -> Value {
    json!({ "request": "join", "ptype": "subscriber", "room": room, "streams": [{ "feed": feed }] })
}

// the id of the first participant marked as publishing, in its wire shape
fn first_publisher(participants: &Value) -> Option<Value> {
    participants
        .as_array()?
        .iter()
        .find(|p| p["publisher"] == true && (p["id"].is_u64() || p["id"].is_string()))
        .map(|p| p["id"].clone())
}

fn offer_sdp(message: &Value) -> Option<&str> {
    (message["jsep"]["type"] == "offer")
        .then(|| message["jsep"]["sdp"].as_str())
        .flatten()
}

fn active_media_streams(data: &Value) -> Option<Vec<&Value>> {
    let streams = data["streams"].as_array()?;
    Some(
        streams
            .iter()
            .filter(|s| s["active"] != false && (s["type"] == "audio" || s["type"] == "video"))
            .collect(),
    )
}

fn check_attached_streams(data: &Value, video_codec: WebRtcVideoCodec) -> Result<(), String> {
    let streams = active_media_streams(data)
        .ok_or_else(|| format!("attached event has no streams: {data}"))?;
    if streams.is_empty() {
        return Err("the feed publishes no audio or video".to_string());
    }
    for stream in streams.iter().filter(|s| s["type"] == "video") {
        let codec = stream["codec"].as_str().unwrap_or_default();
        if codec != video_codec.name() {
            return Err(format!(
                "the feed sends {codec:?} video but video-codec is {}",
                video_codec.name()
            ));
        }
    }
    Ok(())
}

async fn answer_offer<S>(
    rtc: &mut Rtc,
    ws: &mut S,
    signaller: &Signaller,
    offer_sdp: &str,
) -> Result<(), String>
where
    S: Sink<Message, Error = WsError> + Unpin,
{
    let answer = accept_offer_sdp(rtc, offer_sdp)?;
    let jsep = json!({ "type": "answer", "sdp": answer.to_sdp_string(), "trickle": false });
    send_json(
        ws,
        signaller.plugin_message(json!({ "request": "start" }), Some(jsep)),
    )
    .await
}

#[derive(Debug)]
enum ServerEvent {
    Trickle(Candidate),
    Reoffer(String),
    PublisherGone(String),
    Failed(String),
    Other,
}

fn server_event(message: &Value) -> ServerEvent {
    if let Some(reason) = failure(message) {
        return ServerEvent::Failed(reason);
    }
    if let Some(candidate) = trickle_candidate(message) {
        return ServerEvent::Trickle(candidate);
    }
    let data = plugin_data(message);
    if let Some(feed) = data.get("unpublished").or_else(|| data.get("leaving")) {
        return ServerEvent::PublisherGone(format!("publisher {feed} left"));
    }
    if data["videoroom"] != "updated" {
        return ServerEvent::Other;
    }
    // janus marks every stream inactive when the publisher goes away
    let has_media = active_media_streams(data).is_some_and(|s| !s.is_empty());
    match offer_sdp(message) {
        Some(sdp) if has_media => ServerEvent::Reoffer(sdp.to_string()),
        _ => ServerEvent::PublisherGone("the feed stopped publishing".to_string()),
    }
}

static JANUSVRSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "janus-endpoint",
        PropKind::Str,
        "Janus WebSocket endpoint (ws:// or wss://)",
    )
    .with_default(DEFAULT_ENDPOINT),
    PropertySpec::new(
        "room-id",
        PropKind::Str,
        "VideoRoom to subscribe in (a number unless use-string-ids)",
    ),
    PropertySpec::new(
        "feed-id",
        PropKind::Str,
        "publisher id to subscribe to (empty = the room's first publisher)",
    ),
    PropertySpec::new(
        "secret-key",
        PropKind::Str,
        "Janus API secret (apisecret) when the server requires one",
    ),
    PropertySpec::new(
        "use-string-ids",
        PropKind::Bool,
        "send room-id and feed-id as strings (Janus string_ids)",
    )
    .with_default("false"),
    STUN_SERVER_PROP,
    VIDEO_CODEC_PROP,
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "access units across both tracks then EOS (-1 = until the publisher leaves)",
    )
    .with_default("-1")
    .with_range("-1", "9223372036854775807"),
];

impl MultiOutputSource for JanusVrSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }

    fn output_count(&self) -> usize {
        OUTPUT_COUNT
    }

    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        match output {
            VIDEO_PORT => Ok(self.video_codec.caps_placeholder()),
            AUDIO_PORT => Ok(opus_caps(STEREO)),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn properties(&self) -> &'static [PropertySpec] {
        JANUSVRSRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "num-buffers" => {
                return crate::numbuffers::set_num_buffers(&mut self.frame_limit, &value)
            }
            "video-codec" => return set_video_codec_property(&mut self.video_codec, &value),
            "use-string-ids" => {
                self.string_ids = value.as_bool().ok_or(PropError::Type)?;
                return Ok(());
            }
            _ => {}
        }
        let v = value.as_str().ok_or(PropError::Type)?;
        let optional = || (!v.is_empty()).then(|| v.into());
        match name {
            "janus-endpoint" => self.endpoint = v.into(),
            "room-id" => self.room_id = v.into(),
            "feed-id" => self.feed_id = optional(),
            "secret-key" => self.api_secret = optional(),
            "stun-server" => self.stun_server = optional(),
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let text = |value: &Option<String>| PropValue::Str(value.clone().unwrap_or_default());
        match name {
            "janus-endpoint" => Some(PropValue::Str(self.endpoint.clone())),
            "room-id" => Some(PropValue::Str(self.room_id.clone())),
            "feed-id" => Some(text(&self.feed_id)),
            "secret-key" => Some(text(&self.api_secret)),
            "use-string-ids" => Some(PropValue::Bool(self.string_ids)),
            "stun-server" => Some(text(&self.stun_server)),
            "video-codec" => Some(get_video_codec_property(self.video_codec)),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.frame_limit)),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if crate::numbuffers::finished_at_zero_limit_multi(self.frame_limit, OUTPUT_COUNT, out)
                .await?
            {
                return Ok(0);
            }
            let mut s = match self.subscribe().await {
                Ok(subscription) => subscription,
                Err(reason) => {
                    g2g_warn!(self, "{reason}");
                    return Err(G2gError::Hardware(HardwareError::Other));
                }
            };

            let video_codec = self.video_codec;
            out.push_to(
                VIDEO_PORT,
                PipelinePacket::CapsChanged(video_codec.caps_placeholder()),
            )
            .await?;
            out.push_to(AUDIO_PORT, PipelinePacket::CapsChanged(opus_caps(STEREO)))
                .await?;

            let mut media_receiver = MediaReceiver::new(video_codec);
            let mut keepalive_at = Instant::now() + KEEPALIVE_INTERVAL;
            let mut buf = alloc::vec![0u8; UDP_RECV_BYTES];
            let mut turn = TurnSet::empty();
            let mut seq = 0u64;
            macro_rules! finish {
                () => {{
                    leave(&mut s.ws_write, &mut s.ws_read, &s.signaller).await;
                    out.push_to(VIDEO_PORT, PipelinePacket::Eos).await?;
                    out.push_to(AUDIO_PORT, PipelinePacket::Eos).await?;
                    return Ok(seq);
                }};
            }

            loop {
                let mut frames: Vec<(usize, ReceivedUnit)> = Vec::new();
                let deadline = loop {
                    match s.rtc.poll_output() {
                        Ok(Output::Timeout(t)) => break t,
                        Ok(Output::Transmit(t)) => send_transmit(&s.socket, &mut turn, &t).await,
                        Ok(Output::Event(Event::MediaAdded(m))) => {
                            g2g_debug!(self, "{:?} track on mid {:?}", m.kind, m.mid);
                            media_receiver.media_added(&m);
                        }
                        Ok(Output::Event(Event::MediaData(d))) => {
                            let Some(unit) = media_receiver.media_data(&d) else {
                                continue;
                            };
                            if unit.first_keyframe {
                                g2g_debug!(self, "first video keyframe, forwarding");
                            }
                            let port = match unit.kind {
                                MediaKind::Video => VIDEO_PORT,
                                MediaKind::Audio => AUDIO_PORT,
                            };
                            frames.push((port, unit));
                        }
                        Ok(Output::Event(Event::IceConnectionStateChange(
                            IceConnectionState::Disconnected,
                        ))) => {
                            g2g_info!(self, "ice disconnected, ending run");
                            finish!()
                        }
                        Ok(Output::Event(_)) => {}
                        Err(e) => {
                            g2g_warn!(self, "rtc failed: {e:?}, ending run");
                            finish!()
                        }
                    }
                };

                if media_receiver.request_keyframe_if_due(&mut s.rtc) {
                    g2g_debug!(self, "no keyframe yet, sending pli");
                }

                for (port, unit) in frames {
                    let frame = unit.into_frame(seq);
                    out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
                    seq += 1;
                    if seq >= self.frame_limit {
                        finish!();
                    }
                }

                let timeout = deadline.saturating_duration_since(Instant::now());
                tokio::select! {
                    r = s.socket.recv_from(&mut buf) => {
                        let Ok((n, source)) = r else {
                            g2g_warn!(self, "rtc socket recv failed, ending run");
                            finish!()
                        };
                        let _ = feed_datagram(&mut s.rtc, &mut turn, s.local, &buf[..n], source);
                    }
                    frame = s.ws_read.next() => {
                        let Some(Ok(frame)) = frame else {
                            g2g_info!(self, "janus closed the websocket, ending run");
                            finish!()
                        };
                        let message = match parse_frame(&frame) {
                            Ok(message) => message.unwrap_or_default(),
                            Err(reason) => {
                                g2g_warn!(self, "{reason}, ending run");
                                finish!()
                            }
                        };
                        match server_event(&message) {
                            ServerEvent::Trickle(candidate) => s.rtc.add_remote_candidate(candidate),
                            ServerEvent::Reoffer(sdp) => {
                                g2g_debug!(self, "janus updated the subscription, answering");
                                let answered =
                                    answer_offer(&mut s.rtc, &mut s.ws_write, &s.signaller, &sdp).await;
                                if let Err(reason) = answered {
                                    g2g_warn!(self, "{reason}, ending run");
                                    finish!()
                                }
                            }
                            ServerEvent::PublisherGone(reason) => {
                                g2g_info!(self, "{reason}, ending run");
                                finish!()
                            }
                            ServerEvent::Failed(reason) => {
                                g2g_warn!(self, "{reason}, ending run");
                                finish!()
                            }
                            ServerEvent::Other => {}
                        }
                    }
                    _ = tokio::time::sleep_until(tokio::time::Instant::from_std(keepalive_at)) => {
                        if let Err(reason) = send_json(&mut s.ws_write, s.signaller.keepalive()).await {
                            g2g_warn!(self, "{reason}, ending run");
                            finish!()
                        }
                        keepalive_at = Instant::now() + KEEPALIVE_INTERVAL;
                    }
                    _ = tokio::time::sleep(timeout) => {
                        if s.rtc.handle_input(Input::Timeout(Instant::now())).is_err() {
                            g2g_warn!(self, "rtc timeout handling failed, ending run");
                            finish!();
                        }
                    }
                }
            }
        })
    }
}

impl LogSource for JanusVrSrc {
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

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{AudioFormat, VideoCodec};

    const ROOM_ID: &str = "1234";
    const FEED_ID: &str = "77";

    #[test]
    fn outputs_follow_the_video_codec() {
        let src = JanusVrSrc::new(DEFAULT_ENDPOINT).with_video_codec(WebRtcVideoCodec::Vp8);
        assert!(matches!(
            src.output_caps(VIDEO_PORT),
            Ok(Caps::CompressedVideo {
                codec: VideoCodec::Vp8,
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
        assert!(src.output_caps(OUTPUT_COUNT).is_err());
    }

    #[test]
    fn ids_follow_the_string_ids_switch() {
        let numeric = JanusVrSrc::new(DEFAULT_ENDPOINT)
            .with_room(ROOM_ID)
            .with_feed_id(FEED_ID);
        let join = subscriber_join(
            numeric.room_value().unwrap(),
            numeric.configured_feed().unwrap().unwrap(),
        );
        assert_eq!(join["ptype"], "subscriber");
        assert_eq!(join["room"], json!(1234));
        assert_eq!(join["streams"], json!([{ "feed": 77 }]));

        let strings = numeric.with_string_ids(true);
        let join = subscriber_join(
            strings.room_value().unwrap(),
            strings.configured_feed().unwrap().unwrap(),
        );
        assert_eq!(join["room"], json!(ROOM_ID));
        assert_eq!(join["streams"], json!([{ "feed": FEED_ID }]));

        let named = JanusVrSrc::new(DEFAULT_ENDPOINT)
            .with_room("lobby")
            .with_feed_id("camera");
        assert!(named.room_value().is_err());
        assert!(named.configured_feed().is_err());
        assert_eq!(
            JanusVrSrc::new(DEFAULT_ENDPOINT).configured_feed(),
            Ok(None)
        );
    }

    #[test]
    fn the_first_publishing_participant_is_picked() {
        let participants = json!([
            { "id": 5, "display": "viewer", "publisher": false },
            { "id": { "nested": 1 }, "publisher": true },
            { "id": 77, "display": "camera", "publisher": true },
            { "id": 78, "publisher": true },
        ]);
        assert_eq!(first_publisher(&participants), Some(json!(77)));
        assert_eq!(
            first_publisher(&json!([{ "id": "cam", "publisher": true }])),
            Some(json!("cam"))
        );
        assert_eq!(
            first_publisher(&json!([{ "id": 5, "publisher": false }])),
            None
        );
        assert_eq!(first_publisher(&json!({ "id": 5 })), None);
    }

    #[test]
    fn attached_streams_must_match_the_video_codec() {
        let attached = |codec: &str| {
            json!({ "videoroom": "attached", "streams": [
                { "type": "video", "active": true, "mid": "0", "codec": codec },
                { "type": "audio", "active": true, "mid": "1", "codec": "opus" },
            ] })
        };
        assert!(check_attached_streams(&attached("vp8"), WebRtcVideoCodec::Vp8).is_ok());
        let mismatch = check_attached_streams(&attached("vp8"), WebRtcVideoCodec::H264);
        assert!(
            mismatch.as_ref().is_err_and(|e| e.contains("\"vp8\"")),
            "{mismatch:?}"
        );
        let no_media = json!({ "streams": [{ "type": "data", "active": true }] });
        assert!(check_attached_streams(&no_media, WebRtcVideoCodec::Vp8).is_err());
        assert!(check_attached_streams(&json!({}), WebRtcVideoCodec::Vp8).is_err());
    }

    #[test]
    fn server_events_are_classified() {
        let event = |data: Value, jsep: Option<Value>| {
            let mut message = json!({ "janus": "event", "plugindata": { "data": data } });
            if let Some(jsep) = jsep {
                message["jsep"] = jsep;
            }
            message
        };
        let offer = json!({ "type": "offer", "sdp": "v=0" });
        let live =
            json!({ "videoroom": "updated", "streams": [{ "type": "video", "active": true }] });
        assert!(matches!(
            server_event(&event(live, Some(offer.clone()))),
            ServerEvent::Reoffer(sdp) if sdp == "v=0"
        ));
        let gone =
            json!({ "videoroom": "updated", "streams": [{ "type": "video", "active": false }] });
        assert!(matches!(
            server_event(&event(gone, Some(offer))),
            ServerEvent::PublisherGone(_)
        ));
        assert!(matches!(
            server_event(&event(
                json!({ "videoroom": "event", "unpublished": 77 }),
                None
            )),
            ServerEvent::PublisherGone(_)
        ));
        assert!(matches!(
            server_event(&json!({ "janus": "hangup", "reason": "DTLS alert" })),
            ServerEvent::Failed(_)
        ));
        let trickle = json!({ "janus": "trickle", "candidate": {
            "candidate": "candidate:1 1 udp 2130706431 192.0.2.1 5000 typ host", "sdpMid": "0" } });
        assert!(matches!(server_event(&trickle), ServerEvent::Trickle(_)));
        assert!(matches!(
            server_event(&json!({ "janus": "webrtcup" })),
            ServerEvent::Other
        ));
        assert!(matches!(server_event(&Value::Null), ServerEvent::Other));
    }

    #[test]
    fn properties_round_trip() {
        let mut src = JanusVrSrc::new(DEFAULT_ENDPOINT);
        for (name, value) in [
            ("janus-endpoint", "wss://janus.example/ws"),
            ("room-id", ROOM_ID),
            ("feed-id", FEED_ID),
            ("secret-key", "s3"),
            ("stun-server", "stun.example.com:3478"),
            ("video-codec", "vp8"),
        ] {
            src.set_property(name, PropValue::Str(value.into()))
                .unwrap();
            assert_eq!(src.get_property(name), Some(PropValue::Str(value.into())));
        }
        src.set_property("use-string-ids", PropValue::Bool(true))
            .unwrap();
        assert_eq!(
            src.get_property("use-string-ids"),
            Some(PropValue::Bool(true))
        );
        src.set_property("num-buffers", PropValue::Int(12)).unwrap();
        assert_eq!(src.get_property("num-buffers"), Some(PropValue::Int(12)));
        src.set_property("feed-id", PropValue::Str(String::new()))
            .unwrap();
        assert_eq!(src.feed_id, None);
        assert_eq!(
            src.set_property("room-id", PropValue::Bool(true)),
            Err(PropError::Type)
        );
        assert_eq!(
            src.set_property("video-codec", PropValue::Str("theora".into())),
            Err(PropError::Value)
        );
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
            "janusvrsrc name=s janus-endpoint=ws://127.0.0.1:8188 room-id=1234 feed-id=77 use-string-ids=false video-codec=vp8 num-buffers=10  s. ! fakesink  s. ! fakesink",
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
        assert_eq!(fanouts, [NodeKind::FanoutSrc(OUTPUT_COUNT as u8)]);
    }
}

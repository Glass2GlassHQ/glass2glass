//! End-to-end smoke test for the native LiveKit publisher sink (`LiveKitSink`).
//!
//! Pipeline: `PacedH264Src -> LiveKitSink` (join a room, publish H.264 video over
//! one str0m PeerConnection). Ignored by default: it needs a reachable LiveKit
//! server and the sandbox blocks the WebRTC UDP ports, so this is a user-run
//! harness, not a CI gate (like the mediamtx WHIP smoke tests).
//!
//! Recipe (local dev server, docker):
//!
//! ```sh
//! docker run -d --network host --name g2g-livekit livekit/livekit-server --dev
//! # dev credentials: API key `devkey`, secret `secret`.
//! G2G_LIVEKIT_URL=ws://localhost:7880 \
//! G2G_LIVEKIT_API_KEY=devkey G2G_LIVEKIT_API_SECRET=secret \
//! G2G_LIVEKIT_ROOM=g2g-smoke \
//!     cargo test -p g2g-plugins --features webrtc-livekit \
//!     --test livekit_smoke -- --ignored --nocapture
//! ```
//!
//! A green run means the join + publish handshake completed (JoinResponse ->
//! AddTrack -> TrackPublished -> offer/answer -> ICE/DTLS) and access units flowed
//! to str0m. The test also queries the RoomService HTTP API
//! (`ListParticipants`) with an admin token and asserts the publisher participant
//! is in the room with a published track, independent server-side confirmation.

#![cfg(all(target_os = "linux", feature = "webrtc-livekit"))]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_fanin_session, DynSourceLoop, SourceLoop};
use g2g_core::{
    Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, Rate, VideoCodec,
};
use g2g_plugins::livekit_signal::{mint_token, VideoGrant};
use g2g_plugins::livekitsink::LiveKitSink;

/// Build a self-contained Annex-B H.264 stream: every `keyframe_period`th access
/// unit is SPS + PPS + IDR, the rest are P slices. Payload bytes stay in 1..=254
/// so no accidental `00 00 01` start code appears inside a NAL. The bytes are not
/// a decodable picture, but str0m only packetizes NAL units and the LiveKit SFU
/// relays without decoding, so this exercises the real publish path.
/// Split an Annex-B byte stream into NALs, each re-prefixed with a 4-byte start
/// code (same helper as the mediamtx WHIP smoke test).
fn split_annexb(data: &[u8]) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut nals = Vec::new();
    for (k, &s) in starts.iter().enumerate() {
        let payload = s + 3;
        let end = if k + 1 < starts.len() {
            let next = starts[k + 1];
            if next > 0 && data[next - 1] == 0 {
                next - 1
            } else {
                next
            }
        } else {
            data.len()
        };
        let mut nal = vec![0u8, 0, 0, 1];
        nal.extend_from_slice(&data[payload..end]);
        nals.push(nal);
    }
    nals
}

/// Group split NALs into access units: non-VCL NALs (SPS/PPS/SEI/AUD) are
/// prepended to the next VCL NAL (slice types 1/5), which closes the unit.
fn group_access_units(nals: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut aus = Vec::new();
    let mut pending: Vec<u8> = Vec::new();
    for nal in nals {
        let nal_type = nal.get(4).map(|b| b & 0x1f).unwrap_or(0);
        pending.extend_from_slice(&nal);
        if nal_type == 1 || nal_type == 5 {
            aus.push(core::mem::take(&mut pending));
        }
    }
    if !pending.is_empty() {
        aus.push(pending);
    }
    aus
}

fn synthetic_h264(frames: usize) -> Vec<Vec<u8>> {
    fn nal(nal_type: u8, len: usize, salt: usize) -> Vec<u8> {
        let mut v = vec![0u8, 0, 0, 1, nal_type];
        for i in 0..len {
            v.push((((i + salt) % 254) + 1) as u8);
        }
        v
    }
    let mut out = Vec::new();
    for f in 0..frames {
        if f % 30 == 0 {
            let mut au = nal(0x67, 12, f); // SPS
            au.extend(nal(0x68, 6, f)); // PPS
            au.extend(nal(0x65, 2048, f)); // IDR slice
            out.push(au);
        } else {
            out.push(nal(0x41, 768, f)); // P slice
        }
    }
    out
}

/// Loops a synthetic H.264 stream in real time for `duration`, so the sink stays
/// live and publishing across the whole ICE/DTLS handshake window (a flat dump
/// finishes before ICE completes). Mirrors the WHIP smoke test's `PacedH264Src`.
struct PacedH264Src {
    nals: Arc<Vec<Vec<u8>>>,
    duration: Duration,
    width: u32,
    height: u32,
}

impl SourceLoop for PacedH264Src {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(caps_wh(self.width, self.height)))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        let caps = caps_wh(self.width, self.height);
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(caps)).await?;
            let start = Instant::now();
            let mut seq = 0u64;
            let mut idx = 0usize;
            while Instant::now().duration_since(start) < self.duration {
                let nal = self.nals[idx % self.nals.len()].clone();
                idx += 1;
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(nal.into_boxed_slice())),
                    FrameTiming {
                        pts_ns: seq * 5_000_000,
                        ..FrameTiming::default()
                    },
                    seq,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                seq += 1;
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

fn caps_wh(width: u32, height: u32) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(30 << 16),
    }
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Derive the RoomService HTTP base (`http://host:7880`) from the signalling
/// WebSocket URL (`ws://host:7880`).
fn http_base(ws_url: &str) -> String {
    ws_url
        .trim_end_matches('/')
        .replacen("wss://", "https://", 1)
        .replacen("ws://", "http://", 1)
}

/// Query the LiveKit RoomService for the participants in `room`, returning the
/// raw JSON. Uses a hand-minted admin token (roomAdmin grant).
async fn list_participants(
    http_base: &str,
    api_key: &str,
    api_secret: &str,
    room: &str,
) -> Result<String, String> {
    let grant = VideoGrant {
        room_admin: true,
        room: room.to_string(),
        ..Default::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = mint_token(api_key, api_secret, "g2g-admin", &grant, now, 600);
    let url = format!("{http_base}/twirp/livekit.RoomService/ListParticipants");
    let resp = reqwest::Client::new()
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/json")
        .body(format!("{{\"room\":\"{room}\"}}"))
        .send()
        .await
        .map_err(|e| format!("ListParticipants request failed: {e}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!("ListParticipants HTTP {status}: {body}"));
    }
    Ok(body)
}

#[tokio::test]
#[ignore = "needs a LiveKit server (G2G_LIVEKIT_URL + api key/secret + room)"]
async fn livekit_publishes_h264() {
    let (Ok(url), Ok(api_key), Ok(api_secret)) = (
        std::env::var("G2G_LIVEKIT_URL"),
        std::env::var("G2G_LIVEKIT_API_KEY"),
        std::env::var("G2G_LIVEKIT_API_SECRET"),
    ) else {
        eprintln!(
            "skipping: set G2G_LIVEKIT_URL, G2G_LIVEKIT_API_KEY, G2G_LIVEKIT_API_SECRET to run"
        );
        return;
    };
    let room = std::env::var("G2G_LIVEKIT_ROOM").unwrap_or_else(|_| "g2g-smoke".to_string());
    let identity = "g2g-publisher";
    eprintln!("publishing synthetic H.264 -> {url} room={room} as {identity}");

    // `G2G_H264_FIXTURE` swaps in a real (decodable) Annex-B clip, so a browser
    // subscriber in the room renders actual frames; the synthetic stream is
    // structurally valid H.264 that exercises transport but no decoder. The
    // clip's NALs are grouped into access units (SPS/PPS ride with their IDR in
    // one write = one RTP timestamp): an SFU forwards our packetization as-is,
    // and Chrome only treats an IDR as a keyframe when the parameter sets share
    // its frame.
    let nals = match std::env::var("G2G_H264_FIXTURE") {
        Ok(path) => {
            let bytes = std::fs::read(&path).expect("read h264 fixture");
            Arc::new(group_access_units(split_annexb(&bytes)))
        }
        Err(_) => Arc::new(synthetic_h264(300)),
    };

    // Publisher: paced video source fans into the LiveKit sink (1 input).
    let publisher = {
        let (nals, url, room, api_key, api_secret) = (
            nals.clone(),
            url.clone(),
            room.clone(),
            api_key.clone(),
            api_secret.clone(),
        );
        async move {
            // `G2G_PUBLISH_SECS` holds the room live longer for a human /
            // browser subscriber, like the mediamtx browser-publish harness.
            let secs = std::env::var("G2G_PUBLISH_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10);
            let mut src = PacedH264Src {
                nals,
                duration: Duration::from_secs(secs),
                width: 640,
                height: 480,
            };
            let mut sink = LiveKitSink::new(url, room, identity).with_api_key(api_key, api_secret);
            let clock = ZeroClock;
            let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut src];
            let stats = run_fanin_session(sources, &mut sink, &clock, 8).await;
            (stats, sink.frames_sent())
        }
    };

    // Verifier: after the publisher has had time to connect, poll the RoomService
    // API until the participant appears with a published track (or time out).
    let verifier = {
        let (url, room, api_key, api_secret) = (
            url.clone(),
            room.clone(),
            api_key.clone(),
            api_secret.clone(),
        );
        async move {
            let base = http_base(&url);
            let deadline = Instant::now() + Duration::from_secs(9);
            let mut last = String::new();
            while Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(500)).await;
                match list_participants(&base, &api_key, &api_secret, &room).await {
                    Ok(body) => {
                        last = body;
                        // The participant is present with at least one track once
                        // the identity and a track SID (`TR_...`) are in the JSON.
                        if last.contains(identity)
                            && last.contains("\"tracks\"")
                            && last.contains("TR_")
                        {
                            return Ok(last);
                        }
                    }
                    Err(e) => last = e,
                }
            }
            Err(last)
        }
    };

    let secs: u64 = std::env::var("G2G_PUBLISH_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let publisher_fut = tokio::time::timeout(Duration::from_secs(secs + 10), publisher);
    let (pub_res, verified) = tokio::join!(publisher_fut, verifier);
    let (stats, frames_sent) = pub_res.expect("publisher completes in time");

    eprintln!("publisher frames handed to session = {frames_sent}");
    if let Err(e) = &stats {
        eprintln!("publisher session ended with: {e:?}");
    }
    assert!(
        frames_sent > 50,
        "expected a continuous publish, only {frames_sent} access units handed over"
    );

    match verified {
        Ok(body) => eprintln!(
            "RoomService confirmed the publisher + track:\n{}",
            &body[..body.len().min(1200)]
        ),
        Err(last) => panic!(
            "RoomService never showed the publisher with a published track; last response:\n{last}"
        ),
    }
}

/// Two-layer simulcast publish: two paced H.264 sources of different resolutions
/// fan into one [`LiveKitSink`] with `with_simulcast(2)`, so both ride ONE video
/// m-line as rids `h` (pad 0, high) and `q` (pad 1, low). Server-side proof is
/// the RoomService listing the track with two video layers; the SFU logs also
/// show both rid streams binding. Fixtures: `G2G_H264_FIXTURE` (high layer) and
/// `G2G_H264_FIXTURE_LOW` (low layer), both Annex-B.
#[tokio::test]
#[ignore = "needs a LiveKit server + two H.264 fixtures (G2G_H264_FIXTURE / G2G_H264_FIXTURE_LOW)"]
async fn livekit_publishes_simulcast() {
    let (Ok(url), Ok(api_key), Ok(api_secret), Ok(fix_hi), Ok(fix_lo)) = (
        std::env::var("G2G_LIVEKIT_URL"),
        std::env::var("G2G_LIVEKIT_API_KEY"),
        std::env::var("G2G_LIVEKIT_API_SECRET"),
        std::env::var("G2G_H264_FIXTURE"),
        std::env::var("G2G_H264_FIXTURE_LOW"),
    ) else {
        eprintln!("skipping: set G2G_LIVEKIT_URL, api key/secret and both fixtures to run");
        return;
    };
    let room = std::env::var("G2G_LIVEKIT_ROOM").unwrap_or_else(|_| "g2g-simulcast".to_string());
    let identity = "g2g-simulcaster";
    let secs: u64 = std::env::var("G2G_PUBLISH_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    eprintln!("simulcast publish -> {url} room={room} as {identity}");

    let hi = Arc::new(group_access_units(split_annexb(
        &std::fs::read(&fix_hi).expect("read high fixture"),
    )));
    let lo = Arc::new(group_access_units(split_annexb(
        &std::fs::read(&fix_lo).expect("read low fixture"),
    )));
    // Optional third (middle) layer: set `G2G_H264_FIXTURE_MID` to publish 3
    // rids (f/h/q) instead of 2 (h/q).
    let mid = std::env::var("G2G_H264_FIXTURE_MID").ok().map(|p| {
        Arc::new(group_access_units(split_annexb(
            &std::fs::read(&p).expect("read mid fixture"),
        )))
    });

    let publisher = {
        let (url, room, api_key, api_secret) = (
            url.clone(),
            room.clone(),
            api_key.clone(),
            api_secret.clone(),
        );
        async move {
            let mut src_hi = PacedH264Src {
                nals: hi,
                duration: Duration::from_secs(secs),
                width: 640,
                height: 480,
            };
            let mut src_lo = PacedH264Src {
                nals: lo,
                duration: Duration::from_secs(secs),
                width: 320,
                height: 240,
            };
            let mut src_mid = mid.map(|nals| PacedH264Src {
                nals,
                duration: Duration::from_secs(secs),
                width: 480,
                height: 360,
            });
            // `G2G_MAX_SEND_BITRATE` exercises the layer allocator: a cap below
            // the layers' combined nominal rate sheds the top layer live.
            let cap: u64 = std::env::var("G2G_MAX_SEND_BITRATE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let layers = 2 + src_mid.is_some() as usize;
            let mut sink = LiveKitSink::new(url, room, identity)
                .with_api_key(api_key, api_secret)
                .with_simulcast(layers)
                .with_max_send_bitrate(cap);
            let clock = ZeroClock;
            // Pad order is highest resolution first.
            let mut sources: Vec<&mut dyn DynSourceLoop> = Vec::new();
            sources.push(&mut src_hi);
            if let Some(m) = src_mid.as_mut() {
                sources.push(m);
            }
            sources.push(&mut src_lo);
            let stats = run_fanin_session(sources, &mut sink, &clock, 8).await;
            (stats, sink.frames_sent())
        }
    };
    let want_layers = 2 + std::env::var("G2G_H264_FIXTURE_MID").is_ok() as usize;

    // Verifier: the participant's track must list every video layer (the sink
    // announces them in AddTrackRequest, and the SFU updates them from the
    // arriving rid streams).
    let verifier = {
        let (url, room, api_key, api_secret) = (
            url.clone(),
            room.clone(),
            api_key.clone(),
            api_secret.clone(),
        );
        async move {
            let base = http_base(&url);
            let deadline = Instant::now() + Duration::from_secs(9);
            let mut last = String::new();
            while Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(500)).await;
                match list_participants(&base, &api_key, &api_secret, &room).await {
                    Ok(body) => {
                        last = body;
                        let layer_count = last.matches("\"quality\"").count();
                        if last.contains(identity)
                            && last.contains("TR_")
                            && layer_count >= want_layers
                        {
                            return Ok(last);
                        }
                    }
                    Err(e) => last = e,
                }
            }
            Err(last)
        }
    };

    let publisher_fut = tokio::time::timeout(Duration::from_secs(secs + 10), publisher);
    let (pub_res, verified) = tokio::join!(publisher_fut, verifier);
    let (stats, frames_sent) = pub_res.expect("publisher completes in time");
    eprintln!("simulcast publisher frames handed to session = {frames_sent}");
    if let Err(e) = &stats {
        eprintln!("publisher session ended with: {e:?}");
    }
    assert!(
        frames_sent > 100,
        "expected both layers to feed continuously, got {frames_sent}"
    );
    match verified {
        Ok(body) => eprintln!(
            "RoomService confirmed the track with all layers:\n{}",
            &body[..body.len().min(1500)]
        ),
        Err(last) => {
            panic!("RoomService never showed 2 video layers; last response:\n{last}")
        }
    }
}

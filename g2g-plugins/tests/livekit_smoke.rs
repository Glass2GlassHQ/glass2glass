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
use g2g_core::runtime::{
    run_fanin_session, run_fanout_session, run_source_transform_sink, DynSourceLoop, SourceLoop,
};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, ConfigureOutcome, Dim,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate,
    VideoCodec,
};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::livekit_signal::{mint_token, VideoGrant};
use g2g_plugins::livekitsink::LiveKitSink;
use g2g_plugins::livekitsrc::LiveKitSrc;
use g2g_plugins::oggdemux::OggDemux;

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

fn opus_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 48_000,
    }
}

/// Audio analog of `PacedH264Src`: loops a fixture's Opus packets in real time
/// (one 20 ms frame per push), same as the WHIP smoke test's paced audio source.
struct PacedOpusSrc {
    packets: Arc<Vec<Vec<u8>>>,
    duration: Duration,
}

impl SourceLoop for PacedOpusSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(opus_caps()))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(opus_caps())).await?;
            let start = Instant::now();
            let mut seq = 0u64;
            let mut idx = 0usize;
            while Instant::now().duration_since(start) < self.duration {
                let pkt = self.packets[idx % self.packets.len()].clone();
                idx += 1;
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(pkt.into_boxed_slice())),
                    FrameTiming {
                        pts_ns: seq * 20_000_000,
                        ..FrameTiming::default()
                    },
                    seq,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                seq += 1;
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

/// Sink that collects each `DataFrame`'s System-memory payload, used to extract
/// the Opus elementary packets from an Ogg fixture once, before pacing them.
struct CapturingSink {
    out: Arc<std::sync::Mutex<Vec<Vec<u8>>>>,
}

impl AsyncElement for CapturingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                if let MemoryDomain::System(slice) = &frame.domain {
                    self.out.lock().unwrap().push(slice.as_slice().to_vec());
                }
            }
            Ok(())
        })
    }
}

/// Extract the Opus elementary packets from an Ogg-Opus fixture by running the
/// real `FileSrc -> OggDemux` path once (so the paced source loops genuine
/// packets, not synthetic ones).
async fn extract_opus_packets(ogg_path: &str) -> Vec<Vec<u8>> {
    let mut src = FileSrc::new(
        ogg_path,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        },
    );
    let mut demux = OggDemux::new();
    let collected = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut sink = CapturingSink {
        out: collected.clone(),
    };
    let clock = ZeroClock;
    run_source_transform_sink(&mut src, &mut demux, &mut sink, &clock, 8)
        .await
        .expect("ogg->opus extraction should succeed");
    let packets = core::mem::take(&mut *collected.lock().unwrap());
    packets
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
    // `G2G_OPUS_FIXTURE` (Ogg-Opus) adds an Opus audio track alongside the
    // video, so a browser subscriber verifies both payload kinds survive the
    // SFU forwarder.
    let opus = match std::env::var("G2G_OPUS_FIXTURE") {
        Ok(path) => Some(Arc::new(extract_opus_packets(&path).await)),
        Err(_) => None,
    };
    let has_audio = opus.is_some();

    // Publisher: paced video (and optional audio) sources fan into the sink.
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
            let mut asrc = opus.map(|packets| PacedOpusSrc {
                packets,
                duration: Duration::from_secs(secs),
            });
            let mut sink = LiveKitSink::new(url, room, identity).with_api_key(api_key, api_secret);
            if asrc.is_some() {
                sink = sink.with_audio();
            }
            let clock = ZeroClock;
            let mut sources: Vec<&mut dyn DynSourceLoop> = vec![&mut src];
            if let Some(a) = asrc.as_mut() {
                sources.push(a);
            }
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
            let want_tracks = 1 + has_audio as usize;
            let deadline = Instant::now() + Duration::from_secs(9);
            let mut last = String::new();
            while Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(500)).await;
                match list_participants(&base, &api_key, &api_secret, &room).await {
                    Ok(body) => {
                        last = body;
                        // The participant is present once the identity and a
                        // track SID (`TR_...`) per published track are in the
                        // JSON (two SIDs when audio publishes alongside video).
                        if last.contains(identity)
                            && last.contains("\"tracks\"")
                            && last.matches("TR_").count() >= want_tracks
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

/// Sink that counts `DataFrame`s into a shared atomic (same helper as the WHIP
/// smoke test), one per subscriber output.
struct CountingSink {
    frames: Arc<std::sync::atomic::AtomicU64>,
}

impl AsyncElement for CountingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(_) = packet {
                self.frames
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            Ok(())
        })
    }
}

/// M714 LiveKit ingest loopback: `LiveKitSink` publishes A/V into the room and
/// `LiveKitSrc` subscribes to it over the server-offered subscriber PC, so both
/// native elements are validated against the real SFU in one run. Requires the
/// H.264 + Opus fixtures (the subscriber gates video on a decodable keyframe).
#[tokio::test]
#[ignore = "needs a LiveKit server + G2G_H264_FIXTURE + G2G_OPUS_FIXTURE"]
async fn livekit_ingest_loopback() {
    let (Ok(url), Ok(api_key), Ok(api_secret), Ok(fixture)) = (
        std::env::var("G2G_LIVEKIT_URL"),
        std::env::var("G2G_LIVEKIT_API_KEY"),
        std::env::var("G2G_LIVEKIT_API_SECRET"),
        std::env::var("G2G_H264_FIXTURE"),
    ) else {
        eprintln!("skipping: set G2G_LIVEKIT_URL, api key/secret and G2G_H264_FIXTURE");
        return;
    };
    let Ok(opus_fixture) = std::env::var("G2G_OPUS_FIXTURE") else {
        eprintln!("skipping: set G2G_OPUS_FIXTURE");
        return;
    };
    let room = std::env::var("G2G_LIVEKIT_ROOM").unwrap_or_else(|_| "g2g-ingest".to_string());
    let secs: u64 = std::env::var("G2G_PUBLISH_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    eprintln!("ingest loopback -> {url} room={room}");

    let nals = Arc::new(group_access_units(split_annexb(
        &std::fs::read(&fixture).expect("read h264 fixture"),
    )));
    let opus = Arc::new(extract_opus_packets(&opus_fixture).await);

    let publisher = {
        let (url, room, api_key, api_secret) = (
            url.clone(),
            room.clone(),
            api_key.clone(),
            api_secret.clone(),
        );
        async move {
            let mut vsrc = PacedH264Src {
                nals,
                duration: Duration::from_secs(secs),
                width: 640,
                height: 480,
            };
            let mut asrc = PacedOpusSrc {
                packets: opus,
                duration: Duration::from_secs(secs),
            };
            let mut sink = LiveKitSink::new(url, room, "g2g-publisher")
                .with_api_key(api_key, api_secret)
                .with_audio();
            let clock = ZeroClock;
            let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut vsrc, &mut asrc];
            run_fanin_session(sources, &mut sink, &clock, 8).await
        }
    };

    let video_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let audio_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let subscriber = {
        let (url, room, api_key, api_secret) = (url, room, api_key, api_secret);
        let (vc, ac) = (video_count.clone(), audio_count.clone());
        async move {
            // Let the publisher's track land first, then subscribe for the rest
            // of the publish window (bounded so the test always ends).
            tokio::time::sleep(Duration::from_secs(2)).await;
            let mut src = LiveKitSrc::new(url, room, "g2g-subscriber")
                .with_api_key(api_key, api_secret)
                .with_frame_limit(secs.saturating_mul(120));
            let mut vsink = CountingSink { frames: vc };
            let mut asink = CountingSink { frames: ac };
            let clock = ZeroClock;
            let sinks: Vec<&mut dyn g2g_core::element::DynAsyncElement> =
                vec![&mut vsink, &mut asink];
            tokio::time::timeout(
                Duration::from_secs(secs + 5),
                run_fanout_session(&mut src, sinks, &clock, 8),
            )
            .await
        }
    };

    let (pub_res, sub_res) = tokio::join!(publisher, subscriber);
    pub_res.expect("publisher session runs");
    // The subscriber may still be waiting on EOS when the timeout fires; the
    // frame counts are the assertion, not its exit path.
    if let Ok(r) = sub_res {
        r.expect("subscriber session ends cleanly");
    }
    let v = video_count.load(std::sync::atomic::Ordering::SeqCst);
    let a = audio_count.load(std::sync::atomic::Ordering::SeqCst);
    eprintln!("subscriber received video={v} audio={a}");
    assert!(v > 30, "expected a continuous video feed, got {v}");
    assert!(a > 30, "expected a continuous audio feed, got {a}");
}

/// M727 graph-node ingest: the same sink-publish + `LiveKitSrc` subscribe
/// loopback, but the subscriber runs as a `FanoutSrc` GRAPH node
/// (`Graph::add_fanout_src`) feeding two counting sinks, proving the terminal
/// fan-out node against the real SFU.
#[tokio::test]
#[ignore = "needs a LiveKit server + G2G_H264_FIXTURE + G2G_OPUS_FIXTURE"]
async fn livekit_ingest_via_graph_node() {
    let (Ok(url), Ok(api_key), Ok(api_secret), Ok(fixture), Ok(opus_fixture)) = (
        std::env::var("G2G_LIVEKIT_URL"),
        std::env::var("G2G_LIVEKIT_API_KEY"),
        std::env::var("G2G_LIVEKIT_API_SECRET"),
        std::env::var("G2G_H264_FIXTURE"),
        std::env::var("G2G_OPUS_FIXTURE"),
    ) else {
        eprintln!("skipping: set G2G_LIVEKIT_URL, api key/secret and both fixtures");
        return;
    };
    let room = std::env::var("G2G_LIVEKIT_ROOM").unwrap_or_else(|_| "g2g-ingest-graph".to_string());
    let secs: u64 = std::env::var("G2G_PUBLISH_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let nals = Arc::new(group_access_units(split_annexb(
        &std::fs::read(&fixture).expect("read h264 fixture"),
    )));
    let opus = Arc::new(extract_opus_packets(&opus_fixture).await);

    let publisher = {
        let (url, room, api_key, api_secret) = (
            url.clone(),
            room.clone(),
            api_key.clone(),
            api_secret.clone(),
        );
        async move {
            let mut vsrc = PacedH264Src {
                nals,
                duration: Duration::from_secs(secs),
                width: 640,
                height: 480,
            };
            let mut asrc = PacedOpusSrc {
                packets: opus,
                duration: Duration::from_secs(secs),
            };
            let mut sink = LiveKitSink::new(url, room, "g2g-publisher")
                .with_api_key(api_key, api_secret)
                .with_audio();
            let clock = ZeroClock;
            let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut vsrc, &mut asrc];
            run_fanin_session(sources, &mut sink, &clock, 8).await
        }
    };

    let video_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let audio_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let subscriber = {
        let (vc, ac) = (video_count.clone(), audio_count.clone());
        async move {
            tokio::time::sleep(Duration::from_secs(2)).await;
            use g2g_core::runtime::{run_graph, GraphNode};
            let mut g: g2g_core::Graph<GraphNode> = g2g_core::Graph::new();
            let src = g.add_fanout_src(
                GraphNode::fanout_source(
                    LiveKitSrc::new(url, room, "g2g-graph-subscriber")
                        .with_api_key(api_key, api_secret)
                        .with_frame_limit(secs.saturating_mul(120)),
                ),
                2,
            );
            let vsink = g.add_sink(GraphNode::element(CountingSink { frames: vc }));
            let asink = g.add_sink(GraphNode::element(CountingSink { frames: ac }));
            g.link(src.output(0), vsink).unwrap();
            g.link(src.output(1), asink).unwrap();
            tokio::time::timeout(Duration::from_secs(secs + 5), run_graph(g, &ZeroClock, 8)).await
        }
    };

    let (pub_res, sub_res) = tokio::join!(publisher, subscriber);
    pub_res.expect("publisher session runs");
    if let Ok(r) = sub_res {
        r.expect("graph subscriber ends cleanly");
    }
    let v = video_count.load(std::sync::atomic::Ordering::SeqCst);
    let a = audio_count.load(std::sync::atomic::Ordering::SeqCst);
    eprintln!("graph subscriber received video={v} audio={a}");
    assert!(v > 30, "continuous video via the graph node, got {v}");
    assert!(a > 30, "continuous audio via the graph node, got {a}");
}

/// M713 encoder fan graph: the simulcast layers are ENCODED live inside one
/// graph instead of fed from pre-encoded fixtures. One paced raw source tees
/// into per-layer branches (`videoscale -> ffmpegenc`) that end on the
/// [`LiveKitSink`] as a terminal fan-in node (`Graph::add_fanin_sink`), so a
/// remote per-rid PLI travels the graph edges back to the matching encoder.
/// No fixtures needed: the encoders make the video.
#[cfg(feature = "ffmpeg")]
mod fan_graph {
    use super::*;
    use g2g_core::runtime::{run_graph, GraphNode};
    use g2g_core::{Graph, RawVideoFormat};
    use g2g_plugins::ffmpegenc::{Backend, FfmpegH264Enc};
    use g2g_plugins::videoscale::VideoScale;

    fn i420_caps(width: u32, height: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    /// Paced raw I420 source: a moving vertical bar at 30 fps in real time, so
    /// the encoders have live motion to encode across the whole publish window.
    struct PacedI420Src {
        width: u32,
        height: u32,
        duration: Duration,
    }

    impl SourceLoop for PacedI420Src {
        type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
        type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

        fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
            ready(Ok(i420_caps(self.width, self.height)))
        }
        fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
            Ok(ConfigureOutcome::Accepted)
        }
        fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
            let (w, h) = (self.width as usize, self.height as usize);
            let caps = i420_caps(self.width, self.height);
            Box::pin(async move {
                out.push(PipelinePacket::CapsChanged(caps)).await?;
                let start = Instant::now();
                let mut seq = 0u64;
                while Instant::now().duration_since(start) < self.duration {
                    let mut buf = vec![128u8; w * h * 3 / 2];
                    let bar = (seq as usize * 4) % w;
                    for row in buf[..w * h].chunks_exact_mut(w) {
                        row.fill(16);
                        let end = (bar + w / 8).min(w);
                        row[bar..end].fill(235);
                    }
                    let frame = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
                        FrameTiming {
                            pts_ns: seq * 33_333_333,
                            ..FrameTiming::default()
                        },
                        seq,
                    );
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                    seq += 1;
                    tokio::time::sleep(Duration::from_millis(33)).await;
                }
                out.push(PipelinePacket::Eos).await?;
                Ok(seq)
            })
        }
    }

    #[tokio::test]
    #[ignore = "needs a LiveKit server (G2G_LIVEKIT_URL + api key/secret); encodes live via ffmpeg"]
    async fn livekit_publishes_encoder_fan_graph() {
        let (Ok(url), Ok(api_key), Ok(api_secret)) = (
            std::env::var("G2G_LIVEKIT_URL"),
            std::env::var("G2G_LIVEKIT_API_KEY"),
            std::env::var("G2G_LIVEKIT_API_SECRET"),
        ) else {
            eprintln!("skipping: set G2G_LIVEKIT_URL, api key/secret to run");
            return;
        };
        let room =
            std::env::var("G2G_LIVEKIT_ROOM").unwrap_or_else(|_| "g2g-fan-graph".to_string());
        let identity = "g2g-fan-publisher";
        let secs: u64 = std::env::var("G2G_PUBLISH_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(15);
        eprintln!("encoder fan graph publish -> {url} room={room} as {identity}");

        let publisher = {
            let (url, room, api_key, api_secret) = (
                url.clone(),
                room.clone(),
                api_key.clone(),
                api_secret.clone(),
            );
            async move {
                let mut g: Graph<GraphNode> = Graph::new();
                let src = g.add_source(GraphNode::source(PacedI420Src {
                    width: 640,
                    height: 480,
                    duration: Duration::from_secs(secs),
                }));
                let tee = g.add_tee(2);
                let enc_hi = g.add_transform(GraphNode::element(
                    FfmpegH264Enc::new()
                        .with_backend(Backend::Software)
                        .with_bitrate(1_200_000),
                ));
                let scale_lo = g.add_transform(GraphNode::element(VideoScale::new(320, 240)));
                let enc_lo = g.add_transform(GraphNode::element(
                    FfmpegH264Enc::new()
                        .with_backend(Backend::Software)
                        .with_bitrate(300_000),
                ));
                // `G2G_MAX_SEND_BITRATE` caps the BWE estimate so the layer
                // allocator sheds live (the shed layer's encoder then idles on
                // the Bitrate(0) hint, M722).
                let cap: u64 = std::env::var("G2G_MAX_SEND_BITRATE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let sink = LiveKitSink::new(url, room, identity)
                    .with_api_key(api_key, api_secret)
                    .with_simulcast(2)
                    .with_max_send_bitrate(cap);
                let session = g.add_fanin_sink(GraphNode::muxer(sink), 2);
                g.link(src, tee.input()).unwrap();
                g.link(tee.out(0), enc_hi).unwrap();
                g.link(enc_hi, session.input(0)).unwrap();
                g.link(tee.out(1), scale_lo).unwrap();
                g.link(scale_lo, enc_lo).unwrap();
                g.link(enc_lo, session.input(1)).unwrap();
                run_graph(g, &ZeroClock, 4).await
            }
        };

        // Verifier: the RoomService must list the track with both video layers
        // (the sink announces them; the SFU binds them from the arriving rids).
        let verifier = {
            let (url, room, api_key, api_secret) = (
                url.clone(),
                room.clone(),
                api_key.clone(),
                api_secret.clone(),
            );
            async move {
                let base = http_base(&url);
                let deadline = Instant::now() + Duration::from_secs(secs.min(20));
                let mut last = String::new();
                while Instant::now() < deadline {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    match list_participants(&base, &api_key, &api_secret, &room).await {
                        Ok(body) => {
                            last = body;
                            let layer_count = last.matches("\"quality\"").count();
                            if last.contains(identity) && last.contains("TR_") && layer_count >= 2 {
                                return Ok(last);
                            }
                        }
                        Err(e) => last = e,
                    }
                }
                Err(last)
            }
        };

        let publisher_fut = tokio::time::timeout(Duration::from_secs(secs + 15), publisher);
        let (pub_res, verified) = tokio::join!(publisher_fut, verifier);
        let stats = pub_res.expect("publisher completes in time");
        match &stats {
            Ok(s) => eprintln!(
                "fan graph run: emitted {} raw frames, session consumed {}",
                s.frames_emitted, s.frames_consumed
            ),
            Err(e) => eprintln!("fan graph run ended with: {e:?}"),
        }
        let stats = stats.expect("fan graph runs to Eos");
        assert!(
            stats.frames_consumed > 100,
            "expected both encoded layers to feed continuously, got {}",
            stats.frames_consumed
        );
        match verified {
            Ok(body) => eprintln!(
                "RoomService confirmed the encoded track with both layers:\n{}",
                &body[..body.len().min(1500)]
            ),
            Err(last) => {
                panic!("RoomService never showed 2 video layers; last response:\n{last}")
            }
        }
    }
}

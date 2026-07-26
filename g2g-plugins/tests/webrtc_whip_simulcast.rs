//! Live multi-rid validation of the WHIP simulcast session
//! ([`WebRtcSessionSink::with_simulcast`]): publishes three H.264 layers as one
//! rid-tagged m-line (`f` / `h` / `q`, highest resolution on pad 0) to a WHIP
//! server that ingests client simulcast natively, and asserts the server reports
//! all three rids with growing packet counts. mediamtx cannot ingest client
//! simulcast and LiveKit's WHIP ingress transcodes a single layer; Broadcast Box
//! (pion) does, so it is the reference peer here.
//!
//! Ignored by default: it needs a running server and three fixtures.
//!
//! ```sh
//! docker run --rm --network host -e HTTP_ADDRESS=:8085 -e UDP_MUX_PORT=8085 \
//!     --name g2g-bbox glimesh/broadcast-box
//!
//! for wh in 640x480 480x360 320x240; do
//!   ffmpeg -f lavfi -i testsrc=size=$wh:rate=30:duration=10 \
//!          -c:v libx264 -profile:v baseline -bsf:v h264_mp4toannexb \
//!          -f h264 /tmp/sim_$wh.h264
//! done
//!
//! # The per-stream `?key=` status is a summary with no track list; the bare
//! # listing is the one carrying `videoTracks[].rid` / `packetsReceived`.
//! G2G_WHIP_URL=http://localhost:8085/api/whip G2G_WHIP_TOKEN=g2gtest \
//! G2G_BBOX_STATUS_URL=http://localhost:8085/api/status \
//! G2G_H264_FIXTURE=/tmp/sim_640x480.h264 \
//! G2G_H264_FIXTURE_MID=/tmp/sim_480x360.h264 \
//! G2G_H264_FIXTURE_LOW=/tmp/sim_320x240.h264 \
//!     cargo test -p g2g-plugins --features webrtc --test webrtc_whip_simulcast \
//!     -- --ignored --nocapture
//! ```
//!
//! Against the public instance (`https://b.siobud.com/api/whip`) the status API
//! is not assumed: leave `G2G_BBOX_STATUS_URL` unset and the run only proves the
//! publish (ICE / DTLS / SRTP over real NAT) completed.

#![cfg(all(target_os = "linux", feature = "webrtc"))]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_fanin_session, DynSourceLoop, SourceLoop};
use g2g_core::{
    Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, Rate, VideoCodec,
};
use g2g_plugins::webrtcsession::WebRtcSessionSink;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
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

/// Split an Annex-B stream into NAL units, each re-prefixed with a 4-byte start
/// code.
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
        let mut nal = std::vec![0u8, 0, 0, 1];
        nal.extend_from_slice(&data[payload..end]);
        nals.push(nal);
    }
    nals
}

/// Group NALs into access units (parameter sets ride with the slice that
/// follows), so one pushed frame is one complete picture.
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

fn load_layer(path: &str) -> Arc<Vec<Vec<u8>>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"));
    let aus = group_access_units(split_annexb(&bytes));
    assert!(!aus.is_empty(), "fixture {path} had no access units");
    Arc::new(aus)
}

/// Loops a layer's access units at 30 fps for `duration`, announcing that
/// layer's resolution (the rid restrictions in the offer come from these caps).
struct PacedLayerSrc {
    aus: Arc<Vec<Vec<u8>>>,
    duration: Duration,
    width: u32,
    height: u32,
}

impl SourceLoop for PacedLayerSrc {
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
            while start.elapsed() < self.duration {
                let au = self.aus[idx % self.aus.len()].clone();
                idx += 1;
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
                    FrameTiming {
                        pts_ns: seq * 33_000_000,
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

/// Minimal HTTP/1.0 GET so the status probe needs no HTTP client dependency.
/// Plain `http://` only, which is all the local status API is.
async fn http_get(url: &str) -> Option<String> {
    let rest = url.strip_prefix("http://")?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let addr = if authority.contains(':') {
        authority.to_string()
    } else {
        std::format!("{authority}:80")
    };
    let mut stream = tokio::net::TcpStream::connect(&addr).await.ok()?;
    let req = std::format!("GET {path} HTTP/1.0\r\nHost: {authority}\r\nAccept: */*\r\n\r\n");
    stream.write_all(req.as_bytes()).await.ok()?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.ok()?;
    let text = String::from_utf8_lossy(&buf).into_owned();
    Some(text.split("\r\n\r\n").nth(1)?.to_string())
}

/// Every `"<key>": <number>` in a JSON blob, in document order. Enough to read
/// the status API's per-layer counters without a JSON dependency.
fn numbers_for_key(body: &str, key: &str) -> Vec<u64> {
    let needle = std::format!("\"{key}\":");
    let mut out = Vec::new();
    for part in body.split(&needle).skip(1) {
        let digits: String = part
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if let Ok(n) = digits.parse::<u64>() {
            out.push(n);
        }
    }
    out
}

/// Every `"<key>": "<string>"` in a JSON blob, in document order.
fn strings_for_key(body: &str, key: &str) -> Vec<String> {
    let needle = std::format!("\"{key}\":");
    let mut out = Vec::new();
    for part in body.split(&needle).skip(1) {
        let rest = part.trim_start();
        if let Some(rest) = rest.strip_prefix('"') {
            if let Some(end) = rest.find('"') {
                out.push(rest[..end].to_string());
            }
        }
    }
    out
}

/// Narrow a multi-stream status listing to the entry for `key`, so a stream
/// somebody else is publishing cannot satisfy the assertions.
fn scope_to_stream<'a>(body: &'a str, key: &str) -> &'a str {
    let start = match body.find(&std::format!("\"streamKey\":\"{key}\"")) {
        Some(i) => i,
        None => return body,
    };
    let rest = &body[start..];
    match rest[1..].find("\"streamKey\":") {
        Some(i) => &rest[..=i],
        None => rest,
    }
}

/// The rids the status body reports for the stream, with their packet counters.
/// The tracks list them in document order as `rid` then `packetsReceived`.
fn layers_from_status(body: &str, key: Option<&str>) -> Vec<(String, u64)> {
    let scoped = match key {
        Some(k) => scope_to_stream(body, k),
        None => body,
    };
    let rids = strings_for_key(scoped, "rid");
    let packets = numbers_for_key(scoped, "packetsReceived");
    rids.into_iter()
        .zip(packets)
        .filter(|(rid, _)| !rid.is_empty())
        .collect()
}

#[tokio::test]
#[ignore = "needs a simulcast-ingesting WHIP server (Broadcast Box) + three H.264 fixtures"]
async fn whip_simulcast_publishes_three_rids() {
    let (Ok(url), Ok(fix_hi), Ok(fix_mid), Ok(fix_lo)) = (
        std::env::var("G2G_WHIP_URL"),
        std::env::var("G2G_H264_FIXTURE"),
        std::env::var("G2G_H264_FIXTURE_MID"),
        std::env::var("G2G_H264_FIXTURE_LOW"),
    ) else {
        eprintln!(
            "skipping: set G2G_WHIP_URL and the three fixtures \
             (G2G_H264_FIXTURE / _MID / _LOW) to run"
        );
        return;
    };
    let token = std::env::var("G2G_WHIP_TOKEN").ok();
    let status_url = std::env::var("G2G_BBOX_STATUS_URL").ok();
    let secs: u64 = std::env::var("G2G_PUBLISH_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    eprintln!(
        "simulcast publish -> {url} for {secs}s (token: {})",
        token.is_some()
    );

    let (hi, mid, lo) = (
        load_layer(&fix_hi),
        load_layer(&fix_mid),
        load_layer(&fix_lo),
    );
    let duration = Duration::from_secs(secs);

    let publisher = async {
        let mut src_hi = PacedLayerSrc {
            aus: hi,
            duration,
            width: 640,
            height: 480,
        };
        let mut src_mid = PacedLayerSrc {
            aus: mid,
            duration,
            width: 480,
            height: 360,
        };
        let mut src_lo = PacedLayerSrc {
            aus: lo,
            duration,
            width: 320,
            height: 240,
        };
        // Three video pads, no audio: each pad is one rid-tagged layer on a
        // single m-line, pad 0 highest (`f`, then `h`, then `q`).
        let mut sink = WebRtcSessionSink::new(url.clone()).with_inputs(3);
        if let Some(token) = token.clone() {
            sink = sink.with_bearer(token);
        }
        let clock = ZeroClock;
        let sources: Vec<&mut dyn DynSourceLoop> =
            std::vec![&mut src_hi, &mut src_mid, &mut src_lo];
        let stats = run_fanin_session(sources, &mut sink, &clock, 8).await;
        (stats, sink.frames_sent())
    };

    // Poll the server's view while the publish runs: the last sample that
    // reported layers is the verdict, so a late-starting rid still counts. The
    // most recent raw body is kept either way, so a failure prints what the
    // server actually said.
    let watcher = async {
        let Some(status_url) = status_url.clone() else {
            return (String::new(), Vec::new(), Vec::new());
        };
        let mut first = Vec::new();
        let mut last = Vec::new();
        let mut body = String::new();
        let deadline = Instant::now() + duration;
        while Instant::now() < deadline {
            tokio::time::sleep(Duration::from_secs(1)).await;
            let Some(b) = http_get(&status_url).await else {
                continue;
            };
            let layers = layers_from_status(&b, token.as_deref());
            body = b;
            if layers.is_empty() {
                continue;
            }
            if first.is_empty() {
                first = layers.clone();
            }
            last = layers;
        }
        (body, first, last)
    };

    let ((stats, frames_sent), (body, first, last)) = tokio::join!(publisher, watcher);
    let stats = stats.expect("simulcast WHIP publish ran");
    eprintln!(
        "published {} AUs ({} frames handed to the session)",
        stats.frames_emitted, frames_sent
    );
    assert!(
        frames_sent > 0,
        "the session should have taken frames from all three layers"
    );

    let Some(status_url) = status_url else {
        eprintln!("no G2G_BBOX_STATUS_URL: publish-only run, no layer assertions");
        return;
    };
    eprintln!("status ({status_url}) last body: {body}");
    eprintln!("first layers: {first:?}");
    eprintln!("last layers:  {last:?}");

    let mut rids: Vec<&str> = last.iter().map(|(r, _)| r.as_str()).collect();
    rids.sort_unstable();
    rids.dedup();
    assert_eq!(
        rids.len(),
        3,
        "expected three distinct rids ingested, got {rids:?} (raw: {body})"
    );
    // Every layer must have carried packets, and the stream must still be
    // growing between the first and last sample (media, not just a negotiated
    // description).
    for (rid, packets) in &last {
        assert!(*packets > 0, "rid {rid} received no packets");
    }
    // Match by rid: the server lists the tracks in no fixed order.
    let grew = last.iter().any(|(rid, after)| {
        first
            .iter()
            .find(|(r, _)| r == rid)
            .is_some_and(|(_, before)| after > before)
    });
    assert!(
        grew,
        "packet counts should grow across samples: {first:?} -> {last:?}"
    );
}

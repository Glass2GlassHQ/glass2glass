//! Live pipeline dashboard transport (dev tooling, `observe` feature).
//!
//! Serves the [`Observer`] telemetry + [`BusMessage`] events of a running graph
//! to a browser. One TCP port carries both: a plain HTTP `GET /` returns the
//! self-contained dashboard page ([`INDEX_HTML`]); a WebSocket upgrade gets a
//! stream of JSON messages, a `telemetry` snapshot every [`TICK`] plus an `event`
//! per bus message. The JSON is built here (serde_json) so `g2g-core` stays
//! serde-free; the page is dependency-free vanilla JS/SVG so there is no build
//! step.
//!
//! Wire protocol (each WS frame is one JSON object, discriminated by `type`):
//! - `{"type":"telemetry","uptime_ns":N,"nodes":[..],"edges":[{"from":i,"to":j,
//!   "caps":"<gst-string>"|null,"packets":N,"bytes":N,"drops":N,"blocked_ns":N}]}`
//!   where each node is `{"id","name","role","proc":{..}|null,
//!   "push_wait":{..}|null,"transit":{..}|null,"fill_mean_pct","fill_max_pct"}`,
//!   `caps` is the edge's negotiated caps, and the counters are that edge's live
//!   traffic (packets and payload bytes that crossed, frames dropped by a leaky
//!   link, and nanoseconds the producer spent blocked on a full link).
//!   The same message carries `"journey"`: one frame's measured path, or `null`
//!   when none assembles. Its shape is `{"sequence":N,"total_ns":N,
//!   "frame_period_ns":N,"capacity":N,"floor_ns":N,"truncated":bool,
//!   "stages":[{"node":i,"name":"...","wait_ns":N,"work_ns":N,
//!   "blocked_ns":N}]}`, where the stages are upstream-first for that one
//!   sequence id, `work_ns` is compute with `blocked_ns` (that frame's
//!   downstream backpressure) taken out, `total_ns` is the measured end to end,
//!   and `floor_ns` is `2 * capacity * frame_period_ns`, the queueing floor to
//!   compare it against.
//! - `{"type":"event","kind":"eos"|"error"|...,...}` (see [`event_json`]).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast;
use tokio_tungstenite::tungstenite::Message;

use g2g_core::runtime::{LinkInterceptor, Observer, ProbeAction, ProbeSlot, TelemetrySnapshot};
use g2g_core::{BusMessage, Caps, PipelinePacket};

use crate::preview::packet_preview;

/// Per-connection edge subscriptions: edge index -> (the edge's probe slot with
/// our interceptor installed, the shared latest preview it writes).
type EdgeSubs = HashMap<u64, (ProbeSlot, Arc<Mutex<Option<Value>>>)>;

/// Minimum spacing between preview samples on a tapped edge, so the hot path
/// pays a preview conversion at most a few times a second.
const PREVIEW_INTERVAL: Duration = Duration::from_millis(500);

/// A [`LinkInterceptor`] that samples an edge's packets into a shared latest
/// preview, rate-limited. Installed on a subscribe, removed on unsubscribe;
/// always passes the packet through (never drops).
struct PreviewTap {
    caps: Caps,
    latest: Arc<Mutex<Option<Value>>>,
    last_ns: AtomicU64,
    interval_ns: u64,
}

impl LinkInterceptor for PreviewTap {
    fn on_packet(&self, packet: &PipelinePacket) -> ProbeAction {
        let now = g2g_core::metrics::monotonic_ns();
        let last = self.last_ns.load(Ordering::Relaxed);
        if now.saturating_sub(last) >= self.interval_ns
            && self
                .last_ns
                .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            if let Some(v) = packet_preview(packet, &self.caps) {
                if let Ok(mut slot) = self.latest.lock() {
                    *slot = Some(v);
                }
            }
        }
        ProbeAction::Pass
    }
}

/// The self-contained dashboard page, served on `GET /`.
pub const INDEX_HTML: &str = include_str!("../../tools/dashboard/index.html");

/// Telemetry push cadence.
const TICK: Duration = Duration::from_millis(250);

/// Serialize a telemetry snapshot to the wire JSON string: the shared
/// [`toolingjson::telemetry_json`] shape plus the dashboard's `type` tag.
pub fn snapshot_json(snap: &TelemetrySnapshot) -> String {
    let mut v = crate::toolingjson::telemetry_json(snap);
    v["type"] = Value::from("telemetry");
    v.to_string()
}

/// Serialize a bus message to the wire JSON string, or `None` for messages the
/// dashboard does not surface (the heavy `StreamCollection` / `Tag` payloads).
pub fn event_json(msg: &BusMessage) -> Option<String> {
    let v = match msg {
        BusMessage::StreamStart => json!({"kind": "stream-start"}),
        BusMessage::Eos => json!({"kind": "eos"}),
        BusMessage::Info(s) => json!({"kind": "info", "text": s}),
        BusMessage::Error(e) => json!({"kind": "error", "text": format!("{e:?}")}),
        BusMessage::Warning(e) => json!({"kind": "warning", "text": format!("{e:?}")}),
        BusMessage::NegotiationFailed(f) => {
            json!({"kind": "negotiation-failed", "text": format!("{f:?}")})
        }
        BusMessage::StateChanged { old, new } => {
            json!({"kind": "state-changed", "old": format!("{old:?}"), "new": format!("{new:?}")})
        }
        BusMessage::AsyncDone => json!({"kind": "async-done"}),
        BusMessage::Qos {
            running_time_ns,
            jitter_ns,
            processed,
            dropped,
        } => json!({
            "kind": "qos",
            "running_time_ns": running_time_ns,
            "jitter_ns": jitter_ns,
            "processed": processed,
            "dropped": dropped,
        }),
        BusMessage::Buffering { percent, element } => {
            json!({"kind": "buffering", "percent": percent, "element": element})
        }
        BusMessage::DurationChanged { duration_ns } => {
            json!({"kind": "duration-changed", "duration_ns": duration_ns})
        }
        BusMessage::StreamsSelected { ids } => json!({"kind": "streams-selected", "ids": ids}),
        BusMessage::Custom(code) => json!({"kind": "custom", "code": code}),
        // Skip the large structured payloads the dashboard has no view for.
        BusMessage::Tag { .. } | BusMessage::StreamTag { .. } | BusMessage::StreamCollection(_) => {
            return None
        }
    };
    let mut obj = v;
    obj["type"] = json!("event");
    Some(obj.to_string())
}

/// Serve the dashboard on `host:port` until the process ends (`host` is
/// `127.0.0.1` for loopback-only, `0.0.0.0` for all interfaces). Each
/// connection either gets the static page (plain HTTP) or a live telemetry +
/// event stream (WebSocket). `events` is the fan-out channel a bus-drain task
/// publishes [`event_json`] strings into; every WS client subscribes to it.
///
/// Runs forever (the accept loop); drive it with `tokio::select!` against the
/// pipeline run future so it is dropped when the pipeline finishes.
pub async fn serve(
    observer: Observer,
    events: broadcast::Sender<String>,
    host: &str,
    port: u16,
) -> std::io::Result<()> {
    let listener = TcpListener::bind((host, port)).await?;
    serve_on(listener, observer, events).await
}

/// The accept loop, split from the bind so a test can supply an ephemeral-port
/// listener and learn the address.
async fn serve_on(
    listener: TcpListener,
    observer: Observer,
    events: broadcast::Sender<String>,
) -> std::io::Result<()> {
    loop {
        let (stream, _) = listener.accept().await?;
        let observer = observer.clone();
        let ev_rx = events.subscribe();
        tokio::spawn(async move {
            let _ = handle_conn(stream, observer, ev_rx).await;
        });
    }
}

async fn handle_conn(
    mut stream: TcpStream,
    observer: Observer,
    ev_rx: broadcast::Receiver<String>,
) -> std::io::Result<()> {
    // Peek the request head (non-destructive) to branch HTTP vs WebSocket without
    // consuming the bytes the tungstenite handshake needs to re-read.
    let mut peek = [0u8; 1024];
    let n = stream.peek(&mut peek).await?;
    let head = String::from_utf8_lossy(&peek[..n]).to_ascii_lowercase();

    if head.contains("upgrade: websocket") {
        let ws = match tokio_tungstenite::accept_async(stream).await {
            Ok(ws) => ws,
            Err(_) => return Ok(()),
        };
        stream_telemetry(ws, observer, ev_rx).await;
    } else {
        // Static page. Drain the peeked request bytes, then write and close.
        let mut scratch = vec![0u8; n];
        let _ = stream.read(&mut scratch).await;
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            INDEX_HTML.len(),
            INDEX_HTML
        );
        stream.write_all(resp.as_bytes()).await?;
        stream.flush().await?;
    }
    Ok(())
}

async fn stream_telemetry(
    ws: tokio_tungstenite::WebSocketStream<TcpStream>,
    observer: Observer,
    mut ev_rx: broadcast::Receiver<String>,
) {
    let (mut tx, mut rx) = ws.split();
    let mut ticker = tokio::time::interval(TICK);
    // Per-connection edge subscriptions: edge index -> (slot with our interceptor
    // installed, shared latest preview).
    let mut subs: EdgeSubs = HashMap::new();
    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let json = snapshot_json(&observer.snapshot());
                if tx.send(Message::Text(json)).await.is_err() {
                    break;
                }
                // Flush any fresh edge previews sampled since the last tick.
                let mut failed = false;
                for (edge, (_slot, latest)) in subs.iter() {
                    let taken = latest.lock().ok().and_then(|mut l| l.take());
                    if let Some(preview) = taken {
                        let msg = json!({ "type": "preview", "edge": edge, "preview": preview });
                        if tx.send(Message::Text(msg.to_string())).await.is_err() {
                            failed = true;
                            break;
                        }
                    }
                }
                if failed {
                    break;
                }
            }
            ev = ev_rx.recv() => {
                match ev {
                    Ok(json) => {
                        if tx.send(Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                    // Lagged: the client fell behind the event fan-out; skip the
                    // gap and keep streaming. Closed: the bus drainer is gone
                    // (pipeline ended), but keep pushing telemetry until the run
                    // future drops us.
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => {}
                }
            }
            // Inbound frames: subscribe / unsubscribe control, plus close.
            inbound = rx.next() => {
                match inbound {
                    Some(Ok(Message::Text(t))) => handle_client_msg(&t, &observer, &mut subs),
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                }
            }
        }
    }
    // Remove our interceptors so a disconnect leaves no taps sampling the graph.
    for (_edge, (slot, _)) in subs {
        slot.remove();
    }
}

/// Handle a `{type:"subscribe"|"unsubscribe","edge":N}` control message from the
/// dashboard, installing or removing an edge preview tap.
fn handle_client_msg(text: &str, observer: &Observer, subs: &mut EdgeSubs) {
    let Ok(msg) = serde_json::from_str::<Value>(text) else {
        return;
    };
    let edge = match msg.get("edge").and_then(|e| e.as_u64()) {
        Some(e) => e,
        None => return,
    };
    match msg.get("type").and_then(|t| t.as_str()) {
        Some("subscribe") => {
            if subs.contains_key(&edge) {
                return;
            }
            let (Some(slot), Some(caps)) = (
                observer.edge_probe(edge as usize),
                observer.edge_caps(edge as usize),
            ) else {
                return;
            };
            let latest = Arc::new(Mutex::new(None));
            let tap = PreviewTap {
                caps,
                latest: latest.clone(),
                last_ns: AtomicU64::new(0),
                interval_ns: PREVIEW_INTERVAL.as_nanos() as u64,
            };
            slot.install(Arc::new(tap));
            subs.insert(edge, (slot, latest));
        }
        Some("unsubscribe") => {
            if let Some((slot, _)) = subs.remove(&edge) {
                slot.remove();
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use g2g_core::metrics::LatencySnapshot;
    use g2g_core::runtime::NodeRole;
    use g2g_core::runtime::{
        EdgeCounts, EdgeInfo, ElementLatency, FrameJourney, JourneyStage, NodeTelemetry,
        TelemetrySnapshot,
    };

    #[test]
    fn snapshot_json_shape() {
        // Build a snapshot directly from public fields (the runner produces the
        // same shape via the Observer); this keeps the JSON test independent of
        // the runner and the crate-private registration path.
        let snap = TelemetrySnapshot {
            uptime_ns: 1_000,
            nodes: vec![
                NodeTelemetry {
                    id: 0,
                    name: String::from("src0"),
                    role: NodeRole::Source,
                    latency: None,
                },
                NodeTelemetry {
                    id: 1,
                    name: String::from("decode0"),
                    role: NodeRole::Transform,
                    latency: Some(ElementLatency {
                        name: String::from("decode0"),
                        proc: LatencySnapshot {
                            count: 0,
                            mean_ns: 0,
                            max_ns: 0,
                            p50_ns: 0,
                            p95_ns: 0,
                            p99_ns: 0,
                        },
                        push_wait: LatencySnapshot {
                            count: 0,
                            mean_ns: 0,
                            max_ns: 0,
                            p50_ns: 0,
                            p95_ns: 0,
                            p99_ns: 0,
                        },
                        transit: LatencySnapshot {
                            count: 0,
                            mean_ns: 0,
                            max_ns: 0,
                            p50_ns: 0,
                            p95_ns: 0,
                            p99_ns: 0,
                        },
                        fill_mean_pct: 30,
                        fill_max_pct: 50,
                        presentation: None,
                    }),
                },
            ],
            edges: vec![EdgeInfo {
                from: 0,
                to: 1,
                caps: None,
                observed_caps: None,
                counts: EdgeCounts {
                    packets: 12,
                    bytes: 480,
                    drops: 2,
                    blocked_ns: 5_000,
                },
            }],
            journey: None,
        };
        let json = snapshot_json(&snap);
        let v: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "telemetry");
        assert_eq!(v["nodes"].as_array().unwrap().len(), 2);
        // Source: no probe -> proc is null.
        assert_eq!(v["nodes"][0]["role"], "source");
        assert!(v["nodes"][0]["proc"].is_null());
        // Transform: probe present -> proc object + fill reflects the value.
        assert_eq!(v["nodes"][1]["role"], "transform");
        assert_eq!(v["nodes"][1]["proc"]["count"], 0);
        assert_eq!(v["nodes"][1]["fill_max_pct"], 50);
        assert_eq!(v["edges"][0]["from"], 0);
        assert_eq!(v["edges"][0]["to"], 1);
        // Live per-edge counters ride the same message.
        assert_eq!(v["edges"][0]["packets"], 12);
        assert_eq!(v["edges"][0]["bytes"], 480);
        assert_eq!(v["edges"][0]["drops"], 2);
        assert_eq!(v["edges"][0]["blocked_ns"], 5_000);
        // No frame journey assembled yet -> the field is present but null.
        assert!(v["journey"].is_null());
    }

    #[test]
    fn snapshot_json_carries_the_frame_journey() {
        let snap = TelemetrySnapshot {
            uptime_ns: 1_000,
            nodes: vec![],
            edges: vec![],
            journey: Some(FrameJourney {
                sequence: 42,
                stages: vec![
                    JourneyStage {
                        node: 1,
                        name: String::from("videoscale0"),
                        wait_ns: 300,
                        work_ns: 1_200,
                        blocked_ns: 900,
                    },
                    JourneyStage {
                        node: 2,
                        name: String::from("fakesink0"),
                        wait_ns: 100,
                        work_ns: 400,
                        blocked_ns: 0,
                    },
                ],
                total_ns: 2_000,
                frame_period_ns: 33_000_000,
                capacity: 2,
                floor_ns: 132_000_000,
                truncated: false,
            }),
        };
        let v: Value = serde_json::from_str(&snapshot_json(&snap)).unwrap();
        let j = &v["journey"];
        assert_eq!(j["sequence"], 42);
        assert_eq!(j["total_ns"], 2_000);
        assert_eq!(j["frame_period_ns"], 33_000_000);
        assert_eq!(j["capacity"], 2);
        assert_eq!(j["floor_ns"], 132_000_000);
        assert_eq!(j["truncated"], false);
        let stages = j["stages"].as_array().unwrap();
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0]["node"], 1);
        assert_eq!(stages[0]["name"], "videoscale0");
        assert_eq!(stages[0]["wait_ns"], 300);
        assert_eq!(
            stages[0]["blocked_ns"], 900,
            "backpressure is its own segment"
        );
        assert_eq!(stages[1]["work_ns"], 400);
        assert_eq!(stages[1]["blocked_ns"], 0, "a sink pushes nowhere");
    }

    #[test]
    fn event_json_covers_common_messages() {
        let eos = event_json(&BusMessage::Eos).unwrap();
        let v: Value = serde_json::from_str(&eos).unwrap();
        assert_eq!(v["type"], "event");
        assert_eq!(v["kind"], "eos");

        let buf = event_json(&BusMessage::Buffering {
            percent: 75,
            element: Some("videoconvert0".into()),
        })
        .unwrap();
        let v: Value = serde_json::from_str(&buf).unwrap();
        assert_eq!(v["kind"], "buffering");
        assert_eq!(v["percent"], 75);
        assert_eq!(v["element"], "videoconvert0");

        // Heavy payloads are skipped.
        assert!(event_json(&BusMessage::Tag {
            tags: Default::default(),
            program: None,
        })
        .is_none());
    }

    #[tokio::test]
    async fn websocket_streams_telemetry_and_events() {
        use futures_util::StreamExt;

        let observer = Observer::new();
        let (ev_tx, _) = broadcast::channel::<String>(16);
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_on(listener, observer, ev_tx.clone()));

        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/"))
            .await
            .unwrap();

        // The server pushes a telemetry snapshot on its tick.
        let telemetry = loop {
            match ws.next().await.unwrap().unwrap() {
                Message::Text(t) => {
                    let v: Value = serde_json::from_str(&t).unwrap();
                    if v["type"] == "telemetry" {
                        break v;
                    }
                }
                _ => continue,
            }
        };
        assert!(telemetry["nodes"].is_array());

        // An event published to the fan-out reaches the client.
        ev_tx.send(event_json(&BusMessage::Eos).unwrap()).unwrap();
        let event = loop {
            match ws.next().await.unwrap().unwrap() {
                Message::Text(t) => {
                    let v: Value = serde_json::from_str(&t).unwrap();
                    if v["type"] == "event" {
                        break v;
                    }
                }
                _ => continue,
            }
        };
        assert_eq!(event["kind"], "eos");
    }

    #[tokio::test]
    async fn edge_subscribe_streams_a_video_preview() {
        use crate::registry::default_registry;
        use futures_util::StreamExt;
        use g2g_core::runtime::{parse_launch, run_graph_observed};
        use g2g_core::PipelineClock;

        struct ZeroClock;
        impl PipelineClock for ZeroClock {
            fn now_ns(&self) -> u64 {
                0
            }
        }

        let reg = default_registry();
        // A forever source so the run keeps flowing while we tap an edge.
        let graph = parse_launch(
            &reg,
            "videotestsrc ! videoscale width=32 height=24 ! fakesink",
        )
        .expect("parses");
        let observer = Observer::new();
        let (ev_tx, _) = broadcast::channel::<String>(16);
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_on(listener, observer.clone(), ev_tx));

        let clock = ZeroClock;
        let client = async {
            let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/"))
                .await
                .unwrap();
            let mut subscribed = false;
            loop {
                let Some(Ok(Message::Text(t))) = ws.next().await else {
                    continue;
                };
                let v: Value = serde_json::from_str(&t).unwrap();
                // Subscribe to edge 0 (videotestsrc -> videoscale) once the run
                // has registered the graph's edges.
                if !subscribed
                    && v["type"] == "telemetry"
                    && !v["edges"].as_array().unwrap().is_empty()
                {
                    ws.send(Message::Text(r#"{"type":"subscribe","edge":0}"#.into()))
                        .await
                        .unwrap();
                    subscribed = true;
                }
                if v["type"] == "preview" {
                    return v;
                }
            }
        };

        let run = run_graph_observed(graph, &clock, 2, &observer, None);
        tokio::select! {
            _ = run => panic!("run ended before a preview arrived"),
            preview = tokio::time::timeout(Duration::from_secs(15), client) => {
                let preview = preview.expect("preview within deadline");
                assert_eq!(preview["edge"], 0);
                // videotestsrc emits packed RGBA, so the tap yields a thumbnail.
                assert_eq!(preview["preview"]["kind"], "video");
                assert!(preview["preview"]["rgba"].as_array().unwrap().len() >= 4);
            }
        }
    }

    /// A fan-in arm is tappable like any other edge (M849): subscribing to an
    /// edge whose destination is the muxer streams previews of the frames that
    /// input contributes, even when the muxer forwards a different input.
    #[tokio::test]
    async fn muxer_input_edge_subscribe_streams_a_preview() {
        use crate::registry::default_registry;
        use futures_util::StreamExt;
        use g2g_core::runtime::{parse_launch, run_graph_observed};
        use g2g_core::PipelineClock;

        struct ZeroClock;
        impl PipelineClock for ZeroClock {
            fn now_ns(&self) -> u64 {
                0
            }
        }

        let reg = default_registry();
        // input-selector forwards input 0 only, so the sink's sequence check
        // stays happy while both arms keep pushing into the fan-in node.
        let graph = parse_launch(
            &reg,
            "videotestsrc ! m.   videotestsrc ! m.   input-selector name=m ! fakesink",
        )
        .expect("parses");
        let observer = Observer::new();
        let (ev_tx, _) = broadcast::channel::<String>(16);
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_on(listener, observer.clone(), ev_tx));

        let clock = ZeroClock;
        let client = async {
            let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/"))
                .await
                .unwrap();
            let mut target: Option<u64> = None;
            loop {
                let Some(Ok(Message::Text(t))) = ws.next().await else {
                    continue;
                };
                let v: Value = serde_json::from_str(&t).unwrap();
                // Pick the second arm into the fan-in node: the input the muxer
                // does NOT forward, so only the edge tap can see its frames.
                if target.is_none() && v["type"] == "telemetry" {
                    let nodes = v["nodes"].as_array().unwrap();
                    let mux = nodes.iter().find(|n| n["role"] == "muxer");
                    let (Some(mux), edges) = (mux, v["edges"].as_array().unwrap()) else {
                        continue;
                    };
                    let into_mux: Vec<u64> = edges
                        .iter()
                        .enumerate()
                        .filter(|(_, e)| e["to"] == mux["id"])
                        .map(|(i, _)| i as u64)
                        .collect();
                    let Some(&edge) = into_mux.get(1) else {
                        continue;
                    };
                    ws.send(Message::Text(
                        json!({"type": "subscribe", "edge": edge}).to_string(),
                    ))
                    .await
                    .unwrap();
                    target = Some(edge);
                }
                if v["type"] == "preview" {
                    return v;
                }
            }
        };

        let run = run_graph_observed(graph, &clock, 2, &observer, None);
        tokio::select! {
            _ = run => panic!("run ended before a preview arrived"),
            preview = tokio::time::timeout(Duration::from_secs(15), client) => {
                let preview = preview.expect("preview within deadline");
                assert_eq!(preview["preview"]["kind"], "video");
                assert!(preview["preview"]["rgba"].as_array().unwrap().len() >= 4);
            }
        }
    }
}

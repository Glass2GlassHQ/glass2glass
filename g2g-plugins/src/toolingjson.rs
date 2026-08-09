//! JSON tooling shared by `g2g-inspect --json` and the `g2g-mcp` server: the
//! registry dump, a launch-line negotiation check, and a bounded pipeline run
//! (optionally streaming live telemetry while it runs). Kept in one place so the
//! two front-ends serialize the same shapes. serde_json only (no `g2g-core`
//! serde), matching the dashboard's split.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::time::Duration;

use serde_json::{json, Value};

use g2g_core::caps::CapsSet;
use g2g_core::dot::kind_label;
use g2g_core::runtime::{
    negotiate_graph_explained, parse_launch, run_graph, run_graph_observed, ElementDoc, GraphNode,
    NegotiateError, NegotiationFailure, NodeRole, Observer, Registry, RunStats, TelemetrySnapshot,
};
use g2g_core::{G2gError, Graph, NodeId};

use crate::clock::WallClock;

/// Steady-state link depth for a `launch` probe run (matches `g2g-launch`).
const LINK_CAPACITY: usize = 4;

/// One element's introspection as JSON: identity, role, pad caps, and each
/// property's machine type / range / default.
pub fn element_json(d: &ElementDoc) -> Value {
    let props: Vec<Value> = d
        .properties
        .iter()
        .map(|p| {
            json!({
                "name": p.name,
                "blurb": p.blurb,
                "type": p.type_label,
                "default": p.default,
                "range": p.range.as_ref().map(|(a, b)| json!([a, b])),
                "enum_values": p.enum_values,
                "readable": p.readable,
                "writable": p.writable,
            })
        })
        .collect();
    json!({
        "name": d.name,
        "long_name": d.long_name,
        "klass": d.klass,
        "description": d.description,
        "author": d.author,
        "role": d.role,
        "caps": d.caps,
        "pads": d.pads,
        "properties": props,
    })
}

/// The registry (or one element) as `{"elements":[...]}`. `Err` names an unknown
/// element.
pub fn registry_json(reg: &Registry, name: Option<&str>) -> Result<Value, String> {
    let docs = match name {
        Some(n) => match reg.describe(n) {
            Some(d) => alloc::vec![d],
            None => return Err(format!("No such element: {n}")),
        },
        None => reg.describe_all(),
    };
    let elements: Vec<Value> = docs.iter().map(element_json).collect();
    Ok(json!({ "elements": elements }))
}

/// Parse + negotiate a launch line without running it. On success reports every
/// node (index + name) and the negotiated caps per edge (with the edge's
/// endpoint node indices); on a solve conflict, the structured failure naming
/// the offending link.
pub async fn validate_json(reg: &Registry, line: &str) -> Value {
    let graph = match parse_launch(reg, line) {
        Ok(g) => g,
        Err(e) => return json!({ "ok": false, "stage": "parse", "error": format!("{e}") }),
    };
    match negotiate_graph_explained(graph).await {
        Ok((vg, edge_caps, _mem)) => {
            let nodes: Vec<Value> = (0..vg.node_count())
                .map(|i| {
                    let node = NodeId(i as u32);
                    let name = vg
                        .element(node)
                        .map(|e| e.log_category())
                        .unwrap_or_else(|| kind_label(vg.kind(node)));
                    json!({ "index": i, "name": name })
                })
                .collect();
            let edges: Vec<Value> = vg
                .edges()
                .iter()
                .zip(edge_caps.iter())
                .map(|(e, caps)| {
                    json!({
                        "from": e.src.node.0,
                        "to": e.dst.node.0,
                        "caps": caps.to_gst_string(),
                    })
                })
                .collect();
            json!({ "ok": true, "nodes": nodes, "edges": edges })
        }
        Err(NegotiateError::Setup(e)) => {
            json!({ "ok": false, "stage": "setup", "error": format!("{e:?}") })
        }
        Err(NegotiateError::Solve(nf)) => {
            json!({ "ok": false, "stage": "negotiate", "failure": failure_json(&nf) })
        }
    }
}

/// Structured form of a [`NegotiationFailure`]: the conflict kind plus the node
/// indices it names, so a caller (dashboard / MCP client) can highlight the
/// offending link. An `EmptyLink` also carries what each end still allowed
/// (`upstream_caps` / `downstream_caps`, one gst string per alternative) when
/// the solver captured both sides.
fn failure_json(nf: &NegotiationFailure) -> Value {
    match nf {
        NegotiationFailure::EmptyLink {
            upstream,
            downstream,
            conflict,
        } => {
            let mut v =
                json!({ "kind": "empty-link", "upstream": upstream, "downstream": downstream });
            if let Some(c) = conflict {
                v["upstream_caps"] = json!(set_strings(&c.upstream));
                v["downstream_caps"] = json!(set_strings(&c.downstream));
            }
            v
        }
        NegotiationFailure::Unfixable {
            upstream,
            downstream,
        } => {
            json!({ "kind": "unfixable", "upstream": upstream, "downstream": downstream })
        }
        NegotiationFailure::EndpointShapeMismatch { index } => {
            json!({ "kind": "endpoint-shape-mismatch", "index": index })
        }
        NegotiationFailure::Degenerate => json!({ "kind": "degenerate" }),
        NegotiationFailure::Cyclic => json!({ "kind": "cyclic" }),
        NegotiationFailure::NoConsistentFixation => json!({ "kind": "no-consistent-fixation" }),
        NegotiationFailure::MixedLegacyAndNative => json!({ "kind": "mixed-legacy-and-native" }),
    }
}

/// One gst caps string per alternative in the set.
fn set_strings(set: &CapsSet) -> Vec<String> {
    set.alternatives()
        .iter()
        .map(|c| c.to_gst_string())
        .collect()
}

/// Live telemetry hookup for a bounded run: how often to sample the running
/// graph's [`Observer`], and where each sample's JSON goes (the caller wraps it
/// in whatever envelope its transport uses, e.g. an MCP progress notification).
pub struct TelemetryTap<'a> {
    interval: Duration,
    on_snapshot: &'a dyn Fn(Value),
}

impl<'a> TelemetryTap<'a> {
    pub fn new(interval: Duration, on_snapshot: &'a dyn Fn(Value)) -> Self {
        Self {
            interval,
            on_snapshot,
        }
    }
}

impl fmt::Debug for TelemetryTap<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TelemetryTap")
            .field("interval", &self.interval)
            .finish_non_exhaustive()
    }
}

/// Run a launch line for up to `secs` seconds and report the resulting
/// [`RunStats`]. A pipeline that finishes early returns full telemetry; one that
/// hits the deadline reports `timed_out` (a forever source has no final stats).
/// With a `tap`, live snapshots stream out while the run is still going.
pub async fn launch_json(
    reg: &Registry,
    line: &str,
    secs: u64,
    tap: Option<TelemetryTap<'_>>,
) -> Value {
    let graph = match parse_launch(reg, line) {
        Ok(g) => g,
        Err(e) => return json!({ "ok": false, "stage": "parse", "error": format!("{e}") }),
    };
    run_json(graph, secs, tap).await
}

/// Build a declarative document (JSON, or YAML with the `declarative-yaml`
/// build) into a graph and run it exactly as [`launch_json`] runs a launch line:
/// same deadline, stats, and optional live telemetry.
#[cfg(feature = "declarative")]
pub async fn document_json(
    reg: &Registry,
    doc: &str,
    yaml: bool,
    secs: u64,
    tap: Option<TelemetryTap<'_>>,
) -> Value {
    let built = if yaml {
        #[cfg(feature = "declarative-yaml")]
        {
            crate::declarative::from_yaml(reg, doc)
        }
        #[cfg(not(feature = "declarative-yaml"))]
        {
            return json!({
                "ok": false,
                "stage": "build",
                "error": "YAML documents need the `declarative-yaml` build feature",
            });
        }
    } else {
        crate::declarative::from_json(reg, doc)
    };
    match built {
        Ok(graph) => run_json(graph, secs, tap).await,
        Err(e) => json!({ "ok": false, "stage": "build", "error": format!("{e}") }),
    }
}

/// Run a built graph under a deadline, optionally sampling live telemetry into
/// `tap`. Shared by the launch-line and declarative-document entry points so both
/// report the same shape.
async fn run_json(graph: Graph<GraphNode>, secs: u64, tap: Option<TelemetryTap<'_>>) -> Value {
    let clock = WallClock::new();
    let deadline = Duration::from_secs(secs.max(1));
    let outcome = match tap {
        None => tokio::time::timeout(deadline, run_graph(graph, &clock, LINK_CAPACITY))
            .await
            .ok(),
        Some(tap) => {
            let observer = Observer::new();
            let run = run_graph_observed(graph, &clock, LINK_CAPACITY, &observer, None);
            let mut run = core::pin::pin!(run);
            let start = tokio::time::Instant::now();
            loop {
                let left = deadline.saturating_sub(start.elapsed());
                if left.is_zero() {
                    break None;
                }
                // drive the run in interval-sized slices; each expiry is a
                // sampling point, so no second timer task is needed
                match tokio::time::timeout(tap.interval.min(left), run.as_mut()).await {
                    Ok(r) => break Some(r),
                    Err(_) => (tap.on_snapshot)(telemetry_json(&observer.snapshot())),
                }
            }
        }
    };
    run_result_json(outcome)
}

/// `None` is the deadline, `Some` the run's own result.
fn run_result_json(outcome: Option<Result<RunStats, G2gError>>) -> Value {
    match outcome {
        Some(Ok(stats)) => json!({ "ok": true, "stats": stats_json(&stats) }),
        Some(Err(e)) => json!({ "ok": false, "stage": "run", "error": format!("{e:?}") }),
        None => {
            json!({ "ok": true, "timed_out": true, "note": "deadline reached; forever source has no final stats" })
        }
    }
}

/// A live [`TelemetrySnapshot`] as JSON: the same per-node latency / per-edge
/// caps and traffic-counter fields the dashboard streams over its WebSocket.
pub fn telemetry_json(snap: &TelemetrySnapshot) -> Value {
    let nodes: Vec<Value> = snap
        .nodes
        .iter()
        .map(|n| {
            let proc = n.latency.as_ref().map(|l| {
                json!({
                    "count": l.proc.count,
                    "mean_ns": l.proc.mean_ns,
                    "p50_ns": l.proc.p50_ns,
                    "p95_ns": l.proc.p95_ns,
                    "p99_ns": l.proc.p99_ns,
                    "max_ns": l.proc.max_ns,
                })
            });
            // Time blocked pushing into the output link (M947), already out of
            // `proc`. Null for a node the runner attributes no output push to
            // (every sink).
            let push_wait = n
                .latency
                .as_ref()
                .filter(|l| l.push_wait.max_ns > 0)
                .map(|l| {
                    json!({
                        "count": l.push_wait.count,
                        "p50_ns": l.push_wait.p50_ns,
                        "p99_ns": l.push_wait.p99_ns,
                        "max_ns": l.push_wait.max_ns,
                    })
                });
            // Input-link queue-residency (the "wait" half of the latency
            // waterfall). Null when the node's input edge is not instrumented.
            let transit = n.latency.as_ref().filter(|l| l.transit.count > 0).map(|l| {
                json!({
                    "count": l.transit.count,
                    "p50_ns": l.transit.p50_ns,
                    "p99_ns": l.transit.p99_ns,
                    "max_ns": l.transit.max_ns,
                })
            });
            let (fill_mean, fill_max) = n
                .latency
                .as_ref()
                .map(|l| (l.fill_mean_pct, l.fill_max_pct))
                .unwrap_or((0, 0));
            json!({
                "id": n.id,
                "name": n.name,
                "role": role_str(n.role),
                "proc": proc,
                "push_wait": push_wait,
                "transit": transit,
                "fill_mean_pct": fill_mean,
                "fill_max_pct": fill_max,
            })
        })
        .collect();
    let edges: Vec<Value> = snap
        .edges
        .iter()
        .map(|e| {
            json!({
                "from": e.from,
                "to": e.to,
                "caps": e.caps,
                "packets": e.counts.packets,
                "bytes": e.counts.bytes,
                "drops": e.counts.drops,
                "blocked_ns": e.counts.blocked_ns,
            })
        })
        .collect();
    // One frame's path across the linear stages (M851), beside the aggregate
    // per-stage distributions above.
    let journey = snap.journey.as_ref().map(|j| {
        let stages: Vec<Value> = j
            .stages
            .iter()
            .map(|s| {
                json!({
                    "node": s.node,
                    "name": s.name,
                    "wait_ns": s.wait_ns,
                    "work_ns": s.work_ns,
                    "blocked_ns": s.blocked_ns,
                })
            })
            .collect();
        json!({
            "sequence": j.sequence,
            "total_ns": j.total_ns,
            "frame_period_ns": j.frame_period_ns,
            "capacity": j.capacity,
            "floor_ns": j.floor_ns,
            "truncated": j.truncated,
            "stages": stages,
        })
    });
    json!({
        "uptime_ns": snap.uptime_ns,
        "nodes": nodes,
        "edges": edges,
        "journey": journey,
    })
}

fn role_str(role: NodeRole) -> &'static str {
    match role {
        NodeRole::Source => "source",
        NodeRole::Transform => "transform",
        NodeRole::Sink => "sink",
        NodeRole::Tee => "tee",
        NodeRole::Muxer => "muxer",
    }
}

/// A `RunStats` summary as JSON: frame counts plus the measured per-element
/// `process()` p50/p99 and input-link fill.
pub fn stats_json(stats: &RunStats) -> Value {
    let per: Vec<Value> = stats
        .per_element
        .iter()
        .map(|e| {
            json!({
                "name": e.name,
                "proc_count": e.proc.count,
                "proc_p50_ns": e.proc.p50_ns,
                "proc_p99_ns": e.proc.p99_ns,
                "transit_p50_ns": e.transit.p50_ns,
                "transit_p99_ns": e.transit.p99_ns,
                "fill_mean_pct": e.fill_mean_pct,
                "fill_max_pct": e.fill_max_pct,
            })
        })
        .collect();
    json!({
        "frames_emitted": stats.frames_emitted,
        "frames_consumed": stats.frames_consumed,
        "frames_dropped": stats.frames_dropped,
        "per_element": per,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::default_registry;

    #[test]
    fn registry_json_all_and_one() {
        let reg = default_registry();
        let all = registry_json(&reg, None).unwrap();
        assert!(all["elements"].as_array().unwrap().len() > 10);

        let one = registry_json(&reg, Some("videoscale")).unwrap();
        let els = one["elements"].as_array().unwrap();
        assert_eq!(els.len(), 1);
        assert_eq!(els[0]["name"], "videoscale");

        assert!(registry_json(&reg, Some("nope")).is_err());
    }

    #[tokio::test]
    async fn validate_ok_reports_per_edge_caps() {
        let reg = default_registry();
        let ok = validate_json(
            &reg,
            "videotestsrc ! videoscale width=64 height=48 ! fakesink",
        )
        .await;
        assert_eq!(ok["ok"], true);
        let edges = ok["edges"].as_array().unwrap();
        assert!(edges.len() >= 2);
        // Each edge names its endpoints and the negotiated caps.
        assert!(edges[0]["from"].is_number() && edges[0]["to"].is_number());
        assert!(edges[0]["caps"].as_str().unwrap().contains("video"));
    }

    #[tokio::test]
    async fn validate_ok_names_every_node() {
        let reg = default_registry();
        let ok = validate_json(&reg, "videotestsrc ! videoconvert ! fakesink").await;
        let nodes = ok["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 3);
        // Indices are the same ones the edges reference, and each carries a name.
        let names: Vec<&str> = nodes.iter().map(|n| n["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["VideoTestSrc", "VideoConvert", "FakeSink"]);
        for (i, n) in nodes.iter().enumerate() {
            assert_eq!(n["index"], i);
        }
    }

    #[tokio::test]
    async fn validate_parse_error_is_reported() {
        let reg = default_registry();
        let bad = validate_json(&reg, "nosuchelement ! fakesink").await;
        assert_eq!(bad["ok"], false);
        assert_eq!(bad["stage"], "parse");
    }

    #[tokio::test]
    async fn validate_caps_conflict_names_the_link() {
        // Force a negotiation conflict: pin a capsfilter to a format videotestsrc
        // cannot produce so the solver empties that link.
        let reg = default_registry();
        let bad = validate_json(&reg, "videotestsrc ! audio/x-raw,format=S16LE ! fakesink").await;
        assert_eq!(bad["ok"], false);
        // Either the parser rejects the audio caps on a video src, or the solve
        // empties the link; if it reached the solver, the failure is structured.
        if bad["stage"] == "negotiate" {
            assert_eq!(bad["failure"]["kind"], "empty-link");
        }
    }

    #[tokio::test]
    async fn validate_caps_conflict_reports_both_candidate_sets() {
        let reg = default_registry();
        let bad = validate_json(&reg, "videotestsrc ! audio/x-raw,format=S16LE ! fakesink").await;
        assert_eq!(bad["stage"], "negotiate");
        let f = &bad["failure"];
        assert_eq!(f["kind"], "empty-link");
        assert_eq!(f["upstream"], 0);
        assert_eq!(f["downstream"], 1);
        // Both sides' candidate sets, structurally: what the source offered and
        // what the capsfilter demanded.
        let up = f["upstream_caps"].as_array().expect("upstream set");
        let down = f["downstream_caps"].as_array().expect("downstream set");
        assert!(
            up.iter()
                .all(|c| c.as_str().unwrap().starts_with("video/x-raw")),
            "source offers raw video, got {up:?}"
        );
        assert!(
            down.iter()
                .all(|c| c.as_str().unwrap().starts_with("audio/x-raw")),
            "capsfilter demands raw audio, got {down:?}"
        );
    }

    #[tokio::test]
    async fn launch_finite_returns_stats() {
        let reg = default_registry();
        let out = launch_json(&reg, "videotestsrc num-buffers=4 ! fakesink", 10, None).await;
        assert_eq!(out["ok"], true);
        assert_eq!(out["stats"]["frames_consumed"], 4);
    }

    #[cfg(feature = "declarative-yaml")]
    #[tokio::test]
    async fn document_yaml_runs_and_returns_stats() {
        let reg = default_registry();
        let doc = "
nodes:
  - { id: src,  element: videotestsrc, props: { num-buffers: 5 } }
  - { id: sink, element: fakesink }
edges:
  - { from: src, to: sink }
";
        let out = document_json(&reg, doc, true, 10, None).await;
        assert_eq!(out["ok"], true, "{out}");
        assert_eq!(out["stats"]["frames_consumed"], 5);
    }

    #[cfg(feature = "declarative")]
    #[tokio::test]
    async fn document_build_error_is_reported() {
        let reg = default_registry();
        let bad = document_json(&reg, r#"{"nodes":[],"edges":[]}"#, false, 5, None).await;
        assert_eq!(bad["ok"], false);
        assert_eq!(bad["stage"], "build");
    }

    #[tokio::test]
    async fn telemetry_tap_streams_mid_run_snapshots() {
        use std::sync::Mutex;

        let reg = default_registry();
        let seen: Mutex<Vec<Value>> = Mutex::new(Vec::new());
        let sink = |v: Value| seen.lock().unwrap().push(v);
        // Unbounded source: the run only ends at the deadline, so the tap has to
        // produce snapshots while frames are still flowing.
        let out = launch_json(
            &reg,
            "videotestsrc ! fakesink",
            1,
            Some(TelemetryTap::new(Duration::from_millis(20), &sink)),
        )
        .await;
        assert_eq!(out["timed_out"], true);

        let seen = seen.into_inner().unwrap();
        assert!(!seen.is_empty(), "tap saw no mid-run snapshot");
        let last = seen.last().unwrap();
        assert_eq!(last["nodes"].as_array().unwrap().len(), 2);
        let edge = &last["edges"][0];
        assert!(
            edge["packets"].as_u64().unwrap() > 0,
            "live edge counters should advance mid-run, got {edge}"
        );
        assert!(edge["caps"].as_str().unwrap().contains("video"));
    }
}

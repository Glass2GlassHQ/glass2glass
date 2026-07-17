//! JSON tooling shared by `g2g-inspect --json` and the `g2g-mcp` server: the
//! registry dump, a launch-line negotiation check, and a bounded pipeline run.
//! Kept in one place so the two front-ends serialize the same shapes. serde_json
//! only (no `g2g-core` serde), matching the dashboard's split.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use serde_json::{json, Value};

use g2g_core::runtime::{
    negotiate_graph_explained, parse_launch, run_graph, ElementDoc, NegotiateError,
    NegotiationFailure, Registry, RunStats,
};

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

/// Parse + negotiate a launch line without running it. On success reports the
/// negotiated caps per edge (with the edge's endpoint node indices); on a solve
/// conflict, the structured failure naming the offending link.
pub async fn validate_json(reg: &Registry, line: &str) -> Value {
    let graph = match parse_launch(reg, line) {
        Ok(g) => g,
        Err(e) => return json!({ "ok": false, "stage": "parse", "error": format!("{e}") }),
    };
    match negotiate_graph_explained(graph).await {
        Ok((vg, edge_caps, _mem)) => {
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
            json!({ "ok": true, "edges": edges })
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
/// offending link.
fn failure_json(nf: &NegotiationFailure) -> Value {
    match nf {
        NegotiationFailure::EmptyLink { upstream, downstream } => {
            json!({ "kind": "empty-link", "upstream": upstream, "downstream": downstream })
        }
        NegotiationFailure::Unfixable { upstream, downstream } => {
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

/// Run a launch line for up to `secs` seconds and report the resulting
/// [`RunStats`]. A pipeline that finishes early returns full telemetry; one that
/// hits the deadline reports `timed_out` (a forever source has no final stats).
pub async fn launch_json(reg: &Registry, line: &str, secs: u64) -> Value {
    let graph = match parse_launch(reg, line) {
        Ok(g) => g,
        Err(e) => return json!({ "ok": false, "stage": "parse", "error": format!("{e}") }),
    };
    let clock = WallClock::new();
    let run = run_graph(graph, &clock, LINK_CAPACITY);
    match tokio::time::timeout(core::time::Duration::from_secs(secs.max(1)), run).await {
        Ok(Ok(stats)) => json!({ "ok": true, "stats": stats_json(&stats) }),
        Ok(Err(e)) => json!({ "ok": false, "stage": "run", "error": format!("{e:?}") }),
        Err(_) => json!({ "ok": true, "timed_out": true, "note": "deadline reached; forever source has no final stats" }),
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
        let ok = validate_json(&reg, "videotestsrc ! videoscale width=64 height=48 ! fakesink").await;
        assert_eq!(ok["ok"], true);
        let edges = ok["edges"].as_array().unwrap();
        assert!(edges.len() >= 2);
        // Each edge names its endpoints and the negotiated caps.
        assert!(edges[0]["from"].is_number() && edges[0]["to"].is_number());
        assert!(edges[0]["caps"].as_str().unwrap().contains("video"));
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
        let bad = validate_json(
            &reg,
            "videotestsrc ! audio/x-raw,format=S16LE ! fakesink",
        )
        .await;
        assert_eq!(bad["ok"], false);
        // Either the parser rejects the audio caps on a video src, or the solve
        // empties the link; if it reached the solver, the failure is structured.
        if bad["stage"] == "negotiate" {
            assert_eq!(bad["failure"]["kind"], "empty-link");
        }
    }

    #[tokio::test]
    async fn launch_finite_returns_stats() {
        let reg = default_registry();
        let out = launch_json(&reg, "videotestsrc num-buffers=4 ! fakesink", 10).await;
        assert_eq!(out["ok"], true);
        assert_eq!(out["stats"]["frames_consumed"], 4);
    }
}

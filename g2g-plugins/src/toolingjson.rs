//! JSON tooling shared by `g2g-inspect --json` and the `g2g-mcp` server: the
//! registry dump, a launch-line negotiation check, and a bounded pipeline run.
//! Kept in one place so the two front-ends serialize the same shapes. serde_json
//! only (no `g2g-core` serde), matching the dashboard's split.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use serde_json::{json, Value};

use g2g_core::runtime::{negotiate_graph, parse_launch, run_graph, ElementDoc, Registry, RunStats};

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

/// Parse + negotiate a launch line without running it. Reports whether the graph
/// is buildable and, on success, the negotiated caps per edge.
pub async fn validate_json(reg: &Registry, line: &str) -> Value {
    let graph = match parse_launch(reg, line) {
        Ok(g) => g,
        Err(e) => return json!({ "ok": false, "stage": "parse", "error": format!("{e}") }),
    };
    match negotiate_graph(graph).await {
        Ok((_, edge_caps, _mem)) => {
            let caps: Vec<String> = edge_caps.iter().map(|c| c.to_gst_string()).collect();
            json!({ "ok": true, "edge_caps": caps })
        }
        Err(e) => json!({ "ok": false, "stage": "negotiate", "error": format!("{e:?}") }),
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
    async fn validate_ok_and_bad() {
        let reg = default_registry();
        let ok = validate_json(&reg, "videotestsrc ! videoscale width=64 height=48 ! fakesink").await;
        assert_eq!(ok["ok"], true);
        assert!(ok["edge_caps"].as_array().unwrap().len() >= 2);

        let bad = validate_json(&reg, "nosuchelement ! fakesink").await;
        assert_eq!(bad["ok"], false);
    }

    #[tokio::test]
    async fn launch_finite_returns_stats() {
        let reg = default_registry();
        let out = launch_json(&reg, "videotestsrc num-buffers=4 ! fakesink", 10).await;
        assert_eq!(out["ok"], true);
        assert_eq!(out["stats"]["frames_consumed"], 4);
    }
}

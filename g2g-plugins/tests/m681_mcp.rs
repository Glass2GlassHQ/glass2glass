//! M681 MCP server: `g2g-mcp` speaks JSON-RPC 2.0 over stdio and exposes the
//! inspect / validate / launch / run_graph tools for agent-driven dev, plus
//! live-telemetry progress notifications for the two run tools. Drives the built
//! binary end to end (the tool logic is unit-tested in `toolingjson`; this checks
//! the JSON-RPC framing).
//!
//! Needs `observe,multi-thread` (`declarative-yaml` for the `run_graph` test):
//! `cargo test -p g2g-plugins --features observe,multi-thread,declarative-yaml
//! --test m681_mcp`.
#![cfg(all(feature = "tooling-json", feature = "multi-thread"))]

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

/// Feed the JSON-RPC request lines to `g2g-mcp` and split its stdout into
/// (responses, notifications): a notification carries no `id`.
fn session(requests: &[&str]) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_g2g-mcp"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn g2g-mcp");
    {
        let mut stdin = child.stdin.take().unwrap();
        for r in requests {
            writeln!(stdin, "{r}").unwrap();
        }
        // Drop stdin so the server's stdin loop ends and it exits.
    }
    let out = BufReader::new(child.stdout.take().unwrap());
    let (responses, notifications) = out
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(&l.unwrap()).expect("message is JSON"))
        .partition(|m| m.get("id").is_some());
    child.wait().unwrap();
    (responses, notifications)
}

/// The tool result payload of a response, parsed back out of its text block.
fn payload(resp: &serde_json::Value) -> serde_json::Value {
    serde_json::from_str(resp["result"]["content"][0]["text"].as_str().unwrap()).unwrap()
}

#[test]
fn initialize_lists_tools_and_calls_them() {
    let (resp, _) = session(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"validate","arguments":{"pipeline":"videotestsrc ! fakesink"}}}"#,
        r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"launch","arguments":{"pipeline":"videotestsrc num-buffers=3 ! fakesink","duration_secs":10}}}"#,
    ]);

    // The notification (no id) produces no response, so 4 requests -> 4 responses.
    assert_eq!(resp.len(), 4, "one response per id-bearing request");

    // initialize
    assert_eq!(resp[0]["id"], 1);
    assert_eq!(resp[0]["result"]["serverInfo"]["name"], "g2g-mcp");

    // tools/list
    let tools: Vec<&str> = resp[1]["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(tools.contains(&"inspect") && tools.contains(&"validate") && tools.contains(&"launch"));
    assert!(tools.contains(&"start_pipeline") && tools.contains(&"pipeline_status"));
    assert!(tools.contains(&"insert_transform") && tools.contains(&"remove_transform"));
    assert!(tools.contains(&"set_log_level") && tools.contains(&"tail_logs"));
    assert!(tools.contains(&"sample_edge"));

    // validate -> ok
    assert_eq!(payload(&resp[2])["ok"], true);

    // launch -> ran the finite pipeline
    let l = payload(&resp[3]);
    assert_eq!(l["ok"], true);
    assert_eq!(l["stats"]["frames_consumed"], 3);
}

#[test]
fn unknown_method_returns_jsonrpc_error() {
    let (resp, _) = session(&[r#"{"jsonrpc":"2.0","id":9,"method":"no/such/method"}"#]);
    assert_eq!(resp.len(), 1);
    assert_eq!(resp[0]["id"], 9);
    assert_eq!(resp[0]["error"]["code"], -32601);
}

/// The declarative-run tool builds an inline YAML document and runs it under the
/// same deadline / stats conventions as `launch`.
#[cfg(feature = "declarative-yaml")]
#[test]
fn run_graph_runs_an_inline_yaml_document() {
    let doc = "nodes:\n  \
        - { id: src, element: videotestsrc, props: { num-buffers: 6 } }\n  \
        - { id: sink, element: fakesink }\nedges:\n  - { from: src, to: sink }\n";
    let call = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": { "name": "run_graph", "arguments": { "graph": doc, "duration_secs": 10 } },
    });
    let (resp, _) = session(&[
        r#"{"jsonrpc":"2.0","id":0,"method":"tools/list"}"#,
        &call.to_string(),
    ]);

    let tools: Vec<&str> = resp[0]["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(tools.contains(&"run_graph"), "run_graph is advertised");

    let out = payload(&resp[1]);
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(out["stats"]["frames_consumed"], 6);
}

/// A graph *file*: the format follows the extension, so a `.json` document loads
/// through the JSON front-end.
#[cfg(feature = "declarative")]
#[test]
fn run_graph_runs_a_document_file() {
    let path = std::env::temp_dir().join("g2g_mcp_run_graph.json");
    std::fs::write(
        &path,
        r#"{"nodes":[{"id":"src","element":"videotestsrc","props":{"num-buffers":4}},
                    {"id":"sink","element":"fakesink"}],
            "edges":[{"from":"src","to":"sink"}]}"#,
    )
    .unwrap();
    let call = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": { "name": "run_graph", "arguments": { "path": path, "duration_secs": 10 } },
    });
    let (resp, _) = session(&[&call.to_string()]);
    let out = payload(&resp[0]);
    let _ = std::fs::remove_file(&path);
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(out["stats"]["frames_consumed"], 4);
}

/// With a progress token, a run streams live `Observer` snapshots as
/// `notifications/progress` instead of only reporting final stats.
#[test]
fn launch_streams_telemetry_progress_notifications() {
    let call = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "launch",
            // no num-buffers: the run lasts the whole deadline, so snapshots
            // have to arrive mid-run
            "arguments": { "pipeline": "videotestsrc ! fakesink", "duration_secs": 1,
                           "telemetry_interval_ms": 20 },
            "_meta": { "progressToken": "tok-1" },
        },
    });
    let (resp, notes) = session(&[&call.to_string()]);

    assert_eq!(resp.len(), 1);
    assert_eq!(payload(&resp[0])["timed_out"], true);

    assert!(!notes.is_empty(), "expected mid-run progress notifications");
    let n = &notes[0];
    assert_eq!(n["method"], "notifications/progress");
    assert_eq!(n["params"]["progressToken"], "tok-1");
    assert_eq!(n["params"]["progress"], 1);
    // Each notification carries the dashboard's snapshot shape.
    let t = &n["params"]["telemetry"];
    assert_eq!(t["nodes"].as_array().unwrap().len(), 2);
    assert!(t["uptime_ns"].as_u64().unwrap() > 0);
    let edge = &t["edges"][0];
    assert!(edge["caps"].as_str().unwrap().contains("video"));
    assert!(
        notes
            .iter()
            .any(|n| n["params"]["telemetry"]["edges"][0]["packets"]
                .as_u64()
                .unwrap()
                > 0),
        "live edge counters should advance while the run is going"
    );
}

/// Without a progress token the run stays silent: MCP only allows progress
/// notifications for a request that asked for them.
#[test]
fn launch_without_progress_token_emits_no_notifications() {
    let (resp, notes) = session(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"launch","arguments":{"pipeline":"videotestsrc ! fakesink","duration_secs":1,"telemetry_interval_ms":20}}}"#,
    ]);
    assert_eq!(payload(&resp[0])["timed_out"], true);
    assert!(notes.is_empty(), "no token, no notifications: {notes:?}");
}

#[test]
fn manages_and_mutates_a_running_pipeline() {
    let (responses, notifications) = session(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"start_pipeline","arguments":{"pipeline":"videotestsrc name=src ! identity name=base ! fakesink name=sink"}}}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"pipeline_status","arguments":{}}}"#,
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"validate_insertion","arguments":{"target":"base","position":"after","element":"valve","properties":{"drop":false},"expected_revision":0}}}"#,
        r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"insert_transform","arguments":{"target":"base","position":"after","element":"valve","properties":{"drop":false},"expected_revision":0}}}"#,
        r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"insert_transform","arguments":{"target":"base","position":"after","element":"valve","properties":{"drop":false},"expected_revision":0}}}"#,
        r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"pipeline_status","arguments":{}}}"#,
        r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"remove_transform","arguments":{"node":"Valve0","expected_revision":1}}}"#,
        r#"{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"stop_pipeline","arguments":{}}}"#,
    ]);

    assert!(notifications.is_empty());
    assert_eq!(responses.len(), 8);
    assert_eq!(payload(&responses[0])["ok"], true);

    let initial = payload(&responses[1]);
    assert!(matches!(
        initial["state"].as_str(),
        Some("starting" | "running")
    ));
    assert_eq!(initial["revision"], 0);

    assert_eq!(payload(&responses[2])["ok"], true);
    let inserted = payload(&responses[3]);
    assert_eq!(inserted["ok"], true, "{inserted}");
    assert_eq!(inserted["node"], "Valve0");
    assert_eq!(inserted["revision"], 1);

    let stale = payload(&responses[4]);
    assert_eq!(stale["ok"], false);
    assert_eq!(stale["current_revision"], 1);

    let changed = payload(&responses[5]);
    assert_eq!(changed["inserted"]["Valve0"]["target"], "base");
    assert_eq!(changed["inserted"]["Valve0"]["properties"]["drop"], false);
    assert_eq!(changed["revision"], 1);
    assert_eq!(changed["telemetry"]["nodes"].as_array().unwrap().len(), 3);

    let removed = payload(&responses[6]);
    assert_eq!(removed["ok"], true, "{removed}");
    assert_eq!(removed["revision"], 2);
    assert_eq!(payload(&responses[7])["ok"], true);
}

#[test]
fn changes_log_levels_and_samples_live_packets() {
    let (responses, notifications) = session(&[
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"set_log_level","arguments":{"level":"info"}}}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"start_pipeline","arguments":{"pipeline":"videotestsrc name=src ! identity name=base ! fakesink name=sink"}}}"#,
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"sample_edge","arguments":{"edge":0,"count":2,"timeout_ms":2000}}}"#,
        r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"tail_logs","arguments":{"limit":20,"clear":true}}}"#,
        r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"tail_logs","arguments":{}}}"#,
        r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"set_log_level","arguments":{"level":"error"}}}"#,
        r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"stop_pipeline","arguments":{}}}"#,
    ]);

    assert!(notifications.is_empty());
    assert_eq!(responses.len(), 7);

    let enabled = payload(&responses[0]);
    assert_eq!(enabled["ok"], true);
    assert_eq!(enabled["previous_level"], "error");
    assert_eq!(enabled["level"], "info");

    assert_eq!(payload(&responses[1])["ok"], true);
    let sampled = payload(&responses[2]);
    assert_eq!(sampled["ok"], true, "{sampled}");
    assert_eq!(sampled["timed_out"], false, "{sampled}");
    let samples = sampled["samples"].as_array().unwrap();
    assert_eq!(samples.len(), 2);
    for sample in samples {
        assert_eq!(sample["kind"], "frame");
        assert!(sample["sequence"].is_u64());
        assert!(sample["memory"].is_string());
        assert!(sample["preview"].is_object());
    }

    let logs = payload(&responses[3]);
    assert_eq!(logs["ok"], true);
    assert!(
        logs["records"]
            .as_array()
            .unwrap()
            .iter()
            .any(|record| record["message"] == "added to pipeline"),
        "{logs}"
    );
    assert!(logs["records"].as_array().unwrap()[0]["timestamp_ns"].is_u64());
    assert!(logs["records"].as_array().unwrap()[0]["fields"].is_object());
    assert!(payload(&responses[4])["records"]
        .as_array()
        .unwrap()
        .is_empty());

    let restored = payload(&responses[5]);
    assert_eq!(restored["previous_level"], "info");
    assert_eq!(restored["level"], "error");
    assert_eq!(payload(&responses[6])["ok"], true);
}

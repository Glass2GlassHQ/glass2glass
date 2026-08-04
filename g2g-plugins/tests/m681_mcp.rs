//! M681 MCP server: `g2g-mcp` speaks JSON-RPC 2.0 over stdio and exposes the
//! inspect / validate / launch / run_graph tools for agent-driven dev, plus
//! live-telemetry progress notifications for the two run tools. Drives the built
//! binary end to end (the tool logic is unit-tested in `toolingjson`; this checks
//! the JSON-RPC framing).
//!
//! Needs `tooling-json` (`declarative-yaml` for the `run_graph` test):
//! `cargo test -p g2g-plugins --features tooling-json,declarative-yaml
//! --test m681_mcp`.
#![cfg(feature = "tooling-json")]

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

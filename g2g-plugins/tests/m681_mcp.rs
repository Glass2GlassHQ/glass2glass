//! M681 MCP server: `g2g-mcp` speaks JSON-RPC 2.0 over stdio and exposes the
//! inspect / validate / launch tools for agent-driven dev. Drives the built
//! binary end to end (the tool logic is unit-tested in `toolingjson`; this checks
//! the JSON-RPC framing).
//!
//! Needs `tooling-json`: `cargo test -p g2g-plugins --features tooling-json
//! --test m681_mcp`.
#![cfg(feature = "tooling-json")]

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

/// Feed the JSON-RPC request lines to `g2g-mcp` and collect the response lines.
fn session(requests: &[&str]) -> Vec<serde_json::Value> {
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
    let responses: Vec<serde_json::Value> = out
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).expect("response is JSON"))
        .collect();
    child.wait().unwrap();
    responses
}

#[test]
fn initialize_lists_tools_and_calls_them() {
    let resp = session(&[
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
    let tools: Vec<&str> =
        resp[1]["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(tools.contains(&"inspect") && tools.contains(&"validate") && tools.contains(&"launch"));

    // validate -> ok
    let v: serde_json::Value =
        serde_json::from_str(resp[2]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(v["ok"], true);

    // launch -> ran the finite pipeline
    let l: serde_json::Value =
        serde_json::from_str(resp[3]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(l["ok"], true);
    assert_eq!(l["stats"]["frames_consumed"], 3);
}

#[test]
fn unknown_method_returns_jsonrpc_error() {
    let resp = session(&[r#"{"jsonrpc":"2.0","id":9,"method":"no/such/method"}"#]);
    assert_eq!(resp.len(), 1);
    assert_eq!(resp[0]["id"], 9);
    assert_eq!(resp[0]["error"]["code"], -32601);
}

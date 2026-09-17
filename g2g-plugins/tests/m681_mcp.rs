//! M681 MCP server: `g2g-mcp` speaks JSON-RPC 2.0 over stdio and exposes the
//! inspect / validate / launch / run_graph tools for agent-driven dev, plus
//! live-telemetry progress notifications for the two run tools. Drives the built
//! binary end to end (the tool logic is unit-tested in `toolingjson`; this checks
//! the JSON-RPC framing).
//!
//! M1177 adds the record tools (`latest_metadata` / `wait_for_records` /
//! `load_metadata`), the live property tools, `snapshot_frame`, `clip_at` and
//! the README prompts, all driven through the same binary.
//!
//! Needs `mcp` (`declarative-yaml` for the `run_graph` test, `mjpeg` for the
//! clip test, which decodes the AVI it just wrote):
//! `cargo test -p g2g-plugins --features mcp,declarative-yaml,mjpeg
//! --test m681_mcp`.
#![cfg(all(feature = "tooling-json", feature = "multi-thread"))]

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

/// How long a live pipeline gets to post an event or finish before a test
/// gives up polling.
const LIVE_DEADLINE: Duration = Duration::from_secs(10);

/// Feed the JSON-RPC request lines to `g2g-mcp` and split its stdout into
/// (responses, notifications): a notification carries no `id`.
fn session(requests: &[&str]) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    session_with_env(requests, &[])
}

/// [`session`] with `environment` set on the server process.
fn session_with_env(
    requests: &[&str],
    environment: &[(&str, &str)],
) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_g2g-mcp"));
    for (name, value) in environment {
        command.env(name, value);
    }
    let mut child = command
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

/// A request-at-a-time `g2g-mcp` session, for tests that have to poll a live
/// pipeline between requests.
struct LiveSession {
    child: std::process::Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl LiveSession {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_g2g-mcp"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn g2g-mcp");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            stdin,
            stdout,
            next_id: 1,
        }
    }

    /// Call one tool and return its payload.
    fn call(&mut self, tool: &str, arguments: serde_json::Value) -> serde_json::Value {
        payload(&self.request(tool, arguments))
    }

    /// Call one tool and return its whole result, for a tool whose content is
    /// not one text block.
    fn result(&mut self, tool: &str, arguments: serde_json::Value) -> serde_json::Value {
        self.request(tool, arguments)["result"].clone()
    }

    fn request(&mut self, tool: &str, arguments: serde_json::Value) -> serde_json::Value {
        let id = self.next_id;
        self.next_id += 1;
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": { "name": tool, "arguments": arguments },
        });
        writeln!(self.stdin, "{request}").unwrap();
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        let response: serde_json::Value = serde_json::from_str(&line).expect("message is JSON");
        assert_eq!(response["id"], id, "{response}");
        response
    }

    /// Repeat `call` until `done` accepts the payload or the deadline passes.
    fn poll(
        &mut self,
        tool: &str,
        arguments: serde_json::Value,
        done: impl Fn(&serde_json::Value) -> bool,
    ) -> serde_json::Value {
        let deadline = Instant::now() + LIVE_DEADLINE;
        loop {
            let out = self.call(tool, arguments.clone());
            if done(&out) {
                return out;
            }
            assert!(Instant::now() < deadline, "timed out polling {tool}: {out}");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
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
    assert!(tools.contains(&"sample_edge") && tools.contains(&"tail_events"));
    assert!(tools.contains(&"latest_metadata") && tools.contains(&"wait_for_records"));
    assert!(tools.contains(&"load_metadata") && tools.contains(&"clip_at"));
    assert!(tools.contains(&"get_property") && tools.contains(&"set_property"));
    assert!(tools.contains(&"snapshot_frame"));

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

/// The kinds of the events in a `tail_events` payload, in buffer order.
fn event_kinds(events: &serde_json::Value) -> Vec<String> {
    events["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["event"]["kind"].as_str().unwrap().to_string())
        .collect()
}

/// A finite managed pipeline posts its stream-start on the bus, the tail keeps
/// the events in posting order with increasing sequence numbers, and
/// `clear=true` empties it.
#[test]
fn tails_bus_events_from_a_managed_pipeline() {
    let mut session = LiveSession::spawn();
    let started = session.call(
        "start_pipeline",
        serde_json::json!({ "pipeline": "videotestsrc num-buffers=3 name=src ! fakesink name=sink" }),
    );
    assert_eq!(started["ok"], true, "{started}");

    let finished = session.poll("pipeline_status", serde_json::json!({}), |status| {
        status["state"] == "finished"
    });
    assert_eq!(finished["stats"]["frames_consumed"], 3, "{finished}");
    assert!(finished["events"]["buffered"].as_u64().unwrap() > 0);
    assert_eq!(finished["events"]["capacity"], 1024);

    let tail = session.call("tail_events", serde_json::json!({}));
    assert_eq!(tail["ok"], true, "{tail}");
    assert_eq!(tail["overwritten"], 0);
    assert_eq!(tail["cleared"], false);
    let kinds = event_kinds(&tail);
    assert_eq!(kinds[0], "stream-start", "{tail}");
    assert!(kinds.contains(&"buffering".to_string()), "{tail}");
    assert_eq!(kinds.last().map(String::as_str), Some("eos"), "{tail}");
    let sequences: Vec<u64> = tail["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["sequence"].as_u64().unwrap())
        .collect();
    assert!(sequences.windows(2).all(|pair| pair[0] < pair[1]), "{tail}");
    assert!(tail["events"][0]["observed_ns"].is_u64());
    assert_eq!(tail["events"][0]["event"]["type"], "event");

    let limited = session.call("tail_events", serde_json::json!({ "limit": 1 }));
    assert_eq!(limited["events"].as_array().unwrap().len(), 1);
    assert_eq!(limited["events"][0]["sequence"], *sequences.last().unwrap());

    let cleared = session.call("tail_events", serde_json::json!({ "clear": true }));
    assert_eq!(cleared["cleared"], true);
    assert_eq!(cleared["events"].as_array().unwrap().len(), sequences.len());
    let empty = session.call("tail_events", serde_json::json!({}));
    assert!(empty["events"].as_array().unwrap().is_empty(), "{empty}");
    assert_eq!(
        session.call("stop_pipeline", serde_json::json!({}))["ok"],
        true
    );
}

/// A host that runs its own pipeline registers its observer, mutator and bus
/// with an in-process server: the tools read and mutate the host's run, error /
/// warning / QoS / buffering / negotiation events keep their fields, the server
/// refuses to stop what the host owns, and unregistering leaves the run going.
#[cfg(feature = "observe")]
#[test]
fn a_host_registers_its_own_pipeline_in_process() {
    use g2g_core::runtime::{
        parse_launch, run_graph_observed_mutable, select2, Either, GraphMutator,
        NegotiationFailure, Observer,
    };
    use g2g_core::{Bus, BusMessage, G2gError};
    use g2g_plugins::clock::WallClock;
    use g2g_plugins::mcp::McpServer;
    use g2g_plugins::registry::default_registry;

    let graph = parse_launch(
        &default_registry(),
        "videotestsrc name=src ! identity name=base ! fakesink name=sink",
    )
    .expect("launch line parses");
    let observer = Observer::new();
    let (bus, bus_handle) = Bus::new(256);
    let (stop, stop_receiver) = tokio::sync::oneshot::channel::<()>();
    let (mutator_sender, mutator_receiver) =
        std::sync::mpsc::sync_channel::<GraphMutator<'static>>(1);
    let host_observer = observer.clone();
    let host_bus_handle = bus_handle.clone();
    let host = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        // the mutator's element lifetime is the clock's, and MCP wants `'static`
        let clock: &'static WallClock = Box::leak(Box::new(WallClock::new()));
        let (mutator, run) =
            run_graph_observed_mutable(graph, clock, 4, &host_observer, Some(&host_bus_handle));
        mutator_sender.send(mutator).unwrap();
        runtime.block_on(select2(stop_receiver, run))
    });
    let mutator = mutator_receiver.recv().unwrap();

    let mut server = McpServer::with_registry(default_registry());
    let handle = server
        .register_pipeline(observer, mutator, bus)
        .expect("registers");
    let call = |server: &mut McpServer, tool: &str, arguments: serde_json::Value| {
        let params = serde_json::json!({ "name": tool, "arguments": arguments });
        payload(
            &serde_json::json!({ "result": server.dispatch("tools/call", Some(&params)).unwrap() }),
        )
    };

    let status = call(&mut server, "pipeline_status", serde_json::json!({}));
    assert!(
        matches!(status["state"].as_str(), Some("starting" | "running")),
        "{status}"
    );
    assert_eq!(status["revision"], 0);

    assert!(bus_handle.try_post(BusMessage::Error(G2gError::CapsMismatch)));
    assert!(bus_handle.try_post(BusMessage::Warning(G2gError::Shutdown)));
    assert!(bus_handle.try_post(BusMessage::Qos {
        running_time_ns: 40,
        jitter_ns: -7,
        processed: 12,
        dropped: 1,
    }));
    assert!(bus_handle.try_post(BusMessage::Buffering {
        percent: 42,
        element: Some("base".into()),
    }));
    assert!(bus_handle.try_post(BusMessage::NegotiationFailed(
        NegotiationFailure::Degenerate
    )));
    let deadline = Instant::now() + LIVE_DEADLINE;
    let tail = loop {
        let tail = call(&mut server, "tail_events", serde_json::json!({}));
        if event_kinds(&tail).contains(&"negotiation-failed".to_string()) {
            break tail;
        }
        assert!(
            Instant::now() < deadline,
            "posted events never arrived: {tail}"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    let events: Vec<&serde_json::Value> = tail["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| &entry["event"])
        .collect();
    let find = |kind: &str| {
        events
            .iter()
            .find(|event| event["kind"] == kind)
            .copied()
            .unwrap()
    };
    // the host's own posts race the runner's stream-start, so only its presence
    // is decided
    assert!(
        events.iter().any(|event| event["kind"] == "stream-start"),
        "{tail}"
    );
    assert_eq!(
        find("error")["text"],
        format!("{:?}", G2gError::CapsMismatch)
    );
    assert_eq!(find("warning")["text"], format!("{:?}", G2gError::Shutdown));
    let qos = find("qos");
    assert_eq!(qos["running_time_ns"], 40);
    assert_eq!(qos["jitter_ns"], -7);
    assert_eq!(qos["processed"], 12);
    assert_eq!(qos["dropped"], 1);
    let buffering = find("buffering");
    assert_eq!(buffering["percent"], 42);
    assert_eq!(buffering["element"], "base");
    assert_eq!(
        find("negotiation-failed")["text"],
        format!("{:?}", NegotiationFailure::Degenerate)
    );

    let sampled = call(
        &mut server,
        "sample_edge",
        serde_json::json!({ "edge": 0, "count": 1, "timeout_ms": 5000 }),
    );
    assert_eq!(sampled["ok"], true, "{sampled}");
    assert_eq!(sampled["samples"][0]["kind"], "frame");

    let inserted = call(
        &mut server,
        "insert_transform",
        serde_json::json!({
            "target": "base", "position": "after", "element": "valve",
            "properties": { "drop": false }, "expected_revision": 0,
        }),
    );
    assert_eq!(inserted["ok"], true, "{inserted}");
    assert_eq!(inserted["revision"], 1);

    let refused = call(&mut server, "stop_pipeline", serde_json::json!({}));
    assert_eq!(refused["ok"], false);
    assert!(
        refused["error"].as_str().unwrap().contains("host"),
        "{refused}"
    );
    let still = call(&mut server, "pipeline_status", serde_json::json!({}));
    assert_eq!(still["ok"], true);
    assert_eq!(still["revision"], 1);

    stop.send(()).unwrap();
    match host.join().unwrap() {
        Either::Left(_) => handle.stop(),
        Either::Right(Ok(stats)) => handle.finish(&stats),
        Either::Right(Err(error)) => panic!("host run failed: {error:?}"),
    }
    let stopped = call(&mut server, "pipeline_status", serde_json::json!({}));
    assert_eq!(stopped["state"], "stopped", "{stopped}");

    assert!(server.unregister_pipeline());
    assert!(!server.unregister_pipeline());
    let gone = call(&mut server, "pipeline_status", serde_json::json!({}));
    assert_eq!(gone["ok"], false, "{gone}");
}

/// A path under the temporary directory, unique to this process so parallel
/// tests never share a file.
fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("g2g_mcp_{}_{name}", std::process::id()))
}

/// A `videotestsrc` property default, read off the element rather than restated
/// here, so a test expectation follows the element.
fn videotestsrc_default(session: &mut LiveSession, property: &str) -> String {
    let inspected = session.call("inspect", serde_json::json!({ "element": "videotestsrc" }));
    inspected["elements"][0]["properties"]
        .as_array()
        .unwrap()
        .iter()
        .find(|spec| spec["name"] == property)
        .unwrap_or_else(|| panic!("videotestsrc has a {property} property: {inspected}"))["default"]
        .as_str()
        .unwrap()
        .to_string()
}

/// Packets the pipeline's sink has taken, from a `pipeline_status` payload. 0
/// while the run is still starting.
fn sink_packets(status: &serde_json::Value) -> u64 {
    let Some(sink) = status["telemetry"]["nodes"]
        .as_array()
        .and_then(|nodes| nodes.iter().find(|node| node["role"] == "sink"))
    else {
        return 0;
    };
    status["telemetry"]["edges"]
        .as_array()
        .and_then(|edges| edges.iter().find(|edge| edge["to"] == sink["id"]))
        .and_then(|edge| edge["packets"].as_u64())
        .unwrap_or(0)
}

/// The records the replay tests feed in. Each box is an exact binary fraction of
/// the frame, so the pixels survive the normalize / denormalize round trip
/// through `metareplay` unchanged, and only the second record carries an alert.
#[cfg(feature = "analytics-json")]
fn fixture_records() -> Vec<serde_json::Value> {
    Vec::from([
        serde_json::json!({
            "pts": 0.0,
            "detections": [
                { "label": "person", "x": 80, "y": 120, "w": 40, "h": 60, "score": 0.875 }
            ],
        }),
        // the second frame of a 30 fps videotestsrc, inside metareplay's window
        serde_json::json!({
            "pts": 1.0 / 30.0,
            "detections": [
                { "label": "car", "x": 160, "y": 60, "w": 80, "h": 120, "score": 0.5 }
            ],
            "alert": { "rule": "person-in-zone" },
        }),
    ])
}

#[cfg(feature = "analytics-json")]
fn write_fixture(path: &std::path::Path) {
    let lines: Vec<String> = fixture_records()
        .iter()
        .map(serde_json::Value::to_string)
        .collect();
    std::fs::write(path, lines.join("\n") + "\n").expect("write the record fixture");
}

/// A detector's output replayed onto bare frames reaches the record tools: the
/// sink posts each line on the bus, `wait_for_records` hands back the ones
/// posted since the call, and `key` narrows them to the records carrying it.
#[cfg(feature = "analytics-json")]
#[test]
fn reads_the_records_a_metasink_posts() {
    let fixture = temp_path("records.jsonl");
    let written = temp_path("records-written.jsonl");
    write_fixture(&fixture);
    let _ = std::fs::remove_file(&written);
    let expected = fixture_records();

    let mut session = LiveSession::spawn();
    let started = session.call(
        "start_pipeline",
        serde_json::json!({
            "pipeline": format!(
                "videotestsrc num-buffers=5 ! metareplay location=\"{}\" ! metasink location=\"{}\"",
                fixture.display(),
                written.display(),
            ),
        }),
    );
    assert_eq!(started["ok"], true, "{started}");

    let waited = session.call(
        "wait_for_records",
        serde_json::json!({ "count": expected.len(), "timeout_ms": 10000 }),
    );
    assert_eq!(waited["ok"], true, "{waited}");
    let records = waited["records"].as_array().unwrap();
    assert_eq!(records.len(), expected.len(), "{waited}");
    assert_eq!(
        records[0]["detections"], expected[0]["detections"],
        "the boxes come back in the pixels the fixture named: {waited}"
    );
    assert_eq!(
        records[1]["detections"], expected[1]["detections"],
        "{waited}"
    );
    assert_eq!(records[1]["alert"], expected[1]["alert"], "{waited}");
    assert_eq!(records[0]["pts"], 0.0, "{waited}");

    // the run is over, so its whole tail is fresh again and the key filter is
    // what decides which records come back
    session.poll("pipeline_status", serde_json::json!({}), |status| {
        status["state"] == "finished"
    });
    let alerts = session.call(
        "wait_for_records",
        serde_json::json!({ "key": "alert", "timeout_ms": 2000 }),
    );
    let alerts = alerts["records"].as_array().unwrap();
    assert_eq!(
        alerts.len(),
        1,
        "only one record carries an alert: {alerts:?}"
    );
    assert_eq!(alerts[0]["alert"], expected[1]["alert"]);

    let latest = session.call("latest_metadata", serde_json::json!({}));
    let latest = latest["records"].as_array().unwrap();
    assert_eq!(latest.len(), expected.len());
    assert_eq!(latest[0]["pts"], 0.0, "oldest first");
    assert!(latest[1]["pts"].as_f64().unwrap() > 0.0);

    let lines = std::fs::read_to_string(&written).expect("the sink wrote its file");
    assert_eq!(
        lines.lines().count(),
        expected.len(),
        "the bus carried what the file holds"
    );
    let _ = std::fs::remove_file(&fixture);
    let _ = std::fs::remove_file(&written);
}

/// A finished run's file stands in for a pipeline: `load_metadata` fills the
/// store, and a wait over it returns at once instead of blocking on a run that
/// is not there.
#[cfg(feature = "analytics-json")]
#[test]
fn loads_records_from_a_file_with_no_pipeline() {
    let fixture = temp_path("loaded.jsonl");
    write_fixture(&fixture);
    let expected = fixture_records();

    let mut session = LiveSession::spawn();
    let refused = session.call(
        "wait_for_records",
        serde_json::json!({ "timeout_ms": 1000 }),
    );
    assert_eq!(refused["ok"], false, "{refused}");
    assert!(
        refused["error"]
            .as_str()
            .unwrap()
            .contains("start_pipeline"),
        "{refused}"
    );

    let loaded = session.call(
        "load_metadata",
        serde_json::json!({ "path": fixture.display().to_string() }),
    );
    assert_eq!(loaded["records"], expected.len(), "{loaded}");

    let waited = session.call(
        "wait_for_records",
        serde_json::json!({ "count": expected.len(), "timeout_ms": 1000 }),
    );
    assert_eq!(waited["status"]["state"], "none", "{waited}");
    let records = waited["records"].as_array().unwrap();
    assert_eq!(records.len(), expected.len(), "{waited}");
    assert_eq!(records[1]["alert"], expected[1]["alert"]);
    let _ = std::fs::remove_file(&fixture);
}

/// A property written mid-run takes effect between packets: the valve stops
/// feeding the sink, the read back reports the new value, and the two refusals
/// (an unknown name, a source) come back as the mutator's own errors.
#[test]
fn reads_and_writes_live_properties() {
    let mut session = LiveSession::spawn();
    let started = session.call(
        "start_pipeline",
        serde_json::json!({
            "pipeline": "videotestsrc name=src ! valve name=v drop=false ! fakesink name=sink",
        }),
    );
    assert_eq!(started["ok"], true, "{started}");

    let read = session.call(
        "get_property",
        serde_json::json!({ "element": "v", "property": "drop" }),
    );
    assert_eq!(read["ok"], true, "{read}");
    assert_eq!(read["drop"], false, "{read}");

    // frames must be flowing before closing the valve can be seen to stop them
    session.poll("pipeline_status", serde_json::json!({}), |status| {
        sink_packets(status) > 0
    });

    let set = session.call(
        "set_property",
        serde_json::json!({ "element": "v", "property": "drop", "value": true }),
    );
    assert_eq!(set["ok"], true, "{set}");
    assert_eq!(set["drop"], true, "the value is read back off the element");

    // the frames already in flight land first, then the count stops moving
    std::thread::sleep(Duration::from_millis(300));
    let before = sink_packets(&session.call("pipeline_status", serde_json::json!({})));
    std::thread::sleep(Duration::from_millis(300));
    let after = sink_packets(&session.call("pipeline_status", serde_json::json!({})));
    assert!(before > 0, "the sink took frames before the valve closed");
    assert_eq!(before, after, "a dropping valve feeds the sink nothing");

    let unknown = session.call(
        "set_property",
        serde_json::json!({ "element": "v", "property": "nope", "value": 1 }),
    );
    assert_eq!(unknown["ok"], false, "{unknown}");
    assert!(
        unknown["error"]
            .as_str()
            .unwrap()
            .contains("PropertyRejected"),
        "{unknown}"
    );

    let source = session.call(
        "set_property",
        serde_json::json!({ "element": "src", "property": "pattern", "value": "ball" }),
    );
    assert_eq!(source["ok"], false, "{source}");
    assert!(
        source["error"].as_str().unwrap().contains("NotMutable"),
        "{source}"
    );

    assert_eq!(
        session.call("stop_pipeline", serde_json::json!({}))["ok"],
        true
    );
}

/// The PNG magic every encoded snapshot starts with.
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// The newest frame reaching the pipeline's sink comes back as an image content
/// block beside the text one, at the geometry the source negotiated.
#[test]
fn snapshots_the_newest_frame_as_a_png() {
    use base64::Engine as _;

    let mut session = LiveSession::spawn();
    let started = session.call(
        "start_pipeline",
        serde_json::json!({ "pipeline": "videotestsrc name=src ! fakesink name=shown" }),
    );
    assert_eq!(started["ok"], true, "{started}");
    session.poll("pipeline_status", serde_json::json!({}), |status| {
        sink_packets(status) > 0
    });

    let result = session.result("snapshot_frame", serde_json::json!({}));
    let image = &result["content"][0];
    assert_eq!(image["type"], "image", "{result}");
    assert_eq!(image["mimeType"], "image/png");
    let png = base64::engine::general_purpose::STANDARD
        .decode(image["data"].as_str().unwrap())
        .expect("the image block is base64");
    assert_eq!(&png[..PNG_SIGNATURE.len()], &PNG_SIGNATURE, "a PNG file");

    let described: serde_json::Value =
        serde_json::from_str(result["content"][1]["text"].as_str().unwrap()).unwrap();
    assert_eq!(described["ok"], true, "{described}");
    assert_eq!(
        described["element"], "shown",
        "the only sink is the default"
    );
    assert!(described["pts_ns"].is_u64(), "{described}");

    let decoder = png::Decoder::new(std::io::Cursor::new(&png));
    let reader = decoder.read_info().expect("the snapshot decodes");
    let info = reader.info();
    let width: u32 = videotestsrc_default(&mut session, "width").parse().unwrap();
    let height: u32 = videotestsrc_default(&mut session, "height")
        .parse()
        .unwrap();
    assert_eq!((info.width, info.height), (width, height));
    assert_eq!(described["width"], width);
    assert_eq!(described["height"], height);

    assert_eq!(
        session.call("stop_pipeline", serde_json::json!({}))["ok"],
        true
    );
}

/// With several sinks the snapshot goes to the one the results come from, the
/// `metasink`, rather than the display.
#[cfg(feature = "analytics-json")]
#[test]
fn snapshots_the_metasink_among_several_sinks() {
    let written = temp_path("snapshot-records.jsonl");
    let mut session = LiveSession::spawn();
    let started = session.call(
        "start_pipeline",
        serde_json::json!({
            "pipeline": format!(
                "videotestsrc ! tee name=t ! fakesink name=shown t. ! metasink location=\"{}\"",
                written.display(),
            ),
        }),
    );
    assert_eq!(started["ok"], true, "{started}");
    session.poll("pipeline_status", serde_json::json!({}), |status| {
        sink_packets(status) > 0
    });

    let result = session.result("snapshot_frame", serde_json::json!({}));
    let described: serde_json::Value =
        serde_json::from_str(result["content"][1]["text"].as_str().unwrap()).unwrap();
    assert_eq!(described["ok"], true, "{described}");
    assert_ne!(described["element"], "shown", "{described}");
    assert_eq!(result["content"][0]["type"], "image", "{result}");

    assert_eq!(
        session.call("stop_pipeline", serde_json::json!({}))["ok"],
        true
    );
    let _ = std::fs::remove_file(&written);
}

/// Each `###` section of the README's sample pipelines is served as a prompt,
/// addressable by an ASCII slug and carrying that section's code blocks.
#[test]
fn serves_the_readme_pipelines_as_prompts() {
    let readme = concat!(env!("CARGO_MANIFEST_DIR"), "/../README.md");
    let (responses, _) = session_with_env(
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"prompts/list"}"#,
        ],
        &[("G2G_README", readme)],
    );
    assert!(
        responses[0]["result"]["capabilities"]["prompts"].is_object(),
        "{}",
        responses[0]
    );
    let prompts = responses[1]["result"]["prompts"]
        .as_array()
        .unwrap()
        .clone();
    assert!(!prompts.is_empty(), "{}", responses[1]);
    for prompt in &prompts {
        let name = prompt["name"].as_str().unwrap();
        assert!(
            !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
            "{name} is not addressable"
        );
        assert!(prompt["description"].as_str().unwrap().contains("README"));
    }

    let first = prompts[0]["name"].as_str().unwrap();
    let get = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "prompts/get", "params": { "name": first },
    });
    let (responses, _) = session_with_env(&[&get.to_string()], &[("G2G_README", readme)]);
    let message = &responses[0]["result"]["messages"][0];
    assert_eq!(message["role"], "user");
    let text = message["content"]["text"].as_str().unwrap();
    assert!(text.starts_with("These are the"), "{text}");
    assert!(
        text.contains("```"),
        "the section's code blocks ride along: {text}"
    );
}

/// A clip is cut out of a file by decoding it, trimming to the range around the
/// pts and muxing it back: the range is centred on the pts and the output is an
/// AVI holding the frames that second covers.
#[cfg(feature = "mjpeg")]
#[test]
fn cuts_a_clip_out_of_a_video_file() {
    /// 3 s of video at the source's own framerate.
    const SOURCE_FRAMES: u64 = 90;
    const CLIP_PTS: f64 = 1.5;
    const CLIP_SECONDS: f64 = 1.0;

    let source = temp_path("clip-source.avi");
    let _ = std::fs::remove_file(&source);
    let mut session = LiveSession::spawn();
    let built = session.call(
        "launch",
        serde_json::json!({
            "pipeline": format!(
                "videotestsrc num-buffers={SOURCE_FRAMES} ! videoconvert ! mjpegenc ! avimux \
                 ! filesink location=\"{}\"",
                source.display(),
            ),
            "duration_secs": 60,
        }),
    );
    assert_eq!(built["ok"], true, "{built}");
    assert!(source.is_file(), "the source clip was written");

    let clip = session.call(
        "clip_at",
        serde_json::json!({
            "source": source.display().to_string(),
            "pts": CLIP_PTS,
            "seconds": CLIP_SECONDS,
            // bare `avidemux` advertises H.264 before it has read the header
            "decoder": "avidemux stream=mjpeg ! mjpegdec",
        }),
    );
    assert_eq!(clip["ok"], true, "{clip}");
    assert_eq!(clip["start"], CLIP_PTS - CLIP_SECONDS / 2.0, "{clip}");
    assert_eq!(clip["end"], CLIP_PTS + CLIP_SECONDS / 2.0, "{clip}");

    let framerate = videotestsrc_default(&mut session, "framerate");
    let (numerator, denominator) = framerate.split_once('/').expect("fps as n/d");
    let fps = numerator.parse::<f64>().unwrap() / denominator.parse::<f64>().unwrap();
    assert_eq!(
        clip["frames"].as_u64().unwrap(),
        (fps * CLIP_SECONDS) as u64,
        "{clip}"
    );

    let path = std::path::PathBuf::from(clip["path"].as_str().unwrap());
    let bytes = std::fs::read(&path).expect("the clip was written");
    assert_eq!(&bytes[..4], b"RIFF", "an AVI header");
    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&path);
}

/// A `metasink` with no `location` writes its lines to the server's stdout,
/// which carries the protocol, so the run is refused before it starts.
#[cfg(feature = "analytics-json")]
#[test]
fn refuses_a_metasink_that_would_write_to_the_protocol_stream() {
    let mut session = LiveSession::spawn();
    let refused = session.call(
        "start_pipeline",
        serde_json::json!({ "pipeline": "videotestsrc num-buffers=2 ! metasink" }),
    );
    assert_eq!(refused["ok"], false, "{refused}");
    assert!(
        refused["error"].as_str().unwrap().contains("location="),
        "{refused}"
    );
    let status = session.call("pipeline_status", serde_json::json!({}));
    assert_eq!(status["ok"], false, "nothing was started: {status}");
}

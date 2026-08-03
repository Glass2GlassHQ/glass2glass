//! `g2g-mcp`: a Model Context Protocol server over stdio, so an agent can drive
//! g2g development. Speaks newline-delimited JSON-RPC 2.0 and exposes five tools
//! backed by the same internals as `g2g-inspect` / `g2g-launch`:
//!
//!   list_elements             -> the registry (name, role, klass per element)
//!   inspect  {element}        -> one element's full introspection JSON
//!   validate {pipeline}       -> parse + negotiate a launch line, no run
//!   launch   {pipeline, secs} -> run it for up to `secs` and report RunStats
//!   run_graph {path|graph}    -> same, from a declarative JSON / YAML document
//!                                (`declarative` builds only)
//!
//! `launch` and `run_graph` stream live telemetry while the pipeline runs: when
//! the client passes a `_meta.progressToken` with the `tools/call` (the MCP
//! contract for a long-running request), each tick emits a
//! `notifications/progress` whose `telemetry` field is the dashboard's
//! per-element / per-edge snapshot JSON. Without a token the run is silent and
//! only the final stats come back.
//!
//! No MCP framework dependency: the JSON-RPC envelope is hand-rolled over
//! stdin/stdout with serde_json. Needs the `tooling-json` feature (which the
//! registry + runtime imply std).

use std::cell::Cell;
use std::io::{BufRead, Write};
use std::time::Duration;

use serde_json::{json, Value};

use g2g_core::runtime::Registry;
use g2g_plugins::registry::default_registry;
use g2g_plugins::toolingjson::{launch_json, registry_json, validate_json, TelemetryTap};

const PROTOCOL_VERSION: &str = "2024-11-05";

fn main() {
    let reg = default_registry();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue, // malformed line: skip, keep serving
        };
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let id = req.get("id").cloned();
        let result = dispatch(&reg, &rt, method, req.get("params"));

        // A request carries an id and gets a response; a notification (no id)
        // does not.
        let Some(id) = id else { continue };
        let envelope = match result {
            Ok(value) => json!({ "jsonrpc": "2.0", "id": id, "result": value }),
            Err((code, message)) => {
                json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
            }
        };
        let _ = writeln!(stdout, "{envelope}");
        let _ = stdout.flush();
    }
}

fn dispatch(
    reg: &Registry,
    rt: &tokio::runtime::Runtime,
    method: &str,
    params: Option<&Value>,
) -> Result<Value, (i64, String)> {
    match method {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "g2g-mcp", "version": env!("CARGO_PKG_VERSION") },
        })),
        "tools/list" => Ok(json!({ "tools": tool_specs() })),
        "tools/call" => call_tool(reg, rt, params),
        // Notifications and pings need no work.
        "notifications/initialized" | "ping" => Ok(json!({})),
        other => Err((-32601, format!("method not found: {other}"))),
    }
}

fn tool_specs() -> Value {
    #[allow(unused_mut)] // the declarative tool is feature-gated
    let mut tools = json!([
        {
            "name": "list_elements",
            "description": "List every registered g2g element (name, role, klass).",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "inspect",
            "description": "Full introspection of one element: role, pad caps, and typed properties.",
            "inputSchema": {
                "type": "object",
                "properties": { "element": { "type": "string" } },
                "required": ["element"]
            }
        },
        {
            "name": "validate",
            "description": "Parse and negotiate a gst-launch pipeline line without running it.",
            "inputSchema": {
                "type": "object",
                "properties": { "pipeline": { "type": "string" } },
                "required": ["pipeline"]
            }
        },
        {
            "name": "launch",
            "description": "Run a gst-launch pipeline for up to duration_secs and report RunStats. \
                            With a _meta.progressToken, live telemetry streams as notifications/progress.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pipeline": { "type": "string" },
                    "duration_secs": { "type": "integer" },
                    "telemetry_interval_ms": { "type": "integer" }
                },
                "required": ["pipeline"]
            }
        }
    ]);
    #[cfg(feature = "declarative")]
    if let Some(list) = tools.as_array_mut() {
        list.push(json!({
            "name": "run_graph",
            "description": "Run a declarative graph document (the JSON / YAML node+edge format) \
                            for up to duration_secs and report RunStats. Pass `path` to load a file \
                            or `graph` for an inline document. Same telemetry streaming as `launch`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "graph": { "type": "string" },
                    "format": { "type": "string", "enum": ["json", "yaml"] },
                    "duration_secs": { "type": "integer" },
                    "telemetry_interval_ms": { "type": "integer" }
                }
            }
        }));
    }
    tools
}

fn call_tool(
    reg: &Registry,
    rt: &tokio::runtime::Runtime,
    params: Option<&Value>,
) -> Result<Value, (i64, String)> {
    let params = params.ok_or((-32602, "missing params".into()))?;
    let name = params
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or((-32602, "missing tool name".into()))?;
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    // MCP only allows progress notifications for a request that supplied a
    // token, so that is also the switch for live telemetry.
    let token = params
        .get("_meta")
        .and_then(|m| m.get("progressToken"))
        .cloned();
    let ticks = Cell::new(0u64);
    let notify = |telemetry: Value| {
        if let Some(t) = &token {
            ticks.set(ticks.get() + 1);
            emit_progress(t, ticks.get(), telemetry);
        }
    };
    let tap = |args: &Value| {
        token
            .is_some()
            .then(|| TelemetryTap::new(tick_interval(args), &notify))
    };

    let payload: Value = match name {
        "list_elements" => {
            let full = registry_json(reg, None).map_err(|e| (-32603, e))?;
            // Compact listing: identity + role only.
            let list: Vec<Value> = full["elements"]
                .as_array()
                .map(|els| {
                    els.iter()
                        .map(|e| json!({ "name": e["name"], "role": e["role"], "klass": e["klass"] }))
                        .collect()
                })
                .unwrap_or_default();
            json!({ "elements": list })
        }
        "inspect" => {
            let el = args
                .get("element")
                .and_then(|e| e.as_str())
                .ok_or((-32602, "inspect needs `element`".into()))?;
            registry_json(reg, Some(el)).map_err(|e| (-32602, e))?
        }
        "validate" => {
            let line = args
                .get("pipeline")
                .and_then(|p| p.as_str())
                .ok_or((-32602, "validate needs `pipeline`".into()))?;
            rt.block_on(validate_json(reg, line))
        }
        "launch" => {
            let line = args
                .get("pipeline")
                .and_then(|p| p.as_str())
                .ok_or((-32602, "launch needs `pipeline`".into()))?;
            rt.block_on(launch_json(reg, line, duration_secs(&args), tap(&args)))
        }
        #[cfg(feature = "declarative")]
        "run_graph" => {
            let (doc, yaml) = graph_document(&args)?;
            rt.block_on(g2g_plugins::toolingjson::document_json(
                reg,
                &doc,
                yaml,
                duration_secs(&args),
                tap(&args),
            ))
        }
        other => return Err((-32602, format!("unknown tool: {other}"))),
    };

    // MCP tool results wrap output as content blocks; hand back the JSON as text.
    Ok(json!({
        "content": [ { "type": "text", "text": serde_json::to_string_pretty(&payload).unwrap_or_default() } ]
    }))
}

fn duration_secs(args: &Value) -> u64 {
    args.get("duration_secs")
        .and_then(|d| d.as_u64())
        .unwrap_or(5)
}

/// Telemetry cadence, defaulting to the dashboard's 250 ms tick and floored so a
/// client cannot ask for a notification per microsecond.
fn tick_interval(args: &Value) -> Duration {
    let ms = args
        .get("telemetry_interval_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(250)
        .max(10);
    Duration::from_millis(ms)
}

/// The declarative document to run plus whether to parse it as YAML: either an
/// inline `graph` string or a `path` to read, with `format` overriding the guess
/// (file extension, else a leading `{` means JSON).
#[cfg(feature = "declarative")]
fn graph_document(args: &Value) -> Result<(String, bool), (i64, String)> {
    let format = args.get("format").and_then(|f| f.as_str());
    let (doc, from_path) = match (
        args.get("path").and_then(|p| p.as_str()),
        args.get("graph").and_then(|g| g.as_str()),
    ) {
        (Some(path), _) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| (-32602, format!("cannot read '{path}': {e}")))?;
            let lower = path.to_ascii_lowercase();
            (
                text,
                Some(lower.ends_with(".yaml") || lower.ends_with(".yml")),
            )
        }
        (None, Some(text)) => (text.to_string(), None),
        (None, None) => return Err((-32602, "run_graph needs `path` or `graph`".into())),
    };
    let yaml = match format {
        Some("yaml") => true,
        Some("json") => false,
        Some(other) => return Err((-32602, format!("unknown format: {other}"))),
        None => from_path.unwrap_or(!doc.trim_start().starts_with('{')),
    };
    Ok((doc, yaml))
}

/// One `notifications/progress` for a telemetry tick. Written to the same stdout
/// as the responses: the stdio loop is parked in `block_on` while the run drives
/// this, so the writes cannot interleave with a response.
fn emit_progress(token: &Value, progress: u64, telemetry: Value) {
    let note = json!({
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": { "progressToken": token, "progress": progress, "telemetry": telemetry },
    });
    let mut out = std::io::stdout();
    let _ = writeln!(out, "{note}");
    let _ = out.flush();
}

//! `g2g-mcp`: a Model Context Protocol server over stdio, so an agent can drive
//! g2g development. Speaks newline-delimited JSON-RPC 2.0 and exposes four tools
//! backed by the same internals as `g2g-inspect` / `g2g-launch`:
//!
//!   list_elements            -> the registry (name, role, klass per element)
//!   inspect  {element}        -> one element's full introspection JSON
//!   validate {pipeline}       -> parse + negotiate a launch line, no run
//!   launch   {pipeline, secs} -> run it for up to `secs` and report RunStats
//!
//! No MCP framework dependency: the JSON-RPC envelope is hand-rolled over
//! stdin/stdout with serde_json. Needs the `tooling-json` feature (which the
//! registry + runtime imply std).

use std::io::{BufRead, Write};

use serde_json::{json, Value};

use g2g_core::runtime::Registry;
use g2g_plugins::registry::default_registry;
use g2g_plugins::toolingjson::{launch_json, registry_json, validate_json};

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
    json!([
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
            "description": "Run a gst-launch pipeline for up to duration_secs and report RunStats.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pipeline": { "type": "string" },
                    "duration_secs": { "type": "integer" }
                },
                "required": ["pipeline"]
            }
        }
    ])
}

fn call_tool(
    reg: &Registry,
    rt: &tokio::runtime::Runtime,
    params: Option<&Value>,
) -> Result<Value, (i64, String)> {
    let params = params.ok_or((-32602, "missing params".into()))?;
    let name = params.get("name").and_then(|n| n.as_str()).ok_or((-32602, "missing tool name".into()))?;
    let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));

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
            let el = args.get("element").and_then(|e| e.as_str()).ok_or((-32602, "inspect needs `element`".into()))?;
            registry_json(reg, Some(el)).map_err(|e| (-32602, e))?
        }
        "validate" => {
            let line = args.get("pipeline").and_then(|p| p.as_str()).ok_or((-32602, "validate needs `pipeline`".into()))?;
            rt.block_on(validate_json(reg, line))
        }
        "launch" => {
            let line = args.get("pipeline").and_then(|p| p.as_str()).ok_or((-32602, "launch needs `pipeline`".into()))?;
            let secs = args.get("duration_secs").and_then(|d| d.as_u64()).unwrap_or(5);
            rt.block_on(launch_json(reg, line, secs))
        }
        other => return Err((-32602, format!("unknown tool: {other}"))),
    };

    // MCP tool results wrap output as content blocks; hand back the JSON as text.
    Ok(json!({
        "content": [ { "type": "text", "text": serde_json::to_string_pretty(&payload).unwrap_or_default() } ]
    }))
}

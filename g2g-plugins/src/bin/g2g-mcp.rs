//! `g2g-mcp`: a Model Context Protocol server over stdio, so an agent can drive
//! g2g development. Speaks newline-delimited JSON-RPC 2.0 and exposes registry,
//! one-shot run, and managed live-pipeline tools backed by the same internals as
//! `g2g-inspect` / `g2g-launch`.
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
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use g2g_core::log::{LogLevel, LogValue, RingSink};
use g2g_core::property::{takes_undeclared_properties, PropValue};
use g2g_core::runtime::{
    parse_launch, run_graph_observed_mutable, select2, Either, GraphMutator, LinkInterceptor,
    Observer, ProbeAction, Registry,
};
use g2g_core::PipelinePacket;
use g2g_plugins::clock::WallClock;
use g2g_plugins::preview::packet_preview;
use g2g_plugins::registry::default_registry;
use g2g_plugins::toolingjson::{
    launch_json, registry_json, stats_json, telemetry_json, validate_json, TelemetryTap,
};

const PROTOCOL_VERSION: &str = "2024-11-05";
const LOG_CAPACITY: usize = 1024;
const DEFAULT_PACKET_SAMPLE_COUNT: usize = 1;
const MAX_PACKET_SAMPLE_COUNT: usize = 32;
const DEFAULT_PACKET_SAMPLE_TIMEOUT_MS: u64 = 1000;
const MAX_PACKET_SAMPLE_TIMEOUT_MS: u64 = 30_000;
static PIPELINE_CLOCK: OnceLock<WallClock> = OnceLock::new();

fn main() {
    let mut server = Server::new();

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
        let result = server.dispatch(method, req.get("params"));

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

struct Server {
    registry: Registry,
    runtime: tokio::runtime::Runtime,
    pipeline: Option<ManagedPipeline>,
    logs: RingSink,
    default_log_level: LogLevel,
    category_log_levels: BTreeMap<String, LogLevel>,
}

impl Server {
    fn new() -> Self {
        let logs = RingSink::new(LOG_CAPACITY);
        g2g_core::log::set_sink(Box::new(logs.clone()));
        g2g_core::log::set_time_source(g2g_core::log::unix_time_source);
        Self {
            registry: default_registry(),
            runtime: tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build tokio runtime"),
            pipeline: None,
            logs,
            default_log_level: LogLevel::Error,
            category_log_levels: BTreeMap::new(),
        }
    }

    fn dispatch(&mut self, method: &str, params: Option<&Value>) -> Result<Value, (i64, String)> {
        match method {
            "initialize" => Ok(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "g2g-mcp", "version": env!("CARGO_PKG_VERSION") },
            })),
            "tools/list" => Ok(json!({ "tools": tool_specs() })),
            "tools/call" => self.call_tool(params),
            "notifications/initialized" | "ping" => Ok(json!({})),
            other => Err((-32601, format!("method not found: {other}"))),
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(mut pipeline) = self.pipeline.take() {
            pipeline.stop();
        }
    }
}

#[derive(Debug)]
enum PipelineRunState {
    Running,
    Finished(Value),
    Failed(String),
    Stopped,
}

struct ManagedPipeline {
    observer: Observer,
    mutator: GraphMutator<'static>,
    run_state: Arc<Mutex<PipelineRunState>>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
    revision: u64,
    inserted: BTreeMap<String, Value>,
}

impl ManagedPipeline {
    fn stop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if self
            .thread
            .take()
            .is_some_and(|thread| thread.join().is_err())
        {
            *self.run_state.lock().expect("pipeline run state") =
                PipelineRunState::Failed("pipeline thread panicked".into());
        }
    }

    fn status_json(&self) -> Value {
        let snapshot = self.observer.snapshot();
        let state = match &*self.run_state.lock().expect("pipeline run state") {
            PipelineRunState::Running if snapshot.nodes.is_empty() => {
                json!({ "state": "starting" })
            }
            PipelineRunState::Running => json!({ "state": "running" }),
            PipelineRunState::Finished(stats) => {
                json!({ "state": "finished", "stats": stats })
            }
            PipelineRunState::Failed(error) => {
                json!({ "state": "failed", "error": error })
            }
            PipelineRunState::Stopped => json!({ "state": "stopped" }),
        };
        let mut result = json!({
            "ok": true,
            "revision": self.revision,
            "telemetry": telemetry_json(&snapshot),
            "inserted": self.inserted,
        });
        if let (Some(result), Some(state)) = (result.as_object_mut(), state.as_object()) {
            result.extend(state.clone());
        }
        result
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
        },
        {
            "name": "start_pipeline",
            "description": "Start one pipeline in the background for live inspection and mutation.",
            "inputSchema": {
                "type": "object",
                "properties": { "pipeline": { "type": "string" } },
                "required": ["pipeline"]
            }
        },
        {
            "name": "pipeline_status",
            "description": "Read the managed pipeline state, graph revision, and live telemetry.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "set_log_level",
            "description": "Change the process-wide default or one log category while pipelines are running.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "level": {
                        "oneOf": [
                            { "type": "string", "enum": ["off", "error", "warn", "fixme", "info", "debug", "log", "trace"] },
                            { "type": "integer", "minimum": 0, "maximum": 7 }
                        ]
                    },
                    "category": { "type": "string" }
                },
                "required": ["level"]
            }
        },
        {
            "name": "tail_logs",
            "description": "Read structured records from the bounded in-memory log tail.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "minimum": 1, "maximum": 1024 },
                    "clear": { "type": "boolean" }
                }
            }
        },
        {
            "name": "sample_edge",
            "description": "Passively sample bounded packet metadata and content previews from one live edge.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "edge": { "type": "integer", "minimum": 0 },
                    "count": { "type": "integer", "minimum": 1, "maximum": 32 },
                    "timeout_ms": { "type": "integer", "minimum": 1, "maximum": 30000 }
                },
                "required": ["edge"]
            }
        },
        {
            "name": "validate_insertion",
            "description": "Check a transform against the caps currently flowing before or after a node, without changing the graph.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": { "type": "string" },
                    "position": { "type": "string", "enum": ["before", "after"] },
                    "element": { "type": "string" },
                    "properties": { "type": "object" },
                    "expected_revision": { "type": "integer" }
                },
                "required": ["target", "position", "element", "expected_revision"]
            }
        },
        {
            "name": "insert_transform",
            "description": "Insert a transform before or after a node in the managed pipeline.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": { "type": "string" },
                    "position": { "type": "string", "enum": ["before", "after"] },
                    "element": { "type": "string" },
                    "properties": { "type": "object" },
                    "expected_revision": { "type": "integer" }
                },
                "required": ["target", "position", "element", "expected_revision"]
            }
        },
        {
            "name": "remove_transform",
            "description": "Remove a transform previously inserted through this MCP server.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "node": { "type": "string" },
                    "expected_revision": { "type": "integer" }
                },
                "required": ["node", "expected_revision"]
            }
        },
        {
            "name": "stop_pipeline",
            "description": "Stop and release the managed pipeline.",
            "inputSchema": { "type": "object", "properties": {} }
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

impl Server {
    fn call_tool(&mut self, params: Option<&Value>) -> Result<Value, (i64, String)> {
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
                let full = registry_json(&self.registry, None).map_err(|e| (-32603, e))?;
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
                registry_json(&self.registry, Some(el)).map_err(|e| (-32602, e))?
            }
            "validate" => {
                let line = args
                    .get("pipeline")
                    .and_then(|p| p.as_str())
                    .ok_or((-32602, "validate needs `pipeline`".into()))?;
                self.runtime.block_on(validate_json(&self.registry, line))
            }
            "launch" => {
                let line = args
                    .get("pipeline")
                    .and_then(|p| p.as_str())
                    .ok_or((-32602, "launch needs `pipeline`".into()))?;
                self.runtime.block_on(launch_json(
                    &self.registry,
                    line,
                    duration_secs(&args),
                    tap(&args),
                ))
            }
            "start_pipeline" => {
                let line = args
                    .get("pipeline")
                    .and_then(|p| p.as_str())
                    .ok_or((-32602, "start_pipeline needs `pipeline`".into()))?;
                self.start_pipeline(line)
            }
            "pipeline_status" => self
                .pipeline
                .as_ref()
                .map(ManagedPipeline::status_json)
                .unwrap_or_else(|| json!({ "ok": false, "error": "no managed pipeline" })),
            "set_log_level" => self.set_log_level(&args)?,
            "tail_logs" => self.tail_logs(&args)?,
            "sample_edge" => self.sample_edge(&args)?,
            "validate_insertion" => self.validate_insertion(&args)?,
            "insert_transform" => self.insert_transform(&args)?,
            "remove_transform" => self.remove_transform(&args)?,
            "stop_pipeline" => self.stop_pipeline(),
            #[cfg(feature = "declarative")]
            "run_graph" => {
                let (doc, yaml) = graph_document(&args)?;
                self.runtime
                    .block_on(g2g_plugins::toolingjson::document_json(
                        &self.registry,
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

    fn start_pipeline(&mut self, line: &str) -> Value {
        if self.pipeline.is_some() {
            return json!({ "ok": false, "error": "a managed pipeline already exists" });
        }
        let graph = match parse_launch(&self.registry, line) {
            Ok(graph) => graph,
            Err(error) => {
                return json!({ "ok": false, "stage": "parse", "error": format!("{error}") });
            }
        };

        let observer = Observer::new();
        let thread_observer = observer.clone();
        let run_state = Arc::new(Mutex::new(PipelineRunState::Running));
        let thread_run_state = run_state.clone();
        let (stop, stop_receiver) = tokio::sync::oneshot::channel();
        let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
        let thread = match std::thread::Builder::new()
            .name("g2g-mcp-pipeline".into())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let message = format!("cannot build pipeline runtime: {error}");
                        let _ = ready_sender.send(Err(message.clone()));
                        *thread_run_state.lock().expect("pipeline run state") =
                            PipelineRunState::Failed(message);
                        return;
                    }
                };
                let clock = PIPELINE_CLOCK.get_or_init(WallClock::new);
                let (mutator, run) =
                    run_graph_observed_mutable(graph, clock, 4, &thread_observer, None);
                if ready_sender.send(Ok(mutator)).is_err() {
                    return;
                }
                let state = match runtime.block_on(select2(stop_receiver, run)) {
                    Either::Left(_) => PipelineRunState::Stopped,
                    Either::Right(Ok(stats)) => PipelineRunState::Finished(stats_json(&stats)),
                    Either::Right(Err(error)) => PipelineRunState::Failed(format!("{error:?}")),
                };
                *thread_run_state.lock().expect("pipeline run state") = state;
            }) {
            Ok(thread) => thread,
            Err(error) => {
                return json!({ "ok": false, "error": format!("cannot start pipeline thread: {error}") });
            }
        };
        let mutator = match ready_receiver.recv() {
            Ok(Ok(mutator)) => mutator,
            Ok(Err(error)) => {
                let _ = thread.join();
                return json!({ "ok": false, "error": error });
            }
            Err(error) => {
                let _ = thread.join();
                return json!({ "ok": false, "error": format!("pipeline did not start: {error}") });
            }
        };
        self.pipeline = Some(ManagedPipeline {
            observer,
            mutator,
            run_state,
            stop: Some(stop),
            thread: Some(thread),
            revision: 0,
            inserted: BTreeMap::new(),
        });
        json!({ "ok": true, "state": "running", "revision": 0 })
    }

    fn set_log_level(&mut self, args: &Value) -> Result<Value, (i64, String)> {
        let level = parse_log_level(
            args.get("level")
                .ok_or((-32602, "set_log_level needs `level`".into()))?,
        )?;
        let category = args.get("category").and_then(Value::as_str);
        let previous = match category {
            Some(category) if !category.is_empty() => {
                let previous = self
                    .category_log_levels
                    .insert(category.to_string(), level)
                    .unwrap_or(self.default_log_level);
                g2g_core::log::set_category_level(category, level);
                previous
            }
            Some(_) => return Err((-32602, "`category` must not be empty".into())),
            None => {
                let previous = self.default_log_level;
                self.default_log_level = level;
                g2g_core::log::set_default_level(level);
                previous
            }
        };
        Ok(json!({
            "ok": true,
            "category": category,
            "level": level.as_str().to_ascii_lowercase(),
            "previous_level": previous.as_str().to_ascii_lowercase(),
        }))
    }

    fn tail_logs(&self, args: &Value) -> Result<Value, (i64, String)> {
        let limit = args
            .get("limit")
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .filter(|value| (1..=LOG_CAPACITY).contains(value))
                    .ok_or((-32602, "`limit` must be between 1 and 1024".into()))
            })
            .transpose()?
            .unwrap_or(100);
        let clear = args.get("clear").and_then(Value::as_bool).unwrap_or(false);
        let records = if clear {
            self.logs.drain()
        } else {
            self.logs.snapshot()
        };
        let skipped = records.len().saturating_sub(limit);
        let records = records
            .into_iter()
            .skip(skipped)
            .map(log_record_json)
            .collect::<Vec<_>>();
        Ok(json!({
            "ok": true,
            "capacity": self.logs.capacity(),
            "overwritten": self.logs.overwritten(),
            "cleared": clear,
            "records": records,
        }))
    }

    fn sample_edge(&self, args: &Value) -> Result<Value, (i64, String)> {
        let edge = required_u64(args, "edge", "sample_edge")?;
        let edge = usize::try_from(edge).map_err(|_| (-32602, "`edge` is too large".into()))?;
        let count = bounded_usize(
            args.get("count"),
            DEFAULT_PACKET_SAMPLE_COUNT,
            MAX_PACKET_SAMPLE_COUNT,
            "count",
        )?;
        let timeout_ms = bounded_u64(
            args.get("timeout_ms"),
            DEFAULT_PACKET_SAMPLE_TIMEOUT_MS,
            MAX_PACKET_SAMPLE_TIMEOUT_MS,
            "timeout_ms",
        )?;
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(json!({ "ok": false, "error": "no managed pipeline" }));
        };
        let observer = pipeline.observer.clone();
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        let (slot, caps) = loop {
            if let (Some(slot), Some(caps)) = (observer.edge_probe(edge), observer.edge_caps(edge))
            {
                break (slot, caps);
            }
            let edge_count = observer.edge_count();
            if edge_count > 0 && edge >= edge_count {
                return Ok(json!({
                    "ok": false,
                    "error": "edge index is out of range",
                    "edge_count": edge_count,
                }));
            }
            if Instant::now() >= deadline {
                return Ok(json!({
                    "ok": false,
                    "error": "pipeline did not expose the edge before the timeout",
                }));
            }
            std::thread::sleep(Duration::from_millis(1));
        };

        let (sender, receiver) = mpsc::sync_channel(count);
        slot.install(Arc::new(PacketSampler {
            caps: Mutex::new(caps.clone()),
            sender,
            remaining: AtomicUsize::new(count),
        }));
        let mut samples = Vec::with_capacity(count);
        while samples.len() < count {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match receiver.recv_timeout(remaining) {
                Ok(sample) => samples.push(sample),
                Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
        }
        slot.remove();
        Ok(json!({
            "ok": true,
            "edge": edge,
            "caps": caps.to_gst_string(),
            "requested": count,
            "timed_out": samples.len() < count,
            "samples": samples,
        }))
    }

    fn validate_insertion(&mut self, args: &Value) -> Result<Value, (i64, String)> {
        let request = insertion_request(args)?;
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(json!({ "ok": false, "error": "no managed pipeline" }));
        };
        if request.expected_revision != pipeline.revision {
            return Ok(revision_mismatch(pipeline.revision));
        }
        let mutator = pipeline.mutator.clone();
        let element = build_transform(&self.registry, request.element, request.properties)?;
        let result = match request.position {
            "before" => self
                .runtime
                .block_on(mutator.validate_insert_before(request.target, element)),
            "after" => self
                .runtime
                .block_on(mutator.validate_insert_after(request.target, element)),
            _ => return Err((-32602, "position must be `before` or `after`".into())),
        };
        Ok(match result {
            Ok(()) => json!({ "ok": true, "revision": pipeline.revision }),
            Err(error) => json!({ "ok": false, "error": format!("{error:?}") }),
        })
    }

    fn insert_transform(&mut self, args: &Value) -> Result<Value, (i64, String)> {
        let request = insertion_request(args)?;
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(json!({ "ok": false, "error": "no managed pipeline" }));
        };
        if request.expected_revision != pipeline.revision {
            return Ok(revision_mismatch(pipeline.revision));
        }
        let mutator = pipeline.mutator.clone();
        let element = build_transform(&self.registry, request.element, request.properties)?;
        let result = match request.position {
            "before" => self
                .runtime
                .block_on(mutator.insert_before(request.target, element)),
            "after" => self
                .runtime
                .block_on(mutator.insert_after(request.target, element)),
            _ => return Err((-32602, "position must be `before` or `after`".into())),
        };
        let inserted = match result {
            Ok(inserted) => inserted,
            Err(error) => return Ok(json!({ "ok": false, "error": format!("{error:?}") })),
        };
        let pipeline = self.pipeline.as_mut().expect("managed pipeline");
        pipeline.revision = pipeline.revision.saturating_add(1);
        pipeline.inserted.insert(
            inserted.clone(),
            json!({
                "element": request.element,
                "properties": request.properties,
                "position": request.position,
                "target": request.target,
            }),
        );
        Ok(json!({
            "ok": true,
            "node": inserted,
            "revision": pipeline.revision,
        }))
    }

    fn remove_transform(&mut self, args: &Value) -> Result<Value, (i64, String)> {
        let node = required_string(args, "node", "remove_transform")?;
        let expected_revision = required_u64(args, "expected_revision", "remove_transform")?;
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(json!({ "ok": false, "error": "no managed pipeline" }));
        };
        if expected_revision != pipeline.revision {
            return Ok(revision_mismatch(pipeline.revision));
        }
        if !pipeline.inserted.contains_key(node) {
            return Ok(json!({
                "ok": false,
                "error": "only transforms inserted through this MCP server can be removed",
            }));
        }
        let mutator = pipeline.mutator.clone();
        if let Err(error) = self.runtime.block_on(mutator.remove(node)) {
            return Ok(json!({ "ok": false, "error": format!("{error:?}") }));
        }
        let pipeline = self.pipeline.as_mut().expect("managed pipeline");
        pipeline.inserted.remove(node);
        pipeline.revision = pipeline.revision.saturating_add(1);
        Ok(json!({ "ok": true, "revision": pipeline.revision }))
    }

    fn stop_pipeline(&mut self) -> Value {
        let Some(mut pipeline) = self.pipeline.take() else {
            return json!({ "ok": false, "error": "no managed pipeline" });
        };
        pipeline.stop();
        let final_status = pipeline.status_json();
        json!({ "ok": true, "final_status": final_status })
    }
}

struct PacketSampler {
    caps: Mutex<g2g_core::Caps>,
    sender: mpsc::SyncSender<Value>,
    remaining: AtomicUsize,
}

impl LinkInterceptor for PacketSampler {
    fn on_packet(&self, packet: &PipelinePacket) -> ProbeAction {
        if self
            .remaining
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                (remaining > 0).then_some(remaining - 1)
            })
            .is_err()
        {
            return ProbeAction::Pass;
        }
        let mut caps = self.caps.lock().expect("packet sampler caps");
        if let PipelinePacket::CapsChanged(changed) = packet {
            *caps = changed.clone();
        }
        let _ = self.sender.try_send(packet_json(packet, &caps));
        ProbeAction::Pass
    }
}

fn packet_json(packet: &PipelinePacket, caps: &g2g_core::Caps) -> Value {
    match packet {
        PipelinePacket::CapsChanged(caps) => {
            json!({ "kind": "caps", "caps": caps.to_gst_string() })
        }
        PipelinePacket::DataFrame(frame) => json!({
            "kind": "frame",
            "sequence": frame.sequence,
            "pts_ns": frame.timing.pts(),
            "dts_ns": frame.timing.dts_ns,
            "duration_ns": frame.timing.duration_ns,
            "capture_ns": frame.timing.capture_ns,
            "arrival_ns": frame.timing.arrival_ns,
            "keyframe": frame.timing.keyframe,
            "memory": format!("{:?}", frame.domain.kind()).to_ascii_lowercase(),
            "bytes": frame.domain.as_system_slice().map(<[u8]>::len),
            "preview": packet_preview(packet, caps),
        }),
        PipelinePacket::Eos => json!({ "kind": "eos" }),
        PipelinePacket::Flush => json!({ "kind": "flush" }),
        PipelinePacket::Segment(segment) => json!({
            "kind": "segment",
            "rate": segment.rate,
            "applied_rate": segment.applied_rate,
            "base": segment.base,
            "start": segment.start,
            "stop": segment.stop,
            "time": segment.time,
            "position": segment.position,
            "key_units_only": segment.key_units_only,
        }),
        PipelinePacket::Tick => json!({ "kind": "tick" }),
        _ => json!({ "kind": "unknown" }),
    }
}

fn parse_log_level(value: &Value) -> Result<LogLevel, (i64, String)> {
    let parsed = match value {
        Value::String(value) => LogLevel::parse(value),
        Value::Number(value) => value
            .as_u64()
            .and_then(|value| u8::try_from(value).ok())
            .and_then(LogLevel::from_u8),
        _ => None,
    };
    parsed.ok_or((
        -32602,
        "`level` must be off, error, warn, fixme, info, debug, log, trace, or 0 through 7".into(),
    ))
}

fn log_record_json(record: g2g_core::log::OwnedLogRecord) -> Value {
    let fields = record
        .fields
        .into_iter()
        .map(|field| (field.key.into_owned(), log_value_json(field.value)))
        .collect::<serde_json::Map<_, _>>();
    json!({
        "level": record.level.as_str().to_ascii_lowercase(),
        "category": record.category,
        "instance": record.instance,
        "timestamp_ns": record.timestamp_ns,
        "message": record.message,
        "fields": fields,
    })
}

fn log_value_json(value: LogValue<'_>) -> Value {
    match value {
        LogValue::Str(value) => Value::String(value.into_owned()),
        LogValue::Int(value) => Value::from(value),
        LogValue::Uint(value) => Value::from(value),
        LogValue::Float(value) => serde_json::Number::from_f64(value)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        LogValue::Bool(value) => Value::from(value),
    }
}

fn bounded_usize(
    value: Option<&Value>,
    default: usize,
    maximum: usize,
    field: &str,
) -> Result<usize, (i64, String)> {
    let Some(value) = value else {
        return Ok(default);
    };
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| (1..=maximum).contains(value))
        .ok_or((-32602, format!("`{field}` must be between 1 and {maximum}")))
}

fn bounded_u64(
    value: Option<&Value>,
    default: u64,
    maximum: u64,
    field: &str,
) -> Result<u64, (i64, String)> {
    let Some(value) = value else {
        return Ok(default);
    };
    value
        .as_u64()
        .filter(|value| (1..=maximum).contains(value))
        .ok_or((-32602, format!("`{field}` must be between 1 and {maximum}")))
}

struct InsertionRequest<'a> {
    target: &'a str,
    position: &'a str,
    element: &'a str,
    properties: Option<&'a Value>,
    expected_revision: u64,
}

fn insertion_request(args: &Value) -> Result<InsertionRequest<'_>, (i64, String)> {
    Ok(InsertionRequest {
        target: required_string(args, "target", "insertion")?,
        position: required_string(args, "position", "insertion")?,
        element: required_string(args, "element", "insertion")?,
        properties: args.get("properties"),
        expected_revision: required_u64(args, "expected_revision", "insertion")?,
    })
}

fn required_string<'a>(args: &'a Value, field: &str, tool: &str) -> Result<&'a str, (i64, String)> {
    args.get(field)
        .and_then(Value::as_str)
        .ok_or((-32602, format!("{tool} needs `{field}`")))
}

fn required_u64(args: &Value, field: &str, tool: &str) -> Result<u64, (i64, String)> {
    args.get(field)
        .and_then(Value::as_u64)
        .ok_or((-32602, format!("{tool} needs `{field}`")))
}

fn revision_mismatch(current: u64) -> Value {
    json!({
        "ok": false,
        "error": "pipeline revision changed",
        "current_revision": current,
    })
}

fn build_transform(
    registry: &Registry,
    name: &str,
    properties: Option<&Value>,
) -> Result<Box<dyn g2g_core::element::DynAsyncElement>, (i64, String)> {
    let mut element = registry
        .make_element(name)
        .ok_or((-32602, format!("no transform or sink named `{name}`")))?;
    let Some(properties) = properties else {
        return Ok(element);
    };
    let properties = properties
        .as_object()
        .ok_or((-32602, "`properties` must be an object".into()))?;
    for (property_name, property_value) in properties {
        let specs = element.properties();
        let spec = specs.iter().find(|spec| spec.name == property_name);
        let value = match spec {
            Some(spec) => {
                if !spec.flags.writable {
                    return Err((-32602, format!("`{property_name}` is read-only")));
                }
                let raw = property_text(property_value)?;
                spec.parse_value(&raw).map_err(|_| {
                    (
                        -32602,
                        format!("bad value for `{name}.{property_name}`: {raw}"),
                    )
                })?
            }
            None if takes_undeclared_properties(specs) => {
                PropValue::Str(property_text(property_value)?)
            }
            None => {
                return Err((-32602, format!("no property `{property_name}` on `{name}`")));
            }
        };
        element
            .set_property(property_name, value)
            .map_err(|error| {
                (
                    -32602,
                    format!("cannot set `{name}.{property_name}`: {error:?}"),
                )
            })?;
    }
    Ok(element)
}

fn property_text(value: &Value) -> Result<String, (i64, String)> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Number(value) => Ok(value.to_string()),
        Value::Array(values) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or((-32602, "property arrays must contain strings".into()))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|values| values.join("+")),
        _ => Err((
            -32602,
            "property values must be strings, numbers, booleans, or string arrays".into(),
        )),
    }
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

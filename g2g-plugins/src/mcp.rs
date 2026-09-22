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
//! `start_pipeline` keeps one launch-line pipeline running on a background
//! thread for `pipeline_status` / `tail_events` / `tail_logs` / `sample_edge` and
//! the revision-checked `insert_transform` / `remove_transform`. A host
//! application that already runs a pipeline hands its `Observer`,
//! `GraphMutator` and `Bus` to [`McpServer::register_pipeline`] instead and
//! keeps the lifecycle: `stop_pipeline` refuses a registered pipeline.
//!
//! The managed run's results come back through the record tools, the live
//! property tools, and a picture of one edge:
//!
//!   latest_metadata  {count}              -> the newest `metasink` records
//!   wait_for_records {count, key, ...}    -> block until new ones arrive
//!   load_metadata    {path}               -> read a finished run's file instead
//!   get_property     {element, property}  -> read one live element's knob
//!   set_property     {element, ..., value}-> write it between packets
//!   snapshot_frame   {element|edge}       -> the newest frame as a PNG image block
//!   clip_at          {source, pts, secs}  -> cut a clip out of a video file
//!
//! `prompts/list` / `prompts/get` serve one prompt per `###` section of the
//! README's sample pipelines, read from disk at call time (`G2G_README` points
//! elsewhere). `describe_frame` and `search_video` are not here: they need the
//! Python models `pyml_mcp` hosts.
//!
//! No MCP framework dependency: the JSON-RPC envelope is hand-rolled over
//! stdin/stdout with serde_json. Needs the `observe` and `multi-thread`
//! features.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::cell::Cell;
use std::collections::{BTreeMap, VecDeque};
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

#[cfg(feature = "mcp")]
use base64::Engine as _;

use g2g_core::log::{LogLevel, LogValue, RingSink, SinkId};
use g2g_core::property::{takes_undeclared_properties, PropKind, PropValue};
use g2g_core::runtime::{
    parse_launch, run_graph_observed_mutable, select2, Either, GraphMutator, LinkInterceptor,
    Observer, ProbeAction, Registry, RunStats,
};
#[cfg(feature = "mcp")]
use g2g_core::runtime::{NodeRole, TelemetrySnapshot};
use g2g_core::{Bus, BusMessage, PipelinePacket};

use crate::clock::WallClock;
use crate::preview::packet_preview;
use crate::registry::default_registry;
use crate::toolingjson::{
    launch_json, registry_json, stats_json, telemetry_json, validate_json, TelemetryTap,
};

const PROTOCOL_VERSION: &str = "2024-11-05";
const LOG_CAPACITY: usize = 1024;
const EVENT_CAPACITY: usize = 1024;
const DEFAULT_PACKET_SAMPLE_COUNT: usize = 1;
const MAX_PACKET_SAMPLE_COUNT: usize = 32;
const DEFAULT_PACKET_SAMPLE_TIMEOUT_MS: u64 = 1000;
const MAX_PACKET_SAMPLE_TIMEOUT_MS: u64 = 30_000;
const RECORD_CAPACITY: usize = 1000;
const DEFAULT_RECORD_COUNT: usize = 10;
const DEFAULT_RECORD_WAIT_MS: u64 = 30_000;
const MAX_RECORD_WAIT_MS: u64 = 120_000;
/// How long a `wait_for_records` sleep runs before it rechecks the run state,
/// which wakes no one when the pipeline fails.
const RECORD_WAIT_SLICE: Duration = Duration::from_millis(50);
/// The key a record line that is not JSON comes back under.
const RAW_RECORD_KEY: &str = "raw";
#[cfg(feature = "mcp")]
const SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(feature = "mcp")]
const SNAPSHOT_MIME_TYPE: &str = "image/png";
const DEFAULT_CLIP_SECONDS: f64 = 4.0;
const DEFAULT_CLIP_DECODER: &str = "decodebin";
const DEFAULT_CLIP_ENCODER: &str = "mjpegenc ! avimux";
/// The container [`DEFAULT_CLIP_ENCODER`] writes, for the default `location`.
const DEFAULT_CLIP_EXTENSION: &str = ".avi";
/// How long a clip run gets to decode, cut and mux before it is abandoned.
const CLIP_DEADLINE_SECS: u64 = 60;
/// The clip line's own converter, whose processed count is the clip's frames.
const CLIP_FRAME_COUNTER: &str = "clip-frames";
const NS_PER_SECOND: f64 = 1_000_000_000.0;
/// Where the README the prompts are built from lives, and the variable that
/// points somewhere else. Read at call time: the crate is published without it.
const README_PATH_ENV: &str = "G2G_README";
const README_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../README.md");
/// The README heading whose `###` subsections become prompts.
const PIPELINE_SECTION_HEADING: &str = "## Sample pipelines";
static PIPELINE_CLOCK: OnceLock<WallClock> = OnceLock::new();

/// The MCP server: the tool dispatcher plus at most one managed pipeline.
///
/// Constructing one subscribes a 1024-record [`RingSink`] beside the host's log
/// sink (through [`g2g_core::log::add_sink`], so the host keeps its own sink)
/// and installs the UNIX time source when none is set. Dropping it unsubscribes
/// the ring and stops a pipeline it started itself.
pub struct McpServer {
    registry: Registry,
    runtime: tokio::runtime::Runtime,
    pipeline: Option<ManagedPipeline>,
    records: RecordStore,
    logs: RingSink,
    log_sink_id: SinkId,
}

impl core::fmt::Debug for McpServer {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("McpServer")
            .field("pipeline", &self.pipeline.is_some())
            .field("buffered_logs", &self.logs.len())
            .finish_non_exhaustive()
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

impl McpServer {
    /// A server over the standard element registry.
    pub fn new() -> Self {
        Self::with_registry(default_registry())
    }

    /// A server over `registry`, which `inspect`, `validate`, the run tools and
    /// `insert_transform` all build from.
    pub fn with_registry(registry: Registry) -> Self {
        let logs = RingSink::new(LOG_CAPACITY);
        let log_sink_id = g2g_core::log::add_sink(Box::new(logs.clone()));
        if g2g_core::log::timestamp_now().is_none() {
            g2g_core::log::set_time_source(g2g_core::log::unix_time_source);
        }
        Self {
            registry,
            runtime: tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build tokio runtime"),
            pipeline: None,
            records: RecordStore::new(),
            logs,
            log_sink_id,
        }
    }

    /// Serve newline-delimited JSON-RPC on stdin / stdout until stdin closes.
    pub fn serve_stdio(&mut self) {
        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(line) => line,
                Err(_) => break,
            };
            if line.trim().is_empty() {
                continue;
            }
            let request: Value = match serde_json::from_str(&line) {
                Ok(request) => request,
                Err(_) => continue,
            };
            let method = request
                .get("method")
                .and_then(|method| method.as_str())
                .unwrap_or("");
            let id = request.get("id").cloned();
            let result = self.dispatch(method, request.get("params"));
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

    /// Handle one JSON-RPC method (`initialize`, `tools/list`, `tools/call`,
    /// ...) and return its `result`, or the `(code, message)` of its `error`.
    /// An embedding host calls this directly instead of [`serve_stdio`](Self::serve_stdio).
    pub fn dispatch(
        &mut self,
        method: &str,
        params: Option<&Value>,
    ) -> Result<Value, (i64, String)> {
        match method {
            "initialize" => Ok(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": { "tools": {}, "prompts": {} },
                "serverInfo": { "name": "g2g-mcp", "version": env!("CARGO_PKG_VERSION") },
            })),
            "tools/list" => Ok(json!({ "tools": tool_specs() })),
            "tools/call" => self.call_tool(params),
            "prompts/list" => Ok(json!({ "prompts": prompt_specs() })),
            "prompts/get" => prompt_messages(params),
            "notifications/initialized" | "ping" => Ok(json!({})),
            other => Err((-32601, format!("method not found: {other}"))),
        }
    }

    /// Expose a pipeline the host runs itself (its `run_graph_observed_mutable`
    /// handles) to the pipeline tools. The mutator must take `'static`
    /// elements, as one over a `parse_launch` graph does, since the registry
    /// builds the transforms `insert_transform` hands it. The host keeps the
    /// run's lifecycle:
    /// `stop_pipeline` refuses it, and the host reports the run's end through
    /// the returned [`RegisteredPipelineHandle`]. `bus` becomes the server's:
    /// a thread drains it into the `tail_events` buffer, so the host must not
    /// also expect to receive from it. Fails while a pipeline is already
    /// managed or registered.
    pub fn register_pipeline(
        &mut self,
        observer: Observer,
        mutator: GraphMutator<'static>,
        bus: Bus,
    ) -> Result<RegisteredPipelineHandle, RegistrationError> {
        if self.pipeline.is_some() {
            return Err(RegistrationError::PipelineAlreadyRegistered);
        }
        let events = EventBuffer::new(EVENT_CAPACITY);
        self.records.reset();
        let event_collector = EventCollector::spawn(bus, events.clone(), self.records.clone())
            .map_err(RegistrationError::EventCollector)?;
        let run_state = Arc::new(Mutex::new(PipelineRunState::Running));
        let handle = RegisteredPipelineHandle {
            run_state: run_state.clone(),
        };
        self.pipeline = Some(ManagedPipeline {
            observer,
            mutator,
            run_state,
            lifecycle: PipelineLifecycle::Registered,
            events,
            event_collector: Some(event_collector),
            revision: 0,
            inserted: BTreeMap::new(),
        });
        Ok(handle)
    }

    /// Drop the registered pipeline's handles and stop draining its bus. The
    /// host's run keeps going. `false` if no pipeline is registered or managed.
    pub fn unregister_pipeline(&mut self) -> bool {
        let Some(mut pipeline) = self.pipeline.take() else {
            return false;
        };
        pipeline.stop();
        true
    }
}

impl Drop for McpServer {
    fn drop(&mut self) {
        if let Some(mut pipeline) = self.pipeline.take() {
            pipeline.stop();
        }
        g2g_core::log::remove_sink(self.log_sink_id);
    }
}

#[derive(Debug)]
enum PipelineRunState {
    Running,
    Finished(Option<Value>),
    Failed(String),
    Stopped,
}

#[derive(Debug)]
enum PipelineLifecycle {
    Owned {
        stop: Option<tokio::sync::oneshot::Sender<()>>,
        thread: Option<JoinHandle<()>>,
    },
    Registered,
}

/// The host's side of a registered pipeline: how it reports the run's end so
/// `pipeline_status` stops saying `running`.
#[derive(Debug, Clone)]
pub struct RegisteredPipelineHandle {
    run_state: Arc<Mutex<PipelineRunState>>,
}

impl RegisteredPipelineHandle {
    /// The run ended normally with `stats`.
    pub fn finish(&self, stats: &RunStats) {
        *self.run_state.lock().expect("pipeline run state") =
            PipelineRunState::Finished(Some(stats_json(stats)));
    }

    /// The run ended normally, with no stats to report.
    pub fn finish_without_stats(&self) {
        *self.run_state.lock().expect("pipeline run state") = PipelineRunState::Finished(None);
    }

    /// The run ended with `error`.
    pub fn fail(&self, error: impl Into<String>) {
        *self.run_state.lock().expect("pipeline run state") =
            PipelineRunState::Failed(error.into());
    }

    /// The host stopped the run before it ended on its own.
    pub fn stop(&self) {
        *self.run_state.lock().expect("pipeline run state") = PipelineRunState::Stopped;
    }
}

/// Why [`McpServer::register_pipeline`] refused.
#[derive(Debug)]
pub enum RegistrationError {
    /// The server already has a managed or registered pipeline.
    PipelineAlreadyRegistered,
    /// The thread that drains the bus could not be spawned.
    EventCollector(std::io::Error),
}

impl core::fmt::Display for RegistrationError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PipelineAlreadyRegistered => {
                formatter.write_str("a pipeline is already registered")
            }
            Self::EventCollector(error) => {
                write!(formatter, "cannot start bus event collector: {error}")
            }
        }
    }
}

impl std::error::Error for RegistrationError {}

#[derive(Debug, Clone)]
struct EventBuffer {
    inner: Arc<Mutex<EventBufferState>>,
}

#[derive(Debug)]
struct EventBufferState {
    capacity: usize,
    next_sequence: u64,
    overwritten: u64,
    events: VecDeque<Value>,
}

impl EventBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(EventBufferState {
                capacity,
                next_sequence: 0,
                overwritten: 0,
                events: VecDeque::with_capacity(capacity),
            })),
        }
    }

    fn push(&self, event: Value) {
        let mut state = self.inner.lock().expect("bus event buffer");
        if state.events.len() == state.capacity {
            state.events.pop_front();
            state.overwritten = state.overwritten.saturating_add(1);
        }
        let sequence = state.next_sequence;
        state.next_sequence = state.next_sequence.saturating_add(1);
        state.events.push_back(json!({
            "sequence": sequence,
            "observed_ns": g2g_core::metrics::monotonic_ns(),
            "event": event,
        }));
    }

    fn read(&self, limit: usize, clear: bool) -> (Vec<Value>, u64) {
        let mut state = self.inner.lock().expect("bus event buffer");
        let skipped = state.events.len().saturating_sub(limit);
        let events = state.events.iter().skip(skipped).cloned().collect();
        if clear {
            state.events.clear();
        }
        (events, state.overwritten)
    }

    fn len(&self) -> usize {
        self.inner.lock().expect("bus event buffer").events.len()
    }
}

/// The metadata records a `metasink` posted on the bus, oldest first, with a
/// count of everything ever posted (so a waiter can tell its own arrivals from
/// the tail it already saw) and whether anything more can arrive. It belongs to
/// the server rather than a pipeline: `load_metadata` fills it from a file with
/// nothing running.
#[derive(Debug, Clone)]
struct RecordStore {
    inner: Arc<(Mutex<RecordStoreState>, Condvar)>,
}

#[derive(Debug, Default)]
struct RecordStoreState {
    records: VecDeque<Value>,
    posted: u64,
    ended: bool,
}

impl RecordStore {
    fn new() -> Self {
        Self {
            inner: Arc::new((Mutex::new(RecordStoreState::default()), Condvar::new())),
        }
    }

    fn lock(&self) -> MutexGuard<'_, RecordStoreState> {
        self.inner.0.lock().expect("metadata record store")
    }

    fn push(&self, record: Value) {
        push_record(&mut self.lock(), record);
        self.inner.1.notify_all();
    }

    /// Nothing more will arrive: the run reached EOS, or a file was loaded.
    fn end(&self) {
        self.lock().ended = true;
        self.inner.1.notify_all();
    }

    /// Start over for a new run, releasing whoever waits on the old one.
    fn reset(&self) {
        *self.lock() = RecordStoreState::default();
        self.inner.1.notify_all();
    }

    fn ended(&self) -> bool {
        self.lock().ended
    }

    fn latest(&self, count: usize) -> Vec<Value> {
        let state = self.lock();
        let skipped = state.records.len().saturating_sub(count);
        state.records.iter().skip(skipped).cloned().collect()
    }

    /// Replace the store with the JSON lines of a finished run.
    fn load(&self, text: &str) -> u64 {
        let mut state = self.lock();
        *state = RecordStoreState::default();
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            push_record(&mut state, record_value(line));
        }
        state.ended = true;
        self.inner.1.notify_all();
        state.posted
    }

    /// Block until `count` records carrying `key` have been posted since this
    /// call, or nothing more can arrive, or the timeout passes. `run_ended`
    /// answers for the pipeline behind the store, which posts no wakeup of its
    /// own when it fails.
    fn wait(
        &self,
        count: usize,
        key: &str,
        timeout: Duration,
        run_ended: &dyn Fn() -> bool,
    ) -> Vec<Value> {
        let deadline = Instant::now() + timeout;
        let mut state = self.lock();
        // a finished run posts nothing more, so its whole tail is fresh (bus eos may lag the run state)
        let posted_before = if state.ended || run_ended() {
            0
        } else {
            state.posted
        };
        loop {
            let matched = records_since(&state, posted_before, key);
            if matched.len() >= count || state.ended || run_ended() {
                return matched;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return matched;
            }
            state = self
                .inner
                .1
                .wait_timeout(state, left.min(RECORD_WAIT_SLICE))
                .expect("metadata record store")
                .0;
        }
    }
}

fn push_record(state: &mut RecordStoreState, record: Value) {
    if state.records.len() == RECORD_CAPACITY {
        state.records.pop_front();
    }
    state.records.push_back(record);
    state.posted = state.posted.saturating_add(1);
}

/// The records posted after `posted_before` that carry `key`, oldest first. An
/// empty `key` matches every record.
fn records_since(state: &RecordStoreState, posted_before: u64, key: &str) -> Vec<Value> {
    let fresh = usize::try_from(state.posted.saturating_sub(posted_before))
        .unwrap_or(RECORD_CAPACITY)
        .min(state.records.len());
    state
        .records
        .iter()
        .skip(state.records.len() - fresh)
        .filter(|record| key.is_empty() || record.get(key).is_some())
        .cloned()
        .collect()
}

/// One `metasink` line as a value. A line that is not JSON still reaches the
/// agent, under its own key, rather than disappearing.
fn record_value(line: &str) -> Value {
    serde_json::from_str(line).unwrap_or_else(|_| json!({ RAW_RECORD_KEY: line }))
}

#[derive(Debug)]
struct EventCollector {
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl EventCollector {
    fn spawn(bus: Bus, events: EventBuffer, records: RecordStore) -> std::io::Result<Self> {
        let (stop, stop_receiver) = tokio::sync::oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("g2g-mcp-events".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("build bus event runtime");
                runtime.block_on(async move {
                    let drain = async move {
                        while let Some(message) = bus.recv().await {
                            match &message {
                                BusMessage::MetadataRecord { record, .. } => {
                                    records.push(record_value(record));
                                }
                                BusMessage::Eos => records.end(),
                                _ => {}
                            }
                            if let Some(event) = crate::dashboard::event_value(&message) {
                                events.push(event);
                            }
                        }
                    };
                    let _ = select2(stop_receiver, drain).await;
                });
            })?;
        Ok(Self {
            stop: Some(stop),
            thread: Some(thread),
        })
    }

    fn stop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct ManagedPipeline {
    observer: Observer,
    mutator: GraphMutator<'static>,
    run_state: Arc<Mutex<PipelineRunState>>,
    lifecycle: PipelineLifecycle,
    events: EventBuffer,
    event_collector: Option<EventCollector>,
    revision: u64,
    inserted: BTreeMap<String, Value>,
}

impl ManagedPipeline {
    fn stop(&mut self) {
        if let PipelineLifecycle::Owned { stop, thread } = &mut self.lifecycle {
            if let Some(stop) = stop.take() {
                let _ = stop.send(());
            }
            if thread.take().is_some_and(|thread| thread.join().is_err()) {
                *self.run_state.lock().expect("pipeline run state") =
                    PipelineRunState::Failed("pipeline thread panicked".into());
            }
        }
        if let Some(mut collector) = self.event_collector.take() {
            collector.stop();
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
            "events": {
                "buffered": self.events.len(),
                "capacity": EVENT_CAPACITY,
            },
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
            "description": "Start one pipeline in the background for live inspection and mutation. \
                            Give a metasink a location=: a line whose records would land in this \
                            server's stdout, which carries the protocol, is refused.",
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
            "name": "tail_events",
            "description": "Read errors, warnings, QoS, state changes, and other bus events from the bounded event tail.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "minimum": 1, "maximum": 1024 },
                    "clear": { "type": "boolean" }
                }
            }
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
            "name": "latest_metadata",
            "description": "The newest records a metasink posted, oldest first: each has a pts in \
                            seconds plus detections, text, or a JSON blob such as alert.",
            "inputSchema": {
                "type": "object",
                "properties": { "count": { "type": "integer", "minimum": 1, "maximum": 1000 } }
            }
        },
        {
            "name": "wait_for_records",
            "description": "Wait for a metasink to post new records, oldest first, and return them \
                            with the pipeline status. Give a key such as detections or alert to wait \
                            only for records carrying it. Returns whatever arrived when the pipeline \
                            ends, fails, or the timeout passes first.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "count": { "type": "integer", "minimum": 1, "maximum": 1000 },
                    "key": { "type": "string" },
                    "timeout_ms": { "type": "integer", "minimum": 1, "maximum": 120000 }
                }
            }
        },
        {
            "name": "load_metadata",
            "description": "Load the JSON lines a metasink wrote to a file as the current records, \
                            so latest_metadata and wait_for_records read a finished run with no \
                            pipeline running. Stops the managed pipeline.",
            "inputSchema": {
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }
        },
        {
            "name": "get_property",
            "description": "Read a property of a named element of the managed pipeline.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "element": { "type": "string" },
                    "property": { "type": "string" }
                },
                "required": ["element", "property"]
            }
        },
        {
            "name": "set_property",
            "description": "Set a property on a named element of the managed pipeline, applied \
                            between packets, and report the value read back. A source or a tee \
                            takes no live property.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "element": { "type": "string" },
                    "property": { "type": "string" },
                    "value": { "type": ["string", "number", "boolean"] }
                },
                "required": ["element", "property", "value"]
            }
        },
        {
            "name": "clip_at",
            "description": "Cut the seconds of video around a pts out of a video file, through \
                            `filesrc ! decoder ! trim ! videoconvert ! encoder ! filesink`. \
                            Returns where it was written, the range it covers and how many frames \
                            it holds. `location` must carry the extension the encoder's muxer \
                            writes; the default is an AVI in the temporary directory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source": { "type": "string" },
                    "pts": { "type": "number" },
                    "seconds": { "type": "number" },
                    "location": { "type": "string" },
                    "decoder": { "type": "string" },
                    "encoder": { "type": "string" }
                },
                "required": ["source", "pts"]
            }
        },
        {
            "name": "stop_pipeline",
            "description": "Stop and release the managed pipeline.",
            "inputSchema": { "type": "object", "properties": {} }
        }
    ]);
    // The image block needs the png and base64 dependencies the `mcp` feature pulls.
    #[cfg(feature = "mcp")]
    if let Some(list) = tools.as_array_mut() {
        list.push(json!({
            "name": "snapshot_frame",
            "description": "The newest frame reaching an element of the managed pipeline, as a PNG \
                            image. Names the element the frame is about to reach, or an edge index \
                            from pipeline_status. Left empty it takes the metasink, or the pipeline's \
                            only sink.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "element": { "type": "string" },
                    "edge": { "type": "integer", "minimum": 0 }
                }
            }
        }));
    }
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

impl McpServer {
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
                .unwrap_or_else(no_pipeline),
            "tail_events" => self.tail_events(&args)?,
            "set_log_level" => self.set_log_level(&args)?,
            "tail_logs" => self.tail_logs(&args)?,
            "sample_edge" => self.sample_edge(&args)?,
            "latest_metadata" => {
                let count = bounded_usize(
                    args.get("count"),
                    DEFAULT_RECORD_COUNT,
                    RECORD_CAPACITY,
                    "count",
                )?;
                json!({ "ok": true, "records": self.records.latest(count) })
            }
            "wait_for_records" => self.wait_for_records(&args)?,
            "load_metadata" => self.load_metadata(&args)?,
            "get_property" => self.get_property(&args)?,
            "set_property" => self.set_property(&args)?,
            "clip_at" => self.clip_at(&args)?,
            #[cfg(feature = "mcp")]
            "snapshot_frame" => return self.snapshot_frame(&args),
            "validate_insertion" => self.validate_insertion(&args)?,
            "insert_transform" => self.insert_transform(&args)?,
            "remove_transform" => self.remove_transform(&args)?,
            "stop_pipeline" => self.stop_pipeline(),
            #[cfg(feature = "declarative")]
            "run_graph" => {
                let (doc, yaml) = graph_document(&args)?;
                self.runtime.block_on(crate::toolingjson::document_json(
                    &self.registry,
                    &doc,
                    yaml,
                    duration_secs(&args),
                    tap(&args),
                ))
            }
            other => return Err((-32602, format!("unknown tool: {other}"))),
        };

        Ok(text_result(&payload))
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
        if metasink_writes_to_stdout(&graph) {
            return json!({
                "ok": false,
                "error": "give the metasink a location=: with none its records go to this server's stdout, which carries the protocol",
            });
        }

        let observer = Observer::new();
        let thread_observer = observer.clone();
        let (bus, bus_handle) = Bus::new(256);
        let events = EventBuffer::new(EVENT_CAPACITY);
        self.records.reset();
        let mut event_collector = match EventCollector::spawn(
            bus,
            events.clone(),
            self.records.clone(),
        ) {
            Ok(collector) => collector,
            Err(error) => {
                return json!({ "ok": false, "error": format!("cannot start bus event collector: {error}") });
            }
        };
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
                let (mutator, run) = run_graph_observed_mutable(
                    graph,
                    clock,
                    4,
                    &thread_observer,
                    Some(&bus_handle),
                );
                if ready_sender.send(Ok(mutator)).is_err() {
                    return;
                }
                let state = match runtime.block_on(select2(stop_receiver, run)) {
                    Either::Left(_) => PipelineRunState::Stopped,
                    Either::Right(Ok(stats)) => {
                        PipelineRunState::Finished(Some(stats_json(&stats)))
                    }
                    Either::Right(Err(error)) => PipelineRunState::Failed(format!("{error:?}")),
                };
                *thread_run_state.lock().expect("pipeline run state") = state;
            }) {
            Ok(thread) => thread,
            Err(error) => {
                event_collector.stop();
                return json!({ "ok": false, "error": format!("cannot start pipeline thread: {error}") });
            }
        };
        let mutator = match ready_receiver.recv() {
            Ok(Ok(mutator)) => mutator,
            Ok(Err(error)) => {
                let _ = thread.join();
                event_collector.stop();
                return json!({ "ok": false, "error": error });
            }
            Err(error) => {
                let _ = thread.join();
                event_collector.stop();
                return json!({ "ok": false, "error": format!("pipeline did not start: {error}") });
            }
        };
        self.pipeline = Some(ManagedPipeline {
            observer,
            mutator,
            run_state,
            lifecycle: PipelineLifecycle::Owned {
                stop: Some(stop),
                thread: Some(thread),
            },
            events,
            event_collector: Some(event_collector),
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
                let previous = g2g_core::log::level_for(category);
                g2g_core::log::set_category_level(category, level);
                previous
            }
            Some(_) => return Err((-32602, "`category` must not be empty".into())),
            None => {
                let previous = g2g_core::log::default_level();
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

    fn tail_events(&self, args: &Value) -> Result<Value, (i64, String)> {
        let limit = bounded_usize(args.get("limit"), 100, EVENT_CAPACITY, "limit")?;
        let clear = args.get("clear").and_then(Value::as_bool).unwrap_or(false);
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(no_pipeline());
        };
        let (events, overwritten) = pipeline.events.read(limit, clear);
        Ok(json!({
            "ok": true,
            "capacity": EVENT_CAPACITY,
            "overwritten": overwritten,
            "cleared": clear,
            "events": events,
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
            return Ok(no_pipeline());
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

    fn wait_for_records(&self, args: &Value) -> Result<Value, (i64, String)> {
        let count = bounded_usize(args.get("count"), 1, RECORD_CAPACITY, "count")?;
        let key = args.get("key").and_then(Value::as_str).unwrap_or("");
        let timeout_ms = bounded_u64(
            args.get("timeout_ms"),
            DEFAULT_RECORD_WAIT_MS,
            MAX_RECORD_WAIT_MS,
            "timeout_ms",
        )?;
        if self.pipeline.is_none() && !self.records.ended() {
            return Ok(json!({
                "ok": false,
                "error": "no pipeline is running, call start_pipeline first",
            }));
        }
        let run_state = self
            .pipeline
            .as_ref()
            .map(|pipeline| pipeline.run_state.clone());
        let run_ended = move || {
            run_state.as_ref().is_some_and(|state| {
                !matches!(
                    &*state.lock().expect("pipeline run state"),
                    PipelineRunState::Running
                )
            })
        };
        let records = self
            .records
            .wait(count, key, Duration::from_millis(timeout_ms), &run_ended);
        let status = self
            .pipeline
            .as_ref()
            .map(ManagedPipeline::status_json)
            .unwrap_or_else(|| json!({ "state": "none" }));
        Ok(json!({ "ok": true, "records": records, "status": status }))
    }

    fn load_metadata(&mut self, args: &Value) -> Result<Value, (i64, String)> {
        let path = required_string(args, "path", "load_metadata")?.to_string();
        let text = std::fs::read_to_string(&path)
            .map_err(|error| (-32602, format!("cannot read '{path}': {error}")))?;
        self.stop_pipeline();
        Ok(json!({ "ok": true, "records": self.records.load(&text) }))
    }

    fn get_property(&mut self, args: &Value) -> Result<Value, (i64, String)> {
        let element = required_string(args, "element", "get_property")?;
        let property = required_string(args, "property", "get_property")?;
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(no_pipeline());
        };
        let mutator = pipeline.mutator.clone();
        let read = self
            .runtime
            .block_on(mutator.get_property(element, property));
        Ok(property_result(property, read))
    }

    fn set_property(&mut self, args: &Value) -> Result<Value, (i64, String)> {
        let element = required_string(args, "element", "set_property")?;
        let property = required_string(args, "property", "set_property")?;
        let requested = args
            .get("value")
            .ok_or((-32602, "set_property needs `value`".to_string()))?;
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(no_pipeline());
        };
        let mutator = pipeline.mutator.clone();
        // The element's own current value names the kind to parse for, so a JSON
        // 30 reaches an Int property as an Int and "30/1" reaches a Fraction.
        let kind = match self
            .runtime
            .block_on(mutator.get_property(element, property))
        {
            Ok(current) => current.map(|current| current.kind()),
            Err(error) => return Ok(mutation_error(&error)),
        };
        let value = property_value(requested, kind)?;
        if let Err(error) = self
            .runtime
            .block_on(mutator.set_property(element, property, value))
        {
            return Ok(mutation_error(&error));
        }
        let read = self
            .runtime
            .block_on(mutator.get_property(element, property));
        Ok(property_result(property, read))
    }

    /// Cut `seconds` of video centred on `pts` out of a file, decoding, trimming
    /// and re-encoding it through a one-shot launch line.
    fn clip_at(&mut self, args: &Value) -> Result<Value, (i64, String)> {
        let source = required_string(args, "source", "clip_at")?;
        let pts = args
            .get("pts")
            .and_then(Value::as_f64)
            .ok_or((-32602, "clip_at needs `pts`".to_string()))?;
        let seconds = args
            .get("seconds")
            .and_then(Value::as_f64)
            .unwrap_or(DEFAULT_CLIP_SECONDS);
        if seconds <= 0.0 || !seconds.is_finite() {
            return Err((-32602, "`seconds` must be positive".into()));
        }
        let decoder = args
            .get("decoder")
            .and_then(Value::as_str)
            .unwrap_or(DEFAULT_CLIP_DECODER);
        let encoder = args
            .get("encoder")
            .and_then(Value::as_str)
            .unwrap_or(DEFAULT_CLIP_ENCODER);
        if !std::path::Path::new(source).is_file() {
            return Ok(json!({ "ok": false, "error": format!("no video at '{source}'") }));
        }
        let start = (pts - seconds / 2.0).max(0.0);
        let end = start + seconds;
        let location = args.get("location").and_then(Value::as_str).unwrap_or("");
        let path = if location.is_empty() {
            default_clip_path(source, start)
        } else {
            location.to_string()
        };
        let line = format!(
            "filesrc location={source} ! {decoder} ! trim start={start_ns} stop={end_ns} \
             ! videoconvert name={CLIP_FRAME_COUNTER} ! {encoder} ! filesink location={path}",
            source = launch_quote(source),
            path = launch_quote(&path),
            start_ns = seconds_to_ns(start),
            end_ns = seconds_to_ns(end),
        );
        let outcome =
            self.runtime
                .block_on(launch_json(&self.registry, &line, CLIP_DEADLINE_SECS, None));
        if outcome.get("ok").and_then(Value::as_bool) != Some(true) {
            return Ok(outcome);
        }
        // The muxer hands the sink one byte stream, so the run's frame count is
        // the converter's: the frames the trim kept.
        let frames = outcome
            .pointer("/stats/per_element")
            .and_then(Value::as_array)
            .and_then(|elements| {
                elements
                    .iter()
                    .find(|element| element["name"] == CLIP_FRAME_COUNTER)
            })
            .and_then(|element| element["proc_count"].as_u64());
        let Some(frames) = frames else {
            return Ok(json!({
                "ok": false,
                "error": format!("the clip did not finish within {CLIP_DEADLINE_SECS}s"),
                "pipeline": line,
            }));
        };
        Ok(json!({
            "ok": true,
            "path": path,
            "start": start,
            "end": end,
            "frames": frames,
            "pipeline": line,
        }))
    }

    /// The newest frame on the edge feeding an element, as an image content
    /// block. Returns the whole tool result, not a payload: a picture is not
    /// text.
    #[cfg(feature = "mcp")]
    fn snapshot_frame(&self, args: &Value) -> Result<Value, (i64, String)> {
        let element = args.get("element").and_then(Value::as_str).unwrap_or("");
        let requested_edge = args
            .get("edge")
            .map(|edge| {
                edge.as_u64()
                    .and_then(|edge| usize::try_from(edge).ok())
                    .ok_or((-32602, "`edge` must be an edge index".to_string()))
            })
            .transpose()?;
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(text_result(&no_pipeline()));
        };
        // A run registers its topology after negotiation, so a snapshot taken
        // right after start_pipeline waits for it.
        let deadline = Instant::now() + SNAPSHOT_TIMEOUT;
        let topology = loop {
            let topology = pipeline.observer.snapshot();
            if !topology.nodes.is_empty() || Instant::now() >= deadline {
                break topology;
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        let (edge, element) = match snapshot_edge(&topology, element, requested_edge) {
            Ok(found) => found,
            Err(error) => return Ok(text_result(&json!({ "ok": false, "error": error }))),
        };
        let (Some(slot), Some(caps)) = (
            pipeline.observer.edge_probe(edge),
            pipeline.observer.edge_caps(edge),
        ) else {
            return Ok(text_result(
                &json!({ "ok": false, "error": "edge index is out of range" }),
            ));
        };

        let (sender, receiver) = mpsc::sync_channel(1);
        slot.install(Arc::new(FrameCapture {
            caps: Mutex::new(caps),
            sender,
            remaining: AtomicUsize::new(1),
        }));
        let captured = receiver.recv_timeout(deadline.saturating_duration_since(Instant::now()));
        slot.remove();
        let Ok(captured) = captured else {
            return Ok(text_result(&json!({
                "ok": false,
                "error": format!("no frame reached `{element}` within {SNAPSHOT_TIMEOUT:?}"),
            })));
        };
        let image = match snapshot_png(&captured, &element) {
            Ok(image) => image,
            Err(error) => return Ok(text_result(&json!({ "ok": false, "error": error }))),
        };
        let description = json!({
            "ok": true,
            "element": element,
            "edge": edge,
            "width": image.width,
            "height": image.height,
            "pts_ns": captured.pts_ns,
        });
        Ok(json!({
            "content": [
                {
                    "type": "image",
                    "data": base64::engine::general_purpose::STANDARD.encode(&image.png),
                    "mimeType": SNAPSHOT_MIME_TYPE,
                },
                text_block(&description),
            ]
        }))
    }

    fn validate_insertion(&mut self, args: &Value) -> Result<Value, (i64, String)> {
        let request = insertion_request(args)?;
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(no_pipeline());
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
            return Ok(no_pipeline());
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
            return Ok(no_pipeline());
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
        if self
            .pipeline
            .as_ref()
            .is_some_and(|pipeline| matches!(&pipeline.lifecycle, PipelineLifecycle::Registered))
        {
            return json!({
                "ok": false,
                "error": "the host owns this registered pipeline lifecycle",
            });
        }
        let Some(mut pipeline) = self.pipeline.take() else {
            return no_pipeline();
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
                remaining.checked_sub(1)
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

/// An MCP tool result carrying one JSON payload: results are content blocks, and
/// everything but a picture rides as text.
fn text_result(payload: &Value) -> Value {
    json!({ "content": [text_block(payload)] })
}

fn text_block(payload: &Value) -> Value {
    json!({ "type": "text", "text": serde_json::to_string_pretty(payload).unwrap_or_default() })
}

fn no_pipeline() -> Value {
    json!({ "ok": false, "error": "no managed pipeline" })
}

fn mutation_error(error: &impl core::fmt::Debug) -> Value {
    json!({ "ok": false, "error": format!("{error:?}") })
}

/// A property read as the tool reports it: keyed by the property's own name, as
/// `gst-launch` spells it, with `null` for a name the element does not carry.
fn property_result<E: core::fmt::Debug>(
    property: &str,
    read: Result<Option<PropValue>, E>,
) -> Value {
    match read {
        Ok(value) => json!({ "ok": true, property: value.as_ref().map(property_json) }),
        Err(error) => mutation_error(&error),
    }
}

fn property_json(value: &PropValue) -> Value {
    match value {
        PropValue::Bool(value) => json!(value),
        PropValue::Int(value) => json!(value),
        PropValue::Uint(value) => json!(value),
        PropValue::Double(value) => json!(value),
        PropValue::Fraction(numerator, denominator) => json!(format!("{numerator}/{denominator}")),
        PropValue::Str(value) => json!(value),
        PropValue::Flags(nicks) => json!(nicks.join("+")),
        // a kind added since: the agent still sees the value
        other => json!(format!("{other:?}")),
    }
}

/// A JSON argument as a typed property value. With `kind` known (the element's
/// current value) the text is parsed the way a launch line's would be; without
/// it the JSON type decides and the element has the last word.
fn property_value(value: &Value, kind: Option<PropKind>) -> Result<PropValue, (i64, String)> {
    let text = property_text(value)?;
    if let Some(kind) = kind {
        return PropValue::parse(kind, &text).map_err(|error| {
            (
                -32602,
                format!("bad `value` for a {kind:?} property: {error:?}"),
            )
        });
    }
    Ok(match value {
        Value::Bool(flag) => PropValue::Bool(*flag),
        Value::Number(number) => match (number.as_i64(), number.as_u64(), number.as_f64()) {
            (Some(signed), None, _) => PropValue::Int(signed),
            (_, Some(unsigned), _) => PropValue::Uint(unsigned),
            (_, _, Some(double)) => PropValue::Double(double),
            _ => return Err((-32602, "`value` is not a number".into())),
        },
        _ => PropValue::Str(text),
    })
}

/// A path as a launch-line property value. The parser resolves `\` escapes and
/// quoted regions, so a path with spaces survives.
fn launch_quote(path: &str) -> String {
    let mut quoted = String::with_capacity(path.len() + 2);
    quoted.push('"');
    for character in path.chars() {
        if character == '"' || character == '\\' {
            quoted.push('\\');
        }
        quoted.push(character);
    }
    quoted.push('"');
    quoted
}

fn seconds_to_ns(seconds: f64) -> u64 {
    (seconds.max(0.0) * NS_PER_SECOND) as u64
}

/// Where a clip goes when the call named no `location`: beside the temporary
/// directory, under the source's stem and the second it starts at.
fn default_clip_path(source: &str, start: f64) -> String {
    let stem = std::path::Path::new(source)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    std::env::temp_dir()
        .join(format!("{stem}-{start:.2}{DEFAULT_CLIP_EXTENSION}"))
        .to_string_lossy()
        .into_owned()
}

/// Captures the first frame crossing an edge for `snapshot_frame`, under the
/// caps it arrived with.
#[cfg(feature = "mcp")]
struct FrameCapture {
    caps: Mutex<g2g_core::Caps>,
    sender: mpsc::SyncSender<CapturedFrame>,
    remaining: AtomicUsize,
}

#[cfg(feature = "mcp")]
#[derive(Debug)]
struct CapturedFrame {
    caps: g2g_core::Caps,
    pts_ns: Option<u64>,
    memory: String,
    /// `None` for a frame that is not in system memory.
    pixels: Option<Vec<u8>>,
}

#[cfg(feature = "mcp")]
impl LinkInterceptor for FrameCapture {
    fn on_packet(&self, packet: &PipelinePacket) -> ProbeAction {
        let mut caps = self.caps.lock().expect("frame capture caps");
        if let PipelinePacket::CapsChanged(changed) = packet {
            *caps = changed.clone();
        }
        let PipelinePacket::DataFrame(frame) = packet else {
            return ProbeAction::Pass;
        };
        if self
            .remaining
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                remaining.checked_sub(1)
            })
            .is_err()
        {
            return ProbeAction::Pass;
        }
        let _ = self.sender.try_send(CapturedFrame {
            caps: caps.clone(),
            pts_ns: frame.timing.pts(),
            memory: format!("{:?}", frame.domain.kind()).to_ascii_lowercase(),
            pixels: frame.domain.as_system_slice().map(<[u8]>::to_vec),
        });
        ProbeAction::Pass
    }
}

#[cfg(feature = "mcp")]
struct SnapshotImage {
    png: Vec<u8>,
    width: u32,
    height: u32,
}

/// One captured frame as a PNG. Anything that is not already packed RGB or RGBA
/// is converted; the error text names the element to insert instead.
#[cfg(feature = "mcp")]
fn snapshot_png(captured: &CapturedFrame, element: &str) -> Result<SnapshotImage, String> {
    use g2g_core::{Caps, Dim, RawVideoFormat};
    use png::ColorType;

    let Caps::RawVideo {
        format,
        width,
        height,
        colorimetry,
        ..
    } = &captured.caps
    else {
        return Err(format!(
            "`{element}` takes {}, which is not raw video: insert a decoder and a videoconvert ahead of it",
            captured.caps.to_gst_string()
        ));
    };
    let (Dim::Fixed(width), Dim::Fixed(height)) = (width, height) else {
        return Err(format!(
            "the caps reaching `{element}` fix no frame size: {}",
            captured.caps.to_gst_string()
        ));
    };
    let Some(pixels) = captured.pixels.as_deref() else {
        return Err(format!(
            "the frame reaching `{element}` is in {} memory: insert that domain's download element (cudadownload, wgpudownload) ahead of it",
            captured.memory
        ));
    };
    let needed = crate::pixel::frame_byte_size(*format, *width, *height);
    if pixels.len() < needed {
        return Err(format!(
            "the frame holds {} bytes, short of the {needed} its caps describe",
            pixels.len()
        ));
    }
    let (color, converted) = match format {
        RawVideoFormat::Rgb8 => (ColorType::Rgb, None),
        RawVideoFormat::Rgba8 => (ColorType::Rgba, None),
        _ => (
            ColorType::Rgba,
            Some(crate::videoconvert::convert(
                &pixels[..needed],
                *format,
                RawVideoFormat::Rgba8,
                *width as usize,
                *height as usize,
                *colorimetry,
            )),
        ),
    };
    let image = converted.as_deref().unwrap_or(&pixels[..needed]);

    let mut png = Vec::new();
    let mut encoder = png::Encoder::new(&mut png, *width, *height);
    encoder.set_color(color);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|error| format!("cannot write the PNG header: {error}"))?;
    writer
        .write_image_data(image)
        .map_err(|error| format!("cannot write the PNG pixels: {error}"))?;
    writer
        .finish()
        .map_err(|error| format!("cannot finish the PNG: {error}"))?;
    Ok(SnapshotImage {
        png,
        width: *width,
        height: *height,
    })
}

/// The edge whose frames reach the element to snapshot, with that element's
/// name. An explicit `edge` index wins, then a named element, then the sink the
/// results go to.
#[cfg(feature = "mcp")]
fn snapshot_edge(
    snapshot: &TelemetrySnapshot,
    element: &str,
    edge: Option<usize>,
) -> Result<(usize, String), String> {
    if let Some(edge) = edge {
        let info = snapshot
            .edges
            .get(edge)
            .ok_or_else(|| format!("the pipeline has {} edges", snapshot.edges.len()))?;
        let name = snapshot
            .nodes
            .get(info.to)
            .map(|node| node.name.clone())
            .unwrap_or_default();
        return Ok((edge, name));
    }
    let name = if element.is_empty() {
        default_snapshot_element(snapshot)?
    } else {
        element.to_string()
    };
    let node = snapshot
        .nodes
        .iter()
        .find(|node| node.name == name)
        .ok_or_else(|| format!("the pipeline has no element named `{name}`"))?;
    let edge = snapshot
        .edges
        .iter()
        .position(|edge| edge.to == node.id)
        .ok_or_else(|| format!("nothing feeds `{name}`"))?;
    Ok((edge, name))
}

#[cfg(feature = "mcp")]
fn default_snapshot_element(snapshot: &TelemetrySnapshot) -> Result<String, String> {
    let sinks: Vec<&str> = snapshot
        .nodes
        .iter()
        .filter(|node| node.role == NodeRole::Sink)
        .map(|node| node.name.as_str())
        .collect();
    if let Some(name) =
        metasink_prefix().and_then(|prefix| sinks.iter().find(|name| name.starts_with(prefix)))
    {
        return Ok((*name).to_string());
    }
    match sinks.as_slice() {
        [only] => Ok((*only).to_string()),
        [] => Err("the pipeline has no sink: name an `element`".to_string()),
        several => Err(format!(
            "the pipeline has several sinks ({}): name an `element`",
            several.join(", ")
        )),
    }
}

/// Whether the graph holds a `metasink` with no `location`, which would write
/// its records to the stdout this server's protocol rides on.
#[cfg(feature = "analytics-json")]
fn metasink_writes_to_stdout(graph: &g2g_core::Graph<g2g_core::runtime::GraphNode>) -> bool {
    (0..graph.node_count()).any(|node| {
        let Some(g2g_core::runtime::GraphNodeRef::Element(element)) =
            graph.element(g2g_core::NodeId(node as u32))
        else {
            return false;
        };
        element.log_category() == g2g_core::log::short_type_name::<crate::metasink::MetaSink>()
            && element
                .get_property("location")
                .is_some_and(|location| location.as_str() == Some(""))
    })
}

#[cfg(not(feature = "analytics-json"))]
fn metasink_writes_to_stdout(_graph: &g2g_core::Graph<g2g_core::runtime::GraphNode>) -> bool {
    false
}

/// The name an unnamed `metasink` node takes, so a snapshot finds the element
/// the results come from among several sinks.
#[cfg(all(feature = "mcp", feature = "analytics-json"))]
fn metasink_prefix() -> Option<&'static str> {
    Some(g2g_core::log::short_type_name::<crate::metasink::MetaSink>())
}

#[cfg(all(feature = "mcp", not(feature = "analytics-json")))]
fn metasink_prefix() -> Option<&'static str> {
    None
}

/// One `###` section of the README's sample pipelines: the slug it is addressed
/// by, its heading, and the text `prompts/get` returns.
#[derive(Debug)]
struct ReadmePrompt {
    name: String,
    heading: String,
    text: String,
}

impl ReadmePrompt {
    fn description(&self) -> String {
        format!("The README pipelines under {}", self.heading)
    }
}

fn prompt_specs() -> Vec<Value> {
    readme_prompts()
        .iter()
        .map(|prompt| json!({ "name": prompt.name, "description": prompt.description() }))
        .collect()
}

fn prompt_messages(params: Option<&Value>) -> Result<Value, (i64, String)> {
    let name = params
        .and_then(|params| params.get("name"))
        .and_then(Value::as_str)
        .ok_or((-32602, "prompts/get needs `name`".to_string()))?;
    let prompt = readme_prompts()
        .into_iter()
        .find(|prompt| prompt.name == name)
        .ok_or((-32602, format!("unknown prompt: {name}")))?;
    Ok(json!({
        "description": prompt.description(),
        "messages": [
            { "role": "user", "content": { "type": "text", "text": prompt.text } }
        ],
    }))
}

/// A prompt per `###` heading under the README's sample pipelines, carrying that
/// section's code blocks. An unreadable README leaves an agent without prompts,
/// not without a server.
fn readme_prompts() -> Vec<ReadmePrompt> {
    let path = std::env::var(README_PATH_ENV).unwrap_or_else(|_| README_PATH.to_string());
    let Ok(readme) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut prompts: Vec<ReadmePrompt> = Vec::new();
    let mut in_section = false;
    let mut fenced = false;
    for line in readme.lines() {
        if !fenced && line.starts_with("## ") {
            in_section = line.trim_end() == PIPELINE_SECTION_HEADING;
            continue;
        }
        if !in_section {
            continue;
        }
        if !fenced && line.starts_with("### ") {
            let heading = line[4..].trim().to_string();
            prompts.push(ReadmePrompt {
                name: prompt_name(&heading),
                text: prompt_opening(&heading),
                heading,
            });
            continue;
        }
        if line.trim_start().starts_with("```") {
            fenced = !fenced;
        } else if !fenced {
            continue;
        }
        if let Some(prompt) = prompts.last_mut() {
            prompt.text.push('\n');
            prompt.text.push_str(line);
        }
    }
    prompts
}

fn prompt_opening(heading: &str) -> String {
    format!(
        "These are the {heading} pipelines from the g2g README. start_pipeline or launch take each \
         line below, and a display sink can be swapped for metasink to read the results back."
    )
}

/// A heading as a prompt name: ASCII lowercase and `_`, since a heading carries
/// arrows and other punctuation a client cannot address.
fn prompt_name(heading: &str) -> String {
    let mut name = String::with_capacity(heading.len());
    for character in heading.chars() {
        if character.is_ascii_alphanumeric() {
            name.push(character.to_ascii_lowercase());
        } else if !name.ends_with('_') {
            name.push('_');
        }
    }
    name.trim_matches('_').to_string()
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

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use super::*;

    // the runner's finished state can arrive before the bus eos ends the store
    #[test]
    fn wait_treats_a_finished_run_as_ended() {
        let store = RecordStore::new();
        store.push(json!({ "pts": 0.0 }));
        store.push(json!({ "pts": 1.0, "alert": "car" }));
        let run_ended = || true;
        let matched = store.wait(1, "alert", Duration::from_millis(10), &run_ended);
        assert_eq!(matched.len(), 1, "{matched:?}");
        assert_eq!(matched[0]["alert"], "car");
    }
}

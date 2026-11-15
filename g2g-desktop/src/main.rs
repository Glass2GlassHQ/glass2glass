// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use g2g_core::runtime::Observer;
use g2g_core::runtime::{GraphNodeRef, VisitRecorder};
#[cfg(target_os = "linux")]
use g2g_core::G2gError;
#[cfg(target_os = "linux")]
use g2g_desktop::client::{run_client, ClientError, ClientSettings};
use g2g_desktop::host::{inject, run_session, HostCapture, HostSettings, TurnServer};
#[cfg(target_os = "windows")]
use g2g_desktop::sendinput::Injector;
use g2g_desktop::signaling::SessionSlot;
#[cfg(target_os = "linux")]
use g2g_desktop::uinput::Injector;
#[cfg(target_os = "windows")]
use g2g_plugins::d3d11screencapturesrc::{monitor_bounds, D3D11ScreenCaptureSrc};
use g2g_plugins::perfetto_trace::PerfettoTrace;
#[cfg(target_os = "linux")]
use g2g_plugins::pipewirevideosrc::PipeWireVideoSrc;
#[cfg(target_os = "linux")]
use g2g_plugins::screencastportal::PortalSourceTypes;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, watch};

const HOST_USAGE: &str = "  g2g-desktop host --token TOKEN [--port 8080] [--fps 60] [--bitrate 4000000]
                   [--stun-server HOST:PORT] [--turn-server HOST:PORT --turn-user USER --turn-pass PASS]
                   [--trace FILE]";
#[cfg(target_os = "linux")]
const PLATFORM_USAGE: &str =
    "  g2g-desktop client --host HOST --token TOKEN [--port 8080] [--inputs true] [--trace FILE]";
#[cfg(target_os = "windows")]
const PLATFORM_USAGE: &str = "                   [--monitor-index -1]";
const DEFAULT_PORT: u16 = 8080;
const DEFAULT_FPS: u32 = 60;
const DEFAULT_BITRATE: u32 = 4_000_000;
const HOST_FLAGS: [&str; 9] = [
    "port",
    "token",
    "fps",
    "bitrate",
    "stun-server",
    "turn-server",
    "turn-user",
    "turn-pass",
    "trace",
];
#[cfg(target_os = "linux")]
const PLATFORM_HOST_FLAGS: [&str; 0] = [];
#[cfg(target_os = "windows")]
const PLATFORM_HOST_FLAGS: [&str; 1] = ["monitor-index"];
#[cfg(target_os = "windows")]
const PRIMARY_MONITOR: i32 = -1;
#[cfg(target_os = "linux")]
const CLIENT_FLAGS: [&str; 5] = ["host", "port", "token", "inputs", "trace"];

type Flags = BTreeMap<String, String>;

fn parse_flags(
    args: impl Iterator<Item = String>,
    known: &[&str],
) -> Result<Flags, Box<dyn Error>> {
    let mut flags = Flags::new();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let name = arg
            .strip_prefix("--")
            .ok_or_else(|| format!("expected a --flag, got {arg}"))?;
        if !known.contains(&name) {
            return Err(format!("unknown flag --{name}").into());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("--{name} needs a value"))?;
        flags.insert(name.to_string(), value);
    }
    Ok(flags)
}

fn required(flags: &Flags, name: &str) -> Result<String, Box<dyn Error>> {
    flags
        .get(name)
        .cloned()
        .ok_or_else(|| format!("--{name} is required").into())
}

fn parsed<T: FromStr>(flags: &Flags, name: &str, default: T) -> Result<T, Box<dyn Error>>
where
    T::Err: Error + 'static,
{
    match flags.get(name) {
        Some(value) => value
            .parse()
            .map_err(|e| format!("--{name} {value}: {e}").into()),
        None => Ok(default),
    }
}

fn turn_server(flags: &Flags) -> Result<Option<TurnServer>, Box<dyn Error>> {
    let Some(server) = flags.get("turn-server") else {
        return Ok(None);
    };
    Ok(Some(TurnServer {
        server: server.clone(),
        user: required(flags, "turn-user")?,
        pass: required(flags, "turn-pass")?,
    }))
}

#[derive(Debug, Clone, Copy)]
struct CaptureSettings {
    fps: u32,
    #[cfg(target_os = "windows")]
    monitor_index: i32,
}

fn open_trace(path: Option<&Path>) -> Result<Option<Arc<PerfettoTrace>>, Box<dyn Error>> {
    path.map(|path| {
        PerfettoTrace::create(path).map_err(|e| format!("--trace {}: {e}", path.display()).into())
    })
    .transpose()
}

async fn host_one(
    stream: TcpStream,
    settings: HostSettings,
    capture: CaptureSettings,
    trace_path: Option<PathBuf>,
) {
    let trace = match open_trace(trace_path.as_deref()) {
        Ok(trace) => trace,
        Err(e) => {
            eprintln!("g2g-desktop host: {e}");
            return;
        }
    };
    let (events, event_receiver) = mpsc::unbounded_channel();
    let (sizes, size_receiver) = watch::channel(None);
    #[cfg(target_os = "linux")]
    let (host_capture, open_injector) = (
        HostCapture {
            nodes: vec![GraphNodeRef::source(
                PipeWireVideoSrc::new()
                    .with_portal(PortalSourceTypes::Monitor)
                    .with_fps(capture.fps),
            )],
            cursor_shapes: None,
        },
        Injector::new,
    );
    #[cfg(target_os = "windows")]
    let (host_capture, open_injector) = {
        let monitor = match monitor_bounds(capture.monitor_index) {
            Ok(monitor) => monitor,
            Err(e) => {
                eprintln!("g2g-desktop host: monitor {}: {e:?}", capture.monitor_index);
                return;
            }
        };
        let source = D3D11ScreenCaptureSrc::new()
            .with_monitor_index(capture.monitor_index)
            .with_fps(capture.fps);
        (
            HostCapture {
                cursor_shapes: Some(source.cursor_shapes()),
                nodes: vec![GraphNodeRef::source(source)],
            },
            move |screen| Injector::new(screen, monitor),
        )
    };
    let (session, injected) = tokio::join!(
        run_session(
            stream,
            host_capture,
            &settings,
            events,
            sizes,
            trace.clone().map(|trace| trace as Arc<dyn VisitRecorder>),
        ),
        inject(event_receiver, size_receiver, open_injector)
    );
    match session {
        Ok(stats) => eprintln!("{}", stats.report()),
        Err(e) => eprintln!("g2g-desktop host: {e}"),
    }
    if let Err(e) = injected {
        eprintln!("g2g-desktop host: input injection: {e}");
    }
    if let Some(Err(e)) = trace.map(PerfettoTrace::finish) {
        eprintln!("g2g-desktop host: --trace: {e}");
    }
}

async fn host(flags: Flags) -> Result<(), Box<dyn Error>> {
    let port = parsed(&flags, "port", DEFAULT_PORT)?;
    let capture = CaptureSettings {
        fps: parsed(&flags, "fps", DEFAULT_FPS)?,
        #[cfg(target_os = "windows")]
        monitor_index: parsed(&flags, "monitor-index", PRIMARY_MONITOR)?,
    };
    let settings = HostSettings {
        token: required(&flags, "token")?,
        bitrate: parsed(&flags, "bitrate", DEFAULT_BITRATE)?,
        stun_server: flags.get("stun-server").cloned(),
        turn_server: turn_server(&flags)?,
    };
    let trace_path = flags.get("trace").map(PathBuf::from);
    let listener = TcpListener::bind(("0.0.0.0", port)).await?;
    let slot = SessionSlot::default();
    loop {
        let (stream, peer) = listener.accept().await?;
        let Ok(guard) = slot.claim() else {
            eprintln!("g2g-desktop host: {peer} refused, a session is already running");
            continue;
        };
        let settings = settings.clone();
        let trace_path = trace_path.clone();
        tokio::task::spawn_local(async move {
            host_one(stream, settings, capture, trace_path).await;
            drop(guard);
        });
    }
}

#[cfg(target_os = "linux")]
async fn client(flags: Flags) -> Result<(), Box<dyn Error>> {
    let host = required(&flags, "host")?;
    let port = parsed(&flags, "port", DEFAULT_PORT)?;
    let settings = ClientSettings {
        token: required(&flags, "token")?,
        inputs: parsed(&flags, "inputs", true)?,
    };
    let trace = open_trace(flags.get("trace").map(Path::new))?;
    let stream = TcpStream::connect((host.as_str(), port)).await?;
    let observer = Observer::new();
    let session = tokio::select! {
        session = run_client(
            stream,
            &settings,
            trace.clone().map(|trace| trace as Arc<dyn VisitRecorder>),
            &observer,
        ) => Some(session),
        interrupted = tokio::signal::ctrl_c() => {
            interrupted?;
            None
        }
    };
    match &session {
        Some(Ok(stats)) => eprintln!("{}", stats.report()),
        None => {
            eprintln!("g2g-desktop client: interrupted, per-element [measured]:");
            for node in observer.snapshot().nodes {
                if let Some(latency) = node.latency {
                    eprintln!("{}", latency.report_line());
                }
            }
        }
        Some(Err(_)) => {}
    }
    // the observer's element probes hold the trace writer until they drop
    drop(observer);
    if let Some(trace) = trace {
        trace.finish()?;
    }
    match session {
        None | Some(Ok(_)) | Some(Err(ClientError::Session(G2gError::WindowClosed))) => Ok(()),
        Some(Err(e)) => Err(e.into()),
    }
}

fn main() -> ExitCode {
    g2g_core::log::init_from_env();
    let mut args = std::env::args().skip(1);
    let command = args.next();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let local = tokio::task::LocalSet::new();
    let result = match command.as_deref() {
        Some("host") => parse_flags(args, &[&HOST_FLAGS[..], &PLATFORM_HOST_FLAGS].concat())
            .and_then(|flags| runtime.block_on(local.run_until(host(flags)))),
        #[cfg(target_os = "linux")]
        Some("client") => parse_flags(args, &CLIENT_FLAGS)
            .and_then(|flags| runtime.block_on(local.run_until(client(flags)))),
        _ => Err(format!("usage:\n{HOST_USAGE}\n{PLATFORM_USAGE}").into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("g2g-desktop: {e}");
            ExitCode::FAILURE
        }
    }
}

// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::BTreeMap;
#[cfg(target_os = "linux")]
use std::convert::Infallible;
use std::error::Error;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;

use g2g_core::runtime::Observer;
use g2g_core::runtime::{GraphNodeRef, VisitRecorder};
use g2g_core::G2gError;
use g2g_desktop::client::{run_client, ClientError, ClientSettings, ClientTransport};
use g2g_desktop::host::{
    check_token_len, inject, run_session, HostCapture, HostEncoder, HostListener, HostSettings,
};
use g2g_desktop::injection::{InjectError, InjectEvents, ScreenSize};
#[cfg(target_os = "windows")]
use g2g_desktop::sendinput::Injector;
#[cfg(target_os = "linux")]
use g2g_desktop::uinput::Injector;
#[cfg(target_os = "linux")]
use g2g_desktop::waylanddisplayinput::WaylandDisplayInjector;
#[cfg(target_os = "windows")]
use g2g_plugins::d3d11screencapturesrc::{monitor_bounds, D3D11ScreenCaptureSrc};
use g2g_plugins::perfetto_trace::PerfettoTrace;
#[cfg(target_os = "linux")]
use g2g_plugins::pipewirevideosrc::PipeWireVideoSrc;
use g2g_plugins::remotewtio::{generate_certificate, SelfSignedCertificate};
#[cfg(target_os = "linux")]
use g2g_plugins::screencastportal::PortalSourceTypes;
#[cfg(target_os = "linux")]
use g2g_wayland_display::waylanddisplaysrc::{
    render_node_driver, DisplayNames, WaylandDisplay, WaylandDisplaySettings, WaylandDisplaySrc,
    SOFTWARE_RENDERING,
};
use tokio::sync::{mpsc, watch};

const HOST_USAGE: &str =
    "  g2g-desktop host --token TOKEN [--port 8080] [--fps 60] [--bitrate 4000000]
                   [--send-window BYTES] [--video-mode stream|stream-per-frame|datagrams]
                   [--trace FILE]";
const CLIENT_USAGE: &str =
    "  g2g-desktop client --host HOST --cert-hash SHA256 --token TOKEN [--port 8080] [--inputs true]
                     [--transport auto|quic|websocket] [--trace FILE]";
#[cfg(target_os = "linux")]
const PLATFORM_HOST_USAGE: &str =
    "\n                   [--run COMMAND [--size 1920x1080] [--render-node /dev/dri/renderD128]]";
#[cfg(target_os = "windows")]
const PLATFORM_HOST_USAGE: &str =
    "\n                   [--monitor-index -1] [--quality-vs-speed 0-100]";
const DEFAULT_PORT: u16 = 8080;
const DEFAULT_FPS: u32 = 60;
const DEFAULT_BITRATE: u32 = 4_000_000;
const HOST_FLAGS: [&str; 7] = [
    "port",
    "token",
    "fps",
    "bitrate",
    "send-window",
    "video-mode",
    "trace",
];
#[cfg(target_os = "linux")]
const PLATFORM_HOST_FLAGS: [&str; 3] = ["run", "size", "render-node"];
#[cfg(target_os = "windows")]
const PLATFORM_HOST_FLAGS: [&str; 2] = ["monitor-index", "quality-vs-speed"];
#[cfg(target_os = "windows")]
const PRIMARY_MONITOR: i32 = -1;
const CLIENT_FLAGS: [&str; 7] = [
    "host",
    "port",
    "cert-hash",
    "token",
    "inputs",
    "transport",
    "trace",
];

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

fn optional<T: FromStr>(flags: &Flags, name: &str) -> Result<Option<T>, Box<dyn Error>>
where
    T::Err: Error + 'static,
{
    flags
        .get(name)
        .map(|value| {
            value
                .parse()
                .map_err(|e| format!("--{name} {value}: {e}").into())
        })
        .transpose()
}

fn parsed<T: FromStr>(flags: &Flags, name: &str, default: T) -> Result<T, Box<dyn Error>>
where
    T::Err: Error + 'static,
{
    Ok(optional(flags, name)?.unwrap_or(default))
}

#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy)]
struct DisplaySize {
    width: u32,
    height: u32,
}

#[cfg(target_os = "linux")]
fn parse_display_size(value: &str) -> Result<DisplaySize, Box<dyn Error>> {
    let invalid = || format!("--size {value}: expected WIDTHxHEIGHT");
    let (width, height) = value.split_once('x').ok_or_else(invalid)?;
    Ok(DisplaySize {
        width: width.parse().map_err(|_| invalid())?,
        height: height.parse().map_err(|_| invalid())?,
    })
}

#[cfg(target_os = "linux")]
#[derive(Debug, Clone)]
struct HeadlessSettings {
    command: String,
    size: Option<DisplaySize>,
    render_node: Option<String>,
}

#[cfg(target_os = "linux")]
fn headless_settings(flags: &Flags) -> Result<Option<HeadlessSettings>, Box<dyn Error>> {
    let Some(command) = flags.get("run") else {
        if let Some(flag) = ["size", "render-node"]
            .iter()
            .find(|flag| flags.contains_key(**flag))
        {
            return Err(format!("--{flag} needs --run").into());
        }
        return Ok(None);
    };
    Ok(Some(HeadlessSettings {
        command: command.clone(),
        size: flags
            .get("size")
            .map(|size| parse_display_size(size))
            .transpose()?,
        render_node: flags.get("render-node").cloned(),
    }))
}

#[derive(Debug, Clone)]
struct CaptureSettings {
    fps: u32,
    #[cfg(target_os = "linux")]
    portal_restore_token_path: PathBuf,
    #[cfg(target_os = "linux")]
    headless: Option<HeadlessSettings>,
    #[cfg(target_os = "windows")]
    monitor_index: i32,
}

#[cfg(target_os = "linux")]
fn portal_restore_token_path() -> Result<PathBuf, Box<dyn Error>> {
    // XDG base directory spec: an unset or empty XDG_STATE_HOME means $HOME/.local/state
    let state_home = match std::env::var_os("XDG_STATE_HOME").filter(|dir| !dir.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => {
            PathBuf::from(std::env::var_os("HOME").ok_or("HOME is not set")?).join(".local/state")
        }
    };
    Ok(state_home.join("g2g-desktop/portal-restore-token"))
}

#[cfg(target_os = "linux")]
async fn save_portal_restore_tokens(
    mut tokens: watch::Receiver<String>,
    path: &Path,
) -> std::io::Result<()> {
    while tokens.changed().await.is_ok() {
        let token = tokens.borrow_and_update().clone();
        std::fs::write(path, token)?;
    }
    Ok(())
}

fn open_trace(path: Option<&Path>) -> Result<Option<Arc<PerfettoTrace>>, Box<dyn Error>> {
    path.map(|path| {
        PerfettoTrace::create(path).map_err(|e| format!("--trace {}: {e}", path.display()).into())
    })
    .transpose()
}

async fn serve<I: InjectEvents>(
    listener: HostListener,
    settings: &HostSettings,
    capture: HostCapture,
    open_injector: impl Fn(ScreenSize) -> Result<I, InjectError>,
    trace: Option<Arc<PerfettoTrace>>,
) {
    let (events, event_receiver) = mpsc::unbounded_channel();
    let (sizes, size_receiver) = watch::channel(None);
    let (session, injected) = tokio::join!(
        run_session(
            listener,
            capture,
            settings,
            events,
            sizes,
            trace.clone().map(|trace| trace as Arc<dyn VisitRecorder>),
        ),
        inject(event_receiver, size_receiver, open_injector),
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

#[cfg(target_os = "linux")]
async fn host_portal(
    listener: HostListener,
    settings: &HostSettings,
    capture: &CaptureSettings,
    trace: Option<Arc<PerfettoTrace>>,
) {
    let path = &capture.portal_restore_token_path;
    let token = match std::fs::read_to_string(path) {
        Ok(token) => token,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => {
            eprintln!("g2g-desktop host: {}: {e}", path.display());
            return;
        }
    };
    let source = PipeWireVideoSrc::new()
        .with_portal(PortalSourceTypes::Monitor)
        .with_portal_restore_token(token)
        .with_fps(capture.fps);
    let saved_tokens = save_portal_restore_tokens(source.portal_restore_tokens(), path);
    let host_capture = HostCapture {
        nodes: vec![GraphNodeRef::source(source)],
        encoder: HostEncoder::Software,
        cursor_shapes: None,
    };
    let ((), saved) = tokio::join!(
        serve(listener, settings, host_capture, Injector::new, trace),
        saved_tokens
    );
    if let Err(e) = saved {
        eprintln!("g2g-desktop host: saving the portal restore token: {e}");
    }
}

#[cfg(target_os = "linux")]
fn headless_encoder(render_node: &str) -> std::io::Result<HostEncoder> {
    if render_node == SOFTWARE_RENDERING {
        return Ok(HostEncoder::Software);
    }
    let encoder = match render_node_driver(Path::new(render_node))?.as_str() {
        "nvidia" => HostEncoder::Nvenc,
        "amdgpu" | "i915" | "xe" => HostEncoder::Vaapi {
            render_node: render_node.to_owned(),
        },
        _ => HostEncoder::Software,
    };
    Ok(encoder)
}

#[cfg(target_os = "linux")]
fn launch_app(names: &DisplayNames, command: &str) -> std::io::Result<tokio::process::Child> {
    let x11_display = names
        .x11_display
        .as_ref()
        .expect("the headless display runs Xwayland");
    tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .env("WAYLAND_DISPLAY", &names.wayland_socket)
        .env("DISPLAY", x11_display)
        .kill_on_drop(true)
        .spawn()
}

#[cfg(target_os = "linux")]
async fn serve_headless_sessions(
    address: SocketAddr,
    certificate: &SelfSignedCertificate,
    settings: &HostSettings,
    display: &WaylandDisplay,
    encoder: &HostEncoder,
    trace_path: Option<&Path>,
) -> Result<Infallible, Box<dyn Error>> {
    loop {
        let listener = HostListener::bind(address, certificate.clone(), settings)?;
        let trace = open_trace(trace_path)?;
        let input = display.input_sender();
        let host_capture = HostCapture {
            nodes: vec![GraphNodeRef::source(WaylandDisplaySrc::on_display(display))],
            encoder: encoder.clone(),
            cursor_shapes: Some(display.cursor_shapes()),
        };
        let open_injector = move |screen| Ok(WaylandDisplayInjector::new(screen, input.clone()));
        serve(listener, settings, host_capture, open_injector, trace).await;
    }
}

#[cfg(target_os = "linux")]
async fn host_headless(
    address: SocketAddr,
    certificate: &SelfSignedCertificate,
    settings: &HostSettings,
    fps: u32,
    headless: &HeadlessSettings,
    trace_path: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    let mut display_settings = WaylandDisplaySettings {
        framerate_q16: fps << 16,
        xwayland: true,
        ..WaylandDisplaySettings::default()
    };
    if let Some(size) = headless.size {
        display_settings.width = size.width;
        display_settings.height = size.height;
    }
    if let Some(render_node) = &headless.render_node {
        display_settings.render_node = render_node.clone();
    }
    let encoder = headless_encoder(&display_settings.render_node)
        .map_err(|e| format!("{}: {e}", display_settings.render_node))?;
    let display = WaylandDisplay::start(display_settings)
        .await
        .map_err(|e| format!("starting the headless display: {e:?}"))?;
    let mut app =
        launch_app(display.names(), &headless.command).map_err(|e| format!("--run: {e}"))?;
    let sessions = serve_headless_sessions(
        address,
        certificate,
        settings,
        &display,
        &encoder,
        trace_path,
    );
    let status = tokio::select! {
        status = app.wait() => status.map_err(|e| format!("--run: {e}"))?,
        served = sessions => match served? {},
    };
    if !status.success() {
        return Err(format!("--run exited with {status}").into());
    }
    Ok(())
}

#[cfg(target_os = "windows")]
async fn host_monitor(
    listener: HostListener,
    settings: &HostSettings,
    capture: &CaptureSettings,
    trace: Option<Arc<PerfettoTrace>>,
) {
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
    let host_capture = HostCapture {
        cursor_shapes: Some(source.cursor_shapes()),
        nodes: vec![GraphNodeRef::source(source)],
        encoder: HostEncoder::MediaFoundation,
    };
    let open_injector = move |screen| Injector::new(screen, monitor);
    serve(listener, settings, host_capture, open_injector, trace).await;
}

async fn host_one(
    listener: HostListener,
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
    #[cfg(target_os = "linux")]
    host_portal(listener, &settings, &capture, trace).await;
    #[cfg(target_os = "windows")]
    host_monitor(listener, &settings, &capture, trace).await;
}

async fn host(flags: Flags) -> Result<(), Box<dyn Error>> {
    let port = parsed(&flags, "port", DEFAULT_PORT)?;
    let capture = CaptureSettings {
        fps: parsed(&flags, "fps", DEFAULT_FPS)?,
        #[cfg(target_os = "linux")]
        portal_restore_token_path: portal_restore_token_path()?,
        #[cfg(target_os = "linux")]
        headless: headless_settings(&flags)?,
        #[cfg(target_os = "windows")]
        monitor_index: parsed(&flags, "monitor-index", PRIMARY_MONITOR)?,
    };
    let settings = HostSettings {
        token: required(&flags, "token")?,
        bitrate: parsed(&flags, "bitrate", DEFAULT_BITRATE)?,
        send_window: optional(&flags, "send-window")?,
        video_mode: optional(&flags, "video-mode")?,
        #[cfg(target_os = "windows")]
        quality_vs_speed: optional(&flags, "quality-vs-speed")?,
    };
    check_token_len(&settings.token)?;
    #[cfg(target_os = "linux")]
    std::fs::create_dir_all(
        capture
            .portal_restore_token_path
            .parent()
            .expect("the token path is joined onto a state directory"),
    )?;
    let trace_path = flags.get("trace").map(PathBuf::from);
    let certificate = generate_certificate()?;
    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
    eprintln!(
        "g2g-desktop host: listening on UDP and TCP port {port}, certificate sha256 {}",
        certificate.sha256_hex
    );
    #[cfg(target_os = "linux")]
    if let Some(headless) = &capture.headless {
        return host_headless(
            address,
            &certificate,
            &settings,
            capture.fps,
            headless,
            trace_path.as_deref(),
        )
        .await;
    }
    loop {
        let listener = HostListener::bind(address, certificate.clone(), &settings)?;
        host_one(
            listener,
            settings.clone(),
            capture.clone(),
            trace_path.clone(),
        )
        .await;
    }
}

async fn client(flags: Flags) -> Result<(), Box<dyn Error>> {
    let host = required(&flags, "host")?;
    let port = parsed(&flags, "port", DEFAULT_PORT)?;
    let settings = ClientSettings {
        token: required(&flags, "token")?,
        cert_hash: required(&flags, "cert-hash")?,
        inputs: parsed(&flags, "inputs", true)?,
        transport: match flags.get("transport").map(String::as_str) {
            None | Some("auto") => ClientTransport::Auto,
            Some("quic") => ClientTransport::Quic,
            Some("websocket") => ClientTransport::WebSocket,
            Some(other) => return Err(format!("unknown --transport {other}").into()),
        },
    };
    let trace = open_trace(flags.get("trace").map(Path::new))?;
    let observer = Observer::new();
    let session = tokio::select! {
        session = run_client(
            &host,
            port,
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
        Some("client") => parse_flags(args, &CLIENT_FLAGS)
            .and_then(|flags| runtime.block_on(local.run_until(client(flags)))),
        _ => Err(format!("usage:\n{HOST_USAGE}{PLATFORM_HOST_USAGE}\n{CLIENT_USAGE}").into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("g2g-desktop: {e}");
            ExitCode::FAILURE
        }
    }
}

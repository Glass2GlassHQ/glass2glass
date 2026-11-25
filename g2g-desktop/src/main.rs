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

#[cfg(target_os = "linux")]
use g2g_core::element::DynAsyncElement;
use g2g_core::metrics::LatencyHistogram;
#[cfg(all(target_os = "linux", feature = "gamestream"))]
use g2g_core::runtime::parse_chain;
#[cfg(feature = "gamestream")]
use g2g_core::runtime::GraphNode;
use g2g_core::runtime::Observer;
#[cfg(target_os = "linux")]
use g2g_core::runtime::{parse_element, ParseError, SourceLoop};
use g2g_core::runtime::{GraphNodeRef, VisitRecorder};
use g2g_core::G2gError;
#[cfg(target_os = "linux")]
use g2g_core::PropValue;
use g2g_desktop::client::{run_client, ClientError, ClientSettings, ClientTransport};
#[cfg(feature = "gamestream")]
use g2g_desktop::gamestream::ServedDesktop;
#[cfg(all(target_os = "linux", feature = "gamestream"))]
use g2g_desktop::gamestream::{LaunchableApp, ServedApps};
use g2g_desktop::host::{
    check_token_len, inject, run_session, HostCapture, HostEncoder, HostListener, HostSettings,
};
use g2g_desktop::injection::{InjectError, InjectEvents, ScreenSize, WithGamepads};
#[cfg(target_os = "windows")]
use g2g_desktop::sendinput::Injector;
#[cfg(target_os = "linux")]
use g2g_desktop::uinput::Injector;
#[cfg(target_os = "linux")]
use g2g_desktop::uinputgamepad::UinputGamepads;
#[cfg(target_os = "windows")]
use g2g_desktop::vigemgamepad::ViGEmGamepads;
#[cfg(target_os = "linux")]
use g2g_desktop::waylanddisplayinput::WaylandDisplayInjector;
#[cfg(target_os = "linux")]
use g2g_desktop::xtest::XTestInjector;
#[cfg(target_os = "windows")]
use g2g_plugins::d3d11screencapturesrc::{monitor_bounds, D3D11ScreenCaptureSrc};
use g2g_plugins::perfetto_trace::PerfettoTrace;
#[cfg(target_os = "linux")]
use g2g_plugins::pipewirevideosrc::PipeWireVideoSrc;
#[cfg(target_os = "linux")]
use g2g_plugins::registry::default_registry;
use g2g_plugins::remotewtio::generate_certificate;
#[cfg(target_os = "linux")]
use g2g_plugins::remotewtio::SelfSignedCertificate;
#[cfg(target_os = "linux")]
use g2g_plugins::screencastportal::{PortalCursorMode, PortalSourceTypes};
#[cfg(target_os = "linux")]
use g2g_plugins::ximagesrc::XImageSrc;
#[cfg(target_os = "linux")]
use g2g_wayland_display::waylanddisplaysrc::{
    render_node_driver, DisplayNames, WaylandDisplay, WaylandDisplaySettings, WaylandDisplaySrc,
    SOFTWARE_RENDERING,
};
use tokio::sync::{mpsc, watch};

const HOST_USAGE: &str =
    "  g2g-desktop host --token TOKEN [--port 8080] [--fps 60] [--bitrate 4000000]
                   [--send-window BYTES] [--video-mode stream|stream-per-frame|datagrams]
                   [--trace FILE] [--audio true]";
#[cfg(target_os = "linux")]
const GAMESTREAM_USAGE: &str = "  g2g-desktop host --protocol gamestream
                   [--x11-display :0 | --run COMMAND [--size 1920x1080] [--render-node PATH]
                    [--app TITLE=COMMAND]...]
  g2g-desktop pin PIN";
#[cfg(target_os = "windows")]
const GAMESTREAM_USAGE: &str = "  g2g-desktop host --protocol gamestream
  g2g-desktop pin PIN";
const CLIENT_USAGE: &str =
    "  g2g-desktop client --host HOST --cert-hash SHA256 --token TOKEN [--port 8080] [--inputs true]
                     [--transport auto|quic|websocket] [--trace FILE] [--audio true]
                     [--keyboard-grab true]";
#[cfg(target_os = "linux")]
const PLATFORM_HOST_USAGE: &str =
    "\n                   [--run COMMAND [--size 1920x1080] [--render-node /dev/dri/renderD128]
                    | --x11-display :0]";
#[cfg(target_os = "windows")]
const PLATFORM_HOST_USAGE: &str =
    "\n                   [--monitor-index -1] [--quality-vs-speed 0-100]";
const DEFAULT_PORT: u16 = 8080;
const CLOSE_GRACE: std::time::Duration = std::time::Duration::from_millis(100);
const DEFAULT_FPS: u32 = 60;
const DEFAULT_BITRATE: u32 = 4_000_000;
const HOST_FLAGS: [&str; 9] = [
    "protocol",
    "audio",
    "port",
    "token",
    "fps",
    "bitrate",
    "send-window",
    "video-mode",
    "trace",
];
#[cfg(target_os = "linux")]
const PLATFORM_HOST_FLAGS: [&str; 5] = ["run", "size", "render-node", "x11-display", "app"];
#[cfg(target_os = "windows")]
const PLATFORM_HOST_FLAGS: [&str; 2] = ["monitor-index", "quality-vs-speed"];
#[cfg(target_os = "windows")]
const PRIMARY_MONITOR: i32 = -1;
const CLIENT_FLAGS: [&str; 9] = [
    "audio",
    "host",
    "port",
    "cert-hash",
    "token",
    "inputs",
    "transport",
    "trace",
    "keyboard-grab",
];

type Flags = BTreeMap<String, Vec<String>>;

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
        flags.entry(name.to_string()).or_default().push(value);
    }
    Ok(flags)
}

// a flag given more than once takes its last value, unless its reader takes them all
fn flag<'a>(flags: &'a Flags, name: &str) -> Option<&'a String> {
    flags.get(name)?.last()
}

fn required(flags: &Flags, name: &str) -> Result<String, Box<dyn Error>> {
    flag(flags, name)
        .cloned()
        .ok_or_else(|| format!("--{name} is required").into())
}

fn optional<T: FromStr>(flags: &Flags, name: &str) -> Result<Option<T>, Box<dyn Error>>
where
    T::Err: Error + 'static,
{
    flag(flags, name)
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
    if flags.contains_key("run") && flags.contains_key("x11-display") {
        return Err("--run and --x11-display each pick what the host captures".into());
    }
    let Some(command) = flag(flags, "run") else {
        if let Some(flag) = ["size", "render-node", "app"]
            .iter()
            .find(|flag| flags.contains_key(**flag))
        {
            return Err(format!("--{flag} needs --run").into());
        }
        return Ok(None);
    };
    Ok(Some(HeadlessSettings {
        command: command.clone(),
        size: flag(flags, "size")
            .map(|size| parse_display_size(size))
            .transpose()?,
        render_node: flag(flags, "render-node").cloned(),
    }))
}

#[derive(Debug, Clone)]
struct CaptureSettings {
    fps: u32,
    #[cfg(target_os = "linux")]
    portal_restore_token_path: PathBuf,
    #[cfg(target_os = "linux")]
    headless: Option<HeadlessSettings>,
    #[cfg(target_os = "linux")]
    x11_display: Option<String>,
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
    let (rumbled, rumbles) = mpsc::unbounded_channel();
    let (session, injected) = tokio::join!(
        run_session(
            listener,
            capture,
            settings,
            events,
            sizes,
            rumbles,
            trace.clone().map(|trace| trace as Arc<dyn VisitRecorder>),
        ),
        inject(event_receiver, size_receiver, open_injector, &rumbled),
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
fn read_portal_restore_token(path: &Path) -> std::io::Result<String> {
    match std::fs::read_to_string(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        read => read,
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
    let token = match read_portal_restore_token(path) {
        Ok(token) => token,
        Err(e) => {
            eprintln!("g2g-desktop host: {}: {e}", path.display());
            return;
        }
    };
    let source = PipeWireVideoSrc::new()
        .with_portal(PortalSourceTypes::Monitor)
        .with_portal_cursor_mode(PortalCursorMode::Metadata)
        .with_portal_restore_token(token)
        .with_fps(capture.fps);
    let saved_tokens = save_portal_restore_tokens(source.portal_restore_tokens(), path);
    let host_capture = HostCapture {
        cursor_shapes: Some(source.cursor_shapes()),
        nodes: vec![GraphNodeRef::source(source)],
        encoder: HostEncoder::Software,
        clipboard_texts: None,
    };
    let ((), saved) = tokio::join!(
        serve(
            listener,
            settings,
            host_capture,
            |screen| {
                Ok(WithGamepads {
                    desktop: Injector::new(screen)?,
                    gamepads: UinputGamepads::default(),
                })
            },
            trace
        ),
        saved_tokens
    );
    if let Err(e) = saved {
        eprintln!("g2g-desktop host: saving the portal restore token: {e}");
    }
}

// ximagesrc takes its rate from a downstream caps filter, as gst's does
#[cfg(target_os = "linux")]
fn x11_rate_filter(fps: u32) -> Result<Box<dyn DynAsyncElement>, ParseError> {
    parse_element(
        &default_registry(),
        &format!("capsfilter caps=video/x-raw,framerate={fps}/1"),
    )
}

#[cfg(target_os = "linux")]
async fn host_x11(
    listener: HostListener,
    settings: &HostSettings,
    display: &str,
    fps: u32,
    trace: Option<Arc<PerfettoTrace>>,
) {
    let mut source = XImageSrc::new().with_display_name(display);
    let cursor_shapes = source.cursor_shapes();
    SourceLoop::set_property(&mut source, "show-pointer", PropValue::Bool(false))
        .expect("show-pointer is an ximagesrc property");
    let rate = x11_rate_filter(fps).expect("host checked the rate filter before listening");
    let host_capture = HostCapture {
        nodes: vec![GraphNodeRef::source(source), GraphNodeRef::Element(rate)],
        encoder: HostEncoder::Software,
        cursor_shapes: Some(cursor_shapes),
        clipboard_texts: None,
    };
    let open_injector = |screen| {
        Ok(WithGamepads {
            desktop: XTestInjector::new(display, screen)?,
            gamepads: UinputGamepads::default(),
        })
    };
    serve(listener, settings, host_capture, open_injector, trace).await;
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
fn headless_display_settings(fps: u32, headless: &HeadlessSettings) -> WaylandDisplaySettings {
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
    display_settings
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
async fn run_app_while_serving(
    display: &WaylandDisplay,
    command: &str,
    served: impl std::future::Future<Output = Result<(), Box<dyn Error>>>,
) -> Result<(), Box<dyn Error>> {
    let mut app = launch_app(display.names(), command).map_err(|e| format!("--run: {e}"))?;
    let status = tokio::select! {
        status = app.wait() => status.map_err(|e| format!("--run: {e}"))?,
        served = served => return served,
    };
    if !status.success() {
        return Err(format!("--run exited with {status}").into());
    }
    Ok(())
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
        let listener = HostListener::bind_when_free(address, certificate.clone(), settings).await?;
        let trace = open_trace(trace_path)?;
        let host_capture = HostCapture {
            nodes: vec![GraphNodeRef::source(WaylandDisplaySrc::on_display(display))],
            encoder: encoder.clone(),
            cursor_shapes: Some(display.cursor_shapes()),
            clipboard_texts: Some(display.clipboard_texts()),
        };
        let open_injector = |screen| {
            Ok(WithGamepads {
                desktop: WaylandDisplayInjector::new(screen, display),
                gamepads: UinputGamepads::default(),
            })
        };
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
    let display_settings = headless_display_settings(fps, headless);
    let encoder = headless_encoder(&display_settings.render_node)
        .map_err(|e| format!("{}: {e}", display_settings.render_node))?;
    let display = WaylandDisplay::start(display_settings)
        .await
        .map_err(|e| format!("starting the headless display: {e:?}"))?;
    let sessions = serve_headless_sessions(
        address,
        certificate,
        settings,
        &display,
        &encoder,
        trace_path,
    );
    run_app_while_serving(&display, &headless.command, async {
        match sessions.await? {}
    })
    .await
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
        clipboard_texts: None,
        nodes: vec![GraphNodeRef::source(source)],
        encoder: HostEncoder::MediaFoundation,
    };
    let open_injector = move |screen| {
        Ok(WithGamepads {
            desktop: Injector::new(screen, monitor)?,
            gamepads: ViGEmGamepads::default(),
        })
    };
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
    match &capture.x11_display {
        Some(display) => host_x11(listener, &settings, display, capture.fps, trace).await,
        None => host_portal(listener, &settings, &capture, trace).await,
    }
    #[cfg(target_os = "windows")]
    host_monitor(listener, &settings, &capture, trace).await;
}

// a --run app plays through the host's own audio server, which carries every other app too
#[cfg(target_os = "linux")]
fn host_audio(flags: &Flags, capture: &CaptureSettings) -> Result<bool, Box<dyn Error>> {
    match (capture.headless.is_some(), optional(flags, "audio")?) {
        (true, Some(true)) => Err("--run sends no audio".into()),
        (true, _) => Ok(false),
        (false, audio) => Ok(audio.unwrap_or(true)),
    }
}

#[cfg(all(target_os = "linux", feature = "gamestream"))]
fn gamestream_x11_capture(display: &str, fps: u32) -> Result<Vec<GraphNode>, String> {
    let rate = x11_rate_filter(fps).map_err(|e| e.to_string())?;
    let convert = parse_element(&default_registry(), "videoconvert").map_err(|e| e.to_string())?;
    Ok(vec![
        GraphNodeRef::source(XImageSrc::new().with_display_name(display)),
        GraphNodeRef::Element(rate),
        GraphNodeRef::Element(convert),
    ])
}

#[cfg(all(target_os = "linux", feature = "gamestream"))]
fn damage_rate_cap(fps: u32) -> String {
    // the portal and the headless display send frames only when the picture changes
    format!("videorate drop-only=true ! video/x-raw,framerate={fps}/1")
}

#[cfg(all(target_os = "linux", feature = "gamestream"))]
fn gamestream_portal_capture(fps: u32, token_path: &Path) -> Result<Vec<GraphNode>, String> {
    let token = read_portal_restore_token(token_path)
        .map_err(|e| format!("{}: {e}", token_path.display()))?;
    let source = PipeWireVideoSrc::new()
        .with_portal(PortalSourceTypes::Monitor)
        .with_portal_cursor_mode(PortalCursorMode::Embedded)
        .with_portal_restore_token(token)
        .with_fps(fps);
    let tokens = source.portal_restore_tokens();
    let path = token_path.to_owned();
    tokio::task::spawn_local(async move {
        if let Err(e) = save_portal_restore_tokens(tokens, &path).await {
            eprintln!("g2g-desktop host: saving the portal restore token: {e}");
        }
    });
    let convert = parse_chain(
        &default_registry(),
        &format!("{} ! videoconvert", damage_rate_cap(fps)),
    )
    .map_err(|e| e.to_string())?;
    Ok(std::iter::once(GraphNodeRef::source(source))
        .chain(convert.into_iter().map(GraphNodeRef::Element))
        .collect())
}

#[cfg(all(target_os = "linux", feature = "gamestream"))]
fn gamestream_display_capture(
    display: &WaylandDisplay,
    fps: u32,
) -> Result<Vec<GraphNode>, String> {
    let upload = parse_chain(
        &default_registry(),
        &format!("dmabuftocuda ! {}", damage_rate_cap(fps)),
    )
    .map_err(|e| e.to_string())?;
    let source = GraphNodeRef::source(WaylandDisplaySrc::on_display(display));
    Ok(std::iter::once(source)
        .chain(upload.into_iter().map(GraphNodeRef::Element))
        .collect())
}

// moonlight asks for 60 fps unless its user picks another rate
#[cfg(all(target_os = "linux", feature = "gamestream"))]
const GAMESTREAM_DISPLAY_FPS: u32 = 60;

#[cfg(all(target_os = "linux", feature = "gamestream"))]
fn launchable_apps(flags: &Flags) -> Result<Vec<LaunchableApp>, Box<dyn Error>> {
    let mut apps: Vec<LaunchableApp> = Vec::new();
    for value in flags.get("app").into_iter().flatten() {
        let invalid = || format!("--app {value}: expected TITLE=COMMAND");
        let (title, command) = value.split_once('=').ok_or_else(invalid)?;
        if title.is_empty() || command.is_empty() {
            return Err(invalid().into());
        }
        if title == "Desktop" || apps.iter().any(|app| app.title == title) {
            return Err(format!("--app {value}: another app is titled {title}").into());
        }
        apps.push(LaunchableApp {
            title: title.to_owned(),
            command: command.to_owned(),
        });
    }
    Ok(apps)
}

#[cfg(all(target_os = "linux", feature = "gamestream"))]
async fn gamestream_headless(
    headless: &HeadlessSettings,
    apps: Vec<LaunchableApp>,
) -> Result<(), Box<dyn Error>> {
    let display_settings = headless_display_settings(GAMESTREAM_DISPLAY_FPS, headless);
    let render_node = display_settings.render_node.clone();
    let driver =
        render_node_driver(Path::new(&render_node)).map_err(|e| format!("{render_node}: {e}"))?;
    if driver != "nvidia" {
        return Err(format!(
            "--protocol gamestream encodes with nvenc, and {render_node} runs on {driver}: pass an NVIDIA render node with --render-node"
        )
        .into());
    }
    let display = WaylandDisplay::start(display_settings)
        .await
        .map_err(|e| format!("starting the headless display: {e:?}"))?;
    let captured = display.clone();
    let injected = display.clone();
    let launched = display.clone();
    let desktop = ServedDesktop {
        capture: Box::new(move |fps| gamestream_display_capture(&captured, fps)),
        open_injector: Box::new(move |screen| {
            Ok(Box::new(WithGamepads {
                desktop: WaylandDisplayInjector::new(screen, &injected),
                gamepads: UinputGamepads::default(),
            }))
        }),
        apps: Some(ServedApps {
            listed: apps,
            start: Box::new(move |command| launch_app(launched.names(), command)),
        }),
    };
    run_app_while_serving(&display, &headless.command, async {
        Ok(g2g_desktop::gamestream::serve(desktop).await?)
    })
    .await
}

#[cfg(all(target_os = "linux", feature = "gamestream"))]
async fn gamestream_host(flags: &Flags) -> Result<(), Box<dyn Error>> {
    if let Some(flag) = flags.keys().find(|flag| {
        ![
            "protocol",
            "x11-display",
            "run",
            "size",
            "render-node",
            "app",
        ]
        .contains(&flag.as_str())
    }) {
        return Err(format!("--protocol gamestream takes no --{flag}").into());
    }
    if let Some(headless) = headless_settings(flags)? {
        return gamestream_headless(&headless, launchable_apps(flags)?).await;
    }
    let desktop = match flag(flags, "x11-display").cloned() {
        Some(display) => ServedDesktop {
            capture: Box::new({
                let display = display.clone();
                move |fps| gamestream_x11_capture(&display, fps)
            }),
            open_injector: Box::new(move |screen| {
                Ok(Box::new(WithGamepads {
                    desktop: XTestInjector::new(&display, screen)?,
                    gamepads: UinputGamepads::default(),
                }))
            }),
            apps: None,
        },
        None => {
            let token_path = portal_restore_token_path()?;
            ServedDesktop {
                capture: Box::new(move |fps| gamestream_portal_capture(fps, &token_path)),
                open_injector: Box::new(|screen| {
                    Ok(Box::new(WithGamepads {
                        desktop: Injector::new(screen)?,
                        gamepads: UinputGamepads::default(),
                    }))
                }),
                apps: None,
            }
        }
    };
    Ok(g2g_desktop::gamestream::serve(desktop).await?)
}

#[cfg(all(target_os = "windows", feature = "gamestream"))]
fn gamestream_capture(fps: u32) -> Result<Vec<GraphNode>, String> {
    let source = D3D11ScreenCaptureSrc::new()
        .with_monitor_index(PRIMARY_MONITOR)
        .with_fps(fps);
    Ok(vec![GraphNodeRef::source(source)])
}

#[cfg(all(target_os = "windows", feature = "gamestream"))]
async fn gamestream_host(flags: &Flags) -> Result<(), Box<dyn Error>> {
    if let Some(flag) = flags.keys().find(|flag| flag.as_str() != "protocol") {
        return Err(format!("--protocol gamestream takes no --{flag}").into());
    }
    let monitor =
        monitor_bounds(PRIMARY_MONITOR).map_err(|e| format!("monitor {PRIMARY_MONITOR}: {e:?}"))?;
    let desktop = ServedDesktop {
        capture: Box::new(gamestream_capture),
        open_injector: Box::new(move |screen| {
            Ok(Box::new(WithGamepads {
                desktop: Injector::new(screen, monitor)?,
                gamepads: ViGEmGamepads::default(),
            }))
        }),
        apps: None,
    };
    Ok(g2g_desktop::gamestream::serve(desktop).await?)
}

async fn host(flags: Flags) -> Result<(), Box<dyn Error>> {
    match flag(&flags, "protocol").map(String::as_str) {
        None | Some("remotewt") => {}
        #[cfg(feature = "gamestream")]
        Some("gamestream") => return gamestream_host(&flags).await,
        #[cfg(not(feature = "gamestream"))]
        Some("gamestream") => {
            return Err("this g2g-desktop was built without the gamestream feature".into())
        }
        Some(other) => return Err(format!("unknown --protocol {other}").into()),
    }
    #[cfg(target_os = "linux")]
    if flags.contains_key("app") {
        return Err("--app needs --protocol gamestream".into());
    }
    let port = parsed(&flags, "port", DEFAULT_PORT)?;
    let capture = CaptureSettings {
        fps: parsed(&flags, "fps", DEFAULT_FPS)?,
        #[cfg(target_os = "linux")]
        portal_restore_token_path: portal_restore_token_path()?,
        #[cfg(target_os = "linux")]
        headless: headless_settings(&flags)?,
        #[cfg(target_os = "linux")]
        x11_display: flag(&flags, "x11-display").cloned(),
        #[cfg(target_os = "windows")]
        monitor_index: parsed(&flags, "monitor-index", PRIMARY_MONITOR)?,
    };
    let settings = HostSettings {
        token: required(&flags, "token")?,
        bitrate: parsed(&flags, "bitrate", DEFAULT_BITRATE)?,
        #[cfg(target_os = "linux")]
        audio: host_audio(&flags, &capture)?,
        #[cfg(target_os = "windows")]
        audio: parsed(&flags, "audio", true)?,
        send_window: optional(&flags, "send-window")?,
        video_mode: optional(&flags, "video-mode")?,
        #[cfg(target_os = "windows")]
        quality_vs_speed: optional(&flags, "quality-vs-speed")?,
    };
    check_token_len(&settings.token)?;
    #[cfg(target_os = "linux")]
    if capture.x11_display.is_some() {
        x11_rate_filter(capture.fps).map_err(|e| format!("--fps {}: {e}", capture.fps))?;
    }
    #[cfg(target_os = "linux")]
    std::fs::create_dir_all(
        capture
            .portal_restore_token_path
            .parent()
            .expect("the token path is joined onto a state directory"),
    )?;
    let trace_path = flag(&flags, "trace").map(PathBuf::from);
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
        let listener =
            HostListener::bind_when_free(address, certificate.clone(), &settings).await?;
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
        audio: parsed(&flags, "audio", true)?,
        transport: match flag(&flags, "transport").map(String::as_str) {
            None | Some("auto") => ClientTransport::Auto,
            Some("quic") => ClientTransport::Quic,
            Some("websocket") => ClientTransport::WebSocket,
            Some(other) => return Err(format!("unknown --transport {other}").into()),
        },
        keyboard_grab: parsed(&flags, "keyboard-grab", true)?,
    };
    let trace = open_trace(flag(&flags, "trace").map(Path::new))?;
    let observer = Observer::new();
    let glass_to_glass = Arc::new(LatencyHistogram::new());
    let session = tokio::select! {
        session = run_client(
            &host,
            port,
            &settings,
            trace.clone().map(|trace| trace as Arc<dyn VisitRecorder>),
            &observer,
            &glass_to_glass,
        ) => Some(session),
        interrupted = tokio::signal::ctrl_c() => {
            interrupted?;
            None
        }
    };
    if session.is_none() {
        // quinn sends the close of a dropped connection from a driver task, which runs only while the runtime does
        tokio::time::sleep(CLOSE_GRACE).await;
    }
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
    let latency = glass_to_glass.snapshot();
    let nanos_to_ms = |ns: u64| ns as f64 / 1.0e6;
    match latency.count {
        0 => eprintln!("g2g-desktop client: glass-to-glass: no presented frame carried a capture time"),
        count => eprintln!(
            "g2g-desktop client: glass-to-glass mean {:.1} ms, p50 {:.1} ms, p99 {:.1} ms, max {:.1} ms, over {count} frames",
            nanos_to_ms(latency.mean_ns),
            nanos_to_ms(latency.p50_ns),
            nanos_to_ms(latency.p99_ns),
            nanos_to_ms(latency.max_ns),
        ),
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

#[cfg(feature = "gamestream")]
fn submit_pin(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn Error>> {
    let (Some(pin), None) = (args.next(), args.next()) else {
        return Err(format!("usage:\n{GAMESTREAM_USAGE}").into());
    };
    g2g_desktop::gamestream::submit_pin(&pin)?;
    eprintln!("g2g-desktop: PIN accepted");
    Ok(())
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
        #[cfg(feature = "gamestream")]
        Some("pin") => submit_pin(args),
        _ => Err(format!(
            "usage:\n{HOST_USAGE}{PLATFORM_HOST_USAGE}\n{GAMESTREAM_USAGE}\n{CLIENT_USAGE}"
        )
        .into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("g2g-desktop: {e}");
            ExitCode::FAILURE
        }
    }
}

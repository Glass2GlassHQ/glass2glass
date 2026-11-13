// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::BTreeMap;
use std::error::Error;
use std::process::ExitCode;
use std::str::FromStr;

use g2g_core::runtime::GraphNodeRef;
use g2g_core::G2gError;
use g2g_desktop::client::{run_client, ClientError, ClientSettings};
use g2g_desktop::host::{inject, run_session, HostSettings, TurnServer};
use g2g_desktop::signaling::SessionSlot;
use g2g_plugins::pipewirevideosrc::PipeWireVideoSrc;
use g2g_plugins::screencastportal::PortalSourceTypes;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, watch};

const USAGE: &str = "usage:
  g2g-desktop host --token TOKEN [--port 8080] [--fps 60] [--bitrate 4000000]
                   [--stun-server HOST:PORT] [--turn-server HOST:PORT --turn-user USER --turn-pass PASS]
  g2g-desktop client --host HOST --token TOKEN [--port 8080] [--inputs true]";
const DEFAULT_PORT: u16 = 8080;
const DEFAULT_FPS: u32 = 60;
const DEFAULT_BITRATE: u32 = 4_000_000;
const HOST_FLAGS: [&str; 8] = [
    "port",
    "token",
    "fps",
    "bitrate",
    "stun-server",
    "turn-server",
    "turn-user",
    "turn-pass",
];
const CLIENT_FLAGS: [&str; 4] = ["host", "port", "token", "inputs"];

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

async fn host_one(stream: TcpStream, settings: HostSettings, fps: u32) {
    let (events, event_receiver) = mpsc::unbounded_channel();
    let (sizes, size_receiver) = watch::channel(None);
    let capture = vec![GraphNodeRef::source(
        PipeWireVideoSrc::new()
            .with_portal(PortalSourceTypes::Monitor)
            .with_fps(fps),
    )];
    let (session, injected) = tokio::join!(
        run_session(stream, capture, &settings, events, sizes),
        inject(event_receiver, size_receiver)
    );
    if let Err(e) = session {
        eprintln!("g2g-desktop host: {e}");
    }
    if let Err(e) = injected {
        eprintln!("g2g-desktop host: input injection: {e}");
    }
}

async fn host(flags: Flags) -> Result<(), Box<dyn Error>> {
    let port = parsed(&flags, "port", DEFAULT_PORT)?;
    let fps = parsed(&flags, "fps", DEFAULT_FPS)?;
    let settings = HostSettings {
        token: required(&flags, "token")?,
        bitrate: parsed(&flags, "bitrate", DEFAULT_BITRATE)?,
        stun_server: flags.get("stun-server").cloned(),
        turn_server: turn_server(&flags)?,
    };
    let listener = TcpListener::bind(("0.0.0.0", port)).await?;
    let slot = SessionSlot::default();
    loop {
        let (stream, peer) = listener.accept().await?;
        let Ok(guard) = slot.claim() else {
            eprintln!("g2g-desktop host: {peer} refused, a session is already running");
            continue;
        };
        let settings = settings.clone();
        tokio::task::spawn_local(async move {
            host_one(stream, settings, fps).await;
            drop(guard);
        });
    }
}

async fn client(flags: Flags) -> Result<(), Box<dyn Error>> {
    let host = required(&flags, "host")?;
    let port = parsed(&flags, "port", DEFAULT_PORT)?;
    let settings = ClientSettings {
        token: required(&flags, "token")?,
        inputs: parsed(&flags, "inputs", true)?,
    };
    let stream = TcpStream::connect((host.as_str(), port)).await?;
    match run_client(stream, &settings).await {
        Ok(_) | Err(ClientError::Session(G2gError::WindowClosed)) => Ok(()),
        Err(e) => Err(e.into()),
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
        Some("host") => parse_flags(args, &HOST_FLAGS)
            .and_then(|flags| runtime.block_on(local.run_until(host(flags)))),
        Some("client") => parse_flags(args, &CLIENT_FLAGS)
            .and_then(|flags| runtime.block_on(local.run_until(client(flags)))),
        _ => Err(USAGE.into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("g2g-desktop: {e}");
            ExitCode::FAILURE
        }
    }
}

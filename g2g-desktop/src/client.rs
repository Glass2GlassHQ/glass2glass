// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::log::Target;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use g2g_core::metrics::LatencyHistogram;
use g2g_core::runtime::{
    parse_chain, GraphNode, GraphNodeRef, Observer, ParseError, RunBuilder, RunStats, VisitRecorder,
};
use g2g_core::{G2gError, Graph};
use g2g_plugins::clock::WallClock;
use g2g_plugins::cursorshape::CursorShape;
use g2g_plugins::registry::default_registry;
use g2g_plugins::remotewtsrc::RemoteWtSrc;
use g2g_plugins::wgpu;
use g2g_plugins::wgpupresent::WgpuPresentSink;
use g2g_plugins::windowinput::WindowInput;
use tokio::sync::{mpsc, watch};

use crate::audio;
use crate::hid::{usage_from_scancode, ScancodeSet};
use crate::lineargraph::linear_graph;
use crate::packet::{
    CursorImage, Event, EventKind, PacketReader, PacketWriter, RemovableDevice, Side, TouchPhase,
    POINTER_BUTTONS,
};
use crate::router::{InputRouter, Origin};

const CLIENT_CATEGORY: &str = "g2g-desktop-client";
const WINDOW_TITLE: &str = "g2g-desktop";
// glass-to-glass latency grows with link depth
const LINK_CAPACITY: usize = 2;
#[cfg(target_os = "linux")]
const DECODE: &str = "h264parse ! ffmpegdec";
// d3d12h264dec decodes into textures wgpusink presents on the decoder's device
#[cfg(target_os = "windows")]
const DECODE: &str = "h264parse ! d3d12h264dec";
#[cfg(target_os = "linux")]
const WINDOW_SCANCODES: ScancodeSet = ScancodeSet::Evdev;
#[cfg(target_os = "windows")]
const WINDOW_SCANCODES: ScancodeSet = ScancodeSet::Windows;
const KEYBOARD_DEVICE: u8 = 0;
const DISPLAY: u8 = 0;
const FORWARDED_KINDS: [EventKind; 5] = [
    EventKind::Key,
    EventKind::Button,
    EventKind::Position,
    EventKind::Wheel,
    EventKind::Touch,
];

#[derive(Debug, Clone)]
pub struct ClientSettings {
    pub token: String,
    pub cert_hash: String,
    pub inputs: bool,
    pub audio: bool,
    pub transport: ClientTransport,
    #[cfg(target_os = "linux")]
    pub keyboard_grab: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientTransport {
    Auto,
    Quic,
    WebSocket,
}

#[derive(Debug)]
pub enum ClientError {
    Launch(ParseError),
    Session(G2gError),
}

impl core::fmt::Display for ClientError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ClientError::Launch(e) => write!(f, "decode pipeline: {e}"),
            ClientError::Session(e) => write!(f, "session: {e:?}"),
        }
    }
}

impl std::error::Error for ClientError {}

fn event_from(input: WindowInput) -> Option<Event> {
    match input {
        WindowInput::Focus(_) => None,
        WindowInput::Key { scancode, down } => {
            let usage = usage_from_scancode(WINDOW_SCANCODES, scancode)?;
            Some(Event::Key {
                device: KEYBOARD_DEVICE,
                usage,
                down,
            })
        }
        WindowInput::Pointer { x, y } => Some(Event::Position {
            x: i32::try_from(x).ok()?,
            y: i32::try_from(y).ok()?,
            display: DISPLAY,
        }),
        WindowInput::Button { button, down } => {
            let button = POINTER_BUTTONS.iter().position(|known| *known == button)?;
            Some(Event::Button {
                button: u8::try_from(button).ok()?,
                down,
            })
        }
        WindowInput::Wheel { dx, dy } => Some(Event::Wheel { dx, dy }),
        WindowInput::TouchDown { contact, x, y } => Some(Event::Touch {
            contact,
            phase: TouchPhase::Down {
                x: i32::try_from(x).ok()?,
                y: i32::try_from(y).ok()?,
                display: DISPLAY,
            },
        }),
        WindowInput::TouchMove { contact, x, y } => Some(Event::Touch {
            contact,
            phase: TouchPhase::Move {
                x: i32::try_from(x).ok()?,
                y: i32::try_from(y).ok()?,
                display: DISPLAY,
            },
        }),
        WindowInput::TouchUp { contact } => Some(Event::Touch {
            contact,
            phase: TouchPhase::Up,
        }),
        WindowInput::TouchCancelAll => Some(Event::DeviceRemoved {
            kind: RemovableDevice::Touchscreen,
            device: 0,
        }),
    }
}

pub async fn forward_window_input(
    mut window: mpsc::UnboundedReceiver<WindowInput>,
    outgoing: mpsc::UnboundedSender<Vec<u8>>,
) {
    let log = Target::category(CLIENT_CATEGORY);
    let mut router = InputRouter::new(Vec::new());
    let mut network = router.subscribe(Origin::Local, &FORWARDED_KINDS);
    let mut writer = PacketWriter::new(Side::Client);
    while let Some(input) = window.recv().await {
        if let WindowInput::Focus(focused) = input {
            router.set_focused(focused);
        }
        match event_from(input) {
            Some(event) => {
                router.push(Origin::Local, event);
            }
            None if !matches!(input, WindowInput::Focus(_)) => {
                g2g_core::g2g_warn!(log, "{input:?} has no packet form");
            }
            None => {}
        }
        while let Ok(event) = network.try_recv() {
            let packet = writer
                .write(&event)
                .expect("input packets are far under the data channel message cap");
            if outgoing.send(packet).is_err() {
                return;
            }
        }
    }
}

fn cursor_shape(image: &CursorImage) -> Option<CursorShape> {
    if image.width() == 0 || image.height() == 0 {
        return None;
    }
    let (hotspot_x, hotspot_y) = image.hotspot();
    Some(CursorShape {
        width: image.width().into(),
        height: image.height().into(),
        hotspot_x: hotspot_x.into(),
        hotspot_y: hotspot_y.into(),
        rgba: image.rgba().to_vec(),
    })
}

pub async fn follow_host_events(
    mut messages: mpsc::UnboundedReceiver<Vec<u8>>,
    cursor_shapes: Option<watch::Sender<Option<CursorShape>>>,
    host_clipboard: Option<watch::Sender<Option<String>>>,
) {
    let log = Target::category(CLIENT_CATEGORY);
    let mut reader = PacketReader::new(Side::Client);
    while let Some(message) = messages.recv().await {
        match reader.read(&message) {
            Ok(Some(Event::Cursor(image))) => {
                if let Some(shapes) = &cursor_shapes {
                    shapes.send_replace(cursor_shape(&image));
                }
            }
            Ok(Some(Event::Clipboard(text))) => {
                if let Some(host_clipboard) = &host_clipboard {
                    host_clipboard.send_replace(Some(text));
                }
            }
            Ok(Some(event)) => {
                g2g_core::g2g_warn!(log, "the client does not handle {:?}", event.kind());
            }
            Ok(None) => {}
            Err(e) => g2g_core::g2g_warn!(log, "dropped a host packet: {e:?}"),
        }
    }
}

#[cfg(target_os = "linux")]
fn show_host_cursor(
    sink: WgpuPresentSink,
    inputs: bool,
) -> (Option<watch::Sender<Option<CursorShape>>>, WgpuPresentSink) {
    if !inputs {
        return (None, sink);
    }
    let (sender, shapes) = watch::channel(None);
    (Some(sender), sink.with_cursor_shapes(shapes))
}

#[cfg(not(target_os = "linux"))]
fn show_host_cursor(
    sink: WgpuPresentSink,
    _inputs: bool,
) -> (Option<watch::Sender<Option<CursorShape>>>, WgpuPresentSink) {
    (None, sink)
}

async fn forward_local_clipboard(
    mut copies: mpsc::UnboundedReceiver<String>,
    outgoing: mpsc::UnboundedSender<Vec<u8>>,
) {
    let log = Target::category(CLIENT_CATEGORY);
    let mut writer = PacketWriter::new(Side::Client);
    while let Some(text) = copies.recv().await {
        match writer.write(&Event::Clipboard(text)) {
            Ok(packet) => {
                if outgoing.send(packet).is_err() {
                    return;
                }
            }
            Err(e) => g2g_core::g2g_warn!(log, "dropped a clipboard: {e}"),
        }
    }
}

struct ClipboardSync {
    host_texts: watch::Sender<Option<String>>,
    local_copies: mpsc::UnboundedReceiver<String>,
}

#[cfg(target_os = "linux")]
fn grab_keyboard(sink: WgpuPresentSink, settings: &ClientSettings) -> WgpuPresentSink {
    if settings.inputs && settings.keyboard_grab {
        return sink.with_keyboard_grab();
    }
    sink
}

#[cfg(not(target_os = "linux"))]
fn grab_keyboard(sink: WgpuPresentSink, _settings: &ClientSettings) -> WgpuPresentSink {
    sink
}

#[cfg(target_os = "linux")]
fn sync_clipboard(sink: &mut WgpuPresentSink, inputs: bool) -> Option<ClipboardSync> {
    if !inputs {
        return None;
    }
    let (host_texts, remote_texts) = watch::channel(None);
    let local_copies = sink.sync_clipboard(remote_texts);
    Some(ClipboardSync {
        host_texts,
        local_copies,
    })
}

#[cfg(not(target_os = "linux"))]
fn sync_clipboard(_sink: &mut WgpuPresentSink, _inputs: bool) -> Option<ClipboardSync> {
    None
}

fn client_graph(
    source: RemoteWtSrc,
    sink: WgpuPresentSink,
) -> Result<Graph<GraphNode>, ClientError> {
    let decode = parse_chain(&default_registry(), DECODE).map_err(ClientError::Launch)?;
    Ok(linear_graph(
        GraphNodeRef::source(source),
        decode.into_iter().map(GraphNodeRef::Element),
        GraphNodeRef::element(sink),
    ))
}

pub async fn run_client(
    host: &str,
    port: u16,
    settings: &ClientSettings,
    trace: Option<Arc<dyn VisitRecorder>>,
    observer: &Observer,
    glass_to_glass: &Arc<LatencyHistogram>,
) -> Result<RunStats, ClientError> {
    // a caller-mode source binds nothing, the address is unused
    let unbound = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0));
    let quic = format!("quic://{host}:{port}");
    let websocket = format!("wss://{host}:{port}");
    let hash = settings.cert_hash.as_str();
    let mut source = match settings.transport {
        ClientTransport::Auto => RemoteWtSrc::new(unbound)
            .with_caller(quic, hash)
            .with_fallback_location(websocket),
        ClientTransport::Quic => RemoteWtSrc::new(unbound).with_caller(quic, hash),
        ClientTransport::WebSocket => RemoteWtSrc::new(unbound).with_caller(websocket, hash),
    };
    let data_channel = source.open_data_channel();
    let audio_track = settings.audio.then(|| source.open_track());
    data_channel
        .outgoing
        .send(settings.token.as_bytes().to_vec())
        .expect("the source holds the data channel until it runs");
    let window = WgpuPresentSink::new()
        .with_title(WINDOW_TITLE)
        .with_sync(false)
        .with_present_mode(wgpu::PresentMode::Mailbox)
        .with_latency_histogram(Arc::clone(glass_to_glass));
    let (cursor_shapes, mut sink) =
        show_host_cursor(grab_keyboard(window, settings), settings.inputs);
    let window_input = settings.inputs.then(|| sink.input_events());
    let (host_clipboard, local_copies) = sync_clipboard(&mut sink, settings.inputs)
        .map(|sync| (sync.host_texts, sync.local_copies))
        .unzip();
    let graph = client_graph(source, sink)?;

    let clipboard_outgoing = data_channel.outgoing.clone();
    let forward = async {
        if let Some(window_input) = window_input {
            forward_window_input(window_input, data_channel.outgoing).await;
        }
    };
    let copies = async {
        if let Some(local_copies) = local_copies {
            forward_local_clipboard(local_copies, clipboard_outgoing).await;
        }
    };
    let follow = follow_host_events(data_channel.incoming, cursor_shapes, host_clipboard);
    let clock = WallClock::new();
    let mut run = RunBuilder::new(graph, &clock, LINK_CAPACITY).with_observer(observer);
    if let Some(trace) = trace {
        run = run.with_visit_recorder(trace);
    }
    let (run, (), (), (), ()) =
        tokio::join!(run.run(), forward, copies, follow, audio::play(audio_track));
    run.map_err(ClientError::Session)
}

// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::log::Target;
use g2g_core::runtime::{parse_chain, run_graph, GraphNode, GraphNodeRef, ParseError, RunStats};
use g2g_core::{G2gError, Graph};
use g2g_plugins::clock::WallClock;
use g2g_plugins::livesync::LiveSync;
use g2g_plugins::registry::default_registry;
use g2g_plugins::webrtcduplex::{SdpChannel, SignalRole, WebRtcDuplexSession};
use g2g_plugins::wgpupresent::WgpuPresentSink;
use g2g_plugins::windowinput::{PointerButton, WindowInput};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::mpsc;

use crate::hid::{usage_from_scancode, ScancodeSet};
use crate::host::INPUT_LABEL;
use crate::packet::{Event, EventKind, PacketWriter, Side};
use crate::router::{InputRouter, Origin};
use crate::signaling::{ClientEvent, ClientSignaling, SignalingError};

const SIGNALING_READ_BYTES: usize = 4096;
const CLIENT_CATEGORY: &str = "g2g-desktop-client";
const WINDOW_TITLE: &str = "g2g-desktop";
const SDP_CHANNEL_DEPTH: usize = 4;
// glass-to-glass latency grows with link depth
const LINK_CAPACITY: usize = 2;
const DECODE: &str = "h264parse ! ffmpegdec";
const KEYBOARD_DEVICE: u8 = 0;
const DISPLAY: u8 = 0;
const FORWARDED_KINDS: [EventKind; 4] = [
    EventKind::Key,
    EventKind::Button,
    EventKind::Position,
    EventKind::Wheel,
];

#[derive(Debug)]
pub enum ExchangeError {
    Signaling(SignalingError),
    Io(std::io::Error),
    SessionGone,
}

impl core::fmt::Display for ExchangeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ExchangeError::Signaling(e) => write!(f, "signaling: {e}"),
            ExchangeError::Io(e) => write!(f, "signaling connection: {e}"),
            ExchangeError::SessionGone => write!(f, "the session ended before the SDP exchange"),
        }
    }
}

impl std::error::Error for ExchangeError {}

impl From<SignalingError> for ExchangeError {
    fn from(e: SignalingError) -> Self {
        ExchangeError::Signaling(e)
    }
}

impl From<std::io::Error> for ExchangeError {
    fn from(e: std::io::Error) -> Self {
        ExchangeError::Io(e)
    }
}

async fn next_event(
    stream: &mut TcpStream,
    signaling: &mut ClientSignaling,
) -> Result<ClientEvent, ExchangeError> {
    let mut buffer = [0u8; SIGNALING_READ_BYTES];
    loop {
        let read = stream.read(&mut buffer).await?;
        if read == 0 {
            return Err(ExchangeError::Signaling(SignalingError::HungUp));
        }
        if let Some(event) = signaling.receive(&buffer[..read])?.into_iter().next() {
            return Ok(event);
        }
    }
}

pub async fn exchange_sdp(
    mut stream: TcpStream,
    token: &str,
    mut offers: mpsc::Receiver<String>,
    answers: mpsc::Sender<String>,
) -> Result<(), ExchangeError> {
    let (mut signaling, framed_token) = ClientSignaling::new(token)?;
    stream.write_all(&framed_token).await?;
    if next_event(&mut stream, &mut signaling).await? != ClientEvent::Accepted {
        return Err(ExchangeError::Signaling(SignalingError::UnexpectedMessage));
    }
    let offer = offers.recv().await.ok_or(ExchangeError::SessionGone)?;
    stream.write_all(&signaling.offer(&offer)?).await?;
    let ClientEvent::Answer(answer) = next_event(&mut stream, &mut signaling).await? else {
        return Err(ExchangeError::Signaling(SignalingError::UnexpectedMessage));
    };
    answers
        .send(answer)
        .await
        .map_err(|_| ExchangeError::SessionGone)?;
    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).await?;
    signaling.receive(&rest)?;
    Ok(signaling.connection_closed()?)
}

#[derive(Debug, Clone)]
pub struct ClientSettings {
    pub token: String,
    pub inputs: bool,
}

#[derive(Debug)]
pub enum ClientError {
    Exchange(ExchangeError),
    Launch(ParseError),
    Session(G2gError),
}

impl core::fmt::Display for ClientError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ClientError::Exchange(e) => e.fmt(f),
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
            let usage = usage_from_scancode(ScancodeSet::Evdev, scancode)?;
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
            let button = match button {
                PointerButton::Left => 0,
                PointerButton::Right => 1,
                PointerButton::Middle => 2,
            };
            Some(Event::Button { button, down })
        }
        WindowInput::Wheel { dx, dy } => Some(Event::Wheel { dx, dy }),
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

fn client_graph(
    session: WebRtcDuplexSession,
    sink: WgpuPresentSink,
) -> Result<Graph<GraphNode>, ClientError> {
    let decode = parse_chain(&default_registry(), DECODE).map_err(ClientError::Launch)?;
    let mut graph = Graph::new();
    let duplex = graph.add_duplex(GraphNodeRef::duplex(session), 0, 1);
    let mut previous = duplex.output(0);
    for element in decode {
        let node = graph.add_transform(GraphNodeRef::Element(element));
        graph
            .link(previous, node)
            .expect("transforms have one input and one output pad");
        previous = node.into();
    }
    let live = graph.add_muxer(GraphNodeRef::muxer(LiveSync::new()), 1);
    let display = graph.add_sink(GraphNodeRef::element(sink));
    graph
        .link(previous, live.input(0))
        .and_then(|()| graph.link(live.output(), display))
        .expect("livesync has one input and one output pad");
    Ok(graph)
}

pub async fn run_client(
    stream: TcpStream,
    settings: &ClientSettings,
) -> Result<RunStats, ClientError> {
    let (offer_sender, offers) = mpsc::channel(SDP_CHANNEL_DEPTH);
    let (answers, answer_receiver) = mpsc::channel(SDP_CHANNEL_DEPTH);
    let mut session = WebRtcDuplexSession::new(
        SignalRole::Offerer,
        SdpChannel::from_halves(offer_sender, answer_receiver),
        1,
    )
    .with_receive_only();
    let data_channel = session.open_data_channel(INPUT_LABEL);
    let mut sink = WgpuPresentSink::new().with_title(WINDOW_TITLE);
    let window_input = settings.inputs.then(|| sink.input_events());
    let graph = client_graph(session, sink)?;

    let forward = async {
        if let Some(window_input) = window_input {
            forward_window_input(window_input, data_channel.outgoing).await;
        }
    };
    let clock = WallClock::new();
    let (exchanged, run, ()) = tokio::join!(
        exchange_sdp(stream, &settings.token, offers, answers),
        run_graph(graph, &clock, LINK_CAPACITY),
        forward
    );
    exchanged.map_err(ClientError::Exchange)?;
    run.map_err(ClientError::Session)
}

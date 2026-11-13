// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::fmt;
use core::future::Future;
use core::pin::Pin;
use std::io;
use std::time::Duration;

use g2g_core::log::Target;
use g2g_core::runtime::{parse_chain, run_graph, GraphNode, GraphNodeRef, ParseError, RunStats};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, OutputSink,
    PipelinePacket,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::registry::default_registry;
use g2g_plugins::webrtcduplex::{SdpChannel, SignalRole, WebRtcDuplexSession};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, watch};

use crate::packet::{Event, PacketReader, Side};
use crate::signaling::{HostEvent, HostSignaling, SignalingError};
use crate::uinput::{InjectError, Injector, ScreenSize};

pub const INPUT_LABEL: &str = "input";
const HOST_CATEGORY: &str = "g2g-desktop-host";
const SIGNALING_TIMEOUT: Duration = Duration::from_secs(10);
const SIGNALING_READ_BYTES: usize = 4096;
// glass-to-glass latency grows with link depth
const LINK_CAPACITY: usize = 2;
const SDP_CHANNEL_DEPTH: usize = 4;

#[derive(Debug, Clone)]
pub struct TurnServer {
    pub server: String,
    pub user: String,
    pub pass: String,
}

#[derive(Debug, Clone)]
pub struct HostSettings {
    pub token: String,
    pub bitrate: u32,
    pub stun_server: Option<String>,
    pub turn_server: Option<TurnServer>,
}

#[derive(Debug)]
pub enum HostError {
    Signaling(SignalingError),
    SignalingTimeout,
    Io(io::Error),
    Launch(ParseError),
    Session(G2gError),
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostError::Signaling(e) => write!(f, "signaling: {e}"),
            HostError::SignalingTimeout => write!(
                f,
                "client sent no offer within {} seconds",
                SIGNALING_TIMEOUT.as_secs()
            ),
            HostError::Io(e) => write!(f, "signaling connection: {e}"),
            HostError::Launch(e) => write!(f, "capture pipeline: {e}"),
            HostError::Session(e) => write!(f, "session: {e:?}"),
        }
    }
}

impl std::error::Error for HostError {}

impl From<SignalingError> for HostError {
    fn from(e: SignalingError) -> Self {
        HostError::Signaling(e)
    }
}

impl From<io::Error> for HostError {
    fn from(e: io::Error) -> Self {
        HostError::Io(e)
    }
}

fn screen_size(caps: &Caps) -> Result<ScreenSize, G2gError> {
    let Caps::RawVideo {
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        ..
    } = caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    let side = |pixels: &u32| u16::try_from(*pixels).map_err(|_| G2gError::CapsMismatch);
    Ok(ScreenSize {
        width: side(width)?,
        height: side(height)?,
    })
}

#[derive(Debug)]
struct CaptureSize {
    sizes: watch::Sender<Option<ScreenSize>>,
}

impl CaptureSize {
    fn publish(&self, caps: &Caps) -> Result<(), G2gError> {
        let size = Some(screen_size(caps)?);
        self.sizes.send_if_modified(|current| {
            let changed = *current != size;
            *current = size;
            changed
        });
        Ok(())
    }
}

impl AsyncElement for CaptureSize {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.publish(absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::CapsChanged(caps) = &packet {
                self.publish(caps)?;
            }
            if !matches!(packet, PipelinePacket::Eos) {
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

async fn read_offer(
    stream: &mut TcpStream,
    signaling: &mut HostSignaling,
) -> Result<String, HostError> {
    let mut buffer = [0u8; SIGNALING_READ_BYTES];
    loop {
        let read = stream.read(&mut buffer).await?;
        if read == 0 {
            return Err(HostError::Signaling(SignalingError::HungUp));
        }
        for event in signaling.receive(&buffer[..read])? {
            match event {
                HostEvent::Reply(bytes) => stream.write_all(&bytes).await?,
                HostEvent::Offer(sdp) => return Ok(sdp),
            }
        }
    }
}

fn host_graph(
    capture: Vec<GraphNode>,
    settings: &HostSettings,
    session: WebRtcDuplexSession,
    sizes: watch::Sender<Option<ScreenSize>>,
) -> Result<Graph<GraphNode>, HostError> {
    let encode = format!(
        "videoconvert ! ffmpegenc backend=software bitrate={}",
        settings.bitrate
    );
    let encode = parse_chain(&default_registry(), &encode).map_err(HostError::Launch)?;
    let mut capture = capture.into_iter();
    let mut graph = Graph::new();
    let mut previous = graph.add_source(capture.next().expect("capture starts with a source"));
    let transforms = capture
        .chain([GraphNodeRef::element(CaptureSize { sizes })])
        .chain(encode.into_iter().map(GraphNodeRef::Element));
    for transform in transforms {
        let node = graph.add_transform(transform);
        graph
            .link(previous, node)
            .expect("transforms have one input and one output pad");
        previous = node;
    }
    let duplex = graph.add_duplex(GraphNodeRef::duplex(session), 1, 1);
    let drain = graph.add_sink(GraphNodeRef::element(FakeSink::new()));
    graph
        .link(previous, duplex.input(0))
        .and_then(|()| graph.link(duplex.output(0), drain))
        .expect("the duplex node has one input and one output pad");
    Ok(graph)
}

fn duplex_session(settings: &HostSettings, sdp: SdpChannel) -> WebRtcDuplexSession {
    let mut session = WebRtcDuplexSession::new(SignalRole::Answerer, sdp, 1);
    if let Some(server) = &settings.stun_server {
        session = session.with_stun_server(server.clone());
    }
    if let Some(turn) = &settings.turn_server {
        session =
            session.with_turn_server(turn.server.clone(), turn.user.clone(), turn.pass.clone());
    }
    session
}

async fn forward_input(
    mut messages: mpsc::UnboundedReceiver<Vec<u8>>,
    events: mpsc::UnboundedSender<Event>,
) {
    let log = Target::category(HOST_CATEGORY);
    let mut reader = PacketReader::new(Side::Host);
    while let Some(message) = messages.recv().await {
        match reader.read(&message) {
            Ok(Some(event)) => {
                if events.send(event).is_err() {
                    return;
                }
            }
            Ok(None) => {}
            Err(e) => g2g_core::g2g_warn!(log, "dropped an input packet: {e:?}"),
        }
    }
}

pub async fn run_session(
    mut stream: TcpStream,
    capture: Vec<GraphNode>,
    settings: &HostSettings,
    events: mpsc::UnboundedSender<Event>,
    sizes: watch::Sender<Option<ScreenSize>>,
) -> Result<RunStats, HostError> {
    let mut signaling = HostSignaling::new(&settings.token)?;
    let offer = tokio::time::timeout(SIGNALING_TIMEOUT, read_offer(&mut stream, &mut signaling))
        .await
        .map_err(|_| HostError::SignalingTimeout)??;

    let (offer_sender, offer_receiver) = mpsc::channel(SDP_CHANNEL_DEPTH);
    let (answer_sender, mut answer_receiver) = mpsc::channel(SDP_CHANNEL_DEPTH);
    offer_sender
        .send(offer)
        .await
        .expect("the session holds the offer receiver");
    let mut session = duplex_session(
        settings,
        SdpChannel::from_halves(answer_sender, offer_receiver),
    );
    let input = session.open_data_channel(INPUT_LABEL);
    let graph = host_graph(capture, settings, session, sizes)?;

    let send_answer = async move {
        let answer = answer_receiver
            .recv()
            .await
            .ok_or(HostError::Signaling(SignalingError::HungUp))?;
        stream.write_all(&signaling.answer(&answer)?).await?;
        signaling.connection_closed()?;
        Ok::<(), HostError>(())
    };
    let clock = WallClock::new();
    let (run, answered, ()) = tokio::join!(
        run_graph(graph, &clock, LINK_CAPACITY),
        send_answer,
        forward_input(input.incoming, events)
    );
    answered?;
    run.map_err(HostError::Session)
}

pub async fn inject(
    mut events: mpsc::UnboundedReceiver<Event>,
    mut sizes: watch::Receiver<Option<ScreenSize>>,
) -> Result<(), InjectError> {
    let log = Target::category(HOST_CATEGORY);
    let mut injector: Option<Injector> = None;
    loop {
        tokio::select! {
            changed = sizes.changed() => {
                if changed.is_err() {
                    return Ok(());
                }
                let size = *sizes.borrow_and_update();
                injector = size.map(Injector::new).transpose()?;
            }
            event = events.recv() => {
                let Some(event) = event else {
                    return Ok(());
                };
                let Some(injector) = injector.as_mut() else {
                    g2g_core::g2g_warn!(log, "dropped {:?} before the capture size was known", event.kind());
                    continue;
                };
                match injector.inject(&event) {
                    Ok(()) => {}
                    Err(InjectError::Device(e)) => return Err(InjectError::Device(e)),
                    Err(e) => g2g_core::g2g_warn!(log, "did not inject {:?}: {e}", event.kind()),
                }
            }
        }
    }
}

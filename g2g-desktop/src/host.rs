// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::fmt;
use core::future::Future;
use core::pin::Pin;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use g2g_core::log::Target;
use g2g_core::runtime::{
    parse_chain, GraphNode, GraphNodeRef, ParseError, RunBuilder, RunStats, VisitRecorder,
};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, OutputSink,
    PipelinePacket, PropError, PropValue,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::cursorshape::CursorShape;
use g2g_plugins::datachannel::DataChannelEnds;
use g2g_plugins::registry::default_registry;
use g2g_plugins::remotewtio::SelfSignedCertificate;
use g2g_plugins::remotewtsink::RemoteWtSink;
use g2g_plugins::remotewttrack::RemoteWtTrackSink;
use subtle::ConstantTimeEq;
use tokio::sync::{mpsc, watch};

use crate::audio;
use crate::injection::{InjectError, InjectEvents, ScreenSize};
use crate::lineargraph::linear_graph;
use crate::packet::{CursorImage, Event, PacketError, PacketReader, PacketWriter, Side};

const HOST_CATEGORY: &str = "g2g-desktop-host";
const TOKEN_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_TOKEN_LEN: usize = 256;
// quinn keeps a closed connection's UDP socket open while the connection drains
const PORT_FREE_TIMEOUT: Duration = Duration::from_secs(3);
const PORT_POLL_INTERVAL: Duration = Duration::from_millis(20);
// glass-to-glass latency grows with link depth
pub(crate) const LINK_CAPACITY: usize = 2;

#[derive(Debug, Clone)]
pub struct HostSettings {
    pub token: String,
    pub bitrate: u32,
    pub audio: bool,
    pub send_window: Option<u64>,
    pub video_mode: Option<String>,
    #[cfg(target_os = "windows")]
    pub quality_vs_speed: Option<u32>,
}

#[derive(Debug)]
pub enum HostError {
    TokenLength(usize),
    SendWindow(PropError),
    VideoMode(PropError),
    Port(std::io::Error),
    TokenTimeout,
    BadToken,
    HungUp,
    Launch(ParseError),
    Session(G2gError),
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostError::TokenLength(len) => {
                write!(f, "token is {len} bytes, must be 1 to {MAX_TOKEN_LEN}")
            }
            HostError::SendWindow(e) => write!(f, "send window: {e:?}"),
            HostError::VideoMode(e) => write!(f, "video mode: {e:?}"),
            HostError::Port(e) => write!(f, "port: {e}"),
            HostError::TokenTimeout => write!(
                f,
                "client sent no token within {} seconds",
                TOKEN_TIMEOUT.as_secs()
            ),
            HostError::BadToken => write!(f, "client token does not match"),
            HostError::HungUp => write!(f, "client closed the connection before its token"),
            HostError::Launch(e) => write!(f, "capture pipeline: {e}"),
            HostError::Session(e) => write!(f, "session: {e:?}"),
        }
    }
}

impl std::error::Error for HostError {}

pub fn check_token_len(token: &str) -> Result<(), HostError> {
    if token.is_empty() || token.len() > MAX_TOKEN_LEN {
        return Err(HostError::TokenLength(token.len()));
    }
    Ok(())
}

#[derive(Debug)]
pub struct HostListener {
    sink: RemoteWtSink,
    channel: DataChannelEnds,
    audio_track: Option<RemoteWtTrackSink>,
    address: SocketAddr,
}

impl HostListener {
    pub fn bind(
        address: SocketAddr,
        certificate: SelfSignedCertificate,
        settings: &HostSettings,
    ) -> Result<Self, HostError> {
        let mut sink =
            RemoteWtSink::new(format!("quic://{address}")).with_self_signed_listener(certificate);
        AsyncElement::set_property(
            &mut sink,
            "congestion-control",
            PropValue::Str("low-latency".into()),
        )
        .expect("low-latency is a congestion-control value");
        if let Some(bytes) = settings.send_window {
            AsyncElement::set_property(&mut sink, "send-window", PropValue::Uint(bytes))
                .map_err(HostError::SendWindow)?;
        }
        if let Some(mode) = &settings.video_mode {
            AsyncElement::set_property(&mut sink, "video-mode", PropValue::Str(mode.clone()))
                .map_err(HostError::VideoMode)?;
        }
        let channel = sink.open_data_channel();
        let audio_track = settings.audio.then(|| sink.open_track());
        let address = sink.listen().map_err(HostError::Session)?;
        Ok(Self {
            sink,
            channel,
            audio_track,
            address,
        })
    }

    pub async fn bind_when_free(
        address: SocketAddr,
        certificate: SelfSignedCertificate,
        settings: &HostSettings,
    ) -> Result<Self, HostError> {
        let deadline = tokio::time::Instant::now() + PORT_FREE_TIMEOUT;
        loop {
            let probe = std::net::UdpSocket::bind(address);
            match probe {
                Ok(probe) => {
                    drop(probe);
                    return Self::bind(address, certificate, settings);
                }
                Err(error)
                    if error.kind() == std::io::ErrorKind::AddrInUse
                        && tokio::time::Instant::now() < deadline =>
                {
                    tokio::time::sleep(PORT_POLL_INTERVAL).await;
                }
                Err(error) => return Err(HostError::Port(error)),
            }
        }
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.address
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
pub(crate) struct CaptureSize {
    pub(crate) sizes: watch::Sender<Option<ScreenSize>>,
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

    fn output_domain_follows_input(&self) -> bool {
        true
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostEncoder {
    #[cfg(target_os = "linux")]
    Software,
    #[cfg(target_os = "linux")]
    Vaapi { render_node: String },
    #[cfg(target_os = "linux")]
    Nvenc,
    #[cfg(target_os = "windows")]
    MediaFoundation,
}

impl HostEncoder {
    fn launch_chain(&self, settings: &HostSettings) -> String {
        let bitrate = settings.bitrate;
        match self {
            #[cfg(target_os = "linux")]
            HostEncoder::Software => {
                format!("videoconvert ! ffmpegenc backend=software bitrate={bitrate}")
            }
            // vah264enc takes gst's kbit/s
            #[cfg(target_os = "linux")]
            HostEncoder::Vaapi { render_node } => {
                format!(
                    "vah264enc device-path={render_node} bitrate={}",
                    bitrate / 1000
                )
            }
            #[cfg(target_os = "linux")]
            HostEncoder::Nvenc => format!("dmabuftocuda ! nvenc bitrate={bitrate}"),
            #[cfg(target_os = "windows")]
            HostEncoder::MediaFoundation => {
                let speed = settings
                    .quality_vs_speed
                    .map(|quality| format!(" quality-vs-speed={quality}"))
                    .unwrap_or_default();
                format!("d3d11convert ! mfh264enc hardware=true bitrate={bitrate}{speed}")
            }
        }
    }
}

fn host_graph(
    capture: Vec<GraphNode>,
    encoder: &HostEncoder,
    settings: &HostSettings,
    sink: RemoteWtSink,
    sizes: watch::Sender<Option<ScreenSize>>,
) -> Result<Graph<GraphNode>, HostError> {
    let encode = parse_chain(&default_registry(), &encoder.launch_chain(settings))
        .map_err(HostError::Launch)?;
    let mut capture = capture.into_iter();
    let source = capture.next().expect("capture starts with a source");
    let transforms = capture
        .chain([GraphNodeRef::element(CaptureSize { sizes })])
        .chain(encode.into_iter().map(GraphNodeRef::Element));
    Ok(linear_graph(
        source,
        transforms,
        GraphNodeRef::element(sink),
    ))
}

pub struct HostCapture {
    pub nodes: Vec<GraphNode>,
    pub encoder: HostEncoder,
    pub cursor_shapes: Option<watch::Receiver<Option<CursorShape>>>,
    pub clipboard_texts: Option<watch::Receiver<Option<String>>>,
}

impl fmt::Debug for HostCapture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostCapture")
            .field("nodes", &self.nodes.len())
            .field("encoder", &self.encoder)
            .field("cursor_shapes", &self.cursor_shapes)
            .field("clipboard_texts", &self.clipboard_texts)
            .finish()
    }
}

fn cursor_image(shape: Option<&CursorShape>) -> Result<CursorImage, PacketError> {
    let Some(shape) = shape else {
        return CursorImage::new(0, 0, 0, 0, Vec::new());
    };
    let too_large = || PacketError::TooLarge {
        len: shape.rgba.len(),
    };
    CursorImage::new(
        u16::try_from(shape.width).map_err(|_| too_large())?,
        u16::try_from(shape.height).map_err(|_| too_large())?,
        i16::try_from(shape.hotspot_x).map_err(|_| too_large())?,
        i16::try_from(shape.hotspot_y).map_err(|_| too_large())?,
        shape.rgba.clone(),
    )
}

async fn forward_changes<T>(
    values: Option<watch::Receiver<T>>,
    outgoing: &mpsc::UnboundedSender<Vec<u8>>,
    event_for: impl Fn(&T) -> Option<Result<Event, PacketError>>,
) {
    let Some(mut values) = values else {
        return;
    };
    let log = Target::category(HOST_CATEGORY);
    let mut writer = PacketWriter::new(Side::Host);
    loop {
        tokio::select! {
            changed = values.changed() => {
                if changed.is_err() {
                    return;
                }
            }
            () = outgoing.closed() => return,
        }
        let Some(event) = event_for(&values.borrow_and_update()) else {
            continue;
        };
        match event.and_then(|event| writer.write(&event)) {
            Ok(packet) => {
                if outgoing.send(packet).is_err() {
                    return;
                }
            }
            Err(e) => g2g_core::g2g_warn!(log, "dropped a message to the client: {e}"),
        }
    }
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
    listener: HostListener,
    capture: HostCapture,
    settings: &HostSettings,
    events: mpsc::UnboundedSender<Event>,
    sizes: watch::Sender<Option<ScreenSize>>,
    trace: Option<Arc<dyn VisitRecorder>>,
) -> Result<RunStats, HostError> {
    let HostListener {
        mut sink,
        mut channel,
        audio_track,
        ..
    } = listener;
    sink.connect().await.map_err(HostError::Session)?;
    let token = tokio::time::timeout(TOKEN_TIMEOUT, channel.incoming.recv())
        .await
        .map_err(|_| HostError::TokenTimeout)?
        .ok_or(HostError::HungUp)?;
    if !bool::from(settings.token.as_bytes().ct_eq(&token)) {
        return Err(HostError::BadToken);
    }
    let graph = host_graph(capture.nodes, &capture.encoder, settings, sink, sizes)?;
    let clock = WallClock::new();
    let mut run = RunBuilder::new(graph, &clock, LINK_CAPACITY);
    if let Some(trace) = trace {
        run = run.with_visit_recorder(trace);
    }
    let cursor_events =
        |shape: &Option<CursorShape>| Some(cursor_image(shape.as_ref()).map(Event::Cursor));
    let clipboard_events =
        |text: &Option<String>| text.clone().map(|text| Ok(Event::Clipboard(text)));
    if let Some(track) = audio_track {
        audio::send(track);
    }
    let (run, (), (), ()) = tokio::join!(
        run.run(),
        forward_input(channel.incoming, events),
        forward_changes(capture.cursor_shapes, &channel.outgoing, cursor_events),
        forward_changes(capture.clipboard_texts, &channel.outgoing, clipboard_events)
    );
    run.map_err(HostError::Session)
}

pub async fn inject<I: InjectEvents>(
    mut events: mpsc::UnboundedReceiver<Event>,
    mut sizes: watch::Receiver<Option<ScreenSize>>,
    open_injector: impl Fn(ScreenSize) -> Result<I, InjectError>,
) -> Result<(), InjectError> {
    let log = Target::category(HOST_CATEGORY);
    let mut injector: Option<I> = None;
    loop {
        tokio::select! {
            changed = sizes.changed() => {
                if changed.is_err() {
                    return Ok(());
                }
                let size = *sizes.borrow_and_update();
                injector = size.map(&open_injector).transpose()?;
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
                    Err(e @ InjectError::NoUsLayoutKey(_)) => g2g_core::g2g_debug!(log, "{e}"),
                    Err(e) => g2g_core::g2g_warn!(log, "did not inject {:?}: {e}", event.kind()),
                }
            }
        }
    }
}

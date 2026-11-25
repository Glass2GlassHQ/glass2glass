// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::fmt;
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::pin::pin;

use g2g_core::runtime::{parse_chain, GraphNode, GraphNodeRef, ParseError, RunBuilder};
use g2g_core::{AsyncElement, G2gError, PropValue};
use g2g_plugins::clock::WallClock;
use g2g_plugins::moonlight::videopay::MoonlightVideoPay;
use g2g_plugins::registry::default_registry;
use g2g_plugins::udpsink::UdpSink;
use tokio::net::UdpSocket;
use tokio::sync::{oneshot, Notify};

use super::control;
use super::rtsp::StreamSettings;
use crate::host::LINK_CAPACITY;
use crate::lineargraph::linear_graph;

pub(crate) const VIDEO_PORT: u16 = 47998;
pub(crate) const AUDIO_PORT: u16 = 48000;
// moonlight's legacy video ping, which it sends when SETUP returns no ping payload
const VIDEO_PING: &[u8] = b"PING";
const FEC_PERCENTAGE: u32 = 20;

pub type CaptureFactory = Box<dyn Fn(u32) -> Result<Vec<GraphNode>, String>>;

#[derive(Debug)]
pub(crate) enum StreamError {
    Io(io::Error),
    Control(tokio_enet::Error),
    NotPlayed,
    Capture(String),
    Encoder(ParseError),
    Property(&'static str),
    Graph(G2gError),
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Control(e) => write!(f, "control stream: {e}"),
            Self::NotPlayed => f.write_str("the session ended before RTSP PLAY"),
            Self::Capture(e) => write!(f, "capture: {e}"),
            Self::Encoder(e) => write!(f, "encoder: {e}"),
            Self::Property(name) => write!(f, "the stream refused {name}"),
            Self::Graph(e) => write!(f, "{e:?}"),
        }
    }
}

pub(crate) struct SessionSockets {
    control: tokio_enet::Host,
    video: UdpSocket,
}

impl fmt::Debug for SessionSockets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionSockets")
            .field("video", &self.video)
            .finish_non_exhaustive()
    }
}

pub(crate) async fn bind() -> Result<SessionSockets, StreamError> {
    Ok(SessionSockets {
        control: control::bind().map_err(StreamError::Control)?,
        video: UdpSocket::bind((Ipv4Addr::UNSPECIFIED, VIDEO_PORT))
            .await
            .map_err(StreamError::Io)?,
    })
}

async fn wait_for_ping(socket: &UdpSocket, client: IpAddr) -> io::Result<SocketAddr> {
    let mut buffer = [0u8; 64];
    loop {
        let (length, from) = socket.recv_from(&mut buffer).await?;
        if from.ip() == client && &buffer[..length] == VIDEO_PING {
            return Ok(from);
        }
    }
}

// moonlight's bitrate covers the stream with its parity
fn encoder_bitrate(settings: &StreamSettings) -> u64 {
    u64::from(settings.bitrate_kilobits) * 1000 * 100 / u64::from(100 + FEC_PERCENTAGE)
}

#[cfg(target_os = "linux")]
fn encoder_launch(settings: &StreamSettings) -> String {
    use super::rtsp::VideoCodec;
    let codec = match settings.codec {
        VideoCodec::H264 => "h264",
        VideoCodec::Hevc => "hevc",
    };
    format!(
        "cudascale ! video/x-raw,width={},height={} ! nvenc codec={codec} bitrate={} ref-frames={}",
        settings.width,
        settings.height,
        encoder_bitrate(settings),
        settings.reference_frames
    )
}

#[cfg(target_os = "windows")]
fn encoder_launch(settings: &StreamSettings) -> String {
    use super::rtsp::VideoCodec;
    let element = match settings.codec {
        VideoCodec::H264 => "mfh264enc",
        VideoCodec::Hevc => "mfh265enc",
    };
    format!(
        "d3d11convert ! video/x-raw,width={},height={} ! {element} hardware=true bitrate={}",
        settings.width,
        settings.height,
        encoder_bitrate(settings)
    )
}

fn payloader(settings: &StreamSettings, key: &[u8; 16]) -> Result<MoonlightVideoPay, StreamError> {
    let mut pay = MoonlightVideoPay::default();
    let properties = [
        (
            "packet-size",
            PropValue::Uint(u64::from(settings.packet_size)),
        ),
        ("fec-percentage", PropValue::Uint(u64::from(FEC_PERCENTAGE))),
        (
            "min-required-fec-packets",
            PropValue::Uint(u64::from(settings.minimum_fec_packets)),
        ),
        ("encrypt", PropValue::Bool(settings.encrypt_video)),
        ("key", PropValue::Str(hex::encode(key))),
    ];
    for (name, value) in properties {
        pay.set_property(name, value)
            .map_err(|_| StreamError::Property(name))?;
    }
    Ok(pay)
}

async fn send_video(
    settings: &StreamSettings,
    client: SocketAddr,
    capture: &CaptureFactory,
    pay: MoonlightVideoPay,
) -> Result<(), StreamError> {
    let mut nodes = capture(settings.frames_per_second)
        .map_err(StreamError::Capture)?
        .into_iter();
    let source = nodes.next().expect("capture starts with a source");
    let registry = default_registry();
    let encoder =
        parse_chain(&registry, &encoder_launch(settings)).map_err(StreamError::Encoder)?;
    let mut sink = UdpSink::new(client);
    sink.set_property("bind-port", PropValue::Uint(u64::from(VIDEO_PORT)))
        .map_err(|_| StreamError::Property("bind-port"))?;
    let transforms = nodes
        .chain(encoder.into_iter().map(GraphNodeRef::Element))
        .chain([GraphNodeRef::element(pay)]);
    let graph = registry.splice_domain_converters(linear_graph(
        source,
        transforms,
        GraphNodeRef::element(sink),
    ));
    let clock = WallClock::new();
    eprintln!(
        "g2g-desktop host: streaming {:?} at {} fps to {client}, video encryption {}",
        settings.codec,
        settings.frames_per_second,
        if settings.encrypt_video { "on" } else { "off" }
    );
    RunBuilder::new(graph, &clock, LINK_CAPACITY)
        .run()
        .await
        .map(|_| ())
        .map_err(StreamError::Graph)
}

pub(crate) async fn run(
    sockets: SessionSockets,
    client: IpAddr,
    key: [u8; 16],
    play: oneshot::Receiver<StreamSettings>,
    stop: &Notify,
    capture: &CaptureFactory,
) -> Result<(), StreamError> {
    let SessionSockets {
        control,
        video: video_socket,
    } = sockets;
    let settings = tokio::select! {
        played = play => played.map_err(|_| StreamError::NotPlayed)?,
        () = stop.notified() => return Ok(()),
    };
    let pay = payloader(&settings, &key)?;
    let keyframe_requests = pay.keyframe_requests();
    let video = pin!(async {
        let destination = wait_for_ping(&video_socket, client)
            .await
            .map_err(StreamError::Io)?;
        // udpsink binds the video port itself
        drop(video_socket);
        send_video(&settings, destination, capture, pay).await
    });
    let mut control = pin!(control::run(control, &key, keyframe_requests, stop));
    tokio::select! {
        sent = video => {
            stop.notify_one();
            control.await.map_err(StreamError::Control)?;
            sent
        }
        controlled = &mut control => controlled.map_err(StreamError::Control),
    }
}

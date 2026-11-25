// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::fmt;
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::pin::pin;

use g2g_core::runtime::{
    parse_chain, parse_source, GraphNode, GraphNodeRef, ParseError, RunBuilder,
};
use g2g_core::{AsyncElement, G2gError, PropValue};
use g2g_plugins::clock::WallClock;
use g2g_plugins::moonlight::audiopay::MoonlightAudioPay;
use g2g_plugins::moonlight::videopay::MoonlightVideoPay;
use g2g_plugins::registry::default_registry;
use g2g_plugins::udpsink::UdpSink;
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, oneshot, watch, Notify};

use super::control::{self, InputRoute};
use super::rtsp::StreamSettings;
use crate::audio::CAPTURE as AUDIO_CAPTURE;
use crate::host::{inject, CaptureSize, LINK_CAPACITY};
use crate::injection::{InjectError, InjectEvents, ScreenSize};
use crate::lineargraph::linear_graph;

pub(crate) const VIDEO_PORT: u16 = 47998;
pub(crate) const AUDIO_PORT: u16 = 48000;
// moonlight's legacy video and audio ping, which it sends when SETUP returns no ping payload
const LEGACY_PING: &[u8] = b"PING";
const FEC_PERCENTAGE: u32 = 20;
// moonlight decodes stereo Opus at this rate, the one moonshine sends
const AUDIO_BITRATE: u32 = 96_000;

#[derive(Clone, Copy)]
pub(crate) struct SessionKey {
    pub(crate) key: [u8; 16],
    pub(crate) key_id: u32,
}

impl fmt::Debug for SessionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionKey")
            .field("key_id", &self.key_id)
            .finish_non_exhaustive()
    }
}

pub type CaptureFactory = Box<dyn Fn(u32) -> Result<Vec<GraphNode>, String>>;
pub type InjectorFactory = Box<dyn Fn(ScreenSize) -> Result<Box<dyn InjectEvents>, InjectError>>;

pub struct ServedDesktop {
    pub capture: CaptureFactory,
    pub open_injector: InjectorFactory,
    pub apps: Option<super::ServedApps>,
}

impl fmt::Debug for ServedDesktop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServedDesktop").finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub(crate) enum StreamError {
    Io(io::Error),
    Control(tokio_enet::Error),
    NotPlayed,
    Capture(String),
    Encoder(ParseError),
    Audio(ParseError),
    Property(&'static str),
    Graph(G2gError),
    AudioThreadPanicked,
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Control(e) => write!(f, "control stream: {e}"),
            Self::NotPlayed => f.write_str("the session ended before RTSP PLAY"),
            Self::Capture(e) => write!(f, "capture: {e}"),
            Self::Encoder(e) => write!(f, "encoder: {e}"),
            Self::Audio(e) => write!(f, "audio: {e}"),
            Self::Property(name) => write!(f, "the stream refused {name}"),
            Self::Graph(e) => write!(f, "{e:?}"),
            Self::AudioThreadPanicked => f.write_str("the audio thread panicked"),
        }
    }
}

pub(crate) struct SessionSockets {
    control: tokio_enet::Host,
    video: UdpSocket,
    audio: UdpSocket,
}

impl fmt::Debug for SessionSockets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionSockets")
            .field("video", &self.video)
            .field("audio", &self.audio)
            .finish_non_exhaustive()
    }
}

pub(crate) async fn bind() -> Result<SessionSockets, StreamError> {
    Ok(SessionSockets {
        control: control::bind().map_err(StreamError::Control)?,
        video: UdpSocket::bind((Ipv4Addr::UNSPECIFIED, VIDEO_PORT))
            .await
            .map_err(StreamError::Io)?,
        audio: UdpSocket::bind((Ipv4Addr::UNSPECIFIED, AUDIO_PORT))
            .await
            .map_err(StreamError::Io)?,
    })
}

async fn wait_for_ping(socket: &UdpSocket, client: IpAddr) -> io::Result<SocketAddr> {
    let mut buffer = [0u8; 64];
    loop {
        let (length, from) = socket.recv_from(&mut buffer).await?;
        if from.ip() == client && &buffer[..length] == LEGACY_PING {
            return Ok(from);
        }
    }
}

// moonlight's bitrate covers the stream with its parity
fn encoder_bitrate(settings: &StreamSettings) -> u64 {
    u64::from(settings.bitrate_kilobits) * 1000 * 100 / u64::from(100 + FEC_PERCENTAGE)
}

// h264 and hevc keep at most 16, the most frames an invalidation can reach back over
#[cfg(target_os = "linux")]
const HOST_CHOSEN_REFERENCE_FRAMES: u32 = 16;

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
        match settings.reference_frames {
            0 => HOST_CHOSEN_REFERENCE_FRAMES,
            requested => requested,
        }
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
    sizes: watch::Sender<Option<ScreenSize>>,
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
        .chain([GraphNodeRef::element(CaptureSize { sizes })])
        .chain(encoder.into_iter().map(GraphNodeRef::Element))
        .chain([GraphNodeRef::element(pay)]);
    let graph = registry.splice_domain_converters(linear_graph(
        source,
        transforms,
        GraphNodeRef::element(sink),
    ));
    let clock = WallClock::new();
    eprintln!(
        "g2g-desktop host: streaming {:?} at {} fps to {client}, video encryption {}, audio encryption {}",
        settings.codec,
        settings.frames_per_second,
        if settings.encrypt_video { "on" } else { "off" },
        if settings.encrypt_audio { "on" } else { "off" }
    );
    RunBuilder::new(graph, &clock, LINK_CAPACITY)
        .run()
        .await
        .map(|_| ())
        .map_err(StreamError::Graph)
}

fn audio_payloader(
    settings: &StreamSettings,
    key: SessionKey,
) -> Result<MoonlightAudioPay, StreamError> {
    let mut pay = MoonlightAudioPay::default();
    let properties = [
        (
            "packet-duration",
            PropValue::Uint(u64::from(settings.audio_packet_duration_ms)),
        ),
        ("encrypt", PropValue::Bool(settings.encrypt_audio)),
        ("key", PropValue::Str(hex::encode(key.key))),
        ("key-id", PropValue::Uint(u64::from(key.key_id))),
    ];
    for (name, value) in properties {
        pay.set_property(name, value)
            .map_err(|_| StreamError::Property(name))?;
    }
    Ok(pay)
}

async fn send_audio(
    settings: &StreamSettings,
    client: SocketAddr,
    key: SessionKey,
) -> Result<(), StreamError> {
    let registry = default_registry();
    let source = parse_source(&registry, AUDIO_CAPTURE).map_err(StreamError::Audio)?;
    let encoder = parse_chain(
        &registry,
        &format!(
            "audioconvert ! opusenc frame-size={} bitrate={AUDIO_BITRATE} bitrate-type=cbr \
             audio-type=restricted-lowdelay",
            settings.audio_packet_duration_ms
        ),
    )
    .map_err(StreamError::Audio)?;
    let mut sink = UdpSink::new(client);
    sink.set_property("bind-port", PropValue::Uint(u64::from(AUDIO_PORT)))
        .map_err(|_| StreamError::Property("bind-port"))?;
    let transforms = encoder
        .into_iter()
        .map(GraphNodeRef::Element)
        .chain([GraphNodeRef::element(audio_payloader(settings, key)?)]);
    let graph = linear_graph(
        GraphNodeRef::Source(source),
        transforms,
        GraphNodeRef::element(sink),
    );
    let clock = WallClock::new();
    RunBuilder::new(graph, &clock, LINK_CAPACITY)
        .run()
        .await
        .map(|_| ())
        .map_err(StreamError::Graph)
}

// audio on the video graph's thread stops draining the capture while a video element blocks
async fn send_audio_on_own_thread(
    settings: StreamSettings,
    client: SocketAddr,
    key: SessionKey,
) -> Result<(), StreamError> {
    let (_stop, stopped) = oneshot::channel::<()>();
    let (finished, sent) = oneshot::channel();
    std::thread::Builder::new()
        .name("g2g-desktop-gamestream-audio".into())
        .spawn(move || {
            let sent = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(StreamError::Io)
                .and_then(|runtime| {
                    runtime.block_on(async {
                        tokio::select! {
                            sent = send_audio(&settings, client, key) => sent,
                            _ = stopped => Ok(()),
                        }
                    })
                });
            drop(finished.send(sent));
        })
        .map_err(StreamError::Io)?;
    sent.await.map_err(|_| StreamError::AudioThreadPanicked)?
}

pub(crate) async fn run(
    sockets: SessionSockets,
    client: IpAddr,
    key: SessionKey,
    play: oneshot::Receiver<StreamSettings>,
    stop: &Notify,
    desktop: &ServedDesktop,
) -> Result<(), StreamError> {
    let SessionSockets {
        control,
        video: video_socket,
        audio: audio_socket,
    } = sockets;
    let settings = tokio::select! {
        played = play => played.map_err(|_| StreamError::NotPlayed)?,
        () = stop.notified() => return Ok(()),
    };
    let pay = payloader(&settings, &key.key)?;
    let recovery = pay.recovery_requests();
    let (events, event_receiver) = mpsc::unbounded_channel();
    let (sizes, size_receiver) = watch::channel(None);
    let input = InputRoute {
        events,
        sizes: size_receiver.clone(),
    };
    let audio = async {
        let sent = match wait_for_ping(&audio_socket, client).await {
            Ok(destination) => {
                // udpsink binds the audio port itself
                drop(audio_socket);
                send_audio_on_own_thread(settings, destination, key).await
            }
            Err(e) => Err(StreamError::Io(e)),
        };
        match sent {
            Ok(()) => eprintln!("g2g-desktop host: Moonlight audio capture ended"),
            Err(e) => eprintln!("g2g-desktop host: Moonlight audio stopped: {e}"),
        }
        std::future::pending::<()>().await
    };
    let video = pin!(async {
        let destination = wait_for_ping(&video_socket, client)
            .await
            .map_err(StreamError::Io)?;
        // udpsink binds the video port itself
        drop(video_socket);
        tokio::select! {
            sent = send_video(&settings, destination, &desktop.capture, pay, sizes) => sent,
            () = audio => unreachable!("the audio future never completes"),
        }
    });
    let mut control = pin!(control::run(control, &key.key, recovery, input, stop));
    let session = async {
        tokio::select! {
            sent = video => {
                stop.notify_one();
                control.await.map_err(StreamError::Control)?;
                sent
            }
            controlled = &mut control => controlled.map_err(StreamError::Control),
        }
    };
    let (ended, injected) = tokio::join!(
        session,
        inject(event_receiver, size_receiver, &desktop.open_injector)
    );
    if let Err(e) = injected {
        eprintln!("g2g-desktop host: Moonlight input stopped: {e}");
    }
    ended
}

// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use core::time::Duration;

use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_rustls::{TlsAcceptor, TlsConnector};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;
use web_transport_quinn::quinn::rustls;
use web_transport_quinn::quinn::rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};

use g2g_core::g2g_warn;
use g2g_core::log::Target;
use g2g_core::wire::{decode_packet, encode_packet};
use g2g_core::{G2gError, NetworkFailure, PipelinePacket};

use crate::abortondrop::AbortOnDrop;
use crate::datachannel::{DataChannelTransportEnds, MAX_MESSAGE_SIZE};
use crate::remotewire::map_wire;
use crate::remotewtio::{wt_err, SharedDataChannel};
use crate::remotewttrack::{deliver_track_packet, SharedTrackOutgoing};

pub(crate) const WEBSOCKET_SCHEME: &str = "wss";

const VIDEO_TAG: u8 = 0;
const MESSAGE_TAG: u8 = 1;
const KEYFRAME_REQUEST_TAG: u8 = 2;
const TRACK_TAG: u8 = 3;

const HTTP_1_1_ALPN: &[u8] = b"http/1.1";
const QUEUED_VIDEO_PACKETS: usize = 1;
const RECEIVED_VIDEO_PACKETS: usize = 2;

pub(crate) type TlsWebSocket<S> = WebSocketStream<tokio_rustls::TlsStream<S>>;

pub(crate) fn server_tls(
    chain: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
) -> Result<Arc<rustls::ServerConfig>, G2gError> {
    let mut tls = rustls::ServerConfig::builder_with_provider(
        web_transport_quinn::crypto::default_provider(),
    )
    .with_protocol_versions(&[&rustls::version::TLS13])
    .map_err(wt_err)?
    .with_no_client_auth()
    .with_single_cert(chain, key)
    .map_err(wt_err)?;
    tls.alpn_protocols = vec![HTTP_1_1_ALPN.to_vec()];
    Ok(Arc::new(tls))
}

pub(crate) async fn accept(
    listener: &TcpListener,
    tls: &Arc<rustls::ServerConfig>,
) -> Result<TlsWebSocket<TcpStream>, G2gError> {
    let (tcp, _) = listener.accept().await.map_err(wt_err)?;
    tcp.set_nodelay(true).map_err(wt_err)?;
    let tls_stream = TlsAcceptor::from(tls.clone())
        .accept(tcp)
        .await
        .map_err(wt_err)?;
    tokio_tungstenite::accept_async(tokio_rustls::TlsStream::Server(tls_stream))
        .await
        .map_err(wt_err)
}

pub(crate) async fn dial(
    url: &url::Url,
    mut tls: rustls::ClientConfig,
) -> Result<TlsWebSocket<TcpStream>, G2gError> {
    let host = crate::remotewtio::url_host(url)?;
    let port = url.port().ok_or(G2gError::NotConfigured)?;
    let address: SocketAddr = tokio::net::lookup_host((host, port))
        .await
        .map_err(wt_err)?
        .next()
        .ok_or(G2gError::Network(NetworkFailure::Connect))?;
    let tcp = TcpStream::connect(address).await.map_err(wt_err)?;
    tcp.set_nodelay(true).map_err(wt_err)?;
    tls.alpn_protocols = vec![HTTP_1_1_ALPN.to_vec()];
    let server_name = ServerName::try_from(host.to_string()).map_err(wt_err)?;
    let tls_stream = TlsConnector::from(Arc::new(tls))
        .connect(server_name, tcp)
        .await
        .map_err(wt_err)?;
    let (websocket, _) =
        tokio_tungstenite::client_async(url.as_str(), tokio_rustls::TlsStream::Client(tls_stream))
            .await
            .map_err(wt_err)?;
    Ok(websocket)
}

fn tagged(tag: u8, payload: &[u8]) -> Message {
    let mut bytes = Vec::with_capacity(1 + payload.len());
    bytes.push(tag);
    bytes.extend_from_slice(payload);
    Message::Binary(bytes)
}

#[derive(Debug)]
pub struct WsLink {
    video: Option<mpsc::Sender<Vec<u8>>>,
    keyframe_requests: mpsc::UnboundedSender<Message>,
    received_video: mpsc::Receiver<Vec<u8>>,
    keyframe_requested: Arc<AtomicBool>,
    delta_dropped: bool,
    dropping_until_keyframe: bool,
    writer: Option<JoinHandle<()>>,
    _reader: AbortOnDrop,
}

impl WsLink {
    pub(crate) async fn new<S>(
        websocket: TlsWebSocket<S>,
        data_channel: Option<&SharedDataChannel>,
        track_outgoing: Option<&SharedTrackOutgoing>,
        track_incoming: Option<&mpsc::Sender<PipelinePacket>>,
    ) -> Self
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let (socket_sink, socket_stream) = websocket.split();
        let (video, video_out) = mpsc::channel(QUEUED_VIDEO_PACKETS);
        let (keyframe_requests, requests_out) = mpsc::unbounded_channel();
        let (received_video_in, received_video) = mpsc::channel(RECEIVED_VIDEO_PACKETS);
        let keyframe_requested = Arc::new(AtomicBool::new(false));
        let messages_in = match data_channel {
            Some(channel) => Some(channel.lock().await.incoming.clone()),
            None => None,
        };
        let writer = tokio::spawn(write_socket(
            socket_sink,
            video_out,
            requests_out,
            data_channel.cloned(),
            track_outgoing.cloned(),
        ));
        let reader = tokio::spawn(read_socket(
            socket_stream,
            received_video_in,
            messages_in,
            track_incoming.cloned(),
            keyframe_requested.clone(),
        ));
        Self {
            video: Some(video),
            keyframe_requests,
            received_video,
            keyframe_requested,
            delta_dropped: false,
            dropping_until_keyframe: false,
            writer: Some(writer),
            _reader: AbortOnDrop(reader),
        }
    }

    pub(crate) async fn send(&mut self, packet: &PipelinePacket) -> Result<(), G2gError> {
        let body = encode_packet(packet).map_err(map_wire)?;
        let video = self.video.as_ref().ok_or(G2gError::NotConfigured)?;
        let keyframe = match packet {
            PipelinePacket::DataFrame(frame) => frame.timing.keyframe,
            _ => return video.send(body).await.map_err(wt_err),
        };
        if keyframe {
            self.dropping_until_keyframe = false;
            return video.send(body).await.map_err(wt_err);
        }
        if self.dropping_until_keyframe {
            return Ok(());
        }
        match video.try_send(body) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                // later delta frames reference this one
                self.dropping_until_keyframe = true;
                self.delta_dropped = true;
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err(G2gError::Network(NetworkFailure::Stream))
            }
        }
    }

    pub(crate) fn take_keyframe_request(&mut self) -> bool {
        let requested = self.keyframe_requested.swap(false, Ordering::Relaxed);
        core::mem::take(&mut self.delta_dropped) || requested
    }

    pub(crate) fn request_keyframe(&mut self) -> Result<(), G2gError> {
        self.keyframe_requests
            .send(tagged(KEYFRAME_REQUEST_TAG, &[]))
            .map_err(wt_err)
    }

    pub(crate) async fn recv(&mut self) -> Result<Option<PipelinePacket>, G2gError> {
        match self.received_video.recv().await {
            Some(body) => Ok(Some(decode_packet(&body).map_err(map_wire)?)),
            None => Ok(None),
        }
    }

    pub(crate) async fn finish(&mut self, flush_timeout: Duration) {
        self.video = None;
        if let Some(writer) = self.writer.take() {
            let _ = tokio::time::timeout(flush_timeout, writer).await;
        }
    }
}

async fn write_socket<S>(
    mut socket: futures_util::stream::SplitSink<TlsWebSocket<S>, Message>,
    mut video: mpsc::Receiver<Vec<u8>>,
    mut requests: mpsc::UnboundedReceiver<Message>,
    data_channel: Option<SharedDataChannel>,
    track: Option<SharedTrackOutgoing>,
) where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut ends = match &data_channel {
        Some(channel) => Some(channel.lock().await),
        None => None,
    };
    let mut messages_open = ends.is_some();
    let mut track = match &track {
        Some(track) => Some(track.lock().await),
        None => None,
    };
    let mut track_open = track.is_some();
    loop {
        let message = tokio::select! {
            biased;
            Some(request) = requests.recv() => request,
            message = async {
                let DataChannelTransportEnds { outgoing, .. } = &mut **ends.as_mut()?;
                outgoing.recv().await
            }, if messages_open => match message {
                Some(message) => tagged(MESSAGE_TAG, &message),
                None => {
                    messages_open = false;
                    continue;
                }
            },
            body = async { track.as_mut()?.recv().await }, if track_open => match body {
                Some(body) => tagged(TRACK_TAG, &body),
                None => {
                    track_open = false;
                    continue;
                }
            },
            body = video.recv() => match body {
                Some(body) => tagged(VIDEO_TAG, &body),
                None => {
                    let _ = socket.close().await;
                    return;
                }
            },
        };
        if socket.send(message).await.is_err() {
            return;
        }
    }
}

async fn read_socket<S>(
    mut socket: futures_util::stream::SplitStream<TlsWebSocket<S>>,
    video: mpsc::Sender<Vec<u8>>,
    messages: Option<mpsc::UnboundedSender<Vec<u8>>>,
    track: Option<mpsc::Sender<PipelinePacket>>,
    keyframe_requested: Arc<AtomicBool>,
) where
    S: AsyncRead + AsyncWrite + Unpin,
{
    loop {
        let bytes = match socket.next().await {
            None | Some(Ok(Message::Close(_))) => return,
            Some(Ok(Message::Binary(bytes))) => bytes,
            Some(Ok(_)) => continue,
            Some(Err(_)) => break,
        };
        let Some((&tag, payload)) = bytes.split_first() else {
            break;
        };
        match tag {
            VIDEO_TAG => {
                if video.send(payload.to_vec()).await.is_err() {
                    return;
                }
            }
            MESSAGE_TAG if payload.len() <= MAX_MESSAGE_SIZE => {
                if let Some(messages) = &messages {
                    let _ = messages.send(payload.to_vec());
                }
            }
            KEYFRAME_REQUEST_TAG => keyframe_requested.store(true, Ordering::Relaxed),
            TRACK_TAG => {
                let Ok(packet) = decode_packet(payload) else {
                    break;
                };
                if let Some(track) = &track {
                    if deliver_track_packet(track, packet).await.is_err() {
                        return;
                    }
                }
            }
            _ => break,
        }
    }
    g2g_warn!(
        Target::category("remotewt"),
        "websocket closed on a read error or a malformed message"
    );
}

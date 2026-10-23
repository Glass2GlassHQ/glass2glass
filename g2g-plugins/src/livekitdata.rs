use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use core::time::Duration;

use std::net::SocketAddr;
use std::time::Instant;

use futures_util::StreamExt;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use str0m::channel::{ChannelConfig, ChannelId, Reliability};
use str0m::crypto::from_feature_flags;
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

use g2g_core::log::LogSource;
use g2g_core::{g2g_warn, G2gError, HardwareError, NetworkFailure};

use crate::filesink::io_err;
use crate::livekit_signal::{
    candidate_from_init_json, ping_timestamp, recv_signal, send_signal, DataPacketKind,
    IceSettings, RoomSettings, SessionDescription, SignalRequest, SignalResponse, SignalSocket,
    SignalTarget, VideoGrant,
};
use crate::turn::TurnSet;
use crate::webrtc_receive::UDP_RECV_BYTES;
use crate::webrtc_util::{
    accept_answer_sdp, connect_failed, feed_datagram, select_host_ip, send_transmit,
};

const RELIABLE_LABEL: &str = "_reliable";
const LOSSY_LABEL: &str = "_lossy";

const TOKEN_TTL_SECS: u64 = 3600;

#[derive(Debug)]
pub(crate) struct ChannelMessage {
    pub kind: DataPacketKind,
    pub bytes: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct Drained {
    pub messages: Vec<ChannelMessage>,
    pub ended: bool,
    pub deadline: Instant,
}

#[derive(Debug)]
pub(crate) enum Wake {
    Network,
    // None once the queue's senders are gone
    Outgoing(Option<Vec<u8>>),
    // the server closed the signalling socket or sent a Leave
    Closed,
}

pub(crate) struct DataSession {
    rtc: Rtc,
    socket: UdpSocket,
    local: SocketAddr,
    turn: TurnSet,
    turn_refresh_at: Instant,
    ws: SignalSocket,
    ping_period: Duration,
    next_ping: Instant,
    reliable: ChannelId,
    lossy: ChannelId,
    receive_buffer: Vec<u8>,
}

impl core::fmt::Debug for DataSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DataSession")
            .field("local", &self.local)
            .field("reliable", &self.reliable)
            .field("lossy", &self.lossy)
            .finish()
    }
}

impl DataSession {
    pub(crate) async fn open(
        room: &RoomSettings,
        ice: &IceSettings,
        grant: &VideoGrant,
        log: &dyn LogSource,
    ) -> Result<Self, G2gError> {
        let (mut ws, join) = room.join(grant, TOKEN_TTL_SECS, false, log).await?;

        let socket = UdpSocket::bind((select_host_ip(), 0))
            .await
            .map_err(io_err)?;
        let local = socket.local_addr().map_err(io_err)?;
        let mut rtc = RtcConfig::new()
            .set_crypto_provider(Arc::new(from_feature_flags()))
            .build(Instant::now());
        let turn = ice.gather(&mut rtc, &socket, &join, log).await?;

        // with canSubscribe false the server sends data back down these channels
        let (offer, pending, reliable, lossy) = {
            let mut api = rtc.sdp_api();
            let reliable = api.add_channel_with_config(ChannelConfig {
                label: String::from(RELIABLE_LABEL),
                ordered: true,
                reliability: Reliability::Reliable,
                ..Default::default()
            });
            let lossy = api.add_channel_with_config(ChannelConfig {
                label: String::from(LOSSY_LABEL),
                ordered: false,
                reliability: Reliability::MaxRetransmits { retransmits: 0 },
                ..Default::default()
            });
            let (offer, pending) = api
                .apply()
                .ok_or(G2gError::Hardware(HardwareError::Other))?;
            (offer, pending, reliable, lossy)
        };
        send_signal(
            &mut ws,
            &SignalRequest::Offer(SessionDescription {
                sdp_type: "offer".into(),
                sdp: offer.to_sdp_string(),
            }),
        )
        .await?;

        let mut early_candidates = Vec::new();
        let answer = loop {
            match recv_signal(&mut ws).await? {
                Some(SignalResponse::Answer(answer)) => break answer,
                Some(SignalResponse::Trickle(trickle))
                    if trickle.target == SignalTarget::Publisher =>
                {
                    early_candidates.push(trickle.candidate_init)
                }
                Some(SignalResponse::Leave) | None => {
                    g2g_warn!(log, "closed before publisher answer");
                    return Err(G2gError::Network(NetworkFailure::Connect));
                }
                Some(_) => {}
            }
        };
        accept_answer_sdp(&mut rtc, pending, &answer.sdp)
            .map_err(|reason| connect_failed(log, &alloc::format!("publisher {reason}")))?;
        for candidate_init in &early_candidates {
            add_remote_candidate(&mut rtc, candidate_init);
        }

        let ping_period = join.ping_period();
        Ok(Self {
            rtc,
            socket,
            local,
            turn,
            turn_refresh_at: Instant::now() + crate::turn::REFRESH_INTERVAL,
            ws,
            ping_period,
            next_ping: Instant::now() + ping_period,
            reliable,
            lossy,
            receive_buffer: alloc::vec![0u8; UDP_RECV_BYTES],
        })
    }

    fn channel_id(&self, kind: DataPacketKind) -> ChannelId {
        match kind {
            DataPacketKind::Reliable => self.reliable,
            DataPacketKind::Lossy => self.lossy,
        }
    }

    pub(crate) async fn drain(&mut self) -> Drained {
        let mut messages = Vec::new();
        loop {
            match self.rtc.poll_output() {
                Ok(Output::Timeout(deadline)) => {
                    return Drained {
                        messages,
                        ended: false,
                        deadline,
                    }
                }
                Ok(Output::Transmit(transmit)) => {
                    send_transmit(&self.socket, &mut self.turn, &transmit).await
                }
                Ok(Output::Event(Event::ChannelData(data))) => {
                    let kind = if data.id == self.reliable {
                        DataPacketKind::Reliable
                    } else if data.id == self.lossy {
                        DataPacketKind::Lossy
                    } else {
                        continue;
                    };
                    messages.push(ChannelMessage {
                        kind,
                        bytes: data.data,
                    });
                }
                Ok(Output::Event(Event::ChannelClose(_)))
                | Ok(Output::Event(Event::IceConnectionStateChange(
                    IceConnectionState::Disconnected,
                )))
                | Err(_) => {
                    return Drained {
                        messages,
                        ended: true,
                        deadline: Instant::now(),
                    }
                }
                Ok(Output::Event(_)) => {}
            }
        }
    }

    pub(crate) async fn wait(
        &mut self,
        deadline: Instant,
        outgoing: Option<&mut mpsc::Receiver<Vec<u8>>>,
    ) -> Wake {
        let outgoing_message = async {
            match outgoing {
                Some(receiver) => receiver.recv().await,
                None => core::future::pending().await,
            }
        };
        let timeout = deadline.saturating_duration_since(Instant::now());
        tokio::select! {
            message = outgoing_message => Wake::Outgoing(message),
            received = self.socket.recv_from(&mut self.receive_buffer) => {
                let Ok((length, source)) = received else {
                    return Wake::Closed;
                };
                feed_datagram(
                    &mut self.rtc,
                    &mut self.turn,
                    self.local,
                    &self.receive_buffer[..length],
                    source,
                );
                Wake::Network
            }
            message = self.ws.next() => match message {
                Some(Ok(Message::Binary(bytes))) => match SignalResponse::decode(&bytes) {
                    Some(SignalResponse::Trickle(trickle))
                        if trickle.target == SignalTarget::Publisher =>
                    {
                        add_remote_candidate(&mut self.rtc, &trickle.candidate_init);
                        Wake::Network
                    }
                    Some(SignalResponse::Leave) => Wake::Closed,
                    _ => Wake::Network,
                },
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => Wake::Closed,
                Some(Ok(_)) => Wake::Network,
            },
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(self.next_ping)) => {
                let _ = send_signal(&mut self.ws, &SignalRequest::Ping(ping_timestamp())).await;
                self.next_ping = Instant::now() + self.ping_period;
                Wake::Network
            }
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(self.turn_refresh_at)), if !self.turn.is_empty() => {
                self.turn.refresh_all(&self.socket).await;
                self.turn_refresh_at = Instant::now() + crate::turn::REFRESH_INTERVAL;
                Wake::Network
            }
            _ = tokio::time::sleep(timeout) => {
                if self.rtc.handle_input(Input::Timeout(Instant::now())).is_err() {
                    return Wake::Closed;
                }
                Wake::Network
            }
        }
    }

    pub(crate) fn is_open(&mut self, kind: DataPacketKind) -> bool {
        let id = self.channel_id(kind);
        self.rtc.channel(id).is_some()
    }

    // Ok(false) when the channel is not open yet or its send buffer is full
    pub(crate) fn write(&mut self, kind: DataPacketKind, bytes: &[u8]) -> Result<bool, G2gError> {
        let id = self.channel_id(kind);
        let Some(mut channel) = self.rtc.channel(id) else {
            return Ok(false);
        };
        channel
            .write(true, bytes)
            .map_err(|_| G2gError::Network(NetworkFailure::Stream))
    }

    // bytes written that the server has not acknowledged yet
    pub(crate) fn buffered_amount(&mut self, kind: DataPacketKind) -> usize {
        let id = self.channel_id(kind);
        self.rtc
            .channel(id)
            .map(|mut channel| channel.buffered_amount())
            .unwrap_or(0)
    }

    pub(crate) async fn leave(mut self) {
        let _ = send_signal(&mut self.ws, &SignalRequest::Leave).await;
        let _ = self.ws.close(None).await;
    }
}

fn add_remote_candidate(rtc: &mut Rtc, candidate_init: &str) {
    let Some(candidate) = candidate_from_init_json(candidate_init) else {
        return;
    };
    if let Ok(candidate) = Candidate::from_sdp_string(&candidate) {
        rtc.add_remote_candidate(candidate);
    }
}

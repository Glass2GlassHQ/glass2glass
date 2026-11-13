// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use subtle::ConstantTimeEq;

const LENGTH_PREFIX_LEN: usize = 4;
const MAX_TOKEN_LEN: usize = 256;
const MAX_SDP_LEN: usize = 64 * 1024;
const ACCEPT: u8 = 0x01;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignalingError {
    TokenLength(usize),
    BadToken,
    MessageTooLong { len: usize, max: usize },
    NotUtf8,
    BadAccept,
    UnexpectedMessage,
    OfferBeforeAccept,
    AnswerBeforeOffer,
    HungUp,
    Busy,
}

impl fmt::Display for SignalingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignalingError::TokenLength(len) => {
                write!(f, "token is {len} bytes, must be 1 to {MAX_TOKEN_LEN}")
            }
            SignalingError::BadToken => write!(f, "token does not match"),
            SignalingError::MessageTooLong { len, max } => {
                write!(f, "{len} byte signaling message exceeds {max} bytes")
            }
            SignalingError::NotUtf8 => write!(f, "SDP is not UTF-8"),
            SignalingError::BadAccept => write!(f, "host accept message is not the byte 0x01"),
            SignalingError::UnexpectedMessage => {
                write!(
                    f,
                    "peer sent a message after the SDP exchange step it was due"
                )
            }
            SignalingError::OfferBeforeAccept => {
                write!(f, "offer sent before the host accepted the token")
            }
            SignalingError::AnswerBeforeOffer => write!(f, "answer sent before an offer arrived"),
            SignalingError::HungUp => write!(f, "connection closed before the SDP answer"),
            SignalingError::Busy => write!(f, "a session is already running"),
        }
    }
}

impl std::error::Error for SignalingError {}

fn check_token_len(token: &[u8]) -> Result<(), SignalingError> {
    if token.is_empty() || token.len() > MAX_TOKEN_LEN {
        return Err(SignalingError::TokenLength(token.len()));
    }
    Ok(())
}

fn frame(payload: &[u8]) -> Vec<u8> {
    let len = u32::try_from(payload.len()).expect("signaling payloads are bounded by MAX_SDP_LEN");
    let mut framed = Vec::with_capacity(LENGTH_PREFIX_LEN + payload.len());
    framed.extend_from_slice(&len.to_be_bytes());
    framed.extend_from_slice(payload);
    framed
}

fn frame_sdp(sdp: &str) -> Result<Vec<u8>, SignalingError> {
    if sdp.len() > MAX_SDP_LEN {
        return Err(SignalingError::MessageTooLong {
            len: sdp.len(),
            max: MAX_SDP_LEN,
        });
    }
    Ok(frame(sdp.as_bytes()))
}

#[derive(Debug, Default)]
struct Unframer {
    buffer: Vec<u8>,
}

impl Unframer {
    fn next(&mut self, max_len: usize) -> Result<Option<Vec<u8>>, SignalingError> {
        let Some(prefix) = self.buffer.first_chunk::<LENGTH_PREFIX_LEN>() else {
            return Ok(None);
        };
        let len = u32::from_be_bytes(*prefix) as usize;
        if len > max_len {
            return Err(SignalingError::MessageTooLong { len, max: max_len });
        }
        if self.buffer.len() < LENGTH_PREFIX_LEN + len {
            return Ok(None);
        }
        let payload = self.buffer[LENGTH_PREFIX_LEN..LENGTH_PREFIX_LEN + len].to_vec();
        self.buffer.drain(..LENGTH_PREFIX_LEN + len);
        Ok(Some(payload))
    }
}

fn sdp_from(payload: Vec<u8>) -> Result<String, SignalingError> {
    String::from_utf8(payload).map_err(|_| SignalingError::NotUtf8)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostState {
    AwaitingToken,
    AwaitingOffer,
    AwaitingAnswer,
    Answered,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostEvent {
    Reply(Vec<u8>),
    Offer(String),
}

#[derive(Debug)]
pub struct HostSignaling {
    token: Vec<u8>,
    state: HostState,
    unframer: Unframer,
}

impl HostSignaling {
    pub fn new(token: &str) -> Result<Self, SignalingError> {
        check_token_len(token.as_bytes())?;
        Ok(Self {
            token: token.as_bytes().to_vec(),
            state: HostState::AwaitingToken,
            unframer: Unframer::default(),
        })
    }

    pub fn receive(&mut self, bytes: &[u8]) -> Result<Vec<HostEvent>, SignalingError> {
        self.unframer.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        loop {
            if matches!(self.state, HostState::AwaitingAnswer | HostState::Answered) {
                if !self.unframer.buffer.is_empty() {
                    return Err(SignalingError::UnexpectedMessage);
                }
                return Ok(events);
            }
            let max_len = match self.state {
                HostState::AwaitingToken => MAX_TOKEN_LEN,
                _ => MAX_SDP_LEN,
            };
            let Some(payload) = self.unframer.next(max_len)? else {
                return Ok(events);
            };
            if self.state == HostState::AwaitingToken {
                if !bool::from(self.token.ct_eq(&payload)) {
                    return Err(SignalingError::BadToken);
                }
                self.state = HostState::AwaitingOffer;
                events.push(HostEvent::Reply(frame(&[ACCEPT])));
            } else {
                self.state = HostState::AwaitingAnswer;
                events.push(HostEvent::Offer(sdp_from(payload)?));
            }
        }
    }

    pub fn answer(&mut self, sdp: &str) -> Result<Vec<u8>, SignalingError> {
        if self.state != HostState::AwaitingAnswer {
            return Err(SignalingError::AnswerBeforeOffer);
        }
        let framed = frame_sdp(sdp)?;
        self.state = HostState::Answered;
        Ok(framed)
    }

    pub fn connection_closed(self) -> Result<(), SignalingError> {
        if self.state != HostState::Answered {
            return Err(SignalingError::HungUp);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClientState {
    AwaitingAccept,
    Accepted,
    AwaitingAnswer,
    Answered,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientEvent {
    Accepted,
    Answer(String),
}

#[derive(Debug)]
pub struct ClientSignaling {
    state: ClientState,
    unframer: Unframer,
}

impl ClientSignaling {
    pub fn new(token: &str) -> Result<(Self, Vec<u8>), SignalingError> {
        check_token_len(token.as_bytes())?;
        let signaling = Self {
            state: ClientState::AwaitingAccept,
            unframer: Unframer::default(),
        };
        Ok((signaling, frame(token.as_bytes())))
    }

    pub fn receive(&mut self, bytes: &[u8]) -> Result<Vec<ClientEvent>, SignalingError> {
        self.unframer.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        loop {
            if matches!(self.state, ClientState::Accepted | ClientState::Answered) {
                if !self.unframer.buffer.is_empty() {
                    return Err(SignalingError::UnexpectedMessage);
                }
                return Ok(events);
            }
            let max_len = match self.state {
                ClientState::AwaitingAccept => 1,
                _ => MAX_SDP_LEN,
            };
            let Some(payload) = self.unframer.next(max_len)? else {
                return Ok(events);
            };
            if self.state == ClientState::AwaitingAccept {
                if payload != [ACCEPT] {
                    return Err(SignalingError::BadAccept);
                }
                self.state = ClientState::Accepted;
                events.push(ClientEvent::Accepted);
            } else {
                self.state = ClientState::Answered;
                events.push(ClientEvent::Answer(sdp_from(payload)?));
            }
        }
    }

    pub fn offer(&mut self, sdp: &str) -> Result<Vec<u8>, SignalingError> {
        if self.state != ClientState::Accepted {
            return Err(SignalingError::OfferBeforeAccept);
        }
        let framed = frame_sdp(sdp)?;
        self.state = ClientState::AwaitingAnswer;
        Ok(framed)
    }

    pub fn connection_closed(self) -> Result<(), SignalingError> {
        if self.state != ClientState::Answered {
            return Err(SignalingError::HungUp);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct SessionSlot {
    taken: Arc<AtomicBool>,
}

#[derive(Debug)]
pub struct SessionGuard {
    taken: Arc<AtomicBool>,
}

impl SessionSlot {
    pub fn claim(&self) -> Result<SessionGuard, SignalingError> {
        if self.taken.swap(true, Ordering::AcqRel) {
            return Err(SignalingError::Busy);
        }
        Ok(SessionGuard {
            taken: Arc::clone(&self.taken),
        })
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.taken.store(false, Ordering::Release);
    }
}

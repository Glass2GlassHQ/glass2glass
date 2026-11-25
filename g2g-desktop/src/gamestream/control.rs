// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use g2g_core::log::Target;
use ring::aead::{Aad, LessSafeKey, Nonce, Tag, UnboundKey, AES_128_GCM};
use tokio::sync::{mpsc, watch, Notify};
use tokio::time::Instant;
use tokio_enet::{Event, Host, HostConfig, Packet, PacketMode, PeerId};

use super::input::input_event;
use super::LOG_CATEGORY;
use crate::injection::ScreenSize;
use crate::packet::{self, RemovableDevice};

pub(crate) const CONTROL_PORT: u16 = 47999;
// moonlight opens this many ENet channels
const CHANNEL_LIMIT: usize = 0x30;
const MESSAGE_ENCRYPTED: u16 = 0x0001;
const MESSAGE_TERMINATION: u16 = 0x0109;
const MESSAGE_START_B: u16 = 0x0307;
const MESSAGE_REQUEST_IDR: u16 = 0x0302;
const MESSAGE_INPUT: u16 = 0x0206;
// moonlight closes the stream without an error dialog for this reason
const TERMINATION_GRACEFUL: u32 = 0x8003_0023;
const SILENCE_LIMIT: Duration = Duration::from_secs(10);
const HEADER_BYTES: usize = 4;
const SEQUENCE_BYTES: usize = 4;
const TAG_BYTES: usize = 16;
const ENCRYPTED_PREFIX_BYTES: usize = HEADER_BYTES + SEQUENCE_BYTES + TAG_BYTES;
// moonlight marks the last two nonce bytes with who sent the message
const CLIENT_NONCE_MARKER: [u8; 2] = *b"CC";
const HOST_NONCE_MARKER: [u8; 2] = *b"HC";

pub(crate) fn bind() -> Result<Host, tokio_enet::Error> {
    Host::new(HostConfig {
        address: Some(SocketAddr::from((Ipv4Addr::UNSPECIFIED, CONTROL_PORT))),
        peer_count: 1,
        channel_limit: CHANNEL_LIMIT,
        ..HostConfig::default()
    })
}

fn nonce(sequence: u32, marker: [u8; 2]) -> Nonce {
    let mut bytes = [0u8; 12];
    bytes[..SEQUENCE_BYTES].copy_from_slice(&sequence.to_le_bytes());
    bytes[10..].copy_from_slice(&marker);
    Nonce::assume_unique_for_key(bytes)
}

fn message_type(message: &[u8]) -> Option<u16> {
    let header = message.get(..HEADER_BYTES)?;
    let length = u16::from_le_bytes([header[2], header[3]]) as usize;
    (length == message.len() - HEADER_BYTES).then(|| u16::from_le_bytes([header[0], header[1]]))
}

#[derive(Debug)]
struct ControlCipher {
    key: LessSafeKey,
    sent: u32,
}

impl ControlCipher {
    fn new(key: &[u8; 16]) -> Self {
        let key = UnboundKey::new(&AES_128_GCM, key).expect("AES-128-GCM takes a 16-byte key");
        Self {
            key: LessSafeKey::new(key),
            sent: 0,
        }
    }

    fn open(&self, message: &[u8]) -> Option<Vec<u8>> {
        if message_type(message)? != MESSAGE_ENCRYPTED || message.len() < ENCRYPTED_PREFIX_BYTES {
            return None;
        }
        let sequence = u32::from_le_bytes(message[4..8].try_into().ok()?);
        let tag: [u8; TAG_BYTES] = message[8..ENCRYPTED_PREFIX_BYTES].try_into().ok()?;
        let mut plain = message[ENCRYPTED_PREFIX_BYTES..].to_vec();
        self.key
            .open_in_place_separate_tag(
                nonce(sequence, CLIENT_NONCE_MARKER),
                Aad::empty(),
                Tag::from(tag),
                &mut plain,
                0..,
            )
            .ok()?;
        Some(plain)
    }

    fn seal(&mut self, message_type: u16, body: &[u8]) -> Vec<u8> {
        let mut plain = Vec::with_capacity(HEADER_BYTES + body.len());
        plain.extend_from_slice(&message_type.to_le_bytes());
        plain.extend_from_slice(&(body.len() as u16).to_le_bytes());
        plain.extend_from_slice(body);
        let sequence = self.sent;
        self.sent = self.sent.wrapping_add(1);
        let tag = self
            .key
            .seal_in_place_separate_tag(
                nonce(sequence, HOST_NONCE_MARKER),
                Aad::empty(),
                &mut plain,
            )
            .expect("a control message is far below the AES-GCM length limit");
        let length = (SEQUENCE_BYTES + TAG_BYTES + plain.len()) as u16;
        let mut sealed = Vec::with_capacity(ENCRYPTED_PREFIX_BYTES + plain.len());
        sealed.extend_from_slice(&MESSAGE_ENCRYPTED.to_le_bytes());
        sealed.extend_from_slice(&length.to_le_bytes());
        sealed.extend_from_slice(&sequence.to_le_bytes());
        sealed.extend_from_slice(tag.as_ref());
        sealed.extend_from_slice(&plain);
        sealed
    }
}

#[derive(Debug)]
pub(crate) struct InputRoute {
    pub(crate) events: mpsc::UnboundedSender<packet::Event>,
    pub(crate) sizes: watch::Receiver<Option<ScreenSize>>,
}

impl InputRoute {
    fn route(&self, body: &[u8]) {
        let log = Target::category(LOG_CATEGORY);
        match input_event(body, *self.sizes.borrow()) {
            // the injection task logs its own failure before it drops the receiver
            Ok(event) => {
                g2g_core::g2g_debug!(log, "input {event:?}");
                drop(self.events.send(event));
            }
            Err(e) if e.is_malformed() => g2g_core::g2g_warn!(log, "dropped input: {e}"),
            Err(e) => g2g_core::g2g_debug!(log, "dropped input: {e}"),
        }
    }
}

fn handle(
    cipher: &ControlCipher,
    message: &[u8],
    keyframe_requests: &AtomicBool,
    input: &InputRoute,
) {
    let log = Target::category(LOG_CATEGORY);
    let opened = match message_type(message) {
        Some(MESSAGE_ENCRYPTED) => match cipher.open(message) {
            Some(opened) => opened,
            None => {
                g2g_core::g2g_warn!(log, "dropped a control message that does not decrypt");
                return;
            }
        },
        Some(_) => message.to_vec(),
        None => {
            g2g_core::g2g_warn!(log, "dropped a control message with a bad length");
            return;
        }
    };
    match message_type(&opened) {
        Some(MESSAGE_REQUEST_IDR) => {
            g2g_core::g2g_info!(log, "Moonlight asked for a keyframe");
            keyframe_requests.store(true, Ordering::Relaxed);
        }
        Some(MESSAGE_INPUT) => input.route(&opened[HEADER_BYTES..]),
        Some(MESSAGE_START_B) => eprintln!("g2g-desktop host: Moonlight started the stream"),
        other => g2g_core::g2g_debug!(log, "control message {other:04x?}"),
    }
}

pub(crate) async fn run(
    mut host: Host,
    key: &[u8; 16],
    keyframe_requests: Arc<AtomicBool>,
    input: InputRoute,
    stop: &Notify,
) -> Result<(), tokio_enet::Error> {
    let mut cipher = ControlCipher::new(key);
    let mut peer: Option<PeerId> = None;
    let mut deadline = Instant::now() + SILENCE_LIMIT;
    loop {
        let event = tokio::select! {
            event = host.service(deadline.saturating_duration_since(Instant::now())) => event?,
            () = stop.notified() => break,
        };
        match event {
            Some(Event::Connect { peer_id, .. }) => {
                peer = Some(peer_id);
                deadline = Instant::now() + SILENCE_LIMIT;
            }
            Some(Event::Disconnect { .. }) => {
                peer = None;
                break;
            }
            Some(Event::Receive { packet, .. }) => {
                deadline = Instant::now() + SILENCE_LIMIT;
                handle(&cipher, packet.data(), &keyframe_requests, &input);
            }
            None if Instant::now() >= deadline => {
                eprintln!(
                    "g2g-desktop host: Moonlight sent nothing for {} seconds",
                    SILENCE_LIMIT.as_secs()
                );
                break;
            }
            None => {}
        }
    }
    // keys held when Moonlight leaves stay pressed on the host
    drop(input.events.send(packet::Event::DeviceRemoved {
        kind: RemovableDevice::Keyboard,
        device: 0,
    }));
    let Some(peer) = peer.and_then(|peer| host.peer_mut(peer)) else {
        return Ok(());
    };
    let termination = cipher.seal(MESSAGE_TERMINATION, &TERMINATION_GRACEFUL.to_be_bytes());
    peer.send(0, Packet::new(&termination, PacketMode::ReliableSequenced))?;
    host.flush().await
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: [u8; 16] = [7; 16];

    fn client_message(message_type: u16, body: &[u8]) -> Vec<u8> {
        let client = LessSafeKey::new(UnboundKey::new(&AES_128_GCM, &KEY).unwrap());
        let mut plain = Vec::from(message_type.to_le_bytes());
        plain.extend_from_slice(&(body.len() as u16).to_le_bytes());
        plain.extend_from_slice(body);
        let sequence = 5u32;
        let tag = client
            .seal_in_place_separate_tag(
                nonce(sequence, CLIENT_NONCE_MARKER),
                Aad::empty(),
                &mut plain,
            )
            .unwrap();
        let mut message = Vec::from(MESSAGE_ENCRYPTED.to_le_bytes());
        message
            .extend_from_slice(&((SEQUENCE_BYTES + TAG_BYTES + plain.len()) as u16).to_le_bytes());
        message.extend_from_slice(&sequence.to_le_bytes());
        message.extend_from_slice(tag.as_ref());
        message.extend_from_slice(&plain);
        message
    }

    fn input_route() -> (InputRoute, mpsc::UnboundedReceiver<packet::Event>) {
        let (events, received) = mpsc::unbounded_channel();
        let (_, sizes) = watch::channel(None);
        (InputRoute { events, sizes }, received)
    }

    #[test]
    fn a_client_idr_request_opens_and_raises_the_request() {
        let requests = AtomicBool::new(false);
        let (input, _) = input_route();
        let message = client_message(MESSAGE_REQUEST_IDR, &[0, 0]);
        handle(&ControlCipher::new(&KEY), &message, &requests, &input);
        assert!(requests.load(Ordering::Relaxed));
    }

    #[test]
    fn a_client_key_press_reaches_the_injector() {
        // moonlight's input body: big-endian length, little-endian key down magic, flags, key code
        let key_down = [0, 0, 0, 10, 3, 0, 0, 0, 0, b'A', 0x80, 0, 0, 0];
        let (input, mut received) = input_route();
        let message = client_message(MESSAGE_INPUT, &key_down);
        handle(
            &ControlCipher::new(&KEY),
            &message,
            &AtomicBool::new(false),
            &input,
        );
        assert!(matches!(
            received.try_recv(),
            Ok(packet::Event::Key { down: true, .. })
        ));
    }

    #[test]
    fn a_host_message_opens_with_the_host_nonce() {
        let mut cipher = ControlCipher::new(&KEY);
        cipher.seal(MESSAGE_TERMINATION, &[1]);
        let sealed = cipher.seal(MESSAGE_TERMINATION, &TERMINATION_GRACEFUL.to_be_bytes());
        assert_eq!(message_type(&sealed), Some(MESSAGE_ENCRYPTED));
        assert_eq!(sealed[4..8], 1u32.to_le_bytes());
        let tag: [u8; TAG_BYTES] = sealed[8..ENCRYPTED_PREFIX_BYTES].try_into().unwrap();
        let mut body = sealed[ENCRYPTED_PREFIX_BYTES..].to_vec();
        LessSafeKey::new(UnboundKey::new(&AES_128_GCM, &KEY).unwrap())
            .open_in_place_separate_tag(
                nonce(1, HOST_NONCE_MARKER),
                Aad::empty(),
                Tag::from(tag),
                &mut body,
                0..,
            )
            .expect("the host tag authenticates");
        assert_eq!(message_type(&body), Some(MESSAGE_TERMINATION));
        assert_eq!(body[4..], TERMINATION_GRACEFUL.to_be_bytes());
    }
}

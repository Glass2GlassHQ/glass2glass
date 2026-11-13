// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_desktop::packet::{
    CursorImage, Event, EventKind, GamepadAxes, PacketError, PacketReader, PacketWriter,
    RemovableDevice, Side,
};
use g2g_desktop::router::{InputRouter, Origin, Routed};
use g2g_desktop::signaling::{
    ClientEvent, ClientSignaling, HostEvent, HostSignaling, SessionSlot, SignalingError,
};
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::UnboundedReceiver;

// USB HID keyboard page usages
const HID_A: u16 = 0x04;
const HID_ESCAPE: u16 = 0x29;
const HEADER_LEN: usize = 8;
const PAYLOAD_LEN_OFFSET: usize = 6;
const TOKEN: &str = "correct horse battery staple";
const OFFER: &str = "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n";
const ANSWER: &str = "v=0\r\no=- 2 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n";

fn cursor(width: u16, height: u16) -> CursorImage {
    let rgba = vec![0x7f; usize::from(width) * usize::from(height) * 4];
    CursorImage::new(width, height, 1, 2, rgba).unwrap()
}

fn every_event() -> Vec<Event> {
    vec![
        Event::Key {
            device: 1,
            usage: HID_A,
            down: true,
        },
        Event::Button {
            button: 2,
            down: false,
        },
        Event::Move { dx: -3, dy: 70_000 },
        Event::Position {
            x: 1919,
            y: -1,
            display: 1,
        },
        Event::Wheel { dx: 0, dy: -120 },
        Event::Gamepad {
            device: 3,
            buttons: 0x8000_0001,
            axes: GamepadAxes {
                left_x: i16::MIN,
                left_y: i16::MAX,
                right_x: -1,
                right_y: 1,
                left_trigger: 0,
                right_trigger: 255,
            },
        },
        Event::Rumble {
            device: 3,
            low: u16::MAX,
            high: 1,
            duration_ms: 250,
        },
        Event::Cursor(cursor(3, 2)),
        Event::Clipboard("snow \u{2603}".to_string()),
        Event::DeviceRemoved {
            kind: RemovableDevice::Gamepad,
            device: 3,
        },
        Event::DeviceRemoved {
            kind: RemovableDevice::Keyboard,
            device: 1,
        },
    ]
}

#[test]
fn every_event_round_trips() {
    let mut writer = PacketWriter::new(Side::Client);
    let mut reader = PacketReader::new(Side::Host);
    for event in every_event() {
        let packet = writer.write(&event).unwrap();
        assert_eq!(reader.read(&packet).unwrap(), Some(event));
    }
}

#[test]
fn a_packet_shorter_than_its_length_field_fails() {
    let mut writer = PacketWriter::new(Side::Client);
    let mut packet = writer.write(&every_event()[0]).unwrap();
    let declared = packet.len() - HEADER_LEN;
    packet.pop();
    let mut reader = PacketReader::new(Side::Host);
    assert_eq!(
        reader.read(&packet),
        Err(PacketError::LengthMismatch {
            declared,
            actual: declared - 1,
        })
    );
    assert_eq!(
        reader.read(&packet[..HEADER_LEN - 1]),
        Err(PacketError::Truncated {
            len: HEADER_LEN - 1
        })
    );
}

#[test]
fn a_consistently_truncated_payload_fails_its_type_length() {
    let mut writer = PacketWriter::new(Side::Client);
    let mut packet = writer.write(&Event::Move { dx: 1, dy: 2 }).unwrap();
    let expected = packet.len() - HEADER_LEN;
    packet.pop();
    let actual = u16::try_from(expected - 1).unwrap();
    packet[PAYLOAD_LEN_OFFSET..HEADER_LEN].copy_from_slice(&actual.to_le_bytes());
    let mut reader = PacketReader::new(Side::Host);
    assert_eq!(
        reader.read(&packet),
        Err(PacketError::PayloadLength {
            kind: EventKind::Move,
            expected,
            actual: expected - 1,
        })
    );
}

#[test]
fn an_older_sequence_of_the_same_type_is_dropped() {
    let mut writer = PacketWriter::new(Side::Client);
    let older_position = writer
        .write(&Event::Position {
            x: 1,
            y: 1,
            display: 0,
        })
        .unwrap();
    let older_key = writer.write(&every_event()[0]).unwrap();
    let newer_position = Event::Position {
        x: 2,
        y: 2,
        display: 0,
    };
    let newer_position_packet = writer.write(&newer_position).unwrap();

    let mut reader = PacketReader::new(Side::Host);
    assert_eq!(
        reader.read(&newer_position_packet).unwrap(),
        Some(newer_position)
    );
    assert_eq!(reader.read(&older_position).unwrap(), None);
    assert_eq!(
        reader.read(&older_key).unwrap(),
        Some(every_event()[0].clone())
    );
}

#[test]
fn a_cursor_whose_rgba_length_does_not_match_its_size_fails() {
    assert_eq!(
        CursorImage::new(2, 2, 0, 0, vec![0; 15]),
        Err(PacketError::CursorSize {
            width: 2,
            height: 2,
            rgba_len: 15,
        })
    );

    let mut writer = PacketWriter::new(Side::Host);
    let mut packet = writer.write(&Event::Cursor(cursor(2, 2))).unwrap();
    packet[HEADER_LEN..HEADER_LEN + 2].copy_from_slice(&3u16.to_le_bytes());
    let mut reader = PacketReader::new(Side::Client);
    assert_eq!(
        reader.read(&packet),
        Err(PacketError::CursorSize {
            width: 3,
            height: 2,
            rgba_len: 16,
        })
    );
}

#[test]
fn a_cursor_over_the_data_channel_message_cap_is_not_written() {
    let mut writer = PacketWriter::new(Side::Host);
    let too_large = writer.write(&Event::Cursor(cursor(128, 128)));
    assert!(matches!(too_large, Err(PacketError::TooLarge { .. })));
}

#[test]
fn a_packet_flagged_from_the_receiving_side_fails() {
    let mut writer = PacketWriter::new(Side::Host);
    let packet = writer.write(&every_event()[0]).unwrap();
    let mut reader = PacketReader::new(Side::Host);
    assert_eq!(reader.read(&packet), Err(PacketError::WrongDirection));
}

#[test]
fn an_unknown_type_fails() {
    let mut writer = PacketWriter::new(Side::Client);
    let mut packet = writer.write(&every_event()[0]).unwrap();
    packet[0] = 0;
    let mut reader = PacketReader::new(Side::Host);
    assert_eq!(reader.read(&packet), Err(PacketError::UnknownKind(0)));
}

const USER_INPUT: [EventKind; 6] = [
    EventKind::Key,
    EventKind::Button,
    EventKind::Move,
    EventKind::Position,
    EventKind::Wheel,
    EventKind::Gamepad,
];

fn key(usage: u16, down: bool) -> Event {
    Event::Key {
        device: 0,
        usage,
        down,
    }
}

fn drained(receiver: &mut UnboundedReceiver<Event>) -> Vec<Event> {
    let mut events = Vec::new();
    loop {
        match receiver.try_recv() {
            Ok(event) => events.push(event),
            Err(TryRecvError::Empty) => return events,
            Err(TryRecvError::Disconnected) => panic!("router dropped its consumer"),
        }
    }
}

#[test]
fn the_client_router_drops_input_while_unfocused() {
    let mut router = InputRouter::new(Vec::new());
    let mut network = router.subscribe(Origin::Local, &USER_INPUT);
    router.set_focused(false);
    assert_eq!(
        router.push(Origin::Local, Event::Move { dx: 1, dy: 1 }),
        Routed::Unfocused
    );
    assert_eq!(
        router.push(Origin::Local, key(HID_A, true)),
        Routed::Unfocused
    );
    assert!(drained(&mut network).is_empty());

    router.set_focused(true);
    assert_eq!(
        router.push(Origin::Local, key(HID_A, true)),
        Routed::Delivered
    );
    assert_eq!(drained(&mut network), [key(HID_A, true)]);
}

#[test]
fn losing_focus_releases_held_keys_and_buttons() {
    let mut router = InputRouter::new(Vec::new());
    let mut network = router.subscribe(Origin::Local, &USER_INPUT);
    router.push(Origin::Local, key(HID_A, true));
    router.push(
        Origin::Local,
        Event::Button {
            button: 0,
            down: true,
        },
    );
    drained(&mut network);

    router.set_focused(false);
    assert_eq!(
        drained(&mut network),
        [
            key(HID_A, false),
            Event::Button {
                button: 0,
                down: false
            }
        ]
    );
}

#[test]
fn the_client_router_swallows_a_shortcut_key() {
    let mut router = InputRouter::new(vec![HID_ESCAPE]);
    let mut network = router.subscribe(Origin::Local, &USER_INPUT);
    assert_eq!(
        router.push(Origin::Local, key(HID_ESCAPE, true)),
        Routed::ShortcutPressed(HID_ESCAPE)
    );
    assert_eq!(
        router.push(Origin::Local, key(HID_ESCAPE, false)),
        Routed::ShortcutReleased(HID_ESCAPE)
    );
    router.push(Origin::Local, key(HID_A, true));
    assert_eq!(drained(&mut network), [key(HID_A, true)]);
}

#[test]
fn a_remote_cursor_reaches_only_the_cursor_consumer() {
    let mut router = InputRouter::new(Vec::new());
    let mut network = router.subscribe(Origin::Local, &[EventKind::Cursor, EventKind::Clipboard]);
    let mut cursor_consumer = router.subscribe(Origin::Remote, &[EventKind::Cursor]);
    let mut clipboard_consumer = router.subscribe(Origin::Remote, &[EventKind::Clipboard]);

    let remote_cursor = Event::Cursor(cursor(2, 2));
    router.set_focused(false);
    assert_eq!(
        router.push(Origin::Remote, remote_cursor.clone()),
        Routed::Delivered
    );

    assert_eq!(drained(&mut cursor_consumer), [remote_cursor]);
    assert!(drained(&mut network).is_empty());
    assert!(drained(&mut clipboard_consumer).is_empty());
}

#[test]
fn an_event_type_nobody_subscribed_to_is_not_delivered() {
    let mut router = InputRouter::new(Vec::new());
    let mut network = router.subscribe(Origin::Local, &USER_INPUT);
    let clipboard = Event::Clipboard("not synced".to_string());
    assert_eq!(router.push(Origin::Local, clipboard), Routed::NoConsumer);
    assert!(drained(&mut network).is_empty());
}

fn host_receive_bytewise(host: &mut HostSignaling, bytes: &[u8]) -> Vec<HostEvent> {
    let mut events = Vec::new();
    for byte in bytes {
        events.extend(host.receive(&[*byte]).unwrap());
    }
    events
}

#[test]
fn a_session_signals_byte_by_byte() {
    let mut host = HostSignaling::new(TOKEN).unwrap();
    let (mut client, token) = ClientSignaling::new(TOKEN).unwrap();

    let [HostEvent::Reply(accept)] = &host_receive_bytewise(&mut host, &token)[..] else {
        panic!("host did not reply to a correct token");
    };
    assert_eq!(client.receive(accept).unwrap(), [ClientEvent::Accepted]);

    let offer = client.offer(OFFER).unwrap();
    assert_eq!(
        host_receive_bytewise(&mut host, &offer),
        [HostEvent::Offer(OFFER.to_string())]
    );

    let answer = host.answer(ANSWER).unwrap();
    assert_eq!(
        client.receive(&answer).unwrap(),
        [ClientEvent::Answer(ANSWER.to_string())]
    );
    assert_eq!(host.connection_closed(), Ok(()));
    assert_eq!(client.connection_closed(), Ok(()));
}

#[test]
fn a_bad_token_is_refused() {
    let mut host = HostSignaling::new(TOKEN).unwrap();
    let (_, wrong_token) = ClientSignaling::new("wrong").unwrap();
    assert_eq!(host.receive(&wrong_token), Err(SignalingError::BadToken));
}

#[test]
fn an_oversized_token_is_refused_from_its_length_prefix() {
    let mut host = HostSignaling::new(TOKEN).unwrap();
    let declared = 257u32;
    assert_eq!(
        host.receive(&declared.to_be_bytes()),
        Err(SignalingError::MessageTooLong { len: 257, max: 256 })
    );
}

#[test]
fn an_offer_before_the_accept_byte_fails() {
    let (mut client, _) = ClientSignaling::new(TOKEN).unwrap();
    assert_eq!(client.offer(OFFER), Err(SignalingError::OfferBeforeAccept));
}

fn signaled_up_to_the_offer() -> (HostSignaling, ClientSignaling, Vec<u8>) {
    let mut host = HostSignaling::new(TOKEN).unwrap();
    let (mut client, token) = ClientSignaling::new(TOKEN).unwrap();
    let [HostEvent::Reply(accept)] = &host.receive(&token).unwrap()[..] else {
        panic!("host did not reply to a correct token");
    };
    client.receive(accept).unwrap();
    let offer = client.offer(OFFER).unwrap();
    host.receive(&offer).unwrap();
    (host, client, offer)
}

#[test]
fn a_message_after_the_answer_is_a_protocol_error() {
    let (mut host, _, offer) = signaled_up_to_the_offer();
    host.answer(ANSWER).unwrap();
    assert_eq!(host.receive(&offer), Err(SignalingError::UnexpectedMessage));
}

#[test]
fn a_hang_up_before_the_answer_fails_the_session() {
    let (host, client, _) = signaled_up_to_the_offer();
    assert_eq!(host.connection_closed(), Err(SignalingError::HungUp));
    assert_eq!(client.connection_closed(), Err(SignalingError::HungUp));
}

#[test]
fn a_second_connection_is_refused_while_the_first_is_up() {
    let slot = SessionSlot::default();
    let first = slot.claim().unwrap();
    assert_eq!(slot.claim().unwrap_err(), SignalingError::Busy);
    drop(first);
    assert!(slot.claim().is_ok());
}

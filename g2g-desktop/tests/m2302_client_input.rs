// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "session"))]

use g2g_desktop::client::forward_window_input;
use g2g_desktop::packet::{Event, GamepadAxes, PacketReader, RemovableDevice, Side, GAMEPAD_A};
use g2g_plugins::windowinput::{PointerButton, WindowInput};
use tokio::sync::mpsc;

// linux input-event-codes.h KEY_A, and USB HID keyboard usage A
const EVDEV_A: u32 = 30;
const HID_A: u16 = 0x04;
const LEFT_BUTTON: u8 = 0;

#[tokio::test]
async fn window_input_reaches_the_network_and_losing_focus_releases_held_keys() {
    let (window, window_input) = mpsc::unbounded_channel();
    let (outgoing, mut packets) = mpsc::unbounded_channel();
    for input in [
        WindowInput::Focus(true),
        WindowInput::Key {
            scancode: EVDEV_A,
            down: true,
        },
        WindowInput::Pointer { x: 10, y: 20 },
        WindowInput::Button {
            button: PointerButton::Left,
            down: true,
        },
        WindowInput::Wheel { dx: 0, dy: -120 },
        WindowInput::Focus(false),
        WindowInput::Key {
            scancode: EVDEV_A,
            down: false,
        },
        WindowInput::Pointer { x: 30, y: 40 },
    ] {
        window.send(input).unwrap();
    }
    drop(window);
    forward_window_input(window_input, mpsc::unbounded_channel().1, outgoing).await;

    let mut reader = PacketReader::new(Side::Host);
    let mut received = Vec::new();
    while let Ok(packet) = packets.try_recv() {
        received.extend(reader.read(&packet).unwrap());
    }
    assert_eq!(
        received,
        [
            Event::Key {
                device: 0,
                usage: HID_A,
                down: true
            },
            Event::Position {
                x: 10,
                y: 20,
                display: 0
            },
            Event::Button {
                button: LEFT_BUTTON,
                down: true
            },
            Event::Wheel { dx: 0, dy: -120 },
            Event::Key {
                device: 0,
                usage: HID_A,
                down: false
            },
            Event::Button {
                button: LEFT_BUTTON,
                down: false
            },
        ],
        "the key and button still down are released on focus loss, and nothing passes while unfocused"
    );
}

const PAD: u8 = 2;

async fn next_event(
    packets: &mut mpsc::UnboundedReceiver<Vec<u8>>,
    reader: &mut PacketReader,
) -> Event {
    let packet = packets.recv().await.expect("the client forwards a packet");
    reader.read(&packet).unwrap().expect("a whole event")
}

#[tokio::test]
async fn losing_focus_centres_moved_gamepads_and_a_removal_still_passes() {
    let (window, window_input) = mpsc::unbounded_channel();
    let (gamepads, gamepad_states) = mpsc::unbounded_channel();
    let (outgoing, mut packets) = mpsc::unbounded_channel();
    let forward = tokio::spawn(forward_window_input(window_input, gamepad_states, outgoing));
    let mut reader = PacketReader::new(Side::Host);
    let pressed = Event::Gamepad {
        device: PAD,
        buttons: GAMEPAD_A,
        axes: GamepadAxes {
            left_y: i16::MAX,
            ..GamepadAxes::default()
        },
    };

    window.send(WindowInput::Focus(true)).unwrap();
    gamepads.send(pressed.clone()).unwrap();
    assert_eq!(next_event(&mut packets, &mut reader).await, pressed);

    window.send(WindowInput::Focus(false)).unwrap();
    let centred = Event::Gamepad {
        device: PAD,
        buttons: 0,
        axes: GamepadAxes::default(),
    };
    assert_eq!(next_event(&mut packets, &mut reader).await, centred);

    let removed = Event::DeviceRemoved {
        kind: RemovableDevice::Gamepad,
        device: PAD,
    };
    gamepads.send(pressed).unwrap();
    gamepads.send(removed.clone()).unwrap();
    assert_eq!(
        next_event(&mut packets, &mut reader).await,
        removed,
        "an unfocused state is dropped and the removal passes"
    );
    drop(window);
    forward.await.unwrap();
}

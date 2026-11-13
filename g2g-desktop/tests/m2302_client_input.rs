// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "session"))]

use g2g_desktop::client::forward_window_input;
use g2g_desktop::packet::{Event, PacketReader, Side};
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
    forward_window_input(window_input, outgoing).await;

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

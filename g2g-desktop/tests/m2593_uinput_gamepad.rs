// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(target_os = "linux")]

use std::fs::OpenOptions;
use std::time::{Duration, Instant};

use evdev::{AbsoluteAxisCode, Device, KeyCode};
use g2g_desktop::injection::InjectEvents;
use g2g_desktop::packet::{
    Event, GamepadAxes, RemovableDevice, GAMEPAD_A, GAMEPAD_DPAD_UP, GAMEPAD_Y,
};
use g2g_desktop::uinputgamepad::{UinputGamepads, GAMEPAD_NAME};

const DEVICE_APPEARS: Duration = Duration::from_secs(5);
const STICK_UP: i16 = 32767;
const TRIGGER_HALF: i16 = 128;

fn gamepad() -> Option<Device> {
    evdev::enumerate()
        .map(|(_, device)| device)
        .find(|device| device.name() == Some(GAMEPAD_NAME))
}

fn wait_for(appears: bool) -> Option<Device> {
    let deadline = Instant::now() + DEVICE_APPEARS;
    loop {
        let found = gamepad();
        if found.is_some() == appears {
            return found;
        }
        assert!(
            Instant::now() < deadline,
            "{GAMEPAD_NAME} appears: {appears}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn a_gamepad_state_reaches_a_virtual_xbox_pad_and_removal_destroys_it() {
    if OpenOptions::new().write(true).open("/dev/uinput").is_err() {
        return;
    }
    let mut gamepads = UinputGamepads::default();
    let state = |buttons, axes| Event::Gamepad {
        device: 0,
        buttons,
        axes,
    };
    gamepads
        .inject(&state(0, GamepadAxes::default()))
        .expect("creates the pad");
    let mut device = wait_for(true).expect("the pad appears");
    // linux EVIOCGRAB keeps these presses from the desktop session
    device.grab().expect("grabs the pad");

    let axes = GamepadAxes {
        left_y: STICK_UP,
        right_trigger: TRIGGER_HALF,
        ..GamepadAxes::default()
    };
    gamepads
        .inject(&state(GAMEPAD_A | GAMEPAD_Y | GAMEPAD_DPAD_UP, axes))
        .expect("injects");
    let keys = device.get_key_state().expect("reads the buttons");
    assert!(keys.contains(KeyCode::BTN_SOUTH), "A is down");
    assert!(keys.contains(KeyCode::BTN_WEST), "Y is down");
    assert!(!keys.contains(KeyCode::BTN_EAST), "B is up");
    let axis = device.get_abs_state().expect("reads the axes");
    let value = |code: AbsoluteAxisCode| axis[usize::from(code.0)].value;
    assert_eq!(
        value(AbsoluteAxisCode::ABS_Y),
        -32768,
        "linux reads up as negative"
    );
    assert_eq!(value(AbsoluteAxisCode::ABS_RZ), i32::from(TRIGGER_HALF));
    assert_eq!(value(AbsoluteAxisCode::ABS_HAT0Y), -1, "the dpad points up");

    gamepads
        .inject(&Event::DeviceRemoved {
            kind: RemovableDevice::Gamepad,
            device: 0,
        })
        .expect("removes the pad");
    drop(device);
    assert!(wait_for(false).is_none(), "the pad is gone");
}

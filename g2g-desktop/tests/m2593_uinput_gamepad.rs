// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "session"))]

use std::fs::OpenOptions;
use std::future::poll_fn;
use std::time::{Duration, Instant};

use evdev::{AbsoluteAxisCode, Device, FFEffectData, FFEffectKind, FFReplay, FFTrigger, KeyCode};
use g2g_desktop::injection::{GamepadRumble, InjectEvents};
use g2g_desktop::packet::{
    Event, GamepadAxes, MotorLevels, RemovableDevice, GAMEPAD_A, GAMEPAD_DPAD_UP, GAMEPAD_Y,
};
use g2g_desktop::uinputgamepad::{UinputGamepads, GAMEPAD_NAME};

const DEVICE_APPEARS: Duration = Duration::from_secs(5);
const STICK_UP: i16 = 32767;
const TRIGGER_HALF: i16 = 128;
const RUMBLE_ROUND_TRIP: Duration = Duration::from_secs(10);
// every test pad has the same name, so tests that look one up take turns
static PADS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

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

#[tokio::test]
async fn a_gamepad_state_reaches_a_virtual_xbox_pad_and_removal_destroys_it() {
    if OpenOptions::new().write(true).open("/dev/uinput").is_err() {
        return;
    }
    let _turn = PADS.lock().await;
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

#[tokio::test]
async fn a_rumble_a_game_plays_on_the_pad_comes_back_as_motor_levels() {
    if OpenOptions::new().write(true).open("/dev/uinput").is_err() {
        return;
    }
    let _turn = PADS.lock().await;
    let pad = 1;
    let mut gamepads = UinputGamepads::default();
    gamepads
        .inject(&Event::Gamepad {
            device: pad,
            buttons: 0,
            axes: GamepadAxes::default(),
        })
        .expect("creates the pad");
    let device = wait_for(true).expect("the pad appears");
    let motors = MotorLevels {
        low_frequency: 0x8000,
        high_frequency: 0x4000,
    };
    let (stop, stopped) = std::sync::mpsc::channel();
    // linux blocks a game's effect upload and erase until the pad's owner answers them
    let mut game = tokio::task::spawn_blocking(move || {
        let mut device = device;
        let mut effect = device
            .upload_ff_effect(FFEffectData {
                direction: 0,
                trigger: FFTrigger::default(),
                replay: FFReplay {
                    length: 0,
                    delay: 0,
                },
                kind: FFEffectKind::Rumble {
                    strong_magnitude: motors.low_frequency,
                    weak_magnitude: motors.high_frequency,
                },
            })
            .expect("uploads a rumble");
        effect.play(1).expect("plays the rumble");
        stopped.recv().expect("the test asks to stop");
        effect.stop().expect("stops the rumble");
    });
    let mut rumbles = Vec::new();
    tokio::time::timeout(RUMBLE_ROUND_TRIP, async {
        loop {
            tokio::select! {
                rumble = poll_fn(|cx| gamepads.poll_rumble(cx)) => {
                    rumbles.push(rumble.expect("reads the pad"));
                    if rumbles.len() == 1 {
                        stop.send(()).expect("the game waits");
                    }
                }
                finished = &mut game => {
                    finished.expect("the game runs");
                    break;
                }
            }
        }
    })
    .await
    .expect("the rumble round trip finishes");
    assert_eq!(
        rumbles,
        [
            GamepadRumble {
                device: pad,
                motors
            },
            GamepadRumble {
                device: pad,
                motors: MotorLevels::default(),
            },
        ]
    );

    gamepads
        .inject(&Event::DeviceRemoved {
            kind: RemovableDevice::Gamepad,
            device: pad,
        })
        .expect("removes the pad");
    assert!(wait_for(false).is_none(), "the pad is gone");
}

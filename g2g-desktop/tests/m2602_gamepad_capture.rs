// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "session"))]

use std::fs::OpenOptions;
use std::future::poll_fn;
use std::time::Duration;

use g2g_desktop::gamepadcapture::{capture_gamepads, GamepadCapture};
use g2g_desktop::injection::{GamepadRumble, InjectEvents};
use g2g_desktop::packet::{
    Event, GamepadAxes, MotorLevels, RemovableDevice, GAMEPAD_A, GAMEPAD_DPAD_UP,
    GAMEPAD_LEFT_SHOULDER, GAMEPAD_X,
};
use g2g_desktop::uinputgamepad::UinputGamepads;

const HOST_PAD: u8 = 0;
const ROUND_TRIP: Duration = Duration::from_secs(10);
const STATE_WAIT: Duration = Duration::from_millis(200);
const TRIGGER_FULL: i16 = 255;
const STICK_STEP_LOST: u16 = 1;

async fn next_state(
    capture: &mut GamepadCapture,
    pads: &mut UinputGamepads,
    rumbles: &mut Vec<GamepadRumble>,
    within: Duration,
) -> Option<Event> {
    tokio::time::timeout(within, async {
        loop {
            // linux blocks gilrs's effect uploads until the pad's owner answers them
            tokio::select! {
                state = capture.states.recv() => return state.expect("the capture runs"),
                rumble = poll_fn(|cx| pads.poll_rumble(cx)) => {
                    rumbles.push(rumble.expect("reads the pad"));
                }
            }
        }
    })
    .await
    .ok()
}

#[tokio::test]
async fn a_pad_state_comes_back_through_capture_and_a_rumble_reaches_the_pad() {
    if OpenOptions::new().write(true).open("/dev/uinput").is_err() {
        return;
    }
    let mut pads = UinputGamepads::default();
    let mut rumbles = Vec::new();
    let neutral = Event::Gamepad {
        device: HOST_PAD,
        buttons: 0,
        axes: GamepadAxes::default(),
    };
    pads.inject(&neutral).expect("creates the pad");
    let mut capture = capture_gamepads();

    let axes = GamepadAxes {
        left_y: i16::MAX,
        right_trigger: TRIGGER_FULL,
        ..GamepadAxes::default()
    };
    let buttons = GAMEPAD_A | GAMEPAD_X | GAMEPAD_LEFT_SHOULDER | GAMEPAD_DPAD_UP;
    let pressed = Event::Gamepad {
        device: HOST_PAD,
        buttons,
        axes,
    };
    let is_pressed = |state: &Event| {
        let Event::Gamepad {
            buttons: held,
            axes: moved,
            ..
        } = state
        else {
            return false;
        };
        // gilrs reads sticks as f32, which loses the last step of an i16
        let close = |captured: i16, sent: i16| captured.abs_diff(sent) <= STICK_STEP_LOST;
        *held == buttons
            && close(moved.left_y, axes.left_y)
            && close(moved.left_x, axes.left_x)
            && moved.right_trigger == axes.right_trigger
            && moved.left_trigger == axes.left_trigger
    };
    // gilrs opens a pad some time after it appears, so the press repeats until it comes back
    let device = tokio::time::timeout(ROUND_TRIP, async {
        loop {
            pads.inject(&pressed).expect("injects");
            while let Some(state) =
                next_state(&mut capture, &mut pads, &mut rumbles, STATE_WAIT).await
            {
                if let Event::Gamepad { device, .. } = state {
                    if is_pressed(&state) {
                        return device;
                    }
                }
            }
            pads.inject(&neutral).expect("injects");
        }
    })
    .await
    .expect("the pressed state comes back");

    let motors = MotorLevels {
        low_frequency: 0x8000,
        high_frequency: 0x4000,
    };
    capture
        .rumbles
        .send(GamepadRumble { device, motors })
        .expect("the capture runs");
    tokio::time::timeout(ROUND_TRIP, async {
        while !rumbles.iter().any(|rumble| rumble.motors == motors) {
            next_state(&mut capture, &mut pads, &mut rumbles, STATE_WAIT).await;
        }
    })
    .await
    .expect("the rumble reaches the pad");
    assert!(rumbles.iter().all(|rumble| rumble.device == HOST_PAD));

    pads.inject(&Event::DeviceRemoved {
        kind: RemovableDevice::Gamepad,
        device: HOST_PAD,
    })
    .expect("removes the pad");
    let removed = Event::DeviceRemoved {
        kind: RemovableDevice::Gamepad,
        device,
    };
    tokio::time::timeout(ROUND_TRIP, async {
        loop {
            let state = next_state(&mut capture, &mut pads, &mut rumbles, STATE_WAIT).await;
            if state.as_ref() == Some(&removed) {
                return;
            }
        }
    })
    .await
    .expect("the capture reports the removal");
}

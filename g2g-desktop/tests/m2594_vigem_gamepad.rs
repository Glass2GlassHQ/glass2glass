// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "windows", feature = "gamestream"))]

use g2g_desktop::injection::{InjectError, InjectEvents};
use g2g_desktop::packet::{Event, GamepadAxes, RemovableDevice, GAMEPAD_A};
use g2g_desktop::vigemgamepad::ViGEmGamepads;

#[test]
fn a_gamepad_reaches_vigembus_or_is_reported_unavailable_once() {
    let mut gamepads = ViGEmGamepads::default();
    let pressed = Event::Gamepad {
        device: 0,
        buttons: GAMEPAD_A,
        axes: GamepadAxes {
            right_trigger: 200,
            ..GamepadAxes::default()
        },
    };
    match gamepads.inject(&pressed) {
        Ok(()) => {
            let removed = Event::DeviceRemoved {
                kind: RemovableDevice::Gamepad,
                device: 0,
            };
            gamepads.inject(&removed).expect("unplugs the pad");
        }
        Err(InjectError::GamepadUnavailable(e)) => {
            let cause = e.get_ref().and_then(|cause| cause.downcast_ref());
            assert!(
                matches!(cause, Some(vigem_client::Error::BusNotFound)),
                "only a missing ViGEmBus makes gamepads unavailable: {e}"
            );
            assert!(
                gamepads.inject(&pressed).is_ok(),
                "later gamepads are dropped without a second report"
            );
        }
        Err(e) => panic!("ViGEmBus refused the report: {e}"),
    }
}

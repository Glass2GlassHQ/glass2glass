// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::io;

use evdev::uinput::VirtualDevice;
use evdev::{
    AbsInfo, AbsoluteAxisCode, AbsoluteAxisEvent, AttributeSet, BusType, InputEvent, InputId,
    KeyCode, KeyEvent, UinputAbsSetup,
};

use crate::injection::{InjectError, InjectEvents};
use crate::packet::{
    Event, GamepadAxes, RemovableDevice, GAMEPAD_A, GAMEPAD_B, GAMEPAD_BACK, GAMEPAD_DPAD_DOWN,
    GAMEPAD_DPAD_LEFT, GAMEPAD_DPAD_RIGHT, GAMEPAD_DPAD_UP, GAMEPAD_GUIDE, GAMEPAD_LEFT_SHOULDER,
    GAMEPAD_LEFT_THUMB, GAMEPAD_RIGHT_SHOULDER, GAMEPAD_RIGHT_THUMB, GAMEPAD_START, GAMEPAD_X,
    GAMEPAD_Y,
};

pub const GAMEPAD_NAME: &str = "g2g-desktop Xbox 360 pad";
// sdl and steam pick their Xbox 360 mapping from this usb vendor and product
const XBOX_360_VENDOR: u16 = 0x045e;
const XBOX_360_PRODUCT: u16 = 0x028e;
const XBOX_360_VERSION: u16 = 0x0110;

// the linux xpad driver reports X and Y with the codes evdev names north and west
const BUTTON_KEYS: [(u32, KeyCode); 11] = [
    (GAMEPAD_A, KeyCode::BTN_SOUTH),
    (GAMEPAD_B, KeyCode::BTN_EAST),
    (GAMEPAD_X, KeyCode::BTN_NORTH),
    (GAMEPAD_Y, KeyCode::BTN_WEST),
    (GAMEPAD_LEFT_SHOULDER, KeyCode::BTN_TL),
    (GAMEPAD_RIGHT_SHOULDER, KeyCode::BTN_TR),
    (GAMEPAD_BACK, KeyCode::BTN_SELECT),
    (GAMEPAD_START, KeyCode::BTN_START),
    (GAMEPAD_GUIDE, KeyCode::BTN_MODE),
    (GAMEPAD_LEFT_THUMB, KeyCode::BTN_THUMBL),
    (GAMEPAD_RIGHT_THUMB, KeyCode::BTN_THUMBR),
];

// the ranges the linux xpad driver reports for an Xbox 360 pad
fn axis_setups() -> [UinputAbsSetup; 8] {
    let stick = |code| UinputAbsSetup::new(code, AbsInfo::new(0, -32768, 32767, 16, 128, 0));
    let trigger = |code| UinputAbsSetup::new(code, AbsInfo::new(0, 0, 255, 0, 0, 0));
    let hat = |code| UinputAbsSetup::new(code, AbsInfo::new(0, -1, 1, 0, 0, 0));
    [
        stick(AbsoluteAxisCode::ABS_X),
        stick(AbsoluteAxisCode::ABS_Y),
        stick(AbsoluteAxisCode::ABS_RX),
        stick(AbsoluteAxisCode::ABS_RY),
        trigger(AbsoluteAxisCode::ABS_Z),
        trigger(AbsoluteAxisCode::ABS_RZ),
        hat(AbsoluteAxisCode::ABS_HAT0X),
        hat(AbsoluteAxisCode::ABS_HAT0Y),
    ]
}

fn build_gamepad() -> io::Result<VirtualDevice> {
    let keys = AttributeSet::from_iter(BUTTON_KEYS.map(|(_, key)| key));
    let mut builder = VirtualDevice::builder()?
        .name(GAMEPAD_NAME)
        .input_id(InputId::new(
            BusType::BUS_USB,
            XBOX_360_VENDOR,
            XBOX_360_PRODUCT,
            XBOX_360_VERSION,
        ))
        .with_keys(&keys)?;
    for setup in axis_setups() {
        builder = builder.with_absolute_axis(&setup)?;
    }
    builder.build()
}

fn pressed_toward(buttons: u32, negative: u32, positive: u32) -> i32 {
    i32::from(buttons & positive != 0) - i32::from(buttons & negative != 0)
}

// linux reports a stick pushed up as negative Y, xinput as positive
fn state_events(buttons: u32, axes: GamepadAxes) -> Vec<InputEvent> {
    let keys = BUTTON_KEYS
        .iter()
        .map(|(flag, key)| *KeyEvent::new(*key, i32::from(buttons & flag != 0)));
    let axis = |code, value| *AbsoluteAxisEvent::new(code, value);
    let positions = [
        axis(AbsoluteAxisCode::ABS_X, i32::from(axes.left_x)),
        axis(AbsoluteAxisCode::ABS_Y, i32::from(!axes.left_y)),
        axis(AbsoluteAxisCode::ABS_RX, i32::from(axes.right_x)),
        axis(AbsoluteAxisCode::ABS_RY, i32::from(!axes.right_y)),
        axis(AbsoluteAxisCode::ABS_Z, i32::from(axes.left_trigger)),
        axis(AbsoluteAxisCode::ABS_RZ, i32::from(axes.right_trigger)),
        axis(
            AbsoluteAxisCode::ABS_HAT0X,
            pressed_toward(buttons, GAMEPAD_DPAD_LEFT, GAMEPAD_DPAD_RIGHT),
        ),
        axis(
            AbsoluteAxisCode::ABS_HAT0Y,
            pressed_toward(buttons, GAMEPAD_DPAD_UP, GAMEPAD_DPAD_DOWN),
        ),
    ];
    keys.chain(positions).collect()
}

#[derive(Debug, Default)]
pub struct UinputGamepads {
    pads: BTreeMap<u8, VirtualDevice>,
    unavailable: bool,
}

impl InjectEvents for UinputGamepads {
    fn inject(&mut self, event: &Event) -> Result<(), InjectError> {
        match event {
            Event::Gamepad {
                device,
                buttons,
                axes,
            } => {
                // opening /dev/uinput failed once and that failure was reported
                if self.unavailable {
                    return Ok(());
                }
                let pad = match self.pads.entry(*device) {
                    Entry::Occupied(pad) => pad.into_mut(),
                    Entry::Vacant(slot) => match build_gamepad() {
                        Ok(pad) => slot.insert(pad),
                        Err(e) => {
                            self.unavailable = true;
                            return Err(InjectError::GamepadUnavailable(e));
                        }
                    },
                };
                pad.emit(&state_events(*buttons, *axes))
                    .map_err(InjectError::Refused)?;
            }
            Event::DeviceRemoved {
                kind: RemovableDevice::Gamepad,
                device,
            } => {
                self.pads.remove(device);
            }
            _ => return Err(InjectError::NotInjected(event.kind())),
        }
        Ok(())
    }
}

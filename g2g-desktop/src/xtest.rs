// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::BTreeSet;
use std::fmt;
use std::io;

use g2g_plugins::windowinput::PointerButton;
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    Window, BUTTON_PRESS_EVENT, BUTTON_RELEASE_EVENT, KEY_PRESS_EVENT, KEY_RELEASE_EVENT,
    MOTION_NOTIFY_EVENT,
};
use x11rb::protocol::xtest::ConnectionExt as _;
use x11rb::rust_connection::RustConnection;

use crate::hid::evdev_from_usage;
use crate::injection::{check_position, InjectError, InjectEvents, ScreenSize};
use crate::packet::{Event, RemovableDevice, POINTER_BUTTONS};
use crate::uinput::{whole_notches, WheelRemainder};

// X keycodes are evdev codes offset by 8
const X_KEYCODE_OFFSET: u16 = 8;
// XTest moves the pointer by the given amount when the motion detail is 1, to the given point when 0
const ABSOLUTE_MOTION: u8 = 0;
const RELATIVE_MOTION: u8 = 1;
// X11 core buttons 4 to 7 scroll up, down, left and right by one notch per click
const WHEEL_UP: u8 = 4;
const WHEEL_DOWN: u8 = 5;
const WHEEL_LEFT: u8 = 6;
const WHEEL_RIGHT: u8 = 7;

fn x_button(button: PointerButton) -> u8 {
    match button {
        PointerButton::Left => 1,
        PointerButton::Middle => 2,
        PointerButton::Right => 3,
        PointerButton::Back => 8,
        PointerButton::Forward => 9,
    }
}

fn x_failure(error: impl fmt::Display) -> InjectError {
    InjectError::Device(io::Error::other(error.to_string()))
}

pub struct XTestInjector {
    connection: RustConnection,
    root: Window,
    screen: ScreenSize,
    held_keys: BTreeSet<u8>,
    wheel: WheelRemainder,
}

impl fmt::Debug for XTestInjector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XTestInjector")
            .field("root", &self.root)
            .field("screen", &self.screen)
            .field("held_keys", &self.held_keys)
            .finish_non_exhaustive()
    }
}

impl XTestInjector {
    pub fn new(display_name: &str, screen: ScreenSize) -> Result<Self, InjectError> {
        let (connection, screen_number) = x11rb::connect(Some(display_name)).map_err(x_failure)?;
        connection
            .xtest_get_version(2, 2)
            .map_err(x_failure)?
            .reply()
            .map_err(x_failure)?;
        let root = connection.setup().roots[screen_number].root;
        Ok(Self {
            connection,
            root,
            screen,
            held_keys: BTreeSet::new(),
            wheel: WheelRemainder::default(),
        })
    }

    fn fake(&self, kind: u8, detail: u8, x: i16, y: i16) -> Result<(), InjectError> {
        self.connection
            .xtest_fake_input(kind, detail, x11rb::CURRENT_TIME, self.root, x, y, 0)
            .map_err(x_failure)?;
        Ok(())
    }

    fn click(&self, button: u8, times: i32) -> Result<(), InjectError> {
        for _ in 0..times {
            self.fake(BUTTON_PRESS_EVENT, button, 0, 0)?;
            self.fake(BUTTON_RELEASE_EVENT, button, 0, 0)?;
        }
        Ok(())
    }

    fn inject_one(&mut self, event: &Event) -> Result<(), InjectError> {
        match *event {
            Event::Key { usage, down, .. } => {
                let evdev = evdev_from_usage(usage).map_err(InjectError::UnknownUsage)?;
                let keycode = u8::try_from(evdev + X_KEYCODE_OFFSET)
                    .map_err(|_| InjectError::NotInjected(event.kind()))?;
                if down {
                    self.held_keys.insert(keycode);
                    self.fake(KEY_PRESS_EVENT, keycode, 0, 0)
                } else {
                    self.held_keys.remove(&keycode);
                    self.fake(KEY_RELEASE_EVENT, keycode, 0, 0)
                }
            }
            Event::Button { button, down } => {
                let button = POINTER_BUTTONS
                    .get(usize::from(button))
                    .ok_or(InjectError::UnknownButton(button))?;
                let kind = if down {
                    BUTTON_PRESS_EVENT
                } else {
                    BUTTON_RELEASE_EVENT
                };
                self.fake(kind, x_button(*button), 0, 0)
            }
            Event::Move { dx, dy } => {
                let clamp =
                    |delta: i32| delta.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
                self.fake(MOTION_NOTIFY_EVENT, RELATIVE_MOTION, clamp(dx), clamp(dy))
            }
            Event::Position { x, y, display } => {
                check_position(x, y, display, self.screen)?;
                let outside = || InjectError::OutsideScreen { x, y };
                let x = i16::try_from(x).map_err(|_| outside())?;
                let y = i16::try_from(y).map_err(|_| outside())?;
                self.fake(MOTION_NOTIFY_EVENT, ABSOLUTE_MOTION, x, y)
            }
            Event::Wheel { dx, dy } => {
                let vertical = whole_notches(&mut self.wheel.vertical, dy);
                let horizontal = whole_notches(&mut self.wheel.horizontal, dx);
                let up_or_down = if vertical > 0 { WHEEL_UP } else { WHEEL_DOWN };
                let right_or_left = if horizontal > 0 {
                    WHEEL_RIGHT
                } else {
                    WHEEL_LEFT
                };
                self.click(up_or_down, vertical.abs())?;
                self.click(right_or_left, horizontal.abs())
            }
            Event::DeviceRemoved {
                kind: RemovableDevice::Keyboard,
                ..
            } => {
                for keycode in core::mem::take(&mut self.held_keys) {
                    self.fake(KEY_RELEASE_EVENT, keycode, 0, 0)?;
                }
                Ok(())
            }
            _ => Err(InjectError::NotInjected(event.kind())),
        }
    }
}

impl InjectEvents for XTestInjector {
    fn inject(&mut self, event: &Event) -> Result<(), InjectError> {
        self.inject_one(event)?;
        self.connection.flush().map_err(x_failure)?;
        Ok(())
    }
}

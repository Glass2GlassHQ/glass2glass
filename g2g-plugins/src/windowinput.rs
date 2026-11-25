// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
}

impl PointerButton {
    pub fn evdev_code(self) -> u32 {
        // linux input-event-codes.h BTN_LEFT, BTN_RIGHT, BTN_MIDDLE, BTN_SIDE and BTN_EXTRA
        match self {
            PointerButton::Left => 0x110,
            PointerButton::Right => 0x111,
            PointerButton::Middle => 0x112,
            PointerButton::Back => 0x113,
            PointerButton::Forward => 0x114,
        }
    }

    pub fn from_evdev_code(code: u32) -> Option<Self> {
        [
            PointerButton::Left,
            PointerButton::Right,
            PointerButton::Middle,
            PointerButton::Back,
            PointerButton::Forward,
        ]
        .into_iter()
        .find(|button| button.evdev_code() == code)
    }
}

pub const WHEEL_UNITS_PER_NOTCH: i32 = 120;
// weston and mutter scroll 10 pixels per wheel notch
pub const WHEEL_PIXELS_PER_NOTCH: f64 = 10.0;

// wayland and X11 names for plain text, most specific first, which smithay's X11 window manager maps to atoms
pub const CLIPBOARD_TEXT_MIME_TYPES: [&str; 5] = [
    "text/plain;charset=utf-8",
    "UTF8_STRING",
    "text/plain",
    "TEXT",
    "STRING",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowInput {
    Focus(bool),
    Key { scancode: u32, down: bool },
    Pointer { x: u32, y: u32 },
    Button { button: PointerButton, down: bool },
    Wheel { dx: i32, dy: i32 },
    TouchDown { contact: u32, x: u32, y: u32 },
    TouchMove { contact: u32, x: u32, y: u32 },
    TouchUp { contact: u32 },
    TouchCancelAll,
}

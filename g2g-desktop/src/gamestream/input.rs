// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::fmt;

use g2g_plugins::windowinput::PointerButton;
use keycode::{KeyMap, KeyMappingCode};

use crate::injection::ScreenSize;
use crate::packet::{Event, RemovableDevice, TouchPhase, POINTER_BUTTONS};

const KEY_DOWN: u32 = 0x03;
const KEY_UP: u32 = 0x04;
const MOUSE_MOVE_ABSOLUTE: u32 = 0x05;
const MOUSE_MOVE_RELATIVE: u32 = 0x07;
const MOUSE_BUTTON_DOWN: u32 = 0x08;
const MOUSE_BUTTON_UP: u32 = 0x09;
const SCROLL_VERTICAL: u32 = 0x0A;
const SCROLL_HORIZONTAL: u32 = 0x5500_0001;
const UTF8_TEXT: u32 = 0x17;
const TOUCH: u32 = 0x5500_0002;
const PEN: u32 = 0x5500_0003;

// moonlight's LI_TOUCH_EVENT_* values, shared by touch and pen
const TOUCH_DOWN: u8 = 0x01;
const TOUCH_UP: u8 = 0x02;
const TOUCH_MOVE: u8 = 0x03;
const TOUCH_CANCEL: u8 = 0x04;
const TOUCH_CANCEL_ALL: u8 = 0x07;
// moonlight pens have no pointer id, and wayland touch ids are signed
const PEN_CONTACT: u32 = i32::MAX as u32;

const LENGTH_BYTES: usize = 4;
const MAGIC_BYTES: usize = 4;

// moonlight numbers buttons 1 left, 2 middle, 3 right, 4 and 5 the side buttons X1 and X2
const MOONLIGHT_BUTTONS: [(u8, PointerButton); 5] = [
    (1, PointerButton::Left),
    (2, PointerButton::Middle),
    (3, PointerButton::Right),
    (4, PointerButton::Back),
    (5, PointerButton::Forward),
];

const VIRTUAL_KEY_CODES: [(u8, KeyMappingCode); 123] = [
    (0x08, KeyMappingCode::Backspace),
    (0x09, KeyMappingCode::Tab),
    (0x0D, KeyMappingCode::Enter),
    (0x10, KeyMappingCode::ShiftLeft),
    (0x11, KeyMappingCode::ControlLeft),
    (0x12, KeyMappingCode::AltLeft),
    (0x13, KeyMappingCode::Pause),
    (0x14, KeyMappingCode::CapsLock),
    (0x1B, KeyMappingCode::Escape),
    (0x20, KeyMappingCode::Space),
    (0x21, KeyMappingCode::PageUp),
    (0x22, KeyMappingCode::PageDown),
    (0x23, KeyMappingCode::End),
    (0x24, KeyMappingCode::Home),
    (0x25, KeyMappingCode::ArrowLeft),
    (0x26, KeyMappingCode::ArrowUp),
    (0x27, KeyMappingCode::ArrowRight),
    (0x28, KeyMappingCode::ArrowDown),
    (0x2C, KeyMappingCode::PrintScreen),
    (0x2D, KeyMappingCode::Insert),
    (0x2E, KeyMappingCode::Delete),
    (0x2F, KeyMappingCode::Help),
    (b'0', KeyMappingCode::Digit0),
    (b'1', KeyMappingCode::Digit1),
    (b'2', KeyMappingCode::Digit2),
    (b'3', KeyMappingCode::Digit3),
    (b'4', KeyMappingCode::Digit4),
    (b'5', KeyMappingCode::Digit5),
    (b'6', KeyMappingCode::Digit6),
    (b'7', KeyMappingCode::Digit7),
    (b'8', KeyMappingCode::Digit8),
    (b'9', KeyMappingCode::Digit9),
    (b'A', KeyMappingCode::KeyA),
    (b'B', KeyMappingCode::KeyB),
    (b'C', KeyMappingCode::KeyC),
    (b'D', KeyMappingCode::KeyD),
    (b'E', KeyMappingCode::KeyE),
    (b'F', KeyMappingCode::KeyF),
    (b'G', KeyMappingCode::KeyG),
    (b'H', KeyMappingCode::KeyH),
    (b'I', KeyMappingCode::KeyI),
    (b'J', KeyMappingCode::KeyJ),
    (b'K', KeyMappingCode::KeyK),
    (b'L', KeyMappingCode::KeyL),
    (b'M', KeyMappingCode::KeyM),
    (b'N', KeyMappingCode::KeyN),
    (b'O', KeyMappingCode::KeyO),
    (b'P', KeyMappingCode::KeyP),
    (b'Q', KeyMappingCode::KeyQ),
    (b'R', KeyMappingCode::KeyR),
    (b'S', KeyMappingCode::KeyS),
    (b'T', KeyMappingCode::KeyT),
    (b'U', KeyMappingCode::KeyU),
    (b'V', KeyMappingCode::KeyV),
    (b'W', KeyMappingCode::KeyW),
    (b'X', KeyMappingCode::KeyX),
    (b'Y', KeyMappingCode::KeyY),
    (b'Z', KeyMappingCode::KeyZ),
    (0x5B, KeyMappingCode::MetaLeft),
    (0x5C, KeyMappingCode::MetaRight),
    (0x5D, KeyMappingCode::ContextMenu),
    (0x60, KeyMappingCode::Numpad0),
    (0x61, KeyMappingCode::Numpad1),
    (0x62, KeyMappingCode::Numpad2),
    (0x63, KeyMappingCode::Numpad3),
    (0x64, KeyMappingCode::Numpad4),
    (0x65, KeyMappingCode::Numpad5),
    (0x66, KeyMappingCode::Numpad6),
    (0x67, KeyMappingCode::Numpad7),
    (0x68, KeyMappingCode::Numpad8),
    (0x69, KeyMappingCode::Numpad9),
    (0x6A, KeyMappingCode::NumpadMultiply),
    (0x6B, KeyMappingCode::NumpadAdd),
    (0x6D, KeyMappingCode::NumpadSubtract),
    (0x6E, KeyMappingCode::NumpadDecimal),
    (0x6F, KeyMappingCode::NumpadDivide),
    (0x70, KeyMappingCode::F1),
    (0x71, KeyMappingCode::F2),
    (0x72, KeyMappingCode::F3),
    (0x73, KeyMappingCode::F4),
    (0x74, KeyMappingCode::F5),
    (0x75, KeyMappingCode::F6),
    (0x76, KeyMappingCode::F7),
    (0x77, KeyMappingCode::F8),
    (0x78, KeyMappingCode::F9),
    (0x79, KeyMappingCode::F10),
    (0x7A, KeyMappingCode::F11),
    (0x7B, KeyMappingCode::F12),
    (0x7C, KeyMappingCode::F13),
    (0x7D, KeyMappingCode::F14),
    (0x7E, KeyMappingCode::F15),
    (0x7F, KeyMappingCode::F16),
    (0x80, KeyMappingCode::F17),
    (0x81, KeyMappingCode::F18),
    (0x82, KeyMappingCode::F19),
    (0x83, KeyMappingCode::F20),
    (0x84, KeyMappingCode::F21),
    (0x85, KeyMappingCode::F22),
    (0x86, KeyMappingCode::F23),
    (0x87, KeyMappingCode::F24),
    (0x90, KeyMappingCode::NumLock),
    (0x91, KeyMappingCode::ScrollLock),
    (0xA0, KeyMappingCode::ShiftLeft),
    (0xA1, KeyMappingCode::ShiftRight),
    (0xA2, KeyMappingCode::ControlLeft),
    (0xA3, KeyMappingCode::ControlRight),
    (0xA4, KeyMappingCode::AltLeft),
    (0xA5, KeyMappingCode::AltRight),
    (0xAD, KeyMappingCode::AudioVolumeMute),
    (0xAE, KeyMappingCode::AudioVolumeDown),
    (0xAF, KeyMappingCode::AudioVolumeUp),
    (0xBA, KeyMappingCode::Semicolon),
    (0xBB, KeyMappingCode::Equal),
    (0xBC, KeyMappingCode::Comma),
    (0xBD, KeyMappingCode::Minus),
    (0xBE, KeyMappingCode::Period),
    (0xBF, KeyMappingCode::Slash),
    (0xC0, KeyMappingCode::Backquote),
    (0xDB, KeyMappingCode::BracketLeft),
    (0xDC, KeyMappingCode::Backslash),
    (0xDD, KeyMappingCode::BracketRight),
    (0xDE, KeyMappingCode::Quote),
    (0xE2, KeyMappingCode::IntlBackslash),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InputError {
    Truncated,
    LengthMismatch,
    Unsupported(u32),
    UnknownKey(u8),
    UnknownButton(u8),
    NoCaptureSize,
    EmptyReferenceSize,
    NotUtf8,
    UnsupportedTouch(u8),
    NotFinite,
}

impl InputError {
    pub(crate) fn is_malformed(self) -> bool {
        matches!(
            self,
            Self::Truncated | Self::LengthMismatch | Self::NotUtf8 | Self::NotFinite
        )
    }
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("input message is truncated"),
            Self::LengthMismatch => f.write_str("input length does not match the message"),
            Self::Unsupported(magic) => write!(f, "input event {magic:#010x} is not injected"),
            Self::UnknownKey(key) => write!(f, "virtual key {key:#04x} has no HID usage"),
            Self::UnknownButton(button) => write!(f, "mouse button {button} is not injected"),
            Self::NoCaptureSize => {
                f.write_str("absolute position before the capture size is known")
            }
            Self::EmptyReferenceSize => f.write_str("absolute position with a zero reference size"),
            Self::NotUtf8 => f.write_str("text input is not UTF-8"),
            Self::UnsupportedTouch(event) => write!(f, "touch event {event:#04x} is not injected"),
            Self::NotFinite => f.write_str("touch position is not a finite number"),
        }
    }
}

fn virtual_key_usage(virtual_key: u8) -> Option<u16> {
    VIRTUAL_KEY_CODES
        .iter()
        .find(|(key, _)| *key == virtual_key)
        .map(|(_, code)| KeyMap::from(*code).usb)
}

fn be_i16(payload: &[u8], offset: usize) -> Result<i16, InputError> {
    let bytes = payload
        .get(offset..offset + 2)
        .ok_or(InputError::Truncated)?;
    Ok(i16::from_be_bytes([bytes[0], bytes[1]]))
}

fn le_u32(payload: &[u8], offset: usize) -> Result<u32, InputError> {
    let bytes = payload
        .get(offset..offset + 4)
        .ok_or(InputError::Truncated)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn byte(payload: &[u8], offset: usize) -> Result<u8, InputError> {
    payload.get(offset).copied().ok_or(InputError::Truncated)
}

fn scaled(position: i16, reference: i16, captured: u16) -> Result<i32, InputError> {
    if reference <= 0 {
        return Err(InputError::EmptyReferenceSize);
    }
    let scaled = i64::from(position) * i64::from(captured) / i64::from(reference);
    Ok(scaled as i32)
}

fn key(payload: &[u8], down: bool) -> Result<Event, InputError> {
    // the key code is little-endian with the virtual key in its low byte
    let virtual_key = byte(payload, 1)?;
    let usage = virtual_key_usage(virtual_key).ok_or(InputError::UnknownKey(virtual_key))?;
    Ok(Event::Key {
        device: 0,
        usage,
        down,
    })
}

fn button(payload: &[u8], down: bool) -> Result<Event, InputError> {
    let moonlight_button = byte(payload, 0)?;
    let (_, pointer_button) = MOONLIGHT_BUTTONS
        .iter()
        .find(|(number, _)| *number == moonlight_button)
        .ok_or(InputError::UnknownButton(moonlight_button))?;
    let button = POINTER_BUTTONS
        .iter()
        .position(|known| known == pointer_button)
        .expect("POINTER_BUTTONS lists every PointerButton");
    Ok(Event::Button {
        button: button as u8,
        down,
    })
}

fn position(payload: &[u8], screen: Option<ScreenSize>) -> Result<Event, InputError> {
    let x = be_i16(payload, 0)?;
    let y = be_i16(payload, 2)?;
    let reference_width = be_i16(payload, 6)?;
    let reference_height = be_i16(payload, 8)?;
    let screen = screen.ok_or(InputError::NoCaptureSize)?;
    Ok(Event::Position {
        x: scaled(x, reference_width, screen.width)?,
        y: scaled(y, reference_height, screen.height)?,
        display: 0,
    })
}

fn normalized(fraction: f32, captured: u16) -> Result<i32, InputError> {
    if !fraction.is_finite() {
        return Err(InputError::NotFinite);
    }
    // moonlight sends touch positions as fractions from 0 to 1 inclusive of the video's size
    Ok((fraction * f32::from(captured.saturating_sub(1))) as i32)
}

struct ContactReport {
    event: u8,
    contact: u32,
    x: f32,
    y: f32,
}

fn touch_event(report: ContactReport, screen: Option<ScreenSize>) -> Result<Event, InputError> {
    let ContactReport {
        event,
        contact,
        x,
        y,
    } = report;
    let position = || -> Result<(i32, i32), InputError> {
        let screen = screen.ok_or(InputError::NoCaptureSize)?;
        Ok((normalized(x, screen.width)?, normalized(y, screen.height)?))
    };
    let phase = match event {
        TOUCH_DOWN => {
            let (x, y) = position()?;
            TouchPhase::Down { x, y, display: 0 }
        }
        TOUCH_MOVE => {
            let (x, y) = position()?;
            TouchPhase::Move { x, y, display: 0 }
        }
        TOUCH_UP => TouchPhase::Up,
        TOUCH_CANCEL => TouchPhase::Cancel,
        TOUCH_CANCEL_ALL => {
            return Ok(Event::DeviceRemoved {
                kind: RemovableDevice::Touchscreen,
                device: 0,
            })
        }
        other => return Err(InputError::UnsupportedTouch(other)),
    };
    Ok(Event::Touch { contact, phase })
}

pub(crate) fn input_event(body: &[u8], screen: Option<ScreenSize>) -> Result<Event, InputError> {
    let length = body.get(..LENGTH_BYTES).ok_or(InputError::Truncated)?;
    let length = u32::from_be_bytes([length[0], length[1], length[2], length[3]]) as usize;
    let event = &body[LENGTH_BYTES..];
    if length != event.len() {
        return Err(InputError::LengthMismatch);
    }
    let magic = event.get(..MAGIC_BYTES).ok_or(InputError::Truncated)?;
    let magic = u32::from_le_bytes([magic[0], magic[1], magic[2], magic[3]]);
    let payload = &event[MAGIC_BYTES..];
    match magic {
        KEY_DOWN => key(payload, true),
        KEY_UP => key(payload, false),
        MOUSE_MOVE_RELATIVE => Ok(Event::Move {
            dx: i32::from(be_i16(payload, 0)?),
            dy: i32::from(be_i16(payload, 2)?),
        }),
        MOUSE_MOVE_ABSOLUTE => position(payload, screen),
        MOUSE_BUTTON_DOWN => button(payload, true),
        MOUSE_BUTTON_UP => button(payload, false),
        // moonlight scrolls in windows wheel units, 120 per notch, up and right positive
        SCROLL_VERTICAL => Ok(Event::Wheel {
            dx: 0,
            dy: i32::from(be_i16(payload, 0)?),
        }),
        SCROLL_HORIZONTAL => Ok(Event::Wheel {
            dx: i32::from(be_i16(payload, 0)?),
            dy: 0,
        }),
        TOUCH => touch_event(
            ContactReport {
                event: byte(payload, 0)?,
                contact: le_u32(payload, 4)?,
                x: f32::from_bits(le_u32(payload, 8)?),
                y: f32::from_bits(le_u32(payload, 12)?),
            },
            screen,
        ),
        PEN => touch_event(
            ContactReport {
                event: byte(payload, 0)?,
                contact: PEN_CONTACT,
                x: f32::from_bits(le_u32(payload, 4)?),
                y: f32::from_bits(le_u32(payload, 8)?),
            },
            screen,
        ),
        UTF8_TEXT => String::from_utf8(payload.to_vec())
            .map(Event::Text)
            .map_err(|_| InputError::NotUtf8),
        other => Err(InputError::Unsupported(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // HID keyboard usages for A, Enter and left shift
    const USAGE_A: u16 = 0x04;
    const USAGE_ENTER: u16 = 0x28;
    const USAGE_SHIFT_LEFT: u16 = 0xE1;
    const SCREEN: ScreenSize = ScreenSize {
        width: 1920,
        height: 1080,
    };

    fn message(magic: u32, payload: &[u8]) -> Vec<u8> {
        let length = (MAGIC_BYTES + payload.len()) as u32;
        let mut body = Vec::from(length.to_be_bytes());
        body.extend_from_slice(&magic.to_le_bytes());
        body.extend_from_slice(payload);
        body
    }

    #[test]
    fn key_events_map_virtual_keys_to_usages() {
        // moonlight sets the high byte of the key code to 0x80
        let a_down = message(KEY_DOWN, &[0, b'A', 0x80, 0, 0, 0]);
        assert_eq!(
            input_event(&a_down, None),
            Ok(Event::Key {
                device: 0,
                usage: USAGE_A,
                down: true
            })
        );
        let enter_up = message(KEY_UP, &[0, 0x0D, 0x80, 0, 0, 0]);
        assert_eq!(
            input_event(&enter_up, None),
            Ok(Event::Key {
                device: 0,
                usage: USAGE_ENTER,
                down: false
            })
        );
        assert_eq!(virtual_key_usage(0xA0), Some(USAGE_SHIFT_LEFT));
        assert_eq!(
            input_event(&message(KEY_DOWN, &[0, 0xFF, 0x80, 0, 0, 0]), None),
            Err(InputError::UnknownKey(0xFF))
        );
    }

    #[test]
    fn every_virtual_key_has_a_keyboard_usage() {
        for (virtual_key, code) in VIRTUAL_KEY_CODES {
            let usage = virtual_key_usage(virtual_key).expect("listed keys map");
            assert!(
                crate::hid::evdev_from_usage(usage).is_ok(),
                "{code:?} usage {usage:#x}"
            );
        }
    }

    #[test]
    fn mouse_events_become_pointer_events() {
        assert_eq!(
            input_event(&message(MOUSE_BUTTON_DOWN, &[3]), None),
            Ok(Event::Button {
                button: 1,
                down: true
            })
        );
        assert_eq!(
            input_event(&message(MOUSE_BUTTON_UP, &[4]), None),
            Ok(Event::Button {
                button: 3,
                down: false
            })
        );
        assert_eq!(
            input_event(&message(MOUSE_BUTTON_UP, &[6]), None),
            Err(InputError::UnknownButton(6))
        );
        let relative = [(-5i16).to_be_bytes(), 7i16.to_be_bytes()].concat();
        assert_eq!(
            input_event(&message(MOUSE_MOVE_RELATIVE, &relative), None),
            Ok(Event::Move { dx: -5, dy: 7 })
        );
        let scroll = [(-120i16).to_be_bytes(), [0; 2], [0; 2]].concat();
        assert_eq!(
            input_event(&message(SCROLL_VERTICAL, &scroll), None),
            Ok(Event::Wheel { dx: 0, dy: -120 })
        );
        assert_eq!(
            input_event(&message(SCROLL_HORIZONTAL, &120i16.to_be_bytes()), None),
            Ok(Event::Wheel { dx: 120, dy: 0 })
        );
    }

    #[test]
    fn absolute_positions_scale_to_the_capture() {
        let absolute = [
            640i16.to_be_bytes(),
            360i16.to_be_bytes(),
            [0; 2],
            1280i16.to_be_bytes(),
            720i16.to_be_bytes(),
        ]
        .concat();
        assert_eq!(
            input_event(&message(MOUSE_MOVE_ABSOLUTE, &absolute), Some(SCREEN)),
            Ok(Event::Position {
                x: 960,
                y: 540,
                display: 0
            })
        );
        assert_eq!(
            input_event(&message(MOUSE_MOVE_ABSOLUTE, &absolute), None),
            Err(InputError::NoCaptureSize)
        );
    }

    #[test]
    fn utf8_text_becomes_a_text_event() {
        assert_eq!(
            input_event(&message(UTF8_TEXT, "caf\u{e9}".as_bytes()), None),
            Ok(Event::Text("caf\u{e9}".to_string()))
        );
        assert_eq!(
            input_event(&message(UTF8_TEXT, &[0xff]), None),
            Err(InputError::NotUtf8)
        );
    }

    fn touch_payload(event: u8, contact: u32, x: f32, y: f32) -> Vec<u8> {
        let mut payload = vec![event, 0, 0, 0];
        payload.extend_from_slice(&contact.to_le_bytes());
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
        payload.resize(28, 0);
        payload
    }

    fn pen_payload(event: u8, x: f32, y: f32) -> Vec<u8> {
        let mut payload = vec![event, 1, 0, 0];
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
        payload.resize(28, 0);
        payload
    }

    #[test]
    fn touch_and_pen_become_touch_contacts_on_the_capture() {
        let down = message(TOUCH, &touch_payload(TOUCH_DOWN, 9, 0.5, 1.0));
        assert_eq!(
            input_event(&down, Some(SCREEN)),
            Ok(Event::Touch {
                contact: 9,
                phase: TouchPhase::Down {
                    x: 959,
                    y: 1079,
                    display: 0
                }
            })
        );
        let up = message(TOUCH, &touch_payload(TOUCH_UP, 9, 0.5, 1.0));
        assert_eq!(
            input_event(&up, None),
            Ok(Event::Touch {
                contact: 9,
                phase: TouchPhase::Up
            })
        );
        let pen = message(PEN, &pen_payload(TOUCH_MOVE, 0.0, 0.5));
        assert_eq!(
            input_event(&pen, Some(SCREEN)),
            Ok(Event::Touch {
                contact: PEN_CONTACT,
                phase: TouchPhase::Move {
                    x: 0,
                    y: 539,
                    display: 0
                }
            })
        );
        let cancel_all = message(TOUCH, &touch_payload(TOUCH_CANCEL_ALL, 0, 0.0, 0.0));
        assert_eq!(
            input_event(&cancel_all, None),
            Ok(Event::DeviceRemoved {
                kind: RemovableDevice::Touchscreen,
                device: 0
            })
        );
        // moonlight's LI_TOUCH_EVENT_HOVER
        let hover = message(PEN, &pen_payload(0x00, 0.5, 0.5));
        assert_eq!(
            input_event(&hover, Some(SCREEN)),
            Err(InputError::UnsupportedTouch(0x00))
        );
        let not_finite = message(TOUCH, &touch_payload(TOUCH_DOWN, 9, f32::NAN, 0.5));
        assert_eq!(
            input_event(&not_finite, Some(SCREEN)),
            Err(InputError::NotFinite)
        );
    }

    #[test]
    fn malformed_input_fails() {
        let mut short = message(MOUSE_MOVE_RELATIVE, &[0, 1]);
        assert_eq!(input_event(&short, None), Err(InputError::Truncated));
        short.push(0);
        short[3] += 2;
        assert_eq!(input_event(&short, None), Err(InputError::LengthMismatch));
        assert_eq!(input_event(&[0, 0], None), Err(InputError::Truncated));
    }
}

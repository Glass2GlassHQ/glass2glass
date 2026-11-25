// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::fmt;

use keycode::{KeyMap, KeyMapping, KeyMappingCode};

const KEYBOARD_PAGE: u16 = 0x07;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScancodeSet {
    Evdev,
    Windows,
    Mac,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownUsage(pub u16);

impl fmt::Display for UnknownUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HID keyboard usage {:#06x} has no key", self.0)
    }
}

impl std::error::Error for UnknownUsage {}

pub fn usage_from_scancode(set: ScancodeSet, scancode: u32) -> Option<u16> {
    let scancode = u16::try_from(scancode).ok()?;
    let mapping = match set {
        ScancodeSet::Evdev => KeyMapping::Evdev(scancode),
        ScancodeSet::Windows => KeyMapping::Win(scancode),
        ScancodeSet::Mac => KeyMapping::Mac(scancode),
    };
    let key = KeyMap::from_key_mapping(mapping).ok()?;
    // keycode's `usb` field has no page, and a consumer-page key shares low bits with keyboard keys
    let on_keyboard_page = KeyMap::from_usb_code(KEYBOARD_PAGE, key.usb).is_ok_and(|k| k == key);
    (on_keyboard_page && key.usb != 0).then_some(key.usb)
}

pub fn evdev_from_usage(usage: u16) -> Result<u16, UnknownUsage> {
    // keycode maps reserved and unassigned usages to evdev code 0
    KeyMap::from_usb_code(KEYBOARD_PAGE, usage)
        .ok()
        .map(|key| key.evdev)
        .filter(|&code| code != 0)
        .ok_or(UnknownUsage(usage))
}

pub fn windows_scancode_from_usage(usage: u16) -> Result<u16, UnknownUsage> {
    // keycode gives the usb error usages windows codes too, and those rows have no browser key code
    KeyMap::from_usb_code(KEYBOARD_PAGE, usage)
        .ok()
        .filter(|key| key.code.is_some() && key.win != 0)
        .map(|key| key.win)
        .ok_or(UnknownUsage(usage))
}

// each key of a US keyboard with the character it types alone and with shift
const US_LAYOUT: [(KeyMappingCode, char, Option<char>); 50] = [
    (KeyMappingCode::KeyA, 'a', Some('A')),
    (KeyMappingCode::KeyB, 'b', Some('B')),
    (KeyMappingCode::KeyC, 'c', Some('C')),
    (KeyMappingCode::KeyD, 'd', Some('D')),
    (KeyMappingCode::KeyE, 'e', Some('E')),
    (KeyMappingCode::KeyF, 'f', Some('F')),
    (KeyMappingCode::KeyG, 'g', Some('G')),
    (KeyMappingCode::KeyH, 'h', Some('H')),
    (KeyMappingCode::KeyI, 'i', Some('I')),
    (KeyMappingCode::KeyJ, 'j', Some('J')),
    (KeyMappingCode::KeyK, 'k', Some('K')),
    (KeyMappingCode::KeyL, 'l', Some('L')),
    (KeyMappingCode::KeyM, 'm', Some('M')),
    (KeyMappingCode::KeyN, 'n', Some('N')),
    (KeyMappingCode::KeyO, 'o', Some('O')),
    (KeyMappingCode::KeyP, 'p', Some('P')),
    (KeyMappingCode::KeyQ, 'q', Some('Q')),
    (KeyMappingCode::KeyR, 'r', Some('R')),
    (KeyMappingCode::KeyS, 's', Some('S')),
    (KeyMappingCode::KeyT, 't', Some('T')),
    (KeyMappingCode::KeyU, 'u', Some('U')),
    (KeyMappingCode::KeyV, 'v', Some('V')),
    (KeyMappingCode::KeyW, 'w', Some('W')),
    (KeyMappingCode::KeyX, 'x', Some('X')),
    (KeyMappingCode::KeyY, 'y', Some('Y')),
    (KeyMappingCode::KeyZ, 'z', Some('Z')),
    (KeyMappingCode::Digit1, '1', Some('!')),
    (KeyMappingCode::Digit2, '2', Some('@')),
    (KeyMappingCode::Digit3, '3', Some('#')),
    (KeyMappingCode::Digit4, '4', Some('$')),
    (KeyMappingCode::Digit5, '5', Some('%')),
    (KeyMappingCode::Digit6, '6', Some('^')),
    (KeyMappingCode::Digit7, '7', Some('&')),
    (KeyMappingCode::Digit8, '8', Some('*')),
    (KeyMappingCode::Digit9, '9', Some('(')),
    (KeyMappingCode::Digit0, '0', Some(')')),
    (KeyMappingCode::Minus, '-', Some('_')),
    (KeyMappingCode::Equal, '=', Some('+')),
    (KeyMappingCode::BracketLeft, '[', Some('{')),
    (KeyMappingCode::BracketRight, ']', Some('}')),
    (KeyMappingCode::Backslash, '\\', Some('|')),
    (KeyMappingCode::Semicolon, ';', Some(':')),
    (KeyMappingCode::Quote, '\'', Some('"')),
    (KeyMappingCode::Backquote, '`', Some('~')),
    (KeyMappingCode::Comma, ',', Some('<')),
    (KeyMappingCode::Period, '.', Some('>')),
    (KeyMappingCode::Slash, '/', Some('?')),
    (KeyMappingCode::Space, ' ', None),
    (KeyMappingCode::Enter, '\n', None),
    (KeyMappingCode::Tab, '\t', None),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsLayoutKey {
    pub usage: u16,
    pub shifted: bool,
}

pub fn us_layout_key(character: char) -> Option<UsLayoutKey> {
    US_LAYOUT.iter().find_map(|(code, alone, with_shift)| {
        let shifted = if *alone == character {
            false
        } else if *with_shift == Some(character) {
            true
        } else {
            return None;
        };
        Some(UsLayoutKey {
            usage: KeyMap::from(*code).usb,
            shifted,
        })
    })
}

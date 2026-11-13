// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::fmt;

use keycode::{KeyMap, KeyMapping};

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

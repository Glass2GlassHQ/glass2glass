// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_desktop::hid::{
    evdev_from_usage, us_layout_key, usage_from_scancode, windows_scancode_from_usage, ScancodeSet,
    UnknownUsage,
};
use g2g_desktop::injection::{type_us_layout, InjectError, InjectEvents};
use g2g_desktop::packet::Event;

struct PlatformKey {
    name: &'static str,
    usage: u16,
    evdev: u32,
    windows: u32,
    mac: u32,
}

// values from linux input-event-codes.h, windows scancode set 1 and macOS Events.h kVK_* codes
const KEYS_ON_EVERY_PLATFORM: [PlatformKey; 4] = [
    PlatformKey {
        name: "A",
        usage: 0x04,
        evdev: 30,
        windows: 0x1e,
        mac: 0x00,
    },
    PlatformKey {
        name: "Escape",
        usage: 0x29,
        evdev: 1,
        windows: 0x01,
        mac: 0x35,
    },
    PlatformKey {
        name: "ArrowLeft",
        usage: 0x50,
        evdev: 105,
        windows: 0xe04b,
        mac: 0x7b,
    },
    PlatformKey {
        name: "AudioVolumeUp",
        usage: 0x80,
        evdev: 115,
        windows: 0xe030,
        mac: 0x48,
    },
];

#[test]
fn a_key_maps_to_one_usage_from_every_platform() {
    for key in &KEYS_ON_EVERY_PLATFORM {
        for (set, scancode) in [
            (ScancodeSet::Evdev, key.evdev),
            (ScancodeSet::Windows, key.windows),
            (ScancodeSet::Mac, key.mac),
        ] {
            assert_eq!(
                usage_from_scancode(set, scancode),
                Some(key.usage),
                "{} from {set:?}",
                key.name
            );
        }
        assert_eq!(
            evdev_from_usage(key.usage),
            Ok(key.evdev as u16),
            "{}",
            key.name
        );
        assert_eq!(
            windows_scancode_from_usage(key.usage),
            Ok(key.windows as u16),
            "{}",
            key.name
        );
    }
}

#[test]
fn every_injectable_usage_captures_back_to_itself() {
    for usage in 0..=u16::MAX {
        let Ok(evdev) = evdev_from_usage(usage) else {
            continue;
        };
        assert_eq!(
            usage_from_scancode(ScancodeSet::Evdev, u32::from(evdev)),
            Some(usage),
            "usage {usage:#06x} injects as evdev {evdev}"
        );
    }
}

#[test]
fn every_windows_injectable_usage_captures_back_to_itself() {
    for usage in 0..=u16::MAX {
        let Ok(scancode) = windows_scancode_from_usage(usage) else {
            continue;
        };
        assert_eq!(
            usage_from_scancode(ScancodeSet::Windows, u32::from(scancode)),
            Some(usage),
            "usage {usage:#06x} injects as windows scancode {scancode:#06x}"
        );
    }
}

#[test]
fn unknown_usage_fails_instead_of_mapping_to_zero() {
    const RESERVED: u16 = 0x00;
    const ERROR_ROLL_OVER: u16 = 0x01;
    const PAST_KEYBOARD_PAGE: u16 = 0xffff;
    for usage in [RESERVED, ERROR_ROLL_OVER, PAST_KEYBOARD_PAGE] {
        assert_eq!(evdev_from_usage(usage), Err(UnknownUsage(usage)));
        assert_eq!(windows_scancode_from_usage(usage), Err(UnknownUsage(usage)));
    }
}

#[test]
fn consumer_page_key_is_not_a_keyboard_usage() {
    // evdev KEY_PLAYPAUSE is HID consumer usage 0xcd, which keycode stores without its page
    const KEY_PLAYPAUSE: u32 = 164;
    assert_eq!(usage_from_scancode(ScancodeSet::Evdev, KEY_PLAYPAUSE), None);
}

#[test]
fn scancode_wider_than_sixteen_bits_has_no_usage() {
    let too_wide = u32::from(u16::MAX) + 1;
    assert_eq!(usage_from_scancode(ScancodeSet::Windows, too_wide), None);
}

#[test]
fn every_printable_ascii_character_has_an_injectable_us_key() {
    for character in (' '..='~').chain(['\n', '\t']) {
        let key = us_layout_key(character).unwrap_or_else(|| panic!("{character:?} has no key"));
        assert!(evdev_from_usage(key.usage).is_ok(), "{character:?}");
    }
}

#[derive(Default)]
struct RecordedKeys(Vec<(u16, bool)>);

impl InjectEvents for RecordedKeys {
    fn inject(&mut self, event: &Event) -> Result<(), InjectError> {
        let Event::Key { usage, down, .. } = *event else {
            panic!("typing sent {event:?}");
        };
        self.0.push((usage, down));
        Ok(())
    }
}

#[test]
fn text_types_us_keys_with_shift_and_reports_the_rest() {
    // HID keyboard usages for A, 1 and left shift
    const A: u16 = 0x04;
    const DIGIT_1: u16 = 0x1e;
    const SHIFT: u16 = 0xe1;
    let mut keys = RecordedKeys::default();
    let typed = type_us_layout(&mut keys, "a\u{e9}!");
    assert!(matches!(typed, Err(InjectError::NoUsLayoutKey(untyped)) if untyped == "\u{e9}"));
    assert_eq!(
        keys.0,
        [
            (A, true),
            (A, false),
            (SHIFT, true),
            (DIGIT_1, true),
            (DIGIT_1, false),
            (SHIFT, false),
        ]
    );
}

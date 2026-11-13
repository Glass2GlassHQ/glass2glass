// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_desktop::hid::{evdev_from_usage, usage_from_scancode, ScancodeSet, UnknownUsage};

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
fn unknown_usage_fails_instead_of_mapping_to_zero() {
    const RESERVED: u16 = 0x00;
    const ERROR_ROLL_OVER: u16 = 0x01;
    const PAST_KEYBOARD_PAGE: u16 = 0xffff;
    for usage in [RESERVED, ERROR_ROLL_OVER, PAST_KEYBOARD_PAGE] {
        assert_eq!(evdev_from_usage(usage), Err(UnknownUsage(usage)));
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

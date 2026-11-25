// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(target_os = "linux")]

use std::fs::OpenOptions;
use std::time::{Duration, Instant};

use evdev::{AbsoluteAxisCode, Device, EventSummary, KeyCode, SynchronizationCode};
use g2g_desktop::injection::{InjectEvents, ScreenSize, TOUCH_SLOTS};
use g2g_desktop::packet::{Event, RemovableDevice, TouchPhase};
use g2g_desktop::uinput::{Injector, TOUCHSCREEN_NAME};

const SCREEN: ScreenSize = ScreenSize {
    width: 320,
    height: 240,
};
const DEVICE_APPEARS: Duration = Duration::from_secs(5);
// linux marks a slot with no contact by tracking id -1
const NO_CONTACT: i32 = -1;

fn touchscreen() -> Device {
    let deadline = Instant::now() + DEVICE_APPEARS;
    loop {
        let found = evdev::enumerate()
            .map(|(_, device)| device)
            .find(|device| device.name() == Some(TOUCHSCREEN_NAME));
        if let Some(device) = found {
            return device;
        }
        assert!(Instant::now() < deadline, "{TOUCHSCREEN_NAME} appears");
        std::thread::sleep(Duration::from_millis(50));
    }
}

struct Slots {
    current: usize,
    tracking: [i32; TOUCH_SLOTS],
    position: [(i32, i32); TOUCH_SLOTS],
    touching: bool,
}

impl Slots {
    fn new() -> Self {
        Self {
            current: 0,
            tracking: [NO_CONTACT; TOUCH_SLOTS],
            position: [(0, 0); TOUCH_SLOTS],
            touching: false,
        }
    }

    fn read_report(&mut self, device: &mut Device) {
        loop {
            for event in device.fetch_events().expect("reads the touchscreen") {
                match event.destructure() {
                    EventSummary::AbsoluteAxis(_, AbsoluteAxisCode::ABS_MT_SLOT, slot) => {
                        self.current = slot as usize;
                    }
                    EventSummary::AbsoluteAxis(_, AbsoluteAxisCode::ABS_MT_TRACKING_ID, id) => {
                        self.tracking[self.current] = id;
                    }
                    EventSummary::AbsoluteAxis(_, AbsoluteAxisCode::ABS_MT_POSITION_X, x) => {
                        self.position[self.current].0 = x;
                    }
                    EventSummary::AbsoluteAxis(_, AbsoluteAxisCode::ABS_MT_POSITION_Y, y) => {
                        self.position[self.current].1 = y;
                    }
                    EventSummary::Key(_, KeyCode::BTN_TOUCH, value) => self.touching = value == 1,
                    EventSummary::Synchronization(_, SynchronizationCode::SYN_REPORT, _) => {
                        return;
                    }
                    _ => {}
                }
            }
        }
    }
}

#[test]
fn touch_contacts_reach_a_multitouch_device() {
    if OpenOptions::new().write(true).open("/dev/uinput").is_err() {
        return;
    }
    let mut injector = Injector::new(SCREEN).expect("creates the uinput devices");
    let mut device = touchscreen();
    // linux EVIOCGRAB keeps these touches from the desktop session
    device.grab().expect("grabs the touchscreen");
    let mut slots = Slots::new();
    let mut inject = |event: Event, slots: &mut Slots| {
        injector.inject(&event).expect("injects");
        slots.read_report(&mut device);
    };
    let down = |contact, x, y| Event::Touch {
        contact,
        phase: TouchPhase::Down { x, y, display: 0 },
    };

    inject(down(500, 10, 20), &mut slots);
    assert!(slots.touching);
    assert_ne!(slots.tracking[0], NO_CONTACT);
    assert_eq!(slots.position[0], (10, 20));

    inject(down(600, 30, 40), &mut slots);
    assert_ne!(slots.tracking[1], NO_CONTACT);
    assert_ne!(slots.tracking[1], slots.tracking[0]);
    assert_eq!(slots.position[1], (30, 40));

    let moved = Event::Touch {
        contact: 500,
        phase: TouchPhase::Move {
            x: 11,
            y: 21,
            display: 0,
        },
    };
    inject(moved, &mut slots);
    assert_eq!(slots.position[0], (11, 21));

    let up = Event::Touch {
        contact: 500,
        phase: TouchPhase::Up,
    };
    inject(up, &mut slots);
    assert_eq!(slots.tracking[0], NO_CONTACT);
    assert!(slots.touching, "the second contact is still down");

    let removed = Event::DeviceRemoved {
        kind: RemovableDevice::Touchscreen,
        device: 0,
    };
    inject(removed, &mut slots);
    assert_eq!(slots.tracking[1], NO_CONTACT);
    assert!(!slots.touching);
}

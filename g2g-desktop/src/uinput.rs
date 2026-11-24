// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::fmt;
use std::io;

use evdev::uinput::VirtualDevice;
use evdev::{
    AbsInfo, AbsoluteAxisCode, AbsoluteAxisEvent, AttributeSet, InputEvent, KeyCode, KeyEvent,
    RelativeAxisCode, RelativeAxisEvent, UinputAbsSetup,
};

use crate::hid::evdev_from_usage;
use crate::injection::{check_position, InjectError, InjectEvents, ScreenSize};
use crate::packet::{Event, RemovableDevice};

pub const KEYBOARD_NAME_PREFIX: &str = "g2g-desktop keyboard";

// linux reports one wheel notch as 120 high-resolution units
const HI_RES_UNITS_PER_NOTCH: i32 = 120;

const BUTTONS: [KeyCode; 3] = [KeyCode::BTN_LEFT, KeyCode::BTN_RIGHT, KeyCode::BTN_MIDDLE];

fn keyboard_keys() -> AttributeSet<KeyCode> {
    let mut keys = AttributeSet::new();
    for code in (0..=u16::MAX).filter_map(|usage| evdev_from_usage(usage).ok()) {
        keys.insert(KeyCode(code));
    }
    keys
}

fn buttons() -> AttributeSet<KeyCode> {
    let mut set = AttributeSet::new();
    for button in BUTTONS {
        set.insert(button);
    }
    set
}

const MOUSE_NAME: &str = "g2g-desktop mouse";

fn build_mouse() -> io::Result<VirtualDevice> {
    let mut axes = AttributeSet::new();
    for axis in [
        RelativeAxisCode::REL_X,
        RelativeAxisCode::REL_Y,
        RelativeAxisCode::REL_WHEEL,
        RelativeAxisCode::REL_HWHEEL,
        RelativeAxisCode::REL_WHEEL_HI_RES,
        RelativeAxisCode::REL_HWHEEL_HI_RES,
    ] {
        axes.insert(axis);
    }
    VirtualDevice::builder()?
        .name(MOUSE_NAME)
        .with_keys(&buttons())?
        .with_relative_axes(&axes)?
        .build()
}

const TABLET_NAME: &str = "g2g-desktop absolute pointer";

fn build_tablet(screen: ScreenSize) -> io::Result<VirtualDevice> {
    let axis = |code, extent: u16| {
        UinputAbsSetup::new(code, AbsInfo::new(0, 0, i32::from(extent) - 1, 0, 0, 0))
    };
    // udev classifies an absolute device as a mouse only when it also has mouse buttons
    VirtualDevice::builder()?
        .name(TABLET_NAME)
        .with_keys(&buttons())?
        .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_X, screen.width))?
        .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_Y, screen.height))?
        .build()
}

#[derive(Debug, Default)]
struct WheelRemainder {
    vertical: i32,
    horizontal: i32,
}

fn whole_notches(remainder: &mut i32, hi_res: i32) -> i32 {
    *remainder += hi_res;
    let notches = *remainder / HI_RES_UNITS_PER_NOTCH;
    *remainder -= notches * HI_RES_UNITS_PER_NOTCH;
    notches
}

pub struct Injector {
    screen: ScreenSize,
    keyboard_keys: AttributeSet<KeyCode>,
    keyboards: BTreeMap<u8, VirtualDevice>,
    mouse: VirtualDevice,
    tablet: VirtualDevice,
    wheel: WheelRemainder,
}

impl fmt::Debug for Injector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Injector")
            .field("screen", &self.screen)
            .field("keyboards", &self.keyboards.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Injector {
    pub fn new(screen: ScreenSize) -> Result<Self, InjectError> {
        Ok(Self {
            screen,
            keyboard_keys: keyboard_keys(),
            keyboards: BTreeMap::new(),
            mouse: build_mouse()?,
            tablet: build_tablet(screen)?,
            wheel: WheelRemainder::default(),
        })
    }
}

impl InjectEvents for Injector {
    fn inject(&mut self, event: &Event) -> Result<(), InjectError> {
        match *event {
            Event::Key {
                device,
                usage,
                down,
            } => {
                let code = evdev_from_usage(usage).map_err(InjectError::UnknownUsage)?;
                let key = *KeyEvent::new(KeyCode(code), i32::from(down));
                self.keyboard(device)?.emit(&[key])?;
            }
            Event::Button { button, down } => {
                let code = BUTTONS
                    .get(usize::from(button))
                    .ok_or(InjectError::UnknownButton(button))?;
                self.mouse.emit(&[*KeyEvent::new(*code, i32::from(down))])?;
            }
            Event::Move { dx, dy } => self.mouse.emit(&[
                *RelativeAxisEvent::new(RelativeAxisCode::REL_X, dx),
                *RelativeAxisEvent::new(RelativeAxisCode::REL_Y, dy),
            ])?,
            Event::Position { x, y, display } => {
                check_position(x, y, display, self.screen)?;
                self.tablet.emit(&[
                    *AbsoluteAxisEvent::new(AbsoluteAxisCode::ABS_X, x),
                    *AbsoluteAxisEvent::new(AbsoluteAxisCode::ABS_Y, y),
                ])?;
            }
            Event::Wheel { dx, dy } => {
                let vertical = whole_notches(&mut self.wheel.vertical, dy);
                let horizontal = whole_notches(&mut self.wheel.horizontal, dx);
                let events: [InputEvent; 4] = [
                    *RelativeAxisEvent::new(RelativeAxisCode::REL_WHEEL_HI_RES, dy),
                    *RelativeAxisEvent::new(RelativeAxisCode::REL_HWHEEL_HI_RES, dx),
                    *RelativeAxisEvent::new(RelativeAxisCode::REL_WHEEL, vertical),
                    *RelativeAxisEvent::new(RelativeAxisCode::REL_HWHEEL, horizontal),
                ];
                self.mouse.emit(&events)?;
            }
            Event::DeviceRemoved {
                kind: RemovableDevice::Keyboard,
                device,
            } => {
                self.keyboards.remove(&device);
            }
            _ => return Err(InjectError::NotInjected(event.kind())),
        }
        Ok(())
    }
}

impl Injector {
    fn keyboard(&mut self, device: u8) -> io::Result<&mut VirtualDevice> {
        match self.keyboards.entry(device) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let name = format!("{KEYBOARD_NAME_PREFIX} {device}");
                let keyboard = VirtualDevice::builder()?
                    .name(&name)
                    .with_keys(&self.keyboard_keys)?
                    .build()?;
                Ok(entry.insert(keyboard))
            }
        }
    }
}

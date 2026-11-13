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

use crate::hid::{evdev_from_usage, UnknownUsage};
use crate::packet::{Event, EventKind, RemovableDevice};

pub const KEYBOARD_NAME_PREFIX: &str = "g2g-desktop keyboard";

// linux reports one wheel notch as 120 high-resolution units
const HI_RES_UNITS_PER_NOTCH: i32 = 120;

const BUTTONS: [KeyCode; 3] = [KeyCode::BTN_LEFT, KeyCode::BTN_RIGHT, KeyCode::BTN_MIDDLE];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenSize {
    pub width: u16,
    pub height: u16,
}

#[derive(Debug)]
pub enum InjectError {
    UnknownUsage(UnknownUsage),
    UnknownButton(u8),
    UnknownDisplay(u8),
    OutsideScreen { x: i32, y: i32 },
    NotInjected(EventKind),
    Device(io::Error),
}

impl fmt::Display for InjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InjectError::UnknownUsage(usage) => usage.fmt(f),
            InjectError::UnknownButton(button) => write!(f, "button {button} has no evdev code"),
            InjectError::UnknownDisplay(display) => write!(f, "display {display} is not captured"),
            InjectError::OutsideScreen { x, y } => {
                write!(f, "position ({x}, {y}) is outside the captured display")
            }
            InjectError::NotInjected(kind) => write!(f, "{kind:?} events are not injected"),
            InjectError::Device(e) => write!(f, "uinput: {e}"),
        }
    }
}

impl std::error::Error for InjectError {}

impl From<io::Error> for InjectError {
    fn from(e: io::Error) -> Self {
        InjectError::Device(e)
    }
}

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

    pub fn inject(&mut self, event: &Event) -> Result<(), InjectError> {
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
                if display != 0 {
                    return Err(InjectError::UnknownDisplay(display));
                }
                let inside = |value: i32, extent: u16| (0..i32::from(extent)).contains(&value);
                if !inside(x, self.screen.width) || !inside(y, self.screen.height) {
                    return Err(InjectError::OutsideScreen { x, y });
                }
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

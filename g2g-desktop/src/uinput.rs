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
    PropType, RelativeAxisCode, RelativeAxisEvent, UinputAbsSetup,
};

use crate::hid::evdev_from_usage;
use crate::injection::{
    check_position, type_us_layout, InjectError, InjectEvents, ScreenSize, TouchSlots, TOUCH_SLOTS,
};
use crate::packet::{Event, RemovableDevice, TouchPhase};

pub const KEYBOARD_NAME_PREFIX: &str = "g2g-desktop keyboard";

// linux reports one wheel notch as 120 high-resolution units
const HI_RES_UNITS_PER_NOTCH: i32 = 120;

const BUTTONS: [KeyCode; 5] = [
    KeyCode::BTN_LEFT,
    KeyCode::BTN_RIGHT,
    KeyCode::BTN_MIDDLE,
    KeyCode::BTN_SIDE,
    KeyCode::BTN_EXTRA,
];

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

fn axis(code: AbsoluteAxisCode, maximum: i32) -> UinputAbsSetup {
    UinputAbsSetup::new(code, AbsInfo::new(0, 0, maximum, 0, 0, 0))
}

fn build_tablet(screen: ScreenSize) -> io::Result<VirtualDevice> {
    // udev classifies an absolute device as a mouse only when it also has mouse buttons
    VirtualDevice::builder()?
        .name(TABLET_NAME)
        .with_keys(&buttons())?
        .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_X, i32::from(screen.width) - 1))?
        .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_Y, i32::from(screen.height) - 1))?
        .build()
}

pub const TOUCHSCREEN_NAME: &str = "g2g-desktop touchscreen";
// linux names each contact with a tracking id from down to up, and -1 lifts the contact in a slot
const MAX_TRACKING_ID: i32 = 65535;
const LIFTED: i32 = -1;

fn build_touchscreen(screen: ScreenSize) -> io::Result<VirtualDevice> {
    let width = i32::from(screen.width) - 1;
    let height = i32::from(screen.height) - 1;
    VirtualDevice::builder()?
        .name(TOUCHSCREEN_NAME)
        .with_keys(&AttributeSet::from_iter([KeyCode::BTN_TOUCH]))?
        .with_properties(&AttributeSet::from_iter([PropType::DIRECT]))?
        .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_X, width))?
        .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_Y, height))?
        .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_MT_SLOT, TOUCH_SLOTS as i32 - 1))?
        .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_MT_TRACKING_ID, MAX_TRACKING_ID))?
        .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_MT_POSITION_X, width))?
        .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_MT_POSITION_Y, height))?
        .build()
}

fn touch_position(slot: usize, x: i32, y: i32) -> [InputEvent; 5] {
    [
        *AbsoluteAxisEvent::new(AbsoluteAxisCode::ABS_MT_SLOT, slot as i32),
        *AbsoluteAxisEvent::new(AbsoluteAxisCode::ABS_MT_POSITION_X, x),
        *AbsoluteAxisEvent::new(AbsoluteAxisCode::ABS_MT_POSITION_Y, y),
        *AbsoluteAxisEvent::new(AbsoluteAxisCode::ABS_X, x),
        *AbsoluteAxisEvent::new(AbsoluteAxisCode::ABS_Y, y),
    ]
}

fn touch_lift(slot: usize) -> [InputEvent; 2] {
    [
        *AbsoluteAxisEvent::new(AbsoluteAxisCode::ABS_MT_SLOT, slot as i32),
        *AbsoluteAxisEvent::new(AbsoluteAxisCode::ABS_MT_TRACKING_ID, LIFTED),
    ]
}

#[derive(Debug, Default)]
pub(crate) struct WheelRemainder {
    pub(crate) vertical: i32,
    pub(crate) horizontal: i32,
}

pub(crate) fn whole_notches(remainder: &mut i32, hi_res: i32) -> i32 {
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
    touchscreen: VirtualDevice,
    touch_slots: TouchSlots,
    next_tracking_id: i32,
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
            touchscreen: build_touchscreen(screen)?,
            touch_slots: TouchSlots::default(),
            next_tracking_id: 0,
            wheel: WheelRemainder::default(),
        })
    }

    fn touch(&mut self, contact: u32, phase: TouchPhase) -> Result<(), InjectError> {
        let touching = *KeyEvent::new(KeyCode::BTN_TOUCH, 1);
        let released = *KeyEvent::new(KeyCode::BTN_TOUCH, 0);
        let mut events = Vec::new();
        match phase {
            TouchPhase::Down { x, y, display } => {
                check_position(x, y, display, self.screen)?;
                let slot = self.touch_slots.press(contact)?;
                let tracking_id = self.next_tracking_id;
                self.next_tracking_id = (tracking_id + 1) % (MAX_TRACKING_ID + 1);
                events.extend(touch_position(slot, x, y));
                events.push(*AbsoluteAxisEvent::new(
                    AbsoluteAxisCode::ABS_MT_TRACKING_ID,
                    tracking_id,
                ));
                events.push(touching);
            }
            TouchPhase::Move { x, y, display } => {
                check_position(x, y, display, self.screen)?;
                events.extend(touch_position(self.touch_slots.find(contact)?, x, y));
            }
            TouchPhase::Up | TouchPhase::Cancel => {
                events.extend(touch_lift(self.touch_slots.lift(contact)?));
                if self.touch_slots.is_empty() {
                    events.push(released);
                }
            }
        }
        self.touchscreen.emit(&events)?;
        Ok(())
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
            Event::Text(ref text) => return type_us_layout(self, text),
            Event::Touch { contact, phase } => self.touch(contact, phase)?,
            Event::DeviceRemoved {
                kind: RemovableDevice::Touchscreen,
                ..
            } => {
                let mut events: Vec<InputEvent> = self
                    .touch_slots
                    .lift_all()
                    .into_iter()
                    .flat_map(touch_lift)
                    .collect();
                events.push(*KeyEvent::new(KeyCode::BTN_TOUCH, 0));
                self.touchscreen.emit(&events)?;
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

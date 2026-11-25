// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::fmt;
use std::io;
use std::task::{Context, Poll};

use keycode::{KeyMap, KeyMappingCode};

use crate::hid::{us_layout_key, UnknownUsage};
use crate::packet::{Event, EventKind, MotorLevels, RemovableDevice};

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
    OutsideScreen {
        x: i32,
        y: i32,
    },
    MonitorMismatch {
        captured: ScreenSize,
        monitor_width: u32,
        monitor_height: u32,
    },
    NotInjected(EventKind),
    NoUsLayoutKey(String),
    NoFreeTouchSlot,
    UnknownContact(u32),
    Refused(io::Error),
    Device(io::Error),
    GamepadUnavailable(io::Error),
    TriggerOutOfRange(i16),
}

impl fmt::Display for InjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InjectError::UnknownUsage(usage) => usage.fmt(f),
            InjectError::UnknownButton(button) => write!(f, "button {button} has no OS button"),
            InjectError::UnknownDisplay(display) => write!(f, "display {display} is not captured"),
            InjectError::OutsideScreen { x, y } => {
                write!(f, "position ({x}, {y}) is outside the captured display")
            }
            InjectError::MonitorMismatch {
                captured,
                monitor_width,
                monitor_height,
            } => write!(
                f,
                "capture is {}x{} but the monitor is {monitor_width}x{monitor_height}",
                captured.width, captured.height
            ),
            InjectError::NotInjected(kind) => write!(f, "{kind:?} events are not injected"),
            InjectError::NoFreeTouchSlot => {
                write!(f, "all {TOUCH_SLOTS} touch contacts are down")
            }
            InjectError::UnknownContact(contact) => {
                write!(f, "touch contact {contact} is not down")
            }
            InjectError::NoUsLayoutKey(text) => {
                write!(f, "no US keyboard key types {text:?}")
            }
            InjectError::Refused(e) => write!(f, "the OS refused the event: {e}"),
            InjectError::Device(e) => write!(f, "input device: {e}"),
            InjectError::GamepadUnavailable(e) => {
                write!(f, "no virtual gamepad, gamepads are dropped: {e}")
            }
            InjectError::TriggerOutOfRange(value) => {
                write!(f, "trigger value {value} is outside 0 to 255")
            }
        }
    }
}

impl std::error::Error for InjectError {}

impl From<io::Error> for InjectError {
    fn from(e: io::Error) -> Self {
        InjectError::Device(e)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GamepadRumble {
    pub device: u8,
    pub motors: MotorLevels,
}

pub trait InjectEvents {
    fn inject(&mut self, event: &Event) -> Result<(), InjectError>;

    fn poll_rumble(&mut self, _cx: &mut Context<'_>) -> Poll<Result<GamepadRumble, InjectError>> {
        Poll::Pending
    }
}

impl<I: InjectEvents + ?Sized> InjectEvents for Box<I> {
    fn inject(&mut self, event: &Event) -> Result<(), InjectError> {
        (**self).inject(event)
    }

    fn poll_rumble(&mut self, cx: &mut Context<'_>) -> Poll<Result<GamepadRumble, InjectError>> {
        (**self).poll_rumble(cx)
    }
}

#[derive(Debug)]
pub struct WithGamepads<D, G> {
    pub desktop: D,
    pub gamepads: G,
}

impl<D: InjectEvents, G: InjectEvents> InjectEvents for WithGamepads<D, G> {
    fn inject(&mut self, event: &Event) -> Result<(), InjectError> {
        match event {
            Event::Gamepad { .. }
            | Event::DeviceRemoved {
                kind: RemovableDevice::Gamepad,
                ..
            } => self.gamepads.inject(event),
            _ => self.desktop.inject(event),
        }
    }

    fn poll_rumble(&mut self, cx: &mut Context<'_>) -> Poll<Result<GamepadRumble, InjectError>> {
        self.gamepads.poll_rumble(cx)
    }
}

pub fn check_position(x: i32, y: i32, display: u8, screen: ScreenSize) -> Result<(), InjectError> {
    if display != 0 {
        return Err(InjectError::UnknownDisplay(display));
    }
    let inside = |value: i32, extent: u16| (0..i32::from(extent)).contains(&value);
    if !inside(x, screen.width) || !inside(y, screen.height) {
        return Err(InjectError::OutsideScreen { x, y });
    }
    Ok(())
}

pub const TOUCH_SLOTS: usize = 10;

#[derive(Debug, Default)]
pub struct TouchSlots([Option<u32>; TOUCH_SLOTS]);

impl TouchSlots {
    pub fn press(&mut self, contact: u32) -> Result<usize, InjectError> {
        let slot = match self.find(contact) {
            Ok(slot) => slot,
            Err(_) => self
                .0
                .iter()
                .position(Option::is_none)
                .ok_or(InjectError::NoFreeTouchSlot)?,
        };
        self.0[slot] = Some(contact);
        Ok(slot)
    }

    pub fn find(&self, contact: u32) -> Result<usize, InjectError> {
        self.0
            .iter()
            .position(|held| *held == Some(contact))
            .ok_or(InjectError::UnknownContact(contact))
    }

    pub fn lift(&mut self, contact: u32) -> Result<usize, InjectError> {
        let slot = self.find(contact)?;
        self.0[slot] = None;
        Ok(slot)
    }

    pub fn lift_all(&mut self) -> Vec<usize> {
        let held = (0..TOUCH_SLOTS)
            .filter(|&slot| self.0[slot].is_some())
            .collect();
        self.0 = [None; TOUCH_SLOTS];
        held
    }

    pub fn is_empty(&self) -> bool {
        self.0.iter().all(Option::is_none)
    }
}

pub fn type_us_layout(injector: &mut impl InjectEvents, text: &str) -> Result<(), InjectError> {
    let shift = KeyMap::from(KeyMappingCode::ShiftLeft).usb;
    let key = |usage, down| Event::Key {
        device: 0,
        usage,
        down,
    };
    let mut untyped = String::new();
    for character in text.chars() {
        let Some(typed) = us_layout_key(character) else {
            untyped.push(character);
            continue;
        };
        if typed.shifted {
            injector.inject(&key(shift, true))?;
        }
        injector.inject(&key(typed.usage, true))?;
        injector.inject(&key(typed.usage, false))?;
        if typed.shifted {
            injector.inject(&key(shift, false))?;
        }
    }
    if !untyped.is_empty() {
        return Err(InjectError::NoUsLayoutKey(untyped));
    }
    Ok(())
}

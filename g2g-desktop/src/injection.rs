// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::fmt;
use std::io;

use keycode::{KeyMap, KeyMappingCode};

use crate::hid::{us_layout_key, UnknownUsage};
use crate::packet::{Event, EventKind};

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
    Refused(io::Error),
    Device(io::Error),
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
            InjectError::NoUsLayoutKey(text) => {
                write!(f, "no US keyboard key types {text:?}")
            }
            InjectError::Refused(e) => write!(f, "the OS refused the event: {e}"),
            InjectError::Device(e) => write!(f, "input device: {e}"),
        }
    }
}

impl std::error::Error for InjectError {}

impl From<io::Error> for InjectError {
    fn from(e: io::Error) -> Self {
        InjectError::Device(e)
    }
}

pub trait InjectEvents {
    fn inject(&mut self, event: &Event) -> Result<(), InjectError>;
}

impl<I: InjectEvents + ?Sized> InjectEvents for Box<I> {
    fn inject(&mut self, event: &Event) -> Result<(), InjectError> {
        (**self).inject(event)
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

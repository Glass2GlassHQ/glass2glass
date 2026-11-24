// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::fmt;
use std::io;

use crate::hid::UnknownUsage;
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

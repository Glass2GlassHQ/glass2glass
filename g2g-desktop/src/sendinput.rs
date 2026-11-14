// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::mem::size_of;

use g2g_plugins::d3d11screencapturesrc::{with_physical_pixels, MonitorBounds};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT, KEYEVENTF_EXTENDEDKEY,
    KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_HWHEEL,
    MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP,
    MOUSEEVENTF_MOVE, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_VIRTUALDESK,
    MOUSEEVENTF_WHEEL, MOUSEINPUT, MOUSE_EVENT_FLAGS, VIRTUAL_KEY,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetSystemMetrics, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
};

use crate::hid::windows_scancode_from_usage;
use crate::injection::{check_position, InjectError, ScreenSize};
use crate::packet::{Event, RemovableDevice};

// keycode writes an extended key's E0 prefix byte into the high byte of its windows code
const EXTENDED_PREFIX: u16 = 0xe000;
const PREFIX_MASK: u16 = 0xff00;
const SCANCODE_MASK: u16 = 0x00ff;
// windows spreads absolute pointer coordinates over 65536 steps across the virtual desktop
const ABSOLUTE_STEPS: i64 = 65536;

const BUTTONS: [(MOUSE_EVENT_FLAGS, MOUSE_EVENT_FLAGS); 3] = [
    (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
];

#[derive(Debug, Clone, Copy)]
struct VirtualDesktop {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

fn virtual_desktop() -> VirtualDesktop {
    // SAFETY: GetSystemMetrics reads a value and has no other effect.
    with_physical_pixels(|| unsafe {
        VirtualDesktop {
            x: GetSystemMetrics(SM_XVIRTUALSCREEN),
            y: GetSystemMetrics(SM_YVIRTUALSCREEN),
            width: GetSystemMetrics(SM_CXVIRTUALSCREEN),
            height: GetSystemMetrics(SM_CYVIRTUALSCREEN),
        }
    })
}

fn absolute_step(pixel: i32, origin: i32, extent: i32) -> i32 {
    let centre_twice = (i64::from(pixel) - i64::from(origin)) * 2 + 1;
    (centre_twice * ABSOLUTE_STEPS / (2 * i64::from(extent))) as i32
}

fn mouse(dx: i32, dy: i32, data: i32, flags: MOUSE_EVENT_FLAGS) -> INPUT {
    INPUT {
        r#type: INPUT_MOUSE,
        Anonymous: INPUT_0 {
            mi: MOUSEINPUT {
                dx,
                dy,
                mouseData: data as u32,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

fn key(scancode: u16, down: bool) -> INPUT {
    let mut flags = KEYEVENTF_SCANCODE;
    if scancode & PREFIX_MASK == EXTENDED_PREFIX {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    if !down {
        flags |= KEYEVENTF_KEYUP;
    }
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(0),
                wScan: scancode & SCANCODE_MASK,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

fn send(inputs: &[INPUT]) -> Result<(), InjectError> {
    // SAFETY: the slice holds initialised INPUT structs and cbsize is their size.
    let sent = unsafe { SendInput(inputs, size_of::<INPUT>() as i32) };
    if sent as usize != inputs.len() {
        return Err(InjectError::Refused(io::Error::last_os_error()));
    }
    Ok(())
}

#[derive(Debug)]
pub struct Injector {
    screen: ScreenSize,
    monitor: MonitorBounds,
    held_scancodes: BTreeMap<u8, BTreeSet<u16>>,
}

impl Injector {
    pub fn new(screen: ScreenSize, monitor: MonitorBounds) -> Result<Self, InjectError> {
        if (u32::from(screen.width), u32::from(screen.height)) != (monitor.width, monitor.height) {
            return Err(InjectError::MonitorMismatch {
                captured: screen,
                monitor_width: monitor.width,
                monitor_height: monitor.height,
            });
        }
        Ok(Self {
            screen,
            monitor,
            held_scancodes: BTreeMap::new(),
        })
    }

    pub fn inject(&mut self, event: &Event) -> Result<(), InjectError> {
        match *event {
            Event::Key {
                device,
                usage,
                down,
            } => {
                let scancode =
                    windows_scancode_from_usage(usage).map_err(InjectError::UnknownUsage)?;
                send(&[key(scancode, down)])?;
                let held = self.held_scancodes.entry(device).or_default();
                if down {
                    held.insert(scancode);
                } else {
                    held.remove(&scancode);
                }
            }
            Event::Button { button, down } => {
                let (press, release) = BUTTONS
                    .get(usize::from(button))
                    .ok_or(InjectError::UnknownButton(button))?;
                send(&[mouse(0, 0, 0, if down { *press } else { *release })])?;
            }
            Event::Move { dx, dy } => send(&[mouse(dx, dy, 0, MOUSEEVENTF_MOVE)])?,
            Event::Position { x, y, display } => {
                check_position(x, y, display, self.screen)?;
                let desktop = virtual_desktop();
                let flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK;
                send(&[mouse(
                    absolute_step(self.monitor.x + x, desktop.x, desktop.width),
                    absolute_step(self.monitor.y + y, desktop.y, desktop.height),
                    0,
                    flags,
                )])?;
            }
            Event::Wheel { dx, dy } => {
                let mut inputs = Vec::new();
                if dy != 0 {
                    inputs.push(mouse(0, 0, dy, MOUSEEVENTF_WHEEL));
                }
                if dx != 0 {
                    inputs.push(mouse(0, 0, dx, MOUSEEVENTF_HWHEEL));
                }
                send(&inputs)?;
            }
            Event::DeviceRemoved {
                kind: RemovableDevice::Keyboard,
                device,
            } => {
                let held = self.held_scancodes.remove(&device).unwrap_or_default();
                let releases: Vec<INPUT> = held
                    .into_iter()
                    .map(|scancode| key(scancode, false))
                    .collect();
                send(&releases)?;
            }
            _ => return Err(InjectError::NotInjected(event.kind())),
        }
        Ok(())
    }
}

// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::io;

use g2g_plugins::windowinput::WindowInput;
use g2g_wayland_display::waylanddisplaysrc::{InputSender, WaylandDisplay};

use crate::hid::evdev_from_usage;
use crate::injection::{check_position, InjectError, InjectEvents, ScreenSize};
use crate::packet::{Event, RemovableDevice, POINTER_BUTTONS};

#[derive(Debug)]
pub struct WaylandDisplayInjector {
    screen: ScreenSize,
    input: InputSender,
    display: WaylandDisplay,
    pointer: (u32, u32),
}

impl WaylandDisplayInjector {
    pub fn new(screen: ScreenSize, display: &WaylandDisplay) -> Self {
        Self {
            screen,
            input: display.input_sender(),
            display: display.clone(),
            pointer: (0, 0),
        }
    }
}

fn moved_pointer(pointer: (u32, u32), dx: i32, dy: i32, screen: ScreenSize) -> WindowInput {
    // pointers stop at the screen edge
    let step = |from: u32, delta: i32, extent: u16| {
        (i64::from(from) + i64::from(delta)).clamp(0, i64::from(extent) - 1) as u32
    };
    WindowInput::Pointer {
        x: step(pointer.0, dx, screen.width),
        y: step(pointer.1, dy, screen.height),
    }
}

fn window_input(event: &Event, screen: ScreenSize) -> Result<WindowInput, InjectError> {
    let input = match *event {
        Event::Key { usage, down, .. } => WindowInput::Key {
            scancode: u32::from(evdev_from_usage(usage).map_err(InjectError::UnknownUsage)?),
            down,
        },
        Event::Button { button, down } => WindowInput::Button {
            button: *POINTER_BUTTONS
                .get(usize::from(button))
                .ok_or(InjectError::UnknownButton(button))?,
            down,
        },
        Event::Position { x, y, display } => {
            check_position(x, y, display, screen)?;
            let outside = || InjectError::OutsideScreen { x, y };
            WindowInput::Pointer {
                x: u32::try_from(x).map_err(|_| outside())?,
                y: u32::try_from(y).map_err(|_| outside())?,
            }
        }
        Event::Wheel { dx, dy } => WindowInput::Wheel { dx, dy },
        // the compositor releases held keys and buttons on focus loss
        Event::DeviceRemoved {
            kind: RemovableDevice::Keyboard,
            ..
        } => WindowInput::Focus(false),
        _ => return Err(InjectError::NotInjected(event.kind())),
    };
    Ok(input)
}

impl InjectEvents for WaylandDisplayInjector {
    fn inject(&mut self, event: &Event) -> Result<(), InjectError> {
        let display_gone = || InjectError::Device(io::Error::from(io::ErrorKind::BrokenPipe));
        if let Event::Clipboard(text) = event {
            return self
                .display
                .set_clipboard(text.clone())
                .map_err(|_| display_gone());
        }
        let input = match *event {
            Event::Move { dx, dy } => moved_pointer(self.pointer, dx, dy, self.screen),
            _ => window_input(event, self.screen)?,
        };
        if let WindowInput::Pointer { x, y } = input {
            self.pointer = (x, y);
        }
        self.input.send(input).map_err(|_| display_gone())
    }
}

#[cfg(test)]
mod tests {
    use g2g_plugins::windowinput::PointerButton;

    use super::*;

    const SCREEN: ScreenSize = ScreenSize {
        width: 160,
        height: 128,
    };
    // USB HID keyboard usage for A, and linux input-event-codes.h KEY_A
    const USAGE_A: u16 = 0x04;
    const EVDEV_A: u32 = 30;

    #[test]
    fn protocol_events_become_the_compositor_input() {
        assert_eq!(
            window_input(
                &Event::Key {
                    device: 0,
                    usage: USAGE_A,
                    down: true
                },
                SCREEN
            )
            .expect("maps"),
            WindowInput::Key {
                scancode: EVDEV_A,
                down: true
            }
        );
        assert_eq!(
            window_input(
                &Event::Button {
                    button: 1,
                    down: true
                },
                SCREEN
            )
            .expect("maps"),
            WindowInput::Button {
                button: PointerButton::Right,
                down: true
            }
        );
        assert_eq!(
            window_input(
                &Event::Position {
                    x: 10,
                    y: 20,
                    display: 0
                },
                SCREEN
            )
            .expect("maps"),
            WindowInput::Pointer { x: 10, y: 20 }
        );
        assert!(matches!(
            window_input(
                &Event::Position {
                    x: i32::from(SCREEN.width),
                    y: 0,
                    display: 0
                },
                SCREEN
            ),
            Err(InjectError::OutsideScreen { .. })
        ));
    }

    #[test]
    fn relative_motion_moves_the_pointer_within_the_screen() {
        assert_eq!(
            moved_pointer((10, 20), 5, -3, SCREEN),
            WindowInput::Pointer { x: 15, y: 17 }
        );
        assert_eq!(
            moved_pointer((10, 20), -50, 1000, SCREEN),
            WindowInput::Pointer {
                x: 0,
                y: u32::from(SCREEN.height) - 1
            }
        );
    }
}

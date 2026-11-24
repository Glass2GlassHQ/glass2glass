// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "session"))]

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};

use g2g_desktop::injection::{InjectEvents, ScreenSize};
use g2g_desktop::packet::{Event, RemovableDevice};
use g2g_desktop::xtest::XTestInjector;
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    ConnectionExt as _, CreateWindowAux, EventMask, InputFocus, Window, WindowClass,
};
use x11rb::protocol::Event as XEvent;
use x11rb::rust_connection::RustConnection;
use x11rb::COPY_DEPTH_FROM_PARENT;

const SCREEN: ScreenSize = ScreenSize {
    width: 320,
    height: 240,
};
// USB HID keyboard usage for A, and its X keycode: linux KEY_A (30) plus 8
const HID_A: u16 = 0x04;
const X_KEYCODE_A: u8 = 38;
const LEFT_BUTTON_INDEX: u8 = 0;
const X_LEFT_BUTTON: u8 = 1;
const X_WHEEL_UP: u8 = 4;
const ONE_NOTCH: i32 = 120;
const POINTER: (i32, i32) = (10, 20);

struct Xvfb {
    server: Child,
    display: String,
}

impl Xvfb {
    fn start() -> Self {
        let screen = format!("{}x{}x24", SCREEN.width, SCREEN.height);
        // Xvfb writes the display number it picked to the -displayfd descriptor
        let mut server = Command::new("sh")
            .args([
                "-c",
                &format!(
                    "exec Xvfb -displayfd 3 -screen 0 {screen} -nolisten tcp 3>&1 >/dev/null 2>&1"
                ),
            ])
            .stdout(Stdio::piped())
            .spawn()
            .expect("starts Xvfb");
        let mut number = String::new();
        BufReader::new(server.stdout.take().expect("piped"))
            .read_line(&mut number)
            .expect("Xvfb reports its display");
        Self {
            server,
            display: format!(":{}", number.trim()),
        }
    }
}

impl Drop for Xvfb {
    fn drop(&mut self) {
        let _ = self.server.kill();
        let _ = self.server.wait();
    }
}

fn focused_screen_window(display: &str) -> (RustConnection, Window) {
    let (connection, screen) = x11rb::connect(Some(display)).expect("connects to Xvfb");
    let root = connection.setup().roots[screen].root;
    let window = connection.generate_id().expect("window id");
    let events = EventMask::KEY_PRESS
        | EventMask::KEY_RELEASE
        | EventMask::BUTTON_PRESS
        | EventMask::BUTTON_RELEASE
        | EventMask::POINTER_MOTION;
    connection
        .create_window(
            COPY_DEPTH_FROM_PARENT,
            window,
            root,
            0,
            0,
            SCREEN.width,
            SCREEN.height,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &CreateWindowAux::new().event_mask(events),
        )
        .expect("creates the window");
    connection.map_window(window).expect("maps the window");
    connection
        .set_input_focus(InputFocus::POINTER_ROOT, window, x11rb::CURRENT_TIME)
        .expect("focuses the window");
    connection
        .get_input_focus()
        .expect("sends")
        .reply()
        .expect("the server answers");
    (connection, window)
}

#[derive(Debug, PartialEq)]
enum Seen {
    Motion(i16, i16),
    KeyPress(u8),
    KeyRelease(u8),
    ButtonPress(u8),
}

fn next_seen(connection: &RustConnection) -> Seen {
    loop {
        let seen = match connection.wait_for_event().expect("Xvfb stays up") {
            XEvent::MotionNotify(motion) => Seen::Motion(motion.event_x, motion.event_y),
            XEvent::KeyPress(key) => Seen::KeyPress(key.detail),
            XEvent::KeyRelease(key) => Seen::KeyRelease(key.detail),
            XEvent::ButtonPress(button) => Seen::ButtonPress(button.detail),
            _ => continue,
        };
        return seen;
    }
}

#[test]
fn protocol_events_reach_the_focused_x11_window_through_xtest() {
    let server = Xvfb::start();
    let (connection, _window) = focused_screen_window(&server.display);
    let mut injector = XTestInjector::new(&server.display, SCREEN).expect("opens XTest");
    let (x, y) = POINTER;
    let steps = [
        (
            Event::Position { x, y, display: 0 },
            Seen::Motion(x as i16, y as i16),
        ),
        (
            Event::Key {
                device: 0,
                usage: HID_A,
                down: true,
            },
            Seen::KeyPress(X_KEYCODE_A),
        ),
        (
            Event::DeviceRemoved {
                kind: RemovableDevice::Keyboard,
                device: 0,
            },
            Seen::KeyRelease(X_KEYCODE_A),
        ),
        (
            Event::Button {
                button: LEFT_BUTTON_INDEX,
                down: true,
            },
            Seen::ButtonPress(X_LEFT_BUTTON),
        ),
        (
            Event::Wheel {
                dx: 0,
                dy: ONE_NOTCH,
            },
            Seen::ButtonPress(X_WHEEL_UP),
        ),
    ];
    for (event, expected) in steps {
        injector.inject(&event).expect("injects");
        assert_eq!(next_seen(&connection), expected, "after {event:?}");
    }
}

// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "windows", feature = "session"))]

use g2g_desktop::injection::{InjectError, InjectEvents, ScreenSize};
use g2g_desktop::packet::{Event, TouchPhase};
use g2g_desktop::sendinput::Injector;
use g2g_plugins::d3d11screencapturesrc::monitor_bounds;

const PRIMARY_MONITOR: i32 = -1;
const CONTACT: u32 = 77;

// touch injection needs an interactive desktop, which an ssh session or a CI service lacks
#[test]
fn windows_accepts_a_touch_down_move_and_up() {
    let Ok(monitor) = monitor_bounds(PRIMARY_MONITOR) else {
        eprintln!("skipping touch injection: no desktop monitor");
        return;
    };
    let screen = ScreenSize {
        width: u16::try_from(monitor.width).expect("monitor width fits a packet"),
        height: u16::try_from(monitor.height).expect("monitor height fits a packet"),
    };
    let mut injector = Injector::new(screen, monitor).expect("injector for the primary monitor");
    let start = (i32::from(screen.width) / 3, i32::from(screen.height) / 3);
    let end = (start.0 + 40, start.1 + 30);
    let down = Event::Touch {
        contact: CONTACT,
        phase: TouchPhase::Down {
            x: start.0,
            y: start.1,
            display: 0,
        },
    };
    match injector.inject(&down) {
        Ok(()) => {}
        Err(InjectError::Refused(e)) => {
            eprintln!("skipping touch injection: windows refused it, {e}");
            return;
        }
        Err(e) => panic!("touch down: {e}"),
    }
    let moved = Event::Touch {
        contact: CONTACT,
        phase: TouchPhase::Move {
            x: end.0,
            y: end.1,
            display: 0,
        },
    };
    // windows fails a frame whose flags or up position do not follow the contact's earlier frames
    injector.inject(&moved).expect("the contact moves");
    let up = Event::Touch {
        contact: CONTACT,
        phase: TouchPhase::Up,
    };
    injector.inject(&up).expect("the contact lifts");
}

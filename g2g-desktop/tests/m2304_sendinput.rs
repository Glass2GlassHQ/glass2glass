// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "windows", feature = "session"))]

use g2g_desktop::injection::InjectEvents;
use g2g_desktop::injection::{InjectError, ScreenSize};
use g2g_desktop::packet::Event;
use g2g_desktop::sendinput::Injector;
use g2g_plugins::d3d11screencapturesrc::{monitor_bounds, with_physical_pixels};
use windows::Win32::Foundation::POINT;
use windows::Win32::UI::WindowsAndMessaging::GetCursorPos;

const PRIMARY_MONITOR: i32 = -1;

fn cursor() -> Option<POINT> {
    let mut point = POINT::default();
    // SAFETY: the out-param is a valid local.
    with_physical_pixels(|| unsafe { GetCursorPos(&mut point) })
        .ok()
        .map(|()| point)
}

// moving the pointer needs an interactive desktop, which an ssh session or a CI service lacks
#[test]
fn an_absolute_position_lands_on_that_pixel_of_the_monitor() {
    let Ok(monitor) = monitor_bounds(PRIMARY_MONITOR) else {
        eprintln!("skipping pointer injection: no desktop monitor");
        return;
    };
    let screen = ScreenSize {
        width: u16::try_from(monitor.width).expect("monitor width fits a packet"),
        height: u16::try_from(monitor.height).expect("monitor height fits a packet"),
    };
    let mut injector = Injector::new(screen, monitor).expect("injector for the primary monitor");
    let corners = [
        (0, 0),
        (i32::from(screen.width) - 1, i32::from(screen.height) - 1),
        (i32::from(screen.width) / 3, i32::from(screen.height) / 7),
    ];
    for (x, y) in corners {
        match injector.inject(&Event::Position { x, y, display: 0 }) {
            Ok(()) => {}
            Err(InjectError::Refused(e)) => {
                eprintln!("skipping pointer injection: SendInput refused, {e}");
                return;
            }
            Err(e) => panic!("position ({x}, {y}): {e}"),
        }
        let Some(point) = cursor() else {
            eprintln!("skipping pointer injection: no cursor on this desktop");
            return;
        };
        assert_eq!(
            (point.x, point.y),
            (monitor.x + x, monitor.y + y),
            "position ({x}, {y})"
        );
    }
}

#[test]
fn a_capture_that_differs_from_the_monitor_is_refused() {
    let Ok(monitor) = monitor_bounds(PRIMARY_MONITOR) else {
        eprintln!("skipping: no desktop monitor");
        return;
    };
    let wrong = ScreenSize {
        width: u16::try_from(monitor.width + 1).expect("monitor width fits a packet"),
        height: u16::try_from(monitor.height).expect("monitor height fits a packet"),
    };
    assert!(matches!(
        Injector::new(wrong, monitor),
        Err(InjectError::MonitorMismatch { .. })
    ));
}

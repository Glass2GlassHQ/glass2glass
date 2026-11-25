// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::mem::size_of;
use std::sync::{Arc, Mutex, Weak};
use std::thread;
use std::time::{Duration, Instant};

use g2g_core::log::Target;
use g2g_plugins::d3d11screencapturesrc::{with_physical_pixels, MonitorBounds};
use windows::Win32::Foundation::POINT;
use windows::Win32::UI::Controls::{
    CreateSyntheticPointerDevice, DestroySyntheticPointerDevice, HSYNTHETICPOINTERDEVICE,
    POINTER_FEEDBACK_DEFAULT, POINTER_TYPE_INFO, POINTER_TYPE_INFO_0,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT, KEYBD_EVENT_FLAGS,
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE, KEYEVENTF_UNICODE,
    MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_HWHEEL, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, MOUSEEVENTF_MOVE, MOUSEEVENTF_RIGHTDOWN,
    MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_VIRTUALDESK, MOUSEEVENTF_WHEEL, MOUSEEVENTF_XDOWN,
    MOUSEEVENTF_XUP, MOUSEINPUT, MOUSE_EVENT_FLAGS, VIRTUAL_KEY,
};
use windows::Win32::UI::Input::Pointer::{
    InjectSyntheticPointerInput, POINTER_FLAGS, POINTER_FLAG_CANCELED, POINTER_FLAG_DOWN,
    POINTER_FLAG_INCONTACT, POINTER_FLAG_INRANGE, POINTER_FLAG_NONE, POINTER_FLAG_UP,
    POINTER_FLAG_UPDATE, POINTER_TOUCH_INFO,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetSystemMetrics, PT_TOUCH, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN,
    SM_YVIRTUALSCREEN, XBUTTON1, XBUTTON2,
};

use crate::hid::windows_scancode_from_usage;
use crate::injection::{
    check_position, InjectError, InjectEvents, ScreenSize, TouchSlots, TOUCH_SLOTS,
};
use crate::packet::{Event, RemovableDevice, TouchPhase};

// keycode writes an extended key's E0 prefix byte into the high byte of its windows code
const EXTENDED_PREFIX: u16 = 0xe000;
const PREFIX_MASK: u16 = 0xff00;
const SCANCODE_MASK: u16 = 0x00ff;
// windows spreads absolute pointer coordinates over 65536 steps across the virtual desktop
const ABSOLUTE_STEPS: i64 = 65536;

struct ButtonFlags {
    press: MOUSE_EVENT_FLAGS,
    release: MOUSE_EVENT_FLAGS,
    x_button: u16,
}

const BUTTONS: [ButtonFlags; 5] = [
    ButtonFlags {
        press: MOUSEEVENTF_LEFTDOWN,
        release: MOUSEEVENTF_LEFTUP,
        x_button: 0,
    },
    ButtonFlags {
        press: MOUSEEVENTF_RIGHTDOWN,
        release: MOUSEEVENTF_RIGHTUP,
        x_button: 0,
    },
    ButtonFlags {
        press: MOUSEEVENTF_MIDDLEDOWN,
        release: MOUSEEVENTF_MIDDLEUP,
        x_button: 0,
    },
    ButtonFlags {
        press: MOUSEEVENTF_XDOWN,
        release: MOUSEEVENTF_XUP,
        x_button: XBUTTON1,
    },
    ButtonFlags {
        press: MOUSEEVENTF_XDOWN,
        release: MOUSEEVENTF_XUP,
        x_button: XBUTTON2,
    },
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

fn keyboard(scan: u16, mut flags: KEYBD_EVENT_FLAGS, down: bool) -> INPUT {
    if !down {
        flags |= KEYEVENTF_KEYUP;
    }
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(0),
                wScan: scan,
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
    keyboard(scancode & SCANCODE_MASK, flags, down)
}

fn send(inputs: &[INPUT]) -> Result<(), InjectError> {
    // SAFETY: the slice holds initialised INPUT structs and cbsize is their size.
    let sent = unsafe { SendInput(inputs, size_of::<INPUT>() as i32) };
    if sent as usize != inputs.len() {
        return Err(InjectError::Refused(io::Error::last_os_error()));
    }
    Ok(())
}

const TOUCH_LOG_CATEGORY: &str = "g2g-desktop-sendinput";
// windows cancels injected touch contacts that are not injected again within about a second
const TOUCH_REPEAT: Duration = Duration::from_millis(50);
const IN_CONTACT: POINTER_FLAGS = POINTER_FLAGS(POINTER_FLAG_INRANGE.0 | POINTER_FLAG_INCONTACT.0);
const ONE_FRAME_FLAGS: POINTER_FLAGS = POINTER_FLAGS(
    POINTER_FLAG_DOWN.0 | POINTER_FLAG_UPDATE.0 | POINTER_FLAG_UP.0 | POINTER_FLAG_CANCELED.0,
);

#[derive(Debug)]
struct TouchDevice(HSYNTHETICPOINTERDEVICE);

impl Drop for TouchDevice {
    fn drop(&mut self) {
        // SAFETY: the handle came from CreateSyntheticPointerDevice and is destroyed once.
        unsafe { DestroySyntheticPointerDevice(self.0) };
    }
}

#[derive(Debug)]
struct TouchState {
    device: TouchDevice,
    slots: TouchSlots,
    contacts: [POINTER_TOUCH_INFO; TOUCH_SLOTS],
    last_injection: Instant,
}

// SAFETY: the device handle has no thread affinity, the touch infos leave their window and device handles null, and the state is used only under a mutex.
unsafe impl Send for TouchState {}

impl TouchState {
    fn open() -> Result<Self, InjectError> {
        // SAFETY: creating a device takes no pointers.
        let device = unsafe {
            CreateSyntheticPointerDevice(PT_TOUCH, TOUCH_SLOTS as u32, POINTER_FEEDBACK_DEFAULT)
        }
        .map_err(|e| InjectError::Device(io::Error::other(e)))?;
        Ok(Self {
            device: TouchDevice(device),
            slots: TouchSlots::default(),
            contacts: [POINTER_TOUCH_INFO::default(); TOUCH_SLOTS],
            last_injection: Instant::now(),
        })
    }

    fn set(&mut self, slot: usize, flags: POINTER_FLAGS, location: Option<POINT>) {
        let pointer = &mut self.contacts[slot].pointerInfo;
        pointer.pointerType = PT_TOUCH;
        pointer.pointerId = slot as u32;
        pointer.pointerFlags = flags;
        if let Some(location) = location {
            pointer.ptPixelLocation = location;
        }
    }

    fn inject(&mut self) -> Result<(), InjectError> {
        // windows takes every contact that is down in each injection, packed at the front
        let frame: Vec<POINTER_TYPE_INFO> = self
            .contacts
            .iter()
            .filter(|contact| contact.pointerInfo.pointerFlags != POINTER_FLAG_NONE)
            .map(|contact| POINTER_TYPE_INFO {
                r#type: PT_TOUCH,
                Anonymous: POINTER_TYPE_INFO_0 {
                    touchInfo: *contact,
                },
            })
            .collect();
        if frame.is_empty() {
            return Ok(());
        }
        // SAFETY: the frame holds initialised touch infos for this device.
        let injected =
            with_physical_pixels(|| unsafe { InjectSyntheticPointerInput(self.device.0, &frame) });
        self.last_injection = Instant::now();
        for contact in &mut self.contacts {
            contact.pointerInfo.pointerFlags &= !ONE_FRAME_FLAGS;
        }
        injected.map_err(|e| InjectError::Refused(io::Error::other(e)))
    }
}

fn repeat_touches(state: Weak<Mutex<TouchState>>) {
    let log = Target::category(TOUCH_LOG_CATEGORY);
    while let Some(state) = state.upgrade() {
        let mut touches = state.lock().unwrap();
        if touches.last_injection.elapsed() >= TOUCH_REPEAT {
            if let Err(e) = touches.inject() {
                g2g_core::g2g_warn!(log, "did not repeat the held touches: {e}");
            }
        }
        drop(touches);
        drop(state);
        thread::sleep(TOUCH_REPEAT);
    }
}

#[derive(Debug)]
pub struct Injector {
    screen: ScreenSize,
    monitor: MonitorBounds,
    held_scancodes: BTreeMap<u8, BTreeSet<u16>>,
    touches: Option<Arc<Mutex<TouchState>>>,
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
            touches: None,
        })
    }

    fn touch_state(&mut self) -> Result<Arc<Mutex<TouchState>>, InjectError> {
        if let Some(touches) = &self.touches {
            return Ok(Arc::clone(touches));
        }
        let touches = Arc::new(Mutex::new(TouchState::open()?));
        let repeated = Arc::downgrade(&touches);
        thread::Builder::new()
            .name(TOUCH_LOG_CATEGORY.into())
            .spawn(move || repeat_touches(repeated))?;
        self.touches = Some(Arc::clone(&touches));
        Ok(touches)
    }

    fn touch(&mut self, contact: u32, phase: TouchPhase) -> Result<(), InjectError> {
        let (screen, monitor) = (self.screen, self.monitor);
        let desktop = virtual_desktop();
        // windows places synthetic pointers relative to the top left of the virtual desktop
        let location = |x: i32, y: i32, display: u8| -> Result<POINT, InjectError> {
            check_position(x, y, display, screen)?;
            Ok(POINT {
                x: monitor.x + x - desktop.x,
                y: monitor.y + y - desktop.y,
            })
        };
        let touches = self.touch_state()?;
        let mut touches = touches.lock().unwrap();
        match phase {
            TouchPhase::Down { x, y, display } => {
                let location = location(x, y, display)?;
                let slot = touches.slots.press(contact)?;
                let flags = POINTER_FLAGS(IN_CONTACT.0 | POINTER_FLAG_DOWN.0);
                touches.set(slot, flags, Some(location));
            }
            TouchPhase::Move { x, y, display } => {
                let location = location(x, y, display)?;
                let slot = touches.slots.find(contact)?;
                let flags = POINTER_FLAGS(IN_CONTACT.0 | POINTER_FLAG_UPDATE.0);
                touches.set(slot, flags, Some(location));
            }
            // windows needs an up at the contact's last position
            TouchPhase::Up => {
                let slot = touches.slots.lift(contact)?;
                touches.set(slot, POINTER_FLAG_UP, None);
            }
            TouchPhase::Cancel => {
                let slot = touches.slots.lift(contact)?;
                let flags = POINTER_FLAGS(POINTER_FLAG_UP.0 | POINTER_FLAG_CANCELED.0);
                touches.set(slot, flags, None);
            }
        }
        touches.inject()
    }

    fn cancel_touches(&mut self) -> Result<(), InjectError> {
        let Some(touches) = &self.touches else {
            return Ok(());
        };
        let mut touches = touches.lock().unwrap();
        let flags = POINTER_FLAGS(POINTER_FLAG_UP.0 | POINTER_FLAG_CANCELED.0);
        for slot in touches.slots.lift_all() {
            touches.set(slot, flags, None);
        }
        touches.inject()
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
                let flags = BUTTONS
                    .get(usize::from(button))
                    .ok_or(InjectError::UnknownButton(button))?;
                let event = if down { flags.press } else { flags.release };
                send(&[mouse(0, 0, i32::from(flags.x_button), event)])?;
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
            Event::Touch { contact, phase } => self.touch(contact, phase)?,
            Event::DeviceRemoved {
                kind: RemovableDevice::Touchscreen,
                ..
            } => self.cancel_touches()?,
            // windows KEYEVENTF_UNICODE events carry one UTF-16 code unit each
            Event::Text(ref text) => {
                let inputs: Vec<INPUT> = text
                    .encode_utf16()
                    .flat_map(|unit| {
                        [true, false].map(|down| keyboard(unit, KEYEVENTF_UNICODE, down))
                    })
                    .collect();
                send(&inputs)?;
            }
            _ => return Err(InjectError::NotInjected(event.kind())),
        }
        Ok(())
    }
}

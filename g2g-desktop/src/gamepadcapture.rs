// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::BTreeMap;
use std::time::Duration;

use g2g_core::log::Target;
use gilrs::ff::{BaseEffect, BaseEffectType, Effect, EffectBuilder};
use gilrs::{Axis, Button, EventType, Gamepad, GamepadId, Gilrs};
use tokio::sync::mpsc;

use crate::injection::GamepadRumble;
use crate::packet::{
    Event, GamepadAxes, MotorLevels, RemovableDevice, GAMEPAD_A, GAMEPAD_B, GAMEPAD_BACK,
    GAMEPAD_DPAD_DOWN, GAMEPAD_DPAD_LEFT, GAMEPAD_DPAD_RIGHT, GAMEPAD_DPAD_UP, GAMEPAD_GUIDE,
    GAMEPAD_LEFT_SHOULDER, GAMEPAD_LEFT_THUMB, GAMEPAD_RIGHT_SHOULDER, GAMEPAD_RIGHT_THUMB,
    GAMEPAD_START, GAMEPAD_X, GAMEPAD_Y,
};

const GAMEPAD_CATEGORY: &str = "g2g-desktop-gamepad";
const GAMEPAD_POLL_INTERVAL: Duration = Duration::from_millis(10);
const TRIGGER_MAX: f32 = 255.0;

// gilrs names face buttons by position
const BUTTON_BITS: [(u32, Button); 15] = [
    (GAMEPAD_A, Button::South),
    (GAMEPAD_B, Button::East),
    (GAMEPAD_X, Button::West),
    (GAMEPAD_Y, Button::North),
    (GAMEPAD_LEFT_SHOULDER, Button::LeftTrigger),
    (GAMEPAD_RIGHT_SHOULDER, Button::RightTrigger),
    (GAMEPAD_BACK, Button::Select),
    (GAMEPAD_START, Button::Start),
    (GAMEPAD_GUIDE, Button::Mode),
    (GAMEPAD_LEFT_THUMB, Button::LeftThumb),
    (GAMEPAD_RIGHT_THUMB, Button::RightThumb),
    (GAMEPAD_DPAD_UP, Button::DPadUp),
    (GAMEPAD_DPAD_DOWN, Button::DPadDown),
    (GAMEPAD_DPAD_LEFT, Button::DPadLeft),
    (GAMEPAD_DPAD_RIGHT, Button::DPadRight),
];

#[derive(Debug)]
pub struct GamepadCapture {
    pub states: mpsc::UnboundedReceiver<Event>,
    pub rumbles: mpsc::UnboundedSender<GamepadRumble>,
}

pub fn capture_gamepads() -> GamepadCapture {
    let (state_sender, states) = mpsc::unbounded_channel();
    let (rumbles, rumble_receiver) = mpsc::unbounded_channel();
    std::thread::Builder::new()
        .name(String::from("g2g-desktop-gamepads"))
        .spawn(move || run_gamepads(&state_sender, rumble_receiver))
        .expect("the gamepad thread starts");
    GamepadCapture { states, rumbles }
}

fn gamepad_state(device: u8, pad: &Gamepad<'_>) -> Event {
    let buttons = BUTTON_BITS
        .iter()
        .filter(|(_, button)| pad.is_pressed(*button))
        .fold(0, |buttons, (bit, _)| buttons | bit);
    let stick = |axis| (pad.value(axis) * f32::from(i16::MAX)).round() as i16;
    let trigger = |button| {
        pad.button_data(button)
            .map_or(0, |data| (data.value() * TRIGGER_MAX).round() as i16)
    };
    Event::Gamepad {
        device,
        buttons,
        axes: GamepadAxes {
            left_x: stick(Axis::LeftStickX),
            left_y: stick(Axis::LeftStickY),
            right_x: stick(Axis::RightStickX),
            right_y: stick(Axis::RightStickY),
            left_trigger: trigger(Button::LeftTrigger2),
            right_trigger: trigger(Button::RightTrigger2),
        },
    }
}

#[derive(Debug, Default)]
struct PadDeviceNumbers(Vec<(GamepadId, u8)>);

impl PadDeviceNumbers {
    fn device(&self, id: GamepadId) -> Option<u8> {
        self.0
            .iter()
            .find(|(pad, _)| *pad == id)
            .map(|(_, device)| *device)
    }

    fn pad(&self, device: u8) -> Option<GamepadId> {
        self.0
            .iter()
            .find(|(_, numbered)| *numbered == device)
            .map(|(pad, _)| *pad)
    }

    fn plug(&mut self, id: GamepadId) -> u8 {
        if let Some(device) = self.device(id) {
            return device;
        }
        let device = (0..=u8::MAX)
            .find(|device| self.pad(*device).is_none())
            .expect("fewer than 256 gamepads are connected");
        self.0.push((id, device));
        device
    }

    fn unplug(&mut self, id: GamepadId) -> Option<u8> {
        let device = self.device(id)?;
        self.0.retain(|(pad, _)| *pad != id);
        Some(device)
    }
}

fn rumble_effect(
    gilrs: &mut Gilrs,
    id: GamepadId,
    motors: MotorLevels,
) -> Result<Effect, gilrs::ff::Error> {
    let constant = |kind| BaseEffect {
        kind,
        ..BaseEffect::default()
    };
    // gilrs repeats an effect until it is dropped
    let effect = EffectBuilder::new()
        .add_effect(constant(BaseEffectType::Strong {
            magnitude: motors.low_frequency,
        }))
        .add_effect(constant(BaseEffectType::Weak {
            magnitude: motors.high_frequency,
        }))
        .gamepads(&[id])
        .finish(gilrs)?;
    effect.play()?;
    Ok(effect)
}

fn run_gamepads(
    states: &mpsc::UnboundedSender<Event>,
    mut rumbles: mpsc::UnboundedReceiver<GamepadRumble>,
) {
    let log = Target::category(GAMEPAD_CATEGORY);
    let mut gilrs = match Gilrs::new() {
        Ok(gilrs) => gilrs,
        Err(e) => {
            g2g_core::g2g_warn!(log, "no gamepads: {e}");
            return;
        }
    };
    let mut devices = PadDeviceNumbers::default();
    for (id, _) in gilrs.gamepads() {
        devices.plug(id);
    }
    let mut effects: BTreeMap<u8, Effect> = BTreeMap::new();
    while !states.is_closed() {
        while let Ok(GamepadRumble { device, motors }) = rumbles.try_recv() {
            effects.remove(&device);
            let Some(id) = devices.pad(device) else {
                continue;
            };
            if motors == MotorLevels::default() || !gilrs.gamepad(id).is_ff_supported() {
                continue;
            }
            match rumble_effect(&mut gilrs, id, motors) {
                Ok(effect) => {
                    effects.insert(device, effect);
                }
                Err(e) => g2g_core::g2g_warn!(log, "cannot rumble gamepad {device}: {e}"),
            }
        }
        let Some(event) = gilrs.next_event_blocking(Some(GAMEPAD_POLL_INTERVAL)) else {
            continue;
        };
        let state = match event.event {
            EventType::Connected
            | EventType::ButtonPressed(..)
            | EventType::ButtonReleased(..)
            | EventType::ButtonChanged(..)
            | EventType::AxisChanged(..) => {
                gamepad_state(devices.plug(event.id), &gilrs.gamepad(event.id))
            }
            EventType::Disconnected => {
                let Some(device) = devices.unplug(event.id) else {
                    continue;
                };
                effects.remove(&device);
                Event::DeviceRemoved {
                    kind: RemovableDevice::Gamepad,
                    device,
                }
            }
            _ => continue,
        };
        if states.send(state).is_err() {
            return;
        }
    }
}

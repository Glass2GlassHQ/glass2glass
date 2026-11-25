// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use evdev::uinput::{VirtualDevice, VirtualEventStream};
use evdev::{
    AbsInfo, AbsoluteAxisCode, AbsoluteAxisEvent, AttributeSet, BusType, EventSummary,
    FFEffectCode, FFEffectKind, FFReplay, InputEvent, InputId, KeyCode, KeyEvent, UInputCode,
    UinputAbsSetup,
};
use tokio::time::{Instant, Sleep};

use crate::injection::{GamepadRumble, InjectError, InjectEvents};
use crate::packet::{
    Event, GamepadAxes, MotorLevels, RemovableDevice, GAMEPAD_A, GAMEPAD_B, GAMEPAD_BACK,
    GAMEPAD_DPAD_DOWN, GAMEPAD_DPAD_LEFT, GAMEPAD_DPAD_RIGHT, GAMEPAD_DPAD_UP, GAMEPAD_GUIDE,
    GAMEPAD_LEFT_SHOULDER, GAMEPAD_LEFT_THUMB, GAMEPAD_RIGHT_SHOULDER, GAMEPAD_RIGHT_THUMB,
    GAMEPAD_START, GAMEPAD_X, GAMEPAD_Y,
};

pub const GAMEPAD_NAME: &str = "g2g-desktop Xbox 360 pad";
// sdl and steam pick their Xbox 360 mapping from this usb vendor and product
const XBOX_360_VENDOR: u16 = 0x045e;
const XBOX_360_PRODUCT: u16 = 0x028e;
const XBOX_360_VERSION: u16 = 0x0110;
// the linux xpad driver holds this many force feedback effects per pad
const RUMBLE_EFFECTS: u32 = 16;
// linux's memless force feedback layer adds the periodic waveforms and gain to every xpad pad, and gilrs requires them
const FORCE_FEEDBACK_CODES: [FFEffectCode; 8] = [
    FFEffectCode::FF_RUMBLE,
    FFEffectCode::FF_PERIODIC,
    FFEffectCode::FF_SQUARE,
    FFEffectCode::FF_TRIANGLE,
    FFEffectCode::FF_SINE,
    FFEffectCode::FF_SAW_UP,
    FFEffectCode::FF_SAW_DOWN,
    FFEffectCode::FF_GAIN,
];

// the linux xpad driver reports X and Y with the codes evdev names north and west
const BUTTON_KEYS: [(u32, KeyCode); 11] = [
    (GAMEPAD_A, KeyCode::BTN_SOUTH),
    (GAMEPAD_B, KeyCode::BTN_EAST),
    (GAMEPAD_X, KeyCode::BTN_NORTH),
    (GAMEPAD_Y, KeyCode::BTN_WEST),
    (GAMEPAD_LEFT_SHOULDER, KeyCode::BTN_TL),
    (GAMEPAD_RIGHT_SHOULDER, KeyCode::BTN_TR),
    (GAMEPAD_BACK, KeyCode::BTN_SELECT),
    (GAMEPAD_START, KeyCode::BTN_START),
    (GAMEPAD_GUIDE, KeyCode::BTN_MODE),
    (GAMEPAD_LEFT_THUMB, KeyCode::BTN_THUMBL),
    (GAMEPAD_RIGHT_THUMB, KeyCode::BTN_THUMBR),
];

// the ranges the linux xpad driver reports for an Xbox 360 pad
fn axis_setups() -> [UinputAbsSetup; 8] {
    let stick = |code| UinputAbsSetup::new(code, AbsInfo::new(0, -32768, 32767, 16, 128, 0));
    let trigger = |code| UinputAbsSetup::new(code, AbsInfo::new(0, 0, 255, 0, 0, 0));
    let hat = |code| UinputAbsSetup::new(code, AbsInfo::new(0, -1, 1, 0, 0, 0));
    [
        stick(AbsoluteAxisCode::ABS_X),
        stick(AbsoluteAxisCode::ABS_Y),
        stick(AbsoluteAxisCode::ABS_RX),
        stick(AbsoluteAxisCode::ABS_RY),
        trigger(AbsoluteAxisCode::ABS_Z),
        trigger(AbsoluteAxisCode::ABS_RZ),
        hat(AbsoluteAxisCode::ABS_HAT0X),
        hat(AbsoluteAxisCode::ABS_HAT0Y),
    ]
}

fn build_gamepad() -> io::Result<VirtualDevice> {
    let keys = AttributeSet::from_iter(BUTTON_KEYS.map(|(_, key)| key));
    let mut builder = VirtualDevice::builder()?
        .name(GAMEPAD_NAME)
        .input_id(InputId::new(
            BusType::BUS_USB,
            XBOX_360_VENDOR,
            XBOX_360_PRODUCT,
            XBOX_360_VERSION,
        ))
        .with_keys(&keys)?
        .with_ff(&AttributeSet::from_iter(FORCE_FEEDBACK_CODES))?
        .with_ff_effects_max(RUMBLE_EFFECTS);
    for setup in axis_setups() {
        builder = builder.with_absolute_axis(&setup)?;
    }
    builder.build()
}

#[derive(Debug, Clone, Copy)]
struct RumbleEffect {
    motors: MotorLevels,
    replay: FFReplay,
}

#[derive(Debug, Clone, Copy)]
struct PlayingEffect {
    starts: Instant,
    ends: Option<Instant>,
}

impl PlayingEffect {
    fn is_on(&self, now: Instant) -> bool {
        self.starts <= now && self.ends.is_none_or(|ends| now < ends)
    }
}

#[derive(Debug)]
struct RumbleEffects {
    uploaded: BTreeMap<i16, RumbleEffect>,
    playing: BTreeMap<i16, PlayingEffect>,
    gain: u16,
}

impl Default for RumbleEffects {
    fn default() -> Self {
        Self {
            uploaded: BTreeMap::new(),
            playing: BTreeMap::new(),
            gain: u16::MAX,
        }
    }
}

impl RumbleEffects {
    fn upload(&mut self, id: i16, effect: RumbleEffect) {
        self.uploaded.insert(id, effect);
    }

    fn erase(&mut self, id: i16) {
        self.uploaded.remove(&id);
        self.playing.remove(&id);
    }

    // linux plays an effect `count` times in a row and stops it with a count of zero
    fn play(&mut self, id: i16, count: i32, now: Instant) {
        if count == 0 {
            self.playing.remove(&id);
            return;
        }
        // a process on the host can play an id it never uploaded
        let Some(replay) = self.uploaded.get(&id).map(|effect| effect.replay) else {
            return;
        };
        let starts = now + Duration::from_millis(u64::from(replay.delay));
        // linux plays an effect of length zero until it is stopped
        let ends = (replay.length != 0).then(|| {
            starts + Duration::from_millis(u64::from(replay.length)) * count.unsigned_abs()
        });
        self.playing.insert(id, PlayingEffect { starts, ends });
    }

    // linux adds the magnitudes of rumble effects that play at once, then applies the gain
    fn motors(&self, now: Instant) -> MotorLevels {
        let sum = self
            .playing
            .iter()
            .filter(|(_, playing)| playing.is_on(now))
            .map(|(id, _)| self.uploaded[id].motors)
            .fold(MotorLevels::default(), |sum, motors| MotorLevels {
                low_frequency: sum.low_frequency.saturating_add(motors.low_frequency),
                high_frequency: sum.high_frequency.saturating_add(motors.high_frequency),
            });
        let scaled = |level: u16| {
            let level = u32::from(level) * u32::from(self.gain) / u32::from(u16::MAX);
            u16::try_from(level).expect("a gain of at most one keeps the level in range")
        };
        MotorLevels {
            low_frequency: scaled(sum.low_frequency),
            high_frequency: scaled(sum.high_frequency),
        }
    }

    fn next_change(&self, now: Instant) -> Option<Instant> {
        self.playing
            .values()
            .flat_map(|playing| [Some(playing.starts), playing.ends])
            .flatten()
            .filter(|&at| at > now)
            .min()
    }
}

struct Pad {
    device: VirtualEventStream,
    effects: RumbleEffects,
    sent: MotorLevels,
    next_change: Pin<Box<Sleep>>,
}

impl std::fmt::Debug for Pad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pad")
            .field("effects", &self.effects)
            .field("sent", &self.sent)
            .finish_non_exhaustive()
    }
}

impl Pad {
    fn new() -> io::Result<Self> {
        Ok(Self {
            device: build_gamepad()?.into_event_stream()?,
            effects: RumbleEffects::default(),
            sent: MotorLevels::default(),
            next_change: Box::pin(tokio::time::sleep(Duration::ZERO)),
        })
    }

    fn handle(&mut self, event: InputEvent) -> io::Result<()> {
        match event.destructure() {
            EventSummary::UInput(upload, UInputCode::UI_FF_UPLOAD, _) => {
                let upload = self.device.device_mut().process_ff_upload(upload)?;
                let effect = upload.effect();
                let motors = match effect.kind {
                    FFEffectKind::Rumble {
                        strong_magnitude,
                        weak_magnitude,
                    } => MotorLevels {
                        low_frequency: strong_magnitude,
                        high_frequency: weak_magnitude,
                    },
                    // linux's memless layer drives both motors at a periodic effect's magnitude
                    FFEffectKind::Periodic { magnitude, .. } => {
                        let level = u32::from(magnitude.unsigned_abs()) * u32::from(u16::MAX)
                            / i16::MAX as u32;
                        let level = u16::try_from(level).unwrap_or(u16::MAX);
                        MotorLevels {
                            low_frequency: level,
                            high_frequency: level,
                        }
                    }
                    _ => unreachable!("linux uploads only the effect types the pad declares"),
                };
                let replay = effect.replay;
                self.effects
                    .upload(upload.effect_id(), RumbleEffect { motors, replay });
            }
            EventSummary::UInput(erase, UInputCode::UI_FF_ERASE, _) => {
                let erase = self.device.device_mut().process_ff_erase(erase)?;
                // linux effect ids are below RUMBLE_EFFECTS
                self.effects.erase(erase.effect_id() as i16);
            }
            EventSummary::ForceFeedback(_, FFEffectCode::FF_GAIN, gain) => {
                // linux ignores a gain past 0xffff
                if let Ok(gain) = u16::try_from(gain) {
                    self.effects.gain = gain;
                }
            }
            EventSummary::ForceFeedback(_, effect, count) => {
                self.effects.play(effect.0 as i16, count, Instant::now());
            }
            _ => {}
        }
        Ok(())
    }

    fn poll_rumble(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<MotorLevels>> {
        while let Poll::Ready(event) = self.device.poll_event(cx) {
            self.handle(event?)?;
        }
        let now = Instant::now();
        if let Some(at) = self.effects.next_change(now) {
            self.next_change.as_mut().reset(at);
            if self.next_change.as_mut().poll(cx).is_ready() {
                cx.waker().wake_by_ref();
            }
        }
        let motors = self.effects.motors(now);
        if motors == self.sent {
            return Poll::Pending;
        }
        self.sent = motors;
        Poll::Ready(Ok(motors))
    }
}

fn pressed_toward(buttons: u32, negative: u32, positive: u32) -> i32 {
    i32::from(buttons & positive != 0) - i32::from(buttons & negative != 0)
}

// linux reports a stick pushed up as negative Y, xinput as positive
fn state_events(buttons: u32, axes: GamepadAxes) -> Vec<InputEvent> {
    let keys = BUTTON_KEYS
        .iter()
        .map(|(flag, key)| *KeyEvent::new(*key, i32::from(buttons & flag != 0)));
    let axis = |code, value| *AbsoluteAxisEvent::new(code, value);
    let positions = [
        axis(AbsoluteAxisCode::ABS_X, i32::from(axes.left_x)),
        axis(AbsoluteAxisCode::ABS_Y, i32::from(!axes.left_y)),
        axis(AbsoluteAxisCode::ABS_RX, i32::from(axes.right_x)),
        axis(AbsoluteAxisCode::ABS_RY, i32::from(!axes.right_y)),
        axis(AbsoluteAxisCode::ABS_Z, i32::from(axes.left_trigger)),
        axis(AbsoluteAxisCode::ABS_RZ, i32::from(axes.right_trigger)),
        axis(
            AbsoluteAxisCode::ABS_HAT0X,
            pressed_toward(buttons, GAMEPAD_DPAD_LEFT, GAMEPAD_DPAD_RIGHT),
        ),
        axis(
            AbsoluteAxisCode::ABS_HAT0Y,
            pressed_toward(buttons, GAMEPAD_DPAD_UP, GAMEPAD_DPAD_DOWN),
        ),
    ];
    keys.chain(positions).collect()
}

#[derive(Debug, Default)]
pub struct UinputGamepads {
    pads: BTreeMap<u8, Pad>,
    unavailable: bool,
}

impl InjectEvents for UinputGamepads {
    fn inject(&mut self, event: &Event) -> Result<(), InjectError> {
        match event {
            Event::Gamepad {
                device,
                buttons,
                axes,
            } => {
                // opening /dev/uinput failed once and that failure was reported
                if self.unavailable {
                    return Ok(());
                }
                let pad = match self.pads.entry(*device) {
                    Entry::Occupied(pad) => pad.into_mut(),
                    Entry::Vacant(slot) => match Pad::new() {
                        Ok(pad) => slot.insert(pad),
                        Err(e) => {
                            self.unavailable = true;
                            return Err(InjectError::GamepadUnavailable(e));
                        }
                    },
                };
                pad.device
                    .device_mut()
                    .emit(&state_events(*buttons, *axes))
                    .map_err(InjectError::Refused)?;
            }
            Event::DeviceRemoved {
                kind: RemovableDevice::Gamepad,
                device,
            } => {
                self.pads.remove(device);
            }
            _ => return Err(InjectError::NotInjected(event.kind())),
        }
        Ok(())
    }

    fn poll_rumble(&mut self, cx: &mut Context<'_>) -> Poll<Result<GamepadRumble, InjectError>> {
        for (device, pad) in &mut self.pads {
            if let Poll::Ready(motors) = pad.poll_rumble(cx) {
                let motors = motors.map_err(InjectError::Device)?;
                return Poll::Ready(Ok(GamepadRumble {
                    device: *device,
                    motors,
                }));
            }
        }
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn effect(low_frequency: u16, high_frequency: u16, delay: u16, length: u16) -> RumbleEffect {
        RumbleEffect {
            motors: MotorLevels {
                low_frequency,
                high_frequency,
            },
            replay: FFReplay { length, delay },
        }
    }

    #[test]
    fn playing_effects_add_up_within_their_replay_times() {
        let now = Instant::now();
        let at = |milliseconds| now + Duration::from_millis(milliseconds);
        let levels = |low_frequency, high_frequency| MotorLevels {
            low_frequency,
            high_frequency,
        };
        let mut effects = RumbleEffects::default();
        effects.upload(0, effect(0x9000, 0x1000, 0, 100));
        effects.upload(1, effect(0x9000, 0x2000, 50, 0));
        effects.play(0, 2, now);
        effects.play(1, 1, now);

        assert_eq!(effects.motors(now), levels(0x9000, 0x1000));
        assert_eq!(effects.next_change(now), Some(at(50)));
        assert_eq!(effects.motors(at(50)), levels(u16::MAX, 0x3000));
        assert_eq!(
            effects.next_change(at(50)),
            Some(at(200)),
            "two plays of 100 ms"
        );
        assert_eq!(effects.motors(at(200)), levels(0x9000, 0x2000));
        assert_eq!(
            effects.next_change(at(200)),
            None,
            "length zero plays until stopped"
        );

        effects.play(1, 0, at(300));
        assert_eq!(effects.motors(at(300)), MotorLevels::default());
        effects.play(0, 1, at(300));
        effects.erase(0);
        assert_eq!(effects.motors(at(300)), MotorLevels::default());
    }
}

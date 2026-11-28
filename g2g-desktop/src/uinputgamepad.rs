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
    FFEffectCode, FFEffectData, FFEffectKind, FFEnvelope, FFReplay, InputEvent, InputId, KeyCode,
    KeyEvent, UInputCode, UinputAbsSetup,
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
// Linux's memless driver updates the envelope every 50 ms during a ramp.
const ENVELOPE_STEP_MS: u64 = 50;
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
enum RumbleStrength {
    Motors(MotorLevels),
    Periodic {
        magnitude: u16,
        envelope: FFEnvelope,
    },
}

#[derive(Debug, Clone, Copy)]
struct RumbleEffect {
    strength: RumbleStrength,
    replay: FFReplay,
}

impl From<FFEffectData> for RumbleEffect {
    fn from(effect: FFEffectData) -> Self {
        let strength = match effect.kind {
            FFEffectKind::Rumble {
                strong_magnitude,
                weak_magnitude,
            } => RumbleStrength::Motors(MotorLevels {
                low_frequency: strong_magnitude,
                high_frequency: weak_magnitude,
            }),
            FFEffectKind::Periodic {
                magnitude,
                envelope,
                ..
            } => RumbleStrength::Periodic {
                magnitude: magnitude.unsigned_abs(),
                envelope,
            },
            _ => unreachable!("linux uploads only the effect types the pad declares"),
        };
        Self {
            strength,
            replay: effect.replay,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PlayingEffect {
    starts: Instant,
    ends: Option<Instant>,
    replay_ms: u16,
}

impl PlayingEffect {
    fn sample(self, strength: RumbleStrength, now: Instant) -> (MotorLevels, Option<Instant>) {
        if now < self.starts {
            return (MotorLevels::default(), Some(self.starts));
        }
        if self.ends.is_some_and(|end| now >= end) {
            return (MotorLevels::default(), None);
        }
        match strength {
            RumbleStrength::Motors(motors) => (motors, self.ends),
            RumbleStrength::Periodic {
                magnitude,
                envelope,
            } => self.sample_periodic(magnitude, envelope, now),
        }
    }

    fn sample_periodic(
        self,
        magnitude: u16,
        envelope: FFEnvelope,
        now: Instant,
    ) -> (MotorLevels, Option<Instant>) {
        let elapsed_ms = u64::try_from((now - self.starts).as_millis()).unwrap_or(u64::MAX);
        let replay_ms = u64::from(self.replay_ms);
        let position_ms = elapsed_ms.checked_rem(replay_ms).unwrap_or(elapsed_ms);
        let attack_ms = u64::from(envelope.attack_length);
        let fade_ms = u64::from(envelope.fade_length);

        // Each replay starts its own attack. An indefinite effect has no fade.
        let (level, next_ms) = if position_ms < attack_ms {
            let attack_end = if replay_ms == 0 {
                attack_ms
            } else {
                attack_ms.min(replay_ms)
            };
            (
                ramp_level(envelope.attack_level, magnitude, position_ms, attack_ms),
                Some((position_ms + ENVELOPE_STEP_MS).min(attack_end)),
            )
        } else if replay_ms != 0 && fade_ms != 0 {
            let remaining_ms = replay_ms - position_ms;
            if remaining_ms <= fade_ms {
                (
                    ramp_level(envelope.fade_level, magnitude, remaining_ms, fade_ms),
                    Some((position_ms + ENVELOPE_STEP_MS).min(replay_ms)),
                )
            } else {
                (magnitude, Some(replay_ms - fade_ms))
            }
        } else {
            (
                magnitude,
                (replay_ms != 0 && attack_ms != 0).then_some(replay_ms),
            )
        };

        // Linux's memless layer drives both motors at the periodic magnitude.
        let level = u32::from(level) * u32::from(u16::MAX) / i16::MAX as u32;
        let level = u16::try_from(level).unwrap_or(u16::MAX);
        let next = next_ms.map(|position| {
            self.starts + Duration::from_millis(elapsed_ms - position_ms + position)
        });
        (
            MotorLevels {
                low_frequency: level,
                high_frequency: level,
            },
            next.into_iter().chain(self.ends).min(),
        )
    }
}

fn ramp_level(endpoint: u16, magnitude: u16, distance_ms: u64, duration_ms: u64) -> u16 {
    let endpoint = i64::from(endpoint.min(i16::MAX as u16));
    let delta = i64::from(magnitude) - endpoint;
    (endpoint + delta * distance_ms as i64 / duration_ms as i64) as u16
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
        self.playing.insert(
            id,
            PlayingEffect {
                starts,
                ends,
                replay_ms: replay.length,
            },
        );
    }

    // linux adds the magnitudes of rumble effects that play at once, then applies the gain
    fn sample(&self, now: Instant) -> (MotorLevels, Option<Instant>) {
        let mut sum = MotorLevels::default();
        let mut next_change = None;
        for (id, playing) in &self.playing {
            let (motors, next) = playing.sample(self.uploaded[id].strength, now);
            sum.low_frequency = sum.low_frequency.saturating_add(motors.low_frequency);
            sum.high_frequency = sum.high_frequency.saturating_add(motors.high_frequency);
            next_change = next_change.into_iter().chain(next).min();
        }
        let scaled = |level: u16| {
            let level = u32::from(level) * u32::from(self.gain) / u32::from(u16::MAX);
            u16::try_from(level).expect("a gain of at most one keeps the level in range")
        };
        (
            MotorLevels {
                low_frequency: scaled(sum.low_frequency),
                high_frequency: scaled(sum.high_frequency),
            },
            next_change,
        )
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
                self.effects
                    .upload(upload.effect_id(), upload.effect().into());
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
        let (motors, next_change) = self.effects.sample(now);
        if let Some(at) = next_change {
            self.next_change.as_mut().reset(at);
            if self.next_change.as_mut().poll(cx).is_ready() {
                cx.waker().wake_by_ref();
            }
        }
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
    use evdev::{FFTrigger, FFWaveform};

    fn levels(low_frequency: u16, high_frequency: u16) -> MotorLevels {
        MotorLevels {
            low_frequency,
            high_frequency,
        }
    }

    fn effect(low_frequency: u16, high_frequency: u16, delay: u16, length: u16) -> RumbleEffect {
        RumbleEffect {
            strength: RumbleStrength::Motors(levels(low_frequency, high_frequency)),
            replay: FFReplay { length, delay },
        }
    }

    fn periodic(magnitude: i16, delay: u16, length: u16, envelope: FFEnvelope) -> RumbleEffect {
        FFEffectData {
            direction: 0,
            trigger: FFTrigger::default(),
            replay: FFReplay { length, delay },
            kind: FFEffectKind::Periodic {
                waveform: FFWaveform::Sine,
                period: 20,
                magnitude,
                offset: 0,
                phase: 0,
                envelope,
            },
        }
        .into()
    }

    #[test]
    fn playing_effects_add_up_within_their_replay_times() {
        let now = Instant::now();
        let at = |milliseconds| now + Duration::from_millis(milliseconds);
        let mut effects = RumbleEffects::default();
        effects.upload(0, effect(0x9000, 0x1000, 0, 100));
        effects.upload(1, effect(0x9000, 0x2000, 50, 0));
        effects.play(0, 2, now);
        effects.play(1, 1, now);

        assert_eq!(effects.sample(now), (levels(0x9000, 0x1000), Some(at(50))));
        assert_eq!(
            effects.sample(at(50)),
            (levels(u16::MAX, 0x3000), Some(at(200))),
            "two plays of 100 ms"
        );
        assert_eq!(
            effects.sample(at(200)),
            (levels(0x9000, 0x2000), None),
            "length zero plays until stopped"
        );

        effects.play(1, 0, at(300));
        assert_eq!(effects.sample(at(300)), (MotorLevels::default(), None));
        effects.play(0, 1, at(300));
        effects.erase(0);
        assert_eq!(effects.sample(at(300)), (MotorLevels::default(), None));
    }

    #[test]
    fn periodic_effect_schedules_attack_and_fade_on_each_replay() {
        let now = Instant::now();
        let at = |ms| now + Duration::from_millis(ms);
        let mut effects = RumbleEffects::default();
        let envelope = FFEnvelope {
            attack_length: 100,
            attack_level: 0,
            fade_length: 120,
            fade_level: 4_000,
        };
        effects.upload(0, periodic(-24_000, 25, 300, envelope));
        effects.play(0, 2, now);

        for (elapsed, level, next) in [
            (0, 0, Some(25)),
            (25, 0, Some(75)),
            (75, 24_000, Some(125)),
            (125, 48_000, Some(205)),
            (205, 48_000, Some(255)),
            (255, 31_332, Some(305)),
            (305, 14_666, Some(325)),
            (325, 0, Some(375)),
            (605, 14_666, Some(625)),
            (625, 0, None),
        ] {
            assert_eq!(
                effects.sample(at(elapsed)),
                (levels(level, level), next.map(at)),
                "at {elapsed} ms"
            );
        }
    }

    #[test]
    fn an_indefinite_periodic_effect_stays_at_its_magnitude_after_attack() {
        let now = Instant::now();
        let at = |ms| now + Duration::from_millis(ms);
        let mut effects = RumbleEffects::default();
        let envelope = FFEnvelope {
            attack_length: 30,
            attack_level: u16::MAX,
            fade_length: 100,
            fade_level: 0,
        };
        effects.upload(0, periodic(15_000, 0, 0, envelope));
        effects.play(0, 1, now);

        assert_eq!(
            effects.sample(now),
            (levels(u16::MAX, u16::MAX), Some(at(30)))
        );
        assert_eq!(effects.sample(at(30)), (levels(30_000, 30_000), None));
        assert_eq!(effects.sample(at(500)), (levels(30_000, 30_000), None));
    }
}

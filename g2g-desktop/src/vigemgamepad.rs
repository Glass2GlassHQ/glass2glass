// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::io;
use std::rc::Rc;

use vigem_client::{Client, TargetId, XButtons, XGamepad, Xbox360Wired};

use crate::injection::{InjectError, InjectEvents};
use crate::packet::{Event, GamepadAxes, RemovableDevice};

fn report(buttons: u32, axes: GamepadAxes) -> Result<XGamepad, InjectError> {
    let trigger =
        |value: i16| u8::try_from(value).map_err(|_| InjectError::TriggerOutOfRange(value));
    Ok(XGamepad {
        // an Xbox 360 pad has no buttons past the sixteen xinput bits
        buttons: XButtons(buttons as u16),
        left_trigger: trigger(axes.left_trigger)?,
        right_trigger: trigger(axes.right_trigger)?,
        thumb_lx: axes.left_x,
        thumb_ly: axes.left_y,
        thumb_rx: axes.right_x,
        thumb_ry: axes.right_y,
    })
}

fn plug_in(client: &Rc<Client>) -> Result<Xbox360Wired<Rc<Client>>, vigem_client::Error> {
    let mut pad = Xbox360Wired::new(Rc::clone(client), TargetId::XBOX360_WIRED);
    pad.plugin()?;
    pad.wait_ready()?;
    Ok(pad)
}

#[derive(Default)]
pub struct ViGEmGamepads {
    client: Option<Rc<Client>>,
    pads: BTreeMap<u8, Xbox360Wired<Rc<Client>>>,
    unavailable: bool,
}

impl std::fmt::Debug for ViGEmGamepads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViGEmGamepads")
            .field("pads", &self.pads.keys())
            .field("unavailable", &self.unavailable)
            .finish_non_exhaustive()
    }
}

impl ViGEmGamepads {
    fn pad(&mut self, device: u8) -> Result<&mut Xbox360Wired<Rc<Client>>, vigem_client::Error> {
        let client = match &self.client {
            Some(client) => client,
            None => self.client.insert(Rc::new(Client::connect()?)),
        };
        match self.pads.entry(device) {
            Entry::Occupied(pad) => Ok(pad.into_mut()),
            Entry::Vacant(slot) => Ok(slot.insert(plug_in(client)?)),
        }
    }
}

impl InjectEvents for ViGEmGamepads {
    fn inject(&mut self, event: &Event) -> Result<(), InjectError> {
        match event {
            Event::Gamepad {
                device,
                buttons,
                axes,
            } => {
                // ViGEmBus refused a pad once and that failure was reported
                if self.unavailable {
                    return Ok(());
                }
                let pad = match self.pad(*device) {
                    Ok(pad) => pad,
                    Err(e) => {
                        self.unavailable = true;
                        return Err(InjectError::GamepadUnavailable(io::Error::other(e)));
                    }
                };
                pad.update(&report(*buttons, *axes)?)
                    .map_err(|e| InjectError::Refused(io::Error::other(e)))?;
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
}

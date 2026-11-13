// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use tokio::sync::mpsc;

use crate::packet::{Event, EventKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Local,
    Remote,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Routed {
    Delivered,
    NoConsumer,
    Unfocused,
    ShortcutPressed(u16),
    ShortcutReleased(u16),
}

#[derive(Debug)]
struct Subscription {
    origin: Origin,
    kind: EventKind,
    consumer: mpsc::UnboundedSender<Event>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HeldKey {
    device: u8,
    usage: u16,
}

fn is_user_input(kind: EventKind) -> bool {
    matches!(
        kind,
        EventKind::Key
            | EventKind::Button
            | EventKind::Move
            | EventKind::Position
            | EventKind::Wheel
            | EventKind::Gamepad
    )
}

#[derive(Debug)]
pub struct InputRouter {
    subscriptions: Vec<Subscription>,
    focused: bool,
    shortcut_usages: Vec<u16>,
    swallowed_keys: Vec<HeldKey>,
    held_keys: Vec<HeldKey>,
    held_buttons: Vec<u8>,
}

impl InputRouter {
    pub fn new(shortcut_usages: Vec<u16>) -> Self {
        Self {
            subscriptions: Vec::new(),
            focused: true,
            shortcut_usages,
            swallowed_keys: Vec::new(),
            held_keys: Vec::new(),
            held_buttons: Vec::new(),
        }
    }

    pub fn subscribe(
        &mut self,
        origin: Origin,
        kinds: &[EventKind],
    ) -> mpsc::UnboundedReceiver<Event> {
        let (consumer, receiver) = mpsc::unbounded_channel();
        for kind in kinds {
            self.subscriptions.push(Subscription {
                origin,
                kind: *kind,
                consumer: consumer.clone(),
            });
        }
        receiver
    }

    pub fn set_focused(&mut self, focused: bool) {
        if self.focused && !focused {
            // hosts keep a key held down until its key-up arrives
            for key in core::mem::take(&mut self.held_keys) {
                let release = Event::Key {
                    device: key.device,
                    usage: key.usage,
                    down: false,
                };
                self.deliver(Origin::Local, release);
            }
            for button in core::mem::take(&mut self.held_buttons) {
                self.deliver(
                    Origin::Local,
                    Event::Button {
                        button,
                        down: false,
                    },
                );
            }
            self.swallowed_keys.clear();
        }
        self.focused = focused;
    }

    pub fn push(&mut self, origin: Origin, event: Event) -> Routed {
        if origin == Origin::Remote || !is_user_input(event.kind()) {
            return self.deliver(origin, event);
        }
        if !self.focused {
            return Routed::Unfocused;
        }
        match event {
            Event::Key {
                device,
                usage,
                down,
            } => {
                let key = HeldKey { device, usage };
                if down && self.shortcut_usages.contains(&usage) {
                    self.swallowed_keys.push(key);
                    return Routed::ShortcutPressed(usage);
                }
                if !down && self.swallowed_keys.contains(&key) {
                    self.swallowed_keys.retain(|swallowed| *swallowed != key);
                    return Routed::ShortcutReleased(usage);
                }
                self.held_keys.retain(|held| *held != key);
                if down {
                    self.held_keys.push(key);
                }
            }
            Event::Button { button, down } => {
                self.held_buttons.retain(|held| *held != button);
                if down {
                    self.held_buttons.push(button);
                }
            }
            _ => {}
        }
        self.deliver(origin, event)
    }

    fn deliver(&mut self, origin: Origin, event: Event) -> Routed {
        let kind = event.kind();
        self.subscriptions
            .retain(|subscription| !subscription.consumer.is_closed());
        let mut delivered = false;
        for subscription in &self.subscriptions {
            if subscription.origin == origin && subscription.kind == kind {
                delivered |= subscription.consumer.send(event.clone()).is_ok();
            }
        }
        if delivered {
            Routed::Delivered
        } else {
            Routed::NoConsumer
        }
    }
}

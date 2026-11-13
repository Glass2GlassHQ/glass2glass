// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::fmt;

use g2g_plugins::datachannel::MAX_MESSAGE_SIZE;

const HEADER_LEN: usize = 8;
const FROM_HOST: u8 = 1 << 0;
const DEVICE_REMOVED: u8 = 1 << 1;
const KNOWN_FLAGS: u8 = FROM_HOST | DEVICE_REMOVED;
const RGBA_BYTES_PER_PIXEL: usize = 4;
const CURSOR_HEADER_LEN: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Host,
    Client,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    Key = 1,
    Button = 2,
    Move = 3,
    Position = 4,
    Wheel = 5,
    Gamepad = 6,
    Rumble = 7,
    Cursor = 8,
    Clipboard = 9,
}

const EVENT_KINDS: [EventKind; 9] = [
    EventKind::Key,
    EventKind::Button,
    EventKind::Move,
    EventKind::Position,
    EventKind::Wheel,
    EventKind::Gamepad,
    EventKind::Rumble,
    EventKind::Cursor,
    EventKind::Clipboard,
];

impl EventKind {
    fn from_wire(byte: u8) -> Result<Self, PacketError> {
        EVENT_KINDS
            .into_iter()
            .find(|kind| *kind as u8 == byte)
            .ok_or(PacketError::UnknownKind(byte))
    }

    fn index(self) -> usize {
        self as usize - 1
    }

    fn fixed_payload_len(self) -> Option<usize> {
        match self {
            EventKind::Key => Some(4),
            EventKind::Button => Some(2),
            EventKind::Move | EventKind::Wheel => Some(8),
            EventKind::Position => Some(9),
            EventKind::Gamepad => Some(17),
            EventKind::Rumble => Some(7),
            EventKind::Cursor | EventKind::Clipboard => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GamepadAxes {
    pub left_x: i16,
    pub left_y: i16,
    pub right_x: i16,
    pub right_y: i16,
    pub left_trigger: i16,
    pub right_trigger: i16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CursorImage {
    width: u16,
    height: u16,
    hotspot_x: i16,
    hotspot_y: i16,
    rgba: Vec<u8>,
}

impl CursorImage {
    pub fn new(
        width: u16,
        height: u16,
        hotspot_x: i16,
        hotspot_y: i16,
        rgba: Vec<u8>,
    ) -> Result<Self, PacketError> {
        let expected = usize::from(width) * usize::from(height) * RGBA_BYTES_PER_PIXEL;
        if rgba.len() != expected {
            return Err(PacketError::CursorSize {
                width,
                height,
                rgba_len: rgba.len(),
            });
        }
        Ok(Self {
            width,
            height,
            hotspot_x,
            hotspot_y,
            rgba,
        })
    }

    pub fn width(&self) -> u16 {
        self.width
    }

    pub fn height(&self) -> u16 {
        self.height
    }

    pub fn hotspot(&self) -> (i16, i16) {
        (self.hotspot_x, self.hotspot_y)
    }

    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Key {
        device: u8,
        usage: u16,
        down: bool,
    },
    Button {
        button: u8,
        down: bool,
    },
    Move {
        dx: i32,
        dy: i32,
    },
    Position {
        x: i32,
        y: i32,
        display: u8,
    },
    Wheel {
        dx: i32,
        dy: i32,
    },
    Gamepad {
        device: u8,
        buttons: u32,
        axes: GamepadAxes,
    },
    Rumble {
        device: u8,
        low: u16,
        high: u16,
        duration_ms: u16,
    },
    Cursor(CursorImage),
    Clipboard(String),
    DeviceRemoved {
        kind: RemovableDevice,
        device: u8,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemovableDevice {
    Keyboard,
    Gamepad,
}

impl Event {
    pub fn kind(&self) -> EventKind {
        match self {
            Event::Key { .. } => EventKind::Key,
            Event::Button { .. } => EventKind::Button,
            Event::Move { .. } => EventKind::Move,
            Event::Position { .. } => EventKind::Position,
            Event::Wheel { .. } => EventKind::Wheel,
            Event::Gamepad { .. } => EventKind::Gamepad,
            Event::Rumble { .. } => EventKind::Rumble,
            Event::Cursor(_) => EventKind::Cursor,
            Event::Clipboard(_) => EventKind::Clipboard,
            Event::DeviceRemoved {
                kind: RemovableDevice::Keyboard,
                ..
            } => EventKind::Key,
            Event::DeviceRemoved {
                kind: RemovableDevice::Gamepad,
                ..
            } => EventKind::Gamepad,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PacketError {
    Truncated {
        len: usize,
    },
    UnknownKind(u8),
    UnknownFlags(u8),
    WrongDirection,
    LengthMismatch {
        declared: usize,
        actual: usize,
    },
    PayloadLength {
        kind: EventKind,
        expected: usize,
        actual: usize,
    },
    InvalidBool(u8),
    CursorSize {
        width: u16,
        height: u16,
        rgba_len: usize,
    },
    InvalidUtf8,
    TooLarge {
        len: usize,
    },
    SequenceExhausted,
}

impl fmt::Display for PacketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PacketError::Truncated { len } => {
                write!(
                    f,
                    "a {len} byte packet is shorter than the {HEADER_LEN} byte header"
                )
            }
            PacketError::UnknownKind(byte) => write!(f, "unknown packet type {byte}"),
            PacketError::UnknownFlags(flags) => write!(f, "unknown packet flags {flags:#04x}"),
            PacketError::WrongDirection => {
                write!(
                    f,
                    "packet direction flag names the receiving side as its sender"
                )
            }
            PacketError::LengthMismatch { declared, actual } => write!(
                f,
                "payload length field is {declared} but {actual} payload bytes follow"
            ),
            PacketError::PayloadLength {
                kind,
                expected,
                actual,
            } => write!(f, "{kind:?} payload is {actual} bytes, expected {expected}"),
            PacketError::InvalidBool(byte) => write!(f, "pressed byte is {byte}, not 0 or 1"),
            PacketError::CursorSize {
                width,
                height,
                rgba_len,
            } => write!(
                f,
                "{width}x{height} cursor has {rgba_len} RGBA bytes, expected {}",
                usize::from(*width) * usize::from(*height) * RGBA_BYTES_PER_PIXEL
            ),
            PacketError::InvalidUtf8 => write!(f, "clipboard payload is not UTF-8"),
            PacketError::TooLarge { len } => write!(
                f,
                "a {len} byte packet exceeds the {MAX_MESSAGE_SIZE} byte data-channel message"
            ),
            PacketError::SequenceExhausted => write!(f, "packet sequence number reached u32::MAX"),
        }
    }
}

impl std::error::Error for PacketError {}

fn flags_for(side: Side) -> u8 {
    match side {
        Side::Host => FROM_HOST,
        Side::Client => 0,
    }
}

#[derive(Debug)]
pub struct PacketWriter {
    side: Side,
    next_sequence: u32,
}

impl PacketWriter {
    pub fn new(side: Side) -> Self {
        Self {
            side,
            next_sequence: 0,
        }
    }

    pub fn write(&mut self, event: &Event) -> Result<Vec<u8>, PacketError> {
        let mut flags = flags_for(self.side);
        let mut payload = Vec::new();
        match event {
            Event::Key {
                device,
                usage,
                down,
            } => {
                payload.push(*device);
                payload.extend_from_slice(&usage.to_le_bytes());
                payload.push(u8::from(*down));
            }
            Event::Button { button, down } => {
                payload.push(*button);
                payload.push(u8::from(*down));
            }
            Event::Move { dx, dy } | Event::Wheel { dx, dy } => {
                payload.extend_from_slice(&dx.to_le_bytes());
                payload.extend_from_slice(&dy.to_le_bytes());
            }
            Event::Position { x, y, display } => {
                payload.extend_from_slice(&x.to_le_bytes());
                payload.extend_from_slice(&y.to_le_bytes());
                payload.push(*display);
            }
            Event::Gamepad {
                device,
                buttons,
                axes,
            } => {
                payload.push(*device);
                payload.extend_from_slice(&buttons.to_le_bytes());
                for axis in [
                    axes.left_x,
                    axes.left_y,
                    axes.right_x,
                    axes.right_y,
                    axes.left_trigger,
                    axes.right_trigger,
                ] {
                    payload.extend_from_slice(&axis.to_le_bytes());
                }
            }
            Event::Rumble {
                device,
                low,
                high,
                duration_ms,
            } => {
                payload.push(*device);
                payload.extend_from_slice(&low.to_le_bytes());
                payload.extend_from_slice(&high.to_le_bytes());
                payload.extend_from_slice(&duration_ms.to_le_bytes());
            }
            Event::Cursor(cursor) => {
                payload.extend_from_slice(&cursor.width.to_le_bytes());
                payload.extend_from_slice(&cursor.height.to_le_bytes());
                payload.extend_from_slice(&cursor.hotspot_x.to_le_bytes());
                payload.extend_from_slice(&cursor.hotspot_y.to_le_bytes());
                payload.extend_from_slice(&cursor.rgba);
            }
            Event::Clipboard(text) => payload.extend_from_slice(text.as_bytes()),
            Event::DeviceRemoved { device, .. } => {
                flags |= DEVICE_REMOVED;
                payload.push(*device);
            }
        }

        let len = HEADER_LEN + payload.len();
        let payload_len =
            u16::try_from(payload.len()).map_err(|_| PacketError::TooLarge { len })?;
        if len > MAX_MESSAGE_SIZE {
            return Err(PacketError::TooLarge { len });
        }
        let sequence = self.next_sequence;
        self.next_sequence = sequence
            .checked_add(1)
            .ok_or(PacketError::SequenceExhausted)?;

        let mut packet = Vec::with_capacity(len);
        packet.push(event.kind() as u8);
        packet.push(flags);
        packet.extend_from_slice(&sequence.to_le_bytes());
        packet.extend_from_slice(&payload_len.to_le_bytes());
        packet.extend_from_slice(&payload);
        Ok(packet)
    }
}

struct Fields<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Fields<'_> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let mut field = [0; N];
        field.copy_from_slice(&self.bytes[self.at..self.at + N]);
        self.at += N;
        field
    }

    fn u8(&mut self) -> u8 {
        self.take::<1>()[0]
    }

    fn bool(&mut self) -> Result<bool, PacketError> {
        match self.u8() {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(PacketError::InvalidBool(other)),
        }
    }

    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.take())
    }

    fn i16(&mut self) -> i16 {
        i16::from_le_bytes(self.take())
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take())
    }

    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.take())
    }

    fn rest(&mut self) -> &[u8] {
        let rest = &self.bytes[self.at..];
        self.at = self.bytes.len();
        rest
    }
}

fn check_payload_len(kind: EventKind, expected: usize, actual: usize) -> Result<(), PacketError> {
    if expected != actual {
        return Err(PacketError::PayloadLength {
            kind,
            expected,
            actual,
        });
    }
    Ok(())
}

fn decode_event(kind: EventKind, flags: u8, payload: &[u8]) -> Result<Event, PacketError> {
    let mut fields = Fields {
        bytes: payload,
        at: 0,
    };
    if flags & DEVICE_REMOVED != 0 {
        let removed = match kind {
            EventKind::Key => RemovableDevice::Keyboard,
            EventKind::Gamepad => RemovableDevice::Gamepad,
            _ => return Err(PacketError::UnknownFlags(flags)),
        };
        check_payload_len(kind, 1, payload.len())?;
        return Ok(Event::DeviceRemoved {
            kind: removed,
            device: fields.u8(),
        });
    }
    if let Some(expected) = kind.fixed_payload_len() {
        check_payload_len(kind, expected, payload.len())?;
    }
    let event = match kind {
        EventKind::Key => Event::Key {
            device: fields.u8(),
            usage: fields.u16(),
            down: fields.bool()?,
        },
        EventKind::Button => Event::Button {
            button: fields.u8(),
            down: fields.bool()?,
        },
        EventKind::Move => Event::Move {
            dx: fields.i32(),
            dy: fields.i32(),
        },
        EventKind::Position => Event::Position {
            x: fields.i32(),
            y: fields.i32(),
            display: fields.u8(),
        },
        EventKind::Wheel => Event::Wheel {
            dx: fields.i32(),
            dy: fields.i32(),
        },
        EventKind::Gamepad => Event::Gamepad {
            device: fields.u8(),
            buttons: fields.u32(),
            axes: GamepadAxes {
                left_x: fields.i16(),
                left_y: fields.i16(),
                right_x: fields.i16(),
                right_y: fields.i16(),
                left_trigger: fields.i16(),
                right_trigger: fields.i16(),
            },
        },
        EventKind::Rumble => Event::Rumble {
            device: fields.u8(),
            low: fields.u16(),
            high: fields.u16(),
            duration_ms: fields.u16(),
        },
        EventKind::Cursor => {
            if payload.len() < CURSOR_HEADER_LEN {
                return Err(PacketError::PayloadLength {
                    kind,
                    expected: CURSOR_HEADER_LEN,
                    actual: payload.len(),
                });
            }
            let width = fields.u16();
            let height = fields.u16();
            let hotspot_x = fields.i16();
            let hotspot_y = fields.i16();
            Event::Cursor(CursorImage::new(
                width,
                height,
                hotspot_x,
                hotspot_y,
                fields.rest().to_vec(),
            )?)
        }
        EventKind::Clipboard => Event::Clipboard(
            String::from_utf8(fields.rest().to_vec()).map_err(|_| PacketError::InvalidUtf8)?,
        ),
    };
    Ok(event)
}

#[derive(Debug)]
pub struct PacketReader {
    sender_flags: u8,
    newest_sequence: [Option<u32>; EVENT_KINDS.len()],
}

impl PacketReader {
    pub fn new(receiver: Side) -> Self {
        let sender = match receiver {
            Side::Host => Side::Client,
            Side::Client => Side::Host,
        };
        Self {
            sender_flags: flags_for(sender),
            newest_sequence: [None; EVENT_KINDS.len()],
        }
    }

    pub fn read(&mut self, packet: &[u8]) -> Result<Option<Event>, PacketError> {
        let Some((header, payload)) = packet.split_first_chunk::<HEADER_LEN>() else {
            return Err(PacketError::Truncated { len: packet.len() });
        };
        let kind = EventKind::from_wire(header[0])?;
        let flags = header[1];
        if flags & !KNOWN_FLAGS != 0 {
            return Err(PacketError::UnknownFlags(flags));
        }
        if flags & FROM_HOST != self.sender_flags {
            return Err(PacketError::WrongDirection);
        }
        let sequence = u32::from_le_bytes([header[2], header[3], header[4], header[5]]);
        let declared = usize::from(u16::from_le_bytes([header[6], header[7]]));
        if declared != payload.len() {
            return Err(PacketError::LengthMismatch {
                declared,
                actual: payload.len(),
            });
        }
        let event = decode_event(kind, flags, payload)?;

        let newest = &mut self.newest_sequence[kind.index()];
        if newest.is_some_and(|newest| sequence <= newest) {
            return Ok(None);
        }
        *newest = Some(sequence);
        Ok(Some(event))
    }
}

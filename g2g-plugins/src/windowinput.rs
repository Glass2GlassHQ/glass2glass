// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerButton {
    Left,
    Right,
    Middle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowInput {
    Focus(bool),
    Key { scancode: u32, down: bool },
    Pointer { x: u32, y: u32 },
    Button { button: PointerButton, down: bool },
    Wheel { dx: i32, dy: i32 },
}

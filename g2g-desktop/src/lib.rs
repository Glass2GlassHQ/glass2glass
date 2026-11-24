// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "session"))]
mod audio;
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "session"))]
pub mod client;
pub mod hid;
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "session"))]
pub mod host;
pub mod injection;
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "session"))]
mod lineargraph;
pub mod packet;
pub mod router;
#[cfg(all(target_os = "windows", feature = "session"))]
pub mod sendinput;
#[cfg(target_os = "linux")]
pub mod uinput;
#[cfg(all(target_os = "linux", feature = "session"))]
pub mod waylanddisplayinput;
#[cfg(all(target_os = "linux", feature = "session"))]
pub mod xtest;

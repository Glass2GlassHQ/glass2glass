// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::os::fd::{AsFd, BorrowedFd};
use std::path::Path;

use drm::control::Device as ControlDevice;
use drm::Device;

use g2g_core::{G2gError, HardwareError};

pub(crate) const DEFAULT_CARD: &str = "/dev/dri/card0";

#[derive(Debug)]
pub(crate) struct Card(std::fs::File);

impl AsFd for Card {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}
impl Device for Card {}
impl ControlDevice for Card {}

impl Card {
    pub(crate) fn open<P: AsRef<Path>>(path: P) -> Result<Self, G2gError> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        Ok(Card(file))
    }
}

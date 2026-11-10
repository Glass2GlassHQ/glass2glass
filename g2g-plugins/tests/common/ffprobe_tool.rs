// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::process::Command;

pub(crate) fn have_ffprobe() -> bool {
    Command::new("ffprobe").arg("-version").output().is_ok()
}

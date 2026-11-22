// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::process::Command;

pub(crate) fn have_gstreamer_elements(elements: &[&str]) -> bool {
    elements.iter().all(|element| {
        Command::new("gst-inspect-1.0")
            .args(["--exists", element])
            .status()
            .is_ok_and(|status| status.success())
    })
}

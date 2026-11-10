// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![no_main]
// Subtitle text parsing (SRT / WebVTT / SSA-ASS / TTML, auto-detected): byte vs
// char-boundary slicing over attacker-controlled decoded text.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let _ = g2g_plugins::subparse::parse_auto(&text);
});

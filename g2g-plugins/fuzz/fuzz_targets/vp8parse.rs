// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![no_main]
// VP8 keyframe header parsing (start-code + dimension words).
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    g2g_plugins::vp8parse::fuzz_parse(data);
});

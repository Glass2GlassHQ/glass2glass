// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![no_main]
// H.264 SPS geometry parsing: NAL scan + Exp-Golomb bit reader over attacker-controlled access-unit bytes.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    g2g_plugins::h264parse::fuzz_parse(data);
});

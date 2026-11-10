// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![no_main]
// VP9 keyframe uncompressed-header parsing (profile / sync-code / dimension bit reads).
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    g2g_plugins::vp9parse::fuzz_parse(data);
});

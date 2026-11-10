// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![no_main]
// IVF demux over untrusted input: DKIF file header (codec FourCC + geometry) and
// the 12-byte per-frame headers, driven through the element's reassembly path.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    g2g_plugins::ivfdemux::fuzz_parse(data);
});

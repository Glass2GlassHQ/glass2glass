// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

// The keyword graph's front end on the emulated Cortex-M4: MicroFrontend ->
// int8 quantize, with the features digest compared against the host's pinned run.

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hio};
use noalloc_pipeline::keyword::{run_keyword, EXPECTED_FEATURES_DIGEST};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[entry]
fn main() -> ! {
    let features_ok = run_keyword() == Some(EXPECTED_FEATURES_DIGEST);
    if let Ok(mut out) = hio::hstdout() {
        let _ = out.write_all(if features_ok {
            b"g2g-keyword: features bit-exact with the host OK\n".as_slice()
        } else {
            b"g2g-keyword: FAIL, features differ from the host\n".as_slice()
        });
    }
    debug::exit(if features_ok {
        debug::EXIT_SUCCESS
    } else {
        debug::EXIT_FAILURE
    });
    loop {}
}

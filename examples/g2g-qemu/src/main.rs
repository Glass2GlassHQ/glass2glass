//! Emulated Cortex-M execution proof (M628): run the heap-free proof pipeline
//! (the shared `noalloc-pipeline` crate, the same definition the `g2g-noalloc`
//! no-heap / panic-free symbol proofs wrap) on QEMU's MPS2-AN386 Cortex-M4,
//! and report the result over semihosting. The pipeline blits 64 frames
//! through the real SPI display element onto a stub bus and the wire checksum
//! must match the pipeline's expected constant.
//!
//! No formatting machinery: the banner is a static string and the verdict is
//! the semihosting exit code, so this binary adds as little on top of the
//! proven pipeline as possible. A panic would land in the `loop {}` handler
//! below and trip the harness timeout.

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hio};
use noalloc_pipeline::audio::AUDIO_EXPECTED_CHECKSUM;
use noalloc_pipeline::EXPECTED_CHECKSUM;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[entry]
fn main() -> ! {
    let video_ok = noalloc_pipeline::run() == EXPECTED_CHECKSUM;
    // The flagship audio graph (M644): its checksum constant was pinned on
    // the host, so matching it here is the cross-ISA bit-exactness proof.
    let audio_ok = noalloc_pipeline::audio::run_audio() == AUDIO_EXPECTED_CHECKSUM;
    if let Ok(mut out) = hio::hstdout() {
        let _ = out.write_all(match (video_ok, audio_ok) {
            (true, true) => {
                b"g2g-qemu: video + flagship audio ran on emulated Cortex-M4, checksums OK\n"
                    .as_slice()
            }
            (false, _) => b"g2g-qemu: FAIL, wrong video checksum\n".as_slice(),
            (_, false) => b"g2g-qemu: FAIL, wrong audio checksum\n".as_slice(),
        });
    }
    debug::exit(if video_ok && audio_ok { debug::EXIT_SUCCESS } else { debug::EXIT_FAILURE });
    loop {}
}

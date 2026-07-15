//! Embassy executor proof (M632): the heap-free camera -> transform ->
//! SPI-display pipeline (`noalloc-pipeline::run_async`, the same future the
//! g2g-noalloc symbol proofs wrap) driven by a real Embassy executor on
//! QEMU's MPS2-AN386 Cortex-M4. This is the production MCU shape: an Embassy
//! task awaits the static-element pipeline directly, no bespoke poll loop.
//! The verdict is the semihosting exit code plus a static banner.

#![no_std]
#![no_main]

use cortex_m_semihosting::{debug, hio};
use embassy_executor::Spawner;
use noalloc_pipeline::audio::{run_audio_with, SumSender, AUDIO_EXPECTED_CHECKSUM};
use noalloc_pipeline::EXPECTED_CHECKSUM;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let video_ok = noalloc_pipeline::run_async().await == EXPECTED_CHECKSUM;
    // The flagship audio graph (M644), awaited directly like any Embassy
    // future; the host-pinned checksum makes the run verifiable.
    let audio_ok = run_audio_with(SumSender::default()).await.checksum() == AUDIO_EXPECTED_CHECKSUM;
    if let Ok(mut out) = hio::hstdout() {
        let _ = out.write_all(match (video_ok, audio_ok) {
            (true, true) => {
                b"g2g-embassy: video + flagship audio ran under Embassy on Cortex-M4, checksums OK\n"
                    .as_slice()
            }
            (false, _) => b"g2g-embassy: FAIL, wrong video checksum\n".as_slice(),
            (_, false) => b"g2g-embassy: FAIL, wrong audio checksum\n".as_slice(),
        });
    }
    debug::exit(if video_ok && audio_ok { debug::EXIT_SUCCESS } else { debug::EXIT_FAILURE });
}

// The keyword-spotting graph on the emulated Cortex-M4: log-mel -> int8
// quantize -> the MicroFlow-compiled micro_speech model, with the features
// digest and the scores compared bit for bit against the host's pinned run.

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hio};
use noalloc_pipeline::keyword::{run_keyword, EXPECTED_RUN};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[entry]
fn main() -> ! {
    let run = run_keyword();
    let features_ok = run.map(|run| run.features_digest) == Some(EXPECTED_RUN.features_digest);
    let scores_ok =
        run.map(|run| run.scores.map(f32::to_bits)) == Some(EXPECTED_RUN.scores.map(f32::to_bits));
    if let Ok(mut out) = hio::hstdout() {
        let _ = out.write_all(match (features_ok, scores_ok) {
            (true, true) => {
                b"g2g-keyword: features and scores bit-exact with the host OK\n".as_slice()
            }
            (false, _) => b"g2g-keyword: FAIL, features differ from the host\n".as_slice(),
            (_, false) => b"g2g-keyword: FAIL, scores differ from the host\n".as_slice(),
        });
    }
    debug::exit(if features_ok && scores_ok {
        debug::EXIT_SUCCESS
    } else {
        debug::EXIT_FAILURE
    });
    loop {}
}

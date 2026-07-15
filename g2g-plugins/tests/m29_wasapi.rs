//! M29: `AudioTestSrc -> WasapiSink` plays a test tone on the default render
//! endpoint through the real runner. The audible-output mirror of the M25
//! `WavSink` test.
//!
//! Run with:
//!
//! ```powershell
//! cargo test -p g2g-plugins --features wasapi-sink --test m29_wasapi
//! ```
//!
//! The end-to-end test opens a real audio device, so on a headless host (no
//! render endpoint) it skips rather than failing: `configure_pipeline` returns
//! a `Hardware` error, which surfaces as a pipeline error here.

#![cfg(all(target_os = "windows", feature = "wasapi-sink"))]

use g2g_core::element::AsyncElement;
use g2g_core::runtime::run_simple_pipeline;
use g2g_core::{AudioFormat, Caps, G2gError, PipelineClock};
use g2g_plugins::audiotestsrc::AudioTestSrc;
use g2g_plugins::wasapisink::WasapiSink;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn tone_renders_to_the_default_endpoint() {
    // 20 buffers x 10 ms = 200 ms of 1 kHz stereo sine at 48 kHz.
    const BUFFERS: u64 = 20;
    const FRAMES_PER_BUFFER: u64 = 48_000 * 10 / 1000;

    let mut src = AudioTestSrc::new(48_000, 2, 1_000, BUFFERS);
    let mut sink = WasapiSink::new();

    match run_simple_pipeline(&mut src, &mut sink, &NullClock, 4).await {
        Ok(_stats) => {
            assert_eq!(
                sink.frames_rendered(),
                BUFFERS * FRAMES_PER_BUFFER,
                "every produced sample frame must reach the endpoint"
            );
        }
        Err(G2gError::Hardware(_)) => {
            std::eprintln!("skipping: no audio render endpoint on this host");
        }
        Err(e) => panic!("unexpected pipeline error: {e:?}"),
    }
}

#[tokio::test]
async fn rejects_compressed_audio() {
    let mut sink = WasapiSink::new();
    let aac = Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: 48_000,
    };
    let err = sink.configure_pipeline(&aac).expect_err("aac rejected");
    assert_eq!(err, G2gError::CapsMismatch);
}

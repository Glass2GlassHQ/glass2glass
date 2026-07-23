//! M38: `WasapiSrc::with_loopback()` captures the default render endpoint's
//! output (system audio). A background tone is played through `WasapiSink`
//! while the loopback source captures, so the captured buffers carry real
//! playback rather than depending on whatever else is making sound.
//!
//! Run with:
//!
//! ```powershell
//! cargo test -p g2g-plugins --features "wasapi-src wasapi-sink" --test m38_wasapi_loopback
//! ```
//!
//! Needs a render endpoint; a headless host fails the probe and the test skips.

#![cfg(all(target_os = "windows", feature = "wasapi-src", feature = "wasapi-sink"))]

use g2g_core::element::{BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::PipelinePacket;
use g2g_core::runtime::{run_simple_pipeline, SourceLoop};
use g2g_core::{Caps, G2gError, PipelineClock};
use g2g_plugins::audiotestsrc::AudioTestSrc;
use g2g_plugins::wasapisink::WasapiSink;
use g2g_plugins::wasapisrc::WasapiSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Default)]
struct Collect {
    data_frames: usize,
    sample_bytes: usize,
    eos: bool,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.sample_bytes += s.len();
                    }
                    self.data_frames += 1;
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Play ~1 s of tone on the default render endpoint in a background thread, so
/// loopback capture has real audio to pick up.
fn spawn_background_tone() -> std::thread::JoinHandle<()> {
    std::thread::spawn(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(async {
            let mut src = AudioTestSrc::new(48_000, 2, 440, 100); // 100 x 10 ms = 1 s
            let mut sink = WasapiSink::new();
            // Best-effort: if there is no render endpoint the capture probe below
            // skips anyway.
            let _ = run_simple_pipeline(&mut src, &mut sink, &NullClock, 4).await;
        });
    })
}

#[tokio::test]
async fn loopback_captures_system_playback_or_skips() {
    const BUFFERS: u64 = 5;

    let mut src = WasapiSrc::new(BUFFERS).with_loopback();
    assert!(src.is_loopback());

    // Probe first so a host without a render endpoint skips before we spawn the
    // background tone.
    let caps = match src.intercept_caps().await {
        Ok(caps) => caps,
        Err(G2gError::Hardware(_)) => {
            std::eprintln!("skipping: no render endpoint for loopback on this host");
            return;
        }
        Err(e) => panic!("unexpected probe error: {e:?}"),
    };
    assert!(matches!(caps, Caps::Audio { .. }));

    let tone = spawn_background_tone();
    src.configure_pipeline(&caps)
        .expect("configure loopback source");

    let mut out = Collect::default();
    let emitted = src.run(&mut out).await.expect("loopback capture runs");
    let _ = tone.join();

    assert!(out.eos, "source ends with Eos");
    assert_eq!(out.data_frames as u64, emitted, "frames counted");
    assert_eq!(emitted, BUFFERS, "captured the requested loopback buffers");
    assert!(out.sample_bytes > 0, "captured non-empty loopback PCM");
}

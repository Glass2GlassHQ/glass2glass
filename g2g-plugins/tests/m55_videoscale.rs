//! P1.1: `VideoScale` in a real negotiated chain. An RGBA test source at
//! one geometry reaches the sink resampled to the configured target dims,
//! framerate preserved, announced via a `CapsChanged` before the first
//! frame.

use g2g_core::runtime::run_source_transform_sink;
use g2g_core::{Caps, Dim, PipelineClock, Rate, RawVideoFormat};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videoscale::VideoScale;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn upscales_rgba_source_to_target_dims() {
    let mut src = VideoTestSrc::new(32, 16, 30, 5);
    let mut scale = VideoScale::new(64, 32);
    let mut sink = FakeSink::new();

    run_source_transform_sink(&mut src, &mut scale, &mut sink, &NullClock, 4)
        .await
        .expect("RGBA -> scale -> sink negotiates and flows");

    assert_eq!(sink.received(), 5, "every frame scaled and delivered");
    assert!(sink.eos_seen());
    let changes = sink.caps_changes();
    assert!(
        changes.iter().any(|c| matches!(
            &c.caps,
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(64),
                height: Dim::Fixed(32),
                framerate: Rate::Fixed(r),
            } if *r == 30 << 16
        )),
        "sink saw RGBA at the target geometry with framerate preserved, got {changes:?}"
    );
}

#[tokio::test]
async fn downscales_rgba_source_to_target_dims() {
    let mut src = VideoTestSrc::new(64, 48, 30, 3);
    let mut scale = VideoScale::new(32, 24);
    let mut sink = FakeSink::new();

    run_source_transform_sink(&mut src, &mut scale, &mut sink, &NullClock, 4)
        .await
        .expect("downscale chain negotiates and flows");

    assert_eq!(sink.received(), 3);
    let changes = sink.caps_changes();
    assert!(changes.iter().any(|c| matches!(
        &c.caps,
        Caps::RawVideo {
            width: Dim::Fixed(32),
            height: Dim::Fixed(24),
            ..
        }
    )));
}

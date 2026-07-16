//! P1.2: `VideoRate` in a real negotiated chain. A 30 fps RGBA test source
//! reaches the sink resampled to the configured target rate, the new
//! framerate announced via `CapsChanged`. Downsample drops frames, upsample
//! duplicates them.

use g2g_core::runtime::run_source_transform_sink;
use g2g_core::{Caps, PipelineClock, Rate, RawVideoFormat};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videorate::VideoRate;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn downsamples_30_to_10() {
    let mut src = VideoTestSrc::new(32, 16, 30, 9);
    let mut rate = VideoRate::new(10.0);
    let mut sink = FakeSink::new();

    run_source_transform_sink(&mut src, &mut rate, &mut sink, &NullClock, 4)
        .await
        .expect("30 fps -> rate -> 10 fps chain negotiates and flows");

    // 9 inputs at 30 fps against a 10 fps grid: three frames emitted in
    // stream plus the held last frame flushed on EOS.
    assert_eq!(sink.received(), 4, "downsampled ~3:1 plus the EOS-flushed last frame");
    assert!(sink.eos_seen());
    let changes = sink.caps_changes();
    assert!(
        changes.iter().any(|c| matches!(
            &c.caps,
            Caps::RawVideo { format: RawVideoFormat::Rgba8, framerate: Rate::Fixed(r), .. }
            if *r == 10 << 16
        )),
        "output caps carry the 10 fps target, got {changes:?}"
    );
}

#[tokio::test]
async fn upsamples_30_to_60() {
    let mut src = VideoTestSrc::new(32, 16, 30, 5);
    let mut rate = VideoRate::new(60.0);
    let mut sink = FakeSink::new();

    run_source_transform_sink(&mut src, &mut rate, &mut sink, &NullClock, 4)
        .await
        .expect("30 fps -> rate -> 60 fps chain negotiates and flows");

    // duplication produces more output frames than the five inputs.
    assert!(sink.received() > 5, "upsampled, got {}", sink.received());
    let changes = sink.caps_changes();
    assert!(changes.iter().any(|c| matches!(
        &c.caps,
        Caps::RawVideo { framerate: Rate::Fixed(r), .. } if *r == 60 << 16
    )));
}

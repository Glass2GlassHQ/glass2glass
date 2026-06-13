//! Tier-1 A: `VideoFlip` in a real negotiated chain. An 8x4 RGBA test source is
//! rotated 90 degrees clockwise, the swapped 4x8 geometry announced via
//! `CapsChanged`.

use g2g_core::runtime::run_source_transform_sink;
use g2g_core::{Caps, Dim, PipelineClock, Rate, RawVideoFormat};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videoflip::{FlipMethod, VideoFlip};
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn rotates_rgba_source_swapping_geometry() {
    let mut src = VideoTestSrc::new(8, 4, 30, 4);
    let mut flip = VideoFlip::new(FlipMethod::Rotate90Cw);
    let mut sink = FakeSink::new();

    run_source_transform_sink(&mut src, &mut flip, &mut sink, &NullClock, 4)
        .await
        .expect("RGBA -> flip -> sink negotiates and flows");

    assert_eq!(sink.received(), 4, "every frame rotated and delivered");
    assert!(sink.eos_seen());
    let changes = sink.caps_changes();
    assert!(
        changes.iter().any(|c| matches!(
            &c.caps,
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(4),
                height: Dim::Fixed(8),
                framerate: Rate::Fixed(r),
            } if *r == 30 << 16
        )),
        "sink saw RGBA at the swapped 4x8 geometry with framerate preserved, got {changes:?}"
    );
}

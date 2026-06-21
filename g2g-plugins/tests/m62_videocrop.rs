//! Tier-1 A1: `VideoCrop` in a real negotiated chain. An RGBA test source is
//! cropped to a sub-rectangle, the new geometry announced via `CapsChanged`.

use g2g_core::runtime::run_source_transform_sink;
use g2g_core::{Caps, Dim, PipelineClock, Rate, RawVideoFormat};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videocrop::VideoCrop;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn crops_rgba_source_to_rect() {
    let mut src = VideoTestSrc::new(8, 8, 30, 4);
    // Crop 2px off every edge of the 8x8 frame -> 4x4 (gst videocrop insets).
    let mut crop = VideoCrop::new(2, 2, 2, 2);
    let mut sink = FakeSink::new();

    run_source_transform_sink(&mut src, &mut crop, &mut sink, &NullClock, 4)
        .await
        .expect("RGBA -> crop -> sink negotiates and flows");

    assert_eq!(sink.received(), 4, "every frame cropped and delivered");
    assert!(sink.eos_seen());
    let changes = sink.caps_changes();
    assert!(
        changes.iter().any(|c| matches!(
            &c.caps,
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(4),
                height: Dim::Fixed(4),
                framerate: Rate::Fixed(r),
            } if *r == 30 << 16
        )),
        "sink saw RGBA at the crop geometry with framerate preserved, got {changes:?}"
    );
}

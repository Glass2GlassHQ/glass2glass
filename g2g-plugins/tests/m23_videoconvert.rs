//! M23: `VideoConvert` in a real negotiated chain. An RGBA test source
//! reaches an NV12-only sink through the converter.

use g2g_core::runtime::run_source_transform_sink;
use g2g_core::{Caps, Dim, PipelineClock, RawVideoFormat};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn rgba_source_reaches_nv12_only_sink_through_converter() {
    let mut src = VideoTestSrc::new(32, 16, 30, 5);
    let mut conv = VideoConvert::new(RawVideoFormat::Nv12);
    let mut sink = FakeSink::new();

    run_source_transform_sink(&mut src, &mut conv, &mut sink, &NullClock, 4)
        .await
        .expect("RGBA -> convert -> NV12 chain negotiates and flows");

    assert_eq!(sink.received(), 5, "every frame converted and delivered");
    assert!(sink.eos_seen());
    // the converter announces its NV12 output before the first frame.
    let changes = sink.caps_changes();
    assert!(
        changes.iter().any(|c| matches!(
            &c.caps,
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(32),
                height: Dim::Fixed(16),
                ..
            }
        )),
        "sink saw NV12 caps at the source geometry, got {changes:?}"
    );
}

/// A shared-memory source hands the converter an `Arc`-backed `SystemView`
/// (what `videoflip` emits on its zero-copy path); the convert reads it instead
/// of refusing the domain.
#[tokio::test]
async fn shared_memory_source_reaches_nv12_sink_through_converter() {
    let frames = 4;
    let mut src = VideoTestSrc::new(32, 16, 30, frames).with_shared_memory();
    let mut conv = VideoConvert::new(RawVideoFormat::Nv12);
    let mut sink = FakeSink::new();

    run_source_transform_sink(&mut src, &mut conv, &mut sink, &NullClock, 4)
        .await
        .expect("shared RGBA -> convert -> NV12 chain negotiates and flows");

    assert_eq!(sink.received(), frames, "every shared frame converted");
    assert!(sink.eos_seen());
}

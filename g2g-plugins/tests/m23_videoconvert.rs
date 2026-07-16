//! M23: `VideoConvert` in a real negotiated chain. An RGBA test source
//! reaches an NV12-only sink through the converter; without it the same
//! chain fails the solve.

use g2g_core::runtime::run_source_transform_sink;
use g2g_core::{Caps, Dim, G2gError, PipelineClock, Rate, RawVideoFormat};
use g2g_plugins::capsfilter::CapsFilter;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn nv12_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
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
            Caps::RawVideo { format: RawVideoFormat::Nv12, width: Dim::Fixed(32), height: Dim::Fixed(16), .. }
        )),
        "sink saw NV12 caps at the source geometry, got {changes:?}"
    );
}

#[tokio::test]
async fn nv12_only_filter_rejects_rgba_source_without_converter() {
    // control: the same source into an NV12-only filter fails the
    // whole-chain solve, which is exactly the gap VideoConvert closes.
    let mut src = VideoTestSrc::new(32, 16, 30, 5);
    let mut filter = CapsFilter::new(nv12_any());
    let mut sink = FakeSink::new();

    let err = run_source_transform_sink(&mut src, &mut filter, &mut sink, &NullClock, 4)
        .await
        .expect_err("RGBA into an NV12-only filter must fail negotiation");
    assert_eq!(err, G2gError::CapsMismatch);
}

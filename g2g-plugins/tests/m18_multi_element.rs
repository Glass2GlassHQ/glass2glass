//! M18 item 4 — arbitrary-length linear runner (`run_linear_chain`).
//!
//! The fixed-arity runners cap at three elements; `run_linear_chain` lifts
//! that so chains like `source -> id -> capsfilter -> id -> sink` are
//! expressible (DESIGN-M16-caps-nego.md §13.3, §13.4 item 4). These tests
//! drive real plugin elements end to end: negotiation runs the solver over
//! the whole chain at once, and frames flow across every interior hop.

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{run_linear_chain, run_linear_chain_with_bus};
use g2g_core::{
    Bus, BusMessage, Caps, Dim, G2gError, NegotiationFailure, PipelineClock, Rate, RawVideoFormat,
};

use g2g_plugins::capsfilter::CapsFilter;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::identity::IdentityTransform;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

/// Four elements: `VideoTestSrc -> Identity -> Identity -> FakeSink`. Every
/// emitted frame reaches the sink across both interior hops.
#[tokio::test]
async fn four_element_chain_flows_all_frames() {
    let target = 5u64;
    let mut src = VideoTestSrc::new(64, 64, 30, target);
    let mut id1 = IdentityTransform::new();
    let mut id2 = IdentityTransform::new();
    let mut sink = FakeSink::new();
    let clock = ZeroClock;

    let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut id1, &mut id2];
    let stats = run_linear_chain(&mut src, transforms, &mut sink, &clock, 4)
        .await
        .expect("4-element chain runs");

    assert_eq!(stats.frames_emitted, target);
    assert_eq!(stats.frames_consumed, target);
    assert_eq!(sink.received(), target, "every frame crosses both hops");
    assert!(sink.eos_seen(), "EOS propagates the full length");
}

/// Five elements with a real `CapsFilter` mid-chain. The filter pins RGBA
/// (which the source already produces), so negotiation narrows cleanly
/// through the whole chain and data flows.
#[tokio::test]
async fn five_element_chain_with_capsfilter_negotiates_and_flows() {
    let target = 4u64;
    let mut src = VideoTestSrc::new(32, 32, 30, target);
    let mut id1 = IdentityTransform::new();
    let mut filter = CapsFilter::new(rgba_any());
    let mut id2 = IdentityTransform::new();
    let mut sink = FakeSink::new();
    let clock = ZeroClock;

    let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut id1, &mut filter, &mut id2];
    let stats = run_linear_chain(&mut src, transforms, &mut sink, &clock, 4)
        .await
        .expect("5-element chain with a CapsFilter runs");

    assert_eq!(stats.frames_consumed, target);
    assert_eq!(sink.received(), target);
}

/// Zero transforms degenerates to `source -> sink`, the
/// `run_simple_pipeline` shape.
#[tokio::test]
async fn zero_transforms_is_source_to_sink() {
    let target = 3u64;
    let mut src = VideoTestSrc::new(16, 16, 30, target);
    let mut sink = FakeSink::new();
    let clock = ZeroClock;

    let stats = run_linear_chain(&mut src, vec![], &mut sink, &clock, 4)
        .await
        .expect("0-transform chain runs");

    assert_eq!(stats.frames_consumed, target);
    assert_eq!(sink.received(), target);
}

/// A `CapsFilter` pinning a format the source never produces fails the
/// whole-chain solve loud, proving negotiation runs over every element.
#[tokio::test]
async fn incompatible_capsfilter_fails_negotiation() {
    let mut src = VideoTestSrc::new(64, 64, 30, 4);
    let mut filter = CapsFilter::new(Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    });
    let mut sink = FakeSink::new();
    let clock = ZeroClock;

    let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut filter];
    let result = run_linear_chain(&mut src, transforms, &mut sink, &clock, 4).await;

    assert_eq!(
        result.err(),
        Some(G2gError::CapsMismatch),
        "an RGBA source cannot negotiate through an NV12-only filter"
    );
}

/// `run_linear_chain_with_bus` routes the whole-chain solve failure to the
/// bus (M18 item 7): the same incompatible NV12 filter posts a structured
/// `EmptyLink` while the run still errors `CapsMismatch`.
#[tokio::test]
async fn incompatible_chain_posts_negotiation_failure_to_bus() {
    let mut src = VideoTestSrc::new(64, 64, 30, 4);
    let mut filter = CapsFilter::new(Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    });
    let mut sink = FakeSink::new();
    let clock = ZeroClock;
    let (bus, handle) = Bus::new(4);

    let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut filter];
    let result =
        run_linear_chain_with_bus(&mut src, transforms, &mut sink, &clock, 4, &handle).await;

    assert_eq!(result.err(), Some(G2gError::CapsMismatch));
    match bus.try_recv() {
        Some(BusMessage::NegotiationFailed(NegotiationFailure::EmptyLink { .. })) => {}
        other => panic!("expected NegotiationFailed(EmptyLink), got {other:?}"),
    }
}

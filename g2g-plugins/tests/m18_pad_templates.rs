//! M18 item 6 — pad templates as declarative metadata.
//!
//! Demonstrates querying real element *types* without constructing them:
//! listing their pads and running the negotiation solver against two
//! types' static templates to answer "can A's output feed B's input?"
//! before either element exists.

use g2g_core::runtime::solver::NegotiationFailure;
use g2g_core::{
    pad_link, types_can_link, Caps, Dim, PadCaps, PadDirection, PadTemplates, Rate, VideoCodec, RawVideoFormat,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::videotestsrc::VideoTestSrc;

#[test]
fn introspects_pad_templates_without_constructing() {
    // A source type exposes exactly one source pad, here RGBA.
    let src_pads = VideoTestSrc::pad_templates();
    assert_eq!(src_pads.len(), 1);
    assert_eq!(src_pads[0].direction, PadDirection::Source);
    match &src_pads[0].caps {
        PadCaps::Fixed(set) => assert_eq!(
            set.alternatives()[0],
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            }
        ),
        other => panic!("expected fixed RGBA caps, got {other:?}"),
    }

    // A transform exposes both a sink and a source pad.
    let parse_pads = H264Parse::pad_templates();
    assert!(parse_pads.iter().any(|p| p.direction == PadDirection::Sink));
    assert!(parse_pads.iter().any(|p| p.direction == PadDirection::Source));

    // A wildcard sink reports an `Any` pad.
    let sink_pads = FakeSink::pad_templates();
    assert!(matches!(sink_pads[0].caps, PadCaps::Any));
}

#[test]
fn pre_instantiation_query_accepts_compatible_types() {
    // RGBA source -> wildcard sink: compatible.
    assert!(types_can_link::<VideoTestSrc, FakeSink>());
    // H.264 source pad -> wildcard sink: compatible (the link is
    // `Unfixable` since geometry is open, which counts as linkable).
    assert!(types_can_link::<H264Parse, FakeSink>());
}

#[test]
fn pre_instantiation_query_rejects_incompatible_types() {
    // RGBA source cannot feed an H.264-only sink pad.
    assert!(!types_can_link::<VideoTestSrc, H264Parse>());

    // And the detailed query reports exactly why: the caps don't intersect.
    let producer = VideoTestSrc::pad_template(PadDirection::Source).unwrap();
    let consumer = H264Parse::pad_template(PadDirection::Sink).unwrap();
    assert!(
        matches!(pad_link(&producer, &consumer), Err(NegotiationFailure::EmptyLink { .. })),
        "RGBA vs H.264 must be a structured EmptyLink, not a vague failure"
    );
}

#[test]
fn a_source_has_no_sink_pad_so_nothing_feeds_into_it() {
    // Direction matters: a source type can't be the consumer end.
    assert!(VideoTestSrc::pad_template(PadDirection::Sink).is_none());
    assert!(!types_can_link::<H264Parse, VideoTestSrc>());
}

//! M18 item 6 — pad templates as declarative metadata.
//!
//! Demonstrates querying real element *types* without constructing them:
//! listing their pads and running the negotiation solver against two
//! types' static templates to answer "can A's output feed B's input?"
//! before either element exists.

use g2g_core::runtime::solver::NegotiationFailure;
use g2g_core::{
    pad_link, types_can_link, Caps, Dim, PadCaps, PadDirection, PadTemplates, Rate, RawVideoFormat,
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
    assert!(parse_pads
        .iter()
        .any(|p| p.direction == PadDirection::Source));

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
        matches!(
            pad_link(&producer, &consumer),
            Err(NegotiationFailure::EmptyLink { .. })
        ),
        "RGBA vs H.264 must be a structured EmptyLink, not a vague failure"
    );
}

#[test]
fn a_source_has_no_sink_pad_so_nothing_feeds_into_it() {
    // Direction matters: a source type can't be the consumer end.
    assert!(VideoTestSrc::pad_template(PadDirection::Sink).is_none());
    assert!(!types_can_link::<H264Parse, VideoTestSrc>());
}

/// The full Windows decode -> display chain is introspectable before any
/// element is built: H264Parse -> MfDecode -> D3D11Sink all link by type.
#[cfg(all(target_os = "windows", feature = "mf-decode", feature = "d3d11-sink"))]
#[test]
fn windows_decode_to_display_chain_links_by_type() {
    use g2g_plugins::d3d11sink::D3D11Sink;
    use g2g_plugins::mfdecode::MfDecode;

    // H.264 parser -> decoder (both H.264 at the boundary).
    assert!(types_can_link::<H264Parse, MfDecode>());
    // Decoder NV12 output -> NV12 present sink.
    assert!(types_can_link::<MfDecode, D3D11Sink>());
    // An RGBA test source cannot feed the H.264 decoder.
    assert!(!types_can_link::<VideoTestSrc, MfDecode>());
    // The sink is terminal: no source pad to feed anything downstream.
    assert!(D3D11Sink::pad_template(PadDirection::Source).is_none());
}

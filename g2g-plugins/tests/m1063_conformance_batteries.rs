//! M1063: the always-on conformance batteries cover the sans-IO container,
//! packetizer and parser cores, not just the ST 2110 / RTP ones.
//!
//! Each battery withholds its evidence when a check fails rather than asserting,
//! so a silent regression would show up only as a thinner `--maturity` row. This
//! test is what makes that loud: every core listed here must reach the report
//! with `Instantiate` and `RoundTrip` evidence, and the muxers whose battery
//! models packet loss must carry `LossResilience` too.
#![cfg(feature = "std")]

use g2g_core::conformance::{ConformanceDimension as D, MaturityLevel, MaturityRecord};
use g2g_plugins::conformance::report;

/// Every core the in-process batteries cover, by the element name its record
/// carries (the same name the persisted `Oracle` evidence uses, so the two merge
/// into one row).
const COVERED: [&str; 23] = [
    "st2110video",
    "st2110audio",
    "rtph264",
    "rtpjitter",
    "mpegtsmux",
    "tsdemux",
    "mp4mux",
    "fmp4demux",
    "matroskamux",
    "matroskademux",
    "flvmux",
    "flvdemux",
    "oggmux",
    "oggdemux",
    "st2110anc",
    "st2110jxs",
    "h264parse",
    "h265parse",
    "av1parse",
    "aacparse",
    "opusparse",
    "subparse",
    "cea",
];

/// The cores whose battery drops a packet or a page and checks the damage stays
/// contained.
const LOSS_MODELLED: [&str; 6] = [
    "st2110video",
    "rtph264",
    "mpegtsmux",
    "tsdemux",
    "oggmux",
    "oggdemux",
];

/// `rtpjitter` proves itself through the loss / reorder path rather than a
/// round trip, so it is the one core exempt from the `RoundTrip` requirement.
const ROUND_TRIP_EXEMPT: [&str; 1] = ["rtpjitter"];

fn record(name: &str) -> MaturityRecord {
    let report = report();
    report
        .records
        .into_iter()
        .find(|r| r.element == name)
        .unwrap_or_else(|| panic!("{name} has no conformance record"))
}

#[test]
fn every_covered_core_reports_behavioral_evidence() {
    for name in COVERED {
        let rec = record(name);
        assert!(rec.has(D::Instantiate), "{name} withheld Instantiate");
        if !ROUND_TRIP_EXEMPT.contains(&name) {
            assert!(rec.has(D::RoundTrip), "{name} withheld RoundTrip");
        }
        assert_eq!(
            rec.level(),
            MaturityLevel::UnitTested,
            "{name} derives unit-tested from its own checks"
        );
        assert!(
            !rec.has(D::Oracle),
            "{name} claims no interop from an in-process battery"
        );
    }
}

#[test]
fn the_loss_batteries_report_loss_resilience() {
    for name in LOSS_MODELLED {
        assert!(
            record(name).has(D::LossResilience),
            "{name} withheld LossResilience"
        );
    }
}

#[test]
fn the_subtitle_battery_covers_every_text_format() {
    let rec = record("subparse");
    let formats: Vec<&str> = rec
        .evidence
        .iter()
        .filter(|e| e.dimension == D::RoundTrip)
        .filter_map(|e| e.codec.as_deref())
        .collect();
    assert_eq!(
        formats,
        ["srt", "webvtt", "ssa", "ttml"],
        "one cue parsed per format"
    );
}

#[test]
fn the_caption_battery_covers_both_standards_and_their_carriage() {
    let rec = record("cea");
    let codecs: Vec<&str> = rec
        .evidence
        .iter()
        .filter(|e| e.dimension == D::RoundTrip)
        .filter_map(|e| e.codec.as_deref())
        .collect();
    assert_eq!(codecs, ["cea608", "cea708", "cea608+cea708"]);
}

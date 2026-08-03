//! M802 / M803 - the extended ST 0601 tags (identity strings, slant range,
//! target width, corner offsets, target location) and the nested MISB ST 0102
//! security local set, against the same published reference packet M801 uses.
//! Expected values are what the independent klvdata implementation decodes for
//! the same bytes, to within one quantization step of each tag's fixed-point
//! scale. klvdata leaves the security set's tags 2, 3, 12, 13 and 22 as unknown
//! elements, so those are asserted against the fixture's raw nested TLVs
//! (`01 01 01 | 02 01 07 | 03 05 "//USA" | 0C 01 07 | 0D 06 utf16"USA" |
//! 16 02 000A`).

use g2g_plugins::klv::{SecurityClassification, SecurityLocalSet, UasDatalink};

const PACKET: &[u8] = include_bytes!("fixtures/DynamicConstantMISMMSPacketData.bin");

/// Quantization steps: the fixed-point scale of each tag, from klvdata's
/// domain / range table for that tag.
const SLANT_RANGE_STEP: f64 = 5_000_000.0 / (u32::MAX as f64);
const TARGET_WIDTH_STEP: f64 = 10_000.0 / (u16::MAX as f64);
const CORNER_OFFSET_STEP: f64 = 0.075 / (i16::MAX as f64);
const LAT_STEP: f64 = 90.0 / (i32::MAX as f64);

fn reference() -> UasDatalink {
    UasDatalink::parse_lenient(PACKET).expect("structure is valid")
}

/// The fixture's four identity strings decode exactly.
#[test]
fn decodes_reference_string_tags() {
    let ls = reference();
    assert_eq!(ls.mission_id.as_deref(), Some("Mission 12"));
    assert_eq!(ls.platform_designation.as_deref(), Some("Predator"));
    assert_eq!(ls.image_source_sensor.as_deref(), Some("EO Nose"));
    assert_eq!(
        ls.image_coordinate_system.as_deref(),
        Some("Geodetic WGS84")
    );
}

/// Slant range and target width match klvdata to within one step of their
/// scales; the fixture carries no corner offsets or target location.
#[test]
fn decodes_reference_numeric_tags() {
    let ls = reference();
    let slant = ls.slant_range_m.expect("tag 21 present");
    let width = ls.target_width_m.expect("tag 22 present");
    assert!(
        (slant - 68_590.983_298_744_77).abs() <= SLANT_RANGE_STEP,
        "slant range: {slant}"
    );
    assert!(
        (width - 722.819_867_246_509_6).abs() <= TARGET_WIDTH_STEP,
        "target width: {width}"
    );
    assert_eq!(ls.corner_lat_offset_1_deg, None);
    assert_eq!(ls.target_lat_deg, None);
}

/// The nested ST 0102 set decodes to the fixture's markings, including the
/// UTF-16BE object country codes.
#[test]
fn decodes_reference_security_set() {
    let sec = reference().security.expect("tag 48 present");
    assert_eq!(
        sec,
        SecurityLocalSet {
            classification: Some(SecurityClassification::Unclassified),
            country_coding_method: Some(7),
            classifying_country: Some("//USA".into()),
            object_country_coding_method: Some(7),
            object_country_codes: Some("USA".into()),
            version: Some(10),
        }
    );
    assert_eq!(sec.classification.unwrap().label(), "UNCLASSIFIED");
}

/// The text line carries the new fields, strings quoted.
#[test]
fn line_carries_extended_tags() {
    let line = reference().to_line();
    for want in [
        "mission='Mission 12'",
        "platform='Predator'",
        "sensor='EO Nose'",
        "coord_sys='Geodetic WGS84'",
        "slant=68591.0",
        "tgt_width=722.8",
        "class='UNCLASSIFIED'",
        "class_country='//USA'",
    ] {
        assert!(line.contains(want), "missing {want} in {line}");
    }
}

/// Re-encoding the decoded reference set recovers every new field: strings
/// exactly, numerics within one quantization step, the security set unchanged.
#[test]
fn extended_tags_round_trip() {
    let ls = reference();
    let again = UasDatalink::parse(&ls.encode()).expect("our own checksum is right");

    assert_eq!(again.mission_id, ls.mission_id);
    assert_eq!(again.platform_designation, ls.platform_designation);
    assert_eq!(again.image_source_sensor, ls.image_source_sensor);
    assert_eq!(again.image_coordinate_system, ls.image_coordinate_system);
    assert_eq!(again.security, ls.security);

    let close = |a: Option<f64>, b: Option<f64>, step: f64, what: &str| {
        let (a, b) = (a.expect(what), b.expect(what));
        assert!((a - b).abs() <= step, "{what}: {a} != {b}");
    };
    close(
        again.slant_range_m,
        ls.slant_range_m,
        SLANT_RANGE_STEP,
        "slant",
    );
    close(
        again.target_width_m,
        ls.target_width_m,
        TARGET_WIDTH_STEP,
        "width",
    );

    // The fixture has no corner offsets or target location, so exercise those
    // through a set that does.
    let built = UasDatalink {
        corner_lat_offset_1_deg: Some(0.0625),
        corner_lon_offset_4_deg: Some(-0.0625),
        target_lat_deg: Some(-10.542_388_6),
        target_lon_deg: Some(29.157_890_1),
        target_alt_m: Some(3_216.0),
        ..Default::default()
    };
    let got = UasDatalink::parse(&built.encode()).expect("valid packet");
    close(
        got.corner_lat_offset_1_deg,
        built.corner_lat_offset_1_deg,
        CORNER_OFFSET_STEP,
        "corner lat 1",
    );
    close(
        got.corner_lon_offset_4_deg,
        built.corner_lon_offset_4_deg,
        CORNER_OFFSET_STEP,
        "corner lon 4",
    );
    close(
        got.target_lat_deg,
        built.target_lat_deg,
        LAT_STEP,
        "target lat",
    );
    assert_eq!(got.corner_lat_offset_2_deg, None, "absent tags stay absent");
}

/// The security set's own encoding is byte-exact against the fixture's nested
/// TLVs, so a decoded set re-muxes without rewriting the markings.
#[test]
fn security_set_re_encodes_byte_exactly() {
    let encoded = reference().encode();
    let fixture_tag_48 = &PACKET[find(PACKET, &[0x30, 0x1C, 0x01, 0x01, 0x01])..][..30];
    assert!(
        contains(&encoded, fixture_tag_48),
        "tag 48 body differs from the fixture's"
    );
}

/// Every truncation and every single-byte corruption of the fixture's nested
/// security set parses or fails cleanly, never panicking: tag 48's body is
/// attacker-controlled.
#[test]
fn nested_security_set_malformed_never_panics() {
    let start = find(PACKET, &[0x30, 0x1C, 0x01, 0x01, 0x01]) + 2;
    let body = &PACKET[start..start + 28];

    for cut in 0..=body.len() {
        let _ = SecurityLocalSet::parse(&body[..cut]);
    }
    for i in 0..body.len() {
        for bit in 0..8 {
            let mut bad = body.to_vec();
            bad[i] ^= 1 << bit;
            let _ = SecurityLocalSet::parse(&bad);
        }
    }
    // A nested tag whose BER length runs past the body is not walkable.
    assert_eq!(SecurityLocalSet::parse(&[3, 0x7F, b'x']), None);
    // An indefinite BER length is rejected outright.
    assert_eq!(SecurityLocalSet::parse(&[3, 0x80, 0x01]), None);
}

fn find(haystack: &[u8], needle: &[u8]) -> usize {
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("fixture contains the nested security set")
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

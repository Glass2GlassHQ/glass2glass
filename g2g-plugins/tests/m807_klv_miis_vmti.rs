//! M807 / M808 - the ST 1204 MIIS core identifier (ST 0601 tag 94) and the
//! nested ST 0903 VMTI local set (tag 74).
//!
//! The MIIS expectations are the worked example published in ST 1204 (§6.2.1
//! Table 6, the same bytes jmisb's ST 1204 tests use): version 1, usage 0x70
//! (physical sensor, virtual platform), then the two UUIDs. That example is
//! what the M801 fixture carries in its tag 94, so the fixture is checked
//! against the standard rather than against our own encoder.

use g2g_plugins::klv::{MiisCoreId, MiisId, MiisIdType, UasDatalink};
use g2g_plugins::vmti::{TargetLocation, VTarget, VmtiLocalSet};

const PACKET: &[u8] = include_bytes!("fixtures/DynamicConstantMISMMSPacketData.bin");

/// The published sensor UUID, F592F023-7336-4AF8-AA91-62C00F2EB2DA.
const SENSOR_UUID: [u8; 16] = [
    0xF5, 0x92, 0xF0, 0x23, 0x73, 0x36, 0x4A, 0xF8, 0xAA, 0x91, 0x62, 0xC0, 0x0F, 0x2E, 0xB2, 0xDA,
];
/// The published platform UUID, 16B74341-0008-41A0-BE36-5B5AB96A3645.
const PLATFORM_UUID: [u8; 16] = [
    0x16, 0xB7, 0x43, 0x41, 0x00, 0x08, 0x41, 0xA0, 0xBE, 0x36, 0x5B, 0x5A, 0xB9, 0x6A, 0x36, 0x45,
];

fn reference_core_id() -> MiisCoreId {
    MiisCoreId {
        version: 1,
        sensor: Some(MiisId {
            id_type: MiisIdType::Physical,
            uuid: SENSOR_UUID,
        }),
        platform: Some(MiisId {
            id_type: MiisIdType::Virtual,
            uuid: PLATFORM_UUID,
        }),
        window: None,
        minor: None,
    }
}

/// The 34 value bytes ST 1204 publishes for that identifier.
fn reference_value() -> Vec<u8> {
    let mut value = vec![0x01, 0x70];
    value.extend_from_slice(&SENSOR_UUID);
    value.extend_from_slice(&PLATFORM_UUID);
    value
}

/// The fixture's tag 94 decodes to the published identifier and re-encodes to
/// the exact bytes it carries.
#[test]
fn fixture_miis_core_id_round_trips_byte_exact() {
    let value = reference_value();
    // The fixture really does carry these bytes as tag 94 (tag, BER length 34).
    let carried = PACKET
        .windows(2 + value.len())
        .any(|w| w[0] == 94 && w[1] == 34 && w[2..] == value[..]);
    assert!(carried, "fixture carries the reference tag 94 value");

    let ls = UasDatalink::parse_lenient(PACKET).expect("fixture parses leniently");
    let core_id = ls.miis_core_id.as_ref().expect("tag 94 decoded");
    assert_eq!(core_id, &reference_core_id());
    assert_eq!(core_id.encode(), value, "re-encode is byte exact");

    // And it survives a full ST 0601 encode + parse, checksum included.
    let round_tripped = UasDatalink::parse(&ls.encode()).expect("re-encoded set is valid");
    assert_eq!(round_tripped.miis_core_id, Some(reference_core_id()));
}

fn sample_vmti() -> VmtiLocalSet {
    VmtiLocalSet {
        timestamp_us: Some(1_231_798_102_000_000),
        system_name: Some(String::from("g2g detector")),
        version: Some(5),
        total_targets: Some(9),
        reported_targets: Some(3),
        frame_number: Some(77),
        frame_width: Some(1280),
        frame_height: Some(720),
        source_sensor: Some(String::from("EO Nose")),
        hfov_deg: Some(23.5),
        vfov_deg: Some(13.25),
        miis_core_id: Some(reference_core_id()),
        targets: vec![
            VTarget {
                id: 1,
                centroid_pixel: Some(320_641),
                boundary_top_left_pixel: Some(318_081),
                boundary_bottom_right_pixel: Some(324_487),
                priority: Some(1),
                confidence_pct: Some(92),
                centroid_row: Some(251),
                centroid_col: Some(641),
                ..Default::default()
            },
            VTarget {
                id: 4_211,
                centroid_pixel: Some(1),
                confidence_pct: Some(5),
                location: Some(TargetLocation {
                    lat_deg: -10.5423,
                    lon_deg: 29.1578,
                    hae_m: 321.0,
                    accuracy: vec![],
                }),
                ..Default::default()
            },
            VTarget {
                id: 70_000,
                boundary_top_left_pixel: Some(900_000),
                boundary_bottom_right_pixel: Some(921_600),
                ..Default::default()
            },
        ],
    }
}

/// A set with several targets survives the nested encode + parse: ids, pixel
/// geometry and confidences come back unchanged, the geodetic fields within
/// their IMAPB step.
#[test]
fn vmti_set_round_trips() {
    let set = sample_vmti();
    let got = VmtiLocalSet::parse(&set.encode_body()).expect("walkable");
    assert_eq!(got.frame_width, Some(1280));
    assert_eq!(got.frame_height, Some(720));
    assert_eq!(got.total_targets, Some(9));
    assert_eq!(got.reported_targets, Some(3));
    assert_eq!(got.source_sensor.as_deref(), Some("EO Nose"));
    assert_eq!(got.miis_core_id, Some(reference_core_id()));
    assert_eq!(got.targets.len(), 3);

    let ids: Vec<u32> = got.targets.iter().map(|t| t.id).collect();
    assert_eq!(ids, vec![1, 4_211, 70_000]);
    assert_eq!(got.targets[0].centroid_pixel, Some(320_641));
    assert_eq!(got.targets[0].boundary_top_left_pixel, Some(318_081));
    assert_eq!(got.targets[0].boundary_bottom_right_pixel, Some(324_487));
    assert_eq!(got.targets[0].priority, Some(1));
    assert_eq!(got.targets[0].confidence_pct, Some(92));
    assert_eq!(got.targets[0].centroid_row, Some(251));
    assert_eq!(got.targets[0].centroid_col, Some(641));
    assert_eq!(got.targets[2].boundary_bottom_right_pixel, Some(921_600));

    let location = got.targets[1].location.as_ref().expect("location pack");
    assert!((location.lat_deg - (-10.5423)).abs() < 1e-6);
    assert!((location.lon_deg - 29.1578).abs() < 1e-6);
    assert!((location.hae_m - 321.0).abs() < 1.0);
}

/// The VMTI set nested in a full ST 0601 local set survives the round trip, and
/// the outer checksum still validates over the larger body.
#[test]
fn nested_vmti_round_trips_inside_st0601() {
    let ls = UasDatalink {
        timestamp_us: Some(1_231_798_102_000_000),
        platform_designation: Some(String::from("Predator")),
        frame_center_lat_deg: Some(60.18),
        frame_center_lon_deg: Some(24.84),
        version: Some(19),
        vmti: Some(sample_vmti()),
        miis_core_id: Some(reference_core_id()),
        ..Default::default()
    };
    let packet = ls.encode();
    // Strict parse: the checksum must match over the whole packet.
    let got = UasDatalink::parse(&packet).expect("checksum valid over the nested set");
    assert_eq!(got.platform_designation.as_deref(), Some("Predator"));
    assert_eq!(got.miis_core_id, Some(reference_core_id()));
    let vmti = got.vmti.expect("tag 74 decoded");
    assert_eq!(vmti.targets.len(), 3);
    assert_eq!(vmti.frame_width, Some(1280));
    assert_eq!(vmti.encode_body(), sample_vmti().encode_body());

    // A flipped byte anywhere still fails the outer checksum.
    let mut corrupt = packet.clone();
    corrupt[40] ^= 0x01;
    assert_eq!(UasDatalink::parse(&corrupt), None);
}

/// Malformed nested input never panics, and a broken VMTI set fails that tag
/// alone: the enclosing ST 0601 set still decodes.
#[test]
fn malformed_nested_vmti_fails_only_that_tag() {
    // Every truncation of the nested body, and of the whole packet.
    let body = sample_vmti().encode_body();
    for cut in 0..=body.len() {
        let _ = VmtiLocalSet::parse(&body[..cut]);
    }
    let packet = UasDatalink {
        timestamp_us: Some(1),
        vmti: Some(sample_vmti()),
        version: Some(19),
        ..Default::default()
    }
    .encode();
    for cut in 0..=packet.len() {
        let _ = UasDatalink::parse(&packet[..cut]);
        let _ = UasDatalink::parse_lenient(&packet[..cut]);
    }

    // A tag 74 whose nested content is unwalkable: the VMTI field drops, the
    // rest of the ST 0601 set survives.
    let unwalkable = [
        // frame width, then a target series claiming 4 GiB of packs
        vec![8, 2, 0x05, 0x00, 101, 5, 0x84, 0xFF, 0xFF, 0xFF, 0xFF],
        // a BER-OID tag that never terminates
        vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
        // a target pack longer than the series holding it
        vec![101, 3, 0x40, 0x01, 0x02],
        // an indefinite-form BER length
        vec![101, 2, 0x80, 0x01],
    ];
    for nested in unwalkable {
        let got = parse_with_raw_tag_74(&nested);
        assert_eq!(got.timestamp_us, Some(7), "sibling tags still decode");
        assert_eq!(got.version, Some(19));
        assert!(got.vmti.is_none(), "an unwalkable nested set is dropped");
    }

    // Counts a stream can claim but not back up: they decode as written and
    // allocate nothing, because only real bytes produce a target.
    let absurd = parse_with_raw_tag_74(&[5, 3, 0xFF, 0xFF, 0xFF, 6, 3, 0xFF, 0xFF, 0xFF])
        .vmti
        .expect("the set itself is walkable");
    assert_eq!(absurd.total_targets, Some(0xFF_FFFF));
    assert!(absurd.targets.is_empty(), "no targets without target bytes");

    // A value ST 0903 does not define fails its own field only: the target is
    // still reported, without a location.
    let tolerated = parse_with_raw_tag_74(&[101, 6, 0x05, 0x01, 17, 2, 0x00, 0x00])
        .vmti
        .expect("the series is walkable");
    assert_eq!(tolerated.targets.len(), 1);
    assert_eq!(tolerated.targets[0].id, 1);
    assert!(tolerated.targets[0].location.is_none());
}

/// Parse a hand-built ST 0601 set whose tag 74 holds `nested` verbatim, so a
/// test can feed the nested parser bytes our encoder would never produce. The
/// checksum is deliberately absent, hence the lenient parse.
fn parse_with_raw_tag_74(nested: &[u8]) -> UasDatalink {
    let mut body = vec![2, 8, 0, 0, 0, 0, 0, 0, 0, 7, 65, 1, 19, 74];
    body.push(u8::try_from(nested.len()).expect("test values are short"));
    body.extend_from_slice(nested);

    let mut packet = Vec::from(g2g_plugins::klv::UAS_LOCAL_SET_KEY);
    packet.push(u8::try_from(body.len()).expect("test packets are short"));
    packet.extend_from_slice(&body);
    UasDatalink::parse_lenient(&packet).expect("outer set stays walkable")
}

/// The detector bridge: each detection becomes one target with pixel geometry
/// over the declared frame, and a tracked one keeps its tracker id.
#[cfg(feature = "analytics")]
#[test]
fn analytics_meta_becomes_vmti_targets() {
    use g2g_core::{AnalyticsMeta, AnalyticsNode, BBox, ObjectDetection, RelationKind, Tracking};
    use g2g_plugins::vmti::vmti_from_analytics;

    let mut meta = AnalyticsMeta::new();
    for (i, x) in [0.0f32, 0.25, 0.5].iter().enumerate() {
        let det = meta.add_detection(ObjectDetection {
            bbox: BBox {
                x: *x,
                y: 0.25,
                w: 0.125,
                h: 0.5,
            },
            label: i as u32,
            confidence: 0.5,
        });
        if i == 1 {
            let track = meta.push(AnalyticsNode::Tracking(Tracking { object_id: 4_096 }));
            meta.relate(det, track, RelationKind::Tracks);
        }
    }

    let set = vmti_from_analytics(&meta, 640, 480);
    assert_eq!(set.total_targets, Some(3));
    assert_eq!(set.reported_targets, Some(3));
    assert_eq!(set.targets.len(), 3);
    assert_eq!(set.frame_width, Some(640));
    assert_eq!(set.frame_height, Some(480));

    // Untracked targets take their 1-based position, the tracked one its id.
    let ids: Vec<u32> = set.targets.iter().map(|t| t.id).collect();
    assert_eq!(ids, vec![1, 4_096, 3]);
    for target in &set.targets {
        assert_eq!(target.confidence_pct, Some(50));
    }

    // Box 2 covers columns 161..240 and rows 121..360 (1-based) of a 640x480
    // frame, so its corners are those pixel numbers.
    let boxed = &set.targets[1];
    assert_eq!(boxed.boundary_top_left_pixel, Some(120 * 640 + 161));
    assert_eq!(boxed.boundary_bottom_right_pixel, Some(359 * 640 + 240));
    assert_eq!(boxed.centroid_pixel, Some(240 * 640 + 201));

    // The set a detector emits is a valid nested body.
    let parsed = VmtiLocalSet::parse(&set.encode_body()).expect("walkable");
    assert_eq!(parsed.targets.len(), 3);
    assert_eq!(parsed.targets[1].id, 4_096);
}

/// The ST 1204 text representation against jmisb: the published worked
/// example, then jmisb's sensor-only, minor-only, three-UUID, and physical
/// example vectors round-tripped from their binary form.
#[test]
fn miis_text_representation_matches_jmisb() {
    assert_eq!(
        reference_core_id().text_representation(),
        "0170:F592-F023-7336-4AF8-AA91-62C0-0F2E-B2DA/16B7-4341-0008-41A0-BE36-5B5A-B96A-3645:D3"
    );

    for text in [
        "0110:1AB8-231E-17E8-4748-A133-CE93-89A7-A060:25",
        "0102:03DD-9DEE-FB48-477B-8204-B050-6F6B-2A33:25",
        "0154:C7D1-6253-98A2-41C2-BA6E-90F8-FCC7-3914/E047-AB3E-81BE-41ED-9664-09B0-2F44-5FAB/5E71-B0DC-20FE-4920-8216-26D6-4F61-D863:C8",
        "0178:865E-FD9C-EF8A-41C3-8244-B885-AFCC-40BF/ED8A-9AB8-72E2-4165-9979-7E5A-F54A-5B9A:25",
    ] {
        // The binary form is every hex digit before the check value.
        let body = &text[..text.rfind(':').unwrap()];
        let hex: String = body.chars().filter(|c| c.is_ascii_hexdigit()).collect();
        let bytes: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let id = MiisCoreId::parse(&bytes).expect("jmisb vector parses");
        assert_eq!(id.text_representation(), text);
    }
}

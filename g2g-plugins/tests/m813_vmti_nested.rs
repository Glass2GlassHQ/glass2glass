//! M813 - the nested ST 0903 sets a VTarget can carry: VMask (tag 101),
//! VObject (102), VTracker (104), VChip (105).
//!
//! Every byte vector here is from jmisb's ST 0903 test suite (VMaskLSTest,
//! VObjectLSTest, VelocityTest, TrackIdTest, VChipLSTest), so parse and
//! encode are checked against an independent implementation, byte for byte.

use g2g_plugins::vmti::{EnuVector, VChip, VMask, VObject, VTarget, VTracker, VmtiLocalSet};

/// jmisb VMaskLSTest: a three-vertex pixel polygon and a three-run bit mask.
#[test]
fn vmask_matches_jmisb_vectors() {
    let bytes = [
        0x01, 0x09, 0x02, 0x39, 0xAA, 0x02, 0x39, 0xBF, 0x02, 0x3B, 0x0B, // polygon
        0x02, 0x0C, 0x03, 0x01, 0x4A, 0x02, 0x03, 0x01, 0x59, 0x04, 0x03, 0x01, 0x6A,
        0x02, // bit mask
    ];
    let mask = VMask::parse(&bytes).expect("jmisb vector parses");
    assert_eq!(mask.polygon, vec![14_762, 14_783, 15_115]);
    assert_eq!(mask.bitmask, vec![(74, 2), (89, 4), (106, 2)]);
    assert_eq!(mask.encode_body(), bytes);
}

/// jmisb VObjectLSTest `mergedBytes`: ontology URI, class, id 258, and 48.0%
/// confidence (IMAPB 0x3000).
#[test]
fn vobject_matches_jmisb_vectors() {
    let uri = b"https://raw.githubusercontent.com/owlcs/pizza-ontology/master/pizza.owl";
    let mut bytes = vec![0x01, uri.len() as u8];
    bytes.extend_from_slice(uri);
    bytes.extend_from_slice(b"\x02\x08Mushroom");
    bytes.extend_from_slice(&[0x03, 2, 0x01, 0x02, 0x04, 2, 0x30, 0x00]);

    let object = VObject::parse(&bytes).expect("jmisb vector parses");
    assert_eq!(
        object.ontology.as_deref(),
        Some(core::str::from_utf8(uri).unwrap())
    );
    assert_eq!(object.ontology_class.as_deref(), Some("Mushroom"));
    assert_eq!(object.ontology_id, Some(258));
    assert!((object.confidence_pct.unwrap() - 48.0).abs() < 0.001);
    assert_eq!(object.encode_body(), bytes);
}

/// jmisb TrackIdTest / VelocityTest / VTrackerLSTest vectors composed into one
/// tracker set: the UUID, a velocity of (300, 200, 100) m/s with its sigma
/// group, and algorithm id 3.
#[test]
fn vtracker_matches_jmisb_vectors() {
    let track_id = [
        0xF8, 0x1D, 0x4F, 0xAE, 0x7D, 0xEC, 0x11, 0xD0, 0xA7, 0x65, 0x00, 0xA0, 0xC9, 0x1E, 0x6B,
        0xF6,
    ];
    let mut bytes = vec![0x01, 16];
    bytes.extend_from_slice(&track_id);
    bytes.extend_from_slice(&[
        0x0A, 12, 0x4B, 0x00, 0x44, 0xC0, 0x3E, 0x80, 0x25, 0x80, 0x19, 0x00, 0x0C, 0x80,
    ]);
    bytes.extend_from_slice(&[0x0C, 1, 0x03]);

    let tracker = VTracker::parse(&bytes).expect("jmisb vectors parse");
    assert_eq!(tracker.track_id, Some(track_id));
    let velocity = tracker.velocity.as_ref().expect("velocity decoded");
    assert!((velocity.east - 300.0).abs() < 0.1);
    assert!((velocity.north - 200.0).abs() < 0.1);
    assert!((velocity.up - 100.0).abs() < 0.1);
    // The sigma group rides along raw, like TargetLocation::accuracy.
    assert_eq!(velocity.accuracy, [0x25, 0x80, 0x19, 0x00, 0x0C, 0x80]);
    assert_eq!(tracker.algorithm_id, Some(3));
    assert_eq!(tracker.encode_body(), bytes);
}

/// jmisb VChipLSTest: "jpeg" type plus the banner URI, byte for byte.
#[test]
fn vchip_matches_jmisb_vectors() {
    let mut bytes = vec![0x01, 0x04];
    bytes.extend_from_slice(b"jpeg");
    let uri = b"https://www.gwg.nga.mil/misb/images/banner.jpg";
    bytes.push(0x02);
    bytes.push(uri.len() as u8);
    bytes.extend_from_slice(uri);

    let chip = VChip::parse(&bytes).expect("jmisb vector parses");
    assert_eq!(chip.image_type.as_deref(), Some("jpeg"));
    assert_eq!(
        chip.image_uri.as_deref(),
        Some(core::str::from_utf8(uri).unwrap())
    );
    assert_eq!(chip.embedded_image, None);
    assert_eq!(chip.encode_body(), bytes);
}

/// A target carrying all four nested sets survives the full VMTI local set
/// round trip.
#[test]
fn nested_sets_round_trip_inside_vmti() {
    let target = VTarget {
        id: 7,
        confidence_pct: Some(90),
        vmask: Some(VMask {
            polygon: vec![14_762, 14_783, 15_115],
            bitmask: vec![(74, 2), (89, 4), (106, 2)],
        }),
        vobject: Some(VObject {
            ontology: Some("https://example.com/onto.owl".into()),
            ontology_class: Some("Vehicle".into()),
            ontology_id: Some(258),
            confidence_pct: Some(48.0),
        }),
        vtracker: Some(VTracker {
            track_id: Some([0xAB; 16]),
            detection_status: Some(1),
            start_time_us: Some(1_700_000_000_000_000),
            end_time_us: Some(1_700_000_001_000_000),
            algorithm: Some("kcf".into()),
            confidence_pct: Some(75),
            num_track_points: Some(240),
            velocity: Some(EnuVector {
                east: 300.0,
                north: 200.0,
                up: 100.0,
                accuracy: Vec::new(),
            }),
            algorithm_id: Some(3),
            ..Default::default()
        }),
        vchip: Some(VChip {
            image_type: Some("jpeg".into()),
            image_uri: None,
            embedded_image: Some(vec![0xFF, 0xD8, 0xFF, 0xE0]),
        }),
        ..Default::default()
    };
    let set = VmtiLocalSet {
        frame_width: Some(640),
        frame_height: Some(480),
        targets: vec![target],
        ..Default::default()
    };

    let parsed = VmtiLocalSet::parse(&set.encode_body()).expect("round trip parses");
    assert_eq!(parsed, set);
    let t = &parsed.targets[0];
    assert!((t.vtracker.as_ref().unwrap().velocity.as_ref().unwrap().east - 300.0).abs() < 0.1);
    assert_eq!(
        t.vchip.as_ref().unwrap().embedded_image.as_deref(),
        Some(&[0xFF, 0xD8, 0xFF, 0xE0][..])
    );
}

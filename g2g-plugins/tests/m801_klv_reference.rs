//! M801 - the ST 0601 parser against a published reference packet. The fixture
//! is the "Dynamic and Constant MISMMS Packet Data" example (MISB ST 0902, via
//! the klvdata project's data set), a real 228-byte UAS Datalink Local Set with
//! string tags, a nested ST 0102 security set, and an unknown trailing tag.
//! Its declared checksum (0xAA43) does not match the ST 0601 sum (0x3E1E, and
//! klvdata's own `packet_checksum` computes the same), so the strict parse
//! rejects it and the lenient parse decodes it; a checksum-corrected copy
//! passes strictly. Expected values are what the independent klvdata
//! implementation decodes for the same bytes, asserted to within one
//! quantization step of each tag's fixed-point scale.

use g2g_plugins::klv::{split_klv_packets, UasDatalink};

const PACKET: &[u8] = include_bytes!("fixtures/DynamicConstantMISMMSPacketData.bin");

/// The ST 0601 sum of the fixture's bytes (klvdata agrees), unlike the 0xAA43
/// its checksum tag declares.
const ACTUAL_SUM: u16 = 0x3E1E;

fn assert_reference_values(ls: &UasDatalink) {
    // klvdata: 2009-01-12 22:08:22 UTC.
    assert_eq!(ls.timestamp_us, Some(1_231_798_102_000_000));
    assert_eq!(ls.version, Some(6));

    let close = |got: Option<f64>, want: f64, eps: f64, what: &str| {
        let got = got.unwrap_or_else(|| panic!("{what} missing"));
        assert!((got - want).abs() <= eps, "{what}: {got} != {want}");
    };
    close(ls.heading_deg, 159.974_364_843_213_55, 0.006, "heading");
    close(ls.pitch_deg, -0.431_531_723_990_598_7, 0.001, "pitch");
    close(ls.roll_deg, 3.405_865_657_521_289_3, 0.002, "roll");
    close(
        ls.sensor_lat_deg,
        60.176_822_966_978_335,
        1e-6,
        "sensor lat",
    );
    close(
        ls.sensor_lon_deg,
        128.426_759_042_044_52,
        1e-6,
        "sensor lon",
    );
    close(ls.sensor_alt_m, 14_190.719_462_882_427, 0.4, "sensor alt");
    close(ls.hfov_deg, 144.571_297_779_812_3, 0.003, "hfov");
    close(ls.vfov_deg, 152.643_625_543_602_67, 0.003, "vfov");
    close(
        ls.rel_azimuth_deg,
        160.719_211_436_975_57,
        1e-6,
        "rel azimuth",
    );
    close(
        ls.rel_elevation_deg,
        -168.792_324_833_940_85,
        1e-6,
        "rel elevation",
    );
    close(ls.rel_roll_deg, 176.865_437_649_391_94, 1e-6, "rel roll");
    close(
        ls.frame_center_lat_deg,
        -10.542_388_633_146_132,
        1e-6,
        "fc lat",
    );
    close(
        ls.frame_center_lon_deg,
        29.157_890_122_923_02,
        1e-6,
        "fc lon",
    );
    close(
        ls.frame_center_alt_m,
        3_216.037_232_013_427_5,
        0.4,
        "fc elevation",
    );
}

/// The lenient parse decodes the reference packet to klvdata's values; the
/// strict parse rejects it because its declared checksum is wrong.
#[test]
fn parses_published_reference_packet() {
    // The fixture is one whole KLV packet; the splitter agrees.
    assert_eq!(split_klv_packets(PACKET), vec![PACKET]);

    assert_eq!(
        UasDatalink::parse(PACKET),
        None,
        "declared checksum 0xAA43 != actual 0x3E1E, strict parse rejects"
    );
    let ls = UasDatalink::parse_lenient(PACKET).expect("structure is valid");
    assert_reference_values(&ls);
}

/// With the checksum corrected to the real ST 0601 sum, the strict parse
/// accepts the packet and decodes the same values.
#[test]
fn corrected_checksum_passes_strict_parse() {
    let mut fixed = PACKET.to_vec();
    let n = fixed.len();
    fixed[n - 2..].copy_from_slice(&ACTUAL_SUM.to_be_bytes());
    let ls = UasDatalink::parse(&fixed).expect("corrected checksum validates");
    assert_reference_values(&ls);
}

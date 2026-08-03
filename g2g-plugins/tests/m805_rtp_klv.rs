//! M805 - KLV metadata over RTP (RFC 6597, the SMPTE ST 336 payload format).
//! ST 0601 telemetry packets are payloaded into RTP and depayloaded back: the
//! KLVunits come out bit-exact and still parse to the telemetry they encode,
//! their PTS survives the 90 kHz clock to within a tick, a small MTU fragments
//! a unit across packets that share one timestamp and end with the marker, and
//! a lost fragment costs only its own unit.

use g2g_plugins::klv::UasDatalink;
use g2g_plugins::rtpklv::{
    pts_ns_from_rtp, rtp_timestamp_from_pts, RtpKlvDepayloader, RtpKlvPacketizer, RTP_CLOCK_HZ,
};

/// 40 ms apart, the metadata cadence of a 25 fps UAS feed.
const FRAME_NS: u64 = 40_000_000;

fn telemetry(i: u64) -> UasDatalink {
    UasDatalink {
        timestamp_us: Some(1_700_000_000_000_000 + i * 40_000),
        sensor_lat_deg: Some(60.1768 + i as f64 * 0.0001),
        sensor_lon_deg: Some(24.8288),
        sensor_alt_m: Some(145.0),
        heading_deg: Some(87.3),
        hfov_deg: Some(12.5),
        version: Some(19),
        ..Default::default()
    }
}

fn marker(packet: &[u8]) -> bool {
    packet[1] & 0x80 != 0
}
fn timestamp(packet: &[u8]) -> u32 {
    u32::from_be_bytes([packet[4], packet[5], packet[6], packet[7]])
}
fn sequence(packet: &[u8]) -> u16 {
    u16::from_be_bytes([packet[2], packet[3]])
}

/// Payload `count` telemetry units, one per 40 ms, returning the units and the
/// per-unit packet lists.
fn stream(count: u64, max_payload: usize) -> (Vec<Vec<u8>>, Vec<Vec<Vec<u8>>>) {
    let mut pay = RtpKlvPacketizer::new(98, 0x0601_0601).with_max_payload(max_payload);
    let mut units = Vec::new();
    let mut packets = Vec::new();
    for i in 0..count {
        let unit = telemetry(i).encode();
        packets.push(pay.packetize(&unit, rtp_timestamp_from_pts(i * FRAME_NS)));
        units.push(unit);
    }
    (units, packets)
}

#[test]
fn round_trips_telemetry_units_with_their_pts() {
    let (units, packets) = stream(5, 1400);
    let mut depay = RtpKlvDepayloader::new();
    let mut out = Vec::new();
    for pkts in &packets {
        assert_eq!(pkts.len(), 1, "a 0601 local set fits one 1400-byte payload");
        out.extend(pkts.iter().filter_map(|p| depay.depacketize(p)));
    }
    assert_eq!(out.len(), units.len(), "every unit came back");

    let tick_ns = 1_000_000_000 / RTP_CLOCK_HZ;
    for (i, unit) in out.iter().enumerate() {
        assert_eq!(unit.data, units[i], "KLVunit bit-exact through RTP");
        let want = i as u64 * FRAME_NS;
        assert!(
            unit.pts_ns().abs_diff(want) <= tick_ns,
            "unit {i} pts {} != {want} beyond a 90 kHz tick",
            unit.pts_ns()
        );
        assert_eq!(unit.pts_ns(), pts_ns_from_rtp(unit.rtp_timestamp));
        let parsed = UasDatalink::parse(&unit.data).expect("depayloaded unit still parses");
        let want = telemetry(i as u64);
        assert_eq!(parsed.timestamp_us, want.timestamp_us);
        assert_eq!(parsed.version, want.version);
        // Angles are ST 0601 fixed point, so compare within a quantization step.
        assert!((parsed.sensor_lat_deg.unwrap() - want.sensor_lat_deg.unwrap()).abs() < 1e-6);
        assert!((parsed.heading_deg.unwrap() - want.heading_deg.unwrap()).abs() < 0.01);
    }
}

#[test]
fn fragments_a_unit_across_packets_sharing_one_timestamp() {
    // 16-byte payloads split a ~60-byte local set into several packets.
    let (units, packets) = stream(3, 16);
    let mut unit_ts = Vec::new();
    for (i, pkts) in packets.iter().enumerate() {
        assert!(pkts.len() >= 3, "unit {i} fragmented into {}", pkts.len());
        let ts = timestamp(&pkts[0]);
        unit_ts.push(ts);
        for (j, pkt) in pkts.iter().enumerate() {
            assert_eq!(timestamp(pkt), ts, "one timestamp per KLVunit");
            assert_eq!(
                marker(pkt),
                j + 1 == pkts.len(),
                "unit {i} packet {j}: marker only on the last fragment"
            );
            assert!(
                pkt.len() <= 12 + 16,
                "no fragment exceeds the header plus the MTU payload"
            );
        }
    }
    assert!(
        unit_ts.windows(2).all(|w| w[1] > w[0]),
        "each unit advances the 90 kHz clock: {unit_ts:?}"
    );
    // Sequence numbers run continuously across units.
    let flat: Vec<_> = packets.iter().flatten().collect();
    for (i, pkt) in flat.iter().enumerate() {
        assert_eq!(sequence(pkt), i as u16);
    }

    let mut depay = RtpKlvDepayloader::new();
    let out: Vec<_> = flat
        .iter()
        .filter_map(|p| depay.depacketize(p))
        .map(|u| u.data)
        .collect();
    assert_eq!(out, units, "fragments reassemble every unit exactly");
}

#[test]
fn a_lost_fragment_drops_only_its_own_unit() {
    let (units, packets) = stream(4, 16);
    // Drop a middle fragment of unit 1 (not its first, not its marked last).
    let mut wire: Vec<Vec<u8>> = Vec::new();
    for (i, pkts) in packets.iter().enumerate() {
        for (j, pkt) in pkts.iter().enumerate() {
            if i == 1 && j == 1 {
                continue;
            }
            wire.push(pkt.clone());
        }
    }

    let mut depay = RtpKlvDepayloader::new();
    let out: Vec<_> = wire
        .iter()
        .filter_map(|p| depay.depacketize(p))
        .map(|u| u.data)
        .collect();
    assert_eq!(
        out,
        vec![units[0].clone(), units[2].clone(), units[3].clone()],
        "only the damaged unit is discarded, and no half unit is emitted"
    );
}

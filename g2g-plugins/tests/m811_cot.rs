//! M811: the Cursor-on-Target bridge. The builder half asserts the CoT schema
//! facts (required `<event>` attributes, the W3C `xs:dateTime` timestamp
//! profile, the `<point>` sentinels) against a known ST 0601 local set, and that
//! a hostile identity string cannot break the document. The sink half (gated on
//! `udp-egress`, like the other network sinks) runs real KLV packets through
//! `cotsink` and reads the events back off a loopback socket.

use g2g_plugins::cotsink::{cot_event, CotOptions};
use g2g_plugins::klv::{SecurityClassification, SecurityLocalSet, UasDatalink};

/// 2023-11-14T22:13:20.123456Z.
const T: u64 = 1_700_000_000_123_456;

/// A local set with every field the bridge reads.
fn sample() -> UasDatalink {
    UasDatalink {
        timestamp_us: Some(T),
        mission_id: Some("Mission 12".into()),
        heading_deg: Some(87.3),
        platform_designation: Some("Predator".into()),
        sensor_lat_deg: Some(60.176822),
        sensor_lon_deg: Some(24.828835),
        sensor_alt_m: Some(145.2),
        hfov_deg: Some(54.9),
        vfov_deg: Some(31.2),
        rel_azimuth_deg: Some(46.3),
        rel_elevation_deg: Some(-4.9),
        rel_roll_deg: Some(358.2),
        slant_range_m: Some(68_590.9),
        frame_center_lat_deg: Some(60.18),
        frame_center_lon_deg: Some(24.84),
        frame_center_alt_m: Some(12.0),
        security: Some(SecurityLocalSet {
            classification: Some(SecurityClassification::Secret),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// The whole event, byte for byte: the seven required `<event>` attributes, the
/// W3C XML `xs:dateTime` profile (`%Y-%m-%dT%H:%M:%S.%fZ`, 6 fractional digits,
/// literal `Z`) on time / start / stale, `how="m-g"` for a GPS-derived machine
/// position, and `ce` / `le` at the 9999999.0 unknown sentinel.
#[test]
fn known_local_set_builds_the_expected_cot_event() {
    let event = cot_event(&sample(), CotOptions::default()).expect("event");
    assert_eq!(
        event,
        concat!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n",
            "<event version=\"2.0\" uid=\"g2g-uas\" type=\"a-f-A-M-F-Q\"",
            " time=\"2023-11-14T22:13:20.123456Z\"",
            " start=\"2023-11-14T22:13:20.123456Z\"",
            " stale=\"2023-11-14T22:13:30.123456Z\" how=\"m-g\">",
            "<point lat=\"60.176822\" lon=\"24.828835\" hae=\"145.2\"",
            " ce=\"9999999.0\" le=\"9999999.0\"/>",
            "<detail>",
            "<contact callsign=\"Predator\"/>",
            "<track course=\"87.3\"/>",
            "<sensor azimuth=\"133.6\" fov=\"54.9\" vfov=\"31.2\" range=\"68590.9\"",
            " elevation=\"-4.9\" roll=\"-1.8\"/>",
            "<remarks>frame-center=60.180000,24.840000 frame-center-alt=12.0m",
            " mission=Mission 12 classification=SECRET</remarks>",
            "</detail></event>",
        )
    );
}

/// The identity strings are attacker-controlled bitstream data: a designation
/// full of markup is escaped, so it stays one attribute value and the document
/// still has exactly the tags the builder wrote.
#[test]
fn hostile_platform_designation_cannot_break_the_document() {
    let hostile = "\"/><event uid='x'><detail>&pwn;</detail></event><!--";
    let ls = UasDatalink {
        platform_designation: Some(hostile.into()),
        ..sample()
    };
    let event = cot_event(&ls, CotOptions::default()).expect("event");

    assert!(
        event.contains(
            "<contact callsign=\"&quot;/&gt;&lt;event uid=&apos;x&apos;&gt;\
             &lt;detail&gt;&amp;pwn;&lt;/detail&gt;&lt;/event&gt;&lt;!--\"/>"
        ),
        "{event}"
    );
    // One `<` per tag the builder emits (prolog, event, point, detail, contact,
    // track, sensor, remarks, and the two closing tags), and one `>` each: the
    // injected markup added none.
    assert_eq!(event.matches('<').count(), 11, "{event}");
    assert_eq!(event.matches('>').count(), 11, "{event}");
    assert!(event.ends_with("</detail></event>"), "{event}");

    // The same for the mission id, which lands in element text, not an attribute.
    let ls = UasDatalink {
        mission_id: Some("a & b </remarks><script>".into()),
        ..sample()
    };
    let event = cot_event(&ls, CotOptions::default()).expect("event");
    assert!(
        event.contains("mission=a &amp; b &lt;/remarks&gt;&lt;script&gt;"),
        "{event}"
    );
    assert_eq!(event.matches('<').count(), 11, "{event}");
}

/// A set with no platform position yields no event at all: a track at lat/lon 0
/// would put a phantom aircraft in the Gulf of Guinea.
#[test]
fn a_set_without_position_yields_no_event() {
    let no_position = UasDatalink {
        sensor_lat_deg: None,
        sensor_lon_deg: None,
        ..sample()
    };
    assert_eq!(cot_event(&no_position, CotOptions::default()), None);
    // Longitude alone missing is just as fatal.
    let half = UasDatalink {
        sensor_lon_deg: None,
        ..sample()
    };
    assert_eq!(cot_event(&half, CotOptions::default()), None);
}

/// Without an altitude, hae takes the same unknown sentinel as ce / le rather
/// than reading as sea level.
#[test]
fn absent_altitude_becomes_the_unknown_sentinel() {
    let ls = UasDatalink {
        sensor_alt_m: None,
        ..sample()
    };
    let event = cot_event(&ls, CotOptions::default()).expect("event");
    assert!(event.contains("hae=\"9999999.0\""), "{event}");
}

#[cfg(feature = "udp-egress")]
mod sink {
    use super::*;

    use std::net::SocketAddr;
    use std::time::Duration;

    use g2g_core::element::AsyncElement;
    use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::{Caps, G2gError, PropValue};
    use g2g_plugins::cotsink::CotSink;

    struct NullOut;
    impl g2g_core::OutputSink for NullOut {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            packet_slot.take();
            core::task::Poll::Ready(Ok(g2g_core::element::PushOutcome::Accepted))
        }
    }

    /// One KLV frame carrying `n` consecutive local sets, a second apart, as a
    /// TS demuxer hands them over.
    fn klv_frame(count: u64) -> Frame {
        let mut buf = Vec::new();
        for i in 0..count {
            let ls = UasDatalink {
                timestamp_us: Some(T + i * 1_000_000),
                ..sample()
            };
            buf.extend_from_slice(&ls.encode());
        }
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        }
    }

    async fn recv_n(sock: &tokio::net::UdpSocket, n: usize) -> Vec<String> {
        let mut out = Vec::with_capacity(n);
        let mut buf = [0u8; 4096];
        for _ in 0..n {
            let len = tokio::time::timeout(Duration::from_secs(2), sock.recv(&mut buf))
                .await
                .expect("recv timed out: a loopback datagram was lost")
                .expect("recv");
            out.push(String::from_utf8(buf[..len].to_vec()).expect("utf-8 event"));
        }
        out
    }

    /// One datagram per local set, each a complete event carrying that set's own
    /// timestamp.
    #[tokio::test]
    async fn sink_sends_one_datagram_per_local_set() {
        let receiver = tokio::net::UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("bind receiver");
        let dest: SocketAddr = receiver.local_addr().expect("local addr");

        let mut sink = CotSink::new(dest).with_uid("drone-7");
        sink.configure_pipeline(&Caps::Klv).expect("configure");

        let mut out = NullOut;
        sink.process(PipelinePacket::DataFrame(klv_frame(3)), &mut out)
            .await
            .expect("send");

        let events = recv_n(&receiver, 3).await;
        assert_eq!(sink.events_sent(), 3);
        for (i, event) in events.iter().enumerate() {
            assert!(event.contains("uid=\"drone-7\""), "{event}");
            assert!(event.ends_with("</event>"), "{event}");
            let want = format!("time=\"2023-11-14T22:13:2{}.123456Z\"", i);
            assert!(event.contains(&want), "{event} lacks {want}");
        }
        // Nothing further arrives: exactly one event per set.
        let mut buf = [0u8; 4096];
        assert!(
            tokio::time::timeout(Duration::from_millis(100), receiver.recv(&mut buf))
                .await
                .is_err(),
            "a fourth datagram was sent for three local sets"
        );
    }

    /// A packet that is not a valid local set (corrupted checksum) sends nothing.
    #[tokio::test]
    async fn corrupt_local_set_sends_nothing() {
        let receiver = tokio::net::UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("bind receiver");
        let dest: SocketAddr = receiver.local_addr().expect("local addr");
        let mut sink = CotSink::new(dest);
        sink.configure_pipeline(&Caps::Klv).expect("configure");

        let mut bytes = sample().encode();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x01;
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        };

        let mut out = NullOut;
        sink.process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .expect("send");
        assert_eq!(sink.events_sent(), 0);
    }

    /// The TCP path (the transport to a TAK server) writes the same events on a
    /// connection instead of a datagram each.
    #[tokio::test]
    async fn tcp_protocol_writes_events_on_a_connection() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let dest: SocketAddr = listener.local_addr().expect("local addr");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            let mut got = String::new();
            // Two events; read until both closing tags have arrived.
            while got.matches("</event>").count() < 2 {
                stream.readable().await.expect("readable");
                let mut buf = [0u8; 4096];
                match stream.try_read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => got.push_str(&String::from_utf8_lossy(&buf[..n])),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                    Err(e) => panic!("read: {e}"),
                }
            }
            got
        });

        let mut sink = CotSink::new(dest).with_tcp();
        sink.configure_pipeline(&Caps::Klv).expect("configure");
        let mut out = NullOut;
        sink.process(PipelinePacket::DataFrame(klv_frame(2)), &mut out)
            .await
            .expect("send");

        let got = tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .expect("server timed out")
            .expect("server task");
        assert_eq!(sink.events_sent(), 2);
        assert_eq!(got.matches("<event ").count(), 2, "{got}");
    }

    /// The element is reachable by name from a launch line, with its properties
    /// visible to the parser.
    #[test]
    fn registered_for_launch() {
        let reg = g2g_plugins::registry::default_registry();
        let mut element = reg.make_element("cotsink").expect("cotsink registered");
        element
            .set_property("host", PropValue::Str("239.2.3.1".into()))
            .expect("host");
        element
            .set_property("port", PropValue::Uint(6969))
            .expect("port");
        assert_eq!(element.get_property("port"), Some(PropValue::Uint(6969)));
    }

    /// Every knob a launch line can set round-trips onto the field the sink acts
    /// on, and the transport selector rejects an unknown protocol.
    #[test]
    fn properties_round_trip() {
        let mut sink = CotSink::new("239.2.3.1:6969".parse().unwrap());
        for (name, value) in [
            ("host", PropValue::Str("239.2.3.1".into())),
            ("port", PropValue::Uint(8087)),
            ("protocol", PropValue::Str("tcp".into())),
            ("uid", PropValue::Str("drone-7".into())),
            ("cot-type", PropValue::Str("a-u-A-M-H-Q".into())),
            ("stale-seconds", PropValue::Uint(90)),
            ("ttl-mc", PropValue::Uint(8)),
            ("verify-checksum", PropValue::Bool(false)),
        ] {
            assert!(
                sink.properties().iter().any(|s| s.name == name),
                "declares {name}"
            );
            sink.set_property(name, value.clone()).expect(name);
            assert_eq!(sink.get_property(name), Some(value), "{name} round-trips");
        }
        assert!(sink
            .set_property("protocol", PropValue::Str("carrier-pigeon".into()))
            .is_err());
        // The type property really drives the emitted event.
        let event = cot_event(
            &sample(),
            CotOptions {
                cot_type: "a-u-A-M-H-Q",
                uid: "drone-7",
                stale_secs: 90,
            },
        )
        .expect("event");
        assert!(event.contains("type=\"a-u-A-M-H-Q\""), "{event}");
    }
}

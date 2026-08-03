//! M814: the ST 0805.1 Sensor Point of Interest event. The conventions under
//! test (the `b-m-p-s-p-i` type, the platform-uid + sensor-name uid, the
//! `<link relation="p-p">` back to the platform track, `how="m-p"`, target
//! location preferred over frame center) are jmisb's `KlvToCot`.

use g2g_plugins::cotsink::{cot_spi_event, CotOptions};
use g2g_plugins::klv::UasDatalink;

/// 2023-11-14T22:13:20.123456Z.
const T: u64 = 1_700_000_000_123_456;

fn sample() -> UasDatalink {
    UasDatalink {
        timestamp_us: Some(T),
        mission_id: Some("Mission 12".into()),
        platform_designation: Some("Predator".into()),
        image_source_sensor: Some("EO Nose".into()),
        target_lat_deg: Some(60.2),
        target_lon_deg: Some(24.9),
        target_alt_m: Some(15.5),
        frame_center_lat_deg: Some(60.18),
        frame_center_lon_deg: Some(24.84),
        frame_center_alt_m: Some(12.0),
        ..Default::default()
    }
}

/// The whole SPI event, byte for byte, at the target location.
#[test]
fn known_local_set_builds_the_expected_spi_event() {
    let event = cot_spi_event(&sample(), CotOptions::default()).expect("event");
    assert_eq!(
        event,
        concat!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n",
            "<event version=\"2.0\" uid=\"Predator_Mission 12_EO Nose\" type=\"b-m-p-s-p-i\"",
            " time=\"2023-11-14T22:13:20.123456Z\"",
            " start=\"2023-11-14T22:13:20.123456Z\"",
            " stale=\"2023-11-14T22:13:30.123456Z\" how=\"m-p\">",
            "<point lat=\"60.200000\" lon=\"24.900000\" hae=\"15.5\"",
            " ce=\"9999999.0\" le=\"9999999.0\"/>",
            "<detail><link relation=\"p-p\" type=\"a-f-A-M-F-Q\"",
            " uid=\"Predator_Mission 12\"/></detail></event>",
        )
    );
}

/// Without a complete target location the point falls back to the frame
/// center; without either, or without identity strings, the conventions
/// degrade the way jmisb's do.
#[test]
fn fallbacks_match_the_st0805_conventions() {
    // Incomplete target location (no altitude): frame center is the point.
    let mut ls = sample();
    ls.target_alt_m = None;
    let event = cot_spi_event(&ls, CotOptions::default()).expect("event");
    assert!(event.contains("lat=\"60.180000\" lon=\"24.840000\" hae=\"12.0\""));

    // Neither point complete: no event.
    ls.frame_center_alt_m = None;
    assert_eq!(cot_spi_event(&ls, CotOptions::default()), None);

    // No designation / mission pair: the configured uid is the platform uid.
    // No sensor name: the jmisb "UNKNOWN" suffix.
    let ls = UasDatalink {
        timestamp_us: Some(T),
        target_lat_deg: Some(60.2),
        target_lon_deg: Some(24.9),
        target_alt_m: Some(15.5),
        ..Default::default()
    };
    let event = cot_spi_event(&ls, CotOptions::default()).expect("event");
    assert!(event.contains("uid=\"g2g-uas_UNKNOWN\""), "{event}");
    assert!(event.contains("<link relation=\"p-p\" type=\"a-f-A-M-F-Q\" uid=\"g2g-uas\"/>"));
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
        fn push<'a>(
            &'a mut self,
            _packet: PipelinePacket,
        ) -> g2g_core::element::BoxFuture<'a, Result<g2g_core::element::PushOutcome, G2gError>>
        {
            Box::pin(async { Ok(g2g_core::element::PushOutcome::Accepted) })
        }
    }

    /// With `spi=true` each local set produces the platform event and then the
    /// SPI event, as two datagrams.
    #[tokio::test]
    async fn spi_property_sends_a_second_event_per_local_set() {
        let receiver = tokio::net::UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("bind receiver");
        let dest: SocketAddr = receiver.local_addr().expect("local addr");

        let mut ls = sample();
        // The platform event needs a platform position too.
        ls.sensor_lat_deg = Some(60.176822);
        ls.sensor_lon_deg = Some(24.828835);
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(ls.encode().into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        };

        let mut sink = CotSink::new(dest);
        sink.set_property("spi", PropValue::Bool(true))
            .expect("spi prop");
        sink.configure_pipeline(&Caps::Klv).expect("configure");
        let mut out = NullOut;
        sink.process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .expect("send");
        assert_eq!(sink.events_sent(), 2);

        let mut buf = [0u8; 4096];
        let mut events = Vec::new();
        for _ in 0..2 {
            let len = tokio::time::timeout(Duration::from_secs(2), receiver.recv(&mut buf))
                .await
                .expect("recv timed out")
                .expect("recv");
            events.push(String::from_utf8(buf[..len].to_vec()).expect("utf-8"));
        }
        assert!(events[0].contains("type=\"a-f-A-M-F-Q\""), "{}", events[0]);
        assert!(events[1].contains("type=\"b-m-p-s-p-i\""), "{}", events[1]);
        assert!(
            events[1].contains("uid=\"Predator_Mission 12_EO Nose\""),
            "{}",
            events[1]
        );
    }
}

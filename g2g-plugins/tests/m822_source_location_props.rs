//! M822: the feature-gated sources the registry default-builds with a
//! placeholder target must take that target from a launch line. Each test
//! asserts both halves the parser needs: the property is declared (so
//! `parse_launch` can pick its `PropKind`) and it round-trips onto the field the
//! element acts on, then parses in a real launch line. No network / device is
//! touched: construction and property state only.

#[cfg(any(feature = "rtsp", feature = "v4l2", feature = "webrtc"))]
use g2g_core::PropValue;

#[cfg(any(feature = "rtsp", feature = "v4l2", feature = "webrtc"))]
fn declares(specs: &[g2g_core::PropertySpec], name: &str) -> bool {
    specs.iter().any(|s| s.name == name)
}

#[cfg(feature = "rtsp")]
#[test]
fn rtspsrc_location_round_trips_and_parses() {
    use g2g_core::runtime::{parse_launch, SourceLoop};
    use g2g_plugins::rtspsrc::RtspSrc;

    let mut s = RtspSrc::new("");
    assert!(declares(s.properties(), "location"));
    s.set_property(
        "location",
        PropValue::Str("rtsp://cam.local:554/stream1".into()),
    )
    .unwrap();
    assert_eq!(
        s.get_property("location"),
        Some(PropValue::Str("rtsp://cam.local:554/stream1".into()))
    );

    let reg = g2g_plugins::registry::default_registry();
    parse_launch(
        &reg,
        "rtspsrc location=rtsp://cam.local:554/stream1 ! fakesink",
    )
    .expect("launch line sets location");
    assert!(
        parse_launch(&reg, "rtspsrc uri=rtsp://cam.local/s ! fakesink").is_err(),
        "an undeclared property is rejected, never silently dropped"
    );
}

#[cfg(feature = "v4l2")]
#[test]
fn v4l2src_device_round_trips_and_parses() {
    use g2g_core::runtime::{parse_launch, SourceLoop};
    use g2g_plugins::v4l2src::V4l2Src;

    let mut s = V4l2Src::new("/dev/video0");
    assert!(declares(s.properties(), "device"));
    assert_eq!(
        s.get_property("device"),
        Some(PropValue::Str("/dev/video0".into()))
    );
    s.set_property("device", PropValue::Str("/dev/video2".into()))
        .unwrap();
    assert_eq!(
        s.get_property("device"),
        Some(PropValue::Str("/dev/video2".into()))
    );

    let reg = g2g_plugins::registry::default_registry();
    parse_launch(&reg, "v4l2src device=/dev/video2 ! fakesink").expect("launch line sets device");
    assert!(
        parse_launch(&reg, "v4l2src location=/dev/video2 ! fakesink").is_err(),
        "an undeclared property is rejected, never silently dropped"
    );
}

#[cfg(feature = "webrtc")]
#[test]
fn whepsessionsrc_endpoint_and_ice_round_trip() {
    use g2g_core::fanout::MultiOutputSource;
    use g2g_plugins::webrtcwhepsession::WebRtcWhepSessionSrc;

    let mut s = WebRtcWhepSessionSrc::new("");
    for name in [
        "location",
        "auth-token",
        "stun-server",
        "turn-server",
        "turn-user",
        "turn-pass",
        "num-buffers",
    ] {
        assert!(declares(s.properties(), name), "declares {name}");
    }
    s.set_property("location", PropValue::Str("http://sfu/whep".into()))
        .unwrap();
    s.set_property("auth-token", PropValue::Str("tok".into()))
        .unwrap();
    s.set_property("stun-server", PropValue::Str("stun.l:19302".into()))
        .unwrap();
    s.set_property("turn-server", PropValue::Str("relay:3478".into()))
        .unwrap();
    s.set_property("turn-user", PropValue::Str("u".into()))
        .unwrap();
    s.set_property("turn-pass", PropValue::Str("p".into()))
        .unwrap();
    s.set_property("num-buffers", PropValue::Uint(20)).unwrap();

    assert_eq!(
        s.get_property("location"),
        Some(PropValue::Str("http://sfu/whep".into()))
    );
    assert_eq!(
        s.get_property("auth-token"),
        Some(PropValue::Str("tok".into()))
    );
    assert_eq!(
        s.get_property("stun-server"),
        Some(PropValue::Str("stun.l:19302".into()))
    );
    assert_eq!(
        s.get_property("turn-server"),
        Some(PropValue::Str("relay:3478".into()))
    );
    assert_eq!(
        s.get_property("turn-user"),
        Some(PropValue::Str("u".into()))
    );
    assert_eq!(
        s.get_property("turn-pass"),
        Some(PropValue::Str("p".into()))
    );
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Uint(20)));

    // An empty string clears an optional field.
    s.set_property("turn-server", PropValue::Str(String::new()))
        .unwrap();
    assert_eq!(
        s.get_property("turn-server"),
        Some(PropValue::Str(String::new()))
    );
    assert!(
        s.set_property("num-buffers", PropValue::Str("20".into()))
            .is_err(),
        "num-buffers is a Uint"
    );
    assert!(s.set_property("uri", PropValue::Str("x".into())).is_err());
}

#[cfg(feature = "webrtc")]
#[test]
fn whepsessionsrc_launch_line_sets_endpoint() {
    use g2g_core::graph::NodeKind;
    use g2g_core::runtime::parse_launch;

    let reg = g2g_plugins::registry::default_registry();
    let vg = parse_launch(
        &reg,
        "webrtcwhepsessionsrc name=s location=http://sfu/whep num-buffers=4  s. ! fakesink  s. ! fakesink",
    )
    .expect("launch line parses")
    .finish()
    .expect("valid graph");
    let fanouts: Vec<NodeKind> = vg
        .topo()
        .iter()
        .map(|&n| vg.kind(n))
        .filter(|k| matches!(k, NodeKind::FanoutSrc(_)))
        .collect();
    assert_eq!(fanouts, [NodeKind::FanoutSrc(2)]);
    assert!(
        parse_launch(
            &reg,
            "webrtcwhepsessionsrc name=s uri=http://sfu/whep  s. ! fakesink  s. ! fakesink",
        )
        .is_err(),
        "an undeclared property is rejected, never silently dropped"
    );
}

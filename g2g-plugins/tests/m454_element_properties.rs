//! M454: settable element properties for gst-launch parity. Each property newly
//! exposed for `parse_launch` must (a) appear in `properties()` so the parser can
//! look up its `PropKind`, and (b) round-trip through `set_property` /
//! `get_property` onto the real field the element acts on. These assert both,
//! per element, gated on the feature that builds it.

use g2g_core::{AsyncElement, PropValue, PropertySpec};

/// True when a spec table declares a property of this name (the half
/// `parse_launch` reads to determine the value kind).
fn declares(specs: &[PropertySpec], name: &str) -> bool {
    specs.iter().any(|s| s.name == name)
}

#[cfg(feature = "av1-encode")]
#[test]
fn av1enc_bitrate_and_speed() {
    use g2g_plugins::av1enc::Av1Enc;
    let mut e = Av1Enc::new();
    assert!(declares(e.properties(), "bitrate"));
    assert!(declares(e.properties(), "speed"));
    e.set_property("bitrate", PropValue::Uint(2_000_000)).unwrap();
    assert_eq!(e.get_property("bitrate"), Some(PropValue::Uint(2_000_000)));
    e.set_property("speed", PropValue::Uint(6)).unwrap();
    assert_eq!(e.get_property("speed"), Some(PropValue::Uint(6)));
}

#[cfg(feature = "vpx")]
#[test]
fn vpxenc_codec_and_bitrate() {
    use g2g_plugins::vpxenc::VpxEnc;
    let mut e = VpxEnc::new();
    assert!(declares(e.properties(), "codec"));
    e.set_property("codec", PropValue::Str("vp8".into())).unwrap();
    assert_eq!(e.get_property("codec"), Some(PropValue::Str("vp8".into())));
    // bits/second in, folded to libvpx kbps and back: 800 kbps round number.
    e.set_property("bitrate", PropValue::Uint(800_000)).unwrap();
    assert_eq!(e.get_property("bitrate"), Some(PropValue::Uint(800_000)));
    assert!(e.set_property("codec", PropValue::Str("av1".into())).is_err(), "rejects non-VP8/9");
}

#[cfg(feature = "opus")]
#[test]
fn opusenc_bitrate() {
    use g2g_plugins::opusenc::OpusEnc;
    let mut e = OpusEnc::new();
    assert!(declares(e.properties(), "bitrate"));
    e.set_property("bitrate", PropValue::Uint(96_000)).unwrap();
    assert_eq!(e.get_property("bitrate"), Some(PropValue::Uint(96_000)));
    // 0 selects libopus auto.
    e.set_property("bitrate", PropValue::Uint(0)).unwrap();
    assert_eq!(e.get_property("bitrate"), Some(PropValue::Uint(0)));
}

#[cfg(feature = "mjpeg-encode")]
#[test]
fn mjpegenc_quality() {
    use g2g_plugins::mjpegenc::MjpegEnc;
    let mut e = MjpegEnc::new();
    assert!(declares(e.properties(), "quality"));
    e.set_property("quality", PropValue::Uint(50)).unwrap();
    assert_eq!(e.get_property("quality"), Some(PropValue::Uint(50)));
    // Clamped to 100.
    e.set_property("quality", PropValue::Uint(250)).unwrap();
    assert_eq!(e.get_property("quality"), Some(PropValue::Uint(100)));
}

#[cfg(feature = "mjpeg")]
#[test]
fn mjpegdec_output_format() {
    use g2g_plugins::mjpegdec::MjpegDec;
    let mut e = MjpegDec::new();
    assert!(declares(e.properties(), "output-format"));
    e.set_property("output-format", PropValue::Str("i420".into())).unwrap();
    assert_eq!(e.get_property("output-format"), Some(PropValue::Str("i420".into())));
    assert!(e.set_property("output-format", PropValue::Str("rgb565".into())).is_err());
}

#[cfg(feature = "analytics")]
#[test]
fn analyticsoverlay_thickness() {
    use g2g_plugins::analyticsoverlay::AnalyticsOverlay;
    let mut e = AnalyticsOverlay::new();
    assert!(declares(e.properties(), "thickness"));
    e.set_property("thickness", PropValue::Uint(5)).unwrap();
    assert_eq!(e.get_property("thickness"), Some(PropValue::Uint(5)));
    // Clamped to >= 1.
    e.set_property("thickness", PropValue::Uint(0)).unwrap();
    assert_eq!(e.get_property("thickness"), Some(PropValue::Uint(1)));
}

#[test]
fn textoverlay_color_packs_argb() {
    use g2g_plugins::textoverlay::TextOverlay;
    let mut e = TextOverlay::new();
    assert!(declares(e.properties(), "color"));
    // 0xAARRGGBB: opaque red.
    e.set_property("color", PropValue::Uint(0xFFFF_0000)).unwrap();
    assert_eq!(e.get_property("color"), Some(PropValue::Uint(0xFFFF_0000)));
}

#[cfg(feature = "udp-ingress")]
#[test]
fn udpsrc_address_and_port() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::udpsrc::UdpSrc;
    let mut s = UdpSrc::new("0.0.0.0:5004".parse().unwrap());
    assert!(declares(s.properties(), "port"));
    assert!(declares(s.properties(), "address"));
    s.set_property("port", PropValue::Uint(6000)).unwrap();
    s.set_property("address", PropValue::Str("127.0.0.1".into())).unwrap();
    assert_eq!(s.get_property("port"), Some(PropValue::Uint(6000)));
    assert_eq!(s.get_property("address"), Some(PropValue::Str("127.0.0.1".into())));
    assert!(s.set_property("port", PropValue::Uint(70000)).is_err(), "rejects out-of-range port");
}

#[cfg(feature = "srt")]
#[test]
fn srtsrc_latency_and_passphrase() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::srtsrc::SrtSrc;
    let mut s = SrtSrc::new("0.0.0.0:9000".parse().unwrap());
    s.set_property("latency", PropValue::Uint(250)).unwrap();
    assert_eq!(s.get_property("latency"), Some(PropValue::Uint(250)));
    s.set_property("passphrase", PropValue::Str("hunter2hunter2".into())).unwrap();
    assert_eq!(s.get_property("passphrase"), Some(PropValue::Str("hunter2hunter2".into())));
}

#[cfg(feature = "udp-egress")]
#[test]
fn udpsink_host_port_payload() {
    use g2g_plugins::udpsink::UdpSink;
    let mut e = UdpSink::new("127.0.0.1:5004".parse().unwrap());
    e.set_property("host", PropValue::Str("10.0.0.5".into())).unwrap();
    e.set_property("port", PropValue::Uint(5600)).unwrap();
    e.set_property("payload-type", PropValue::Uint(97)).unwrap();
    assert_eq!(e.get_property("host"), Some(PropValue::Str("10.0.0.5".into())));
    assert_eq!(e.get_property("port"), Some(PropValue::Uint(5600)));
    assert_eq!(e.get_property("payload-type"), Some(PropValue::Uint(97)));
    assert!(e.set_property("payload-type", PropValue::Uint(200)).is_err(), "PT must be <= 127");
}

#[test]
fn h264parse_config_interval() {
    use g2g_plugins::h264parse::H264Parse;
    let mut e = H264Parse::reframing();
    assert!(declares(e.properties(), "config-interval"));
    e.set_property("config-interval", PropValue::Int(-1)).unwrap();
    assert_eq!(e.get_property("config-interval"), Some(PropValue::Int(-1)));
    e.set_property("config-interval", PropValue::Int(2)).unwrap();
    assert_eq!(e.get_property("config-interval"), Some(PropValue::Int(2)));
    assert!(e.set_property("config-interval", PropValue::Int(-2)).is_err(), "rejects < -1");
}

#[test]
fn h265parse_config_interval() {
    use g2g_plugins::h265parse::H265Parse;
    let mut e = H265Parse::reframing();
    assert!(declares(e.properties(), "config-interval"));
    e.set_property("config-interval", PropValue::Int(-1)).unwrap();
    assert_eq!(e.get_property("config-interval"), Some(PropValue::Int(-1)));
}

#[test]
fn tsmux_pat_pmt_interval() {
    use g2g_plugins::tsmux::TsMux;
    let mut e = TsMux::new();
    assert!(declares(e.properties(), "pat-interval"));
    assert!(declares(e.properties(), "pmt-interval"));
    e.set_property("pat-interval", PropValue::Uint(100)).unwrap();
    assert_eq!(e.get_property("pat-interval"), Some(PropValue::Uint(100)));
    // pat / pmt share one cadence (the tables are emitted together).
    e.set_property("pmt-interval", PropValue::Uint(250)).unwrap();
    assert_eq!(e.get_property("pat-interval"), Some(PropValue::Uint(250)));
}

#[test]
fn mkvmux_streamable() {
    use g2g_plugins::mkvmux::MkvMux;
    let mut e = MkvMux::new();
    assert!(declares(e.properties(), "streamable"));
    e.set_property("streamable", PropValue::Bool(true)).unwrap();
    assert_eq!(e.get_property("streamable"), Some(PropValue::Bool(true)));
}

#[cfg(feature = "std")]
#[test]
fn mp4mux_fragment_duration() {
    use g2g_plugins::mp4mux::Mp4Mux;
    let mut e = Mp4Mux::new();
    assert!(declares(e.properties(), "fragment-duration"));
    e.set_property("fragment-duration", PropValue::Uint(2000)).unwrap();
    assert_eq!(e.get_property("fragment-duration"), Some(PropValue::Uint(2000)));
}

// parse_launch end to end: the parser looks up the kind in properties() and calls
// set_property, so a pipeline that sets a newly exposed property must parse, and an
// undeclared property must be rejected.
#[cfg(feature = "mjpeg-encode")]
#[test]
fn parse_launch_sets_encoder_property() {
    use g2g_core::runtime::parse_launch;
    use g2g_plugins::registry::default_registry;
    let reg = default_registry();
    assert!(
        parse_launch(&reg, "videotestsrc num-buffers=2 ! mjpegenc quality=50 ! fakesink").is_ok(),
        "a launch line setting the new quality property parses"
    );
    assert!(
        parse_launch(&reg, "videotestsrc num-buffers=2 ! mjpegenc bogus=1 ! fakesink").is_err(),
        "an undeclared property is rejected"
    );
}

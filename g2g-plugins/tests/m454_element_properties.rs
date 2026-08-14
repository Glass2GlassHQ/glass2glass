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

/// M888: the CMAF chunked-consumption switch is settable from a launch line.
#[cfg(feature = "dash")]
#[test]
fn dashsrc_low_latency() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::dashsrc::DashSrc;
    let mut e = DashSrc::new("http://h/manifest.mpd");
    assert!(declares(e.properties(), "low-latency"));
    e.set_property("low-latency", PropValue::Bool(true))
        .unwrap();
    assert_eq!(e.get_property("low-latency"), Some(PropValue::Bool(true)));
}

#[cfg(feature = "av1-encode")]
#[test]
fn av1enc_bitrate_speed_and_quantizer() {
    use g2g_plugins::av1enc::Av1Enc;
    let mut e = Av1Enc::new();
    assert!(declares(e.properties(), "bitrate"));
    assert!(declares(e.properties(), "speed"));
    assert!(declares(e.properties(), "quantizer"));
    e.set_property("bitrate", PropValue::Uint(2_000_000))
        .unwrap();
    assert_eq!(e.get_property("bitrate"), Some(PropValue::Uint(2_000_000)));
    e.set_property("speed", PropValue::Uint(6)).unwrap();
    assert_eq!(e.get_property("speed"), Some(PropValue::Uint(6)));
    // Constant quality and a rate target are exclusive: setting one clears the
    // other (rav1e reads only one of them).
    e.set_property("quantizer", PropValue::Uint(90)).unwrap();
    assert_eq!(e.get_property("quantizer"), Some(PropValue::Uint(90)));
    assert_eq!(e.get_property("bitrate"), Some(PropValue::Uint(0)));
}

#[cfg(feature = "vpx")]
#[test]
fn vpxenc_codec_and_bitrate() {
    use g2g_plugins::vpxenc::VpxEnc;
    let mut e = VpxEnc::new();
    assert!(declares(e.properties(), "codec"));
    e.set_property("codec", PropValue::Str("vp8".into()))
        .unwrap();
    assert_eq!(e.get_property("codec"), Some(PropValue::Str("vp8".into())));
    // bits/second in, folded to libvpx kbps and back: 800 kbps round number.
    e.set_property("bitrate", PropValue::Uint(800_000)).unwrap();
    assert_eq!(e.get_property("bitrate"), Some(PropValue::Uint(800_000)));
    assert!(
        e.set_property("codec", PropValue::Str("av1".into()))
            .is_err(),
        "rejects non-VP8/9"
    );
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

#[cfg(feature = "opus")]
#[test]
fn opusenc_frame_size_and_complexity() {
    use g2g_plugins::opusenc::OpusEnc;
    let mut e = OpusEnc::new();
    assert!(declares(e.properties(), "frame-size"));
    assert!(declares(e.properties(), "complexity"));
    // gst's enum integers: whole ms, except 2 = 2.5 ms.
    e.set_property("frame-size", PropValue::Uint(2)).unwrap();
    assert_eq!(e.get_property("frame-size"), Some(PropValue::Uint(2)));
    assert!(
        e.set_property("frame-size", PropValue::Uint(30)).is_err(),
        "rejects a duration Opus has no frame of"
    );
    e.set_property("complexity", PropValue::Uint(4)).unwrap();
    assert_eq!(e.get_property("complexity"), Some(PropValue::Uint(4)));
    assert!(
        e.set_property("complexity", PropValue::Uint(11)).is_err(),
        "rejects complexity above 10"
    );
}

#[cfg(feature = "opus")]
#[test]
fn opusenc_audio_type() {
    use g2g_plugins::opusenc::{OpusAudioType, OpusEnc};
    let mut e = OpusEnc::new();
    assert!(declares(e.properties(), "audio-type"));
    assert_eq!(
        e.get_property("audio-type"),
        Some(PropValue::Str("generic".into())),
        "gst opusenc's default"
    );
    e.set_property("audio-type", PropValue::Str("voice".into()))
        .unwrap();
    assert_eq!(
        e.get_property("audio-type"),
        Some(PropValue::Str("voice".into()))
    );
    assert!(
        e.set_property("audio-type", PropValue::Str("music".into()))
            .is_err(),
        "rejects a mode libopus has no application for"
    );
    // the mode has to reach libopus: a voice encoder builds, and the low-delay
    // one builds with a shorter lookahead than the default mode's.
    let caps = g2g_core::Caps::Audio {
        format: g2g_core::AudioFormat::PcmS16Le,
        channels: 1,
        sample_rate: 48_000,
    };
    e.configure_pipeline(&caps)
        .expect("voice encoder initializes");
    let voice_lookahead = e.lookahead().expect("live encoder built");

    let mut low = OpusEnc::new().with_audio_type(OpusAudioType::RestrictedLowDelay);
    assert_eq!(
        low.get_property("audio-type"),
        Some(PropValue::Str("restricted-lowdelay".into())),
        "builder and property agree"
    );
    low.configure_pipeline(&caps).unwrap();
    assert!(
        low.lookahead().unwrap() < voice_lookahead,
        "restricted-lowdelay drops the SILK lookahead"
    );
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
    e.set_property("output-format", PropValue::Str("i420".into()))
        .unwrap();
    assert_eq!(
        e.get_property("output-format"),
        Some(PropValue::Str("i420".into()))
    );
    assert!(e
        .set_property("output-format", PropValue::Str("rgb565".into()))
        .is_err());
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

#[cfg(feature = "analytics")]
#[test]
fn analyticsoverlay_mask_alpha() {
    use g2g_plugins::analyticsoverlay::AnalyticsOverlay;
    let mut e = AnalyticsOverlay::new();
    assert!(declares(e.properties(), "mask-alpha"));
    e.set_property("mask-alpha", PropValue::Uint(140)).unwrap();
    assert_eq!(e.get_property("mask-alpha"), Some(PropValue::Uint(140)));
    // Clamped to a byte.
    e.set_property("mask-alpha", PropValue::Uint(4000)).unwrap();
    assert_eq!(e.get_property("mask-alpha"), Some(PropValue::Uint(255)));
}

#[test]
fn textoverlay_color_packs_argb() {
    use g2g_plugins::textoverlay::TextOverlay;
    let mut e = TextOverlay::new();
    assert!(declares(e.properties(), "color"));
    // 0xAARRGGBB: opaque red.
    e.set_property("color", PropValue::Uint(0xFFFF_0000))
        .unwrap();
    assert_eq!(e.get_property("color"), Some(PropValue::Uint(0xFFFF_0000)));
}

/// The shared placement / styling half of both timestamp overlays, plus
/// `timeoverlay`'s own `time-mode`.
#[test]
fn timeoverlay_time_mode_and_placement() {
    use g2g_plugins::timeoverlay::TimeOverlay;
    let mut e = TimeOverlay::new();
    for name in [
        "time-mode",
        "scale",
        "halignment",
        "valignment",
        "xpad",
        "ypad",
        "color",
        "shaded-background",
    ] {
        assert!(declares(e.properties(), name), "{name} is declared");
    }

    e.set_property("time-mode", PropValue::Str("elapsed-running-time".into()))
        .unwrap();
    assert_eq!(
        e.get_property("time-mode"),
        Some(PropValue::Str("elapsed-running-time".into()))
    );
    // `time-code` / `reference-timestamp` need frame-meta g2g has no producer
    // for, so they are not accepted rather than silently drawing something else.
    assert!(e
        .set_property("time-mode", PropValue::Str("time-code".into()))
        .is_err());

    e.set_property("halignment", PropValue::Str("right".into()))
        .unwrap();
    e.set_property("valignment", PropValue::Str("bottom".into()))
        .unwrap();
    e.set_property("xpad", PropValue::Uint(12)).unwrap();
    e.set_property("ypad", PropValue::Uint(4)).unwrap();
    e.set_property("shaded-background", PropValue::Bool(false))
        .unwrap();
    // 0xAARRGGBB, the textoverlay `color` packing: opaque green.
    e.set_property("color", PropValue::Uint(0xFF00_FF00))
        .unwrap();
    assert_eq!(
        e.get_property("halignment"),
        Some(PropValue::Str("right".into()))
    );
    assert_eq!(
        e.get_property("valignment"),
        Some(PropValue::Str("bottom".into()))
    );
    assert_eq!(e.get_property("xpad"), Some(PropValue::Uint(12)));
    assert_eq!(e.get_property("ypad"), Some(PropValue::Uint(4)));
    assert_eq!(
        e.get_property("shaded-background"),
        Some(PropValue::Bool(false))
    );
    assert_eq!(e.get_property("color"), Some(PropValue::Uint(0xFF00_FF00)));

    // Alignments g2g does not implement, and a zero magnification, are rejected.
    assert!(e
        .set_property("valignment", PropValue::Str("baseline".into()))
        .is_err());
    assert!(e.set_property("scale", PropValue::Uint(0)).is_err());
}

#[cfg(feature = "std")]
#[test]
fn clockoverlay_time_format() {
    use g2g_plugins::clockoverlay::ClockOverlay;
    let mut e = ClockOverlay::new();
    assert!(declares(e.properties(), "time-format"));
    assert_eq!(
        e.get_property("time-format"),
        Some(PropValue::Str("%H:%M:%S".into())),
        "gst's default clock format"
    );
    e.set_property("time-format", PropValue::Str("%F %T".into()))
        .unwrap();
    assert_eq!(
        e.get_property("time-format"),
        Some(PropValue::Str("%F %T".into()))
    );
    // The styling half is shared with timeoverlay.
    assert!(declares(e.properties(), "halignment"));
    e.set_property("halignment", PropValue::Str("center".into()))
        .unwrap();
    assert_eq!(
        e.get_property("halignment"),
        Some(PropValue::Str("center".into()))
    );
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
    s.set_property("address", PropValue::Str("127.0.0.1".into()))
        .unwrap();
    assert_eq!(s.get_property("port"), Some(PropValue::Uint(6000)));
    assert_eq!(
        s.get_property("address"),
        Some(PropValue::Str("127.0.0.1".into()))
    );
    assert!(
        s.set_property("port", PropValue::Uint(70000)).is_err(),
        "rejects out-of-range port"
    );
}

#[cfg(feature = "srt")]
#[test]
fn srtsrc_latency_and_passphrase() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::srtsrc::SrtSrc;
    let mut s = SrtSrc::new("0.0.0.0:9000".parse().unwrap());
    s.set_property("latency", PropValue::Uint(250)).unwrap();
    assert_eq!(s.get_property("latency"), Some(PropValue::Uint(250)));
    s.set_property("passphrase", PropValue::Str("hunter2hunter2".into()))
        .unwrap();
    assert_eq!(
        s.get_property("passphrase"),
        Some(PropValue::Str("hunter2hunter2".into()))
    );
}

#[cfg(feature = "udp-egress")]
#[test]
fn udpsink_host_port_payload() {
    use g2g_plugins::udpsink::UdpSink;
    let mut e = UdpSink::new("127.0.0.1:5004".parse().unwrap());
    e.set_property("host", PropValue::Str("10.0.0.5".into()))
        .unwrap();
    e.set_property("port", PropValue::Uint(5600)).unwrap();
    e.set_property("payload-type", PropValue::Uint(97)).unwrap();
    assert_eq!(
        e.get_property("host"),
        Some(PropValue::Str("10.0.0.5".into()))
    );
    assert_eq!(e.get_property("port"), Some(PropValue::Uint(5600)));
    assert_eq!(e.get_property("payload-type"), Some(PropValue::Uint(97)));
    assert!(
        e.set_property("payload-type", PropValue::Uint(200))
            .is_err(),
        "PT must be <= 127"
    );
}

#[cfg(feature = "http-src")]
#[test]
fn httpsrc_prebuffer_bytes() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::httpsrc::HttpSrc;
    let mut s = HttpSrc::new(
        "http://127.0.0.1/x.ts",
        g2g_core::Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::MpegTs,
        },
    );
    assert!(declares(s.properties(), "prebuffer-bytes"));
    s.set_property("prebuffer-bytes", PropValue::Uint(65536))
        .unwrap();
    assert_eq!(
        s.get_property("prebuffer-bytes"),
        Some(PropValue::Uint(65536))
    );
    assert!(
        s.set_property("prebuffer-bytes", PropValue::Uint(u64::MAX))
            .is_err(),
        "rejects an absurd window"
    );
}

#[cfg(feature = "rtsp-server")]
#[test]
fn rtspserversink_bind_rtp_and_session_knobs() {
    use g2g_plugins::rtspserversink::RtspServerSink;
    let mut e = RtspServerSink::new("0.0.0.0:8554".parse().unwrap());
    for name in [
        "address",
        "port",
        "payload-type",
        "ssrc",
        "rtcp-sr-interval",
        "timeout",
    ] {
        assert!(declares(e.properties(), name), "declares {name}");
    }
    e.set_property("address", PropValue::Str("127.0.0.1".into()))
        .unwrap();
    e.set_property("port", PropValue::Uint(9554)).unwrap();
    e.set_property("payload-type", PropValue::Uint(97)).unwrap();
    e.set_property("ssrc", PropValue::Uint(0xABCD_0001))
        .unwrap();
    e.set_property("rtcp-sr-interval", PropValue::Uint(2000))
        .unwrap();
    e.set_property("timeout", PropValue::Uint(30)).unwrap();
    assert_eq!(
        e.get_property("address"),
        Some(PropValue::Str("127.0.0.1".into()))
    );
    assert_eq!(e.get_property("port"), Some(PropValue::Uint(9554)));
    assert_eq!(e.get_property("payload-type"), Some(PropValue::Uint(97)));
    assert_eq!(e.get_property("ssrc"), Some(PropValue::Uint(0xABCD_0001)));
    assert_eq!(
        e.get_property("rtcp-sr-interval"),
        Some(PropValue::Uint(2000))
    );
    assert_eq!(e.get_property("timeout"), Some(PropValue::Uint(30)));
    // 0 disables reaping and reads back as 0.
    e.set_property("timeout", PropValue::Uint(0)).unwrap();
    assert_eq!(e.get_property("timeout"), Some(PropValue::Uint(0)));
    assert!(
        e.set_property("payload-type", PropValue::Uint(200))
            .is_err(),
        "PT must be <= 127"
    );
    assert!(
        e.set_property("rtcp-sr-interval", PropValue::Uint(0))
            .is_err(),
        "a zero SR interval is rejected"
    );
}

#[cfg(feature = "udp-egress")]
#[test]
fn udpsink_fec_knobs() {
    use g2g_plugins::udpsink::UdpSink;
    let mut e = UdpSink::new("127.0.0.1:5004".parse().unwrap());
    for name in [
        "fec-mode",
        "fec-columns",
        "fec-rows",
        "fec-payload-type",
        "fec-ssrc",
    ] {
        assert!(declares(e.properties(), name), "declares {name}");
    }
    e.set_property("fec-mode", PropValue::Str("flexfec".into()))
        .unwrap();
    e.set_property("fec-columns", PropValue::Uint(4)).unwrap();
    e.set_property("fec-rows", PropValue::Uint(3)).unwrap();
    e.set_property("fec-payload-type", PropValue::Uint(98))
        .unwrap();
    e.set_property("fec-ssrc", PropValue::Uint(0xFEC0_0004))
        .unwrap();
    assert_eq!(
        e.get_property("fec-mode"),
        Some(PropValue::Str("flexfec".into()))
    );
    assert_eq!(e.get_property("fec-columns"), Some(PropValue::Uint(4)));
    assert_eq!(e.get_property("fec-rows"), Some(PropValue::Uint(3)));
    assert_eq!(
        e.get_property("fec-payload-type"),
        Some(PropValue::Uint(98))
    );
    assert_eq!(
        e.get_property("fec-ssrc"),
        Some(PropValue::Uint(0xFEC0_0004))
    );
    assert!(
        e.set_property("fec-mode", PropValue::Str("bogus".into()))
            .is_err(),
        "unknown FEC scheme rejected"
    );
    assert!(
        e.set_property("fec-columns", PropValue::Uint(0)).is_err(),
        "block width must be >= 1"
    );
    assert!(
        e.set_property("fec-payload-type", PropValue::Uint(200))
            .is_err(),
        "FEC PT must be <= 127"
    );
}

#[test]
fn h264parse_config_interval() {
    use g2g_plugins::h264parse::H264Parse;
    let mut e = H264Parse::reframing();
    assert!(declares(e.properties(), "config-interval"));
    e.set_property("config-interval", PropValue::Int(-1))
        .unwrap();
    assert_eq!(e.get_property("config-interval"), Some(PropValue::Int(-1)));
    e.set_property("config-interval", PropValue::Int(2))
        .unwrap();
    assert_eq!(e.get_property("config-interval"), Some(PropValue::Int(2)));
    assert!(
        e.set_property("config-interval", PropValue::Int(-2))
            .is_err(),
        "rejects < -1"
    );
}

#[test]
fn h265parse_config_interval() {
    use g2g_plugins::h265parse::H265Parse;
    let mut e = H265Parse::reframing();
    assert!(declares(e.properties(), "config-interval"));
    e.set_property("config-interval", PropValue::Int(-1))
        .unwrap();
    assert_eq!(e.get_property("config-interval"), Some(PropValue::Int(-1)));
}

#[test]
fn tsmux_pat_pmt_interval() {
    use g2g_plugins::tsmux::TsMux;
    let mut e = TsMux::new();
    assert!(declares(e.properties(), "pat-interval"));
    assert!(declares(e.properties(), "pmt-interval"));
    e.set_property("pat-interval", PropValue::Uint(100))
        .unwrap();
    assert_eq!(e.get_property("pat-interval"), Some(PropValue::Uint(100)));
    // pat / pmt share one cadence (the tables are emitted together).
    e.set_property("pmt-interval", PropValue::Uint(250))
        .unwrap();
    assert_eq!(e.get_property("pat-interval"), Some(PropValue::Uint(250)));
}

#[test]
fn tsmux_pcr_interval() {
    // pcr-interval on both the single-input and fan-in muxers: declared, default
    // 3600 (matching GStreamer mpegtsmux), and round-trips through set/get.
    let mut single = g2g_plugins::tsmux::TsMux::new();
    assert!(declares(single.properties(), "pcr-interval"));
    assert_eq!(
        single.get_property("pcr-interval"),
        Some(PropValue::Uint(3600))
    );
    single
        .set_property("pcr-interval", PropValue::Uint(1800))
        .unwrap();
    assert_eq!(
        single.get_property("pcr-interval"),
        Some(PropValue::Uint(1800))
    );

    use g2g_core::MultiInputElement;
    let mut fanin = g2g_plugins::tsmuxn::TsMux::new(2);
    assert!(declares(fanin.properties(), "pcr-interval"));
    assert_eq!(
        fanin.get_property("pcr-interval"),
        Some(PropValue::Uint(3600))
    );
    fanin
        .set_property("pcr-interval", PropValue::Uint(9000))
        .unwrap();
    assert_eq!(
        fanin.get_property("pcr-interval"),
        Some(PropValue::Uint(9000))
    );
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
    e.set_property("fragment-duration", PropValue::Uint(2000))
        .unwrap();
    assert_eq!(
        e.get_property("fragment-duration"),
        Some(PropValue::Uint(2000))
    );
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
        parse_launch(
            &reg,
            "videotestsrc num-buffers=2 ! mjpegenc quality=50 ! fakesink"
        )
        .is_ok(),
        "a launch line setting the new quality property parses"
    );
    assert!(
        parse_launch(
            &reg,
            "videotestsrc num-buffers=2 ! mjpegenc bogus=1 ! fakesink"
        )
        .is_err(),
        "an undeclared property is rejected"
    );
}

#[test]
fn oggmux_serial() {
    use g2g_plugins::oggmux::OggMux;
    let mut e = OggMux::new();
    assert!(declares(e.properties(), "serial"));
    e.set_property("serial", PropValue::Uint(0xDEAD_BEEF))
        .unwrap();
    assert_eq!(e.get_property("serial"), Some(PropValue::Uint(0xDEAD_BEEF)));
    assert!(
        e.set_property("serial", PropValue::Uint(1 << 40)).is_err(),
        "a serial number is 32 bits"
    );
}

#[cfg(feature = "std")]
#[test]
fn oggmux_is_a_launch_element() {
    use g2g_core::runtime::parse_launch;
    use g2g_plugins::registry::default_registry;
    let reg = default_registry();
    assert!(
        parse_launch(
            &reg,
            "filesrc location=in.ogg ! oggdemux ! oggmux serial=7 ! fakesink"
        )
        .is_ok(),
        "oggmux is registered as a launch element"
    );
}

#[test]
fn compositor_canvas_and_flattened_pad_placement() {
    use g2g_core::{MultiInputElement, PropError};
    use g2g_plugins::compositor::{Compositor, CompositorPad};
    let mut e = Compositor::new(
        320,
        240,
        Vec::from([CompositorPad::at(0, 0), CompositorPad::at(0, 0)]),
    );
    for name in [
        "width",
        "height",
        "framerate",
        "background-color",
        "timed-output",
        "format",
        "sink0-xpos",
        "sink1-ypos",
        "sink7-height",
    ] {
        assert!(declares(e.properties(), name), "{name} is declared");
    }
    assert!(
        !declares(e.properties(), "sink8-xpos"),
        "the flattened pad names are bounded"
    );

    e.set_property("width", PropValue::Uint(640)).unwrap();
    e.set_property("height", PropValue::Uint(480)).unwrap();
    assert_eq!(e.get_property("width"), Some(PropValue::Uint(640)));
    assert_eq!(e.get_property("height"), Some(PropValue::Uint(480)));
    e.set_property("framerate", PropValue::Fraction(60, 1))
        .unwrap();
    assert_eq!(
        e.get_property("framerate"),
        Some(PropValue::Fraction(60, 1))
    );
    // 0xAARRGGBB, the textoverlay `color` packing: opaque green.
    e.set_property("background-color", PropValue::Uint(0xFF00FF00))
        .unwrap();
    assert_eq!(
        e.get_property("background-color"),
        Some(PropValue::Uint(0xFF00FF00))
    );
    e.set_property("format", PropValue::Str("nv12".into()))
        .unwrap();
    assert_eq!(
        e.get_property("format"),
        Some(PropValue::Str("nv12".into()))
    );
    // Timed output drives the declared tick interval off the framerate above.
    assert_eq!(e.tick_interval_ns(), None);
    e.set_property("timed-output", PropValue::Bool(true))
        .unwrap();
    assert_eq!(e.get_property("timed-output"), Some(PropValue::Bool(true)));
    assert_eq!(
        e.tick_interval_ns(),
        Some(1_000_000_000 * 65536 / (60 << 16))
    );
    e.set_property("timed-output", PropValue::Bool(false))
        .unwrap();
    assert_eq!(e.tick_interval_ns(), None);

    // Per-pad placement, gst's `sink_1::xpos` request-pad properties flattened.
    for (name, value) in [
        ("sink1-xpos", PropValue::Int(-8)),
        ("sink1-ypos", PropValue::Int(12)),
        ("sink1-zorder", PropValue::Uint(3)),
        ("sink1-alpha", PropValue::Uint(128)),
        ("sink1-width", PropValue::Uint(64)),
        ("sink1-height", PropValue::Uint(48)),
    ] {
        e.set_property(name, value.clone()).unwrap();
        assert_eq!(e.get_property(name), Some(value), "{name} round-trips");
    }

    assert_eq!(
        e.set_property("width", PropValue::Str("640".into())),
        Err(PropError::Type),
        "a canvas dimension is not a string"
    );
    assert_eq!(
        e.set_property("sink1-xpos", PropValue::Uint(4)),
        Err(PropError::Type),
        "a pad position is signed"
    );
    assert_eq!(
        e.set_property("sink1-alpha", PropValue::Uint(300)),
        Err(PropError::Value),
        "alpha is a byte"
    );
    assert_eq!(
        e.set_property("format", PropValue::Str("bgrx".into())),
        Err(PropError::Value),
        "only the formats the element composites"
    );
    // Declared for eight pads, but this instance has two: a placement that would
    // silently vanish is an error instead.
    assert_eq!(
        e.set_property("sink5-xpos", PropValue::Int(4)),
        Err(PropError::Value)
    );
    assert_eq!(e.get_property("sink5-xpos"), None);
    assert_eq!(
        e.set_property("bogus", PropValue::Int(0)),
        Err(PropError::Unknown)
    );
}

#[cfg(all(target_os = "linux", feature = "pipewire"))]
#[tokio::test]
async fn pipewiresrc_format_rate_and_channels() {
    use g2g_core::runtime::SourceLoop;
    use g2g_core::{AudioFormat, Caps, PropError};
    use g2g_plugins::pipewiresrc::PipeWireSrc;
    let mut e = PipeWireSrc::new();
    for name in [
        "target-object",
        "format",
        "samplerate",
        "channels",
        "num-buffers",
    ] {
        assert!(declares(e.properties(), name), "{name} is declared");
    }
    e.set_property("format", PropValue::Str("F32LE".into()))
        .unwrap();
    e.set_property("samplerate", PropValue::Uint(44_100))
        .unwrap();
    e.set_property("channels", PropValue::Uint(1)).unwrap();
    assert_eq!(
        e.get_property("format"),
        Some(PropValue::Str("F32LE".into()))
    );
    assert_eq!(e.get_property("samplerate"), Some(PropValue::Uint(44_100)));
    assert_eq!(e.get_property("channels"), Some(PropValue::Uint(1)));
    // all three land on the caps the element opens the stream with
    assert_eq!(
        e.intercept_caps().await,
        Ok(Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: 44_100,
        })
    );
    // a format with no PCM stream behind it is rejected, never silently kept
    assert_eq!(
        e.set_property("format", PropValue::Str("AAC".into())),
        Err(PropError::Value)
    );
}

/// The sink takes its format, rate and channels from the negotiated caps, so it
/// declares none of them rather than knobs it would have to ignore (same shape
/// as `alsasink` / `pulsesink`).
#[cfg(all(target_os = "linux", feature = "pipewire"))]
#[test]
fn pipewiresink_declares_only_the_node_target() {
    use g2g_core::PropError;
    use g2g_plugins::pipewiresink::PipeWireSink;
    let mut e = PipeWireSink::new();
    assert!(declares(e.properties(), "target-object"));
    for name in ["format", "samplerate", "channels"] {
        assert!(!declares(e.properties(), name), "{name} is not a sink knob");
        assert_eq!(
            e.set_property(name, PropValue::Uint(1)),
            Err(PropError::Unknown)
        );
    }
    e.set_property("target-object", PropValue::Str("spk0".into()))
        .unwrap();
    assert_eq!(
        e.get_property("target-object"),
        Some(PropValue::Str("spk0".into()))
    );
}

#[cfg(all(target_os = "linux", feature = "pipewire"))]
#[tokio::test]
async fn pipewirevideosrc_format_pin_and_geometry() {
    use g2g_core::runtime::SourceLoop;
    use g2g_core::{Caps, Dim, Interlace, PropError, Rate, RawVideoFormat};
    use g2g_plugins::pipewirevideosrc::PipeWireVideoSrc;
    let mut e = PipeWireVideoSrc::new();
    for name in [
        "target-object",
        "width",
        "height",
        "framerate",
        "format",
        "num-buffers",
    ] {
        assert!(declares(e.properties(), name), "{name} is declared");
    }
    e.set_property("width", PropValue::Uint(1280)).unwrap();
    e.set_property("height", PropValue::Uint(720)).unwrap();
    e.set_property("framerate", PropValue::Uint(60)).unwrap();
    e.set_property("format", PropValue::Str("NV12".into()))
        .unwrap();
    assert_eq!(
        e.get_property("format"),
        Some(PropValue::Str("NV12".into()))
    );
    // the pin is what the element advertises, so negotiation commits to the
    // format up front instead of waiting for a CapsChanged
    assert_eq!(
        e.intercept_caps().await,
        Ok(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Fixed(60 << 16),
            interlace: Interlace::Any,
        })
    );
    // an empty pin is the open negotiation the element defaults to
    e.set_property("format", PropValue::Str(String::new()))
        .unwrap();
    assert_eq!(
        e.get_property("format"),
        Some(PropValue::Str(String::new()))
    );
    assert_eq!(
        e.set_property("format", PropValue::Str("nonsense".into())),
        Err(PropError::Value)
    );
}

/// The other half of the contract: `parse_launch` looks each name up in
/// `properties()` for its `PropKind` and hands the parsed value to
/// `set_property`, so a launch line sets every one of them and a value the
/// element rejects fails the parse instead of being dropped.
#[cfg(all(target_os = "linux", feature = "pipewire"))]
#[test]
fn pipewire_launch_lines_set_every_property() {
    use g2g_core::runtime::parse_launch;
    use g2g_plugins::registry::default_registry;
    let reg = default_registry();
    for line in [
        "pipewiresrc target-object=mic0 format=F32LE samplerate=44100 channels=1 num-buffers=2 ! fakesink",
        "pipewirevideosrc target-object=cam0 format=NV12 width=1280 height=720 framerate=60 num-buffers=2 ! fakesink",
        "audiotestsrc num-buffers=2 ! pipewiresink target-object=spk0",
    ] {
        assert!(
            parse_launch(&reg, line).is_ok(),
            "launch line should build a graph: {line}"
        );
    }
    for line in [
        "pipewiresrc format=AAC ! fakesink",
        "pipewirevideosrc format=nonsense ! fakesink",
    ] {
        assert!(
            parse_launch(&reg, line).is_err(),
            "a format the element cannot open must fail the parse: {line}"
        );
    }
}

#[cfg(feature = "wgpu-sink")]
#[test]
fn wgpucompositor_shares_the_compositor_properties() {
    use g2g_core::{MultiInputElement, PropError};
    use g2g_plugins::compositor::CompositorPad;
    use g2g_plugins::wgpucompositor::WgpuCompositor;
    let mut e = WgpuCompositor::new(
        320,
        240,
        Vec::from([CompositorPad::at(0, 0), CompositorPad::at(0, 0)]),
    );
    for name in [
        "width",
        "height",
        "framerate",
        "background-color",
        "timed-output",
        "sink1-xpos",
        "sink7-height",
    ] {
        assert!(declares(e.properties(), name), "{name} is declared");
    }
    // RGBA8 only, so it declares no output format at all rather than a knob it
    // would have to reject every value of.
    assert!(!declares(e.properties(), "format"));
    assert_eq!(
        e.set_property("format", PropValue::Str("nv12".into())),
        Err(PropError::Unknown)
    );

    e.set_property("width", PropValue::Uint(1280)).unwrap();
    e.set_property("height", PropValue::Uint(720)).unwrap();
    assert_eq!(e.get_property("width"), Some(PropValue::Uint(1280)));
    assert_eq!(e.get_property("height"), Some(PropValue::Uint(720)));
    e.set_property("background-color", PropValue::Uint(0xFF102030))
        .unwrap();
    assert_eq!(
        e.get_property("background-color"),
        Some(PropValue::Uint(0xFF102030))
    );
    e.set_property("sink1-zorder", PropValue::Uint(2)).unwrap();
    e.set_property("sink1-alpha", PropValue::Uint(64)).unwrap();
    assert_eq!(e.get_property("sink1-zorder"), Some(PropValue::Uint(2)));
    assert_eq!(e.get_property("sink1-alpha"), Some(PropValue::Uint(64)));
    assert_eq!(
        e.set_property("sink3-xpos", PropValue::Int(0)),
        Err(PropError::Value),
        "two pads, so pad 3 does not exist"
    );
}

/// M956: the dmabuf-export io-mode is settable from a launch line, and a V4L2
/// streaming method the element does not implement is refused.
#[cfg(all(target_os = "linux", feature = "v4l2"))]
#[test]
fn v4l2src_io_mode() {
    use g2g_core::runtime::SourceLoop;
    use g2g_core::{MemoryDomainKind, PropError};
    use g2g_plugins::v4l2src::V4l2Src;
    let mut s = V4l2Src::new("/dev/video0");
    assert!(declares(s.properties(), "io-mode"));
    s.set_property("io-mode", PropValue::Str("dmabuf".into()))
        .unwrap();
    assert_eq!(
        s.get_property("io-mode"),
        Some(PropValue::Str("dmabuf".into()))
    );
    // the declared output domain follows the mode, so the solver and the DOT
    // dump show what a consumer will really be handed.
    assert_eq!(s.output_memory(), MemoryDomainKind::DmaBuf);
    assert_eq!(
        s.set_property("io-mode", PropValue::Str("userptr".into())),
        Err(PropError::Value),
        "userptr is not implemented, so it must not be accepted"
    );
}

/// M1038: the Android mic source's PCM shape and run length are settable from a
/// launch line, and a shape AAudio cannot open is refused.
#[cfg(all(target_os = "android", feature = "aaudio"))]
#[test]
fn aaudiosrc_shape_and_num_buffers() {
    use g2g_core::runtime::SourceLoop;
    use g2g_core::PropError;
    use g2g_plugins::aaudio::AAudioSrc;
    let mut s = AAudioSrc::new(48_000, 2, u64::MAX);
    for name in ["samplerate", "channels", "num-buffers"] {
        assert!(declares(s.properties(), name), "{name} must be declared");
    }
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(-1)));
    s.set_property("samplerate", PropValue::Uint(16_000))
        .unwrap();
    s.set_property("channels", PropValue::Uint(1)).unwrap();
    s.set_property("num-buffers", PropValue::Int(25)).unwrap();
    assert_eq!(s.get_property("samplerate"), Some(PropValue::Uint(16_000)));
    assert_eq!(s.get_property("channels"), Some(PropValue::Uint(1)));
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(25)));
    assert_eq!(
        s.set_property("samplerate", PropValue::Uint(0)),
        Err(PropError::Value),
        "a zero rate is not a stream AAudio can open"
    );
    assert_eq!(
        s.set_property("channels", PropValue::Uint(0)),
        Err(PropError::Value)
    );
}

/// M1038: the Android camera source's id, geometry and run length are settable
/// from a launch line, and the geometry reaches the produced caps.
#[cfg(all(target_os = "android", feature = "camera2"))]
#[test]
fn camera2src_device_geometry_and_num_buffers() {
    use g2g_core::runtime::{block_on, SourceLoop};
    use g2g_core::{Caps, Dim, Interlace, PropError, Rate, RawVideoFormat};
    use g2g_plugins::camera2src::Camera2Src;
    let mut s = Camera2Src::new(640, 480, u64::MAX);
    for name in ["device", "width", "height", "num-buffers"] {
        assert!(declares(s.properties(), name), "{name} must be declared");
    }
    s.set_property("device", PropValue::Str("1".into()))
        .unwrap();
    s.set_property("width", PropValue::Uint(1280)).unwrap();
    s.set_property("height", PropValue::Uint(720)).unwrap();
    s.set_property("num-buffers", PropValue::Int(90)).unwrap();
    assert_eq!(s.get_property("device"), Some(PropValue::Str("1".into())));
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(90)));
    // The geometry properties feed negotiation, not just the readback. Caps
    // come straight off the fields here, so this opens no camera.
    assert_eq!(
        block_on(s.intercept_caps()).expect("caps"),
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Any,
            interlace: Interlace::Any,
        }
    );
    // Empty returns to the first camera the manager reports.
    s.set_property("device", PropValue::Str(String::new()))
        .unwrap();
    assert_eq!(
        s.get_property("device"),
        Some(PropValue::Str(String::new()))
    );
    assert_eq!(
        s.set_property("width", PropValue::Uint(0)),
        Err(PropError::Value),
        "an ImageReader the camera can never fill is refused"
    );
}

/// M1038: the Core Audio capture source's PCM shape and run length join its
/// device UID as launch-line properties.
#[cfg(all(target_os = "macos", feature = "coreaudio"))]
#[test]
fn coreaudiosrc_shape_and_num_buffers() {
    use g2g_core::runtime::SourceLoop;
    use g2g_core::PropError;
    use g2g_plugins::coreaudio::CoreAudioSrc;
    let mut s = CoreAudioSrc::new(48_000, 2, u64::MAX);
    for name in ["device", "samplerate", "channels", "num-buffers"] {
        assert!(declares(s.properties(), name), "{name} must be declared");
    }
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(-1)));
    s.set_property("samplerate", PropValue::Uint(44_100))
        .unwrap();
    s.set_property("channels", PropValue::Uint(1)).unwrap();
    s.set_property("num-buffers", PropValue::Int(50)).unwrap();
    assert_eq!(s.get_property("samplerate"), Some(PropValue::Uint(44_100)));
    assert_eq!(s.get_property("channels"), Some(PropValue::Uint(1)));
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(50)));
    assert_eq!(
        s.set_property("samplerate", PropValue::Uint(0)),
        Err(PropError::Value),
        "a zero rate would divide by zero in the run loop"
    );
}

/// M1038: the AVFoundation camera source's run length is settable from a launch
/// line, alongside the device id and zero-copy switch it already took.
#[cfg(all(target_os = "macos", feature = "avfoundation"))]
#[test]
fn avfvideosrc_num_buffers() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::avf::AvfVideoSrc;
    let mut s = AvfVideoSrc::new(u64::MAX);
    assert!(declares(s.properties(), "num-buffers"));
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(-1)));
    s.set_property("num-buffers", PropValue::Int(300)).unwrap();
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(300)));
    s.set_property("num-buffers", PropValue::Int(-1)).unwrap();
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(-1)));
}

/// M1038: the AVFoundation mic source takes properties at all, matching its
/// camera sibling and the Core Audio capture source.
#[cfg(all(target_os = "macos", feature = "avfoundation"))]
#[test]
fn avfaudiosrc_device_shape_and_num_buffers() {
    use g2g_core::runtime::SourceLoop;
    use g2g_core::PropError;
    use g2g_plugins::avf::AvfAudioSrc;
    let mut s = AvfAudioSrc::new(48_000, 2, u64::MAX);
    for name in ["device", "samplerate", "channels", "num-buffers"] {
        assert!(declares(s.properties(), name), "{name} must be declared");
    }
    s.set_property("device", PropValue::Str("uid-1".into()))
        .unwrap();
    s.set_property("samplerate", PropValue::Uint(16_000))
        .unwrap();
    s.set_property("channels", PropValue::Uint(1)).unwrap();
    s.set_property("num-buffers", PropValue::Int(10)).unwrap();
    assert_eq!(
        s.get_property("device"),
        Some(PropValue::Str("uid-1".into()))
    );
    assert_eq!(s.get_property("samplerate"), Some(PropValue::Uint(16_000)));
    assert_eq!(s.get_property("channels"), Some(PropValue::Uint(1)));
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(10)));
    assert_eq!(
        s.set_property("channels", PropValue::Uint(0)),
        Err(PropError::Value),
        "a zero channel count would divide by zero in the run loop"
    );
}

/// M1038: the ScreenCaptureKit source's run length is settable from a launch
/// line. Its geometry stays display-derived, so it is not a property.
#[cfg(all(target_os = "macos", feature = "screencapture"))]
#[test]
fn screencapturesrc_num_buffers() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::sck::ScreenCaptureSrc;
    let mut s = ScreenCaptureSrc::new(u64::MAX);
    assert!(declares(s.properties(), "num-buffers"));
    assert!(
        !declares(s.properties(), "width"),
        "geometry comes from the display, so it must not look settable"
    );
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(-1)));
    s.set_property("num-buffers", PropValue::Int(120)).unwrap();
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(120)));
}

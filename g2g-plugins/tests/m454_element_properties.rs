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

/// The value a spec's declared `default` text parses to. A `gst-inspect` dump
/// prints it, so it has to be what a freshly built element actually reports.
fn declared_default(specs: &[PropertySpec], name: &str) -> PropValue {
    let spec = specs
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("`{name}` is declared"));
    let text = spec
        .default
        .unwrap_or_else(|| panic!("`{name}` declares a default"));
    spec.parse_value(text)
        .unwrap_or_else(|_| panic!("`{name}`'s default parses for its kind"))
}

/// The `(min, max)` a spec declares for an unsigned property, as numbers, so a
/// test can hold the declared text and the value the element enforces together.
fn declared_range(specs: &[PropertySpec], name: &str) -> (usize, usize) {
    let spec = specs
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("`{name}` is declared"));
    let (minimum, maximum) = spec
        .range
        .unwrap_or_else(|| panic!("`{name}` declares a range"));
    let parse = |text: &str| {
        text.parse::<usize>()
            .unwrap_or_else(|_| panic!("`{name}` declares `{text}` as a number"))
    };
    (parse(minimum), parse(maximum))
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
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
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

/// The block size both ADPCM halves are written / read with. It is the only
/// property they have, and out-of-range values are refused rather than clamped.
#[test]
fn adpcm_blockalign() {
    use g2g_plugins::adpcm::{AdpcmDec, AdpcmEnc};
    let mut encoder = AdpcmEnc::new();
    assert!(declares(encoder.properties(), "blockalign"));
    encoder
        .set_property("blockalign", PropValue::Uint(512))
        .unwrap();
    assert_eq!(
        encoder.get_property("blockalign"),
        Some(PropValue::Uint(512))
    );
    assert_eq!(encoder.block_align(), 512);
    assert!(encoder
        .set_property("blockalign", PropValue::Uint(16))
        .is_err());

    let mut decoder = AdpcmDec::new();
    assert!(declares(decoder.properties(), "blockalign"));
    decoder
        .set_property("blockalign", PropValue::Uint(2048))
        .unwrap();
    assert_eq!(
        decoder.get_property("blockalign"),
        Some(PropValue::Uint(2048))
    );
    assert_eq!(decoder.block_align(), 2048);
    assert!(decoder
        .set_property("blockalign", PropValue::Uint(65_536))
        .is_err());
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

/// M1098: `srtpenc` takes its key as hexadecimal digits, at any of the four
/// cipher lengths, and never reads it back. A length matching none of them is
/// refused rather than silently truncated.
#[cfg(feature = "srtp")]
#[test]
fn srtpenc_key_and_rtcp_encrypt() {
    use g2g_plugins::srtpenc::SrtpEnc;
    /// 16 key bytes then the 12-byte salt, and 32 key bytes then the same salt.
    const AES_128_KEY: &str = "000102030405060708090a0b0c0d0e0f517569642070726f2071756f";
    const AES_256_KEY: &str = "000102030405060708090a0b0c0d0e0f\
                               101112131415161718191a1b1c1d1e1f\
                               517569642070726f2071756f";

    let mut e = SrtpEnc::default();
    assert!(declares(e.properties(), "key"));
    for key in [AES_128_KEY, AES_256_KEY] {
        e.set_property("key", PropValue::Str(key.into())).unwrap();
    }
    // Reads back empty: the element never hands key material out.
    assert_eq!(e.get_property("key"), Some(PropValue::Str(String::new())));
    // One byte short of a cipher, and a non-hexadecimal digit.
    assert!(e
        .set_property("key", PropValue::Str(AES_128_KEY[..54].into()))
        .is_err());
    assert!(e
        .set_property("key", PropValue::Str(AES_128_KEY.replace("0a", "0z")))
        .is_err());

    assert_eq!(
        e.get_property("rtcp-encrypt"),
        Some(declared_default(e.properties(), "rtcp-encrypt"))
    );
    e.set_property("rtcp-encrypt", PropValue::Bool(false))
        .unwrap();
    assert_eq!(e.get_property("rtcp-encrypt"), Some(PropValue::Bool(false)));
}

/// M1101: the cipher and authentication pair of each flow, on both SRTP
/// elements. A fresh element reports the declared gst defaults; a key length
/// moves them; a value the key length cannot key is refused.
#[cfg(feature = "srtp")]
#[test]
fn srtp_elements_carry_a_cipher_and_authentication_per_flow() {
    use g2g_plugins::srtp::{AUTHENTICATION_VALUES, CIPHER_VALUES};
    use g2g_plugins::srtpdec::SrtpDec;
    use g2g_plugins::srtpenc::SrtpEnc;

    /// 16 key bytes then the 14-byte salt: the AES-128-ICM length gst defaults
    /// to, and the 12-byte-salt AES-128-GCM one beside it.
    const COUNTER_MODE_KEY: &str = "000102030405060708090a0b0c0d0e0f517569642070726f2071756f2121";
    const GCM_KEY: &str = "000102030405060708090a0b0c0d0e0f517569642070726f2071756f";
    const PROTECTION_PROPERTIES: [&str; 4] = ["rtp-cipher", "rtcp-cipher", "rtp-auth", "rtcp-auth"];

    fn check<E: AsyncElement>(label: &str, element: &mut E) {
        for name in PROTECTION_PROPERTIES {
            assert!(
                declares(element.properties(), name),
                "{label} must declare {name}"
            );
            assert_eq!(
                element.get_property(name),
                Some(declared_default(element.properties(), name)),
                "{label} {name} reports its declared default"
            );
        }
        // The declared choice list is the closed set the parser validates
        // against, so it has to name every value the element takes.
        for (name, values) in [
            ("rtp-cipher", CIPHER_VALUES),
            ("rtp-auth", AUTHENTICATION_VALUES),
        ] {
            let spec = element
                .properties()
                .iter()
                .find(|spec| spec.name == name)
                .expect("the property is declared");
            assert_eq!(spec.enum_values, Some(values), "{label} {name}");
        }

        element
            .set_property("key", PropValue::Str(GCM_KEY.into()))
            .unwrap_or_else(|_| panic!("{label} takes an AES-128-GCM key"));
        assert_eq!(
            element.get_property("rtp-cipher"),
            Some(PropValue::Str("aes-128-gcm".into())),
            "{label}: an unset cipher follows the key length"
        );
        assert_eq!(
            element.get_property("rtp-auth"),
            Some(PropValue::Str("null".into())),
            "{label}: an AEAD cipher carries its own tag"
        );
        // A separate transform beside an AEAD cipher, and a cipher this key
        // length cannot key.
        assert!(element
            .set_property("rtp-auth", PropValue::Str("hmac-sha1-80".into()))
            .is_err());
        assert!(element
            .set_property("rtp-cipher", PropValue::Str("aes-128-icm".into()))
            .is_err());

        element
            .set_property("key", PropValue::Str(COUNTER_MODE_KEY.into()))
            .unwrap_or_else(|_| panic!("{label} takes an AES-128-ICM key"));
        assert_eq!(
            element.get_property("rtcp-cipher"),
            Some(PropValue::Str("aes-128-icm".into())),
            "{label}"
        );
        assert_eq!(
            element.get_property("rtcp-auth"),
            Some(PropValue::Str("hmac-sha1-80".into())),
            "{label}"
        );
        // Only the flow that was set moves.
        element
            .set_property("rtcp-auth", PropValue::Str("hmac-sha1-32".into()))
            .unwrap_or_else(|_| panic!("{label} takes an RTCP tag length"));
        assert_eq!(
            element.get_property("rtp-auth"),
            Some(PropValue::Str("hmac-sha1-80".into())),
            "{label}: the RTP pair is untouched"
        );
        // A value the enum does not name.
        assert!(element
            .set_property("rtp-cipher", PropValue::Str("aes-192-icm".into()))
            .is_err());
        assert!(element
            .set_property("rtp-auth", PropValue::Str("hmac-sha256-80".into()))
            .is_err());
    }

    check("srtpenc", &mut SrtpEnc::default());
    check("srtpdec", &mut SrtpDec::default());
}

/// M1099: the session knobs `srtpenc` shares with gst. `stats` stays
/// programmatic (g2g has no structure-valued property kind), so it is unknown
/// here.
#[cfg(feature = "srtp")]
#[test]
fn srtpenc_replay_window_repeat_transmission_and_mki() {
    use g2g_plugins::srtp::{
        DEFAULT_REPLAY_WINDOW, MAXIMUM_MKI_LENGTH, MAXIMUM_REPLAY_WINDOW, MINIMUM_REPLAY_WINDOW,
    };
    use g2g_plugins::srtpenc::SrtpEnc;

    let mut e = SrtpEnc::default();
    let window = |packets: usize| PropValue::Uint(packets as u64);

    assert_eq!(
        e.get_property("replay-window-size"),
        Some(window(DEFAULT_REPLAY_WINDOW))
    );
    assert_eq!(
        declared_default(e.properties(), "replay-window-size"),
        window(DEFAULT_REPLAY_WINDOW),
        "the declared default is the one a fresh element reports"
    );
    assert_eq!(
        declared_range(e.properties(), "replay-window-size"),
        (MINIMUM_REPLAY_WINDOW, MAXIMUM_REPLAY_WINDOW),
        "the declared range is the one the element enforces"
    );
    for packets in [MINIMUM_REPLAY_WINDOW, MAXIMUM_REPLAY_WINDOW] {
        e.set_property("replay-window-size", window(packets))
            .unwrap();
        assert_eq!(e.get_property("replay-window-size"), Some(window(packets)));
    }
    for packets in [MINIMUM_REPLAY_WINDOW - 1, MAXIMUM_REPLAY_WINDOW + 1] {
        assert!(
            e.set_property("replay-window-size", window(packets))
                .is_err(),
            "{packets} is outside the declared range"
        );
    }

    assert_eq!(
        e.get_property("allow-repeat-tx"),
        Some(declared_default(e.properties(), "allow-repeat-tx"))
    );
    e.set_property("allow-repeat-tx", PropValue::Bool(true))
        .unwrap();
    assert_eq!(
        e.get_property("allow-repeat-tx"),
        Some(PropValue::Bool(true))
    );

    // The MKI is hexadecimal in and hexadecimal back out: it names the key, it
    // is not the key.
    assert_eq!(
        e.get_property("mki"),
        Some(declared_default(e.properties(), "mki"))
    );
    e.set_property("mki", PropValue::Str("00ff10".into()))
        .unwrap();
    assert_eq!(e.get_property("mki"), Some(PropValue::Str("00ff10".into())));
    e.set_property("mki", PropValue::Str(String::new()))
        .unwrap();
    assert_eq!(e.get_property("mki"), Some(PropValue::Str(String::new())));
    // An odd digit count, and one byte past libsrtp's SRTP_MAX_MKI_LEN.
    assert!(e.set_property("mki", PropValue::Str("00f".into())).is_err());
    assert!(e
        .set_property("mki", PropValue::Str("aa".repeat(MAXIMUM_MKI_LENGTH + 1)))
        .is_err());

    assert!(e
        .set_property("stats", PropValue::Str(String::new()))
        .is_err());
}

/// M1099: `srtpdec`'s replay window sizes the contexts it creates afterwards.
#[cfg(feature = "srtp")]
#[test]
fn srtpdec_replay_window() {
    use g2g_plugins::srtp::{DEFAULT_REPLAY_WINDOW, MAXIMUM_REPLAY_WINDOW, MINIMUM_REPLAY_WINDOW};
    use g2g_plugins::srtpdec::SrtpDec;

    let mut d = SrtpDec::default();
    let window = |packets: usize| PropValue::Uint(packets as u64);

    assert_eq!(
        d.get_property("replay-window-size"),
        Some(window(DEFAULT_REPLAY_WINDOW))
    );
    assert_eq!(
        declared_default(d.properties(), "replay-window-size"),
        window(DEFAULT_REPLAY_WINDOW)
    );
    assert_eq!(
        declared_range(d.properties(), "replay-window-size"),
        (MINIMUM_REPLAY_WINDOW, MAXIMUM_REPLAY_WINDOW)
    );
    for packets in [MINIMUM_REPLAY_WINDOW, MAXIMUM_REPLAY_WINDOW] {
        d.set_property("replay-window-size", window(packets))
            .unwrap();
        assert_eq!(d.get_property("replay-window-size"), Some(window(packets)));
    }
    for packets in [MINIMUM_REPLAY_WINDOW - 1, MAXIMUM_REPLAY_WINDOW + 1] {
        assert!(
            d.set_property("replay-window-size", window(packets))
                .is_err(),
            "{packets} is outside the declared range"
        );
    }
    assert!(d
        .set_property("stats", PropValue::Str(String::new()))
        .is_err());
}

/// M1098: `srtpdec` takes the same key plus the rollover counter a context
/// created later starts from.
#[cfg(feature = "srtp")]
#[test]
fn srtpdec_key_and_rollover_counter() {
    use g2g_plugins::srtpdec::SrtpDec;
    const AES_128_KEY: &str = "000102030405060708090a0b0c0d0e0f517569642070726f2071756f";

    let mut d = SrtpDec::default();
    assert!(declares(d.properties(), "key"));
    d.set_property("key", PropValue::Str(AES_128_KEY.into()))
        .unwrap();
    assert_eq!(d.get_property("key"), Some(PropValue::Str(String::new())));
    assert!(d
        .set_property("key", PropValue::Str(AES_128_KEY[..54].into()))
        .is_err());

    assert_eq!(
        d.get_property("roc"),
        Some(declared_default(d.properties(), "roc"))
    );
    d.set_property("roc", PropValue::Uint(u64::from(u32::MAX)))
        .unwrap();
    assert_eq!(
        d.get_property("roc"),
        Some(PropValue::Uint(u64::from(u32::MAX)))
    );
    assert!(d
        .set_property("roc", PropValue::Uint(u64::from(u32::MAX) + 1))
        .is_err());
}

/// M1068: `tcpserversrc` takes the gst `tcp` property set, and reports the port
/// it actually bound through the read-only `current-port`.
#[cfg(feature = "tcp")]
#[test]
fn tcpserversrc_endpoint_blocksize_and_format() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::tcp::TcpServerSrc;
    let mut s = TcpServerSrc::default();
    for name in [
        "host",
        "port",
        "current-port",
        "blocksize",
        "num-buffers",
        "bytestream-format",
    ] {
        assert!(declares(s.properties(), name), "{name} must be declared");
    }
    for name in [
        "host",
        "port",
        "blocksize",
        "num-buffers",
        "bytestream-format",
    ] {
        assert_eq!(
            s.get_property(name),
            Some(declared_default(s.properties(), name)),
            "a fresh element reports `{name}`'s declared default"
        );
    }
    assert!(
        !s.properties()
            .iter()
            .find(|spec| spec.name == "current-port")
            .unwrap()
            .flags
            .writable,
        "current-port is derived from the socket, so it must not look settable"
    );
    assert_eq!(
        s.get_property("current-port"),
        Some(PropValue::Uint(0)),
        "no port is bound before the pipeline configures"
    );

    s.set_property("host", PropValue::Str("127.0.0.1".into()))
        .unwrap();
    s.set_property("port", PropValue::Uint(0)).unwrap();
    s.set_property("blocksize", PropValue::Uint(8192)).unwrap();
    s.set_property("num-buffers", PropValue::Int(12)).unwrap();
    s.set_property("bytestream-format", PropValue::Str("matroska".into()))
        .unwrap();
    assert_eq!(
        s.get_property("host"),
        Some(PropValue::Str("127.0.0.1".into()))
    );
    assert_eq!(s.get_property("port"), Some(PropValue::Uint(0)));
    assert_eq!(s.get_property("blocksize"), Some(PropValue::Uint(8192)));
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(12)));
    assert_eq!(
        s.get_property("bytestream-format"),
        Some(PropValue::Str("matroska".into()))
    );

    assert!(
        s.set_property("port", PropValue::Uint(70_000)).is_err(),
        "rejects an out-of-range port"
    );
    assert!(
        s.set_property("blocksize", PropValue::Uint(0)).is_err(),
        "a zero read size would make no progress"
    );
    assert!(
        s.set_property("bytestream-format", PropValue::Str("mp3".into()))
            .is_err(),
        "rejects a container it cannot declare"
    );

    // The port only exists once something bound; `port=0` binds one the OS picks.
    let bound = s.bind().expect("bind an ephemeral port");
    assert_ne!(bound, 0);
    assert_eq!(
        s.get_property("current-port"),
        Some(PropValue::Uint(bound as u64))
    );
}

/// M1068: `tcpclientsrc` takes the same read properties, minus the bind-only
/// `current-port`.
#[cfg(feature = "tcp")]
#[test]
fn tcpclientsrc_endpoint_blocksize_and_format() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::tcp::TcpClientSrc;
    let mut s = TcpClientSrc::default();
    for name in [
        "host",
        "port",
        "blocksize",
        "num-buffers",
        "bytestream-format",
    ] {
        assert!(declares(s.properties(), name), "{name} must be declared");
        assert_eq!(
            s.get_property(name),
            Some(declared_default(s.properties(), name)),
            "a fresh element reports `{name}`'s declared default"
        );
    }
    assert!(
        !declares(s.properties(), "current-port"),
        "a client binds nothing, so it must not look like it reports a bound port"
    );

    s.set_property("host", PropValue::Str("example.test".into()))
        .unwrap();
    s.set_property("port", PropValue::Uint(5000)).unwrap();
    s.set_property("blocksize", PropValue::Uint(1500)).unwrap();
    s.set_property("num-buffers", PropValue::Int(3)).unwrap();
    s.set_property("bytestream-format", PropValue::Str("flv".into()))
        .unwrap();
    assert_eq!(
        s.get_property("host"),
        Some(PropValue::Str("example.test".into())),
        "a host name is kept as written, since it is resolved at connect time"
    );
    assert_eq!(s.get_property("port"), Some(PropValue::Uint(5000)));
    assert_eq!(s.get_property("blocksize"), Some(PropValue::Uint(1500)));
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(3)));
    assert_eq!(
        s.get_property("bytestream-format"),
        Some(PropValue::Str("flv".into()))
    );
}

/// M1068: `tcpserversink` takes the endpoint, reports its bound port, and lets
/// the head-of-stream wait be turned off.
#[cfg(feature = "tcp")]
#[test]
fn tcpserversink_endpoint_and_wait_for_connection() {
    use g2g_plugins::tcp::TcpServerSink;
    let mut e = TcpServerSink::default();
    for name in ["host", "port", "current-port", "wait-for-connection"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    for name in ["host", "port", "wait-for-connection"] {
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "a fresh element reports `{name}`'s declared default"
        );
    }
    assert_eq!(
        e.get_property("current-port"),
        Some(PropValue::Uint(0)),
        "no port is bound before the pipeline configures"
    );

    e.set_property("host", PropValue::Str("127.0.0.1".into()))
        .unwrap();
    e.set_property("port", PropValue::Uint(0)).unwrap();
    e.set_property("wait-for-connection", PropValue::Bool(false))
        .unwrap();
    assert_eq!(
        e.get_property("host"),
        Some(PropValue::Str("127.0.0.1".into()))
    );
    assert_eq!(
        e.get_property("wait-for-connection"),
        Some(PropValue::Bool(false))
    );
    assert!(
        e.set_property("port", PropValue::Uint(70_000)).is_err(),
        "rejects an out-of-range port"
    );

    let bound = e.bind().expect("bind an ephemeral port");
    assert_ne!(bound, 0);
    assert_eq!(
        e.get_property("current-port"),
        Some(PropValue::Uint(bound as u64))
    );
}

/// M1068: `tcpclientsink` takes the endpoint it dials.
#[cfg(feature = "tcp")]
#[test]
fn tcpclientsink_endpoint() {
    use g2g_plugins::tcp::TcpClientSink;
    let mut e = TcpClientSink::default();
    for name in ["host", "port"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "a fresh element reports `{name}`'s declared default"
        );
    }
    e.set_property("host", PropValue::Str("10.0.0.5".into()))
        .unwrap();
    e.set_property("port", PropValue::Uint(5000)).unwrap();
    assert_eq!(
        e.get_property("host"),
        Some(PropValue::Str("10.0.0.5".into()))
    );
    assert_eq!(e.get_property("port"), Some(PropValue::Uint(5000)));
    assert!(
        e.set_property("port", PropValue::Uint(70_000)).is_err(),
        "rejects an out-of-range port"
    );
}

/// M1081: `shmsink` takes the gst `shm` sink property set, and reports the
/// generated area name once the area exists.
#[cfg(all(unix, feature = "shm"))]
#[test]
fn shmsink_socket_area_and_backlog_properties() {
    use g2g_plugins::shm::ShmSink;
    let mut e = ShmSink::default();
    for name in [
        "socket-path",
        "shm-size",
        "perms",
        "wait-for-connection",
        "buffer-time",
        "shm-area-name",
    ] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    for name in ["shm-size", "perms", "wait-for-connection", "buffer-time"] {
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "a fresh element reports `{name}`'s declared default"
        );
    }
    assert_eq!(
        e.get_property("shm-area-name"),
        Some(PropValue::Str(String::new())),
        "no area exists before the pipeline configures"
    );

    let socket_path = std::env::temp_dir().join(format!("g2g_m454_shmsink_{}", std::process::id()));
    let _ = std::fs::remove_file(&socket_path);
    e.set_property(
        "socket-path",
        PropValue::Str(socket_path.to_string_lossy().into_owned()),
    )
    .unwrap();
    e.set_property("shm-size", PropValue::Uint(65_536)).unwrap();
    e.set_property("perms", PropValue::Uint(0o600)).unwrap();
    e.set_property("wait-for-connection", PropValue::Bool(false))
        .unwrap();
    e.set_property("buffer-time", PropValue::Int(2_000_000))
        .unwrap();
    assert_eq!(e.get_property("shm-size"), Some(PropValue::Uint(65_536)));
    assert_eq!(e.get_property("perms"), Some(PropValue::Uint(0o600)));
    assert_eq!(
        e.get_property("wait-for-connection"),
        Some(PropValue::Bool(false))
    );
    assert_eq!(
        e.get_property("buffer-time"),
        Some(PropValue::Int(2_000_000))
    );
    assert!(
        e.set_property("shm-size", PropValue::Uint(0)).is_err(),
        "rejects an empty area"
    );
    assert!(
        e.set_property("perms", PropValue::Uint(0o10000)).is_err(),
        "rejects mode bits outside 0 - 4095"
    );

    e.open().expect("the control socket and shm area open");
    let name = e.get_property("shm-area-name").expect("the area is named");
    assert!(
        matches!(&name, PropValue::Str(s) if s.starts_with("/shmpipe.")),
        "the area name is the one gst's writer would generate, got {name:?}"
    );
    let _ = std::fs::remove_file(&socket_path);
}

/// M1081: `shmsrc` takes the socket it reads and either way of declaring what
/// the bytes are.
#[cfg(all(unix, feature = "shm"))]
#[test]
fn shmsrc_socket_and_declared_caps() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::shm::ShmSrc;
    let mut e = ShmSrc::default();
    for name in [
        "socket-path",
        "bytestream-format",
        "caps",
        "num-buffers",
        "shm-area-name",
    ] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    for name in ["bytestream-format", "num-buffers"] {
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "a fresh element reports `{name}`'s declared default"
        );
    }

    e.set_property("socket-path", PropValue::Str("/tmp/g2g-m454-shm".into()))
        .unwrap();
    e.set_property("bytestream-format", PropValue::Str("matroska".into()))
        .unwrap();
    e.set_property("num-buffers", PropValue::Int(5)).unwrap();
    assert_eq!(
        e.get_property("socket-path"),
        Some(PropValue::Str("/tmp/g2g-m454-shm".into()))
    );
    assert_eq!(
        e.get_property("bytestream-format"),
        Some(PropValue::Str("matroska".into()))
    );
    assert_eq!(e.get_property("num-buffers"), Some(PropValue::Int(5)));

    let caps = "video/x-raw,format=i420,width=320,height=240,framerate=30/1";
    e.set_property("caps", PropValue::Str(caps.into())).unwrap();
    assert_eq!(e.get_property("caps"), Some(PropValue::Str(caps.into())));
    assert_eq!(
        e.configured_output_caps(),
        Some(g2g_core::Caps::RawVideo {
            format: g2g_core::RawVideoFormat::I420,
            width: g2g_core::Dim::Fixed(320),
            height: g2g_core::Dim::Fixed(240),
            framerate: g2g_core::Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN
        }),
        "the caps property is what the source declares"
    );
    assert!(
        e.set_property("caps", PropValue::Str("nonsense".into()))
            .is_err(),
        "rejects a caps string that does not parse"
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
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
    );
    // a format with no PCM stream behind it is rejected, never silently kept
    assert_eq!(
        e.set_property("format", PropValue::Str("AAC".into())),
        Err(PropError::Value)
    );
}

/// M1067: `imagefreeze` takes its output rate and length from properties, and
/// declares nothing it cannot honour.
#[test]
fn imagefreeze_framerate_and_num_buffers() {
    use g2g_plugins::imagefreeze::ImageFreeze;
    let mut e = ImageFreeze::new();
    assert!(declares(e.properties(), "framerate"));
    assert!(declares(e.properties(), "num-buffers"));
    assert_eq!(
        e.get_property("framerate"),
        Some(PropValue::Fraction(25, 1)),
        "25/1 with no property set"
    );
    e.set_property("framerate", PropValue::Fraction(30, 1))
        .unwrap();
    assert_eq!(
        e.get_property("framerate"),
        Some(PropValue::Fraction(30, 1))
    );
    assert_eq!(
        e.get_property("num-buffers"),
        Some(PropValue::Int(-1)),
        "gst's unlimited default"
    );
    e.set_property("num-buffers", PropValue::Int(10)).unwrap();
    assert_eq!(e.get_property("num-buffers"), Some(PropValue::Int(10)));
    // neither can be honoured in this model, so neither is offered.
    for name in ["allow-replace", "is-live"] {
        assert!(!declares(e.properties(), name), "{name}");
    }
}

/// M1066: `audiorate`'s two settings round-trip, and its four sample counters
/// read back but refuse a write.
#[test]
fn audiorate_tolerance_skip_to_first_and_counters() {
    use g2g_core::PropError;
    use g2g_plugins::audiorate::AudioRate;
    let mut e = AudioRate::new();
    assert!(declares(e.properties(), "tolerance"));
    assert!(declares(e.properties(), "skip-to-first"));
    // gst's default, in ns.
    assert_eq!(
        e.get_property("tolerance"),
        Some(PropValue::Uint(40_000_000))
    );
    e.set_property("tolerance", PropValue::Uint(0)).unwrap();
    assert_eq!(e.get_property("tolerance"), Some(PropValue::Uint(0)));
    e.set_property("skip-to-first", PropValue::Bool(true))
        .unwrap();
    assert_eq!(e.get_property("skip-to-first"), Some(PropValue::Bool(true)));
    for name in ["in", "out", "add", "drop"] {
        assert_eq!(e.get_property(name), Some(PropValue::Uint(0)), "{name}");
        assert_eq!(
            e.set_property(name, PropValue::Uint(1)),
            Err(PropError::Value),
            "{name} is a counter, not a knob"
        );
    }
    // gst's `silent` notify switch has no analog here, so it is not declared.
    assert!(!declares(e.properties(), "silent"));
}

/// M1075: `scaletempo`'s three window settings round-trip, and the current
/// playback rate reads back but refuses a write.
#[test]
fn scaletempo_windows_and_read_only_rate() {
    use g2g_core::PropError;
    use g2g_plugins::scaletempo::ScaleTempo;
    let mut e = ScaleTempo::new();
    // gst's defaults: 30 ms strides, 20 % overlap, 14 ms of search.
    assert_eq!(e.get_property("stride"), Some(PropValue::Uint(30)));
    assert_eq!(e.get_property("overlap"), Some(PropValue::Double(0.2)));
    assert_eq!(e.get_property("search"), Some(PropValue::Uint(14)));
    e.set_property("stride", PropValue::Uint(20)).unwrap();
    assert_eq!(e.get_property("stride"), Some(PropValue::Uint(20)));
    e.set_property("overlap", PropValue::Double(0.5)).unwrap();
    assert_eq!(e.get_property("overlap"), Some(PropValue::Double(0.5)));
    e.set_property("search", PropValue::Uint(0)).unwrap();
    assert_eq!(e.get_property("search"), Some(PropValue::Uint(0)));
    // the rate comes from the segment, so it reports but does not take one.
    assert_eq!(e.get_property("rate"), Some(PropValue::Double(1.0)));
    assert_eq!(
        e.set_property("rate", PropValue::Double(2.0)),
        Err(PropError::Value)
    );
    // gst's `mode` (fit-down) has no analog here, so it is not declared.
    assert!(!declares(e.properties(), "mode"));
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN
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

/// M1070: the valve's two knobs, both of which change what a closed valve does.
#[test]
fn valve_drop_and_drop_mode() {
    use g2g_plugins::valve::Valve;
    let mut e = Valve::new();
    assert!(declares(e.properties(), "drop"));
    assert!(declares(e.properties(), "drop-mode"));
    assert_eq!(
        e.get_property("drop"),
        Some(PropValue::Bool(false)),
        "gst valve's default"
    );
    e.set_property("drop", PropValue::Bool(true)).unwrap();
    assert_eq!(e.get_property("drop"), Some(PropValue::Bool(true)));
    assert_eq!(
        e.get_property("drop-mode"),
        Some(PropValue::Str("drop-all".into())),
        "gst valve's default"
    );
    e.set_property("drop-mode", PropValue::Str("forward-sticky-events".into()))
        .unwrap();
    assert_eq!(
        e.get_property("drop-mode"),
        Some(PropValue::Str("forward-sticky-events".into()))
    );
    assert!(
        e.set_property("drop-mode", PropValue::Str("transform-to-gap".into()))
            .is_err(),
        "rejects the gst mode g2g has no gap packet for"
    );
}

/// M1070: `fakesrc`'s run length, buffer size and fill.
#[test]
fn fakesrc_num_buffers_sizemax_and_filltype() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::fakesrc::FakeSrc;
    let mut e = FakeSrc::new();
    for name in ["num-buffers", "sizemax", "filltype"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(e.get_property("num-buffers"), Some(PropValue::Int(-1)));
    assert_eq!(
        e.get_property("sizemax"),
        Some(PropValue::Uint(4096)),
        "gst fakesrc's default"
    );
    assert_eq!(
        e.get_property("filltype"),
        Some(PropValue::Str("nothing".into())),
        "gst fakesrc's default"
    );
    e.set_property("num-buffers", PropValue::Int(20)).unwrap();
    e.set_property("sizemax", PropValue::Uint(100)).unwrap();
    e.set_property("filltype", PropValue::Str("pattern".into()))
        .unwrap();
    assert_eq!(e.get_property("num-buffers"), Some(PropValue::Int(20)));
    assert_eq!(e.get_property("sizemax"), Some(PropValue::Uint(100)));
    assert_eq!(
        e.get_property("filltype"),
        Some(PropValue::Str("pattern".into()))
    );
    assert!(
        e.set_property("filltype", PropValue::Str("pattern-span".into()))
            .is_err(),
        "rejects a fill it does not implement"
    );
}

/// M1070: the descriptor `fdsrc` reads and how much it reads at a time.
#[cfg(all(feature = "std", unix))]
#[test]
fn fdsrc_fd_blocksize_and_num_buffers() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::fd::FdSrc;
    let mut e = FdSrc::default();
    for name in ["fd", "blocksize", "num-buffers"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(
        e.get_property("fd"),
        Some(PropValue::Int(0)),
        "gst fdsrc reads stdin by default"
    );
    assert_eq!(
        e.get_property("blocksize"),
        Some(PropValue::Uint(4096)),
        "gst basesrc's default"
    );
    e.set_property("fd", PropValue::Int(7)).unwrap();
    e.set_property("blocksize", PropValue::Uint(512)).unwrap();
    e.set_property("num-buffers", PropValue::Int(3)).unwrap();
    assert_eq!(e.get_property("fd"), Some(PropValue::Int(7)));
    assert_eq!(e.get_property("blocksize"), Some(PropValue::Uint(512)));
    assert_eq!(e.get_property("num-buffers"), Some(PropValue::Int(3)));
    assert!(
        e.set_property("fd", PropValue::Int(-1)).is_err(),
        "rejects a descriptor that names nothing"
    );
    assert!(
        e.set_property("blocksize", PropValue::Uint(0)).is_err(),
        "a zero read would spin without progress"
    );
}

/// M1070: the descriptor `fdsink` writes to.
#[cfg(all(feature = "std", unix))]
#[test]
fn fdsink_fd() {
    use g2g_plugins::fd::FdSink;
    let mut e = FdSink::default();
    assert!(declares(e.properties(), "fd"));
    assert_eq!(
        e.get_property("fd"),
        Some(PropValue::Int(1)),
        "gst fdsink writes stdout by default"
    );
    e.set_property("fd", PropValue::Int(9)).unwrap();
    assert_eq!(e.get_property("fd"), Some(PropValue::Int(9)));
    assert!(
        e.set_property("fd", PropValue::Int(-1)).is_err(),
        "rejects a descriptor that names nothing"
    );
}

/// M1072: the interleaver declares the PCM shape of its pads and merged output,
/// since a fan-in names its output caps before the pads negotiate.
#[test]
fn interleave_format_and_rate() {
    use g2g_core::{AudioFormat, Caps, MultiInputElement};
    use g2g_plugins::interleave::Interleave;
    let mut element = Interleave::new(2);
    for name in ["format", "rate"] {
        assert!(
            declares(element.properties(), name),
            "{name} must be declared"
        );
    }
    assert_eq!(
        element.get_property("format"),
        Some(PropValue::Str("S16LE".into()))
    );
    element
        .set_property("format", PropValue::Str("F32LE".into()))
        .unwrap();
    element
        .set_property("rate", PropValue::Uint(44_100))
        .unwrap();
    assert_eq!(
        element.get_property("format"),
        Some(PropValue::Str("F32LE".into()))
    );
    assert_eq!(element.get_property("rate"), Some(PropValue::Uint(44_100)));
    assert_eq!(
        element.output_caps().unwrap(),
        Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 2,
            sample_rate: 44_100,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        },
        "the properties are what the merged output declares"
    );
}

/// M1072: the splitter's fan-out form declares the same shape, one channel per
/// port, and its single-output form picks a channel instead.
#[test]
fn deinterleave_format_rate_and_channel() {
    use g2g_core::{AudioFormat, Caps, MultiOutputElement};
    use g2g_plugins::deinterleave::{Deinterleave, DeinterleaveN};
    let mut fanout = DeinterleaveN::new(2);
    for name in ["format", "rate"] {
        assert!(
            declares(fanout.properties(), name),
            "{name} must be declared"
        );
    }
    fanout
        .set_property("format", PropValue::Str("S32LE".into()))
        .unwrap();
    fanout
        .set_property("rate", PropValue::Uint(16_000))
        .unwrap();
    assert_eq!(
        fanout.get_property("format"),
        Some(PropValue::Str("S32LE".into()))
    );
    assert_eq!(fanout.get_property("rate"), Some(PropValue::Uint(16_000)));
    assert_eq!(
        fanout.port_output_caps(1),
        Some(Caps::Audio {
            format: AudioFormat::PcmS32Le,
            channels: 1,
            sample_rate: 16_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }),
        "every port carries one channel of the declared shape"
    );

    let mut picker = Deinterleave::new();
    assert!(declares(AsyncElement::properties(&picker), "channel"));
    assert_eq!(picker.get_property("channel"), Some(PropValue::Uint(0)));
    picker.set_property("channel", PropValue::Uint(3)).unwrap();
    assert_eq!(picker.get_property("channel"), Some(PropValue::Uint(3)));
}

/// M1077: the watchdog's stall deadline.
#[cfg(feature = "std")]
#[test]
fn watchdog_timeout() {
    use g2g_plugins::watchdog::Watchdog;
    let mut e = Watchdog::new();
    assert!(declares(e.properties(), "timeout"));
    assert_eq!(
        e.get_property("timeout"),
        Some(PropValue::Uint(1000)),
        "gst watchdog's default"
    );
    e.set_property("timeout", PropValue::Uint(2500)).unwrap();
    assert_eq!(e.get_property("timeout"), Some(PropValue::Uint(2500)));
    assert!(
        e.set_property("timeout", PropValue::Uint(u64::from(u32::MAX)))
            .is_err(),
        "past gst's 32-bit signed bound"
    );
}

/// M1077: what `capssetter` writes, and the two switches deciding how.
#[test]
fn capssetter_caps_join_and_replace() {
    use g2g_plugins::capssetter::CapsSetter;
    let mut e = CapsSetter::new();
    for name in ["caps", "join", "replace"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(
        e.get_property("join"),
        Some(PropValue::Bool(true)),
        "gst capssetter's default"
    );
    assert_eq!(
        e.get_property("replace"),
        Some(PropValue::Bool(false)),
        "gst capssetter's default"
    );
    let caps = "video/x-raw,framerate=60/1";
    e.set_property("caps", PropValue::Str(caps.into())).unwrap();
    assert_eq!(e.get_property("caps"), Some(PropValue::Str(caps.into())));
    e.set_property("join", PropValue::Bool(false)).unwrap();
    e.set_property("replace", PropValue::Bool(true)).unwrap();
    assert_eq!(e.get_property("join"), Some(PropValue::Bool(false)));
    assert_eq!(e.get_property("replace"), Some(PropValue::Bool(true)));
    assert!(
        e.set_property("caps", PropValue::Str("nonsense".into()))
            .is_err(),
        "an unparseable description is refused"
    );
}

/// M1077: the tag list `taginject` posts.
#[test]
fn taginject_tags() {
    use g2g_core::Tag;
    use g2g_plugins::taginject::TagInject;
    let mut e = TagInject::new();
    assert!(declares(e.properties(), "tags"));
    assert_eq!(e.get_property("tags"), None, "gst taginject's default");
    let tags = "title=\"A Title\",artist=Someone";
    e.set_property("tags", PropValue::Str(tags.into())).unwrap();
    assert_eq!(e.get_property("tags"), Some(PropValue::Str(tags.into())));
    assert_eq!(
        e.tags().tags(),
        [Tag::Title("A Title".into()), Tag::Artist("Someone".into())],
        "a quoted value keeps its space and the comma stays a separator"
    );
    assert!(
        e.set_property("tags", PropValue::Str("title".into()))
            .is_err(),
        "a pair with no `=` is refused"
    );
}

/// M1077: the chunk bounds and seed `rndbuffersize` cuts with.
#[test]
fn rndbuffersize_min_max_and_seed() {
    use g2g_plugins::rndbuffersize::RndBufferSize;
    let mut e = RndBufferSize::new();
    for name in ["min", "max", "seed"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(
        e.get_property("min"),
        Some(PropValue::Uint(1)),
        "gst rndbuffersize's default"
    );
    assert_eq!(
        e.get_property("max"),
        Some(PropValue::Uint(8192)),
        "gst rndbuffersize's default"
    );
    assert_eq!(
        e.get_property("seed"),
        Some(PropValue::Uint(0)),
        "gst rndbuffersize's default"
    );
    e.set_property("min", PropValue::Uint(100)).unwrap();
    e.set_property("max", PropValue::Uint(500)).unwrap();
    e.set_property("seed", PropValue::Uint(7)).unwrap();
    assert_eq!(e.get_property("min"), Some(PropValue::Uint(100)));
    assert_eq!(e.get_property("max"), Some(PropValue::Uint(500)));
    assert_eq!(e.get_property("seed"), Some(PropValue::Uint(7)));
    assert!(
        e.set_property("max", PropValue::Uint(0)).is_err(),
        "gst's `max` starts at one byte"
    );
}

/// M1080: both multipart elements take the boundary from a launch line, and
/// refuse one no sender could have written.
#[test]
fn multipart_boundaries() {
    use g2g_core::PropError;
    use g2g_plugins::multipart::{MultipartDemux, MultipartMux};

    let mut demux = MultipartDemux::new();
    assert!(declares(demux.properties(), "boundary"));
    assert_eq!(
        demux.get_property("boundary"),
        Some(PropValue::Str(String::new())),
        "empty until set: the stream's first line declares it"
    );
    demux
        .set_property("boundary", PropValue::Str("ffmpeg".into()))
        .unwrap();
    assert_eq!(
        demux.get_property("boundary"),
        Some(PropValue::Str("ffmpeg".into()))
    );

    let mut mux = MultipartMux::new();
    assert!(declares(mux.properties(), "boundary"));
    assert_eq!(
        mux.get_property("boundary"),
        Some(PropValue::Str("ThisRandomString".into())),
        "gst's default"
    );
    mux.set_property("boundary", PropValue::Str("ffmpeg".into()))
        .unwrap();
    assert_eq!(
        mux.get_property("boundary"),
        Some(PropValue::Str("ffmpeg".into()))
    );
    assert_eq!(
        mux.set_property("boundary", PropValue::Str("with a space".into())),
        Err(PropError::Value)
    );
    // gst's `single-stream` only decides when no-more-pads fires, and this
    // element has one static output pad.
    assert!(!declares(demux.properties(), "single-stream"));
}

/// M1079: the raw-datagram knobs on `udpsrc`, and the RTP-only ones it refuses
/// once a container is declared.
#[cfg(feature = "udp-ingress")]
#[test]
fn udpsrc_bytestream_and_multicast() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::udpsrc::UdpSrc;
    let mut s = UdpSrc::new("0.0.0.0:5004".parse().unwrap());
    for name in [
        "bytestream-format",
        "multicast-group",
        "auto-multicast",
        "current-port",
    ] {
        assert!(declares(s.properties(), name), "declares {name}");
    }
    assert_eq!(
        s.get_property("bytestream-format"),
        Some(PropValue::Str(String::new())),
        "unset means the RTP path"
    );
    s.set_property("multicast-group", PropValue::Str("239.1.2.3".into()))
        .unwrap();
    assert_eq!(
        s.get_property("address"),
        Some(PropValue::Str("239.1.2.3".into())),
        "multicast-group is the same setting as address"
    );
    s.set_property("auto-multicast", PropValue::Bool(false))
        .unwrap();
    assert_eq!(
        s.get_property("auto-multicast"),
        Some(PropValue::Bool(false))
    );
    assert_eq!(
        s.get_property("current-port"),
        Some(PropValue::Uint(0)),
        "nothing bound yet"
    );
    assert!(
        s.set_property("current-port", PropValue::Uint(5000))
            .is_err(),
        "current-port is read-only"
    );

    s.set_property("bytestream-format", PropValue::Str("mpegts".into()))
        .unwrap();
    assert_eq!(
        s.get_property("bytestream-format"),
        Some(PropValue::Str("mpegts".into()))
    );
    assert!(
        s.set_property("jitter-latency", PropValue::Uint(20))
            .is_err(),
        "an RTP-only property is refused in raw mode, not ignored"
    );
    assert!(
        s.set_property("bytestream-format", PropValue::Str("nonsense".into()))
            .is_err(),
        "an unknown container is refused"
    );
}

/// M1079: the `multiudpsink` fan-out and multicast knobs on `udpsink`.
#[cfg(feature = "udp-egress")]
#[test]
fn udpsink_clients_and_multicast() {
    use g2g_plugins::udpsink::UdpSink;
    let mut e = UdpSink::new("127.0.0.1:5004".parse().unwrap());
    for name in ["clients", "auto-multicast", "ttl-mc"] {
        assert!(declares(e.properties(), name), "declares {name}");
    }
    assert_eq!(
        e.get_property("clients"),
        Some(PropValue::Str(String::new())),
        "no fan-out until it is asked for"
    );
    e.set_property(
        "clients",
        PropValue::Str("127.0.0.1:5000,127.0.0.1:5002".into()),
    )
    .unwrap();
    assert_eq!(
        e.get_property("clients"),
        Some(PropValue::Str("127.0.0.1:5000,127.0.0.1:5002".into()))
    );
    assert!(
        e.set_property("clients", PropValue::Str("127.0.0.1".into()))
            .is_err(),
        "an entry without a port is refused"
    );
    e.set_property("ttl-mc", PropValue::Uint(8)).unwrap();
    assert_eq!(e.get_property("ttl-mc"), Some(PropValue::Uint(8)));
    assert!(
        e.set_property("ttl-mc", PropValue::Uint(256)).is_err(),
        "the multicast TTL is a byte"
    );
    e.set_property("auto-multicast", PropValue::Bool(false))
        .unwrap();
    assert_eq!(
        e.get_property("auto-multicast"),
        Some(PropValue::Bool(false))
    );
}

/// M1078: `audiodynamic`'s two enums and its two curve numbers.
#[test]
fn audiodynamic_mode_characteristics_threshold_and_ratio() {
    use g2g_plugins::audiodynamic::AudioDynamic;
    let mut e = AudioDynamic::new();
    for name in ["mode", "characteristics", "threshold", "ratio"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(
        e.get_property("mode"),
        Some(PropValue::Str("compressor".into()))
    );
    assert_eq!(
        e.get_property("characteristics"),
        Some(PropValue::Str("hard-knee".into()))
    );
    e.set_property("mode", PropValue::Str("expander".into()))
        .unwrap();
    assert_eq!(
        e.get_property("mode"),
        Some(PropValue::Str("expander".into()))
    );
    e.set_property("characteristics", PropValue::Str("soft-knee".into()))
        .unwrap();
    assert_eq!(
        e.get_property("characteristics"),
        Some(PropValue::Str("soft-knee".into()))
    );
    e.set_property("threshold", PropValue::Double(0.5)).unwrap();
    assert_eq!(e.get_property("threshold"), Some(PropValue::Double(0.5)));
    e.set_property("ratio", PropValue::Double(0.25)).unwrap();
    assert_eq!(e.get_property("ratio"), Some(PropValue::Double(0.25)));
    // the threshold is a fraction of full scale, so 2 is out of range.
    assert!(e.set_property("threshold", PropValue::Double(2.0)).is_err());
    assert!(e
        .set_property("mode", PropValue::Str("limiter".into()))
        .is_err());
}

/// M1078: `audioinvert`'s single blend knob.
#[test]
fn audioinvert_degree() {
    use g2g_plugins::audioinvert::AudioInvert;
    let mut e = AudioInvert::new();
    assert!(declares(e.properties(), "degree"));
    assert_eq!(e.get_property("degree"), Some(PropValue::Double(0.0)));
    e.set_property("degree", PropValue::Double(1.0)).unwrap();
    assert_eq!(e.get_property("degree"), Some(PropValue::Double(1.0)));
    assert!(e.set_property("degree", PropValue::Double(-0.1)).is_err());
}

/// M1078: `audiokaraoke`'s two levels and its mono filter shape.
#[test]
fn audiokaraoke_levels_and_filter() {
    use g2g_plugins::audiokaraoke::AudioKaraoke;
    let mut e = AudioKaraoke::new();
    for name in ["level", "mono-level", "filter-band", "filter-width"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(e.get_property("level"), Some(PropValue::Double(1.0)));
    assert_eq!(
        e.get_property("filter-band"),
        Some(PropValue::Double(220.0))
    );
    assert_eq!(
        e.get_property("filter-width"),
        Some(PropValue::Double(100.0))
    );
    e.set_property("level", PropValue::Double(0.75)).unwrap();
    assert_eq!(e.get_property("level"), Some(PropValue::Double(0.75)));
    e.set_property("mono-level", PropValue::Double(0.25))
        .unwrap();
    assert_eq!(e.get_property("mono-level"), Some(PropValue::Double(0.25)));
    e.set_property("filter-band", PropValue::Double(180.0))
        .unwrap();
    assert_eq!(
        e.get_property("filter-band"),
        Some(PropValue::Double(180.0))
    );
    e.set_property("filter-width", PropValue::Double(50.0))
        .unwrap();
    assert_eq!(
        e.get_property("filter-width"),
        Some(PropValue::Double(50.0))
    );
    // the band tops out at 441 Hz, as in the reference.
    assert!(e
        .set_property("filter-band", PropValue::Double(1000.0))
        .is_err());
}

/// M1078: `audiowsinclimit`'s kernel knobs, including the odd-length rounding.
#[test]
fn audiowsinclimit_mode_cutoff_length_and_window() {
    use g2g_plugins::audiowsinclimit::AudioWsincLimit;
    let mut e = AudioWsincLimit::new();
    for name in ["mode", "cutoff", "length", "window"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(
        e.get_property("mode"),
        Some(PropValue::Str("low-pass".into()))
    );
    assert_eq!(e.get_property("length"), Some(PropValue::Int(101)));
    assert_eq!(
        e.get_property("window"),
        Some(PropValue::Str("hamming".into()))
    );
    e.set_property("mode", PropValue::Str("high-pass".into()))
        .unwrap();
    assert_eq!(
        e.get_property("mode"),
        Some(PropValue::Str("high-pass".into()))
    );
    e.set_property("cutoff", PropValue::Double(1000.0)).unwrap();
    assert_eq!(e.get_property("cutoff"), Some(PropValue::Double(1000.0)));
    // an even length is rounded up so the kernel has a centre tap.
    e.set_property("length", PropValue::Int(200)).unwrap();
    assert_eq!(e.get_property("length"), Some(PropValue::Int(201)));
    for window in ["blackman", "gaussian", "cosine", "hann", "hamming"] {
        e.set_property("window", PropValue::Str(window.into()))
            .unwrap();
        assert_eq!(
            e.get_property("window"),
            Some(PropValue::Str(window.into()))
        );
    }
    assert!(e
        .set_property("window", PropValue::Str("kaiser".into()))
        .is_err());
}

/// M1078: `audiowsincband`'s kernel knobs.
#[test]
fn audiowsincband_mode_band_length_and_window() {
    use g2g_plugins::audiowsincband::AudioWsincBand;
    let mut e = AudioWsincBand::new();
    for name in [
        "mode",
        "lower-frequency",
        "upper-frequency",
        "length",
        "window",
    ] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(
        e.get_property("mode"),
        Some(PropValue::Str("band-pass".into()))
    );
    e.set_property("mode", PropValue::Str("band-reject".into()))
        .unwrap();
    assert_eq!(
        e.get_property("mode"),
        Some(PropValue::Str("band-reject".into()))
    );
    e.set_property("lower-frequency", PropValue::Double(1000.0))
        .unwrap();
    e.set_property("upper-frequency", PropValue::Double(4000.0))
        .unwrap();
    assert_eq!(
        e.get_property("lower-frequency"),
        Some(PropValue::Double(1000.0))
    );
    assert_eq!(
        e.get_property("upper-frequency"),
        Some(PropValue::Double(4000.0))
    );
    e.set_property("length", PropValue::Int(64)).unwrap();
    assert_eq!(e.get_property("length"), Some(PropValue::Int(65)));
    e.set_property("window", PropValue::Str("blackman".into()))
        .unwrap();
    assert_eq!(
        e.get_property("window"),
        Some(PropValue::Str("blackman".into()))
    );
}

/// M1078: `audiocheblimit`'s pole count, ripple and filter type.
#[test]
fn audiocheblimit_mode_cutoff_poles_ripple_and_type() {
    use g2g_plugins::audiocheblimit::AudioChebLimit;
    let mut e = AudioChebLimit::new();
    for name in ["mode", "cutoff", "poles", "ripple", "type"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(e.get_property("poles"), Some(PropValue::Int(4)));
    assert_eq!(e.get_property("ripple"), Some(PropValue::Double(0.25)));
    assert_eq!(e.get_property("type"), Some(PropValue::Int(1)));
    e.set_property("mode", PropValue::Str("high-pass".into()))
        .unwrap();
    assert_eq!(
        e.get_property("mode"),
        Some(PropValue::Str("high-pass".into()))
    );
    e.set_property("cutoff", PropValue::Double(2000.0)).unwrap();
    assert_eq!(e.get_property("cutoff"), Some(PropValue::Double(2000.0)));
    // poles come in conjugate pairs, so an odd count rounds up.
    e.set_property("poles", PropValue::Int(7)).unwrap();
    assert_eq!(e.get_property("poles"), Some(PropValue::Int(8)));
    e.set_property("ripple", PropValue::Double(0.5)).unwrap();
    assert_eq!(e.get_property("ripple"), Some(PropValue::Double(0.5)));
    e.set_property("type", PropValue::Int(2)).unwrap();
    assert_eq!(e.get_property("type"), Some(PropValue::Int(2)));
    assert!(e.set_property("type", PropValue::Int(3)).is_err());
}

/// M1078: `audiochebband`'s band, pole count, ripple and filter type.
#[test]
fn audiochebband_mode_band_poles_ripple_and_type() {
    use g2g_plugins::audiochebband::AudioChebBand;
    let mut e = AudioChebBand::new();
    for name in [
        "mode",
        "lower-frequency",
        "upper-frequency",
        "poles",
        "ripple",
        "type",
    ] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(
        e.get_property("mode"),
        Some(PropValue::Str("band-pass".into()))
    );
    e.set_property("mode", PropValue::Str("band-reject".into()))
        .unwrap();
    assert_eq!(
        e.get_property("mode"),
        Some(PropValue::Str("band-reject".into()))
    );
    e.set_property("lower-frequency", PropValue::Double(1000.0))
        .unwrap();
    e.set_property("upper-frequency", PropValue::Double(4000.0))
        .unwrap();
    assert_eq!(
        e.get_property("lower-frequency"),
        Some(PropValue::Double(1000.0))
    );
    assert_eq!(
        e.get_property("upper-frequency"),
        Some(PropValue::Double(4000.0))
    );
    // each section is fourth order, so the pole count rounds up to a multiple
    // of four.
    e.set_property("poles", PropValue::Int(6)).unwrap();
    assert_eq!(e.get_property("poles"), Some(PropValue::Int(8)));
    e.set_property("ripple", PropValue::Double(1.0)).unwrap();
    assert_eq!(e.get_property("ripple"), Some(PropValue::Double(1.0)));
    e.set_property("type", PropValue::Int(2)).unwrap();
    assert_eq!(e.get_property("type"), Some(PropValue::Int(2)));
    assert!(e.set_property("poles", PropValue::Int(64)).is_err());
}

/// M1085: `audiochannelmix`'s four stereo gains.
#[test]
fn audiochannelmix_gains() {
    use g2g_plugins::audiochannelmix::AudioChannelMix;
    let mut e = AudioChannelMix::new();
    for name in [
        "left-to-left",
        "left-to-right",
        "right-to-left",
        "right-to-right",
    ] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    // the identity mix is the default: each channel to itself.
    assert_eq!(e.get_property("left-to-left"), Some(PropValue::Double(1.0)));
    assert_eq!(
        e.get_property("left-to-right"),
        Some(PropValue::Double(0.0))
    );
    e.set_property("left-to-right", PropValue::Double(0.5))
        .unwrap();
    assert_eq!(
        e.get_property("left-to-right"),
        Some(PropValue::Double(0.5))
    );
    e.set_property("right-to-left", PropValue::Double(-0.5))
        .unwrap();
    assert_eq!(
        e.get_property("right-to-left"),
        Some(PropValue::Double(-0.5))
    );
}

/// M1085: `audiomixmatrix`'s mode, channel counts and matrix string.
#[test]
fn audiomixmatrix_mode_channels_and_matrix() {
    use g2g_plugins::audiomixmatrix::AudioMixMatrix;
    let mut e = AudioMixMatrix::new();
    for name in ["mode", "in-channels", "out-channels", "matrix"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(
        e.get_property("mode"),
        Some(PropValue::Str("manual".into()))
    );
    e.set_property("mode", PropValue::Str("first-channels".into()))
        .unwrap();
    assert_eq!(
        e.get_property("mode"),
        Some(PropValue::Str("first-channels".into()))
    );
    assert!(e
        .set_property("mode", PropValue::Str("none".into()))
        .is_err());
    e.set_property("in-channels", PropValue::Uint(6)).unwrap();
    e.set_property("out-channels", PropValue::Uint(2)).unwrap();
    assert_eq!(e.get_property("in-channels"), Some(PropValue::Uint(6)));
    assert_eq!(e.get_property("out-channels"), Some(PropValue::Uint(2)));
    // more channels than the reference's 64 is out of range.
    assert!(e.set_property("in-channels", PropValue::Uint(65)).is_err());
    // rows are separated by ';', gains within a row by ','.
    e.set_property("matrix", PropValue::Str("1,0;0,1".into()))
        .unwrap();
    assert_eq!(
        e.get_property("matrix"),
        Some(PropValue::Str("1,0;0,1".into()))
    );
    assert!(e
        .set_property("matrix", PropValue::Str("1,0;0,x".into()))
        .is_err());
}

/// M1085: `stereo`'s active switch and widening amount.
#[test]
fn stereo_active_and_widening() {
    use g2g_plugins::stereo::Stereo;
    let mut e = Stereo::new();
    for name in ["active", "stereo"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(e.get_property("active"), Some(PropValue::Bool(true)));
    e.set_property("active", PropValue::Bool(false)).unwrap();
    assert_eq!(e.get_property("active"), Some(PropValue::Bool(false)));
    e.set_property("stereo", PropValue::Double(0.5)).unwrap();
    assert_eq!(e.get_property("stereo"), Some(PropValue::Double(0.5)));
    assert!(e.set_property("stereo", PropValue::Double(1.5)).is_err());
}

/// M1085: `audiofirfilter`'s kernel list and latency.
#[test]
fn audiofirfilter_kernel_and_latency() {
    use g2g_plugins::audiofirfilter::AudioFirFilter;
    let mut e = AudioFirFilter::new();
    for name in ["kernel", "latency"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    // the default kernel is the unit impulse, a pass-through.
    assert_eq!(e.get_property("kernel"), Some(PropValue::Str("1".into())));
    assert_eq!(e.get_property("latency"), Some(PropValue::Uint(0)));
    e.set_property("kernel", PropValue::Str("0.25,0.5,0.25".into()))
        .unwrap();
    assert_eq!(
        e.get_property("kernel"),
        Some(PropValue::Str("0.25,0.5,0.25".into()))
    );
    e.set_property("latency", PropValue::Uint(1)).unwrap();
    assert_eq!(e.get_property("latency"), Some(PropValue::Uint(1)));
    assert!(e
        .set_property("kernel", PropValue::Str("0.25,nope".into()))
        .is_err());
}

/// M1085: `audioiirfilter`'s two coefficient lists.
#[test]
fn audioiirfilter_coefficients() {
    use g2g_plugins::audioiirfilter::AudioIirFilter;
    let mut e = AudioIirFilter::new();
    for name in ["a", "b"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    // unity on both sides is the default, a pass-through.
    assert_eq!(e.get_property("a"), Some(PropValue::Str("1".into())));
    assert_eq!(e.get_property("b"), Some(PropValue::Str("1".into())));
    e.set_property("a", PropValue::Str("1,-0.5".into()))
        .unwrap();
    e.set_property("b", PropValue::Str("0.5".into())).unwrap();
    assert_eq!(e.get_property("a"), Some(PropValue::Str("1,-0.5".into())));
    assert_eq!(e.get_property("b"), Some(PropValue::Str("0.5".into())));
    // a leading zero in the denominator would leave the recurrence undefined.
    assert!(e.set_property("a", PropValue::Str("0,1".into())).is_err());
    assert!(e.set_property("b", PropValue::Str("".into())).is_err());
}

/// M1085: `removesilence`'s detector and removal settings.
#[test]
fn removesilence_detector_and_removal() {
    use g2g_plugins::removesilence::RemoveSilence;
    let mut e = RemoveSilence::new();
    for name in [
        "remove",
        "hysteresis",
        "threshold",
        "squash",
        "silent",
        "minimum-silence-buffers",
        "minimum-silence-time",
    ] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(e.get_property("remove"), Some(PropValue::Bool(false)));
    assert_eq!(e.get_property("hysteresis"), Some(PropValue::Uint(480)));
    assert_eq!(e.get_property("threshold"), Some(PropValue::Int(-60)));
    assert_eq!(e.get_property("silent"), Some(PropValue::Bool(true)));
    e.set_property("remove", PropValue::Bool(true)).unwrap();
    assert_eq!(e.get_property("remove"), Some(PropValue::Bool(true)));
    e.set_property("hysteresis", PropValue::Uint(960)).unwrap();
    assert_eq!(e.get_property("hysteresis"), Some(PropValue::Uint(960)));
    e.set_property("threshold", PropValue::Int(-40)).unwrap();
    assert_eq!(e.get_property("threshold"), Some(PropValue::Int(-40)));
    e.set_property("squash", PropValue::Bool(true)).unwrap();
    assert_eq!(e.get_property("squash"), Some(PropValue::Bool(true)));
    e.set_property("minimum-silence-buffers", PropValue::Uint(4))
        .unwrap();
    assert_eq!(
        e.get_property("minimum-silence-buffers"),
        Some(PropValue::Uint(4))
    );
    e.set_property("minimum-silence-time", PropValue::Uint(100_000_000))
        .unwrap();
    assert_eq!(
        e.get_property("minimum-silence-time"),
        Some(PropValue::Uint(100_000_000))
    );
    // the reference's bounds: hysteresis of zero, and both minimums capped.
    assert!(e.set_property("hysteresis", PropValue::Uint(0)).is_err());
    assert!(e.set_property("threshold", PropValue::Int(80)).is_err());
    assert!(e
        .set_property("minimum-silence-buffers", PropValue::Uint(10_001))
        .is_err());
}

/// M1085: `audiobuffersplit`'s framing and discontinuity settings.
#[test]
fn audiobuffersplit_framing_and_discont() {
    use g2g_plugins::audiobuffersplit::AudioBufferSplit;
    let mut e = AudioBufferSplit::new();
    for name in [
        "output-buffer-duration",
        "output-buffer-size",
        "strict-buffer-size",
        "gapless",
        "alignment-threshold",
        "discont-wait",
        "max-silence-time",
    ] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(
        e.get_property("output-buffer-duration"),
        Some(PropValue::Fraction(1, 50))
    );
    assert_eq!(
        e.get_property("alignment-threshold"),
        Some(PropValue::Uint(40_000_000))
    );
    assert_eq!(
        e.get_property("discont-wait"),
        Some(PropValue::Uint(1_000_000_000))
    );
    e.set_property("output-buffer-duration", PropValue::Fraction(1, 100))
        .unwrap();
    assert_eq!(
        e.get_property("output-buffer-duration"),
        Some(PropValue::Fraction(1, 100))
    );
    assert!(e
        .set_property("output-buffer-duration", PropValue::Fraction(0, 100))
        .is_err());
    e.set_property("output-buffer-size", PropValue::Uint(1024))
        .unwrap();
    assert_eq!(
        e.get_property("output-buffer-size"),
        Some(PropValue::Uint(1024))
    );
    e.set_property("strict-buffer-size", PropValue::Bool(true))
        .unwrap();
    assert_eq!(
        e.get_property("strict-buffer-size"),
        Some(PropValue::Bool(true))
    );
    e.set_property("gapless", PropValue::Bool(true)).unwrap();
    assert_eq!(e.get_property("gapless"), Some(PropValue::Bool(true)));
    e.set_property("alignment-threshold", PropValue::Uint(0))
        .unwrap();
    assert_eq!(
        e.get_property("alignment-threshold"),
        Some(PropValue::Uint(0))
    );
    e.set_property("discont-wait", PropValue::Uint(0)).unwrap();
    assert_eq!(e.get_property("discont-wait"), Some(PropValue::Uint(0)));
    e.set_property("max-silence-time", PropValue::Uint(500_000_000))
        .unwrap();
    assert_eq!(
        e.get_property("max-silence-time"),
        Some(PropValue::Uint(500_000_000))
    );
}

/// M1085: `speed`'s playback-rate factor.
#[test]
fn speed_factor() {
    use g2g_plugins::speed::Speed;
    let mut e = Speed::new();
    assert!(declares(e.properties(), "speed"));
    assert_eq!(e.get_property("speed"), Some(PropValue::Double(1.0)));
    e.set_property("speed", PropValue::Double(2.0)).unwrap();
    assert_eq!(e.get_property("speed"), Some(PropValue::Double(2.0)));
    // the reference's range is 0.1 to 40.
    assert!(e.set_property("speed", PropValue::Double(0.05)).is_err());
    assert!(e.set_property("speed", PropValue::Double(41.0)).is_err());
}

/// M1084: `coloreffects`'s preset selects one of the colour tables.
#[test]
fn coloreffects_preset() {
    use g2g_plugins::coloreffects::ColorEffects;
    let mut e = ColorEffects::new();
    assert!(declares(e.properties(), "preset"));
    assert_eq!(
        e.get_property("preset"),
        Some(declared_default(e.properties(), "preset"))
    );
    e.set_property("preset", PropValue::Str("sepia".into()))
        .unwrap();
    assert_eq!(
        e.get_property("preset"),
        Some(PropValue::Str("sepia".into()))
    );
    assert!(e
        .set_property("preset", PropValue::Str("mauve".into()))
        .is_err());
}

/// M1084: `chromahold`'s target colour and hue tolerance.
#[test]
fn chromahold_target_and_tolerance() {
    use g2g_plugins::chromahold::ChromaHold;
    let mut e = ChromaHold::new();
    for name in ["target-r", "target-g", "target-b", "tolerance"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "{name} reports its declared default"
        );
    }
    e.set_property("target-r", PropValue::Uint(0)).unwrap();
    e.set_property("target-g", PropValue::Uint(255)).unwrap();
    e.set_property("tolerance", PropValue::Uint(45)).unwrap();
    assert_eq!(e.get_property("target-r"), Some(PropValue::Uint(0)));
    assert_eq!(e.get_property("target-g"), Some(PropValue::Uint(255)));
    assert_eq!(e.get_property("tolerance"), Some(PropValue::Uint(45)));
    // out of range on both kinds of knob.
    assert!(e.set_property("target-b", PropValue::Uint(256)).is_err());
    assert!(e.set_property("tolerance", PropValue::Uint(181)).is_err());
}

/// M1084: `zebrastripe`'s exposure threshold.
#[test]
fn zebrastripe_threshold() {
    use g2g_plugins::zebrastripe::ZebraStripe;
    let mut e = ZebraStripe::new();
    assert!(declares(e.properties(), "threshold"));
    assert_eq!(
        e.get_property("threshold"),
        Some(declared_default(e.properties(), "threshold"))
    );
    e.set_property("threshold", PropValue::Int(50)).unwrap();
    assert_eq!(e.get_property("threshold"), Some(PropValue::Int(50)));
    assert!(e.set_property("threshold", PropValue::Int(101)).is_err());
}

/// M1084: `gaussianblur`'s sigma, negative for sharpen.
#[test]
fn gaussianblur_sigma() {
    use g2g_plugins::gaussianblur::GaussianBlur;
    let mut e = GaussianBlur::new();
    assert!(declares(e.properties(), "sigma"));
    assert_eq!(
        e.get_property("sigma"),
        Some(declared_default(e.properties(), "sigma"))
    );
    e.set_property("sigma", PropValue::Double(-3.5)).unwrap();
    assert_eq!(e.get_property("sigma"), Some(PropValue::Double(-3.5)));
    assert!(e.set_property("sigma", PropValue::Double(21.0)).is_err());
}

/// M1084: `videodiff`'s motion threshold, GStreamer's internal constant made
/// settable.
#[test]
fn videodiff_threshold() {
    use g2g_plugins::videodiff::VideoDiff;
    let mut e = VideoDiff::new();
    assert!(declares(e.properties(), "threshold"));
    assert_eq!(
        e.get_property("threshold"),
        Some(declared_default(e.properties(), "threshold"))
    );
    e.set_property("threshold", PropValue::Int(40)).unwrap();
    assert_eq!(e.get_property("threshold"), Some(PropValue::Int(40)));
    assert!(e.set_property("threshold", PropValue::Int(256)).is_err());
}

/// M1084: `videomedian`'s window size and chroma switch.
#[test]
fn videomedian_filtersize_and_lum_only() {
    use g2g_plugins::videomedian::VideoMedian;
    let mut e = VideoMedian::new();
    for name in ["filtersize", "lum-only"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "{name} reports its declared default"
        );
    }
    e.set_property("filtersize", PropValue::Int(9)).unwrap();
    e.set_property("lum-only", PropValue::Bool(false)).unwrap();
    assert_eq!(e.get_property("filtersize"), Some(PropValue::Int(9)));
    assert_eq!(e.get_property("lum-only"), Some(PropValue::Bool(false)));
    // only the two window sizes GStreamer offers.
    assert!(e.set_property("filtersize", PropValue::Int(7)).is_err());
}

/// M1084: `smooth`'s activity switch, tolerance, reach and chroma switch.
#[test]
fn smooth_active_tolerance_filter_size_and_luma_only() {
    use g2g_plugins::smooth::Smooth;
    let mut e = Smooth::new();
    for name in ["active", "tolerance", "filter-size", "luma-only"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "{name} reports its declared default"
        );
    }
    e.set_property("active", PropValue::Bool(false)).unwrap();
    e.set_property("tolerance", PropValue::Int(24)).unwrap();
    e.set_property("filter-size", PropValue::Int(5)).unwrap();
    e.set_property("luma-only", PropValue::Bool(false)).unwrap();
    assert_eq!(e.get_property("active"), Some(PropValue::Bool(false)));
    assert_eq!(e.get_property("tolerance"), Some(PropValue::Int(24)));
    assert_eq!(e.get_property("filter-size"), Some(PropValue::Int(5)));
    assert_eq!(e.get_property("luma-only"), Some(PropValue::Bool(false)));
}

/// M1084: `aspectratiocrop`'s target ratio.
#[test]
fn aspectratiocrop_aspect_ratio() {
    use g2g_plugins::aspectratiocrop::AspectRatioCrop;
    let mut e = AspectRatioCrop::new();
    assert!(declares(e.properties(), "aspect-ratio"));
    assert_eq!(
        e.get_property("aspect-ratio"),
        Some(declared_default(e.properties(), "aspect-ratio"))
    );
    e.set_property("aspect-ratio", PropValue::Fraction(16, 9))
        .unwrap();
    assert_eq!(
        e.get_property("aspect-ratio"),
        Some(PropValue::Fraction(16, 9))
    );
    // a zero denominator names no ratio at all.
    assert!(e
        .set_property("aspect-ratio", PropValue::Fraction(16, 0))
        .is_err());
}

/// M1083: `breakmydata`'s corruption knobs, gst's names, bounds and defaults.
#[test]
fn breakmydata_probability_seed_set_to_and_skip() {
    use g2g_plugins::breakmydata::BreakMyData;
    let mut e = BreakMyData::new();
    for name in ["probability", "seed", "set-to", "skip"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(e.get_property("probability"), Some(PropValue::Double(0.0)));
    assert_eq!(e.get_property("set-to"), Some(PropValue::Int(-1)));
    e.set_property("probability", PropValue::Double(0.25))
        .unwrap();
    assert_eq!(e.get_property("probability"), Some(PropValue::Double(0.25)));
    e.set_property("seed", PropValue::Uint(7)).unwrap();
    assert_eq!(e.get_property("seed"), Some(PropValue::Uint(7)));
    e.set_property("set-to", PropValue::Int(0xaa)).unwrap();
    assert_eq!(e.get_property("set-to"), Some(PropValue::Int(0xaa)));
    e.set_property("skip", PropValue::Uint(64)).unwrap();
    assert_eq!(e.get_property("skip"), Some(PropValue::Uint(64)));
    assert!(e
        .set_property("probability", PropValue::Double(1.5))
        .is_err());
    assert!(e.set_property("set-to", PropValue::Int(256)).is_err());
}

/// M1083: `chopmydata`'s three sizes, which gst bounds away from zero.
#[test]
fn chopmydata_min_max_and_step_size() {
    use g2g_plugins::chopmydata::ChopMyData;
    let mut e = ChopMyData::new();
    for name in ["min-size", "max-size", "step-size"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(e.get_property("min-size"), Some(PropValue::Int(1)));
    assert_eq!(e.get_property("max-size"), Some(PropValue::Int(4096)));
    assert_eq!(e.get_property("step-size"), Some(PropValue::Int(1)));
    for (name, value) in [("min-size", 8), ("max-size", 32), ("step-size", 8)] {
        e.set_property(name, PropValue::Int(value)).unwrap();
        assert_eq!(e.get_property(name), Some(PropValue::Int(value)));
    }
    assert!(e.set_property("step-size", PropValue::Int(0)).is_err());
}

/// M1083: `errorignore`'s four switches and the failure it reports instead.
#[test]
fn errorignore_switches_and_convert_to() {
    use g2g_plugins::errorignore::ErrorIgnore;
    let mut e = ErrorIgnore::new();
    for name in [
        "ignore-error",
        "ignore-notlinked",
        "ignore-notnegotiated",
        "ignore-eos",
        "convert-to",
    ] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    // gst ignores an error and a failed negotiation by default, nothing else.
    assert_eq!(e.get_property("ignore-error"), Some(PropValue::Bool(true)));
    assert_eq!(
        e.get_property("ignore-notnegotiated"),
        Some(PropValue::Bool(true))
    );
    assert_eq!(
        e.get_property("ignore-notlinked"),
        Some(PropValue::Bool(false))
    );
    assert_eq!(e.get_property("ignore-eos"), Some(PropValue::Bool(false)));
    assert_eq!(
        e.get_property("convert-to"),
        Some(PropValue::Str("not-linked".into()))
    );
    for name in [
        "ignore-error",
        "ignore-notlinked",
        "ignore-notnegotiated",
        "ignore-eos",
    ] {
        e.set_property(name, PropValue::Bool(true)).unwrap();
        assert_eq!(e.get_property(name), Some(PropValue::Bool(true)));
        e.set_property(name, PropValue::Bool(false)).unwrap();
        assert_eq!(e.get_property(name), Some(PropValue::Bool(false)));
    }
    e.set_property("convert-to", PropValue::Str("ok".into()))
        .unwrap();
    assert_eq!(
        e.get_property("convert-to"),
        Some(PropValue::Str("ok".into()))
    );
    assert!(e
        .set_property("convert-to", PropValue::Str("custom-success".into()))
        .is_err());
}

/// M1083: `checksumsink`'s hash, the one knob of gst's that has a g2g meaning.
#[test]
fn checksumsink_hash() {
    use g2g_plugins::checksumsink::ChecksumSink;
    let mut e = ChecksumSink::new();
    assert!(declares(e.properties(), "hash"));
    assert_eq!(
        e.get_property("hash"),
        Some(PropValue::Str("sha1".into())),
        "gst's default"
    );
    for hash in ["md5", "sha1", "sha256", "sha512"] {
        e.set_property("hash", PropValue::Str(hash.into())).unwrap();
        assert_eq!(e.get_property("hash"), Some(PropValue::Str(hash.into())));
    }
    assert!(e
        .set_property("hash", PropValue::Str("crc32".into()))
        .is_err());
}

/// M1083: the media-typed fake sinks' per-buffer line, off by default.
#[test]
fn fakevideosink_and_fakeaudiosink_silent_and_last_message() {
    use g2g_plugins::fakemediasink::{FakeAudioSink, FakeVideoSink};
    let mut video = FakeVideoSink::new();
    let mut audio = FakeAudioSink::new();
    for name in ["silent", "last-message"] {
        assert!(
            declares(video.properties(), name),
            "{name} must be declared"
        );
        assert!(
            declares(audio.properties(), name),
            "{name} must be declared"
        );
    }
    assert_eq!(video.get_property("silent"), Some(PropValue::Bool(true)));
    assert_eq!(audio.get_property("silent"), Some(PropValue::Bool(true)));
    assert_eq!(
        video.get_property("last-message"),
        Some(PropValue::Str(String::new()))
    );
    video
        .set_property("silent", PropValue::Bool(false))
        .unwrap();
    assert_eq!(video.get_property("silent"), Some(PropValue::Bool(false)));
    audio
        .set_property("silent", PropValue::Bool(false))
        .unwrap();
    assert_eq!(audio.get_property("silent"), Some(PropValue::Bool(false)));
    assert!(video
        .set_property("last-message", PropValue::Str("mine".into()))
        .is_err());
}

/// M1083: `fpsdisplaysink`'s child, its report interval and the read-only
/// counters an application polls.
#[cfg(feature = "std")]
#[test]
fn fpsdisplaysink_child_interval_and_counters() {
    use g2g_plugins::fpsdisplaysink::FpsDisplaySink;
    let mut e = FpsDisplaySink::new();
    for name in [
        "video-sink",
        "fps-update-interval",
        "silent",
        "frames-rendered",
        "frames-dropped",
        "max-fps",
        "min-fps",
        "last-message",
    ] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(
        e.get_property("video-sink"),
        Some(PropValue::Str("autovideosink".into()))
    );
    assert_eq!(
        e.get_property("fps-update-interval"),
        Some(PropValue::Int(500))
    );
    assert_eq!(e.get_property("max-fps"), Some(PropValue::Double(-1.0)));
    assert_eq!(e.get_property("min-fps"), Some(PropValue::Double(-1.0)));
    assert_eq!(e.get_property("frames-rendered"), Some(PropValue::Uint(0)));
    assert_eq!(e.get_property("frames-dropped"), Some(PropValue::Uint(0)));
    e.set_property("video-sink", PropValue::Str("fakesink".into()))
        .unwrap();
    assert_eq!(
        e.get_property("video-sink"),
        Some(PropValue::Str("fakesink".into()))
    );
    e.set_property("fps-update-interval", PropValue::Int(200))
        .unwrap();
    assert_eq!(
        e.get_property("fps-update-interval"),
        Some(PropValue::Int(200))
    );
    e.set_property("silent", PropValue::Bool(true)).unwrap();
    assert_eq!(e.get_property("silent"), Some(PropValue::Bool(true)));
    assert!(e
        .set_property("fps-update-interval", PropValue::Int(0))
        .is_err());
    assert!(e
        .set_property("frames-rendered", PropValue::Uint(9))
        .is_err());
}

/// M1086: the shape `rawvideoparse` cuts a headerless dump into, and the
/// per-frame stride of a dump whose frames are spaced apart.
#[test]
fn rawvideoparse_shape_and_frame_size() {
    use g2g_plugins::rawvideoparse::RawVideoParse;
    let mut e = RawVideoParse::new();
    for name in ["format", "width", "height", "framerate", "frame-size"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "{name} reports its declared default"
        );
    }
    e.set_property("format", PropValue::Str("NV12".into()))
        .unwrap();
    assert_eq!(
        e.get_property("format"),
        Some(PropValue::Str("NV12".into()))
    );
    e.set_property("width", PropValue::Uint(640)).unwrap();
    e.set_property("height", PropValue::Uint(480)).unwrap();
    assert_eq!(e.get_property("width"), Some(PropValue::Uint(640)));
    assert_eq!(e.get_property("height"), Some(PropValue::Uint(480)));
    e.set_property("framerate", PropValue::Fraction(30000, 1001))
        .unwrap();
    assert_eq!(
        e.get_property("framerate"),
        Some(PropValue::Fraction(30000, 1001))
    );
    e.set_property("frame-size", PropValue::Uint(460_800))
        .unwrap();
    assert_eq!(e.get_property("frame-size"), Some(PropValue::Uint(460_800)));
    // M1093: the padded-plane lists, as comma-separated byte counts.
    for name in ["plane-strides", "plane-offsets"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
        assert_eq!(
            e.get_property(name),
            Some(PropValue::Str(String::new())),
            "{name} is empty until a padded layout is declared"
        );
        e.set_property(name, PropValue::Str("704,352,352".into()))
            .unwrap();
        assert_eq!(
            e.get_property(name),
            Some(PropValue::Str("704,352,352".into()))
        );
        assert!(e
            .set_property(name, PropValue::Str("704,nope".into()))
            .is_err());
    }
    assert!(e
        .set_property("framerate", PropValue::Fraction(25, 0))
        .is_err());
}

/// M1086: the sample shape `rawaudioparse` cuts a headerless dump into. The
/// `format` / `pcm-format` pair is gst's: the first names the encoding, the
/// second the PCM sample layout.
#[test]
fn rawaudioparse_format_rate_and_channels() {
    use g2g_plugins::rawaudioparse::RawAudioParse;
    let mut e = RawAudioParse::new();
    for name in ["format", "pcm-format", "sample-rate", "num-channels"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "{name} reports its declared default"
        );
    }
    e.set_property("pcm-format", PropValue::Str("F32LE".into()))
        .unwrap();
    assert_eq!(
        e.get_property("pcm-format"),
        Some(PropValue::Str("F32LE".into()))
    );
    e.set_property("sample-rate", PropValue::Uint(48_000))
        .unwrap();
    e.set_property("num-channels", PropValue::Uint(1)).unwrap();
    assert_eq!(e.get_property("sample-rate"), Some(PropValue::Uint(48_000)));
    assert_eq!(e.get_property("num-channels"), Some(PropValue::Uint(1)));
    e.set_property("format", PropValue::Str("alaw".into()))
        .unwrap();
    assert_eq!(
        e.get_property("format"),
        Some(PropValue::Str("alaw".into()))
    );
    assert!(e
        .set_property("format", PropValue::Str("adpcm".into()))
        .is_err());
}

/// M1088: `splitfilesrc`'s pattern, read size and container override.
#[cfg(feature = "std")]
#[test]
fn splitfilesrc_location_blocksize_and_format() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::splitfilesrc::SplitFileSrc;
    let mut e = SplitFileSrc::new("clip.ts.part*");
    for name in ["location", "blocksize", "bytestream-format"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(
        e.get_property("blocksize"),
        Some(declared_default(e.properties(), "blocksize"))
    );
    assert_eq!(
        e.get_property("location"),
        Some(PropValue::Str("clip.ts.part*".into()))
    );
    e.set_property("blocksize", PropValue::Uint(4096)).unwrap();
    assert_eq!(e.get_property("blocksize"), Some(PropValue::Uint(4096)));
    e.set_property("bytestream-format", PropValue::Str("mpegts".into()))
        .unwrap();
    assert_eq!(
        e.get_property("bytestream-format"),
        Some(PropValue::Str("mpegts".into()))
    );
    assert!(e.set_property("blocksize", PropValue::Uint(0)).is_err());
    assert!(e
        .set_property("bytestream-format", PropValue::Str("nosuch".into()))
        .is_err());
}

/// M1088: `dataurisrc`'s URI and push size.
#[cfg(feature = "std")]
#[test]
fn dataurisrc_uri_and_blocksize() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::dataurisrc::DataUriSrc;
    let mut e = DataUriSrc::new("data:,hello");
    for name in ["uri", "blocksize"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(
        e.get_property("blocksize"),
        Some(declared_default(e.properties(), "blocksize"))
    );
    e.set_property("uri", PropValue::Str("data:text/plain;base64,aGk=".into()))
        .unwrap();
    assert_eq!(
        e.get_property("uri"),
        Some(PropValue::Str("data:text/plain;base64,aGk=".into()))
    );
    e.set_property("blocksize", PropValue::Uint(128)).unwrap();
    assert_eq!(e.get_property("blocksize"), Some(PropValue::Uint(128)));
    assert!(e.set_property("blocksize", PropValue::Uint(0)).is_err());
}

/// M1088: the framerate that turns `multifilesrc` into `imagesequencesrc`,
/// unstamped until it is set.
#[cfg(feature = "std")]
#[test]
fn multifilesrc_framerate() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::multifilesrc::MultiFileSrc;
    let mut e = MultiFileSrc::new("img%05d.jpg");
    assert!(declares(e.properties(), "framerate"));
    assert_eq!(
        e.get_property("framerate"),
        Some(PropValue::Fraction(0, 1)),
        "a plain multifilesrc leaves the files unstamped"
    );
    e.set_property("framerate", PropValue::Fraction(30, 1))
        .unwrap();
    assert_eq!(
        e.get_property("framerate"),
        Some(PropValue::Fraction(30, 1))
    );
    assert!(e
        .set_property("framerate", PropValue::Fraction(30, 0))
        .is_err());
}

/// M1094: the audio tag writers take their tags from a `tags=` property, in the
/// gst taglist syntax `taginject` accepts, because a `TagList` reaches an
/// application only on the bus and an element cannot read the bus. `id3v2mux`
/// adds the ID3v2 version to write and the ID3v1 trailer switch.
#[test]
fn m1094_audio_tag_writer_properties() {
    use g2g_core::Tag;
    use g2g_plugins::apev2mux::ApeV2Mux;
    use g2g_plugins::flactag::FlacTag;
    use g2g_plugins::id3v2mux::Id3V2Mux;
    use g2g_plugins::vorbistag::VorbisTag;
    use g2g_plugins::xingmux::XingMux;

    /// A quoted value holding a comma, the case the syntax needs quotes for.
    const TAGS: &str = "title=\"A, Title\",artist=Someone";
    let expected = [Tag::Title("A, Title".into()), Tag::Artist("Someone".into())];

    /// The `tags` half every writer shares: declared, unset until written, and
    /// rejecting a pair with no `=`.
    fn check_tags_property(element: &mut impl AsyncElement) {
        assert!(
            declares(element.properties(), "tags"),
            "`tags` must be declared"
        );
        assert_eq!(
            element.get_property("tags"),
            None,
            "unset until it is written"
        );
        element
            .set_property("tags", PropValue::Str(TAGS.into()))
            .unwrap();
        assert_eq!(
            element.get_property("tags"),
            Some(PropValue::Str(TAGS.into()))
        );
        assert!(element
            .set_property("tags", PropValue::Str("no equals sign".into()))
            .is_err());
    }

    let mut id3 = Id3V2Mux::new();
    check_tags_property(&mut id3);
    assert_eq!(id3.tags().tags(), expected, "the property feeds the writer");

    let mut ape = ApeV2Mux::new();
    check_tags_property(&mut ape);
    assert_eq!(ape.tags().tags(), expected);

    let mut vorbis = VorbisTag::new();
    check_tags_property(&mut vorbis);
    assert_eq!(vorbis.tags().tags(), expected);

    let mut flac = FlacTag::new();
    check_tags_property(&mut flac);
    assert_eq!(flac.tags().tags(), expected);

    let mut id3 = Id3V2Mux::new();
    for name in ["v2-version", "write-v1"] {
        assert!(declares(id3.properties(), name), "{name} must be declared");
        assert_eq!(
            id3.get_property(name),
            Some(declared_default(id3.properties(), name))
        );
    }
    id3.set_property("v2-version", PropValue::Uint(4)).unwrap();
    assert_eq!(id3.get_property("v2-version"), Some(PropValue::Uint(4)));
    // ID3v2.2 frame ids are three bytes, which nothing here reads or writes.
    assert!(id3.set_property("v2-version", PropValue::Uint(2)).is_err());
    id3.set_property("write-v1", PropValue::Bool(true)).unwrap();
    assert_eq!(id3.get_property("write-v1"), Some(PropValue::Bool(true)));

    // `xingmux` takes no tags: a Xing header is a seek table, not metadata, and
    // gst's element has no properties either.
    assert!(XingMux::new().properties().is_empty());
}

/// M1095: the legacy video parsers. `mpeg4videoparse` takes gst's
/// `config-interval`; all three report the sample aspect their sequence header
/// signalled through a read-only `pixel-aspect-ratio`, since caps carry no field
/// for it.
#[test]
fn legacy_video_parsers_properties() {
    use g2g_plugins::mpeg4videoparse::Mpeg4VideoParse;
    use g2g_plugins::mpegvideoparse::MpegVideoParse;
    use g2g_plugins::vc1parse::Vc1Parse;

    let mut mpeg4 = Mpeg4VideoParse::new();
    assert!(declares(mpeg4.properties(), "config-interval"));
    assert_eq!(
        mpeg4.get_property("config-interval"),
        Some(declared_default(mpeg4.properties(), "config-interval"))
    );
    mpeg4
        .set_property("config-interval", PropValue::Int(-1))
        .unwrap();
    assert_eq!(
        mpeg4.get_property("config-interval"),
        Some(PropValue::Int(-1))
    );
    assert!(mpeg4
        .set_property("config-interval", PropValue::Int(-2))
        .is_err());
    assert!(mpeg4
        .set_property("config-interval", PropValue::Int(3601))
        .is_err());

    // The other two apply no configuration re-insertion, so they declare none.
    assert!(!declares(
        MpegVideoParse::new().properties(),
        "config-interval"
    ));
    assert!(!declares(Vc1Parse::new().properties(), "config-interval"));

    for properties in [
        MpegVideoParse::new().properties(),
        Mpeg4VideoParse::new().properties(),
        Vc1Parse::new().properties(),
    ] {
        assert!(declares(properties, "pixel-aspect-ratio"));
    }
    let mut mpeg2 = MpegVideoParse::new();
    assert_eq!(
        mpeg2.get_property("pixel-aspect-ratio"),
        Some(PropValue::Fraction(0, 1)),
        "unknown until a sequence header has been parsed"
    );
    assert!(mpeg2
        .set_property("pixel-aspect-ratio", PropValue::Fraction(1, 1))
        .is_err());
}

/// M1096: the subtitle writers' cue-window offsets, the caption converter's
/// layout pair, and the combiner's scheduling / meta-merge knobs.
#[test]
fn srtenc_and_webvttenc_cue_window_offsets() {
    use g2g_plugins::srtenc::SrtEnc;
    use g2g_plugins::webvttenc::WebVttEnc;
    let mut srt = SrtEnc::new();
    let mut vtt = WebVttEnc::new();
    for name in ["timestamp", "duration"] {
        assert!(declares(srt.properties(), name), "srtenc {name}");
        assert!(declares(vtt.properties(), name), "webvttenc {name}");
        assert_eq!(
            srt.get_property(name),
            Some(declared_default(srt.properties(), name))
        );
    }
    srt.set_property("timestamp", PropValue::Int(-500_000_000))
        .unwrap();
    assert_eq!(
        srt.get_property("timestamp"),
        Some(PropValue::Int(-500_000_000))
    );
    vtt.set_property("duration", PropValue::Int(250_000_000))
        .unwrap();
    assert_eq!(
        vtt.get_property("duration"),
        Some(PropValue::Int(250_000_000))
    );
    assert!(srt.set_property("timestamp", PropValue::Uint(1)).is_err());
}

/// M1096: `ccconverter`'s input / output layouts, the line-21 field and the
/// frame rate a written CDP declares.
#[test]
fn ccconverter_layout_pair_field_and_framerate() {
    use g2g_plugins::ccconverter::CcConverter;
    let mut e = CcConverter::new();
    for name in ["in-format", "out-format", "field", "framerate"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "{name} reports its declared default"
        );
    }
    e.set_property("in-format", PropValue::Str("s334-1a".into()))
        .unwrap();
    assert_eq!(
        e.get_property("in-format"),
        Some(PropValue::Str("s334-1a".into()))
    );
    e.set_property("out-format", PropValue::Str("raw".into()))
        .unwrap();
    assert_eq!(
        e.get_property("out-format"),
        Some(PropValue::Str("raw".into()))
    );
    e.set_property("field", PropValue::Uint(1)).unwrap();
    assert_eq!(e.get_property("field"), Some(PropValue::Uint(1)));
    e.set_property("framerate", PropValue::Fraction(25, 1))
        .unwrap();
    assert_eq!(
        e.get_property("framerate"),
        Some(PropValue::Fraction(25, 1))
    );
    assert!(e
        .set_property("in-format", PropValue::Str("nosuch".into()))
        .is_err());
    assert!(e.set_property("field", PropValue::Uint(2)).is_err());
    assert!(e
        .set_property("framerate", PropValue::Fraction(25, 0))
        .is_err());
}

/// M1096: `cccombiner`'s caption queue cap and how it merges with caption meta
/// the video frame already carries.
#[cfg(feature = "metadata")]
#[test]
fn cccombiner_scheduling_and_meta_merge() {
    use g2g_core::MultiInputElement;
    use g2g_plugins::cccombiner::CcCombiner;
    let mut e = CcCombiner::new();
    for name in ["max-scheduled", "input-meta-processing", "field"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "{name} reports its declared default"
        );
    }
    e.set_property("max-scheduled", PropValue::Uint(4)).unwrap();
    assert_eq!(e.get_property("max-scheduled"), Some(PropValue::Uint(4)));
    e.set_property("input-meta-processing", PropValue::Str("favor".into()))
        .unwrap();
    assert_eq!(
        e.get_property("input-meta-processing"),
        Some(PropValue::Str("favor".into()))
    );
    e.set_property("field", PropValue::Uint(1)).unwrap();
    assert_eq!(e.get_property("field"), Some(PropValue::Uint(1)));
    assert!(e
        .set_property("input-meta-processing", PropValue::Str("nosuch".into()))
        .is_err());
    assert!(e.set_property("field", PropValue::Uint(2)).is_err());
}

/// M1100: the DTLS-SRTP pair's `connection-id` / `is-client` / `pem` knobs and
/// the two read-only ones. The gst `key` / `srtp-cipher` / `srtp-auth` overrides
/// that would disable DTLS are not accepted: a fixed key is `srtpenc` /
/// `srtpdec`'s job.
#[cfg(feature = "dtls-srtp")]
#[test]
fn dtls_srtp_pair_carries_its_connection_knobs() {
    use g2g_core::{MultiInputElement, MultiOutputElement};
    use g2g_plugins::dtlssrtpdec::DtlsSrtpDec;
    use g2g_plugins::dtlssrtpenc::DtlsSrtpEnc;

    let mut encoder = DtlsSrtpEnc::new(2);
    for name in [
        "connection-id",
        "is-client",
        "connection-state",
        "peer-pem",
        "peer-fingerprint",
    ] {
        assert!(
            declares(MultiInputElement::properties(&encoder), name),
            "dtlssrtpenc must declare {name}"
        );
        assert_eq!(
            MultiInputElement::get_property(&encoder, name),
            Some(declared_default(
                MultiInputElement::properties(&encoder),
                name
            )),
            "dtlssrtpenc {name} reports its declared default"
        );
    }
    MultiInputElement::set_property(
        &mut encoder,
        "connection-id",
        PropValue::Str("session-a".into()),
    )
    .unwrap();
    MultiInputElement::set_property(&mut encoder, "is-client", PropValue::Bool(true)).unwrap();
    assert_eq!(
        MultiInputElement::get_property(&encoder, "connection-id"),
        Some(PropValue::Str("session-a".into()))
    );
    assert_eq!(
        MultiInputElement::get_property(&encoder, "is-client"),
        Some(PropValue::Bool(true))
    );
    // Read-only, and no key override: DTLS delivers the key.
    assert!(MultiInputElement::set_property(
        &mut encoder,
        "connection-state",
        PropValue::Str("connected".into())
    )
    .is_err());
    for name in [
        "key",
        "srtp-cipher",
        "srtp-auth",
        "srtcp-cipher",
        "srtcp-auth",
    ] {
        assert!(
            MultiInputElement::set_property(&mut encoder, name, PropValue::Str(String::new()))
                .is_err(),
            "dtlssrtpenc must not accept {name}"
        );
    }

    let mut decoder = DtlsSrtpDec::default();
    for name in [
        "connection-id",
        "pem",
        "connection-state",
        "peer-pem",
        "peer-fingerprint",
    ] {
        assert!(
            declares(MultiOutputElement::properties(&decoder), name),
            "dtlssrtpdec must declare {name}"
        );
        assert_eq!(
            MultiOutputElement::get_property(&decoder, name),
            Some(declared_default(
                MultiOutputElement::properties(&decoder),
                name
            )),
            "dtlssrtpdec {name} reports its declared default"
        );
    }
    MultiOutputElement::set_property(
        &mut decoder,
        "connection-id",
        PropValue::Str("session-a".into()),
    )
    .unwrap();
    MultiOutputElement::set_property(
        &mut decoder,
        "pem",
        PropValue::Str("-----BEGIN CERTIFICATE-----".into()),
    )
    .unwrap();
    assert_eq!(
        MultiOutputElement::get_property(&decoder, "connection-id"),
        Some(PropValue::Str("session-a".into()))
    );
    assert!(MultiOutputElement::set_property(
        &mut decoder,
        "peer-pem",
        PropValue::Str(String::new())
    )
    .is_err());
    for name in ["key", "roc", "srtp-cipher", "srtcp-auth"] {
        assert!(
            MultiOutputElement::set_property(&mut decoder, name, PropValue::Str(String::new()))
                .is_err(),
            "dtlssrtpdec must not accept {name}"
        );
    }

    // M1101: the pinned peer certificate, in the SDP `a=fingerprint` value
    // form. It reads back with the hash name and uppercase octets whichever
    // way it was written, and an unparseable value is refused.
    const LOWERCASE_FINGERPRINT: &str = "sha-256 \
                                         ab:cd:ef:01:23:45:67:89:ab:cd:ef:01:23:45:67:89:\
                                         ab:cd:ef:01:23:45:67:89:ab:cd:ef:01:23:45:67:89";
    let expected = PropValue::Str(
        LOWERCASE_FINGERPRINT
            .to_uppercase()
            .replace("SHA-256", "sha-256"),
    );
    MultiInputElement::set_property(
        &mut encoder,
        "peer-fingerprint",
        PropValue::Str(LOWERCASE_FINGERPRINT.into()),
    )
    .expect("dtlssrtpenc takes a fingerprint");
    assert_eq!(
        MultiInputElement::get_property(&encoder, "peer-fingerprint"),
        Some(expected.clone())
    );
    // The hash name is optional: the digest alone names the same certificate.
    let digest_only = LOWERCASE_FINGERPRINT
        .split_once(' ')
        .expect("the hash name and the digest")
        .1;
    MultiOutputElement::set_property(
        &mut decoder,
        "peer-fingerprint",
        PropValue::Str(digest_only.into()),
    )
    .expect("dtlssrtpdec takes a bare digest");
    assert_eq!(
        MultiOutputElement::get_property(&decoder, "peer-fingerprint"),
        Some(expected)
    );
    // Empty clears the pin, and a short or misnamed digest is refused.
    MultiOutputElement::set_property(
        &mut decoder,
        "peer-fingerprint",
        PropValue::Str(String::new()),
    )
    .expect("an empty fingerprint accepts any peer");
    assert_eq!(
        MultiOutputElement::get_property(&decoder, "peer-fingerprint"),
        Some(PropValue::Str(String::new()))
    );
    for refused in [
        &digest_only[..digest_only.len() - 3],
        &format!("sha-1 {digest_only}"),
        &digest_only.replace(':', ""),
    ] {
        assert!(
            MultiOutputElement::set_property(
                &mut decoder,
                "peer-fingerprint",
                PropValue::Str(refused.to_string())
            )
            .is_err(),
            "dtlssrtpdec must refuse `{refused}`"
        );
    }
}

/// M1103: gaudieffects knobs round-trip through `set_property` / `get_property`.
#[test]
fn solarize_threshold_start_end() {
    use g2g_plugins::gaudieffects::Solarize;
    let mut e = Solarize::new();
    for name in ["threshold", "start", "end"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "{name} reports its declared default"
        );
    }
    e.set_property("threshold", PropValue::Uint(100)).unwrap();
    assert_eq!(e.get_property("threshold"), Some(PropValue::Uint(100)));
    assert!(e.set_property("threshold", PropValue::Uint(257)).is_err());
}

#[test]
fn chromium_edge_a_and_edge_b() {
    use g2g_plugins::gaudieffects::Chromium;
    let mut e = Chromium::new();
    assert!(declares(e.properties(), "edge-a"));
    assert!(declares(e.properties(), "edge-b"));
    e.set_property("edge-a", PropValue::Uint(10)).unwrap();
    e.set_property("edge-b", PropValue::Uint(20)).unwrap();
    assert_eq!(e.get_property("edge-a"), Some(PropValue::Uint(10)));
    assert_eq!(e.get_property("edge-b"), Some(PropValue::Uint(20)));
}

#[test]
fn exclusion_factor_and_burn_adjustment() {
    use g2g_plugins::gaudieffects::{Burn, Exclusion};
    let mut exclusion = Exclusion::new();
    assert!(declares(exclusion.properties(), "factor"));
    exclusion
        .set_property("factor", PropValue::Uint(50))
        .unwrap();
    assert_eq!(exclusion.get_property("factor"), Some(PropValue::Uint(50)));
    assert!(exclusion
        .set_property("factor", PropValue::Uint(0))
        .is_err());

    let mut burn = Burn::new();
    assert!(declares(burn.properties(), "adjustment"));
    burn.set_property("adjustment", PropValue::Uint(10))
        .unwrap();
    assert_eq!(burn.get_property("adjustment"), Some(PropValue::Uint(10)));
}

#[test]
fn dilate_erode() {
    use g2g_plugins::gaudieffects::Dilate;
    let mut e = Dilate::new();
    assert!(declares(e.properties(), "erode"));
    assert_eq!(
        e.get_property("erode"),
        Some(declared_default(e.properties(), "erode"))
    );
    e.set_property("erode", PropValue::Bool(true)).unwrap();
    assert_eq!(e.get_property("erode"), Some(PropValue::Bool(true)));
}

/// M1104: knobs round-trip through `set_property` / `get_property`.
#[test]
fn pnmenc_ascii() {
    use g2g_plugins::pnm::PnmEnc;
    let mut e = PnmEnc::new();
    assert!(declares(e.properties(), "ascii"));
    assert_eq!(
        e.get_property("ascii"),
        Some(declared_default(e.properties(), "ascii"))
    );
    e.set_property("ascii", PropValue::Bool(true)).unwrap();
    assert_eq!(e.get_property("ascii"), Some(PropValue::Bool(true)));
}

#[test]
fn tonegeneratesrc_freq_volume_samplesperbuffer() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::tonegeneratesrc::ToneGenerateSrc;
    let mut e = ToneGenerateSrc::new();
    assert!(declares(e.properties(), "freq"));
    assert!(declares(e.properties(), "volume"));
    assert!(declares(e.properties(), "samplesperbuffer"));
    assert_eq!(
        e.get_property("freq"),
        Some(declared_default(e.properties(), "freq"))
    );
    e.set_property("freq", PropValue::Double(1000.0)).unwrap();
    assert_eq!(e.get_property("freq"), Some(PropValue::Double(1000.0)));
    e.set_property("volume", PropValue::Double(0.5)).unwrap();
    assert_eq!(e.get_property("volume"), Some(PropValue::Double(0.5)));
    e.set_property("samplesperbuffer", PropValue::Int(256))
        .unwrap();
    assert_eq!(
        e.get_property("samplesperbuffer"),
        Some(PropValue::Int(256))
    );
    assert!(e.set_property("volume", PropValue::Double(1.5)).is_err());
}

#[test]
fn dtmfsrc_interval_number_volume() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::dtmf::DtmfSrc;
    let mut e = DtmfSrc::new();
    for name in [
        "interval",
        "min-pulse-duration",
        "min-inter-digit-interval",
        "number",
        "volume",
    ] {
        assert!(declares(e.properties(), name), "{name} must be declared");
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "{name} reports its declared default"
        );
    }
    e.set_property("interval", PropValue::Uint(40)).unwrap();
    assert_eq!(e.get_property("interval"), Some(PropValue::Uint(40)));
    e.set_property("number", PropValue::Uint(11)).unwrap();
    assert_eq!(e.get_property("number"), Some(PropValue::Uint(11)));
    assert!(e.set_property("number", PropValue::Uint(17)).is_err());
    e.set_property("volume", PropValue::Uint(8)).unwrap();
    assert_eq!(e.get_property("volume"), Some(PropValue::Uint(8)));
}

#[test]
fn debugspy_checksum_type_and_silent() {
    use g2g_plugins::debugspy::DebugSpy;
    let mut e = DebugSpy::new();
    assert!(declares(e.properties(), "checksum-type"));
    assert!(declares(e.properties(), "silent"));
    assert_eq!(
        e.get_property("checksum-type"),
        Some(declared_default(e.properties(), "checksum-type"))
    );
    e.set_property("checksum-type", PropValue::Str("md5".into()))
        .unwrap();
    assert_eq!(
        e.get_property("checksum-type"),
        Some(PropValue::Str("md5".into()))
    );
    assert!(e
        .set_property("checksum-type", PropValue::Str("crc32".into()))
        .is_err());
    e.set_property("silent", PropValue::Bool(true)).unwrap();
    assert_eq!(e.get_property("silent"), Some(PropValue::Bool(true)));
}

#[test]
fn videoanalyse_message() {
    use g2g_plugins::videoanalyse::VideoAnalyse;
    let mut e = VideoAnalyse::new();
    assert!(declares(e.properties(), "message"));
    assert_eq!(
        e.get_property("message"),
        Some(declared_default(e.properties(), "message"))
    );
    e.set_property("message", PropValue::Bool(false)).unwrap();
    assert_eq!(e.get_property("message"), Some(PropValue::Bool(false)));
}

/// M1105: hsv / roundedcorners knobs.
#[test]
fn hsvfilter_hue_shift() {
    use g2g_plugins::hsv::HsvFilter;
    let mut e = HsvFilter::new();
    assert!(declares(e.properties(), "hue-shift"));
    assert_eq!(
        e.get_property("hue-shift"),
        Some(declared_default(e.properties(), "hue-shift"))
    );
    e.set_property("hue-shift", PropValue::Double(180.0))
        .unwrap();
    assert_eq!(e.get_property("hue-shift"), Some(PropValue::Double(180.0)));
}

#[test]
fn hsvdetector_hue_var_range() {
    use g2g_plugins::hsv::HsvDetector;
    let mut e = HsvDetector::new();
    assert!(declares(e.properties(), "hue-var"));
    e.set_property("hue-var", PropValue::Double(45.0)).unwrap();
    assert_eq!(e.get_property("hue-var"), Some(PropValue::Double(45.0)));
    assert!(e.set_property("hue-var", PropValue::Double(181.0)).is_err());
}

#[test]
fn roundedcorners_border_radius_px() {
    use g2g_plugins::roundedcorners::RoundedCorners;
    let mut e = RoundedCorners::new();
    assert!(declares(e.properties(), "border-radius-px"));
    assert_eq!(
        e.get_property("border-radius-px"),
        Some(declared_default(e.properties(), "border-radius-px"))
    );
    e.set_property("border-radius-px", PropValue::Uint(16))
        .unwrap();
    assert_eq!(
        e.get_property("border-radius-px"),
        Some(PropValue::Uint(16))
    );
}

/// M1119: the software decoder's worker-thread count, gst `avdec_*`'s
/// `max-threads`. That the value reaches libavcodec's `thread_count` is proved
/// in `ffmpegdec`'s own `max_threads_reaches_the_codec_context`, which opens a
/// decoder and reads it back; this is the launch-line half, plus the bound that
/// keeps a pasted line from asking for a thread count that is a resource
/// exhaustion rather than a decode setting.
#[cfg(feature = "ffmpeg")]
#[test]
fn ffmpegdec_max_threads() {
    use g2g_plugins::ffmpegdec::FfmpegVideoDec;
    let mut e = FfmpegVideoDec::new();
    assert!(declares(e.properties(), "max-threads"));
    assert_eq!(
        e.get_property("max-threads"),
        Some(declared_default(e.properties(), "max-threads")),
        "gst's default is auto, spelled 0"
    );
    e.set_property("max-threads", PropValue::Uint(4)).unwrap();
    assert_eq!(e.get_property("max-threads"), Some(PropValue::Uint(4)));
    assert_eq!(e.max_threads(), 4, "onto the field the decoder opens with");

    let (_, maximum) = declared_range(e.properties(), "max-threads");
    e.set_property("max-threads", PropValue::Uint(maximum as u64))
        .expect("the declared maximum is accepted");
    assert!(
        e.set_property("max-threads", PropValue::Uint(maximum as u64 + 1))
            .is_err(),
        "past the declared maximum is refused"
    );
}

/// M1119: which threading method the decoder may use, gst `avdec_*`'s
/// `thread-type` with its three nicks. `frame` is the one that trades latency
/// for throughput, so the launch line has to be able to ask for it by name.
#[cfg(feature = "ffmpeg")]
#[test]
fn ffmpegdec_thread_type() {
    use g2g_plugins::ffmpegdec::{FfmpegVideoDec, ThreadType};
    let mut e = FfmpegVideoDec::new();
    assert!(declares(e.properties(), "thread-type"));
    assert_eq!(
        e.get_property("thread-type"),
        Some(declared_default(e.properties(), "thread-type")),
        "gst's default nick is auto"
    );
    for (nick, expected) in [
        ("frame", ThreadType::Frame),
        ("slice", ThreadType::Slice),
        ("auto", ThreadType::Auto),
    ] {
        e.set_property("thread-type", PropValue::Str(nick.into()))
            .unwrap();
        assert_eq!(e.thread_type(), expected, "`{nick}` onto the field");
        assert_eq!(
            e.get_property("thread-type"),
            Some(PropValue::Str(nick.into())),
            "and reads back as the nick it was set with"
        );
    }
    assert!(
        e.set_property("thread-type", PropValue::Str("frame+slice".into()))
            .is_err(),
        "gst's flag-combining spelling is not one of our nicks"
    );
}

/// M1130: `audioreverse`'s batch size, in nanoseconds.
#[test]
fn audioreverse_chunk_duration() {
    use g2g_plugins::audioreverse::AudioReverse;
    let mut e = AudioReverse::new();
    assert!(declares(e.properties(), "chunk-duration"));
    assert_eq!(
        e.get_property("chunk-duration"),
        Some(declared_default(e.properties(), "chunk-duration"))
    );
    e.set_property("chunk-duration", PropValue::Uint(250_000_000))
        .unwrap();
    assert_eq!(
        e.get_property("chunk-duration"),
        Some(PropValue::Uint(250_000_000))
    );
}

/// M1131: `ebur128`'s reporting interval and measurement switch.
#[test]
fn ebur128_interval_and_post_messages() {
    use g2g_plugins::ebur128::Ebur128;
    let mut e = Ebur128::new();
    for name in ["interval", "post-messages"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "{name} reports its declared default"
        );
    }
    e.set_property("interval", PropValue::Uint(400_000_000))
        .unwrap();
    assert_eq!(
        e.get_property("interval"),
        Some(PropValue::Uint(400_000_000))
    );
    e.set_property("post-messages", PropValue::Bool(false))
        .unwrap();
    assert_eq!(
        e.get_property("post-messages"),
        Some(PropValue::Bool(false))
    );
}

/// M1153: `colorspace`'s tone map takes the PQ source peak from a property, and
/// only inside the range it declares.
#[test]
fn colorspace_hdr_peak_nits() {
    use g2g_plugins::colorspace::{
        Colorspace, DEFAULT_HDR_PEAK_NITS, MAXIMUM_HDR_PEAK_NITS, MINIMUM_HDR_PEAK_NITS,
    };

    let mut e = Colorspace::new();
    let nits = |value: u32| PropValue::Uint(u64::from(value));

    assert!(declares(e.properties(), "hdr-peak-nits"));
    assert_eq!(
        e.get_property("hdr-peak-nits"),
        Some(nits(DEFAULT_HDR_PEAK_NITS))
    );
    assert_eq!(
        declared_default(e.properties(), "hdr-peak-nits"),
        nits(DEFAULT_HDR_PEAK_NITS),
        "the declared default is the one a fresh element reports"
    );
    assert_eq!(
        declared_range(e.properties(), "hdr-peak-nits"),
        (
            MINIMUM_HDR_PEAK_NITS as usize,
            MAXIMUM_HDR_PEAK_NITS as usize
        ),
        "the declared range is the one the element enforces"
    );
    for peak in [MINIMUM_HDR_PEAK_NITS, MAXIMUM_HDR_PEAK_NITS] {
        e.set_property("hdr-peak-nits", nits(peak)).unwrap();
        assert_eq!(e.get_property("hdr-peak-nits"), Some(nits(peak)));
    }
    for peak in [MINIMUM_HDR_PEAK_NITS - 1, MAXIMUM_HDR_PEAK_NITS + 1] {
        assert!(
            e.set_property("hdr-peak-nits", nits(peak)).is_err(),
            "{peak} is outside the declared range"
        );
    }
}

/// M1155: `togglerecord`'s three gst properties plus the two that stand in for
/// gst's request pads. `record` and `recording` live on the shared group, so
/// they read back through it rather than off the element.
#[cfg(feature = "std")]
#[test]
fn togglerecord_record_group_and_main() {
    use g2g_plugins::togglerecord::{RecordGroup, ToggleRecord};
    let mut e = ToggleRecord::new();
    for name in ["record", "recording", "is-live", "group", "main"] {
        assert!(declares(e.properties(), name), "{name} must be declared");
    }
    assert_eq!(
        e.get_property("record"),
        Some(declared_default(e.properties(), "record"))
    );
    assert_eq!(
        e.get_property("recording"),
        Some(declared_default(e.properties(), "recording"))
    );
    assert_eq!(
        e.get_property("is-live"),
        Some(declared_default(e.properties(), "is-live"))
    );
    assert_eq!(
        e.get_property("group"),
        Some(declared_default(e.properties(), "group"))
    );
    assert_eq!(
        e.get_property("main"),
        Some(declared_default(e.properties(), "main"))
    );

    e.set_property("is-live", PropValue::Bool(true)).unwrap();
    assert_eq!(e.get_property("is-live"), Some(PropValue::Bool(true)));
    e.set_property("main", PropValue::Bool(false)).unwrap();
    assert_eq!(e.get_property("main"), Some(PropValue::Bool(false)));
    e.set_property("group", PropValue::Str("m454-take".into()))
        .unwrap();
    assert_eq!(
        e.get_property("group"),
        Some(PropValue::Str("m454-take".into()))
    );
    e.set_property("record", PropValue::Bool(true)).unwrap();
    assert!(
        RecordGroup::named("m454-take").record(),
        "`record` is the group's flag, not the element's"
    );
    assert_eq!(
        e.set_property("recording", PropValue::Bool(true)),
        Err(g2g_core::PropError::ReadOnly)
    );
}

/// M1154: `fallbackswitch`'s health-rule knobs, all five settable from a launch
/// line and all reporting the default `gst-inspect` prints.
#[cfg(feature = "std")]
#[test]
fn fallbackswitch_switching_rule() {
    use g2g_core::{MultiInputElement, PropError};
    use g2g_plugins::fallbackswitch::FallbackSwitch;
    let mut e = FallbackSwitch::new(2);
    for name in [
        "active-pad",
        "auto-switch",
        "immediate-fallback",
        "timeout",
        "stop-on-eos",
    ] {
        assert!(declares(e.properties(), name), "{name} is declared");
        assert_eq!(
            e.get_property(name),
            Some(declared_default(e.properties(), name)),
            "{name} reports its declared default"
        );
    }
    e.set_property("timeout", PropValue::Uint(250_000_000))
        .unwrap();
    assert_eq!(
        e.get_property("timeout"),
        Some(PropValue::Uint(250_000_000))
    );
    e.set_property("auto-switch", PropValue::Bool(false))
        .unwrap();
    e.set_property("active-pad", PropValue::Uint(1)).unwrap();
    assert_eq!(e.get_property("active-pad"), Some(PropValue::Uint(1)));
    e.set_property("immediate-fallback", PropValue::Bool(true))
        .unwrap();
    assert_eq!(
        e.get_property("immediate-fallback"),
        Some(PropValue::Bool(true))
    );
    e.set_property("stop-on-eos", PropValue::Bool(true))
        .unwrap();
    assert_eq!(e.get_property("stop-on-eos"), Some(PropValue::Bool(true)));
    // Only the pads that exist can be selected.
    assert_eq!(
        e.set_property("active-pad", PropValue::Uint(2))
            .unwrap_err(),
        PropError::Value
    );
}

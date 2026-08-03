//! M818: runtime properties for every meaningful builder knob (the M454
//! discipline swept across the elements that had drifted). Each property must
//! (a) appear in `properties()` so `parse_launch` can look up its `PropKind`,
//! and (b) round-trip through `set_property` / `get_property` onto the real
//! field the element acts on.

#![cfg(any(
    feature = "rtsp",
    feature = "rtsp-server",
    feature = "srt",
    feature = "udp-ingress",
    feature = "udp-egress",
    feature = "std",
))]

use g2g_core::{PropValue, PropertySpec};

/// True when a spec table declares a property of this name.
#[allow(dead_code)]
fn declares(specs: &[PropertySpec], name: &str) -> bool {
    specs.iter().any(|s| s.name == name)
}

#[cfg(feature = "rtsp-server")]
#[test]
fn rtspserversrc_full_property_set() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::rtspserversrc::RtspServerSrc;
    let mut s = RtspServerSrc::new("0.0.0.0:8554".parse().unwrap());
    for name in [
        "address",
        "port",
        "payload-type",
        "ssrc",
        "width",
        "height",
        "framerate",
        "num-buffers",
        "jitter-latency",
        "jitter-depth",
        "rtcp-rr-interval",
        "nack",
        "rtx-payload-type",
        "rtx-apt",
        "fec-payload-type",
        "flexfec-payload-type",
    ] {
        assert!(declares(s.properties(), name), "missing spec: {name}");
    }
    s.set_property("port", PropValue::Uint(9554)).unwrap();
    assert_eq!(s.get_property("port"), Some(PropValue::Uint(9554)));
    s.set_property("payload-type", PropValue::Uint(97)).unwrap();
    assert_eq!(s.get_property("payload-type"), Some(PropValue::Uint(97)));
    assert!(s
        .set_property("payload-type", PropValue::Uint(128))
        .is_err());
    s.set_property("ssrc", PropValue::Uint(0xDEAD)).unwrap();
    assert_eq!(s.get_property("ssrc"), Some(PropValue::Uint(0xDEAD)));
    s.set_property("width", PropValue::Uint(1920)).unwrap();
    s.set_property("height", PropValue::Uint(1080)).unwrap();
    s.set_property("framerate", PropValue::Uint(60)).unwrap();
    assert_eq!(s.get_property("width"), Some(PropValue::Uint(1920)));
    assert_eq!(s.get_property("height"), Some(PropValue::Uint(1080)));
    assert_eq!(s.get_property("framerate"), Some(PropValue::Uint(60)));
    // num-buffers: -1 = unlimited (internal 0), 0 rejected.
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(-1)));
    s.set_property("num-buffers", PropValue::Int(25)).unwrap();
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(25)));
    assert!(s.set_property("num-buffers", PropValue::Int(0)).is_err());
    s.set_property("num-buffers", PropValue::Int(-1)).unwrap();
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(-1)));
    // Receive-path tuning (shared dispatch with UdpSrc).
    s.set_property("jitter-latency", PropValue::Uint(80))
        .unwrap();
    s.set_property("jitter-depth", PropValue::Uint(128))
        .unwrap();
    assert_eq!(s.get_property("jitter-latency"), Some(PropValue::Uint(80)));
    assert_eq!(s.get_property("jitter-depth"), Some(PropValue::Uint(128)));
    s.set_property("rtcp-rr-interval", PropValue::Uint(500))
        .unwrap();
    s.set_property("nack", PropValue::Bool(true)).unwrap();
    assert_eq!(
        s.get_property("rtcp-rr-interval"),
        Some(PropValue::Uint(500))
    );
    assert_eq!(s.get_property("nack"), Some(PropValue::Bool(true)));
}

#[cfg(feature = "udp-ingress")]
#[test]
fn udpsrc_recv_tuning_properties() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::udpsrc::UdpSrc;
    let mut s = UdpSrc::new("0.0.0.0:5004".parse().unwrap());
    for name in [
        "width",
        "height",
        "framerate",
        "num-buffers",
        "jitter-latency",
        "jitter-depth",
        "rtcp-rr-interval",
        "nack",
        "rtx-payload-type",
        "rtx-apt",
        "fec-payload-type",
        "flexfec-payload-type",
    ] {
        assert!(declares(s.properties(), name), "missing spec: {name}");
    }
    s.set_property("width", PropValue::Uint(640)).unwrap();
    s.set_property("height", PropValue::Uint(480)).unwrap();
    s.set_property("framerate", PropValue::Uint(25)).unwrap();
    assert_eq!(s.get_property("width"), Some(PropValue::Uint(640)));
    assert_eq!(s.get_property("height"), Some(PropValue::Uint(480)));
    assert_eq!(s.get_property("framerate"), Some(PropValue::Uint(25)));
    s.set_property("num-buffers", PropValue::Int(100)).unwrap();
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(100)));
    // Defaults reflect RtpRecvConfig::default (1 s RR, NACK on, 50 ms / 64).
    assert_eq!(
        s.get_property("rtcp-rr-interval"),
        Some(PropValue::Uint(1000))
    );
    assert_eq!(s.get_property("nack"), Some(PropValue::Bool(true)));
    assert_eq!(s.get_property("jitter-latency"), Some(PropValue::Uint(50)));
    assert_eq!(s.get_property("jitter-depth"), Some(PropValue::Uint(64)));
    s.set_property("jitter-latency", PropValue::Uint(120))
        .unwrap();
    s.set_property("jitter-depth", PropValue::Uint(0)).unwrap();
    assert_eq!(s.get_property("jitter-latency"), Some(PropValue::Uint(120)));
    assert_eq!(s.get_property("jitter-depth"), Some(PropValue::Uint(0)));
    // rtx legs may arrive in either order; pt 0 disables.
    s.set_property("rtx-apt", PropValue::Uint(96)).unwrap();
    s.set_property("rtx-payload-type", PropValue::Uint(97))
        .unwrap();
    assert_eq!(
        s.get_property("rtx-payload-type"),
        Some(PropValue::Uint(97))
    );
    assert_eq!(s.get_property("rtx-apt"), Some(PropValue::Uint(96)));
    s.set_property("rtx-payload-type", PropValue::Uint(0))
        .unwrap();
    assert_eq!(s.get_property("rtx-payload-type"), Some(PropValue::Uint(0)));
    s.set_property("fec-payload-type", PropValue::Uint(100))
        .unwrap();
    assert_eq!(
        s.get_property("fec-payload-type"),
        Some(PropValue::Uint(100))
    );
    s.set_property("flexfec-payload-type", PropValue::Uint(101))
        .unwrap();
    assert_eq!(
        s.get_property("flexfec-payload-type"),
        Some(PropValue::Uint(101))
    );
    assert!(s
        .set_property("fec-payload-type", PropValue::Uint(128))
        .is_err());
}

#[cfg(feature = "udp-egress")]
#[test]
fn udpsink_send_tuning_properties() {
    use g2g_core::AsyncElement;
    use g2g_plugins::udpsink::UdpSink;
    let mut s = UdpSink::new("127.0.0.1:5004".parse().unwrap());
    for name in [
        "max-payload",
        "retransmit",
        "retx-capacity",
        "rtcp-sr-interval",
        "rtx-payload-type",
        "rtx-ssrc",
    ] {
        assert!(declares(s.properties(), name), "missing spec: {name}");
    }
    s.set_property("max-payload", PropValue::Uint(1200))
        .unwrap();
    assert_eq!(s.get_property("max-payload"), Some(PropValue::Uint(1200)));
    assert!(s.set_property("max-payload", PropValue::Uint(0)).is_err());
    s.set_property("retransmit", PropValue::Bool(false))
        .unwrap();
    assert_eq!(s.get_property("retransmit"), Some(PropValue::Bool(false)));
    s.set_property("retx-capacity", PropValue::Uint(256))
        .unwrap();
    assert_eq!(s.get_property("retx-capacity"), Some(PropValue::Uint(256)));
    // 0 = off (None internally).
    assert_eq!(s.get_property("rtcp-sr-interval"), Some(PropValue::Uint(0)));
    s.set_property("rtcp-sr-interval", PropValue::Uint(5000))
        .unwrap();
    assert_eq!(
        s.get_property("rtcp-sr-interval"),
        Some(PropValue::Uint(5000))
    );
    s.set_property("rtx-ssrc", PropValue::Uint(0xBEEF)).unwrap();
    s.set_property("rtx-payload-type", PropValue::Uint(97))
        .unwrap();
    assert_eq!(
        s.get_property("rtx-payload-type"),
        Some(PropValue::Uint(97))
    );
    assert_eq!(s.get_property("rtx-ssrc"), Some(PropValue::Uint(0xBEEF)));
}

#[cfg(feature = "rtsp")]
#[test]
fn rtspsrc_limit_dims_and_reconnect_properties() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::rtspsrc::RtspSrc;
    let mut s = RtspSrc::new("rtsp://example/stream");
    for name in [
        "num-buffers",
        "width",
        "height",
        "reconnect",
        "reconnect-backoff",
        "reconnect-backoff-max",
    ] {
        assert!(declares(s.properties(), name), "missing spec: {name}");
    }
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(-1)));
    s.set_property("num-buffers", PropValue::Int(30)).unwrap();
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(30)));
    s.set_property("width", PropValue::Uint(1920)).unwrap();
    s.set_property("height", PropValue::Uint(1080)).unwrap();
    assert_eq!(s.get_property("width"), Some(PropValue::Uint(1920)));
    assert_eq!(s.get_property("height"), Some(PropValue::Uint(1080)));
    // A bare `reconnect=` fills the backoff defaults like with_reconnect.
    s.set_property("reconnect", PropValue::Uint(3)).unwrap();
    assert_eq!(s.get_property("reconnect"), Some(PropValue::Uint(3)));
    assert_eq!(
        s.get_property("reconnect-backoff"),
        Some(PropValue::Uint(250))
    );
    assert_eq!(
        s.get_property("reconnect-backoff-max"),
        Some(PropValue::Uint(5000))
    );
    s.set_property("reconnect-backoff", PropValue::Uint(100))
        .unwrap();
    s.set_property("reconnect-backoff-max", PropValue::Uint(2000))
        .unwrap();
    assert_eq!(
        s.get_property("reconnect-backoff"),
        Some(PropValue::Uint(100))
    );
    assert_eq!(
        s.get_property("reconnect-backoff-max"),
        Some(PropValue::Uint(2000))
    );
}

/// A half-set width/height pair must not advertise a 0 dim: intercept_caps
/// falls back to the probe path unless both are nonzero.
#[cfg(feature = "rtsp")]
#[tokio::test]
async fn rtspsrc_half_set_dims_do_not_fix_caps() {
    use g2g_core::runtime::SourceLoop;
    use g2g_core::{Caps, Dim};
    use g2g_plugins::rtspsrc::RtspSrc;
    let mut s = RtspSrc::new("rtsp://127.0.0.1:1/none");
    s.set_property("width", PropValue::Uint(1920)).unwrap();
    // Only width set: the probe path runs (and fails against the dead URL),
    // rather than advertising width=1920 height=0.
    assert!(s.intercept_caps().await.is_err());
    s.set_property("height", PropValue::Uint(1080)).unwrap();
    let caps = s.intercept_caps().await.unwrap();
    let Caps::CompressedVideo { width, height, .. } = caps else {
        panic!("expected video caps");
    };
    assert_eq!(width, Dim::Fixed(1920));
    assert_eq!(height, Dim::Fixed(1080));
}

#[cfg(feature = "srt")]
#[test]
fn srtsrc_num_buffers() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::srtsrc::SrtSrc;
    let mut s = SrtSrc::new("0.0.0.0:9000".parse().unwrap());
    assert!(declares(s.properties(), "num-buffers"));
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(-1)));
    s.set_property("num-buffers", PropValue::Int(10)).unwrap();
    assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(10)));
    assert!(s.set_property("num-buffers", PropValue::Int(0)).is_err());
}

#[cfg(feature = "std")]
#[test]
fn filesrc_blocksize() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::filesrc::FileSrc;
    let mut s = FileSrc::untyped();
    assert!(declares(s.properties(), "blocksize"));
    assert_eq!(s.get_property("blocksize"), Some(PropValue::Uint(65536)));
    s.set_property("blocksize", PropValue::Uint(4096)).unwrap();
    assert_eq!(s.get_property("blocksize"), Some(PropValue::Uint(4096)));
    assert!(s.set_property("blocksize", PropValue::Uint(0)).is_err());
}

#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
#[test]
fn ffmpegdec_backend_properties() {
    use g2g_core::AsyncElement;
    use g2g_plugins::ffmpegdec::FfmpegH264Dec;
    let mut d = FfmpegH264Dec::new();
    for name in ["backend", "cuvid-surfaces", "low-delay"] {
        assert!(declares(d.properties(), name), "missing spec: {name}");
    }
    assert_eq!(
        d.get_property("backend"),
        Some(PropValue::Str("software".into()))
    );
    assert_eq!(d.get_property("low-delay"), Some(PropValue::Bool(false)));
    // Selecting cuvid applies the same latency defaults as with_backend.
    d.set_property("backend", PropValue::Str("nvdec-cuvid".into()))
        .unwrap();
    assert_eq!(
        d.get_property("backend"),
        Some(PropValue::Str("nvdec-cuvid".into()))
    );
    assert_eq!(d.get_property("cuvid-surfaces"), Some(PropValue::Uint(4)));
    assert_eq!(d.get_property("low-delay"), Some(PropValue::Bool(true)));
    // Explicit overrides after backend selection stick.
    d.set_property("cuvid-surfaces", PropValue::Uint(8))
        .unwrap();
    assert_eq!(d.get_property("cuvid-surfaces"), Some(PropValue::Uint(8)));
    d.set_property("low-delay", PropValue::Bool(false)).unwrap();
    assert_eq!(d.get_property("low-delay"), Some(PropValue::Bool(false)));
    // Switching back to software resets them.
    d.set_property("backend", PropValue::Str("software".into()))
        .unwrap();
    assert_eq!(d.get_property("cuvid-surfaces"), Some(PropValue::Uint(0)));
    assert_eq!(d.get_property("low-delay"), Some(PropValue::Bool(false)));
    assert!(d
        .set_property("backend", PropValue::Str("quicksync".into()))
        .is_err());
}

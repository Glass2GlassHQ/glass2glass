//! M1040: the capture and ingest sources spell `num-buffers` the way gst
//! `basesrc` does, so -1 runs forever, n emits exactly n, and 0 emits nothing
//! and goes straight to EOS. They used to read 0 as "forever" (v4l2src,
//! mfvideosrc) or refuse it outright (the RTP / SRT ingest family), so this pins
//! the converted behavior.
//!
//! The run tests need no peer at all: a zero limit ends the source before it
//! listens, which is the point. They are wrapped in a timeout, since the
//! pre-M1040 reading of 0 blocks on the socket forever.
//!
//! ```sh
//! cargo test -p g2g-plugins --features udp-ingress,srt,rtsp-server,v4l2 \
//!     --test m1040_num_buffers_zero
//! ```

#![cfg(any(
    feature = "udp-ingress",
    feature = "srt",
    feature = "rtsp-server",
    all(target_os = "linux", feature = "v4l2")
))]

use g2g_core::PropValue;

/// -1 reads back as -1, n as n, and 0 as 0: a real count now, not a rejected
/// value and not a second spelling of "forever".
macro_rules! assert_num_buffers_round_trips {
    ($source:expr) => {{
        let source = &mut $source;
        assert_eq!(
            source.get_property("num-buffers"),
            Some(PropValue::Int(-1)),
            "a fresh source is unlimited"
        );
        source
            .set_property("num-buffers", PropValue::Int(7))
            .unwrap();
        assert_eq!(source.get_property("num-buffers"), Some(PropValue::Int(7)));
        source
            .set_property("num-buffers", PropValue::Int(0))
            .unwrap();
        assert_eq!(
            source.get_property("num-buffers"),
            Some(PropValue::Int(0)),
            "0 is a count of zero, not unlimited"
        );
        source
            .set_property("num-buffers", PropValue::Int(-1))
            .unwrap();
        assert_eq!(source.get_property("num-buffers"), Some(PropValue::Int(-1)));
    }};
}

#[cfg(any(feature = "udp-ingress", feature = "srt", feature = "rtsp-server"))]
mod zero_limit_run {
    use core::time::Duration;

    use g2g_core::element::PushOutcome;
    use g2g_core::{G2gError, OutputSink, PipelinePacket, PropValue};

    /// How long a zero-limit `run` may take before we call it hung. Generous:
    /// the correct path does no IO at all.
    const ZERO_LIMIT_DEADLINE: Duration = Duration::from_secs(5);

    /// Collects every packet a directly-driven source pushes.
    #[derive(Default)]
    struct Collect {
        packets: Vec<PipelinePacket>,
    }

    impl OutputSink for Collect {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            self.packets.push(packet);
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    /// A zero limit produced exactly one packet and it was the EOS.
    fn assert_only_eos(out: &Collect, emitted: u64) {
        assert_eq!(emitted, 0, "a zero limit emits no buffers");
        assert_eq!(out.packets.len(), 1, "Eos is the only packet");
        assert!(matches!(out.packets[0], PipelinePacket::Eos));
    }

    #[cfg(feature = "udp-ingress")]
    #[tokio::test]
    async fn udpsrc_num_buffers_zero_emits_only_eos() {
        use g2g_core::runtime::SourceLoop;
        use g2g_plugins::udpsrc::UdpSrc;

        let mut src = UdpSrc::new("127.0.0.1:0".parse().unwrap());
        src.set_property("num-buffers", PropValue::Int(0)).unwrap();
        let caps = src.intercept_caps().await.unwrap();
        src.configure_pipeline(&caps).unwrap();

        let mut out = Collect::default();
        let emitted = tokio::time::timeout(ZERO_LIMIT_DEADLINE, src.run(&mut out))
            .await
            .expect("a zero limit must not wait on the socket")
            .expect("run");
        assert_only_eos(&out, emitted);
    }

    #[cfg(feature = "srt")]
    #[tokio::test]
    async fn srtsrc_num_buffers_zero_emits_only_eos() {
        use g2g_core::runtime::SourceLoop;
        use g2g_plugins::srtsrc::SrtSrc;

        let socket = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let mut src = SrtSrc::from_socket(socket).unwrap();
        src.set_property("num-buffers", PropValue::Int(0)).unwrap();
        let caps = src.intercept_caps().await.unwrap();
        src.configure_pipeline(&caps).unwrap();

        let mut out = Collect::default();
        let emitted = tokio::time::timeout(ZERO_LIMIT_DEADLINE, src.run(&mut out))
            .await
            .expect("a zero limit must not wait on the handshake")
            .expect("run");
        assert_only_eos(&out, emitted);
    }

    #[cfg(feature = "rtsp-server")]
    #[tokio::test]
    async fn rtspserversrc_num_buffers_zero_emits_only_eos() {
        use g2g_core::runtime::SourceLoop;
        use g2g_plugins::rtspserversrc::RtspServerSrc;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut src = RtspServerSrc::from_listener(listener).unwrap();
        src.set_property("num-buffers", PropValue::Int(0)).unwrap();
        let caps = src.intercept_caps().await.unwrap();
        src.configure_pipeline(&caps).unwrap();

        let mut out = Collect::default();
        let emitted = tokio::time::timeout(ZERO_LIMIT_DEADLINE, src.run(&mut out))
            .await
            .expect("a zero limit must not wait for a publisher")
            .expect("run");
        assert_only_eos(&out, emitted);
    }
}

#[cfg(feature = "udp-ingress")]
#[test]
fn udpsrc_num_buffers_round_trips() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::udpsrc::UdpSrc;
    assert_num_buffers_round_trips!(UdpSrc::new("127.0.0.1:0".parse().unwrap()));
}

#[cfg(feature = "srt")]
#[test]
fn srtsrc_num_buffers_round_trips() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::srtsrc::SrtSrc;
    assert_num_buffers_round_trips!(SrtSrc::new("127.0.0.1:9000".parse().unwrap()));
}

#[cfg(feature = "rtsp-server")]
#[test]
fn rtspserversrc_num_buffers_round_trips() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::rtspserversrc::RtspServerSrc;
    assert_num_buffers_round_trips!(RtspServerSrc::new("127.0.0.1:8554".parse().unwrap()));
}

#[cfg(feature = "rtsp-server")]
#[test]
fn rtspserversrcn_num_buffers_round_trips() {
    use g2g_core::MultiOutputSource;
    use g2g_plugins::rtspserversrcn::RtspServerSrcN;
    assert_num_buffers_round_trips!(RtspServerSrcN::new("127.0.0.1:8554".parse().unwrap(), 2));
}

/// v4l2src's capture path needs a real `/dev/videoN`, so only the property half
/// runs here. 0 used to read back as -1 (the old "0 = forever" sentinel).
#[cfg(all(target_os = "linux", feature = "v4l2"))]
#[test]
fn v4l2src_num_buffers_round_trips() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::v4l2src::V4l2Src;
    assert_num_buffers_round_trips!(V4l2Src::new("/dev/video0"));
}

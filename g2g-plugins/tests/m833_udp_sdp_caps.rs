//! `UdpSrc` caps discovery (M833): an RTP receiver has no in-band stream
//! description, so `UdpSrc` used to produce a declared geometry hint and leave
//! the correction to a downstream decoder. It now takes the description from the
//! SDP a sender publishes (`sdp` property, parsed by `g2g_plugins::sdp`) and,
//! failing that, refines its caps at runtime from the stream's own SPS
//! (`CapsChanged` before the frame that carries the new geometry).
//!
//! The reference-peer half (`ffmpeg_sdp_and_stream_configure_udpsrc`) is ignored
//! by default (needs ffmpeg, opens a local UDP socket). Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features udp-ingress --test m833_udp_sdp_caps \
//!     -- --ignored --nocapture
//! ```
#![cfg(feature = "udp-ingress")]

use core::future::Future;
use core::pin::Pin;
use std::net::UdpSocket as StdUdpSocket;
use std::process::Command;
use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AudioFormat, Caps, Dim, G2gError, OutputSink, PipelinePacket, PropValue, PushOutcome, Rate,
    VideoCodec,
};
use g2g_plugins::rtpdepay::RtpH264Depayloader;
use g2g_plugins::rtprecv::SpsCapsRefiner;
use g2g_plugins::sdp::SdpMedia;
use g2g_plugins::udpsrc::UdpSrc;

/// The SDP `ffmpeg -f rtp -sdp_file` writes for 320x240@15 H.264 with
/// `-flags +global_header` (so the parameter sets ride in the `fmtp`).
const FFMPEG_SDP: &str = "v=0\r\n\
    o=- 0 0 IN IP4 127.0.0.1\r\n\
    s=No Name\r\n\
    c=IN IP4 127.0.0.1\r\n\
    t=0 0\r\n\
    m=video 45999 RTP/AVP 96\r\n\
    a=framerate:15\r\n\
    a=rtpmap:96 H264/90000\r\n\
    a=fmtp:96 packetization-mode=1; \
    sprop-parameter-sets=Z/QADJGWgUH7ARAAAAMAEAAAAwHg8UKq,aM4PGSA=; profile-level-id=F4000C\r\n";

/// The SPS / PPS of that same stream, Annex-B payload (no start code): 320x240.
const SPS: &[u8] = &[
    0x67, 0xf4, 0x00, 0x0c, 0x91, 0x96, 0x81, 0x41, 0xfb, 0x01, 0x10, 0x00, 0x00, 0x03, 0x00, 0x10,
    0x00, 0x00, 0x03, 0x01, 0xe0, 0xf1, 0x42, 0xaa,
];
const PPS: &[u8] = &[0x68, 0xce, 0x0f, 0x19, 0x20];

fn h264_caps(width: u32, height: u32, fps: u32) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(fps << 16),
    }
}

/// One single-NAL RTP packet (payload type 96) carrying `nal` verbatim.
fn rtp_packet(seq: u16, ts: u32, marker: bool, nal: &[u8]) -> Vec<u8> {
    let mut p = vec![0x80u8, if marker { 0x80 | 96 } else { 96 }];
    p.extend_from_slice(&seq.to_be_bytes());
    p.extend_from_slice(&ts.to_be_bytes());
    p.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]); // ssrc
    p.extend_from_slice(nal);
    p
}

/// One access unit's worth of packets: SPS, PPS, then a marked IDR slice.
fn keyframe_packets(first_seq: u16, ts: u32) -> Vec<Vec<u8>> {
    vec![
        rtp_packet(first_seq, ts, false, SPS),
        rtp_packet(first_seq + 1, ts, false, PPS),
        rtp_packet(first_seq + 2, ts, true, &[0x65, 0x88, 0x84, 0x00]),
    ]
}

/// Records the packet kinds a source pushes, in order.
#[derive(Default)]
struct OrderedSink {
    caps: Vec<Caps>,
    /// `"caps"` / `"frame"` / `"eos"` in push order.
    order: Vec<&'static str>,
}

impl OutputSink for OrderedSink {
    fn push<'a>(
        &'a mut self,
        p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        match p {
            PipelinePacket::CapsChanged(c) => {
                self.caps.push(c);
                self.order.push("caps");
            }
            PipelinePacket::DataFrame(_) => self.order.push("frame"),
            PipelinePacket::Eos => self.order.push("eos"),
            _ => {}
        }
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

#[tokio::test]
async fn sdp_configures_geometry_rate_and_port() {
    let mut src = UdpSrc::new("0.0.0.0:5004".parse().unwrap())
        .with_sdp(FFMPEG_SDP)
        .expect("the SDP describes an H.264 stream");
    assert_eq!(
        src.intercept_caps().await.unwrap(),
        h264_caps(320, 240, 15),
        "geometry from the sprop SPS, rate from a=framerate"
    );
    assert_eq!(
        src.get_property("port"),
        Some(PropValue::Uint(45999)),
        "the m= port is where the sender is sending"
    );
    assert_eq!(
        src.get_property("sdp"),
        Some(PropValue::Str(FFMPEG_SDP.into())),
        "the property reads back what was set"
    );
}

#[tokio::test]
async fn sdp_property_accepts_a_file_path() {
    let path = std::env::temp_dir().join("m833_udpsrc_probe.sdp");
    std::fs::write(&path, FFMPEG_SDP).expect("write the sdp file");
    let mut src = UdpSrc::new("0.0.0.0:5004".parse().unwrap());
    src.set_property("sdp", PropValue::Str(path.to_string_lossy().into_owned()))
        .expect("a path is read as a document");
    assert_eq!(src.intercept_caps().await.unwrap(), h264_caps(320, 240, 15));
    std::fs::remove_file(&path).ok();
}

#[test]
fn opus_section_maps_to_audio_caps_and_is_refused_by_the_h264_source() {
    let text = "v=0\r\n\
        c=IN IP4 127.0.0.1\r\n\
        m=audio 5006 RTP/AVP 111\r\n\
        a=rtpmap:111 opus/48000/2\r\n\
        a=fmtp:111 minptime=10;useinbandfec=1\r\n";
    let media = SdpMedia::parse(text).expect("the opus section maps");
    assert_eq!(
        media.caps,
        Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: 48_000,
        }
    );
    assert_eq!(media.payload_type, 111);

    // UdpSrc depayloads H.264 only, so it must refuse the description outright
    // rather than accept a property it cannot honour.
    let mut src = UdpSrc::new("0.0.0.0:5004".parse().unwrap());
    assert!(
        !src.apply_sdp(&media),
        "an audio SDP is not receivable here"
    );
    assert!(src
        .set_property("sdp", PropValue::Str(text.into()))
        .is_err());
    assert_eq!(
        src.get_property("port"),
        Some(PropValue::Uint(5004)),
        "a refused SDP changes nothing"
    );
}

#[tokio::test]
async fn declared_hint_is_the_fallback_and_the_override() {
    // No SDP: the declared hint is what negotiation runs on.
    let mut src = UdpSrc::new("0.0.0.0:5004".parse().unwrap());
    assert_eq!(
        src.intercept_caps().await.unwrap(),
        h264_caps(1280, 720, 30)
    );

    // An explicit hint overrides the SDP when it is set afterwards.
    let mut src = UdpSrc::new("0.0.0.0:5004".parse().unwrap())
        .with_sdp(FFMPEG_SDP)
        .expect("sdp applies")
        .with_video_size(640, 480)
        .with_framerate(25);
    assert_eq!(src.intercept_caps().await.unwrap(), h264_caps(640, 480, 25));
}

#[test]
fn sps_refines_declared_caps_once_from_a_canned_rtp_sequence() {
    // The receive path's units, driven directly: depayload canned RTP into an
    // access unit, then refine the declared caps from its SPS.
    let mut depay = RtpH264Depayloader::new();
    let mut refiner = SpsCapsRefiner::new(h264_caps(1280, 720, 30));

    let mut refined = Vec::new();
    for ts in [9000u32, 12_000] {
        let seq = if ts == 9000 { 0 } else { 3 };
        for packet in keyframe_packets(seq, ts) {
            if let Some(au) = depay.depacketize(&packet) {
                if let Some(caps) = refiner.refine(&au.data) {
                    refined.push(caps);
                }
            }
        }
    }
    assert_eq!(
        refined,
        vec![h264_caps(320, 240, 15)],
        "the first SPS corrects the hint; the identical second one is suppressed"
    );
    assert_eq!(refiner.caps(), &h264_caps(320, 240, 15));
}

#[tokio::test]
async fn udpsrc_emits_capschanged_before_the_frame_it_describes() {
    let sock = StdUdpSocket::bind("127.0.0.1:0").expect("bind udp");
    let port = sock.local_addr().unwrap().port();
    let mut src = UdpSrc::from_socket(sock)
        .expect("adopt socket")
        .with_frame_limit(1);
    src.configure_pipeline(&h264_caps(1280, 720, 30))
        .expect("configure");

    // Blast the canned access unit at it; the kernel buffers until run() reads.
    let tx = StdUdpSocket::bind("127.0.0.1:0").expect("bind sender");
    for packet in keyframe_packets(0, 9000) {
        tx.send_to(&packet, ("127.0.0.1", port)).expect("send rtp");
    }

    let mut sink = OrderedSink::default();
    let received = tokio::time::timeout(Duration::from_secs(10), src.run(&mut sink))
        .await
        .expect("the access unit arrives")
        .expect("run");
    assert_eq!(received, 1);
    assert_eq!(
        sink.order,
        vec!["caps", "frame", "eos"],
        "the refined caps precede the frame they describe"
    );
    assert_eq!(
        sink.caps,
        vec![h264_caps(320, 240, 15)],
        "the declared 1280x720@30 hint is corrected from the stream's SPS"
    );
}

#[tokio::test]
#[ignore = "needs ffmpeg with the rtp muxer; opens a local UDP socket"]
async fn ffmpeg_sdp_and_stream_configure_udpsrc() {
    // Bind first so the SDP ffmpeg writes names the port we listen on.
    let sock = StdUdpSocket::bind("127.0.0.1:0").expect("bind udp");
    let port = sock.local_addr().unwrap().port();
    let sdp_path = std::env::temp_dir().join(format!("m833_ffmpeg_{port}.sdp"));
    std::fs::remove_file(&sdp_path).ok();

    const N: u64 = 20;
    let url = format!("rtp://127.0.0.1:{port}");
    let sdp_arg = sdp_path.to_string_lossy().into_owned();
    let ffmpeg = tokio::task::spawn_blocking(move || {
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-an",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=15:duration=3",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                // Carry SPS/PPS out of band, so the SDP declares the geometry.
                "-flags",
                "+global_header",
                "-payload_type",
                "96",
                "-f",
                "rtp",
                "-sdp_file",
                &sdp_arg,
                &url,
            ])
            .status()
    });

    // ffmpeg writes the SDP as it opens the output; wait for it to land.
    let mut text = String::new();
    for _ in 0..100 {
        if let Ok(s) = std::fs::read_to_string(&sdp_path) {
            if s.contains("m=video") {
                text = s;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(!text.is_empty(), "ffmpeg wrote an SDP at {sdp_path:?}");

    // What ffmpeg declared: 320x240 H.264 at 15 fps on our port, PT 96.
    let media = SdpMedia::parse(&text).expect("the ffmpeg SDP maps to caps");
    assert_eq!(media.payload_type, 96);
    assert_eq!(media.clock_rate, 90_000);
    assert_eq!(media.port, port);
    assert_eq!(
        media.caps,
        h264_caps(320, 240, 15),
        "discovered caps match ffmpeg's declaration\n{text}"
    );

    let mut src = UdpSrc::from_socket(sock)
        .expect("adopt socket")
        .with_frame_limit(N);
    assert!(
        src.apply_sdp(&media),
        "the ffmpeg SDP configures the source"
    );
    src.configure_pipeline(&h264_caps(320, 240, 15))
        .expect("configure");

    let mut sink = OrderedSink::default();
    let received = tokio::time::timeout(Duration::from_secs(30), src.run(&mut sink))
        .await
        .expect("UdpSrc receives N access units within 30s")
        .expect("UdpSrc runs");
    let _ = ffmpeg.await;
    std::fs::remove_file(&sdp_path).ok();

    assert_eq!(received, N, "depayloaded the requested access units");
    // The in-band SPS agrees with the SDP, so no correction is emitted: the
    // stream ffmpeg sends is the stream its SDP described.
    assert!(
        sink.caps.is_empty(),
        "SDP-configured caps need no in-band correction, got {:?}",
        sink.caps
    );
    assert!(
        sink.order.iter().filter(|k| **k == "frame").count() as u64 == N,
        "every access unit reached the sink"
    );

    use g2g_core::conformance::{ConformanceDimension, Evidence};
    g2g_plugins::conformance::persist::record_evidence(
        "udpsrc",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264")
            .detail("caps discovered from the ffmpeg-published SDP match its declared stream"),
    )
    .expect("record oracle evidence");
}

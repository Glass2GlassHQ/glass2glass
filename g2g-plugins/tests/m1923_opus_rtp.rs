#![cfg(all(feature = "udp-ingress", feature = "udp-egress", feature = "opus"))]

use core::future::Future;
use core::pin::Pin;
use std::net::UdpSocket as StdUdpSocket;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use g2g_core::runtime::{
    parse_launch, run_simple_pipeline, run_source_transform_sink, LatencyProfile, SourceLoop,
};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, ConfigureOutcome, G2gError, OutputSink, PipelineClock,
    PipelinePacket, PropValue,
};

use g2g_plugins::audiotestsrc::AudioTestSrc;
use g2g_plugins::opusenc::{OpusEnc, OpusFrameSize};
use g2g_plugins::opusparse::OPUS_RATE_HZ;
use g2g_plugins::registry::default_registry;
use g2g_plugins::rtpdepay::RtpEncoding;
use g2g_plugins::rtppay::RtpOpusPacketizer;
use g2g_plugins::sdp::{OPUS_CLOCK_HZ, OPUS_RTPMAP_CHANNELS};
use g2g_plugins::udpsink::UdpSink;
use g2g_plugins::udpsrc::UdpSrc;

// a whole number of 20 ms Opus frames, so opusenc pads no tail
const SOURCE_BUFFERS: u64 = 100;
const SOURCE_CHANNELS: u8 = 2;
const SOURCE_TONE_HZ: u32 = 440;

const PAYLOAD_TYPE: u8 = 96;
const SSRC: u32 = 0x6732_1923;

const LOOPBACK_DEADLINE: Duration = Duration::from_secs(20);

// UDP drops what is sent before the gst receiver binds its socket
const PEER_LISTEN_WAIT: Duration = Duration::from_millis(1500);

const PEER_DEADLINE: Duration = Duration::from_secs(60);

// gst opusenc sends one more packet at EOS, the receiver stops after these
const GST_PACKETS: u64 = 50;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn opus_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: OPUS_RTPMAP_CHANNELS,
        sample_rate: OPUS_CLOCK_HZ,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

#[derive(Default)]
struct Recorder {
    caps: Vec<Caps>,
    frames: Vec<(Vec<u8>, u64)>,
}

impl Recorder {
    fn payloads(&self) -> Vec<&[u8]> {
        self.frames
            .iter()
            .map(|(bytes, _)| bytes.as_slice())
            .collect()
    }

    fn pts(&self) -> Vec<u64> {
        self.frames.iter().map(|(_, pts)| *pts).collect()
    }
}

impl AsyncElement for Recorder {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, caps: &Caps) -> Result<Caps, G2gError> {
        Ok(caps.clone())
    }

    fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.caps.push(caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let bytes = frame
                        .domain
                        .as_system_slice()
                        .expect("RTP frames are in system memory")
                        .to_vec();
                    self.frames.push((bytes, frame.timing.pts_ns));
                }
                PipelinePacket::CapsChanged(caps) => self.caps.push(caps),
                _ => {}
            }
            Ok(())
        })
    }
}

fn test_tone() -> AudioTestSrc {
    AudioTestSrc::new(
        OPUS_RATE_HZ,
        SOURCE_CHANNELS,
        SOURCE_TONE_HZ,
        SOURCE_BUFFERS,
    )
}

// libopus encodes the same input to the same bytes
async fn encoded_reference() -> Recorder {
    let mut recorder = Recorder::default();
    run_source_transform_sink(
        &mut test_tone(),
        &mut OpusEnc::new(),
        &mut recorder,
        &ZeroClock,
        LatencyProfile::Live.link_capacity(),
    )
    .await
    .expect("the encode pipeline runs");
    assert!(!recorder.frames.is_empty(), "opusenc produced packets");
    recorder
}

// bound up front so the socket queues what arrives before the receive loop runs
fn bound_opus_source(frame_limit: u64) -> (UdpSrc, u16) {
    let socket = StdUdpSocket::bind("127.0.0.1:0").expect("bind a receiver");
    let src = UdpSrc::from_socket(socket)
        .expect("adopt the socket")
        .with_encoding(RtpEncoding::Opus)
        .with_frame_limit(frame_limit);
    let port = src.local_port().expect("the bound port");
    (src, port)
}

async fn receive(mut source: UdpSrc) -> Recorder {
    let mut recorder = Recorder::default();
    run_simple_pipeline(
        &mut source,
        &mut recorder,
        &ZeroClock,
        LatencyProfile::Live.link_capacity(),
    )
    .await
    .expect("the receive pipeline runs");
    recorder
}

fn assert_even_spacing(pts: &[u64], step_ns: u64) {
    for (index, pair) in pts.windows(2).enumerate() {
        assert_eq!(pair[1] - pair[0], step_ns, "PTS step after frame {index}");
    }
}

// byte equality relies on loopback delivering every datagram in order
#[tokio::test]
async fn opusenc_udpsink_into_opus_udpsrc() {
    let reference = encoded_reference().await;
    let packets = reference.frames.len() as u64;
    let (source, port) = bound_opus_source(packets);

    let send = async {
        let mut sink =
            UdpSink::new(format!("127.0.0.1:{port}").parse().unwrap()).with_rtp(PAYLOAD_TYPE, SSRC);
        run_source_transform_sink(
            &mut test_tone(),
            &mut OpusEnc::new(),
            &mut sink,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        )
        .await
        .expect("the send pipeline runs");
        sink.packets_sent()
    };
    let (received, sent) = tokio::time::timeout(LOOPBACK_DEADLINE, async {
        tokio::join!(receive(source), send)
    })
    .await
    .expect("the receiver finishes");

    assert_eq!(sent, packets, "one RTP packet per Opus packet");
    assert_eq!(received.caps.first(), Some(&opus_caps()), "Opus caps");
    assert!(
        received.payloads() == reference.payloads(),
        "the received Opus packets differ from the encoded ones"
    );
    assert_eq!(
        received.pts(),
        reference.pts(),
        "PTS survives the 48 kHz clock"
    );
    assert_eq!(received.pts().first(), Some(&0), "PTS starts at zero");
    assert_even_spacing(&received.pts(), OpusFrameSize::default().nanos());
}

#[tokio::test]
async fn short_and_empty_packets_are_dropped() {
    let mut packetizer = RtpOpusPacketizer::new(PAYLOAD_TYPE, SSRC);
    let frame_ticks = OpusFrameSize::default().samples() as u32;
    let first = [0xFCu8, 0x01, 0x02];
    let second = [0xFCu8, 0x03, 0x04];
    let header_only = packetizer.packetize(&[], 0).remove(0);
    let valid_first = packetizer.packetize(&first, frame_ticks).remove(0);
    // padding bit set, with a padding count longer than the packet
    let mut padding_overrun = packetizer.packetize(&[0xFC], 2 * frame_ticks).remove(0);
    padding_overrun[0] |= 0x20;
    *padding_overrun.last_mut().unwrap() = u8::MAX;
    let valid_second = packetizer.packetize(&second, 3 * frame_ticks).remove(0);

    let (source, port) = bound_opus_source(2);
    let sender = StdUdpSocket::bind("127.0.0.1:0").expect("bind a sender");
    let destination = format!("127.0.0.1:{port}");
    let truncated = &valid_first[..valid_first.len() / 2];
    for datagram in [
        &[][..],
        truncated,
        &header_only,
        &valid_first,
        &padding_overrun,
        &valid_second,
    ] {
        sender.send_to(datagram, &destination).expect("send");
    }

    let received = tokio::time::timeout(LOOPBACK_DEADLINE, receive(source))
        .await
        .expect("the receiver finishes");
    assert_eq!(received.payloads(), [&first[..], &second[..]]);
}

const OPUS_SDP: &str = "v=0\r\n\
    o=- 0 0 IN IP4 127.0.0.1\r\n\
    s=forward\r\n\
    c=IN IP4 127.0.0.1\r\n\
    t=0 0\r\n\
    m=audio 5006 RTP/AVP 111\r\n\
    a=rtpmap:111 opus/48000/2\r\n\
    a=fmtp:111 minptime=10;useinbandfec=1\r\n";

#[tokio::test]
async fn opus_sdp_selects_opus() {
    let mut src = UdpSrc::new("0.0.0.0:5004".parse().unwrap());
    src.set_property("sdp", PropValue::Str(OPUS_SDP.into()))
        .expect("an Opus SDP is receivable");
    assert_eq!(src.intercept_caps().await.unwrap(), opus_caps());
    assert_eq!(src.get_property("port"), Some(PropValue::Uint(5006)));
    assert_eq!(
        src.get_property("encoding-name"),
        Some(PropValue::Str("OPUS".into()))
    );
}

#[tokio::test]
async fn encoding_name_defaults_to_h264_and_opus_refuses_geometry() {
    let mut src = UdpSrc::new("0.0.0.0:5004".parse().unwrap());
    assert_eq!(
        src.get_property("encoding-name"),
        Some(PropValue::Str("H264".into()))
    );
    src.set_property("encoding-name", PropValue::Str("OPUS".into()))
        .expect("OPUS is an encoding name");
    assert_eq!(src.intercept_caps().await.unwrap(), opus_caps());
    assert!(
        src.set_property("width", PropValue::Uint(640)).is_err(),
        "Opus has no geometry to hint"
    );
    assert!(src
        .set_property("encoding-name", PropValue::Str("VP8".into()))
        .is_err());
}

#[test]
fn opus_rtp_launch_lines_build() {
    let registry = default_registry();
    for line in [
        "udpsrc address=127.0.0.1 port=0 encoding-name=OPUS ! opusdec ! fakesink",
        "audiotestsrc ! clocksync ! opusenc ! udpsink host=127.0.0.1 port=5004",
    ] {
        assert!(
            parse_launch(&registry, line).is_ok(),
            "`{line}` builds a graph"
        );
    }
}

// ---- GStreamer reference peer ----

fn have_gstreamer_elements(elements: &[&str]) -> bool {
    elements.iter().all(|element| {
        Command::new("gst-inspect-1.0")
            .args(["--exists", element])
            .status()
            .is_ok_and(|status| status.success())
    })
}

fn wait_for(mut child: Child) -> std::process::Output {
    let deadline = std::time::Instant::now() + PEER_DEADLINE;
    while child.try_wait().expect("poll the peer").is_none() {
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            panic!("the peer did not finish within {PEER_DEADLINE:?}");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    child.wait_with_output().expect("collect the peer output")
}

#[tokio::test]
async fn gst_rtpopuspay_feeds_opus_udpsrc() {
    if !have_gstreamer_elements(&["audiotestsrc", "opusenc", "rtpopuspay", "udpsink"]) {
        eprintln!("gst-launch-1.0 with opusenc / rtpopuspay is not present; skipping");
        return;
    }
    let frame = OpusFrameSize::default();
    let (source, port) = bound_opus_source(GST_PACKETS);
    let peer = Command::new("gst-launch-1.0")
        .arg("-q")
        .args(["audiotestsrc", &format!("num-buffers={GST_PACKETS}")])
        .arg(format!("samplesperbuffer={}", frame.samples()))
        .args(["!", &format!("audio/x-raw,rate={OPUS_RATE_HZ},channels=1")])
        .args([
            "!",
            "opusenc",
            &format!("frame-size={}", frame.property_value()),
        ])
        .args(["!", "rtpopuspay", &format!("pt={PAYLOAD_TYPE}")])
        .args(["!", "udpsink", "host=127.0.0.1", &format!("port={port}")])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("gst-launch-1.0 is on PATH");

    let received = tokio::time::timeout(PEER_DEADLINE, receive(source))
        .await
        .expect("the receiver finishes");
    let output = wait_for(peer);
    assert!(
        output.status.success(),
        "gst sender failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert_eq!(received.frames.len() as u64, GST_PACKETS);
    assert_eq!(received.caps.first(), Some(&opus_caps()));
    assert!(received
        .payloads()
        .iter()
        .all(|payload| !payload.is_empty()));
    let pts = received.pts();
    assert_eq!(pts.first(), Some(&0), "PTS starts at zero");
    // gst opusenc stamps its first packet shorter by the encoder lookahead
    assert!(
        pts[1] > 0 && pts[1] < frame.nanos(),
        "first step {}",
        pts[1]
    );
    assert_even_spacing(&pts[1..], frame.nanos());
}

// `fakesink silent=false` prints this once per buffer under `gst-launch-1.0 -v`
const FAKESINK_BUFFER_LINE: &str = "last-message = chain";

#[tokio::test]
async fn opus_udpsink_feeds_gst_rtpopusdepay() {
    if !have_gstreamer_elements(&["udpsrc", "rtpopusdepay", "opusdec", "fakesink"]) {
        eprintln!("gst-launch-1.0 with rtpopusdepay / opusdec is not present; skipping");
        return;
    }
    let packets = encoded_reference().await.frames.len() as u64;
    let port = StdUdpSocket::bind("127.0.0.1:0")
        .and_then(|socket| socket.local_addr())
        .expect("pick a free port")
        .port();
    let peer = Command::new("gst-launch-1.0")
        .arg("-v")
        .args([
            "udpsrc",
            "address=127.0.0.1",
            &format!("port={port}"),
            &format!("num-buffers={packets}"),
            &format!(
                "caps=application/x-rtp,media=audio,encoding-name=OPUS,\
                 clock-rate={OPUS_CLOCK_HZ},payload={PAYLOAD_TYPE}"
            ),
        ])
        .args(["!", "rtpopusdepay", "!", "opusdec"])
        .args(["!", "fakesink", "silent=false"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("gst-launch-1.0 is on PATH");

    tokio::time::sleep(PEER_LISTEN_WAIT).await;
    let registry = default_registry();
    let line = format!(
        "audiotestsrc num-buffers={SOURCE_BUFFERS} ! opusenc \
         ! udpsink host=127.0.0.1 port={port} payload-type={PAYLOAD_TYPE} ssrc={SSRC}"
    );
    let graph = parse_launch(&registry, &line).expect("the sender pipeline parses");
    tokio::time::timeout(
        LOOPBACK_DEADLINE,
        g2g_core::runtime::run_graph(graph, &ZeroClock, 4),
    )
    .await
    .expect("the sender finishes")
    .expect("the send pipeline runs");

    let output = wait_for(peer);
    let log = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "gst receiver failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let decoded = log
        .lines()
        .filter(|line| line.contains(FAKESINK_BUFFER_LINE))
        .count() as u64;
    assert_eq!(decoded, packets, "gst decoded one buffer per packet: {log}");
}

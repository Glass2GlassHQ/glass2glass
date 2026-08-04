//! M863: concurrent multi-publisher RTSP ingest (`RtspServerSrcN`). One RTSP
//! endpoint, one recording publisher per output pad, all sessions live at the
//! same time: each pad emits exactly the stream its own publisher pushed, and a
//! publisher arriving with every pad busy is refused with 503.
//!
//! The loopback tests drive hand-rolled publishers (one TCP-interleaved, one
//! unicast UDP, concurrently); the ffmpeg interop test (ignored by default)
//! drives the same paths from two reference RTSP clients at once:
//!
//! ```sh
//! cargo test -p g2g-plugins --features rtsp-server \
//!     --test m863_rtsp_ingest_concurrent -- --ignored --nocapture
//! ```
#![cfg(feature = "rtsp-server")]

use core::future::Future;
use core::pin::Pin;
use std::net::{SocketAddr, TcpListener as StdTcpListener};
use std::time::Duration;

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::run_fanout_session;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, MultiOutputSource, OutputSink,
    PipelineClock, PipelinePacket,
};

use g2g_plugins::nalparse::NalCodec;
use g2g_plugins::rtppay::RtpH264Packetizer;
use g2g_plugins::rtspserversrcn::RtspServerSrcN;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// One output pad's downstream: the access units it received (tag byte and PTS)
/// plus its EOS count.
#[derive(Default)]
struct Capture {
    frames: Vec<(u8, u64)>,
    aus: Vec<Vec<u8>>,
    eos: usize,
}

impl Capture {
    fn tags(&self) -> Vec<u8> {
        self.frames.iter().map(|(tag, _)| *tag).collect()
    }

    /// The geometry the SPS in this pad's stream declares, if any carried one.
    fn sps_size(&self) -> Option<(u32, u32)> {
        self.aus
            .iter()
            .find_map(|au| g2g_plugins::h264parse::H264Codec::extract_sps_info(au))
            .map(|sps| (sps.width, sps.height))
    }
}

impl AsyncElement for Capture {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
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
                    if let Some(slice) = frame.domain.as_system_slice() {
                        self.frames
                            .push((slice.get(5).copied().unwrap_or(0), frame.timing.pts_ns));
                        self.aus.push(slice.to_vec());
                    }
                }
                PipelinePacket::Eos => self.eos += 1,
                _ => {}
            }
            Ok(())
        })
    }
}

const SDP: &str =
    "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=g2g\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n";
const URL: &str = "rtsp://127.0.0.1/stream";

/// One small Annex-B IDR access unit, tagged at byte 5.
fn access_unit(tag: u8) -> Vec<u8> {
    vec![0u8, 0, 0, 1, 0x65, tag, 0xAB, 0xCD]
}

/// Frame an RTP packet as interleaved binary (RFC 2326 §10.12).
fn interleaved(channel: u8, rtp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + rtp.len());
    out.push(0x24);
    out.push(channel);
    out.extend_from_slice(&(rtp.len() as u16).to_be_bytes());
    out.extend_from_slice(rtp);
    out
}

/// Read one full RTSP response (header block + any Content-Length body).
async fn read_response(sock: &mut tokio::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 2048];
    loop {
        let n = sock.read(&mut tmp).await.expect("read response");
        assert!(n > 0, "server closed the control connection");
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..pos]).to_string();
            let want_body = head
                .lines()
                .find_map(|l| l.strip_prefix("Content-Length:"))
                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                .unwrap_or(0);
            if buf.len() >= pos + 4 + want_body {
                return String::from_utf8_lossy(&buf).to_string();
            }
        }
    }
}

/// OPTIONS + ANNOUNCE, the prelude both transports share.
async fn announce(rtsp_addr: SocketAddr) -> tokio::net::TcpStream {
    let mut ctrl = tokio::net::TcpStream::connect(rtsp_addr)
        .await
        .expect("connect rtsp");
    ctrl.write_all(format!("OPTIONS {URL} RTSP/1.0\r\nCSeq: 1\r\n\r\n").as_bytes())
        .await
        .unwrap();
    let options = read_response(&mut ctrl).await;
    assert!(options.contains("200 OK"), "a free pad answers: {options}");
    ctrl.write_all(
        format!(
            "ANNOUNCE {URL} RTSP/1.0\r\nCSeq: 2\r\nContent-Type: application/sdp\r\nContent-Length: {}\r\n\r\n{SDP}",
            SDP.len()
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    assert!(read_response(&mut ctrl).await.contains("200 OK"));
    ctrl
}

/// Publish over unicast UDP: the control stream plus an RTP socket already
/// pointed at the server RTP port this session's SETUP advertised.
async fn publish_udp(rtsp_addr: SocketAddr) -> (tokio::net::TcpStream, tokio::net::UdpSocket) {
    let mut ctrl = announce(rtsp_addr).await;
    let rtp = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind publisher rtp");
    let client_rtp_port = rtp.local_addr().unwrap().port();
    ctrl.write_all(
        format!(
            "SETUP {URL}/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP;unicast;client_port={client_rtp_port}-{};mode=record\r\n\r\n",
            client_rtp_port + 1
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    let setup = read_response(&mut ctrl).await;
    let server_port: u16 = setup
        .split("server_port=")
        .nth(1)
        .and_then(|s| s.split(['-', ';', '\r']).next())
        .and_then(|s| s.trim().parse().ok())
        .expect("SETUP advertises a server_port");
    ctrl.write_all(
        format!("RECORD {URL} RTSP/1.0\r\nCSeq: 4\r\nSession: 12345678\r\n\r\n").as_bytes(),
    )
    .await
    .unwrap();
    assert!(read_response(&mut ctrl).await.contains("200 OK"));
    rtp.connect(("127.0.0.1", server_port))
        .await
        .expect("connect rtp dest");
    (ctrl, rtp)
}

/// Publish over TCP-interleaved: the control stream also carries the RTP.
async fn publish_interleaved(rtsp_addr: SocketAddr) -> tokio::net::TcpStream {
    let mut ctrl = announce(rtsp_addr).await;
    ctrl.write_all(
        format!("SETUP {URL}/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1;mode=record\r\n\r\n")
            .as_bytes(),
    )
    .await
    .unwrap();
    let setup = read_response(&mut ctrl).await;
    assert!(setup.contains("interleaved=0-1"), "{setup}");
    ctrl.write_all(
        format!("RECORD {URL} RTSP/1.0\r\nCSeq: 4\r\nSession: 12345678\r\n\r\n").as_bytes(),
    )
    .await
    .unwrap();
    assert!(read_response(&mut ctrl).await.contains("200 OK"));
    ctrl
}

/// Send one access unit as RTP over UDP.
async fn send_udp_au(rtp: &tokio::net::UdpSocket, pktz: &mut RtpH264Packetizer, tag: u8) {
    for pkt in pktz.packetize(&access_unit(tag), tag as u32 * 3000) {
        rtp.send(&pkt).await.expect("send rtp");
    }
}

/// Send one access unit as `$`-framed RTP on the control connection.
async fn send_interleaved_au(
    ctrl: &mut tokio::net::TcpStream,
    pktz: &mut RtpH264Packetizer,
    tag: u8,
) {
    for pkt in pktz.packetize(&access_unit(tag), tag as u32 * 3000) {
        ctrl.write_all(&interleaved(0, &pkt))
            .await
            .expect("send interleaved rtp");
    }
}

fn bound_src(pads: usize) -> (RtspServerSrcN, SocketAddr) {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp");
    let src = RtspServerSrcN::from_listener(listener, pads)
        .expect("adopt listener")
        .with_rtp(96, 0x1234_5678)
        .with_video_size(320, 240);
    let addr = ([127, 0, 0, 1], src.local_port().expect("bound")).into();
    (src, addr)
}

/// Run the two-pad source to completion (within `budget`) through the fan-out
/// runner, reporting each pad's capture alongside the session bookkeeping.
async fn serve(mut src: RtspServerSrcN, budget: Duration) -> (u64, Capture, Capture, u64, u64) {
    assert_eq!(src.output_count(), 2, "these tests drive two pads");
    let mut pad_a = Capture::default();
    let mut pad_b = Capture::default();
    let frames = {
        let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut pad_a, &mut pad_b];
        let stats =
            tokio::time::timeout(budget, run_fanout_session(&mut src, sinks, &ZeroClock, 4))
                .await
                .expect("the source completes within the budget")
                .expect("the source runs");
        stats.frames_consumed
    };
    (
        frames,
        pad_a,
        pad_b,
        src.sessions_served(),
        src.sessions_refused(),
    )
}

/// Two publishers recording at once land on separate pads, and each pad emits
/// exactly its own publisher's access units: no interleaving of the two streams,
/// no session state shared between them. They negotiate different transports, so
/// both receive paths run concurrently.
#[tokio::test]
async fn two_publishers_record_concurrently_on_their_own_pads() {
    let (src, addr) = bound_src(2);
    let src = src.with_frame_limit(5);
    let first: Vec<u8> = (0..5).collect();
    let second: Vec<u8> = (100..105).collect();

    let publishers = async move {
        // Both sessions are live before either sends media.
        let mut ctrl_a = publish_interleaved(addr).await;
        let (ctrl_b, rtp_b) = publish_udp(addr).await;
        let mut pktz_a = RtpH264Packetizer::new(96, 0x1111_1111);
        let mut pktz_b = RtpH264Packetizer::new(96, 0x2222_2222);
        for i in 0..5 {
            send_interleaved_au(&mut ctrl_a, &mut pktz_a, i).await;
            send_udp_au(&rtp_b, &mut pktz_b, 100 + i).await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        drop(ctrl_a);
        drop(ctrl_b);
        drop(rtp_b);
    };

    let (frames, pad_a, pad_b, served, refused) =
        tokio::join!(publishers, serve(src, Duration::from_secs(20))).1;

    assert_eq!(frames, 10, "both publishes were ingested");
    assert_eq!(served, 2, "two publishers recorded at the same time");
    assert_eq!(refused, 0, "neither publisher was turned away");
    let mut got = [pad_a.tags(), pad_b.tags()];
    got.sort();
    assert_eq!(
        got,
        [first, second],
        "each pad carries one publisher's stream, whole and unmixed"
    );
    assert_eq!((pad_a.eos, pad_b.eos), (1, 1), "every pad ends on its EOS");
    for pad in [&pad_a, &pad_b] {
        for pair in pad.frames.windows(2) {
            assert!(
                pair[1].1 > pair[0].1,
                "PTS rises within a pad: {:?}",
                pad.frames
            );
        }
    }
}

/// A third publisher arriving while both pads are recording has nowhere to go, so
/// it is refused with 503 rather than queued, and the two live sessions run on.
#[tokio::test]
async fn a_third_publisher_is_refused_when_every_pad_is_busy() {
    let (src, addr) = bound_src(2);
    let src = src.with_frame_limit(3);

    let publishers = async move {
        let mut ctrl_a = publish_interleaved(addr).await;
        let (ctrl_b, rtp_b) = publish_udp(addr).await;

        let mut third = tokio::net::TcpStream::connect(addr)
            .await
            .expect("a third publisher still reaches the listener");
        third
            .write_all(format!("OPTIONS {URL} RTSP/1.0\r\nCSeq: 7\r\n\r\n").as_bytes())
            .await
            .unwrap();
        let refusal = read_response(&mut third).await;
        assert!(
            refusal.starts_with("RTSP/1.0 503 Service Unavailable"),
            "the third publisher is refused, not queued: {refusal}"
        );
        assert!(refusal.contains("CSeq: 7"), "the refusal echoes CSeq");
        drop(third);

        let mut pktz_a = RtpH264Packetizer::new(96, 0x1111_1111);
        let mut pktz_b = RtpH264Packetizer::new(96, 0x2222_2222);
        for i in 0..3 {
            send_interleaved_au(&mut ctrl_a, &mut pktz_a, i).await;
            send_udp_au(&rtp_b, &mut pktz_b, 100 + i).await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        drop(ctrl_a);
        drop(ctrl_b);
        drop(rtp_b);
    };

    let (frames, pad_a, pad_b, served, refused) =
        tokio::join!(publishers, serve(src, Duration::from_secs(20))).1;

    assert_eq!(
        frames, 6,
        "the two recording sessions ran through the refusal"
    );
    assert_eq!(served, 2);
    assert_eq!(refused, 1, "the publisher with no free pad was refused");
    let mut got = [pad_a.tags(), pad_b.tags()];
    got.sort();
    assert_eq!(got, [vec![0, 1, 2], vec![100, 101, 102]]);
}

/// A publisher that leaves frees its pad without ending the branch: the next one
/// takes the same pad over, its stream continues the pad's sequence, and PTS keeps
/// moving forward.
#[tokio::test]
async fn a_freed_pad_is_taken_over_by_the_next_publisher() {
    let (src, addr) = bound_src(2);
    let src = src.with_frame_limit(4);

    let publishers = async move {
        let mut ctrl_a = publish_interleaved(addr).await;
        let (ctrl_b, rtp_b) = publish_udp(addr).await;
        let mut pktz_a = RtpH264Packetizer::new(96, 0x1111_1111);
        let mut pktz_b = RtpH264Packetizer::new(96, 0x2222_2222);
        for i in 0..2 {
            send_interleaved_au(&mut ctrl_a, &mut pktz_a, i).await;
            send_udp_au(&rtp_b, &mut pktz_b, 100 + i).await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        // The interleaved publisher leaves; a new one takes its pad.
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(ctrl_a);
        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut ctrl_c = publish_interleaved(addr).await;
        let mut pktz_c = RtpH264Packetizer::new(96, 0x3333_3333);
        for i in 0..2 {
            send_interleaved_au(&mut ctrl_c, &mut pktz_c, 10 + i).await;
            send_udp_au(&rtp_b, &mut pktz_b, 102 + i).await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        drop(ctrl_c);
        drop(ctrl_b);
        drop(rtp_b);
    };

    let (frames, pad_a, pad_b, served, refused) =
        tokio::join!(publishers, serve(src, Duration::from_secs(20))).1;

    assert_eq!(frames, 8);
    assert_eq!(served, 3, "three publishers recorded over two pads");
    assert_eq!(refused, 0);
    let mut got = [pad_a.tags(), pad_b.tags()];
    got.sort();
    assert_eq!(
        got,
        [vec![0, 1, 10, 11], vec![100, 101, 102, 103]],
        "the replacement publisher continued its pad's stream"
    );
    let handover = if pad_a.tags() == vec![0, 1, 10, 11] {
        pad_a
    } else {
        pad_b
    };
    for pair in handover.frames.windows(2) {
        assert!(
            pair[1].1 > pair[0].1,
            "PTS keeps moving forward across the handover: {:?}",
            handover.frames
        );
    }
}

/// The pad / session knobs are runtime properties, and the launch registry builds
/// one pad per linked output.
#[test]
fn properties_and_registry_expose_the_pad_count() {
    use g2g_core::{PropValue, PropertySpec};
    let mut src = RtspServerSrcN::new("0.0.0.0:8554".parse().unwrap(), 2);
    let declares = |specs: &[PropertySpec], name: &str| specs.iter().any(|s| s.name == name);
    assert!(declares(
        MultiOutputSource::properties(&src),
        "max-sessions"
    ));
    assert!(declares(MultiOutputSource::properties(&src), "timeout"));

    src.set_property("max-sessions", PropValue::Uint(4))
        .unwrap();
    assert_eq!(src.get_property("max-sessions"), Some(PropValue::Uint(4)));
    assert_eq!(src.output_count(), 4, "the property is the pad count");
    assert!(src
        .set_property("max-sessions", PropValue::Uint(0))
        .is_err());
    src.set_property("timeout", PropValue::Uint(15)).unwrap();
    assert_eq!(src.get_property("timeout"), Some(PropValue::Uint(15)));
    src.set_property("port", PropValue::Uint(9554)).unwrap();
    assert_eq!(src.get_property("port"), Some(PropValue::Uint(9554)));

    let reg = g2g_plugins::registry::default_registry();
    assert!(reg.is_fanout_src("rtspserversrcn"));
    let built = reg
        .make_fanout_src("rtspserversrcn", 3)
        .expect("registry builds the element");
    assert_eq!(built.output_count(), 3, "one pad per linked output");
}

/// Reference peer: two ffmpeg publishes running *at the same time* into one
/// source, each at its own resolution. Every pad must carry exactly one of them,
/// which the in-band SPS geometry proves.
#[tokio::test]
#[ignore = "needs ffmpeg with RTSP; opens local TCP + UDP sockets"]
async fn two_ffmpeg_publishers_record_concurrently() {
    if std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_err()
    {
        eprintln!("skipping: ffmpeg is not on PATH");
        return;
    }
    let (src, addr) = bound_src(2);
    let src = src.with_frame_limit(15);
    let url = format!("rtsp://127.0.0.1:{}/stream", addr.port());

    let wide = url.clone();
    let first = tokio::task::spawn_blocking(move || ffmpeg_publish(&wide, "tcp", "320x240", 10));
    let small = tokio::task::spawn_blocking(move || ffmpeg_publish(&url, "udp", "176x144", 10));

    let publishers = async {
        let (a, b) = tokio::join!(first, small);
        (a.unwrap(), b.unwrap())
    };
    let (frames, pad_a, pad_b, served, refused) =
        tokio::join!(publishers, serve(src, Duration::from_secs(60))).1;

    assert_eq!(served, 2, "both ffmpeg publishes recorded at once");
    assert_eq!(refused, 0, "two publishers fit in two pads");
    assert_eq!(frames, 30, "each pad ingested its num-buffers");
    let mut sizes = [pad_a.sps_size(), pad_b.sps_size()];
    sizes.sort();
    assert_eq!(
        sizes,
        [Some((176, 144)), Some((320, 240))],
        "each pad carries one publisher's stream, told apart by its SPS geometry"
    );
}

/// Publish `secs` seconds of `size` H.264 to `url` with ffmpeg. The element stops
/// reading once its pads hit `num-buffers`, so the exit status is not meaningful.
fn ffmpeg_publish(url: &str, transport: &str, size: &str, secs: u32) -> bool {
    std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-re",
            "-an",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc=size={size}:rate=15:duration={secs}"),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            // in-band SPS/PPS, so the ingested access units carry the geometry
            // (ffmpeg otherwise leaves them to the SDP alone)
            "-bsf:v",
            "dump_extra",
            "-f",
            "rtsp",
            "-rtsp_transport",
            transport,
            url,
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

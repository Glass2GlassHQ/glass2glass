//! M1079: raw (non-RTP) UDP byte streams, the broadcast `mpegtsmux ! udpsink`
//! and `udpsrc port=5000 ! tsdemux` pair, plus the `clients` fan-out gst splits
//! into `multiudpsink`.
//!
//! Every loopback runs a g2g sender and a g2g receiver on `127.0.0.1`. UDP
//! guarantees neither order nor delivery, so the byte-equality assertions rest
//! on the loopback interface delivering in order and dropping nothing: the
//! receiving socket is bound before the sender starts, and the fixture is small
//! enough to sit in the kernel receive buffer.
//!
//! The `#[ignore]`d interop legs pair each element with `gst-launch-1.0`, which
//! a g2g <-> g2g loopback cannot check: both ends would share a bug. Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features udp-ingress,udp-egress \
//!     --test m1079_raw_udp -- --ignored --nocapture
//! ```
#![cfg(all(feature = "udp-ingress", feature = "udp-egress"))]

use std::net::UdpSocket as StdUdpSocket;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::Duration;

use core::future::Future;
use core::pin::Pin;

use g2g_core::runtime::{parse_launch, run_graph, run_simple_pipeline, LatencyProfile, SourceLoop};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, ConfigureOutcome, G2gError, OutputSink, PipelineClock,
    PipelinePacket, PropValue,
};

use g2g_plugins::bytestream::{datagram_chunk, TS_DATAGRAM_PAYLOAD, TS_PACKET_SIZE};
use g2g_plugins::filesink::FileSink;
use g2g_plugins::registry::default_registry;
use g2g_plugins::udpsrc::UdpSrc;

/// The stream every leg carries: a small MPEG-TS clip, the container raw UDP
/// broadcast actually carries.
const FIXTURE: &str = "tests/fixtures/av_h264_aac44100.ts";

/// The `bytestream-format` value naming that container.
const TS_FORMAT: &str = "mpegts";

/// `udpsink`'s `max-payload` default, the cap the sender splits on.
const DEFAULT_MAX_PAYLOAD: usize = 1400;

/// Long enough that a loopback transfer of the fixture cannot legitimately still
/// be running, short enough that a deadlock is reported instead of hanging CI.
const LOOPBACK_DEADLINE: Duration = Duration::from_secs(20);

/// How long a `gst-launch-1.0` peer gets to bind its socket before the g2g side
/// starts sending. UDP drops what nobody is listening for, so the receiver of
/// each interop leg is always started first and given this long.
const PEER_LISTEN_WAIT: Duration = Duration::from_millis(1500);

/// How long a peer process gets to finish writing its output file.
const PEER_DEADLINE: Duration = Duration::from_secs(60);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE)
}

fn output_path(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    let _ = std::fs::remove_file(&path);
    path
}

/// Datagrams the sender splits the fixture into, so a receiver can ask for
/// exactly that many `num-buffers` and then emit EOS.
fn datagram_count() -> u64 {
    let len = std::fs::metadata(fixture_path())
        .expect("stat the fixture")
        .len() as usize;
    let chunk = datagram_chunk(ByteStreamEncoding::MpegTs, DEFAULT_MAX_PAYLOAD);
    len.div_ceil(chunk) as u64
}

/// A `UdpSrc` in raw mode on an already-bound ephemeral port, so the sender
/// knows where to send and the socket is queueing before the transfer starts.
fn bound_raw_source(frame_limit: u64) -> (UdpSrc, u16) {
    let socket = StdUdpSocket::bind("127.0.0.1:0").expect("bind a receiver");
    let src = UdpSrc::from_socket(socket)
        .expect("adopt the socket")
        .with_bytestream(ByteStreamEncoding::MpegTs)
        .with_frame_limit(frame_limit);
    let port = src.local_port().expect("the bound port");
    (src, port)
}

/// A port nothing is listening on, for the legs where a peer process binds and
/// the port has to be named up front.
fn free_port() -> u16 {
    let socket = StdUdpSocket::bind("127.0.0.1:0").expect("bind an ephemeral port");
    socket.local_addr().expect("bound address").port()
}

async fn send_fixture_to(destination: String) {
    let registry = default_registry();
    let graph = parse_launch(
        &registry,
        &format!(
            "filesrc location={} bytestream-format={TS_FORMAT} ! udpsink {destination}",
            fixture_path().display()
        ),
    )
    .expect("the sender pipeline parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the send pipeline runs");
}

fn assert_identical(received: &Path) {
    let sent_bytes = std::fs::read(fixture_path()).expect("read the fixture");
    let received_bytes = std::fs::read(received).expect("read the received file");
    assert_eq!(
        received_bytes.len(),
        sent_bytes.len(),
        "the whole byte stream arrived"
    );
    assert!(
        received_bytes == sent_bytes,
        "the received bytes differ from the sent ones"
    );
}

/// `filesrc ! udpsink` into `udpsrc bytestream-format=mpegts ! filesink` is the
/// fixture back, byte for byte.
#[tokio::test]
async fn raw_udp_loopback_is_byte_exact() {
    let out = output_path("g2g_m1079_loopback.ts");
    let (mut source, port) = bound_raw_source(datagram_count());
    let mut recorder = FileSink::new(&out);

    let receive = tokio::time::timeout(
        LOOPBACK_DEADLINE,
        run_simple_pipeline(
            &mut source,
            &mut recorder,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    );
    let send = send_fixture_to(format!("host=127.0.0.1 port={port}"));

    let (received, ()) = tokio::join!(receive, send);
    let stats = received
        .expect("the receiver finishes")
        .expect("the receive pipeline runs");

    assert_eq!(
        stats.frames_emitted,
        datagram_count(),
        "one DataFrame per datagram"
    );
    assert!(recorder.eos_seen(), "num-buffers ended the stream on Eos");
    assert_identical(&out);
}

/// Records the byte length of every frame it is handed.
#[derive(Default)]
struct SizeSink {
    sizes: Vec<usize>,
}

impl AsyncElement for SizeSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, caps: &Caps) -> Result<Caps, G2gError> {
        Ok(caps.clone())
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                if let Some(slice) = frame.domain.as_system_slice() {
                    self.sizes.push(slice.len());
                }
            }
            Ok(())
        })
    }
}

/// Every datagram but the last carries a whole number of TS packets, so a
/// receiver never has to stitch one back together across two of them.
#[tokio::test]
async fn datagrams_carry_whole_ts_packets() {
    let (mut source, port) = bound_raw_source(datagram_count());
    let mut sizes = SizeSink::default();

    let receive = tokio::time::timeout(
        LOOPBACK_DEADLINE,
        run_simple_pipeline(
            &mut source,
            &mut sizes,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    );
    let send = send_fixture_to(format!("host=127.0.0.1 port={port}"));
    let (received, ()) = tokio::join!(receive, send);
    received
        .expect("the receiver finishes")
        .expect("the receive pipeline runs");

    let chunk = datagram_chunk(ByteStreamEncoding::MpegTs, DEFAULT_MAX_PAYLOAD);
    let total = std::fs::metadata(fixture_path()).unwrap().len() as usize;
    assert_eq!(chunk % TS_PACKET_SIZE, 0, "the split is packet aligned");
    assert_eq!(
        sizes.sizes.iter().sum::<usize>(),
        total,
        "{:?}",
        sizes.sizes
    );
    for (index, size) in sizes.sizes.iter().enumerate() {
        assert_eq!(size % TS_PACKET_SIZE, 0, "datagram {index} is {size} bytes");
        assert!(*size <= chunk, "datagram {index} exceeds max-payload");
    }
}

/// The received datagrams demux exactly like the file does: `udpsrc ! tsdemux`
/// yields the frame count `filesrc ! tsdemux` yields for the same fixture.
#[tokio::test]
async fn raw_udpsrc_feeds_tsdemux() {
    let registry = default_registry();
    let from_file = parse_launch(
        &registry,
        &format!(
            "filesrc location={} bytestream-format={TS_FORMAT} ! tsdemux stream=h264 ! fakesink",
            fixture_path().display()
        ),
    )
    .expect("the file pipeline parses");
    let expected = run_graph(from_file, &ZeroClock, 4)
        .await
        .expect("the file pipeline runs")
        .frames_consumed;
    assert!(expected > 0, "the fixture demuxes to video frames");

    let port = free_port();
    let count = datagram_count();
    let over_udp = parse_launch(
        &registry,
        &format!(
            "udpsrc address=127.0.0.1 port={port} bytestream-format={TS_FORMAT} \
             num-buffers={count} ! tsdemux stream=h264 ! fakesink"
        ),
    )
    .expect("the receiver pipeline parses");
    // The receiver binds its socket on its first poll, so the sender waits for
    // it: UDP drops what nobody is listening for.
    let send = async {
        tokio::time::sleep(PEER_LISTEN_WAIT).await;
        send_fixture_to(format!("host=127.0.0.1 port={port}")).await;
    };
    let (received, ()) = tokio::time::timeout(LOOPBACK_DEADLINE, async {
        tokio::join!(run_graph(over_udp, &ZeroClock, 4), send)
    })
    .await
    .expect("the receiver finishes");
    let stats = received.expect("the receive pipeline runs");
    assert_eq!(
        stats.frames_consumed, expected,
        "the demuxed frame count matches the file path"
    );
}

/// `clients=host:port,host:port` delivers the whole stream to both receivers.
#[tokio::test]
async fn clients_fan_the_stream_out_to_two_receivers() {
    let first = output_path("g2g_m1079_client_a.ts");
    let second = output_path("g2g_m1079_client_b.ts");
    let (mut source_a, port_a) = bound_raw_source(datagram_count());
    let (mut source_b, port_b) = bound_raw_source(datagram_count());
    let mut recorder_a = FileSink::new(&first);
    let mut recorder_b = FileSink::new(&second);

    let receive_a = run_simple_pipeline(
        &mut source_a,
        &mut recorder_a,
        &ZeroClock,
        LatencyProfile::Live.link_capacity(),
    );
    let receive_b = run_simple_pipeline(
        &mut source_b,
        &mut recorder_b,
        &ZeroClock,
        LatencyProfile::Live.link_capacity(),
    );
    let send = send_fixture_to(format!("clients=127.0.0.1:{port_a},127.0.0.1:{port_b}"));

    let both = tokio::time::timeout(LOOPBACK_DEADLINE, async {
        tokio::join!(receive_a, receive_b, send)
    })
    .await
    .expect("both receivers finish");
    both.0.expect("the first receive pipeline runs");
    both.1.expect("the second receive pipeline runs");

    assert_identical(&first);
    assert_identical(&second);
}

/// The RTP-only knobs are refused once `bytestream-format` puts the source in
/// raw mode, so a launch line cannot look configured and be ignored.
#[test]
fn raw_mode_rejects_the_rtp_only_properties() {
    let mut source = UdpSrc::new("127.0.0.1:0".parse().unwrap());
    source
        .set_property("bytestream-format", PropValue::Str(TS_FORMAT.into()))
        .expect("raw mode is settable");
    for (name, value) in [
        ("sdp", PropValue::Str("v=0\r\n".into())),
        ("jitter-latency", PropValue::Uint(20)),
        ("jitter-depth", PropValue::Uint(8)),
        ("nack", PropValue::Bool(false)),
        ("rtcp-rr-interval", PropValue::Uint(500)),
        ("rtx-payload-type", PropValue::Uint(97)),
        ("rtx-apt", PropValue::Uint(96)),
        ("fec-payload-type", PropValue::Uint(98)),
        ("flexfec-payload-type", PropValue::Uint(99)),
        ("width", PropValue::Uint(640)),
        ("height", PropValue::Uint(480)),
    ] {
        assert!(
            source.set_property(name, value).is_err(),
            "`{name}` must be refused in raw mode"
        );
    }
    // The knobs that still mean something stay settable.
    source
        .set_property("port", PropValue::Uint(5000))
        .expect("port applies in both modes");
    source
        .set_property("auto-multicast", PropValue::Bool(false))
        .expect("auto-multicast applies in both modes");
}

/// Unset `bytestream-format` leaves the source on the RTP path it has always
/// taken, RTP-only properties included.
#[test]
fn rtp_mode_is_still_the_default() {
    let mut source = UdpSrc::new("127.0.0.1:0".parse().unwrap());
    assert_eq!(
        source.get_property("bytestream-format"),
        Some(PropValue::Str(String::new())),
        "no container declared means RTP"
    );
    source
        .set_property("jitter-latency", PropValue::Uint(20))
        .expect("the jitter buffer is configurable in RTP mode");
    assert_eq!(
        source.get_property("jitter-latency"),
        Some(PropValue::Uint(20))
    );
}

/// Both raw names build from a launch line, `multiudpsink` through its alias.
#[test]
fn raw_names_build_from_a_launch_line() {
    let registry = default_registry();
    for line in [
        &format!("udpsrc address=127.0.0.1 port=0 bytestream-format={TS_FORMAT} ! fakesink"),
        &format!(
            "filesrc location={} bytestream-format={TS_FORMAT} ! udpsink host=127.0.0.1 port=5004",
            fixture_path().display()
        ),
        &format!(
            "filesrc location={} bytestream-format={TS_FORMAT} \
             ! multiudpsink clients=127.0.0.1:5004,127.0.0.1:5006",
            fixture_path().display()
        ),
    ] {
        let line: &str = line;
        assert!(
            parse_launch(&registry, line).is_ok(),
            "`{line}` builds a graph"
        );
    }
}

// ---- real-peer interop (ignored: needs gst-launch-1.0 / ffmpeg) ----

fn spawn(program: &str, arguments: &str) -> Child {
    Command::new(program)
        .args(arguments.split_whitespace())
        .spawn()
        .unwrap_or_else(|_| panic!("{program} is on PATH"))
}

fn wait_for(mut child: Child) {
    let deadline = std::time::Instant::now() + PEER_DEADLINE;
    loop {
        match child.try_wait().expect("poll the peer") {
            Some(status) => {
                assert!(status.success(), "the peer exited with {status}");
                return;
            }
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                panic!("the peer did not finish within {PEER_DEADLINE:?}");
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

/// Packets a peer's `ffprobe -count_packets` reports for `path`.
fn packet_count(path: &Path) -> u64 {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .expect("ffprobe is on PATH");
    assert!(output.status.success(), "ffprobe failed on {path:?}");
    // The csv writer emits a blank line per unselected stream, so take the one
    // row that carries the count.
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .map(str::trim)
        .find_map(|line| line.parse().ok())
        .unwrap_or_else(|| panic!("ffprobe reported no packet count for {path:?}: {text:?}"))
}

/// How long a counting pass waits for the next datagram before calling the
/// peer's stream finished. A `-re` paced peer sends the small fixture in about
/// a second, so a gap this long is the end of it.
const PEER_QUIET: Duration = Duration::from_secs(5);

/// Datagrams one run of a peer sends. ffmpeg splits on its own muxer
/// boundaries, not on `pkt_size` alone, so the count cannot be derived from the
/// fixture the way the g2g sender's can; the receive leg needs it to know when
/// the stream is over.
fn count_peer_datagrams(program: &str, arguments: &dyn Fn(u16) -> String) -> u64 {
    let socket = StdUdpSocket::bind("127.0.0.1:0").expect("bind a counter");
    socket
        .set_read_timeout(Some(PEER_QUIET))
        .expect("set a read timeout");
    let port = socket.local_addr().expect("bound address").port();
    let peer = spawn(program, &arguments(port));
    let mut buf = [0u8; 65_535];
    let mut datagrams = 0;
    while socket.recv_from(&mut buf).is_ok() {
        datagrams += 1;
    }
    wait_for(peer);
    assert!(datagrams > 0, "{program} sent nothing");
    datagrams
}

/// A real sender: gst pushes the fixture into raw UDP a datagram at a time and
/// g2g records it byte for byte. `blocksize` is what makes gst's split
/// predictable, so the receiver knows how many datagrams to expect.
#[tokio::test]
#[ignore = "needs gst-launch-1.0"]
async fn gst_udpsink_feeds_raw_udpsrc() {
    let out = output_path("g2g_m1079_gst_sends.ts");
    let (mut source, port) = bound_raw_source(datagram_count());
    let mut recorder = FileSink::new(&out);

    let peer = spawn(
        "gst-launch-1.0",
        &format!(
            "-q filesrc blocksize={TS_DATAGRAM_PAYLOAD} location={} \
             ! udpsink host=127.0.0.1 port={port}",
            fixture_path().display()
        ),
    );

    tokio::time::timeout(
        PEER_DEADLINE,
        run_simple_pipeline(
            &mut source,
            &mut recorder,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("the receiver finishes")
    .expect("the receive pipeline runs");
    wait_for(peer);
    assert_identical(&out);
}

/// A real receiver: g2g pushes the fixture into raw UDP and gst records it byte
/// for byte. The receiver starts first, since UDP drops what nobody awaits.
#[tokio::test]
#[ignore = "needs gst-launch-1.0"]
async fn raw_udpsink_feeds_gst_udpsrc() {
    let out = output_path("g2g_m1079_gst_receives.ts");
    let port = free_port();
    let peer = spawn(
        "gst-launch-1.0",
        &format!(
            "-q -e udpsrc address=127.0.0.1 port={port} num-buffers={} ! filesink location={}",
            datagram_count(),
            out.display()
        ),
    );
    tokio::time::sleep(PEER_LISTEN_WAIT).await;

    send_fixture_to(format!("host=127.0.0.1 port={port}")).await;
    wait_for(peer);
    assert_identical(&out);
}

/// ffmpeg is the other reference sender, over the `pkt_size=1316` MPEG-TS
/// framing raw UDP broadcast uses. It is run twice: once to learn how many
/// datagrams its muxer emits, once for the receive leg itself.
#[tokio::test]
#[ignore = "needs ffmpeg and ffprobe"]
async fn ffmpeg_udp_feeds_raw_udpsrc() {
    let out = output_path("g2g_m1079_ffmpeg_sends.ts");
    let stream_to = |port: u16| {
        format!(
            "-hide_banner -loglevel error -re -i {} -c copy -f mpegts \
             udp://127.0.0.1:{port}?pkt_size={TS_DATAGRAM_PAYLOAD}",
            fixture_path().display()
        )
    };
    let datagrams = count_peer_datagrams("ffmpeg", &stream_to);

    let (mut source, port) = bound_raw_source(datagrams);
    let mut recorder = FileSink::new(&out);
    let peer = spawn("ffmpeg", &stream_to(port));

    tokio::time::timeout(
        PEER_DEADLINE,
        run_simple_pipeline(
            &mut source,
            &mut recorder,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("the receiver finishes")
    .expect("the receive pipeline runs");
    wait_for(peer);

    assert_eq!(
        packet_count(&out),
        packet_count(&fixture_path()),
        "the recorded stream carries the source's packets"
    );
}

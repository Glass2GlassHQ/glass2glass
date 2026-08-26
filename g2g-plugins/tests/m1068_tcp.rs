//! M1068: the plain TCP byte-stream elements. Each loopback pairs a g2g client
//! with a g2g server over `127.0.0.1` on an OS-picked port (`port=0`, read back
//! through `current-port`) and asserts the received file is byte-for-byte the
//! sent one. One side of every pair is built with `parse_launch`, so the
//! registry names and property parsing are exercised too; the server side is
//! driven directly, because its port is only known once it has bound and a
//! `parse_launch` graph gives no handle on the element inside it.
//!
//! The four `#[ignore]`d interop tests pair each element with its GStreamer
//! counterpart, which a g2g <-> g2g loopback cannot check: both ends would share
//! a bug. They need `gst-launch-1.0`. Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features tcp --test m1068_tcp -- --ignored --nocapture
//! ```
#![cfg(feature = "tcp")]

use std::net::TcpListener as StdTcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::Duration;

use g2g_core::runtime::{parse_launch, run_graph, run_simple_pipeline, LatencyProfile};
use g2g_core::{ByteStreamEncoding, Caps, PipelineClock};

use g2g_plugins::filesink::FileSink;
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::registry::default_registry;
use g2g_plugins::tcp::{TcpServerSink, TcpServerSrc};

/// The transfer both loopbacks carry: a small MPEG-TS clip, so the declared
/// `bytestream-format` default matches what actually crosses the wire.
const FIXTURE: &str = "tests/fixtures/av_h264_aac44100.ts";

/// Long enough that a loopback transfer of the fixture cannot legitimately still
/// be running, short enough that a deadlock is reported instead of hanging CI.
const LOOPBACK_DEADLINE: Duration = Duration::from_secs(20);

/// How long a `gst-launch-1.0` peer gets to bind its listening socket before the
/// g2g client dials it. A probe connection cannot be used to detect readiness:
/// `tcpserversrc` accepts exactly one client and `tcpserversink` would start
/// serving the probe.
const GST_LISTEN_WAIT: Duration = Duration::from_millis(1000);

/// Microseconds gst `identity` sleeps before each buffer in the interop case
/// where gst holds the listening socket and pushes the file into it. Its
/// `tcpserversink` serves a joining client from the latest buffer, so this has to
/// exceed [`GST_LISTEN_WAIT`] by enough that the g2g client is attached before
/// the very first push.
const GST_PUSH_PACING_US: u64 = 1_500_000;

/// How long a gst peer gets to finish writing its output file.
const GST_DEADLINE: Duration = Duration::from_secs(60);

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

fn ts_bytestream() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

/// A port nothing is listening on, for the cases where the gst peer binds and we
/// have to name the port up front.
fn free_port() -> u16 {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
    listener.local_addr().expect("bound address").port()
}

fn assert_identical(sent: &Path, received: &Path) {
    let sent_bytes = std::fs::read(sent).expect("read the sent file");
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

#[tokio::test]
async fn tcpclientsink_reaches_tcpserversrc() {
    let fixture = fixture_path();
    let out = output_path("g2g_m1068_client_to_server.ts");

    let mut server = TcpServerSrc::new("127.0.0.1", 0);
    let port = server.bind().expect("bind an ephemeral port");
    assert_ne!(port, 0, "`port=0` reports the port the OS picked");
    assert_eq!(server.current_port(), Some(port));

    let mut recorder = FileSink::new(&out);
    let receive = tokio::time::timeout(
        LOOPBACK_DEADLINE,
        run_simple_pipeline(
            &mut server,
            &mut recorder,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    );

    let send = async {
        let registry = default_registry();
        let graph = parse_launch(
            &registry,
            &format!(
                "filesrc location={} ! tcpclientsink host=127.0.0.1 port={port}",
                fixture.display()
            ),
        )
        .expect("the sender pipeline parses");
        run_graph(graph, &ZeroClock, 4).await
    };

    let (received, sent) = tokio::join!(receive, send);
    let stats = received
        .expect("the receiver finishes")
        .expect("the receive pipeline runs");
    sent.expect("the send pipeline runs");

    assert!(stats.frames_emitted > 0, "the source emitted chunks");
    assert!(recorder.eos_seen(), "the stream ended on Eos");
    assert_identical(&fixture, &out);
}

#[tokio::test]
async fn tcpserversink_reaches_tcpclientsrc() {
    let fixture = fixture_path();
    let out = output_path("g2g_m1068_server_to_client.ts");

    let mut server = TcpServerSink::new("127.0.0.1", 0);
    let port = server.bind().expect("bind an ephemeral port");
    assert_ne!(port, 0, "`port=0` reports the port the OS picked");
    assert_eq!(server.current_port(), Some(port));

    let mut player = FileSrc::new(&fixture, ts_bytestream());
    let send = tokio::time::timeout(
        LOOPBACK_DEADLINE,
        run_simple_pipeline(
            &mut player,
            &mut server,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    );

    let receive = async {
        let registry = default_registry();
        let graph = parse_launch(
            &registry,
            &format!(
                "tcpclientsrc host=127.0.0.1 port={port} ! filesink location={}",
                out.display()
            ),
        )
        .expect("the receiver pipeline parses");
        run_graph(graph, &ZeroClock, 4).await
    };

    let (sent, received) = tokio::join!(send, receive);
    sent.expect("the sender finishes")
        .expect("the send pipeline runs");
    let stats = received.expect("the receive pipeline runs");

    assert!(stats.frames_consumed > 0, "the sink recorded chunks");
    assert_eq!(
        server.bytes_written(),
        std::fs::metadata(&fixture).unwrap().len(),
        "the sink wrote the whole file to its one client"
    );
    assert_identical(&fixture, &out);
}

/// Every one of the four names resolves in the default registry and takes its
/// `host` / `port` from a launch line. `port=0` keeps the parse from touching a
/// port a parallel test might want.
#[test]
fn the_four_elements_build_from_a_launch_line() {
    let registry = default_registry();
    for line in [
        "tcpserversrc host=127.0.0.1 port=0 ! fakesink",
        "tcpclientsrc host=127.0.0.1 port=0 ! fakesink",
        "audiotestsrc num-buffers=1 ! tcpserversink host=127.0.0.1 port=0",
        "audiotestsrc num-buffers=1 ! tcpclientsink host=127.0.0.1 port=0",
    ] {
        assert!(
            parse_launch(&registry, line).is_ok(),
            "`{line}` builds a graph"
        );
    }
}

// ---- real-peer interop against gst-launch-1.0 (ignored: needs GStreamer) ----

fn spawn_gst(description: &str) -> Child {
    Command::new("gst-launch-1.0")
        .arg("-q")
        .args(description.split_whitespace())
        .spawn()
        .expect("gst-launch-1.0 is on PATH")
}

fn wait_for_gst(mut child: Child) {
    let deadline = std::time::Instant::now() + GST_DEADLINE;
    loop {
        match child.try_wait().expect("poll the gst peer") {
            Some(status) => {
                assert!(status.success(), "the gst peer exited with {status}");
                return;
            }
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                panic!("the gst peer did not finish within {GST_DEADLINE:?}");
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

/// g2g serves, gst reads: `tcpserversink` -> `gst tcpclientsrc ! filesink`.
#[tokio::test]
#[ignore = "needs gst-launch-1.0"]
async fn tcpserversink_feeds_gst_tcpclientsrc() {
    let fixture = fixture_path();
    let out = output_path("g2g_m1068_gst_reads_our_server.ts");

    let mut server = TcpServerSink::new("127.0.0.1", 0);
    let port = server.bind().expect("bind an ephemeral port");
    let peer = spawn_gst(&format!(
        "tcpclientsrc host=127.0.0.1 port={port} ! filesink location={}",
        out.display()
    ));

    let mut player = FileSrc::new(&fixture, ts_bytestream());
    tokio::time::timeout(
        GST_DEADLINE,
        run_simple_pipeline(
            &mut player,
            &mut server,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("the sender finishes")
    .expect("the send pipeline runs");

    wait_for_gst(peer);
    assert_identical(&fixture, &out);
}

/// g2g dials, gst serves: `tcpclientsink` -> `gst tcpserversrc ! filesink`.
#[tokio::test]
#[ignore = "needs gst-launch-1.0"]
async fn tcpclientsink_feeds_gst_tcpserversrc() {
    let fixture = fixture_path();
    let out = output_path("g2g_m1068_gst_serves_our_client_sink.ts");
    let port = free_port();

    let peer = spawn_gst(&format!(
        "tcpserversrc host=127.0.0.1 port={port} ! filesink location={}",
        out.display()
    ));
    tokio::time::sleep(GST_LISTEN_WAIT).await;

    let registry = default_registry();
    let graph = parse_launch(
        &registry,
        &format!(
            "filesrc location={} ! tcpclientsink host=127.0.0.1 port={port}",
            fixture.display()
        ),
    )
    .expect("the sender pipeline parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the send pipeline runs");

    wait_for_gst(peer);
    assert_identical(&fixture, &out);
}

/// gst dials, g2g serves: `gst filesrc ! tcpclientsink` -> `tcpserversrc`.
#[tokio::test]
#[ignore = "needs gst-launch-1.0"]
async fn tcpserversrc_reads_gst_tcpclientsink() {
    let fixture = fixture_path();
    let out = output_path("g2g_m1068_gst_dials_our_server.ts");

    let mut server = TcpServerSrc::new("127.0.0.1", 0);
    let port = server.bind().expect("bind an ephemeral port");
    let peer = spawn_gst(&format!(
        "filesrc location={} ! tcpclientsink host=127.0.0.1 port={port}",
        fixture.display()
    ));

    let mut recorder = FileSink::new(&out);
    tokio::time::timeout(
        GST_DEADLINE,
        run_simple_pipeline(
            &mut server,
            &mut recorder,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("the receiver finishes")
    .expect("the receive pipeline runs");

    wait_for_gst(peer);
    assert_identical(&fixture, &out);
}

/// gst serves, g2g dials: `gst filesrc ! tcpserversink` -> `tcpclientsrc`.
#[tokio::test]
#[ignore = "needs gst-launch-1.0"]
async fn tcpclientsrc_reads_gst_tcpserversink() {
    let fixture = fixture_path();
    let out = output_path("g2g_m1068_gst_serves_our_client_src.ts");
    let port = free_port();

    let peer = spawn_gst(&format!(
        "filesrc location={} ! identity sleep-time={GST_PUSH_PACING_US} ! \
         tcpserversink host=127.0.0.1 port={port}",
        fixture.display()
    ));
    tokio::time::sleep(GST_LISTEN_WAIT).await;

    let registry = default_registry();
    let graph = parse_launch(
        &registry,
        &format!(
            "tcpclientsrc host=127.0.0.1 port={port} ! filesink location={}",
            out.display()
        ),
    )
    .expect("the receiver pipeline parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the receive pipeline runs");

    wait_for_gst(peer);
    assert_identical(&fixture, &out);
}

//! M1081: the shared-memory IPC pair. The loopbacks run a g2g `shmsink` and a
//! g2g `shmsrc` over one control socket and assert the bytes and the frame count
//! come through; the codec tests cover what a hostile peer can put on that
//! socket.
//!
//! The two `#[ignore]`d interop tests pair each element with its GStreamer
//! counterpart, which a g2g <-> g2g loopback cannot check: both ends would share
//! a bug. They need `gst-launch-1.0` with gst-plugins-bad. Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features shm --test m1081_shm -- --ignored --nocapture
//! ```
#![cfg(all(unix, feature = "shm"))]

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use g2g_core::runtime::{parse_launch, run_graph, run_simple_pipeline, LatencyProfile};
use g2g_core::PipelineClock;

use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::registry::default_registry;
use g2g_plugins::shm::{ShmSink, ShmSrc};
use g2g_plugins::shmpipe::{buffer_range, Command as ShmCommand, COMMAND_BYTES};
use g2g_plugins::videotestsrc::VideoTestSrc;

/// The transfer the byte-exact loopback carries.
const FIXTURE: &str = "tests/fixtures/av_h264_aac44100.ts";

/// Small enough that the fixture cycles the allocator many times instead of
/// fitting whole, so a block is only reused once the source has acknowledged it.
const TEST_SHM_SIZE: usize = 65_536;

/// Long enough that a loopback of the fixture cannot legitimately still be
/// running, short enough that a deadlock is reported instead of hanging CI.
const LOOPBACK_DEADLINE: Duration = Duration::from_secs(30);

/// How long a `gst-launch-1.0` peer gets to finish.
const GST_DEADLINE: Duration = Duration::from_secs(60);

/// The `videotestsrc` geometry the frame-count tests use.
const TEST_WIDTH: u32 = 320;
const TEST_HEIGHT: u32 = 240;
const TEST_FRAMERATE: u32 = 30;
const TEST_FRAMES: u64 = 20;
/// Frames the gst interop legs move, more than [`TEST_FRAMES`] so a truncated
/// run cannot pass as a short one.
const INTEROP_FRAMES: u64 = 30;
/// Bytes per pixel of what `videotestsrc` draws.
const RGBA_BYTES_PER_PIXEL: usize = 4;
/// Bytes per pixel of I420, as the numerator over 2.
const I420_HALF_BYTES_PER_PIXEL: usize = 3;
/// One `videotestsrc` frame at the test geometry.
const RGBA_FRAME_BYTES: usize = TEST_WIDTH as usize * TEST_HEIGHT as usize * RGBA_BYTES_PER_PIXEL;
/// Area for the video tests: three frames, so the allocator still has to
/// recycle a block the source has acknowledged rather than never filling.
const VIDEO_SHM_SIZE: usize = RGBA_FRAME_BYTES * 3;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE)
}

/// A path under the temp dir nothing else in this run uses, so the tests can
/// share a process.
fn unique_path(name: &str) -> PathBuf {
    static SERIAL: AtomicU32 = AtomicU32::new(0);
    let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("g2g_m1081_{name}_{}_{serial}", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

fn rgba_caps_text() -> String {
    format!(
        "video/x-raw,format=rgba,width={TEST_WIDTH},height={TEST_HEIGHT},framerate={TEST_FRAMERATE}/1"
    )
}

/// A sink already holding its control socket, for the pairings where the peer
/// is started by hand rather than by `run_graph`.
fn opened_sink(socket_path: &Path, shm_size: usize) -> ShmSink {
    let mut sink = ShmSink::new(socket_path).with_shm_size(shm_size);
    sink.open().expect("the control socket and shm area open");
    sink
}

/// `filesrc ! shmsink` into `shmsrc ! filesink`, byte for byte, both sides from
/// a launch line.
#[tokio::test]
async fn shmsink_feeds_shmsrc_byte_exactly() {
    let fixture = fixture_path();
    let socket_path = unique_path("loopback.sock");
    let out = unique_path("loopback.ts");
    let registry = default_registry();

    let send = async {
        let graph = parse_launch(
            &registry,
            &format!(
                "filesrc location={} ! shmsink socket-path={} shm-size={TEST_SHM_SIZE}",
                fixture.display(),
                socket_path.display()
            ),
        )
        .expect("the sender pipeline parses");
        run_graph(graph, &ZeroClock, 4).await
    };

    let receive = async {
        let graph = parse_launch(
            &registry,
            &format!(
                "shmsrc socket-path={} bytestream-format=mpegts ! filesink location={}",
                socket_path.display(),
                out.display()
            ),
        )
        .expect("the receiver pipeline parses");
        run_graph(graph, &ZeroClock, 4).await
    };

    let (sent, received) =
        tokio::time::timeout(LOOPBACK_DEADLINE, async { tokio::join!(send, receive) })
            .await
            .expect("the loopback finishes");
    sent.expect("the send pipeline runs");
    let stats = received.expect("the receive pipeline runs");
    assert!(stats.frames_consumed > 0, "the sink recorded chunks");

    let sent_bytes = std::fs::read(&fixture).expect("read the sent file");
    let received_bytes = std::fs::read(&out).expect("read the received file");
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

/// `videotestsrc ! shmsink` into `shmsrc caps=... ! capsfilter ! fakesink`: the
/// frames arrive whole, and the caps the source was told to declare are what
/// negotiates downstream. The `capsfilter` naming the same caps is the
/// assertion: the graph would not solve if `shmsrc` declared anything else.
#[tokio::test]
async fn shmsrc_declares_its_caps_downstream() {
    let socket_path = unique_path("caps.sock");
    let registry = default_registry();
    let caps_text = rgba_caps_text();

    let send = async {
        let graph = parse_launch(
            &registry,
            &format!(
                "videotestsrc num-buffers={TEST_FRAMES} width={TEST_WIDTH} height={TEST_HEIGHT} \
                 framerate={TEST_FRAMERATE}/1 ! shmsink socket-path={} shm-size={VIDEO_SHM_SIZE}",
                socket_path.display()
            ),
        )
        .expect("the sender pipeline parses");
        run_graph(graph, &ZeroClock, 4).await
    };

    let receive = async {
        let graph = parse_launch(
            &registry,
            &format!(
                "shmsrc socket-path={} caps={caps_text} ! capsfilter caps={caps_text} ! fakesink",
                socket_path.display()
            ),
        )
        .expect("the receiver pipeline parses");
        run_graph(graph, &ZeroClock, 4).await
    };

    let (sent, received) =
        tokio::time::timeout(LOOPBACK_DEADLINE, async { tokio::join!(send, receive) })
            .await
            .expect("the loopback finishes");
    sent.expect("the send pipeline runs");
    let stats = received.expect("the receive pipeline runs");
    assert_eq!(
        stats.frames_consumed, TEST_FRAMES,
        "every frame arrived under the declared caps"
    );
}

/// Frames cross whole rather than split at a block boundary: the same count of
/// frames arrives as was sent, and they weigh exactly that many whole frames.
/// The area holds only three, so the run turns the allocator over.
#[tokio::test]
async fn shmsrc_emits_whole_frames() {
    let socket_path = unique_path("frames.sock");
    let out = unique_path("frames.rgba");
    let registry = default_registry();

    let send = async {
        let graph = parse_launch(
            &registry,
            &format!(
                "videotestsrc num-buffers={TEST_FRAMES} width={TEST_WIDTH} height={TEST_HEIGHT} \
                 framerate={TEST_FRAMERATE}/1 ! shmsink socket-path={} shm-size={VIDEO_SHM_SIZE}",
                socket_path.display()
            ),
        )
        .expect("the sender pipeline parses");
        run_graph(graph, &ZeroClock, 4).await
    };

    let receive = async {
        let graph = parse_launch(
            &registry,
            &format!(
                "shmsrc socket-path={} caps={} ! filesink location={}",
                socket_path.display(),
                rgba_caps_text(),
                out.display()
            ),
        )
        .expect("the receiver pipeline parses");
        run_graph(graph, &ZeroClock, 4).await
    };

    let (sent, received) =
        tokio::time::timeout(LOOPBACK_DEADLINE, async { tokio::join!(send, receive) })
            .await
            .expect("the loopback finishes");
    sent.expect("the send pipeline runs");
    let stats = received.expect("the receive pipeline runs");

    assert_eq!(stats.frames_consumed, TEST_FRAMES, "every frame arrived");
    let written = std::fs::metadata(&out)
        .expect("the receiver wrote a file")
        .len();
    assert_eq!(
        written,
        RGBA_FRAME_BYTES as u64 * TEST_FRAMES,
        "those frames weigh exactly {TEST_FRAMES} whole frames"
    );
}

/// Both names resolve in the default registry and take their properties from a
/// launch line.
#[test]
fn both_elements_build_from_a_launch_line() {
    let registry = default_registry();
    let socket_path = unique_path("launch.sock");
    for line in [
        format!(
            "shmsrc socket-path={} bytestream-format=mpegts num-buffers=1 ! fakesink",
            socket_path.display()
        ),
        format!(
            "videotestsrc num-buffers=1 ! shmsink socket-path={} shm-size=65536 \
             wait-for-connection=false perms=384 buffer-time=1000000",
            socket_path.display()
        ),
    ] {
        assert!(
            parse_launch(&registry, &line).is_ok(),
            "`{line}` builds a graph"
        );
    }
}

// ---- the command codec, which every byte a peer sends goes through ----

/// The wire struct is the 24 bytes a C probe of `shmpipe.c`'s `CommandBuffer`
/// reports on this ABI, so a gst peer's write is exactly one command.
#[test]
#[cfg(target_pointer_width = "64")]
fn a_command_is_the_size_the_c_struct_is() {
    assert_eq!(COMMAND_BYTES, 24);
}

#[test]
fn a_truncated_or_overlong_command_is_rejected() {
    let encoded = ShmCommand::NewBuffer {
        area_id: 1,
        offset: 0,
        size: 16,
    }
    .encode();
    assert_eq!(
        ShmCommand::decode(&encoded[..COMMAND_BYTES - 1]),
        None,
        "a short read is not a command"
    );
    let mut overlong = Vec::from(encoded);
    overlong.push(0);
    assert_eq!(
        ShmCommand::decode(&overlong),
        None,
        "two partial commands are not one command"
    );
    assert!(
        ShmCommand::decode(&encoded).is_some(),
        "the whole command still decodes"
    );
}

#[test]
fn a_buffer_outside_the_area_is_rejected() {
    let area_len = 4096usize;
    assert_eq!(
        buffer_range(area_len, 0, area_len as u64),
        Some(0..area_len)
    );
    assert_eq!(
        buffer_range(area_len, area_len as u64 - 8, 16),
        None,
        "a buffer running past the end is rejected"
    );
    assert_eq!(
        buffer_range(area_len, area_len as u64 + 1, 8),
        None,
        "an offset past the end is rejected"
    );
    assert_eq!(
        buffer_range(area_len, u64::MAX, 8),
        None,
        "an offset plus size that wraps is rejected"
    );
    assert_eq!(
        buffer_range(area_len, 0, 0),
        None,
        "an empty buffer is not a buffer"
    );
}

// ---- real-peer interop against gst-launch-1.0 (ignored: needs GStreamer) ----

fn spawn_gst(description: &str) -> Child {
    Command::new("gst-launch-1.0")
        .arg("-q")
        .args(description.split_whitespace())
        .spawn()
        .expect("gst-launch-1.0 is on PATH")
}

/// Wait for the gst peer to exit, without asking it to exit cleanly. The gst
/// `shm` elements have no end-of-stream on the wire: `shmsrc` reports a closed
/// control socket as a read error, and `shmsink`'s poll thread errors on
/// teardown, so both exit nonzero against a gst peer as well. What the transfer
/// moved is what the caller asserts on.
fn wait_for_gst(mut child: Child) {
    let deadline = std::time::Instant::now() + GST_DEADLINE;
    loop {
        match child.try_wait().expect("poll the gst peer") {
            Some(_) => return,
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                panic!("the gst peer did not finish within {GST_DEADLINE:?}");
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

/// gst serves, g2g reads: `gst videotestsrc ! shmsink` -> `shmsrc ! fakesink`.
#[tokio::test]
#[ignore = "needs gst-launch-1.0 with gst-plugins-bad"]
async fn shmsrc_reads_gst_shmsink() {
    let socket_path = unique_path("gst_sink.sock");
    let frame_bytes = TEST_WIDTH as usize * TEST_HEIGHT as usize * I420_HALF_BYTES_PER_PIXEL / 2;

    let peer = spawn_gst(&format!(
        "videotestsrc num-buffers={INTEROP_FRAMES} ! \
         video/x-raw,format=I420,width={TEST_WIDTH},height={TEST_HEIGHT} ! \
         shmsink socket-path={} wait-for-connection=true",
        socket_path.display()
    ));

    let mut source = ShmSrc::new(&socket_path);
    let mut recorder = FakeSink::new();
    tokio::time::timeout(
        GST_DEADLINE,
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

    wait_for_gst(peer);
    assert_eq!(
        recorder.received(),
        INTEROP_FRAMES,
        "every gst frame arrived, {frame_bytes} bytes each"
    );
    assert!(
        source.shm_area_name().is_some(),
        "the gst sink announced its shm area"
    );
}

/// g2g serves, gst reads: `videotestsrc ! shmsink` -> `gst shmsrc ! filesink`.
#[tokio::test]
#[ignore = "needs gst-launch-1.0 with gst-plugins-bad"]
async fn shmsink_feeds_gst_shmsrc() {
    let socket_path = unique_path("gst_src.sock");
    let out = unique_path("gst_src.rgba");
    let mut sink = opened_sink(&socket_path, VIDEO_SHM_SIZE);
    let peer = spawn_gst(&format!(
        "shmsrc socket-path={} ! \
         video/x-raw,format=RGBA,width={TEST_WIDTH},height={TEST_HEIGHT},framerate={TEST_FRAMERATE}/1 ! \
         filesink location={}",
        socket_path.display(),
        out.display()
    ));

    let mut source = VideoTestSrc::new(TEST_WIDTH, TEST_HEIGHT, TEST_FRAMERATE, INTEROP_FRAMES);
    tokio::time::timeout(
        GST_DEADLINE,
        run_simple_pipeline(
            &mut source,
            &mut sink,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("the sender finishes")
    .expect("the send pipeline runs");

    wait_for_gst(peer);
    let written = std::fs::metadata(&out)
        .expect("the gst peer wrote a file")
        .len();
    assert_eq!(
        written,
        RGBA_FRAME_BYTES as u64 * INTEROP_FRAMES,
        "gst wrote every frame whole"
    );
}

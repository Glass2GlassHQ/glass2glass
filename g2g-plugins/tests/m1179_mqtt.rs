#![cfg(feature = "mqtt")]
//! M1179: `mqttsink` and `mqttsrc` against a real broker. mosquitto runs on a
//! free port with a password file. For the sink, `mosquitto_sub` subscribes and
//! the assertions read what it received; for the source, `mosquitto_pub` sends
//! and the assertions read the frames that came out. Both check what the broker
//! logged about the connection, never the element's own counters alone.
//! Self-skips without the mosquitto binaries (CI installs them).

use std::io::{BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AnalyticsMeta, BBox, BlobMeta, Caps, Colorimetry, Dim, G2gError, Interlace, ObjectDetection,
    OutputSink, PropValue, PushOutcome, Rate, RawVideoFormat, TextFormat,
};
use g2g_plugins::mqttsink::MqttSink;
use g2g_plugins::mqttsrc::MqttSrc;
use serde_json::{json, Value};

const Q16_SHIFT: u32 = 16;
const RECORD_FPS: u32 = 30;
const FRAME_WIDTH: u32 = 320;
const FRAME_HEIGHT: u32 = 240;
const CLASS_NAMES: &[&str] = &["person", "car"];
/// The frame carrying a detection: `(pts_ns, label, x, y, w, h, score)`, the
/// box normalized in exact binary fractions.
const DETECTED: (u64, u32, f32, f32, f32, f32, f32) =
    (1_000_000_000, 0, 0.25, 0.5, 0.125, 0.25, 0.875);
/// The frame carrying a JSON blob and nothing else.
const BLOB_PTS_NS: u64 = 2_000_000_000;
const BLOB_HEADER: &str = "note";
const BLOB_PAYLOAD: &[u8] = b"{\"camera\":\"north\"}";
/// A frame with nothing on it, which publishes nothing.
const BARE_PTS_NS: u64 = 3_000_000_000;
const NS_PER_SECOND: f64 = 1_000_000_000.0;

const TOPIC: &str = "g2g/m1179/records";
const CLIENT_ID: &str = "g2g-m1179-sink";
/// The control topic the source test subscribes to and the messages sent on it.
const CONTROL_TOPIC: &str = "g2g/m1179/control";
const SOURCE_CLIENT_ID: &str = "g2g-m1179-src";
const CONTROL_MESSAGES: &[&str] = &[
    "{\"command\":\"start\"}",
    "{\"command\":\"snapshot\"}",
    "{\"command\":\"stop\"}",
];
const USERNAME: &str = "tester";
const PASSWORD: &str = "sesame";
const QOS: u64 = 1;
/// How long the broker gets to come up, the subscriber to register, and the
/// subscriber to receive everything.
const PEER_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Default)]
struct RecordingSink;

impl OutputSink for RecordingSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn rgba_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(FRAME_WIDTH),
        height: Dim::Fixed(FRAME_HEIGHT),
        framerate: Rate::Fixed(RECORD_FPS << Q16_SHIFT),
        interlace: Interlace::Progressive,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

fn black_frame(pts_ns: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 4].into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        sequence: pts_ns,
        meta: Default::default(),
    }
}

fn detected_frame() -> Frame {
    let (pts_ns, label, x, y, w, h, score) = DETECTED;
    let mut frame = black_frame(pts_ns);
    let mut analytics = AnalyticsMeta::new();
    analytics.set_class_names(CLASS_NAMES.iter().copied());
    analytics.add_detection(ObjectDetection {
        bbox: BBox { x, y, w, h },
        label,
        confidence: score,
    });
    frame.meta.attach(analytics);
    frame
}

fn blob_frame() -> Frame {
    let mut frame = black_frame(BLOB_PTS_NS);
    let mut blobs = BlobMeta::new();
    blobs.push(BLOB_HEADER, Vec::from(BLOB_PAYLOAD));
    frame.meta.attach(blobs);
    frame
}

/// The records the subscriber must receive, in order: the detection in whole
/// pixels of the negotiated geometry, then the blob under its header.
fn expected_records() -> Vec<Value> {
    let (pts_ns, label, x, y, w, h, score) = DETECTED;
    let pixels = |normalized: f32, span: u32| (normalized * span as f32).round() as i64;
    vec![
        json!({
            "pts": pts_ns as f64 / NS_PER_SECOND,
            "detections": [{
                "label": CLASS_NAMES[label as usize],
                "x": pixels(x, FRAME_WIDTH),
                "y": pixels(y, FRAME_HEIGHT),
                "w": pixels(w, FRAME_WIDTH),
                "h": pixels(h, FRAME_HEIGHT),
                "score": f64::from(score),
            }],
        }),
        json!({
            "pts": BLOB_PTS_NS as f64 / NS_PER_SECOND,
            BLOB_HEADER: serde_json::from_slice::<Value>(BLOB_PAYLOAD).expect("the blob is JSON"),
        }),
    ]
}

fn have_mosquitto() -> bool {
    [
        "mosquitto",
        "mosquitto_sub",
        "mosquitto_pub",
        "mosquitto_passwd",
    ]
    .iter()
    .all(|binary| Command::new(binary).arg("--help").output().is_ok())
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind an ephemeral port")
        .local_addr()
        .expect("local address")
        .port()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m1179-{}-{name}", std::process::id()))
}

/// A child process killed when the test ends, however it ends.
struct BrokerProcess(Child);

impl Drop for BrokerProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// mosquitto on `port`, accepting only `USERNAME` / `PASSWORD`, with its log
/// collected line by line.
fn start_broker(port: u16) -> (BrokerProcess, Arc<Mutex<Vec<String>>>) {
    let password_file = temp_path(&format!("{port}-passwd"));
    let status = Command::new("mosquitto_passwd")
        .args(["-b", "-c"])
        .arg(&password_file)
        .args([USERNAME, PASSWORD])
        .status()
        .expect("mosquitto_passwd runs");
    assert!(
        status.success(),
        "mosquitto_passwd writes the password file"
    );
    let config_file = temp_path(&format!("{port}-mosquitto.conf"));
    std::fs::write(
        &config_file,
        format!(
            "listener {port} 127.0.0.1\nallow_anonymous false\npassword_file {}\nlog_dest stderr\nlog_type all\n",
            password_file.display()
        ),
    )
    .expect("write the broker config");
    let mut child = Command::new("mosquitto")
        .arg("-c")
        .arg(&config_file)
        .arg("-v")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("mosquitto starts");
    let stderr = child.stderr.take().expect("piped stderr");
    let log = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&log);
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            sink.lock().expect("log lock").push(line);
        }
    });
    let deadline = Instant::now() + PEER_TIMEOUT;
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(
            Instant::now() < deadline,
            "the broker listens within the timeout"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    (BrokerProcess(child), log)
}

async fn wait_for_log(log: &Mutex<Vec<String>>, needle: &str) {
    let deadline = Instant::now() + PEER_TIMEOUT;
    loop {
        if log
            .lock()
            .expect("log lock")
            .iter()
            .any(|line| line.contains(needle))
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the broker logs {needle:?} within the timeout"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn mqttsink_publishes_each_record_to_a_mosquitto_subscriber() {
    if !have_mosquitto() {
        eprintln!("skipping: the mosquitto binaries are not installed");
        return;
    }
    let port = free_port();
    let (_broker, log) = start_broker(port);
    let expected = expected_records();
    let subscriber = Command::new("mosquitto_sub")
        .args(["-h", "127.0.0.1", "-p", &port.to_string()])
        .args(["-u", USERNAME, "-P", PASSWORD])
        .args(["-t", TOPIC, "-q", &QOS.to_string()])
        .args(["-C", &expected.len().to_string()])
        .args(["-W", &PEER_TIMEOUT.as_secs().to_string()])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("mosquitto_sub starts");
    wait_for_log(&log, "Received SUBSCRIBE").await;

    let mut sink = MqttSink::new()
        .with_host("127.0.0.1")
        .with_port(port)
        .with_topic(TOPIC);
    for (name, value) in [
        ("client-id", PropValue::Str(String::from(CLIENT_ID))),
        ("username", PropValue::Str(String::from(USERNAME))),
        ("password", PropValue::Str(String::from(PASSWORD))),
        ("qos", PropValue::Uint(QOS)),
    ] {
        sink.set_property(name, value)
            .expect("the property is settable");
    }
    sink.configure_pipeline(&rgba_caps())
        .expect("mqttsink accepts the caps");
    let mut out = RecordingSink;
    for frame in [detected_frame(), blob_frame(), black_frame(BARE_PTS_NS)] {
        sink.process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .expect("mqttsink queues the record");
    }
    sink.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("mqttsink drains on eos");
    assert_eq!(
        sink.published(),
        expected.len() as u64,
        "the bare frame publishes nothing"
    );
    assert_eq!(sink.dropped(), 0);

    let output = tokio::task::spawn_blocking(move || subscriber.wait_with_output())
        .await
        .expect("join")
        .expect("mosquitto_sub exits");
    assert!(
        output.status.success(),
        "mosquitto_sub received every message before its timeout"
    );
    let received: Vec<Value> = String::from_utf8(output.stdout)
        .expect("payloads are UTF-8")
        .lines()
        .map(|line| serde_json::from_str(line).expect("each payload is one JSON object"))
        .collect();
    assert_eq!(received, expected);

    let log = log.lock().expect("log lock");
    let connected = log
        .iter()
        .find(|line| line.contains("New client connected") && line.contains(CLIENT_ID))
        .expect("the broker saw the sink's client id");
    assert!(
        connected.contains(&format!("u'{USERNAME}'")),
        "the broker saw the sink log in as {USERNAME}: {connected}"
    );
    let publishes: Vec<&String> = log
        .iter()
        .filter(|line| line.contains("Received PUBLISH from") && line.contains(CLIENT_ID))
        .collect();
    assert_eq!(publishes.len(), expected.len());
    for line in publishes {
        assert!(
            line.contains(&format!("q{QOS}, r0")) && line.contains(&format!("'{TOPIC}'")),
            "each publish carries the sink's qos, retain and topic: {line}"
        );
    }
}

#[derive(Default)]
struct CollectingSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectingSink {
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

/// Publish `message` on the control topic through the reference client, at
/// QoS 1 so the call returns only once the broker has it.
async fn publish_control(port: u16, message: &'static str) {
    let status = tokio::task::spawn_blocking(move || {
        Command::new("mosquitto_pub")
            .args(["-h", "127.0.0.1", "-p", &port.to_string()])
            .args(["-u", USERNAME, "-P", PASSWORD])
            .args(["-t", CONTROL_TOPIC, "-q", &QOS.to_string(), "-m", message])
            .status()
    })
    .await
    .expect("join")
    .expect("mosquitto_pub runs");
    assert!(status.success(), "mosquitto_pub delivers {message}");
}

#[tokio::test]
async fn mqttsrc_emits_each_published_message_as_a_text_frame() {
    if !have_mosquitto() {
        eprintln!("skipping: the mosquitto binaries are not installed");
        return;
    }
    let port = free_port();
    let (_broker, log) = start_broker(port);

    let mut source = MqttSrc::new()
        .with_host("127.0.0.1")
        .with_port(port)
        .with_topic(CONTROL_TOPIC)
        .with_message_limit(CONTROL_MESSAGES.len() as u64);
    for (name, value) in [
        ("client-id", PropValue::Str(String::from(SOURCE_CLIENT_ID))),
        ("username", PropValue::Str(String::from(USERNAME))),
        ("password", PropValue::Str(String::from(PASSWORD))),
        ("qos", PropValue::Uint(QOS)),
    ] {
        source
            .set_property(name, value)
            .expect("the property is settable");
    }
    let caps = source
        .intercept_caps()
        .await
        .expect("the source names its caps");
    assert_eq!(
        caps,
        Caps::Text {
            format: TextFormat::Utf8
        }
    );
    source
        .configure_pipeline(&caps)
        .expect("mqttsrc accepts its own caps");

    let mut out = CollectingSink::default();
    let publisher = async {
        wait_for_log(&log, &format!("Received SUBSCRIBE from {SOURCE_CLIENT_ID}")).await;
        for message in CONTROL_MESSAGES {
            publish_control(port, message).await;
        }
    };
    let (run, ()) = tokio::join!(source.run(&mut out), publisher);
    let emitted = run.expect("the source ends at its message limit");
    assert_eq!(emitted, CONTROL_MESSAGES.len() as u64);
    assert_eq!(source.received(), emitted);

    let payloads: Vec<&str> = out
        .packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(
                core::str::from_utf8(frame.domain.as_system_slice().expect("host bytes"))
                    .expect("the payload is UTF-8"),
            ),
            _ => None,
        })
        .collect();
    assert_eq!(
        payloads, CONTROL_MESSAGES,
        "one frame per message, in order"
    );
    assert!(
        matches!(out.packets.last(), Some(PipelinePacket::Eos)),
        "the limit ends the stream with Eos"
    );

    let log = log.lock().expect("log lock");
    let connected = log
        .iter()
        .find(|line| line.contains("New client connected") && line.contains(SOURCE_CLIENT_ID))
        .expect("the broker saw the source's client id");
    assert!(
        connected.contains(&format!("u'{USERNAME}'")),
        "the broker saw the source log in as {USERNAME}: {connected}"
    );
    assert!(
        log.iter()
            .any(|line| line.contains(&format!("{CONTROL_TOPIC} (QoS {QOS})"))),
        "the broker saw the subscription at the source's qos"
    );
}

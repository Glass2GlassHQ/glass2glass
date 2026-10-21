// M1784: livekitdatasink and livekitdatasrc against a real LiveKit SFU, with the
// official LiveKit Python SDK (tools/livekit-data-peer.py) as the other participant.
// Needs a server and uv on PATH, the first run downloads the SDK wheels:
//
//   livekit-server --dev &   # release binary from github.com/livekit/livekit
//   G2G_LIVEKIT_URL=ws://127.0.0.1:7880 G2G_LIVEKIT_API_KEY=devkey G2G_LIVEKIT_API_SECRET=secret \
//       cargo test -p g2g-plugins --features webrtc-livekit --test m1784_livekit_data -- --ignored

#![cfg(all(target_os = "linux", feature = "webrtc-livekit"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use tokio::sync::mpsc;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{AsyncElement, FrameTiming, MemoryDomain, PipelinePacket};
use g2g_plugins::livekit_signal::DataPacketKind;
use g2g_plugins::livekitdatasink::LiveKitDataSink;
use g2g_plugins::livekitdatasrc::LiveKitDataSrc;

use collect_sink::CollectSink;

const TOPIC: &str = "input";
const RELIABLE_LENGTHS: [usize; 6] = [1, 7, 64, 255, 1200, 12000];
// each lossy payload fits one SCTP chunk, one lost chunk drops a whole message
const LOSSY_LENGTHS: [usize; 5] = [1, 7, 64, 255, 1200];
const WARMUP_PAYLOAD: &[u8] = b"warmup";
const STRAY_PAYLOAD: &[u8] = b"for nobody";
// the peer exits after this long without a packet
const PEER_IDLE_SECONDS: &str = "15";
const PEER_EVENT_TIMEOUT: Duration = Duration::from_secs(60);
const SOURCE_RUN_TIMEOUT: Duration = Duration::from_secs(60);
const PEER_SCRIPT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/livekit-data-peer.py");

struct Server {
    url: String,
    api_key: String,
    api_secret: String,
}

impl Server {
    fn from_env() -> Self {
        let var = |name: &str| {
            std::env::var(name)
                .unwrap_or_else(|_| panic!("{name} is required, see the module docs"))
        };
        Self {
            url: var("G2G_LIVEKIT_URL"),
            api_key: var("G2G_LIVEKIT_API_KEY"),
            api_secret: var("G2G_LIVEKIT_API_SECRET"),
        }
    }
}

fn unique_room(direction: &str) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    format!("m1784-{direction}-{}-{now}", std::process::id())
}

#[derive(Debug, Clone)]
struct PeerLine {
    event: String,
    fields: HashMap<String, String>,
}

impl PeerLine {
    fn parse(line: &str) -> Option<Self> {
        let mut words = line.strip_prefix("peer ")?.split(' ');
        let event = words.next()?.to_string();
        let fields = words
            .filter_map(|word| word.split_once('='))
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        Some(Self { event, fields })
    }

    fn field(&self, key: &str) -> &str {
        self.fields
            .get(key)
            .unwrap_or_else(|| panic!("{self:?} has no {key}"))
    }

    fn payload(&self) -> Vec<u8> {
        let hex = self.field("payload");
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }
}

struct Peer {
    child: Child,
    lines: mpsc::UnboundedReceiver<PeerLine>,
}

impl Peer {
    fn spawn(server: &Server, room: &str, identity: &str, mode: &[&str]) -> Self {
        let mut child = Command::new("uv")
            .args([
                "run",
                "-q",
                "--with",
                "livekit",
                "--with",
                "livekit-api",
                "python",
            ])
            .arg(PEER_SCRIPT)
            .args(["--url", &server.url, "--api-key", &server.api_key])
            .args(["--api-secret", &server.api_secret, "--room", room])
            .args(["--identity", identity, "--timeout", PEER_IDLE_SECONDS])
            .args(mode)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("uv runs the LiveKit Python peer");
        let stdout = child.stdout.take().unwrap();
        let (tx, lines) = mpsc::unbounded_channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Some(line) = PeerLine::parse(&line) {
                    if tx.send(line).is_err() {
                        return;
                    }
                }
            }
        });
        Self { child, lines }
    }

    async fn next(&mut self) -> PeerLine {
        tokio::time::timeout(PEER_EVENT_TIMEOUT, self.lines.recv())
            .await
            .expect("the peer prints its next line in time")
            .expect("the peer exited early")
    }

    async fn wait_for(&mut self, event: &str) -> PeerLine {
        loop {
            let line = self.next().await;
            if line.event == event {
                return line;
            }
        }
    }

    async fn finish(mut self) -> Vec<PeerLine> {
        let mut lines = Vec::new();
        loop {
            let line = self.next().await;
            if line.event == "done" {
                break;
            }
            lines.push(line);
        }
        let status = self.child.wait().unwrap();
        assert!(status.success(), "the peer exited with {status}");
        lines
    }
}

fn lengths_arg(lengths: &[usize]) -> String {
    lengths
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn data_payloads(packets: &[PipelinePacket]) -> Vec<Vec<u8>> {
    packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame.domain.as_system_slice()?.to_vec()),
            _ => None,
        })
        .collect()
}

fn frame(payload: Vec<u8>, sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
        FrameTiming::default(),
        sequence,
    ))
}

fn pattern(length: usize, seed: u8) -> Vec<u8> {
    (0..length)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

#[tokio::test]
#[ignore = "needs a LiveKit server (G2G_LIVEKIT_URL + api key/secret) and uv"]
async fn python_packets_reach_livekitdatasrc() {
    const SOURCE_IDENTITY: &str = "g2g-datasrc";
    const PEER_IDENTITY: &str = "py-sender";
    let server = Server::from_env();
    let room = unique_room("to-g2g");
    let reliable_lengths = lengths_arg(&RELIABLE_LENGTHS);
    let lossy_lengths = lengths_arg(&LOSSY_LENGTHS);
    let peer = Peer::spawn(
        &server,
        &room,
        PEER_IDENTITY,
        &[
            "send",
            "--target",
            SOURCE_IDENTITY,
            "--topic",
            TOPIC,
            "--reliable-lengths",
            &reliable_lengths,
            "--lossy-lengths",
            &lossy_lengths,
        ],
    );
    let expected_count = (RELIABLE_LENGTHS.len() + LOSSY_LENGTHS.len()) as u64;
    let mut source = LiveKitDataSrc::new(&server.url, &room, SOURCE_IDENTITY)
        .with_api_key(&server.api_key, &server.api_secret)
        .with_topic(TOPIC)
        .with_sender_identity(PEER_IDENTITY)
        .with_message_limit(expected_count);
    let mut out = CollectSink::default();
    let run = tokio::time::timeout(SOURCE_RUN_TIMEOUT, source.run(&mut out)).await;
    let sent = peer.finish().await;

    let sent_to_source = |kind: &str| -> Vec<Vec<u8>> {
        sent.iter()
            .filter(|line| line.event == "sent" && line.field("kind") == kind)
            .filter(|line| line.field("topic") == TOPIC)
            .filter(|line| line.field("destinations") == SOURCE_IDENTITY)
            .map(PeerLine::payload)
            .collect()
    };
    let expected_reliable = sent_to_source("reliable");
    let expected_lossy = sent_to_source("lossy");
    assert_eq!(expected_reliable.len(), RELIABLE_LENGTHS.len());
    assert_eq!(expected_lossy.len(), LOSSY_LENGTHS.len());
    let filtered: Vec<Vec<u8>> = sent
        .iter()
        .filter(|line| line.event == "sent")
        .map(PeerLine::payload)
        .filter(|payload| !expected_reliable.contains(payload) && !expected_lossy.contains(payload))
        .collect();
    assert!(
        !filtered.is_empty(),
        "the peer sends packets the filters drop"
    );

    let received = data_payloads(&out.packets);
    assert_eq!(run.ok(), Some(Ok(expected_count)), "received {received:?}");
    assert!(matches!(
        out.packets.first(),
        Some(PipelinePacket::CapsChanged(_))
    ));
    assert!(matches!(out.packets.last(), Some(PipelinePacket::Eos)));
    let received_reliable: Vec<Vec<u8>> = received
        .iter()
        .filter(|payload| expected_reliable.contains(payload))
        .cloned()
        .collect();
    assert_eq!(received_reliable, expected_reliable, "reliable, in order");
    for payload in &expected_lossy {
        assert!(received.contains(payload), "lossy payload {payload:02x?}");
    }
    for payload in &filtered {
        assert!(
            !received.contains(payload),
            "filtered payload {payload:02x?}"
        );
    }
    assert_eq!(
        source.emitted_from(DataPacketKind::Reliable),
        expected_reliable.len() as u64
    );
    assert_eq!(
        source.emitted_from(DataPacketKind::Lossy),
        expected_lossy.len() as u64
    );
}

#[tokio::test]
#[ignore = "needs a LiveKit server (G2G_LIVEKIT_URL + api key/secret) and uv"]
async fn livekitdatasink_packets_reach_python() {
    const PEER_IDENTITY: &str = "py-receiver";
    const RELIABLE_IDENTITY: &str = "g2g-datasink-reliable";
    const LOSSY_IDENTITY: &str = "g2g-datasink-lossy";
    const STRAY_IDENTITY: &str = "g2g-datasink-stray";
    const RELIABLE_SEED: u8 = 1;
    const LOSSY_SEED: u8 = 2;
    let server = Server::from_env();
    let room = unique_room("from-g2g");
    // one warmup packet from each of the two sinks the peer receives from
    let peer_count = (2 + RELIABLE_LENGTHS.len() + LOSSY_LENGTHS.len()).to_string();
    let mut peer = Peer::spawn(
        &server,
        &room,
        PEER_IDENTITY,
        &["receive", "--count", &peer_count],
    );
    peer.wait_for("ready").await;

    let sink = |identity: &str| {
        LiveKitDataSink::new(&server.url, &room, identity)
            .with_api_key(&server.api_key, &server.api_secret)
            .with_topic(TOPIC)
    };
    let mut reliable = sink(RELIABLE_IDENTITY).with_destination_identities([PEER_IDENTITY]);
    let mut lossy = sink(LOSSY_IDENTITY).with_reliable(false);
    let mut stray = sink(STRAY_IDENTITY).with_destination_identities(["nobody"]);
    let mut out = CollectSink::default();
    for element in [&mut reliable, &mut lossy] {
        element
            .process(frame(WARMUP_PAYLOAD.to_vec(), 0), &mut out)
            .await
            .unwrap();
    }
    stray
        .process(frame(STRAY_PAYLOAD.to_vec(), 0), &mut out)
        .await
        .unwrap();

    let mut joined = Vec::new();
    while !(joined.iter().any(|identity| identity == RELIABLE_IDENTITY)
        && joined.iter().any(|identity| identity == LOSSY_IDENTITY))
    {
        let line = peer.wait_for("joined").await;
        joined.push(line.field("identity").to_string());
    }

    let expected_reliable: Vec<Vec<u8>> = RELIABLE_LENGTHS
        .iter()
        .map(|&length| pattern(length, RELIABLE_SEED))
        .collect();
    let expected_lossy: Vec<Vec<u8>> = LOSSY_LENGTHS
        .iter()
        .map(|&length| pattern(length, LOSSY_SEED))
        .collect();
    for (sequence, payload) in expected_reliable.iter().enumerate() {
        reliable
            .process(frame(payload.clone(), sequence as u64 + 1), &mut out)
            .await
            .unwrap();
    }
    for (sequence, payload) in expected_lossy.iter().enumerate() {
        lossy
            .process(frame(payload.clone(), sequence as u64 + 1), &mut out)
            .await
            .unwrap();
    }
    for element in [&mut reliable, &mut lossy, &mut stray] {
        element
            .process(PipelinePacket::Eos, &mut out)
            .await
            .unwrap();
    }
    assert_eq!(reliable.sent(), 1 + RELIABLE_LENGTHS.len() as u64);
    assert_eq!(lossy.sent(), 1 + LOSSY_LENGTHS.len() as u64);

    let received: Vec<PeerLine> = peer
        .finish()
        .await
        .into_iter()
        .filter(|line| line.event == "received")
        .collect();
    let from = |identity: &str| -> Vec<&PeerLine> {
        received
            .iter()
            .filter(|line| line.field("sender") == identity)
            .filter(|line| line.payload() != WARMUP_PAYLOAD)
            .collect()
    };
    let from_reliable = from(RELIABLE_IDENTITY);
    let from_lossy = from(LOSSY_IDENTITY);
    for line in from_reliable.iter().chain(&from_lossy) {
        assert_eq!(line.field("topic"), TOPIC);
    }
    assert!(from_reliable
        .iter()
        .all(|line| line.field("kind") == "reliable"));
    assert!(from_lossy.iter().all(|line| line.field("kind") == "lossy"));
    let reliable_payloads: Vec<Vec<u8>> = from_reliable.iter().map(|line| line.payload()).collect();
    assert_eq!(reliable_payloads, expected_reliable, "reliable, in order");
    let lossy_payloads: Vec<Vec<u8>> = from_lossy.iter().map(|line| line.payload()).collect();
    for payload in &expected_lossy {
        assert!(
            lossy_payloads.contains(payload),
            "lossy payload {payload:02x?}"
        );
    }
    assert!(
        received.iter().all(|line| line.payload() != STRAY_PAYLOAD),
        "a packet addressed elsewhere reached the peer"
    );
}

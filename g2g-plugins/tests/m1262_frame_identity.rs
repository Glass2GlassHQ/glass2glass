#![cfg(all(feature = "udp-ingress", feature = "udp-egress"))]

use core::future::Future;
use core::pin::Pin;
use std::io::{BufRead, BufReader};
use std::net::{SocketAddr, UdpSocket as StdUdpSocket};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::graph::Graph;
use g2g_core::memory::SystemSlice;
use g2g_core::rtp::{ABS_CAPTURE_TIME_URI, NTP_64_URI};
use g2g_core::runtime::{
    run_graph_with_progress, run_simple_pipeline, GraphNodeRef, PipelineProgress, SourceLoop,
};
use g2g_core::{
    ntp_from_unix_nanos, AsyncElement, Bus, BusMessage, Caps, Colorimetry, ConfigureOutcome, Dim,
    FrameIdentityStats, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, PropValue, Rate, VideoCodec, WallClockMeta, NTP_TO_UNIX_EPOCH_SECS,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::udpsink::UdpSink;
use g2g_plugins::udpsrc::UdpSrc;

// x264's 16x16 IDR: SPS, PPS, slice
const ACCESS_UNIT: [u8; 46] = [
    0, 0, 0, 1, 0x67, 0x42, 0xc0, 0x0a, 0xdd, 0xec, 0x04, 0x40, 0x00, 0x00, 0x03, 0x00, 0x40, 0x00,
    0x00, 0x0f, 0x23, 0xc4, 0x89, 0xe0, 0, 0, 0, 1, 0x68, 0xce, 0x0f, 0xc8, 0, 0, 0, 1, 0x65, 0x88,
    0x84, 0x3a, 0x26, 0x28, 0x00, 0x09, 0x02, 0xe0,
];
const FRAME_SIZE: u32 = 16;
const FRAMES_PER_SECOND: u32 = 30;
const FRAME_PERIOD_NS: i64 = 1_000_000_000 / FRAMES_PER_SECOND as i64;
const NANOS_PER_SECOND: i64 = 1_000_000_000;
const SEND_SPACING: Duration = Duration::from_millis(5);
const RUN_TIMEOUT: Duration = Duration::from_secs(20);
const LINK_CAPACITY: usize = 4;
const BUS_CAPACITY: usize = 64;
const STREAM_FRAMES: usize = 8;
const ABS_CAPTURE_TIME_ID: u8 = 3;
const NTP_64_ID: u8 = 5;
const TWO_BYTE_FORM_ID: u8 = 20;
const PEER_SCRIPT: &str = "tests/m1262_gst_ntp64_peer.py";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(FRAME_SIZE),
        height: Dim::Fixed(FRAME_SIZE),
        framerate: Rate::Fixed(FRAMES_PER_SECOND << 16),
        colorimetry: Colorimetry::UNKNOWN,
    }
}

// capture times a second in the past, one frame period apart
fn capture_times(frames: &[i64]) -> Vec<i64> {
    let base = g2g_core::metrics::unix_time_ns() - NANOS_PER_SECOND;
    frames
        .iter()
        .map(|frame| base + frame * FRAME_PERIOD_NS)
        .collect()
}

fn stream_capture_times() -> Vec<i64> {
    capture_times(&(0..STREAM_FRAMES as i64).collect::<Vec<_>>())
}

fn ntp_nanos(unix_nanos: i64) -> i64 {
    unix_nanos + NTP_TO_UNIX_EPOCH_SECS as i64 * NANOS_PER_SECOND
}

struct StampedSource {
    capture_unix_ns: Vec<i64>,
}

impl SourceLoop for StampedSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(h264_caps()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for (index, &unix_nanos) in self.capture_unix_ns.iter().enumerate() {
                let pts_ns = index as u64 * FRAME_PERIOD_NS as u64;
                let mut frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(ACCESS_UNIT.into())),
                    FrameTiming {
                        pts_ns,
                        dts_ns: pts_ns,
                        keyframe: true,
                        ..FrameTiming::default()
                    },
                    index as u64,
                );
                frame.meta.attach(WallClockMeta { unix_nanos });
                out.push(PipelinePacket::DataFrame(frame)).await?;
                tokio::time::sleep(SEND_SPACING).await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.capture_unix_ns.len() as u64)
        })
    }
}

#[derive(Default)]
struct WallClockSink {
    capture_unix_ns: Arc<Mutex<Vec<Option<i64>>>>,
}

impl AsyncElement for WallClockSink {
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
                let wall_clock = frame.meta.get::<WallClockMeta>().map(|w| w.unix_nanos);
                self.capture_unix_ns.lock().unwrap().push(wall_clock);
            }
            Ok(())
        })
    }
}

fn bound_receiver() -> (StdUdpSocket, SocketAddr) {
    let socket = StdUdpSocket::bind("127.0.0.1:0").expect("bind receiver");
    let address = socket.local_addr().unwrap();
    (socket, address)
}

fn sender(destination: SocketAddr, abs_capture_time_id: u8, ntp_64_id: u8) -> UdpSink {
    let mut sink = UdpSink::new(destination);
    AsyncElement::set_property(
        &mut sink,
        "abs-capture-time-id",
        PropValue::Uint(abs_capture_time_id.into()),
    )
    .unwrap();
    AsyncElement::set_property(&mut sink, "ntp-64-id", PropValue::Uint(ntp_64_id.into())).unwrap();
    sink
}

async fn send(capture_unix_ns: Vec<i64>, sink: &mut UdpSink) {
    let mut source = StampedSource { capture_unix_ns };
    run_simple_pipeline(&mut source, sink, &ZeroClock, LINK_CAPACITY)
        .await
        .expect("send pipeline");
}

#[tokio::test]
async fn the_sink_reports_a_gap_and_a_duplicate() {
    // frames 5 and 6 never arrive, frame 8 arrives twice
    let capture_unix_ns = capture_times(&[0, 1, 2, 3, 4, 7, 8, 8, 9]);
    let mut graph: Graph<GraphNodeRef<'static>> = Graph::new();
    let source = graph.add_source(GraphNodeRef::source(StampedSource { capture_unix_ns }));
    let sink = graph.add_sink(GraphNodeRef::element(FakeSink::new()));
    graph.link(source, sink).unwrap();
    let (bus, bus_handle) = Bus::new(BUS_CAPACITY);
    run_graph_with_progress(
        graph,
        &ZeroClock,
        LINK_CAPACITY,
        &PipelineProgress::new(),
        Some(&bus_handle),
    )
    .await
    .expect("run");

    let mut reports: Vec<(String, FrameIdentityStats)> = Vec::new();
    while let Some(message) = bus.try_recv() {
        if let BusMessage::FrameIdentity { element, stats } = message {
            reports.push((element, stats));
        }
    }
    let (element, stats) = reports.last().expect("a report at end of stream");
    assert!(!element.is_empty(), "the report names its sink");
    assert_eq!(stats.frames, 8);
    assert_eq!(stats.lost, 2);
    assert_eq!(stats.duplicates, 1);
    assert!(
        stats.latency_min_ns > 0 && stats.latency_min_ns <= stats.latency_max_ns,
        "{stats:?}"
    );
}

#[tokio::test]
async fn capture_time_crosses_a_g2g_hop_in_ids_an_sdp_negotiates() {
    let (socket, address) = bound_receiver();
    let sdp = format!(
        "v=0\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n\
         a=extmap:{ABS_CAPTURE_TIME_ID} {ABS_CAPTURE_TIME_URI}\r\n\
         a=extmap:{TWO_BYTE_FORM_ID} {NTP_64_URI}\r\n"
    );
    let mut receiver = UdpSrc::from_socket(socket)
        .unwrap()
        .with_frame_limit(STREAM_FRAMES as u64);
    receiver.set_property("sdp", PropValue::Str(sdp)).unwrap();
    assert_eq!(
        receiver.get_property("ntp-64-id"),
        Some(PropValue::Uint(TWO_BYTE_FORM_ID.into())),
        "the SDP's extmap sets the id"
    );
    let mut collected = WallClockSink::default();
    let received = collected.capture_unix_ns.clone();
    let expected = stream_capture_times();
    let mut sink = sender(address, ABS_CAPTURE_TIME_ID, TWO_BYTE_FORM_ID);

    let receive = run_simple_pipeline(&mut receiver, &mut collected, &ZeroClock, LINK_CAPACITY);
    let (received_run, ()) = tokio::time::timeout(RUN_TIMEOUT, async {
        tokio::join!(receive, send(expected.clone(), &mut sink))
    })
    .await
    .expect("both ends finish");
    received_run.expect("receive pipeline");

    let received: Vec<Option<i64>> = received.lock().unwrap().clone();
    assert_eq!(received, expected.into_iter().map(Some).collect::<Vec<_>>());
}

fn peer_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(PEER_SCRIPT)
}

fn gstreamer_peer_available() -> bool {
    let status = Command::new("python3")
        .arg("-I")
        .arg(peer_script())
        .arg("check")
        .status();
    let available = matches!(status, Ok(status) if status.success());
    if !available {
        eprintln!("skipping: python3 with GStreamer's rtphdrextntp64 is not available");
    }
    available
}

fn access_unit_hex() -> String {
    ACCESS_UNIT
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[tokio::test]
async fn gstreamer_reads_the_capture_time_g2g_writes() {
    if !gstreamer_peer_available() {
        return;
    }
    let mut peer = Command::new("python3")
        .arg("-I")
        .arg(peer_script())
        .args([
            "receive",
            &NTP_64_ID.to_string(),
            &ABS_CAPTURE_TIME_ID.to_string(),
            &STREAM_FRAMES.to_string(),
        ])
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn the GStreamer receiver");
    let mut lines = BufReader::new(peer.stdout.take().unwrap()).lines();
    let ready = lines.next().expect("a ready line").unwrap();
    let port: u16 = ready
        .strip_prefix("ready ")
        .and_then(|port| port.parse().ok())
        .unwrap_or_else(|| panic!("unexpected peer line {ready:?}"));

    let expected = stream_capture_times();
    let destination = SocketAddr::from(([127, 0, 0, 1], port));
    send(
        expected.clone(),
        &mut sender(destination, ABS_CAPTURE_TIME_ID, NTP_64_ID),
    )
    .await;

    let output: Vec<String> = lines.map(Result::unwrap).collect();
    peer.wait().unwrap();
    let values = |kind: &str| -> Vec<String> {
        output
            .iter()
            .filter_map(|line| line.strip_prefix(kind).map(str::to_string))
            .collect()
    };
    let ntp_nanos_read: Vec<String> = values("ntp ");
    let expected_ntp_nanos: Vec<String> = expected
        .iter()
        .map(|&unix| ntp_nanos(unix).to_string())
        .collect();
    assert_eq!(
        ntp_nanos_read, expected_ntp_nanos,
        "rtphdrextntp64 reads the same nanosecond"
    );
    let abs_read = values("abs ");
    let expected_abs: Vec<String> = expected
        .iter()
        .map(|&unix| format!("{:016x}", ntp_from_unix_nanos(unix).unwrap()))
        .collect();
    assert_eq!(
        abs_read, expected_abs,
        "GStreamer's RFC 8285 parser finds abs-capture-time"
    );
}

#[tokio::test]
async fn g2g_reads_the_capture_time_gstreamer_writes() {
    if !gstreamer_peer_available() {
        return;
    }
    let (socket, address) = bound_receiver();
    let mut receiver = UdpSrc::from_socket(socket)
        .unwrap()
        .with_frame_limit(STREAM_FRAMES as u64);
    receiver
        .set_property("ntp-64-id", PropValue::Uint(NTP_64_ID.into()))
        .unwrap();
    let mut collected = WallClockSink::default();
    let received = collected.capture_unix_ns.clone();

    let expected = stream_capture_times();
    let mut peer = Command::new("python3")
        .arg("-I")
        .arg(peer_script())
        .args(["send", &address.port().to_string(), &NTP_64_ID.to_string()])
        .arg(access_unit_hex())
        .args(expected.iter().map(|&unix| ntp_nanos(unix).to_string()))
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn the GStreamer sender");
    tokio::time::timeout(
        RUN_TIMEOUT,
        run_simple_pipeline(&mut receiver, &mut collected, &ZeroClock, LINK_CAPACITY),
    )
    .await
    .expect("every frame arrives")
    .expect("receive pipeline");
    peer.wait().unwrap();

    let received: Vec<Option<i64>> = received.lock().unwrap().clone();
    assert_eq!(received, expected.into_iter().map(Some).collect::<Vec<_>>());
}

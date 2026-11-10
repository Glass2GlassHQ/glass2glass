// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "udp-ingress")]

use std::net::UdpSocket as StdUdpSocket;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use core::future::Future;
use core::pin::Pin;

use g2g_core::graph::Graph;
use g2g_core::runtime::{parse_launch, run_graph, GraphNode, GraphNodeRef};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, ClockPriority, ConfigureOutcome, G2gError, OutputSink,
    PipelineClock, PipelinePacket, PropValue, Segment,
};

use g2g_plugins::bytestream::datagram_chunk;
use g2g_plugins::mpegts::{
    ticks_90khz_to_ns, TsDemuxer, TsPacketSplitter, SYSTEM_CLOCK_TICKS_PER_90KHZ,
};
use g2g_plugins::mpegtslivesrc::MpegTsLiveSrc;
use g2g_plugins::registry::default_registry;
use g2g_plugins::tsdemux::TsDemux;
use g2g_plugins::udpsrc::UdpSrc;

const FIXTURE: &str = "tests/fixtures/av_h264_aac44100.ts";
const DATAGRAM_PAYLOAD: usize = 1316;
const DEADLINE: Duration = Duration::from_secs(20);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture() -> Vec<u8> {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    std::fs::read(path).expect("read the fixture")
}

fn first_pcr_27mhz(ts: &[u8]) -> u64 {
    let mut reader = TsDemuxer::pcr_reader();
    let mut splitter = TsPacketSplitter::new();
    splitter.push(ts, &mut reader);
    reader.take_pcr_samples()[0].pcr_27mhz
}

// queued on a bound socket before the receiver runs, so loopback drops none
fn send_datagrams(ts: &[u8], to: u16) -> u64 {
    let chunk = datagram_chunk(ByteStreamEncoding::MpegTs, DATAGRAM_PAYLOAD);
    let sender = StdUdpSocket::bind("127.0.0.1:0").expect("bind a sender");
    for datagram in ts.chunks(chunk) {
        sender.send_to(datagram, ("127.0.0.1", to)).expect("send");
    }
    ts.len().div_ceil(chunk) as u64
}

#[derive(Debug, Default)]
struct Recording {
    segment_before_first_frame: Option<Segment>,
    frames: u64,
}

#[derive(Default)]
struct SegmentSink {
    recording: Arc<Mutex<Recording>>,
}

impl AsyncElement for SegmentSink {
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
            let mut recording = self.recording.lock().unwrap();
            match packet {
                PipelinePacket::Segment(segment) if recording.frames == 0 => {
                    recording.segment_before_first_frame = Some(segment);
                }
                PipelinePacket::DataFrame(_) => recording.frames += 1,
                _ => {}
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn tsdemux_runs_on_the_pcr_clock_and_segment() {
    let ts = fixture();
    let first_pcr_ns = ticks_90khz_to_ns(first_pcr_27mhz(&ts) / SYSTEM_CLOCK_TICKS_PER_90KHZ);
    let socket = StdUdpSocket::bind("127.0.0.1:0").expect("bind a receiver");
    let port = socket.local_addr().unwrap().port();
    let datagrams = send_datagrams(&ts, port);
    let source = UdpSrc::from_socket(socket)
        .expect("adopt the socket")
        .with_bytestream(ByteStreamEncoding::MpegTs)
        .with_frame_limit(datagrams);
    let mut demux = TsDemux::new();
    demux
        .set_property("stream", PropValue::Str("h264".into()))
        .unwrap();
    let recorder = SegmentSink::default();
    let recording = recorder.recording.clone();
    let stats = {
        let mut graph: Graph<GraphNode> = Graph::new();
        let src = graph.add_source(GraphNodeRef::Source(Box::new(source)));
        let clock = graph.add_transform(GraphNodeRef::element(MpegTsLiveSrc::new()));
        let demux = graph.add_transform(GraphNodeRef::element(demux));
        let sink = graph.add_sink(GraphNodeRef::element(recorder));
        graph.link(src, clock).unwrap();
        graph.link(clock, demux).unwrap();
        graph.link(demux, sink).unwrap();
        tokio::time::timeout(DEADLINE, run_graph(graph, &ZeroClock, 4))
            .await
            .expect("the run finishes")
            .expect("the run succeeds")
    };
    assert_eq!(stats.clock_priority, ClockPriority::LiveSource);
    let recording = recording.lock().unwrap();
    assert!(recording.frames > 0, "frames reached the sink");
    let segment = recording
        .segment_before_first_frame
        .expect("a segment came before the first frame");
    assert_eq!(segment.start, first_pcr_ns, "{segment:?}");
}

#[test]
fn the_gst_source_property_builds_the_source_ahead() {
    let registry = default_registry();
    let line = "mpegtslivesrc source=\"udpsrc address=127.0.0.1 port=0 bytestream-format=mpegts\" \
                ! tsdemux stream=h264 ! fakesink";
    assert!(parse_launch(&registry, line).is_ok(), "`{line}` builds");
    let without_source = "mpegtslivesrc ! tsdemux ! fakesink";
    assert!(parse_launch(&registry, without_source).is_err());
}

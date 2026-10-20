#![cfg(all(
    feature = "metadata",
    any(
        all(feature = "rtsp-client-sink", feature = "rtsp-server"),
        all(target_os = "linux", feature = "webrtc")
    )
))]

#[cfg(all(target_os = "linux", feature = "webrtc"))]
mod webrtc_streams_common;

use core::task::{Context, Poll};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome, WallClockMeta,
};

// 2026-10-20, a capture time no clock in the path would produce by itself
const CAPTURE_BASE_UNIX_NS: i64 = 1_792_454_400_000_000_000;
const SERVER_DEADLINE: Duration = Duration::from_secs(20);
const POLL_INTERVAL: Duration = Duration::from_millis(5);

fn capture_unix_ns(pts_ns: u64) -> i64 {
    CAPTURE_BASE_UNIX_NS + pts_ns as i64
}

fn stamped_frame(data: Vec<u8>, pts_ns: u64) -> Frame {
    let mut frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        0,
    );
    frame.meta.attach(WallClockMeta {
        unix_nanos: capture_unix_ns(pts_ns),
    });
    frame
}

#[derive(Debug, Clone, PartialEq)]
struct ReceivedFrame {
    data: Vec<u8>,
    capture_unix_ns: Option<i64>,
}

fn received_frame(frame: &Frame) -> ReceivedFrame {
    ReceivedFrame {
        data: frame
            .domain
            .as_system_slice()
            .expect("system frame")
            .to_vec(),
        capture_unix_ns: frame.meta.get::<WallClockMeta>().map(|w| w.unix_nanos),
    }
}

#[derive(Default, Clone)]
struct Collected {
    frames: Arc<Mutex<Vec<ReceivedFrame>>>,
}

impl Collected {
    fn take(&self) -> Vec<ReceivedFrame> {
        core::mem::take(&mut *self.frames.lock().unwrap())
    }
}

impl OutputSink for Collected {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet.take() {
            self.frames.lock().unwrap().push(received_frame(&frame));
        }
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

#[cfg(all(feature = "rtsp-client-sink", feature = "rtsp-server"))]
mod rtsp {
    use super::*;

    use std::io::{BufRead, BufReader};
    use std::net::TcpListener as StdTcpListener;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

    use g2g_core::runtime::SourceLoop;
    use g2g_core::{
        Caps, Dim, MultiInputElement, PropValue, Rate, VideoCodec, NTP_TO_UNIX_EPOCH_SECS,
    };
    use g2g_plugins::rtspclient::LowerTransport;
    use g2g_plugins::rtspclientsink::RtspClientSink;
    use g2g_plugins::rtspserversrc::RtspServerSrc;

    const SPS: &[u8] = &[0x67, 0x64, 0x00, 0x28, 0xAC, 0xD9];
    const PPS: &[u8] = &[0x68, 0xEB, 0xEC, 0xB2];
    const IDR_NAL_HEADER: u8 = 0x65;
    const START_CODE: &[u8] = &[0, 0, 0, 1];
    const ACCESS_UNITS: usize = 8;
    const ABS_CAPTURE_TIME_ID: u8 = 3;
    const NTP_64_ID: u8 = 5;
    const EXTENSION_OFF: u8 = 0;
    // x264's 16x16 IDR: SPS, PPS, slice
    const X264_ACCESS_UNIT: [u8; 46] = [
        0, 0, 0, 1, 0x67, 0x42, 0xc0, 0x0a, 0xdd, 0xec, 0x04, 0x40, 0x00, 0x00, 0x03, 0x00, 0x40,
        0x00, 0x00, 0x0f, 0x23, 0xc4, 0x89, 0xe0, 0, 0, 0, 1, 0x68, 0xce, 0x0f, 0xc8, 0, 0, 0, 1,
        0x65, 0x88, 0x84, 0x3a, 0x26, 0x28, 0x00, 0x09, 0x02, 0xe0,
    ];
    const NANOS_PER_SECOND: i64 = 1_000_000_000;
    const PEER_SCRIPT: &str = "tests/m1710_gst_rtsp_publisher.py";
    const FRAME_PERIOD_NS: u64 = 33_333_333;

    fn h264_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    // every unit is an IDR whose last byte is its index, so a received unit names its pts
    fn access_unit(index: usize) -> Vec<u8> {
        [SPS, PPS, &[IDR_NAL_HEADER, index as u8]]
            .iter()
            .flat_map(|nal| START_CODE.iter().chain(nal.iter()).copied())
            .collect()
    }

    fn pts_of(index: usize) -> u64 {
        index as u64 * FRAME_PERIOD_NS
    }

    fn one_shot_server() -> (RtspServerSrc, u16) {
        let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp");
        let port = listener.local_addr().unwrap().port();
        let server = RtspServerSrc::from_listener(listener)
            .unwrap()
            .with_max_sessions(1);
        (server, port)
    }

    async fn publish_into_server(
        transport: LowerTransport,
        abs_capture_time_id: u8,
        ntp_64_id: u8,
    ) -> Vec<ReceivedFrame> {
        let (mut server, port) = one_shot_server();
        server.configure_pipeline(&h264_caps()).expect("configure");
        let mut sink = RtspClientSink::new(1, format!("rtsp://127.0.0.1:{port}/stream"))
            .with_protocols(vec![transport]);
        for (name, id) in [
            ("abs-capture-time-id", abs_capture_time_id),
            ("ntp-64-id", ntp_64_id),
        ] {
            sink.set_property(name, PropValue::Uint(id.into())).unwrap();
        }
        let collected = Collected::default();
        let mut output = collected.clone();
        let receive = async {
            tokio::time::timeout(SERVER_DEADLINE, server.run(&mut output))
                .await
                .expect("server finishes")
                .expect("server runs")
        };
        let send = async {
            let mut discard = Collected::default();
            sink.configure_pipeline(0, &h264_caps()).unwrap();
            for index in 0..ACCESS_UNITS {
                let frame = stamped_frame(access_unit(index), pts_of(index));
                sink.process(0, PipelinePacket::DataFrame(frame), &mut discard)
                    .await
                    .unwrap();
            }
            // UDP media can arrive after the TEARDOWN, so EOS waits for the server
            while collected.frames.lock().unwrap().len() < ACCESS_UNITS {
                tokio::time::sleep(POLL_INTERVAL).await;
            }
            sink.process(0, PipelinePacket::Eos, &mut discard)
                .await
                .unwrap();
        };
        tokio::join!(receive, send);
        collected.take()
    }

    fn assert_every_unit_kept_its_capture_time(received: &[ReceivedFrame]) {
        assert_eq!(received.len(), ACCESS_UNITS);
        for frame in received {
            let index = usize::from(*frame.data.last().expect("a unit"));
            assert_eq!(frame.data, access_unit(index));
            assert_eq!(
                frame.capture_unix_ns,
                Some(capture_unix_ns(pts_of(index))),
                "unit {index}"
            );
        }
    }

    #[tokio::test]
    async fn both_extensions_over_tcp_interleaved_reach_the_ingested_frames() {
        let received =
            publish_into_server(LowerTransport::Tcp, ABS_CAPTURE_TIME_ID, NTP_64_ID).await;
        assert_every_unit_kept_its_capture_time(&received);
    }

    #[tokio::test]
    async fn ntp_64_over_udp_reaches_the_ingested_frames() {
        let received = publish_into_server(LowerTransport::Udp, EXTENSION_OFF, NTP_64_ID).await;
        assert_every_unit_kept_its_capture_time(&received);
    }

    #[tokio::test]
    async fn no_extension_id_leaves_the_ingested_frames_unstamped() {
        let received = publish_into_server(LowerTransport::Tcp, EXTENSION_OFF, EXTENSION_OFF).await;
        assert_eq!(received.len(), ACCESS_UNITS);
        assert!(received.iter().all(|f| f.capture_unix_ns.is_none()));
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

    fn ntp_nanos(unix_nanos: i64) -> i64 {
        unix_nanos + NTP_TO_UNIX_EPOCH_SECS as i64 * NANOS_PER_SECOND
    }

    #[tokio::test]
    async fn g2g_ingests_the_capture_time_a_gstreamer_publisher_writes() {
        if !gstreamer_peer_available() {
            return;
        }
        // the publisher's TEARDOWN ends the one session and the run
        let (mut server, port) = one_shot_server();
        server.configure_pipeline(&h264_caps()).expect("configure");
        let expected: Vec<i64> = (0..ACCESS_UNITS)
            .map(|index| capture_unix_ns(pts_of(index)))
            .collect();
        let access_unit_hex: String = X264_ACCESS_UNIT
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let mut peer = Command::new("python3")
            .arg("-I")
            .arg(peer_script())
            .args(["publish", &port.to_string(), &NTP_64_ID.to_string()])
            .arg(access_unit_hex)
            .args(expected.iter().map(|&unix| ntp_nanos(unix).to_string()))
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn the GStreamer publisher");
        let collected = Collected::default();
        let mut output = collected.clone();
        tokio::time::timeout(SERVER_DEADLINE, server.run(&mut output))
            .await
            .expect("every frame arrives")
            .expect("server runs");
        let lines: Vec<String> = BufReader::new(peer.stdout.take().unwrap())
            .lines()
            .map(Result::unwrap)
            .collect();
        assert!(peer.wait().unwrap().success(), "publisher output {lines:?}");

        let received: Vec<Option<i64>> = collected
            .take()
            .into_iter()
            .map(|frame| frame.capture_unix_ns)
            .collect();
        assert_eq!(received, expected.into_iter().map(Some).collect::<Vec<_>>());
    }
}

#[cfg(all(target_os = "linux", feature = "webrtc"))]
mod webrtc {
    use super::*;

    use std::collections::HashMap;

    use g2g_core::{AsyncElement, MultiOutputSink, MultiOutputSource, PropValue};
    use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
    use g2g_plugins::webrtcsink::WebRtcSink;
    use g2g_plugins::whipserversrc::WhipServerSrc;

    use crate::webrtc_streams_common::{fixed_caps, synthetic_stream, FPS};

    const VIDEO_PORT: usize = 0;
    const PORTS: usize = 2;
    const VIDEO_UNITS: usize = 60;
    const MIN_DELIVERED_PERCENT: usize = 90;
    // let the sender's pacer flush before EOS deletes the resource
    const DRAIN_BEFORE_EOS: Duration = Duration::from_millis(500);
    // str0m converts abs-capture-time through f64 seconds on both ends
    const STR0M_PRECISION_NS: i64 = 1_000;

    #[derive(Default, Clone)]
    struct VideoPort {
        collected: Collected,
        announced: Arc<Mutex<usize>>,
    }

    impl MultiOutputSink for VideoPort {
        fn poll_push_to(
            &mut self,
            cx: &mut Context<'_>,
            port: usize,
            packet: &mut Option<PipelinePacket>,
        ) -> Poll<Result<PushOutcome, G2gError>> {
            if matches!(packet, Some(PipelinePacket::CapsChanged(_))) {
                *self.announced.lock().unwrap() += 1;
            }
            if port != VIDEO_PORT {
                packet.take();
                return Poll::Ready(Ok(PushOutcome::Accepted));
            }
            self.collected.poll_push(cx, packet)
        }

        fn port_count(&self) -> usize {
            PORTS
        }
    }

    fn free_port() -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    }

    #[tokio::test]
    async fn abs_capture_time_crosses_a_whip_session() {
        let codec = WebRtcVideoCodec::H264;
        let interval = Duration::from_secs(1) / FPS;
        let units = synthetic_stream(codec, VIDEO_UNITS);
        let pts_by_unit: HashMap<Vec<u8>, u64> = units
            .iter()
            .enumerate()
            .map(|(index, unit)| (unit.clone(), index as u64 * interval.as_nanos() as u64))
            .collect();
        assert_eq!(pts_by_unit.len(), units.len(), "every unit is distinct");

        let host_addr = format!("http://127.0.0.1:{}", free_port());
        let mut server = WhipServerSrc::new(host_addr.clone());
        server
            .set_property("video-codec", PropValue::Str(codec.name().into()))
            .unwrap();
        let mut output = VideoPort::default();
        let received = output.collected.clone();
        let announced = output.announced.clone();
        let run = server.run(&mut output);
        let publish = async {
            // both caps announcements follow the bind, so the server is listening
            while *announced.lock().unwrap() < PORTS {
                tokio::time::sleep(POLL_INTERVAL).await;
            }
            let mut sink = WebRtcSink::new(format!("{host_addr}/whip"));
            sink.configure_pipeline(&fixed_caps(codec)).unwrap();
            let mut discard = Collected::default();
            for (index, unit) in units.iter().enumerate() {
                let frame = stamped_frame(unit.clone(), index as u64 * interval.as_nanos() as u64);
                sink.process(PipelinePacket::DataFrame(frame), &mut discard)
                    .await
                    .unwrap();
                tokio::time::sleep(interval).await;
            }
            tokio::time::sleep(DRAIN_BEFORE_EOS).await;
            sink.process(PipelinePacket::Eos, &mut discard)
                .await
                .unwrap();
        };
        let (result, ()) =
            tokio::time::timeout(SERVER_DEADLINE, async { tokio::join!(run, publish) })
                .await
                .expect("loopback run timed out");
        result.expect("server run failed");

        let received = received.take();
        assert!(
            received.len() >= units.len() * MIN_DELIVERED_PERCENT / 100,
            "{} of {} units arrived",
            received.len(),
            units.len()
        );
        for frame in &received {
            let pts_ns = pts_by_unit[&frame.data];
            let stamped = frame
                .capture_unix_ns
                .expect("every unit carries its capture time");
            let error_ns = (stamped - capture_unix_ns(pts_ns)).abs();
            assert!(
                error_ns <= STR0M_PRECISION_NS,
                "unit at {pts_ns} ns is off by {error_ns} ns"
            );
        }
    }
}

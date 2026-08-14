//! M1044: the canonical "watch it and record it" graph, on real hardware.
//!
//! ```text
//! RtspSrc -> tee -+-> FfmpegH264Dec -> WaylandSink
//!                 `-> Mp4Mux -> FileSink
//! ```
//!
//! Until now the tee only had fake-element coverage. This drives one live RTSP
//! H.264 feed into both a display branch and a recording branch, then reads the
//! written file back with g2g's own fMP4 demuxer.
//!
//! Ignored by default. Requires:
//! - A running Wayland session (`WAYLAND_DISPLAY` set in the environment).
//! - An RTSP feed at `G2G_RTSP_TEST_URL` (default `rtsp://127.0.0.1:8554/pattern`).
//!
//! ```sh
//! G2G_RTSP_TEST_URL=rtsp://127.0.0.1:8554/pattern \
//!     cargo test -p g2g-plugins \
//!     --features "rtsp ffmpeg wayland-sink" \
//!     --test m1044_hw_tee_decode_mux -- --ignored --nocapture
//! ```
//!
//! A window titled "g2g M1044" shows the feed for a couple of seconds.
//! `G2G_TARGET_FRAMES` bounds the run (the source's `num-buffers`), and
//! `G2G_LINK_CAP` overrides the default `LatencyProfile::Live` depth.

#![cfg(all(
    target_os = "linux",
    feature = "rtsp",
    feature = "ffmpeg",
    feature = "wayland-sink"
))]

use std::path::PathBuf;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{run_graph, GraphNodeRef, LatencyProfile, LinkCapacity};
use g2g_core::{
    ByteStreamEncoding, Caps, G2gError, Graph, OutputSink, PipelineClock, PipelinePacket,
    PushOutcome, VideoCodec,
};
use g2g_plugins::ffmpegdec::{FfmpegH264Dec, OutputFormat};
use g2g_plugins::filesink::FileSink;
use g2g_plugins::fmp4demux::Fmp4Demux;
use g2g_plugins::mp4mux::Mp4Mux;
use g2g_plugins::rtspsrc::RtspSrc;
use g2g_plugins::waylandsink::WaylandSink;

const DEFAULT_RTSP_URL: &str = "rtsp://127.0.0.1:8554/pattern";
const DEFAULT_TARGET_FRAMES: u64 = 90;
const WINDOW_TITLE: &str = "g2g M1044";
/// Enough decoded frames to prove the display branch really ran, low enough to
/// tolerate the mid-GOP tune-in the source drops before its first IDR.
const MIN_DECODED_FRAMES: u64 = 10;
/// Same idea for the recording branch: a couple of GOPs of access units.
const MIN_MUXED_ACCESS_UNITS: usize = 10;
/// Fixed setup tax (RTSP DESCRIBE/SETUP, decoder open, Wayland surface) on top
/// of the wall time the frames themselves take.
const SETUP_BUDGET_S: u64 = 40;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Removes the recording on the way out, including on a failed assertion.
struct TempRecording(PathBuf);
impl Drop for TempRecording {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Collects what the demuxer pushes downstream so the readback can assert on the
/// track caps and the recovered access units.
#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    access_units: Vec<Vec<u8>>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        match packet {
            PipelinePacket::CapsChanged(caps) => self.caps.push(caps),
            PipelinePacket::DataFrame(frame) => {
                if let Some(slice) = frame.domain.as_system_slice() {
                    self.access_units.push(slice.to_vec());
                }
            }
            _ => {}
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Reads the recording back through `Fmp4Demux`, the in-repo inverse of `Mp4Mux`.
async fn demux_recording(bytes: Vec<u8>) -> CaptureSink {
    let mut demux = Fmp4Demux::new();
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .expect("fmp4 demux accepts an ISO-BMFF byte stream");

    let mut sink = CaptureSink::default();
    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };
    demux
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("demuxing the recording");
    demux
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("demux end of stream");
    sink
}

#[tokio::test]
#[ignore = "needs a Wayland session + a live RTSP feed (set G2G_RTSP_TEST_URL)"]
async fn tee_feeds_a_wayland_display_and_an_mp4_recording() {
    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("skipping: no WAYLAND_DISPLAY in env (run under a Wayland session)");
        return;
    }

    let url = std::env::var("G2G_RTSP_TEST_URL").unwrap_or_else(|_| DEFAULT_RTSP_URL.to_string());
    let target: u64 = std::env::var("G2G_TARGET_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TARGET_FRAMES);
    let link_cap: LinkCapacity = match std::env::var("G2G_LINK_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        Some(n) => LinkCapacity::new(n),
        None => LatencyProfile::Live.link_capacity(),
    };
    eprintln!(
        "connecting to {url} (target frames = {target}, link capacity = {})",
        link_cap.get()
    );

    let recording =
        TempRecording(std::env::temp_dir().join(format!("g2g-m1044-{}.mp4", std::process::id())));

    let mut source = RtspSrc::new(url).with_frame_limit(target);
    let mut decoder = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
    let mut display = WaylandSink::new().with_title(WINDOW_TITLE);
    let mut muxer = Mp4Mux::new();
    let mut file = FileSink::new(recording.0.clone());

    let stats = {
        let mut graph: Graph<GraphNodeRef> = Graph::new();
        let source_node = graph.add_source(GraphNodeRef::source_ref(&mut source));
        let tee = graph.add_tee(2);
        let decoder_node = graph.add_transform(GraphNodeRef::element_ref(&mut decoder));
        let display_node = graph.add_sink(GraphNodeRef::element_ref(&mut display));
        let muxer_node = graph.add_transform(GraphNodeRef::element_ref(&mut muxer));
        let file_node = graph.add_sink(GraphNodeRef::element_ref(&mut file));
        graph.link(source_node, tee.input()).unwrap();
        graph.link(tee.out(0), decoder_node).unwrap();
        graph.link(decoder_node, display_node).unwrap();
        graph.link(tee.out(1), muxer_node).unwrap();
        graph.link(muxer_node, file_node).unwrap();

        let budget = std::time::Duration::from_secs(SETUP_BUDGET_S + target / 10);
        tokio::time::timeout(budget, run_graph(graph, &ZeroClock, link_cap))
            .await
            .unwrap_or_else(|_| panic!("pipeline should complete within {budget:?}"))
            .expect("tee graph should run end to end")
    };

    eprintln!(
        "stats: emitted={} consumed={} decoded={} presented={} muxed_frames={} bytes_written={}",
        stats.frames_emitted,
        stats.frames_consumed,
        decoder.decoded_count(),
        display.frames_presented(),
        muxer.emitted(),
        file.bytes_written(),
    );

    assert!(
        decoder.decoded_count() >= MIN_DECODED_FRAMES,
        "display branch decoded only {} frames",
        decoder.decoded_count()
    );
    assert!(file.eos_seen(), "recording branch never saw end of stream");
    assert!(
        file.bytes_written() > 0,
        "recording branch wrote an empty file"
    );

    let bytes = std::fs::read(&recording.0).expect("read the recording back");
    assert_eq!(
        bytes.len() as u64,
        file.bytes_written(),
        "the file on disk holds every byte the sink reported"
    );

    let readback = demux_recording(bytes).await;
    let track = readback
        .caps
        .iter()
        .find(|caps| {
            matches!(
                caps,
                Caps::CompressedVideo {
                    codec: VideoCodec::H264,
                    ..
                }
            )
        })
        .unwrap_or_else(|| panic!("no H.264 track in the recording: {:?}", readback.caps));
    eprintln!(
        "readback: track={track:?} access_units={}",
        readback.access_units.len()
    );

    assert!(
        readback.access_units.len() >= MIN_MUXED_ACCESS_UNITS,
        "recovered only {} access units from the recording",
        readback.access_units.len()
    );
    assert_eq!(
        readback.access_units.len() as u64,
        muxer.emitted(),
        "every access unit the muxer wrote came back out of the file"
    );
}

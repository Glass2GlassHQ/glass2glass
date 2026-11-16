// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{block_on, SeekController};
use g2g_core::{G2gError, MultiOutputElement, MultiOutputSink, PushOutcome, Seek};
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};

// ffmpeg -i av_seek_h264_aac.mp4 -c copy -movflags +faststart, so the moov comes before the mdat
const FIXTURE: &str = "av_seek_h264_aac_faststart.mp4";
// a sixteenth of the 17 KB fixture, so the file arrives over many packets
const CHUNK_LEN: usize = 1024;
// between the keyframes at 0.8 s and 1.2 s of video time
const TARGET_NS: u64 = 1_020_000_000;
// the fixture's video: 25 fps for 2 s, encoded with -g 10
const FRAME_PERIOD_NS: u64 = 40_000_000;
const FRAMES_PER_KEYFRAME: usize = 10;
const VIDEO_FRAME_COUNT: usize = 50;

#[derive(Debug, Clone, PartialEq)]
struct Unit {
    pts_ns: u64,
    keyframe: bool,
    bytes: Vec<u8>,
}

#[derive(Debug)]
struct Ports {
    units: Vec<Vec<Unit>>,
    segment_starts: Vec<Vec<u64>>,
}

impl Ports {
    fn new(count: usize) -> Self {
        Self {
            units: vec![Vec::new(); count],
            segment_starts: vec![Vec::new(); count],
        }
    }

    fn clear(&mut self) {
        self.units.iter_mut().for_each(Vec::clear);
        self.segment_starts.iter_mut().for_each(Vec::clear);
    }
}

impl MultiOutputSink for Ports {
    fn port_count(&self) -> usize {
        self.units.len()
    }

    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        match packet_slot.take().expect("poll_push without a packet") {
            PipelinePacket::DataFrame(frame) => self.units[port].push(Unit {
                pts_ns: frame.timing.pts_ns,
                keyframe: frame.timing.keyframe,
                bytes: frame
                    .domain
                    .as_system_slice()
                    .expect("system frame")
                    .to_vec(),
            }),
            PipelinePacket::Segment(segment) => self.segment_starts[port].push(segment.start),
            _ => {}
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn fixture_path() -> String {
    format!("{}/tests/fixtures/{FIXTURE}", env!("CARGO_MANIFEST_DIR"))
}

fn fixture() -> Vec<u8> {
    std::fs::read(fixture_path()).expect("the fixture reads")
}

fn demuxer(file: &[u8]) -> Mp4DemuxN {
    Mp4DemuxN::new(
        forwardable_streams(file)
            .into_iter()
            .map(|stream| Mp4Port {
                track_id: stream.track_id,
                caps: stream.caps,
            })
            .collect(),
    )
}

// as filesrc wires it: the app seeks the demuxer, which byte-seeks the source through `upstream`
fn seekable_demuxer(file: &[u8]) -> (Mp4DemuxN, SeekController, SeekController) {
    let mut demux = demuxer(file);
    let app = SeekController::new();
    let upstream = SeekController::new();
    upstream.set_stream_len(file.len() as u64);
    MultiOutputElement::set_seek(&mut demux, app.clone(), upstream.clone());
    (demux, app, upstream)
}

async fn feed(demux: &mut Mp4DemuxN, bytes: &[u8], ports: &mut Ports) {
    for chunk in bytes.chunks(CHUNK_LEN) {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        demux
            .process(PipelinePacket::DataFrame(frame), ports)
            .await
            .expect("the bytes demux");
    }
}

async fn end(demux: &mut Mp4DemuxN, ports: &mut Ports) {
    demux
        .process(PipelinePacket::Eos, ports)
        .await
        .expect("the stream ends");
}

async fn whole_file_units(file: &[u8]) -> Ports {
    let mut demux = demuxer(file);
    let mut ports = Ports::new(demux.port_count());
    feed(&mut demux, file, &mut ports).await;
    assert!(
        ports.units.iter().all(Vec::is_empty),
        "without a seekable source the file is held whole until Eos"
    );
    end(&mut demux, &mut ports).await;
    ports
}

#[test]
fn a_faststart_file_streams_before_eos_and_matches_the_whole_file_parse() {
    block_on(async {
        let file = fixture();
        let whole = whole_file_units(&file).await;
        let (mut demux, _app, _upstream) = seekable_demuxer(&file);
        let mut streamed = Ports::new(demux.port_count());
        let half = file.len() / 2;
        feed(&mut demux, &file[..half], &mut streamed).await;
        assert!(
            streamed.units.iter().all(|units| !units.is_empty()),
            "every port has frames out before the second half of the file arrives"
        );
        feed(&mut demux, &file[half..], &mut streamed).await;
        end(&mut demux, &mut streamed).await;
        assert_eq!(streamed.units, whole.units);
    });
}

#[test]
fn an_app_seek_reads_on_from_the_indexed_keyframe() {
    block_on(async {
        let file = fixture();
        let whole = whole_file_units(&file).await;
        let (mut demux, app, upstream) = seekable_demuxer(&file);
        let mut ports = Ports::new(demux.port_count());
        let half = file.len() / 2;
        feed(&mut demux, &file[..half], &mut ports).await;
        app.seek(Seek::flush_to(TARGET_NS));
        // the demuxer takes the seek at its next packet, and drops that packet's bytes
        feed(&mut demux, &file[half..half + 1], &mut ports).await;
        let byte_seek = upstream
            .take_seek(Ok)
            .expect("the demuxer byte-seeks its source");
        let landing = byte_seek.start as usize;
        assert!(
            landing > 0,
            "the sample index places the seek, not a re-scan from byte 0"
        );

        ports.clear();
        demux
            .process(PipelinePacket::Flush, &mut ports)
            .await
            .expect("the flush");
        feed(&mut demux, &file[landing..], &mut ports).await;
        end(&mut demux, &mut ports).await;

        let stream_start_ns = whole
            .units
            .iter()
            .filter_map(|units| units.first())
            .map(|unit| unit.pts_ns)
            .min()
            .expect("the whole file has frames");
        let target_pts_ns = stream_start_ns + TARGET_NS;
        for (port, seeked) in ports.units.iter().enumerate() {
            let resumed_at = whole.units[port]
                .iter()
                .rposition(|unit| unit.keyframe && unit.pts_ns <= target_pts_ns)
                .expect("a keyframe at or before the target");
            let expected = &whole.units[port][resumed_at..];
            let pts = |units: &[Unit]| units.iter().map(|unit| unit.pts_ns).collect::<Vec<_>>();
            assert_eq!(
                pts(seeked),
                pts(expected),
                "port {port} resumes at its keyframe"
            );
            // the first frame after a flush carries the track's codec config, so its bytes can differ
            assert_eq!(
                seeked[1..],
                expected[1..],
                "port {port} frames after the first"
            );
            assert_eq!(
                ports.segment_starts[port].last(),
                Some(&target_pts_ns),
                "port {port} plays from the target"
            );
        }
    });
}

fn launch_checksums(options: &[&str]) -> Vec<String> {
    let pipeline = format!(
        "filesrc location={} ! qtdemux name=d d.video_0 ! checksumsink d.audio_0 ! fakesink",
        fixture_path()
    );
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .arg("-q")
        .args(options)
        .args(pipeline.split_whitespace())
        .output()
        .expect("g2g-launch starts");
    assert!(
        output.status.success(),
        "g2g-launch fails: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn a_seek_after_the_end_reads_on_from_the_indexed_keyframe() {
    let unseeked = launch_checksums(&[]);
    assert_eq!(unseeked.len(), VIDEO_FRAME_COUNT);
    let last_frame_seconds =
        ((VIDEO_FRAME_COUNT - 1) as u64 * FRAME_PERIOD_NS) as f64 / 1_000_000_000.0;
    let target_seconds = TARGET_NS as f64 / 1_000_000_000.0;
    let seeked = launch_checksums(&[
        "--seek-at",
        &format!("{last_frame_seconds}:{target_seconds}"),
    ]);
    assert_eq!(
        seeked[..unseeked.len()],
        unseeked,
        "the whole run comes before the seek"
    );
    let resumed = &seeked[unseeked.len()..];
    let keyframe =
        (TARGET_NS / FRAME_PERIOD_NS) as usize / FRAMES_PER_KEYFRAME * FRAMES_PER_KEYFRAME;
    let expected = &unseeked[keyframe..];
    let pts = |lines: &[String]| {
        lines
            .iter()
            .map(|line| line.split_whitespace().next().map(str::to_string))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        pts(resumed),
        pts(expected),
        "the run resumes at the keyframe before the target"
    );
    // the first frame after the flush carries the parameter sets, so its digest differs
    assert_eq!(resumed[1..], expected[1..]);
}

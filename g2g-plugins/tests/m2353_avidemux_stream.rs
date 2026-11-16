// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use std::time::Duration;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{
    attach_seek_controller, block_on, parse_launch, RunBuilder, SeekController,
};
use g2g_core::{Bus, BusMessage, G2gError, MultiOutputElement, MultiOutputSink, PushOutcome, Seek};
use g2g_plugins::avidemux::{forwardable_streams, AviDemuxN};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

// ffmpeg -f lavfi -i testsrc=size=64x48:rate=25:duration=2 -f lavfi -i sine=frequency=440:sample_rate=48000:duration=2
//   -ac 1 -c:v libx264 -g 10 -keyint_min 10 -sc_threshold 0 -bf 0 -c:a libmp3lame, the idx1 after the movi list
const FIXTURE: &str = "av_seek_h264_mp3.avi";
// ffprobe and gst-discoverer report 2.04 s, the 85 mp3 frames lame writes for 2 s of audio
const DURATION_NS: u64 = 2_040_000_000;
const CHUNK_LEN: usize = 1024;
// between the keyframes at 0.8 s and 1.2 s of video time
const TARGET_NS: u64 = 1_020_000_000;
const FRAMES_PER_KEYFRAME: usize = 10;
const LINK_CAPACITY: usize = 4;
const BUS_CAPACITY: usize = 1024;
const RUN_TIMEOUT: Duration = Duration::from_secs(10);
const CHUNK_HEADER_LEN: usize = 8;
const LIST_HEADER_LEN: usize = 12;
const VIDEO_PORT: usize = 0;

#[derive(Debug, Clone, PartialEq)]
struct Unit {
    pts_ns: u64,
    keyframe: bool,
    bytes: Vec<u8>,
}

#[derive(Debug, Default)]
struct Ports {
    units: Vec<Vec<Unit>>,
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
        if let PipelinePacket::DataFrame(frame) = packet_slot.take().expect("a packet") {
            self.units[port].push(Unit {
                pts_ns: frame.timing.pts_ns,
                keyframe: frame.timing.keyframe,
                bytes: frame
                    .domain
                    .as_system_slice()
                    .expect("system frame")
                    .to_vec(),
            });
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

// the offset of a chunk directly inside the `RIFF AVI ` list, by id, or by form type for a list
fn top_level_offset(file: &[u8], id: &[u8; 4]) -> usize {
    let mut at = LIST_HEADER_LEN;
    while at + LIST_HEADER_LEN <= file.len() {
        let size =
            u32::from_le_bytes(file[at + 4..at + 8].try_into().expect("four bytes")) as usize;
        if &file[at..at + 4] == id || &file[at + 8..at + LIST_HEADER_LEN] == id {
            return at;
        }
        at += CHUNK_HEADER_LEN + size + size % 2;
    }
    panic!("the fixture has no top-level {id:?}");
}

fn demuxer(file: &[u8]) -> AviDemuxN {
    AviDemuxN::new(forwardable_streams(file))
}

fn ports(demux: &AviDemuxN) -> Ports {
    Ports {
        units: vec![Vec::new(); demux.port_count()],
    }
}

async fn feed(demux: &mut AviDemuxN, bytes: &[u8], ports: &mut Ports) {
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

async fn send(demux: &mut AviDemuxN, packet: PipelinePacket, ports: &mut Ports) {
    demux
        .process(packet, ports)
        .await
        .expect("the packet goes through");
}

async fn whole_file_units(file: &[u8]) -> Ports {
    let mut demux = demuxer(file);
    let mut ports = ports(&demux);
    feed(&mut demux, file, &mut ports).await;
    assert!(
        ports.units.iter().all(Vec::is_empty),
        "without a parked source the file is held whole until Eos"
    );
    send(&mut demux, PipelinePacket::Eos, &mut ports).await;
    ports
}

struct Streaming {
    demux: AviDemuxN,
    app: SeekController,
    upstream: SeekController,
    ports: Ports,
    first_chunk_at: usize,
    idx1_at: usize,
}

// fed up to the first chunk of movi, after the seek to the idx1 and back
async fn streaming_from_the_first_chunk(file: &[u8]) -> Streaming {
    let mut demux = demuxer(file);
    let app = SeekController::new();
    app.enable_seeks_after_end();
    let upstream = SeekController::new();
    upstream.set_stream_len(file.len() as u64);
    MultiOutputElement::set_seek(&mut demux, app.clone(), upstream.clone());
    let mut ports = ports(&demux);
    let first_chunk_at = top_level_offset(file, b"movi") + LIST_HEADER_LEN;
    let idx1_at = top_level_offset(file, b"idx1");

    feed(&mut demux, &file[..first_chunk_at], &mut ports).await;
    let to_index = upstream
        .take_seek(Ok)
        .expect("the demuxer seeks to the idx1");
    assert_eq!(to_index.start, idx1_at as u64);
    send(&mut demux, PipelinePacket::Flush, &mut ports).await;
    feed(&mut demux, &file[idx1_at..], &mut ports).await;
    let back = upstream
        .take_seek(Ok)
        .expect("the demuxer seeks back to the chunks");
    assert_eq!(back.start, first_chunk_at as u64);
    send(&mut demux, PipelinePacket::Flush, &mut ports).await;
    assert!(ports.units.iter().all(Vec::is_empty));
    Streaming {
        demux,
        app,
        upstream,
        ports,
        first_chunk_at,
        idx1_at,
    }
}

#[test]
fn the_chunks_go_out_as_they_arrive_with_the_idx1_keyframe_flags() {
    block_on(async {
        let file = fixture();
        let whole = whole_file_units(&file).await;
        let mut streaming = streaming_from_the_first_chunk(&file).await;
        let half = (streaming.first_chunk_at + streaming.idx1_at) / 2;
        feed(
            &mut streaming.demux,
            &file[streaming.first_chunk_at..half],
            &mut streaming.ports,
        )
        .await;
        assert!(
            streaming.ports.units.iter().all(|units| !units.is_empty()),
            "every port has frames out before the second half of the chunks arrives"
        );
        feed(
            &mut streaming.demux,
            &file[half..streaming.idx1_at],
            &mut streaming.ports,
        )
        .await;
        assert_eq!(streaming.ports.units, whole.units);
        assert_eq!(
            MultiOutputElement::query_duration(&streaming.demux),
            Some(DURATION_NS)
        );
    });
}

#[test]
fn an_app_seek_lands_on_the_keyframe_the_index_names() {
    block_on(async {
        let file = fixture();
        let whole = whole_file_units(&file).await;
        let mut streaming = streaming_from_the_first_chunk(&file).await;
        let half = (streaming.first_chunk_at + streaming.idx1_at) / 2;
        feed(
            &mut streaming.demux,
            &file[streaming.first_chunk_at..half],
            &mut streaming.ports,
        )
        .await;
        streaming.app.seek(Seek::flush_to(TARGET_NS));
        // the demuxer takes the seek at its next packet, and drops that packet's bytes
        feed(
            &mut streaming.demux,
            &file[half..half + 1],
            &mut streaming.ports,
        )
        .await;
        let landing = streaming
            .upstream
            .take_seek(Ok)
            .expect("the demuxer byte-seeks its source")
            .start as usize;
        assert!(
            landing > streaming.first_chunk_at,
            "the index places the seek, not a re-read from the first chunk"
        );

        streaming.ports.units.iter_mut().for_each(Vec::clear);
        send(
            &mut streaming.demux,
            PipelinePacket::Flush,
            &mut streaming.ports,
        )
        .await;
        feed(
            &mut streaming.demux,
            &file[landing..streaming.idx1_at],
            &mut streaming.ports,
        )
        .await;

        let video = &streaming.ports.units[VIDEO_PORT];
        let whole_video = &whole.units[VIDEO_PORT];
        let skipped_frames = whole_video.len() - video.len();
        assert!(
            skipped_frames > 0 && skipped_frames.is_multiple_of(FRAMES_PER_KEYFRAME),
            "video restarts at a keyframe after skipping {skipped_frames} frames"
        );
        assert!(video[0].keyframe);
        assert_eq!(video[1..], whole_video[skipped_frames + 1..]);
        assert!(video[0].pts_ns <= TARGET_NS && video[FRAMES_PER_KEYFRAME].pts_ns > TARGET_NS);
    });
}

async fn run_checksums(with_seek_controller: bool) -> Vec<(String, u64, String)> {
    let line = format!(
        "filesrc location={} blocksize={CHUNK_LEN} ! avidemux name=d \
         d.video_0 ! checksumsink name=video  d.audio_0 ! checksumsink name=audio",
        fixture_path()
    );
    let mut graph = parse_launch(&default_registry(), &line).expect("the line negotiates");
    let app = with_seek_controller
        .then(|| attach_seek_controller(&mut graph).expect("avidemux takes seeks"));
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let clock = WallClock::new();
    let mut builder = RunBuilder::new(graph, &clock, LINK_CAPACITY).with_bus(&handle);
    if let Some(app) = &app {
        builder = builder.with_seek_controller(app);
    }
    tokio::time::timeout(RUN_TIMEOUT, builder.run())
        .await
        .expect("the run ends")
        .expect("the pipeline runs");
    let mut lines = Vec::new();
    while let Some(message) = bus.try_recv() {
        if let BusMessage::Checksum {
            element,
            pts_ns,
            digest,
        } = message
        {
            lines.push((element, pts_ns, digest));
        }
    }
    lines.sort();
    lines
}

#[tokio::test]
async fn a_run_that_takes_seeks_streams_the_file_to_its_end() {
    let held_whole = run_checksums(false).await;
    assert!(!held_whole.is_empty());
    assert_eq!(run_checksums(true).await, held_whole);
}

#[tokio::test]
async fn a_seek_after_the_end_replays_video_from_the_keyframe_before_the_target() {
    let whole = run_checksums(false).await;
    let whole_video: Vec<u64> = whole
        .iter()
        .filter(|(element, ..)| element == "video")
        .map(|(_, pts_ns, _)| *pts_ns)
        .collect();
    let line = format!(
        "filesrc location={} blocksize={CHUNK_LEN} ! avidemux name=d \
         d.video_0 ! checksumsink name=video  d.audio_0 ! fakesink",
        fixture_path()
    );
    let mut graph = parse_launch(&default_registry(), &line).expect("the line negotiates");
    let app = attach_seek_controller(&mut graph).expect("avidemux takes seeks");
    app.keep_run_open();
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let clock = WallClock::new();
    let run = RunBuilder::new(graph, &clock, LINK_CAPACITY)
        .with_bus(&handle)
        .with_seek_controller(&app)
        .run();
    let mut video_pts_ns = Vec::new();
    let mut record = |message: BusMessage| {
        if let BusMessage::Checksum { pts_ns, .. } = message {
            video_pts_ns.push(pts_ns);
            if video_pts_ns.len() == whole_video.len() {
                app.seek(Seek::flush_to(TARGET_NS));
                app.let_run_end();
            }
        }
    };
    tokio::time::timeout(RUN_TIMEOUT, async {
        tokio::pin!(run);
        loop {
            tokio::select! {
                result = &mut run => {
                    result.expect("the pipeline runs");
                    while let Some(message) = bus.try_recv() {
                        record(message);
                    }
                    return;
                }
                Some(message) = bus.recv() => record(message),
            }
        }
    })
    .await
    .expect("the run ends");
    let replayed = video_pts_ns.split_off(whole_video.len());
    assert_eq!(video_pts_ns, whole_video);
    let keyframe_before_target = whole_video
        .iter()
        .step_by(FRAMES_PER_KEYFRAME)
        .copied()
        .filter(|pts_ns| *pts_ns <= TARGET_NS)
        .max()
        .expect("a keyframe before the target");
    let resumed_at = whole_video
        .iter()
        .position(|pts_ns| *pts_ns == keyframe_before_target)
        .expect("the keyframe is in the whole run");
    assert_eq!(replayed, whole_video[resumed_at..]);
}

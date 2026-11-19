// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use std::time::Duration;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{block_on, parse_launch, RunBuilder, SeekController};
use g2g_core::{
    Bus, BusMessage, ByteStreamEncoding, G2gError, MultiOutputElement, MultiOutputSink,
    PushOutcome, Seek,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::mxfdemux::{forwardable_tracks, MxfDemuxN};
use g2g_plugins::registry::default_registry;

#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;

// every fixture is ten frames at 25 fps, see tools/mxf-fixtures.sh
const FRAMES: usize = 10;
const FRAME_NS: u64 = 40_000_000;
const DURATION_NS: u64 = FRAMES as u64 * FRAME_NS;
const CHUNK_LEN: usize = 1024;
const LINK_CAPACITY: usize = 4;
const BUS_CAPACITY: usize = 1024;
const RUN_TIMEOUT: Duration = Duration::from_secs(20);
// the h.264 fixture's second gop starts at frame five
const SECOND_GOP_FRAME: usize = 5;
const TARGET_NS: u64 = 300_000_000;
// 24-bit stereo
const IMF_BLOCK_BYTES: usize = 6;
const IMF_SAMPLE_RATE: u64 = 48_000;
// inside the imf file's 2000 samples
const SOUND_TARGET_NS: u64 = 20_000_000;

#[derive(Debug, Clone, PartialEq)]
struct Unit {
    pts_ns: u64,
    dts_ns: u64,
    keyframe: bool,
    bytes: Vec<u8>,
}

#[derive(Debug, Default)]
struct Ports {
    units: Vec<Vec<Unit>>,
}

impl Ports {
    fn bytes(&self, port: usize) -> Vec<u8> {
        self.units[port]
            .iter()
            .flat_map(|unit| unit.bytes.iter().copied())
            .collect()
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
        if let PipelinePacket::DataFrame(frame) = packet_slot.take().expect("a packet") {
            self.units[port].push(Unit {
                pts_ns: frame.timing.pts_ns,
                dts_ns: frame.timing.dts_ns,
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

fn fixture_path(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(fixture_path(name)).expect("the fixture reads")
}

fn demuxer(file: &[u8]) -> (MxfDemuxN, Ports) {
    let tracks = forwardable_tracks(file);
    let ports = Ports {
        units: vec![Vec::new(); tracks.len()],
    };
    (MxfDemuxN::new(tracks), ports)
}

async fn feed(demux: &mut MxfDemuxN, bytes: &[u8], ports: &mut Ports) {
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

async fn send(demux: &mut MxfDemuxN, packet: PipelinePacket, ports: &mut Ports) {
    demux
        .process(packet, ports)
        .await
        .expect("the packet goes through");
}

async fn whole_file_units(file: &[u8]) -> Ports {
    let (mut demux, mut ports) = demuxer(file);
    feed(&mut demux, file, &mut ports).await;
    send(&mut demux, PipelinePacket::Eos, &mut ports).await;
    ports
}

#[test]
fn every_track_carries_the_packets_ffmpeg_copies() {
    let cases = [
        ("mxf_h264.mxf", ["v"].as_slice()),
        ("mxf_prores.mxf", &["v"]),
        ("mxf_jpeg2000.mxf", &["v"]),
        ("mxf_dnxhr.mxf", &["v"]),
        ("mxf_mpeg2_pcm.mxf", &["v", "a"]),
        ("mxf_pcm_opatom.mxf", &["a"]),
        ("mxf_imf_pcm_clip.mxf", &["a"]),
    ];
    block_on(async {
        for (name, streams) in cases {
            let file = fixture(name);
            assert_eq!(
                g2g_plugins::typefind::sniff(&file),
                Some(ByteStreamEncoding::Mxf)
            );
            let ports = whole_file_units(&file).await;
            assert_eq!(ports.units.len(), streams.len(), "{name}");
            for (port, stream) in streams.iter().enumerate() {
                let copied = fixture(&format!("{name}.{stream}.data"));
                assert!(
                    ports.bytes(port) == copied,
                    "{name} track {port} differs from ffmpeg's copy"
                );
            }
        }
    });
}

#[test]
fn pictures_present_in_the_order_the_index_reorders_them_to() {
    block_on(async {
        let ports = whole_file_units(&fixture("mxf_mpeg2_pcm.mxf")).await;
        let video = &ports.units[0];
        assert_eq!(video.len(), FRAMES);
        assert!(video.iter().all(|unit| unit.pts_ns >= unit.dts_ns));
        let mut presented: Vec<u64> = video.iter().map(|unit| unit.pts_ns).collect();
        assert_ne!(presented, {
            let mut sorted = presented.clone();
            sorted.sort();
            sorted
        });
        presented.sort();
        assert!(presented
            .windows(2)
            .all(|pair| pair[1] - pair[0] == FRAME_NS));
        let first_picture_ns = presented[0];
        let sound = &ports.units[1];
        assert_eq!(
            sound[0].pts_ns, first_picture_ns,
            "sound starts with the first picture shown, not the first stored"
        );
    });
}

struct Streaming {
    demux: MxfDemuxN,
    app: SeekController,
    upstream: SeekController,
    ports: Ports,
}

// fed the header, then the tail it seeks to, then back at byte 0
async fn streaming(file: &[u8]) -> Streaming {
    streaming_with_probes(file, false).await
}

async fn streaming_with_probes(file: &[u8], expect_probes: bool) -> Streaming {
    let (mut demux, mut ports) = demuxer(file);
    let app = SeekController::new();
    app.enable_seeks_after_end();
    let upstream = SeekController::new();
    upstream.set_stream_len(file.len() as u64);
    MultiOutputElement::set_seek(&mut demux, app.clone(), upstream.clone());
    let mut fed = 0;
    let to_tail = loop {
        feed(&mut demux, &file[fed..fed + CHUNK_LEN], &mut ports).await;
        fed += CHUNK_LEN;
        if let Some(seek) = upstream.take_seek(Ok) {
            break seek.start as usize;
        }
    };
    send(&mut demux, PipelinePacket::Flush, &mut ports).await;
    feed(&mut demux, &file[to_tail..], &mut ports).await;
    send(&mut demux, PipelinePacket::Eos, &mut ports).await;
    let mut probes = 0;
    let mut probe_bytes = 0;
    loop {
        let seek = upstream
            .take_seek(Ok)
            .expect("the demuxer seeks for the index or rewinds to play");
        let start = seek.start as usize;
        if start == 0 && (!expect_probes || probes > 0) {
            break;
        }
        assert!(expect_probes && start < to_tail);
        probes += 1;
        send(&mut demux, PipelinePacket::Flush, &mut ports).await;
        for chunk in file[start..].chunks(CHUNK_LEN) {
            feed(&mut demux, chunk, &mut ports).await;
            probe_bytes += chunk.len();
            if upstream.has_pending() {
                break;
            }
        }
        if !upstream.has_pending() {
            send(&mut demux, PipelinePacket::Eos, &mut ports).await;
        }
    }
    if expect_probes {
        assert!(probes >= 3, "probe multiple partitions outside the tail");
        assert!(
            probe_bytes < file.len() / 4,
            "skip essence during the probes"
        );
    }
    send(&mut demux, PipelinePacket::Flush, &mut ports).await;
    assert!(ports.units.iter().all(Vec::is_empty));
    Streaming {
        demux,
        app,
        upstream,
        ports,
    }
}

#[test]
fn a_seekable_run_streams_what_a_held_file_sends() {
    block_on(async {
        for name in ["mxf_h264.mxf", "mxf_mpeg2_pcm.mxf"] {
            let file = fixture(name);
            let whole = whole_file_units(&file).await;
            let mut streaming = streaming(&file).await;
            let split = split_point(&file);
            feed(&mut streaming.demux, &file[..split], &mut streaming.ports).await;
            assert!(
                streaming.ports.units.iter().all(|units| !units.is_empty()),
                "{name} sends every track before its last quarter arrives"
            );
            feed(&mut streaming.demux, &file[split..], &mut streaming.ports).await;
            assert_eq!(streaming.ports.units, whole.units, "{name}");
            assert_eq!(
                MultiOutputElement::query_duration(&streaming.demux),
                Some(DURATION_NS),
                "{name}"
            );
        }
    });
}

fn split_point(file: &[u8]) -> usize {
    file.len() * 3 / 4
}

// seek targets count from the first unit sent, so the seek waits for one on every port
async fn seek_after(streaming: &mut Streaming, file: &[u8], target_ns: u64) -> usize {
    let mut fed = 0;
    while streaming.ports.units.iter().any(Vec::is_empty) {
        feed(
            &mut streaming.demux,
            &file[fed..fed + CHUNK_LEN],
            &mut streaming.ports,
        )
        .await;
        fed += CHUNK_LEN;
    }
    streaming.app.seek(Seek::flush_to(target_ns));
    // the demuxer takes the seek at its next packet, and drops that packet's bytes
    feed(
        &mut streaming.demux,
        &file[fed..fed + 1],
        &mut streaming.ports,
    )
    .await;
    let landing = streaming
        .upstream
        .take_seek(Ok)
        .expect("the demuxer byte-seeks its source")
        .start as usize;
    streaming.ports.units.iter_mut().for_each(Vec::clear);
    send(
        &mut streaming.demux,
        PipelinePacket::Flush,
        &mut streaming.ports,
    )
    .await;
    feed(&mut streaming.demux, &file[landing..], &mut streaming.ports).await;
    landing
}

#[test]
fn a_picture_seek_lands_on_the_keyframe_the_index_names() {
    block_on(async {
        let file = fixture("mxf_h264.mxf");
        let whole = whole_file_units(&file).await;
        let mut streaming = streaming(&file).await;
        let landing = seek_after(&mut streaming, &file, TARGET_NS).await;
        assert!(landing > 0, "the index places the seek");
        let video = &streaming.ports.units[0];
        assert!(video[0].keyframe);
        assert_eq!(video[..], whole.units[0][SECOND_GOP_FRAME..]);
    });
}

#[test]
fn a_seek_into_an_unplayed_body_partition_uses_the_prefetched_index() {
    if !ffmpeg_tool::have_ffmpeg() {
        eprintln!("skipping the multi-partition MXF test without ffmpeg");
        return;
    }
    let path = format!(
        "{}/mxf_multiple_partitions.mxf",
        env!("CARGO_TARGET_TMPDIR")
    );
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=25:duration=34",
            "-c:v",
            "mpeg2video",
            "-g",
            "25",
            "-bf",
            "0",
            "-q:v",
            "8",
            "-f",
            "mxf",
            &path,
        ])
        .status()
        .expect("ffmpeg starts");
    assert!(status.success(), "ffmpeg writes the multi-partition MXF");
    let file = std::fs::read(&path).expect("the file reads");
    std::fs::remove_file(&path).expect("the temporary file is removed");
    block_on(async {
        let whole = whole_file_units(&file).await;
        let mut streaming = streaming_with_probes(&file, true).await;
        let landing = seek_after(&mut streaming, &file, 12_480_000_000).await;
        assert!(landing > file.len() / 4, "seek into a later body partition");
        let video = &streaming.ports.units[0];
        assert!(video[0].keyframe);
        assert_eq!(video[0].pts_ns, 12_000_000_000);
        assert_eq!(&video[..], &whole.units[0][300..]);
    });
}

#[test]
fn a_clip_wrapped_sound_seek_lands_on_its_sample() {
    block_on(async {
        let file = fixture("mxf_imf_pcm_clip.mxf");
        let copied = fixture("mxf_imf_pcm_clip.mxf.a.data");
        let mut streaming = streaming(&file).await;
        seek_after(&mut streaming, &file, SOUND_TARGET_NS).await;
        let sound = &streaming.ports.units[0];
        assert_eq!(sound[0].pts_ns, SOUND_TARGET_NS);
        let target_byte =
            (SOUND_TARGET_NS * IMF_SAMPLE_RATE / 1_000_000_000) as usize * IMF_BLOCK_BYTES;
        assert!(streaming.ports.bytes(0) == copied[target_byte..]);
    });
}

async fn checksum_count(line: &str) -> usize {
    let graph = parse_launch(&default_registry(), line).expect("the line negotiates");
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let clock = WallClock::new();
    tokio::time::timeout(
        RUN_TIMEOUT,
        RunBuilder::new(graph, &clock, LINK_CAPACITY)
            .with_bus(&handle)
            .run(),
    )
    .await
    .expect("the run ends")
    .expect("the pipeline runs");
    let mut count = 0;
    while let Some(message) = bus.try_recv() {
        if matches!(message, BusMessage::Checksum { .. }) {
            count += 1;
        }
    }
    count
}

#[tokio::test]
async fn a_named_demuxer_fans_out_its_tracks() {
    let line = format!(
        "filesrc location={} ! mxfdemux name=d \
         d.video_0 ! checksumsink  d.audio_0 ! fakesink",
        fixture_path("mxf_mpeg2_pcm.mxf")
    );
    assert_eq!(checksum_count(&line).await, FRAMES);
}

#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn decodebin_decodes_every_picture_essence() {
    for name in [
        "mxf_h264.mxf",
        "mxf_prores.mxf",
        "mxf_jpeg2000.mxf",
        "mxf_jpeg2000_rgb24.mxf",
        "mxf_dnxhr.mxf",
        "mxf_mpeg2_pcm.mxf",
    ] {
        let line = format!(
            "filesrc location={} ! decodebin name=d d.video_0 ! checksumsink",
            fixture_path(name)
        );
        assert_eq!(checksum_count(&line).await, FRAMES, "{name}");
    }
}

#[test]
fn an_mxf_holding_program_stream_bytes_plays_as_mxf() {
    // a dcp's jpeg 2000 essence can hold the byte runs a program stream starts with
    let mut file = fixture("mxf_jpeg2000.mxf");
    file.extend(fixture("av_mpeg2_mp2.mpg"));
    let path = format!(
        "{}/mxf_holding_program_stream.mxf",
        env!("CARGO_TARGET_TMPDIR")
    );
    std::fs::write(&path, file).expect("the test file writes");
    let uri = format!("file://{path}");
    let registry = default_registry();
    let program_stream = g2g_plugins::uridecodebin::ps_uri_fanout(&registry, &uri)
        .expect("the program stream probe runs");
    let mxf =
        g2g_plugins::uridecodebin::mxf_uri_fanout(&registry, &uri).expect("the mxf probe runs");
    assert!(program_stream.is_none());
    assert!(mxf.is_some());
}

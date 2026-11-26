// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use std::time::Duration;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::runtime::{
    attach_seek_controller, block_on, parse_launch, RunBuilder, SeekController,
};
use g2g_core::{Bus, BusMessage, G2gError, MultiOutputElement, MultiOutputSink, PushOutcome};
use g2g_plugins::clock::WallClock;
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};
use g2g_plugins::registry::default_registry;

// ffmpeg's default layout, the moov written after the mdat
const FIXTURE: &str = "av_seek_h264_aac.mp4";
const CHUNK_LEN: usize = 1024;
const LINK_CAPACITY: usize = 4;
const BUS_CAPACITY: usize = 1024;
const RUN_TIMEOUT: Duration = Duration::from_secs(10);
const BOX_HEADER_LEN: usize = 8;

#[derive(Debug, Default)]
struct Ports {
    units: Vec<Vec<(u64, Vec<u8>)>>,
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
            let bytes = frame
                .domain
                .as_system_slice()
                .expect("system frame")
                .to_vec();
            self.units[port].push((frame.timing.pts_ns, bytes));
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn fixture_path() -> String {
    format!("{}/tests/fixtures/{FIXTURE}", env!("CARGO_MANIFEST_DIR"))
}

// the offset of each top-level box, by fourcc
fn box_offset(file: &[u8], kind: &[u8; 4]) -> usize {
    let mut at = 0;
    while at + BOX_HEADER_LEN <= file.len() {
        let size = u32::from_be_bytes(file[at..at + 4].try_into().expect("four bytes")) as usize;
        if &file[at + 4..at + BOX_HEADER_LEN] == kind {
            return at;
        }
        at += size;
    }
    panic!("the fixture has no top-level {kind:?}");
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

fn ports(demux: &Mp4DemuxN) -> Ports {
    Ports {
        units: vec![Vec::new(); demux.port_count()],
    }
}

async fn feed(demux: &mut Mp4DemuxN, bytes: &[u8], ports: &mut Ports) {
    for chunk in bytes.chunks(CHUNK_LEN) {
        let frame = Frame::from_system_bytes(chunk.to_vec(), FrameTiming::default(), 0);
        demux
            .process(PipelinePacket::DataFrame(frame), ports)
            .await
            .expect("the bytes demux");
    }
}

async fn flush(demux: &mut Mp4DemuxN, ports: &mut Ports) {
    demux
        .process(PipelinePacket::Flush, ports)
        .await
        .expect("the flush");
}

async fn whole_file_units(file: &[u8]) -> Ports {
    let mut demux = demuxer(file);
    let mut ports = ports(&demux);
    feed(&mut demux, file, &mut ports).await;
    demux
        .process(PipelinePacket::Eos, &mut ports)
        .await
        .expect("the stream ends");
    ports
}

#[test]
fn the_demuxer_reads_the_moov_past_the_mdat_then_streams_the_samples() {
    block_on(async {
        let file = std::fs::read(fixture_path()).expect("the fixture reads");
        let mdat_at = box_offset(&file, b"mdat");
        let moov_at = box_offset(&file, b"moov");
        assert!(mdat_at < moov_at, "the fixture stores its moov last");
        let whole = whole_file_units(&file).await;

        let mut demux = demuxer(&file);
        let app = SeekController::new();
        app.enable_seeks_after_end();
        let upstream = SeekController::new();
        upstream.set_stream_len(file.len() as u64);
        MultiOutputElement::set_seek(&mut demux, app, upstream.clone());
        let mut streamed = ports(&demux);

        feed(&mut demux, &file[..mdat_at + BOX_HEADER_LEN], &mut streamed).await;
        let to_moov = upstream
            .take_seek(Ok)
            .expect("the demuxer seeks past the mdat");
        assert_eq!(to_moov.start, moov_at as u64);
        flush(&mut demux, &mut streamed).await;
        feed(&mut demux, &file[moov_at..], &mut streamed).await;
        let to_samples = upstream
            .take_seek(Ok)
            .expect("the demuxer seeks back to the samples");
        let landing = to_samples.start as usize;
        assert!(
            (mdat_at + BOX_HEADER_LEN..moov_at).contains(&landing),
            "the seek back lands inside the mdat, at byte {landing}"
        );
        assert!(streamed.units.iter().all(Vec::is_empty));

        flush(&mut demux, &mut streamed).await;
        feed(&mut demux, &file[landing..moov_at], &mut streamed).await;
        assert_eq!(
            streamed.units, whole.units,
            "every sample goes out before the source reaches the moov again"
        );
    });
}

async fn run_checksums(with_seek_controller: bool) -> Vec<(String, u64, String)> {
    let line = format!(
        "filesrc location={} blocksize={CHUNK_LEN} ! qtdemux name=d \
         d.video_0 ! checksumsink name=video  d.audio_0 ! checksumsink name=audio",
        fixture_path()
    );
    let mut graph = parse_launch(&default_registry(), &line).expect("the line negotiates");
    let app = with_seek_controller
        .then(|| attach_seek_controller(&mut graph).expect("qtdemux takes seeks"));
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
async fn a_run_that_takes_seeks_plays_the_moov_last_file_to_its_end() {
    let held_whole = run_checksums(false).await;
    assert!(!held_whole.is_empty());
    assert_eq!(run_checksums(true).await, held_whole);
}

//! M1130: `audioreverse` re-emits each chunk of audio with its sample frames in
//! the opposite order, and flushes the last partial chunk at `Eos`.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, PushOutcome,
};
use g2g_plugins::audioreverse::AudioReverse;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

const RATE: u32 = 48_000;
const CHANNELS: u8 = 1;
/// 480 sample frames at 48 kHz, small enough to check by hand.
const CHUNK_DURATION_NS: u64 = 10_000_000;
const CHUNK_FRAMES: usize = (RATE as u64 * CHUNK_DURATION_NS / 1_000_000_000) as usize;
const NS_PER_SECOND: u64 = 1_000_000_000;

fn caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: CHANNELS,
        sample_rate: RATE,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// A ramp `0, 1, 2, ...`: every sample says where it sat in the input, so the
/// expected output order is readable straight off the values.
fn ramp(frames: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(frames * 2);
    for index in 0..frames {
        bytes.extend_from_slice(&(index as i16).to_le_bytes());
    }
    bytes
}

fn samples(bytes: &[u8]) -> Vec<i16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|s| i16::from_le_bytes(*s))
        .collect()
}

#[derive(Default)]
struct Collect {
    buffers: Vec<(u64, u64, Vec<i16>)>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            if let Some(slice) = frame.domain.as_system_slice() {
                self.buffers.push((
                    frame.timing.pts_ns,
                    frame.timing.duration_ns,
                    samples(slice),
                ));
            }
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn frame(bytes: Vec<u8>, pts_ns: u64) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..Default::default()
        },
        0,
    )
}

/// Push one buffer then `Eos`, collecting everything the element emits.
fn run(element: &mut AudioReverse, bytes: Vec<u8>) -> Collect {
    let mut sink = Collect::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime.block_on(async {
        element
            .process(PipelinePacket::DataFrame(frame(bytes, 0)), &mut sink)
            .await
            .unwrap();
        element
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
    });
    sink
}

fn configured() -> AudioReverse {
    let mut element = AudioReverse::new().with_chunk_duration(CHUNK_DURATION_NS);
    element.configure_pipeline(&caps()).unwrap();
    element
}

/// The frames a chunk starting at `start` holds, in the order they should leave.
fn expected_chunk(start: usize, frames: usize) -> Vec<i16> {
    (0..frames)
        .map(|index| (start + frames - 1 - index) as i16)
        .collect()
}

#[test]
fn whole_chunks_come_out_reversed_and_in_order() {
    let chunks = 3;
    let mut element = configured();
    let collected = run(&mut element, ramp(CHUNK_FRAMES * chunks));

    assert_eq!(collected.buffers.len(), chunks);
    for (index, (pts, duration, values)) in collected.buffers.iter().enumerate() {
        assert_eq!(
            *values,
            expected_chunk(index * CHUNK_FRAMES, CHUNK_FRAMES),
            "chunk {index} runs backwards over its own samples"
        );
        assert_eq!(*pts, index as u64 * CHUNK_DURATION_NS, "chunk {index} pts");
        assert_eq!(*duration, CHUNK_DURATION_NS, "chunk {index} duration");
    }
    // the whole run is monotonically ascending, chunk to chunk.
    assert!(collected
        .buffers
        .windows(2)
        .all(|pair| pair[0].0 < pair[1].0));
}

#[test]
fn eos_flushes_the_partial_chunk_reversed() {
    let tail_frames = CHUNK_FRAMES / 4;
    let mut element = configured();
    let collected = run(&mut element, ramp(CHUNK_FRAMES + tail_frames));

    assert_eq!(collected.buffers.len(), 2, "the tail is not dropped");
    let (pts, duration, values) = &collected.buffers[1];
    assert_eq!(*values, expected_chunk(CHUNK_FRAMES, tail_frames));
    assert_eq!(*pts, CHUNK_DURATION_NS);
    assert_eq!(
        *duration,
        tail_frames as u64 * NS_PER_SECOND / RATE as u64,
        "the tail is stamped for the samples it actually holds"
    );
}

#[test]
fn a_zero_duration_reverses_the_whole_stream_at_eos() {
    let frames = CHUNK_FRAMES * 3;
    let mut element = AudioReverse::new().with_chunk_duration(0);
    element.configure_pipeline(&caps()).unwrap();
    let collected = run(&mut element, ramp(frames));

    assert_eq!(collected.buffers.len(), 1, "nothing leaves before Eos");
    assert_eq!(collected.buffers[0].2, expected_chunk(0, frames));
}

/// A launch line reaches the element and its `chunk-duration` takes effect:
/// zero holds every source buffer back until `Eos`, so the sink sees one.
#[tokio::test]
async fn launch_sets_the_chunk_duration() {
    let registry = default_registry();
    let graph = parse_launch(
        &registry,
        "audiotestsrc num-buffers=5 ! audioreverse chunk-duration=0 ! fakesink",
    )
    .expect("the audioreverse pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the audioreverse pipeline runs");
    assert_eq!(
        stats.frames_consumed, 1,
        "the five source buffers leave as one reversed buffer"
    );
}

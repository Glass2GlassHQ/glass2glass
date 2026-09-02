//! M1066 `audiorate`: a gap in the timestamps is filled with silence, samples
//! that overlap what was already emitted are dropped, and jitter under the
//! tolerance is only re-stamped, so the output pts advances by exactly the
//! samples each frame carries.
#![cfg(feature = "std")]

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, G2gError, MemoryDomain, OutputSink, PipelineClock, PropValue,
    PushOutcome,
};
use g2g_plugins::audiorate::AudioRate;
use g2g_plugins::registry::default_registry;

const NS_PER_SECOND: u64 = 1_000_000_000;
const RATE: u32 = 48_000;
const CHANNELS: u8 = 2;
const BYTES_PER_FRAME: usize = 2 * CHANNELS as usize;
/// One test buffer: 480 sample frames, 10 ms at 48 kHz.
const BUFFER_SAMPLES: u64 = 480;
const BUFFER_NS: u64 = 10_000_000;

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl Collect {
    /// (pts, duration, payload) of every data frame pushed.
    fn frames(&self) -> Vec<(u64, u64, Vec<u8>)> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some((
                    f.timing.pts_ns,
                    f.timing.duration_ns,
                    f.domain.as_system_slice().expect("system frame").to_vec(),
                )),
                _ => None,
            })
            .collect()
    }
}

fn caps(format: AudioFormat, channels: u8, sample_rate: u32) -> Caps {
    Caps::Audio {
        format,
        channels,
        sample_rate,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// A buffer of `samples` sample frames stamped at `pts`, each byte `fill` so a
/// re-stamped or trimmed payload is identifiable.
fn frame(pts: u64, samples: u64, bytes_per_frame: usize, fill: u8) -> PipelinePacket {
    let bytes = vec![fill; samples as usize * bytes_per_frame];
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming {
            pts_ns: pts,
            dts_ns: pts,
            duration_ns: samples * NS_PER_SECOND / RATE as u64,
            ..Default::default()
        },
        0,
    ))
}

fn counters(element: &AudioRate) -> (u64, u64, u64, u64) {
    (
        element.get_property("in").and_then(as_uint).expect("in"),
        element.get_property("out").and_then(as_uint).expect("out"),
        element.get_property("add").and_then(as_uint).expect("add"),
        element
            .get_property("drop")
            .and_then(as_uint)
            .expect("drop"),
    )
}

fn as_uint(value: PropValue) -> Option<u64> {
    match value {
        PropValue::Uint(v) => Some(v),
        _ => None,
    }
}

#[tokio::test]
async fn gap_is_filled_with_exactly_the_missing_samples() {
    let mut element = AudioRate::new().with_tolerance(0);
    element
        .configure_pipeline(&caps(AudioFormat::PcmS16Le, CHANNELS, RATE))
        .expect("s16 stereo configures");
    let mut out = Collect::default();
    element
        .process(frame(0, BUFFER_SAMPLES, BYTES_PER_FRAME, 0x11), &mut out)
        .await
        .unwrap();
    // 20 ms late: two buffers' worth of silence is missing.
    let late = 3 * BUFFER_NS;
    element
        .process(frame(late, BUFFER_SAMPLES, BYTES_PER_FRAME, 0x22), &mut out)
        .await
        .unwrap();

    let frames = out.frames();
    assert_eq!(frames.len(), 3, "the gap is one silence buffer");
    let gap_samples = 2 * BUFFER_SAMPLES;
    assert_eq!(
        frames[1].2.len(),
        gap_samples as usize * BYTES_PER_FRAME,
        "silence covers exactly the missing samples"
    );
    assert!(
        frames[1].2.iter().all(|&b| b == 0),
        "s16 silence is all-zero"
    );
    // pts advances by each frame's own samples, and the late frame keeps its
    // original timestamp because the fill made the stream contiguous.
    assert_eq!(frames[0].0, 0);
    assert_eq!(frames[1].0, BUFFER_NS);
    assert_eq!(frames[2].0, late);
    assert_eq!(frames[0].1, BUFFER_NS, "duration is samples over the rate");
    assert_eq!(frames[1].1, 2 * BUFFER_NS);
    assert_eq!(
        counters(&element),
        (2 * BUFFER_SAMPLES, 4 * BUFFER_SAMPLES, gap_samples, 0)
    );
}

#[tokio::test]
async fn overlapping_samples_are_dropped() {
    let mut element = AudioRate::new().with_tolerance(0);
    element
        .configure_pipeline(&caps(AudioFormat::PcmS16Le, CHANNELS, RATE))
        .expect("s16 stereo configures");
    let mut out = Collect::default();
    element
        .process(frame(0, BUFFER_SAMPLES, BYTES_PER_FRAME, 0x11), &mut out)
        .await
        .unwrap();
    // starts 5 ms back into what was already emitted: half of it overlaps.
    let overlap_samples = BUFFER_SAMPLES / 2;
    element
        .process(
            frame(BUFFER_NS / 2, BUFFER_SAMPLES, BYTES_PER_FRAME, 0x22),
            &mut out,
        )
        .await
        .unwrap();

    let frames = out.frames();
    assert_eq!(frames.len(), 2);
    assert_eq!(
        frames[1].2.len(),
        (BUFFER_SAMPLES - overlap_samples) as usize * BYTES_PER_FRAME,
        "only the leading overlap is cut"
    );
    assert!(
        frames[1].2.iter().all(|&b| b == 0x22),
        "the kept tail is the new frame's own samples"
    );
    assert_eq!(
        frames[1].0, BUFFER_NS,
        "the remainder starts where the stream left off"
    );
    assert_eq!(counters(&element).3, overlap_samples);

    // a frame that ends before the expected time is dropped whole.
    element
        .process(frame(0, BUFFER_SAMPLES, BYTES_PER_FRAME, 0x33), &mut out)
        .await
        .unwrap();
    assert_eq!(out.frames().len(), 2, "nothing new reached the sink");
    assert_eq!(counters(&element).3, overlap_samples + BUFFER_SAMPLES);
}

#[tokio::test]
async fn jitter_within_tolerance_is_only_restamped() {
    // gst's 40 ms default tolerance; the 2 ms of jitter below is well inside it.
    let mut element = AudioRate::new();
    element
        .configure_pipeline(&caps(AudioFormat::PcmS16Le, CHANNELS, RATE))
        .expect("s16 stereo configures");
    let mut out = Collect::default();
    element
        .process(frame(0, BUFFER_SAMPLES, BYTES_PER_FRAME, 0x11), &mut out)
        .await
        .unwrap();
    element
        .process(
            frame(BUFFER_NS + 2_000_000, BUFFER_SAMPLES, BYTES_PER_FRAME, 0x22),
            &mut out,
        )
        .await
        .unwrap();

    let frames = out.frames();
    assert_eq!(frames.len(), 2, "no silence inserted, nothing dropped");
    assert_eq!(frames[1].0, BUFFER_NS, "the jitter is re-stamped away");
    assert_eq!(
        counters(&element),
        (2 * BUFFER_SAMPLES, 2 * BUFFER_SAMPLES, 0, 0)
    );
}

#[tokio::test]
async fn u8_silence_sits_at_the_offset_binary_midpoint() {
    const U8_RATE: u32 = 8_000;
    const U8_SAMPLES: u64 = 8;
    const U8_BUFFER_NS: u64 = 1_000_000;
    let mut element = AudioRate::new().with_tolerance(0);
    element
        .configure_pipeline(&caps(AudioFormat::PcmU8, 1, U8_RATE))
        .expect("u8 mono configures");
    let mut out = Collect::default();
    element
        .process(frame(0, U8_SAMPLES, 1, 0x11), &mut out)
        .await
        .unwrap();
    element
        .process(frame(2 * U8_BUFFER_NS, U8_SAMPLES, 1, 0x22), &mut out)
        .await
        .unwrap();

    let frames = out.frames();
    assert_eq!(frames.len(), 3);
    assert_eq!(frames[1].2.len(), U8_SAMPLES as usize);
    assert!(
        frames[1].2.iter().all(|&b| b == 0x80),
        "u8 silence is 0x80, not 0"
    );
}

#[tokio::test]
async fn flush_restarts_the_grid() {
    let mut element = AudioRate::new().with_tolerance(0);
    element
        .configure_pipeline(&caps(AudioFormat::PcmS16Le, CHANNELS, RATE))
        .expect("s16 stereo configures");
    let mut out = Collect::default();
    element
        .process(frame(0, BUFFER_SAMPLES, BYTES_PER_FRAME, 0x11), &mut out)
        .await
        .unwrap();
    element
        .process(PipelinePacket::Flush, &mut out)
        .await
        .unwrap();
    // after the flush the next frame defines the start again, so its far-off
    // timestamp is kept rather than filled up to.
    let resume = 5 * BUFFER_NS;
    element
        .process(
            frame(resume, BUFFER_SAMPLES, BYTES_PER_FRAME, 0x22),
            &mut out,
        )
        .await
        .unwrap();

    let frames = out.frames();
    assert_eq!(frames.len(), 2, "no silence bridges a flush");
    assert_eq!(frames[1].0, resume);
    assert_eq!(counters(&element).2, 0);
}

#[tokio::test]
async fn ragged_buffer_fails_loud() {
    let mut element = AudioRate::new();
    element
        .configure_pipeline(&caps(AudioFormat::PcmS16Le, CHANNELS, RATE))
        .expect("s16 stereo configures");
    let mut out = Collect::default();
    // 3 bytes is not a whole s16 stereo sample frame (4 bytes).
    let bytes = vec![0u8; 3];
    let packet = PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ));
    assert_eq!(
        element.process(packet, &mut out).await.unwrap_err(),
        G2gError::CapsMismatch
    );
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn audiorate_runs_in_a_text_pipeline() {
    let reg = default_registry();
    let graph = parse_launch(&reg, "audiotestsrc num-buffers=5 ! audiorate ! fakesink")
        .expect("pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("audiorate pipeline runs");
    assert_eq!(
        stats.frames_consumed, 5,
        "a contiguous source passes through untouched"
    );
}

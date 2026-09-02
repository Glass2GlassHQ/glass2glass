//! M1075 `scaletempo`: a segment rate other than 1 stretches the audio to the
//! playback speed without moving its pitch, re-stamps the output onto a rate-1
//! timeline, and rewrites the segment it forwards.
#![cfg(feature = "std")]

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::segment::Segment;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, G2gError, MemoryDomain, OutputSink, PipelineClock, PropValue,
    PushOutcome,
};
use g2g_plugins::registry::default_registry;
use g2g_plugins::scaletempo::ScaleTempo;

const NS_PER_SECOND: u64 = 1_000_000_000;
const SAMPLE_RATE: u32 = 48_000;
/// Mono, so the zero-crossing estimate reads one waveform.
const CHANNELS: u8 = 1;
const BYTES_PER_FRAME: usize = 2 * CHANNELS as usize;
const TONE_HZ: u32 = 440;
/// Half full scale, the level `audiotestsrc` generates at.
const TONE_AMPLITUDE: f64 = i16::MAX as f64 / 2.0;
/// One test buffer: 480 sample frames, 10 ms at 48 kHz.
const FRAMES_PER_BUFFER: u64 = 480;
const TONE_SECONDS: u64 = 2;
const FAST_RATE: f64 = 2.0;
const SLOW_RATE: f64 = 0.5;
/// Zero-crossing estimates are noisy at the stride seams; 5 % separates 440 Hz
/// from the 880 Hz plain decimation would give.
const PITCH_TOLERANCE: f64 = 0.05;

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
    /// (pts, payload) of every data frame pushed.
    fn frames(&self) -> Vec<(u64, Vec<u8>)> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some((
                    f.timing.pts_ns,
                    f.domain.as_system_slice().expect("system frame").to_vec(),
                )),
                _ => None,
            })
            .collect()
    }

    fn segments(&self) -> Vec<Segment> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::Segment(s) => Some(*s),
                _ => None,
            })
            .collect()
    }

    /// Every output sample, in order.
    fn samples(&self) -> Vec<i16> {
        self.frames()
            .iter()
            .flat_map(|(_, bytes)| {
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| i16::from_le_bytes(*c))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn sample_frames(&self) -> u64 {
        self.samples().len() as u64 / CHANNELS as u64
    }
}

fn caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: CHANNELS,
        sample_rate: SAMPLE_RATE,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// `frames` sample frames of the test tone, starting at sample index `first`.
fn sine_buffer(first: u64, frames: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(frames as usize * BYTES_PER_FRAME);
    for i in 0..frames {
        let seconds = (first + i) as f64 / SAMPLE_RATE as f64;
        let angle = core::f64::consts::TAU * TONE_HZ as f64 * seconds;
        let value = (TONE_AMPLITUDE * angle.sin()) as i16;
        for _ in 0..CHANNELS {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn data_packet(pts: u64, bytes: Vec<u8>) -> PipelinePacket {
    let frames = bytes.len() as u64 / BYTES_PER_FRAME as u64;
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming {
            pts_ns: pts,
            dts_ns: pts,
            duration_ns: frames * NS_PER_SECOND / SAMPLE_RATE as u64,
            ..Default::default()
        },
        0,
    ))
}

fn segment_packet(rate: f64) -> PipelinePacket {
    PipelinePacket::Segment(Segment {
        rate,
        ..Segment::new()
    })
}

fn pts_of(sample: u64) -> u64 {
    sample * NS_PER_SECOND / SAMPLE_RATE as u64
}

fn uint_property(element: &ScaleTempo, name: &str) -> u64 {
    match element.get_property(name) {
        Some(PropValue::Uint(v)) => v,
        other => panic!("`{name}` reads back as a uint, got {other:?}"),
    }
}

fn double_property(element: &ScaleTempo, name: &str) -> f64 {
    match element.get_property(name) {
        Some(PropValue::Double(v)) => v,
        other => panic!("`{name}` reads back as a double, got {other:?}"),
    }
}

/// Sample frames of one output stride, and the frames the element has to queue
/// before it can emit the first one, both read from the element's own settings.
fn stride_and_priming(element: &ScaleTempo) -> (u64, u64) {
    let per_ms = SAMPLE_RATE as u64 / 1000;
    let stride = uint_property(element, "stride") * per_ms;
    let overlap = (stride as f64 * double_property(element, "overlap")) as u64;
    let search = uint_property(element, "search") * per_ms;
    (stride, stride + overlap + search)
}

/// Push `TONE_SECONDS` of tone through a freshly configured element at `rate`,
/// returning the collected output and the sample frames fed in.
async fn stretch_tone(rate: f64) -> (ScaleTempo, Collect, u64) {
    let mut element = ScaleTempo::new();
    element.configure_pipeline(&caps()).expect("s16 mono");
    let mut out = Collect::default();
    element
        .process(segment_packet(rate), &mut out)
        .await
        .unwrap();
    let total = TONE_SECONDS * SAMPLE_RATE as u64;
    let mut pushed = 0;
    while pushed < total {
        let frames = FRAMES_PER_BUFFER.min(total - pushed);
        element
            .process(
                data_packet(pts_of(pushed), sine_buffer(pushed, frames)),
                &mut out,
            )
            .await
            .unwrap();
        pushed += frames;
    }
    element
        .process(PipelinePacket::Eos, &mut out)
        .await
        .unwrap();
    (element, out, total)
}

/// Dominant frequency of a mono signal, from its zero crossings: a full period
/// crosses zero twice.
fn dominant_frequency(samples: &[i16]) -> f64 {
    let mut crossings = 0u64;
    let mut previous = 0i16;
    for &sample in samples {
        if sample == 0 {
            continue;
        }
        if previous != 0 && (sample > 0) != (previous > 0) {
            crossings += 1;
        }
        previous = sample;
    }
    let seconds = samples.len() as f64 / SAMPLE_RATE as f64;
    crossings as f64 / (2.0 * seconds)
}

/// Every output frame starts where the previous one ended, on the rate-1
/// timeline (its own sample count, not the input's).
fn assert_contiguous(frames: &[(u64, Vec<u8>)], first_pts: u64) {
    let mut expected = first_pts;
    let mut emitted = 0u64;
    for (index, (pts, bytes)) in frames.iter().enumerate() {
        assert_eq!(*pts, expected, "frame {index} follows the previous one");
        emitted += bytes.len() as u64 / BYTES_PER_FRAME as u64;
        expected = first_pts + emitted * NS_PER_SECOND / SAMPLE_RATE as u64;
    }
}

#[tokio::test]
async fn rate_one_passes_the_bytes_and_timestamps_through() {
    let mut element = ScaleTempo::new();
    element.configure_pipeline(&caps()).expect("s16 mono");
    let mut out = Collect::default();
    // no segment: the element starts at rate 1.
    let mut sent = Vec::new();
    for buffer in 0..3u64 {
        let first = buffer * FRAMES_PER_BUFFER;
        let bytes = sine_buffer(first, FRAMES_PER_BUFFER);
        sent.push((pts_of(first), bytes.clone()));
        element
            .process(data_packet(pts_of(first), bytes), &mut out)
            .await
            .unwrap();
    }
    assert_eq!(out.frames(), sent, "rate 1 changes neither bytes nor pts");
}

#[tokio::test]
async fn rate_one_runs_in_a_text_pipeline() {
    const SOURCE_BUFFERS: u64 = 10;
    let reg = default_registry();
    let line = format!("audiotestsrc num-buffers={SOURCE_BUFFERS} ! scaletempo ! fakesink");
    let graph = parse_launch(&reg, &line).expect("pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("scaletempo pipeline runs");
    assert_eq!(stats.frames_consumed, SOURCE_BUFFERS);
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn rate_two_halves_the_stream_and_rewrites_the_segment() {
    let (element, out, input_frames) = stretch_tone(FAST_RATE).await;
    let (stride, priming) = stride_and_priming(&element);

    let expected = (input_frames as f64 / FAST_RATE) as u64;
    let produced = out.sample_frames();
    // the queue has to fill before the first stride comes out, so the output is
    // short by that priming (stretched) plus the stride the division rounds off.
    let tolerance = (priming as f64 / FAST_RATE) as u64 + stride;
    assert!(
        produced <= expected && expected - produced <= tolerance,
        "{produced} frames out for {input_frames} in, want ~{expected} (tolerance {tolerance})"
    );

    assert_contiguous(&out.frames(), 0);

    let segments = out.segments();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].rate, 1.0, "downstream sees a rate-1 stream");
    assert_eq!(segments[0].applied_rate, FAST_RATE);
}

#[tokio::test]
async fn rate_half_doubles_the_stream() {
    let (element, out, input_frames) = stretch_tone(SLOW_RATE).await;
    let (stride, priming) = stride_and_priming(&element);

    let expected = (input_frames as f64 / SLOW_RATE) as u64;
    let produced = out.sample_frames();
    let tolerance = (priming as f64 / SLOW_RATE) as u64 + stride;
    assert!(
        produced <= expected && expected - produced <= tolerance,
        "{produced} frames out for {input_frames} in, want ~{expected} (tolerance {tolerance})"
    );
    assert_eq!(out.segments()[0].applied_rate, SLOW_RATE);
}

#[tokio::test]
async fn double_speed_keeps_the_pitch() {
    let (_, out, _) = stretch_tone(FAST_RATE).await;
    let measured = dominant_frequency(&out.samples());
    let wanted = TONE_HZ as f64;
    assert!(
        (measured - wanted).abs() <= wanted * PITCH_TOLERANCE,
        "{measured} Hz out of a {wanted} Hz tone at {FAST_RATE}x; plain decimation would give {}",
        wanted * FAST_RATE
    );
}

#[tokio::test]
async fn flush_drops_the_queue_and_restarts_the_timeline() {
    let mut element = ScaleTempo::new();
    element.configure_pipeline(&caps()).expect("s16 mono");
    let mut out = Collect::default();
    element
        .process(segment_packet(FAST_RATE), &mut out)
        .await
        .unwrap();
    let (_, priming) = stride_and_priming(&element);
    let buffers = priming.div_ceil(FRAMES_PER_BUFFER);

    for buffer in 0..buffers {
        let first = buffer * FRAMES_PER_BUFFER;
        element
            .process(
                data_packet(pts_of(first), sine_buffer(first, FRAMES_PER_BUFFER)),
                &mut out,
            )
            .await
            .unwrap();
    }
    assert!(!out.frames().is_empty(), "a full queue produced a stride");
    let before_flush = out.frames().len();

    element
        .process(PipelinePacket::Flush, &mut out)
        .await
        .unwrap();

    // resume a second into the stream: the output restarts at that timestamp
    // mapped onto the compressed timeline, with nothing bridging the flush.
    let resume = SAMPLE_RATE as u64;
    for buffer in 0..buffers {
        let first = resume + buffer * FRAMES_PER_BUFFER;
        element
            .process(
                data_packet(pts_of(first), sine_buffer(first, FRAMES_PER_BUFFER)),
                &mut out,
            )
            .await
            .unwrap();
    }
    let frames = out.frames();
    assert!(
        frames.len() > before_flush,
        "the queue refilled after the flush"
    );
    let after = &frames[before_flush..];
    let expected_pts = (pts_of(resume) as f64 / FAST_RATE) as u64;
    assert_eq!(after[0].0, expected_pts);
    assert_contiguous(after, expected_pts);
}

#[tokio::test]
async fn ragged_buffer_fails_loud() {
    let mut element = ScaleTempo::new();
    element.configure_pipeline(&caps()).expect("s16 mono");
    let mut out = Collect::default();
    // 3 bytes is not a whole s16 mono sample frame pair; one byte short of two.
    let packet = data_packet(0, vec![0u8; BYTES_PER_FRAME + 1]);
    assert_eq!(
        element.process(packet, &mut out).await.unwrap_err(),
        G2gError::CapsMismatch
    );
}

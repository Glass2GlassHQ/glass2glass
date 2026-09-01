//! M1067 `imagefreeze`: one still frame becomes a stream at the configured
//! framerate, bounded by `num-buffers` or by the output push failing.
#![cfg(feature = "std")]

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, Interlace, MemoryDomain, OutputSink, PipelineClock,
    PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::imagefreeze::ImageFreeze;
use g2g_plugins::registry::default_registry;

const WIDTH: u32 = 16;
const HEIGHT: u32 = 8;
const FRAMERATE: u32 = 30;
const PERIOD_NS: u64 = 1_000_000_000 / FRAMERATE as u64;
const FRAMES: u64 = 10;

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
    fn timings(&self) -> Vec<FrameTiming> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f.timing),
                _ => None,
            })
            .collect()
    }

    fn caps_changes(&self) -> Vec<Caps> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }
}

/// A sink that accepts `limit` data frames and then fails, standing in for the
/// pipeline shutting an unlimited run down.
struct FailAfter {
    limit: usize,
    frames: usize,
}

impl OutputSink for FailAfter {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        if matches!(packet, PipelinePacket::DataFrame(_)) {
            if self.frames == self.limit {
                return core::task::Poll::Ready(Err(G2gError::Shutdown));
            }
            self.frames += 1;
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn caps(framerate: Rate) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate,
        interlace: Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn still() -> PipelinePacket {
    let bytes = vec![0x7f_u8; (WIDTH * HEIGHT * 4) as usize];
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

#[tokio::test]
async fn bounded_run_lands_on_the_exact_frame_grid() {
    let mut element = ImageFreeze::new()
        .with_framerate(FRAMERATE, 1)
        .with_num_buffers(FRAMES);
    element
        .configure_pipeline(&caps(Rate::Fixed(10 << 16)))
        .expect("raw video configures");
    let mut out = Collect::default();
    element.process(still(), &mut out).await.unwrap();

    let timings = out.timings();
    assert_eq!(timings.len(), FRAMES as usize);
    for (n, timing) in timings.iter().enumerate() {
        assert_eq!(timing.pts_ns, n as u64 * PERIOD_NS, "frame {n}");
        assert_eq!(timing.dts_ns, timing.pts_ns);
        assert_eq!(timing.duration_ns, PERIOD_NS);
    }
    // the fixed output framerate is announced once, before the first frame.
    assert_eq!(
        out.caps_changes(),
        [caps(Rate::Fixed(FRAMERATE << 16))],
        "one CapsChanged carrying the property's framerate"
    );

    // a second input frame has nowhere to go: the still is already chosen.
    element.process(still(), &mut out).await.unwrap();
    assert_eq!(out.timings().len(), FRAMES as usize);
}

#[tokio::test]
async fn unlimited_run_stops_when_the_sink_fails() {
    const ACCEPTED: usize = 7;
    let mut element = ImageFreeze::new().with_framerate(FRAMERATE, 1);
    element
        .configure_pipeline(&caps(Rate::Any))
        .expect("raw video configures");
    let mut out = FailAfter {
        limit: ACCEPTED,
        frames: 0,
    };
    assert_eq!(
        element.process(still(), &mut out).await.unwrap_err(),
        G2gError::Shutdown,
        "the failing push ends the unlimited loop"
    );
    assert_eq!(out.frames, ACCEPTED);
}

#[tokio::test]
async fn flush_lets_a_new_still_take_over() {
    let mut element = ImageFreeze::new()
        .with_framerate(FRAMERATE, 1)
        .with_num_buffers(FRAMES);
    element
        .configure_pipeline(&caps(Rate::Any))
        .expect("raw video configures");
    let mut out = Collect::default();
    element.process(still(), &mut out).await.unwrap();
    element
        .process(PipelinePacket::Flush, &mut out)
        .await
        .unwrap();
    element.process(still(), &mut out).await.unwrap();
    let timings = out.timings();
    assert_eq!(timings.len(), 2 * FRAMES as usize);
    assert_eq!(
        timings[FRAMES as usize].pts_ns, 0,
        "the counter restarts after a flush"
    );
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn imagefreeze_runs_in_a_text_pipeline() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=1 ! imagefreeze num-buffers=10 framerate=30/1 ! fakesink",
    )
    .expect("pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("imagefreeze pipeline runs");
    assert_eq!(
        stats.frames_consumed, FRAMES,
        "one source frame becomes num-buffers stream frames"
    );
}

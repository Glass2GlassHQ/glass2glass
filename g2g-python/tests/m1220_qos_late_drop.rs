#![cfg(feature = "analytics")]

use g2g_core::memory::SystemSlice;
use g2g_core::metrics::monotonic_ns;
use g2g_core::{
    AsyncElement, BlobMeta, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PropValue, PushOutcome, QosMessage, Rate, RawVideoFormat, Segment,
};
use g2g_python::PyTransform;

const WIDTH: u32 = 2;
const HEIGHT: u32 = 1;
const FRAME_PERIOD_NS: u64 = 33_000_000;
const JITTER_NS: i64 = 70_000_000;
const EARLY_NS: i64 = 20_000_000;
const DOWNSTREAM_DELAY: std::time::Duration = std::time::Duration::from_millis(50);
const CALLS_HEADER: &str = "calls";

#[derive(Default)]
struct LateSink {
    packets: Vec<PipelinePacket>,
    report: Option<QosMessage>,
}

impl OutputSink for LateSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.packets.push(packet);
        let outcome = match self.report.take() {
            Some(mut report) => {
                report.taken_ns = monotonic_ns();
                PushOutcome::Qos(report)
            }
            None => PushOutcome::Accepted,
        };
        core::task::Poll::Ready(Ok(outcome))
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn frame(pts_ns: u64, sequence: u64) -> Frame {
    let bytes = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    }
}

fn calls(packet: &PipelinePacket) -> u8 {
    let PipelinePacket::DataFrame(frame) = packet else {
        panic!("expected a DataFrame downstream");
    };
    let blobs = frame.meta.get::<BlobMeta>().expect("the fixture's blob");
    blobs.get(CALLS_HEADER).expect("the call count").payload[0]
}

fn hosted(module: &str, class: &str) -> PyTransform {
    std::env::set_var(
        "PYTHONPATH",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
    );
    let mut element = PyTransform::new(module, class);
    element.configure_pipeline(&caps()).unwrap();
    element
}

fn counting_element() -> PyTransform {
    hosted("m1178_element", "CountingTransform")
}

fn run_three_frames(element: &mut PyTransform, sink: &mut LateSink) {
    sink.report = Some(QosMessage {
        jitter_ns: JITTER_NS,
        running_time_ns: 0,
        taken_ns: 0,
    });
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for sequence in 0..3u64 {
        runtime
            .block_on(element.process(
                PipelinePacket::DataFrame(frame(sequence * FRAME_PERIOD_NS, sequence)),
                sink,
            ))
            .unwrap();
    }
}

#[test]
fn a_frame_inside_the_late_window_is_dropped_without_a_python_call() {
    let mut element = counting_element();
    assert!(
        element.handles_qos(),
        "the runner hands the report to the host"
    );
    let mut sink = LateSink::default();
    run_three_frames(&mut element, &mut sink);

    assert_eq!(
        sink.packets.len(),
        2,
        "the late frame never went downstream"
    );
    assert_eq!(element.late_dropped_count(), 1);
    assert_eq!(element.emitted_count(), 2);
    assert_eq!(calls(&sink.packets[0]), 1);
    assert_eq!(
        calls(&sink.packets[1]),
        2,
        "the hosted class saw two calls, none for the dropped frame"
    );
}

#[test]
fn qos_false_runs_every_frame_and_leaves_the_report_to_the_runner() {
    let mut element = counting_element();
    element
        .set_property("qos", PropValue::Bool(false))
        .expect("qos is a declared property");
    assert!(!element.handles_qos());
    let mut sink = LateSink::default();
    run_three_frames(&mut element, &mut sink);

    assert_eq!(sink.packets.len(), 3);
    assert_eq!(element.late_dropped_count(), 0);
    assert_eq!(calls(&sink.packets[2]), 3);
}

#[test]
fn lateness_is_measured_in_running_time_and_a_flush_clears_it() {
    let mut element = counting_element();
    let mut sink = LateSink::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    // A segment starting at 1 s: PTS 1 s is running time 0.
    let segment_start_ns = 1_000_000_000;
    let mut segment = Segment::new();
    segment.start = segment_start_ns;
    runtime
        .block_on(element.process(PipelinePacket::Segment(segment), &mut sink))
        .unwrap();
    sink.report = Some(QosMessage {
        jitter_ns: JITTER_NS,
        running_time_ns: 0,
        taken_ns: 0,
    });
    runtime
        .block_on(element.process(
            PipelinePacket::DataFrame(frame(segment_start_ns, 0)),
            &mut sink,
        ))
        .unwrap();
    runtime
        .block_on(element.process(
            PipelinePacket::DataFrame(frame(segment_start_ns + FRAME_PERIOD_NS, 1)),
            &mut sink,
        ))
        .unwrap();
    assert_eq!(
        element.late_dropped_count(),
        1,
        "running time 33 ms is late"
    );

    runtime
        .block_on(element.process(PipelinePacket::Flush, &mut sink))
        .unwrap();
    runtime
        .block_on(element.process(
            PipelinePacket::DataFrame(frame(segment_start_ns + FRAME_PERIOD_NS, 2)),
            &mut sink,
        ))
        .unwrap();
    assert_eq!(
        element.late_dropped_count(),
        1,
        "the same frame runs after a flush"
    );
    let frames = sink
        .packets
        .iter()
        .filter(|packet| matches!(packet, PipelinePacket::DataFrame(_)))
        .count();
    assert_eq!(frames, 2);
}

#[test]
fn a_frame_that_would_be_late_after_the_run_time_is_dropped() {
    let mut element = hosted("m1220_element", "SlowTransform");
    let mut sink = LateSink::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    sink.report = Some(QosMessage {
        jitter_ns: JITTER_NS,
        running_time_ns: 0,
        taken_ns: 0,
    });
    for sequence in 0..4u64 {
        runtime
            .block_on(element.process(
                PipelinePacket::DataFrame(frame(sequence * FRAME_PERIOD_NS, sequence)),
                &mut sink,
            ))
            .unwrap();
    }
    assert_eq!(
        element.late_dropped_count(),
        2,
        "33 ms and 66 ms are skipped"
    );
    assert_eq!(sink.packets.len(), 2);
    assert_eq!(calls(&sink.packets[1]), 2);
}

#[test]
fn an_early_report_ends_the_drops_a_late_one_started() {
    let mut element = counting_element();
    let mut sink = LateSink::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let process = |element: &mut PyTransform, sink: &mut LateSink, pts_ns, sequence| {
        runtime
            .block_on(element.process(PipelinePacket::DataFrame(frame(pts_ns, sequence)), sink))
            .unwrap();
    };
    sink.report = Some(QosMessage {
        jitter_ns: JITTER_NS,
        running_time_ns: 0,
        taken_ns: 0,
    });
    process(&mut element, &mut sink, 0, 0);
    process(&mut element, &mut sink, FRAME_PERIOD_NS, 1);
    assert_eq!(element.late_dropped_count(), 1);

    // An untimed frame is never late, so it carries the sink's next report.
    sink.report = Some(QosMessage {
        jitter_ns: -EARLY_NS,
        running_time_ns: 0,
        taken_ns: 0,
    });
    process(&mut element, &mut sink, FrameTiming::PTS_NONE, 2);
    process(&mut element, &mut sink, 2 * FRAME_PERIOD_NS, 3);
    assert_eq!(
        element.late_dropped_count(),
        1,
        "66 ms runs once the sink reports it is early"
    );
    assert_eq!(sink.packets.len(), 3);
}

// The sink reports its clock at frame 2's running time, when frame 0 reached it
// `delay` after this element pushed it.
fn frame_two_is_dropped_after(delay: std::time::Duration) -> bool {
    let mut element = counting_element();
    let mut sink = LateSink::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime
        .block_on(element.process(PipelinePacket::DataFrame(frame(0, 0)), &mut sink))
        .unwrap();
    std::thread::sleep(delay);
    sink.report = Some(QosMessage {
        jitter_ns: 2 * FRAME_PERIOD_NS as i64,
        running_time_ns: 0,
        taken_ns: 0,
    });
    runtime
        .block_on(element.process(
            PipelinePacket::DataFrame(frame(FRAME_PERIOD_NS, 1)),
            &mut sink,
        ))
        .unwrap();
    runtime
        .block_on(element.process(
            PipelinePacket::DataFrame(frame(2 * FRAME_PERIOD_NS, 2)),
            &mut sink,
        ))
        .unwrap();
    element.late_dropped_count() == 1
}

#[test]
fn the_learned_downstream_delay_counts_toward_lateness() {
    assert!(
        !frame_two_is_dropped_after(std::time::Duration::ZERO),
        "a frame due at the sink's clock is within the sink's bound"
    );
    assert!(
        frame_two_is_dropped_after(DOWNSTREAM_DELAY),
        "it reaches the sink a delay later, past the bound"
    );
}

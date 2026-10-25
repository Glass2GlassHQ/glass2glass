//! M1893: at `log` level the runner logs every packet each element receives
//! (and each one a source pushes) under that element's category and instance
//! name, the `GST_DEBUG=GST_SCHEDULING:6,GST_EVENT:6` analog. Paths that drop
//! data say so at `debug`.
#![cfg(feature = "std")]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use g2g_core::log::{self, short_type_name, LogLevel, OwnedLogRecord, RingSink, PACING_CATEGORY};
use g2g_core::runtime::{block_on, parse_launch, run_graph};
use g2g_core::{ClockSync, Pace, PipelineClock, PresentationPacer};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::registry::default_registry;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videotestsrc::VideoTestSrc;

const BUFFERS: usize = 2;
const RING_CAPACITY: usize = 4096;
const FRAME_PERIOD_NS: u64 = 33_000_000;

// The log sink and thresholds are process-global, so the tests here take turns.
static LOGGING: Mutex<()> = Mutex::new(());

fn capture(spec: &str) -> (RingSink, MutexGuard<'static, ()>) {
    let guard = LOGGING
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let ring = RingSink::new(RING_CAPACITY);
    log::set_sink(Box::new(ring.clone()));
    log::configure(spec);
    (ring, guard)
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

struct ManualClock(Arc<AtomicU64>);
impl PipelineClock for ManualClock {
    fn now_ns(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

fn lines<'r>(
    records: &'r [OwnedLogRecord],
    category: &str,
    level: LogLevel,
    prefix: &str,
) -> Vec<&'r OwnedLogRecord> {
    records
        .iter()
        .filter(|r| r.category == category && r.level == level && r.message.starts_with(prefix))
        .collect()
}

#[test]
fn each_element_logs_its_buffers_and_eos_under_its_instance_name() {
    let (ring, _guard) = capture("*:log");
    let graph = parse_launch(
        &default_registry(),
        &format!("videotestsrc num-buffers={BUFFERS} ! videoconvert ! fakesink"),
    )
    .expect("parses");
    block_on(run_graph(graph, &ZeroClock, 4)).expect("runs");
    log::reset();
    let records = ring.drain();

    let source = short_type_name::<VideoTestSrc>();
    let pushed = lines(&records, source, LogLevel::Log, "pushing buffer pts ");
    assert_eq!(pushed.len(), BUFFERS, "the source's pushes: {records:#?}");
    assert_eq!(
        lines(&records, source, LogLevel::Debug, "pushing event eos").len(),
        1
    );
    assert!(pushed
        .iter()
        .all(|r| r.instance.as_deref() == Some(format!("{source}0").as_str())));

    for category in [
        short_type_name::<VideoConvert>(),
        short_type_name::<FakeSink>(),
    ] {
        let instance = format!("{category}0");
        let buffers = lines(&records, category, LogLevel::Log, "have buffer pts ");
        assert_eq!(buffers.len(), BUFFERS, "{category} buffers: {records:#?}");
        assert!(buffers
            .iter()
            .all(|r| r.instance.as_deref() == Some(instance.as_str())));
        let eos = lines(&records, category, LogLevel::Debug, "have event eos");
        assert_eq!(eos.len(), 1, "{category} eos: {records:#?}");
        assert_eq!(eos[0].instance.as_deref(), Some(instance.as_str()));
    }
}

#[test]
fn a_run_of_late_drops_logs_once_when_it_starts_and_once_when_it_ends() {
    let (ring, _guard) = capture("*:debug");
    let now = Arc::new(AtomicU64::new(0));
    let mut pacer = PresentationPacer::new();
    pacer.set_clock_sync(ClockSync::new(Arc::new(ManualClock(now.clone())), 0));
    pacer.set_max_lateness_ns(0);

    // The first frame anchors the clock on itself, so it is due at once.
    assert_eq!(pacer.judge(0, 0), Pace::Now);
    // Three frame periods later the next two frames are already past due.
    now.store(3 * FRAME_PERIOD_NS, Ordering::Relaxed);
    let late = [FRAME_PERIOD_NS, 2 * FRAME_PERIOD_NS];
    for pts in late {
        assert_eq!(pacer.judge(pts, 1), Pace::Drop);
    }
    assert!(matches!(pacer.judge(4 * FRAME_PERIOD_NS, 1), Pace::Wait(_)));
    log::reset();
    let records = ring.drain();

    let started = lines(
        &records,
        PACING_CATEGORY,
        LogLevel::Debug,
        "dropping late frames",
    );
    assert_eq!(started.len(), 1, "one line for the run: {records:#?}");
    let ended = lines(
        &records,
        PACING_CATEGORY,
        LogLevel::Debug,
        "presenting again",
    );
    assert_eq!(ended.len(), 1, "{records:#?}");
    assert!(
        ended[0]
            .message
            .ends_with(&format!("after dropping {} late frames", late.len())),
        "{}",
        ended[0].message
    );
}

#[cfg(feature = "mjpeg")]
#[path = "common/collect_sink.rs"]
mod collect_sink;

#[cfg(feature = "mjpeg")]
#[test]
fn a_decoder_logs_each_frame_it_clips_to_the_segment() {
    use g2g_core::memory::SystemSlice;
    use g2g_core::{
        AsyncElement, BitstreamAlignment, Caps, Colorimetry, Dim, Frame, FrameTiming, MemoryDomain,
        PipelinePacket, Rate, Segment, VideoCodec,
    };
    use g2g_plugins::mjpegdec::MjpegDec;

    let (ring, _guard) = capture("*:log");
    let mut decoder = MjpegDec::new();
    decoder
        .configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::Mjpeg,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: Colorimetry::UNKNOWN,
            alignment: BitstreamAlignment::Unspecified,
        })
        .expect("configures");
    let segment = Segment {
        start: FRAME_PERIOD_NS,
        ..Segment::new()
    };
    // the clip comes before the decode, so the frame needs no JPEG in it
    let before_the_segment = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8]))),
        FrameTiming {
            pts_ns: 0,
            ..FrameTiming::default()
        },
        0,
    );
    let mut out = collect_sink::CollectSink::default();
    block_on(async {
        decoder
            .process(PipelinePacket::Segment(segment), &mut out)
            .await
            .expect("takes the segment");
        decoder
            .process(PipelinePacket::DataFrame(before_the_segment), &mut out)
            .await
            .expect("clips the frame");
    });
    log::reset();
    let records = ring.drain();

    let clipped = lines(
        &records,
        short_type_name::<MjpegDec>(),
        LogLevel::Log,
        "dropping frame pts ",
    );
    assert_eq!(clipped.len(), 1, "{records:#?}");
    assert!(
        !out.packets
            .iter()
            .any(|packet| matches!(packet, PipelinePacket::DataFrame(_))),
        "the clipped frame is not pushed"
    );
}

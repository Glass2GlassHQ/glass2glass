//! M843: the QoS reports a synchronizing sink posts, through the real
//! `SyncSink`. Two cadences: the running stats on a report interval (pipeline
//! clock, not wall clock), and the per-drop report a late frame triggers. Both
//! come from the shared `QosTracker`, which is what the display sinks use too,
//! so this covers their reporting without a display attached.
#![cfg(feature = "std")]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncClock, AsyncElement, Bus, BusMessage, ByteStreamEncoding, Caps, FrameTiming, G2gError,
    OutputSink, PipelineClock, PipelinePacket, PushOutcome,
};
use g2g_plugins::syncsink::SyncSink;

/// A pipeline clock the test drives by hand; sleeping is instant, so a frame's
/// deadline never blocks the test.
struct ManualClock(Arc<AtomicU64>);

impl PipelineClock for ManualClock {
    fn now_ns(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

impl AsyncClock for ManualClock {
    type SleepFuture<'a>
        = Ready<()>
    where
        Self: 'a;
    fn sleep_until_ns(&self, _deadline_ns: u64) -> Ready<()> {
        ready(())
    }
}

struct NullOut;
impl OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn frame(pts_ns: u64, sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8]))),
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        sequence,
    ))
}

fn configured(sink: &mut SyncSink<ManualClock>) {
    sink.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    })
    .expect("wildcard sink accepts any caps");
}

#[tokio::test]
async fn a_report_interval_posts_running_stats_while_frames_flow() {
    let (bus, handle) = Bus::new(16);
    let clock = Arc::new(AtomicU64::new(0));
    let mut sink = SyncSink::new(ManualClock(clock.clone()))
        .with_bus(handle)
        .with_qos_interval_ns(20_000_000);
    configured(&mut sink);
    let mut out = NullOut;

    // Three on-time frames 20 ms apart: the first arms the interval, the next
    // two each cross it. Nothing is late, so every report is a periodic one.
    for i in 0..3u64 {
        clock.store(i * 20_000_000, Ordering::Relaxed);
        sink.process(frame(i * 20_000_000, i), &mut out)
            .await
            .unwrap();
    }
    assert_eq!(sink.received(), 3, "every frame was presented");
    assert_eq!(sink.dropped(), 0, "nothing was late");

    let mut reports = 0;
    while let Some(m) = bus.try_recv() {
        match m {
            BusMessage::Qos {
                dropped, processed, ..
            } => {
                assert_eq!(dropped, 0);
                assert!(processed > 0, "running stats carry the presented count");
                reports += 1;
            }
            other => panic!("unexpected message {other:?}"),
        }
    }
    assert_eq!(reports, 2, "one report per elapsed interval");
}

#[tokio::test]
async fn no_interval_reports_only_drops() {
    let (bus, handle) = Bus::new(16);
    let clock = Arc::new(AtomicU64::new(0));
    let mut sink = SyncSink::new(ManualClock(clock.clone()))
        .with_bus(handle)
        .with_max_lateness_ns(0);
    configured(&mut sink);
    let mut out = NullOut;

    // On time: presented, and with no interval set nothing is reported.
    sink.process(frame(0, 0), &mut out).await.unwrap();
    assert_eq!(bus.try_recv(), None, "an on-time frame reports nothing");

    // 5 ms past a zero bound: dropped, reported, and signalled upstream.
    clock.store(15_000_000, Ordering::Relaxed);
    sink.process(frame(10_000_000, 1), &mut out).await.unwrap();
    assert_eq!(sink.dropped(), 1);
    assert_eq!(
        bus.try_recv(),
        Some(BusMessage::Qos {
            running_time_ns: 10_000_000,
            jitter_ns: 5_000_000,
            processed: 1,
            dropped: 1,
        })
    );
    assert_eq!(
        AsyncElement::take_qos(&mut sink).map(|q| q.jitter_ns),
        Some(5_000_000),
        "the same lateness travels upstream for load shedding"
    );
}

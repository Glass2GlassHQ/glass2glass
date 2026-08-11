//! Dynamic duplex-session runner (`run_duplex_session_dynamic`, M1014): the pad
//! COUNT of a live duplex session grows in both directions, with nothing reserved
//! up front.
//!
//! Two deterministic fake sessions, both driven with `block_on` (no executor
//! needed): one grows its **send** side, taking a source attached mid-run through
//! `DynamicDuplexHandle::add_send_track` and routing that pad's PLI back to it
//! through `DuplexInbound::reverse_channel`; the other grows its **recv** side,
//! calling `MultiOutputSink::add_port` mid-run and having the runner's sink
//! factory build the element that drains the new port.

#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{
    block_on, run_duplex_session_dynamic, run_duplex_session_dynamic_observed, DynSourceLoop,
    Join2, NodeRole, Observer, SourceLoop,
};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, ConfigureOutcome, Dim, DuplexInbound,
    G2gError, MultiDuplexSession, MultiOutputSink, OutputSink, PipelineClock, PipelinePacket,
    PushOutcome, Rate, RawVideoFormat,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn video_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::I420,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

/// A second, clearly different shape, so the caps a grown pad carries can be
/// told apart from the declared pad's.
fn audio_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 48_000,
    }
}

fn frame(seq: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(g2g_core::frame::Frame::new(
        g2g_core::MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
            std::vec![0u8; 4].into_boxed_slice(),
        )),
        g2g_core::FrameTiming {
            pts_ns: seq,
            ..Default::default()
        },
        seq,
    ))
}

/// Send source: announces `caps`, pushes `n` frames then EOS, and records whether
/// any push came back as a keyframe request (the reverse-channel round trip).
struct CountedSource {
    caps: Caps,
    n: u64,
    keyframe_asked: Arc<AtomicBool>,
}

impl SourceLoop for CountedSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(self.caps.clone()))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(self.caps.clone()))
                .await?;
            for seq in 0..self.n {
                if let PushOutcome::Reconfigure(_) = out.push(frame(seq)).await? {
                    self.keyframe_asked.store(true, Ordering::SeqCst);
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.n)
        })
    }
}

/// What a recv sink saw, in order, so a pad's caps event can be placed relative
/// to its first frame.
#[derive(Debug, Clone, PartialEq)]
enum Seen {
    Configured(Caps),
    Caps(Caps),
    Frame,
    Eos,
}

/// Recv sink recording its configure calls and packet order into a shared log.
struct RecordSink {
    log: Arc<Mutex<Vec<Seen>>>,
}

impl AsyncElement for RecordSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.log.lock().unwrap().push(Seen::Configured(c.clone()));
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let seen = match packet {
                PipelinePacket::CapsChanged(c) => Some(Seen::Caps(c)),
                PipelinePacket::DataFrame(_) => Some(Seen::Frame),
                PipelinePacket::Eos => Some(Seen::Eos),
                _ => None,
            };
            if let Some(seen) = seen {
                self.log.lock().unwrap().push(seen);
            }
            Ok(())
        })
    }
}

/// What a fake session observed on its send side.
#[derive(Debug, Clone, PartialEq)]
enum Inbound {
    /// A pad announced itself: its index and caps, plus whether the runner had a
    /// reverse channel registered for it by then.
    Pad(usize, Caps, bool),
    Frame(usize),
}

/// Fake duplex session that takes send pads it never declared: any index it has
/// not seen before is learned from that index's `CapsChanged`, and a keyframe is
/// requested on it through the reverse channel the runner registered. Frames are
/// echoed to output 0 so the recv side stays exercised.
struct GrowSendDuplex {
    inputs: usize,
    outputs: usize,
    log: Arc<Mutex<Vec<Inbound>>>,
}

impl MultiDuplexSession for GrowSendDuplex {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.inputs
    }
    fn output_count(&self) -> usize {
        self.outputs
    }
    fn intercept_caps(&self, _i: usize, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_input(&self, _i: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_input(&mut self, _i: usize, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_caps(&self, _o: usize) -> Result<Caps, G2gError> {
        Ok(video_caps())
    }
    fn run<'a>(
        &'a mut self,
        inbound: &'a mut dyn DuplexInbound,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::RunFuture<'a> {
        let declared = self.inputs;
        let outputs = self.outputs;
        let log = self.log.clone();
        Box::pin(async move {
            let mut received = 0u64;
            // Pads learned mid-run. A source that announces its own caps sends a
            // second `CapsChanged` on the index, so this is a re-announce, exactly
            // as it is for a declared pad.
            let mut learned: Vec<usize> = Vec::new();
            loop {
                // Bound the recv borrow to this statement: the pad lookup below
                // needs `inbound` again, which a `while let` scrutinee would still
                // be holding.
                let next = inbound.recv().await;
                let Some((idx, packet)) = next else { break };
                match packet {
                    PipelinePacket::CapsChanged(caps)
                        if idx >= declared && !learned.contains(&idx) =>
                    {
                        learned.push(idx);
                        let reverse = inbound.reverse_channel(idx);
                        log.lock()
                            .unwrap()
                            .push(Inbound::Pad(idx, caps, reverse.is_some()));
                        if let Some(reverse) = reverse {
                            reverse.request_keyframe();
                        }
                    }
                    PipelinePacket::DataFrame(f) => {
                        log.lock().unwrap().push(Inbound::Frame(idx));
                        out.push_to(idx % outputs, PipelinePacket::DataFrame(f))
                            .await?;
                        received += 1;
                    }
                    _ => {}
                }
            }
            for o in 0..outputs {
                out.push_to(o, PipelinePacket::Eos).await?;
            }
            Ok(received)
        })
    }
}

#[test]
fn a_send_track_added_mid_run_reaches_the_session_with_a_reverse_route() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let keyframe_asked = Arc::new(AtomicBool::new(false));
    let mut declared_source = CountedSource {
        caps: video_caps(),
        n: 5,
        keyframe_asked: Arc::new(AtomicBool::new(false)),
    };
    let mut session = GrowSendDuplex {
        inputs: 1,
        outputs: 1,
        log: log.clone(),
    };
    let sink_log = Arc::new(Mutex::new(Vec::new()));
    let mut sink = RecordSink {
        log: sink_log.clone(),
    };
    let clock = ZeroClock;

    let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut declared_source];
    let sinks: Vec<&mut dyn DynAsyncElement> = std::vec![&mut sink];
    let (handle, run) = run_duplex_session_dynamic(
        sources,
        &mut session,
        sinks,
        &clock,
        4,
        // The recv side never grows here.
        |_port, _caps| None,
    );

    let late = Box::new(CountedSource {
        caps: audio_caps(),
        n: 8,
        keyframe_asked: keyframe_asked.clone(),
    });
    let control = async {
        let input = handle
            .add_send_track(late)
            .expect("the running session takes a new send track");
        // Dropping the handle is what lets the send side end.
        drop(handle);
        input
    };

    let (stats, input) = block_on(Join2::new(run, control));
    let stats = stats.expect("dynamic duplex run completes");
    let log = log.lock().unwrap().clone();

    assert_eq!(
        input, 1,
        "the grown track takes the index past the declared"
    );
    assert_eq!(
        log.iter()
            .filter(|e| matches!(e, Inbound::Pad(1, _, _)))
            .count(),
        1,
        "the session learns of pad 1 exactly once, log: {log:?}"
    );
    assert!(
        log.contains(&Inbound::Pad(1, audio_caps(), true)),
        "pad 1 announces the late source's own caps and resolves a reverse channel, log: {log:?}"
    );
    assert_eq!(
        log.iter().filter(|e| **e == Inbound::Frame(1)).count(),
        8,
        "every frame of the grown track reaches the session under index 1"
    );
    assert_eq!(
        log.iter().filter(|e| **e == Inbound::Frame(0)).count(),
        5,
        "the declared track keeps flowing"
    );
    // The pad's caps arrive before any of its frames.
    let pad_at = log
        .iter()
        .position(|e| matches!(e, Inbound::Pad(1, _, _)))
        .expect("pad 1 announced");
    let first_frame_at = log
        .iter()
        .position(|e| *e == Inbound::Frame(1))
        .expect("pad 1 has frames");
    assert!(pad_at < first_frame_at, "caps precede the first frame");
    assert!(
        keyframe_asked.load(Ordering::SeqCst),
        "the PLI the session sent on the grown pad reaches its source"
    );
    assert_eq!(stats.frames_emitted, 13, "5 declared + 8 grown");
    assert_eq!(stats.frames_consumed, 13, "all echoed to the recv sink");
}

/// Fake duplex session that grows its RECV side: after the second frame it asks
/// for a port it never declared and publishes a second stream on it.
struct GrowRecvDuplex {
    outputs: usize,
    /// The caps the grown port is created with, echoed to the test.
    grown_caps: Caps,
    grown_port: Arc<AtomicUsize>,
}

/// Sentinel for "the session never got a grown port".
const NO_PORT: usize = usize::MAX;

impl MultiDuplexSession for GrowRecvDuplex {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        1
    }
    fn output_count(&self) -> usize {
        self.outputs
    }
    fn intercept_caps(&self, _i: usize, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_input(&self, _i: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_input(&mut self, _i: usize, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_caps(&self, _o: usize) -> Result<Caps, G2gError> {
        Ok(video_caps())
    }
    fn run<'a>(
        &'a mut self,
        inbound: &'a mut dyn DuplexInbound,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::RunFuture<'a> {
        let outputs = self.outputs;
        let grown_caps = self.grown_caps.clone();
        let grown_port = self.grown_port.clone();
        Box::pin(async move {
            let mut received = 0u64;
            let mut grown: Option<usize> = None;
            loop {
                let next = inbound.recv().await;
                let Some((_, PipelinePacket::DataFrame(f))) = next else {
                    if next.is_none() {
                        break;
                    }
                    continue;
                };
                received += 1;
                out.push_to(0, PipelinePacket::DataFrame(f)).await?;
                // Mid-run: a track the declared pads cannot carry.
                if received == 2 && grown.is_none() {
                    if let Some(port) = out.add_port(&grown_caps) {
                        grown_port.store(port, core::sync::atomic::Ordering::SeqCst);
                        grown = Some(port);
                        out.push_to(port, PipelinePacket::CapsChanged(grown_caps.clone()))
                            .await?;
                    }
                }
                if let Some(port) = grown {
                    out.push_to(port, frame(received)).await?;
                }
            }
            for o in 0..outputs + usize::from(grown.is_some()) {
                out.push_to(o, PipelinePacket::Eos).await?;
            }
            Ok(received)
        })
    }
}

#[test]
fn a_recv_port_the_session_adds_mid_run_gets_a_sink_from_the_factory() {
    let mut source = CountedSource {
        caps: video_caps(),
        n: 5,
        keyframe_asked: Arc::new(AtomicBool::new(false)),
    };
    let grown_port = Arc::new(AtomicUsize::new(NO_PORT));
    let mut session = GrowRecvDuplex {
        outputs: 1,
        grown_caps: audio_caps(),
        grown_port: grown_port.clone(),
    };
    let declared_log = Arc::new(Mutex::new(Vec::new()));
    let mut declared_sink = RecordSink {
        log: declared_log.clone(),
    };
    let clock = ZeroClock;

    // What the factory was asked for, and the log of the sink it handed back.
    let asked: Arc<Mutex<Vec<(usize, Caps)>>> = Arc::new(Mutex::new(Vec::new()));
    let grown_log = Arc::new(Mutex::new(Vec::new()));

    let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut source];
    let sinks: Vec<&mut dyn DynAsyncElement> = std::vec![&mut declared_sink];
    let (handle, run) = {
        let asked = asked.clone();
        let grown_log = grown_log.clone();
        run_duplex_session_dynamic(
            sources,
            &mut session,
            sinks,
            &clock,
            4,
            move |port, caps: &Caps| {
                asked.lock().unwrap().push((port, caps.clone()));
                let sink: Box<dyn DynAsyncElement> = Box::new(RecordSink {
                    log: grown_log.clone(),
                });
                Some(sink)
            },
        )
    };
    // No send track is added here; the handle only has to go away so the send
    // side can end.
    drop(handle);

    let stats = block_on(run).expect("dynamic duplex run completes");

    assert_eq!(
        grown_port.load(Ordering::SeqCst),
        1,
        "the session got the port past its declared one"
    );
    assert_eq!(
        *asked.lock().unwrap(),
        std::vec![(1usize, audio_caps())],
        "the factory is asked once, for the port and caps the session declared"
    );
    let grown = grown_log.lock().unwrap().clone();
    assert_eq!(
        grown.first(),
        Some(&Seen::Configured(audio_caps())),
        "the grown sink is configured with the port's caps before anything else, log: {grown:?}"
    );
    let first_frame = grown
        .iter()
        .position(|s| *s == Seen::Frame)
        .expect("the grown port carries frames");
    let caps_at = grown
        .iter()
        .position(|s| *s == Seen::Caps(audio_caps()))
        .expect("the grown port announces its caps");
    assert!(
        caps_at < first_frame,
        "caps reach the grown sink before its first frame, log: {grown:?}"
    );
    assert_eq!(
        grown.iter().filter(|s| **s == Seen::Frame).count(),
        4,
        "frames 2..5 flow on the grown port, log: {grown:?}"
    );
    assert_eq!(grown.last(), Some(&Seen::Eos), "the grown port is EOSed");
    assert_eq!(
        declared_log
            .lock()
            .unwrap()
            .iter()
            .filter(|s| **s == Seen::Frame)
            .count(),
        5,
        "the declared port keeps its own stream"
    );
    assert_eq!(stats.frames_consumed, 9, "5 declared + 4 grown");
    assert_eq!(stats.frames_dropped, 0, "every grown frame found a sink");
}

/// Yields to the executor once, so a control future can interleave with the run
/// under `block_on` the way a real caller on the same thread would.
struct YieldOnce(bool);

impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut core::task::Context<'_>) -> core::task::Poll<()> {
        if self.0 {
            core::task::Poll::Ready(())
        } else {
            self.0 = true;
            cx.waker().wake_by_ref();
            core::task::Poll::Pending
        }
    }
}

/// Session that grows a recv port on every third frame, so recv growth and
/// send-track adds land in the same control-arm pass.
struct GrowOftenDuplex;

impl MultiDuplexSession for GrowOftenDuplex {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        1
    }
    fn output_count(&self) -> usize {
        1
    }
    fn intercept_caps(&self, _i: usize, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_input(&self, _i: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_input(&mut self, _i: usize, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_caps(&self, _o: usize) -> Result<Caps, G2gError> {
        Ok(video_caps())
    }
    fn run<'a>(
        &'a mut self,
        inbound: &'a mut dyn DuplexInbound,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut received = 0u64;
            loop {
                let next = inbound.recv().await;
                let Some((_, packet)) = next else { break };
                let PipelinePacket::DataFrame(f) = packet else {
                    continue;
                };
                received += 1;
                out.push_to(0, PipelinePacket::DataFrame(f)).await?;
                if received % 3 == 0 {
                    if let Some(port) = out.add_port(&audio_caps()) {
                        out.push_to(port, PipelinePacket::CapsChanged(audio_caps()))
                            .await?;
                        out.push_to(port, frame(received)).await?;
                        out.push_to(port, PipelinePacket::Eos).await?;
                    }
                }
            }
            out.push_to(0, PipelinePacket::Eos).await?;
            Ok(received)
        })
    }
}

/// Regression for the M1014 verification finding: more pending adds than the arm
/// channel holds used to turn `SendError::Full` into `G2gError::Shutdown` and
/// kill the whole live session. At link capacity 2 (the recommended live value),
/// two queued send tracks plus one grown recv port crossed it.
#[test]
fn a_burst_of_adds_larger_than_link_capacity_is_backpressure_not_teardown() {
    let mut source = CountedSource {
        caps: video_caps(),
        n: 300,
        keyframe_asked: Arc::new(AtomicBool::new(false)),
    };
    let mut session = GrowOftenDuplex;
    let mut sink = RecordSink {
        log: Arc::new(Mutex::new(Vec::new())),
    };
    let clock = ZeroClock;

    let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut source];
    let sinks: Vec<&mut dyn DynAsyncElement> = std::vec![&mut sink];
    let (handle, run) =
        run_duplex_session_dynamic(sources, &mut session, sinks, &clock, 2, |_port, _caps| None);

    let accepted = Arc::new(AtomicUsize::new(0));
    let control = {
        let accepted = accepted.clone();
        async move {
            for _ in 0..60 {
                for _ in 0..4 {
                    let late = Box::new(CountedSource {
                        caps: audio_caps(),
                        n: 1,
                        keyframe_asked: Arc::new(AtomicBool::new(false)),
                    });
                    if handle.add_send_track(late).is_ok() {
                        accepted.fetch_add(1, Ordering::SeqCst);
                    }
                }
                YieldOnce(false).await;
            }
            drop(handle);
        }
    };

    let (result, _) = block_on(Join2::new(run, control));
    let stats = result.expect("a burst of adds must backpressure, not tear the session down");
    assert!(
        accepted.load(Ordering::SeqCst) > 2,
        "the burst outran the capacity-2 channels at least once"
    );
    assert!(stats.frames_emitted >= 300, "the declared track survived");
}

/// Regression for the M1014 verification finding: `add_send_track` used to
/// reserve its index with a `fetch_add` separate from the enqueue, so two
/// concurrent callers could reach the session out of order and the lower index
/// was orphaned for the life of the session. Reserving and enqueueing now share
/// a lock; the session must see first announces in strictly increasing order.
/// Cross-thread handles only exist under `multi-thread` (elements are `Send`
/// there), which is also the only configuration the race could occur in.
#[cfg(feature = "multi-thread")]
#[test]
fn concurrent_adds_reach_the_session_in_index_order() {
    for attempt in 0..30 {
        let mut source = CountedSource {
            caps: video_caps(),
            n: 400,
            keyframe_asked: Arc::new(AtomicBool::new(false)),
        };
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut session = GrowSendDuplex {
            inputs: 1,
            outputs: 1,
            log: log.clone(),
        };
        let mut sink = RecordSink {
            log: Arc::new(Mutex::new(Vec::new())),
        };
        let clock = ZeroClock;

        let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut source];
        let sinks: Vec<&mut dyn DynAsyncElement> = std::vec![&mut sink];
        let (handle, run) =
            run_duplex_session_dynamic(sources, &mut session, sinks, &clock, 16, |_p, _c| None);

        std::thread::scope(|scope| {
            for _ in 0..2 {
                let handle = handle.clone();
                scope.spawn(move || {
                    for _ in 0..15 {
                        let late = Box::new(CountedSource {
                            caps: audio_caps(),
                            n: 1,
                            keyframe_asked: Arc::new(AtomicBool::new(false)),
                        });
                        let _ = handle.add_send_track(late);
                        std::thread::yield_now();
                    }
                });
            }
            drop(handle);
            block_on(run).expect("dynamic duplex run completes");
        });

        let announced: Vec<usize> = log
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                Inbound::Pad(idx, _, _) => Some(*idx),
                Inbound::Frame(_) => None,
            })
            .collect();
        let mut sorted = announced.clone();
        sorted.sort_unstable();
        assert_eq!(
            announced, sorted,
            "attempt {attempt}: grown pads must announce in index order, got {announced:?}"
        );
    }
}

#[test]
fn the_observer_gains_a_node_for_every_pad_the_run_grows() {
    let observer = Observer::new();
    let mut source = CountedSource {
        caps: video_caps(),
        n: 5,
        keyframe_asked: Arc::new(AtomicBool::new(false)),
    };
    let mut session = GrowRecvDuplex {
        outputs: 1,
        grown_caps: audio_caps(),
        grown_port: Arc::new(AtomicUsize::new(NO_PORT)),
    };
    let mut declared_sink = RecordSink {
        log: Arc::new(Mutex::new(Vec::new())),
    };
    let clock = ZeroClock;

    let sources: Vec<&mut dyn DynSourceLoop> = std::vec![&mut source];
    let sinks: Vec<&mut dyn DynAsyncElement> = std::vec![&mut declared_sink];
    let (handle, run) = run_duplex_session_dynamic_observed(
        sources,
        &mut session,
        sinks,
        &clock,
        4,
        |_port, _caps| {
            let sink: Box<dyn DynAsyncElement> = Box::new(RecordSink {
                log: Arc::new(Mutex::new(Vec::new())),
            });
            Some(sink)
        },
        &observer,
    );
    let control = async {
        handle
            .add_send_track(Box::new(CountedSource {
                caps: audio_caps(),
                n: 4,
                keyframe_asked: Arc::new(AtomicBool::new(false)),
            }))
            .expect("the running session takes a new send track");
        drop(handle);
    };
    block_on(Join2::new(run, control))
        .0
        .expect("observed dynamic duplex run completes");

    // Declared: source, session, sink. Grown: one of each.
    let snapshot = observer.snapshot();
    let sources: Vec<usize> = snapshot
        .nodes
        .iter()
        .filter(|n| n.role == NodeRole::Source)
        .map(|n| n.id)
        .collect();
    let sinks: Vec<usize> = snapshot
        .nodes
        .iter()
        .filter(|n| n.role == NodeRole::Sink)
        .map(|n| n.id)
        .collect();
    assert_eq!(snapshot.nodes.len(), 5, "nodes: {:?}", snapshot.nodes);
    assert_eq!(sources.len(), 2, "the grown send track appears as a source");
    assert_eq!(sinks.len(), 2, "the grown recv port appears as a sink");
    let session_id = 1;
    for source in sources {
        assert!(
            snapshot
                .edges
                .iter()
                .any(|e| e.from == source && e.to == session_id),
            "source {source} links into the session, edges: {:?}",
            snapshot.edges
        );
    }
    for sink in sinks {
        assert!(
            snapshot
                .edges
                .iter()
                .any(|e| e.from == session_id && e.to == sink),
            "the session links into sink {sink}, edges: {:?}",
            snapshot.edges
        );
    }
}

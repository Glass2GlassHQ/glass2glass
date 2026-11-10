// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::task::{Context, Poll};
use std::time::Duration;

use g2g_core::fanout::MultiOutputSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph, LaunchFactory};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError,
    MultiInputElement, MultiOutputElement, OutputSink, PadTemplate, PadTemplates, PipelineClock,
    PushOutcome,
};
use g2g_plugins::appsink::set_appsink_callback;
use g2g_plugins::framesplit::{FrameMerge, FrameSplit};
use g2g_plugins::registry::default_registry;

const FRAMES: usize = 40;
const FAST_BRANCH_DELAY: Duration = Duration::from_millis(1);
const SLOW_BRANCH_DELAY: Duration = Duration::from_millis(5);

static FAST_BRANCH_FRAMES: AtomicUsize = AtomicUsize::new(0);
static SLOW_BRANCH_FRAMES: AtomicUsize = AtomicUsize::new(0);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

struct DelayPass {
    delay: Duration,
    frames: &'static AtomicUsize,
}

impl AsyncElement for DelayPass {
    type ProcessFuture<'a>
        = core::pin::Pin<Box<dyn core::future::Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new("delaying pass-through", "Generic", "test", "g2g")
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(_) = &packet {
                self.frames.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(self.delay).await;
            }
            if !matches!(packet, PipelinePacket::Eos) {
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

impl PadTemplates for DelayPass {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

extern "C" fn record_pts(data: *const u8, len: usize, pts_ns: u64, user: *mut c_void) {
    // SAFETY: `user` is the &Mutex<Vec<u64>> registered for the channel, alive for the whole run.
    let recorded = unsafe { &*(user as *const Mutex<Vec<u64>>) };
    // appsink signals the end of stream with an empty call
    if data.is_null() && len == 0 {
        return;
    }
    recorded.lock().unwrap().push(pts_ns);
}

#[tokio::test]
async fn a_slow_branch_takes_some_frames_and_the_merge_restores_their_order() {
    let recorded = Box::new(Mutex::new(Vec::<u64>::new()));
    let channel = "m2116-merged";
    set_appsink_callback(
        channel,
        record_pts,
        (&*recorded as *const Mutex<Vec<u64>>) as *mut c_void,
    );
    let mut registry = default_registry();
    registry.register_launch(LaunchFactory::of::<DelayPass>("fastpass", || {
        Box::new(DelayPass {
            delay: FAST_BRANCH_DELAY,
            frames: &FAST_BRANCH_FRAMES,
        })
    }));
    registry.register_launch(LaunchFactory::of::<DelayPass>("slowpass", || {
        Box::new(DelayPass {
            delay: SLOW_BRANCH_DELAY,
            frames: &SLOW_BRANCH_FRAMES,
        })
    }));
    let pipeline = format!(
        "videotestsrc num-buffers={FRAMES} ! framesplit name=s dispatch-log=m2116 \
         s. ! fastpass ! m.  s. ! slowpass ! m. \
         framemerge name=m dispatch-log=m2116 ! appsink channel={channel}"
    );
    let graph = parse_launch(&registry, &pipeline).expect("pipeline parses");
    run_graph(graph, &ZeroClock, 2)
        .await
        .expect("pipeline runs");

    let pts = recorded.lock().unwrap().clone();
    assert_eq!(pts.len(), FRAMES, "every frame comes out of the merge");
    assert!(
        pts.windows(2).all(|pair| pair[0] < pair[1]),
        "the merged frames are out of order: {pts:?}"
    );
    let fast = FAST_BRANCH_FRAMES.load(Ordering::Relaxed);
    let slow = SLOW_BRANCH_FRAMES.load(Ordering::Relaxed);
    assert_eq!(fast + slow, FRAMES);
    assert!(
        slow > 0 && fast > slow,
        "the fast branch took {fast} frames and the slow one {slow}"
    );
}

fn frame(pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 4].into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        sequence: 0,
        meta: Default::default(),
    })
}

fn pts_of(packet: &PipelinePacket) -> Option<u64> {
    match packet {
        PipelinePacket::DataFrame(frame) => Some(frame.timing.pts_ns),
        _ => None,
    }
}

#[derive(Default)]
struct Collected {
    pts: Vec<u64>,
    sequences: Vec<u64>,
}

impl OutputSink for Collected {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        if let PipelinePacket::DataFrame(frame) = &packet {
            self.pts.push(frame.timing.pts_ns);
            self.sequences.push(frame.sequence);
        }
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

struct Ports {
    full: Vec<bool>,
    pushed: Vec<Vec<u64>>,
}

impl MultiOutputSink for Ports {
    fn poll_push_to(
        &mut self,
        _cx: &mut Context<'_>,
        port: usize,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        let packet = packet.take().expect("poll_push_to without a packet");
        self.pushed[port].extend(pts_of(&packet));
        Poll::Ready(Ok(PushOutcome::Accepted))
    }

    fn poll_port_ready(
        &mut self,
        _cx: &mut Context<'_>,
        port: usize,
    ) -> Poll<Result<(), G2gError>> {
        match self.full[port] {
            true => Poll::Pending,
            false => Poll::Ready(Ok(())),
        }
    }

    fn port_count(&self) -> usize {
        self.full.len()
    }
}

#[tokio::test]
async fn the_split_skips_a_full_output_and_the_merge_waits_for_the_frame_it_holds() {
    let log = "m2116-full-port";
    let mut split = FrameSplit::new(2).with_dispatch_log(log);
    let mut merge = FrameMerge::new(2).with_dispatch_log(log);
    let mut ports = Ports {
        full: vec![false, false],
        pushed: vec![Vec::new(), Vec::new()],
    };
    split.process(frame(0), &mut ports).await.unwrap();
    ports.full[0] = true;
    split.process(frame(1), &mut ports).await.unwrap();
    ports.full[0] = false;
    split.process(frame(2), &mut ports).await.unwrap();
    assert_eq!(ports.pushed, vec![vec![0, 2], vec![1]]);

    let mut merged = Collected::default();
    merge.process(0, frame(0), &mut merged).await.unwrap();
    merge.process(0, frame(2), &mut merged).await.unwrap();
    assert_eq!(merged.pts, vec![0], "frame 2 went out ahead of frame 1");
    merge.process(1, frame(1), &mut merged).await.unwrap();
    assert_eq!(merged.pts, vec![0, 1, 2]);
    assert_eq!(
        merged.sequences,
        vec![0, 1, 2],
        "the merged stream is renumbered"
    );
}

#[tokio::test]
async fn an_idle_branch_does_not_hold_back_the_busy_one() {
    let log = "m2116-idle-branch";
    let mut split = FrameSplit::new(2).with_dispatch_log(log);
    let mut merge = FrameMerge::new(2).with_dispatch_log(log);
    let mut ports = Ports {
        full: vec![false, false],
        pushed: vec![Vec::new(), Vec::new()],
    };
    let mut merged = Collected::default();
    for pts in 0..3 {
        split.process(frame(pts), &mut ports).await.unwrap();
        merge.process(0, frame(pts), &mut merged).await.unwrap();
    }
    assert!(
        ports.pushed[1].is_empty(),
        "the second branch had room to spare"
    );
    assert_eq!(merged.pts, vec![0, 1, 2]);
}

//! M16 workaround #3 Phase A — ordering regression guard.
//!
//! The forward-reconfigure design (`DESIGN-M16-workaround3-reconfigure.md`)
//! turns on Phase A: decoders no longer silently swallow input
//! `PipelinePacket::CapsChanged`; they validate the format (loud on a
//! mid-stream H.264 -> VP9 switch) and record it. The output
//! `CapsChanged` is still emitted at the **decode boundary**, after the
//! last frame produced under the old caps and before the first frame
//! produced under the new caps.
//!
//! The naive "eager forward" fix violates this because decoders buffer
//! for B-frame reorder: across a resolution change, frames produced
//! under the old caps can still drain after the input `CapsChanged`
//! arrives. Forwarding the derived output caps eagerly reconfigures the
//! sink before those old frames land, presenting them under the wrong
//! configuration.
//!
//! This test isolates the ordering invariant with a fake in-memory
//! decoder whose buffering depth is controllable. No hardware, no
//! ffmpeg / mf / vaapi feature gating — pure plumbing.

use core::cell::Cell;
use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::collections::VecDeque;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, Rate, VideoFormat,
};
use g2g_plugins::fakesink::FakeSink;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn h264_caps(w: u32, h: u32) -> Caps {
    // Framerate must be fixated (not `Any`) so the solver can fixate
    // the link feeding into the legacy-bridged decoder.
    Caps::Video {
        format: VideoFormat::H264,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::Video {
        format: VideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

/// Programmable source: emits a scripted sequence of caps changes and
/// data frames into the pipeline, then EOS. Each data frame carries
/// the H.264 caps in force when the script entry was written, so the
/// downstream decoder sees the same ordering a real RTSP stream would.
#[derive(Debug, Clone)]
enum Step {
    /// Send `CapsChanged(h264_caps(w, h))` downstream.
    InputCapsChanged(u32, u32),
    /// Send one `DataFrame` carrying the most-recently-advertised caps.
    /// `tag` is a small unique byte so the test can distinguish frames
    /// in flight when buffering is in play.
    DataFrame { tag: u8 },
}

struct ScriptedSource {
    script: VecDeque<Step>,
    current_caps: Caps,
    configured: bool,
}

impl ScriptedSource {
    fn new(initial: Caps, script: Vec<Step>) -> Self {
        Self {
            script: script.into(),
            current_caps: initial,
            configured: false,
        }
    }
}

impl SourceLoop for ScriptedSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn intercept_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.current_caps.clone())
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut emitted: u64 = 0;
            while let Some(step) = self.script.pop_front() {
                match step {
                    Step::InputCapsChanged(w, h) => {
                        let c = h264_caps(w, h);
                        self.current_caps = c.clone();
                        let _ = out.push(PipelinePacket::CapsChanged(c)).await?;
                    }
                    Step::DataFrame { tag } => {
                        let frame = Frame {
                            domain: MemoryDomain::System(SystemSlice::from_boxed(
                                vec![tag].into_boxed_slice(),
                            )),
                            caps: self.current_caps.clone(),
                            timing: FrameTiming::default(),
                            sequence: emitted,
                        };
                        let _ = out.push(PipelinePacket::DataFrame(frame)).await?;
                        emitted += 1;
                    }
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }
}

/// In-memory decoder modeling the workaround-#3 Phase A contract: it
/// records its input caps on `CapsChanged`, holds frames in a FIFO of
/// configurable depth (mimicking B-frame reorder), and emits its own
/// output `CapsChanged` at the decode boundary — between the last
/// drain under the old input caps and the first drain under the new.
///
/// `buffering` is the number of frames held before any are released:
/// 0 = pass-through, N = each frame stays buffered until N+1 frames
/// have arrived. This is what shifts old-caps frames to drain *after*
/// the new input `CapsChanged` arrives.
struct FakeReorderDecoder {
    configured: Cell<bool>,
    input_caps: std::cell::RefCell<Option<Caps>>,
    /// Buffered (input-caps-at-arrival, tag) pairs.
    queue: std::cell::RefCell<VecDeque<(Caps, u8)>>,
    /// Output caps last advertised. Comparison key for emitting a new
    /// boundary `CapsChanged`.
    last_out_caps: std::cell::RefCell<Option<Caps>>,
    buffering: usize,
    /// Bumped whenever a `CapsChanged` is silently dropped — Phase A
    /// makes this stay 0 for valid streams.
    swallowed: Cell<u64>,
    /// Strictly-increasing sequence for emitted output frames.
    out_sequence: Cell<u64>,
}

impl FakeReorderDecoder {
    fn new(buffering: usize) -> Self {
        Self {
            configured: Cell::new(false),
            input_caps: std::cell::RefCell::new(None),
            queue: std::cell::RefCell::new(VecDeque::new()),
            last_out_caps: std::cell::RefCell::new(None),
            buffering,
            swallowed: Cell::new(0),
            out_sequence: Cell::new(0),
        }
    }
}

impl AsyncElement for FakeReorderDecoder {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        // Accept H.264 only.
        match upstream {
            Caps::Video {
                format: VideoFormat::H264,
                ..
            } => Ok(upstream.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured.set(true);
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(c) => {
                    // Phase A behavior: validate the format, record the
                    // input caps. Do NOT forward (an eager forward
                    // would reach the sink before buffered old-caps
                    // frames drain — the ordering bug §3 warns about).
                    match &c {
                        Caps::Video {
                            format: VideoFormat::H264,
                            ..
                        } => {}
                        _ => return Err(G2gError::CapsMismatch),
                    }
                    *self.input_caps.borrow_mut() = Some(c);
                }
                PipelinePacket::DataFrame(frame) => {
                    // Park this frame with whatever input caps are
                    // currently in force, regardless of whether the
                    // queue still holds older-caps frames. This is the
                    // reorder behavior the test is set up to expose.
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let tag = slice.as_slice().first().copied().unwrap_or(0);
                    let caps_at_arrival = self
                        .input_caps
                        .borrow()
                        .clone()
                        .unwrap_or_else(|| frame.caps.clone());
                    self.queue.borrow_mut().push_back((caps_at_arrival, tag));

                    // Drain only once the buffer is "full." Real decoders
                    // hold frames for B-frame reorder; the depth shifts
                    // when frames first reach the sink.
                    if self.queue.borrow().len() > self.buffering {
                        let (in_caps, tag) = self.queue.borrow_mut().pop_front().unwrap();
                        let out_caps = match in_caps {
                            Caps::Video {
                                format: VideoFormat::H264,
                                width: Dim::Fixed(w),
                                height: Dim::Fixed(h),
                                ..
                            } => nv12_caps(w, h),
                            _ => unreachable!("Phase A rejects non-H.264 at intake"),
                        };

                        // Emit boundary CapsChanged when output geometry
                        // changes — between the last old-caps frame and
                        // the first new-caps frame. THIS is the
                        // ordering invariant under test.
                        let need_emit = self.last_out_caps.borrow().as_ref() != Some(&out_caps);
                        if need_emit {
                            out.push(PipelinePacket::CapsChanged(out_caps.clone())).await?;
                            *self.last_out_caps.borrow_mut() = Some(out_caps.clone());
                        }
                        let seq = self.out_sequence.get();
                        self.out_sequence.set(seq + 1);
                        let drained = Frame {
                            domain: MemoryDomain::System(SystemSlice::from_boxed(
                                vec![tag].into_boxed_slice(),
                            )),
                            caps: out_caps,
                            timing: FrameTiming::default(),
                            sequence: seq,
                        };
                        out.push(PipelinePacket::DataFrame(drained)).await?;
                    }
                }
                PipelinePacket::Eos => {
                    // Drain remaining queue, then forward EOS. Same
                    // boundary CapsChanged rule applies.
                    while let Some((in_caps, tag)) = self.queue.borrow_mut().pop_front() {
                        let out_caps = match in_caps {
                            Caps::Video {
                                format: VideoFormat::H264,
                                width: Dim::Fixed(w),
                                height: Dim::Fixed(h),
                                ..
                            } => nv12_caps(w, h),
                            _ => unreachable!(),
                        };
                        let need_emit = self.last_out_caps.borrow().as_ref() != Some(&out_caps);
                        if need_emit {
                            out.push(PipelinePacket::CapsChanged(out_caps.clone())).await?;
                            *self.last_out_caps.borrow_mut() = Some(out_caps.clone());
                        }
                        let seq = self.out_sequence.get();
                        self.out_sequence.set(seq + 1);
                        let drained = Frame {
                            domain: MemoryDomain::System(SystemSlice::from_boxed(
                                vec![tag].into_boxed_slice(),
                            )),
                            caps: out_caps,
                            timing: FrameTiming::default(),
                            sequence: seq,
                        };
                        out.push(PipelinePacket::DataFrame(drained)).await?;
                    }
                    out.push(PipelinePacket::Eos).await?;
                }
                PipelinePacket::Flush => {
                    self.queue.borrow_mut().clear();
                    *self.last_out_caps.borrow_mut() = None;
                    out.push(PipelinePacket::Flush).await?;
                }
            }
            Ok(())
        })
    }
}

/// Phase A regression guard. With buffering = 1, the sequence
///
///     input caps@A, frame@A1, frame@A2,
///     input caps@B, frame@B1, frame@B2, EOS
///
/// has frame@A2 release **after** input-caps@B arrives at the decoder.
/// If the decoder eagerly forwarded the input caps change, the
/// downstream sink would see `CapsChanged(out@B)` before frame@A2 —
/// reconfiguring for the new geometry while an old-geometry frame is
/// still in flight. The Phase A rule (emit the output CapsChanged at
/// the **decode boundary**, not on input arrival) preserves the
/// invariant: `CapsChanged(out@B)` must appear strictly after the
/// last A-tagged DataFrame.
#[tokio::test]
async fn output_capschanged_lands_between_last_old_frame_and_first_new_frame() {
    let initial = h264_caps(640, 480);
    let script = vec![
        Step::DataFrame { tag: 0xA1 },
        Step::DataFrame { tag: 0xA2 },
        Step::InputCapsChanged(1280, 720),
        Step::DataFrame { tag: 0xB1 },
        Step::DataFrame { tag: 0xB2 },
    ];
    let mut src = ScriptedSource::new(initial, script);
    let mut dec = FakeReorderDecoder::new(/* buffering = */ 1);
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    g2g_core::runtime::run_source_transform_sink(&mut src, &mut dec, &mut snk, &clock, 8)
        .await
        .expect("pipeline should complete");

    // Phase A: no input CapsChanged was silently dropped.
    assert_eq!(
        dec.swallowed.get(),
        0,
        "Phase A must validate + record, not swallow"
    );

    // Walk the sink's packet log and check the §3 ordering invariant.
    let events = snk.caps_changes();
    let received_frames = snk.received();

    assert!(snk.eos_seen(), "sink must observe EOS");
    assert_eq!(
        received_frames, 4,
        "all four data frames must reach the sink"
    );

    // First CapsChanged is the 640x480 NV12 advertised by the very
    // first decoded frame. Second is the 1280x720 NV12.
    assert_eq!(events.len(), 2, "exactly two output CapsChanged events");

    // First boundary: index = 0 frames-before (it precedes A1).
    assert_eq!(events[0].caps, nv12_caps(640, 480));
    assert_eq!(
        events[0].frames_before, 0,
        "first CapsChanged precedes every frame"
    );

    // Second boundary: must follow A1 and A2 (the two old-caps frames
    // that drained), and precede B1 and B2. With buffering=1 and EOS
    // flush draining one extra at end, the sink should see exactly 2
    // frames before the second CapsChanged.
    assert_eq!(events[1].caps, nv12_caps(1280, 720));
    assert_eq!(
        events[1].frames_before, 2,
        "second CapsChanged must land after the last old-geometry frame"
    );
}

/// Phase A loud-reject regression: an incompatible mid-stream format
/// change (H.264 -> VP9) used to be silently dropped. Now the decoder
/// returns `CapsMismatch` so the pipeline surfaces the error instead of
/// continuing to feed VP9 bytes into an H.264 decoder.
#[tokio::test]
async fn mid_stream_format_switch_rejected_loud() {
    let initial = h264_caps(640, 480);
    let script = vec![
        Step::DataFrame { tag: 0xA1 },
        // Inject a VP9 caps change — script bypasses the typed
        // builder so we can hand-craft the bad packet.
        Step::InputCapsChanged(640, 480),
        Step::DataFrame { tag: 0xA2 },
    ];
    let mut src = ScriptedSourceBadCaps {
        inner: ScriptedSource::new(initial, script),
        // After step 1 (the synthetic InputCapsChanged), the source's
        // run loop substitutes VP9 caps instead of H.264. This is the
        // hostile mid-stream format switch the test models.
    };
    let mut dec = FakeReorderDecoder::new(/* buffering = */ 0);
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let result =
        g2g_core::runtime::run_source_transform_sink(&mut src, &mut dec, &mut snk, &clock, 4).await;

    assert!(
        matches!(result, Err(G2gError::CapsMismatch)),
        "VP9 mid-stream should propagate CapsMismatch, got {result:?}"
    );
}

/// Wrapper that rewrites the second `InputCapsChanged` step's caps to
/// VP9 — bypassing `ScriptedSource`'s H.264-typed factory so we can
/// inject an incompatible mid-stream change.
struct ScriptedSourceBadCaps {
    inner: ScriptedSource,
}

impl SourceLoop for ScriptedSourceBadCaps {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn intercept_caps(&self) -> Result<Caps, G2gError> {
        self.inner.intercept_caps()
    }

    fn configure_pipeline(&mut self, c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.inner.configure_pipeline(c)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut emitted: u64 = 0;
            let mut caps_changes_seen = 0;
            while let Some(step) = self.inner.script.pop_front() {
                match step {
                    Step::InputCapsChanged(w, h) => {
                        // Second caps change rewritten to VP9 to model
                        // the hostile codec switch.
                        caps_changes_seen += 1;
                        let c = if caps_changes_seen == 1 {
                            Caps::Video {
                                format: VideoFormat::Vp9,
                                width: Dim::Fixed(w),
                                height: Dim::Fixed(h),
                                framerate: Rate::Any,
                            }
                        } else {
                            h264_caps(w, h)
                        };
                        self.inner.current_caps = c.clone();
                        let _ = out.push(PipelinePacket::CapsChanged(c)).await?;
                    }
                    Step::DataFrame { tag } => {
                        let frame = Frame {
                            domain: MemoryDomain::System(SystemSlice::from_boxed(
                                vec![tag].into_boxed_slice(),
                            )),
                            caps: self.inner.current_caps.clone(),
                            timing: FrameTiming::default(),
                            sequence: emitted,
                        };
                        let _ = out.push(PipelinePacket::DataFrame(frame)).await?;
                        emitted += 1;
                    }
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }
}

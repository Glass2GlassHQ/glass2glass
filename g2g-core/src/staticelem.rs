//! Static (heap-free) element model for the no-alloc / MCU path (Phase 2 of the
//! alloc-optional core): the generic twin of the object-safe [`AsyncElement`] /
//! [`OutputSink`], which box a future per frame (`element.rs`, the honest
//! per-frame allocation boundary pinned by M616). Elements here are concrete types
//! wired by direct calls and driven by a const-arity runner, so a whole pipeline
//! monomorphizes to unboxed `async` state machines: no `dyn`, no `Box`, no
//! allocation. This is the M620 concrete-chain pattern promoted to an API.
//!
//! The traits use `async fn` in trait (stable on MSRV 1.75), so a stage's future
//! is an anonymous type inlined into the caller, never boxed. The runners are
//! generic and executor-agnostic: on an MCU an Embassy task `.await`s them, on a
//! host `block_on` drives them. Because nothing here allocates, a chain built from
//! these traits links on a target with no global allocator (proven end to end by
//! `examples/g2g-noalloc`).
//!
//! [`AsyncElement`]: crate::element::AsyncElement
//! [`OutputSink`]: crate::element::OutputSink

use core::future::Future;
use core::pin::pin;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use crate::error::G2gError;
use crate::frame::Frame;

/// A heap-free source: yields frames until the stream ends (`Ok(None)` at EOS).
///
/// The `#[allow(async_fn_in_trait)]` is intentional: this model targets a single
/// executor (an Embassy task on an MCU, `block_on` on a host), so the auto-trait
/// (`Send`) leakage the lint warns about is a non-issue, and avoiding it (a boxed
/// or `-> impl Future + Send` return) would reintroduce the allocation this model
/// exists to remove.
#[allow(async_fn_in_trait)]
pub trait StaticSource {
    /// Produce the next frame, or `Ok(None)` at end of stream.
    async fn next(&mut self) -> Result<Option<Frame>, G2gError>;
}

/// A heap-free 1:(0 or 1) transform: consumes a frame and optionally emits one (a
/// filter/decimator returns `Ok(None)` to drop it).
#[allow(async_fn_in_trait)]
pub trait StaticTransform {
    /// Transform `input`, optionally producing an output frame.
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError>;
}

/// A heap-free terminal sink.
#[allow(async_fn_in_trait)]
pub trait StaticSink {
    /// Consume one frame.
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError>;
}

/// A heap-free 2:(0 or 1) fan-in stage (a mixer, an interleaver): consumes one
/// frame from each of two inputs and optionally emits one. Const-arity like
/// everything in this model: the input count is fixed in the trait, so the
/// stage monomorphizes with no pad list. Returning `Ok(None)` drops the pair.
#[allow(async_fn_in_trait)]
pub trait StaticFanIn2 {
    /// Combine one frame from each input, optionally producing an output frame.
    async fn process2(&mut self, a: Frame, b: Frame) -> Result<Option<Frame>, G2gError>;
}

/// Compose two transforms into one, running `A` then `B` on its output, so a
/// static chain can carry more than one middle stage without a heap-allocated
/// element list: `Chain(a, Chain(b, c))` is a three-transform pipeline that still
/// monomorphizes to one unboxed future. `A` dropping a frame (`Ok(None)`)
/// short-circuits `B`.
#[derive(Debug)]
pub struct Chain<A, B>(pub A, pub B);

impl<A: StaticTransform, B: StaticTransform> StaticTransform for Chain<A, B> {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        match self.0.process(input).await? {
            Some(mid) => self.1.process(mid).await,
            None => Ok(None),
        }
    }
}

/// Fuse a transform onto a source, yielding the transform's output: the
/// static analog of a `source ! transform` bin. A const-arity runner slot
/// that takes one source can then carry a whole linear branch, which is how a
/// fan-in graph gets per-input chains ([`run_sources_fanin_sink`] with a
/// `SourceChain` in each source slot). A frame the transform drops
/// (`Ok(None)`) is pulled past (the fused source polls the inner source
/// again), so downstream sees only surviving frames; end of stream is the
/// inner source's.
#[derive(Debug)]
pub struct SourceChain<S, T>(pub S, pub T);

impl<S: StaticSource, T: StaticTransform> StaticSource for SourceChain<S, T> {
    async fn next(&mut self) -> Result<Option<Frame>, G2gError> {
        loop {
            let Some(frame) = self.0.next().await? else { return Ok(None) };
            if let Some(out) = self.1.process(frame).await? {
                return Ok(Some(out));
            }
        }
    }
}

/// Fuse a transform onto a sink: the static analog of a `transform ! sink`
/// bin, giving a const-arity runner's sink slot a processing tail (a fan-in
/// graph's `mix -> encode -> send`). A frame the transform drops never
/// reaches the sink.
#[derive(Debug)]
pub struct SinkChain<T, K>(pub T, pub K);

impl<T: StaticTransform, K: StaticSink> StaticSink for SinkChain<T, K> {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        if let Some(out) = self.0.process(frame).await? {
            self.1.consume(out).await?;
        }
        Ok(())
    }
}

/// Drive a `source -> sink` chain to end of stream. Fully monomorphized; no `Box`,
/// no `dyn`, no allocation.
pub async fn run_source_sink<S, K>(mut src: S, mut sink: K) -> Result<(), G2gError>
where
    S: StaticSource,
    K: StaticSink,
{
    while let Some(frame) = src.next().await? {
        sink.consume(frame).await?;
    }
    Ok(())
}

/// Drive a `source -> transform -> sink` chain to end of stream. A transform that
/// returns `Ok(None)` drops the frame (the sink is not called for it). Compose
/// transforms with [`Chain`] for longer pipelines. Fully monomorphized.
pub async fn run_source_transform_sink<S, T, K>(
    mut src: S,
    mut transform: T,
    mut sink: K,
) -> Result<(), G2gError>
where
    S: StaticSource,
    T: StaticTransform,
    K: StaticSink,
{
    while let Some(frame) = src.next().await? {
        if let Some(out) = transform.process(frame).await? {
            sink.consume(out).await?;
        }
    }
    Ok(())
}

/// Drive a `{source_a, source_b} -> fan-in -> sink` graph to end of stream:
/// the const-arity fan-in analog of [`run_source_transform_sink`]. Pull is
/// lockstep and deterministic, one frame from each source per iteration (`a`
/// first), and the stream ends when either source ends; a fan-in needing
/// rate adaptation puts a resampler upstream, not a queue (there is none on
/// this path by design). A fan-in that returns `Ok(None)` drops the pair.
/// Fully monomorphized; no `Box`, no `dyn`, no allocation.
pub async fn run_sources_fanin_sink<SA, SB, F, K>(
    mut src_a: SA,
    mut src_b: SB,
    mut fanin: F,
    mut sink: K,
) -> Result<(), G2gError>
where
    SA: StaticSource,
    SB: StaticSource,
    F: StaticFanIn2,
    K: StaticSink,
{
    loop {
        let Some(a) = src_a.next().await? else { return Ok(()) };
        let Some(b) = src_b.next().await? else { return Ok(()) };
        if let Some(out) = fanin.process2(a, b).await? {
            sink.consume(out).await?;
        }
    }
}

/// The outcome of one [`step_source_sink`] iteration: the frame-at-a-time analog
/// of the `run_*` runners, for a caller that owns the loop rather than handing it
/// to the runner. This is what lets an external scheduler drive a static pipeline
/// one frame at a time and get control back, e.g. a C superloop calling in per
/// frame over the FFI seam (`g2g-mcu::cffi`), or an RTOS task interleaving the
/// pipeline with other work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// One frame was pulled from the source and delivered to the sink (or
    /// dropped by a fused transform); call again for the next.
    Advanced,
    /// The source reported end of stream; no frame this step, stop calling.
    Eos,
    /// A stage suspended (`Poll::Pending`). The step model is for stages that
    /// complete synchronously (polling drivers, the `g2g-mcu` mock/C seams); a
    /// genuinely suspending pipeline belongs on a real executor (Embassy), not
    /// a per-step caller. Reported, never silently looped.
    Pending,
}

/// The one-frame body, as a named `async fn` driven by a single [`drive_ready`]
/// poll (the same shape the `run_*` runners use). Kept a named fn, not an inline
/// `async {}`, so it monomorphizes like the runners.
async fn step_once<S, K>(src: &mut S, sink: &mut K) -> Result<bool, G2gError>
where
    S: StaticSource,
    K: StaticSink,
{
    match src.next().await? {
        Some(frame) => {
            sink.consume(frame).await?;
            Ok(true)
        }
        None => Ok(false),
    }
}

/// Run exactly one frame through a `source -> sink` chain and return, instead of
/// looping to end of stream like [`run_source_sink`]. The caller owns the loop:
/// call it once per frame and yield in between (a C superloop over the
/// `g2g-mcu::cffi` seam, an RTOS task). Compose a processing tail into the sink
/// with [`SinkChain`] (and a head with [`SourceChain`] / [`Chain`]), so this one
/// primitive steps any linear graph shape. Heap-free and single-poll like
/// [`drive_ready`]; the source's and sink's streaming state persist across calls
/// in the caller's `&mut` borrows.
///
/// Panic surface: a single-frame future does not give the optimizer the
/// termination proof a run-to-EOS loop does, so it leaves the compiler's
/// resumed-after-completion guard statically present in the archive. That guard
/// is runtime-unreachable (each per-step future is polled exactly once, never
/// again) and is NOT a data panic (no bounds / overflow / unwrap path);
/// `tools/cffi-check.sh` asserts the step path stays heap-free and free of data
/// panics, permitting only that one benign re-poll guard.
pub fn step_source_sink<S, K>(src: &mut S, sink: &mut K) -> Result<Step, G2gError>
where
    S: StaticSource,
    K: StaticSink,
{
    match drive_ready(step_once(src, sink)) {
        Some(Ok(true)) => Ok(Step::Advanced),
        Some(Ok(false)) => Ok(Step::Eos),
        Some(Err(e)) => Err(e),
        None => Ok(Step::Pending),
    }
}

// Blanket impls so a `&mut` to a stage is itself one: the runners take ownership,
// but a caller that keeps its sink for inspection passes a `&mut`. Heap-free (a
// reference forward).
impl<S: StaticSource> StaticSource for &mut S {
    async fn next(&mut self) -> Result<Option<Frame>, G2gError> {
        (**self).next().await
    }
}

impl<T: StaticTransform> StaticTransform for &mut T {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        (**self).process(input).await
    }
}

impl<K: StaticSink> StaticSink for &mut K {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        (**self).consume(frame).await
    }
}

impl<F: StaticFanIn2> StaticFanIn2 for &mut F {
    async fn process2(&mut self, a: Frame, b: Frame) -> Result<Option<Frame>, G2gError> {
        (**self).process2(a, b).await
    }
}

/// Drive an always-ready future with a single noop-waker poll: the minimal
/// executor for a static chain whose stages never suspend (every `g2g-mcu`
/// mock-peripheral element, the g2g-noalloc proof pipeline). Returns `None`
/// if the future is `Pending`, which with a noop waker could never be woken
/// again anyway; a suspending pipeline belongs on a real executor (Embassy).
///
/// Safe to call, so an application crate under `#![forbid(unsafe_code)]` can
/// run a whole pipeline (the `unsafe` waker plumbing lives here, once).
/// Polling exactly once (not a re-poll loop) also lets the optimizer discharge
/// the compiler's resumed-after-completion panic arm, which the panic-free
/// symbol proof (`tools/noalloc-check.sh`) relies on.
///
/// `#[inline]`: the future is taken by value, and inlining lets the caller's
/// future be polled in place instead of being memmoved into this frame (a
/// pipeline state machine is KBs; the footprint budgets count on the elision).
#[inline]
pub fn drive_ready<F: Future>(fut: F) -> Option<F::Output> {
    const VTABLE: RawWakerVTable =
        RawWakerVTable::new(|_| RawWaker::new(core::ptr::null(), &VTABLE), |_| {}, |_| {}, |_| {});
    // SAFETY: the vtable's clone returns an equivalent no-op waker and wake /
    // drop are no-ops, satisfying the Waker contract.
    let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    let fut = pin!(fut);
    match fut.poll(&mut cx) {
        Poll::Ready(v) => Some(v),
        Poll::Pending => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameTiming;
    use crate::memory::{MemoryDomain, SystemSlice};
    use core::future::Future;
    use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    // A no-op waker so the tests drive a future to completion without an executor
    // or any allocation (the static-chain futures never yield Pending here). Keeps
    // the test itself heap-free, matching the model under test.
    fn noop_waker() -> Waker {
        const VTABLE: RawWakerVTable =
            RawWakerVTable::new(|_| RawWaker::new(core::ptr::null(), &VTABLE), |_| {}, |_| {}, |_| {});
        // SAFETY: the vtable's clone returns an equivalent no-op RawWaker and the
        // wake/drop arms are no-ops, so the waker upholds the Waker contract.
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) }
    }

    fn drive<F: Future>(fut: F) -> F::Output {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut fut = core::pin::pin!(fut);
        loop {
            if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                return v;
            }
        }
    }

    // Static buffers the sources lend zero-copy (the MCU pattern: no per-frame
    // allocation, the bytes live in a fixed region).
    static SAMPLES: [u8; 4] = [10, 20, 30, 40];
    static SAMPLES_B: [u8; 3] = [15, 5, 25];

    /// Emits one frame per byte of a 'static buffer, lending the byte zero-copy.
    struct ByteSource {
        data: &'static [u8],
        idx: usize,
    }
    impl ByteSource {
        fn over(data: &'static [u8]) -> Self {
            Self { data, idx: 0 }
        }
    }
    impl StaticSource for ByteSource {
        async fn next(&mut self) -> Result<Option<Frame>, G2gError> {
            if self.idx >= self.data.len() {
                return Ok(None);
            }
            let i = self.idx;
            self.idx += 1;
            // SAFETY: the buffer is 'static and never mutated; the lent slice covers
            // exactly one valid byte, and `free` is None (no reclamation needed).
            let slice = unsafe {
                SystemSlice::from_foreign(self.data.as_ptr().add(i), 1, None, core::ptr::null_mut())
            };
            Ok(Some(Frame::new(
                MemoryDomain::System(slice),
                FrameTiming { pts_ns: i as u64, ..FrameTiming::default() },
                i as u64,
            )))
        }
    }

    /// Drops odd-indexed frames (a decimator), proving the `Ok(None)` drop path.
    struct KeepEven;
    impl StaticTransform for KeepEven {
        async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
            if input.sequence % 2 == 0 {
                Ok(Some(input))
            } else {
                Ok(None)
            }
        }
    }

    /// Records the first payload byte of each frame it receives.
    struct CollectSink {
        seen: [u8; 8],
        n: usize,
    }
    impl StaticSink for CollectSink {
        async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
            if let Some(s) = frame.domain.as_system_slice() {
                self.seen[self.n] = s[0];
                self.n += 1;
            }
            Ok(())
        }
    }

    #[test]
    fn source_transform_sink_runs_the_static_chain() {
        let mut sink = CollectSink { seen: [0; 8], n: 0 };
        // Run source -> KeepEven -> sink; the runner consumes `sink` so collect the
        // result by reading a shared cell instead. Simpler: build, run, inspect.
        let src = ByteSource::over(&SAMPLES);
        // Move sink in and out via a wrapper that borrows.
        drive(run_source_transform_sink(src, KeepEven, &mut sink)).unwrap();
        // Frames 0 and 2 survive KeepEven (seq 0,2); their bytes are SAMPLES[0], [2].
        assert_eq!(&sink.seen[..sink.n], &[10, 30], "even-sequence frames reached the sink");
    }

    #[test]
    fn chain_composes_two_transforms() {
        // KeepEven then a pass-through: same survivors, proving Chain wires A->B.
        struct PassThrough;
        impl StaticTransform for PassThrough {
            async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
                Ok(Some(input))
            }
        }
        let mut sink = CollectSink { seen: [0; 8], n: 0 };
        drive(run_source_transform_sink(
            ByteSource::over(&SAMPLES),
            Chain(KeepEven, PassThrough),
            &mut sink,
        ))
        .unwrap();
        assert_eq!(&sink.seen[..sink.n], &[10, 30], "chained transforms preserve behavior");
    }

    #[test]
    fn source_sink_visits_every_frame() {
        let mut sink = CollectSink { seen: [0; 8], n: 0 };
        drive(run_source_sink(ByteSource::over(&SAMPLES), &mut sink)).unwrap();
        assert_eq!(&sink.seen[..sink.n], &[10, 20, 30, 40], "no transform: every frame arrives");
    }

    #[test]
    fn source_chain_and_sink_chain_fuse_transforms() {
        // The same KeepEven behavior, fused on the source side...
        let mut sink = CollectSink { seen: [0; 8], n: 0 };
        drive(run_source_sink(SourceChain(ByteSource::over(&SAMPLES), KeepEven), &mut sink))
            .unwrap();
        assert_eq!(&sink.seen[..sink.n], &[10, 30], "SourceChain skips dropped frames");
        // ...and on the sink side, must agree with the plain runner.
        let mut sink = CollectSink { seen: [0; 8], n: 0 };
        drive(run_source_sink(ByteSource::over(&SAMPLES), SinkChain(KeepEven, &mut sink)))
            .unwrap();
        assert_eq!(&sink.seen[..sink.n], &[10, 30], "SinkChain drops before the sink");
    }

    /// Emits whichever frame of the pair has the larger first payload byte,
    /// dropping pairs whose `a` sequence is odd (the fan-in drop path).
    struct PickMaxDropOdd;
    impl StaticFanIn2 for PickMaxDropOdd {
        async fn process2(&mut self, a: Frame, b: Frame) -> Result<Option<Frame>, G2gError> {
            if a.sequence % 2 != 0 {
                return Ok(None);
            }
            fn first_byte(f: &Frame) -> u8 {
                f.domain.as_system_slice().map_or(0, |s| s[0])
            }
            Ok(Some(if first_byte(&b) > first_byte(&a) { b } else { a }))
        }
    }

    #[test]
    fn step_drives_one_frame_at_a_time_then_reports_eos() {
        // The caller owns the loop: step returns Advanced per frame, then Eos.
        let mut src = ByteSource::over(&SAMPLES);
        let mut sink = CollectSink { seen: [0; 8], n: 0 };
        let mut steps = 0;
        loop {
            match step_source_sink(&mut src, &mut sink).unwrap() {
                Step::Advanced => steps += 1,
                Step::Eos => break,
                Step::Pending => panic!("synchronous stages never suspend"),
            }
        }
        assert_eq!(steps, 4, "one Advanced per source frame");
        assert_eq!(&sink.seen[..sink.n], &[10, 20, 30, 40], "every frame delivered, in order");
    }

    #[test]
    fn step_threads_a_fused_transform_tail_and_persists_state() {
        // With a SinkChain tail the step primitive covers a linear graph; the
        // transform's drop path and the sink's cross-call state both hold.
        let mut src = ByteSource::over(&SAMPLES);
        let mut sink = CollectSink { seen: [0; 8], n: 0 };
        let mut tail = SinkChain(KeepEven, &mut sink);
        for _ in 0..4 {
            let _ = step_source_sink(&mut src, &mut tail).unwrap();
        }
        // Frames 0 and 2 survive KeepEven across the four steps.
        assert_eq!(&sink.seen[..sink.n], &[10, 30], "fused transform + persistent sink state");
    }

    #[test]
    fn fanin_pulls_lockstep_and_ends_at_shorter_source() {
        let mut sink = CollectSink { seen: [0; 8], n: 0 };
        drive(run_sources_fanin_sink(
            ByteSource::over(&SAMPLES),
            ByteSource::over(&SAMPLES_B),
            PickMaxDropOdd,
            &mut sink,
        ))
        .unwrap();
        // Pairs: (10,15) -> 15 (b wins); (20,5) -> dropped (odd seq);
        // (30,25) -> 30 (a wins); then B ends, so A's 40 is never paired.
        assert_eq!(&sink.seen[..sink.n], &[15, 30], "lockstep pairing, drop path, EOS at min");
    }
}

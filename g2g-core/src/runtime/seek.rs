//! Seek request channel and segment-done back-channel.
//!
//! A cloneable handle carrying a pending [`Seek`] from the application to a
//! seek-aware source. The app calls [`SeekController::seek`]; the source's run
//! loop calls [`SeekController::take_pending`] between frames and, on a flushing
//! seek, emits `Flush`, repositions, emits the post-flush
//! [`Segment`](crate::segment::Segment), and resumes from the new position.
//!
//! Seeks travel upstream to the source in GStreamer; here the app holds the
//! controller and a clone lives in the source, so a seek reaches the producer
//! without a back-reference. The latest request wins: an app scrubbing fast
//! only needs the final target, so a new `seek` replaces any prior unhandled
//! one. A producing source polls between frames; a source parked at the end of
//! the stream waits in [`wait_event`](SeekController::wait_event), which a
//! `seek` wakes.
//!
//! ## Segment seeks
//!
//! A [`SeekFlags::SEGMENT`](crate::segment::SeekFlags::SEGMENT) seek asks the
//! source to play `[start, stop]` and, on reaching `stop`, report
//! **segment-done** instead of running to `Eos`, so the app can loop seamlessly
//! with a non-flushing accumulating seek (gapless, see
//! [`Segment::accumulate_seek`](crate::segment::Segment::accumulate_seek)). g2g
//! has no `PipelinePacket` for this (the GStreamer `SEGMENT_DONE` event/message);
//! it would force a new control packet through every element's exhaustive match.
//! Instead the controller carries it back on the same app<->source channel that
//! already exists: the source calls [`notify_segment_done`](Self::notify_segment_done)
//! at `stop`, the app observes [`segment_done_count`](Self::segment_done_count) /
//! [`take_segment_done`](Self::take_segment_done) and either re-arms a loop seek
//! or calls [`shutdown`](Self::shutdown) to end the loop. An app that attached a
//! bus ([`set_bus`](Self::set_bus)) also gets each completion as
//! [`BusMessage::SegmentDone`] at the same moment. A segment-looping
//! source that is idle between loops waits in `wait_event` and checks
//! [`is_shutdown`](Self::is_shutdown) so it can emit `Eos` and terminate.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::{Context, Poll, Waker};

use alloc::sync::Arc;

use spin::Mutex;

#[cfg(feature = "std")]
use alloc::vec::Vec;

use crate::bus::{BusHandle, BusMessage};
#[cfg(feature = "std")]
use crate::element::DynAsyncElement;
#[cfg(feature = "std")]
use crate::graph::{Graph, NodeId};
#[cfg(feature = "std")]
use crate::runtime::{DynSourceLoop, GraphNode, GraphNodeRef};
use crate::segment::Seek;

/// Source -> app segment-done state. `count` is monotonic so the app can detect
/// a *new* completion (e.g. spin until it advances); `fresh` is the take-once
/// flag for [`take_segment_done`](SeekController::take_segment_done).
#[derive(Debug, Default, Clone, Copy)]
struct SegmentDone {
    /// Number of `SEGMENT` segments the source has completed (monotonic).
    count: u64,
    /// Stream-time position (ns) where the most recent segment ended.
    position_ns: u64,
    /// Set by `notify_segment_done`, cleared by `take_segment_done`.
    fresh: bool,
}

#[derive(Debug, Default)]
struct SeekInner {
    pending: Mutex<Option<Seek>>,
    /// Segment-done back-channel (source -> app), the `SEGMENT_DONE` analog.
    segment_done: Mutex<SegmentDone>,
    /// App -> source stop request: ends a segment-looping source idling between
    /// loop seeks. Latches once set (a run is torn down, not resumed).
    shutdown: AtomicBool,
    /// Total length of the stream the seek positions address, once the source
    /// knows it. Latest-set wins.
    stream_len: Mutex<Option<u64>>,
    /// Pipeline bus, when the app attached one, so a segment-done also reaches
    /// it as [`BusMessage::SegmentDone`]. `None` leaves the back-channel the
    /// only route.
    bus: Mutex<Option<BusHandle>>,
    /// Waker a source parked in [`SeekController::wait_event`] registered, woken
    /// by `seek` / `shutdown` so an idle segment-looping source resumes without
    /// busy-polling. `None` when no source is parked.
    waker: Mutex<Option<Waker>>,
    seek_counts: Mutex<SeekCounts>,
    takes_seeks_after_end: AtomicBool,
    // the byte source a demuxer repositions, it stops and seeks after the end with this controller
    upstream: Mutex<Option<SeekController>>,
    run_held_open: AtomicBool,
    run_end_waker: Mutex<Option<Waker>>,
}

#[derive(Debug, Default, Clone, Copy)]
struct SeekCounts {
    flushing_seeks_taken: u64,
    // the flushing seek count at which every sink took the end of the stream
    stream_ended_at: Option<u64>,
}

/// Cloneable seek channel. Every clone shares one pending-seek slot, one
/// segment-done back-channel, and one shutdown flag.
#[derive(Debug, Clone, Default)]
pub struct SeekController {
    inner: Arc<SeekInner>,
}

impl SeekController {
    /// A controller with no pending seek.
    pub fn new() -> Self {
        Self::default()
    }

    /// Application side: request a seek. Replaces any prior unhandled request
    /// (latest-wins).
    pub fn seek(&self, seek: Seek) {
        *self.inner.pending.lock() = Some(seek);
        self.wake();
    }

    /// Source side: take and clear the pending seek, or `None` if none is set.
    pub fn take_pending(&self) -> Option<Seek> {
        let mut pending = self.inner.pending.lock();
        let seek = pending.take();
        if seek.is_some_and(|seek| seek.is_flush()) {
            self.inner.seek_counts.lock().flushing_seeks_taken += 1;
        }
        seek
    }

    /// Whether a seek is currently pending (not yet taken).
    pub fn has_pending(&self) -> bool {
        self.inner.pending.lock().is_some()
    }

    /// Source side: publish the total length of the stream, in the same unit as
    /// the seek positions on this controller (bytes for a demuxer-to-byte-source
    /// controller), when the source knows it (e.g. the file size). Latest-set
    /// wins.
    pub fn set_stream_len(&self, len: u64) {
        *self.inner.stream_len.lock() = Some(len);
    }

    /// Consumer side: the total stream length the source published, or `None`
    /// when it has not (or cannot). A seeking consumer bounds a guessed offset
    /// with it so a guess cannot land past the end.
    pub fn stream_len(&self) -> Option<u64> {
        *self.inner.stream_len.lock()
    }

    /// Application side: attach the pipeline bus, so every segment-done is also
    /// posted as [`BusMessage::SegmentDone`]. The app that owns both the
    /// controller and the bus wires them together; the back-channel works the
    /// same either way.
    pub fn set_bus(&self, bus: BusHandle) {
        *self.inner.bus.lock() = Some(bus);
    }

    /// Source side: report that a `SEGMENT` segment finished at stream-time
    /// `position_ns` (the `SEGMENT_DONE` signal). Bumps the completion count,
    /// arms [`take_segment_done`](Self::take_segment_done), and posts
    /// [`BusMessage::SegmentDone`] when a bus is attached.
    pub fn notify_segment_done(&self, position_ns: u64) {
        {
            let mut d = self.inner.segment_done.lock();
            d.count = d.count.saturating_add(1);
            d.position_ns = position_ns;
            d.fresh = true;
        }
        if let Some(bus) = self.inner.bus.lock().as_ref() {
            bus.try_post(BusMessage::SegmentDone { position_ns });
        }
    }

    /// Application side: take the stream-time position of the most recent
    /// segment-done, or `None` if none is unconsumed since the last take. The
    /// app reacts by re-arming a (typically non-flushing) loop seek.
    pub fn take_segment_done(&self) -> Option<u64> {
        let mut d = self.inner.segment_done.lock();
        if d.fresh {
            d.fresh = false;
            Some(d.position_ns)
        } else {
            None
        }
    }

    /// Application side: total `SEGMENT` segments the source has completed
    /// (monotonic). Lets the app wait for the next completion without consuming
    /// the take-once slot.
    pub fn segment_done_count(&self) -> u64 {
        self.inner.segment_done.lock().count
    }

    /// Application side: ask a segment-looping source to stop and emit `Eos`.
    /// Latching: a torn-down run is not resumed.
    pub fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::SeqCst);
        self.wake();
        if let Some(upstream) = self.upstream() {
            upstream.shutdown();
        }
    }

    /// Source side: whether the app has requested shutdown.
    pub fn is_shutdown(&self) -> bool {
        self.inner.shutdown.load(Ordering::SeqCst)
    }

    /// Source side: park until a seek is pending or shutdown is requested, then
    /// resolve. The wakeful idle for a segment-looping source between loops (the
    /// poll-free analog of pausing the source task): `seek` / `shutdown` wake the
    /// registered waker. The caller re-checks `take_pending` / `is_shutdown`
    /// after it resolves (a spurious early resolve is harmless, just a re-poll).
    pub fn wait_event(&self) -> WaitEvent<'_> {
        WaitEvent { ctl: self }
    }

    // a source that holds this controller then parks after its Eos instead of returning
    pub fn takes_seeks_after_end(&self) -> bool {
        self.inner.takes_seeks_after_end.load(Ordering::SeqCst)
    }

    pub(crate) fn enable_seeks_after_end(&self) {
        self.inner
            .takes_seeks_after_end
            .store(true, Ordering::SeqCst);
        if let Some(upstream) = self.upstream() {
            upstream.enable_seeks_after_end();
        }
    }

    // the run stays up after every sink took the end of the stream, until let_run_end
    pub fn keep_run_open(&self) {
        self.inner.run_held_open.store(true, Ordering::SeqCst);
    }

    pub fn let_run_end(&self) {
        self.inner.run_held_open.store(false, Ordering::SeqCst);
        self.wake_run_end();
    }

    pub(crate) fn run_held_open(&self) -> bool {
        self.inner.run_held_open.load(Ordering::SeqCst)
    }

    // every sink took the end of the stream of the latest flushing seek, and no seek is pending
    pub fn stream_ended(&self) -> bool {
        let counts = *self.inner.seek_counts.lock();
        counts.stream_ended_at == Some(counts.flushing_seeks_taken) && !self.has_pending()
    }

    pub(crate) fn flushing_seeks_taken(&self) -> u64 {
        self.inner.seek_counts.lock().flushing_seeks_taken
    }

    pub(crate) fn mark_stream_ended(&self, flushing_seeks_taken: u64) {
        self.inner.seek_counts.lock().stream_ended_at = Some(flushing_seeks_taken);
    }

    pub(crate) fn register_run_end_waker(&self, waker: &Waker) {
        *self.inner.run_end_waker.lock() = Some(waker.clone());
    }

    pub(crate) fn wake_run_end(&self) {
        if let Some(waker) = self.inner.run_end_waker.lock().take() {
            waker.wake();
        }
    }

    fn upstream(&self) -> Option<SeekController> {
        self.inner.upstream.lock().clone()
    }

    /// Resolve any parked [`wait_event`](Self::wait_event).
    fn wake(&self) {
        if let Some(w) = self.inner.waker.lock().take() {
            w.wake();
        }
    }

    /// Whether a seek is pending or shutdown is set (the `wait_event` ready
    /// condition).
    fn has_event(&self) -> bool {
        self.has_pending() || self.is_shutdown()
    }
}

/// Future returned by [`SeekController::wait_event`]. Resolves when a seek is
/// pending or shutdown is requested.
#[derive(Debug)]
pub struct WaitEvent<'a> {
    ctl: &'a SeekController,
}

impl Future for WaitEvent<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.ctl.has_event() {
            return Poll::Ready(());
        }
        // Register before the final check so a `seek` / `shutdown` that lands in
        // the gap still wakes us (it either sees the waker, or we see its state).
        *self.ctl.inner.waker.lock() = Some(cx.waker().clone());
        if self.ctl.has_event() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

// gst's `GstFormat` names for the two units a seek position comes in
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekFormat {
    Time,
    Bytes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ElementSeekRoute {
    // from the whole stream it holds, so its upstream never repositions
    Buffered,
    // by byte-seeking the source that feeds it
    UpstreamBytes,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachSeekError {
    NothingSeekable,
    SeveralSeekable(Vec<&'static str>),
}

#[cfg(feature = "std")]
impl core::fmt::Display for AttachSeekError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NothingSeekable => write!(
                f,
                "nothing in the pipeline can seek: a seek needs an MP4 file source \
                 (uridecodebin uri=file://...mp4), qtdemux, or a demuxer fed straight by a filesrc"
            ),
            Self::SeveralSeekable(elements) => write!(
                f,
                "{} elements can seek ({}), a seek needs exactly one",
                elements.len(),
                elements.join(", ")
            ),
        }
    }
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy)]
pub(crate) enum SeekableNode {
    TimeSource(NodeId),
    BufferedElement(NodeId),
    Demuxer {
        demuxer: NodeId,
        byte_source: NodeId,
    },
}

#[cfg(feature = "std")]
impl SeekableNode {
    pub(crate) fn time_node(self) -> NodeId {
        match self {
            Self::TimeSource(node) | Self::BufferedElement(node) => node,
            Self::Demuxer { demuxer, .. } => demuxer,
        }
    }

    pub(crate) fn byte_source(self) -> Option<NodeId> {
        match self {
            Self::TimeSource(_) | Self::BufferedElement(_) => None,
            Self::Demuxer { byte_source, .. } => Some(byte_source),
        }
    }
}

#[cfg(feature = "std")]
fn seekable_node(graph: &Graph<GraphNodeRef<'_>>, node: NodeId) -> Option<SeekableNode> {
    match graph.element(node)? {
        GraphNodeRef::Source(source) if source.seek_format() == Some(SeekFormat::Time) => {
            Some(SeekableNode::TimeSource(node))
        }
        GraphNodeRef::Element(element)
            if element.seek_route() == Some(ElementSeekRoute::Buffered) =>
        {
            Some(SeekableNode::BufferedElement(node))
        }
        GraphNodeRef::Element(element)
            if element.seek_route() == Some(ElementSeekRoute::UpstreamBytes) =>
        {
            let byte_source = graph
                .edges()
                .iter()
                .find(|edge| edge.dst.node == node)?
                .src
                .node;
            match graph.element(byte_source)? {
                GraphNodeRef::Source(source) if source.seek_format() == Some(SeekFormat::Bytes) => {
                    Some(SeekableNode::Demuxer {
                        demuxer: node,
                        byte_source,
                    })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(feature = "std")]
fn log_category(graph: &Graph<GraphNodeRef<'_>>, node: NodeId) -> &'static str {
    match graph.element(node) {
        Some(GraphNodeRef::Source(source)) => source.log_category(),
        Some(GraphNodeRef::Element(element)) => element.log_category(),
        _ => unreachable!("a seekable node is a source or an element"),
    }
}

#[cfg(feature = "std")]
fn source_mut(graph: &mut Graph<GraphNode>, node: NodeId) -> &mut (dyn DynSourceLoop + 'static) {
    match graph.element_mut(node) {
        Some(GraphNodeRef::Source(source)) => source.as_mut(),
        _ => unreachable!("seekable_node found a source here"),
    }
}

#[cfg(feature = "std")]
fn element_mut(graph: &mut Graph<GraphNode>, node: NodeId) -> &mut (dyn DynAsyncElement + 'static) {
    match graph.element_mut(node) {
        Some(GraphNodeRef::Element(element)) => element.as_mut(),
        _ => unreachable!("seekable_node found an element here"),
    }
}

#[cfg(feature = "std")]
pub fn attach_seek(
    graph: &mut Graph<GraphNode>,
    first_seek: Seek,
) -> Result<SeekController, AttachSeekError> {
    attach(graph, Some(first_seek))
}

// the application seeks through the controller once the run is under way
#[cfg(feature = "std")]
pub fn attach_seek_controller(
    graph: &mut Graph<GraphNode>,
) -> Result<SeekController, AttachSeekError> {
    attach(graph, None)
}

#[cfg(feature = "std")]
pub(crate) fn seek_target(
    graph: &Graph<GraphNodeRef<'_>>,
) -> Result<SeekableNode, AttachSeekError> {
    let seekable: Vec<SeekableNode> = (0..graph.node_count())
        .filter_map(|index| seekable_node(graph, NodeId(index as u32)))
        .collect();
    match seekable[..] {
        [target] => Ok(target),
        [] => Err(AttachSeekError::NothingSeekable),
        _ => Err(AttachSeekError::SeveralSeekable(
            seekable
                .iter()
                .map(|node| log_category(graph, node.time_node()))
                .collect(),
        )),
    }
}

#[cfg(feature = "std")]
fn attach(
    graph: &mut Graph<GraphNode>,
    first_seek: Option<Seek>,
) -> Result<SeekController, AttachSeekError> {
    let target = seek_target(graph)?;
    let app = SeekController::new();
    // queued before the element gets the controller, so it lands before the first frame
    if let Some(seek) = first_seek {
        app.seek(seek);
    }
    match target {
        SeekableNode::TimeSource(node) => source_mut(graph, node).set_seek(app.clone()),
        SeekableNode::BufferedElement(node) => {
            element_mut(graph, node).set_buffered_seek(app.clone())
        }
        SeekableNode::Demuxer {
            demuxer,
            byte_source,
        } => {
            let upstream = SeekController::new();
            *app.inner.upstream.lock() = Some(upstream.clone());
            source_mut(graph, byte_source).set_seek(upstream.clone());
            element_mut(graph, demuxer).set_seek(app.clone(), upstream);
        }
    }
    Ok(app)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::{Seek, SeekFlags, SeekType};

    #[test]
    fn take_pending_clears_and_returns() {
        let c = SeekController::new();
        assert!(!c.has_pending());
        assert_eq!(c.take_pending(), None);

        c.seek(Seek::flush_to(5_000));
        assert!(c.has_pending());
        let s = c.take_pending().expect("a seek was pending");
        assert_eq!(s.start, 5_000);
        assert!(s.is_flush());
        // Taken: now empty.
        assert!(!c.has_pending());
        assert_eq!(c.take_pending(), None);
    }

    #[test]
    fn latest_seek_wins() {
        let c = SeekController::new();
        c.seek(Seek::flush_to(1_000));
        c.seek(Seek {
            rate: 1.0,
            flags: SeekFlags::FLUSH,
            start_type: SeekType::Set,
            start: 9_000,
            stop_type: SeekType::None,
            stop: 0,
        });
        // Only the final request survives.
        assert_eq!(c.take_pending().map(|s| s.start), Some(9_000));
    }

    #[test]
    fn clones_share_one_slot() {
        let app = SeekController::new();
        let src = app.clone();
        app.seek(Seek::flush_to(42));
        // The source-side clone observes the app-side request.
        assert_eq!(src.take_pending().map(|s| s.start), Some(42));
        assert!(!app.has_pending());
    }

    #[test]
    fn stream_len_is_published_source_side() {
        let app = SeekController::new();
        let src = app.clone();
        assert_eq!(app.stream_len(), None, "unknown until the source says");
        src.set_stream_len(1_234);
        assert_eq!(app.stream_len(), Some(1_234));
        // Latest-set wins (a re-opened / grown stream).
        src.set_stream_len(2_000);
        assert_eq!(app.stream_len(), Some(2_000));
    }

    #[test]
    fn segment_done_counts_and_takes_once() {
        let app = SeekController::new();
        let src = app.clone();
        assert_eq!(app.segment_done_count(), 0);
        assert_eq!(app.take_segment_done(), None);

        // Source reports a completed segment; the app sees the count advance and
        // can take the position exactly once.
        src.notify_segment_done(5_000);
        assert_eq!(app.segment_done_count(), 1);
        assert_eq!(app.take_segment_done(), Some(5_000));
        assert_eq!(app.take_segment_done(), None, "take is once per notify");

        // A second completion bumps the monotonic count and re-arms the take.
        src.notify_segment_done(10_000);
        assert_eq!(app.segment_done_count(), 2);
        assert_eq!(app.take_segment_done(), Some(10_000));
    }

    #[test]
    fn segment_done_posts_to_an_attached_bus_only() {
        use crate::bus::{Bus, BusMessage};

        // No bus attached: the back-channel still works, nothing is posted.
        let unattached = SeekController::new();
        unattached.notify_segment_done(1_000);
        assert_eq!(unattached.segment_done_count(), 1);

        let (bus, handle) = Bus::new(4);
        let app = SeekController::new();
        app.set_bus(handle);
        let src = app.clone();

        src.notify_segment_done(5_000);
        src.notify_segment_done(9_000);
        assert_eq!(
            bus.try_recv(),
            Some(BusMessage::SegmentDone { position_ns: 5_000 })
        );
        assert_eq!(
            bus.try_recv(),
            Some(BusMessage::SegmentDone { position_ns: 9_000 })
        );
        assert_eq!(bus.try_recv(), None, "one message per completed segment");
    }

    #[test]
    fn shutdown_latches_and_is_visible_to_the_source() {
        let app = SeekController::new();
        let src = app.clone();
        assert!(!src.is_shutdown());
        app.shutdown();
        assert!(src.is_shutdown());
        // Latching: it stays set (a run is torn down, not resumed).
        assert!(src.is_shutdown());
    }

    // `block_on` parks a thread, so it exists only on the std runtime.
    #[cfg(feature = "std")]
    #[test]
    fn wait_event_resolves_and_wakes() {
        use crate::runtime::block_on;
        // Already-pending: resolves immediately.
        let c = SeekController::new();
        c.seek(Seek::flush_to(1));
        block_on(c.wait_event());

        // A wake from `seek` resolves a previously-pending wait, and a wake from
        // `shutdown` resolves the idle case. Drive a parked wait by polling it
        // once (Pending), then satisfying it and polling again (Ready).
        let c2 = SeekController::new();
        let mut cx = Context::from_waker(Waker::noop());
        let mut fut = core::pin::pin!(c2.wait_event());
        assert!(fut.as_mut().poll(&mut cx).is_pending(), "no event yet");
        c2.shutdown();
        assert!(
            fut.as_mut().poll(&mut cx).is_ready(),
            "shutdown resolves the wait"
        );
    }
}

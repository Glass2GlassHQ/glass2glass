//! Gapless playback source (M383): concatenates a playlist of sources into one
//! continuous, monotonically-timed stream, the analog of GStreamer playbin's
//! `about-to-finish` + next-`uri` gapless playback.
//!
//! `GaplessSrc` wraps a current [`DynSourceLoop`] and a shared
//! [`GaplessController`]. It plays the current item, and when nothing is queued
//! behind it posts **about-to-finish** so the app can enqueue the next item
//! *during* playback (a seamless swap). On the current item's EOS it pulls the
//! next from the queue, rebases that item's timestamps onto the running timeline
//! (the previous item's end becomes the new item's zero), and continues, so the
//! downstream decode chain is reused without a flush or a gap. The inner items'
//! EOS packets are swallowed; the only terminal `Eos` is emitted once the app
//! [`finish`](GaplessController::finish)es the playlist and the queue drains.
//!
//! Scope (v1): the concatenated items must share a codec (the decode chain is
//! reused), the read-side analog of GStreamer reusing one decodebin across URIs.
//! A per-item caps refinement (e.g. a resolution change) still flows downstream
//! via the inner source's `CapsChanged`. An instant (flushing) switch that
//! preempts the current item is also supported (M384,
//! [`GaplessController::switch_now`], the `instant-uri` analog); an audio/video
//! offset is a follow-up.
//!
//! ```text
//! GaplessSrc(clip1) ! h264parse ! decoder ! sink     // app enqueues clip2, clip3, ... ; then finish()
//! ```

use core::fmt;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::runtime::{
    select2, DynSourceLoop, Either, GaplessController, GraphNode, Registry, SourceLoop, UriError,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, G2gError, Graph, OutputSink, PipelinePacket, PushOutcome,
};

/// A source that plays a playlist of sources back-to-back as one gapless stream.
/// See the module docs. Driven by any runner that accepts a [`SourceLoop`].
pub struct GaplessSrc {
    /// The item currently negotiated / about to play. The first item is set at
    /// construction (and configured by [`configure_pipeline`](SourceLoop::configure_pipeline));
    /// `run` takes it and then pulls successors from the controller.
    current: Option<Box<dyn DynSourceLoop>>,
    /// The app <-> source playlist + about-to-finish channel.
    ctl: GaplessController,
    configured: bool,
}

impl fmt::Debug for GaplessSrc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GaplessSrc")
            .field("has_current", &self.current.is_some())
            .field("configured", &self.configured)
            .field("ctl", &self.ctl)
            .finish()
    }
}

impl GaplessSrc {
    /// Build a gapless source playing `first`, then the items the app enqueues on
    /// `ctl` (the app keeps a clone of `ctl` to `enqueue` successors and `finish`
    /// the playlist). `first` is negotiated + configured through the normal
    /// `SourceLoop` startup; enqueued successors are negotiated + configured by
    /// `GaplessSrc` itself before each plays.
    pub fn new(first: Box<dyn DynSourceLoop>, ctl: GaplessController) -> Self {
        Self {
            current: Some(first),
            ctl,
            configured: false,
        }
    }
}

impl SourceLoop for GaplessSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        Box::pin(async move {
            let src = self.current.as_mut().ok_or(G2gError::NotConfigured)?;
            src.intercept_caps().await
        })
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let src = self.current.as_mut().ok_or(G2gError::NotConfigured)?;
        let outcome = src.configure_pipeline(absolute_caps)?;
        self.configured = true;
        Ok(outcome)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let mut total = 0u64;
            // Running-time offset added to each item's timestamps so the playlist
            // is one monotonic timeline (the previous item's end is the next
            // item's zero). The first item plays at offset 0.
            let mut offset = 0u64;
            let mut current = self.current.take();
            // The first item is already configured (configure_pipeline above);
            // every successor pulled from the queue needs its own negotiate +
            // configure before it can run.
            let mut needs_config = false;

            while let Some(mut src) = current.take() {
                if needs_config {
                    let caps = src.intercept_caps().await?;
                    src.configure_pipeline(&caps)?;
                }
                // Nothing queued behind this item (and no instant switch pending):
                // signal about-to-finish now so the app has this item's whole
                // playback to enqueue the next one (the swap is then gap-free).
                if !self.ctl.has_next() && !self.ctl.is_finished() && !self.ctl.has_instant() {
                    self.ctl.notify_about_to_finish();
                }

                // Play the item, but let an instant switch (M384) preempt it: race
                // its `run` against `wait_instant`. If the switch wins, `select2`
                // drops the `run` future, cancelling the inner source mid-stream
                // (its remaining frames are abandoned, the instant-uri contract).
                // The adapter is scoped so its borrow on `out` ends before the
                // flush / next-item handling below reuses `out`; `frames` and
                // `end` are copied out (the inner's own count is lost when a
                // preemption drops its run future mid-stream).
                let (preempted, frames, end) = {
                    let mut adapter = ShiftSink {
                        out: &mut *out,
                        offset,
                        max_end: offset,
                        frames: 0,
                    };
                    let preempted =
                        match select2(src.run(&mut adapter), self.ctl.wait_instant()).await {
                            Either::Left(res) => {
                                res?; // propagate a run error
                                false
                            }
                            Either::Right(()) => true,
                        };
                    (preempted, adapter.frames, adapter.max_end)
                };
                total = total.saturating_add(frames);
                if !preempted {
                    // Advance the offset past this item's end (max emitted
                    // PTS+duration), so the next item starts where it stopped.
                    offset = end;
                }

                if preempted {
                    // Instant switch: flush the chain and reset the timeline, then
                    // play the requested source at offset 0.
                    out.push(PipelinePacket::Flush).await?;
                    offset = 0;
                    current = self.ctl.take_instant();
                    needs_config = true;
                    continue;
                }

                // Clean end: pick the next item. An instant switch still wins (a
                // flush + reset), else the gapless queue, else park until the app
                // enqueues, switches, or finishes the playlist.
                let mut flush = false;
                current = loop {
                    if let Some(s) = self.ctl.take_instant() {
                        flush = true;
                        break Some(s);
                    }
                    if let Some(next) = self.ctl.take_next() {
                        break Some(next);
                    }
                    if self.ctl.is_finished() {
                        break None;
                    }
                    self.ctl.wait_event().await;
                };
                if flush {
                    out.push(PipelinePacket::Flush).await?;
                    offset = 0;
                }
                needs_config = true;
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(total)
        })
    }
}

/// An [`OutputSink`] adapter wrapping the real downstream output for one playlist
/// item: it shifts each `DataFrame`'s PTS/DTS by `offset` (so items concatenate
/// onto one timeline), tracks the highest end time reached (`max_end`, the next
/// item's offset), and swallows the item's `Eos` (the gapless stream ends only at
/// playlist end). Caps / flush / segment packets pass through unchanged.
struct ShiftSink<'o> {
    out: &'o mut dyn OutputSink,
    offset: u64,
    /// Highest `pts + duration` forwarded so far (seeded with `offset`, so an
    /// empty item leaves the offset unchanged).
    max_end: u64,
    /// `DataFrame`s forwarded for this item, so `GaplessSrc` counts frames even
    /// when a preemption drops the inner source's run future (losing its count).
    frames: u64,
}

impl OutputSink for ShiftSink<'_> {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(mut f) => {
                    f.timing.pts_ns = f.timing.pts_ns.saturating_add(self.offset);
                    f.timing.dts_ns = f.timing.dts_ns.saturating_add(self.offset);
                    let end = f.timing.pts_ns.saturating_add(f.timing.duration_ns);
                    if end > self.max_end {
                        self.max_end = end;
                    }
                    self.frames = self.frames.saturating_add(1);
                    self.out.push(PipelinePacket::DataFrame(f)).await
                }
                // Swallow the inner item's EOS: only playlist-end emits a terminal
                // Eos (from `GaplessSrc::run`).
                PipelinePacket::Eos => Ok(PushOutcome::Accepted),
                // A per-item caps refinement / segment / flush still reaches the
                // chain unchanged.
                other => self.out.push(other).await,
            }
        })
    }
}

/// Why [`gapless_playbin`] could not assemble a graph.
#[derive(Debug)]
pub enum GaplessPlaybinError {
    /// The playlist was empty (a gapless source needs at least a first item).
    EmptyPlaylist,
    /// A URI could not be turned into a source, or its decode chain could not be
    /// plugged (wraps the [`UriError`], whose `Decode` variant carries the chain
    /// failure).
    Uri(UriError),
}

impl From<UriError> for GaplessPlaybinError {
    fn from(e: UriError) -> Self {
        GaplessPlaybinError::Uri(e)
    }
}

/// Build a gapless-playback graph from a playlist of URIs (M387), the convenience
/// over hand-wiring a [`GaplessSrc`]. Constructs the first URI's source, wraps it
/// in a `GaplessSrc`, enqueues the remaining URIs' sources on a shared
/// [`GaplessController`], and auto-plugs one decode chain (reused across items) to
/// `sink`. Returns the runnable `GaplessSrc -> decode -> sink` graph plus the
/// controller, so the caller can `enqueue` more items, `switch_now` (M384), and
/// must `finish()` the playlist when no more will be added (else the source parks
/// after the last item rather than emitting `Eos`).
///
/// All `uris` must decode-compatibly (same codec): the decode chain is plugged
/// once from the first item's caps and reused, the gapless contract (a per-item
/// caps refinement still flows via `CapsChanged`). `target` is the usual shape
/// predicate (`is_raw_video` / `is_raw_audio`).
pub fn gapless_playbin<Sk: AsyncElement + 'static>(
    reg: &Registry,
    uris: &[&str],
    sink: Sk,
    target: &dyn Fn(&Caps) -> bool,
    max_depth: usize,
) -> Result<(Graph<GraphNode>, GaplessController), GaplessPlaybinError> {
    let (first, rest) = uris
        .split_first()
        .ok_or(GaplessPlaybinError::EmptyPlaylist)?;
    let (first_src, caps) = reg.build_uri_source(first)?;
    let ctl = GaplessController::new();
    for uri in rest {
        let (src, _caps) = reg.build_uri_source(uri)?;
        ctl.enqueue(src);
    }
    let gapless = GaplessSrc::new(first_src, ctl.clone());
    let graph = reg.build_source_decodebin(Box::new(gapless), &caps, sink, target, max_depth)?;
    Ok((graph, ctl))
}

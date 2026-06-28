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
//! via the inner source's `CapsChanged`. Instant (flush) URI switching and an
//! audio/video offset are follow-ups.
//!
//! ```text
//! GaplessSrc(clip1) ! h264parse ! decoder ! sink     // app enqueues clip2, clip3, ... ; then finish()
//! ```

use core::fmt;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::runtime::{DynSourceLoop, GaplessController, SourceLoop};
use g2g_core::{
    Caps, ConfigureOutcome, G2gError, OutputSink, PipelinePacket, PushOutcome,
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
        Self { current: Some(first), ctl, configured: false }
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
                // Nothing queued behind this item: signal about-to-finish now so
                // the app has this item's whole playback to enqueue the next one
                // (the swap is then gap-free).
                if !self.ctl.has_next() && !self.ctl.is_finished() {
                    self.ctl.notify_about_to_finish();
                }

                let mut adapter = ShiftSink { out: &mut *out, offset, max_end: offset };
                total = total.saturating_add(src.run(&mut adapter).await?);
                // Advance the offset past this item's end (max emitted PTS+duration),
                // so the next item starts where this one stopped.
                offset = adapter.max_end;

                // Pull the next item, parking (wakefully) until the app enqueues
                // one or finishes the playlist.
                current = loop {
                    if let Some(next) = self.ctl.take_next() {
                        break Some(next);
                    }
                    if self.ctl.is_finished() {
                        break None;
                    }
                    self.ctl.wait_event().await;
                };
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

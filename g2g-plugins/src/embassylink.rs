//! Embassy zero-alloc inter-task link: a statically-sized channel carrying
//! `PipelinePacket`s between embedded tasks (DESIGN.md §6.2 "stack channels"),
//! the embassy-sync counterpart of the spin-based runtime channel. The app owns
//! the `PacketChannel` (e.g. in a `StaticCell` or `static`) and hands its `sink`
//! to a producer and its `receiver` to a consumer.
//!
//! The channel storage is static (no allocation), and the `OutputSink`
//! adapter pushes through the poll form, so a push costs no heap either. The
//! fully static element model (concrete future types, no `dyn`) remains the
//! static-graph layer (§4.8.1).

use embassy_sync::blocking_mutex::raw::{CriticalSectionRawMutex, NoopRawMutex, RawMutex};
use embassy_sync::channel::{Channel, Receiver, Sender};

use g2g_core::{G2gError, OutputSink, PipelinePacket, PushOutcome};

/// Statically-sized packet link. `M` is the embassy `RawMutex` and `N` the
/// queue depth. See [`SinglePacketChannel`] for the single-executor default.
#[allow(missing_debug_implementations)]
pub struct PacketChannel<M: RawMutex, const N: usize> {
    inner: Channel<M, PipelinePacket, N>,
}

impl<M: RawMutex, const N: usize> PacketChannel<M, N> {
    pub const fn new() -> Self {
        Self {
            inner: Channel::new(),
        }
    }

    /// An [`OutputSink`] that pushes packets into this channel; hand it to a
    /// producing source or transform.
    pub fn sink(&self) -> EmbassySink<'_, M, N> {
        EmbassySink {
            sender: self.inner.sender(),
        }
    }

    /// The receiving end for the consumer task.
    pub fn receiver(&self) -> Receiver<'_, M, PipelinePacket, N> {
        self.inner.receiver()
    }
}

impl<M: RawMutex, const N: usize> Default for PacketChannel<M, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// A [`PacketChannel`] for a single Embassy executor (`NoopRawMutex`). Use a
/// [`SharedPacketChannel`] if the link is shared with an interrupt handler or
/// must live in a `static`.
pub type SinglePacketChannel<const N: usize> = PacketChannel<NoopRawMutex, N>;

/// A [`PacketChannel`] over `CriticalSectionRawMutex`, which (unlike
/// `NoopRawMutex`) is `Sync`. Use this when the link is shared with an interrupt
/// handler, or when it must live in a `static` so spawned Embassy tasks reach it
/// by `&'static` reference (an executor's tasks take `'static` arguments). Needs
/// a `critical-section` impl at link. See `m264_embassy_multitask.rs`.
pub type SharedPacketChannel<const N: usize> = PacketChannel<CriticalSectionRawMutex, N>;

/// [`OutputSink`] over a [`PacketChannel`] sender, so an element pushes packets
/// into the embassy-sync channel (awaiting capacity under backpressure).
#[allow(missing_debug_implementations)]
pub struct EmbassySink<'a, M: RawMutex, const N: usize> {
    sender: Sender<'a, M, PipelinePacket, N>,
}

impl<M: RawMutex, const N: usize> OutputSink for EmbassySink<'_, M, N> {
    fn poll_push(
        &mut self,
        cx: &mut core::task::Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        use core::task::Poll;
        match self.sender.poll_ready_to_send(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(()) => {
                let taken = packet.take().expect("poll_push without a packet");
                match self.sender.try_send(taken) {
                    Ok(()) => Poll::Ready(Ok(PushOutcome::Accepted)),
                    // Lost the slot between readiness and send (another sender
                    // on the shared channel); park again for the next wake.
                    Err(embassy_sync::channel::TrySendError::Full(returned)) => {
                        *packet = Some(returned);
                        let _ = self.sender.poll_ready_to_send(cx);
                        Poll::Pending
                    }
                }
            }
        }
    }
}

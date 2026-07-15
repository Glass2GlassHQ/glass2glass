//! Embassy zero-alloc inter-task link: a statically-sized channel carrying
//! `PipelinePacket`s between embedded tasks (DESIGN.md §6.2 "stack channels"),
//! the embassy-sync counterpart of the spin-based runtime channel. The app owns
//! the `PacketChannel` (e.g. in a `StaticCell` or `static`) and hands its `sink`
//! to a producer and its `receiver` to a consumer.
//!
//! The channel storage is static (no allocation). The `OutputSink` adapter
//! still boxes its push future, since that trait is dyn-safe; a fully
//! allocation-free element model (concrete future types, no boxing) is the
//! static-graph layer (§4.8.1).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

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
        Self { inner: Channel::new() }
    }

    /// An [`OutputSink`] that pushes packets into this channel; hand it to a
    /// producing source or transform.
    pub fn sink(&self) -> EmbassySink<'_, M, N> {
        EmbassySink { sender: self.inner.sender() }
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
    fn push<'b>(
        &'b mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'b>> {
        Box::pin(async move {
            self.sender.send(packet).await;
            Ok(PushOutcome::Accepted)
        })
    }
}

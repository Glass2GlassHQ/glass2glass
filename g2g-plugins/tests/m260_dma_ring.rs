#![cfg(feature = "embassy-link")]
//! M260: a no-alloc DMA-ring capture source streams zero-copy frames through the
//! embassy-sync stack channel to a consumer, all under `block_on`. Proves the
//! full embedded path is heap-free end to end: a fixed `StaticLendRing` of `N`
//! slots, the source fills the next free slot (the DMA target) and lends it as a
//! `Frame` that *borrows* the slot, the frame crosses the static channel, and the
//! slot is reclaimed when the consumer drops the frame. With more frames than
//! slots, the same physical buffers recur (no per-frame allocation), and the
//! consumer's bytes alias the ring (zero copy).
//!
//! A real capture wires a DMA-completion ISR / HAL into the same ring; here the
//! `fill` callback stands in for the DMA write so the mechanism is host-testable.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError, OutputSink, PipelinePacket, Rate,
    RawVideoFormat, StaticLendRing,
};
use g2g_plugins::embassylink::SinglePacketChannel;

/// Per-frame observation: (sequence, first byte, slot base ptr, aliases the ring).
type FrameLog = Arc<Mutex<Vec<(u64, u8, usize, bool)>>>;

const SLOTS: usize = 4;
const BYTES: usize = 8;
const PAYLOAD: usize = 4;

/// A DMA-ring capture `SourceLoop`: for each of `total` frames it reserves a free
/// ring slot, fills it via `fill` (the DMA write stand-in), and lends it
/// zero-copy as a `System` frame. No heap: the ring's slots are the only frame
/// storage and they recycle as the consumer drops frames.
struct DmaRingSource<'r> {
    ring: &'r StaticLendRing<SLOTS, BYTES>,
    total: u64,
    fill: fn(&mut [u8], u64),
}

impl SourceLoop for DmaRingSource<'_> {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<g2g_core::CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(g2g_core::CapsConstraint::Produces(
            g2g_core::CapsSet::one(caps()),
        )))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..self.total {
                // Reserve a free slot; the ring is sized to exceed the in-flight
                // depth, so a downstream drop has freed one by the time we loop.
                let mut slot = self.ring.acquire().ok_or(G2gError::CapsMismatch)?;
                (self.fill)(slot.buf_mut(), seq);
                // SAFETY: `ring` outlives this source's `run` (the test keeps it on
                // the stack across `block_on`), hence outlives every lent frame.
                let payload = unsafe { slot.publish(PAYLOAD) };
                out.push(PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(payload),
                    timing: FrameTiming::default(),
                    sequence: seq,
                    meta: Default::default(),
                }))
                .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.total)
        })
    }
}

fn caps() -> Caps {
    // One RGBA pixel == PAYLOAD (4) bytes; the geometry is incidental, the test is
    // about the zero-copy lend, not pixel semantics.
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(1),
        height: Dim::Fixed(1),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Fill stands in for a DMA write: stamp the whole payload with the frame index.
fn stamp(buf: &mut [u8], seq: u64) {
    for b in buf[..PAYLOAD].iter_mut() {
        *b = seq as u8;
    }
}

#[test]
fn dma_ring_streams_zero_copy_through_embassy_channel_no_alloc() {
    let ring: StaticLendRing<SLOTS, BYTES> = StaticLendRing::new();
    let channel: SinglePacketChannel<1> = SinglePacketChannel::new();
    let total = 10u64;

    let mut src = DmaRingSource {
        ring: &ring,
        total,
        fill: stamp,
    };
    let mut sink = channel.sink();
    let rx = channel.receiver();

    // (seq, value-byte, distinct buffer base ptr, aliases-ring) per received frame.
    let log: FrameLog = Arc::new(Mutex::new(Vec::new()));

    let producer = src.run(&mut sink);
    let consumer = {
        let log = Arc::clone(&log);
        let ring = &ring;
        async move {
            loop {
                match rx.receive().await {
                    PipelinePacket::DataFrame(frame) => {
                        let Some(slice) = frame.domain.as_system_slice() else {
                            panic!("expected a System frame");
                        };
                        let bytes = slice;
                        let base = bytes.as_ptr() as usize;
                        log.lock().unwrap().push((
                            frame.sequence,
                            bytes[0],
                            base,
                            ring.contains(bytes.as_ptr()),
                        ));
                        // frame drops here: its slot returns to the ring.
                    }
                    PipelinePacket::Eos => break,
                    _ => {}
                }
            }
        }
    };

    let (run, ()) = embassy_futures::block_on(embassy_futures::join::join(producer, consumer));
    run.expect("source run completes");

    let log = log.lock().unwrap();
    assert_eq!(log.len(), total as usize, "every frame crossed the channel");
    // Zero-copy: each frame's bytes alias the ring and carry the producer's stamp
    // read back through the *same* memory (the consumer never sees a copy).
    for (seq, value, _base, in_ring) in log.iter() {
        assert!(*in_ring, "frame {seq} bytes live in the ring (zero copy)");
        assert_eq!(
            *value, *seq as u8,
            "frame {seq} carries its stamp through the lent slot"
        );
    }
    // No per-frame allocation: 10 frames reuse at most SLOTS physical buffers.
    let distinct: BTreeSet<usize> = log.iter().map(|(_, _, base, _)| *base).collect();
    assert!(
        distinct.len() <= SLOTS,
        "buffers recycle: {} distinct for {total} frames",
        distinct.len()
    );
    assert_eq!(ring.leased_count(), 0, "all slots reclaimed after drain");
}

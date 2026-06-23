//! Shared frame-emission helper for the packet-producing encoders
//! (`opusenc`, `vpxenc`, `av1enc`). They each turn a batch of encoded
//! `(payload, pts_ns)` packets into downstream `DataFrame`s, announcing the
//! output caps exactly once before the first frame, so the loop lives here.

use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{Caps, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket};

/// Push a batch of encoded `(payload, pts_ns)` packets downstream.
///
/// `caps` is announced via `CapsChanged` once, before the first frame is ever
/// emitted; `caps_sent` tracks that across calls so it fires at most once.
/// Each payload becomes a System-memory `DataFrame` with `dts == pts` and a
/// monotonic sequence number drawn from `emitted`. An empty batch is a no-op,
/// so the caps stay unannounced until real data arrives.
pub(crate) async fn emit_packets(
    caps_sent: &mut bool,
    emitted: &mut u64,
    packets: Vec<(Vec<u8>, u64)>,
    caps: &Caps,
    out: &mut dyn OutputSink,
) -> Result<(), G2gError> {
    if !packets.is_empty() && !*caps_sent {
        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
        *caps_sent = true;
    }
    for (data, pts_ns) in packets {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
            *emitted,
        );
        *emitted += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
    }
    Ok(())
}

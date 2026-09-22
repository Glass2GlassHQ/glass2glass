//! The `meta-only` reply mode the remote-transform elements share.
//!
//! A full round trip carries the whole frame both ways. For a stage that only
//! attaches metadata (inference, analytics) the return pixels are waste: the
//! client keeps its own frame, the peer replies with an empty payload carrying
//! the meta alone, and the client emits its retained frame with that meta on it.
//! Both ends have to agree, so a payload arriving in this mode is a protocol
//! error rather than something to discard.

use g2g_core::{G2gError, HardwareError, PipelinePacket, PropKind, PropertySpec};

/// The property every transform's transport lists, so a launch line can turn
/// the mode on.
pub const META_ONLY_PROPERTY: PropertySpec = PropertySpec::new(
    "meta-only",
    PropKind::Bool,
    "the peer replies with metadata alone and the frame is kept locally",
)
.with_default("false");

/// Put the peer's metadata on the frame that was sent. The reply carries the
/// meta alone, so anything else (a payload, a packet that is not a frame) means
/// the peer is not running this mode.
pub fn merge_meta(sent: PipelinePacket, reply: PipelinePacket) -> Result<PipelinePacket, G2gError> {
    let (PipelinePacket::DataFrame(mut kept), PipelinePacket::DataFrame(reply)) = (sent, reply)
    else {
        return Err(G2gError::Hardware(HardwareError::Other));
    };
    if reply.domain.as_system_slice().is_none_or(|s| !s.is_empty()) {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    kept.meta = reply.meta;
    Ok(PipelinePacket::DataFrame(kept))
}

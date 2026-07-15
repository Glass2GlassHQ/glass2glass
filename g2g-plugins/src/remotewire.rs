//! Shared helper for the distributed-graph transport elements (the `remote` TCP
//! pair and the `remote-ws` WebSocket pair): mapping a [`WireError`] from the
//! `g2g-core` codec to the pipeline error type. Both transports serialize the
//! identical `PipelinePacket` stream through [`g2g_core::wire`]; only the byte
//! transport under it differs.

use g2g_core::wire::WireError;
use g2g_core::{G2gError, HardwareError};

/// Map a [`WireError`] to the pipeline error type. A device / foreign memory
/// domain surfaces as `UnsupportedDomain` (the same error a CPU sink raises);
/// anything else is an internal encode / decode fault.
pub(crate) fn map_wire(e: WireError) -> G2gError {
    match e {
        WireError::UnsupportedDomain => G2gError::UnsupportedDomain,
        WireError::Truncated | WireError::BadTag => G2gError::Hardware(HardwareError::Other),
    }
}

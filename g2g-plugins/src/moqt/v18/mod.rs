//! IETF MoQ Transport draft-18, beside the draft-16 modules it grew out of.
//!
//! The version is picked per session: the client offers `moqt-18` and
//! `moqt-16` as WebTransport subprotocols and the server's choice selects the
//! codec. Draft-18 restructured the wire (its own `vi64` varint, a single
//! SETUP message on a pair of unidirectional control streams, one
//! bidirectional stream per request, bit-table data-plane headers), so the
//! wire layer lives here; what is genuinely version-agnostic (namespaces,
//! Key-Value-Pairs, object reordering, the catalog) is shared with the
//! draft-16 modules rather than copied.
//!
//! - [`coding`]: the `vi64` integer, namespaces, Key-Value-Pairs, and the
//!   typed control-message parameters that replaced KVP parameters.
//! - [`message`]: the control and request stream message set (Table 5).
//! - [`data`]: the bit-table SUBGROUP_HEADER and per-object fields.
//! - [`datagram`]: the bit-table OBJECT_DATAGRAM.
//! - [`session`]: the SETUP exchange, request streams, and the live data plane.

pub mod coding;
pub mod data;
pub mod datagram;
pub mod message;
pub mod session;

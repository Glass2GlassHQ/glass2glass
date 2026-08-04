//! IETF MoQ Transport (MOQT) draft-16, implemented in-tree over the M901
//! WebTransport carrier.
//!
//! The dialect is the IETF draft, version `0xff000010` (draft-16), which is
//! what Cloudflare's `moq-relay-ietf` runs. It is *not* moq-lite: that is a
//! single-vendor dialect with its own ALPN and cannot talk to IETF endpoints.
//! No crate implements the IETF draft on this workspace's MSRV, so the wire
//! layer is written here the way g2g's SRT and ST 2110 stacks were: read the
//! draft, read the reference implementation
//! (`cloudflare/moq-rs`, `moq-transport/src/{coding,setup,message,data}`), and
//! validate against the reference peer.
//!
//! The split:
//!
//! - [`coding`]: varints, byte strings, track namespaces and names, the
//!   delta-coded Key-Value-Pair sequences.
//! - [`message`]: the control-stream message set and its framing.
//! - [`data`]: the subgroup stream header and per-object header.
//! - [`datagram`]: the datagram object, the unreliable MTU-bounded carriage of
//!   one object.
//! - [`reassembly`]: decoding a subgroup stream, and putting the objects from
//!   many concurrent streams back into (group, object) order.
//! - [`catalog`]: the JSON track list, written by the publisher and read by the
//!   subscriber.
//! - [`session`]: the SETUP exchange and the live control / data streams.
//!
//! Everything but [`session`] is pure `alloc` and decodes byte vectors, so the
//! wire layer is unit-testable without a network.

pub mod catalog;
pub mod coding;
pub mod data;
pub mod datagram;
pub mod message;
pub mod reassembly;
pub mod session;

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
//! - [`fetch`]: the FETCH response stream, which carries already-published
//!   objects in order and is shared by both drafts.
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
pub mod fetch;
pub mod message;
pub mod reassembly;
pub mod session;
pub mod v18;

use alloc::vec::Vec;

use g2g_core::{G2gError, HardwareError};
use web_transport_quinn::RecvStream;

use coding::{MoqtError, Reader};

/// The draft versions this crate speaks, negotiated per session: the client
/// offers each as a WebTransport subprotocol in preference order and the
/// server's pick selects the codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoqtVersion {
    V16,
    V18,
}

impl MoqtVersion {
    /// A cursor over this version's integer flavour: draft-16 reads QUIC
    /// varints, draft-18 its own `vi64`.
    pub fn reader(self, buf: &[u8]) -> Reader<'_> {
        match self {
            Self::V16 => Reader::new(buf),
            Self::V18 => Reader::new_vi64(buf),
        }
    }

    /// Append an integer in this version's flavour.
    pub fn put_int(self, out: &mut Vec<u8>, v: u64) {
        match self {
            Self::V16 => coding::put_varint(out, v),
            Self::V18 => v18::coding::put_vi64(out, v),
        }
    }

    /// The WebTransport subprotocol that names this version on CONNECT.
    pub fn protocol(self) -> &'static str {
        match self {
            Self::V16 => session::MOQT_PROTOCOL,
            Self::V18 => v18::session::MOQT_PROTOCOL,
        }
    }

    fn from_protocol(protocol: &str) -> Option<Self> {
        [Self::V16, Self::V18]
            .into_iter()
            .find(|v| v.protocol() == protocol)
    }
}

/// Read just enough of a unidirectional stream to decode the type varint that
/// opens it, returning it and every byte read (the type included, since the
/// decoder that follows needs it).
///
/// Both drafts open every data stream this way, so one reader serves both: only
/// the integer flavour differs.
pub async fn read_stream_type(
    version: MoqtVersion,
    stream: &mut RecvStream,
) -> Result<(u64, Vec<u8>), G2gError> {
    // A varint is at most nine bytes, so this cannot grow on a peer's say-so.
    let mut buf = Vec::with_capacity(9);
    let mut chunk = [0u8; 9];
    loop {
        match version.reader(&buf).varint() {
            Ok(code) => return Ok((code, buf)),
            Err(MoqtError::Incomplete) if buf.len() < 9 => {}
            Err(_) => return Err(G2gError::Hardware(HardwareError::Other)),
        }
        match stream.read(&mut chunk[..9 - buf.len()]).await {
            Ok(Some(n)) if n > 0 => buf.extend_from_slice(&chunk[..n]),
            _ => return Err(G2gError::Hardware(HardwareError::Other)),
        }
    }
}

/// Parse a `versions` property: comma-separated draft numbers in preference
/// order (`"18,16"`). An empty list or an unknown number is refused, so a typo
/// fails the property rather than dialling with nothing.
pub fn parse_versions(list: &str) -> Result<Vec<MoqtVersion>, G2gError> {
    let mut versions = Vec::new();
    for part in list.split(',') {
        let version = match part.trim() {
            "16" => MoqtVersion::V16,
            "18" => MoqtVersion::V18,
            _ => return Err(G2gError::NotConfigured),
        };
        if !versions.contains(&version) {
            versions.push(version);
        }
    }
    if versions.is_empty() {
        return Err(G2gError::NotConfigured);
    }
    Ok(versions)
}

/// The version a dialled session negotiated. The server's subprotocol pick
/// decides (`moq-relay-ietf` echoes `moqt-16` when it is offered). A server
/// that names none predates multi-version offers, and every such server is a
/// draft-16 peer, so the fallback is draft-16 when it was offered; the SETUP
/// handshake that follows still validates the choice either way.
pub fn negotiated_version(
    session: &web_transport_quinn::Session,
    offered: &[MoqtVersion],
) -> Result<MoqtVersion, G2gError> {
    match session.response().protocol.as_deref() {
        Some(protocol) => {
            MoqtVersion::from_protocol(protocol).ok_or(G2gError::Hardware(HardwareError::Other))
        }
        None if offered.contains(&MoqtVersion::V16) => Ok(MoqtVersion::V16),
        None => Ok(offered[0]),
    }
}

//! Hand-rolled TURN client (RFC 5766 / 8656) for the str0m-based WebRTC
//! elements, the relay fallback for the NAT cases a STUN server-reflexive
//! candidate cannot punch through (symmetric NAT, restrictive firewalls). It is
//! the relay analog of the hand-rolled STUN Binding in [`crate::webrtc_util`].
//!
//! str0m is sans-IO and does not drive TURN itself: it only offers the
//! `Candidate::relayed` constructor. The data plane is the caller's job, which
//! is what this module provides. The integration in the elements is:
//!
//! - At setup, [`TurnClient::allocate`] does the two-step Allocate handshake
//!   (unauthenticated request, then a long-term-credential retry against the
//!   `401`'s `REALM` / `NONCE`) and learns the server-allocated relay address.
//!   The element adds that address to str0m as a relayed local candidate.
//! - str0m emits every `Output::Transmit` from a relay pair with
//!   `source == relay_addr` (the relayed candidate's base). The run loop routes
//!   those through the relay: [`TurnClient::wrap_send`] boxes the payload in a
//!   TURN Send indication addressed to the peer and sends it to the TURN
//!   server. Direct (host / server-reflexive) transmits are unchanged.
//! - Inbound UDP from the TURN server arrives as a Data indication;
//!   [`TurnClient::parse_data`] unwraps it to `(peer, payload)` and the element
//!   feeds str0m an `Input::Receive` with `source = peer`, `destination =
//!   relay_addr`. The server only relays peer traffic once a permission for that
//!   peer's IP exists, so the run loop calls [`TurnClient::ensure_permission`]
//!   (a `CreatePermission` request) for each new peer before relaying to it.
//! - [`TurnClient::refresh`] keeps the allocation (and, by re-permitting, the
//!   per-peer permissions) alive past their lifetimes.
//!
//! IPv4 only in v1 (matching the server-reflexive path); channel binding (the
//! lower-overhead alternative to Send/Data indications) is a follow-up. Behind
//! the `webrtc` feature.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use core::time::Duration;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use hmac::{Hmac, Mac};
use md5::{Digest, Md5};
use sha1::Sha1;
use str0m::{Candidate, Rtc};
use tokio::net::UdpSocket;

use g2g_core::{G2gError, HardwareError};

type HmacSha1 = Hmac<Sha1>;

/// STUN magic cookie (RFC 5389).
const MAGIC: u32 = 0x2112_A442;

// STUN/TURN message types (class + method, RFC 5389 §6, RFC 5766 §3).
const ALLOCATE_REQUEST: u16 = 0x0003;
const ALLOCATE_SUCCESS: u16 = 0x0103;
const REFRESH_REQUEST: u16 = 0x0004;
const CREATE_PERMISSION_REQUEST: u16 = 0x0008;
const SEND_INDICATION: u16 = 0x0016;
const DATA_INDICATION: u16 = 0x0017;

// Attribute types.
const ATTR_USERNAME: u16 = 0x0006;
const ATTR_MESSAGE_INTEGRITY: u16 = 0x0008;
const ATTR_ERROR_CODE: u16 = 0x0009;
const ATTR_LIFETIME: u16 = 0x000D;
const ATTR_XOR_PEER_ADDRESS: u16 = 0x0012;
const ATTR_DATA: u16 = 0x0013;
const ATTR_REALM: u16 = 0x0014;
const ATTR_NONCE: u16 = 0x0015;
const ATTR_XOR_RELAYED_ADDRESS: u16 = 0x0016;
const ATTR_REQUESTED_TRANSPORT: u16 = 0x0019;

/// Desired allocation lifetime requested from the server (seconds). The server
/// may shorten it; the actual lifetime comes back in the success response.
const DEFAULT_LIFETIME_SECS: u32 = 600;

/// IANA protocol number for UDP, the REQUESTED-TRANSPORT value.
const TRANSPORT_UDP: u8 = 17;

fn hw() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// An active TURN allocation plus the long-term credentials needed to keep
/// authenticating follow-up requests (CreatePermission / Refresh). Owns no
/// socket: the element passes its ICE `UdpSocket` to the async methods, the
/// same socket str0m drives, so the relay sees a single NAT binding.
#[derive(Debug)]
pub(crate) struct TurnClient {
    /// The TURN server's transport address (resolved `host:port`).
    server: SocketAddr,
    username: String,
    /// Long-term credential key: `MD5(username ":" realm ":" password)`.
    key: Vec<u8>,
    realm: String,
    nonce: String,
    /// Server-allocated relay address; the relayed ICE candidate's address.
    relay: SocketAddr,
    /// Peer IPs we have already issued a CreatePermission for this lifetime.
    permitted: BTreeSet<IpAddr>,
    /// Monotonic transaction-id counter (responses are matched by server source
    /// + message class, so uniqueness within the socket is all that is needed).
    txn_counter: u32,
}

impl TurnClient {
    /// The relay transport address; add this to str0m via `Candidate::relayed`.
    pub(crate) fn relay_addr(&self) -> SocketAddr {
        self.relay
    }

    /// The TURN server's transport address (where wrapped datagrams are sent).
    pub(crate) fn server_addr(&self) -> SocketAddr {
        self.server
    }

    /// True if `addr` is the TURN server (used to demux inbound UDP: traffic
    /// from the server is TURN-framed, everything else is a direct str0m path).
    pub(crate) fn is_server(&self, addr: SocketAddr) -> bool {
        addr == self.server
    }

    /// Do the TURN Allocate handshake on `socket` and return the live client.
    ///
    /// RFC 5766 §6.1: the first Allocate is sent without credentials and is
    /// answered with `401` carrying the `REALM` and `NONCE`; the retry adds
    /// USERNAME / REALM / NONCE / MESSAGE-INTEGRITY computed with the long-term
    /// key. Runs during element setup, before str0m owns the socket, so there is
    /// no read contention (same contract as the STUN srflx gather).
    pub(crate) async fn allocate(
        socket: &UdpSocket,
        server: SocketAddr,
        username: &str,
        password: &str,
    ) -> Result<Self, G2gError> {
        let mut counter = 1u32;

        // First Allocate: unauthenticated, expect 401 with realm + nonce.
        let txn = make_txn(socket, &mut counter);
        let mut msg = MessageBuilder::new(ALLOCATE_REQUEST, txn);
        msg.push_requested_transport();
        msg.push_lifetime(DEFAULT_LIFETIME_SECS);
        let first = round_trip(socket, server, msg.finish()).await?;

        let (realm, nonce) = match parse_error(&first) {
            // 401 Unauthorized carries the realm/nonce we must echo.
            Some((401, _)) | None => {
                let realm = find_str(&first, ATTR_REALM).ok_or_else(hw)?;
                let nonce = find_str(&first, ATTR_NONCE).ok_or_else(hw)?;
                (realm, nonce)
            }
            Some(_) => return Err(hw()),
        };

        let key = long_term_key(username, &realm, password);

        // Authenticated Allocate retry.
        let txn = make_txn(socket, &mut counter);
        let mut msg = MessageBuilder::new(ALLOCATE_REQUEST, txn);
        msg.push_requested_transport();
        msg.push_lifetime(DEFAULT_LIFETIME_SECS);
        msg.push_str_attr(ATTR_USERNAME, username);
        msg.push_str_attr(ATTR_REALM, &realm);
        msg.push_str_attr(ATTR_NONCE, &nonce);
        let bytes = msg.finish_with_integrity(&key);
        let resp = round_trip(socket, server, bytes).await?;

        if message_type(&resp) != ALLOCATE_SUCCESS {
            return Err(hw());
        }
        let relay = find_xor_addr(&resp, ATTR_XOR_RELAYED_ADDRESS, &txn).ok_or_else(hw)?;

        Ok(Self {
            server,
            username: username.into(),
            key,
            realm,
            nonce,
            relay,
            permitted: BTreeSet::new(),
            txn_counter: counter,
        })
    }

    /// Ensure the server will relay to/from `peer`'s IP, sending a
    /// CreatePermission request the first time we see that IP this lifetime.
    /// Fire-and-forget: ICE retransmits its connectivity checks, so a check
    /// dropped before the permission lands is re-sent and succeeds. Permissions
    /// are re-established after [`Self::refresh`] clears the set.
    pub(crate) async fn ensure_permission(
        &mut self,
        socket: &UdpSocket,
        peer: SocketAddr,
    ) -> Result<(), G2gError> {
        if !self.permitted.insert(peer.ip()) {
            return Ok(());
        }
        self.txn_counter = self.txn_counter.wrapping_add(1);
        let txn = txn_bytes(self.server, self.txn_counter);
        let mut msg = MessageBuilder::new(CREATE_PERMISSION_REQUEST, txn);
        msg.push_xor_peer_address(peer, &txn);
        msg.push_str_attr(ATTR_USERNAME, &self.username);
        msg.push_str_attr(ATTR_REALM, &self.realm);
        msg.push_str_attr(ATTR_NONCE, &self.nonce);
        let bytes = msg.finish_with_integrity(&self.key);
        socket.send_to(&bytes, self.server).await.map_err(|_| hw())?;
        Ok(())
    }

    /// Wrap `payload` (an str0m datagram bound for `peer`) in a Send indication
    /// for the TURN server to relay. Send indications are unauthenticated (RFC
    /// 5766 §10.1), so this is allocation-free of the credential dance and cheap
    /// enough for the per-packet hot path.
    pub(crate) fn wrap_send(&mut self, peer: SocketAddr, payload: &[u8]) -> Vec<u8> {
        self.txn_counter = self.txn_counter.wrapping_add(1);
        let txn = txn_bytes(self.server, self.txn_counter);
        let mut msg = MessageBuilder::new(SEND_INDICATION, txn);
        msg.push_xor_peer_address(peer, &txn);
        msg.push_bytes_attr(ATTR_DATA, payload);
        msg.finish()
    }

    /// If `msg` (received from the TURN server) is a Data indication, unwrap it
    /// to `(peer, payload)` to be fed back into str0m. Returns `None` for the
    /// server's control responses (CreatePermission / Refresh success), which
    /// the run loop discards.
    pub(crate) fn parse_data(&self, msg: &[u8]) -> Option<(SocketAddr, Vec<u8>)> {
        if message_type(msg) != DATA_INDICATION {
            return None;
        }
        let txn = txn_of(msg)?;
        let peer = find_xor_addr(msg, ATTR_XOR_PEER_ADDRESS, &txn)?;
        let data = find_bytes(msg, ATTR_DATA)?;
        Some((peer, data.to_vec()))
    }

    /// Refresh the allocation (RFC 5766 §7) and drop the permission set so the
    /// run loop re-issues CreatePermission lazily. Call well inside the
    /// permission lifetime (300 s) and the allocation lifetime. Fire-and-forget:
    /// the success response is consumed as ordinary server traffic in the loop.
    pub(crate) async fn refresh(&mut self, socket: &UdpSocket) -> Result<(), G2gError> {
        self.txn_counter = self.txn_counter.wrapping_add(1);
        let txn = txn_bytes(self.server, self.txn_counter);
        let mut msg = MessageBuilder::new(REFRESH_REQUEST, txn);
        msg.push_lifetime(DEFAULT_LIFETIME_SECS);
        msg.push_str_attr(ATTR_USERNAME, &self.username);
        msg.push_str_attr(ATTR_REALM, &self.realm);
        msg.push_str_attr(ATTR_NONCE, &self.nonce);
        let bytes = msg.finish_with_integrity(&self.key);
        socket.send_to(&bytes, self.server).await.map_err(|_| hw())?;
        self.permitted.clear();
        Ok(())
    }
}

/// Refresh well inside the 300 s permission lifetime so peer permissions never
/// lapse mid-session.
pub(crate) const REFRESH_INTERVAL: Duration = Duration::from_secs(240);

/// Resolve `server` (`host:port`), do the Allocate handshake on `socket`, and
/// add the resulting relay address to `rtc` as a relayed ICE candidate. Returns
/// the live client to drive the relay data plane, or `None` if the server is
/// unreachable / refuses the allocation, degrading gracefully to STUN/host (the
/// run continues, just without the relay candidate). `socket` must not yet be
/// owned by str0m's loop (the handshake reads replies directly).
pub(crate) async fn setup(
    rtc: &mut Rtc,
    socket: &UdpSocket,
    server: &str,
    username: &str,
    password: &str,
) -> Option<TurnClient> {
    let server_addr = tokio::net::lookup_host(server).await.ok()?.next()?;
    let client = TurnClient::allocate(socket, server_addr, username, password).await.ok()?;
    let local = socket.local_addr().ok()?;
    if let Ok(c) = Candidate::relayed(client.relay_addr(), local, "udp") {
        rtc.add_local_candidate(c);
    }
    Some(client)
}

/// `MD5(username ":" realm ":" password)`, the long-term credential key used as
/// the HMAC-SHA1 key for MESSAGE-INTEGRITY (RFC 5389 §15.4).
fn long_term_key(username: &str, realm: &str, password: &str) -> Vec<u8> {
    let mut h = Md5::new();
    h.update(username.as_bytes());
    h.update(b":");
    h.update(realm.as_bytes());
    h.update(b":");
    h.update(password.as_bytes());
    h.finalize().to_vec()
}

/// Send `bytes` to the TURN server and read one reply, with a short timeout.
/// Used only during the setup Allocate handshake.
async fn round_trip(
    socket: &UdpSocket,
    server: SocketAddr,
    bytes: Vec<u8>,
) -> Result<Vec<u8>, G2gError> {
    socket.send_to(&bytes, server).await.map_err(|_| hw())?;
    let mut buf = [0u8; 1500];
    let (n, _) = tokio::time::timeout(Duration::from_secs(3), socket.recv_from(&mut buf))
        .await
        .map_err(|_| hw())?
        .map_err(|_| hw())?;
    Ok(buf[..n].to_vec())
}

/// Derive a 12-byte transaction id from the socket's local port and a counter.
fn make_txn(socket: &UdpSocket, counter: &mut u32) -> [u8; 12] {
    let port = socket.local_addr().map(|a| a.port()).unwrap_or(0);
    *counter = counter.wrapping_add(1);
    let mut txn = [0u8; 12];
    txn[0..2].copy_from_slice(&port.to_be_bytes());
    txn[2..6].copy_from_slice(&MAGIC.to_be_bytes());
    txn[6..10].copy_from_slice(&counter.to_be_bytes());
    txn
}

/// Transaction id seeded from the server port + a counter (for in-loop requests
/// where re-querying the socket each time is needless).
fn txn_bytes(server: SocketAddr, counter: u32) -> [u8; 12] {
    let mut txn = [0u8; 12];
    txn[0..2].copy_from_slice(&server.port().to_be_bytes());
    txn[2..6].copy_from_slice(&MAGIC.to_be_bytes());
    txn[6..10].copy_from_slice(&counter.to_be_bytes());
    txn
}

/// Builds a STUN/TURN message: 20-byte header then 4-byte-aligned attributes.
struct MessageBuilder {
    buf: Vec<u8>,
}

impl MessageBuilder {
    fn new(msg_type: u16, txn: [u8; 12]) -> Self {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(&msg_type.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes()); // length, patched in finish()
        buf.extend_from_slice(&MAGIC.to_be_bytes());
        buf.extend_from_slice(&txn);
        Self { buf }
    }

    /// Append one attribute, padding the value to a 4-byte boundary.
    fn push_attr(&mut self, atype: u16, value: &[u8]) {
        self.buf.extend_from_slice(&atype.to_be_bytes());
        self.buf.extend_from_slice(&(value.len() as u16).to_be_bytes());
        self.buf.extend_from_slice(value);
        let pad = (4 - (value.len() % 4)) % 4;
        self.buf.extend(core::iter::repeat(0u8).take(pad));
    }

    fn push_str_attr(&mut self, atype: u16, s: &str) {
        self.push_attr(atype, s.as_bytes());
    }

    fn push_bytes_attr(&mut self, atype: u16, b: &[u8]) {
        self.push_attr(atype, b);
    }

    fn push_requested_transport(&mut self) {
        self.push_attr(ATTR_REQUESTED_TRANSPORT, &[TRANSPORT_UDP, 0, 0, 0]);
    }

    fn push_lifetime(&mut self, secs: u32) {
        self.push_attr(ATTR_LIFETIME, &secs.to_be_bytes());
    }

    fn push_xor_peer_address(&mut self, addr: SocketAddr, txn: &[u8; 12]) {
        self.push_attr(ATTR_XOR_PEER_ADDRESS, &encode_xor_addr(addr, txn));
    }

    /// Patch the header length and return the bytes.
    fn finish(mut self) -> Vec<u8> {
        let attrs_len = (self.buf.len() - 20) as u16;
        self.buf[2..4].copy_from_slice(&attrs_len.to_be_bytes());
        self.buf
    }

    /// Append MESSAGE-INTEGRITY (RFC 5389 §15.4): HMAC-SHA1 of the message over a
    /// header whose length already counts the 24-byte MI attribute, then the
    /// attribute itself.
    fn finish_with_integrity(mut self, key: &[u8]) -> Vec<u8> {
        // Length must include the MESSAGE-INTEGRITY attribute (4 + 20) for the
        // HMAC input, even though it is not yet appended.
        let len_with_mi = (self.buf.len() - 20 + 24) as u16;
        self.buf[2..4].copy_from_slice(&len_with_mi.to_be_bytes());

        let mut mac = HmacSha1::new_from_slice(key).expect("HMAC accepts any key length");
        mac.update(&self.buf);
        let digest = mac.finalize().into_bytes();
        self.push_attr(ATTR_MESSAGE_INTEGRITY, &digest);
        // push_attr already advanced the buffer; the header length set above is
        // exactly correct (MI adds 24 bytes, no padding).
        self.buf
    }
}

/// Encode a socket address as an XOR-(PEER|RELAYED|MAPPED)-ADDRESS value (RFC
/// 5389 §15.2). IPv4 only: port XORed with the high cookie half, address with
/// the full cookie.
fn encode_xor_addr(addr: SocketAddr, _txn: &[u8; 12]) -> Vec<u8> {
    let SocketAddr::V4(v4) = addr else {
        // IPv6 relayed addresses are a v1 limitation; encode loopback so the
        // caller still produces a well-formed (if unreachable) attribute.
        return Vec::from([0u8, 0x01, 0, 0, 0, 0, 0, 0]);
    };
    let magic = MAGIC.to_be_bytes();
    let xport = v4.port() ^ (MAGIC >> 16) as u16;
    let ip = v4.ip().octets();
    let mut out = Vec::with_capacity(8);
    out.push(0); // reserved
    out.push(0x01); // family IPv4
    out.extend_from_slice(&xport.to_be_bytes());
    for k in 0..4 {
        out.push(ip[k] ^ magic[k]);
    }
    out
}

fn message_type(msg: &[u8]) -> u16 {
    if msg.len() < 2 {
        return 0;
    }
    u16::from_be_bytes([msg[0], msg[1]])
}

fn txn_of(msg: &[u8]) -> Option<[u8; 12]> {
    if msg.len() < 20 {
        return None;
    }
    let mut txn = [0u8; 12];
    txn.copy_from_slice(&msg[8..20]);
    Some(txn)
}

fn find_str(msg: &[u8], atype: u16) -> Option<String> {
    find_bytes(msg, atype).and_then(|b| core::str::from_utf8(b).ok().map(String::from))
}

fn find_bytes(msg: &[u8], atype: u16) -> Option<&[u8]> {
    if msg.len() < 20 {
        return None;
    }
    let mut i = 20;
    while i + 4 <= msg.len() {
        let t = u16::from_be_bytes([msg[i], msg[i + 1]]);
        let alen = u16::from_be_bytes([msg[i + 2], msg[i + 3]]) as usize;
        let start = i + 4;
        if start + alen > msg.len() {
            break;
        }
        if t == atype {
            return Some(&msg[start..start + alen]);
        }
        i = start + alen + ((4 - (alen % 4)) % 4);
    }
    None
}

/// Find and decode an XOR-address attribute (peer / relayed / mapped). IPv4.
fn find_xor_addr(msg: &[u8], atype: u16, _txn: &[u8; 12]) -> Option<SocketAddr> {
    let val = find_bytes(msg, atype)?;
    if val.len() < 8 || val[1] != 0x01 {
        return None;
    }
    let magic = MAGIC.to_be_bytes();
    let port = u16::from_be_bytes([val[2], val[3]]) ^ (MAGIC >> 16) as u16;
    let mut ip = [val[4], val[5], val[6], val[7]];
    for k in 0..4 {
        ip[k] ^= magic[k];
    }
    Some(SocketAddr::from((Ipv4Addr::from(ip), port)))
}

/// Decode an ERROR-CODE attribute (RFC 5389 §15.6) as `(code, reason)`. Returns
/// `None` when the message carries no ERROR-CODE (e.g. a success response).
fn parse_error(msg: &[u8]) -> Option<(u16, String)> {
    let val = find_bytes(msg, ATTR_ERROR_CODE)?;
    if val.len() < 4 {
        return None;
    }
    // Bytes 0..2 reserved; byte 2 = class (hundreds), byte 3 = number.
    let code = (val[2] as u16) * 100 + (val[3] as u16);
    let reason = core::str::from_utf8(&val[4..]).unwrap_or("").into();
    Some((code, reason))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_term_key_matches_rfc_md5() {
        // MD5("user:realm:pass") computed independently.
        let key = long_term_key("user", "realm", "pass");
        assert_eq!(key.len(), 16);
        let expected = {
            let mut h = Md5::new();
            h.update(b"user:realm:pass");
            h.finalize().to_vec()
        };
        assert_eq!(key, expected);
    }

    #[test]
    fn xor_addr_round_trips() {
        let addr = SocketAddr::from((Ipv4Addr::new(203, 0, 113, 7), 51234));
        let txn = [0u8; 12];
        let encoded = encode_xor_addr(addr, &txn);
        // Wrap it in a minimal message with one XOR-PEER-ADDRESS attribute.
        let mut msg = MessageBuilder::new(DATA_INDICATION, txn);
        msg.push_attr(ATTR_XOR_PEER_ADDRESS, &encoded);
        let bytes = msg.finish();
        let decoded = find_xor_addr(&bytes, ATTR_XOR_PEER_ADDRESS, &txn).expect("decodes");
        assert_eq!(decoded, addr);
    }

    #[test]
    fn message_integrity_is_verifiable() {
        // Build an authenticated CreatePermission and re-derive the HMAC the way
        // a server would, confirming the length field and HMAC input agree.
        let key = long_term_key("u", "r", "p");
        let txn = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let peer = SocketAddr::from((Ipv4Addr::new(10, 0, 0, 9), 4000));
        let mut msg = MessageBuilder::new(CREATE_PERMISSION_REQUEST, txn);
        msg.push_xor_peer_address(peer, &txn);
        msg.push_str_attr(ATTR_USERNAME, "u");
        msg.push_str_attr(ATTR_REALM, "r");
        msg.push_str_attr(ATTR_NONCE, "n");
        let bytes = msg.finish_with_integrity(&key);

        // Locate the MESSAGE-INTEGRITY attribute and recompute over the prefix.
        let mut mi_offset = None;
        for_each_attr_with_offset(&bytes, |t, off| {
            if t == ATTR_MESSAGE_INTEGRITY {
                mi_offset = Some(off);
            }
        });
        let off = mi_offset.expect("has MI");
        // The HMAC input is the header+attrs up to the MI attribute, with the
        // header length already counting MI; that is exactly the on-wire length
        // here, so recompute over bytes[..off-4] with the stored length.
        let mut prefix = bytes[..off - 4].to_vec();
        let len_with_mi = (bytes.len() - 20) as u16;
        prefix[2..4].copy_from_slice(&len_with_mi.to_be_bytes());
        let mut mac = HmacSha1::new_from_slice(&key).unwrap();
        mac.update(&prefix);
        let expected = mac.finalize().into_bytes();
        assert_eq!(&bytes[off..off + 20], &expected[..]);
    }

    #[test]
    fn data_indication_unwraps_to_peer_and_payload() {
        let txn = [9u8; 12];
        let peer = SocketAddr::from((Ipv4Addr::new(198, 51, 100, 2), 7000));
        let payload = b"hello-relayed-rtp";
        let mut msg = MessageBuilder::new(DATA_INDICATION, txn);
        msg.push_xor_peer_address(peer, &txn);
        msg.push_bytes_attr(ATTR_DATA, payload);
        let bytes = msg.finish();

        let client = TurnClient {
            server: SocketAddr::from((Ipv4Addr::LOCALHOST, 3478)),
            username: "u".into(),
            key: Vec::new(),
            realm: "r".into(),
            nonce: "n".into(),
            relay: SocketAddr::from((Ipv4Addr::LOCALHOST, 5000)),
            permitted: BTreeSet::new(),
            txn_counter: 0,
        };
        let (got_peer, got_data) = client.parse_data(&bytes).expect("data indication");
        assert_eq!(got_peer, peer);
        assert_eq!(got_data, payload);
        // A non-Data message yields None.
        let other = MessageBuilder::new(ALLOCATE_SUCCESS, txn).finish();
        assert!(client.parse_data(&other).is_none());
    }

    #[test]
    fn parses_error_code() {
        let txn = [0u8; 12];
        let mut msg = MessageBuilder::new(0x0113, txn); // Allocate error
        msg.push_attr(ATTR_ERROR_CODE, &[0, 0, 4, 1, b'x']); // 401
        let bytes = msg.finish();
        assert_eq!(parse_error(&bytes).map(|(c, _)| c), Some(401));
    }

    /// Test-only attribute walker that also yields the byte offset of each
    /// attribute value (used to locate MESSAGE-INTEGRITY for verification).
    fn for_each_attr_with_offset(msg: &[u8], mut f: impl FnMut(u16, usize)) {
        let mut i = 20;
        while i + 4 <= msg.len() {
            let atype = u16::from_be_bytes([msg[i], msg[i + 1]]);
            let alen = u16::from_be_bytes([msg[i + 2], msg[i + 3]]) as usize;
            let start = i + 4;
            if start + alen > msg.len() {
                break;
            }
            f(atype, start);
            i = start + alen + ((4 - (alen % 4)) % 4);
        }
    }
}

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
//!   those through the relay: [`TurnClient::wrap_send`] frames the payload for
//!   the TURN server (ChannelData or a Send indication, below) and sends it to
//!   the server. Direct (host / server-reflexive) transmits are unchanged.
//! - Inbound UDP from the TURN server is a ChannelData frame or a Data
//!   indication; [`TurnClient::parse_data`] unwraps either to `(peer, payload)`
//!   and the element feeds str0m an `Input::Receive` with `source = peer`,
//!   `destination = relay_addr`. The server only relays peer traffic once a
//!   permission for that peer's IP exists, so the run loop calls
//!   [`TurnClient::ensure_permission`] for each new peer before relaying to it.
//! - [`TurnClient::refresh`] keeps the allocation, the channel bindings, and
//!   the per-peer permissions alive past their lifetimes.
//!
//! Channel binding (RFC 5766 §11) upgrades the per-peer data plane: for each
//! new peer the client sends a ChannelBind (which also installs the IP
//! permission server-side, so no separate CreatePermission is needed) and keeps
//! using Send indications until the bind success lands; from then on datagrams
//! ride 4-byte-header ChannelData frames instead of 36-byte indications, both
//! directions. A `438 Stale Nonce` on any authenticated request updates the
//! stored nonce from the error response and un-caches the affected state, so
//! the existing lazy paths retry with fresh credentials.
//!
//! IPv4 only in v1 (matching the server-reflexive path). Behind the `webrtc`
//! feature.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use core::time::Duration;
use std::net::{Ipv4Addr, SocketAddr};

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
const CHANNEL_BIND_REQUEST: u16 = 0x0009;
const CHANNEL_BIND_SUCCESS: u16 = 0x0109;
const SEND_INDICATION: u16 = 0x0016;
const DATA_INDICATION: u16 = 0x0017;
/// Error-class bit pattern (RFC 5389 §6): request type | 0x0110.
const ERROR_CLASS: u16 = 0x0110;

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
const ATTR_CHANNEL_NUMBER: u16 = 0x000C;

/// Client-assigned channel numbers (RFC 5766 §11: 0x4000-0x7FFF).
const CHANNEL_MIN: u16 = 0x4000;
const CHANNEL_MAX: u16 = 0x7FFF;
/// 438 Stale Nonce (RFC 5389 §15.6).
const STALE_NONCE: u16 = 438;

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
    /// Per-peer channel state: assigned number + whether the bind confirmed
    /// (ChannelData may only be sent after the ChannelBind success, §11.3).
    channels: BTreeMap<SocketAddr, Channel>,
    /// Next channel number to assign (wraps within the client range).
    next_channel: u16,
    /// In-flight authenticated requests by transaction id, so their success /
    /// 438-stale-nonce responses can be matched in the data plane.
    pending: BTreeMap<[u8; 12], Pending>,
    /// Monotonic transaction-id counter (responses are matched by server source
    /// + message class, so uniqueness within the socket is all that is needed).
    txn_counter: u32,
}

/// One peer's channel binding.
#[derive(Debug, Clone, Copy)]
struct Channel {
    number: u16,
    bound: bool,
}

/// What an in-flight authenticated request was for, so its response can update
/// (success) or revert (438) the matching client state.
#[derive(Debug, Clone, Copy)]
enum Pending {
    Bind(SocketAddr),
    Refresh,
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
        let first = round_trip(socket, server, msg.finish(), &txn).await?;

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
        let resp = round_trip(socket, server, bytes, &txn).await?;

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
            channels: BTreeMap::new(),
            next_channel: CHANNEL_MIN,
            pending: BTreeMap::new(),
            txn_counter: counter,
        })
    }

    /// Append the long-term-credential attributes + MESSAGE-INTEGRITY and record
    /// the request as pending under `txn`.
    fn finish_authenticated(
        &mut self,
        mut msg: MessageBuilder,
        txn: [u8; 12],
        p: Pending,
    ) -> Vec<u8> {
        msg.push_str_attr(ATTR_USERNAME, &self.username);
        msg.push_str_attr(ATTR_REALM, &self.realm);
        msg.push_str_attr(ATTR_NONCE, &self.nonce);
        self.pending.insert(txn, p);
        msg.finish_with_integrity(&self.key)
    }

    /// Send a ChannelBind for `peer` on channel `number` (installs / refreshes
    /// the peer-IP permission server-side too, RFC 5766 §11.2).
    async fn send_channel_bind(
        &mut self,
        socket: &UdpSocket,
        peer: SocketAddr,
        number: u16,
    ) -> Result<(), G2gError> {
        self.txn_counter = self.txn_counter.wrapping_add(1);
        let txn = txn_bytes(self.server, self.txn_counter);
        let mut msg = MessageBuilder::new(CHANNEL_BIND_REQUEST, txn);
        msg.push_attr(
            ATTR_CHANNEL_NUMBER,
            &[(number >> 8) as u8, number as u8, 0, 0],
        );
        msg.push_xor_peer_address(peer, &txn);
        let bytes = self.finish_authenticated(msg, txn, Pending::Bind(peer));
        socket
            .send_to(&bytes, self.server)
            .await
            .map_err(|_| hw())?;
        Ok(())
    }

    /// Ensure the server will relay to/from `peer`: the first transmit to a new
    /// peer sends a ChannelBind (RFC 5766 §11), which both installs the peer-IP
    /// permission and starts the channel; [`Self::wrap_send`] uses Send
    /// indications until the bind success lands. Fire-and-forget: ICE
    /// retransmits its connectivity checks, so a check dropped in the one-RTT
    /// window before the server processes the bind is re-sent and succeeds.
    pub(crate) async fn ensure_permission(
        &mut self,
        socket: &UdpSocket,
        peer: SocketAddr,
    ) -> Result<(), G2gError> {
        if self.channels.contains_key(&peer) {
            return Ok(());
        }
        let number = self.next_channel;
        self.next_channel = if self.next_channel >= CHANNEL_MAX {
            CHANNEL_MIN
        } else {
            self.next_channel + 1
        };
        // Insert only after the request is actually sent, so a transient send
        // failure leaves the peer unknown and the next transmit retries.
        self.send_channel_bind(socket, peer, number).await?;
        self.channels.insert(
            peer,
            Channel {
                number,
                bound: false,
            },
        );
        Ok(())
    }

    /// Wrap `payload` (an str0m datagram bound for `peer`) for the TURN server
    /// to relay: a 4-byte-header ChannelData frame once the peer's channel is
    /// bound, else a Send indication (unauthenticated, RFC 5766 §10.1). Both are
    /// cheap enough for the per-packet hot path; ChannelData saves the 36-byte
    /// indication envelope in the steady state.
    pub(crate) fn wrap_send(&mut self, peer: SocketAddr, payload: &[u8]) -> Vec<u8> {
        if let Some(ch) = self.channels.get(&peer) {
            if ch.bound {
                let mut out = Vec::with_capacity(4 + payload.len());
                out.extend_from_slice(&ch.number.to_be_bytes());
                out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
                out.extend_from_slice(payload);
                return out;
            }
        }
        self.txn_counter = self.txn_counter.wrapping_add(1);
        let txn = txn_bytes(self.server, self.txn_counter);
        let mut msg = MessageBuilder::new(SEND_INDICATION, txn);
        msg.push_xor_peer_address(peer, &txn);
        msg.push_bytes_attr(ATTR_DATA, payload);
        msg.finish()
    }

    /// Unwrap one datagram received from the TURN server. Relayed peer traffic
    /// (a ChannelData frame or a Data indication) returns `(peer, payload)` to
    /// be fed back into str0m. Control responses return `None` after updating
    /// client state: a ChannelBind success flips that peer to ChannelData
    /// framing, and a `438 Stale Nonce` on any pending request adopts the error
    /// response's fresh nonce and un-caches the affected state so the lazy
    /// paths retry with it.
    pub(crate) fn parse_data(&mut self, msg: &[u8]) -> Option<(SocketAddr, Vec<u8>)> {
        // ChannelData: first two bits 0b01 (STUN messages start 0b00, §11.4).
        if msg.len() >= 4 && msg[0] & 0xC0 == 0x40 {
            let number = u16::from_be_bytes([msg[0], msg[1]]);
            let len = u16::from_be_bytes([msg[2], msg[3]]) as usize;
            let end = 4usize.checked_add(len)?;
            if end > msg.len() {
                return None;
            }
            let peer = self
                .channels
                .iter()
                .find(|(_, c)| c.number == number)
                .map(|(p, _)| *p)?;
            return Some((peer, msg[4..end].to_vec()));
        }
        let mtype = message_type(msg);
        if mtype == DATA_INDICATION {
            let txn = txn_of(msg)?;
            let peer = find_xor_addr(msg, ATTR_XOR_PEER_ADDRESS, &txn)?;
            let data = find_bytes(msg, ATTR_DATA)?;
            return Some((peer, data.to_vec()));
        }
        // A response to one of our authenticated requests.
        let txn = txn_of(msg)?;
        let pending = self.pending.remove(&txn)?;
        if mtype == CHANNEL_BIND_SUCCESS {
            if let Pending::Bind(peer) = pending {
                if let Some(ch) = self.channels.get_mut(&peer) {
                    ch.bound = true;
                }
            }
        } else if mtype & ERROR_CLASS == ERROR_CLASS {
            if let Some((STALE_NONCE, _)) = parse_error(msg) {
                if let Some(nonce) = find_str(msg, ATTR_NONCE) {
                    self.nonce = nonce;
                }
            }
            // Un-cache the failed state (stale nonce or otherwise), so the next
            // transmit / refresh tick retries the request with current fields.
            match pending {
                Pending::Bind(peer) => {
                    self.channels.remove(&peer);
                }
                Pending::Refresh => {}
            }
        }
        None
    }

    /// Refresh the allocation (RFC 5766 §7) and re-send each peer's ChannelBind,
    /// which refreshes both the channel binding (600 s lifetime) and its peer
    /// permission (300 s). Call well inside the permission lifetime.
    /// Fire-and-forget: the responses are consumed by [`Self::parse_data`] as
    /// ordinary server traffic in the loop.
    pub(crate) async fn refresh(&mut self, socket: &UdpSocket) -> Result<(), G2gError> {
        self.txn_counter = self.txn_counter.wrapping_add(1);
        let txn = txn_bytes(self.server, self.txn_counter);
        let msg = {
            let mut m = MessageBuilder::new(REFRESH_REQUEST, txn);
            m.push_lifetime(DEFAULT_LIFETIME_SECS);
            m
        };
        let bytes = self.finish_authenticated(msg, txn, Pending::Refresh);
        socket
            .send_to(&bytes, self.server)
            .await
            .map_err(|_| hw())?;
        let rebind: Vec<(SocketAddr, u16)> =
            self.channels.iter().map(|(p, c)| (*p, c.number)).collect();
        for (peer, number) in rebind {
            self.send_channel_bind(socket, peer, number).await?;
        }
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
    let client = TurnClient::allocate(socket, server_addr, username, password)
        .await
        .ok()?;
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

/// Send `bytes` to the TURN server and read the reply matching `txn`, with a
/// short overall timeout. Because the offer is POSTed before this handshake runs
/// (trickle ICE), the peer's ICE connectivity checks also land on the socket, so
/// skip any datagram whose transaction id is not ours (matched on bytes 8..20).
async fn round_trip(
    socket: &UdpSocket,
    server: SocketAddr,
    bytes: Vec<u8>,
    txn: &[u8; 12],
) -> Result<Vec<u8>, G2gError> {
    socket.send_to(&bytes, server).await.map_err(|_| hw())?;
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    let mut buf = [0u8; 1500];
    loop {
        let remaining = deadline
            .checked_duration_since(std::time::Instant::now())
            .ok_or_else(hw)?;
        let (n, _) = tokio::time::timeout(remaining, socket.recv_from(&mut buf))
            .await
            .map_err(|_| hw())?
            .map_err(|_| hw())?;
        if n >= 20 && &buf[8..20] == txn {
            return Ok(buf[..n].to_vec());
        }
    }
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
        self.buf
            .extend_from_slice(&(value.len() as u16).to_be_bytes());
        self.buf.extend_from_slice(value);
        let pad = (4 - (value.len() % 4)) % 4;
        self.buf.extend(core::iter::repeat_n(0u8, pad));
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
        // Build an authenticated ChannelBind and re-derive the HMAC the way
        // a server would, confirming the length field and HMAC input agree.
        let key = long_term_key("u", "r", "p");
        let txn = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let peer = SocketAddr::from((Ipv4Addr::new(10, 0, 0, 9), 4000));
        let mut msg = MessageBuilder::new(CHANNEL_BIND_REQUEST, txn);
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

        let mut client = test_client();
        let (got_peer, got_data) = client.parse_data(&bytes).expect("data indication");
        assert_eq!(got_peer, peer);
        assert_eq!(got_data, payload);
        // A non-Data message that matches no pending request yields None.
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

    fn test_client() -> TurnClient {
        TurnClient {
            server: SocketAddr::from((Ipv4Addr::LOCALHOST, 3478)),
            username: "u".into(),
            key: long_term_key("u", "r", "p"),
            realm: "r".into(),
            nonce: "n1".into(),
            relay: SocketAddr::from((Ipv4Addr::LOCALHOST, 50000)),
            channels: BTreeMap::new(),
            next_channel: CHANNEL_MIN,
            pending: BTreeMap::new(),
            txn_counter: 0,
        }
    }

    fn peer() -> SocketAddr {
        SocketAddr::from((Ipv4Addr::new(10, 0, 0, 9), 4000))
    }

    /// A bare response of `msg_type` matching `txn` (success shapes carry no
    /// attributes the client reads).
    fn response(msg_type: u16, txn: [u8; 12]) -> Vec<u8> {
        MessageBuilder::new(msg_type, txn).finish()
    }

    #[test]
    fn wrap_send_upgrades_to_channel_data_after_bind_success() {
        let mut c = test_client();
        c.channels.insert(
            peer(),
            Channel {
                number: CHANNEL_MIN,
                bound: false,
            },
        );
        let txn = [9u8; 12];
        c.pending.insert(txn, Pending::Bind(peer()));

        // Unbound: a Send indication (STUN-framed, starts 0b00).
        let ind = c.wrap_send(peer(), b"payload");
        assert_eq!(message_type(&ind), SEND_INDICATION);

        // Bind success flips the peer to ChannelData framing.
        assert!(c.parse_data(&response(CHANNEL_BIND_SUCCESS, txn)).is_none());
        let framed = c.wrap_send(peer(), b"payload");
        assert_eq!(&framed[..2], &CHANNEL_MIN.to_be_bytes());
        assert_eq!(&framed[2..4], &7u16.to_be_bytes());
        assert_eq!(&framed[4..], b"payload");
    }

    #[test]
    fn parse_data_unwraps_channel_data_frames() {
        let mut c = test_client();
        c.channels.insert(
            peer(),
            Channel {
                number: 0x4005,
                bound: true,
            },
        );
        let mut frame = Vec::from(0x4005u16.to_be_bytes());
        frame.extend_from_slice(&5u16.to_be_bytes());
        frame.extend_from_slice(b"hello");
        assert_eq!(c.parse_data(&frame), Some((peer(), b"hello".to_vec())));
        // Unknown channel number: dropped.
        frame[0..2].copy_from_slice(&0x4009u16.to_be_bytes());
        assert_eq!(c.parse_data(&frame), None);
        // Truncated length: dropped, not panicked (network input).
        let short = [0x40u8, 0x05, 0x00, 0xFF, b'x'];
        assert_eq!(c.parse_data(&short), None);
    }

    #[test]
    fn stale_nonce_adopts_new_nonce_and_reverts_bind() {
        let mut c = test_client();
        c.channels.insert(
            peer(),
            Channel {
                number: CHANNEL_MIN,
                bound: false,
            },
        );
        let txn = [7u8; 12];
        c.pending.insert(txn, Pending::Bind(peer()));

        // 438 error response carrying the fresh nonce.
        let mut msg = MessageBuilder::new(CHANNEL_BIND_REQUEST | ERROR_CLASS, txn);
        msg.push_attr(ATTR_ERROR_CODE, &[0, 0, 4, 38]);
        msg.push_str_attr(ATTR_NONCE, "n2");
        assert!(c.parse_data(&msg.finish()).is_none());

        assert_eq!(c.nonce, "n2", "stale nonce adopted from the error response");
        assert!(
            !c.channels.contains_key(&peer()),
            "bind state reverted so the next transmit retries with the new nonce"
        );
        assert!(c.pending.is_empty());
    }

    /// Interop against a real TURN server (coturn):
    /// `docker run -d --network host coturn/coturn -n --no-tls --no-dtls
    ///  --listening-port=3478 --fingerprint --lt-cred-mech --user=g2g:g2gpass
    ///  --realm=g2g --no-multicast-peers`, then
    /// `G2G_TURN_SERVER=127.0.0.1:3478 G2G_TURN_USER=g2g G2G_TURN_PASS=g2gpass`.
    /// Allocates, channel-binds a local peer socket, and round-trips payloads
    /// both ways through the relay over ChannelData framing.
    #[tokio::test]
    #[ignore = "needs a reachable TURN server (G2G_TURN_SERVER/_USER/_PASS)"]
    async fn channel_binding_relays_both_ways_against_real_server() {
        let (Ok(server), Ok(user), Ok(pass)) = (
            std::env::var("G2G_TURN_SERVER"),
            std::env::var("G2G_TURN_USER"),
            std::env::var("G2G_TURN_PASS"),
        ) else {
            std::eprintln!("skipping: set G2G_TURN_SERVER, G2G_TURN_USER, G2G_TURN_PASS");
            return;
        };
        let server: SocketAddr = tokio::net::lookup_host(&server)
            .await
            .expect("resolve server")
            .next()
            .expect("server addr");

        let host_ip = crate::webrtc_util::select_host_ip();
        let client_sock = UdpSocket::bind((host_ip, 0)).await.expect("bind client");
        let peer_sock = UdpSocket::bind((host_ip, 0)).await.expect("bind peer");
        let peer_addr = peer_sock.local_addr().expect("peer addr");

        let mut c = TurnClient::allocate(&client_sock, server, &user, &pass)
            .await
            .expect("allocate against the real server");
        let relay = c.relay_addr();

        // Bind a channel to the peer and wait for the success response.
        c.ensure_permission(&client_sock, peer_addr)
            .await
            .expect("channel bind sent");
        let mut buf = [0u8; 1500];
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !c.channels.get(&peer_addr).is_some_and(|ch| ch.bound) {
            let remaining = deadline
                .checked_duration_since(std::time::Instant::now())
                .expect("bind confirmed before timeout");
            let (n, _) = tokio::time::timeout(remaining, client_sock.recv_from(&mut buf))
                .await
                .expect("bind response arrives")
                .expect("recv");
            let _ = c.parse_data(&buf[..n]);
        }

        // Client -> peer through the relay: the peer sees the raw payload from
        // the relay address.
        let framed = c.wrap_send(peer_addr, b"to-peer");
        assert_eq!(
            &framed[..2],
            &CHANNEL_MIN.to_be_bytes(),
            "ChannelData framing"
        );
        client_sock
            .send_to(&framed, c.server_addr())
            .await
            .expect("send");
        let (n, from) = tokio::time::timeout(Duration::from_secs(3), peer_sock.recv_from(&mut buf))
            .await
            .expect("relayed datagram arrives")
            .expect("recv");
        assert_eq!(&buf[..n], b"to-peer");
        assert_eq!(from, relay, "peer sees the relay as the sender");

        // Peer -> client: arrives as a ChannelData frame that unwraps to the
        // peer + payload.
        peer_sock.send_to(b"to-client", relay).await.expect("send");
        let unwrapped = loop {
            let (n, _) =
                tokio::time::timeout(Duration::from_secs(3), client_sock.recv_from(&mut buf))
                    .await
                    .expect("relayed frame arrives")
                    .expect("recv");
            assert_eq!(buf[0] & 0xC0, 0x40, "server relays over the bound channel");
            if let Some(v) = c.parse_data(&buf[..n]) {
                break v;
            }
        };
        assert_eq!(unwrapped, (peer_addr, b"to-client".to_vec()));

        // Refresh re-issues the bind (and the allocation) without erroring.
        c.refresh(&client_sock).await.expect("refresh");
    }
}

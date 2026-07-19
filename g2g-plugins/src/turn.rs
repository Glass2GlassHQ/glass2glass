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
//! Address-family agnostic (M718): the XOR address codec handles IPv4 and
//! IPv6, and a v6-bound client requests a v6 relayed address (RFC 6156).
//! Behind the `webrtc` feature.

use alloc::collections::BTreeMap;
use alloc::format;
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
/// RFC 6156: request an IPv6 relayed address for a v6 client.
const ATTR_REQUESTED_ADDRESS_FAMILY: u16 = 0x0017;

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
        // A v6 client asks for a v6 relayed address (RFC 6156); v4 keeps the
        // default family (the attribute is omitted).
        let want_v6 = socket.local_addr().map(|a| a.is_ipv6()).unwrap_or(false);

        // First Allocate: unauthenticated, expect 401 with realm + nonce.
        let txn = make_txn(socket, &mut counter);
        let mut msg = MessageBuilder::new(ALLOCATE_REQUEST, txn);
        msg.push_requested_transport();
        if want_v6 {
            msg.push_attr(ATTR_REQUESTED_ADDRESS_FAMILY, &[0x02, 0, 0, 0]);
        }
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
        if want_v6 {
            msg.push_attr(ATTR_REQUESTED_ADDRESS_FAMILY, &[0x02, 0, 0, 0]);
        }
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

/// Resolve `server`, do the Allocate handshake on `socket`, and add the
/// resulting relay address to `rtc` as a relayed ICE candidate. Returns the
/// live client to drive the relay data plane, or `None` if the server is
/// unreachable / refuses the allocation, degrading gracefully to STUN/host (the
/// run continues, just without the relay candidate). `socket` must not yet be
/// owned by str0m's loop (the handshake reads replies directly).
///
/// `server` is a bare `host:port` (UDP, the default) or an RFC 7065-style URI:
/// `turn:host[:port][?transport=udp|tcp]` or `turns:host[:port]` (TLS, always
/// over TCP). For the stream transports a local bridge task (below) tunnels the
/// client's datagrams over one TCP / TLS connection to the server, so the
/// `TurnClient` and every element run loop stay transport-agnostic; the
/// allocation itself still requests UDP relaying toward peers (RFC 5766 §2.1,
/// not RFC 6062 TCP allocations).
pub(crate) async fn setup(
    rtc: &mut Rtc,
    socket: &UdpSocket,
    server: &str,
    username: &str,
    password: &str,
) -> Option<TurnClient> {
    let local_ip = socket.local_addr().ok()?.ip();
    let server_addr = resolve_transport(server, local_ip).await?;
    let client = TurnClient::allocate(socket, server_addr, username, password)
        .await
        .ok()?;
    let local = socket.local_addr().ok()?;
    if let Ok(c) = Candidate::relayed(client.relay_addr(), local, "udp") {
        rtc.add_local_candidate(c);
    }
    Some(client)
}

/// The TURN client-to-server leg's transport (RFC 7065 URI schemes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnTransport {
    Udp,
    Tcp,
    Tls,
}

/// A set of TURN allocations, one per configured server (M719): each
/// contributes its own relayed candidate, and the data plane routes by which
/// relay / server address a transmit or datagram matches. The `Option`-like
/// surface keeps the element run loops one-liner simple.
#[derive(Debug, Default)]
pub(crate) struct TurnSet(Vec<TurnClient>);

impl TurnSet {
    pub(crate) fn empty() -> Self {
        Self(Vec::new())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Allocate on every server in the comma-separated `servers` list (each a
    /// `host:port`, RFC 7065 `turn:` / `turns:` URI, or GStreamer-style
    /// `turn://user:pass@host:port` with embedded credentials; entries without
    /// credentials use the element-level `user` / `pass`). A server that fails
    /// to allocate is skipped, so one dead relay never blocks the rest.
    pub(crate) async fn setup(
        rtc: &mut Rtc,
        socket: &UdpSocket,
        servers: &str,
        user: &str,
        pass: &str,
    ) -> Self {
        let mut set = Vec::new();
        for entry in servers.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (server, entry_user, entry_pass) = split_turn_credentials(entry);
            let user = entry_user.as_deref().unwrap_or(user);
            let pass = entry_pass.as_deref().unwrap_or(pass);
            if let Some(client) = setup(rtc, socket, &server, user, pass).await {
                set.push(client);
            }
        }
        Self(set)
    }

    /// The client whose relayed candidate `relay` is (an str0m transmit from a
    /// relay pair carries it as `source`).
    pub(crate) fn for_relay(&mut self, relay: SocketAddr) -> Option<&mut TurnClient> {
        self.0.iter_mut().find(|c| c.relay_addr() == relay)
    }

    /// The client for the server (or bridge) at `addr`.
    pub(crate) fn for_server(&mut self, addr: SocketAddr) -> Option<&mut TurnClient> {
        self.0.iter_mut().find(|c| c.is_server(addr))
    }

    /// Each allocation's relayed candidate address.
    pub(crate) fn relay_addrs(&self) -> Vec<SocketAddr> {
        self.0.iter().map(|c| c.relay_addr()).collect()
    }

    /// Refresh every allocation (and its channel bindings / permissions).
    pub(crate) async fn refresh_all(&mut self, socket: &UdpSocket) {
        for c in &mut self.0 {
            let _ = c.refresh(socket).await;
        }
    }
}

/// Split GStreamer-style embedded credentials (`turn://user:pass@rest`) off a
/// server entry, returning the credential-free server string (scheme
/// normalized back to the RFC 7065 form) and the credentials when present.
fn split_turn_credentials(entry: &str) -> (String, Option<String>, Option<String>) {
    let (scheme, rest) = if let Some(r) = entry.strip_prefix("turns://") {
        ("turns:", r)
    } else if let Some(r) = entry.strip_prefix("turn://") {
        ("turn:", r)
    } else {
        return (entry.into(), None, None);
    };
    match rest.split_once('@') {
        Some((userinfo, host)) => {
            let (user, pass) = match userinfo.split_once(':') {
                Some((u, p)) => (u.into(), p.into()),
                None => (userinfo.into(), String::new()),
            };
            (format!("{scheme}{host}"), Some(user), Some(pass))
        }
        None => (format!("{scheme}{rest}"), None, None),
    }
}

/// Resolve a TURN server string to the transport address the client speaks UDP
/// to: the server itself, or a freshly spawned stream bridge for `turn:...
/// ?transport=tcp` / `turns:` servers.
async fn resolve_transport(server: &str, local_ip: IpAddr) -> Option<SocketAddr> {
    let (hostport, transport, sni) = parse_turn_server(server);
    match transport {
        TurnTransport::Udp => tokio::net::lookup_host(hostport.as_str())
            .await
            .ok()?
            .next(),
        TurnTransport::Tcp | TurnTransport::Tls => {
            spawn_stream_bridge(
                &hostport,
                matches!(transport, TurnTransport::Tls),
                &sni,
                local_ip,
            )
            .await
        }
    }
}

/// Parse a TURN server string into `(host:port, transport, sni-host)`. A bare
/// `host:port` keeps the historical UDP behavior; `turn:` / `turns:` URIs pick
/// the transport (`turns` implies TLS-over-TCP), with the RFC 7065 default
/// ports (3478 / 5349) when omitted.
fn parse_turn_server(server: &str) -> (String, TurnTransport, String) {
    let (rest, mut transport, default_port) = if let Some(r) = server.strip_prefix("turns:") {
        (r, TurnTransport::Tls, 5349u16)
    } else if let Some(r) = server.strip_prefix("turn:") {
        (r, TurnTransport::Udp, 3478u16)
    } else {
        (server, TurnTransport::Udp, 3478u16)
    };
    let (hostport, query) = match rest.split_once('?') {
        Some((h, q)) => (h, Some(q)),
        None => (rest, None),
    };
    if transport != TurnTransport::Tls {
        if let Some(q) = query {
            if q.split('&').any(|kv| kv == "transport=tcp") {
                transport = TurnTransport::Tcp;
            }
        }
    }
    let (host, port) = match hostport.rsplit_once(':') {
        // Guard against a bare IPv6 literal (no port) misparsing at its last colon.
        Some((h, p)) if p.parse::<u16>().is_ok() => (h, p.parse::<u16>().unwrap_or(default_port)),
        _ => (hostport, default_port),
    };
    (format!("{host}:{port}"), transport, host.into())
}

/// Spawn the datagram <-> stream bridge: a task owning one TCP (or TLS)
/// connection to the TURN server and a local UDP socket the `TurnClient`
/// treats as its server. Element -> server datagrams are written to the stream
/// (a ChannelData frame padded to 4 bytes, RFC 5766 §11.5; STUN messages
/// as-is); the stream is re-delimited into messages (self-describing lengths)
/// and each is forwarded back as one datagram. Returns the bridge's UDP
/// address. The task ends when the stream closes.
async fn spawn_stream_bridge(
    hostport: &str,
    tls: bool,
    sni: &str,
    local_ip: IpAddr,
) -> Option<SocketAddr> {
    let tcp = tokio::net::TcpStream::connect(hostport).await.ok()?;
    let _ = tcp.set_nodelay(true);
    let udp = UdpSocket::bind((local_ip, 0)).await.ok()?;
    let addr = udp.local_addr().ok()?;
    if tls {
        let connector = tokio_native_tls::TlsConnector::from(
            tokio_native_tls::native_tls::TlsConnector::new().ok()?,
        );
        let stream = connector.connect(sni, tcp).await.ok()?;
        tokio::spawn(bridge_loop(udp, stream));
    } else {
        tokio::spawn(bridge_loop(udp, tcp));
    }
    Some(addr)
}

async fn bridge_loop<S>(udp: UdpSocket, mut stream: S)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    // The element's socket address, learned from its first datagram.
    let mut element: Option<SocketAddr> = None;
    let mut dgram = alloc::vec![0u8; 65_535];
    let mut chunk = [0u8; 8192];
    let mut inbound: Vec<u8> = Vec::with_capacity(4096);
    loop {
        tokio::select! {
            r = udp.recv_from(&mut dgram) => {
                let Ok((n, from)) = r else { break };
                element = Some(from);
                if stream.write_all(&dgram[..n]).await.is_err() {
                    break;
                }
                // Over a stream a ChannelData message is padded to 4 bytes.
                if n >= 1 && dgram[0] & 0xC0 == 0x40 {
                    let pad = (4 - (n % 4)) % 4;
                    if pad > 0 && stream.write_all(&[0u8; 3][..pad]).await.is_err() {
                        break;
                    }
                }
            }
            r = stream.read(&mut chunk) => {
                let n = match r {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                inbound.extend_from_slice(&chunk[..n]);
                loop {
                    match stream_message_bounds(&inbound) {
                        Some((consumed, msg_len)) => {
                            if let Some(el) = element {
                                let _ = udp.send_to(&inbound[..msg_len], el).await;
                            }
                            inbound.drain(..consumed);
                        }
                        // Incomplete: wait for more bytes. A first byte that is
                        // neither STUN nor ChannelData means the stream lost
                        // sync: unrecoverable, drop the connection.
                        None if inbound.first().is_some_and(|b| b & 0xC0 > 0x40) => return,
                        None => break,
                    }
                }
            }
        }
    }
}

/// Delimit the next complete TURN message at the head of a stream buffer,
/// returning `(bytes_to_consume_incl_padding, message_len)`, or `None` when the
/// buffer holds only a partial message.
fn stream_message_bounds(buf: &[u8]) -> Option<(usize, usize)> {
    if buf.len() < 4 {
        return None;
    }
    let len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
    if buf[0] & 0xC0 == 0x40 {
        // ChannelData: 4-byte header + data, padded to 4 on the stream.
        let msg = 4 + len;
        let total = msg + ((4 - (msg % 4)) % 4);
        (buf.len() >= total).then_some((total, msg))
    } else if buf[0] & 0xC0 == 0 {
        // STUN: 20-byte header + attributes (already 4-aligned).
        let total = 20 + len;
        (buf.len() >= total).then_some((total, total))
    } else {
        None
    }
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
/// 5389 §15.2): port XORed with the high cookie half; an IPv4 address with the
/// cookie, an IPv6 address with the cookie concatenated with the transaction id.
fn encode_xor_addr(addr: SocketAddr, txn: &[u8; 12]) -> Vec<u8> {
    let magic = MAGIC.to_be_bytes();
    let xport = addr.port() ^ (MAGIC >> 16) as u16;
    match addr {
        SocketAddr::V4(v4) => {
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
        SocketAddr::V6(v6) => {
            let ip = v6.ip().octets();
            let mut out = Vec::with_capacity(20);
            out.push(0);
            out.push(0x02); // family IPv6
            out.extend_from_slice(&xport.to_be_bytes());
            for k in 0..16 {
                let x = if k < 4 { magic[k] } else { txn[k - 4] };
                out.push(ip[k] ^ x);
            }
            out
        }
    }
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

/// Find and decode an XOR-address attribute (peer / relayed / mapped), IPv4 or
/// IPv6 (the latter XORed with cookie + transaction id, RFC 5389 §15.2).
fn find_xor_addr(msg: &[u8], atype: u16, txn: &[u8; 12]) -> Option<SocketAddr> {
    let val = find_bytes(msg, atype)?;
    if val.len() < 8 {
        return None;
    }
    let magic = MAGIC.to_be_bytes();
    let port = u16::from_be_bytes([val[2], val[3]]) ^ (MAGIC >> 16) as u16;
    match val[1] {
        0x01 => {
            let mut ip = [val[4], val[5], val[6], val[7]];
            for k in 0..4 {
                ip[k] ^= magic[k];
            }
            Some(SocketAddr::from((Ipv4Addr::from(ip), port)))
        }
        0x02 if val.len() >= 20 => {
            let mut ip = [0u8; 16];
            ip.copy_from_slice(&val[4..20]);
            for k in 0..16 {
                ip[k] ^= if k < 4 { magic[k] } else { txn[k - 4] };
            }
            Some(SocketAddr::from((std::net::Ipv6Addr::from(ip), port)))
        }
        _ => None,
    }
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

    #[test]
    fn xor_addr_round_trips_v6() {
        let txn = [3u8; 12];
        let addr: SocketAddr = "[2001:db8::17]:6001".parse().unwrap();
        let mut msg = MessageBuilder::new(DATA_INDICATION, txn);
        msg.push_xor_peer_address(addr, &txn);
        let bytes = msg.finish();
        assert_eq!(
            find_xor_addr(&bytes, ATTR_XOR_PEER_ADDRESS, &txn),
            Some(addr)
        );
    }

    #[test]
    fn parses_turn_server_forms() {
        let udp = parse_turn_server("relay.example:3478");
        assert_eq!(
            udp,
            (
                "relay.example:3478".into(),
                TurnTransport::Udp,
                "relay.example".into()
            )
        );
        let uri_udp = parse_turn_server("turn:relay.example");
        assert_eq!(uri_udp.0, "relay.example:3478");
        assert_eq!(uri_udp.1, TurnTransport::Udp);
        let tcp = parse_turn_server("turn:relay.example:3478?transport=tcp");
        assert_eq!(tcp.1, TurnTransport::Tcp);
        let tls = parse_turn_server("turns:relay.example");
        assert_eq!(
            tls,
            (
                "relay.example:5349".into(),
                TurnTransport::Tls,
                "relay.example".into()
            )
        );
    }

    #[test]
    fn stream_delimiting_handles_partials_and_padding() {
        // Partial STUN header: incomplete.
        assert_eq!(stream_message_bounds(&[0x01, 0x13, 0x00]), None);
        // STUN message with 4 attribute bytes: 24 total, no extra padding.
        let mut stun = MessageBuilder::new(ALLOCATE_SUCCESS, [0u8; 12]);
        stun.push_lifetime(60);
        let stun = stun.finish();
        let mut buf = stun.clone();
        buf.extend_from_slice(&[0x40, 0x00]); // trailing partial frame
        assert_eq!(stream_message_bounds(&buf), Some((stun.len(), stun.len())));
        // ChannelData of length 5 is padded to 12 on the stream, payload len 9.
        let frame = [
            0x40u8, 0x00, 0x00, 0x05, b'h', b'e', b'l', b'l', b'o', 0, 0, 0,
        ];
        assert_eq!(stream_message_bounds(&frame), Some((12, 9)));
        assert_eq!(
            stream_message_bounds(&frame[..10]),
            None,
            "padding not yet arrived"
        );
        // A non-STUN, non-ChannelData first byte is a desync, not a message.
        assert_eq!(stream_message_bounds(&[0x80, 0, 0, 0, 0]), None);
    }

    #[test]
    fn splits_gstreamer_style_credentials() {
        assert_eq!(
            split_turn_credentials("turn://alice:s3cret@relay.example:3478?transport=tcp"),
            (
                "turn:relay.example:3478?transport=tcp".into(),
                Some("alice".into()),
                Some("s3cret".into())
            )
        );
        assert_eq!(
            split_turn_credentials("turns://relay.example"),
            ("turns:relay.example".into(), None, None)
        );
        // Bare host:port passes through untouched.
        assert_eq!(
            split_turn_credentials("relay.example:3478"),
            ("relay.example:3478".into(), None, None)
        );
    }

    /// One allocation per configured server: `G2G_TURN_SERVER` may be a
    /// comma-separated list (e.g. the same coturn over UDP and TCP), and the
    /// set must expose one relayed candidate per entry.
    #[tokio::test]
    #[ignore = "needs a reachable TURN server (G2G_TURN_SERVER/_USER/_PASS)"]
    async fn turn_set_allocates_per_server() {
        let (Ok(servers), Ok(user), Ok(pass)) = (
            std::env::var("G2G_TURN_SERVER"),
            std::env::var("G2G_TURN_USER"),
            std::env::var("G2G_TURN_PASS"),
        ) else {
            std::eprintln!("skipping: set G2G_TURN_SERVER (may be a list), _USER, _PASS");
            return;
        };
        let want = servers.split(',').filter(|s| !s.trim().is_empty()).count();
        let host_ip = crate::webrtc_util::select_host_ip();
        let socket = UdpSocket::bind((host_ip, 0)).await.expect("bind");
        let mut rtc = str0m::RtcConfig::new().build(std::time::Instant::now());
        let set = TurnSet::setup(&mut rtc, &socket, &servers, &user, &pass).await;
        assert_eq!(
            set.relay_addrs().len(),
            want,
            "every server in the list allocated a relay"
        );
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
        let host_ip = crate::webrtc_util::select_host_ip();
        // The same resolution the elements use: a `turn:...?transport=tcp` or
        // `turns:` server spawns the stream bridge here.
        let server: SocketAddr = resolve_transport(&server, host_ip)
            .await
            .expect("resolve server / spawn bridge");
        // Match the server's address family (a `[::1]:3478` server exercises
        // the v6 XOR codec + RFC 6156 relay allocation end to end).
        let bind_ip: IpAddr = if server.is_ipv6() {
            IpAddr::from(std::net::Ipv6Addr::LOCALHOST)
        } else {
            host_ip
        };

        let client_sock = UdpSocket::bind((bind_ip, 0)).await.expect("bind client");
        let peer_sock = UdpSocket::bind((bind_ip, 0)).await.expect("bind peer");
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
            // Surface server rejections (this is a user-run diagnostic harness).
            if let Some((code, reason)) = parse_error(&buf[..n]) {
                std::eprintln!(
                    "turn: response {:#06x} error {code} {reason}",
                    message_type(&buf[..n])
                );
            }
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

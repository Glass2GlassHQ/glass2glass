//! SRT egress sink (SrtSink, `srt` feature). Calls an SRT listener, waits for a
//! caller, or meets a rendezvous peer over UDP, runs the HSv5 handshake, then carries an incoming
//! MPEG-TS byte stream (`Caps::ByteStream{MpegTs}`, as produced by `mpegtsmux`)
//! as SRT data packets, retransmitting on the receiver's NAK loss reports. The
//! sans-IO [`srt`](crate::srt) module does the protocol work; this element is the
//! tokio UDP I/O around it.
//!
//! Scope: one connection in the caller, listener or rendezvous role, NAK-based ARQ, AES-128/256 encryption
//! with optional mid-stream key rotation (`with_key_rotation`); a rekey KM is
//! retransmitted until the peer returns a KMRSP, so it survives packet loss.
//! Congestion control is a follow-up.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use g2g_core::log::Target;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    NetworkFailure, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

use crate::bytestream::{mpegts_caps, TS_DATAGRAM_PAYLOAD};
use crate::filesink::io_err;
use crate::srt::{self, LiveCc, SrtHandshake, SrtSender, CC_DEFAULT_OVERHEAD};
use crate::srtcrypto::{AesKeySize, SrtCrypto, KM_KK_EVEN};
use crate::srturi::{bind_toward, parse_host, parse_srt_uri, SrtMode, SrtPeer};

/// Our SRT socket id.
const SOCKET_ID: u32 = 0x6732_7363; // "g2sc"
/// Initial data sequence number.
const INIT_SEQ: u32 = 1;
/// Send-buffer depth for retransmission.
const SEND_CAPACITY: usize = 8192;
/// Default target latency advertised in the handshake.
const DEFAULT_LATENCY_MS: u16 = 120;
/// Resend an unacknowledged mid-stream KM (rekey) this often until the peer
/// returns a KMRSP, so a rekey survives KM-packet loss on a lossy link.
const KM_RETRANSMIT_US: u64 = 5_000;
/// Stop retransmitting the KM after this many tries (a peer that never KMRSPs,
/// e.g. an older receiver); the new key is already active locally regardless.
const KM_MAX_RETRANSMITS: u32 = 20;
// libsrt resends an unanswered handshake packet this often
const HANDSHAKE_RESEND_NS: u64 = 250_000_000;
// a call that failed is not retried sooner, frames in between are dropped
pub(crate) const RECONNECT_INTERVAL_NS: u64 = 1_000_000_000;
const LOG_CATEGORY: &str = "srtsink";
// a receiver sends a full ACK every 10 ms
const ACK_POLL_NS: u64 = 5_000_000;

// the first frame sent on a connection, where its srt timestamps start
#[derive(Debug, Clone, Copy)]
struct ConnectionStart {
    pts_ns: u64,
    sent_at_ns: u64,
}

/// A mid-stream rekey KM awaiting the peer's KMRSP. Retransmitted on a timer
/// until acknowledged (or [`KM_MAX_RETRANSMITS`] is reached).
#[derive(Debug)]
struct PendingKm {
    /// The full KM control packet, resent verbatim on each retry.
    pkt: Vec<u8>,
    /// The KM blob (control CIF), matched against the KMRSP to confirm the ack.
    km: Vec<u8>,
    /// Earliest monotonic microsecond time for the next retransmit.
    next_at_us: u64,
    retries: u32,
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::srtsink::SrtSink;
///
/// let element = SrtSink::new("127.0.0.1:9000".parse().unwrap())
///     .with_stream_id("live")
///     .with_latency(200);
/// ```
#[derive(Debug)]
pub struct SrtSink {
    // the listener a caller calls, or the peer a rendezvous meets
    remote: SrtPeer,
    mode: SrtMode,
    // set by a uri, which is the only way to name the port a listener binds
    bind: Option<SocketAddr>,
    latency_ms: u16,
    stream_id: Option<String>,
    passphrase: Option<String>,
    key_size: AesKeySize,
    /// Rekey the stream cipher every N data packets (encrypted streams only);
    /// `None` keeps one key for the whole session.
    rekey_interval: Option<u64>,
    packets_since_rekey: u64,
    rekeys: u64,
    /// A rekey KM awaiting the peer's KMRSP, retransmitted until acknowledged.
    pending_km: Option<PendingKm>,
    /// Count of KM retransmissions (for tests / metrics).
    km_retransmits: u64,
    /// Live-mode congestion control (output pacing); `None` sends as fast as the
    /// pipeline produces.
    cc: Option<LiveCc>,
    /// Earliest monotonic microsecond time the next packet may go (pacing cursor).
    next_send_us: u64,
    socket: Option<tokio::net::UdpSocket>,
    handshake: Option<SrtHandshake>,
    sender: Option<SrtSender>,
    configured: bool,
    frames_sent: u64,
    packets_sent: u64,
    retransmits: u64,
    eos_seen: bool,
    auto_reconnect: bool,
    connection_start: Option<ConnectionStart>,
    // how far the last frame's timestamp is ahead of the clock, which delays its delivery
    timestamps_ahead_ns: u64,
    connected_at_ns: u64,
    last_peer_packet_ns: u64,
    // monotonic time before which a failed connection is not called again
    reconnect_at_ns: u64,
}

impl SrtSink {
    /// Connect to an SRT listener at `dest` (e.g. `127.0.0.1:9000`).
    pub fn new(dest: SocketAddr) -> Self {
        Self {
            remote: dest.into(),
            mode: SrtMode::Caller,
            bind: None,
            latency_ms: DEFAULT_LATENCY_MS,
            stream_id: None,
            passphrase: None,
            key_size: AesKeySize::Aes128,
            rekey_interval: None,
            packets_since_rekey: 0,
            rekeys: 0,
            pending_km: None,
            km_retransmits: 0,
            cc: None,
            next_send_us: 0,
            socket: None,
            handshake: None,
            sender: None,
            configured: false,
            frames_sent: 0,
            packets_sent: 0,
            retransmits: 0,
            eos_seen: false,
            auto_reconnect: true,
            connection_start: None,
            timestamps_ahead_ns: 0,
            connected_at_ns: 0,
            last_peer_packet_ns: 0,
            reconnect_at_ns: 0,
        }
    }

    /// Set the SRT `streamid` carried in the handshake (the listener routes on it).
    pub fn with_stream_id(mut self, stream_id: impl Into<String>) -> Self {
        self.stream_id = Some(stream_id.into());
        self
    }

    /// Set the advertised target latency (ms).
    pub fn with_latency(mut self, latency_ms: u16) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    /// Encrypt the stream with AES-CTR under a key derived from `passphrase`. A
    /// fresh random stream key is generated per connection and exchanged
    /// (wrapped) in the handshake KM extension; the listener needs the same
    /// passphrase to decrypt.
    pub fn with_passphrase(mut self, passphrase: impl Into<String>) -> Self {
        self.passphrase = Some(passphrase.into());
        self
    }

    /// Use AES-256 (instead of the default AES-128) for the stream cipher. The
    /// key size rides in the KM `KLen` field, so the listener picks it up from
    /// the handshake with no extra configuration. Only meaningful with a
    /// passphrase set.
    pub fn with_aes256(mut self) -> Self {
        self.key_size = AesKeySize::Aes256;
        self
    }

    /// Rekey the stream cipher every `every_packets` data packets: a fresh random
    /// key is generated in the other parity slot, announced to the receiver in a
    /// KM control packet, and made active, so no single key encrypts the whole
    /// session (limiting exposure if a key is ever recovered). Encrypted streams
    /// only (needs a passphrase); off by default.
    pub fn with_key_rotation(mut self, every_packets: u64) -> Self {
        self.rekey_interval = Some(every_packets.max(1));
        self
    }

    /// Number of mid-stream rekeys performed.
    pub fn rekeys(&self) -> u64 {
        self.rekeys
    }

    /// Number of KM retransmissions sent while awaiting a KMRSP.
    pub fn km_retransmits(&self) -> u64 {
        self.km_retransmits
    }

    /// Enable live-mode congestion control (output pacing). A positive
    /// `max_bytes_per_sec` caps the egress at that bandwidth; `0` follows the
    /// measured input rate plus a retransmit headroom, so SRT does not
    /// burst-flood the path. Off by default (the pipeline's own cadence paces
    /// the stream).
    pub fn with_max_bandwidth(mut self, max_bytes_per_sec: u64) -> Self {
        self.cc = Some(LiveCc::new(max_bytes_per_sec, CC_DEFAULT_OVERHEAD));
        self
    }

    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    pub fn packets_sent(&self) -> u64 {
        self.packets_sent
    }

    pub fn retransmits(&self) -> u64 {
        self.retransmits
    }

    pub fn eos_seen(&self) -> bool {
        self.eos_seen
    }

    fn set_uri(&mut self, uri: &str) -> Result<(), PropError> {
        let uri = parse_srt_uri(uri)?;
        let (bind, remote) = uri.endpoints()?;
        // lossmaxttl only tunes a receiver
        let too_long_stream_id = uri
            .stream_id
            .as_ref()
            .is_some_and(|id| id.len() > crate::srt::MAX_STREAM_ID_BYTES);
        if uri.loss_max_ttl.is_some() || too_long_stream_id {
            return Err(PropError::Value);
        }
        self.mode = uri.mode;
        self.bind = Some(bind);
        if let Some(remote) = remote {
            self.remote = remote;
        }
        self.latency_ms = uri.latency_ms.unwrap_or(self.latency_ms);
        self.key_size = uri.key_size.unwrap_or(self.key_size);
        if uri.passphrase.is_some() {
            self.passphrase = uri.passphrase;
        }
        if uri.stream_id.is_some() {
            self.stream_id = uri.stream_id;
        }
        Ok(())
    }

    fn uri(&self) -> String {
        match (self.mode, self.bind) {
            (SrtMode::Listener, Some(bind)) => alloc::format!("srt://{bind}?mode=listener"),
            (SrtMode::Listener, None) => String::new(),
            (mode, _) => alloc::format!("srt://{}?mode={}", self.remote, mode.name()),
        }
    }

    fn bind_address(&self, remote: Option<SocketAddr>) -> Result<SocketAddr, G2gError> {
        let any = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
        // without a uri a caller binds any port, and a rendezvous the port it meets
        let bind = match (self.mode, self.bind) {
            (_, Some(bind)) => bind,
            (SrtMode::Caller, None) => SocketAddr::new(any, 0),
            (SrtMode::Rendezvous, None) => SocketAddr::new(any, self.remote.port),
            (SrtMode::Listener, None) => return Err(G2gError::NotConfigured),
        };
        Ok(remote.map_or(bind, |remote| bind_toward(bind, remote)))
    }

    fn new_handshake(&self, km: Option<Vec<u8>>) -> SrtHandshake {
        let stream_id = self.stream_id.clone();
        match self.mode {
            SrtMode::Caller => {
                SrtHandshake::new_caller(SOCKET_ID, INIT_SEQ, self.latency_ms, stream_id, km)
            }
            SrtMode::Rendezvous => SrtHandshake::new_rendezvous(
                SOCKET_ID,
                INIT_SEQ,
                self.latency_ms,
                stream_id,
                km,
                unpredictable_cookie(),
            ),
            SrtMode::Listener => {
                let mut handshake =
                    SrtHandshake::new_listener(SOCKET_ID, self.latency_ms, unpredictable_cookie());
                handshake.admit_only(stream_id, None);
                handshake
            }
        }
    }

    /// Bind the UDP socket and run the handshake to established, arming the
    /// sender (its destination is the peer's socket id).
    async fn connect_and_handshake(&mut self) -> Result<(), G2gError> {
        let remote = match self.mode {
            SrtMode::Listener => None,
            SrtMode::Caller | SrtMode::Rendezvous => Some(self.remote.resolve().await?),
        };
        let socket = tokio::net::UdpSocket::bind(self.bind_address(remote)?)
            .await
            .map_err(io_err)?;

        // For an encrypted stream, generate a fresh key + salt, advertise the
        // wrapped key in the handshake KM extension, and arm the sender's cipher.
        let crypto = self
            .passphrase
            .as_ref()
            .map(|_| SrtCrypto::generate(self.key_size));
        let km = crypto
            .as_ref()
            .zip(self.passphrase.as_ref())
            .map(|(c, pass)| c.build_km(pass, KM_KK_EVEN));

        let mut hs = self.new_handshake(km);
        // a caller fails at once on a refused port, a rendezvous peer may come up later
        if let (SrtMode::Caller, Some(remote)) = (self.mode, remote) {
            socket.connect(remote).await.map_err(io_err)?;
        }
        let peer = match remote {
            None => accept_call(&socket, &mut hs).await?,
            Some(remote) => {
                if let Err(error) = complete_handshake(&socket, remote, &mut hs).await {
                    if let Some(rejection) = hs.rejection() {
                        g2g_core::g2g_warn!(
                            Target::category(LOG_CATEGORY),
                            "{remote} rejected the call with handshake type {rejection}"
                        );
                    }
                    return Err(error);
                }
                remote
            }
        };
        socket.connect(peer).await.map_err(io_err)?;
        // the peer's KM carries the key of the caller, or of the rendezvous initiator
        let crypto = match (self.mode, &self.passphrase) {
            (SrtMode::Caller, _) | (_, None) => crypto,
            (_, Some(passphrase)) => {
                let km = hs
                    .peer_km()
                    .ok_or(G2gError::Network(NetworkFailure::Connect))?;
                let crypto = SrtCrypto::from_km(km, passphrase)
                    .ok_or(G2gError::Network(NetworkFailure::Connect))?;
                Some(crypto)
            }
        };
        self.connected_at_ns = g2g_core::metrics::monotonic_ns();
        self.last_peer_packet_ns = self.connected_at_ns;
        let mut sender = SrtSender::new(hs.peer_socket_id(), hs.init_seq(), SEND_CAPACITY);
        if let Some(c) = crypto {
            sender.set_crypto(c);
        }
        self.sender = Some(sender);
        self.handshake = Some(hs);
        self.socket = Some(socket);
        Ok(())
    }

    /// Drain pending control packets (NAK / ACK) without blocking and resend.
    async fn service_control(&mut self) -> Result<(), G2gError> {
        let socket = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
        let mut resends = alloc::vec::Vec::new();
        let mut buf = [0u8; 2048];
        loop {
            match socket.try_recv(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    self.last_peer_packet_ns = g2g_core::metrics::monotonic_ns();
                    if let Some(ctrl) = srt::parse_control(&buf[..n]) {
                        if ctrl == srt::Control::Shutdown {
                            return Err(G2gError::Network(NetworkFailure::Stream));
                        }
                        // a peer that missed our last handshake packet sends its own again
                        if let srt::Control::Handshake(_) = ctrl {
                            let handshake =
                                self.handshake.as_mut().ok_or(G2gError::NotConfigured)?;
                            resends.extend(handshake.on_packet(&buf[..n]).reply);
                            continue;
                        }
                        // A KMRSP acknowledges the outstanding rekey: stop
                        // retransmitting the KM once its blob matches.
                        if let srt::Control::KeyMaterial { rsp: true, km } = &ctrl {
                            if self.pending_km.as_ref().is_some_and(|p| &p.km == km) {
                                self.pending_km = None;
                            }
                            continue;
                        }
                        // control packets carry the microseconds since the connection, wrapping at 32 bits
                        let control_timestamp =
                            ((self.last_peer_packet_ns - self.connected_at_ns) / 1000) as u32;
                        let sender = self.sender.as_mut().ok_or(G2gError::NotConfigured)?;
                        resends.extend(sender.on_control(&ctrl, control_timestamp));
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                // the listener's port refused a datagram
                Err(e) => return Err(io_err(e)),
            }
        }
        let socket = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
        for pkt in &resends {
            socket.send(pkt).await.map_err(io_err)?;
        }
        // Retransmit an unacknowledged rekey KM until the peer KMRSPs (or we hit
        // the retry cap, at which point the new key is still active locally).
        if self.pending_km.is_some() {
            let now_us = g2g_core::metrics::monotonic_ns() / 1000;
            let due = self
                .pending_km
                .as_ref()
                .is_some_and(|p| now_us >= p.next_at_us);
            if due {
                let exhausted = self
                    .pending_km
                    .as_ref()
                    .is_some_and(|p| p.retries >= KM_MAX_RETRANSMITS);
                if exhausted {
                    self.pending_km = None;
                } else {
                    let pkt = {
                        let p = self.pending_km.as_mut().expect("pending");
                        p.retries += 1;
                        p.next_at_us = now_us + KM_RETRANSMIT_US;
                        p.pkt.clone()
                    };
                    self.km_retransmits += 1;
                    let socket = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
                    socket.send(&pkt).await.map_err(io_err)?;
                }
            }
        }
        self.retransmits = self.sender.as_ref().map(|s| s.retransmits()).unwrap_or(0);
        Ok(())
    }

    async fn send_frame(&mut self, bytes: &[u8], pts_ns: u64) -> Result<(), G2gError> {
        // srt timestamps count from the start of the connection
        let now_ns = g2g_core::metrics::monotonic_ns();
        let start = *self.connection_start.get_or_insert(ConnectionStart {
            pts_ns,
            sent_at_ns: now_ns,
        });
        let since_start_ns = pts_ns.saturating_sub(start.pts_ns);
        self.timestamps_ahead_ns = (start.sent_at_ns + since_start_ns).saturating_sub(now_ns);
        let timestamp = (since_start_ns / 1000) as u32;
        // Fragment the TS byte stream into SRT payloads (7 TS packets).
        let mut sent = Vec::new();
        {
            let sender = self.sender.as_mut().ok_or(G2gError::NotConfigured)?;
            for chunk in bytes.chunks(TS_DATAGRAM_PAYLOAD) {
                sent.push(sender.send(chunk, timestamp));
            }
        }
        let n_packets = sent.len() as u64;
        for pkt in &sent {
            // Congestion control: pace each packet to the target rate
            // so the egress does not burst-flood the path.
            if let Some(cc) = self.cc.as_mut() {
                let now = g2g_core::metrics::monotonic_ns() / 1000;
                if now < self.next_send_us {
                    tokio::time::sleep(core::time::Duration::from_micros(self.next_send_us - now))
                        .await;
                }
                let sent_at = g2g_core::metrics::monotonic_ns() / 1000;
                cc.on_packet(pkt.len(), sent_at);
                let period = cc.snd_period_us(pkt.len());
                self.next_send_us = sent_at.max(self.next_send_us).saturating_add(period);
            }
            let socket = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
            socket.send(pkt).await.map_err(io_err)?;
            self.packets_sent += 1;
        }
        self.frames_sent += 1;
        self.service_control().await?;

        // Mid-stream rekey when due (encrypted streams only): roll the
        // cipher to a fresh key in the other parity slot, announce it,
        // and make it active for the next packets. The KM goes out
        // before any packet under the new key so the receiver installs
        // it first; the previous key stays live for retransmits.
        if let Some(interval) = self.rekey_interval {
            self.packets_since_rekey += n_packets;
            if self.packets_since_rekey >= interval {
                self.packets_since_rekey = 0;
                if let Some(pass) = self.passphrase.clone() {
                    let new = SrtCrypto::generate(self.key_size);
                    let km_pkt = {
                        let sender = self.sender.as_mut().ok_or(G2gError::NotConfigured)?;
                        sender.rekey(new, &pass)
                    };
                    let socket = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
                    socket.send(&km_pkt).await.map_err(io_err)?;
                    self.rekeys += 1;
                    // Track it for KMRSP-acked retransmission so the
                    // rekey survives KM-packet loss.
                    let km_blob = match srt::parse_control(&km_pkt) {
                        Some(srt::Control::KeyMaterial { km, .. }) => km,
                        _ => Vec::new(),
                    };
                    let now_us = g2g_core::metrics::monotonic_ns() / 1000;
                    self.pending_km = Some(PendingKm {
                        pkt: km_pkt,
                        km: km_blob,
                        next_at_us: now_us + KM_RETRANSMIT_US,
                        retries: 0,
                    });
                }
            }
        }

        Ok(())
    }

    // a libsrt receiver drops the packets TSBPD still holds when a SHUTDOWN arrives
    async fn deliver_tail(&mut self) -> Result<(), G2gError> {
        let deadline = g2g_core::metrics::monotonic_ns() + srt::PEER_IDLE_TIMEOUT_NS;
        while !self
            .sender
            .as_ref()
            .ok_or(G2gError::NotConfigured)?
            .all_acknowledged()
        {
            if g2g_core::metrics::monotonic_ns() >= deadline {
                return Err(G2gError::Network(NetworkFailure::Stream));
            }
            tokio::time::sleep(core::time::Duration::from_nanos(ACK_POLL_NS)).await;
            self.service_control().await?;
        }
        // the receiver releases the last packet one latency after its timestamp comes due
        let handshake = self.handshake.as_ref().ok_or(G2gError::NotConfigured)?;
        let latency = core::time::Duration::from_millis(handshake.latency_ms().into());
        let ahead = core::time::Duration::from_nanos(self.timestamps_ahead_ns);
        tokio::time::sleep(latency + ahead).await;
        Ok(())
    }

    fn peer_idle(&self) -> bool {
        g2g_core::metrics::monotonic_ns().saturating_sub(self.last_peer_packet_ns)
            >= srt::PEER_IDLE_TIMEOUT_NS
    }

    fn connection_failed(&mut self, error: G2gError) -> Result<(), G2gError> {
        if !self.auto_reconnect {
            return Err(error);
        }
        g2g_core::g2g_warn!(
            Target::category(LOG_CATEGORY),
            "connection to {} failed ({error:?}), calling again",
            self.remote
        );
        self.socket = None;
        self.handshake = None;
        self.sender = None;
        self.pending_km = None;
        self.connection_start = None;
        Ok(())
    }
}

// Seeded from the monotonic clock so an off-path attacker can't predict it
// from the public socket id (anti-spoof).
pub(crate) fn unpredictable_cookie() -> u32 {
    let t = g2g_core::metrics::monotonic_ns();
    ((t ^ (t >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as u32 | 1
}

// answers callers until one completes the handshake, returning its address
pub(crate) async fn accept_call(
    socket: &tokio::net::UdpSocket,
    handshake: &mut SrtHandshake,
) -> Result<SocketAddr, G2gError> {
    let mut buf = [0u8; 2048];
    loop {
        let (n, from) = socket.recv_from(&mut buf).await.map_err(io_err)?;
        let step = handshake.on_packet(&buf[..n]);
        if let Some(reply) = step.reply {
            socket.send_to(&reply, from).await.map_err(io_err)?;
        }
        if step.established {
            return Ok(from);
        }
    }
}

// sends the handshake's packets to `peer` until it is established, resending the
// last one every HANDSHAKE_RESEND_NS since a lost packet otherwise stalls the call
pub(crate) async fn complete_handshake(
    socket: &tokio::net::UdpSocket,
    peer: SocketAddr,
    hs: &mut SrtHandshake,
) -> Result<(), G2gError> {
    let mut last_sent = hs.start();
    if let Some(first) = &last_sent {
        socket.send_to(first, peer).await.map_err(io_err)?;
    }
    let mut buf = [0u8; 2048];
    // Bound the *whole* handshake, not each recv: a peer that keeps sending
    // non-converging packets (a handshake mismatch, a flood) must not reset
    // the deadline every datagram and hang the pipeline forever.
    let deadline = tokio::time::Instant::now() + core::time::Duration::from_secs(8);
    let resend = core::time::Duration::from_nanos(HANDSHAKE_RESEND_NS);
    while !hs.is_established() {
        if tokio::time::Instant::now() >= deadline {
            return Err(G2gError::Network(NetworkFailure::Connect));
        }
        let (n, from) = match tokio::time::timeout(resend, socket.recv_from(&mut buf)).await {
            Ok(r) => r.map_err(io_err)?,
            Err(_) => {
                if let Some(packet) = &last_sent {
                    socket.send_to(packet, peer).await.map_err(io_err)?;
                }
                continue;
            }
        };
        if from != peer {
            continue;
        }
        let step = hs.on_packet(&buf[..n]);
        if hs.rejection().is_some() {
            return Err(G2gError::Network(NetworkFailure::Connect));
        }
        if let Some(reply) = step.reply {
            socket.send_to(&reply, peer).await.map_err(io_err)?;
            last_sent = Some(reply);
        }
    }
    Ok(())
}

impl AsyncElement for SrtSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&mpegts_caps())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(mpegts_caps()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SRT sink",
            "Sink/Network",
            "Sends an MPEG-TS byte stream over SRT (caller, listener or rendezvous), NAK-retransmitting",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "host",
                PropKind::Str,
                "host name or IP of the listener a caller calls, or of the rendezvous peer",
            )
            .with_default("127.0.0.1"),
            PropertySpec::new(
                "port",
                PropKind::Uint,
                "SRT port a caller calls, or the rendezvous peer's",
            )
                .with_range("0", "65535"),
            PropertySpec::new(
                "latency",
                PropKind::Uint,
                "advertised target latency, milliseconds",
            ),
            PropertySpec::new(
                "passphrase",
                PropKind::Str,
                "AES passphrase (empty = unencrypted)",
            ),
            PropertySpec::new(
                "streamid",
                PropKind::Str,
                "stream id a caller sends, or the only one a listener accepts (empty = none)",
            ),
            PropertySpec::new(
                "auto-reconnect",
                PropKind::Bool,
                "connect again when the connection fails, ends or goes silent, dropping frames until the peer answers (a listener waits for the next caller)",
            )
            .with_default("true"),
            PropertySpec::new(
                "pbkeylen",
                PropKind::Uint,
                "AES key length in bytes of the key a caller or rendezvous peer makes, 16, 24 or 32",
            )
            .with_default("16"),
            PropertySpec::new(
                "uri",
                PropKind::Str,
                "srt://host:port calls, srt://:port listens, with mode, localaddress, localport, latency, passphrase, pbkeylen and streamid query options",
            ),
            PropertySpec::new("mode", PropKind::Str, "SRT connection mode")
                .with_default("caller")
                .with_enum_values(SrtMode::NAMES),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "host" => {
                self.remote.host = parse_host(value.as_str().ok_or(PropError::Type)?)?;
                Ok(())
            }
            "port" => {
                let port = value.as_uint().ok_or(PropError::Type)?;
                self.remote.port = u16::try_from(port).map_err(|_| PropError::Value)?;
                Ok(())
            }
            "latency" => {
                let ms = value.as_uint().ok_or(PropError::Type)?;
                self.latency_ms = ms.min(u16::MAX as u64) as u16;
                Ok(())
            }
            "passphrase" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.passphrase = (!s.is_empty()).then(|| s.to_string());
                Ok(())
            }
            "streamid" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                if s.len() > crate::srt::MAX_STREAM_ID_BYTES {
                    return Err(PropError::Value);
                }
                self.stream_id = (!s.is_empty()).then(|| s.to_string());
                Ok(())
            }
            "auto-reconnect" => {
                self.auto_reconnect = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "pbkeylen" => {
                let bytes = value.as_uint().ok_or(PropError::Type)?;
                self.key_size = AesKeySize::from_bytes(bytes).ok_or(PropError::Value)?;
                Ok(())
            }
            "uri" => self.set_uri(value.as_str().ok_or(PropError::Type)?),
            "mode" => {
                let name = value.as_str().ok_or(PropError::Type)?;
                self.mode = SrtMode::from_name(name).ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "host" => Some(PropValue::Str(self.remote.host.clone())),
            "port" => Some(PropValue::Uint(self.remote.port.into())),
            "latency" => Some(PropValue::Uint(self.latency_ms as u64)),
            "passphrase" => Some(PropValue::Str(self.passphrase.clone().unwrap_or_default())),
            "streamid" => Some(PropValue::Str(self.stream_id.clone().unwrap_or_default())),
            "auto-reconnect" => Some(PropValue::Bool(self.auto_reconnect)),
            "pbkeylen" => Some(PropValue::Uint(self.key_size.bytes() as u64)),
            "uri" => Some(PropValue::Str(self.uri())),
            "mode" => Some(PropValue::Str(self.mode.name().to_string())),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if !self.configured {
                        return Err(G2gError::NotConfigured);
                    }
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    if self.socket.is_some() {
                        // acks queued while the pipeline stalled are not silence
                        if let Err(error) = self.service_control().await {
                            self.connection_failed(error)?;
                        }
                    }
                    if self.socket.is_some() && self.peer_idle() {
                        self.connection_failed(G2gError::Network(NetworkFailure::Stream))?;
                    }
                    if self.socket.is_none() {
                        if g2g_core::metrics::monotonic_ns() < self.reconnect_at_ns {
                            return Ok(());
                        }
                        if let Err(error) = self.connect_and_handshake().await {
                            self.connection_failed(error)?;
                            self.reconnect_at_ns =
                                g2g_core::metrics::monotonic_ns() + RECONNECT_INTERVAL_NS;
                            return Ok(());
                        }
                    }
                    if let Err(error) = self.send_frame(slice, frame.timing.pts_ns).await {
                        self.connection_failed(error)?;
                    }
                }
                PipelinePacket::Eos => {
                    if self.socket.is_some() {
                        if let Err(error) = self.deliver_tail().await {
                            self.connection_failed(error)?;
                        }
                    }
                    // Signal the peer to close cleanly (RTP-style flows have no
                    // in-band end; SRT has SHUTDOWN).
                    if let (Some(socket), Some(handshake)) = (&self.socket, &self.handshake) {
                        let shutdown = srt::build_control(
                            &srt::Control::Shutdown,
                            0,
                            handshake.peer_socket_id(),
                        );
                        let _ = socket.send(&shutdown).await;
                    }
                    self.eos_seen = true;
                }
                PipelinePacket::Flush
                | PipelinePacket::CapsChanged(_)
                | PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for SrtSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(mpegts_caps()))])
    }
}

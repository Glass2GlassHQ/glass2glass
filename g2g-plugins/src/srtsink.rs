//! SRT egress sink (SrtSink, `srt` feature): the caller side. Connects out to an
//! SRT listener over UDP, runs the HSv5 handshake, then carries an incoming
//! MPEG-TS byte stream (`Caps::ByteStream{MpegTs}`, as produced by `mpegtsmux`)
//! as SRT data packets, retransmitting on the listener's NAK loss reports. The
//! sans-IO [`srt`](crate::srt) module does the protocol work; this element is the
//! tokio UDP I/O around it.
//!
//! Scope: one connection, the caller role, NAK-based ARQ, AES-128/256 encryption
//! with optional mid-stream key rotation (`with_key_rotation`); a rekey KM is
//! retransmitted until the peer returns a KMRSP, so it survives packet loss.
//! Congestion control is a follow-up.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use std::net::SocketAddr;

use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::filesink::io_err;
use crate::srt::{self, LiveCc, SrtHandshake, SrtSender, CC_DEFAULT_OVERHEAD};
use crate::srtcrypto::{AesKeySize, SrtCrypto, KM_KK_EVEN};

/// Max SRT payload bytes per packet: 7 x 188-byte TS packets, the SRT default.
const SRT_PAYLOAD: usize = 1316;
/// Our SRT socket id (caller).
const CALLER_SOCKET_ID: u32 = 0x6732_7363; // "g2sc"
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

fn ts_bytestream() -> Caps {
    Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }
}

#[derive(Debug)]
pub struct SrtSink {
    dest: SocketAddr,
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
    sender: Option<SrtSender>,
    configured: bool,
    frames_sent: u64,
    packets_sent: u64,
    retransmits: u64,
    eos_seen: bool,
}

impl SrtSink {
    /// Connect to an SRT listener at `dest` (e.g. `127.0.0.1:9000`).
    pub fn new(dest: SocketAddr) -> Self {
        Self {
            dest,
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
            sender: None,
            configured: false,
            frames_sent: 0,
            packets_sent: 0,
            retransmits: 0,
            eos_seen: false,
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

    /// Connect the UDP socket and run the caller handshake to established,
    /// returning the armed sender (its destination is the peer's socket id).
    async fn connect_and_handshake(&mut self) -> Result<(), G2gError> {
        let socket = tokio::net::UdpSocket::bind(("0.0.0.0", 0)).await.map_err(io_err)?;
        socket.connect(self.dest).await.map_err(io_err)?;

        // For an encrypted stream, generate a fresh key + salt, advertise the
        // wrapped key in the handshake KM extension, and arm the sender's cipher.
        let crypto = self.passphrase.as_ref().map(|_| SrtCrypto::generate(self.key_size));
        let km = crypto
            .as_ref()
            .zip(self.passphrase.as_ref())
            .map(|(c, pass)| c.build_km(pass, KM_KK_EVEN));

        let mut hs = SrtHandshake::new_caller(
            CALLER_SOCKET_ID,
            INIT_SEQ,
            self.latency_ms,
            self.stream_id.clone(),
            km,
        );
        if let Some(first) = hs.start() {
            socket.send(&first).await.map_err(io_err)?;
        }
        let mut buf = [0u8; 2048];
        // Bound the *whole* handshake, not each recv: a peer that keeps sending
        // non-converging packets (a handshake mismatch, a flood) must not reset
        // the deadline every datagram and hang the pipeline forever.
        let deadline = tokio::time::Instant::now() + core::time::Duration::from_secs(8);
        while !hs.is_established() {
            let n = match tokio::time::timeout_at(deadline, socket.recv(&mut buf)).await {
                Ok(r) => r.map_err(io_err)?,
                Err(_) => return Err(G2gError::Hardware(HardwareError::Other)),
            };
            let step = hs.on_packet(&buf[..n]);
            if let Some(reply) = step.reply {
                socket.send(&reply).await.map_err(io_err)?;
            }
        }
        let mut sender = SrtSender::new(hs.peer_socket_id(), INIT_SEQ, SEND_CAPACITY);
        if let Some(c) = crypto {
            sender.set_crypto(c);
        }
        self.sender = Some(sender);
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
                    if let Some(ctrl) = srt::parse_control(&buf[..n]) {
                        // A KMRSP acknowledges the outstanding rekey: stop
                        // retransmitting the KM once its blob matches.
                        if let srt::Control::KeyMaterial { rsp: true, km } = &ctrl {
                            if self.pending_km.as_ref().is_some_and(|p| &p.km == km) {
                                self.pending_km = None;
                            }
                            continue;
                        }
                        let sender = self.sender.as_mut().ok_or(G2gError::NotConfigured)?;
                        resends.extend(sender.on_control(&ctrl, 0));
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => break,
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
            let due = self.pending_km.as_ref().is_some_and(|p| now_us >= p.next_at_us);
            if due {
                let exhausted =
                    self.pending_km.as_ref().is_some_and(|p| p.retries >= KM_MAX_RETRANSMITS);
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
}

impl AsyncElement for SrtSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&ts_bytestream())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(ts_bytestream()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SRT sink",
            "Sink/Network",
            "Sends an MPEG-TS byte stream over SRT (caller), NAK-retransmitting",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("host", PropKind::Str, "destination host (IP of the SRT listener)")
                .with_default("127.0.0.1"),
            PropertySpec::new("port", PropKind::Uint, "destination SRT port")
                .with_range("0", "65535"),
            PropertySpec::new("latency", PropKind::Uint, "advertised target latency, milliseconds"),
            PropertySpec::new("passphrase", PropKind::Str, "AES passphrase (empty = unencrypted)"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(r) = crate::netprop::set_addr_prop(&mut self.dest, "host", name, &value) {
            return r;
        }
        match name {
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
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(v) = crate::netprop::get_addr_prop(&self.dest, "host", name) {
            return Some(v);
        }
        match name {
            "latency" => Some(PropValue::Uint(self.latency_ms as u64)),
            "passphrase" => Some(PropValue::Str(self.passphrase.clone().unwrap_or_default())),
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    if self.socket.is_none() {
                        self.connect_and_handshake().await?;
                    }
                    let timestamp = (frame.timing.pts_ns / 1000) as u32;
                    // Fragment the TS byte stream into SRT payloads (7 TS packets).
                    let bytes = slice.as_slice();
                    let mut sent = Vec::new();
                    {
                        let sender = self.sender.as_mut().ok_or(G2gError::NotConfigured)?;
                        for chunk in bytes.chunks(SRT_PAYLOAD) {
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
                                tokio::time::sleep(core::time::Duration::from_micros(
                                    self.next_send_us - now,
                                ))
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
                                    let sender =
                                        self.sender.as_mut().ok_or(G2gError::NotConfigured)?;
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
                }
                PipelinePacket::Eos => {
                    // Signal the listener to close cleanly (RTP-style flows have no
                    // in-band end; SRT has SHUTDOWN).
                    if let Some(socket) = self.socket.as_ref() {
                        let shutdown = srt::build_control(&srt::Control::Shutdown, 0, CALLER_SOCKET_ID);
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
        Vec::from([PadTemplate::sink(CapsSet::one(ts_bytestream()))])
    }
}

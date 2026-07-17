//! SRT ingress source (SrtSrc, `srt` feature): the listener side. Binds a UDP
//! port, accepts an SRT caller's HSv5 handshake, then receives the data packets,
//! reorders them, NAKs gaps for retransmission, and emits the reassembled
//! MPEG-TS byte stream downstream as `Caps::ByteStream{MpegTs}` (for `tsdemux`).
//! The sans-IO [`srt`](crate::srt) module does the protocol work; this element is
//! the tokio UDP I/O around it, the receive-side inverse of [`SrtSink`](crate::srtsink).
//!
//! Scope: one caller, the listener role, NAK-based ARQ, and TSBPD delivery
//! timing (the advertised latency holds packets back into a steady output).
//! Congestion control is a follow-up.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};

use std::net::{SocketAddr, UdpSocket as StdUdpSocket};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    FrameTiming, G2gError, HardwareError, MemoryDomain, OutputSink, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

use crate::filesink::io_err;
use crate::srt::{self, Control, SrtHandshake, SrtReceiver};
use crate::srtcrypto::SrtCrypto;

/// Our SRT socket id (listener).
const LISTENER_SOCKET_ID: u32 = 0x6732_736C; // "g2sl"
/// Default advertised target latency (ms).
const DEFAULT_LATENCY_MS: u16 = 120;
/// UDP receive buffer.
const RECV_BUF: usize = 2048;
/// Send a NAK at most this often (ns), so a gap is not re-requested every packet.
const NACK_MIN_INTERVAL_NS: u64 = 20_000_000;
/// Receive timeout (ms) so the TSBPD buffer still flushes due packets when no new
/// datagram arrives; bounds the extra delivery jitter a silent gap can add.
const TSBPD_WAKE_MS: u64 = 5;

fn ts_bytestream() -> Caps {
    Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }
}

#[derive(Debug)]
pub struct SrtSrc {
    bind: SocketAddr,
    latency_ms: u16,
    frame_limit: u64,
    passphrase: Option<String>,
    std_socket: Option<StdUdpSocket>,
    configured: bool,
}

impl SrtSrc {
    /// Listen for an SRT caller on `bind` (e.g. `0.0.0.0:9000`).
    pub fn new(bind: SocketAddr) -> Self {
        Self {
            bind,
            latency_ms: DEFAULT_LATENCY_MS,
            frame_limit: 0,
            passphrase: None,
            std_socket: None,
            configured: false,
        }
    }

    /// Use an already-bound socket (so a test can pick an ephemeral port).
    pub fn from_socket(socket: StdUdpSocket) -> Result<Self, G2gError> {
        let bind = socket.local_addr().map_err(io_err)?;
        socket.set_nonblocking(true).map_err(io_err)?;
        Ok(Self { std_socket: Some(socket), configured: true, ..Self::new(bind) })
    }

    /// Stop after `n` payloads and emit EOS (the bounded / test path).
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// Decrypt an encrypted stream using a key derived from `passphrase`. The
    /// caller's wrapped stream key arrives in the handshake KM extension; the
    /// passphrase must match the caller's or the connection fails.
    pub fn with_passphrase(mut self, passphrase: impl Into<String>) -> Self {
        self.passphrase = Some(passphrase.into());
        self
    }

    /// The bound port, once a socket exists.
    pub fn local_port(&self) -> Option<u16> {
        self.std_socket.as_ref().and_then(|s| s.local_addr().ok()).map(|a| a.port())
    }
}

fn ts_frame(bytes: alloc::vec::Vec<u8>, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming { arrival_ns: g2g_core::metrics::monotonic_ns(), ..FrameTiming::default() },
        sequence,
        meta: Default::default(),
    }
}

impl SourceLoop for SrtSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(ts_bytestream()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(ts_bytestream()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.std_socket.is_none() {
            let socket = StdUdpSocket::bind(self.bind).map_err(io_err)?;
            socket.set_nonblocking(true).map_err(io_err)?;
            self.std_socket = Some(socket);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SRT source",
            "Source/Network",
            "Receives an MPEG-TS byte stream over SRT (listener), NAK-recovering loss",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("address", PropKind::Str, "local bind address (IP to listen on)")
                .with_default("0.0.0.0"),
            PropertySpec::new("port", PropKind::Uint, "local SRT port to listen on")
                .with_range("0", "65535"),
            PropertySpec::new("latency", PropKind::Uint, "SRT receiver latency, milliseconds"),
            PropertySpec::new("passphrase", PropKind::Str, "AES passphrase (empty = unencrypted)"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(r) = crate::netprop::set_addr_prop(&mut self.bind, "address", name, &value) {
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
        if let Some(v) = crate::netprop::get_addr_prop(&self.bind, "address", name) {
            return Some(v);
        }
        match name {
            "latency" => Some(PropValue::Uint(self.latency_ms as u64)),
            "passphrase" => Some(PropValue::Str(self.passphrase.clone().unwrap_or_default())),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let std = self.std_socket.take().ok_or(G2gError::NotConfigured)?;
            std.set_nonblocking(true).map_err(io_err)?;
            let socket = tokio::net::UdpSocket::from_std(std).map_err(io_err)?;

            // Listener handshake: answer the caller until established. Seed the
            // SYN cookie from the monotonic clock so an off-path attacker can't
            // predict it from the public listener socket id (anti-spoof).
            let cookie = {
                let t = g2g_core::metrics::monotonic_ns();
                ((t ^ (t >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as u32 | 1
            };
            let mut hs = SrtHandshake::new_listener(LISTENER_SOCKET_ID, self.latency_ms, cookie);
            let mut buf = [0u8; RECV_BUF];
            let mut peer: Option<SocketAddr> = None;
            while !hs.is_established() {
                let (n, from) = socket.recv_from(&mut buf).await.map_err(io_err)?;
                peer = Some(from);
                let step = hs.on_packet(&buf[..n]);
                if let Some(reply) = step.reply {
                    socket.send_to(&reply, from).await.map_err(io_err)?;
                }
            }
            let peer = peer.ok_or(G2gError::NotConfigured)?;
            let peer_socket_id = hs.peer_socket_id();

            // The MPEG-TS byte stream the depayloaded packets reconstruct.
            out.push(PipelinePacket::CapsChanged(ts_bytestream())).await?;

            let mut receiver = SrtReceiver::new();
            // Hold packets back to the advertised latency (TSBPD): the negotiated
            // delay smooths network jitter into a steady output stream.
            receiver.set_tsbpd((self.latency_ms as u32) * 1000);
            // Derive the shared stream key from the caller's KM and our passphrase.
            if let Some(pass) = &self.passphrase {
                let km = hs.peer_km().ok_or(G2gError::Hardware(HardwareError::Other))?;
                let crypto =
                    SrtCrypto::from_km(km, pass).ok_or(G2gError::Hardware(HardwareError::Other))?;
                receiver.set_crypto(crypto);
            }
            let limit = self.frame_limit;
            let mut emitted = 0u64;
            let mut last_nack_ns = 0u64;
            loop {
                // A short receive timeout so buffered packets still flush on their
                // TSBPD schedule when no new datagram arrives (a silent gap).
                let recv = tokio::time::timeout(
                    core::time::Duration::from_millis(TSBPD_WAKE_MS),
                    socket.recv_from(&mut buf),
                )
                .await;
                let now = g2g_core::metrics::monotonic_ns();
                let now_us = now / 1000;

                match recv {
                    // Timer wake: no datagram, just drain due packets below.
                    Err(_elapsed) => {}
                    Ok(r) => {
                        let (n, from) = r.map_err(io_err)?;
                        if from == peer {
                            if srt::is_control(&buf[..n]) {
                                match srt::parse_control(&buf[..n]) {
                                    Some(Control::Shutdown) => break,
                                    // Mid-stream rekey (KMREQ): unwrap the new key
                                    // and file it into the slot its parity names,
                                    // so packets under the new key decrypt, then
                                    // KMRSP so the sender stops retransmitting.
                                    Some(Control::KeyMaterial { rsp: false, km }) => {
                                        if let Some(pass) = &self.passphrase {
                                            if let (Some(kk), Some(crypto)) = (
                                                SrtCrypto::km_kk(&km),
                                                SrtCrypto::from_km(&km, pass),
                                            ) {
                                                receiver.install_key(kk, crypto);
                                            }
                                        }
                                        let rsp = srt::build_control(
                                            &Control::KeyMaterial { rsp: true, km },
                                            0,
                                            peer_socket_id,
                                        );
                                        socket.send_to(&rsp, peer).await.map_err(io_err)?;
                                    }
                                    // Keepalive / handshake retries: ignored.
                                    _ => {}
                                }
                            } else if let Some(pkt) = srt::parse_data_packet(&buf[..n]) {
                                receiver.on_data(pkt);
                            }
                        }
                    }
                }

                // Deliver packets whose TSBPD delivery time is due, in order.
                for payload in receiver.take_ready_at(now_us) {
                    out.push(PipelinePacket::DataFrame(ts_frame(payload, emitted))).await?;
                    emitted += 1;
                    if limit != 0 && emitted >= limit {
                        out.push(PipelinePacket::Eos).await?;
                        return Ok(emitted);
                    }
                }

                // Request retransmission of any open gaps, rate-limited, and ACK.
                if now.saturating_sub(last_nack_ns) >= NACK_MIN_INTERVAL_NS {
                    let missing = receiver.missing();
                    if !missing.is_empty() {
                        let nak = srt::build_control(&Control::Nak { loss: missing }, 0, peer_socket_id);
                        socket.send_to(&nak, peer).await.map_err(io_err)?;
                        last_nack_ns = now;
                    }
                    let ack = srt::build_control(
                        &Control::Ack { ack_no: 0, ack_seq: receiver.ack_seq() },
                        0,
                        peer_socket_id,
                    );
                    socket.send_to(&ack, peer).await.map_err(io_err)?;
                }
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }
}

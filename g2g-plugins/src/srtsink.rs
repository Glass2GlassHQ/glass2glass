//! SRT egress sink (SrtSink, `srt` feature): the caller side. Connects out to an
//! SRT listener over UDP, runs the HSv5 handshake, then carries an incoming
//! MPEG-TS byte stream (`Caps::ByteStream{MpegTs}`, as produced by `mpegtsmux`)
//! as SRT data packets, retransmitting on the listener's NAK loss reports. The
//! sans-IO [`srt`](crate::srt) module does the protocol work; this element is the
//! tokio UDP I/O around it.
//!
//! Scope: one connection, the caller role, cleartext (no encryption), NAK-based
//! ARQ. The TSBPD timing model and congestion control are follow-ups.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use std::net::SocketAddr;

use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket,
};

use crate::filesink::io_err;
use crate::srt::{self, SrtHandshake, SrtSender};
use crate::srtcrypto::SrtCrypto;

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

fn ts_bytestream() -> Caps {
    Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }
}

#[derive(Debug)]
pub struct SrtSink {
    dest: SocketAddr,
    latency_ms: u16,
    stream_id: Option<String>,
    passphrase: Option<String>,
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
        let crypto = self.passphrase.as_ref().map(|_| SrtCrypto::generate());
        let km = crypto
            .as_ref()
            .zip(self.passphrase.as_ref())
            .map(|(c, pass)| c.build_km(pass));

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
        while !hs.is_established() {
            // Bound the handshake so a missing listener cannot hang the pipeline.
            let n = match tokio::time::timeout(core::time::Duration::from_secs(5), socket.recv(&mut buf)).await {
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
                    let socket = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
                    for pkt in &sent {
                        socket.send(pkt).await.map_err(io_err)?;
                        self.packets_sent += 1;
                    }
                    self.frames_sent += 1;
                    self.service_control().await?;
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

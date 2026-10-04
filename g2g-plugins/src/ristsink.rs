//! RIST sender (RistSink, `rist` feature), VSF TR-06-1 Simple Profile. It takes
//! an MPEG-TS byte stream and sends it as RTP MP2T (payload type 33, seven TS
//! packets per RTP packet, like `rtpmp2tpay`) to `address`:`port`, with RTCP
//! SR + SDES to `port + 1` from the same socket. Receiver NACKs in either RIST
//! form are answered from a history `sender-buffer` long, and RTT echo requests
//! are answered. At EOS it keeps answering NACKs for `sender-buffer`, then sends
//! BYE. The GStreamer equivalent is `rtpmp2tpay ! ristsink`.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, UdpSocket as StdUdpSocket};

use g2g_core::rtp::{RtpHeader, RTP_HEADER_LEN};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::bytestream::{mpegts_caps, TS_DATAGRAM_PAYLOAD};
use crate::filesink::io_err;
use crate::rist::{self, LinkSettings, RetransmissionHistory, MP2T_PAYLOAD_TYPE};
use crate::rtcp;

/// GStreamer's `ristsink` sends to this host by default.
const DEFAULT_ADDRESS: &str = "localhost";
const DEFAULT_SENDER_BUFFER_MS: u64 = 1200;
const NS_PER_MS: u64 = 1_000_000;

/// # Example
///
/// ```no_run
/// use g2g_core::{AsyncElement, PropValue};
/// use g2g_plugins::ristsink::RistSink;
///
/// let mut element = RistSink::default();
/// element.set_property("address", PropValue::Str("10.0.0.1".into())).unwrap();
/// element.set_property("port", PropValue::Uint(5004)).unwrap();
/// ```
#[derive(Debug)]
pub struct RistSink {
    link: LinkSettings,
    sender_buffer_ms: u64,
    ssrc: u32,
    next_sequence: u16,
    destination: Option<SocketAddr>,
    std_socket: Option<StdUdpSocket>,
    socket: Option<tokio::net::UdpSocket>,
    history: RetransmissionHistory,
    /// Sender report counters: media packets and payload octets, wrapping as
    /// the RFC 3550 fields do.
    report_packets: u32,
    report_octets: u32,
    last_rtp_timestamp: u32,
    last_media_sent_ns: u64,
    last_rtcp_sent_ns: Option<u64>,
    packets_sent: u64,
    retransmissions_sent: u64,
    eos_seen: bool,
}

impl Default for RistSink {
    fn default() -> Self {
        let ssrc = rist::media_ssrc(rist::random_u32());
        Self {
            link: LinkSettings::new(DEFAULT_ADDRESS, format!("g2g-{ssrc:08x}")),
            sender_buffer_ms: DEFAULT_SENDER_BUFFER_MS,
            ssrc,
            next_sequence: rist::random_u32() as u16,
            destination: None,
            std_socket: None,
            socket: None,
            history: RetransmissionHistory::new(DEFAULT_SENDER_BUFFER_MS * NS_PER_MS),
            report_packets: 0,
            report_octets: 0,
            last_rtp_timestamp: 0,
            last_media_sent_ns: 0,
            last_rtcp_sent_ns: None,
            packets_sent: 0,
            retransmissions_sent: 0,
            eos_seen: false,
        }
    }
}

impl RistSink {
    /// The SSRC of the original packets, which retransmissions carry with the
    /// low bit set.
    pub fn ssrc(&self) -> u32 {
        self.ssrc
    }

    pub fn packets_sent(&self) -> u64 {
        self.packets_sent
    }

    pub fn retransmissions_sent(&self) -> u64 {
        self.retransmissions_sent
    }

    pub fn eos_seen(&self) -> bool {
        self.eos_seen
    }

    fn rtcp_destination(&self) -> Result<SocketAddr, G2gError> {
        let mut destination = self.destination.ok_or(G2gError::NotConfigured)?;
        destination.set_port(destination.port() + 1);
        Ok(destination)
    }

    fn ensure_socket(&mut self) -> Result<&tokio::net::UdpSocket, G2gError> {
        if self.socket.is_none() {
            let std_socket = self.std_socket.take().ok_or(G2gError::NotConfigured)?;
            self.socket = Some(tokio::net::UdpSocket::from_std(std_socket).map_err(io_err)?);
        }
        self.socket.as_ref().ok_or(G2gError::NotConfigured)
    }

    async fn send_media(&mut self, payload: &[u8], timestamp: u32) -> Result<(), G2gError> {
        let header = RtpHeader {
            payload_type: MP2T_PAYLOAD_TYPE,
            marker: false,
            sequence: self.next_sequence,
            timestamp,
            ssrc: self.ssrc,
        };
        let mut packet = Vec::with_capacity(RTP_HEADER_LEN + payload.len());
        packet.extend_from_slice(&header.to_bytes());
        packet.extend_from_slice(payload);
        let destination = self.destination.ok_or(G2gError::NotConfigured)?;
        self.ensure_socket()?
            .send_to(&packet, destination)
            .await
            .map_err(io_err)?;
        let now_ns = g2g_core::metrics::monotonic_ns();
        self.history.record(header, payload, now_ns);
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.report_packets = self.report_packets.wrapping_add(1);
        self.report_octets = self.report_octets.wrapping_add(payload.len() as u32);
        self.last_rtp_timestamp = timestamp;
        self.last_media_sent_ns = now_ns;
        self.packets_sent += 1;
        Ok(())
    }

    /// SR + SDES, the compound every sender RTCP packet starts with.
    fn report(&self, now_ns: u64) -> Vec<u8> {
        let elapsed_ns = now_ns.saturating_sub(self.last_media_sent_ns);
        let rtp_timestamp = self
            .last_rtp_timestamp
            .wrapping_add(crate::rtpklv::rtp_timestamp_from_pts(elapsed_ns));
        let mut compound = rtcp::build_sender_report(
            self.ssrc,
            rtcp::ntp_now(),
            rtp_timestamp,
            self.report_packets,
            self.report_octets,
            &[],
        );
        compound.extend(rtcp::build_sdes_cname(self.ssrc, &self.link.cname));
        compound
    }

    async fn send_report_if_due(&mut self) -> Result<(), G2gError> {
        let now_ns = g2g_core::metrics::monotonic_ns();
        let due = self
            .last_rtcp_sent_ns
            .is_none_or(|last| now_ns.saturating_sub(last) >= self.link.rtcp_interval_ns());
        if !due {
            return Ok(());
        }
        let report = self.report(now_ns);
        let destination = self.rtcp_destination()?;
        self.ensure_socket()?
            .send_to(&report, destination)
            .await
            .map_err(io_err)?;
        self.last_rtcp_sent_ns = Some(now_ns);
        Ok(())
    }

    /// Answer one RTCP datagram from the receiver: resend what its NACKs ask
    /// for and echo an RTT request back to where it came from.
    async fn handle_rtcp(&mut self, datagram: &[u8], from: SocketAddr) -> Result<(), G2gError> {
        let mut retransmissions = Vec::new();
        let mut echo_responses = Vec::new();
        for packet in rtcp::parse_compound(datagram) {
            if let Some(ranges) = rist::requested_ranges(&packet, self.ssrc) {
                retransmissions.extend(self.history.retransmissions(&ranges));
            } else if let Some(response) = rist::rtt_echo_response(&packet) {
                echo_responses.push(response);
            }
        }
        let destination = self.destination.ok_or(G2gError::NotConfigured)?;
        for packet in &retransmissions {
            self.ensure_socket()?
                .send_to(packet, destination)
                .await
                .map_err(io_err)?;
        }
        self.retransmissions_sent += retransmissions.len() as u64;
        for response in echo_responses {
            let mut compound = self.report(g2g_core::metrics::monotonic_ns());
            compound.extend(response);
            self.ensure_socket()?
                .send_to(&compound, from)
                .await
                .map_err(io_err)?;
        }
        Ok(())
    }

    /// Read every RTCP datagram already queued without waiting.
    async fn service_rtcp(&mut self) -> Result<(), G2gError> {
        let mut buffer = alloc::vec![0u8; rist::MAX_DATAGRAM];
        loop {
            match self.ensure_socket()?.try_recv_from(&mut buffer) {
                Ok((length, from)) => self.handle_rtcp(&buffer[..length], from).await?,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) if rist::is_peer_unreachable(&error) => continue,
                Err(error) => return Err(io_err(error)),
            }
        }
        self.send_report_if_due().await
    }

    /// Keep answering NACKs for as long as the receiver may still ask, then
    /// say goodbye.
    async fn drain_and_close(&mut self) -> Result<(), G2gError> {
        let deadline = tokio::time::Instant::now() + Duration::from_millis(self.sender_buffer_ms);
        let mut buffer = alloc::vec![0u8; rist::MAX_DATAGRAM];
        while tokio::time::Instant::now() < deadline {
            let wait = Duration::from_millis(rist::TIMER_CHECK_MS);
            let received =
                tokio::time::timeout(wait, self.ensure_socket()?.recv_from(&mut buffer)).await;
            match received {
                Ok(Ok((length, from))) => self.handle_rtcp(&buffer[..length], from).await?,
                Ok(Err(error)) if !rist::is_peer_unreachable(&error) => return Err(io_err(error)),
                _ => {}
            }
            self.send_report_if_due().await?;
        }
        let mut goodbye = self.report(g2g_core::metrics::monotonic_ns());
        goodbye.extend(rtcp::build_bye(self.ssrc));
        let destination = self.rtcp_destination()?;
        self.ensure_socket()?
            .send_to(&goodbye, destination)
            .await
            .map_err(io_err)?;
        Ok(())
    }
}

impl AsyncElement for RistSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

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
        let destination = self.link.media_address().map_err(io_err)?;
        let local = if destination.is_ipv6() {
            SocketAddr::from((Ipv6Addr::UNSPECIFIED, 0))
        } else {
            SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))
        };
        let socket = StdUdpSocket::bind(local).map_err(io_err)?;
        socket.set_nonblocking(true).map_err(io_err)?;
        self.destination = Some(destination);
        self.std_socket = Some(socket);
        self.socket = None;
        self.history = RetransmissionHistory::new(self.sender_buffer_ms * NS_PER_MS);
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "RIST sink",
            "Sink/Network",
            "Sends an MPEG-TS byte stream over RIST (TR-06-1 Simple Profile), resending on NACK",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "address",
                PropKind::Str,
                "address to send packets to (IPv4, IPv6 or a host name)",
            )
            .with_default(DEFAULT_ADDRESS),
            PropertySpec::new(
                "port",
                PropKind::Uint,
                "RTP port, RTCP goes to this value + 1; must be even",
            )
            .with_default("5004")
            .with_range("2", "65534"),
            PropertySpec::new(
                "sender-buffer",
                PropKind::Uint,
                "how long sent packets stay available for retransmission, ms",
            )
            .with_default("1200")
            .with_range("0", "4294967295"),
            PropertySpec::new(
                "min-rtcp-interval",
                PropKind::Uint,
                "interval between two regular RTCP packets, ms",
            )
            .with_default("100")
            .with_range("0", "100"),
            PropertySpec::new(
                "cname",
                PropKind::Str,
                "CNAME in the SDES block of the sender report",
            ),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(result) = self.link.set_property(name, &value) {
            return result;
        }
        match name {
            "sender-buffer" => {
                let ms = value.as_uint().ok_or(PropError::Type)?;
                if ms > u64::from(u32::MAX) {
                    return Err(PropError::Value);
                }
                self.sender_buffer_ms = ms;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = self.link.get_property(name) {
            return Some(value);
        }
        match name {
            "sender-buffer" => Some(PropValue::Uint(self.sender_buffer_ms)),
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
                    let bytes = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let timestamp = crate::rtpklv::rtp_timestamp_from_pts(frame.timing.pts_ns);
                    for payload in bytes.chunks(TS_DATAGRAM_PAYLOAD) {
                        self.send_media(payload, timestamp).await?;
                    }
                    self.service_rtcp().await?;
                }
                PipelinePacket::Eos => {
                    if self.packets_sent > 0 {
                        self.drain_and_close().await?;
                    }
                    self.eos_seen = true;
                }
                _ => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for RistSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(mpegts_caps()))])
    }
}

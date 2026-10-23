//! RIST sender (RistSink, `rist` feature), VSF TR-06-1 Simple Profile. It takes
//! an MPEG-TS byte stream and sends it as RTP MP2T (payload type 33, seven TS
//! packets per RTP packet, like `rtpmp2tpay`) to `address`:`port`, with RTCP
//! SR + SDES to `port + 1` from the same socket. Receiver NACKs in either RIST
//! form are answered from a history `sender-buffer` long, and RTT echo requests
//! are answered. At EOS it keeps answering NACKs for `sender-buffer`, then sends
//! BYE. The GStreamer equivalent is `rtpmp2tpay ! ristsink`.
//!
//! `bonding-addresses` lists several links, each with its own socket and RTCP.
//! `bonding-method=broadcast` sends every packet on every link,
//! `round-robin` one link after the other, and a NACK is answered on the link
//! it came in on with the packets that link carried. A multicast address is
//! sent to with `multicast-ttl` / `multicast-loopback`, and its receivers'
//! RTCP is read from the group. `sequence-number-extension` and
//! `drop-null-ts-packets` add the TR-06-2 RTP header extension, as GStreamer's
//! `ristrtpext` inside its `ristsink`.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

use std::net::{SocketAddr, UdpSocket as StdUdpSocket};

use g2g_core::rtp::RtpHeader;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::bytestream::{mpegts_caps, TS_DATAGRAM_PAYLOAD};
use crate::filesink::io_err;
use crate::rist::{
    self, LinkAddress, LinkSettings, RetransmissionHistory, RistExtensionSettings, SentOn,
    MP2T_PAYLOAD_TYPE,
};
use crate::rtcp;

/// GStreamer's `ristsink` sends to this host by default.
const DEFAULT_ADDRESS: &str = "localhost";
const DEFAULT_SENDER_BUFFER_MS: u64 = 1200;
const NS_PER_MS: u64 = 1_000_000;
// the `seqnum-offset` of `rtpmp2tpay` that picks a random first sequence number
const RANDOM_SEQUENCE_NUMBER_OFFSET: i64 = -1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BondingMethod {
    Broadcast,
    RoundRobin,
}

impl BondingMethod {
    fn from_nick(nick: &str) -> Option<Self> {
        match nick {
            "broadcast" => Some(Self::Broadcast),
            "round-robin" => Some(Self::RoundRobin),
            _ => None,
        }
    }

    fn nick(self) -> &'static str {
        match self {
            Self::Broadcast => "broadcast",
            Self::RoundRobin => "round-robin",
        }
    }
}

#[derive(Debug)]
struct SenderLink {
    destination: SocketAddr,
    std_socket: Option<StdUdpSocket>,
    socket: Option<tokio::net::UdpSocket>,
    /// Sender report counters: media packets and payload octets sent on this
    /// link, wrapping as the RFC 3550 fields do.
    report_packets: u32,
    report_octets: u32,
    last_rtcp_sent_ns: Option<u64>,
}

impl SenderLink {
    fn open(settings: &LinkSettings, address: &LinkAddress) -> Result<Self, G2gError> {
        let destination = address.resolve().map_err(io_err)?;
        let ip = destination.ip();
        let socket = if ip.is_multicast() {
            // the receivers of a group send their RTCP to the group
            let rtcp_port = destination.port() + 1;
            let local = rist::unspecified_address(destination, rtcp_port);
            settings.link_socket(address, local, Some(ip))
        } else {
            let local = rist::unspecified_address(destination, 0);
            settings.link_socket(address, local, None)
        }
        .map_err(io_err)?;
        Ok(Self {
            destination,
            std_socket: Some(socket),
            socket: None,
            report_packets: 0,
            report_octets: 0,
            last_rtcp_sent_ns: None,
        })
    }

    fn rtcp_destination(&self) -> SocketAddr {
        let mut destination = self.destination;
        destination.set_port(destination.port() + 1);
        destination
    }

    // the tokio socket needs the runtime, which `configure_pipeline` can run outside of
    fn socket(&mut self) -> Result<&tokio::net::UdpSocket, G2gError> {
        if self.socket.is_none() {
            let std_socket = self.std_socket.take().ok_or(G2gError::NotConfigured)?;
            self.socket = Some(tokio::net::UdpSocket::from_std(std_socket).map_err(io_err)?);
        }
        self.socket.as_ref().ok_or(G2gError::NotConfigured)
    }
}

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
    settings: LinkSettings,
    bonding_method: BondingMethod,
    extension: RistExtensionSettings,
    sender_buffer_ms: u64,
    sequence_number_offset: Option<u16>,
    ssrc: u32,
    // the low 16 bits go in the RTP header, the high 16 in the RIST header extension
    next_sequence: u32,
    links: Vec<SenderLink>,
    next_round_robin_link: usize,
    history: RetransmissionHistory,
    last_rtp_timestamp: u32,
    last_media_sent_ns: u64,
    packets_sent: u64,
    retransmissions_sent: u64,
    eos_seen: bool,
}

impl Default for RistSink {
    fn default() -> Self {
        let ssrc = rist::media_ssrc(rist::random_u32());
        Self {
            settings: LinkSettings::new(DEFAULT_ADDRESS, format!("g2g-{ssrc:08x}")),
            bonding_method: BondingMethod::Broadcast,
            extension: RistExtensionSettings::default(),
            sender_buffer_ms: DEFAULT_SENDER_BUFFER_MS,
            sequence_number_offset: None,
            ssrc,
            next_sequence: 0,
            links: Vec::new(),
            next_round_robin_link: 0,
            history: RetransmissionHistory::new(DEFAULT_SENDER_BUFFER_MS * NS_PER_MS),
            last_rtp_timestamp: 0,
            last_media_sent_ns: 0,
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

    async fn send_media(&mut self, payload: &[u8], timestamp: u32) -> Result<(), G2gError> {
        if self.links.is_empty() {
            return Err(G2gError::NotConfigured);
        }
        let header = RtpHeader {
            payload_type: MP2T_PAYLOAD_TYPE,
            marker: false,
            sequence: self.next_sequence as u16,
            timestamp,
            ssrc: self.ssrc,
        };
        let sequence_high_bits = (self.next_sequence >> u16::BITS) as u16;
        let packet = rist::build_media_packet(header, sequence_high_bits, payload, self.extension)?;
        let sent_on = match self.bonding_method {
            BondingMethod::Broadcast => SentOn::EveryLink,
            BondingMethod::RoundRobin => {
                let link = self.next_round_robin_link;
                self.next_round_robin_link = (link + 1) % self.links.len();
                SentOn::Link(link)
            }
        };
        for (index, link) in self.links.iter_mut().enumerate() {
            if !sent_on.includes(index) {
                continue;
            }
            let destination = link.destination;
            link.socket()?
                .send_to(&packet, destination)
                .await
                .map_err(io_err)?;
            link.report_packets = link.report_packets.wrapping_add(1);
            link.report_octets = link.report_octets.wrapping_add(payload.len() as u32);
        }
        let now_ns = g2g_core::metrics::monotonic_ns();
        self.history
            .record(header.sequence, packet, sent_on, now_ns);
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.last_rtp_timestamp = timestamp;
        self.last_media_sent_ns = now_ns;
        self.packets_sent += 1;
        Ok(())
    }

    /// SR + SDES, the compound every sender RTCP packet starts with, counting
    /// what went out on `link`.
    fn report(&self, link: &SenderLink, now_ns: u64) -> Vec<u8> {
        let elapsed_ns = now_ns.saturating_sub(self.last_media_sent_ns);
        let rtp_timestamp = self
            .last_rtp_timestamp
            .wrapping_add(crate::rtpklv::rtp_timestamp_from_pts(elapsed_ns));
        let mut compound = rtcp::build_sender_report(
            self.ssrc,
            rtcp::ntp_now(),
            rtp_timestamp,
            link.report_packets,
            link.report_octets,
            &[],
        );
        compound.extend(rtcp::build_sdes_cname(self.ssrc, &self.settings.cname));
        compound
    }

    async fn send_reports_if_due(&mut self) -> Result<(), G2gError> {
        let interval_ns = self.settings.rtcp_interval_ns();
        for index in 0..self.links.len() {
            let now_ns = g2g_core::metrics::monotonic_ns();
            let due = self.links[index]
                .last_rtcp_sent_ns
                .is_none_or(|last| now_ns.saturating_sub(last) >= interval_ns);
            if !due {
                continue;
            }
            let report = self.report(&self.links[index], now_ns);
            let link = &mut self.links[index];
            let destination = link.rtcp_destination();
            link.socket()?
                .send_to(&report, destination)
                .await
                .map_err(io_err)?;
            link.last_rtcp_sent_ns = Some(now_ns);
        }
        Ok(())
    }

    /// Answer one RTCP datagram that came in on link `index`: resend what its
    /// NACKs ask for on that link and echo an RTT request back to where it
    /// came from.
    async fn handle_rtcp(
        &mut self,
        index: usize,
        datagram: &[u8],
        from: SocketAddr,
    ) -> Result<(), G2gError> {
        let mut retransmissions = Vec::new();
        let mut echo_responses = Vec::new();
        for packet in rtcp::parse_compound(datagram) {
            if let Some(ranges) = rist::requested_ranges(&packet, self.ssrc) {
                retransmissions.extend(self.history.retransmissions(&ranges, index));
            } else if let Some(response) = rist::rtt_echo_response(&packet) {
                echo_responses.push(response);
            }
        }
        let now_ns = g2g_core::metrics::monotonic_ns();
        let echo_compounds: Vec<Vec<u8>> = echo_responses
            .into_iter()
            .map(|response| {
                let mut compound = self.report(&self.links[index], now_ns);
                compound.extend(response);
                compound
            })
            .collect();
        let link = &mut self.links[index];
        let destination = link.destination;
        for packet in &retransmissions {
            link.socket()?
                .send_to(packet, destination)
                .await
                .map_err(io_err)?;
        }
        for compound in &echo_compounds {
            link.socket()?
                .send_to(compound, from)
                .await
                .map_err(io_err)?;
        }
        self.retransmissions_sent += retransmissions.len() as u64;
        Ok(())
    }

    /// Read every RTCP datagram already queued on any link without waiting.
    async fn service_rtcp(&mut self) -> Result<(), G2gError> {
        let mut buffer = alloc::vec![0u8; rist::MAX_DATAGRAM];
        for index in 0..self.links.len() {
            loop {
                let received = self.links[index].socket()?.try_recv_from(&mut buffer);
                match received {
                    Ok((length, from)) => self.handle_rtcp(index, &buffer[..length], from).await?,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(error) if rist::is_peer_unreachable(&error) => continue,
                    Err(error) => return Err(io_err(error)),
                }
            }
        }
        self.send_reports_if_due().await
    }

    /// Keep answering NACKs for as long as the receiver may still ask, then
    /// say goodbye on every link.
    async fn drain_and_close(&mut self) -> Result<(), G2gError> {
        let deadline = tokio::time::Instant::now() + Duration::from_millis(self.sender_buffer_ms);
        for link in &mut self.links {
            link.socket()?;
        }
        while tokio::time::Instant::now() < deadline {
            let wait = Duration::from_millis(rist::TIMER_CHECK_MS);
            let sockets = self.links.iter().filter_map(|link| link.socket.as_ref());
            if let Ok(Err(error)) = tokio::time::timeout(wait, rist::any_readable(sockets)).await {
                return Err(io_err(error));
            }
            self.service_rtcp().await?;
        }
        for index in 0..self.links.len() {
            let mut goodbye = self.report(&self.links[index], g2g_core::metrics::monotonic_ns());
            goodbye.extend(rtcp::build_bye(self.ssrc));
            let link = &mut self.links[index];
            let destination = link.rtcp_destination();
            link.socket()?
                .send_to(&goodbye, destination)
                .await
                .map_err(io_err)?;
        }
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
        self.links = self
            .settings
            .links()
            .iter()
            .map(|address| SenderLink::open(&self.settings, address))
            .collect::<Result<_, _>>()?;
        self.next_round_robin_link = 0;
        let first_sequence = self
            .sequence_number_offset
            .unwrap_or_else(|| rist::random_u32() as u16);
        self.next_sequence = u32::from(first_sequence);
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
                "bonding-addresses",
                PropKind::Str,
                "comma separated list of <address>:<port> to send to, replacing address and port",
            )
            .with_default("localhost:5004"),
            PropertySpec::new(
                "bonding-method",
                PropKind::Str,
                "send every packet on every link (broadcast) or one link after the other",
            )
            .with_enum_values("broadcast | round-robin")
            .with_default("broadcast"),
            PropertySpec::new(
                "multicast-iface",
                PropKind::Str,
                "network interface multicast packets go out on",
            ),
            PropertySpec::new(
                "multicast-loopback",
                PropKind::Bool,
                "deliver multicast packets to receivers on this host too",
            )
            .with_default("false"),
            PropertySpec::new("multicast-ttl", PropKind::Int, "multicast time-to-live")
                .with_default("1")
                .with_range("0", "255"),
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
            PropertySpec::new(
                "drop-null-ts-packets",
                PropKind::Bool,
                "drop null MPEG-TS packets and mark them in the RIST header extension",
            )
            .with_default("false"),
            PropertySpec::new(
                "sequence-number-extension",
                PropKind::Bool,
                "carry the high 16 bits of a 32-bit sequence number in the RIST header extension",
            )
            .with_default("false"),
            PropertySpec::new(
                "seqnum-offset",
                PropKind::Int,
                "first RTP sequence number, -1 picks a random one",
            )
            .with_default("-1")
            .with_range("-1", "65535"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(result) = self.settings.set_property(name, &value) {
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
            "bonding-method" => {
                let nick = value.as_str().ok_or(PropError::Type)?;
                self.bonding_method = BondingMethod::from_nick(nick).ok_or(PropError::Value)?;
                Ok(())
            }
            "drop-null-ts-packets" => {
                self.extension.drop_null_packets = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "sequence-number-extension" => {
                self.extension.sequence_number_extension =
                    value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "seqnum-offset" => {
                let offset = value.as_int().ok_or(PropError::Type)?;
                self.sequence_number_offset = if offset == RANDOM_SEQUENCE_NUMBER_OFFSET {
                    None
                } else {
                    Some(u16::try_from(offset).map_err(|_| PropError::Value)?)
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = self.settings.get_property(name) {
            return Some(value);
        }
        match name {
            "sender-buffer" => Some(PropValue::Uint(self.sender_buffer_ms)),
            "bonding-method" => Some(PropValue::Str(self.bonding_method.nick().into())),
            "drop-null-ts-packets" => Some(PropValue::Bool(self.extension.drop_null_packets)),
            "sequence-number-extension" => {
                Some(PropValue::Bool(self.extension.sequence_number_extension))
            }
            "seqnum-offset" => Some(PropValue::Int(
                self.sequence_number_offset
                    .map_or(RANDOM_SEQUENCE_NUMBER_OFFSET, i64::from),
            )),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        // librist reads its sockets every 5 ms, so two periods put the report in an earlier pass
        const FIRST_REPORT_LEAD: Duration = Duration::from_millis(10);
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let bytes = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let timestamp = crate::rtpklv::rtp_timestamp_from_pts(frame.timing.pts_ns);
                    // librist's `ristreceiver` drops media that comes before the first sender report
                    if self.packets_sent == 0 {
                        self.send_reports_if_due().await?;
                        tokio::time::sleep(FIRST_REPORT_LEAD).await;
                    }
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

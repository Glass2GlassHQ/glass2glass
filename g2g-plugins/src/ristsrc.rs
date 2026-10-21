//! RIST receiver (RistSrc, `rist` feature), VSF TR-06-1 Simple Profile. It
//! listens for RTP MP2T on `address`:`port` and RTCP on `port + 1`, holds every
//! packet `receiver-buffer` ms and releases the MPEG-TS payloads in order. A gap
//! older than `reorder-section` is NACKed, in whichever RIST form is shorter, up
//! to `max-rtx-retries` times spread over the buffer. A retransmission (SSRC low
//! bit set) fills its gap. RR + SDES go back to the source address of the
//! sender's last RTCP packet, and the stream ends once the sender's BYE has
//! been drained. The GStreamer equivalent is `ristsrc ! rtpmp2tdepay`.
//!
//! `bonding-addresses` lists several links, each with its own sockets and
//! RTCP. Their packets merge into one stream, the copies of a sequence number
//! after the first dropped, and NACKs go out on every link, as in GStreamer. A
//! multicast address joins the group and sends RTCP to it. The TR-06-2 RTP
//! header extension comes off every packet and the null packets it deleted go
//! back in, as GStreamer's `ristrtpdeext` inside its `ristsrc`.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

use std::net::{SocketAddr, UdpSocket as StdUdpSocket};

use g2g_core::rtp::RtpHeader;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError, LatencyReport,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::bytestream::{byte_frame, mpegts_caps};
use crate::filesink::io_err;
use crate::rist::{
    self, LinkAddress, LinkSettings, NackScheduler, RistExtensionRemover, MP2T_PAYLOAD_TYPE,
};
use crate::rtcp::{self, ReceptionStats, RtcpPacket};
use crate::rtpjitter::{JitterConfig, RtpJitterBuffer};

/// GStreamer's `ristsrc` listens on every interface by default.
const DEFAULT_ADDRESS: &str = "0.0.0.0";
const DEFAULT_RECEIVER_BUFFER_MS: u64 = 1000;
const DEFAULT_REORDER_SECTION_MS: u64 = 70;
const DEFAULT_MAX_RTX_RETRIES: u32 = 7;
/// Media clock of RTP MP2T.
const MP2T_CLOCK_HZ: u32 = 90_000;
/// Bound on packets held, so a flood cannot grow the buffer without limit.
const MAX_BUFFERED_PACKETS: usize = 1 << 15;
/// Sequence numbers one NACK may name, so the compound stays under an MTU.
const MAX_NACKED_PER_REPORT: usize = 256;
/// Holes scheduled at once, the oldest first, so one forged far-ahead sequence
/// number cannot make every check walk the whole buffer.
const MAX_TRACKED_LOSSES: usize = 4096;
const NS_PER_MS: u64 = 1_000_000;

/// # Example
///
/// ```no_run
/// use g2g_core::runtime::SourceLoop;
/// use g2g_core::PropValue;
/// use g2g_plugins::ristsrc::RistSrc;
///
/// let mut source = RistSrc::default();
/// source.set_property("port", PropValue::Uint(5004)).unwrap();
/// source.set_property("receiver-buffer", PropValue::Uint(500)).unwrap();
/// ```
#[derive(Debug)]
pub struct RistSrc {
    settings: LinkSettings,
    receiver_buffer_ms: u64,
    reorder_section_ms: u64,
    max_rtx_retries: u32,
    ssrc: u32,
    bound_links: Vec<BoundLink>,
}

impl Default for RistSrc {
    fn default() -> Self {
        let ssrc = rist::random_u32();
        Self {
            settings: LinkSettings::new(DEFAULT_ADDRESS, format!("g2g-{ssrc:08x}")),
            receiver_buffer_ms: DEFAULT_RECEIVER_BUFFER_MS,
            reorder_section_ms: DEFAULT_REORDER_SECTION_MS,
            max_rtx_retries: DEFAULT_MAX_RTX_RETRIES,
            ssrc,
            bound_links: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum RtcpDestination {
    Group(SocketAddr),
    // the source of the sender's last valid RTCP packet on the link
    LastSender(Option<SocketAddr>),
}

impl RtcpDestination {
    fn address(self) -> Option<SocketAddr> {
        match self {
            RtcpDestination::Group(group) => Some(group),
            RtcpDestination::LastSender(sender) => sender,
        }
    }

    fn on_sender_rtcp(&mut self, from: SocketAddr) {
        if let RtcpDestination::LastSender(sender) = self {
            *sender = Some(from);
        }
    }
}

#[derive(Debug)]
struct BoundLink {
    media_socket: StdUdpSocket,
    rtcp_socket: StdUdpSocket,
    rtcp_destination: RtcpDestination,
}

impl BoundLink {
    fn bind(settings: &LinkSettings, address: &LinkAddress) -> Result<Self, G2gError> {
        let media_address = address.resolve().map_err(io_err)?;
        let rtcp_port = media_address.port() + 1;
        let ip = media_address.ip();
        if ip.is_multicast() {
            let media_local = rist::unspecified_address(media_address, media_address.port());
            let rtcp_local = rist::unspecified_address(media_address, rtcp_port);
            return Ok(Self {
                media_socket: settings
                    .link_socket(address, media_local, Some(ip))
                    .map_err(io_err)?,
                rtcp_socket: settings
                    .link_socket(address, rtcp_local, Some(ip))
                    .map_err(io_err)?,
                rtcp_destination: RtcpDestination::Group(SocketAddr::new(ip, rtcp_port)),
            });
        }
        let mut rtcp_address = media_address;
        rtcp_address.set_port(rtcp_port);
        Ok(Self {
            media_socket: settings
                .link_socket(address, media_address, None)
                .map_err(io_err)?,
            rtcp_socket: settings
                .link_socket(address, rtcp_address, None)
                .map_err(io_err)?,
            rtcp_destination: RtcpDestination::LastSender(None),
        })
    }

    fn start(self) -> Result<ReceiverLink, G2gError> {
        Ok(ReceiverLink {
            media_socket: tokio::net::UdpSocket::from_std(self.media_socket).map_err(io_err)?,
            rtcp_socket: tokio::net::UdpSocket::from_std(self.rtcp_socket).map_err(io_err)?,
            statistics: None,
            rtcp_destination: self.rtcp_destination,
        })
    }
}

#[derive(Debug)]
struct ReceiverLink {
    media_socket: tokio::net::UdpSocket,
    rtcp_socket: tokio::net::UdpSocket,
    // what arrived on this link, from its first original packet on
    statistics: Option<ReceptionStats>,
    rtcp_destination: RtcpDestination,
}

/// What the receive loop knows about the one stream it serves, whichever link
/// a packet came in on.
#[derive(Debug)]
struct StreamState {
    /// Original-packet SSRC, taken from the first original packet.
    media_ssrc: Option<u32>,
    said_goodbye: bool,
    extension_remover: RistExtensionRemover,
}

impl RistSrc {
    fn receiver_buffer_ns(&self) -> u64 {
        self.receiver_buffer_ms * NS_PER_MS
    }

    /// RR (with a report block once the link has seen the stream) + SDES.
    fn report(&self, statistics: &mut Option<ReceptionStats>, now_ns: u64) -> Vec<u8> {
        let blocks: Vec<rtcp::ReportBlock> = statistics
            .as_mut()
            .map(|statistics| statistics.report_block(now_ns))
            .into_iter()
            .collect();
        let mut compound = rtcp::build_receiver_report(self.ssrc, &blocks);
        compound.extend(rtcp::build_sdes_cname(self.ssrc, &self.settings.cname));
        compound
    }
}

/// Buffer one media datagram if it is RTP MP2T from the stream being received.
/// An original packet adopts the stream on first sight, a retransmission only
/// fills a gap in it. A copy of a sequence number already buffered, from
/// another bonded link, is dropped by the jitter buffer.
fn accept_media(
    datagram: &[u8],
    now_ns: u64,
    stream: &mut StreamState,
    statistics: &mut Option<ReceptionStats>,
    jitter: &mut RtpJitterBuffer,
) {
    let Some(parsed) = RtpHeader::parse(datagram) else {
        return;
    };
    let header = parsed.header;
    if header.payload_type != MP2T_PAYLOAD_TYPE {
        return;
    }
    let media_ssrc = rist::media_ssrc(header.ssrc);
    let retransmission = rist::is_retransmission(header.ssrc);
    match stream.media_ssrc {
        None if retransmission => return,
        None => stream.media_ssrc = Some(media_ssrc),
        Some(known) if known != media_ssrc => return,
        Some(_) => {}
    }
    let Some(stripped) = stream.extension_remover.strip(datagram) else {
        return;
    };
    if !retransmission {
        statistics
            .get_or_insert_with(|| ReceptionStats::new(0, MP2T_CLOCK_HZ))
            .on_rtp(header.ssrc, header.sequence, header.timestamp, now_ns);
    }
    jitter.push(&stripped, now_ns);
}

impl SourceLoop for RistSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(mpegts_caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(mpegts_caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.bound_links = self
            .settings
            .links()
            .iter()
            .map(|address| BoundLink::bind(&self.settings, address))
            .collect::<Result<_, _>>()?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn latency(&self) -> LatencyReport {
        LatencyReport::live(self.receiver_buffer_ns(), None)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "RIST source",
            "Source/Network",
            "Receives an MPEG-TS byte stream over RIST (TR-06-1 Simple Profile), NACK-recovering loss",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "address",
                PropKind::Str,
                "address to receive packets on (IPv4, IPv6 or a host name)",
            )
            .with_default(DEFAULT_ADDRESS),
            PropertySpec::new(
                "port",
                PropKind::Uint,
                "RTP port to listen on, RTCP uses this value + 1; must be even",
            )
            .with_default("5004")
            .with_range("2", "65534"),
            PropertySpec::new(
                "bonding-addresses",
                PropKind::Str,
                "comma separated list of <address>:<port> to receive on, replacing address and port",
            )
            .with_default("0.0.0.0:5004"),
            PropertySpec::new(
                "multicast-iface",
                PropKind::Str,
                "network interface the multicast group is joined on",
            ),
            PropertySpec::new(
                "multicast-loopback",
                PropKind::Bool,
                "deliver the RTCP sent to the multicast group to senders on this host too",
            )
            .with_default("false"),
            PropertySpec::new(
                "multicast-ttl",
                PropKind::Int,
                "time-to-live of the RTCP sent to the multicast group",
            )
            .with_default("1")
            .with_range("0", "255"),
            PropertySpec::new("receiver-buffer", PropKind::Uint, "buffering duration, ms")
                .with_default("1000")
                .with_range("0", "4294967295"),
            PropertySpec::new(
                "reorder-section",
                PropKind::Uint,
                "time to wait before requesting a retransmission, ms",
            )
            .with_default("70")
            .with_range("0", "4294967295"),
            PropertySpec::new(
                "max-rtx-retries",
                PropKind::Uint,
                "maximum number of retransmission requests for a lost packet",
            )
            .with_default("7")
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
                "CNAME in the SDES block of the receiver report",
            ),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(result) = self.settings.set_property(name, &value) {
            return result;
        }
        let target = match name {
            "receiver-buffer" => &mut self.receiver_buffer_ms,
            "reorder-section" => &mut self.reorder_section_ms,
            "max-rtx-retries" => {
                let retries = value.as_uint().ok_or(PropError::Type)?;
                self.max_rtx_retries = u32::try_from(retries).map_err(|_| PropError::Value)?;
                return Ok(());
            }
            _ => return Err(PropError::Unknown),
        };
        let ms = value.as_uint().ok_or(PropError::Type)?;
        if ms > u64::from(u32::MAX) {
            return Err(PropError::Value);
        }
        *target = ms;
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = self.settings.get_property(name) {
            return Some(value);
        }
        match name {
            "receiver-buffer" => Some(PropValue::Uint(self.receiver_buffer_ms)),
            "reorder-section" => Some(PropValue::Uint(self.reorder_section_ms)),
            "max-rtx-retries" => Some(PropValue::Uint(u64::from(self.max_rtx_retries))),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let bound_links = core::mem::take(&mut self.bound_links);
            if bound_links.is_empty() {
                return Err(G2gError::NotConfigured);
            }
            let mut links = bound_links
                .into_iter()
                .map(BoundLink::start)
                .collect::<Result<Vec<_>, _>>()?;

            out.push(PipelinePacket::CapsChanged(mpegts_caps())).await?;

            let buffer_ns = self.receiver_buffer_ns();
            let mut jitter = RtpJitterBuffer::new(JitterConfig {
                max_hold_ns: buffer_ns,
                max_depth: MAX_BUFFERED_PACKETS,
            })
            .with_constant_delay();
            let mut scheduler = NackScheduler::new(
                self.reorder_section_ms * NS_PER_MS,
                buffer_ns,
                self.max_rtx_retries,
            );
            let mut stream = StreamState {
                media_ssrc: None,
                said_goodbye: false,
                extension_remover: RistExtensionRemover::default(),
            };
            let timer_check = Duration::from_millis(rist::TIMER_CHECK_MS);
            let timer_check_ns = rist::TIMER_CHECK_MS * NS_PER_MS;
            let mut datagram = alloc::vec![0u8; rist::MAX_DATAGRAM];
            let mut last_report_ns: Option<u64> = None;
            let mut next_nack_check_ns = 0u64;
            let mut emitted = 0u64;
            loop {
                let media_sockets = links.iter().map(|link| &link.media_socket);
                let readable =
                    tokio::time::timeout(timer_check, rist::any_readable(media_sockets)).await;
                if let Ok(Err(error)) = readable {
                    return Err(io_err(error));
                }
                let now_ns = g2g_core::metrics::monotonic_ns();
                // one datagram per link per pass, so a busy link cannot starve the others
                for link in links.iter_mut() {
                    match link.media_socket.try_recv_from(&mut datagram) {
                        Ok((length, _)) => accept_media(
                            &datagram[..length],
                            now_ns,
                            &mut stream,
                            &mut link.statistics,
                            &mut jitter,
                        ),
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                        Err(error) if rist::is_peer_unreachable(&error) => {}
                        Err(error) => return Err(io_err(error)),
                    }
                }

                for link in links.iter_mut() {
                    loop {
                        let (length, from) = match link.rtcp_socket.try_recv_from(&mut datagram) {
                            Ok(received) => received,
                            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                            Err(error) if rist::is_peer_unreachable(&error) => continue,
                            Err(error) => return Err(io_err(error)),
                        };
                        let packets = rtcp::parse_compound(&datagram[..length]);
                        if packets.is_empty() {
                            continue;
                        }
                        link.rtcp_destination.on_sender_rtcp(from);
                        for packet in &packets {
                            match packet {
                                RtcpPacket::SenderReport { ssrc, ntp, .. }
                                    if stream.media_ssrc == Some(*ssrc) =>
                                {
                                    if let Some(statistics) = link.statistics.as_mut() {
                                        statistics.on_sender_report(*ntp, now_ns);
                                    }
                                }
                                RtcpPacket::Bye { ssrc } => {
                                    let known = stream.media_ssrc;
                                    if known.is_some_and(|media| ssrc.contains(&media)) {
                                        stream.said_goodbye = true;
                                    }
                                }
                                _ => {
                                    if let Some(response) = rist::rtt_echo_response(packet) {
                                        let mut compound =
                                            self.report(&mut link.statistics, now_ns);
                                        compound.extend(response);
                                        link.rtcp_socket
                                            .send_to(&compound, from)
                                            .await
                                            .map_err(io_err)?;
                                    }
                                }
                            }
                        }
                    }
                }

                while let Some(packet) = jitter.pop(now_ns) {
                    let Some(parsed) = RtpHeader::parse(&packet) else {
                        continue;
                    };
                    let payload_end = parsed.payload_offset + parsed.payload_len;
                    let payload = packet[parsed.payload_offset..payload_end].to_vec();
                    out.push(PipelinePacket::DataFrame(byte_frame(payload, emitted)))
                        .await?;
                    emitted += 1;
                }
                if stream.said_goodbye && jitter.buffered() == 0 {
                    out.push(PipelinePacket::Eos).await?;
                    return Ok(emitted);
                }

                let no_rtcp_destination = links
                    .iter()
                    .all(|link| link.rtcp_destination.address().is_none());
                if no_rtcp_destination {
                    continue;
                }
                let mut nack = Vec::new();
                if now_ns >= next_nack_check_ns {
                    next_nack_check_ns = now_ns + timer_check_ns;
                    let missing = jitter.missing_seqs();
                    let tracked = &missing[..missing.len().min(MAX_TRACKED_LOSSES)];
                    let requests = scheduler.due(tracked, now_ns, MAX_NACKED_PER_REPORT);
                    if let (Some(media_ssrc), false) = (stream.media_ssrc, requests.is_empty()) {
                        nack = rist::build_nack_feedback(self.ssrc, media_ssrc, &requests);
                    }
                }
                let report_due = last_report_ns.is_none_or(|last| {
                    now_ns.saturating_sub(last) >= self.settings.rtcp_interval_ns()
                });
                if !report_due && nack.is_empty() {
                    continue;
                }
                // the NACK goes out on every link, whichever one carried the lost packet
                for link in links.iter_mut() {
                    let Some(destination) = link.rtcp_destination.address() else {
                        continue;
                    };
                    let mut compound = self.report(&mut link.statistics, now_ns);
                    compound.extend_from_slice(&nack);
                    link.rtcp_socket
                        .send_to(&compound, destination)
                        .await
                        .map_err(io_err)?;
                }
                last_report_ns = Some(now_ns);
            }
        })
    }
}

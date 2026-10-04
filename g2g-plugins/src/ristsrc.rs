//! RIST receiver (RistSrc, `rist` feature), VSF TR-06-1 Simple Profile. It
//! listens for RTP MP2T on `address`:`port` and RTCP on `port + 1`, holds every
//! packet `receiver-buffer` ms and releases the MPEG-TS payloads in order. A gap
//! older than `reorder-section` is NACKed, in whichever RIST form is shorter, up
//! to `max-rtx-retries` times spread over the buffer. A retransmission (SSRC low
//! bit set) fills its gap. RR + SDES go back to the source address of the
//! sender's last RTCP packet, and the stream ends once the sender's BYE has
//! been drained. The GStreamer equivalent is `ristsrc ! rtpmp2tdepay`.

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
use crate::rist::{self, LinkSettings, NackScheduler, MP2T_PAYLOAD_TYPE};
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
    link: LinkSettings,
    receiver_buffer_ms: u64,
    reorder_section_ms: u64,
    max_rtx_retries: u32,
    ssrc: u32,
    media_socket: Option<StdUdpSocket>,
    rtcp_socket: Option<StdUdpSocket>,
}

impl Default for RistSrc {
    fn default() -> Self {
        let ssrc = rist::random_u32();
        Self {
            link: LinkSettings::new(DEFAULT_ADDRESS, format!("g2g-{ssrc:08x}")),
            receiver_buffer_ms: DEFAULT_RECEIVER_BUFFER_MS,
            reorder_section_ms: DEFAULT_REORDER_SECTION_MS,
            max_rtx_retries: DEFAULT_MAX_RTX_RETRIES,
            ssrc,
            media_socket: None,
            rtcp_socket: None,
        }
    }
}

/// What the receive loop knows about the one sender it serves.
#[derive(Debug)]
struct SenderState {
    /// Original-packet SSRC, taken from the first original packet.
    media_ssrc: Option<u32>,
    statistics: ReceptionStats,
    /// Where RTCP goes: the source of the sender's last valid RTCP packet.
    rtcp_address: Option<SocketAddr>,
    said_goodbye: bool,
}

impl RistSrc {
    fn receiver_buffer_ns(&self) -> u64 {
        self.receiver_buffer_ms * NS_PER_MS
    }

    /// RR (with a report block once the stream is known) + SDES.
    fn report(&self, sender: &mut SenderState, now_ns: u64) -> Vec<u8> {
        let blocks: Vec<rtcp::ReportBlock> = sender
            .media_ssrc
            .map(|_| sender.statistics.report_block(now_ns))
            .into_iter()
            .collect();
        let mut compound = rtcp::build_receiver_report(self.ssrc, &blocks);
        compound.extend(rtcp::build_sdes_cname(self.ssrc, &self.link.cname));
        compound
    }
}

/// Buffer one media datagram if it is RTP MP2T from the stream being received.
/// An original packet adopts the stream on first sight, a retransmission only
/// fills a gap in it.
fn accept_media(
    datagram: &[u8],
    now_ns: u64,
    sender: &mut SenderState,
    jitter: &mut RtpJitterBuffer,
) {
    let Some(parsed) = RtpHeader::parse(datagram) else {
        return;
    };
    let header = parsed.header;
    if header.payload_type != MP2T_PAYLOAD_TYPE {
        return;
    }
    let stream = rist::media_ssrc(header.ssrc);
    let retransmission = rist::is_retransmission(header.ssrc);
    match sender.media_ssrc {
        None if retransmission => return,
        None => sender.media_ssrc = Some(stream),
        Some(known) if known != stream => return,
        Some(_) => {}
    }
    if !retransmission {
        sender
            .statistics
            .on_rtp(header.ssrc, header.sequence, header.timestamp, now_ns);
    }
    jitter.push(datagram, now_ns);
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
        let media_address = self.link.media_address().map_err(io_err)?;
        let mut rtcp_address = media_address;
        rtcp_address.set_port(media_address.port() + 1);
        let media_socket = StdUdpSocket::bind(media_address).map_err(io_err)?;
        let rtcp_socket = StdUdpSocket::bind(rtcp_address).map_err(io_err)?;
        media_socket.set_nonblocking(true).map_err(io_err)?;
        rtcp_socket.set_nonblocking(true).map_err(io_err)?;
        self.media_socket = Some(media_socket);
        self.rtcp_socket = Some(rtcp_socket);
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
        if let Some(result) = self.link.set_property(name, &value) {
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
        if let Some(value) = self.link.get_property(name) {
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
            let media_socket = self.media_socket.take().ok_or(G2gError::NotConfigured)?;
            let rtcp_socket = self.rtcp_socket.take().ok_or(G2gError::NotConfigured)?;
            let media_socket = tokio::net::UdpSocket::from_std(media_socket).map_err(io_err)?;
            let rtcp_socket = tokio::net::UdpSocket::from_std(rtcp_socket).map_err(io_err)?;

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
            let mut sender = SenderState {
                media_ssrc: None,
                statistics: ReceptionStats::new(0, MP2T_CLOCK_HZ),
                rtcp_address: None,
                said_goodbye: false,
            };
            let timer_check = Duration::from_millis(rist::TIMER_CHECK_MS);
            let timer_check_ns = rist::TIMER_CHECK_MS * NS_PER_MS;
            let mut datagram = alloc::vec![0u8; rist::MAX_DATAGRAM];
            let mut last_report_ns: Option<u64> = None;
            let mut next_nack_check_ns = 0u64;
            let mut emitted = 0u64;
            loop {
                let received =
                    tokio::time::timeout(timer_check, media_socket.recv_from(&mut datagram)).await;
                let now_ns = g2g_core::metrics::monotonic_ns();
                match received {
                    Ok(Ok((length, _))) => {
                        accept_media(&datagram[..length], now_ns, &mut sender, &mut jitter)
                    }
                    Ok(Err(error)) if !rist::is_peer_unreachable(&error) => {
                        return Err(io_err(error))
                    }
                    _ => {}
                }

                loop {
                    let (length, from) = match rtcp_socket.try_recv_from(&mut datagram) {
                        Ok(received) => received,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                        Err(error) if rist::is_peer_unreachable(&error) => continue,
                        Err(error) => return Err(io_err(error)),
                    };
                    let packets = rtcp::parse_compound(&datagram[..length]);
                    if packets.is_empty() {
                        continue;
                    }
                    sender.rtcp_address = Some(from);
                    for packet in &packets {
                        match packet {
                            RtcpPacket::SenderReport { ssrc, ntp, .. }
                                if sender.media_ssrc == Some(*ssrc) =>
                            {
                                sender.statistics.on_sender_report(*ntp, now_ns);
                            }
                            RtcpPacket::Bye { ssrc } => {
                                let known = sender.media_ssrc;
                                if known.is_some_and(|media| ssrc.contains(&media)) {
                                    sender.said_goodbye = true;
                                }
                            }
                            _ => {
                                if let Some(response) = rist::rtt_echo_response(packet) {
                                    let mut compound = self.report(&mut sender, now_ns);
                                    compound.extend(response);
                                    rtcp_socket.send_to(&compound, from).await.map_err(io_err)?;
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
                if sender.said_goodbye && jitter.buffered() == 0 {
                    out.push(PipelinePacket::Eos).await?;
                    return Ok(emitted);
                }

                let Some(rtcp_address) = sender.rtcp_address else {
                    continue;
                };
                let mut nack = Vec::new();
                if now_ns >= next_nack_check_ns {
                    next_nack_check_ns = now_ns + timer_check_ns;
                    let missing = jitter.missing_seqs();
                    let tracked = &missing[..missing.len().min(MAX_TRACKED_LOSSES)];
                    let requests = scheduler.due(tracked, now_ns, MAX_NACKED_PER_REPORT);
                    if let (Some(media_ssrc), false) = (sender.media_ssrc, requests.is_empty()) {
                        nack = rist::build_nack_feedback(self.ssrc, media_ssrc, &requests);
                    }
                }
                let report_due = last_report_ns
                    .is_none_or(|last| now_ns.saturating_sub(last) >= self.link.rtcp_interval_ns());
                if !report_due && nack.is_empty() {
                    continue;
                }
                let mut compound = self.report(&mut sender, now_ns);
                compound.extend(nack);
                rtcp_socket
                    .send_to(&compound, rtcp_address)
                    .await
                    .map_err(io_err)?;
                last_report_ns = Some(now_ns);
            }
        })
    }
}

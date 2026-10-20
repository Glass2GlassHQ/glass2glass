use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::net::{IpAddr, SocketAddr};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, UdpSocket};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use g2g_core::log::{io_err, short_type_name, LogName, LogSource};
use g2g_core::mediaclock::MediaClock;
use g2g_core::metrics::{monotonic_ns, unix_time_ns};
use g2g_core::{
    g2g_info, g2g_warn, ntp_from_unix_nanos, AudioFormat, Caps, CapsConstraint, ConfigureOutcome,
    Dim, ElementMetadata, G2gError, HardwareError, MultiInputElement, OutputSink, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, VideoCodec,
};

use crate::aacparse::{
    audio_specific_config_info, is_leading_aac_config, muxer_access_unit,
    muxer_audio_specific_config,
};
use crate::annexb::{au_is_keyframe, parameter_sets, split_annexb};
use crate::opusparse::{is_opus_config, parse_opus_head, parse_toc};
use crate::rtcp::{build_sdes_cname, SenderCounters};
use crate::rtpklv::rtp_timestamp_at_clock;
use crate::rtppay::{RtpAacPacketizer, RtpH264Packetizer, RtpH265Packetizer, RtpOpusPacketizer};
use crate::rtspclient::{
    keepalive_interval_s, record_transport_tcp, record_transport_udp, transports_from_flags,
    transports_to_flags, Credentials, LowerTransport, ReconnectPolicy, ReconnectState,
    RtspClientError, RtspPublishRequests, RtspResponse, DEFAULT_INITIAL_BACKOFF_MS,
    DEFAULT_MAX_BACKOFF_MS, STATUS_OK, STATUS_UNAUTHORIZED,
};
use crate::rtspserver::{
    interleaved_frame, parse_interleaved_channels, parse_rtp_port, push_interleaved,
    INTERLEAVED_HEADER_LEN, INTERLEAVED_MARKER,
};
use crate::sdp::{
    aac_media_section, h264_media_section, h265_media_section, opus_media_section,
    session_description, stream_control, OPUS_CLOCK_HZ,
};

const DEFAULT_USER_AGENT: &str = concat!("g2g/", env!("CARGO_PKG_VERSION"));
const DEFAULT_RTSP_PORT: u16 = 554;
const DEFAULT_TCP_TIMEOUT_US: u64 = 20_000_000;
const FIRST_DYNAMIC_PAYLOAD_TYPE: u8 = 96;
const LAST_DYNAMIC_PAYLOAD_TYPE: u8 = 127;
const MAX_OPUS_CHANNELS: u8 = 2;
// frames held over all pads while the ANNOUNCE waits for every pad's first frame
const MAX_HELD_FRAMES: usize = 256;
const READ_BUF_LEN: usize = 8192;
// a response still incomplete at this size never completes
const MAX_PENDING: usize = 64 * 1024;
// as gst's `retry` default
const UDP_PORT_PAIR_ATTEMPTS: usize = 20;
// RFC 3550's nominal sender-report interval, the gst rtpsession default
const DEFAULT_REPORT_INTERVAL_MS: u64 = 5_000;
const MAX_REPORT_INTERVAL_MS: u64 = 60_000;
const NS_PER_MS: u64 = 1_000_000;
const NS_PER_SECOND: u64 = 1_000_000_000;
// gst's margin leaves nothing of a one-second timeout, and a zero interval sends keepalives back to back
const MIN_KEEPALIVE_INTERVAL_S: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamCodec {
    H264,
    H265,
    Aac,
    // zero when the caps leave the count to the stream
    Opus { channels: u8 },
}

impl StreamCodec {
    fn from_caps(caps: &Caps) -> Option<Self> {
        match caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            } => Some(StreamCodec::H264),
            Caps::CompressedVideo {
                codec: VideoCodec::H265,
                ..
            } => Some(StreamCodec::H265),
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            } => Some(StreamCodec::Aac),
            Caps::Audio {
                format: AudioFormat::Opus,
                channels,
                ..
            } if *channels <= MAX_OPUS_CHANNELS => Some(StreamCodec::Opus {
                channels: *channels,
            }),
            _ => None,
        }
    }
}

// what the ANNOUNCE describes for one stream, read from its first frame
#[derive(Debug, Clone)]
enum StreamConfig {
    H264 {
        sps: Vec<u8>,
        pps: Vec<u8>,
    },
    H265 {
        vps: Vec<u8>,
        sps: Vec<u8>,
        pps: Vec<u8>,
    },
    Aac {
        audio_specific_config: Vec<u8>,
        sample_rate: u32,
        channels: u8,
    },
    Opus {
        channels: u8,
    },
}

impl StreamConfig {
    fn from_first_frame(codec: StreamCodec, frame: &[u8]) -> Result<Self, G2gError> {
        let category = short_type_name::<RtspClientSink>();
        match codec {
            StreamCodec::H264 => {
                let sets = parameter_sets(VideoCodec::H264, &split_annexb(frame))?;
                Ok(StreamConfig::H264 {
                    sps: sets[0].to_vec(),
                    pps: sets[1].to_vec(),
                })
            }
            StreamCodec::H265 => {
                let sets = parameter_sets(VideoCodec::H265, &split_annexb(frame))?;
                Ok(StreamConfig::H265 {
                    vps: sets[0].to_vec(),
                    sps: sets[1].to_vec(),
                    pps: sets[2].to_vec(),
                })
            }
            StreamCodec::Aac => {
                let audio_specific_config =
                    muxer_audio_specific_config(frame, category)?.ok_or(G2gError::CapsMismatch)?;
                let info = audio_specific_config_info(&audio_specific_config)
                    .ok_or(G2gError::CapsMismatch)?;
                Ok(StreamConfig::Aac {
                    audio_specific_config,
                    sample_rate: info.sample_rate,
                    channels: info.channels,
                })
            }
            StreamCodec::Opus { channels } if channels != 0 => Ok(StreamConfig::Opus { channels }),
            // a demuxer leaves the count unknown, a leading OpusHead or the TOC byte says it
            StreamCodec::Opus { .. } => {
                let channels = if is_opus_config(frame) {
                    parse_opus_head(frame).map(|(channels, _)| channels)
                } else {
                    parse_toc(frame).map(|toc| toc.channels)
                };
                let channels = channels
                    .filter(|channels| *channels <= MAX_OPUS_CHANNELS)
                    .ok_or(G2gError::CapsMismatch)?;
                Ok(StreamConfig::Opus { channels })
            }
        }
    }

    // a demuxed stream's leading config frame describes the stream and is not sent as media
    fn is_config_frame(&self, frame: &[u8]) -> bool {
        match self {
            StreamConfig::Aac {
                audio_specific_config,
                ..
            } => is_leading_aac_config(audio_specific_config, frame),
            StreamConfig::Opus { .. } => is_opus_config(frame),
            StreamConfig::H264 { .. } | StreamConfig::H265 { .. } => false,
        }
    }

    fn media_section(&self, payload_type: u8, stream_id: usize) -> String {
        match self {
            StreamConfig::H264 { sps, pps } => {
                h264_media_section(payload_type, Some(sps), Some(pps), stream_id)
            }
            StreamConfig::H265 { vps, sps, pps } => {
                h265_media_section(payload_type, vps, sps, pps, stream_id)
            }
            StreamConfig::Aac {
                audio_specific_config,
                sample_rate,
                channels,
            } => aac_media_section(
                payload_type,
                audio_specific_config,
                *sample_rate,
                *channels,
                stream_id,
            ),
            StreamConfig::Opus { channels } => {
                opus_media_section(payload_type, *channels, stream_id)
            }
        }
    }

    fn video_codec(&self) -> Option<VideoCodec> {
        match self {
            StreamConfig::H264 { .. } => Some(VideoCodec::H264),
            StreamConfig::H265 { .. } => Some(VideoCodec::H265),
            StreamConfig::Aac { .. } | StreamConfig::Opus { .. } => None,
        }
    }

    fn clock_rate(&self) -> u32 {
        match self {
            StreamConfig::H264 { .. } | StreamConfig::H265 { .. } => MediaClock::VIDEO_RATE_HZ,
            StreamConfig::Aac { sample_rate, .. } => *sample_rate,
            StreamConfig::Opus { .. } => OPUS_CLOCK_HZ,
        }
    }

    fn packetizer(&self, payload_type: u8, ssrc: u32) -> StreamPacketizer {
        match self {
            StreamConfig::H264 { .. } => {
                StreamPacketizer::H264(RtpH264Packetizer::new(payload_type, ssrc))
            }
            StreamConfig::H265 { .. } => {
                StreamPacketizer::H265(RtpH265Packetizer::new(payload_type, ssrc))
            }
            StreamConfig::Aac {
                audio_specific_config,
                ..
            } => StreamPacketizer::Aac {
                packetizer: RtpAacPacketizer::new(payload_type, ssrc),
                audio_specific_config: audio_specific_config.clone(),
            },
            StreamConfig::Opus { .. } => {
                StreamPacketizer::Opus(RtpOpusPacketizer::new(payload_type, ssrc))
            }
        }
    }
}

#[derive(Debug)]
enum StreamPacketizer {
    H264(RtpH264Packetizer),
    H265(RtpH265Packetizer),
    Aac {
        packetizer: RtpAacPacketizer,
        audio_specific_config: Vec<u8>,
    },
    Opus(RtpOpusPacketizer),
}

impl StreamPacketizer {
    fn packetize(
        &mut self,
        frame: &[u8],
        rtp_timestamp: u32,
    ) -> Result<Vec<Vec<u8>>, PublishError> {
        match self {
            StreamPacketizer::H264(p) => Ok(p.packetize(frame, rtp_timestamp)),
            StreamPacketizer::H265(p) => Ok(p.packetize(frame, rtp_timestamp)),
            StreamPacketizer::Aac {
                packetizer,
                audio_specific_config,
            } => packetizer
                .packetize(
                    muxer_access_unit(audio_specific_config, frame),
                    rtp_timestamp,
                )
                .ok_or(PublishError::AccessUnitTooLarge),
            StreamPacketizer::Opus(p) => Ok(p.packetize(frame, rtp_timestamp)),
        }
    }
}

#[derive(Debug)]
enum PublishError {
    Rtsp(RtspClientError),
    Io(std::io::Error),
    InvalidLocation,
    CredentialsInLocation,
    TooManyStreams,
    NoUdpPortPair,
    NoTransports,
    TimedOut,
    Closed,
    AccessUnitTooLarge,
    TimestampOutsideNtpRange,
}

impl PublishError {
    // errors from the element's own settings or input repeat on every new session
    fn is_transient(&self) -> bool {
        !matches!(
            self,
            PublishError::InvalidLocation
                | PublishError::CredentialsInLocation
                | PublishError::TooManyStreams
                | PublishError::NoTransports
                | PublishError::AccessUnitTooLarge
                | PublishError::TimestampOutsideNtpRange
        )
    }
}

impl From<RtspClientError> for PublishError {
    fn from(e: RtspClientError) -> Self {
        PublishError::Rtsp(e)
    }
}

impl From<std::io::Error> for PublishError {
    fn from(e: std::io::Error) -> Self {
        PublishError::Io(e)
    }
}

impl core::fmt::Display for PublishError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PublishError::Rtsp(e) => write!(f, "{e}"),
            PublishError::Io(e) => write!(f, "{e}"),
            PublishError::InvalidLocation => write!(f, "location is not an rtsp:// URL"),
            PublishError::CredentialsInLocation => {
                write!(f, "location carries credentials, set user-id and user-pw instead")
            }
            PublishError::TooManyStreams => write!(
                f,
                "more streams than dynamic RTP payload types {FIRST_DYNAMIC_PAYLOAD_TYPE}..={LAST_DYNAMIC_PAYLOAD_TYPE}"
            ),
            PublishError::NoUdpPortPair => write!(
                f,
                "no adjacent even and odd UDP ports after {UDP_PORT_PAIR_ATTEMPTS} attempts"
            ),
            PublishError::NoTransports => write!(f, "protocols lists no transport"),
            PublishError::TimedOut => write!(f, "server did not answer within tcp-timeout"),
            PublishError::Closed => write!(f, "server closed the RTSP connection"),
            PublishError::AccessUnitTooLarge => {
                write!(f, "AAC access unit exceeds the 13-bit RTP AU size")
            }
            PublishError::TimestampOutsideNtpRange => {
                write!(f, "frame PTS maps outside the NTP timestamp range")
            }
        }
    }
}

async fn within<T, E>(
    limit: Option<Duration>,
    future: impl Future<Output = Result<T, E>>,
) -> Result<T, PublishError>
where
    PublishError: From<E>,
{
    match limit {
        Some(limit) => Ok(tokio::time::timeout(limit, future)
            .await
            .map_err(|_| PublishError::TimedOut)??),
        None => Ok(future.await?),
    }
}

fn require_ok(method: &str, response: &RtspResponse) -> Result<(), RtspClientError> {
    if response.status != STATUS_OK {
        return Err(RtspClientError::Refused {
            method: method.to_string(),
            status_line: response.status_line.clone(),
        });
    }
    Ok(())
}

#[derive(Debug)]
struct ControlConnection {
    stream: TcpStream,
    requests: RtspPublishRequests,
    inbound: Vec<u8>,
    timeout: Option<Duration>,
}

impl ControlConnection {
    // one request and its response, resent once with credentials when the server asks for them
    async fn exchange(
        &mut self,
        method: &str,
        uri: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<RtspResponse, PublishError> {
        let response = self.round_trip(method, uri, headers, body).await?;
        if response.status != STATUS_UNAUTHORIZED {
            return Ok(response);
        }
        self.requests.answer_challenge(&response)?;
        self.round_trip(method, uri, headers, body).await
    }

    async fn round_trip(
        &mut self,
        method: &str,
        uri: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<RtspResponse, PublishError> {
        let request = self.requests.request(method, uri, headers, body);
        within(self.timeout, self.stream.write_all(&request)).await?;
        let mut response = self.read_response().await?;
        while self.requests.answers_earlier_request(&response) {
            response = self.read_response().await?;
        }
        self.requests.check_cseq(&response)?;
        Ok(response)
    }

    async fn read_response(&mut self) -> Result<RtspResponse, PublishError> {
        let mut buf = [0u8; READ_BUF_LEN];
        loop {
            if let Some(response) = self.take_response()? {
                return Ok(response);
            }
            let n = within(self.timeout, self.stream.read(&mut buf)).await?;
            if n == 0 {
                return Err(PublishError::Closed);
            }
            self.inbound.extend_from_slice(&buf[..n]);
            if self.inbound.len() > MAX_PENDING {
                return Err(RtspClientError::MalformedResponse.into());
            }
        }
    }

    // skips the RTCP the server interleaves on the connection, then takes one complete response
    fn take_response(&mut self) -> Result<Option<RtspResponse>, PublishError> {
        while self.inbound.first() == Some(&INTERLEAVED_MARKER) {
            let Some(frame) = interleaved_frame(&self.inbound) else {
                return Ok(None);
            };
            let consumed = frame.consumed;
            self.inbound.drain(..consumed);
        }
        let Some((response, consumed)) = RtspResponse::parse(&self.inbound)? else {
            return Ok(None);
        };
        self.inbound.drain(..consumed);
        Ok(Some(response))
    }

    // reads what arrived since the last frame without waiting, so the server's receiver reports never fill the socket
    fn discard_inbound(&mut self) -> Result<(), PublishError> {
        let mut buf = [0u8; READ_BUF_LEN];
        loop {
            match self.stream.try_read(&mut buf) {
                Ok(0) => return Err(PublishError::Closed),
                Ok(n) => self.inbound.extend_from_slice(&buf[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) => return Err(e.into()),
            }
        }
        while self.take_response()?.is_some() {}
        if self.inbound.len() > MAX_PENDING {
            return Err(RtspClientError::MalformedResponse.into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum PacketKind {
    Rtp,
    Rtcp,
}

#[derive(Debug)]
enum MediaDestination {
    Interleaved {
        rtp_channel: u8,
        rtcp_channel: u8,
    },
    Udp {
        rtp_socket: UdpSocket,
        server_rtp: SocketAddr,
        rtcp_socket: UdpSocket,
        server_rtcp: SocketAddr,
    },
}

impl MediaDestination {
    async fn deliver(
        &self,
        control: &mut ControlConnection,
        kind: PacketKind,
        packets: &[Vec<u8>],
    ) -> Result<(), PublishError> {
        match self {
            MediaDestination::Interleaved {
                rtp_channel,
                rtcp_channel,
            } => {
                let channel = match kind {
                    PacketKind::Rtp => *rtp_channel,
                    PacketKind::Rtcp => *rtcp_channel,
                };
                let framed_len = packets
                    .iter()
                    .map(|packet| INTERLEAVED_HEADER_LEN + packet.len())
                    .sum();
                let mut framed = Vec::with_capacity(framed_len);
                for packet in packets {
                    push_interleaved(&mut framed, channel, packet);
                }
                within(control.timeout, control.stream.write_all(&framed)).await
            }
            MediaDestination::Udp {
                rtp_socket,
                server_rtp,
                rtcp_socket,
                server_rtcp,
            } => {
                let (socket, address) = match kind {
                    PacketKind::Rtp => (rtp_socket, server_rtp),
                    PacketKind::Rtcp => (rtcp_socket, server_rtcp),
                };
                for packet in packets {
                    socket.send_to(packet, *address).await?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug)]
struct SessionStream {
    packetizer: StreamPacketizer,
    ssrc: u32,
    clock_rate: u32,
    destination: MediaDestination,
    counters: SenderCounters,
    next_report_ns: u64,
    last_sent: Option<SentFrame>,
}

// the last frame a stream sent, whose timestamps a sender report pairs
#[derive(Debug, Clone, Copy)]
struct SentFrame {
    rtp_timestamp: u32,
    pts_ns: u64,
}

#[derive(Debug)]
struct Keepalive {
    method: &'static str,
    interval_ns: u64,
    last_request_ns: u64,
}

#[derive(Debug)]
struct PublishSession {
    control: ControlConnection,
    request_url: String,
    keepalive: Option<Keepalive>,
    streams: Vec<SessionStream>,
    cname: String,
    // the wall-clock time of PTS zero, so every stream's sender reports share one NTP mapping
    pts_zero_unix_ns: i64,
    report_interval_ns: u64,
    // what stopped the timer task, reported on the next frame
    timer_failure: Option<PublishError>,
}

impl PublishSession {
    async fn send(&mut self, input: usize, frame: &[u8], pts_ns: u64) -> Result<(), PublishError> {
        if let Some(failure) = self.timer_failure.take() {
            return Err(failure);
        }
        self.control.discard_inbound()?;
        let stream = &mut self.streams[input];
        let rtp_timestamp = rtp_timestamp_at_clock(pts_ns, u64::from(stream.clock_rate));
        let packets = stream.packetizer.packetize(frame, rtp_timestamp)?;
        for packet in &packets {
            stream.counters.count(packet);
        }
        stream
            .destination
            .deliver(&mut self.control, PacketKind::Rtp, &packets)
            .await?;
        stream.last_sent = Some(SentFrame {
            rtp_timestamp,
            pts_ns,
        });
        self.report_if_due(input).await
    }

    async fn report_if_due(&mut self, index: usize) -> Result<(), PublishError> {
        let stream = &mut self.streams[index];
        let Some(sent) = stream.last_sent else {
            return Ok(());
        };
        let now = monotonic_ns();
        if now < stream.next_report_ns {
            return Ok(());
        }
        stream.next_report_ns = now.saturating_add(randomized_interval_ns(self.report_interval_ns));
        let ntp = ntp_from_unix_nanos(self.pts_zero_unix_ns.saturating_add_unsigned(sent.pts_ns))
            .ok_or(PublishError::TimestampOutsideNtpRange)?;
        let mut report = stream
            .counters
            .sender_report(stream.ssrc, ntp, sent.rtp_timestamp);
        report.extend(build_sdes_cname(stream.ssrc, &self.cname));
        stream
            .destination
            .deliver(&mut self.control, PacketKind::Rtcp, &[report])
            .await
    }

    async fn on_timer(&mut self) -> Result<(), PublishError> {
        self.control.discard_inbound()?;
        self.keep_alive().await?;
        for index in 0..self.streams.len() {
            self.report_if_due(index).await?;
        }
        Ok(())
    }

    // capped at half the report interval, the soonest a frame sent meanwhile can make a report due
    fn next_timer_ns(&self) -> u64 {
        let keepalive = self
            .keepalive
            .as_ref()
            .map(|k| k.last_request_ns.saturating_add(k.interval_ns));
        let reports = self
            .streams
            .iter()
            .filter(|stream| stream.last_sent.is_some())
            .map(|stream| stream.next_report_ns);
        let cap = monotonic_ns().saturating_add(self.report_interval_ns / 2);
        keepalive.into_iter().chain(reports).fold(cap, u64::min)
    }

    // the answer is read and dropped with the server's RTCP, as gst does not wait for it either
    async fn keep_alive(&mut self) -> Result<(), PublishError> {
        let Some(keepalive) = &mut self.keepalive else {
            return Ok(());
        };
        let now = monotonic_ns();
        if now.saturating_sub(keepalive.last_request_ns) < keepalive.interval_ns {
            return Ok(());
        }
        keepalive.last_request_ns = now;
        let request = self
            .control
            .requests
            .request(keepalive.method, &self.request_url, &[], b"");
        within(
            self.control.timeout,
            self.control.stream.write_all(&request),
        )
        .await
    }

    async fn teardown(&mut self) -> Result<(), PublishError> {
        let url = self.request_url.clone();
        let response = self.control.exchange("TEARDOWN", &url, &[], b"").await?;
        Ok(require_ok("TEARDOWN", &response)?)
    }
}

// keepalives and sender reports run on a timer so a stalled upstream keeps the session
async fn run_session_timers(session: Arc<Mutex<PublishSession>>) {
    loop {
        let wake_ns = session.lock().await.next_timer_ns();
        let delay_ns = wake_ns.saturating_sub(monotonic_ns());
        tokio::time::sleep(Duration::from_nanos(delay_ns)).await;
        let mut session = session.lock().await;
        if let Err(failure) = session.on_timer().await {
            session.timer_failure = Some(failure);
            return;
        }
    }
}

#[derive(Debug)]
struct LiveSession {
    session: Arc<Mutex<PublishSession>>,
    timers: JoinHandle<()>,
}

impl LiveSession {
    fn start(session: PublishSession) -> Self {
        let session = Arc::new(Mutex::new(session));
        let timers = tokio::spawn(run_session_timers(session.clone()));
        Self { session, timers }
    }
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        self.timers.abort();
    }
}

struct PublishTarget {
    host: String,
    port: u16,
    request_url: String,
}

fn publish_target(location: &str) -> Result<PublishTarget, PublishError> {
    let url = url::Url::parse(location).map_err(|_| PublishError::InvalidLocation)?;
    if url.scheme() != "rtsp" {
        return Err(PublishError::InvalidLocation);
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(PublishError::CredentialsInLocation);
    }
    let host = match url.host().ok_or(PublishError::InvalidLocation)? {
        url::Host::Domain(domain) => domain.to_string(),
        url::Host::Ipv4(address) => address.to_string(),
        url::Host::Ipv6(address) => address.to_string(),
    };
    Ok(PublishTarget {
        host,
        port: url.port().unwrap_or(DEFAULT_RTSP_PORT),
        request_url: url.as_str().to_string(),
    })
}

fn payload_type(index: usize) -> Result<u8, PublishError> {
    u8::try_from(index)
        .ok()
        .and_then(|i| FIRST_DYNAMIC_PAYLOAD_TYPE.checked_add(i))
        .filter(|pt| *pt <= LAST_DYNAMIC_PAYLOAD_TYPE)
        .ok_or(PublishError::TooManyStreams)
}

fn random_u32() -> u32 {
    let mut bytes = [0u8; 4];
    getrandom::getrandom(&mut bytes).expect("OS RNG for the RTP SSRC and RTCP timing");
    u32::from_be_bytes(bytes)
}

// RFC 3550 §6.3.1 spreads each report over 0.5 to 1.5 times the nominal interval
fn randomized_interval_ns(nominal_ns: u64) -> u64 {
    let spread = (u128::from(nominal_ns) * u128::from(random_u32())) >> u32::BITS;
    nominal_ns / 2 + spread as u64
}

async fn bind_udp_port_pair(local_ip: IpAddr) -> Result<(UdpSocket, UdpSocket), PublishError> {
    for _ in 0..UDP_PORT_PAIR_ATTEMPTS {
        let rtp_socket = UdpSocket::bind((local_ip, 0)).await?;
        let port = rtp_socket.local_addr()?.port();
        // RTP takes the even port of the pair (RFC 3550 §11)
        if port % 2 != 0 {
            continue;
        }
        if let Ok(rtcp_socket) = UdpSocket::bind((local_ip, port + 1)).await {
            return Ok((rtp_socket, rtcp_socket));
        }
    }
    Err(PublishError::NoUdpPortPair)
}

async fn setup_stream(
    control: &mut ControlConnection,
    control_url: &str,
    index: usize,
    transport: LowerTransport,
) -> Result<MediaDestination, PublishError> {
    match transport {
        LowerTransport::Tcp => {
            // payload_type has bounded the index, so the channel pair fits a u8
            let requested_channel = (index * 2) as u8;
            let transport_header = record_transport_tcp(requested_channel);
            let response = control
                .exchange(
                    "SETUP",
                    control_url,
                    &[("Transport", &transport_header)],
                    b"",
                )
                .await?;
            require_ok("SETUP", &response)?;
            control.requests.accept_session(&response)?;
            let (rtp_channel, rtcp_channel) = response
                .transport
                .as_deref()
                .and_then(parse_interleaved_channels)
                .ok_or(RtspClientError::MissingTransport)?;
            Ok(MediaDestination::Interleaved {
                rtp_channel,
                rtcp_channel,
            })
        }
        LowerTransport::Udp => {
            let local_ip = control.stream.local_addr()?.ip();
            let (rtp_socket, rtcp_socket) = bind_udp_port_pair(local_ip).await?;
            let transport_header = record_transport_udp(rtp_socket.local_addr()?.port());
            let response = control
                .exchange(
                    "SETUP",
                    control_url,
                    &[("Transport", &transport_header)],
                    b"",
                )
                .await?;
            require_ok("SETUP", &response)?;
            control.requests.accept_session(&response)?;
            let server_port = response
                .transport
                .as_deref()
                .and_then(|t| parse_rtp_port(t, "server_port="))
                .ok_or(RtspClientError::MissingTransport)?;
            let server_rtcp_port = server_port
                .checked_add(1)
                .ok_or(RtspClientError::MissingTransport)?;
            let server_ip = control.stream.peer_addr()?.ip();
            Ok(MediaDestination::Udp {
                rtp_socket,
                server_rtp: SocketAddr::new(server_ip, server_port),
                rtcp_socket,
                server_rtcp: SocketAddr::new(server_ip, server_rtcp_port),
            })
        }
    }
}

#[derive(Debug, Default)]
struct Pad {
    codec: Option<StreamCodec>,
    config: Option<StreamConfig>,
    ended: bool,
    // a new session's video starts at a keyframe
    awaiting_keyframe: bool,
}

#[derive(Debug)]
struct Outage {
    retry: ReconnectState,
    next_attempt_ns: u64,
}

#[derive(Debug)]
enum PublishState {
    // frames wait here until every pad has described its stream
    Holding(Vec<HeldFrame>),
    Live(LiveSession),
    Down(Outage),
}

#[derive(Debug)]
struct HeldFrame {
    input: usize,
    pts_ns: u64,
    data: Vec<u8>,
}

#[derive(Debug)]
pub struct RtspClientSink {
    location: String,
    transports: Vec<LowerTransport>,
    user_id: Option<String>,
    user_pw: String,
    user_agent: String,
    tcp_timeout_us: u64,
    keep_alive: bool,
    report_interval_ms: u64,
    reconnect: ReconnectPolicy,
    pads: Vec<Pad>,
    state: PublishState,
    log_name: LogName,
}

impl LogSource for RtspClientSink {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

static RTSP_CLIENT_SINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "rtsp:// URL to publish to (ANNOUNCE / RECORD)",
    ),
    PropertySpec::new(
        "protocols",
        PropKind::Flags,
        "lower transports to request at SETUP, tried in the order written",
    )
    .with_enum_values("udp | tcp")
    .with_default("tcp"),
    PropertySpec::new("user-id", PropKind::Str, "RTSP auth username"),
    PropertySpec::new("user-pw", PropKind::Str, "RTSP auth password"),
    PropertySpec::new(
        "user-agent",
        PropKind::Str,
        "User-Agent header sent on every request",
    )
    .with_default(DEFAULT_USER_AGENT),
    PropertySpec::new(
        "tcp-timeout",
        PropKind::Uint,
        "fail when the server does not answer or accept data within this many microseconds (0 = never)",
    )
    .with_default("20000000"),
    PropertySpec::new(
        "do-rtsp-keep-alive",
        PropKind::Bool,
        "send GET_PARAMETER, or OPTIONS when the server lacks it, ahead of the session timeout",
    )
    .with_default("true"),
    PropertySpec::new(
        "rtcp-sr-interval",
        PropKind::Uint,
        "nominal RTCP sender-report interval in milliseconds, randomized per RFC 3550",
    )
    .with_default("5000")
    .with_range("1", "60000"),
    PropertySpec::new(
        "reconnect",
        PropKind::Uint,
        "max reconnect attempts after each loss of the session, frames are dropped meanwhile (0 = no reconnect)",
    )
    .with_default("0"),
    PropertySpec::new(
        "reconnect-backoff",
        PropKind::Uint,
        "wait before the first reconnect attempt, milliseconds (doubles per retry)",
    )
    .with_default("250"),
    PropertySpec::new(
        "reconnect-backoff-max",
        PropKind::Uint,
        "cap on the doubling reconnect backoff, milliseconds",
    )
    .with_default("5000"),
];

impl RtspClientSink {
    pub fn new(inputs: usize, location: impl Into<String>) -> Self {
        Self {
            location: location.into(),
            transports: Vec::from([LowerTransport::Tcp]),
            user_id: None,
            user_pw: String::new(),
            user_agent: DEFAULT_USER_AGENT.to_string(),
            tcp_timeout_us: DEFAULT_TCP_TIMEOUT_US,
            keep_alive: true,
            report_interval_ms: DEFAULT_REPORT_INTERVAL_MS,
            reconnect: ReconnectPolicy {
                max_attempts: 0,
                initial_backoff_ms: DEFAULT_INITIAL_BACKOFF_MS,
                max_backoff_ms: DEFAULT_MAX_BACKOFF_MS,
            },
            pads: (0..inputs).map(|_| Pad::default()).collect(),
            state: PublishState::Holding(Vec::new()),
            log_name: LogName::new(),
        }
    }

    pub fn with_protocols(mut self, transports: Vec<LowerTransport>) -> Self {
        self.transports = transports;
        self
    }

    pub fn with_credentials(
        mut self,
        user_id: impl Into<String>,
        user_pw: impl Into<String>,
    ) -> Self {
        self.user_id = Some(user_id.into());
        self.user_pw = user_pw.into();
        self
    }

    fn tcp_timeout(&self) -> Option<Duration> {
        (self.tcp_timeout_us > 0).then(|| Duration::from_micros(self.tcp_timeout_us))
    }

    fn publish_failed(&self, error: PublishError) -> G2gError {
        g2g_warn!(self, "publishing to {} failed: {error}", self.location);
        match error {
            PublishError::Io(e) => io_err(e),
            PublishError::TimedOut => G2gError::Timeout,
            _ => G2gError::Hardware(HardwareError::Other),
        }
    }

    // a refused SETUP moves on to the next transport `protocols` lists
    async fn setup_with_fallback(
        &self,
        control: &mut ControlConnection,
        control_url: &str,
        index: usize,
        candidates: &[LowerTransport],
    ) -> Result<(MediaDestination, LowerTransport), PublishError> {
        for (attempt, transport) in candidates.iter().enumerate() {
            match setup_stream(control, control_url, index, *transport).await {
                Ok(destination) => return Ok((destination, *transport)),
                Err(PublishError::Rtsp(RtspClientError::Refused { status_line, .. }))
                    if attempt + 1 < candidates.len() =>
                {
                    g2g_warn!(
                        self,
                        "SETUP over {} refused: {status_line}",
                        transport.nick()
                    );
                }
                Err(e) => return Err(e),
            }
        }
        Err(PublishError::NoTransports)
    }

    async fn start_session(
        &self,
        configs: &[StreamConfig],
        first_pts_ns: u64,
    ) -> Result<PublishSession, PublishError> {
        let target = publish_target(&self.location)?;
        let timeout = self.tcp_timeout();
        let stream = within(
            timeout,
            TcpStream::connect((target.host.as_str(), target.port)),
        )
        .await?;
        let credentials = self.user_id.clone().map(|user| Credentials {
            user,
            password: self.user_pw.clone(),
        });
        let mut control = ControlConnection {
            stream,
            requests: RtspPublishRequests::new(self.user_agent.clone(), credentials),
            inbound: Vec::new(),
            timeout,
        };
        let url = target.request_url;

        let options = control.exchange("OPTIONS", &url, &[], b"").await?;
        require_ok("OPTIONS", &options)?;

        let payload_types = (0..configs.len())
            .map(payload_type)
            .collect::<Result<Vec<u8>, _>>()?;
        let sections: Vec<String> = configs
            .iter()
            .zip(&payload_types)
            .enumerate()
            .map(|(i, (config, pt))| config.media_section(*pt, i))
            .collect();
        let sdp = session_description(&sections);
        let announce = control
            .exchange(
                "ANNOUNCE",
                &url,
                &[("Content-Type", "application/sdp")],
                sdp.as_bytes(),
            )
            .await?;
        require_ok("ANNOUNCE", &announce)?;

        let mut chosen: Option<LowerTransport> = None;
        let mut streams = Vec::with_capacity(configs.len());
        for (i, (config, pt)) in configs.iter().zip(&payload_types).enumerate() {
            let control_url = format!("{url}/{}", stream_control(i));
            // the first stream settles the transport the rest of the session uses
            let candidates = match chosen {
                Some(transport) => Vec::from([transport]),
                None => self.transports.clone(),
            };
            let (destination, transport) = self
                .setup_with_fallback(&mut control, &control_url, i, &candidates)
                .await?;
            chosen = Some(transport);
            let ssrc = random_u32();
            streams.push(SessionStream {
                packetizer: config.packetizer(*pt, ssrc),
                ssrc,
                clock_rate: config.clock_rate(),
                destination,
                counters: SenderCounters::default(),
                next_report_ns: 0,
                last_sent: None,
            });
        }

        let record = control
            .exchange("RECORD", &url, &[("Range", "npt=0.000-")], b"")
            .await?;
        require_ok("RECORD", &record)?;
        let keepalive_method = if options.allows("GET_PARAMETER") {
            "GET_PARAMETER"
        } else {
            "OPTIONS"
        };
        let keepalive = self.keep_alive.then(|| Keepalive {
            method: keepalive_method,
            interval_ns: keepalive_interval_s(control.requests.session_timeout_s())
                .max(MIN_KEEPALIVE_INTERVAL_S)
                .saturating_mul(NS_PER_SECOND),
            last_request_ns: monotonic_ns(),
        });
        Ok(PublishSession {
            control,
            request_url: url,
            keepalive,
            streams,
            cname: format!("g2g-{:08x}", random_u32()),
            pts_zero_unix_ns: unix_time_ns().saturating_sub_unsigned(first_pts_ns),
            report_interval_ns: self.report_interval_ms * NS_PER_MS,
            timer_failure: None,
        })
    }

    async fn on_frame(&mut self, input: usize, frame: &[u8], pts_ns: u64) -> Result<(), G2gError> {
        let pad = self.pads.get(input).ok_or(G2gError::NotConfigured)?;
        let codec = pad.codec.ok_or(G2gError::NotConfigured)?;
        if pad.config.is_none() {
            let config = StreamConfig::from_first_frame(codec, frame).inspect_err(|_| {
                g2g_warn!(
                    self,
                    "pad {input}: first frame does not describe its stream"
                );
            })?;
            self.pads[input].config = Some(config);
        }
        if self.pads[input]
            .config
            .as_ref()
            .is_some_and(|c| c.is_config_frame(frame))
        {
            return Ok(());
        }
        let PublishState::Holding(held) = &mut self.state else {
            return self.publish(input, frame, pts_ns).await;
        };
        held.push(HeldFrame {
            input,
            pts_ns,
            data: frame.to_vec(),
        });
        if let Some(waiting) = self.pads.iter().position(|pad| pad.config.is_none()) {
            if held.len() > MAX_HELD_FRAMES {
                g2g_warn!(
                    self,
                    "pad {waiting} sent no frame within {MAX_HELD_FRAMES} frames of the others"
                );
                return Err(G2gError::NotConfigured);
            }
            return Ok(());
        }
        let mut held = core::mem::take(held);
        held.sort_by_key(|frame| frame.pts_ns);
        match self
            .start_session(&self.stream_configs(), held[0].pts_ns)
            .await
        {
            Ok(session) => self.state = PublishState::Live(LiveSession::start(session)),
            Err(e) => return self.lose_session(e),
        }
        for frame in held {
            self.publish(frame.input, &frame.data, frame.pts_ns).await?;
        }
        Ok(())
    }

    fn stream_configs(&self) -> Vec<StreamConfig> {
        self.pads
            .iter()
            .filter_map(|pad| pad.config.clone())
            .collect()
    }

    async fn publish(&mut self, input: usize, frame: &[u8], pts_ns: u64) -> Result<(), G2gError> {
        let pad = &self.pads[input];
        let video_codec = pad.config.as_ref().and_then(StreamConfig::video_codec);
        let opens_stream = video_codec.is_none_or(|codec| au_is_keyframe(codec, frame));
        if pad.awaiting_keyframe && !opens_stream {
            return Ok(());
        }
        if let PublishState::Down(outage) = &self.state {
            if monotonic_ns() < outage.next_attempt_ns {
                return Ok(());
            }
            let retry = outage.retry.clone();
            match self.start_session(&self.stream_configs(), pts_ns).await {
                Ok(session) => {
                    g2g_info!(
                        self,
                        "republishing to {} after {} attempts",
                        self.location,
                        retry.attempt
                    );
                    self.state = PublishState::Live(LiveSession::start(session));
                }
                Err(e) => return self.schedule_reconnect(retry, e),
            }
        }
        let PublishState::Live(live) = &mut self.state else {
            return Ok(());
        };
        let sent = live.session.lock().await.send(input, frame, pts_ns).await;
        match sent {
            Ok(()) => {
                self.pads[input].awaiting_keyframe = false;
                Ok(())
            }
            Err(e) => self.lose_session(e),
        }
    }

    // errors only when the policy has no attempt left for this loss
    fn lose_session(&mut self, error: PublishError) -> Result<(), G2gError> {
        let retry = ReconnectState::new(&self.reconnect);
        self.schedule_reconnect(retry, error)
    }

    fn schedule_reconnect(
        &mut self,
        mut retry: ReconnectState,
        error: PublishError,
    ) -> Result<(), G2gError> {
        let delay_ms = retry
            .next_delay_ms(&self.reconnect)
            .filter(|_| error.is_transient());
        let Some(delay_ms) = delay_ms else {
            self.state = PublishState::Holding(Vec::new());
            return Err(self.publish_failed(error));
        };
        g2g_warn!(
            self,
            "publishing to {} failed: {error}, reconnect {}/{} in {delay_ms} ms",
            self.location,
            retry.attempt,
            self.reconnect.max_attempts
        );
        self.state = PublishState::Down(Outage {
            retry,
            next_attempt_ns: monotonic_ns().saturating_add(delay_ms.saturating_mul(NS_PER_MS)),
        });
        for pad in &mut self.pads {
            pad.awaiting_keyframe = true;
        }
        Ok(())
    }

    async fn on_eos(&mut self, input: usize) -> Result<(), G2gError> {
        let pad = self.pads.get_mut(input).ok_or(G2gError::NotConfigured)?;
        pad.ended = true;
        if matches!(self.state, PublishState::Holding(_)) && pad.config.is_none() {
            g2g_warn!(
                self,
                "pad {input} ended before its first frame, nothing was published"
            );
            return Err(G2gError::NotConfigured);
        }
        if !self.pads.iter().all(|pad| pad.ended) {
            return Ok(());
        }
        let PublishState::Live(live) = &mut self.state else {
            return Ok(());
        };
        live.timers.abort();
        let result = live.session.lock().await.teardown().await;
        self.state = PublishState::Holding(Vec::new());
        result.map_err(|e| self.publish_failed(e))
    }
}

fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

impl MultiInputElement for RtspClientSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn input_count(&self) -> usize {
        self.pads.len()
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        StreamCodec::from_caps(upstream_caps).ok_or(G2gError::CapsMismatch)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let codec = StreamCodec::from_caps(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.pads
            .get_mut(input)
            .ok_or(G2gError::NotConfigured)?
            .codec = Some(codec);
        Ok(ConfigureOutcome::Accepted)
    }

    // terminal: the server is the destination, so this placeholder is never consulted
    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(h264_any())
    }

    fn is_terminal(&self) -> bool {
        true
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "RTSP RECORD client",
            "Sink/Network",
            "Publishes H.264 / H.265 video and AAC / Opus audio to an RTSP server with ANNOUNCE and RECORD",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let data = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    let Some(pts_ns) = frame.timing.pts() else {
                        g2g_warn!(self, "pad {input}: frame has no PTS for its RTP timestamp");
                        return Err(G2gError::CapsMismatch);
                    };
                    self.on_frame(input, data, pts_ns).await
                }
                PipelinePacket::Eos => self.on_eos(input).await,
                _ => Ok(()),
            }
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        RTSP_CLIENT_SINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let text = || value.as_str().map(str::to_string).ok_or(PropError::Type);
        match name {
            "location" => self.location = text()?,
            "protocols" => self.transports = transports_from_flags(&value)?,
            "user-id" => self.user_id = Some(text()?).filter(|id| !id.is_empty()),
            "user-pw" => self.user_pw = text()?,
            "user-agent" => self.user_agent = text()?,
            "tcp-timeout" => self.tcp_timeout_us = value.as_uint().ok_or(PropError::Type)?,
            "do-rtsp-keep-alive" => self.keep_alive = value.as_bool().ok_or(PropError::Type)?,
            "rtcp-sr-interval" => {
                let ms = value.as_uint().ok_or(PropError::Type)?;
                if !(1..=MAX_REPORT_INTERVAL_MS).contains(&ms) {
                    return Err(PropError::Value);
                }
                self.report_interval_ms = ms;
            }
            "reconnect" => {
                let attempts = value.as_uint().ok_or(PropError::Type)?;
                self.reconnect.max_attempts =
                    u32::try_from(attempts).map_err(|_| PropError::Value)?;
            }
            "reconnect-backoff" => {
                self.reconnect.initial_backoff_ms = value.as_uint().ok_or(PropError::Type)?
            }
            "reconnect-backoff-max" => {
                self.reconnect.max_backoff_ms = value.as_uint().ok_or(PropError::Type)?
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let text = |s: &str| Some(PropValue::Str(s.to_string()));
        match name {
            "location" => text(&self.location),
            "protocols" => Some(transports_to_flags(&self.transports)),
            "user-id" => text(self.user_id.as_deref().unwrap_or("")),
            "user-pw" => text(&self.user_pw),
            "user-agent" => text(&self.user_agent),
            "tcp-timeout" => Some(PropValue::Uint(self.tcp_timeout_us)),
            "do-rtsp-keep-alive" => Some(PropValue::Bool(self.keep_alive)),
            "rtcp-sr-interval" => Some(PropValue::Uint(self.report_interval_ms)),
            "reconnect" => Some(PropValue::Uint(u64::from(self.reconnect.max_attempts))),
            "reconnect-backoff" => Some(PropValue::Uint(self.reconnect.initial_backoff_ms)),
            "reconnect-backoff-max" => Some(PropValue::Uint(self.reconnect.max_backoff_ms)),
            _ => None,
        }
    }
}

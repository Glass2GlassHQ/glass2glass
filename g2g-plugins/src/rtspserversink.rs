// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! RTSP server sink (RtspServerSink, `rtsp-server` feature): hosts a pipeline's
//! H.264, and ONVIF metadata on further pads, as an RTSP endpoint and serves it
//! over RTP to connecting players (the
//! OBS / surveillance / contribution-server shape), on either transport a player
//! SETUPs: unicast RTP/UDP, or TCP-interleaved (RFC 2326 §10.12, `$`-framed RTP
//! on the control connection, what `ffmpeg -rtsp_transport tcp` uses, for players
//! behind a firewall that blocks the UDP ports). The sans-IO
//! [`rtspserver::RtspResponder`](crate::rtspserver) does the protocol work
//! (OPTIONS / DESCRIBE / SETUP / PLAY); this element is the tokio TCP control
//! channel + the RTP transport around it, reusing the
//! [`RtpH264Packetizer`](crate::rtppay) the UDP sink uses.
//!
//! Multi-client: the listener is bound in `configure_pipeline`; every buffer
//! accepts new players and advances their handshakes without blocking, then
//! broadcasts the frame to every PLAYing player on its own RTP session. A
//! buffer with no PLAYing player is dropped, so a player starts at the current
//! frame instead of a backlog held while nobody watched. Players that disconnect are reaped. Each track's
//! RTP UDP socket sends to each player's negotiated address. ANNOUNCE/RECORD ingest is the
//! separate [`RtspServerSrc`](crate::rtspserversrc).
//!
//! RTCP + keepalive during PLAY: sender reports (NTP <-> RTP mapping, RFC 3550)
//! go to each player periodically, over UDP from the socket adjacent to the RTP
//! one (the advertised `server_port` pair) or `$`-framed on the RTCP channel for
//! an interleaved player, with a BYE at EOS. The SETUP response advertises
//! `Session: id;timeout=N`; a player silent past the timeout on both the control
//! channel (GET_PARAMETER / OPTIONS keepalive) and RTCP (receiver reports) is
//! reaped, so departed players do not accumulate.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::net::{IpAddr, SocketAddr, TcpListener as StdTcpListener};

use tokio::io::AsyncWriteExt;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, Frame,
    G2gError, HardwareError, MultiInputElement, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, Reconfigure, VideoCodec,
};

use crate::clock::PtsWallClock;
use crate::filesink::io_err;
use crate::rtcp;
use crate::rtpklv::{pts_ns_at_rtp_tick, RtpKlvPacketizer};
use crate::rtppay::RtpH264Packetizer;
use crate::rtspserver::{
    interleaved_frame, push_interleaved, RtspEvent, RtspRequest, RtspResponder, TrackTransport,
    DEFAULT_SESSION_TIMEOUT_SECS, INTERLEAVED_HEADER_LEN, INTERLEAVED_MARKER,
};
use crate::sdp::{h264_media_section, onvif_metadata_media_section, session_description};

/// Default dynamic RTP payload type for H.264.
const DEFAULT_PAYLOAD_TYPE: u8 = 96;
/// Default max RTP payload bytes, leaving headroom under a 1500-byte MTU.
const DEFAULT_MAX_PAYLOAD: usize = 1400;
/// TCP read buffer for RTSP control requests.
const CTRL_BUF: usize = 8192;
const RTP_SEQUENCE_RANGE: core::ops::Range<usize> = 2..4;
/// Cap on buffered-but-unparsed control bytes. A real RTSP request (even an
/// ANNOUNCE carrying SDP) is far smaller; the bound reaps a client that drips a
/// never-terminating request or an oversized Content-Length (slow-loris DoS).
const MAX_PENDING: usize = 64 * 1024;
/// Default RTCP sender-report interval (RFC 3550's nominal 5 s).
const DEFAULT_SR_INTERVAL_NS: u64 = 5_000_000_000;

/// H.264-at-any-geometry caps (geometry rides in-band in the SPS).
fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    }
}

fn served_caps() -> CapsSet {
    CapsSet::from_alternatives(Vec::from([h264_any(), Caps::OnvifMetadata]))
}

fn intercept_served_caps(upstream_caps: &Caps) -> Result<Caps, G2gError> {
    upstream_caps
        .intersect(&h264_any())
        .or_else(|_| upstream_caps.intersect(&Caps::OnvifMetadata))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrackMedia {
    H264,
    OnvifMetadata,
}

impl TrackMedia {
    fn of(caps: &Caps) -> Option<Self> {
        match caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            } => Some(Self::H264),
            Caps::OnvifMetadata => Some(Self::OnvifMetadata),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
struct ServedTrack {
    media: Option<TrackMedia>,
    rtp_socket: Option<tokio::net::UdpSocket>,
    rtcp_socket: Option<tokio::net::UdpSocket>,
    ended: bool,
}

#[derive(Debug, Default)]
struct PlayerTrack {
    dest: Option<SocketAddr>,
    /// Where this player's RTCP lives when on UDP: the second port of the
    /// `client_port` pair. Sender reports go there; its receiver reports come
    /// from there.
    rtcp_dest: Option<SocketAddr>,
    /// The RTP channel once a TCP-interleaved SETUP was negotiated (RFC 2326
    /// §10.12): RTP rides the control connection as `$`-framed binary instead of
    /// its own UDP port. Mutually exclusive with `dest` in practice.
    interleaved: Option<u8>,
    /// The RTCP channel for an interleaved player: SRs out, RRs in, `$`-framed.
    rtcp_channel: Option<u8>,
    /// This track's own RTP sequence numbering, armed at PLAY.
    next_sequence: Option<u16>,
    /// RFC 3550 sender counters for this track's RTP session.
    sender_counters: rtcp::SenderCounters,
    last_rtp_ts: Option<u32>,
    last_pts_ns: u64,
    last_sr_ns: u64,
}

impl PlayerTrack {
    /// PLAYing once SETUP gave a transport (a UDP destination or an interleaved
    /// channel) and PLAY armed the sequence numbering.
    fn playing(&self) -> bool {
        self.next_sequence.is_some() && (self.dest.is_some() || self.interleaved.is_some())
    }
}

/// One connected player: its control channel, protocol responder, and one
/// [`PlayerTrack`] per served track.
#[derive(Debug)]
struct Client {
    control: tokio::net::TcpStream,
    responder: RtspResponder,
    pending: Vec<u8>,
    peer_ip: IpAddr,
    tracks: Vec<PlayerTrack>,
    // a player that starts between keyframes has no reference for the frames before the next one
    awaiting_keyframe: bool,
    /// Last control-channel or RTCP activity, for session-timeout reaping.
    last_activity_ns: u64,
}

impl Client {
    fn new(control: tokio::net::TcpStream, responder: RtspResponder, peer_ip: IpAddr) -> Self {
        let tracks = responder
            .track_transports()
            .iter()
            .map(|_| PlayerTrack::default())
            .collect();
        Self {
            control,
            responder,
            pending: Vec::new(),
            peer_ip,
            tracks,
            awaiting_keyframe: false,
            last_activity_ns: g2g_core::metrics::monotonic_ns(),
        }
    }

    // PLAY starts every track this player SETUP, on the transport its SETUP negotiated
    fn start_tracks(&mut self) {
        for (track, transport) in self
            .tracks
            .iter_mut()
            .zip(self.responder.track_transports())
        {
            match *transport {
                Some(TrackTransport::Udp { client_rtp_port }) => {
                    track.dest = Some(SocketAddr::new(self.peer_ip, client_rtp_port));
                    track.rtcp_dest = Some(SocketAddr::new(
                        self.peer_ip,
                        client_rtp_port.saturating_add(1),
                    ));
                }
                Some(TrackTransport::Interleaved {
                    rtp_channel,
                    rtcp_channel,
                }) => {
                    track.interleaved = Some(rtp_channel);
                    track.rtcp_channel = Some(rtcp_channel);
                }
                None => continue,
            }
            track.next_sequence = Some(0);
        }
        self.awaiting_keyframe = true;
    }

    /// Drain whatever control bytes are readable now (non-blocking) and answer
    /// the requests they complete, advancing toward PLAY. Returns `false` if the
    /// player disconnected or tore down (the caller reaps it).
    async fn advance(&mut self, stream_rtptime: Option<u32>) -> bool {
        let mut buf = [0u8; CTRL_BUF];
        loop {
            match self.control.try_read(&mut buf) {
                Ok(0) => return false, // closed
                Ok(n) => self.pending.extend_from_slice(&buf[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => return false,
            }
        }
        loop {
            // An interleaved player sends its RTCP (receiver reports) `$`-framed
            // on the control connection; consume the frame before request parsing
            // so the binary is not misread as a request, and count it as
            // keepalive activity.
            if self.pending.first() == Some(&INTERLEAVED_MARKER) {
                let Some(frame) = interleaved_frame(&self.pending) else {
                    break; // frame not fully arrived
                };
                let consumed = frame.consumed;
                self.pending.drain(..consumed);
                self.last_activity_ns = g2g_core::metrics::monotonic_ns();
                continue;
            }
            let Some((req, consumed)) = RtspRequest::parse(&self.pending) else {
                break;
            };
            self.pending.drain(..consumed);
            self.last_activity_ns = g2g_core::metrics::monotonic_ns();
            if let Some(rtptime) = stream_rtptime {
                self.responder.set_stream_rtptime(rtptime);
            }
            let (response, event) = self.responder.handle_request(&req);
            if self.control.write_all(&response).await.is_err() {
                return false;
            }
            match event {
                RtspEvent::Play => self.start_tracks(),
                RtspEvent::Pause => {
                    for track in &mut self.tracks {
                        track.next_sequence = None;
                    }
                }
                RtspEvent::Teardown => return false,
                _ => {}
            }
        }
        // A partial request that never completes must not grow without bound.
        self.pending.len() <= MAX_PENDING
    }

    /// Stamp this player's sequence numbers on one track's `packets` and send
    /// them over that track's negotiated transport: UDP to `dest`, or `$`-framed
    /// on the control connection for a TCP-interleaved client. Returns the packet
    /// count, or `Err` if the send failed (the caller reaps it).
    async fn send_packets(
        &mut self,
        track_index: usize,
        socket: &tokio::net::UdpSocket,
        packets: &mut [Vec<u8>],
        timestamp: u32,
        pts_ns: u64,
    ) -> Result<u64, ()> {
        let track = &mut self.tracks[track_index];
        let Some(first_sequence) = track.next_sequence else {
            return Ok(0);
        };
        for (offset, packet) in packets.iter_mut().enumerate() {
            let sequence = first_sequence.wrapping_add(offset as u16);
            packet[RTP_SEQUENCE_RANGE].copy_from_slice(&sequence.to_be_bytes());
        }
        track.next_sequence = Some(first_sequence.wrapping_add(packets.len() as u16));
        track.last_rtp_ts = Some(timestamp);
        track.last_pts_ns = pts_ns;
        let mut sent = 0;
        if let Some(channel) = track.interleaved {
            let framed_len = packets
                .iter()
                .map(|packet| INTERLEAVED_HEADER_LEN + packet.len())
                .sum();
            let mut framed = Vec::with_capacity(framed_len);
            for packet in packets.iter() {
                push_interleaved(&mut framed, channel, packet);
            }
            write_to_player(&mut self.control, &framed).await?;
            sent = packets.len() as u64;
        } else if let Some(dest) = track.dest {
            for packet in packets.iter() {
                if socket.send_to(packet, dest).await.is_err() {
                    return Err(());
                }
                sent += 1;
            }
        }
        for packet in packets.iter() {
            track.sender_counters.count(packet);
        }
        Ok(sent)
    }

    /// Send one RTCP packet over one track's negotiated RTCP transport:
    /// `$`-framed on the RTCP channel for an interleaved player, else UDP from
    /// the track's server RTCP socket to the client's RTCP port. `Err` if the
    /// send failed (the caller reaps it); `Ok(false)` if no transport is
    /// negotiated.
    async fn send_rtcp(
        &mut self,
        track_index: usize,
        rtcp_socket: &tokio::net::UdpSocket,
        packet: &[u8],
    ) -> Result<bool, ()> {
        let track = &self.tracks[track_index];
        if let Some(channel) = track.rtcp_channel {
            let mut framed = Vec::with_capacity(INTERLEAVED_HEADER_LEN + packet.len());
            push_interleaved(&mut framed, channel, packet);
            write_to_player(&mut self.control, &framed).await?;
            Ok(true)
        } else if let Some(dest) = track.rtcp_dest {
            if rtcp_socket.send_to(packet, dest).await.is_err() {
                return Err(());
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

// gst-rtsp-server drops a TCP player whose backlog grows, so one slow player does not stall the rest
const PLAYER_WRITE_TIMEOUT: core::time::Duration = core::time::Duration::from_secs(2);

async fn write_to_player(control: &mut tokio::net::TcpStream, bytes: &[u8]) -> Result<(), ()> {
    match tokio::time::timeout(PLAYER_WRITE_TIMEOUT, control.write_all(bytes)).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(_)) | Err(_) => Err(()),
    }
}

/// # Example
///
/// ```no_run
/// use core::time::Duration;
/// use g2g_plugins::rtspserversink::RtspServerSink;
///
/// let sink = RtspServerSink::new("0.0.0.0:8554".parse().unwrap())
///     .with_rtp(96, 0x1234_5678)
///     .with_rtcp_sr_interval(Duration::from_secs(5));
/// ```
#[derive(Debug)]
pub struct RtspServerSink {
    rtsp_addr: SocketAddr,
    payload_type: u8,
    ssrc: u32,
    max_payload: usize,
    listener: Option<StdTcpListener>,
    sr_interval_ns: u64,
    session_timeout_ns: u64,
    // one per input, in SDP order
    tracks: Vec<ServedTrack>,
    // Runtime, established lazily on the first buffer.
    tcp: Option<tokio::net::TcpListener>,
    clients: Vec<Client>,
    started: bool,
    configured: bool,
    frames_sent: u64,
    packets_sent: u64,
    sender_reports_sent: u64,
    keyframe_request_pending: bool,
    newest_pts_ns: Option<u64>,
    /// Last SPS and PPS seen in the stream, offered in the SDP as
    /// `sprop-parameter-sets`. Refreshed as they flow, so a mid-stream
    /// resolution change reaches players that DESCRIBE after it.
    sps: Option<Vec<u8>>,
    pps: Option<Vec<u8>>,
    // sender reports of every track map PTS to wall clock with this one clock
    wall_clock: Option<PtsWallClock>,
}

impl RtspServerSink {
    /// Listen for RTSP players on `rtsp_addr` (e.g. `0.0.0.0:8554`).
    pub fn new(rtsp_addr: SocketAddr) -> Self {
        Self {
            rtsp_addr,
            payload_type: DEFAULT_PAYLOAD_TYPE,
            ssrc: 0,
            max_payload: DEFAULT_MAX_PAYLOAD,
            listener: None,
            sr_interval_ns: DEFAULT_SR_INTERVAL_NS,
            session_timeout_ns: DEFAULT_SESSION_TIMEOUT_SECS as u64 * 1_000_000_000,
            tracks: Vec::from([ServedTrack::default()]),
            tcp: None,
            clients: Vec::new(),
            started: false,
            configured: false,
            frames_sent: 0,
            packets_sent: 0,
            sender_reports_sent: 0,
            keyframe_request_pending: false,
            newest_pts_ns: None,
            sps: None,
            pps: None,
            wall_clock: None,
        }
    }

    /// Use an already-bound listener (so a test can pick an ephemeral port).
    pub fn from_listener(listener: StdTcpListener) -> Result<Self, G2gError> {
        let addr = listener.local_addr().map_err(io_err)?;
        Ok(Self {
            listener: Some(listener),
            configured: true,
            ..Self::new(addr)
        })
    }

    pub fn with_tracks(mut self, count: usize) -> Self {
        self.tracks = (0..count).map(|_| ServedTrack::default()).collect();
        self
    }

    /// Set the RTP payload type and SSRC carried in every packet. Track `n`
    /// sends as SSRC `ssrc + n`.
    pub fn with_rtp(mut self, payload_type: u8, ssrc: u32) -> Self {
        self.payload_type = payload_type & 0x7F;
        self.ssrc = ssrc;
        self
    }

    /// RTCP sender-report interval (default 5 s).
    pub fn with_rtcp_sr_interval(mut self, interval: Duration) -> Self {
        self.sr_interval_ns = interval.as_nanos() as u64;
        self
    }

    /// Session timeout: a player silent past this on both the control channel
    /// and RTCP is reaped. Advertised (rounded up to whole seconds) in the
    /// SETUP response so clients know their keepalive cadence.
    pub fn with_session_timeout(mut self, timeout: Duration) -> Self {
        self.session_timeout_ns = (timeout.as_nanos() as u64).max(1);
        self
    }

    /// The timeout to advertise in `Session: id;timeout=N`, in whole seconds.
    /// With reaping disabled the RFC default is advertised, so clients keep a
    /// standard keepalive cadence.
    fn session_timeout_secs(&self) -> u32 {
        if self.session_timeout_ns == u64::MAX {
            return DEFAULT_SESSION_TIMEOUT_SECS;
        }
        self.session_timeout_ns.div_ceil(1_000_000_000).max(1) as u32
    }

    fn track_ssrc(&self, track_index: usize) -> u32 {
        self.ssrc.wrapping_add(track_index as u32)
    }

    fn sdp(&self) -> Result<alloc::string::String, G2gError> {
        let mut sections = Vec::with_capacity(self.tracks.len());
        for (index, track) in self.tracks.iter().enumerate() {
            sections.push(match track.media.ok_or(G2gError::NotConfigured)? {
                TrackMedia::H264 => h264_media_section(
                    self.payload_type,
                    self.sps.as_deref(),
                    self.pps.as_deref(),
                    index,
                ),
                TrackMedia::OnvifMetadata => onvif_metadata_media_section(self.payload_type, index),
            });
        }
        Ok(session_description(&sections))
    }

    /// Cache the parameter sets an access unit carries, so the SDP offered to the
    /// next player describes the stream. Called before the handshake work in
    /// `process`, so the very first buffer's SPS / PPS are already in hand when
    /// the first DESCRIBE is answered.
    fn latch_parameter_sets(&mut self, au: &[u8]) {
        let (sps, pps) = crate::annexb::h264_parameter_sets(au);
        if let Some(s) = sps.into_iter().next() {
            self.sps = Some(s);
        }
        if let Some(p) = pps.into_iter().next() {
            self.pps = Some(p);
        }
    }

    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    pub fn packets_sent(&self) -> u64 {
        self.packets_sent
    }

    pub fn sender_reports_sent(&self) -> u64 {
        self.sender_reports_sent
    }

    /// Number of players currently connected (PLAYing or mid-handshake).
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    pub fn eos_seen(&self) -> bool {
        self.tracks.iter().all(|track| track.ended)
    }

    /// Bind each track's RTP socket pair and promote the listener to tokio, on
    /// the first buffer.
    async fn start(&mut self) -> Result<(), G2gError> {
        let std_listener = self.listener.take().ok_or(G2gError::NotConfigured)?;
        std_listener.set_nonblocking(true).map_err(io_err)?;
        self.tcp = Some(tokio::net::TcpListener::from_std(std_listener).map_err(io_err)?);
        for track in &mut self.tracks {
            let (rtp_socket, rtcp_socket) = bind_rtp_pair().await?;
            track.rtp_socket = Some(rtp_socket);
            track.rtcp_socket = Some(rtcp_socket);
        }
        Ok(())
    }

    /// Accept every player whose TCP connection is already queued (non-blocking),
    /// adding each as a handshaking client.
    async fn accept_new(&mut self) -> Result<(), G2gError> {
        let Some(listener) = self.tcp.as_ref() else {
            return Ok(());
        };
        let mut server_rtp_ports = Vec::with_capacity(self.tracks.len());
        for track in &self.tracks {
            let socket = track.rtp_socket.as_ref().ok_or(G2gError::NotConfigured)?;
            server_rtp_ports.push(socket.local_addr().map_err(io_err)?.port());
        }
        let ssrcs: Vec<u32> = (0..self.tracks.len())
            .map(|index| self.track_ssrc(index))
            .collect();
        let sdp = self.sdp()?;
        // A zero timeout polls accept once: take a queued connection or stop.
        while let Ok(Ok((control, peer))) =
            tokio::time::timeout(Duration::from_millis(0), listener.accept()).await
        {
            let responder = RtspResponder::new(sdp.clone(), server_rtp_ports[0], self.ssrc)
                .serving_tracks(server_rtp_ports.clone(), ssrcs.clone())
                .with_session_timeout_secs(self.session_timeout_secs());
            self.clients
                .push(Client::new(control, responder, peer.ip()));
        }
        Ok(())
    }

    /// Service every client's control channel (non-blocking), reaping any that
    /// disconnected, tore down, or fell silent past the session timeout.
    /// PLAYing clients are advanced too, so a mid-stream TEARDOWN or
    /// control-channel close is detected and keepalive requests are answered.
    async fn advance_handshakes(&mut self) {
        let now = g2g_core::metrics::monotonic_ns();
        let stream_rtptime = self
            .newest_pts_ns
            .map(crate::rtpklv::rtp_timestamp_from_pts);
        let mut i = 0;
        while i < self.clients.len() {
            let keep = self.clients[i].advance(stream_rtptime).await
                && now.saturating_sub(self.clients[i].last_activity_ns) <= self.session_timeout_ns;
            if keep {
                i += 1;
            } else {
                self.clients.swap_remove(i);
            }
        }
    }

    /// The RTCP pass: drain each track's server RTCP socket (a player's receiver
    /// report counts as keepalive activity), then send a sender report on every
    /// PLAYing track whose interval elapsed, reaping any player whose send fails.
    async fn service_rtcp(&mut self) -> Result<(), G2gError> {
        let now = g2g_core::metrics::monotonic_ns();
        let mut buf = [0u8; 1500];
        for (track_index, track) in self.tracks.iter().enumerate() {
            let Some(rtcp_socket) = track.rtcp_socket.as_ref() else {
                continue;
            };
            while let Ok((n, src)) = rtcp_socket.try_recv_from(&mut buf) {
                if !rtcp::is_rtcp(&buf[..n]) {
                    continue;
                }
                // Attribute by the negotiated RTCP address, falling back to the
                // peer IP (a client may send from an unbound source port).
                let by_addr = self
                    .clients
                    .iter_mut()
                    .find(|c| c.tracks[track_index].rtcp_dest == Some(src));
                let stamped = match by_addr {
                    Some(c) => Some(c),
                    None => self.clients.iter_mut().find(|c| c.peer_ip == src.ip()),
                };
                if let Some(c) = stamped {
                    c.last_activity_ns = now;
                }
            }
        }
        let Some(wall_clock) = self.wall_clock else {
            return Ok(());
        };
        let mut i = 0;
        'clients: while i < self.clients.len() {
            for track_index in 0..self.tracks.len() {
                let Some(rtcp_socket) = self.tracks[track_index].rtcp_socket.as_ref() else {
                    continue;
                };
                let track = &self.clients[i].tracks[track_index];
                let due =
                    track.playing() && now.saturating_sub(track.last_sr_ns) >= self.sr_interval_ns;
                let Some(last_rtp_ts) = track.last_rtp_ts.filter(|_| due) else {
                    continue;
                };
                // the report names the instant of the RTP timestamp, so both tracks place a frame alike
                let unix_nanos = wall_clock.unix_nanos_at(pts_ns_at_rtp_tick(track.last_pts_ns));
                let ntp =
                    g2g_core::ntp_from_unix_nanos(unix_nanos).ok_or(G2gError::CapsMismatch)?;
                let sr = track.sender_counters.sender_report(
                    self.ssrc.wrapping_add(track_index as u32),
                    ntp,
                    last_rtp_ts,
                );
                match self.clients[i]
                    .send_rtcp(track_index, rtcp_socket, &sr)
                    .await
                {
                    Ok(true) => {
                        self.clients[i].tracks[track_index].last_sr_ns = now;
                        self.sender_reports_sent += 1;
                    }
                    Ok(false) => {}
                    Err(()) => {
                        self.clients.swap_remove(i);
                        continue 'clients;
                    }
                }
            }
            i += 1;
        }
        Ok(())
    }

    /// Broadcast one frame of one track to every player PLAYing it, reaping any
    /// whose send fails.
    async fn broadcast(
        &mut self,
        track_index: usize,
        media: TrackMedia,
        bytes: &[u8],
        pts_ns: u64,
    ) -> Result<(), G2gError> {
        let timestamp = crate::rtpklv::rtp_timestamp_from_pts(pts_ns);
        self.newest_pts_ns = Some(
            self.newest_pts_ns
                .map_or(pts_ns, |newest| newest.max(pts_ns)),
        );
        let ssrc = self.track_ssrc(track_index);
        let socket = self.tracks[track_index]
            .rtp_socket
            .as_ref()
            .ok_or(G2gError::NotConfigured)?;
        // packetized once, each player stamps its own sequence numbers
        let (mut packets, keyframe) = match media {
            TrackMedia::H264 => (
                RtpH264Packetizer::new(self.payload_type, ssrc)
                    .with_max_payload(self.max_payload)
                    .packetize(bytes, timestamp),
                Some(crate::annexb::au_is_keyframe(VideoCodec::H264, bytes)),
            ),
            // the ONVIF Streaming spec splits a document across packets and marks the last, as RFC 6597 does KLV
            TrackMedia::OnvifMetadata => (
                RtpKlvPacketizer::new(self.payload_type, ssrc)
                    .with_max_payload(self.max_payload)
                    .packetize(bytes, timestamp),
                None,
            ),
        };
        let mut i = 0;
        while i < self.clients.len() {
            let client = &mut self.clients[i];
            if let Some(keyframe) = keyframe {
                client.awaiting_keyframe &= !keyframe;
                if client.awaiting_keyframe {
                    i += 1;
                    continue;
                }
            }
            if !client.tracks[track_index].playing() {
                i += 1;
                continue;
            }
            match client
                .send_packets(track_index, socket, &mut packets, timestamp, pts_ns)
                .await
            {
                Ok(pkts) => {
                    self.packets_sent += pkts;
                    i += 1;
                }
                Err(()) => {
                    self.clients.swap_remove(i);
                }
            }
        }
        Ok(())
    }

    fn configure_track(&mut self, input: usize, caps: &Caps) -> Result<(), G2gError> {
        let media = TrackMedia::of(caps).ok_or(G2gError::CapsMismatch)?;
        self.tracks
            .get_mut(input)
            .ok_or(G2gError::NotConfigured)?
            .media = Some(media);
        if self.listener.is_none() && self.tcp.is_none() {
            self.listener = Some(StdTcpListener::bind(self.rtsp_addr).map_err(io_err)?);
        }
        self.configured = true;
        Ok(())
    }

    async fn send_frame(&mut self, input: usize, frame: Frame) -> Result<(), G2gError> {
        if !self.configured {
            return Err(G2gError::NotConfigured);
        }
        let media = self
            .tracks
            .get(input)
            .and_then(|track| track.media)
            .ok_or(G2gError::NotConfigured)?;
        let slice = frame
            .domain
            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
        // Before any handshake: the SDP a DESCRIBE gets back has to
        // describe this stream.
        if media == TrackMedia::H264 {
            self.latch_parameter_sets(slice);
        }
        if !self.started {
            self.start().await?;
            self.started = true;
        }
        self.wall_clock
            .get_or_insert_with(|| PtsWallClock::from_first_frame(&frame));
        // a buffer never waits for a player, so a player starts at the current frame
        self.accept_new().await?;
        self.advance_handshakes().await;
        let pts_ns = frame.timing.pts_ns;
        if media == TrackMedia::H264 {
            let keyframe = crate::annexb::au_is_keyframe(VideoCodec::H264, slice);
            self.keyframe_request_pending |= self.clients.iter().any(|client| {
                client.tracks[input].playing() && client.awaiting_keyframe && !keyframe
            });
        }
        self.broadcast(input, media, slice, pts_ns).await?;
        self.frames_sent += 1;
        self.service_rtcp().await
    }

    // RTP has no in-band end marker: an RTCP BYE tells each player the track ended cleanly rather than stalling to a timeout
    async fn end_track(&mut self, input: usize) {
        let ssrc = self.track_ssrc(input);
        let Some(track) = self.tracks.get_mut(input) else {
            return;
        };
        track.ended = true;
        let Some(rtcp_socket) = track.rtcp_socket.as_ref() else {
            return;
        };
        let bye = rtcp::build_bye(ssrc);
        for client in &mut self.clients {
            if client.tracks[input].playing() {
                let _ = client.send_rtcp(input, rtcp_socket, &bye).await;
            }
        }
    }

    async fn handle_packet(
        &mut self,
        input: usize,
        packet: PipelinePacket,
    ) -> Result<(), G2gError> {
        match packet {
            PipelinePacket::DataFrame(frame) => self.send_frame(input, frame).await,
            PipelinePacket::Eos => {
                self.end_track(input).await;
                Ok(())
            }
            // Sequence numbers persist across a seek (loss is tracked by gaps),
            // and geometry refinement lives in the in-band SPS, not in RTP/SDP.
            _ => Ok(()),
        }
    }
}

const RTSP_SERVER_SINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new("address", PropKind::Str, "IP to listen for RTSP players on")
        .with_default("0.0.0.0"),
    PropertySpec::new("port", PropKind::Uint, "RTSP TCP port to listen on")
        .with_default("8554")
        .with_range("0", "65535"),
    PropertySpec::new(
        "payload-type",
        PropKind::Uint,
        "RTP payload type (96..=127)",
    )
    .with_default("96")
    .with_range("0", "127"),
    PropertySpec::new(
        "ssrc",
        PropKind::Uint,
        "RTP synchronization source id of the first track, the next tracks counting up from it",
    ),
    PropertySpec::new(
        "rtcp-sr-interval",
        PropKind::Uint,
        "RTCP sender-report interval in milliseconds",
    )
    .with_default("5000")
    .with_range("1", "60000"),
    PropertySpec::new(
        "timeout",
        PropKind::Uint,
        "session timeout in seconds (a player silent past it is reaped; 0 = never)",
    )
    .with_default("60")
    .with_range("0", "604800"),
];

impl RtspServerSink {
    fn set_sink_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(r) = crate::netprop::set_addr_prop(&mut self.rtsp_addr, "address", name, &value)
        {
            return r;
        }
        match name {
            "payload-type" => {
                let pt = value.as_uint().ok_or(PropError::Type)?;
                if pt > 127 {
                    return Err(PropError::Value);
                }
                self.payload_type = pt as u8;
                Ok(())
            }
            "ssrc" => {
                self.ssrc = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "rtcp-sr-interval" => {
                let ms = value.as_uint().ok_or(PropError::Type)?;
                if !(1..=60_000).contains(&ms) {
                    return Err(PropError::Value);
                }
                self.sr_interval_ns = ms * 1_000_000;
                Ok(())
            }
            "timeout" => {
                let secs = value.as_uint().ok_or(PropError::Type)?;
                if secs > 604_800 {
                    return Err(PropError::Value);
                }
                self.session_timeout_ns = match secs {
                    0 => u64::MAX, // never reap
                    s => s * 1_000_000_000,
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_sink_property(&self, name: &str) -> Option<PropValue> {
        if let Some(v) = crate::netprop::get_addr_prop(&self.rtsp_addr, "address", name) {
            return Some(v);
        }
        match name {
            "payload-type" => Some(PropValue::Uint(self.payload_type as u64)),
            "ssrc" => Some(PropValue::Uint(self.ssrc as u64)),
            "rtcp-sr-interval" => Some(PropValue::Uint(self.sr_interval_ns / 1_000_000)),
            "timeout" => Some(PropValue::Uint(match self.session_timeout_ns {
                u64::MAX => 0,
                _ => self.session_timeout_secs() as u64,
            })),
            _ => None,
        }
    }
}

fn element_metadata() -> ElementMetadata {
    ElementMetadata::new(
        "RTSP server sink",
        "Sink/Network",
        "Hosts an RTSP endpoint serving H.264 and ONVIF metadata over RTP to multiple players",
        crate::AARON_BOXER,
    )
}

/// Bind the server's RTP/RTCP UDP sockets on adjacent ports, so the SETUP
/// response's `server_port=N-(N+1)` pair is real on both halves. The ephemeral
/// RTP port's neighbor may be taken; retry with a fresh pair a few times.
pub(crate) async fn bind_rtp_pair(
) -> Result<(tokio::net::UdpSocket, tokio::net::UdpSocket), G2gError> {
    for _ in 0..16 {
        let rtp = tokio::net::UdpSocket::bind(("0.0.0.0", 0))
            .await
            .map_err(io_err)?;
        let port = rtp.local_addr().map_err(io_err)?.port();
        let Some(rtcp_port) = port.checked_add(1) else {
            continue;
        };
        if let Ok(rtcp) = tokio::net::UdpSocket::bind(("0.0.0.0", rtcp_port)).await {
            return Ok((rtp, rtcp));
        }
    }
    Err(G2gError::Hardware(HardwareError::Other))
}

impl AsyncElement for RtspServerSink {
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
        intercept_served_caps(upstream_caps)
    }

    fn take_reconfigure(&mut self) -> Option<Reconfigure> {
        // gst-rtsp-server's ensure-keyunit-on-start asks the encoder for a keyframe when a player starts
        core::mem::take(&mut self.keyframe_request_pending).then_some(Reconfigure::ForceKeyframe)
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(served_caps())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configure_track(0, absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        element_metadata()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        RTSP_SERVER_SINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.set_sink_property(name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.get_sink_property(name)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(self.handle_packet(0, packet))
    }
}

impl MultiInputElement for RtspServerSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn input_count(&self) -> usize {
        self.tracks.len()
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        intercept_served_caps(upstream_caps)
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(served_caps())
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.configure_track(input, absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    // terminal: the players are the destination, so this placeholder is never consulted
    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(h264_any())
    }

    fn is_terminal(&self) -> bool {
        true
    }

    fn metadata(&self) -> ElementMetadata {
        element_metadata()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        RTSP_SERVER_SINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.set_sink_property(name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.get_sink_property(name)
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(self.handle_packet(input, packet))
    }
}

impl PadTemplates for RtspServerSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(served_caps())])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtspserver::sdp_h264;
    use alloc::vec;
    use tokio::io::AsyncWriteExt;

    async fn client_pair() -> (Client, tokio::net::TcpStream) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let peer = tokio::net::TcpStream::connect(addr).await.unwrap();
        let (server, _) = listener.accept().await.unwrap();
        let client = Client::new(
            server,
            RtspResponder::new(sdp_h264(96, None, None), 6000, 0x1234_5678)
                .serving_tracks(vec![6000], vec![0x1234_5678]),
            std::net::IpAddr::from([127, 0, 0, 1]),
        );
        (client, peer)
    }

    #[tokio::test]
    async fn oversized_pending_request_reaps_the_client() {
        let (mut client, mut peer) = client_pair().await;
        // A never-terminating request (no double CRLF): the writer keeps the
        // connection open, so any reap is from the buffer cap, not a close.
        let writer = tokio::spawn(async move {
            let junk = vec![b'A'; MAX_PENDING + CTRL_BUF + 16];
            let _ = peer.write_all(&junk).await;
            peer // hold the socket open
        });
        let mut reaped = false;
        for _ in 0..10_000 {
            if !client.advance(None).await {
                reaped = true;
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(reaped, "a client overflowing the control buffer is reaped");
        let _ = writer.await;
    }

    #[tokio::test]
    async fn playing_client_is_reaped_when_control_channel_closes() {
        let (mut client, peer) = client_pair().await;
        client.tracks[0].dest = Some(SocketAddr::new(client.peer_ip, 5000));
        client.tracks[0].next_sequence = Some(0);
        assert!(client.tracks[0].playing());
        drop(peer); // peer closes the control channel mid-stream
        let mut reaped = false;
        for _ in 0..10_000 {
            if !client.advance(None).await {
                reaped = true;
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(
            reaped,
            "a playing client whose control channel closed is reaped"
        );
    }
}

//! Native P2P WebRTC data-channel elements on the sans-IO [`str0m`] stack
//! (ICE / DTLS / SCTP), the native unification of the wasm-only browser
//! data-channel [`crate::webrtcsrc::WebRtcSrc`]. Two elements sharing one
//! handshake:
//!
//! - [`WebRtcDataSrc`] (`SourceLoop`): owns an `Rtc` + `UdpSocket`, does the
//!   P2P SDP handshake over an [`SdpChannel`], then drains
//!   [`Event::ChannelData`] and emits each binary message as a system-memory
//!   `Caps::ByteStream` `DataFrame`, ending on the sink's end-of-stream marker.
//!   Mirrors the wasm `WebRtcSrc` surface so the native and browser ingest paths
//!   look the same downstream.
//! - [`WebRtcDataSink`] (`AsyncElement`): on the first frame it runs the
//!   handshake and spawns a task owning the `Rtc` (so the element stays `Send`,
//!   as `WebRtcSink` does), then writes each incoming `DataFrame`'s bytes as one
//!   binary SCTP channel message. On input EOS it sends a one-byte text-flagged
//!   message as an end-of-stream marker (str0m does not surface a remote channel
//!   close as an event, so the marker, not a stream reset, ends the src), which
//!   the ordered reliable channel delivers after the data.
//!
//! Why standalone elements, not a lane folded into [`WebRtcDuplexSession`]: that
//! session is a media element (H.264 / Opus RTP writers, NACK / RTX counters,
//! PLI / BWE reverse channels). A data channel is SCTP bytes with none of that,
//! and the existing element taxonomy already pairs a `SourceLoop` ingest with an
//! `AsyncElement` egress (`WebRtcSrc` / `WebRtcSink`). Keeping them separate is
//! the smaller change and matches the wasm surface; both reuse the P2P
//! [`SdpChannel`] signalling seam and the shared handshake below, so the offer /
//! answer logic is written once.
//!
//! The offerer adds the channel ([`SdpApi::add_channel_with_config`]); the
//! answerer inherits it from the offer. Reliability (`ordered` / retransmits /
//! packet-lifetime) is set by the channel creator, so it applies in the Offerer
//! role and is inherited from the peer as the Answerer, exactly as the browser
//! `RTCPeerConnection.createDataChannel` model works.
//!
//! Message size: str0m caps a single data-channel message at 64 KiB (65536
//! bytes, sctp-proto's default max receive message size); SCTP fragments a
//! message across DATA chunks and reassembles it, so one `write` up to that size
//! arrives as one `ChannelData`. Larger payloads must be chunked by the caller.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::net::SocketAddr;
use std::time::Instant;

use tokio::net::UdpSocket;
use tokio::sync::mpsc;

use str0m::change::{SdpAnswer, SdpOffer};
use str0m::channel::{ChannelConfig, ChannelId, Reliability};
use str0m::crypto::from_feature_flags;
use str0m::net::{Protocol, Receive};
use str0m::{Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming,
    G2gError, HardwareError, MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

use crate::filesink::io_err;
use crate::webrtc_util::{add_ice_candidates, select_host_ip};
use crate::webrtcduplex::{SdpChannel, SignalRole};

/// Bounded depth of the element->session channel in the sink: backpressures the
/// pipeline if the SCTP association falls behind.
const DEFAULT_QUEUE_DEPTH: usize = 256;

/// The largest single message str0m accepts on a data channel (sctp-proto's
/// default max receive message size, 64 KiB). SCTP fragments and reassembles up
/// to this; a larger payload is rejected on write.
pub const MAX_MESSAGE_SIZE: usize = 65536;

/// Reliability knobs for the data channel, shared by both elements. Applied by
/// whichever peer creates the channel (the Offerer); the Answerer inherits the
/// peer's settings. Mirrors the browser `RTCDataChannelInit`.
#[derive(Debug, Clone)]
struct ChannelSettings {
    label: String,
    ordered: bool,
    /// Max retransmits before giving up (`None` = reliable, unbounded).
    max_retransmits: Option<u16>,
    /// Max time (ms) to retransmit a message (`None` = no limit).
    max_packet_lifetime: Option<u16>,
}

impl Default for ChannelSettings {
    fn default() -> Self {
        Self {
            label: String::from("g2g"),
            ordered: true,
            max_retransmits: None,
            max_packet_lifetime: None,
        }
    }
}

impl ChannelSettings {
    /// Resolve to a str0m [`ChannelConfig`]. Packet-lifetime wins over
    /// retransmits (the two SCTP reliability modes are mutually exclusive);
    /// neither set means reliable.
    fn to_config(&self) -> ChannelConfig {
        let reliability = if let Some(lifetime) = self.max_packet_lifetime {
            Reliability::MaxPacketLifetime { lifetime }
        } else if let Some(retransmits) = self.max_retransmits {
            Reliability::MaxRetransmits { retransmits }
        } else {
            Reliability::Reliable
        };
        ChannelConfig {
            label: self.label.clone(),
            ordered: self.ordered,
            reliability,
            ..Default::default()
        }
    }
}

/// The properties both data-channel elements expose (the sink handles them; the
/// `SourceLoop` trait carries no property surface, so the src configures these
/// only via builders). Reliability applies when this peer creates the channel.
static DATA_CHANNEL_PROPS: &[PropertySpec] = &[
    PropertySpec::new("label", PropKind::Str, "data-channel label"),
    PropertySpec::new(
        "ordered",
        PropKind::Bool,
        "deliver messages in order (applies when this peer creates the channel)",
    ),
    PropertySpec::new(
        "max-retransmits",
        PropKind::Int,
        "max retransmits before dropping a message, -1 = reliable (creator only)",
    ),
    PropertySpec::new(
        "max-packet-lifetime",
        PropKind::Int,
        "max ms to retransmit a message, -1 = unset; wins over max-retransmits (creator only)",
    ),
];

/// Apply one `DATA_CHANNEL_PROPS` name to the shared settings.
fn set_channel_prop(
    settings: &mut ChannelSettings,
    name: &str,
    value: PropValue,
) -> Result<(), PropError> {
    match name {
        "label" => {
            settings.label = value.as_str().ok_or(PropError::Type)?.into();
            Ok(())
        }
        "ordered" => {
            settings.ordered = value.as_bool().ok_or(PropError::Type)?;
            Ok(())
        }
        "max-retransmits" => {
            let v = value.as_int().ok_or(PropError::Type)?;
            settings.max_retransmits = to_u16_opt(v);
            // The two reliability modes are exclusive: setting one clears the other.
            if settings.max_retransmits.is_some() {
                settings.max_packet_lifetime = None;
            }
            Ok(())
        }
        "max-packet-lifetime" => {
            let v = value.as_int().ok_or(PropError::Type)?;
            settings.max_packet_lifetime = to_u16_opt(v);
            if settings.max_packet_lifetime.is_some() {
                settings.max_retransmits = None;
            }
            Ok(())
        }
        _ => Err(PropError::Unknown),
    }
}

/// Read one `DATA_CHANNEL_PROPS` name back (`-1` for an unset reliability knob).
fn get_channel_prop(settings: &ChannelSettings, name: &str) -> Option<PropValue> {
    match name {
        "label" => Some(PropValue::Str(settings.label.clone())),
        "ordered" => Some(PropValue::Bool(settings.ordered)),
        "max-retransmits" => Some(PropValue::Int(
            settings.max_retransmits.map(|v| v as i64).unwrap_or(-1),
        )),
        "max-packet-lifetime" => Some(PropValue::Int(
            settings.max_packet_lifetime.map(|v| v as i64).unwrap_or(-1),
        )),
        _ => None,
    }
}

/// A negative value clears the knob; otherwise it is clamped into `u16`.
fn to_u16_opt(v: i64) -> Option<u16> {
    if v < 0 {
        None
    } else {
        Some(v.min(u16::MAX as i64) as u16)
    }
}

/// Build the `Rtc` + socket and run the P2P SDP handshake over `sig`. The
/// Offerer adds the data channel with `settings`; the Answerer inherits it.
/// Returns the connected `Rtc`, its socket, and the local address for feeding
/// received datagrams back in.
async fn connect(
    role: SignalRole,
    mut sig: SdpChannel,
    settings: &ChannelSettings,
) -> Result<(Rtc, UdpSocket, SocketAddr), G2gError> {
    let hw = || G2gError::Hardware(HardwareError::Other);

    let host_ip = select_host_ip();
    let socket = UdpSocket::bind((host_ip, 0)).await.map_err(io_err)?;
    let local = socket.local_addr().map_err(io_err)?;

    let mut rtc = RtcConfig::new()
        .set_crypto_provider(Arc::new(from_feature_flags()))
        .build(Instant::now());
    // Host candidate rides in the SDP, so it must be added before offer / answer.
    add_ice_candidates(&mut rtc, &socket, None).await?;

    match role {
        SignalRole::Offerer => {
            let (offer_sdp, pending) = {
                let mut api = rtc.sdp_api();
                api.add_channel_with_config(settings.to_config());
                let (offer, pending) = api.apply().ok_or_else(hw)?;
                (offer.to_sdp_string(), pending)
            };
            if !sig.send_sdp(offer_sdp).await {
                return Err(hw());
            }
            let answer_sdp = sig.recv_sdp().await.ok_or_else(hw)?;
            let answer = SdpAnswer::from_sdp_string(&answer_sdp).map_err(|_| hw())?;
            rtc.sdp_api()
                .accept_answer(pending, answer)
                .map_err(|_| hw())?;
        }
        SignalRole::Answerer => {
            let offer_sdp = sig.recv_sdp().await.ok_or_else(hw)?;
            let offer = SdpOffer::from_sdp_string(&offer_sdp).map_err(|_| hw())?;
            let answer = rtc.sdp_api().accept_offer(offer).map_err(|_| hw())?;
            if !sig.send_sdp(answer.to_sdp_string()).await {
                return Err(hw());
            }
        }
    }

    Ok((rtc, socket, local))
}

/// Feed one received datagram into str0m. `false` means the association is dead.
fn feed(rtc: &mut Rtc, local: SocketAddr, source: SocketAddr, contents: &[u8]) -> bool {
    let Ok(contents) = contents.try_into() else {
        return true;
    };
    let input = Input::Receive(
        Instant::now(),
        Receive {
            proto: Protocol::Udp,
            source,
            destination: local,
            contents,
        },
    );
    rtc.handle_input(input).is_ok()
}

/// Native P2P WebRTC data-channel ingest. See the module docs.
#[derive(Debug)]
pub struct WebRtcDataSrc {
    role: SignalRole,
    sig: Option<SdpChannel>,
    settings: ChannelSettings,
    encoding: ByteStreamEncoding,
    configured: bool,
}

impl WebRtcDataSrc {
    /// A data-channel source with the given signalling `role` (typically
    /// [`SignalRole::Answerer`], accepting the peer's channel) and SDP channel.
    /// Emits `Caps::ByteStream` (default `MpegTs`, override with
    /// [`Self::with_encoding`]).
    pub fn new(role: SignalRole, sig: SdpChannel) -> Self {
        Self {
            role,
            sig: Some(sig),
            settings: ChannelSettings::default(),
            encoding: ByteStreamEncoding::MpegTs,
            configured: false,
        }
    }

    /// Declare the byte-stream container the messages carry (default `MpegTs`).
    /// The transport is codec-agnostic; this only sets the emitted caps.
    pub fn with_encoding(mut self, encoding: ByteStreamEncoding) -> Self {
        self.encoding = encoding;
        self
    }

    /// Channel label, used when this source is the Offerer (it inherits the
    /// peer's label as Answerer).
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.settings.label = label.into();
        self
    }

    /// Deliver messages in order (default true). Applies when this source
    /// creates the channel (Offerer role).
    pub fn with_ordered(mut self, ordered: bool) -> Self {
        self.settings.ordered = ordered;
        self
    }

    /// Cap retransmits per message, dropping the reliable default. Applies when
    /// this source creates the channel (Offerer role).
    pub fn with_max_retransmits(mut self, retransmits: u16) -> Self {
        self.settings.max_retransmits = Some(retransmits);
        self.settings.max_packet_lifetime = None;
        self
    }

    /// Cap the retransmit lifetime (ms) per message. Applies when this source
    /// creates the channel (Offerer role).
    pub fn with_max_packet_lifetime(mut self, lifetime_ms: u16) -> Self {
        self.settings.max_packet_lifetime = Some(lifetime_ms);
        self.settings.max_retransmits = None;
        self
    }

    fn caps(&self) -> Caps {
        Caps::ByteStream {
            encoding: self.encoding,
        }
    }
}

impl SourceLoop for WebRtcDataSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(self.caps()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        let role = self.role;
        let settings = self.settings.clone();
        let caps = self.caps();
        let sig = self.sig.take();
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let hw = || G2gError::Hardware(HardwareError::Other);
            let sig = sig.ok_or_else(hw)?;
            let (mut rtc, socket, local) = connect(role, sig, &settings).await?;

            out.push(PipelinePacket::CapsChanged(caps)).await?;

            let mut buf = alloc::vec![0u8; 2048];
            let mut sequence = 0u64;

            'outer: loop {
                let mut messages: Vec<Vec<u8>> = Vec::new();
                let mut ended = false;
                let deadline = loop {
                    match rtc.poll_output() {
                        Ok(Output::Timeout(t)) => break t,
                        Ok(Output::Transmit(t)) => {
                            let _ = socket.send_to(&t.contents, t.destination).await;
                        }
                        // A text-flagged message is the peer's end-of-stream
                        // marker (see the sink); binary messages are data.
                        Ok(Output::Event(Event::ChannelData(cd))) => {
                            if cd.binary {
                                messages.push(cd.data);
                            } else {
                                ended = true;
                                break Instant::now();
                            }
                        }
                        Ok(Output::Event(Event::ChannelClose(_)))
                        | Ok(Output::Event(Event::IceConnectionStateChange(
                            IceConnectionState::Disconnected,
                        ))) => {
                            ended = true;
                            break Instant::now();
                        }
                        Ok(Output::Event(_)) => {}
                        Err(_) => {
                            ended = true;
                            break Instant::now();
                        }
                    }
                };

                for data in messages {
                    let frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(
                            data.into_boxed_slice(),
                        )),
                        // Raw ingest carries no timing; recovered downstream, as
                        // with FileSrc / the wasm WebRtcSrc.
                        timing: FrameTiming::default(),
                        sequence,
                        meta: Default::default(),
                    };
                    sequence += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }

                if ended {
                    break 'outer;
                }

                let timeout = deadline.saturating_duration_since(Instant::now());
                tokio::select! {
                    r = socket.recv_from(&mut buf) => {
                        let Ok((n, source)) = r else { break 'outer };
                        if !feed(&mut rtc, local, source, &buf[..n]) {
                            break 'outer;
                        }
                    }
                    _ = tokio::time::sleep(timeout) => {
                        if rtc.handle_input(Input::Timeout(Instant::now())).is_err() {
                            break 'outer;
                        }
                    }
                }
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }
}

/// Native P2P WebRTC data-channel egress. See the module docs.
pub struct WebRtcDataSink {
    role: SignalRole,
    sig: Option<SdpChannel>,
    settings: ChannelSettings,
    queue_depth: usize,
    /// How long the session task keeps pumping after sending the end-of-stream
    /// marker, so the final messages reach the peer before the task exits.
    linger: Duration,
    configured: bool,
    /// Set on the first frame, after the handshake spawns the session task.
    tx: Option<mpsc::Sender<Vec<u8>>>,
    frames_sent: u64,
}

impl core::fmt::Debug for WebRtcDataSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WebRtcDataSink")
            .field("role", &self.role)
            .field("settings", &self.settings)
            .field("configured", &self.configured)
            .field("frames_sent", &self.frames_sent)
            .finish()
    }
}

impl WebRtcDataSink {
    /// A data-channel sink with the given signalling `role` (typically
    /// [`SignalRole::Offerer`], creating the channel it writes to) and SDP
    /// channel.
    pub fn new(role: SignalRole, sig: SdpChannel) -> Self {
        Self {
            role,
            sig: Some(sig),
            settings: ChannelSettings::default(),
            queue_depth: DEFAULT_QUEUE_DEPTH,
            linger: Duration::from_millis(1500),
            configured: false,
            tx: None,
            frames_sent: 0,
        }
    }

    /// Channel label, used when this sink creates the channel (Offerer role).
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.settings.label = label.into();
        self
    }

    /// Deliver messages in order (default true). Applies when this sink creates
    /// the channel (Offerer role).
    pub fn with_ordered(mut self, ordered: bool) -> Self {
        self.settings.ordered = ordered;
        self
    }

    /// Cap retransmits per message, dropping the reliable default. Applies when
    /// this sink creates the channel (Offerer role).
    pub fn with_max_retransmits(mut self, retransmits: u16) -> Self {
        self.settings.max_retransmits = Some(retransmits);
        self.settings.max_packet_lifetime = None;
        self
    }

    /// Cap the retransmit lifetime (ms) per message. Applies when this sink
    /// creates the channel (Offerer role).
    pub fn with_max_packet_lifetime(mut self, lifetime_ms: u16) -> Self {
        self.settings.max_packet_lifetime = Some(lifetime_ms);
        self.settings.max_retransmits = None;
        self
    }

    /// Override the bounded element->session channel depth.
    pub fn with_queue_depth(mut self, depth: usize) -> Self {
        self.queue_depth = depth.max(1);
        self
    }

    /// Override the post-input linger window (default 1.5 s).
    pub fn with_linger(mut self, linger: Duration) -> Self {
        self.linger = linger;
        self
    }

    /// Messages handed to the session so far.
    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    /// Run the handshake and spawn the session task (on the first frame, since
    /// it is async and the runner drives `process` inside a tokio runtime).
    async fn start_session(&mut self) -> Result<(), G2gError> {
        let sig = self
            .sig
            .take()
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let (rtc, socket, local) = connect(self.role, sig, &self.settings).await?;
        let (tx, rx) = mpsc::channel::<Vec<u8>>(self.queue_depth);
        tokio::spawn(run_egress(rtc, socket, local, rx, self.linger));
        self.tx = Some(tx);
        Ok(())
    }
}

impl AsyncElement for WebRtcDataSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::ByteStream { .. } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::ByteStream { .. } => {
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DATA_CHANNEL_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        set_channel_prop(&mut self.settings, name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        get_channel_prop(&self.settings, name)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let data = slice.to_vec();
                    if self.tx.is_none() {
                        self.start_session().await?;
                    }
                    if let Some(tx) = &self.tx {
                        // Bounded send backpressures the pipeline; a closed
                        // channel means the session ended.
                        tx.send(data).await.map_err(|_| G2gError::Shutdown)?;
                    }
                    self.frames_sent += 1;
                }
                // Dropping `tx` signals the session task to flush and close the
                // channel, then linger; it winds down on its own.
                PipelinePacket::Eos => {
                    self.tx = None;
                }
                _ => {}
            }
            Ok(())
        })
    }
}

/// The sans-IO egress loop: owns the `Rtc` + socket, writes queued messages on
/// the data channel once it opens, and after the input ends closes the channel
/// and lingers so the last messages and the stream reset reach the peer.
async fn run_egress(
    mut rtc: Rtc,
    socket: UdpSocket,
    local: SocketAddr,
    mut rx: mpsc::Receiver<Vec<u8>>,
    linger: Duration,
) {
    let mut buf = alloc::vec![0u8; 2048];
    let mut channel: Option<ChannelId> = None;
    // A message awaiting buffer space (str0m rejected the last write as full).
    let mut pending: Option<Vec<u8>> = None;
    let mut input_done = false;
    let mut marker_sent = false;
    let mut drain_deadline: Option<Instant> = None;

    loop {
        let deadline = loop {
            match rtc.poll_output() {
                Ok(Output::Timeout(t)) => break t,
                Ok(Output::Transmit(t)) => {
                    let _ = socket.send_to(&t.contents, t.destination).await;
                }
                Ok(Output::Event(Event::ChannelOpen(id, _))) => channel = Some(id),
                Ok(Output::Event(Event::ChannelClose(_)))
                | Ok(Output::Event(Event::IceConnectionStateChange(
                    IceConnectionState::Disconnected,
                ))) => return,
                Ok(Output::Event(_)) => {}
                Err(_) => return,
            }
        };

        // Flush a pending data message once the channel is open. `write` returns
        // Ok(false) when the send buffer cannot hold it; keep it and retry.
        if let (Some(id), Some(msg)) = (channel, pending.take()) {
            match rtc.channel(id).map(|mut c| c.write(true, &msg)) {
                Some(Ok(true)) => {}
                Some(Ok(false)) | None => pending = Some(msg),
                Some(Err(_)) => return,
            }
        }

        // Input ended and all data is queued: send the end-of-stream marker (a
        // one-byte text-flagged message the src reads as EOS, ordered after the
        // data on the reliable channel), then linger so it reaches the peer. A
        // zero-length message is not delivered, so the marker carries one byte.
        if input_done && pending.is_none() && !marker_sent {
            if let Some(id) = channel {
                match rtc.channel(id).map(|mut c| c.write(false, &[0u8])) {
                    Some(Ok(true)) => {
                        marker_sent = true;
                        drain_deadline = Some(Instant::now() + linger);
                    }
                    // Buffer full or channel not ready: retry next iteration.
                    Some(Ok(false)) | None => {}
                    Some(Err(_)) => return,
                }
            }
        }
        if drain_deadline.is_some_and(|dl| Instant::now() >= dl) {
            return;
        }

        // Retry quickly while a message (or the marker) is waiting on buffer space.
        let base = deadline.saturating_duration_since(Instant::now());
        let timeout = if pending.is_some() || (input_done && !marker_sent) {
            base.min(Duration::from_millis(5))
        } else {
            base
        };

        tokio::select! {
            r = socket.recv_from(&mut buf) => {
                let Ok((n, source)) = r else { return };
                if !feed(&mut rtc, local, source, &buf[..n]) {
                    return;
                }
            }
            msg = rx.recv(), if channel.is_some() && pending.is_none() && !input_done => {
                match msg {
                    Some(m) => pending = Some(m),
                    None => input_done = true,
                }
            }
            _ = tokio::time::sleep(timeout) => {
                if rtc.handle_input(Input::Timeout(Instant::now())).is_err() {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_default_is_reliable_ordered() {
        let c = ChannelSettings::default().to_config();
        assert!(c.ordered);
        assert_eq!(c.reliability, Reliability::Reliable);
        assert_eq!(c.label, "g2g");
    }

    #[test]
    fn packet_lifetime_wins_over_retransmits() {
        let s = ChannelSettings {
            max_retransmits: Some(3),
            max_packet_lifetime: Some(500),
            ..Default::default()
        };
        assert_eq!(
            s.to_config().reliability,
            Reliability::MaxPacketLifetime { lifetime: 500 }
        );
        // Retransmits alone map to MaxRetransmits.
        let s2 = ChannelSettings {
            max_retransmits: Some(3),
            ..Default::default()
        };
        assert_eq!(
            s2.to_config().reliability,
            Reliability::MaxRetransmits { retransmits: 3 }
        );
    }

    #[test]
    fn props_round_trip_and_are_mutually_exclusive() {
        let mut s = ChannelSettings::default();
        set_channel_prop(&mut s, "label", PropValue::Str("chan".into())).unwrap();
        set_channel_prop(&mut s, "ordered", PropValue::Bool(false)).unwrap();
        assert_eq!(
            get_channel_prop(&s, "label"),
            Some(PropValue::Str("chan".into()))
        );
        assert_eq!(
            get_channel_prop(&s, "ordered"),
            Some(PropValue::Bool(false))
        );
        // Unset reliability reads back as -1.
        assert_eq!(
            get_channel_prop(&s, "max-retransmits"),
            Some(PropValue::Int(-1))
        );

        set_channel_prop(&mut s, "max-retransmits", PropValue::Int(5)).unwrap();
        assert_eq!(
            get_channel_prop(&s, "max-retransmits"),
            Some(PropValue::Int(5))
        );
        // Setting packet-lifetime clears retransmits.
        set_channel_prop(&mut s, "max-packet-lifetime", PropValue::Int(200)).unwrap();
        assert_eq!(
            get_channel_prop(&s, "max-retransmits"),
            Some(PropValue::Int(-1))
        );
        assert_eq!(
            get_channel_prop(&s, "max-packet-lifetime"),
            Some(PropValue::Int(200))
        );
        // A negative value clears the knob.
        set_channel_prop(&mut s, "max-packet-lifetime", PropValue::Int(-1)).unwrap();
        assert_eq!(s.max_packet_lifetime, None);

        assert_eq!(
            set_channel_prop(&mut s, "nope", PropValue::Int(1)),
            Err(PropError::Unknown)
        );
        assert_eq!(
            set_channel_prop(&mut s, "ordered", PropValue::Int(1)),
            Err(PropError::Type)
        );
    }

    #[test]
    fn sink_accepts_bytestream_rejects_others() {
        let (a, _b) = SdpChannel::pair();
        let sink = WebRtcDataSink::new(SignalRole::Offerer, a);
        assert!(sink
            .intercept_caps(&Caps::ByteStream {
                encoding: ByteStreamEncoding::MpegTs
            })
            .is_ok());
        let bad = Caps::Audio {
            format: g2g_core::AudioFormat::Opus,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(sink.intercept_caps(&bad), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn sink_props_apply_through_element() {
        let (a, _b) = SdpChannel::pair();
        let mut sink = WebRtcDataSink::new(SignalRole::Offerer, a);
        sink.set_property("label", PropValue::Str("dc".into()))
            .unwrap();
        sink.set_property("max-retransmits", PropValue::Int(0))
            .unwrap();
        assert_eq!(
            sink.settings.to_config().reliability,
            Reliability::MaxRetransmits { retransmits: 0 }
        );
        assert_eq!(
            sink.get_property("label"),
            Some(PropValue::Str("dc".into()))
        );
    }

    #[test]
    fn src_emits_declared_bytestream_caps() {
        let (a, _b) = SdpChannel::pair();
        let src =
            WebRtcDataSrc::new(SignalRole::Answerer, a).with_encoding(ByteStreamEncoding::Mp4);
        assert_eq!(
            src.caps(),
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Mp4
            }
        );
    }
}

//! `dtlssrtpenc`: runs the DTLS-SRTP handshake and protects the media it keys.
//!
//! One input pad per flow, `application/x-rtp` or `application/x-rtcp`, and one
//! `application/x-dtls` output carrying both the handshake records and the
//! protected packets, which is what a socket on the far end demultiplexes by the
//! RFC 7983 first byte. Which flow a pad carries comes from the caps its link
//! settled on, the way `srtpenc` takes it, so the pad names `rtp_%u` / `rtcp_%u`
//! select by position and the caps say what each one is.
//!
//! The decoder of the same session is the only source of inbound handshake
//! records, so this element is only half a connection: pair it with a
//! `dtlssrtpdec` on the same `connection-id`, exactly as GStreamer does.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::collections::VecDeque;
use std::time::Instant;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::log::LogSource;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::runtime::{PadKind, PadRequest};
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    MemoryDomain, MultiInputElement, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::dtlssrtp::{
    connection_for_id, format_fingerprint, lock_connection, parse_fingerprint, DtlsSrtpError,
    DtlsSrtpHandle, DtlsSrtpState, FINGERPRINT_LENGTH, PEER_FINGERPRINT_PROPERTY,
};
use crate::srtp::{
    flow_caps, is_fatal_send_error, protect_for_flow, push_frame_bytes, synchronization_source,
    RtcpProtectionMode, SrtpFlow, SrtpMasterKey, SrtpSender,
};

/// How often the connection is serviced while the inputs are silent: often
/// enough that a handshake flight is not left sitting through a media gap,
/// rarely enough that an idle session costs nothing measurable.
const CONNECTION_TICK_NANOSECONDS: u64 = 20_000_000;

/// The RTCP body is encrypted, not only authenticated: a DTLS-keyed session has
/// no reason to leave the reports in the clear, and gst's `dtlssrtpenc` does the
/// same. Unlike `srtpenc` there is no property, since there is no key to share
/// with a peer that expects the other mode.
const RTCP_PROTECTION: RtcpProtectionMode = RtcpProtectionMode::Encrypt;

/// Packets one input holds while the handshake runs. Deep enough to cover a
/// handshake on a link with a normal round trip, shallow enough that a peer that
/// never answers cannot grow the queue without bound.
const HELD_PACKETS_PER_INPUT: usize = 128;

const DTLSSRTPENC_PROPERTIES: &[PropertySpec] = &[
    PropertySpec::new(
        "connection-id",
        PropKind::Str,
        "name the paired `dtlssrtpdec` also carries, so the two drive one DTLS handshake",
    )
    .with_default(""),
    PropertySpec::new(
        "is-client",
        PropKind::Bool,
        "start the handshake instead of waiting for the peer to start it",
    )
    .with_default("false"),
    PropertySpec::new(
        "connection-state",
        PropKind::Str,
        "the handshake's progress: new | connecting | connected | failed | closed. Read only",
    )
    .with_default("new"),
    PropertySpec::new(
        "peer-pem",
        PropKind::Str,
        "the peer's certificate as PEM, empty until the handshake presented it. Read only",
    )
    .with_default(""),
    PEER_FINGERPRINT_PROPERTY,
];

/// What one `dtlssrtpenc` protected, held back, and dropped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DtlsSrtpEncStats {
    pub packets_protected: u64,
    pub packets_dropped: u64,
    pub records_sent: u64,
}

/// One input pad: the flow its caps named, the sender built from its first
/// packet, and the media it holds while the handshake runs.
#[derive(Debug, Default)]
struct EncoderInput {
    flow: Option<SrtpFlow>,
    sender: Option<SrtpSender>,
    held: VecDeque<Frame>,
}

/// Protects each input's RTP or RTCP under the key a DTLS handshake exported,
/// and carries that handshake's records on the same output.
#[derive(Debug)]
pub struct DtlsSrtpEnc {
    inputs: Vec<EncoderInput>,
    connection: Option<DtlsSrtpHandle>,
    connection_id: String,
    is_client: bool,
    expected_peer_fingerprint: Option<[u8; FINGERPRINT_LENGTH]>,
    packets_protected: u64,
    packets_dropped: u64,
    records_sent: u64,
}

impl DtlsSrtpEnc {
    pub fn new(inputs: usize) -> Self {
        Self {
            inputs: (0..inputs.max(1))
                .map(|_| EncoderInput::default())
                .collect(),
            connection: None,
            connection_id: String::new(),
            is_client: false,
            expected_peer_fingerprint: None,
            packets_protected: 0,
            packets_dropped: 0,
            records_sent: 0,
        }
    }

    /// Drive this connection instead of looking one up by `connection-id`, for
    /// an application that builds the pair itself.
    pub fn with_connection(mut self, connection: DtlsSrtpHandle) -> Self {
        self.connection = Some(connection);
        self
    }

    /// Start the handshake rather than waiting for the peer to start it.
    pub fn with_client_role(mut self, is_client: bool) -> Self {
        self.is_client = is_client;
        self
    }

    /// Refuse any peer whose certificate does not hash to this SHA-256
    /// fingerprint, the value signalling carried out of band.
    pub fn with_peer_fingerprint(mut self, fingerprint: [u8; FINGERPRINT_LENGTH]) -> Self {
        self.expected_peer_fingerprint = Some(fingerprint);
        self
    }

    pub fn stats(&self) -> DtlsSrtpEncStats {
        DtlsSrtpEncStats {
            packets_protected: self.packets_protected,
            packets_dropped: self.packets_dropped,
            records_sent: self.records_sent,
        }
    }

    /// Resolve the session and tell it which end starts the handshake, once, at
    /// configure time. An element with neither a handle nor a `connection-id`
    /// has no decoder to answer it, so that pipeline is refused here rather than
    /// running with a handshake that can never complete.
    fn open_connection(&mut self) -> Result<(), G2gError> {
        if self.connection.is_none() {
            if self.connection_id.is_empty() {
                g2g_core::g2g_error!(
                    self,
                    "no `connection-id`: the paired `dtlssrtpdec` carries the same name, and \
                     without it no peer can answer the handshake"
                );
                return Err(G2gError::NotConfigured);
            }
            self.connection = Some(connection_for_id(&self.connection_id).map_err(report(self))?);
        }
        let handle = self.connection.clone().ok_or(G2gError::NotConfigured)?;
        let mut connection = lock_connection(&handle).map_err(report(self))?;
        connection.set_client(self.is_client);
        if let Some(fingerprint) = self.expected_peer_fingerprint {
            connection.set_expected_peer_fingerprint(fingerprint);
        }
        Ok(())
    }

    fn connection(&self) -> Result<&DtlsSrtpHandle, G2gError> {
        self.connection.as_ref().ok_or(G2gError::NotConfigured)
    }

    fn connection_state(&self) -> DtlsSrtpState {
        self.connection
            .as_ref()
            .and_then(|handle| lock_connection(handle).ok())
            .map_or(DtlsSrtpState::New, |connection| connection.state())
    }

    fn peer_certificate_pem(&self) -> String {
        self.connection
            .as_ref()
            .and_then(|handle| lock_connection(handle).ok())
            .and_then(|connection| connection.peer_certificate_pem())
            .unwrap_or_default()
    }

    /// Advance the handshake and put every record it produced on the wire. Any
    /// DTLS failure ends the run: a session that cannot be keyed must not fall
    /// back to sending the media in the clear.
    async fn service_connection(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let now = Instant::now();
        let records = {
            let handle = self.connection()?.clone();
            let mut connection = lock_connection(&handle).map_err(report(self))?;
            connection.drive(now).map_err(report(self))?;
            let mut records = Vec::new();
            while let Some(record) = connection.take_outbound() {
                records.push(record);
            }
            records
        };
        for record in records {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(record.into_boxed_slice())),
                FrameTiming::default(),
                self.records_sent,
            );
            self.records_sent += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }

    /// Whether the handshake has keyed this end, asked once per packet.
    fn is_keyed(&self) -> Result<bool, G2gError> {
        let handle = self.connection()?;
        let connection = lock_connection(handle).map_err(report(self))?;
        Ok(connection.keys().is_some())
    }

    /// The key this end protects with, `None` until the handshake exported it.
    /// Read once per input, when that pad's first packet builds its sender.
    fn send_key(&self) -> Result<Option<SrtpMasterKey>, G2gError> {
        let handle = self.connection()?;
        let connection = lock_connection(handle).map_err(report(self))?;
        Ok(connection.keys().map(|keys| keys.send.clone()))
    }

    /// Send everything the inputs held while the handshake ran, in the order it
    /// was queued. Run on every packet and every tick once the session is keyed,
    /// so a stream held through the handshake is released without waiting for
    /// the next packet on its own pad.
    async fn release_held(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        for input in 0..self.inputs.len() {
            let mut pending: VecDeque<Frame> = core::mem::take(&mut self.inputs[input].held);
            while let Some(frame) = pending.pop_front() {
                self.protect(input, frame, out).await?;
            }
        }
        Ok(())
    }

    async fn protect(
        &mut self,
        input: usize,
        frame: Frame,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let flow = self.inputs[input].flow.ok_or(G2gError::NotConfigured)?;
        let bytes = frame
            .domain
            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
        if self.inputs[input].sender.is_none() {
            let Some(source) = synchronization_source(bytes, flow) else {
                self.packets_dropped += 1;
                g2g_core::g2g_warn!(self, "dropping a frame that is not a valid {flow:?} packet");
                return Ok(());
            };
            let Some(key) = self.send_key()? else {
                return Err(G2gError::NotConfigured);
            };
            let sender = SrtpSender::new(key.policy(), key.master_key(), key.master_salt(), source)
                .map_err(|error| {
                    g2g_core::g2g_error!(self, "cannot protect this stream: {error}");
                    G2gError::NotConfigured
                })?;
            self.inputs[input].sender = Some(sender);
        }
        let sender = self.inputs[input]
            .sender
            .as_mut()
            .ok_or(G2gError::NotConfigured)?;
        match protect_for_flow(sender, flow, bytes, RTCP_PROTECTION) {
            Ok(protected) => {
                self.packets_protected += 1;
                push_frame_bytes(frame, protected, out).await
            }
            Err(error) if is_fatal_send_error(error) => {
                self.packets_dropped += 1;
                g2g_core::g2g_error!(self, "stopping the stream: {error}");
                Err(G2gError::Hardware(g2g_core::HardwareError::Other))
            }
            Err(error) => {
                self.packets_dropped += 1;
                g2g_core::g2g_warn!(self, "dropping a packet: {error}");
                Ok(())
            }
        }
    }

    /// Hold a frame until the handshake keys this end, dropping the oldest once
    /// the queue is full so a peer that never answers cannot grow it.
    fn hold(&mut self, input: usize, frame: Frame) {
        if self.inputs[input].held.len() >= HELD_PACKETS_PER_INPUT {
            self.inputs[input].held.pop_front();
            self.packets_dropped += 1;
            g2g_core::g2g_warn!(
                self,
                "dropping the oldest of {HELD_PACKETS_PER_INPUT} packets held on input {input}: \
                 the handshake has not keyed this stream yet"
            );
        }
        self.inputs[input].held.push_back(frame);
    }

    async fn process_packet(
        &mut self,
        input: usize,
        packet: PipelinePacket,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        self.service_connection(out).await?;
        let keyed = self.is_keyed()?;
        if keyed {
            self.release_held(out).await?;
        }
        match packet {
            PipelinePacket::DataFrame(frame) => {
                if keyed {
                    self.protect(input, frame, out).await
                } else {
                    self.hold(input, frame);
                    Ok(())
                }
            }
            // The fan-in arm forwards caps and Eos itself, the tick only serviced the connection.
            PipelinePacket::CapsChanged(_) | PipelinePacket::Eos | PipelinePacket::Tick => Ok(()),
            other => out.push(other).await.map(|_| ()),
        }
    }
}

/// Log a connection failure on this element before it stops the run.
fn report(element: &DtlsSrtpEnc) -> impl Fn(DtlsSrtpError) -> G2gError + '_ {
    move |error| {
        g2g_core::g2g_error!(element, "{error}");
        G2gError::from(error)
    }
}

impl LogSource for DtlsSrtpEnc {
    fn log_category(&self) -> &'static str {
        "dtlssrtpenc"
    }
}

impl MultiInputElement for DtlsSrtpEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Serviced on a period of its own, so the handshake advances while the
    /// inputs are silent and a held stream is released as soon as it is keyed.
    fn tick_interval_ns(&self) -> Option<u64> {
        Some(CONNECTION_TICK_NANOSECONDS)
    }

    /// Both pad names are positional: `rtp_0` and `rtcp_0` carry no kind the
    /// parser recognizes, and the flow comes from each pad's caps instead.
    fn input_pad_index(&self, req: &PadRequest, ordinal: usize) -> Option<usize> {
        match req.kind {
            PadKind::Any => Some(req.index),
            _ => {
                let _ = ordinal;
                None
            }
        }
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        SrtpFlow::from_plain_caps(upstream_caps).ok_or(G2gError::CapsMismatch)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::from_alternatives(flow_caps(
            SrtpFlow::plain_encoding,
        )))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let flow = SrtpFlow::from_plain_caps(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.open_connection()?;
        self.inputs
            .get_mut(input)
            .ok_or(G2gError::CapsMismatch)?
            .flow = Some(flow);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(Caps::ByteStream {
            encoding: ByteStreamEncoding::Dtls,
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "DTLS-SRTP protector",
            "Filter/Network/SRTP",
            "Runs a DTLS-SRTP handshake and protects the RTP and RTCP it keys",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DTLSSRTPENC_PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "connection-id" => {
                let id = value.as_str().ok_or(PropError::Type)?;
                if self.connection.is_some() {
                    return Err(PropError::Value);
                }
                self.connection_id = id.to_string();
                Ok(())
            }
            "is-client" => {
                self.is_client = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "peer-fingerprint" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                self.expected_peer_fingerprint = match text.is_empty() {
                    true => None,
                    false => Some(parse_fingerprint(text).ok_or(PropError::Value)?),
                };
                Ok(())
            }
            "connection-state" | "peer-pem" => Err(PropError::Value),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "connection-id" => Some(PropValue::Str(self.connection_id.clone())),
            "is-client" => Some(PropValue::Bool(self.is_client)),
            "connection-state" => {
                Some(PropValue::Str(self.connection_state().as_str().to_string()))
            }
            "peer-pem" => Some(PropValue::Str(self.peer_certificate_pem())),
            "peer-fingerprint" => Some(PropValue::Str(
                self.expected_peer_fingerprint
                    .map(|fingerprint| format_fingerprint(&fingerprint))
                    .unwrap_or_default(),
            )),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(self.process_packet(input, packet, out))
    }
}

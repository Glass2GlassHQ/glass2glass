//! `dtlssrtpdec`: answers the DTLS-SRTP handshake and recovers the media it keys.
//!
//! One `application/x-dtls` input carrying handshake records multiplexed with
//! protected media, split on the RFC 7983 first byte, and two outputs: recovered
//! RTP on port 0 and RTCP on port 1, told apart by the RFC 5761 payload-type
//! rule.
//!
//! This element answers the handshake but never sends: it has no socket path.
//! The records it produces leave through the `dtlssrtpenc` on the same
//! `connection-id`, so a decoder without its paired encoder in the same process
//! never completes a handshake. GStreamer's pair has the same shape.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::time::Instant;

use g2g_core::frame::Frame;
use g2g_core::log::LogSource;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, MemoryDomain,
    MultiOutputElement, MultiOutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::dtlssrtp::{
    classify_datagram, connection_for_id, format_fingerprint, lock_connection, parse_fingerprint,
    DtlsSrtpDatagram, DtlsSrtpHandle, DtlsSrtpState, FINGERPRINT_LENGTH, PEER_FINGERPRINT_PROPERTY,
};
use crate::rtcp::is_rtcp;
use crate::srtp::{
    SrtpFlow, SrtpKeyProvider, SrtpKeyingMaterial, SrtpReceiverSet, SrtpStreamStats,
};

/// Recovered RTP leaves on this port, RTCP on the next one.
pub const RTP_PORT: usize = 0;
pub const RTCP_PORT: usize = 1;
/// Both ports together, and the most a graph can link.
pub const PORT_COUNT: usize = 2;

const DTLSSRTPDEC_PROPERTIES: &[PropertySpec] = &[
    PropertySpec::new(
        "connection-id",
        PropKind::Str,
        "name the paired `dtlssrtpenc` also carries, so the two drive one DTLS handshake",
    )
    .with_default(""),
    PropertySpec::new(
        "pem",
        PropKind::Str,
        "this end's certificate and private key as PEM. Unset generates a self-signed ECDSA \
         certificate. Reads back the certificate only",
    )
    .with_default(""),
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

/// What one `dtlssrtpdec` took in, recovered, and dropped.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DtlsSrtpDecStats {
    pub packets_received: u64,
    pub packets_recovered: u64,
    pub packets_dropped: u64,
    pub records_handled: u64,
    pub streams: Vec<SrtpStreamStats>,
}

/// Hands every synchronization source the key the handshake exported for the
/// peer's direction. Empty until the handshake keyed the session, so a packet
/// that arrives first is dropped and the source is keyed on the next one.
#[derive(Debug, Default)]
struct DtlsSrtpKeys {
    connection: Option<DtlsSrtpHandle>,
}

impl SrtpKeyProvider for DtlsSrtpKeys {
    fn keys_for(&mut self, _synchronization_source: u32) -> Vec<SrtpKeyingMaterial> {
        self.connection
            .as_ref()
            .and_then(|handle| lock_connection(handle).ok())
            .map(|connection| connection.receive_keying_material())
            .unwrap_or_default()
    }
}

/// Answers the DTLS handshake and recovers the RTP and RTCP it keys.
#[derive(Debug)]
pub struct DtlsSrtpDec {
    receivers: SrtpReceiverSet<DtlsSrtpKeys>,
    connection_id: String,
    certificate_pem: String,
    expected_peer_fingerprint: Option<[u8; FINGERPRINT_LENGTH]>,
    announced: [bool; PORT_COUNT],
    ports: usize,
    packets_received: u64,
    packets_recovered: u64,
    packets_dropped: u64,
    records_handled: u64,
}

impl Default for DtlsSrtpDec {
    fn default() -> Self {
        Self::new(PORT_COUNT)
    }
}

impl DtlsSrtpDec {
    /// `outputs` is what the launch parser counted. A line that links only one
    /// branch gets one port, and the flow the other port would have carried is
    /// dropped rather than pushed at a port the graph does not have.
    pub fn new(outputs: usize) -> Self {
        Self {
            receivers: SrtpReceiverSet::new(DtlsSrtpKeys::default()),
            connection_id: String::new(),
            certificate_pem: String::new(),
            expected_peer_fingerprint: None,
            announced: [false; PORT_COUNT],
            ports: outputs.clamp(1, PORT_COUNT),
            packets_received: 0,
            packets_recovered: 0,
            packets_dropped: 0,
            records_handled: 0,
        }
    }

    /// Answer this connection instead of looking one up by `connection-id`.
    pub fn with_connection(mut self, connection: DtlsSrtpHandle) -> Self {
        self.receivers.key_provider_mut().connection = Some(connection);
        self
    }

    /// Use this certificate and private key rather than generating one.
    pub fn with_certificate_pem(mut self, pem: &str) -> Self {
        self.certificate_pem = pem.to_string();
        self
    }

    /// Refuse any peer whose certificate does not hash to this SHA-256
    /// fingerprint, the value signalling carried out of band.
    pub fn with_peer_fingerprint(mut self, fingerprint: [u8; FINGERPRINT_LENGTH]) -> Self {
        self.expected_peer_fingerprint = Some(fingerprint);
        self
    }

    pub fn stats(&self) -> DtlsSrtpDecStats {
        DtlsSrtpDecStats {
            packets_received: self.packets_received,
            packets_recovered: self.packets_recovered,
            packets_dropped: self.packets_dropped,
            records_handled: self.records_handled,
            streams: self.receivers.stream_statistics(),
        }
    }

    fn connection(&self) -> Option<&DtlsSrtpHandle> {
        self.receivers.key_provider().connection.as_ref()
    }

    fn connection_state(&self) -> DtlsSrtpState {
        self.connection()
            .and_then(|handle| lock_connection(handle).ok())
            .map_or(DtlsSrtpState::New, |connection| connection.state())
    }

    fn peer_certificate_pem(&self) -> String {
        self.connection()
            .and_then(|handle| lock_connection(handle).ok())
            .and_then(|connection| connection.peer_certificate_pem())
            .unwrap_or_default()
    }

    /// Resolve the session and hand it the certificate, once, at configure time.
    fn open_connection(&mut self) -> Result<(), G2gError> {
        if self.connection().is_none() {
            if self.connection_id.is_empty() {
                g2g_core::g2g_error!(
                    self,
                    "no `connection-id`: the paired `dtlssrtpenc` carries the same name, and \
                     without it this element has no way to send its handshake records"
                );
                return Err(G2gError::NotConfigured);
            }
            let handle = connection_for_id(&self.connection_id).map_err(|error| {
                g2g_core::g2g_error!(self, "{error}");
                G2gError::from(error)
            })?;
            self.receivers.key_provider_mut().connection = Some(handle);
        }
        let handle = self.connection().ok_or(G2gError::NotConfigured)?.clone();
        let mut connection = lock_connection(&handle).map_err(|error| {
            g2g_core::g2g_error!(self, "{error}");
            G2gError::from(error)
        })?;
        if let Some(fingerprint) = self.expected_peer_fingerprint {
            connection.set_expected_peer_fingerprint(fingerprint);
        }
        if self.certificate_pem.is_empty() {
            return Ok(());
        }
        connection
            .with_certificate_pem(&self.certificate_pem)
            .map_err(|error| {
                g2g_core::g2g_error!(self, "{error}");
                G2gError::from(error)
            })
    }

    /// Take one handshake record in. Any DTLS failure ends the run: a session
    /// that cannot be keyed must not go on accepting media.
    fn handle_record(&mut self, record: &[u8]) -> Result<(), G2gError> {
        let handle = self.connection().ok_or(G2gError::NotConfigured)?.clone();
        self.records_handled += 1;
        let mut connection = lock_connection(&handle).map_err(|error| {
            g2g_core::g2g_error!(self, "{error}");
            G2gError::from(error)
        })?;
        connection
            .handle_datagram(record, Instant::now())
            .map_err(|error| {
                g2g_core::g2g_error!(self, "{error}");
                G2gError::from(error)
            })
    }

    async fn deliver(
        &mut self,
        flow: SrtpFlow,
        frame: Frame,
        recovered: Vec<u8>,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        let port = match flow {
            SrtpFlow::Rtp => RTP_PORT,
            SrtpFlow::Rtcp => RTCP_PORT,
        };
        if port >= self.ports {
            self.packets_dropped += 1;
            g2g_core::g2g_warn!(
                self,
                "dropping a {flow:?} packet: nothing is linked to port {port}"
            );
            return Ok(());
        }
        if !self.announced[port] {
            self.announced[port] = true;
            out.push_to(port, PipelinePacket::CapsChanged(port_caps(port)))
                .await?;
        }
        let mut frame = frame;
        frame.domain = MemoryDomain::System(SystemSlice::from_boxed(recovered.into_boxed_slice()));
        out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
        self.packets_recovered += 1;
        Ok(())
    }

    async fn process_frame(
        &mut self,
        frame: Frame,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        let datagram = frame
            .domain
            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
        self.packets_received += 1;
        match classify_datagram(datagram) {
            Some(DtlsSrtpDatagram::Dtls) => {
                let record = datagram.to_vec();
                self.handle_record(&record)
            }
            Some(DtlsSrtpDatagram::Media) => {
                let flow = if is_rtcp(datagram) {
                    SrtpFlow::Rtcp
                } else {
                    SrtpFlow::Rtp
                };
                let recovered = match flow {
                    SrtpFlow::Rtp => self.receivers.unprotect_rtp(datagram),
                    SrtpFlow::Rtcp => self.receivers.unprotect_rtcp(datagram),
                };
                match recovered {
                    Ok(recovered) => self.deliver(flow, frame, recovered, out).await,
                    Err(error) => {
                        self.packets_dropped += 1;
                        g2g_core::g2g_warn!(self, "dropping a {flow:?} packet: {error}");
                        Ok(())
                    }
                }
            }
            None => {
                self.packets_dropped += 1;
                g2g_core::g2g_warn!(
                    self,
                    "dropping a datagram that is neither DTLS nor protected media"
                );
                Ok(())
            }
        }
    }

    async fn process_packet(
        &mut self,
        packet: PipelinePacket,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        match packet {
            PipelinePacket::DataFrame(frame) => self.process_frame(frame, out).await,
            PipelinePacket::Segment(segment) => {
                for port in 0..self.ports {
                    out.push_to(port, PipelinePacket::Segment(segment)).await?;
                }
                Ok(())
            }
            PipelinePacket::Flush => {
                self.announced = [false; PORT_COUNT];
                for port in 0..self.ports {
                    out.push_to(port, PipelinePacket::Flush).await?;
                }
                Ok(())
            }
            // The runner's fan-out arm owns the port caps and the merged Eos.
            _ => Ok(()),
        }
    }
}

fn port_caps(port: usize) -> Caps {
    Caps::ByteStream {
        encoding: if port == RTCP_PORT {
            ByteStreamEncoding::Rtcp
        } else {
            ByteStreamEncoding::Rtp
        },
    }
}

fn input_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Dtls,
    }
}

impl LogSource for DtlsSrtpDec {
    fn log_category(&self) -> &'static str {
        "dtlssrtpdec"
    }
}

impl MultiOutputElement for DtlsSrtpDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&input_caps())
    }

    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(input_caps()))
    }

    /// Each port carries its own protocol, so a branch negotiates against the
    /// one it will receive instead of against the multiplexed input.
    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        (port < self.ports).then(|| port_caps(port))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        absolute_caps.intersect(&input_caps())?;
        self.open_connection()?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DTLSSRTPDEC_PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "connection-id" => {
                let id = value.as_str().ok_or(PropError::Type)?;
                if self.connection().is_some() {
                    return Err(PropError::Value);
                }
                self.connection_id = id.to_string();
                Ok(())
            }
            "pem" => {
                self.certificate_pem = value.as_str().ok_or(PropError::Type)?.to_string();
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
            // Never hands the private key back out: only the certificate half.
            "pem" => Some(PropValue::Str(
                self.connection()
                    .and_then(|handle| lock_connection(handle).ok())
                    .and_then(|mut connection| connection.certificate_pem().ok())
                    .unwrap_or_default(),
            )),
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
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(self.process_packet(packet, out))
    }
}

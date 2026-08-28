//! DTLS-SRTP key delivery (RFC 5764): the DTLS session the `dtlssrtpenc` /
//! `dtlssrtpdec` pair share.
//!
//! One [`DtlsSrtpConnection`] carries one handshake. The decoder feeds it the
//! DTLS records it demultiplexes out of the media socket, the encoder drains the
//! records it wants to send and drives its retransmission timer, and once the
//! handshake exports keying material both read the SRTP master keys from it.
//! The two elements find each other through [`connection_for_id`], the analog of
//! GStreamer's `connection-id`.
//!
//! The DTLS state machine itself is `dimpl`, which owns no socket: this module
//! is the glue between it, the RFC 5764 key split, and the RFC 7983 byte that
//! says whether a datagram is DTLS or protected media.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::Instant;

use base64::Engine as _;
use dimpl::certificate::calculate_fingerprint;
use dimpl::{Config, Dtls, DtlsCertificate, Output};
use g2g_core::G2gError;
use zeroize::Zeroizing;

use crate::srtp::{
    SrtpAuthentication, SrtpCipher, SrtpError, SrtpKeyingMaterial, SrtpMasterKey, SrtpPolicy,
};

/// Largest DTLS record this session builds, matching `udpsink`'s `max-payload`
/// default: a handshake flight that exceeds it is fragmented rather than sent as
/// a datagram the sink would have to split.
pub const DTLS_MAXIMUM_RECORD: usize = 1400;

/// Rollover counter every context keyed by a fresh handshake starts from.
const INITIAL_ROLLOVER_COUNTER: u32 = 0;

/// SHA-256 fingerprint length, the hash RFC 8122 pins for DTLS-SRTP.
pub const FINGERPRINT_LENGTH: usize = 32;
/// The hash function name an SDP `a=fingerprint` line carries before the digest.
pub const FINGERPRINT_HASH_NAME: &str = "sha-256";

/// The pinned-peer property both DTLS-SRTP elements declare, since either one
/// may be the half a launch line configures.
pub const PEER_FINGERPRINT_PROPERTY: g2g_core::PropertySpec = g2g_core::PropertySpec::new(
    "peer-fingerprint",
    g2g_core::PropKind::Str,
    "refuse any peer whose certificate does not hash to this SHA-256 fingerprint, in the SDP \
     `a=fingerprint` value form `sha-256 AB:CD:...`. Empty accepts whatever certificate the \
     handshake presents",
)
.with_default("");

const CERTIFICATE_PEM_LABEL: &str = "CERTIFICATE";
/// The two labels an EC private key arrives under: PKCS#8 and SEC1. `dimpl`
/// reads either DER encoding, so only the base64 body matters here.
const PRIVATE_KEY_PEM_LABELS: [&str; 2] = ["PRIVATE KEY", "EC PRIVATE KEY"];
/// PEM bodies wrap at 64 characters (RFC 7468).
const PEM_LINE_LENGTH: usize = 64;

/// First-byte ranges RFC 7983 assigns on a port that multiplexes DTLS with
/// protected media. Anything outside both is neither, and is dropped.
const DTLS_FIRST_BYTES: core::ops::RangeInclusive<u8> = 20..=63;
const MEDIA_FIRST_BYTES: core::ops::RangeInclusive<u8> = 128..=191;

/// What a datagram arriving on a DTLS-SRTP socket carries, by its first byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtlsSrtpDatagram {
    Dtls,
    Media,
}

/// Which of the two protocols RFC 7983 says `datagram` carries, `None` for a
/// first byte that belongs to neither (STUN, TURN, ZRTP).
pub fn classify_datagram(datagram: &[u8]) -> Option<DtlsSrtpDatagram> {
    let first = *datagram.first()?;
    if DTLS_FIRST_BYTES.contains(&first) {
        return Some(DtlsSrtpDatagram::Dtls);
    }
    if MEDIA_FIRST_BYTES.contains(&first) {
        return Some(DtlsSrtpDatagram::Media);
    }
    None
}

/// The handshake's progress, named the way gst's `connection-state` enum is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtlsSrtpState {
    New,
    Connecting,
    Connected,
    Failed,
    Closed,
}

impl DtlsSrtpState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::New => "new",
            Self::Connecting => "connecting",
            Self::Connected => "connected",
            Self::Failed => "failed",
            Self::Closed => "closed",
        }
    }
}

/// A failure that ends the DTLS session. Every one of these is fatal: the
/// connection is closed and both elements stop the run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DtlsSrtpError {
    /// The peer's certificate does not hash to the fingerprint the application
    /// was told to expect, so the peer is not who signalling said it is.
    FingerprintMismatch,
    /// The handshake settled on a protection profile the packet layer does not
    /// implement.
    UnsupportedProfile(&'static str),
    /// The exported block is not the length the negotiated profile calls for.
    ShortKeyingMaterial { expected: usize, actual: usize },
    /// The `pem` property does not hold a certificate and a private key.
    InvalidCertificatePem,
    /// No certificate could be generated for this session.
    CertificateGenerationFailed,
    /// The DTLS state machine reported an error, which is always terminal.
    Dtls(String),
    /// The SRTP packet layer refused the exported key material.
    Srtp(SrtpError),
    /// A thread panicked while holding the connection, so its state is unknown.
    Poisoned,
}

impl fmt::Display for DtlsSrtpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FingerprintMismatch => {
                formatter.write_str("the peer certificate does not match the expected fingerprint")
            }
            Self::UnsupportedProfile(profile) => write!(
                formatter,
                "the handshake negotiated {profile}, which the SRTP packet layer does not \
                 implement"
            ),
            Self::ShortKeyingMaterial { expected, actual } => write!(
                formatter,
                "the DTLS-SRTP export is {actual} bytes, expected {expected}"
            ),
            Self::InvalidCertificatePem => formatter
                .write_str("the `pem` property needs a CERTIFICATE and a private key block"),
            Self::CertificateGenerationFailed => {
                formatter.write_str("cannot generate a self-signed certificate")
            }
            Self::Dtls(message) => write!(formatter, "the DTLS session failed: {message}"),
            Self::Srtp(error) => {
                write!(formatter, "the exported key material is unusable: {error}")
            }
            Self::Poisoned => formatter.write_str("the DTLS connection state is unrecoverable"),
        }
    }
}

impl std::error::Error for DtlsSrtpError {}

impl From<DtlsSrtpError> for G2gError {
    /// A DTLS-SRTP failure stops the run, the way `srtpenc` stops it when the
    /// SRTP key lifetime ends: there is no correct way to keep sending.
    fn from(_: DtlsSrtpError) -> Self {
        G2gError::Hardware(g2g_core::HardwareError::Other)
    }
}

/// The two SRTP master keys one handshake produces: the one this end
/// protects with and the one it recovers the peer's packets with.
#[derive(Clone, Debug)]
pub struct DtlsSrtpKeys {
    pub send: SrtpMasterKey,
    pub receive: SrtpMasterKey,
}

/// The DTLS session `dtlssrtpenc` and `dtlssrtpdec` share, held behind a
/// [`DtlsSrtpHandle`] so both elements drive the one handshake.
pub struct DtlsSrtpConnection {
    dtls: Option<Dtls>,
    certificate: Option<DtlsCertificate>,
    is_client: bool,
    expected_peer_fingerprint: Option<[u8; FINGERPRINT_LENGTH]>,
    peer_certificate: Option<Vec<u8>>,
    state: DtlsSrtpState,
    failure: Option<DtlsSrtpError>,
    keys: Option<DtlsSrtpKeys>,
    outbound: VecDeque<Vec<u8>>,
    next_timeout: Option<Instant>,
}

impl fmt::Debug for DtlsSrtpConnection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DtlsSrtpConnection")
            .field("is_client", &self.is_client)
            .field("state", &self.state)
            .field("keyed", &self.keys.is_some())
            .field("pending_records", &self.outbound.len())
            .finish_non_exhaustive()
    }
}

impl Default for DtlsSrtpConnection {
    fn default() -> Self {
        Self::new()
    }
}

impl DtlsSrtpConnection {
    pub fn new() -> Self {
        Self {
            dtls: None,
            certificate: None,
            is_client: false,
            expected_peer_fingerprint: None,
            peer_certificate: None,
            state: DtlsSrtpState::New,
            failure: None,
            keys: None,
            outbound: VecDeque::new(),
            next_timeout: None,
        }
    }

    /// A handle a caller can hand both elements directly, instead of pairing
    /// them through a `connection-id`.
    pub fn shared() -> DtlsSrtpHandle {
        Arc::new(Mutex::new(Self::new()))
    }

    /// Use this certificate and private key instead of generating one. `pem`
    /// holds a `CERTIFICATE` block and a `PRIVATE KEY` or `EC PRIVATE KEY` one.
    /// Refused once the handshake has started.
    pub fn with_certificate_pem(&mut self, pem: &str) -> Result<(), DtlsSrtpError> {
        let certificate = parse_certificate_pem(pem)?;
        self.certificate = Some(certificate);
        Ok(())
    }

    /// Whether this end starts the handshake. Refused once it has started.
    pub fn set_client(&mut self, is_client: bool) {
        self.is_client = is_client;
    }

    pub fn is_client(&self) -> bool {
        self.is_client
    }

    /// Refuse any peer whose certificate does not hash to this SHA-256
    /// fingerprint, the value signalling carried out of band. Without one every
    /// peer certificate is accepted, which is the DTLS-SRTP default: the
    /// handshake proves possession of a key, and only the fingerprint ties that
    /// key to the peer the application meant to reach.
    pub fn set_expected_peer_fingerprint(&mut self, fingerprint: [u8; FINGERPRINT_LENGTH]) {
        self.expected_peer_fingerprint = Some(fingerprint);
    }

    /// The fingerprint this end refuses any other peer for, `None` when it
    /// accepts whatever certificate the handshake presents.
    pub fn expected_peer_fingerprint(&self) -> Option<[u8; FINGERPRINT_LENGTH]> {
        self.expected_peer_fingerprint
    }

    /// The protection profile the handshake settled on, `None` until it did.
    pub fn policy(&self) -> Option<SrtpPolicy> {
        Some(self.keys.as_ref()?.send.policy())
    }

    /// This end's certificate as PEM, generating one on first use.
    pub fn certificate_pem(&mut self) -> Result<String, DtlsSrtpError> {
        let certificate = self.certificate()?;
        Ok(encode_pem(CERTIFICATE_PEM_LABEL, &certificate.certificate))
    }

    /// This end's SHA-256 certificate fingerprint, generating the certificate on
    /// first use. This is the value the peer is told to expect.
    pub fn local_fingerprint(&mut self) -> Result<Vec<u8>, DtlsSrtpError> {
        let certificate = self.certificate()?;
        Ok(calculate_fingerprint(&certificate.certificate))
    }

    /// The peer's certificate as PEM, `None` until the handshake presented it.
    pub fn peer_certificate_pem(&self) -> Option<String> {
        self.peer_certificate
            .as_deref()
            .map(|der| encode_pem(CERTIFICATE_PEM_LABEL, der))
    }

    pub fn state(&self) -> DtlsSrtpState {
        self.state
    }

    /// Why the session failed, `None` while it has not. An application that
    /// signalled a fingerprint reads this to tell a refused peer from a peer
    /// whose profile this stack cannot key.
    pub fn failure(&self) -> Option<&DtlsSrtpError> {
        self.failure.as_ref()
    }

    pub fn keys(&self) -> Option<&DtlsSrtpKeys> {
        self.keys.as_ref()
    }

    /// The key material every inbound synchronization source is decrypted with,
    /// empty until the handshake exported it.
    pub fn receive_keying_material(&self) -> Vec<SrtpKeyingMaterial> {
        self.keys
            .as_ref()
            .and_then(|keys| keys.receive.keying_material(INITIAL_ROLLOVER_COUNTER).ok())
            .into_iter()
            .collect()
    }

    /// Take in one DTLS record from the socket and advance the handshake.
    pub fn handle_datagram(&mut self, datagram: &[u8], now: Instant) -> Result<(), DtlsSrtpError> {
        self.start(now)?;
        let result = self
            .dtls
            .as_mut()
            .ok_or(DtlsSrtpError::CertificateGenerationFailed)?
            .handle_packet(datagram);
        if let Err(error) = result {
            return Err(self.fail(DtlsSrtpError::Dtls(error.to_string())));
        }
        self.pump()
    }

    /// Run the retransmission timer if it is due, and refresh what is pending.
    pub fn drive(&mut self, now: Instant) -> Result<(), DtlsSrtpError> {
        self.start(now)?;
        if self.next_timeout.is_some_and(|deadline| now >= deadline) {
            self.next_timeout = None;
            let result = self
                .dtls
                .as_mut()
                .ok_or(DtlsSrtpError::CertificateGenerationFailed)?
                .handle_timeout(now);
            if let Err(error) = result {
                return Err(self.fail(DtlsSrtpError::Dtls(error.to_string())));
            }
        }
        self.pump()
    }

    /// The next DTLS record to put on the wire, in the order they were produced.
    pub fn take_outbound(&mut self) -> Option<Vec<u8>> {
        self.outbound.pop_front()
    }

    /// Build the DTLS state machine on first use, so the certificate and role
    /// properties can be set after the connection was created.
    fn start(&mut self, now: Instant) -> Result<(), DtlsSrtpError> {
        if self.dtls.is_some() {
            return Ok(());
        }
        let certificate = self.certificate()?.clone();
        let config = Config::builder()
            .mtu(DTLS_MAXIMUM_RECORD)
            .build()
            .map_err(|error| DtlsSrtpError::Dtls(error.to_string()))?;
        // DTLS 1.2 only: OpenSSL-backed peers may lack the 1.3 version negotiation.
        let mut dtls = Dtls::new_12(Arc::new(config), certificate, now);
        dtls.set_active(self.is_client);
        self.dtls = Some(dtls);
        self.state = DtlsSrtpState::Connecting;
        Ok(())
    }

    fn certificate(&mut self) -> Result<&DtlsCertificate, DtlsSrtpError> {
        if self.certificate.is_none() {
            let generated = dimpl::certificate::generate_self_signed_certificate()
                .map_err(|_| DtlsSrtpError::CertificateGenerationFailed)?;
            self.certificate = Some(generated);
        }
        self.certificate
            .as_ref()
            .ok_or(DtlsSrtpError::CertificateGenerationFailed)
    }

    /// Drain every event the state machine has, queueing the records to send and
    /// absorbing the handshake results.
    fn pump(&mut self) -> Result<(), DtlsSrtpError> {
        let mut buffer = alloc::vec![0_u8; DTLS_MAXIMUM_RECORD];
        loop {
            let Some(dtls) = self.dtls.as_mut() else {
                return Ok(());
            };
            let mut grow_to = None;
            let mut peer_certificate = None;
            let mut exported = None;
            let mut done = false;
            match dtls.poll_output(&mut buffer) {
                Output::Packet(record) => self.outbound.push_back(record.to_vec()),
                Output::BufferTooSmall { needed } => grow_to = Some(needed),
                Output::Timeout(deadline) => {
                    self.next_timeout = Some(deadline);
                    done = true;
                }
                Output::Connected => {
                    if self.state == DtlsSrtpState::Connecting {
                        self.state = DtlsSrtpState::Connected;
                    }
                }
                Output::PeerCert(der) => peer_certificate = Some(der.to_vec()),
                Output::KeyingMaterial(material, profile) => {
                    exported = Some((Zeroizing::new(material.to_vec()), profile));
                }
                Output::ApplicationData(_) => {}
                Output::CloseNotify => {
                    self.state = DtlsSrtpState::Closed;
                    done = true;
                }
                // `Output` is not exhaustive: an event this build does not know
                // carries nothing this session acts on.
                _ => {}
            }
            if let Some(needed) = grow_to {
                buffer.resize(needed, 0);
                continue;
            }
            if let Some(der) = peer_certificate {
                self.accept_peer_certificate(der)?;
            }
            if let Some((material, profile)) = exported {
                self.install_keys(&material, profile)?;
            }
            if done {
                return Ok(());
            }
        }
    }

    fn accept_peer_certificate(&mut self, der: Vec<u8>) -> Result<(), DtlsSrtpError> {
        if let Some(expected) = self.expected_peer_fingerprint {
            let actual = calculate_fingerprint(&der);
            if actual.as_slice() != expected.as_slice() {
                return Err(self.fail(DtlsSrtpError::FingerprintMismatch));
            }
        }
        self.peer_certificate = Some(der);
        Ok(())
    }

    /// Split the RFC 5764 export and build the two master keys. The block is
    /// `client_key || server_key || client_salt || server_salt`, and each end
    /// protects with its own half and recovers the peer's packets with the other.
    fn install_keys(
        &mut self,
        material: &[u8],
        profile: dimpl::SrtpProfile,
    ) -> Result<(), DtlsSrtpError> {
        let policy = match srtp_policy(profile) {
            Ok(policy) => policy,
            Err(error) => return Err(self.fail(error)),
        };
        let key_length = policy.cipher.master_key_lengths()[0];
        let salt_length = policy.cipher.master_salt_length();
        let expected = 2 * (key_length + salt_length);
        if material.len() != expected {
            return Err(self.fail(DtlsSrtpError::ShortKeyingMaterial {
                expected,
                actual: material.len(),
            }));
        }
        let (keys, salts) = material.split_at(2 * key_length);
        let (client_key, server_key) = keys.split_at(key_length);
        let (client_salt, server_salt) = salts.split_at(salt_length);
        let (send_key, send_salt, receive_key, receive_salt) = if self.is_client {
            (client_key, client_salt, server_key, server_salt)
        } else {
            (server_key, server_salt, client_key, client_salt)
        };
        let built = SrtpMasterKey::new(policy, send_key, send_salt).and_then(|send| {
            Ok(DtlsSrtpKeys {
                send,
                receive: SrtpMasterKey::new(policy, receive_key, receive_salt)?,
            })
        });
        match built {
            Ok(keys) => {
                self.keys = Some(keys);
                Ok(())
            }
            Err(error) => Err(self.fail(DtlsSrtpError::Srtp(error))),
        }
    }

    /// Tear the session down and remember why. Nothing is sent or recovered on a
    /// failed connection, and the keys it may already hold are dropped.
    fn fail(&mut self, error: DtlsSrtpError) -> DtlsSrtpError {
        if let Some(dtls) = self.dtls.as_mut() {
            let _ = dtls.close();
        }
        self.state = DtlsSrtpState::Failed;
        self.failure = Some(error.clone());
        self.keys = None;
        self.outbound.clear();
        error
    }
}

/// The shared session two paired elements drive.
pub type DtlsSrtpHandle = Arc<Mutex<DtlsSrtpConnection>>;

/// Borrow the connection, reporting a lock another thread panicked under rather
/// than panicking again on a session whose state is unknown.
pub fn lock_connection(
    handle: &DtlsSrtpHandle,
) -> Result<MutexGuard<'_, DtlsSrtpConnection>, DtlsSrtpError> {
    handle.lock().map_err(|_| DtlsSrtpError::Poisoned)
}

/// Connections a `connection-id` names, kept weakly so a pair that both dropped
/// leaves nothing behind.
fn connections() -> &'static Mutex<BTreeMap<String, Weak<Mutex<DtlsSrtpConnection>>>> {
    static CONNECTIONS: OnceLock<Mutex<BTreeMap<String, Weak<Mutex<DtlsSrtpConnection>>>>> =
        OnceLock::new();
    CONNECTIONS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// The connection this `connection-id` names, created on first use. The encoder
/// and the decoder of one session name the same id and so share one handshake,
/// the way GStreamer's `dtlssrtpenc` / `dtlssrtpdec` pair up.
pub fn connection_for_id(id: &str) -> Result<DtlsSrtpHandle, DtlsSrtpError> {
    let mut registry = connections().lock().map_err(|_| DtlsSrtpError::Poisoned)?;
    if let Some(existing) = registry.get(id).and_then(Weak::upgrade) {
        return Ok(existing);
    }
    let handle = DtlsSrtpConnection::shared();
    registry.insert(id.to_string(), Arc::downgrade(&handle));
    registry.retain(|_, weak| weak.strong_count() > 0);
    Ok(handle)
}

/// The packet-layer policy a negotiated DTLS-SRTP profile names.
fn srtp_policy(profile: dimpl::SrtpProfile) -> Result<SrtpPolicy, DtlsSrtpError> {
    let (cipher, authentication) = match profile {
        dimpl::SrtpProfile::AEAD_AES_128_GCM => (SrtpCipher::Aes128Gcm, SrtpAuthentication::Null),
        dimpl::SrtpProfile::AEAD_AES_256_GCM => (SrtpCipher::Aes256Gcm, SrtpAuthentication::Null),
        dimpl::SrtpProfile::AES128_CM_SHA1_80 => (
            SrtpCipher::Aes128CounterMode,
            SrtpAuthentication::HmacSha1Tag80,
        ),
        _ => {
            return Err(DtlsSrtpError::UnsupportedProfile(
                "an unknown SRTP protection profile",
            ))
        }
    };
    SrtpPolicy::new(cipher, authentication)
        .map_err(|_| DtlsSrtpError::UnsupportedProfile("an unusable SRTP protection profile"))
}

/// The SHA-256 fingerprint an SDP `a=fingerprint` value spells: an optional
/// `sha-256` prefix then 32 colon-separated hexadecimal octets, either case.
pub fn parse_fingerprint(text: &str) -> Option<[u8; FINGERPRINT_LENGTH]> {
    let digits = match text.split_once(char::is_whitespace) {
        Some((name, digits)) if name.eq_ignore_ascii_case(FINGERPRINT_HASH_NAME) => digits,
        Some(_) => return None,
        None => text,
    };
    let mut fingerprint = [0_u8; FINGERPRINT_LENGTH];
    let mut octets = digits.trim().split(':');
    for byte in &mut fingerprint {
        *byte = u8::from_str_radix(octets.next()?, 16).ok()?;
    }
    octets.next().is_none().then_some(fingerprint)
}

/// A fingerprint in the SDP `a=fingerprint` value form, the way it is signalled.
pub fn format_fingerprint(fingerprint: &[u8]) -> String {
    use core::fmt::Write as _;

    let mut text = String::from(FINGERPRINT_HASH_NAME);
    for (index, byte) in fingerprint.iter().enumerate() {
        let _ = write!(text, "{}{byte:02X}", if index == 0 { ' ' } else { ':' });
    }
    text
}

/// The DER body of the first `label` block in `pem`.
fn pem_block(pem: &str, label: &str) -> Option<Vec<u8>> {
    let begin = alloc::format!("-----BEGIN {label}-----");
    let end = alloc::format!("-----END {label}-----");
    let start = pem.find(&begin)? + begin.len();
    let body = &pem[start..];
    let stop = body.find(&end)?;
    let base64_body: String = body[..stop]
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect();
    base64::engine::general_purpose::STANDARD
        .decode(base64_body)
        .ok()
}

fn parse_certificate_pem(pem: &str) -> Result<DtlsCertificate, DtlsSrtpError> {
    let certificate =
        pem_block(pem, CERTIFICATE_PEM_LABEL).ok_or(DtlsSrtpError::InvalidCertificatePem)?;
    let private_key = PRIVATE_KEY_PEM_LABELS
        .iter()
        .find_map(|label| pem_block(pem, label))
        .ok_or(DtlsSrtpError::InvalidCertificatePem)?;
    Ok(DtlsCertificate {
        certificate,
        private_key,
    })
}

fn encode_pem(label: &str, der: &[u8]) -> String {
    let body = base64::engine::general_purpose::STANDARD.encode(der);
    let mut text = alloc::format!("-----BEGIN {label}-----\n");
    for line in body.as_bytes().chunks(PEM_LINE_LENGTH) {
        text.push_str(core::str::from_utf8(line).unwrap_or_default());
        text.push('\n');
    }
    text.push_str(&alloc::format!("-----END {label}-----\n"));
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc_7983_splits_dtls_from_protected_media() {
        assert_eq!(classify_datagram(&[]), None);
        // STUN and TURN sit below the DTLS range, ZRTP above the media one.
        assert_eq!(classify_datagram(&[0x00]), None);
        assert_eq!(classify_datagram(&[0x40]), None);
        assert_eq!(classify_datagram(&[0xc0]), None);
        for first in [20_u8, 22, 63] {
            assert_eq!(
                classify_datagram(&[first]),
                Some(DtlsSrtpDatagram::Dtls),
                "{first:#04x}"
            );
        }
        // An RTP header's first byte is version 2 plus flags, an RTCP one the same.
        for first in [128_u8, 0x80 | 0x20, 191] {
            assert_eq!(
                classify_datagram(&[first]),
                Some(DtlsSrtpDatagram::Media),
                "{first:#04x}"
            );
        }
    }

    #[test]
    fn pem_round_trips_through_a_certificate() {
        let certificate =
            dimpl::certificate::generate_self_signed_certificate().expect("generate a certificate");
        let pem = alloc::format!(
            "{}{}",
            encode_pem(CERTIFICATE_PEM_LABEL, &certificate.certificate),
            encode_pem(PRIVATE_KEY_PEM_LABELS[0], &certificate.private_key)
        );
        let parsed = parse_certificate_pem(&pem).expect("parse the PEM back");
        assert_eq!(parsed.certificate, certificate.certificate);
        assert_eq!(parsed.private_key, certificate.private_key);

        // Neither block, then a certificate with no private key beside it.
        for incomplete in [
            String::from("not a certificate"),
            encode_pem(CERTIFICATE_PEM_LABEL, &certificate.certificate),
        ] {
            assert_eq!(
                parse_certificate_pem(&incomplete).err(),
                Some(DtlsSrtpError::InvalidCertificatePem)
            );
        }
    }

    /// The SDP `a=fingerprint` value form, as RFC 8122 writes it and as a
    /// launch line may write it: the hash name optional, the hex either case.
    #[test]
    fn a_fingerprint_round_trips_through_the_sdp_value_form() {
        let certificate =
            dimpl::certificate::generate_self_signed_certificate().expect("generate a certificate");
        let fingerprint = calculate_fingerprint(&certificate.certificate);
        let text = format_fingerprint(&fingerprint);
        assert!(text.starts_with(FINGERPRINT_HASH_NAME));
        assert_eq!(
            parse_fingerprint(&text).map(Vec::from),
            Some(fingerprint.clone())
        );

        let digest = text
            .split_once(' ')
            .expect("the hash name and the digest")
            .1
            .to_string();
        for accepted in [digest.clone(), digest.to_lowercase()] {
            assert_eq!(
                parse_fingerprint(&accepted).map(Vec::from),
                Some(fingerprint.clone()),
                "{accepted}"
            );
        }
        for refused in [
            String::new(),
            digest[..digest.len() - 3].to_string(),
            alloc::format!("{digest}:00"),
            digest.replace(':', ""),
            alloc::format!("sha-1 {digest}"),
        ] {
            assert_eq!(parse_fingerprint(&refused), None, "{refused}");
        }
    }

    #[test]
    fn one_connection_id_names_one_connection() {
        let first = connection_for_id("m1100-shared").expect("a connection");
        let second = connection_for_id("m1100-shared").expect("the same connection");
        assert!(Arc::ptr_eq(&first, &second));
        assert!(!Arc::ptr_eq(
            &first,
            &connection_for_id("m1100-other").expect("a different connection")
        ));

        drop(second);
        drop(first);
        // The registry holds the entry weakly, so the next request builds a new
        // session rather than handing back a dead one.
        let rebuilt = connection_for_id("m1100-shared").expect("a fresh connection");
        assert_eq!(
            lock_connection(&rebuilt).expect("lock").state(),
            DtlsSrtpState::New
        );
    }
}

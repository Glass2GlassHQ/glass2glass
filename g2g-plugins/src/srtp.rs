//! Authenticated encryption for SRTP and SRTCP: the RFC 7714 AES-GCM profiles
//! and the RFC 3711 / RFC 6188 AES counter-mode and HMAC-SHA1 ones.
//!
//! [`SrtpSender`] and [`SrtpReceiver`] each protect one synchronization
//! source. [`SrtpReceiverSet`] selects independent contexts through an
//! [`SrtpKeyProvider`]. The contexts derive distinct RTP and RTCP keys from a
//! master key and its master salt, track packet indices, and reject repeated
//! packets. The module has no socket or key-exchange code.
//!
//! An [`SrtpPolicy`] pairs the cipher with the authentication transform, and a
//! [`SrtpMasterKey`]'s byte count picks the default policy the way GStreamer's
//! `srtpenc` does. The two families put the optional MKI in different places:
//! RFC 7714 appends it after the AEAD tag, RFC 3711 puts it between the
//! protected body and the authentication tag.
//!
//! [`SrtpMasterKey`], [`SrtpKeySettings`], [`SrtpFlow`] and
//! [`forward_until_data_frame`] are the pieces the [`srtpenc`](crate::srtpenc)
//! and [`srtpdec`](crate::srtpdec) elements share, kept here so neither element
//! file owns the other's half.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;
use core::ops::Range;

use aes_gcm::aead::{array::Array, Aead, Payload};
use aes_gcm::aes::cipher::{BlockCipherEncrypt, KeyInit as AesKeyInit};
use aes_gcm::aes::{Aes128, Aes256};
use aes_gcm::{Aes128Gcm, Aes256Gcm};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::rtp::RtpHeader;
use g2g_core::{
    ByteStreamEncoding, Caps, G2gError, MemoryDomain, OutputSink, PipelinePacket, PropKind,
    PropertySpec,
};
use hmac::{Hmac, Mac};
use sha1::Sha1;
use zeroize::{Zeroize, Zeroizing};

type HmacSha1 = Hmac<Sha1>;

/// RFC 7714 always uses a 16-byte AES-GCM authentication tag.
pub const AUTHENTICATION_TAG_LENGTH: usize = 16;
/// The 80-bit HMAC-SHA1 tag of RFC 3711 section 4.2.1.
pub const HMAC_SHA1_80_TAG_LENGTH: usize = 10;
/// The 32-bit HMAC-SHA1 tag of RFC 3711 section 4.2.1.
pub const HMAC_SHA1_32_TAG_LENGTH: usize = 4;
/// RFC 7714 master and session salts are 12 bytes.
pub const AEAD_MASTER_SALT_LENGTH: usize = 12;
/// RFC 3711 master and session salts are 14 bytes.
pub const COUNTER_MODE_MASTER_SALT_LENGTH: usize = 14;
/// DTLS-SRTP protection profile id for `SRTP_AES128_CM_HMAC_SHA1_80`.
pub const DTLS_SRTP_AES128_CM_HMAC_SHA1_80: u16 = 0x0001;
/// DTLS-SRTP protection profile id for `SRTP_AES128_CM_HMAC_SHA1_32`.
pub const DTLS_SRTP_AES128_CM_HMAC_SHA1_32: u16 = 0x0002;
/// DTLS-SRTP protection profile id for `SRTP_NULL_HMAC_SHA1_80`.
pub const DTLS_SRTP_NULL_HMAC_SHA1_80: u16 = 0x0005;
/// DTLS-SRTP protection profile id for `SRTP_NULL_HMAC_SHA1_32`.
pub const DTLS_SRTP_NULL_HMAC_SHA1_32: u16 = 0x0006;
/// DTLS-SRTP protection profile id for `SRTP_AEAD_AES_128_GCM`.
pub const DTLS_SRTP_AEAD_AES_128_GCM: u16 = 0x0007;
/// DTLS-SRTP protection profile id for `SRTP_AEAD_AES_256_GCM`.
pub const DTLS_SRTP_AEAD_AES_256_GCM: u16 = 0x0008;
/// Default policy margin before the fixed SRTP key lifetime ends.
///
/// RFC 3711 section 9.2 fixes the hard lifetime but leaves the rekey point to
/// local policy.
pub const DEFAULT_SRTP_REKEY_MARGIN: u64 = 1 << 16;
/// Default policy margin before the fixed SRTCP key lifetime ends.
///
/// RFC 3711 section 9.2 fixes the hard lifetime but leaves the rekey point to
/// local policy.
pub const DEFAULT_SRTCP_REKEY_MARGIN: u32 = 1 << 16;
/// The RFC 3711 section 9.2 lifetime of one SRTP key instantiation.
pub const MAXIMUM_SRTP_KEY_INVOCATIONS: u64 = 1_u64 << 48;
/// The RFC 3711 section 9.2 lifetime of one SRTCP key instantiation.
pub const MAXIMUM_SRTCP_KEY_INVOCATIONS: u32 = 1_u32 << 31;
/// Smallest replay window a context accepts, in packets. The gst
/// `replay-window-size` range starts here.
pub const MINIMUM_REPLAY_WINDOW: usize = 64;
/// Largest replay window a context accepts, in packets.
pub const MAXIMUM_REPLAY_WINDOW: usize = 32768;
/// Replay window a context starts with, matching the gst default.
pub const DEFAULT_REPLAY_WINDOW: usize = 128;
/// Longest Master Key Identifier a key may carry, libsrtp's `SRTP_MAX_MKI_LEN`.
pub const MAXIMUM_MKI_LENGTH: usize = 128;
/// The E-flag and 31-bit SRTCP index every SRTCP packet ends with, before an
/// optional MKI.
pub const SRTCP_INDEX_LENGTH: usize = 4;

const RTP_ENCRYPTION_KEY_LABEL: u8 = 0x00;
const RTP_AUTHENTICATION_KEY_LABEL: u8 = 0x01;
const RTP_SALT_LABEL: u8 = 0x02;
const RTCP_ENCRYPTION_KEY_LABEL: u8 = 0x03;
const RTCP_AUTHENTICATION_KEY_LABEL: u8 = 0x04;
const RTCP_SALT_LABEL: u8 = 0x05;
const SEQUENCE_HALF_RANGE: u16 = 1 << 15;
const REPLAY_BITMAP_WORD_BITS: usize = 64;
const MAXIMUM_RTP_PACKET_INDEX: u64 = (1_u64 << 48) - 1;
const MAXIMUM_SRTCP_INDEX: u32 = (1_u32 << 31) - 1;
/// The top bit of the SRTCP index word, set when the body is encrypted.
const SRTCP_ENCRYPTION_FLAG: u32 = 1 << 31;
const RTCP_AUTHENTICATED_PREFIX_LENGTH: usize = 8;
/// The version every RTP and RTCP packet's first two bits carry.
const RTP_VERSION: u8 = 2;
/// Where an RTCP packet carries its synchronization source.
const RTCP_SOURCE_RANGE: Range<usize> = 4..8;

const AES_128_KEY_LENGTH: usize = 16;
const AES_256_KEY_LENGTH: usize = 32;
const AES_BLOCK_LENGTH: usize = 16;
/// RFC 3711 section 4.2.1 derives a 160-bit HMAC-SHA1 session key.
const SESSION_AUTHENTICATION_KEY_LENGTH: usize = 20;
/// The 48-bit SRTP packet index occupies these bytes of the counter block, the
/// session salt having been shifted left 16 bits.
const COUNTER_BLOCK_INDEX_RANGE: Range<usize> = 8..14;
/// Where the SSRC lands in the counter block, from its multiplication by 2^64.
const COUNTER_BLOCK_SOURCE_RANGE: Range<usize> = 4..8;

/// The encryption transform one flow runs, `null` through `aes-256-gcm` in
/// GStreamer's `GstSrtpCipherType` order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SrtpCipher {
    Null,
    Aes128CounterMode,
    Aes256CounterMode,
    Aes128Gcm,
    Aes256Gcm,
}

impl SrtpCipher {
    /// The property value GStreamer's `rtp-cipher` takes.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Aes128CounterMode => "aes-128-icm",
            Self::Aes256CounterMode => "aes-256-icm",
            Self::Aes128Gcm => "aes-128-gcm",
            Self::Aes256Gcm => "aes-256-gcm",
        }
    }

    pub fn from_property_value(text: &str) -> Option<Self> {
        [
            Self::Null,
            Self::Aes128CounterMode,
            Self::Aes256CounterMode,
            Self::Aes128Gcm,
            Self::Aes256Gcm,
        ]
        .into_iter()
        .find(|cipher| cipher.as_str() == text)
    }

    /// The master key lengths this cipher keys from. The NULL cipher derives
    /// only an authentication key, so either counter-mode length keys it: that
    /// is what GStreamer's 30- and 46-byte `key` buffers mean.
    pub const fn master_key_lengths(self) -> &'static [usize] {
        match self {
            Self::Null => &[AES_128_KEY_LENGTH, AES_256_KEY_LENGTH],
            Self::Aes128CounterMode | Self::Aes128Gcm => &[AES_128_KEY_LENGTH],
            Self::Aes256CounterMode | Self::Aes256Gcm => &[AES_256_KEY_LENGTH],
        }
    }

    /// The master salt that follows the master key in the `key` property.
    pub const fn master_salt_length(self) -> usize {
        match self {
            Self::Aes128Gcm | Self::Aes256Gcm => AEAD_MASTER_SALT_LENGTH,
            _ => COUNTER_MODE_MASTER_SALT_LENGTH,
        }
    }

    /// The session salt the transform itself reads, none for the NULL cipher.
    const fn session_salt_length(self) -> usize {
        match self {
            Self::Null => 0,
            _ => self.master_salt_length(),
        }
    }

    /// Whether the cipher carries its own authentication tag, so a separate
    /// authentication transform would be a second tag over the same bytes.
    pub const fn is_authenticated_encryption(self) -> bool {
        matches!(self, Self::Aes128Gcm | Self::Aes256Gcm)
    }
}

/// The message authentication transform one flow runs, matching GStreamer's
/// `GstSrtpAuthType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SrtpAuthentication {
    Null,
    HmacSha1Tag32,
    HmacSha1Tag80,
}

impl SrtpAuthentication {
    /// The property value GStreamer's `rtp-auth` takes.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::HmacSha1Tag32 => "hmac-sha1-32",
            Self::HmacSha1Tag80 => "hmac-sha1-80",
        }
    }

    pub fn from_property_value(text: &str) -> Option<Self> {
        [Self::Null, Self::HmacSha1Tag32, Self::HmacSha1Tag80]
            .into_iter()
            .find(|authentication| authentication.as_str() == text)
    }

    /// Octets of HMAC-SHA1 output the packet carries.
    pub const fn tag_length(self) -> usize {
        match self {
            Self::Null => 0,
            Self::HmacSha1Tag32 => HMAC_SHA1_32_TAG_LENGTH,
            Self::HmacSha1Tag80 => HMAC_SHA1_80_TAG_LENGTH,
        }
    }
}

/// How one flow is protected: the cipher and the authentication transform,
/// GStreamer's `rtp-cipher` / `rtp-auth` pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SrtpPolicy {
    pub cipher: SrtpCipher,
    pub authentication: SrtpAuthentication,
}

impl SrtpPolicy {
    /// A GCM cipher already authenticates, so it takes no separate transform.
    pub fn new(cipher: SrtpCipher, authentication: SrtpAuthentication) -> Result<Self, SrtpError> {
        if cipher.is_authenticated_encryption() && authentication != SrtpAuthentication::Null {
            return Err(SrtpError::AeadCipherTakesNoAuthentication);
        }
        Ok(Self {
            cipher,
            authentication,
        })
    }

    /// The authentication a cipher takes when nothing asked for one: none for
    /// the AEAD ciphers, GStreamer's `hmac-sha1-80` default for the rest.
    pub const fn default_authentication(cipher: SrtpCipher) -> SrtpAuthentication {
        if cipher.is_authenticated_encryption() {
            SrtpAuthentication::Null
        } else {
            SrtpAuthentication::HmacSha1Tag80
        }
    }

    /// The master key length this policy takes for `material_length` bytes of
    /// key plus salt, `None` when the count matches no key length.
    pub fn master_key_length_of(&self, material_length: usize) -> Option<usize> {
        let key_length = material_length.checked_sub(self.cipher.master_salt_length())?;
        self.cipher
            .master_key_lengths()
            .contains(&key_length)
            .then_some(key_length)
    }

    /// Whether every protected packet carries a tag a receiver checks.
    pub const fn is_authenticated(&self) -> bool {
        self.cipher.is_authenticated_encryption()
            || !matches!(self.authentication, SrtpAuthentication::Null)
    }

    /// The DTLS-SRTP protection profile id RFC 5764 or RFC 7714 assigns this
    /// policy. AES-256 counter mode has none: RFC 5764 reserved 0x0003 and
    /// 0x0004 without assigning them.
    pub const fn dtls_protection_profile(&self) -> Option<u16> {
        Some(match (self.cipher, self.authentication) {
            (SrtpCipher::Aes128CounterMode, SrtpAuthentication::HmacSha1Tag80) => {
                DTLS_SRTP_AES128_CM_HMAC_SHA1_80
            }
            (SrtpCipher::Aes128CounterMode, SrtpAuthentication::HmacSha1Tag32) => {
                DTLS_SRTP_AES128_CM_HMAC_SHA1_32
            }
            (SrtpCipher::Null, SrtpAuthentication::HmacSha1Tag80) => DTLS_SRTP_NULL_HMAC_SHA1_80,
            (SrtpCipher::Null, SrtpAuthentication::HmacSha1Tag32) => DTLS_SRTP_NULL_HMAC_SHA1_32,
            (SrtpCipher::Aes128Gcm, _) => DTLS_SRTP_AEAD_AES_128_GCM,
            (SrtpCipher::Aes256Gcm, _) => DTLS_SRTP_AEAD_AES_256_GCM,
            _ => return None,
        })
    }
}

/// Master key material returned by an [`SrtpKeyProvider`].
#[derive(Debug)]
pub struct SrtpKeyingMaterial {
    key: SrtpMasterKey,
    initial_rollover_counter: u32,
}

impl SrtpKeyingMaterial {
    pub fn new(
        policy: SrtpPolicy,
        master_key: &[u8],
        master_salt: &[u8],
        initial_rollover_counter: u32,
    ) -> Result<Self, SrtpError> {
        Ok(Self {
            key: SrtpMasterKey::new(policy, master_key, master_salt)?,
            initial_rollover_counter,
        })
    }

    /// Select this key with a Master Key Identifier, 1..=[`MAXIMUM_MKI_LENGTH`]
    /// bytes. Every key of one context has to carry an MKI of the same length.
    pub fn with_mki(mut self, mki: &[u8]) -> Result<Self, SrtpError> {
        self.key = self.key.with_mki(mki)?;
        Ok(self)
    }

    pub fn mki(&self) -> Option<&[u8]> {
        self.key.mki()
    }

    pub fn policy(&self) -> SrtpPolicy {
        self.key.policy()
    }

    fn derive_keys(&self) -> Result<SessionKeys, SrtpError> {
        SessionKeys::derive(
            self.key.policy(),
            self.key.master_key(),
            self.key.master_salt(),
        )
    }
}

/// The policy a master key and its salt occupy `length` bytes of. The GCM
/// profiles pair a 16- or 32-byte key with a 12-byte salt (28 and 44 bytes),
/// the counter-mode ones with a 14-byte salt (30 and 46), so the length alone
/// picks the cipher, and the authentication follows the cipher.
pub fn policy_for_key_material(length: usize) -> Option<SrtpPolicy> {
    [
        SrtpCipher::Aes128Gcm,
        SrtpCipher::Aes256Gcm,
        SrtpCipher::Aes128CounterMode,
        SrtpCipher::Aes256CounterMode,
    ]
    .into_iter()
    .find(|cipher| {
        cipher
            .master_key_lengths()
            .contains(&(length.saturating_sub(cipher.master_salt_length())))
            && length > cipher.master_salt_length()
    })
    .map(|cipher| SrtpPolicy {
        cipher,
        authentication: SrtpPolicy::default_authentication(cipher),
    })
}

/// A master key immediately followed by its master salt, the layout an
/// element's `key` property carries, with the policy it is used under.
#[derive(Clone)]
pub struct SrtpMasterKey {
    policy: SrtpPolicy,
    key_length: usize,
    bytes: Zeroizing<Vec<u8>>,
    mki: Option<Vec<u8>>,
}

impl fmt::Debug for SrtpMasterKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SrtpMasterKey")
            .field("policy", &self.policy)
            .field("mki_length", &self.mki.as_ref().map_or(0, Vec::len))
            .finish_non_exhaustive()
    }
}

impl SrtpMasterKey {
    pub fn new(
        policy: SrtpPolicy,
        master_key: &[u8],
        master_salt: &[u8],
    ) -> Result<Self, SrtpError> {
        if !policy
            .cipher
            .master_key_lengths()
            .contains(&master_key.len())
        {
            return Err(SrtpError::InvalidMasterKeyLength {
                expected: policy.cipher.master_key_lengths()[0],
                actual: master_key.len(),
            });
        }
        if master_salt.len() != policy.cipher.master_salt_length() {
            return Err(SrtpError::InvalidMasterSaltLength {
                expected: policy.cipher.master_salt_length(),
                actual: master_salt.len(),
            });
        }
        let mut bytes = Zeroizing::new(Vec::with_capacity(master_key.len() + master_salt.len()));
        bytes.extend_from_slice(master_key);
        bytes.extend_from_slice(master_salt);
        Ok(Self {
            policy,
            key_length: master_key.len(),
            bytes,
            mki: None,
        })
    }

    /// Hexadecimal digits, the master key immediately followed by the master
    /// salt: 56 digits for AES-128-GCM, 88 for AES-256-GCM, 60 for
    /// AES-128-ICM, 92 for AES-256-ICM. `None` when the text is not hexadecimal
    /// or its length matches no cipher.
    pub fn from_hexadecimal(text: &str) -> Option<Self> {
        let bytes = decode_hexadecimal(text)?;
        let policy = policy_for_key_material(bytes.len())?;
        Some(Self {
            policy,
            key_length: policy.master_key_length_of(bytes.len())?,
            bytes,
            mki: None,
        })
    }

    /// Tag this key with a Master Key Identifier, 1..=[`MAXIMUM_MKI_LENGTH`]
    /// bytes. A sender appends it to every packet it protects.
    pub fn with_mki(mut self, mki: &[u8]) -> Result<Self, SrtpError> {
        self.mki = Some(validated_mki(mki)?);
        Ok(self)
    }

    pub fn mki(&self) -> Option<&[u8]> {
        self.mki.as_deref()
    }

    pub fn policy(&self) -> SrtpPolicy {
        self.policy
    }

    pub fn master_key(&self) -> &[u8] {
        &self.bytes[..self.key_length]
    }

    pub fn master_salt(&self) -> &[u8] {
        &self.bytes[self.key_length..]
    }

    /// This key as the material an [`SrtpKeyProvider`] hands a new context.
    pub fn keying_material(
        &self,
        initial_rollover_counter: u32,
    ) -> Result<SrtpKeyingMaterial, SrtpError> {
        Ok(SrtpKeyingMaterial {
            key: self.clone(),
            initial_rollover_counter,
        })
    }
}

/// The cipher an element reports before it has a key to pick one from, the
/// GStreamer `rtp-cipher` default.
pub const DEFAULT_CIPHER: SrtpCipher = SrtpCipher::Aes128CounterMode;

/// The closed set of `rtp-cipher` / `rtcp-cipher` values, GStreamer's
/// `GstSrtpCipherType` nicks.
pub const CIPHER_VALUES: &str = "null | aes-128-icm | aes-256-icm | aes-128-gcm | aes-256-gcm";
/// The closed set of `rtp-auth` / `rtcp-auth` values, GStreamer's
/// `GstSrtpAuthType` nicks.
pub const AUTHENTICATION_VALUES: &str = "null | hmac-sha1-32 | hmac-sha1-80";

/// The four protection properties `srtpenc` and `srtpdec` both declare. Only
/// the pair naming the flow an instance carries is applied, the way
/// `rtcp-encrypt` is read on an RTCP instance alone.
pub const RTP_CIPHER_PROPERTY: PropertySpec = PropertySpec::new(
    "rtp-cipher",
    PropKind::Str,
    "cipher the RTP flow is protected with. Unset follows the `key` length: 28 or 44 bytes is \
     AES-GCM, 30 or 46 is AES counter mode",
)
.with_default("aes-128-icm")
.with_enum_values(CIPHER_VALUES);
pub const RTCP_CIPHER_PROPERTY: PropertySpec = PropertySpec::new(
    "rtcp-cipher",
    PropKind::Str,
    "cipher the RTCP flow is protected with. Unset follows the `key` length: 28 or 44 bytes is \
     AES-GCM, 30 or 46 is AES counter mode",
)
.with_default("aes-128-icm")
.with_enum_values(CIPHER_VALUES);
pub const RTP_AUTHENTICATION_PROPERTY: PropertySpec = PropertySpec::new(
    "rtp-auth",
    PropKind::Str,
    "authentication the RTP flow carries. Unset is hmac-sha1-80 under a counter-mode or NULL \
     cipher and null under an AES-GCM one, which carries its own tag",
)
.with_default("hmac-sha1-80")
.with_enum_values(AUTHENTICATION_VALUES);
pub const RTCP_AUTHENTICATION_PROPERTY: PropertySpec = PropertySpec::new(
    "rtcp-auth",
    PropKind::Str,
    "authentication the RTCP flow carries. Unset is hmac-sha1-80 under a counter-mode or NULL \
     cipher and null under an AES-GCM one, which carries its own tag",
)
.with_default("hmac-sha1-80")
.with_enum_values(AUTHENTICATION_VALUES);

/// The `key` property's description, the same on both elements.
pub const KEY_PROPERTY_BLURB: &str =
    "hexadecimal master key immediately followed by the master salt: 56 digits for AES-128-GCM, \
     88 for AES-256-GCM, 60 for AES-128-ICM, 92 for AES-256-ICM. Reads back empty";

/// Every packet an unauthenticated policy protects can be forged, which the
/// element says once when a launch line asked for one.
pub const UNAUTHENTICATED_POLICY_WARNING: &str =
    "this flow carries no authentication tag, so a forged packet cannot be told from a real one";

/// The flow a protection property names, `None` for any other property.
pub fn protection_property_flow(name: &str) -> Option<SrtpFlow> {
    match name {
        "rtp-cipher" | "rtp-auth" => Some(SrtpFlow::Rtp),
        "rtcp-cipher" | "rtcp-auth" => Some(SrtpFlow::Rtcp),
        _ => None,
    }
}

/// What a flow's `cipher` and `auth` properties asked for, each `None` until
/// something set it.
#[derive(Clone, Copy, Debug, Default)]
struct RequestedProtection {
    cipher: Option<SrtpCipher>,
    authentication: Option<SrtpAuthentication>,
}

/// The `key` bytes plus the per-flow cipher and authentication a launch line
/// named, resolved into one [`SrtpPolicy`] once the element knows which flow it
/// carries. `srtpenc` and `srtpdec` share it so their property halves cannot
/// drift.
#[derive(Default)]
pub struct SrtpKeySettings {
    key_material: Option<Zeroizing<Vec<u8>>>,
    rtp: RequestedProtection,
    rtcp: RequestedProtection,
}

impl fmt::Debug for SrtpKeySettings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SrtpKeySettings")
            .field("has_key", &self.key_material.is_some())
            .field("rtp", &self.rtp)
            .field("rtcp", &self.rtcp)
            .finish()
    }
}

impl SrtpKeySettings {
    /// Both flows pinned to one key and the policy it was built with, the route
    /// a programmatic caller takes instead of setting properties.
    pub fn from_master_key(key: &SrtpMasterKey) -> Self {
        let requested = RequestedProtection {
            cipher: Some(key.policy().cipher),
            authentication: Some(key.policy().authentication),
        };
        let mut material = Zeroizing::new(Vec::new());
        material.extend_from_slice(key.master_key());
        material.extend_from_slice(key.master_salt());
        Self {
            key_material: Some(material),
            rtp: requested,
            rtcp: requested,
        }
    }

    /// Hexadecimal digits, the master key immediately followed by the master
    /// salt. Refused when the byte count matches no cipher, or contradicts a
    /// cipher a property already named.
    pub fn set_key_hexadecimal(&mut self, text: &str) -> Result<(), SrtpError> {
        let bytes = decode_hexadecimal(text).ok_or(SrtpError::UnsupportedKeyMaterialLength {
            actual: text.len() / 2,
        })?;
        if policy_for_key_material(bytes.len()).is_none() {
            return Err(SrtpError::UnsupportedKeyMaterialLength {
                actual: bytes.len(),
            });
        }
        for (flow, requested) in [(SrtpFlow::Rtp, self.rtp), (SrtpFlow::Rtcp, self.rtcp)] {
            if requested.cipher.is_none() {
                continue;
            }
            self.policy(flow)?.master_key_length_of(bytes.len()).ok_or(
                SrtpError::UnsupportedKeyMaterialLength {
                    actual: bytes.len(),
                },
            )?;
        }
        self.key_material = Some(bytes);
        Ok(())
    }

    pub fn has_key(&self) -> bool {
        self.key_material.is_some()
    }

    pub fn set_cipher(&mut self, flow: SrtpFlow, cipher: SrtpCipher) -> Result<(), SrtpError> {
        let requested = self.requested_mut(flow);
        let previous = requested.cipher;
        requested.cipher = Some(cipher);
        self.check(flow).inspect_err(|_| {
            self.requested_mut(flow).cipher = previous;
        })
    }

    pub fn set_authentication(
        &mut self,
        flow: SrtpFlow,
        authentication: SrtpAuthentication,
    ) -> Result<(), SrtpError> {
        let requested = self.requested_mut(flow);
        let previous = requested.authentication;
        requested.authentication = Some(authentication);
        self.check(flow).inspect_err(|_| {
            self.requested_mut(flow).authentication = previous;
        })
    }

    /// The cipher this flow runs: the one a property named, else the one the
    /// key length picks, else the GStreamer default.
    pub fn cipher(&self, flow: SrtpFlow) -> SrtpCipher {
        self.requested(flow)
            .cipher
            .or_else(|| {
                policy_for_key_material(self.key_material.as_ref()?.len())
                    .map(|policy| policy.cipher)
            })
            .unwrap_or(DEFAULT_CIPHER)
    }

    /// The authentication this flow runs: the one a property named, else the
    /// one the cipher implies.
    pub fn authentication(&self, flow: SrtpFlow) -> SrtpAuthentication {
        self.requested(flow)
            .authentication
            .unwrap_or_else(|| SrtpPolicy::default_authentication(self.cipher(flow)))
    }

    pub fn policy(&self, flow: SrtpFlow) -> Result<SrtpPolicy, SrtpError> {
        SrtpPolicy::new(self.cipher(flow), self.authentication(flow))
    }

    /// The key this flow protects with, split where its policy says.
    pub fn master_key(&self, flow: SrtpFlow) -> Result<SrtpMasterKey, SrtpError> {
        let bytes = self.key_material.as_ref().ok_or(SrtpError::MissingKey)?;
        let policy = self.policy(flow)?;
        let key_length = policy.master_key_length_of(bytes.len()).ok_or(
            SrtpError::UnsupportedKeyMaterialLength {
                actual: bytes.len(),
            },
        )?;
        SrtpMasterKey::new(policy, &bytes[..key_length], &bytes[key_length..])
    }

    /// Whether the resolved policy and the key length still agree.
    fn check(&self, flow: SrtpFlow) -> Result<(), SrtpError> {
        let policy = self.policy(flow)?;
        let Some(bytes) = &self.key_material else {
            return Ok(());
        };
        policy.master_key_length_of(bytes.len()).map(|_| ()).ok_or(
            SrtpError::UnsupportedKeyMaterialLength {
                actual: bytes.len(),
            },
        )
    }

    fn requested(&self, flow: SrtpFlow) -> &RequestedProtection {
        match flow {
            SrtpFlow::Rtp => &self.rtp,
            SrtpFlow::Rtcp => &self.rtcp,
        }
    }

    fn requested_mut(&mut self, flow: SrtpFlow) -> &mut RequestedProtection {
        match flow {
            SrtpFlow::Rtp => &mut self.rtp,
            SrtpFlow::Rtcp => &mut self.rtcp,
        }
    }
}

const fn hexadecimal_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// The bytes `text` spells, two hexadecimal digits each. `None` when it is not
/// an even run of hexadecimal digits. Zeroized because the master key comes
/// through here.
pub(crate) fn decode_hexadecimal(text: &str) -> Option<Zeroizing<Vec<u8>>> {
    let (pairs, remainder) = text.as_bytes().as_chunks::<2>();
    if !remainder.is_empty() {
        return None;
    }
    let mut bytes = Zeroizing::new(Vec::with_capacity(pairs.len()));
    for pair in pairs {
        bytes.push(hexadecimal_digit(pair[0])? << 4 | hexadecimal_digit(pair[1])?);
    }
    Some(bytes)
}

/// `bytes` as lowercase hexadecimal digits, the form a property reads back.
pub(crate) fn encode_hexadecimal(bytes: &[u8]) -> alloc::string::String {
    use core::fmt::Write;

    let mut text = alloc::string::String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

/// An MKI is 1..=[`MAXIMUM_MKI_LENGTH`] bytes, libsrtp's `SRTP_MAX_MKI_LEN`.
pub(crate) fn validated_mki(mki: &[u8]) -> Result<Vec<u8>, SrtpError> {
    if mki.is_empty() || mki.len() > MAXIMUM_MKI_LENGTH {
        return Err(SrtpError::InvalidMkiLength { actual: mki.len() });
    }
    Ok(mki.to_vec())
}

/// Supplies master keys when a receiver first observes a synchronization source.
pub trait SrtpKeyProvider {
    /// Every key the context for this source is built from, empty when there is
    /// none. More than one key means each carries an MKI of the same length and
    /// the sender picks between them per packet.
    fn keys_for(&mut self, synchronization_source: u32) -> Vec<SrtpKeyingMaterial>;
}

impl<F> SrtpKeyProvider for F
where
    F: FnMut(u32) -> Vec<SrtpKeyingMaterial>,
{
    fn keys_for(&mut self, synchronization_source: u32) -> Vec<SrtpKeyingMaterial> {
        self(synchronization_source)
    }
}

/// Whether an SRTCP packet encrypts its body or only authenticates it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RtcpProtectionMode {
    Encrypt,
    AuthenticateOnly,
}

/// Whether a sender should request a replacement master key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyUsage {
    Normal,
    SoftLimitReached,
}

/// Per-protocol usage of the current sender master key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SrtpKeyUsage {
    pub srtp: KeyUsage,
    pub srtcp: KeyUsage,
}

/// Caller-selected rekey thresholds below the fixed RFC 3711 hard limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SrtpSoftLimits {
    pub srtp_packets: u64,
    pub srtcp_packets: u32,
}

impl SrtpSoftLimits {
    /// Whether both thresholds sit below the fixed RFC 3711 hard limits, so a
    /// rekey is still possible when one is reached.
    pub fn is_valid(self) -> bool {
        self.srtp_packets < MAXIMUM_SRTP_KEY_INVOCATIONS
            && self.srtcp_packets < MAXIMUM_SRTCP_KEY_INVOCATIONS
    }
}

impl Default for SrtpSoftLimits {
    fn default() -> Self {
        Self {
            srtp_packets: MAXIMUM_SRTP_KEY_INVOCATIONS - DEFAULT_SRTP_REKEY_MARGIN,
            srtcp_packets: MAXIMUM_SRTCP_KEY_INVOCATIONS - DEFAULT_SRTCP_REKEY_MARGIN,
        }
    }
}

/// A packet or context error reported by the SRTP layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SrtpError {
    InvalidMasterKeyLength { expected: usize, actual: usize },
    InvalidMasterSaltLength { expected: usize, actual: usize },
    InvalidRtpPacket,
    InvalidRtcpPacket,
    WrongSynchronizationSource { expected: u32, actual: u32 },
    AuthenticationFailed,
    RepeatedPacket,
    PacketTooOld,
    PacketIndexExhausted,
    KeyLifetimeExhausted,
    MissingKey,
    InvalidSoftLimit,
    PacketTooLarge,
    InvalidReplayWindow { size: usize },
    InvalidMkiLength { actual: usize },
    InconsistentMki,
    AeadCipherTakesNoAuthentication,
    UnsupportedKeyMaterialLength { actual: usize },
}

impl fmt::Display for SrtpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMasterKeyLength { expected, actual } => write!(
                formatter,
                "invalid SRTP master key length, expected {expected}, got {actual}"
            ),
            Self::InvalidMasterSaltLength { expected, actual } => write!(
                formatter,
                "invalid SRTP master salt length, expected {expected}, got {actual}"
            ),
            Self::InvalidRtpPacket => formatter.write_str("invalid RTP packet"),
            Self::InvalidRtcpPacket => formatter.write_str("invalid RTCP packet"),
            Self::WrongSynchronizationSource { expected, actual } => write!(
                formatter,
                "wrong synchronization source, expected {expected:#010x}, got {actual:#010x}"
            ),
            Self::AuthenticationFailed => formatter.write_str("SRTP authentication failed"),
            Self::RepeatedPacket => formatter.write_str("SRTP packet index was already used"),
            Self::PacketTooOld => formatter.write_str("SRTP packet is outside the replay window"),
            Self::PacketIndexExhausted => formatter.write_str("SRTP packet index is exhausted"),
            Self::KeyLifetimeExhausted => formatter.write_str("SRTP key lifetime is exhausted"),
            Self::MissingKey => formatter.write_str("no SRTP key is available for this source"),
            Self::InvalidSoftLimit => formatter.write_str("invalid SRTP soft key-use limit"),
            Self::PacketTooLarge => {
                formatter.write_str("SRTP packet exceeds the cipher's per-packet limit")
            }
            Self::InvalidReplayWindow { size } => write!(
                formatter,
                "replay window of {size} packets is outside {MINIMUM_REPLAY_WINDOW}..={MAXIMUM_REPLAY_WINDOW}"
            ),
            Self::InvalidMkiLength { actual } => write!(
                formatter,
                "MKI length {actual} is outside 1..={MAXIMUM_MKI_LENGTH}"
            ),
            Self::InconsistentMki => formatter
                .write_str("the keys of one SRTP context need one shared MKI length"),
            Self::AeadCipherTakesNoAuthentication => formatter.write_str(
                "an AES-GCM cipher carries its own tag, so it takes no separate authentication",
            ),
            Self::UnsupportedKeyMaterialLength { actual } => write!(
                formatter,
                "{actual} bytes of key material match no SRTP cipher's key plus salt"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SrtpError {}

/// AES as a raw block cipher, the primitive both the RFC 3711 key derivation
/// and the counter-mode transform build their keystreams from.
enum AesBlockCipher {
    Aes128(Box<Aes128>),
    Aes256(Box<Aes256>),
}

impl fmt::Debug for AesBlockCipher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("AesBlockCipher { .. }")
    }
}

impl AesBlockCipher {
    fn new(key: &[u8]) -> Result<Self, SrtpError> {
        match key.len() {
            AES_128_KEY_LENGTH => Ok(Self::Aes128(Box::new(
                Aes128::new_from_slice(key).map_err(|_| invalid_key_length(key.len()))?,
            ))),
            AES_256_KEY_LENGTH => Ok(Self::Aes256(Box::new(
                Aes256::new_from_slice(key).map_err(|_| invalid_key_length(key.len()))?,
            ))),
            actual => Err(invalid_key_length(actual)),
        }
    }

    fn encrypt_block(&self, input: [u8; AES_BLOCK_LENGTH]) -> [u8; AES_BLOCK_LENGTH] {
        let mut block = Array(input);
        match self {
            Self::Aes128(cipher) => cipher.encrypt_block(&mut block),
            Self::Aes256(cipher) => cipher.encrypt_block(&mut block),
        }
        block.0
    }
}

fn invalid_key_length(actual: usize) -> SrtpError {
    SrtpError::InvalidMasterKeyLength {
        expected: AES_128_KEY_LENGTH,
        actual,
    }
}

enum AesGcmCipher {
    Aes128(Box<Aes128Gcm>),
    Aes256(Box<Aes256Gcm>),
}

impl fmt::Debug for AesGcmCipher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("AesGcmCipher { .. }")
    }
}

impl AesGcmCipher {
    fn new(cipher: SrtpCipher, key: &[u8]) -> Result<Self, SrtpError> {
        match cipher {
            SrtpCipher::Aes256Gcm => Ok(Self::Aes256(Box::new(
                Aes256Gcm::new_from_slice(key).map_err(|_| invalid_key_length(key.len()))?,
            ))),
            _ => Ok(Self::Aes128(Box::new(
                Aes128Gcm::new_from_slice(key).map_err(|_| invalid_key_length(key.len()))?,
            ))),
        }
    }

    fn encrypt(
        &self,
        initialization_vector: [u8; 12],
        associated_data: &[u8],
        plaintext: &[u8],
    ) -> Result<Vec<u8>, SrtpError> {
        let payload = Payload {
            msg: plaintext,
            aad: associated_data,
        };
        let result = match self {
            Self::Aes128(cipher) => cipher.encrypt(&Array(initialization_vector), payload),
            Self::Aes256(cipher) => cipher.encrypt(&Array(initialization_vector), payload),
        };
        result.map_err(|_| SrtpError::PacketTooLarge)
    }

    fn decrypt(
        &self,
        initialization_vector: [u8; 12],
        associated_data: &[u8],
        ciphertext: &[u8],
    ) -> Result<Vec<u8>, SrtpError> {
        let payload = Payload {
            msg: ciphertext,
            aad: associated_data,
        };
        let result = match self {
            Self::Aes128(cipher) => cipher.decrypt(&Array(initialization_vector), payload),
            Self::Aes256(cipher) => cipher.decrypt(&Array(initialization_vector), payload),
        };
        result.map_err(|_| SrtpError::AuthenticationFailed)
    }
}

/// One flow's derived transform state: the cipher with its session salt, and
/// the HMAC-SHA1 key when the policy authenticates separately.
struct FlowKeys {
    cipher: FlowCipher,
    authentication: SrtpAuthentication,
    authentication_key: Zeroizing<[u8; SESSION_AUTHENTICATION_KEY_LENGTH]>,
}

enum FlowCipher {
    Null,
    CounterMode {
        cipher: AesBlockCipher,
        salt: [u8; COUNTER_MODE_MASTER_SALT_LENGTH],
    },
    AuthenticatedEncryption {
        cipher: AesGcmCipher,
        salt: [u8; AEAD_MASTER_SALT_LENGTH],
    },
}

impl fmt::Debug for FlowKeys {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("FlowKeys { .. }")
    }
}

impl Drop for FlowCipher {
    fn drop(&mut self) {
        match self {
            Self::Null => {}
            Self::CounterMode { salt, .. } => salt.zeroize(),
            Self::AuthenticatedEncryption { salt, .. } => salt.zeroize(),
        }
    }
}

impl FlowKeys {
    fn derive(
        policy: SrtpPolicy,
        master: &AesBlockCipher,
        master_salt: &[u8; COUNTER_MODE_MASTER_SALT_LENGTH],
        labels: FlowLabels,
    ) -> Result<Self, SrtpError> {
        let key_length = *policy
            .cipher
            .master_key_lengths()
            .first()
            .ok_or(SrtpError::MissingKey)?;
        let mut session_key = Zeroizing::new([0_u8; AES_256_KEY_LENGTH]);
        let mut session_salt = Zeroizing::new([0_u8; COUNTER_MODE_MASTER_SALT_LENGTH]);
        let mut authentication_key = Zeroizing::new([0_u8; SESSION_AUTHENTICATION_KEY_LENGTH]);
        let salt_length = policy.cipher.session_salt_length();
        let cipher = match policy.cipher {
            SrtpCipher::Null => FlowCipher::Null,
            SrtpCipher::Aes128CounterMode | SrtpCipher::Aes256CounterMode => {
                derive_session_value(
                    master,
                    master_salt,
                    labels.encryption_key,
                    &mut session_key[..key_length],
                )?;
                derive_session_value(
                    master,
                    master_salt,
                    labels.salt,
                    &mut session_salt[..salt_length],
                )?;
                FlowCipher::CounterMode {
                    cipher: AesBlockCipher::new(&session_key[..key_length])?,
                    salt: *session_salt,
                }
            }
            SrtpCipher::Aes128Gcm | SrtpCipher::Aes256Gcm => {
                derive_session_value(
                    master,
                    master_salt,
                    labels.encryption_key,
                    &mut session_key[..key_length],
                )?;
                derive_session_value(
                    master,
                    master_salt,
                    labels.salt,
                    &mut session_salt[..salt_length],
                )?;
                FlowCipher::AuthenticatedEncryption {
                    cipher: AesGcmCipher::new(policy.cipher, &session_key[..key_length])?,
                    salt: session_salt[..AEAD_MASTER_SALT_LENGTH]
                        .try_into()
                        .map_err(|_| SrtpError::InvalidMasterSaltLength {
                            expected: AEAD_MASTER_SALT_LENGTH,
                            actual: salt_length,
                        })?,
                }
            }
        };
        if policy.authentication != SrtpAuthentication::Null {
            derive_session_value(
                master,
                master_salt,
                labels.authentication_key,
                &mut authentication_key[..],
            )?;
        }
        Ok(Self {
            cipher,
            authentication: policy.authentication,
            authentication_key,
        })
    }

    fn encrypts(&self) -> bool {
        !matches!(self.cipher, FlowCipher::Null)
    }

    /// Where the MKI sits in a protected packet of `packet_length` bytes. RFC
    /// 7714 appends it after the AEAD tag, so at the very end; RFC 3711 figures
    /// 1 and 2 put it between the protected body and the authentication tag.
    fn mki_range(&self, packet_length: usize, mki_length: usize) -> Option<Range<usize>> {
        let end = match self.cipher {
            FlowCipher::AuthenticatedEncryption { .. } => packet_length,
            _ => packet_length.checked_sub(self.authentication.tag_length())?,
        };
        Some(end.checked_sub(mki_length)?..end)
    }

    /// XOR the counter-mode keystream over `data`, in place and in both
    /// directions. A NULL cipher leaves the bytes alone.
    fn apply_cipher(
        &self,
        data: &mut [u8],
        synchronization_source: u32,
        packet_index: u64,
    ) -> Result<(), SrtpError> {
        let FlowCipher::CounterMode { cipher, salt } = &self.cipher else {
            return Ok(());
        };
        let block = counter_mode_block(salt, synchronization_source, packet_index);
        apply_counter_mode(cipher, block, data)
    }

    fn authentication_tag(&self, message: &[&[u8]]) -> Option<Vec<u8>> {
        let tag_length = self.authentication.tag_length();
        if tag_length == 0 {
            return None;
        }
        let mut mac = self.mac();
        for part in message {
            mac.update(part);
        }
        Some(mac.finalize().into_bytes()[..tag_length].to_vec())
    }

    fn verify_authentication_tag(&self, message: &[&[u8]], tag: &[u8]) -> Result<(), SrtpError> {
        if self.authentication.tag_length() == 0 {
            return Ok(());
        }
        let mut mac = self.mac();
        for part in message {
            mac.update(part);
        }
        mac.verify_truncated_left(tag)
            .map_err(|_| SrtpError::AuthenticationFailed)
    }

    fn mac(&self) -> HmacSha1 {
        HmacSha1::new_from_slice(&self.authentication_key[..])
            .expect("HMAC-SHA1 accepts a key of any length")
    }
}

/// The three RFC 3711 key-derivation labels of one flow.
#[derive(Clone, Copy)]
struct FlowLabels {
    encryption_key: u8,
    authentication_key: u8,
    salt: u8,
}

const RTP_LABELS: FlowLabels = FlowLabels {
    encryption_key: RTP_ENCRYPTION_KEY_LABEL,
    authentication_key: RTP_AUTHENTICATION_KEY_LABEL,
    salt: RTP_SALT_LABEL,
};
const RTCP_LABELS: FlowLabels = FlowLabels {
    encryption_key: RTCP_ENCRYPTION_KEY_LABEL,
    authentication_key: RTCP_AUTHENTICATION_KEY_LABEL,
    salt: RTCP_SALT_LABEL,
};

struct SessionKeys {
    rtp: FlowKeys,
    rtcp: FlowKeys,
}

impl fmt::Debug for SessionKeys {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SessionKeys { .. }")
    }
}

impl SessionKeys {
    fn derive(
        policy: SrtpPolicy,
        master_key: &[u8],
        master_salt: &[u8],
    ) -> Result<Self, SrtpError> {
        if !policy
            .cipher
            .master_key_lengths()
            .contains(&master_key.len())
        {
            return Err(SrtpError::InvalidMasterKeyLength {
                expected: policy.cipher.master_key_lengths()[0],
                actual: master_key.len(),
            });
        }
        if master_salt.len() != policy.cipher.master_salt_length() {
            return Err(SrtpError::InvalidMasterSaltLength {
                expected: policy.cipher.master_salt_length(),
                actual: master_salt.len(),
            });
        }
        // The RFC 7714 salt is two octets shorter than the 112-bit input the
        // RFC 3711 key derivation reads, so it is padded on the right.
        let mut padded_master_salt = Zeroizing::new([0_u8; COUNTER_MODE_MASTER_SALT_LENGTH]);
        padded_master_salt[..master_salt.len()].copy_from_slice(master_salt);
        let master = AesBlockCipher::new(master_key)?;
        Ok(Self {
            rtp: FlowKeys::derive(policy, &master, &padded_master_salt, RTP_LABELS)?,
            rtcp: FlowKeys::derive(policy, &master, &padded_master_salt, RTCP_LABELS)?,
        })
    }

    /// A context built straight from session values, for the RFC 7714 vectors,
    /// which give the session key and salt rather than a master key.
    #[cfg(test)]
    fn from_session_values(
        policy: SrtpPolicy,
        key: &[u8],
        salt: [u8; AEAD_MASTER_SALT_LENGTH],
    ) -> Self {
        let flow = || FlowKeys {
            cipher: FlowCipher::AuthenticatedEncryption {
                cipher: AesGcmCipher::new(policy.cipher, key).expect("valid test key"),
                salt,
            },
            authentication: policy.authentication,
            authentication_key: Zeroizing::new([0_u8; SESSION_AUTHENTICATION_KEY_LENGTH]),
        };
        Self {
            rtp: flow(),
            rtcp: flow(),
        }
    }

    fn flow_keys(&self, flow: SrtpFlow) -> &FlowKeys {
        match flow {
            SrtpFlow::Rtp => &self.rtp,
            SrtpFlow::Rtcp => &self.rtcp,
        }
    }

    fn protect_rtp(
        &self,
        packet: &[u8],
        packet_index: u64,
        mki: Option<&[u8]>,
    ) -> Result<Vec<u8>, SrtpError> {
        let parsed = RtpHeader::parse(packet).ok_or(SrtpError::InvalidRtpPacket)?;
        let rollover_counter =
            u32::try_from(packet_index >> 16).map_err(|_| SrtpError::PacketIndexExhausted)?;
        if let FlowCipher::AuthenticatedEncryption { cipher, salt } = &self.rtp.cipher {
            let initialization_vector = rtp_initialization_vector(
                *salt,
                parsed.header.ssrc,
                rollover_counter,
                parsed.header.sequence,
            );
            let mut output = packet[..parsed.payload_offset].to_vec();
            let ciphertext = cipher.encrypt(
                initialization_vector,
                &packet[..parsed.payload_offset],
                &packet[parsed.payload_offset..],
            )?;
            output.extend_from_slice(&ciphertext);
            append_mki(&mut output, mki);
            return Ok(output);
        }

        let mut protected = packet.to_vec();
        self.rtp.apply_cipher(
            &mut protected[parsed.payload_offset..],
            parsed.header.ssrc,
            packet_index,
        )?;
        // RFC 3711 section 4.2: M is the authenticated portion followed by the
        // rollover counter, and the MKI is outside it.
        let counter_bytes = rollover_counter.to_be_bytes();
        let tag = self.rtp.authentication_tag(&[&protected, &counter_bytes]);
        append_mki(&mut protected, mki);
        if let Some(tag) = tag {
            protected.extend_from_slice(&tag);
        }
        Ok(protected)
    }

    fn unprotect_rtp(
        &self,
        packet: &[u8],
        packet_index: u64,
        mki_length: usize,
    ) -> Result<Vec<u8>, SrtpError> {
        let parsed = RtpHeader::parse_header(packet).ok_or(SrtpError::InvalidRtpPacket)?;
        let rollover_counter =
            u32::try_from(packet_index >> 16).map_err(|_| SrtpError::PacketIndexExhausted)?;
        if let FlowCipher::AuthenticatedEncryption { cipher, salt } = &self.rtp.cipher {
            let body_end = packet
                .len()
                .checked_sub(mki_length)
                .ok_or(SrtpError::InvalidRtpPacket)?;
            if body_end.saturating_sub(parsed.payload_offset) < AUTHENTICATION_TAG_LENGTH {
                return Err(SrtpError::InvalidRtpPacket);
            }
            let initialization_vector = rtp_initialization_vector(
                *salt,
                parsed.header.ssrc,
                rollover_counter,
                parsed.header.sequence,
            );
            let plaintext = cipher.decrypt(
                initialization_vector,
                &packet[..parsed.payload_offset],
                &packet[parsed.payload_offset..body_end],
            )?;
            let mut output = packet[..parsed.payload_offset].to_vec();
            output.extend_from_slice(&plaintext);
            RtpHeader::parse(&output).ok_or(SrtpError::InvalidRtpPacket)?;
            return Ok(output);
        }

        let tag_length = self.rtp.authentication.tag_length();
        let body_end = packet
            .len()
            .checked_sub(tag_length + mki_length)
            .filter(|end| *end >= parsed.payload_offset)
            .ok_or(SrtpError::InvalidRtpPacket)?;
        let counter_bytes = rollover_counter.to_be_bytes();
        self.rtp.verify_authentication_tag(
            &[&packet[..body_end], &counter_bytes],
            &packet[packet.len() - tag_length..],
        )?;
        let mut output = packet[..body_end].to_vec();
        self.rtp.apply_cipher(
            &mut output[parsed.payload_offset..],
            parsed.header.ssrc,
            packet_index,
        )?;
        RtpHeader::parse(&output).ok_or(SrtpError::InvalidRtpPacket)?;
        Ok(output)
    }

    fn protect_rtcp(
        &self,
        packet: &[u8],
        srtcp_index: u32,
        mode: RtcpProtectionMode,
        mki: Option<&[u8]>,
    ) -> Result<Vec<u8>, SrtpError> {
        let synchronization_source = validate_rtcp_packet(packet)?;
        let encryption_flag = matches!(mode, RtcpProtectionMode::Encrypt) && self.rtcp.encrypts();
        let index_word = srtcp_index
            | if encryption_flag {
                SRTCP_ENCRYPTION_FLAG
            } else {
                0
            };

        if let FlowCipher::AuthenticatedEncryption { cipher, salt } = &self.rtcp.cipher {
            let initialization_vector =
                rtcp_initialization_vector(*salt, synchronization_source, srtcp_index);
            if encryption_flag {
                let mut associated_data =
                    [0_u8; RTCP_AUTHENTICATED_PREFIX_LENGTH + SRTCP_INDEX_LENGTH];
                associated_data[..RTCP_AUTHENTICATED_PREFIX_LENGTH]
                    .copy_from_slice(&packet[..RTCP_AUTHENTICATED_PREFIX_LENGTH]);
                associated_data[RTCP_AUTHENTICATED_PREFIX_LENGTH..]
                    .copy_from_slice(&index_word.to_be_bytes());
                let ciphertext = cipher.encrypt(
                    initialization_vector,
                    &associated_data,
                    &packet[RTCP_AUTHENTICATED_PREFIX_LENGTH..],
                )?;
                let mut output = packet[..RTCP_AUTHENTICATED_PREFIX_LENGTH].to_vec();
                output.extend_from_slice(&ciphertext);
                output.extend_from_slice(&index_word.to_be_bytes());
                append_mki(&mut output, mki);
                return Ok(output);
            }
            let mut associated_data = packet.to_vec();
            associated_data.extend_from_slice(&index_word.to_be_bytes());
            let authentication_tag =
                cipher.encrypt(initialization_vector, &associated_data, &[])?;
            let mut output = packet.to_vec();
            output.extend_from_slice(&authentication_tag);
            output.extend_from_slice(&index_word.to_be_bytes());
            append_mki(&mut output, mki);
            return Ok(output);
        }

        let mut protected = packet.to_vec();
        if encryption_flag {
            self.rtcp.apply_cipher(
                &mut protected[RTCP_AUTHENTICATED_PREFIX_LENGTH..],
                synchronization_source,
                u64::from(srtcp_index),
            )?;
        }
        protected.extend_from_slice(&index_word.to_be_bytes());
        let tag = self.rtcp.authentication_tag(&[&protected]);
        append_mki(&mut protected, mki);
        if let Some(tag) = tag {
            protected.extend_from_slice(&tag);
        }
        Ok(protected)
    }

    /// The index word a protected SRTCP packet carries, read before the packet
    /// is authenticated so the replay check can run first.
    fn srtcp_index(&self, packet: &[u8], mki_length: usize) -> Result<u32, SrtpError> {
        let offset = self
            .rtcp_index_offset(packet.len(), mki_length)
            .ok_or(SrtpError::InvalidRtcpPacket)?;
        let word = packet
            .get(offset..offset + SRTCP_INDEX_LENGTH)
            .and_then(|bytes| bytes.try_into().ok())
            .map(u32::from_be_bytes)
            .ok_or(SrtpError::InvalidRtcpPacket)?;
        Ok(word & MAXIMUM_SRTCP_INDEX)
    }

    /// Where the E-flag and index word start. RFC 7714 puts the MKI after them,
    /// RFC 3711 puts the MKI and the tag after them.
    fn rtcp_index_offset(&self, packet_length: usize, mki_length: usize) -> Option<usize> {
        let trailer = match self.rtcp.cipher {
            FlowCipher::AuthenticatedEncryption { .. } => mki_length + SRTCP_INDEX_LENGTH,
            _ => self.rtcp.authentication.tag_length() + mki_length + SRTCP_INDEX_LENGTH,
        };
        packet_length.checked_sub(trailer)
    }

    fn unprotect_rtcp(
        &self,
        packet: &[u8],
        mki_length: usize,
    ) -> Result<(Vec<u8>, u32), SrtpError> {
        let index_offset = self
            .rtcp_index_offset(packet.len(), mki_length)
            .filter(|offset| *offset >= RTCP_AUTHENTICATED_PREFIX_LENGTH)
            .ok_or(SrtpError::InvalidRtcpPacket)?;
        if packet[0] >> 6 != RTP_VERSION {
            return Err(SrtpError::InvalidRtcpPacket);
        }
        let index_end = index_offset + SRTCP_INDEX_LENGTH;
        let index_word = u32::from_be_bytes(
            packet[index_offset..index_end]
                .try_into()
                .map_err(|_| SrtpError::InvalidRtcpPacket)?,
        );
        let encryption_flag = index_word & SRTCP_ENCRYPTION_FLAG != 0;
        let srtcp_index = index_word & MAXIMUM_SRTCP_INDEX;
        let synchronization_source = u32::from_be_bytes(
            packet[RTCP_SOURCE_RANGE]
                .try_into()
                .map_err(|_| SrtpError::InvalidRtcpPacket)?,
        );

        if let FlowCipher::AuthenticatedEncryption { cipher, salt } = &self.rtcp.cipher {
            let initialization_vector =
                rtcp_initialization_vector(*salt, synchronization_source, srtcp_index);
            let output = if encryption_flag {
                let mut associated_data =
                    [0_u8; RTCP_AUTHENTICATED_PREFIX_LENGTH + SRTCP_INDEX_LENGTH];
                associated_data[..RTCP_AUTHENTICATED_PREFIX_LENGTH]
                    .copy_from_slice(&packet[..RTCP_AUTHENTICATED_PREFIX_LENGTH]);
                associated_data[RTCP_AUTHENTICATED_PREFIX_LENGTH..]
                    .copy_from_slice(&index_word.to_be_bytes());
                let plaintext = cipher.decrypt(
                    initialization_vector,
                    &associated_data,
                    &packet[RTCP_AUTHENTICATED_PREFIX_LENGTH..index_offset],
                )?;
                let mut output = packet[..RTCP_AUTHENTICATED_PREFIX_LENGTH].to_vec();
                output.extend_from_slice(&plaintext);
                output
            } else {
                let tag_offset = index_offset
                    .checked_sub(AUTHENTICATION_TAG_LENGTH)
                    .filter(|offset| *offset >= RTCP_AUTHENTICATED_PREFIX_LENGTH)
                    .ok_or(SrtpError::InvalidRtcpPacket)?;
                let mut associated_data = packet[..tag_offset].to_vec();
                associated_data.extend_from_slice(&index_word.to_be_bytes());
                cipher.decrypt(
                    initialization_vector,
                    &associated_data,
                    &packet[tag_offset..index_offset],
                )?;
                packet[..tag_offset].to_vec()
            };
            validate_rtcp_packet(&output)?;
            return Ok((output, srtcp_index));
        }

        let tag_length = self.rtcp.authentication.tag_length();
        self.rtcp.verify_authentication_tag(
            &[&packet[..index_end]],
            &packet[packet.len() - tag_length..],
        )?;
        let mut output = packet[..index_offset].to_vec();
        if encryption_flag {
            self.rtcp.apply_cipher(
                &mut output[RTCP_AUTHENTICATED_PREFIX_LENGTH..],
                synchronization_source,
                u64::from(srtcp_index),
            )?;
        }
        validate_rtcp_packet(&output)?;
        Ok((output, srtcp_index))
    }
}

fn append_mki(packet: &mut Vec<u8>, mki: Option<&[u8]>) {
    if let Some(mki) = mki {
        packet.extend_from_slice(mki);
    }
}

/// The RFC 3711 section 4.1.1 initial counter block: the session salt shifted
/// left 16 bits, XOR the source at 2^64 and the packet index at 2^16.
fn counter_mode_block(
    salt: &[u8; COUNTER_MODE_MASTER_SALT_LENGTH],
    synchronization_source: u32,
    packet_index: u64,
) -> [u8; AES_BLOCK_LENGTH] {
    let mut block = [0_u8; AES_BLOCK_LENGTH];
    block[..COUNTER_MODE_MASTER_SALT_LENGTH].copy_from_slice(salt);
    for (byte, source_byte) in block[COUNTER_BLOCK_SOURCE_RANGE]
        .iter_mut()
        .zip(synchronization_source.to_be_bytes())
    {
        *byte ^= source_byte;
    }
    let index_bytes = packet_index.to_be_bytes();
    for (byte, index_byte) in block[COUNTER_BLOCK_INDEX_RANGE]
        .iter_mut()
        .zip(&index_bytes[index_bytes.len() - COUNTER_BLOCK_INDEX_RANGE.len()..])
    {
        *byte ^= index_byte;
    }
    block
}

/// XOR the AES counter-mode keystream over `data`. The initial block leaves its
/// last two octets zero, which is where the block counter goes, so RFC 3711's
/// cap of 2^16 blocks for one initial value is what the counter's width allows.
fn apply_counter_mode(
    cipher: &AesBlockCipher,
    initial_block: [u8; AES_BLOCK_LENGTH],
    data: &mut [u8],
) -> Result<(), SrtpError> {
    let mut block = Zeroizing::new(initial_block);
    for (number, chunk) in data.chunks_mut(AES_BLOCK_LENGTH).enumerate() {
        let number = u16::try_from(number).map_err(|_| SrtpError::PacketTooLarge)?;
        block[AES_BLOCK_LENGTH - 2..].copy_from_slice(&number.to_be_bytes());
        let keystream = cipher.encrypt_block(*block);
        for (byte, keystream_byte) in chunk.iter_mut().zip(keystream) {
            *byte ^= keystream_byte;
        }
    }
    Ok(())
}

fn derive_session_value(
    master: &AesBlockCipher,
    master_salt: &[u8; COUNTER_MODE_MASTER_SALT_LENGTH],
    label: u8,
    output: &mut [u8],
) -> Result<(), SrtpError> {
    let mut input = Zeroizing::new([0_u8; AES_BLOCK_LENGTH]);
    input[..COUNTER_MODE_MASTER_SALT_LENGTH].copy_from_slice(master_salt);
    input[7] ^= label;
    output.fill(0);
    apply_counter_mode(master, *input, output)
}

fn rtp_initialization_vector(
    salt: [u8; 12],
    synchronization_source: u32,
    rollover_counter: u32,
    sequence: u16,
) -> [u8; 12] {
    let mut packet_vector = [0_u8; 12];
    packet_vector[2..6].copy_from_slice(&synchronization_source.to_be_bytes());
    packet_vector[6..10].copy_from_slice(&rollover_counter.to_be_bytes());
    packet_vector[10..12].copy_from_slice(&sequence.to_be_bytes());
    xor_initialization_vector(salt, packet_vector)
}

fn rtcp_initialization_vector(
    salt: [u8; 12],
    synchronization_source: u32,
    srtcp_index: u32,
) -> [u8; 12] {
    let mut packet_vector = [0_u8; 12];
    packet_vector[2..6].copy_from_slice(&synchronization_source.to_be_bytes());
    packet_vector[8..12].copy_from_slice(&(srtcp_index & MAXIMUM_SRTCP_INDEX).to_be_bytes());
    xor_initialization_vector(salt, packet_vector)
}

fn xor_initialization_vector(salt: [u8; 12], packet_vector: [u8; 12]) -> [u8; 12] {
    let mut initialization_vector = salt;
    for (byte, packet_byte) in initialization_vector.iter_mut().zip(packet_vector) {
        *byte ^= packet_byte;
    }
    initialization_vector
}

fn validate_rtcp_packet(packet: &[u8]) -> Result<u32, SrtpError> {
    if packet.len() < RTCP_AUTHENTICATED_PREFIX_LENGTH || !packet.len().is_multiple_of(4) {
        return Err(SrtpError::InvalidRtcpPacket);
    }
    if packet[0] >> 6 != RTP_VERSION {
        return Err(SrtpError::InvalidRtcpPacket);
    }
    let synchronization_source = u32::from_be_bytes(
        packet[RTCP_SOURCE_RANGE]
            .try_into()
            .map_err(|_| SrtpError::InvalidRtcpPacket)?,
    );
    Ok(synchronization_source)
}

/// Whether `packets` is a replay window an element may ask for.
pub fn validate_replay_window(packets: usize) -> Result<(), SrtpError> {
    if !(MINIMUM_REPLAY_WINDOW..=MAXIMUM_REPLAY_WINDOW).contains(&packets) {
        return Err(SrtpError::InvalidReplayWindow { size: packets });
    }
    Ok(())
}

#[derive(Debug)]
struct PacketIndexState {
    initial_rollover_counter: u32,
    highest_index: Option<u64>,
    /// One bit per index within the window, bit 0 of word 0 being
    /// `highest_index` itself. Sized once, never grown per packet.
    replay_bitmap: Vec<u64>,
    window: usize,
}

impl PacketIndexState {
    fn new(initial_rollover_counter: u32, window: usize) -> Result<Self, SrtpError> {
        validate_replay_window(window)?;
        Ok(Self {
            initial_rollover_counter,
            highest_index: None,
            replay_bitmap: alloc::vec![0; window.div_ceil(REPLAY_BITMAP_WORD_BITS)],
            window,
        })
    }

    /// Resize the window, dropping what it recorded so far.
    fn resize(&mut self, window: usize) -> Result<(), SrtpError> {
        *self = Self::new(self.initial_rollover_counter, window)?;
        Ok(())
    }

    /// The rollover counter the next packet is judged against.
    fn rollover_counter(&self) -> u32 {
        match self.highest_index {
            Some(index) => u32::try_from(index >> 16).unwrap_or(u32::MAX),
            None => self.initial_rollover_counter,
        }
    }

    fn is_set(&self, distance: usize) -> bool {
        let word = self.replay_bitmap[distance / REPLAY_BITMAP_WORD_BITS];
        word & (1_u64 << (distance % REPLAY_BITMAP_WORD_BITS)) != 0
    }

    fn set(&mut self, distance: usize) {
        self.replay_bitmap[distance / REPLAY_BITMAP_WORD_BITS] |=
            1_u64 << (distance % REPLAY_BITMAP_WORD_BITS);
    }

    /// Move every recorded index `distance` bits further back, dropping the ones
    /// that fall out of the window.
    fn shift(&mut self, distance: u64) {
        let word_bits = REPLAY_BITMAP_WORD_BITS as u64;
        let words = usize::try_from(distance / word_bits).unwrap_or(usize::MAX);
        if words >= self.replay_bitmap.len() {
            self.replay_bitmap.fill(0);
            return;
        }
        let bits = (distance % word_bits) as u32;
        for target in (0..self.replay_bitmap.len()).rev() {
            let carried = match target.checked_sub(words + 1) {
                Some(source) if bits > 0 => {
                    self.replay_bitmap[source] >> (REPLAY_BITMAP_WORD_BITS as u32 - bits)
                }
                _ => 0,
            };
            self.replay_bitmap[target] = match target.checked_sub(words) {
                Some(source) => (self.replay_bitmap[source] << bits) | carried,
                None => 0,
            };
        }
    }

    fn estimate_rtp_index(&self, sequence: u16) -> Result<u64, SrtpError> {
        let Some(highest_index) = self.highest_index else {
            return Ok((u64::from(self.initial_rollover_counter) << 16) | u64::from(sequence));
        };
        let highest_rollover_counter = highest_index >> 16;
        let highest_sequence = highest_index as u16;
        // RFC 3711 appendix A in signed arithmetic: only a sequence across the wrap moves the counter.
        let half_range = i32::from(SEQUENCE_HALF_RANGE);
        let sequence_offset = i32::from(sequence) - i32::from(highest_sequence);
        let before_the_wrap =
            highest_sequence < SEQUENCE_HALF_RANGE && sequence_offset > half_range;
        let after_the_wrap =
            highest_sequence >= SEQUENCE_HALF_RANGE && sequence_offset < -half_range;
        let rollover_counter = if before_the_wrap {
            highest_rollover_counter
                .checked_sub(1)
                .ok_or(SrtpError::PacketTooOld)?
        } else if after_the_wrap {
            highest_rollover_counter
                .checked_add(1)
                .ok_or(SrtpError::PacketIndexExhausted)?
        } else {
            highest_rollover_counter
        };
        let packet_index = (rollover_counter << 16) | u64::from(sequence);
        if packet_index > MAXIMUM_RTP_PACKET_INDEX {
            return Err(SrtpError::PacketIndexExhausted);
        }
        Ok(packet_index)
    }

    fn check(&self, packet_index: u64) -> Result<(), SrtpError> {
        let Some(highest_index) = self.highest_index else {
            return Ok(());
        };
        if packet_index > highest_index {
            return Ok(());
        }
        let distance =
            usize::try_from(highest_index - packet_index).map_err(|_| SrtpError::PacketTooOld)?;
        if distance >= self.window {
            return Err(SrtpError::PacketTooOld);
        }
        if self.is_set(distance) {
            return Err(SrtpError::RepeatedPacket);
        }
        Ok(())
    }

    fn accept(&mut self, packet_index: u64) {
        match self.highest_index {
            None => {
                self.set(0);
                self.highest_index = Some(packet_index);
            }
            Some(highest_index) if packet_index > highest_index => {
                self.shift(packet_index - highest_index);
                self.set(0);
                self.highest_index = Some(packet_index);
            }
            Some(highest_index) => {
                let distance = usize::try_from(highest_index - packet_index).unwrap_or(usize::MAX);
                if distance < self.window {
                    self.set(distance);
                }
            }
        }
    }
}

/// Protects outbound RTP and RTCP packets for one synchronization source.
#[derive(Debug)]
pub struct SrtpSender {
    keys: SessionKeys,
    mki: Option<Vec<u8>>,
    synchronization_source: u32,
    rtp_indices: PacketIndexState,
    next_srtcp_index: Option<u32>,
    soft_limits: SrtpSoftLimits,
    allow_repeat_transmission: bool,
    srtp_key_invocations: u64,
    srtcp_key_invocations: u32,
}

impl SrtpSender {
    pub fn new(
        policy: SrtpPolicy,
        master_key: &[u8],
        master_salt: &[u8],
        synchronization_source: u32,
    ) -> Result<Self, SrtpError> {
        Self::new_with_soft_limits(
            policy,
            master_key,
            master_salt,
            synchronization_source,
            SrtpSoftLimits::default(),
        )
    }

    pub fn new_with_soft_limits(
        policy: SrtpPolicy,
        master_key: &[u8],
        master_salt: &[u8],
        synchronization_source: u32,
        soft_limits: SrtpSoftLimits,
    ) -> Result<Self, SrtpError> {
        validate_soft_limits(soft_limits)?;
        Ok(Self {
            keys: SessionKeys::derive(policy, master_key, master_salt)?,
            mki: None,
            synchronization_source,
            rtp_indices: PacketIndexState::new(0, DEFAULT_REPLAY_WINDOW)?,
            next_srtcp_index: Some(0),
            soft_limits,
            allow_repeat_transmission: false,
            srtp_key_invocations: 0,
            srtcp_key_invocations: 0,
        })
    }

    /// Size the window the repeat check reads, 64..=32768 packets. Resizing
    /// clears what it recorded, so set it before the first packet.
    pub fn set_replay_window(&mut self, packets: usize) -> Result<(), SrtpError> {
        self.rtp_indices.resize(packets)
    }

    /// Protect a packet whose index was already protected instead of rejecting
    /// it. Every repeat has to carry the same RTP payload: the keystream
    /// repeats with the index, and two payloads under one keystream break the
    /// cipher.
    pub fn set_repeat_transmission(&mut self, allow: bool) {
        self.allow_repeat_transmission = allow;
    }

    /// Put this Master Key Identifier in every packet protected from here on,
    /// after the tag under the RFC 7714 profiles and before it under the RFC
    /// 3711 ones. `None` removes it.
    pub fn set_mki(&mut self, mki: Option<&[u8]>) -> Result<(), SrtpError> {
        self.mki = mki.map(validated_mki).transpose()?;
        Ok(())
    }

    pub fn mki(&self) -> Option<&[u8]> {
        self.mki.as_deref()
    }

    pub fn replace_key(
        &mut self,
        policy: SrtpPolicy,
        master_key: &[u8],
        master_salt: &[u8],
    ) -> Result<(), SrtpError> {
        self.keys = SessionKeys::derive(policy, master_key, master_salt)?;
        self.srtp_key_invocations = 0;
        self.srtcp_key_invocations = 0;
        Ok(())
    }

    pub fn set_soft_limits(&mut self, soft_limits: SrtpSoftLimits) -> Result<(), SrtpError> {
        validate_soft_limits(soft_limits)?;
        self.soft_limits = soft_limits;
        Ok(())
    }

    pub fn key_usage(&self) -> SrtpKeyUsage {
        SrtpKeyUsage {
            srtp: key_usage(self.srtp_key_invocations, self.soft_limits.srtp_packets),
            srtcp: key_usage(
                u64::from(self.srtcp_key_invocations),
                u64::from(self.soft_limits.srtcp_packets),
            ),
        }
    }

    pub fn protect_rtp(&mut self, packet: &[u8]) -> Result<Vec<u8>, SrtpError> {
        if self.srtp_key_invocations >= MAXIMUM_SRTP_KEY_INVOCATIONS {
            return Err(SrtpError::KeyLifetimeExhausted);
        }
        let parsed = RtpHeader::parse(packet).ok_or(SrtpError::InvalidRtpPacket)?;
        self.check_synchronization_source(parsed.header.ssrc)?;
        let packet_index = self
            .rtp_indices
            .estimate_rtp_index(parsed.header.sequence)?;
        self.check_repeat(packet_index)?;
        let protected = self
            .keys
            .protect_rtp(packet, packet_index, self.mki.as_deref())?;
        self.rtp_indices.accept(packet_index);
        self.srtp_key_invocations += 1;
        Ok(protected)
    }

    /// The replay check against the sender's own indices. A repeat passes only
    /// when repeated transmission is allowed; an index that already fell out of
    /// the window never does, the way libsrtp's `allow_repeat_tx` works.
    fn check_repeat(&self, packet_index: u64) -> Result<(), SrtpError> {
        match self.rtp_indices.check(packet_index) {
            Err(SrtpError::RepeatedPacket) if self.allow_repeat_transmission => Ok(()),
            other => other,
        }
    }

    pub fn protect_rtcp(
        &mut self,
        packet: &[u8],
        mode: RtcpProtectionMode,
    ) -> Result<Vec<u8>, SrtpError> {
        if self.srtcp_key_invocations >= MAXIMUM_SRTCP_KEY_INVOCATIONS {
            return Err(SrtpError::KeyLifetimeExhausted);
        }
        let actual_source = validate_rtcp_packet(packet)?;
        self.check_synchronization_source(actual_source)?;
        let packet_index = self
            .next_srtcp_index
            .ok_or(SrtpError::PacketIndexExhausted)?;
        let protected = self
            .keys
            .protect_rtcp(packet, packet_index, mode, self.mki.as_deref())?;
        self.next_srtcp_index = packet_index
            .checked_add(1)
            .filter(|index| *index <= MAXIMUM_SRTCP_INDEX);
        self.srtcp_key_invocations += 1;
        Ok(protected)
    }

    fn check_synchronization_source(&self, actual: u32) -> Result<(), SrtpError> {
        if actual != self.synchronization_source {
            return Err(SrtpError::WrongSynchronizationSource {
                expected: self.synchronization_source,
                actual,
            });
        }
        Ok(())
    }
}

fn validate_soft_limits(soft_limits: SrtpSoftLimits) -> Result<(), SrtpError> {
    if !soft_limits.is_valid() {
        return Err(SrtpError::InvalidSoftLimit);
    }
    Ok(())
}

fn key_usage(invocations: u64, soft_limit: u64) -> KeyUsage {
    if invocations >= soft_limit {
        KeyUsage::SoftLimitReached
    } else {
        KeyUsage::Normal
    }
}

/// One master key of a receive context, with the MKI that selects it.
#[derive(Debug)]
struct KeyedSession {
    keys: SessionKeys,
    mki: Option<Vec<u8>>,
}

/// The MKI length every key of one context shares, 0 when none carries one.
/// RFC 3711 fixes the length for a session, so a mixture is refused, as is a
/// second key with no MKI to tell it apart by.
fn shared_mki_length(keys: &[SrtpKeyingMaterial]) -> Result<usize, SrtpError> {
    let first = keys.first().ok_or(SrtpError::MissingKey)?;
    let length = first.mki().map_or(0, <[u8]>::len);
    let mixed = keys
        .iter()
        .any(|key| key.mki().map_or(0, <[u8]>::len) != length);
    if mixed || (length == 0 && keys.len() > 1) {
        return Err(SrtpError::InconsistentMki);
    }
    Ok(length)
}

/// Authenticates and decrypts inbound RTP and RTCP packets for one source.
#[derive(Debug)]
pub struct SrtpReceiver {
    sessions: Vec<KeyedSession>,
    mki_length: usize,
    synchronization_source: u32,
    rtp_indices: PacketIndexState,
    srtcp_indices: PacketIndexState,
}

impl SrtpReceiver {
    pub fn new(
        policy: SrtpPolicy,
        master_key: &[u8],
        master_salt: &[u8],
        synchronization_source: u32,
    ) -> Result<Self, SrtpError> {
        Self::new_with_rollover_counter(policy, master_key, master_salt, synchronization_source, 0)
    }

    pub fn new_with_rollover_counter(
        policy: SrtpPolicy,
        master_key: &[u8],
        master_salt: &[u8],
        synchronization_source: u32,
        initial_rollover_counter: u32,
    ) -> Result<Self, SrtpError> {
        let material =
            SrtpKeyingMaterial::new(policy, master_key, master_salt, initial_rollover_counter)?;
        Self::new_with_keys(synchronization_source, Vec::from([material]))
    }

    /// A context keyed by one key, or by several the sender selects between
    /// through the MKI it appends. Every key has to carry an MKI of the same
    /// length, or the single key may carry none.
    pub fn new_with_keys(
        synchronization_source: u32,
        keys: Vec<SrtpKeyingMaterial>,
    ) -> Result<Self, SrtpError> {
        Self::new_with_keys_and_window(synchronization_source, keys, DEFAULT_REPLAY_WINDOW)
    }

    fn new_with_keys_and_window(
        synchronization_source: u32,
        keys: Vec<SrtpKeyingMaterial>,
        replay_window: usize,
    ) -> Result<Self, SrtpError> {
        let mki_length = shared_mki_length(&keys)?;
        let initial_rollover_counter = keys
            .first()
            .ok_or(SrtpError::MissingKey)?
            .initial_rollover_counter;
        let sessions = keys
            .iter()
            .map(|material| {
                Ok(KeyedSession {
                    keys: material.derive_keys()?,
                    mki: material.mki().map(<[u8]>::to_vec),
                })
            })
            .collect::<Result<Vec<KeyedSession>, SrtpError>>()?;
        Ok(Self {
            sessions,
            mki_length,
            synchronization_source,
            rtp_indices: PacketIndexState::new(initial_rollover_counter, replay_window)?,
            srtcp_indices: PacketIndexState::new(0, replay_window)?,
        })
    }

    /// Replace every key of this context with one new key, which carries no MKI.
    pub fn replace_key(
        &mut self,
        policy: SrtpPolicy,
        master_key: &[u8],
        master_salt: &[u8],
    ) -> Result<(), SrtpError> {
        self.sessions = Vec::from([KeyedSession {
            keys: SessionKeys::derive(policy, master_key, master_salt)?,
            mki: None,
        }]);
        self.mki_length = 0;
        Ok(())
    }

    /// Size both replay windows, dropping what they recorded so far.
    pub fn set_replay_window(&mut self, packets: usize) -> Result<(), SrtpError> {
        self.rtp_indices.resize(packets)?;
        self.srtcp_indices.resize(packets)
    }

    /// The rollover counter the next RTP packet is judged against.
    pub fn rollover_counter(&self) -> u32 {
        self.rtp_indices.rollover_counter()
    }

    pub fn unprotect_rtp(&mut self, packet: &[u8]) -> Result<Vec<u8>, SrtpError> {
        let session = self.session_for(packet, SrtpFlow::Rtp)?;
        let parsed = RtpHeader::parse_header(packet).ok_or(SrtpError::InvalidRtpPacket)?;
        self.check_synchronization_source(parsed.header.ssrc)?;
        let packet_index = self
            .rtp_indices
            .estimate_rtp_index(parsed.header.sequence)?;
        self.rtp_indices.check(packet_index)?;
        let unprotected =
            self.sessions[session]
                .keys
                .unprotect_rtp(packet, packet_index, self.mki_length)?;
        self.rtp_indices.accept(packet_index);
        Ok(unprotected)
    }

    pub fn unprotect_rtcp(&mut self, packet: &[u8]) -> Result<Vec<u8>, SrtpError> {
        let session = self.session_for(packet, SrtpFlow::Rtcp)?;
        let actual_source = packet
            .get(RTCP_SOURCE_RANGE)
            .and_then(|bytes| bytes.try_into().ok())
            .map(u32::from_be_bytes)
            .ok_or(SrtpError::InvalidRtcpPacket)?;
        self.check_synchronization_source(actual_source)?;
        let packet_index = self.sessions[session]
            .keys
            .srtcp_index(packet, self.mki_length)?;
        self.srtcp_indices.check(u64::from(packet_index))?;
        let (unprotected, authenticated_index) = self.sessions[session]
            .keys
            .unprotect_rtcp(packet, self.mki_length)?;
        debug_assert_eq!(packet_index, authenticated_index);
        self.srtcp_indices.accept(u64::from(packet_index));
        Ok(unprotected)
    }

    /// The session a packet's MKI names. Where the MKI sits depends on the
    /// key's own policy, so each candidate is compared at its own offset.
    fn session_for(&self, packet: &[u8], flow: SrtpFlow) -> Result<usize, SrtpError> {
        if self.mki_length == 0 {
            if self.sessions.is_empty() {
                return Err(SrtpError::MissingKey);
            }
            return Ok(0);
        }
        self.sessions
            .iter()
            .position(|session| {
                session
                    .keys
                    .flow_keys(flow)
                    .mki_range(packet.len(), self.mki_length)
                    .and_then(|range| packet.get(range))
                    == session.mki.as_deref()
            })
            .ok_or(SrtpError::MissingKey)
    }

    fn check_synchronization_source(&self, actual: u32) -> Result<(), SrtpError> {
        if actual != self.synchronization_source {
            return Err(SrtpError::WrongSynchronizationSource {
                expected: self.synchronization_source,
                actual,
            });
        }
        Ok(())
    }
}

/// One receive context's identity and progress, the pair gst's `stats`
/// `streams` list carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SrtpStreamStats {
    pub synchronization_source: u32,
    pub rollover_counter: u32,
}

/// Selects an independent receive context for each synchronization source.
#[derive(Debug)]
pub struct SrtpReceiverSet<P> {
    key_provider: P,
    receivers: Vec<SrtpReceiver>,
    replay_window: usize,
}

impl<P> SrtpReceiverSet<P>
where
    P: SrtpKeyProvider,
{
    pub fn new(key_provider: P) -> Self {
        Self {
            key_provider,
            receivers: Vec::new(),
            replay_window: DEFAULT_REPLAY_WINDOW,
        }
    }

    /// The window every context created from here on gets. The ones that
    /// already exist keep theirs, so their replay history survives.
    pub fn set_replay_window(&mut self, packets: usize) -> Result<(), SrtpError> {
        validate_replay_window(packets)?;
        self.replay_window = packets;
        Ok(())
    }

    pub fn replay_window(&self) -> usize {
        self.replay_window
    }

    /// One entry per context that has seen a packet.
    pub fn stream_statistics(&self) -> Vec<SrtpStreamStats> {
        self.receivers
            .iter()
            .map(|receiver| SrtpStreamStats {
                synchronization_source: receiver.synchronization_source,
                rollover_counter: receiver.rollover_counter(),
            })
            .collect()
    }

    pub fn unprotect_rtp(&mut self, packet: &[u8]) -> Result<Vec<u8>, SrtpError> {
        let parsed = RtpHeader::parse_header(packet).ok_or(SrtpError::InvalidRtpPacket)?;
        let receiver = self.receiver_for(parsed.header.ssrc)?;
        receiver.unprotect_rtp(packet)
    }

    pub fn unprotect_rtcp(&mut self, packet: &[u8]) -> Result<Vec<u8>, SrtpError> {
        let synchronization_source = packet
            .get(RTCP_SOURCE_RANGE)
            .and_then(|bytes| bytes.try_into().ok())
            .map(u32::from_be_bytes)
            .ok_or(SrtpError::InvalidRtcpPacket)?;
        let receiver = self.receiver_for(synchronization_source)?;
        receiver.unprotect_rtcp(packet)
    }

    pub fn replace_key(
        &mut self,
        synchronization_source: u32,
        policy: SrtpPolicy,
        master_key: &[u8],
        master_salt: &[u8],
    ) -> Result<(), SrtpError> {
        let receiver = self
            .receivers
            .iter_mut()
            .find(|receiver| receiver.synchronization_source == synchronization_source)
            .ok_or(SrtpError::MissingKey)?;
        receiver.replace_key(policy, master_key, master_salt)
    }

    /// The sources that already have a context, so a caller replacing the key
    /// can re-key every one of them without disturbing their packet indices.
    pub fn synchronization_sources(&self) -> Vec<u32> {
        self.receivers
            .iter()
            .map(|receiver| receiver.synchronization_source)
            .collect()
    }

    pub fn key_provider(&self) -> &P {
        &self.key_provider
    }

    /// The provider itself, so a caller can hand it a new key. Contexts already
    /// built keep the key they were given; re-key those through
    /// [`replace_key`](Self::replace_key).
    pub fn key_provider_mut(&mut self) -> &mut P {
        &mut self.key_provider
    }

    pub fn remove_context(&mut self, synchronization_source: u32) -> bool {
        let Some(position) = self
            .receivers
            .iter()
            .position(|receiver| receiver.synchronization_source == synchronization_source)
        else {
            return false;
        };
        self.receivers.swap_remove(position);
        true
    }

    fn receiver_for(
        &mut self,
        synchronization_source: u32,
    ) -> Result<&mut SrtpReceiver, SrtpError> {
        if let Some(position) = self
            .receivers
            .iter()
            .position(|receiver| receiver.synchronization_source == synchronization_source)
        {
            return Ok(&mut self.receivers[position]);
        }
        let keys = self.key_provider.keys_for(synchronization_source);
        if keys.is_empty() {
            return Err(SrtpError::MissingKey);
        }
        let receiver = SrtpReceiver::new_with_keys_and_window(
            synchronization_source,
            keys,
            self.replay_window,
        )?;
        self.receivers.push(receiver);
        self.receivers.last_mut().ok_or(SrtpError::MissingKey)
    }
}

/// The protocol one element instance carries. `srtpenc` and `srtpdec` each
/// handle a single flow, fixed by the caps the link settled on, the way
/// GStreamer's `srtpenc` splits its RTP and RTCP pads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SrtpFlow {
    Rtp,
    Rtcp,
}

impl SrtpFlow {
    /// The caps encoding this flow carries before protection.
    pub const fn plain_encoding(self) -> ByteStreamEncoding {
        match self {
            Self::Rtp => ByteStreamEncoding::Rtp,
            Self::Rtcp => ByteStreamEncoding::Rtcp,
        }
    }

    /// The caps encoding this flow carries after protection.
    pub const fn protected_encoding(self) -> ByteStreamEncoding {
        match self {
            Self::Rtp => ByteStreamEncoding::Srtp,
            Self::Rtcp => ByteStreamEncoding::Srtcp,
        }
    }

    /// The flow `caps` names before protection, `None` for anything else.
    pub fn from_plain_caps(caps: &Caps) -> Option<Self> {
        Self::of_caps(caps, Self::plain_encoding)
    }

    /// The flow `caps` names after protection, `None` for anything else.
    pub fn from_protected_caps(caps: &Caps) -> Option<Self> {
        Self::of_caps(caps, Self::protected_encoding)
    }

    fn of_caps(caps: &Caps, side: fn(Self) -> ByteStreamEncoding) -> Option<Self> {
        let Caps::ByteStream { encoding } = caps else {
            return None;
        };
        [Self::Rtp, Self::Rtcp]
            .into_iter()
            .find(|flow| side(*flow) == *encoding)
    }
}

/// The source an RTP or RTCP packet names, `None` when it is not one. The value
/// a sender context is built around.
pub fn synchronization_source(packet: &[u8], flow: SrtpFlow) -> Option<u32> {
    match flow {
        SrtpFlow::Rtp => Some(RtpHeader::parse_header(packet)?.header.ssrc),
        SrtpFlow::Rtcp => packet
            .get(RTCP_SOURCE_RANGE)
            .and_then(|bytes| bytes.try_into().ok())
            .map(u32::from_be_bytes),
    }
}

/// Protect one packet of `flow`, the RTCP mode ignored on the RTP flow.
pub fn protect_for_flow(
    sender: &mut SrtpSender,
    flow: SrtpFlow,
    packet: &[u8],
    mode: RtcpProtectionMode,
) -> Result<Vec<u8>, SrtpError> {
    match flow {
        SrtpFlow::Rtp => sender.protect_rtp(packet),
        SrtpFlow::Rtcp => sender.protect_rtcp(packet, mode),
    }
}

/// Whether a send failure ends the stream. The RFC 3711 packet-index and key
/// lifetimes may not be exceeded, so a sender that reached one stops rather than
/// protecting a second packet under a keystream it already used.
pub fn is_fatal_send_error(error: SrtpError) -> bool {
    matches!(
        error,
        SrtpError::KeyLifetimeExhausted | SrtpError::PacketIndexExhausted
    )
}

/// Both flows' caps on one side, for a pad template or a solver constraint.
pub fn flow_caps(side: fn(SrtpFlow) -> ByteStreamEncoding) -> Vec<Caps> {
    [SrtpFlow::Rtp, SrtpFlow::Rtcp]
        .into_iter()
        .map(|flow| Caps::ByteStream {
            encoding: side(flow),
        })
        .collect()
}

/// Forward every packet an SRTP element passes through untouched, announcing
/// `output_caps` in place of an incoming `CapsChanged`, and hand back the frame
/// the element has to protect or unprotect itself.
///
/// `Eos` is consumed, not forwarded: the runner emits the sentinel once
/// `process` returns.
pub async fn forward_until_data_frame(
    packet: PipelinePacket,
    out: &mut dyn OutputSink,
    output_caps: Caps,
) -> Result<Option<Frame>, G2gError> {
    match packet {
        PipelinePacket::DataFrame(frame) => return Ok(Some(frame)),
        PipelinePacket::CapsChanged(_) => {
            out.push(PipelinePacket::CapsChanged(output_caps)).await?;
        }
        PipelinePacket::Flush => {
            out.push(PipelinePacket::Flush).await?;
        }
        PipelinePacket::Segment(segment) => {
            out.push(PipelinePacket::Segment(segment)).await?;
        }
        PipelinePacket::Eos => {}
        other => {
            out.push(other).await?;
        }
    }
    Ok(None)
}

/// Send `bytes` downstream in `frame`'s place, keeping its timing, sequence and
/// metadata.
pub async fn push_frame_bytes(
    mut frame: Frame,
    bytes: Vec<u8>,
    out: &mut dyn OutputSink,
) -> Result<(), G2gError> {
    frame.domain = MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice()));
    out.push(PipelinePacket::DataFrame(frame)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SALT: [u8; AEAD_MASTER_SALT_LENGTH] = *b"Quid pro quo";
    /// A packet-layer leg that carries no Master Key Identifier.
    const NO_MKI: Option<&[u8]> = None;
    const NO_MKI_LENGTH: usize = 0;
    const AES_128_GCM: SrtpPolicy = SrtpPolicy {
        cipher: SrtpCipher::Aes128Gcm,
        authentication: SrtpAuthentication::Null,
    };
    const AES_256_GCM: SrtpPolicy = SrtpPolicy {
        cipher: SrtpCipher::Aes256Gcm,
        authentication: SrtpAuthentication::Null,
    };
    const AES_128_COUNTER_MODE_80: SrtpPolicy = SrtpPolicy {
        cipher: SrtpCipher::Aes128CounterMode,
        authentication: SrtpAuthentication::HmacSha1Tag80,
    };
    const AES_256_COUNTER_MODE_80: SrtpPolicy = SrtpPolicy {
        cipher: SrtpCipher::Aes256CounterMode,
        authentication: SrtpAuthentication::HmacSha1Tag80,
    };
    const AES_128_COUNTER_MODE_32: SrtpPolicy = SrtpPolicy {
        cipher: SrtpCipher::Aes128CounterMode,
        authentication: SrtpAuthentication::HmacSha1Tag32,
    };
    const NULL_CIPHER_80: SrtpPolicy = SrtpPolicy {
        cipher: SrtpCipher::Null,
        authentication: SrtpAuthentication::HmacSha1Tag80,
    };

    fn hexadecimal(input: &str) -> Vec<u8> {
        let digits: Vec<u8> = input
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect();
        let (pairs, remainder) = digits.as_chunks::<2>();
        assert!(remainder.is_empty());
        pairs
            .iter()
            .map(|pair| {
                let high = hexadecimal_digit(pair[0]).expect("hex digit");
                let low = hexadecimal_digit(pair[1]).expect("hex digit");
                high << 4 | low
            })
            .collect()
    }

    fn rfc_rtp_packet() -> Vec<u8> {
        hexadecimal(
            "8040f17b 8041f8d3 5501a0b2 47616c6c
             69612065 7374206f 6d6e6973 20646976
             69736120 696e2070 61727465 73207472 6573",
        )
    }

    fn rfc_rtcp_packet() -> Vec<u8> {
        hexadecimal(
            "81c8000d 4d617273 4e545031 4e545032
             52545020 0000042a 0000e930 4c756e61
             deadbeef deadbeef deadbeef deadbeef deadbeef",
        )
    }

    /// The keys and salts one policy takes, for a round-trip leg.
    fn test_key_material(policy: SrtpPolicy, fill: u8) -> (Vec<u8>, Vec<u8>) {
        let key_length = policy.cipher.master_key_lengths()[0];
        (
            alloc::vec![fill; key_length],
            alloc::vec![fill ^ 0xff; policy.cipher.master_salt_length()],
        )
    }

    #[test]
    fn hexadecimal_key_material_carries_its_profile() {
        let key_128 = "000102030405060708090a0b0c0d0e0f517569642070726f2071756f";
        let parsed = SrtpMasterKey::from_hexadecimal(key_128).expect("28 bytes is AES-128-GCM");
        assert_eq!(parsed.policy(), AES_128_GCM);
        assert_eq!(parsed.master_key(), &hexadecimal(&key_128[..32])[..]);
        assert_eq!(parsed.master_salt(), &TEST_SALT[..]);

        let key_256 = "00112233445566778899aabbccddeeff\
                       00112233445566778899aabbccddeeff\
                       517569642070726f2071756f";
        let parsed = SrtpMasterKey::from_hexadecimal(key_256).expect("44 bytes is AES-256-GCM");
        assert_eq!(parsed.policy(), AES_256_GCM);
        assert_eq!(parsed.master_salt(), &TEST_SALT[..]);

        assert!(SrtpMasterKey::from_hexadecimal(&key_128[..30]).is_none());
        assert!(SrtpMasterKey::from_hexadecimal(&key_128[..55]).is_none());
        assert!(SrtpMasterKey::from_hexadecimal(&key_128.replace("0a", "0z")).is_none());
    }

    #[test]
    fn flows_map_each_caps_pair_to_one_protocol() {
        for flow in [SrtpFlow::Rtp, SrtpFlow::Rtcp] {
            let plain = Caps::ByteStream {
                encoding: flow.plain_encoding(),
            };
            let protected = Caps::ByteStream {
                encoding: flow.protected_encoding(),
            };
            assert_eq!(SrtpFlow::from_plain_caps(&plain), Some(flow));
            assert_eq!(SrtpFlow::from_protected_caps(&protected), Some(flow));
            assert_eq!(SrtpFlow::from_plain_caps(&protected), None);
            assert_eq!(SrtpFlow::from_protected_caps(&plain), None);
        }
    }

    /// The cipher key, cipher salt and auth key of RFC 3711 appendix B.3 and
    /// RFC 6188 section 7.2, the three values one master key derives per flow.
    #[test]
    fn aes_cm_key_derivation_matches_rfc_3711_and_rfc_6188() {
        let derive = |master_key: &[u8], master_salt: &str, label: u8, length: usize| {
            let salt: [u8; COUNTER_MODE_MASTER_SALT_LENGTH] = hexadecimal(master_salt)
                .try_into()
                .expect("a 14-byte master salt");
            let cipher = AesBlockCipher::new(master_key).expect("an AES master key");
            let mut derived = alloc::vec![0_u8; length];
            derive_session_value(&cipher, &salt, label, &mut derived).expect("derivation");
            derived
        };

        const MASTER_SALT_128: &str = "0EC675AD498AFEEBB6960B3AABE6";
        let master_key_128 = hexadecimal("E1F97A0D3E018BE0D64FA32C06DE4139");
        assert_eq!(
            derive(
                &master_key_128,
                MASTER_SALT_128,
                RTP_ENCRYPTION_KEY_LABEL,
                AES_128_KEY_LENGTH
            ),
            hexadecimal("C61E7A93744F39EE10734AFE3FF7A087"),
            "cipher key"
        );
        assert_eq!(
            derive(
                &master_key_128,
                MASTER_SALT_128,
                RTP_SALT_LABEL,
                COUNTER_MODE_MASTER_SALT_LENGTH
            ),
            hexadecimal("30CBBC08863D8C85D49DB34A9AE1"),
            "cipher salt"
        );
        assert_eq!(
            derive(
                &master_key_128,
                MASTER_SALT_128,
                RTP_AUTHENTICATION_KEY_LABEL,
                SESSION_AUTHENTICATION_KEY_LENGTH
            ),
            hexadecimal("CEBE321F6FF7716B6FD4AB49AF256A156D38BAA4"),
            "auth key"
        );

        const MASTER_SALT_256: &str = "3b04803de51ee7c96423ab5b78d2";
        let master_key_256 = hexadecimal(
            "f0f04914b513f2763a1b1fa130f10e29
             98f6f6e43e4309d1e622a0e332b9f1b6",
        );
        assert_eq!(
            derive(
                &master_key_256,
                MASTER_SALT_256,
                RTP_ENCRYPTION_KEY_LABEL,
                AES_256_KEY_LENGTH
            ),
            hexadecimal("5ba1064e30ec51613cad926c5a28ef731ec7fb397f70a960653caf06554cd8c4"),
            "cipher key"
        );
        assert_eq!(
            derive(
                &master_key_256,
                MASTER_SALT_256,
                RTP_SALT_LABEL,
                COUNTER_MODE_MASTER_SALT_LENGTH
            ),
            hexadecimal("fa31791685ca444a9e07c6c64e93"),
            "cipher salt"
        );
        assert_eq!(
            derive(
                &master_key_256,
                MASTER_SALT_256,
                RTP_AUTHENTICATION_KEY_LABEL,
                SESSION_AUTHENTICATION_KEY_LENGTH
            ),
            hexadecimal("fd9c32d39ed5fbb5a9dc96b30818454d1313dc05"),
            "auth key"
        );
    }

    /// The AES counter-mode keystream of RFC 3711 appendix B.2 and RFC 6188
    /// section 7.1. Both fix SSRC and index at zero, so the counter block is the
    /// session salt shifted left 16 bits, and give the first three output blocks.
    #[test]
    fn aes_counter_mode_keystream_matches_rfc_3711_and_rfc_6188() {
        const SESSION_SALT: &str = "f0f1f2f3f4f5f6f7f8f9fafbfcfd";
        const NO_SOURCE: u32 = 0;
        const FIRST_PACKET: u64 = 0;
        /// The three blocks each RFC prints before its elision.
        const BLOCKS: usize = 3;

        let salt: [u8; COUNTER_MODE_MASTER_SALT_LENGTH] = hexadecimal(SESSION_SALT)
            .try_into()
            .expect("a 14-byte session salt");
        let block = counter_mode_block(&salt, NO_SOURCE, FIRST_PACKET);
        let mut expected_block = [0_u8; AES_BLOCK_LENGTH];
        expected_block[..COUNTER_MODE_MASTER_SALT_LENGTH].copy_from_slice(&salt);
        assert_eq!(
            block, expected_block,
            "with no source and index zero the counter block is the shifted salt"
        );

        for (session_key, expected) in [
            (
                "2B7E151628AED2A6ABF7158809CF4F3C",
                "E03EAD0935C95E80E166B16DD92B4EB4
                 D23513162B02D0F72A43A2FE4A5F97AB
                 41E95B3BB0A2E8DD477901E4FCA894C0",
            ),
            (
                "57f82fe3613fd170a85ec93c40b1f092
                 2ec4cb0dc025b58272147cc438944a98",
                "92bdd28a93c3f52511c677d08b5515a4
                 9da71b2378a854f67050756ded165bac
                 63c4868b7096d88421b563b8c94c9a31",
            ),
        ] {
            let cipher =
                AesBlockCipher::new(&hexadecimal(session_key)).expect("an AES session key");
            // The keystream is what encrypting zeros produces.
            let mut keystream = alloc::vec![0_u8; BLOCKS * AES_BLOCK_LENGTH];
            apply_counter_mode(&cipher, block, &mut keystream).expect("keystream");
            assert_eq!(keystream, hexadecimal(expected));
        }
    }

    /// HMAC-SHA1 itself, from RFC 2202 section 3, truncated the way RFC 3711
    /// section 4.2.1 truncates it.
    #[test]
    fn hmac_sha1_tags_match_rfc_2202_truncated_to_each_length() {
        const KEY: &str = "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b";
        const DATA: &[u8] = b"Hi There";
        const DIGEST: &str = "b617318655057264e28bc0b6fb378c8ef146be00";

        let mut authentication_key = Zeroizing::new([0_u8; SESSION_AUTHENTICATION_KEY_LENGTH]);
        authentication_key.copy_from_slice(&hexadecimal(KEY));
        let digest = hexadecimal(DIGEST);
        for authentication in [
            SrtpAuthentication::HmacSha1Tag80,
            SrtpAuthentication::HmacSha1Tag32,
        ] {
            let keys = FlowKeys {
                cipher: FlowCipher::Null,
                authentication,
                authentication_key: authentication_key.clone(),
            };
            let tag = keys.authentication_tag(&[DATA]).expect("a tag");
            assert_eq!(
                tag,
                digest[..authentication.tag_length()],
                "{authentication:?}"
            );
            assert_eq!(keys.verify_authentication_tag(&[DATA], &tag), Ok(()));
            let mut wrong = tag.clone();
            wrong[0] ^= 1;
            assert_eq!(
                keys.verify_authentication_tag(&[DATA], &wrong),
                Err(SrtpError::AuthenticationFailed)
            );
        }
    }

    /// Every RFC 3711 combination round trips, and the protected packet grows by
    /// exactly the tag the policy declares.
    #[test]
    fn rfc_3711_policies_round_trip_rtp_and_rtcp() {
        const SOURCE: u32 = 0x0a0b_0c0d;
        const SEQUENCE: u16 = 4_242;

        for policy in [
            AES_128_COUNTER_MODE_80,
            AES_128_COUNTER_MODE_32,
            AES_256_COUNTER_MODE_80,
            NULL_CIPHER_80,
            SrtpPolicy {
                cipher: SrtpCipher::Aes128CounterMode,
                authentication: SrtpAuthentication::Null,
            },
        ] {
            let (key, salt) = test_key_material(policy, 0x5a);
            let mut sender = SrtpSender::new(policy, &key, &salt, SOURCE).unwrap();
            let mut receiver = SrtpReceiver::new(policy, &key, &salt, SOURCE).unwrap();

            let packet = test_rtp_packet(SEQUENCE, SOURCE, b"counter mode payload");
            let protected = sender.protect_rtp(&packet).unwrap();
            assert_eq!(
                protected.len(),
                packet.len() + policy.authentication.tag_length(),
                "{policy:?} tag length"
            );
            let header_length = RtpHeader::parse(&packet).unwrap().payload_offset;
            assert_eq!(
                protected[header_length..packet.len()] == packet[header_length..],
                policy.cipher == SrtpCipher::Null,
                "{policy:?} encrypted the payload"
            );
            assert_eq!(receiver.unprotect_rtp(&protected).unwrap(), packet);

            let rtcp = hexadecimal("81c8000d 0a0b0c0d 4e545031 4e545032 52545020 0000042a");
            for mode in [
                RtcpProtectionMode::Encrypt,
                RtcpProtectionMode::AuthenticateOnly,
            ] {
                let protected = sender.protect_rtcp(&rtcp, mode).unwrap();
                assert_eq!(
                    protected.len(),
                    rtcp.len() + SRTCP_INDEX_LENGTH + policy.authentication.tag_length(),
                    "{policy:?} {mode:?}"
                );
                let index_word = u32::from_be_bytes(
                    protected[rtcp.len()..rtcp.len() + SRTCP_INDEX_LENGTH]
                        .try_into()
                        .unwrap(),
                );
                let encrypted =
                    policy.cipher != SrtpCipher::Null && mode == RtcpProtectionMode::Encrypt;
                assert_eq!(
                    index_word & SRTCP_ENCRYPTION_FLAG != 0,
                    encrypted,
                    "{policy:?} {mode:?} E flag"
                );
                assert_eq!(
                    protected[RTCP_AUTHENTICATED_PREFIX_LENGTH..rtcp.len()]
                        == rtcp[RTCP_AUTHENTICATED_PREFIX_LENGTH..],
                    !encrypted,
                    "{policy:?} {mode:?} encrypted the body"
                );
                assert_eq!(receiver.unprotect_rtcp(&protected).unwrap(), rtcp);
            }
        }
    }

    /// A flipped bit anywhere in the authenticated portion is caught, at both
    /// tag lengths. Without a tag the same packet is handed on unchecked, which
    /// is what an unauthenticated policy means.
    #[test]
    fn hmac_sha1_catches_tampering_at_both_tag_lengths() {
        const SOURCE: u32 = 0x1122_3344;

        for policy in [
            AES_128_COUNTER_MODE_80,
            AES_128_COUNTER_MODE_32,
            SrtpPolicy {
                cipher: SrtpCipher::Aes128CounterMode,
                authentication: SrtpAuthentication::Null,
            },
        ] {
            let (key, salt) = test_key_material(policy, 0x31);
            let mut sender = SrtpSender::new(policy, &key, &salt, SOURCE).unwrap();
            let packet = test_rtp_packet(9, SOURCE, b"tamper with me");
            let protected = sender.protect_rtp(&packet).unwrap();
            let header_length = RtpHeader::parse(&packet).unwrap().payload_offset;

            for damaged_byte in [0, header_length, protected.len() - 1] {
                let mut receiver = SrtpReceiver::new(policy, &key, &salt, SOURCE).unwrap();
                let mut damaged = protected.clone();
                damaged[damaged_byte] ^= 1;
                let recovered = receiver.unprotect_rtp(&damaged);
                if policy.authentication == SrtpAuthentication::Null {
                    // The header byte carries the version, so only that one is
                    // refused, and then by the RTP parser rather than a tag.
                    assert!(recovered.is_ok() || damaged_byte == 0, "{policy:?}");
                    continue;
                }
                assert_eq!(
                    recovered,
                    Err(SrtpError::AuthenticationFailed),
                    "{policy:?} byte {damaged_byte}"
                );
            }
        }
    }

    /// Where the MKI sits differs by profile family: RFC 7714 appends it after
    /// the AEAD tag, RFC 3711 puts it before the authentication tag.
    #[test]
    fn the_mki_offset_follows_the_profile_family() {
        const MKI: &[u8] = b"pick-me!";
        const SOURCE: u32 = 0x0102_0304;

        for (policy, trailing_tag) in [
            (AES_128_GCM, AUTHENTICATION_TAG_LENGTH),
            (AES_128_COUNTER_MODE_80, 0),
            (AES_128_COUNTER_MODE_32, 0),
        ] {
            let (key, salt) = test_key_material(policy, 0x77);
            let mut sender = SrtpSender::new(policy, &key, &salt, SOURCE).unwrap();
            sender.set_mki(Some(MKI)).unwrap();
            let keys = Vec::from([SrtpKeyingMaterial::new(policy, &key, &salt, 0)
                .unwrap()
                .with_mki(MKI)
                .unwrap()]);
            let mut receiver = SrtpReceiver::new_with_keys(SOURCE, keys).unwrap();

            let packet = test_rtp_packet(3, SOURCE, b"mki placement");
            let protected = sender.protect_rtp(&packet).unwrap();
            let mki_end = protected.len() - policy.authentication.tag_length();
            assert_eq!(&protected[mki_end - MKI.len()..mki_end], MKI, "{policy:?}");
            assert_eq!(
                protected.len(),
                packet.len() + trailing_tag + MKI.len() + policy.authentication.tag_length()
            );
            assert_eq!(receiver.unprotect_rtp(&protected).unwrap(), packet);

            let rtcp = hexadecimal("80c90001 01020304");
            let protected = sender
                .protect_rtcp(&rtcp, RtcpProtectionMode::Encrypt)
                .unwrap();
            let mki_end = protected.len() - policy.authentication.tag_length();
            assert_eq!(&protected[mki_end - MKI.len()..mki_end], MKI, "{policy:?}");
            assert_eq!(receiver.unprotect_rtcp(&protected).unwrap(), rtcp);
        }
    }

    /// The byte count alone picks a policy, and the DTLS-SRTP profile ids are
    /// the ones RFC 5764 and RFC 7714 assign.
    #[test]
    fn key_material_lengths_and_protection_profile_ids() {
        for (length, policy) in [
            (28, AES_128_GCM),
            (44, AES_256_GCM),
            (30, AES_128_COUNTER_MODE_80),
            (46, AES_256_COUNTER_MODE_80),
        ] {
            assert_eq!(policy_for_key_material(length), Some(policy), "{length}");
        }
        for length in [0, 12, 14, 27, 29, 31, 43, 45, 47] {
            assert_eq!(policy_for_key_material(length), None, "{length}");
        }

        assert_eq!(
            AES_128_COUNTER_MODE_80.dtls_protection_profile(),
            Some(DTLS_SRTP_AES128_CM_HMAC_SHA1_80)
        );
        assert_eq!(
            AES_128_COUNTER_MODE_32.dtls_protection_profile(),
            Some(DTLS_SRTP_AES128_CM_HMAC_SHA1_32)
        );
        assert_eq!(
            NULL_CIPHER_80.dtls_protection_profile(),
            Some(DTLS_SRTP_NULL_HMAC_SHA1_80)
        );
        assert_eq!(
            AES_128_GCM.dtls_protection_profile(),
            Some(DTLS_SRTP_AEAD_AES_128_GCM)
        );
        assert_eq!(
            AES_256_GCM.dtls_protection_profile(),
            Some(DTLS_SRTP_AEAD_AES_256_GCM)
        );
        // RFC 5764 reserved 0x0003 and 0x0004 without assigning them, so the
        // AES-256 counter-mode suite has no DTLS-SRTP profile to negotiate.
        assert_eq!(AES_256_COUNTER_MODE_80.dtls_protection_profile(), None);

        assert_eq!(
            SrtpPolicy::new(SrtpCipher::Aes128Gcm, SrtpAuthentication::HmacSha1Tag80),
            Err(SrtpError::AeadCipherTakesNoAuthentication)
        );
    }

    /// The properties resolve one policy per flow: an explicit value wins, an
    /// unset one follows the key length, and a contradiction is refused.
    #[test]
    fn key_settings_resolve_one_policy_per_flow() {
        /// 16 key bytes then a 14-byte salt: the counter-mode lengths.
        const COUNTER_MODE_KEY: &str =
            "000102030405060708090a0b0c0d0e0f517569642070726f2071756f2121";
        /// 16 key bytes then a 12-byte salt: the AES-128-GCM lengths.
        const GCM_KEY: &str = "000102030405060708090a0b0c0d0e0f517569642070726f2071756f";

        let mut settings = SrtpKeySettings::default();
        // With nothing set the element reports the GStreamer defaults.
        assert_eq!(settings.cipher(SrtpFlow::Rtp), DEFAULT_CIPHER);
        assert_eq!(
            settings.authentication(SrtpFlow::Rtp),
            SrtpAuthentication::HmacSha1Tag80
        );

        settings.set_key_hexadecimal(GCM_KEY).unwrap();
        assert_eq!(settings.cipher(SrtpFlow::Rtp), SrtpCipher::Aes128Gcm);
        assert_eq!(
            settings.authentication(SrtpFlow::Rtcp),
            SrtpAuthentication::Null,
            "an AEAD cipher carries its own tag"
        );
        assert_eq!(settings.policy(SrtpFlow::Rtp), Ok(AES_128_GCM));

        // A cipher the key length cannot key is refused and leaves the previous
        // one in place.
        assert!(settings
            .set_cipher(SrtpFlow::Rtp, SrtpCipher::Aes128CounterMode)
            .is_err());
        assert_eq!(settings.cipher(SrtpFlow::Rtp), SrtpCipher::Aes128Gcm);
        assert!(settings
            .set_authentication(SrtpFlow::Rtp, SrtpAuthentication::HmacSha1Tag80)
            .is_err());

        settings.set_key_hexadecimal(COUNTER_MODE_KEY).unwrap();
        assert_eq!(settings.policy(SrtpFlow::Rtp), Ok(AES_128_COUNTER_MODE_80));
        // Only the flow that was set moves.
        settings
            .set_authentication(SrtpFlow::Rtcp, SrtpAuthentication::HmacSha1Tag32)
            .unwrap();
        assert_eq!(settings.policy(SrtpFlow::Rtp), Ok(AES_128_COUNTER_MODE_80));
        assert_eq!(settings.policy(SrtpFlow::Rtcp), Ok(AES_128_COUNTER_MODE_32));

        // The NULL cipher takes either counter-mode key length.
        settings
            .set_cipher(SrtpFlow::Rtp, SrtpCipher::Null)
            .unwrap();
        assert_eq!(settings.policy(SrtpFlow::Rtp), Ok(NULL_CIPHER_80));
        let key = settings.master_key(SrtpFlow::Rtp).unwrap();
        assert_eq!(key.master_key().len(), AES_128_KEY_LENGTH);
        assert_eq!(key.master_salt().len(), COUNTER_MODE_MASTER_SALT_LENGTH);
    }

    #[test]
    fn srtp_encryption_matches_both_rfc_7714_profiles() {
        let plaintext = rfc_rtp_packet();
        let expected_128 = hexadecimal(
            "8040f17b 8041f8d3 5501a0b2 f24de3a3
             fb34de6c acba861c 9d7e4bca be633bd5
             0d294e6f 42a5f47a 51c7d19b 36de3adf
             8833899d 7f27beb1 6a9152cf 765ee439 0cce",
        );
        let expected_256 = hexadecimal(
            "8040f17b 8041f8d3 5501a0b2 32b1de78
             a822fe12 ef9f78fa 332e33aa b1801238
             9a58e2f3 b50b2a02 76ffae0f 1ba63799
             b87b7aa3 db36dfff d6b0f9bb 7878d7a7 6c13",
        );

        let key_128: Vec<u8> = (0_u8..16).collect();
        let keys_128 = SessionKeys::from_session_values(AES_128_GCM, &key_128, TEST_SALT);
        assert_eq!(
            keys_128.protect_rtp(&plaintext, 0xf17b, NO_MKI).unwrap(),
            expected_128
        );

        let key_256: Vec<u8> = (0_u8..32).collect();
        let keys_256 = SessionKeys::from_session_values(AES_256_GCM, &key_256, TEST_SALT);
        assert_eq!(
            keys_256.protect_rtp(&plaintext, 0xf17b, NO_MKI).unwrap(),
            expected_256
        );
        assert_eq!(
            keys_256
                .unprotect_rtp(&expected_256, 0xf17b, NO_MKI_LENGTH)
                .unwrap(),
            plaintext
        );
    }

    #[test]
    fn srtcp_encryption_and_authentication_match_rfc_7714() {
        let plaintext = rfc_rtcp_packet();
        let key_128: Vec<u8> = (0_u8..16).collect();
        let keys = SessionKeys::from_session_values(AES_128_GCM, &key_128, TEST_SALT);
        let encrypted = hexadecimal(
            "81c8000d 4d617273 63e94885 dcdab67c
             a727d766 2f6b7e99 7ff5c0f7 6c06f32d
             c676a5f1 730d6fda 4ce09b46 86303ded
             0bb9275b c84aa458 96cf4d2f c5abf872
             45d9eade 800005d4",
        );
        assert_eq!(
            keys.protect_rtcp(&plaintext, 0x05d4, RtcpProtectionMode::Encrypt, NO_MKI)
                .unwrap(),
            encrypted
        );
        assert_eq!(
            keys.unprotect_rtcp(&encrypted, NO_MKI_LENGTH).unwrap().0,
            plaintext
        );

        let authenticated = hexadecimal(
            "81c8000d 4d617273 4e545031 4e545032
             52545020 0000042a 0000e930 4c756e61
             deadbeef deadbeef deadbeef deadbeef
             deadbeef 841dd968 3dd78ec9 2ae58790
             125f62b3 000005d4",
        );
        assert_eq!(
            keys.protect_rtcp(
                &plaintext,
                0x05d4,
                RtcpProtectionMode::AuthenticateOnly,
                NO_MKI
            )
            .unwrap(),
            authenticated
        );
        assert_eq!(
            keys.unprotect_rtcp(&authenticated, NO_MKI_LENGTH)
                .unwrap()
                .0,
            plaintext
        );
    }

    #[test]
    fn public_contexts_round_trip_wraparound_and_reject_replay() {
        let master_key = [0x31_u8; 16];
        let master_salt = [0x72_u8; 12];
        let synchronization_source = 0x1020_3040;
        let mut sender = SrtpSender::new(
            AES_128_GCM,
            &master_key,
            &master_salt,
            synchronization_source,
        )
        .unwrap();
        let mut receiver = SrtpReceiver::new(
            AES_128_GCM,
            &master_key,
            &master_salt,
            synchronization_source,
        )
        .unwrap();

        for sequence in [u16::MAX, 0, 1] {
            let header = RtpHeader {
                payload_type: 96,
                marker: true,
                sequence,
                timestamp: u32::from(sequence),
                ssrc: synchronization_source,
            };
            let mut packet = header.to_bytes().to_vec();
            packet.extend_from_slice(b"payload");
            let protected = sender.protect_rtp(&packet).unwrap();
            assert_eq!(receiver.unprotect_rtp(&protected).unwrap(), packet);
            assert_eq!(
                receiver.unprotect_rtp(&protected),
                Err(SrtpError::RepeatedPacket)
            );
        }

        let rtcp = hexadecimal("80c90001 10203040");
        for mode in [
            RtcpProtectionMode::Encrypt,
            RtcpProtectionMode::AuthenticateOnly,
        ] {
            let protected = sender.protect_rtcp(&rtcp, mode).unwrap();
            assert_eq!(receiver.unprotect_rtcp(&protected).unwrap(), rtcp);
            assert_eq!(
                receiver.unprotect_rtcp(&protected),
                Err(SrtpError::RepeatedPacket)
            );
        }
    }

    #[test]
    fn authentication_failure_does_not_consume_the_packet_index() {
        let key = [7_u8; 32];
        let salt = [9_u8; 12];
        let synchronization_source = 0x1122_3344;
        let mut sender = SrtpSender::new(AES_256_GCM, &key, &salt, synchronization_source).unwrap();
        let mut receiver =
            SrtpReceiver::new(AES_256_GCM, &key, &salt, synchronization_source).unwrap();
        let header = RtpHeader {
            payload_type: 96,
            marker: false,
            sequence: 42,
            timestamp: 17,
            ssrc: synchronization_source,
        };
        let mut packet = header.to_bytes().to_vec();
        packet.extend_from_slice(b"authenticated payload");
        let protected = sender.protect_rtp(&packet).unwrap();
        let mut damaged = protected.clone();
        *damaged.last_mut().unwrap() ^= 1;
        assert_eq!(
            receiver.unprotect_rtp(&damaged),
            Err(SrtpError::AuthenticationFailed)
        );
        assert_eq!(receiver.unprotect_rtp(&protected).unwrap(), packet);
    }

    #[test]
    fn rtp_csrc_extension_and_padding_are_protected_and_recovered() {
        let key = [0x41_u8; 16];
        let salt = [0x29_u8; 12];
        let synchronization_source = 0x1122_3344;
        let mut sender = SrtpSender::new(AES_128_GCM, &key, &salt, synchronization_source).unwrap();
        let mut receiver =
            SrtpReceiver::new(AES_128_GCM, &key, &salt, synchronization_source).unwrap();
        let packet = hexadecimal(
            "b160002a 01020304 11223344 55667788
             bede0001 aabbccdd 7061796c 6f616400 0003",
        );
        let header_length = RtpHeader::parse(&packet).unwrap().payload_offset;

        let protected = sender.protect_rtp(&packet).unwrap();
        assert_eq!(&protected[..header_length], &packet[..header_length]);
        assert_ne!(
            &protected[header_length..protected.len() - AUTHENTICATION_TAG_LENGTH],
            &packet[header_length..]
        );
        assert_eq!(receiver.unprotect_rtp(&protected).unwrap(), packet);
    }

    #[test]
    fn receiver_uses_the_configured_initial_rollover_counter() {
        let key = [0x61_u8; 16];
        let salt = [0x17_u8; 12];
        let synchronization_source = 0x1234_5678;
        let rollover_counter = 7;
        let header = RtpHeader {
            payload_type: 96,
            marker: false,
            sequence: 500,
            timestamp: 29,
            ssrc: synchronization_source,
        };
        let mut packet = header.to_bytes().to_vec();
        packet.extend_from_slice(b"joined stream");
        let keys = SessionKeys::derive(AES_128_GCM, &key, &salt).unwrap();
        let packet_index = (u64::from(rollover_counter) << 16) | u64::from(header.sequence);
        let protected = keys.protect_rtp(&packet, packet_index, NO_MKI).unwrap();
        let mut receiver = SrtpReceiver::new_with_rollover_counter(
            AES_128_GCM,
            &key,
            &salt,
            synchronization_source,
            rollover_counter,
        )
        .unwrap();

        assert_eq!(receiver.unprotect_rtp(&protected).unwrap(), packet);
    }

    #[test]
    fn receiver_set_requests_one_key_for_each_source() {
        use core::cell::Cell;

        let key = [0x25_u8; 16];
        let salt = [0x38_u8; 12];
        let sources = [0x1111_1111, 0x2222_2222];
        let mut senders: Vec<SrtpSender> = sources
            .iter()
            .map(|source| SrtpSender::new(AES_128_GCM, &key, &salt, *source).unwrap())
            .collect();
        let provider_calls = Cell::new(0);
        let mut receivers = SrtpReceiverSet::new(|source| {
            provider_calls.set(provider_calls.get() + 1);
            if !sources.contains(&source) {
                return Vec::new();
            }
            Vec::from([SrtpKeyingMaterial::new(AES_128_GCM, &key, &salt, 0).unwrap()])
        });

        for (sequence, (source, sender)) in sources.iter().zip(&mut senders).enumerate() {
            let header = RtpHeader {
                payload_type: 96,
                marker: false,
                sequence: sequence as u16,
                timestamp: sequence as u32,
                ssrc: *source,
            };
            let mut packet = header.to_bytes().to_vec();
            packet.extend_from_slice(b"source payload");
            let protected = sender.protect_rtp(&packet).unwrap();
            assert_eq!(receivers.unprotect_rtp(&protected).unwrap(), packet);
        }

        assert_eq!(provider_calls.get(), sources.len());
        let unknown_header = RtpHeader {
            payload_type: 96,
            marker: false,
            sequence: 0,
            timestamp: 0,
            ssrc: 0x3333_3333,
        };
        assert_eq!(
            receivers.unprotect_rtp(&unknown_header.to_bytes()),
            Err(SrtpError::MissingKey)
        );
    }

    #[test]
    fn replacing_keys_preserves_indices_and_resets_usage() {
        let first_key = [0x19_u8; 16];
        let first_salt = [0x20_u8; 12];
        let second_key = [0x31_u8; 16];
        let second_salt = [0x42_u8; 12];
        let synchronization_source = 0x1020_3040;
        let soft_limits = SrtpSoftLimits {
            srtp_packets: 1,
            srtcp_packets: 1,
        };
        let mut sender = SrtpSender::new_with_soft_limits(
            AES_128_GCM,
            &first_key,
            &first_salt,
            synchronization_source,
            soft_limits,
        )
        .unwrap();
        let mut receiver =
            SrtpReceiver::new(AES_128_GCM, &first_key, &first_salt, synchronization_source)
                .unwrap();
        let rtcp = hexadecimal("80c90001 10203040");
        let first_rtcp = sender
            .protect_rtcp(&rtcp, RtcpProtectionMode::Encrypt)
            .unwrap();
        assert_eq!(receiver.unprotect_rtcp(&first_rtcp).unwrap(), rtcp);
        assert_eq!(sender.key_usage().srtcp, KeyUsage::SoftLimitReached);

        sender
            .replace_key(AES_128_GCM, &second_key, &second_salt)
            .unwrap();
        receiver
            .replace_key(AES_128_GCM, &second_key, &second_salt)
            .unwrap();
        assert_eq!(
            sender.key_usage(),
            SrtpKeyUsage {
                srtp: KeyUsage::Normal,
                srtcp: KeyUsage::Normal,
            }
        );
        let second_rtcp = sender
            .protect_rtcp(&rtcp, RtcpProtectionMode::Encrypt)
            .unwrap();
        assert_eq!(receiver.unprotect_rtcp(&second_rtcp).unwrap(), rtcp);
        assert_eq!(
            &first_rtcp[first_rtcp.len() - 4..],
            &0x8000_0000_u32.to_be_bytes()
        );
        assert_eq!(
            &second_rtcp[second_rtcp.len() - 4..],
            &0x8000_0001_u32.to_be_bytes()
        );
    }

    /// A packet with `payload`, from `synchronization_source`, at `sequence`.
    fn test_rtp_packet(sequence: u16, synchronization_source: u32, payload: &[u8]) -> Vec<u8> {
        let header = RtpHeader {
            payload_type: 96,
            marker: false,
            sequence,
            timestamp: u32::from(sequence),
            ssrc: synchronization_source,
        };
        let mut packet = header.to_bytes().to_vec();
        packet.extend_from_slice(payload);
        packet
    }

    /// A sequence number below the newest one is a reordered packet, not a
    /// wrap: RFC 3711 appendix A compares in signed arithmetic, so the rollover
    /// counter only moves for a packet on the far side of the wrap.
    #[test]
    fn a_reordered_sequence_number_keeps_the_current_rollover_counter() {
        /// Below the half range, where a late packet used to read as a wrap.
        const LOW_SEQUENCE: u16 = 1_100;
        /// Above the half range, with a forward jump wider than the half range
        /// stays inside the current counter.
        const HIGH_SEQUENCE: u16 = 40_000;
        const REORDER_DISTANCE: u16 = 100;

        let mut indices = PacketIndexState::new(0, DEFAULT_REPLAY_WINDOW).unwrap();
        indices.accept(u64::from(LOW_SEQUENCE));
        assert_eq!(
            indices.estimate_rtp_index(LOW_SEQUENCE - REORDER_DISTANCE),
            Ok(u64::from(LOW_SEQUENCE - REORDER_DISTANCE))
        );

        let mut indices = PacketIndexState::new(0, DEFAULT_REPLAY_WINDOW).unwrap();
        indices.accept(u64::from(HIGH_SEQUENCE));
        assert_eq!(
            indices.estimate_rtp_index(HIGH_SEQUENCE - REORDER_DISTANCE),
            Ok(u64::from(HIGH_SEQUENCE - REORDER_DISTANCE))
        );
        // A jump forward of more than the half range is still this counter.
        assert_eq!(
            indices.estimate_rtp_index(u16::MAX),
            Ok(u64::from(u16::MAX))
        );
        // Only a sequence on the other side of the wrap advances it.
        assert_eq!(
            indices.estimate_rtp_index(0),
            Ok(1_u64 << 16),
            "a wrap advances the rollover counter"
        );
    }

    #[test]
    fn the_replay_window_spans_exactly_its_configured_size() {
        /// Far enough in that the whole window sits above zero.
        const HIGHEST_INDEX: u64 = 10_000;

        let mut indices = PacketIndexState::new(0, MINIMUM_REPLAY_WINDOW).unwrap();
        indices.accept(HIGHEST_INDEX);
        let oldest_covered = HIGHEST_INDEX - (MINIMUM_REPLAY_WINDOW as u64 - 1);
        assert_eq!(indices.check(oldest_covered), Ok(()));
        assert_eq!(
            indices.check(oldest_covered - 1),
            Err(SrtpError::PacketTooOld)
        );
        indices.accept(oldest_covered);
        assert_eq!(
            indices.check(oldest_covered),
            Err(SrtpError::RepeatedPacket)
        );
    }

    #[test]
    fn the_replay_window_carries_bits_across_word_boundaries() {
        /// Four 64-bit words, so a shift moves bits from one to the next.
        const WINDOW: usize = 4 * REPLAY_BITMAP_WORD_BITS;
        /// Not a multiple of the word width, so each shift splits every word.
        const STEP: u64 = 100;
        const FIRST_INDEX: u64 = 5_000;

        let accepted = [FIRST_INDEX, FIRST_INDEX + STEP, FIRST_INDEX + 2 * STEP];
        let mut indices = PacketIndexState::new(0, WINDOW).unwrap();
        for index in accepted {
            indices.accept(index);
        }
        for index in accepted {
            assert_eq!(
                indices.check(index),
                Err(SrtpError::RepeatedPacket),
                "index {index} was recorded"
            );
        }
        assert_eq!(indices.check(FIRST_INDEX + 1), Ok(()));
        assert_eq!(indices.check(FIRST_INDEX + STEP - 1), Ok(()));

        let newest = FIRST_INDEX + 2 * STEP;
        let oldest_covered = newest - (WINDOW as u64 - 1);
        assert_eq!(indices.check(oldest_covered), Ok(()));
        assert_eq!(
            indices.check(oldest_covered - 1),
            Err(SrtpError::PacketTooOld)
        );

        // A jump of a whole window leaves nothing of the old one behind.
        indices.accept(newest + WINDOW as u64);
        assert_eq!(indices.check(newest), Err(SrtpError::PacketTooOld));
    }

    #[test]
    fn repeated_transmission_reprotects_only_an_index_still_in_the_window() {
        let key = [0x66_u8; 16];
        let salt = [0x77_u8; 12];
        let synchronization_source = 0x0102_0304;
        let mut sender = SrtpSender::new(AES_128_GCM, &key, &salt, synchronization_source).unwrap();
        let first = test_rtp_packet(1, synchronization_source, b"repeat me");

        let protected = sender.protect_rtp(&first).unwrap();
        assert_eq!(sender.protect_rtp(&first), Err(SrtpError::RepeatedPacket));
        sender.set_repeat_transmission(true);
        assert_eq!(sender.protect_rtp(&first).unwrap(), protected);
        assert_eq!(
            sender.srtp_key_invocations, 2,
            "the repeat uses the key too"
        );

        // Once the index leaves the window it is refused again, the way
        // libsrtp's allow_repeat_tx does.
        let last_sequence = u16::try_from(DEFAULT_REPLAY_WINDOW).unwrap() + 2;
        for sequence in 2..=last_sequence {
            sender
                .protect_rtp(&test_rtp_packet(
                    sequence,
                    synchronization_source,
                    b"filler",
                ))
                .unwrap();
        }
        assert_eq!(sender.protect_rtp(&first), Err(SrtpError::PacketTooOld));
    }

    #[test]
    fn an_mki_selects_one_of_a_context_key_set_and_is_stripped() {
        const FIRST_MKI: &[u8] = b"key-one!";
        const SECOND_MKI: &[u8] = b"key-two!";
        const UNKNOWN_MKI: &[u8] = b"key-xxx!";

        let first_key = [0x11_u8; 16];
        let second_key = [0x22_u8; 16];
        let salt = [0x33_u8; 12];
        let synchronization_source = 0x0a0b_0c0d;

        let mut sender =
            SrtpSender::new(AES_128_GCM, &second_key, &salt, synchronization_source).unwrap();
        sender.set_mki(Some(SECOND_MKI)).unwrap();
        let keys = Vec::from([
            SrtpKeyingMaterial::new(AES_128_GCM, &first_key, &salt, 0)
                .unwrap()
                .with_mki(FIRST_MKI)
                .unwrap(),
            SrtpKeyingMaterial::new(AES_128_GCM, &second_key, &salt, 0)
                .unwrap()
                .with_mki(SECOND_MKI)
                .unwrap(),
        ]);
        let mut receiver = SrtpReceiver::new_with_keys(synchronization_source, keys).unwrap();

        let packet = test_rtp_packet(7, synchronization_source, b"mki payload");
        let protected = sender.protect_rtp(&packet).unwrap();
        assert_eq!(
            protected.len(),
            packet.len() + AUTHENTICATION_TAG_LENGTH + SECOND_MKI.len()
        );
        assert_eq!(&protected[protected.len() - SECOND_MKI.len()..], SECOND_MKI);
        assert_eq!(receiver.unprotect_rtp(&protected).unwrap(), packet);

        // RFC 7714 figure 5 puts the SRTCP MKI after the index word.
        let rtcp = hexadecimal("80c90001 0a0b0c0d");
        let protected = sender
            .protect_rtcp(&rtcp, RtcpProtectionMode::Encrypt)
            .unwrap();
        let index_end = protected.len() - SECOND_MKI.len();
        assert_eq!(&protected[index_end..], SECOND_MKI);
        assert_eq!(
            &protected[index_end - SRTCP_INDEX_LENGTH..index_end],
            &0x8000_0000_u32.to_be_bytes()
        );
        assert_eq!(receiver.unprotect_rtcp(&protected).unwrap(), rtcp);

        let mut unknown = protected[..index_end].to_vec();
        unknown.extend_from_slice(UNKNOWN_MKI);
        assert_eq!(
            receiver.unprotect_rtcp(&unknown),
            Err(SrtpError::MissingKey)
        );
    }

    #[test]
    fn one_context_takes_one_mki_length() {
        let key = [0x44_u8; 16];
        let salt = [0x55_u8; 12];
        let material = || SrtpKeyingMaterial::new(AES_128_GCM, &key, &salt, 0).unwrap();

        assert_eq!(
            material().with_mki(&[]).unwrap_err(),
            SrtpError::InvalidMkiLength { actual: 0 }
        );
        assert_eq!(
            material()
                .with_mki(&[0_u8; MAXIMUM_MKI_LENGTH + 1])
                .unwrap_err(),
            SrtpError::InvalidMkiLength {
                actual: MAXIMUM_MKI_LENGTH + 1
            }
        );
        assert!(material().with_mki(&[0_u8; MAXIMUM_MKI_LENGTH]).is_ok());

        let source = 0x0000_0001;
        for keys in [
            Vec::from([
                material().with_mki(b"one").unwrap(),
                material().with_mki(b"longer").unwrap(),
            ]),
            Vec::from([material().with_mki(b"one").unwrap(), material()]),
            Vec::from([material(), material()]),
        ] {
            assert_eq!(
                SrtpReceiver::new_with_keys(source, keys).unwrap_err(),
                SrtpError::InconsistentMki
            );
        }
    }

    #[test]
    fn a_replay_window_outside_the_accepted_range_is_refused() {
        for size in [MINIMUM_REPLAY_WINDOW - 1, MAXIMUM_REPLAY_WINDOW + 1] {
            assert_eq!(
                PacketIndexState::new(0, size).unwrap_err(),
                SrtpError::InvalidReplayWindow { size }
            );
        }
        assert!(PacketIndexState::new(0, MINIMUM_REPLAY_WINDOW).is_ok());
        assert!(PacketIndexState::new(0, MAXIMUM_REPLAY_WINDOW).is_ok());
    }

    #[test]
    fn hexadecimal_text_round_trips_through_both_directions() {
        const MKI: &[u8] = &[0x00, 0x9f, 0xff, 0x10];
        assert_eq!(encode_hexadecimal(MKI), "009fff10");
        assert_eq!(decode_hexadecimal("009FFF10").unwrap().as_slice(), MKI);
        assert!(decode_hexadecimal("009").is_none());
        assert!(decode_hexadecimal("00zz").is_none());
    }

    #[test]
    fn sender_reports_soft_limits_and_enforces_rfc_hard_limits() {
        let key = [0x51_u8; 16];
        let salt = [0x63_u8; 12];
        let synchronization_source = 0x1020_3040;
        let mut sender = SrtpSender::new_with_soft_limits(
            AES_128_GCM,
            &key,
            &salt,
            synchronization_source,
            SrtpSoftLimits {
                srtp_packets: 1,
                srtcp_packets: 1,
            },
        )
        .unwrap();
        let header = RtpHeader {
            payload_type: 96,
            marker: false,
            sequence: 1,
            timestamp: 1,
            ssrc: synchronization_source,
        };
        let mut packet = header.to_bytes().to_vec();
        packet.extend_from_slice(b"payload");
        sender.protect_rtp(&packet).unwrap();
        assert_eq!(sender.key_usage().srtp, KeyUsage::SoftLimitReached);

        sender.srtp_key_invocations = MAXIMUM_SRTP_KEY_INVOCATIONS;
        assert_eq!(
            sender.protect_rtp(&packet),
            Err(SrtpError::KeyLifetimeExhausted)
        );
        let rtcp = hexadecimal("80c90001 10203040");
        sender.srtcp_key_invocations = MAXIMUM_SRTCP_KEY_INVOCATIONS;
        assert_eq!(
            sender.protect_rtcp(&rtcp, RtcpProtectionMode::Encrypt),
            Err(SrtpError::KeyLifetimeExhausted)
        );
        assert_eq!(
            SrtpSender::new_with_soft_limits(
                AES_128_GCM,
                &key,
                &salt,
                synchronization_source,
                SrtpSoftLimits {
                    srtp_packets: MAXIMUM_SRTP_KEY_INVOCATIONS,
                    srtcp_packets: 1,
                },
            )
            .unwrap_err(),
            SrtpError::InvalidSoftLimit
        );
    }
}

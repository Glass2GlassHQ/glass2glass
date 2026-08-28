//! `srtpenc`: SRTP or SRTCP protection for one RTP or RTCP flow.
//!
//! Which flow an instance carries comes from the caps the link settled on
//! (`application/x-rtp` or `application/x-rtcp`), the way GStreamer's `srtpenc`
//! splits its `rtp_sink` and `rtcp_sink` pads. The cipher and authentication
//! follow the `key` length unless `rtp-cipher` / `rtp-auth` (or the `rtcp-`
//! pair, on an RTCP instance) name them.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::log::LogSource;
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{
    AsyncElement, BusHandle, BusMessage, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, HardwareError, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::srtp::{
    decode_hexadecimal, encode_hexadecimal, flow_caps, forward_until_data_frame,
    is_fatal_send_error, protect_for_flow, protection_property_flow, push_frame_bytes,
    synchronization_source, validate_replay_window, validated_mki, KeyUsage, RtcpProtectionMode,
    SrtpAuthentication, SrtpCipher, SrtpError, SrtpFlow, SrtpKeySettings, SrtpKeyUsage,
    SrtpMasterKey, SrtpPolicy, SrtpSender, SrtpSoftLimits, DEFAULT_REPLAY_WINDOW,
    KEY_PROPERTY_BLURB, RTCP_AUTHENTICATION_PROPERTY, RTCP_CIPHER_PROPERTY,
    RTP_AUTHENTICATION_PROPERTY, RTP_CIPHER_PROPERTY, UNAUTHENTICATED_POLICY_WARNING,
};

/// Posted once when the key reaches its soft use limit, the analog of the gst
/// `soft-limit` signal.
const SOFT_LIMIT_MESSAGE: &str =
    "srtpenc: the SRTP master key reached its soft use limit, replace it before the RFC 3711 \
     lifetime ends";

const SRTPENC_PROPERTIES: &[PropertySpec] = &[
    PropertySpec::new("key", PropKind::Str, KEY_PROPERTY_BLURB),
    RTP_CIPHER_PROPERTY,
    RTCP_CIPHER_PROPERTY,
    RTP_AUTHENTICATION_PROPERTY,
    RTCP_AUTHENTICATION_PROPERTY,
    PropertySpec::new(
        "rtcp-encrypt",
        PropKind::Bool,
        "encrypt the RTCP body instead of only authenticating it",
    )
    .with_default("true"),
    PropertySpec::new(
        "replay-window-size",
        PropKind::Uint,
        "size of the replay protection window, in packets. Read when the first packet builds the \
         sender context",
    )
    .with_range("64", "32768")
    .with_default("128"),
    PropertySpec::new(
        "allow-repeat-tx",
        PropKind::Bool,
        "whether retransmissions of packets with the same sequence number are allowed (note that \
         such repeated transmissions must have the same RTP payload, or a severe security \
         weakness is introduced)",
    )
    .with_default("false"),
    PropertySpec::new(
        "mki",
        PropKind::Str,
        "hexadecimal Master Key Identifier appended to every protected packet, 2 to 256 digits. \
         Empty means no MKI",
    )
    .with_default(""),
];

/// What `srtpenc` protected and what it dropped. The counts run from the first
/// packet and a rekey does not reset them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SrtpEncStats {
    pub packets_protected: u64,
    pub packets_dropped: u64,
}

/// Protects one system-memory RTP or RTCP packet per frame, under one master
/// key and one synchronization source.
///
/// The source is taken from the first packet and the sender context is built
/// there; a later packet from a different source is dropped, as is one that is
/// not a valid RTP or RTCP packet. Running out of key lifetime is fatal: the
/// RFC 3711 limit may not be exceeded, so the run stops rather than reusing a
/// packet index.
#[derive(Debug)]
pub struct SrtpEnc {
    keys: SrtpKeySettings,
    soft_limits: SrtpSoftLimits,
    encrypt_rtcp: bool,
    replay_window: usize,
    allow_repeat_transmission: bool,
    mki: Option<Vec<u8>>,
    bus: Option<BusHandle>,
    flow: Option<SrtpFlow>,
    sender: Option<SrtpSender>,
    soft_limit_reported: bool,
    packets_protected: u64,
    packets_dropped: u64,
}

impl Default for SrtpEnc {
    fn default() -> Self {
        Self {
            keys: SrtpKeySettings::default(),
            soft_limits: SrtpSoftLimits::default(),
            encrypt_rtcp: true,
            replay_window: DEFAULT_REPLAY_WINDOW,
            allow_repeat_transmission: false,
            mki: None,
            bus: None,
            flow: None,
            sender: None,
            soft_limit_reported: false,
            packets_protected: 0,
            packets_dropped: 0,
        }
    }
}

impl SrtpEnc {
    pub fn new(
        policy: SrtpPolicy,
        master_key: &[u8],
        master_salt: &[u8],
    ) -> Result<Self, SrtpError> {
        Ok(Self {
            keys: SrtpKeySettings::from_master_key(&SrtpMasterKey::new(
                policy,
                master_key,
                master_salt,
            )?),
            ..Self::default()
        })
    }

    /// Protect the RTP flow with this cipher and authentication instead of the
    /// pair the `key` length picks. Read on an RTP instance.
    pub fn with_rtp_protection(
        mut self,
        cipher: SrtpCipher,
        authentication: SrtpAuthentication,
    ) -> Result<Self, SrtpError> {
        self.keys.set_cipher(SrtpFlow::Rtp, cipher)?;
        self.keys
            .set_authentication(SrtpFlow::Rtp, authentication)?;
        Ok(self)
    }

    /// The same for the RTCP flow, read on an RTCP instance.
    pub fn with_rtcp_protection(
        mut self,
        cipher: SrtpCipher,
        authentication: SrtpAuthentication,
    ) -> Result<Self, SrtpError> {
        self.keys.set_cipher(SrtpFlow::Rtcp, cipher)?;
        self.keys
            .set_authentication(SrtpFlow::Rtcp, authentication)?;
        Ok(self)
    }

    /// Rekey thresholds below the fixed RFC 3711 hard limits. Ones at or above
    /// a hard limit are refused when the pipeline is configured.
    pub fn with_soft_limits(mut self, soft_limits: SrtpSoftLimits) -> Self {
        self.soft_limits = soft_limits;
        self
    }

    /// Where the soft-limit notice is posted. Without a bus the notice is only
    /// logged.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Encrypt the RTCP body (the default) or only authenticate it. Read on the
    /// RTCP flow; an RTP instance protects its payload either way.
    pub fn with_rtcp_encryption(mut self, encrypt: bool) -> Self {
        self.encrypt_rtcp = encrypt;
        self
    }

    /// The window the repeat check reads, 64..=32768 packets. Read when the
    /// first packet builds the sender context.
    pub fn with_replay_window(mut self, packets: usize) -> Result<Self, SrtpError> {
        validate_replay_window(packets)?;
        self.replay_window = packets;
        Ok(self)
    }

    /// Protect a packet whose index was already protected again instead of
    /// dropping it. Every repeat has to carry the same RTP payload: the
    /// keystream repeats with the index, and two payloads under one keystream
    /// break the cipher.
    pub fn with_repeat_transmission(mut self, allow: bool) -> Self {
        self.allow_repeat_transmission = allow;
        if let Some(sender) = &mut self.sender {
            sender.set_repeat_transmission(allow);
        }
        self
    }

    /// Append this Master Key Identifier to every packet, 1 to 128 bytes.
    pub fn with_mki(mut self, mki: &[u8]) -> Result<Self, SrtpError> {
        self.apply_mki(Some(mki))?;
        Ok(self)
    }

    /// What this element protected and what it dropped.
    pub fn stats(&self) -> SrtpEncStats {
        SrtpEncStats {
            packets_protected: self.packets_protected,
            packets_dropped: self.packets_dropped,
        }
    }

    /// Replace the master key mid-run. Packet indices continue, so the stream
    /// is unbroken, and the key-use counters restart.
    pub fn replace_key(
        &mut self,
        policy: SrtpPolicy,
        master_key: &[u8],
        master_salt: &[u8],
    ) -> Result<(), SrtpError> {
        self.keys =
            SrtpKeySettings::from_master_key(&SrtpMasterKey::new(policy, master_key, master_salt)?);
        self.soft_limit_reported = false;
        self.rekey_sender()
    }

    /// How far into its lifetime the current key is, `None` until the first
    /// packet built the sender context.
    pub fn key_usage(&self) -> Option<SrtpKeyUsage> {
        self.sender.as_ref().map(SrtpSender::key_usage)
    }

    /// Set or clear the MKI on the element and on a sender that already exists,
    /// so a mid-run change reaches the next packet.
    fn apply_mki(&mut self, mki: Option<&[u8]>) -> Result<(), SrtpError> {
        let validated = mki.map(validated_mki).transpose()?;
        if let Some(sender) = &mut self.sender {
            sender.set_mki(validated.as_deref())?;
        }
        self.mki = validated;
        Ok(())
    }

    /// Push a key or policy change into a sender that already exists, so a
    /// mid-run property change reaches the next packet with the indices intact.
    fn rekey_sender(&mut self) -> Result<(), SrtpError> {
        let (Some(flow), true) = (self.flow, self.sender.is_some()) else {
            return Ok(());
        };
        let key = self.keys.master_key(flow)?;
        if let Some(sender) = &mut self.sender {
            sender.replace_key(key.policy(), key.master_key(), key.master_salt())?;
        }
        Ok(())
    }

    fn output_caps(&self) -> Caps {
        Caps::ByteStream {
            encoding: self.flow.unwrap_or(SrtpFlow::Rtp).protected_encoding(),
        }
    }

    fn rtcp_mode(&self) -> RtcpProtectionMode {
        if self.encrypt_rtcp {
            RtcpProtectionMode::Encrypt
        } else {
            RtcpProtectionMode::AuthenticateOnly
        }
    }

    /// The sender for this stream, built from the first packet's source.
    /// `None` when the packet is not a valid RTP / RTCP packet to read one from.
    fn sender_for(
        &mut self,
        packet: &[u8],
        flow: SrtpFlow,
    ) -> Result<Option<&mut SrtpSender>, G2gError> {
        if self.sender.is_none() {
            let Some(source) = synchronization_source(packet, flow) else {
                return Ok(None);
            };
            let master_key = self.keys.master_key(flow).map_err(|error| {
                g2g_core::g2g_error!(self, "cannot protect this stream: {error}");
                G2gError::NotConfigured
            })?;
            let sender = Self::build_sender(
                &master_key,
                source,
                self.soft_limits,
                self.replay_window,
                self.allow_repeat_transmission,
                self.mki.as_deref(),
            )
            .map_err(|error| {
                g2g_core::g2g_error!(self, "cannot protect this stream: {error}");
                G2gError::NotConfigured
            })?;
            self.sender = Some(sender);
        }
        Ok(self.sender.as_mut())
    }

    fn build_sender(
        master_key: &SrtpMasterKey,
        synchronization_source: u32,
        soft_limits: SrtpSoftLimits,
        replay_window: usize,
        allow_repeat_transmission: bool,
        mki: Option<&[u8]>,
    ) -> Result<SrtpSender, SrtpError> {
        let mut sender = SrtpSender::new_with_soft_limits(
            master_key.policy(),
            master_key.master_key(),
            master_key.master_salt(),
            synchronization_source,
            soft_limits,
        )?;
        sender.set_replay_window(replay_window)?;
        sender.set_repeat_transmission(allow_repeat_transmission);
        sender.set_mki(mki)?;
        Ok(sender)
    }

    fn report_soft_limit(&mut self) {
        if self.soft_limit_reported {
            return;
        }
        let Some(usage) = self.key_usage() else {
            return;
        };
        if usage.srtp != KeyUsage::SoftLimitReached && usage.srtcp != KeyUsage::SoftLimitReached {
            return;
        }
        self.soft_limit_reported = true;
        g2g_core::g2g_warn!(self, "{SOFT_LIMIT_MESSAGE}");
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::Info(SOFT_LIMIT_MESSAGE.to_string()));
        }
    }

    async fn process_packet(
        &mut self,
        packet: PipelinePacket,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let flow = self.flow.ok_or(G2gError::NotConfigured)?;
        let Some(frame) = forward_until_data_frame(packet, out, self.output_caps()).await? else {
            return Ok(());
        };
        let bytes = frame
            .domain
            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
        let mode = self.rtcp_mode();
        let Some(sender) = self.sender_for(bytes, flow)? else {
            self.packets_dropped += 1;
            g2g_core::g2g_warn!(self, "dropping a frame that is not a valid {flow:?} packet");
            return Ok(());
        };
        match protect_for_flow(sender, flow, bytes, mode) {
            Ok(protected) => {
                self.packets_protected += 1;
                push_frame_bytes(frame, protected, out).await?;
                self.report_soft_limit();
                Ok(())
            }
            Err(error) if is_fatal_send_error(error) => {
                self.packets_dropped += 1;
                g2g_core::g2g_error!(self, "stopping the stream: {error}");
                Err(G2gError::Hardware(HardwareError::Other))
            }
            Err(error) => {
                self.packets_dropped += 1;
                g2g_core::g2g_warn!(self, "dropping a packet: {error}");
                Ok(())
            }
        }
    }
}

impl LogSource for SrtpEnc {
    fn log_category(&self) -> &'static str {
        "srtpenc"
    }
}

impl AsyncElement for SrtpEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SRTP protector",
            "Filter/Network/SRTP",
            "Authenticates and encrypts RTP or RTCP packets, RFC 7714 AES-GCM or RFC 3711 \
             counter mode with HMAC-SHA1",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SRTPENC_PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "key" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                self.keys
                    .set_key_hexadecimal(text)
                    .map_err(|_| PropError::Value)?;
                self.soft_limit_reported = false;
                self.rekey_sender().map_err(|_| PropError::Value)
            }
            "rtp-cipher" | "rtcp-cipher" => {
                let flow = protection_property_flow(name).ok_or(PropError::Unknown)?;
                let text = value.as_str().ok_or(PropError::Type)?;
                let cipher = SrtpCipher::from_property_value(text).ok_or(PropError::Value)?;
                self.keys
                    .set_cipher(flow, cipher)
                    .map_err(|_| PropError::Value)?;
                self.rekey_sender().map_err(|_| PropError::Value)
            }
            "rtp-auth" | "rtcp-auth" => {
                let flow = protection_property_flow(name).ok_or(PropError::Unknown)?;
                let text = value.as_str().ok_or(PropError::Type)?;
                let authentication =
                    SrtpAuthentication::from_property_value(text).ok_or(PropError::Value)?;
                self.keys
                    .set_authentication(flow, authentication)
                    .map_err(|_| PropError::Value)?;
                self.rekey_sender().map_err(|_| PropError::Value)
            }
            "rtcp-encrypt" => {
                self.encrypt_rtcp = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "replay-window-size" => {
                let value = value.as_uint().ok_or(PropError::Type)?;
                let packets = usize::try_from(value).map_err(|_| PropError::Value)?;
                validate_replay_window(packets).map_err(|_| PropError::Value)?;
                self.replay_window = packets;
                Ok(())
            }
            "allow-repeat-tx" => {
                self.allow_repeat_transmission = value.as_bool().ok_or(PropError::Type)?;
                if let Some(sender) = &mut self.sender {
                    sender.set_repeat_transmission(self.allow_repeat_transmission);
                }
                Ok(())
            }
            "mki" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                if text.is_empty() {
                    return self.apply_mki(None).map_err(|_| PropError::Value);
                }
                let mki = decode_hexadecimal(text).ok_or(PropError::Value)?;
                self.apply_mki(Some(&mki)).map_err(|_| PropError::Value)
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            // Never hands the key back out.
            "key" => Some(PropValue::Str(String::new())),
            "rtp-cipher" | "rtcp-cipher" => Some(PropValue::Str(
                self.keys
                    .cipher(protection_property_flow(name)?)
                    .as_str()
                    .into(),
            )),
            "rtp-auth" | "rtcp-auth" => Some(PropValue::Str(
                self.keys
                    .authentication(protection_property_flow(name)?)
                    .as_str()
                    .into(),
            )),
            "rtcp-encrypt" => Some(PropValue::Bool(self.encrypt_rtcp)),
            "replay-window-size" => Some(PropValue::Uint(self.replay_window as u64)),
            "allow-repeat-tx" => Some(PropValue::Bool(self.allow_repeat_transmission)),
            // The MKI is not secret: it names the key, it is not the key.
            "mki" => Some(PropValue::Str(
                self.mki
                    .as_deref()
                    .map(encode_hexadecimal)
                    .unwrap_or_default(),
            )),
            _ => None,
        }
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        SrtpFlow::from_plain_caps(upstream_caps).ok_or(G2gError::CapsMismatch)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Mapping(
            [SrtpFlow::Rtp, SrtpFlow::Rtcp]
                .into_iter()
                .map(|flow| {
                    (
                        CapsSet::one(Caps::ByteStream {
                            encoding: flow.plain_encoding(),
                        }),
                        CapsSet::one(Caps::ByteStream {
                            encoding: flow.protected_encoding(),
                        }),
                    )
                })
                .collect(),
        )
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let flow = SrtpFlow::from_plain_caps(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        if !self.keys.has_key() {
            g2g_core::g2g_error!(self, "no master key: set the `key` property");
            return Err(G2gError::NotConfigured);
        }
        let policy = self.keys.master_key(flow).map_err(|error| {
            g2g_core::g2g_error!(self, "{error}");
            G2gError::NotConfigured
        })?;
        if !policy.policy().is_authenticated() {
            g2g_core::g2g_warn!(self, "{UNAUTHENTICATED_POLICY_WARNING}");
        }
        if !self.soft_limits.is_valid() {
            g2g_core::g2g_error!(
                self,
                "soft limits {:?} leave no room below the RFC 3711 hard limits",
                self.soft_limits
            );
            return Err(G2gError::NotConfigured);
        }
        self.flow = Some(flow);
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(self.process_packet(packet, out))
    }
}

impl PadTemplates for SrtpEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(flow_caps(
                SrtpFlow::plain_encoding,
            ))),
            PadTemplate::source(CapsSet::from_alternatives(flow_caps(
                SrtpFlow::protected_encoding,
            ))),
        ])
    }
}

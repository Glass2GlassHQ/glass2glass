//! `srtpdec`: authenticates and decrypts one SRTP or SRTCP flow.
//!
//! Which flow an instance carries comes from the caps the link settled on
//! (`application/x-srtp` or `application/x-srtcp`). Every synchronization source
//! gets its own context, keyed either from the `key` and `roc` properties (one
//! key for any source, the way gst's `srtpdec` uses the key it was given) or
//! from an [`SrtpKeyProvider`] the application installs. A packet that fails to
//! authenticate, repeats an index, or names a source with no key is dropped and
//! the stream continues.

use core::fmt;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use zeroize::Zeroizing;

use g2g_core::log::LogSource;
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::srtp::{
    flow_caps, forward_until_data_frame, protection_property_flow, push_frame_bytes,
    SrtpAuthentication, SrtpCipher, SrtpError, SrtpFlow, SrtpKeyProvider, SrtpKeySettings,
    SrtpKeyingMaterial, SrtpMasterKey, SrtpPolicy, SrtpReceiverSet, SrtpStreamStats,
    KEY_PROPERTY_BLURB, RTCP_AUTHENTICATION_PROPERTY, RTCP_CIPHER_PROPERTY,
    RTP_AUTHENTICATION_PROPERTY, RTP_CIPHER_PROPERTY, UNAUTHENTICATED_POLICY_WARNING,
};

const SRTPDEC_PROPERTIES: &[PropertySpec] = &[
    PropertySpec::new("key", PropKind::Str, KEY_PROPERTY_BLURB),
    RTP_CIPHER_PROPERTY,
    RTCP_CIPHER_PROPERTY,
    RTP_AUTHENTICATION_PROPERTY,
    RTCP_AUTHENTICATION_PROPERTY,
    PropertySpec::new(
        "roc",
        PropKind::Uint,
        "rollover counter the next context starts from, for joining a stream already past a \
         sequence-number wrap",
    )
    .with_range("0", "4294967295")
    .with_default("0"),
    PropertySpec::new(
        "replay-window-size",
        PropKind::Uint,
        "size of the replay protection window, in packets. Sizes every context created \
         afterwards; the ones already running keep their history",
    )
    .with_range("64", "32768")
    .with_default("128"),
];

/// What `srtpdec` took in and handed on, and where each of its contexts stands.
/// The counts run from the first packet and a rekey does not reset them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SrtpDecStats {
    pub packets_received: u64,
    pub packets_dropped: u64,
    pub streams: Vec<SrtpStreamStats>,
}

/// The keys `srtpdec` hands each new context. An installed provider answers for
/// every source; otherwise the `key` property answers for all of them, with the
/// `roc` property as their starting rollover counter.
#[derive(Default)]
struct SrtpDecKeys {
    settings: SrtpKeySettings,
    /// The flow the caps settled on, so the `rtp-` or `rtcp-` half of the
    /// protection properties is the one every context is keyed under.
    flow: Option<SrtpFlow>,
    rollover_counter: u32,
    provider: Option<Box<dyn SrtpKeyProvider + Send>>,
}

impl fmt::Debug for SrtpDecKeys {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SrtpDecKeys")
            .field("settings", &self.settings)
            .field("rollover_counter", &self.rollover_counter)
            .field("has_provider", &self.provider.is_some())
            .finish_non_exhaustive()
    }
}

impl SrtpDecKeys {
    /// The key every context is built from, `None` until the caps named the
    /// flow whose policy applies or while a provider answers instead.
    fn master_key(&self) -> Option<SrtpMasterKey> {
        self.settings.master_key(self.flow?).ok()
    }
}

impl SrtpKeyProvider for SrtpDecKeys {
    fn keys_for(&mut self, synchronization_source: u32) -> Vec<SrtpKeyingMaterial> {
        if let Some(provider) = &mut self.provider {
            return provider.keys_for(synchronization_source);
        }
        self.master_key()
            .and_then(|key| key.keying_material(self.rollover_counter).ok())
            .into_iter()
            .collect()
    }
}

/// Recovers one system-memory RTP or RTCP packet per frame, one context per
/// synchronization source.
#[derive(Debug)]
pub struct SrtpDec {
    receivers: SrtpReceiverSet<SrtpDecKeys>,
    flow: Option<SrtpFlow>,
    packets_received: u64,
    packets_dropped: u64,
}

impl Default for SrtpDec {
    fn default() -> Self {
        Self {
            receivers: SrtpReceiverSet::new(SrtpDecKeys::default()),
            flow: None,
            packets_received: 0,
            packets_dropped: 0,
        }
    }
}

impl SrtpDec {
    pub fn new(
        policy: SrtpPolicy,
        master_key: &[u8],
        master_salt: &[u8],
    ) -> Result<Self, SrtpError> {
        let mut element = Self::default();
        element.apply_key(SrtpMasterKey::new(policy, master_key, master_salt)?)?;
        Ok(element)
    }

    /// Recover the RTP flow under this cipher and authentication instead of the
    /// pair the `key` length picks. Read on an RTP instance.
    pub fn with_rtp_protection(
        mut self,
        cipher: SrtpCipher,
        authentication: SrtpAuthentication,
    ) -> Result<Self, SrtpError> {
        let keys = self.receivers.key_provider_mut();
        keys.settings.set_cipher(SrtpFlow::Rtp, cipher)?;
        keys.settings
            .set_authentication(SrtpFlow::Rtp, authentication)?;
        Ok(self)
    }

    /// The same for the RTCP flow, read on an RTCP instance.
    pub fn with_rtcp_protection(
        mut self,
        cipher: SrtpCipher,
        authentication: SrtpAuthentication,
    ) -> Result<Self, SrtpError> {
        let keys = self.receivers.key_provider_mut();
        keys.settings.set_cipher(SrtpFlow::Rtcp, cipher)?;
        keys.settings
            .set_authentication(SrtpFlow::Rtcp, authentication)?;
        Ok(self)
    }

    /// Answer every source from this provider instead of the `key` property, so
    /// an application that learns keys out of band (a signalling channel, a
    /// DTLS-SRTP handshake) can serve a different key per source.
    pub fn with_key_provider(mut self, provider: Box<dyn SrtpKeyProvider + Send>) -> Self {
        self.receivers.key_provider_mut().provider = Some(provider);
        self
    }

    /// The rollover counter a context created later starts from.
    pub fn with_rollover_counter(mut self, rollover_counter: u32) -> Self {
        self.receivers.key_provider_mut().rollover_counter = rollover_counter;
        self
    }

    /// The replay window a context created later gets, 64..=32768 packets.
    pub fn with_replay_window(mut self, packets: usize) -> Result<Self, SrtpError> {
        self.receivers.set_replay_window(packets)?;
        Ok(self)
    }

    /// What this element took in and handed on, plus one entry per context.
    pub fn stats(&self) -> SrtpDecStats {
        SrtpDecStats {
            packets_received: self.packets_received,
            packets_dropped: self.packets_dropped,
            streams: self.receivers.stream_statistics(),
        }
    }

    /// Replace the master key mid-run: every existing context is re-keyed with
    /// its packet indices intact, and contexts created later take the same key.
    /// This is the property route, so it also drops an installed provider.
    pub fn replace_key(
        &mut self,
        policy: SrtpPolicy,
        master_key: &[u8],
        master_salt: &[u8],
    ) -> Result<(), SrtpError> {
        self.apply_key(SrtpMasterKey::new(policy, master_key, master_salt)?)
    }

    fn apply_key(&mut self, master_key: SrtpMasterKey) -> Result<(), SrtpError> {
        let keys = self.receivers.key_provider_mut();
        keys.provider = None;
        keys.settings = SrtpKeySettings::from_master_key(&master_key);
        self.rekey_contexts()
    }

    /// Re-key every context that already exists, keeping its packet indices, so
    /// a key or policy change mid-run reaches the next packet.
    fn rekey_contexts(&mut self) -> Result<(), SrtpError> {
        let Some(master_key) = self.receivers.key_provider().master_key() else {
            return Ok(());
        };
        let policy = master_key.policy();
        let key = Zeroizing::new(master_key.master_key().to_vec());
        let salt = Zeroizing::new(master_key.master_salt().to_vec());
        for source in self.receivers.synchronization_sources() {
            self.receivers.replace_key(source, policy, &key, &salt)?;
        }
        Ok(())
    }

    fn has_key(&self) -> bool {
        let keys = self.receivers.key_provider();
        keys.settings.has_key() || keys.provider.is_some()
    }

    fn rollover_counter(&self) -> u32 {
        self.receivers.key_provider().rollover_counter
    }

    fn output_caps(&self) -> Caps {
        Caps::ByteStream {
            encoding: self.flow.unwrap_or(SrtpFlow::Rtp).plain_encoding(),
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
        self.packets_received += 1;
        let recovered = match flow {
            SrtpFlow::Rtp => self.receivers.unprotect_rtp(bytes),
            SrtpFlow::Rtcp => self.receivers.unprotect_rtcp(bytes),
        };
        match recovered {
            Ok(recovered) => push_frame_bytes(frame, recovered, out).await,
            Err(error) => {
                self.packets_dropped += 1;
                g2g_core::g2g_warn!(self, "dropping a packet: {error}");
                Ok(())
            }
        }
    }
}

impl LogSource for SrtpDec {
    fn log_category(&self) -> &'static str {
        "srtpdec"
    }
}

impl AsyncElement for SrtpDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SRTP unprotector",
            "Filter/Network/SRTP",
            "Authenticates and decrypts SRTP or SRTCP packets, RFC 7714 AES-GCM or RFC 3711 \
             counter mode with HMAC-SHA1",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SRTPDEC_PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "key" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                let keys = self.receivers.key_provider_mut();
                keys.provider = None;
                keys.settings
                    .set_key_hexadecimal(text)
                    .map_err(|_| PropError::Value)?;
                self.rekey_contexts().map_err(|_| PropError::Value)
            }
            "rtp-cipher" | "rtcp-cipher" => {
                let flow = protection_property_flow(name).ok_or(PropError::Unknown)?;
                let text = value.as_str().ok_or(PropError::Type)?;
                let cipher = SrtpCipher::from_property_value(text).ok_or(PropError::Value)?;
                self.receivers
                    .key_provider_mut()
                    .settings
                    .set_cipher(flow, cipher)
                    .map_err(|_| PropError::Value)?;
                self.rekey_contexts().map_err(|_| PropError::Value)
            }
            "rtp-auth" | "rtcp-auth" => {
                let flow = protection_property_flow(name).ok_or(PropError::Unknown)?;
                let text = value.as_str().ok_or(PropError::Type)?;
                let authentication =
                    SrtpAuthentication::from_property_value(text).ok_or(PropError::Value)?;
                self.receivers
                    .key_provider_mut()
                    .settings
                    .set_authentication(flow, authentication)
                    .map_err(|_| PropError::Value)?;
                self.rekey_contexts().map_err(|_| PropError::Value)
            }
            "roc" => {
                let value = value.as_uint().ok_or(PropError::Type)?;
                self.receivers.key_provider_mut().rollover_counter =
                    u32::try_from(value).map_err(|_| PropError::Value)?;
                Ok(())
            }
            "replay-window-size" => {
                let value = value.as_uint().ok_or(PropError::Type)?;
                let packets = usize::try_from(value).map_err(|_| PropError::Value)?;
                self.receivers
                    .set_replay_window(packets)
                    .map_err(|_| PropError::Value)
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            // Never hands the key back out.
            "key" => Some(PropValue::Str(String::new())),
            "rtp-cipher" | "rtcp-cipher" => Some(PropValue::Str(
                self.receivers
                    .key_provider()
                    .settings
                    .cipher(protection_property_flow(name)?)
                    .as_str()
                    .into(),
            )),
            "rtp-auth" | "rtcp-auth" => Some(PropValue::Str(
                self.receivers
                    .key_provider()
                    .settings
                    .authentication(protection_property_flow(name)?)
                    .as_str()
                    .into(),
            )),
            "roc" => Some(PropValue::Uint(u64::from(self.rollover_counter()))),
            "replay-window-size" => Some(PropValue::Uint(self.receivers.replay_window() as u64)),
            _ => None,
        }
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        SrtpFlow::from_protected_caps(upstream_caps).ok_or(G2gError::CapsMismatch)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Mapping(
            [SrtpFlow::Rtp, SrtpFlow::Rtcp]
                .into_iter()
                .map(|flow| {
                    (
                        CapsSet::one(Caps::ByteStream {
                            encoding: flow.protected_encoding(),
                        }),
                        CapsSet::one(Caps::ByteStream {
                            encoding: flow.plain_encoding(),
                        }),
                    )
                })
                .collect(),
        )
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let flow = SrtpFlow::from_protected_caps(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        if !self.has_key() {
            g2g_core::g2g_error!(
                self,
                "no key material: set the `key` property or install a key provider"
            );
            return Err(G2gError::NotConfigured);
        }
        self.receivers.key_provider_mut().flow = Some(flow);
        let keys = self.receivers.key_provider();
        if keys.provider.is_none() {
            let policy = keys.settings.policy(flow).map_err(|error| {
                g2g_core::g2g_error!(self, "{error}");
                G2gError::NotConfigured
            })?;
            if !policy.is_authenticated() {
                g2g_core::g2g_warn!(self, "{UNAUTHENTICATED_POLICY_WARNING}");
            }
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

impl PadTemplates for SrtpDec {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(flow_caps(
                SrtpFlow::protected_encoding,
            ))),
            PadTemplate::source(CapsSet::from_alternatives(flow_caps(
                SrtpFlow::plain_encoding,
            ))),
        ])
    }
}

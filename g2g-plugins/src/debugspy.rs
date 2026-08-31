//! Buffer spy (`debugspy`). A 1:1 passthrough that hashes each buffer.
//! `checksum-type` picks the digest; `silent=true` keeps the hash off the log.
//! Reuses [`crate::checksumsink::digest_hex`] so the two debug hashers cannot
//! drift.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_debug, AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::checksumsink::{digest_hex, DEFAULT_HASH, HASH_VALUES};

/// # Example
///
/// ```no_run
/// use g2g_plugins::debugspy::DebugSpy;
///
/// let spy = DebugSpy::new();
/// ```
#[derive(Debug)]
pub struct DebugSpy {
    checksum_type: &'static str,
    silent: bool,
    last: String,
    seen: u64,
    configured: bool,
    log_name: LogName,
}

impl Default for DebugSpy {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugSpy {
    pub fn new() -> Self {
        Self {
            checksum_type: DEFAULT_HASH,
            silent: false,
            last: String::new(),
            seen: 0,
            configured: false,
            log_name: LogName::new(),
        }
    }

    /// Most recent digest hex, empty before the first buffer.
    pub fn last_checksum(&self) -> &str {
        &self.last
    }

    pub fn seen(&self) -> u64 {
        self.seen
    }
}

impl AsyncElement for DebugSpy {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "DebugSpy",
            "Filter/Analyzer/Debug",
            "A spy element that can provide information on buffers going through it",
            "g2g",
        )
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if let Some(bytes) = frame.domain.as_system_slice() {
                        self.last = digest_hex(self.checksum_type, bytes);
                        self.seen += 1;
                        if !self.silent {
                            g2g_debug!(self, "{}", self.last);
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DEBUGSPY_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "checksum-type" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                self.checksum_type = DEBUGSPY_PROPS
                    .iter()
                    .find(|spec| spec.name == name)
                    .and_then(|spec| spec.enum_nicks().find(|nick| *nick == text))
                    .ok_or(PropError::Value)?;
            }
            "silent" => self.silent = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "checksum-type" => Some(PropValue::Str(self.checksum_type.to_string())),
            "silent" => Some(PropValue::Bool(self.silent)),
            _ => None,
        }
    }
}

impl LogSource for DebugSpy {
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

static DEBUGSPY_PROPS: &[PropertySpec] = &[
    PropertySpec::new("checksum-type", PropKind::Str, "Checksum algorithm to use")
        .with_enum_values(HASH_VALUES)
        .with_default(DEFAULT_HASH),
    PropertySpec::new("silent", PropKind::Bool, "Produce verbose output ?").with_default("false"),
];

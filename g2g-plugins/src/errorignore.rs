//! Error ignore: passes data through and turns a failure from downstream into
//! one the run survives, the gst `errorignore` analog. Put it above a branch
//! that is allowed to die (a preview sink on a display that went away) so the
//! rest of the pipeline keeps running.
//!
//! gst decides on the `GstFlowReturn` its push gave back; a g2g push returns a
//! [`G2gError`] instead, so each ignore property covers the errors that mean
//! what its flow return means:
//!
//! | gst flow return | property | g2g error |
//! | :-- | :-- | :-- |
//! | `GST_FLOW_NOT_NEGOTIATED` | `ignore-notnegotiated` | [`G2gError::CapsMismatch`], [`G2gError::FixationFailed`] |
//! | `GST_FLOW_NOT_LINKED`, `GST_FLOW_EOS` | `ignore-notlinked`, `ignore-eos` | [`G2gError::Shutdown`] |
//! | `GST_FLOW_ERROR` | `ignore-error` | every other error |
//!
//! g2g has one error for "downstream is gone" where gst has two flow returns, so
//! `ignore-notlinked` and `ignore-eos` cover the same [`G2gError::Shutdown`] and
//! either one ignores it. `convert-to` names the error returned in place of an
//! ignored one, and its `eos` and `not-linked` values coincide for the same
//! reason; `ok` returns success, which is the value that keeps a run going.
//!
//! Every packet is offered downstream, one after a failed push included: like
//! gst's, this element decides on the push it just made and remembers nothing,
//! so a branch that comes back to life is pushed to again.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_debug, AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError,
    HardwareError, OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

/// The `convert-to` choices, gst's flow-return nicks that have a g2g meaning.
const CONVERT_TO_VALUES: &str = "ok | error | not-linked | not-negotiated | eos";
/// gst `errorignore`'s `convert-to` default.
const DEFAULT_CONVERT_TO: &str = "not-linked";

/// # Example
///
/// ```no_run
/// use g2g_plugins::errorignore::ErrorIgnore;
///
/// // gst-launch equivalent: errorignore ignore-error=true convert-to=ok
/// let element = ErrorIgnore::new();
/// ```
#[derive(Debug)]
pub struct ErrorIgnore {
    ignore_error: bool,
    ignore_notlinked: bool,
    ignore_notnegotiated: bool,
    ignore_eos: bool,
    convert_to: &'static str,
    ignored: u64,
    configured: bool,
    log_name: LogName,
}

impl Default for ErrorIgnore {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorIgnore {
    pub fn new() -> Self {
        Self {
            ignore_error: true,
            ignore_notlinked: false,
            ignore_notnegotiated: true,
            ignore_eos: false,
            convert_to: DEFAULT_CONVERT_TO,
            ignored: 0,
            configured: false,
            log_name: LogName::new(),
        }
    }

    /// Errors converted rather than passed on.
    pub fn ignored(&self) -> u64 {
        self.ignored
    }

    /// Whether this error is one the properties say to ignore.
    fn ignores(&self, error: &G2gError) -> bool {
        match error {
            G2gError::Shutdown => self.ignore_notlinked || self.ignore_eos,
            G2gError::CapsMismatch | G2gError::FixationFailed => self.ignore_notnegotiated,
            _ => self.ignore_error,
        }
    }

    /// What `convert-to` names, as the result this element returns.
    fn converted(&self) -> Result<(), G2gError> {
        match self.convert_to {
            "ok" => Ok(()),
            "error" => Err(G2gError::Hardware(HardwareError::Other)),
            "not-negotiated" => Err(G2gError::CapsMismatch),
            // "not-linked" and "eos" both mean downstream is gone.
            _ => Err(G2gError::Shutdown),
        }
    }

    /// Push `packet`, converting a failure the properties cover.
    async fn forward(
        &mut self,
        packet: PipelinePacket,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        match out.push(packet).await {
            // The reverse-channel outcome is the producer's business; this
            // element only reacts to the failure.
            Ok(_) => Ok(()),
            Err(error) => {
                if !self.ignores(&error) {
                    return Err(error);
                }
                self.ignored += 1;
                g2g_debug!(
                    self,
                    "converting {:?} from downstream to {}",
                    error,
                    self.convert_to
                );
                self.converted()
            }
        }
    }
}

impl AsyncElement for ErrorIgnore {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ErrorIgnore",
            "Generic",
            "Passes data through, converting a failure from downstream",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Wildcard pass-through: the data is untouched whatever the media type is.
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
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            self.forward(packet, out).await
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        ERRORIGNORE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if name == "convert-to" {
            let text = value.as_str().ok_or(PropError::Type)?;
            let known = ERRORIGNORE_PROPS
                .iter()
                .find(|s| s.name == name)
                .and_then(|s| s.enum_nicks().find(|nick| *nick == text))
                .ok_or(PropError::Value)?;
            self.convert_to = known;
            return Ok(());
        }
        let on = value.as_bool().ok_or(PropError::Type)?;
        match name {
            "ignore-error" => self.ignore_error = on,
            "ignore-notlinked" => self.ignore_notlinked = on,
            "ignore-notnegotiated" => self.ignore_notnegotiated = on,
            "ignore-eos" => self.ignore_eos = on,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "ignore-error" => Some(PropValue::Bool(self.ignore_error)),
            "ignore-notlinked" => Some(PropValue::Bool(self.ignore_notlinked)),
            "ignore-notnegotiated" => Some(PropValue::Bool(self.ignore_notnegotiated)),
            "ignore-eos" => Some(PropValue::Bool(self.ignore_eos)),
            "convert-to" => Some(PropValue::Str(self.convert_to.into())),
            _ => None,
        }
    }
}

impl LogSource for ErrorIgnore {
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

/// `ErrorIgnore`'s settable properties, named and defaulted as gst
/// `errorignore`.
static ERRORIGNORE_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "ignore-error",
        PropKind::Bool,
        "ignore a failure none of the other properties name",
    )
    .with_default("true"),
    PropertySpec::new(
        "ignore-notlinked",
        PropKind::Bool,
        "ignore downstream being gone",
    )
    .with_default("false"),
    PropertySpec::new(
        "ignore-notnegotiated",
        PropKind::Bool,
        "ignore a caps mismatch downstream",
    )
    .with_default("true"),
    PropertySpec::new(
        "ignore-eos",
        PropKind::Bool,
        "ignore downstream having ended",
    )
    .with_default("false"),
    PropertySpec::new(
        "convert-to",
        PropKind::Str,
        "what an ignored failure is reported as instead",
    )
    .with_enum_values(CONVERT_TO_VALUES)
    .with_default(DEFAULT_CONVERT_TO),
];

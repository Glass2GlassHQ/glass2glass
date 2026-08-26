//! Valve: a pass-through that discards data while it is closed (`drop=true`),
//! the gst `valve` analog. Use it to mute one branch of a tee without tearing
//! the graph down.
//!
//! `Eos` and `Flush` always pass, closed or not: a swallowed `Eos` would leave
//! the runner waiting for a stream that already ended.
//!
//! Per the transform contract (see `run_source_transform_sink`), this element
//! does NOT emit `Eos` itself: the runner forwards the EOS sentinel after
//! `process(Eos)` returns.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Segment,
};

/// What a closed valve does with the ordered control packets that carry stream
/// state (`CapsChanged`, `Segment`), the g2g half of gst's `drop-mode`. gst's
/// third mode, `transform-to-gap`, has no analog: g2g has no gap packet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DropMode {
    /// Hold the latest caps and segment back while closed, and emit them when
    /// the valve reopens, so downstream never sees state for frames it did not
    /// get. gst's default.
    #[default]
    DropAll,
    /// Forward caps and segment as they arrive even while closed, so downstream
    /// tracks the stream's state through the closed stretch.
    ForwardStickyEvents,
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::valve::Valve;
///
/// // gst-launch equivalent: valve drop=true
/// let element = Valve::new();
/// assert_eq!(element.dropped(), 0);
/// ```
#[derive(Debug, Default)]
pub struct Valve {
    closed: bool,
    drop_mode: DropMode,
    dropped: u64,
    configured: bool,
    /// The caps and segment withheld by a `drop-all` close, latest wins, emitted
    /// when the valve reopens.
    held_caps: Option<Caps>,
    held_segment: Option<Segment>,
}

impl Valve {
    pub fn new() -> Self {
        Self::default()
    }

    /// Frames discarded while closed.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}

impl AsyncElement for Valve {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Valve",
            "Filter",
            "Drops buffers and events or lets them through",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Wildcard pass-through: a valve constrains neither side, whatever the
    /// surrounding endpoints settle on flows through it.
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
            if !self.closed {
                if let Some(caps) = self.held_caps.take() {
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                if let Some(segment) = self.held_segment.take() {
                    out.push(PipelinePacket::Segment(segment)).await?;
                }
            }
            let hold_state = self.closed && self.drop_mode == DropMode::DropAll;
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if self.closed {
                        self.dropped += 1;
                    } else {
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                    }
                }
                PipelinePacket::CapsChanged(caps) if hold_state => self.held_caps = Some(caps),
                PipelinePacket::Segment(segment) if hold_state => self.held_segment = Some(segment),
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        VALVE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "drop" => {
                self.closed = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "drop-mode" => {
                let mode = value.as_str().ok_or(PropError::Type)?;
                self.drop_mode = drop_mode_from_str(mode).ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "drop" => Some(PropValue::Bool(self.closed)),
            "drop-mode" => Some(PropValue::Str(drop_mode_to_str(self.drop_mode).into())),
            _ => None,
        }
    }
}

/// `Valve`'s settable properties, named and defaulted as gst `valve`.
static VALVE_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "drop",
        PropKind::Bool,
        "discard frames instead of forwarding them",
    )
    .with_default("false"),
    PropertySpec::new(
        "drop-mode",
        PropKind::Str,
        "what a closed valve does with caps and segment",
    )
    .with_enum_values("drop-all | forward-sticky-events")
    .with_default("drop-all"),
];

/// Parse a `drop-mode` property string. The names are gst's enum nicks.
fn drop_mode_from_str(s: &str) -> Option<DropMode> {
    match s {
        "drop-all" => Some(DropMode::DropAll),
        "forward-sticky-events" => Some(DropMode::ForwardStickyEvents),
        _ => None,
    }
}

/// The `drop-mode` property string for a [`DropMode`].
fn drop_mode_to_str(mode: DropMode) -> &'static str {
    match mode {
        DropMode::DropAll => "drop-all",
        DropMode::ForwardStickyEvents => "forward-sticky-events",
    }
}

//! Caps setter: rewrites the caps travelling with a stream without touching a
//! byte of it, the gst `capssetter` analog. Use it to correct a source that
//! declares the wrong framerate or format.
//!
//! gst caps are field maps, so gst can merge one map over another. g2g caps are
//! typed variants, so "merge" is spelled out per variant:
//!
//! - `Caps::RawVideo`: `format`, `width`, `height`, `framerate` and
//!   `interlace-mode` are overwritten one by one, whichever of them the `caps`
//!   property names.
//! - `Caps::Audio`: `format`, `channels` and `rate`, the same way.
//! - every other variant has no separately settable field, so it is replaced
//!   whole and the `caps` property has to describe exactly one alternative.
//!
//! `join=true` (the default) requires the incoming caps to be the same variant
//! as the property's and fails the run otherwise. `join=false` leaves caps of
//! another variant untouched instead. `replace=true` overrides both: the
//! property's caps become the output whatever arrived.
//!
//! Which fields to overwrite is read from the property text rather than the
//! parsed caps, because the shared parser fills an unnamed audio field with a
//! default (2 channels, 48 kHz) rather than a wildcard.
//!
//! Negotiation is a pass-through (like `typefind`): the rewrite happens at
//! runtime, on the negotiated caps and on every later `CapsChanged`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::capsfilter::parse_caps_set;

/// The caps-string field names this element can overwrite one at a time.
const FORMAT_FIELD: &str = "format";
const WIDTH_FIELD: &str = "width";
const HEIGHT_FIELD: &str = "height";
const FRAMERATE_FIELD: &str = "framerate";
const INTERLACE_FIELD: &str = "interlace-mode";
const CHANNELS_FIELD: &str = "channels";
const RATE_FIELD: &str = "rate";

/// # Example
///
/// ```no_run
/// use g2g_plugins::capssetter::CapsSetter;
///
/// // gst-launch equivalent: capssetter caps="video/x-raw,framerate=60/1"
/// let element = CapsSetter::new();
/// ```
#[derive(Debug)]
pub struct CapsSetter {
    /// The caps the `caps` property named. `None` until it is set, which makes
    /// the element a plain pass-through.
    setting: Option<Caps>,
    /// Whether the property text described exactly one caps: a description with
    /// alternatives (`format={nv12,i420}`, or no format at all) can only supply
    /// the fields it names, never a whole replacement.
    single: bool,
    /// The field names the property text carries.
    fields: Vec<String>,
    caps_str: String,
    join: bool,
    replace: bool,
    configured: bool,
    /// The negotiated caps, held until the first frame so the rewritten ones
    /// reach downstream even when the runner sends no `CapsChanged` of its own.
    negotiated: Option<Caps>,
    /// The caps last sent downstream, so an unchanged one is not re-emitted.
    declared: Option<Caps>,
}

impl Default for CapsSetter {
    fn default() -> Self {
        Self::new()
    }
}

impl CapsSetter {
    pub fn new() -> Self {
        Self {
            setting: None,
            single: false,
            fields: Vec::new(),
            caps_str: String::new(),
            join: true,
            replace: false,
            configured: false,
            negotiated: None,
            declared: None,
        }
    }

    /// Whether the `caps` property text names this field.
    fn names(&self, field: &str) -> bool {
        self.fields.iter().any(|f| f == field)
    }

    /// The caps `incoming` becomes on the way out.
    fn rewrite(&self, incoming: &Caps) -> Result<Caps, G2gError> {
        let Some(setting) = &self.setting else {
            return Ok(incoming.clone());
        };
        if self.replace {
            return self.whole(setting);
        }
        match (incoming, setting) {
            (
                Caps::RawVideo {
                    format,
                    width,
                    height,
                    framerate,
                    interlace,
                    ..
                },
                Caps::RawVideo {
                    format: set_format,
                    width: set_width,
                    height: set_height,
                    framerate: set_framerate,
                    interlace: set_interlace,
                    ..
                },
            ) => Ok(Caps::RawVideo {
                format: if self.names(FORMAT_FIELD) {
                    *set_format
                } else {
                    *format
                },
                width: if self.names(WIDTH_FIELD) {
                    set_width.clone()
                } else {
                    width.clone()
                },
                height: if self.names(HEIGHT_FIELD) {
                    set_height.clone()
                } else {
                    height.clone()
                },
                framerate: if self.names(FRAMERATE_FIELD) {
                    set_framerate.clone()
                } else {
                    framerate.clone()
                },
                interlace: if self.names(INTERLACE_FIELD) {
                    *set_interlace
                } else {
                    *interlace
                },
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            }),
            (
                Caps::Audio {
                    format,
                    channels,
                    sample_rate,
                    ..
                },
                Caps::Audio {
                    format: set_format,
                    channels: set_channels,
                    sample_rate: set_rate,
                    ..
                },
            ) => Ok(Caps::Audio {
                format: if self.names(FORMAT_FIELD) {
                    *set_format
                } else {
                    *format
                },
                channels: if self.names(CHANNELS_FIELD) {
                    *set_channels
                } else {
                    *channels
                },
                sample_rate: if self.names(RATE_FIELD) {
                    *set_rate
                } else {
                    *sample_rate
                },
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            }),
            // Same variant, no field-by-field merge to do.
            _ if core::mem::discriminant(incoming) == core::mem::discriminant(setting) => {
                self.whole(setting)
            }
            // A different variant: `join` decides whether that is a mistake or a
            // stream this setter has nothing to say about.
            _ if self.join => Err(G2gError::CapsMismatch),
            _ => Ok(incoming.clone()),
        }
    }

    /// The property's caps as a whole replacement, which needs it to describe
    /// exactly one.
    fn whole(&self, setting: &Caps) -> Result<Caps, G2gError> {
        if !self.single {
            return Err(G2gError::CapsMismatch);
        }
        Ok(setting.clone())
    }

    /// Rewrite `caps` and send the result on, unless it is what downstream
    /// already has.
    async fn declare(&mut self, caps: &Caps, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let rewritten = self.rewrite(caps)?;
        if self.declared.as_ref() == Some(&rewritten) {
            return Ok(());
        }
        out.push(PipelinePacket::CapsChanged(rewritten.clone()))
            .await?;
        self.declared = Some(rewritten);
        Ok(())
    }
}

impl AsyncElement for CapsSetter {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "CapsSetter",
            "Generic",
            "Overwrites the caps of a stream, passing the data through",
            "g2g",
        )
    }

    /// Pass-through at negotiation: the rewrite happens on the data path, so
    /// upstream and downstream negotiate as if this element were an identity.
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Fail here rather than at the first frame if the rewrite cannot apply.
        self.rewrite(absolute_caps)?;
        self.negotiated = Some(absolute_caps.clone());
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if let Some(caps) = self.negotiated.take() {
                        self.declare(&caps, out).await?;
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.negotiated = None;
                    self.declare(&caps, out).await?;
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
        CAPSSETTER_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "caps" => {
                let desc = value.as_str().ok_or(PropError::Type)?;
                let set = parse_caps_set(desc).ok_or(PropError::Value)?;
                let alternatives = set.alternatives();
                let first = alternatives.first().ok_or(PropError::Value)?;
                let fields = named_fields(desc);
                let single = alternatives.len() == 1;
                // A named `format` that expands to several is a list, and there
                // is no picking one to write.
                if fields.iter().any(|f| f == FORMAT_FIELD) && !single {
                    return Err(PropError::Value);
                }
                self.setting = Some(first.clone());
                self.single = single;
                self.fields = fields;
                self.caps_str = desc.to_string();
                Ok(())
            }
            "join" => {
                self.join = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "replace" => {
                self.replace = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "caps" if !self.caps_str.is_empty() => Some(PropValue::Str(self.caps_str.clone())),
            "join" => Some(PropValue::Bool(self.join)),
            "replace" => Some(PropValue::Bool(self.replace)),
            _ => None,
        }
    }
}

/// The field names a caps description sets. A name inside a `{a,b}` list is not
/// one: only the text before an `=` counts.
fn named_fields(desc: &str) -> Vec<String> {
    desc.split(',')
        .skip(1)
        .filter_map(|part| part.split_once('='))
        .map(|(key, _)| key.trim().to_string())
        .collect()
}

/// `CapsSetter`'s settable properties, named and defaulted as gst `capssetter`.
static CAPSSETTER_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "caps",
        PropKind::Str,
        "caps to set, gst-launch syntax: e.g. video/x-raw,framerate=60/1",
    ),
    PropertySpec::new(
        "join",
        PropKind::Bool,
        "fail on incoming caps of another media type instead of passing them through",
    )
    .with_default("true"),
    PropertySpec::new(
        "replace",
        PropKind::Bool,
        "replace the incoming caps outright instead of overwriting fields",
    )
    .with_default("false"),
];

//! Tag inject: posts a hand-written tag list on the bus, the gst `taginject`
//! analog. Use it to give a stream a title / artist a muxer downstream can
//! write, or to exercise an application's tag handling without a tagged file.
//!
//! `tags` takes gst's taglist syntax, parsed by [`crate::tagproperty`].
//!
//! gst's `scope` property is not here: g2g's per-stream
//! [`BusMessage::StreamTag`] names the stream it tags with an id from a
//! `StreamCollection`, which this element has no way to know, so only the
//! global scope ([`BusMessage::Tag`]) can be honoured.
//!
//! Without a bus the element is a pass-through: there is nowhere for the tags
//! to go, and it says so on the debug log.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_debug, AsyncElement, BusHandle, BusMessage, Caps, CapsConstraint, ConfigureOutcome,
    ElementMetadata, G2gError, OutputSink, PipelinePacket, PropError, PropValue, PropertySpec,
    TagList,
};

use crate::tagproperty::TagsProperty;

/// # Example
///
/// ```no_run
/// use g2g_plugins::taginject::TagInject;
///
/// // gst-launch equivalent: taginject tags="title=A Title,artist=Someone"
/// let element = TagInject::new();
/// ```
#[derive(Debug, Default)]
pub struct TagInject {
    tags: TagsProperty,
    bus: Option<BusHandle>,
    posted: bool,
    configured: bool,
    log_name: LogName,
}

impl TagInject {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach the pipeline bus the tags are posted on.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// The tags parsed from the `tags` property.
    pub fn tags(&self) -> &TagList {
        self.tags.tags()
    }

    /// Post the tags, once, ahead of the first frame.
    fn post_tags(&mut self) {
        if self.posted || self.tags.tags().is_empty() {
            return;
        }
        self.posted = true;
        let Some(bus) = &self.bus else {
            g2g_debug!(
                self,
                "no bus attached, so the {} injected tags go nowhere",
                self.tags.tags().len()
            );
            return;
        };
        bus.try_post(BusMessage::Tag {
            tags: self.tags.tags().clone(),
            program: None,
        });
    }
}

impl AsyncElement for TagInject {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "TagInject",
            "Generic",
            "Posts hand-written stream metadata on the bus",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Wildcard pass-through: tags describe a stream of any media type.
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    self.post_tags();
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
        TAGINJECT_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "tags" => self.tags.set(value.as_str().ok_or(PropError::Type)?),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "tags" => self.tags.value(),
            _ => None,
        }
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }
}

impl LogSource for TagInject {
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

/// `TagInject`'s settable properties, named as gst `taginject`.
static TAGINJECT_PROPS: &[PropertySpec] = &[TagsProperty::SPEC];

//! Vorbis comment-header rewriter element (`vorbistag`): a packetized Vorbis
//! stream in, the same packets out with the comment header replaced.
//!
//! A Vorbis stream opens with three header packets: identification, comment and
//! setup. The comment packet (`\x03vorbis`) is the one that carries the
//! metadata, so rewriting it is all there is to tagging a Vorbis stream. Every
//! other packet passes through untouched, the vendor string included: rewriting
//! the tags should not relabel who encoded the audio.
//!
//! The tags come from the `tags` property ([`crate::tagproperty`]), not from
//! upstream: a [`TagList`] travels out of band on the bus and
//! an element cannot read the bus, so there is no in-band tag event to pick up.
//!
//! # Example
//!
//! ```no_run
//! use g2g_core::AsyncElement;
//! use g2g_core::PropValue;
//! use g2g_plugins::vorbistag::VorbisTag;
//!
//! // gst-launch equivalent:
//! //   filesrc location=in.ogg ! oggdemux ! vorbistag tags="title=A Title" ! oggmux ! filesink location=out.ogg
//! let mut element = VorbisTag::new();
//! element
//!     .set_property("tags", PropValue::Str("title=A Title".into()))
//!     .unwrap();
//! ```

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::short_type_name;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropValue, PropertySpec, TagList,
};

use crate::tagproperty::TagsProperty;
use crate::vorbiscomment::{vorbis_comment, vorbis_comment_vendor, VENDOR, VORBIS_COMMENT_MAGIC};

/// # Example
///
/// ```no_run
/// use g2g_plugins::vorbistag::VorbisTag;
///
/// let element = VorbisTag::new();
/// ```
#[derive(Debug, Default)]
pub struct VorbisTag {
    configured: bool,
    tags: TagsProperty,
    sequence: u64,
}

impl VorbisTag {
    pub fn new() -> Self {
        Self::default()
    }

    /// The tags parsed from the `tags` property.
    pub fn tags(&self) -> &TagList {
        self.tags.tags()
    }

    /// The comment packet this writes in place of `original`, keeping the vendor
    /// string the stream already declared.
    fn rewritten(&self, original: &[u8]) -> Vec<u8> {
        let vendor = vorbis_comment_vendor(original);
        let vendor = vendor.as_deref().map_or(VENDOR, str::as_bytes);
        vorbis_comment(VORBIS_COMMENT_MAGIC, vendor, self.tags.tags())
    }
}

impl AsyncElement for VorbisTag {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio {
                format: AudioFormat::Vorbis,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Pass-through identity over Vorbis of any channels / rate: the tags do not
    /// touch the audio shape, which the identification header already fixed.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Audio {
                format: AudioFormat::Vorbis,
                ..
            } => {
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Vorbis tag writer",
            "Formatter/Metadata",
            "Rewrites the comment header packet of a Vorbis stream",
            "g2g",
        )
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
                    let slice = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    if !slice.starts_with(VORBIS_COMMENT_MAGIC) {
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        return Ok(());
                    }
                    let rewritten = self.rewritten(slice);
                    let replacement = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(rewritten.into_boxed_slice())),
                        frame.timing,
                        self.sequence,
                    );
                    self.sequence += 1;
                    out.push(PipelinePacket::DataFrame(replacement)).await?;
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
        VORBISTAG_PROPS
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
}

impl PadTemplates for VorbisTag {
    fn pad_templates() -> Vec<PadTemplate> {
        // `Caps::Audio` has no open dims; pin the common stereo / 44.1 kHz shape.
        let vorbis = Caps::Audio {
            format: AudioFormat::Vorbis,
            channels: 2,
            sample_rate: 44_100,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(vorbis.clone())),
            PadTemplate::source(CapsSet::one(vorbis)),
        ])
    }
}

/// `VorbisTag`'s settable properties, named as gst `vorbistag`.
static VORBISTAG_PROPS: &[PropertySpec] = &[TagsProperty::SPEC];

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{FrameTiming, PushOutcome, Tag};

    use crate::vorbiscomment::parse_vorbis_comment;

    /// The Vorbis identification header's magic, the packet ahead of the
    /// comment one.
    const VORBIS_IDENT_MAGIC: &[u8] = b"\x01vorbis";
    /// The vendor string the incoming stream declares, which the rewrite keeps.
    const UPSTREAM_VENDOR: &[u8] = b"libVorbis 1.3.7";

    #[derive(Default)]
    struct RecordingSink {
        packets: Vec<Vec<u8>>,
    }

    impl OutputSink for RecordingSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            if let PipelinePacket::DataFrame(frame) =
                packet_slot.take().expect("poll_push without a packet")
            {
                let slice = frame.domain.require_system_slice("test").expect("system");
                self.packets.push(Vec::from(slice));
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn vorbis_caps() -> Caps {
        Caps::Audio {
            format: AudioFormat::Vorbis,
            channels: 2,
            sample_rate: 44_100,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    fn tags() -> TagList {
        [Tag::Title("Sine".into()), Tag::Artist("g2g".into())]
            .into_iter()
            .collect()
    }

    async fn run(packets: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut element = VorbisTag::new();
        element
            .set_property("tags", PropValue::Str("title=Sine,artist=g2g".into()))
            .expect("a valid taglist");
        element.configure_pipeline(&vorbis_caps()).expect("vorbis");
        let mut out = RecordingSink::default();
        for packet in packets {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(packet.clone().into_boxed_slice())),
                FrameTiming::default(),
                0,
            );
            element
                .process(PipelinePacket::DataFrame(frame), &mut out)
                .await
                .expect("process");
        }
        out.packets
    }

    /// The rewritten comment packet read back by this crate's own reader, with
    /// the identification and setup packets untouched around it.
    #[tokio::test]
    async fn rewrites_the_comment_packet_only() {
        let ident = Vec::from(VORBIS_IDENT_MAGIC);
        let setup = Vec::from(&b"\x05vorbis codebooks"[..]);
        let comment = vorbis_comment(
            VORBIS_COMMENT_MAGIC,
            UPSTREAM_VENDOR,
            &[Tag::Title("Old".into())].into_iter().collect(),
        );
        let written = run(&[ident.clone(), comment, setup.clone()]).await;
        assert_eq!(written.len(), 3);
        assert_eq!(written[0], ident);
        assert_eq!(written[2], setup);
        assert_eq!(parse_vorbis_comment(&written[1]).tags(), tags().tags());
        assert_eq!(
            vorbis_comment_vendor(&written[1]).as_deref(),
            Some(core::str::from_utf8(UPSTREAM_VENDOR).unwrap()),
            "the encoder's vendor string survives the rewrite"
        );
        assert_eq!(
            *written[1].last().expect("a non-empty packet"),
            1,
            "the framing bit the Vorbis mapping mandates"
        );
    }

    /// Only the comment packet is replaced, so a stream carrying none reaches
    /// the far side unchanged.
    #[tokio::test]
    async fn a_stream_without_a_comment_packet_passes_through() {
        let ident = Vec::from(VORBIS_IDENT_MAGIC);
        let written = run(core::slice::from_ref(&ident)).await;
        assert_eq!(written, [ident]);
    }
}

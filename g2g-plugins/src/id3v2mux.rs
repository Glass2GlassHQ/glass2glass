//! ID3v2 tag writer element (`id3v2mux`, gst's other name for it `id3mux`): the
//! coded byte stream in, the same stream with an ID3v2 tag ahead of it out. An
//! ID3v2 tag already at the head is replaced, so running the element twice
//! leaves one tag.
//!
//! The tags come from the `tags` property ([`crate::tagproperty`]), not from
//! upstream: a [`TagList`](g2g_core::TagList) travels out of band on the bus and
//! an element cannot read the bus, so there is no in-band tag event to pick up.
//!
//! `write-v1` adds the 128-byte ID3v1 block behind the stream, the lossy
//! 30-bytes-a-field summary a player with no ID3v2 support reads.
//!
//! # Example
//!
//! ```no_run
//! use g2g_core::AsyncElement;
//! use g2g_core::PropValue;
//! use g2g_plugins::id3v2mux::Id3V2Mux;
//!
//! // gst-launch equivalent:
//! //   filesrc location=in.mp3 ! id3v2mux tags="title=A Title" ! filesink location=out.mp3
//! let mut element = Id3V2Mux::new();
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
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, MemoryDomain,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, TagList,
};

use crate::id3::{id3v2_len, write_id3v1, write_id3v2, ID3V2_HEADER_LEN, ID3V2_VERSION_2_3};
use crate::tagproperty::TagsProperty;

/// # Example
///
/// ```no_run
/// use g2g_plugins::id3v2mux::Id3V2Mux;
///
/// let element = Id3V2Mux::new();
/// ```
#[derive(Debug)]
pub struct Id3V2Mux {
    configured: bool,
    tags: TagsProperty,
    version: u8,
    write_v1: bool,
    /// Bytes of the ID3v2 tag being replaced that are still to be dropped from
    /// the front of the stream.
    replaced_skip: usize,
    /// Whether the head has been examined for a tag to replace.
    head_examined: bool,
    /// Bytes not yet forwarded: the head until the tag is decided.
    buf: Vec<u8>,
    header_written: bool,
    sequence: u64,
}

impl Default for Id3V2Mux {
    fn default() -> Self {
        Self::new()
    }
}

impl Id3V2Mux {
    pub fn new() -> Self {
        Self {
            configured: false,
            tags: TagsProperty::default(),
            version: ID3V2_VERSION_2_3,
            write_v1: false,
            replaced_skip: 0,
            head_examined: false,
            buf: Vec::new(),
            header_written: false,
            sequence: 0,
        }
    }

    /// The tags parsed from the `tags` property.
    pub fn tags(&self) -> &TagList {
        self.tags.tags()
    }

    /// The ID3v2 tag this writes, as it goes on the wire.
    fn tag_bytes(&self) -> Vec<u8> {
        write_id3v2(self.tags.tags(), self.version)
    }

    /// Drop the ID3v2 tag at the head of the stream, if there is one. Returns
    /// whether the payload behind it has been reached.
    fn skip_existing_tag(&mut self, eos: bool) -> bool {
        if !self.head_examined {
            if self.buf.len() < ID3V2_HEADER_LEN && !eos {
                return false;
            }
            self.head_examined = true;
            self.replaced_skip = id3v2_len(&self.buf).unwrap_or(0);
        }
        let drop = self.replaced_skip.min(self.buf.len());
        self.buf.drain(..drop);
        self.replaced_skip -= drop;
        self.replaced_skip == 0
    }

    async fn emit(&mut self, data: Vec<u8>, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if data.is_empty() {
            return Ok(());
        }
        // Unstamped: a buffer here can carry bytes from two inputs, so neither
        // input's time describes it, the same as `id3demux`.
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            Default::default(),
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    /// Write the tag once, then forward everything past the replaced one.
    async fn drain(&mut self, eos: bool, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if !self.skip_existing_tag(eos) {
            return Ok(());
        }
        if !self.header_written {
            self.header_written = true;
            let tag = self.tag_bytes();
            self.emit(tag, out).await?;
        }
        let data: Vec<u8> = core::mem::take(&mut self.buf);
        self.emit(data, out).await
    }
}

impl AsyncElement for Id3V2Mux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    /// The tag is not part of the media type, so the tagged stream is whatever
    /// the untagged one was declared as.
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

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ID3v2 tag writer",
            "Formatter/Metadata",
            "Writes an ID3v2 tag ahead of a stream, replacing one already there",
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
                    self.buf.extend_from_slice(slice);
                    self.drain(false, out).await?;
                }
                PipelinePacket::Eos => {
                    self.drain(true, out).await?;
                    if self.write_v1 {
                        let block = write_id3v1(self.tags.tags());
                        self.emit(block, out).await?;
                    }
                }
                PipelinePacket::Flush => {
                    self.buf.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        ID3V2MUX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "tags" => self.tags.set(value.as_str().ok_or(PropError::Type)?),
            "v2-version" => {
                let version = value.as_uint().ok_or(PropError::Type)?;
                if version != u64::from(ID3V2_VERSION_2_3)
                    && version != u64::from(crate::id3::ID3V2_VERSION_2_4)
                {
                    return Err(PropError::Value);
                }
                self.version = version as u8;
                Ok(())
            }
            "write-v1" => {
                self.write_v1 = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "tags" => self.tags.value(),
            "v2-version" => Some(PropValue::Uint(u64::from(self.version))),
            "write-v1" => Some(PropValue::Bool(self.write_v1)),
            _ => None,
        }
    }
}

/// `Id3V2Mux`'s settable properties, named as gst `id3mux` / `id3v2mux`.
static ID3V2MUX_PROPS: &[PropertySpec] = &[
    TagsProperty::SPEC,
    PropertySpec::new(
        "v2-version",
        PropKind::Uint,
        "ID3v2 major version to write: 3 (2.3) or 4 (2.4)",
    )
    .with_default("3")
    .with_range("3", "4"),
    PropertySpec::new(
        "write-v1",
        PropKind::Bool,
        "also write the 128-byte ID3v1 block behind the stream",
    )
    .with_default("false"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{PushOutcome, Tag, TagList};

    use crate::id3::{parse_id3v1, parse_id3v2, ID3V1_LEN};

    /// Payload bytes standing in for the coded stream the tag rides ahead of.
    const PAYLOAD: &[u8] = b"\xff\xfb\x90\x00 audio bytes";

    #[derive(Default)]
    struct RecordingSink {
        bytes: Vec<u8>,
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
                self.bytes.extend_from_slice(slice);
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn tags() -> TagList {
        [Tag::Title("Sine".into()), Tag::Artist("g2g".into())]
            .into_iter()
            .collect()
    }

    fn with_tags() -> Id3V2Mux {
        let mut element = Id3V2Mux::new();
        element
            .set_property("tags", PropValue::Str("title=Sine,artist=g2g".into()))
            .expect("a valid taglist");
        element
    }

    /// Run the element over `input` cut into `chunk` byte pieces, then EOS.
    async fn run(element: &mut Id3V2Mux, input: &[u8], chunk: usize) -> Vec<u8> {
        let mut out = RecordingSink::default();
        element
            .configure_pipeline(&Caps::ByteStream {
                encoding: g2g_core::ByteStreamEncoding::Raw,
            })
            .expect("any byte stream");
        for piece in input.chunks(chunk) {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(Vec::from(piece).into_boxed_slice())),
                Default::default(),
                0,
            );
            element
                .process(PipelinePacket::DataFrame(frame), &mut out)
                .await
                .expect("process");
        }
        element
            .process(PipelinePacket::Eos, &mut out)
            .await
            .expect("eos");
        out.bytes
    }

    /// The written stream split into the tags the parser reads back and the
    /// bytes behind the tag.
    fn split(written: &[u8]) -> (TagList, Vec<u8>) {
        let len = id3v2_len(written).expect("the stream opens on an ID3v2 tag");
        (parse_id3v2(&written[..len]), Vec::from(&written[len..]))
    }

    /// The tag the element writes, read back by `id3.rs`, with the payload
    /// behind it untouched.
    #[tokio::test]
    async fn writes_a_tag_the_parser_reads_back() {
        let written = run(&mut with_tags(), PAYLOAD, PAYLOAD.len()).await;
        let (read, payload) = split(&written);
        assert_eq!(read.tags(), tags().tags());
        assert_eq!(payload, PAYLOAD);
    }

    /// A stream arriving one byte at a time: the head must still be examined
    /// whole before the tag goes out.
    #[tokio::test]
    async fn a_leading_tag_is_replaced_not_repeated() {
        let mut stream = write_id3v2(
            &[Tag::Title("Old".into())].into_iter().collect(),
            ID3V2_VERSION_2_3,
        );
        stream.extend_from_slice(PAYLOAD);
        let written = run(&mut with_tags(), &stream, 1).await;
        let (read, payload) = split(&written);
        assert_eq!(read.tags(), tags().tags());
        assert_eq!(payload, PAYLOAD, "the old tag is gone, the audio is not");
    }

    #[tokio::test]
    async fn v2_version_4_writes_a_2_4_tag() {
        let mut element = with_tags();
        element
            .set_property("v2-version", PropValue::Uint(4))
            .expect("2.4 is a version this writes");
        let written = run(&mut element, PAYLOAD, PAYLOAD.len()).await;
        /// The ID3v2 header's major-version byte, behind the 3-byte magic.
        const VERSION_BYTE: usize = 3;
        assert_eq!(written[VERSION_BYTE], crate::id3::ID3V2_VERSION_2_4);
        assert_eq!(split(&written).0.tags(), tags().tags());
        assert!(element
            .set_property("v2-version", PropValue::Uint(2))
            .is_err());
    }

    #[tokio::test]
    async fn write_v1_appends_the_trailer() {
        let mut element = with_tags();
        element
            .set_property("write-v1", PropValue::Bool(true))
            .expect("write-v1");
        let written = run(&mut element, PAYLOAD, PAYLOAD.len()).await;
        let (read, payload) = split(&written);
        assert_eq!(read.tags(), tags().tags());
        let (audio, trailer) = payload.split_at(payload.len() - ID3V1_LEN);
        assert_eq!(audio, PAYLOAD);
        assert_eq!(
            parse_id3v1(trailer).expect("a TAG block").tags(),
            tags().tags()
        );
    }
}

//! APEv2 tag writer element (`apev2mux`): the coded byte stream in, the same
//! stream with an APEv2 tag at its tail out.
//!
//! An APEv2 tag sits at the very end of the file, except that it goes *ahead* of
//! an ID3v1 block when there is one: the ID3v1 block is defined as the last 128
//! bytes, so nothing may follow it. This element therefore holds the tail of the
//! stream back until EOS, splits off an ID3v1 block, replaces an APEv2 tag
//! already there, and writes [tag][ID3v1].
//!
//! The tags come from the `tags` property ([`crate::tagproperty`]), not from
//! upstream: a [`TagList`](g2g_core::TagList) travels out of band on the bus and
//! an element cannot read the bus, so there is no in-band tag event to pick up.
//!
//! # Example
//!
//! ```no_run
//! use g2g_core::AsyncElement;
//! use g2g_core::PropValue;
//! use g2g_plugins::apev2mux::ApeV2Mux;
//!
//! // gst-launch equivalent:
//! //   filesrc location=in.mpc ! apev2mux tags="artist=Someone" ! filesink location=out.mpc
//! let mut element = ApeV2Mux::new();
//! element
//!     .set_property("tags", PropValue::Str("artist=Someone".into()))
//!     .unwrap();
//! ```

use core::future::Future;
use core::pin::Pin;

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::short_type_name;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, MemoryDomain,
    OutputSink, PipelinePacket, PropError, PropValue, PropertySpec, Tag, TagList,
};

use crate::id3::{parse_id3v1, ID3V1_LEN};
use crate::tagproperty::TagsProperty;

/// The 8 bytes an APEv2 header and footer both open with.
pub(crate) const APE_PREAMBLE: &[u8; 8] = b"APETAGEX";
/// Header / footer length: preamble, version, tag size, item count, flags, and
/// 8 reserved bytes.
pub(crate) const APE_FOOTER_LEN: usize = 32;
/// Version field value of an APEv2 tag.
const APE_VERSION_2: u32 = 2000;
/// Tag flag bit 31: the tag carries a header as well as a footer.
const APE_FLAG_HAS_HEADER: u32 = 1 << 31;
/// Tag flag bit 29: this 32-byte block is the header, not the footer.
const APE_FLAG_IS_HEADER: u32 = 1 << 29;
/// Item flags: value type 00 (UTF-8 text), not read-only.
const APE_ITEM_FLAGS_UTF8: u32 = 0;

/// The longest APEv2 tag these elements hold whole: `apev2mux` to replace one,
/// `apedemux` to read one. A tag past this is carrying artwork rather than
/// text; the muxer leaves it where it is, behind the new tag whose footer a
/// reader finds first.
pub(crate) const MAX_REPLACED_APE_TAG: usize = 1 << 16;

/// Bytes held back until EOS: enough for an ID3v1 block plus the largest APEv2
/// tag these elements handle.
pub(crate) const TAIL_HELD: usize = MAX_REPLACED_APE_TAG + ID3V1_LEN;

/// # Example
///
/// ```no_run
/// use g2g_plugins::apev2mux::ApeV2Mux;
///
/// let element = ApeV2Mux::new();
/// ```
#[derive(Debug, Default)]
pub struct ApeV2Mux {
    configured: bool,
    tags: TagsProperty,
    /// The tail of the stream, held back so EOS can rewrite it.
    buf: Vec<u8>,
    sequence: u64,
}

impl ApeV2Mux {
    pub fn new() -> Self {
        Self::default()
    }

    /// The tags parsed from the `tags` property.
    pub fn tags(&self) -> &TagList {
        self.tags.tags()
    }

    /// The APEv2 tag this writes, as it goes on the wire.
    fn tag_bytes(&self) -> Vec<u8> {
        write_apev2(self.tags.tags())
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

    /// Forward everything but the held-back tail.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let Some(len) = self.buf.len().checked_sub(TAIL_HELD) else {
            return Ok(());
        };
        let data: Vec<u8> = self.buf.drain(..len).collect();
        self.emit(data, out).await
    }

    /// Rewrite the tail: payload, then the new tag, then whatever ID3v1 block
    /// was there.
    async fn finish(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let mut tail: Vec<u8> = core::mem::take(&mut self.buf);
        let id3v1 = split_id3v1(&mut tail);
        strip_apev2(&mut tail);
        tail.extend_from_slice(&self.tag_bytes());
        tail.extend_from_slice(&id3v1);
        self.emit(tail, out).await
    }
}

/// Split a trailing ID3v1 block off `tail`, returning it (empty when there is
/// none). The APEv2 tag goes ahead of it, so it has to come off first.
fn split_id3v1(tail: &mut Vec<u8>) -> Vec<u8> {
    let Some(start) = tail.len().checked_sub(ID3V1_LEN) else {
        return Vec::new();
    };
    if parse_id3v1(&tail[start..]).is_none() {
        return Vec::new();
    }
    tail.split_off(start)
}

/// Parse a trailing APEv2 tag: the tags it carries and how many bytes at the
/// end of `data` they occupy (header, items, and footer). `None` when there is
/// no tag, the footer is truncated, or a declared size would overrun.
pub(crate) fn parse_apev2_tail(data: &[u8]) -> Option<(TagList, usize)> {
    let footer_at = data.len().checked_sub(APE_FOOTER_LEN)?;
    let footer = data.get(footer_at..)?;
    if &footer[..APE_PREAMBLE.len()] != APE_PREAMBLE {
        return None;
    }
    let version = u32::from_le_bytes(footer[VERSION_AT..VERSION_AT + 4].try_into().ok()?);
    if version != APE_VERSION_2 {
        return None;
    }
    let size = u32::from_le_bytes(footer[SIZE_AT..SIZE_AT + 4].try_into().ok()?) as usize;
    let count = u32::from_le_bytes(footer[COUNT_AT..COUNT_AT + 4].try_into().ok()?);
    let flags = u32::from_le_bytes(footer[FLAGS_AT..FLAGS_AT + 4].try_into().ok()?);
    if !(APE_FOOTER_LEN..=MAX_REPLACED_APE_TAG).contains(&size) || count > MAX_APE_ITEMS {
        return None;
    }
    let header = if flags & APE_FLAG_HAS_HEADER != 0 {
        APE_FOOTER_LEN
    } else {
        0
    };
    let total = size.checked_add(header)?;
    let start = data.len().checked_sub(total)?;
    let items = data.get(start + header..footer_at)?;
    let tags = parse_ape_items(items, count)?;
    Some((tags, total))
}

/// Return the total length declared by an APEv2 header at the start of a
/// stream. The returned length includes the header, items, and footer.
pub(crate) fn apev2_start_len(data: &[u8]) -> Option<usize> {
    let header = data.get(..APE_FOOTER_LEN)?;
    if &header[..APE_PREAMBLE.len()] != APE_PREAMBLE {
        return None;
    }
    let version = u32::from_le_bytes(header[VERSION_AT..VERSION_AT + 4].try_into().ok()?);
    let size = u32::from_le_bytes(header[SIZE_AT..SIZE_AT + 4].try_into().ok()?) as usize;
    let count = u32::from_le_bytes(header[COUNT_AT..COUNT_AT + 4].try_into().ok()?);
    let flags = u32::from_le_bytes(header[FLAGS_AT..FLAGS_AT + 4].try_into().ok()?);
    if version != APE_VERSION_2
        || !(APE_FOOTER_LEN..=MAX_REPLACED_APE_TAG).contains(&size)
        || count > MAX_APE_ITEMS
        || flags & APE_FLAG_HAS_HEADER == 0
        || flags & APE_FLAG_IS_HEADER == 0
    {
        return None;
    }
    size.checked_add(APE_FOOTER_LEN)
}

/// Field offsets within a 32-byte APEv2 header / footer, shared with the tests.
const VERSION_AT: usize = 8;
const SIZE_AT: usize = 12;
const COUNT_AT: usize = 16;
const FLAGS_AT: usize = 20;
/// An item count past this is artwork or a corrupt footer, not a text tag.
const MAX_APE_ITEMS: u32 = 256;

fn parse_ape_items(mut items: &[u8], count: u32) -> Option<TagList> {
    let mut tags = TagList::new();
    for _ in 0..count {
        if items.len() < 8 {
            return None;
        }
        let value_size = u32::from_le_bytes(items[0..4].try_into().ok()?) as usize;
        let flags = u32::from_le_bytes(items[4..8].try_into().ok()?);
        items = items.get(8..)?;
        let key_end = items.iter().position(|&b| b == 0)?;
        let key = core::str::from_utf8(items.get(..key_end)?).ok()?;
        items = items.get(key_end + 1..)?;
        let value = items.get(..value_size)?;
        items = items.get(value_size..)?;
        // Bits 1-2 of the item flags are the type: 00 is UTF-8 text. Binary
        // (artwork) and locator items are skipped, not posted as tags.
        if flags & 0x6 != 0 {
            continue;
        }
        let text = core::str::from_utf8(value).ok()?;
        tags.push(Tag::from_key_value(key, text));
    }
    Some(tags)
}

/// Drop a trailing APEv2 tag from `tail`, so rewriting does not leave the old
/// one behind. A tag longer than [`MAX_REPLACED_APE_TAG`] is out of the held
/// window and stays where it is.
fn strip_apev2(tail: &mut Vec<u8>) {
    let Some(footer_at) = tail.len().checked_sub(APE_FOOTER_LEN) else {
        return;
    };
    let footer = &tail[footer_at..];
    if &footer[..APE_PREAMBLE.len()] != APE_PREAMBLE {
        return;
    }
    // The size field counts the items plus the footer, never the header, so a
    // tag written with one is 32 bytes longer than it says.
    let size = u32::from_le_bytes([footer[12], footer[13], footer[14], footer[15]]) as usize;
    let flags = u32::from_le_bytes([footer[20], footer[21], footer[22], footer[23]]);
    let header = if flags & APE_FLAG_HAS_HEADER != 0 {
        APE_FOOTER_LEN
    } else {
        0
    };
    let Some(total) = size.checked_add(header) else {
        return;
    };
    if let Some(start) = tail.len().checked_sub(total) {
        tail.truncate(start);
    }
}

/// Write a complete APEv2 tag: a header, one UTF-8 text item per tag, and a
/// footer. Both 32-byte blocks carry the same size and count; only the header
/// sets [`APE_FLAG_IS_HEADER`].
fn write_apev2(tags: &TagList) -> Vec<u8> {
    let mut items = Vec::new();
    let mut count = 0u32;
    for tag in tags.tags() {
        let key = ape_key(tag);
        if !key.bytes().all(|b| (0x20..=0x7E).contains(&b)) {
            continue; // an APEv2 key is printable ASCII, nothing else fits
        }
        let value = tag.value_string();
        items.extend_from_slice(&(value.len() as u32).to_le_bytes());
        items.extend_from_slice(&APE_ITEM_FLAGS_UTF8.to_le_bytes());
        items.extend_from_slice(key.as_bytes());
        items.push(0);
        items.extend_from_slice(value.as_bytes());
        count += 1;
    }
    let size = (items.len() + APE_FOOTER_LEN) as u32;
    let block = |flags: u32| {
        let mut b = Vec::from(*APE_PREAMBLE);
        b.extend_from_slice(&APE_VERSION_2.to_le_bytes());
        b.extend_from_slice(&size.to_le_bytes());
        b.extend_from_slice(&count.to_le_bytes());
        b.extend_from_slice(&flags.to_le_bytes());
        b.extend_from_slice(&[0u8; 8]); // reserved
        b
    };
    let mut tag = block(APE_FLAG_HAS_HEADER | APE_FLAG_IS_HEADER);
    tag.extend_from_slice(&items);
    tag.extend_from_slice(&block(APE_FLAG_HAS_HEADER));
    tag
}

/// The item key a tag is written under. APEv2 spells the common keys in title
/// case ("Title", "Artist", ...); a key that came in verbatim keeps its own.
fn ape_key(tag: &Tag) -> Cow<'_, str> {
    match tag {
        Tag::Title(_) => Cow::Borrowed("Title"),
        Tag::Artist(_) => Cow::Borrowed("Artist"),
        Tag::Album(_) => Cow::Borrowed("Album"),
        Tag::Encoder(_) => Cow::Borrowed("Encoder"),
        Tag::Language(_) => Cow::Borrowed("Language"),
        Tag::Comment(_) => Cow::Borrowed("Comment"),
        _ => tag.key(),
    }
}

impl AsyncElement for ApeV2Mux {
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
            "APEv2 tag writer",
            "Formatter/Metadata",
            "Writes an APEv2 tag at the tail of a stream, ahead of any ID3v1 block",
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
                    self.drain(out).await?;
                }
                // Only the end of the stream can tell a trailing tag from
                // payload, and only there is the write position known.
                PipelinePacket::Eos => self.finish(out).await?,
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
        APEV2MUX_PROPS
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

/// `ApeV2Mux`'s settable properties, named as gst `apev2mux`.
static APEV2MUX_PROPS: &[PropertySpec] = &[TagsProperty::SPEC];

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::PushOutcome;

    use crate::id3::write_id3v1;

    /// Payload bytes standing in for the coded stream the tag rides behind.
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

    fn u32_at(block: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]])
    }

    fn with_tags() -> ApeV2Mux {
        let mut element = ApeV2Mux::new();
        element
            .set_property("tags", PropValue::Str("title=Sine,artist=g2g".into()))
            .expect("a valid taglist");
        element
    }

    async fn run(element: &mut ApeV2Mux, input: &[u8], chunk: usize) -> Vec<u8> {
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

    /// Layout asserted field by field: preamble, version, the size that counts
    /// items plus one 32-byte block, the item count, and the two flag words that
    /// tell the header from the footer.
    #[tokio::test]
    async fn writes_a_header_items_and_a_footer() {
        let written = run(&mut with_tags(), PAYLOAD, PAYLOAD.len()).await;
        let (payload, tag) = written.split_at(PAYLOAD.len());
        assert_eq!(payload, PAYLOAD);
        let (header, rest) = tag.split_at(APE_FOOTER_LEN);
        let (items, footer) = rest.split_at(rest.len() - APE_FOOTER_LEN);
        for block in [header, footer] {
            assert_eq!(&block[..APE_PREAMBLE.len()], APE_PREAMBLE);
            assert_eq!(u32_at(block, VERSION_AT), APE_VERSION_2);
            assert_eq!(
                u32_at(block, SIZE_AT) as usize,
                items.len() + APE_FOOTER_LEN
            );
            assert_eq!(u32_at(block, COUNT_AT), 2);
        }
        assert_eq!(
            u32_at(header, FLAGS_AT),
            APE_FLAG_HAS_HEADER | APE_FLAG_IS_HEADER
        );
        assert_eq!(u32_at(footer, FLAGS_AT), APE_FLAG_HAS_HEADER);
        // One item: value length, flags, "Title\0", the value.
        let mut expected = Vec::new();
        expected.extend_from_slice(&4u32.to_le_bytes());
        expected.extend_from_slice(&APE_ITEM_FLAGS_UTF8.to_le_bytes());
        expected.extend_from_slice(b"Title\0Sine");
        assert_eq!(&items[..expected.len()], &expected[..]);
    }

    /// The APEv2 tag goes ahead of an ID3v1 block: nothing may follow the last
    /// 128 bytes of the file.
    #[tokio::test]
    async fn the_tag_goes_ahead_of_an_id3v1_block() {
        let id3v1 = write_id3v1(&[Tag::Title("Sine".into())].into_iter().collect());
        let mut stream = Vec::from(PAYLOAD);
        stream.extend_from_slice(&id3v1);
        let written = run(&mut with_tags(), &stream, 1).await;
        assert_eq!(&written[written.len() - ID3V1_LEN..], &id3v1[..]);
        let tag_end = written.len() - ID3V1_LEN;
        assert_eq!(
            &written[tag_end - APE_FOOTER_LEN..tag_end][..APE_PREAMBLE.len()],
            APE_PREAMBLE
        );
        assert_eq!(&written[..PAYLOAD.len()], PAYLOAD);
    }

    /// Running the writer twice leaves one tag, not two.
    #[tokio::test]
    async fn an_existing_tag_is_replaced() {
        let once = run(&mut with_tags(), PAYLOAD, PAYLOAD.len()).await;
        let twice = run(&mut with_tags(), &once, 7).await;
        assert_eq!(twice, once);
    }
}

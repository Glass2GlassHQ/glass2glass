//! FLAC tag writer element (`flactag`): a native `.flac` byte stream in, the
//! same stream with its VORBIS_COMMENT metadata block rewritten out.
//!
//! A FLAC file opens with the `fLaC` marker and a run of metadata blocks, one of
//! which (type 4) holds the VorbisComment fields. This element replaces that
//! block and leaves every other one alone: STREAMINFO stays first, as the format
//! requires, and PADDING / SEEKTABLE / PICTURE keep their order behind it. The
//! audio frames pass through untouched.
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
//! use g2g_plugins::flactag::FlacTag;
//!
//! // gst-launch equivalent:
//! //   filesrc location=in.flac ! flactag tags="artist=Someone" ! filesink location=out.flac
//! let mut element = FlacTag::new();
//! element
//!     .set_property("tags", PropValue::Str("artist=Someone".into()))
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

use crate::flacparse::complete_header_len;
use crate::tagproperty::TagsProperty;
use crate::vorbiscomment::{
    comment_body_vendor, vorbis_comment, FLAC_BLOCK_HEADER_LEN, FLAC_COMMENT_BLOCK_TYPE, VENDOR,
};

/// The 4 bytes a native FLAC stream opens with.
const FLAC_MARKER: &[u8; 4] = b"fLaC";
/// Metadata block header bit 7: this is the last block before the audio.
const FLAC_LAST_BLOCK: u8 = 0x80;
/// A metadata block's length field is 24 bits, so nothing longer can be written.
const MAX_FLAC_BLOCK_LEN: usize = (1 << 24) - 1;

/// # Example
///
/// ```no_run
/// use g2g_plugins::flactag::FlacTag;
///
/// let element = FlacTag::new();
/// ```
#[derive(Debug, Default)]
pub struct FlacTag {
    configured: bool,
    tags: TagsProperty,
    /// Bytes not yet forwarded: the header until it has all arrived.
    buf: Vec<u8>,
    header_written: bool,
    sequence: u64,
}

impl FlacTag {
    pub fn new() -> Self {
        Self::default()
    }

    /// The tags parsed from the `tags` property.
    pub fn tags(&self) -> &TagList {
        self.tags.tags()
    }

    async fn emit(&mut self, data: Vec<u8>, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if data.is_empty() {
            return Ok(());
        }
        // Unstamped: a buffer here can carry bytes from two inputs, so neither
        // input's time describes it. The parser behind this stamps from the
        // frame headers.
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            Default::default(),
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    /// Rewrite the header once it is whole, then forward the audio behind it.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if !self.header_written {
            let Some(header_len) = complete_header_len(&self.buf)? else {
                return Ok(()); // header still arriving
            };
            let header: Vec<u8> = self.buf.drain(..header_len).collect();
            let rewritten =
                rewrite_header(&header, self.tags.tags()).ok_or(G2gError::CapsMismatch)?;
            self.header_written = true;
            self.emit(rewritten, out).await?;
        }
        let data: Vec<u8> = core::mem::take(&mut self.buf);
        self.emit(data, out).await
    }
}

/// One metadata block of a FLAC header, as it sits in the stream.
struct MetadataBlock<'a> {
    kind: u8,
    body: &'a [u8],
}

/// Split a complete FLAC header into its metadata blocks. `None` when a block
/// header or body runs past the buffer, which [`complete_header_len`] has
/// already ruled out for a header it accepted.
fn metadata_blocks(header: &[u8]) -> Option<Vec<MetadataBlock<'_>>> {
    if header.get(..FLAC_MARKER.len())? != FLAC_MARKER {
        return None;
    }
    let mut blocks = Vec::new();
    let mut at = FLAC_MARKER.len();
    loop {
        let head = header.get(at..at.checked_add(FLAC_BLOCK_HEADER_LEN)?)?;
        let last = head[0] & FLAC_LAST_BLOCK != 0;
        let kind = head[0] & !FLAC_LAST_BLOCK;
        let len = u32::from_be_bytes([0, head[1], head[2], head[3]]) as usize;
        let start = at.checked_add(FLAC_BLOCK_HEADER_LEN)?;
        let end = start.checked_add(len)?;
        blocks.push(MetadataBlock {
            kind,
            body: header.get(start..end)?,
        });
        if last {
            return Some(blocks);
        }
        at = end;
    }
}

/// The FLAC header with its VORBIS_COMMENT block replaced: every other block in
/// its original order, the new comment block behind STREAMINFO, and the
/// last-block flag on whatever now comes last. `None` when the header is
/// malformed or the new comment block is longer than a 24-bit length can code.
fn rewrite_header(header: &[u8], tags: &TagList) -> Option<Vec<u8>> {
    let blocks = metadata_blocks(header)?;
    let vendor = blocks
        .iter()
        .find(|b| b.kind == FLAC_COMMENT_BLOCK_TYPE)
        .and_then(|b| comment_body_vendor(b.body));
    let vendor = vendor.as_deref().map_or(VENDOR, str::as_bytes);
    let comment = vorbis_comment(&[], vendor, tags);

    // STREAMINFO must stay first; the comment goes right behind it, and the rest
    // keep the order they arrived in.
    let mut bodies: Vec<(u8, &[u8])> = Vec::new();
    for block in &blocks {
        if block.kind == FLAC_COMMENT_BLOCK_TYPE {
            continue;
        }
        bodies.push((block.kind, block.body));
        if bodies.len() == 1 {
            bodies.push((FLAC_COMMENT_BLOCK_TYPE, &comment));
        }
    }
    if bodies.is_empty() {
        return None; // no STREAMINFO: not a FLAC header
    }

    let mut out = Vec::from(*FLAC_MARKER);
    let last = bodies.len() - 1;
    for (i, (kind, body)) in bodies.iter().enumerate() {
        if body.len() > MAX_FLAC_BLOCK_LEN {
            return None;
        }
        let flag = if i == last { FLAC_LAST_BLOCK } else { 0 };
        out.push(flag | kind);
        out.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
        out.extend_from_slice(body);
    }
    Some(out)
}

impl AsyncElement for FlacTag {
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
                format: AudioFormat::Flac,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Pass-through identity over FLAC of any channels / rate: the tags do not
    /// touch the audio shape, which STREAMINFO already fixed.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Audio {
                format: AudioFormat::Flac,
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
            "FLAC tag writer",
            "Formatter/Metadata",
            "Rewrites the VORBIS_COMMENT metadata block of a native FLAC stream",
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
                PipelinePacket::Eos => self.drain(out).await?,
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
        FLACTAG_PROPS
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

impl PadTemplates for FlacTag {
    fn pad_templates() -> Vec<PadTemplate> {
        // `Caps::Audio` has no open dims; pin the common stereo / 44.1 kHz shape.
        let flac = Caps::Audio {
            format: AudioFormat::Flac,
            channels: 2,
            sample_rate: 44_100,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(flac.clone())),
            PadTemplate::source(CapsSet::one(flac)),
        ])
    }
}

/// `FlacTag`'s settable properties, named as gst `flactag`.
static FLACTAG_PROPS: &[PropertySpec] = &[TagsProperty::SPEC];

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{PushOutcome, Tag};

    use crate::flacparse::parse_streaminfo;
    use crate::vorbiscomment::parse_comment_body;

    /// STREAMINFO's body is a fixed 34 bytes.
    const STREAMINFO_LEN: usize = 34;
    /// FLAC metadata block types this test builds around the comment one.
    const STREAMINFO_TYPE: u8 = 0;
    const PADDING_TYPE: u8 = 1;
    /// The vendor string the incoming header declares, which the rewrite keeps.
    const UPSTREAM_VENDOR: &[u8] = b"reference libFLAC 1.4.3";
    /// Audio bytes standing in for the frames behind the header.
    const AUDIO: &[u8] = b"\xff\xf8\x69\x18 frame bytes";

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

    fn block(kind: u8, last: bool, body: &[u8]) -> Vec<u8> {
        let mut out = Vec::from([if last { FLAC_LAST_BLOCK | kind } else { kind }]);
        out.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
        out.extend_from_slice(body);
        out
    }

    /// STREAMINFO coding 44100 Hz stereo: a 20-bit rate at byte 10, then
    /// channels-1 in the 3 bits behind it.
    fn streaminfo() -> Vec<u8> {
        const SAMPLE_RATE: u32 = 44_100;
        const CHANNELS: u8 = 2;
        let mut body = alloc::vec![0u8; STREAMINFO_LEN];
        body[10] = (SAMPLE_RATE >> 12) as u8;
        body[11] = ((SAMPLE_RATE >> 4) & 0xFF) as u8;
        body[12] = (((SAMPLE_RATE & 0x0F) as u8) << 4) | ((CHANNELS - 1) << 1);
        body
    }

    fn flac_stream(comment_tags: &TagList) -> Vec<u8> {
        let mut stream = Vec::from(*FLAC_MARKER);
        stream.extend(block(STREAMINFO_TYPE, false, &streaminfo()));
        stream.extend(block(
            FLAC_COMMENT_BLOCK_TYPE,
            false,
            &vorbis_comment(&[], UPSTREAM_VENDOR, comment_tags),
        ));
        stream.extend(block(PADDING_TYPE, true, &[0u8; 16]));
        stream.extend_from_slice(AUDIO);
        stream
    }

    fn flac_caps() -> Caps {
        Caps::Audio {
            format: AudioFormat::Flac,
            channels: 2,
            sample_rate: 44_100,
        }
    }

    fn tags() -> TagList {
        [Tag::Title("Sine".into()), Tag::Artist("g2g".into())]
            .into_iter()
            .collect()
    }

    async fn run(input: &[u8], chunk: usize) -> Vec<u8> {
        let mut element = FlacTag::new();
        element
            .set_property("tags", PropValue::Str("title=Sine,artist=g2g".into()))
            .expect("a valid taglist");
        element.configure_pipeline(&flac_caps()).expect("flac in");
        let mut out = RecordingSink::default();
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

    /// The rewritten block read back by this crate's own VorbisComment reader,
    /// with STREAMINFO, the padding block and the audio all intact around it.
    #[tokio::test]
    async fn rewrites_the_comment_block_only() {
        let stream = flac_stream(&[Tag::Title("Old".into())].into_iter().collect());
        let written = run(&stream, 7).await;
        let header_len = complete_header_len(&written)
            .expect("a FLAC header")
            .expect("the whole header");
        assert_eq!(&written[header_len..], AUDIO, "the audio is untouched");
        let blocks = metadata_blocks(&written[..header_len]).expect("the header parses");
        assert_eq!(
            blocks.iter().map(|b| b.kind).collect::<Vec<_>>(),
            [STREAMINFO_TYPE, FLAC_COMMENT_BLOCK_TYPE, PADDING_TYPE],
            "STREAMINFO stays first and the padding block keeps its place"
        );
        assert_eq!(blocks[0].body, streaminfo());
        assert_eq!(parse_comment_body(blocks[1].body).tags(), tags().tags());
        assert_eq!(
            comment_body_vendor(blocks[1].body).as_deref(),
            Some(core::str::from_utf8(UPSTREAM_VENDOR).unwrap()),
            "the encoder's vendor string survives the rewrite"
        );
        // The rewritten header is still one `flacparse` reads the geometry from.
        let info = parse_streaminfo(&written[..header_len]).expect("STREAMINFO");
        assert_eq!((info.sample_rate, info.channels), (44_100, 2));
    }

    /// A header with no comment block gains one, behind STREAMINFO.
    #[tokio::test]
    async fn a_header_without_a_comment_block_gains_one() {
        let mut stream = Vec::from(*FLAC_MARKER);
        stream.extend(block(STREAMINFO_TYPE, true, &streaminfo()));
        stream.extend_from_slice(AUDIO);
        let written = run(&stream, 64).await;
        let header_len = complete_header_len(&written)
            .expect("a FLAC header")
            .expect("the whole header");
        let blocks = metadata_blocks(&written[..header_len]).expect("the header parses");
        assert_eq!(
            blocks.iter().map(|b| b.kind).collect::<Vec<_>>(),
            [STREAMINFO_TYPE, FLAC_COMMENT_BLOCK_TYPE]
        );
        assert_eq!(parse_comment_body(blocks[1].body).tags(), tags().tags());
        assert_eq!(&written[header_len..], AUDIO);
    }

    /// Running the writer twice leaves one comment block, not two.
    #[tokio::test]
    async fn an_existing_comment_block_is_replaced() {
        let stream = flac_stream(&TagList::new());
        let once = run(&stream, 1024).await;
        let twice = run(&once, 3).await;
        assert_eq!(twice, once);
    }
}

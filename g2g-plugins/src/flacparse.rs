//! FLAC stream parser element (`flacparse`): a native `.flac` byte
//! stream in (`Caps::Audio{Flac}`, arbitrary chunks from `filesrc`),
//! frame-aligned FLAC out. The audio sibling of the re-framing `h264parse`:
//! FLAC frames carry no length field, so a byte stream must be split by frame
//! sync before a decoder can consume it (Matroska hands over whole frames, a
//! bare `.flac` file does not).
//!
//! The stream opens with the `fLaC` marker and metadata blocks; STREAMINFO
//! (the mandatory first block) provides the concrete sample rate / channels
//! for the refined `CapsChanged`. The whole header (marker through the last
//! metadata block) is forwarded in-band as the first `DataFrame`, the
//! `fLaC`-prefixed extradata convention `ffmpegaudiodec` and the
//! Matroska `CodecPrivate` path already share. Frames are then split on the
//! 14-bit sync code, each candidate validated by parsing the whole frame
//! header and checking its CRC-8 (audio bytes alias the sync pattern, so an
//! unvalidated sync would mis-split). A frame ends at the next validated sync
//! whose number follows the current frame's or that closes a valid CRC-16, and a
//! last frame failing its CRC-16 is dropped. PTS / duration accumulate from each
//! header's block size, so the emitted timeline is sample-accurate. A `Segment`
//! also ends the last frame, and a `fLaC` header after it (a chained Ogg link)
//! is forwarded again and re-announces the caps if they changed.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, BusHandle, BusMessage, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, Tag, TagList,
};

use crate::flactag::metadata_blocks;
use crate::vorbiscomment::{
    parse_comment_body, parse_flac_picture, FLAC_COMMENT_BLOCK_TYPE, FLAC_PICTURE_BLOCK_TYPE,
};

/// Sample rate / channel count out of the mandatory STREAMINFO block.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StreamInfo {
    pub(crate) sample_rate: u32,
    pub(crate) channels: u8,
}

/// A validated frame header at a sync candidate: the frame's per-channel
/// sample count (its block size).
pub(crate) struct FrameHeader {
    pub(crate) block_size: u32,
    // the coded frame number, or the first sample's number for variable block sizes
    number: u64,
    variable_block_size: bool,
}

impl FrameHeader {
    fn next_number(&self) -> u64 {
        if self.variable_block_size {
            self.number.saturating_add(u64::from(self.block_size))
        } else {
            self.number.saturating_add(1)
        }
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::flacparse::FlacParse;
///
/// let parser = FlacParse::new();
/// assert_eq!(parser.frames_emitted(), 0);
/// ```
pub struct FlacParse {
    configured: bool,
    /// Unconsumed input bytes (the header until complete, then frame data from
    /// the current frame's sync).
    buf: Vec<u8>,
    info: Option<StreamInfo>,
    /// How many `fLaC` header blocks have been emitted downstream.
    headers_sent: u64,
    /// Per-channel samples emitted so far (the PTS accumulator).
    samples: u64,
    sequence: u64,
    bus: Option<BusHandle>,
    tags: TagList,
}

impl Default for FlacParse {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for FlacParse {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FlacParse")
            .field("buffered", &self.buf.len())
            .field("headers_sent", &self.headers_sent)
            .field("sequence", &self.sequence)
            .finish()
    }
}

impl FlacParse {
    pub fn new() -> Self {
        Self {
            configured: false,
            buf: Vec::new(),
            info: None,
            headers_sent: 0,
            samples: 0,
            sequence: 0,
            bus: None,
            tags: TagList::new(),
        }
    }

    // the stream's comments and pictures reach the application as a `BusMessage::Tag`
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    // the comments and pictures of the latest header, empty until one is parsed
    pub fn tags(&self) -> &TagList {
        &self.tags
    }

    /// Count of frames emitted (excluding the in-band header blocks).
    pub fn frames_emitted(&self) -> u64 {
        self.sequence.saturating_sub(self.headers_sent)
    }

    fn flac_caps(&self) -> Caps {
        let (channels, sample_rate) = self
            .info
            .map(|i| (i.channels, i.sample_rate))
            .unwrap_or((0, 0));
        Caps::Audio {
            format: AudioFormat::Flac,
            channels,
            sample_rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    async fn emit(&mut self, data: Vec<u8>, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let (pts_ns, duration_ns) = match (self.info, parse_frame_header(&data)) {
            (Some(info), Some(h)) if info.sample_rate > 0 => {
                let ns = |samples: u64| {
                    (samples as u128 * 1_000_000_000 / info.sample_rate as u128) as u64
                };
                let pts = ns(self.samples);
                let dur = ns(u64::from(h.block_size));
                self.samples += u64::from(h.block_size);
                (pts, dur)
            }
            // The header block (no frame header) rides at time zero.
            _ => (0, 0),
        };
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns,
                ..FrameTiming::default()
            },
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    // the last frame ends at the end of the stream, not at a next sync
    async fn flush_last_frame(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        self.drain(out).await?;
        if self.headers_sent > 0 && parse_frame_header(&self.buf).is_some() && crc16(&self.buf) == 0
        {
            let tail = core::mem::take(&mut self.buf);
            self.emit(tail, out).await?;
        }
        Ok(())
    }

    fn post_tags(&self) {
        if let (Some(bus), false) = (&self.bus, self.tags.is_empty()) {
            bus.try_post(BusMessage::Tag {
                tags: self.tags.clone(),
                program: None,
            });
        }
    }

    /// Consume as much of `buf` as is settled: the header once its last
    /// metadata block is complete, then every frame whose end (the next
    /// validated sync) has arrived.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        // A chained Ogg link or a re-read after a flush re-sends the header in-band.
        while self.headers_sent == 0 || self.buf.starts_with(b"fLaC") {
            let Some(header_len) = complete_header_len(&self.buf)? else {
                return Ok(()); // header still arriving
            };
            let caps_before = self.info.map(|_| self.flac_caps());
            self.info =
                Some(parse_streaminfo(&self.buf[..header_len]).ok_or(G2gError::CapsMismatch)?);
            let caps = self.flac_caps();
            if caps_before.as_ref() != Some(&caps) {
                out.push(PipelinePacket::CapsChanged(caps)).await?;
            }
            let tags = header_tags(&self.buf[..header_len])?;
            if tags != self.tags {
                self.tags = tags;
                self.post_tags();
            }
            let header: Vec<u8> = self.buf.drain(..header_len).collect();
            self.headers_sent += 1;
            self.emit(header, out).await?;
        }
        // Frames: buf starts at the current frame's sync; emit each frame whose
        // end (the next validated sync) is in the buffer.
        loop {
            if parse_frame_header(&self.buf).is_none() {
                // Not at a frame start (mid-header arrival or junk): wait; a
                // longer buffer either validates or the resync below advances.
                return Ok(());
            }
            let Some(end) = frame_end(&self.buf) else {
                return Ok(()); // the current frame's end has not arrived
            };
            let frame: Vec<u8> = self.buf.drain(..end).collect();
            self.emit(frame, out).await?;
        }
    }
}

fn header_tags(header: &[u8]) -> Result<TagList, G2gError> {
    let blocks = metadata_blocks(header).ok_or(G2gError::CapsMismatch)?;
    Ok(blocks
        .iter()
        .flat_map(|block| -> Vec<Tag> {
            match block.kind {
                FLAC_COMMENT_BLOCK_TYPE => parse_comment_body(block.body).tags().to_vec(),
                FLAC_PICTURE_BLOCK_TYPE => parse_flac_picture(block.body).into_iter().collect(),
                _ => Vec::new(),
            }
        })
        .collect())
}

/// Byte length of the complete `fLaC` header (marker + all metadata blocks), or
/// `None` while it is still arriving. `Err` when the stream does not open with
/// the marker (not a FLAC stream: fail loud, not silent). Shared with
/// [`crate::flactag`], which rewrites one of those metadata blocks.
pub(crate) fn complete_header_len(buf: &[u8]) -> Result<Option<usize>, G2gError> {
    if buf.len() < 4 {
        return Ok(None);
    }
    if &buf[..4] != b"fLaC" {
        return Err(G2gError::CapsMismatch);
    }
    let mut at = 4usize;
    loop {
        let Some(block) = buf.get(at..at + 4) else {
            return Ok(None);
        };
        let last = block[0] & 0x80 != 0;
        let len = u32::from_be_bytes([0, block[1], block[2], block[3]]) as usize;
        at = match at.checked_add(4 + len) {
            Some(v) => v,
            None => return Err(G2gError::CapsMismatch),
        };
        if last {
            return Ok((at <= buf.len()).then_some(at));
        }
    }
}

/// Sample rate / channels out of the header's STREAMINFO block (type 0, the
/// mandatory first metadata block). `None` if absent or malformed. Shared with
/// the Ogg-FLAC mapping in [`crate::ogg`] (its first packet embeds this header).
pub(crate) fn parse_streaminfo(header: &[u8]) -> Option<StreamInfo> {
    let block = header.get(4..)?;
    if block.first()? & 0x7F != 0 {
        return None; // STREAMINFO must be the first block
    }
    let body = block.get(4..38)?;
    // 20-bit sample rate at byte 10, then 3 bits of channels-1.
    let sample_rate =
        (u32::from(body[10]) << 12) | (u32::from(body[11]) << 4) | (u32::from(body[12]) >> 4);
    let channels = ((body[12] >> 1) & 0x07) + 1;
    (sample_rate > 0).then_some(StreamInfo {
        sample_rate,
        channels,
    })
}

/// CRC-8 (polynomial 0x07, init 0), the FLAC frame-header checksum.
fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
        }
    }
    crc
}

// CRC-16 (polynomial 0x8005, init 0), the FLAC frame footer checksum
fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &b in data {
        crc ^= u16::from(b) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// Parse and validate a FLAC frame header at the start of `data`: the 14-bit
/// sync code, the fixed fields (rejecting reserved values), the UTF-8-coded
/// frame/sample number, the optional block-size / sample-rate bytes, and the
/// trailing CRC-8 over all of it. `None` unless everything checks out, so an
/// audio byte pair that aliases the sync pattern cannot mis-split a frame.
/// Shared with [`crate::oggdemux`], which times Ogg-FLAC packets by block size.
pub(crate) fn parse_frame_header(data: &[u8]) -> Option<FrameHeader> {
    let b1 = *data.get(1)?;
    let variable_block_size = b1 & 0x01 != 0;
    // 14 sync bits (0b11111111_111110) and the reserved bit after them.
    if data[0] != 0xFF || b1 & 0xFC != 0xF8 || b1 & 0x02 != 0 {
        return None;
    }
    let b2 = *data.get(2)?;
    let bs_code = b2 >> 4;
    let sr_code = b2 & 0x0F;
    if bs_code == 0 || sr_code == 15 {
        return None; // reserved / invalid
    }
    let b3 = *data.get(3)?;
    let channel_assignment = b3 >> 4;
    let sample_size_code = (b3 >> 1) & 0x07;
    if channel_assignment > 10 || sample_size_code == 3 || sample_size_code == 7 || b3 & 1 != 0 {
        return None; // reserved
    }
    // UTF-8-style coded frame/sample number (1..=7 bytes).
    let mut at = 4usize;
    let first = *data.get(at)?;
    let extra = match first.leading_ones() {
        0 => 0,
        n @ 2..=7 => n as usize - 1,
        _ => return None, // 0b10xxxxxx or 0xFF: not a valid first byte
    };
    at += 1;
    // a multi-byte first byte spends its leading ones plus a zero on the length
    let first_value_mask = if extra == 0 {
        0x7F
    } else {
        0x7F >> (extra + 1)
    };
    let mut number = u64::from(first & first_value_mask);
    for _ in 0..extra {
        let continuation = *data.get(at)?;
        if continuation & 0xC0 != 0x80 {
            return None;
        }
        number = (number << 6) | u64::from(continuation & 0x3F);
        at += 1;
    }
    // Explicit block-size / sample-rate bytes when the codes call for them.
    let block_size = match bs_code {
        1 => 192,
        2..=5 => 576u32 << (bs_code - 2),
        6 => {
            let v = u32::from(*data.get(at)?);
            at += 1;
            v + 1
        }
        7 => {
            let v = (u32::from(*data.get(at)?) << 8) | u32::from(*data.get(at + 1)?);
            at += 2;
            v + 1
        }
        _ => 256u32 << (bs_code - 8),
    };
    match sr_code {
        12 => at += 1,
        13 | 14 => at += 2,
        _ => {}
    }
    let crc = *data.get(at)?;
    (crc8(&data[..at]) == crc).then_some(FrameHeader {
        block_size,
        number,
        variable_block_size,
    })
}

// the next validated sync that follows in number or closes a frame whose CRC-16 passes
fn frame_end(data: &[u8]) -> Option<usize> {
    let next_number = parse_frame_header(data)?.next_number();
    let mut at = 1;
    while at + 1 < data.len() {
        if data[at] == 0xFF && data[at + 1] & 0xFC == 0xF8 {
            let follows = parse_frame_header(&data[at..])
                .is_some_and(|next| next.number == next_number || crc16(&data[..at]) == 0);
            if follows {
                return Some(at);
            }
        }
        at += 1;
    }
    None
}

impl AsyncElement for FlacParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
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

    /// Pass-through identity over FLAC of any channels/rate (STREAMINFO refines
    /// them mid-stream; the media type never changes).
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
            "FLAC parser",
            "Codec/Parser/Audio",
            "Splits a native FLAC stream into frames and refines caps",
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
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    self.buf.extend_from_slice(slice);
                    self.drain(out).await?;
                }
                PipelinePacket::Eos => self.flush_last_frame(out).await?,
                // The previous stream's last frame goes out ahead of a new segment
                // (a chained Ogg link).
                PipelinePacket::Segment(segment) => {
                    self.flush_last_frame(out).await?;
                    out.push(PipelinePacket::Segment(segment)).await?;
                }
                PipelinePacket::Flush => {
                    self.buf.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for FlacParse {
    fn pad_templates() -> Vec<PadTemplate> {
        // `Caps::Audio` has no open dims; pin the common stereo/44.1 kHz shape.
        let flac = Caps::Audio {
            format: AudioFormat::Flac,
            channels: 2,
            sample_rate: 44_100,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(flac.clone())),
            PadTemplate::source(CapsSet::one(flac)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flactag::FLAC_MARKER;
    use crate::testutil::{flac_block, flac_picture_body, flac_streaminfo, run, STREAMINFO_TYPE};
    use crate::vorbiscomment::vorbis_comment;

    #[test]
    fn reads_the_header_comments_and_pictures() {
        let png = b"\x89PNG";
        let comments: TagList = [Tag::Title("Sine".into())].into_iter().collect();
        let mut stream = Vec::from(*FLAC_MARKER);
        stream.extend(flac_block(STREAMINFO_TYPE, false, &flac_streaminfo()));
        stream.extend(flac_block(
            FLAC_COMMENT_BLOCK_TYPE,
            false,
            &vorbis_comment(&[], b"g2g", &comments),
        ));
        stream.extend(flac_block(
            FLAC_PICTURE_BLOCK_TYPE,
            true,
            &flac_picture_body("image/png", png),
        ));
        let mut parser = FlacParse::new();
        let caps = Caps::Audio {
            format: AudioFormat::Flac,
            channels: 0,
            sample_rate: 0,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        run(&mut parser, &caps, &[&stream]);
        assert_eq!(
            parser.tags().tags(),
            &[
                Tag::Title("Sine".into()),
                Tag::Image {
                    codec: g2g_core::VideoCodec::Png,
                    data: png.to_vec()
                }
            ]
        );
    }
}

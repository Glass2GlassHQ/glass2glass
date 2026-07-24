//! FLAC stream parser element (`flacparse`, M774): a native `.flac` byte
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
//! `fLaC`-prefixed extradata convention [`crate::ffmpegaudiodec`] and the
//! Matroska `CodecPrivate` path already share. Frames are then split on the
//! 14-bit sync code, each candidate validated by parsing the whole frame
//! header and checking its CRC-8 (audio bytes alias the sync pattern, so an
//! unvalidated sync would mis-split). A frame ends at the next validated sync
//! (or EOF), and PTS / duration accumulate from each header's block size, so
//! the emitted timeline is sample-accurate.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
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
}

pub struct FlacParse {
    configured: bool,
    /// Unconsumed input bytes (the header until complete, then frame data from
    /// the current frame's sync).
    buf: Vec<u8>,
    info: Option<StreamInfo>,
    /// Whether the `fLaC` header block has been emitted downstream.
    header_sent: bool,
    /// Per-channel samples emitted so far (the PTS accumulator).
    samples: u64,
    sequence: u64,
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
            .field("header_sent", &self.header_sent)
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
            header_sent: false,
            samples: 0,
            sequence: 0,
        }
    }

    /// Count of frames emitted (excluding the in-band header block).
    pub fn frames_emitted(&self) -> u64 {
        self.sequence.saturating_sub(u64::from(self.header_sent))
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

    /// Consume as much of `buf` as is settled: the header once its last
    /// metadata block is complete, then every frame whose end (the next
    /// validated sync) has arrived.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if !self.header_sent {
            let Some(header_len) = complete_header_len(&self.buf)? else {
                return Ok(()); // header still arriving
            };
            self.info =
                Some(parse_streaminfo(&self.buf[..header_len]).ok_or(G2gError::CapsMismatch)?);
            out.push(PipelinePacket::CapsChanged(self.flac_caps()))
                .await?;
            let header: Vec<u8> = self.buf.drain(..header_len).collect();
            self.header_sent = true;
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
            let Some(end) = next_sync(&self.buf, 1) else {
                return Ok(()); // the current frame's end has not arrived
            };
            let frame: Vec<u8> = self.buf.drain(..end).collect();
            self.emit(frame, out).await?;
        }
    }
}

/// Byte length of the complete `fLaC` header (marker + all metadata blocks), or
/// `None` while it is still arriving. `Err` when the stream does not open with
/// the marker (not a FLAC stream: fail loud, not silent).
fn complete_header_len(buf: &[u8]) -> Result<Option<usize>, G2gError> {
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

/// Parse and validate a FLAC frame header at the start of `data`: the 14-bit
/// sync code, the fixed fields (rejecting reserved values), the UTF-8-coded
/// frame/sample number, the optional block-size / sample-rate bytes, and the
/// trailing CRC-8 over all of it. `None` unless everything checks out, so an
/// audio byte pair that aliases the sync pattern cannot mis-split a frame.
/// Shared with [`crate::oggdemux`], which times Ogg-FLAC packets by block size.
pub(crate) fn parse_frame_header(data: &[u8]) -> Option<FrameHeader> {
    let b1 = *data.get(1)?;
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
    for _ in 0..extra {
        if data.get(at)? & 0xC0 != 0x80 {
            return None;
        }
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
    (crc8(&data[..at]) == crc).then_some(FrameHeader { block_size })
}

/// The next validated frame-sync offset at or past `from`, or `None`.
fn next_sync(data: &[u8], from: usize) -> Option<usize> {
    let mut at = from;
    while at + 1 < data.len() {
        if data[at] == 0xFF
            && data[at + 1] & 0xFC == 0xF8
            && parse_frame_header(&data[at..]).is_some()
        {
            return Some(at);
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.buf.extend_from_slice(slice);
                    self.drain(out).await?;
                }
                // The last frame ends at EOF, not at a next sync: flush it.
                PipelinePacket::Eos => {
                    self.drain(out).await?;
                    if self.header_sent && parse_frame_header(&self.buf).is_some() {
                        let tail = core::mem::take(&mut self.buf);
                        self.emit(tail, out).await?;
                    }
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
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(flac.clone())),
            PadTemplate::source(CapsSet::one(flac)),
        ])
    }
}

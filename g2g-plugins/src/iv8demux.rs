use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::time::ticks_to_ns;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, VideoCodec,
};

use crate::fifobuffer::FifoBuffer;

const PACKET_HEADER_LEN: usize = 16;
// the size field also counts the twelve header bytes that follow it
const HEADER_BYTES_AFTER_SIZE: usize = 12;
const PACKET_TYPE_VIDEO: u16 = 257;
const FLAG_LAST_PACKET_OF_FRAME: u16 = 0x80;
const IV8_CLOCK_HZ: u32 = 90_000;
// far above any MPEG-4 Part 2 frame, a stream that never ends a frame fails here
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
struct PacketHeader {
    is_video: bool,
    payload_len: usize,
    ends_frame: bool,
    pts_90khz: u32,
}

fn parse_packet_header(header: &[u8; PACKET_HEADER_LEN]) -> Result<PacketHeader, G2gError> {
    let be16 = |at: usize| u16::from_be_bytes([header[at], header[at + 1]]);
    let size = usize::from(be16(2));
    let payload_len = size
        .checked_sub(HEADER_BYTES_AFTER_SIZE)
        .filter(|&len| len > 0)
        .ok_or(G2gError::CapsMismatch)?;
    Ok(PacketHeader {
        // type 258 and any other type carry no video
        is_video: be16(0) == PACKET_TYPE_VIDEO,
        payload_len,
        ends_frame: be16(4) & FLAG_LAST_PACKET_OF_FRAME != 0,
        pts_90khz: u32::from_be_bytes([header[8], header[9], header[10], header[11]]),
    })
}

#[derive(Debug, Default)]
struct FrameAssembly {
    data: Vec<u8>,
    pts_90khz: u32,
}

impl FrameAssembly {
    fn append(&mut self, header: PacketHeader, payload: &[u8]) -> Result<(), G2gError> {
        if self.data.len().saturating_add(payload.len()) > MAX_FRAME_BYTES {
            return Err(G2gError::CapsMismatch);
        }
        if self.data.is_empty() {
            self.pts_90khz = header.pts_90khz;
        }
        self.data.extend_from_slice(payload);
        Ok(())
    }

    fn clear(&mut self) {
        self.data.clear();
    }
}

#[derive(Debug, Default)]
pub struct Iv8Demux {
    configured: bool,
    buf: FifoBuffer<u8>,
    assembly: FrameAssembly,
    emitted: u64,
}

impl Iv8Demux {
    pub fn new() -> Self {
        Self::default()
    }

    fn input_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Iv8,
        }
    }

    pub fn output_caps() -> Caps {
        crate::typefind::elementary_video_caps(VideoCodec::Mpeg4Part2)
    }

    async fn emit_frame(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let data = core::mem::take(&mut self.assembly.data);
        let pts_ns = ticks_to_ns(u64::from(self.assembly.pts_90khz), IV8_CLOCK_HZ);
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                ..FrameTiming::default()
            },
            self.emitted,
        );
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        loop {
            let pending = self.buf.as_slice();
            let Some(header_bytes) = pending.first_chunk::<PACKET_HEADER_LEN>() else {
                return Ok(());
            };
            let header = parse_packet_header(header_bytes)?;
            let end = PACKET_HEADER_LEN + header.payload_len;
            let Some(payload) = pending.get(PACKET_HEADER_LEN..end) else {
                return Ok(());
            };
            if header.is_video {
                self.assembly.append(header, payload)?;
            }
            self.buf.discard(end);
            if header.is_video && header.ends_frame {
                self.emit_frame(out).await?;
            }
        }
    }

    // ffmpeg also keeps what arrived of a cut video payload and drops a cut header
    async fn finish(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        self.drain(out).await?;
        let pending = self.buf.as_slice();
        if let Some(header_bytes) = pending.first_chunk::<PACKET_HEADER_LEN>() {
            let header = parse_packet_header(header_bytes)?;
            if header.is_video {
                self.assembly
                    .append(header, &pending[PACKET_HEADER_LEN..])?;
            }
            if header.is_video && !self.assembly.data.is_empty() {
                self.emit_frame(out).await?;
            }
        }
        self.buf.clear();
        self.assembly.clear();
        Ok(())
    }
}

impl AsyncElement for Iv8Demux {
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "IV8 demuxer",
            "Codec/Demuxer",
            "Demuxes an IndigoVision 8000 file into its MPEG-4 Part 2 frames",
            "g2g",
        )
    }
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Iv8,
            } => CapsSet::one(Self::output_caps()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if *absolute_caps != Self::input_caps() {
            return Err(G2gError::CapsMismatch);
        }
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
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    self.buf.extend_from_slice(slice);
                    self.drain(out).await?;
                }
                PipelinePacket::Eos => {
                    self.finish(out).await?;
                }
                PipelinePacket::Flush => {
                    self.buf.clear();
                    self.assembly.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
                // the parser downstream reads the geometry from the bitstream
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for Iv8Demux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(CapsSet::one(Self::output_caps())),
        ])
    }
}

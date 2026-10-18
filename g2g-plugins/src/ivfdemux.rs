//! IVF demuxer: the simple raw container libvpx / libaom conformance vectors ship
//! in. A 32-byte `DKIF` header (FourCC codec + geometry + timebase), then before
//! each frame a 12-byte header (frame byte size + timestamp). One video elementary
//! stream out (VP8 / VP9 / AV1), the codec read from the header FourCC and the
//! concrete geometry emitted via `CapsChanged`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, Rate, VideoCodec,
};

use crate::fifobuffer::FifoBuffer;

const FILE_HEADER_LEN: usize = 32;
const FRAME_HEADER_LEN: usize = 12;

/// The 32-byte `DKIF` header's fixed fields.
#[derive(Debug, Clone, Copy)]
struct IvfHeader {
    codec: VideoCodec,
    width: u32,
    height: u32,
    /// The frame timebase is `scale / rate` seconds; a frame's timestamp is in
    /// these units, so `pts = ts * scale / rate`.
    rate: u32,
    scale: u32,
}

/// Demuxes an IVF byte stream into its single video elementary stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::ivfdemux::IvfDemux;
///
/// let demux = IvfDemux::new();
/// ```
#[derive(Debug, Default)]
pub struct IvfDemux {
    configured: bool,
    /// Bytes accumulated across input chunks, consumed as whole units: the file
    /// header first, then each 12-byte-prefixed frame. A frame split across chunks
    /// stays buffered until the rest arrives.
    buf: FifoBuffer<u8>,
    header: Option<IvfHeader>,
    caps_sent: bool,
    emitted: u64,
}

impl IvfDemux {
    pub fn new() -> Self {
        Self::default()
    }

    /// The input this element accepts: an IVF byte stream.
    fn input_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Ivf,
        }
    }

    /// The video codecs an IVF stream can carry, advertised at negotiation; the
    /// concrete one is fixed via `CapsChanged` once the `DKIF` header is read.
    fn output_alternatives() -> CapsSet {
        CapsSet::from_alternatives(Vec::from([
            Self::compressed(VideoCodec::Vp8),
            Self::compressed(VideoCodec::Vp9),
            Self::compressed(VideoCodec::Av1),
        ]))
    }

    fn compressed(codec: VideoCodec) -> Caps {
        Caps::CompressedVideo {
            codec,
            width: Dim::Range {
                min: 16,
                max: 65_535,
            },
            height: Dim::Range {
                min: 16,
                max: 65_535,
            },
            framerate: crate::typefind::PLACEHOLDER_FRAMERATE,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    pub fn stream_caps(file_prefix: &[u8]) -> Option<Caps> {
        let header = Self::parse_header(file_prefix)?;
        let declared = Self::compressed(header.codec);
        let header_size = Caps::CompressedVideo {
            codec: header.codec,
            width: Dim::Fixed(header.width),
            height: Dim::Fixed(header.height),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        // a header size outside the declared range is not trusted
        Some(declared.intersect(&header_size).unwrap_or(declared))
    }

    fn parse_header(buf: &[u8]) -> Option<IvfHeader> {
        if buf.len() < FILE_HEADER_LEN || &buf[0..4] != b"DKIF" {
            return None;
        }
        let codec = match &buf[8..12] {
            b"VP80" => VideoCodec::Vp8,
            b"VP90" => VideoCodec::Vp9,
            b"AV01" => VideoCodec::Av1,
            _ => return None,
        };
        let u16le = |o: usize| u16::from_le_bytes([buf[o], buf[o + 1]]) as u32;
        let u32le = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        Some(IvfHeader {
            codec,
            width: u16le(12),
            height: u16le(14),
            rate: u32le(16).max(1),
            scale: u32le(20).max(1),
        })
    }

    /// Parse the file header once (emitting the concrete output caps), then emit
    /// every complete frame currently buffered.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if self.header.is_none() {
            if self.buf.len() < FILE_HEADER_LEN {
                return Ok(());
            }
            let header = Self::parse_header(self.buf.as_slice()).ok_or(G2gError::CapsMismatch)?;
            self.buf.discard(FILE_HEADER_LEN);
            self.header = Some(header);
        }
        let header = self.header.expect("parsed above");
        if !self.caps_sent {
            // rate / scale is the timestamp timebase, not a frame rate
            out.push(PipelinePacket::CapsChanged(Caps::CompressedVideo {
                codec: header.codec,
                width: Dim::Fixed(header.width),
                height: Dim::Fixed(header.height),
                framerate: Rate::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            }))
            .await?;
            self.caps_sent = true;
        }
        loop {
            let pending = self.buf.as_slice();
            if pending.len() < FRAME_HEADER_LEN {
                return Ok(());
            }
            let size =
                u32::from_le_bytes([pending[0], pending[1], pending[2], pending[3]]) as usize;
            let ts = u64::from_le_bytes([
                pending[4],
                pending[5],
                pending[6],
                pending[7],
                pending[8],
                pending[9],
                pending[10],
                pending[11],
            ]);
            let end = FRAME_HEADER_LEN.saturating_add(size);
            let Some(data) = pending.get(FRAME_HEADER_LEN..end) else {
                return Ok(()); // partial frame; wait for the next chunk
            };
            let data = data.to_vec();
            self.buf.discard(end);
            let pts_ns = (u128::from(ts) * u128::from(header.scale) * 1_000_000_000
                / u128::from(header.rate)) as u64;
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
        }
    }
}

impl AsyncElement for IvfDemux {
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "IVF demuxer",
            "Codec/Demuxer",
            "Demuxes an IVF file into its VP8 / VP9 / AV1 frames",
            "g2g",
        )
    }
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
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // The concrete codec is unknown until the DKIF header is read, so offer
        // every one an IVF file can carry and let `CapsChanged` fix it at runtime.
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Ivf,
            } => Self::output_alternatives(),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Ivf
            }
        ) {
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
                    self.drain(out).await?;
                }
                // ByteStream caps carry no geometry; the concrete caps are emitted
                // from the parsed header instead.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for IvfDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(Self::output_alternatives()),
        ])
    }
}

#[cfg(fuzzing)]
pub fn fuzz_parse(data: &[u8]) {
    let _ = IvfDemux::parse_header(data);

    let mut demux = IvfDemux::new();
    if demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Ivf,
        })
        .is_err()
    {
        return;
    }
    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(data.to_vec().into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };
    let _ = crate::fuzz_process(&mut demux, PipelinePacket::DataFrame(frame));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::CollectSink;
    use alloc::vec;
    use g2g_core::FrameTiming;

    fn ivf_stream(fourcc: &[u8; 4], w: u16, h: u16, frames: &[&[u8]]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"DKIF");
        v.extend_from_slice(&0u16.to_le_bytes()); // version
        v.extend_from_slice(&32u16.to_le_bytes()); // header length
        v.extend_from_slice(fourcc);
        v.extend_from_slice(&w.to_le_bytes());
        v.extend_from_slice(&h.to_le_bytes());
        v.extend_from_slice(&30u32.to_le_bytes()); // timebase rate
        v.extend_from_slice(&1u32.to_le_bytes()); // timebase scale
        v.extend_from_slice(&(frames.len() as u32).to_le_bytes()); // frame count
        v.extend_from_slice(&0u32.to_le_bytes()); // unused
        for (i, f) in frames.iter().enumerate() {
            v.extend_from_slice(&(f.len() as u32).to_le_bytes());
            v.extend_from_slice(&(i as u64).to_le_bytes());
            v.extend_from_slice(f);
        }
        v
    }

    fn data_frame(bytes: Vec<u8>) -> Frame {
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        }
    }

    fn configured() -> IvfDemux {
        let mut demux = IvfDemux::new();
        demux
            .configure_pipeline(&Caps::ByteStream {
                encoding: ByteStreamEncoding::Ivf,
            })
            .unwrap();
        demux
    }

    fn payloads(packets: &[PipelinePacket]) -> Vec<&[u8]> {
        packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => f.domain.as_system_slice(),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn demuxes_ivf_into_codec_caps_then_frames() {
        let stream = ivf_stream(b"VP90", 176, 144, &[&[1, 2, 3], &[4, 5, 6, 7]]);
        let mut demux = configured();
        let mut sink = CollectSink::default();
        demux
            .process(PipelinePacket::DataFrame(data_frame(stream)), &mut sink)
            .await
            .unwrap();

        assert!(matches!(
            sink.packets[0],
            PipelinePacket::CapsChanged(Caps::CompressedVideo {
                codec: VideoCodec::Vp9,
                width: Dim::Fixed(176),
                height: Dim::Fixed(144),
                ..
            })
        ));
        assert_eq!(
            payloads(&sink.packets),
            vec![&[1u8, 2, 3][..], &[4, 5, 6, 7]]
        );
    }

    #[tokio::test]
    async fn reads_the_codec_fourcc() {
        for (fourcc, want) in [
            (b"VP80", VideoCodec::Vp8),
            (b"VP90", VideoCodec::Vp9),
            (b"AV01", VideoCodec::Av1),
        ] {
            let stream = ivf_stream(fourcc, 64, 48, &[&[0xAB]]);
            let mut demux = configured();
            let mut sink = CollectSink::default();
            demux
                .process(PipelinePacket::DataFrame(data_frame(stream)), &mut sink)
                .await
                .unwrap();
            match &sink.packets[0] {
                PipelinePacket::CapsChanged(Caps::CompressedVideo { codec, .. }) => {
                    assert_eq!(*codec, want)
                }
                p => panic!("expected CapsChanged, got {p:?}"),
            }
        }
    }

    #[tokio::test]
    async fn reassembles_a_frame_split_across_chunks() {
        let stream = ivf_stream(b"AV01", 64, 48, &[&[9u8; 100]]);
        let mut demux = configured();
        let mut sink = CollectSink::default();
        let split = 40; // mid-frame: past the file header, inside the payload
        demux
            .process(
                PipelinePacket::DataFrame(data_frame(stream[..split].to_vec())),
                &mut sink,
            )
            .await
            .unwrap();
        demux
            .process(
                PipelinePacket::DataFrame(data_frame(stream[split..].to_vec())),
                &mut sink,
            )
            .await
            .unwrap();
        assert_eq!(payloads(&sink.packets), vec![&[9u8; 100][..]]);
    }
}

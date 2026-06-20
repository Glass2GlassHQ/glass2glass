//! Fragmented-MP4 / CMAF byte-stream demuxer (Fmp4Demux): `ByteStream{IsoBmff}`
//! in, `CompressedVideo{H264|H265}` Annex-B access units out. The streaming
//! counterpart of the file-based [`Mp4Src`](crate::mp4src); both share the
//! [`fmp4`](crate::fmp4) parser. This is what an HLS/DASH fMP4 segment stream
//! (init segment + media fragments) feeds into, the analog of `tsdemux` for the
//! TS path.
//!
//! Bytes arrive in arbitrary chunks (whole segments, or split mid-box by a
//! generic source), so it buffers and processes one complete top-level box at a
//! time: `moov` yields the codec/geometry (emitted as `CapsChanged`) and the
//! parameter sets; each `moof`+`mdat` pair yields samples. The out-of-band
//! parameter sets are prepended to the first emitted access unit so a decoder can
//! start. Single video track; the profile `Mp4Sink` writes (see `fmp4`).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    Rate, VideoCodec,
};

use crate::fmp4::{parse_fragments, parse_header, starts_with_param_set, Header};

#[derive(Debug)]
pub struct Fmp4Demux {
    buffer: Vec<u8>,
    header: Option<Header>,
    /// A `moof` box awaiting its following `mdat` to form a complete fragment.
    pending_moof: Option<Vec<u8>>,
    /// Negotiation-time output codec (refined from the `moov` at runtime).
    out_codec: VideoCodec,
    /// Prepend the config-record parameter sets to the first access unit.
    need_param_sets: bool,
    caps_sent: bool,
    sequence: u64,
    configured: bool,
}

impl Default for Fmp4Demux {
    fn default() -> Self {
        Self::new()
    }
}

impl Fmp4Demux {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            header: None,
            pending_moof: None,
            out_codec: VideoCodec::H264,
            need_param_sets: true,
            caps_sent: false,
            sequence: 0,
            configured: false,
        }
    }

    fn input_caps() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }
    }

    fn output_caps(codec: VideoCodec, width: Dim, height: Dim) -> Caps {
        Caps::CompressedVideo { codec, width, height, framerate: Rate::Any }
    }

    /// Process every complete top-level box now buffered, emitting access units.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        while let Some(total) = next_box_len(&self.buffer) {
            if self.buffer.len() < total {
                break; // wait for the rest of this box
            }
            let box_bytes: Vec<u8> = self.buffer.drain(..total).collect();
            let kind: [u8; 4] = box_bytes[4..8].try_into().expect("8-byte box header");
            match &kind {
                b"moov" => {
                    let header = parse_header(&box_bytes)?;
                    let caps = Self::output_caps(
                        header.codec,
                        Dim::Fixed(header.width),
                        Dim::Fixed(header.height),
                    );
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                    self.out_codec = header.codec;
                    self.caps_sent = true;
                    self.header = Some(header);
                }
                b"moof" => self.pending_moof = Some(box_bytes),
                b"mdat" => {
                    let Some(mut frag) = self.pending_moof.take() else {
                        return Err(G2gError::CapsMismatch); // mdat without moof
                    };
                    // header must exist (moov precedes the first fragment)
                    let Some(header) = self.header.as_ref() else {
                        return Err(G2gError::CapsMismatch);
                    };
                    let (timescale, codec) = (header.timescale, header.codec);
                    let param_sets = header.param_sets.clone();

                    frag.extend_from_slice(&box_bytes);
                    let samples = parse_fragments(&frag, timescale, codec)?;
                    for s in samples {
                        let mut annexb = s.annexb;
                        if self.need_param_sets && !starts_with_param_set(&annexb, codec) {
                            let mut with_sets = Vec::new();
                            for set in &param_sets {
                                with_sets.extend_from_slice(&[0, 0, 0, 1]);
                                with_sets.extend_from_slice(set);
                            }
                            with_sets.extend_from_slice(&annexb);
                            annexb = with_sets;
                        }
                        self.need_param_sets = false;
                        let frame = Frame {
                            domain: MemoryDomain::System(SystemSlice::from_boxed(
                                annexb.into_boxed_slice(),
                            )),
                            timing: FrameTiming {
                                pts_ns: s.pts_ns,
                                dts_ns: s.pts_ns,
                                duration_ns: s.duration_ns,
                                capture_ns: s.pts_ns,
                                arrival_ns: g2g_core::metrics::monotonic_ns(),
                            },
                            sequence: self.sequence,
                            meta: Default::default(),
                        };
                        self.sequence += 1;
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                    }
                }
                // ftyp / styp / sidx / free / etc.: not needed for demux
                _ => {}
            }
        }
        Ok(())
    }
}

/// Total length of the box at the start of `buf`, or `None` if the 8-byte header
/// (or the 64-bit large-size header) isn't fully buffered yet. A size below 8
/// (including the size-0 "to end of stream" form) can't be framed and returns
/// `None`; the writer profile we consume always uses explicit sizes.
fn next_box_len(buf: &[u8]) -> Option<usize> {
    if buf.len() < 8 {
        return None;
    }
    let size = u32::from_be_bytes(buf[0..4].try_into().expect("4 bytes"));
    let total = if size == 1 {
        if buf.len() < 16 {
            return None;
        }
        u64::from_be_bytes(buf[8..16].try_into().expect("8 bytes")) as usize
    } else {
        size as usize
    };
    (total >= 8).then_some(total)
}

impl AsyncElement for Fmp4Demux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // ByteStream{IsoBmff} in -> the video track out. The default codec is
        // refined from the moov via CapsChanged at runtime (like tsdemux).
        let codec = self.out_codec;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff } => {
                CapsSet::one(Self::output_caps(codec, Dim::Any, Dim::Any))
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(absolute_caps, Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }) {
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.buffer.extend_from_slice(slice.as_slice());
                    self.drain(out).await?;
                }
                // Nothing to flush (incomplete trailing boxes are dropped); the
                // runner's transform arm forwards the EOS itself.
                PipelinePacket::Eos => {}
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for Fmp4Demux {
    fn pad_templates() -> Vec<PadTemplate> {
        let video = |codec| Self::output_caps(codec, Dim::Any, Dim::Any);
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(CapsSet::from_alternatives(Vec::from([
                video(VideoCodec::H264),
                video(VideoCodec::H265),
            ]))),
        ])
    }
}

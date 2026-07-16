//! Progressive / whole-file MP4 demuxer (Mp4Demux, M479): `ByteStream{Mp4}` in,
//! the video track's Annex-B access units out. The single-output, whole-file
//! sibling of [`Fmp4Demux`](crate::fmp4demux) (which streams a live fragmented
//! `ByteStream{IsoBmff}`): a progressive `.mp4` / `.mov` keeps its `moov` sample
//! tables (`stbl`) and an `mdat`, and the `moov` may sit at the *end* of the file
//! with absolute `stco` chunk offsets, so the whole file is buffered before the
//! [`fmp4::parse_progressive`](crate::fmp4::parse_progressive) pass runs at `Eos`.
//!
//! This is what a bare `filesrc location=movie.mp4 ! decodebin` auto-plugs: a file
//! source types itself `ByteStream{Mp4}` (M478), the auto-plugger picks this
//! whole-file demuxer for it, and the fragmented `fmp4demux` still serves the
//! streaming `IsoBmff` that HLS / DASH produce. Multi-track fan-out (video, audio,
//! text) stays on [`Mp4DemuxN`](crate::mp4demuxn) via `qtdemux name=d ...`; this
//! element exposes only the video track, the one stream a linear `decodebin` chain
//! decodes.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata,
    G2gError, OutputSink, PadTemplate, PadTemplates, Rate, VideoCodec,
};

use crate::fmp4::{parse_header, parse_progressive, prepend_param_sets, starts_with_param_set};

#[derive(Debug)]
pub struct Mp4Demux {
    /// The whole file, accumulated across chunks; parsed once at `Eos` (the `moov`
    /// may be at the end, so no sample can be emitted before the file is complete).
    buffer: Vec<u8>,
    /// Negotiation-time output codec, refined from the `moov` via `CapsChanged`.
    out_codec: VideoCodec,
    caps_sent: bool,
    sequence: u64,
    configured: bool,
    drained: bool,
}

impl Default for Mp4Demux {
    fn default() -> Self {
        Self::new()
    }
}

impl Mp4Demux {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            out_codec: VideoCodec::H264,
            caps_sent: false,
            sequence: 0,
            configured: false,
            drained: false,
        }
    }

    fn input_caps() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::Mp4 }
    }

    fn output_caps(codec: VideoCodec, width: Dim, height: Dim) -> Caps {
        Caps::CompressedVideo { codec, width, height, framerate: Rate::Any }
    }

    /// The negotiation placeholder: a fixatable `Range` (not `Dim::Any`, which
    /// fails Phase-2 fixate, e.g. a downstream `h264parse`), refined to the concrete
    /// moov geometry via `CapsChanged` at `Eos`. Mirrors `tsdemux` / `matroskademux`.
    fn nego_caps(codec: VideoCodec) -> Caps {
        Caps::CompressedVideo {
            codec,
            width: Dim::Range { min: 16, max: 65_535 },
            height: Dim::Range { min: 16, max: 65_535 },
            framerate: Rate::Range { min_q16: 1 << 16, max_q16: 240 << 16 },
        }
    }

    /// Parse the buffered file and emit the video track: announce the concrete
    /// caps from the `moov`, then every access unit (parameter sets prepended to
    /// the first, matching `fmp4demux`). Idempotent; runs once, at `Eos`.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if self.drained {
            return Ok(());
        }
        self.drained = true;
        let header = parse_header(&self.buffer)?;
        if !self.caps_sent {
            let caps =
                Self::output_caps(header.codec, Dim::Fixed(header.width), Dim::Fixed(header.height));
            out.push(PipelinePacket::CapsChanged(caps)).await?;
            self.out_codec = header.codec;
            self.caps_sent = true;
        }
        let codec = header.codec;
        let samples = parse_progressive(&self.buffer, header.timescale)?;
        let mut need_param_sets = true;
        for s in samples {
            let mut annexb = s.annexb;
            // The moov's config-record parameter sets are out-of-band, so prepend
            // them to the first access unit if it does not already carry them.
            if need_param_sets && !starts_with_param_set(&annexb, codec) {
                annexb = prepend_param_sets(&annexb, &header.param_sets, codec);
            }
            need_param_sets = false;
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(annexb.into_boxed_slice())),
                timing: FrameTiming {
                    pts_ns: s.pts_ns,
                    dts_ns: s.pts_ns,
                    duration_ns: s.duration_ns,
                    capture_ns: s.pts_ns,
                    arrival_ns: g2g_core::metrics::monotonic_ns(),
                    keyframe: s.keyframe,
                },
                sequence: self.sequence,
                meta: Default::default(),
            };
            self.sequence += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for Mp4Demux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // ByteStream{Mp4} in -> the video track out. The codec defaults to H.264
        // and is refined from the moov via CapsChanged at Eos (like fmp4demux).
        let codec = self.out_codec;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream { encoding: ByteStreamEncoding::Mp4 } => {
                CapsSet::one(Self::nego_caps(codec))
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(absolute_caps, Caps::ByteStream { encoding: ByteStreamEncoding::Mp4 }) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Progressive MP4 demuxer",
            "Codec/Demuxer",
            "Demuxes a whole-file (progressive) MP4 / QuickTime byte stream to its video track",
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // Accumulate; the moov may be at the end, so parse only at Eos.
                    self.buffer.extend_from_slice(slice.as_slice());
                }
                // The whole file is in hand: parse and emit, then the runner's
                // transform arm forwards the EOS.
                PipelinePacket::Eos => self.drain(out).await?,
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for Mp4Demux {
    fn pad_templates() -> Vec<PadTemplate> {
        let video = Self::nego_caps;
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(CapsSet::from_alternatives(Vec::from([
                video(VideoCodec::H264),
                video(VideoCodec::H265),
            ]))),
        ])
    }
}

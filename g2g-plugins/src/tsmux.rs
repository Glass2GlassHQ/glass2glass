//! MPEG-TS multiplexer element (M114): one elementary stream in
//! (`Caps::CompressedVideo{H264|H265}` Annex-B, or `Caps::Audio{Aac}` ADTS), an
//! MPEG-TS byte stream out (`Caps::ByteStream{MpegTs}`).
//!
//! Wraps the pure [`crate::mpegts::TsMuxer`], the inverse of
//! [`crate::tsdemux::TsDemux`]: each input access unit becomes a PES packet split
//! across 188-byte TS packets, with PAT + PMT emitted once up front. The PMT
//! stream type is read from the input caps at configure. CPU, `no_std` baseline.
//!
//! ```text
//! ... ! h264parse ! mpegtsmux ! filesink location=out.ts
//! ```
//!
//! Scope (v1): one program / one stream (a single input pad), mirroring the
//! single-stream `TsDemux`; multi-stream (A+V) muxing, PCR, and periodic PSI
//! re-insertion are follow-ups (DESIGN.md §4.17).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    Rate, VideoCodec,
};

use crate::mpegts::{TsMuxer, STREAM_TYPE_AAC, STREAM_TYPE_H264, STREAM_TYPE_H265};

/// The PMT `stream_type` for an input caps, or `None` if unsupported. Shared by
/// the single-input [`TsMux`] and the multi-input `tsmuxn::TsMux`.
pub(crate) fn stream_type_for(caps: &Caps) -> Option<u8> {
    match caps {
        Caps::CompressedVideo { codec: VideoCodec::H264, .. } => Some(STREAM_TYPE_H264),
        Caps::CompressedVideo { codec: VideoCodec::H265, .. } => Some(STREAM_TYPE_H265),
        Caps::Audio { format: AudioFormat::Aac, .. } => Some(STREAM_TYPE_AAC),
        _ => None,
    }
}

/// Muxes one elementary stream into an MPEG-TS byte stream.
#[derive(Debug)]
pub struct TsMux {
    /// Built at configure, once the input codec (and so the PMT stream type) is
    /// known.
    mux: Option<TsMuxer>,
    configured: bool,
    emitted: u64,
}

impl Default for TsMux {
    fn default() -> Self {
        Self::new()
    }
}

impl TsMux {
    pub fn new() -> Self {
        Self { mux: None, configured: false, emitted: 0 }
    }

    /// Count of TS byte frames forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The output it produces: an MPEG-TS byte stream.
    fn output_caps() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }
    }

    /// The elementary streams this muxer accepts on its sink pad.
    fn input_alternatives() -> Vec<Caps> {
        let video = |codec| Caps::CompressedVideo {
            codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([
            video(VideoCodec::H264),
            video(VideoCodec::H265),
            Caps::Audio { format: AudioFormat::Aac, channels: 0, sample_rate: 0 },
        ])
    }
}

impl AsyncElement for TsMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if stream_type_for(upstream_caps).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // Any supported elementary stream maps to one MPEG-TS byte stream.
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if stream_type_for(input).is_some() {
                CapsSet::one(Self::output_caps())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let stream_type = stream_type_for(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.mux = Some(TsMuxer::new(stream_type));
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
                    let mux = self.mux.as_mut().ok_or(G2gError::NotConfigured)?;
                    let pts_90khz = (frame.timing.pts_ns as u128 * 90_000 / 1_000_000_000) as u64;
                    let ts = mux.push_au(slice.as_slice(), Some(pts_90khz));
                    let out_frame = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(ts.into_boxed_slice())),
                        FrameTiming { pts_ns: frame.timing.pts_ns, ..FrameTiming::default() },
                        self.emitted,
                    );
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                // The runner's transform arm forwards EOS; nothing to flush here.
                PipelinePacket::Eos => {}
                // Input geometry / params don't change the TS framing.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for TsMux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(Self::input_alternatives())),
            PadTemplate::source(CapsSet::one(Self::output_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tsdemux::TsDemux;
    use g2g_core::{PushOutcome, RawVideoFormat};

    fn h264_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Any,
        }
    }

    #[derive(Default)]
    struct CaptureSink {
        frames: Vec<Vec<u8>>,
        eos: bool,
    }
    impl OutputSink for CaptureSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                match packet {
                    PipelinePacket::DataFrame(f) => {
                        if let MemoryDomain::System(s) = &f.domain {
                            self.frames.push(s.as_slice().to_vec());
                        }
                    }
                    PipelinePacket::Eos => self.eos = true,
                    _ => {}
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn h264_frame(au: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            FrameTiming { pts_ns, ..FrameTiming::default() },
            0,
        ))
    }

    #[test]
    fn caps_codec_in_byte_stream_out() {
        let m = TsMux::new();
        assert!(m.intercept_caps(&h264_caps()).is_ok());
        // Raw video / an existing byte stream have nothing to mux.
        let raw = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert!(m.intercept_caps(&raw).is_err());

        let CapsConstraint::DerivedOutput(f) = m.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        assert!(matches!(
            f(&h264_caps()).alternatives(),
            [Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }]
        ));
    }

    #[tokio::test]
    async fn element_round_trips_through_tsdemux() {
        let au0 = alloc::vec![0u8, 0, 0, 1, 0x65, 0xAA];
        let au1 = alloc::vec![0u8, 0, 0, 1, 0x41, 0xBB, 0xCC];

        let mut mux = TsMux::new();
        mux.configure_pipeline(&h264_caps()).unwrap();
        let mut ts_sink = CaptureSink::default();
        mux.process(h264_frame(au0.clone(), 10_000_000), &mut ts_sink).await.unwrap();
        mux.process(h264_frame(au1.clone(), 20_000_000), &mut ts_sink).await.unwrap();
        mux.process(PipelinePacket::Eos, &mut ts_sink).await.unwrap();
        assert!(!ts_sink.eos, "EOS is forwarded by the runner's arm, not the element");

        // Feed the muxed TS bytes back through the demuxer.
        let mut ts = Vec::new();
        for f in &ts_sink.frames {
            ts.extend_from_slice(f);
        }
        let mut demux = TsDemux::new();
        demux.configure_pipeline(&Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }).unwrap();
        let mut au_sink = CaptureSink::default();
        let ts_frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(ts.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        demux.process(PipelinePacket::DataFrame(ts_frame), &mut au_sink).await.unwrap();
        demux.process(PipelinePacket::Eos, &mut au_sink).await.unwrap();

        assert_eq!(au_sink.frames, alloc::vec![au0, au1], "AUs recovered through mux + demux");
        assert_eq!(mux.emitted(), 2);
    }
}

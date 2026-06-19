//! Ogg demuxer element (M116): `Caps::ByteStream{Ogg}` in, the Opus audio
//! elementary stream out (`Caps::Audio{Opus}`).
//!
//! Wraps the pure [`crate::ogg::OggDemuxer`], the Ogg sibling of
//! [`crate::mkvdemux::MkvDemux`]: it reassembles the logical bitstream's packets,
//! skips the codec setup headers, and forwards the audio packets. Once `OpusHead`
//! is parsed the channel count is known, so the demuxer refines the caps via
//! `CapsChanged` before the first frame. CPU, `no_std` baseline.
//!
//! ```text
//! filesrc(location=x.opus, caps=ByteStream{Ogg}) ! oggdemux ! <opus decoder>
//! ```
//!
//! Scope (v1): one logical bitstream, Opus output (a non-Opus Ogg is parsed but
//! not forwarded, since the output pad is Opus-typed). Granule-position timing
//! and Vorbis output are follow-ups (packets carry no PTS yet).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

use crate::ogg::{OggCodec, OggDemuxer};

/// Demuxes an Ogg byte stream into its Opus audio elementary stream.
#[derive(Debug)]
pub struct OggDemux {
    demux: OggDemuxer,
    configured: bool,
    emitted: u64,
    last_caps: Option<Caps>,
}

impl Default for OggDemux {
    fn default() -> Self {
        Self::new()
    }
}

impl OggDemux {
    pub fn new() -> Self {
        Self { demux: OggDemuxer::new(), configured: false, emitted: 0, last_caps: None }
    }

    /// Count of audio packets forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn input_caps() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::Ogg }
    }

    /// The placeholder output: Opus with a sentinel channels/rate, refined from
    /// `OpusHead` via `CapsChanged` once the stream is parsed.
    fn output_caps() -> Caps {
        Caps::Audio { format: AudioFormat::Opus, channels: 0, sample_rate: 0 }
    }

    /// The concrete Opus caps once `OpusHead` is parsed, or `None` until then /
    /// for a non-Opus stream.
    fn concrete_caps(&self) -> Option<Caps> {
        let info = self.demux.info()?;
        if info.codec == OggCodec::Opus && info.sample_rate > 0 {
            Some(Caps::Audio {
                format: AudioFormat::Opus,
                channels: info.channels.max(1),
                sample_rate: info.sample_rate,
            })
        } else {
            None
        }
    }

    /// Emit a `CapsChanged` once the Opus parameters are known, then forward each
    /// audio packet. A non-Opus stream is drained and dropped (Opus-typed pad).
    async fn emit_ready(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if let Some(caps) = self.concrete_caps() {
            if self.last_caps.as_ref() != Some(&caps) {
                out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                self.last_caps = Some(caps);
            }
        }
        let is_opus = self.demux.info().map(|i| i.codec) == Some(OggCodec::Opus);
        for packet in self.demux.take_packets() {
            if !is_opus {
                continue;
            }
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(packet.into_boxed_slice())),
                FrameTiming::default(),
                self.emitted,
            );
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for OggDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::ByteStream { encoding: ByteStreamEncoding::Ogg } => {
                CapsSet::one(Self::output_caps())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(absolute_caps, Caps::ByteStream { encoding: ByteStreamEncoding::Ogg }) {
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
                    self.demux.push_data(slice.as_slice());
                    self.emit_ready(out).await?;
                }
                PipelinePacket::Eos => {
                    // Emit any final packets; the runner's transform arm forwards EOS.
                    self.emit_ready(out).await?;
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

impl PadTemplates for OggDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(CapsSet::one(Self::output_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{PushOutcome, RawVideoFormat, Dim, Rate};

    /// Build one Ogg page carrying `packets` for `serial` (mirrors the parser
    /// test helper).
    fn page(header_type: u8, serial: u32, seq: u32, packets: &[&[u8]]) -> Vec<u8> {
        let mut table = Vec::new();
        let mut body = Vec::new();
        for p in packets {
            let mut n = p.len();
            loop {
                let seg = n.min(255);
                table.push(seg as u8);
                n -= seg;
                if seg < 255 {
                    break;
                }
            }
            body.extend_from_slice(p);
        }
        let mut out = b"OggS".to_vec();
        out.push(0);
        out.push(header_type);
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&serial.to_le_bytes());
        out.extend_from_slice(&seq.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.push(table.len() as u8);
        out.extend_from_slice(&table);
        out.extend_from_slice(&body);
        out
    }

    fn opus_head(channels: u8) -> Vec<u8> {
        let mut h = b"OpusHead".to_vec();
        h.push(1);
        h.push(channels);
        h.extend_from_slice(&[0, 0]);
        h.extend_from_slice(&48_000u32.to_le_bytes());
        h.extend_from_slice(&[0, 0, 0]);
        h
    }

    #[derive(Default)]
    struct CaptureSink {
        caps: Vec<Caps>,
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
                    PipelinePacket::CapsChanged(c) => self.caps.push(c),
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

    #[test]
    fn caps_byte_stream_in_opus_out() {
        let d = OggDemux::new();
        assert!(d.intercept_caps(&OggDemux::input_caps()).is_ok());
        let raw = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert!(d.intercept_caps(&raw).is_err());
        // The Matroska byte stream is the wrong container.
        let mkv = Caps::ByteStream { encoding: ByteStreamEncoding::Matroska };
        assert!(d.intercept_caps(&mkv).is_err());
    }

    #[tokio::test]
    async fn demuxes_opus_with_refined_caps() {
        let serial = 7;
        let mut stream = Vec::new();
        stream.extend_from_slice(&page(0x02, serial, 0, &[&opus_head(2)]));
        stream.extend_from_slice(&page(0x00, serial, 1, &[b"OpusTags"]));
        stream.extend_from_slice(&page(0x00, serial, 2, &[&[0x11, 0x22], &[0x33]]));

        let mut d = OggDemux::new();
        d.configure_pipeline(&OggDemux::input_caps()).unwrap();
        let mut sink = CaptureSink::default();
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        d.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();
        d.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        assert_eq!(
            sink.caps,
            alloc::vec![Caps::Audio { format: AudioFormat::Opus, channels: 2, sample_rate: 48_000 }]
        );
        assert_eq!(sink.frames, alloc::vec![alloc::vec![0x11, 0x22], alloc::vec![0x33]]);
        assert!(!sink.eos, "EOS is forwarded by the runner's arm, not the element");
        assert_eq!(d.emitted(), 2);
    }
}

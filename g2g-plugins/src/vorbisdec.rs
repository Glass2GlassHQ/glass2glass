//! Vorbis audio decoder element (VorbisDec, `vorbis` feature, M777):
//! `Audio{Vorbis}` in, interleaved `Audio{PcmS16Le}` out, via symphonia's
//! pure-Rust decoder (no system library, unlike the libopus-backed
//! [`crate::opusdec::OpusDec`]).
//!
//! Vorbis packets are container-framed (one packet per Ogg packet / Matroska
//! block) and the decoder needs two of the stream's three header packets: the
//! identification header (`\x01vorbis`, channels + rate) and the setup header
//! (`\x05vorbis`, the codebooks). [`crate::oggdemux::OggDemux`] forwards all
//! three in-band ahead of the audio; the prefixes are unambiguous (an audio
//! packet's first byte always has bit 0 clear), so the decoder stashes ident +
//! setup, ignores the comment header, and builds once setup arrives. Output PCM
//! carries the demuxer's packet timing when present (M778: durations from the
//! setup-header mode tables, end-granule clamped, so the decoded PCM is trimmed
//! to match), else it is stamped from the decoded sample count.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use symphonia_codec_vorbis::VorbisDecoder;
use symphonia_core::audio::GenericAudioBufferRef;
use symphonia_core::codecs::audio::well_known::CODEC_ID_VORBIS;
use symphonia_core::codecs::audio::{AudioCodecParameters, AudioDecoder, AudioDecoderOptions};
use symphonia_core::packet::PacketRef;
use symphonia_core::units::{Duration, Timestamp};

/// Decodes a Vorbis elementary stream into raw interleaved S16LE PCM.
pub struct VorbisDec {
    /// The identification header (`\x01vorbis`), held until setup arrives.
    ident: Option<Vec<u8>>,
    dec: Option<VorbisDecoder>,
    channels: u8,
    sample_rate: u32,
    /// Last emitted output caps, to suppress an unchanged `CapsChanged`.
    last_out: Option<Caps>,
    /// Decoded samples (per channel) so far, the PTS accumulator.
    samples: u64,
    sequence: u64,
    configured: bool,
}

impl core::fmt::Debug for VorbisDec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // symphonia's VorbisDecoder is not Debug; report the configuration.
        f.debug_struct("VorbisDec")
            .field("channels", &self.channels)
            .field("sample_rate", &self.sample_rate)
            .field("sequence", &self.sequence)
            .field("configured", &self.configured)
            .finish()
    }
}

impl Default for VorbisDec {
    fn default() -> Self {
        Self::new()
    }
}

impl VorbisDec {
    pub fn new() -> Self {
        Self {
            ident: None,
            dec: None,
            channels: 0,
            sample_rate: 0,
            last_out: None,
            samples: 0,
            sequence: 0,
            configured: false,
        }
    }

    /// Build the symphonia decoder from the stashed identification header plus
    /// the just-arrived setup header (its non-laced extradata form is the two
    /// concatenated). Channels / rate come from the identification header.
    fn build_decoder(&mut self, setup: &[u8]) -> Result<(), G2gError> {
        let ident = self.ident.as_ref().ok_or(G2gError::NotConfigured)?;
        // Identification header: magic(7), version(4), channels at offset 11,
        // sample rate (LE u32) at offset 12 (validated by the parse below).
        if ident.len() < 16 {
            return Err(G2gError::CapsMismatch);
        }
        self.channels = ident[11];
        self.sample_rate = u32::from_le_bytes([ident[12], ident[13], ident[14], ident[15]]);
        let mut extra = Vec::with_capacity(ident.len() + setup.len());
        extra.extend_from_slice(ident);
        extra.extend_from_slice(setup);
        let mut params = AudioCodecParameters::new();
        params
            .for_codec(CODEC_ID_VORBIS)
            .with_extra_data(extra.into_boxed_slice());
        self.dec = Some(
            VorbisDecoder::try_new(&params, &AudioDecoderOptions::default())
                .map_err(|_| G2gError::CapsMismatch)?,
        );
        Ok(())
    }

    fn output_caps(&self) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: self.channels,
            sample_rate: self.sample_rate,
        }
    }

    /// Decode one Vorbis packet to interleaved i16 samples (empty for the
    /// stream-priming first packet, whose window has no predecessor).
    fn decode(&mut self, packet: &[u8]) -> Result<Vec<i16>, G2gError> {
        let dec = self.dec.as_mut().ok_or(G2gError::NotConfigured)?;
        let pkt = PacketRef::new(0, Timestamp::ZERO, Duration::ZERO, packet);
        let buf: GenericAudioBufferRef<'_> = dec
            .decode_ref(&pkt)
            .map_err(|_| G2gError::Hardware(g2g_core::HardwareError::Other))?;
        let mut pcm = alloc::vec![0i16; buf.samples_interleaved()];
        buf.copy_to_slice_interleaved::<i16, _>(&mut pcm[..]);
        Ok(pcm)
    }
}

impl AsyncElement for VorbisDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio {
                format: AudioFormat::Vorbis,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Native `DerivedOutput`: Vorbis in -> interleaved `PcmS16Le` out. Channel
    /// count and rate are only known once the identification header is parsed,
    /// so the output advertises the wildcards with a concrete-rate first
    /// alternative (the fixate fallback; see the same shape in
    /// [`crate::ffmpegaudiodec`]) and the real values arrive via `CapsChanged`.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio {
                format: AudioFormat::Vorbis,
                ..
            } => {
                let pcm = |sample_rate| Caps::Audio {
                    format: AudioFormat::PcmS16Le,
                    channels: ANY_CHANNELS,
                    sample_rate,
                };
                CapsSet::from_alternatives(alloc::vec![pcm(48_000), pcm(ANY_SAMPLE_RATE)])
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Audio {
                format: AudioFormat::Vorbis,
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
            "Vorbis audio decoder",
            "Codec/Decoder/Audio",
            "Decodes Vorbis to interleaved PcmS16Le (pure Rust, symphonia)",
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
                    // In-band header packets (audio packets have bit 0 of the
                    // first byte clear, so the prefixes cannot alias): stash
                    // ident, skip the comment, build on setup. Codec config,
                    // no PCM out.
                    if slice.starts_with(b"\x01vorbis") {
                        self.ident = Some(slice.to_vec());
                        return Ok(());
                    }
                    if slice.starts_with(b"\x03vorbis") {
                        return Ok(());
                    }
                    if slice.starts_with(b"\x05vorbis") {
                        let setup = slice.to_vec();
                        self.build_decoder(&setup)?;
                        return Ok(());
                    }
                    let pcm = self.decode(slice)?;
                    // The first audio packet primes the overlap window and
                    // decodes to nothing; consume it without an empty frame.
                    if pcm.is_empty() {
                        return Ok(());
                    }
                    let new_caps = self.output_caps();
                    if self.last_out.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
                        self.last_out = Some(new_caps);
                    }
                    let channels = usize::from(self.channels.max(1));
                    let decoded = pcm.len() / channels;
                    let ns = |s: u64| {
                        (s as u128 * 1_000_000_000 / self.sample_rate.max(1) as u128) as u64
                    };
                    // Demux-provided timing (M778): the packet's timeline
                    // duration arrives end-granule clamped, so trim the decoded
                    // PCM to it (the final block's encoder padding drops here).
                    // Untimed input (duration 0) self-stamps from the decoded
                    // sample count.
                    let (pts_ns, duration_ns, keep) = if frame.timing.duration_ns > 0 {
                        let k = ((frame.timing.duration_ns as u128
                            * u128::from(self.sample_rate.max(1))
                            + 500_000_000)
                            / 1_000_000_000) as usize;
                        (
                            frame.timing.pts_ns,
                            frame.timing.duration_ns,
                            k.min(decoded),
                        )
                    } else {
                        let pts = ns(self.samples);
                        let dur = ns(self.samples + decoded as u64).saturating_sub(pts);
                        (pts, dur, decoded)
                    };
                    self.samples += keep as u64;
                    let mut bytes = Vec::with_capacity(keep * channels * 2);
                    for s in &pcm[..keep * channels] {
                        bytes.extend_from_slice(&s.to_le_bytes());
                    }
                    let decoded = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                        FrameTiming {
                            pts_ns,
                            dts_ns: pts_ns,
                            duration_ns,
                            ..FrameTiming::default()
                        },
                        self.sequence,
                    );
                    self.sequence += 1;
                    out.push(PipelinePacket::DataFrame(decoded)).await?;
                }
                PipelinePacket::Flush => {
                    // A seek/flush restarts the DSP window and sample clock; the
                    // re-read stream re-sends its headers.
                    if let Some(dec) = self.dec.as_mut() {
                        dec.reset();
                    }
                    self.samples = 0;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::CapsChanged(c) => match &c {
                    // The demuxer's input refine: the decoder reads its real
                    // parameters from the in-band headers, nothing to do.
                    Caps::Audio {
                        format: AudioFormat::Vorbis,
                        ..
                    } => {}
                    // The runner's pre-fixed forward output caps: forward on.
                    Caps::Audio {
                        format: AudioFormat::PcmS16Le,
                        ..
                    } => {
                        out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                        self.last_out = Some(c);
                    }
                    _ => return Err(G2gError::CapsMismatch),
                },
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for VorbisDec {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Caps::Audio {
                format: AudioFormat::Vorbis,
                channels: ANY_CHANNELS,
                sample_rate: ANY_SAMPLE_RATE,
            })),
            PadTemplate::source(CapsSet::one(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 48_000,
            })),
        ])
    }
}

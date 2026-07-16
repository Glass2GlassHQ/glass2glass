//! Opus audio decoder element (OpusDec, `opus` feature): `Audio{Opus}` in,
//! `Audio{PcmS16Le}` out, via libopus through the `audiopus` crate. The decode
//! sibling of [`crate::opusenc::OpusEnc`]; it consumes the packets
//! [`crate::opusparse::OpusParse`] frames.
//!
//! Each Opus packet is self-contained, so decode is one packet in, one PCM frame
//! out. libopus always decodes at 48 kHz ([`crate::opusparse::OPUS_RATE_HZ`])
//! regardless of the coded bandwidth, so the output rate is constant; the channel
//! count is fixed at configure (the decoder is created for a channel count). A
//! `CapsChanged` carries the output format before the first frame.
//!
//! Scope (v1): 48 kHz mono/stereo, S16LE output. Float output and packet-loss
//! concealment (decode of a `None` packet) are follow-ups.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, ANY_CHANNELS,
    ANY_SAMPLE_RATE,
};

use audiopus::coder::Decoder;
use audiopus::packet::Packet;
use audiopus::{Channels, MutSignals, SampleRate};

use crate::opusparse::OPUS_RATE_HZ;

/// Largest Opus frame is 120 ms; at 48 kHz that is 5760 samples per channel. The
/// decode output buffer is sized for it so any single packet fits.
const MAX_FRAME_SAMPLES: usize = (OPUS_RATE_HZ as usize * 120) / 1000;

/// Decodes an Opus elementary stream into raw interleaved S16LE PCM.
pub struct OpusDec {
    channels: u8,
    dec: Option<Decoder>,
    caps_sent: bool,
    sequence: u64,
    configured: bool,
}

impl core::fmt::Debug for OpusDec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // audiopus' Decoder is not Debug; report the configuration instead.
        f.debug_struct("OpusDec")
            .field("channels", &self.channels)
            .field("sequence", &self.sequence)
            .field("configured", &self.configured)
            .finish()
    }
}

impl Default for OpusDec {
    fn default() -> Self {
        Self::new()
    }
}

impl OpusDec {
    pub fn new() -> Self {
        Self { channels: 0, dec: None, caps_sent: false, sequence: 0, configured: false }
    }

    /// Sink pad template: Opus at any channel count / nominal rate. The auto-plug
    /// matcher intersects this against the demuxer's caps, which carry a concrete
    /// channel count (mono or stereo) but the "unknown until parsed" rate
    /// placeholder (compressed rate intersects strictly, so a fixed rate here would
    /// not match `rate: 0`). OpusDec ignores the nominal rate anyway: Opus always
    /// decodes at 48 kHz, and the real channel count is read in `configure_pipeline`.
    fn input_template() -> Caps {
        Caps::Audio { format: AudioFormat::Opus, channels: ANY_CHANNELS, sample_rate: ANY_SAMPLE_RATE }
    }

    fn output_caps(&self) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: self.channels,
            sample_rate: OPUS_RATE_HZ,
        }
    }

    /// Decode one Opus packet into interleaved S16LE bytes.
    fn decode(&mut self, opus: &[u8]) -> Result<Vec<u8>, G2gError> {
        let channels = self.channels as usize;
        let dec = self.dec.as_mut().ok_or(G2gError::NotConfigured)?;
        let packet = Packet::try_from(opus).map_err(|_| G2gError::CapsMismatch)?;
        let mut pcm = alloc::vec![0i16; MAX_FRAME_SAMPLES * channels];
        let per_channel = {
            let signals = MutSignals::try_from(&mut pcm[..]).map_err(|_| G2gError::CapsMismatch)?;
            dec.decode(Some(packet), signals, false).map_err(|_| G2gError::CapsMismatch)?
        };
        pcm.truncate(per_channel * channels);
        // Serialize interleaved i16 to little-endian bytes.
        let mut bytes = Vec::with_capacity(pcm.len() * 2);
        for s in pcm {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        Ok(bytes)
    }
}

impl AsyncElement for OpusDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio { format: AudioFormat::Opus, .. } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio { format: AudioFormat::Opus, channels, .. } if *channels == 1 || *channels == 2 => {
                CapsSet::one(Caps::Audio {
                    format: AudioFormat::PcmS16Le,
                    channels: *channels,
                    sample_rate: OPUS_RATE_HZ,
                })
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::Audio { format: AudioFormat::Opus, channels, .. } = absolute_caps else {
            return Err(G2gError::CapsMismatch);
        };
        let ch = match channels {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            _ => return Err(G2gError::CapsMismatch),
        };
        let dec = Decoder::new(SampleRate::Hz48000, ch).map_err(|_| G2gError::CapsMismatch)?;
        self.dec = Some(dec);
        self.channels = *channels;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Opus audio decoder",
            "Codec/Decoder/Audio",
            "Decodes Opus to raw S16LE PCM",
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
                    let pcm = self.decode(slice.as_slice())?;
                    if !self.caps_sent {
                        out.push(PipelinePacket::CapsChanged(self.output_caps())).await?;
                        self.caps_sent = true;
                    }
                    let decoded = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
                        frame.timing,
                        self.sequence,
                    );
                    self.sequence += 1;
                    out.push(PipelinePacket::DataFrame(decoded)).await?;
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

impl PadTemplates for OpusDec {
    fn pad_templates() -> Vec<PadTemplate> {
        let out =
            Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: OPUS_RATE_HZ };
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_template())),
            PadTemplate::source(CapsSet::one(out)),
        ])
    }
}

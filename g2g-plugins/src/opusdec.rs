//! Opus audio decoder element (OpusDec, `opus` feature): `Audio{Opus}` in,
//! `Audio{PcmS16Le}` out, via libopus through the `audiopus` crate. The decode
//! sibling of [`crate::opusenc::OpusEnc`]; it consumes the packets
//! [`crate::opusparse::OpusParse`] frames.
//!
//! Each Opus packet is self-contained, so decode is one packet in, one PCM frame
//! out. libopus always decodes at 48 kHz ([`crate::opusparse::OPUS_RATE_HZ`])
//! regardless of the coded bandwidth, so the output rate is constant. The channel
//! count comes from `OpusHead`; a demuxer (OggDemux) only knows it once it has
//! parsed the stream, so at negotiation the input channels can be the
//! `ANY_CHANNELS` placeholder. The output therefore advertises `ANY_CHANNELS`
//! (fixated to stereo for the edge) and the decoder is (re)built when the real
//! channel count arrives via a `CapsChanged`. A `CapsChanged` carries the output
//! format before the first frame.
//!
//! Pre-skip / end-trim: Opus streams carry encoder lookahead (pre-skip) at the
//! head and codec padding at the tail. `OggDemux` forwards the `OpusHead` in-band
//! (its pre-skip drops the leading output samples) and marks the final packet(s)
//! short via `duration_ns` (the end-of-stream granule trim), so the decoded PCM
//! matches ffmpeg / gstreamer sample-for-sample. Streams with no `OpusHead` and
//! no per-frame duration (RTP) decode untrimmed, as before.
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
    /// Last emitted output caps, to suppress re-emitting an unchanged
    /// `CapsChanged` and to detect a channel-count change.
    last_out: Option<Caps>,
    sequence: u64,
    configured: bool,
    /// Leading 48 kHz output samples (per channel) to discard: the Opus encoder
    /// lookahead from `OpusHead`. `0` when no header was seen (e.g. the RTP path).
    pre_skip: u32,
    /// Running count of decoded samples (per channel) across all frames, to place
    /// the pre-skip window against the stream, not each frame in isolation.
    decoded_samples: u64,
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
        Self {
            channels: 0,
            dec: None,
            last_out: None,
            sequence: 0,
            configured: false,
            pre_skip: 0,
            decoded_samples: 0,
        }
    }

    /// (Re)create the libopus decoder for a concrete channel count. Called from
    /// `configure_pipeline` when the negotiated input already carries a real
    /// count, and from `process` when the demuxer's `CapsChanged` delivers it.
    fn build_decoder(&mut self, channels: u8) -> Result<(), G2gError> {
        let ch = match channels {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            _ => return Err(G2gError::CapsMismatch),
        };
        self.dec = Some(Decoder::new(SampleRate::Hz48000, ch).map_err(|_| G2gError::CapsMismatch)?);
        self.channels = channels;
        Ok(())
    }

    /// Sink pad template: Opus at any channel count / nominal rate. The auto-plug
    /// matcher intersects this against the demuxer's caps, which carry a concrete
    /// channel count (mono or stereo) but the "unknown until parsed" rate
    /// placeholder (compressed rate intersects strictly, so a fixed rate here would
    /// not match `rate: 0`). OpusDec ignores the nominal rate anyway: Opus always
    /// decodes at 48 kHz, and the real channel count is read in `configure_pipeline`.
    fn input_template() -> Caps {
        Caps::Audio {
            format: AudioFormat::Opus,
            channels: ANY_CHANNELS,
            sample_rate: ANY_SAMPLE_RATE,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: self.channels,
            sample_rate: OPUS_RATE_HZ,
        }
    }

    /// Decode one Opus packet, returning interleaved i16 samples and the
    /// per-channel sample count.
    fn decode(&mut self, opus: &[u8]) -> Result<(Vec<i16>, usize), G2gError> {
        let channels = self.channels as usize;
        let dec = self.dec.as_mut().ok_or(G2gError::NotConfigured)?;
        let packet = Packet::try_from(opus).map_err(|_| G2gError::CapsMismatch)?;
        let mut pcm = alloc::vec![0i16; MAX_FRAME_SAMPLES * channels];
        let per_channel = {
            let signals = MutSignals::try_from(&mut pcm[..]).map_err(|_| G2gError::CapsMismatch)?;
            dec.decode(Some(packet), signals, false)
                .map_err(|_| G2gError::CapsMismatch)?
        };
        pcm.truncate(per_channel * channels);
        Ok((pcm, per_channel))
    }

    /// Decode `opus` and serialize only the valid window to little-endian bytes.
    /// The window drops the pre-skip lookahead at the stream head and any padding
    /// past `keep` (per-channel valid count for this frame, from `duration_ns`;
    /// `None` keeps the whole frame). Returns the trimmed S16LE bytes.
    fn decode_trimmed(&mut self, opus: &[u8], keep: Option<u64>) -> Result<Vec<u8>, G2gError> {
        let channels = self.channels as usize;
        let (pcm, per_channel) = self.decode(opus)?;
        let n = per_channel as u64;
        // Head drop: pre-skip samples still ahead of this frame's start.
        let head = self
            .pre_skip
            .saturating_sub(self.decoded_samples.min(u32::MAX as u64) as u32)
            as u64;
        let head = head.min(n);
        // Tail cap: keep at most `keep` per-channel samples from the frame start.
        let end = keep.map_or(n, |k| k.min(n)).max(head);
        self.decoded_samples = self.decoded_samples.saturating_add(n);
        let start_i = head as usize * channels;
        let end_i = end as usize * channels;
        let mut bytes = Vec::with_capacity((end_i - start_i) * 2);
        for &s in &pcm[start_i..end_i] {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        Ok(bytes)
    }
}

/// Per-channel valid sample count encoded in a frame's `duration_ns` (48 kHz),
/// or `None` when unset (`0`). Rounds to the nearest sample; the demuxer's
/// truncating ns conversion round-trips back to the exact count.
fn duration_to_samples(duration_ns: u64) -> Option<u64> {
    if duration_ns == 0 {
        return None;
    }
    Some(
        (duration_ns
            .saturating_mul(48_000)
            .saturating_add(500_000_000))
            / 1_000_000_000,
    )
}

/// Channel count and pre-skip from an in-band `OpusHead` (RFC 7845), or `None`
/// if `packet` is not one. Offset 9 is the channel count, offset 10 the LE u16
/// pre-skip. A full family-0 header is 19 bytes; only the fixed prefix is read.
fn parse_opus_head(packet: &[u8]) -> Option<(u8, u16)> {
    if packet.len() >= 12 && packet.starts_with(b"OpusHead") {
        let channels = packet[9];
        let pre_skip = u16::from_le_bytes([packet[10], packet[11]]);
        (channels >= 1).then_some((channels, pre_skip))
    } else {
        None
    }
}

impl AsyncElement for OpusDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio {
                format: AudioFormat::Opus,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Native `DerivedOutput`: Opus in -> interleaved `PcmS16Le` out at 48 kHz.
    /// The output channel count is the `ANY_CHANNELS` placeholder, not the input
    /// count: a demuxer only knows the real count once it parses `OpusHead`, so
    /// the negotiated input can be `ANY_CHANNELS`. `fixate` collapses the output
    /// placeholder to stereo for the edge; the real count arrives via the
    /// `CapsChanged` the demuxer emits and the decoded frame carries.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio {
                format: AudioFormat::Opus,
                ..
            } => CapsSet::one(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: ANY_CHANNELS,
                sample_rate: OPUS_RATE_HZ,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::Audio {
            format: AudioFormat::Opus,
            channels,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        self.configured = true;
        // A concrete count (the direct OpusParse path) builds the decoder now; the
        // `ANY_CHANNELS` (0) placeholder defers it to the demuxer's `CapsChanged`.
        if *channels == 1 || *channels == 2 {
            self.build_decoder(*channels)?;
        }
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
                    // An in-band OpusHead is codec config, not audio: read its
                    // channel count + pre-skip, (re)build the decoder, and consume
                    // it (no PCM out). The demuxer forwards it before the audio.
                    if let Some((channels, pre_skip)) = parse_opus_head(slice.as_slice()) {
                        if self.channels != channels {
                            self.build_decoder(channels)?;
                        }
                        self.pre_skip = pre_skip as u32;
                        self.decoded_samples = 0;
                        return Ok(());
                    }
                    let keep = duration_to_samples(frame.timing.duration_ns);
                    let pcm = self.decode_trimmed(slice.as_slice(), keep)?;
                    // A frame fully inside the pre-skip window trims to nothing;
                    // consume it without emitting an empty PCM frame.
                    if pcm.is_empty() {
                        return Ok(());
                    }
                    let new_caps = self.output_caps();
                    if self.last_out.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
                        self.last_out = Some(new_caps);
                    }
                    let decoded = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
                        frame.timing,
                        self.sequence,
                    );
                    self.sequence += 1;
                    out.push(PipelinePacket::DataFrame(decoded)).await?;
                }
                PipelinePacket::Flush => {
                    // A seek/flush restarts sample accounting; the re-read stream
                    // re-sends its OpusHead, which resets pre-skip again.
                    self.pre_skip = 0;
                    self.decoded_samples = 0;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::CapsChanged(c) => match &c {
                    // The demuxer's input refine carries the real channel count
                    // (from `OpusHead`); (re)build the decoder for it. The decoder
                    // re-derives its own output, so this is not forwarded.
                    Caps::Audio {
                        format: AudioFormat::Opus,
                        channels,
                        ..
                    } => {
                        self.build_decoder(*channels)?;
                    }
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

impl PadTemplates for OpusDec {
    fn pad_templates() -> Vec<PadTemplate> {
        let out = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: OPUS_RATE_HZ,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_template())),
            PadTemplate::source(CapsSet::one(out)),
        ])
    }
}

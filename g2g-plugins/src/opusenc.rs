//! Opus audio encoder element (OpusEnc, `opus` feature): `Audio{PcmS16Le}` in,
//! `Audio{Opus}` out, via libopus through the `audiopus` crate. The encode
//! sibling of [`crate::opusdec::OpusDec`] and the producer that
//! [`crate::opusparse::OpusParse`] reads.
//!
//! Opus only encodes whole frames of one of a fixed set of durations (2.5..60
//! ms). PCM `DataFrame`s arrive at arbitrary sizes, so the element *buffers*
//! interleaved samples and emits one Opus packet per fixed frame
//! ([`FRAME_MS`], 20 ms = 960 samples/channel at 48 kHz, the common default).
//! At EOS a partial tail is zero-padded to one full frame so no audio is lost.
//!
//! Scope (v1): 48 kHz mono/stereo S16LE. 48 kHz because Opus always *decodes* at
//! 48 kHz ([`crate::opusparse::OPUS_RATE_HZ`]), so the whole pipeline stays at
//! that rate without a resample; other input rates need an upstream
//! `AudioResample`. Bitrate is builder-set (`with_bitrate`), default libopus
//! auto. Float input + other frame durations are follow-ups.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

use audiopus::coder::Encoder;
use audiopus::{Application, Bitrate, Channels, SampleRate};

use crate::opusparse::OPUS_RATE_HZ;

/// Opus frame duration this encoder emits, in milliseconds. 20 ms is the Opus
/// default and a good latency/efficiency balance; at 48 kHz it is 960
/// samples/channel.
pub const FRAME_MS: u32 = 20;

/// Samples per channel in one emitted frame (48 kHz * 20 ms / 1000).
const FRAME_SAMPLES: usize = (OPUS_RATE_HZ as usize * FRAME_MS as usize) / 1000;

/// One frame's duration in nanoseconds, the PTS step between emitted packets.
const FRAME_NS: u64 = (FRAME_MS as u64) * 1_000_000;

/// Maximum Opus packet size for a 20 ms stereo frame; the libopus-recommended
/// output scratch (`1275 * 3 + 7`).
const MAX_PACKET: usize = 4_000;

/// Encodes raw interleaved S16LE PCM into an Opus elementary stream.
pub struct OpusEnc {
    channels: u8,
    bitrate: Bitrate,
    enc: Option<Encoder>,
    /// Interleaved S16 samples not yet packed into a full Opus frame.
    buf: Vec<i16>,
    /// PTS for the next packet, anchored to the first input frame's PTS and
    /// advanced one frame duration per emitted packet.
    next_pts_ns: Option<u64>,
    caps_sent: bool,
    emitted: u64,
    configured: bool,
}

impl core::fmt::Debug for OpusEnc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // audiopus' Encoder is not Debug; report the configuration instead.
        f.debug_struct("OpusEnc")
            .field("channels", &self.channels)
            .field("buffered_samples", &self.buf.len())
            .field("emitted", &self.emitted)
            .field("configured", &self.configured)
            .finish()
    }
}

impl Default for OpusEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl OpusEnc {
    pub fn new() -> Self {
        Self {
            channels: 0,
            bitrate: Bitrate::Auto,
            enc: None,
            buf: Vec::new(),
            next_pts_ns: None,
            caps_sent: false,
            emitted: 0,
            configured: false,
        }
    }

    /// Set the target bitrate in bits per second (e.g. 64_000). Default is
    /// libopus auto (rate chosen from the signal and frame size).
    pub fn with_bitrate(mut self, bits_per_second: i32) -> Self {
        self.bitrate = Bitrate::BitsPerSecond(bits_per_second);
        self
    }

    /// Count of Opus packets emitted.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn input_template() -> Caps {
        // Audio caps carry no `Any`; pin the supported shape (48 kHz stereo).
        Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: OPUS_RATE_HZ }
    }

    fn output_caps(&self) -> Caps {
        Caps::Audio { format: AudioFormat::Opus, channels: self.channels, sample_rate: OPUS_RATE_HZ }
    }

    /// Encode one full interleaved frame (`FRAME_SAMPLES * channels` samples)
    /// into an owned Opus packet.
    fn encode_frame(&self, frame: &[i16]) -> Result<Vec<u8>, G2gError> {
        let enc = self.enc.as_ref().ok_or(G2gError::NotConfigured)?;
        let mut out = alloc::vec![0u8; MAX_PACKET];
        let len = enc.encode(frame, &mut out).map_err(|_| G2gError::CapsMismatch)?;
        out.truncate(len);
        Ok(out)
    }

    /// Drain as many full frames as `buf` holds, returning `(packet, pts)` for
    /// each. PTS advances one frame duration per packet from the anchor.
    fn drain_frames(&mut self) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let frame_len = FRAME_SAMPLES * self.channels as usize;
        let mut packets = Vec::new();
        while self.buf.len() >= frame_len {
            let frame: Vec<i16> = self.buf.drain(..frame_len).collect();
            let packet = self.encode_frame(&frame)?;
            let pts = self.next_pts_ns.unwrap_or(0);
            self.next_pts_ns = Some(pts + FRAME_NS);
            packets.push((packet, pts));
        }
        Ok(packets)
    }

    /// At EOS, zero-pad a partial tail to one full frame and encode it, so the
    /// final samples are not dropped. Returns the flushed packet, if any.
    fn flush(&mut self) -> Result<Option<(Vec<u8>, u64)>, G2gError> {
        if self.buf.is_empty() {
            return Ok(None);
        }
        let frame_len = FRAME_SAMPLES * self.channels as usize;
        self.buf.resize(frame_len, 0); // pad with silence to a whole frame
        let frame: Vec<i16> = self.buf.drain(..frame_len).collect();
        let packet = self.encode_frame(&frame)?;
        let pts = self.next_pts_ns.unwrap_or(0);
        self.next_pts_ns = Some(pts + FRAME_NS);
        Ok(Some((packet, pts)))
    }

    async fn emit(
        &mut self,
        packets: Vec<(Vec<u8>, u64)>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        if !packets.is_empty() && !self.caps_sent {
            out.push(PipelinePacket::CapsChanged(self.output_caps())).await?;
            self.caps_sent = true;
        }
        for (data, pts_ns) in packets {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
                FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
                self.emitted,
            );
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for OpusEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio { format: AudioFormat::PcmS16Le, channels, sample_rate }
                if (*channels == 1 || *channels == 2) && *sample_rate == OPUS_RATE_HZ =>
            {
                Ok(upstream_caps.clone())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio { format: AudioFormat::PcmS16Le, channels, sample_rate }
                if (*channels == 1 || *channels == 2) && *sample_rate == OPUS_RATE_HZ =>
            {
                CapsSet::one(Caps::Audio {
                    format: AudioFormat::Opus,
                    channels: *channels,
                    sample_rate: OPUS_RATE_HZ,
                })
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::Audio { format: AudioFormat::PcmS16Le, channels, sample_rate } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if *sample_rate != OPUS_RATE_HZ {
            return Err(G2gError::CapsMismatch);
        }
        let ch = match channels {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            _ => return Err(G2gError::CapsMismatch),
        };
        let mut enc = Encoder::new(SampleRate::Hz48000, ch, Application::Audio)
            .map_err(|_| G2gError::CapsMismatch)?;
        enc.set_bitrate(self.bitrate).map_err(|_| G2gError::CapsMismatch)?;
        self.enc = Some(enc);
        self.channels = *channels;
        self.buf.clear();
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
                    // Anchor the output timeline to the first input PTS.
                    if self.next_pts_ns.is_none() {
                        self.next_pts_ns = Some(frame.timing.pts_ns);
                    }
                    // Append interleaved S16LE samples to the pending buffer.
                    let bytes = slice.as_slice();
                    self.buf.extend(
                        bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])),
                    );
                    let packets = self.drain_frames()?;
                    self.emit(packets, out).await?;
                }
                PipelinePacket::Eos => {
                    // Flush a partial tail (zero-padded); the runner forwards EOS.
                    if let Some(p) = self.flush()? {
                        self.emit(alloc::vec![p], out).await?;
                    }
                }
                PipelinePacket::Flush => {
                    // Drop buffered samples and re-anchor on the next frame.
                    self.buf.clear();
                    self.next_pts_ns = None;
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

impl PadTemplates for OpusEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        let out = Caps::Audio { format: AudioFormat::Opus, channels: 2, sample_rate: OPUS_RATE_HZ };
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_template())),
            PadTemplate::source(CapsSet::one(out)),
        ])
    }
}

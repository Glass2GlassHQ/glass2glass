//! Linux audio decode element using ffmpeg / libavcodec (M422).
//!
//! Consumes compressed audio access units (`Caps::Audio { format: Aac, .. }`,
//! ADTS-framed as MPEG-TS / HLS carry it) and emits interleaved little-endian
//! `PcmS16Le`. The audio sibling of [`FfmpegVideoDec`](crate::ffmpegdec): it
//! wraps a libavcodec decoder, sends each access unit, and drains decoded
//! frames, converting libavcodec's native sample layout (AAC decodes to planar
//! float, `FLTP`) to interleaved S16. Linux + the `ffmpeg` feature; libavcodec
//! must include the AAC decoder (it does in every standard build).
//!
//! Negotiation: the channel count and sample rate are only known once a frame
//! decodes, so the element advertises a `PcmS16Le { ANY_CHANNELS, ANY_SAMPLE_RATE }`
//! output at negotiation (both wildcards; `fixate` pins channels to a stereo
//! placeholder for the edge) and emits a `CapsChanged` with the real channels /
//! rate before the first decoded `DataFrame`. A downstream `audioconvert`
//! retargets the real channel layout (mono fan-out, multichannel downmix) and
//! `audioresample` (which tolerates the `ANY_SAMPLE_RATE` placeholder and learns
//! the real rate from that `CapsChanged`) retargets to the sink's fixed rate.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use ffmpeg_next as ffmpeg;
use ffmpeg::codec::{self, Id};
use ffmpeg::format::sample::{Sample, Type};
use ffmpeg::frame::Audio as FfAudio;
use ffmpeg::packet::Packet;
use ffmpeg::Error as FfError;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, HardwareError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, ANY_CHANNELS,
    ANY_SAMPLE_RATE,
};

/// One decoded audio frame: interleaved `PcmS16Le` bytes plus its real layout.
struct DecodedAudio {
    pcm: Vec<u8>,
    channels: u8,
    rate: u32,
}

pub struct FfmpegAudioDec {
    decoder: Option<ffmpeg::decoder::Audio>,
    configured: bool,
    emitted: u64,
    /// Last emitted output caps, to suppress re-emitting an unchanged `CapsChanged`.
    last_out: Option<Caps>,
}

// SAFETY: `ffmpeg::decoder::Audio` wraps a raw `*mut AVCodecContext` and is
// `!Send` by default. The runner drives the element through `&mut self` only,
// never concurrently, so the context is owned and moved, never aliased; the
// same contract `FfmpegH264Dec` documents.
unsafe impl Send for FfmpegAudioDec {}

impl core::fmt::Debug for FfmpegAudioDec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FfmpegAudioDec")
            .field("configured", &self.configured)
            .field("emitted", &self.emitted)
            .finish()
    }
}

impl Default for FfmpegAudioDec {
    fn default() -> Self {
        Self::new()
    }
}

impl FfmpegAudioDec {
    pub fn new() -> Self {
        Self { decoder: None, configured: false, emitted: 0, last_out: None }
    }

    /// Count of decoded `DataFrame`s pushed downstream. Useful in tests.
    pub fn decoded_count(&self) -> u64 {
        self.emitted
    }

    /// Drain whatever the decoder is ready to release into `out_frames`,
    /// converting each to interleaved `PcmS16Le`. libavcodec may buffer, so a
    /// send commonly yields zero frames.
    fn drain(&mut self, out_frames: &mut Vec<DecodedAudio>) -> Result<(), G2gError> {
        let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
        loop {
            let mut frame = FfAudio::empty();
            match decoder.receive_frame(&mut frame) {
                Ok(()) => out_frames.push(to_s16_interleaved(&frame)?),
                Err(FfError::Other { errno }) if errno == ffmpeg::error::EAGAIN => return Ok(()),
                Err(FfError::Eof) => return Ok(()),
                Err(_) => return Err(G2gError::Hardware(HardwareError::Other)),
            }
        }
    }
}

impl AsyncElement for FfmpegAudioDec {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio { format: AudioFormat::Aac, .. } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Native `DerivedOutput`: AAC in -> interleaved `PcmS16Le` out at an
    /// `ANY_CHANNELS`, any-rate placeholder. The output channel count must not
    /// depend on the input: an input-coupled count reads as a passthrough field,
    /// and the solver then couples the output's concrete channels back onto the
    /// input. With the channels wildcard that back-coupling is no longer fatal
    /// (`Aac{N} ∩ Aac{0} = Aac{N}`), but a wildcard output is still cleaner: the
    /// decoder genuinely does not know the layout until it parses a frame, so it
    /// advertises `ANY_CHANNELS` (like the video decoder's `Dim::Any`) and the real
    /// channel count / rate arrive via the `CapsChanged` a decoded frame emits.
    /// `fixate` collapses the placeholder to stereo for the negotiated edge, which
    /// the downstream `audioconvert` retargets (mono fan-out, multichannel downmix).
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio { format: AudioFormat::Aac, .. } => CapsSet::one(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: ANY_CHANNELS,
                // A concrete default rate (a raw-PCM `ANY_SAMPLE_RATE` is
                // deliberately unfixable, M187); the real rate arrives via the
                // CapsChanged a decoded frame emits, which the downstream
                // audioresample retargets back to this rate.
                sample_rate: 48_000,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::Audio { format: AudioFormat::Aac, .. } = absolute_caps else {
            return Err(G2gError::CapsMismatch);
        };
        ffmpeg::init().map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let codec = codec::decoder::find(Id::AAC).ok_or(G2gError::Hardware(HardwareError::Other))?;
        let decoder = codec::decoder::new()
            .open_as(codec)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?
            .audio()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.decoder = Some(decoder);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ffmpeg audio decoder",
            "Codec/Decoder/Audio",
            "Decodes AAC (libavcodec) to interleaved PcmS16Le",
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
            let mut decoded = Vec::new();
            let mut timing = FrameTiming::default();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    timing = frame.timing;
                    // libavcodec's AAC decoder wants one access unit per packet,
                    // but a demuxed buffer (one MPEG-TS PES) carries several ADTS
                    // frames back to back; split on the ADTS frame_length and send
                    // each, the audio analog of access-unit framing for video.
                    for au in adts_frames(slice.as_slice()) {
                        let pkt = Packet::copy(au);
                        let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
                        decoder
                            .send_packet(&pkt)
                            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                        self.drain(&mut decoded)?;
                    }
                }
                PipelinePacket::CapsChanged(c) => {
                    // An upstream AAC input caps refine is consumed (the decoder
                    // re-derives its output from decoded frames); the runner's
                    // pre-fixed forward output PcmS16Le caps are passed on.
                    match &c {
                        Caps::Audio { format: AudioFormat::Aac, .. } => {}
                        Caps::Audio { format: AudioFormat::PcmS16Le, .. } => {
                            out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                            self.last_out = Some(c);
                        }
                        _ => return Err(G2gError::CapsMismatch),
                    }
                    return Ok(());
                }
                PipelinePacket::Flush => {
                    if let Some(d) = self.decoder.as_mut() {
                        d.flush();
                    }
                    self.last_out = None;
                    out.push(PipelinePacket::Flush).await?;
                    return Ok(());
                }
                PipelinePacket::Eos => {
                    if let Some(d) = self.decoder.as_mut() {
                        d.send_eof().map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    }
                    self.drain(&mut decoded)?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                    return Ok(());
                }
                other => {
                    out.push(other).await?;
                    return Ok(());
                }
            }

            for d in decoded {
                let new_caps = Caps::Audio {
                    format: AudioFormat::PcmS16Le,
                    channels: d.channels,
                    sample_rate: d.rate,
                };
                if self.last_out.as_ref() != Some(&new_caps) {
                    out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                    self.last_out = Some(new_caps);
                }
                let out_frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(d.pcm.into_boxed_slice())),
                    timing,
                    sequence: self.emitted,
                    meta: Default::default(),
                };
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(out_frame)).await?;
            }
            Ok(())
        })
    }
}

impl PadTemplates for FfmpegAudioDec {
    fn pad_templates() -> Vec<PadTemplate> {
        let aac = Caps::Audio { format: AudioFormat::Aac, channels: ANY_CHANNELS, sample_rate: 0 };
        let pcm = Caps::Audio { format: AudioFormat::PcmS16Le, channels: ANY_CHANNELS, sample_rate: ANY_SAMPLE_RATE };
        Vec::from([PadTemplate::sink(CapsSet::one(aac)), PadTemplate::source(CapsSet::one(pcm))])
    }
}

/// Split a buffer of back-to-back ADTS AAC frames into individual frames, so each
/// reaches the decoder as one access unit. Each ADTS frame begins with a 12-bit
/// syncword (`0xFFF`) and carries a 13-bit `aac_frame_length` (total header +
/// payload) at bits spanning bytes 3..6. Walking stops at the first byte that is
/// not a valid sync or a length that overruns the buffer, so a truncated tail is
/// dropped rather than mis-fed. A buffer that is already a single frame yields it
/// once.
fn adts_frames(buf: &[u8]) -> Vec<&[u8]> {
    let mut frames = Vec::new();
    let mut pos = 0;
    while pos + 7 <= buf.len() {
        // Syncword: byte0 == 0xFF and the top 4 bits of byte1 == 0xF.
        if buf[pos] != 0xFF || (buf[pos + 1] & 0xF0) != 0xF0 {
            break;
        }
        let len = (((buf[pos + 3] & 0x03) as usize) << 11)
            | ((buf[pos + 4] as usize) << 3)
            | ((buf[pos + 5] >> 5) as usize);
        if len < 7 || pos + len > buf.len() {
            break;
        }
        frames.push(&buf[pos..pos + len]);
        pos += len;
    }
    frames
}

/// Convert one decoded libavcodec audio frame to interleaved little-endian
/// `PcmS16Le`. Handles the layouts libavcodec emits for the codecs we decode:
/// planar / packed 32-bit float (AAC's native `FLTP`) and planar / packed S16.
/// Float samples are clamped to [-1, 1] and scaled; an unsupported layout is a
/// loud `CapsMismatch` rather than silent noise.
fn to_s16_interleaved(frame: &FfAudio) -> Result<DecodedAudio, G2gError> {
    let channels = frame.channels() as usize;
    let samples = frame.samples();
    if channels == 0 {
        return Err(G2gError::CapsMismatch);
    }
    let mut out = alloc::vec![0i16; samples * channels];
    let f32_to_i16 = |v: f32| (v.clamp(-1.0, 1.0) * 32767.0) as i16;
    match frame.format() {
        Sample::F32(Type::Planar) => {
            for (c, slot) in (0..channels).zip(0..) {
                let plane = frame.plane::<f32>(c);
                for i in 0..samples {
                    out[i * channels + slot] = f32_to_i16(plane[i]);
                }
            }
        }
        Sample::F32(Type::Packed) => {
            let plane = frame.plane::<f32>(0);
            for (i, s) in plane.iter().take(samples * channels).enumerate() {
                out[i] = f32_to_i16(*s);
            }
        }
        Sample::I16(Type::Planar) => {
            for (c, slot) in (0..channels).zip(0..) {
                let plane = frame.plane::<i16>(c);
                for i in 0..samples {
                    out[i * channels + slot] = plane[i];
                }
            }
        }
        Sample::I16(Type::Packed) => {
            let plane = frame.plane::<i16>(0);
            out.copy_from_slice(&plane[..samples * channels]);
        }
        _ => return Err(G2gError::CapsMismatch),
    }
    let mut pcm = Vec::with_capacity(out.len() * 2);
    for s in out {
        pcm.extend_from_slice(&s.to_le_bytes());
    }
    Ok(DecodedAudio { pcm, channels: channels as u8, rate: frame.rate() })
}

#[cfg(test)]
mod tests {
    use super::adts_frames;
    use alloc::vec;
    use alloc::vec::Vec;

    /// A minimal `len`-byte ADTS frame: a 7-byte header (sync + `aac_frame_length`
    /// = `len`) zero-padded to `len`.
    fn adts(len: usize) -> Vec<u8> {
        let mut f = vec![0u8; len];
        f[0] = 0xFF;
        f[1] = 0xF1; // syncword low nibble + MPEG-4, layer 0, no CRC
        f[3] = ((len >> 11) & 0x03) as u8;
        f[4] = ((len >> 3) & 0xFF) as u8;
        f[5] = (((len & 0x07) << 5) as u8) | 0x1F;
        f
    }

    #[test]
    fn splits_concatenated_adts_frames() {
        let mut buf = adts(20);
        buf.extend_from_slice(&adts(13));
        let frames = adts_frames(&buf);
        assert_eq!(frames.len(), 2, "two back-to-back ADTS frames split apart");
        assert_eq!(frames[0].len(), 20);
        assert_eq!(frames[1].len(), 13);
    }

    #[test]
    fn drops_a_truncated_tail() {
        let mut buf = adts(20);
        buf.extend_from_slice(&adts(30)[..10]); // a second frame claiming 30 bytes, only 10 present
        let frames = adts_frames(&buf);
        assert_eq!(frames.len(), 1, "the complete frame is kept, the truncated tail dropped");
        assert_eq!(frames[0].len(), 20);
    }

    #[test]
    fn bails_without_a_syncword() {
        assert!(adts_frames(&[0, 1, 2, 3, 4, 5, 6, 7]).is_empty());
    }
}

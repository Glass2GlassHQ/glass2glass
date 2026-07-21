//! Linux audio decode element using ffmpeg / libavcodec (M422).
//!
//! Consumes compressed audio access units (`Caps::Audio { format: Aac | Mp2, .. }`;
//! AAC ADTS-framed as MPEG-TS / HLS carry it, MPEG audio as raw self-syncing
//! frames) and emits interleaved little-endian `PcmS16Le`. The audio sibling of
//! [`FfmpegVideoDec`](crate::ffmpegdec): it wraps a libavcodec decoder, sends
//! each access unit, and drains decoded frames, converting libavcodec's native
//! sample layout (AAC decodes to planar float, `FLTP`) to interleaved S16.
//! Linux + the `ffmpeg` feature; libavcodec must include the AAC and MP2
//! decoders (it does in every standard build).
//!
//! Negotiation: the channel count and sample rate are only known once a frame
//! decodes, so the element advertises a `PcmS16Le { ANY_CHANNELS, .. }` output at
//! negotiation, channels a wildcard (`fixate` pins it to a stereo placeholder for
//! the edge) and the rate two alternatives, a concrete `48_000` default plus an
//! `ANY_SAMPLE_RATE` wildcard (M754), so a downstream `rate=44100` / `rate=48000`
//! pin negotiates directly while an any-rate sink still fixates to the default. It
//! then emits a `CapsChanged` with the real channels / rate before the first
//! decoded `DataFrame`. A downstream `audioconvert` retargets the real channel
//! layout (mono fan-out, multichannel downmix) and `audioresample` (which tolerates
//! the placeholder rate and learns the real rate from that `CapsChanged`) retargets
//! to the sink's fixed rate.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use ffmpeg::codec::{self, Id};
use ffmpeg::format::sample::{Sample, Type};
use ffmpeg::frame::Audio as FfAudio;
use ffmpeg::packet::Packet;
use ffmpeg::Error as FfError;
use ffmpeg_next as ffmpeg;

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
    /// The compressed input format this instance decodes, set at configure.
    /// Drives the libavcodec codec choice and the per-format access-unit split.
    format: AudioFormat,
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
        Self {
            decoder: None,
            format: AudioFormat::Aac,
            configured: false,
            emitted: 0,
            last_out: None,
        }
    }

    /// Whether `format` is a compressed input this element decodes.
    fn decodes(format: AudioFormat) -> bool {
        matches!(format, AudioFormat::Aac | AudioFormat::Mp2)
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
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio { format, .. } if Self::decodes(*format) => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Native `DerivedOutput`: AAC in -> interleaved `PcmS16Le` out at an
    /// `ANY_CHANNELS` placeholder, over two rate alternatives. The output channel
    /// count must not depend on the input: an input-coupled count reads as a
    /// passthrough field, and the solver then couples the output's concrete
    /// channels back onto the input. With the channels wildcard that back-coupling
    /// is no longer fatal (`Aac{N} ∩ Aac{0} = Aac{N}`), but a wildcard output is
    /// still cleaner: the decoder genuinely does not know the layout until it parses
    /// a frame, so it advertises `ANY_CHANNELS` (like the video decoder's `Dim::Any`)
    /// and the real channel count / rate arrive via the `CapsChanged` a decoded frame
    /// emits. `fixate` collapses the placeholder to stereo for the negotiated edge,
    /// which the downstream `audioconvert` retargets (mono fan-out, downmix).
    ///
    /// The rate is two alternatives (M754): a concrete default first (`48_000`, the
    /// fixate fallback when nothing downstream pins the rate, e.g. an any-rate sink)
    /// plus an `ANY_SAMPLE_RATE` wildcard second, so a downstream concrete pin (a
    /// `rate=44100` or `rate=48000` capsfilter) intersects to that rate and no
    /// resampler is forced. A lone `ANY_SAMPLE_RATE` is deliberately unfixable
    /// (M187), hence the concrete first alternative; this mirrors audioresample's
    /// `[passthrough, ANY_SAMPLE_RATE]` set. The real rate still arrives via the
    /// CapsChanged a decoded frame emits, so a 44.1 kHz decode reaching a 48 kHz pin
    /// with no `audioresample` still fails loud at runtime (M749).
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio { format, .. } if Self::decodes(*format) => {
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
        let Caps::Audio { format, .. } = absolute_caps else {
            return Err(G2gError::CapsMismatch);
        };
        if !Self::decodes(*format) {
            return Err(G2gError::CapsMismatch);
        }
        self.format = *format;
        ffmpeg::init().map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        // MP2 = libavcodec's layer I/II decoder, matching what `-c:a mp2` writes.
        let id = match self.format {
            AudioFormat::Mp2 => Id::MP2,
            _ => Id::AAC,
        };
        let codec = codec::decoder::find(id).ok_or(G2gError::Hardware(HardwareError::Other))?;
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
            "Decodes AAC / MPEG audio (libavcodec) to interleaved PcmS16Le",
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
                    // libavcodec wants one access unit per packet, but a demuxed
                    // buffer (one MPEG-TS PES) carries several frames back to
                    // back; split on the codec's frame length (ADTS header for
                    // AAC, MPEG audio header for mp2) and send each, the audio
                    // analog of access-unit framing for video.
                    let aus = match self.format {
                        AudioFormat::Mp2 => mpa_frames(slice.as_slice()),
                        _ => adts_frames(slice.as_slice()),
                    };
                    for au in aus {
                        let pkt = Packet::copy(au);
                        let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
                        decoder
                            .send_packet(&pkt)
                            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                        self.drain(&mut decoded)?;
                    }
                }
                PipelinePacket::CapsChanged(c) => {
                    // An upstream compressed-input caps refine is consumed (the
                    // decoder re-derives its output from decoded frames); the
                    // runner's pre-fixed forward output PcmS16Le caps are passed on.
                    match &c {
                        Caps::Audio { format, .. } if Self::decodes(*format) => {}
                        Caps::Audio {
                            format: AudioFormat::PcmS16Le,
                            ..
                        } => {
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
                        d.send_eof()
                            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
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
                    out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                        .await?;
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
        let compressed = |format| Caps::Audio {
            format,
            channels: ANY_CHANNELS,
            sample_rate: 0,
        };
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: ANY_CHANNELS,
            sample_rate: ANY_SAMPLE_RATE,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(Vec::from([
                compressed(AudioFormat::Aac),
                compressed(AudioFormat::Mp2),
            ]))),
            PadTemplate::source(CapsSet::one(pcm)),
        ])
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

/// Bitrates (kbit/s) for MPEG-1 Layer II, indexed by the header's 4-bit field.
/// Index 0 ("free format") and 15 are invalid here and fail the parse.
const MP2_BITRATES_KBPS: [u32; 16] = [
    0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0,
];
/// Sample rates for MPEG-1 (index 3 is reserved). MPEG-2 halves these.
const MPA_RATES_HZ: [u32; 4] = [44_100, 48_000, 32_000, 0];

/// Split a buffer of back-to-back MPEG audio (Layer II, `mp2`) frames into
/// individual frames, the MPEG-audio sibling of [`adts_frames`]. Each frame's
/// 4-byte header carries an 11-bit sync (`0xFFE`), version, layer, bitrate and
/// sample-rate indices, and a padding bit; Layer II frame length =
/// `144 * bitrate / rate + padding` (MPEG-2's low-rate extension halves the
/// rate, same formula). Walking stops at the first invalid header or a length
/// overrunning the buffer, so a truncated tail is dropped rather than mis-fed.
fn mpa_frames(buf: &[u8]) -> Vec<&[u8]> {
    let mut frames = Vec::new();
    let mut pos = 0;
    while pos + 4 <= buf.len() {
        let h = &buf[pos..pos + 4];
        // Sync: 11 set bits. Version: bits 4..3 of byte1 (3 = MPEG-1, 2 = MPEG-2,
        // 0 = MPEG-2.5, 1 reserved). Layer: bits 2..1 (2 = Layer II).
        if h[0] != 0xFF || (h[1] & 0xE0) != 0xE0 {
            break;
        }
        let version = (h[1] >> 3) & 0x03;
        let layer = (h[1] >> 1) & 0x03;
        if version == 1 || layer != 2 {
            break; // reserved version, or not Layer II
        }
        let bitrate = MP2_BITRATES_KBPS[((h[2] >> 4) & 0x0F) as usize].saturating_mul(1_000);
        let mut rate = MPA_RATES_HZ[((h[2] >> 2) & 0x03) as usize];
        if version != 3 {
            rate /= 2; // MPEG-2 / MPEG-2.5 low-sample-rate extension
        }
        if bitrate == 0 || rate == 0 {
            break; // free-format / reserved: cannot compute a frame length
        }
        let padding = ((h[2] >> 1) & 1) as usize;
        let len = (144 * bitrate / rate) as usize + padding;
        if len < 4 || pos + len > buf.len() {
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
    Ok(DecodedAudio {
        pcm,
        channels: channels as u8,
        rate: frame.rate(),
    })
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
        assert_eq!(
            frames.len(),
            1,
            "the complete frame is kept, the truncated tail dropped"
        );
        assert_eq!(frames[0].len(), 20);
    }

    #[test]
    fn bails_without_a_syncword() {
        assert!(adts_frames(&[0, 1, 2, 3, 4, 5, 6, 7]).is_empty());
    }

    /// A minimal MPEG-1 Layer II frame header: 384 kbit/s at 48 kHz, no padding
    /// (frame length = 144 * 384000 / 48000 = 1152 bytes), zero-padded.
    fn mpa() -> Vec<u8> {
        let mut f = vec![0u8; 1152];
        f[0] = 0xFF;
        f[1] = 0xFD; // sync + MPEG-1 + Layer II + no CRC
        f[2] = 0xE4; // bitrate index 14 (384k) + rate index 1 (48k) + no padding
        f
    }

    #[test]
    fn splits_concatenated_mpa_frames() {
        let mut buf = mpa();
        buf.extend_from_slice(&mpa());
        let frames = super::mpa_frames(&buf);
        assert_eq!(frames.len(), 2, "two back-to-back mp2 frames split apart");
        assert!(frames.iter().all(|f| f.len() == 1152));
    }

    #[test]
    fn mpa_drops_truncated_tail_and_bad_headers() {
        let mut buf = mpa();
        buf.extend_from_slice(&mpa()[..40]); // truncated second frame
        assert_eq!(super::mpa_frames(&buf).len(), 1);
        // free-format bitrate (index 0) has no computable length: bail, no panic.
        let mut free = mpa();
        free[2] = 0x04;
        assert!(super::mpa_frames(&free).is_empty());
        assert!(super::mpa_frames(&[0xFF, 0xFD]).is_empty());
    }
}

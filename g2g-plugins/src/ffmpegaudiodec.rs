//! Linux audio decode element using ffmpeg / libavcodec (M422).
//!
//! Consumes compressed audio access units (`Caps::Audio { format: Aac | Mp2 |
//! Ac3 | Flac, .. }`; AAC ADTS-framed as MPEG-TS / HLS carry it, MPEG audio and
//! AC-3 as self-syncing frames, FLAC container-framed one frame per packet) and
//! emits interleaved little-endian `PcmS16Le`. The audio sibling of
//! [`FfmpegVideoDec`](crate::ffmpegdec): it wraps a libavcodec decoder, sends
//! each access unit, and drains decoded frames, converting libavcodec's native
//! sample layout (AAC / AC-3 decode to planar float `FLTP`, FLAC to S16 / S32)
//! to interleaved S16. Linux + the `ffmpeg` feature; libavcodec must include the
//! AAC / MP2 / AC-3 / FLAC decoders (it does in every standard build).
//!
//! FLAC is the one format that needs setup data: the decoder takes the stream's
//! STREAMINFO as extradata. The containers hand it over as the native `fLaC`
//! header (mkv `CodecPrivate`, the first Ogg packet), which the demuxer forwards
//! in-band as a leading `fLaC`-prefixed frame; this element takes it as extradata
//! and opens the decoder lazily (the AAC / MP2 / AC-3 decoders open eagerly at
//! configure, their frames being self-describing).
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
        matches!(
            format,
            AudioFormat::Aac | AudioFormat::Mp2 | AudioFormat::Ac3 | AudioFormat::Flac
        )
    }

    /// The libavcodec decoder Id for a decoded format.
    fn codec_id(format: AudioFormat) -> Id {
        match format {
            // MP2 = libavcodec's layer I/II decoder, matching what `-c:a mp2` writes.
            AudioFormat::Mp2 => Id::MP2,
            AudioFormat::Ac3 => Id::AC3,
            AudioFormat::Flac => Id::FLAC,
            _ => Id::AAC,
        }
    }

    /// Open the libavcodec decoder for `self.format`, optionally seeding it with
    /// `extradata` (FLAC's STREAMINFO). libavcodec owns the extradata buffer (frees
    /// it on close) and requires the `AV_INPUT_BUFFER_PADDING_SIZE` trailing zero
    /// bytes, the same setup [`FfmpegH264Dec`](crate::ffmpegdec) uses for parameter
    /// sets. Idempotent: a second call while open is a no-op.
    fn open_decoder(&mut self, extradata: Option<&[u8]>) -> Result<(), G2gError> {
        if self.decoder.is_some() {
            return Ok(());
        }
        ffmpeg::init().map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let codec = codec::decoder::find(Self::codec_id(self.format))
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let mut decoder_ctx = codec::decoder::new();
        if let Some(extradata) = extradata {
            // SAFETY: `decoder_ctx` is freshly allocated and not yet opened.
            // `av_mallocz` returns a zeroed buffer (so the required trailing
            // padding is already zero); we copy `extradata` into it and hand
            // ownership to the context via its raw `extradata`/`extradata_size`
            // fields, the canonical way to set decoder extradata.
            unsafe {
                let size = extradata.len();
                let total = size + ffmpeg::ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize;
                let buf = ffmpeg::ffi::av_mallocz(total) as *mut u8;
                if buf.is_null() {
                    return Err(G2gError::Hardware(HardwareError::Other));
                }
                core::ptr::copy_nonoverlapping(extradata.as_ptr(), buf, size);
                let raw = decoder_ctx.as_mut_ptr();
                (*raw).extradata = buf;
                (*raw).extradata_size = size as i32;
            }
        }
        let decoder = decoder_ctx
            .open_as(codec)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?
            .audio()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.decoder = Some(decoder);
        Ok(())
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
        // FLAC opens lazily on its `fLaC` STREAMINFO header (needed as extradata);
        // the rest open eagerly, their frames being self-describing.
        if self.format != AudioFormat::Flac {
            self.open_decoder(None)?;
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ffmpeg audio decoder",
            "Codec/Decoder/Audio",
            "Decodes AAC / MPEG audio / AC-3 / FLAC (libavcodec) to interleaved PcmS16Le",
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
                    let buf = slice.as_slice();
                    // FLAC arrives container-framed (one frame per packet), led by
                    // the native `fLaC` STREAMINFO header the demuxer forwards
                    // in-band: take it as extradata and open the decoder, emitting
                    // nothing (it is codec config, not audio). Later frames go to the
                    // decoder whole, since FLAC frame boundaries are not cheaply
                    // self-syncing so g2g relies on the container framing.
                    if self.format == AudioFormat::Flac {
                        if buf.starts_with(b"fLaC") {
                            self.open_decoder(Some(buf))?;
                            return Ok(());
                        }
                        // A raw .flac stream with no leading header: open without
                        // extradata (frame headers are self-describing for standard
                        // rates), then decode the frame.
                        self.open_decoder(None)?;
                        let pkt = Packet::copy(buf);
                        let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
                        decoder
                            .send_packet(&pkt)
                            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                        self.drain(&mut decoded)?;
                    } else {
                        // libavcodec wants one access unit per packet, but a demuxed
                        // buffer (one MPEG-TS PES) carries several frames back to
                        // back; split on the codec's frame length (ADTS header for
                        // AAC, MPEG audio header for mp2, `0x0B77` sync for AC-3) and
                        // send each, the audio analog of access-unit framing for video.
                        let aus = match self.format {
                            AudioFormat::Mp2 => mpa_frames(buf),
                            AudioFormat::Ac3 => ac3_frames(buf),
                            _ => adts_frames(buf),
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
                compressed(AudioFormat::Ac3),
                compressed(AudioFormat::Flac),
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

/// AC-3 syncframe size in 16-bit words, indexed `[frmsizecod][fscod]`, where
/// `fscod` selects 48 / 44.1 / 32 kHz (ATSC A/52 Table 5.18). The 44.1 kHz column
/// carries the +1-word padding some rates need. `frmsizecod` above 37 is invalid.
const AC3_FRAME_SIZE_WORDS: [[u16; 3]; 38] = [
    [64, 69, 96],
    [64, 70, 96],
    [80, 87, 120],
    [80, 88, 120],
    [96, 104, 144],
    [96, 105, 144],
    [112, 121, 168],
    [112, 122, 168],
    [128, 139, 192],
    [128, 140, 192],
    [160, 174, 240],
    [160, 175, 240],
    [192, 208, 288],
    [192, 209, 288],
    [224, 243, 336],
    [224, 244, 336],
    [256, 278, 384],
    [256, 279, 384],
    [320, 348, 480],
    [320, 349, 480],
    [384, 417, 576],
    [384, 418, 576],
    [448, 487, 672],
    [448, 488, 672],
    [512, 557, 768],
    [512, 558, 768],
    [640, 696, 960],
    [640, 697, 960],
    [768, 835, 1152],
    [768, 836, 1152],
    [896, 975, 1344],
    [896, 976, 1344],
    [1024, 1114, 1536],
    [1024, 1115, 1536],
    [1152, 1253, 1728],
    [1152, 1254, 1728],
    [1280, 1393, 1920],
    [1280, 1394, 1920],
];

/// Split a buffer of back-to-back AC-3 syncframes into individual frames, the AC-3
/// sibling of [`adts_frames`]. Each frame starts with the `0x0B77` syncword; byte 4
/// holds `fscod` (top 2 bits) and `frmsizecod` (low 6 bits), whose pair gives the
/// frame length in 16-bit words via [`AC3_FRAME_SIZE_WORDS`] (bytes = words * 2).
/// Walking stops at the first bad sync, a reserved `fscod` (3) / `frmsizecod`
/// (>= 38), or a length overrunning the buffer, so a truncated tail is dropped
/// rather than mis-fed.
fn ac3_frames(buf: &[u8]) -> Vec<&[u8]> {
    let mut frames = Vec::new();
    let mut pos = 0;
    while pos + 5 <= buf.len() {
        if buf[pos] != 0x0B || buf[pos + 1] != 0x77 {
            break;
        }
        let fscod = (buf[pos + 4] >> 6) as usize;
        let frmsizecod = (buf[pos + 4] & 0x3F) as usize;
        if fscod >= 3 || frmsizecod >= AC3_FRAME_SIZE_WORDS.len() {
            break; // reserved sample rate or frame-size code
        }
        let len = (AC3_FRAME_SIZE_WORDS[frmsizecod][fscod] as usize).saturating_mul(2);
        if len < 5 || pos + len > buf.len() {
            break;
        }
        frames.push(&buf[pos..pos + len]);
        pos += len;
    }
    frames
}

/// The raw interleaved bytes of a packed audio frame, sliced to exactly
/// `samples * channels` elements of `T`. `frame.plane::<T>(0)` cannot be used
/// here: ffmpeg-next sizes a plane as `samples` elements regardless of the
/// channel count, so a packed multi-channel plane comes back one channel long
/// (indexing past it panicked on stereo FLAC). `data(0)` is the real buffer;
/// its tail past the sample data is allocator padding, and a buffer shorter
/// than the frame claims is a loud `CapsMismatch`, not a truncated read.
fn packed_bytes<T>(frame: &FfAudio, samples: usize, channels: usize) -> Result<&[u8], G2gError> {
    let need = samples
        .checked_mul(channels)
        .and_then(|n| n.checked_mul(core::mem::size_of::<T>()))
        .ok_or(G2gError::CapsMismatch)?;
    frame.data(0).get(..need).ok_or(G2gError::CapsMismatch)
}

/// Convert one decoded libavcodec audio frame to interleaved little-endian
/// `PcmS16Le`. Handles the layouts libavcodec emits for the codecs we decode:
/// planar / packed 32-bit float (AAC / AC-3's native `FLTP`), planar / packed S16
/// (FLAC <= 16-bit), and planar / packed S32 (FLAC 24-bit, high bits kept). Float
/// samples are clamped to [-1, 1] and scaled; an unsupported layout is a loud
/// `CapsMismatch` rather than silent noise.
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
            let data = packed_bytes::<f32>(frame, samples, channels)?;
            for (i, b) in data.chunks_exact(4).enumerate() {
                out[i] = f32_to_i16(f32::from_ne_bytes([b[0], b[1], b[2], b[3]]));
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
            let data = packed_bytes::<i16>(frame, samples, channels)?;
            for (i, b) in data.chunks_exact(2).enumerate() {
                out[i] = i16::from_ne_bytes([b[0], b[1]]);
            }
        }
        Sample::I32(Type::Planar) => {
            for (c, slot) in (0..channels).zip(0..) {
                let plane = frame.plane::<i32>(c);
                for i in 0..samples {
                    out[i * channels + slot] = (plane[i] >> 16) as i16;
                }
            }
        }
        Sample::I32(Type::Packed) => {
            let data = packed_bytes::<i32>(frame, samples, channels)?;
            for (i, b) in data.chunks_exact(4).enumerate() {
                out[i] = (i32::from_ne_bytes([b[0], b[1], b[2], b[3]]) >> 16) as i16;
            }
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

    /// A minimal AC-3 frame: `0x0B77` sync, `fscod`/`frmsizecod` in byte 4, padded
    /// to its computed length. `frmsizecod` 20 at 48 kHz = 384 words = 768 bytes.
    fn ac3(fscod: u8, frmsizecod: u8) -> Vec<u8> {
        let words = super::AC3_FRAME_SIZE_WORDS[frmsizecod as usize][fscod as usize] as usize;
        let mut f = vec![0u8; words * 2];
        f[0] = 0x0B;
        f[1] = 0x77;
        f[4] = (fscod << 6) | (frmsizecod & 0x3F);
        f
    }

    #[test]
    fn splits_concatenated_ac3_frames() {
        let mut buf = ac3(0, 20); // 48 kHz, 768 bytes
        buf.extend_from_slice(&ac3(0, 20));
        let frames = super::ac3_frames(&buf);
        assert_eq!(frames.len(), 2, "two back-to-back AC-3 frames split apart");
        assert!(frames.iter().all(|f| f.len() == 768));
    }

    #[test]
    fn ac3_drops_truncated_tail_and_rejects_bad_headers() {
        let mut buf = ac3(0, 20);
        buf.extend_from_slice(&ac3(0, 20)[..100]); // truncated second frame
        assert_eq!(super::ac3_frames(&buf).len(), 1);
        // reserved fscod (3) has no frame length: bail, no panic.
        let mut bad_rate = ac3(0, 20);
        bad_rate[4] = (3 << 6) | 20;
        assert!(super::ac3_frames(&bad_rate).is_empty());
        // no syncword, and a too-short buffer: both empty, no panic.
        assert!(super::ac3_frames(&[0, 1, 2, 3, 4]).is_empty());
        assert!(super::ac3_frames(&[0x0B, 0x77]).is_empty());
    }
}

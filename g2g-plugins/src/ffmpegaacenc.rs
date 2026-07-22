//! libavcodec AAC-LC audio encoder (M292): interleaved PCM in
//! (`Caps::Audio{PcmS16Le|PcmF32Le}`), ADTS-framed AAC out
//! (`Caps::Audio{Aac}`). The audio companion of [`crate::ffmpegenc::FfmpegH264Enc`],
//! so the Linux path finally has an AAC encoder for `mp4mux` / `mpegtsmux` /
//! `matroskamux` / `flvmux` (the existing `MfAacEncode` is Windows-only):
//!
//! ```text
//! audiotestsrc ! audioconvert ! avenc_aac ! mpegtsmux ! filesink location=a.ts
//! ```
//!
//! ADTS framing (a 7-byte header per access unit) is the elementary-stream
//! convention the rest of the tree expects (`aacparse` recovers channel / rate
//! from it; muxers that need the AudioSpecificConfig synthesise it from the ADTS
//! header). The ffmpeg `aac` encoder wants planar float (`FLTP`) at a fixed 1024
//! samples per frame, so input PCM is converted to FLTP and buffered into
//! frame-sized chunks; a partial tail is flushed at EOS.
//!
//! Threading: like `FfmpegH264Enc`, the raw `AVCodecContext` is single-owner and
//! the element asserts `Send` under that contract.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use ffmpeg::codec::encoder::audio::Encoder as AudioEncoder;
use ffmpeg::format::sample::Type as SampleType;
use ffmpeg::format::Sample as SampleFmt;
use ffmpeg::frame::Audio as FfAudio;
use ffmpeg::packet::Packet;
use ffmpeg::ChannelLayout;
use ffmpeg::Error as FfError;
use ffmpeg_next as ffmpeg;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    FrameTiming, G2gError, HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

/// Default constant bitrate: 128 kbps, the usual stereo-AAC streaming target.
const DEFAULT_BITRATE_BPS: usize = 128_000;

/// ADTS / ASC sampling-frequency-index table (ISO/IEC 14496-3). The encoder only
/// opens at a rate in this table (libavcodec rejects others).
const SAMPLE_RATES: [u32; 13] = [
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
];

fn sample_rate_index(rate: u32) -> Option<u8> {
    SAMPLE_RATES
        .iter()
        .position(|&r| r == rate)
        .map(|i| i as u8)
}

pub struct FfmpegAacEnc {
    sample_rate: u32,
    channels: u8,
    /// Input PCM format (`PcmS16Le` or `PcmF32Le`), fixed at negotiation.
    input_format: AudioFormat,
    bitrate_bps: usize,
    encoder: Option<AudioEncoder>,
    /// libavcodec frame size (samples per channel per AAC frame, 1024 for LC).
    frame_size: usize,
    /// Interleaved f32 PCM not yet encoded (carried across input buffers so each
    /// AAC frame is exactly `frame_size` samples).
    pending: Vec<f32>,
    /// Total input samples (per channel) consumed, the next frame's PTS in
    /// sample units (`time_base` = 1/rate).
    samples_in: u64,
    caps_sent: bool,
    configured: bool,
    emitted: u64,
}

// SAFETY: `AudioEncoder` wraps a raw `*mut AVCodecContext`; like `FfmpegH264Enc`
// the element is single-owner (one executor thread drives it) and never shares
// the context, so the `Send` contract holds.
unsafe impl Send for FfmpegAacEnc {}

impl core::fmt::Debug for FfmpegAacEnc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FfmpegAacEnc")
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("bitrate_bps", &self.bitrate_bps)
            .field("open", &self.encoder.is_some())
            .finish()
    }
}

impl Default for FfmpegAacEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl FfmpegAacEnc {
    pub fn new() -> Self {
        Self {
            sample_rate: 0,
            channels: 0,
            input_format: AudioFormat::PcmS16Le,
            bitrate_bps: DEFAULT_BITRATE_BPS,
            encoder: None,
            frame_size: 1024,
            pending: Vec::new(),
            samples_in: 0,
            caps_sent: false,
            configured: false,
            emitted: 0,
        }
    }

    pub fn with_bitrate(mut self, bps: usize) -> Self {
        self.bitrate_bps = bps.max(1);
        self
    }

    /// Count of ADTS access units emitted.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn input_template() -> CapsSet {
        let pcm = |format| Caps::Audio {
            format,
            channels: 0,
            sample_rate: 0,
        };
        CapsSet::from_alternatives(Vec::from([
            pcm(AudioFormat::PcmS16Le),
            pcm(AudioFormat::PcmF32Le),
        ]))
    }

    fn output_caps(&self) -> Caps {
        Caps::Audio {
            format: AudioFormat::Aac,
            channels: self.channels,
            sample_rate: self.sample_rate,
        }
    }

    fn open_encoder(&mut self) -> Result<(), G2gError> {
        let codec =
            ffmpeg::encoder::find_by_name("aac").ok_or(G2gError::Hardware(HardwareError::Other))?;
        // Allocate with the codec so codec-appropriate defaults apply (the M289
        // lesson: a codec-less context trips encoder default-validation).
        let mut audio = ffmpeg::codec::context::Context::new_with_codec(codec)
            .encoder()
            .audio()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        audio.set_rate(self.sample_rate as i32);
        audio.set_channel_layout(ChannelLayout::default(self.channels as i32));
        // The native ffmpeg AAC encoder accepts only planar float.
        audio.set_format(SampleFmt::F32(SampleType::Planar));
        audio.set_bit_rate(self.bitrate_bps);
        audio.set_time_base(ffmpeg::Rational::new(1, self.sample_rate as i32));
        let opened = audio
            .open_as(codec)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        // frame_size is known once opened (0 means "any", e.g. a variable-frame
        // encoder, but AAC-LC is always 1024).
        let fs = opened.frame_size() as usize;
        self.frame_size = if fs > 0 { fs } else { 1024 };
        self.encoder = Some(opened);
        Ok(())
    }

    /// Decode an input PCM buffer into interleaved f32 samples appended to `pending`.
    fn ingest(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        match self.input_format {
            AudioFormat::PcmS16Le => {
                if bytes.len() % 2 != 0 {
                    return Err(G2gError::CapsMismatch);
                }
                self.pending.extend(
                    bytes
                        .chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0),
                );
            }
            AudioFormat::PcmF32Le => {
                if bytes.len() % 4 != 0 {
                    return Err(G2gError::CapsMismatch);
                }
                self.pending.extend(
                    bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                );
            }
            _ => return Err(G2gError::CapsMismatch),
        }
        Ok(())
    }

    /// Encode one frame from `interleaved` (`n_samples` per channel) and drain the
    /// resulting ADTS access units.
    fn encode_frame(
        &mut self,
        interleaved: &[f32],
        n_samples: usize,
    ) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let ch = self.channels as usize;
        let mut frame = FfAudio::new(
            SampleFmt::F32(SampleType::Planar),
            n_samples,
            ChannelLayout::default(self.channels as i32),
        );
        frame.set_rate(self.sample_rate);
        frame.set_pts(Some(self.samples_in as i64));
        // Deinterleave into the per-channel planar float planes.
        for c in 0..ch {
            let plane = frame.plane_mut::<f32>(c);
            for (i, slot) in plane.iter_mut().enumerate().take(n_samples) {
                *slot = interleaved[i * ch + c];
            }
        }
        self.samples_in += n_samples as u64;
        let enc = self.encoder.as_mut().ok_or(G2gError::NotConfigured)?;
        enc.send_frame(&frame)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.drain()
    }

    fn flush(&mut self) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let mut out = Vec::new();
        // Encode the partial tail (libavcodec pads the last AAC frame).
        let ch = self.channels as usize;
        if ch > 0 && self.pending.len() >= ch {
            let tail = core::mem::take(&mut self.pending);
            let n = tail.len() / ch;
            out.extend(self.encode_frame(&tail, n)?);
        }
        if let Some(enc) = self.encoder.as_mut() {
            enc.send_eof()
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        }
        out.extend(self.drain()?);
        Ok(out)
    }

    /// Drain ready packets, ADTS-wrapping each, as `(adts_au, pts_ns)`.
    fn drain(&mut self) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let sr_index = sample_rate_index(self.sample_rate).ok_or(G2gError::CapsMismatch)?;
        let mut out = Vec::new();
        let enc = self.encoder.as_mut().ok_or(G2gError::NotConfigured)?;
        loop {
            let mut packet = Packet::empty();
            match enc.receive_packet(&mut packet) {
                Ok(()) => {
                    let pts_units = packet.pts().unwrap_or(0).max(0) as u128;
                    let pts_ns =
                        (pts_units * 1_000_000_000 / self.sample_rate.max(1) as u128) as u64;
                    if let Some(data) = packet.data() {
                        out.push((adts_wrap(data, sr_index, self.channels), pts_ns));
                    }
                }
                Err(FfError::Other { errno }) if errno == ffmpeg::error::EAGAIN => break,
                Err(FfError::Eof) => break,
                Err(_) => return Err(G2gError::Hardware(HardwareError::Other)),
            }
        }
        Ok(out)
    }

    async fn emit(
        &mut self,
        packets: Vec<(Vec<u8>, u64)>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        if !packets.is_empty() && !self.caps_sent {
            out.push(PipelinePacket::CapsChanged(self.output_caps()))
                .await?;
            self.caps_sent = true;
        }
        for (au, pts_ns) in packets {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
                FrameTiming {
                    pts_ns,
                    dts_ns: pts_ns,
                    ..FrameTiming::default()
                },
                self.emitted,
            );
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

/// Prepend a 7-byte ADTS header (no CRC) to a raw AAC-LC access unit.
fn adts_wrap(aac: &[u8], sr_index: u8, channels: u8) -> Vec<u8> {
    let frame_len = aac.len() + 7;
    // profile = AudioObjectType - 1; AAC-LC is AOT 2 -> profile 1.
    let mut h = [0u8; 7];
    h[0] = 0xFF;
    h[1] = 0xF1; // syncword | MPEG-4 | layer 0 | protection_absent
    h[2] = (1 << 6) | (sr_index << 2) | ((channels >> 2) & 1);
    h[3] = ((channels & 3) << 6) | ((frame_len >> 11) & 3) as u8;
    h[4] = ((frame_len >> 3) & 0xFF) as u8;
    h[5] = (((frame_len & 7) << 5) as u8) | 0x1F; // buffer fullness 0x7FF (top bits)
    h[6] = 0xFC; // buffer fullness (low) | num_raw_data_blocks = 0
    let mut out = Vec::with_capacity(frame_len);
    out.extend_from_slice(&h);
    out.extend_from_slice(aac);
    out
}

impl AsyncElement for FfmpegAacEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        for alt in Self::input_template().alternatives() {
            if let Ok(c) = upstream_caps.intersect(alt) {
                return Ok(c);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    /// Native `DerivedOutput`: PCM (any rate/channels in the AAC table) -> AAC at
    /// the same rate + channels.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio {
                format: AudioFormat::PcmS16Le | AudioFormat::PcmF32Le,
                channels,
                sample_rate,
            } => CapsSet::one(Caps::Audio {
                format: AudioFormat::Aac,
                channels: *channels,
                sample_rate: *sample_rate,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::Audio {
            format,
            channels,
            sample_rate,
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !matches!(format, AudioFormat::PcmS16Le | AudioFormat::PcmF32Le) {
            return Err(G2gError::CapsMismatch);
        }
        if *channels == 0 || sample_rate_index(*sample_rate).is_none() {
            return Err(G2gError::CapsMismatch);
        }
        ffmpeg::init().map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.input_format = *format;
        self.channels = *channels;
        self.sample_rate = *sample_rate;
        self.open_encoder()?;
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.ingest(slice)?;
                    let ch = self.channels as usize;
                    let frame_len = self.frame_size * ch;
                    let mut packets = Vec::new();
                    while self.pending.len() >= frame_len {
                        let chunk: Vec<f32> = self.pending.drain(..frame_len).collect();
                        packets.extend(self.encode_frame(&chunk, self.frame_size)?);
                    }
                    self.emit(packets, out).await?;
                }
                PipelinePacket::Eos => {
                    let packets = self.flush()?;
                    self.emit(packets, out).await?;
                }
                // A mid-stream PCM caps change would need a re-open; reject the
                // unsupported case, ignore an identical re-announce.
                PipelinePacket::CapsChanged(c) => {
                    if let Caps::Audio {
                        format,
                        channels,
                        sample_rate,
                    } = &c
                    {
                        if *channels != self.channels
                            || *sample_rate != self.sample_rate
                            || !matches!(format, AudioFormat::PcmS16Le | AudioFormat::PcmF32Le)
                        {
                            return Err(G2gError::CapsMismatch);
                        }
                    }
                }
                PipelinePacket::Flush => {
                    self.pending.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AAC_ENC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "libavcodec AAC encoder",
            "Codec/Encoder/Audio",
            "Encodes PCM to ADTS-framed AAC-LC via libavcodec",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "bitrate" => {
                self.bitrate_bps = (value.as_uint().ok_or(PropError::Type)? as usize).max(1);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "bitrate" => Some(PropValue::Uint(self.bitrate_bps as u64)),
            _ => None,
        }
    }
}

static AAC_ENC_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "bitrate",
    PropKind::Uint,
    "target bitrate in bits/second",
)];

impl PadTemplates for FfmpegAacEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 0,
            sample_rate: 0,
        };
        Vec::from([
            PadTemplate::sink(Self::input_template()),
            PadTemplate::source(CapsSet::one(aac)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adts_header_encodes_length_and_params() {
        // 48 kHz (index 3), stereo (2), a 9-byte AAC payload -> 16-byte AU.
        let au = adts_wrap(&[0u8; 9], 3, 2);
        assert_eq!(au.len(), 16);
        assert_eq!(au[0], 0xFF);
        assert_eq!(au[1], 0xF1);
        // profile=1(LC), sr_index=3, chan high bit 0.
        assert_eq!(au[2], (1 << 6) | (3 << 2));
        // frame_length = 16: byte3 low 2 bits 0, byte4 = 16>>3 = 2, byte5 top 3 bits = 0.
        assert_eq!(au[3] >> 6, 2, "channel config 2 in the top 2 bits");
        assert_eq!(au[4], 2, "frame_length[10:3] = 16>>3");
    }

    #[test]
    fn sample_rate_index_table() {
        assert_eq!(sample_rate_index(48000), Some(3));
        assert_eq!(sample_rate_index(44100), Some(4));
        assert_eq!(sample_rate_index(16000), Some(8));
        assert_eq!(sample_rate_index(12345), None);
    }
}

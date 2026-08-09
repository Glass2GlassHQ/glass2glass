//! Opus audio encoder element (OpusEnc, `opus` feature): `Audio{PcmS16Le}` or
//! `Audio{PcmF32Le}` in (the float path uses libopus' `opus_encode_float`, no
//! S16 quantize), `Audio{Opus}` out, via libopus through the `audiopus` crate.
//! The encode sibling of [`crate::opusdec::OpusDec`] and the producer that
//! [`crate::opusparse::OpusParse`] reads.
//!
//! Opus only encodes whole frames of one of a fixed set of durations (2.5..60
//! ms). PCM `DataFrame`s arrive at arbitrary sizes, so the element *buffers*
//! interleaved samples and emits one Opus packet per [`OpusFrameSize`]
//! (default 20 ms = 960 samples/channel at 48 kHz). At EOS a partial tail is
//! zero-padded to one full frame so no audio is lost.
//!
//! Scope (v1): 48 kHz mono/stereo S16LE. 48 kHz because Opus always *decodes* at
//! 48 kHz ([`crate::opusparse::OPUS_RATE_HZ`]), so the whole pipeline stays at
//! that rate without a resample; other input rates need an upstream
//! `AudioResample`. Bitrate, frame size, complexity, audio type and the in-band FEC pair
//! (`inband-fec` / `packet-loss-percentage`, which make each packet carry a
//! redundant copy of the previous frame for [`crate::opusdec`] to recover a lost
//! one from) are builder-set and runtime properties.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

use audiopus::coder::Encoder;
use audiopus::{Application, Bitrate, Channels, SampleRate};

use crate::opusparse::OPUS_RATE_HZ;

/// Maximum Opus packet size for a 60 ms stereo frame; the libopus-recommended
/// output scratch (`1275 * 3 + 7`).
const MAX_PACKET: usize = 4_000;

/// Highest accepted value of the `complexity` property (libopus' maximum).
const MAX_COMPLEXITY: u8 = 10;

/// Highest accepted value of the `packet-loss-percentage` property.
const MAX_PACKET_LOSS_PERC: u8 = 100;

/// libopus' own default computational complexity, applied when nothing sets the
/// `complexity` property.
const DEFAULT_COMPLEXITY: u8 = 9;

/// The Opus frame durations libopus can encode (RFC 6716 §2.1.4).
///
/// The `frame-size` property carries these as GStreamer `opusenc`'s enum
/// integers, which are whole milliseconds except that `2` means 2.5 ms (the
/// only fractional duration, and the reason the property is not simply a
/// duration in ms).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpusFrameSize {
    /// 2.5 ms, `frame-size=2`. 120 samples/channel at 48 kHz.
    Ms2_5,
    /// 5 ms, `frame-size=5`. 240 samples/channel at 48 kHz.
    Ms5,
    /// 10 ms, `frame-size=10`. 480 samples/channel at 48 kHz.
    Ms10,
    /// 20 ms, `frame-size=20`. 960 samples/channel at 48 kHz. The Opus default
    /// and a good latency/efficiency balance.
    #[default]
    Ms20,
    /// 40 ms, `frame-size=40`. 1920 samples/channel at 48 kHz.
    Ms40,
    /// 60 ms, `frame-size=60`. 2880 samples/channel at 48 kHz.
    Ms60,
}

impl OpusFrameSize {
    /// Every supported duration, shortest first.
    pub const ALL: [OpusFrameSize; 6] = [
        OpusFrameSize::Ms2_5,
        OpusFrameSize::Ms5,
        OpusFrameSize::Ms10,
        OpusFrameSize::Ms20,
        OpusFrameSize::Ms40,
        OpusFrameSize::Ms60,
    ];

    /// This duration in microseconds. Microseconds because 2.5 ms is not a whole
    /// number of milliseconds; every duration divides the 48 kHz clock exactly.
    pub const fn micros(self) -> u32 {
        match self {
            OpusFrameSize::Ms2_5 => 2_500,
            OpusFrameSize::Ms5 => 5_000,
            OpusFrameSize::Ms10 => 10_000,
            OpusFrameSize::Ms20 => 20_000,
            OpusFrameSize::Ms40 => 40_000,
            OpusFrameSize::Ms60 => 60_000,
        }
    }

    /// Samples per channel in one frame at 48 kHz, exact for every duration.
    pub const fn samples(self) -> usize {
        (self.micros() as usize * OPUS_RATE_HZ as usize) / 1_000_000
    }

    /// One frame's duration in nanoseconds, the PTS step between packets.
    pub const fn nanos(self) -> u64 {
        self.micros() as u64 * 1_000
    }

    /// The `frame-size` property value for this duration.
    pub const fn property_value(self) -> u64 {
        match self {
            OpusFrameSize::Ms2_5 => 2,
            OpusFrameSize::Ms5 => 5,
            OpusFrameSize::Ms10 => 10,
            OpusFrameSize::Ms20 => 20,
            OpusFrameSize::Ms40 => 40,
            OpusFrameSize::Ms60 => 60,
        }
    }

    /// The duration a `frame-size` property value selects, or `None` for a value
    /// Opus has no frame of.
    pub const fn from_property_value(value: u64) -> Option<Self> {
        Some(match value {
            2 => OpusFrameSize::Ms2_5,
            5 => OpusFrameSize::Ms5,
            10 => OpusFrameSize::Ms10,
            20 => OpusFrameSize::Ms20,
            40 => OpusFrameSize::Ms40,
            60 => OpusFrameSize::Ms60,
            _ => return None,
        })
    }
}

/// What libopus optimizes the encode for (`OPUS_SET_APPLICATION`), carrying
/// GStreamer `opusenc`'s `audio-type` nicks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpusAudioType {
    /// `generic`: high-fidelity, decoded audio as close as possible to the input.
    #[default]
    Generic,
    /// `voice`: VoIP / conferencing, intelligibility over fidelity.
    Voice,
    /// `restricted-lowdelay`: lowest achievable latency, no SILK layer (so no
    /// in-band FEC).
    RestrictedLowDelay,
}

impl OpusAudioType {
    /// The `audio-type` property value for this mode.
    pub const fn property_value(self) -> &'static str {
        match self {
            OpusAudioType::Generic => "generic",
            OpusAudioType::Voice => "voice",
            OpusAudioType::RestrictedLowDelay => "restricted-lowdelay",
        }
    }

    /// The mode an `audio-type` property value selects, or `None` for a value
    /// libopus has no application for.
    pub fn from_property_value(value: &str) -> Option<Self> {
        Some(match value {
            "generic" => OpusAudioType::Generic,
            "voice" => OpusAudioType::Voice,
            "restricted-lowdelay" => OpusAudioType::RestrictedLowDelay,
            _ => return None,
        })
    }

    fn application(self) -> Application {
        match self {
            OpusAudioType::Generic => Application::Audio,
            OpusAudioType::Voice => Application::Voip,
            OpusAudioType::RestrictedLowDelay => Application::LowDelay,
        }
    }
}

/// Encodes raw interleaved S16LE PCM into an Opus elementary stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::opusenc::{OpusEnc, OpusFrameSize};
///
/// let encoder = OpusEnc::new()
///     .with_bitrate(64_000)
///     .with_frame_size(OpusFrameSize::Ms20);
/// ```
pub struct OpusEnc {
    channels: u8,
    bitrate: Bitrate,
    /// Duration of the frames fed to libopus: it sets the accumulation size, the
    /// PTS step, and (through the sample count passed to `opus_encode`) the
    /// packet duration libopus codes into the TOC.
    frame_size: OpusFrameSize,
    /// libopus computational complexity, 0..=10.
    complexity: u8,
    /// What libopus optimizes for, applied at encoder construction only
    /// (`OPUS_SET_APPLICATION` is not a live ctl on a running encoder).
    audio_type: OpusAudioType,
    /// Code a redundant (LBRR) copy of each frame into the next packet, so a
    /// decoder that lost one packet can rebuild it (see [`crate::opusdec`]).
    inband_fec: bool,
    /// Expected packet loss, 0..=100 %. libopus needs it above 0 to spend bits
    /// on the FEC copy at all, and it also steers the mode choice: FEC only
    /// exists in the SILK layer, so a high enough loss keeps the encoder out of
    /// CELT-only mode.
    packet_loss_perc: u8,
    /// The bitrate actually applied to the live encoder, so a repeated BWE
    /// estimate is not re-applied every batch (`OPUS_SET_BITRATE` is cheap and
    /// glitch-free, but the ctl still need not run per packet).
    applied_bps: Option<i32>,
    enc: Option<Encoder>,
    /// Float (`PcmF32Le`) input: encode through libopus' float API instead of
    /// quantizing to S16 first. Set from the negotiated input caps.
    in_f32: bool,
    /// Interleaved S16 samples not yet packed into a full Opus frame.
    buf: Vec<i16>,
    /// The float twin of `buf` (only one is in use, per `in_f32`).
    buf_f32: Vec<f32>,
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
            .field("frame_size", &self.frame_size)
            .field("complexity", &self.complexity)
            .field("audio_type", &self.audio_type)
            .field("inband_fec", &self.inband_fec)
            .field("packet_loss_perc", &self.packet_loss_perc)
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
            frame_size: OpusFrameSize::default(),
            complexity: DEFAULT_COMPLEXITY,
            audio_type: OpusAudioType::Generic,
            inband_fec: false,
            packet_loss_perc: 0,
            applied_bps: None,
            enc: None,
            in_f32: false,
            buf: Vec::new(),
            buf_f32: Vec::new(),
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

    /// Set the duration of the emitted Opus frames. Default 20 ms. Shorter
    /// frames cut latency and cost bitrate; longer ones the reverse.
    pub fn with_frame_size(mut self, frame_size: OpusFrameSize) -> Self {
        self.frame_size = frame_size;
        self
    }

    /// The frame duration this encoder emits.
    pub fn frame_size(&self) -> OpusFrameSize {
        self.frame_size
    }

    /// Set libopus' computational complexity, 0 (cheapest) to 10 (best quality).
    /// Values above 10 clamp to 10; the `complexity` property rejects them
    /// instead. Default 9, libopus' own.
    pub fn with_complexity(mut self, complexity: u8) -> Self {
        self.complexity = complexity.min(MAX_COMPLEXITY);
        self
    }

    /// The complexity the live encoder is running at, read back from libopus.
    /// Falls back to the configured value before `configure_pipeline`.
    pub fn complexity(&self) -> u8 {
        self.enc
            .as_ref()
            .and_then(|e| e.complexity().ok())
            .unwrap_or(self.complexity)
    }

    /// Set what libopus optimizes the encode for. Default `Generic`, as in
    /// GStreamer's opusenc. Takes effect when the encoder is built, so set it
    /// before `configure_pipeline`.
    pub fn with_audio_type(mut self, audio_type: OpusAudioType) -> Self {
        self.audio_type = audio_type;
        self
    }

    /// The mode the encoder is (or will be) built with.
    pub fn audio_type(&self) -> OpusAudioType {
        self.audio_type
    }

    /// Carry a redundant copy of each frame in the next packet, so one lost
    /// packet is recoverable downstream (`opusdec use-inband-fec=true`). Costs
    /// bitrate and only takes effect together with a non-zero
    /// [`with_packet_loss_percentage`](Self::with_packet_loss_percentage).
    /// Off by default, as in GStreamer's opusenc.
    pub fn with_inband_fec(mut self, enabled: bool) -> Self {
        self.inband_fec = enabled;
        self
    }

    /// The FEC setting the live encoder is running with, read back from libopus.
    pub fn inband_fec(&self) -> bool {
        self.enc
            .as_ref()
            .and_then(|e| e.inband_fec().ok())
            .unwrap_or(self.inband_fec)
    }

    /// Tell libopus how much loss to encode for, 0..=100 %. Values above 100
    /// clamp; the `packet-loss-percentage` property rejects them instead.
    pub fn with_packet_loss_percentage(mut self, percent: u8) -> Self {
        self.packet_loss_perc = percent.min(MAX_PACKET_LOSS_PERC);
        self
    }

    /// The loss percentage the live encoder is running with, read back from
    /// libopus.
    pub fn packet_loss_percentage(&self) -> u8 {
        self.enc
            .as_ref()
            .and_then(|e| e.packet_loss_perc().ok())
            .unwrap_or(self.packet_loss_perc)
    }

    /// Count of Opus packets emitted.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The live encoder's lookahead in 48 kHz samples: how far its output lags
    /// its input, which is the pre-skip a container must declare for this
    /// stream. `None` before `configure_pipeline` builds the encoder.
    pub fn lookahead(&self) -> Option<u32> {
        self.enc.as_ref().and_then(|e| e.lookahead().ok())
    }

    fn input_templates() -> Vec<Caps> {
        // Audio caps carry no `Any`; pin the supported shapes (48 kHz stereo).
        let pcm = |format| Caps::Audio {
            format,
            channels: 2,
            sample_rate: OPUS_RATE_HZ,
        };
        Vec::from([pcm(AudioFormat::PcmS16Le), pcm(AudioFormat::PcmF32Le)])
    }

    /// Whether `caps` is an encodable PCM shape: `(float input, channels)`.
    fn pcm_shape(caps: &Caps) -> Option<(bool, u8)> {
        match caps {
            Caps::Audio {
                format: format @ (AudioFormat::PcmS16Le | AudioFormat::PcmF32Le),
                channels,
                sample_rate,
            } if (*channels == 1 || *channels == 2) && *sample_rate == OPUS_RATE_HZ => {
                Some((*format == AudioFormat::PcmF32Le, *channels))
            }
            _ => None,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::Audio {
            format: AudioFormat::Opus,
            channels: self.channels,
            sample_rate: OPUS_RATE_HZ,
        }
    }

    /// Interleaved sample count of one whole frame at the current frame size.
    fn frame_len(&self) -> usize {
        self.frame_size.samples() * self.channels as usize
    }

    /// Encode one full interleaved frame (`frame_len` samples) into an owned
    /// Opus packet. libopus reads the frame duration from the sample count, so
    /// the TOC of the packet carries whatever `frame_size` selected.
    fn encode_frame(&self, frame: &[i16]) -> Result<Vec<u8>, G2gError> {
        let enc = self.enc.as_ref().ok_or(G2gError::NotConfigured)?;
        let mut out = alloc::vec![0u8; MAX_PACKET];
        let len = enc
            .encode(frame, &mut out)
            .map_err(|_| G2gError::CapsMismatch)?;
        out.truncate(len);
        Ok(out)
    }

    /// The float twin of [`encode_frame`](Self::encode_frame) (libopus'
    /// `opus_encode_float`).
    fn encode_frame_f32(&self, frame: &[f32]) -> Result<Vec<u8>, G2gError> {
        let enc = self.enc.as_ref().ok_or(G2gError::NotConfigured)?;
        let mut out = alloc::vec![0u8; MAX_PACKET];
        let len = enc
            .encode_float(frame, &mut out)
            .map_err(|_| G2gError::CapsMismatch)?;
        out.truncate(len);
        Ok(out)
    }

    /// Encode the next full frame from the active buffer, or `None` when less
    /// than a frame is buffered.
    fn encode_next(&mut self) -> Result<Option<Vec<u8>>, G2gError> {
        let frame_len = self.frame_len();
        if self.in_f32 {
            if self.buf_f32.len() < frame_len {
                return Ok(None);
            }
            let frame: Vec<f32> = self.buf_f32.drain(..frame_len).collect();
            self.encode_frame_f32(&frame).map(Some)
        } else {
            if self.buf.len() < frame_len {
                return Ok(None);
            }
            let frame: Vec<i16> = self.buf.drain(..frame_len).collect();
            self.encode_frame(&frame).map(Some)
        }
    }

    /// Drain as many full frames as the buffer holds, returning `(packet, pts)`
    /// for each. PTS advances one frame duration per packet from the anchor.
    fn drain_frames(&mut self) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let mut packets = Vec::new();
        while let Some(packet) = self.encode_next()? {
            let pts = self.next_pts_ns.unwrap_or(0);
            self.next_pts_ns = Some(pts + self.frame_size.nanos());
            packets.push((packet, pts));
        }
        Ok(packets)
    }

    /// At EOS, zero-pad a partial tail to one full frame and encode it, so the
    /// final samples are not dropped. Returns the flushed packet, if any.
    fn flush(&mut self) -> Result<Option<(Vec<u8>, u64)>, G2gError> {
        if self.buf.is_empty() && self.buf_f32.is_empty() {
            return Ok(None);
        }
        let frame_len = self.frame_len();
        // pad with silence to a whole frame
        if self.in_f32 {
            self.buf_f32.resize(frame_len, 0.0);
        } else {
            self.buf.resize(frame_len, 0);
        }
        let packet = self.encode_next()?.expect("padded to a full frame");
        let pts = self.next_pts_ns.unwrap_or(0);
        self.next_pts_ns = Some(pts + self.frame_size.nanos());
        Ok(Some((packet, pts)))
    }

    async fn emit(
        &mut self,
        packets: Vec<(Vec<u8>, u64)>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let caps = self.output_caps();
        let feedback = crate::encoder_base::emit_packets(
            &mut self.caps_sent,
            &mut self.emitted,
            packets,
            &caps,
            out,
        )
        .await?;
        // Runtime bitrate adaptation (M721): a downstream BWE estimate
        // retargets the live encoder via `OPUS_SET_BITRATE` (no rebuild, no
        // glitch). Audio has no keyframes, so `force_keyframe` is ignored.
        if let Some(bps) = feedback.bitrate_bps {
            self.retarget(bps);
        }
        Ok(())
    }

    /// Apply a downstream bitrate target to the live encoder, clamped to the
    /// libopus-valid range, skipping a repeat of the already-applied value.
    fn retarget(&mut self, bps: u32) {
        let bps = bps.clamp(500, 512_000) as i32;
        if self.applied_bps == Some(bps) {
            return;
        }
        if let Some(enc) = self.enc.as_mut() {
            if enc.set_bitrate(Bitrate::BitsPerSecond(bps)).is_ok() {
                self.applied_bps = Some(bps);
            }
        }
    }
}

impl AsyncElement for OpusEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Self::pcm_shape(upstream_caps)
            .map(|_| upstream_caps.clone())
            .ok_or(G2gError::CapsMismatch)
    }

    fn handles_bitrate_requests(&self) -> bool {
        true
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match Self::pcm_shape(input) {
            Some((_, channels)) => CapsSet::one(Caps::Audio {
                format: AudioFormat::Opus,
                channels,
                sample_rate: OPUS_RATE_HZ,
            }),
            None => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (in_f32, channels) = Self::pcm_shape(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.in_f32 = in_f32;
        let ch = match channels {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            _ => return Err(G2gError::CapsMismatch),
        };
        let mut enc = Encoder::new(SampleRate::Hz48000, ch, self.audio_type.application())
            .map_err(|_| G2gError::CapsMismatch)?;
        enc.set_bitrate(self.bitrate)
            .map_err(|_| G2gError::CapsMismatch)?;
        enc.set_complexity(self.complexity)
            .map_err(|_| G2gError::CapsMismatch)?;
        enc.set_inband_fec(self.inband_fec)
            .map_err(|_| G2gError::CapsMismatch)?;
        enc.set_packet_loss_perc(self.packet_loss_perc)
            .map_err(|_| G2gError::CapsMismatch)?;
        self.enc = Some(enc);
        self.channels = channels;
        self.buf.clear();
        self.buf_f32.clear();
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Opus audio encoder",
            "Codec/Encoder/Audio",
            "Encodes raw S16LE / F32LE PCM to Opus",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "bitrate",
                PropKind::Uint,
                "target bitrate, bits/second (0 = libopus auto)",
            )
            .with_default("0"),
            PropertySpec::new(
                "frame-size",
                PropKind::Uint,
                "duration of one Opus frame, in ms (2 means 2.5)",
            )
            .with_default("20")
            .with_enum_values("2 (2.5 ms) | 5 | 10 | 20 | 40 | 60"),
            PropertySpec::new(
                "complexity",
                PropKind::Uint,
                "libopus computational complexity, 0 (cheapest) to 10 (best)",
            )
            .with_default("9")
            .with_range("0", "10"),
            PropertySpec::new(
                "audio-type",
                PropKind::Str,
                "what libopus optimizes for: generic | voice | restricted-lowdelay",
            )
            .with_default("generic")
            .with_enum_values("generic | voice | restricted-lowdelay"),
            PropertySpec::new(
                "inband-fec",
                PropKind::Bool,
                "carry a redundant copy of each frame in the next packet",
            )
            .with_default("false"),
            PropertySpec::new(
                "packet-loss-percentage",
                PropKind::Uint,
                "loss the encoder codes for, 0-100 % (0 spends no bits on FEC)",
            )
            .with_default("0")
            .with_range("0", "100"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            // bits/second; 0 lets libopus pick a bitrate from the sample rate / channels.
            "bitrate" => {
                let bps = value.as_uint().ok_or(PropError::Type)?;
                self.bitrate = if bps == 0 {
                    Bitrate::Auto
                } else {
                    Bitrate::BitsPerSecond(bps.min(i32::MAX as u64) as i32)
                };
                Ok(())
            }
            // GStreamer opusenc's enum integers: whole ms, except 2 = 2.5 ms.
            "frame-size" => {
                let value = value.as_uint().ok_or(PropError::Type)?;
                self.frame_size =
                    OpusFrameSize::from_property_value(value).ok_or(PropError::Value)?;
                Ok(())
            }
            // OPUS_SET_COMPLEXITY is a live ctl, so a running encoder retargets
            // without a rebuild.
            "complexity" => {
                let value = value.as_uint().ok_or(PropError::Type)?;
                if value > MAX_COMPLEXITY as u64 {
                    return Err(PropError::Value);
                }
                self.complexity = value as u8;
                if let Some(enc) = self.enc.as_mut() {
                    enc.set_complexity(self.complexity)
                        .map_err(|_| PropError::Value)?;
                }
                Ok(())
            }
            // The application is fixed at encoder construction, so this only
            // has an effect while the element is unconfigured.
            "audio-type" => {
                let value = value.as_str().ok_or(PropError::Type)?;
                self.audio_type =
                    OpusAudioType::from_property_value(value).ok_or(PropError::Value)?;
                Ok(())
            }
            // Both FEC ctls are live, like complexity: a running encoder starts
            // (or stops) coding the redundant copy without a rebuild.
            "inband-fec" => {
                self.inband_fec = value.as_bool().ok_or(PropError::Type)?;
                if let Some(enc) = self.enc.as_mut() {
                    enc.set_inband_fec(self.inband_fec)
                        .map_err(|_| PropError::Value)?;
                }
                Ok(())
            }
            "packet-loss-percentage" => {
                let value = value.as_uint().ok_or(PropError::Type)?;
                if value > MAX_PACKET_LOSS_PERC as u64 {
                    return Err(PropError::Value);
                }
                self.packet_loss_perc = value as u8;
                if let Some(enc) = self.enc.as_mut() {
                    enc.set_packet_loss_perc(self.packet_loss_perc)
                        .map_err(|_| PropError::Value)?;
                }
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "bitrate" => Some(PropValue::Uint(match self.bitrate {
                Bitrate::BitsPerSecond(b) => b.max(0) as u64,
                _ => 0,
            })),
            "frame-size" => Some(PropValue::Uint(self.frame_size.property_value())),
            "complexity" => Some(PropValue::Uint(self.complexity() as u64)),
            "audio-type" => Some(PropValue::Str(self.audio_type.property_value().into())),
            "inband-fec" => Some(PropValue::Bool(self.inband_fec())),
            "packet-loss-percentage" => Some(PropValue::Uint(self.packet_loss_percentage() as u64)),
            _ => None,
        }
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
                    // Anchor the output timeline to the first input PTS.
                    if self.next_pts_ns.is_none() {
                        self.next_pts_ns = Some(frame.timing.pts_ns);
                    }
                    // Append interleaved samples (in the negotiated width) to
                    // the pending buffer.
                    let bytes = slice;
                    if self.in_f32 {
                        self.buf_f32.extend(
                            bytes
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                        );
                    } else {
                        self.buf.extend(
                            bytes
                                .chunks_exact(2)
                                .map(|c| i16::from_le_bytes([c[0], c[1]])),
                        );
                    }
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
                    self.buf_f32.clear();
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
        let out = Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: OPUS_RATE_HZ,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(Self::input_templates())),
            PadTemplate::source(CapsSet::one(out)),
        ])
    }
}

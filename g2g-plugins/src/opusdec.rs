//! Opus audio decoder element (OpusDec, `opus` feature): `Audio{Opus}` in,
//! `Audio{PcmS16Le}` (or negotiated `PcmF32Le`) out, via libopus through the `audiopus` crate. The decode
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
//! A chained file (M827) hands the decoder a fresh `OpusHead` per physical
//! stream, behind the `Segment` that opens the chain. That header rebuilds the
//! decoder, so each chain decodes exactly as it would on its own instead of
//! carrying the previous stream's state across the boundary (M830). A header
//! merely re-stated mid-stream, identical bytes and no segment, does not.
//!
//! Pre-skip / end-trim: Opus streams carry encoder lookahead (pre-skip) at the
//! head and codec padding at the tail. `OggDemux` forwards the `OpusHead` in-band
//! (its pre-skip drops the leading output samples) and marks the final packet(s)
//! short via `duration_ns` (the end-of-stream granule trim), so the decoded PCM
//! matches ffmpeg / gstreamer sample-for-sample. Streams with no `OpusHead` and
//! no per-frame duration (RTP) decode untrimmed, as before.
//!
//! Packet-loss concealment (`plc`, off by default like GStreamer's opusdec):
//! nothing upstream signals loss (no depayloader or demuxer marks a gap), so a
//! loss is inferred from the timeline. Each packet's TOC gives its duration, so
//! the decoder knows where the next one is due; a packet arriving later than
//! that leaves a hole, which libopus fills by decoding a `None` packet for the
//! missing duration. Runs longer than [`MAX_PLC_SAMPLES`] are a seek or a
//! stream restart rather than loss, and re-anchor the timeline with no fill.
//!
//! In-band FEC (`use-inband-fec`): a stream encoded with FEC carries a
//! low-bitrate redundant copy (LBRR) of each frame in the *next* packet, so a
//! gap whose next packet did arrive can be reconstructed instead of concealed
//! blind. The gap is filled from its end backwards: the last frame duration
//! comes from the arriving packet decoded with `fec=1`, anything earlier (a
//! multi-packet loss) stays PLC. A packet carrying no LBRR (CELT-only, or an
//! encoder with FEC off) decodes as PLC inside libopus, so the fill is never
//! worse than concealment. Either knob on fills the gap, with the same PTS span
//! and sample accounting.
//!
//! Scope: 48 kHz mono/stereo, S16LE (default) or F32LE output.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use audiopus::coder::Decoder;
use audiopus::packet::Packet;
use audiopus::{Channels, MutSignals, SampleRate};

use crate::opusparse::{packet_frame_samples, packet_samples, parse_opus_head, OPUS_RATE_HZ};

/// Largest Opus frame is 120 ms; at 48 kHz that is 5760 samples per channel. The
/// decode output buffer is sized for it so any single packet fits.
const MAX_FRAME_SAMPLES: usize = (OPUS_RATE_HZ as usize * 120) / 1000;

/// Concealment granularity: libopus fills whole 2.5 ms steps and cannot conceal
/// anything shorter, so a gap is rounded to a multiple of this (120 samples at
/// 48 kHz) before it is filled.
const PLC_STEP_SAMPLES: u64 = OPUS_RATE_HZ as u64 / 400;

/// Longest concealment run synthesized for one gap, 200 ms. libopus' PLC fades
/// to silence within a handful of frames, so a longer fill adds nothing, and a
/// timeline jump that large is a seek or a stream restart rather than packet
/// loss. GStreamer's opusdec has no equivalent bound because it is told where
/// the gaps are; inferring them from timestamps needs one.
const MAX_PLC_SAMPLES: u64 = (OPUS_RATE_HZ as u64 * 200) / 1000;

/// Concealment step used until a packet has been seen, 20 ms (GStreamer
/// opusdec's assumption for an unknown duration).
const DEFAULT_PLC_CHUNK_SAMPLES: u64 = (OPUS_RATE_HZ as u64 * 20) / 1000;

/// Decodes an Opus elementary stream into raw interleaved S16LE PCM.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::opusdec::OpusDec;
///
/// let decoder = OpusDec::new().with_plc(true).with_inband_fec(true);
/// assert!(decoder.plc());
/// ```
pub struct OpusDec {
    channels: u8,
    /// Negotiated output sample format: `PcmS16Le` (default) or `PcmF32Le`
    /// (libopus' float API, no i16 quantize). Settled by the solver via
    /// `configure_output` / the runner's pre-fixed output `CapsChanged`.
    out_format: AudioFormat,
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
    /// Packet-loss concealment, off by default (GStreamer opusdec's default).
    plc: bool,
    /// Use the in-band FEC (LBRR) copy in the packet after a gap to rebuild the
    /// last lost frame, off by default (GStreamer opusdec's default).
    fec: bool,
    /// PTS the next packet is due at: the last packet's PTS plus its coded
    /// duration. A packet arriving later than this left a gap. `None` until a
    /// packet has been seen, and reset by a flush or a fresh `OpusHead`.
    next_pts_ns: Option<u64>,
    /// Coded duration (per-channel samples) of the last packet, the step
    /// concealment is synthesized in.
    prev_samples: u64,
    /// A `Segment` arrived and nothing has been decoded since. The next in-band
    /// `OpusHead` then opens a new logical stream (a chained file's next
    /// physical stream, M830) rather than re-stating the current one, so the
    /// decoder is rebuilt even when the header is identical.
    new_segment: bool,
}

impl core::fmt::Debug for OpusDec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // audiopus' Decoder is not Debug; report the configuration instead.
        f.debug_struct("OpusDec")
            .field("channels", &self.channels)
            .field("sequence", &self.sequence)
            .field("configured", &self.configured)
            .field("plc", &self.plc)
            .field("fec", &self.fec)
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
            out_format: AudioFormat::PcmS16Le,
            dec: None,
            last_out: None,
            sequence: 0,
            configured: false,
            pre_skip: 0,
            decoded_samples: 0,
            plc: false,
            fec: false,
            next_pts_ns: None,
            prev_samples: 0,
            new_segment: false,
        }
    }

    /// Enable packet-loss concealment: a gap in the packet timeline is filled
    /// with audio libopus synthesizes from the decoder state rather than left as
    /// a jump. Off by default.
    pub fn with_plc(mut self, plc: bool) -> Self {
        self.plc = plc;
        self
    }

    pub fn plc(&self) -> bool {
        self.plc
    }

    /// Use the redundant copy carried by the packet after a gap to rebuild the
    /// last lost frame, instead of concealing it blind. Off by default; on, it
    /// fills a gap even with `plc` off.
    pub fn with_inband_fec(mut self, fec: bool) -> Self {
        self.fec = fec;
        self
    }

    pub fn inband_fec(&self) -> bool {
        self.fec
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
            format: self.out_format,
            channels: self.channels,
            sample_rate: OPUS_RATE_HZ,
        }
    }

    /// Decode one Opus packet, returning interleaved i16 samples and the
    /// per-channel sample count. `opus = None` is a lost packet: libopus
    /// conceals it, and then `capacity` is not a bound but the exact duration to
    /// synthesize, so it must be a whole number of 2.5 ms steps. `fec` decodes
    /// the redundant copy of the frame *before* `opus` under that same
    /// exact-duration rule, rather than the packet's own audio.
    fn decode(
        &mut self,
        opus: Option<&[u8]>,
        capacity: usize,
        fec: bool,
    ) -> Result<(Vec<i16>, usize), G2gError> {
        let channels = self.channels as usize;
        let dec = self.dec.as_mut().ok_or(G2gError::NotConfigured)?;
        let packet = opus
            .map(|o| Packet::try_from(o).map_err(|_| G2gError::CapsMismatch))
            .transpose()?;
        let mut pcm = alloc::vec![0i16; capacity * channels];
        let per_channel = {
            let signals = MutSignals::try_from(&mut pcm[..]).map_err(|_| G2gError::CapsMismatch)?;
            dec.decode(packet, signals, fec)
                .map_err(|_| G2gError::CapsMismatch)?
        };
        pcm.truncate(per_channel * channels);
        Ok((pcm, per_channel))
    }

    /// The float twin of [`decode`](Self::decode) (libopus' `opus_decode_float`,
    /// used when `PcmF32Le` output is negotiated: no i16 quantize).
    fn decode_f32(
        &mut self,
        opus: Option<&[u8]>,
        capacity: usize,
        fec: bool,
    ) -> Result<(Vec<f32>, usize), G2gError> {
        let channels = self.channels as usize;
        let dec = self.dec.as_mut().ok_or(G2gError::NotConfigured)?;
        let packet = opus
            .map(|o| Packet::try_from(o).map_err(|_| G2gError::CapsMismatch))
            .transpose()?;
        let mut pcm = alloc::vec![0f32; capacity * channels];
        let per_channel = {
            let signals = MutSignals::try_from(&mut pcm[..]).map_err(|_| G2gError::CapsMismatch)?;
            dec.decode_float(packet, signals, fec)
                .map_err(|_| G2gError::CapsMismatch)?
        };
        pcm.truncate(per_channel * channels);
        Ok((pcm, per_channel))
    }

    /// The interleaved-sample window of a decoded frame: drops the pre-skip
    /// lookahead at the stream head and any padding past `keep` (per-channel
    /// valid count from `duration_ns`; `None` keeps the whole frame). Returns
    /// `(start, end)` interleaved-sample indices and advances the accounting.
    fn valid_window(&mut self, per_channel: usize, keep: Option<u64>) -> (usize, usize) {
        let channels = self.channels as usize;
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
        (head as usize * channels, end as usize * channels)
    }

    /// Decode `opus` (or conceal a lost packet, see [`decode`](Self::decode))
    /// and serialize only the valid window (see
    /// [`valid_window`](Self::valid_window)) to little-endian bytes of the
    /// negotiated output format.
    fn decode_trimmed(
        &mut self,
        opus: Option<&[u8]>,
        capacity: usize,
        keep: Option<u64>,
        fec: bool,
    ) -> Result<Vec<u8>, G2gError> {
        if self.out_format == AudioFormat::PcmF32Le {
            let (pcm, per_channel) = self.decode_f32(opus, capacity, fec)?;
            let (start_i, end_i) = self.valid_window(per_channel, keep);
            let mut bytes = Vec::with_capacity((end_i - start_i) * 4);
            for &s in &pcm[start_i..end_i] {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            return Ok(bytes);
        }
        let (pcm, per_channel) = self.decode(opus, capacity, fec)?;
        let (start_i, end_i) = self.valid_window(per_channel, keep);
        let mut bytes = Vec::with_capacity((end_i - start_i) * 2);
        for &s in &pcm[start_i..end_i] {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        Ok(bytes)
    }

    /// Bytes one interleaved sample of every channel occupies in the output.
    fn bytes_per_sample(&self) -> usize {
        self.channels as usize
            * if self.out_format == AudioFormat::PcmF32Le {
                4
            } else {
                2
            }
    }

    /// Fill PCM for the hole in front of the packet `next` arriving at `pts_ns`,
    /// and the PTS it starts at. `None` when there is nothing to fill: no
    /// timeline yet, both `plc` and `fec` off, the packet on time, or a jump
    /// past [`MAX_PLC_SAMPLES`] (a seek, which re-anchors instead).
    ///
    /// The gap is rounded to the nearest whole 2.5 ms, which is what libopus
    /// conceals in. With `fec`, its final frame duration is decoded from `next`'s
    /// redundant copy of it; the rest is synthesized in steps of the last
    /// packet's own duration, since one `opus_decode` call cannot exceed the
    /// 120 ms decode buffer.
    fn fill_gap(&mut self, pts_ns: u64, next: &[u8]) -> Result<Option<(u64, Vec<u8>)>, G2gError> {
        let Some(expected) = self.next_pts_ns else {
            return Ok(None);
        };
        if !self.plc && !self.fec {
            return Ok(None);
        }
        let missing = ns_to_samples(pts_ns.saturating_sub(expected));
        let gap = ((missing + PLC_STEP_SAMPLES / 2) / PLC_STEP_SAMPLES) * PLC_STEP_SAMPLES;
        if gap == 0 || gap > MAX_PLC_SAMPLES {
            return Ok(None);
        }
        // The step comes from a packet's TOC, so clamp it into the decode buffer
        // and onto the 2.5 ms grid rather than trusting the stream's frame count.
        let chunk = match self.prev_samples {
            0 => DEFAULT_PLC_CHUNK_SAMPLES,
            n => n.min(MAX_FRAME_SAMPLES as u64),
        };
        let chunk = (chunk / PLC_STEP_SAMPLES).max(1) * PLC_STEP_SAMPLES;
        // `next` carries the redundant copy of the frame immediately before it,
        // so FEC reaches back exactly one of its frames: everything earlier in a
        // multi-packet gap is still blind concealment.
        let recovered = if self.fec {
            u64::from(packet_frame_samples(next)).min(gap)
        } else {
            0
        };
        let concealed = gap - recovered;

        let mut pcm = Vec::with_capacity(gap as usize * self.bytes_per_sample());
        let mut done = 0;
        while done < concealed {
            let want = chunk.min(concealed - done);
            pcm.extend_from_slice(&self.decode_trimmed(None, want as usize, None, false)?);
            done += want;
        }
        if recovered > 0 {
            pcm.extend_from_slice(&self.decode_trimmed(
                Some(next),
                recovered as usize,
                None,
                true,
            )?);
        }
        Ok(Some((expected, pcm)))
    }

    /// Push `pcm` as one output frame, preceded by a `CapsChanged` whenever the
    /// output caps changed since the last one.
    async fn emit(
        &mut self,
        pcm: Vec<u8>,
        timing: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let new_caps = self.output_caps();
        if self.last_out.as_ref() != Some(&new_caps) {
            out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                .await?;
            self.last_out = Some(new_caps);
        }
        let decoded = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
            timing,
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(decoded)).await?;
        Ok(())
    }
}

/// Per-channel 48 kHz sample count for a duration in nanoseconds, rounded to the
/// nearest sample.
fn ns_to_samples(ns: u64) -> u64 {
    (ns.saturating_mul(OPUS_RATE_HZ as u64)
        .saturating_add(500_000_000))
        / 1_000_000_000
}

/// The exact inverse for whole samples: 48 kHz divides a second evenly.
fn samples_to_ns(samples: u64) -> u64 {
    samples.saturating_mul(1_000_000_000) / OPUS_RATE_HZ as u64
}

/// Per-channel valid sample count encoded in a frame's `duration_ns` (48 kHz),
/// or `None` when unset (`0`). Rounds to the nearest sample; the demuxer's
/// truncating ns conversion round-trips back to the exact count.
fn duration_to_samples(duration_ns: u64) -> Option<u64> {
    if duration_ns == 0 {
        return None;
    }
    Some(ns_to_samples(duration_ns))
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

    /// Native `DerivedOutput`: Opus in -> interleaved `PcmS16Le` (preferred) or
    /// `PcmF32Le` out at 48 kHz.
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
            } => CapsSet::from_alternatives(Vec::from([
                Caps::Audio {
                    format: AudioFormat::PcmS16Le,
                    channels: ANY_CHANNELS,
                    sample_rate: OPUS_RATE_HZ,
                },
                Caps::Audio {
                    format: AudioFormat::PcmF32Le,
                    channels: ANY_CHANNELS,
                    sample_rate: OPUS_RATE_HZ,
                },
            ])),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    /// The solver's chosen output format (S16 by default, F32 when a
    /// float-consuming downstream negotiated it).
    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        match output_caps {
            Caps::Audio {
                format: format @ (AudioFormat::PcmS16Le | AudioFormat::PcmF32Le),
                ..
            } => {
                self.out_format = *format;
                Ok(())
            }
            _ => Err(G2gError::CapsMismatch),
        }
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
            "Decodes Opus to raw S16LE / F32LE PCM",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "plc",
                PropKind::Bool,
                "conceal lost packets with audio synthesized from the decoder state",
            )
            .with_default("false"),
            PropertySpec::new(
                "use-inband-fec",
                PropKind::Bool,
                "rebuild a lost frame from the redundant copy in the next packet",
            )
            .with_default("false"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "plc" => {
                self.plc = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "use-inband-fec" => {
                self.fec = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "plc" => Some(PropValue::Bool(self.plc)),
            "use-inband-fec" => Some(PropValue::Bool(self.fec)),
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
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    // An in-band OpusHead is codec config, not audio: read its
                    // channel count + pre-skip, (re)build the decoder, and consume
                    // it (no PCM out). The demuxer forwards it before the audio.
                    if let Some((channels, pre_skip)) = parse_opus_head(slice) {
                        // A header opening a new logical stream (a channel-count
                        // change, or a chained physical stream, which a `Segment`
                        // announces, M830) needs a decoder that has not carried
                        // the previous stream's state into it: that stream's
                        // encoder started cold, so the decoder must too. A header
                        // merely re-stated mid-stream keeps the running decoder.
                        if self.channels != channels || self.new_segment {
                            self.build_decoder(channels)?;
                        }
                        // Fresh stream, fresh pre-skip window and PLC cadence.
                        self.pre_skip = pre_skip as u32;
                        self.decoded_samples = 0;
                        self.next_pts_ns = None;
                        self.prev_samples = 0;
                        self.new_segment = false;
                        return Ok(());
                    }
                    self.new_segment = false;
                    // Fill any hole this packet's PTS opens up before decoding
                    // it, so the filled audio keeps the output contiguous and
                    // the real packet still lands at its own PTS. This packet is
                    // also where the FEC copy of the lost frame lives.
                    if let Some((pts_ns, pcm)) = self.fill_gap(frame.timing.pts_ns, slice)? {
                        if !pcm.is_empty() {
                            let samples = (pcm.len() / self.bytes_per_sample()) as u64;
                            self.emit(
                                pcm,
                                FrameTiming {
                                    pts_ns,
                                    dts_ns: pts_ns,
                                    duration_ns: samples_to_ns(samples),
                                    ..FrameTiming::default()
                                },
                                out,
                            )
                            .await?;
                        }
                    }
                    // The coded duration, not the trimmed one: a container
                    // advances the next packet's PTS by the whole frame.
                    self.prev_samples = u64::from(packet_samples(slice));
                    self.next_pts_ns = Some(
                        frame
                            .timing
                            .pts_ns
                            .saturating_add(samples_to_ns(self.prev_samples)),
                    );

                    let keep = duration_to_samples(frame.timing.duration_ns);
                    let pcm = self.decode_trimmed(Some(slice), MAX_FRAME_SAMPLES, keep, false)?;
                    // A frame fully inside the pre-skip window trims to nothing;
                    // consume it without emitting an empty PCM frame.
                    if pcm.is_empty() {
                        return Ok(());
                    }
                    self.emit(pcm, frame.timing, out).await?;
                }
                PipelinePacket::Flush => {
                    // A seek/flush restarts sample accounting; the re-read stream
                    // re-sends its OpusHead, which resets pre-skip again.
                    self.pre_skip = 0;
                    self.decoded_samples = 0;
                    self.next_pts_ns = None;
                    self.prev_samples = 0;
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
                    // The runner's pre-fixed forward output caps: adopt the
                    // chosen sample format and forward on.
                    Caps::Audio {
                        format: format @ (AudioFormat::PcmS16Le | AudioFormat::PcmF32Le),
                        ..
                    } => {
                        self.out_format = *format;
                        out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                        self.last_out = Some(c);
                    }
                    _ => return Err(G2gError::CapsMismatch),
                },
                // A new segment marks a stream boundary (a chained file's next
                // physical stream): remember it for the header that follows.
                PipelinePacket::Segment(seg) => {
                    self.new_segment = true;
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // the runner forwards Eos after process(Eos) returns; re-emitting
                // it here races the sink's exit on the first one.
                PipelinePacket::Eos => {}
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
        let out = |format| Caps::Audio {
            format,
            channels: 2,
            sample_rate: OPUS_RATE_HZ,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_template())),
            PadTemplate::source(CapsSet::from_alternatives(Vec::from([
                out(AudioFormat::PcmS16Le),
                out(AudioFormat::PcmF32Le),
            ]))),
        ])
    }
}

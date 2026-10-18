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
//! Scope: 48 kHz, the `OpusHead` channel mapping or the RFC 7845 defaults for
//! the channel count, S16LE (default) or F32LE output.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::time::{ns_to_ticks, ticks_to_ns};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ChannelLayout, ChannelPosition,
    ConfigureOutcome, ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    Segment, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use audiopus::ffi;

use crate::opusparse::{
    mapping_len, packet_frame_samples, packet_samples, parse_opus_head, OPUS_HEAD_FIXED,
    OPUS_RATE_HZ,
};
use crate::pcmclip::clip_pcm_to_segment;

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

const MAPPING_FAMILY_RTP: u8 = 0;
const MAPPING_FAMILY_VORBIS: u8 = 1;
const MAPPING_FAMILY_OFFSET: usize = OPUS_HEAD_FIXED - 1;

// RFC 7845 section 5.1.1.2: streams, coupled streams, table per channel count
const VORBIS_MAPPINGS: [(u8, u8, &[u8]); 8] = [
    (1, 0, &[0]),
    (1, 1, &[0, 1]),
    (2, 1, &[0, 2, 1]),
    (2, 2, &[0, 1, 2, 3]),
    (3, 2, &[0, 4, 1, 2, 3]),
    (4, 2, &[0, 4, 1, 2, 3, 5]),
    (4, 3, &[0, 4, 1, 2, 3, 5, 6]),
    (5, 3, &[0, 6, 1, 2, 3, 4, 5, 7]),
];

// the Vorbis-order channel behind each channel of the ffmpeg default order
fn vorbis_layout(channels: u8) -> Option<(ChannelLayout, &'static [usize])> {
    use ChannelPosition::*;
    Some(match channels {
        3 => (ChannelLayout::of(&[Fl, Fr, Fc]), &[0, 2, 1]),
        4 => (ChannelLayout::of(&[Fl, Fr, Bl, Br]), &[0, 1, 2, 3]),
        5 => (ChannelLayout::of(&[Fl, Fr, Fc, Bl, Br]), &[0, 2, 1, 3, 4]),
        6 => (ChannelLayout::SURROUND_5_1, &[0, 2, 1, 5, 3, 4]),
        7 => (
            ChannelLayout::of(&[Fl, Fr, Fc, Lfe, Bc, Sl, Sr]),
            &[0, 2, 1, 6, 5, 3, 4],
        ),
        8 => (ChannelLayout::SURROUND_7_1, &[0, 2, 1, 7, 5, 6, 3, 4]),
        _ => return None,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChannelMapping {
    streams: u8,
    coupled_streams: u8,
    table: Vec<u8>,
    layout: ChannelLayout,
}

impl ChannelMapping {
    fn default_for(channels: u8) -> Option<Self> {
        let family = if channels <= 2 {
            MAPPING_FAMILY_RTP
        } else {
            MAPPING_FAMILY_VORBIS
        };
        let &(streams, coupled_streams, table) =
            VORBIS_MAPPINGS.get(usize::from(channels).checked_sub(1)?)?;
        Self::new(family, streams, coupled_streams, table)
    }

    fn from_opus_head(head: &[u8]) -> Option<Self> {
        let (channels, _) = parse_opus_head(head)?;
        let family = *head.get(MAPPING_FAMILY_OFFSET)?;
        if family == MAPPING_FAMILY_RTP {
            return Self::default_for(channels).filter(|_| channels <= 2);
        }
        let table = head.get(OPUS_HEAD_FIXED..OPUS_HEAD_FIXED + mapping_len(family, channels))?;
        let (&streams, rest) = table.split_first()?;
        let (&coupled_streams, table) = rest.split_first()?;
        Self::new(family, streams, coupled_streams, table)
    }

    // family 1 output is permuted to the ffmpeg default order of its layout
    fn new(family: u8, streams: u8, coupled_streams: u8, table: &[u8]) -> Option<Self> {
        let coded_channels = u16::from(streams) + u16::from(coupled_streams);
        if streams == 0 || coupled_streams > streams || coded_channels > u16::from(u8::MAX) {
            return None;
        }
        let channels = u8::try_from(table.len()).ok()?;
        let reorder = (family == MAPPING_FAMILY_VORBIS)
            .then(|| vorbis_layout(channels))
            .flatten();
        let Some((layout, vorbis_index)) = reorder else {
            return Some(Self {
                streams,
                coupled_streams,
                table: table.to_vec(),
                layout: ChannelLayout::UNSPECIFIED,
            });
        };
        Some(Self {
            streams,
            coupled_streams,
            table: vorbis_index.iter().map(|&index| table[index]).collect(),
            layout,
        })
    }

    fn channels(&self) -> u8 {
        self.table.len() as u8
    }
}

struct MultistreamDecoder(core::ptr::NonNull<ffi::OpusMSDecoder>);

// SAFETY: libopus decoder state has no thread affinity and is only reached via `&mut self`.
unsafe impl Send for MultistreamDecoder {}

impl MultistreamDecoder {
    fn new(mapping: &ChannelMapping) -> Result<Self, G2gError> {
        let mut error = 0;
        // SAFETY: `table` has one entry per output channel and `error` is a valid out-pointer.
        let decoder = unsafe {
            ffi::opus_multistream_decoder_create(
                OPUS_RATE_HZ as i32,
                i32::from(mapping.channels()),
                i32::from(mapping.streams),
                i32::from(mapping.coupled_streams),
                mapping.table.as_ptr(),
                &mut error,
            )
        };
        match core::ptr::NonNull::new(decoder) {
            Some(decoder) if error == ffi::OPUS_OK => Ok(Self(decoder)),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn checked_samples(decoded: i32) -> Result<usize, G2gError> {
        usize::try_from(decoded).map_err(|_| G2gError::CapsMismatch)
    }
}

impl Drop for MultistreamDecoder {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `opus_multistream_decoder_create`, destroyed once here.
        unsafe { ffi::opus_multistream_decoder_destroy(self.0.as_ptr()) }
    }
}

// a lost packet is null data for libopus
fn packet_parts(opus: Option<&[u8]>) -> Result<(*const u8, i32), G2gError> {
    match opus {
        Some(data) => Ok((
            data.as_ptr(),
            i32::try_from(data.len()).map_err(|_| G2gError::CapsMismatch)?,
        )),
        None => Ok((core::ptr::null(), 0)),
    }
}

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
    mapping: Option<ChannelMapping>,
    /// Negotiated output sample format: `PcmS16Le` (default) or `PcmF32Le`
    /// (libopus' float API, no i16 quantize). Settled by the solver via
    /// `configure_output` / the runner's pre-fixed output `CapsChanged`.
    out_format: AudioFormat,
    dec: Option<MultistreamDecoder>,
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
    segment: Segment,
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
            mapping: None,
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
            segment: Segment::new(),
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

    /// (Re)create the libopus decoder for a channel mapping. Called from
    /// `configure_pipeline` when the negotiated input already carries a real
    /// count, from `process` when the demuxer's `CapsChanged` delivers it, and
    /// from an in-band `OpusHead`.
    fn build_decoder(&mut self, mapping: ChannelMapping) -> Result<(), G2gError> {
        self.dec = Some(MultistreamDecoder::new(&mapping)?);
        self.channels = mapping.channels();
        self.mapping = Some(mapping);
        Ok(())
    }

    fn build_default_decoder(&mut self, channels: u8) -> Result<(), G2gError> {
        self.build_decoder(ChannelMapping::default_for(channels).ok_or(G2gError::CapsMismatch)?)
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
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::Audio {
            format: self.out_format,
            channels: self.channels,
            sample_rate: OPUS_RATE_HZ,
            channel_layout: self
                .mapping
                .as_ref()
                .map_or(ChannelLayout::UNSPECIFIED, |mapping| mapping.layout),
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
        let (data, len) = packet_parts(opus)?;
        let mut pcm = alloc::vec![0i16; capacity * channels];
        let frame_size = i32::try_from(capacity).map_err(|_| G2gError::CapsMismatch)?;
        // SAFETY: `data` is `len` bytes or null, `pcm` holds `frame_size` samples per channel.
        let decoded = unsafe {
            ffi::opus_multistream_decode(
                dec.0.as_ptr(),
                data,
                len,
                pcm.as_mut_ptr(),
                frame_size,
                i32::from(fec),
            )
        };
        let per_channel = MultistreamDecoder::checked_samples(decoded)?;
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
        let (data, len) = packet_parts(opus)?;
        let mut pcm = alloc::vec![0f32; capacity * channels];
        let frame_size = i32::try_from(capacity).map_err(|_| G2gError::CapsMismatch)?;
        // SAFETY: `data` is `len` bytes or null, `pcm` holds `frame_size` samples per channel.
        let decoded = unsafe {
            ffi::opus_multistream_decode_float(
                dec.0.as_ptr(),
                data,
                len,
                pcm.as_mut_ptr(),
                frame_size,
                i32::from(fec),
            )
        };
        let per_channel = MultistreamDecoder::checked_samples(decoded)?;
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
        let missing = ns_to_ticks(pts_ns.saturating_sub(expected), OPUS_RATE_HZ);
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

    /// Push the part of `pcm` inside the segment as one output frame, preceded
    /// by a `CapsChanged` whenever the output caps changed since the last one.
    async fn emit(
        &mut self,
        pcm: Vec<u8>,
        timing: FrameTiming,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let Some((timing, pcm)) = clip_pcm_to_segment(
            &self.segment,
            timing,
            pcm,
            self.bytes_per_sample(),
            OPUS_RATE_HZ,
        )?
        else {
            return Ok(());
        };
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

/// Per-channel valid sample count encoded in a frame's `duration_ns` (48 kHz),
/// or `None` when unset (`0`). Rounds to the nearest sample; the demuxer's
/// ns conversion round-trips back to the exact count.
fn duration_to_samples(duration_ns: u64) -> Option<u64> {
    if duration_ns == 0 {
        return None;
    }
    Some(ns_to_ticks(duration_ns, OPUS_RATE_HZ))
}

impl AsyncElement for OpusDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

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
                    channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
                },
                Caps::Audio {
                    format: AudioFormat::PcmF32Le,
                    channels: ANY_CHANNELS,
                    sample_rate: OPUS_RATE_HZ,
                    channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
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
        if *channels != ANY_CHANNELS {
            self.build_default_decoder(*channels)?;
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
                    // channel mapping + pre-skip, (re)build the decoder, and consume
                    // it (no PCM out). The demuxer forwards it before the audio.
                    if let Some((_, pre_skip)) = parse_opus_head(slice) {
                        let mapping =
                            ChannelMapping::from_opus_head(slice).ok_or(G2gError::CapsMismatch)?;
                        // A header opening a new logical stream (a channel-mapping
                        // change, or a chained physical stream, which a `Segment`
                        // announces, M830) needs a decoder that has not carried
                        // the previous stream's state into it: that stream's
                        // encoder started cold, so the decoder must too. A header
                        // merely re-stated mid-stream keeps the running decoder.
                        if self.mapping.as_ref() != Some(&mapping) || self.new_segment {
                            self.build_decoder(mapping)?;
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
                                    duration_ns: ticks_to_ns(samples, OPUS_RATE_HZ),
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
                            .saturating_add(ticks_to_ns(self.prev_samples, OPUS_RATE_HZ)),
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
                        self.build_default_decoder(*channels)?;
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
                    self.segment = seg;
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
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
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

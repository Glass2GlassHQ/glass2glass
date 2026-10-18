//! AAC access-unit parser that frames an ADTS stream and refines source-side
//! `Caps`.
//!
//! The audio sibling of `h264parse` / `h265parse`: it reads the ADTS header
//! (12-bit `0xFFF` syncword) for the channel count and sample rate and emits a
//! `CapsChanged` before the frames it describes. This lets a raw ADTS AAC
//! elementary stream be restreamed or muxed with concrete channel/rate caps.
//!
//! An ADTS stream is also split into one access unit per buffer, the way
//! `mpegaudioparse` splits MPEG audio: a bare `.aac` file arrives from `filesrc`
//! in arbitrary chunks, and `ffmpegaudiodec` takes one access unit per packet.
//! Partial frames are carried across input buffers, and a candidate sync is
//! trusted only when a valid header sits at its frame length (or the stream ends
//! there), which is also how the parser resynchronizes after garbage.
//! Presentation time runs off a sample counter at 1024 samples per access unit,
//! re-based whenever an upstream buffer carries a real time, so a demuxer's own
//! timestamps survive.
//!
//! `Caps::Audio` has no open (`Any`) field, so a source advertising AAC before
//! the first header lands uses sentinel `channels`/`sample_rate` 0; the
//! negotiated constraint is therefore `IdentityAny` (forward whatever AAC the
//! upstream produces) rather than the video parsers' `Identity(any geometry)`.
//! The AAC-only guard lives in `intercept_caps`.
//!
//! Both AAC framings are handled: ADTS (the common elementary-stream sync) and
//! LOAS/LATM (the MPEG-TS / broadcast `AudioSyncStream`), whose `StreamMuxConfig`
//! embeds an `AudioSpecificConfig` the parser reads for the channel count and
//! rate. The LATM path handles the common `audioMuxVersion == 0` layout and
//! bails safely (caps unrefined, never wrong) on the rare version-1 / config-reuse
//! variants. Neither framing needs exp-Golomb or emulation prevention, so this
//! shares none of the `annexb` machinery the H.264 / H.265 parsers use (just a
//! small local MSB-first bit reader for the LATM fields). A LATM stream is
//! forwarded buffer for buffer: its frames come from the broadcast container
//! already, not from a byte stream this has to cut up.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use crate::annexb::{BitReader, BitWriter};

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_warn, AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket,
};

use crate::audioframe::{locate_frame, Located, SyncFrameHeader};

/// Nanoseconds per second, the presentation-time unit.
const NS_PER_SECOND: u128 = 1_000_000_000;

/// Samples per channel one AAC access unit decodes to. An SBR / HE-AAC stream
/// doubles this against its ADTS-declared rate, which the header cannot tell us,
/// so the core-rate figure is used throughout.
const SAMPLES_PER_ACCESS_UNIT: u32 = 1024;

/// Bytes of ADTS header before the payload, without and with the CRC that
/// `protection_absent == 0` adds.
const ADTS_HEADER_LEN: usize = 7;
const ADTS_HEADER_LEN_CRC: usize = 9;

/// ADTS sampling-frequency-index table (ISO/IEC 14496-3). Indices 13/14 are
/// reserved and 15 (explicit rate) is forbidden in ADTS, so only 0..=12 map.
pub(crate) const SAMPLE_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// Synthesise the 2-byte AAC AudioSpecificConfig from an ADTS header.
pub(crate) fn asc_from_adts(au: &[u8]) -> Option<[u8; 2]> {
    if au.len() < ADTS_HEADER_LEN || au[0] != 0xFF || (au[1] & 0xF0) != 0xF0 {
        return None;
    }
    let object_type = ((au[2] >> 6) & 0x03) + 1; // profile + 1
    let sr_index = (au[2] >> 2) & 0x0F;
    let channel_config = ((au[2] & 0x01) << 2) | ((au[3] >> 6) & 0x03);
    Some([
        (object_type << 3) | (sr_index >> 1),
        ((sr_index & 1) << 7) | (channel_config << 3),
    ])
}

/// Strip the ADTS header (7 bytes, or 9 with CRC) from an AAC access unit.
pub(crate) fn strip_adts(au: &[u8]) -> &[u8] {
    if au.len() >= ADTS_HEADER_LEN && au[0] == 0xFF && (au[1] & 0xF0) == 0xF0 {
        au.get(adts_header_len(au[1])..).unwrap_or(&[])
    } else {
        au
    }
}

/// Build an ADTS-framed AAC access unit from the track's AudioSpecificConfig
/// and the raw access unit: a 7-byte ADTS header (no CRC) carrying the profile,
/// rate index and channel configuration, then the AU. A configuration of 0
/// puts the ASC's program config element in front of the AU, as ffmpeg's ADTS
/// muxer does. The inverse of the muxers' de-ADTS write, so the demuxed audio is
/// self-describing. `None` when ADTS cannot carry the configuration (an object
/// type past AAC LTP, an explicit rate, a 960-sample frame, malformed bits) or
/// the frame exceeds the 13-bit ADTS length. Shared by the MP4 and FLV
/// demuxers (M662).
pub(crate) fn adts_from_asc(asc: &[u8], au: &[u8]) -> Option<Vec<u8>> {
    let mut r = BitReader::new(asc);
    let mut aot = r.read_bits(5)?;
    let sr_index = r.read_bits(4)?;
    let channel_config = r.read_bits(4)?;
    if aot == AOT_SBR || aot == AOT_PS {
        // explicit SBR signalling: the core rate came first, then the SBR rate and the core type
        if r.read_bits(4)? == EXPLICIT_RATE_INDEX {
            r.read_bits(24)?;
        }
        aot = r.read_bits(5)?;
    }
    if !(AOT_AAC_MAIN..=AOT_AAC_LTP).contains(&aot) || sr_index as usize >= SAMPLE_RATES.len() {
        return None;
    }
    let mut payload = Vec::new();
    if channel_config == 0 {
        // GASpecificConfig: frameLengthFlag, dependsOnCoreCoder, extensionFlag
        if r.read_bits(1)? == 1 {
            return None; // 960-sample frames
        }
        if r.read_bits(1)? == 1 {
            r.read_bits(14)?;
        }
        if r.read_bits(1)? == 1 {
            return None;
        }
        let mut w = BitWriter::default();
        w.write_bits(ID_PCE, ELEMENT_ID_BITS);
        copy_program_config(&mut r, Some(&mut w))?;
        payload = w.into_bytes();
    }
    payload.extend_from_slice(au);
    adts_frame(
        (aot - 1) as u8,
        sr_index as u8,
        channel_config as u8,
        &payload,
    )
}

// a demuxer's AAC access unit as ADTS, failing loud on a config ADTS cannot frame
#[cfg(feature = "std")]
pub(crate) fn require_adts(
    asc: &[u8],
    au: &[u8],
    demuxer: &'static str,
) -> Result<Vec<u8>, G2gError> {
    adts_from_asc(asc, au).ok_or_else(|| {
        g2g_warn!(
            g2g_core::log::Target::category(demuxer),
            "AAC config {asc:02x?} cannot be ADTS framed"
        );
        G2gError::CapsMismatch
    })
}

// ADTS has no profile for ER AAC, ELD or USAC, so a demuxer sends their access units raw behind this config as a frame
pub(crate) fn is_raw_aac_config(config: &[u8]) -> bool {
    read_object_type(&mut BitReader::new(config))
        .is_some_and(|object_type| RAW_AAC_OBJECT_TYPES.contains(&object_type))
}

// longest AudioSpecificConfig a muxer takes from the frame leading a raw AAC stream
const MAX_RAW_AAC_CONFIG_LEN: usize = 256;

// an AAC stream's AudioSpecificConfig from its first frame: an ADTS header's, or the frame itself when it leads raw units
pub(crate) fn muxer_audio_specific_config(
    first_frame: &[u8],
    muxer: &'static str,
) -> Result<Option<Vec<u8>>, G2gError> {
    if let Some(asc) = asc_from_adts(first_frame) {
        return Ok(Some(asc.to_vec()));
    }
    if !is_raw_aac_config(first_frame) {
        return Ok(None);
    }
    if first_frame.len() > MAX_RAW_AAC_CONFIG_LEN {
        g2g_warn!(
            g2g_core::log::Target::category(muxer),
            "AAC config of {} bytes exceeds {MAX_RAW_AAC_CONFIG_LEN}",
            first_frame.len()
        );
        return Err(G2gError::CapsMismatch);
    }
    Ok(Some(first_frame.to_vec()))
}

// the config a muxer captured for an AAC pad, failing loud when its stream carried neither ADTS headers nor a leading config
#[cfg(feature = "std")]
pub(crate) fn require_muxer_config<'a>(
    config: Option<&'a [u8]>,
    muxer: &'static str,
) -> Result<&'a [u8], G2gError> {
    config.ok_or_else(|| {
        g2g_warn!(
            g2g_core::log::Target::category(muxer),
            "AAC stream has neither ADTS headers nor a leading AudioSpecificConfig"
        );
        G2gError::CapsMismatch
    })
}

// whether `frame` is the config frame leading the raw access units of an AAC stream declared by `config`
pub(crate) fn is_leading_aac_config(config: &[u8], frame: &[u8]) -> bool {
    is_raw_aac_config(config) && frame == config
}

// the access unit a muxer stores for one frame of an AAC stream declared by `config`
pub(crate) fn muxer_access_unit<'a>(config: &[u8], frame: &'a [u8]) -> &'a [u8] {
    if is_raw_aac_config(config) {
        frame
    } else {
        strip_adts(frame)
    }
}

fn read_object_type(r: &mut BitReader) -> Option<u32> {
    match r.read_bits(5)? {
        AOT_ESCAPE => Some(AOT_ESCAPE + 1 + r.read_bits(6)?),
        object_type => Some(object_type),
    }
}

// audio object types (ISO/IEC 14496-3 1.5.1.1)
const AOT_AAC_MAIN: u32 = 1;
const AOT_AAC_LTP: u32 = 4;
const AOT_SBR: u32 = 5;
const AOT_PS: u32 = 29;
const AOT_ESCAPE: u32 = 31;
// ER AAC LC, ER AAC LTP, ER AAC LD, ER AAC ELD, USAC
const RAW_AAC_OBJECT_TYPES: [u32; 5] = [17, 19, 23, 39, 42];
const EXPLICIT_RATE_INDEX: u32 = 15;
// raw_data_block element id of a program_config_element
const ID_PCE: u32 = 5;
const ELEMENT_ID_BITS: u32 = 3;

// a front / side / back element: is_cpe, then the tag
const CHANNEL_ELEMENT_BITS: u32 = 5;
const LFE_ELEMENT_BITS: u32 = 4;
const ASSOC_DATA_ELEMENT_BITS: u32 = 4;
const CC_ELEMENT_BITS: u32 = 5;

// copies a program_config_element to `w`, each side byte aligned from its own start, and counts its channels
fn copy_program_config(r: &mut BitReader, mut w: Option<&mut BitWriter>) -> Option<u8> {
    let w = &mut w;
    copy_bits(r, w, 4)?; // element_instance_tag
    copy_bits(r, w, 2)?; // object_type
    copy_bits(r, w, 4)?; // sampling_frequency_index
    let front = copy_bits(r, w, 4)?;
    let side = copy_bits(r, w, 4)?;
    let back = copy_bits(r, w, 4)?;
    let lfe = copy_bits(r, w, 2)?;
    let assoc_data = copy_bits(r, w, 3)?;
    let cc = copy_bits(r, w, 4)?;
    // mono_mixdown, stereo_mixdown, matrix_mixdown: a present flag, then the field
    for mixdown_bits in [4, 4, 3] {
        if copy_bits(r, w, 1)? == 1 {
            copy_bits(r, w, mixdown_bits)?;
        }
    }
    let mut channels = lfe;
    for _ in 0..front + side + back {
        let is_channel_pair = copy_bits(r, w, CHANNEL_ELEMENT_BITS)? >> 4 == 1;
        channels += if is_channel_pair { 2 } else { 1 };
    }
    for (count, bits) in [
        (lfe, LFE_ELEMENT_BITS),
        (assoc_data, ASSOC_DATA_ELEMENT_BITS),
        (cc, CC_ELEMENT_BITS),
    ] {
        for _ in 0..count {
            copy_bits(r, w, bits)?;
        }
    }
    while !r.bit_pos().is_multiple_of(8) {
        r.read_bit()?;
    }
    if let Some(w) = w.as_deref_mut() {
        w.align_to_byte();
    }
    let comment_bytes = copy_bits(r, w, 8)?;
    for _ in 0..comment_bytes {
        copy_bits(r, w, 8)?;
    }
    u8::try_from(channels).ok().filter(|&channels| channels > 0)
}

fn copy_bits(r: &mut BitReader, w: &mut Option<&mut BitWriter>, bits: u32) -> Option<u32> {
    let value = r.read_bits(bits)?;
    if let Some(w) = w.as_deref_mut() {
        w.write_bits(value, bits);
    }
    Some(value)
}

/// Frame `au` behind a 7-byte ADTS header (no CRC) for `profile` (audio object
/// type minus one), sampling-frequency index `sr_index` and `channel_config`.
/// `None` when the frame exceeds the 13-bit ADTS frame length.
pub(crate) fn adts_frame(
    profile: u8,
    sr_index: u8,
    channel_config: u8,
    au: &[u8],
) -> Option<Vec<u8>> {
    let frame_len = au.len() + ADTS_HEADER_LEN;
    if frame_len > 0x1FFF {
        return None; // ADTS frame_length is 13 bits
    }
    let mut out = Vec::with_capacity(frame_len);
    out.extend_from_slice(&[
        0xFF,
        0xF1, // syncword | MPEG-4 | layer 0 | protection_absent (no CRC)
        (profile << 6) | (sr_index << 2) | ((channel_config >> 2) & 1),
        ((channel_config & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F, // buffer fullness (top bits)
        0xFC,                                  // buffer fullness (low) | num_raw_data_blocks = 0
    ]);
    out.extend_from_slice(au);
    Some(out)
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::aacparse::AacParse;
///
/// let parse = AacParse::new();
/// assert_eq!(parse.caps_changes_emitted(), 0);
/// ```
#[derive(Debug, Default)]
pub struct AacParse {
    configured: bool,
    last_emitted_caps: Option<Caps>,
    headers_emitted: u64,
    framing: Framing,
    /// Unconsumed ADTS bytes, starting at stream offset `buf_offset`.
    buf: Vec<u8>,
    buf_offset: u64,
    /// Samples emitted since the last time base, the presentation-time counter.
    samples: u64,
    base_ns: u64,
    /// A presentation time from upstream and the stream offset its buffer began
    /// at, applied to the first access unit that starts at or past it.
    pending_rebase: Option<(u64, u64)>,
    sequence: u64,
    log_name: LogName,
}

/// Which framing the stream turned out to carry. Decided from the first buffer
/// that parses as one, and held for the rest of the stream.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Framing {
    #[default]
    Undecided,
    /// ADTS: a byte stream this splits into one access unit per buffer.
    Adts,
    /// LOAS/LATM: already framed by the broadcast container, forwarded as is.
    Latm,
    // raw access units behind a leading AudioSpecificConfig frame, forwarded as is
    Raw,
}

impl AacParse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count of `CapsChanged` packets pushed downstream, for tests asserting
    /// re-emission is suppressed when the ADTS parameters are unchanged.
    pub fn caps_changes_emitted(&self) -> u64 {
        self.headers_emitted
    }

    /// Count of access units emitted, once the stream framed as ADTS.
    pub fn frames_emitted(&self) -> u64 {
        self.sequence
    }

    /// Announce `caps` if they differ from what downstream was last told.
    async fn announce(&mut self, caps: Caps, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if self.last_emitted_caps.as_ref() == Some(&caps) {
            return Ok(());
        }
        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
        self.last_emitted_caps = Some(caps);
        self.headers_emitted += 1;
        Ok(())
    }

    /// Emit every access unit the buffer holds whole.
    async fn drain(&mut self, eos: bool, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        loop {
            let Located::Frame { start, len } = locate_frame::<AdtsHeader>(&self.buf, eos) else {
                return Ok(());
            };
            if start > 0 {
                g2g_warn!(self, "resynchronized past {start} bytes of non-ADTS");
                self.buf.drain(..start);
                self.buf_offset += start as u64;
            }
            let header = adts_header(&self.buf).ok_or(G2gError::CapsMismatch)?;
            let start_offset = self.buf_offset;
            let data: Vec<u8> = self.buf.drain(..len).collect();
            self.buf_offset += len as u64;
            self.emit(data, &header, start_offset, out).await?;
        }
    }

    /// Push one access unit with the caps its header declares and a presentation
    /// time from the running sample count.
    async fn emit(
        &mut self,
        data: Vec<u8>,
        header: &AdtsHeader,
        start_offset: u64,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        self.announce(header.info().caps(), out).await?;
        if let Some((at, pts_ns)) = self.pending_rebase {
            if start_offset >= at {
                self.base_ns = pts_ns;
                self.samples = 0;
                self.pending_rebase = None;
            }
        }
        let rate = u128::from(header.sample_rate);
        let ns = |samples: u64| (u128::from(samples) * NS_PER_SECOND / rate) as u64;
        let pts_ns = self.base_ns + ns(self.samples);
        let duration_ns = ns(u64::from(SAMPLES_PER_ACCESS_UNIT));
        self.samples += u64::from(SAMPLES_PER_ACCESS_UNIT);
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns,
                // Every AAC access unit decodes on its own.
                keyframe: true,
                ..FrameTiming::default()
            },
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

impl AsyncElement for AacParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Pass-through identity over AAC of any channel/rate (the parser refines
    /// those mid-stream from the ADTS header but never changes media type).
    /// `IdentityAny`, not `Identity(set)`, because audio caps cannot express
    /// "AAC at any channels/rate" in a single `Caps`.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            } => {
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AAC parser",
            "Codec/Parser/Audio",
            "Parses an AAC ADTS or LOAS/LATM stream and refines caps",
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
                    let bytes = frame.domain.as_system_slice().unwrap_or(&[]);
                    if self.framing == Framing::Undecided && !bytes.is_empty() {
                        self.framing = if parse_adts(bytes).is_some() {
                            Framing::Adts
                        } else if is_raw_aac_config(bytes) {
                            Framing::Raw
                        } else {
                            Framing::Latm
                        };
                    }
                    match self.framing {
                        Framing::Adts => {
                            // A byte source stamps every chunk 0, so only a real
                            // time (a demuxer's) re-bases the counter.
                            if let Some(pts) = frame.timing.pts().filter(|pts| *pts != 0) {
                                self.pending_rebase =
                                    Some((self.buf_offset + self.buf.len() as u64, pts));
                            }
                            self.buf.extend_from_slice(bytes);
                            self.drain(false, out).await?;
                        }
                        // raw access unit bytes can alias an ADTS or LOAS sync, so they are not scanned
                        Framing::Raw => {
                            out.push(PipelinePacket::DataFrame(frame)).await?;
                        }
                        Framing::Undecided | Framing::Latm => {
                            if let Some(info) = parse_aac(bytes) {
                                self.announce(info.caps(), out).await?;
                            }
                            out.push(PipelinePacket::DataFrame(frame)).await?;
                        }
                    }
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    self.last_emitted_caps = None;
                    self.buf.clear();
                    self.pending_rebase = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // The last access unit ends at the end of the stream, so it has
                // no successor to confirm its sync against.
                PipelinePacket::Eos => {
                    if self.framing == Framing::Adts {
                        self.drain(true, out).await?;
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    /// Reads host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }
}

impl LogSource for AacParse {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

impl PadTemplates for AacParse {
    fn pad_templates() -> Vec<PadTemplate> {
        // `Caps::Audio` has no open dims; pin the common stereo/48 kHz shape.
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(aac.clone())),
            PadTemplate::source(CapsSet::one(aac)),
        ])
    }
}

/// Channel count and sample rate recovered from an AAC bitstream header (ADTS or
/// LOAS/LATM).
pub(crate) struct AacInfo {
    pub(crate) channels: u8,
    pub(crate) sample_rate: u32,
}

impl AacInfo {
    fn caps(&self) -> Caps {
        Caps::Audio {
            format: AudioFormat::Aac,
            channels: self.channels,
            sample_rate: self.sample_rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }
}

/// The decoded fields of one ADTS header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AdtsHeader {
    sample_rate: u32,
    channels: u8,
    /// Bytes this access unit occupies, header and any CRC included.
    pub(crate) frame_len: usize,
    /// Bytes of header ahead of the payload: 7, or 9 with a CRC.
    pub(crate) header_len: usize,
    profile: u8,
    freq_index: u8,
    channel_config: u8,
}

impl AdtsHeader {
    fn info(&self) -> AacInfo {
        AacInfo {
            channels: self.channels,
            sample_rate: self.sample_rate,
        }
    }

    /// Whether two headers declare the same coding: the profile, sampling
    /// frequency index and channel configuration an ADTS stream repeats in every
    /// header. The content sniffer wants this stricter agreement before it types
    /// a file off two headers alone.
    pub(crate) fn same_configuration(&self, other: &Self) -> bool {
        self.profile == other.profile
            && self.freq_index == other.freq_index
            && self.channel_config == other.channel_config
    }
}

impl SyncFrameHeader for AdtsHeader {
    const HEADER_LEN: usize = ADTS_HEADER_LEN;

    fn parse(buf: &[u8]) -> Option<Self> {
        adts_header(buf)
    }

    fn coded_bytes(&self) -> usize {
        self.frame_len
    }

    /// A parsed ADTS header at the frame length is confirmation enough: it is 12
    /// sync bits plus a rate index, a channel configuration and a length that
    /// lands exactly here. Demanding the same coding as well would drop the
    /// frame ahead of a mid-stream parameter change, which this parser reports
    /// through `CapsChanged` instead.
    fn confirms_sync(&self, _next: &Self) -> bool {
        true
    }
}

// the channel count of the program config element a raw data block opens with
fn pce_channels(payload: &[u8]) -> Option<u8> {
    let mut r = BitReader::new(payload);
    if r.read_bits(ELEMENT_ID_BITS)? != ID_PCE {
        return None;
    }
    // the PCE's byte alignment counts from the raw data block start, the reader's start
    copy_program_config(&mut r, None)
}

/// Bytes of ADTS header ahead of the payload, from byte 1 of the header:
/// `protection_absent == 0` puts a 2-byte CRC behind the fixed fields.
fn adts_header_len(byte1: u8) -> usize {
    if byte1 & 0x01 == 0 {
        ADTS_HEADER_LEN_CRC
    } else {
        ADTS_HEADER_LEN
    }
}

/// Decode the ADTS header at the start of `buf`, or `None` when `buf` does not
/// open on a usable one: too short, no `0xFFF` syncword with layer 00, a
/// reserved sampling-frequency index, a channel configuration that does not pin
/// a channel count (0 without a leading program config element), or an
/// `aac_frame_length` shorter than its own header.
pub(crate) fn adts_header(buf: &[u8]) -> Option<AdtsHeader> {
    let head = buf.get(..ADTS_HEADER_LEN)?;
    // Syncword 0xFFF (12 bits) + layer 00: byte0 all ones, byte1 high nibble all
    // ones and the two layer bits zero.
    if head[0] != 0xFF || (head[1] & 0xF6) != 0xF0 {
        return None;
    }
    let profile = (head[2] >> 6) & 0x03;
    let freq_index = (head[2] >> 2) & 0x0F;
    let channel_config = ((head[2] & 0x01) << 2) | (head[3] >> 6);
    let &sample_rate = SAMPLE_RATES.get(freq_index as usize)?;
    let header_len = adts_header_len(head[1]);
    let channels = match channel_config {
        1..=6 => channel_config, // 1ch..5.1
        7 => 8,                  // 7.1
        // 0: a program config element opens the payload, or the layout is unknown
        _ => pce_channels(buf.get(header_len..)?)?,
    };
    let frame_len =
        (((head[3] & 0x03) as usize) << 11) | ((head[4] as usize) << 3) | ((head[5] >> 5) as usize);
    (frame_len >= header_len).then_some(AdtsHeader {
        sample_rate,
        channels,
        frame_len,
        header_len,
        profile,
        freq_index,
        channel_config,
    })
}

/// Recover the channel count and sample rate from either AAC framing: ADTS (the
/// elementary-stream sync) first, then LOAS/LATM (the MPEG-TS / broadcast sync).
fn parse_aac(au: &[u8]) -> Option<AacInfo> {
    parse_adts(au).or_else(|| parse_loas(au))
}

/// Scan `au` for the first valid ADTS header and take its channel count and
/// sample rate. `None` if no header parses.
fn parse_adts(au: &[u8]) -> Option<AacInfo> {
    let last = au.len().checked_sub(ADTS_HEADER_LEN)?;
    (0..=last).find_map(|i| adts_header(&au[i..]).map(|h| h.info()))
}

/// Scan `au` for the first LOAS `AudioSyncStream` frame (syncword 0x2B7, 11 bits)
/// and recover the AAC channel count and sample rate from the LATM
/// `StreamMuxConfig`'s embedded `AudioSpecificConfig`. `None` if no frame parses.
fn parse_loas(au: &[u8]) -> Option<AacInfo> {
    let last = au.len().checked_sub(3)?;
    for i in 0..=last {
        // Syncword 0x2B7 byte-aligned: byte0 = 0x56, top 3 bits of byte1 = 111.
        if au[i] != 0x56 || (au[i + 1] & 0xE0) != 0xE0 {
            continue;
        }
        // audioMuxLengthBytes: the 13 bits after the 11-bit syncword.
        let mux_len = (((au[i + 1] & 0x1F) as usize) << 8) | au[i + 2] as usize;
        let Some(payload) = au.get(i + 3..).and_then(|p| p.get(..mux_len)) else {
            continue;
        };
        if let Some(info) = parse_audio_mux_element(payload) {
            return Some(info);
        }
    }
    None
}

/// Parse a LATM `AudioMuxElement` (muxConfigPresent = 1, the LOAS case) far
/// enough to reach the `AudioSpecificConfig`. Handles the common
/// `audioMuxVersion == 0` layout; the rarer version-1 (variable-length values)
/// and stream-mux-config reuse yield `None` (caps stay unrefined, never wrong).
fn parse_audio_mux_element(data: &[u8]) -> Option<AacInfo> {
    let mut r = BitReader::new(data);
    if r.read_bits(1)? == 1 {
        return None; // useSameStreamMux: reuses a prior config we did not retain
    }
    // StreamMuxConfig
    if r.read_bits(1)? != 0 {
        return None; // audioMuxVersion 1 (LatmGetValue lengths) not handled
    }
    r.read_bits(1)?; // allStreamsSameTimeFraming
    r.read_bits(6)?; // numSubFrames
    r.read_bits(4)?; // numProgram (only program 0 is parsed)
    r.read_bits(3)?; // numLayer of program 0 (only layer 0 is parsed)
                     // program 0, layer 0: useSameConfig is implied 0, so AudioSpecificConfig follows.
    parse_audio_specific_config(&mut r)
}

pub(crate) fn audio_specific_config_info(config: &[u8]) -> Option<AacInfo> {
    parse_audio_specific_config(&mut BitReader::new(config))
}

/// Read the leading fields of an `AudioSpecificConfig` (ISO/IEC 14496-3): the
/// audio object type, sampling frequency (index or explicit), and channel
/// configuration, enough to pin the channel count and sample rate.
fn parse_audio_specific_config(r: &mut BitReader) -> Option<AacInfo> {
    // the object type does not affect the channel count / rate
    read_object_type(r)?;
    let sr_index = r.read_bits(4)? as usize;
    let sample_rate = if sr_index == 0x0F {
        r.read_bits(24)? // explicit 24-bit sampling frequency
    } else {
        *SAMPLE_RATES.get(sr_index)?
    };
    let channel_config = r.read_bits(4)?;
    let channels = match channel_config {
        1..=6 => channel_config as u8,
        7 => 8,
        _ => return None, // 0 = further config; reserved otherwise
    };
    Some(AacInfo {
        channels,
        sample_rate,
    })
}

/// Fuzzing entry: parse an AAC access unit (ADTS / LOAS-LATM dispatch, then the
/// AudioSpecificConfig bit reader). Exposed only under `--cfg fuzzing`
/// (cargo-fuzz) so the normal public API is unchanged.
#[cfg(fuzzing)]
pub fn fuzz_parse(data: &[u8]) {
    let _ = parse_aac(data);
}

/// Synthetic ADTS access units, shared by the tests of the parser and of the
/// content sniffer (both need a stream that frames the way a real one does).
#[cfg(test)]
pub(crate) mod test_frames {
    use alloc::vec;
    use alloc::vec::Vec;

    /// Build a 7-byte (no-CRC) ADTS header for `channel_config` at
    /// `freq_index`, followed by `payload_len` bytes of `fill`, framed as one
    /// AAC-LC access unit.
    pub(crate) fn adts_frame(
        channel_config: u8,
        freq_index: u8,
        payload_len: usize,
        fill: u8,
    ) -> Vec<u8> {
        /// AAC-LC is AOT 2, and the ADTS profile field is AOT - 1.
        const PROFILE_AAC_LC: u8 = 1;
        let frame_len = super::ADTS_HEADER_LEN + payload_len;
        let mut f = vec![fill; frame_len];
        f[0] = 0xFF;
        f[1] = 0xF1; // syncword low, MPEG-4, layer 00, protection_absent = 1
        f[2] = (PROFILE_AAC_LC << 6) | ((freq_index & 0x0F) << 2) | ((channel_config >> 2) & 0x01);
        f[3] = ((channel_config & 0x03) << 6) | (((frame_len >> 11) & 0x03) as u8);
        f[4] = ((frame_len >> 3) & 0xFF) as u8;
        f[5] = (((frame_len & 0x07) << 5) as u8) | 0x1F;
        f[6] = 0xFC; // buffer fullness low + num_raw_blocks (0)
        f
    }

    /// One 48 kHz stereo AAC-LC access unit carrying `payload`.
    #[cfg(feature = "std")]
    pub(crate) fn adts_au(payload: &[u8]) -> Vec<u8> {
        const STEREO: u8 = 2;
        const FREQ_INDEX_48000: u8 = 3;
        let mut au = adts_frame(STEREO, FREQ_INDEX_48000, payload.len(), 0);
        au[super::ADTS_HEADER_LEN..].copy_from_slice(payload);
        au
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{frame_with_bytes, CollectSink};
    use alloc::vec;

    use test_frames::adts_frame as build_adts;

    /// One access unit with a zero payload, the shape most tests here want.
    fn adts_frame(channel_config: u8, freq_index: u8, payload_len: usize) -> Vec<u8> {
        build_adts(channel_config, freq_index, payload_len, 0)
    }

    #[test]
    fn recovers_stereo_44100() {
        let info = parse_adts(&adts_frame(2, 4, 16)).expect("ADTS must parse");
        assert_eq!((info.channels, info.sample_rate), (2, 44_100));
    }

    #[test]
    fn recovers_mono_48000() {
        let info = parse_adts(&adts_frame(1, 3, 8)).expect("ADTS must parse");
        assert_eq!((info.channels, info.sample_rate), (1, 48_000));
    }

    #[test]
    fn maps_channel_config_7_to_eight_channels() {
        let info = parse_adts(&adts_frame(7, 3, 8)).expect("ADTS must parse");
        assert_eq!(info.channels, 8);
    }

    #[test]
    fn rejects_reserved_sampling_index() {
        // freq_index 13 is reserved; no valid rate, so the header is skipped.
        assert!(parse_adts(&adts_frame(2, 13, 16)).is_none());
    }

    #[test]
    fn adts_carries_the_program_config_of_a_channel_config_zero_asc() {
        const AOT_AAC_LC: u32 = 2;
        const RATE_INDEX_44100: u32 = 4;
        let mut asc = BitWriter::default();
        asc.write_bits(AOT_AAC_LC, 5);
        asc.write_bits(RATE_INDEX_44100, 4);
        asc.write_bits(0, 4); // channel configuration in the PCE
        asc.write_bits(0, 3); // frameLengthFlag, dependsOnCoreCoder, extensionFlag
        asc.write_bits(0, 4); // element_instance_tag
        asc.write_bits(AOT_AAC_LC - 1, 2);
        asc.write_bits(RATE_INDEX_44100, 4);
        // one front channel pair, one LFE: three channels
        for (count, bits) in [(1, 4), (0, 4), (0, 4), (1, 2), (0, 3), (0, 4)] {
            asc.write_bits(count, bits);
        }
        asc.write_bits(0, 3); // no mixdowns
        asc.write_bits(0b1_0000, CHANNEL_ELEMENT_BITS); // is_cpe
        asc.write_bits(0, LFE_ELEMENT_BITS);
        asc.align_to_byte();
        asc.write_bits(0, 8); // no comment
        let au = [0xAA, 0xBB];
        let frame = adts_from_asc(&asc.into_bytes(), &au).expect("ADTS with a PCE");
        let header = adts_header(&frame).expect("the header parses");
        assert_eq!((header.channel_config, header.channels), (0, 3));
        assert_eq!(header.sample_rate, SAMPLE_RATES[RATE_INDEX_44100 as usize]);
        assert_eq!(header.frame_len, frame.len());
        assert!(frame.ends_with(&au));
    }

    #[test]
    fn adts_refuses_configs_it_cannot_carry() {
        // AAC-ELD (escaped object type 39), 48 kHz stereo
        const ELD_ASC: [u8; 4] = [0xF8, 0xE6, 0x40, 0x00];
        // AAC LC 48 kHz, frameLengthFlag set (960-sample frames), PCE follows
        const SHORT_FRAME_ASC: [u8; 2] = [0x11, 0x84];
        assert!(adts_from_asc(&ELD_ASC, &[0xAA]).is_none());
        assert!(adts_from_asc(&SHORT_FRAME_ASC, &[0xAA]).is_none());
        #[cfg(feature = "std")]
        assert!(require_adts(&ELD_ASC, &[0xAA], "test").is_err());
    }

    #[test]
    fn adts_takes_the_core_profile_of_explicit_sbr() {
        // AOT 5 (SBR), core 24 kHz index 6, stereo, SBR rate index 3, core AOT 2
        let mut asc = BitWriter::default();
        for (value, bits) in [(5, 5), (6, 4), (2, 4), (3, 4), (2, 5)] {
            asc.write_bits(value, bits);
        }
        let frame = adts_from_asc(&asc.into_bytes(), &[0xAA]).expect("ADTS");
        let header = adts_header(&frame).expect("the header parses");
        const PROFILE_AAC_LC: u8 = 1;
        assert_eq!(header.profile, PROFILE_AAC_LC);
        assert_eq!(header.freq_index, 6);
        assert_eq!(header.channels, 2);
    }

    #[test]
    fn rejects_channel_config_zero() {
        // config 0 means the channel count lives in the AOT config, not ADTS.
        assert!(parse_adts(&adts_frame(0, 4, 16)).is_none());
    }

    #[test]
    fn finds_header_after_leading_bytes() {
        let mut stream = vec![0x00, 0x11, 0x22];
        stream.extend_from_slice(&adts_frame(2, 4, 16));
        let info = parse_adts(&stream).expect("ADTS after junk must parse");
        assert_eq!((info.channels, info.sample_rate), (2, 44_100));
    }

    #[test]
    fn returns_none_on_non_adts() {
        assert!(parse_adts(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00]).is_none());
        assert!(parse_adts(&[]).is_none());
    }

    // -- LOAS / LATM (broadcast framing) -----------------------------------

    /// Pack the LATM `AudioMuxElement` bits (audioMuxVersion 0) up to and
    /// including the `AudioSpecificConfig` channel configuration.
    fn latm_mux_element(aot: u32, sr_index: u32, channel_config: u32) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bits(0, 1); // useSameStreamMux = 0
        w.write_bits(0, 1); // audioMuxVersion = 0
        w.write_bits(1, 1); // allStreamsSameTimeFraming
        w.write_bits(0, 6); // numSubFrames
        w.write_bits(0, 4); // numProgram (program 0 only)
        w.write_bits(0, 3); // numLayer (layer 0 only)
        w.write_bits(aot, 5); // AudioSpecificConfig: audio object type
        w.write_bits(sr_index, 4); // sampling frequency index
        w.write_bits(channel_config, 4); // channel configuration
        w.into_bytes()
    }

    /// Wrap a LATM `AudioMuxElement` payload in a LOAS `AudioSyncStream` frame
    /// (11-bit syncword 0x2B7 + 13-bit length).
    fn loas_frame(payload: &[u8]) -> Vec<u8> {
        let mux_len = payload.len();
        let mut f = vec![
            0x56,
            0xE0 | ((mux_len >> 8) as u8 & 0x1F),
            (mux_len & 0xFF) as u8,
        ];
        f.extend_from_slice(payload);
        f
    }

    #[test]
    fn recovers_loas_latm_stereo_44100() {
        // AAC-LC (AOT 2), sr index 4 (44100), channel config 2 (stereo).
        let frame = loas_frame(&latm_mux_element(2, 4, 2));
        let info = parse_aac(&frame).expect("LOAS/LATM must parse");
        assert_eq!((info.channels, info.sample_rate), (2, 44_100));
    }

    #[test]
    fn recovers_loas_latm_5_1() {
        // channel config 6 = 5.1.
        let info = parse_aac(&loas_frame(&latm_mux_element(2, 3, 6))).expect("LATM 5.1 parses");
        assert_eq!((info.channels, info.sample_rate), (6, 48_000));
    }

    #[test]
    fn loas_found_after_leading_bytes() {
        let mut stream = vec![0x00, 0x11];
        stream.extend_from_slice(&loas_frame(&latm_mux_element(2, 4, 1)));
        let info = parse_aac(&stream).expect("LOAS after junk parses");
        assert_eq!((info.channels, info.sample_rate), (1, 44_100));
    }

    #[test]
    fn latm_audiomux_version_1_bails_safely() {
        // Flip audioMuxVersion to 1: unsupported, so no (wrong) caps are produced.
        let mut payload = latm_mux_element(2, 4, 2);
        payload[0] |= 0b0100_0000; // set bit 1 (audioMuxVersion) after useSameStreamMux
        assert!(parse_audio_mux_element(&payload).is_none());
    }

    #[test]
    fn parse_aac_accepts_both_framings() {
        assert_eq!(
            parse_aac(&adts_frame(2, 4, 16)).map(|i| i.channels),
            Some(2)
        );
        assert_eq!(
            parse_aac(&loas_frame(&latm_mux_element(2, 4, 2))).map(|i| i.channels),
            Some(2)
        );
    }

    // -- Element-level tests (drive AacParse::process directly) -------------

    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{FrameTiming, MemoryDomain};

    fn aac_caps() -> Caps {
        // Sentinel pre-parse caps: format pinned, channels/rate unknown.
        Caps::Audio {
            format: AudioFormat::Aac,
            channels: 0,
            sample_rate: 0,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    /// Push every buffer in `buffers`, then end the stream.
    async fn run(buffers: &[Vec<u8>]) -> (AacParse, CollectSink) {
        let mut parse = AacParse::new();
        parse.configure_pipeline(&aac_caps()).unwrap();
        let mut sink = CollectSink::default();
        for (seq, bytes) in buffers.iter().enumerate() {
            parse
                .process(
                    PipelinePacket::DataFrame(frame_with_bytes(seq as u64, bytes.clone())),
                    &mut sink,
                )
                .await
                .unwrap();
        }
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        (parse, sink)
    }

    /// The (channels, rate) of every `CapsChanged` pushed, in order.
    fn caps_params(sink: &CollectSink) -> Vec<(u8, u32)> {
        sink.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(Caps::Audio {
                    channels,
                    sample_rate,
                    ..
                }) => Some((*channels, *sample_rate)),
                _ => None,
            })
            .collect()
    }

    fn data_frames(sink: &CollectSink) -> Vec<&Frame> {
        sink.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let (parse, sink) = run(&[adts_frame(2, 4, 16)]).await;

        assert_eq!(sink.packets.len(), 2, "expected CapsChanged then DataFrame");
        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::Audio {
                format: AudioFormat::Aac,
                channels,
                sample_rate,
                ..
            }) => {
                assert_eq!(*channels, 2);
                assert_eq!(*sample_rate, 44_100);
            }
            other => panic!("expected AAC CapsChanged first, got {other:?}"),
        }
        assert!(matches!(sink.packets[1], PipelinePacket::DataFrame(_)));
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn does_not_re_emit_caps_when_unchanged() {
        let unit = adts_frame(2, 4, 16);
        let (parse, sink) = run(&[unit.clone(), unit.clone(), unit]).await;

        assert_eq!(
            caps_params(&sink),
            vec![(2, 44_100)],
            "CapsChanged fires once for identical ADTS params"
        );
        assert_eq!(parse.caps_changes_emitted(), 1);
        assert_eq!(parse.frames_emitted(), 3);
    }

    #[tokio::test]
    async fn re_emits_caps_on_parameter_change() {
        // stereo/44100 then mono/48000.
        let (parse, sink) = run(&[adts_frame(2, 4, 16), adts_frame(1, 3, 8)]).await;

        assert_eq!(caps_params(&sink), vec![(2, 44_100), (1, 48_000)]);
        assert_eq!(parse.caps_changes_emitted(), 2);
    }

    #[tokio::test]
    async fn splits_a_byte_stream_into_access_units() {
        // An odd chunk size, so access units straddle buffer boundaries.
        const CHUNK_LEN: usize = 13;
        const PAYLOAD_LEN: usize = 24;
        let mut stream = Vec::new();
        for fill in 0..5u8 {
            stream.extend(build_adts(2, 4, PAYLOAD_LEN, fill));
        }
        let buffers: Vec<Vec<u8>> = stream.chunks(CHUNK_LEN).map(<[u8]>::to_vec).collect();
        let (parse, sink) = run(&buffers).await;

        assert_eq!(parse.frames_emitted(), 5);
        let frames = data_frames(&sink);
        assert!(frames
            .iter()
            .all(|f| f.domain.as_system_slice().map(<[u8]>::len)
                == Some(ADTS_HEADER_LEN + PAYLOAD_LEN)));
        // Presentation time comes off the running sample count at 1024 samples
        // an access unit, not a sum of rounded durations.
        const FRAME_RATE_HZ: u64 = 44_100;
        let pts = |units: u64| {
            units * u64::from(SAMPLES_PER_ACCESS_UNIT) * NS_PER_SECOND as u64 / FRAME_RATE_HZ
        };
        let times: Vec<u64> = frames.iter().map(|f| f.timing.pts_ns).collect();
        assert_eq!(times, (0..5).map(pts).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn resynchronizes_past_leading_garbage() {
        let mut stream = vec![0u8; 20];
        // A lone sync byte pair that no valid header follows: skipped too.
        stream.extend_from_slice(&[0xFF, 0xF1, 0x00, 0x00, 0x00, 0x00, 0x00]);
        for fill in 0..3u8 {
            stream.extend(build_adts(2, 4, 16, fill));
        }
        let (parse, _) = run(&[stream]).await;
        assert_eq!(parse.frames_emitted(), 3, "every unit behind the garbage");
    }

    #[tokio::test]
    async fn drops_a_truncated_tail() {
        let mut stream = build_adts(2, 4, 16, 0);
        stream.extend(build_adts(2, 4, 16, 1));
        stream.extend_from_slice(&build_adts(2, 4, 16, 2)[..10]);
        let (parse, _) = run(&[stream]).await;
        assert_eq!(parse.frames_emitted(), 2, "the half unit is not emitted");
    }

    #[tokio::test]
    async fn re_bases_on_an_upstream_presentation_time() {
        const DEMUXER_PTS_NS: u64 = 3_000_000_000;
        let mut parse = AacParse::new();
        parse.configure_pipeline(&aac_caps()).unwrap();
        let mut sink = CollectSink::default();
        let mut stream = adts_frame(2, 4, 16);
        stream.extend(adts_frame(2, 4, 16));
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
            FrameTiming {
                pts_ns: DEMUXER_PTS_NS,
                ..FrameTiming::default()
            },
            0,
        );
        parse
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        assert_eq!(data_frames(&sink)[0].timing.pts_ns, DEMUXER_PTS_NS);
    }

    #[tokio::test]
    async fn loas_latm_drives_caps_through_process() {
        let mut parse = AacParse::new();
        parse.configure_pipeline(&aac_caps()).unwrap();
        let mut sink = CollectSink::default();

        let frame = frame_with_bytes(0, loas_frame(&latm_mux_element(2, 3, 6)));
        parse
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        assert_eq!(
            sink.packets.len(),
            2,
            "CapsChanged then the forwarded frame"
        );
        assert!(matches!(
            sink.packets[0],
            PipelinePacket::CapsChanged(Caps::Audio {
                channels: 6,
                sample_rate: 48_000,
                ..
            })
        ));
        assert!(matches!(sink.packets[1], PipelinePacket::DataFrame(_)));
    }

    #[tokio::test]
    async fn rejects_non_aac_caps_in_intercept() {
        let parse = AacParse::new();
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(parse.intercept_caps(&pcm), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_constraint_is_identity_any() {
        let parse = AacParse::new();
        assert!(matches!(
            parse.caps_constraint_as_transform(),
            CapsConstraint::IdentityAny
        ));
    }
}

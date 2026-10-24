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
//! rate. The LATM path handles `audioMuxVersion` 0 and 1 and bails safely
//! (caps unrefined, never wrong) on the config-reuse variants. Neither framing
//! needs exp-Golomb or emulation prevention, so this shares none of the
//! `annexb` machinery the H.264 / H.265 parsers use (just a small local
//! MSB-first bit reader for the LATM fields). A LATM stream is
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
    g2g_error, g2g_warn, AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket,
};

use crate::audioframe::{locate_frame, Located, SyncFrameHeader};
use crate::frame_clock::FrameClock;
use crate::id3::{parse_id3v1, ID3V1_LEN};
use g2g_core::time::ticks_to_ns;

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

const ADTS_AUDIO_SPECIFIC_CONFIG_LEN: usize = 2;

/// Synthesise the 2-byte AAC AudioSpecificConfig from an ADTS header.
pub(crate) fn asc_from_adts(au: &[u8]) -> Option<[u8; ADTS_AUDIO_SPECIFIC_CONFIG_LEN]> {
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
/// type past AAC LTP, an explicit rate, a 960-sample frame, an SBR output rate
/// other than double the core rate, malformed bits) or the frame exceeds the
/// 13-bit ADTS length. Shared by the MP4 and FLV demuxers (M662).
pub(crate) fn adts_from_asc(asc: &[u8], au: &[u8]) -> Option<Vec<u8>> {
    if has_undoubled_sbr(asc) {
        return None;
    }
    let mut r = BitReader::new(asc);
    let core = read_adts_core_config(&mut r)?;
    // GASpecificConfig: frameLengthFlag, dependsOnCoreCoder, extensionFlag
    if r.read_bits(1)? == 1 {
        return None; // 960-sample frames
    }
    let mut payload = Vec::new();
    if core.channel_config == 0 {
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
        (core.object_type - 1) as u8,
        core.sr_index as u8,
        core.channel_config as u8,
        &payload,
    )
}

// the AudioSpecificConfig fields an ADTS header repeats
struct AdtsCoreConfig {
    object_type: u32,
    sr_index: u32,
    channel_config: u32,
}

// reads up to the GASpecificConfig, `None` unless the core object type has an ADTS profile
fn read_adts_core_config(r: &mut BitReader) -> Option<AdtsCoreConfig> {
    let mut object_type = r.read_bits(5)?;
    let sr_index = r.read_bits(4)?;
    let channel_config = r.read_bits(4)?;
    if object_type == AOT_SBR || object_type == AOT_PS {
        // explicit SBR signalling: the core rate came first, then the SBR rate and the core type
        if r.read_bits(4)? == EXPLICIT_RATE_INDEX {
            r.read_bits(24)?;
        }
        object_type = r.read_bits(5)?;
    }
    let has_adts_profile = (AOT_AAC_MAIN..=AOT_AAC_LTP).contains(&object_type);
    (has_adts_profile && (sr_index as usize) < SAMPLE_RATES.len()).then_some(AdtsCoreConfig {
        object_type,
        sr_index,
        channel_config,
    })
}

// a demuxer's AAC access unit as ADTS, failing loud on a config ADTS cannot frame
#[cfg(feature = "std")]
pub(crate) fn require_adts(
    asc: &[u8],
    au: &[u8],
    demuxer: &'static str,
) -> Result<Vec<u8>, G2gError> {
    adts_from_asc(asc, au).ok_or_else(|| {
        g2g_error!(
            g2g_core::log::Target::category(demuxer),
            "AAC config {asc:02x?} cannot be ADTS framed"
        );
        G2gError::CapsMismatch
    })
}

// ADTS has no profile for ER AAC, ELD or USAC and no field for 960-sample frames or an SBR output rate, so a demuxer sends their access units raw behind this config as a frame
pub(crate) fn is_raw_aac_config(config: &[u8]) -> bool {
    read_object_type(&mut BitReader::new(config))
        .is_some_and(|object_type| RAW_AAC_OBJECT_TYPES.contains(&object_type))
        || has_960_sample_frames(config)
        || has_undoubled_sbr(config)
}

pub(crate) fn is_usac_config(config: &[u8]) -> bool {
    read_object_type(&mut BitReader::new(config)) == Some(AOT_USAC)
}

// the object type of a config a demuxer can neither ADTS-frame nor send raw (ALS, SLS, ...)
#[cfg(feature = "std")]
pub(crate) fn unframable_object_type(config: &[u8]) -> Option<u32> {
    if is_raw_aac_config(config) || adts_from_asc(config, &[]).is_some() {
        return None;
    }
    read_object_type(&mut BitReader::new(config))
}

fn has_960_sample_frames(config: &[u8]) -> bool {
    let mut r = BitReader::new(config);
    read_adts_core_config(&mut r).is_some() && r.read_bits(1) == Some(1)
}

// A decoder of ADTS finds SBR data on its own and doubles the core rate, so
// ADTS cannot carry SBR the config gives any other output rate.
fn has_undoubled_sbr(config: &[u8]) -> bool {
    sbr_rates(config).is_some_and(|rates| Some(rates.output) != rates.core.checked_mul(2))
}

#[derive(Debug, Clone, Copy)]
struct SbrRates {
    core: u32,
    output: u32,
}

// the AudioSpecificConfig's own `syncExtensionType` that announces SBR after the core config
const SBR_SYNC_EXTENSION_TYPE: u32 = 0x2B7;

// the core and output rates of a config that signals SBR, hierarchically or after the core config
fn sbr_rates(config: &[u8]) -> Option<SbrRates> {
    let mut r = BitReader::new(config);
    let object_type = read_object_type(&mut r)?;
    let core = read_sample_rate(&mut r)?;
    let channel_config = r.read_bits(4)?;
    if object_type == AOT_SBR || object_type == AOT_PS {
        let output = read_sample_rate(&mut r)?;
        return Some(SbrRates { core, output });
    }
    if !(AOT_AAC_MAIN..=AOT_AAC_LTP).contains(&object_type) {
        return None;
    }
    // GASpecificConfig: frameLengthFlag, dependsOnCoreCoder and its delay, extensionFlag
    r.read_bits(1)?;
    if r.read_bits(1)? == 1 {
        r.read_bits(14)?;
    }
    let extension_flag = r.read_bits(1)?;
    if channel_config == 0 {
        copy_program_config(&mut r, None)?;
    }
    if extension_flag == 1 {
        r.read_bits(1)?; // extensionFlag3
    }
    if r.read_bits(11)? != SBR_SYNC_EXTENSION_TYPE || read_object_type(&mut r)? != AOT_SBR {
        return None;
    }
    if r.read_bits(1)? == 0 {
        return None; // sbrPresentFlag
    }
    let output = read_sample_rate(&mut r)?;
    Some(SbrRates { core, output })
}

// a samplingFrequencyIndex, or the explicit 24-bit rate index 15 stands for
fn read_sample_rate(r: &mut BitReader) -> Option<u32> {
    match r.read_bits(4)? {
        EXPLICIT_RATE_INDEX => r.read_bits(24),
        index => SAMPLE_RATES.get(index as usize).copied(),
    }
}

// longest AudioSpecificConfig a muxer takes from the frame leading a raw AAC stream
const MAX_RAW_AAC_CONFIG_LEN: usize = 256;

// an AAC stream's AudioSpecificConfig from its first frame: an ADTS header's, or the frame itself when it leads raw units
pub(crate) fn muxer_audio_specific_config(
    first_frame: &[u8],
    muxer: &'static str,
) -> Result<Option<Vec<u8>>, G2gError> {
    if let Some(header) = asc_from_adts(first_frame) {
        // an ADTS stream of channel configuration 0 need not send a program config element
        let with_program_config = match asc_channel_config(&header) {
            Some(0) => asc_with_program_config(header, strip_adts(first_frame)),
            _ => None,
        };
        return Ok(Some(with_program_config.unwrap_or_else(|| header.to_vec())));
    }
    if opens_loas_frame(first_frame) {
        g2g_error!(
            g2g_core::log::Target::category(muxer),
            "cannot carry LOAS/LATM AAC"
        );
        return Err(G2gError::CapsMismatch);
    }
    if !is_raw_aac_config(first_frame) {
        return Ok(None);
    }
    if first_frame.len() > MAX_RAW_AAC_CONFIG_LEN {
        g2g_error!(
            g2g_core::log::Target::category(muxer),
            "AAC config of {} bytes exceeds {MAX_RAW_AAC_CONFIG_LEN}",
            first_frame.len()
        );
        return Err(G2gError::CapsMismatch);
    }
    Ok(Some(first_frame.to_vec()))
}

// whether `frame` is the config frame leading the raw access units of an AAC stream declared by `config`
pub(crate) fn is_leading_aac_config(config: &[u8], frame: &[u8]) -> bool {
    is_raw_aac_config(config) && frame == config
}

// the access unit a muxer stores for one frame of an AAC stream declared by `config`
pub(crate) fn muxer_access_unit<'a>(config: &[u8], frame: &'a [u8]) -> &'a [u8] {
    if is_raw_aac_config(config) {
        return frame;
    }
    let payload = strip_adts(frame);
    let config_has_program_config =
        asc_channel_config(config) == Some(0) && config.len() > ADTS_AUDIO_SPECIFIC_CONFIG_LEN;
    if !config_has_program_config {
        return payload;
    }
    // a copy of the config's program config element in band would be read twice
    let mut r = BitReader::new(payload);
    let program_config =
        r.read_bits(ELEMENT_ID_BITS) == Some(ID_PCE) && copy_program_config(&mut r, None).is_some();
    if program_config {
        // a program config element ends byte aligned, on its comment bytes
        &payload[r.bit_pos() / 8..]
    } else {
        payload
    }
}

fn asc_channel_config(asc: &[u8]) -> Option<u8> {
    asc.get(1).map(|byte| (byte >> 3) & 0x0F)
}

// a channel configuration of 0 leaves the layout to the program config element an ADTS payload opens with
fn asc_with_program_config(
    header: [u8; ADTS_AUDIO_SPECIFIC_CONFIG_LEN],
    payload: &[u8],
) -> Option<Vec<u8>> {
    let mut r = BitReader::new(payload);
    if r.read_bits(ELEMENT_ID_BITS)? != ID_PCE {
        return None;
    }
    let mut w = BitWriter::default();
    // the header's last three bits are the GASpecificConfig flags, all clear for ADTS
    w.write_bits(u32::from(u16::from_be_bytes(header)), 16);
    copy_program_config(&mut r, Some(&mut w))?;
    Some(w.into_bytes())
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
const AOT_USAC: u32 = 42;
// ER AAC LC, ER AAC LTP, ER AAC LD, ER AAC ELD, USAC
const RAW_AAC_OBJECT_TYPES: [u32; 5] = [17, 19, 23, 39, AOT_USAC];
// AAC Main, LC, SSR, LTP, ER AAC LC, ER AAC LTP, ER AAC LD: a GASpecificConfig with no layer number ahead of its program config
const GA_SPECIFIC_CONFIG_OBJECT_TYPES: [u32; 7] = [1, 2, 3, 4, 17, 19, 23];
const EXPLICIT_RATE_INDEX: u32 = 15;
// raw_data_block element id of a program_config_element
const ID_PCE: u32 = 5;
const ID_DSE: u32 = 4;
const ELEMENT_ID_BITS: u32 = 3;
// a data_stream_element's first byte: the element id, a four-bit tag, then data_byte_align_flag
const DSE_BYTE_ALIGN_FLAG: u8 = 0x01;

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

// the LOAS header (ISO/IEC 14496-3 1.7.2): an 11-bit syncword, then a 13-bit length
const LOAS_SYNC_WORD: u32 = 0x2B7;
const LOAS_LENGTH_BITS: u32 = 13;
const LOAS_HEADER_LEN: usize = 3;
// a latmBufferFullness of all ones marks a variable-rate stream
const LATM_VARIABLE_RATE_FULLNESS: u32 = 0xFF;
// PayloadLengthInfo writes a payload length as this byte value per full step, then the rest
const LATM_LENGTH_STEP: usize = 255;

// audioMuxVersion 1 carries `asc` verbatim behind its bit length, with no parse of its end
pub(crate) fn loas_from_asc(asc: &[u8], au: &[u8]) -> Option<Vec<u8>> {
    // the config's byte padding counts as the fill bits ascLen allows
    let asc_bits = u32::try_from(asc.len().checked_mul(8)?).ok()?;
    let mut w = BitWriter::default();
    w.write_bits(0, 1); // useSameStreamMux
    w.write_bits(1, 1); // audioMuxVersion
    w.write_bits(0, 1); // audioMuxVersionA
    write_latm_value(&mut w, LATM_VARIABLE_RATE_FULLNESS); // taraBufferFullness
    w.write_bits(1, 1); // allStreamsSameTimeFraming
    w.write_bits(0, 6); // numSubFrames
    w.write_bits(0, 4); // numProgram
    w.write_bits(0, 3); // numLayer
    write_latm_value(&mut w, asc_bits); // ascLen
    for &byte in asc {
        w.write_bits(u32::from(byte), 8);
    }
    w.write_bits(0, 3); // frameLengthType: a payload counted in bytes
    w.write_bits(LATM_VARIABLE_RATE_FULLNESS, 8); // latmBufferFullness
    w.write_bits(0, 1); // otherDataPresent
    w.write_bits(0, 1); // crcCheckPresent
    for _ in 0..au.len() / LATM_LENGTH_STEP {
        w.write_bits(LATM_LENGTH_STEP as u32, 8);
    }
    w.write_bits((au.len() % LATM_LENGTH_STEP) as u32, 8);
    // decoders byte-align a leading data_stream_element against the stream, and a LATM payload sits off byte boundaries
    let clears_dse_alignment = au.first().is_some_and(|&byte| {
        u32::from(byte >> (8 - ELEMENT_ID_BITS)) == ID_DSE && byte & DSE_BYTE_ALIGN_FLAG != 0
    }) && read_object_type(&mut BitReader::new(asc))
        .is_some_and(|object_type| !RAW_AAC_OBJECT_TYPES.contains(&object_type));
    for (index, &byte) in au.iter().enumerate() {
        let byte = if index == 0 && clears_dse_alignment {
            byte & !DSE_BYTE_ALIGN_FLAG
        } else {
            byte
        };
        w.write_bits(u32::from(byte), 8);
    }
    w.align_to_byte();
    let element = w.into_bytes();
    let length = u32::try_from(element.len())
        .ok()
        .filter(|&length| length < 1 << LOAS_LENGTH_BITS)?;
    let header = (LOAS_SYNC_WORD << LOAS_LENGTH_BITS) | length;
    let mut frame = Vec::with_capacity(LOAS_HEADER_LEN + element.len());
    frame.extend_from_slice(&header.to_be_bytes()[size_of::<u32>() - LOAS_HEADER_LEN..]);
    frame.extend_from_slice(&element);
    Some(frame)
}

// LatmGetValue's coding: the byte count less one in two bits, then the value in that many bytes
fn write_latm_value(w: &mut BitWriter, value: u32) {
    let bytes = (u32::BITS - value.leading_zeros()).div_ceil(8).max(1);
    w.write_bits(bytes - 1, 2);
    w.write_bits(value, bytes * 8);
}

fn read_latm_value(r: &mut BitReader) -> Option<u32> {
    let bytes = r.read_bits(2)? + 1;
    r.read_bits(bytes * 8)
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
    // the buffers that arrived while the framing was undecided
    undecided: Vec<Frame>,
    // LATM frames dropped because no caps were known when they arrived
    frames_dropped_before_caps: u64,
    /// Unconsumed ADTS bytes, starting at stream offset `buf_offset`.
    buf: Vec<u8>,
    buf_offset: u64,
    clock: FrameClock,
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

    // a trailing ID3v1 block leaves the last frame no successor header to confirm its sync
    fn strip_id3v1(&mut self) {
        let Some(start) = self.buf.len().checked_sub(ID3V1_LEN) else {
            return;
        };
        if parse_id3v1(&self.buf[start..]).is_some() {
            self.buf.truncate(start);
        }
    }

    // The framing the held buffers show, `None` while they show none: a file can
    // open on junk that runs past the first read before its first sync.
    fn held_framing(&self) -> Option<Framing> {
        // past this much, the stream is taken as LATM framed by its container
        const MAX_UNDECIDED_BYTES: usize = 64 * 1024;
        let held: Vec<u8> = self
            .undecided
            .iter()
            .flat_map(|frame| frame.domain.as_system_slice().unwrap_or(&[]))
            .copied()
            .collect();
        let first = self.undecided.first()?.domain.as_system_slice()?;
        // a LOAS payload is unaligned bits that can hold a valid ADTS header
        if opens_loas_frame(&held) && parse_loas(&held).is_some() {
            Some(Framing::Latm)
        } else if parse_adts(&held).is_some() {
            Some(Framing::Adts)
        } else if is_raw_aac_config(first) {
            Some(Framing::Raw)
        } else if held.len() > MAX_UNDECIDED_BYTES {
            Some(Framing::Latm)
        } else {
            None
        }
    }

    async fn forward_held(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        for frame in core::mem::take(&mut self.undecided) {
            self.forward(frame, out).await?;
        }
        Ok(())
    }

    async fn forward(&mut self, frame: Frame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let bytes = frame.domain.as_system_slice().unwrap_or(&[]);
        match self.framing {
            Framing::Adts => {
                // A byte source stamps every chunk 0, so only a real
                // time (a demuxer's) re-bases the counter.
                if let Some(pts) = frame.timing.pts().filter(|pts| *pts != 0) {
                    self.clock.note_chunk(
                        self.buf_offset,
                        self.buf_offset + self.buf.len() as u64,
                        pts,
                    );
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
                } else if self.last_emitted_caps.is_none() {
                    // a decoder cannot open on a frame that reuses a config it never saw
                    self.frames_dropped_before_caps += 1;
                    return Ok(());
                }
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
        }
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
        let frame_samples = u64::from(SAMPLES_PER_ACCESS_UNIT);
        let pts_ns = self
            .clock
            .frame_pts(start_offset, frame_samples, header.sample_rate);
        let duration_ns = ticks_to_ns(frame_samples, header.sample_rate);
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
                    if self.framing != Framing::Undecided {
                        return self.forward(frame, out).await;
                    }
                    self.undecided.push(frame);
                    let Some(framing) = self.held_framing() else {
                        return Ok(());
                    };
                    self.framing = framing;
                    self.forward_held(out).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    self.last_emitted_caps = None;
                    self.buf.clear();
                    self.undecided.clear();
                    self.clock.clear_pending();
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // The last access unit ends at the end of the stream, so it has
                // no successor to confirm its sync against.
                PipelinePacket::Eos => {
                    if self.framing == Framing::Undecided && !self.undecided.is_empty() {
                        self.framing = Framing::Latm;
                        self.forward_held(out).await?;
                    }
                    if self.framing == Framing::Adts {
                        self.strip_id3v1();
                        self.drain(true, out).await?;
                    }
                    if self.framing == Framing::Latm && self.headers_emitted == 0 {
                        g2g_error!(
                            self,
                            "none of {} frames carried an ADTS header or a LATM StreamMuxConfig this can read",
                            self.frames_dropped_before_caps
                        );
                        return Err(G2gError::CapsMismatch);
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
#[derive(Clone, Copy)]
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

/// Recover the channel count and sample rate from either AAC framing: LOAS/LATM
/// (the MPEG-TS / broadcast sync) first, since its unaligned payload can hold a
/// valid ADTS header, then ADTS (the elementary-stream sync).
fn parse_aac(au: &[u8]) -> Option<AacInfo> {
    parse_loas(au).or_else(|| parse_adts(au))
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
pub(crate) fn parse_loas(au: &[u8]) -> Option<AacInfo> {
    (0..au.len()).find_map(|i| {
        let frame = au[i..].get(..loas_frame_len(&au[i..])?)?;
        parse_audio_mux_element(&frame[LOAS_HEADER_LEN..])
    })
}

// syncword 0x2B7 byte-aligned: byte0 = 0x56, top 3 bits of byte1 = 111
pub(crate) fn opens_loas_frame(data: &[u8]) -> bool {
    matches!(data, [0x56, second, ..] if second & 0xE0 == 0xE0)
}

// the bytes of the LOAS frame `data` opens on, its header included
pub(crate) fn loas_frame_len(data: &[u8]) -> Option<usize> {
    if !opens_loas_frame(data) {
        return None;
    }
    let header = data.get(..LOAS_HEADER_LEN)?;
    // audioMuxLengthBytes: the 13 bits after the 11-bit syncword
    let element_len = (usize::from(header[1] & 0x1F) << 8) | usize::from(header[2]);
    Some(LOAS_HEADER_LEN + element_len)
}

/// Parse a LATM `AudioMuxElement` (muxConfigPresent = 1, the LOAS case) far
/// enough to reach the `AudioSpecificConfig`. Handles `audioMuxVersion` 0 and
/// 1 (the version whose values carry their own byte counts); the undefined
/// `audioMuxVersionA` 1 and stream-mux-config reuse yield `None` (caps stay
/// unrefined, never wrong).
fn parse_audio_mux_element(data: &[u8]) -> Option<AacInfo> {
    let mut r = BitReader::new(data);
    if r.read_bits(1)? == 1 {
        return None; // useSameStreamMux: reuses a prior config we did not retain
    }
    // StreamMuxConfig
    let audio_mux_version = r.read_bits(1)?;
    if audio_mux_version == 1 {
        if r.read_bits(1)? != 0 {
            return None; // audioMuxVersionA 1 has no defined syntax
        }
        read_latm_value(&mut r)?; // taraBufferFullness
    }
    r.read_bits(1)?; // allStreamsSameTimeFraming
    r.read_bits(6)?; // numSubFrames
    r.read_bits(4)?; // numProgram (only program 0 is parsed)
    r.read_bits(3)?; // numLayer of program 0 (only layer 0 is parsed)
                     // program 0, layer 0: useSameConfig is implied 0, so AudioSpecificConfig follows.
    if audio_mux_version == 1 {
        read_latm_value(&mut r)?; // ascLen
    }
    // a program config element aligns against the config's start, which the element puts off a byte boundary
    let mut config = BitWriter::default();
    while let Some(bit) = r.read_bit() {
        config.write_bit(bit);
    }
    parse_audio_specific_config(&mut BitReader::new(&config.into_bytes()))
}

pub(crate) fn audio_specific_config_info(config: &[u8]) -> Option<AacInfo> {
    parse_audio_specific_config(&mut BitReader::new(config))
}

/// Read the leading fields of an `AudioSpecificConfig` (ISO/IEC 14496-3): the
/// audio object type, sampling frequency (index or explicit), and channel
/// configuration, enough to pin the channel count and sample rate.
fn parse_audio_specific_config(r: &mut BitReader) -> Option<AacInfo> {
    let mut object_type = read_object_type(r)?;
    let sample_rate = read_sample_rate(r)?;
    let channel_config = r.read_bits(4)?;
    let channels = match channel_config {
        1..=6 => channel_config as u8,
        7 => 8,
        0 => {
            // explicit SBR signalling puts the SBR rate and the core type ahead of the GASpecificConfig
            if object_type == AOT_SBR || object_type == AOT_PS {
                read_sample_rate(r)?;
                object_type = read_object_type(r)?;
            }
            if !GA_SPECIFIC_CONFIG_OBJECT_TYPES.contains(&object_type) {
                return None;
            }
            // GASpecificConfig: frameLengthFlag, dependsOnCoreCoder and its delay, extensionFlag
            r.read_bits(1)?;
            if r.read_bits(1)? == 1 {
                r.read_bits(14)?;
            }
            r.read_bits(1)?;
            copy_program_config(r, None)?
        }
        _ => return None, // reserved
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

    const AOT_AAC_LC: u32 = 2;
    const RATE_INDEX_44100: u32 = 4;

    // AAC LC 44.1 kHz whose program config element declares a front channel pair and an LFE
    fn channel_config_zero_asc() -> Vec<u8> {
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
        asc.into_bytes()
    }

    #[test]
    fn adts_carries_the_program_config_of_a_channel_config_zero_asc() {
        let au = [0xAA, 0xBB];
        let frame = adts_from_asc(&channel_config_zero_asc(), &au).expect("ADTS with a PCE");
        let header = adts_header(&frame).expect("the header parses");
        assert_eq!((header.channel_config, header.channels), (0, 3));
        assert_eq!(header.sample_rate, SAMPLE_RATES[RATE_INDEX_44100 as usize]);
        assert_eq!(header.frame_len, frame.len());
        assert!(frame.ends_with(&au));
    }

    #[test]
    fn a_muxer_moves_the_in_band_program_config_back_into_the_config() {
        const MUXER: &str = "test";
        let asc = channel_config_zero_asc();
        let au = [0xAA, 0xBB];
        let frame = adts_from_asc(&asc, &au).expect("ADTS with a PCE");
        let config = muxer_audio_specific_config(&frame, MUXER)
            .expect("an ADTS frame")
            .expect("a config");
        assert_eq!(config, asc);
        assert_eq!(muxer_access_unit(&config, &frame), au);
    }

    #[test]
    fn a_muxer_refuses_loas_frames() {
        const MUXER: &str = "test";
        // AAC LC, 48 kHz, stereo
        const LC_CONFIG: [u8; 2] = [0x11, 0x90];
        let frame = loas_from_asc(&LC_CONFIG, &[0xAA, 0xBB]).expect("a LOAS frame");
        assert_eq!(
            muxer_audio_specific_config(&frame, MUXER),
            Err(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn adts_refuses_configs_it_cannot_carry() {
        // AAC-ELD (escaped object type 39), 48 kHz stereo
        const ELD_ASC: [u8; 4] = [0xF8, 0xE6, 0x40, 0x00];
        // AAC LC 48 kHz, frameLengthFlag set (960-sample frames), PCE follows
        const SHORT_FRAME_ASC: [u8; 2] = [0x11, 0x84];
        // AAC LC 48 kHz mono, frameLengthFlag set
        const SHORT_FRAME_MONO_ASC: [u8; 2] = [0x11, 0x8C];
        // AAC LC 48 kHz stereo
        const LC_ASC: [u8; 2] = [0x11, 0x90];
        assert!(adts_from_asc(&ELD_ASC, &[0xAA]).is_none());
        assert!(adts_from_asc(&SHORT_FRAME_ASC, &[0xAA]).is_none());
        assert!(adts_from_asc(&SHORT_FRAME_MONO_ASC, &[0xAA]).is_none());
        for raw in [&ELD_ASC[..], &SHORT_FRAME_ASC, &SHORT_FRAME_MONO_ASC] {
            assert!(is_raw_aac_config(raw), "{raw:02x?} goes raw");
        }
        assert!(!is_raw_aac_config(&LC_ASC));
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

    // AAC-LC at `core_index`, stereo, then the sync extension announcing SBR at `output_index`
    fn backward_compatible_sbr_config(core_index: u32, output_index: u32) -> Vec<u8> {
        const AOT_AAC_LC: u32 = 2;
        const STEREO: u32 = 2;
        let mut asc = BitWriter::default();
        let fields = [
            (AOT_AAC_LC, 5),
            (core_index, 4),
            (STEREO, 4),
            (0, 3), // frameLengthFlag, dependsOnCoreCoder, extensionFlag
            (SBR_SYNC_EXTENSION_TYPE, 11),
            (AOT_SBR, 5),
            (1, 1), // sbrPresentFlag
            (output_index, 4),
        ];
        for (value, bits) in fields {
            asc.write_bits(value, bits);
        }
        asc.into_bytes()
    }

    fn rate_index(rate: u32) -> u32 {
        SAMPLE_RATES.iter().position(|&r| r == rate).unwrap() as u32
    }

    #[test]
    fn sbr_at_the_core_rate_goes_raw() {
        const RATE: u32 = 48_000;
        // fate-suite aac/al_sbr_sr_48_2_fsaac48.mp4 signals SBR this way
        let same_rate = backward_compatible_sbr_config(rate_index(RATE), rate_index(RATE));
        assert!(is_raw_aac_config(&same_rate));
        let doubled = backward_compatible_sbr_config(rate_index(RATE / 2), rate_index(RATE));
        assert!(
            !is_raw_aac_config(&doubled),
            "ADTS implies the doubled rate"
        );
        assert!(adts_from_asc(&same_rate, &[0xAA]).is_none());
        assert!(adts_from_asc(&doubled, &[0xAA]).is_some());
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
    fn latm_audio_mux_version_a_bails_safely() {
        // Flip audioMuxVersion to 1: the allStreamsSameTimeFraming bit behind it
        // then reads as audioMuxVersionA 1, which has no syntax, so no (wrong)
        // caps are produced.
        let mut payload = latm_mux_element(2, 4, 2);
        payload[0] |= 0b0100_0000; // set bit 1 (audioMuxVersion) after useSameStreamMux
        assert!(parse_audio_mux_element(&payload).is_none());
    }

    // AAC-ELD, 48 kHz, stereo
    const ELD_ASC: [u8; 4] = [0xF8, 0xE6, 0x40, 0x00];

    #[test]
    fn loas_from_asc_reads_back_as_version_1_latm() {
        let frame = loas_from_asc(&ELD_ASC, &[0x7E, 0x89, 0xB4]).expect("a short unit frames");
        assert!(opens_loas_frame(&frame));
        let mux_len = (((frame[1] & 0x1F) as usize) << 8) | frame[2] as usize;
        assert_eq!(mux_len, frame.len() - LOAS_HEADER_LEN);
        let info = parse_aac(&frame).expect("the version 1 config parses");
        assert_eq!((info.channels, info.sample_rate), (2, 48_000));
    }

    #[test]
    fn a_channel_config_zero_loas_frame_counts_its_program_config_channels() {
        let frame = loas_from_asc(&channel_config_zero_asc(), &[0xAA, 0xBB]).expect("a LOAS frame");
        let info = parse_loas(&frame).expect("the config parses");
        assert_eq!(
            (info.channels, info.sample_rate),
            (3, SAMPLE_RATES[RATE_INDEX_44100 as usize])
        );
    }

    #[test]
    fn loas_from_asc_clears_a_leading_dse_byte_alignment_in_a_ga_payload_only() {
        // AAC LC, 48 kHz, stereo
        const LC_ASC: [u8; 2] = [0x11, 0x90];
        // a data_stream_element, tag 15, data_byte_align_flag set, then its count
        const ALIGNED_DSE: [u8; 2] = [0x9F, 0x02];
        let unaligned_dse = [ALIGNED_DSE[0] & !DSE_BYTE_ALIGN_FLAG, ALIGNED_DSE[1]];
        assert_eq!(
            loas_from_asc(&LC_ASC, &ALIGNED_DSE),
            loas_from_asc(&LC_ASC, &unaligned_dse)
        );
        // an ER payload has no element ids, so its first byte is data
        assert_ne!(
            loas_from_asc(&ELD_ASC, &ALIGNED_DSE),
            loas_from_asc(&ELD_ASC, &unaligned_dse)
        );
    }

    #[test]
    fn loas_from_asc_refuses_a_frame_past_the_13_bit_length() {
        let longest_element = (1 << LOAS_LENGTH_BITS) - 1;
        assert!(loas_from_asc(&ELD_ASC, &vec![0; longest_element]).is_none());
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
            ticks_to_ns(
                units * u64::from(SAMPLES_PER_ACCESS_UNIT),
                FRAME_RATE_HZ as u32,
            )
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
    async fn emits_the_last_unit_ahead_of_an_id3v1_trailer() {
        let mut stream = build_adts(2, 4, 16, 0);
        stream.extend(build_adts(2, 4, 16, 1));
        let mut trailer = vec![0u8; crate::id3::ID3V1_LEN];
        trailer[..3].copy_from_slice(b"TAG");
        stream.extend(trailer);
        let (parse, _) = run(&[stream]).await;
        assert_eq!(parse.frames_emitted(), 2);
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
    async fn keeps_the_unit_ahead_of_a_mid_stream_id3v2_tag() {
        let tags: g2g_core::TagList = [g2g_core::Tag::Title("segment".into())]
            .into_iter()
            .collect();
        let mut stream = build_adts(2, 4, 16, 0);
        stream.extend(build_adts(2, 4, 16, 1));
        stream.extend(crate::id3::write_id3v2(&tags, 4));
        stream.extend(build_adts(2, 4, 16, 2));
        let (parse, _) = run(&[stream]).await;
        assert_eq!(
            parse.frames_emitted(),
            3,
            "the tag confirms the unit before it"
        );
    }

    #[tokio::test]
    async fn junk_past_the_first_buffer_still_reaches_the_first_sync() {
        // more junk than one read holds, so the first buffer shows no framing at all
        let junk_buffers = [vec![0u8; 100], vec![0u8; 100]];
        let mut stream = Vec::new();
        for fill in 0..3u8 {
            stream.extend(build_adts(2, 4, 16, fill));
        }
        let mut buffers = junk_buffers.to_vec();
        buffers.push(stream);
        let (parse, sink) = run(&buffers).await;
        assert_eq!(parse.frames_emitted(), 3, "every unit behind the junk");
        assert_eq!(data_frames(&sink).len(), 3, "the junk is not forwarded");
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
    async fn a_loas_frame_holding_an_adts_header_stays_latm() {
        // 16 kHz mono, a header a byte scan accepts
        const ADTS_HEADER: [u8; 7] = [0xFF, 0xF1, 0x60, 0x40, 0x00, 0xFF, 0xFC];
        // bits ahead of the payload in a LOAS frame of a 4-byte config and a unit under 255 bytes
        const PAYLOAD_START_BIT: u32 = LOAS_HEADER_LEN as u32 * 8 + 90;
        let unaligned_by = 8 - PAYLOAD_START_BIT % 8;
        let mut w = BitWriter::default();
        w.write_bits(0, unaligned_by);
        for byte in ADTS_HEADER {
            w.write_bits(u32::from(byte), 8);
        }
        let frame = loas_from_asc(&ELD_ASC, &w.into_bytes()).unwrap();
        assert!(
            parse_adts(&frame).is_some(),
            "the frame holds an aligned ADTS header"
        );

        let mut parse = AacParse::new();
        parse.configure_pipeline(&aac_caps()).unwrap();
        let mut sink = CollectSink::default();
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, frame.clone())),
                &mut sink,
            )
            .await
            .unwrap();
        assert!(matches!(
            sink.packets[0],
            PipelinePacket::CapsChanged(Caps::Audio {
                channels: 2,
                sample_rate: 48_000,
                ..
            })
        ));
        let PipelinePacket::DataFrame(forwarded) = &sink.packets[1] else {
            panic!("the LOAS frame is forwarded");
        };
        assert_eq!(forwarded.domain.as_system_slice(), Some(frame.as_slice()));
    }

    // a LOAS frame whose AudioMuxElement sets useSameStreamMux, so it carries no config
    fn loas_frame_reusing_the_config() -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bits(1, 1);
        w.write_bits(0, 7);
        loas_frame(&w.into_bytes())
    }

    async fn push_frames(
        parse: &mut AacParse,
        sink: &mut CollectSink,
        frames: &[Vec<u8>],
    ) -> Result<(), G2gError> {
        for (sequence, bytes) in frames.iter().enumerate() {
            let frame = frame_with_bytes(sequence as u64, bytes.clone());
            parse
                .process(PipelinePacket::DataFrame(frame), sink)
                .await?;
        }
        parse.process(PipelinePacket::Eos, sink).await
    }

    #[tokio::test]
    async fn a_loas_stream_with_no_config_fails_at_eos() {
        let mut parse = AacParse::new();
        parse.configure_pipeline(&aac_caps()).unwrap();
        let mut sink = CollectSink::default();
        let frames = vec![loas_frame_reusing_the_config(); 3];
        assert_eq!(
            push_frames(&mut parse, &mut sink, &frames).await,
            Err(G2gError::CapsMismatch)
        );
        assert!(sink.packets.is_empty(), "nothing undescribed goes out");
    }

    #[tokio::test]
    async fn loas_frames_ahead_of_the_first_config_are_dropped() {
        let mut parse = AacParse::new();
        parse.configure_pipeline(&aac_caps()).unwrap();
        let mut sink = CollectSink::default();
        let with_config = loas_frame(&latm_mux_element(2, 4, 2));
        let reusing = loas_frame_reusing_the_config();
        let frames = [reusing.clone(), with_config.clone(), reusing.clone()];
        push_frames(&mut parse, &mut sink, &frames).await.unwrap();
        assert_eq!(caps_params(&sink), [(2, 44_100)]);
        assert!(matches!(sink.packets[0], PipelinePacket::CapsChanged(_)));
        let forwarded: Vec<&[u8]> = data_frames(&sink)
            .iter()
            .filter_map(|frame| frame.domain.as_system_slice())
            .collect();
        assert_eq!(forwarded, [with_config.as_slice(), reusing.as_slice()]);
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

//! Frame headers and lengths of the self-syncing audio bitstreams (AC-3 and
//! MPEG audio), read from a frame's own header.
//!
//! Several callers need the same arithmetic and must not drift: `FfmpegAudioDec`
//! splits a container-framed access unit into the individual frames it sends to
//! libavcodec, `PsDemux` re-aligns an MPEG program stream's sector-cut PES
//! payloads onto frame boundaries before it emits them, and `MpegAudioParse`
//! splits a bare `.mp3` / `.mp2` byte stream (the content sniffer reads a header
//! through it too, to type such a file). Kept on the `no_std` baseline because
//! the demuxer is.
//!
//! Every field here comes off the wire. A header whose fields are reserved,
//! free-format, or otherwise unusable yields `None` rather than a guessed
//! length, so a caller resynchronizes instead of mis-framing.

/// AC-3 syncword, the first two bytes of every syncframe.
pub(crate) const AC3_SYNC: [u8; 2] = [0x0B, 0x77];

/// Bytes of header needed to compute an AC-3 frame length.
pub(crate) const AC3_HEADER_LEN: usize = 5;

/// Bytes of header needed to compute an MPEG audio frame length.
pub(crate) const MPA_HEADER_LEN: usize = 4;

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

/// Bitrates (kbit/s) indexed by the header's 4-bit field, per version and layer.
/// MPEG-1 has one table per layer; MPEG-2 / 2.5 (the low-sample-rate extension)
/// has one for Layer I and one shared by Layers II and III. Index 0 ("free
/// format") and 15 are invalid here and fail the parse.
const MP1_V1_BITRATES_KBPS: [u32; 16] = [
    0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0,
];
const MP1_V2_BITRATES_KBPS: [u32; 16] = [
    0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0,
];
const MP2_V1_BITRATES_KBPS: [u32; 16] = [
    0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0,
];
const MP3_V1_BITRATES_KBPS: [u32; 16] = [
    0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
];
const LSF_V2_BITRATES_KBPS: [u32; 16] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
];
/// Sample rates for MPEG-1 (index 3 is reserved). MPEG-2 halves these, MPEG-2.5
/// quarters them.
const MPA_RATES_HZ: [u32; 4] = [44_100, 48_000, 32_000, 0];

/// Length in bytes of the AC-3 syncframe starting at `buf`, or `None` when `buf`
/// does not open on one: too short, no `0x0B77` syncword, or a reserved `fscod`
/// (3) / `frmsizecod` (>= 38). Byte 4 holds `fscod` in its top 2 bits and
/// `frmsizecod` in the low 6, whose pair gives the length in 16-bit words.
pub(crate) fn ac3_frame_len(buf: &[u8]) -> Option<usize> {
    let head = buf.get(..AC3_HEADER_LEN)?;
    if head[0] != AC3_SYNC[0] || head[1] != AC3_SYNC[1] {
        return None;
    }
    let fscod = (head[4] >> 6) as usize;
    let frmsizecod = (head[4] & 0x3F) as usize;
    if fscod >= 3 || frmsizecod >= AC3_FRAME_SIZE_WORDS.len() {
        return None; // reserved sample rate or frame-size code
    }
    let len = (AC3_FRAME_SIZE_WORDS[frmsizecod][fscod] as usize).saturating_mul(2);
    (len >= AC3_HEADER_LEN).then_some(len)
}

/// Which layer an MPEG audio frame is coded in. Layer I is decoded so a parser
/// can name it in the error it fails with; nothing here decodes Layer I audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MpaLayer {
    One,
    Two,
    Three,
}

/// Which MPEG audio version a frame is coded in: the sample rate and, for
/// Layer III, the frame's sample count both follow from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MpaVersion {
    Mpeg1,
    Mpeg2,
    Mpeg25,
}

/// The decoded fields of one MPEG audio frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MpaHeader {
    pub(crate) version: MpaVersion,
    pub(crate) layer: MpaLayer,
    pub(crate) sample_rate: u32,
    pub(crate) channels: u8,
    /// Samples per channel this frame decodes to, the presentation-time step.
    pub(crate) samples_per_frame: u32,
    pub(crate) frame_len: usize,
}

impl MpaHeader {
    /// Whether two headers describe the same stream, the test a resync uses to
    /// confirm a candidate sync against the frame behind it.
    pub(crate) fn same_stream(&self, other: &Self) -> bool {
        self.version == other.version
            && self.layer == other.layer
            && self.sample_rate == other.sample_rate
    }
}

/// Decode the MPEG audio frame header at the start of `buf`, or `None` when
/// `buf` does not open on one. The 4-byte header carries an 11-bit sync
/// (`0xFFE`), version, layer, bitrate and sample-rate indices, a padding bit and
/// the channel mode; the length is `samples_per_frame / 8 * bitrate / rate`
/// plus the padding slot (4 bytes in Layer I, 1 otherwise). A free-format or
/// reserved header has no computable length and fails.
pub(crate) fn mpa_header(buf: &[u8]) -> Option<MpaHeader> {
    let h = buf.get(..MPA_HEADER_LEN)?;
    if h[0] != 0xFF || (h[1] & 0xE0) != 0xE0 {
        return None;
    }
    let version = match (h[1] >> 3) & 0x03 {
        3 => MpaVersion::Mpeg1,
        2 => MpaVersion::Mpeg2,
        0 => MpaVersion::Mpeg25,
        _ => return None, // reserved version
    };
    let layer = match (h[1] >> 1) & 0x03 {
        3 => MpaLayer::One,
        2 => MpaLayer::Two,
        1 => MpaLayer::Three,
        _ => return None, // reserved layer
    };
    let lsf = version != MpaVersion::Mpeg1;
    let bitrates = match (layer, lsf) {
        (MpaLayer::One, false) => &MP1_V1_BITRATES_KBPS,
        (MpaLayer::One, true) => &MP1_V2_BITRATES_KBPS,
        (MpaLayer::Two, false) => &MP2_V1_BITRATES_KBPS,
        (MpaLayer::Three, false) => &MP3_V1_BITRATES_KBPS,
        (_, true) => &LSF_V2_BITRATES_KBPS,
    };
    let bitrate = bitrates[((h[2] >> 4) & 0x0F) as usize].saturating_mul(1_000);
    let mut sample_rate = MPA_RATES_HZ[((h[2] >> 2) & 0x03) as usize];
    // The low-sample-rate extensions: MPEG-2 halves, MPEG-2.5 quarters.
    sample_rate >>= match version {
        MpaVersion::Mpeg1 => 0,
        MpaVersion::Mpeg2 => 1,
        MpaVersion::Mpeg25 => 2,
    };
    if bitrate == 0 || sample_rate == 0 {
        return None; // free-format / reserved: no computable frame length
    }
    let samples_per_frame = match (layer, lsf) {
        (MpaLayer::One, _) => 384,
        (MpaLayer::Three, true) => 576,
        _ => 1152,
    };
    let padding = ((h[2] >> 1) & 1) as usize;
    let frame_len = match layer {
        // Layer I is coded in 4-byte slots, so its padding is one slot too.
        MpaLayer::One => (12 * bitrate / sample_rate) as usize * 4 + padding * 4,
        _ => (samples_per_frame / 8 * bitrate / sample_rate) as usize + padding,
    };
    // Mode 3 is single_channel; the other three modes are two-channel codings.
    let channels = if (h[3] >> 6) & 0x03 == 3 { 1 } else { 2 };
    (frame_len >= MPA_HEADER_LEN).then_some(MpaHeader {
        version,
        layer,
        sample_rate,
        channels,
        samples_per_frame,
        frame_len,
    })
}

/// Length in bytes of the MPEG audio frame starting at `buf`, for the callers
/// that split a Layer II (`mp2`) / Layer III (`mp3`) stream. Layer I fails here
/// rather than widening what those callers accept as a sync.
pub(crate) fn mpa_frame_len(buf: &[u8]) -> Option<usize> {
    let header = mpa_header(buf)?;
    (header.layer != MpaLayer::One).then_some(header.frame_len)
}

/// Synthetic MPEG audio frames, shared by the tests of the parser and of the
/// content sniffer (both need a stream that frames the way a real one does).
#[cfg(test)]
pub(crate) mod test_frames {
    use alloc::vec;
    use alloc::vec::Vec;

    /// Bitrate index 9 is 128 kbit/s in the MPEG-1 Layer III table and rate
    /// index 0 is 44100 Hz, which makes every frame this long.
    pub(crate) const MP3_128K_44100_LEN: usize = 417;
    const BITRATE_INDEX_128K: u8 = 9;
    const RATE_INDEX_44100: u8 = 0;
    /// Channel mode 3 is single_channel, mode 0 (stereo) is the default.
    const MODE_MONO: u8 = 3;

    /// One MPEG-1 Layer III frame at 128 kbit/s / 44100 Hz, its payload filled
    /// with `fill`.
    pub(crate) fn mp3_frame(mono: bool, fill: u8) -> Vec<u8> {
        let mode = if mono { MODE_MONO } else { 0 };
        let mut frame = vec![
            0xFF,
            0xFB, // sync, MPEG-1, Layer III, no CRC
            (BITRATE_INDEX_128K << 4) | (RATE_INDEX_44100 << 2),
            mode << 6,
        ];
        frame.resize(MP3_128K_44100_LEN, fill);
        frame
    }
}

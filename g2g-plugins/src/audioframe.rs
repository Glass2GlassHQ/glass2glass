//! Frame lengths of the self-syncing audio bitstreams (AC-3 and MPEG audio),
//! read from a frame's own header.
//!
//! Two callers need the same arithmetic and must not drift: `FfmpegAudioDec`
//! splits a container-framed access unit into the individual frames it sends to
//! libavcodec, and `PsDemux` re-aligns an MPEG program stream's sector-cut PES
//! payloads onto frame boundaries before it emits them. Kept on the `no_std`
//! baseline because the demuxer is.
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

/// Bitrates (kbit/s) indexed by the header's 4-bit field, per version and layer:
/// MPEG-1 Layer II, MPEG-1 Layer III, and MPEG-2 / 2.5 Layer III (whose low-rate
/// extension has its own table). Index 0 ("free format") and 15 are invalid here
/// and fail the parse.
const MP2_BITRATES_KBPS: [u32; 16] = [
    0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0,
];
const MP3_V1_BITRATES_KBPS: [u32; 16] = [
    0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
];
const MP3_V2_BITRATES_KBPS: [u32; 16] = [
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

/// Length in bytes of the MPEG audio (Layer II `mp2` / Layer III `mp3`) frame
/// starting at `buf`, or `None` when `buf` does not open on one. The 4-byte
/// header carries an 11-bit sync (`0xFFE`), version, layer, bitrate and
/// sample-rate indices, and a padding bit; the length is
/// `samples_per_frame / 8 * bitrate / rate + padding`, a frame being 1152
/// samples except MPEG-2 / 2.5 Layer III's 576. A free-format or reserved header
/// has no computable length and fails.
pub(crate) fn mpa_frame_len(buf: &[u8]) -> Option<usize> {
    let h = buf.get(..MPA_HEADER_LEN)?;
    if h[0] != 0xFF || (h[1] & 0xE0) != 0xE0 {
        return None;
    }
    let version = (h[1] >> 3) & 0x03;
    let layer = (h[1] >> 1) & 0x03;
    let bitrate_index = ((h[2] >> 4) & 0x0F) as usize;
    let (bitrates, bytes_per_frame_num) = match (version, layer) {
        (1, _) => return None,                  // reserved version
        (3, 2) => (&MP2_BITRATES_KBPS, 144),    // MPEG-1 Layer II
        (_, 2) => (&MP2_BITRATES_KBPS, 144),    // MPEG-2 / 2.5 Layer II
        (3, 1) => (&MP3_V1_BITRATES_KBPS, 144), // MPEG-1 Layer III
        (_, 1) => (&MP3_V2_BITRATES_KBPS, 72),  // MPEG-2 / 2.5 Layer III
        _ => return None,                       // Layer I or reserved
    };
    let bitrate = bitrates[bitrate_index].saturating_mul(1_000);
    let mut rate = MPA_RATES_HZ[((h[2] >> 2) & 0x03) as usize];
    // The low-sample-rate extensions: MPEG-2 halves, MPEG-2.5 quarters.
    rate >>= match version {
        3 => 0,
        2 => 1,
        _ => 2,
    };
    if bitrate == 0 || rate == 0 {
        return None; // free-format / reserved: no computable frame length
    }
    let padding = ((h[2] >> 1) & 1) as usize;
    let len = (bytes_per_frame_num * bitrate / rate) as usize + padding;
    (len >= MPA_HEADER_LEN).then_some(len)
}

use alloc::vec::Vec;

use ffmpeg::codec::Id;
use ffmpeg::ffi;
use ffmpeg::format::Pixel;
use ffmpeg::frame::Video as FfVideo;
use ffmpeg::packet::{Mut, Packet};
use ffmpeg_next as ffmpeg;

use g2g_core::{AudioFormat, Caps, CodecTag, G2gError, TagNamespace, VideoCodec};

use crate::mp4box::boxes;
use crate::riff::{read_u16, read_u32};

// The codec libavformat's tag tables name, `None` when they name none or
// libavcodec has no decoder for it. A RIFF video tag falls back to the MOV
// table as avidec does, a MOV one to the RIFF table as mov.c does.
pub(crate) fn video_codec_id(tag: CodecTag) -> Option<Id> {
    // SAFETY: the table getters return pointers to libavformat's static tables.
    let (riff, mov) = unsafe {
        (
            ffi::avformat_get_riff_video_tags(),
            ffi::avformat_get_mov_video_tags(),
        )
    };
    let tables = match tag.namespace {
        TagNamespace::Riff => [riff, mov],
        TagNamespace::Mov => [mov, riff],
    };
    tables
        .into_iter()
        .find_map(|table| table_codec_id(table, tag.tag))
        .filter(|id| has_decoder(*id))
}

// a QuickTime `ms` / `TS` twocc carries a WAVEFORMATEX tag in its low half, as mov.c reads it
const MS_TWOCC: u32 = u32::from_le_bytes([b'm', b's', 0, 0]);
const TS_TWOCC: u32 = u32::from_le_bytes([b'T', b'S', 0, 0]);
const TWOCC_MASK: u32 = 0xFFFF;

pub(crate) fn audio_codec_id(tag: CodecTag) -> Option<Id> {
    // SAFETY: the table getters return pointers to libavformat's static tables.
    let (riff, mov) = unsafe {
        (
            ffi::avformat_get_riff_audio_tags(),
            ffi::avformat_get_mov_audio_tags(),
        )
    };
    let id = match tag.namespace {
        TagNamespace::Riff => table_codec_id(riff, tag.tag),
        TagNamespace::Mov => table_codec_id(mov, tag.tag).or_else(|| {
            let twocc = tag.tag & TWOCC_MASK;
            (twocc == MS_TWOCC || twocc == TS_TWOCC)
                .then(|| table_codec_id(riff, tag.tag.swap_bytes() & TWOCC_MASK))
                .flatten()
        }),
    };
    id.filter(|id| has_decoder(*id))
}

fn table_codec_id(table: *const ffi::AVCodecTag, tag: u32) -> Option<Id> {
    let tables = [table, core::ptr::null()];
    // SAFETY: `tables` is a null-terminated list of libavformat's static tag
    // tables, which `av_codec_get_id` only reads.
    let id = Id::from(unsafe { ffi::av_codec_get_id(tables.as_ptr(), tag) });
    (id != Id::None).then_some(id)
}

fn has_decoder(id: Id) -> bool {
    ffmpeg::codec::decoder::find(id).is_some()
}

pub(crate) fn decodes_video(caps: &Caps) -> bool {
    matches!(caps, Caps::CompressedVideo { codec: VideoCodec::Tagged(tag), .. } if video_codec_id(*tag).is_some())
}

pub(crate) fn decodes_audio(caps: &Caps) -> bool {
    matches!(caps, Caps::Audio { format: AudioFormat::Tagged(tag), .. } if audio_codec_id(*tag).is_some())
}

pub(crate) fn decodes(caps: &Caps) -> bool {
    decodes_video(caps) || decodes_audio(caps)
}

// the descriptor flags of a pixel format libswscale reads as RGB
const RGB_FLAGS: u64 = (ffi::AV_PIX_FMT_FLAG_RGB | ffi::AV_PIX_FMT_FLAG_PAL) as u64;
const LUMA_AND_CHROMA: u8 = 3;

// The pixel format g2g packs that is nearest a decoded one it has no mapping for:
// RGBA or RGB for RGB and palettized sources by alpha, A420 for YUV with alpha,
// else planar YUV of the same subsampling at the nearest depth, 4:2:0 when g2g
// has none of that subsampling.
pub(crate) fn nearest_output_pixel(source: Pixel) -> Pixel {
    const ALPHA_FLAG: u64 = ffi::AV_PIX_FMT_FLAG_ALPHA as u64;
    let Some(descriptor) = source.descriptor() else {
        return Pixel::YUV420P;
    };
    // SAFETY: `descriptor` points at libavutil's static descriptor table.
    let (flags, depth) = unsafe {
        let raw = &*descriptor.as_ptr();
        (raw.flags, raw.comp[0].depth)
    };
    let alpha = flags & ALPHA_FLAG != 0;
    if flags & RGB_FLAGS != 0 {
        return if alpha { Pixel::RGBA } else { Pixel::RGB24 };
    }
    if alpha {
        return Pixel::YUVA420P;
    }
    let depths = match (
        descriptor.log2_chroma_w(),
        descriptor.log2_chroma_h(),
        descriptor.nb_components(),
    ) {
        (0, 0, LUMA_AND_CHROMA) => [Pixel::YUV444P, Pixel::YUV444P10LE, Pixel::YUV444P12LE],
        (1, 0, LUMA_AND_CHROMA) => [Pixel::YUV422P, Pixel::YUV422P10LE, Pixel::YUV422P12LE],
        _ => [Pixel::YUV420P, Pixel::YUV420P10LE, Pixel::YUV420P12LE],
    };
    match depth {
        ..=8 => depths[0],
        9..=10 => depths[1],
        _ => depths[2],
    }
}

// Converts a decoded picture to another pixel format with libswscale, set up
// the way the ffmpeg CLI's format filter drives it: the frame API with the
// default bicubic scaler on one thread, the source's chroma siting left
// unspecified and its fields scaled as one frame.
pub(crate) struct FormatConverter {
    // null until the first conversion allocates it
    context: *mut ffi::SwsContext,
}

impl Default for FormatConverter {
    fn default() -> Self {
        Self {
            context: core::ptr::null_mut(),
        }
    }
}

impl core::fmt::Debug for FormatConverter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FormatConverter")
            .field("allocated", &!self.context.is_null())
            .finish()
    }
}

impl Drop for FormatConverter {
    fn drop(&mut self) {
        // SAFETY: the context is null, which `sws_freeContext` ignores, or one
        // `sws_alloc_context` returned and nothing uses after this.
        unsafe { ffi::sws_freeContext(self.context) };
    }
}

// SAFETY: the `SwsContext` is owned and only touched through `&mut`, moving with
// the decoder that holds it under the same one-thread-at-a-time contract as
// the decoder's own `AVCodecContext`.
unsafe impl Send for FormatConverter {}

impl FormatConverter {
    pub(crate) fn convert(
        &mut self,
        frame: &mut FfVideo,
        target: Pixel,
    ) -> Result<FfVideo, G2gError> {
        let failed = || G2gError::Hardware(g2g_core::HardwareError::Other);
        if self.context.is_null() {
            // SAFETY: a plain allocation, checked for null below.
            self.context = unsafe { ffi::sws_alloc_context() };
            if self.context.is_null() {
                return Err(failed());
            }
        }
        let mut converted = FfVideo::empty();
        // SAFETY: `frame` is a decoded frame and `converted` a fresh one, both
        // owned here. The destination takes the source's properties, then the
        // target format and size. `sws_scale_frame` allocates its planes and
        // reconfigures the context whenever the frames' parameters change.
        let ret = unsafe {
            let source = frame.as_mut_ptr();
            // the format filter scales progressive, whatever the picture says
            (*source).flags &= !ffi::AV_FRAME_FLAG_INTERLACED;
            (*source).chroma_location = ffi::AVChromaLocation::AVCHROMA_LOC_UNSPECIFIED;
            let destination = converted.as_mut_ptr();
            if ffi::av_frame_copy_props(destination, source) < 0 {
                return Err(failed());
            }
            // the CLI's format filter keeps a chroma source's range and matrix, and
            // leaves an RGB, palettized or grey one's to the scaler's limited BT.601
            if !has_chroma(frame.format()) {
                (*destination).color_range = ffi::AVColorRange::AVCOL_RANGE_UNSPECIFIED;
                (*destination).colorspace = ffi::AVColorSpace::AVCOL_SPC_UNSPECIFIED;
            }
            (*destination).format = ffi::AVPixelFormat::from(target) as i32;
            (*destination).width = (*source).width;
            (*destination).height = (*source).height;
            ffi::sws_scale_frame(self.context, destination, source)
        };
        if ret < 0 {
            return Err(failed());
        }
        Ok(converted)
    }
}

fn has_chroma(format: Pixel) -> bool {
    format.descriptor().is_some_and(|descriptor| {
        // SAFETY: `descriptor` points at libavutil's static descriptor table.
        let flags = unsafe { (*descriptor.as_ptr()).flags };
        flags & RGB_FLAGS == 0 && descriptor.nb_components() >= LUMA_AND_CHROMA
    })
}

// hands the decoder a container palette beside an access unit, as libavformat does on the first packet
pub(crate) fn attach_palette(packet: &mut Packet, palette: &[u8]) -> Result<(), G2gError> {
    // SAFETY: `packet` owns a valid AVPacket. The side data buffer is allocated
    // at `palette.len()` bytes and owned by the packet from then on.
    unsafe {
        let data = ffi::av_packet_new_side_data(
            packet.as_mut_ptr(),
            ffi::AVPacketSideDataType::AV_PKT_DATA_PALETTE,
            palette.len(),
        );
        if data.is_null() {
            return Err(G2gError::Hardware(g2g_core::HardwareError::Other));
        }
        core::ptr::copy_nonoverlapping(palette.as_ptr(), data, palette.len());
    }
    Ok(())
}

// What a video stream's container header gives the decoder: the codec context
// fields, the extradata, and the palette its first packet carries.
#[derive(Debug)]
pub(crate) struct TaggedVideo {
    pub id: Id,
    pub fields: VideoContextFields,
    pub extradata: Vec<u8>,
    // AVPALETTE_SIZE bytes of native-endian 0xAARRGGBB words
    pub palette: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct VideoContextFields {
    pub codec_tag: u32,
    pub bits_per_coded_sample: i32,
    pub width: i32,
    pub height: i32,
}

impl VideoContextFields {
    // SAFETY: `context` must be a valid, not yet opened codec context.
    pub(crate) unsafe fn set_on(self, context: *mut ffi::AVCodecContext) {
        // SAFETY: the caller hands a valid unopened context, and these are plain
        // fields libavcodec reads at open.
        unsafe {
            (*context).codec_tag = self.codec_tag;
            (*context).bits_per_coded_sample = self.bits_per_coded_sample;
            (*context).width = self.width;
            (*context).height = self.height;
        }
    }
}

// The container header leads a tagged stream in band: a RIFF stream's whole
// BITMAPINFOHEADER, a MOV stream's whole sample entry.
pub(crate) fn tagged_video(tag: CodecTag, header: &[u8]) -> Result<TaggedVideo, G2gError> {
    let id = video_codec_id(tag).ok_or(G2gError::CapsMismatch)?;
    match tag.namespace {
        TagNamespace::Riff => riff_video(id, tag.tag, header),
        TagNamespace::Mov => mov_video(id, tag.tag, header),
    }
}

const PALETTE_BYTES: usize = ffi::AVPALETTE_SIZE as usize;
const PALETTE_ENTRY_BYTES: usize = 4;
const OPAQUE: u32 = 0xFF00_0000;

// avidec appends this to a BI_RGB stream's extradata so rawdec flips the bottom-up rows
const BOTTOM_UP: &[u8] = b"BottomUp\0";
const BITMAPINFO_LEN: usize = 40;
const BITMAPINFO_SIZE: usize = 0;
const BITMAPINFO_WIDTH: usize = 4;
const BITMAPINFO_HEIGHT: usize = 8;
const BITMAPINFO_BIT_COUNT: usize = 14;
const AVRN: u32 = u32::from_le_bytes(*b"AVRn");
// avidec decodes an `AVRn` stream as MJPEG unless its extradata says 1:1 here
const AVRN_ASPECT_AT: core::ops::Range<usize> = 28..31;

fn riff_video(id: Id, tag: u32, header: &[u8]) -> Result<TaggedVideo, G2gError> {
    let field = |at| read_u32(header, at).ok_or(G2gError::CapsMismatch);
    let declared_len = field(BITMAPINFO_SIZE)? as usize;
    let width = i32::try_from(field(BITMAPINFO_WIDTH)?).map_err(|_| G2gError::CapsMismatch)?;
    let height = field(BITMAPINFO_HEIGHT)? as i32;
    let bits = read_u16(header, BITMAPINFO_BIT_COUNT).ok_or(G2gError::CapsMismatch)?;
    // a writer that padded the chunk to even length but declared the odd size
    let extradata_end = if declared_len + 1 == header.len() && declared_len % 2 == 1 {
        declared_len
    } else {
        header.len()
    };
    let mut extradata = header
        .get(BITMAPINFO_LEN..extradata_end)
        .unwrap_or_default()
        .to_vec();
    // avidec reads a palettized codec's palette off the end of its extradata
    let palette = (!extradata.is_empty() && bits <= 8).then(|| {
        let palette_len = (PALETTE_ENTRY_BYTES << bits).min(extradata.len());
        let mut start = extradata.len() - palette_len;
        if start >= BOTTOM_UP.len() && extradata.ends_with(BOTTOM_UP) {
            start -= BOTTOM_UP.len();
        }
        let mut palette = alloc::vec![0u8; PALETTE_BYTES];
        let entries = extradata[start..start + palette_len].chunks_exact(PALETTE_ENTRY_BYTES);
        for (slot, entry) in palette.chunks_exact_mut(PALETTE_ENTRY_BYTES).zip(entries) {
            let rgb = u32::from_le_bytes(entry.try_into().expect("4 bytes"));
            slot.copy_from_slice(&(OPAQUE | rgb).to_ne_bytes());
        }
        palette
    });
    if tag == 0 && height > 0 {
        extradata.extend_from_slice(BOTTOM_UP);
    }
    let id = if id == Id::AVRN && tag == AVRN && extradata.get(AVRN_ASPECT_AT) != Some(b"1:1") {
        Id::MJPEG
    } else {
        id
    };
    Ok(TaggedVideo {
        id,
        fields: VideoContextFields {
            codec_tag: tag,
            bits_per_coded_sample: i32::from(bits),
            width,
            height: height.checked_abs().ok_or(G2gError::CapsMismatch)?,
        },
        extradata,
        palette,
    })
}

// offsets into a video sample entry, its size and fourcc included
const VISUAL_ENTRY_WIDTH: usize = 32;
const VISUAL_ENTRY_HEIGHT: usize = 34;
const VISUAL_ENTRY_DEPTH: usize = 82;
const VISUAL_ENTRY_COLOR_TABLE_ID: usize = 84;
const VISUAL_ENTRY_COLOR_TABLE: usize = 86;
const VISUAL_ENTRY_CHILDREN: usize = 86;
const DEPTH_BITS: u16 = 0x1F;
const DEPTH_GREYSCALE: u16 = 0x20;

fn be16(data: &[u8], at: usize) -> Result<u16, G2gError> {
    data.get(at..at + 2)
        .map(|bytes| u16::from_be_bytes(bytes.try_into().expect("2 bytes")))
        .ok_or(G2gError::CapsMismatch)
}

fn be32(data: &[u8], at: usize) -> Result<u32, G2gError> {
    crate::mp4box::be32(data, at)
}

fn mov_video(id: Id, tag: u32, entry: &[u8]) -> Result<TaggedVideo, G2gError> {
    let depth = be16(entry, VISUAL_ENTRY_DEPTH)?;
    let palette = qt_palette(id, entry, depth)?;
    let bits = if palette.is_some() {
        depth & DEPTH_BITS
    } else {
        depth
    };
    let mut extradata = Vec::new();
    for (kind, payload) in boxes(entry.get(VISUAL_ENTRY_CHILDREN..).unwrap_or_default()) {
        extension_atom(id, kind, payload, &mut extradata);
    }
    Ok(TaggedVideo {
        id,
        fields: VideoContextFields {
            codec_tag: tag,
            bits_per_coded_sample: i32::from(bits),
            width: i32::from(be16(entry, VISUAL_ENTRY_WIDTH)?),
            height: i32::from(be16(entry, VISUAL_ENTRY_HEIGHT)?),
        },
        extradata,
        palette,
    })
}

// ff_get_qtpalette: a 1-, 2-, 4- or 8-bit entry is palettized, from its own
// colour table, a greyscale ramp, or the default Macintosh palette
fn qt_palette(id: Id, entry: &[u8], depth: u16) -> Result<Option<Vec<u8>>, G2gError> {
    let bits = depth & DEPTH_BITS;
    let greyscale = depth & DEPTH_GREYSCALE != 0;
    let color_table_id = be16(entry, VISUAL_ENTRY_COLOR_TABLE_ID)?;
    if greyscale && id == Id::CINEPAK {
        return Ok(None);
    }
    if !matches!(bits, 1 | 2 | 4 | 8) {
        return Ok(None);
    }
    let colors = 1usize << bits;
    let mut words = [0u32; PALETTE_BYTES / PALETTE_ENTRY_BYTES];
    if greyscale && bits > 1 && color_table_id != 0 {
        let step = 256 / (colors as i32 - 1);
        let mut level: i32 = 255;
        for word in words.iter_mut().take(colors) {
            *word = opaque_rgb([level as u8; 3]);
            level = (level - step).max(0);
        }
    } else if color_table_id != 0 {
        for (index, word) in words.iter_mut().take(colors).enumerate() {
            *word = opaque_rgb(default_mac_color(bits, index));
        }
    } else {
        // start (32 bits), flags (16), end (16), then 8 bytes per colour: index, r, g, b as 16-bit
        const COLOR_RECORD_LEN: usize = 8;
        const COLOR_RECORD_RGB_AT: usize = 2;
        let start = be32(entry, VISUAL_ENTRY_COLOR_TABLE)? as usize;
        let end = be16(entry, VISUAL_ENTRY_COLOR_TABLE + 6)? as usize;
        if start <= end && end < words.len() {
            let records = entry
                .get(VISUAL_ENTRY_COLOR_TABLE + 8..)
                .unwrap_or_default()
                .chunks_exact(COLOR_RECORD_LEN);
            for (word, record) in words[start..=end].iter_mut().zip(records) {
                let rgb = &record[COLOR_RECORD_RGB_AT..];
                // the top byte of each 16-bit component
                *word = opaque_rgb([rgb[0], rgb[2], rgb[4]]);
            }
        }
    }
    Ok(Some(
        words.iter().flat_map(|word| word.to_ne_bytes()).collect(),
    ))
}

fn opaque_rgb([r, g, b]: [u8; 3]) -> u32 {
    OPAQUE | u32::from(r) << 16 | u32::from(g) << 8 | u32::from(b)
}

// libavformat's qt_default_palette tables, which the 256-colour one builds as a
// 6x6x6 cube stepping down from white then red, green, blue and grey ramps
fn default_mac_color(bits: u16, index: usize) -> [u8; 3] {
    const TWO: [[u8; 3]; 2] = [[0xFF; 3], [0x00; 3]];
    const FOUR: [u8; 4] = [0xFF, 0xAC, 0x55, 0x00];
    const SIXTEEN: [[u8; 3]; 16] = [
        [0xFF, 0xFF, 0xFF],
        [0xFC, 0xF3, 0x05],
        [0xFF, 0x64, 0x02],
        [0xDD, 0x08, 0x06],
        [0xF2, 0x08, 0x84],
        [0x46, 0x00, 0xA5],
        [0x00, 0x00, 0xD4],
        [0x02, 0xAB, 0xEA],
        [0x1F, 0xB7, 0x14],
        [0x00, 0x64, 0x11],
        [0x56, 0x2C, 0x05],
        [0x90, 0x71, 0x3A],
        [0xC0, 0xC0, 0xC0],
        [0x80, 0x80, 0x80],
        [0x40, 0x40, 0x40],
        [0x00, 0x00, 0x00],
    ];
    match bits {
        1 => TWO[index],
        2 => [FOUR[index]; 3],
        4 => SIXTEEN[index],
        _ => default_mac_color_256(index),
    }
}

fn default_mac_color_256(index: usize) -> [u8; 3] {
    const CUBE_STEP: u8 = 0x33;
    const CUBE_SIDE: usize = 6;
    const CUBE_LEN: usize = CUBE_SIDE * CUBE_SIDE * CUBE_SIDE - 1;
    // each ramp skips the cube's levels, 0xEE down to 0x11
    const RAMP: [u8; 10] = [0xEE, 0xDD, 0xBB, 0xAA, 0x88, 0x77, 0x55, 0x44, 0x22, 0x11];
    if index < CUBE_LEN {
        let level = |digit: usize| 0xFF - CUBE_STEP * digit as u8;
        return [
            level(index / (CUBE_SIDE * CUBE_SIDE)),
            level(index / CUBE_SIDE % CUBE_SIDE),
            level(index % CUBE_SIDE),
        ];
    }
    let ramp_index = index - CUBE_LEN;
    let Some(&value) = RAMP.get(ramp_index % RAMP.len()) else {
        return [0; 3];
    };
    match ramp_index / RAMP.len() {
        0 => [value, 0, 0],
        1 => [0, value, 0],
        2 => [0, 0, value],
        3 => [value; 3],
        _ => [0; 3],
    }
}

// the extension atoms mov.c turns into extradata
fn extension_atom(id: Id, kind: &[u8; 4], payload: &[u8], extradata: &mut Vec<u8>) {
    const BITMAPINFO_IN_STRF: usize = BITMAPINFO_LEN;
    match kind {
        b"glbl" if extradata.len() <= 1 => *extradata = payload.to_vec(),
        b"strf" if payload.len() > BITMAPINFO_IN_STRF => {
            *extradata = payload[BITMAPINFO_IN_STRF..].to_vec()
        }
        b"SMI " if id == Id::SVQ3 => append_atom(kind, payload, extradata),
        b"alac" if id == Id::ALAC => append_atom(kind, payload, extradata),
        b"wave" if matches!(id, Id::QDM2 | Id::QDMC | Id::SPEEX) => *extradata = payload.to_vec(),
        b"wave" => {
            for (kind, payload) in boxes(payload) {
                extension_atom(id, kind, payload, extradata);
            }
        }
        _ => {}
    }
}

fn append_atom(kind: &[u8; 4], payload: &[u8], extradata: &mut Vec<u8>) {
    const ATOM_HEADER_LEN: usize = 8;
    let size = u32::try_from(payload.len() + ATOM_HEADER_LEN).unwrap_or(u32::MAX);
    extradata.extend_from_slice(&size.to_be_bytes());
    extradata.extend_from_slice(kind);
    extradata.extend_from_slice(payload);
}

// What an audio stream's container header gives the decoder.
#[derive(Debug)]
pub(crate) struct TaggedAudio {
    pub id: Id,
    pub codec_tag: u32,
    pub channels: i32,
    pub sample_rate: i32,
    pub block_align: i32,
    pub bits_per_coded_sample: i32,
    pub bit_rate: i64,
    pub extradata: Vec<u8>,
}

impl TaggedAudio {
    // SAFETY: `context` must be a valid, not yet opened codec context.
    pub(crate) unsafe fn set_context_fields(&self, context: *mut ffi::AVCodecContext) {
        // SAFETY: the caller hands a valid unopened context, whose fresh channel
        // layout holds no allocated map. The rest are plain fields read at open.
        unsafe {
            (*context).codec_tag = self.codec_tag;
            (*context).sample_rate = self.sample_rate;
            (*context).block_align = self.block_align;
            (*context).bits_per_coded_sample = self.bits_per_coded_sample;
            (*context).bit_rate = self.bit_rate;
            (*context).ch_layout.order = ffi::AVChannelOrder::AV_CHANNEL_ORDER_UNSPEC;
            (*context).ch_layout.nb_channels = self.channels;
        }
    }
}

// a RIFF stream's whole WAVEFORMATEX, a MOV stream's whole sample entry
pub(crate) fn tagged_audio(tag: CodecTag, header: &[u8]) -> Result<TaggedAudio, G2gError> {
    let id = audio_codec_id(tag).ok_or(G2gError::CapsMismatch)?;
    match tag.namespace {
        TagNamespace::Riff => riff_audio(id, tag.tag, header),
        TagNamespace::Mov => mov_audio(id, tag.tag, header),
    }
}

const WAVEFORMAT_LEN: usize = 14;
const WAVEFORMAT_CHANNELS: usize = 2;
const WAVEFORMAT_SAMPLE_RATE: usize = 4;
const WAVEFORMAT_BYTE_RATE: usize = 8;
const WAVEFORMAT_BLOCK_ALIGN: usize = 12;
const WAVEFORMAT_BITS: usize = 14;
const WAVEFORMAT_EXTRA_SIZE: usize = 16;
const WAVEFORMATEX_LEN: usize = 18;
// a plain WAVEFORMAT carries no sample width, which ff_get_wav_header takes as 8
const WAVEFORMAT_DEFAULT_BITS: u16 = 8;

fn riff_audio(id: Id, tag: u32, header: &[u8]) -> Result<TaggedAudio, G2gError> {
    if header.len() < WAVEFORMAT_LEN {
        return Err(G2gError::CapsMismatch);
    }
    let field16 = |at| read_u16(header, at).ok_or(G2gError::CapsMismatch);
    let field32 = |at| read_u32(header, at).ok_or(G2gError::CapsMismatch);
    let sample_rate = field32(WAVEFORMAT_SAMPLE_RATE)?;
    let bit_rate = i64::from(field32(WAVEFORMAT_BYTE_RATE)?) * 8;
    let bits = read_u16(header, WAVEFORMAT_BITS).unwrap_or(WAVEFORMAT_DEFAULT_BITS);
    let extradata = match read_u16(header, WAVEFORMAT_EXTRA_SIZE) {
        Some(declared) if header.len() >= WAVEFORMATEX_LEN => {
            let end = WAVEFORMATEX_LEN + usize::from(declared).min(header.len() - WAVEFORMATEX_LEN);
            header[WAVEFORMATEX_LEN..end].to_vec()
        }
        _ => Vec::new(),
    };
    let sample_rate = i32::try_from(sample_rate)
        .ok()
        .filter(|rate| *rate > 0)
        .ok_or(G2gError::CapsMismatch)?;
    // ff_get_wav_header reads G.726's sample width off its bit rate
    let bits = if id == Id::ADPCM_G726 {
        i32::try_from(bit_rate / i64::from(sample_rate)).map_err(|_| G2gError::CapsMismatch)?
    } else {
        i32::from(bits)
    };
    Ok(TaggedAudio {
        id,
        codec_tag: tag,
        channels: i32::from(field16(WAVEFORMAT_CHANNELS)?),
        sample_rate,
        block_align: i32::from(field16(WAVEFORMAT_BLOCK_ALIGN)?),
        bits_per_coded_sample: bits,
        bit_rate,
        extradata,
    })
}

// the codecs whose block_align mov.c takes from the sound description's bytes per frame
const FRAMED_BY_BYTES_PER_FRAME: [Id; 7] = [
    Id::GSM,
    Id::ADPCM_MS,
    Id::ADPCM_IMA_WAV,
    Id::ILBC,
    Id::MACE3,
    Id::MACE6,
    Id::QDM2,
];
const ALAC_EXTRADATA_LEN: usize = 36;
const ALAC_CHANNELS_AT: usize = 21;
const ALAC_SAMPLE_RATE_AT: usize = 32;

fn mov_audio(id: Id, tag: u32, entry: &[u8]) -> Result<TaggedAudio, G2gError> {
    let payload = entry
        .get(crate::mp4box::BOX_HEADER_LEN..)
        .ok_or(G2gError::CapsMismatch)?;
    let description = crate::fmp4::SoundDescription::parse(payload, tag)?;
    let mut extradata = Vec::new();
    for (kind, payload) in boxes(payload.get(description.children_at..).unwrap_or_default()) {
        extension_atom(id, kind, payload, &mut extradata);
    }
    let mut channels = i32::from(description.channels);
    let mut sample_rate = description.sample_rate;
    if id == Id::ALAC && extradata.len() == ALAC_EXTRADATA_LEN {
        channels = i32::from(extradata[ALAC_CHANNELS_AT]);
        sample_rate = be32(&extradata, ALAC_SAMPLE_RATE_AT)?;
    }
    // 3GPP speech is mono at its codec's rate whatever the sound description says
    const AMR_NB_RATE: u32 = 8_000;
    const AMR_WB_RATE: u32 = 16_000;
    match id {
        Id::AMR_NB => (channels, sample_rate) = (1, AMR_NB_RATE),
        Id::AMR_WB => (channels, sample_rate) = (1, AMR_WB_RATE),
        _ => {}
    }
    // SAFETY: a pure lookup on a codec id.
    let codec_bits = unsafe { ffi::av_get_bits_per_sample(id.into()) };
    let block_align = if FRAMED_BY_BYTES_PER_FRAME.contains(&id) {
        description.bytes_per_frame
    } else {
        0
    };
    Ok(TaggedAudio {
        id,
        codec_tag: tag,
        channels,
        sample_rate: i32::try_from(sample_rate).map_err(|_| G2gError::CapsMismatch)?,
        block_align: i32::try_from(block_align).map_err(|_| G2gError::CapsMismatch)?,
        bits_per_coded_sample: if codec_bits > 0 {
            codec_bits
        } else {
            i32::from(description.sample_bits)
        },
        bit_rate: 0,
        extradata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn riff(fourcc: &[u8; 4]) -> CodecTag {
        CodecTag {
            namespace: TagNamespace::Riff,
            tag: u32::from_le_bytes(*fourcc),
        }
    }

    fn mov(fourcc: &[u8; 4]) -> CodecTag {
        CodecTag {
            namespace: TagNamespace::Mov,
            tag: u32::from_le_bytes(*fourcc),
        }
    }

    #[test]
    fn resolves_riff_and_mov_tags_through_libavformat() {
        assert_eq!(video_codec_id(riff(b"cvid")), Some(Id::CINEPAK));
        assert_eq!(video_codec_id(riff(b"ULY0")), Some(Id::UTVIDEO));
        assert_eq!(video_codec_id(mov(b"rle ")), Some(Id::QTRLE));
        assert_eq!(video_codec_id(mov(b"SVQ3")), Some(Id::SVQ3));
        const TRUESPEECH: u32 = 0x0022;
        assert_eq!(
            audio_codec_id(CodecTag {
                namespace: TagNamespace::Riff,
                tag: TRUESPEECH,
            }),
            Some(Id::TRUESPEECH)
        );
        assert_eq!(audio_codec_id(mov(b"MAC3")), Some(Id::MACE3));
        assert_eq!(audio_codec_id(mov(b"ms\0\x02")), Some(Id::ADPCM_MS));
    }

    #[test]
    fn rejects_a_tag_libavformat_does_not_know() {
        assert_eq!(video_codec_id(riff(b"ZZZ9")), None);
        assert_eq!(video_codec_id(mov(b"ZZZ9")), None);
        assert_eq!(audio_codec_id(mov(b"ZZZ9")), None);
        let unknown = Caps::CompressedVideo {
            codec: VideoCodec::Tagged(riff(b"ZZZ9")),
            width: g2g_core::Dim::Any,
            height: g2g_core::Dim::Any,
            framerate: g2g_core::Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert!(!decodes(&unknown));
    }

    fn bitmapinfo(fourcc: &[u8; 4], bits: u16, tail: &[u8]) -> Vec<u8> {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 48;
        let mut header = alloc::vec![0u8; BITMAPINFO_LEN];
        header[..4].copy_from_slice(&(BITMAPINFO_LEN as u32).to_le_bytes());
        header[4..8].copy_from_slice(&WIDTH.to_le_bytes());
        header[8..12].copy_from_slice(&HEIGHT.to_le_bytes());
        header[14..16].copy_from_slice(&bits.to_le_bytes());
        header[16..20].copy_from_slice(fourcc);
        header.extend_from_slice(tail);
        header
    }

    #[test]
    fn a_palettized_bitmapinfo_header_carries_its_palette_and_depth() {
        const BITS: u16 = 2;
        let colors: Vec<u8> = (0..4u8).flat_map(|i| [i, i + 10, i + 20, 0]).collect();
        let video = tagged_video(riff(b"CRAM"), &bitmapinfo(b"CRAM", BITS, &colors))
            .expect("CRAM is msvideo1");
        assert_eq!(video.id, Id::MSVIDEO1);
        assert_eq!(video.fields.bits_per_coded_sample, i32::from(BITS));
        assert_eq!(video.extradata, colors);
        let palette = video.palette.expect("a 2-bit stream has a palette");
        let second = u32::from_ne_bytes(palette[4..8].try_into().unwrap());
        assert_eq!(second, OPAQUE | u32::from_le_bytes([1, 11, 21, 0]));
    }

    #[test]
    fn the_default_mac_palette_matches_libavformat() {
        // entries from libavformat's qt_default_palette_256
        assert_eq!(default_mac_color_256(1), [0xFF, 0xFF, 0xCC]);
        assert_eq!(default_mac_color_256(214), [0x00, 0x00, 0x33]);
        assert_eq!(default_mac_color_256(215), [0xEE, 0x00, 0x00]);
        assert_eq!(default_mac_color_256(234), [0x00, 0x11, 0x00]);
        assert_eq!(default_mac_color_256(245), [0xEE, 0xEE, 0xEE]);
        assert_eq!(default_mac_color_256(255), [0x00, 0x00, 0x00]);
    }
}

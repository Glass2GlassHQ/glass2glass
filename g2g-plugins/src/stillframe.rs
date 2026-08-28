//! Byte-stream framing and header geometry for the still-image formats, shared
//! by the still-image parsers (`jpegparse`, `pngparse`) and decoders (`PngDec`,
//! `WebPDec`).
//!
//! This exists because a byte source hands over whatever a read returned, not
//! whole files: `filesrc` emits 64 KiB chunks, so an image larger than that
//! arrives in pieces, while `multifilesrc` hands over one complete image per
//! buffer. [`ImageAssembler`] covers both, walking with the format's own length
//! function.
//!
//! Every length and dimension read here is the file's own, so each step is
//! folded with checked arithmetic and held under a fixed budget: a malformed
//! header fails the parse rather than sizing an allocation.

use alloc::vec::Vec;

use g2g_core::G2gError;

/// Ceiling on the encoded bytes held while waiting for the rest of an image.
/// Bounds a byte stream that opens with a plausible signature and then never
/// completes, which would otherwise grow the buffer for as long as it flows.
pub(crate) const MAX_ENCODED_BYTES: usize = 64 * 1024 * 1024;

/// Length of the complete image at the start of `data`, or `None` when more
/// bytes are needed. `Err` when the bytes cannot be an image of this format at
/// all, or when the length it claims is past [`MAX_ENCODED_BYTES`].
pub(crate) type FrameLength = fn(&[u8]) -> Result<Option<usize>, G2gError>;

/// Reassembles whole encoded images from a byte stream.
///
/// A buffer that already holds exactly one image passes through with a single
/// copy; anything else accumulates until the format's own length says the image
/// is complete, so a source's read size never decides whether a file decodes.
#[derive(Debug, Default)]
pub(crate) struct ImageAssembler {
    pending: Vec<u8>,
}

impl ImageAssembler {
    /// Take one buffer of the stream and hand back every image it completed.
    pub(crate) fn push(
        &mut self,
        bytes: &[u8],
        frame_length: FrameLength,
    ) -> Result<Vec<Vec<u8>>, G2gError> {
        if self.pending.len().saturating_add(bytes.len()) > MAX_ENCODED_BYTES {
            return Err(G2gError::CapsMismatch);
        }
        self.pending.extend_from_slice(bytes);

        let mut images = Vec::new();
        while let Some(length) = frame_length(&self.pending)? {
            let rest = self.pending.split_off(length);
            images.push(core::mem::replace(&mut self.pending, rest));
        }
        Ok(images)
    }

    /// End of stream: bytes still held are an image that never completed.
    pub(crate) fn finish(&mut self) -> Result<(), G2gError> {
        if self.pending.is_empty() {
            Ok(())
        } else {
            self.pending = Vec::new();
            Err(G2gError::CapsMismatch)
        }
    }

    /// Drop what a flushing seek made stale.
    pub(crate) fn reset(&mut self) {
        self.pending = Vec::new();
    }
}

/// The 8 bytes every PNG opens with.
pub(crate) const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// Each chunk opens with a 4-byte payload length and a 4-byte type, and closes
/// with a 4-byte CRC.
const PNG_CHUNK_HEADER: usize = 8;
pub(crate) const PNG_CHUNK_CRC: usize = 4;
const PNG_CHUNK_OVERHEAD: usize = PNG_CHUNK_HEADER + PNG_CHUNK_CRC;
/// The last chunk of every PNG.
pub(crate) const PNG_END_CHUNK: [u8; 4] = *b"IEND";
/// The first chunk of every PNG, and the one carrying the geometry.
const PNG_HEADER_CHUNK: [u8; 4] = *b"IHDR";

/// Walk a PNG's chunk list to the end of its `IEND` chunk, so a byte stream that
/// splits or joins files is framed back into whole images.
pub(crate) fn png_frame_length(data: &[u8]) -> Result<Option<usize>, G2gError> {
    if data.len() < PNG_SIGNATURE.len() {
        // Not yet enough to tell a PNG from anything else.
        if PNG_SIGNATURE.starts_with(data) {
            return Ok(None);
        }
        return Err(G2gError::CapsMismatch);
    }
    if !data.starts_with(&PNG_SIGNATURE) {
        return Err(G2gError::CapsMismatch);
    }

    let mut offset = PNG_SIGNATURE.len();
    loop {
        let Some(header) = data.get(offset..offset + PNG_CHUNK_HEADER) else {
            return Ok(None);
        };
        let payload = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let chunk_type: [u8; 4] = [header[4], header[5], header[6], header[7]];
        let end = offset
            .checked_add(PNG_CHUNK_OVERHEAD)
            .and_then(|used| used.checked_add(payload))
            .filter(|end| *end <= MAX_ENCODED_BYTES)
            .ok_or(G2gError::CapsMismatch)?;
        if chunk_type == PNG_END_CHUNK {
            return Ok((data.len() >= end).then_some(end));
        }
        offset = end;
    }
}

/// `(width, height)` from a PNG's `IHDR`, the first chunk of the file. `None`
/// when the image does not open with one, or declares a zero side.
pub(crate) fn png_geometry(image: &[u8]) -> Option<(u32, u32)> {
    let chunk = image.get(PNG_SIGNATURE.len()..)?;
    if chunk.get(4..8)? != PNG_HEADER_CHUNK {
        return None;
    }
    let body = chunk.get(PNG_CHUNK_HEADER..PNG_CHUNK_HEADER + 8)?;
    let dimension =
        |at: usize| u32::from_be_bytes([body[at], body[at + 1], body[at + 2], body[at + 3]]);
    let (width, height) = (dimension(0), dimension(4));
    (width > 0 && height > 0).then_some((width, height))
}

/// Marker prefix byte. A run of them is padding before the next marker.
const JPEG_MARKER_PREFIX: u8 = 0xFF;
/// Start and end of image, the markers that frame a file.
const JPEG_START_OF_IMAGE: u8 = 0xD8;
const JPEG_END_OF_IMAGE: u8 = 0xD9;
/// Start of scan: its segment header is followed by entropy-coded data, which is
/// not length-declared and has to be scanned for the next marker.
const JPEG_START_OF_SCAN: u8 = 0xDA;
/// Restart markers, which appear inside the entropy-coded data.
const JPEG_RESTART_FIRST: u8 = 0xD0;
const JPEG_RESTART_LAST: u8 = 0xD7;
/// Standalone markers with no length field: `TEM` plus the two above.
const JPEG_TEMPORARY: u8 = 0x01;
/// The frame headers carrying the geometry: `SOF0`..`SOF15`, less the three
/// codes in that range that mean something else.
const JPEG_FRAME_HEADER_FIRST: u8 = 0xC0;
const JPEG_FRAME_HEADER_LAST: u8 = 0xCF;
const JPEG_HUFFMAN_TABLE: u8 = 0xC4;
const JPEG_EXTENSIONS: u8 = 0xC8;
const JPEG_ARITHMETIC_CONDITIONING: u8 = 0xCC;
/// Length field, itself included in the count.
const JPEG_SEGMENT_LENGTH: usize = 2;

fn jpeg_is_standalone(marker: u8) -> bool {
    matches!(
        marker,
        JPEG_START_OF_IMAGE | JPEG_END_OF_IMAGE | JPEG_TEMPORARY | JPEG_RESTART_FIRST
            ..=JPEG_RESTART_LAST
    )
}

fn jpeg_is_frame_header(marker: u8) -> bool {
    matches!(marker, JPEG_FRAME_HEADER_FIRST..=JPEG_FRAME_HEADER_LAST)
        && !matches!(
            marker,
            JPEG_HUFFMAN_TABLE | JPEG_EXTENSIONS | JPEG_ARITHMETIC_CONDITIONING
        )
}

/// The offset of the next marker's code at or after `at`, skipping entropy-coded
/// data: a `FF 00` pair is a stuffed data byte and a restart marker belongs to
/// the scan, so neither ends it. `None` while the data runs to the end of what
/// has arrived.
fn jpeg_next_marker(data: &[u8], at: usize) -> Option<usize> {
    let mut offset = at;
    loop {
        let prefix = data[offset..]
            .iter()
            .position(|b| *b == JPEG_MARKER_PREFIX)?;
        let code = offset + prefix + 1;
        let marker = *data.get(code)?;
        let stuffing_or_restart = marker == 0x00
            || marker == JPEG_MARKER_PREFIX
            || matches!(marker, JPEG_RESTART_FIRST..=JPEG_RESTART_LAST);
        if !stuffing_or_restart {
            return Some(code);
        }
        offset = code;
    }
}

/// Walk a JPEG's markers to the end of its `EOI`, so a byte stream that splits
/// or joins files (an MJPEG dump, a `filesrc` chunking) is framed back into
/// whole images.
pub(crate) fn jpeg_frame_length(data: &[u8]) -> Result<Option<usize>, G2gError> {
    const SIGNATURE: [u8; 2] = [JPEG_MARKER_PREFIX, JPEG_START_OF_IMAGE];
    if data.len() < SIGNATURE.len() {
        if SIGNATURE.starts_with(data) {
            return Ok(None);
        }
        return Err(G2gError::CapsMismatch);
    }
    if data[..SIGNATURE.len()] != SIGNATURE {
        return Err(G2gError::CapsMismatch);
    }

    let mut offset = SIGNATURE.len();
    loop {
        if offset > MAX_ENCODED_BYTES {
            return Err(G2gError::CapsMismatch);
        }
        // Padding: any number of FF bytes may precede a marker code.
        let Some(rest) = data.get(offset..) else {
            return Ok(None);
        };
        let Some(skip) = rest.iter().position(|b| *b != JPEG_MARKER_PREFIX) else {
            return Ok(None);
        };
        if skip == 0 {
            // A marker prefix must follow the previous segment.
            return Err(G2gError::CapsMismatch);
        }
        let code = offset + skip;
        let marker = data[code];
        if marker == JPEG_END_OF_IMAGE {
            return Ok(Some(code + 1));
        }
        if jpeg_is_standalone(marker) {
            offset = code + 1;
            continue;
        }
        let Some(header) = data.get(code + 1..code + 1 + JPEG_SEGMENT_LENGTH) else {
            return Ok(None);
        };
        let length = usize::from(u16::from_be_bytes([header[0], header[1]]));
        if length < JPEG_SEGMENT_LENGTH {
            return Err(G2gError::CapsMismatch);
        }
        let body_end = code
            .checked_add(1)
            .and_then(|after| after.checked_add(length))
            .filter(|end| *end <= MAX_ENCODED_BYTES)
            .ok_or(G2gError::CapsMismatch)?;
        if marker != JPEG_START_OF_SCAN {
            offset = body_end;
            continue;
        }
        // The scan's entropy-coded data follows its header; the next marker that
        // is neither stuffing nor a restart ends it.
        if body_end > data.len() {
            return Ok(None);
        }
        match jpeg_next_marker(data, body_end) {
            Some(next) => offset = next - 1,
            None => return Ok(None),
        }
    }
}

/// `(width, height)` from a JPEG's frame header (`SOF`). `None` when the image
/// carries no frame header, or declares a zero side.
pub(crate) fn jpeg_geometry(image: &[u8]) -> Option<(u32, u32)> {
    /// Sample precision, then the two 16-bit sides.
    const PRECISION: usize = 1;
    let mut offset = 2;
    loop {
        let skip = image
            .get(offset..)?
            .iter()
            .position(|b| *b != JPEG_MARKER_PREFIX)?;
        if skip == 0 {
            return None;
        }
        let code = offset + skip;
        let marker = *image.get(code)?;
        if marker == JPEG_END_OF_IMAGE {
            return None;
        }
        if jpeg_is_standalone(marker) {
            offset = code + 1;
            continue;
        }
        let header = image.get(code + 1..code + 1 + JPEG_SEGMENT_LENGTH)?;
        let length = usize::from(u16::from_be_bytes([header[0], header[1]]));
        let body = code + 1 + JPEG_SEGMENT_LENGTH;
        if jpeg_is_frame_header(marker) {
            let sides = image.get(body + PRECISION..body + PRECISION + 4)?;
            let height = u32::from(u16::from_be_bytes([sides[0], sides[1]]));
            let width = u32::from(u16::from_be_bytes([sides[2], sides[3]]));
            return (width > 0 && height > 0).then_some((width, height));
        }
        // Past a scan there is no frame header left to find in a baseline file,
        // and a progressive one repeats its SOF before the first scan.
        if marker == JPEG_START_OF_SCAN {
            return None;
        }
        offset = code.checked_add(1)?.checked_add(length)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// A stand-in image format for the assembler's own behaviour, which does not
    /// depend on PNG or JPEG: a 4-byte big-endian length, then that many bytes.
    const LENGTH_PREFIX: usize = 4;

    fn prefixed_frame_length(data: &[u8]) -> Result<Option<usize>, G2gError> {
        let Some(header) = data.get(..LENGTH_PREFIX) else {
            return Ok(None);
        };
        let payload = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let total = payload
            .checked_add(LENGTH_PREFIX)
            .filter(|total| *total <= MAX_ENCODED_BYTES)
            .ok_or(G2gError::CapsMismatch)?;
        Ok((data.len() >= total).then_some(total))
    }

    fn prefixed_bytes(payload: usize) -> Vec<u8> {
        let mut file = Vec::from((payload as u32).to_be_bytes());
        file.extend_from_slice(&vec![0u8; payload]);
        file
    }

    /// A PNG shaped file: signature, one `IHDR` chunk of `payload` bytes, then an
    /// empty `IEND`. The CRCs are zeroed, since framing never checks them.
    fn png_bytes(payload: usize) -> Vec<u8> {
        let zero_crc = [0u8; PNG_CHUNK_CRC];
        let mut file = Vec::from(PNG_SIGNATURE);
        file.extend_from_slice(&(payload as u32).to_be_bytes());
        file.extend_from_slice(b"IHDR");
        file.extend_from_slice(&vec![0u8; payload]);
        file.extend_from_slice(&zero_crc);
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(&PNG_END_CHUNK);
        file.extend_from_slice(&zero_crc);
        file
    }

    /// A PNG shaped file whose `IHDR` declares `width` x `height`.
    fn png_bytes_sized(width: u32, height: u32) -> Vec<u8> {
        /// `IHDR`: two 32-bit sides then depth, colour type, compression, filter,
        /// interlace.
        const IHDR_PAYLOAD: usize = 13;
        let mut file = png_bytes(IHDR_PAYLOAD);
        let at = PNG_SIGNATURE.len() + PNG_CHUNK_HEADER;
        file[at..at + 4].copy_from_slice(&width.to_be_bytes());
        file[at + 4..at + 8].copy_from_slice(&height.to_be_bytes());
        file
    }

    /// A baseline-JPEG shaped file: `SOI`, an `APP0` segment, a `SOF0` declaring
    /// `width` x `height`, a `SOS` with `scan` bytes of entropy-coded data, `EOI`.
    fn jpeg_bytes(width: u16, height: u16, scan: &[u8]) -> Vec<u8> {
        let mut file = vec![JPEG_MARKER_PREFIX, JPEG_START_OF_IMAGE];
        // APP0, a length and one payload byte.
        file.extend_from_slice(&[JPEG_MARKER_PREFIX, 0xE0, 0x00, 0x03, 0x00]);
        file.extend_from_slice(&[
            JPEG_MARKER_PREFIX,
            JPEG_FRAME_HEADER_FIRST,
            0x00,
            0x0B,
            0x08,
        ]);
        file.extend_from_slice(&height.to_be_bytes());
        file.extend_from_slice(&width.to_be_bytes());
        // One component, its sampling factors and table selector.
        file.extend_from_slice(&[0x01, 0x01, 0x11, 0x00]);
        file.extend_from_slice(&[JPEG_MARKER_PREFIX, JPEG_START_OF_SCAN, 0x00, 0x08]);
        file.extend_from_slice(&[0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]);
        file.extend_from_slice(scan);
        file.extend_from_slice(&[JPEG_MARKER_PREFIX, JPEG_END_OF_IMAGE]);
        file
    }

    #[test]
    fn assembler_rejoins_a_split_image_and_splits_a_joined_one() {
        let file = prefixed_bytes(40);
        // Split across three buffers, the `filesrc` case: nothing comes out
        // until the last piece lands, and then it is the whole file.
        let mut assembler = ImageAssembler::default();
        let (a, rest) = file.split_at(2);
        let (b, c) = rest.split_at(20);
        assert_eq!(assembler.push(a, prefixed_frame_length), Ok(Vec::new()));
        assert_eq!(assembler.push(b, prefixed_frame_length), Ok(Vec::new()));
        assert_eq!(
            assembler.push(c, prefixed_frame_length),
            Ok(vec![file.clone()])
        );
        assert_eq!(assembler.finish(), Ok(()));

        // Two whole images in one buffer come out as two.
        let mut joined = file.clone();
        joined.extend_from_slice(&file);
        let mut assembler = ImageAssembler::default();
        assert_eq!(
            assembler.push(&joined, prefixed_frame_length),
            Ok(vec![file.clone(), file.clone()])
        );
    }

    #[test]
    fn assembler_reports_a_stream_that_ends_mid_image() {
        let file = prefixed_bytes(40);
        let mut assembler = ImageAssembler::default();
        assert_eq!(
            assembler.push(&file[..12], prefixed_frame_length),
            Ok(Vec::new())
        );
        assert_eq!(
            assembler.finish(),
            Err(G2gError::CapsMismatch),
            "a half-received image is not silently dropped"
        );
        // The failed image is cleared, so the next one starts clean.
        assert_eq!(assembler.push(&file, prefixed_frame_length), Ok(vec![file]));
    }

    #[test]
    fn assembler_bounds_a_stream_that_never_completes() {
        // A length claiming almost the whole ceiling, so the length itself is
        // allowed and only the held bytes can stop the stream. Feeding it
        // forever must hit the ceiling rather than growing with the stream.
        const CHUNK_BYTES: usize = 8 * 1024 * 1024;
        let header = (MAX_ENCODED_BYTES as u32 - 32).to_be_bytes();

        let mut assembler = ImageAssembler::default();
        assert_eq!(
            assembler.push(&header, prefixed_frame_length),
            Ok(Vec::new())
        );
        let chunk = vec![0u8; CHUNK_BYTES];
        let mut pushes = 0;
        loop {
            match assembler.push(&chunk, prefixed_frame_length) {
                Ok(images) => {
                    assert!(images.is_empty());
                    pushes += 1;
                    assert!(pushes <= MAX_ENCODED_BYTES / CHUNK_BYTES, "must stop");
                }
                Err(e) => {
                    assert_eq!(e, G2gError::CapsMismatch);
                    break;
                }
            }
        }
    }

    #[test]
    fn png_frame_length_walks_the_chunk_list() {
        let file = png_bytes(13);
        assert_eq!(png_frame_length(&file), Ok(Some(file.len())));
        // Every prefix is "need more", never a wrong length.
        for cut in 0..file.len() {
            assert_eq!(png_frame_length(&file[..cut]), Ok(None), "prefix of {cut}");
        }
        // Trailing bytes do not extend the image.
        let mut two = file.clone();
        two.extend_from_slice(&file);
        assert_eq!(png_frame_length(&two), Ok(Some(file.len())));
    }

    #[test]
    fn png_frame_length_refuses_a_non_png_and_an_absurd_chunk() {
        assert_eq!(
            png_frame_length(b"not a png at all"),
            Err(G2gError::CapsMismatch)
        );
        assert_eq!(
            png_frame_length(&[0x89, b'P', b'X']),
            Err(G2gError::CapsMismatch)
        );
        // A chunk claiming 4 GB, past the encoded-byte ceiling.
        let mut huge = Vec::from(PNG_SIGNATURE);
        huge.extend_from_slice(&u32::MAX.to_be_bytes());
        huge.extend_from_slice(b"IDAT");
        assert_eq!(png_frame_length(&huge), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn png_geometry_reads_the_ihdr() {
        assert_eq!(png_geometry(&png_bytes_sized(640, 480)), Some((640, 480)));
        assert_eq!(png_geometry(&png_bytes_sized(0, 480)), None);
        // No IHDR first, and a file too short to hold one.
        let mut mangled = png_bytes_sized(640, 480);
        mangled[PNG_SIGNATURE.len() + 4..PNG_SIGNATURE.len() + 8].copy_from_slice(b"IDAT");
        assert_eq!(png_geometry(&mangled), None);
        assert_eq!(png_geometry(&PNG_SIGNATURE), None);
    }

    #[test]
    fn jpeg_frame_length_walks_the_markers() {
        let file = jpeg_bytes(64, 48, &[0x11; 100]);
        assert_eq!(jpeg_frame_length(&file), Ok(Some(file.len())));
        for cut in 0..file.len() {
            assert_eq!(jpeg_frame_length(&file[..cut]), Ok(None), "prefix of {cut}");
        }
        // Two concatenated images (an MJPEG dump) frame one at a time.
        let mut two = file.clone();
        two.extend_from_slice(&file);
        assert_eq!(jpeg_frame_length(&two), Ok(Some(file.len())));
    }

    #[test]
    fn jpeg_frame_length_skips_stuffing_and_restarts_in_the_scan() {
        // A stuffed FF, a restart marker, and an FF pair: none of them ends the
        // scan, so the length is still the whole file.
        let scan = [
            0x00,
            JPEG_MARKER_PREFIX,
            0x00,
            JPEG_MARKER_PREFIX,
            JPEG_RESTART_LAST,
            0x42,
        ];
        let file = jpeg_bytes(64, 48, &scan);
        assert_eq!(jpeg_frame_length(&file), Ok(Some(file.len())));
    }

    #[test]
    fn jpeg_frame_length_refuses_a_non_jpeg_and_a_bogus_segment() {
        assert_eq!(
            jpeg_frame_length(b"not a jpeg"),
            Err(G2gError::CapsMismatch)
        );
        assert_eq!(
            jpeg_frame_length(&[0xFF, 0xD7]),
            Err(G2gError::CapsMismatch)
        );
        // A segment length under the length field's own two bytes.
        let bogus = [
            JPEG_MARKER_PREFIX,
            JPEG_START_OF_IMAGE,
            JPEG_MARKER_PREFIX,
            0xE0,
            0x00,
            0x01,
        ];
        assert_eq!(jpeg_frame_length(&bogus), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn jpeg_geometry_reads_the_frame_header() {
        assert_eq!(
            jpeg_geometry(&jpeg_bytes(64, 48, &[0x11; 8])),
            Some((64, 48))
        );
        assert_eq!(jpeg_geometry(&jpeg_bytes(0, 48, &[0x11; 8])), None);
        // No frame header at all: a file that is only SOI + EOI.
        assert_eq!(
            jpeg_geometry(&[
                JPEG_MARKER_PREFIX,
                JPEG_START_OF_IMAGE,
                JPEG_MARKER_PREFIX,
                JPEG_END_OF_IMAGE
            ]),
            None
        );
    }
}

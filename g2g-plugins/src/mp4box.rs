//! Shared fragmented-MP4 box primitives for the MP4 muxer/demuxer elements
//! (`mp4sink`/`mp4src` and their audio counterparts). Writers build
//! size-prefixed boxes; readers walk the box tree. std-gated like its callers.

use alloc::vec::Vec;

use g2g_core::G2gError;

/// Unity 3x3 transform matrix (16.16 / 2.30 fixed point) for `tkhd`/`mvhd`.
pub(crate) const MATRIX: [u32; 9] = [0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000];

// --- writers ---------------------------------------------------------------

/// A size-prefixed box: `[u32 size][4cc kind][payload]`.
pub(crate) fn mp4_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(8 + payload.len());
    b.extend_from_slice(&((payload.len() as u32 + 8).to_be_bytes()));
    b.extend_from_slice(kind);
    b.extend_from_slice(payload);
    b
}

/// A full box: a version byte plus 24-bit flags, then the payload.
pub(crate) fn full_box(kind: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut p = Vec::with_capacity(4 + payload.len());
    p.push(version);
    p.extend_from_slice(&flags.to_be_bytes()[1..]);
    p.extend_from_slice(payload);
    mp4_box(kind, &p)
}

/// The `ftyp` box (iso5/isom brands), identical for the video and audio muxers.
pub(crate) fn ftyp() -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(b"iso5"); // major brand
    p.extend_from_slice(&512u32.to_be_bytes()); // minor version
    p.extend_from_slice(b"iso5");
    p.extend_from_slice(b"isom");
    mp4_box(b"ftyp", &p)
}

// --- readers ---------------------------------------------------------------

pub(crate) fn be32(data: &[u8], at: usize) -> Result<u32, G2gError> {
    data.get(at..at + 4)
        .map(|b| u32::from_be_bytes(b.try_into().expect("4 bytes")))
        .ok_or(G2gError::CapsMismatch)
}

pub(crate) fn be64(data: &[u8], at: usize) -> Result<u64, G2gError> {
    data.get(at..at + 8)
        .map(|b| u64::from_be_bytes(b.try_into().expect("8 bytes")))
        .ok_or(G2gError::CapsMismatch)
}

/// Iterate the child boxes of `data`, yielding `(fourcc, payload)`.
pub(crate) fn boxes(data: &[u8]) -> impl Iterator<Item = (&[u8; 4], &[u8])> {
    let mut i = 0usize;
    core::iter::from_fn(move || {
        if i + 8 > data.len() {
            return None;
        }
        let size = u32::from_be_bytes(data[i..i + 4].try_into().expect("4 bytes")) as usize;
        if size < 8 || i + size > data.len() {
            return None;
        }
        let kind: &[u8; 4] = data[i + 4..i + 8].try_into().expect("4 bytes");
        let payload = &data[i + 8..i + size];
        i += size;
        Some((kind, payload))
    })
}

pub(crate) fn find_box<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    boxes(data).find(|(k, _)| *k == kind).map(|(_, p)| p)
}

/// Descend a path of nested boxes.
pub(crate) fn find_path<'a>(mut data: &'a [u8], path: &[&[u8; 4]]) -> Option<&'a [u8]> {
    for kind in path {
        data = find_box(data, kind)?;
    }
    Some(data)
}

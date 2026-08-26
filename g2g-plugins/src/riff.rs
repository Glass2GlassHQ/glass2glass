//! RIFF chunk walking, shared by the RIFF containers g2g reads and writes:
//! [`wavparse`](crate::wavparse) for `RIFF....WAVE` and
//! [`avidemux`](crate::avidemux) / [`avimux`](crate::avimux) for `RIFF....AVI `.
//!
//! A RIFF file is a 12-byte `RIFF` + size + form-type header, then a flat list
//! of `id` + size + body chunks whose bodies are padded to an even length. A
//! `LIST` chunk's body opens with its own 4-byte form type and then nests more
//! chunks. Every size comes from the file, so [`Chunks`] validates each one
//! against the enclosing range instead of trusting it, and reports a body that
//! overruns rather than indexing past it.

use core::ops::Range;

/// A four-character code: a chunk id, a `LIST` form type, or a codec tag.
pub(crate) type FourCc = [u8; 4];

pub(crate) const FOURCC_LEN: usize = 4;
/// A chunk's 4-byte id plus its 4-byte little-endian size.
pub(crate) const CHUNK_HEADER_LEN: usize = 8;
/// `RIFF` + size + the form type (`WAVE`, `AVI `).
pub(crate) const RIFF_HEADER_LEN: usize = CHUNK_HEADER_LEN + FOURCC_LEN;
/// The outermost chunk id of every RIFF file.
pub(crate) const RIFF_FOURCC: FourCc = *b"RIFF";
/// A chunk whose body nests more chunks behind a form type.
pub(crate) const LIST_FOURCC: FourCc = *b"LIST";

/// A chunk body is padded to an even length. `None` on overflow.
pub(crate) fn padded_len(size: usize) -> Option<usize> {
    size.checked_add(size % 2)
}

pub(crate) fn read_u16(data: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(data.get(at..at + 2)?.try_into().ok()?))
}

pub(crate) fn read_u32(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?))
}

pub(crate) fn read_fourcc(data: &[u8], at: usize) -> Option<FourCc> {
    data.get(at..at + FOURCC_LEN)?.try_into().ok()
}

/// One chunk found by [`Chunks`]: its id and the byte range of its body within
/// the buffer being walked (the padding byte is excluded).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Chunk {
    pub id: FourCc,
    pub body: Range<usize>,
}

impl Chunk {
    /// The chunk's body bytes.
    pub(crate) fn body<'a>(&self, data: &'a [u8]) -> &'a [u8] {
        data.get(self.body.clone()).unwrap_or_default()
    }

    /// The form type of a `LIST` chunk (`hdrl`, `movi`, `INFO`, ...), or `None`
    /// for any other chunk or a `LIST` too short to carry one.
    pub(crate) fn list_form(&self, data: &[u8]) -> Option<FourCc> {
        (self.id == LIST_FOURCC)
            .then(|| read_fourcc(data, self.body.start))
            .flatten()
    }

    /// The range of a `LIST` chunk's nested chunks: its body past the form type.
    pub(crate) fn list_body(&self) -> Range<usize> {
        (self.body.start + FOURCC_LEN).min(self.body.end)..self.body.end
    }
}

/// Walks the chunks in `range` of `data`. Iteration stops at the end of the
/// range, at a trailing run too short to hold another header, or on a chunk
/// whose declared body overruns the range, which sets [`Chunks::overran`].
#[derive(Debug)]
pub(crate) struct Chunks<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
    overran: bool,
}

/// Walk the chunks of `range`, which is clamped to what `data` actually holds.
pub(crate) fn chunks(data: &[u8], range: Range<usize>) -> Chunks<'_> {
    Chunks {
        data,
        pos: range.start.min(data.len()),
        end: range.end.min(data.len()),
        overran: false,
    }
}

impl Chunks<'_> {
    /// True once a chunk declared a body longer than the range holds, so a
    /// caller can fail the parse instead of treating the short walk as the
    /// whole list.
    pub(crate) fn overran(&self) -> bool {
        self.overran
    }
}

impl Iterator for Chunks<'_> {
    type Item = Chunk;

    fn next(&mut self) -> Option<Chunk> {
        let header_end = self.pos.checked_add(CHUNK_HEADER_LEN)?;
        if header_end > self.end {
            return None;
        }
        let id = read_fourcc(self.data, self.pos)?;
        let size = read_u32(self.data, self.pos + FOURCC_LEN)? as usize;
        let body_end = header_end.checked_add(size)?;
        let next = header_end.checked_add(padded_len(size)?)?;
        if body_end > self.end {
            self.overran = true;
            return None;
        }
        self.pos = next;
        Some(Chunk {
            id,
            body: header_end..body_end,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = Vec::from(*id);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        if body.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    #[test]
    fn walks_chunks_and_skips_the_odd_size_pad() {
        let mut data = Vec::new();
        data.extend_from_slice(&chunk(b"fmt ", &[1, 2, 3]));
        data.extend_from_slice(&chunk(b"data", &[4, 5]));
        let found: Vec<Chunk> = chunks(&data, 0..data.len()).collect();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].id, *b"fmt ");
        assert_eq!(found[0].body(&data), &[1, 2, 3]);
        assert_eq!(found[1].id, *b"data");
        assert_eq!(found[1].body(&data), &[4, 5]);
    }

    #[test]
    fn reports_a_body_longer_than_the_range() {
        let mut data = Vec::from(*b"junk");
        data.extend_from_slice(&u32::MAX.to_le_bytes());
        data.extend_from_slice(&[0; 4]);
        let mut walk = chunks(&data, 0..data.len());
        assert_eq!(walk.next(), None);
        assert!(walk.overran(), "a size past the range must be reported");
    }

    #[test]
    fn a_trailing_partial_header_ends_the_walk_cleanly() {
        let mut data = chunk(b"data", &[7]);
        data.extend_from_slice(&[0, 0, 0]);
        let mut walk = chunks(&data, 0..data.len());
        assert_eq!(walk.next().map(|c| c.id), Some(*b"data"));
        assert_eq!(walk.next(), None);
        assert!(!walk.overran());
    }

    #[test]
    fn reads_a_list_form_and_its_nested_range() {
        let mut inner = Vec::from(*b"hdrl");
        inner.extend_from_slice(&chunk(b"avih", &[9; 4]));
        let data = chunk(b"LIST", &inner);
        let list = chunks(&data, 0..data.len()).next().expect("the LIST");
        assert_eq!(list.list_form(&data), Some(*b"hdrl"));
        let nested: Vec<Chunk> = chunks(&data, list.list_body()).collect();
        assert_eq!(nested.len(), 1);
        assert_eq!(nested[0].id, *b"avih");
    }
}

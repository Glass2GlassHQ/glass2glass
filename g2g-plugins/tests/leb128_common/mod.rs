#![allow(dead_code)] // no one test file uses both directions

/// LEB128 unsigned (AV1 4.10.5) at `offset`: the value and the bytes it occupied.
pub(crate) fn read_leb128(stream: &[u8], offset: usize) -> Option<(u64, usize)> {
    let mut value = 0u64;
    let mut p = offset;
    for i in 0..8 {
        let b = *stream.get(p)?;
        p += 1;
        value |= ((b & 0x7f) as u64) << (7 * i);
        if b & 0x80 == 0 {
            return Some((value, p - offset));
        }
    }
    None
}

/// `v` as LEB128 bytes.
pub(crate) fn leb128(mut v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if v == 0 {
            break;
        }
    }
    out
}

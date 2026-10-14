fn be_array<const N: usize>(data: &[u8], at: usize) -> Option<[u8; N]> {
    data.get(at..at.checked_add(N)?)?.try_into().ok()
}

pub(crate) fn be_u16(data: &[u8], at: usize) -> Option<u16> {
    be_array(data, at).map(u16::from_be_bytes)
}

pub(crate) fn be_u24(data: &[u8], at: usize) -> Option<u32> {
    be_array::<3>(data, at).map(|[high, mid, low]| u32::from_be_bytes([0, high, mid, low]))
}

pub(crate) fn be_u32(data: &[u8], at: usize) -> Option<u32> {
    be_array(data, at).map(u32::from_be_bytes)
}

#[cfg(feature = "std")]
pub(crate) fn be_u64(data: &[u8], at: usize) -> Option<u64> {
    be_array(data, at).map(u64::from_be_bytes)
}

/// Read a big-endian `u16` at `*pos` and step past it.
pub(crate) fn take_be_u16(data: &[u8], pos: &mut usize) -> Option<u16> {
    let value = be_u16(data, *pos)?;
    *pos += 2;
    Some(value)
}

/// Read a big-endian `u32` at `*pos` and step past it.
pub(crate) fn take_be_u32(data: &[u8], pos: &mut usize) -> Option<u32> {
    let value = be_u32(data, *pos)?;
    *pos += 4;
    Some(value)
}

/// Read a big-endian `u64` at `*pos` and step past it.
#[cfg(feature = "dash")]
pub(crate) fn take_be_u64(data: &[u8], pos: &mut usize) -> Option<u64> {
    let value = be_u64(data, *pos)?;
    *pos += 8;
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_stop_at_the_end_instead_of_panicking() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05];
        assert_eq!(be_u24(&data, 2), Some(0x03_04_05));
        assert_eq!(be_u32(&data, 1), Some(0x02_03_04_05));
        assert_eq!(be_u32(&data, 2), None);
        assert_eq!(be_u16(&data, usize::MAX), None);
        let mut pos = 3;
        assert_eq!(take_be_u16(&data, &mut pos), Some(0x04_05));
        assert_eq!(pos, 5);
        assert_eq!(take_be_u16(&data, &mut pos), None);
        assert_eq!(pos, 5, "a short read leaves the cursor");
    }
}

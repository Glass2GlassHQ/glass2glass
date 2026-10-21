// the keystream runs on across the ranges, as over one buffer of their bytes in order
pub(crate) fn apply_aes128_ctr(
    buf: &mut [u8],
    ranges: impl IntoIterator<Item = (usize, usize)>,
    key: &[u8; 16],
    counter_block: &[u8; 16],
) {
    use aes::cipher::{KeyIvInit, StreamCipher};
    let mut cipher = ctr::Ctr128BE::<aes::Aes128>::new(key.into(), counter_block.into());
    for (start, end) in ranges {
        cipher.apply_keystream(&mut buf[start..end]);
    }
}

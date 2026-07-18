//! MPEG-CENC `cbcs` sample decryption, shared by the HLS fMP4 path
//! ([`Fmp4Demux`](crate::fmp4demux), `hls` feature) and the multi-track MP4
//! demuxer ([`Mp4DemuxN`](crate::mp4demuxn), `mp4-cenc` feature). Pure AES-128-CBC
//! pattern decryption over a sample's `senc` subsample map; the key is supplied by
//! the caller (an HLS `#EXT-X-KEY` or an app clear key), the constant IV and
//! crypt/skip pattern come from the init segment's `tenc`.

use alloc::vec::Vec;

use crate::fmp4::Subsample;

/// Decrypt one cbcs sample in place: walk its `senc` subsamples, decrypting each
/// protected range (an empty map means the whole sample is one protected range).
pub(crate) fn cbcs_decrypt_sample(
    buf: &mut [u8],
    subsamples: &[Subsample],
    key: &[u8; 16],
    iv: &[u8; 16],
    crypt: u8,
    skip: u8,
) {
    if subsamples.is_empty() {
        decrypt_protected_range(buf, key, iv, crypt, skip);
        return;
    }
    let mut pos = 0usize;
    for s in subsamples {
        pos = (pos + s.clear as usize).min(buf.len());
        let end = (pos + s.protected as usize).min(buf.len());
        if pos < end {
            decrypt_protected_range(&mut buf[pos..end], key, iv, crypt, skip);
        }
        pos = end;
    }
}

/// cbcs pattern decrypt over one protected range: AES-128-CBC the encrypted
/// 16-byte blocks (a `crypt`:`skip` block pattern, or every block when either is
/// zero), the IV reset to the constant IV at the range start, CBC chaining across
/// the encrypted blocks only. A trailing partial block is left clear.
fn decrypt_protected_range(range: &mut [u8], key: &[u8; 16], iv: &[u8; 16], crypt: u8, skip: u8) {
    use aes::cipher::{block_padding::NoPadding, BlockDecryptMut, KeyIvInit};
    type Dec = cbc::Decryptor<aes::Aes128>;

    let block_count = range.len() / 16;
    let offsets: Vec<usize> = if crypt == 0 || skip == 0 {
        (0..block_count).map(|b| b * 16).collect()
    } else {
        let span = (crypt + skip) as usize;
        (0..block_count)
            .filter(|b| b % span < crypt as usize)
            .map(|b| b * 16)
            .collect()
    };
    if offsets.is_empty() {
        return;
    }
    let mut gathered: Vec<u8> = offsets
        .iter()
        .flat_map(|&o| range[o..o + 16].iter().copied())
        .collect();
    Dec::new(&(*key).into(), &(*iv).into())
        .decrypt_padded_mut::<NoPadding>(&mut gathered)
        .expect("cbcs region is block-aligned");
    for (i, &o) in offsets.iter().enumerate() {
        range[o..o + 16].copy_from_slice(&gathered[i * 16..i * 16 + 16]);
    }
}

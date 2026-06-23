//! SRT payload encryption (`srt` feature): AES-128-CTR over the data payload,
//! with the Stream Encrypting Key (SEK) derived/exchanged the SRT way:
//!
//! - A passphrase is stretched into a Key Encrypting Key (KEK) with
//!   PBKDF2-HMAC-SHA1 (2048 iterations), salted by the bottom 8 bytes of the
//!   stream salt.
//! - The random SEK is wrapped with the KEK using RFC 3394 AES key wrap and
//!   carried in the KM (Keying Material) message in the handshake KMREQ/KMRSP
//!   extension. The receiver unwraps it with the same passphrase.
//! - Each data packet's payload is AES-CTR encrypted; the 128-bit counter block
//!   is the salt with the packet sequence XORed in, so encrypt and decrypt are
//!   the same keystream operation (haicrypt-style construction).
//!
//! v1 is one AES-128 even key, no rekeying. The wire layout follows the SRT KM
//! message so it is a faithful basis for real-peer interop; that interop is
//! unverified here (the sandbox blocks external peers), as with the rest of the
//! SRT element. Validated g2g <-> g2g.

use alloc::vec::Vec;

use aes::Aes128;
use aes_kw::Kek;
use ctr::cipher::{KeyIvInit, StreamCipher};
use hmac::Hmac;
use sha1::Sha1;

/// AES-128: 16-byte SEK / KEK / salt.
const KEY_LEN: usize = 16;
const SALT_LEN: usize = 16;
/// PBKDF2 iteration count SRT uses for the passphrase -> KEK stretch.
const KEK_ITERS: u32 = 2048;
/// KM message signature ("HAI" marker) and constants (SRT Keying Material).
const KM_SIGN: u16 = 0x2029;
const KM_VERS_PT: u8 = 0x12; // Vers=1, PT=2 (KM message)
const KM_KK_EVEN: u8 = 0x01; // even key only
const KM_CIPHER_CTR: u8 = 2; // AES-CTR
const KM_SE_DATA: u8 = 2; // data stream element
/// Fixed KM header length before the salt; salt then wrapped key follow.
const KM_HDR_LEN: usize = 16;
/// AES-CTR with a big-endian 128-bit counter (the full IV is the start block).
type Aes128Ctr = ctr::Ctr128BE<Aes128>;

/// Per-stream cipher state: the symmetric key + salt both peers share after the
/// KM exchange.
#[derive(Clone)]
pub struct SrtCrypto {
    sek: [u8; KEY_LEN],
    salt: [u8; SALT_LEN],
}

impl core::fmt::Debug for SrtCrypto {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Never print key material.
        f.debug_struct("SrtCrypto").finish_non_exhaustive()
    }
}

impl SrtCrypto {
    /// Construct from an explicit key + salt (a test injects fixed values for
    /// determinism; the sender uses [`generate`](Self::generate)).
    pub fn new(sek: [u8; KEY_LEN], salt: [u8; SALT_LEN]) -> Self {
        Self { sek, salt }
    }

    /// Generate a fresh random stream key + salt from the OS RNG (the caller
    /// side of an encrypted stream).
    pub fn generate() -> Self {
        let mut sek = [0u8; KEY_LEN];
        let mut salt = [0u8; SALT_LEN];
        getrandom::getrandom(&mut sek).expect("OS RNG for the SRT stream key");
        getrandom::getrandom(&mut salt).expect("OS RNG for the SRT stream salt");
        Self { sek, salt }
    }

    /// Encrypt or decrypt `buf` in place for packet `seq` (AES-CTR is symmetric:
    /// the same call both ways).
    pub fn process(&self, seq: u32, buf: &mut [u8]) {
        let iv = self.counter_block(seq);
        let mut cipher = Aes128Ctr::new((&self.sek).into(), (&iv).into());
        cipher.apply_keystream(buf);
    }

    /// The 128-bit AES-CTR start counter: the salt's first 14 bytes, with the
    /// packet sequence XORed into bytes 10..14, and a 2-byte block counter (0).
    fn counter_block(&self, seq: u32) -> [u8; 16] {
        let mut ctr = [0u8; 16];
        ctr[..14].copy_from_slice(&self.salt[..14]);
        let seqb = seq.to_be_bytes();
        for i in 0..4 {
            ctr[10 + i] ^= seqb[i];
        }
        ctr
    }

    /// Build the KM (Keying Material) message: the fixed header, the salt, and
    /// the SEK wrapped with the passphrase-derived KEK (RFC 3394). This is the
    /// payload of the KMREQ/KMRSP handshake extension.
    pub fn build_km(&self, passphrase: &str) -> Vec<u8> {
        let kek = derive_kek(passphrase, &self.salt);
        let wrapped = Kek::<Aes128>::from(kek)
            .wrap_vec(&self.sek)
            .expect("AES-KW wrap of a 16-byte key always succeeds");

        let mut km = Vec::with_capacity(KM_HDR_LEN + SALT_LEN + wrapped.len());
        km.push(KM_VERS_PT);
        km.extend_from_slice(&KM_SIGN.to_be_bytes());
        km.push(KM_KK_EVEN);
        km.extend_from_slice(&0u32.to_be_bytes()); // KEKI
        km.push(KM_CIPHER_CTR);
        km.push(0); // Auth
        km.push(KM_SE_DATA);
        km.push(0); // Resv2
        km.extend_from_slice(&0u16.to_be_bytes()); // Resv3
        km.push((SALT_LEN / 4) as u8); // SLen/4
        km.push((KEY_LEN / 4) as u8); // KLen/4
        km.extend_from_slice(&self.salt);
        km.extend_from_slice(&wrapped);
        km
    }

    /// Parse a KM message and unwrap the SEK with `passphrase`, yielding the
    /// shared cipher state. `None` on a malformed message or a passphrase that
    /// fails to unwrap (the integrity check in AES key unwrap).
    pub fn from_km(km: &[u8], passphrase: &str) -> Option<Self> {
        if km.len() < KM_HDR_LEN + SALT_LEN || km[0] != KM_VERS_PT {
            return None;
        }
        if u16::from_be_bytes(km[1..3].try_into().ok()?) != KM_SIGN {
            return None;
        }
        let slen = km[14] as usize * 4;
        let klen = km[15] as usize * 4;
        if slen != SALT_LEN || klen != KEY_LEN {
            return None; // v1 supports only AES-128 + a 16-byte salt
        }
        let salt: [u8; SALT_LEN] = km[KM_HDR_LEN..KM_HDR_LEN + SALT_LEN].try_into().ok()?;
        let wrapped = km.get(KM_HDR_LEN + SALT_LEN..)?;
        if wrapped.len() != KEY_LEN + 8 {
            return None;
        }
        let kek = derive_kek(passphrase, &salt);
        let sek_vec = Kek::<Aes128>::from(kek).unwrap_vec(wrapped).ok()?;
        let sek: [u8; KEY_LEN] = sek_vec.as_slice().try_into().ok()?;
        Some(Self { sek, salt })
    }
}

/// Derive the Key Encrypting Key from the passphrase via PBKDF2-HMAC-SHA1,
/// salted by the bottom 8 bytes of the stream salt (the SRT/haicrypt rule).
fn derive_kek(passphrase: &str, salt: &[u8; SALT_LEN]) -> [u8; KEY_LEN] {
    let mut kek = [0u8; KEY_LEN];
    pbkdf2::pbkdf2::<Hmac<Sha1>>(passphrase.as_bytes(), &salt[SALT_LEN - 8..], KEK_ITERS, &mut kek)
        .expect("PBKDF2 into a 16-byte buffer never fails on length");
    kek
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn ctr_round_trips_a_payload() {
        let c = SrtCrypto::new([7u8; 16], [3u8; 16]);
        let plain = b"the quick brown fox jumps".to_vec();
        let mut buf = plain.clone();
        c.process(42, &mut buf);
        assert_ne!(buf, plain, "ciphertext differs from plaintext");
        c.process(42, &mut buf);
        assert_eq!(buf, plain, "decrypt (same op, same seq) recovers the plaintext");
    }

    #[test]
    fn different_seq_gives_different_keystream() {
        let c = SrtCrypto::new([7u8; 16], [3u8; 16]);
        let mut a = vec![0u8; 32];
        let mut b = vec![0u8; 32];
        c.process(1, &mut a);
        c.process(2, &mut b);
        assert_ne!(a, b, "the sequence number diversifies the counter block");
    }

    #[test]
    fn km_round_trips_the_key_under_the_passphrase() {
        let sender = SrtCrypto::new([0xABu8; 16], [0xCDu8; 16]);
        let km = sender.build_km("s3cret");
        let recv = SrtCrypto::from_km(&km, "s3cret").expect("unwrap with the right passphrase");
        // The recovered cipher must produce the same keystream.
        let mut x = vec![1u8, 2, 3, 4, 5];
        let mut y = x.clone();
        sender.process(9, &mut x);
        recv.process(9, &mut y);
        assert_eq!(x, y, "receiver derived the same SEK + salt");
    }

    #[test]
    fn km_rejects_a_wrong_passphrase() {
        let km = SrtCrypto::new([0xABu8; 16], [0xCDu8; 16]).build_km("right");
        assert!(SrtCrypto::from_km(&km, "wrong").is_none(), "AES-KW integrity check fails");
    }
}

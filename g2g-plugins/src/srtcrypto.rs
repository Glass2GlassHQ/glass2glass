//! SRT payload encryption (`srt` feature): AES-CTR over the data payload (AES-128
//! or AES-256), with the Stream Encrypting Key (SEK) derived/exchanged the SRT way:
//!
//! - A passphrase is stretched into a Key Encrypting Key (KEK) with
//!   PBKDF2-HMAC-SHA1 (2048 iterations), salted by the bottom 8 bytes of the
//!   stream salt. The KEK length matches the cipher (128 or 256 bits).
//! - The random SEK is wrapped with the KEK using RFC 3394 AES key wrap and
//!   carried in the KM (Keying Material) message in the handshake KMREQ/KMRSP
//!   extension. The receiver unwraps it with the same passphrase.
//! - Each data packet's payload is AES-CTR encrypted; the 128-bit counter block
//!   is the salt with the packet sequence XORed in, so encrypt and decrypt are
//!   the same keystream operation (haicrypt-style construction).
//!
//! The KM carries the key length (`KLen`) and a parity flag (`KK`: even / odd),
//! so AES-128 and AES-256 interoperate off the same wire layout, and the parity
//! is what mid-stream rekeying ([`crate::srtsink`]) toggles. Real-peer interop is
//! unverified here (the sandbox blocks external peers); validated g2g <-> g2g.

use alloc::vec::Vec;

use aes::{Aes128, Aes256};
use aes_kw::Kek;
use ctr::cipher::{KeyIvInit, StreamCipher};
use hmac::Hmac;
use sha1::Sha1;

const SALT_LEN: usize = 16;
/// PBKDF2 iteration count SRT uses for the passphrase -> KEK stretch.
const KEK_ITERS: u32 = 2048;
/// KM message signature ("HAI" marker) and constants (SRT Keying Material).
const KM_SIGN: u16 = 0x2029;
const KM_VERS_PT: u8 = 0x12; // Vers=1, PT=2 (KM message)
/// KM key-parity flags (`KK`): which of the two key slots the SEK fills. SRT
/// alternates even / odd across a rekey so in-flight packets under the old key
/// still decrypt.
pub const KM_KK_EVEN: u8 = 0x01;
pub const KM_KK_ODD: u8 = 0x02;
const KM_CIPHER_CTR: u8 = 2; // AES-CTR
const KM_SE_DATA: u8 = 2; // data stream element
/// Fixed KM header length before the salt; salt then wrapped key follow.
const KM_HDR_LEN: usize = 16;
/// AES-CTR with a big-endian 128-bit counter (the full IV is the start block).
type Aes128Ctr = ctr::Ctr128BE<Aes128>;
type Aes256Ctr = ctr::Ctr128BE<Aes256>;

/// The stream cipher key size. AES-128 stays the default; AES-256 is opt-in
/// (`SrtSink::with_aes256`) and negotiated through the KM `KLen` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AesKeySize {
    Aes128,
    Aes256,
}

impl AesKeySize {
    /// Key length in bytes (16 or 32).
    pub fn bytes(self) -> usize {
        match self {
            AesKeySize::Aes128 => 16,
            AesKeySize::Aes256 => 32,
        }
    }
}

/// The Stream Encrypting Key, sized to the negotiated cipher.
#[derive(Clone)]
enum Sek {
    Aes128([u8; 16]),
    Aes256([u8; 32]),
}

impl Sek {
    fn as_bytes(&self) -> &[u8] {
        match self {
            Sek::Aes128(k) => k,
            Sek::Aes256(k) => k,
        }
    }
}

/// Per-stream cipher state: the symmetric key + salt both peers share after the
/// KM exchange.
#[derive(Clone)]
pub struct SrtCrypto {
    sek: Sek,
    salt: [u8; SALT_LEN],
}

impl core::fmt::Debug for SrtCrypto {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Never print key material.
        f.debug_struct("SrtCrypto").finish_non_exhaustive()
    }
}

impl SrtCrypto {
    /// Construct an AES-128 cipher from an explicit key + salt (a test injects
    /// fixed values for determinism; the sender uses [`generate`](Self::generate)).
    pub fn new(sek: [u8; 16], salt: [u8; SALT_LEN]) -> Self {
        Self { sek: Sek::Aes128(sek), salt }
    }

    /// Construct an AES-256 cipher from an explicit 32-byte key + salt.
    pub fn new_aes256(sek: [u8; 32], salt: [u8; SALT_LEN]) -> Self {
        Self { sek: Sek::Aes256(sek), salt }
    }

    /// Generate a fresh random stream key (of `size`) + salt from the OS RNG (the
    /// caller side of an encrypted stream).
    pub fn generate(size: AesKeySize) -> Self {
        let mut salt = [0u8; SALT_LEN];
        getrandom::getrandom(&mut salt).expect("OS RNG for the SRT stream salt");
        let sek = match size {
            AesKeySize::Aes128 => {
                let mut k = [0u8; 16];
                getrandom::getrandom(&mut k).expect("OS RNG for the SRT stream key");
                Sek::Aes128(k)
            }
            AesKeySize::Aes256 => {
                let mut k = [0u8; 32];
                getrandom::getrandom(&mut k).expect("OS RNG for the SRT stream key");
                Sek::Aes256(k)
            }
        };
        Self { sek, salt }
    }

    /// Encrypt or decrypt `buf` in place for packet `seq` (AES-CTR is symmetric:
    /// the same call both ways). The counter block is key-size independent (a
    /// 128-bit CTR block); only the key schedule differs between AES-128/256.
    pub fn process(&self, seq: u32, buf: &mut [u8]) {
        let iv = self.counter_block(seq);
        match &self.sek {
            Sek::Aes128(k) => Aes128Ctr::new(k.into(), (&iv).into()).apply_keystream(buf),
            Sek::Aes256(k) => Aes256Ctr::new(k.into(), (&iv).into()).apply_keystream(buf),
        }
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

    /// Build the KM (Keying Material) message for key parity `kk` (`KM_KK_EVEN`
    /// / `KM_KK_ODD`): the fixed header, the salt, and the SEK wrapped with the
    /// passphrase-derived KEK (RFC 3394). This is the payload of the KMREQ/KMRSP
    /// handshake extension and of a mid-stream rekey control packet. `KLen`
    /// records the key size so AES-128/256 are told apart on unwrap.
    pub fn build_km(&self, passphrase: &str, kk: u8) -> Vec<u8> {
        let key_len = self.sek.as_bytes().len();
        let wrapped = match &self.sek {
            Sek::Aes128(k) => Kek::<Aes128>::from(derive_kek::<16>(passphrase, &self.salt))
                .wrap_vec(k)
                .expect("AES-KW wrap of a 16-byte key always succeeds"),
            Sek::Aes256(k) => Kek::<Aes256>::from(derive_kek::<32>(passphrase, &self.salt))
                .wrap_vec(k)
                .expect("AES-KW wrap of a 32-byte key always succeeds"),
        };

        let mut km = Vec::with_capacity(KM_HDR_LEN + SALT_LEN + wrapped.len());
        km.push(KM_VERS_PT);
        km.extend_from_slice(&KM_SIGN.to_be_bytes());
        km.push(kk);
        km.extend_from_slice(&0u32.to_be_bytes()); // KEKI
        km.push(KM_CIPHER_CTR);
        km.push(0); // Auth
        km.push(KM_SE_DATA);
        km.push(0); // Resv2
        km.extend_from_slice(&0u16.to_be_bytes()); // Resv3
        km.push((SALT_LEN / 4) as u8); // SLen/4
        km.push((key_len / 4) as u8); // KLen/4
        km.extend_from_slice(&self.salt);
        km.extend_from_slice(&wrapped);
        km
    }

    /// The key parity (`KM_KK_EVEN` / `KM_KK_ODD`) a KM message carries, without
    /// unwrapping it. The receiver uses this to file a rekey into the right slot.
    pub fn km_kk(km: &[u8]) -> Option<u8> {
        if km.len() < KM_HDR_LEN + SALT_LEN || km[0] != KM_VERS_PT {
            return None;
        }
        Some(km[3])
    }

    /// Parse a KM message and unwrap the SEK with `passphrase`, yielding the
    /// shared cipher state (AES-128 or AES-256 per the `KLen` field). `None` on a
    /// malformed message or a passphrase that fails to unwrap (the integrity
    /// check in AES key unwrap).
    pub fn from_km(km: &[u8], passphrase: &str) -> Option<Self> {
        if km.len() < KM_HDR_LEN + SALT_LEN || km[0] != KM_VERS_PT {
            return None;
        }
        if u16::from_be_bytes(km[1..3].try_into().ok()?) != KM_SIGN {
            return None;
        }
        let slen = km[14] as usize * 4;
        let klen = km[15] as usize * 4;
        if slen != SALT_LEN {
            return None; // a 16-byte salt is the only layout we emit
        }
        let salt: [u8; SALT_LEN] = km[KM_HDR_LEN..KM_HDR_LEN + SALT_LEN].try_into().ok()?;
        let wrapped = km.get(KM_HDR_LEN + SALT_LEN..)?;
        if wrapped.len() != klen + 8 {
            return None; // wrapped key is the SEK plus the 8-byte AES-KW IV
        }
        let sek = match klen {
            16 => {
                let v = Kek::<Aes128>::from(derive_kek::<16>(passphrase, &salt))
                    .unwrap_vec(wrapped)
                    .ok()?;
                Sek::Aes128(v.as_slice().try_into().ok()?)
            }
            32 => {
                let v = Kek::<Aes256>::from(derive_kek::<32>(passphrase, &salt))
                    .unwrap_vec(wrapped)
                    .ok()?;
                Sek::Aes256(v.as_slice().try_into().ok()?)
            }
            _ => return None, // only AES-128 / AES-256
        };
        Some(Self { sek, salt })
    }
}

/// Derive an `N`-byte Key Encrypting Key from the passphrase via PBKDF2-HMAC-SHA1,
/// salted by the bottom 8 bytes of the stream salt (the SRT/haicrypt rule). The
/// KEK length matches the cipher (16 for AES-128, 32 for AES-256).
fn derive_kek<const N: usize>(passphrase: &str, salt: &[u8; SALT_LEN]) -> [u8; N] {
    let mut kek = [0u8; N];
    pbkdf2::pbkdf2::<Hmac<Sha1>>(passphrase.as_bytes(), &salt[SALT_LEN - 8..], KEK_ITERS, &mut kek)
        .expect("PBKDF2 into a fixed buffer never fails on length");
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
        let km = sender.build_km("s3cret", KM_KK_EVEN);
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
        let km = SrtCrypto::new([0xABu8; 16], [0xCDu8; 16]).build_km("right", KM_KK_EVEN);
        assert!(SrtCrypto::from_km(&km, "wrong").is_none(), "AES-KW integrity check fails");
    }

    #[test]
    fn aes256_ctr_round_trips_a_payload() {
        let c = SrtCrypto::new_aes256([9u8; 32], [4u8; 16]);
        let plain = b"the quick brown fox jumps over".to_vec();
        let mut buf = plain.clone();
        c.process(17, &mut buf);
        assert_ne!(buf, plain, "AES-256 ciphertext differs from plaintext");
        c.process(17, &mut buf);
        assert_eq!(buf, plain, "AES-256 decrypt recovers the plaintext");
    }

    #[test]
    fn aes256_km_round_trips_a_32_byte_key() {
        // The KM carries KLen, so the receiver recovers a 32-byte key and the same
        // AES-256 keystream, distinct from an AES-128 key under the same salt.
        let sender = SrtCrypto::new_aes256([0x5Au8; 32], [0xC3u8; 16]);
        let km = sender.build_km("p@ss", KM_KK_ODD);
        assert_eq!(km[15], 32 / 4, "KLen field records the 256-bit key");
        assert_eq!(SrtCrypto::km_kk(&km), Some(KM_KK_ODD), "parity flag preserved");
        let recv = SrtCrypto::from_km(&km, "p@ss").expect("unwrap the AES-256 key");
        let mut x = vec![7u8; 24];
        let mut y = x.clone();
        sender.process(3, &mut x);
        recv.process(3, &mut y);
        assert_eq!(x, y, "receiver derived the same AES-256 SEK");
    }

    #[test]
    fn aes256_km_rejects_a_wrong_passphrase() {
        let km = SrtCrypto::new_aes256([0x5Au8; 32], [0xC3u8; 16]).build_km("right", KM_KK_EVEN);
        assert!(SrtCrypto::from_km(&km, "wrong").is_none(), "AES-256 AES-KW integrity check fails");
    }
}

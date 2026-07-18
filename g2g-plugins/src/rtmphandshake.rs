//! RTMP "genuine FMS/FP" digest handshake (M521), the HMAC-SHA256 complex
//! handshake strict CDNs (Adobe FMS, some Wowza / Facebook Live configs) require
//! in place of the plain byte-echo simple handshake.
//!
//! Both C1 (client) and S1 (server) embed a 32-byte HMAC-SHA256 digest at a
//! data-dependent offset (one of two schemes); the peer proves it knows the
//! shared "genuine" key by computing that digest and, in C2 / S2, a signature
//! keyed off the other side's digest. A server that validates the handshake
//! rejects a connection whose C1 digest does not verify, which is why the simple
//! handshake fails against it.
//!
//! Reference: the well-known librtmp / ffmpeg handshake. Keys and offset schemes
//! are the published constants; nothing here is g2g-specific.
//!
//! This module is `rtmp`-feature gated because it pulls the HMAC-SHA256 crypto
//! (`hmac` + `sha2`) and OS randomness (`getrandom`); the sans-IO RTMP core
//! stays crypto-free for the `no_std` baseline and uses the simple handshake.

use hmac::{Hmac, Mac};
use sha2::Sha256;

/// A handshake signature block (C1 / S1 / C2 / S2) is 1536 bytes.
pub const SIG_SIZE: usize = 1536;
/// The HMAC-SHA256 digest embedded in a signature block is 32 bytes.
const DIGEST_LEN: usize = 32;

/// The 32-byte magic suffix shared by both genuine keys.
const KEY_SUFFIX: [u8; 32] = [
    0xf0, 0xee, 0xc2, 0x4a, 0x80, 0x68, 0xbe, 0xe8, 0x2e, 0x00, 0xd0, 0xd1, 0x02, 0x9e, 0x7e, 0x57,
    0x6e, 0xec, 0x5d, 0x2d, 0x29, 0x80, 0x6f, 0xab, 0x93, 0xb8, 0xe6, 0x36, 0xcf, 0xeb, 0x31, 0xae,
];
/// The client (Flash Player) key: the 30-byte name + the shared suffix (62 bytes).
/// The C1 digest keys on the first 30 bytes; the C2 response keys on all 62.
const FP_NAME: &[u8] = b"Genuine Adobe Flash Player 001";
/// The server (Flash Media Server) key: the 36-byte name + the suffix (68 bytes).
/// The S1 digest keys on the first 36 bytes; the S2 response keys on all 68.
const FMS_NAME: &[u8] = b"Genuine Adobe Flash Media Server 001";

fn fp_key() -> [u8; 62] {
    let mut k = [0u8; 62];
    k[..30].copy_from_slice(FP_NAME);
    k[30..].copy_from_slice(&KEY_SUFFIX);
    k
}

fn fms_key() -> [u8; 68] {
    let mut k = [0u8; 68];
    k[..36].copy_from_slice(FMS_NAME);
    k[36..].copy_from_slice(&KEY_SUFFIX);
    k
}

/// HMAC-SHA256 over the concatenation of `parts`, keyed by `key`.
fn hmac(key: &[u8], parts: &[&[u8]]) -> [u8; DIGEST_LEN] {
    let mut mac = <Hmac<Sha256>>::new_from_slice(key).expect("HMAC accepts any key length");
    for p in parts {
        mac.update(p);
    }
    mac.finalize().into_bytes().into()
}

/// Digest offset scheme 0: four little counter bytes at [8..12] pick a slot in
/// the first data block.
fn digest_offset_scheme0(buf: &[u8]) -> usize {
    let sum = buf[8] as usize + buf[9] as usize + buf[10] as usize + buf[11] as usize;
    (sum % 728) + 12
}

/// Digest offset scheme 1: four counter bytes at [772..776] pick a slot in the
/// second data block.
fn digest_offset_scheme1(buf: &[u8]) -> usize {
    let sum = buf[772] as usize + buf[773] as usize + buf[774] as usize + buf[775] as usize;
    (sum % 728) + 776
}

const SCHEMES: [fn(&[u8]) -> usize; 2] = [digest_offset_scheme0, digest_offset_scheme1];

/// The digest over `buf` with the 32 bytes at `offset` excluded (the value the
/// digest at that offset must equal).
fn digest_excluding(buf: &[u8], offset: usize, key: &[u8]) -> [u8; DIGEST_LEN] {
    hmac(key, &[&buf[..offset], &buf[offset + DIGEST_LEN..]])
}

/// Fill `buf` with OS randomness (falling back to a data-dependent pattern if
/// the OS RNG is unavailable; the random block is not security-critical, only
/// the keyed digest is).
fn fill_random(buf: &mut [u8]) {
    if getrandom::getrandom(buf).is_err() {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(31).wrapping_add(7);
        }
    }
}

/// Build a client C1 carrying a valid digest (scheme 1). `time` is the client
/// uptime field; the version field is set non-zero to request the complex
/// handshake.
pub fn build_c1(time: u32) -> [u8; SIG_SIZE] {
    let mut c1 = [0u8; SIG_SIZE];
    c1[0..4].copy_from_slice(&time.to_be_bytes());
    // A non-zero client version signals the digest handshake (a Flash Player
    // build number; the exact value is not validated, only its presence).
    c1[4..8].copy_from_slice(&[0x80, 0x00, 0x07, 0x02]);
    fill_random(&mut c1[8..]);
    let off = digest_offset_scheme1(&c1);
    let key = fp_key();
    let digest = digest_excluding(&c1, off, &key[..30]);
    c1[off..off + DIGEST_LEN].copy_from_slice(&digest);
    c1
}

/// Locate and return the peer's 32-byte digest in `sig`, verifying it against
/// `name_key` (the first 30 / 36 bytes of the FP / FMS key) under whichever
/// scheme validates. `None` means no valid digest is present (the peer used the
/// simple handshake).
fn find_digest(sig: &[u8], name_key: &[u8]) -> Option<(usize, [u8; DIGEST_LEN])> {
    if sig.len() < SIG_SIZE {
        return None;
    }
    for scheme in SCHEMES {
        let off = scheme(sig);
        if off + DIGEST_LEN > SIG_SIZE {
            continue;
        }
        let expect = digest_excluding(sig, off, name_key);
        if expect == sig[off..off + DIGEST_LEN] {
            let mut d = [0u8; DIGEST_LEN];
            d.copy_from_slice(&sig[off..off + DIGEST_LEN]);
            return Some((off, d));
        }
    }
    None
}

/// Build the client C2 responding to the server's `s1`. Returns `None` if `s1`
/// carries no valid FMS digest (the server spoke the simple handshake), so the
/// caller falls back to echoing S1 as C2.
pub fn build_c2(s1: &[u8]) -> Option<[u8; SIG_SIZE]> {
    let fms = fms_key();
    let (_, server_digest) = find_digest(s1, &fms[..36])?;
    // Key the C2 signature off the server's digest using the full FP key.
    let key = hmac(&fp_key(), &[&server_digest]);
    let mut c2 = [0u8; SIG_SIZE];
    fill_random(&mut c2);
    let sig = hmac(&key, &[&c2[..SIG_SIZE - DIGEST_LEN]]);
    c2[SIG_SIZE - DIGEST_LEN..].copy_from_slice(&sig);
    Some(c2)
}

/// Build a server S1 carrying a valid FMS digest (scheme 1).
pub fn build_s1(time: u32) -> [u8; SIG_SIZE] {
    let mut s1 = [0u8; SIG_SIZE];
    s1[0..4].copy_from_slice(&time.to_be_bytes());
    s1[4..8].copy_from_slice(&[0x0d, 0x0e, 0x0a, 0x0d]);
    fill_random(&mut s1[8..]);
    let off = digest_offset_scheme1(&s1);
    let key = fms_key();
    let digest = digest_excluding(&s1, off, &key[..36]);
    s1[off..off + DIGEST_LEN].copy_from_slice(&digest);
    s1
}

/// Build the server S2 responding to the client's `c1`. Returns `None` if `c1`
/// carries no valid FP digest (the client spoke the simple handshake), so the
/// caller falls back to echoing C1 as S2.
pub fn build_s2(c1: &[u8]) -> Option<[u8; SIG_SIZE]> {
    let fp = fp_key();
    let (_, client_digest) = find_digest(c1, &fp[..30])?;
    // Key the S2 signature off the client's digest using the full FMS key.
    let key = hmac(&fms_key(), &[&client_digest]);
    let mut s2 = [0u8; SIG_SIZE];
    fill_random(&mut s2);
    let sig = hmac(&key, &[&s2[..SIG_SIZE - DIGEST_LEN]]);
    s2[SIG_SIZE - DIGEST_LEN..].copy_from_slice(&sig);
    Some(s2)
}

/// Whether `c1` carries a valid client (FP) digest, i.e. requests the complex
/// handshake. A server uses this to choose between the complex and simple reply.
pub fn c1_has_digest(c1: &[u8]) -> bool {
    let fp = fp_key();
    find_digest(c1, &fp[..30]).is_some()
}

/// Verify the peer's C2 / S2 response `resp` proves knowledge of `our_digest`
/// (the digest we embedded in our own S1 / C1), keyed by the peer's full key
/// (`peer_full_key` = FP key when we are the server validating C2, FMS key when
/// we are the client validating S2). Used to prove the round-trip in tests and
/// by a validating server.
pub fn verify_response(resp: &[u8], our_digest: &[u8], peer_full_key: &[u8]) -> bool {
    if resp.len() < SIG_SIZE {
        return false;
    }
    let key = hmac(peer_full_key, &[our_digest]);
    let expect = hmac(&key, &[&resp[..SIG_SIZE - DIGEST_LEN]]);
    expect == resp[SIG_SIZE - DIGEST_LEN..SIG_SIZE]
}

/// The digest we embedded in a signature block we built (scheme 1), for a later
/// `verify_response`. Returns the 32-byte digest at the scheme-1 offset.
pub fn own_digest_scheme1(sig: &[u8]) -> [u8; DIGEST_LEN] {
    let off = digest_offset_scheme1(sig);
    let mut d = [0u8; DIGEST_LEN];
    d.copy_from_slice(&sig[off..off + DIGEST_LEN]);
    d
}

/// The full genuine keys, exposed for callers that verify a peer response.
pub fn genuine_fp_key() -> [u8; 62] {
    fp_key()
}
pub fn genuine_fms_key() -> [u8; 68] {
    fms_key()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_have_the_published_lengths() {
        assert_eq!(FP_NAME.len(), 30, "FP name is 30 bytes");
        assert_eq!(FMS_NAME.len(), 36, "FMS name is 36 bytes");
        assert_eq!(fp_key().len(), 62);
        assert_eq!(fms_key().len(), 68);
    }

    #[test]
    fn c1_embeds_a_self_consistent_fp_digest() {
        // A strict server recomputes the C1 digest over C1-minus-digest with the
        // FP name key and rejects the connection unless it matches. Prove ours
        // matches (the whole point of the complex handshake).
        let c1 = build_c1(0);
        let fp = fp_key();
        assert!(c1_has_digest(&c1), "C1 carries a valid FP digest");
        let (off, embedded) = find_digest(&c1, &fp[..30]).expect("valid digest present");
        let recomputed = digest_excluding(&c1, off, &fp[..30]);
        assert_eq!(
            embedded, recomputed,
            "embedded digest matches recomputation"
        );
    }

    #[test]
    fn simple_handshake_block_has_no_digest() {
        // A byte-echo simple S1 (deterministic pattern, no keyed digest) must not
        // spuriously validate, so build_c2 falls back to echo against it.
        let mut simple = [0u8; SIG_SIZE];
        for (i, b) in simple.iter_mut().enumerate().skip(8) {
            *b = (i & 0xFF) as u8;
        }
        assert!(
            build_c2(&simple).is_none(),
            "no FMS digest => C2 falls back to echo"
        );
        assert!(!c1_has_digest(&simple), "no FP digest in a simple block");
    }

    #[test]
    fn full_client_server_digest_round_trip_validates() {
        // Emulate exactly what a genuine-FMS server + Flash client verify:
        //   client C1 --(FP digest)--> server validates, builds S1(FMS digest)+S2
        //   server S1/S2 --> client validates S1, builds C2 keyed off S1 digest
        //   server validates C2 against the client digest it saw in C1.
        let c1 = build_c1(0);
        // Server: validate client C1 and build its reply.
        assert!(c1_has_digest(&c1), "server accepts the complex C1");
        let s1 = build_s1(123);
        let s2 = build_s2(&c1).expect("server builds a digest S2 from a digest C1");

        // Client: build C2 from S1 (finds the FMS digest, signs with FP full key).
        let c2 = build_c2(&s1).expect("client builds a digest C2 from a digest S1");

        // Client verifies the server's S2 proves knowledge of the client's C1
        // digest, using the full FMS key.
        let client_c1_digest = own_digest_scheme1(&c1);
        assert!(
            verify_response(&s2, &client_c1_digest, &genuine_fms_key()),
            "server S2 proves it validated our C1 digest"
        );
        // Server verifies the client's C2 proves knowledge of the server's S1
        // digest, using the full FP key.
        let server_s1_digest = own_digest_scheme1(&s1);
        assert!(
            verify_response(&c2, &server_s1_digest, &genuine_fp_key()),
            "client C2 proves it validated the server S1 digest"
        );
    }
}

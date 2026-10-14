use alloc::string::String;
use alloc::vec::Vec;

const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const BASE64_PAD: char = '=';
const BITS_PER_SEXTET: u32 = 6;
const BITS_PER_BYTE: u32 = 8;
const SEXTET_MASK: u32 = 0x3F;
const NIBBLE_MASK: u8 = 0x0F;

pub fn hex_encode(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(HEX_DIGITS[usize::from(byte >> 4)] as char);
        text.push(HEX_DIGITS[usize::from(byte & NIBBLE_MASK)] as char);
    }
    text
}

pub fn hex_decode(text: &str) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    hex_decode_into(text, &mut bytes)?;
    Some(bytes)
}

// reserved up front so a caller zeroizing a key never leaves a reallocated copy
pub fn hex_decode_into(text: &str, out: &mut Vec<u8>) -> Option<()> {
    let (pairs, remainder) = text.as_bytes().as_chunks::<2>();
    if !remainder.is_empty() {
        return None;
    }
    out.reserve(pairs.len());
    for [high, low] in pairs {
        out.push(hex_digit(*high)? << 4 | hex_digit(*low)?);
    }
    Some(())
}

const fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

pub fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let group = (u32::from(chunk[0]) << 16)
            | (u32::from(chunk.get(1).copied().unwrap_or(0)) << 8)
            | u32::from(chunk.get(2).copied().unwrap_or(0));
        for sextet in 0..4 {
            // A chunk of n bytes codes n+1 characters, the rest is padding.
            out.push(if sextet <= chunk.len() {
                let shift = 18 - BITS_PER_SEXTET * sextet as u32;
                BASE64_ALPHABET[((group >> shift) & SEXTET_MASK) as usize] as char
            } else {
                BASE64_PAD
            });
        }
    }
    out
}

// padding optional, a truncated final group is rejected
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    fn sextet(c: u8) -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a') as u32 + 26,
            b'0'..=b'9' => (c - b'0') as u32 + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        })
    }
    let bytes: &[u8] = s.trim_end_matches(BASE64_PAD).as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for &c in bytes {
        acc = (acc << BITS_PER_SEXTET) | sextet(c)?;
        bits += BITS_PER_SEXTET;
        if bits >= BITS_PER_BYTE {
            bits -= BITS_PER_BYTE;
            out.push((acc >> bits) as u8);
        }
    }
    // A leftover group of one sextet (2 bits short of a byte) is truncated input.
    if bits >= BITS_PER_SEXTET {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips_and_rejects_bad_text() {
        const BYTES: [u8; 4] = [0x00, 0x9F, 0xFF, 0x10];
        assert_eq!(hex_encode(&BYTES), "009fff10");
        assert_eq!(hex_decode("009FFF10").unwrap(), BYTES);
        assert_eq!(hex_decode("009"), None, "odd length");
        assert_eq!(hex_decode("00zz"), None, "not a digit");
        assert_eq!(hex_decode("\u{e9}0"), None, "non-ASCII");
    }

    // checked against the spec rather than against the decoder
    #[test]
    fn base64_matches_the_rfc_4648_vectors() {
        for (input, expected) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(
                base64_encode(input.as_bytes()),
                expected,
                "base64 of {input:?}"
            );
        }
    }

    #[test]
    fn base64_round_trips_every_padding_length() {
        for input in ["", "h", "hi", "hi!", "hi!!"] {
            let encoded = base64_encode(input.as_bytes());
            assert_eq!(encoded.len() % 4, 0, "padded to whole groups");
            assert_eq!(base64_decode(&encoded).unwrap(), input.as_bytes());
        }
        assert_eq!(base64_encode(b"hi"), "aGk=");
        assert_eq!(base64_decode("aGk"), Some(b"hi".to_vec()));
        assert_eq!(base64_decode("a"), None, "a lone sextet is truncated");
        assert_eq!(base64_decode("a b"), None, "space is outside the alphabet");
    }
}

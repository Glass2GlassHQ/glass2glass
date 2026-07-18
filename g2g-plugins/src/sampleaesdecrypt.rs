//! HLS SAMPLE-AES sample decryptor (`hls` feature), the post-demux half of HLS
//! encryption. Unlike whole-segment AES-128 (decrypted by [`HlsSrc`](crate::hlssrc)
//! before the demuxer), SAMPLE-AES encrypts only the media samples *inside* the
//! container, leaving the PES / codec framing in the clear, so it must run after
//! the demuxer, per access unit: `tsdemux ! sampleaesdecrypt ! h264parse`.
//!
//! Scope: MPEG-2 TS H.264 (Annex-B) and AAC (ADTS), per the Apple "MPEG-2 Stream
//! Encryption Format for HTTP Live Streaming" spec, cross-checked against hls.js
//! and FFmpeg. The key and IV are configured on the element. Auto-wiring the
//! playlist `#EXT-X-KEY` material from `HlsSrc` through the demuxer, fMP4 `cbcs`,
//! and AC-3 are follow-ups (DESIGN_TODO).
//!
//! H.264: for each slice NAL (type 1 / 5) longer than 48 bytes, the
//! emulation-prevention bytes are stripped, the first 32 bytes stay clear, then a
//! 16-encrypted / 144-clear block pattern is AES-128-CBC decrypted (chaining over
//! the encrypted blocks only, IV reset per NAL), and the bytes are re-escaped.
//! AAC: per ADTS frame, the header plus 16 leader bytes stay clear and the whole
//! 16-byte blocks after them are CBC decrypted (IV reset per frame).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;
use std::sync::{Arc, Mutex};

use aes::cipher::{block_padding::NoPadding, BlockDecryptMut, KeyIvInit};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, ConfigureOutcome, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, VideoCodec,
};

use crate::annexb::{add_emulation_prevention, next_start_code, strip_emulation_prevention};

type Aes128CbcDec = cbc::Decryptor<aes::Aes128>;

/// AES-128-CBC decrypt a block-aligned buffer in place, no padding (the
/// SAMPLE-AES protected region is always a whole number of 16-byte blocks).
fn cbc_decrypt_blocks(buf: &mut [u8], key: &[u8; 16], iv: &[u8; 16]) {
    Aes128CbcDec::new(&(*key).into(), &(*iv).into())
        .decrypt_padded_mut::<NoPadding>(buf)
        .expect("SAMPLE-AES region is block-aligned");
}

/// Which codec's sample-encryption rule to apply, resolved from the input caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Codec {
    H264,
    Aac,
}

/// SAMPLE-AES key material: the 16-byte AES key (the one `#EXT-X-KEY` names) and
/// the constant segment IV, reset at each NAL unit / audio frame.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SampleAesKey {
    pub key: [u8; 16],
    pub iv: [u8; 16],
}

// Redact the key/IV from Debug so secrets don't leak into logs.
impl core::fmt::Debug for SampleAesKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SampleAesKey").finish_non_exhaustive()
    }
}

/// Shared slot a key publisher ([`HlsSrc`](crate::hlssrc)) writes once it has
/// fetched the playlist `#EXT-X-KEY` material, and the decryptor reads. `None`
/// until the publisher fills it; this is the auto-wiring path that spares the
/// caller from configuring the key by hand.
pub type SampleAesKeyHandle = Arc<Mutex<Option<SampleAesKey>>>;

/// A fresh, empty key handle to wire a publisher and a decryptor together.
pub fn new_key_handle() -> SampleAesKeyHandle {
    Arc::new(Mutex::new(None))
}

pub struct SampleAesDecrypt {
    /// Directly configured key (the [`new`](Self::new) path).
    key: Option<SampleAesKey>,
    /// Shared key source (the [`from_key_handle`](Self::from_key_handle) path);
    /// takes precedence when set.
    key_handle: Option<SampleAesKeyHandle>,
    codec: Option<Codec>,
    configured: bool,
}

// Redact the key/IV from Debug so secrets don't leak into logs.
impl core::fmt::Debug for SampleAesDecrypt {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SampleAesDecrypt")
            .field("codec", &self.codec)
            .field("configured", &self.configured)
            .finish_non_exhaustive()
    }
}

impl SampleAesDecrypt {
    /// Decrypt with a key known up front.
    pub fn new(key: [u8; 16], iv: [u8; 16]) -> Self {
        Self {
            key: Some(SampleAesKey { key, iv }),
            key_handle: None,
            codec: None,
            configured: false,
        }
    }

    /// Decrypt with the key a publisher writes into the shared `handle` at
    /// runtime. A frame that arrives before the handle is filled passes through
    /// unchanged (in the HLS chain `HlsSrc` fills it before pushing any bytes).
    pub fn from_key_handle(handle: SampleAesKeyHandle) -> Self {
        Self {
            key: None,
            key_handle: Some(handle),
            codec: None,
            configured: false,
        }
    }

    /// The key in effect: the shared handle if wired, else the direct key.
    fn current_key(&self) -> Option<SampleAesKey> {
        match &self.key_handle {
            Some(handle) => *handle.lock().expect("sample-aes key handle poisoned"),
            None => self.key,
        }
    }
}

impl AsyncElement for SampleAesDecrypt {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }
            | Caps::Audio {
                format: AudioFormat::Aac,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.codec = match absolute_caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            } => Some(Codec::H264),
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            } => Some(Codec::Aac),
            _ => return Err(G2gError::CapsMismatch),
        };
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let key = self.current_key();
                    let decrypted = match (&frame.domain, self.codec, key) {
                        (MemoryDomain::System(s), Some(Codec::H264), Some(k)) => {
                            Some(decrypt_avc(s.as_slice(), &k.key, &k.iv))
                        }
                        (MemoryDomain::System(s), Some(Codec::Aac), Some(k)) => {
                            Some(decrypt_aac(s.as_slice(), &k.key, &k.iv))
                        }
                        // No key yet (or non-system domain): forward unchanged.
                        _ => None,
                    };
                    let frame = match decrypted {
                        Some(bytes) => Frame {
                            domain: MemoryDomain::System(SystemSlice::from_boxed(
                                bytes.into_boxed_slice(),
                            )),
                            ..frame
                        },
                        None => frame,
                    };
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // EOS is forwarded by the runner's transform arm, not here.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Decrypt the SAMPLE-AES-protected slice NALs of an Annex-B access unit,
/// preserving the original start codes and all non-slice / short NALs verbatim.
fn decrypt_avc(au: &[u8], key: &[u8; 16], iv: &[u8; 16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(au.len());
    let mut pos = 0;
    let mut found = false;
    while let Some((_, nal_start)) = next_start_code(au, pos) {
        found = true;
        // Copy any leading bytes plus the start code itself, unchanged.
        out.extend_from_slice(&au[pos..nal_start]);
        let nal_end = next_start_code(au, nal_start).map_or(au.len(), |(sc, _)| sc);
        out.extend_from_slice(&decrypt_avc_nal(&au[nal_start..nal_end], key, iv));
        pos = nal_end;
    }
    if !found {
        return au.to_vec();
    }
    out
}

/// One NAL: encrypted only for slice types (1 non-IDR, 5 IDR) over 48 bytes.
fn decrypt_avc_nal(nal: &[u8], key: &[u8; 16], iv: &[u8; 16]) -> Vec<u8> {
    let nal_type = nal.first().map_or(0, |b| b & 0x1F);
    if nal.len() <= 48 || (nal_type != 1 && nal_type != 5) {
        return nal.to_vec();
    }
    let mut rbsp = strip_emulation_prevention(nal);
    decrypt_avc_pattern(&mut rbsp, key, iv);
    add_emulation_prevention(&rbsp)
}

/// Apply the 16-encrypted / 144-clear pattern from offset 32 of the de-escaped
/// NAL. CBC chains across the encrypted blocks only; the IV is the segment IV.
fn decrypt_avc_pattern(buf: &mut [u8], key: &[u8; 16], iv: &[u8; 16]) {
    let mut block_offsets = Vec::new();
    let mut pos = 32;
    while buf.len().saturating_sub(pos) > 16 {
        block_offsets.push(pos);
        pos += 16;
        pos += core::cmp::min(144, buf.len() - pos);
    }
    if block_offsets.is_empty() {
        return;
    }
    let mut gathered = Vec::with_capacity(block_offsets.len() * 16);
    for &b in &block_offsets {
        gathered.extend_from_slice(&buf[b..b + 16]);
    }
    cbc_decrypt_blocks(&mut gathered, key, iv);
    for (i, &b) in block_offsets.iter().enumerate() {
        buf[b..b + 16].copy_from_slice(&gathered[i * 16..i * 16 + 16]);
    }
}

/// Decrypt the SAMPLE-AES-protected AAC frames in an ADTS buffer (a PES payload
/// may carry several). Non-ADTS or trailing bytes pass through unchanged.
fn decrypt_aac(buf: &[u8], key: &[u8; 16], iv: &[u8; 16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(buf.len());
    let mut pos = 0;
    while pos + 7 <= buf.len() {
        // ADTS syncword is 12 set bits.
        if buf[pos] != 0xFF || buf[pos + 1] & 0xF0 != 0xF0 {
            break;
        }
        let header_len = if buf[pos + 1] & 0x01 == 1 { 7 } else { 9 };
        let frame_len = ((buf[pos + 3] as usize & 0x03) << 11)
            | ((buf[pos + 4] as usize) << 3)
            | ((buf[pos + 5] as usize & 0xE0) >> 5);
        if frame_len < header_len || pos + frame_len > buf.len() {
            break;
        }
        out.extend_from_slice(&decrypt_aac_frame(
            &buf[pos..pos + frame_len],
            header_len,
            key,
            iv,
        ));
        pos += frame_len;
    }
    out.extend_from_slice(&buf[pos..]);
    out
}

/// One ADTS frame: header + 16 leader bytes clear, whole 16-byte blocks after
/// them CBC decrypted with the segment IV.
fn decrypt_aac_frame(frame: &[u8], header_len: usize, key: &[u8; 16], iv: &[u8; 16]) -> Vec<u8> {
    let enc_start = header_len + 16;
    let block_count = frame.len().saturating_sub(enc_start) / 16;
    if block_count == 0 {
        return frame.to_vec();
    }
    let mut out = frame.to_vec();
    cbc_decrypt_blocks(&mut out[enc_start..enc_start + block_count * 16], key, iv);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes::cipher::{BlockEncryptMut, KeyIvInit};
    use alloc::vec;

    const KEY: [u8; 16] = *b"sample-aes-key!!";
    const IV: [u8; 16] = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6];

    type Aes128CbcEnc = cbc::Encryptor<aes::Aes128>;

    fn cbc_encrypt_blocks(buf: &mut [u8], key: &[u8; 16], iv: &[u8; 16]) {
        let n = buf.len();
        Aes128CbcEnc::new(&(*key).into(), &(*iv).into())
            .encrypt_padded_mut::<NoPadding>(buf, n)
            .unwrap();
    }

    // The encrypt side mirrors decrypt exactly, so the round-trip exercises the
    // real decrypt path against independently produced ciphertext.
    fn encrypt_avc_pattern(buf: &mut [u8], key: &[u8; 16], iv: &[u8; 16]) {
        let mut offsets = Vec::new();
        let mut pos = 32;
        while buf.len().saturating_sub(pos) > 16 {
            offsets.push(pos);
            pos += 16;
            pos += core::cmp::min(144, buf.len() - pos);
        }
        if offsets.is_empty() {
            return;
        }
        let mut gathered = Vec::with_capacity(offsets.len() * 16);
        for &b in &offsets {
            gathered.extend_from_slice(&buf[b..b + 16]);
        }
        cbc_encrypt_blocks(&mut gathered, key, iv);
        for (i, &b) in offsets.iter().enumerate() {
            buf[b..b + 16].copy_from_slice(&gathered[i * 16..i * 16 + 16]);
        }
    }

    fn encrypt_avc_nal(nal: &[u8], key: &[u8; 16], iv: &[u8; 16]) -> Vec<u8> {
        let t = nal[0] & 0x1F;
        if nal.len() <= 48 || (t != 1 && t != 5) {
            return nal.to_vec();
        }
        let mut rbsp = strip_emulation_prevention(nal);
        encrypt_avc_pattern(&mut rbsp, key, iv);
        add_emulation_prevention(&rbsp)
    }

    fn encrypt_avc(au: &[u8], key: &[u8; 16], iv: &[u8; 16]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut pos = 0;
        while let Some((_, nal_start)) = next_start_code(au, pos) {
            out.extend_from_slice(&au[pos..nal_start]);
            let nal_end = next_start_code(au, nal_start).map_or(au.len(), |(sc, _)| sc);
            out.extend_from_slice(&encrypt_avc_nal(&au[nal_start..nal_end], key, iv));
            pos = nal_end;
        }
        out
    }

    /// Build a NAL of `nal_type` whose RBSP is `header_byte` + `len` payload
    /// bytes, escaped to a canonical bytestream NAL (so strip/add round-trips).
    fn make_nal(nal_type: u8, len: usize) -> Vec<u8> {
        let mut rbsp = vec![nal_type & 0x1F];
        // Seed some 0x00 0x00 runs in the clear leader so emulation-prevention
        // is exercised, then varied bytes for the encrypted region.
        rbsp.extend_from_slice(&[0, 0, 1, 0, 0, 2, 0, 0, 3]);
        rbsp.extend((0..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add(7)));
        add_emulation_prevention(&rbsp)
    }

    fn annexb(nals: &[Vec<u8>]) -> Vec<u8> {
        let mut au = Vec::new();
        for nal in nals {
            au.extend_from_slice(&[0, 0, 0, 1]);
            au.extend_from_slice(nal);
        }
        au
    }

    #[test]
    fn avc_idr_slice_roundtrips_through_decrypt() {
        let cleartext = annexb(&[make_nal(5, 300), make_nal(1, 200)]);
        let ciphertext = encrypt_avc(&cleartext, &KEY, &IV);
        assert_ne!(ciphertext, cleartext, "encryption must change the bytes");
        let recovered = decrypt_avc(&ciphertext, &KEY, &IV);
        assert_eq!(
            recovered, cleartext,
            "SAMPLE-AES AVC decrypt recovers the cleartext AU"
        );
    }

    #[test]
    fn avc_pattern_encrypts_only_the_spec_offsets() {
        // Pin the SAMPLE-AES AVC pattern to literal spec offsets, independent of
        // the offset loop the encrypt oracle and decrypt share: a 32-byte clear
        // leader, then a 16-byte encrypted block every 160 bytes, the trailing
        // short run left clear. For a 452-byte de-escaped RBSP that is exactly
        // blocks at 32, 192, 352. A geometry bug shared by both sides would still
        // round-trip cleanly, so the round-trip test alone cannot catch it; this
        // checks the actual encrypted positions against the hand-derived spec.
        let n = 452usize;
        let clear: Vec<u8> = (0..n)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
            .collect();
        let mut buf = clear.clone();
        encrypt_avc_pattern(&mut buf, &KEY, &IV);

        const ENCRYPTED: [usize; 3] = [32, 192, 352];
        for off in 0..n {
            let in_block = ENCRYPTED.iter().any(|&b| off >= b && off < b + 16);
            if !in_block {
                assert_eq!(
                    buf[off], clear[off],
                    "byte {off} outside the pattern must stay clear"
                );
            }
        }
        for &b in &ENCRYPTED {
            assert_ne!(
                buf[b..b + 16],
                clear[b..b + 16],
                "the spec block at {b} must be encrypted"
            );
        }
    }

    #[test]
    fn avc_leaves_short_and_non_slice_nals_clear() {
        // SPS (type 7), a short slice (<= 48), and a parameter-set NAL must pass
        // through untouched.
        let sps = make_nal(7, 300);
        let short_slice = annexb(&[vec![0x65, 1, 2, 3]]); // type 5 but only 4 bytes
        let aud = make_nal(9, 100);
        let au = annexb(&[sps.clone(), aud.clone()]);
        let unchanged = encrypt_avc(&au, &KEY, &IV);
        assert_eq!(unchanged, au, "non-slice NALs are never encrypted");
        assert_eq!(
            decrypt_avc(&short_slice, &KEY, &IV),
            short_slice,
            "<=48-byte slice is clear"
        );
    }

    fn adts_frame(payload: &[u8]) -> Vec<u8> {
        let total = 7 + payload.len();
        let mut frame = vec![
            0xFF,
            0xF1, // syncword + MPEG-4 + protection_absent=1 (7-byte header)
            0x50,
            (((total >> 11) & 0x03) as u8),
            (((total >> 3) & 0xFF) as u8),
            ((((total & 0x07) << 5) as u8) | 0x1F),
            0xFC,
        ];
        frame.extend_from_slice(payload);
        frame
    }

    fn encrypt_aac_frame(
        frame: &[u8],
        header_len: usize,
        key: &[u8; 16],
        iv: &[u8; 16],
    ) -> Vec<u8> {
        let enc_start = header_len + 16;
        let blocks = frame.len().saturating_sub(enc_start) / 16;
        if blocks == 0 {
            return frame.to_vec();
        }
        let mut out = frame.to_vec();
        cbc_encrypt_blocks(&mut out[enc_start..enc_start + blocks * 16], key, iv);
        out
    }

    #[test]
    fn aac_adts_frames_roundtrip_through_decrypt() {
        let payload0: Vec<u8> = (0..80u32).map(|i| (i % 251) as u8).collect();
        let payload1: Vec<u8> = (0..67u32).map(|i| (i % 233) as u8 ^ 0x44).collect();
        let clear0 = adts_frame(&payload0);
        let clear1 = adts_frame(&payload1);
        let cleartext = [clear0.clone(), clear1.clone()].concat();

        let cipher = [
            encrypt_aac_frame(&clear0, 7, &KEY, &IV),
            encrypt_aac_frame(&clear1, 7, &KEY, &IV),
        ]
        .concat();
        assert_ne!(cipher, cleartext, "encryption must change the bytes");

        let recovered = decrypt_aac(&cipher, &KEY, &IV);
        assert_eq!(
            recovered, cleartext,
            "both ADTS frames decrypt back, IV reset per frame"
        );
    }

    #[test]
    fn aac_short_frame_is_left_clear() {
        // Payload <= 16 leaves no whole block to encrypt.
        let frame = adts_frame(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(decrypt_aac(&frame, &KEY, &IV), frame);
    }

    // -- element-level: drive process() like the chain would ----------------

    use core::pin::Pin;
    use g2g_core::frame::FrameTiming;
    use g2g_core::{Dim, PushOutcome, Rate};

    #[derive(Default)]
    struct RecordingSink {
        frames: Vec<Vec<u8>>,
    }
    impl OutputSink for RecordingSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(f) = packet {
                    if let MemoryDomain::System(s) = &f.domain {
                        self.frames.push(s.as_slice().to_vec());
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    #[tokio::test]
    async fn element_decrypts_avc_data_frame() {
        let cleartext = annexb(&[make_nal(5, 256)]);
        let ciphertext = encrypt_avc(&cleartext, &KEY, &IV);

        let mut elem = SampleAesDecrypt::new(KEY, IV);
        elem.configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        })
        .unwrap();

        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(ciphertext.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        };
        let mut sink = RecordingSink::default();
        elem.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        assert_eq!(
            sink.frames,
            vec![cleartext],
            "element emits the decrypted access unit"
        );
    }

    #[tokio::test]
    async fn element_reads_key_from_shared_handle() {
        let cleartext = annexb(&[make_nal(5, 256)]);
        let ciphertext = encrypt_avc(&cleartext, &KEY, &IV);

        // Publisher fills the handle (as HlsSrc would) before the frame flows.
        let handle = new_key_handle();
        *handle.lock().unwrap() = Some(SampleAesKey { key: KEY, iv: IV });

        let mut elem = SampleAesDecrypt::from_key_handle(handle);
        elem.configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        })
        .unwrap();

        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(ciphertext.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        };
        let mut sink = RecordingSink::default();
        elem.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        assert_eq!(
            sink.frames,
            vec![cleartext],
            "key from the shared handle decrypts the AU"
        );
    }

    #[tokio::test]
    async fn empty_handle_forwards_unchanged() {
        let bytes = annexb(&[make_nal(5, 256)]);
        let mut elem = SampleAesDecrypt::from_key_handle(new_key_handle());
        elem.configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        })
        .unwrap();
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.clone().into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        };
        let mut sink = RecordingSink::default();
        elem.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        assert_eq!(
            sink.frames,
            vec![bytes],
            "no key in the handle: pass through untouched"
        );
    }

    #[test]
    fn rejects_non_h264_non_aac_caps() {
        let mut elem = SampleAesDecrypt::new(KEY, IV);
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert!(matches!(
            elem.configure_pipeline(&vp9),
            Err(G2gError::CapsMismatch)
        ));
    }
}

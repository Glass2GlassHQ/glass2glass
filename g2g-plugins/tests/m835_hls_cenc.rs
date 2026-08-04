//! M835 HLS / CMAF Common Encryption: mid-stream `#EXT-X-KEY` rotation, `cbcs`
//! audio, per-sample IVs (`cenc`), `cens` pattern AES-CTR (M867),
//! `saiz`/`saio`-located sample auxiliary information and `seig` sample groups at
//! both fragment and movie level.
//!
//! Two kinds of vector. The `cenc` (AES-CTR) fixtures are authored by ffmpeg
//! (`-encryption_scheme cenc-aes-ctr`), the strongest reference available: this
//! host's ffmpeg cannot author `cbcs`, `cens` or SAMPLE-AES HLS at all (its mp4
//! muxer offers only `none` and `cenc-aes-ctr`), so those are hand-built box by
//! box with an encrypt oracle that mirrors the spec, and the decrypted output is
//! checked against the clear source bytes either way.
//!
//! Network-free: the rotation tests serve their playlists and segments from an
//! in-process `TcpListener`, like the other HLS suites.

#![cfg(all(feature = "hls", feature = "mp4-cenc"))]

use core::future::Future;
use core::pin::Pin;

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use aes::cipher::{block_padding::NoPadding, BlockEncryptMut, KeyIvInit};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    G2gError, MemoryDomain, MultiOutputElement, MultiOutputSink, OutputSink, PipelinePacket,
    PushOutcome,
};
use g2g_plugins::cenc::{new_key_handle, CencKeyHandle};
use g2g_plugins::hlssrc::HlsSrc;
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};

/// The content key the ffmpeg fixtures are encrypted under (a test vector, not a
/// secret: it is the value passed to `-encryption_key` when authoring them).
const FFMPEG_KEY: [u8; 16] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
];
/// The matching `-encryption_kid`, which lands in the fixtures' `tenc`.
const FFMPEG_KID: [u8; 16] = [
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
];

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

// --- test sinks ------------------------------------------------------------

#[derive(Default)]
struct PortCapture {
    frames: Vec<Vec<Vec<u8>>>,
}

impl PortCapture {
    fn new(ports: usize) -> Self {
        Self {
            frames: vec![Vec::new(); ports],
        }
    }
}

impl MultiOutputSink for PortCapture {
    fn port_count(&self) -> usize {
        self.frames.len()
    }

    fn push_to<'a>(
        &'a mut self,
        port: usize,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames[port].push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn byte_frame(bytes: Vec<u8>) -> PipelinePacket {
    use g2g_core::frame::{Frame, FrameTiming};
    use g2g_core::memory::SystemSlice;
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

/// Demux a whole fMP4 through `Mp4DemuxN`, returning each port's samples.
/// `keys`, when given, is the store the encrypted file's keys come from.
async fn demux_all(
    file: Vec<u8>,
    keys: Option<CencKeyHandle>,
) -> Result<Vec<Vec<Vec<u8>>>, G2gError> {
    let streams = forwardable_streams(&file);
    if streams.is_empty() {
        // Unparsable protection metadata sinks the whole track at init parse.
        return Err(G2gError::CapsMismatch);
    }
    let ports: Vec<Mp4Port> = streams
        .iter()
        .map(|s| Mp4Port {
            track_id: s.track_id,
            caps: s.caps.clone(),
        })
        .collect();
    let n = ports.len();
    let mut demux = Mp4DemuxN::new(ports);
    if let Some(keys) = keys {
        demux = demux.with_cenc_key_handle(keys);
    }
    let mut out = PortCapture::new(n);
    demux.process(byte_frame(file), &mut out).await?;
    demux.process(PipelinePacket::Eos, &mut out).await?;
    Ok(out.frames)
}

/// The elementary payloads of demuxed AAC samples: `Mp4DemuxN` frames each one
/// with a 7-byte ADTS header, which is framing, not sample data.
fn adts_payloads(frames: &[Vec<u8>]) -> Vec<Vec<u8>> {
    frames
        .iter()
        .map(|f| {
            assert_eq!(&f[..2], &[0xFF, 0xF1], "AAC frames come out as ADTS");
            f[7..].to_vec()
        })
        .collect()
}

/// A key store holding one content key under `kid`.
fn store_for_kid(kid: [u8; 16], key: [u8; 16]) -> CencKeyHandle {
    let handle = new_key_handle();
    handle.lock().unwrap().insert_kid(kid, key);
    handle
}

// --- ffmpeg-authored `cenc` (AES-CTR) vectors ------------------------------

/// H.264 with subsample encryption: ffmpeg writes a per-sample 8-byte IV and a
/// subsample map (clear NAL headers, protected payload) into `senc`, and locates
/// it with `saiz` (per-sample sizes) + `saio` (an offset from the `moof` start).
/// Decrypting must reproduce the clear file's access units byte for byte.
#[tokio::test]
async fn ffmpeg_cenc_video_saiz_saio_decrypts_to_the_clear_samples() {
    let clear = demux_all(fixture("cenc_h264_clear.mp4"), None)
        .await
        .expect("clear demux");
    let keys = store_for_kid(FFMPEG_KID, FFMPEG_KEY);
    let got = demux_all(fixture("cenc_h264_ctr.mp4"), Some(keys))
        .await
        .expect("encrypted demux");

    assert!(!clear[0].is_empty(), "the fixture carries video samples");
    assert_eq!(
        got[0], clear[0],
        "cenc AES-CTR decrypt reproduces the clear access units"
    );
}

/// AAC: ffmpeg's `cenc` audio track has no subsample map at all (the whole
/// sample is protected) and a uniform `saiz` `default_sample_info_size` of 8, the
/// per-sample IV. This is the audio shape of the aux-info path.
#[tokio::test]
async fn ffmpeg_cenc_audio_full_sample_decrypts_to_the_clear_samples() {
    let clear = demux_all(fixture("cenc_aac_clear.mp4"), None)
        .await
        .expect("clear demux");
    let keys = store_for_kid(FFMPEG_KID, FFMPEG_KEY);
    let got = demux_all(fixture("cenc_aac_ctr.mp4"), Some(keys))
        .await
        .expect("encrypted demux");

    assert!(!clear[0].is_empty(), "the fixture carries audio samples");
    assert_eq!(
        got[0], clear[0],
        "cenc AES-CTR decrypt reproduces the clear AAC samples"
    );
}

/// The wrong key must not silently yield plausible output: with a key registered
/// under a KID the file never names, the sample stays undecryptable and the parse
/// fails loud rather than emitting ciphertext as media.
#[tokio::test]
async fn unknown_kid_fails_loud() {
    let keys = store_for_kid([0xAB; 16], FFMPEG_KEY);
    assert_eq!(
        demux_all(fixture("cenc_aac_ctr.mp4"), Some(keys)).await,
        Err(G2gError::CapsMismatch),
        "no key for the sample's KID: fail, never emit ciphertext",
    );
}

// --- hand-built `cbcs` vectors ---------------------------------------------
//
// ffmpeg's mp4 muxer only authors `cenc-aes-ctr`, so the `cbcs` cases (what HLS
// SAMPLE-AES maps to for fMP4) and the aux-info / sample-group edge cases are
// built box by box here, with an encrypt oracle that follows the spec: whole-
// sample AES-CBC for audio (no pattern, no clear leader, trailing partial block
// left clear) and a 1:9 crypt:skip pattern for video.

const CBCS_KEY: [u8; 16] = *b"m835-cbcs-key!!!";
const CBCS_IV: [u8; 16] = [0x5A; 16];
const CBCS_KID: [u8; 16] = [0x31; 16];
const TIMESCALE: u32 = 48_000;

type CbcEnc = cbc::Encryptor<aes::Aes128>;

fn mp4_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut b = (payload.len() as u32 + 8).to_be_bytes().to_vec();
    b.extend_from_slice(kind);
    b.extend_from_slice(payload);
    b
}

fn full_box(kind: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut p = vec![version];
    p.extend_from_slice(&flags.to_be_bytes()[1..]);
    p.extend_from_slice(payload);
    mp4_box(kind, &p)
}

/// Encrypt the whole-block prefix of `sample` in place under AES-CBC, the shape
/// `cbcs` audio uses (pattern 0:0). The trailing partial block stays clear.
fn cbcs_encrypt_whole(sample: &[u8], key: &[u8; 16], iv: &[u8; 16]) -> Vec<u8> {
    let mut out = sample.to_vec();
    let blocks = (out.len() / 16) * 16;
    if blocks == 0 {
        return out;
    }
    CbcEnc::new(key.into(), iv.into())
        .encrypt_padded_mut::<NoPadding>(&mut out[..blocks], blocks)
        .expect("block aligned");
    out
}

/// A `tenc` with the given pattern: either constant-IV (the `cbcs` shape) or
/// per-sample-IV (`cenc` / `cens`, where each `senc` record carries a 16-byte IV).
fn tenc(kid: [u8; 16], crypt: u8, skip: u8, per_sample_iv: bool) -> Vec<u8> {
    let iv_size = if per_sample_iv { 16 } else { 0 };
    let mut p = vec![0u8, (crypt << 4) | skip, 1, iv_size];
    p.extend_from_slice(&kid);
    if !per_sample_iv {
        p.push(16);
        p.extend_from_slice(&CBCS_IV);
    }
    full_box(b"tenc", 1, 0, &p)
}

fn sinf(original: &[u8; 4], scheme: &[u8; 4], kid: [u8; 16], crypt: u8, skip: u8) -> Vec<u8> {
    let schm = full_box(
        b"schm",
        0,
        0,
        &[&scheme[..], &0x0001_0000u32.to_be_bytes()].concat(),
    );
    mp4_box(
        b"sinf",
        &[
            mp4_box(b"frma", original),
            schm,
            mp4_box(b"schi", &tenc(kid, crypt, skip, per_sample_iv(scheme))),
        ]
        .concat(),
    )
}

/// The counter-mode schemes carry a per-sample IV; the CBC ones in this file use
/// the `tenc` constant IV.
fn per_sample_iv(scheme: &[u8; 4]) -> bool {
    matches!(scheme, b"cenc" | b"cens")
}

/// An `esds` carrying a 2-byte AAC-LC AudioSpecificConfig.
fn esds(config: &[u8]) -> Vec<u8> {
    let mut dec_specific = vec![0x05u8, config.len() as u8];
    dec_specific.extend_from_slice(config);
    let mut dec_config = vec![0x04u8, (13 + dec_specific.len()) as u8, 0x40, 0x15];
    dec_config.extend_from_slice(&[0u8; 11]);
    dec_config.extend_from_slice(&dec_specific);
    let mut es = vec![0x03u8, (3 + dec_config.len()) as u8, 0, 0, 0];
    es.extend_from_slice(&dec_config);
    full_box(b"esds", 0, 0, &es)
}

/// A `sgpd` for grouping type `seig` (version 1, per-entry lengths) over
/// `entries`.
fn seig_sgpd(entries: &[Vec<u8>]) -> Vec<u8> {
    let mut p = b"seig".to_vec();
    p.extend_from_slice(&0u32.to_be_bytes()); // default_length: variable
    p.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for e in entries {
        p.extend_from_slice(&(e.len() as u32).to_be_bytes());
        p.extend_from_slice(e);
    }
    full_box(b"sgpd", 1, 0, &p)
}

/// An encrypted AAC init segment: `moov` with an `enca` sample entry whose `sinf`
/// declares `cbcs` whole-sample protection (pattern 0:0) under `kid`.
fn aac_init(kid: [u8; 16]) -> Vec<u8> {
    aac_init_full(kid, b"cbcs", 0, 0, &[])
}

/// The same init segment with an explicit scheme and crypt:skip pattern, plus an
/// optional movie-level `seig` table in the track's `stbl` (what a fragment's
/// `sbgp` addresses with a group description index below 0x10000).
fn aac_init_full(
    kid: [u8; 16],
    scheme: &[u8; 4],
    crypt: u8,
    skip: u8,
    movie_seig: &[Vec<u8>],
) -> Vec<u8> {
    let mut tkhd_p = vec![0u8; 80];
    tkhd_p[8..12].copy_from_slice(&1u32.to_be_bytes()); // track_ID
    let tkhd = full_box(b"tkhd", 0, 0, &tkhd_p);
    let mut mdhd_p = vec![0u8; 16];
    mdhd_p[8..12].copy_from_slice(&TIMESCALE.to_be_bytes());
    let mdhd = full_box(b"mdhd", 0, 0, &mdhd_p);
    let mut hdlr_p = vec![0u8; 20];
    hdlr_p[4..8].copy_from_slice(b"soun");
    let hdlr = full_box(b"hdlr", 0, 0, &hdlr_p);

    let enca = {
        let mut p = vec![0u8; 28];
        p[16..18].copy_from_slice(&2u16.to_be_bytes()); // channelcount
        p.extend_from_slice(&esds(&[0x11, 0x90]));
        p.extend_from_slice(&sinf(b"mp4a", scheme, kid, crypt, skip));
        mp4_box(b"enca", &p)
    };
    let stsd = {
        let mut p = 1u32.to_be_bytes().to_vec();
        p.extend_from_slice(&enca);
        full_box(b"stsd", 0, 0, &p)
    };
    let mut stbl = stsd;
    if !movie_seig.is_empty() {
        stbl.extend_from_slice(&seig_sgpd(movie_seig));
    }
    let minf = mp4_box(b"minf", &mp4_box(b"stbl", &stbl));
    let mdia = mp4_box(b"mdia", &[mdhd, hdlr, minf].concat());
    let trak = mp4_box(b"trak", &[tkhd, mdia].concat());
    let mvex = mp4_box(b"mvex", &full_box(b"trex", 0, 0, &[0u8; 20]));
    mp4_box(b"moov", &[trak, mvex].concat())
}

/// How a fragment carries its sample auxiliary information.
enum Aux {
    /// A `senc` box, the common layout.
    Senc,
    /// A `senc` whose records lead with a 16-byte per-sample IV (the counter-mode
    /// shape), one per sample in order.
    SencWithIvs(Vec<[u8; 16]>),
    /// `saiz` + `saio` with one offset per sample, the blobs deliberately stored
    /// out of order with padding between them (a non-contiguous layout `senc`
    /// alone cannot express).
    ScatteredSaizSaio,
    /// No aux info at all: the constant IV protects the whole sample.
    None,
}

/// A `seig` mapping: `(run length, group_description_index)` pairs plus the group
/// description entries they index. An empty entry list emits the `sbgp` with no
/// `traf`-local `sgpd`, so the indices address the movie-level table.
type SeigGroups = (Vec<(u32, u32)>, Vec<Vec<u8>>);

/// One encrypted fragment for track 1 (`moof` + `mdat`). `samples` are the
/// already-encrypted sample payloads, `subsamples` the per-sample map to declare
/// (empty for whole-sample protection), `groups` the optional `seig` mapping as
/// `(run length, group_description_index)` pairs with their entries.
fn fragment(
    base_time: u64,
    samples: &[Vec<u8>],
    subsamples: &[Vec<(u16, u32)>],
    aux: Aux,
    groups: Option<SeigGroups>,
) -> Vec<u8> {
    // tfhd: default-base-is-moof (0x020000) + default_sample_duration (0x08).
    let mut tfhd_p = 1u32.to_be_bytes().to_vec();
    tfhd_p.extend_from_slice(&1024u32.to_be_bytes());
    let tfhd = full_box(b"tfhd", 0, 0x02_0008, &tfhd_p);
    let tfdt = full_box(b"tfdt", 1, 0, &base_time.to_be_bytes());
    let trun = {
        // flags: data-offset (0x1) + sample-size (0x200)
        let mut p = (samples.len() as u32).to_be_bytes().to_vec();
        p.extend_from_slice(&0u32.to_be_bytes());
        for s in samples {
            p.extend_from_slice(&(s.len() as u32).to_be_bytes());
        }
        full_box(b"trun", 0, 0x201, &p)
    };

    // Per-sample aux blobs: a constant-IV track carries no IV, only the map.
    let blobs: Vec<Vec<u8>> = (0..samples.len())
        .map(|i| {
            let mut b = Vec::new();
            if let Aux::SencWithIvs(ivs) = &aux {
                b.extend_from_slice(&ivs[i]);
            }
            if let Some(map) = subsamples.get(i) {
                if !map.is_empty() {
                    b.extend_from_slice(&(map.len() as u16).to_be_bytes());
                    for (clear, protected) in map {
                        b.extend_from_slice(&clear.to_be_bytes());
                        b.extend_from_slice(&protected.to_be_bytes());
                    }
                }
            }
            b
        })
        .collect();
    let has_subsamples = subsamples.iter().any(|m| !m.is_empty());

    let mut group_boxes = Vec::new();
    if let Some((runs, entries)) = &groups {
        let mut sbgp_p = b"seig".to_vec();
        sbgp_p.extend_from_slice(&(runs.len() as u32).to_be_bytes());
        for (count, index) in runs {
            sbgp_p.extend_from_slice(&count.to_be_bytes());
            sbgp_p.extend_from_slice(&index.to_be_bytes());
        }
        group_boxes.extend_from_slice(&full_box(b"sbgp", 0, 0, &sbgp_p));
        if !entries.is_empty() {
            group_boxes.extend_from_slice(&seig_sgpd(entries));
        }
    }

    // The aux boxes' sizes are fixed, so the moof layout (and the saio offsets
    // into it) can be computed before the box is assembled.
    let head = [tfhd, tfdt, trun].concat();
    let (aux_boxes, tail) = match aux {
        Aux::None => (Vec::new(), Vec::new()),
        Aux::Senc | Aux::SencWithIvs(_) => {
            let mut p = (samples.len() as u32).to_be_bytes().to_vec();
            for b in &blobs {
                p.extend_from_slice(b);
            }
            let flags = if has_subsamples { 0x2 } else { 0 };
            (full_box(b"senc", 0, flags, &p), Vec::new())
        }
        Aux::ScatteredSaizSaio => {
            // Sizes in sample order; offsets point into a trailing `free` box
            // that stores the blobs in reverse order with 3 bytes of padding
            // between them, so nothing about the layout is contiguous.
            let mut saiz_p = vec![0u8]; // default_sample_info_size: per-sample
            saiz_p.extend_from_slice(&(blobs.len() as u32).to_be_bytes());
            for b in &blobs {
                saiz_p.push(b.len() as u8);
            }
            let saiz = full_box(b"saiz", 0, 0, &saiz_p);
            // A saio with one offset per sample; the values are patched in once
            // the enclosing moof's size is known, so build a placeholder first.
            let saio_len = 8 + 4 + 4 + 4 * blobs.len();
            let free_header = 8usize;
            // moof header + traf header + the traf's boxes + the free header, all
            // fixed sizes, so the blob offsets are known before assembly.
            let blob_area =
                8 + 8 + head.len() + group_boxes.len() + saiz.len() + saio_len + free_header;
            let mut store = Vec::new();
            let mut offsets = vec![0u64; blobs.len()];
            for (i, b) in blobs.iter().enumerate().rev() {
                store.extend_from_slice(&[0xAA, 0xBB, 0xCC]); // padding
                offsets[i] = (blob_area + store.len()) as u64;
                store.extend_from_slice(b);
            }
            let mut saio_p = (blobs.len() as u32).to_be_bytes().to_vec();
            for o in &offsets {
                saio_p.extend_from_slice(&(*o as u32).to_be_bytes());
            }
            let saio = full_box(b"saio", 0, 0, &saio_p);
            assert_eq!(saio.len(), saio_len, "saio size must be predictable");
            (
                [saiz, saio].concat(),
                mp4_box(b"free", &store), // holds the scattered blobs
            )
        }
    };

    let traf = mp4_box(b"traf", &[head, group_boxes, aux_boxes, tail].concat());
    let moof = mp4_box(b"moof", &traf);
    let mdat = mp4_box(b"mdat", &samples.concat());
    [moof, mdat].concat()
}

/// A `seig` group entry: pattern, protection flag, constant IV and KID.
fn seig_entry(kid: [u8; 16], protected: bool, crypt: u8, skip: u8) -> Vec<u8> {
    let mut e = vec![0u8, (crypt << 4) | skip, u8::from(protected), 0];
    e.extend_from_slice(&kid);
    e.push(16);
    e.extend_from_slice(&CBCS_IV);
    e
}

/// Distinct AAC-sized sample payloads (not block multiples, so the trailing
/// partial block exercises the "left clear" rule).
fn aac_samples(n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| {
            (0..(100 + i * 7))
                .map(|b| ((b * 31 + i * 17) % 251) as u8)
                .collect()
        })
        .collect()
}

/// cbcs audio: the whole sample is one protected range with no subsample map and
/// no clear leader, using the `tenc` constant IV. The trailing partial block must
/// come back untouched, which the byte-exact comparison covers.
#[tokio::test]
async fn cbcs_aac_audio_decrypts_whole_samples() {
    let clear = aac_samples(4);
    let cipher: Vec<Vec<u8>> = clear
        .iter()
        .map(|s| cbcs_encrypt_whole(s, &CBCS_KEY, &CBCS_IV))
        .collect();
    assert_ne!(cipher, clear, "the oracle must actually encrypt");
    // A sample whose length is not a block multiple keeps its tail in the clear.
    assert!(
        cipher[0].ends_with(&clear[0][clear[0].len() / 16 * 16..]),
        "the trailing partial block stays clear"
    );

    let mut file = aac_init(CBCS_KID);
    file.extend_from_slice(&fragment(0, &cipher, &[], Aux::Senc, None));

    let keys = store_for_kid(CBCS_KID, CBCS_KEY);
    let got = demux_all(file, Some(keys)).await.expect("demux");
    assert_eq!(
        adts_payloads(&got[0]),
        clear,
        "cbcs whole-sample audio decrypt"
    );
}

/// The same audio fragment with no aux info at all: the `tenc` constant IV alone
/// protects each whole sample.
#[tokio::test]
async fn cbcs_audio_without_aux_info_uses_the_constant_iv() {
    let clear = aac_samples(3);
    let cipher: Vec<Vec<u8>> = clear
        .iter()
        .map(|s| cbcs_encrypt_whole(s, &CBCS_KEY, &CBCS_IV))
        .collect();
    let mut file = aac_init(CBCS_KID);
    file.extend_from_slice(&fragment(0, &cipher, &[], Aux::None, None));

    let keys = store_for_kid(CBCS_KID, CBCS_KEY);
    let got = demux_all(file, Some(keys)).await.expect("demux");
    assert_eq!(
        adts_payloads(&got[0]),
        clear,
        "constant-IV whole-sample decrypt"
    );
}

/// `saiz` + `saio` locate the aux info authoritatively: here each sample's blob
/// sits at its own offset, in reverse order with padding between, and there is no
/// `senc` to fall back on. Only a reader that honors the offsets recovers the
/// subsample maps (and so the cleartext).
#[tokio::test]
async fn saiz_saio_locate_aux_info_at_non_contiguous_offsets() {
    let clear = aac_samples(4);
    // A clear leader per sample, so the maps differ per sample and a mis-located
    // blob decrypts the wrong bytes.
    let maps: Vec<Vec<(u16, u32)>> = clear
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let lead = (8 + i * 4) as u16;
            vec![(lead, (s.len() - lead as usize) as u32)]
        })
        .collect();
    let cipher: Vec<Vec<u8>> = clear
        .iter()
        .zip(&maps)
        .map(|(s, map)| {
            let lead = map[0].0 as usize;
            let mut out = s[..lead].to_vec();
            out.extend_from_slice(&cbcs_encrypt_whole(&s[lead..], &CBCS_KEY, &CBCS_IV));
            out
        })
        .collect();

    let mut file = aac_init(CBCS_KID);
    file.extend_from_slice(&fragment(0, &cipher, &maps, Aux::ScatteredSaizSaio, None));

    let keys = store_for_kid(CBCS_KID, CBCS_KEY);
    let got = demux_all(file, Some(keys)).await.expect("demux");
    assert_eq!(
        adts_payloads(&got[0]),
        clear,
        "saio-located subsample maps drive the decrypt"
    );
}

/// A `seig` sample group overrides the track defaults for a run of samples: the
/// first two samples are declared unprotected (they must pass through untouched,
/// never "decrypted"), the rest carry a second KID whose key is a different one.
#[tokio::test]
async fn seig_group_marks_samples_clear_and_rekeys_the_rest() {
    const KID2: [u8; 16] = [0x77; 16];
    const KEY2: [u8; 16] = *b"m835-second-key!";

    let clear = aac_samples(4);
    let mut stored = clear.clone();
    // Samples 2 and 3 are encrypted under the group's own key; 0 and 1 stay clear.
    stored[2] = cbcs_encrypt_whole(&clear[2], &KEY2, &CBCS_IV);
    stored[3] = cbcs_encrypt_whole(&clear[3], &KEY2, &CBCS_IV);

    let groups = (
        // Fragment-local group indices are offset by 0x10000.
        vec![(2u32, 0x1_0001u32), (2, 0x1_0002)],
        vec![
            seig_entry([0; 16], false, 0, 0), // clear samples
            seig_entry(KID2, true, 0, 0),     // re-keyed samples
        ],
    );
    let mut file = aac_init(CBCS_KID);
    file.extend_from_slice(&fragment(0, &stored, &[], Aux::Senc, Some(groups)));

    // Only the group's key is available: the track default KID has none, which
    // proves the clear samples never went near the decryptor.
    let keys = store_for_kid(KID2, KEY2);
    let got = demux_all(file, Some(keys)).await.expect("demux");
    assert_eq!(
        adts_payloads(&got[0]),
        clear,
        "seig clear samples pass through and grouped samples use the group key"
    );
}

/// The same override carried by the movie-level `seig` table instead: the `sgpd`
/// sits in the track's `stbl` and the fragment's `sbgp` names its entries with
/// plain indices (below 0x10000), with no `traf`-local `sgpd` to fall back on.
#[tokio::test]
async fn movie_level_seig_table_resolves_fragment_groups() {
    const KID2: [u8; 16] = [0x66; 16];
    const KEY2: [u8; 16] = *b"m867-movie-seig!";

    let clear = aac_samples(4);
    let mut stored = clear.clone();
    stored[2] = cbcs_encrypt_whole(&clear[2], &KEY2, &CBCS_IV);
    stored[3] = cbcs_encrypt_whole(&clear[3], &KEY2, &CBCS_IV);

    let movie_seig = vec![
        seig_entry([0; 16], false, 0, 0), // entry 1: clear samples
        seig_entry(KID2, true, 0, 0),     // entry 2: re-keyed samples
    ];
    let mut file = aac_init_full(CBCS_KID, b"cbcs", 0, 0, &movie_seig);
    file.extend_from_slice(&fragment(
        0,
        &stored,
        &[],
        Aux::Senc,
        // Movie-level indices: 1-based into the `stbl` table, no 0x10000 offset,
        // and the fragment carries no `sgpd` of its own.
        Some((vec![(2u32, 1u32), (2, 2)], Vec::new())),
    ));

    // Only the group's key is registered: the track default KID has none, so a
    // reader that ignored the movie-level table could not produce this output.
    let keys = store_for_kid(KID2, KEY2);
    let got = demux_all(file, Some(keys)).await.expect("demux");
    assert_eq!(
        adts_payloads(&got[0]),
        clear,
        "movie-level seig entries drive the per-sample crypto"
    );
}

/// A movie-level index with no movie-level table, and one past the end of the
/// table that is there: both must decline rather than silently fall back to the
/// track defaults (which would decrypt a sample the group left clear).
#[tokio::test]
async fn movie_level_seig_index_without_a_table_fails_loud() {
    let clear = aac_samples(2);
    let cipher: Vec<Vec<u8>> = clear
        .iter()
        .map(|s| cbcs_encrypt_whole(s, &CBCS_KEY, &CBCS_IV))
        .collect();
    let frag =
        |runs: Vec<(u32, u32)>| fragment(0, &cipher, &[], Aux::Senc, Some((runs, Vec::new())));

    let mut no_table = aac_init(CBCS_KID);
    no_table.extend_from_slice(&frag(vec![(2, 1)]));

    let mut past_end = aac_init_full(CBCS_KID, b"cbcs", 0, 0, &[seig_entry(CBCS_KID, true, 0, 0)]);
    past_end.extend_from_slice(&frag(vec![(2, 2)]));

    for (name, file) in [
        ("movie-level index with no stbl sgpd", no_table),
        ("movie-level index past the table", past_end),
    ] {
        assert_eq!(
            demux_all(file, Some(store_for_kid(CBCS_KID, CBCS_KEY))).await,
            Err(G2gError::CapsMismatch),
            "{name}: must fail the parse",
        );
    }
}

// --- hand-built `cens` (pattern AES-CTR) vectors ----------------------------
//
// Nothing on this host authors `cens`, so these vectors come from an oracle built
// on the raw block cipher (not the `ctr` crate the decryptor uses), following
// ISO/IEC 23001-7 §9.6 / §10.3: only the pattern's crypt blocks are encrypted, the
// counter advances once per encrypted block (a skipped block consumes none of the
// keystream), the IV applies once per sample so the keystream runs continuously
// across that sample's protected ranges, and the pattern restarts at each range.

const CENS_KEY: [u8; 16] = *b"m867-cens-key!!!";
const CENS_KID: [u8; 16] = [0x2C; 16];

/// The keystream block for counter offset `n`: the IV's low 64 bits are the block
/// counter, incremented big-endian, and AES-128 encrypts the counter block.
fn keystream_block(key: &[u8; 16], iv: &[u8; 16], n: u64) -> [u8; 16] {
    use aes::cipher::{BlockEncrypt, KeyInit};
    let mut counter_block = *iv;
    let base = u64::from_be_bytes(iv[8..].try_into().unwrap());
    counter_block[8..].copy_from_slice(&base.wrapping_add(n).to_be_bytes());
    let mut block = aes::cipher::generic_array::GenericArray::clone_from_slice(&counter_block);
    aes::Aes128::new(&aes::cipher::generic_array::GenericArray::clone_from_slice(
        key,
    ))
    .encrypt_block(&mut block);
    let mut out = [0u8; 16];
    out.copy_from_slice(&block);
    out
}

/// Encrypt `ranges` of `sample` in place under `cens`. `crypt`/`skip` of 0 means
/// no pattern, so the whole range (trailing partial block included) is encrypted,
/// which is plain `cenc`.
fn cens_encrypt(sample: &mut [u8], ranges: &[(usize, usize)], iv: &[u8; 16], crypt: u8, skip: u8) {
    let mut xor = |at: usize, len: usize, counter: &mut u64| {
        for off in (0..len).step_by(16) {
            let ks = keystream_block(&CENS_KEY, iv, *counter);
            for (i, b) in sample[at + off..(at + off + 16).min(at + len)]
                .iter_mut()
                .enumerate()
            {
                *b ^= ks[i];
            }
            *counter += 1;
        }
    };
    let mut counter = 0u64;
    for &(start, end) in ranges {
        if crypt == 0 || skip == 0 {
            xor(start, end - start, &mut counter);
            continue;
        }
        let cycle = (crypt as usize + skip as usize) * 16;
        let mut at = start;
        while at < end {
            // Whole blocks only: a partial block at the end of a range stays clear.
            let take = (crypt as usize * 16).min((end - at) / 16 * 16);
            if take == 0 {
                break;
            }
            xor(at, take, &mut counter);
            at += cycle;
        }
    }
}

/// A distinct per-sample IV, as a `cens` track carries in its `senc`.
fn cens_ivs(n: usize) -> Vec<[u8; 16]> {
    (0..n)
        .map(|i| {
            let mut iv = [0u8; 16];
            iv[..8].copy_from_slice(&(0xC0DE_0000_0000_0000u64 + i as u64).to_be_bytes());
            iv
        })
        .collect()
}

/// `cens` with a 1:9 crypt:skip pattern and two protected ranges per sample: the
/// pattern restarts at each range while the counter runs on, so a decryptor that
/// restarted the keystream per range, or advanced it over skipped blocks, produces
/// different bytes from the second range on.
#[tokio::test]
async fn cens_pattern_ctr_decrypts_across_subsample_ranges() {
    // Two subsamples per sample, each a clear leader plus a 16-multiple protected
    // range (what 23001-7 §10.3 requires of `cens` content).
    let clear: Vec<Vec<u8>> = (0..3usize)
        .map(|i| {
            (0..(8 + 176) * 2)
                .map(|b| ((b * 37 + i * 11) % 251) as u8)
                .collect()
        })
        .collect();
    let maps: Vec<Vec<(u16, u32)>> = clear.iter().map(|_| vec![(8, 176), (8, 176)]).collect();
    let ranges = [(8usize, 184usize), (192, 368)];
    let ivs = cens_ivs(clear.len());
    let cipher: Vec<Vec<u8>> = clear
        .iter()
        .zip(&ivs)
        .map(|(s, iv)| {
            let mut out = s.clone();
            cens_encrypt(&mut out, &ranges, iv, 1, 9);
            out
        })
        .collect();
    assert_ne!(cipher, clear, "the oracle must actually encrypt");
    // The pattern's skipped blocks are untouched: block 1 of the first range
    // (bytes 24..40) is clear data, block 0 (8..24) is not.
    assert_eq!(
        cipher[0][24..40],
        clear[0][24..40],
        "skipped block stays clear"
    );
    assert_ne!(
        cipher[0][8..24],
        clear[0][8..24],
        "crypt block is encrypted"
    );

    let mut file = aac_init_full(CENS_KID, b"cens", 1, 9, &[]);
    file.extend_from_slice(&fragment(0, &cipher, &maps, Aux::SencWithIvs(ivs), None));

    let keys = store_for_kid(CENS_KID, CENS_KEY);
    let got = demux_all(file, Some(keys)).await.expect("demux");
    assert_eq!(
        adts_payloads(&got[0]),
        clear,
        "cens pattern AES-CTR decrypt reproduces the clear samples"
    );
}

/// A whole-sample `cens` range that is not a block multiple: the pattern covers
/// what whole blocks it reaches and the trailing partial block stays clear.
#[tokio::test]
async fn cens_leaves_a_trailing_partial_block_clear() {
    let clear = aac_samples(3); // lengths 100, 107, 114: none a multiple of 16
    let ivs = cens_ivs(clear.len());
    let cipher: Vec<Vec<u8>> = clear
        .iter()
        .zip(&ivs)
        .map(|(s, iv)| {
            let mut out = s.clone();
            cens_encrypt(&mut out, &[(0, s.len())], iv, 1, 9);
            out
        })
        .collect();
    let tail = clear[0].len() / 16 * 16;
    assert_eq!(
        cipher[0][tail..],
        clear[0][tail..],
        "the trailing partial block stays clear"
    );

    let mut file = aac_init_full(CENS_KID, b"cens", 1, 9, &[]);
    file.extend_from_slice(&fragment(0, &cipher, &[], Aux::SencWithIvs(ivs), None));

    let keys = store_for_kid(CENS_KID, CENS_KEY);
    let got = demux_all(file, Some(keys)).await.expect("demux");
    assert_eq!(adts_payloads(&got[0]), clear, "cens whole-sample decrypt");
}

/// A `cens` track whose pattern is 0:0 is not an error: per §9.6.1 pattern
/// encryption applies only when both fields are non-zero, so the whole protected
/// range is encrypted exactly as `cenc` would, trailing partial block included.
#[tokio::test]
async fn cens_without_a_pattern_encrypts_the_whole_range() {
    let clear = aac_samples(2);
    let ivs = cens_ivs(clear.len());
    let cipher: Vec<Vec<u8>> = clear
        .iter()
        .zip(&ivs)
        .map(|(s, iv)| {
            let mut out = s.clone();
            cens_encrypt(&mut out, &[(0, s.len())], iv, 0, 0);
            out
        })
        .collect();
    // Unlike the patterned case the short tail is encrypted too, under the counter
    // that follows the last whole block.
    let tail = clear[0].len() / 16 * 16;
    let ks = keystream_block(&CENS_KEY, &ivs[0], (tail / 16) as u64);
    let expect: Vec<u8> = clear[0][tail..]
        .iter()
        .zip(&ks)
        .map(|(p, k)| p ^ k)
        .collect();
    assert_eq!(
        cipher[0][tail..],
        expect[..],
        "a 0:0 pattern encrypts the trailing partial block too"
    );

    let mut file = aac_init_full(CENS_KID, b"cens", 0, 0, &[]);
    file.extend_from_slice(&fragment(0, &cipher, &[], Aux::SencWithIvs(ivs), None));

    let keys = store_for_kid(CENS_KID, CENS_KEY);
    let got = demux_all(file, Some(keys)).await.expect("demux");
    assert_eq!(adts_payloads(&got[0]), clear, "unpatterned cens decrypt");
}

/// The `cens` shapes must fail as loudly as the others: no key for the KID, and a
/// subsample map whose protected run runs past the end of the sample.
#[tokio::test]
async fn cens_failures_are_loud() {
    let clear = aac_samples(2);
    let ivs = cens_ivs(clear.len());
    let cipher: Vec<Vec<u8>> = clear
        .iter()
        .zip(&ivs)
        .map(|(s, iv)| {
            let mut out = s.clone();
            cens_encrypt(&mut out, &[(0, s.len())], iv, 1, 9);
            out
        })
        .collect();
    let build = |maps: &[Vec<(u16, u32)>]| {
        let mut file = aac_init_full(CENS_KID, b"cens", 1, 9, &[]);
        file.extend_from_slice(&fragment(
            0,
            &cipher,
            maps,
            Aux::SencWithIvs(ivs.clone()),
            None,
        ));
        file
    };

    assert_eq!(
        demux_all(build(&[]), Some(store_for_kid([0x99; 16], CENS_KEY))).await,
        Err(G2gError::CapsMismatch),
        "no key for the cens KID: fail, never emit ciphertext",
    );

    // A subsample map claiming more bytes than the sample holds must fail, not
    // decrypt the part that fits and pass the rest off as plaintext.
    let long: Vec<Vec<(u16, u32)>> = clear.iter().map(|_| vec![(4, 0xFFFF_0000)]).collect();
    assert_eq!(
        demux_all(build(&long), Some(store_for_kid(CENS_KID, CENS_KEY))).await,
        Err(G2gError::CapsMismatch),
        "a subsample map past the end of the sample must fail the parse",
    );
}

// --- malformed protection metadata -----------------------------------------

/// Overwrite `len` bytes at `at` bytes past the start of the first `kind` box's
/// payload. Used to corrupt one field of an otherwise valid fragment.
fn patch_box(file: &mut [u8], kind: &[u8; 4], at: usize, bytes: &[u8]) {
    let pos = file
        .windows(4)
        .position(|w| w == kind)
        .unwrap_or_else(|| panic!("no {} box", core::str::from_utf8(kind).unwrap()));
    let start = pos + 4 + at;
    file[start..start + bytes.len()].copy_from_slice(bytes);
}

/// A valid encrypted audio file with subsample maps, as the base for corruption.
fn mapped_file(aux: Aux) -> Vec<u8> {
    let clear = aac_samples(3);
    let maps: Vec<Vec<(u16, u32)>> = clear
        .iter()
        .map(|s| vec![(8u16, (s.len() - 8) as u32)])
        .collect();
    let cipher: Vec<Vec<u8>> = clear
        .iter()
        .map(|s| {
            let mut out = s[..8].to_vec();
            out.extend_from_slice(&cbcs_encrypt_whole(&s[8..], &CBCS_KEY, &CBCS_IV));
            out
        })
        .collect();
    let mut file = aac_init(CBCS_KID);
    file.extend_from_slice(&fragment(0, &cipher, &maps, aux, None));
    file
}

/// The same file with a `traf`-local `seig` group over its samples, as the base
/// for corrupting the sample-group boxes.
fn mapped_seig_file() -> Vec<u8> {
    let clear = aac_samples(2);
    let cipher: Vec<Vec<u8>> = clear
        .iter()
        .map(|s| cbcs_encrypt_whole(s, &CBCS_KEY, &CBCS_IV))
        .collect();
    let mut file = aac_init(CBCS_KID);
    file.extend_from_slice(&fragment(
        0,
        &cipher,
        &[],
        Aux::Senc,
        Some((vec![(2, 0x1_0001)], vec![seig_entry(CBCS_KID, true, 0, 0)])),
    ));
    file
}

/// Every one of these lies about the size, count or position of something the
/// parser would otherwise trust. Each must surface an error, never panic and
/// never emit ciphertext as media.
#[tokio::test]
async fn malformed_protection_metadata_fails_without_panicking() {
    let keys = || Some(store_for_kid(CBCS_KID, CBCS_KEY));

    // A `saio` offset past the end of the fragment.
    let mut saio_past_end = mapped_file(Aux::ScatteredSaizSaio);
    patch_box(
        &mut saio_past_end,
        b"saio",
        8,
        &0xFFFF_0000u32.to_be_bytes(),
    );

    // A `saiz` sample count far beyond the sizes the box carries.
    let mut saiz_count = mapped_file(Aux::ScatteredSaizSaio);
    patch_box(&mut saiz_count, b"saiz", 5, &0x7FFF_FFFFu32.to_be_bytes());

    // A `senc` subsample count that claims more entries than the box holds.
    let mut senc_subs = mapped_file(Aux::Senc);
    patch_box(&mut senc_subs, b"senc", 8, &0xFFFFu16.to_be_bytes());

    // A `senc` sample count beyond the fragment's samples.
    let mut senc_count = mapped_file(Aux::Senc);
    patch_box(&mut senc_count, b"senc", 4, &0x00FF_FFFFu32.to_be_bytes());

    // The `senc` versions the multi-key layout uses (1 is what the one packager
    // that writes it emits, 2 has no established layout at all): both are declined
    // rather than mis-sliced as version 0.
    let mut senc_v1 = mapped_file(Aux::Senc);
    patch_box(&mut senc_v1, b"senc", 0, &[1]);
    let mut senc_v2 = mapped_file(Aux::Senc);
    patch_box(&mut senc_v2, b"senc", 0, &[2]);

    // A `tenc` per-sample IV size no scheme defines.
    let mut bad_iv_size = mapped_file(Aux::Senc);
    patch_box(&mut bad_iv_size, b"tenc", 7, &[7]);

    // A `sgpd` version whose entry offsets differ between editions of 14496-12.
    let mut sgpd_v2 = mapped_seig_file();
    patch_box(&mut sgpd_v2, b"sgpd", 0, &[2]);

    for (name, file) in [
        ("saio offset past the fragment", saio_past_end),
        ("saiz count beyond the box", saiz_count),
        ("senc subsample count beyond the box", senc_subs),
        ("senc sample count beyond the fragment", senc_count),
        ("senc multi-key version 1", senc_v1),
        ("senc multi-key version 2", senc_v2),
        ("undefined per-sample IV size", bad_iv_size),
        ("sgpd version 2", sgpd_v2),
    ] {
        assert_eq!(
            demux_all(file, keys()).await,
            Err(G2gError::CapsMismatch),
            "{name}: must fail the parse, not panic or emit ciphertext",
        );
    }
}

/// A `seig` mapping that points at a group description the `sgpd` does not have,
/// and one whose entry declares the multi-key layout: both must decline rather
/// than fall back to the track defaults (which could decrypt a clear sample or
/// use the wrong key).
#[tokio::test]
async fn malformed_sample_groups_fail_without_panicking() {
    let clear = aac_samples(2);
    let cipher: Vec<Vec<u8>> = clear
        .iter()
        .map(|s| cbcs_encrypt_whole(s, &CBCS_KEY, &CBCS_IV))
        .collect();

    let build = |runs: Vec<(u32, u32)>, entries: Vec<Vec<u8>>| {
        let mut file = aac_init(CBCS_KID);
        file.extend_from_slice(&fragment(0, &cipher, &[], Aux::Senc, Some((runs, entries))));
        file
    };
    let dangling = build(vec![(2, 0x1_0009)], vec![seig_entry(CBCS_KID, true, 0, 0)]);
    // The two descriptions of the multi-key entry put its flag in different bits of
    // the reserved byte, so both encodings must decline.
    let flagged = |bit: u8| {
        let mut e = seig_entry(CBCS_KID, true, 0, 0);
        e[0] = bit;
        build(vec![(2, 0x1_0001)], vec![e])
    };

    for (name, file) in [
        ("group index with no description", dangling),
        ("multi-key group entry (bit 7)", flagged(0x80)),
        ("multi-key group entry (bit 0)", flagged(0x01)),
    ] {
        assert_eq!(
            demux_all(file, Some(store_for_kid(CBCS_KID, CBCS_KEY))).await,
            Err(G2gError::CapsMismatch),
            "{name}: must fail the parse",
        );
    }
}

// --- mid-stream key rotation over HLS --------------------------------------

/// Serve a fixed set of paths, plus a playlist that changes on each request (the
/// live-reload case): `playlists[i]` answers the i-th playlist fetch, the last
/// one repeating.
fn serve(playlists: Vec<String>, files: Vec<(String, Vec<u8>)>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        let mut playlist_hits = 0usize;
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { break };
            let mut req = Vec::new();
            let mut byte = [0u8; 1];
            while stream.read(&mut byte).unwrap_or(0) == 1 {
                req.push(byte[0]);
                if req.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let line = String::from_utf8_lossy(&req);
            let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
            let body: Option<Vec<u8>> = if path == "/stream.m3u8" {
                let i = playlist_hits.min(playlists.len() - 1);
                playlist_hits += 1;
                Some(playlists[i].as_bytes().to_vec())
            } else {
                files
                    .iter()
                    .find(|(name, _)| path == format!("/{name}"))
                    .map(|(_, body)| body.clone())
            };
            match body {
                Some(body) => {
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(&body);
                }
                None => {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                }
            }
        }
    });
    format!("http://127.0.0.1:{port}/stream.m3u8")
}

/// Queues everything `HlsSrc` emits instead of demuxing it inline. The demuxer
/// runs only once the source has finished, which is the worst case of the link
/// buffering that always exists between the two: every key the playlist declares
/// has been fetched and published before a single fragment is parsed, so a store
/// that just held "the key in force" would decrypt the whole stream with the last
/// one.
#[derive(Default)]
struct ByteQueue {
    segments: Vec<Vec<u8>>,
}

impl OutputSink for ByteQueue {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.segments.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// A rotating stream: the clear samples per segment, the per-segment keys, and
/// the files (init, segments, key resources) the fixture server hosts.
type RotatingStream = (Vec<Vec<Vec<u8>>>, Vec<[u8; 16]>, Vec<(String, Vec<u8>)>);

/// Segments each encrypted under their own key, with the files that serve them.
fn rotating_stream(seg_count: usize) -> RotatingStream {
    let mut clear_by_seg = Vec::new();
    let mut segment_keys = Vec::new();
    let mut files = vec![("init.mp4".into(), aac_init(CBCS_KID))];
    for seg in 0..seg_count {
        let mut key = CBCS_KEY;
        key[0] = b'0' + seg as u8; // a distinct key per segment
        let clear = aac_samples(3 + seg);
        let cipher: Vec<Vec<u8>> = clear
            .iter()
            .map(|s| cbcs_encrypt_whole(s, &key, &CBCS_IV))
            .collect();
        files.push((
            format!("seg{seg}.m4s"),
            fragment(seg as u64 * 4096, &cipher, &[], Aux::Senc, None),
        ));
        files.push((format!("k{seg}.key"), key.to_vec()));
        clear_by_seg.push(clear);
        segment_keys.push(key);
    }
    (clear_by_seg, segment_keys, files)
}

/// A media playlist body over `segments` indices, with a fresh `#EXT-X-KEY`
/// before each (back-to-back rotation) and `#EXT-X-ENDLIST` when `end`.
fn rotating_playlist(segments: &[usize], end: bool) -> String {
    let mut p = String::from(
        "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:0\n#EXT-X-MAP:URI=\"init.mp4\"\n",
    );
    for seg in segments {
        p.push_str(&format!(
            "#EXT-X-KEY:METHOD=SAMPLE-AES,URI=\"k{seg}.key\"\n#EXTINF:1.0,\nseg{seg}.m4s\n"
        ));
    }
    if end {
        p.push_str("#EXT-X-ENDLIST\n");
    }
    p
}

/// Run the chain and return the demuxed samples of the single audio port.
async fn run_hls(url: String, prebuffer_ms: u64, keys: CencKeyHandle) -> Vec<Vec<u8>> {
    let mut queue = ByteQueue::default();
    let mut src = HlsSrc::new(url)
        .with_sample_aes_key_handle(keys.clone())
        .with_prebuffer_ms(prebuffer_ms)
        .with_reload_interval_ms(20);
    src.configure_pipeline(&g2g_core::Caps::ByteStream {
        encoding: g2g_core::ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    src.run(&mut queue).await.expect("hls run");

    let ports = vec![Mp4Port {
        track_id: 1,
        caps: g2g_core::Caps::Audio {
            format: g2g_core::AudioFormat::Aac,
            channels: 0,
            sample_rate: 0,
        },
    }];
    let mut demux = Mp4DemuxN::new(ports).with_cenc_key_handle(keys);
    let mut capture = PortCapture::new(1);
    for bytes in queue.segments {
        demux
            .process(byte_frame(bytes), &mut capture)
            .await
            .expect("demux segment");
    }
    demux
        .process(PipelinePacket::Eos, &mut capture)
        .await
        .expect("demux eos");
    adts_payloads(&capture.frames[0])
}

/// A new `#EXT-X-KEY` before every segment: each fragment must decrypt with the
/// key its own segment declared. The prebuffer is deliberately large enough to
/// hold the whole playlist, so every key is fetched before the first byte is
/// emitted: a key store that simply kept "the last key fetched" would decrypt
/// every segment with the last one.
#[tokio::test]
async fn key_rotation_switches_at_each_segment_boundary() {
    let (clear, keys, files) = rotating_stream(3);
    assert_ne!(keys[0], keys[1], "the segments really do use distinct keys");
    let url = serve(vec![rotating_playlist(&[0, 1, 2], true)], files);

    let got = run_hls(url, 60_000, new_key_handle()).await;
    let expected: Vec<Vec<u8>> = clear.into_iter().flatten().collect();
    assert_eq!(
        got, expected,
        "every segment decrypts with the key in force for it"
    );
}

/// The same rotation, but the second key only appears when a live playlist is
/// reloaded (no `#EXT-X-ENDLIST` on the first fetch). The key published on the
/// reload must govern only the segment it precedes.
#[tokio::test]
async fn key_rotation_on_live_reload() {
    let (clear, _, files) = rotating_stream(2);
    let url = serve(
        vec![
            rotating_playlist(&[0], false),
            rotating_playlist(&[0, 1], true),
        ],
        files,
    );

    let got = run_hls(url, 0, new_key_handle()).await;
    let expected: Vec<Vec<u8>> = clear.into_iter().flatten().collect();
    assert_eq!(
        got, expected,
        "the key added on the live reload decrypts only its own segment"
    );
}

/// A `KEYID` on the `#EXT-X-KEY` binds the playlist key to the CENC key
/// identifier the segments name, which is how a re-keyed stream picks a key
/// per sample rather than per position.
#[tokio::test]
async fn keyid_binds_the_playlist_key_to_the_container_kid() {
    let clear = aac_samples(3);
    let cipher: Vec<Vec<u8>> = clear
        .iter()
        .map(|s| cbcs_encrypt_whole(s, &CBCS_KEY, &CBCS_IV))
        .collect();
    let files = vec![
        ("init.mp4".to_string(), aac_init(CBCS_KID)),
        (
            "seg0.m4s".to_string(),
            fragment(0, &cipher, &[], Aux::Senc, None),
        ),
        ("k0.key".to_string(), CBCS_KEY.to_vec()),
    ];
    let kid_hex: String = CBCS_KID.iter().map(|b| format!("{b:02x}")).collect();
    let playlist = format!(
        "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXT-X-MEDIA-SEQUENCE:0\n\
         #EXT-X-MAP:URI=\"init.mp4\"\n\
         #EXT-X-KEY:METHOD=SAMPLE-AES,URI=\"k0.key\",KEYID=0x{kid_hex}\n\
         #EXTINF:1.0,\nseg0.m4s\n#EXT-X-ENDLIST\n"
    );
    let url = serve(vec![playlist], files);

    let got = run_hls(url, 0, new_key_handle()).await;
    assert_eq!(got, clear, "the KEYID-registered key decrypts the samples");
}

/// A `sbgp` that maps more samples than the fragment holds is rejected before it
/// is used to size anything (an oversized run would otherwise allocate on a
/// number the stream chose).
#[tokio::test]
async fn oversized_sample_group_run_is_rejected() {
    let clear = aac_samples(2);
    let cipher: Vec<Vec<u8>> = clear
        .iter()
        .map(|s| cbcs_encrypt_whole(s, &CBCS_KEY, &CBCS_IV))
        .collect();
    let mut file = aac_init(CBCS_KID);
    file.extend_from_slice(&fragment(
        0,
        &cipher,
        &[],
        Aux::Senc,
        Some((
            vec![(0x00FF_FFFF, 0x1_0001)],
            vec![seig_entry(CBCS_KID, true, 0, 0)],
        )),
    ));
    assert_eq!(
        demux_all(file, Some(store_for_kid(CBCS_KID, CBCS_KEY))).await,
        Err(G2gError::CapsMismatch),
        "a run longer than the fragment's samples must fail the parse",
    );
}

//! M164 fMP4 `cbcs` (MPEG Common Encryption) decryption in `Fmp4Demux`. Builds an
//! encrypted CMAF stream by hand (an `encv`/`sinf`/`tenc` init segment + a
//! `moof` whose `senc` maps subsamples over `cbcs`-encrypted `mdat` samples),
//! then checks `Fmp4Demux`, given the key via the shared handle, recovers the
//! original access units. The encrypt side mirrors the decryptor, so the real
//! decrypt path runs against independent ciphertext. Also covers the movie-level
//! `seig` table on this demuxer's init path (M867).

#![cfg(feature = "hls")]

use core::future::Future;
use core::pin::Pin;

use aes::cipher::{block_padding::NoPadding, BlockEncryptMut, KeyIvInit};
use g2g_core::element::AsyncElement;
use g2g_core::{
    ByteStreamEncoding, Caps, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome,
};
use g2g_plugins::fmp4demux::Fmp4Demux;
use g2g_plugins::sampleaesdecrypt::{new_key_handle, SampleAesKey};

const KEY: [u8; 16] = *b"cbcs-test-key!!!";
const IV: [u8; 16] = [0x44; 16];
const CRYPT: u8 = 1;
const SKIP: u8 = 9;
const TIMESCALE: u32 = 90_000;

// --- MP4 box builders ------------------------------------------------------

fn mp4_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut b = ((payload.len() as u32 + 8).to_be_bytes()).to_vec();
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

fn avcc(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let mut p = vec![1, sps[1], sps[2], sps[3], 0xFF, 0xE1]; // version, profile/compat/level, lenSize-1, 111+spsCount=1
    p.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    p.extend_from_slice(sps);
    p.push(1); // pps count
    p.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    p.extend_from_slice(pps);
    mp4_box(b"avcC", &p)
}

fn sinf() -> Vec<u8> {
    let frma = mp4_box(b"frma", b"avc1");
    let schm = full_box(
        b"schm",
        0,
        0,
        &[&b"cbcs"[..], &0x0001_0000u32.to_be_bytes()].concat(),
    );
    let mut tp = vec![0u8]; // tenc[4] reserved
    tp.push((CRYPT << 4) | SKIP); // tenc[5] crypt/skip pattern
    tp.push(1); // default_isProtected
    tp.push(0); // default_Per_Sample_IV_Size (cbcs constant IV)
    tp.extend_from_slice(&[0u8; 16]); // default_KID
    tp.push(16); // constant_iv_size
    tp.extend_from_slice(&IV);
    let schi = mp4_box(b"schi", &full_box(b"tenc", 1, 0, &tp));
    mp4_box(b"sinf", &[frma, schm, schi].concat())
}

/// A `seig` group description entry: pattern, protection flag, KID and the
/// constant IV, the same record shape as the `tenc` fields.
fn seig_entry(kid: [u8; 16]) -> Vec<u8> {
    let mut e = vec![0u8, (CRYPT << 4) | SKIP, 1, 0];
    e.extend_from_slice(&kid);
    e.push(16);
    e.extend_from_slice(&IV);
    e
}

fn moov(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    moov_with_seig(sps, pps, &[])
}

/// The same init segment with a movie-level `seig` table in the track's `stbl`,
/// which a fragment's `sbgp` addresses with an index below 0x10000.
fn moov_with_seig(sps: &[u8], pps: &[u8], movie_seig: &[Vec<u8>]) -> Vec<u8> {
    let mut tkhd_p = vec![0u8; 80];
    tkhd_p[72..76].copy_from_slice(&(64u32 << 16).to_be_bytes()); // width
    tkhd_p[76..80].copy_from_slice(&(48u32 << 16).to_be_bytes()); // height
    let tkhd = full_box(b"tkhd", 0, 0, &tkhd_p);

    let mut mdhd_p = vec![0u8; 16];
    mdhd_p[8..12].copy_from_slice(&TIMESCALE.to_be_bytes());
    let mdhd = full_box(b"mdhd", 0, 0, &mdhd_p);

    let mut encv_p = vec![0u8; 78]; // visual sample entry fixed fields
    encv_p.extend_from_slice(&avcc(sps, pps));
    encv_p.extend_from_slice(&sinf());
    let encv = mp4_box(b"encv", &encv_p);

    let mut stbl_p = full_box(b"stsd", 0, 0, &[&1u32.to_be_bytes()[..], &encv].concat());
    if !movie_seig.is_empty() {
        let mut sgpd_p = b"seig".to_vec();
        sgpd_p.extend_from_slice(&0u32.to_be_bytes()); // default_length: variable
        sgpd_p.extend_from_slice(&(movie_seig.len() as u32).to_be_bytes());
        for e in movie_seig {
            sgpd_p.extend_from_slice(&(e.len() as u32).to_be_bytes());
            sgpd_p.extend_from_slice(e);
        }
        stbl_p.extend_from_slice(&full_box(b"sgpd", 1, 0, &sgpd_p));
    }
    let stbl = mp4_box(b"stbl", &stbl_p);
    let minf = mp4_box(b"minf", &stbl);
    let mdia = mp4_box(b"mdia", &[mdhd, minf].concat());
    let trak = mp4_box(b"trak", &[tkhd, mdia].concat());
    mp4_box(b"moov", &trak)
}

// --- cbcs encrypt mirror (inverse of fmp4demux's decrypt_protected_range) ---

type Aes128CbcEnc = cbc::Encryptor<aes::Aes128>;

fn encrypted_block_offsets(len: usize) -> Vec<usize> {
    let block_count = len / 16;
    let span = (CRYPT + SKIP) as usize;
    (0..block_count)
        .filter(|b| b % span < CRYPT as usize)
        .map(|b| b * 16)
        .collect()
}

fn cbcs_encrypt_range(range: &mut [u8], key: &[u8; 16]) {
    let offsets = encrypted_block_offsets(range.len());
    if offsets.is_empty() {
        return;
    }
    let mut gathered: Vec<u8> = offsets
        .iter()
        .flat_map(|&o| range[o..o + 16].iter().copied())
        .collect();
    let n = gathered.len();
    Aes128CbcEnc::new(key.into(), &IV.into())
        .encrypt_padded_mut::<NoPadding>(&mut gathered, n)
        .unwrap();
    for (i, &o) in offsets.iter().enumerate() {
        range[o..o + 16].copy_from_slice(&gathered[i * 16..i * 16 + 16]);
    }
}

/// One IDR-slice access unit per sample, encrypted under `KEY`.
fn make_fragment(nals: &[Vec<u8>]) -> Vec<u8> {
    make_fragment_keyed(nals, &[], None)
}

/// The `moof`+`mdat` fragment for `nals`, one sample each: `keys[i]` encrypts
/// sample `i` (falling back to `KEY`), and `groups` adds a `seig` `sbgp` mapping
/// runs of samples to group description indices.
fn make_fragment_keyed(
    nals: &[Vec<u8>],
    keys: &[[u8; 16]],
    groups: Option<Vec<(u32, u32)>>,
) -> Vec<u8> {
    let clear_leader = 16usize; // length prefix + NAL header + slice-header bytes
    let mut mdat_payload = Vec::new();
    let mut sizes = Vec::new();
    let mut senc_entries = Vec::new();
    for (i, nal) in nals.iter().enumerate() {
        let mut sample = (nal.len() as u32).to_be_bytes().to_vec();
        sample.extend_from_slice(nal);
        let protected = sample.len() - clear_leader;
        cbcs_encrypt_range(&mut sample[clear_leader..], keys.get(i).unwrap_or(&KEY));
        sizes.push(sample.len() as u32);
        senc_entries.push((clear_leader as u16, protected as u32));
        mdat_payload.extend_from_slice(&sample);
    }

    let tfdt = full_box(b"tfdt", 0, 0, &0u32.to_be_bytes());

    let mut trun_p = (nals.len() as u32).to_be_bytes().to_vec();
    for size in &sizes {
        trun_p.extend_from_slice(&3000u32.to_be_bytes()); // duration
        trun_p.extend_from_slice(&size.to_be_bytes());
    }
    let trun = full_box(b"trun", 0, 0x300, &trun_p);

    let mut senc_p = (nals.len() as u32).to_be_bytes().to_vec();
    for (clear, protected) in &senc_entries {
        senc_p.extend_from_slice(&1u16.to_be_bytes()); // subsample count
        senc_p.extend_from_slice(&clear.to_be_bytes());
        senc_p.extend_from_slice(&protected.to_be_bytes());
    }
    let senc = full_box(b"senc", 0, 0x2, &senc_p);

    let sbgp = match groups {
        Some(runs) => {
            let mut p = b"seig".to_vec();
            p.extend_from_slice(&(runs.len() as u32).to_be_bytes());
            for (count, index) in &runs {
                p.extend_from_slice(&count.to_be_bytes());
                p.extend_from_slice(&index.to_be_bytes());
            }
            full_box(b"sbgp", 0, 0, &p)
        }
        None => Vec::new(),
    };

    let traf = mp4_box(b"traf", &[tfdt, trun, sbgp, senc].concat());
    let moof = mp4_box(b"moof", &traf);
    [moof, mp4_box(b"mdat", &mdat_payload)].concat()
}

fn idr_nals() -> Vec<Vec<u8>> {
    (0..3u8)
        .map(|s| {
            let mut nal = vec![0x65u8]; // IDR slice NAL header
            nal.extend((0..400u32).map(|i| (i as u8).wrapping_mul(31).wrapping_add(s)));
            nal
        })
        .collect()
}

#[derive(Default)]
struct CaptureSink {
    frames: Vec<Vec<u8>>,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn data_frame(bytes: Vec<u8>) -> PipelinePacket {
    use g2g_core::frame::{Frame, FrameTiming};
    use g2g_core::memory::SystemSlice;
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    })
}

#[tokio::test]
async fn fmp4_cbcs_decrypts_with_key_handle() {
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let nals = idr_nals();
    let moov = moov(&sps, &pps);
    let fragment = make_fragment(&nals);

    let handle = new_key_handle();
    handle.lock().unwrap().set_current(SampleAesKey {
        key: KEY,
        iv: [0; 16],
    }); // IV unused (tenc constant IV)

    let mut demux = Fmp4Demux::new().with_cbcs_key_handle(handle);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .unwrap();
    let mut sink = CaptureSink::default();
    demux.process(data_frame(moov), &mut sink).await.unwrap();
    demux
        .process(data_frame(fragment), &mut sink)
        .await
        .unwrap();

    assert_eq!(sink.frames.len(), nals.len(), "one access unit per sample");
    for (frame, nal) in sink.frames.iter().zip(&nals) {
        let expected_tail = [&[0, 0, 0, 1][..], nal].concat();
        assert!(
            frame.ends_with(&expected_tail),
            "decrypted access unit ends with the original IDR NAL",
        );
    }
}

/// A movie-level `seig` table (the `sgpd` in the track's `stbl`) resolves the
/// fragment's `sbgp`: the first sample keeps the `tenc` default KID and the rest
/// are re-keyed by group entry 1, addressed by a plain index below 0x10000. Only
/// both keys together recover the stream, so the movie-level table has to be read.
#[tokio::test]
async fn fmp4_movie_level_seig_rekeys_a_run_of_samples() {
    const KID_TRACK: [u8; 16] = [0; 16]; // the tenc default KID this init declares
    const KID_GROUP: [u8; 16] = [0x5E; 16];
    const KEY_GROUP: [u8; 16] = *b"m867-seig-key!!!";

    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let nals = idr_nals();
    let moov = moov_with_seig(&sps, &pps, &[seig_entry(KID_GROUP)]);
    // Sample 0 under the track key, samples 1 and 2 under the group's key.
    let fragment = make_fragment_keyed(
        &nals,
        &[KEY, KEY_GROUP, KEY_GROUP],
        Some(vec![(1, 0), (2, 1)]),
    );

    let handle = new_key_handle();
    handle.lock().unwrap().insert_kid(KID_TRACK, KEY);
    handle.lock().unwrap().insert_kid(KID_GROUP, KEY_GROUP);

    let mut demux = Fmp4Demux::new().with_cbcs_key_handle(handle);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .unwrap();
    let mut sink = CaptureSink::default();
    demux.process(data_frame(moov), &mut sink).await.unwrap();
    demux
        .process(data_frame(fragment), &mut sink)
        .await
        .unwrap();

    assert_eq!(sink.frames.len(), nals.len(), "one access unit per sample");
    for (frame, nal) in sink.frames.iter().zip(&nals) {
        let expected_tail = [&[0, 0, 0, 1][..], nal].concat();
        assert!(
            frame.ends_with(&expected_tail),
            "movie-level seig group key decrypts its run of samples",
        );
    }
}

#[tokio::test]
async fn fmp4_cbcs_without_key_fails_loud() {
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let moov = moov(&sps, &pps);
    let fragment = make_fragment(&idr_nals());

    let mut demux = Fmp4Demux::new(); // no key handle
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .unwrap();
    let mut sink = CaptureSink::default();
    demux.process(data_frame(moov), &mut sink).await.unwrap();
    assert_eq!(
        demux.process(data_frame(fragment), &mut sink).await,
        Err(G2gError::CapsMismatch),
        "an encrypted fragment without a key fails loud, not silently garbled",
    );
}

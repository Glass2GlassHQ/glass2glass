//! MPEG Common Encryption (ISO/IEC 23001-7), shared by the HLS fMP4 path
//! ([`Fmp4Demux`](crate::fmp4demux), `hls` feature) and the multi-track MP4
//! demuxer ([`Mp4DemuxN`](crate::mp4demuxn), `mp4-cenc` feature).
//!
//! Two halves. The box half (always compiled with `std`) reads the protection
//! metadata: the init segment's `sinf`/`schm`/`tenc` track defaults, and per
//! fragment the sample auxiliary information (`senc`, or `saiz`+`saio` when the
//! aux data is located out of line) plus the `seig` sample-group overrides that
//! can switch KID, IV size, pattern or protection on a run of samples (described
//! either in the `traf` or by the track's movie-level table). The
//! cipher half (behind `hls` / `mp4-cenc`) applies the scheme: `cbcs` (pattern
//! AES-CBC), `cbc1` (full AES-CBC), `cenc` (AES-CTR) and `cens` (pattern
//! AES-CTR).
//!
//! Everything here reads attacker-controlled bytes: counts, sizes and offsets
//! are bounds-checked with checked arithmetic and a malformed box fails the
//! parse (`CapsMismatch`) rather than panicking or decrypting garbage.

use alloc::vec::Vec;
use std::sync::{Arc, Mutex};

use g2g_core::G2gError;

use crate::mp4box::{be32, be64, find_box};

/// The Common Encryption scheme a track's `schm` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CencScheme {
    /// AES-CTR, per-sample IV, keystream continuous over the protected bytes.
    Cenc,
    /// AES-CTR over a crypt:skip block pattern, per-sample IV. The keystream is
    /// continuous over the encrypted blocks only: a skipped block consumes none
    /// of it (ISO/IEC 23001-7 §9.6, §10.3).
    Cens,
    /// AES-CBC over whole blocks, IV reset per protected range, no pattern.
    Cbc1,
    /// AES-CBC over a crypt:skip block pattern, IV reset per protected range.
    Cbcs,
}

/// A track's protection metadata from the init segment: the `tenc` defaults plus
/// the movie-level `seig` table (the `sgpd` in the track's `stbl`), which a
/// fragment's `sbgp` addresses with a group description index below 0x10000.
#[derive(Debug, Clone)]
pub(crate) struct CencTrack {
    pub(crate) defaults: CencDefaults,
    /// Movie-level `seig` group entries in `sgpd` order, empty when the track
    /// carries no such table.
    pub(crate) movie_seig: Vec<CencDefaults>,
}

/// Protection defaults for a track (`tenc`) or for a `seig` sample group. The IV
/// is either per-sample (`per_sample_iv_size` bytes in the sample aux info) or
/// the `constant_iv` here (`per_sample_iv_size == 0`, the cbcs shape).
#[derive(Debug, Clone)]
pub(crate) struct CencDefaults {
    pub(crate) scheme: CencScheme,
    pub(crate) is_protected: bool,
    pub(crate) crypt_byte_block: u8,
    pub(crate) skip_byte_block: u8,
    pub(crate) per_sample_iv_size: u8,
    pub(crate) constant_iv: Vec<u8>,
    /// Default key identifier: which content key decrypts this track / group.
    pub(crate) kid: [u8; 16],
}

/// One subsample range: `clear` bytes pass through, the next `protected` bytes
/// are encrypted (byte counts over the sample as stored).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Subsample {
    pub(crate) clear: u32,
    pub(crate) protected: u32,
}

/// The crypto in force for one sample, after folding the track defaults with any
/// `seig` group override and the sample's own auxiliary information.
// Read by the cipher half, which the decryption features gate.
#[cfg_attr(
    not(any(feature = "hls", feature = "mp4-cenc")),
    allow(dead_code, reason = "read only by the feature-gated decryptor")
)]
#[derive(Debug, Clone)]
pub(crate) struct SampleCrypt {
    pub(crate) kid: [u8; 16],
    pub(crate) scheme: CencScheme,
    pub(crate) crypt_byte_block: u8,
    pub(crate) skip_byte_block: u8,
    pub(crate) iv: [u8; 16],
    /// Empty means the whole sample is one protected range.
    pub(crate) subsamples: Vec<Subsample>,
    /// `false` for a sample a `seig` group (or the track) marks unprotected: it
    /// is already clear and must be passed through untouched.
    pub(crate) protected: bool,
}

impl SampleCrypt {
    /// A clear sample: no key needed, bytes pass through.
    fn clear() -> Self {
        Self {
            kid: [0; 16],
            scheme: CencScheme::Cbcs,
            crypt_byte_block: 0,
            skip_byte_block: 0,
            iv: [0; 16],
            subsamples: Vec::new(),
            protected: false,
        }
    }
}

/// Content keys available to a demuxer, filled by a key publisher
/// ([`HlsSrc`](crate::hlssrc) fetching `#EXT-X-KEY` material, or an app clear
/// key) and read per sample.
///
/// A sample's key is resolved in this order: the key registered for the sample's
/// KID (CENC / EME clear-key semantics: the container names which key it needs),
/// then the rotation timeline (the key covering the sample's byte position in the
/// stream), then the current key. The timeline is what makes mid-stream
/// `#EXT-X-KEY` rotation exact: entries are keyed by the byte offset at which
/// their segment enters the stream, so a key never overtakes the segments the
/// previous key still governs, no matter how far the source runs ahead.
#[derive(Default)]
pub struct CencKeyStore {
    current: Option<ContentKey>,
    /// `(first stream byte offset the key governs, key)`, in ascending order.
    timeline: Vec<(u64, ContentKey)>,
    by_kid: Vec<([u8; 16], [u8; 16])>,
}

// Debug reports only how many keys are held: the KID map is raw key material, so
// no derived formatting may reach a log.
impl core::fmt::Debug for CencKeyStore {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CencKeyStore")
            .field("has_current", &self.current.is_some())
            .field("rotations", &self.timeline.len())
            .field("keys_by_kid", &self.by_kid.len())
            .finish()
    }
}

/// A rotation timeline this long covers minutes of per-segment rotation; the
/// oldest entries describe media the demuxer consumed long ago.
const MAX_TIMELINE: usize = 256;

/// A 16-byte AES content key plus the IV that goes with it (the HLS `#EXT-X-KEY`
/// pair). The IV matters for the TS SAMPLE-AES decryptor; the fMP4 CENC path
/// takes its IV from the sample aux info or the `tenc` constant IV instead.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ContentKey {
    pub key: [u8; 16],
    pub iv: [u8; 16],
}

// Redact the key/IV from Debug so secrets don't leak into logs.
impl core::fmt::Debug for ContentKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ContentKey").finish_non_exhaustive()
    }
}

impl CencKeyStore {
    /// Set the key in force with no stream position attached (the direct,
    /// non-rotating case).
    pub fn set_current(&mut self, key: ContentKey) {
        self.current = Some(key);
    }

    /// The key in force, ignoring the rotation timeline.
    pub fn current(&self) -> Option<ContentKey> {
        self.current
    }

    /// Publish `key` as governing the stream from byte `offset` on (the offset of
    /// the first byte of the segment it decrypts, in the byte stream the source
    /// emits). Also becomes the current key.
    pub fn publish_at(&mut self, offset: u64, key: ContentKey) {
        // Re-publishing the same boundary (a re-emitted init on an ABR switch)
        // replaces rather than stacks.
        match self.timeline.last_mut() {
            Some((at, slot)) if *at == offset => *slot = key,
            _ => self.timeline.push((offset, key)),
        }
        if self.timeline.len() > MAX_TIMELINE {
            self.timeline.remove(0);
        }
        self.current = Some(key);
    }

    /// The key governing stream byte `offset`: the last entry published at or
    /// before it.
    pub fn key_at(&self, offset: u64) -> Option<ContentKey> {
        self.timeline
            .iter()
            .rev()
            .find(|(at, _)| *at <= offset)
            .map(|(_, k)| *k)
    }

    /// Drop the rotation timeline at a flush: the byte coordinate restarts at
    /// zero on both sides of a seek, so stale boundaries must not linger. Keys
    /// registered by KID and the current key survive (they are position-free).
    pub fn reset_timeline(&mut self) {
        self.timeline.clear();
    }

    /// Register a content key for a key identifier (clear-key: the container's
    /// `tenc` / `seig` KID names which key a sample needs).
    pub fn insert_kid(&mut self, kid: [u8; 16], key: [u8; 16]) {
        match self.by_kid.iter_mut().find(|(k, _)| *k == kid) {
            Some((_, slot)) => *slot = key,
            None => self.by_kid.push((kid, key)),
        }
    }

    /// The key for a sample with `kid` at stream byte `offset`: KID first, then
    /// the rotation timeline, then the current key.
    #[cfg_attr(
        not(any(feature = "hls", feature = "mp4-cenc")),
        allow(dead_code, reason = "called only by the feature-gated decryptor")
    )]
    pub(crate) fn resolve(&self, kid: &[u8; 16], offset: u64) -> Option<[u8; 16]> {
        if let Some((_, key)) = self.by_kid.iter().find(|(k, _)| k == kid) {
            return Some(*key);
        }
        self.key_at(offset)
            .or(self.current)
            .map(|k: ContentKey| k.key)
    }
}

/// Shared key store a publisher fills and a demuxer reads.
pub type CencKeyHandle = Arc<Mutex<CencKeyStore>>;

/// A fresh, empty key store to wire a publisher and a demuxer together.
pub fn new_key_handle() -> CencKeyHandle {
    Arc::new(Mutex::new(CencKeyStore::default()))
}

// -- box parsing ------------------------------------------------------------

/// Read the protection defaults out of a `sinf`: `schm` names the scheme and
/// `schi/tenc` carries the pattern, per-sample IV size, default KID and (for a
/// constant-IV scheme) the constant IV. An unsupported scheme fails loud.
pub(crate) fn parse_sinf(sinf: &[u8]) -> Result<CencDefaults, G2gError> {
    let schm = find_box(sinf, b"schm").ok_or(G2gError::CapsMismatch)?;
    let scheme = match schm.get(4..8) {
        Some(b"cbcs") => CencScheme::Cbcs,
        Some(b"cbc1") => CencScheme::Cbc1,
        Some(b"cenc") => CencScheme::Cenc,
        Some(b"cens") => CencScheme::Cens,
        _ => return Err(G2gError::CapsMismatch),
    };
    let schi = find_box(sinf, b"schi").ok_or(G2gError::CapsMismatch)?;
    let tenc = find_box(schi, b"tenc").ok_or(G2gError::CapsMismatch)?;
    let version = *tenc.first().ok_or(G2gError::CapsMismatch)?;
    // tenc payload: version/flags(4), reserved(1), pattern(1, v1+), isProtected(1),
    // Per_Sample_IV_Size(1), KID(16), then the constant IV when there is one.
    let mut d = parse_protection_fields(tenc.get(4..).ok_or(G2gError::CapsMismatch)?, scheme)?;
    if version == 0 {
        // v0 has no pattern field: full-sample / full-block encryption.
        d.crypt_byte_block = 0;
        d.skip_byte_block = 0;
    }
    Ok(d)
}

/// The 20-byte protection record shared by `tenc` (after its version/flags) and a
/// `seig` sample group entry: reserved, crypt/skip pattern, isProtected,
/// Per_Sample_IV_Size, KID, and the constant IV when the IV is not per-sample.
fn parse_protection_fields(b: &[u8], scheme: CencScheme) -> Result<CencDefaults, G2gError> {
    // The first byte is reserved (0) in both boxes except that a multi-key `seig`
    // entry signals itself there, and the two descriptions of the multi-key layout
    // put the flag in different bits (the 2016 MPEG proposal in bit 7, GPAC's
    // on-disk key-info blob in bit 0). The record that follows a set flag is a
    // different shape, so any nonzero value is declined rather than read at what
    // would be the wrong offsets.
    if *b.first().ok_or(G2gError::CapsMismatch)? != 0 {
        return Err(G2gError::CapsMismatch);
    }
    let packed = *b.get(1).ok_or(G2gError::CapsMismatch)?;
    let is_protected = b.get(2) == Some(&1);
    let per_sample_iv_size = *b.get(3).ok_or(G2gError::CapsMismatch)?;
    if !matches!(per_sample_iv_size, 0 | 8 | 16) {
        return Err(G2gError::CapsMismatch);
    }
    let kid: [u8; 16] = b
        .get(4..20)
        .ok_or(G2gError::CapsMismatch)?
        .try_into()
        .expect("16 bytes");
    let constant_iv = if is_protected && per_sample_iv_size == 0 {
        let size = *b.get(20).ok_or(G2gError::CapsMismatch)? as usize;
        if size != 16 && size != 8 {
            return Err(G2gError::CapsMismatch);
        }
        b.get(21..21 + size).ok_or(G2gError::CapsMismatch)?.to_vec()
    } else {
        Vec::new()
    };
    Ok(CencDefaults {
        scheme,
        is_protected,
        crypt_byte_block: packed >> 4,
        skip_byte_block: packed & 0x0F,
        per_sample_iv_size,
        constant_iv,
        kid,
    })
}

/// Read the track's movie-level `seig` table out of its `stbl`: the `sgpd` whose
/// grouping type is `seig`, parsed with the track's scheme. A track without one
/// yields an empty table (only fragment-local groups can then apply).
pub(crate) fn parse_movie_seig(
    stbl: &[u8],
    scheme: CencScheme,
) -> Result<Vec<CencDefaults>, G2gError> {
    match find_grouping(stbl, b"sgpd", b"seig") {
        Some(sgpd) => parse_sgpd_entries(sgpd, scheme),
        None => Ok(Vec::new()),
    }
}

/// Resolve the per-sample crypto for one fragment's `sample_count` samples.
///
/// `from_moof` is the fragment bytes starting at the enclosing `moof` box header
/// (a `moof` immediately followed by its `mdat`, which is how CMAF stores them
/// and how the demuxers buffer them): `saio` offsets are anchored there, per
/// ISO/IEC 14496-12 for a `default-base-is-moof` fragment. An offset that lands
/// outside the fragment fails the parse.
///
/// The sample auxiliary information is taken from `saiz`+`saio` when both are
/// present (they locate it authoritatively, including a non-contiguous layout),
/// otherwise from the `senc` records. `seig` sample groups override the track
/// defaults over a run of samples, including marking samples clear.
pub(crate) fn fragment_sample_crypt(
    traf: &[u8],
    from_moof: &[u8],
    track: &CencTrack,
    sample_count: usize,
) -> Result<Vec<SampleCrypt>, G2gError> {
    let defaults = &track.defaults;
    let groups = parse_seig(traf, track, sample_count)?;
    // Per-sample protection settings before the aux info is folded in.
    let settings: Vec<&CencDefaults> = (0..sample_count)
        .map(|i| match groups.get(i) {
            Some(Some(g)) => g,
            _ => defaults,
        })
        .collect();

    let aux = sample_aux_info(traf, from_moof, &settings)?;
    let mut out = Vec::with_capacity(sample_count);
    for (i, d) in settings.iter().enumerate() {
        if !d.is_protected {
            out.push(SampleCrypt::clear());
            continue;
        }
        let blob = aux.get(i).copied().unwrap_or(&[]);
        out.push(sample_crypt_from_aux(blob, d)?);
    }
    Ok(out)
}

/// Fold one sample's auxiliary information (`[IV][subsample map]`) with the
/// protection settings in force for it.
fn sample_crypt_from_aux(blob: &[u8], d: &CencDefaults) -> Result<SampleCrypt, G2gError> {
    let iv_size = d.per_sample_iv_size as usize;
    let mut iv = [0u8; 16];
    if iv_size == 0 {
        // Constant IV: a short (8-byte) constant IV is used as the high half of
        // the 16-byte block, the low half zero.
        let civ = &d.constant_iv;
        if civ.is_empty() {
            return Err(G2gError::CapsMismatch);
        }
        iv[..civ.len()].copy_from_slice(civ);
    } else {
        let raw = blob.get(..iv_size).ok_or(G2gError::CapsMismatch)?;
        // A per-sample IV shorter than the block is the high half, zero-extended
        // (an 8-byte cenc IV is the counter's high 64 bits, block counter 0).
        iv[..iv_size].copy_from_slice(raw);
    }
    let mut subsamples = Vec::new();
    if blob.len() > iv_size {
        let rest = &blob[iv_size..];
        let count = u16::from_be_bytes(
            rest.get(0..2)
                .ok_or(G2gError::CapsMismatch)?
                .try_into()
                .expect("2 bytes"),
        ) as usize;
        // Each entry is 6 bytes; a count larger than the blob holds is a lie.
        if count
            .checked_mul(6)
            .is_none_or(|n| n > rest.len().saturating_sub(2))
        {
            return Err(G2gError::CapsMismatch);
        }
        subsamples.reserve(count);
        for e in 0..count {
            let at = 2 + e * 6;
            let clear = u16::from_be_bytes(
                rest.get(at..at + 2)
                    .ok_or(G2gError::CapsMismatch)?
                    .try_into()
                    .expect("2 bytes"),
            ) as u32;
            let protected = be32(rest, at + 2)?;
            subsamples.push(Subsample { clear, protected });
        }
    }
    Ok(SampleCrypt {
        kid: d.kid,
        scheme: d.scheme,
        crypt_byte_block: d.crypt_byte_block,
        skip_byte_block: d.skip_byte_block,
        iv,
        subsamples,
        protected: true,
    })
}

/// Locate each sample's auxiliary information blob: from `saiz`+`saio` when both
/// are present, else from the `senc` records. An absent aux info (a fragment that
/// relies on the `tenc` constant IV with no subsample map) yields no blobs, which
/// [`sample_crypt_from_aux`] reads as "whole sample protected".
fn sample_aux_info<'a>(
    traf: &'a [u8],
    from_moof: &'a [u8],
    settings: &[&CencDefaults],
) -> Result<Vec<&'a [u8]>, G2gError> {
    let senc = find_box(traf, b"senc");
    if moof_anchored(traf)? {
        if let (Some(saiz), Some(saio)) = (find_box(traf, b"saiz"), find_box(traf, b"saio")) {
            return aux_from_saiz_saio(saiz, saio, from_moof);
        }
    }
    match senc {
        Some(senc) => aux_from_senc(senc, settings),
        None => Ok(Vec::new()),
    }
}

/// Whether this fragment addresses its data from the start of the enclosing
/// `moof`, which is what makes a `saio` offset resolvable here: the `tfhd`
/// declares `default-base-is-moof` (0x020000) and no explicit `base_data_offset`
/// (0x000001, an absolute file position a streamed fragment cannot resolve).
/// A fragment with no `tfhd` at all cannot be anchored either.
fn moof_anchored(traf: &[u8]) -> Result<bool, G2gError> {
    let Some(tfhd) = find_box(traf, b"tfhd") else {
        return Ok(false);
    };
    let flags = be32(tfhd, 0)? & 0x00FF_FFFF;
    Ok(flags & 0x02_0000 != 0 && flags & 0x1 == 0)
}

/// Skip a `saiz` / `saio` full box's optional `aux_info_type` +
/// `aux_info_type_parameter` pair (present when `flags & 1`), returning the
/// payload after the version/flags and that pair.
fn after_aux_info_type(b: &[u8]) -> Result<&[u8], G2gError> {
    let flags = be32(b, 0)? & 0x00FF_FFFF;
    let at = if flags & 1 != 0 { 12 } else { 4 };
    b.get(at..).ok_or(G2gError::CapsMismatch)
}

/// Per-sample aux blobs located by `saiz` (sizes) and `saio` (offsets from the
/// start of the enclosing `moof`). `saio` carries either one offset (the blobs
/// run contiguously from it) or one per sample (a non-contiguous layout).
fn aux_from_saiz_saio<'a>(
    saiz: &[u8],
    saio: &[u8],
    from_moof: &'a [u8],
) -> Result<Vec<&'a [u8]>, G2gError> {
    let sizes = {
        let b = after_aux_info_type(saiz)?;
        let default_size = *b.first().ok_or(G2gError::CapsMismatch)? as usize;
        let count = be32(b, 1)? as usize;
        if default_size == 0 {
            // One size byte per sample must actually be present.
            let raw = b.get(5..5 + count).ok_or(G2gError::CapsMismatch)?;
            raw.iter().map(|&s| s as usize).collect::<Vec<_>>()
        } else {
            // A uniform size still has to fit the fragment, checked below per entry.
            if count > from_moof.len() {
                return Err(G2gError::CapsMismatch);
            }
            Vec::from_iter(core::iter::repeat_n(default_size, count))
        }
    };
    let offsets = {
        let b = after_aux_info_type(saio)?;
        let version = *saio.first().ok_or(G2gError::CapsMismatch)?;
        let count = be32(b, 0)? as usize;
        let width = if version == 0 { 4 } else { 8 };
        if count
            .checked_mul(width)
            .is_none_or(|n| n > b.len().saturating_sub(4))
        {
            return Err(G2gError::CapsMismatch);
        }
        (0..count)
            .map(|i| {
                let at = 4 + i * width;
                if width == 4 {
                    be32(b, at).map(u64::from)
                } else {
                    be64(b, at)
                }
            })
            .collect::<Result<Vec<u64>, _>>()?
    };
    // One offset covers all samples contiguously; otherwise there must be one
    // offset per sample. Anything else is a layout we cannot address safely.
    if offsets.len() != 1 && offsets.len() != sizes.len() {
        return Err(G2gError::CapsMismatch);
    }
    let mut out = Vec::with_capacity(sizes.len());
    let mut running = *offsets.first().ok_or(G2gError::CapsMismatch)?;
    for (i, &size) in sizes.iter().enumerate() {
        let at = if offsets.len() == 1 {
            running
        } else {
            offsets[i]
        };
        let start = usize::try_from(at).map_err(|_| G2gError::CapsMismatch)?;
        let end = start.checked_add(size).ok_or(G2gError::CapsMismatch)?;
        out.push(from_moof.get(start..end).ok_or(G2gError::CapsMismatch)?);
        running = at.checked_add(size as u64).ok_or(G2gError::CapsMismatch)?;
    }
    Ok(out)
}

/// Per-sample aux blobs read straight out of a `senc`: version/flags, sample
/// count, then per sample the IV and (when `flags & 2`) the subsample map. The
/// IV size can differ per sample when a `seig` group overrides it, so the records
/// are walked with each sample's own settings.
fn aux_from_senc<'a>(
    senc: &'a [u8],
    settings: &[&CencDefaults],
) -> Result<Vec<&'a [u8]>, G2gError> {
    // A nonzero version carries the multi-key record layout (per-sample key
    // indices, an IV whose length comes from a key list in the `seig` entry, and
    // wider subsample counts). Only one implementation writes it and the `seig`
    // half of the layout could not be established, so reading it as version 0
    // would mis-slice every sample: decline.
    if senc.first() != Some(&0) {
        return Err(G2gError::CapsMismatch);
    }
    let flags = be32(senc, 0)? & 0x00FF_FFFF;
    let has_subsamples = flags & 0x2 != 0;
    let count = be32(senc, 4)? as usize;
    let mut at = 8usize;
    // A version-0 `senc` describes exactly the fragment's samples, and a record
    // can be empty (constant IV, no subsample map), so the byte length is no
    // bound at all: hold the count to the sample count instead.
    if count > settings.len() {
        return Err(G2gError::CapsMismatch);
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let iv_size = settings.get(i).map_or(0, |d| d.per_sample_iv_size as usize);
        let mut len = iv_size;
        if has_subsamples {
            let sub_count = u16::from_be_bytes(
                senc.get(at + len..at + len + 2)
                    .ok_or(G2gError::CapsMismatch)?
                    .try_into()
                    .expect("2 bytes"),
            ) as usize;
            let map = sub_count.checked_mul(6).ok_or(G2gError::CapsMismatch)?;
            len = len.checked_add(2 + map).ok_or(G2gError::CapsMismatch)?;
        }
        let end = at.checked_add(len).ok_or(G2gError::CapsMismatch)?;
        out.push(senc.get(at..end).ok_or(G2gError::CapsMismatch)?);
        at = end;
    }
    Ok(out)
}

/// Per-sample `seig` overrides: `sbgp` maps runs of samples to `sgpd` entries,
/// each a protection record that replaces the track defaults for those samples
/// (a different KID, IV size or pattern, or `isProtected = 0` for clear samples).
/// `None` for a sample means "use the track defaults". Absent boxes yield an
/// empty map.
///
/// Index 0 means the sample is in no group. Per ISO/IEC 14496-12 a
/// `group_description_index` of 0x10000 or more names an entry of the `traf`'s own
/// `sgpd` (local index = index - 0x10000), and a smaller one an entry of the
/// movie-level table in the track's `stbl`. An index whose table is absent fails
/// the parse: falling back to the other table, or to the track defaults, could
/// decrypt a sample the group left clear or key it wrongly.
fn parse_seig(
    traf: &[u8],
    track: &CencTrack,
    sample_count: usize,
) -> Result<Vec<Option<CencDefaults>>, G2gError> {
    let Some(sbgp) = find_grouping(traf, b"sbgp", b"seig") else {
        return Ok(Vec::new());
    };
    let local = match find_grouping(traf, b"sgpd", b"seig") {
        Some(sgpd) => parse_sgpd_entries(sgpd, track.defaults.scheme)?,
        None => Vec::new(),
    };

    let version = *sbgp.first().ok_or(G2gError::CapsMismatch)?;
    // payload: version/flags(4), grouping_type(4), grouping_type_parameter(4, v1),
    // entry_count(4), then (sample_count, group_description_index) pairs.
    let mut at = if version == 1 { 12 } else { 8 };
    let count = be32(sbgp, at)? as usize;
    at += 4;
    if count
        .checked_mul(8)
        .is_none_or(|n| n > sbgp.len().saturating_sub(at))
    {
        return Err(G2gError::CapsMismatch);
    }
    let mut out = Vec::new();
    for _ in 0..count {
        let run = be32(sbgp, at)? as usize;
        let index = be32(sbgp, at + 4)?;
        at += 8;
        // The runs describe this fragment's samples: mapping more than it has is
        // a lie, and one long enough would otherwise allocate on it. Fewer is
        // fine, the rest fall back to the track defaults.
        if run > sample_count.saturating_sub(out.len()) {
            return Err(G2gError::CapsMismatch);
        }
        let entry = match index {
            0 => None,
            i => {
                let (table, at) = if i >= 0x1_0000 {
                    (&local, i - 0x1_0000)
                } else {
                    (&track.movie_seig, i)
                };
                let e = table
                    .get((at as usize).checked_sub(1).ok_or(G2gError::CapsMismatch)?)
                    .ok_or(G2gError::CapsMismatch)?;
                Some(e.clone())
            }
        };
        out.extend(core::iter::repeat_n(entry, run));
    }
    Ok(out)
}

/// Find a sample-group box (`sbgp` / `sgpd`) whose `grouping_type` matches; a
/// `traf` (or `stbl`) can carry several for different grouping types.
fn find_grouping<'a>(container: &'a [u8], kind: &[u8; 4], grouping: &[u8; 4]) -> Option<&'a [u8]> {
    crate::mp4box::boxes(container)
        .filter(|(k, _)| *k == kind)
        .find(|(_, b)| b.get(4..8) == Some(&grouping[..]))
        .map(|(_, b)| b)
}

/// `sgpd` entries for grouping type `seig`: version/flags(4), grouping_type(4),
/// `default_length`(4, v1+), `default_sample_description_index`(4, v2+),
/// entry_count(4), then the entries (each preceded by its own length when v1
/// declares `default_length == 0`).
fn parse_sgpd_entries(sgpd: &[u8], scheme: CencScheme) -> Result<Vec<CencDefaults>, G2gError> {
    let version = *sgpd.first().ok_or(G2gError::CapsMismatch)?;
    if version > 1 {
        // v2 inserts a default sample description index, but the editions of
        // 14496-12 disagree on whether `default_length` and the per-entry lengths
        // survive alongside it (2015 has them only at v1, 2022 from v1 up), so the
        // two layouts differ in where the entries start. Decline rather than
        // mis-read: v1 is what CENC packagers write.
        return Err(G2gError::CapsMismatch);
    }
    let mut at = 8usize;
    let default_length = if version == 1 {
        let l = be32(sgpd, at)? as usize;
        at += 4;
        l
    } else {
        0
    };
    let count = be32(sgpd, at)? as usize;
    at += 4;
    // A seig entry is at least 20 bytes, so the count cannot exceed what remains.
    if count > sgpd.len().saturating_sub(at) / 20 {
        return Err(G2gError::CapsMismatch);
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let len = if version == 1 && default_length == 0 {
            let l = be32(sgpd, at)? as usize;
            at += 4;
            l
        } else if version == 1 {
            default_length
        } else {
            // v0 has no length: the rest of the box is one entry's worth.
            sgpd.len().saturating_sub(at)
        };
        let body = sgpd.get(at..at + len).ok_or(G2gError::CapsMismatch)?;
        // A seig entry has the same shape as a tenc's fields from its reserved
        // byte on: reserved, pattern, isProtected, IV size, KID, constant IV.
        out.push(parse_protection_fields(body, scheme)?);
        at += len;
    }
    Ok(out)
}

// -- decryption -------------------------------------------------------------

/// Decrypt one sample in place under `key`, following its resolved
/// [`SampleCrypt`]: an empty subsample map protects the whole sample, otherwise
/// each subsample's `protected` run is decrypted and its `clear` run passes
/// through. A clear sample is left untouched.
///
/// Errors when the subsample map describes more bytes than the sample holds: the
/// map then does not belong to this sample, and decrypting the part that fits
/// would emit the rest as if it were plaintext.
#[cfg(any(feature = "hls", feature = "mp4-cenc"))]
pub(crate) fn decrypt_sample(
    buf: &mut [u8],
    crypt: &SampleCrypt,
    key: &[u8; 16],
) -> Result<(), G2gError> {
    if !crypt.protected {
        return Ok(());
    }
    // `cenc` (AES-CTR) and `cbc1` (AES-CBC) both run one continuous cipher over
    // the sample's protected bytes, skipping the clear runs, so those gather the
    // ranges first. `cens` gathers too, but only the pattern's encrypted blocks:
    // its counter advances over those alone. `cbcs` restarts from the IV at each
    // protected range, so it works range by range.
    let ranges = protected_ranges(buf.len(), &crypt.subsamples).ok_or(G2gError::CapsMismatch)?;
    let pattern = (crypt.crypt_byte_block, crypt.skip_byte_block);
    match crypt.scheme {
        CencScheme::Cenc => gathered(buf, &ranges, |g| {
            ctr_decrypt(g, key, &crypt.iv);
        }),
        CencScheme::Cens => {
            // With no pattern `cens` is `cenc`: the whole protected range, trailing
            // partial block included.
            let blocks = match pattern {
                (0, _) | (_, 0) => ranges.clone(),
                (c, s) => pattern_blocks(&ranges, c, s),
            };
            gathered(buf, &blocks, |g| {
                ctr_decrypt(g, key, &crypt.iv);
            });
        }
        CencScheme::Cbc1 => gathered(buf, &ranges, |g| {
            cbc_decrypt_blocks(g, key, &crypt.iv, 0, 0);
        }),
        CencScheme::Cbcs => {
            for &(start, end) in &ranges {
                cbc_decrypt_blocks(
                    &mut buf[start..end],
                    key,
                    &crypt.iv,
                    crypt.crypt_byte_block,
                    crypt.skip_byte_block,
                );
            }
        }
    }
    Ok(())
}

/// AES-CTR over `g` from `iv` as the initial counter block. A short per-sample IV
/// has already been zero-extended into the low half, which is the block counter.
#[cfg(any(feature = "hls", feature = "mp4-cenc"))]
fn ctr_decrypt(g: &mut [u8], key: &[u8; 16], iv: &[u8; 16]) {
    use aes::cipher::{KeyIvInit, StreamCipher};
    type Ctr = ctr::Ctr128BE<aes::Aes128>;
    Ctr::new(&(*key).into(), &(*iv).into()).apply_keystream(g);
}

/// The extents a crypt:skip block pattern encrypts within each protected range,
/// as absolute `(start, end)` pairs (one per pattern cycle's crypt span).
///
/// The pattern restarts at the head of every protected range, per ISO/IEC 23001-7
/// §9.6.1: only whole 16-byte blocks are encrypted, so a range's trailing partial
/// block stays clear even when the pattern would cover it. A final crypt span the
/// range truncates keeps the whole blocks it does hold (what GPAC and Shaka
/// Packager do; ffmpeg drops that span instead, a divergence conformant content
/// avoids because 23001-7 §10.3 requires 16-byte-aligned protected ranges).
#[cfg(any(feature = "hls", feature = "mp4-cenc"))]
fn pattern_blocks(ranges: &[(usize, usize)], crypt: u8, skip: u8) -> Vec<(usize, usize)> {
    let span = (crypt as usize + skip as usize) * 16;
    let crypt_len = crypt as usize * 16;
    let mut out = Vec::new();
    for &(start, end) in ranges {
        let mut at = start;
        while at < end {
            // Whole blocks only: truncate the crypt span to what the range holds.
            let take = crypt_len.min((end - at) / 16 * 16);
            if take == 0 {
                break;
            }
            out.push((at, at + take));
            at = at.saturating_add(span);
        }
    }
    out
}

/// Run `cipher` over the sample's protected bytes as one contiguous buffer, then
/// scatter them back: the schemes whose cipher state carries across subsamples
/// (the `cenc` counter, the `cbc1` chain) need the protected bytes adjacent.
#[cfg(any(feature = "hls", feature = "mp4-cenc"))]
fn gathered(buf: &mut [u8], ranges: &[(usize, usize)], cipher: impl FnOnce(&mut [u8])) {
    let mut g: Vec<u8> = ranges
        .iter()
        .flat_map(|&(s, e)| buf[s..e].iter().copied())
        .collect();
    if g.is_empty() {
        return;
    }
    cipher(&mut g);
    let mut at = 0usize;
    for &(s, e) in ranges {
        let n = e - s;
        buf[s..e].copy_from_slice(&g[at..at + n]);
        at += n;
    }
}

/// The protected byte ranges of a sample. An empty subsample map means the whole
/// sample is protected. `None` when the map runs past the end of the sample: it
/// describes a different sample than the one in hand, so no part of this one can
/// be trusted to be where the map says.
#[cfg(any(feature = "hls", feature = "mp4-cenc"))]
fn protected_ranges(len: usize, subsamples: &[Subsample]) -> Option<Vec<(usize, usize)>> {
    if subsamples.is_empty() {
        return Some(Vec::from([(0, len)]));
    }
    let mut out = Vec::with_capacity(subsamples.len());
    let mut pos = 0usize;
    for s in subsamples {
        pos = pos.checked_add(s.clear as usize)?;
        let end = pos.checked_add(s.protected as usize)?;
        if end > len {
            return None;
        }
        if pos < end {
            out.push((pos, end));
        }
        pos = end;
    }
    Some(out)
}

/// AES-CBC over a protected run: with a `crypt`:`skip` pattern only the crypt
/// blocks of each span are deciphered (the `cbcs` shape), with a zero pattern
/// every whole block is. Chaining runs over the deciphered blocks only, from the
/// IV; a trailing partial block is left clear.
#[cfg(any(feature = "hls", feature = "mp4-cenc"))]
fn cbc_decrypt_blocks(run: &mut [u8], key: &[u8; 16], iv: &[u8; 16], crypt: u8, skip: u8) {
    use aes::cipher::{block_padding::NoPadding, BlockDecryptMut, KeyIvInit};
    type Dec = cbc::Decryptor<aes::Aes128>;

    let block_count = run.len() / 16;
    let offsets: Vec<usize> = if crypt != 0 && skip != 0 {
        let span = (crypt as usize) + (skip as usize);
        (0..block_count)
            .filter(|b| b % span < crypt as usize)
            .map(|b| b * 16)
            .collect()
    } else {
        (0..block_count).map(|b| b * 16).collect()
    };
    if offsets.is_empty() {
        return;
    }
    let mut g: Vec<u8> = offsets
        .iter()
        .flat_map(|&o| run[o..o + 16].iter().copied())
        .collect();
    Dec::new(&(*key).into(), &(*iv).into())
        .decrypt_padded_mut::<NoPadding>(&mut g)
        .expect("gathered whole blocks");
    for (i, &o) in offsets.iter().enumerate() {
        run[o..o + 16].copy_from_slice(&g[i * 16..i * 16 + 16]);
    }
}

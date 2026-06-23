//! Fragmented-MP4 / CMAF byte-stream demuxer (Fmp4Demux): `ByteStream{IsoBmff}`
//! in, `CompressedVideo{H264|H265}` Annex-B access units out. The streaming
//! counterpart of the file-based [`Mp4Src`](crate::mp4src); both share the
//! [`fmp4`](crate::fmp4) parser. This is what an HLS/DASH fMP4 segment stream
//! (init segment + media fragments) feeds into, the analog of `tsdemux` for the
//! TS path.
//!
//! Bytes arrive in arbitrary chunks (whole segments, or split mid-box by a
//! generic source), so it buffers and processes one complete top-level box at a
//! time: `moov` yields the codec/geometry (emitted as `CapsChanged`) and the
//! parameter sets; each `moof`+`mdat` pair yields samples. The out-of-band
//! parameter sets are prepended to the first emitted access unit so a decoder can
//! start. Single video track; the profile `Mp4Sink` writes (see `fmp4`).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, Rate, VideoCodec,
};

use crate::fmp4::{parse_fragments, parse_header, starts_with_param_set, CencDefaults, Header, Sample};
#[cfg(feature = "hls")]
use crate::fmp4::Subsample;

#[derive(Debug)]
pub struct Fmp4Demux {
    buffer: Vec<u8>,
    header: Option<Header>,
    /// A `moof` box awaiting its following `mdat` to form a complete fragment.
    pending_moof: Option<Vec<u8>>,
    /// Negotiation-time output codec (refined from the `moov` at runtime).
    out_codec: VideoCodec,
    /// Prepend the config-record parameter sets to the first access unit.
    need_param_sets: bool,
    /// cbcs decryption key (shared with `HlsSrc`); the constant IV comes from the
    /// init segment's `tenc`. Without it an encrypted track fails loud.
    #[cfg(feature = "hls")]
    cbcs_key: Option<crate::sampleaesdecrypt::SampleAesKeyHandle>,
    caps_sent: bool,
    sequence: u64,
    configured: bool,
}

impl Default for Fmp4Demux {
    fn default() -> Self {
        Self::new()
    }
}

impl Fmp4Demux {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            header: None,
            pending_moof: None,
            out_codec: VideoCodec::H264,
            need_param_sets: true,
            #[cfg(feature = "hls")]
            cbcs_key: None,
            caps_sent: false,
            sequence: 0,
            configured: false,
        }
    }

    /// Share the cbcs key handle a `HlsSrc` publishes into (the auto-wired HLS
    /// SAMPLE-AES path for fMP4/CMAF). The decryptor pairs it with the constant
    /// IV from the segment's `tenc`.
    #[cfg(feature = "hls")]
    pub fn with_cbcs_key_handle(
        mut self,
        handle: crate::sampleaesdecrypt::SampleAesKeyHandle,
    ) -> Self {
        self.cbcs_key = Some(handle);
        self
    }

    /// Parse a fragment's samples, decrypting in place when the track is cbcs
    /// (the key from the shared handle, the constant IV + pattern from `tenc`).
    #[cfg(feature = "hls")]
    fn parse_fragment_samples(
        &self,
        frag: &[u8],
        timescale: u32,
        codec: VideoCodec,
        cenc: Option<&CencDefaults>,
    ) -> Result<Vec<Sample>, G2gError> {
        let Some(c) = cenc else {
            return parse_fragments(frag, timescale, codec, None, None);
        };
        let key = self
            .cbcs_key
            .as_ref()
            .and_then(|h| *h.lock().expect("key handle poisoned"))
            .map(|k| k.key)
            .ok_or(G2gError::CapsMismatch)?;
        let iv: [u8; 16] =
            c.constant_iv.as_slice().try_into().map_err(|_| G2gError::CapsMismatch)?;
        let (crypt, skip) = (c.crypt_byte_block, c.skip_byte_block);
        let mut decrypt = move |buf: &mut [u8], subs: &[Subsample]| {
            cbcs_decrypt_sample(buf, subs, &key, &iv, crypt, skip);
        };
        parse_fragments(frag, timescale, codec, Some(c), Some(&mut decrypt))
    }

    /// Without the `hls` feature there is no AES: an encrypted track fails loud.
    #[cfg(not(feature = "hls"))]
    fn parse_fragment_samples(
        &self,
        frag: &[u8],
        timescale: u32,
        codec: VideoCodec,
        cenc: Option<&CencDefaults>,
    ) -> Result<Vec<Sample>, G2gError> {
        parse_fragments(frag, timescale, codec, cenc, None)
    }

    fn input_caps() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }
    }

    fn output_caps(codec: VideoCodec, width: Dim, height: Dim) -> Caps {
        Caps::CompressedVideo { codec, width, height, framerate: Rate::Any }
    }

    /// Process every complete top-level box now buffered, emitting access units.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        while let Some(total) = next_box_len(&self.buffer) {
            if self.buffer.len() < total {
                break; // wait for the rest of this box
            }
            let box_bytes: Vec<u8> = self.buffer.drain(..total).collect();
            let kind: [u8; 4] = box_bytes[4..8].try_into().expect("8-byte box header");
            match &kind {
                b"moov" => {
                    let header = parse_header(&box_bytes)?;
                    let caps = Self::output_caps(
                        header.codec,
                        Dim::Fixed(header.width),
                        Dim::Fixed(header.height),
                    );
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                    self.out_codec = header.codec;
                    self.caps_sent = true;
                    self.header = Some(header);
                }
                b"moof" => self.pending_moof = Some(box_bytes),
                b"mdat" => {
                    let Some(mut frag) = self.pending_moof.take() else {
                        return Err(G2gError::CapsMismatch); // mdat without moof
                    };
                    // header must exist (moov precedes the first fragment)
                    let Some(header) = self.header.as_ref() else {
                        return Err(G2gError::CapsMismatch);
                    };
                    let (timescale, codec) = (header.timescale, header.codec);
                    let param_sets = header.param_sets.clone();
                    let cenc = header.cenc.clone();

                    frag.extend_from_slice(&box_bytes);
                    let samples = self.parse_fragment_samples(&frag, timescale, codec, cenc.as_ref())?;
                    for s in samples {
                        let mut annexb = s.annexb;
                        if self.need_param_sets && !starts_with_param_set(&annexb, codec) {
                            let mut with_sets = Vec::new();
                            for set in &param_sets {
                                with_sets.extend_from_slice(&[0, 0, 0, 1]);
                                with_sets.extend_from_slice(set);
                            }
                            with_sets.extend_from_slice(&annexb);
                            annexb = with_sets;
                        }
                        self.need_param_sets = false;
                        let frame = Frame {
                            domain: MemoryDomain::System(SystemSlice::from_boxed(
                                annexb.into_boxed_slice(),
                            )),
                            timing: FrameTiming {
                                pts_ns: s.pts_ns,
                                dts_ns: s.pts_ns,
                                duration_ns: s.duration_ns,
                                capture_ns: s.pts_ns,
                                arrival_ns: g2g_core::metrics::monotonic_ns(),
                                keyframe: s.keyframe, // fMP4 trun keyframe flag
                            },
                            sequence: self.sequence,
                            meta: Default::default(),
                        };
                        self.sequence += 1;
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                    }
                }
                // ftyp / styp / sidx / free / etc.: not needed for demux
                _ => {}
            }
        }
        Ok(())
    }
}

/// Total length of the box at the start of `buf`, or `None` if the 8-byte header
/// (or the 64-bit large-size header) isn't fully buffered yet. A size below 8
/// (including the size-0 "to end of stream" form) can't be framed and returns
/// `None`; the writer profile we consume always uses explicit sizes.
fn next_box_len(buf: &[u8]) -> Option<usize> {
    if buf.len() < 8 {
        return None;
    }
    let size = u32::from_be_bytes(buf[0..4].try_into().expect("4 bytes"));
    let total = if size == 1 {
        if buf.len() < 16 {
            return None;
        }
        u64::from_be_bytes(buf[8..16].try_into().expect("8 bytes")) as usize
    } else {
        size as usize
    };
    (total >= 8).then_some(total)
}

/// Decrypt one cbcs sample in place: walk its `senc` subsamples, decrypting each
/// protected range (an empty map means the whole sample is one protected range).
#[cfg(feature = "hls")]
fn cbcs_decrypt_sample(
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
#[cfg(feature = "hls")]
fn decrypt_protected_range(range: &mut [u8], key: &[u8; 16], iv: &[u8; 16], crypt: u8, skip: u8) {
    use aes::cipher::{block_padding::NoPadding, BlockDecryptMut, KeyIvInit};
    type Dec = cbc::Decryptor<aes::Aes128>;

    let block_count = range.len() / 16;
    let offsets: Vec<usize> = if crypt == 0 || skip == 0 {
        (0..block_count).map(|b| b * 16).collect()
    } else {
        let span = (crypt + skip) as usize;
        (0..block_count).filter(|b| b % span < crypt as usize).map(|b| b * 16).collect()
    };
    if offsets.is_empty() {
        return;
    }
    let mut gathered: Vec<u8> =
        offsets.iter().flat_map(|&o| range[o..o + 16].iter().copied()).collect();
    Dec::new(&(*key).into(), &(*iv).into())
        .decrypt_padded_mut::<NoPadding>(&mut gathered)
        .expect("cbcs region is block-aligned");
    for (i, &o) in offsets.iter().enumerate() {
        range[o..o + 16].copy_from_slice(&gathered[i * 16..i * 16 + 16]);
    }
}

impl AsyncElement for Fmp4Demux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // ByteStream{IsoBmff} in -> the video track out. The default codec is
        // refined from the moov via CapsChanged at runtime (like tsdemux).
        let codec = self.out_codec;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff } => {
                CapsSet::one(Self::output_caps(codec, Dim::Any, Dim::Any))
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(absolute_caps, Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "fMP4 / CMAF demuxer",
            "Codec/Demuxer",
            "Demuxes a fragmented-MP4 / CMAF byte stream",
            "g2g",
        )
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.buffer.extend_from_slice(slice.as_slice());
                    self.drain(out).await?;
                }
                // Nothing to flush (incomplete trailing boxes are dropped); the
                // runner's transform arm forwards the EOS itself.
                PipelinePacket::Eos => {}
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for Fmp4Demux {
    fn pad_templates() -> Vec<PadTemplate> {
        let video = |codec| Self::output_caps(codec, Dim::Any, Dim::Any);
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(CapsSet::from_alternatives(Vec::from([
                video(VideoCodec::H264),
                video(VideoCodec::H265),
            ]))),
        ])
    }
}

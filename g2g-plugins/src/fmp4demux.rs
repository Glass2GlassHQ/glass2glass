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
//! start. Single video track; the profile `Mp4Mux` writes (see `fmp4`).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SeekController;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, Rate, Seek, Segment, VideoCodec,
};

use crate::demuxseek::{Admit, DemuxSeek};
#[cfg(feature = "hls")]
use crate::fmp4::Subsample;
use crate::fmp4::{
    parse_fragments, parse_header, prepend_param_sets, starts_with_param_set, CencDefaults, Header,
    Sample,
};

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
    /// Seek support (M362): app time seeks drive an upstream byte-seek and a
    /// re-sync. Inert unless `with_seek` wired the controllers.
    seek: DemuxSeek,
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
            seek: DemuxSeek::default(),
        }
    }

    /// Make the demuxer seekable (M362): `app` carries app time seeks; `upstream`
    /// is the byte source's ([`FileSrc`](crate::filesrc)) byte-seek controller,
    /// which a time seek drives to reposition the stream. On the resulting
    /// `Flush` the parser resets and re-syncs from the keyframe at/after target.
    pub fn with_seek(mut self, app: SeekController, upstream: SeekController) -> Self {
        self.seek.with(app, upstream);
        self
    }

    /// Reset the parser for a discontinuity (a `Flush` / seek): drop buffered
    /// bytes and any half-formed fragment, and re-prepend parameter sets to the
    /// next emitted access unit. The codec / caps are unchanged (same file), so
    /// `header` / `caps_sent` are kept (no redundant `CapsChanged`).
    fn reset_parser(&mut self) {
        self.buffer.clear();
        self.pending_moof = None;
        self.need_param_sets = true;
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
        let iv: [u8; 16] = c
            .constant_iv
            .as_slice()
            .try_into()
            .map_err(|_| G2gError::CapsMismatch)?;
        let (crypt, skip) = (c.crypt_byte_block, c.skip_byte_block);
        let mut decrypt = move |buf: &mut [u8], subs: &[Subsample]| {
            crate::cenc::cbcs_decrypt_sample(buf, subs, &key, &iv, crypt, skip);
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
        Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        }
    }

    fn output_caps(codec: VideoCodec, width: Dim, height: Dim) -> Caps {
        Caps::CompressedVideo {
            codec,
            width,
            height,
            framerate: Rate::Any,
        }
    }

    /// Process every complete top-level box now buffered, emitting access units.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        while let Some(total) = next_box_len(&self.buffer)? {
            if self.buffer.len() < total {
                break; // wait for the rest of this box
            }
            let box_bytes: Vec<u8> = self.buffer.drain(..total).collect();
            let kind: [u8; 4] = box_bytes[4..8].try_into().expect("8-byte box header");
            match &kind {
                b"moov" => {
                    let header = parse_header(&box_bytes)?;
                    // Re-reading the moov on a seek re-parses the header (for
                    // timescale / parameter sets) but does not re-announce the
                    // unchanged caps.
                    if !self.caps_sent {
                        let caps = Self::output_caps(
                            header.codec,
                            Dim::Fixed(header.width),
                            Dim::Fixed(header.height),
                        );
                        out.push(PipelinePacket::CapsChanged(caps)).await?;
                        self.out_codec = header.codec;
                        self.caps_sent = true;
                    }
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
                    let samples =
                        self.parse_fragment_samples(&frag, timescale, codec, cenc.as_ref())?;
                    for s in samples {
                        // M362 seek: drop samples until the keyframe at/after the
                        // target; the resuming keyframe emits a fresh segment.
                        match self.seek.admit(s.pts_ns, s.keyframe) {
                            Admit::Drop => continue,
                            Admit::Resume(start) => {
                                let seg = Segment::for_flush_seek(&Seek::flush_to(start), None);
                                out.push(PipelinePacket::Segment(seg)).await?;
                            }
                            Admit::Emit => {}
                        }
                        let mut annexb = s.annexb;
                        if self.need_param_sets && !starts_with_param_set(&annexb, codec) {
                            annexb = prepend_param_sets(&annexb, &param_sets, codec);
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

/// Total length of the box at the start of `buf`. `Ok(None)` means the 8-byte
/// header (or the 64-bit large-size header) isn't fully buffered yet. Once the
/// size field is in hand, a value below 8 (including the size-0 "to end of
/// stream" form) is malformed and fails loud rather than stalling the demuxer
/// with an unconsumable box.
fn next_box_len(buf: &[u8]) -> Result<Option<usize>, G2gError> {
    if buf.len() < 8 {
        return Ok(None);
    }
    let size = u32::from_be_bytes(buf[0..4].try_into().expect("4 bytes"));
    let total = if size == 1 {
        if buf.len() < 16 {
            return Ok(None);
        }
        u64::from_be_bytes(buf[8..16].try_into().expect("8 bytes")) as usize
    } else {
        size as usize
    };
    if total < 8 {
        return Err(G2gError::CapsMismatch);
    }
    Ok(Some(total))
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
            Caps::ByteStream {
                encoding: ByteStreamEncoding::IsoBmff,
            } => CapsSet::one(Self::output_caps(codec, Dim::Any, Dim::Any)),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::IsoBmff
            }
        ) {
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
            // M362: a pending app seek triggers an upstream byte-seek; until its
            // `Flush` returns, drop input so no stale pre-seek units are emitted.
            self.seek.poll_request();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if self.seek.dropping_input() {
                        return Ok(());
                    }
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.buffer.extend_from_slice(slice.as_slice());
                    self.drain(out).await?;
                }
                // The upstream byte-seek's flush: reset the parser, then re-sync
                // from the re-read stream. Forward it downstream.
                PipelinePacket::Flush => {
                    self.seek.on_flush();
                    self.reset_parser();
                    out.push(PipelinePacket::Flush).await?;
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

#[cfg(fuzzing)]
pub fn fuzz_parse(data: &[u8]) {
    use g2g_core::PushOutcome;

    struct NoopSink;
    impl OutputSink for NoopSink {
        fn push<'a>(
            &'a mut self,
            _packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }
    }

    let mut demux = Fmp4Demux::new();
    if demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .is_err()
    {
        return;
    }
    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(data.to_vec().into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };
    let mut sink = NoopSink;
    let _ = crate::fuzz_block_on(demux.process(PipelinePacket::DataFrame(frame), &mut sink));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_box_len_distinguishes_incomplete_from_malformed() {
        // Fewer than 8 bytes: header not yet buffered, keep waiting.
        assert_eq!(next_box_len(&[0, 0, 0, 16, b'm']), Ok(None));
        // size==1 large-size form without the full 64-bit field yet: keep waiting.
        assert_eq!(
            next_box_len(&[0, 0, 0, 1, b'm', b'd', b'a', b't', 0, 0, 0, 0]),
            Ok(None)
        );
        // A framed box reports its total length.
        assert_eq!(
            next_box_len(&[0, 0, 0, 16, b'f', b't', b'y', b'p']),
            Ok(Some(16))
        );
        // size < 8 is malformed: fail loud instead of stalling the demuxer.
        assert_eq!(
            next_box_len(&[0, 0, 0, 0, b'm', b'o', b'o', b'v']),
            Err(G2gError::CapsMismatch)
        );
        assert_eq!(
            next_box_len(&[0, 0, 0, 7, b'f', b'r', b'e', b'e']),
            Err(G2gError::CapsMismatch)
        );
    }
}
